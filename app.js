const canvas = document.getElementById("model-canvas");
const zoomReadout = document.getElementById("zoom-readout");
const latlonReadout = document.getElementById("latlon-readout");
const renderReadout = document.getElementById("render-readout");
const forecastHour = document.getElementById("forecast-hour");
const forecastValue = document.getElementById("forecast-value");
const fpsReadout = document.getElementById("fps-readout");
const frameReadout = document.getElementById("frame-readout");
const playToggle = document.getElementById("play-toggle");
const loopToggle = document.getElementById("loop-toggle");
const phaseReadout = document.getElementById("phase-readout");
const lifecycleReadout = document.getElementById("lifecycle-readout");
const rateReadout = document.getElementById("rate-readout");
const solverReadout = document.getElementById("solver-readout");

const DOMAIN = Object.freeze({ width: 25, depth: 25, top: 10, duration: 90 });
const VARIABLE_INDEX = {
  reflectivity: 0,
  temperature: 1,
  dewpoint: 2,
  wind: 3,
  cape: 4,
  vorticity: 5,
  vertical: 6,
  pressure: 7
};

const defaultProfile = {
  heights: [0, 1.2, 2.5, 4.5, 6.5, 8.2, 10],
  temperature: [27, 22, 14, 4, -12, -28, -45],
  dewpoint: [21, 17, 8, -4, -20, -36, -50]
};

const setup = {
  cape: 2800,
  cin: 50,
  shear: 45,
  srh: 210,
  moisture: 68,
  lcl: 1150,
  heights: [...defaultProfile.heights],
  temperature: [...defaultProfile.temperature],
  dewpoint: [...defaultProfile.dewpoint]
};

const model = {
  phase: "setup",
  selectedVariable: "reflectivity",
  viewMode: 0,
  simTime: 0,
  playing: false,
  loop: true,
  timeScale: 4,
  adaptiveMinutesPerSecond: 0,
  stormMode: "Classic Supercell",
  capStatus: "Breakable",
  eventLog: ["Setup environment loaded."],
  probe: { x: 0, y: 0, z: 0, source: "2D" }
};

const viewport = {
  zoom: 1,
  targetZoom: 1,
  offsetX: 0,
  offsetY: 0,
  targetOffsetX: 0,
  targetOffsetY: 0,
  cameraYaw: 34,
  cameraPitch: 52,
  cameraRange: 1.35,
  pointerMode: null,
  pointerMoved: false,
  pointerStartX: 0,
  pointerStartY: 0,
  lastX: 0,
  lastY: 0
};

const gl = canvas.getContext("webgl2", {
  antialias: true,
  alpha: false,
  powerPreference: "high-performance"
});

let glState = null;
let lastFrameTime = performance.now();
let fpsAccumulator = 0;
let fpsFrames = 0;
let lastFps = 60;
let skewDrag = null;

function resizeCanvas() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.floor(canvas.clientWidth * dpr);
  const height = Math.floor(canvas.clientHeight * dpr);

  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
    if (gl) {
      gl.viewport(0, 0, width, height);
    }
  }
}

function compileShader(type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);

  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(shader) || "Shader compilation failed");
  }

  return shader;
}

function initWebGL() {
  if (!gl) {
    renderReadout.textContent = "2D fallback";
    return null;
  }

  const vertexSource = `#version 300 es
    precision highp float;
    in vec2 position;
    out vec2 vUv;

    void main() {
      vUv = position * 0.5 + 0.5;
      gl_Position = vec4(position, 0.0, 1.0);
    }
  `;

  const fragmentSource = `#version 300 es
    precision highp float;

    in vec2 vUv;
    out vec4 fragColor;

    uniform vec2 resolution;
    uniform vec2 offset;
    uniform float zoom;
    uniform float simTime;
    uniform float phase;
    uniform float cape;
    uniform float cin;
    uniform float shear;
    uniform float srh;
    uniform float moisture;
    uniform float cameraYaw;
    uniform float cameraPitch;
    uniform float cameraRange;
    uniform float variableIndex;
    uniform float viewMode;

    const float PI = 3.14159265359;

    float hash(vec2 p) {
      p = fract(p * vec2(123.34, 456.21));
      p += dot(p, p + 45.32);
      return fract(p.x * p.y);
    }

    float noise(vec2 p) {
      vec2 i = floor(p);
      vec2 f = fract(p);
      float a = hash(i);
      float b = hash(i + vec2(1.0, 0.0));
      float c = hash(i + vec2(0.0, 1.0));
      float d = hash(i + vec2(1.0, 1.0));
      vec2 u = f * f * (3.0 - 2.0 * f);
      return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
    }

    float fbm(vec2 p) {
      float value = 0.0;
      float amp = 0.5;
      for (int i = 0; i < 5; i++) {
        value += amp * noise(p);
        p *= 2.03;
        amp *= 0.5;
      }
      return value;
    }

    float gaussian(vec2 p, vec2 c, vec2 radius) {
      vec2 d = (p - c) / radius;
      return exp(-dot(d, d));
    }

    float lifecycle(float t) {
      float grow = smoothstep(0.05, 0.34, t);
      float decay = 1.0 - smoothstep(0.78, 1.0, t);
      return grow * decay;
    }

    vec2 stormCenter(float t, float shearN) {
      return vec2(-0.24 + t * (0.45 + shearN * 0.08), 0.18 - t * 0.28 + sin(t * PI * 2.0) * 0.05);
    }

    float stormField2d(vec2 p, float t) {
      float instability = clamp(cape / 5000.0, 0.0, 1.0);
      float shearN = clamp(shear / 90.0, 0.0, 1.0);
      float capN = clamp(cin / 250.0, 0.0, 1.0);
      vec2 center = stormCenter(t, shearN);
      float mature = lifecycle(t);
      float core = gaussian(p, center, vec2(0.12 + instability * 0.05, 0.10 + shearN * 0.07));
      float anvil = gaussian(p, center + vec2(0.13 + shearN * 0.18, 0.05), vec2(0.28, 0.16));
      float line = gaussian(p, vec2(center.x - 0.18, center.y - 0.18), vec2(0.11, 0.35));
      float texNoise = fbm(p * 9.0 + vec2(t * 4.0, -t * 2.5));
      float capDelay = smoothstep(capN * 0.42, capN * 0.42 + 0.2, t);
      return clamp((core * 1.25 + anvil * 0.52 + line * shearN * 0.44 + texNoise * 0.16) * mature * capDelay, 0.0, 1.4);
    }

    float vortexField(vec2 p, float t) {
      float shearN = clamp(shear / 90.0, 0.0, 1.0);
      float srhN = clamp(srh / 500.0, 0.0, 1.0);
      vec2 center = stormCenter(t, shearN) + vec2(-0.045, -0.055);
      float mature = smoothstep(0.38, 0.55, t) * (1.0 - smoothstep(0.74, 0.88, t));
      return gaussian(p, center, vec2(0.026, 0.036)) * mature * shearN * srhN;
    }

    float microburstField(vec2 p, float t) {
      float instability = clamp(cape / 5000.0, 0.0, 1.0);
      float dryAir = 1.0 - clamp(moisture / 100.0, 0.0, 1.0);
      float shearN = clamp(shear / 90.0, 0.0, 1.0);
      vec2 center = stormCenter(t, shearN) + vec2(0.07, -0.12);
      float stage = smoothstep(0.58, 0.68, t) * (1.0 - smoothstep(0.84, 0.96, t));
      return gaussian(p, center, vec2(0.10, 0.10)) * stage * (0.4 + instability * 0.55 + dryAir * 0.4);
    }

    vec3 variableColor(float v, float vort, float downburst, vec2 p, float t) {
      float idx = floor(variableIndex + 0.5);
      float instability = clamp(cape / 5000.0, 0.0, 1.0);
      float shearN = clamp(shear / 90.0, 0.0, 1.0);
      vec3 cold = vec3(0.015, 0.045, 0.075);
      vec3 rain = vec3(0.02, 0.42, 0.48);
      vec3 hail = vec3(0.70, 0.86, 0.36);
      vec3 core = vec3(0.95, 0.25, 0.18);
      vec3 refl = mix(cold, rain, smoothstep(0.07, 0.40, v));
      refl = mix(refl, hail, smoothstep(0.45, 0.72, v));
      refl = mix(refl, core, smoothstep(0.75, 1.05, v));

      if (idx == 1.0) {
        float temp = clamp(0.55 + p.y * 0.22 - v * 0.32 + instability * 0.14, 0.0, 1.0);
        return mix(vec3(0.08, 0.34, 0.62), vec3(0.96, 0.40, 0.20), temp);
      }
      if (idx == 2.0) {
        float dew = clamp(moisture / 100.0 + v * 0.25 - p.x * 0.10, 0.0, 1.0);
        return mix(vec3(0.08, 0.18, 0.28), vec3(0.20, 0.78, 0.70), dew);
      }
      if (idx == 3.0) {
        float wind = clamp(shearN * 0.52 + v * 0.45 + downburst * 0.72, 0.0, 1.0);
        return mix(vec3(0.08, 0.16, 0.30), vec3(0.98, 0.72, 0.22), wind);
      }
      if (idx == 4.0) {
        float localCape = clamp(instability * (1.0 - v * 0.25) + fbm(p * 4.0) * 0.18, 0.0, 1.0);
        return mix(vec3(0.06, 0.15, 0.10), vec3(0.78, 0.88, 0.28), localCape);
      }
      if (idx == 5.0) {
        return mix(vec3(0.05, 0.05, 0.13), vec3(0.95, 0.22, 0.78), clamp(vort * 4.0 + v * 0.2, 0.0, 1.0));
      }
      if (idx == 6.0) {
        float w = clamp(v * 0.95 - downburst * 0.45 + 0.25, 0.0, 1.0);
        return mix(vec3(0.08, 0.14, 0.35), vec3(0.76, 0.92, 1.0), w);
      }
      if (idx == 7.0) {
        float pressure = clamp(0.52 - vort * 0.75 + downburst * 0.34, 0.0, 1.0);
        return mix(vec3(0.18, 0.12, 0.36), vec3(0.88, 0.82, 0.58), pressure);
      }
      return refl;
    }

    vec3 renderPlan(vec2 uv) {
      vec2 aspect = vec2(resolution.x / resolution.y, 1.0);
      vec2 world = ((uv - 0.5) * aspect) / zoom + offset;
      float t = clamp(simTime / 90.0, 0.0, 1.0);
      float terrain = fbm(world * 2.0 + vec2(-1.4, 0.7));
      float field = stormField2d(world, t);
      float vort = vortexField(world, t);
      float downburst = microburstField(world, t);
      vec3 color = variableColor(field, vort, downburst, world, t);
      color += vec3(0.07, 0.09, 0.08) * terrain * 0.38;

      vec2 domain = abs(world);
      float outside = step(0.5, max(domain.x, domain.y));
      vec2 grid = abs(fract((world + 0.5) * 10.0) - 0.5);
      float gridLine = 1.0 - smoothstep(0.0, 0.015 / zoom, min(grid.x, grid.y));
      float border = 1.0 - smoothstep(0.0, 0.006 / zoom, abs(max(domain.x, domain.y) - 0.5));
      color += vec3(0.13, 0.20, 0.22) * gridLine * (1.0 - outside) * 0.22;
      color += vec3(0.72, 0.86, 0.88) * border * 0.34;
      color = mix(color, vec3(0.01, 0.02, 0.035), outside * 0.68);

      if (phase < 0.5) {
        float setupHalo = 1.0 - smoothstep(0.20, 0.50, length(world));
        color = mix(color, vec3(0.08, 0.18, 0.23), setupHalo * 0.28);
      }

      if (vort > 0.22) {
        color = mix(color, vec3(1.0, 0.12, 0.08), clamp(vort * 3.0, 0.0, 1.0));
      }
      if (downburst > 0.35) {
        color = mix(color, vec3(0.58, 0.88, 1.0), clamp(downburst * 0.9, 0.0, 1.0));
      }

      return color * (1.0 - length(uv - 0.5) * 0.2);
    }

    vec2 boxIntersect(vec3 ro, vec3 rd, vec3 bmin, vec3 bmax) {
      vec3 invD = 1.0 / rd;
      vec3 t0 = (bmin - ro) * invD;
      vec3 t1 = (bmax - ro) * invD;
      vec3 tsmaller = min(t0, t1);
      vec3 tbigger = max(t0, t1);
      float tnear = max(max(tsmaller.x, tsmaller.y), tsmaller.z);
      float tfar = min(min(tbigger.x, tbigger.y), tbigger.z);
      return vec2(tnear, tfar);
    }

    float boxEdge(vec3 p) {
      vec3 d = min(abs(p - vec3(-0.5, -0.5, 0.0)), abs(p - vec3(0.5, 0.5, 0.4)));
      return 1.0 - smoothstep(0.0, 0.008, min(min(d.x, d.y), d.z));
    }

    vec3 renderScene(vec2 uv) {
      vec2 p = (uv - 0.5) * vec2(1.55, 1.2);
      float yaw = radians(cameraYaw);
      float pitch = radians(cameraPitch);
      vec3 target = vec3(0.0, 0.0, 0.18);
      vec3 ro = target + vec3(cos(yaw) * cos(pitch), sin(yaw) * cos(pitch), sin(pitch)) * cameraRange;
      vec3 forward = normalize(target - ro);
      vec3 right = normalize(cross(forward, vec3(0.0, 0.0, 1.0)));
      vec3 up = normalize(cross(right, forward));
      vec3 rd = normalize(forward + right * p.x + up * p.y);
      vec2 hit = boxIntersect(ro, rd, vec3(-0.5, -0.5, 0.0), vec3(0.5, 0.5, 0.4));
      vec3 sky = mix(vec3(0.01, 0.025, 0.04), vec3(0.03, 0.08, 0.11), uv.y);

      if (hit.x > hit.y || hit.y < 0.0) {
        return sky;
      }

      float t0 = max(hit.x, 0.0);
      float t1 = hit.y;
      float stepSize = (t1 - t0) / 46.0;
      vec3 accum = vec3(0.0);
      float alpha = 0.0;
      float simN = clamp(simTime / 90.0, 0.0, 1.0);
      float shearN = clamp(shear / 90.0, 0.0, 1.0);
      vec2 center = stormCenter(simN, shearN);

      for (int i = 0; i < 46; i++) {
        float ti = t0 + stepSize * (float(i) + 0.5);
        vec3 pos = ro + rd * ti;
        vec2 horiz = pos.xy;
        float zNorm = clamp(pos.z / 0.4, 0.0, 1.0);
        float updraftColumn = gaussian(horiz, center, vec2(0.12, 0.12));
        float anvil = gaussian(horiz, center + vec2(0.22, 0.06), vec2(0.30, 0.18));
        float zCore = (zNorm - 0.54) * 2.0;
        float verticalShape = exp(-(zCore * zCore)) + updraftColumn * smoothstep(0.15, 0.85, zNorm);
        float density = stormField2d(horiz, simN) * verticalShape * (0.58 + anvil * 0.3);
        float vort = vortexField(horiz, simN) * (1.0 - smoothstep(0.0, 0.34, zNorm));
        float micro = microburstField(horiz, simN) * (1.0 - smoothstep(0.05, 0.22, zNorm));
        vec3 sampleColor = variableColor(density, vort, micro, horiz, simN);
        sampleColor = mix(sampleColor, vec3(0.84, 0.90, 0.95), smoothstep(0.62, 1.0, zNorm) * density * 0.35);
        float a = clamp(density * 0.10 + vort * 0.20 + micro * 0.10, 0.0, 0.18);
        accum += (1.0 - alpha) * sampleColor * a;
        alpha += (1.0 - alpha) * a;
      }

      vec3 groundHit = ro + rd * t1;
      float groundGrid = 0.0;
      if (groundHit.z < 0.01) {
        vec2 g = abs(fract((groundHit.xy + 0.5) * 10.0) - 0.5);
        groundGrid = 1.0 - smoothstep(0.0, 0.025, min(g.x, g.y));
      }
      float edge = boxEdge(groundHit);
      vec3 boxColor = vec3(0.18, 0.33, 0.38) * groundGrid + vec3(0.68, 0.84, 0.88) * edge;
      vec3 color = mix(sky + boxColor * 0.4, accum + boxColor, clamp(alpha + edge * 0.35 + groundGrid * 0.12, 0.0, 1.0));
      return color;
    }

    void main() {
      bool split = viewMode < 0.5;
      bool planOnly = viewMode > 0.5 && viewMode < 1.5;
      vec3 color;

      if (split) {
        if (vUv.x < 0.5) {
          color = renderPlan(vec2(vUv.x * 2.0, vUv.y));
        } else {
          color = renderScene(vec2((vUv.x - 0.5) * 2.0, vUv.y));
        }

        float divider = 1.0 - smoothstep(0.0, 0.002, abs(vUv.x - 0.5));
        color += vec3(0.45, 0.65, 0.70) * divider * 0.38;
      } else if (planOnly) {
        color = renderPlan(vUv);
      } else {
        color = renderScene(vUv);
      }

      fragColor = vec4(color, 1.0);
    }
  `;

  const program = gl.createProgram();
  gl.attachShader(program, compileShader(gl.VERTEX_SHADER, vertexSource));
  gl.attachShader(program, compileShader(gl.FRAGMENT_SHADER, fragmentSource));
  gl.linkProgram(program);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(program) || "Program link failed");
  }

  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);

  const position = gl.getAttribLocation(program, "position");
  gl.enableVertexAttribArray(position);
  gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);

  renderReadout.textContent = "WebGL2 split mesh active";

  return {
    program,
    resolution: gl.getUniformLocation(program, "resolution"),
    offset: gl.getUniformLocation(program, "offset"),
    zoom: gl.getUniformLocation(program, "zoom"),
    simTime: gl.getUniformLocation(program, "simTime"),
    phase: gl.getUniformLocation(program, "phase"),
    cape: gl.getUniformLocation(program, "cape"),
    cin: gl.getUniformLocation(program, "cin"),
    shear: gl.getUniformLocation(program, "shear"),
    srh: gl.getUniformLocation(program, "srh"),
    moisture: gl.getUniformLocation(program, "moisture"),
    cameraYaw: gl.getUniformLocation(program, "cameraYaw"),
    cameraPitch: gl.getUniformLocation(program, "cameraPitch"),
    cameraRange: gl.getUniformLocation(program, "cameraRange"),
    variableIndex: gl.getUniformLocation(program, "variableIndex"),
    viewMode: gl.getUniformLocation(program, "viewMode")
  };
}

function renderFallback2d() {
  const context = canvas.getContext("2d");
  if (!context) {
    if (gl) {
      gl.clearColor(0.02, 0.05, 0.08, 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
    }
    return;
  }

  const gradient = context.createLinearGradient(0, 0, canvas.width, canvas.height);
  gradient.addColorStop(0, "#071323");
  gradient.addColorStop(0.48, "#0b2630");
  gradient.addColorStop(1, "#03060b");
  context.fillStyle = gradient;
  context.fillRect(0, 0, canvas.width, canvas.height);
}

function render(now) {
  resizeCanvas();

  const elapsed = Math.min((now - lastFrameTime) / 1000, 0.05);
  lastFrameTime = now;

  if (model.phase === "simulation" && model.playing) {
    const speed = getAdaptiveMinutesPerSecond();
    model.adaptiveMinutesPerSecond = speed;
    model.simTime += elapsed * speed;

    if (model.simTime > DOMAIN.duration) {
      model.simTime = model.loop ? 0 : DOMAIN.duration;
      model.playing = model.loop;
    }

    forecastHour.value = String(Math.round(model.simTime));
  } else {
    model.adaptiveMinutesPerSecond = 0;
  }

  viewport.zoom += (viewport.targetZoom - viewport.zoom) * 0.16;
  viewport.offsetX += (viewport.targetOffsetX - viewport.offsetX) * 0.16;
  viewport.offsetY += (viewport.targetOffsetY - viewport.offsetY) * 0.16;

  if (gl && glState) {
    gl.useProgram(glState.program);
    gl.uniform2f(glState.resolution, canvas.width, canvas.height);
    gl.uniform2f(glState.offset, viewport.offsetX, viewport.offsetY);
    gl.uniform1f(glState.zoom, viewport.zoom);
    gl.uniform1f(glState.simTime, model.simTime);
    gl.uniform1f(glState.phase, model.phase === "simulation" ? 1 : 0);
    gl.uniform1f(glState.cape, setup.cape);
    gl.uniform1f(glState.cin, setup.cin);
    gl.uniform1f(glState.shear, setup.shear);
    gl.uniform1f(glState.srh, setup.srh);
    gl.uniform1f(glState.moisture, setup.moisture);
    gl.uniform1f(glState.cameraYaw, viewport.cameraYaw);
    gl.uniform1f(glState.cameraPitch, viewport.cameraPitch);
    gl.uniform1f(glState.cameraRange, viewport.cameraRange);
    gl.uniform1f(glState.variableIndex, VARIABLE_INDEX[model.selectedVariable] || 0);
    gl.uniform1f(glState.viewMode, model.viewMode);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  } else {
    renderFallback2d();
  }

  updateReadouts(elapsed);
  requestAnimationFrame(render);
}

function getAdaptiveMinutesPerSecond() {
  const base = [2.5, 4, 6.5, 9, 12][model.timeScale - 1] || 9;
  const fpsBoost = lastFps > 55 ? 1.35 : lastFps > 42 ? 1.0 : 0.72;
  return base * fpsBoost;
}

function updateReadouts(elapsed) {
  zoomReadout.textContent = model.viewMode === 2
    ? `3D Range ${viewport.cameraRange.toFixed(2)}x`
    : `2D Zoom ${viewport.zoom.toFixed(2)}x`;
  forecastValue.textContent = String(Math.round(model.simTime)).padStart(3, "0");
  phaseReadout.textContent = model.phase === "setup" ? "Setup" : "Simulation";
  solverReadout.textContent = model.phase === "setup" ? "Setup" : "Local";
  lifecycleReadout.textContent = getLifecycleStage(model.simTime / DOMAIN.duration);
  rateReadout.textContent = model.adaptiveMinutesPerSecond > 0
    ? `${model.adaptiveMinutesPerSecond.toFixed(1)} min/s`
    : "Idle";

  fpsAccumulator += elapsed;
  fpsFrames += 1;

  if (fpsAccumulator >= 0.5) {
    lastFps = Math.round(fpsFrames / fpsAccumulator);
    fpsReadout.textContent = String(lastFps);
    frameReadout.textContent = `${(1000 / Math.max(lastFps, 1)).toFixed(1)} ms`;
    fpsAccumulator = 0;
    fpsFrames = 0;
  }
}

function getLifecycleStage(t) {
  if (model.phase === "setup") return "Pre-initiation";
  if (t < 0.16) return "Initiation";
  if (t < 0.36) return "Deepening";
  if (t < 0.62) return model.stormMode;
  if (t < 0.80) return "Outflow Dominant";
  return "Dissipating";
}

function updateSetupReadouts() {
  const fields = [
    ["cape", "cape-value"],
    ["cin", "cin-value"],
    ["shear", "shear-value"],
    ["srh", "srh-value"],
    ["moisture", "moisture-value"],
    ["lcl", "lcl-value"]
  ];

  fields.forEach(([key, id]) => {
    document.getElementById(id).textContent = String(Math.round(setup[key]));
  });

  model.stormMode = classifyStormMode();
  model.capStatus = setup.cin > 160 ? "Strong" : setup.cin > 90 ? "Conditional" : "Breakable";
  document.getElementById("storm-mode-readout").textContent = model.stormMode;
  document.getElementById("cap-status-readout").textContent = model.capStatus;
  document.getElementById("profile-readout").textContent = `Surface ${Math.round(setup.temperature[0])} C / Td ${Math.round(setup.dewpoint[0])} C`;

  const torRisk = getTornadoRisk();
  const mbRisk = getMicroburstRisk();
  document.getElementById("tornado-meter").value = torRisk;
  document.getElementById("microburst-meter").value = mbRisk;
  document.getElementById("tornado-value").textContent = `${Math.round(torRisk * 100)}%`;
  document.getElementById("microburst-value").textContent = `${Math.round(mbRisk * 100)}%`;
  writeEventLog();
}

function classifyStormMode() {
  if (setup.cin > 185 && setup.cape < 1800) return "Capped Elevated";
  if (setup.shear >= 38 && setup.cape >= 1600 && setup.srh >= 140) return "Classic Supercell";
  if (setup.shear >= 52 && setup.cape >= 2200) return "HP Supercell";
  if (setup.shear >= 24) return "Multicell Cluster";
  if (setup.cape >= 2400) return "Pulse Severe";
  return "Weak Convection";
}

function getTornadoRisk() {
  const shearTerm = smoothstep(24, 58, setup.shear);
  const srhTerm = smoothstep(90, 330, setup.srh);
  const capeTerm = smoothstep(900, 3300, setup.cape);
  const lclTerm = 1 - smoothstep(900, 2100, setup.lcl);
  const capPenalty = 1 - smoothstep(130, 240, setup.cin) * 0.55;
  return clamp((shearTerm * 0.32 + srhTerm * 0.34 + capeTerm * 0.20 + lclTerm * 0.14) * capPenalty, 0, 1);
}

function getMicroburstRisk() {
  const capeTerm = smoothstep(1200, 4300, setup.cape);
  const dryTerm = 1 - setup.moisture / 100;
  const shearPenalty = 1 - smoothstep(48, 85, setup.shear) * 0.35;
  return clamp(capeTerm * 0.48 + dryTerm * 0.38 + smoothstep(850, 2200, setup.lcl) * 0.22, 0, 1) * shearPenalty;
}

function writeEventLog() {
  const log = document.getElementById("event-log");
  log.innerHTML = "";
  model.eventLog.slice(-4).forEach((item) => {
    const row = document.createElement("span");
    row.textContent = item;
    log.append(row);
  });
}

function setPhase(phase) {
  model.phase = phase;
  document.querySelectorAll("[data-phase-button]").forEach((button) => {
    button.classList.toggle("active", button.dataset.phaseButton === phase);
  });

  if (phase === "setup") {
    model.playing = false;
    playToggle.textContent = "Play";
    model.simTime = 0;
    forecastHour.value = "0";
    model.eventLog.push("Returned to setup phase.");
  } else {
    model.playing = true;
    playToggle.textContent = "Pause";
    model.eventLog.push(`${model.stormMode} simulation started.`);
  }

  updateSetupReadouts();
}

function resetEnvironment() {
  Object.assign(setup, {
    cape: 2800,
    cin: 50,
    shear: 45,
    srh: 210,
    moisture: 68,
    lcl: 1150,
    heights: [...defaultProfile.heights],
    temperature: [...defaultProfile.temperature],
    dewpoint: [...defaultProfile.dewpoint]
  });

  syncInputsFromSetup();
  drawSkewT();
  model.eventLog.push("Setup profile reset.");
  updateSetupReadouts();
}

function syncInputsFromSetup() {
  document.getElementById("cape-input").value = setup.cape;
  document.getElementById("cin-input").value = setup.cin;
  document.getElementById("shear-input").value = setup.shear;
  document.getElementById("srh-input").value = setup.srh;
  document.getElementById("moisture-input").value = setup.moisture;
  document.getElementById("lcl-input").value = setup.lcl;
}

function initSetupControls() {
  const inputs = [
    ["cape-input", "cape"],
    ["cin-input", "cin"],
    ["shear-input", "shear"],
    ["srh-input", "srh"],
    ["moisture-input", "moisture"],
    ["lcl-input", "lcl"]
  ];

  inputs.forEach(([id, key]) => {
    document.getElementById(id).addEventListener("input", (event) => {
      setup[key] = Number(event.target.value);
      updateProfileFromBulkSetup();
      drawSkewT();
      updateSetupReadouts();
    });
  });

  document.getElementById("start-simulation").addEventListener("click", () => setPhase("simulation"));
  document.getElementById("reset-environment").addEventListener("click", resetEnvironment);
  document.querySelectorAll("[data-phase-button]").forEach((button) => {
    button.addEventListener("click", () => setPhase(button.dataset.phaseButton));
  });
}

function updateProfileFromBulkSetup() {
  const moistureBoost = (setup.moisture - 68) * 0.05;
  const capeBoost = (setup.cape - 2800) * 0.0018;
  const capBoost = setup.cin * 0.012;
  setup.temperature = defaultProfile.temperature.map((value, index) => {
    const lowLevel = 1 - index / (defaultProfile.temperature.length - 1);
    return value + capeBoost * lowLevel * 4 + (index === 1 ? capBoost : 0);
  });
  setup.dewpoint = defaultProfile.dewpoint.map((value, index) => value + moistureBoost * (1 - index * 0.08));
}

function drawSkewT() {
  const tempLine = document.getElementById("temp-line");
  const dewLine = document.getElementById("dew-line");
  const handles = document.getElementById("skewt-handles");
  const tempPoints = setup.temperature.map((temp, index) => profilePoint(temp, index));
  const dewPoints = setup.dewpoint.map((temp, index) => profilePoint(temp, index));

  tempLine.setAttribute("points", tempPoints.map((point) => `${point.x},${point.y}`).join(" "));
  dewLine.setAttribute("points", dewPoints.map((point) => `${point.x},${point.y}`).join(" "));
  handles.innerHTML = "";

  [
    ["temp", tempPoints],
    ["dew", dewPoints]
  ].forEach(([type, points]) => {
    points.forEach((point, index) => {
      const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      circle.setAttribute("class", `skew-handle ${type}`);
      circle.setAttribute("cx", point.x);
      circle.setAttribute("cy", point.y);
      circle.setAttribute("r", "6");
      circle.dataset.type = type;
      circle.dataset.index = String(index);
      handles.append(circle);
    });
  });
}

function profilePoint(temp, index) {
  const y = 226 - (setup.heights[index] / DOMAIN.top) * 204;
  const skew = (226 - y) * 0.32;
  const x = clamp(126 + temp * 3.0 + skew, 34, 286);
  return { x, y };
}

function tempFromPoint(x, index) {
  const y = 226 - (setup.heights[index] / DOMAIN.top) * 204;
  const skew = (226 - y) * 0.32;
  return (clamp(x, 34, 286) - 126 - skew) / 3.0;
}

function initSkewTDrag() {
  const svg = document.getElementById("skewt-editor");

  svg.addEventListener("pointerdown", (event) => {
    const target = event.target.closest(".skew-handle");
    if (!target) return;
    skewDrag = {
      type: target.dataset.type,
      index: Number(target.dataset.index)
    };
    svg.setPointerCapture(event.pointerId);
  });

  svg.addEventListener("pointermove", (event) => {
    if (!skewDrag) return;
    const rect = svg.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * 300;
    const value = tempFromPoint(x, skewDrag.index);
    const profile = skewDrag.type === "temp" ? setup.temperature : setup.dewpoint;
    profile[skewDrag.index] = value;

    if (skewDrag.type === "dew") {
      profile[skewDrag.index] = Math.min(profile[skewDrag.index], setup.temperature[skewDrag.index] - 1);
    } else {
      setup.dewpoint[skewDrag.index] = Math.min(setup.dewpoint[skewDrag.index], profile[skewDrag.index] - 1);
    }

    deriveBulkSetupFromProfile();
    syncInputsFromSetup();
    drawSkewT();
    updateSetupReadouts();
  });

  svg.addEventListener("pointerup", (event) => {
    if (skewDrag) {
      model.eventLog.push("Sounding profile edited.");
      writeEventLog();
    }
    skewDrag = null;
    svg.releasePointerCapture(event.pointerId);
  });
}

function deriveBulkSetupFromProfile() {
  const lowLevelLapse = setup.temperature[0] - setup.temperature[2];
  const midDryness = setup.temperature[3] - setup.dewpoint[3];
  const surfaceMoisture = setup.dewpoint[0];
  setup.cape = clamp(Math.round(900 + lowLevelLapse * 145 + surfaceMoisture * 58), 250, 5000);
  setup.cin = clamp(Math.round(35 + Math.max(0, setup.temperature[1] - setup.temperature[0] + 5) * 23 + midDryness * 2.2), 0, 250);
  setup.moisture = clamp(Math.round(34 + surfaceMoisture * 1.55), 20, 95);
}

function initTimeControls() {
  forecastHour.addEventListener("input", () => {
    model.simTime = Number(forecastHour.value);
    updateDataProbe(model.probe);
  });

  document.getElementById("speed-input").addEventListener("input", (event) => {
    model.timeScale = Number(event.target.value);
    document.getElementById("speed-value").textContent = model.timeScale >= 4 ? "Fastest stable" : `${model.timeScale}x`;
  });

  playToggle.addEventListener("click", togglePlayback);
  loopToggle.addEventListener("click", () => {
    model.loop = !model.loop;
    loopToggle.classList.toggle("active", model.loop);
  });
  document.getElementById("rewind-time").addEventListener("click", () => {
    model.simTime = 0;
    forecastHour.value = "0";
    updateDataProbe(model.probe);
  });
  document.getElementById("step-back").addEventListener("click", () => stepTime(-3));
  document.getElementById("step-forward").addEventListener("click", () => stepTime(3));
}

function stepTime(delta) {
  model.simTime = clamp(model.simTime + delta, 0, DOMAIN.duration);
  forecastHour.value = String(Math.round(model.simTime));
  updateDataProbe(model.probe);
}

function togglePlayback() {
  if (model.phase === "setup") {
    setPhase("simulation");
    return;
  }

  model.playing = !model.playing;
  playToggle.textContent = model.playing ? "Pause" : "Play";
}

function initViewControls() {
  document.querySelectorAll("[data-view-mode]").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll("[data-view-mode]").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      model.viewMode = button.dataset.viewMode === "split" ? 0 : button.dataset.viewMode === "plan" ? 1 : 2;
    });
  });

  const viewInputs = [
    ["yaw-input", "cameraYaw", "yaw-value", (value) => `${Math.round(value)}`],
    ["pitch-input", "cameraPitch", "pitch-value", (value) => `${Math.round(value)}`],
    ["range-input", "cameraRange", "range-value", (value) => `${value.toFixed(2)}`]
  ];

  viewInputs.forEach(([id, key, readout, formatter]) => {
    document.getElementById(id).addEventListener("input", (event) => {
      const value = key === "cameraRange" ? Number(event.target.value) / 100 : Number(event.target.value);
      viewport[key] = value;
      document.getElementById(readout).textContent = formatter(value);
    });
  });
}

function initVariableSelection() {
  document.querySelectorAll("input[name='variable']").forEach((input) => {
    input.addEventListener("change", () => {
      model.selectedVariable = input.value;
      document.querySelectorAll(".field-row").forEach((row) => row.classList.remove("active"));
      input.closest(".field-row").classList.add("active");
    });
  });
}

function screenToModel(clientX, clientY) {
  const rect = canvas.getBoundingClientRect();
  let nx = (clientX - rect.left) / rect.width;
  const ny = (clientY - rect.top) / rect.height;

  if (model.viewMode === 0 && nx < 0.5) {
    nx *= 2;
  }

  const aspect = rect.width / rect.height;
  const xNorm = ((nx - 0.5) * aspect) / viewport.zoom + viewport.offsetX;
  const yNorm = -(((ny - 0.5) / viewport.zoom) + viewport.offsetY);
  return {
    x: clamp(xNorm * DOMAIN.width, -DOMAIN.width / 2, DOMAIN.width / 2),
    y: clamp(yNorm * DOMAIN.depth, -DOMAIN.depth / 2, DOMAIN.depth / 2),
    z: 0,
    source: "2D"
  };
}

function screenToSceneProbe(clientX, clientY) {
  const rect = canvas.getBoundingClientRect();
  let nx = (clientX - rect.left) / rect.width;
  const ny = (clientY - rect.top) / rect.height;

  if (model.viewMode === 0) {
    nx = (nx - 0.5) * 2;
  }

  const x = clamp((nx - 0.5) * DOMAIN.width * 1.15, -DOMAIN.width / 2, DOMAIN.width / 2);
  const y = clamp((0.5 - ny) * DOMAIN.depth * 1.15, -DOMAIN.depth / 2, DOMAIN.depth / 2);
  const z = clamp((0.82 - ny) * DOMAIN.top * 0.9, 0, DOMAIN.top);
  return { x, y, z, source: "3D" };
}

function isSceneArea(clientX) {
  if (model.viewMode === 2) return true;
  if (model.viewMode === 1) return false;
  const rect = canvas.getBoundingClientRect();
  return (clientX - rect.left) / rect.width >= 0.5;
}

canvas.addEventListener("pointerdown", (event) => {
  const scene = isSceneArea(event.clientX);
  viewport.pointerMode = scene ? "orbit" : "pan";
  viewport.pointerMoved = false;
  viewport.pointerStartX = event.clientX;
  viewport.pointerStartY = event.clientY;
  viewport.lastX = event.clientX;
  viewport.lastY = event.clientY;
  canvas.setPointerCapture(event.pointerId);
});

canvas.addEventListener("pointermove", (event) => {
  const dx = event.clientX - viewport.lastX;
  const dy = event.clientY - viewport.lastY;
  const total = Math.hypot(event.clientX - viewport.pointerStartX, event.clientY - viewport.pointerStartY);
  viewport.pointerMoved = viewport.pointerMoved || total > 4;

  const probe = isSceneArea(event.clientX) ? screenToSceneProbe(event.clientX, event.clientY) : screenToModel(event.clientX, event.clientY);
  latlonReadout.textContent = `X ${probe.x.toFixed(1)} mi / Y ${probe.y.toFixed(1)} mi`;

  if (viewport.pointerMode === "pan") {
    const rect = canvas.getBoundingClientRect();
    viewport.targetOffsetX -= dx / rect.width * (2.0 / viewport.targetZoom);
    viewport.targetOffsetY += dy / rect.height * (2.0 / viewport.targetZoom);
  } else if (viewport.pointerMode === "orbit") {
    viewport.cameraYaw = wrapDegrees(viewport.cameraYaw + dx * 0.35);
    viewport.cameraPitch = clamp(viewport.cameraPitch - dy * 0.18, 20, 78);
    document.getElementById("yaw-input").value = viewport.cameraYaw;
    document.getElementById("pitch-input").value = viewport.cameraPitch;
    document.getElementById("yaw-value").textContent = String(Math.round(viewport.cameraYaw));
    document.getElementById("pitch-value").textContent = String(Math.round(viewport.cameraPitch));
  }

  viewport.lastX = event.clientX;
  viewport.lastY = event.clientY;
});

canvas.addEventListener("pointerup", (event) => {
  if (!viewport.pointerMoved) {
    const probe = isSceneArea(event.clientX) ? screenToSceneProbe(event.clientX, event.clientY) : screenToModel(event.clientX, event.clientY);
    updateDataProbe(probe);
  }

  viewport.pointerMode = null;
  canvas.releasePointerCapture(event.pointerId);
});

canvas.addEventListener("wheel", (event) => {
  event.preventDefault();
  const scene = isSceneArea(event.clientX);

  if (scene) {
    const zoomFactor = Math.exp(event.deltaY * 0.001);
    viewport.cameraRange = clamp(viewport.cameraRange * zoomFactor, 0.85, 2.2);
    document.getElementById("range-input").value = Math.round(viewport.cameraRange * 100);
    document.getElementById("range-value").textContent = viewport.cameraRange.toFixed(2);
  } else {
    const zoomFactor = Math.exp(-event.deltaY * 0.001);
    viewport.targetZoom = clamp(viewport.targetZoom * zoomFactor, 0.65, 14);
  }
}, { passive: false });

canvas.addEventListener("dblclick", (event) => {
  if (isSceneArea(event.clientX)) {
    viewport.cameraRange = clamp(viewport.cameraRange * 0.82, 0.85, 2.2);
  } else {
    viewport.targetZoom = clamp(viewport.targetZoom * 1.8, 0.65, 14);
  }
});

canvas.addEventListener("contextmenu", (event) => event.preventDefault());

window.addEventListener("keydown", (event) => {
  const step = 0.08 / viewport.targetZoom;

  if (event.key === "+" || event.key === "=") {
    viewport.targetZoom = clamp(viewport.targetZoom * 1.2, 0.65, 14);
  } else if (event.key === "-" || event.key === "_") {
    viewport.targetZoom = clamp(viewport.targetZoom / 1.2, 0.65, 14);
  } else if (event.key === "ArrowLeft") {
    viewport.targetOffsetX -= step;
  } else if (event.key === "ArrowRight") {
    viewport.targetOffsetX += step;
  } else if (event.key === "ArrowUp") {
    viewport.targetOffsetY -= step;
  } else if (event.key === "ArrowDown") {
    viewport.targetOffsetY += step;
  } else if (event.key.toLowerCase() === "h") {
    viewport.targetZoom = 1;
    viewport.targetOffsetX = 0;
    viewport.targetOffsetY = 0;
    viewport.cameraYaw = 34;
    viewport.cameraPitch = 52;
    viewport.cameraRange = 1.35;
  } else if (event.code === "Space") {
    event.preventDefault();
    togglePlayback();
  }
});

function updateDataProbe(probe) {
  model.probe = probe;
  const sample = sampleStorm(probe.x, probe.y, probe.z);
  document.getElementById("sample-x").textContent = `${probe.x.toFixed(1)} mi`;
  document.getElementById("sample-y").textContent = `${probe.y.toFixed(1)} mi`;
  document.getElementById("sample-z").textContent = probe.source === "3D" ? `${probe.z.toFixed(1)} mi` : "SFC";
  document.getElementById("sample-temp").textContent = `${sample.temperature.toFixed(0)} C`;
  document.getElementById("sample-dew").textContent = `${sample.dewpoint.toFixed(0)} C`;
  document.getElementById("sample-ref").textContent = `${sample.reflectivity.toFixed(0)} dBZ`;
  document.getElementById("sample-gust").textContent = `${sample.windGust.toFixed(0)} mph`;
  document.getElementById("sample-updraft").textContent = `${sample.updraft.toFixed(1)} m/s`;
  document.getElementById("sample-vort").textContent = `${sample.vorticity.toFixed(2)} s-1`;
  document.getElementById("sample-hazard").textContent = sample.hazard;
}

function sampleStorm(xMiles, yMiles, zMiles) {
  const x = xMiles / DOMAIN.width;
  const y = yMiles / DOMAIN.depth;
  const t = clamp(model.simTime / DOMAIN.duration, 0, 1);
  const center = stormCenter(t);
  const dx = x - center.x;
  const dy = y - center.y;
  const core = Math.exp(-((dx * dx) / 0.018 + (dy * dy) / 0.015));
  const anvil = Math.exp(-(((x - center.x - 0.16) ** 2) / 0.07 + ((y - center.y - 0.05) ** 2) / 0.035));
  const lifecycle = stormLifecycle(t);
  const zFactor = Math.max(0.12, 1 - zMiles / DOMAIN.top);
  const vortex = getTornadoRisk() * Math.exp(-(((x - center.x + 0.045) ** 2) / 0.0014 + ((y - center.y + 0.055) ** 2) / 0.0016)) * matureWindow(t);
  const micro = getMicroburstRisk() * Math.exp(-(((x - center.x - 0.07) ** 2) / 0.012 + ((y - center.y + 0.12) ** 2) / 0.012)) * outflowWindow(t);
  const intensity = (core * 1.1 + anvil * 0.35) * lifecycle;
  const temp = setup.temperature[0] - zMiles * 6.0 - intensity * 5 + micro * 2;
  const dew = setup.dewpoint[0] - zMiles * 5.4 + intensity * 1.5;
  const reflectivity = clamp(intensity * 68 + vortex * 15, 0, 82);
  const windGust = clamp(18 + setup.shear * 0.55 + intensity * 28 + micro * 52 + vortex * 22, 0, 115);
  const updraft = clamp(intensity * (setup.cape / 5000) * 44 * zFactor - micro * 13, -24, 68);
  const vorticity = clamp(vortex * 0.85 + setup.srh / 1200 * intensity, 0, 1.25);

  let hazard = "None";
  if (vortex > 0.42) hazard = "Tornado circulation";
  else if (micro > 0.48) hazard = "Microburst";
  else if (windGust > 58) hazard = "Severe gust";
  else if (reflectivity > 50) hazard = "Storm core";

  return { temperature: temp, dewpoint: dew, reflectivity, windGust, updraft, vorticity, hazard };
}

function stormCenter(t) {
  const shearN = clamp(setup.shear / 90, 0, 1);
  return {
    x: -0.24 + t * (0.45 + shearN * 0.08),
    y: 0.18 - t * 0.28 + Math.sin(t * Math.PI * 2) * 0.05
  };
}

function stormLifecycle(t) {
  return smoothstep(0.05, 0.34, t) * (1 - smoothstep(0.78, 1, t)) * smoothstep(setup.cin / 250 * 0.42, setup.cin / 250 * 0.42 + 0.2, t);
}

function matureWindow(t) {
  return smoothstep(0.38, 0.55, t) * (1 - smoothstep(0.74, 0.88, t));
}

function outflowWindow(t) {
  return smoothstep(0.58, 0.68, t) * (1 - smoothstep(0.84, 0.96, t));
}

function initWindows() {
  document.querySelectorAll("[data-window]").forEach((windowEl, index) => {
    normalizePosition(windowEl);
    windowEl.style.zIndex = String(20 + index);

    const titlebar = windowEl.querySelector(".window-titlebar");
    const collapseButton = windowEl.querySelector("[data-collapse]");
    const resizeHandle = windowEl.querySelector("[data-resize]");

    titlebar.addEventListener("pointerdown", (event) => startDrag(event, windowEl));
    collapseButton.addEventListener("click", () => {
      windowEl.classList.toggle("is-collapsed");
      const collapsed = windowEl.classList.contains("is-collapsed");
      collapseButton.textContent = collapsed ? "+" : "-";
      collapseButton.title = collapsed ? "Expand window" : "Collapse window";
    });

    if (resizeHandle) {
      resizeHandle.addEventListener("pointerdown", (event) => startResize(event, windowEl));
    }
  });
}

function normalizePosition(windowEl) {
  const rect = windowEl.getBoundingClientRect();
  const style = window.getComputedStyle(windowEl);

  if (style.right !== "auto" && style.left === "auto") {
    windowEl.style.left = `${document.documentElement.clientWidth - rect.width - parseFloat(style.right)}px`;
    windowEl.style.right = "auto";
  }

  if (style.bottom !== "auto" && style.top === "auto") {
    windowEl.style.top = `${document.documentElement.clientHeight - rect.height - parseFloat(style.bottom)}px`;
    windowEl.style.bottom = "auto";
  }

  if (style.right !== "auto" && style.left !== "auto") {
    windowEl.style.width = `${rect.width}px`;
    windowEl.style.right = "auto";
  }
}

function startDrag(event, windowEl) {
  if (event.target.closest("button, input, svg, meter")) {
    return;
  }

  event.preventDefault();
  bringForward(windowEl);

  const rect = windowEl.getBoundingClientRect();
  const drag = {
    startX: event.clientX,
    startY: event.clientY,
    left: rect.left,
    top: rect.top
  };

  windowEl.classList.add("is-dragging");
  windowEl.setPointerCapture(event.pointerId);

  const move = (moveEvent) => {
    const nextLeft = drag.left + moveEvent.clientX - drag.startX;
    const nextTop = drag.top + moveEvent.clientY - drag.startY;
    const snapped = snapWindow(nextLeft, nextTop, rect.width, rect.height);
    windowEl.style.left = `${snapped.left}px`;
    windowEl.style.top = `${snapped.top}px`;
  };

  const up = (upEvent) => {
    windowEl.classList.remove("is-dragging");
    windowEl.releasePointerCapture(upEvent.pointerId);
    windowEl.removeEventListener("pointermove", move);
    windowEl.removeEventListener("pointerup", up);
  };

  windowEl.addEventListener("pointermove", move);
  windowEl.addEventListener("pointerup", up);
}

function startResize(event, windowEl) {
  event.preventDefault();
  event.stopPropagation();
  bringForward(windowEl);

  const rect = windowEl.getBoundingClientRect();
  const resize = {
    startX: event.clientX,
    startY: event.clientY,
    width: rect.width,
    height: rect.height
  };

  windowEl.classList.add("is-resizing");
  windowEl.setPointerCapture(event.pointerId);

  const move = (moveEvent) => {
    const maxWidth = document.documentElement.clientWidth - rect.left - 18;
    const maxHeight = document.documentElement.clientHeight - rect.top - 18;
    const width = clamp(resize.width + moveEvent.clientX - resize.startX, 220, maxWidth);
    const height = clamp(resize.height + moveEvent.clientY - resize.startY, 112, maxHeight);
    windowEl.style.width = `${width}px`;
    windowEl.style.height = `${height}px`;
  };

  const up = (upEvent) => {
    windowEl.classList.remove("is-resizing");
    windowEl.releasePointerCapture(upEvent.pointerId);
    windowEl.removeEventListener("pointermove", move);
    windowEl.removeEventListener("pointerup", up);
  };

  windowEl.addEventListener("pointermove", move);
  windowEl.addEventListener("pointerup", up);
}

function snapWindow(left, top, width, height) {
  const margin = 18;
  const snap = 14;
  const maxLeft = document.documentElement.clientWidth - width - margin;
  const maxTop = document.documentElement.clientHeight - height - margin;

  let nextLeft = clamp(left, margin, maxLeft);
  let nextTop = clamp(top, margin, maxTop);

  if (Math.abs(nextLeft - margin) < snap) nextLeft = margin;
  if (Math.abs(nextTop - margin) < snap) nextTop = margin;
  if (Math.abs(nextLeft - maxLeft) < snap) nextLeft = maxLeft;
  if (Math.abs(nextTop - maxTop) < snap) nextTop = maxTop;

  return { left: nextLeft, top: nextTop };
}

function bringForward(windowEl) {
  const maxZ = Array.from(document.querySelectorAll("[data-window]"))
    .reduce((highest, item) => Math.max(highest, Number(item.style.zIndex) || 20), 20);
  windowEl.style.zIndex = String(maxZ + 1);
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function smoothstep(edge0, edge1, x) {
  const t = clamp((x - edge0) / (edge1 - edge0), 0, 1);
  return t * t * (3 - 2 * t);
}

function wrapDegrees(value) {
  if (value > 180) return value - 360;
  if (value < -180) return value + 360;
  return value;
}

try {
  glState = initWebGL();
} catch (error) {
  console.warn(error);
  renderReadout.textContent = "2D fallback";
}

initWindows();
initSetupControls();
initSkewTDrag();
initTimeControls();
initViewControls();
initVariableSelection();
syncInputsFromSetup();
drawSkewT();
updateSetupReadouts();
updateDataProbe(model.probe);
resizeCanvas();
requestAnimationFrame(render);
window.addEventListener("resize", resizeCanvas);
