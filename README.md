# MesoMaxxed

MesoMaxxed is a browser-based atmospheric modeling workstation prototype inspired by CM1-style idealized simulation workflows. It now uses a phased flow: users first build the environment in a setup phase, then launch a linked 2D/3D storm simulation inside a fixed 25 mile by 25 mile by 10 mile domain.

## Structure

```text
MesoMaxxed/
  frontend/
    index.html
    styles.css
    app.js
  backend/
    server.py
```

## Run

Open `frontend/index.html` directly in a modern desktop browser, or run the Python backend:

```powershell
cd backend
python -m pip install -r requirements.txt
python server.py
```

Then visit `http://127.0.0.1:8000`.

## Phase 1 Scope

- Full-screen atmospheric model canvas with WebGL2 rendering and a guarded fallback.
- Dark, dense workstation UI tuned for desktop monitors.
- Floating slate-toned windows for setup, variables, linked views, fast time evolution, lifecycle hazards, performance, cross sections, vertical profiles, and data inspection.
- Custom JavaScript pointer handlers for dragging, resizing, collapsing, focus stacking, and viewport edge snapping.
- Setup phase with editable CAPE, CIN, shear, SRH, moisture, LCL, and a draggable Skew-T style sounding profile.
- Simulation phase with shared 2D mesh and 3D storm-box views of the same 25 x 25 x 10 mile domain.
- Parameter-driven storm lifecycle, storm mode classification, tornado circulation and microburst diagnostics, and click-to-probe storm data.
- Canvas mouse wheel zoom, 2D panning, 3D orbit/range controls, double-click zoom, and keyboard navigation shortcuts.
- Fast time evolution with slider scrubbing and loop playback.
- FastAPI backend shell with health, phase metadata, and WebSocket simulation endpoints.
