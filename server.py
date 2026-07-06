from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles
except ImportError as exc:  # pragma: no cover - import guard for clearer local setup errors.
    raise SystemExit(
        "MesoMaxxed backend requires FastAPI and Uvicorn. "
        "Install them with: python -m pip install fastapi uvicorn"
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT / "frontend"

app = FastAPI(
    title="MesoMaxxed Simulation Server",
    version="0.1.0",
    description="Phase 1 backend shell for streaming atmospheric model state to the browser.",
)

MODEL_DOMAIN = {
    "width_miles": 25,
    "depth_miles": 25,
    "height_miles": 10,
    "views": ["2d_mesh", "3d_scene"],
}

MODEL_PHASES = [
    {
        "id": "setup",
        "name": "Environment setup",
        "editable_fields": ["sounding_profile", "cape", "cin", "bulk_shear", "srh", "moisture", "lcl"],
    },
    {
        "id": "simulation",
        "name": "Linked 2D/3D storm simulation",
        "domain": MODEL_DOMAIN,
        "simulated_hazards": ["tornado_circulation", "microburst", "severe_wind_gust"],
    },
]

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ready",
        "service": "MesoMaxxed backend",
        "phase": 1,
        "simulation": "idle",
        "domain": MODEL_DOMAIN,
    }


@app.get("/model/phases")
async def model_phases() -> dict[str, Any]:
    return {
        "phases": MODEL_PHASES,
        "domain": MODEL_DOMAIN,
        "time": {
            "duration_minutes": 90,
            "mode": "fastest_stable_loop",
        },
    }


@app.get("/{asset_name}")
async def frontend_asset(asset_name: str) -> FileResponse:
    asset_path = FRONTEND_DIR / asset_name
    if asset_path.is_file() and asset_path.parent == FRONTEND_DIR:
        return FileResponse(asset_path)
    raise HTTPException(status_code=404, detail="Frontend asset not found")


@app.websocket("/ws/simulation")
async def simulation_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    await websocket.send_json(
        {
            "type": "hello",
            "message": "MesoMaxxed simulation socket ready",
            "binaryStreams": False,
        }
    )

    try:
        while True:
            payload = await websocket.receive_text()
            command = json.loads(payload)
            await websocket.send_json(
                {
                    "type": "ack",
                    "command": command.get("type", "unknown"),
                    "simulation": "idle",
                }
            )
    except WebSocketDisconnect:
        return


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
