# ===----------------------------------------------------------------------===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===//

#!/usr/bin/env python3

import os
import sys
import json
import argparse
import mimetypes
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from pathlib import Path

# Add parent directories to path for imports
current_dir = Path(__file__).parent
tools_dir = current_dir.parent
sys.path.insert(0, str(tools_dir))

from common.collector import ArtifactCollector
from common.models import FileType

# Import API endpoints
from api.health import HealthEndpoint
from api.units import UnitsEndpoint, UnitDetailEndpoint
from api.summary import SummaryEndpoint
from api.files import FileContentEndpoint
from api.artifacts import ArtifactsEndpoint, ArtifactTypesEndpoint
from api.explorer import ExplorerEndpoint
from api.specialized_router import SpecializedRouter, SPECIALIZED_ENDPOINTS_DOCS


class APIHandler(BaseHTTPRequestHandler):
    """API handler"""

    def __init__(self, data_dir, *args, **kwargs):
        self.data_dir = data_dir
        self.collector = ArtifactCollector()

        # Set up frontend paths
        current_dir = Path(__file__).parent
        self.frontend_dir = current_dir / "frontend"
        self.static_dir = self.frontend_dir / "static"
        self.templates_dir = self.frontend_dir / "templates"

        # Initialize endpoints
        self.endpoints = {
            "health": HealthEndpoint(data_dir, self.collector),
            "units": UnitsEndpoint(data_dir, self.collector),
            "unit_detail": UnitDetailEndpoint(data_dir, self.collector),
            "summary": SummaryEndpoint(data_dir, self.collector),
            "files": FileContentEndpoint(data_dir, self.collector),
            "artifacts": ArtifactsEndpoint(data_dir, self.collector),
            "artifact_types": ArtifactTypesEndpoint(data_dir, self.collector),
            "explorer": ExplorerEndpoint(data_dir, self.collector),
        }

        # Initialize specialized router
        self.specialized_router = SpecializedRouter(data_dir, self.collector)

        super().__init__(*args, **kwargs)

    def _send_json_response(self, response_data):
        """Send JSON response with proper headers"""
        if "status" in response_data:
            status = response_data["status"]
        else:
            status = 200

        response_json = json.dumps(response_data, indent=2, default=str)

        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(response_json.encode("utf-8"))

    def _send_error(self, message, status=500):
        """Send error response"""
        self._send_json_response({"success": False, "error": message, "status": status})

    def _send_static_file(self, file_path):
        """Send a static file with appropriate headers"""
        try:
            if not file_path.exists():
                self.send_error(404, "File not found")
                return

            # Determine content type
            content_type, _ = mimetypes.guess_type(str(file_path))
            if content_type is None:
                content_type = "application/octet-stream"

            # Read file content
            with open(file_path, "rb") as f:
                content = f.read()

            # Send response
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(content)

        except Exception as e:
            self.send_error(500, f"Error serving file: {str(e)}")

    def _send_html_file(self, file_path):
        """Send an HTML file"""
        try:
            if not file_path.exists():
                self.send_error(404, "File not found")
                return

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content.encode("utf-8"))))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(content.encode("utf-8"))

        except Exception as e:
            self.send_error(500, f"Error serving HTML file: {str(e)}")

    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        """Route GET requests to endpoints or serve static files"""
        try:
            parsed_url = urlparse(self.path)
            path = parsed_url.path.rstrip("/")
            path_parts = path.strip("/").split("/")
            query_params = parse_qs(parsed_url.query)

            # Serve main UI at root
            if path == "" or path == "/":
                index_file = self.templates_dir / "index.html"
                self._send_html_file(index_file)
                return

            # Serve static files
            if path.startswith("/static/"):
                # Remove '/static' from path and get relative path
                static_path = path[8:]  # Remove '/static/'
                file_path = self.static_dir / static_path
                self._send_static_file(file_path)
                return

            # API Documentation endpoint
            if path == "/api" or path == "/api/":
                response = self._get_api_documentation()
                self._send_json_response(response)
                return

            # Route to appropriate API endpoints
            if path == "/api/health":
                response = self.endpoints["health"].handle(path_parts, query_params)
            elif path == "/api/units":
                response = self.endpoints["units"].handle(path_parts, query_params)
            elif path == "/api/summary":
                response = self.endpoints["summary"].handle(path_parts, query_params)
            elif path == "/api/artifacts":
                response = self.endpoints["artifact_types"].handle(
                    path_parts, query_params
                )
            elif path.startswith("/api/units/") and len(path_parts) >= 3:
                response = self.endpoints["unit_detail"].handle(
                    path_parts, query_params
                )
            elif path.startswith("/api/file/") and len(path_parts) >= 5:
                response = self.endpoints["files"].handle(path_parts, query_params)
            elif path.startswith("/api/artifacts/") and len(path_parts) >= 3:
                response = self.endpoints["artifacts"].handle(path_parts, query_params)
            elif path.startswith("/api/explorer/") and len(path_parts) >= 3:
                response = self.endpoints["explorer"].handle(path_parts, query_params)
            elif self._is_specialized_endpoint(path_parts):
                # Route to specialized file-type endpoints
                response = self.specialized_router.route_request(
                    path_parts, query_params
                )
            else:
                response = {
                    "success": False,
                    "error": "Endpoint not found",
                    "status": 404,
                    "available_endpoints": self._get_available_endpoints(),
                }

            self._send_json_response(response)

        except Exception as e:
            self._send_error(f"Internal server error: {str(e)}")

    def _is_specialized_endpoint(self, path_parts: list) -> bool:
        """Check if this is a specialized file-type endpoint"""
        if len(path_parts) >= 2 and path_parts[0] == "api":
            file_type = path_parts[1]
            specialized_types = [
                "remarks",
                "diagnostics",
                "compilation-phases",
                "time-trace",
                "runtime-trace",
                "binary-size",
                "ast-json",
                "sarif",
                "symbols",
                "ir",
                "assembly",
                "preprocessed",
                "macro-expansion",
            ]
            return file_type in specialized_types
        return False

    def _get_api_documentation(self):
        """Generate API documentation"""
        return {
            "success": True,
            "data": {
                "llvm_advisor_api": "1.0",
                "description": "API for LLVM Advisor compilation data",
                "data_directory": self.data_dir,
                "endpoints": self._get_available_endpoints(),
                "specialized_endpoints": self.specialized_router.get_available_endpoints(),
                "supported_file_types": [ft.value for ft in FileType],
            },
            "status": 200,
        }

    def _get_available_endpoints(self):
        """Get list of available endpoints with descriptions"""
        return {
            "GET /api/health": {
                "description": "System health check and data directory status",
                "returns": "Health status, data directory info, basic statistics",
            },
            "GET /api/units": {
                "description": "List all compilation units with basic metadata",
                "returns": "Array of compilation units with file counts by type",
            },
            "GET /api/units/{unit_name}": {
                "description": "Detailed information for a specific compilation unit",
                "returns": "Complete metadata for all files in the unit",
            },
            "GET /api/summary": {
                "description": "Overall statistics summary across all units",
                "returns": "Aggregated statistics, error counts, success rates",
            },
            "GET /api/artifacts": {
                "description": "List all available artifact types with global counts",
                "returns": "Summary of all file types found across units",
            },
            "GET /api/artifacts/{file_type}": {
                "description": "Aggregated data for a specific file type across all units",
                "returns": "All files of the specified type with summary statistics",
            },
            "GET /api/file/{unit_name}/{file_type}/{file_name}": {
                "description": "Get parsed content of a specific file",
                "returns": "Complete parsed data for the requested file",
                "query_params": {
                    "full": "Set to 'true' to get complete data for large files (default: summary only)"
                },
            },
            "Specialized Endpoints": {
                "description": "File-type specific analysis endpoints",
                "pattern": "GET /api/{file_type}/{analysis_type}",
                "examples": {
                    "GET /api/remarks/overview": "Optimization remarks statistics",
                    "GET /api/diagnostics/patterns": "Common diagnostic patterns",
                    "GET /api/compilation-phases/bottlenecks": "Compilation bottlenecks",
                    "GET /api/time-trace/hotspots": "Performance hotspots",
                    "GET /api/binary-size/optimization": "Size optimization opportunities",
                },
            },
        }


def create_handler(data_dir):
    """Factory function to create handler with data directory"""

    def handler(*args, **kwargs):
        return APIHandler(data_dir, *args, **kwargs)

    return handler


def main():
    parser = argparse.ArgumentParser(description="LLVM Advisor API Server")
    parser.add_argument(
        "--data-dir", required=True, help="Directory containing .llvm-advisor data"
    )
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--host", default="localhost", help="Host to bind to")

    args = parser.parse_args()

    # Verify data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory does not exist: {args.data_dir}")
        sys.exit(1)

    # Create handler with data directory
    handler_class = create_handler(args.data_dir)

    # Start server
    server = HTTPServer((args.host, args.port), handler_class)

    print(f"LLVM Advisor Web Interface")
    print(f"==========================")
    print(f"Starting web server on http://{args.host}:{args.port}")
    print(f"Loading data from: {args.data_dir}")
    print(f"Press Ctrl+C to stop the server")
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
