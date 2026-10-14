import http.server
import os
import subprocess
import sys
import threading
import time


class DelayingHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", "1000000")
        self.end_headers()
        for _ in range(10):
            self.wfile.write(b"a" * 4096)
            self.wfile.flush()
        # The test sets DEBUGINFOD_TIMEOUT=1
        time.sleep(5)


httpd = http.server.HTTPServer(("localhost", 0), DelayingHandler)
port = httpd.socket.getsockname()[1]

try:
    t = threading.Thread(target=httpd.serve_forever).start()
    os.environ["DEBUGINFOD_URLS"] = f"http://localhost:{port}"
    result = subprocess.run(sys.argv[1:], capture_output=True)
    print(result.stderr.decode())
finally:
    httpd.shutdown()
