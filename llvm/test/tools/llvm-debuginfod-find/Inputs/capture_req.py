import http.server
import os
import subprocess
import sys
import threading


class TrivialHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(501)

    def log_request(self, *args, **kwargs):
        print(self.requestline)
        print(self.headers)


httpd = http.server.HTTPServer(("", 0), TrivialHandler)
port = httpd.socket.getsockname()[1]

try:
    t = threading.Thread(target=httpd.serve_forever).start()
    os.environ["DEBUGINFOD_URLS"] = f"http://localhost:{port}"
    subprocess.run(sys.argv[1:], capture_output=True)
finally:
    httpd.shutdown()
