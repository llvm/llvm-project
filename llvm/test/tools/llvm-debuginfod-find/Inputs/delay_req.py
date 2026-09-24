import http.server
import os
import subprocess
import sys
import threading
import time


class DelayingHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        # The test sets DEBUGINFOD_TIMEOUT=1
        time.sleep(5)
        self.send_response(501)


httpd = http.server.HTTPServer(("localhost", 0), DelayingHandler)
port = httpd.socket.getsockname()[1]

try:
    t = threading.Thread(target=httpd.serve_forever).start()
    os.environ["DEBUGINFOD_URLS"] = f"http://localhost:{port}"
    result = subprocess.run(sys.argv[1:], capture_output=True)
    # e.g. Build ID 00: curl_easy_perform() failed: Timeout was reached
    print(result.stderr.decode())
finally:
    httpd.shutdown()
