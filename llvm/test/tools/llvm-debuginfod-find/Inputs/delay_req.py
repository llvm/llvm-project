import http.server
import os
import subprocess
import sys
import threading


class DelayingHandler(http.server.BaseHTTPRequestHandler):
    can_respond = threading.Event()

    def do_GET(self):
        # Wait until the client has finished running.
        self.can_respond.wait()

httpd = http.server.HTTPServer(("localhost", 0), DelayingHandler)
port = httpd.socket.getsockname()[1]

try:
    t = threading.Thread(target=httpd.serve_forever).start()
    os.environ["DEBUGINFOD_URLS"] = f"http://localhost:{port}"
    result = subprocess.run(sys.argv[1:], capture_output=True, timeout=600)
    # e.g. Build ID 00: curl_easy_perform() failed: Timeout was reached
    print(result.stderr.decode())
finally:
    DelayingHandler.can_respond.set()
    httpd.shutdown()
