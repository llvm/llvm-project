import http.server
import threading
import urllib.request
import time
import sys


captured_req = False


class TrivialHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()

    def log_request(self, *args, **kwargs):
        if len(self.requestline) > 0:
            if len(self.headers) > 0:
                global captured_req
                captured_req = True


# Start server on random free port, query it and check response
def can_capture_req_to_http_server():
    if sys.platform != "win32":
        return True
    try:
        httpd = http.server.HTTPServer(("127.0.0.1", 0), TrivialHandler)
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        time.sleep(0.2)
        url = f"http://127.0.0.1:{port}/"
        with urllib.request.urlopen(url) as response:
            if response.status == 200:
                return captured_req
    except Exception as e:
        print("Failed to start and query HTTP server: ", repr(e))
        return False
    finally:
        httpd.shutdown()
        thread.join()


if can_capture_req_to_http_server():
    config.available_features.add("http-server")
