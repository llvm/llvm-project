void log_response(int reqest_response) {
  // fake logging logic
}

int slow_handle_request(int id) {
  // "slow" request handling logic
  for (int i = 0; i < 10; i++)
    id += 2;
  return id;
}

int fast_handle_request(int id) {
  // "fast" request handling logic
  return id + 2;
}

void iterative_handle_request_by_id(int id, int reps) {
  int response;
  for (int i = 0; i < reps; i++) {
    if (i % 2 == 0)
      response = fast_handle_request(id);
    else
      response = slow_handle_request(id);
    log_response(response);
  }
}

int main() {
  int n_requests = 10;
  for (int id = 0; id < n_requests; id++) {
    iterative_handle_request_by_id(id, 3);
  }
  return 0;
}
