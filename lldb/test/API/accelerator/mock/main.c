// The mock accelerator plugin sets its breakpoints on these dedicated, uniquely
// named functions.
void mock_gpu_accelerator_initialize(void) {}

int mock_gpu_accelerator_compute(int x) { return x * 2; }

int mock_gpu_accelerator_finish(void) { return 0; }

// When the plugin's connection-trigger breakpoint on this function is hit, it
// asks the client to create a second target and connect to the mock accelerator
// GDB server.
void mock_gpu_accelerator_connect(void) {}

int main(void) {
  mock_gpu_accelerator_initialize();
  mock_gpu_accelerator_connect();
  int result = mock_gpu_accelerator_compute(21);
  mock_gpu_accelerator_finish();
  return result == 42 ? 0 : 1;
}
