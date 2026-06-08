// The mock accelerator plugin sets its initialize breakpoint on this function.
// Using a dedicated, uniquely named function (rather than "main") ensures the
// mock plugin only affects this test program and not other inferiors launched
// by lldb-server.
void mock_gpu_accelerator_initialize(void) {}

int mock_gpu_accelerator_compute(int x) { return x * 2; }

int mock_gpu_accelerator_finish(void) { return 0; }

int main(void) {
  mock_gpu_accelerator_initialize();
  int result = mock_gpu_accelerator_compute(21);
  mock_gpu_accelerator_finish();
  return result == 42 ? 0 : 1;
}
