#include <unistd.h>

// The mock accelerator plugin sets breakpoints on these hooks.
void mock_gpu_accelerator_initialize(void) {}
void mock_gpu_accelerator_connect(void) {}
int mock_gpu_accelerator_compute(int x) { return x * 2; }
int mock_gpu_accelerator_finish(void) { return 0; }

// Hitting this hook makes the plugin halt the native process once it resumes.
void mock_accelerator_halt_native_process(void) {}

volatile int g_wait_for_accelerator_halt = 1;

void wait_for_accelerator_halt(void) {
  mock_accelerator_halt_native_process();
  for (int i = 0; i < 100 && g_wait_for_accelerator_halt; ++i)
    usleep(100 * 1000);
}

int main(void) {
  mock_gpu_accelerator_initialize();
  mock_gpu_accelerator_connect();
  wait_for_accelerator_halt(); // step over this line
  return 0;
}
