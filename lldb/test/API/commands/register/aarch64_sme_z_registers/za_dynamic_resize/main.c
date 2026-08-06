#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <sys/prctl.h>

// Important notes for this test:
// * Making a syscall will disable streaming mode.
// * LLDB writing to vg while in streaming mode will disable ZA
//   (this is just how ptrace works).
// * Using an instruction to write to an inactive ZA produces a SIGILL
//   (doing the same thing via ptrace does not, as the kernel activates ZA for
//   us in that case).

#ifndef PR_SME_SET_VL
#define PR_SME_SET_VL 63
#endif

#define SM_INST(c) asm volatile("msr s0_3_c4_c" #c "_3, xzr")
#define SMSTART_SM SM_INST(3)
#define SMSTART_ZA SM_INST(5)

void set_za_register(int svl, int value_offset) {
#define MAX_VL_BYTES 256
  uint8_t data[MAX_VL_BYTES];

  // ldr za will actually wrap the selected vector row, by the number of rows
  // you have. So setting one that didn't exist would actually set one that did.
  // That's why we need the streaming vector length here.
  for (int i = 0; i < svl; ++i) {
    // This may involve instructions that require the smefa64 extension.
    memset(data, i + value_offset, MAX_VL_BYTES);
    // Each one of these loads a VL sized row of ZA.
    asm volatile("mov w12, %w0\n\t"
                 "ldr za[w12, 0], [%1]\n\t" ::"r"(i),
                 "r"(&data)
                 : "w12");
  }
}

// These are used to make sure we only break in each thread once both of the
// threads have been started. Otherwise when the test does "process continue"
// it could stop in one thread and wait forever for the other one to start.
atomic_bool threadX_ready = false;
atomic_bool threadY_ready = false;

void *threadX_func(void *x_arg) {
  threadX_ready = true;
  while (!threadY_ready) {
  }

  prctl(PR_SME_SET_VL, 8 * 4);
  SMSTART_SM;
  SMSTART_ZA;
  set_za_register(8 * 4, 2);
  SMSTART_ZA; // Thread X breakpoint 1
  set_za_register(8 * 2, 2);
  return NULL; // Thread X breakpoint 2
}

void *threadY_func(void *y_arg) {
  threadY_ready = true;
  while (!threadX_ready) {
  }

  prctl(PR_SME_SET_VL, 8 * 2);
  SMSTART_SM;
  SMSTART_ZA;
  set_za_register(8 * 2, 3);
  SMSTART_ZA; // Thread Y breakpoint 1
  set_za_register(8 * 4, 3);
  return NULL; // Thread Y breakpoint 2
}

int main(int argc, char *argv[]) {
  // Expecting argument to tell us whether to enable ZA on the main thread.
  if (argc != 2)
    return 1;

  prctl(PR_SME_SET_VL, 8 * 8);
  SMSTART_SM;

  if (argv[1][0] == '1') {
    SMSTART_ZA;
    set_za_register(8 * 8, 1);
  }
  // else we do not enable ZA and lldb will show 0s for it.

  pthread_t x_thread;
  if (pthread_create(&x_thread, NULL, threadX_func, 0)) // Break in main thread
    return 1;

  pthread_t y_thread;
  if (pthread_create(&y_thread, NULL, threadY_func, 0))
    return 1;

  if (pthread_join(x_thread, NULL))
    return 2;

  if (pthread_join(y_thread, NULL))
    return 2;

  return 0;
}
