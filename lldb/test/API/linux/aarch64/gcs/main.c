#include <asm/hwcap.h>
#include <sys/auxv.h>
#include <sys/prctl.h>

#ifndef HWCAP2_GCS
#define HWCAP2_GCS (1UL << 63)
#endif

#define PR_GET_SHADOW_STACK_STATUS 74
#define PR_SET_SHADOW_STACK_STATUS 75
#define PR_SHADOW_STACK_ENABLE (1UL)
#define PRCTL_SYSCALL_NO 167

// Once we enable GCS, we cannot return from the function that made the syscall
// to enable it. This is because the control stack is empty, there is no valid
// address for us to return to. So for the initial enable we must use inline asm
// instead of the libc's prctl wrapper function.
#define my_prctl(option, arg2, arg3, arg4, arg5)                               \
  ({                                                                           \
    register unsigned long x0 __asm__("x0") = option;                          \
    register unsigned long x1 __asm__("x1") = arg2;                            \
    register unsigned long x2 __asm__("x2") = arg3;                            \
    register unsigned long x3 __asm__("x3") = arg4;                            \
    register unsigned long x4 __asm__("x4") = arg5;                            \
    register unsigned long x8 __asm__("x8") = PRCTL_SYSCALL_NO;                \
    __asm__ __volatile__("svc #0\n"                                            \
                         : "=r"(x0)                                            \
                         : "r"(x0), "r"(x1), "r"(x2), "r"(x3), "r"(x4),        \
                           "r"(x8)                                             \
                         : "cc", "memory");                                    \
  })

unsigned long get_gcs_status() {
  unsigned long mode = 0;
  prctl(PR_GET_SHADOW_STACK_STATUS, &mode, 0, 0, 0);
  return mode;
}

void gcs_signal() {
  // If we enabled GCS manually, then we could just return from main to generate
  // a signal. However, if the C library enabled it, then we'd just exit
  // normally. Assume the latter, and try to return to some bogus address to
  // generate the signal.
  __asm__ __volatile__(
      // Corrupt the link register. This could be many numbers but 16 is a
      // nicely aligned value that is unlikely to result in a fault because the
      // PC is misaligned, which would hide the GCS fault.
      "add x30, x30, #10\n"
      "ret\n");
}

int main() {
  if (!(getauxval(AT_HWCAP2) & HWCAP2_GCS))
    return 1;

  unsigned long mode = get_gcs_status();
  if ((mode & 1) == 0) {
    // If GCS wasn't already enabled by the C library, enable it.
    my_prctl(PR_SET_SHADOW_STACK_STATUS, PR_SHADOW_STACK_ENABLE, 0, 0, 0);
    // From this point on, we cannot return from main without faulting because
    // the return address from main, and every function before that, is not on
    // the guarded control stack.
  }

  // By now we should have one memory region where the GCS is stored.
  gcs_signal(); // Set break point at this line.

  return 0;
}
