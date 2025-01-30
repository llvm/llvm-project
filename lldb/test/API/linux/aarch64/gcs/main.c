#include <asm/hwcap.h>
#include <stdbool.h>
#include <sys/auxv.h>
#include <sys/prctl.h>

#ifndef HWCAP_GCS
#define HWCAP_GCS (1UL << 32)
#endif

#define PR_GET_SHADOW_STACK_STATUS 74
#define PR_SET_SHADOW_STACK_STATUS 75
#define PR_LOCK_SHADOW_STACK_STATUS 76

#define PR_SHADOW_STACK_ENABLE (1UL << 0)
#define PR_SHADOW_STACK_WRITE (1UL << 1)
#define PR_SHADOW_STACK_PUSH (1UL << 2)

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

extern void _start();
bool change_gcs_config(bool enable) {
  // The test unlocks and disables all features (excluding the main enable bit)
  // before calling this expression. Enable them again.
  unsigned long new_status =
      enable | PR_SHADOW_STACK_PUSH | PR_SHADOW_STACK_WRITE;

  if (enable) {
    // We would not be able to return from prctl().
    my_prctl(PR_SET_SHADOW_STACK_STATUS, new_status, 0, 0, 0);

    // This is a stack, so we must push in reverse order to the pops we want to
    // have later. So push the return of __lldb_expr (_start), then the return
    // address of this function (__lldb_expr).
    __asm__ __volatile__("sys	#3, C7, C7, #0, %0\n"  // gcspushm _start
                         "sys	#3, C7, C7, #0, x30\n" // gcspushm x30
                         :
                         : "r"(_start));
  } else {
    if (prctl(PR_SET_SHADOW_STACK_STATUS, new_status, 0, 0, 0) != 0)
      return false;
  }

  // Turn back on all locks.
  if (prctl(PR_LOCK_SHADOW_STACK_STATUS, ~(0UL), 0, 0, 0) != 0)
    return false;

  return true;
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

// These functions are used to observe gcspr_el0 changing as we enter them, and
// the fault we cause by changing its value. Also used to check expression
// eval can handle function calls.
int test_func2() { return 99; }

int test_func() { return test_func2(); }

int main() {
  if (!(getauxval(AT_HWCAP) & HWCAP_GCS))
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

  // For register read/write tests.
  volatile int i = test_func();

  // If this was a register test, we would have disabled GCS during the
  // test_func call. We cannot re-enable it from ptrace so skip this part in
  // this case.
  mode = get_gcs_status();
  if ((mode & 1) == 1)
    gcs_signal(); // Set break point at this line.

  return 0;
}
