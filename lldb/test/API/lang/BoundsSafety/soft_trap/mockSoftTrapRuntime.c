#include <bounds_safety_soft_traps.h>
#include <ptrcheck.h>
#include <stdio.h>

#if __CLANG_BOUNDS_SAFETY_SOFT_TRAP_API_VERSION > 0
#error API version changed
#endif

// FIXME: The runtimes really shouldn't be built with `-fbounds-safety` in
// soft trap mode because of the risk of infinite recursion. However,
// there's currently no way to have source files built with different flags

void __bounds_safety_soft_trap_s(const char *reason) {
  printf("BoundsSafety check FAILED: message:\"%s\"\n", reason ? reason : "");
}

void __bounds_safety_soft_trap(void) { printf("BoundsSafety check FAILED\n"); }
