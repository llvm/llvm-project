#include <bounds_safety_soft_traps.h>
#include <ptrcheck.h>
#include <stdio.h>

#if __CLANG_BOUNDS_SAFETY_SOFT_TRAP_API_VERSION > 0
#error API version changed
#endif

#if __has_ptrcheck
#error Do not compile the runtime with -fbounds-safety enabled due to potential for infinite recursion
#endif



void __bounds_safety_soft_trap_s(const char *reason) {
    printf("BoundsSafety check FAILED: message:\"%s\"\n", reason? reason: "");
}

void __bounds_safety_soft_trap(void) {
    printf("BoundsSafety check FAILED\n");
}
