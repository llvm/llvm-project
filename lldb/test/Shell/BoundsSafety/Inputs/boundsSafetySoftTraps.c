#include <stdio.h>
#include <bounds_safety_soft_traps.h>

#if __CLANG_BOUNDS_SAFETY_SOFT_TRAP_API_VERSION > 0
#error API version changed
#endif

void __bounds_safety_soft_trap_s(const char *reason) {
    printf("BoundsSafety check FAILED: message:\"%s\"\n", reason? reason: "");
}

void __bounds_safety_soft_trap(void) {
    printf("BoundsSafety check FAILED\n");
}

int bad_read(int index) {
  int array[] = {0, 1, 2};
  return array[index];
}

int main(int argc, char** argv) {
  bad_read(10);
  printf("Execution continued\n");
  return 0;
}
