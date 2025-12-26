#include <bounds_safety_soft_traps.h>
#include <stdio.h>

int main(void) {
  __bounds_safety_soft_trap_s(0);
  printf("Execution continued\n");
  return 0;
}
