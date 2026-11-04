// Program used to generate guid-roundtrip.proftext. See guid-roundtrip.test
// for how to regenerate it.

#include <stdio.h>

static int local_callee(int x) { return x + 1; }

int external_callee(int x) { return x + 2; }

// Volatile so the indirect calls aren't optimized away, and get
// value-profiled.
typedef int (*callee_t)(int);
volatile callee_t indirect_callee;

int main(void) {
  int sum = 0;

  indirect_callee = local_callee;
  for (int i = 0; i < 100; i++)
    sum += indirect_callee(i);

  indirect_callee = external_callee;
  for (int i = 0; i < 10; i++)
    sum += indirect_callee(i);

  printf("%d\n", sum);
  return 0;
}
