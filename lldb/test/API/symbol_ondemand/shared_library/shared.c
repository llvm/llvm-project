#include "foo.h"
#include <stdio.h>

int global_shared = 897;
int main(void) {
  puts("This is a shared library test...");
  foo(); // Set breakpoint 0 here.
  return 0;
}
