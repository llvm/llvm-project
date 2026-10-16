#include <ptrcheck.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int bad_call(int *__counted_by(count) ptr, int count) {}

int main(int argc, char **argv) {
  const int num_bytes = sizeof(int) * 2;
  int *array = (int *)malloc(num_bytes);
  memset(array, 0, num_bytes);

  // The count argument is too large and will cause a trap.
  // This code pattern is currently missing a trap reason (rdar://100346924) and
  // so we can use it to test how `InstrumentationRuntimeBoundsSafety` handles
  // this.
  bad_call(array, 3);
  printf("Execution continued\n");
  return 0;
}
