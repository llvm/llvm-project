#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

int compare_ints(const void *a, const void *b) {
  int arg1 = *(const int *)a;
  int arg2 = *(const int *)b;

  // breakpoint 1

  if (arg1 < arg2)
    return -1;
  if (arg1 > arg2)
    return 1;
  return 0;
}

int main(void) {
  int ints[] = {-2, 99, 0, -743, 2, INT_MIN, 4};
  int size = sizeof ints / sizeof *ints;

  qsort(ints, size, sizeof(int), compare_ints);

  for (int i = 0; i < size; i++) {
    printf("%d ", ints[i]);
  }

  printf("\n");
  return 0;
}