#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int comp(const void *first, const void *second) {
  const int a = *((const int *)first);
  const int b = *((const int *)second);
  if (a == b) // qsort call
    return 0;
  if (a > b)
    return 1;
  return -1;
}

int main(int argc, char const *argv[]) {
  int numbers[] = {4, 5, 2, 3, 1, 0, 9, 8, 6, 7};
  qsort(numbers, sizeof(numbers) / sizeof(int), sizeof(int), comp);

  return 0;
}
