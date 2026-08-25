#include <stddef.h>

static int square(int x) { return x * x; }

int sum_of_squares(const int *arr, size_t n) {
  int total = 0;
  for (size_t i = 0; i < n; ++i)
    total += square(arr[i]);
  return total;
}

void normalize(int *arr, size_t n) {
  if (n == 0) return;
  int total = sum_of_squares(arr, n);
  if (total == 0) return;
  for (size_t i = 0; i < n; ++i)
    arr[i] = (arr[i] * 100) / total;
}
