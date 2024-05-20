#include <stdint.h>

int global = 10;

int main() {
  int count = 5;
  int *count_p = &count;

  // Add some metadata in the top byte (this will crash unless the
  // test is running with TBI enabled, but we won't dereference it)

  intptr_t scratch = (intptr_t)count_p;
  scratch |= (3ULL << 60);
  int *count_invalid_p = (int *)scratch;

  int (*main_p)() = main;
  scratch = (intptr_t)main_p;
  scratch |= (3ULL << 60);
  int (*main_invalid_p)() = (int (*)())scratch;

  int *global_p = &global;
  scratch = (intptr_t)global_p;
  scratch |= (3ULL << 60);
  int *global_invalid_p = (int *)scratch;

  return count; // break here
}
