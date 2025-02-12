#include <ptrcheck.h>
#include <stdio.h>
#include <stdlib.h>

int *__bidi_indexable array_of_bounds_safety_pointers[2];

int main() {
  puts("// break here 1");
  array_of_bounds_safety_pointers[0] =
      __unsafe_forge_bidi_indexable(int *, (int *)malloc(16), 16);

  puts("// break here 2");
  return 0;
}
