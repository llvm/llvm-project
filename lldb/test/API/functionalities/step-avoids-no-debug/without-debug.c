#include <stdlib.h>

typedef int (*debug_callee) (int);
int no_debug_caller(int, debug_callee);
int no_debug_caller_intermediate(int, debug_callee);
void no_debug_during(void);
void no_debug_deeper(void);

int
no_debug_caller_intermediate(int input, debug_callee callee)
{
  // We want to test the case where the function we are stepping out from
  // gets called recursively while stepping out.  To avoid infinite
  // recursion, we treat input == 1 as a sign not to continue recursing.

  int return_value = 0;
  if (input < 3)
    return return_value;

  return_value = callee(input);
  no_debug_during();
  return return_value;
}

void no_debug_during() {
  no_debug_caller_intermediate(0, NULL);
  no_debug_deeper();
  no_debug_caller_intermediate(2, NULL);
}

void no_debug_deeper() { no_debug_caller_intermediate(1, NULL); }

int
no_debug_caller (int input, debug_callee callee)
{
  int return_value = 0;
  if (input != 1)
    return_value = no_debug_caller_intermediate(input, callee);
  return return_value;
}
