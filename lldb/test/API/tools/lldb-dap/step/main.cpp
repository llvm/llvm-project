#include "other.h"

int function(int x) {
  if ((x % 2) == 0)
    return function(x - 1) + x; // breakpoint 1
  else
    return x;
}

int function2() {
  int volatile value = 3; // breakpoint 2
  inlined_fn();           // position_after_step_over

  return value;
}

int main(int argc, char const *argv[]) {
  int func_result = function2();
  return function(2) - func_result; // returns 0
}
