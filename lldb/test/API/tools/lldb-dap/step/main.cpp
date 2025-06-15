#include "Number.h"

int function(int x) {
  if ((x % 2) == 0)
    return function(x - 1) + x; // breakpoint 1
  else
    return x;
}

int function2() {
  Numbers<10> list;

  int result = 0;
  for (const int val : list) // position_after_step_over
    result++;                // breakpoint 2

  return result;
}

int main(int argc, char const *argv[]) {
  int func_result = function2();
  return function(2) + func_result; // returns 13
}
