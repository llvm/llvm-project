#include <stdio.h>

static void unique_function_name() {
  puts(__PRETTY_FUNCTION__); // foo breakpoint 2
}

int foo(int x) {
  // foo breakpoint 1
  unique_function_name();
  return x + 42;
}
