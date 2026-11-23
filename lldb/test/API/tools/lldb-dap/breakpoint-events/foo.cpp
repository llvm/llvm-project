#include <stdio.h>

static void unique_function_name() {
  puts(__PRETTY_FUNCTION__); // call puts
}

int foo(int x) {
  int value = 100;
  unique_function_name();
  return x + 42;
}
