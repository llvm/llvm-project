#include <stdio.h>

extern "C" int a_function();
extern "C" int c_function();
extern "C" int b_function();
extern "C" int d_function();

int main() {
  a_function();
  b_function();
  c_function();
  d_function();

  puts("running"); // breakpoint here
  return 0;
}
