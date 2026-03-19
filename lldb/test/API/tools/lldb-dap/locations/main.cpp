#include <cstdio>

void greet() { printf("Hello"); }

int main(void) {
  int var1 = 1;
  void (*func_ptr)() = &greet;
  void (&func_ref)() = greet;
  return 0; // break here
}
