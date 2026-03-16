#include <cstdio>

void greet() { printf("Hello"); }

struct test {
  void foo() {}
};

int main(void) {
  int var1 = 1;
  void (*func_ptr)() = &greet;
  void (&func_ref)() = greet;
  auto member_ptr = &test::foo;
  return 0; // break here
}
