#include <cstdio>

void greet() { printf("Hello"); }

struct Test {
  void foo() {}
  virtual void bar() {}
};

int main(void) {
  int var1 = 1;
  void (*func_ptr)() = &greet;
  void (&func_ref)() = greet;
  auto member_ptr = &Test::foo;
  auto virtual_member_ptr = &Test::bar;
  return 0; // break here
}
