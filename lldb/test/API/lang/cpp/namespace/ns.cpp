#include "ns.h"

int foo()
{
  std::printf("global foo()\n");
  return 42;
}
int func()
{
  std::printf("global func()\n");
  return 1;
}
int func(int a)
{
  std::printf("global func(int)\n");
  return a + 1;
}
void test_lookup_at_global_scope()
{
  // BP_global_scope
  std::printf("at global scope: foo() = %d\n", foo()); // eval foo(), exp: 42
  std::printf("at global scope: func() = %d\n", func()); // eval func(), exp: 1
}
