#include <stdio.h>

void foo (int &&i)
{
  printf("%d\n", i); // breakpoint 1
}

int main()
{
  foo(3);
  return 0; // breakpoint 2
}
