#include <stdio.h>
int global = 0;
int main()
{
  global = 5; // Set a breakpoint here
  __builtin_debugtrap();
  global = 10;
  __builtin_trap();
  global = 15;
  return global;
}
