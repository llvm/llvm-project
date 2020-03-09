#include <stdio.h>

int global_var = 10;

int
main()
{
  printf("Set a breakpoint here: %d.\n", global_var);
  global_var = 20;
  printf("We should have stopped on the previous line: %d.\n", global_var);
  global_var = 30;
  return 0;
}
