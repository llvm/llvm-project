#include <stdio.h>

int global_test_var = 10;

int
main()
{
  int test_var = 10;
  printf ("Set a breakpoint here: %d.\n", test_var);
  return global_test_var;
}
