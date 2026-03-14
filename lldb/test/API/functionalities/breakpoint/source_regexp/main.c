#include <stdio.h>
#include "a.h"

int
main_func(int input)
{
  return printf("Set B breakpoint here: %d.\n", input);
}

int
main()
{
  a_func(10);
  main_func(10);
  printf("Set a breakpoint here:\n");
  return 0;
}
