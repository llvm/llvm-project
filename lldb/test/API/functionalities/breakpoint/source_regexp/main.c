#include <stdio.h>
#include "a.h"

int
main() { int argc = 0; char **argv = (char **)0;

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
