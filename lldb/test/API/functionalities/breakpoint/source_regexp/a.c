#include <stdio.h>

#include "a.h"

static int
main() { int argc = 0; char **argv = (char **)0;

  return printf("Set B breakpoint here: %d", input);
}

int
a_func(int input)
{
  input += 1; // Set A breakpoint here;
  return main_func(input);
}
