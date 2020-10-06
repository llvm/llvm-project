#include <stdio.h>

static int g_var = 0;

int step_out_of_me()
{
  return g_var; // Set a breakpoint here and step out.
}

int
main()
{
  int result = step_out_of_me(); // Stop here first
  return result;
}
