#include <stdio.h>

static int g_var = 0;

int step_out_of_me()
{
  return g_var; // Set a breakpoint here and step out.
}

void
increment_gvar() {
  g_var++;
}

int
main()
{
  int result = step_out_of_me(); // Stop here first
  increment_gvar(); // Continue to here
  return result;
}
