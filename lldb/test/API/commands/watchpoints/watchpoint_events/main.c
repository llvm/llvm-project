#include <stdio.h>

int 
main() { int argc = 0; char **argv = (char **)0;

  int local_var = 10; 
  printf ("local_var is: %d.\n", local_var++); // Put a breakpoint here.
  return local_var;
}
