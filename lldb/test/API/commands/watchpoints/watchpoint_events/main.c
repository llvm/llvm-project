#include <stdio.h>

int 
main (int argc, char **argv)
{
  int local_var = 10; 
  printf ("local_var is: %d.\n", local_var++); // Put a breakpoint here.
  return local_var;
}
