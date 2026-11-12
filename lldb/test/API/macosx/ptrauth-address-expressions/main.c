#include <stdio.h>

int foo () { return 10; }

int main () 
{
  int (*fptr)() = foo;
  printf ("%p\n", fptr); // break here
  return fptr();
}
