#include <stdio.h>
#include "foo.h"

int 
main ()
{
  struct foo *my_foo_ptr;
  my_foo_ptr = GetMeAFoo();
  
  printf ("My sub foo has: %d.\n", GetMeASubFoo(my_foo_ptr)->sub_1); // Set breakpoint 0 here.

  return 0;
}
