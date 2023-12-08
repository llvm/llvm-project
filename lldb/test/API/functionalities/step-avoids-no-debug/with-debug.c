#include <stdio.h>

typedef int (*debug_callee) (int);

extern int no_debug_caller (int, debug_callee);

int
called_from_nodebug_actual(int some_value)
{
  int return_value = 0;
  return_value  = printf ("Length: %d.\n", some_value);
  return return_value; // Stop here and step out of me
}

int
called_from_nodebug(int some_value)
{
  int intermediate_return_value = 0;
  intermediate_return_value = called_from_nodebug_actual(some_value);
  return intermediate_return_value;
}

int
main()
{
  int return_value = no_debug_caller(5, called_from_nodebug);
  printf ("I got: %d.\n", return_value);
  return 0;
}
