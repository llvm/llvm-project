#include <stdio.h>

int
call_me(int input)
{
  return input;
}

int
main()
{
  int value = call_me(0); // Set a breakpoint here
  while (value < 10)
    {
      printf("Add conditions to this breakpoint: %d.\n", value++);
    }
  return 0;
}
