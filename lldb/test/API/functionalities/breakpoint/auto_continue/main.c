#include <stdio.h>

void
call_me()
{
  printf("Set another breakpoint here.\n");
}

int
main()
{
  int change_me = 0;
  for (int i = 0; i < 2; i++)
  {
    printf ("Set a breakpoint here: %d with: %d.\n", i, change_me);
  }
  call_me();
  return 0;
}
