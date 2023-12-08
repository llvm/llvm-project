#include <stdio.h>

int
test_func()
{
  return printf("I am a test function.");
}

void
break_on_me()
{
  printf("I was called.\n");
}

int
main()
{
  break_on_me();
  test_func();
  return 0;
}
