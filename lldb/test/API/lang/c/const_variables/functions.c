#include <stdio.h>

void foo()
{
  printf("foo()\n");
}

int bar()
{
  int ret = 3;
  printf("bar()->%d\n", ret);
  return ret;
}

void baaz(int i)
{
  printf("baaz(%d)\n", i);
}
