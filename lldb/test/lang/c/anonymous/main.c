#include <stdio.h>

struct container {
  struct {
    struct {
      int a;
      int b;
    };
    struct {
      int c;
      int d;
    } foo;
  };
};

int processor (struct container *c)
{
  return c->foo.d + c->b; // Set breakpoint 0 here.
}

int main()
{
  struct container c = { 0, 2, 0, 4 };
  
  printf("%d\n", processor(&c));

  return 0;
}
