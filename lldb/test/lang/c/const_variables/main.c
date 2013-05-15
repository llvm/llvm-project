#include <stdint.h>

extern int foo();
extern int bar();
extern int baaz(int i);

int main()
{
  int32_t index;

  foo();

  index = 512;

  if (bar())
    index = 256;

  baaz(index);
}
