#include "objc-ivar-offsets.h"

int
main ()
{
  DerivedClass *mine = [[DerivedClass alloc] init];
  mine.backed_int = 1111;
  mine.unbacked_int = 2222;
  mine.derived_backed_int = 3333;
  mine.derived_unbacked_int = 4444;
  mine->flag1 = 1;
  mine->flag2 = 7;

  return 0;  // Set breakpoint here.
}
