#include "dylib.h"

int sink;
void *sinkPtr;

int
doSomething()
{
  // Set a breakpoint here.
  if (absent_weak_int)
    sink = absent_weak_int;
  if (absent_weak_function)
    sinkPtr = absent_weak_function;
  if (present_weak_int)
    sink = present_weak_int;
  if (present_weak_function)
    sinkPtr = present_weak_function;
}

int
main()
{
  return doSomething();
}
