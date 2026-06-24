#include "dylib.h"

int sink = 0;

int main() {
  // Set a breakpoint here
  if (absent_weak_function)
    sink = 6;
  if (present_weak_function)
    sink = 7;

  return sink;
}
