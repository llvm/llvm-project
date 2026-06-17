#include "foo.h"

int main(int argc, char const *argv[]) {
  int bp = 0; // main breakpoint 1
  foo();
  return 0;
}
