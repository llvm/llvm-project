#include <iostream>
#include "foo.h"

int main(int argc, char const *argv[]) {
  std::cout << "Hello World!" << std::endl; // main breakpoint 1
  foo();
  return 0;
}
