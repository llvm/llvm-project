#include "module1.h"
#include "module2.h"

#include <cstdio>

int main() {
  ClassInMod1 FromMod1;
  ClassInMod2 FromMod2;

  FromMod1.VecInMod1.Member = 137;
  FromMod2.VecInMod2.Member = 42;

  std::puts("Break here");
  return 0;
}
