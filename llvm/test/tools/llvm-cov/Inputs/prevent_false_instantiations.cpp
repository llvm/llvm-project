#include "prevent_false_instantiations.h"

void func1() {
  DO_SOMETHING();
}

void func2() {
  DO_SOMETHING();
}

int main() {
  func1();
  func2();
  return 0;
}
