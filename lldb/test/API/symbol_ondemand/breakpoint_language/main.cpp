#include <stdio.h>
extern "C" int func_from_c();
extern int func_from_cpp();

int main() {
  func_from_c();
  func_from_cpp();
  return 0;
}
