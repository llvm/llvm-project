#include <stdio.h>

class A {
public:
  int __attribute__((abi_tag("cxx11"))) test_abi_tag() {
      return 1;
  }
  int test_asm_name() asm("A_test_asm") {
      return 2;
  }
};

int main(int argc, char **argv) {
  A a;
  // Break here
  a.test_abi_tag();
  a.test_asm_name();
  return 0;
}
