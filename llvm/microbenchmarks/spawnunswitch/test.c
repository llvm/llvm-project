#include <cilk/cilk.h>

int foo();

int bar();

int main() {
  cilk_for (int i=0; i < 1000; i++) {
    foo();
  }
  return foo();
}
