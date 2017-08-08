#include <cilk/cilk.h>

int foo();

int bar();

int main() {
  int c = foo();
  int d = bar();
  cilk_spawn {
    if (c) {
      foo();
    }
  }
  return foo();
}
