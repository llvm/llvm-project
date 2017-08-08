#include <cilk/cilk.h>

int foo();

int bar();

int main() {
  int c;
  cilk_spawn {
    cilk_spawn {
      foo();
      foo();
    }
  }
  return c;
}
