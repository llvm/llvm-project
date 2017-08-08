#include <cilk/cilk.h>

int foo();

int bar();

int main() {
  int c;
  cilk_spawn {
    foo();
    bar();
    c = 2;
  }
  cilk_spawn {
    foo();
    foo();
  }
  bar();
  return 0;
}
