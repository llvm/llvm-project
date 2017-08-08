#include <cilk/cilk.h>

int foo();

int bar();

int main() {
  int c = foo();
  if (c*2 > 1) {
    cilk_spawn {
      if (c > 1) {
        bar();
      } else {
        foo();
      }
    }
  } else if (c*3 < 1) {
    cilk_spawn {
      bar();
    }
  } else {
    cilk_spawn {
      foo();
    }
  }
  return c;
}
