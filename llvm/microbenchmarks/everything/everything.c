#include <cilk/cilk.h>
#include <math.h>

int foo() {
  return 10;
}

int bar();

int main() {
  double c = foo();
  cilk_spawn {
    c += sin(c);
    c += sin(c);
    c += sin(c);
  }
  cilk_spawn {
    cilk_spawn {
      c += sin(c);
      c += sin(c);
      c += sin(c);
    }
  }
  cilk_spawn {
    if (c) {
      c += sin(c);
      c += sin(c);
      c += sin(c);
    }
  }
  return c;
}
