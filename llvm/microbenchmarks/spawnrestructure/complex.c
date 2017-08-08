#include <cilk/cilk.h>

int foo();

int bar();

int main() {
  int c;
  cilk_spawn {
    cilk_spawn {
      foo();
      bar();
      c = 2;
    }
    bar();
    cilk_spawn {
      cilk_spawn {
        cilk_spawn {
          foo();
        }
      }
      bar();
    }
    cilk_spawn {
      cilk_spawn {
        foo();
        foo();
      }
    }
  }
  return c;
}
