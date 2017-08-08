#include <cilk/cilk.h>

int foo();

int bar();

int main() {
  int c = 0;
  for (int i=0; i < 1000; i++) {
    cilk_spawn {
      foo();
    }
  }
  return c;
}
