#include <cilk/cilk.h>

int foo();

int bar();

int main() {
  if (foo()) {
    cilk_spawn {
      bar();
    }
  }
  return foo();
}
