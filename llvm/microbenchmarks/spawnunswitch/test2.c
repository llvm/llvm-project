#include <cilk/cilk.h>

int foo();

int bar();

int main() {
  cilk_spawn {
    bar();
  }
  return foo();
}
