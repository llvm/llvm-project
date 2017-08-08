#include <cilk/cilk.h>

int foo();

int bar();

int main() {
  int c = foo();
  if (c > 0) {
    bar();
  } else {
    foo();
  }
  return c;
}
