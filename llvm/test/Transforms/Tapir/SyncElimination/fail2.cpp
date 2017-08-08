#include <cilk/cilk.h>

void func(int *a, int *b) {
  cilk_spawn {
    *a = 1;
  }
  cilk_sync;
  *b = 2;
  cilk_sync;
}
