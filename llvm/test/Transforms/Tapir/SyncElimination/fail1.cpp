#include <cilk/cilk.h>

void func() {
  int a;
  cilk_spawn {
    a = 1;
  }
  cilk_sync;
}
