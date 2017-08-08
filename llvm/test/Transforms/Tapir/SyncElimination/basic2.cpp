#include <cilk/cilk.h>

void func() {
  cilk_spawn {
  }
  cilk_sync;
  cilk_sync;
}
