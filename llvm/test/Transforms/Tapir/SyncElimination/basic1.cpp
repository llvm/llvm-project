#include <cilk/cilk.h>

void func() {
  cilk_sync;
  cilk_sync;
}
