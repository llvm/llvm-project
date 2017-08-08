#include <cilk/cilk.h>

void func() {
  cilk_for (int i = 0; i < 10; i++) {
  }
  cilk_for (int i = 0; i < 10; i++) {
  }
}
