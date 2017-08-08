#include <cilk/cilk.h>

void func() {
  cilk_for (int i = 0; i < 100; i++) {
    cilk_for (int j = 0; j < 3; j++) {
    }
  }
}
