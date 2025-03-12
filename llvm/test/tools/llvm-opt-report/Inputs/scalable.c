#include <stddef.h>

void foo(size_t N, float A[restrict N], float B[N]) {
  #pragma clang loop vectorize_width(4, scalable)
  for (size_t i = 0; i < N; i++) {
    A[i] = B[i] * 42.f;
  }
}

