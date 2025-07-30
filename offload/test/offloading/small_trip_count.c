// clang-format off
// RUN: %libomptarget-compile-generic
// RUN: env LIBOMPTARGET_INFO=16 \
// RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefix=DEFAULT
// RUN: env LIBOMPTARGET_INFO=16 LIBOMPTARGET_MIN_THREADS_FOR_LOW_TRIP_COUNT=8 \
// RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefix=EIGHT

// REQUIRES: gpu

#define N 128

__attribute__((optnone)) void optnone() {}

int main() {
  // DEFAULT: Launching kernel {{.+_main_.+}} with [4,1,1] blocks and [32,1,1] threads in SPMD mode
  // EIGHT: Launching kernel {{.+_main_.+}} with [16,1,1] blocks and [8,1,1] threads in SPMD mode
#pragma omp target teams distribute parallel for simd
  for (int i = 0; i < N; ++i) {
    optnone();
  }
  // DEFAULT: Launching kernel {{.+_main_.+}} with [4,1,1] blocks and [32,1,1] threads in SPMD mode
  // EIGHT: Launching kernel {{.+_main_.+}} with [16,1,1] blocks and [8,1,1] threads in SPMD mode
#pragma omp target teams distribute parallel for simd
  for (int i = 0; i < N - 1; ++i) {
    optnone();
  }
  // DEFAULT: Launching kernel {{.+_main_.+}} with [5,1,1] blocks and [32,1,1] threads in SPMD mode
  // EIGHT: Launching kernel {{.+_main_.+}} with [17,1,1] blocks and [8,1,1] threads in SPMD mode
#pragma omp target teams distribute parallel for simd
  for (int i = 0; i < N + 1; ++i) {
    optnone();
  }
  // DEFAULT: Launching kernel {{.+_main_.+}} with [32,1,1] blocks and [4,1,1] threads in SPMD mode
  // EIGHT: Launching kernel {{.+_main_.+}} with [32,1,1] blocks and [4,1,1] threads in SPMD mode
#pragma omp target teams distribute parallel for simd thread_limit(4)
  for (int i = 0; i < N; ++i) {
    optnone();
  }
}

