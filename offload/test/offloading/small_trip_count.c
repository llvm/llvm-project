// clang-format off
// RUN: %libomptarget-compile-generic
// RUN: env LIBOMPTARGET_INFO=16 LIBOMPTARGET_MIN_THREADS_FOR_LOW_TRIP_COUNT=32 \
// RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefix=DEFAULT
// RUN: env LIBOMPTARGET_INFO=16 LIBOMPTARGET_MIN_THREADS_FOR_LOW_TRIP_COUNT=8 \
// RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefix=EIGHT
// No-loop and big-jump-loop kernels, which only AMDGPU has, follow the same
// rule.
// RUN: %if amdgpu %{ %libomptarget-compile-generic -fopenmp-target-fast %}
// RUN: %if amdgpu %{ env LIBOMPTARGET_INFO=16 LIBOMPTARGET_MIN_THREADS_FOR_LOW_TRIP_COUNT=32 \
// RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefix=NOLOOP %}
// RUN: %if amdgpu %{ %libomptarget-compile-generic -fopenmp-target-fast -fno-openmp-target-no-loop %}
// RUN: %if amdgpu %{ env LIBOMPTARGET_INFO=16 LIBOMPTARGET_MIN_THREADS_FOR_LOW_TRIP_COUNT=32 \
// RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefix=BJL %}
// Downstream: the opt-in AMDGPU low-trip-count knobs, for loops of up to 128
// iterations. A no-loop kernel ignores the number of blocks.
// RUN: %if amdgpu %{ %libomptarget-compile-generic %}
// RUN: %if amdgpu %{ env LIBOMPTARGET_INFO=16 LIBOMPTARGET_AMDGPU_LOW_TRIPCOUNT=128 \
// RUN:   LIBOMPTARGET_BLOCKS_FOR_LOW_TRIP_COUNT=2 LIBOMPTARGET_MIN_THREADS_FOR_LOW_TRIP_COUNT=16 \
// RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefix=KNOBS %}
// RUN: %if amdgpu %{ %libomptarget-compile-generic -fopenmp-target-fast %}
// RUN: %if amdgpu %{ env LIBOMPTARGET_INFO=16 LIBOMPTARGET_AMDGPU_LOW_TRIPCOUNT=128 \
// RUN:   LIBOMPTARGET_BLOCKS_FOR_LOW_TRIP_COUNT=2 LIBOMPTARGET_MIN_THREADS_FOR_LOW_TRIP_COUNT=16 \
// RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefix=KNOBS-NOLOOP %}
// RUN: %if amdgpu %{ %libomptarget-compile-generic -fopenmp-target-fast -fno-openmp-target-no-loop %}
// RUN: %if amdgpu %{ env LIBOMPTARGET_INFO=16 LIBOMPTARGET_AMDGPU_LOW_TRIPCOUNT=128 \
// RUN:   LIBOMPTARGET_BLOCKS_FOR_LOW_TRIP_COUNT=2 LIBOMPTARGET_MIN_THREADS_FOR_LOW_TRIP_COUNT=16 \
// RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefix=KNOBS-BJL %}

// REQUIRES: gpu

#define N 128

__attribute__((optnone)) void optnone() {}

int main() {
  // DEFAULT: Launching kernel {{.+_main_.+}} with [4,1,1] blocks and [32,1,1] threads in SPMD mode
  // EIGHT: Launching kernel {{.+_main_.+}} with [16,1,1] blocks and [8,1,1] threads in SPMD mode
  // NOLOOP: Launching kernel {{.+_main_.+}} with [4,1,1] blocks and [32,1,1] threads in SPMD-No-Loop mode
  // BJL: Launching kernel {{.+_main_.+}} with [4,1,1] blocks and [32,1,1] threads in SPMD-Big-Jump-Loop mode
  // KNOBS: Launching kernel {{.+_main_.+}} with [2,1,1] blocks and [16,1,1] threads in SPMD mode
  // KNOBS-NOLOOP: Launching kernel {{.+_main_.+}} with [8,1,1] blocks and [16,1,1] threads in SPMD-No-Loop mode
  // KNOBS-BJL: Launching kernel {{.+_main_.+}} with [2,1,1] blocks and [16,1,1] threads in SPMD-Big-Jump-Loop mode
#pragma omp target teams distribute parallel for simd
  for (int i = 0; i < N; ++i) {
    optnone();
  }
  // DEFAULT: Launching kernel {{.+_main_.+}} with [4,1,1] blocks and [32,1,1] threads in SPMD mode
  // EIGHT: Launching kernel {{.+_main_.+}} with [16,1,1] blocks and [8,1,1] threads in SPMD mode
  // NOLOOP: Launching kernel {{.+_main_.+}} with [4,1,1] blocks and [32,1,1] threads in SPMD-No-Loop mode
  // BJL: Launching kernel {{.+_main_.+}} with [4,1,1] blocks and [32,1,1] threads in SPMD-Big-Jump-Loop mode
  // KNOBS: Launching kernel {{.+_main_.+}} with [2,1,1] blocks and [16,1,1] threads in SPMD mode
  // KNOBS-NOLOOP: Launching kernel {{.+_main_.+}} with [8,1,1] blocks and [16,1,1] threads in SPMD-No-Loop mode
  // KNOBS-BJL: Launching kernel {{.+_main_.+}} with [2,1,1] blocks and [16,1,1] threads in SPMD-Big-Jump-Loop mode
#pragma omp target teams distribute parallel for simd
  for (int i = 0; i < N - 1; ++i) {
    optnone();
  }
  // DEFAULT: Launching kernel {{.+_main_.+}} with [5,1,1] blocks and [32,1,1] threads in SPMD mode
  // EIGHT: Launching kernel {{.+_main_.+}} with [17,1,1] blocks and [8,1,1] threads in SPMD mode
  // NOLOOP: Launching kernel {{.+_main_.+}} with [5,1,1] blocks and [32,1,1] threads in SPMD-No-Loop mode
  // BJL: Launching kernel {{.+_main_.+}} with [5,1,1] blocks and [32,1,1] threads in SPMD-Big-Jump-Loop mode
  // KNOBS: Launching kernel {{.+_main_.+}} with [9,1,1] blocks and [16,1,1] threads in SPMD mode
  // KNOBS-NOLOOP: Launching kernel {{.+_main_.+}} with [9,1,1] blocks and [16,1,1] threads in SPMD-No-Loop mode
  // KNOBS-BJL: Launching kernel {{.+_main_.+}} with [9,1,1] blocks and [16,1,1] threads in SPMD-Big-Jump-Loop mode
#pragma omp target teams distribute parallel for simd
  for (int i = 0; i < N + 1; ++i) {
    optnone();
  }
  // DEFAULT: Launching kernel {{.+_main_.+}} with [32,1,1] blocks and [4,1,1] threads in SPMD mode
  // EIGHT: Launching kernel {{.+_main_.+}} with [32,1,1] blocks and [4,1,1] threads in SPMD mode
  // NOLOOP: Launching kernel {{.+_main_.+}} with [32,1,1] blocks and [4,1,1] threads in SPMD-No-Loop mode
  // BJL: Launching kernel {{.+_main_.+}} with [32,1,1] blocks and [4,1,1] threads in SPMD-Big-Jump-Loop mode
  // KNOBS: Launching kernel {{.+_main_.+}} with [32,1,1] blocks and [4,1,1] threads in SPMD mode
  // KNOBS-NOLOOP: Launching kernel {{.+_main_.+}} with [32,1,1] blocks and [4,1,1] threads in SPMD-No-Loop mode
  // KNOBS-BJL: Launching kernel {{.+_main_.+}} with [32,1,1] blocks and [4,1,1] threads in SPMD-Big-Jump-Loop mode
#pragma omp target teams distribute parallel for simd thread_limit(4)
  for (int i = 0; i < N; ++i) {
    optnone();
  }
  // A cross-team reduction.
  // KNOBS: Launching kernel {{.+_main_.+}} with [2,1,1] blocks and [16,1,1] threads in SPMD mode
  // KNOBS-NOLOOP: Launching kernel {{.+_main_.+}} with [2,1,1] blocks and [16,1,1] threads in SPMD mode
  // KNOBS-BJL: Launching kernel {{.+_main_.+}} with [2,1,1] blocks and [16,1,1] threads in SPMD mode
  int Sum = 0;
#pragma omp target teams distribute parallel for reduction(+ : Sum)
  for (int i = 0; i < N; ++i)
    Sum += 1;
  return Sum != N;
}

