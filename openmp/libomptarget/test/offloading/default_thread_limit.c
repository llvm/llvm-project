// clang-format off
// RUN: %libomptarget-compile-generic
// RUN: env LIBOMPTARGET_INFO=16 \
// RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefix=DEFAULT

// UNSUPPORTED: nvptx64-nvidia-cuda
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO
// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
// UNSUPPORTED: x86_64-pc-linux-gnu
// UNSUPPORTED: x86_64-pc-linux-gnu-LTO

__attribute__((optnone)) int optnone() { return 1; }

int main() {
  int N = optnone() * 4098 * 32;

// DEFAULT: [[NT:(128|256)]] (MaxFlatWorkGroupSize: [[NT]]
#pragma omp target teams distribute parallel for simd
  for (int i = 0; i < N; ++i) {
    optnone();
  }
// DEFAULT: [[NT:(128|256)]] (MaxFlatWorkGroupSize: [[NT]]
#pragma omp target teams distribute parallel for simd
  for (int i = 0; i < N; ++i) {
    optnone();
  }
// DEFAULT: [[NT:(128|256)]] (MaxFlatWorkGroupSize: [[NT]]
#pragma omp target teams distribute parallel for simd
  for (int i = 0; i < N; ++i) {
    optnone();
  }
// DEFAULT: [[NT:(128|256)]] (MaxFlatWorkGroupSize: [[NT]]
#pragma omp target 
#pragma omp teams distribute parallel for
  for (int i = 0; i < N; ++i) {
    optnone();
  }
// DEFAULT: 42 (MaxFlatWorkGroupSize: 1024
#pragma omp target thread_limit(optnone() * 42)
#pragma omp teams distribute parallel for
  for (int i = 0; i < N; ++i) {
    optnone();
  }
// DEFAULT: 42 (MaxFlatWorkGroupSize: 42
#pragma omp target thread_limit(optnone() * 42) ompx_attribute(__attribute__((amdgpu_flat_work_group_size(42, 42))))
#pragma omp teams distribute parallel for
  for (int i = 0; i < N; ++i) {
    optnone();
  }
// DEFAULT: 42 (MaxFlatWorkGroupSize: 42
#pragma omp target ompx_attribute(__attribute__((amdgpu_flat_work_group_size(42, 42))))
#pragma omp teams distribute parallel for
  for (int i = 0; i < N; ++i) {
    optnone();
  }
// DEFAULT: MaxFlatWorkGroupSize: 1024
#pragma omp target
#pragma omp teams distribute parallel for num_threads(optnone() * 42)
  for (int i = 0; i < N; ++i) {
    optnone();
  }
// DEFAULT: MaxFlatWorkGroupSize: 1024
#pragma omp target teams distribute parallel for thread_limit(optnone() * 42)
  for (int i = 0; i < N; ++i) {
    optnone();
  }
// DEFAULT: MaxFlatWorkGroupSize: 1024
#pragma omp target teams distribute parallel for num_threads(optnone() * 42)
  for (int i = 0; i < N; ++i) {
    optnone();
  }
// DEFAULT: 9 (MaxFlatWorkGroupSize: 9
#pragma omp target
#pragma omp teams distribute parallel for num_threads(9)
  for (int i = 0; i < N; ++i) {
    optnone();
  }
// DEFAULT: 4 (MaxFlatWorkGroupSize: 4
#pragma omp target thread_limit(4)
#pragma omp teams distribute parallel for
  for (int i = 0; i < N; ++i) {
    optnone();
  }
// DEFAULT: 4 (MaxFlatWorkGroupSize: 4
#pragma omp target
#pragma omp teams distribute parallel for thread_limit(4)
  for (int i = 0; i < N; ++i) {
    optnone();
  }
// DEFAULT: 9 (MaxFlatWorkGroupSize: 9
#pragma omp target teams distribute parallel for num_threads(9)
  for (int i = 0; i < N; ++i) {
    optnone();
  }
// DEFAULT: 4 (MaxFlatWorkGroupSize: 4
#pragma omp target teams distribute parallel for simd thread_limit(4)
  for (int i = 0; i < N; ++i) {
    optnone();
  }
}

