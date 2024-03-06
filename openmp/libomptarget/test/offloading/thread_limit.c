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

int main() {
  int n = 1 << 20;
  int th = 12;
  int te = n / th;
// DEFAULT: 12 (MaxFlatWorkGroupSize: 
#pragma omp target
#pragma omp teams loop num_teams(te), thread_limit(th)
  for (int i = 0; i < n; i++) {
  }

// DEFAULT: 13 (MaxFlatWorkGroupSize: 
  #pragma omp target
  #pragma omp teams distribute parallel for simd num_teams(te), thread_limit(th+1) simdlen(64)
  for(int i = 0; i < n; i++) {
  }
  return 0;
}
