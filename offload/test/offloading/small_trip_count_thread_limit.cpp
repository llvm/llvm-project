// clang-format off
// RUN: %libomptarget-compilexx-generic
// RUN: env LIBOMPTARGET_INFO=16 \
// RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic

// REQUIRES: gpu

int main(int argc, char *argv[]) {
  constexpr const int block_size = 256;
  constexpr const int grid_size = 4;
  constexpr const int count = block_size * grid_size;

  int *data = new int[count];

#pragma omp target teams distribute parallel for thread_limit(block_size) map(from: data[0:count])
  for (int i = 0; i < count; ++i)
    data[i] = i;

  for (int i = 0; i < count; ++i)
    if (data[i] != i)
      return 1;

  delete[] data;

  return 0;
}

// CHECK: Launching kernel {{.*}} with [4,1,1] blocks and [256,1,1] threads in SPMD mode
