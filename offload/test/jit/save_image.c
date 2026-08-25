// clang-format off
// RUN: %libomptarget-compileopt-generic -fopenmp-target-jit
// RUN: rm -f %t.image
// RUN: env LIBOMPTARGET_JIT_SAVE_IMAGE_FILENAME=%t.image %libomptarget-run-generic
// RUN: test -s %t.image
// clang-format on

// REQUIRES: gpu

int main() {
  int X = 0;

#pragma omp target map(tofrom : X)
  {
    X = 1;
  }

  return X != 1;
}
