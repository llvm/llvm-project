// RUN: %libomptarget-compilexx-nvptx64-nvidia-cuda && %libomptarget-run-fail-nvptx64-nvidia-cuda

int main(int argc, char *argv[]) {
#pragma omp target
  { __builtin_trap(); }

  return 0;
}
