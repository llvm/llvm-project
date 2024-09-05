// RUN: %libomp-compile-and-run
// RUN: env OMP_NUM_THREADS=1 %libomp-run

int main() {
  int i;
  int a[2];
  volatile int attempt = 0;

  for (i = 0; i < 10; ++i) {
    a[0] = a[1] = 0;
#pragma omp parallel for
    for (int i = 0; i < 2; i++) {
      a[i] = 2;
    }
    if (a[0] != 2 || a[1] != 2)
      return 1;

#pragma omp teams distribute parallel for if (attempt >= 2)
    for (int i = 0; i < 2; i++) {
      a[i] = 1;
    }
    if (a[0] != 1 || a[1] != 1)
      return 1;
  }

  return 0;
}
