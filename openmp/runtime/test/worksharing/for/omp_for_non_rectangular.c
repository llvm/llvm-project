// RUN: %libomp-compile-and-run

// Support for collapse of non-rectangular loop nests was added in GCC 11
// UNSUPPORTED: gcc-4, gcc-5, gcc-6, gcc-7, gcc-8, gcc-9, gcc-10

#define N 10
int arr[N][N][N];

int collapsed(int mp) {
  int Cnt = 0;
#pragma omp for collapse(3)
  for (int j = 0; j < mp; ++j) {
    for (int i = j; i < mp; ++i) {
      for (int i0 = 0; i0 < N; ++i0) {
        arr[j][i][i0] = 1;
        ++Cnt;
      }
    }
  }
  return Cnt;
}

int test(int mp) {
  int Failed = 0;
  for (int j = 0; j < mp; ++j) {
    for (int i = 0; i < mp; ++i) {
      for (int i0 = 0; i0 < N; ++i0) {
        if ((i < j && arr[j][i][i0] == 0) || (i >= j && arr[j][i][i0] == 1))
          continue;
        ++Failed;
      }
    }
  }
  return Failed;
}

int main() {
  int Res = collapsed(N);
  return Res == N * (N + 1) && test(N) == 0;
}
