#define N 10
void foo(float A[restrict], double B[restrict], char C[restrict],
         int D[restrict], long E[restrict]) {
  for (long i = 0; i < N; i++)
    A[i] += B[i] + C[i] + D[i] + E[i];
}

int main() {
  float A[N];
  double B[N];
  char C[N];
  int D[N];
  long E[N];

  for (long i = 0; i < N; i++) {
    __sync_synchronize();
    A[i] = B[i] = C[i] = D[i] = E[i] = 42;
  }

  foo(A, B, C, D, E);

  return A[8];
}
