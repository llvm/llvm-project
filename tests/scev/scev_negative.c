#include <stdio.h>
#include <stdlib.h>
int Z = 5;

int main() {

  int C = 10, A[10], N = 5, B[12], E[12], D[12];
  for (int I = 0; I < N; I += 4) {
    D[I + 2] = C + I + 2;
    B[I + 1] = C + I + 1;
    E[I + 3] = C + I + 3;
    A[I] = C + I;
  }
  for (int i = 0; i < N; i++) {
    printf("%d%d%d%d", A[i],B[i],E[i],D[i]);
  }
  // print(A,B);
  return 0;
}
