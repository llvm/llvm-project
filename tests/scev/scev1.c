#include <stdio.h>
#include <stdlib.h>
int Z = 5;

int main() {
  int A[100], B[100];
  int N = 20;
  int C = 10;
  for (int I = 0; I < N; I += 4) {
    A[I + 2] = C + I + 2;
    B[I + 1] = C + I + 1;
    A[I + 3] = C + I + 3;
    A[I] = C + I;
  }
  for (int I = 0; I < N; I++) {
    printf("%d%d", A[I], B[I]);
  }
  return 0;
}
