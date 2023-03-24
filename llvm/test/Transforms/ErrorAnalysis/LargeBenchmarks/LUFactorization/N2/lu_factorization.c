#include<stdio.h>

void lu_factorization(float A[2][2], float L[2][2], float U[2][2], float B[2])
{
  int i, j, k, n=2;

  for(j=0; j<n; j++) {
    for(i=0; i<n; i++) {
      if(i<=j) {
        U[i][j]=A[i][j];
        for(k=0; k<=i-1; k++)
          U[i][j]-=L[i][k]*U[k][j];
        if(i==j)
          L[i][j]=1;
        else
          L[i][j]=0;
      }
      else {
        L[i][j]=A[i][j];
        for(k=0; k<=j-1; k++)
          L[i][j]-=L[i][k]*U[k][j];
        L[i][j]/=U[j][j];
        U[i][j]=0;
      }
    }
  }

  // Printing matrix L
  printf("[L]: \n");
  for(i=0; i<n; i++) {
    for(j=0; j<n; j++)
      printf("%9.3f",L[i][j]);
    printf("\n");
  }

  // Printing matrix U
  printf("\n\n[U]: \n");
  for(i=0; i<n; i++) {
    for(j=0; j<n; j++)
      printf("%9.3f",U[i][j]);
    printf("\n");
  }
  printf("\n");
}

int main()
{
  float A[2][2]= { { 4, 12 }, { 12, 6 } }, // Matrix A
        L[2][2]= {0}, // Lower triangular matrix
        U[2][2];            // Upper triangular matrix

  float B[2]= {166, 112}; // Constant terms

  lu_factorization(A, L, U, B);

  return 0;
}