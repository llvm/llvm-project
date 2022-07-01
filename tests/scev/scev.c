#include<stdio.h>
#include<stdlib.h>
int Z=5;
// void print(int *A,int *B)
// {
//   printf("A=%d, B=%d", *A,*B);
// }

int main()
{

 int C = 10,A[10],N=5,B[12];
 for(int I = 0; I < N; I += 4) {
  // A[I+2] = C+I+2;
  // A[I+1] = C+I+1;


  // A[I+3] = C+I+3;
  // A[I] = C+I;
  A[I] = C+I;
  A[I+1] = C+I+1;
  A[I+2] = C+I+2;
  A[I+3] = C+I+3;

}
for (int i=0; i<N;i++) {
  printf("%d", A[i] );
}
// print(A,B);
return 0;
}
