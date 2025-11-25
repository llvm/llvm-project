// Testing spmd mode
#include <stdio.h>

#pragma omp declare target
void test_breakpoint() {
  asm("");
}
#pragma omp end declare target

void vec_mult(int N)
{
  int i;
  float p[N], v1[N], v2[N];
  //init(v1, v2, N);
  #pragma omp target map(v1, v2, p)
  {
  #pragma omp parallel for
  for (i=0; i<N; i++)
  {
    test_breakpoint();
    p[i] = v1[i] * v2[i];
  }
  }
//output(p, N);
}
int main() {
  printf("calling vec_mul...\n");
  vec_mult(2048);
  printf("done\n");
  return 0;
}
