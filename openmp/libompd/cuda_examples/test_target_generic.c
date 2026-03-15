// Testing generic mode of nvptx devRtl
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
    test_breakpoint();
  #pragma omp parallel for
  for (i=0; i<N; i++)
  {
    test_breakpoint();
    p[i] = v1[i] * v2[i];
  }
    test_breakpoint();
  }
//output(p, N);
}
int main() {
  printf("calling vec_mul...\n");
  vec_mult(64);
  printf("done\n");
  return 0;
}
