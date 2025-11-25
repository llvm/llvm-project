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
  #pragma omp target map(v1, v2, p)
  {
    test_breakpoint();
    p[0] = v[0] * v[0];
  }
}
int main() {
  printf("calling vec_mul...\n");
  vec_mult(64);
  printf("done\n");
  return 0;
}
