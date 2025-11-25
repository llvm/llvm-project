#include <stdio.h>
#include <omp.h>

#pragma omp declare target
void test_breakpoint() {
  asm("");
}
#pragma omp end declare target

void vec_mult(int N)
{
  int i;
  float p[N], v1[N], v2[N];
  omp_set_nested(1);
  #pragma omp target map(v1, v2, p)
  {
    omp_set_nested(1);
  #pragma omp parallel shared(v1, v2, p, N) num_threads(4)
  {
    printf("Outer region - thread ID: %d\n", omp_get_thread_num());
    #pragma omp for
    for (int i = 0; i < N; ++i)
    {
      float acc = 0;
      #pragma omp parallel shared(v1, v2, p, N) num_threads(4)
      #pragma omp for
      for(int j = 0; j < N; ++j)
      {
        test_breakpoint();
        p[i] += v1[i] + v2[i];
      }
    }
  }
    printf("End of target region\n");
  }
//output(p, N);
}
int main() {
  printf("calling vec_mul...\n");
  vec_mult(64);
  printf("done\n");
  return 0;
}
