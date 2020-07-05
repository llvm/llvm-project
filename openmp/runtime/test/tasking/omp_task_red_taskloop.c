// RUN: %libomp-compile-and-run

// Parsing error until gcc8:
// UNSUPPORTED: gcc-4, gcc-5, gcc-6, gcc-7, gcc-8

// Missing GOMP_taskgroup_reduction_(un)register in LLVM/OpenMP
// Should be removed once the functions are implemented
// XFAIL: gcc-9, gcc-10

#include <stdio.h>
#include <omp.h>

int r;

int work(int k, int l)
{
  return k + l + 1;
}
void bar(int i) {
  #pragma omp taskgroup task_reduction(+:r)
 { int th_gen = omp_get_thread_num();
  #pragma omp task in_reduction(+:r) firstprivate(i, th_gen)
  {
    r += work(i, 0);
printf("executing task (%d, 0), th %d (gen by th %d)\n", i, omp_get_thread_num(), th_gen);
  }
  #pragma omp task in_reduction(+:r) firstprivate(i, th_gen)
  {
    r += work(i, 1);
printf("executing task (%d, 1), th %d (gen by th %d)\n", i, omp_get_thread_num(), th_gen);
  }
 }
}
int foo() {
  int i;
  int th_gen = omp_get_thread_num();
  #pragma omp taskgroup task_reduction(+:r)
  {
    bar(0);
  }
printf("th %d passed bar0\n", th_gen);
  #pragma omp taskloop reduction(+:r) firstprivate(th_gen)
  for (i = 1; i < 4; ++i) {
    bar(i);
printf("th %d (gen by th %d) passed bar%d in taskloop\n", omp_get_thread_num(), th_gen, i);
  #pragma omp task in_reduction(+:r)
    r += i;
  }
  return 0;
}
// res = 2*((1+2)+(2+3)+(3+4)+(4+5)+1+2+3) = 60
#define res 60
int main()
{
  r = 0;
  #pragma omp parallel num_threads(2)
    foo();
  if (r == res) {
    return 0;
  } else {
    printf("error r = %d (!= %d)\n", r, res);
    return 1;
  }
}
