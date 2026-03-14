// RUN: %libomp-compile-and-run

// https://bugs.llvm.org/show_bug.cgi?id=26540 requested
// stack size to be propagated from master to workers.
// Library implements propagation of not too big stack
// for Linux x86_64 platform (skipped Windows for now).
//
// The test checks that workers can use more than 4MB
// of stack (4MB - was historical default for
// stack size of worker thread in runtime library).

#include <stdio.h>
#include <omp.h>
#if !defined(_WIN32)
#include <sys/resource.h> // getrlimit
#endif

#define STK 4800000

double foo(int n, int th)
{
  double arr[n];
  int i;
  double res = 0.0;
  for (i = 0; i < n; ++i) {
    arr[i] = (double)i / (n + 2);
  }
  for (i = 0; i < n; ++i) {
    res += arr[i] / n;
  }
  return res;
}

int main(int argc, char *argv[])
{
#if defined(_WIN32)
  // don't test Windows
  printf("stack propagation not implemented, skipping test...\n");
  return 0;
#else
  int status;
  double val = 0.0;
  int m = STK / 8; // > 4800000 bytes per thread
  // read stack size of calling thread, save it as default
  struct rlimit rlim;
  status = getrlimit(RLIMIT_STACK, &rlim);
  if (sizeof(void *) > 4 &&                 // do not test 32-bit systems,
      status == 0 && rlim.rlim_cur > STK) { // or small initial stack size
#pragma omp parallel reduction(+:val)
    {
      val += foo(m, omp_get_thread_num());
    }
  } else {
    printf("too small stack size limit (needs about 8MB), skipping test...\n");
    return 0;
  }
  if (val > 0.1) {
    printf("passed\n");
    return 0;
  } else {
    printf("failed, val = %f\n", val);
    return 1;
  }
#endif // _WIN32
}
