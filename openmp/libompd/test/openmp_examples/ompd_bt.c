// RUN: %gdb-compile 2>&1 | tee %t.compile
// RUN: %gdb-test -x %S/ompd_bt.cmd %t 2>&1 | tee %t.out | FileCheck %s

#include <omp.h>

void subdomain(float *x, int istart, int ipoints) {
  int i;

  for (i = 0; i < ipoints; i++)
    x[istart + i] = 123.456;
}

void sub(float *x, int npoints) {
  int iam, nt, ipoints, istart;

#pragma omp parallel default(shared) private(iam, nt, ipoints, istart)
  {
    iam = omp_get_thread_num();
    nt = omp_get_num_threads();
    ipoints = npoints / nt; /* size of partition */
    istart = iam * ipoints; /* starting array index */
    if (iam == nt - 1)      /* last thread may do more */
      ipoints = npoints - istart;
    subdomain(x, istart, ipoints);
  }
}

int main() {

  omp_set_num_threads(5);
  float array[10000];

  sub(array, 10000);

  return 0;
}

// CHECK: Loaded OMPD lib successfully!

// CHECK: Enabled filter for "bt" output successfully.
// CHECK-NOT: {{__kmp.*}}

// CHECK: Disabled filter for "bt" output successfully
// CHECK: {{__kmp.*}}

// CHECK-NOT: Python Exception
// CHECK-NOT: The program is not being run.
// CHECK-NOT: No such file or directory
