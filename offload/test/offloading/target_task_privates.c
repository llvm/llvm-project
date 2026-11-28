// RUN: %libomptarget-compile-run-and-check-generic
// RUN: %libomptarget-compileopt-run-and-check-generic

// Check private, firstprivate and lastprivate variables of tasks and taskloops
// in a target region, and the local variables of an untied task. Pointers to
// all of them are passed to the task's privates mapping function, which takes
// them in the default address space; on AMDGPU, stack variables are in a
// different one.

#include <stdio.h>

#define N 8

struct S {
  int a;
  long b[3];
};

int main(void) {
  int res[16] = {0};

#pragma omp target map(tofrom : res)
  {
    // A task with private and firstprivate scalars, arrays and structs. The
    // task's changes to its firstprivate copies are not seen by the originals.
    long fp = 5;
    int arr[N];
    for (int i = 0; i < N; ++i)
      arr[i] = i;
    struct S s = {7, {1, 2, 3}};
    long p;
#pragma omp task private(p) firstprivate(fp, arr, s) shared(res)
    {
      p = fp * 2;
      fp += 1;
      arr[0] = 100;
      s.b[1] = 20;
      res[0] = p;
      res[1] = fp;
      res[2] = arr[3] + arr[0];
      res[3] = s.a + s.b[1];
    }
#pragma omp taskwait
    res[4] = fp;
    res[5] = arr[0];
    res[6] = s.b[1];

    // A taskloop with lastprivate, firstprivate and private variables.
    // FIXME: The device runtime runs a taskloop as a single task without
    // setting its last iteration flag, so the value of a lastprivate variable
    // is not copied out. Check the task's copy instead.
    int last = -1;
    int base = 10;
    int tmp;
#pragma omp taskloop lastprivate(last) firstprivate(base) private(tmp)         \
    shared(res)
    for (int i = 0; i < N; ++i) {
      tmp = base + i;
      last = tmp;
      if (i == N - 1)
        res[7] = last;
    }

    // An untied task with local variables.
#pragma omp task untied shared(res)
    {
      int local = 42;
      long local_arr[2] = {3, 4};
      res[8] = local + local_arr[1];
    }
#pragma omp taskwait
  }

  // CHECK: task: p=10 fp=6 arr=103 s=27
  printf("task: p=%d fp=%d arr=%d s=%d\n", res[0], res[1], res[2], res[3]);
  // CHECK: originals: fp=5 arr[0]=0 s.b[1]=2
  printf("originals: fp=%d arr[0]=%d s.b[1]=%d\n", res[4], res[5], res[6]);
  // CHECK: taskloop: last=17
  printf("taskloop: last=%d\n", res[7]);
  // CHECK: untied: 46
  printf("untied: %d\n", res[8]);
  return 0;
}
