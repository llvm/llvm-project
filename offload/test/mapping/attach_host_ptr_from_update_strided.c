// RUN: %libomptarget-compile-generic -fopenmp-version=61
// RUN: %libomptarget-run-generic 2>&1 | %fcheck-generic
// UNSUPPORTED: apu

// Strided target updates of an array of structs whose pointers are all
// attached. arr[0:2:2] covers only arr[0] and arr[2]:
//
//   arr:     [ p0 x0 ][ p1 x1 ][ p2 x2 ][ p3 x3 ]
//   update:  [=======]         [=======]
//
// The device pointers p0/p2 stay attached after "update to", and after
// "update from" the host pointers p0/p2 keep their values from before the
// transfer, while p1/p3 are not touched, even if they were re-pointed on the
// host (allowed since OpenMP 6.1).

#include <stdio.h>

typedef struct {
  int *p;
  int x;
} S;

int a[4] = {10, 11, 12, 13}, b[4];
S arr[4];

static void check_host(const char *what, int *const want_p[4],
                       const int want_x[4]) {
  int ok = 1;
  for (int i = 0; i < 4; ++i)
    ok &= arr[i].p == want_p[i] && arr[i].x == want_x[i];
  printf("%s: %s\n", what, ok ? "ok" : "wrong");
}

int main() {
  for (int i = 0; i < 4; ++i)
    arr[i] = (S){&a[i], i};
#pragma omp target enter data map(to : a)
#pragma omp target enter data map(to : arr, arr[0].p[0 : 1], arr[1].p[0 : 1],  \
                                      arr[2].p[0 : 1], arr[3].p[0 : 1])

  // update to: x0/x2 are updated, all device pointers stay attached.
  for (int i = 0; i < 4; ++i)
    arr[i].x = 100 + i;
#pragma omp target update to(arr[0 : 2 : 2])
  int dev_p[4], dev_x[4];
#pragma omp target map(from : dev_p, dev_x)
  for (int i = 0; i < 4; ++i) {
    dev_p[i] = *arr[i].p;
    dev_x[i] = arr[i].x;
  }
  // CHECK: update to: *p = 10 11 12 13, x = 100 1 102 3
  printf("update to: *p = %d %d %d %d, x = %d %d %d %d\n", dev_p[0], dev_p[1],
         dev_p[2], dev_p[3], dev_x[0], dev_x[1], dev_x[2], dev_x[3]);

  // update from, after re-pointing p1 and p2 on the host.
#pragma omp target
  for (int i = 0; i < 4; ++i)
    arr[i].x = 200 + i;
  arr[1].p = &b[1];
  arr[2].p = &b[2];
#pragma omp target update from(arr[0 : 2 : 2])
  {
    int *const want_p[4] = {&a[0], &b[1], &b[2], &a[3]};
    const int want_x[4] = {200, 101, 202, 103};
    // CHECK: update from: ok
    check_host("update from", want_p, want_x);
  }

  // The same with nowait: the restore happens when the update completes.
#pragma omp target
  for (int i = 0; i < 4; ++i)
    arr[i].x = 300 + i;
  for (int i = 0; i < 4; ++i)
    arr[i].p = &b[i];
#pragma omp parallel num_threads(2)
#pragma omp single
  {
#pragma omp target update from(arr[0 : 2 : 2]) nowait
#pragma omp taskwait
  }
  {
    int *const want_p[4] = {&b[0], &b[1], &b[2], &b[3]};
    const int want_x[4] = {300, 101, 302, 103};
    // CHECK: update from nowait: ok
    check_host("update from nowait", want_p, want_x);
  }

  for (int i = 0; i < 4; ++i)
    arr[i].p = &a[i];
#pragma omp target exit data map(delete : arr, a)
  return 0;
}
