// RUN: %libomptarget-compile-generic -fopenmp-version=61
// RUN: %libomptarget-run-generic 2>&1 | %fcheck-generic

// Since OpenMP 6.1, the host pointer of an attached pointer may be re-pointed
// while its device pointer stays attached. A FROM transfer must then leave
// the host pointer with its value before the transfer, and a skipped FROM
// transfer must not touch it at all:
//
//   enter data map(to: s, s.p[0:1], ...)    ; device s.p -> device a[0]
//   s.p = &a[1]                             ; host only
//   <construct with FROM of s>              ; host s.p must stay &a[1]

#include <stdio.h>

typedef struct {
  int *p;
  int x;
  int *q;
} S;

int a[2], b[2];
int *ptrs[4];

int main() {
#pragma omp target enter data map(to : a, b)

  // FROM skipped: ref count of s is 2 at the end of the target region.
  {
    S s = {&a[0], 0, &b[0]};
#pragma omp target enter data map(to : s, s.p[0 : 1])
    s.p = &a[1];
#pragma omp target map(tofrom : s)
    {
      s.x = 1;
    }
    // CHECK: skipped FROM: s.p kept: 1, s.x not copied back: 1
    printf("skipped FROM: s.p kept: %d, s.x not copied back: %d\n",
           s.p == &a[1], s.x == 0);
    s.p = &a[0];
#pragma omp target exit data map(delete : s) map(release : s.p[0 : 1])
  }

  // FROM performed, covering the re-pointed s.p.
  {
    S s = {&a[0], 0, &b[0]};
#pragma omp target enter data map(to : s, s.p[0 : 1])
    s.p = &a[1];
#pragma omp target exit data map(from : s) map(release : s.p[0 : 1])
    // CHECK: real FROM: s.p kept: 1
    printf("real FROM: s.p kept: %d\n", s.p == &a[1]);
  }

  // FROM of only s.p: s.p is restored; the re-pointed s.q is not touched.
  {
    S s = {&a[0], 0, &b[0]};
#pragma omp target enter data map(to : s, s.p[0 : 1], s.q[0 : 1])
    s.q = &b[1];
#pragma omp target map(always, from : s.p)
    {
    }
    // CHECK: partial FROM: s.p restored: 1, s.q kept: 1
    printf("partial FROM: s.p restored: %d, s.q kept: %d\n", s.p == &a[0],
           s.q == &b[1]);
    s.q = &b[0];
#pragma omp target exit data map(delete : s)                                   \
    map(release : s.p[0 : 1], s.q[0 : 1])
  }

  // target update from of a member that does not cover the pointer.
  {
    S s = {&a[0], 0, &b[0]};
#pragma omp target enter data map(to : s, s.p[0 : 1])
    s.p = &a[1];
#pragma omp target update from(s.x)
    // CHECK: update from(s.x): s.p kept: 1
    printf("update from(s.x): s.p kept: %d\n", s.p == &a[1]);
    s.p = &a[0];
#pragma omp target exit data map(delete : s) map(release : s.p[0 : 1])
  }

  // Strided target update from: ptrs[0] and ptrs[2] are transferred, ptrs[2]
  // is attached and restored, the re-pointed ptrs[1] is not touched.
  {
    for (int i = 0; i < 4; ++i)
      ptrs[i] = i < 2 ? &a[i] : &b[i - 2];
#pragma omp target enter data map(to : ptrs, ptrs[1][0 : 1], ptrs[2][0 : 1])
    ptrs[1] = &a[0];
#pragma omp target update from(ptrs[0 : 2 : 2])
    // CHECK: strided update from: ptrs[1] kept: 1, ptrs[2] restored: 1
    printf("strided update from: ptrs[1] kept: %d, ptrs[2] restored: %d\n",
           ptrs[1] == &a[0], ptrs[2] == &b[0]);
    ptrs[1] = &a[1];
#pragma omp target exit data map(delete : ptrs)                                \
    map(release : ptrs[1][0 : 1], ptrs[2][0 : 1])
  }

#pragma omp target exit data map(delete : a, b)
  return 0;
}
