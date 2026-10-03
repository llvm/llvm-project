// Inbounds: the pointee region is fully present; the present check passes at
// every OpenMP version.
// RUN: %libomptarget-compile-generic -fopenmp-version=52
// RUN: %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefix=CHECK-52
// RUN: %libomptarget-compile-generic -fopenmp-version=60
// RUN: %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefix=CHECK-60

// Out-of-bounds: s.p[0:20] extends beyond the mapped region x[0:10]. At OpenMP
// 6.0 the present modifier is propagated to the pointee s.p[0:20], so the
// update fails the present check; at <= 5.2 present is not applied to the
// pointee, so it succeeds.
// RUN: %libomptarget-compile-generic -fopenmp-version=52 -DOUT_OF_BOUNDS
// RUN: %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic --check-prefix=CHECK-52-OOB
// RUN: %libomptarget-compile-generic -fopenmp-version=60 -DOUT_OF_BOUNDS
// RUN: %libomptarget-run-fail-generic 2>&1 \
// RUN: | %fcheck-generic --check-prefix=CHECK-60-OOB

#include <stdio.h>

int x[10];
typedef struct {
  int y;
  int *p;
} S;

#ifdef OUT_OF_BOUNDS
#pragma omp declare mapper(S s) map(s.y, s.p[0 : 20])
#else
#pragma omp declare mapper(S s) map(s.y, s.p[0 : 2])
#endif
S s;

void f1() {
#pragma omp target update to(present : s)

#pragma omp target data use_device_addr(s, x)
#pragma omp target has_device_addr(s, x)
  {
    s.y = s.y + 222;
    x[0] = x[0] + 222;
  }
}

int main() {
  x[0] = 111;
  s.y = 111;
  s.p = &x[0];

  // CHECK-52-OOB: addr=0x[[#%x,HOST_ADDR:]], size=[[#%u,SIZE:]]
  // CHECK-60-OOB: addr=0x[[#%x,HOST_ADDR:]], size=[[#%u,SIZE:]]
  fprintf(stderr, "addr=%p, size=%zu\n", &s.p[0], 20 * sizeof(s.p[0]));

  // At 6.0 the out-of-bounds pointee fails the present check inside f1().
  // clang-format off
  // CHECK-60-OOB: message: device mapping required by 'present' motion modifier does not exist for host address 0x{{0*}}[[#HOST_ADDR]] ([[#SIZE]] bytes)
  // CHECK-60-OOB: fatal error 1: failure of target construct while offloading is mandatory
  // clang-format on
#pragma omp target data map(from : s.y, x)
  {
    f1();
  }

  // Inbounds: present check passes at both versions.
  // CHECK-52: 333 333
  // CHECK-60: 333 333
  fprintf(stderr, "%d %d\n", x[0], s.y);

  // 5.2 out-of-bounds: present is not applied to the pointee, so the update
  // completes.
  // FIXME: even at 5.2, the update should still have happened for the subset of
  // the pointee that is present (x[0:10]), instead of being silently ignored
  // (x[0] currently reads back as garbage). Once that is fixed:
  //   EXPECTED-52-OOB: 333 333
  // CHECK-52-OOB: done
  fprintf(stderr, "done\n");
}
