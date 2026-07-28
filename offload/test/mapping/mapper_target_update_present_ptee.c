// Inbounds: the pointee region is fully present; the present check passes at
// every OpenMP version.
// RUN: %libomptarget-compile-generic -fopenmp-version=52
// RUN: %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefix=CHECK-52
// RUN: %libomptarget-compile-generic -fopenmp-version=60
// RUN: %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefix=CHECK-60

// Out-of-bounds: s.p[0:20] extends beyond the mapped region x[0:10]. At OpenMP
// 6.0 the present modifier should be propagated to the pointee s.p[0:20], so
// this should run-fail with a present error; at <= 5.2 present is (correctly)
// not applied to the pointee, so the run succeeds.
// FIXME: the present modifier is not yet propagated to the pointee, so the OOB
// run currently succeeds at 6.0 too; propagating it is done in a follow-on.
//   EXPECTED (6.0): run-fail with a present-modifier error for s.p[0:20].
// RUN: %libomptarget-compile-generic -fopenmp-version=52 -DOUT_OF_BOUNDS
// RUN: %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic --check-prefix=CHECK-52-OOB
// RUN: %libomptarget-compile-generic -fopenmp-version=60 -DOUT_OF_BOUNDS
// RUN: %libomptarget-run-generic 2>&1 \
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

#pragma omp target data map(from : s.y, x)
  {
    f1();
  }

  // Inbounds: present check passes at both versions.
  // CHECK-52: 333 333
  // CHECK-60: 333 333
  fprintf(stderr, "%d %d\n", x[0], s.y);

  // Out-of-bounds: present is not yet propagated to the pointee, so both
  // versions currently complete past the update.
  //
  // 5.2: present is (correctly) never applied to the pointee, so the update
  // completes.
  // FIXME: even at 5.2, the update should still have happened for the subset of
  // the pointee that is present (x[0:10]), instead of being silently ignored
  // (x[0] currently reads back as garbage). Once that is fixed:
  //   EXPECTED-52-OOB: 333 333
  // CHECK-52-OOB: done
  //
  // 6.0: present should be propagated to the pointee, so the update should fail
  // the present check for s.p[0:20]. That propagation is done in a follow-on;
  // until then the run completes.
  //   EXPECTED-60-OOB: run-fail with a present-modifier error for s.p[0:20].
  // CHECK-60-OOB: done
  fprintf(stderr, "done\n");
}
