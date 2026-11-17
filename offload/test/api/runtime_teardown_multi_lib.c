// RUN: rm -rf %t.testdir && mkdir -p %t.testdir
// RUN: %clang-generic -DLIB_A -fPIC -shared %s -o %t.testdir/libompa.so
// RUN: %clang-generic -DLIB_B -fPIC -shared %s -o %t.testdir/libompb.so
// RUN: %clang-generic -DMAIN %s -Wl,-rpath,%t.testdir -L%t.testdir -lompa -lompb -o %t
// RUN: env LIBOMPTARGET_DEBUG=1 %t 2>&1 | %fcheck-generic

// The runtime is torn down only after the last module is unregistered.

// REQUIRES: libomptarget-debug

#if defined(LIB_A) || defined(LIB_B)

#ifdef LIB_A
#define FN run_a
#else
#define FN run_b
#endif

int FN(int V) {
  int X = V;
#pragma omp target map(tofrom : X)
  X += 1;
  return X;
}

#else

#include <stdio.h>

int run_a(int V);
int run_b(int V);

int main() {
  // The executable has its own offload image too, so three modules are
  // registered.
  int X = 39;
#pragma omp target map(tofrom : X)
  X += 1;
  int R = run_b(run_a(X));
  printf("R = %d\n", R);
  fflush(stdout);
  return R != 42;
}

#endif

// CHECK: R = 42
// CHECK: Done unregistering library!
// CHECK: Done unregistering library!
// CHECK: Done unregistering library!
// CHECK: Tearing down offload library!
// CHECK-NOT: Tearing down offload library!
// CHECK-NOT: Done unregistering library!
