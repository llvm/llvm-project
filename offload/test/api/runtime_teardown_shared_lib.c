// RUN: rm -rf %t.testdir && mkdir -p %t.testdir
// RUN: %clang-generic -DSHARED_LIB -fPIC -shared %s \
// RUN:   -o %t.testdir/libompshared.so
// RUN: %clang %s -Wl,-rpath,%t.testdir -L%t.testdir -lompshared -o %t
// RUN: env LIBOMPTARGET_DEBUG=1 %t 2>&1 | %fcheck-generic

// OpenMP only in the shared library; the runtime is torn down exactly once.

// REQUIRES: libomptarget-debug

#ifdef SHARED_LIB

// Library, built with OpenMP offloading.
int run_on_device(int V) {
  int X = V;
#pragma omp target map(tofrom : X)
  X += 1;
  return X;
}

#else

// Executable, built without OpenMP.
#include <stdio.h>

int run_on_device(int V);

int main() {
  int R = run_on_device(41);
  printf("R = %d\n", R);
  return R != 42;
}

#endif

// CHECK-DAG: R = 42
// CHECK-DAG: Tearing down offload library!
// CHECK-NOT: Tearing down offload library!
