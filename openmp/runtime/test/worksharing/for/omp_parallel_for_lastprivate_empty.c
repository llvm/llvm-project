// RUN: %libomp-compile-and-run | FileCheck %s --match-full-lines

// An empty worksharing loop still runs the for-init of a lastprivate loop
// counter. Variables that were never assigned keep their original values.

#include <stdio.h>
#include <stdlib.h>

static void reference(void) {
  int I = -1;
  int J = -1;
  int Zero = 0;
  for (I = 0; I < Zero; ++I)
    for (J = 0; J < 3; ++J)
      ;
  printf("reference: i=%d j=%d\n", I, J);
}

// Runtime-empty path: omp.precond.empty copy-back for I only.
static void empty_lastprivate(void) {
  int I = -1;
  int J = -1;
  int Zero = 0;
#pragma omp parallel for lastprivate(I, J)
  for (I = 0; I < Zero; ++I)
    for (J = 0; J < 3; ++J)
      ;
  printf("empty: i=%d j=%d\n", I, J);
}

// Same shape with conditional lastprivate.
static void empty_conditional(void) {
  int I = -1;
  int J = -1;
  int Zero = 0;
#pragma omp parallel for lastprivate(conditional : I, J)
  for (I = 0; I < Zero; ++I)
    for (J = 0; J < 3; ++J)
      ;
  printf("conditional: i=%d j=%d\n", I, J);
}

// Constant-empty path: Sema folds the precondition away.
static void const_empty(void) {
  int I = -1;
  int J = -1;
#pragma omp parallel for lastprivate(I, J)
  for (I = 0; I < 0; ++I)
    for (J = 0; J < 3; ++J)
      ;
  printf("const-empty: i=%d j=%d\n", I, J);
}

// Outer loop runs; middle loop is empty. I finishes normally, J keeps its
// for-init, and K is not lastprivate so it stays untouched.
static void middle_empty(void) {
  int I = -1;
  int J = -1;
  int K = -1;
  int Zero = 0;
#pragma omp parallel for lastprivate(I, J)
  for (I = 0; I < 2; ++I)
    for (J = 0; J < Zero; ++J)
      for (K = 0; K < 4; ++K)
        ;
  printf("mid-empty: i=%d j=%d k=%d\n", I, J, K);
}

// Non-empty control: normal lastprivate copy-back is unchanged.
static void nonempty(void) {
  int I = -1;
  int J = -1;
#pragma omp parallel for lastprivate(I, J)
  for (I = 0; I < 2; ++I)
    for (J = 0; J < 3; ++J)
      ;
  printf("nonempty: i=%d j=%d\n", I, J);
}

// The loop counter is declared in the for. The lastprivate variable is only
// assigned in the body, which does not run.
static void decl_for_empty(void) {
  int I = -1;
#pragma omp parallel for lastprivate(I)
  for (int T = 0; T < 0; ++T)
    I = T;
  printf("decl-for: i=%d\n", I);
}

int main(void) {
  reference();
  empty_lastprivate();
  empty_conditional();
  const_empty();
  middle_empty();
  nonempty();
  decl_for_empty();
  return EXIT_SUCCESS;
}

// CHECK:      reference: i=0 j=-1
// CHECK-NEXT: empty: i=0 j=-1
// CHECK-NEXT: conditional: i=0 j=-1
// CHECK-NEXT: const-empty: i=0 j=-1
// CHECK-NEXT: mid-empty: i=2 j=0 k=-1
// CHECK-NEXT: nonempty: i=2 j=3
// CHECK-NEXT: decl-for: i=-1
