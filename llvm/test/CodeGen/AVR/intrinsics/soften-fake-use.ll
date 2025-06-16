; RUN: llc -O0 < %s -mtriple=avr 2>&1 --stop-after=finalize-isel | FileCheck %s

;; Tests that we can soften float operands to llvm.fake.use intrinsics.

define double @idd(double %d) {
entry:
  notail call void (...) @llvm.fake.use(double %d)
  ret double %d
}
