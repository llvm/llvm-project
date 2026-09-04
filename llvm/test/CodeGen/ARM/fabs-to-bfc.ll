; RUN: llc < %s -mtriple=armv5e-none-linux-gnueabi -mattr=+vfp2  | FileCheck %s -check-prefix=CHECK-VABS
; RUN: llc < %s -mtriple=armv7-none-linux-gnueabi  -mattr=+vfp3 | FileCheck  %s -check-prefix=CHECK-BFC


define double @test(double %tx) {
;CHECK-LABEL: test:
  %call = tail call double @llvm.fabs.f64(double %tx)
  ret double %call
;CHECK-VABS: vabs.f64
;CHECK-BFC: bfc
}

declare double @llvm.fabs.f64(double) readnone

