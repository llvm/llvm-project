; RUN: llc %s -o - -mtriple=powerpc-apple-darwin9 | FileCheck %s
define ppc_fp128 @test1(i64 %X) nounwind readnone {
entry:
  %0 = sitofp i64 %X to ppc_fp128
  ret ppc_fp128 %0
}

; CHECK: _test1:
; CHECK: bl ___floatditf
