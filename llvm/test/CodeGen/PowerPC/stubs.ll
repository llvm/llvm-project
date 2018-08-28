; RUN: llc -verify-machineinstrs %s -o - -mtriple=powerpc-unknown-linux-gnu | FileCheck %s
define ppc_fp128 @test1(i64 %X) nounwind readnone {
entry:
  %0 = sitofp i64 %X to ppc_fp128
  ret ppc_fp128 %0
}

; CHECK: test1:
; CHECK: bl __floatditf@PLT
