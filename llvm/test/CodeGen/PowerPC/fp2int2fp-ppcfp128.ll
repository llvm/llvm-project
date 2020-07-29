; RUN: llc -verify-machineinstrs -mcpu=a2 < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-bgq-linux"

define linkonce_odr double @test1(ppc_fp128 %input) {
entry:
  %conv6.i.i = fptosi ppc_fp128 %input to i64
  %conv.i = sitofp i64 %conv6.i.i to double
  ret double %conv.i

; CHECK-LABEL: @test1
; CHECK: bl __fixtfdi
; CHECK: fcfid
; CHECK: blr
}

