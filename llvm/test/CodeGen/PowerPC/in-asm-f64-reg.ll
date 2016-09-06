; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu -no-integrated-as | FileCheck %s

define void @f() {
; CHECK: @f

entry:
  %0 = tail call double* asm sideeffect "qvstfdux $2,$0,$1", "=b,{r7},{f11},0,~{memory}"(i32 64, double undef, double* undef)
  ret void

; CHECK: qvstfdux 11,{{[0-9]+}},7
}
