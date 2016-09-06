; RUN: llc -verify-machineinstrs -mcpu=ppc64 < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; Function Attrs: nounwind readonly
define signext i32 @test(i32* nocapture readonly %P) #0 {
entry:
  %0 = load i32, i32* %P, align 4
  %shr = lshr i32 %0, 24
  ret i32 %shr

; CHECK-LABEL: @test
; CHECK: lbz 3, 0(3)
; CHECK: blr
}

attributes #0 = { nounwind readonly }

