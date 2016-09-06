; RUN: llc < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @test(i256* %a, i256* %b, i256* %out) #0 {
entry:
  %av = load i256, i256* %a
  %bv = load i256, i256* %b
  %r = mul i256 %av, %bv
  store i256 %r, i256* %out
  ret void
}

; CHECK-LABEL: @test
; There is a lot of inter-register motion, and so matching the instruction
; sequence will be fragile. There should be 6 underlying multiplications.
; CHECK: imulq
; CHECK: imulq
; CHECK: imulq
; CHECK: imulq
; CHECK: imulq
; CHECK: imulq
; CHECK-NOT: imulq
; CHECK: retq

attributes #0 = { norecurse nounwind uwtable "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" }

