; RUN: llc -march=hexagon -mattr=+duplex -mcpu=hexagonv67t < %s | FileCheck %s

; Check that we generate two memory operations in tiny core if duplexes
; are enabled.

; CHECK: {
; CHECK: memw
; CHECK-NEXT: memw
; CHECK: }
; CHECK: {
; CHECK: memw
; CHECK-NEXT: memw
; CHECK: }

define i32 @test(ptr noalias nocapture readonly %a, ptr noalias nocapture readonly %b, i32 %n) local_unnamed_addr #0 {
entry:
  %0 = load i32, ptr %a, align 4
  %1 = load i32, ptr %b, align 4
  %mul = mul nsw i32 %1, %0
  %arrayidx.inc = getelementptr i32, ptr %a, i32 1
  %arrayidx1.inc = getelementptr i32, ptr %b, i32 1
  %2 = load i32, ptr %arrayidx.inc, align 4
  %3 = load i32, ptr %arrayidx1.inc, align 4
  %mul.1 = mul nsw i32 %3, %2
  %add.1 = add nsw i32 %mul.1, %mul
  ret i32 %add.1
}
