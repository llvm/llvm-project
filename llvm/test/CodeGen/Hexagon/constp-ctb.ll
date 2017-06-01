; RUN: llc < %s
; REQUIRES: asserts

target datalayout = "e-m:e-p:32:32-i1:32-i64:64-a:0-v32:32-n16:32"
target triple = "hexagon-unknown--elf"

; Function Attrs: nounwind readnone
define i64 @foo() #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.S2.ct0p(i64 18)
  %1 = tail call i32 @llvm.hexagon.S2.ct1p(i64 27)
  %2 = tail call i64 @llvm.hexagon.A2.combinew(i32 %0, i32 %1)
  ret i64 %2
}

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.S2.ct0p(i64) #0

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.S2.ct1p(i64) #0

; Function Attrs: nounwind readnone
declare i64 @llvm.hexagon.A2.combinew(i32, i32) #0

attributes #0 = { nounwind readnone }

