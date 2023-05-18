; RUN: llc -march=hexagon < %s
; REQUIRES: asserts

target triple = "hexagon-unknown--elf"

; Function Attrs: nounwind readnone
define i64 @foo() #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.S2.clbp(i64 291)
  %1 = tail call i64 @llvm.hexagon.A4.combineir(i32 0, i32 %0)
  ret i64 %1
}

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.S2.clbp(i64) #0

; Function Attrs: nounwind readnone
declare i64 @llvm.hexagon.A4.combineir(i32, i32) #0

attributes #0 = { nounwind readnone }

