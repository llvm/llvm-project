; RUN: llc -mcpu=hexagonv5 < %s
; REQUIRES: asserts

target datalayout = "e-m:e-p:32:32-i1:32-i64:64-a:0-v32:32-n16:32"
target triple = "hexagon-unknown--elf"

; Function Attrs: nounwind readnone
define i64 @foo() #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.S2.clbp(i64 291)
  %1 = tail call i64 @llvm.hexagon.A4.combineir(i32 0, i32 %0)
  ret i64 %1
}

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.S2.clbp(i64) #1

; Function Attrs: nounwind readnone
declare i64 @llvm.hexagon.A4.combineir(i32, i32) #1

attributes #0 = { nounwind readnone "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

