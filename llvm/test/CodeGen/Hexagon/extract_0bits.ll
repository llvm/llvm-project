; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK: r{{[0-9:]+}} = #0

; Function Attrs: nounwind readnone
define i32 @f0() #0 {
b0:
  %v0 = tail call i64 @llvm.hexagon.S4.extractp(i64 -1, i32 0, i32 1)
  %v1 = trunc i64 %v0 to i32
  ret i32 %v1
}

; Function Attrs: nounwind readnone
declare i64 @llvm.hexagon.S4.extractp(i64, i32, i32) #0

attributes #0 = { nounwind readnone }
