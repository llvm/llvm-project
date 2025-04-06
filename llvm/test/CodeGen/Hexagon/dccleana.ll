; RUN: llc -mtriple=hexagon < %s | FileCheck %s
; REQUIRES: asserts

; CHECK: dccleana

; Function Attrs: nounwind
declare void @llvm.hexagon.Y2.dccleana(ptr) #0

define i32 @f0(ptr %a0) {
b0:
  tail call void @llvm.hexagon.Y2.dccleana(ptr %a0)
  %v0 = load i8, ptr %a0
  %v1 = zext i8 %v0 to i32
  ret i32 %v1
}

attributes #0 = { nounwind }
