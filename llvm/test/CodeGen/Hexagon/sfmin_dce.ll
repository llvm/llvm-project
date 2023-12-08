; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK: f0
; CHECK-NOT: sfmin

; Function Attrs: nounwind
define void @f0(i32 %a0, i32 %a1) #0 {
b0:
  %v0 = bitcast i32 %a0 to float
  %v1 = bitcast i32 %a1 to float
  %v2 = tail call float @llvm.hexagon.F2.sfmin(float %v0, float %v1) #0
  ret void
}

; Function Attrs: readnone
declare float @llvm.hexagon.F2.sfmin(float, float) #1

attributes #0 = { nounwind }
attributes #1 = { readnone }
