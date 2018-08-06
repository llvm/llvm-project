; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK: if{{.*}}add
; CHECK: if{{.*}}sub

; Function Attrs: nounwind
define i32 @f0(i32 %a0, i32 %a1, i32 %a2) #0 {
b0:
  %v0 = add i32 %a0, %a2
  %v1 = sub i32 %a1, %a2
  %v2 = select i1 undef, i32 %v0, i32 %v1
  ret i32 %v2
}

attributes #0 = { nounwind }
