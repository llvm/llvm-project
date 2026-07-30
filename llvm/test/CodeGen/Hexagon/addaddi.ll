; RUN: llc -mtriple=hexagon < %s | FileCheck %s
; Check for S4_addaddi:
; CHECK: r{{[0-9]+}} = add(r{{[0-9]+}},add(r{{[0-9]+}},#2))

define i32 @fred(i32 %a0, i32 %a1, ptr nocapture %a2) #0 {
b3:
  %v4 = add nsw i32 %a0, 2
  %v5 = add nsw i32 %v4, %a1
  store i32 %v5, ptr %a2, align 4
  ret i32 undef
}

attributes #0 = { nounwind }
