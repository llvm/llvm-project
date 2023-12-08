; RUN: llc -mtriple=hexagon  -O2 -spill-func-threshold=2 < %s | FileCheck %s

declare i32 @f0(i32, i32, i32, i32, i32, i32)

; CHECK-LABEL: f1:
; CHECK: save_r16_through_r23
define i32 @f1(i32 %a0, i32 %a11, i32 %a22, i32 %a33, i32 %a4, i32 %a5) #0 {
b0:
  %v0 = call i32 @f0(i32 0, i32 1, i32 2, i32 3, i32 4, i32 5)
  %v1 = call i32 @f0(i32 %a0, i32 %a11, i32 %a22, i32 %a33, i32 %a4, i32 %a5)
  %v2 = add i32 %v0, %v1
  ret i32 %v2
}

declare i32 @f2(i32, i32, i32, i32)

; CHECK-LABEL: f3:
; CHECK: save_r16_through_r21
define i32 @f3(i32 %a0, i32 %a11, i32 %a22, i32 %a33, i32 %a44, i32 %a5) #0 {
b0:
  %v0 = call i32 @f2(i32 0, i32 1, i32 2, i32 3)
  %v1 = call i32 @f2(i32 %a0, i32 %a11, i32 %a22, i32 %a33)
  %v2 = add i32 %v0, %v1
  %v3 = add i32 %v2, %a44
  ret i32 %v3
}

declare i32 @f4(i32, i32)

; CHECK-LABEL: f5:
; CHECK-NOT: save_r16_through_r19
define i32 @f5(i32 %a0, i32 %a11, i32 %a22, i32 %a33, i32 %a4, i32 %a5) #0 {
b0:
  %v0 = call i32 @f4(i32 0, i32 1)
  %v1 = call i32 @f4(i32 %a0, i32 %a11)
  %v2 = add i32 %v0, %v1
  ret i32 %v2
}

declare i32 @f6(i32)

; CHECK-LABEL: f7:
; CHECK-NOT: save_r16_through_r17
define i32 @f7(i32 %a0, i32 %a11, i32 %a22, i32 %a3, i32 %a4, i32 %a5) #0 {
b0:
  %v0 = call i32 @f6(i32 0)
  %v1 = call i32 @f6(i32 %a0)
  ret i32 %v0
}

attributes #0 = { nounwind }
