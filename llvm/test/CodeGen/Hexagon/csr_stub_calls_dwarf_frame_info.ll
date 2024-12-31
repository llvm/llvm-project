; RUN: llc -mtriple=hexagon -O2 < %s | FileCheck %s

target triple = "hexagon-unknown-linux-gnu"

declare i32 @f0(i32, i32)

; CHECK: __save_r16_through_r21
; CHECK: __restore_r16_through_r21_and_deallocframe

; Function Attrs: optsize
define i32 @f1(i32 %a0, i32 %a11, i32 %a22, i32 %a33, i32 %a44) #0 {
b0:
  %v0 = call i32 @f0(i32 1, i32 1)
  %v1 = call i32 @f0(i32 %a0, i32 %a11)
  %v2 = call i32 @f0(i32 %a22, i32 %a33)
  %v3 = call i32 @f0(i32 %a0, i32 %a44)
  ret i32 %v3
}

attributes #0 = { optsize }
