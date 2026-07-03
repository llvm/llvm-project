; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; Test that we generate 64-bit mutiply accumulate/subtract.

; CHECK-LABEL: f0:
; CHECK: r{{[0-9]+}}:{{[0-9]+}} += mpyu
define i64 @f0(i64 %a0, i32 %a1, i32 %a2) #0 {
b0:
  %v0 = zext i32 %a1 to i64
  %v1 = zext i32 %a2 to i64
  %v2 = mul nsw i64 %v1, %v0
  %v3 = add nsw i64 %v2, %a0
  ret i64 %v3
}

; CHECK-LABEL: f1:
; CHECK: r{{[0-9]+}}:{{[0-9]+}} -= mpyu
define i64 @f1(i64 %a0, i32 %a1, i32 %a2) #0 {
b0:
  %v0 = zext i32 %a1 to i64
  %v1 = zext i32 %a2 to i64
  %v2 = mul nsw i64 %v1, %v0
  %v3 = sub nsw i64 %a0, %v2
  ret i64 %v3
}

attributes #0 = { nounwind readnone }
