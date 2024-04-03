; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 | FileCheck %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Check comparisons with zero. If the tested value is live after the
; comparison, load and test cannot be used to the same register.

; Compared value is used afterwards.
define i64 @f1(i64 %a, i64 %b, float %V, ptr %dst) {
; CHECK-LABEL: f1:
; CHECK: ltebr %f1, %f0
  %cond = fcmp oeq float %V, 0.0
  %res = select i1 %cond, i64 %a, i64 %b
  store volatile float %V, ptr %dst
  ret i64 %res
}

define i64 @f1m(i64 %a, i64 %b, float %V, ptr %dst) {
; CHECK-LABEL: f1m:
; CHECK: ltebr %f1, %f0
  %cond = fcmp oeq float %V, -0.0
  %res = select i1 %cond, i64 %a, i64 %b
  store volatile float %V, ptr %dst
  ret i64 %res
}

; Value only used in comparison.
define i64 @f2(i64 %a, i64 %b, float %V) {
; CHECK-LABEL: f2:
; CHECK: ltebr %f0, %f0
  %cond = fcmp oeq float %V, 0.0
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}

define i64 @f2m(i64 %a, i64 %b, float %V) {
; CHECK-LABEL: f2m:
; CHECK: ltebr %f0, %f0
  %cond = fcmp oeq float %V, -0.0
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}

; Same for double
define i64 @f3(i64 %a, i64 %b, double %V, ptr %dst) {
; CHECK-LABEL: f3:
; CHECK: ltdbr %f1, %f0
  %cond = fcmp oeq double %V, 0.0
  %res = select i1 %cond, i64 %a, i64 %b
  store volatile double %V, ptr %dst
  ret i64 %res
}

define i64 @f3m(i64 %a, i64 %b, double %V, ptr %dst) {
; CHECK-LABEL: f3m:
; CHECK: ltdbr %f1, %f0
  %cond = fcmp oeq double %V, -0.0
  %res = select i1 %cond, i64 %a, i64 %b
  store volatile double %V, ptr %dst
  ret i64 %res
}

define i64 @f4(i64 %a, i64 %b, double %V) {
; CHECK-LABEL: f4:
; CHECK: ltdbr %f0, %f0
  %cond = fcmp oeq double %V, 0.0
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}

define i64 @f4m(i64 %a, i64 %b, double %V) {
; CHECK-LABEL: f4m:
; CHECK: ltdbr %f0, %f0
  %cond = fcmp oeq double %V, -0.0
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}

; Same for fp128
define i64 @f5(i64 %a, i64 %b, fp128 %V, ptr %dst) {
; CHECK-LABEL: f5:
; CHECK: ltxbr %f1, %f0
  %cond = fcmp oeq fp128 %V, 0xL00000000000000008000000000000000
  %res = select i1 %cond, i64 %a, i64 %b
  store volatile fp128 %V, ptr %dst
  ret i64 %res
}

define i64 @f6(i64 %a, i64 %b, fp128 %V) {
; CHECK-LABEL: f6:
; CHECK: ltxbr %f0, %f0
  %cond = fcmp oeq fp128 %V, 0xL00000000000000008000000000000000
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}
