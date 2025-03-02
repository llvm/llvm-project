; Test vector multiplication on arch15.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=arch15 | FileCheck %s

; Test a v2i64 multiplication.
define <2 x i64> @f1(<2 x i64> %dummy, <2 x i64> %val1, <2 x i64> %val2) {
; CHECK-LABEL: f1:
; CHECK: vmlg %v24, %v26, %v28
; CHECK: br %r14
  %ret = mul <2 x i64> %val1, %val2
  ret <2 x i64> %ret
}

; Test a v2i64 multiply-and-add.
define <2 x i64> @f2(<2 x i64> %dummy, <2 x i64> %val1, <2 x i64> %val2,
                     <2 x i64> %val3) {
; CHECK-LABEL: f2:
; CHECK: vmalg %v24, %v26, %v28, %v30
; CHECK: br %r14
  %mul = mul <2 x i64> %val1, %val2
  %ret = add <2 x i64> %mul, %val3
  ret <2 x i64> %ret
}

