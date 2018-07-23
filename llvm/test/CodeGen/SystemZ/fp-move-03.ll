; Test 32-bit floating-point loads.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Test the low end of the LE range.
define float @f1(float *%src) {
; CHECK-LABEL: f1:
; CHECK: le %f0, 0(%r2)
; CHECK: br %r14
  %val = load float, float *%src
  ret float %val
}

; Test the high end of the LE range.
define float @f2(float *%src) {
; CHECK-LABEL: f2:
; CHECK: le %f0, 4092(%r2)
; CHECK: br %r14
  %ptr = getelementptr float, float *%src, i64 1023
  %val = load float, float *%ptr
  ret float %val
}

; Check the next word up, which should use LEY instead of LE.
define float @f3(float *%src) {
; CHECK-LABEL: f3:
; CHECK: ley %f0, 4096(%r2)
; CHECK: br %r14
  %ptr = getelementptr float, float *%src, i64 1024
  %val = load float, float *%ptr
  ret float %val
}

; Check the high end of the aligned LEY range.
define float @f4(float *%src) {
; CHECK-LABEL: f4:
; CHECK: ley %f0, 524284(%r2)
; CHECK: br %r14
  %ptr = getelementptr float, float *%src, i64 131071
  %val = load float, float *%ptr
  ret float %val
}

; Check the next word up, which needs separate address logic.
; Other sequences besides this one would be OK.
define float @f5(float *%src) {
; CHECK-LABEL: f5:
; CHECK: agfi %r2, 524288
; CHECK: le %f0, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr float, float *%src, i64 131072
  %val = load float, float *%ptr
  ret float %val
}

; Check the high end of the negative aligned LEY range.
define float @f6(float *%src) {
; CHECK-LABEL: f6:
; CHECK: ley %f0, -4(%r2)
; CHECK: br %r14
  %ptr = getelementptr float, float *%src, i64 -1
  %val = load float, float *%ptr
  ret float %val
}

; Check the low end of the LEY range.
define float @f7(float *%src) {
; CHECK-LABEL: f7:
; CHECK: ley %f0, -524288(%r2)
; CHECK: br %r14
  %ptr = getelementptr float, float *%src, i64 -131072
  %val = load float, float *%ptr
  ret float %val
}

; Check the next word down, which needs separate address logic.
; Other sequences besides this one would be OK.
define float @f8(float *%src) {
; CHECK-LABEL: f8:
; CHECK: agfi %r2, -524292
; CHECK: le %f0, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr float, float *%src, i64 -131073
  %val = load float, float *%ptr
  ret float %val
}

; Check that LE allows an index.
define float @f9(i64 %src, i64 %index) {
; CHECK-LABEL: f9:
; CHECK: le %f0, 4092({{%r3,%r2|%r2,%r3}})
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 4092
  %ptr = inttoptr i64 %add2 to float *
  %val = load float, float *%ptr
  ret float %val
}

; Check that LEY allows an index.
define float @f10(i64 %src, i64 %index) {
; CHECK-LABEL: f10:
; CHECK: ley %f0, 4096({{%r3,%r2|%r2,%r3}})
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 4096
  %ptr = inttoptr i64 %add2 to float *
  %val = load float, float *%ptr
  ret float %val
}
