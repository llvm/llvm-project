; REQUIRES: asserts
; RUN: opt -S -debug-counter=partially-inline-libcalls-transform-skip=1,partially-inline-libcalls-transform-count=1 \
; RUN:     -partially-inline-libcalls -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck %s
;; Test that, with debug counters on, we will skip the first optimization opportunity, perform next 1,
;; and ignore all the others left.

define float @f1(float %val) {
; CHECK-LABEL: @f1(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[RES:%.*]] = tail call float @sqrtf(float [[VAL:%.*]])
; CHECK-NEXT:    ret float [[RES:%.*]]
entry:
  %res = tail call float @sqrtf(float %val)
  ret float %res
}

define float @f2(float %val) {
; CHECK-LABEL: @f2(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[RES:%.*]] = tail call float @sqrtf(float [[VAL:%.*]]) #0
; CHECK-NEXT:    [[TMP0:%.*]] = fcmp oge float [[VAL]], 0.000000e+00
; CHECK-NEXT:    br i1 [[TMP0]], label [[ENTRY_SPLIT:%.*]], label [[CALL_SQRT:%.*]]
; CHECK:       call.sqrt:
; CHECK-NEXT:    [[TMP1:%.*]] = tail call float @sqrtf(float [[VAL]])
; CHECK-NEXT:    br label [[ENTRY_SPLIT]]
; CHECK:       entry.split:
; CHECK-NEXT:    [[TMP2:%.*]] = phi float [ [[RES]], [[ENTRY:%.*]] ], [ [[TMP1]], [[CALL_SQRT]] ]
; CHECK-NEXT:    ret float [[TMP2]]
entry:
  %res = tail call float @sqrtf(float %val)
  ret float %res
}

define float @f3(float %val) {
; CHECK-LABEL: @f3(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[RES:%.*]] = tail call float @sqrtf(float [[VAL:%.*]])
; CHECK-NEXT:    ret float [[RES:%.*]]
entry:
  %res = tail call float @sqrtf(float %val)
  ret float %res
}

declare float @sqrtf(float)
