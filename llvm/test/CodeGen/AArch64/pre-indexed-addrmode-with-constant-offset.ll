; RUN: llc -mtriple=aarch64-linux-gnu < %s | FileCheck %s

; Reduced test from https://github.com/llvm/llvm-project/issues/60645.
; To check that we are generating -32 as offset for the first store.

define ptr @pr60645(ptr %ptr, i64 %t0) {
; CHECK-LABEL: pr60645:
; CHECK:       // %bb.0:
; CHECK-NEXT:    sub x8, x0, x1, lsl #2
; CHECK-NEXT:    str wzr, [x8, #-32]!
; CHECK-NEXT:    stur wzr, [x8, #-8]
; CHECK-NEXT:    ret
  %t1 = add nuw nsw i64 %t0, 8
  %t2 = mul i64 %t1, -4
  %t3 = getelementptr i8, ptr %ptr, i64 %t2
  %t4 = bitcast ptr %t3 to ptr
  store i32 0, ptr %t4, align 4
  %t5 = shl i64 %t1, 2
  %t6 = sub nuw nsw i64 -8, %t5
  %t7 = getelementptr i8, ptr %ptr, i64 %t6
  %t8 = bitcast ptr %t7 to ptr
  store i32 0, ptr %t8, align 4
  ret ptr %ptr
}
