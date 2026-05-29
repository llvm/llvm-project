; Negative test: only the i64 form of `air.simd_shuffle` is split; an i32
; shuffle must pass through untouched.
; RUN: llc -mtriple=air %s -o - | FileCheck %s

; CHECK-LABEL: define void @kernel
; CHECK-NOT: trunc i64
; CHECK-NOT: lshr i64
; CHECK: call i32 @air.simd_shuffle.i32(i32 %v
; CHECK: ret void

declare i32 @air.simd_shuffle.i32(i32, i16)

define void @kernel(i32 %v, i16 %off, ptr %out) {
entry:
  %s = call i32 @air.simd_shuffle.i32(i32 %v, i16 %off)
  store i32 %s, ptr %out
  ret void
}
