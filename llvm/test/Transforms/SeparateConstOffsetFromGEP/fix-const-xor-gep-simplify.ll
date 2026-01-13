; RUN: opt -S -passes=separate-const-offset-from-gep %s | FileCheck %s

define ptr @src(i32 %0) {
; CHECK-LABEL: @src(
; CHECK-NEXT: %base = alloca [4 x i32], align 16
; CHECK-NEXT: %2 = xor i64 0, 3
; CHECK-NEXT: %gep = getelementptr [4 x i32], ptr %base, i64 0, i64 %2
; CHECK-NEXT: ret ptr %gep

; CHECK-NOT: getelementptr i8, ptr %gep, i64 12
  %base = alloca [4 x i32], align 16
  %2 = xor i64 0, 3
  %gep = getelementptr [4 x i32], ptr %base, i64 0, i64 %2
  ret ptr %gep
}
