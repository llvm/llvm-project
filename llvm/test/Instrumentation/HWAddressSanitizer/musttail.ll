; Regression test for a compiler bug that caused a crash when instrumenting code
; using musttail.

; RUN: opt -S -passes=hwasan -hwasan-use-stack-safety=0 %s -o - | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-arm-unknown-eabi"

define dso_local noundef i32 @_Z3bari(i32 noundef %0) sanitize_hwaddress {
  %2 = alloca i32, align 4
  store i32 %0, i32* %2, align 4
  %3 = load i32, i32* %2, align 4
  ret i32 %3
}

define dso_local noundef i32 @_Z3fooi(i32 noundef %0) sanitize_hwaddress {
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  store i32 %0, i32* %2, align 4
  store volatile i32 5, i32* %3, align 4
  %4 = load i32, i32* %2, align 4
  %5 = load volatile i32, i32* %3, align 4
  %6 = add nsw i32 %4, %5
  ; Check we untag before the musttail.
  ; CHECK: call void @llvm.memset.p0i8.i64
  ; CHECK: musttail call
  ; CHECK-NOT: call void @llvm.memset.p0i8.i64
  %7 = musttail call noundef i32 @_Z3bari(i32 noundef %6)
  ret i32 %7
}
