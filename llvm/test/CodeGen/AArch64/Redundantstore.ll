; RUN: llc < %s -O3 -mtriple=aarch64 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
@end_of_array = common global ptr null, align 8

; CHECK-LABEL: @test
; CHECK: stur
; CHECK-NOT: stur
define ptr @test(i32 %size) {
entry:
  %0 = load ptr, ptr @end_of_array, align 8
  %conv = sext i32 %size to i64
  %and = and i64 %conv, -8
  %conv2 = trunc i64 %and to i32
  %add.ptr.sum = add nsw i64 %and, -4
  %add.ptr3 = getelementptr inbounds i8, ptr %0, i64 %add.ptr.sum
  store i32 %conv2, ptr %add.ptr3, align 4
  %add.ptr.sum9 = add nsw i64 %and, -4
  %add.ptr5 = getelementptr inbounds i8, ptr %0, i64 %add.ptr.sum9
  store i32 %conv2, ptr %add.ptr5, align 4
  ret ptr %0
}

