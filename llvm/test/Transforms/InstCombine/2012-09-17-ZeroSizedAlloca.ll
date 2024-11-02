; RUN: opt -S -passes=instcombine < %s | FileCheck %s

; When merging zero sized alloca check that requested alignments of the allocas
; are obeyed.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

@x = global ptr null, align 8
@y = global ptr null, align 8

; CHECK-LABEL: @f(
; CHECK-NEXT: alloca [0 x i8], align 1024
; CHECK-NOT: alloca
; CHECK: ret void
define void @f() {
  %1 = alloca [0 x i8], align 1
  %2 = alloca [0 x i8], align 1024
  store ptr %1, ptr @x, align 8
  store ptr %2, ptr @y, align 8
  ret void
}
