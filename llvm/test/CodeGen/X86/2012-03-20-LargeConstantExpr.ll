; RUN: llc < %s -mtriple=x86_64-apple-darwin | FileCheck %s
; <rdar://problem/11070338>
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

; CHECK:      _.memset_pattern:
; CHECK-NEXT: .quad   4575657222473777152
; CHECK-NEXT: .quad   4575657222473777152

@.memset_pattern = internal unnamed_addr constant i128 or (i128 zext (i64 bitcast (<2 x float> <float 1.000000e+00, float 1.000000e+00> to i64) to i128), i128 shl (i128 zext (i64 bitcast (<2 x float> <float 1.000000e+00, float 1.000000e+00> to i64) to i128), i128 64)), align 16

define void @foo(i8* %a, i64 %b) {
  call void @memset_pattern16(i8* %a, i8* bitcast (i128* @.memset_pattern to i8*), i64 %b)
  ret void
}

declare void @memset_pattern16(i8*, i8*, i64)
