; RUN: llc < %s | FileCheck %s
; <rdar://problem/10497732>

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32"
target triple = "thumbv7-apple-darwin10"

@x1 = internal global i32 1
@x2 = internal global i64 12

define i64 @f() {
  %ax = load i32* @x1
  %a = zext i32 %ax to i64
  %b = load i64* @x2
  %c = add i64 %a, %b
  ret i64 %c
}

; We can global-merge the i64 in theory, but the current code doesn't handle
; the alignment correctly; for the moment, just check that we don't do it.
; See also 

; CHECK-NOT: MergedGlobals
; CHECK: _x2
; CHECK-NOT: MergedGlobals
