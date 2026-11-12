; RUN: llc < %s | FileCheck %s

; tail call inside a function where byval argument is splitted between
; registers and stack is currently unsupported.
; XFAIL: *

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:32:64-v128:32:128-a0:0:32-n32"
target triple = "thumbv7-apple-ios5.0"

%struct.A = type <{ i16, i16, i32, i16, i16, i32, i16, [8 x %struct.B], [418 x i8], %struct.C }>
%struct.B = type <{ i32, i16, i16 }>
%struct.C = type { i16, i32, i16, i16 }

; CHECK: f
; CHECK: push {r1, r2, r3}
; CHECK: add sp, #12
; CHECK: b.w _puts

define void @f(ptr %s, ptr nocapture byval(%struct.A) %a) nounwind optsize {
entry:
  %puts = tail call i32 @puts(ptr %s)
  ret void
}

declare i32 @puts(ptr nocapture) nounwind
