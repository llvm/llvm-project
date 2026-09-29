; RUN: llc -verify-machineinstrs < %s -mcpu=a2 -mtriple=powerpc64-unknown-linux-gnu | FileCheck %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"

define i32 @zytest(i32 %a) nounwind {
entry:
; CHECK: @zytest
  %r = call i32 asm "lwbrx $0, ${1:y}", "=r,Z"(i32 %a) nounwind, !srcloc !0
  ret i32 %r
; CHECK: lwbrx 3, 0,
}

!0 = !{i32 101688}

