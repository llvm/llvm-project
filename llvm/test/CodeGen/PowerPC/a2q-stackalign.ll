; RUN: llc -verify-machineinstrs < %s -mtriple=ppc64-- -mcpu=a2 | FileCheck -check-prefix=CHECK-A2 %s
; RUN: llc -verify-machineinstrs < %s -mtriple=ppc64-- -mcpu=a2q | FileCheck -check-prefix=CHECK-A2Q %s
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-bgq-linux -mcpu=a2 | FileCheck -check-prefix=CHECK-BGQ %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

declare i32 @bar(i8* %a) nounwind;
define i32 @foo() nounwind {
  %p = alloca i8, i8 115
  store i8 0, i8* %p
  %r = call i32 @bar(i8* %p)
  ret i32 %r
}

; Without QPX, the allocated stack frame is 240 bytes, but with QPX
; (because we require 32-byte alignment), it is 256 bytes.
; CHECK-A2: @foo
; CHECK-A2: stdu 1, -240(1)
; CHECK-A2Q: @foo
; CHECK-A2Q: stdu 1, -256(1)
; CHECK-BGQ: @foo
; CHECK-BGQ: stdu 1, -256(1)

