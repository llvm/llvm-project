; RUN: llc < %s | FileCheck %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define void @test(i64 %n) nounwind {
entry:
  %0 = alloca i8, i64 %n, align 1
  %1 = alloca i8, i64 %n, align 1
  call void @use(ptr %0, ptr %1) nounwind
  ret void
}

declare void @use(ptr, ptr)

; Check we actually have two instances of dynamic stack allocation,
; identified by the stdux used to update the back-chain link.
; CHECK: stdux
; CHECK: stdux
