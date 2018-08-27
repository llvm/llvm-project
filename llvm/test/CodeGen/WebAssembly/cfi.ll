; RUN: opt -S -lowertypetests < %s | llc -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-keep-registers | FileCheck %s

; Tests that we correctly assign indexes for control flow integrity.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

@0 = private unnamed_addr constant [2 x void (...)*] [void (...)* bitcast (void ()* @f to void (...)*), void (...)* bitcast (void ()* @g to void (...)*)], align 16

; CHECK-LABEL: h:
; CHECK-NOT: .indidx
define void @h() !type !0 {
  ret void
}

; CHECK-LABEL: f:
; CHECK: .indidx 1
define void @f() !type !0 {
  ret void
}

; CHECK-LABEL: g:
; CHECK: .indidx 2
define void @g() !type !1 {
  ret void
}

!0 = !{i32 0, !"typeid1"}
!1 = !{i32 0, !"typeid2"}

declare i1 @llvm.type.test(i8* %ptr, metadata %bitset) nounwind readnone
declare void @llvm.trap() nounwind noreturn

; CHECK-LABEL: foo:
; CHECK: br_if
; CHECK: br_if
; CHECK: unreachable
define i1 @foo(i8* %p) {
  %x = call i1 @llvm.type.test(i8* %p, metadata !"typeid1")
  br i1 %x, label %contx, label %trap

trap:
  tail call void @llvm.trap() #1
  unreachable

contx:
  %y = call i1 @llvm.type.test(i8* %p, metadata !"typeid2")
  br i1 %y, label %conty, label %trap

conty:
  %z = add i1 %x, %y
  ret i1 %z
}
