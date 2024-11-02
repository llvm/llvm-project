; RUN: opt -passes=objc-arc-contract -S < %s | FileCheck %s

; This file makes sure that clang.arc.used is removed even if no other ARC
; interesting calls are in the module.

declare void @llvm.objc.clang.arc.use(...) nounwind

; Kill calls to @llvm.objc.clang.arc.use(...)
; CHECK-LABEL: define void @test0(
; CHECK-NOT: clang.arc.use
; CHECK: }
define void @test0(i8* %a, i8* %b) {
  call void (...) @llvm.objc.clang.arc.use(i8* %a, i8* %b) nounwind
  ret void
}

