; RUN: llc -mtriple=aarch64-windows %s -o -| FileCheck %s
; RUN: llc -mtriple=aarch64-windows -fast-isel %s -o - | FileCheck %s
; RUN: llc -mtriple=aarch64-windows -global-isel %s -o - | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnu %s -o -| FileCheck %s
; RUN: llc -mtriple=arm64-apple-ios -global-isel %s -o - | FileCheck %s
; RUN: llc -mtriple=arm64-apple-macosx -fast-isel %s -o - | FileCheck %s

; CHECK-LABEL: test1:
; CHECK: brk #0xf000
define void @test1() noreturn nounwind  {
entry:
  tail call void @llvm.debugtrap( )
  ret void
}

declare void @llvm.debugtrap() nounwind 
