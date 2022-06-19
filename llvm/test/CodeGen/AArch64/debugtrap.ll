; RUN: llc -mtriple=aarch64-windows %s -o -| FileCheck %s
; RUN: llc -mtriple=aarch64-windows -fast-isel %s -o - | FileCheck %s --check-prefix=FASTISEL
; RUN: llc -mtriple=aarch64-windows -global-isel %s -o - | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnu %s -o -| FileCheck %s
; RUN: llc -mtriple=arm64-apple-ios -global-isel %s -o - | FileCheck %s
; RUN: llc -mtriple=arm64-apple-macosx -fast-isel %s -o - | FileCheck %s --check-prefix=FASTISEL

; CHECK-LABEL: test1:
; CHECK: brk #0xf000
; FASTISEL: brk #0xf000
define void @test1() noreturn nounwind  {
entry:
  tail call void @llvm.debugtrap( )
  ret void
}

declare void @llvm.debugtrap() nounwind 

; CHECK-LABEL: test_trap_func:
; CHECK: bl {{.*}}wibble

; FastISel doesn't handle trap-func-name for debugtrap.
; FASTISEL: brk
define void @test_trap_func() noreturn nounwind  {
entry:
  tail call void @llvm.debugtrap( ) "trap-func-name"="wibble"
  ret void
}