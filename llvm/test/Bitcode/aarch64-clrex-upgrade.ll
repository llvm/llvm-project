; RUN: llvm-as < %s | llvm-dis | FileCheck %s

; The nullary llvm.aarch64.clrex was given a CRm operand; the old form is
; upgraded to pass the previous implicit CRm value of 15.

define void @test_clrex() {
; CHECK-LABEL: define void @test_clrex()
; CHECK: call void @llvm.aarch64.clrex(i32 15)
  call void @llvm.aarch64.clrex()
  ret void
}

; CHECK: declare void @llvm.aarch64.clrex(i32)
declare void @llvm.aarch64.clrex()
