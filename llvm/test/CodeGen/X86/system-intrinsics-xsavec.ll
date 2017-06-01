; RUN: llc < %s -mtriple=i686-unknown-unknown -mattr=+xsave,+xsavec | FileCheck %s

define void @test_xsavec(i8* %ptr, i32 %hi, i32 %lo) {
; CHECK-LABEL: test_xsavec
; CHECK: movl   8(%esp), %edx
; CHECK: movl   12(%esp), %eax
; CHECK: movl   4(%esp), %ecx
; CHECK: xsavec (%ecx)
  call void @llvm.x86.xsavec(i8* %ptr, i32 %hi, i32 %lo)
  ret void;
}
declare void @llvm.x86.xsavec(i8*, i32, i32)
