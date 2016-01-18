; RUN: llc < %s -mtriple=i686-unknown-unknown -mattr=+xsave,+xsaves | FileCheck %s

define void @test_xsaves(i8* %ptr, i32 %hi, i32 %lo) {
; CHECK-LABEL: test_xsaves
; CHECK: movl   8(%esp), %edx
; CHECK: movl   12(%esp), %eax
; CHECK: movl   4(%esp), %ecx
; CHECK: xsaves (%ecx)
  call void @llvm.x86.xsaves(i8* %ptr, i32 %hi, i32 %lo)
  ret void;
}
declare void @llvm.x86.xsaves(i8*, i32, i32)

define void @test_xrstors(i8* %ptr, i32 %hi, i32 %lo) {
; CHECK-LABEL: test_xrstors
; CHECK: movl    8(%esp), %edx
; CHECK: movl    12(%esp), %eax
; CHECK: movl    4(%esp), %ecx
; CHECK: xrstors (%ecx)
  call void @llvm.x86.xrstors(i8* %ptr, i32 %hi, i32 %lo)
  ret void;
}
declare void @llvm.x86.xrstors(i8*, i32, i32)
