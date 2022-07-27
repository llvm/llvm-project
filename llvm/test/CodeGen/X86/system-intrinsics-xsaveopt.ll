; RUN: llc < %s -mtriple=i686-unknown-unknown -mattr=+xsave,+xsaveopt | FileCheck %s

define void @test_xsaveopt(ptr %ptr, i32 %hi, i32 %lo) {
; CHECK-LABEL: test_xsaveopt
; CHECK: movl     8(%esp), %edx
; CHECK: movl     12(%esp), %eax
; CHECK: movl     4(%esp), %ecx
; CHECK: xsaveopt (%ecx)
  call void @llvm.x86.xsaveopt(ptr %ptr, i32 %hi, i32 %lo)
  ret void;
}
declare void @llvm.x86.xsaveopt(ptr, i32, i32)
