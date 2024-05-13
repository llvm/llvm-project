; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+xsave,+xsavec | FileCheck %s

define void @test_xsavec(ptr %ptr, i32 %hi, i32 %lo) {
; CHECK-LABEL: test_xsavec
; CHECK: movl   %edx, %eax
; CHECK: movl   %esi, %edx
; CHECK: xsavec (%rdi)
  call void @llvm.x86.xsavec(ptr %ptr, i32 %hi, i32 %lo)
  ret void;
}
declare void @llvm.x86.xsavec(ptr, i32, i32)

define void @test_xsavec64(ptr %ptr, i32 %hi, i32 %lo) {
; CHECK-LABEL: test_xsavec64
; CHECK: movl     %edx, %eax
; CHECK: movl     %esi, %edx
; CHECK: xsavec64 (%rdi)
  call void @llvm.x86.xsavec64(ptr %ptr, i32 %hi, i32 %lo)
  ret void;
}
declare void @llvm.x86.xsavec64(ptr, i32, i32)
