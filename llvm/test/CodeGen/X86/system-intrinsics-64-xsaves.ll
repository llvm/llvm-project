; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+xsave,+xsaves | FileCheck %s

define void @test_xsaves(i8* %ptr, i32 %hi, i32 %lo) {
; CHECK-LABEL: test_xsaves
; CHECK: movl   %edx, %eax
; CHECK: movl   %esi, %edx
; CHECK: xsaves (%rdi)
  call void @llvm.x86.xsaves(i8* %ptr, i32 %hi, i32 %lo)
  ret void;
}
declare void @llvm.x86.xsaves(i8*, i32, i32)

define void @test_xsaves64(i8* %ptr, i32 %hi, i32 %lo) {
; CHECK-LABEL: test_xsaves64
; CHECK: movl     %edx, %eax
; CHECK: movl     %esi, %edx
; CHECK: xsaves64 (%rdi)
  call void @llvm.x86.xsaves64(i8* %ptr, i32 %hi, i32 %lo)
  ret void;
}
declare void @llvm.x86.xsaves64(i8*, i32, i32)

define void @test_xrstors(i8* %ptr, i32 %hi, i32 %lo) {
; CHECK-LABEL: test_xrstors
; CHECK: movl    %edx, %eax
; CHECK: movl    %esi, %edx
; CHECK: xrstors (%rdi)
  call void @llvm.x86.xrstors(i8* %ptr, i32 %hi, i32 %lo)
  ret void;
}
declare void @llvm.x86.xrstors(i8*, i32, i32)

define void @test_xrstors64(i8* %ptr, i32 %hi, i32 %lo) {
; CHECK-LABEL: test_xrstors64
; CHECK: movl      %edx, %eax
; CHECK: movl      %esi, %edx
; CHECK: xrstors64 (%rdi)
  call void @llvm.x86.xrstors64(i8* %ptr, i32 %hi, i32 %lo)
  ret void;
}
declare void @llvm.x86.xrstors64(i8*, i32, i32)
