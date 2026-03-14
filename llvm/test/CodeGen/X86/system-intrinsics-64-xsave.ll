; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+xsave | FileCheck %s

define void @test_xsave(ptr %ptr, i32 %hi, i32 %lo) {
; CHECK-LABEL: test_xsave
; CHECK: movl  %edx, %eax
; CHECK: movl  %esi, %edx
; CHECK: xsave (%rdi)
  call void @llvm.x86.xsave(ptr %ptr, i32 %hi, i32 %lo)
  ret void;
}
declare void @llvm.x86.xsave(ptr, i32, i32)

define void @test_xsave64(ptr %ptr, i32 %hi, i32 %lo) {
; CHECK-LABEL: test_xsave64
; CHECK: movl    %edx, %eax
; CHECK: movl    %esi, %edx
; CHECK: xsave64 (%rdi)
  call void @llvm.x86.xsave64(ptr %ptr, i32 %hi, i32 %lo)
  ret void;
}
declare void @llvm.x86.xsave64(ptr, i32, i32)

define void @test_xrstor(ptr %ptr, i32 %hi, i32 %lo) {
; CHECK-LABEL: test_xrstor
; CHECK: movl   %edx, %eax
; CHECK: movl   %esi, %edx
; CHECK: xrstor (%rdi)
  call void @llvm.x86.xrstor(ptr %ptr, i32 %hi, i32 %lo)
  ret void;
}
declare void @llvm.x86.xrstor(ptr, i32, i32)

define void @test_xrstor64(ptr %ptr, i32 %hi, i32 %lo) {
; CHECK-LABEL: test_xrstor64
; CHECK: movl     %edx, %eax
; CHECK: movl     %esi, %edx
; CHECK: xrstor64 (%rdi)
  call void @llvm.x86.xrstor64(ptr %ptr, i32 %hi, i32 %lo)
  ret void;
}
declare void @llvm.x86.xrstor64(ptr, i32, i32)
