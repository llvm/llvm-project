; RUN: llc < %s -mtriple=x86_64-apple-macosx10.7.0 -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-apple-macosx10.7.0 -verify-machineinstrs -O0 | FileCheck %s

define void @test1(ptr %ptr, i32 %val1) {
; CHECK: test1
; CHECK: xchgl	%esi, (%rdi)
  store atomic i32 %val1, ptr %ptr seq_cst, align 4
  ret void
}

define void @test2(ptr %ptr, i32 %val1) {
; CHECK: test2
; CHECK: movl	%esi, (%rdi)
  store atomic i32 %val1, ptr %ptr release, align 4
  ret void
}

define i32 @test3(ptr %ptr) {
; CHECK: test3
; CHECK: movl	(%rdi), %eax
  %val = load atomic i32, ptr %ptr seq_cst, align 4
  ret i32 %val
}
