; RUN: llc < %s -fast-isel -verify-machineinstrs -mtriple=x86_64-apple-darwin10
; RUN: llc < %s -fast-isel -verify-machineinstrs -mtriple=x86_64-pc-win32 | FileCheck %s -check-prefix=WIN32
; RUN: llc < %s -fast-isel -verify-machineinstrs -mtriple=x86_64-pc-win64 | FileCheck %s -check-prefix=WIN64

; Previously, this would cause an assert.
define i31 @t1(i31 %a, i31 %b, i31 %c) {
entry:
  %add = add nsw i31 %b, %a
  %add1 = add nsw i31 %add, %c
  ret i31 %add1
}

; We don't handle the Windows CC, yet.
define i32 @foo(ptr %p) {
entry:
; WIN32: foo
; WIN32: movl (%rcx), %eax
; WIN64: foo
; WIN64: movl (%rdi), %eax
  %0 = load i32, ptr %p, align 4
  ret i32 %0
}
