; The purpose of this test to verify that the fltused symbol is
; emitted when floating point operations are used on Windows.

; RUN: llc < %s -mtriple i686-pc-win32 | FileCheck %s --check-prefix WIN32
; RUN: llc < %s -mtriple x86_64-pc-win32 | FileCheck %s --check-prefix WIN64
; RUN: llc < %s -O0 -mtriple i686-pc-win32 | FileCheck %s --check-prefix WIN32
; RUN: llc < %s -O0 -mtriple x86_64-pc-win32 | FileCheck %s --check-prefix WIN64

define i32 @foo(i32 %a) nounwind {
entry:
  %da = sitofp i32 %a to double
  %div = fdiv double %da, 3.100000e+00
  %res = fptosi double %div to i32
  ret i32 %res
}

; WIN32: .globl __fltused
; WIN64: .globl _fltused
