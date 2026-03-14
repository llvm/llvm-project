; The purpose of this test to verify that the fltused symbol is
; emitted when a floating point call is made on Windows.

; RUN: llc < %s -mtriple i686-pc-win32 | FileCheck %s --check-prefix WIN32
; RUN: llc < %s -mtriple x86_64-pc-win32 | FileCheck %s --check-prefix WIN64
; RUN: llc < %s -O0 -mtriple i686-pc-win32 | FileCheck %s --check-prefix WIN32
; RUN: llc < %s -O0 -mtriple x86_64-pc-win32 | FileCheck %s --check-prefix WIN64

@.str = private constant [4 x i8] c"%f\0A\00"

define i32 @main() nounwind {
entry:
  %call = tail call i32 (ptr, ...) @printf(ptr @.str, double 1.000000e+000) nounwind
  ret i32 0
}

declare i32 @printf(ptr nocapture, ...) nounwind

; WIN32: .globl __fltused
; WIN64: .globl _fltused
