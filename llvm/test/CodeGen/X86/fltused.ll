; The purpose of this test to to verify that the fltused symbol is emitted when
; any function is called with floating point arguments on Windows. And that it
; is not emitted otherwise.

; RUN: llc < %s -mtriple i686-pc-win32 | FileCheck %s --check-prefix WIN32
; RUN: llc < %s -mtriple x86_64-pc-win32 | FileCheck %s --check-prefix WIN64
; RUN: llc < %s -O0 -mtriple i686-pc-win32 | FileCheck %s --check-prefix WIN32
; RUN: llc < %s -O0 -mtriple x86_64-pc-win32 | FileCheck %s --check-prefix WIN64

@.str = private constant [4 x i8] c"%f\0A\00"

define i32 @main() nounwind {
entry:
  %call = tail call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), double 1.000000e+000) nounwind
  ret i32 0
}

declare i32 @printf(i8* nocapture, ...) nounwind

; WIN32: .globl __fltused
; WIN64: .globl _fltused
