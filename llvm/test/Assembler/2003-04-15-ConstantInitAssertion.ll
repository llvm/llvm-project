; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

; Test the case of a misformed constant initializer
; This should cause an assembler error, not an assertion failure!

; CHECK: struct initializer doesn't match struct element type
@0 = constant { i32 } { float 1.0 }
