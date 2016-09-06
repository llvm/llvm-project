; RUN: not llvm-as < %s >/dev/null 2> %t
; RUN: grep "struct initializer doesn't match struct element type" %t
; Test the case of a misformed constant initializer
; This should cause an assembler error, not an assertion failure!
@0 = constant { i32 } { float 1.0 }
