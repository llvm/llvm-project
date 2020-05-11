; RUN: not --crash llc < %s -mtriple powerpc-ibm-aix-xcoff 2>&1 | FileCheck %s
; RUN: not --crash llc < %s -mtriple powerpc64-ibm-aix-xcoff 2>&1 | FileCheck %s

; Check that, while generation of aliases on AIX remains unimplemented, llc dies
; with an appropriate message instead of generating incorrect output when an
; alias is encountered.

define i32 @a() { ret i32 0 }
; CHECK: ERROR: module has aliases
@b = internal alias i32 (), i32 ()* @a
