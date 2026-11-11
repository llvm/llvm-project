; Test that a sub-8-byte address constant (ADCON) in a C_WSA64 section produces
; an error. The z/OS binder requires all ADCONs in C_WSA64 to be exactly 8
; bytes and rejects 4-byte ADCONs with IEW2572E.
;
; RUN: not llc < %s -mtriple=s390x-ibm-zos --filetype=obj 2>&1 | FileCheck %s
; RUN: not llc < %s -mtriple=s390x-ibm-zos 2>&1 | FileCheck %s

; CHECK: error: address constant in C_WSA64 must be 8 bytes (z/OS binder restriction IEW2572E)

@foo = global i32 42
@arr = global [2 x i32] [
  i32 trunc (i64 ptrtoint (ptr @foo to i64) to i32),
  i32 99
]
