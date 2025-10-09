; RUN: not llvm-as < %s 2>&1 | FileCheck %s

@var = global i32 0

; CHECK: error: constant ptrauth base pointer must be a pointer
@auth_var = global ptr ptrauth (i32 42, i32 0)
