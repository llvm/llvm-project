; RUN: not llvm-as < %s 2>&1 | FileCheck %s

@var = global i32 0

; CHECK: error: constant ptrauth key must be i32 constant
@auth_var = global ptr ptrauth (ptr @var, i32 ptrtoint (ptr @var to i32))
