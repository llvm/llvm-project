; RUN: not llvm-as < %s 2>&1 | FileCheck %s

@var = global i32 0

; CHECK: error: constant ptrauth integer discriminator must be i64 constant
@auth_var = global ptr ptrauth (ptr @var, i32 2, ptr @var))
