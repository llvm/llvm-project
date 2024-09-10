; RUN: not llvm-as < %s 2>&1 | FileCheck %s

@var = global i32 0

; CHECK: error: constant ptrauth address discriminator must be a pointer
@auth_var = global ptr ptrauth (ptr @var, i32 2, i64 65535, i8 0)
