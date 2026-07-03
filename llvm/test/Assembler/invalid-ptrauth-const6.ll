; RUN: not llvm-as < %s 2>&1 | FileCheck %s

@var = global i32 0

; CHECK: error: constant ptrauth deactivation symbol must be a pointer
@ptr = global ptr ptrauth (ptr @var, i32 0, i64 65535, ptr null, i64 0)
