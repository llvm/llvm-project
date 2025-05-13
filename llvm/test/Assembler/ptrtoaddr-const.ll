; RUN: not llvm-as < %s 2>&1 | FileCheck %s
; ptrtoaddr is not currently support in constant expressions

@i = global i32 0
@global_cast = global i32 ptrtoaddr (ptr @i0 to i32)
; CHECK: [[#@LINE-1]]:27: error: expected value token
