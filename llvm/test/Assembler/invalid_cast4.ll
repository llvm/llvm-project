; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; CHECK: invalid cast opcode for cast from 'i64' to 'i64'
@0 = global i64* inttoptr (i64 0 to i64)
