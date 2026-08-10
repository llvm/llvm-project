; RUN: not llvm-link %s %p/unnamed-addr-err-b.ll -S -o - 2>&1 | FileCheck %s

@foo = appending unnamed_addr global [1 x i32] [i32 42]
; CHECK: Appending variables with different unnamed_addr need to be linked
