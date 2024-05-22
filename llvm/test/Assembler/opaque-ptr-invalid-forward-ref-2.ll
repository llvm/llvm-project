; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; CHECK: forward reference and definition of global have different types

@a = alias i32, ptr addrspace(1) @g
@g = global i32 0
