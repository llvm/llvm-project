; RUN: not llvm-as < %s 2>&1 | FileCheck %s
; CHECK: <stdin>:4:12: error: explicit pointee type doesn't match operand's pointee type (i1 vs i2)
@y = global i2 0
@x = alias i1, i2* @y
