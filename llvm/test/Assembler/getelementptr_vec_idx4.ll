; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; CHECK: getelementptr vector index has a wrong number of elements

@0 = global <2 x ptr> getelementptr ([3 x {i32, i32}], <4 x ptr> zeroinitializer, <2 x i32> <i32 1, i32 2>, <2 x i32> <i32 2, i32 3>, <2 x i32> <i32 1, i32 1>)
