; RUN: llvm-as --opaque-pointers=0 < %s | llvm-dis --opaque-pointers=0 | FileCheck %s

@v1 = global i32 0
; CHECK: @v1 = global i32 0

@v2 = global [1 x i32] zeroinitializer
; CHECK: @v2 = global [1 x i32] zeroinitializer

@v3 = global [2 x i16] zeroinitializer
; CHECK: @v3 = global [2 x i16] zeroinitializer

@a1 = alias i16, bitcast (i32* @v1 to i16*)
; CHECK: @a1 = alias i16, bitcast (i32* @v1 to i16*)

@a2 = alias i32, bitcast([1 x i32]* @v2 to i32*)
; CHECK: @a2 = alias i32, getelementptr inbounds ([1 x i32], [1 x i32]* @v2, i32 0, i32 0)

@a3 = alias i32, addrspacecast (i32* @v1 to i32 addrspace(2)*)
; CHECK: @a3 = alias i32, addrspacecast (i32* @v1 to i32 addrspace(2)*)

@a4 = alias i16, bitcast (i32* @v1 to i16*)
; CHECK: @a4 = alias i16, bitcast (i32* @v1 to i16*)

@a5 = thread_local(localdynamic) alias i32, i32* @v1
; CHECK: @a5 = thread_local(localdynamic) alias i32, i32* @v1

@a6 = alias i16, getelementptr ([2 x i16], [2 x i16]* @v3, i32 1, i32 1)
; CHECK: @a6 = alias i16, getelementptr ([2 x i16], [2 x i16]* @v3, i32 1, i32 1)
