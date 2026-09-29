; RUN: llvm-as < %s | llvm-dis | FileCheck %s

@v1 = global i32 0
; CHECK: @v1 = global i32 0

@v2 = global [1 x i32] zeroinitializer
; CHECK: @v2 = global [1 x i32] zeroinitializer

@v3 = global [2 x i16] zeroinitializer
; CHECK: @v3 = global [2 x i16] zeroinitializer

@a1 = alias i16, ptr @v1
; CHECK: @a1 = alias i16, ptr @v1

@a2 = alias i32, ptr @v2
; CHECK: @a2 = alias i32, ptr @v2

@a3 = alias i32, addrspacecast (ptr @v1 to ptr addrspace(2))
; CHECK: @a3 = alias i32, addrspacecast (ptr @v1 to ptr addrspace(2))

@a4 = alias i16, ptr @v1
; CHECK: @a4 = alias i16, ptr @v1

@a5 = thread_local(localdynamic) alias i32, ptr @v1
; CHECK: @a5 = thread_local(localdynamic) alias i32, ptr @v1

@a6 = alias i16, getelementptr ([2 x i16], ptr @v3, i32 1, i32 1)
; CHECK: @a6 = alias i16, getelementptr ([2 x i16], ptr @v3, i32 1, i32 1)
