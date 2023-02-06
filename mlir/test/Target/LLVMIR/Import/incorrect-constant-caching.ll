; RUN: mlir-translate --import-llvm %s | FileCheck %s

; Testing the fix for issue where llvm.getelementptr translated from the second
; ConstantExpr-GEP tried to use llvm.constant(0: i32)-s that were below itself,
; because it's reusing the cached ConstantInt-type zero value translated earlier.

; This test is primarily used to make sure an verification is passed. Thus, we
; only wrote minimum level of checks.

%my_struct = type {i32, ptr}
; CHECK-DAG: llvm.mlir.constant(8 : i32) : i32
; CHECK-DAG: llvm.mlir.addressof @str0 : !llvm.ptr
; CHECK-DAG: llvm.mlir.constant(0 : i32) : i32
; CHECK-DAG: llvm.getelementptr
; CHECK-DAG: llvm.mlir.undef : !llvm.struct<"my_struct", (i32, ptr)>
; CHECK-DAG: llvm.insertvalue
; CHECK-DAG: llvm.insertvalue
; CHECK-DAG: llvm.mlir.constant(7 : i32) : i32
; CHECK-DAG: llvm.mlir.addressof @str1 : !llvm.ptr
; CHECK-DAG: llvm.getelementptr
; CHECK-DAG: llvm.mlir.undef : !llvm.struct<"my_struct", (i32, ptr)>
; CHECK-DAG: llvm.insertvalue
; CHECK-DAG: llvm.insertvalue
; CHECK-DAG: llvm.mlir.undef : !llvm.array<2 x struct<"my_struct", (i32, ptr)>>
; CHECK-DAG: llvm.insertvalue
; CHECK-DAG: llvm.insertvalue
; CHECK-DAG: llvm.return
@str0 = private unnamed_addr constant [5 x i8] c"aaaa\00"
@str1 = private unnamed_addr constant [5 x i8] c"bbbb\00"
@g = global [2 x %my_struct] [%my_struct {i32 8, ptr getelementptr ([5 x i8], ptr @str0, i32 0, i32 1)}, %my_struct {i32 7, ptr getelementptr ([5 x i8], ptr @str1, i32 0, i32 1)}]
