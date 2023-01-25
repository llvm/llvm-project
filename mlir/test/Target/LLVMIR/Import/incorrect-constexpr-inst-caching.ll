; RUN: mlir-translate -opaque-pointers=0 --import-llvm %s | FileCheck %s
; REQUIRES: asserts

; This test is primarily used to make sure an assertion is not triggered.
; Thus, we only wrote minimum level of checks.

%my_struct = type {i32, i8*}
; CHECK: llvm.mlir.constant(8 : i32) : i32
; CHECK: llvm.mlir.addressof @str0 : !llvm.ptr<array<5 x i8>>
; CHECK: llvm.mlir.constant(0 : i32) : i32
; CHECK: llvm.mlir.constant(1 : i32) : i32
; CHECK: llvm.getelementptr
; CHECK: llvm.mlir.undef : !llvm.struct<"my_struct", (i32, ptr<i8>)>
; CHECK: llvm.insertvalue
; CHECK: llvm.insertvalue
; CHECK: llvm.mlir.constant(7 : i32) : i32
; CHECK: llvm.mlir.addressof @str1 : !llvm.ptr<array<5 x i8>>
; CHECK: llvm.mlir.constant(2 : i32) : i32
; CHECK: llvm.mlir.constant(3 : i32) : i32
; CHECK: llvm.getelementptr
; CHECK: llvm.mlir.undef : !llvm.struct<"my_struct", (i32, ptr<i8>)>
; CHECK: llvm.insertvalue
; CHECK: llvm.insertvalue
; CHECK: llvm.mlir.undef : !llvm.array<2 x struct<"my_struct", (i32, ptr<i8>)>>
; CHECK: llvm.insertvalue
; CHECK: llvm.insertvalue
; CHECK: llvm.return
@str0 = private unnamed_addr constant [5 x i8] c"aaaa\00"
@str1 = private unnamed_addr constant [5 x i8] c"bbbb\00"
@g = global [2 x %my_struct] [%my_struct {i32 8, i8* getelementptr ([5 x i8], [5 x i8]* @str0, i32 0, i32 1)}, %my_struct {i32 7, i8* getelementptr ([5 x i8], [5 x i8]* @str1, i32 2, i32 3)}]
