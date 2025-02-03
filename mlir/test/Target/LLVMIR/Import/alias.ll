; RUN: mlir-translate --import-llvm %s | FileCheck %s

@foo_alias = alias ptr, ptr @callee

; CHECK: llvm.mlir.alias external @foo_alias {addr_space = 0 : i32} : !llvm.ptr {
; CHECK:   %0 = llvm.mlir.addressof @callee : !llvm.ptr
; CHECK:   llvm.return %0 : !llvm.ptr
; CHECK: }

define internal ptr @callee() {
entry:
  ret ptr null
}

@zed = global i32 42
@foo = alias i32, ptr @zed
@foo2 = alias i16, ptr @zed

; CHECK: llvm.mlir.alias external @foo {addr_space = 0 : i32} : i32 {
; CHECK:   %0 = llvm.mlir.addressof @zed : !llvm.ptr
; CHECK:   llvm.return %0 : !llvm.ptr
; CHECK: }
; CHECK: llvm.mlir.alias external @foo2 {addr_space = 0 : i32} : i16 {
; CHECK:   %0 = llvm.mlir.addressof @zed : !llvm.ptr
; CHECK:   llvm.return %0 : !llvm.ptr
; CHECK: }

@v1 = global i32 0
@a3 = alias i32, addrspacecast (ptr @v1 to ptr addrspace(2))
; CHECK: llvm.mlir.alias external @a3 {addr_space = 2 : i32} : i32 {
; CHECK:   %0 = llvm.mlir.addressof @v1 : !llvm.ptr
; CHECK:   %1 = llvm.addrspacecast %0 : !llvm.ptr to !llvm.ptr<2>
; CHECK:   llvm.return %1 : !llvm.ptr<2>
; CHECK: }

@some_name = constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr null, ptr null] }
@_ZTV1D = alias { [3 x ptr] }, ptr @some_name

; CHECK: llvm.mlir.alias external @_ZTV1D {addr_space = 0 : i32} : !llvm.struct<(array<3 x ptr>)> {
; CHECK:   %0 = llvm.mlir.addressof @some_name : !llvm.ptr
; CHECK:   llvm.return %0 : !llvm.ptr
; CHECK: }

@glob.private = private constant [32 x i32] zeroinitializer
@glob = linkonce_odr hidden alias [32 x i32], inttoptr (i64 add (i64 ptrtoint (ptr @glob.private to i64), i64 1234) to ptr)

; CHECK: llvm.mlir.alias linkonce_odr hidden @glob {addr_space = 0 : i32, dso_local} : !llvm.array<32 x i32> {
; CHECK: %0 = llvm.mlir.constant(1234 : i64) : i64
; CHECK: %1 = llvm.mlir.addressof @glob.private : !llvm.ptr
; CHECK: %2 = llvm.ptrtoint %1 : !llvm.ptr to i64
; CHECK: %3 = llvm.add %2, %0 : i64
; CHECK: %4 = llvm.inttoptr %3 : i64 to !llvm.ptr
; CHECK: llvm.return %4 : !llvm.ptr
