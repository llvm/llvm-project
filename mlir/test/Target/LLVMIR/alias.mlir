// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

llvm.mlir.alias external @foo_alias {addr_space = 0 : i32} : !llvm.ptr {
  %0 = llvm.mlir.addressof @callee : !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

// CHECK: @foo_alias = alias ptr, ptr @callee

llvm.mlir.alias external @foo {addr_space = 0 : i32} : i32 {
  %0 = llvm.mlir.addressof @zed : !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

// CHECK: @foo = alias i32, ptr @zed

llvm.mlir.alias external @foo2 {addr_space = 0 : i32} : i16 {
  %0 = llvm.mlir.addressof @zed : !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

// CHECK: @foo2 = alias i16, ptr @zed

llvm.mlir.global external @zed(42 : i32) {addr_space = 0 : i32} : i32

llvm.func internal @callee() -> !llvm.ptr attributes {dso_local} {
  %0 = llvm.mlir.zero : !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

llvm.mlir.alias external @_ZTV1D {addr_space = 0 : i32} : !llvm.struct<(array<3 x ptr>)> {
  %0 = llvm.mlir.addressof @callee : !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

// CHECK: @_ZTV1D = alias { [3 x ptr] }, ptr @callee

llvm.mlir.global private constant @glob.private(dense<0> : tensor<32xi32>) {addr_space = 0 : i32, dso_local} : !llvm.array<32 x i32>

llvm.mlir.alias linkonce_odr hidden @glob {addr_space = 0 : i32, dso_local} : !llvm.array<32 x i32> {
  %0 = llvm.mlir.constant(1234 : i64) : i64
  %1 = llvm.mlir.addressof @glob.private : !llvm.ptr
  %2 = llvm.ptrtoint %1 : !llvm.ptr to i64
  %3 = llvm.add %2, %0 : i64
  %4 = llvm.inttoptr %3 : i64 to !llvm.ptr
  llvm.return %4 : !llvm.ptr
}

// CHECK: @glob = linkonce_odr hidden alias [32 x i32], inttoptr (i64 add (i64 ptrtoint (ptr @glob.private to i64), i64 1234) to ptr)