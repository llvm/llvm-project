// RUN: mlir-translate -mlir-to-llvmir %s -split-input-file | FileCheck %s

llvm.func internal @callee() -> !llvm.ptr attributes {dso_local} {
  %0 = llvm.mlir.zero : !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

llvm.mlir.alias external @foo_alias : !llvm.ptr {
  %0 = llvm.mlir.addressof @callee : !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

llvm.mlir.alias external @_ZTV1D : !llvm.struct<(array<3 x ptr>)> {
  %0 = llvm.mlir.addressof @callee : !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

// CHECK: @foo_alias = alias ptr, ptr @callee
// CHECK: @_ZTV1D = alias { [3 x ptr] }, ptr @callee

// -----

llvm.mlir.global external @zed(42 : i32) : i32

llvm.mlir.alias external @foo : i32 {
  %0 = llvm.mlir.addressof @zed : !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

llvm.mlir.alias external @foo2 : i16 {
  %0 = llvm.mlir.addressof @zed : !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

// CHECK: @foo = alias i32, ptr @zed
// CHECK: @foo2 = alias i16, ptr @zed

// -----

llvm.mlir.global private constant @glob.private(dense<0> : tensor<32xi32>) {dso_local} : !llvm.array<32 x i32>

llvm.mlir.alias linkonce_odr hidden @glob {dso_local} : !llvm.array<32 x i32> {
  %0 = llvm.mlir.constant(1234 : i64) : i64
  %1 = llvm.mlir.addressof @glob.private : !llvm.ptr
  %2 = llvm.ptrtoint %1 : !llvm.ptr to i64
  %3 = llvm.add %2, %0 : i64
  %4 = llvm.inttoptr %3 : i64 to !llvm.ptr
  llvm.return %4 : !llvm.ptr
}

// CHECK: @glob = linkonce_odr hidden alias [32 x i32], inttoptr (i64 add (i64 ptrtoint (ptr @glob.private to i64), i64 1234) to ptr)

// -----

llvm.mlir.global external @v1(0 : i32) : i32
llvm.mlir.alias external @a3 : i32 {
  %0 = llvm.mlir.addressof @v1 : !llvm.ptr
  %1 = llvm.addrspacecast %0 : !llvm.ptr to !llvm.ptr<2>
  llvm.return %1 : !llvm.ptr<2>
}

// CHECK: @a3 = alias i32, addrspacecast (ptr @v1 to ptr addrspace(2))

// -----

llvm.mlir.global private @g1(0 : i32) {dso_local} : i32

llvm.mlir.alias private @a1 {dso_local} : i32 {
  %0 = llvm.mlir.addressof @g1 : !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

llvm.mlir.global internal constant @g2() {dso_local} : !llvm.ptr {
  %0 = llvm.mlir.addressof @a1 : !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

llvm.mlir.alias private @a2 {dso_local} : !llvm.ptr {
  %0 = llvm.mlir.addressof @a1 : !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

llvm.mlir.global internal constant @g3() {dso_local} : !llvm.ptr {
  %0 = llvm.mlir.addressof @a2 : !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

// CHECK: @g1 = private global i32 0
// CHECK: @g2 = internal constant ptr @a1
// CHECK: @g3 = internal constant ptr @a2
// CHECK: @a1 = private alias i32, ptr @g1
// CHECK: @a2 = private alias ptr, ptr @a1
