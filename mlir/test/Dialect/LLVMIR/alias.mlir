// RUN: mlir-opt %s -split-input-file --verify-roundtrip | FileCheck %s

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

// CHECK: llvm.mlir.alias external @foo_alias : !llvm.ptr {
// CHECK:   %[[ADDR:.*]] = llvm.mlir.addressof @callee : !llvm.ptr
// CHECK:   llvm.return %[[ADDR]] : !llvm.ptr
// CHECK: }
// CHECK: llvm.mlir.alias external @_ZTV1D : !llvm.struct<(array<3 x ptr>)> {
// CHECK:   %[[ADDR:.*]] = llvm.mlir.addressof @callee : !llvm.ptr
// CHECK:   llvm.return %[[ADDR]] : !llvm.ptr
// CHECK: }

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

// CHECK: llvm.mlir.alias external @foo : i32 {
// CHECK:   %[[ADDR:.*]] = llvm.mlir.addressof @zed : !llvm.ptr
// CHECK:   llvm.return %[[ADDR]] : !llvm.ptr
// CHECK: }
// CHECK: llvm.mlir.alias external @foo2 : i16 {
// CHECK:   %[[ADDR:.*]] = llvm.mlir.addressof @zed : !llvm.ptr
// CHECK:   llvm.return %[[ADDR]] : !llvm.ptr
// CHECK: }

// -----

llvm.mlir.global private constant @glob.private(dense<0> : tensor<32xi32>) : !llvm.array<32 x i32>

llvm.mlir.alias linkonce_odr hidden @glob {dso_local} : !llvm.array<32 x i32> {
  %0 = llvm.mlir.constant(1234 : i64) : i64
  %1 = llvm.mlir.addressof @glob.private : !llvm.ptr
  %2 = llvm.ptrtoint %1 : !llvm.ptr to i64
  %3 = llvm.add %2, %0 : i64
  %4 = llvm.inttoptr %3 : i64 to !llvm.ptr
  llvm.return %4 : !llvm.ptr
}

// CHECK: llvm.mlir.global private constant @glob.private(dense<0> : tensor<32xi32>)
// CHECK: llvm.mlir.alias linkonce_odr hidden @glob {dso_local} : !llvm.array<32 x i32> {
// CHECK:   %[[CST:.*]] = llvm.mlir.constant(1234 : i64) : i64
// CHECK:   %[[ADDR:.*]] = llvm.mlir.addressof @glob.private : !llvm.ptr
// CHECK:   %[[INTADDR:.*]] = llvm.ptrtoint %[[ADDR]] : !llvm.ptr to i64
// CHECK:   %[[BACKTOPTR:.*]] = llvm.add %[[INTADDR]], %[[CST]] : i64
// CHECK:   %[[RET_ADDR:.*]]  = llvm.inttoptr %[[BACKTOPTR]] : i64 to !llvm.ptr
// CHECK:   llvm.return %[[RET_ADDR]] : !llvm.ptr
// CHECK: }

// -----

llvm.mlir.global external @v1(0 : i32) : i32
llvm.mlir.alias external @a3 : i32 {
  %0 = llvm.mlir.addressof @v1 : !llvm.ptr
  %1 = llvm.addrspacecast %0 : !llvm.ptr to !llvm.ptr<2>
  llvm.return %1 : !llvm.ptr<2>
}

// CHECK: llvm.mlir.alias external @a3 : i32 {
// CHECK:   %[[ADDR:.*]] = llvm.mlir.addressof @v1 : !llvm.ptr
// CHECK:   %1 = llvm.addrspacecast %[[ADDR]] : !llvm.ptr to !llvm.ptr<2>
// CHECK:   llvm.return %1 : !llvm.ptr<2>
// CHECK: }

// -----

llvm.mlir.global private @g1(0 : i32) {dso_local} : i32

llvm.mlir.alias private @a1 {dso_local} : i32 {
  %0 = llvm.mlir.addressof @g1 : !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

llvm.mlir.global internal constant @g2() : !llvm.ptr {
  %0 = llvm.mlir.addressof @a1 : !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

llvm.mlir.alias private @a2 {dso_local} : !llvm.ptr {
  %0 = llvm.mlir.addressof @a1 : !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

llvm.mlir.global internal constant @g3() : !llvm.ptr {
  %0 = llvm.mlir.addressof @a2 : !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

// CHECK: llvm.mlir.alias private @a1 {dso_local} : i32 {
// CHECK:   %[[ADDR:.*]] = llvm.mlir.addressof @g1 : !llvm.ptr
// CHECK:   llvm.return %[[ADDR]] : !llvm.ptr
// CHECK: }
// CHECK: llvm.mlir.global internal constant @g2()
// CHECK:   %[[ADDR:.*]] = llvm.mlir.addressof @a1 : !llvm.ptr
// CHECK:   llvm.return %[[ADDR]] : !llvm.ptr
// CHECK: }
// CHECK: llvm.mlir.alias private @a2 {dso_local} : !llvm.ptr {
// CHECK:   %[[ADDR:.*]] = llvm.mlir.addressof @a1 : !llvm.ptr
// CHECK:   llvm.return %[[ADDR]] : !llvm.ptr
// CHECK: }
// CHECK: llvm.mlir.global internal constant @g3()
// CHECK:   %[[ADDR:.*]] = llvm.mlir.addressof @a2 : !llvm.ptr
// CHECK:   llvm.return %[[ADDR]] : !llvm.ptr
// CHECK: }

// -----

llvm.mlir.global private @g30(0 : i32) {dso_local} : i32

llvm.mlir.alias private unnamed_addr thread_local @a30 {dso_local} : i32 {
  %0 = llvm.mlir.addressof @g30 : !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

// CHECK: llvm.mlir.alias private unnamed_addr thread_local @a30 {dso_local} : i32 {
// CHECK:   %0 = llvm.mlir.addressof @g30 : !llvm.ptr
// CHECK:   llvm.return %0 : !llvm.ptr
// CHECK: }
