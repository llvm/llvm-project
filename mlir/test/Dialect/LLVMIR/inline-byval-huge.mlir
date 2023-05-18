// RUN: mlir-opt -inline --split-input-file %s | FileCheck %s

// CHECK-LABEL: @byval_2_000_000_000
llvm.func @byval_2_000_000_000(%ptr : !llvm.ptr) {
  // CHECK: %[[SIZE:.+]] = llvm.mlir.constant(2000000000 : i64)
  // CHECK: "llvm.intr.memcpy"(%{{.*}}, %{{.*}}, %[[SIZE]]

  llvm.call @with_byval_arg(%ptr) : (!llvm.ptr) -> ()
  llvm.return
}

llvm.func @with_byval_arg(%ptr : !llvm.ptr { llvm.byval = !llvm.array<1000 x array<1000 x array <500 x i32>>> }) {
  llvm.return
}

// -----

// This exceeds representational capacity of 32-bit unsigned value.

// CHECK-LABEL: @byval_8_000_000_000
llvm.func @byval_8_000_000_000(%ptr : !llvm.ptr) {
  // CHECK: %[[SIZE:.+]] = llvm.mlir.constant(8000000000 : i64)
  // CHECK: "llvm.intr.memcpy"(%{{.*}}, %{{.*}}, %[[SIZE]]

  llvm.call @with_byval_arg(%ptr) : (!llvm.ptr) -> ()
  llvm.return
}

llvm.func @with_byval_arg(%ptr : !llvm.ptr { llvm.byval = !llvm.array<2000 x array<2000 x array <500 x i32>>> }) {
  llvm.return
}
