// RUN: mlir-opt %s --pass-pipeline="builtin.module(llvm.func(canonicalize))" --split-input-file | FileCheck %s

// CHECK-LABEL: llvm.func @zext_basic
llvm.func @zext_basic() -> i64 {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.zext %0 : i32 to i64
  // CHECK: %[[RES:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK: llvm.return %[[RES]] : i64
  llvm.return %1 : i64
}

// CHECK-LABEL: llvm.func @zext_neg
llvm.func @zext_neg() -> i64 {
  %0 = llvm.mlir.constant(-1 : i32) : i32
  %1 = llvm.zext %0 : i32 to i64
  // CHECK: %[[RES:.*]] = llvm.mlir.constant(4294967295 : i64) : i64
  // CHECK: llvm.return %[[RES]] : i64
  llvm.return %1 : i64
}

// -----

// CHECK-LABEL: llvm.func @shl_basic
llvm.func @shl_basic() -> i32 {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.mlir.constant(1 : i32) : i32
  %2 = llvm.shl %0, %1 : i32
  // CHECK: %[[RES:.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK: llvm.return %[[RES]] : i32
  llvm.return %2 : i32
}

// CHECK-LABEL: llvm.func @shl_multiple
llvm.func @shl_multiple() -> i32 {
  %0 = llvm.mlir.constant(45 : i32) : i32
  %1 = llvm.mlir.constant(7 : i32) : i32
  %2 = llvm.shl %0, %1 : i32
  // CHECK: %[[RES:.*]] = llvm.mlir.constant(5760 : i32) : i32
  // CHECK: llvm.return %[[RES]] : i32
  llvm.return %2 : i32
}

// -----

// CHECK-LABEL: llvm.func @or_basic
llvm.func @or_basic() -> i32 {
  %0 = llvm.mlir.constant(5 : i32) : i32
  %1 = llvm.mlir.constant(9 : i32) : i32
  %2 = llvm.or %0, %1 : i32
  // CHECK: %[[RES:.*]] = llvm.mlir.constant(13 : i32) : i32
  // CHECK: llvm.return %[[RES]] : i32
  llvm.return %2 : i32
}
