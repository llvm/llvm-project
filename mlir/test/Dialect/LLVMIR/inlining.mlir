// RUN: mlir-opt %s -inline | FileCheck %s

// CHECK-LABEL: func.func @test_inline() -> i32 {
// CHECK-NEXT: %[[RES:.*]] = llvm.mlir.constant(42 : i32) : i32
// CHECK-NEXT: return %[[RES]] : i32
func.func @test_inline() -> i32 {
  %0 = call @inner_func_inlinable() : () -> i32
  return %0 : i32
}

func.func @inner_func_inlinable() -> i32 {
  %0 = llvm.mlir.constant(42 : i32) : i32
  return %0 : i32
}

// CHECK-LABEL: func.func @test_not_inline() -> !llvm.ptr<f64> {
// CHECK-NEXT: %[[RES:.*]] = call @inner_func_not_inlinable() : () -> !llvm.ptr<f64>
// CHECK-NEXT: return %[[RES]] : !llvm.ptr<f64>
func.func @test_not_inline() -> !llvm.ptr<f64> {
  %0 = call @inner_func_not_inlinable() : () -> !llvm.ptr<f64>
  return %0 : !llvm.ptr<f64>
}

func.func @inner_func_not_inlinable() -> !llvm.ptr<f64> {
  %0 = llvm.mlir.constant(0 : i32) : i32
  %1 = llvm.alloca %0 x f64 : (i32) -> !llvm.ptr<f64>
  return %1 : !llvm.ptr<f64>
}
