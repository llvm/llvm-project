// RUN: mlir-opt %s -inline -split-input-file | FileCheck %s

func.func @inner_func_inlinable(%ptr : !llvm.ptr) -> i32 {
  %0 = llvm.mlir.constant(42 : i32) : i32
  llvm.store %0, %ptr { alignment = 8 } : i32, !llvm.ptr
  %1 = llvm.load %ptr { alignment = 8 } : !llvm.ptr -> i32
  return %1 : i32
}

// CHECK-LABEL: func.func @test_inline(
// CHECK-SAME: %[[PTR:[a-zA-Z0-9_]+]]
// CHECK-NEXT: %[[CST:.*]] = llvm.mlir.constant(42 : i32) : i32
// CHECK-NEXT: llvm.store %[[CST]], %[[PTR]]
// CHECK-NEXT: %[[RES:.+]] = llvm.load %[[PTR]]
// CHECK-NEXT: return %[[RES]] : i32
func.func @test_inline(%ptr : !llvm.ptr) -> i32 {
  %0 = call @inner_func_inlinable(%ptr) : (!llvm.ptr) -> i32
  return %0 : i32
}

// -----

func.func @inner_func_not_inlinable() -> !llvm.ptr<f64> {
  %0 = llvm.mlir.constant(0 : i32) : i32
  %1 = llvm.alloca %0 x f64 : (i32) -> !llvm.ptr<f64>
  return %1 : !llvm.ptr<f64>
}

// CHECK-LABEL: func.func @test_not_inline() -> !llvm.ptr<f64> {
// CHECK-NEXT: %[[RES:.*]] = call @inner_func_not_inlinable() : () -> !llvm.ptr<f64>
// CHECK-NEXT: return %[[RES]] : !llvm.ptr<f64>
func.func @test_not_inline() -> !llvm.ptr<f64> {
  %0 = call @inner_func_not_inlinable() : () -> !llvm.ptr<f64>
  return %0 : !llvm.ptr<f64>
}

// -----

llvm.metadata @metadata {
  llvm.access_group @group
  llvm.return
}

func.func private @with_mem_attr(%ptr : !llvm.ptr) -> () {
  %0 = llvm.mlir.constant(42 : i32) : i32
  // Do not inline load/store operations that carry attributes requiring
  // handling while inlining, until this is supported by the inliner.
  llvm.store %0, %ptr { access_groups = [@metadata::@group] }: i32, !llvm.ptr
  return
}

// CHECK-LABEL: func.func @test_not_inline
// CHECK-NEXT: call @with_mem_attr
// CHECK-NEXT: return
func.func @test_not_inline(%ptr : !llvm.ptr) -> () {
  call @with_mem_attr(%ptr) : (!llvm.ptr) -> ()
  return
}

// -----
// Check that llvm.return is correctly handled

func.func @func(%arg0 : i32) -> i32  {
  llvm.return %arg0 : i32
}
// CHECK-LABEL: @llvm_ret
// CHECK-NOT: call
// CHECK:  return %arg0
func.func @llvm_ret(%arg0 : i32) -> i32 {
  %res = call @func(%arg0) : (i32) -> (i32)
  return %res : i32
}
