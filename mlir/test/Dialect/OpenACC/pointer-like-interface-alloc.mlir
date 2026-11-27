// RUN: mlir-opt %s --split-input-file --pass-pipeline="builtin.module(func.func(test-acc-pointer-like-interface{test-mode=alloc}))" 2>&1 | FileCheck %s

func.func @test_static_memref_alloc() {
  %0 = memref.alloca() {test.ptr} : memref<10x20xf32>
  // CHECK: Successfully generated alloc for operation: %[[ORIG:.*]] = memref.alloca() {test.ptr} : memref<10x20xf32>
  // CHECK: Generated: %{{.*}} = memref.alloca() {acc.var_name = #acc.var_name<"test_alloc">} : memref<10x20xf32>
  return
}

// -----

func.func @test_dynamic_memref_alloc() {
  %c10 = arith.constant 10 : index
  %c20 = arith.constant 20 : index
  %orig = memref.alloc(%c10, %c20) {test.ptr} : memref<?x?xf32>
  
  // CHECK: Successfully generated alloc for operation: %[[ORIG:.*]] = memref.alloc(%[[C10:.*]], %[[C20:.*]]) {test.ptr} : memref<?x?xf32>
  // CHECK: Generated: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: Generated: %[[DIM0:.*]] = memref.dim %[[ORIG]], %[[C0]] : memref<?x?xf32>
  // CHECK: Generated: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: Generated: %[[DIM1:.*]] = memref.dim %[[ORIG]], %[[C1]] : memref<?x?xf32>
  // CHECK: Generated: %{{.*}} = memref.alloc(%[[DIM0]], %[[DIM1]]) {acc.var_name = #acc.var_name<"test_alloc">} : memref<?x?xf32>
  return
}
