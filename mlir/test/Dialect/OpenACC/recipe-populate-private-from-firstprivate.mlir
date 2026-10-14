// RUN: mlir-opt %s --split-input-file --pass-pipeline="builtin.module(test-acc-recipe-populate{recipe-type=private_from_firstprivate})" | FileCheck %s

// Verify that we can create a private recipe using the convenience overload
// that takes an existing firstprivate recipe as input. For a simple scalar
// alloca-backed memref, only an init region is expected (no destroy).
// CHECK: acc.private.recipe @private_from_firstprivate_scalar : memref<f32> init {
// CHECK: ^bb0(%{{.*}}: memref<f32>):
// CHECK:   %[[ALLOC:.*]] = memref.alloca() {acc.var_name = #acc.var_name<"scalar">} : memref<f32>
// CHECK:   acc.yield %[[ALLOC]] : memref<f32>
// CHECK: }

func.func @test_scalar_from_firstprivate() {
  %0 = memref.alloca() {test.var = "scalar"} : memref<f32>
  return
}

// -----

// Verify that destroy regions are also present when creating a private recipe
// from a firstprivate recipe that requires dynamic deallocation.
// CHECK: acc.private.recipe @private_from_firstprivate_dynamic_d2 : memref<?x?xf32> init {
// CHECK: ^bb0(%[[ARG:.*]]: memref<?x?xf32>):
// CHECK:   %[[C0:.*]] = arith.constant 0 : index
// CHECK:   %[[DIM0:.*]] = memref.dim %[[ARG]], %[[C0]] : memref<?x?xf32>
// CHECK:   %[[C1:.*]] = arith.constant 1 : index
// CHECK:   %[[DIM1:.*]] = memref.dim %[[ARG]], %[[C1]] : memref<?x?xf32>
// CHECK:   %[[ALLOC:.*]] = memref.alloc(%[[DIM0]], %[[DIM1]]) {acc.var_name = #acc.var_name<"dynamic_d2">} : memref<?x?xf32>
// CHECK:   acc.yield %[[ALLOC]] : memref<?x?xf32>
// CHECK: } destroy {
// CHECK: ^bb0(%{{.*}}: memref<?x?xf32>, %[[VAL:.*]]: memref<?x?xf32>):
// CHECK:   memref.dealloc %[[VAL]] : memref<?x?xf32>
// CHECK:   acc.terminator
// CHECK: }

func.func @test_dynamic_from_firstprivate(%arg0: index, %arg1: index) {
  %0 = memref.alloc(%arg0, %arg1) {test.var = "dynamic_d2"} : memref<?x?xf32>
  return
}
