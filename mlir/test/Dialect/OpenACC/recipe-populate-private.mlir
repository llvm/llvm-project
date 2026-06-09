// RUN: mlir-opt %s --split-input-file --pass-pipeline="builtin.module(test-acc-recipe-populate{recipe-type=private})" | FileCheck %s

// CHECK: acc.private.recipe @private_scalar : memref<f32> init {
// CHECK: ^bb0(%{{.*}}: memref<f32>):
// CHECK:   %[[ALLOC:.*]] = memref.alloca() {acc.var_name = #acc.var_name<"scalar">} : memref<f32>
// CHECK:   acc.yield %[[ALLOC]] : memref<f32>
// CHECK: }
// CHECK-NOT: destroy

func.func @test_scalar() {
  %0 = memref.alloca() {test.var = "scalar"} : memref<f32>
  return
}

// -----

// CHECK: acc.private.recipe @private_static_2d : memref<10x20xf32> init {
// CHECK: ^bb0(%{{.*}}: memref<10x20xf32>):
// CHECK:   %[[ALLOC:.*]] = memref.alloca() {acc.var_name = #acc.var_name<"static_2d">} : memref<10x20xf32>
// CHECK:   acc.yield %[[ALLOC]] : memref<10x20xf32>
// CHECK: }
// CHECK-NOT: destroy

func.func @test_static_2d() {
  %0 = memref.alloca() {test.var = "static_2d"} : memref<10x20xf32>
  return
}

// -----

// CHECK: acc.private.recipe @private_dynamic_2d : memref<?x?xf32> init {
// CHECK: ^bb0(%[[ARG:.*]]: memref<?x?xf32>):
// CHECK:   %[[C0:.*]] = arith.constant 0 : index
// CHECK:   %[[DIM0:.*]] = memref.dim %[[ARG]], %[[C0]] : memref<?x?xf32>
// CHECK:   %[[C1:.*]] = arith.constant 1 : index
// CHECK:   %[[DIM1:.*]] = memref.dim %[[ARG]], %[[C1]] : memref<?x?xf32>
// CHECK:   %[[ALLOC:.*]] = memref.alloc(%[[DIM0]], %[[DIM1]]) {acc.var_name = #acc.var_name<"dynamic_2d">} : memref<?x?xf32>
// CHECK:   acc.yield %[[ALLOC]] : memref<?x?xf32>
// CHECK: } destroy {
// CHECK: ^bb0(%{{.*}}: memref<?x?xf32>, %[[VAL:.*]]: memref<?x?xf32>):
// CHECK:   memref.dealloc %[[VAL]] : memref<?x?xf32>
// CHECK:   acc.terminator
// CHECK: }

func.func @test_dynamic_2d(%arg0: index, %arg1: index) {
  %0 = memref.alloc(%arg0, %arg1) {test.var = "dynamic_2d"} : memref<?x?xf32>
  return
}

// -----

// CHECK: acc.private.recipe @private_mixed_dims : memref<10x?xf32> init {
// CHECK: ^bb0(%[[ARG:.*]]: memref<10x?xf32>):
// CHECK:   %[[C1:.*]] = arith.constant 1 : index
// CHECK:   %[[DIM1:.*]] = memref.dim %[[ARG]], %[[C1]] : memref<10x?xf32>
// CHECK:   %[[ALLOC:.*]] = memref.alloc(%[[DIM1]]) {acc.var_name = #acc.var_name<"mixed_dims">} : memref<10x?xf32>
// CHECK:   acc.yield %[[ALLOC]] : memref<10x?xf32>
// CHECK: } destroy {
// CHECK: ^bb0(%{{.*}}: memref<10x?xf32>, %[[VAL:.*]]: memref<10x?xf32>):
// CHECK:   memref.dealloc %[[VAL]] : memref<10x?xf32>
// CHECK:   acc.terminator
// CHECK: }

func.func @test_mixed_dims(%arg0: index) {
  %0 = memref.alloc(%arg0) {test.var = "mixed_dims"} : memref<10x?xf32>
  return
}

// -----

// CHECK: acc.private.recipe @private_scalar_int : memref<i32> init {
// CHECK: ^bb0(%{{.*}}: memref<i32>):
// CHECK:   %[[ALLOC:.*]] = memref.alloca() {acc.var_name = #acc.var_name<"scalar_int">} : memref<i32>
// CHECK:   acc.yield %[[ALLOC]] : memref<i32>
// CHECK: }
// CHECK-NOT: destroy

func.func @test_scalar_int() {
  %0 = memref.alloca() {test.var = "scalar_int"} : memref<i32>
  return
}

