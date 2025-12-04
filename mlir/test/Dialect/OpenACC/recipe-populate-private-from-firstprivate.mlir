// RUN: mlir-opt %s --split-input-file --pass-pipeline="builtin.module(test-acc-recipe-populate{recipe-type=private_from_firstprivate})" | FileCheck %s

// Verify that we can create a private recipe using the convenience overload
// that takes an existing firstprivate recipe as input.
// CHECK: acc.firstprivate.recipe @first_firstprivate_scalar : memref<f32> init
// CHECK: acc.private.recipe @private_from_firstprivate_scalar : memref<f32> init {
// CHECK: ^bb0(%{{.*}}: memref<f32>):
// CHECK:   %[[ALLOC:.*]] = memref.alloca() {acc.var_name = #acc.var_name<"scalar">} : memref<f32>
// CHECK:   acc.yield %[[ALLOC]] : memref<f32>
// CHECK: }

func.func @test_scalar() {
  %0 = memref.alloca() {test.var = "scalar"} : memref<f32>
  return
}


