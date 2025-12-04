// RUN: mlir-opt %s --split-input-file --pass-pipeline="builtin.module(test-acc-recipe-populate{recipe-type=private})" | FileCheck %s --check-prefix=CHECK-PRIVATE
// RUN: mlir-opt %s --split-input-file --pass-pipeline="builtin.module(test-acc-recipe-populate{recipe-type=private_from_firstprivate})" | FileCheck %s --check-prefix=CHECK-PRIV-FROM-FIRST

// CHECK-PRIVATE: acc.private.recipe @private_scalar : memref<f32> init {
// CHECK-PRIVATE: ^bb0(%{{.*}}: memref<f32>):
// CHECK-PRIVATE:   %[[ALLOC:.*]] = memref.alloca() {acc.var_name = #acc.var_name<"scalar">} : memref<f32>
// CHECK-PRIVATE:   acc.yield %[[ALLOC]] : memref<f32>
// CHECK-PRIVATE: }
// CHECK-PRIVATE-NOT: destroy

func.func @test_scalar() {
  %0 = memref.alloca() {test.var = "scalar"} : memref<f32>
  return
}

// -----

// CHECK-PRIVATE: acc.private.recipe @private_static_2d : memref<10x20xf32> init {
// CHECK-PRIVATE: ^bb0(%{{.*}}: memref<10x20xf32>):
// CHECK-PRIVATE:   %[[ALLOC:.*]] = memref.alloca() {acc.var_name = #acc.var_name<"static_2d">} : memref<10x20xf32>
// CHECK-PRIVATE:   acc.yield %[[ALLOC]] : memref<10x20xf32>
// CHECK-PRIVATE: }
// CHECK-PRIVATE-NOT: destroy

func.func @test_static_2d() {
  %0 = memref.alloca() {test.var = "static_2d"} : memref<10x20xf32>
  return
}

// -----

// CHECK-PRIVATE: acc.private.recipe @private_dynamic_2d : memref<?x?xf32> init {
// CHECK-PRIVATE: ^bb0(%[[ARG:.*]]: memref<?x?xf32>):
// CHECK-PRIVATE:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-PRIVATE:   %[[DIM0:.*]] = memref.dim %[[ARG]], %[[C0]] : memref<?x?xf32>
// CHECK-PRIVATE:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-PRIVATE:   %[[DIM1:.*]] = memref.dim %[[ARG]], %[[C1]] : memref<?x?xf32>
// CHECK-PRIVATE:   %[[ALLOC:.*]] = memref.alloc(%[[DIM0]], %[[DIM1]]) {acc.var_name = #acc.var_name<"dynamic_2d">} : memref<?x?xf32>
// CHECK-PRIVATE:   acc.yield %[[ALLOC]] : memref<?x?xf32>
// CHECK-PRIVATE: } destroy {
// CHECK-PRIVATE: ^bb0(%{{.*}}: memref<?x?xf32>, %[[VAL:.*]]: memref<?x?xf32>):
// CHECK-PRIVATE:   memref.dealloc %[[VAL]] : memref<?x?xf32>
// CHECK-PRIVATE:   acc.terminator
// CHECK-PRIVATE: }

func.func @test_dynamic_2d(%arg0: index, %arg1: index) {
  %0 = memref.alloc(%arg0, %arg1) {test.var = "dynamic_2d"} : memref<?x?xf32>
  return
}

// -----

// CHECK-PRIVATE: acc.private.recipe @private_mixed_dims : memref<10x?xf32> init {
// CHECK-PRIVATE: ^bb0(%[[ARG:.*]]: memref<10x?xf32>):
// CHECK-PRIVATE:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-PRIVATE:   %[[DIM1:.*]] = memref.dim %[[ARG]], %[[C1]] : memref<10x?xf32>
// CHECK-PRIVATE:   %[[ALLOC:.*]] = memref.alloc(%[[DIM1]]) {acc.var_name = #acc.var_name<"mixed_dims">} : memref<10x?xf32>
// CHECK-PRIVATE:   acc.yield %[[ALLOC]] : memref<10x?xf32>
// CHECK-PRIVATE: } destroy {
// CHECK-PRIVATE: ^bb0(%{{.*}}: memref<10x?xf32>, %[[VAL:.*]]: memref<10x?xf32>):
// CHECK-PRIVATE:   memref.dealloc %[[VAL]] : memref<10x?xf32>
// CHECK-PRIVATE:   acc.terminator
// CHECK-PRIVATE: }

func.func @test_mixed_dims(%arg0: index) {
  %0 = memref.alloc(%arg0) {test.var = "mixed_dims"} : memref<10x?xf32>
  return
}

// -----

// CHECK-PRIVATE: acc.private.recipe @private_scalar_int : memref<i32> init {
// CHECK-PRIVATE: ^bb0(%{{.*}}: memref<i32>):
// CHECK-PRIVATE:   %[[ALLOC:.*]] = memref.alloca() {acc.var_name = #acc.var_name<"scalar_int">} : memref<i32>
// CHECK-PRIVATE:   acc.yield %[[ALLOC]] : memref<i32>
// CHECK-PRIVATE: }
// CHECK-PRIVATE-NOT: destroy

func.func @test_scalar_int() {
  %0 = memref.alloca() {test.var = "scalar_int"} : memref<i32>
  return
}

// -----

// Verify that we can create a private recipe using the convenience overload
// that takes an existing firstprivate recipe as input.
// CHECK-PRIV-FROM-FIRST: acc.firstprivate.recipe @first_firstprivate_scalar : memref<f32> init
// CHECK-PRIV-FROM-FIRST: acc.private.recipe @private_from_firstprivate_scalar : memref<f32> init {
// CHECK-PRIV-FROM-FIRST: ^bb0(%{{.*}}: memref<f32>):
// CHECK-PRIV-FROM-FIRST:   %[[ALLOC:.*]] = memref.alloca() {acc.var_name = #acc.var_name<"scalar">} : memref<f32>
// CHECK-PRIV-FROM-FIRST:   acc.yield %[[ALLOC]] : memref<f32>
// CHECK-PRIV-FROM-FIRST: }

func.func @test_scalar_from_firstprivate() {
  %0 = memref.alloca() {test.var = "scalar"} : memref<f32>
  return
}

// -----

// Verify that destroy regions are also cloned when creating a private recipe
// from a firstprivate recipe that requires deallocation.
// CHECK-PRIV-FROM-FIRST: acc.firstprivate.recipe @first_firstprivate_dynamic_d2 : memref<?x?xf32> init {
// CHECK-PRIV-FROM-FIRST: } copy {
// CHECK-PRIV-FROM-FIRST: } destroy {
// CHECK-PRIV-FROM-FIRST:   memref.dealloc
// CHECK-PRIV-FROM-FIRST:   acc.terminator
// CHECK-PRIV-FROM-FIRST: }
// CHECK-PRIV-FROM-FIRST: acc.private.recipe @private_from_firstprivate_dynamic_d2 : memref<?x?xf32> init {
// CHECK-PRIV-FROM-FIRST: } destroy {
// CHECK-PRIV-FROM-FIRST:   memref.dealloc
// CHECK-PRIV-FROM-FIRST:   acc.terminator
// CHECK-PRIV-FROM-FIRST: }

func.func @test_dynamic_from_firstprivate(%arg0: index, %arg1: index) {
  %0 = memref.alloc(%arg0, %arg1) {test.var = "dynamic_d2"} : memref<?x?xf32>
  return
}

