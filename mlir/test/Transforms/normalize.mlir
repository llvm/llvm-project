// RUN: mlir-opt %s -pass-pipeline="builtin.module(func.func(normalize))" -split-input-file | FileCheck %s
// RUN: mlir-opt %s -pass-pipeline="builtin.module(func.func(normalize{fold-depth=1}))" -mlir-use-nameloc-as-prefix -split-input-file | FileCheck %s -check-prefix=CHECK-NAMELOC
// RUN: mlir-opt %s -pass-pipeline="builtin.module(func.func(normalize{fold-depth=0}))" -mlir-use-nameloc-as-prefix -split-input-file | FileCheck %s -check-prefix=CHECK-NAMELOC-DEPTH-0
// RUN: mlir-opt %s -pass-pipeline="builtin.module(func.func(normalize{fold-depth=2}))" -mlir-use-nameloc-as-prefix -split-input-file | FileCheck %s -check-prefix=CHECK-NAMELOC-DEPTH-2

// This test verifies op ordering and the sorting of commutative operands.

// CHECK-LABEL: func @multiple_memref_store
//  CHECK-SAME:   %[[ARG0:.*]]: index,
//  CHECK-SAME:   %[[ARG1:.*]]: memref<?xf32>
func.func @multiple_memref_store(%arg0: index, %arg1 : memref<?xf32>) {
  %f0 = arith.constant 0.0 : f32
  %f1 = arith.constant 1.0 : f32
  %add = arith.addi %arg0, %arg0 : index
  %sub = arith.addi %add, %arg0 : index
  memref.store %f0, %arg1[%add] : memref<?xf32>
  memref.store %f1, %arg1[%sub] : memref<?xf32>
  return
}

// CHECK-NEXT: %[[C_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT: %[[ADD_0:.*]] = arith.addi %[[ARG0]], %[[ARG0]] : index
// CHECK-NEXT: memref.store %[[C_0]], %[[ARG1]]{{\[}}%[[ADD_0]]] : memref<?xf32>
// CHECK-NEXT: %[[CONSTANT_1:.*]] = arith.constant 1.000000e+00 : f32
// CHECK-NEXT: %[[ADD_1:.*]] = arith.addi %[[ARG0]], %[[ADD_0]] : index
// CHECK-NEXT: memref.store %[[CONSTANT_1]], %[[ARG1]]{{\[}}%[[ADD_1]]] : memref<?xf32>

// CHECK-NAMELOC-LABEL: func @multiple_memref_store
//  CHECK-NAMELOC-NEXT:   %vl48293 = arith.constant 0.000000e+00 : f32
//  CHECK-NAMELOC-NEXT:   %op13600$-a0.a0-$ = arith.addi %a0, %a0 : index
//  CHECK-NAMELOC-NEXT:   memref.store %vl48293, %a1[%op13600$-a0.a0-$] : memref<?xf32>
//  CHECK-NAMELOC-NEXT:   %vl15553 = arith.constant 1.000000e+00 : f32
//  CHECK-NAMELOC-NEXT:   %op69768$-a0.op13600-$ = arith.addi %a0, %op13600$-a0.a0-$ : index
//  CHECK-NAMELOC-NEXT:   memref.store %vl15553, %a1[%op69768$-a0.op13600-$] : memref<?xf32>

// -----

// This test verifies an 'output' op with multiple operands that are all results of another op.

// CHECK-LABEL: func @return_multiple_operands
//  CHECK-SAME:   %[[ARG0:.*]]: index,
//  CHECK-SAME:   %[[ARG1:.*]]: index
func.func @return_multiple_operands (%arg0: index, %arg1: index) -> (index, index) {
  %0 = arith.addi %arg0, %arg1 : index
  %1 = arith.subi %arg0, %arg1 : index
  return %1, %0 : index, index
}

// CHECK-NEXT: %[[SUB:.*]] = arith.subi %[[ARG0]], %[[ARG1]] : index
// CHECK-NEXT: %[[ADD:.*]] = arith.addi %[[ARG0]], %[[ARG1]] : index
// CHECK-NEXT: return %[[SUB]], %[[ADD]] : index, index

// CHECK-NAMELOC-LABEL: func @return_multiple_operands
//  CHECK-NAMELOC-NEXT:   %op89776$-a0.a1-$ = arith.subi %a0, %a1 : index
//  CHECK-NAMELOC-NEXT:   %op13600$-a0.a1-$ = arith.addi %a0, %a1 : index
//  CHECK-NAMELOC-NEXT:   return %op89776$-a0.a1-$, %op13600$-a0.a1-$ : index, index

// -----

// This test checks if '%add' is scheduled down to the second 'memref.store' site.

// CHECK-LABEL: func @cross_region
//  CHECK-SAME:   %[[ARG0:.*]]: f32,
//  CHECK-SAME:   %[[ARG1:.*]]: memref<10xf32>
func.func @cross_region(%arg0: f32, %arg1 : memref<10xf32>) {
  %add = arith.addf %arg0, %arg0 : f32
  affine.for %i = 0 to 5 {
    memref.store %add, %arg1[%i] : memref<10xf32>
  }
  %exp = math.log2 %add : f32
  affine.for %i = 6 to 10 {
    memref.store %exp, %arg1[%i] : memref<10xf32>
  } 
  return
}

//      CHECK: affine.for %[[IV:.*]] = 6 to 10 {
// CHECK-NEXT:   %[[LOG:.*]] = math.log2
// CHECK-NEXT:   memref.store %[[LOG]], %[[ARG1]]{{\[}}%[[IV]]] : memref<10xf32>
// CHECK-NEXT: }

// CHECK-NAMELOC-LABEL: func @cross_region
//       CHECK-NAMELOC:   affine.for %a3 = 6 to 10 {
//  CHECK-NAMELOC-NEXT:     %op16592$-op10970-$ = math.log2 %op10970$-a0.a0-$ : f32
//  CHECK-NAMELOC-NEXT:     memref.store %op16592$-op10970-$, %a1[%a3] : memref<10xf32>
//  CHECK-NAMELOC-NEXT:   }

// -----

// This test verifies the reordering of scf.for ops.
// The memref.store within the scf.for causes the loop to have side effects.
// The lower bound of the scf.for remains in its original position
// because the upper bound depends on it, but the step has been reordered.

// CHECK-LABEL: func @side_effect_loop_op
//  CHECK-SAME:   %[[ARG0:.*]]: memref<?xf32>
func.func @side_effect_loop_op(%arg1 : memref<?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %upper = memref.dim %arg1, %c0 : memref<?xf32>
  %f1 = arith.constant 1.0 : f32
  scf.for %i = %c0 to %upper step %c1 {
    memref.store %f1, %arg1[%i] : memref<?xf32>
  }
  return
}

// CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT: %[[DIM:.*]] = memref.dim %[[ARG0]], %[[C0]] : memref<?xf32>
// CHECK-NEXT: %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT: scf.for %[[IV:.*]] = %[[C0]] to %[[DIM]] step %[[C1]] {
// CHECK-NEXT:   %[[F1:.*]] = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:   memref.store %[[F1]], %[[ARG0]]{{\[}}%[[IV]]] : memref<?xf32>
// CHECK-NEXT: }

// CHECK-NAMELOC-LABEL: func @side_effect_loop_op
//  CHECK-NAMELOC-NEXT:   %vl15499 = arith.constant 0 : index
//  CHECK-NAMELOC-NEXT:   %op18509$-a0.vl15499-$ = memref.dim %a0, %vl15499 : memref<?xf32>
//  CHECK-NAMELOC-NEXT:   %vl14483 = arith.constant 1 : index
//  CHECK-NAMELOC-NEXT:   scf.for %a1 = %vl15499 to %op18509$-a0.vl15499-$ step %vl14483 {
//  CHECK-NAMELOC-NEXT:     %vl71256 = arith.constant 1.000000e+00 : f32
//  CHECK-NAMELOC-NEXT:     memref.store %vl71256, %a0[%a1] : memref<?xf32>
//  CHECK-NAMELOC-NEXT:   }

// -----

// This test verifies the naming of func.call operations.

func.func private @callee_0(%arg0: f32) -> f32
func.func private @callee_1(%arg0: f32) -> f32

func.func @test_call_operation(%arg0: f32) -> f32 {
  %0 = func.call @callee_0(%arg0) : (f32) -> f32
  %1 = func.call @callee_1(%0) : (f32) -> f32
  return %1 : f32
}

// CHECK-NAMELOC-LABEL: func @test_call_operation
//  CHECK-NAMELOC-NEXT:   %op71372callee_0$-a0-$ = call @callee_0(%a0) : (f32) -> f32
//  CHECK-NAMELOC-NEXT:   %op15508callee_1$-op71372callee_0-$ = call @callee_1(%op71372callee_0$-a0-$) : (f32) -> f32
//  CHECK-NAMELOC-NEXT:   return %op15508callee_1$-op71372callee_0-$ : f32

// -----

func.func @deep_use_chain(%arg0: i32, %arg1: i32) -> i32 {
  %0 = arith.addi %arg0, %arg1 : i32
  %1 = arith.muli %arg0, %0 : i32
  %2 = arith.addi %arg0, %1 : i32
  %3 = arith.muli %2, %2 : i32
  %4 = arith.addi %arg1, %3 : i32
  return %4 : i32
}

// CHECK-NAMELOC-DEPTH-0-LABEL: func @deep_use_chain
//  CHECK-NAMELOC-DEPTH-0-NEXT:   %op11483 = arith.addi %a0, %a1 : i32
//  CHECK-NAMELOC-DEPTH-0-NEXT:   %op80011 = arith.muli %a0, %op11483 : i32
//  CHECK-NAMELOC-DEPTH-0-NEXT:   %op14133 = arith.addi %a0, %op80011 : i32
//  CHECK-NAMELOC-DEPTH-0-NEXT:   %op27579 = arith.muli %op14133, %op14133 : i32
//  CHECK-NAMELOC-DEPTH-0-NEXT:   %op17770 = arith.addi %a1, %op27579 : i32
//  CHECK-NAMELOC-DEPTH-0-NEXT:   return %op17770 : i32

// CHECK-NAMELOC-LABEL: func @deep_use_chain
//  CHECK-NAMELOC-NEXT:   %op11483$-a0.a1-$ = arith.addi %a0, %a1 : i32
//  CHECK-NAMELOC-NEXT:   %op80011$-a0.op11483-$ = arith.muli %a0, %op11483$-a0.a1-$ : i32
//  CHECK-NAMELOC-NEXT:   %op14133$-a0.op80011-$ = arith.addi %a0, %op80011$-a0.op11483-$ : i32
//  CHECK-NAMELOC-NEXT:   %op27579$-op14133-$ = arith.muli %op14133$-a0.op80011-$, %op14133$-a0.op80011-$ : i32
//  CHECK-NAMELOC-NEXT:   %op17770$-a1.op27579-$ = arith.addi %a1, %op27579$-op14133-$ : i32
//  CHECK-NAMELOC-NEXT:   return %op17770$-a1.op27579-$ : i32

// CHECK-NAMELOC-DEPTH-2-LABEL: func @deep_use_chain
//  CHECK-NAMELOC-DEPTH-2-NEXT:   %op11483$-a0.a1-$ = arith.addi %a0, %a1 : i32
//  CHECK-NAMELOC-DEPTH-2-NEXT:   %op80011$-a0.op11483$-a0.a1-$-$ = arith.muli %a0, %op11483$-a0.a1-$ : i32
//  CHECK-NAMELOC-DEPTH-2-NEXT:   %op14133$-a0.op80011$-a0.op11483-$-$ = arith.addi %a0, %op80011$-a0.op11483$-a0.a1-$-$ : i32
//  CHECK-NAMELOC-DEPTH-2-NEXT:   %op27579$-op14133$-a0.op80011-$-$ = arith.muli %op14133$-a0.op80011$-a0.op11483-$-$, %op14133$-a0.op80011$-a0.op11483-$-$ : i32
//  CHECK-NAMELOC-DEPTH-2-NEXT:   %op17770$-a1.op27579$-op14133-$-$ = arith.addi %a1, %op27579$-op14133$-a0.op80011-$-$ : i32
//  CHECK-NAMELOC-DEPTH-2-NEXT:   return %op17770$-a1.op27579$-op14133-$-$ : i32
