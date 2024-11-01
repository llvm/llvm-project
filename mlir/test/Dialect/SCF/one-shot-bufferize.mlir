// RUN: mlir-opt %s -allow-unregistered-dialect -one-shot-bufferize="allow-return-allocs bufferize-function-boundaries" -drop-equivalent-buffer-results -buffer-deallocation -split-input-file | FileCheck %s
// RUN: mlir-opt %s -allow-unregistered-dialect -one-shot-bufferize="allow-return-allocs bufferize-function-boundaries" -drop-equivalent-buffer-results -split-input-file | FileCheck %s --check-prefix=CHECK-NO-DEALLOC-PASS

// Run fuzzer with different seeds.
// RUN: mlir-opt %s -allow-unregistered-dialect -one-shot-bufferize="allow-return-allocs test-analysis-only analysis-fuzzer-seed=23 bufferize-function-boundaries" -split-input-file -o /dev/null
// RUN: mlir-opt %s -allow-unregistered-dialect -one-shot-bufferize="allow-return-allocs test-analysis-only analysis-fuzzer-seed=59 bufferize-function-boundaries" -split-input-file -o /dev/null
// RUN: mlir-opt %s -allow-unregistered-dialect -one-shot-bufferize="allow-return-allocs test-analysis-only analysis-fuzzer-seed=91 bufferize-function-boundaries" -split-input-file -o /dev/null

// Test bufferization using memref types that have no layout map.
// RUN: mlir-opt %s -allow-unregistered-dialect -one-shot-bufferize="allow-return-allocs unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map bufferize-function-boundaries" -buffer-deallocation -split-input-file -o /dev/null

// CHECK-LABEL: func @scf_for_yield_only(
//  CHECK-SAME:   %[[A:[a-zA-Z0-9]*]]: memref<?xf32, strided<[?], offset: ?>>,
//  CHECK-SAME:   %[[t:[a-zA-Z0-9]*]]: memref<?xf32, strided<[?], offset: ?>>
//  CHECK-SAME:   ) -> memref<?xf32> {
func.func @scf_for_yield_only(
    %A : tensor<?xf32> {bufferization.writable = false},
    %B : tensor<?xf32> {bufferization.writable = true},
    %lb : index, %ub : index, %step : index)
  -> (tensor<?xf32>, tensor<?xf32>)
{
  //     CHECK:   %[[ALLOC_FOR_A:.*]] = memref.alloc
  //     CHECK:   memref.copy %[[A]], %[[ALLOC_FOR_A]]

  // The first scf.for remains but just turns into dead code.
  %r0 = scf.for %i = %lb to %ub step %step iter_args(%t = %A) -> (tensor<?xf32>) {
    scf.yield %t : tensor<?xf32>
  }

  // The second scf.for remains but just turns into dead code.
  %r1 = scf.for %i = %lb to %ub step %step iter_args(%t = %B) -> (tensor<?xf32>) {
    scf.yield %t : tensor<?xf32>
  }

  //     CHECK:   return %[[ALLOC_FOR_A]] : memref<?xf32>
  // CHECK-NOT:   dealloc
  return %r0, %r1: tensor<?xf32>, tensor<?xf32>
}

// -----

// CHECK-LABEL: func @scf_for_is_reading(
//  CHECK-SAME:     %[[A:.*]]: memref<?xf32, strided<[?], offset: ?>>, %[[B:.*]]: memref<?xf32, strided<[?], offset: ?>>
func.func @scf_for_is_reading(%A : tensor<?xf32>, %B : tensor<?xf32>,
                              %lb : index, %ub : index)
  -> (f32, f32)
{
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.0 : f32

  // This is a regression test to make sure that an alloc + copy is emitted.

  // CHECK: %[[alloc:.*]] = memref.alloc
  // CHECK: memref.copy %[[A]], %[[alloc]]
  // CHECK: %[[clone:.*]] = bufferization.clone %[[alloc]]
  // CHECK: scf.for {{.*}} iter_args(%{{.*}} = %[[clone]])
  %0 = scf.for %iv = %lb to %ub step %c1 iter_args(%1 = %A) -> tensor<?xf32> {
    %r = linalg.fill ins(%cst : f32) outs(%1 : tensor<?xf32>) -> tensor<?xf32>
    scf.yield %B : tensor<?xf32>
  }
  %1 = tensor.extract %0[%c1] : tensor<?xf32>
  %2 = tensor.extract %A[%c1] : tensor<?xf32>
  return %1, %2 : f32, f32
}

// -----

// Ensure that the function bufferizes without error. This tests pre-order
// traversal of scf.for loops during bufferization. No need to check the IR,
// just want to make sure that it does not crash.

// CHECK-LABEL: func @nested_scf_for
func.func @nested_scf_for(%A : tensor<?xf32> {bufferization.writable = true},
                          %v : vector<5xf32>) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %r1 = scf.for %i = %c0 to %c10 step %c1 iter_args(%B = %A) -> tensor<?xf32> {
    %r2 = scf.for %j = %c0 to %c10 step %c1 iter_args(%C = %B) -> tensor<?xf32> {
      %w = vector.transfer_write %v, %C[%c0] : vector<5xf32>, tensor<?xf32>
      scf.yield %w : tensor<?xf32>
    }
    scf.yield %r2 : tensor<?xf32>
  }
  return %r1 : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @scf_for_with_tensor.insert_slice
//  CHECK-SAME:   %[[A:[a-zA-Z0-9]*]]: memref<?xf32, strided<[?], offset: ?>>
//  CHECK-SAME:   %[[B:[a-zA-Z0-9]*]]: memref<?xf32, strided<[?], offset: ?>>
//  CHECK-SAME:   %[[C:[a-zA-Z0-9]*]]: memref<4xf32, strided<[?], offset: ?>>
func.func @scf_for_with_tensor.insert_slice(
    %A : tensor<?xf32> {bufferization.writable = false},
    %B : tensor<?xf32> {bufferization.writable = true},
    %C : tensor<4xf32> {bufferization.writable = false},
    %lb : index, %ub : index, %step : index)
  -> (tensor<?xf32>, tensor<?xf32>)
{
  //     CHECK:   %[[ALLOC_FOR_A:.*]] = memref.alloc
  //     CHECK:   memref.copy %[[A]], %[[ALLOC_FOR_A]]

  //     CHECK: %[[svA:.*]] = memref.subview %[[ALLOC_FOR_A]][0] [4] [1]
  //     CHECK: %[[svB:.*]] = memref.subview %[[B]][0] [4] [1]

  //     CHECK:   scf.for {{.*}}
  // CHECK-NOT: iter_args
  %r0:2 = scf.for %i = %lb to %ub step %step iter_args(%tA = %A, %tB = %B)
      -> (tensor<?xf32>, tensor<?xf32>)
  {
    // %ttA bufferizes to direct copy of %BUFFER_CAST_C into %svA
    //     CHECK: memref.copy %[[C]], %[[svA]]
    %ttA = tensor.insert_slice %C into %tA[0][4][1] : tensor<4xf32> into tensor<?xf32>

    // %ttB bufferizes to direct copy of %BUFFER_CAST_C into %BUFFER_CAST_B
    //     CHECK:   memref.copy %[[C]], %[[svB]]
    %ttB = tensor.insert_slice %C into %tB[0][4][1] : tensor<4xf32> into tensor<?xf32>

    // CHECK-NOT:   scf.yield
    scf.yield %ttA, %ttB : tensor<?xf32>, tensor<?xf32>
  }

  //     CHECK:  return %[[ALLOC_FOR_A]] : memref<?xf32>
  return %r0#0, %r0#1: tensor<?xf32>, tensor<?xf32>
}

// -----

// CHECK-LABEL: func @execute_region_with_conflict(
//  CHECK-SAME:     %[[m1:.*]]: memref<?xf32
func.func @execute_region_with_conflict(
    %t1 : tensor<?xf32> {bufferization.writable = true})
  -> (f32, tensor<?xf32>, f32)
{
  %f1 = arith.constant 0.0 : f32
  %idx = arith.constant 7 : index

  // scf.execute_region is canonicalized away after bufferization. So just the
  // memref.store is left over.

  // CHECK: %[[alloc:.*]] = memref.alloc
  // CHECK: memref.copy %[[m1]], %[[alloc]]
  // CHECK: memref.store %{{.*}}, %[[alloc]][%{{.*}}]
  %0, %1, %2 = scf.execute_region -> (f32, tensor<?xf32>, f32) {
    %t2 = tensor.insert %f1 into %t1[%idx] : tensor<?xf32>
    scf.yield %f1, %t2, %f1 : f32, tensor<?xf32>, f32
  }

  // CHECK: %[[load:.*]] = memref.load %[[m1]]
  %3 = tensor.extract %t1[%idx] : tensor<?xf32>

  // CHECK: return %{{.*}}, %[[alloc]], %[[load]] : f32, memref<?xf32>, f32
  return %0, %1, %3 : f32, tensor<?xf32>, f32
}

// -----

// CHECK-LABEL: func @scf_if_inplace(
//  CHECK-SAME:     %[[cond:.*]]: i1, %[[t1:.*]]: memref<?xf32{{.*}}>, %[[v:.*]]: vector
func.func @scf_if_inplace(%cond: i1,
                          %t1: tensor<?xf32> {bufferization.writable = true},
                          %v: vector<5xf32>, %idx: index) -> tensor<?xf32> {

  //      CHECK: scf.if %[[cond]] {
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   vector.transfer_write %[[v]], %[[t1]]
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  %r = scf.if %cond -> (tensor<?xf32>) {
    scf.yield %t1 : tensor<?xf32>
  } else {
    %t2 = vector.transfer_write %v, %t1[%idx] : vector<5xf32>, tensor<?xf32>
    scf.yield %t2 : tensor<?xf32>
  }
  return %r : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @scf_if_inside_scf_for
//   CHECK-DAG:   %[[c0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[c1:.*]] = arith.constant 1 : index
//   CHECK-DAG:   %[[c10:.*]] = arith.constant 10 : index
//       CHECK:   scf.for %{{.*}} = %[[c0]] to %[[c10]] step %[[c1]] {
//       CHECK:     scf.if %{{.*}} {
//       CHECK:     } else {
//       CHECK:       vector.transfer_write
//       CHECK:     }
//       CHECK:   }
func.func @scf_if_inside_scf_for(
    %t1: tensor<?xf32> {bufferization.writable = true},
    %v: vector<5xf32>, %idx: index,
    %cond: i1)
  -> tensor<?xf32>
{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %r = scf.for %iv = %c0 to %c10 step %c1 iter_args(%bb = %t1) -> (tensor<?xf32>) {
    %r2 = scf.if %cond -> (tensor<?xf32>) {
      scf.yield %bb : tensor<?xf32>
    } else {
      %t2 = vector.transfer_write %v, %bb[%idx] : vector<5xf32>, tensor<?xf32>
      scf.yield %t2 : tensor<?xf32>
    }
    scf.yield %r2 : tensor<?xf32>
  }
  return %r : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @scf_if_non_equiv_yields(
//  CHECK-SAME:     %[[cond:.*]]: i1, %[[A:.*]]: memref<{{.*}}>, %[[B:.*]]: memref<{{.*}}>) -> memref<{{.*}}>
func.func @scf_if_non_equiv_yields(
    %b : i1,
    %A : tensor<4xf32> {bufferization.writable = false},
    %B : tensor<4xf32> {bufferization.writable = false})
  -> tensor<4xf32>
{
  // CHECK: %[[r:.*]] = arith.select %[[cond]], %[[A]], %[[B]]
  %r = scf.if %b -> (tensor<4xf32>) {
    scf.yield %A : tensor<4xf32>
  } else {
    scf.yield %B : tensor<4xf32>
  }
  // CHECK: return %[[r]]
  return %r: tensor<4xf32>
}

// -----

// Note: This bufferization is inefficient, but it bufferizes correctly.

// CHECK-LABEL: func @scf_execute_region_yield_non_equivalent(
//       CHECK:   %[[alloc:.*]] = memref.alloc(%{{.*}})
//       CHECK:   %[[r:.*]] = memref.load %[[alloc]][%{{.*}}]
//       CHECK:   memref.dealloc %[[alloc]]
//       CHECK:   return %[[r]]
func.func @scf_execute_region_yield_non_equivalent(%i: index, %j: index) -> f32 {
  %r = scf.execute_region -> (tensor<?xf32>) {
    %t2 = bufferization.alloc_tensor(%i) : tensor<?xf32>
    scf.yield %t2 : tensor<?xf32>
  }
  %f = tensor.extract %r[%j] : tensor<?xf32>
  return %f : f32
}

// -----

// Note: This bufferizes to inefficient code, but bufferization should not see
// such IR in the first place. The iter_arg would canonicalize away. This test
// case is just to ensure that the bufferization generates correct code.

// CHECK-LABEL: func @scf_for_yield_non_equivalent(
//  CHECK-SAME:     %[[t:.*]]: memref<?xf32
//       CHECK:   %[[alloc:.*]] = memref.alloc(%{{.*}})
//       CHECK:   memref.copy %[[t]], %[[alloc]]
//       CHECK:   %[[cloned:.*]] = bufferization.clone %[[t]]
//       CHECK:   %[[for:.*]] = scf.for {{.*}} iter_args(%[[iter:.*]] = %[[cloned]])
//   CHECK-DAG:     memref.dealloc %[[iter]]
//   CHECK-DAG:     %[[alloc2:.*]] = memref.alloc(%{{.*}})
//       CHECK:     memref.copy %[[alloc]], %[[alloc2]]
//       CHECK:     %[[alloc2_casted:.*]] = memref.cast %[[alloc2]]
//       CHECK:     %[[cloned2:.*]] = bufferization.clone %[[alloc2_casted]]
//       CHECK:     memref.dealloc %[[alloc2]]
//       CHECK:     scf.yield %[[cloned2]]
//       CHECK:   memref.dealloc %[[alloc]]
//       CHECK:   return %[[for]]
func.func @scf_for_yield_non_equivalent(
    %t: tensor<?xf32>, %lb : index, %ub : index, %step : index) -> tensor<?xf32> {
  %r = scf.for %i = %lb to %ub step %step iter_args(%a = %t) -> tensor<?xf32> {
    scf.yield %t : tensor<?xf32>
  }

  return %r : tensor<?xf32>
}

// -----

// Note: This bufferizes to inefficient code, but bufferization should not see
// such IR in the first place. The iter_arg would canonicalize away. This test
// case is just to ensure that the bufferization generates correct code.

// CHECK-LABEL: func @scf_for_yield_allocation(
//  CHECK-SAME:     %[[t:.*]]: memref<?xf32
//       CHECK:   %[[cloned:.*]] = bufferization.clone %[[t]]
//       CHECK:   %[[for:.*]] = scf.for {{.*}} iter_args(%[[iter:.*]] = %[[cloned]])
// This alloc is for the bufferization.alloc_tensor.
//   CHECK-DAG:     %[[alloc2:.*]] = memref.alloc(%{{.*}})
//   CHECK-DAG:     memref.dealloc %[[iter]]
// This alloc is for the scf.yield.
//       CHECK:     %[[alloc3:.*]] = memref.alloc(%{{.*}})
//       CHECK:     memref.copy %[[alloc2]], %[[alloc3]]
//       CHECK:     memref.dealloc %[[alloc2]]
//       CHECK:     %[[casted3:.*]] = memref.cast %[[alloc3]]
//       CHECK:     %[[cloned3:.*]] = bufferization.clone %[[casted3]]
//       CHECK:     memref.dealloc %[[alloc3]]
//       CHECK:     scf.yield %[[cloned3]]
//       CHECK:   return %[[for]]
func.func @scf_for_yield_allocation(%t: tensor<?xf32>, %lb : index, %ub : index,
                               %step : index) -> tensor<?xf32> {
  %r = scf.for %i = %lb to %ub step %step iter_args(%a = %t) -> tensor<?xf32> {
    %t2 = bufferization.alloc_tensor(%i) : tensor<?xf32>
    scf.yield %t2 : tensor<?xf32>
  }

  return %r : tensor<?xf32>
}

// -----

// TODO: The scf.yield could bufferize to 1 alloc and 2 copies (instead of
// 2 allocs and 2 copies).

// CHECK-LABEL: func @scf_for_swapping_yields(
//  CHECK-SAME:     %[[A:.*]]: memref<?xf32, strided{{.*}}>, %[[B:.*]]: memref<?xf32, strided{{.*}}>
func.func @scf_for_swapping_yields(
    %A : tensor<?xf32>, %B : tensor<?xf32> {bufferization.writable = true},
    %C : tensor<4xf32>, %lb : index, %ub : index, %step : index)
  -> (f32, f32)
{
//   CHECK-DAG:   %[[clone1:.*]] = bufferization.clone %[[A]]
//   CHECK-DAG:   %[[clone2:.*]] = bufferization.clone %[[B]]
//       CHECK:   %[[for:.*]]:2 = scf.for {{.*}} iter_args(%[[iter1:.*]] = %[[clone1]], %[[iter2:.*]] = %[[clone2]])
  %r0:2 = scf.for %i = %lb to %ub step %step iter_args(%tA = %A, %tB = %B)
      -> (tensor<?xf32>, tensor<?xf32>)
  {
//       CHECK:     %[[sv1:.*]] = memref.subview %[[iter1]]
//       CHECK:     memref.copy %{{.*}}, %[[sv1]]
    %ttA = tensor.insert_slice %C into %tA[0][4][1] : tensor<4xf32> into tensor<?xf32>
//       CHECK:     %[[sv2:.*]] = memref.subview %[[iter2]]
//       CHECK:     memref.copy %{{.*}}, %[[sv2]]
    %ttB = tensor.insert_slice %C into %tB[0][4][1] : tensor<4xf32> into tensor<?xf32>

//       CHECK:     %[[alloc2:.*]] = memref.alloc(%{{.*}})
//       CHECK:     memref.copy %[[iter2]], %[[alloc2]]
//       CHECK:     memref.dealloc %[[iter2]]
//       CHECK:     %[[alloc1:.*]] = memref.alloc(%{{.*}})
//       CHECK:     memref.copy %[[iter1]], %[[alloc1]]
//       CHECK:     memref.dealloc %[[iter1]]
//       CHECK:     %[[casted2:.*]] = memref.cast %[[alloc2]]
//       CHECK:     %[[casted1:.*]] = memref.cast %[[alloc1]]
//       CHECK:     %[[cloned1:.*]] = bufferization.clone %[[casted1]]
//       CHECK:     memref.dealloc %[[alloc1]]
//       CHECK:     %[[cloned2:.*]] = bufferization.clone %[[casted2]]
//       CHECK:     memref.dealloc %[[alloc2]]
//       CHECK:     scf.yield %[[cloned2]], %[[cloned1]]
    // Yield tensors in different order.
    scf.yield %ttB, %ttA : tensor<?xf32>, tensor<?xf32>
  }

//       CHECK:     %[[r0:.*]] = memref.load %[[for]]#0
//       CHECK:     memref.dealloc %[[for]]#0
//       CHECK:     %[[r1:.*]] = memref.load %[[for]]#1
//       CHECK:     memref.dealloc %[[for]]#1
  %f0 = tensor.extract %r0#0[%step] : tensor<?xf32>
  %f1 = tensor.extract %r0#1[%step] : tensor<?xf32>
//       CHECK:     return %[[r0]], %[[r1]]
  return %f0, %f1: f32, f32
}

// -----

// CHECK-LABEL: func @scf_while(
//  CHECK-SAME:     %[[arg0:.*]]: memref<?xi1, strided{{.*}}>
func.func @scf_while(%arg0: tensor<?xi1>, %idx: index) -> tensor<?xi1> {
  // CHECK: scf.while : () -> () {
  %res:2 = scf.while (%arg1 = %arg0, %i = %idx) :
      (tensor<?xi1>, index) -> (tensor<?xi1>, index) {
    // CHECK: %[[condition:.*]] = memref.load %[[arg0]]
    // CHECK: scf.condition(%[[condition]])
    %condition = tensor.extract %arg1[%idx] : tensor<?xi1>
    scf.condition(%condition) %arg1, %idx : tensor<?xi1>, index
  } do {
  ^bb0(%arg2: tensor<?xi1>, %i: index):
    // CHECK: } do {
    // CHECK: memref.store %{{.*}}, %[[arg0]]
    // CHECK: scf.yield
    // CHECK: }
    %pos = "dummy.some_op"() : () -> (index)
    %val = "dummy.another_op"() : () -> (i1)
    %1 = tensor.insert %val into %arg2[%pos] : tensor<?xi1>
    scf.yield %1, %i : tensor<?xi1>, index
  }

  // CHECK: return
  return %res#0 : tensor<?xi1>
}

// -----

// The loop condition yields non-equivalent buffers.

// CHECK-LABEL: func @scf_while_non_equiv_condition(
//  CHECK-SAME:     %[[arg0:.*]]: memref<5xi1, strided{{.*}}>, %[[arg1:.*]]: memref<5xi1, strided{{.*}}>
func.func @scf_while_non_equiv_condition(%arg0: tensor<5xi1>,
                                         %arg1: tensor<5xi1>,
                                         %idx: index)
  -> (tensor<5xi1>, tensor<5xi1>)
{
  // CHECK: %[[clone1:.*]] = bufferization.clone %[[arg1]]
  // CHECK: %[[clone0:.*]] = bufferization.clone %[[arg0]]
  // CHECK: %[[loop:.*]]:2 = scf.while (%[[w0:.*]] = %[[clone0]], %[[w1:.*]] = %[[clone1]]) {{.*}} {
  %r0, %r1 = scf.while (%w0 = %arg0, %w1 = %arg1)
      : (tensor<5xi1>, tensor<5xi1>) -> (tensor<5xi1>, tensor<5xi1>) {
    // CHECK: %[[condition:.*]] = memref.load %[[w0]]
    // CHECK: %[[a1:.*]] = memref.alloc() {{.*}} : memref<5xi1>
    // CHECK: memref.copy %[[w1]], %[[a1]]
    // CHECK: memref.dealloc %[[w1]]
    // CHECK: %[[a0:.*]] = memref.alloc() {{.*}} : memref<5xi1>
    // CHECK: memref.copy %[[w0]], %[[a0]]
    // CHECK: memref.dealloc %[[w0]]
    // CHECK: %[[cloned1:.*]] = bufferization.clone %[[a1]]
    // CHECK: memref.dealloc %[[a1]]
    // CHECK: %[[cloned0:.*]] = bufferization.clone %[[a0]]
    // CHECK: memref.dealloc %[[a0]]
    // CHECK: scf.condition(%[[condition]]) %[[cloned1]], %[[cloned0]]
    %condition = tensor.extract %w0[%idx] : tensor<5xi1>
    scf.condition(%condition) %w1, %w0 : tensor<5xi1>, tensor<5xi1>
  } do {
  ^bb0(%b0: tensor<5xi1>, %b1: tensor<5xi1>):
    // CHECK: } do {
    // CHECK: ^bb0(%[[b0:.*]]: memref<5xi1>, %[[b1:.*]]: memref<5xi1>):
    // CHECK: memref.store %{{.*}}, %[[b0]]
    // CHECK: %[[casted0:.*]] = memref.cast %[[b0]] : memref<5xi1> to memref<5xi1, strided{{.*}}>
    // CHECK: %[[casted1:.*]] = memref.cast %[[b1]] : memref<5xi1> to memref<5xi1, strided{{.*}}>
    // CHECK: %[[cloned2:.*]] = bufferization.clone %[[casted1]]
    // CHECK: memref.dealloc %[[b1]]
    // CHECK: %[[cloned3:.*]] = bufferization.clone %[[casted0]]
    // CHECK: memref.dealloc %[[b0]]
    // CHECK: scf.yield %[[cloned3]], %[[cloned2]]
    // CHECK: }
    %pos = "dummy.some_op"() : () -> (index)
    %val = "dummy.another_op"() : () -> (i1)
    %1 = tensor.insert %val into %b0[%pos] : tensor<5xi1>
    scf.yield %1, %b1 : tensor<5xi1>, tensor<5xi1>
  }

  // CHECK: return %[[loop]]#0, %[[loop]]#1
  return %r0, %r1 : tensor<5xi1>, tensor<5xi1>
}

// -----

// Both the loop condition and the loop buffer yield non-equivalent buffers.

// CHECK-LABEL: func @scf_while_non_equiv_condition_and_body(
//  CHECK-SAME:     %[[arg0:.*]]: memref<5xi1, strided{{.*}}>, %[[arg1:.*]]: memref<5xi1, strided{{.*}}>
func.func @scf_while_non_equiv_condition_and_body(%arg0: tensor<5xi1>,
                                                  %arg1: tensor<5xi1>,
                                                  %idx: index)
  -> (tensor<5xi1>, tensor<5xi1>)
{
  // CHECK-DAG: %[[clone1:.*]] = bufferization.clone %[[arg1]]
  // CHECK-DAG: %[[clone0:.*]] = bufferization.clone %[[arg0]]
  // CHECK: %[[loop:.*]]:2 = scf.while (%[[w0:.*]] = %[[clone0]], %[[w1:.*]] = %[[clone1]]) {{.*}} {
  %r0, %r1 = scf.while (%w0 = %arg0, %w1 = %arg1)
      : (tensor<5xi1>, tensor<5xi1>) -> (tensor<5xi1>, tensor<5xi1>) {
    // CHECK: %[[condition:.*]] = memref.load %[[w0]]
    // CHECK: %[[a1:.*]] = memref.alloc() {{.*}} : memref<5xi1>
    // CHECK: memref.copy %[[w1]], %[[a1]]
    // CHECK: memref.dealloc %[[w1]]
    // CHECK: %[[a0:.*]] = memref.alloc() {{.*}} : memref<5xi1>
    // CHECK: memref.copy %[[w0]], %[[a0]]
    // CHECK: memref.dealloc %[[w0]]
    // CHECK: %[[cloned1:.*]] = bufferization.clone %[[a1]]
    // CHECK: memref.dealloc %[[a1]]
    // CHECK: %[[cloned0:.*]] = bufferization.clone %[[a0]]
    // CHECK: memref.dealloc %[[a0]]
    // CHECK: scf.condition(%[[condition]]) %[[cloned1]], %[[cloned0]]
    %condition = tensor.extract %w0[%idx] : tensor<5xi1>
    scf.condition(%condition) %w1, %w0 : tensor<5xi1>, tensor<5xi1>
  } do {
  ^bb0(%b0: tensor<5xi1>, %b1: tensor<5xi1>):
    // CHECK: } do {
    // CHECK: ^bb0(%[[b0:.*]]: memref<5xi1>, %[[b1:.*]]: memref<5xi1>):
    // CHECK: memref.store %{{.*}}, %[[b0]]
    // CHECK: %[[casted1:.*]] = memref.cast %[[b1]]
    // CHECK: %[[casted0:.*]] = memref.cast %[[b0]]
    // CHECK: %[[cloned1:.*]] = bufferization.clone %[[casted1]]
    // CHECK: memref.dealloc %[[b1]]
    // CHECK: %[[cloned0:.*]] = bufferization.clone %[[casted0]]
    // CHECK: memref.dealloc %[[b0]]
    // CHECK: scf.yield %[[cloned1]], %[[cloned0]]
    // CHECK: }
    %pos = "dummy.some_op"() : () -> (index)
    %val = "dummy.another_op"() : () -> (i1)
    %1 = tensor.insert %val into %b0[%pos] : tensor<5xi1>
    scf.yield %b1, %1 : tensor<5xi1>, tensor<5xi1>
  }

  // CHECK: return %[[loop]]#0, %[[loop]]#1
  return %r0, %r1 : tensor<5xi1>, tensor<5xi1>
}

// -----

// CHECK-LABEL: func @scf_while_iter_arg_result_mismatch(
//  CHECK-SAME:     %[[arg0:.*]]: memref<5xi1, strided{{.*}}>, %[[arg1:.*]]: memref<5xi1, strided{{.*}}>
//       CHECK:   %[[clone:.*]] = bufferization.clone %[[arg1]]
//       CHECK:   scf.while (%[[arg3:.*]] = %[[clone]]) : (memref<5xi1, strided{{.*}}) -> () {
//   CHECK-DAG:     memref.dealloc %[[arg3]]
//   CHECK-DAG:     %[[load:.*]] = memref.load %[[arg0]]
//       CHECK:     scf.condition(%[[load]])
//       CHECK:   } do {
//       CHECK:     %[[alloc2:.*]] = memref.alloc() {{.*}} : memref<5xi1>
//       CHECK:     memref.copy %[[arg0]], %[[alloc2]]
//       CHECK:     memref.store %{{.*}}, %[[alloc2]]
//       CHECK:     %[[casted:.*]] = memref.cast %[[alloc2]] : memref<5xi1> to memref<5xi1, strided{{.*}}>
//       CHECK:     %[[cloned:.*]] = bufferization.clone %[[casted]]
//       CHECK:     memref.dealloc %[[alloc2]]
//       CHECK:     scf.yield %[[cloned]]
//       CHECK:   }
func.func @scf_while_iter_arg_result_mismatch(%arg0: tensor<5xi1>,
                                              %arg1: tensor<5xi1>,
                                              %arg2: index) {
  scf.while (%arg3 = %arg1) : (tensor<5xi1>) -> () {
    %0 = tensor.extract %arg0[%arg2] : tensor<5xi1>
    scf.condition(%0)
  } do {
    %0 = "dummy.some_op"() : () -> index
    %1 = "dummy.another_op"() : () -> i1
    %2 = tensor.insert %1 into %arg0[%0] : tensor<5xi1>
    scf.yield %2 : tensor<5xi1>
  }
  return
}

// -----

// CHECK-LABEL: func.func @parallel_insert_slice_no_conflict(
//  CHECK-SAME:     %[[idx:.*]]: index, %[[idx2:.*]]: index,
//  CHECK-SAME:     %[[arg1:.*]]: memref<?xf32, strided{{.*}}>,
//  CHECK-SAME:     %[[arg2:.*]]: memref<?xf32, strided{{.*}}>
func.func @parallel_insert_slice_no_conflict(
    %idx: index,
    %idx2: index,
    %arg1: tensor<?xf32> {bufferization.writable = true},
    %arg2: tensor<?xf32> {bufferization.writable = true}) -> (tensor<?xf32>, f32) {
  %cst = arith.constant 4.200000e+01 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // CHECK: scf.forall (%[[tidx:.*]]) in (%[[idx2]])
  %2 = scf.forall (%arg3) in (%idx2) shared_outs(%o = %arg2) -> (tensor<?xf32>) {
      // CHECK: %[[subview:.*]] = memref.subview %[[arg2]][5] [%[[idx]]] [1]
      %6 = tensor.extract_slice %o[5] [%idx] [%c1] : tensor<?xf32> to tensor<?xf32>
      // CHECK: linalg.fill ins(%{{.*}}) outs(%[[subview]] : memref<?xf32
      %8 = linalg.fill ins(%cst : f32) outs(%6 : tensor<?xf32>) -> tensor<?xf32>
      // Self-copy will DCE away later.
      // CHECK: memref.copy %[[subview]], %[[subview]]

      // Empty terminator is elided from pretty-printing.
      // CHECK-NOT: scf.forall.in_parallel
      // CHECK-NOT: parallel_insert_slice
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %8 into %o[5] [%idx] [%c1] :
          tensor<?xf32> into tensor<?xf32>
      }
  }

  // CHECK: %[[load:.*]] = memref.load %[[arg2]]
  %f = tensor.extract %2[%c0] : tensor<?xf32>

  // CHECK: return %[[load]] : f32
  return %2, %f : tensor<?xf32>, f32
}

// -----

// CHECK-LABEL: func.func @parallel_insert_slice_with_conflict(
//  CHECK-SAME:     %[[idx:.*]]: index, %[[idx2:.*]]: index,
//  CHECK-SAME:     %[[arg1:.*]]: memref<?xf32, strided{{.*}}>,
//  CHECK-SAME:     %[[arg2:.*]]: memref<?xf32, strided{{.*}}>
func.func @parallel_insert_slice_with_conflict(
    %idx: index,
    %idx2: index,
    %arg1: tensor<?xf32> {bufferization.writable = true},
    %arg2: tensor<?xf32> {bufferization.writable = true}) -> (f32, f32)
{
  %cst = arith.constant 4.200000e+01 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // The parallel_insert_slice_op bufferizes out-of-place due to a RAW conflict
  // on %arg2, so we need an allocation.
  // CHECK: %[[alloc1:.*]] = memref.alloc
  // CHECK: memref.copy %[[arg2]], %[[alloc1]]

  // CHECK: scf.forall (%[[tidx:.*]]) in (%[[idx2]])
  %2 = scf.forall (%arg3) in (%idx2) shared_outs(%o = %arg2) -> (tensor<?xf32>) {
      // CHECK: %[[subview1:.*]] = memref.subview %[[alloc1]][5] [%[[idx]]] [1]
      %6 = tensor.extract_slice %o[5] [%idx] [%c1] : tensor<?xf32> to tensor<?xf32>

      // CHECK: linalg.fill ins(%{{.*}}) outs(%[[subview1]] : memref<?xf32
      %8 = linalg.fill ins(%cst : f32) outs(%6 : tensor<?xf32>) -> tensor<?xf32>

      // Now the copy of the actual insert_slice. (It will fold away.)
      // CHECK: memref.copy %[[subview1]], %[[subview1]]

      // Empty terminator is elided from pretty-printing.
      // CHECK-NOT: scf.forall.in_parallel
      // CHECK-NOT: parallel_insert_slice
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %8 into %o[5] [%idx] [%c1] :
          tensor<?xf32> into tensor<?xf32>
      }
  }

  // CHECK: %[[load:.*]] = memref.load %[[arg2]]
  // CHECK: %[[load2:.*]] = memref.load %[[alloc1]]
  // CHECK: memref.dealloc %[[alloc1]]
  %f = tensor.extract %arg2[%c0] : tensor<?xf32>
  %f2 = tensor.extract %2[%c0] : tensor<?xf32>

  // CHECK: return %[[load2]], %[[load]] : f32, f32
  return %f2, %f : f32, f32
}

// -----

#map0 = affine_map<(d0) -> (d0 * 4)>
#map1 = affine_map<(d0) -> (d0 * 2)>

// CHECK-LABEL: func.func @matmul
func.func @matmul(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>, %arg2: tensor<8x8xf32> {bufferization.writable = true}) -> tensor<8x8xf32> {
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index

  // CHECK: scf.forall {{.*}}
  %0 = scf.forall (%arg3, %arg4) in (%c2, %c4) shared_outs(%o = %arg2) -> (tensor<8x8xf32>) {
    %1 = affine.apply #map0(%arg3)
    %3 = tensor.extract_slice %arg0[%1, 0] [4, 8] [1, 1] : tensor<8x8xf32> to tensor<4x8xf32>
    %4 = affine.apply #map1(%arg4)
    %6 = tensor.extract_slice %arg1[0, %4] [8, 4] [1, 1] : tensor<8x8xf32> to tensor<8x4xf32>
    %7 = tensor.extract_slice %o[%1, %4] [4, 4] [1, 1] : tensor<8x8xf32> to tensor<4x4xf32>

    //      CHECK: linalg.matmul ins({{.*}}memref<4x8xf32, strided<[?, ?], offset: ?>>, memref<8x4xf32, strided<[?, ?], offset: ?>>) outs({{.*}} : memref<4x4xf32, strided<[?, ?], offset: ?>>)
    %8 = linalg.matmul ins(%3, %6 : tensor<4x8xf32>, tensor<8x4xf32>) outs(%7 : tensor<4x4xf32>) -> tensor<4x4xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %8 into %o[%1, %4] [4, 4] [1, 1] : tensor<4x4xf32> into tensor<8x8xf32>
    }
  }
  return %0 : tensor<8x8xf32>
}

// -----

// CHECK-LABEL: func @scf_foreach_private_var(
//  CHECK-SAME:     %[[t:.*]]: memref<10xf32
func.func @scf_foreach_private_var(%t: tensor<10xf32>) -> f32 {
  %c2 = arith.constant 2 : index
  %c5 = arith.constant 5 : index

  // A copy is inserted for the uses of %t in the loop.
  // CHECK: %[[t_copy:.*]] = memref.alloc() {{.*}} : memref<10xf32>
  // CHECK: memref.copy %[[t]], %[[t_copy]]

  // CHECK: scf.forall (%{{.*}}) in (2) {

  // Load from the copy and store into the shared output.
  // CHECK:   %[[subview:.*]] = memref.subview %[[t]]
  // CHECK:   memref.load %[[t_copy]]
  // CHECK:   memref.store %{{.*}}, %[[subview]]
  %0 = scf.forall (%tid) in (%c2) shared_outs(%o = %t) -> tensor<10xf32> {
    %offset = arith.muli %c5, %tid : index
    %slice = tensor.extract_slice %o[%offset] [5] [1]
        : tensor<10xf32> to tensor<5xf32>
    %r2 = tensor.extract %t[%tid] : tensor<10xf32>
    %i = tensor.insert %r2 into %slice[%c2] : tensor<5xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %i into %o[%offset] [5] [1]
          : tensor<5xf32> into tensor<10xf32>
    }
  }

  %r = tensor.extract %0[%c2] : tensor<10xf32>
  return %r : f32
}

// -----

// CHECK-LABEL: func.func @scf_foreach_privatized_but_not_copied(
//  CHECK-SAME:     %[[t0:.*]]: memref<10xf32, {{.*}}>, %[[t1:.*]]: memref<10xf32
func.func @scf_foreach_privatized_but_not_copied(
    %t0: tensor<10xf32>, %t1: tensor<10xf32>) -> f32 {
  %c2 = arith.constant 2 : index
  %c5 = arith.constant 5 : index

  // CHECK-NOT: memref.alloc
  // CHECK-NOT: memref.copy
  // CHECK: scf.forall {{.*}} {
  %0 = scf.forall (%tid) in (%c2) shared_outs(%o = %t0) -> tensor<10xf32> {
    %offset = arith.muli %c5, %tid : index
    %slice = tensor.extract_slice %o[%offset] [5] [1]
        : tensor<10xf32> to tensor<5xf32>

    // %t1 is never written in here, so no copy is needed
    // CHECK: memref.load %[[t1]]
    %r2 = tensor.extract %t1[%tid] : tensor<10xf32>
    %i = tensor.insert %r2 into %slice[%c2] : tensor<5xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %i into %o[%offset] [5] [1]
          : tensor<5xf32> into tensor<10xf32>
    }
  }

  %r = tensor.extract %0[%c2] : tensor<10xf32>
  return %r : f32
}

// -----

// CHECK-LABEL: func @scf_if_memory_space
func.func @scf_if_memory_space(%c: i1, %f: f32, %cst: f32) -> (f32, f32)
{
  %c0 = arith.constant 0 : index
  // CHECK: %[[alloc:.*]] = memref.alloc() {{.*}} : memref<5xf32, 1>
  %alloc = bufferization.alloc_tensor() {memory_space = 1 : i64} : tensor<5xf32>
  // CHECK: linalg.fill {{.*}} outs(%[[alloc]] : memref<5xf32, 1>)
  %filled = linalg.fill ins(%cst : f32) outs(%alloc : tensor<5xf32>) -> tensor<5xf32>
  // CHECK: scf.if %{{.*}} -> (memref<5xf32, 1>) {
  %1 = scf.if %c -> tensor<5xf32> {
    // CHECK: %[[cloned:.*]] = bufferization.clone %[[alloc]]
    // CHECK: scf.yield %[[cloned]]
    scf.yield %filled : tensor<5xf32>
  } else {
    // CHECK: %[[alloc2:.*]] = memref.alloc() {{.*}} : memref<5xf32, 1>
    // CHECK: memref.store %{{.*}}, %[[alloc2]]
    // CHECK: %[[cloned2:.*]] = bufferization.clone %[[alloc2]]
    // CHECK: memref.dealloc %[[alloc2]]
    // CHECK: scf.yield %[[cloned2]]
    %2 = tensor.insert %f into %filled[%c0] : tensor<5xf32>
    scf.yield %2 : tensor<5xf32>
  }
  %r0 = tensor.extract %filled[%c0] : tensor<5xf32>
  %r1 = tensor.extract %1[%c0] : tensor<5xf32>
  return %r0, %r1 : f32, f32
}

// -----

// CHECK-LABEL: func @scf_execute_region_memory_space
// CHECK: memref.alloc() {{.*}} : memref<5xf32, 1>
// CHECK: memref.store
// CHECK: memref.load
// CHECK: memref.dealloc
func.func @scf_execute_region_memory_space(%f: f32) -> f32 {
  %c0 = arith.constant 0 : index
  %0 = scf.execute_region -> tensor<5xf32> {
    %1 = bufferization.alloc_tensor() {memory_space = 1 : i64} : tensor<5xf32>
    %2 = tensor.insert %f into %1[%c0] : tensor<5xf32>
    scf.yield %2 : tensor<5xf32>
  }
  %r = tensor.extract %0[%c0] : tensor<5xf32>
  return %r : f32
}

// -----

// Additional allocs are inserted in the loop body. We just check that all
// allocs have the correct memory space.

// CHECK-LABEL: func @scf_for_swapping_yields_memory_space
func.func @scf_for_swapping_yields_memory_space(
    %sz: index, %C : tensor<4xf32>, %lb : index, %ub : index, %step : index)
  -> (f32, f32)
{
  // CHECK: memref.alloc(%{{.*}}) {{.*}} : memref<?xf32, 1>
  // CHECK: memref.alloc(%{{.*}}) {{.*}} : memref<?xf32, 1>
  %A = bufferization.alloc_tensor(%sz) {memory_space = 1 : i64} : tensor<?xf32>
  %B = bufferization.alloc_tensor(%sz) {memory_space = 1 : i64} : tensor<?xf32>

  // CHECK: scf.for {{.*}} {
  %r0:2 = scf.for %i = %lb to %ub step %step iter_args(%tA = %A, %tB = %B)
      -> (tensor<?xf32>, tensor<?xf32>)
  {
    // CHECK: memref.alloc(%{{.*}}) {{.*}} : memref<?xf32, 1>
    // CHECK: memref.alloc(%{{.*}}) {{.*}} : memref<?xf32, 1>
    %ttA = tensor.insert_slice %C into %tA[0][4][1] : tensor<4xf32> into tensor<?xf32>
    %ttB = tensor.insert_slice %C into %tB[0][4][1] : tensor<4xf32> into tensor<?xf32>
    // Yield tensors in different order.
    scf.yield %ttB, %ttA : tensor<?xf32>, tensor<?xf32>
  }
  // CHECK: }
  %f0 = tensor.extract %r0#0[%step] : tensor<?xf32>
  %f1 = tensor.extract %r0#1[%step] : tensor<?xf32>
  return %f0, %f1: f32, f32
}

// -----

// CHECK-LABEL: func @scf_for_yield_alias_of_non_equivalent(
func.func @scf_for_yield_alias_of_non_equivalent(%sz: index) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 5.0 : f32

  // CHECK: %[[generate:.*]] = memref.alloc
  %0 = tensor.generate %sz {
  ^bb0(%i: index):
    tensor.yield %cst : f32
  } : tensor<?xf32>

  // A copy is inserted because %t is used inside the loop.
  // CHECK: %[[generate_copy:.*]] = memref.alloc
  // CHECK: memref.copy %[[generate]], %[[generate_copy]]
  // CHECK: scf.for
  %r = scf.for %iv = %c0 to %sz step %c1 iter_args(%t = %0) -> tensor<?xf32> {
    %iv_sub = arith.subi %iv, %c1 : index
    // CHECK: memref.subview %[[generate_copy]]
    %ll = tensor.extract_slice %0[%iv_sub][%sz][1] : tensor<?xf32> to tensor<?xf32>
    %l = tensor.extract %ll[%c0] : tensor<?xf32>
    %double = arith.mulf %cst, %l : f32
    // CHECK: memref.store %{{.*}}, %[[generate]]
    %s = tensor.insert %double into %t[%iv] : tensor<?xf32>
    scf.yield %s : tensor<?xf32>
  }
  return %r : tensor<?xf32>
}

// -----

// We just check that this example bufferizes to valid IR.

// CHECK-LABEL: func @scf_for_buffer_type_mismatch
func.func @scf_for_buffer_type_mismatch(%sz: index, %sz2: index) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %0 = bufferization.alloc_tensor(%sz) : tensor<?xf32>
  %e2 = tensor.extract_slice %0[1][%sz2][1] : tensor<?xf32> to tensor<?xf32>
  // init_arg and iter_arg have different buffer types. This must be resolved
  // with casts.
  %r = scf.for %iv = %c0 to %c10 step %c1 iter_args(%t = %e2) -> tensor<?xf32> {
    %s = "test.dummy"() : () -> (index)
    %e = tensor.extract_slice %t[1][%s][1] : tensor<?xf32> to tensor<?xf32>
    scf.yield %e : tensor<?xf32>
  }
  %x = tensor.extract %r[%c1] : tensor<?xf32>
  return %x : f32
}

// -----

// We just check that this example bufferizes to valid IR.

// CHECK-LABEL: func @scf_while_buffer_type_mismatch
func.func @scf_while_buffer_type_mismatch(%sz: index, %sz2: index) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %cst = arith.constant 5.5 : f32
  %0 = bufferization.alloc_tensor(%sz) : tensor<?xf32>
  %e2 = tensor.extract_slice %0[1][%sz2][1] : tensor<?xf32> to tensor<?xf32>
  // init_arg and iter_arg have different buffer types. This must be resolved
  // with casts.
  %r = scf.while (%t = %e2) : (tensor<?xf32>) -> (tensor<?xf32>) {
    %c = "test.condition"() : () -> (i1)
    %s = "test.dummy"() : () -> (index)
    %e = tensor.extract_slice %t[1][%s][1] : tensor<?xf32> to tensor<?xf32>
    scf.condition(%c) %e : tensor<?xf32>
  } do {
  ^bb0(%b0: tensor<?xf32>):
    %s2 = "test.dummy"() : () -> (index)
    %n = tensor.insert %cst into %b0[%s2] : tensor<?xf32>
    scf.yield %n : tensor<?xf32>
  }
  %x = tensor.extract %r[%c1] : tensor<?xf32>
  return %x : f32
}

// -----

// CHECK-LABEL: func @non_tensor_for_arg
func.func @non_tensor_for_arg(%A : tensor<?xf32> {bufferization.writable = true})
    -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2.0 : f32
  %c10 = arith.constant 10 : index
  %r1:2 = scf.for %i = %c0 to %c10 step %c1 iter_args(%idx = %c1, %t = %A) -> (index, tensor<?xf32>) {
    %t2 = tensor.insert %c2 into %t[%idx] : tensor<?xf32>
    scf.yield %idx, %t2 : index, tensor<?xf32>
  }
  return %r1#1 : tensor<?xf32>
}

// -----

// This is a regression test. Just check that the IR bufferizes.

// CHECK-LABEL: func @buffer_type_of_collapse_shape
func.func @buffer_type_of_collapse_shape(%arg0: tensor<f64>) {
  %true = arith.constant true
  %0 = scf.while (%arg1 = %arg0) : (tensor<f64>) -> (tensor<f64>) {
    scf.condition(%true) %arg1 : tensor<f64>
  } do {
  ^bb0(%_: tensor<f64>):
    %3 = bufferization.alloc_tensor() : tensor<1xf64>
    %16 = tensor.collapse_shape %3 [] : tensor<1xf64> into tensor<f64>
    scf.yield %16 : tensor<f64>
  }
  return
}

// -----

// This is a regression test. Just check that the IR bufferizes.

// CHECK-LABEL: func @non_block_argument_yield
func.func @non_block_argument_yield() {
  %true = arith.constant true
  %0 = bufferization.alloc_tensor() : tensor<i32>
  %1 = scf.while (%arg0 = %0) : (tensor<i32>) -> (tensor<i32>) {
    scf.condition(%true) %arg0 : tensor<i32>
  } do {
  ^bb0(%arg0: tensor<i32>):
    %ret = scf.while (%arg1 = %0) : (tensor<i32>) -> (tensor<i32>) {
      scf.condition(%true) %arg1 : tensor<i32>
    } do {
    ^bb0(%arg7: tensor<i32>):
      scf.yield %0 : tensor<i32>
    }
    scf.yield %ret : tensor<i32>
  }
  return
}

// -----

// This is a regression test. Make sure that bufferization succeeds.

// CHECK-LABEL: func @regression_cast_in_loop(
func.func @regression_cast_in_loop() -> tensor<2xindex> {
  %false = arith.constant false
  %c0 = arith.constant 0 : index
  %0 = bufferization.alloc_tensor() : tensor<2xindex>
  // CHECK: scf.while (%{{.*}} = %{{.*}}) : (memref<2xindex>) -> memref<2xindex>
  %1 = scf.while (%arg0 = %0) : (tensor<2xindex>) -> tensor<2xindex> {
    scf.condition(%false) %arg0 : tensor<2xindex>
  } do {
  // CHECK: ^bb0(%{{.*}}: memref<2xindex>):
  ^bb0(%arg0: tensor<2xindex>):
    %cast = tensor.cast %0 : tensor<2xindex> to tensor<?xindex>
    %inserted = tensor.insert %c0 into %cast[%c0] : tensor<?xindex>
    %cast_0 = tensor.cast %inserted : tensor<?xindex> to tensor<2xindex>
    scf.yield %cast_0 : tensor<2xindex>
  }
  return %1 : tensor<2xindex>
}

// -----

// This test does not compute anything meaningful but it tests that
// bufferizesToMemoryWrite is correctly propagated through regions.

// CHECK-NO-DEALLOC-PASS-LABEL: func @elide_copy_of_non_writing_scf_if(
func.func @elide_copy_of_non_writing_scf_if(%c: i1, %p1: index, %p2: index, %f: f32)
  -> (tensor<10xf32>, f32)
{
  %r = scf.if %c -> tensor<10xf32> {
    // CHECK-NO-DEALLOC-PASS: memref.alloc
    %t1 = bufferization.alloc_tensor() : tensor<10xf32>
    scf.yield %t1 : tensor<10xf32>
  } else {
    // CHECK-NO-DEALLOC-PASS: memref.alloc
    %t2 = bufferization.alloc_tensor() : tensor<10xf32>
    scf.yield %t2 : tensor<10xf32>
  }

  // No copy should be inserted because %r does not bufferize to a memory write.
  // I.e., %r does not have defined contents and the copy can be elided.
  // CHECK-NO-DEALLOC-PASS-NOT: memref.alloc
  // CHECK-NO-DEALLOC-PASS-NOT: memref.copy
  %r2 = tensor.insert %f into %r[%p1] : tensor<10xf32>
  %r3 = tensor.extract %r[%p2] : tensor<10xf32>
  return %r2, %r3 : tensor<10xf32>, f32
}
