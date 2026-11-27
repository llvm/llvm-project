// RUN: mlir-opt %s -test-tensor-copy-insertion=allow-return-allocs-from-loops -allow-unregistered-dialect -split-input-file | FileCheck %s
// RUN: mlir-opt %s -test-tensor-copy-insertion="allow-return-allocs-from-loops bufferize-function-boundaries" -split-input-file | FileCheck %s --check-prefix=CHECK-FUNC

// CHECK-LABEL: func @scf_for(
//  CHECK-SAME:     %[[A:.*]]: tensor<?xf32>, %[[B:.*]]: tensor<?xf32>
func.func @scf_for(%A : tensor<?xf32>, %B : tensor<?xf32>,
                   %lb : index, %ub : index, %step : index)
  -> (tensor<?xf32>, tensor<?xf32>)
{
  // CHECK: %[[A_copy:.*]] = bufferization.alloc_tensor() copy(%[[A]]) : tensor<?xf32>
  // CHECK: %[[B_copy:.*]] = bufferization.alloc_tensor() copy(%[[B]]) : tensor<?xf32>
  // CHECK:   %[[for:.*]]:2 = scf.for {{.*}} iter_args(%[[iter1:.*]] = %[[A_copy]], %[[iter2:.*]] = %[[B_copy]])
  %r0:2 = scf.for %i = %lb to %ub step %step iter_args(%tA = %A, %tB = %B)
      -> (tensor<?xf32>, tensor<?xf32>)
  {
    // CHECK: scf.yield %[[iter1]], %[[iter2]]
    scf.yield %tA, %tB : tensor<?xf32>, tensor<?xf32>
  }

  return %r0#0, %r0#1 : tensor<?xf32>, tensor<?xf32>
}

// -----

// CHECK-LABEL: func @scf_for_swapping_yields(
//  CHECK-SAME:     %[[A:.*]]: tensor<?xf32>, %[[B:.*]]: tensor<?xf32>
func.func @scf_for_swapping_yields(%A : tensor<?xf32>, %B : tensor<?xf32>,
                                   %lb : index, %ub : index, %step : index)
  -> (tensor<?xf32>, tensor<?xf32>)
{
  // CHECK: %[[A_copy:.*]] = bufferization.alloc_tensor() copy(%[[A]]) : tensor<?xf32>
  // CHECK: %[[B_copy:.*]] = bufferization.alloc_tensor() copy(%[[B]]) : tensor<?xf32>
  // CHECK:   %[[for:.*]]:2 = scf.for {{.*}} iter_args(%[[iter1:.*]] = %[[A_copy]], %[[iter2:.*]] = %[[B_copy]])
  %r0:2 = scf.for %i = %lb to %ub step %step iter_args(%tA = %A, %tB = %B)
      -> (tensor<?xf32>, tensor<?xf32>)
  {
    // Yield tensors in different order.
    // CHECK-DAG: %[[yield1:.*]] = bufferization.alloc_tensor() copy(%[[iter2]]) : tensor<?xf32>
    // CHECK-DAG: %[[yield2:.*]] = bufferization.alloc_tensor() copy(%[[iter1]]) : tensor<?xf32>
    // CHECK: scf.yield %[[yield1]], %[[yield2]]
    scf.yield %tB, %tA : tensor<?xf32>, tensor<?xf32>
  }

  return %r0#0, %r0#1 : tensor<?xf32>, tensor<?xf32>
}

// -----

// CHECK-LABEL: func @scf_while(
//  CHECK-SAME:     %[[A:.*]]: tensor<5xi1>, %[[B:.*]]: tensor<5xi1>
func.func @scf_while(%A: tensor<5xi1>, %B: tensor<5xi1>, %idx: index)
  -> (tensor<5xi1>, tensor<5xi1>)
{
  // CHECK: %[[A_copy:.*]] = bufferization.alloc_tensor() copy(%[[A]]) : tensor<5xi1>
  // CHECK: %[[B_copy:.*]] = bufferization.alloc_tensor() copy(%[[B]]) : tensor<5xi1>
  // CHECK: %[[loop:.*]]:2 = scf.while (%[[w0:.*]] = %[[A_copy]], %[[w1:.*]] = %[[B_copy]]) {{.*}} {
  %r0, %r1 = scf.while (%w0 = %A, %w1 = %B)
      : (tensor<5xi1>, tensor<5xi1>) -> (tensor<5xi1>, tensor<5xi1>) {
    // CHECK: %[[condition:.*]] = tensor.extract %[[w0]]
    %condition = tensor.extract %w0[%idx] : tensor<5xi1>
    // Yield tensors in different order.
    // CHECK: scf.condition(%[[condition]]) %[[w0]], %[[w1]]
    scf.condition(%condition) %w0, %w1 : tensor<5xi1>, tensor<5xi1>
  } do {
  ^bb0(%b0: tensor<5xi1>, %b1: tensor<5xi1>):
    // CHECK: } do {
    // CHECK: ^bb0(%[[b0:.*]]: tensor<5xi1>, %[[b1:.*]]: tensor<5xi1>):
    // CHECK: scf.yield %[[b0]], %[[b1]]
    // CHECK: }
    scf.yield %b0, %b1 : tensor<5xi1>, tensor<5xi1>
  }

  return %r0, %r1 : tensor<5xi1>, tensor<5xi1>
}

// -----

// CHECK-LABEL: func @scf_while_non_equiv_condition_and_body(
//  CHECK-SAME:     %[[A:.*]]: tensor<5xi1>, %[[B:.*]]: tensor<5xi1>
func.func @scf_while_non_equiv_condition_and_body(%A: tensor<5xi1>,
                                                  %B: tensor<5xi1>,
                                                  %idx: index)
  -> (tensor<5xi1>, tensor<5xi1>)
{
  // CHECK: %[[A_copy:.*]] = bufferization.alloc_tensor() copy(%[[A]]) : tensor<5xi1>
  // CHECK: %[[B_copy:.*]] = bufferization.alloc_tensor() copy(%[[B]]) : tensor<5xi1>
  // CHECK: %[[loop:.*]]:2 = scf.while (%[[w0:.*]] = %[[A_copy]], %[[w1:.*]] = %[[B_copy]]) {{.*}} {
  %r0, %r1 = scf.while (%w0 = %A, %w1 = %B)
      : (tensor<5xi1>, tensor<5xi1>) -> (tensor<5xi1>, tensor<5xi1>) {
    // CHECK: %[[condition:.*]] = tensor.extract %[[w0]]
    %condition = tensor.extract %w0[%idx] : tensor<5xi1>
    // Yield tensors in different order.
    // CHECK-DAG: %[[yield0:.*]] = bufferization.alloc_tensor() copy(%[[w1]]) : tensor<5xi1>
    // CHECK-DAG: %[[yield1:.*]] = bufferization.alloc_tensor() copy(%[[w0]]) : tensor<5xi1>
    // CHECK: scf.condition(%[[condition]]) %[[yield0]], %[[yield1]]
    scf.condition(%condition) %w1, %w0 : tensor<5xi1>, tensor<5xi1>
  } do {
  ^bb0(%b0: tensor<5xi1>, %b1: tensor<5xi1>):
    // CHECK: } do {
    // CHECK: ^bb0(%[[b0:.*]]: tensor<5xi1>, %[[b1:.*]]: tensor<5xi1>):
    // CHECK: scf.yield %[[b1]], %[[b0]]
    // CHECK: }
    scf.yield %b1, %b0 : tensor<5xi1>, tensor<5xi1>
  }

  return %r0, %r1 : tensor<5xi1>, tensor<5xi1>
}

// -----

// The "after" region yields tensors that are defined outside of the loop: an
// allocation and a constant. Both must be copied: They are passed to the
// "before" region in the next iteration, which may write to them in-place.

// CHECK-LABEL: func @scf_while_yield_external(
func.func @scf_while_yield_external() -> (tensor<4xf32>, tensor<4xf32>) {
  %true = arith.constant true
  %c0 = arith.constant 0 : index
  %c100 = arith.constant 100.0 : f32
  // CHECK: %[[cst:.*]] = arith.constant dense<7.000000e+00> : tensor<4xf32>
  %cst = arith.constant dense<7.0> : tensor<4xf32>
  %A = bufferization.alloc_tensor() : tensor<4xf32>
  %B = bufferization.alloc_tensor() : tensor<4xf32>
  // CHECK: %[[ext:.*]] = bufferization.alloc_tensor() copy(%[[cst]]) : tensor<4xf32>
  %ext = bufferization.alloc_tensor() copy(%cst) : tensor<4xf32>
  // CHECK: scf.while
  %r:2 = scf.while (%w0 = %A, %w1 = %B) : (tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
    %w0w = tensor.insert %c100 into %w0[%c0] : tensor<4xf32>
    scf.condition(%true) %w0w, %w1 : tensor<4xf32>, tensor<4xf32>
  } do {
  ^bb0(%b0: tensor<4xf32>, %b1: tensor<4xf32>):
    // CHECK: %[[ext_copy:.*]] = bufferization.alloc_tensor() copy(%[[ext]]) : tensor<4xf32>
    // CHECK: %[[cst_copy:.*]] = bufferization.alloc_tensor() copy(%[[cst]]) : tensor<4xf32>
    // CHECK: scf.yield %[[ext_copy]], %[[cst_copy]]
    scf.yield %ext, %cst : tensor<4xf32>, tensor<4xf32>
  }
  return %r#0, %r#1 : tensor<4xf32>, tensor<4xf32>
}

// -----

// The "after" region yields a tensor that is allocated inside the loop. It does
// not alias any value defined outside of the loop, so it can be yielded without
// a copy (allow-return-allocs-from-loops).

// CHECK-LABEL: func @scf_while_yield_alloc(
func.func @scf_while_yield_alloc() -> tensor<4xf32> {
  %true = arith.constant true
  %A = bufferization.alloc_tensor() : tensor<4xf32>
  %r = scf.while (%w = %A) : (tensor<4xf32>) -> tensor<4xf32> {
    // CHECK: scf.condition
    scf.condition(%true) %w : tensor<4xf32>
  } do {
  ^bb0(%b: tensor<4xf32>):
    // CHECK: %[[alloc:.*]] = bufferization.alloc_tensor() : tensor<4xf32>
    %new = bufferization.alloc_tensor() : tensor<4xf32>
    // CHECK: scf.yield %[[alloc]]
    scf.yield %new : tensor<4xf32>
  }
  return %r : tensor<4xf32>
}

// -----

// Two slots yield the same loop-carried tensor. One of them must be copied so
// that the two loop-carried values do not alias in the next iteration.

// CHECK-LABEL: func @scf_while_yield_same_value(
func.func @scf_while_yield_same_value() -> (tensor<4xf32>, tensor<4xf32>) {
  %true = arith.constant true
  %A = bufferization.alloc_tensor() : tensor<4xf32>
  %B = bufferization.alloc_tensor() : tensor<4xf32>
  %r:2 = scf.while (%w0 = %A, %w1 = %B) : (tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
    // CHECK: scf.condition
    scf.condition(%true) %w0, %w1 : tensor<4xf32>, tensor<4xf32>
  } do {
  ^bb0(%b0: tensor<4xf32>, %b1: tensor<4xf32>):
    // CHECK: } do {
    // CHECK: ^bb0(%[[b0:.*]]: tensor<4xf32>, %[[b1:.*]]: tensor<4xf32>):
    // CHECK: %[[copy:.*]] = bufferization.alloc_tensor() copy(%[[b1]]) : tensor<4xf32>
    // CHECK: scf.yield %[[b1]], %[[copy]]
    scf.yield %b1, %b1 : tensor<4xf32>, tensor<4xf32>
  }
  return %r#0, %r#1 : tensor<4xf32>, tensor<4xf32>
}

// -----

// CHECK-LABEL: func @scf_forall_out_of_place(
//  CHECK-SAME:     %[[arg0:.*]]: tensor<100xf32>, %[[arg1:.*]]: tensor<100xf32>
// CHECK-FUNC-LABEL: func @scf_forall_out_of_place(
func.func @scf_forall_out_of_place(%in: tensor<100xf32>,
                                   %out: tensor<100xf32>) {
  %c1 = arith.constant 1 : index
  %num_threads = arith.constant 100 : index

  // CHECK-FUNC-NOT: alloc_tensor
  // CHECK: %[[alloc:.*]] = bufferization.alloc_tensor() copy(%[[arg1]]) : tensor<100xf32>
  // CHECK: scf.forall {{.*}} shared_outs(%[[o:.*]] = %[[alloc]])
  %result = scf.forall (%thread_idx) in (%num_threads) shared_outs(%o = %out) -> tensor<100xf32> {
      // CHECK: tensor.extract_slice
      // CHECK: scf.forall.in_parallel
      // CHECK: tensor.parallel_insert_slice %{{.*}} into %[[o]]
      %1 = tensor.extract_slice %in[%thread_idx][1][1] : tensor<100xf32> to tensor<1xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %1 into %o[%thread_idx][1][1] :
          tensor<1xf32> into tensor<100xf32>
      }
  // CHECK: } {mapping = [#gpu.thread<x>]}
  } {mapping = [#gpu.thread<x>]}
  return
}

// -----

// CHECK-LABEL: func @in_order_multiple_parallel_writes
func.func @in_order_multiple_parallel_writes(%2: tensor<320xf32>,
                                             %3: tensor<320xf32>)
  -> (tensor<320xf32>, tensor<320xf32>)
{
  %c0 = arith.constant 0 : index
  %cst = arith.constant -0.000000e+00 : f32
  %c320 = arith.constant 320 : index
  %4:2 = scf.forall (%arg0) in (%c320) shared_outs(%arg1 = %2, %arg2 = %3) -> (tensor<320xf32>, tensor<320xf32>) {
    // CHECK: tensor.extract_slice {{.*}}
    %6 = tensor.extract_slice %arg1[%arg0] [1] [1] : tensor<320xf32> to tensor<1xf32>
    // CHECK: tensor.extract_slice {{.*}}
    %7 = tensor.extract_slice %arg2[%arg0] [1] [1] : tensor<320xf32> to tensor<1xf32>
    // CHECK: linalg.fill {{.*}}
    %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<1xf32>) -> tensor<1xf32>

    // CHECK: tensor.parallel_insert_slice {{.*}}
    // CHECK: tensor.parallel_insert_slice {{.*}}
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %6 into %arg2[%arg0] [1] [1] : tensor<1xf32> into tensor<320xf32>
      tensor.parallel_insert_slice %8 into %arg1[%arg0] [1] [1] : tensor<1xf32> into tensor<320xf32>
    }
  }
  return %4#0, %4#1 : tensor<320xf32>, tensor<320xf32>
}
