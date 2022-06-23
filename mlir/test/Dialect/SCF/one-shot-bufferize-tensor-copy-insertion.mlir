// RUN: mlir-opt %s -tensor-copy-insertion="allow-return-allocs" -allow-unregistered-dialect -split-input-file | FileCheck %s
// RUN: mlir-opt %s -tensor-copy-insertion="bufferize-function-boundaries allow-return-allocs" -split-input-file | FileCheck %s --check-prefix=CHECK-FUNC

// CHECK-LABEL: func @scf_for(
//  CHECK-SAME:     %[[A:.*]]: tensor<?xf32>, %[[B:.*]]: tensor<?xf32>
func.func @scf_for(%A : tensor<?xf32>, %B : tensor<?xf32>,
                   %lb : index, %ub : index, %step : index)
  -> (tensor<?xf32>, tensor<?xf32>)
{
  // CHECK: %[[A_copy:.*]] = bufferization.alloc_tensor() copy(%[[A]]) {bufferization.escape = [false]} : tensor<?xf32>
  // CHECK: %[[B_copy:.*]] = bufferization.alloc_tensor() copy(%[[B]]) {bufferization.escape = [false]} : tensor<?xf32>
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
  // CHECK: %[[A_copy:.*]] = bufferization.alloc_tensor() copy(%[[A]]) {bufferization.escape = [false]} : tensor<?xf32>
  // CHECK: %[[B_copy:.*]] = bufferization.alloc_tensor() copy(%[[B]]) {bufferization.escape = [false]} : tensor<?xf32>
  // CHECK:   %[[for:.*]]:2 = scf.for {{.*}} iter_args(%[[iter1:.*]] = %[[A_copy]], %[[iter2:.*]] = %[[B_copy]])
  %r0:2 = scf.for %i = %lb to %ub step %step iter_args(%tA = %A, %tB = %B)
      -> (tensor<?xf32>, tensor<?xf32>)
  {
    // Yield tensors in different order.
    // CHECK-DAG: %[[yield1:.*]] = bufferization.alloc_tensor() copy(%[[iter2]]) {bufferization.escape = [true]} : tensor<?xf32>
    // CHECK-DAG: %[[yield2:.*]] = bufferization.alloc_tensor() copy(%[[iter1]]) {bufferization.escape = [true]} : tensor<?xf32>
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
  // CHECK: %[[A_copy:.*]] = bufferization.alloc_tensor() copy(%[[A]]) {bufferization.escape = [false]} : tensor<5xi1>
  // CHECK: %[[B_copy:.*]] = bufferization.alloc_tensor() copy(%[[B]]) {bufferization.escape = [false]} : tensor<5xi1>
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
  // CHECK: %[[A_copy:.*]] = bufferization.alloc_tensor() copy(%[[A]]) {bufferization.escape = [false]} : tensor<5xi1>
  // CHECK: %[[B_copy:.*]] = bufferization.alloc_tensor() copy(%[[B]]) {bufferization.escape = [false]} : tensor<5xi1>
  // CHECK: %[[loop:.*]]:2 = scf.while (%[[w0:.*]] = %[[A_copy]], %[[w1:.*]] = %[[B_copy]]) {{.*}} {
  %r0, %r1 = scf.while (%w0 = %A, %w1 = %B)
      : (tensor<5xi1>, tensor<5xi1>) -> (tensor<5xi1>, tensor<5xi1>) {
    // CHECK: %[[condition:.*]] = tensor.extract %[[w0]]
    %condition = tensor.extract %w0[%idx] : tensor<5xi1>
    // Yield tensors in different order.
    // CHECK-DAG: %[[yield0:.*]] = bufferization.alloc_tensor() copy(%[[w1]]) {bufferization.escape = [true]} : tensor<5xi1>
    // CHECK-DAG: %[[yield1:.*]] = bufferization.alloc_tensor() copy(%[[w0]]) {bufferization.escape = [true]} : tensor<5xi1>
    // CHECK: scf.condition(%[[condition]]) %[[yield0]], %[[yield1]]
    scf.condition(%condition) %w1, %w0 : tensor<5xi1>, tensor<5xi1>
  } do {
  ^bb0(%b0: tensor<5xi1>, %b1: tensor<5xi1>):
    // CHECK: } do {
    // CHECK: ^bb0(%[[b0:.*]]: tensor<5xi1>, %[[b1:.*]]: tensor<5xi1>):
    // CHECK-DAG: %[[yield2:.*]] = bufferization.alloc_tensor() copy(%[[b1]]) {bufferization.escape = [true]} : tensor<5xi1>
    // CHECK-DAG: %[[yield3:.*]] = bufferization.alloc_tensor() copy(%[[b0]]) {bufferization.escape = [true]} : tensor<5xi1>
    // CHECK: scf.yield %[[yield2]], %[[yield3]]
    // CHECK: }
    scf.yield %b1, %b0 : tensor<5xi1>, tensor<5xi1>
  }

  return %r0, %r1 : tensor<5xi1>, tensor<5xi1>
}

// -----

// CHECK-LABEL: func @scf_foreach_thread_out_of_place(
//  CHECK-SAME:     %[[arg0:.*]]: tensor<100xf32>, %[[arg1:.*]]: tensor<100xf32>
// CHECK-FUNC-LABEL: func @scf_foreach_thread_out_of_place(
func.func @scf_foreach_thread_out_of_place(%in: tensor<100xf32>,
                                           %out: tensor<100xf32>) {
  %c1 = arith.constant 1 : index
  %num_threads = arith.constant 100 : index

  // CHECK-FUNC-NOT: alloc_tensor
  // CHECK: %[[alloc:.*]] = bufferization.alloc_tensor() copy(%[[arg1]]) {bufferization.escape = [false]} : tensor<100xf32>
  // CHECK: scf.foreach_thread
  %result = scf.foreach_thread (%thread_idx) in (%num_threads) -> tensor<100xf32> {
      // CHECK: tensor.extract_slice
      // CHECK: scf.foreach_thread.perform_concurrently
      // CHECK: scf.foreach_thread.parallel_insert_slice %{{.*}} into %[[alloc]]
      %1 = tensor.extract_slice %in[%thread_idx][1][1] : tensor<100xf32> to tensor<1xf32>
      scf.foreach_thread.perform_concurrently {
        scf.foreach_thread.parallel_insert_slice %1 into %out[%thread_idx][1][1] :
          tensor<1xf32> into tensor<100xf32>
      }
  }
  return
}
