// RUN: mlir-opt %s -linalg-comprehensive-func-bufferize=test-analysis-only -split-input-file | FileCheck %s

//===----------------------------------------------------------------------===//
// Simple cases
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @subtensor_fun
func @subtensor_fun(%A : tensor<?xf32>, %B : tensor<?xf32> {linalg.inplaceable = true})
  -> (tensor<4xf32>, tensor<8xf32>)
{
  // subtensor is not used in a write, it is not compelled to bufferize out of
  // place. Let callers decide whether they want to create aliasing subviews at
  // all call sites or whether they allocate.
  // This is true irrespective of whether the function argument is inplaceable.
  //     CHECK: subtensor
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  %r0 = subtensor %A[0][4][1] : tensor<?xf32> to tensor<4xf32>

  //     CHECK: subtensor
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  %r1 = subtensor %B[0][8][1] : tensor<?xf32> to tensor<8xf32>

  return %r0, %r1: tensor<4xf32>, tensor<8xf32>
}

// -----

// CHECK-LABEL: func @subtensor_insert_fun
func @subtensor_insert_fun(
    %A : tensor<?xf32>,
    %B : tensor<?xf32> {linalg.inplaceable = true},
    %C : tensor<4xf32>)
  -> (tensor<?xf32>, tensor<?xf32>)
{
  // must bufferize out of place.
  //     CHECK: subtensor_insert
  // CHECK-SAME: {__inplace_results_attr__ = ["false"]}
  %r0 = subtensor_insert %C into %A[0][4][1] : tensor<4xf32> into tensor<?xf32>

  // bufferizes inplace.
  //     CHECK: subtensor_insert
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  %r1 = subtensor_insert %C into %B[0][4][1] : tensor<4xf32> into tensor<?xf32>

  return %r0, %r1: tensor<?xf32>, tensor<?xf32>
}

// -----

// CHECK-LABEL: func @conflict_on_B
func @conflict_on_B(
    %A : tensor<4x4xf32> {linalg.inplaceable = true},
    %B : tensor<4x4xf32> {linalg.inplaceable = true})
  -> (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>)
{
  // matmul output operand interferes with input operand.
  //     CHECK: linalg.matmul
  // CHECK-SAME: {__inplace_results_attr__ = ["false"]}
  %C = linalg.matmul  ins(%A, %B: tensor<4x4xf32>, tensor<4x4xf32>)
                     outs(%B: tensor<4x4xf32>)
    -> tensor<4x4xf32>

  // matmul output operand interferes with input operand.
  //     CHECK: linalg.matmul
  // CHECK-SAME: {__inplace_results_attr__ = ["false"]}
  %D = linalg.matmul  ins(%B, %A: tensor<4x4xf32>, tensor<4x4xf32>)
                     outs(%B: tensor<4x4xf32>)
    -> tensor<4x4xf32>

  // matmul output operand does not interferes with input operand.
  //     CHECK: linalg.matmul
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  %E = linalg.matmul  ins(%A, %A: tensor<4x4xf32>, tensor<4x4xf32>)
                     outs(%B: tensor<4x4xf32>)
    -> tensor<4x4xf32>

  return %C, %D, %E: tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>
}

//===----------------------------------------------------------------------===//
// Length-1 producer-consumer cases.
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @subtensor_subtensor
func @subtensor_subtensor(
    %A : tensor<?xf32> {linalg.inplaceable = true}, %B : tensor<?xf32>)
  -> (tensor<2xf32>, tensor<2xf32>)
{
  // subtensor is not used in a write, it is not compelled to bufferize out of
  // place. Let callers decide whether they want to create aliasing subviews at
  // all call sites or whether they allocate.
  // This is true irrespective of whether the function argument is inplaceable.
  // CHECK: {__inplace_results_attr__ = ["true"]}
  %r0 = subtensor %A[0][4][1] : tensor<?xf32> to tensor<4xf32>

  // CHECK: {__inplace_results_attr__ = ["true"]}
  %r1 = subtensor %r0[0][2][1] : tensor<4xf32> to tensor<2xf32>

  // CHECK: {__inplace_results_attr__ = ["true"]}
  %r2 = subtensor %B[0][4][1] : tensor<?xf32> to tensor<4xf32>

  // CHECK: {__inplace_results_attr__ = ["true"]}
  %r3 = subtensor %r2[0][2][1] : tensor<4xf32> to tensor<2xf32>

  return %r1, %r3: tensor<2xf32>, tensor<2xf32>
}

// -----

// CHECK-LABEL: func @subtensor_insert_subtensor_insert
func @subtensor_insert_subtensor_insert(
    %A : tensor<?xf32> {linalg.inplaceable = true},
    %A2 : tensor<4xf32> {linalg.inplaceable = true},
    %A3 : tensor<2xf32> {linalg.inplaceable = true},
    %B : tensor<?xf32>, %B2 : tensor<4xf32>, %B3 : tensor<2xf32>)
  -> (tensor<?xf32>, tensor<?xf32>)
{
  // CHECK: {__inplace_results_attr__ = ["true"]}
  %r0 = subtensor_insert %A3 into %A2[0][2][1] : tensor<2xf32> into tensor<4xf32>

  // CHECK: {__inplace_results_attr__ = ["true"]}
  %r1 = subtensor_insert %r0 into %A[0][4][1] : tensor<4xf32> into tensor<?xf32>

  // CHECK: {__inplace_results_attr__ = ["false"]}
  %r2 = subtensor_insert %B3 into %B2[0][2][1] : tensor<2xf32> into tensor<4xf32>

  // CHECK: {__inplace_results_attr__ = ["false"]}
  %r3 = subtensor_insert %r2 into %B[0][4][1] : tensor<4xf32> into tensor<?xf32>

  return %r1, %r3: tensor<?xf32>, tensor<?xf32>
}

// -----

// CHECK-LABEL: func @subtensor_nonmatching_subtensor_insert
func @subtensor_nonmatching_subtensor_insert(
    %A : tensor<?xf32> {linalg.inplaceable = true},
    %B : tensor<?xf32>, %idx: index)
  -> (tensor<?xf32>, tensor<?xf32>)
{
  // %r1 bufferizes inplace because %A is inplaceable.
  // %r0 is an overlapping subtensor that does not match, it must be out of place.
  //      CHECK: subtensor
  // CHECK-SAME: {__inplace_results_attr__ = ["false"]}
  %r0 = subtensor %A[0][4][1] : tensor<?xf32> to tensor<4xf32>

  // %r1 can bufferize inplace fine.
  //      CHECK: subtensor_insert
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  %r1 = subtensor_insert %r0 into %A[%idx][4][1] : tensor<4xf32> into tensor<?xf32>

  // %r3 does bufferizes inplace because %B is not inplaceable.
  // %r0 is an overlapping subtensor that does not match, but does not alias with
  // the buffer coming from %r3 so it can actually bufferize inplace.
  //      CHECK: subtensor
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  %r2 = subtensor %B[0][4][1] : tensor<?xf32> to tensor<4xf32>

  // %r3 cannot bufferize inplace since %B is not inplaceable.
  //      CHECK: subtensor_insert
  // CHECK-SAME: {__inplace_results_attr__ = ["false"]}
  %r3 = subtensor_insert %r2 into %B[%idx][4][1] : tensor<4xf32> into tensor<?xf32>

  return %r1, %r3: tensor<?xf32>, tensor<?xf32>
}

// -----

// CHECK-LABEL: func @subtensor_matching_subtensor_insert
func @subtensor_matching_subtensor_insert(
    %A : tensor<?xf32> {linalg.inplaceable = true},
    %B : tensor<?xf32>)
  -> (tensor<?xf32>, tensor<?xf32>)
{
  // %r1 bufferizes inplace because %A is inplaceable.
  // %r0 is a subtensor that matches, it can also be bufferized inplace.
  //      CHECK: subtensor
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  %r0 = subtensor %A[0][4][1] : tensor<?xf32> to tensor<4xf32>

  //      CHECK: subtensor_insert
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  %r1 = subtensor_insert %r0 into %A[0][4][1] : tensor<4xf32> into tensor<?xf32>

  // %r2 is a subtensor that matches %r3, it can be bufferized inplace.
  //      CHECK: subtensor
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  %r2 = subtensor %B[0][4][1] : tensor<?xf32> to tensor<4xf32>

  // subtensor_insert cannot bufferize inplace.
  // This should have been captured by a canonicalization pattern and it would
  // be unproductive to have special logic in bufferization to encode matching
  // subtensor_insert(subtensor(A), A).
  //      CHECK: subtensor_insert
  // CHECK-SAME: {__inplace_results_attr__ = ["false"]}
  %r3 = subtensor_insert %r2 into %B[0][4][1] : tensor<4xf32> into tensor<?xf32>

  return %r1, %r3: tensor<?xf32>, tensor<?xf32>
}

// -----

// CHECK-LABEL: func @subtensor_linalg_readonly_use
func @subtensor_linalg_readonly_use(
    %A : tensor<?x?xf32>,
    %B : tensor<4x4xf32>,
    %C : tensor<4x4xf32> {linalg.inplaceable = true})
  ->  (tensor<4x4xf32>, tensor<4x4xf32>)
{
  // subtensor is only used as a read, no interference irrespective of user's
  // inplace status.
  //     CHECK: subtensor
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  %sA = subtensor %A[0, 0][4, 4][1, 1] : tensor<?x?xf32> to tensor<4x4xf32>

  // matmul output operand is not inplaceable at the function boundary.
  //     CHECK: linalg.matmul
  // CHECK-SAME: {__inplace_results_attr__ = ["false"]}
  %D = linalg.matmul  ins(%sA, %B: tensor<4x4xf32>, tensor<4x4xf32>)
                     outs(%B: tensor<4x4xf32>)
    -> tensor<4x4xf32>

  // matmul output operand is inplaceable at the function boundary.
  //     CHECK: linalg.matmul
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  %E = linalg.matmul  ins(%sA, %B: tensor<4x4xf32>, tensor<4x4xf32>)
                     outs(%C: tensor<4x4xf32>)
    -> tensor<4x4xf32>

  return %D, %E: tensor<4x4xf32>, tensor<4x4xf32>
}

// -----

// CHECK-LABEL: func @subtensor_to_linalg_write_use
func @subtensor_to_linalg_write_use(
    %A : tensor<4x4xf32>,
    %B : tensor<?x?xf32>,
    %C : tensor<?x?xf32> {linalg.inplaceable = true})
  ->  (tensor<4x4xf32>, tensor<4x4xf32>)
{
  // Step 3. %sB forward propagates to a write in %D but it is not inplace.
  // So this is only ever read and can bufferize inplace.
  //     CHECK: subtensor
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  %sB = subtensor %B[0, 0][4, 4][1, 1] : tensor<?x?xf32> to tensor<4x4xf32>

  // Step 2. %sB has a read interference in %E, it does not bufferize inplace.
  //     CHECK: linalg.matmul
  // CHECK-SAME: {__inplace_results_attr__ = ["false"]}
  %D = linalg.matmul  ins(%B, %C: tensor<?x?xf32>, tensor<?x?xf32>)
                     outs(%sB: tensor<4x4xf32>)
    -> tensor<4x4xf32>

  // Step 4. %sC forward propagates to an inplace write in %E.
  // %sC backward propagates to %C which is inplaceable.
  // As a consequence this is bufferized inplace.
  //     CHECK: subtensor
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  %sC = subtensor %C[0, 0][4, 4][1, 1] : tensor<?x?xf32> to tensor<4x4xf32>

  // Step 1. %sC backprops to the subtensor producer which is not considered an
  // interference. This bufferizes inplace.
  //     CHECK: linalg.matmul
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  %E = linalg.matmul  ins(%A, %sB: tensor<4x4xf32>, tensor<4x4xf32>)
                     outs(%sC: tensor<4x4xf32>)
    -> tensor<4x4xf32>

  return %D, %E: tensor<4x4xf32>, tensor<4x4xf32>
}

//===----------------------------------------------------------------------===//
// Transitive cases
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @subtensor_to_linalg_write_use
func @subtensor_to_linalg_write_use(
    %A : tensor<4x4xf32>,
    %B : tensor<?x?xf32>,
    %C : tensor<?x?xf32> {linalg.inplaceable = true})
  ->  (tensor<4x4xf32>, tensor<4x4xf32>)
{
  // Step 4. %sB forward propagates to an inplace write in %D.
  // %sB backward propagates to %B which is not inplaceable.
  // As a consequence this is bufferized out of place.
  //     CHECK: subtensor
  // CHECK-SAME: {__inplace_results_attr__ = ["false"]}
  %sB = subtensor %B[0, 0][4, 4][1, 1] : tensor<?x?xf32> to tensor<4x4xf32>

  // Step 1. %sB backprops to the subtensor producer which is not considered an
  // interference. This bufferizes inplace.
  //     CHECK: linalg.matmul
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  %D = linalg.matmul  ins(%B, %C: tensor<?x?xf32>, tensor<?x?xf32>)
                     outs(%sB: tensor<4x4xf32>)
    -> tensor<4x4xf32>

  // Step 3. %sC forward propagates to an inplace write in %E.
  // %sC backward propagates to %C which is inplaceable.
  // As a consequence this is bufferized inplace.
  //     CHECK: subtensor
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  %sC = subtensor %C[0, 0][4, 4][1, 1] : tensor<?x?xf32> to tensor<4x4xf32>

  // Step 1. %sC backprops to the subtensor producer which is not considered an
  // interference. This bufferizes inplace.
  //     CHECK: linalg.matmul
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  %E = linalg.matmul  ins(%A, %A: tensor<4x4xf32>, tensor<4x4xf32>)
                     outs(%sC: tensor<4x4xf32>)
    -> tensor<4x4xf32>

  return %D, %E: tensor<4x4xf32>, tensor<4x4xf32>
}

// -----

// CHECK-LABEL: func @nested_subtensor_and_insert
func @nested_subtensor_and_insert(
    %A : tensor<?x?xf32>,
    %B : tensor<?x?xf32> {linalg.inplaceable = true},
    %C : tensor<?x?xf32> {linalg.inplaceable = true},
    %idx : index)
  ->  (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>)
{
  %f0 = constant 0.0 : f32

  // 2-level matching subtensor / subtensor_insert into non inplaceable %A.
  //   - %rA is not inplaceable because %A is not inplaceable at function boundary.
  //   - once %rA is deemed not inplaceable, nothing prevent %rsA to be inplaceable
  //   - this propagates to %FA and %ssA being inplaceable.
  //   - %sA would then bufferize to an inplace write (i.e. %FA) but %A is not
  //     inplaceable and so %sA is not inplaceable.
  //     CHECK: subtensor
  // CHECK-SAME: {__inplace_results_attr__ = ["false"]}
  // CHECK-NEXT: subtensor
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  // CHECK-NEXT: fill
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  // CHECK-NEXT: subtensor_insert
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  // CHECK-NEXT: subtensor_insert
  // CHECK-SAME: {__inplace_results_attr__ = ["false"]}
  %sA = subtensor %A[0, 0][%idx, %idx][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
  %ssA = subtensor %sA[0, 0][4, 4][1, 1] : tensor<?x?xf32> to tensor<4x4xf32>
  %FA = linalg.fill(%ssA, %f0) : tensor<4x4xf32>, f32 -> tensor<4x4xf32>
  %rsA = subtensor_insert %FA into %sA[0, 0][4, 4][1, 1] : tensor<4x4xf32> into tensor<?x?xf32>
  %rA = subtensor_insert %rsA into %A[0, 0][%idx, %idx][1, 1] : tensor<?x?xf32> into tensor<?x?xf32>

  // 3-level matching subtensor / subtensor_insert into inplaceable %B.
  // CHECK-NEXT: subtensor
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  // CHECK-NEXT: subtensor
  // Atm, this 2nd subtensor fails to bufferize inplace because clobbering
  // analysis conservatively test for equivalent buffers.
  // TODO: This is currently too restrictive and misses clobberings.
  // When available, use container-containee analysis.
  // CHECK-SAME: {__inplace_results_attr__ = ["false"]}
  // CHECK-NEXT: subtensor
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  // CHECK-NEXT: fill
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  // CHECK-NEXT: subtensor_insert
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  // CHECK-NEXT: subtensor_insert
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  // CHECK-NEXT: subtensor_insert
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  %sB = subtensor %B[0, 0][%idx, %idx][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
  %ssB = subtensor %sB[0, 0][4, %idx][1, 1] : tensor<?x?xf32> to tensor<4x?xf32>
  %sssB = subtensor %ssB[0, 0][4, 4][1, 1] : tensor<4x?xf32> to tensor<4x4xf32>
  %FB = linalg.fill(%sssB, %f0) : tensor<4x4xf32>, f32 -> tensor<4x4xf32>
  %rssB = subtensor_insert %FB into %ssB[0, 0][4, 4][1, 1] : tensor<4x4xf32> into tensor<4x?xf32>
  %rsB = subtensor_insert %rssB into %sB[0, 0][4, %idx][1, 1] : tensor<4x?xf32> into tensor<?x?xf32>
  %rB = subtensor_insert %rsB into %B[0, 0][%idx, %idx][1, 1] : tensor<?x?xf32> into tensor<?x?xf32>

  // 2-level matching subtensor / subtensor_insert into inplaceable %C with a twist.
  // Throw a wrench in the system: %rsC production sizes do not match %ssC.
  // CHECK-NEXT: subtensor
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  // The subtensor_insert that would be candidate for matching does not actually
  // match. That subtensor_insert can still be bufferized inplace nonetheless
  // but this subtensor, which bufferizes to an inplace write, cannot.
  // CHECK-NEXT: subtensor
  // CHECK-SAME: {__inplace_results_attr__ = ["false"]}
  // CHECK-NEXT: fill
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  // CHECK-NEXT: subtensor_insert
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  // CHECK-NEXT: subtensor_insert
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  %sC = subtensor %C[0, 0][%idx, %idx][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
  %ssC = subtensor %sC[0, 0][4, 4][1, 1] : tensor<?x?xf32> to tensor<4x4xf32>
  %FC = linalg.fill(%ssC, %f0) : tensor<4x4xf32>, f32 -> tensor<4x4xf32>
  %rsC = subtensor_insert %FC into %sC[0, 0][12345, 67890][1, 1] : tensor<4x4xf32> into tensor<?x?xf32>
  %rC = subtensor_insert %rsC into %C[0, 0][%idx, %idx][1, 1] : tensor<?x?xf32> into tensor<?x?xf32>

  return %rA, %rB, %rC: tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>
}

