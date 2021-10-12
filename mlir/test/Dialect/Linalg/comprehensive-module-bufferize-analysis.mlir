// RUN: mlir-opt %s -linalg-comprehensive-module-bufferize=test-analysis-only -split-input-file | FileCheck %s

//===----------------------------------------------------------------------===//
// Simple cases
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @extract_slice_fun
func @extract_slice_fun(%A : tensor<?xf32>, %B : tensor<?xf32> {linalg.inplaceable = true})
  -> (tensor<4xf32>, tensor<8xf32>)
{
  // tensor.extract_slice is not used in a write, it is not compelled to
  // bufferize out of place. Let callers decide whether they want to create
  // aliasing subviews at all call sites or whether they allocate.
  // This is true irrespective of whether the function argument is inplaceable.
  //     CHECK: tensor.extract_slice
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  %r0 = tensor.extract_slice %A[0][4][1] : tensor<?xf32> to tensor<4xf32>

  //     CHECK: tensor.extract_slice
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  %r1 = tensor.extract_slice %B[0][8][1] : tensor<?xf32> to tensor<8xf32>

  return %r0, %r1: tensor<4xf32>, tensor<8xf32>
}

// -----

// CHECK-LABEL: func @insert_slice_fun
func @insert_slice_fun(
    %A : tensor<?xf32>,
    %B : tensor<?xf32> {linalg.inplaceable = true},
    %C : tensor<4xf32>)
  -> (tensor<?xf32>, tensor<?xf32>)
{
  // must bufferize out of place.
  //     CHECK: tensor.insert_slice
  // CHECK-SAME: {__inplace_results_attr__ = ["false"]}
  %r0 = tensor.insert_slice %C into %A[0][4][1] : tensor<4xf32> into tensor<?xf32>

  // bufferizes inplace.
  //     CHECK: tensor.insert_slice
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  %r1 = tensor.insert_slice %C into %B[0][4][1] : tensor<4xf32> into tensor<?xf32>

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

// CHECK-LABEL: func @extract_slice_extract_slice
func @extract_slice_extract_slice(
    %A : tensor<?xf32> {linalg.inplaceable = true}, %B : tensor<?xf32>)
  -> (tensor<2xf32>, tensor<2xf32>)
{
  // tensor.extract_slice is not used in a write, it is not compelled to
  // bufferize out of place. Let callers decide whether they want to create
  // aliasing subviews at all call sites or whether they allocate.
  // This is true irrespective of whether the function argument is inplaceable.
  // CHECK: {__inplace_results_attr__ = ["true"]}
  %r0 = tensor.extract_slice %A[0][4][1] : tensor<?xf32> to tensor<4xf32>

  // CHECK: {__inplace_results_attr__ = ["true"]}
  %r1 = tensor.extract_slice %r0[0][2][1] : tensor<4xf32> to tensor<2xf32>

  // CHECK: {__inplace_results_attr__ = ["true"]}
  %r2 = tensor.extract_slice %B[0][4][1] : tensor<?xf32> to tensor<4xf32>

  // CHECK: {__inplace_results_attr__ = ["true"]}
  %r3 = tensor.extract_slice %r2[0][2][1] : tensor<4xf32> to tensor<2xf32>

  return %r1, %r3: tensor<2xf32>, tensor<2xf32>
}

// -----

// CHECK-LABEL: func @insert_slice_insert_slice
func @insert_slice_insert_slice(
    %A : tensor<?xf32> {linalg.inplaceable = true},
    %A2 : tensor<4xf32> {linalg.inplaceable = true},
    %A3 : tensor<2xf32> {linalg.inplaceable = true},
    %B : tensor<?xf32>, %B2 : tensor<4xf32>, %B3 : tensor<2xf32>)
  -> (tensor<?xf32>, tensor<?xf32>)
{
  // CHECK: {__inplace_results_attr__ = ["true"]}
  %r0 = tensor.insert_slice %A3 into %A2[0][2][1] : tensor<2xf32> into tensor<4xf32>

  // CHECK: {__inplace_results_attr__ = ["true"]}
  %r1 = tensor.insert_slice %r0 into %A[0][4][1] : tensor<4xf32> into tensor<?xf32>

  // CHECK: {__inplace_results_attr__ = ["false"]}
  %r2 = tensor.insert_slice %B3 into %B2[0][2][1] : tensor<2xf32> into tensor<4xf32>

  // CHECK: {__inplace_results_attr__ = ["false"]}
  %r3 = tensor.insert_slice %r2 into %B[0][4][1] : tensor<4xf32> into tensor<?xf32>

  return %r1, %r3: tensor<?xf32>, tensor<?xf32>
}

// -----

// CHECK-LABEL: func @extract_slice_nonmatching_insert_slice
func @extract_slice_nonmatching_insert_slice(
    %A : tensor<?xf32> {linalg.inplaceable = true},
    %B : tensor<?xf32>, %idx: index)
  -> (tensor<?xf32>, tensor<?xf32>)
{
  // %r1 bufferizes inplace because %A is inplaceable.
  // %r0 is an overlapping tensor.extract_slice that does not match, it must be
  // out of place.
  //      CHECK: tensor.extract_slice
  // CHECK-SAME: {__inplace_results_attr__ = ["false"]}
  %r0 = tensor.extract_slice %A[0][4][1] : tensor<?xf32> to tensor<4xf32>

  // %r1 can bufferize inplace fine.
  //      CHECK: tensor.insert_slice
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  %r1 = tensor.insert_slice %r0 into %A[%idx][4][1] : tensor<4xf32> into tensor<?xf32>

  // %r3 does bufferizes inplace because %B is not inplaceable.
  // %r0 is an overlapping tensor.extract_slice that does not match, but does
  // not alias with the buffer coming from %r3 so it can actually bufferize
  // inplace.
  //      CHECK: tensor.extract_slice
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  %r2 = tensor.extract_slice %B[0][4][1] : tensor<?xf32> to tensor<4xf32>

  // %r3 cannot bufferize inplace since %B is not inplaceable.
  //      CHECK: tensor.insert_slice
  // CHECK-SAME: {__inplace_results_attr__ = ["false"]}
  %r3 = tensor.insert_slice %r2 into %B[%idx][4][1] : tensor<4xf32> into tensor<?xf32>

  return %r1, %r3: tensor<?xf32>, tensor<?xf32>
}

// -----

// CHECK-LABEL: func @extract_slice_matching_insert_slice
func @extract_slice_matching_insert_slice(
    %A : tensor<?xf32> {linalg.inplaceable = true},
    %B : tensor<?xf32>)
  -> (tensor<?xf32>, tensor<?xf32>)
{
  // %r1 bufferizes inplace because %A is inplaceable.
  // %r0 is a tensor.extract_slice that matches, it can also be bufferized
  // inplace.
  //      CHECK: tensor.extract_slice
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  %r0 = tensor.extract_slice %A[0][4][1] : tensor<?xf32> to tensor<4xf32>

  //      CHECK: tensor.insert_slice
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  %r1 = tensor.insert_slice %r0 into %A[0][4][1] : tensor<4xf32> into tensor<?xf32>

  // %r2 is a tensor.extract_slice that matches %r3, it can be bufferized
  // inplace.
  //      CHECK: tensor.extract_slice
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  %r2 = tensor.extract_slice %B[0][4][1] : tensor<?xf32> to tensor<4xf32>

  // tensor.insert_slice cannot bufferize inplace.
  // This should have been captured by a canonicalization pattern and it would
  // be unproductive to have special logic in bufferization to encode matching
  // insert_slice(extract_slice(A), A).
  //      CHECK: tensor.insert_slice
  // CHECK-SAME: {__inplace_results_attr__ = ["false"]}
  %r3 = tensor.insert_slice %r2 into %B[0][4][1] : tensor<4xf32> into tensor<?xf32>

  return %r1, %r3: tensor<?xf32>, tensor<?xf32>
}

// -----

// CHECK-LABEL: func @extract_slice_linalg_readonly_use
func @extract_slice_linalg_readonly_use(
    %A : tensor<?x?xf32>,
    %B : tensor<4x4xf32>,
    %C : tensor<4x4xf32> {linalg.inplaceable = true})
  ->  (tensor<4x4xf32>, tensor<4x4xf32>)
{
  // tensor.extract_slice is only used as a read, no interference irrespective
  // of user's inplace status.
  //     CHECK: tensor.extract_slice
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  %sA = tensor.extract_slice %A[0, 0][4, 4][1, 1] : tensor<?x?xf32> to tensor<4x4xf32>

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

// CHECK-LABEL: func @extract_slice_to_linalg_write_use
func @extract_slice_to_linalg_write_use(
    %A : tensor<4x4xf32>,
    %B : tensor<?x?xf32>,
    %C : tensor<?x?xf32> {linalg.inplaceable = true})
  ->  (tensor<4x4xf32>, tensor<4x4xf32>)
{
  // Step 4. %sB forward propagates to a write in %D but it is not inplace.
  // So this is only ever read and can bufferize inplace.
  //     CHECK: tensor.extract_slice
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  %sB = tensor.extract_slice %B[0, 0][4, 4][1, 1] : tensor<?x?xf32> to tensor<4x4xf32>

  // Step 3. %sB has a read interference in %E, it does not bufferize inplace.
  //     CHECK: linalg.matmul
  // CHECK-SAME: {__inplace_results_attr__ = ["false"]}
  %D = linalg.matmul  ins(%B, %C: tensor<?x?xf32>, tensor<?x?xf32>)
                     outs(%sB: tensor<4x4xf32>)
    -> tensor<4x4xf32>

  // Step 2. %sC forward propagates to an inplace write in %E.
  // %sC backward propagates to %C which is inplaceable.
  // As a consequence this is bufferized inplace.
  //     CHECK: tensor.extract_slice
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  %sC = tensor.extract_slice %C[0, 0][4, 4][1, 1] : tensor<?x?xf32> to tensor<4x4xf32>

  // Step 1. %sC backprops to the tensor.extract_slice producer which is not
  // considered an interference. This bufferizes inplace.
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

// CHECK-LABEL: func @extract_slice_to_linalg_write_use
func @extract_slice_to_linalg_write_use(
    %A : tensor<4x4xf32>,
    %B : tensor<?x?xf32>,
    %C : tensor<?x?xf32> {linalg.inplaceable = true})
  ->  (tensor<4x4xf32>, tensor<4x4xf32>)
{
  // Step 4. %sB forward propagates to an inplace write in %D.
  // %sB backward propagates to %B which is not inplaceable.
  // As a consequence this is bufferized out of place.
  //     CHECK: tensor.extract_slice
  // CHECK-SAME: {__inplace_results_attr__ = ["false"]}
  %sB = tensor.extract_slice %B[0, 0][4, 4][1, 1] : tensor<?x?xf32> to tensor<4x4xf32>

  // Step 3. %sB backprops to the tensor.extract_slice producer which is not
  // considered an interference. This bufferizes inplace.
  //     CHECK: linalg.matmul
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  %D = linalg.matmul  ins(%B, %C: tensor<?x?xf32>, tensor<?x?xf32>)
                     outs(%sB: tensor<4x4xf32>)
    -> tensor<4x4xf32>

  // Step 2. %sC forward propagates to an inplace write in %E.
  // %sC backward propagates to %C which is inplaceable.
  // As a consequence this is bufferized inplace.
  //     CHECK: tensor.extract_slice
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  %sC = tensor.extract_slice %C[0, 0][4, 4][1, 1] : tensor<?x?xf32> to tensor<4x4xf32>

  // Step 1. %sC backprops to the tensor.extract_slice producer which is not
  // considered an interference. This bufferizes inplace.
  //     CHECK: linalg.matmul
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  %E = linalg.matmul  ins(%A, %A: tensor<4x4xf32>, tensor<4x4xf32>)
                     outs(%sC: tensor<4x4xf32>)
    -> tensor<4x4xf32>

  return %D, %E: tensor<4x4xf32>, tensor<4x4xf32>
}

// -----

// CHECK-LABEL: func @nested_extract_slice_and_insert
func @nested_extract_slice_and_insert(
    %A : tensor<?x?xf32>,
    %B : tensor<?x?xf32> {linalg.inplaceable = true},
    %C : tensor<?x?xf32> {linalg.inplaceable = true},
    %idx : index)
  ->  (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>)
{
  %f0 = arith.constant 0.0 : f32

  // 2-level matching tensor.extract_slice / tensor.insert_slice into non
  // inplaceable %A.
  //   - %rA is not inplaceable because %A is not inplaceable at function boundary.
  //   - once %rA is deemed not inplaceable, nothing prevent %rsA to be inplaceable
  //   - this propagates to %FA and %ssA being inplaceable.
  //   - %sA would then bufferize to an inplace write (i.e. %FA) but %A is not
  //     inplaceable and so %sA is not inplaceable.
  //     CHECK: tensor.extract_slice
  // CHECK-SAME: {__inplace_results_attr__ = ["false"]}
  // CHECK-NEXT: tensor.extract_slice
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  // CHECK-NEXT: fill
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  // CHECK-NEXT: tensor.insert_slice
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  // CHECK-NEXT: tensor.insert_slice
  // CHECK-SAME: {__inplace_results_attr__ = ["false"]}
  %sA = tensor.extract_slice %A[0, 0][%idx, %idx][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
  %ssA = tensor.extract_slice %sA[0, 0][4, 4][1, 1] : tensor<?x?xf32> to tensor<4x4xf32>
  %FA = linalg.fill(%f0, %ssA) : f32, tensor<4x4xf32> -> tensor<4x4xf32>
  %rsA = tensor.insert_slice %FA into %sA[0, 0][4, 4][1, 1] : tensor<4x4xf32> into tensor<?x?xf32>
  %rA = tensor.insert_slice %rsA into %A[0, 0][%idx, %idx][1, 1] : tensor<?x?xf32> into tensor<?x?xf32>

  // 3-level matching tensor.extract_slice / tensor.insert_slice into
  // inplaceable %B.
  // CHECK-NEXT: tensor.extract_slice
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  // CHECK-NEXT: tensor.extract_slice
  // Atm, this 2nd tensor.extract_slice fails to bufferize inplace because
  // clobbering analysis conservatively test for equivalent buffers.
  // TODO: This is currently too restrictive and misses clobberings.
  // When available, use container-containee analysis.
  // CHECK-SAME: {__inplace_results_attr__ = ["false"]}
  // CHECK-NEXT: tensor.extract_slice
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  // CHECK-NEXT: fill
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  // CHECK-NEXT: tensor.insert_slice
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  // CHECK-NEXT: tensor.insert_slice
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  // CHECK-NEXT: tensor.insert_slice
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  %sB = tensor.extract_slice %B[0, 0][%idx, %idx][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
  %ssB = tensor.extract_slice %sB[0, 0][4, %idx][1, 1] : tensor<?x?xf32> to tensor<4x?xf32>
  %sssB = tensor.extract_slice %ssB[0, 0][4, 4][1, 1] : tensor<4x?xf32> to tensor<4x4xf32>
  %FB = linalg.fill(%f0, %sssB) : f32, tensor<4x4xf32> -> tensor<4x4xf32>
  %rssB = tensor.insert_slice %FB into %ssB[0, 0][4, 4][1, 1] : tensor<4x4xf32> into tensor<4x?xf32>
  %rsB = tensor.insert_slice %rssB into %sB[0, 0][4, %idx][1, 1] : tensor<4x?xf32> into tensor<?x?xf32>
  %rB = tensor.insert_slice %rsB into %B[0, 0][%idx, %idx][1, 1] : tensor<?x?xf32> into tensor<?x?xf32>

  // 2-level matching tensor.extract_slice / tensor.insert_slice into
  // inplaceable %C with a twist.
  // Throw a wrench in the system: %rsC production sizes do not match %ssC.
  // CHECK-NEXT: tensor.extract_slice
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  // The tensor.insert_slice that would be candidate for matching does not actually
  // match. That tensor.insert_slice can still be bufferized inplace nonetheless
  // but this tensor.extract_slice, which bufferizes to an inplace write, cannot.
  // CHECK-NEXT: tensor.extract_slice
  // CHECK-SAME: {__inplace_results_attr__ = ["false"]}
  // CHECK-NEXT: fill
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  // CHECK-NEXT: tensor.insert_slice
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  // CHECK-NEXT: tensor.insert_slice
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  %sC = tensor.extract_slice %C[0, 0][%idx, %idx][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
  %ssC = tensor.extract_slice %sC[0, 0][4, 4][1, 1] : tensor<?x?xf32> to tensor<4x4xf32>
  %FC = linalg.fill(%f0, %ssC) : f32, tensor<4x4xf32> -> tensor<4x4xf32>
  %rsC = tensor.insert_slice %FC into %sC[0, 0][12345, 67890][1, 1] : tensor<4x4xf32> into tensor<?x?xf32>
  %rC = tensor.insert_slice %rsC into %C[0, 0][%idx, %idx][1, 1] : tensor<?x?xf32> into tensor<?x?xf32>

  return %rA, %rB, %rC: tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>
}

//===----------------------------------------------------------------------===//
// Simple loop cases
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @scf_for_yield_only
func @scf_for_yield_only(%A : tensor<?xf32>,
                         %B : tensor<?xf32> {linalg.inplaceable = true},
                         %lb : index, %ub : index, %step : index)
  -> (tensor<?xf32>, tensor<?xf32>)
{
  //      CHECK: scf.for
  // CHECK-NEXT: scf.yield
  // CHECK-NEXT: {__inplace_results_attr__ = ["false"]}
  %r0 = scf.for %i = %lb to %ub step %step iter_args(%t = %A) -> (tensor<?xf32>) {
    scf.yield %t : tensor<?xf32>
  }

  //      CHECK: scf.for
  // CHECK-NEXT: scf.yield
  // CHECK-NEXT: {__inplace_results_attr__ = ["true"]}
  %r1 = scf.for %i = %lb to %ub step %step iter_args(%t = %B) -> (tensor<?xf32>) {
    scf.yield %t : tensor<?xf32>
  }

  return %r0, %r1: tensor<?xf32>, tensor<?xf32>
}

// -----

// CHECK-LABEL: func @scf_for_with_tensor.insert_slice
func @scf_for_with_tensor.insert_slice(%A : tensor<?xf32>,
              %B : tensor<?xf32> {linalg.inplaceable = true},
              %C : tensor<4xf32>,
              %lb : index, %ub : index, %step : index)
  -> (tensor<?xf32>, tensor<?xf32>)
{
  //      CHECK: scf.for
  // scf.for bbArgs are always inplaceable seen from ops inside the body:
  //   1. Either the matching tensor is not inplaceable and an alloc occurs
  //      which makes bbArg inplaceable.
  //   2. Or it is already inplaceable and so is bbArg.
  // CHECK-NEXT:   tensor.insert_slice
  // CHECK-SAME:     {__inplace_results_attr__ = ["true"]}
  // CHECK-NEXT:   tensor.insert_slice
  // CHECK-SAME:     {__inplace_results_attr__ = ["true"]}
  // CHECK-NEXT:   scf.yield
  // CHECK-NEXT: {__inplace_results_attr__ = ["false", "true"]}
  %r0:2 = scf.for %i = %lb to %ub step %step iter_args(%tA = %A, %tB = %B)
      -> (tensor<?xf32>, tensor<?xf32>)
  {
    %ttA = tensor.insert_slice %C into %tA[0][4][1] : tensor<4xf32> into tensor<?xf32>
    %ttB = tensor.insert_slice %C into %tB[0][4][1] : tensor<4xf32> into tensor<?xf32>
    scf.yield %ttA, %ttB : tensor<?xf32>, tensor<?xf32>
  }

  return %r0#0, %r0#1: tensor<?xf32>, tensor<?xf32>
}

// -----

func private @some_use(tensor<?xf32>) -> ()

// CHECK-LABEL: func @scf_for_deps
func @scf_for_deps(%A : tensor<?xf32> {linalg.inplaceable = true},
                   %B : tensor<?xf32> {linalg.inplaceable = true},
                   %lb : index, %ub : index, %step : index)
  -> (tensor<?xf32>, tensor<?xf32>)
{
  // %r0 must be out of place because one use of %t in the subsequent production
  // of %r1 is read.
  //      CHECK: scf.for
  // CHECK-NEXT: call
  // CHECK-NEXT: scf.yield
  // CHECK-NEXT: {__inplace_results_attr__ = ["false"]}
  %r0 = scf.for %i = %lb to %ub step %step iter_args(%t = %A) -> (tensor<?xf32>) {
    call @some_use(%t) : (tensor<?xf32>) -> ()
    scf.yield %t : tensor<?xf32>
  }

  // %r1 bufferizes inplace fine.
  //      CHECK: scf.for
  // CHECK-NEXT: call
  // CHECK-NEXT: scf.yield
  // CHECK-NEXT: {__inplace_results_attr__ = ["true"]}
  %r1 = scf.for %i = %lb to %ub step %step iter_args(%t = %A) -> (tensor<?xf32>) {
    call @some_use(%t) : (tensor<?xf32>) -> ()
    scf.yield %t : tensor<?xf32>
  }

  // %r2 must be out of place because one use of %t in the subsequent production
  // of %r3 is read.
  //      CHECK: linalg.tiled_loop
  // CHECK-NEXT: call
  // CHECK-NEXT: linalg.yield
  // CHECK-NEXT: {__inplace_results_attr__ = ["false"]}
  %r2 = linalg.tiled_loop (%i) = (%lb) to (%ub) step (%step)
        ins()
        outs(%t = %B: tensor<?xf32>) {
    call @some_use(%t) : (tensor<?xf32>) -> ()
    linalg.yield %t : tensor<?xf32>
  }

  // %r3 bufferizes inplace fine.
  //      CHECK: linalg.tiled_loop
  // CHECK-NEXT: call
  // CHECK-NEXT: linalg.yield
  // CHECK-NEXT: {__inplace_results_attr__ = ["true"]}
  %r3 = linalg.tiled_loop (%i) = (%lb) to (%ub) step (%step)
        ins()
        outs(%t = %B: tensor<?xf32>) {
    call @some_use(%t) : (tensor<?xf32>) -> ()
    linalg.yield %t : tensor<?xf32>
  }

  return %r1, %r3: tensor<?xf32>, tensor<?xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Cross function boundary cases.
//===----------------------------------------------------------------------===//

func private @foo(tensor<64xf32>)

// CHECK-LABEL: dependence_through_call
func @dependence_through_call(%I : tensor<64xf32> {linalg.inplaceable = true}) {
  %f1 = arith.constant 1.000000e+00 : f32
  %f2 = arith.constant 2.000000e+00 : f32

  // 2. %B already bufferizes inplace, %A would alias and have a different
  // value. The calls to `foo` are determined to read conservatively, so %A
  // cannot bufferize inplace.
  //     CHECK: fill
  // CHECK-SAME: {__inplace_results_attr__ = ["false"]}
  %A = linalg.fill(%f1, %I) : f32, tensor<64xf32> -> tensor<64xf32>

  // 1. Bufferizes inplace: no alias to %A is yet possible.
  //     CHECK: fill
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  %B = linalg.fill(%f2, %I) : f32, tensor<64xf32> -> tensor<64xf32>

  call @foo(%A) : (tensor<64xf32>) -> ()
  call @foo(%B) : (tensor<64xf32>) -> ()

  return
}

// -----

func private @foo(tensor<64xf32>)

func private @bar(%A : tensor<64xf32>) {
  call @foo(%A) : (tensor<64xf32>) -> ()
  return
}

func @read_dependence_through_scf_and_call(
    %I : tensor<64xf32> {linalg.inplaceable = true},
    %I2 : tensor<64xf32> {linalg.inplaceable = true}) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %f1 = arith.constant 1.000000e+00 : f32
  %f2 = arith.constant 2.000000e+00 : f32

  // 5. %B bufferizes inplace, %A would alias and have a different value.
  // The calls to `foo` are determined to read conservatively, so %A cannot
  // bufferize inplace.
  //     CHECK: fill
  // CHECK-SAME: {__inplace_results_attr__ = ["false"]}
  %A = linalg.fill(%f1, %I) : f32, tensor<64xf32> -> tensor<64xf32>

  // 4. Bufferizes inplace: no alias to %A is yet possible.
  //     CHECK: fill
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  %B = linalg.fill(%f2, %I) : f32, tensor<64xf32> -> tensor<64xf32>

  // 3. Does not read or write, bufferizes inplace.
  //     CHECK: scf.for
  //     CHECK: {__inplace_results_attr__ = ["true", "true"]}
  %r:2 = scf.for %i = %c0 to %c10 step %c1 iter_args(%0 = %A, %1 = %B)
    -> (tensor<64xf32>, tensor<64xf32>)
  {
    scf.yield %0, %1 : tensor<64xf32>, tensor<64xf32>
  }
  call @foo(%r#0) : (tensor<64xf32>) -> ()
  call @foo(%r#1) : (tensor<64xf32>) -> ()

  // 2. %B2 already bufferizes inplace, %A2 would alias and have a different
  // value. The calls to `foo` are determined to read conservatively, so %A2
  // cannot bufferize inplace.
  //     CHECK: fill
  // CHECK-SAME: {__inplace_results_attr__ = ["false"]}
  %A2 = linalg.fill(%f1, %I2) : f32, tensor<64xf32> -> tensor<64xf32>

  // 1. Bufferizes inplace: no alias to %A2 is yet possible.
  //     CHECK: fill
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  %B2 = linalg.fill(%f2, %I2) : f32, tensor<64xf32> -> tensor<64xf32>

  call @bar(%A2) : (tensor<64xf32>) -> ()
  call @bar(%B2) : (tensor<64xf32>) -> ()
  return
}

// -----

//===----------------------------------------------------------------------===//
// Transitive cases through extract_slice.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @write_into_constant_via_alias
func @write_into_constant_via_alias(%v : vector<5xi32>,
                                    %s1 : index, %s2 : index,
                                    %s3 : index) -> tensor<?xi32> {
  %A = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
  //      CHECK: tensor.extract_slice
  // CHECK-SAME: {__inplace_results_attr__ = ["false"]}
  %b = tensor.extract_slice %A[%s1][%s2][1] : tensor<4xi32> to tensor<?xi32>
  //      CHECK: vector.transfer_write
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  %r = vector.transfer_write %v, %b[%s3] : vector<5xi32>, tensor<?xi32>
  return %r : tensor<?xi32>
}

// -----

builtin.func @matmul_on_tensors(
    %arg0: tensor<518x518xf32> {linalg.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false},
    %arg1: tensor<518x518xf32> {linalg.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false},
    %arg2: tensor<256x256xf32> {linalg.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = true})
    -> tensor<256x256xf32>
{
  %c0 = arith.constant 0 : index
  %cst_0 = arith.constant 0.000000e+00 : f32
  %cst_1 = arith.constant 1.000000e+00 : f32

  %7 = linalg.init_tensor [256, 256] : tensor<256x256xf32>

  //      CHECK: linalg.fill
  // CHECK-SAME: {__inplace_results_attr__ = ["false"]}
  //      CHECK: linalg.fill
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  %8 = linalg.fill(%cst_0, %7) : f32, tensor<256x256xf32> -> tensor<256x256xf32>
  %11 = linalg.fill(%cst_1, %7) : f32, tensor<256x256xf32> -> tensor<256x256xf32>

  //      CHECK: tensor.extract_slice
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  //      CHECK: tensor.extract_slice
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  //      CHECK: linalg.matmul
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  %sA = tensor.extract_slice %8[0, 0][256, 16][1, 1]: tensor<256x256xf32> to tensor<256x16xf32>
  %sB = tensor.extract_slice %11[0, 0][16, 256][1, 1]: tensor<256x256xf32> to tensor<16x256xf32>
  %r = linalg.matmul
         ins(%sA, %sB : tensor<256x16xf32>, tensor<16x256xf32>)
        outs(%arg2 : tensor<256x256xf32>) -> tensor<256x256xf32>

  return %r : tensor<256x256xf32>
}

// -----

builtin.func @matmul_on_tensors(
    %arg0: tensor<518x518xf32> {linalg.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false},
    %arg1: tensor<518x518xf32> {linalg.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false},
    %arg2: tensor<256x256xf32> {linalg.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = true})
    -> tensor<256x256xf32>
{
  %c0 = arith.constant 0 : index
  %cst_0 = arith.constant 0.000000e+00 : f32
  %cst_1 = arith.constant 1.000000e+00 : f32

  %7 = linalg.init_tensor [256, 256] : tensor<256x256xf32>

  //     CHECK: linalg.fill
  // CHECK-SAME: {__inplace_results_attr__ = ["false"]}
  //      CHECK: vector.transfer_write
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]
  %8 = linalg.fill(%cst_0, %7) : f32, tensor<256x256xf32> -> tensor<256x256xf32>
  %9 = vector.transfer_read %arg0[%c0, %c0], %cst_0 {in_bounds = [false, true]} : tensor<518x518xf32>, vector<256x256xf32>
  %10 = vector.transfer_write %9, %8[%c0, %c0] {in_bounds = [true, true]} : vector<256x256xf32>, tensor<256x256xf32>

  //      CHECK: linalg.fill
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  //      CHECK: vector.transfer_write
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]
  %11 = linalg.fill(%cst_1, %7) : f32, tensor<256x256xf32> -> tensor<256x256xf32>
  %12 = vector.transfer_read %arg1[%c0, %c0], %cst_0 {in_bounds = [false, true]} : tensor<518x518xf32>, vector<256x256xf32>
  %13 = vector.transfer_write %12, %11[%c0, %c0] {in_bounds = [true, true]} : vector<256x256xf32>, tensor<256x256xf32>

  //      CHECK: tensor.extract_slice
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  //      CHECK: tensor.extract_slice
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  //      CHECK: linalg.matmul
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]}
  %sA = tensor.extract_slice %10[0, 0][256, 16][1, 1]: tensor<256x256xf32> to tensor<256x16xf32>
  %sB = tensor.extract_slice %13[0, 0][16, 256][1, 1]: tensor<256x256xf32> to tensor<16x256xf32>
  %r = linalg.matmul
         ins(%sA, %sB : tensor<256x16xf32>, tensor<16x256xf32>)
        outs(%arg2 : tensor<256x256xf32>) -> tensor<256x256xf32>

  return %r : tensor<256x256xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Chain of tensor.insert_slice is better traversed in reverse order without
// prioritizing  the tensor.insert_slice ops.
//===----------------------------------------------------------------------===//

func @insert_slice_chain(
    %v1: vector<32x90xf32>,
    %v2: vector<30x90xf32>,
    %arg0: tensor<62x126xf32> {linalg.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false},
    %arg1: tensor<126x90xf32> {linalg.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false},
    %arg2: tensor<62x90xf32> {linalg.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = true})
  -> tensor<62x90xf32> attributes {passthrough = [["target-cpu", "skylake-avx512"], ["prefer-vector-width", "512"]]}
{
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32

  //      CHECK: linalg.fill
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]
  %0 = linalg.fill(%cst, %arg2) : f32, tensor<62x90xf32> -> tensor<62x90xf32>

  //      CHECK: tensor.extract_slice
  // CHECK-SAME: {__inplace_results_attr__ = ["false"]
  // TODO: in order to have this extract_slice bufferize inplace, we need to write a range
  // analysis and determine that intersection([0, 32)x[0, 90), [32, 62)x[0, 90)) is empty.
  %2 = tensor.extract_slice %0[0, 0] [32, 90] [1, 1] : tensor<62x90xf32> to tensor<32x90xf32>
  //      CHECK: vector.transfer_write
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]
  %7 = vector.transfer_write %v1, %2[%c0, %c0] {in_bounds = [true, true]} : vector<32x90xf32>, tensor<32x90xf32>
  //      CHECK: tensor.insert_slice
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]
  %8 = tensor.insert_slice %7 into %0[0, 0] [32, 90] [1, 1] : tensor<32x90xf32> into tensor<62x90xf32>

  //      CHECK: tensor.extract_slice
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]
  %10 = tensor.extract_slice %8[32, 0] [30, 90] [1, 1] : tensor<62x90xf32> to tensor<30x90xf32>
  //      CHECK: vector.transfer_write
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]
  %14 = vector.transfer_write %v2, %10[%c0, %c0] {in_bounds = [true, true]} : vector<30x90xf32>, tensor<30x90xf32>
  //      CHECK: tensor.insert_slice
  // CHECK-SAME: {__inplace_results_attr__ = ["true"]
  %15 = tensor.insert_slice %14 into %8[32, 0] [30, 90] [1, 1] : tensor<30x90xf32> into tensor<62x90xf32>

  return %15 : tensor<62x90xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Insert point issue cases.
//===----------------------------------------------------------------------===//

// Only test IR validity wrt dominance.
// CHECK-LABEL: func @ip
func @ip(%t: tensor<10x20xf32> {linalg.inplaceable = true},
         %x: index, %y: index, %v: vector<5x6xf32>)
  -> tensor<10x20xf32>
{
  %c0 = arith.constant 0 : index
  %c256 = arith.constant 256 : index
  %c257 = arith.constant 257 : index
  %r = scf.for %arg0 = %c0 to %c257 step %c256 iter_args(%arg1 = %t) -> (tensor<10x20xf32>) {
    %t1 = tensor.extract_slice %arg1[%x, 0] [5, %y] [1, 1] : tensor<10x20xf32> to tensor<5x?xf32>
    %t11 = tensor.extract_slice %t1[0, 0] [5, %y] [1, 1] : tensor<5x?xf32> to tensor<5x?xf32>
    %t2 = vector.transfer_write %v, %t11[%c0, %c0] : vector<5x6xf32>, tensor<5x?xf32>
    %t3 = tensor.insert_slice %t2 into %arg1[%x, 0] [5, %y] [1, 1] : tensor<5x?xf32> into tensor<10x20xf32>
    scf.yield %t3 : tensor<10x20xf32>
  }
 return %r : tensor<10x20xf32>
}

