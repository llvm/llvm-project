// RUN: mlir-opt %s -one-shot-bufferize="bufferize-function-boundaries test-analysis-only allow-return-allocs" -split-input-file | FileCheck %s

// Run fuzzer with different seeds.
// RUN: mlir-opt %s -one-shot-bufferize="bufferize-function-boundaries test-analysis-only allow-return-allocs analysis-fuzzer-seed=23" -split-input-file -o /dev/null
// RUN: mlir-opt %s -one-shot-bufferize="bufferize-function-boundaries test-analysis-only allow-return-allocs analysis-fuzzer-seed=59" -split-input-file -o /dev/null
// RUN: mlir-opt %s -one-shot-bufferize="bufferize-function-boundaries test-analysis-only allow-return-allocs analysis-fuzzer-seed=91" -split-input-file -o /dev/null

// TODO: Extract op-specific test cases and move them to their respective
// dialects.

//===----------------------------------------------------------------------===//
// Simple cases
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @extract_slice_fun(
func.func @extract_slice_fun(%A : tensor<?xf32> {bufferization.writable = false},
//  CHECK-SAME:              bufferization.access = "read"
                             %B : tensor<?xf32> {bufferization.writable = true})
//  CHECK-SAME:              bufferization.access = "read"
  -> (tensor<4xf32>, tensor<8xf32>)
{
  // tensor.extract_slice is not used in a write, it is not compelled to
  // bufferize out of place. Let callers decide whether they want to create
  // aliasing subviews at all call sites or whether they allocate.
  // This is true irrespective of whether the function argument is inplaceable.
  //     CHECK: tensor.extract_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true"]}
  %r0 = tensor.extract_slice %A[0][4][1] : tensor<?xf32> to tensor<4xf32>

  //     CHECK: tensor.extract_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true"]}
  %r1 = tensor.extract_slice %B[0][8][1] : tensor<?xf32> to tensor<8xf32>

  return %r0, %r1: tensor<4xf32>, tensor<8xf32>
}

// -----

// CHECK-LABEL: func @insert_slice_fun(
func.func @insert_slice_fun(%A : tensor<?xf32> {bufferization.writable = false},
//  CHECK-SAME:             bufferization.access = "read"
                            %B : tensor<?xf32> {bufferization.writable = true},
//  CHECK-SAME:             bufferization.access = "read-write"
                            %C : tensor<4xf32> {bufferization.writable = false})
//  CHECK-SAME:             bufferization.access = "read"
  -> (tensor<?xf32>, tensor<?xf32>)
{
  // must bufferize out of place.
  //      CHECK: tensor.insert_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "false"]}
  %r0 = tensor.insert_slice %C into %A[0][4][1] : tensor<4xf32> into tensor<?xf32>

  // bufferizes inplace.
  //      CHECK: tensor.insert_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true"]}
  %r1 = tensor.insert_slice %C into %B[0][4][1] : tensor<4xf32> into tensor<?xf32>

  //      CHECK: return
  // CHECK-SAME: __equivalent_func_args__ = [-1, 1]
  return %r0, %r1: tensor<?xf32>, tensor<?xf32>
}

// -----

// CHECK-LABEL: func @conflict_on_B(
func.func @conflict_on_B(%A : tensor<4x4xf32> {bufferization.writable = true},
//  CHECK-SAME:          bufferization.access = "read"
                         %B : tensor<4x4xf32> {bufferization.writable = true})
//  CHECK-SAME:          bufferization.access = "read-write"
  -> (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>)
{
  // matmul output operand interferes with input operand.
  //     CHECK: linalg.matmul
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true", "false"]}
  %C = linalg.matmul  ins(%A, %B: tensor<4x4xf32>, tensor<4x4xf32>)
                     outs(%B: tensor<4x4xf32>)
    -> tensor<4x4xf32>

  // matmul output operand interferes with input operand.
  //     CHECK: linalg.matmul
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true", "false"]}
  %D = linalg.matmul  ins(%B, %A: tensor<4x4xf32>, tensor<4x4xf32>)
                     outs(%B: tensor<4x4xf32>)
    -> tensor<4x4xf32>

  // matmul output operand does not interferes with input operand.
  //     CHECK: linalg.matmul
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true", "true"]}
  %E = linalg.matmul  ins(%A, %A: tensor<4x4xf32>, tensor<4x4xf32>)
                     outs(%B: tensor<4x4xf32>)
    -> tensor<4x4xf32>

  //      CHECK: return
  // CHECK-SAME: __equivalent_func_args__ = [-1, -1, 1]
  return %C, %D, %E: tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>
}

//===----------------------------------------------------------------------===//
// Length-1 producer-consumer cases.
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @extract_slice_extract_slice(
func.func @extract_slice_extract_slice(
    %A : tensor<?xf32> {bufferization.writable = true},
//  CHECK-SAME:         bufferization.access = "read"
    %B : tensor<?xf32> {bufferization.writable = false})
//  CHECK-SAME:         bufferization.access = "read"
  -> (tensor<2xf32>, tensor<2xf32>)
{
  // tensor.extract_slice is not used in a write, it is not compelled to
  // bufferize out of place. Let callers decide whether they want to create
  // aliasing subviews at all call sites or whether they allocate.
  // This is true irrespective of whether the function argument is inplaceable.
  // CHECK: {__inplace_operands_attr__ = ["true"]}
  %r0 = tensor.extract_slice %A[0][4][1] : tensor<?xf32> to tensor<4xf32>

  // CHECK: {__inplace_operands_attr__ = ["true"]}
  %r1 = tensor.extract_slice %r0[0][2][1] : tensor<4xf32> to tensor<2xf32>

  // CHECK: {__inplace_operands_attr__ = ["true"]}
  %r2 = tensor.extract_slice %B[0][4][1] : tensor<?xf32> to tensor<4xf32>

  // CHECK: {__inplace_operands_attr__ = ["true"]}
  %r3 = tensor.extract_slice %r2[0][2][1] : tensor<4xf32> to tensor<2xf32>

  return %r1, %r3: tensor<2xf32>, tensor<2xf32>
}

// -----

// CHECK-LABEL: func @insert_slice_insert_slice(
func.func @insert_slice_insert_slice(
    %A : tensor<?xf32> {bufferization.writable = true},
//  CHECK-SAME:         bufferization.access = "read-write"
    %A2 : tensor<4xf32> {bufferization.writable = true},
//  CHECK-SAME:          bufferization.access = "read-write"
    %A3 : tensor<2xf32> {bufferization.writable = true},
//  CHECK-SAME:          bufferization.access = "read"
    %B : tensor<?xf32> {bufferization.writable = false},
//  CHECK-SAME:         bufferization.access = "read"
    %B2 : tensor<4xf32> {bufferization.writable = false},
//  CHECK-SAME:          bufferization.access = "read"
    %B3 : tensor<2xf32> {bufferization.writable = false})
//  CHECK-SAME:          bufferization.access = "read"
  -> (tensor<?xf32>, tensor<?xf32>)
{
  // CHECK: {__inplace_operands_attr__ = ["true", "true"]}
  %r0 = tensor.insert_slice %A3 into %A2[0][2][1] : tensor<2xf32> into tensor<4xf32>

  // CHECK: {__inplace_operands_attr__ = ["true", "true"]}
  %r1 = tensor.insert_slice %r0 into %A[0][4][1] : tensor<4xf32> into tensor<?xf32>

  // CHECK: {__inplace_operands_attr__ = ["true", "false"]}
  %r2 = tensor.insert_slice %B3 into %B2[0][2][1] : tensor<2xf32> into tensor<4xf32>

  // CHECK: {__inplace_operands_attr__ = ["true", "false"]}
  %r3 = tensor.insert_slice %r2 into %B[0][4][1] : tensor<4xf32> into tensor<?xf32>

  //      CHECK: return
  // CHECK-SAME: __equivalent_func_args__ = [0, -1]
  return %r1, %r3: tensor<?xf32>, tensor<?xf32>
}

// -----

// CHECK-LABEL: func @extract_slice_nonmatching_insert_slice
func.func @extract_slice_nonmatching_insert_slice(
    %A : tensor<?xf32> {bufferization.writable = true},
    %B : tensor<?xf32> {bufferization.writable = false},
    %idx: index)
  -> (tensor<?xf32>, tensor<?xf32>)
{
  // %r1 bufferizes inplace because %A is inplaceable.
  // %r0 is an overlapping tensor.extract_slice that does not match, it must be
  // out of place.
  //      CHECK: tensor.extract_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["false"]}
  %r0 = tensor.extract_slice %A[0][4][1] : tensor<?xf32> to tensor<4xf32>

  // %r1 can bufferize inplace fine.
  //      CHECK: tensor.insert_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true", "none"]}
  %r1 = tensor.insert_slice %r0 into %A[%idx][4][1] : tensor<4xf32> into tensor<?xf32>

  // %r3 does bufferizes inplace because %B is not inplaceable.
  // %r0 is an overlapping tensor.extract_slice that does not match, but does
  // not alias with the buffer coming from %r3 so it can actually bufferize
  // inplace.
  //      CHECK: tensor.extract_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true"]}
  %r2 = tensor.extract_slice %B[0][4][1] : tensor<?xf32> to tensor<4xf32>

  // %r3 cannot bufferize inplace since %B is not inplaceable.
  //      CHECK: tensor.insert_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "false", "none"]}
  %r3 = tensor.insert_slice %r2 into %B[%idx][4][1] : tensor<4xf32> into tensor<?xf32>

  //      CHECK: return
  // CHECK-SAME: __equivalent_func_args__ = [0, -1]
  return %r1, %r3: tensor<?xf32>, tensor<?xf32>
}

// -----

// CHECK-LABEL: func @extract_slice_matching_insert_slice
func.func @extract_slice_matching_insert_slice(
    %A : tensor<?xf32> {bufferization.writable = true},
    %B : tensor<?xf32> {bufferization.writable = false})
  -> (tensor<?xf32>, tensor<?xf32>)
{
  // %r1 bufferizes inplace because %A is inplaceable.
  // %r0 is a tensor.extract_slice that matches, it can also be bufferized
  // inplace.
  //      CHECK: tensor.extract_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true"]}
  %r0 = tensor.extract_slice %A[0][4][1] : tensor<?xf32> to tensor<4xf32>

  //      CHECK: tensor.insert_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true"]}
  %r1 = tensor.insert_slice %r0 into %A[0][4][1] : tensor<4xf32> into tensor<?xf32>

  // %r2 is a tensor.extract_slice that matches %r3, it can be bufferized
  // inplace.
  //      CHECK: tensor.extract_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true"]}
  %r2 = tensor.extract_slice %B[0][4][1] : tensor<?xf32> to tensor<4xf32>

  // tensor.insert_slice cannot bufferize inplace.
  // This should have been captured by a canonicalization pattern and it would
  // be unproductive to have special logic in bufferization to encode matching
  // insert_slice(extract_slice(A), A).
  //      CHECK: tensor.insert_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "false"]}
  %r3 = tensor.insert_slice %r2 into %B[0][4][1] : tensor<4xf32> into tensor<?xf32>

  //      CHECK: return
  // CHECK-SAME: __equivalent_func_args__ = [0, -1]
  return %r1, %r3: tensor<?xf32>, tensor<?xf32>
}

// -----

// CHECK-LABEL: @read_of_matching_insert_slice_source
func.func @read_of_matching_insert_slice_source(
    %A : tensor<?xf32> {bufferization.writable = true},
    %idx : index,
    %idx2 : index)
  -> (tensor<?xf32>, vector<5xf32>)
{
  %cst = arith.constant 0.0 : f32
  %cst2 = arith.constant 1.0 : f32

  //      CHECK: tensor.extract_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "none", "none"]}
  %0 = tensor.extract_slice %A[%idx][%idx][1] : tensor<?xf32> to tensor<?xf32>

  //      CHECK: linalg.fill
  // CHECK-SAME: {__inplace_operands_attr__ = ["none", "true"]}
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?xf32>) -> tensor<?xf32>

  //      CHECK: tensor.insert_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true", "none", "none"]}
  %2 = tensor.insert_slice %1 into %A[%idx][%idx][1] : tensor<?xf32> into tensor<?xf32>

  %3 = vector.transfer_read %1[%idx2], %cst2 : tensor<?xf32>, vector<5xf32>

  //      CHECK: return
  // CHECK-SAME: __equivalent_func_args__ = [0, -1]
  return %2, %3 : tensor<?xf32>, vector<5xf32>
}

// -----

// CHECK-LABEL: @read_of_matching_insert_slice_source_interleaved
func.func @read_of_matching_insert_slice_source_interleaved(
    %A : tensor<?xf32> {bufferization.writable = true},
    %idx : index,
    %idx2 : index,
    %idx3 : index)
  -> (tensor<?xf32>, vector<5xf32>)
{
  %cst = arith.constant 0.0 : f32
  %cst2 = arith.constant 1.0 : f32

  //      CHECK: tensor.extract_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["false", "none", "none"]}
  %0 = tensor.extract_slice %A[%idx][%idx][1] : tensor<?xf32> to tensor<?xf32>

  //      CHECK: linalg.fill
  // CHECK-SAME: {__inplace_operands_attr__ = ["none", "true"]}
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?xf32>) -> tensor<?xf32>

  //      CHECK: tensor.insert_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true", "none", "none"]}
  %2 = tensor.insert_slice %1 into %A[%idx][%idx][1] : tensor<?xf32> into tensor<?xf32>

  //      CHECK: tensor.extract_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "none", "none"]}
  %4 = tensor.extract_slice %2[%idx3][%idx3][1] : tensor<?xf32> to tensor<?xf32>

  //      CHECK: linalg.fill
  // CHECK-SAME: {__inplace_operands_attr__ = ["none", "true"]}
  %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<?xf32>) -> tensor<?xf32>

  %3 = vector.transfer_read %1[%idx2], %cst2 : tensor<?xf32>, vector<5xf32>

  //      CHECK: tensor.insert_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true", "none", "none"]}
  %6 = tensor.insert_slice %5 into %2[%idx3][%idx3][1] : tensor<?xf32> into tensor<?xf32>

  //      CHECK: return
  // CHECK-SAME: __equivalent_func_args__ = [0, -1]
  return %6, %3 : tensor<?xf32>, vector<5xf32>
}

// -----

// CHECK-LABEL: func @extract_slice_linalg_readonly_use
func.func @extract_slice_linalg_readonly_use(
    %A : tensor<?x?xf32> {bufferization.writable = false},
    %B : tensor<4x4xf32> {bufferization.writable = false},
    %C : tensor<4x4xf32> {bufferization.writable = true})
  ->  (tensor<4x4xf32>, tensor<4x4xf32>)
{
  // tensor.extract_slice is only used as a read, no interference irrespective
  // of user's inplace status.
  //     CHECK: tensor.extract_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true"]}
  %sA = tensor.extract_slice %A[0, 0][4, 4][1, 1] : tensor<?x?xf32> to tensor<4x4xf32>

  // matmul output operand is not inplaceable at the function boundary.
  //     CHECK: linalg.matmul
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true", "false"]}
  %D = linalg.matmul  ins(%sA, %B: tensor<4x4xf32>, tensor<4x4xf32>)
                     outs(%B: tensor<4x4xf32>)
    -> tensor<4x4xf32>

  // matmul output operand is inplaceable at the function boundary.
  //     CHECK: linalg.matmul
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true", "true"]}
  %E = linalg.matmul  ins(%sA, %B: tensor<4x4xf32>, tensor<4x4xf32>)
                     outs(%C: tensor<4x4xf32>)
    -> tensor<4x4xf32>

  //      CHECK: return
  // CHECK-SAME: __equivalent_func_args__ = [-1, 2]
  return %D, %E: tensor<4x4xf32>, tensor<4x4xf32>
}

// -----

// CHECK-LABEL: func @extract_slice_to_linalg_write_use
func.func @extract_slice_to_linalg_write_use(
    %A : tensor<4x4xf32> {bufferization.writable = false},
    %B : tensor<?x?xf32> {bufferization.writable = false},
    %C : tensor<?x?xf32> {bufferization.writable = true})
  ->  (tensor<4x4xf32>, tensor<4x4xf32>)
{
  // Step 4. %sB forward propagates to a write in %D but it is not inplace.
  // So this is only ever read and can bufferize inplace.
  //     CHECK: tensor.extract_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true"]}
  %sB = tensor.extract_slice %B[0, 0][4, 4][1, 1] : tensor<?x?xf32> to tensor<4x4xf32>

  // Step 3. %sB has a read interference in %E, it does not bufferize inplace.
  //     CHECK: linalg.matmul
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true", "false"]}
  %D = linalg.matmul  ins(%B, %C: tensor<?x?xf32>, tensor<?x?xf32>)
                     outs(%sB: tensor<4x4xf32>)
    -> tensor<4x4xf32>

  // Step 2. %sC forward propagates to an inplace write in %E.
  // %sC backward propagates to %C which is inplaceable.
  // As a consequence this is bufferized inplace.
  //     CHECK: tensor.extract_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true"]}
  %sC = tensor.extract_slice %C[0, 0][4, 4][1, 1] : tensor<?x?xf32> to tensor<4x4xf32>

  // Step 1. %sC backprops to the tensor.extract_slice producer which is not
  // considered an interference. This bufferizes inplace.
  //     CHECK: linalg.matmul
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true", "true"]}
  %E = linalg.matmul  ins(%A, %sB: tensor<4x4xf32>, tensor<4x4xf32>)
                     outs(%sC: tensor<4x4xf32>)
    -> tensor<4x4xf32>

  return %D, %E: tensor<4x4xf32>, tensor<4x4xf32>
}

// -----

// CHECK-LABEL: func @insert_slice_double_extract_slice
func.func @insert_slice_double_extract_slice(
    %s1: index,
    %s2: index,
    %s3: index,
    %s4: index,
    %A: tensor<8x6xf32> {bufferization.writable = false},
    %B: tensor<6x6xf32> {bufferization.writable = false},
    %C: tensor<30x20xf32> {bufferization.writable = true})
  -> tensor<30x20xf32>
{
  //      CHECK: tensor.extract_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "none", "none", "none", "none"]}
  %15 = tensor.extract_slice %C[%s3, %s4] [%s1, %s2] [1, 1] : tensor<30x20xf32> to tensor<?x?xf32>

  //      CHECK: linalg.matmul
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true", "true"]}
  %18 = linalg.matmul ins(%A, %B : tensor<8x6xf32>, tensor<6x6xf32>) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>

  //      CHECK: tensor.extract_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "none", "none"]}
  %19 = tensor.extract_slice %18[0, 0] [%s1, %s2] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>

  //      CHECK: tensor.insert_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true", "none", "none", "none", "none"]}
  %20 = tensor.insert_slice %19 into %C[%s3, %s4] [%s1, %s2] [1, 1] : tensor<?x?xf32> into tensor<30x20xf32>

  //      CHECK: return
  // CHECK-SAME: __equivalent_func_args__ = [6]
  return %20 : tensor<30x20xf32>
}

//===----------------------------------------------------------------------===//
// Transitive cases
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @extract_slice_to_linalg_write_use
func.func @extract_slice_to_linalg_write_use(
    %A : tensor<4x4xf32> {bufferization.writable = false},
    %B : tensor<?x?xf32> {bufferization.writable = false},
    %C : tensor<?x?xf32> {bufferization.writable = true})
  ->  (tensor<4x4xf32>, tensor<4x4xf32>)
{
  // Step 4. %sB forward propagates to an inplace write in %D.
  // %sB backward propagates to %B which is not inplaceable.
  // As a consequence this is bufferized out of place.
  //     CHECK: tensor.extract_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["false"]}
  %sB = tensor.extract_slice %B[0, 0][4, 4][1, 1] : tensor<?x?xf32> to tensor<4x4xf32>

  // Step 3. %sB backprops to the tensor.extract_slice producer which is not
  // considered an interference. This bufferizes inplace.
  //     CHECK: linalg.matmul
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true", "true"]}
  %D = linalg.matmul  ins(%B, %C: tensor<?x?xf32>, tensor<?x?xf32>)
                     outs(%sB: tensor<4x4xf32>)
    -> tensor<4x4xf32>

  // Step 2. %sC forward propagates to an inplace write in %E.
  // %sC backward propagates to %C which is inplaceable.
  // As a consequence this is bufferized inplace.
  //     CHECK: tensor.extract_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true"]}
  %sC = tensor.extract_slice %C[0, 0][4, 4][1, 1] : tensor<?x?xf32> to tensor<4x4xf32>

  // Step 1. %sC backprops to the tensor.extract_slice producer which is not
  // considered an interference. This bufferizes inplace.
  //     CHECK: linalg.matmul
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true", "true"]}
  %E = linalg.matmul  ins(%A, %A: tensor<4x4xf32>, tensor<4x4xf32>)
                     outs(%sC: tensor<4x4xf32>)
    -> tensor<4x4xf32>

  return %D, %E: tensor<4x4xf32>, tensor<4x4xf32>
}

// -----

// CHECK-LABEL: func @nested_extract_slice_and_insert
func.func @nested_extract_slice_and_insert(
    %A : tensor<?x?xf32> {bufferization.writable = false},
    %B : tensor<?x?xf32> {bufferization.writable = true},
    %C : tensor<?x?xf32> {bufferization.writable = true},
    %idx : index,
    %sz1 : index,
    %sz2 : index)
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
  // CHECK-SAME: {__inplace_operands_attr__ = ["false", "none", "none"]}
  // CHECK-NEXT: tensor.extract_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true"]}
  // CHECK-NEXT: fill
  // CHECK-SAME: {__inplace_operands_attr__ = ["none", "true"]}
  // CHECK-NEXT: tensor.insert_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true"]}
  // CHECK-NEXT: tensor.insert_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "false", "none", "none"]}
  %sA = tensor.extract_slice %A[0, 0][%idx, %idx][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
  %ssA = tensor.extract_slice %sA[0, 0][4, 4][1, 1] : tensor<?x?xf32> to tensor<4x4xf32>
  %FA = linalg.fill ins(%f0 : f32) outs(%ssA : tensor<4x4xf32>) -> tensor<4x4xf32>
  %rsA = tensor.insert_slice %FA into %sA[0, 0][4, 4][1, 1] : tensor<4x4xf32> into tensor<?x?xf32>
  %rA = tensor.insert_slice %rsA into %A[0, 0][%idx, %idx][1, 1] : tensor<?x?xf32> into tensor<?x?xf32>

  // 3-level matching tensor.extract_slice / tensor.insert_slice into
  // inplaceable %B.
  // CHECK-NEXT: tensor.extract_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "none", "none"]}
  // CHECK-NEXT: tensor.extract_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "none"]}
  // CHECK-NEXT: tensor.extract_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true"]}
  // CHECK-NEXT: fill
  // CHECK-SAME: {__inplace_operands_attr__ = ["none", "true"]}
  // CHECK-NEXT: tensor.insert_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true"]}
  // CHECK-NEXT: tensor.insert_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true", "none"]}
  // CHECK-NEXT: tensor.insert_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true", "none", "none"]}
  %sB = tensor.extract_slice %B[0, 0][%idx, %idx][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
  %ssB = tensor.extract_slice %sB[0, 0][4, %idx][1, 1] : tensor<?x?xf32> to tensor<4x?xf32>
  %sssB = tensor.extract_slice %ssB[0, 0][4, 4][1, 1] : tensor<4x?xf32> to tensor<4x4xf32>
  %FB = linalg.fill ins(%f0 : f32) outs(%sssB : tensor<4x4xf32>) -> tensor<4x4xf32>
  %rssB = tensor.insert_slice %FB into %ssB[0, 0][4, 4][1, 1] : tensor<4x4xf32> into tensor<4x?xf32>
  %rsB = tensor.insert_slice %rssB into %sB[0, 0][4, %idx][1, 1] : tensor<4x?xf32> into tensor<?x?xf32>
  %rB = tensor.insert_slice %rsB into %B[0, 0][%idx, %idx][1, 1] : tensor<?x?xf32> into tensor<?x?xf32>

  // 2-level matching tensor.extract_slice / tensor.insert_slice into
  // inplaceable %C with a twist.
  // Throw a wrench in the system: %rsC production sizes do not match %ssC.
  // CHECK-NEXT: tensor.extract_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "none", "none"]}
  // The tensor.insert_slice that would be candidate for matching does not actually
  // match. That tensor.insert_slice can still be bufferized inplace nonetheless
  // but this tensor.extract_slice, which bufferizes to an inplace write, cannot.
  // CHECK-NEXT: tensor.extract_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["false", "none"]}
  // CHECK-NEXT: fill
  // CHECK-SAME: {__inplace_operands_attr__ = ["none", "true"]}
  // CHECK-NEXT: tensor.insert_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true", "none"]}
  // CHECK-NEXT: tensor.insert_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true", "none", "none"]}
  %sC = tensor.extract_slice %C[0, 0][%idx, %idx][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
  %ssC = tensor.extract_slice %sC[0, 0][%sz1, 4][1, 1] : tensor<?x?xf32> to tensor<?x4xf32>
  %FC = linalg.fill ins(%f0 : f32) outs(%ssC : tensor<?x4xf32>) -> tensor<?x4xf32>
  %rsC = tensor.insert_slice %FC into %sC[0, 0][%sz2, 4][1, 1] : tensor<?x4xf32> into tensor<?x?xf32>
  %rC = tensor.insert_slice %rsC into %C[0, 0][%idx, %idx][1, 1] : tensor<?x?xf32> into tensor<?x?xf32>

  //      CHECK: return
  // CHECK-SAME: __equivalent_func_args__ = [-1, 1, 2]
  return %rA, %rB, %rC: tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Cross function boundary cases.
//===----------------------------------------------------------------------===//

func.func private @foo(tensor<64xf32>)

// CHECK-LABEL: dependence_through_call
func.func @dependence_through_call(%I : tensor<64xf32> {bufferization.writable = true}) {
  %f1 = arith.constant 1.000000e+00 : f32
  %f2 = arith.constant 2.000000e+00 : f32

  // 2. %B already bufferizes inplace, %A would alias and have a different
  // value. The calls to `foo` are determined to read conservatively, so %A
  // cannot bufferize inplace.
  //     CHECK: fill
  // CHECK-SAME: {__inplace_operands_attr__ = ["none", "false"]}
  %A = linalg.fill ins(%f1 : f32) outs(%I : tensor<64xf32>) -> tensor<64xf32>

  // 1. Bufferizes inplace: no alias to %A is yet possible.
  //     CHECK: fill
  // CHECK-SAME: {__inplace_operands_attr__ = ["none", "true"]}
  %B = linalg.fill ins(%f2 : f32) outs(%I : tensor<64xf32>) -> tensor<64xf32>

  call @foo(%A) : (tensor<64xf32>) -> ()
  call @foo(%B) : (tensor<64xf32>) -> ()

  return
}

// -----

func.func private @foo(tensor<64xf32>)

func.func private @bar(%A : tensor<64xf32>) {
  call @foo(%A) : (tensor<64xf32>) -> ()
  return
}

func.func @read_dependence_through_scf_and_call(
    %I : tensor<64xf32> {bufferization.writable = true},
    %I2 : tensor<64xf32> {bufferization.writable = true}) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %f1 = arith.constant 1.000000e+00 : f32
  %f2 = arith.constant 2.000000e+00 : f32

  // 5. %B bufferizes inplace, %A would alias and have a different value.
  // The calls to `foo` are determined to read conservatively, so %A cannot
  // bufferize inplace.
  //     CHECK: fill
  // CHECK-SAME: {__inplace_operands_attr__ = ["none", "false"]}
  %A = linalg.fill ins(%f1 : f32) outs(%I : tensor<64xf32>) -> tensor<64xf32>

  // 4. Bufferizes inplace: no alias to %A is yet possible.
  //     CHECK: fill
  // CHECK-SAME: {__inplace_operands_attr__ = ["none", "true"]}
  %B = linalg.fill ins(%f2 : f32) outs(%I : tensor<64xf32>) -> tensor<64xf32>

  // 3. Does not read or write, bufferizes inplace.
  //      CHECK: scf.for
  // CHECK-NEXT: scf.yield
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true"]}
  //      CHECK: } {__inplace_operands_attr__ = ["none", "none", "none", "true", "true"]}
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
  // CHECK-SAME: {__inplace_operands_attr__ = ["none", "false"]}
  %A2 = linalg.fill ins(%f1 : f32) outs(%I2 : tensor<64xf32>) -> tensor<64xf32>

  // 1. Bufferizes inplace: no alias to %A2 is yet possible.
  //     CHECK: fill
  // CHECK-SAME: {__inplace_operands_attr__ = ["none", "true"]}
  %B2 = linalg.fill ins(%f2 : f32) outs(%I2 : tensor<64xf32>) -> tensor<64xf32>

  call @bar(%A2) : (tensor<64xf32>) -> ()
  call @bar(%B2) : (tensor<64xf32>) -> ()
  return
}

// -----

//===----------------------------------------------------------------------===//
// Transitive cases through extract_slice.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @write_into_constant_via_alias
func.func @write_into_constant_via_alias(%v : vector<5xi32>,
                                    %s1 : index, %s2 : index,
                                    %s3 : index) -> tensor<?xi32> {
  %A = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
  //      CHECK: tensor.extract_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["false", "none", "none"]}
  %b = tensor.extract_slice %A[%s1][%s2][1] : tensor<4xi32> to tensor<?xi32>
  //      CHECK: vector.transfer_write
  // CHECK-SAME: {__inplace_operands_attr__ = ["none", "true", "none"]}
  %r = vector.transfer_write %v, %b[%s3] : vector<5xi32>, tensor<?xi32>
  return %r : tensor<?xi32>
}

// -----

func.func @matmul_on_tensors(
    %arg0: tensor<518x518xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, bufferization.writable = false},
    %arg1: tensor<518x518xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, bufferization.writable = false},
    %arg2: tensor<256x256xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, bufferization.writable = true})
    -> tensor<256x256xf32>
{
  %c0 = arith.constant 0 : index
  %cst_0 = arith.constant 0.000000e+00 : f32
  %cst_1 = arith.constant 1.000000e+00 : f32

  %7 = bufferization.alloc_tensor() : tensor<256x256xf32>

  //      CHECK: linalg.fill
  // CHECK-SAME: {__inplace_operands_attr__ = ["none", "false"]}
  //      CHECK: linalg.fill
  // CHECK-SAME: {__inplace_operands_attr__ = ["none", "true"]}
  %8 = linalg.fill ins(%cst_0 : f32) outs(%7 : tensor<256x256xf32>) -> tensor<256x256xf32>
  %11 = linalg.fill ins(%cst_1 : f32) outs(%7 : tensor<256x256xf32>) -> tensor<256x256xf32>

  //      CHECK: tensor.extract_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true"]}
  //      CHECK: tensor.extract_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true"]}
  //      CHECK: linalg.matmul
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true", "true"]}
  %sA = tensor.extract_slice %8[0, 0][256, 16][1, 1]: tensor<256x256xf32> to tensor<256x16xf32>
  %sB = tensor.extract_slice %11[0, 0][16, 256][1, 1]: tensor<256x256xf32> to tensor<16x256xf32>
  %r = linalg.matmul
         ins(%sA, %sB : tensor<256x16xf32>, tensor<16x256xf32>)
        outs(%arg2 : tensor<256x256xf32>) -> tensor<256x256xf32>

  //      CHECK: return
  // CHECK-SAME: __equivalent_func_args__ = [2]
  return %r : tensor<256x256xf32>
}

// -----

func.func @matmul_on_tensors(
    %arg0: tensor<518x518xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, bufferization.writable = false},
    %arg1: tensor<518x518xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, bufferization.writable = false},
    %arg2: tensor<256x256xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, bufferization.writable = true})
    -> tensor<256x256xf32>
{
  %c0 = arith.constant 0 : index
  %cst_0 = arith.constant 0.000000e+00 : f32
  %cst_1 = arith.constant 1.000000e+00 : f32

  %7 = bufferization.alloc_tensor() : tensor<256x256xf32>

  //     CHECK: linalg.fill
  // CHECK-SAME: {__inplace_operands_attr__ = ["none", "false"]}
  //      CHECK: vector.transfer_write
  // CHECK-SAME: {__inplace_operands_attr__ = ["none", "true", "none", "none"]
  %8 = linalg.fill ins(%cst_0 : f32) outs(%7 : tensor<256x256xf32>) -> tensor<256x256xf32>
  %9 = vector.transfer_read %arg0[%c0, %c0], %cst_0 {in_bounds = [false, true]} : tensor<518x518xf32>, vector<256x256xf32>
  %10 = vector.transfer_write %9, %8[%c0, %c0] {in_bounds = [true, true]} : vector<256x256xf32>, tensor<256x256xf32>

  //      CHECK: linalg.fill
  // CHECK-SAME: {__inplace_operands_attr__ = ["none", "true"]}
  //      CHECK: vector.transfer_write
  // CHECK-SAME: {__inplace_operands_attr__ = ["none", "true", "none", "none"]
  %11 = linalg.fill ins(%cst_1 : f32) outs(%7 : tensor<256x256xf32>) -> tensor<256x256xf32>
  %12 = vector.transfer_read %arg1[%c0, %c0], %cst_0 {in_bounds = [false, true]} : tensor<518x518xf32>, vector<256x256xf32>
  %13 = vector.transfer_write %12, %11[%c0, %c0] {in_bounds = [true, true]} : vector<256x256xf32>, tensor<256x256xf32>

  //      CHECK: tensor.extract_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true"]}
  //      CHECK: tensor.extract_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true"]}
  //      CHECK: linalg.matmul
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true", "true"]}
  %sA = tensor.extract_slice %10[0, 0][256, 16][1, 1]: tensor<256x256xf32> to tensor<256x16xf32>
  %sB = tensor.extract_slice %13[0, 0][16, 256][1, 1]: tensor<256x256xf32> to tensor<16x256xf32>
  %r = linalg.matmul
         ins(%sA, %sB : tensor<256x16xf32>, tensor<16x256xf32>)
        outs(%arg2 : tensor<256x256xf32>) -> tensor<256x256xf32>

  //      CHECK: return
  // CHECK-SAME: __equivalent_func_args__ = [2]
  return %r : tensor<256x256xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Chain of tensor.insert_slice is better traversed in reverse order without
// prioritizing  the tensor.insert_slice ops.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @insert_slice_chain(
func.func @insert_slice_chain(
    %v1: vector<32x90xf32>,
    %v2: vector<30x90xf32>,
    %arg0: tensor<62x126xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, bufferization.writable = false},
// CHECK-SAME: bufferization.access = "none"
    %arg1: tensor<126x90xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, bufferization.writable = false},
// CHECK-SAME: bufferization.access = "none"
    %arg2: tensor<62x90xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, bufferization.writable = true})
// CHECK-SAME: bufferization.access = "write"
  -> tensor<62x90xf32> attributes {passthrough = [["target-cpu", "skylake-avx512"], ["prefer-vector-width", "512"]]}
{
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32

  //      CHECK: linalg.fill
  // CHECK-SAME: {__inplace_operands_attr__ = ["none", "true"]
  %0 = linalg.fill ins(%cst : f32) outs(%arg2 : tensor<62x90xf32>) -> tensor<62x90xf32>

  //      CHECK: tensor.extract_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true"]
  %2 = tensor.extract_slice %0[0, 0] [32, 90] [1, 1] : tensor<62x90xf32> to tensor<32x90xf32>
  //      CHECK: vector.transfer_write
  // CHECK-SAME: {__inplace_operands_attr__ = ["none", "true", "none", "none"]
  %7 = vector.transfer_write %v1, %2[%c0, %c0] {in_bounds = [true, true]} : vector<32x90xf32>, tensor<32x90xf32>
  //      CHECK: tensor.insert_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true"]
  %8 = tensor.insert_slice %7 into %0[0, 0] [32, 90] [1, 1] : tensor<32x90xf32> into tensor<62x90xf32>

  //      CHECK: tensor.extract_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true"]
  %10 = tensor.extract_slice %8[32, 0] [30, 90] [1, 1] : tensor<62x90xf32> to tensor<30x90xf32>
  //      CHECK: vector.transfer_write
  // CHECK-SAME: {__inplace_operands_attr__ = ["none", "true", "none", "none"]
  %14 = vector.transfer_write %v2, %10[%c0, %c0] {in_bounds = [true, true]} : vector<30x90xf32>, tensor<30x90xf32>
  //      CHECK: tensor.insert_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true"]
  %15 = tensor.insert_slice %14 into %8[32, 0] [30, 90] [1, 1] : tensor<30x90xf32> into tensor<62x90xf32>

  //      CHECK: return
  // CHECK-SAME: __equivalent_func_args__ = [4]
  return %15 : tensor<62x90xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Insert point issue cases.
//===----------------------------------------------------------------------===//

// Only test IR validity wrt dominance.
// CHECK-LABEL: func @ip
func.func @ip(%t: tensor<10x20xf32> {bufferization.writable = true},
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

  //      CHECK: return
  // CHECK-SAME: __equivalent_func_args__ = [0]
 return %r : tensor<10x20xf32>
}

// -----

#accesses = [
  affine_map<(i) -> (i)>,
  affine_map<(i) -> (i)>,
  affine_map<(i) -> (i)>
]
#trait = {
  indexing_maps = #accesses,
  iterator_types = ["parallel"]
}

// CHECK-LABEL: func @linalg_op_same_out_tensors(
func.func @linalg_op_same_out_tensors(
    %t1: tensor<?xf32> {bufferization.writable = true},
// CHECK-SAME:          bufferization.access = "read"
    %t2: tensor<?xf32> {bufferization.writable = true})
// CHECK-SAME:          bufferization.access = "write"
  -> (tensor<?xf32>, tensor<?xf32>){

  //      CHECK: linalg.generic
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true", "false"]
  %o:2 = linalg.generic #trait ins(%t1 : tensor<?xf32>)
                               outs (%t2, %t2 : tensor<?xf32>, tensor<?xf32>) {
      ^bb(%0: f32, %1: f32, %2 : f32) :
        linalg.yield %0, %0 : f32, f32
    } -> (tensor<?xf32>, tensor<?xf32>)

  //      CHECK: return
  // CHECK-SAME: __equivalent_func_args__ = [1, -1]
  return %o#0, %o#1 : tensor<?xf32>, tensor<?xf32>
}

// -----

#accesses = [
  affine_map<(i) -> (i)>,
  affine_map<(i) -> (i)>,
  affine_map<(i) -> (i)>,
  affine_map<(i) -> (i)>
]
#trait = {
  indexing_maps = #accesses,
  iterator_types = ["parallel"]
}

// CHECK-LABEL: func @linalg_op_same_out_tensors_2(
func.func @linalg_op_same_out_tensors_2(
    %t1: tensor<?xf32> {bufferization.writable = true},
// CHECK-SAME:          bufferization.access = "read"
    %t2: tensor<?xf32> {bufferization.writable = true})
// CHECK-SAME:          bufferization.access = "write"
        -> (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>){

  //      CHECK: linalg.generic
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true", "false", "false"]
  %o:3 = linalg.generic #trait
          ins(%t1 : tensor<?xf32>)
          outs (%t2, %t2, %t2 : tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) {
      ^bb(%0: f32, %1: f32, %2 : f32, %3 : f32) :
        linalg.yield %0, %0, %0 : f32, f32, f32
    } -> (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>)

  //      CHECK: return
  // CHECK-SAME: __equivalent_func_args__ = [1, -1, -1]
  return %o#0, %o#1, %o#2 : tensor<?xf32>, tensor<?xf32>, tensor<?xf32>
}

// -----

// CHECK-LABEL: func @double_insert_slice_into_alias
func.func @double_insert_slice_into_alias(
    %v1: vector<32x90xf32>,
    %v2: vector<30x90xf32>,
    %arg2: tensor<62x90xf32> {bufferization.writable = true},
    %s1: index, %s2: index, %s3: index, %s4: index)
  -> (tensor<62x90xf32>, tensor<?x?xf32>)
{
  %c0 = arith.constant 0 : index

  // Cannot bufferize inplace this extract_slice because both operand and result
  // are modified and returned separately.
  //      CHECK: tensor.extract_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["false", "none", "none", "none", "none"]
  %e = tensor.extract_slice %arg2[%s1, %s2][%s3, %s4][1, 1] : tensor<62x90xf32> to tensor<?x?xf32>

  //      CHECK: tensor.extract_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true"]
  %2 = tensor.extract_slice %arg2[0, 0] [32, 90] [1, 1] : tensor<62x90xf32> to tensor<32x90xf32>
  //      CHECK: vector.transfer_write
  // CHECK-SAME: {__inplace_operands_attr__ = ["none", "true", "none", "none"]
  %7 = vector.transfer_write %v1, %2[%c0, %c0] {in_bounds = [true, true]} : vector<32x90xf32>, tensor<32x90xf32>
  //      CHECK: tensor.insert_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true"]
  %8 = tensor.insert_slice %7 into %arg2[0, 0] [32, 90] [1, 1] : tensor<32x90xf32> into tensor<62x90xf32>

  //      CHECK: tensor.extract_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true"]
  %10 = tensor.extract_slice %e[32, 0] [30, 90] [1, 1] : tensor<?x?xf32> to tensor<30x90xf32>
  //      CHECK: vector.transfer_write
  // CHECK-SAME: {__inplace_operands_attr__ = ["none", "true", "none", "none"]
  %14 = vector.transfer_write %v2, %10[%c0, %c0] {in_bounds = [true, true]} : vector<30x90xf32>, tensor<30x90xf32>
  //      CHECK: tensor.insert_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true"]
  %15 = tensor.insert_slice %14 into %e[32, 0] [30, 90] [1, 1] : tensor<30x90xf32> into tensor<?x?xf32>

  //      CHECK: return
  // CHECK-SAME: __equivalent_func_args__ = [2, -1]
  return %8, %15 : tensor<62x90xf32>, tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func @interleaved_extract_insert_slice_chain_1
func.func @interleaved_extract_insert_slice_chain_1(
    %arg2: tensor<62x90xf32> {bufferization.writable = true})
  -> (tensor<62x90xf32>)
{
  //      CHECK: tensor.extract_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true"]
  %2 = tensor.extract_slice %arg2[0, 0] [32, 90] [1, 1] : tensor<62x90xf32> to tensor<32x90xf32>

  // TODO: This should bufferize inplace once we have a proper range analysis.
  //      CHECK: tensor.extract_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["false"]
  %10 = tensor.extract_slice %arg2[32, 0] [30, 90] [1, 1] : tensor<62x90xf32> to tensor<30x90xf32>


  //      CHECK: tensor.insert_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true"]
  %8 = tensor.insert_slice %2 into %arg2[0, 0] [32, 90] [1, 1] : tensor<32x90xf32> into tensor<62x90xf32>


  //      CHECK: tensor.insert_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true"]
  %15 = tensor.insert_slice %10 into %8[32, 0] [30, 90] [1, 1] : tensor<30x90xf32> into tensor<62x90xf32>

  //      CHECK: return
  // CHECK-SAME: __equivalent_func_args__ = [0]
  return %15 : tensor<62x90xf32>
}

// -----

// CHECK-LABEL: func @interleaved_extract_insert_slice_chain_2
func.func @interleaved_extract_insert_slice_chain_2(
    %arg2: tensor<62x90xf32> {bufferization.writable = true})
  -> (tensor<62x90xf32>)
{
  //      CHECK: tensor.extract_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true"]
  %2 = tensor.extract_slice %arg2[0, 0] [32, 90] [1, 1] : tensor<62x90xf32> to tensor<32x90xf32>

  // The slices are overlapping, so this can never bufferize inplace.
  //      CHECK: tensor.extract_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["false"]
  %10 = tensor.extract_slice %arg2[31, 0] [30, 90] [1, 1] : tensor<62x90xf32> to tensor<30x90xf32>


  //      CHECK: tensor.insert_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true"]
  %8 = tensor.insert_slice %2 into %arg2[0, 0] [32, 90] [1, 1] : tensor<32x90xf32> into tensor<62x90xf32>


  //      CHECK: tensor.insert_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true"]
  %15 = tensor.insert_slice %10 into %8[31, 0] [30, 90] [1, 1] : tensor<30x90xf32> into tensor<62x90xf32>

  //      CHECK: return
  // CHECK-SAME: __equivalent_func_args__ = [0]
  return %15 : tensor<62x90xf32>
}

// -----

// CHECK-LABEL: func @extract_once_insert_twice
func.func @extract_once_insert_twice(
    %arg2: tensor<62x90xf32> {bufferization.writable = true})
  -> (tensor<62x90xf32>)
{
  //      CHECK: tensor.extract_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["false"]
  %2 = tensor.extract_slice %arg2[0, 0] [32, 90] [1, 1] : tensor<62x90xf32> to tensor<32x90xf32>

  //      CHECK: tensor.insert_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true"]
  %8 = tensor.insert_slice %2 into %arg2[0, 0] [32, 90] [1, 1] : tensor<32x90xf32> into tensor<62x90xf32>

  //      CHECK: tensor.insert_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true"]
  %15 = tensor.insert_slice %2 into %8[15, 0] [32, 90] [1, 1] : tensor<32x90xf32> into tensor<62x90xf32>

  //      CHECK: return
  // CHECK-SAME: __equivalent_func_args__ = [0]
  return %15 : tensor<62x90xf32>
}

// -----

// CHECK-LABEL: func @some_use
func.func @some_use(%A : tensor<?xf32> {bufferization.writable = true},
                    %v : vector<5xf32>) -> (tensor<?xf32>) {
  %idx = arith.constant 0 : index
  //      CHECK: vector.transfer_write
  // CHECK-SAME: {__inplace_operands_attr__ = ["none", "true", "none"]
  %0 = vector.transfer_write %v, %A[%idx] : vector<5xf32>, tensor<?xf32>
  return %0 : tensor<?xf32>
}


// CHECK-LABEL: func @main_func
func.func @main_func(%A : tensor<?xf32> {bufferization.writable = true},
                     %v : vector<5xf32>) -> (tensor<?xf32>) {
  //      CHECK: call
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "none"]
  %0 = call @some_use(%A, %v) : (tensor<?xf32>, vector<5xf32>) -> (tensor<?xf32>)
  return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @to_tensor_op_not_writable
func.func @to_tensor_op_not_writable(%m: memref<?xf32>, %v:  vector<5xf32>,
                                %idx1: index, %idx2: index)
    -> vector<10xf32> {
  %0 = bufferization.to_tensor %m : memref<?xf32>

  // Write to the tensor. Cannot be inplace due to tensor_load.
  //      CHECK: vector.transfer_write
  // CHECK-SAME: {__inplace_operands_attr__ = ["none", "false", "none"]
  %w = vector.transfer_write %v, %0[%idx1] : vector<5xf32>, tensor<?xf32>

  // Read from the tensor and return result.
  %cst = arith.constant 0.0 : f32
  %r = vector.transfer_read %w[%idx2], %cst : tensor<?xf32>, vector<10xf32>
  return %r : vector<10xf32>
}

// -----

// CHECK-LABEL: func @inner_func
func.func @inner_func(%t: tensor<?xf32>) -> tensor<?xf32> {
  //      CHECK: return
  // CHECK-SAME: __equivalent_func_args__ = [0]
  return %t : tensor<?xf32>
}

func.func @equivalent_func_arg(%c0: index, %c10: index, %c1: index, %t0: tensor<?xf32>) -> tensor<?xf32> {
  // This test does not check IR. It just asserts there is no failure due to
  // non-equivalent scf.for yield values.
  %1 = scf.for %iv = %c0 to %c10 step %c1 iter_args(%t1 = %t0) -> (tensor<?xf32>) {
    %3 = func.call @inner_func(%t1) : (tensor<?xf32>) -> tensor<?xf32>
    scf.yield %3 : tensor<?xf32>
  }
  return %1: tensor<?xf32>
}

// -----

// CHECK-LABEL: func @inner_func_2
func.func @inner_func_2(%t: tensor<?xf32>) -> tensor<?xf32> {
  %f = arith.constant 1.0 : f32
  %c0 = arith.constant 0 : index
  %0 = tensor.insert %f into %t[%c0] : tensor<?xf32>
  //      CHECK: return
  // CHECK-SAME: __equivalent_func_args__ = [0]
  return %0 : tensor<?xf32>
}

func.func @equivalent_func_arg_2(%c0: index, %c10: index, %c1: index, %t0: tensor<?xf32>) -> tensor<?xf32> {
  // This test does not check IR. It just asserts there is no failure due to
  // non-equivalent scf.for yield values.
  %1 = scf.for %iv = %c0 to %c10 step %c1 iter_args(%t1 = %t0) -> (tensor<?xf32>) {
    %3 = func.call @inner_func_2(%t1) : (tensor<?xf32>) -> tensor<?xf32>
    scf.yield %3 : tensor<?xf32>
  }
  return %1: tensor<?xf32>
}

// -----

// CHECK-LABEL: func @write_after_select_read_one
//  CHECK-SAME:     %[[t1:.*]]: tensor<?xf32> {{.*}}, %[[t2:.*]]: tensor<?xf32>
func.func @write_after_select_read_one(
    %t1 : tensor<?xf32> {bufferization.writable = true},
    %t2 : tensor<?xf32> {bufferization.writable = true},
    %c : i1)
  -> (f32, tensor<?xf32>)
{
  %cst = arith.constant 0.0 : f32
  %idx = arith.constant 0 : index

  //      CHECK: arith.select %{{.*}}, %[[t1]], %[[t2]]
  // CHECK-SAME:   {__inplace_operands_attr__ = ["none", "false", "true"]}
  %s = arith.select %c, %t1, %t2 : tensor<?xf32>
  //      CHECK: tensor.insert
  // CHECK-SAME:   {__inplace_operands_attr__ = ["none", "true", "none"]}
  %w = tensor.insert %cst into %s[%idx] : tensor<?xf32>
  //      CHECK: tensor.extract
  // CHECK-SAME:   {__inplace_operands_attr__ = ["true", "none"]}
  %f = tensor.extract %t1[%idx] : tensor<?xf32>

  return %f, %w : f32, tensor<?xf32>
}

// -----

// CHECK-LABEL: func @write_after_select_read_both
//  CHECK-SAME:     %[[t1:.*]]: tensor<?xf32> {{.*}}, %[[t2:.*]]: tensor<?xf32>
func.func @write_after_select_read_both(
    %t1 : tensor<?xf32> {bufferization.writable = true},
    %t2 : tensor<?xf32> {bufferization.writable = true},
    %c : i1)
  -> (f32, f32, tensor<?xf32>)
{
  %cst = arith.constant 0.0 : f32
  %idx = arith.constant 0 : index

  //      CHECK: arith.select %{{.*}}, %[[t1]], %[[t2]]
  // CHECK-SAME:   {__inplace_operands_attr__ = ["none", "false", "false"]}
  %s = arith.select %c, %t1, %t2 : tensor<?xf32>
  //      CHECK: tensor.insert
  // CHECK-SAME:   {__inplace_operands_attr__ = ["none", "true", "none"]}
  %w = tensor.insert %cst into %s[%idx] : tensor<?xf32>
  //      CHECK: tensor.extract
  // CHECK-SAME:   {__inplace_operands_attr__ = ["true", "none"]}
  %f = tensor.extract %t1[%idx] : tensor<?xf32>
  //      CHECK: tensor.extract
  // CHECK-SAME:   {__inplace_operands_attr__ = ["true", "none"]}
  %f2 = tensor.extract %t2[%idx] : tensor<?xf32>

  return %f, %f2, %w : f32, f32, tensor<?xf32>
}

// -----

// CHECK-LABEL: func @write_after_select_no_conflict
//  CHECK-SAME:     %[[t1:.*]]: tensor<?xf32> {{.*}}, %[[t2:.*]]: tensor<?xf32>
func.func @write_after_select_no_conflict(
    %t1 : tensor<?xf32> {bufferization.writable = true},
    %t2 : tensor<?xf32> {bufferization.writable = true},
    %c : i1)
  -> (f32, tensor<?xf32>)
{
  %cst = arith.constant 0.0 : f32
  %idx = arith.constant 0 : index

  //      CHECK: arith.select %{{.*}}, %[[t1]], %[[t2]]
  // CHECK-SAME:   {__inplace_operands_attr__ = ["none", "true", "true"]}
  %s = arith.select %c, %t1, %t2 : tensor<?xf32>
  //      CHECK: tensor.insert
  // CHECK-SAME:   {__inplace_operands_attr__ = ["none", "true", "none"]}
  %w = tensor.insert %cst into %s[%idx] : tensor<?xf32>
  //      CHECK: tensor.extract
  // CHECK-SAME:   {__inplace_operands_attr__ = ["true", "none"]}
  %f = tensor.extract %w[%idx] : tensor<?xf32>

  return %f, %w : f32, tensor<?xf32>
}

// -----

// CHECK-LABEL: func @write_to_same_tensor_in_loop_out_of_place(
func.func @write_to_same_tensor_in_loop_out_of_place(
    %A : tensor<?xf32> {bufferization.writable = true},
    %B : tensor<?xf32> {bufferization.writable = true},
    %lb : index, %ub : index, %step : index, %sz: index)
  -> (tensor<?xf32>)
{
  // CHECK: scf.for {{.*}} {
  %r0 = scf.for %i = %lb to %ub step %step iter_args(%t = %A) -> (tensor<?xf32>) {
    %i2 = arith.index_cast %i : index to i32
    %i3 = arith.sitofp %i2 : i32 to f32
    // The tensor.insert is out-of-place because the %B is written multiple
    // times inside a loop.
    //      CHECK: tensor.insert
    // CHECK-SAME:   {__inplace_operands_attr__ = ["none", "false", "none"]}
    %B2 = tensor.insert %i3 into %B[%i] : tensor<?xf32>
    //      CHECK: tensor.insert_slice
    // CHECK-SAME:   {__inplace_operands_attr__ = ["true", "true", "none", "none"]}
    %A2 = tensor.insert_slice %B2 into %t[%i][%sz][1] : tensor<?xf32> into tensor<?xf32>
    scf.yield %A2 : tensor<?xf32>
  }
  // CHECK: } {__inplace_operands_attr__ = ["none", "none", "none", "true"]}

  return %r0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @write_to_same_alloc_tensor_in_place(
func.func @write_to_same_alloc_tensor_in_place(
    %A : tensor<?xf32> {bufferization.writable = true},
    %lb : index, %ub : index, %step : index, %sz: index, %sz2: index)
  -> (tensor<?xf32>)
{
  %B = bufferization.alloc_tensor(%sz2) : tensor<?xf32>

  // CHECK: scf.for {{.*}} {
  %r0 = scf.for %i = %lb to %ub step %step iter_args(%t = %A) -> (tensor<?xf32>) {
    %i2 = arith.index_cast %i : index to i32
    %i3 = arith.sitofp %i2 : i32 to f32
    // %B is written multiple times inside a loop, but it is an alloc_tensor.
    //      CHECK: tensor.insert
    // CHECK-SAME:   {__inplace_operands_attr__ = ["none", "true", "none"]}
    %B2 = tensor.insert %i3 into %B[%i] : tensor<?xf32>
    //      CHECK: tensor.insert_slice
    // CHECK-SAME:   {__inplace_operands_attr__ = ["true", "true", "none", "none"]}
    %A2 = tensor.insert_slice %B2 into %t[%i][%sz][1] : tensor<?xf32> into tensor<?xf32>
    scf.yield %A2 : tensor<?xf32>
  }
  // CHECK: } {__inplace_operands_attr__ = ["none", "none", "none", "true"]}

  return %r0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @write_to_same_alloc_tensor_out_of_place(
func.func @write_to_same_alloc_tensor_out_of_place(
    %A : tensor<?xf32> {bufferization.writable = true},
    %lb : index, %ub : index, %step : index, %sz: index, %sz2: index, %f: f32)
  -> (tensor<?xf32>)
{
  %B = bufferization.alloc_tensor(%sz2) : tensor<?xf32>
  %C = tensor.insert %f into %B[%lb] : tensor<?xf32>

  // CHECK: scf.for {{.*}} {
  %r0 = scf.for %i = %lb to %ub step %step iter_args(%t = %A) -> (tensor<?xf32>) {
    %i2 = arith.index_cast %i : index to i32
    %i3 = arith.sitofp %i2 : i32 to f32
    // %C is written multiple times inside a loop. Even though %C aliases with
    // an alloc_tensor, out-of-bounds bufferization is necessary because there
    // is another alias (%C) outside of the loop.
    //      CHECK: tensor.insert
    // CHECK-SAME:   {__inplace_operands_attr__ = ["none", "false", "none"]}
    %B2 = tensor.insert %i3 into %C[%i] : tensor<?xf32>
    //      CHECK: tensor.insert_slice
    // CHECK-SAME:   {__inplace_operands_attr__ = ["true", "true", "none", "none"]}
    %A2 = tensor.insert_slice %B2 into %t[%i][%sz][1] : tensor<?xf32> into tensor<?xf32>
    scf.yield %A2 : tensor<?xf32>
  }
  // CHECK: } {__inplace_operands_attr__ = ["none", "none", "none", "true"]}

  return %r0 : tensor<?xf32>
}
