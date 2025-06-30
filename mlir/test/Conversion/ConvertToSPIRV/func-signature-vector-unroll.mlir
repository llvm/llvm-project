// RUN: mlir-opt -test-spirv-func-signature-conversion -split-input-file %s | FileCheck %s

// CHECK-LABEL: @simple_scalar
// CHECK-SAME: (%[[ARG0:.+]]: i32)
func.func @simple_scalar(%arg0 : i32) -> i32 {
  // CHECK: return %[[ARG0]] : i32
  return %arg0 : i32
}

// -----

// CHECK-LABEL: @simple_vector_4
// CHECK-SAME: (%[[ARG0:.+]]: vector<4xi32>)
func.func @simple_vector_4(%arg0 : vector<4xi32>) -> vector<4xi32> {
  // CHECK: return %[[ARG0]] : vector<4xi32>
  return %arg0 : vector<4xi32>
}

// -----

// CHECK-LABEL: @simple_vector_5
// CHECK-SAME: (%[[ARG0:.+]]: vector<1xi32>, %[[ARG1:.+]]: vector<1xi32>, %[[ARG2:.+]]: vector<1xi32>, %[[ARG3:.+]]: vector<1xi32>, %[[ARG4:.+]]: vector<1xi32>)
func.func @simple_vector_5(%arg0 : vector<5xi32>) -> vector<5xi32> {
  // CHECK: %[[CST:.*]] = arith.constant dense<0> : vector<5xi32>
  // CHECK: %[[INSERT0:.*]] = vector.insert_strided_slice %[[ARG0]], %[[CST]] {offsets = [0], strides = [1]} : vector<1xi32> into vector<5xi32>
  // CHECK: %[[INSERT1:.*]] = vector.insert_strided_slice %[[ARG1]], %[[INSERT0]] {offsets = [1], strides = [1]} : vector<1xi32> into vector<5xi32>
  // CHECK: %[[INSERT2:.*]] = vector.insert_strided_slice %[[ARG2]], %[[INSERT1]] {offsets = [2], strides = [1]} : vector<1xi32> into vector<5xi32>
  // CHECK: %[[INSERT3:.*]] = vector.insert_strided_slice %[[ARG3]], %[[INSERT2]] {offsets = [3], strides = [1]} : vector<1xi32> into vector<5xi32>
  // CHECK: %[[INSERT4:.*]] = vector.insert_strided_slice %[[ARG4]], %[[INSERT3]] {offsets = [4], strides = [1]} : vector<1xi32> into vector<5xi32>
  // CHECK: %[[EXTRACT0:.*]] = vector.extract_strided_slice %[[INSERT4]] {offsets = [0], sizes = [1], strides = [1]} : vector<5xi32> to vector<1xi32>
  // CHECK: %[[EXTRACT1:.*]] = vector.extract_strided_slice %[[INSERT4]] {offsets = [1], sizes = [1], strides = [1]} : vector<5xi32> to vector<1xi32>
  // CHECK: %[[EXTRACT2:.*]] = vector.extract_strided_slice %[[INSERT4]] {offsets = [2], sizes = [1], strides = [1]} : vector<5xi32> to vector<1xi32>
  // CHECK: %[[EXTRACT3:.*]] = vector.extract_strided_slice %[[INSERT4]] {offsets = [3], sizes = [1], strides = [1]} : vector<5xi32> to vector<1xi32>
  // CHECK: %[[EXTRACT4:.*]] = vector.extract_strided_slice %[[INSERT4]] {offsets = [4], sizes = [1], strides = [1]} : vector<5xi32> to vector<1xi32>
  // CHECK: return %[[EXTRACT0]], %[[EXTRACT1]], %[[EXTRACT2]], %[[EXTRACT3]], %[[EXTRACT4]] : vector<1xi32>, vector<1xi32>, vector<1xi32>, vector<1xi32>, vector<1xi32>
  return %arg0 : vector<5xi32>
}

// -----

// CHECK-LABEL: @simple_vector_6
// CHECK-SAME: (%[[ARG0:.+]]: vector<3xi32>, %[[ARG1:.+]]: vector<3xi32>)
func.func @simple_vector_6(%arg0 : vector<6xi32>) -> vector<6xi32> {
  // CHECK: %[[CST:.*]] = arith.constant dense<0> : vector<6xi32>
  // CHECK: %[[INSERT0:.*]] = vector.insert_strided_slice %[[ARG0]], %[[CST]] {offsets = [0], strides = [1]} : vector<3xi32> into vector<6xi32>
  // CHECK: %[[INSERT1:.*]] = vector.insert_strided_slice %[[ARG1]], %[[INSERT0]] {offsets = [3], strides = [1]} : vector<3xi32> into vector<6xi32>
  // CHECK: %[[EXTRACT0:.*]] = vector.extract_strided_slice %[[INSERT1]] {offsets = [0], sizes = [3], strides = [1]} : vector<6xi32> to vector<3xi32>
  // CHECK: %[[EXTRACT1:.*]] = vector.extract_strided_slice %[[INSERT1]] {offsets = [3], sizes = [3], strides = [1]} : vector<6xi32> to vector<3xi32>
  // CHECK: return %[[EXTRACT0]], %[[EXTRACT1]] : vector<3xi32>, vector<3xi32>
  return %arg0 : vector<6xi32>
}

// -----

// CHECK-LABEL: @simple_vector_8
// CHECK-SAME: (%[[ARG0:.+]]: vector<4xi32>, %[[ARG1:.+]]: vector<4xi32>)
func.func @simple_vector_8(%arg0 : vector<8xi32>) -> vector<8xi32> {
  // CHECK: %[[CST:.*]] = arith.constant dense<0> : vector<8xi32>
  // CHECK: %[[INSERT0:.*]] = vector.insert_strided_slice %[[ARG0]], %[[CST]] {offsets = [0], strides = [1]} : vector<4xi32> into vector<8xi32>
  // CHECK: %[[INSERT1:.*]] = vector.insert_strided_slice %[[ARG1]], %[[INSERT0]] {offsets = [4], strides = [1]} : vector<4xi32> into vector<8xi32>
  // CHECK: %[[EXTRACT0:.*]] = vector.extract_strided_slice %[[INSERT1]] {offsets = [0], sizes = [4], strides = [1]} : vector<8xi32> to vector<4xi32>
  // CHECK: %[[EXTRACT1:.*]] = vector.extract_strided_slice %[[INSERT1]] {offsets = [4], sizes = [4], strides = [1]} : vector<8xi32> to vector<4xi32>
  // CHECK: return %[[EXTRACT0]], %[[EXTRACT1]] : vector<4xi32>, vector<4xi32>
  return %arg0 : vector<8xi32>
}

// -----

// CHECK-LABEL: @simple_vector_2d
// CHECK-SAME: (%[[ARG0:.+]]: vector<4xi32>, %[[ARG1:.+]]: vector<4xi32>, %[[ARG2:.+]]: vector<4xi32>, %[[ARG3:.+]]: vector<4xi32>)
func.func @simple_vector_2d(%arg0 : vector<4x4xi32>) -> vector<4x4xi32> {
  // CHECK: %[[CST:.*]] = arith.constant dense<0> : vector<4x4xi32>
  // CHECK: %[[INSERT0:.*]] = vector.insert_strided_slice %[[ARG0]], %[[CST]] {offsets = [0, 0], strides = [1]} : vector<4xi32> into vector<4x4xi32>
  // CHECK: %[[INSERT1:.*]] = vector.insert_strided_slice %[[ARG1]], %[[INSERT0]] {offsets = [1, 0], strides = [1]} : vector<4xi32> into vector<4x4xi32>
  // CHECK: %[[INSERT2:.*]] = vector.insert_strided_slice %[[ARG2]], %[[INSERT1]] {offsets = [2, 0], strides = [1]} : vector<4xi32> into vector<4x4xi32>
  // CHECK: %[[INSERT3:.*]] = vector.insert_strided_slice %[[ARG3]], %[[INSERT2]] {offsets = [3, 0], strides = [1]} : vector<4xi32> into vector<4x4xi32>
  // CHECK: %[[EXTRACT0:.*]] = vector.extract_strided_slice %[[INSERT3]] {offsets = [0, 0], sizes = [1, 4], strides = [1, 1]} : vector<4x4xi32> to vector<1x4xi32>
  // CHECK: %[[EXTRACT0_1:.*]] = vector.extract %[[EXTRACT0]][0] : vector<4xi32> from vector<1x4xi32>
  // CHECK: %[[EXTRACT1:.*]] = vector.extract_strided_slice %[[INSERT3]] {offsets = [1, 0], sizes = [1, 4], strides = [1, 1]} : vector<4x4xi32> to vector<1x4xi32>
  // CHECK: %[[EXTRACT1_1:.*]] = vector.extract %[[EXTRACT1]][0] : vector<4xi32> from vector<1x4xi32>
  // CHECK: %[[EXTRACT2:.*]] = vector.extract_strided_slice %[[INSERT3]] {offsets = [2, 0], sizes = [1, 4], strides = [1, 1]} : vector<4x4xi32> to vector<1x4xi32>
  // CHECK: %[[EXTRACT2_1:.*]] = vector.extract %[[EXTRACT2]][0] : vector<4xi32> from vector<1x4xi32>
  // CHECK: %[[EXTRACT3:.*]] = vector.extract_strided_slice %[[INSERT3]] {offsets = [3, 0], sizes = [1, 4], strides = [1, 1]} : vector<4x4xi32> to vector<1x4xi32>
  // CHECK: %[[EXTRACT3_1:.*]] = vector.extract %[[EXTRACT3]][0] : vector<4xi32> from vector<1x4xi32>
  // CHECK: return %[[EXTRACT0_1]], %[[EXTRACT1_1]], %[[EXTRACT2_1]], %[[EXTRACT3_1]] : vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>
  return %arg0 : vector<4x4xi32>
}

// -----

// CHECK-LABEL: @vector_6and8
// CHECK-SAME: (%[[ARG0:.+]]: vector<3xi32>, %[[ARG1:.+]]: vector<3xi32>, %[[ARG2:.+]]: vector<4xi32>, %[[ARG3:.+]]: vector<4xi32>)
func.func @vector_6and8(%arg0 : vector<6xi32>, %arg1 : vector<8xi32>) -> (vector<6xi32>, vector<8xi32>) {
  // CHECK: %[[CST:.*]] = arith.constant dense<0> : vector<8xi32>
  // CHECK: %[[CST0:.*]] = arith.constant dense<0> : vector<6xi32>
  // CHECK: %[[INSERT0:.*]] = vector.insert_strided_slice %[[ARG0]], %[[CST0]] {offsets = [0], strides = [1]} : vector<3xi32> into vector<6xi32>
  // CHECK: %[[INSERT1:.*]] = vector.insert_strided_slice %[[ARG1]], %[[INSERT0]] {offsets = [3], strides = [1]} : vector<3xi32> into vector<6xi32>
  // CHECK: %[[INSERT2:.*]] = vector.insert_strided_slice %[[ARG2]], %[[CST]] {offsets = [0], strides = [1]} : vector<4xi32> into vector<8xi32>
  // CHECK: %[[INSERT3:.*]] = vector.insert_strided_slice %[[ARG3]], %[[INSERT2]] {offsets = [4], strides = [1]} : vector<4xi32> into vector<8xi32>
  // CHECK: %[[EXTRACT0:.*]] = vector.extract_strided_slice %[[INSERT1]] {offsets = [0], sizes = [3], strides = [1]} : vector<6xi32> to vector<3xi32>
  // CHECK: %[[EXTRACT1:.*]] = vector.extract_strided_slice %[[INSERT1]] {offsets = [3], sizes = [3], strides = [1]} : vector<6xi32> to vector<3xi32>
  // CHECK: %[[EXTRACT2:.*]] = vector.extract_strided_slice %[[INSERT3]] {offsets = [0], sizes = [4], strides = [1]} : vector<8xi32> to vector<4xi32>
  // CHECK: %[[EXTRACT3:.*]] = vector.extract_strided_slice %[[INSERT3]] {offsets = [4], sizes = [4], strides = [1]} : vector<8xi32> to vector<4xi32>
  // CHECK: return %[[EXTRACT0]], %[[EXTRACT1]], %[[EXTRACT2]], %[[EXTRACT3]] : vector<3xi32>, vector<3xi32>, vector<4xi32>, vector<4xi32>
  return %arg0, %arg1 : vector<6xi32>, vector<8xi32>
}

// -----

// CHECK-LABEL: @vector_3and8
// CHECK-SAME: (%[[ARG0:.+]]: vector<3xi32>, %[[ARG1:.+]]: vector<4xi32>, %[[ARG2:.+]]: vector<4xi32>)
func.func @vector_3and8(%arg0 : vector<3xi32>, %arg1 : vector<8xi32>) -> (vector<3xi32>, vector<8xi32>) {
  // CHECK: %[[CST:.*]] = arith.constant dense<0> : vector<8xi32>
  // CHECK: %[[INSERT0:.*]] = vector.insert_strided_slice %[[ARG1]], %[[CST]] {offsets = [0], strides = [1]} : vector<4xi32> into vector<8xi32>
  // CHECK: %[[INSERT1:.*]] = vector.insert_strided_slice %[[ARG2]], %[[INSERT0]] {offsets = [4], strides = [1]} : vector<4xi32> into vector<8xi32>
  // CHECK: %[[EXTRACT0:.*]] = vector.extract_strided_slice %[[INSERT1]] {offsets = [0], sizes = [4], strides = [1]} : vector<8xi32> to vector<4xi32>
  // CHECK: %[[EXTRACT1:.*]] = vector.extract_strided_slice %[[INSERT1]] {offsets = [4], sizes = [4], strides = [1]} : vector<8xi32> to vector<4xi32>
  // CHECK: return %[[ARG0]], %[[EXTRACT0]], %[[EXTRACT1]] : vector<3xi32>, vector<4xi32>, vector<4xi32>
  return %arg0, %arg1 : vector<3xi32>, vector<8xi32>
}

// -----

// CHECK-LABEL: @scalar_vector
// CHECK-SAME: (%[[ARG0:.+]]: vector<4xi32>, %[[ARG1:.+]]: vector<4xi32>, %[[ARG2:.+]]: vector<3xi32>, %[[ARG3:.+]]: i32)
func.func @scalar_vector(%arg0 : vector<8xi32>, %arg1 : vector<3xi32>, %arg2 : i32) -> (vector<8xi32>, vector<3xi32>, i32) {
  // CHECK: %[[CST:.*]] = arith.constant dense<0> : vector<8xi32>
  // CHECK: %[[INSERT0:.*]] = vector.insert_strided_slice %[[ARG0]], %[[CST]] {offsets = [0], strides = [1]} : vector<4xi32> into vector<8xi32>
  // CHECK: %[[INSERT1:.*]] = vector.insert_strided_slice %[[ARG1]], %[[INSERT0]] {offsets = [4], strides = [1]} : vector<4xi32> into vector<8xi32>
  // CHECK: %[[EXTRACT0:.*]] = vector.extract_strided_slice %[[INSERT1]] {offsets = [0], sizes = [4], strides = [1]} : vector<8xi32> to vector<4xi32>
  // CHECK: %[[EXTRACT1:.*]] = vector.extract_strided_slice %[[INSERT1]] {offsets = [4], sizes = [4], strides = [1]} : vector<8xi32> to vector<4xi32>
  // CHECK: return %[[EXTRACT0]], %[[EXTRACT1]], %[[ARG2]], %[[ARG3]] : vector<4xi32>, vector<4xi32>, vector<3xi32>, i32
  return %arg0, %arg1, %arg2 : vector<8xi32>, vector<3xi32>, i32
}

// -----

// CHECK-LABEL: @vector_2dand1d
// CHECK-SAME: (%[[ARG0:.+]]: vector<3xi32>, %[[ARG1:.+]]: vector<3xi32>, %[[ARG2:.+]]: vector<3xi32>, %[[ARG3:.+]]: vector<3xi32>, %[[ARG4:.+]]: vector<4xi32>)
func.func @vector_2dand1d(%arg0 : vector<2x6xi32>, %arg1 : vector<4xi32>) -> (vector<2x6xi32>, vector<4xi32>) {
  // CHECK: %[[CST:.*]] = arith.constant dense<0> : vector<2x6xi32>
  // CHECK: %[[INSERT0:.*]] = vector.insert_strided_slice %[[ARG0]], %[[CST]] {offsets = [0, 0], strides = [1]} : vector<3xi32> into vector<2x6xi32>
  // CHECK: %[[INSERT1:.*]] = vector.insert_strided_slice %[[ARG1]], %[[INSERT0]] {offsets = [0, 3], strides = [1]} : vector<3xi32> into vector<2x6xi32>
  // CHECK: %[[INSERT2:.*]] = vector.insert_strided_slice %[[ARG2]], %[[INSERT1]] {offsets = [1, 0], strides = [1]} : vector<3xi32> into vector<2x6xi32>
  // CHECK: %[[INSERT3:.*]] = vector.insert_strided_slice %[[ARG3]], %[[INSERT2]] {offsets = [1, 3], strides = [1]} : vector<3xi32> into vector<2x6xi32>
  // CHECK: %[[EXTRACT0:.*]]  = vector.extract_strided_slice %[[INSERT3]] {offsets = [0, 0], sizes = [1, 3], strides = [1, 1]} : vector<2x6xi32> to vector<1x3xi32>
  // CHECK: %[[EXTRACT0_1:.*]]  = vector.extract %[[EXTRACT0]][0] : vector<3xi32> from vector<1x3xi32>
  // CHECK: %[[EXTRACT1:.*]]  = vector.extract_strided_slice %[[INSERT3]] {offsets = [0, 3], sizes = [1, 3], strides = [1, 1]} : vector<2x6xi32> to vector<1x3xi32>
  // CHECK: %[[EXTRACT1_1:.*]]  = vector.extract %[[EXTRACT1]][0] : vector<3xi32> from vector<1x3xi32>
  // CHECK: %[[EXTRACT2:.*]]  = vector.extract_strided_slice %[[INSERT3]] {offsets = [1, 0], sizes = [1, 3], strides = [1, 1]} : vector<2x6xi32> to vector<1x3xi32>
  // CHECK: %[[EXTRACT2_1:.*]]  = vector.extract %[[EXTRACT2]][0] : vector<3xi32> from vector<1x3xi32>
  // CHECK: %[[EXTRACT3:.*]]  = vector.extract_strided_slice %[[INSERT3]] {offsets = [1, 3], sizes = [1, 3], strides = [1, 1]} : vector<2x6xi32> to vector<1x3xi32>
  // CHECK: %[[EXTRACT3_1:.*]]  = vector.extract %[[EXTRACT3]][0] : vector<3xi32> from vector<1x3xi32>
  // CHECK: return %[[EXTRACT0_1]], %[[EXTRACT1_1]], %[[EXTRACT2_1]], %[[EXTRACT3_1]], %[[ARG4]] : vector<3xi32>, vector<3xi32>, vector<3xi32>, vector<3xi32>, vector<4xi32>
  return %arg0, %arg1 : vector<2x6xi32>, vector<4xi32>
}

// -----

// CHECK-LABEL: @reduction
// CHECK-SAME: (%[[ARG0:.+]]: vector<4xi32>, %[[ARG1:.+]]: vector<4xi32>, %[[ARG2:.+]]: vector<4xi32>, %[[ARG3:.+]]: vector<4xi32>, %[[ARG4:.+]]: i32)
func.func @reduction(%arg0 : vector<8xi32>, %arg1 : vector<8xi32>, %arg2 : i32) -> (i32) {
  // CHECK: %[[CST:.*]] = arith.constant dense<0> : vector<8xi32>
  // CHECK: %[[INSERT0:.*]] = vector.insert_strided_slice %[[ARG0]], %[[CST]] {offsets = [0], strides = [1]} : vector<4xi32> into vector<8xi32>
  // CHECK: %[[INSERT1:.*]] = vector.insert_strided_slice %[[ARG1]], %[[INSERT0]] {offsets = [4], strides = [1]} : vector<4xi32> into vector<8xi32>
  // CHECK: %[[INSERT2:.*]] = vector.insert_strided_slice %[[ARG2]], %[[CST]] {offsets = [0], strides = [1]} : vector<4xi32> into vector<8xi32>
  // CHECK: %[[INSERT3:.*]] = vector.insert_strided_slice %[[ARG3]], %[[INSERT2]] {offsets = [4], strides = [1]} : vector<4xi32> into vector<8xi32>
  // CHECK: %[[ADDI:.*]] = arith.addi %[[INSERT1]], %[[INSERT3]] : vector<8xi32>
  // CHECK: %[[REDUCTION:.*]] = vector.reduction <add>, %[[ADDI]] : vector<8xi32> into i32
  // CHECK: %[[RET:.*]] = arith.addi %[[REDUCTION]], %[[ARG4]] : i32
  // CHECK: return %[[RET]] : i32
  %0 = arith.addi %arg0, %arg1 : vector<8xi32>
  %1 = vector.reduction <add>, %0 : vector<8xi32> into i32
  %2 = arith.addi %1, %arg2 : i32
  return %2 : i32
}

// -----

// CHECK-LABEL: func.func private @unsupported_decl(vector<8xi32>)
func.func private @unsupported_decl(vector<8xi32>)

// -----

// CHECK-LABEL: @unsupported_scalable
// CHECK-SAME: (%[[ARG0:.+]]: vector<[8]xi32>)
func.func @unsupported_scalable(%arg0 : vector<[8]xi32>) -> (vector<[8]xi32>) {
  // CHECK: return %[[ARG0]] : vector<[8]xi32>
  return %arg0 : vector<[8]xi32>
}

// -----

// Check that already legal function parameters are properly preserved across multiple blocks.

// CHECK-LABEL: func.func @legal_params_multiple_blocks_simple
// CHECK-SAME: (%[[ARG0:.+]]: i32, %[[ARG1:.+]]: i32) -> i32
func.func @legal_params_multiple_blocks_simple(%arg0: i32, %arg1: i32) -> i32 {
  // CHECK: %[[ADD0:.*]] = arith.addi %[[ARG0]], %[[ARG1]] : i32
  // CHECK: %[[ADD1:.*]] = arith.addi %[[ADD0]], %[[ARG1]] : i32
  // CHECK: return %[[ADD1]] : i32
  cf.br ^bb1(%arg0 : i32)
^bb1(%acc0: i32):
  %acc1_val = arith.addi %acc0, %arg1 : i32
  cf.br ^bb2(%acc1_val : i32)
^bb2(%acc1: i32):
  %acc2_val = arith.addi %acc1, %arg1 : i32
  cf.br ^bb3(%acc2_val : i32)
^bb3(%acc_final: i32):
  return %acc_final : i32
}

// -----

// Check that legal parameters and existing `vector.insert_strided_slice`s are properly preserved across multiple blocks.

// CHECK-LABEL: func.func @legal_params_with_vec_insert_multiple_blocks
// CHECK-SAME: (%[[ARG0:.+]]: i32, %[[ARG1:.+]]: i32, %[[ARG2:.+]]: vector<4xi32>) -> vector<4xi32>
func.func @legal_params_with_vec_insert_multiple_blocks(%arg0: i32, %arg1: i32, %arg2: vector<4xi32>) -> vector<4xi32> {
  // CHECK: %[[ADD0:.*]] = arith.addi %[[ARG0]], %[[ARG1]] : i32
  // CHECK: %[[ADD1:.*]] = arith.addi %[[ADD0]], %[[ARG1]] : i32
  // CHECK: %[[VEC1D:.*]] = vector.broadcast %[[ADD1]] : i32 to vector<1xi32>
  // CHECK: %[[VEC0:.*]] = vector.insert_strided_slice %[[VEC1D]], %[[ARG2]] {offsets = [1], strides = [1]} : vector<1xi32> into vector<4xi32>
  // CHECK: %[[VEC1:.*]] = vector.insert_strided_slice %[[VEC1D]], %[[VEC0]] {offsets = [2], strides = [1]} : vector<1xi32> into vector<4xi32>
  // CHECK: %[[RESULT:.*]] = vector.insert_strided_slice %[[VEC1D]], %[[VEC1]] {offsets = [3], strides = [1]} : vector<1xi32> into vector<4xi32>
  // CHECK: return %[[RESULT]] : vector<4xi32>
  cf.br ^bb1(%arg0 : i32)
^bb1(%acc0: i32):
  %acc1_val = arith.addi %acc0, %arg1 : i32
  cf.br ^bb2(%acc1_val : i32)
^bb2(%acc1: i32):
  %acc2_val = arith.addi %acc1, %arg1 : i32
  cf.br ^bb3(%acc2_val : i32)
^bb3(%acc_final: i32):
  %scalar_vec = vector.broadcast %acc_final : i32 to vector<1xi32>
  %vec0 = vector.insert_strided_slice %scalar_vec, %arg2 {offsets = [1], strides = [1]} : vector<1xi32> into vector<4xi32>
  %vec1 = vector.insert_strided_slice %scalar_vec, %vec0 {offsets = [2], strides = [1]} : vector<1xi32> into vector<4xi32>
  %result = vector.insert_strided_slice %scalar_vec, %vec1 {offsets = [3], strides = [1]} : vector<1xi32> into vector<4xi32>
  return %result : vector<4xi32>
}

// -----

// Check that already legal function parameters are preserved across a loop (which contains multiple blocks).

// CHECK-LABEL: @legal_params_for_loop
// CHECK-SAME: (%[[ARG0:.+]]: i32, %[[ARG1:.+]]: i32, %[[ARG2:.+]]: i32)
func.func @legal_params_for_loop(%arg0: i32, %arg1: i32, %arg2: i32) -> i32 {
  // CHECK: %[[CST0:.*]] = arith.constant 0 : index
  // CHECK: %[[CST1:.*]] = arith.constant 1 : index
  // CHECK: %[[UB:.*]] = arith.index_cast %[[ARG2]] : i32 to index
  // CHECK: %[[RESULT:.*]] = scf.for %[[STEP:.*]] = %[[CST0]] to %[[UB]] step %[[CST1]] iter_args(%[[ACC:.*]] = %[[ARG0]]) -> (i32) {
  // CHECK:   %[[ADD:.*]] = arith.addi %[[ACC]], %[[ARG1]] : i32
  // CHECK:   scf.yield %[[ADD]] : i32
  // CHECK: return %[[RESULT]] : i32
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %ub = arith.index_cast %arg2 : i32 to index
  %result = scf.for %i = %zero to %ub step %one iter_args(%acc = %arg0) -> (i32) {
    %new_acc = arith.addi %acc, %arg1 : i32
    scf.yield %new_acc : i32
  }
  return %result : i32
}
