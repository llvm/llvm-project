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

