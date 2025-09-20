// RUN: mlir-opt -test-spirv-vector-unrolling -split-input-file %s | FileCheck %s

// CHECK-LABEL: @vaddi
// CHECK-SAME: (%[[ARG0:.+]]: vector<3xi32>, %[[ARG1:.+]]: vector<3xi32>, %[[ARG2:.+]]: vector<3xi32>, %[[ARG3:.+]]: vector<3xi32>)
func.func @vaddi(%arg0 : vector<6xi32>, %arg1 : vector<6xi32>) -> (vector<6xi32>) {
  // CHECK: %[[ADD0:.*]] = arith.addi %[[ARG0]], %[[ARG2]] : vector<3xi32>
  // CHECK: %[[ADD1:.*]] = arith.addi %[[ARG1]], %[[ARG3]] : vector<3xi32>
  // CHECK: return %[[ADD0]], %[[ADD1]] : vector<3xi32>, vector<3xi32>
  %0 = arith.addi %arg0, %arg1 : vector<6xi32>
  return %0 : vector<6xi32>
}

// CHECK-LABEL: @vaddi_2d
// CHECK-SAME: (%[[ARG0:.+]]: vector<2xi32>, %[[ARG1:.+]]: vector<2xi32>, %[[ARG2:.+]]: vector<2xi32>, %[[ARG3:.+]]: vector<2xi32>)
func.func @vaddi_2d(%arg0 : vector<2x2xi32>, %arg1 : vector<2x2xi32>) -> (vector<2x2xi32>) {
  // CHECK: %[[ADD0:.*]] = arith.addi %[[ARG0]], %[[ARG2]] : vector<2xi32>
  // CHECK: %[[ADD1:.*]] = arith.addi %[[ARG1]], %[[ARG3]] : vector<2xi32>
  // CHECK: return %[[ADD0]], %[[ADD1]] : vector<2xi32>, vector<2xi32>
  %0 = arith.addi %arg0, %arg1 : vector<2x2xi32>
  return %0 : vector<2x2xi32>
}

// CHECK-LABEL: @vaddi_2d_8
// CHECK-SAME: (%[[ARG0:.+]]: vector<4xi32>, %[[ARG1:.+]]: vector<4xi32>, %[[ARG2:.+]]: vector<4xi32>, %[[ARG3:.+]]: vector<4xi32>, %[[ARG4:.+]]: vector<4xi32>, %[[ARG5:.+]]: vector<4xi32>, %[[ARG6:.+]]: vector<4xi32>, %[[ARG7:.+]]: vector<4xi32>)
func.func @vaddi_2d_8(%arg0 : vector<2x8xi32>, %arg1 : vector<2x8xi32>) -> (vector<2x8xi32>) {
  // CHECK: %[[ADD0:.*]] = arith.addi %[[ARG0]], %[[ARG4]] : vector<4xi32>
  // CHECK: %[[ADD1:.*]] = arith.addi %[[ARG1]], %[[ARG5]] : vector<4xi32>
  // CHECK: %[[ADD2:.*]] = arith.addi %[[ARG2]], %[[ARG6]] : vector<4xi32>
  // CHECK: %[[ADD3:.*]] = arith.addi %[[ARG3]], %[[ARG7]] : vector<4xi32>
  // CHECK: return %[[ADD0]], %[[ADD1]], %[[ADD2]], %[[ADD3]] : vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>
  %0 = arith.addi %arg0, %arg1 : vector<2x8xi32>
  return %0 : vector<2x8xi32>
}

// -----

// CHECK-LABEL: @reduction_5
// CHECK-SAME: (%[[ARG0:.+]]: vector<1xi32>, %[[ARG1:.+]]: vector<1xi32>, %[[ARG2:.+]]: vector<1xi32>, %[[ARG3:.+]]: vector<1xi32>, %[[ARG4:.+]]: vector<1xi32>)
func.func @reduction_5(%arg0 : vector<5xi32>) -> (i32) {
  // CHECK: %[[EXTRACT0:.*]] = vector.extract %[[ARG0]][0] : i32 from vector<1xi32>
  // CHECK: %[[EXTRACT1:.*]] = vector.extract %[[ARG1]][0] : i32 from vector<1xi32>
  // CHECK: %[[ADD0:.*]] = arith.addi %[[EXTRACT0]], %[[EXTRACT1]] : i32
  // CHECK: %[[EXTRACT2:.*]] = vector.extract %[[ARG2]][0] : i32 from vector<1xi32>
  // CHECK: %[[ADD1:.*]] = arith.addi %[[ADD0]], %[[EXTRACT2]] : i32
  // CHECK: %[[EXTRACT3:.*]] = vector.extract %[[ARG3]][0] : i32 from vector<1xi32>
  // CHECK: %[[ADD2:.*]] = arith.addi %[[ADD1]], %[[EXTRACT3]] : i32
  // CHECK: %[[EXTRACT4:.*]] = vector.extract %[[ARG4]][0] : i32 from vector<1xi32>
  // CHECK: %[[ADD3:.*]] = arith.addi %[[ADD2]], %[[EXTRACT4]] : i32
  // CHECK: return %[[ADD3]] : i32
  %0 = vector.reduction <add>, %arg0 : vector<5xi32> into i32
  return %0 : i32
}

// CHECK-LABEL: @reduction_8
// CHECK-SAME: (%[[ARG0:.+]]: vector<4xi32>, %[[ARG1:.+]]: vector<4xi32>)
func.func @reduction_8(%arg0 : vector<8xi32>) -> (i32) {
  // CHECK: %[[REDUCTION0:.*]] = vector.reduction <add>, %[[ARG0]] : vector<4xi32> into i32
  // CHECK: %[[REDUCTION1:.*]] = vector.reduction <add>, %[[ARG1]] : vector<4xi32> into i32
  // CHECK: %[[ADD:.*]] = arith.addi %[[REDUCTION0]], %[[REDUCTION1]] : i32
  // CHECK: return %[[ADD]] : i32
  %0 = vector.reduction <add>, %arg0 : vector<8xi32> into i32
  return %0 : i32
}

// -----

// CHECK-LABEL: @vaddi_reduction
// CHECK-SAME: (%[[ARG0:.+]]: vector<4xi32>, %[[ARG1:.+]]: vector<4xi32>, %[[ARG2:.+]]: vector<4xi32>, %[[ARG3:.+]]: vector<4xi32>)
func.func @vaddi_reduction(%arg0 : vector<8xi32>, %arg1 : vector<8xi32>) -> (i32) {
  // CHECK: %[[ADD0:.*]] = arith.addi %[[ARG0]], %[[ARG2]] : vector<4xi32>
  // CHECK: %[[ADD1:.*]] = arith.addi %[[ARG1]], %[[ARG3]] : vector<4xi32>
  // CHECK: %[[REDUCTION0:.*]] = vector.reduction <add>, %[[ADD0]] : vector<4xi32> into i32
  // CHECK: %[[REDUCTION1:.*]] = vector.reduction <add>, %[[ADD1]] : vector<4xi32> into i32
  // CHECK: %[[ADD2:.*]] = arith.addi %[[REDUCTION0]], %[[REDUCTION1]] : i32
  // CHECK: return %[[ADD2]] : i32
  %0 = arith.addi %arg0, %arg1 : vector<8xi32>
  %1 = vector.reduction <add>, %0 : vector<8xi32> into i32
  return %1 : i32
}

// -----

// CHECK-LABEL: @transpose
// CHECK-SAME: (%[[ARG0:.+]]: vector<3xi32>, %[[ARG1:.+]]: vector<3xi32>)
func.func @transpose(%arg0 : vector<2x3xi32>) -> (vector<3x2xi32>) {
  // CHECK: %[[EXTRACT0:.*]] = vector.extract %[[ARG0]][0] : i32 from vector<3xi32>
  // CHECK: %[[EXTRACT1:.*]] = vector.extract %[[ARG1]][0] : i32 from vector<3xi32>
  // CHECK: %[[FROM_ELEMENTS0:.*]] = vector.from_elements %[[EXTRACT0]], %[[EXTRACT1]] : vector<2xi32>
  // CHECK: %[[EXTRACT2:.*]] = vector.extract %[[ARG0]][1] : i32 from vector<3xi32>
  // CHECK: %[[EXTRACT3:.*]] = vector.extract %[[ARG1]][1] : i32 from vector<3xi32>
  // CHECK: %[[FROM_ELEMENTS1:.*]] = vector.from_elements %[[EXTRACT2]], %[[EXTRACT3]] : vector<2xi32>
  // CHECK: %[[EXTRACT4:.*]] = vector.extract %[[ARG0]][2] : i32 from vector<3xi32>
  // CHECK: %[[EXTRACT5:.*]] = vector.extract %[[ARG1]][2] : i32 from vector<3xi32>
  // CHECK: %[[FROM_ELEMENTS2:.*]] = vector.from_elements %[[EXTRACT4]], %[[EXTRACT5]] : vector<2xi32>
  // CHECK: return %[[FROM_ELEMENTS0]], %[[FROM_ELEMENTS1]], %[[FROM_ELEMENTS2]] : vector<2xi32>, vector<2xi32>, vector<2xi32>
  %0 = vector.transpose %arg0, [1, 0] : vector<2x3xi32> to vector<3x2xi32>
  return %0 : vector<3x2xi32>
}

// -----

// In order to verify that the pattern is applied,
// we need to make sure that the the 2d vector does not
// come from the parameters. Otherwise, the pattern
// in unrollVectorsInSignatures which splits the 2d vector
// parameter will take precedent. Similarly, let's avoid
// returning a vector as another pattern would take precendence.

// CHECK-LABEL: @unroll_to_elements_2d
func.func @unroll_to_elements_2d() -> (f32, f32, f32, f32) {
  %1 = "test.op"() : () -> (vector<2x2xf32>)
  // CHECK: %[[VEC2D:.+]] = "test.op"
  // CHECK: %[[VEC0:.+]] = vector.extract %[[VEC2D]][0] : vector<2xf32> from vector<2x2xf32>
  // CHECK: %[[VEC1:.+]] = vector.extract %[[VEC2D]][1] : vector<2xf32> from vector<2x2xf32>
  // CHECK: %[[RES0:.+]]:2 = vector.to_elements %[[VEC0]]
  // CHECK: %[[RES1:.+]]:2 = vector.to_elements %[[VEC1]]
  %2:4 = vector.to_elements %1 : vector<2x2xf32>
  return %2#0, %2#1, %2#2, %2#3 : f32, f32, f32, f32
}

// -----

// In order to verify that the pattern is applied,
// we need to make sure that the the 2d vector is used
// by an operation and that extracts are not folded away.
// In other words we can't use "test.op" nor return the
// value `%0 = vector.from_elements`

// CHECK-LABEL: @unroll_from_elements_2d
// CHECK-SAME: (%[[ARG0:.+]]: f32, %[[ARG1:.+]]: f32, %[[ARG2:.+]]: f32, %[[ARG3:.+]]: f32)
func.func @unroll_from_elements_2d(%arg0: f32, %arg1: f32, %arg2: f32, %arg3: f32) -> (vector<2x2xf32>) {
  // CHECK: %[[VEC0:.+]] = vector.from_elements %[[ARG0]], %[[ARG1]] : vector<2xf32>
  // CHECK: %[[VEC1:.+]] = vector.from_elements %[[ARG2]], %[[ARG3]] : vector<2xf32>
  %0 = vector.from_elements %arg0, %arg1, %arg2, %arg3 : vector<2x2xf32>

  // CHECK: %[[RES0:.+]] = arith.addf %[[VEC0]], %[[VEC0]]
  // CHECK: %[[RES1:.+]] = arith.addf %[[VEC1]], %[[VEC1]]
  %1 = arith.addf %0, %0 : vector<2x2xf32>

  // return %[[RES0]], %%[[RES1]] : vector<2xf32>, vector<2xf32>
  return %1 : vector<2x2xf32>
}
