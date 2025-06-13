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
  // CHECK: %[[UB:.*]] = ub.poison : vector<2xi32>
  // CHECK: %[[EXTRACT0:.*]] = vector.extract %[[ARG0]][0] : i32 from vector<3xi32>
  // CHECK: %[[INSERT0:.*]]= vector.insert %[[EXTRACT0]], %[[UB]] [0] : i32 into vector<2xi32>
  // CHECK: %[[EXTRACT1:.*]] = vector.extract %[[ARG1]][0] : i32 from vector<3xi32>
  // CHECK: %[[INSERT1:.*]] = vector.insert %[[EXTRACT1]], %[[INSERT0]][1] : i32 into vector<2xi32>
  // CHECK: %[[EXTRACT2:.*]] = vector.extract %[[ARG0]][1] : i32 from vector<3xi32>
  // CHECK: %[[INSERT2:.*]] = vector.insert %[[EXTRACT2]], %[[UB]] [0] : i32 into vector<2xi32>
  // CHECK: %[[EXTRACT3:.*]] = vector.extract %[[ARG1]][1] : i32 from vector<3xi32>
  // CHECK: %[[INSERT3:.*]] = vector.insert %[[EXTRACT3]], %[[INSERT2]] [1] : i32 into vector<2xi32>
  // CHECK: %[[EXTRACT4:.*]] = vector.extract %[[ARG0]][2] : i32 from vector<3xi32>
  // CHECK: %[[INSERT4:.*]] = vector.insert %[[EXTRACT4]], %[[UB]] [0] : i32 into vector<2xi32>
  // CHECK: %[[EXTRACT5:.*]] = vector.extract %[[ARG1]][2] : i32 from vector<3xi32>
  // CHECK: %[[INSERT5:.*]] = vector.insert %[[EXTRACT5]], %[[INSERT4]] [1] : i32 into vector<2xi32>
  // CHECK: return %[[INSERT1]], %[[INSERT3]], %[[INSERT5]] : vector<2xi32>, vector<2xi32>, vector<2xi32>
  %0 = vector.transpose %arg0, [1, 0] : vector<2x3xi32> to vector<3x2xi32>
  return %0 : vector<3x2xi32>
}
