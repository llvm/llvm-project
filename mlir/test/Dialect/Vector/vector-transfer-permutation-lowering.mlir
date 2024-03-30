// RUN: mlir-opt %s --transform-interpreter --split-input-file | FileCheck %s

// CHECK-LABEL: func @lower_permutation_with_mask_fixed_width(
//       CHECK:   %[[vec:.*]] = arith.constant dense<-2.000000e+00> : vector<7x1xf32>
//       CHECK:   %[[mask:.*]] = arith.constant dense<[true, false, true, false, true, true, true]> : vector<7xi1>
//       CHECK:   %[[b:.*]] = vector.broadcast %[[mask]] : vector<7xi1> to vector<1x7xi1>
//       CHECK:   %[[tp:.*]] = vector.transpose %[[b]], [1, 0] : vector<1x7xi1> to vector<7x1xi1>
//       CHECK:   vector.transfer_write %[[vec]], %{{.*}}[%{{.*}}, %{{.*}}], %[[tp]] {in_bounds = [false, true]} : vector<7x1xf32>, memref<?x?xf32>
func.func @lower_permutation_with_mask_fixed_width(%A : memref<?x?xf32>, %base1 : index,
                                       %base2 : index) {
  %fn1 = arith.constant -2.0 : f32
  %vf0 = vector.splat %fn1 : vector<7xf32>
  %mask = arith.constant dense<[1, 0, 1, 0, 1, 1, 1]> : vector<7xi1>
  vector.transfer_write %vf0, %A[%base1, %base2], %mask
    {permutation_map = affine_map<(d0, d1) -> (d0)>, in_bounds = [false]}
    : vector<7xf32>, memref<?x?xf32>
  return
}

// CHECK-LABEL:   func.func @permutation_with_mask_scalable(
// CHECK-SAME:      %[[ARG_0:.*]]: memref<?x?xf32>,
// CHECK-SAME:      %[[IDX_1:.*]]: index,
// CHECK-SAME:      %[[IDX_2:.*]]: index) -> vector<8x[4]x2xf32> {
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[PASS_THROUGH:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[MASK:.*]] = vector.create_mask %[[IDX_2]], %[[IDX_1]] : vector<2x[4]xi1>
// CHECK:           %[[T_READ:.*]] = vector.transfer_read %[[ARG_0]]{{\[}}%[[C0]], %[[C0]]], %[[PASS_THROUGH]], %[[MASK]] {in_bounds = [true, true]} : memref<?x?xf32>, vector<2x[4]xf32>
// CHECK:           %[[BCAST:.*]] = vector.broadcast %[[T_READ]] : vector<2x[4]xf32> to vector<8x2x[4]xf32>
// CHECK:           %[[TRANSPOSE:.*]] = vector.transpose %[[BCAST]], [0, 2, 1] : vector<8x2x[4]xf32> to vector<8x[4]x2xf32>
// CHECK:           return %[[TRANSPOSE]] : vector<8x[4]x2xf32>
// CHECK:         }
func.func @permutation_with_mask_scalable(%2: memref<?x?xf32>, %dim_1: index, %dim_2: index) -> (vector<8x[4]x2xf32>) {

  %c0 = arith.constant 0 : index
  %cst_0 = arith.constant 0.000000e+00 : f32

  %mask = vector.create_mask %dim_2, %dim_1 : vector<2x[4]xi1>
  %1 = vector.transfer_read %2[%c0, %c0], %cst_0, %mask
    {in_bounds = [true, true, true], permutation_map = affine_map<(d0, d1) -> (0, d1, d0)>}
    : memref<?x?xf32>, vector<8x[4]x2xf32>
  return %1 : vector<8x[4]x2xf32>
}

// CHECK:           func.func @permutation_with_mask_transfer_write_scalable(
// CHECK-SAME:        %[[ARG_0:.*]]: vector<4x[8]xi16>,
// CHECK-SAME:        %[[ARG_1:.*]]: memref<1x4x?x1x1x1x1xi16>,
// CHECK-SAME:        %[[MASK:.*]]: vector<4x[8]xi1>) {
// CHECK:             %[[C0:.*]] = arith.constant 0 : index
// CHECK:             %[[BCAST_1:.*]] = vector.broadcast %[[ARG_0]] : vector<4x[8]xi16> to vector<1x1x1x1x4x[8]xi16>
// CHECK:             %[[BCAST_2:.*]] = vector.broadcast %[[MASK]] : vector<4x[8]xi1> to vector<1x1x1x1x4x[8]xi1>
// CHECK:             %[[TRANSPOSE_1:.*]] = vector.transpose %[[BCAST_2]], [4, 5, 0, 1, 2, 3] : vector<1x1x1x1x4x[8]xi1> to vector<4x[8]x1x1x1x1xi1>
// CHECK:             %[[TRANSPOSE_2:.*]] = vector.transpose %[[BCAST_1]], [4, 5, 0, 1, 2, 3] : vector<1x1x1x1x4x[8]xi16> to vector<4x[8]x1x1x1x1xi16>
// CHECK:             vector.transfer_write %[[TRANSPOSE_2]], %[[ARG_1]]{{\[}}%[[C0]], %[[C0]], %[[C0]], %[[C0]], %[[C0]], %[[C0]], %[[C0]]], %[[TRANSPOSE_1]] {in_bounds = [true, true, true, true, true, true]} : vector<4x[8]x1x1x1x1xi16>, memref<1x4x?x1x1x1x1xi16>
// CHECK:             return
func.func @permutation_with_mask_transfer_write_scalable(%arg0: vector<4x[8]xi16>, %arg1: memref<1x4x?x1x1x1x1xi16>, %mask:  vector<4x[8]xi1>){
     %c0 = arith.constant 0 : index
      vector.transfer_write %arg0, %arg1[%c0, %c0, %c0, %c0, %c0, %c0, %c0], %mask {in_bounds = [true, true], permutation_map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d2)>
} : vector<4x[8]xi16>, memref<1x4x?x1x1x1x1xi16>

    return
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %module_op
      : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f {
      transform.apply_patterns.vector.transfer_permutation_patterns
    } : !transform.any_op
    transform.yield
  }
}
