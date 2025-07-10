// RUN: mlir-opt %s -split-input-file -test-vector-linearize -verify-diagnostics | FileCheck %s

// **--------------------------------------------------------**
//              Tests of vectorizable ops
//             [CollapseInnerVectorizable]
// **--------------------------------------------------------**

// Constant linearization happens here because of the vector.shape_cast folder.
// The linearization of math.sin and arith.addf happens because of the pattern, CollapseInnerVectorizable.

// CHECK-LABEL: linearize_constant_and_elementwise
//  CHECK-SAME: (%[[ORIG_ARG:.*]]: vector<2x2xf32>)
//   CHECK-DAG: %[[ARG:.*]] = vector.shape_cast %[[ORIG_ARG]] : vector<2x2xf32> to vector<4xf32>
//   CHECK-DAG: %[[CST:.*]] = arith.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : vector<4xf32>
//       CHECK: %[[SIN:.*]] = math.sin %[[ARG]] : vector<4xf32>
//       CHECK: %[[ADD:.*]] = arith.addf %[[SIN]], %[[CST]] : vector<4xf32>
//       CHECK: %[[RES:.*]] = vector.shape_cast %[[ADD]] : vector<4xf32> to vector<2x2xf32>
//       CHECK: return %[[RES]] : vector<2x2xf32>
func.func @linearize_constant_and_elementwise(%arg0: vector<2x2xf32>) -> vector<2x2xf32> {
  %0 = arith.constant dense<[[1., 2.], [3., 4.]]> : vector<2x2xf32>
  %1 = math.sin %arg0 : vector<2x2xf32>
  %2 = arith.addf %1, %0 :  vector<2x2xf32>
  return %2 : vector<2x2xf32>
}

// -----

// The pattern CollapseInnerVectorizable is applied twice (4D->3D, then 3D-2D).

func.func @linearize_elemenetwise_4D(%arg0 : vector<2x3x5x7xi8>, %arg1 : vector<2x3x5x7xi8>) -> vector<210xi8> {
  // CHECK-LABEL: linearize_elemenetwise_4D
  //  CHECK-SAME: (%[[ARG0:.*]]: vector<2x3x5x7xi8>, %[[ARG1:.*]]: vector<2x3x5x7xi8>) -> vector<210xi8> {
  //   CHECK-DAG: %[[SC0:.*]] = vector.shape_cast %[[ARG0]] : vector<2x3x5x7xi8> to vector<210xi8>
  //   CHECK-DAG: %[[SC1:.*]] = vector.shape_cast %[[ARG1]] : vector<2x3x5x7xi8> to vector<210xi8>
  //       CHECK: %[[ADD:.*]] = arith.addi %[[SC0]], %[[SC1]] : vector<210xi8>
  //       CHECK: return %[[ADD]] : vector<210xi8>
  %0 = arith.addi %arg0, %arg1 : vector<2x3x5x7xi8>
  %1 = vector.shape_cast %0 : vector<2x3x5x7xi8> to vector<210xi8>
  return %1 : vector<210xi8>
}

// -----

// Poison linearization happens here because of the vector.shape_cast folder.

// CHECK-LABEL: linearize_poison
//       CHECK: %[[POISON:.*]] = ub.poison : vector<4xf32>
//       CHECK: return %[[POISON]] : vector<4xf32>
func.func @linearize_poison() -> vector<4xf32> {
  %0 = ub.poison : vector<2x2xf32>
  %1 = vector.shape_cast %0 : vector<2x2xf32> to vector<4xf32>
  return %1 : vector<4xf32>
}

// -----

// Check that linearization does not happen if the operands of the vectorizable operation are not vectors.

// CHECK-LABEL: tensor_no_linearize
//       CHECK: %[[MULF:.*]] = arith.mulf %arg0, %arg1 : tensor<2x2xf32>
func.func @tensor_no_linearize(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>) {
   %0 = arith.mulf %arg0, %arg1 : tensor<2x2xf32>
   return %0, %arg0 : tensor<2x2xf32>, tensor<2x2xf32>
}

// -----

// Check that linearization happens as long as the combined dimensions have at most 1 scalable dimension.

// CHECK-LABEL: func.func @scalable_linearize(
//  CHECK-SAME: %[[ARG_0:.*]]: vector<2x[2]xf32>) -> vector<2x[2]xf32> {
//   CHECK-DAG: %[[SC:.*]]   = vector.shape_cast %[[ARG_0]] : vector<2x[2]xf32> to vector<[4]xf32>
//   CHECK-DAG: %[[CST:.*]]  = arith.constant dense<3.000000e+00> : vector<[4]xf32>
//       CHECK: %[[SIN:.*]]  = math.sin %[[SC]] : vector<[4]xf32>
//       CHECK: %[[ADDF:.*]] = arith.addf %[[SIN]], %[[CST]] : vector<[4]xf32>
//       CHECK: %[[RES:.*]]  = vector.shape_cast %[[ADDF]] : vector<[4]xf32> to vector<2x[2]xf32>
//       CHECK: return %[[RES]] : vector<2x[2]xf32>
func.func @scalable_linearize(%arg0: vector<2x[2]xf32>) -> vector<2x[2]xf32> {
  %0 = arith.constant dense<[[3., 3.], [3., 3.]]> : vector<2x[2]xf32>
  %1 = math.sin %arg0 : vector<2x[2]xf32>
  %2 = arith.addf %0, %1 : vector<2x[2]xf32>
  return %2 : vector<2x[2]xf32>
}

// -----

// In this case there are 2 scalable dimensions, these cannot be combined.

// CHECK-LABEL: func.func @scalable_no_linearize(
//  CHECK-SAME: %[[VAL_0:.*]]: vector<[2]x[2]xf32>) -> vector<[2]x[2]xf32> {
//       CHECK: %[[CST:.*]] = arith.constant dense<2.000000e+00> : vector<[2]x[2]xf32>
//       CHECK: %[[SIN:.*]] = math.sin %[[VAL_0]] : vector<[2]x[2]xf32>
//       CHECK: %[[RES:.*]] = arith.addf %[[SIN]], %[[CST]] : vector<[2]x[2]xf32>
//       CHECK: return %[[RES]] : vector<[2]x[2]xf32>
func.func @scalable_no_linearize(%arg0: vector<[2]x[2]xf32>) -> vector<[2]x[2]xf32> {
  %0 = arith.constant dense<[[2., 2.], [2., 2.]]> : vector<[2]x[2]xf32>
  %1 = math.sin %arg0 : vector<[2]x[2]xf32>
  %2 = arith.addf %1, %0 : vector<[2]x[2]xf32>
  return %2 : vector<[2]x[2]xf32>
}

// -----

// In this case, the innermost 2 dimensions can be combined, because only 1 of them is scalable.
// However, at a subsequent application of the pattern the innermost 2 dimensions are now both
// scalable, and so the pattern fails to collapse 2D -> 1D.

// CHECK-LABEL: func.func @scalable_partial_linearize(
//  CHECK-SAME: %[[VAL_0:.*]]: vector<[2]x[2]x4xi8>) -> vector<[2]x[2]x4xi8> {
//       CHECK: %[[SC:.*]] = vector.shape_cast %[[VAL_0]] : vector<[2]x[2]x4xi8> to vector<[2]x[8]xi8>
//       CHECK: %[[COS:.*]] = math.absi %[[SC]] : vector<[2]x[8]xi8>
//       CHECK: %[[RES:.*]] = vector.shape_cast %[[COS]] : vector<[2]x[8]xi8> to vector<[2]x[2]x4xi8>
//       CHECK: return %[[RES]] : vector<[2]x[2]x4xi8>
func.func @scalable_partial_linearize(%arg0: vector<[2]x[2]x4xi8>) -> vector<[2]x[2]x4xi8> {
  %0 = math.absi %arg0 : vector<[2]x[2]x4xi8>
  return %0 : vector<[2]x[2]x4xi8>
}

// -----

// Check that rank-0 vectors are not converted to rank-1.

// CHECK-LABEL: func.func @vector_0d
//       CHECK: %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<f32>
//       CHECK: return %[[CST]]
func.func @vector_0d() -> vector<f32> {
  %0 = arith.constant dense<0.> : vector<f32>
  return %0 : vector<f32>
}


// -----

// **--------------------------------------------------------**
//              Tests of vector.shuffle
//            [CollapseInnnerVectorShuffle]
// **--------------------------------------------------------**

// CHECK-LABEL: vector_shuffle_2D
//  CHECK-SAME: (%[[ORIG_ARG0:.*]]: vector<4x2xf32>, %[[ORIG_ARG1:.*]]: vector<4x2xf32>) -> vector<8x2xf32> {
//   CHECK-DAG: %[[ARG0:.*]] = vector.shape_cast %[[ORIG_ARG0]] : vector<4x2xf32> to vector<8xf32>
//   CHECK-DAG: %[[ARG1:.*]] = vector.shape_cast %[[ORIG_ARG1]] : vector<4x2xf32> to vector<8xf32>
//       CHECK: %[[SHUFFLE:.*]] = vector.shuffle %[[ARG0]], %[[ARG1]]
//  CHECK-SAME: [0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15] : vector<8xf32>, vector<8xf32>
//       CHECK: %[[RES:.*]]  = vector.shape_cast %[[SHUFFLE]] : vector<16xf32> to vector<8x2xf32>
//       CHECK: return %[[RES]] : vector<8x2xf32>
func.func @vector_shuffle_2D(%arg0: vector<4x2xf32>, %arg1: vector<4x2xf32>) -> vector<8x2xf32> {
  %0 = vector.shuffle %arg0, %arg1 [0, 4, 1, 5, 2, 6, 3, 7] : vector<4x2xf32>, vector<4x2xf32>
  return %0 : vector<8x2xf32>
}

// -----

// CHECK-LABEL: vector_shuffle_5D
//  CHECK-SAME: (%[[ARG0:.*]]: vector<2x1x2x1x2xf32>, %[[ARG1:.*]]: vector<2x1x2x1x2xf32>)
//   CHECK-DAG: %[[SC0:.*]] = vector.shape_cast %[[ARG0]] : vector<2x1x2x1x2xf32> to vector<8xf32>
//   CHECK-DAG: %[[SC1:.*]] = vector.shape_cast %[[ARG1]] : vector<2x1x2x1x2xf32> to vector<8xf32>
//       CHECK: %[[SHUFFLE:.*]] = vector.shuffle %[[SC0]], %[[SC1]]
//  CHECK-SAME: [12, 13, 14, 15, 8, 9, 10, 11, 0, 1, 2, 3] : vector<8xf32>, vector<8xf32>
//       CHECK: %[[RES:.*]] = vector.shape_cast %[[SHUFFLE]] : vector<12xf32> to vector<3x1x2x1x2xf32>
//       CHECK: return %[[RES]] : vector<3x1x2x1x2xf32>
func.func @vector_shuffle_5D(%arg0: vector<2x1x2x1x2xf32>, %arg1: vector<2x1x2x1x2xf32>) -> vector<3x1x2x1x2xf32> {
  %0 = vector.shuffle %arg0, %arg1 [3, 2, 0] : vector<2x1x2x1x2xf32>, vector<2x1x2x1x2xf32>
  return %0 : vector<3x1x2x1x2xf32>
}

// -----

// **--------------------------------------------------------**
//              Tests of vector.bitcast
//            [CollapseInnerVectorBitcast]
// **--------------------------------------------------------**

// CHECK-LABEL: vector_bitcast_2D
//  CHECK-SAME: %[[ARG_0:.*]]: vector<4x4xf32>
//       CHECK: %[[DOWNCAST:.*]] = vector.shape_cast %[[ARG_0]] : vector<4x4xf32> to vector<16xf32>
//       CHECK: %[[BITCAST:.*]]  = vector.bitcast %[[DOWNCAST]] : vector<16xf32> to vector<32xf16>
//       CHECK: %[[UPCAST:.*]]   = vector.shape_cast %[[BITCAST]] : vector<32xf16> to vector<4x8xf16>
func.func @vector_bitcast_2D(%arg0: vector<4x4xf32>) -> vector<4x8xf16> {
  %1 = vector.bitcast %arg0 : vector<4x4xf32> to vector<4x8xf16>
  return %1 : vector<4x8xf16>
}

// -----

// CHECK-LABEL: vector_bitcast_3D
//  CHECK-SAME: %[[ARG_0:.*]]: vector<10x4x2xf32>
//       CHECK: %[[DOWNCAST:.*]] = vector.shape_cast %[[ARG_0]] : vector<10x4x2xf32> to vector<80xf32>
//       CHECK: %[[BITCAST:.*]]  = vector.bitcast %[[DOWNCAST]] : vector<80xf32> to vector<160xf16>
//       CHECK: %[[UPCAST:.*]]   = vector.shape_cast %[[BITCAST]] : vector<160xf16> to vector<10x4x4xf16>
func.func @vector_bitcast_3D(%arg0: vector<10x4x2xf32>) -> vector<10x4x4xf16> {
  %1 = vector.bitcast %arg0 : vector<10x4x2xf32> to vector<10x4x4xf16>
  return %1 : vector<10x4x4xf16>
}

// -----

// CHECK-LABEL: vector_bitcast_scalable
//  CHECK-SAME: %[[ARG_0:.*]]: vector<4x[2]xf32>
//       CHECK: %[[DOWNCAST:.*]] = vector.shape_cast %[[ARG_0]] : vector<4x[2]xf32> to vector<[8]xf32>
//       CHECK: %[[BITCAST:.*]]  = vector.bitcast %[[DOWNCAST]] : vector<[8]xf32> to vector<[16]xf16>
//       CHECK: %[[UPCAST:.*]]   = vector.shape_cast %[[BITCAST]] : vector<[16]xf16> to vector<4x[4]xf16>
func.func @vector_bitcast_scalable(%arg0: vector<4x[2]xf32>) -> vector<4x[4]xf16> {
  %1 = vector.bitcast %arg0 : vector<4x[2]xf32> to vector<4x[4]xf16>
  return %1 : vector<4x[4]xf16>
}

// -----

// CHECK-LABEL: vector_bitcast_scalable_3D
//  CHECK-SAME: %[[ARG_0:.*]]: vector<1x3x[8]xi8>
//       CHECK: %[[DOWNCAST:.*]] = vector.shape_cast %[[ARG_0]] : vector<1x3x[8]xi8> to vector<[24]xi8>
//       CHECK: %[[BITCAST:.*]]  = vector.bitcast %[[DOWNCAST]] : vector<[24]xi8> to vector<[6]xi32>
//       CHECK: %[[UPCAST:.*]]   = vector.shape_cast %[[BITCAST]] : vector<[6]xi32> to vector<1x3x[2]xi32>
//       CHECK: return %[[UPCAST]] : vector<1x3x[2]xi32>
func.func @vector_bitcast_scalable_3D(%arg0: vector<1x3x[8]xi8>) -> vector<1x3x[2]xi32> {
  %1 = vector.bitcast %arg0 :  vector<1x3x[8]xi8> to vector<1x3x[2]xi32>
  return %1 : vector<1x3x[2]xi32>
}

// -----

// **--------------------------------------------------------**
//              Tests of vector.create_mask
//              [SqueezeCreateMaskUnitDims]
// **--------------------------------------------------------**

// CHECK-LABEL: linearize_create_mask_2D
//  CHECK-SAME: (%[[ARG0:.*]]: index, %[[ARG1:.*]]: index) -> vector<1x16xi1>
//       CHECK: %[[C0:.*]]        = arith.constant 0 : index
//       CHECK: %[[CMP:.*]]       = arith.cmpi sgt, %[[ARG0]], %[[C0]] : index
//       CHECK: %[[INDEXCAST:.*]] = arith.index_cast %[[CMP]] : i1 to index
//       CHECK: %[[MULI:.*]]      = arith.muli %[[INDEXCAST]], %[[ARG1]] : index
//       CHECK: %[[MASK_1D:.*]]   = vector.create_mask %[[MULI]] : vector<16xi1>
//       CHECK: %[[CAST:.*]]      = vector.shape_cast %[[MASK_1D]] : vector<16xi1> to vector<1x16xi1>
//       CHECK: return %[[CAST]] : vector<1x16xi1>
func.func @linearize_create_mask_2D(%arg0 : index, %arg1 : index) -> vector<1x16xi1> {
  %0 = vector.create_mask %arg0, %arg1 : vector<1x16xi1>
  return %0 : vector<1x16xi1>
}

// -----

// CHECK-LABEL: linearize_create_mask_4D
//  CHECK-SAME: (%[[ARG0:.*]]: index, %[[ARG1:.*]]: index) -> vector<1x16x1x1xi1> {
//       CHECK: %[[C0:.*]]        = arith.constant 0 : index
//       CHECK: %[[CMP0:.*]]      = arith.cmpi sgt, %[[ARG0]], %[[C0]] : index
//       CHECK: %[[CMP1:.*]]      = arith.cmpi sgt, %[[ARG0]], %[[C0]] : index
//       CHECK: %[[MULI:.*]]      = arith.muli %[[CMP0]], %[[CMP1]] : i1
//       CHECK: %[[INDEXCAST:.*]] = arith.index_cast %[[MULI]] : i1 to index
//       CHECK: %[[MULI2:.*]]     = arith.muli %[[INDEXCAST]], %[[ARG1]] : index
//       CHECK: %[[MASK_1D:.*]]   = vector.create_mask %[[MULI2]] : vector<16xi1>
//       CHECK: %[[CAST:.*]]      = vector.shape_cast %[[MASK_1D]] :
//  CHECK-SAME: vector<16xi1> to vector<1x16x1x1xi1>
//       CHECK: return %[[CAST]] : vector<1x16x1x1xi1>
func.func @linearize_create_mask_4D(%arg0 : index, %arg1 : index) -> vector<1x16x1x1xi1> {
  %cst1 = arith.constant 1 : index
  %0 = vector.create_mask %arg0, %arg1, %cst1, %arg0 : vector<1x16x1x1xi1>
  return %0 : vector<1x16x1x1xi1>
}

// -----

// If any of the indices to the vector.create_mask is 0, the arithmetic is greatly
// simplified, because the mask will always be all false.

// CHECK-LABEL: linearize_create_mask_4D_false
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[MASK_1D:.*]] = vector.create_mask %[[C0]] : vector<16xi1>
// CHECK: vector.shape_cast %[[MASK_1D]] : vector<16xi1> to vector<1x16x1x1xi1>
func.func @linearize_create_mask_4D_false(%arg0 : index, %arg1 : index) -> vector<1x16x1x1xi1> {
  %cst0 = arith.constant 0  : index
  %0 = vector.create_mask %arg0, %arg1, %cst0, %arg0 : vector<1x16x1x1xi1>
  return %0 : vector<1x16x1x1xi1>
}

// -----

// CHECK-LABEL: linearize_scalable_create_mask
//       CHECK: vector.create_mask {{%.*}} : vector<[16]xi1>
func.func @linearize_scalable_create_mask(%arg0 : index, %arg1 : index) -> vector<1x[16]xi1> {
  %0 = vector.create_mask %arg0, %arg1 : vector<1x[16]xi1>
  return %0 : vector<1x[16]xi1>
}

// -----

// The mask being created in this test has 2 dimensions that are not 1, so it is not linearized.

// CHECK-LABEL: negative_create_mask
//       CHECK: vector.create_mask {{.*}} vector<2x2xi1>
func.func @negative_create_mask(%arg0 : index, %arg1 : index) -> vector<2x2xi1> {
  %0 = vector.create_mask %arg0, %arg1 : vector<2x2xi1>
  return %0 : vector<2x2xi1>
}

// -----

// **--------------------------------------------------------**
//              Tests of scf.for
// **--------------------------------------------------------**

// This test illustrates how type conversion can be used to linearize structured
// op types.

// CHECK-LABEL: linearize_across_for
//       CHECK: scf.for {{.*}} -> (vector<4xi8>)
//       CHECK: arith.addi {{.*}} : vector<4xi8>
//       CHECK: scf.yield {{.*}} : vector<4xi8>
func.func @linearize_across_for(%arg0 : vector<4xi8>) -> vector<4xi8> {
  %0 = vector.shape_cast %arg0 : vector<4xi8> to vector<2x2xi8>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %1 = scf.for %i = %c0 to %c4 step %c1 iter_args(%arg1 = %0) -> (vector<2x2xi8>) {
    %2 = arith.addi %arg1, %0 : vector<2x2xi8>
    scf.yield %2 : vector<2x2xi8>
  }
  %3 = vector.shape_cast %1 : vector<2x2xi8> to vector<4xi8>
  return %3 : vector<4xi8>
}

// -----

// **--------------------------------------------------------**
//              Tests of vector.splat
//               [CollapseInnerSplat]
// **--------------------------------------------------------**

// CHECK-LABEL: linearize_vector_splat
//  CHECK-SAME: (%[[ARG:.*]]: i32) -> vector<8xi32>
//       CHECK: %[[SPLAT:.*]] = vector.splat %[[ARG]] : vector<8xi32>
//       CHECK: return %[[SPLAT]] : vector<8xi32>
func.func @linearize_vector_splat(%arg0: i32) -> vector<8xi32> {
  %0 = vector.splat %arg0 : vector<4x2xi32>
  %1 = vector.shape_cast %0 : vector<4x2xi32> to vector<8xi32>
  return %1 : vector<8xi32>
}

// -----

// CHECK-LABEL: linearize_scalable_vector_splat
//  CHECK-SAME: (%[[ARG:.*]]: i32) -> vector<[8]xi32>
//       CHECK: %[[SPLAT:.*]] = vector.splat %[[ARG]] : vector<[8]xi32>
//       CHECK: return %[[SPLAT]] : vector<[8]xi32>
func.func @linearize_scalable_vector_splat(%arg0: i32) -> vector<[8]xi32> {
  %0 = vector.splat %arg0 : vector<4x[2]xi32>
  %1 = vector.shape_cast %0 : vector<4x[2]xi32> to vector<[8]xi32>
  return %1 : vector<[8]xi32>
}


// **-------------------------------------------------------------------**
//                     Tests of vector.insert
//   [ConvertInsertToShuffle, CollapseInnerInsert, CollapseOuterInsert]
// **-------------------------------------------------------------------**

// -----

// vector.insert where the destination is 1D vector is always unchanged.

// CHECK-LABEL: insert_scalar_to_1D(
//  CHECK-SAME: %[[A0:.*]]: i8, %[[A1:.*]]: vector<4xi8>
//       CHECK: %[[IN0:.*]] = vector.insert %[[A0]], %[[A1]] [2] : i8 into vector<4xi8>
//       CHECK: return %[[IN0]] : vector<4xi8>
func.func @insert_scalar_to_1D(%arg0 : i8, %arg1 : vector<4xi8>) -> vector<4xi8>
{
  %inserted = vector.insert %arg0, %arg1[2] : i8 into vector<4xi8>
  return %inserted : vector<4xi8>
}

// -----

// vector.insert of scalar always becomes insert of scalar into 1-D vector.
//
// CHECK-LABEL: insert_scalar_to_2D(
//  CHECK-SAME: %[[A0:.*]]: i8, %[[A1:.*]]: vector<3x4xi8>
//       CHECK: %[[SC0:.*]] = vector.shape_cast %[[A1]] : vector<3x4xi8> to vector<12xi8>
//       CHECK: %[[IN0:.*]] = vector.insert %[[A0]], %[[SC0]] [9] : i8 into vector<12xi8>
//       CHECK: %[[SC1:.*]] = vector.shape_cast %[[IN0]] : vector<12xi8> to vector<3x4xi8>
//       CHECK: return %[[SC1]] : vector<3x4xi8>
func.func @insert_scalar_to_2D(%arg0 : i8, %arg1 : vector<3x4xi8>) -> vector<3x4xi8>
{
  %inserted = vector.insert %arg0, %arg1[2, 1] : i8 into vector<3x4xi8>
  return %inserted : vector<3x4xi8>
}

// -----

// Another test of inserting a scalar into a vector.

// CHECK-LABEL: insert_scalar_to_2D_f32
//  CHECK-SAME: (%[[DEST:.*]]: vector<2x4xf32>, %[[SRC:.*]]: f32) -> vector<2x4xf32> {
//       CHECK: %[[DEST_1D:.*]] = vector.shape_cast %[[DEST]] : vector<2x4xf32> to vector<8xf32>
//       CHECK: %[[INSERT_1D:.*]] = vector.insert %[[SRC]], %[[DEST_1D]] [6] : f32 into vector<8xf32>
//       CHECK: %[[RES:.*]] = vector.shape_cast %[[INSERT_1D]] : vector<8xf32> to vector<2x4xf32>
//       CHECK: return %[[RES]] : vector<2x4xf32>
func.func @insert_scalar_to_2D_f32(%arg0: vector<2x4xf32>, %arg1: f32) -> vector<2x4xf32> {

  %0 = vector.insert %arg1, %arg0[1, 2]: f32 into vector<2x4xf32>
  return %0 : vector<2x4xf32>
}

// -----

// vector.insert where the source isn't a scalar. This case: 1D -> 2D.

//    CHECK-LABEL: insert_1D_to_2D(
//      CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11]
func.func @insert_1D_to_2D(%arg0 : vector<4xi8>, %arg1 : vector<3x4xi8>) -> vector<3x4xi8> {
  %inserted = vector.insert %arg0, %arg1[1] : vector<4xi8> into vector<3x4xi8>
  return %inserted : vector<3x4xi8>
}

// -----

// CHECK-LABEL: insert_2D_to_3D
//  CHECK-SAME: (%[[DEST:.*]]: vector<2x8x4xf32>, %[[SRC:.*]]: vector<8x4xf32>) -> vector<2x8x4xf32> {
//   CHECK-DAG: %[[ARG_SRC:.*]] = vector.shape_cast %[[SRC]] : vector<8x4xf32> to vector<32xf32>
//   CHECK-DAG: %[[ARG_DEST:.*]] = vector.shape_cast %[[DEST]] : vector<2x8x4xf32> to vector<64xf32>
//       CHECK: %[[SHUFFLE:.*]] = vector.shuffle %[[ARG_DEST]], %[[ARG_SRC]]
//  CHECK-SAME: [64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87,
//  CHECK-SAME: 88, 89, 90, 91, 92, 93, 94, 95, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
//  CHECK-SAME: 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<64xf32>, vector<32xf32>
//       CHECK: %[[RES:.*]] = vector.shape_cast %[[SHUFFLE]] : vector<64xf32> to vector<2x8x4xf32>
//       CHECK: return %[[RES]] : vector<2x8x4xf32>
func.func @insert_2D_to_3D(%arg0: vector<2x8x4xf32>, %arg1: vector<8x4xf32>) -> vector<2x8x4xf32> {
  %0 = vector.insert %arg1, %arg0[0]: vector<8x4xf32> into vector<2x8x4xf32>
  return %0 : vector<2x8x4xf32>
}

// -----

//    CHECK-LABEL: insert_2D_to_4D(
//  CHECK-COUNT-2: shape_cast
//          CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19] :
//     CHECK-SAME: vector<16xi8>, vector<4xi8>
func.func @insert_2D_to_4D(%arg0 : vector<2x2xi8>, %arg1 : vector<2x2x2x2xi8>) -> vector<2x2x2x2xi8> {
  %inserted = vector.insert %arg0, %arg1[1, 1] : vector<2x2xi8> into vector<2x2x2x2xi8>
  return %inserted : vector<2x2x2x2xi8>
}

// -----

// CHECK-LABEL: func.func @insert_scalable(
//  CHECK-SAME: %[[ARG0:.*]]: vector<2x8x[4]xf32>, %[[ARG1:.*]]: vector<8x[4]xf32>) -> vector<2x8x[4]xf32> {
//   CHECK-DAG: %[[SHAPE_CAST0:.*]] = vector.shape_cast %[[ARG0]] : vector<2x8x[4]xf32> to vector<2x[32]xf32>
//   CHECK-DAG: %[[SHAPE_CAST1:.*]] = vector.shape_cast %[[ARG1]] : vector<8x[4]xf32> to vector<[32]xf32>
//       CHECK: %[[INSERT:.*]]      = vector.insert %[[SHAPE_CAST1]], %[[SHAPE_CAST0]] [0] : vector<[32]xf32> into vector<2x[32]xf32>
//       CHECK: %[[RESULT:.*]]      = vector.shape_cast %[[INSERT]] : vector<2x[32]xf32> to vector<2x8x[4]xf32>
//       CHECK: return %[[RESULT]] : vector<2x8x[4]xf32>
func.func @insert_scalable(%arg0: vector<2x8x[4]xf32>, %arg1: vector<8x[4]xf32>) -> vector<2x8x[4]xf32> {
  %0 = vector.insert %arg1, %arg0[0]: vector<8x[4]xf32> into vector<2x8x[4]xf32>
  return %0 : vector<2x8x[4]xf32>
}

// -----

// **--------------------------------------------------------------------------**
//                             Tests of vector.extract
//     [ConvertExtractToShuffle, CollapseInnerExtract, CollapseOuterExtract]
// **--------------------------------------------------------------------------**

// vector.extract where the source is 1D vector is always unchanged.

// CHECK-LABEL: extract_scalar_from_1D(
//  CHECK-SAME: %[[A0:.*]]: vector<4xi8>
//       CHECK: %[[EX0:.*]] = vector.extract %[[A0]][2] : i8 from vector<4xi8>
//       CHECK: return %[[EX0]] : i8
func.func @extract_scalar_from_1D(%arg0 : vector<4xi8>) -> i8 {
  %extracted = vector.extract %arg0[2] : i8 from vector<4xi8>
  return %extracted : i8
}

// -----

// CHECK-LABEL: extract_scalar_from_1D_dynamic
//   CHECK-NOT: vector.shuffle
//       CHECK: vector.extract
//   CHECK-NOT: vector.shuffle
func.func @extract_scalar_from_1D_dynamic(%idx : index) -> i32 {
  %cst = arith.constant dense<[1, 2, 3, 4]> : vector<4xi32>
  %0 = vector.extract %cst[%idx] : i32 from vector<4xi32>
  return %0 : i32
}

// -----


// CHECK-LABEL: extract_scalar_from_2D(
//  CHECK-SAME: %[[A0:.*]]: vector<12xi8>
//       CHECK: %[[EX0:.*]] = vector.extract %[[A0]][9] : i8 from vector<12xi8>
//       CHECK: return %[[EX0]] : i8
func.func @extract_scalar_from_2D(%arg0 : vector<12xi8>) -> i8 {
  %sc = vector.shape_cast %arg0 : vector<12xi8> to vector<3x4xi8>
  %extracted = vector.extract %sc[2, 1] : i8 from vector<3x4xi8>
  return %extracted : i8
}

// -----

// CHECK-LABEL: extract_1D_from_2D(
//       CHECK: vector.shuffle
//  CHECK-SAME: [4, 5, 6, 7] : vector<12xi8>, vector<12xi8>
func.func @extract_1D_from_2D(%arg0 : vector<3x4xi8>) -> vector<4xi8> {
  %extracted = vector.extract %arg0[1] : vector<4xi8> from vector<3x4xi8>
  return %extracted : vector<4xi8>
}

// -----

// CHECK-LABEL: extract_2D_from_3D
//  CHECK-SAME: (%[[ORIG_ARG:.*]]: vector<2x8x2xf32>) -> vector<8x2xf32> {
//       CHECK: %[[ARG:.*]]     = vector.shape_cast %[[ORIG_ARG]] : vector<2x8x2xf32> to vector<32xf32>
//       CHECK: %[[SHUFFLE:.*]] = vector.shuffle %[[ARG]], %[[ARG]]
//  CHECK-SAME: [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<32xf32>, vector<32xf32>
//       CHECK: %[[RES:.*]]     = vector.shape_cast %[[SHUFFLE]] : vector<16xf32> to vector<8x2xf32>
//       CHECK: return %[[RES]] : vector<8x2xf32>
func.func @extract_2D_from_3D(%arg0: vector<2x8x2xf32>) -> vector<8x2xf32> {
  %0 = vector.extract %arg0[1]: vector<8x2xf32> from vector<2x8x2xf32>
  return %0 : vector<8x2xf32>
}

// -----

//    CHECK-LABEL: extract_2D_from_4D(
//          CHECK: vector.shuffle
//     CHECK-SAME: [10, 11] : vector<24xi8>, vector<24xi8>
func.func @extract_2D_from_4D(%arg0 : vector<4x3x2x1xi8>) -> vector<2x1xi8> {
  %extracted = vector.extract %arg0[1, 2] : vector<2x1xi8> from vector<4x3x2x1xi8>
  return %extracted : vector<2x1xi8>
}

// -----

// In this test, the dynamic extract dimension prevents linearization all the way to a shuffle operation.
// The outermost 2 and the innermost 2 dimensions are linearized, however.

//    CHECK-LABEL: extract_2D_from_5D_dynamic(
//     CHECK-SAME: %[[ARG0:.*]]: vector<5x4x3x2x1xi8>, %[[IDX:.*]]: index) -> vector<2x1xi8> {
//          CHECK: %[[SC:.*]] = vector.shape_cast %[[ARG0]] : vector<5x4x3x2x1xi8> to vector<20x3x2xi8>
//          CHECK: %[[EXTRACT:.*]] = vector.extract %[[SC]][9, %[[IDX]]] : vector<2xi8> from vector<20x3x2xi8>
//          CHECK: %[[RES:.*]] = vector.shape_cast %[[EXTRACT]] : vector<2xi8> to vector<2x1xi8>
func.func @extract_2D_from_5D_dynamic(%arg0 : vector<5x4x3x2x1xi8>, %idx : index) -> vector<2x1xi8> {
  %extracted = vector.extract %arg0[2, 1, %idx] : vector<2x1xi8> from vector<5x4x3x2x1xi8>
  return %extracted : vector<2x1xi8>
}

// -----

// CHECK-LABEL: func.func @extract_scalable(
//  CHECK-SAME: %[[ARG0:.*]]: vector<2x8x[2]xf32>) -> vector<8x[2]xf32> {
//       CHECK: %[[SC:.*]]      = vector.shape_cast %[[ARG0]] : vector<2x8x[2]xf32> to vector<2x[16]xf32>
//       CHECK: %[[EXTRACT:.*]] = vector.extract %[[SC]][1] : vector<[16]xf32> from vector<2x[16]xf32>
//       CHECK: %[[RES:.*]]     = vector.shape_cast %[[EXTRACT]] : vector<[16]xf32> to vector<8x[2]xf32>
//       CHECK: return %[[RES]] : vector<8x[2]xf32>
func.func @extract_scalable(%arg0: vector<2x8x[2]xf32>) -> vector<8x[2]xf32> {
  %0 = vector.extract %arg0[1]: vector<8x[2]xf32> from vector<2x8x[2]xf32>
  return %0 : vector<8x[2]xf32>
}


// **------------------------------------------------------------**
//                 Tests of vector.insert_strided_slice
//    [ConvertInsertStridedToShuffle, CollapseInnerInsertStride]
// **------------------------------------------------------------**

// -----

// Test of insert_strided_slice -> shuffle.
// This is a contiguous insertion of 4 elements at offset 6 into a vector of 12 elements.
// CHECK-LABEL: insert_strided_slice_2D_into_4D
//   CHECK-DAG: %[[ARG0:.*]] = vector.shape_cast {{.*}}  to vector<4xi8>
//   CHECK-DAG: %[[ARG1:.*]] = vector.shape_cast {{.*}}  to vector<12xi8>
//       CHECK: vector.shuffle %[[ARG1]], %[[ARG0]]
//  CHECK-SAME: [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 10, 11] : vector<12xi8>, vector<4xi8>
//       CHECK: %[[RES:.*]]  = vector.shape_cast {{.*}} to vector<2x1x3x2xi8>
//       CHECK: return %[[RES]] : vector<2x1x3x2xi8>
func.func @insert_strided_slice_2D_into_4D(%arg0 : vector<2x2xi8>, %arg1 : vector<2x1x3x2xi8>) -> vector<2x1x3x2xi8> {
  %0 = vector.insert_strided_slice %arg0, %arg1 {offsets = [1, 0, 0, 0], strides = [1, 1]} : vector<2x2xi8> into vector<2x1x3x2xi8>
  return %0 : vector<2x1x3x2xi8>
}

// -----

// Test of insert_strided_slice -> shuffle.
// [[[0, 1], [2, 3], [4, 5]], [[6, 7], [8, 9], [10, 11]], [[12, 13], [14, 15]], [[16, 17]]]
//                                         ^         ^
//                                         |         |
//                          where the 2 elements are inserted into the 3x3x2 vector
// CHECK-LABEL: insert_strided_slice_3D
//   CHECK-DAG: %[[ARG0:.*]] = vector.shape_cast {{.*}}  to vector<2xi8>
//   CHECK-DAG: %[[ARG1:.*]] = vector.shape_cast {{.*}}  to vector<18xi8>
//       CHECK: vector.shuffle %[[ARG1]], %[[ARG0]]
//  CHECK-SAME: [0, 1, 2, 3, 4, 5, 6, 7, 8, 18, 10, 19, 12, 13, 14, 15, 16, 17] : vector<18xi8>, vector<2xi8>
//       CHECK: %[[RES:.*]]  = vector.shape_cast {{.*}} to vector<3x3x2xi8>
//       CHECK: return %[[RES]] : vector<3x3x2xi8>
func.func @insert_strided_slice_3D(%arg0 : vector<1x2x1xi8>, %arg1 : vector<3x3x2xi8>) -> vector<3x3x2xi8> {
  %0 = vector.insert_strided_slice %arg0, %arg1 {offsets = [1, 1, 1], sizes = [1, 2, 1], strides = [1, 1, 1]} : vector<1x2x1xi8> into vector<3x3x2xi8>
  return %0 : vector<3x3x2xi8>
}

// -----

// CHECK-LABEL: insert_strided_slice_2D_higher_offsets
//       CHECK: [0, 1, 2, 3, 10, 11, 12, 13, 8, 9]
//                           ^^^ ^^^ ^^^ ^^^
//                          insertion indices
//       CHECK: [0, 1, 2, 3, 10, 5, 11, 7, 8, 9]
//                           ^^^    ^^^
//                        insertion indices
//       CHECK: [0, 1, 2, 3, 4, 5, 6, 10, 8, 11]
//                                    ^^^    ^^^
//                                 insertion indices
func.func @insert_strided_slice_2D_higher_offsets(%arg0 : vector<2x1xi8>, %arg1 : vector<2x2xi8>, %arg2 : vector<5x2xi8>) -> vector<5x2xi8> {
  %0 = vector.insert_strided_slice %arg1, %arg2 {offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : vector<2x2xi8> into vector<5x2xi8>
  %1 = vector.insert_strided_slice %arg0, %0 {offsets = [2, 0], sizes = [2, 1], strides = [1, 1]} : vector<2x1xi8> into vector<5x2xi8>
  %2 = vector.insert_strided_slice %arg0, %1 {offsets = [3, 1], sizes = [2, 1], strides = [1, 1]} : vector<2x1xi8> into vector<5x2xi8>
  return %2 : vector<5x2xi8>
}

// -----

// CHECK-LABEL: negative_insert_strided_slice_scalable_shapes_differ
// CHECK-NOT:   vector.shuffle
// CHECK:       return
func.func @negative_insert_strided_slice_scalable_shapes_differ(%arg0 : vector<1x[2]xi8>, %arg1 : vector<2x[2]xi8>) -> vector<2x[2]xi8> {
  %0 = vector.insert_strided_slice %arg0, %arg1 {offsets = [0, 0], strides = [1,1]} : vector<1x[2]xi8> into vector<2x[2]xi8>
  return %0 : vector<2x[2]xi8>
}

// -----

// CHECK-LABEL: insert_strided_slice_scalable_common_pre
//  CHECK-SAME:  (%[[ARG0:.*]]: vector<3x1x[4]x2xi8>, %[[ARG1:.*]]: vector<3x1x[4]x5xi8>)
//   CHECK-DAG: %[[SC0:.*]] = vector.shape_cast %[[ARG0]] : vector<3x1x[4]x2xi8> to vector<[12]x2xi8>
//   CHECK-DAG: %[[SC1:.*]] = vector.shape_cast %[[ARG1]] : vector<3x1x[4]x5xi8> to vector<[12]x5xi8>
//       CHECK: %[[INSERT:.*]] = vector.insert_strided_slice %[[SC0]], %[[SC1]]
//  CHECK-SAME: {offsets = [0, 2], strides = [1, 1]} : vector<[12]x2xi8> into vector<[12]x5xi8>
//       CHECK: %[[RES:.*]] = vector.shape_cast %[[INSERT]] : vector<[12]x5xi8> to vector<3x1x[4]x5xi8>
//       CHECK: return %[[RES]] : vector<3x1x[4]x5xi8>

func.func @insert_strided_slice_scalable_common_pre(%arg0 : vector<3x1x[4]x2xi8>, %arg1 : vector<3x1x[4]x5xi8>) -> vector<3x1x[4]x5xi8> {
  %1 = vector.insert_strided_slice %arg0, %arg1 {offsets = [0, 0, 0, 2], strides = [1, 1, 1, 1]} : vector<3x1x[4]x2xi8> into vector<3x1x[4]x5xi8>
  return %1 : vector<3x1x[4]x5xi8>
}

// -----

// CHECK-LABEL: insert_strided_slice_scalable_common_post
//  CHECK-SAME: (%[[ARG0:.*]]: vector<1x[2]x3xi8>, %[[ARG1:.*]]: vector<5x[2]x3xi8>)
//   CHECK-DAG: %[[SC0:.*]] = vector.shape_cast %[[ARG0]] : vector<1x[2]x3xi8> to vector<1x[6]xi8>
//   CHECK-DAG: %[[SC1:.*]] = vector.shape_cast %[[ARG1]] : vector<5x[2]x3xi8> to vector<5x[6]xi8>
//       CHECK: %[[INSERT:.*]] = vector.insert_strided_slice %[[SC0]], %[[SC1]]
//  CHECK-SAME: {offsets = [3, 0], strides = [1, 1]} : vector<1x[6]xi8> into vector<5x[6]xi8>
//       CHECK: %[[RES:.*]] = vector.shape_cast %[[INSERT]] : vector<5x[6]xi8> to vector<5x[2]x3xi8>
//        CHECK: return %[[RES]] : vector<5x[2]x3xi8>

func.func @insert_strided_slice_scalable_common_post(%arg0 : vector<1x[2]x3xi8>, %arg1 : vector<5x[2]x3xi8>) -> vector<5x[2]x3xi8> {
  %1 = vector.insert_strided_slice %arg0, %arg1 {offsets = [3, 0, 0], strides = [1, 1, 1]} : vector<1x[2]x3xi8> into vector<5x[2]x3xi8>
  return %1 : vector<5x[2]x3xi8>
}

//  -----

// CHECK-LABEL: insert_strided_slice_1D(
//       CHECK: shuffle {{.*}} [0, 8, 9, 3, 4, 5, 6, 7]
func.func @insert_strided_slice_1D(%arg0 : vector<2xi8>, %arg1 : vector<8xi8>) -> vector<8xi8> {
  %inserted = vector.insert_strided_slice %arg0, %arg1 {offsets = [1], strides = [1]} : vector<2xi8> into vector<8xi8>
  return %inserted : vector<8xi8>
}

// -----

// CHECK-LABEL: insert_strided_slice_4D_contiguous(
//       CHECK: vector.shuffle
//  CHECK-SAME: 52, 53, 120, 121
//  CHECK-SAME: 130, 131, 66, 67
//  CHECK-SAME: vector<120xi8>, vector<12xi8>

func.func @insert_strided_slice_4D_contiguous(%arg0 : vector<1x2x2x3xi8>, %arg1 : vector<5x4x2x3xi8>) -> vector<5x4x2x3xi8> {
  %inserted = vector.insert_strided_slice %arg0, %arg1 {offsets = [2, 1, 0, 0], strides = [1, 1, 1, 1]} : vector<1x2x2x3xi8> into vector<5x4x2x3xi8>
  return %inserted : vector<5x4x2x3xi8>
}

// -----

// This insert_strided_slice is not contiguous, and so it is always linearized to a 1D vector.shuffle

// CHECK-LABEL: insert_strided_slice_4D_noncontiguous(
//       CHECK: vector.shuffle
//  CHECK-SAME: [0, 1, 2, 8, 4, 5, 6, 9] : vector<8xi8>, vector<2xi8>

func.func @insert_strided_slice_4D_noncontiguous(%arg0 : vector<1x2x1x1xi8>, %arg1 : vector<1x2x2x2xi8>) -> vector<1x2x2x2xi8> {
  %inserted = vector.insert_strided_slice %arg0, %arg1 {offsets = [0, 0, 1, 1], strides = [1, 1, 1, 1]} : vector<1x2x1x1xi8> into vector<1x2x2x2xi8>
  return %inserted : vector<1x2x2x2xi8>
}

// -----

// **----------------------------------------------------------------**
//              Tests of vector.extract_strided_slice
//    [ConvertExtractStridedToShuffle, CollapseInnerExtractStride]
// **----------------------------------------------------------------**

// CHECK-LABEL: extract_strided_slice_2D
//  CHECK-SAME: (%[[ORIG_ARG:.*]]: vector<4x8xf32>) -> vector<2x2xf32> {
//       CHECK: %[[ARG:.*]] = vector.shape_cast %[[ORIG_ARG]] : vector<4x8xf32> to vector<32xf32>
//       CHECK: %[[SHUFFLE:.*]] = vector.shuffle %[[ARG]], %[[ARG]]
//  CHECK-SAME: [4, 5, 12, 13] : vector<32xf32>, vector<32xf32>
//       CHECK: %[[RES:.*]] = vector.shape_cast %[[SHUFFLE]] : vector<4xf32> to vector<2x2xf32>
//       CHECK: return %[[RES]] : vector<2x2xf32
func.func @extract_strided_slice_2D(%arg0 : vector<4x8xf32>) -> vector<2x2xf32> {
  %0 = vector.extract_strided_slice %arg0 { sizes = [2, 2], strides = [1, 1], offsets = [0, 4]}
     : vector<4x8xf32> to vector<2x2xf32>
  return %0 : vector<2x2xf32>
}

// -----

// CHECK-LABEL: func.func @extract_strided_slice_2D_scalable(
//  CHECK-SAME: %[[VAL_0:.*]]: vector<4x[8]xf32>) -> vector<2x[8]xf32> {
//   CHECK-NOT: vector.shuffle
//   CHECK-NOT: vector.shape_cast
//       CHECK: %[[RES:.*]] = vector.extract_strided_slice %[[VAL_0]]
//       CHECK: return %[[RES]] : vector<2x[8]xf32>
func.func @extract_strided_slice_2D_scalable(%arg0: vector<4x[8]xf32>) -> vector<2x[8]xf32> {
  %0 = vector.extract_strided_slice %arg0 { sizes = [2, 8], strides = [1, 1], offsets = [1, 0] } : vector<4x[8]xf32> to vector<2x[8]xf32>
  return %0 : vector<2x[8]xf32>
}

// -----

// CHECK-LABEL: extract_strided_slice_3D
//  CHECK-SAME: (%[[ORIG_ARG:.*]]: vector<2x8x2xf32>) -> vector<1x4x2xf32> {
//       CHECK: %[[ARG:.*]] = vector.shape_cast %[[ORIG_ARG]] : vector<2x8x2xf32> to vector<32xf32>
//       CHECK: %[[SHUFFLE:.*]] = vector.shuffle %[[ARG]], %[[ARG]]
//  CHECK-SAME: [20, 21, 22, 23, 24, 25, 26, 27] : vector<32xf32>, vector<32xf32>
//       CHECK: %[[RES:.*]] = vector.shape_cast %[[SHUFFLE]] : vector<8xf32> to vector<1x4x2xf32>
//       CHECK: return %[[RES]] : vector<1x4x2xf32>
func.func @extract_strided_slice_3D(%arg0 : vector<2x8x2xf32>) -> vector<1x4x2xf32> {
  %0 = vector.extract_strided_slice %arg0 { offsets = [1, 2], strides = [1, 1], sizes = [1, 4] }
    : vector<2x8x2xf32> to vector<1x4x2xf32>
  return %0 : vector<1x4x2xf32>
}

// -----


// CHECK-LABEL: extract_strided_slice_1D(
//       CHECK: vector.shuffle {{.*}} [1, 2]
func.func @extract_strided_slice_1D(%arg0 : vector<8xi8>) -> vector<2xi8> {
  %extracted = vector.extract_strided_slice %arg0 {offsets = [1], sizes = [2], strides = [1]} : vector<8xi8> to vector<2xi8>
  return %extracted : vector<2xi8>
}

// -----

//CHECK-LABEL: extract_strided_slice_4D_contiguous_1(
//      CHECK: vector.shuffle
// CHECK-SAME: [3, 4, 5]
// CHECK-SAME: vector<6xi8>, vector<6xi8>
func.func @extract_strided_slice_4D_contiguous_1(%arg0 : vector<2x1x3x1xi8>) -> vector<1x1x3x1xi8> {
  %extracted = vector.extract_strided_slice %arg0 {offsets = [1, 0, 0, 0], sizes = [1, 1, 3, 1], strides = [1, 1, 1, 1]} : vector<2x1x3x1xi8> to vector<1x1x3x1xi8>
  return %extracted : vector<1x1x3x1xi8>
}

// -----

//CHECK-LABEL: extract_strided_slice_4D_contiguous_2(
//      CHECK: vector.shuffle
// CHECK-SAME: [3, 4]
// CHECK-SAME: vector<6xi8>, vector<6xi8>
func.func @extract_strided_slice_4D_contiguous_2(%arg0 : vector<2x1x3x1xi8>) -> vector<1x1x2x1xi8> {
  %extracted = vector.extract_strided_slice %arg0 {offsets = [1, 0, 0, 0], sizes = [1, 1, 2, 1], strides = [1, 1, 1, 1]} : vector<2x1x3x1xi8> to vector<1x1x2x1xi8>
  return %extracted : vector<1x1x2x1xi8>
}

// -----

// CHECK-LABEL: extract_strided_slice_4D_noncontiguous(
//       CHECK: vector.shuffle
//  CHECK-SAME: [0, 1, 3, 4]
//  CHECK-SAME: vector<6xi8>, vector<6xi8>
func.func @extract_strided_slice_4D_noncontiguous(%arg0 : vector<2x1x3x1xi8>) -> vector<2x1x2x1xi8> {
  %extracted = vector.extract_strided_slice %arg0 {offsets = [0, 0, 0, 0], sizes = [2, 1, 2, 1], strides = [1, 1, 1, 1]} : vector<2x1x3x1xi8> to vector<2x1x2x1xi8>
  return %extracted : vector<2x1x2x1xi8>
}
