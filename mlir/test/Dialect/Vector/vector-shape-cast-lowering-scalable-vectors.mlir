// RUN: mlir-opt %s --transform-interpreter | FileCheck %s

/// This tests that shape casts of scalable vectors (with one trailing scalable dim)
/// can be correctly lowered to vector.scalable.insert/extract.

// CHECK-LABEL: i32_3d_to_1d_last_dim_scalable
// CHECK-SAME: %[[arg0:.*]]: vector<2x1x[4]xi32>
func.func @i32_3d_to_1d_last_dim_scalable(%arg0: vector<2x1x[4]xi32>) -> vector<[8]xi32>
{
  // CHECK-NEXT: %[[ub:.*]] = ub.poison : vector<[8]xi32>
  // CHECK-NEXT: %[[subvec0:.*]] = vector.extract %[[arg0]][0, 0] : vector<[4]xi32> from vector<2x1x[4]xi32>
  // CHECK-NEXT: %[[res0:.*]] = vector.scalable.insert %[[subvec0]], %[[ub]][0] : vector<[4]xi32> into vector<[8]xi32>
  // CHECK-NEXT: %[[subvec1:.*]] = vector.extract %[[arg0]][1, 0] : vector<[4]xi32> from vector<2x1x[4]xi32>
  // CHECK-NEXT: %[[res1:.*]] = vector.scalable.insert %[[subvec1]], %[[res0]][4] : vector<[4]xi32> into vector<[8]xi32>
  %flat = vector.shape_cast %arg0 : vector<2x1x[4]xi32> to vector<[8]xi32>
  // CHECK-NEXT: return %[[res1]] : vector<[8]xi32>
  return %flat : vector<[8]xi32>
}

// -----

// CHECK-LABEL: i32_1d_to_3d_last_dim_scalable
// CHECK-SAME: %[[arg0:.*]]: vector<[8]xi32>
func.func @i32_1d_to_3d_last_dim_scalable(%arg0: vector<[8]xi32>) -> vector<2x1x[4]xi32> {
  // CHECK-NEXT: %[[ub:.*]] = ub.poison : vector<2x1x[4]xi32>
  // CHECK-NEXT: %[[subvec0:.*]] = vector.scalable.extract %[[arg0]][0] : vector<[4]xi32> from vector<[8]xi32>
  // CHECK-NEXT: %[[res0:.*]] = vector.insert %[[subvec0]], %[[ub]] [0, 0] : vector<[4]xi32> into vector<2x1x[4]xi32>
  // CHECK-NEXT: %[[subvec1:.*]] = vector.scalable.extract %[[arg0]][4] : vector<[4]xi32> from vector<[8]xi32>
  // CHECK-NEXT: %[[res1:.*]] = vector.insert %[[subvec1]], %[[res0]] [1, 0] : vector<[4]xi32> into vector<2x1x[4]xi32>
  %unflat = vector.shape_cast %arg0 : vector<[8]xi32> to vector<2x1x[4]xi32>
  // CHECK-NEXT: return %[[res1]] : vector<2x1x[4]xi32>
  return %unflat : vector<2x1x[4]xi32>
}

// -----

// CHECK-LABEL: i8_2d_to_1d_last_dim_scalable
// CHECK-SAME: %[[arg0:.*]]: vector<4x[8]xi8>
func.func @i8_2d_to_1d_last_dim_scalable(%arg0: vector<4x[8]xi8>) -> vector<[32]xi8> {
  // CHECK-NEXT: %[[ub:.*]] = ub.poison : vector<[32]xi8>
  // CHECK-NEXT: %[[subvec0:.*]] = vector.extract %[[arg0]][0] : vector<[8]xi8> from vector<4x[8]xi8>
  // CHECK-NEXT: %[[res0:.*]] = vector.scalable.insert %[[subvec0]], %[[ub]][0] : vector<[8]xi8> into vector<[32]xi8>
  // CHECK-NEXT: %[[subvec1:.*]] = vector.extract %[[arg0]][1] : vector<[8]xi8> from vector<4x[8]xi8>
  // CHECK-NEXT: %[[res1:.*]] = vector.scalable.insert %[[subvec1]], %[[res0]][8] : vector<[8]xi8> into vector<[32]xi8>
  // CHECK-NEXT: %[[subvec2:.*]] = vector.extract %[[arg0]][2] : vector<[8]xi8> from vector<4x[8]xi8>
  // CHECK-NEXT: %[[res2:.*]] = vector.scalable.insert %[[subvec2]], %[[res1]][16] : vector<[8]xi8> into vector<[32]xi8>
  // CHECK-NEXT: %[[subvec3:.*]] = vector.extract %[[arg0]][3] : vector<[8]xi8> from vector<4x[8]xi8>
  // CHECK-NEXT: %[[res3:.*]] = vector.scalable.insert %[[subvec3]], %[[res2]][24] : vector<[8]xi8> into vector<[32]xi8>
  %flat = vector.shape_cast %arg0 : vector<4x[8]xi8> to vector<[32]xi8>
  // CHECK-NEXT: return %[[res3]] : vector<[32]xi8>
  return %flat : vector<[32]xi8>
}

// -----

// CHECK-LABEL: i8_1d_to_2d_last_dim_scalable
// CHECK-SAME: %[[arg0:.*]]: vector<[32]xi8>
func.func @i8_1d_to_2d_last_dim_scalable(%arg0: vector<[32]xi8>) -> vector<4x[8]xi8> {
  // CHECK-NEXT: %[[ub:.*]] = ub.poison : vector<4x[8]xi8>
  // CHECK-NEXT: %[[subvec0:.*]] = vector.scalable.extract %[[arg0]][0] : vector<[8]xi8> from vector<[32]xi8>
  // CHECK-NEXT: %[[res0:.*]] = vector.insert %[[subvec0]], %[[ub]] [0] : vector<[8]xi8> into vector<4x[8]xi8>
  // CHECK-NEXT: %[[subvec1:.*]] = vector.scalable.extract %[[arg0]][8] : vector<[8]xi8> from vector<[32]xi8>
  // CHECK-NEXT: %[[res1:.*]] = vector.insert %[[subvec1]], %[[res0]] [1] : vector<[8]xi8> into vector<4x[8]xi8>
  // CHECK-NEXT: %[[subvec2:.*]] = vector.scalable.extract %[[arg0]][16] : vector<[8]xi8> from vector<[32]xi8>
  // CHECK-NEXT: %[[res2:.*]] = vector.insert %[[subvec2]], %[[res1]] [2] : vector<[8]xi8> into vector<4x[8]xi8>
  // CHECK-NEXT: %[[subvec3:.*]] = vector.scalable.extract %[[arg0]][24] : vector<[8]xi8> from vector<[32]xi8>
  // CHECK-NEXT: %[[res3:.*]] = vector.insert %[[subvec3]], %[[res2]] [3] : vector<[8]xi8> into vector<4x[8]xi8>
  %unflat = vector.shape_cast %arg0 : vector<[32]xi8> to vector<4x[8]xi8>
  // CHECK-NEXT: return %[[res3]] : vector<4x[8]xi8>
  return %unflat : vector<4x[8]xi8>
}

// -----

// CHECK-LABEL: f32_permute_leading_non_scalable_dims
// CHECK-SAME: %[[arg0:.*]]: vector<2x3x[4]xf32>
func.func @f32_permute_leading_non_scalable_dims(%arg0: vector<2x3x[4]xf32>) -> vector<3x2x[4]xf32> {
  // CHECK-NEXT: %[[ub:.*]] = ub.poison : vector<3x2x[4]xf32>
  // CHECK-NEXT: %[[subvec0:.*]] = vector.extract %[[arg0]][0, 0] : vector<[4]xf32> from vector<2x3x[4]xf32>
  // CHECK-NEXT: %[[res0:.*]] = vector.insert %[[subvec0]], %[[ub]] [0, 0] : vector<[4]xf32> into vector<3x2x[4]xf32>
  // CHECK-NEXT: %[[subvec1:.*]] = vector.extract %[[arg0]][0, 1] : vector<[4]xf32> from vector<2x3x[4]xf32>
  // CHECK-NEXT: %[[res1:.*]] = vector.insert %[[subvec1]], %[[res0]] [0, 1] : vector<[4]xf32> into vector<3x2x[4]xf32>
  // CHECK-NEXT: %[[subvec2:.*]] = vector.extract %[[arg0]][0, 2] : vector<[4]xf32> from vector<2x3x[4]xf32>
  // CHECK-NEXT: %[[res2:.*]] = vector.insert %[[subvec2]], %[[res1]] [1, 0] : vector<[4]xf32> into vector<3x2x[4]xf32>
  // CHECK-NEXT: %[[subvec3:.*]] = vector.extract %[[arg0]][1, 0] : vector<[4]xf32> from vector<2x3x[4]xf32>
  // CHECK-NEXT: %[[res3:.*]] = vector.insert %[[subvec3]], %[[res2]] [1, 1] : vector<[4]xf32> into vector<3x2x[4]xf32>
  // CHECK-NEXT: %[[subvec4:.*]] = vector.extract %[[arg0]][1, 1] : vector<[4]xf32> from vector<2x3x[4]xf32>
  // CHECK-NEXT: %[[res4:.*]] = vector.insert %[[subvec4]], %[[res3]] [2, 0] : vector<[4]xf32> into vector<3x2x[4]xf32>
  // CHECK-NEXT: %[[subvec5:.*]] = vector.extract %[[arg0]][1, 2] : vector<[4]xf32> from vector<2x3x[4]xf32>
  // CHECK-NEXT: %[[res5:.*]] = vector.insert %[[subvec5]], %[[res4]] [2, 1] : vector<[4]xf32> into vector<3x2x[4]xf32>
  %res = vector.shape_cast %arg0: vector<2x3x[4]xf32> to vector<3x2x[4]xf32>
  // CHECK-NEXT: return %[[res5]] : vector<3x2x[4]xf32>
  return %res : vector<3x2x[4]xf32>
}

// -----

// CHECK-LABEL: f64_flatten_leading_non_scalable_dims
// CHECK-SAME: %[[arg0:.*]]: vector<2x2x[2]xf64>
func.func @f64_flatten_leading_non_scalable_dims(%arg0: vector<2x2x[2]xf64>) -> vector<4x[2]xf64>
{
  // CHECK-NEXT: %[[ub:.*]] = ub.poison : vector<4x[2]xf64>
  // CHECK-NEXT: %[[subvec0:.*]] = vector.extract %[[arg0]][0, 0] : vector<[2]xf64> from vector<2x2x[2]xf64>
  // CHECK-NEXT: %[[res0:.*]] = vector.insert %[[subvec0]], %[[ub]] [0] : vector<[2]xf64> into vector<4x[2]xf64>
  // CHECK-NEXT: %[[subvec1:.*]] = vector.extract %[[arg0]][0, 1] : vector<[2]xf64> from vector<2x2x[2]xf64>
  // CHECK-NEXT: %[[res1:.*]] = vector.insert %[[subvec1]], %[[res0]] [1] : vector<[2]xf64> into vector<4x[2]xf64>
  // CHECK-NEXT: %[[subvec2:.*]] = vector.extract %[[arg0]][1, 0] : vector<[2]xf64> from vector<2x2x[2]xf64>
  // CHECK-NEXT: %[[res2:.*]] = vector.insert %[[subvec2]], %[[res1]] [2] : vector<[2]xf64> into vector<4x[2]xf64>
  // CHECK-NEXT: %[[subvec3:.*]] = vector.extract %[[arg0]][1, 1] : vector<[2]xf64> from vector<2x2x[2]xf64>
  // CHECK-NEXT: %[[res3:.*]] = vector.insert %[[subvec3]], %[[res2]] [3] : vector<[2]xf64> into vector<4x[2]xf64>
  %res = vector.shape_cast %arg0: vector<2x2x[2]xf64> to vector<4x[2]xf64>
  // CHECK-NEXT: return %[[res3:.*]] : vector<4x[2]xf64>
  return %res : vector<4x[2]xf64>
}

// -----

// CHECK-LABEL: f32_reduce_trailing_scalable_dim
// CHECK-SAME: %[[arg0:.*]]: vector<3x[4]xf32>
func.func @f32_reduce_trailing_scalable_dim(%arg0: vector<3x[4]xf32>) -> vector<6x[2]xf32>
{
  // CHECK-NEXT: %[[ub:.*]] = ub.poison : vector<6x[2]xf32>
  // CHECK-NEXT: %[[srcvec0:.*]] = vector.extract %[[arg0]][0] : vector<[4]xf32> from vector<3x[4]xf32>
  // CHECK-NEXT: %[[subvec0:.*]] = vector.scalable.extract %[[srcvec0]][0] : vector<[2]xf32> from vector<[4]xf32>
  // CHECK-NEXT: %[[res0:.*]] = vector.insert %[[subvec0]], %[[ub]] [0] : vector<[2]xf32> into vector<6x[2]xf32>
  // CHECK-NEXT: %[[subvec1:.*]] = vector.scalable.extract %[[srcvec0]][2] : vector<[2]xf32> from vector<[4]xf32>
  // CHECK-NEXT: %[[res1:.*]] = vector.insert %[[subvec1]], %[[res0]] [1] : vector<[2]xf32> into vector<6x[2]xf32>
  // CHECK-NEXT: %[[srcvec1:.*]] = vector.extract %[[arg0]][1] : vector<[4]xf32> from vector<3x[4]xf32>
  // CHECK-NEXT: %[[subvec2:.*]] = vector.scalable.extract %[[srcvec1]][0] : vector<[2]xf32> from vector<[4]xf32>
  // CHECK-NEXT: %[[res2:.*]] = vector.insert %[[subvec2]], %[[res1]] [2] : vector<[2]xf32> into vector<6x[2]xf32>
  // CHECK-NEXT: %[[subvec3:.*]] = vector.scalable.extract %[[srcvec1]][2] : vector<[2]xf32> from vector<[4]xf32>
  // CHECK-NEXT: %[[res3:.*]] = vector.insert %[[subvec3]], %[[res2]] [3] : vector<[2]xf32> into vector<6x[2]xf32>
  // CHECK-NEXT: %[[srcvec2:.*]] = vector.extract %[[arg0]][2] : vector<[4]xf32> from vector<3x[4]xf32>
  // CHECK-NEXT: %[[subvec4:.*]] = vector.scalable.extract %[[srcvec2]][0] : vector<[2]xf32> from vector<[4]xf32>
  // CHECK-NEXT: %[[res4:.*]] = vector.insert %[[subvec4]], %[[res3]] [4] : vector<[2]xf32> into vector<6x[2]xf32>
  // CHECK-NEXT: %[[subvec5:.*]] = vector.scalable.extract %[[srcvec2]][2] : vector<[2]xf32> from vector<[4]xf32>
  // CHECK-NEXT: %[[res5:.*]] = vector.insert %[[subvec5]], %[[res4]] [5] : vector<[2]xf32> into vector<6x[2]xf32>
  %res = vector.shape_cast %arg0: vector<3x[4]xf32> to vector<6x[2]xf32>
  // CHECK-NEXT: return %[[res5]] : vector<6x[2]xf32>
  return %res: vector<6x[2]xf32>
}

// -----

// CHECK-LABEL: f32_increase_trailing_scalable_dim
// CHECK-SAME: %[[arg0:.*]]: vector<4x[2]xf32>
func.func @f32_increase_trailing_scalable_dim(%arg0: vector<4x[2]xf32>) -> vector<2x[4]xf32>
{
  // CHECK-NEXT: %[[ub:.*]] = ub.poison : vector<2x[4]xf32>
  // CHECK-NEXT: %[[subvec0:.*]] = vector.extract %[[arg0]][0] : vector<[2]xf32> from vector<4x[2]xf32>
  // CHECK-NEXT: %[[resvec0:.*]] = vector.extract %[[ub]][0] : vector<[4]xf32> from vector<2x[4]xf32>
  // CHECK-NEXT: %[[resvec1:.*]] = vector.scalable.insert %[[subvec0]], %[[resvec0]][0] : vector<[2]xf32> into vector<[4]xf32>
  // CHECK-NEXT: %[[subvec1:.*]] = vector.extract %[[arg0]][1] : vector<[2]xf32> from vector<4x[2]xf32>
  // CHECK-NEXT: %[[resvec2:.*]] = vector.scalable.insert %[[subvec1]], %[[resvec1]][2] : vector<[2]xf32> into vector<[4]xf32>
  // CHECK-NEXT: %[[res0:.*]] = vector.insert %[[resvec2]], %[[ub]] [0] : vector<[4]xf32> into vector<2x[4]xf32>
  // CHECK-NEXT: %[[subvec3:.*]] = vector.extract %[[arg0]][2] : vector<[2]xf32> from vector<4x[2]xf32>
  // CHECK-NEXT: %[[resvec3:.*]] = vector.extract %[[ub]][1] : vector<[4]xf32> from vector<2x[4]xf32>
  // CHECK-NEXT: %[[resvec4:.*]] = vector.scalable.insert %[[subvec3]], %[[resvec3]][0] : vector<[2]xf32> into vector<[4]xf32>
  // CHECK-NEXT: %[[subvec4:.*]] = vector.extract %[[arg0]][3] : vector<[2]xf32> from vector<4x[2]xf32>
  // CHECK-NEXT: %[[resvec5:.*]] = vector.scalable.insert %[[subvec4]], %[[resvec4]][2] : vector<[2]xf32> into vector<[4]xf32>
  // CHECK-NEXT: %[[res1:.*]] = vector.insert %[[resvec5]], %[[res0]] [1] : vector<[4]xf32> into vector<2x[4]xf32>
  %res = vector.shape_cast %arg0: vector<4x[2]xf32> to vector<2x[4]xf32>
  // CHECK-NEXT: return %[[res1]] : vector<2x[4]xf32>
  return %res: vector<2x[4]xf32>
}

// -----

/// The following shape_casts are not supported as the types cannot be
/// represented in LLVM (and likely won't be supported soon), and currently
/// there's no ops that could do the extracts/inserts required.

// -----

// CHECK-LABEL: cannot_cast_to_non_trailing_scalable_dim
// CHECK-SAME: %[[arg0:.*]]: vector<[4]xf32>
func.func @cannot_cast_to_non_trailing_scalable_dim(%arg0: vector<[4]xf32>) -> vector<[2]x2xf32> {
  // CHECK-NEXT: %[[res:.*]] = vector.shape_cast %[[arg0]] : vector<[4]xf32> to vector<[2]x2xf32>
  %res = vector.shape_cast %arg0 : vector<[4]xf32> to vector<[2]x2xf32>
  // CHECK-NEXT: return %[[res]] : vector<[2]x2xf32>
  return %res: vector<[2]x2xf32>
}

// -----

// CHECK-LABEL: cannot_shape_cast_from_non_trailing_scalable_dim
// CHECK-SAME: %[[arg0:.*]]: vector<[2]x2xf32>
func.func @cannot_shape_cast_from_non_trailing_scalable_dim(%arg0: vector<[2]x2xf32>) -> vector<[4]xf32> {
  // CHECK-NEXT: %[[res:.*]] = vector.shape_cast %[[arg0]] : vector<[2]x2xf32> to vector<[4]xf32>
  %res = vector.shape_cast %arg0 : vector<[2]x2xf32> to vector<[4]xf32>
  // CHECK-NEXT: return %[[res]] : vector<[4]xf32>
  return %res: vector<[4]xf32>
}

// -----

// CHECK-LABEL: cannot_shape_cast_more_than_one_scalable_dim
// CHECK-SAME: %[[arg0:.*]]: vector<[4]x[4]xf32>
func.func @cannot_shape_cast_more_than_one_scalable_dim(%arg0: vector<[4]x[4]xf32>) -> vector<2x[2]x[4]xf32>  {
  // CHECK-NEXT: %[[res:.*]] = vector.shape_cast %[[arg0]] : vector<[4]x[4]xf32> to vector<2x[2]x[4]xf32>
  %res = vector.shape_cast %arg0 : vector<[4]x[4]xf32> to vector<2x[2]x[4]xf32>
  // CHECK-NEXT: return %[[res]] : vector<2x[2]x[4]xf32>
  return %res: vector<2x[2]x[4]xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %module_op
      : (!transform.any_op) -> !transform.any_op

    transform.apply_patterns to %f {
      transform.apply_patterns.vector.lower_shape_cast
    } : !transform.any_op
    transform.yield
  }
}
