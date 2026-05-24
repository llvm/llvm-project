// RUN: mlir-opt %s --transform-interpreter | FileCheck %s

// CHECK-LABEL: func @nop_shape_cast
//  CHECK-SAME:    %[[A:.*]]: vector<16xf32>
//       CHECK: return %[[A]] : vector<16xf32>
func.func @nop_shape_cast(%arg0: vector<16xf32>) -> vector<16xf32> {
  %0 = vector.shape_cast %arg0 : vector<16xf32> to vector<16xf32>
  return %0 : vector<16xf32>
}

// CHECK-LABEL: func @cancel_shape_cast
//  CHECK-SAME:    %[[A:.*]]: vector<16xf32>
//       CHECK: return %[[A]] : vector<16xf32>
func.func @cancel_shape_cast(%arg0: vector<16xf32>) -> vector<16xf32> {
  %0 = vector.shape_cast %arg0 : vector<16xf32> to vector<4x4xf32>
  %1 = vector.shape_cast %0 : vector<4x4xf32> to vector<16xf32>
  return %1 : vector<16xf32>
}

// Collapse 2-D to 1-D.
// CHECK-LABEL: func @shape_cast_2d1d
//  CHECK-SAME:    %[[A:.*]]: vector<2x2xf32>) -> vector<4xf32> {
//       CHECK: %[[UB:.*]] = ub.poison : vector<4xf32>
//
//       CHECK: %[[EX0:.*]] = vector.extract %[[A]][0] : vector<2xf32> from vector<2x2xf32>
//       CHECK: %[[IN0:.*]] = vector.insert_strided_slice %[[EX0]], %[[UB]]
//  CHECK-SAME:    {offsets = [0], strides = [1]} : vector<2xf32> into vector<4xf32>
//
//       CHECK: %[[EX1:.*]] = vector.extract %{{.*}}[1] : vector<2xf32> from vector<2x2xf32>
//       CHECK: %[[IN2:.*]] = vector.insert_strided_slice %[[EX1]], %[[IN0]]
//  CHECK-SAME:    {offsets = [2], strides = [1]} : vector<2xf32> into vector<4xf32>
//       CHECK: return %[[IN2]] : vector<4xf32>
func.func @shape_cast_2d1d(%a: vector<2x2xf32>) -> (vector<4xf32>) {
  %0 = vector.shape_cast %a : vector<2x2xf32> to vector<4xf32>
  return %0 : vector<4xf32>
}

// Collapse 3-D to 1-D.
// CHECK-LABEL: func @shape_cast_3d1d
//  CHECK-SAME:    %[[A:.*]]: vector<1x3x2xf32>
//       CHECK: %[[UB:.*]] = ub.poison : vector<6xf32>
//
//       CHECK: %[[T0:.*]] = vector.extract %[[A]][0, 0] : vector<2xf32> from vector<1x3x2xf32>
//       CHECK: %[[T1:.*]] = vector.insert_strided_slice %[[T0]], %[[UB]]
//  CHECK-SAME:    {offsets = [0], strides = [1]} : vector<2xf32> into vector<6xf32>
//
//       CHECK: %[[T2:.*]] = vector.extract %[[A]][0, 1] : vector<2xf32> from vector<1x3x2xf32>
//       CHECK: %[[T3:.*]] = vector.insert_strided_slice %[[T2]], %[[T1]]
//  CHECK-SAME:    {offsets = [2], strides = [1]} : vector<2xf32> into vector<6xf32>
//
//       CHECK: %[[T4:.*]] = vector.extract %[[A]][0, 2] : vector<2xf32> from vector<1x3x2xf32>
//       CHECK: %[[T5:.*]] = vector.insert_strided_slice %[[T4]], %[[T3]]
//  CHECK-SAME:    {offsets = [4], strides = [1]} : vector<2xf32> into vector<6xf32>
//       CHECK: return %[[T5]] : vector<6xf32>
func.func @shape_cast_3d1d(%arg0 : vector<1x3x2xf32>) -> vector<6xf32> {
  %s = vector.shape_cast %arg0 : vector<1x3x2xf32> to vector<6xf32>
  return %s : vector<6xf32>
}

// Expand 1-D to 2-D.
// CHECK-LABEL: func.func @shape_cast_1d2d(
//  CHECK-SAME:    %[[A:.*]]: vector<4xf32>) -> vector<2x2xf32> {
//       CHECK: %[[UB:.*]] = ub.poison : vector<2x2xf32>
//
//       CHECK: %[[SS0:.*]] = vector.extract_strided_slice %[[A]]
//  CHECK-SAME:    {offsets = [0], sizes = [2], strides = [1]} :
//  CHECK-SAME:    vector<4xf32> to vector<2xf32>
//       CHECK: %[[res0:.*]] = vector.insert %[[SS0]], %[[UB]] [0] :
//  CHECK-SAME:    vector<2xf32> into vector<2x2xf32>
//
//       CHECK: %[[SS2:.*]] = vector.extract_strided_slice %[[A]]
//  CHECK-SAME:    {offsets = [2], sizes = [2], strides = [1]} :
//  CHECK-SAME:    vector<4xf32> to vector<2xf32>
//       CHECK: %[[res1:.*]] = vector.insert %[[SS2]], %[[res0]] [1] :
//  CHECK-SAME:    vector<2xf32> into vector<2x2xf32>
//       CHECK: return  %[[res1]] :  vector<2x2xf32>
func.func @shape_cast_1d2d(%a: vector<4xf32>) -> (vector<2x2xf32>) {
  %1 = vector.shape_cast %a: vector<4xf32> to vector<2x2xf32>
  return %1 : vector<2x2xf32>
}

// Expand 1-D to 3-D.
// CHECK-LABEL: func @shape_cast_1d3d
//  CHECK-SAME:    %[[A:.*]]: vector<6xf32>
//       CHECK: %[[UB:.*]] = ub.poison : vector<2x1x3xf32>
//
//       CHECK: %[[T0:.*]] = vector.extract_strided_slice %[[A]]
//  CHECK-SAME:    {offsets = [0], sizes = [3], strides = [1]} :
//  CHECK-SAME:    vector<6xf32> to vector<3xf32>
//       CHECK: %[[T1:.*]] = vector.insert %[[T0]], %[[UB]] [0, 0] :
//  CHECK-SAME:    vector<3xf32> into vector<2x1x3xf32>
//
//       CHECK: %[[T2:.*]] = vector.extract_strided_slice %[[A]]
//  CHECK-SAME:    {offsets = [3], sizes = [3], strides = [1]} :
//  CHECK-SAME:    vector<6xf32> to vector<3xf32>
//       CHECK: %[[T3:.*]] = vector.insert %[[T2]], %[[T1]] [1, 0] :
//  CHECK-SAME:    vector<3xf32> into vector<2x1x3xf32>
//       CHECK: return %[[T3]] : vector<2x1x3xf32>
func.func @shape_cast_1d3d(%arg0 : vector<6xf32>) -> vector<2x1x3xf32> {
  %s = vector.shape_cast %arg0 : vector<6xf32> to vector<2x1x3xf32>
  return %s : vector<2x1x3xf32>
}

// 2-D to 2-D where the inner-most dimensions have no common factors. This
// case requires scalar element by element extraction and insertion.
// CHECK-LABEL: func @shape_cast_2d2d
//  CHECK-SAME:    %[[A:.*]]: vector<3x2xf32>
//       CHECK: %[[UB:.*]] = ub.poison : vector<2x3xf32>
//
//       CHECK: %[[T0:.*]] = vector.extract %[[A]][0, 0] : f32 from vector<3x2xf32>
//       CHECK: %[[T1:.*]] = vector.insert %[[T0]], %[[UB]] [0, 0] :
//
//       CHECK: %[[T2:.*]] = vector.extract %[[A]][0, 1] : f32 from vector<3x2xf32>
//       CHECK: %[[T3:.*]] = vector.insert %[[T2]], %[[T1]] [0, 1] :
//
//       CHECK: %[[T4:.*]] = vector.extract %[[A]][1, 0] : f32 from vector<3x2xf32>
//       CHECK: %[[T5:.*]] = vector.insert %[[T4]], %[[T3]] [0, 2] :
//
//       CHECK: %[[T6:.*]] = vector.extract %[[A]][1, 1] : f32 from vector<3x2xf32>
//       CHECK: %[[T7:.*]] = vector.insert %[[T6]], %[[T5]] [1, 0] :
//
//       CHECK: %[[T8:.*]] = vector.extract %[[A]][2, 0] : f32 from vector<3x2xf32>
//       CHECK: %[[T9:.*]] = vector.insert %[[T8]], %[[T7]] [1, 1] :
//
//       CHECK: %[[T10:.*]] = vector.extract %[[A]][2, 1] : f32 from vector<3x2xf32>
//       CHECK: %[[T11:.*]] = vector.insert %[[T10]], %[[T9]] [1, 2] :
//
//       CHECK: return %[[T11]] : vector<2x3xf32>
func.func @shape_cast_2d2d(%arg0 : vector<3x2xf32>) -> vector<2x3xf32> {
  %s = vector.shape_cast %arg0: vector<3x2xf32> to vector<2x3xf32>
  return %s : vector<2x3xf32>
}

// CHECK-LABEL: func.func @shape_cast_0d1d(
//  CHECK-SAME:    %[[A:.*]]: vector<f32>) -> vector<1xf32> {
//       CHECK: %[[UB:.*]] = ub.poison : vector<1xf32>
//
//       CHECK: %[[EXTRACT0:.*]] = vector.extract %[[A]][] : f32 from vector<f32>
//       CHECK: %[[RES:.*]] = vector.insert %[[EXTRACT0]], %[[UB]] [0] :
//       CHECK: return %[[RES]] : vector<1xf32>
func.func @shape_cast_0d1d(%arg0 : vector<f32>) -> vector<1xf32> {
  %s = vector.shape_cast %arg0 : vector<f32> to vector<1xf32>
  return %s : vector<1xf32>
}

// CHECK-LABEL: func.func @shape_cast_1d0d(
//  CHECK-SAME:    %[[A:.*]]: vector<1xf32>) -> vector<f32> {
//       CHECK: %[[UB:.*]] = ub.poison : vector<f32>
//
//       CHECK: %[[EXTRACT0:.*]] = vector.extract %[[A]][0] : f32 from vector<1xf32>
//       CHECK: %[[RES:.*]] = vector.insert %[[EXTRACT0]], %[[UB]] [] :
//       CHECK: return %[[RES]] : vector<f32>
func.func @shape_cast_1d0d(%arg0 : vector<1xf32>) -> vector<f32> {
  %s = vector.shape_cast %arg0 : vector<1xf32> to vector<f32>
  return %s : vector<f32>
}

// The shapes have 2 inner dimension sizes in common, so the extract result is rank-2.
// CHECK-LABEL: func.func @squeeze_out_prefix_unit_dim(
//  CHECK-SAME:    %[[A:.*]]: vector<1x2x3xf32>) -> vector<2x3xf32> {
//
//       CHECK: %[[EXTRACTED:.*]] = vector.extract %[[A]][0] :
//  CHECK-SAME:    vector<2x3xf32> from vector<1x2x3xf32>
//       CHECK: return %[[EXTRACTED]] : vector<2x3xf32>
func.func @squeeze_out_prefix_unit_dim(%arg0 : vector<1x2x3xf32>) -> vector<2x3xf32> {
  %s = vector.shape_cast %arg0 : vector<1x2x3xf32> to vector<2x3xf32>
  return %s : vector<2x3xf32>
}

// The shapes have 1 inner dimension size in common, so the extract results are rank-1.
// CHECK-LABEL: func.func @squeeze_out_middle_unit_dim(
//  CHECK-SAME:    %[[A:.*]]: vector<2x1x3xf32>) -> vector<2x3xf32> {
//       CHECK: %[[UB:.*]] = ub.poison : vector<2x3xf32>
//
//       CHECK: %[[E0:.*]] = vector.extract %[[A]][0, 0] : vector<3xf32>
//       CHECK: %[[I0:.*]] = vector.insert %[[E0]], %[[UB]] [0] :
//
//       CHECK: %[[E1:.*]] = vector.extract %[[A]][1, 0] : vector<3xf32>
//       CHECK: %[[I1:.*]] = vector.insert %[[E1]], %[[I0]] [1] :
//       CHECK: return %[[I1]] : vector<2x3xf32>
func.func @squeeze_out_middle_unit_dim(%arg0 : vector<2x1x3xf32>) -> vector<2x3xf32> {
  %s = vector.shape_cast %arg0 : vector<2x1x3xf32> to vector<2x3xf32>
  return %s : vector<2x3xf32>
}

// CHECK-LABEL: func.func @prepend_unit_dim(
//  CHECK-SAME:    %[[A:.*]]: vector<2x3xf32>) -> vector<1x2x3xf32> {
//       CHECK: %[[UB:.*]] = ub.poison : vector<1x2x3xf32>
//
//       CHECK: %[[I0:.*]] = vector.insert %[[A]], %[[UB]] [0]
//       CHECK: return %[[I0]] : vector<1x2x3xf32>
func.func @prepend_unit_dim(%arg0 : vector<2x3xf32>) -> vector<1x2x3xf32> {
  %s = vector.shape_cast %arg0 : vector<2x3xf32> to vector<1x2x3xf32>
  return %s : vector<1x2x3xf32>
}

// CHECK-LABEL: func.func @insert_middle_unit_dim(
//  CHECK-SAME:    %[[A:.*]]: vector<2x3xf32>) -> vector<2x1x3xf32> {
//       CHECK: %[[UB:.*]] = ub.poison : vector<2x1x3xf32>
//
//       CHECK: %[[E0:.*]] = vector.extract %[[A]][0] : vector<3xf32>
//       CHECK: %[[I0:.*]] = vector.insert %[[E0]], %[[UB]] [0, 0] : vector<3xf32>
//
//       CHECK: %[[E1:.*]] = vector.extract %[[A]][1] : vector<3xf32>
//       CHECK: %[[I1:.*]] = vector.insert %[[E1]], %[[I0]] [1, 0] : vector<3xf32>
//       CHECK: return %[[I1]] : vector<2x1x3xf32>
func.func @insert_middle_unit_dim(%arg0 : vector<2x3xf32>) -> vector<2x1x3xf32> {
  %s = vector.shape_cast %arg0 : vector<2x3xf32> to vector<2x1x3xf32>
  return %s : vector<2x1x3xf32>
}

//   CHECK-LABEL: func.func @postpend_unit_dims(
//    CHECK-SAME:    %[[A:.*]]: vector<4xf32>) -> vector<4x1x1xf32> {
//         CHECK: %[[UB:.*]] = ub.poison : vector<4x1x1xf32>
//         CHECK: %[[E0:.*]] = vector.extract %[[A]][0] : f32 from vector<4xf32>
//         CHECK: %[[I0:.*]] = vector.insert %[[E0]], %[[UB]] [0, 0, 0] : f32 into vector<4x1x1xf32>
//         CHECK: %[[E1:.*]] = vector.extract %[[A]][1] : f32 from vector<4xf32>
//         CHECK: %[[I1:.*]] = vector.insert %[[E1]], %[[I0]] [1, 0, 0] : f32 into vector<4x1x1xf32>
//         CHECK: vector.extract %[[A]][2]
//         CHECK: vector.insert {{.*}} [2, 0, 0]
//         CHECK: vector.extract %[[A]][3]
//         CHECK: vector.insert {{.*}} [3, 0, 0]
//         CHECK: return
func.func @postpend_unit_dims(%arg0 : vector<4xf32>) -> vector<4x1x1xf32> {
  %s = vector.shape_cast %arg0 : vector<4xf32> to vector<4x1x1xf32>
  return %s : vector<4x1x1xf32>
}

// CHECK-LABEL: func.func @expand_inner_dims(
//  CHECK-SAME:    %[[A:.*]]: vector<2x10xf32>) -> vector<2x2x5xf32> {
//       CHECK: %[[UB:.*]] = ub.poison : vector<2x2x5xf32>
//
//       CHECK: %[[E0:.*]] = vector.extract %[[A]][0] : vector<10xf32>
//       CHECK: %[[S0:.*]] = vector.extract_strided_slice %[[E0]]
//  CHECK-SAME:    {offsets = [0], sizes = [5], {{.*}} to vector<5xf32>
//       CHECK: %[[I0:.*]] = vector.insert %[[S0]], %[[UB]] [0, 0]
//
//       CHECK: %[[S1:.*]] = vector.extract_strided_slice %[[E0]]
//  CHECK-SAME:    {offsets = [5], sizes = [5], {{.*}} to vector<5xf32>
//       CHECK: %[[I1:.*]] = vector.insert %[[S1]], %[[I0]] [0, 1]
//
//       CHECK: %[[E1:.*]] = vector.extract %[[A]][1] : vector<10xf32>
//       CHECK: %[[S2:.*]] = vector.extract_strided_slice %[[E1]]
//  CHECK-SAME:    {offsets = [0], sizes = [5], {{.*}} to vector<5xf32>
//       CHECK: %[[I2:.*]] = vector.insert %[[S2]], %[[I1]] [1, 0]
//
//       CHECK: %[[S3:.*]] = vector.extract_strided_slice %[[E1]]
//  CHECK-SAME:    {offsets = [5], sizes = [5], {{.*}} to vector<5xf32>
//       CHECK: %[[I3:.*]] = vector.insert %[[S3]], %[[I2]] [1, 1]
//       CHECK: return %[[I3]] : vector<2x2x5xf32>
func.func @expand_inner_dims(%arg0 : vector<2x10xf32>) -> vector<2x2x5xf32> {
  %s = vector.shape_cast %arg0 : vector<2x10xf32> to vector<2x2x5xf32>
  return %s : vector<2x2x5xf32>
}


// Some pseudocode describing how this function is lowered:
//
// func collapse_inner_dims(A : vector<2x2x5xi8>) -> vector<1x2x1x10xi8> {
//   v0          = empty of shape (10)
//   v1          = empty of shape (1,2,1,10)
//   v0[0:5]     = A[0,0,:]
//   v0[5:10]    = A[0,1,:]
//   v1[0,0,0,:] = v0
//   v0[0:5]     = A[1,0,:]
//   v0[5:10]    = A[1,1,:]
//   v1[0,1,0,:] = v0
//   return v1;
// }
// CHECK-LABEL: func.func @collapse_inner_dims(
//  CHECK-SAME:    %[[A:.*]]: vector<2x2x5xi8>) -> vector<1x2x1x10xi8> {
//   CHECK-DAG: %[[UBSMALL:.*]] = ub.poison : vector<10xi8>
//   CHECK-DAG: %[[UBLARGE:.*]] = ub.poison : vector<1x2x1x10xi8>
//
//       CHECK: %[[EX0:.*]] = vector.extract %[[A]][0, 0]
//       CHECK: %[[IN0:.*]] = vector.insert_strided_slice %[[EX0]], %[[UBSMALL]]
//  CHECK-SAME:    {offsets = [0], {{.*}}
//       CHECK: %[[EX1:.*]] = vector.extract %[[A]][0, 1]
//       CHECK: %[[IN1:.*]] = vector.insert_strided_slice %[[EX1]], %[[IN0]]
//  CHECK-SAME:    {offsets = [5], {{.*}}
//       CHECK: %[[IN2:.*]] = vector.insert %[[IN1]], %[[UBLARGE]] [0, 0, 0]
//
//       CHECK: %[[EX2:.*]] = vector.extract %[[A]][1, 0]
//       CHECK: %[[IN3:.*]] = vector.insert_strided_slice %[[EX2]], %[[UBSMALL]]
//  CHECK-SAME:    {offsets = [0], {{.*}}
//       CHECK: %[[EX3:.*]] = vector.extract %[[A]][1, 1]
//       CHECK: %[[IN4:.*]] = vector.insert_strided_slice %[[EX3]], %[[IN3]]
//  CHECK-SAME:    {offsets = [5], {{.*}}
//       CHECK: %[[IN5:.*]] = vector.insert %[[IN4]], %[[IN2]] [0, 1, 0]
//       CHECK: return %[[IN5]] : vector<1x2x1x10xi8>
func.func @collapse_inner_dims(%arg0 : vector<2x2x5xi8>) -> vector<1x2x1x10xi8> {
  %s = vector.shape_cast %arg0 : vector<2x2x5xi8> to vector<1x2x1x10xi8>
  return %s : vector<1x2x1x10xi8>
}

// Some alternative pseudocode describing how this function is lowered:
//
// func non_dividing_gcd_decreasing(A : vector<2x15xi8>) -> vector<3x10xi8> {
//   v0          = empty of shape (10)
//   v1          = empty of shape (3,10)
//   e0          = A[0,:]
//   v0[0:5]     = e0[0:5]
//   v0[5:10]    = e0[5:10]
//   v1[0,:]     = v0
//   v0[0,0:5]   = e0[10:15]
//   e1          = A[1,:]
//   v0[0,5:10]  = e1[0:5]
//   v1[1,:]     = v0
//   v0[0,0:5]   = e1[5:10]
//   v0[0,5:10]  = e1[10:15]
//   v1[2,:]     = v0
//   return v1;
// }
// CHECK-LABEL: func.func @non_dividing_gcd_decreasing(
//  CHECK-SAME:    %[[A:.*]]: vector<2x15xi8>) -> vector<3x10xi8> {
//   CHECK-DAG: %[[UB0:.*]] = ub.poison : vector<10xi8>
//   CHECK-DAG: %[[UB1:.*]] = ub.poison : vector<3x10xi8>
//
// First 10 elements:
//       CHECK: %[[EX0:.*]] = vector.extract %[[A]][0] : vector<15xi8> from vector<2x15xi8>
//       CHECK: %[[SS0:.*]] = vector.extract_strided_slice %[[EX0]]
//  CHECK-SAME:    {offsets = [0], {{.*}} to vector<5xi8>
//       CHECK: %[[IN0:.*]] = vector.insert_strided_slice %[[SS0]], %[[UB0]]
//  CHECK-SAME:    {offsets = [0], {{.*}}
//       CHECK: %[[SS1:.*]] = vector.extract_strided_slice %[[EX0]]
//  CHECK-SAME:    {offsets = [5], {{.*}} to vector<5xi8>
//       CHECK: %[[IN1:.*]] = vector.insert_strided_slice %[[SS1]], %[[IN0]]
//  CHECK-SAME:    {offsets = [5], {{.*}}
//       CHECK: %[[IN2:.*]] = vector.insert %[[IN1]], %[[UB1]] [0] : vector<10xi8> into vector<3x10xi8>
//
// Next 10 elements:
//       CHECK: %[[SS2:.*]] = vector.extract_strided_slice %[[EX0]]
//  CHECK-SAME:    {offsets = [10], {{.*}} to vector<5xi8>
//       CHECK: %[[IN3:.*]] = vector.insert_strided_slice %[[SS2]], %[[UB0]]
//  CHECK-SAME:    {offsets = [0], {{.*}}
//       CHECK: %[[EX1:.*]] = vector.extract %[[A]][1] : vector<15xi8> from vector<2x15xi8>
//       CHECK: %[[SS3:.*]] = vector.extract_strided_slice %[[EX1]]
//  CHECK-SAME:    {offsets = [0], {{.*}} to vector<5xi8>
//       CHECK: %[[IN4:.*]] = vector.insert_strided_slice %[[SS3]], %[[IN3]]
//  CHECK-SAME:    {offsets = [5], {{.*}}
//       CHECK: %[[IN5:.*]] = vector.insert %[[IN4]], %[[IN2]] [1] : vector<10xi8> into vector<3x10xi8>
//
// Final 10 elements:
//       CHECK: %[[SS4:.*]] = vector.extract_strided_slice %[[EX1]]
//  CHECK-SAME:    {offsets = [5], {{.*}} to vector<5xi8>
//       CHECK: %[[IN6:.*]] = vector.insert_strided_slice %[[SS4]], %[[UB0]]
//  CHECK-SAME:    {offsets = [0], {{.*}}
//       CHECK: %[[SS5:.*]] = vector.extract_strided_slice %[[EX1]]
//  CHECK-SAME:    {offsets = [10], {{.*}} to vector<5xi8>
//       CHECK: %[[IN7:.*]] = vector.insert_strided_slice %[[SS5]], %[[IN6]]
//  CHECK-SAME:    {offsets = [5], {{.*}}
//       CHECK: %[[IN8:.*]] = vector.insert %[[IN7]], %[[IN5]] [2] : vector<10xi8> into vector<3x10xi8>
//       CHECK: return %[[IN8]] : vector<3x10xi8>
func.func @non_dividing_gcd_decreasing(%arg0 : vector<2x15xi8>) -> vector<3x10xi8> {
  %0 = vector.shape_cast %arg0 : vector<2x15xi8> to vector<3x10xi8>
  return %0 : vector<3x10xi8>
}

// CHECK-LABEL: func.func @non_dividing_gcd_increasing(
//  CHECK-SAME:    %[[A:.*]]: vector<3x10xi8>) -> vector<2x15xi8> {
//
//   CHECK-DAG: ub.poison : vector<15xi8>
//   CHECK-DAG: ub.poison : vector<2x15xi8>
//
// Collect the first 15 elements, and insert into the first row of the result.
//       CHECK: vector.extract %[[A]][0]
//       CHECK: extract_strided_slice
//       CHECK: insert_strided_slice
//       CHECK: extract_strided_slice
//       CHECK: insert_strided_slice
//       CHECK: vector.extract %[[A]][1]
//       CHECK: extract_strided_slice
//       CHECK: insert_strided_slice
//       CHECK: vector.insert {{.*}} [0] : vector<15xi8> into vector<2x15xi8>
//
// Collect the next 15 elements, and insert into the second row of the result.
//       CHECK: extract_strided_slice
//       CHECK: insert_strided_slice
//       CHECK: vector.extract %[[A]][2]
//       CHECK: extract_strided_slice
//       CHECK: insert_strided_slice
//       CHECK: extract_strided_slice
//       CHECK: insert_strided_slice
//       CHECK: vector.insert {{.*}} [1] : vector<15xi8> into vector<2x15xi8>
func.func @non_dividing_gcd_increasing(%arg0 : vector<3x10xi8>) -> vector<2x15xi8> {
  %0 = vector.shape_cast %arg0 : vector<3x10xi8> to vector<2x15xi8>
  return %0 : vector<2x15xi8>
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
