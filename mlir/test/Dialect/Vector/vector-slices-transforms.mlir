// RUN: mlir-opt %s -test-vector-slices-conversion | FileCheck %s

// CHECK-LABEL: func @extract_slices(%arg0: vector<3x3xf32>)
//       CHECK: %[[SS:.*]] = vector.extract_strided_slice %arg0 {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]}
//       CHECK: return %[[SS]]

func @extract_slices(%arg0: vector<3x3xf32>) -> vector<2x2xf32> {
  %0 = vector.extract_slices %arg0, [2, 2], [1, 1]
    : vector<3x3xf32> into tuple<vector<2x2xf32>, vector<2x1xf32>, vector<1x2xf32>, vector<1x1xf32>>
  %1 = vector.tuple_get %0, 0 : tuple<vector<2x2xf32>, vector<2x1xf32>, vector<1x2xf32>, vector<1x1xf32>>
  return %1 : vector<2x2xf32>
}

// CHECK-LABEL: func @insert_slices(%arg0: vector<2x2xf32>, %arg1: vector<2x1xf32>, %arg2: vector<1x2xf32>, %arg3: vector<1x1xf32>)
//       CHECK: %[[C0:.*]] = constant dense<0.000000e+00> : vector<3x3xf32>
//       CHECK: %[[I0:.*]] = vector.insert_strided_slice %arg0, %[[C0]] {offsets = [0, 0], strides = [1, 1]}
//       CHECK: %[[I1:.*]] = vector.insert_strided_slice %arg1, %[[I0]] {offsets = [0, 2], strides = [1, 1]}
//       CHECK: %[[I2:.*]] = vector.insert_strided_slice %arg2, %[[I1]] {offsets = [2, 0], strides = [1, 1]}
//       CHECK: %[[I3:.*]] = vector.insert_strided_slice %arg3, %[[I2]] {offsets = [2, 2], strides = [1, 1]}
//       CHECK: return %[[I3]]

func @insert_slices(%arg0: vector<2x2xf32>,
                    %arg1: vector<2x1xf32>,
                    %arg2: vector<1x2xf32>,
                    %arg3: vector<1x1xf32>) -> vector<3x3xf32> {
  %0 = vector.tuple %arg0, %arg1, %arg2, %arg3
    : vector<2x2xf32>, vector<2x1xf32>, vector<1x2xf32>, vector<1x1xf32>
  %1 = vector.insert_slices %0, [2, 2], [1, 1]
    : tuple<vector<2x2xf32>, vector<2x1xf32>, vector<1x2xf32>, vector<1x1xf32>> into vector<3x3xf32>
  return %1 : vector<3x3xf32>
}

// CHECK-LABEL: func @extract_insert_slices(%arg0: vector<3x3xf32>)
//       CHECK: %[[C:.*]] = constant dense<0.000000e+00> : vector<3x3xf32>
//       CHECK: %[[X0:.*]] = vector.extract_strided_slice %arg0 {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]}
//       CHECK: %[[X1:.*]] = vector.extract_strided_slice %arg0 {offsets = [0, 2], sizes = [2, 1], strides = [1, 1]}
//       CHECK: %[[X2:.*]] = vector.extract_strided_slice %arg0 {offsets = [2, 0], sizes = [1, 2], strides = [1, 1]}
//       CHECK: %[[X3:.*]] = vector.extract_strided_slice %arg0 {offsets = [2, 2], sizes = [1, 1], strides = [1, 1]}
//       CHECK: %[[X4:.*]] = vector.insert_strided_slice %[[X0]], %[[C]] {offsets = [0, 0], strides = [1, 1]}
//       CHECK: %[[X5:.*]] = vector.insert_strided_slice %[[X1]], %[[X4]] {offsets = [0, 2], strides = [1, 1]}
//       CHECK: %[[X6:.*]] = vector.insert_strided_slice %[[X2]], %[[X5]] {offsets = [2, 0], strides = [1, 1]}
//       CHECK: %[[X7:.*]] = vector.insert_strided_slice %[[X3]], %[[X6]] {offsets = [2, 2], strides = [1, 1]}
//       CHECK:return %[[X7]]

func @extract_insert_slices(%arg0: vector<3x3xf32>) -> vector<3x3xf32> {
  %0 = vector.extract_slices %arg0, [2, 2], [1, 1]
    : vector<3x3xf32> into tuple<vector<2x2xf32>, vector<2x1xf32>, vector<1x2xf32>, vector<1x1xf32>>
  %1 = vector.insert_slices %0, [2, 2], [1, 1]
    : tuple<vector<2x2xf32>, vector<2x1xf32>, vector<1x2xf32>, vector<1x1xf32>> into vector<3x3xf32>
  return %1 : vector<3x3xf32>
}

// CHECK-LABEL: func @extract_slices_tuple_leaks(%arg0: vector<4xf32>)
//       CHECK: %[[X0:.*]] = vector.extract_strided_slice %arg0 {offsets = [0], sizes = [2], strides = [1]}
//       CHECK: %[[X1:.*]] = vector.extract_strided_slice %arg0 {offsets = [2], sizes = [2], strides = [1]}
//       CHECK: %[[X2:.*]] = vector.tuple %[[X0]], %[[X1]]
//       CHECK: return %[[X2]]

func @extract_slices_tuple_leaks(%arg0: vector<4xf32>) -> tuple<vector<2xf32>, vector<2xf32>> {
  %0 = vector.extract_slices %arg0, [2], [1] : vector<4xf32> into tuple<vector<2xf32>, vector<2xf32>>
  return %0 : tuple<vector<2xf32>, vector<2xf32>>
}

