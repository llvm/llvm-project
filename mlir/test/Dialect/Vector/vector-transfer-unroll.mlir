// RUN: mlir-opt %s -test-vector-transfer-unrolling-patterns --split-input-file | FileCheck %s

// CHECK-LABEL: func @transfer_read_unroll
//       CHECK-DAG:   %[[C2:.*]] = constant 2 : index
//       CHECK-DAG:   %[[C0:.*]] = constant 0 : index
//       CHECK:   %[[VTR0:.*]] = vector.transfer_read {{.*}}[%[[C0]], %[[C0]]], %{{.*}} : memref<4x4xf32>, vector<2x2xf32>
//  CHECK-NEXT:   %[[VEC0:.*]] = vector.insert_strided_slice %[[VTR0]], %{{.*}} {offsets = [0, 0], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>
//  CHECK-NEXT:   %[[VTR1:.*]] = vector.transfer_read {{.*}}[%[[C0]], %[[C2]]], %{{.*}} : memref<4x4xf32>, vector<2x2xf32>
//  CHECK-NEXT:   %[[VEC1:.*]] = vector.insert_strided_slice %[[VTR1]], %[[VEC0]] {offsets = [0, 2], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>
//  CHECK-NEXT:   %[[VTR2:.*]] = vector.transfer_read {{.*}}[%[[C2]], %[[C0]]], %{{.*}} : memref<4x4xf32>, vector<2x2xf32>
//  CHECK-NEXT:   %[[VEC2:.*]] = vector.insert_strided_slice %[[VTR2]], %[[VEC1]] {offsets = [2, 0], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>
//  CHECK-NEXT:   %[[VTR3:.*]] = vector.transfer_read {{.*}}[%[[C2]], %[[C2]]], %{{.*}} : memref<4x4xf32>, vector<2x2xf32>
//  CHECK-NEXT:   %[[VEC3:.*]] = vector.insert_strided_slice %[[VTR3]], %[[VEC2]] {offsets = [2, 2], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>
//  CHECK-NEXT:   return %[[VEC3]] : vector<4x4xf32>

func @transfer_read_unroll(%arg0 : memref<4x4xf32>) -> vector<4x4xf32> {
  %c0 = constant 0 : index
  %cf0 = constant 0.0 : f32
  %0 = vector.transfer_read %arg0[%c0, %c0], %cf0 : memref<4x4xf32>, vector<4x4xf32>
  return %0 : vector<4x4xf32>
}

// CHECK-LABEL: func @transfer_write_unroll
//       CHECK-DAG:   %[[C2:.*]] = constant 2 : index
//       CHECK-DAG:   %[[C0:.*]] = constant 0 : index
//       CHECK:   %[[S0:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
//  CHECK-NEXT:   vector.transfer_write %[[S0]], {{.*}}[%[[C0]], %[[C0]]] {{.*}} : vector<2x2xf32>, memref<4x4xf32>
//  CHECK-NEXT:   %[[S1:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [0, 2], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
//  CHECK-NEXT:   vector.transfer_write %[[S1]], {{.*}}[%[[C0]], %[[C2]]] {{.*}} : vector<2x2xf32>, memref<4x4xf32>
//  CHECK-NEXT:   %[[S2:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
//  CHECK-NEXT:   vector.transfer_write %[[S2]], {{.*}}[%[[C2]], %[[C0]]] {{.*}} : vector<2x2xf32>, memref<4x4xf32>
//  CHECK-NEXT:   %[[S3:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [2, 2], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
//  CHECK-NEXT:   vector.transfer_write %[[S3]], {{.*}}[%[[C2]], %[[C2]]] {{.*}} : vector<2x2xf32>, memref<4x4xf32>
//  CHECK-NEXT:   return

func @transfer_write_unroll(%arg0 : memref<4x4xf32>, %arg1 : vector<4x4xf32>) {
  %c0 = constant 0 : index
  vector.transfer_write %arg1, %arg0[%c0, %c0] : vector<4x4xf32>, memref<4x4xf32>
  return
}

// CHECK-LABEL: func @transfer_readwrite_unroll
//       CHECK-DAG:   %[[C2:.*]] = constant 2 : index
//       CHECK-DAG:   %[[C0:.*]] = constant 0 : index
//       CHECK:   %[[VTR0:.*]] = vector.transfer_read {{.*}}[%[[C0]], %[[C0]]], %{{.*}} : memref<4x4xf32>, vector<2x2xf32>
//  CHECK-NEXT:   %[[VTR1:.*]] = vector.transfer_read {{.*}}[%[[C0]], %[[C2]]], %{{.*}} : memref<4x4xf32>, vector<2x2xf32>
//  CHECK-NEXT:   %[[VTR2:.*]] = vector.transfer_read {{.*}}[%[[C2]], %[[C0]]], %{{.*}} : memref<4x4xf32>, vector<2x2xf32>
//  CHECK-NEXT:   %[[VTR3:.*]] = vector.transfer_read {{.*}}[%[[C2]], %[[C2]]], %{{.*}} : memref<4x4xf32>, vector<2x2xf32>
//  CHECK-NEXT:   vector.transfer_write %[[VTR0]], {{.*}}[%[[C0]], %[[C0]]] {{.*}} : vector<2x2xf32>, memref<4x4xf32>
//  CHECK-NEXT:   vector.transfer_write %[[VTR1]], {{.*}}[%[[C0]], %[[C2]]] {{.*}} : vector<2x2xf32>, memref<4x4xf32>
//  CHECK-NEXT:   vector.transfer_write %[[VTR2]], {{.*}}[%[[C2]], %[[C0]]] {{.*}} : vector<2x2xf32>, memref<4x4xf32>
//  CHECK-NEXT:   vector.transfer_write %[[VTR3]], {{.*}}[%[[C2]], %[[C2]]] {{.*}} : vector<2x2xf32>, memref<4x4xf32>
//  CHECK-NEXT:   return

func @transfer_readwrite_unroll(%arg0 : memref<4x4xf32>) {
  %c0 = constant 0 : index
  %cf0 = constant 0.0 : f32
  %0 = vector.transfer_read %arg0[%c0, %c0], %cf0 : memref<4x4xf32>, vector<4x4xf32>
  vector.transfer_write %0, %arg0[%c0, %c0] : vector<4x4xf32>, memref<4x4xf32>
  return
}

// CHECK-LABEL: func @transfer_read_unroll_tensor
//       CHECK-DAG:   %[[C2:.*]] = constant 2 : index
//       CHECK-DAG:   %[[C0:.*]] = constant 0 : index
//       CHECK:   %[[VTR0:.*]] = vector.transfer_read {{.*}}[%[[C0]], %[[C0]]], %{{.*}} : tensor<4x4xf32>, vector<2x2xf32>
//  CHECK-NEXT:   %[[VEC0:.*]] = vector.insert_strided_slice %[[VTR0]], %{{.*}} {offsets = [0, 0], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>
//  CHECK-NEXT:   %[[VTR1:.*]] = vector.transfer_read {{.*}}[%[[C0]], %[[C2]]], %{{.*}} : tensor<4x4xf32>, vector<2x2xf32>
//  CHECK-NEXT:   %[[VEC1:.*]] = vector.insert_strided_slice %[[VTR1]], %[[VEC0]] {offsets = [0, 2], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>
//  CHECK-NEXT:   %[[VTR2:.*]] = vector.transfer_read {{.*}}[%[[C2]], %[[C0]]], %{{.*}} : tensor<4x4xf32>, vector<2x2xf32>
//  CHECK-NEXT:   %[[VEC2:.*]] = vector.insert_strided_slice %[[VTR2]], %[[VEC1]] {offsets = [2, 0], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>
//  CHECK-NEXT:   %[[VTR3:.*]] = vector.transfer_read {{.*}}[%[[C2]], %[[C2]]], %{{.*}} : tensor<4x4xf32>, vector<2x2xf32>
//  CHECK-NEXT:   %[[VEC3:.*]] = vector.insert_strided_slice %[[VTR3]], %[[VEC2]] {offsets = [2, 2], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>
//  CHECK-NEXT:   return %[[VEC3]] : vector<4x4xf32>

func @transfer_read_unroll_tensor(%arg0 : tensor<4x4xf32>) -> vector<4x4xf32> {
  %c0 = constant 0 : index
  %cf0 = constant 0.0 : f32
  %0 = vector.transfer_read %arg0[%c0, %c0], %cf0 : tensor<4x4xf32>, vector<4x4xf32>
  return %0 : vector<4x4xf32>
}

// CHECK-LABEL: func @transfer_write_unroll_tensor
//       CHECK-DAG:   %[[C2:.*]] = constant 2 : index
//       CHECK-DAG:   %[[C0:.*]] = constant 0 : index
//       CHECK:   %[[S0:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
//  CHECK-NEXT:   %[[VTW0:.*]] = vector.transfer_write %[[S0]], {{.*}}[%[[C0]], %[[C0]]] {{.*}} : vector<2x2xf32>, tensor<4x4xf32>
//  CHECK-NEXT:   %[[S1:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [0, 2], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
//  CHECK-NEXT:   %[[VTW1:.*]] = vector.transfer_write %[[S1]], %[[VTW0]][%[[C0]], %[[C2]]] {{.*}} : vector<2x2xf32>, tensor<4x4xf32>
//  CHECK-NEXT:   %[[S2:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
//  CHECK-NEXT:   %[[VTW2:.*]] = vector.transfer_write %[[S2]], %[[VTW1]][%[[C2]], %[[C0]]] {{.*}} : vector<2x2xf32>, tensor<4x4xf32>
//  CHECK-NEXT:   %[[S3:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [2, 2], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
//  CHECK-NEXT:   %[[VTW3:.*]] = vector.transfer_write %[[S3]], %[[VTW2]][%[[C2]], %[[C2]]] {{.*}} : vector<2x2xf32>, tensor<4x4xf32>
//  CHECK-NEXT:   return %[[VTW3]] : tensor<4x4xf32>

func @transfer_write_unroll_tensor(%arg0 : tensor<4x4xf32>,
  %arg1 : vector<4x4xf32>) -> tensor<4x4xf32> {
  %c0 = constant 0 : index
  %r = vector.transfer_write %arg1, %arg0[%c0, %c0] :
    vector<4x4xf32>, tensor<4x4xf32>
  return %r: tensor<4x4xf32>
}

// CHECK-LABEL: func @transfer_readwrite_unroll_tensor
//       CHECK-DAG:   %[[C2:.*]] = constant 2 : index
//       CHECK-DAG:   %[[C0:.*]] = constant 0 : index
//       CHECK:   %[[VTR0:.*]] = vector.transfer_read {{.*}}[%[[C0]], %[[C0]]], %{{.*}} : tensor<4x4xf32>, vector<2x2xf32>
//  CHECK-NEXT:   %[[VTR1:.*]] = vector.transfer_read {{.*}}[%[[C0]], %[[C2]]], %{{.*}} : tensor<4x4xf32>, vector<2x2xf32>
//  CHECK-NEXT:   %[[VTR2:.*]] = vector.transfer_read {{.*}}[%[[C2]], %[[C0]]], %{{.*}} : tensor<4x4xf32>, vector<2x2xf32>
//  CHECK-NEXT:   %[[VTR3:.*]] = vector.transfer_read {{.*}}[%[[C2]], %[[C2]]], %{{.*}} : tensor<4x4xf32>, vector<2x2xf32>
//  CHECK-NEXT:   %[[VTW0:.*]] = vector.transfer_write %[[VTR0]], {{.*}}[%[[C0]], %[[C0]]] {{.*}} : vector<2x2xf32>, tensor<4x4xf32>
//  CHECK-NEXT:   %[[VTW1:.*]] = vector.transfer_write %[[VTR1]], %[[VTW0]][%[[C0]], %[[C2]]] {{.*}} : vector<2x2xf32>, tensor<4x4xf32>
//  CHECK-NEXT:   %[[VTW2:.*]] = vector.transfer_write %[[VTR2]], %[[VTW1]][%[[C2]], %[[C0]]] {{.*}} : vector<2x2xf32>, tensor<4x4xf32>
//  CHECK-NEXT:   %[[VTW3:.*]] = vector.transfer_write %[[VTR3]], %[[VTW2]][%[[C2]], %[[C2]]] {{.*}} : vector<2x2xf32>, tensor<4x4xf32>
//  CHECK-NEXT:   return %[[VTW3]] : tensor<4x4xf32>

func @transfer_readwrite_unroll_tensor(%arg0 : tensor<4x4xf32>, %arg1 : tensor<4x4xf32>) ->
  tensor<4x4xf32> {
  %c0 = constant 0 : index
  %cf0 = constant 0.0 : f32
  %0 = vector.transfer_read %arg0[%c0, %c0], %cf0 : tensor<4x4xf32>, vector<4x4xf32>
  %r = vector.transfer_write %0, %arg1[%c0, %c0] : vector<4x4xf32>, tensor<4x4xf32>
  return %r: tensor<4x4xf32>
}

// -----

// CHECK-LABEL: func @transfer_read_unroll_permutation
//       CHECK-DAG:   %[[C4:.*]] = constant 4 : index
//       CHECK-DAG:   %[[C2:.*]] = constant 2 : index
//       CHECK-DAG:   %[[C0:.*]] = constant 0 : index
//       CHECK:   %[[VTR0:.*]] = vector.transfer_read {{.*}}[%[[C0]], %[[C0]]], %{{.*}} : memref<6x4xf32>, vector<2x2xf32>
//  CHECK-NEXT:   %[[VEC0:.*]] = vector.insert_strided_slice %[[VTR0]], %{{.*}} {offsets = [0, 0], strides = [1, 1]} : vector<2x2xf32> into vector<4x6xf32>
//  CHECK-NEXT:   %[[VTR1:.*]] = vector.transfer_read {{.*}}[%[[C2]], %[[C0]]], %{{.*}} : memref<6x4xf32>, vector<2x2xf32>
//  CHECK-NEXT:   %[[VEC1:.*]] = vector.insert_strided_slice %[[VTR1]], %[[VEC0]] {offsets = [0, 2], strides = [1, 1]} : vector<2x2xf32> into vector<4x6xf32>
//  CHECK-NEXT:   %[[VTR2:.*]] = vector.transfer_read {{.*}}[%[[C4]], %[[C0]]], %{{.*}} : memref<6x4xf32>, vector<2x2xf32>
//  CHECK-NEXT:   %[[VEC2:.*]] = vector.insert_strided_slice %[[VTR2]], %[[VEC1]] {offsets = [0, 4], strides = [1, 1]} : vector<2x2xf32> into vector<4x6xf32>
//  CHECK-NEXT:   %[[VTR3:.*]] = vector.transfer_read {{.*}}[%[[C0]], %[[C2]]], %{{.*}} : memref<6x4xf32>, vector<2x2xf32>
//  CHECK-NEXT:   %[[VEC3:.*]] = vector.insert_strided_slice %[[VTR3]], %[[VEC2]] {offsets = [2, 0], strides = [1, 1]} : vector<2x2xf32> into vector<4x6xf32>
//  CHECK-NEXT:   %[[VTR4:.*]] = vector.transfer_read {{.*}}[%[[C2]], %[[C2]]], %{{.*}} : memref<6x4xf32>, vector<2x2xf32>
//  CHECK-NEXT:   %[[VEC4:.*]] = vector.insert_strided_slice %[[VTR4]], %[[VEC3]] {offsets = [2, 2], strides = [1, 1]} : vector<2x2xf32> into vector<4x6xf32>
//  CHECK-NEXT:   %[[VTR5:.*]] = vector.transfer_read {{.*}}[%[[C4]], %[[C2]]], %{{.*}} : memref<6x4xf32>, vector<2x2xf32>
//  CHECK-NEXT:   %[[VEC5:.*]] = vector.insert_strided_slice %[[VTR5]], %[[VEC4]] {offsets = [2, 4], strides = [1, 1]} : vector<2x2xf32> into vector<4x6xf32>
//  CHECK-NEXT:   return %[[VEC5]] : vector<4x6xf32>
#map0 = affine_map<(d0, d1) -> (d1, d0)>
func @transfer_read_unroll_permutation(%arg0 : memref<6x4xf32>) -> vector<4x6xf32> {
  %c0 = constant 0 : index
  %cf0 = constant 0.0 : f32
  %0 = vector.transfer_read %arg0[%c0, %c0], %cf0 {permutation_map = #map0} : memref<6x4xf32>, vector<4x6xf32>
  return %0 : vector<4x6xf32>
}

// -----

// CHECK-LABEL: func @transfer_read_unroll_broadcast
//       CHECK-DAG:   %[[C2:.*]] = constant 2 : index
//       CHECK-DAG:   %[[C0:.*]] = constant 0 : index
//       CHECK:   %[[VTR0:.*]] = vector.transfer_read {{.*}}[%[[C0]], %[[C0]]], %{{.*}} : memref<6x4xf32>, vector<2x2xf32>
//  CHECK-NEXT:   %[[VEC0:.*]] = vector.insert_strided_slice %[[VTR0]], %{{.*}} {offsets = [0, 0], strides = [1, 1]} : vector<2x2xf32> into vector<6x4xf32>
//  CHECK-NEXT:   %[[VTR1:.*]] = vector.transfer_read {{.*}}[%[[C0]], %[[C2]]], %{{.*}} : memref<6x4xf32>, vector<2x2xf32>
//  CHECK-NEXT:   %[[VEC1:.*]] = vector.insert_strided_slice %[[VTR1]], %[[VEC0]] {offsets = [0, 2], strides = [1, 1]} : vector<2x2xf32> into vector<6x4xf32>
//  CHECK-NEXT:   %[[VTR2:.*]] = vector.transfer_read {{.*}}[%[[C0]], %[[C0]]], %{{.*}} : memref<6x4xf32>, vector<2x2xf32>
//  CHECK-NEXT:   %[[VEC2:.*]] = vector.insert_strided_slice %[[VTR2]], %[[VEC1]] {offsets = [2, 0], strides = [1, 1]} : vector<2x2xf32> into vector<6x4xf32>
//  CHECK-NEXT:   %[[VTR3:.*]] = vector.transfer_read {{.*}}[%[[C0]], %[[C2]]], %{{.*}} : memref<6x4xf32>, vector<2x2xf32>
//  CHECK-NEXT:   %[[VEC3:.*]] = vector.insert_strided_slice %[[VTR3]], %[[VEC2]] {offsets = [2, 2], strides = [1, 1]} : vector<2x2xf32> into vector<6x4xf32>
//  CHECK-NEXT:   %[[VTR4:.*]] = vector.transfer_read {{.*}}[%[[C0]], %[[C0]]], %{{.*}} : memref<6x4xf32>, vector<2x2xf32>
//  CHECK-NEXT:   %[[VEC4:.*]] = vector.insert_strided_slice %[[VTR4]], %[[VEC3]] {offsets = [4, 0], strides = [1, 1]} : vector<2x2xf32> into vector<6x4xf32>
//  CHECK-NEXT:   %[[VTR5:.*]] = vector.transfer_read {{.*}}[%[[C0]], %[[C2]]], %{{.*}} : memref<6x4xf32>, vector<2x2xf32>
//  CHECK-NEXT:   %[[VEC5:.*]] = vector.insert_strided_slice %[[VTR5]], %[[VEC4]] {offsets = [4, 2], strides = [1, 1]} : vector<2x2xf32> into vector<6x4xf32>
//  CHECK-NEXT:   return %[[VEC5]] : vector<6x4xf32>
#map0 = affine_map<(d0, d1) -> (0, d1)>
func @transfer_read_unroll_broadcast(%arg0 : memref<6x4xf32>) -> vector<6x4xf32> {
  %c0 = constant 0 : index
  %cf0 = constant 0.0 : f32
  %0 = vector.transfer_read %arg0[%c0, %c0], %cf0 {permutation_map = #map0} : memref<6x4xf32>, vector<6x4xf32>
  return %0 : vector<6x4xf32>
}

// -----

// CHECK-LABEL: func @transfer_read_unroll_broadcast_permuation
//       CHECK-DAG:   %[[C4:.*]] = constant 4 : index
//       CHECK-DAG:   %[[C2:.*]] = constant 2 : index
//       CHECK-DAG:   %[[C0:.*]] = constant 0 : index
//       CHECK:   %[[VTR0:.*]] = vector.transfer_read {{.*}}[%[[C0]], %[[C0]]], %{{.*}} : memref<6x4xf32>, vector<2x2xf32>
//  CHECK-NEXT:   %[[VEC0:.*]] = vector.insert_strided_slice %[[VTR0]], %{{.*}} {offsets = [0, 0], strides = [1, 1]} : vector<2x2xf32> into vector<4x6xf32>
//  CHECK-NEXT:   %[[VTR1:.*]] = vector.transfer_read {{.*}}[%[[C2]], %[[C0]]], %{{.*}} : memref<6x4xf32>, vector<2x2xf32>
//  CHECK-NEXT:   %[[VEC1:.*]] = vector.insert_strided_slice %[[VTR1]], %[[VEC0]] {offsets = [0, 2], strides = [1, 1]} : vector<2x2xf32> into vector<4x6xf32>
//  CHECK-NEXT:   %[[VTR2:.*]] = vector.transfer_read {{.*}}[%[[C4]], %[[C0]]], %{{.*}} : memref<6x4xf32>, vector<2x2xf32>
//  CHECK-NEXT:   %[[VEC2:.*]] = vector.insert_strided_slice %[[VTR2]], %[[VEC1]] {offsets = [0, 4], strides = [1, 1]} : vector<2x2xf32> into vector<4x6xf32>
//  CHECK-NEXT:   %[[VTR3:.*]] = vector.transfer_read {{.*}}[%[[C0]], %[[C0]]], %{{.*}} : memref<6x4xf32>, vector<2x2xf32>
//  CHECK-NEXT:   %[[VEC3:.*]] = vector.insert_strided_slice %[[VTR3]], %[[VEC2]] {offsets = [2, 0], strides = [1, 1]} : vector<2x2xf32> into vector<4x6xf32>
//  CHECK-NEXT:   %[[VTR4:.*]] = vector.transfer_read {{.*}}[%[[C2]], %[[C0]]], %{{.*}} : memref<6x4xf32>, vector<2x2xf32>
//  CHECK-NEXT:   %[[VEC4:.*]] = vector.insert_strided_slice %[[VTR4]], %[[VEC3]] {offsets = [2, 2], strides = [1, 1]} : vector<2x2xf32> into vector<4x6xf32>
//  CHECK-NEXT:   %[[VTR5:.*]] = vector.transfer_read {{.*}}[%[[C4]], %[[C0]]], %{{.*}} : memref<6x4xf32>, vector<2x2xf32>
//  CHECK-NEXT:   %[[VEC5:.*]] = vector.insert_strided_slice %[[VTR5]], %[[VEC4]] {offsets = [2, 4], strides = [1, 1]} : vector<2x2xf32> into vector<4x6xf32>
//  CHECK-NEXT:   return %[[VEC5]] : vector<4x6xf32>
#map0 = affine_map<(d0, d1) -> (0, d0)>
func @transfer_read_unroll_broadcast_permuation(%arg0 : memref<6x4xf32>) -> vector<4x6xf32> {
  %c0 = constant 0 : index
  %cf0 = constant 0.0 : f32
  %0 = vector.transfer_read %arg0[%c0, %c0], %cf0 {permutation_map = #map0} : memref<6x4xf32>, vector<4x6xf32>
  return %0 : vector<4x6xf32>
}

// -----

// CHECK-LABEL: func @transfer_read_unroll_different_rank
//       CHECK-DAG:   %[[C4:.*]] = constant 4 : index
//       CHECK-DAG:   %[[C2:.*]] = constant 2 : index
//       CHECK-DAG:   %[[C0:.*]] = constant 0 : index
//       CHECK:   %[[VTR0:.*]] = vector.transfer_read {{.*}}[%[[C0]], %[[C0]], %[[C0]]], %{{.*}} : memref<?x?x?xf32>, vector<2x2xf32>
//  CHECK-NEXT:   %[[VEC0:.*]] = vector.insert_strided_slice %[[VTR0]], %{{.*}} {offsets = [0, 0], strides = [1, 1]} : vector<2x2xf32> into vector<6x4xf32>
//  CHECK-NEXT:   %[[VTR1:.*]] = vector.transfer_read {{.*}}[%[[C2]], %[[C0]], %[[C0]]], %{{.*}} : memref<?x?x?xf32>, vector<2x2xf32>
//  CHECK-NEXT:   %[[VEC1:.*]] = vector.insert_strided_slice %[[VTR1]], %[[VEC0]] {offsets = [0, 2], strides = [1, 1]} : vector<2x2xf32> into vector<6x4xf32>
//  CHECK-NEXT:   %[[VTR2:.*]] = vector.transfer_read {{.*}}[%[[C0]], %[[C0]], %[[C2]]], %{{.*}} : memref<?x?x?xf32>, vector<2x2xf32>
//  CHECK-NEXT:   %[[VEC2:.*]] = vector.insert_strided_slice %[[VTR2]], %[[VEC1]] {offsets = [2, 0], strides = [1, 1]} : vector<2x2xf32> into vector<6x4xf32>
//  CHECK-NEXT:   %[[VTR3:.*]] = vector.transfer_read {{.*}}[%[[C2]], %[[C0]], %[[C2]]], %{{.*}} : memref<?x?x?xf32>, vector<2x2xf32>
//  CHECK-NEXT:   %[[VEC3:.*]] = vector.insert_strided_slice %[[VTR3]], %[[VEC2]] {offsets = [2, 2], strides = [1, 1]} : vector<2x2xf32> into vector<6x4xf32>
//  CHECK-NEXT:   %[[VTR4:.*]] = vector.transfer_read {{.*}}[%[[C0]], %[[C0]], %[[C4]]], %{{.*}} : memref<?x?x?xf32>, vector<2x2xf32>
//  CHECK-NEXT:   %[[VEC4:.*]] = vector.insert_strided_slice %[[VTR4]], %[[VEC3]] {offsets = [4, 0], strides = [1, 1]} : vector<2x2xf32> into vector<6x4xf32>
//  CHECK-NEXT:   %[[VTR5:.*]] = vector.transfer_read {{.*}}[%[[C2]], %[[C0]], %[[C4]]], %{{.*}} : memref<?x?x?xf32>, vector<2x2xf32>
//  CHECK-NEXT:   %[[VEC5:.*]] = vector.insert_strided_slice %[[VTR5]], %[[VEC4]] {offsets = [4, 2], strides = [1, 1]} : vector<2x2xf32> into vector<6x4xf32>
//  CHECK-NEXT:   return %[[VEC5]] : vector<6x4xf32>
#map0 = affine_map<(d0, d1, d2) -> (d2, d0)>
func @transfer_read_unroll_different_rank(%arg0 : memref<?x?x?xf32>) -> vector<6x4xf32> {
  %c0 = constant 0 : index
  %cf0 = constant 0.0 : f32
  %0 = vector.transfer_read %arg0[%c0, %c0, %c0], %cf0 {permutation_map = #map0} : memref<?x?x?xf32>, vector<6x4xf32>
  return %0 : vector<6x4xf32>
}
