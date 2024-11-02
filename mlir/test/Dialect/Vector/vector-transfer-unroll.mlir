// RUN: mlir-opt %s -test-vector-transfer-unrolling-patterns --split-input-file | FileCheck %s --check-prefix=ALL
// RUN: mlir-opt %s -test-vector-transfer-unrolling-patterns=reverse-unroll-order --split-input-file | FileCheck %s --check-prefixes=ALL,ORDER

// ALL-LABEL:   func @transfer_read_unroll
// CHECK-DAG:     %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[VTR0:.*]] = vector.transfer_read {{.*}}[%[[C0]], %[[C0]]], %{{.*}} : memref<4x4xf32>, vector<2x2xf32>
// CHECK-NEXT:    %[[VEC0:.*]] = vector.insert_strided_slice %[[VTR0]], %{{.*}} {offsets = [0, 0], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>
// CHECK-NEXT:    %[[VTR1:.*]] = vector.transfer_read {{.*}}[%[[C0]], %[[C2]]], %{{.*}} : memref<4x4xf32>, vector<2x2xf32>
// CHECK-NEXT:    %[[VEC1:.*]] = vector.insert_strided_slice %[[VTR1]], %[[VEC0]] {offsets = [0, 2], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>
// CHECK-NEXT:    %[[VTR2:.*]] = vector.transfer_read {{.*}}[%[[C2]], %[[C0]]], %{{.*}} : memref<4x4xf32>, vector<2x2xf32>
// CHECK-NEXT:    %[[VEC2:.*]] = vector.insert_strided_slice %[[VTR2]], %[[VEC1]] {offsets = [2, 0], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>
// CHECK-NEXT:    %[[VTR3:.*]] = vector.transfer_read {{.*}}[%[[C2]], %[[C2]]], %{{.*}} : memref<4x4xf32>, vector<2x2xf32>
// CHECK-NEXT:    %[[VEC3:.*]] = vector.insert_strided_slice %[[VTR3]], %[[VEC2]] {offsets = [2, 2], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>
// CHECK-NEXT:    return %[[VEC3]] : vector<4x4xf32>

// ORDER-DAG:     %[[C2:.*]] = arith.constant 2 : index
// ORDER-DAG:     %[[C0:.*]] = arith.constant 0 : index
// ORDER:         %[[VTR0:.*]] = vector.transfer_read {{.*}}[%[[C0]], %[[C0]]], %{{.*}} : memref<4x4xf32>, vector<2x2xf32>
// ORDER-NEXT:    %[[VEC0:.*]] = vector.insert_strided_slice %[[VTR0]], %{{.*}} {offsets = [0, 0], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>
// ORDER-NEXT:    %[[VTR1:.*]] = vector.transfer_read {{.*}}[%[[C2]], %[[C0]]], %{{.*}} : memref<4x4xf32>, vector<2x2xf32>
// ORDER-NEXT:    %[[VEC1:.*]] = vector.insert_strided_slice %[[VTR1]], %[[VEC0]] {offsets = [2, 0], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>
// ORDER-NEXT:    %[[VTR2:.*]] = vector.transfer_read {{.*}}[%[[C0]], %[[C2]]], %{{.*}} : memref<4x4xf32>, vector<2x2xf32>
// ORDER-NEXT:    %[[VEC2:.*]] = vector.insert_strided_slice %[[VTR2]], %[[VEC1]] {offsets = [0, 2], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>
// ORDER-NEXT:    %[[VTR3:.*]] = vector.transfer_read {{.*}}[%[[C2]], %[[C2]]], %{{.*}} : memref<4x4xf32>, vector<2x2xf32>
// ORDER-NEXT:    %[[VEC3:.*]] = vector.insert_strided_slice %[[VTR3]], %[[VEC2]] {offsets = [2, 2], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>
// ORDER-NEXT:    return %[[VEC3]] : vector<4x4xf32>

func.func @transfer_read_unroll(%mem : memref<4x4xf32>) -> vector<4x4xf32> {
  %c0 = arith.constant 0 : index
  %cf0 = arith.constant 0.0 : f32
  %res = vector.transfer_read %mem[%c0, %c0], %cf0 : memref<4x4xf32>, vector<4x4xf32>
  return %res : vector<4x4xf32>
}

// -----

// ALL-LABEL:   func @transfer_write_unroll
// CHECK-DAG:     %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[S0:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
// CHECK-NEXT:    vector.transfer_write %[[S0]], {{.*}}[%[[C0]], %[[C0]]] {{.*}} : vector<2x2xf32>, memref<4x4xf32>
// CHECK-NEXT:    %[[S1:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [0, 2], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
// CHECK-NEXT:    vector.transfer_write %[[S1]], {{.*}}[%[[C0]], %[[C2]]] {{.*}} : vector<2x2xf32>, memref<4x4xf32>
// CHECK-NEXT:    %[[S2:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
// CHECK-NEXT:    vector.transfer_write %[[S2]], {{.*}}[%[[C2]], %[[C0]]] {{.*}} : vector<2x2xf32>, memref<4x4xf32>
// CHECK-NEXT:    %[[S3:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [2, 2], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
// CHECK-NEXT:    vector.transfer_write %[[S3]], {{.*}}[%[[C2]], %[[C2]]] {{.*}} : vector<2x2xf32>, memref<4x4xf32>
// CHECK-NEXT:    return

// ORDER-DAG:     %[[C2:.*]] = arith.constant 2 : index
// ORDER-DAG:     %[[C0:.*]] = arith.constant 0 : index
// ORDER:         %[[S0:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
// ORDER-NEXT:    vector.transfer_write %[[S0]], {{.*}}[%[[C0]], %[[C0]]] {{.*}} : vector<2x2xf32>, memref<4x4xf32>
// ORDER-NEXT:    %[[S1:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
// ORDER-NEXT:    vector.transfer_write %[[S1]], {{.*}}[%[[C2]], %[[C0]]] {{.*}} : vector<2x2xf32>, memref<4x4xf32>
// ORDER-NEXT:    %[[S2:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [0, 2], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
// ORDER-NEXT:    vector.transfer_write %[[S2]], {{.*}}[%[[C0]], %[[C2]]] {{.*}} : vector<2x2xf32>, memref<4x4xf32>
// ORDER-NEXT:    %[[S3:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [2, 2], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
// ORDER-NEXT:    vector.transfer_write %[[S3]], {{.*}}[%[[C2]], %[[C2]]] {{.*}} : vector<2x2xf32>, memref<4x4xf32>
// ORDER-NEXT:    return

func.func @transfer_write_unroll(%mem : memref<4x4xf32>, %vec : vector<4x4xf32>) {
  %c0 = arith.constant 0 : index
  vector.transfer_write %vec, %mem[%c0, %c0] : vector<4x4xf32>, memref<4x4xf32>
  return
}

// -----

// CHECK-LABEL: func @transfer_readwrite_unroll
// CHECK-DAG:     %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[VTR0:.*]] = vector.transfer_read {{.*}}[%[[C0]], %[[C0]]], %{{.*}} : memref<4x4xf32>, vector<2x2xf32>
// CHECK-NEXT:    %[[VTR1:.*]] = vector.transfer_read {{.*}}[%[[C0]], %[[C2]]], %{{.*}} : memref<4x4xf32>, vector<2x2xf32>
// CHECK-NEXT:    %[[VTR2:.*]] = vector.transfer_read {{.*}}[%[[C2]], %[[C0]]], %{{.*}} : memref<4x4xf32>, vector<2x2xf32>
// CHECK-NEXT:    %[[VTR3:.*]] = vector.transfer_read {{.*}}[%[[C2]], %[[C2]]], %{{.*}} : memref<4x4xf32>, vector<2x2xf32>
// CHECK-NEXT:    vector.transfer_write %[[VTR0]], {{.*}}[%[[C0]], %[[C0]]] {{.*}} : vector<2x2xf32>, memref<4x4xf32>
// CHECK-NEXT:    vector.transfer_write %[[VTR1]], {{.*}}[%[[C0]], %[[C2]]] {{.*}} : vector<2x2xf32>, memref<4x4xf32>
// CHECK-NEXT:    vector.transfer_write %[[VTR2]], {{.*}}[%[[C2]], %[[C0]]] {{.*}} : vector<2x2xf32>, memref<4x4xf32>
// CHECK-NEXT:    vector.transfer_write %[[VTR3]], {{.*}}[%[[C2]], %[[C2]]] {{.*}} : vector<2x2xf32>, memref<4x4xf32>
// CHECK-NEXT:    return

func.func @transfer_readwrite_unroll(%mem : memref<4x4xf32>) {
  %c0 = arith.constant 0 : index
  %cf0 = arith.constant 0.0 : f32
  %0 = vector.transfer_read %mem[%c0, %c0], %cf0 : memref<4x4xf32>, vector<4x4xf32>
  vector.transfer_write %0, %mem[%c0, %c0] : vector<4x4xf32>, memref<4x4xf32>
  return
}

// -----

// CHECK-LABEL: func @transfer_read_unroll_tensor
// CHECK-DAG:     %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[VTR0:.*]] = vector.transfer_read {{.*}}[%[[C0]], %[[C0]]], %{{.*}} : tensor<4x4xf32>, vector<2x2xf32>
// CHECK-NEXT:    %[[VEC0:.*]] = vector.insert_strided_slice %[[VTR0]], %{{.*}} {offsets = [0, 0], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>
// CHECK-NEXT:    %[[VTR1:.*]] = vector.transfer_read {{.*}}[%[[C0]], %[[C2]]], %{{.*}} : tensor<4x4xf32>, vector<2x2xf32>
// CHECK-NEXT:    %[[VEC1:.*]] = vector.insert_strided_slice %[[VTR1]], %[[VEC0]] {offsets = [0, 2], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>
// CHECK-NEXT:    %[[VTR2:.*]] = vector.transfer_read {{.*}}[%[[C2]], %[[C0]]], %{{.*}} : tensor<4x4xf32>, vector<2x2xf32>
// CHECK-NEXT:    %[[VEC2:.*]] = vector.insert_strided_slice %[[VTR2]], %[[VEC1]] {offsets = [2, 0], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>
// CHECK-NEXT:    %[[VTR3:.*]] = vector.transfer_read {{.*}}[%[[C2]], %[[C2]]], %{{.*}} : tensor<4x4xf32>, vector<2x2xf32>
// CHECK-NEXT:    %[[VEC3:.*]] = vector.insert_strided_slice %[[VTR3]], %[[VEC2]] {offsets = [2, 2], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>
// CHECK-NEXT:    return %[[VEC3]] : vector<4x4xf32>

func.func @transfer_read_unroll_tensor(%arg0 : tensor<4x4xf32>) -> vector<4x4xf32> {
  %c0 = arith.constant 0 : index
  %cf0 = arith.constant 0.0 : f32
  %res = vector.transfer_read %arg0[%c0, %c0], %cf0 : tensor<4x4xf32>, vector<4x4xf32>
  return %res : vector<4x4xf32>
}

// -----

// CHECK-LABEL: func @transfer_write_unroll_tensor
// CHECK-DAG:     %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[S0:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
// CHECK-NEXT:    %[[VTW0:.*]] = vector.transfer_write %[[S0]], {{.*}}[%[[C0]], %[[C0]]] {{.*}} : vector<2x2xf32>, tensor<4x4xf32>
// CHECK-NEXT:    %[[S1:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [0, 2], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
// CHECK-NEXT:    %[[VTW1:.*]] = vector.transfer_write %[[S1]], %[[VTW0]][%[[C0]], %[[C2]]] {{.*}} : vector<2x2xf32>, tensor<4x4xf32>
// CHECK-NEXT:    %[[S2:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
// CHECK-NEXT:    %[[VTW2:.*]] = vector.transfer_write %[[S2]], %[[VTW1]][%[[C2]], %[[C0]]] {{.*}} : vector<2x2xf32>, tensor<4x4xf32>
// CHECK-NEXT:    %[[S3:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [2, 2], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf32> to vector<2x2xf32>
// CHECK-NEXT:    %[[VTW3:.*]] = vector.transfer_write %[[S3]], %[[VTW2]][%[[C2]], %[[C2]]] {{.*}} : vector<2x2xf32>, tensor<4x4xf32>
// CHECK-NEXT:    return %[[VTW3]] : tensor<4x4xf32>

func.func @transfer_write_unroll_tensor(%arg0 : tensor<4x4xf32>,
  %vec : vector<4x4xf32>) -> tensor<4x4xf32> {
  %c0 = arith.constant 0 : index
  %res = vector.transfer_write %vec, %arg0[%c0, %c0] :
    vector<4x4xf32>, tensor<4x4xf32>
  return %res: tensor<4x4xf32>
}

// -----

// CHECK-LABEL: func @transfer_readwrite_unroll_tensor
// CHECK-DAG:     %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[VTR0:.*]] = vector.transfer_read {{.*}}[%[[C0]], %[[C0]]], %{{.*}} : tensor<4x4xf32>, vector<2x2xf32>
// CHECK-NEXT:    %[[VTR1:.*]] = vector.transfer_read {{.*}}[%[[C0]], %[[C2]]], %{{.*}} : tensor<4x4xf32>, vector<2x2xf32>
// CHECK-NEXT:    %[[VTR2:.*]] = vector.transfer_read {{.*}}[%[[C2]], %[[C0]]], %{{.*}} : tensor<4x4xf32>, vector<2x2xf32>
// CHECK-NEXT:    %[[VTR3:.*]] = vector.transfer_read {{.*}}[%[[C2]], %[[C2]]], %{{.*}} : tensor<4x4xf32>, vector<2x2xf32>
// CHECK-NEXT:    %[[VTW0:.*]] = vector.transfer_write %[[VTR0]], {{.*}}[%[[C0]], %[[C0]]] {{.*}} : vector<2x2xf32>, tensor<4x4xf32>
// CHECK-NEXT:    %[[VTW1:.*]] = vector.transfer_write %[[VTR1]], %[[VTW0]][%[[C0]], %[[C2]]] {{.*}} : vector<2x2xf32>, tensor<4x4xf32>
// CHECK-NEXT:    %[[VTW2:.*]] = vector.transfer_write %[[VTR2]], %[[VTW1]][%[[C2]], %[[C0]]] {{.*}} : vector<2x2xf32>, tensor<4x4xf32>
// CHECK-NEXT:    %[[VTW3:.*]] = vector.transfer_write %[[VTR3]], %[[VTW2]][%[[C2]], %[[C2]]] {{.*}} : vector<2x2xf32>, tensor<4x4xf32>
// CHECK-NEXT:    return %[[VTW3]] : tensor<4x4xf32>

func.func @transfer_readwrite_unroll_tensor(%arg0 : tensor<4x4xf32>, %arg1 : tensor<4x4xf32>) ->
  tensor<4x4xf32> {
  %c0 = arith.constant 0 : index
  %cf0 = arith.constant 0.0 : f32
  %0 = vector.transfer_read %arg0[%c0, %c0], %cf0 : tensor<4x4xf32>, vector<4x4xf32>
  %res = vector.transfer_write %0, %arg1[%c0, %c0] : vector<4x4xf32>, tensor<4x4xf32>
  return %res: tensor<4x4xf32>
}

// -----

// CHECK-LABEL: func @transfer_read_unroll_permutation
// CHECK-DAG:     %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:     %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[VTR0:.*]] = vector.transfer_read {{.*}}[%[[C0]], %[[C0]]], %{{.*}} : memref<6x4xf32>, vector<2x2xf32>
// CHECK-NEXT:    %[[VEC0:.*]] = vector.insert_strided_slice %[[VTR0]], %{{.*}} {offsets = [0, 0], strides = [1, 1]} : vector<2x2xf32> into vector<4x6xf32>
// CHECK-NEXT:    %[[VTR1:.*]] = vector.transfer_read {{.*}}[%[[C2]], %[[C0]]], %{{.*}} : memref<6x4xf32>, vector<2x2xf32>
// CHECK-NEXT:    %[[VEC1:.*]] = vector.insert_strided_slice %[[VTR1]], %[[VEC0]] {offsets = [0, 2], strides = [1, 1]} : vector<2x2xf32> into vector<4x6xf32>
// CHECK-NEXT:    %[[VTR2:.*]] = vector.transfer_read {{.*}}[%[[C4]], %[[C0]]], %{{.*}} : memref<6x4xf32>, vector<2x2xf32>
// CHECK-NEXT:    %[[VEC2:.*]] = vector.insert_strided_slice %[[VTR2]], %[[VEC1]] {offsets = [0, 4], strides = [1, 1]} : vector<2x2xf32> into vector<4x6xf32>
// CHECK-NEXT:    %[[VTR3:.*]] = vector.transfer_read {{.*}}[%[[C0]], %[[C2]]], %{{.*}} : memref<6x4xf32>, vector<2x2xf32>
// CHECK-NEXT:    %[[VEC3:.*]] = vector.insert_strided_slice %[[VTR3]], %[[VEC2]] {offsets = [2, 0], strides = [1, 1]} : vector<2x2xf32> into vector<4x6xf32>
// CHECK-NEXT:    %[[VTR4:.*]] = vector.transfer_read {{.*}}[%[[C2]], %[[C2]]], %{{.*}} : memref<6x4xf32>, vector<2x2xf32>
// CHECK-NEXT:    %[[VEC4:.*]] = vector.insert_strided_slice %[[VTR4]], %[[VEC3]] {offsets = [2, 2], strides = [1, 1]} : vector<2x2xf32> into vector<4x6xf32>
// CHECK-NEXT:    %[[VTR5:.*]] = vector.transfer_read {{.*}}[%[[C4]], %[[C2]]], %{{.*}} : memref<6x4xf32>, vector<2x2xf32>
// CHECK-NEXT:    %[[VEC5:.*]] = vector.insert_strided_slice %[[VTR5]], %[[VEC4]] {offsets = [2, 4], strides = [1, 1]} : vector<2x2xf32> into vector<4x6xf32>
// CHECK-NEXT:    return %[[VEC5]] : vector<4x6xf32>
#map0 = affine_map<(d0, d1) -> (d1, d0)>
func.func @transfer_read_unroll_permutation(%mem : memref<6x4xf32>) -> vector<4x6xf32> {
  %c0 = arith.constant 0 : index
  %cf0 = arith.constant 0.0 : f32
  %res = vector.transfer_read %mem[%c0, %c0], %cf0 {permutation_map = #map0} : memref<6x4xf32>, vector<4x6xf32>
  return %res : vector<4x6xf32>
}

// -----

// CHECK-LABEL: func @transfer_read_unroll_broadcast
// CHECK-DAG:     %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[VTR0:.*]] = vector.transfer_read {{.*}}[%[[C0]], %[[C0]]], %{{.*}} : memref<6x4xf32>, vector<2x2xf32>
// CHECK-NEXT:    %[[VEC0:.*]] = vector.insert_strided_slice %[[VTR0]], %{{.*}} {offsets = [0, 0], strides = [1, 1]} : vector<2x2xf32> into vector<6x4xf32>
// CHECK-NEXT:    %[[VTR1:.*]] = vector.transfer_read {{.*}}[%[[C0]], %[[C2]]], %{{.*}} : memref<6x4xf32>, vector<2x2xf32>
// CHECK-NEXT:    %[[VEC1:.*]] = vector.insert_strided_slice %[[VTR1]], %[[VEC0]] {offsets = [0, 2], strides = [1, 1]} : vector<2x2xf32> into vector<6x4xf32>
// CHECK-NEXT:    %[[VTR2:.*]] = vector.transfer_read {{.*}}[%[[C0]], %[[C0]]], %{{.*}} : memref<6x4xf32>, vector<2x2xf32>
// CHECK-NEXT:    %[[VEC2:.*]] = vector.insert_strided_slice %[[VTR2]], %[[VEC1]] {offsets = [2, 0], strides = [1, 1]} : vector<2x2xf32> into vector<6x4xf32>
// CHECK-NEXT:    %[[VTR3:.*]] = vector.transfer_read {{.*}}[%[[C0]], %[[C2]]], %{{.*}} : memref<6x4xf32>, vector<2x2xf32>
// CHECK-NEXT:    %[[VEC3:.*]] = vector.insert_strided_slice %[[VTR3]], %[[VEC2]] {offsets = [2, 2], strides = [1, 1]} : vector<2x2xf32> into vector<6x4xf32>
// CHECK-NEXT:    %[[VTR4:.*]] = vector.transfer_read {{.*}}[%[[C0]], %[[C0]]], %{{.*}} : memref<6x4xf32>, vector<2x2xf32>
// CHECK-NEXT:    %[[VEC4:.*]] = vector.insert_strided_slice %[[VTR4]], %[[VEC3]] {offsets = [4, 0], strides = [1, 1]} : vector<2x2xf32> into vector<6x4xf32>
// CHECK-NEXT:    %[[VTR5:.*]] = vector.transfer_read {{.*}}[%[[C0]], %[[C2]]], %{{.*}} : memref<6x4xf32>, vector<2x2xf32>
// CHECK-NEXT:    %[[VEC5:.*]] = vector.insert_strided_slice %[[VTR5]], %[[VEC4]] {offsets = [4, 2], strides = [1, 1]} : vector<2x2xf32> into vector<6x4xf32>
// CHECK-NEXT:    return %[[VEC5]] : vector<6x4xf32>
#map0 = affine_map<(d0, d1) -> (0, d1)>
func.func @transfer_read_unroll_broadcast(%mem : memref<6x4xf32>) -> vector<6x4xf32> {
  %c0 = arith.constant 0 : index
  %cf0 = arith.constant 0.0 : f32
  %res = vector.transfer_read %mem[%c0, %c0], %cf0 {in_bounds = [true, false], permutation_map = #map0} : memref<6x4xf32>, vector<6x4xf32>
  return %res : vector<6x4xf32>
}

// -----

// CHECK-LABEL: func @transfer_read_unroll_broadcast_permuation
// CHECK-DAG:     %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:     %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[VTR0:.*]] = vector.transfer_read {{.*}}[%[[C0]], %[[C0]]], %{{.*}} : memref<6x4xf32>, vector<2x2xf32>
// CHECK-NEXT:    %[[VEC0:.*]] = vector.insert_strided_slice %[[VTR0]], %{{.*}} {offsets = [0, 0], strides = [1, 1]} : vector<2x2xf32> into vector<4x6xf32>
// CHECK-NEXT:    %[[VTR1:.*]] = vector.transfer_read {{.*}}[%[[C2]], %[[C0]]], %{{.*}} : memref<6x4xf32>, vector<2x2xf32>
// CHECK-NEXT:    %[[VEC1:.*]] = vector.insert_strided_slice %[[VTR1]], %[[VEC0]] {offsets = [0, 2], strides = [1, 1]} : vector<2x2xf32> into vector<4x6xf32>
// CHECK-NEXT:    %[[VTR2:.*]] = vector.transfer_read {{.*}}[%[[C4]], %[[C0]]], %{{.*}} : memref<6x4xf32>, vector<2x2xf32>
// CHECK-NEXT:    %[[VEC2:.*]] = vector.insert_strided_slice %[[VTR2]], %[[VEC1]] {offsets = [0, 4], strides = [1, 1]} : vector<2x2xf32> into vector<4x6xf32>
// CHECK-NEXT:    %[[VTR3:.*]] = vector.transfer_read {{.*}}[%[[C0]], %[[C0]]], %{{.*}} : memref<6x4xf32>, vector<2x2xf32>
// CHECK-NEXT:    %[[VEC3:.*]] = vector.insert_strided_slice %[[VTR3]], %[[VEC2]] {offsets = [2, 0], strides = [1, 1]} : vector<2x2xf32> into vector<4x6xf32>
// CHECK-NEXT:    %[[VTR4:.*]] = vector.transfer_read {{.*}}[%[[C2]], %[[C0]]], %{{.*}} : memref<6x4xf32>, vector<2x2xf32>
// CHECK-NEXT:    %[[VEC4:.*]] = vector.insert_strided_slice %[[VTR4]], %[[VEC3]] {offsets = [2, 2], strides = [1, 1]} : vector<2x2xf32> into vector<4x6xf32>
// CHECK-NEXT:    %[[VTR5:.*]] = vector.transfer_read {{.*}}[%[[C4]], %[[C0]]], %{{.*}} : memref<6x4xf32>, vector<2x2xf32>
// CHECK-NEXT:    %[[VEC5:.*]] = vector.insert_strided_slice %[[VTR5]], %[[VEC4]] {offsets = [2, 4], strides = [1, 1]} : vector<2x2xf32> into vector<4x6xf32>
// CHECK-NEXT:    return %[[VEC5]] : vector<4x6xf32>
#map0 = affine_map<(d0, d1) -> (0, d0)>
func.func @transfer_read_unroll_broadcast_permuation(%mem : memref<6x4xf32>) -> vector<4x6xf32> {
  %c0 = arith.constant 0 : index
  %cf0 = arith.constant 0.0 : f32
  %res = vector.transfer_read %mem[%c0, %c0], %cf0 {in_bounds = [true, false], permutation_map = #map0} : memref<6x4xf32>, vector<4x6xf32>
  return %res : vector<4x6xf32>
}

// -----

// ALL-LABEL:   func @transfer_read_unroll_different_rank
// CHECK-DAG:     %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:     %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[VTR0:.*]] = vector.transfer_read {{.*}}[%[[C0]], %[[C0]], %[[C0]]], %{{.*}} : memref<?x?x?xf32>, vector<2x2xf32>
// CHECK-NEXT:    %[[VEC0:.*]] = vector.insert_strided_slice %[[VTR0]], %{{.*}} {offsets = [0, 0], strides = [1, 1]} : vector<2x2xf32> into vector<6x4xf32>
// CHECK-NEXT:    %[[VTR1:.*]] = vector.transfer_read {{.*}}[%[[C2]], %[[C0]], %[[C0]]], %{{.*}} : memref<?x?x?xf32>, vector<2x2xf32>
// CHECK-NEXT:    %[[VEC1:.*]] = vector.insert_strided_slice %[[VTR1]], %[[VEC0]] {offsets = [0, 2], strides = [1, 1]} : vector<2x2xf32> into vector<6x4xf32>
// CHECK-NEXT:    %[[VTR2:.*]] = vector.transfer_read {{.*}}[%[[C0]], %[[C0]], %[[C2]]], %{{.*}} : memref<?x?x?xf32>, vector<2x2xf32>
// CHECK-NEXT:    %[[VEC2:.*]] = vector.insert_strided_slice %[[VTR2]], %[[VEC1]] {offsets = [2, 0], strides = [1, 1]} : vector<2x2xf32> into vector<6x4xf32>
// CHECK-NEXT:    %[[VTR3:.*]] = vector.transfer_read {{.*}}[%[[C2]], %[[C0]], %[[C2]]], %{{.*}} : memref<?x?x?xf32>, vector<2x2xf32>
// CHECK-NEXT:    %[[VEC3:.*]] = vector.insert_strided_slice %[[VTR3]], %[[VEC2]] {offsets = [2, 2], strides = [1, 1]} : vector<2x2xf32> into vector<6x4xf32>
// CHECK-NEXT:    %[[VTR4:.*]] = vector.transfer_read {{.*}}[%[[C0]], %[[C0]], %[[C4]]], %{{.*}} : memref<?x?x?xf32>, vector<2x2xf32>
// CHECK-NEXT:    %[[VEC4:.*]] = vector.insert_strided_slice %[[VTR4]], %[[VEC3]] {offsets = [4, 0], strides = [1, 1]} : vector<2x2xf32> into vector<6x4xf32>
// CHECK-NEXT:    %[[VTR5:.*]] = vector.transfer_read {{.*}}[%[[C2]], %[[C0]], %[[C4]]], %{{.*}} : memref<?x?x?xf32>, vector<2x2xf32>
// CHECK-NEXT:    %[[VEC5:.*]] = vector.insert_strided_slice %[[VTR5]], %[[VEC4]] {offsets = [4, 2], strides = [1, 1]} : vector<2x2xf32> into vector<6x4xf32>
// CHECK-NEXT:    return %[[VEC5]] : vector<6x4xf32>

// ORDER-DAG:     %[[C4:.*]] = arith.constant 4 : index
// ORDER-DAG:     %[[C2:.*]] = arith.constant 2 : index
// ORDER-DAG:     %[[C0:.*]] = arith.constant 0 : index
// ORDER:         %[[VTR0:.*]] = vector.transfer_read {{.*}}[%[[C0]], %[[C0]], %[[C0]]], %{{.*}} : memref<?x?x?xf32>, vector<2x2xf32>
// ORDER-NEXT:    %[[VEC0:.*]] = vector.insert_strided_slice %[[VTR0]], %{{.*}} {offsets = [0, 0], strides = [1, 1]} : vector<2x2xf32> into vector<6x4xf32>
// ORDER-NEXT:    %[[VTR1:.*]] = vector.transfer_read {{.*}}[%[[C0]], %[[C0]], %[[C2]]], %{{.*}} : memref<?x?x?xf32>, vector<2x2xf32>
// ORDER-NEXT:    %[[VEC1:.*]] = vector.insert_strided_slice %[[VTR1]], %[[VEC0]] {offsets = [2, 0], strides = [1, 1]} : vector<2x2xf32> into vector<6x4xf32>
// ORDER-NEXT:    %[[VTR2:.*]] = vector.transfer_read {{.*}}[%[[C0]], %[[C0]], %[[C4]]], %{{.*}} : memref<?x?x?xf32>, vector<2x2xf32>
// ORDER-NEXT:    %[[VEC2:.*]] = vector.insert_strided_slice %[[VTR2]], %[[VEC1]] {offsets = [4, 0], strides = [1, 1]} : vector<2x2xf32> into vector<6x4xf32>
// ORDER-NEXT:    %[[VTR3:.*]] = vector.transfer_read {{.*}}[%[[C2]], %[[C0]], %[[C0]]], %{{.*}} : memref<?x?x?xf32>, vector<2x2xf32>
// ORDER-NEXT:    %[[VEC3:.*]] = vector.insert_strided_slice %[[VTR3]], %[[VEC2]] {offsets = [0, 2], strides = [1, 1]} : vector<2x2xf32> into vector<6x4xf32>
// ORDER-NEXT:    %[[VTR4:.*]] = vector.transfer_read {{.*}}[%[[C2]], %[[C0]], %[[C2]]], %{{.*}} : memref<?x?x?xf32>, vector<2x2xf32>
// ORDER-NEXT:    %[[VEC4:.*]] = vector.insert_strided_slice %[[VTR4]], %[[VEC3]] {offsets = [2, 2], strides = [1, 1]} : vector<2x2xf32> into vector<6x4xf32>
// ORDER-NEXT:    %[[VTR5:.*]] = vector.transfer_read {{.*}}[%[[C2]], %[[C0]], %[[C4]]], %{{.*}} : memref<?x?x?xf32>, vector<2x2xf32>
// ORDER-NEXT:    %[[VEC5:.*]] = vector.insert_strided_slice %[[VTR5]], %[[VEC4]] {offsets = [4, 2], strides = [1, 1]} : vector<2x2xf32> into vector<6x4xf32>
// ORDER-NEXT:    return %[[VEC5]] : vector<6x4xf32>

#map0 = affine_map<(d0, d1, d2) -> (d2, d0)>
func.func @transfer_read_unroll_different_rank(%mem : memref<?x?x?xf32>) -> vector<6x4xf32> {
  %c0 = arith.constant 0 : index
  %cf0 = arith.constant 0.0 : f32
  %res = vector.transfer_read %mem[%c0, %c0, %c0], %cf0 {permutation_map = #map0} : memref<?x?x?xf32>, vector<6x4xf32>
  return %res : vector<6x4xf32>
}

// -----

// ALL-LABEL:   func @vector_gather_unroll
// ALL-SAME:      %[[MEM:.*]]: memref<?x?x?xf32>
// ALL-SAME:      %[[INDICES:.*]]: vector<6x4xindex>
// ALL-SAME:      %[[MASK:.*]]: vector<6x4xi1>
// ALL-SAME:      %[[PT:.*]]: vector<6x4xf32>

// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[IDX0:.*]] = vector.extract_strided_slice %[[INDICES]] {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : vector<6x4xindex> to vector<2x2xindex>
// CHECK-NEXT:    %[[MASK0:.*]] = vector.extract_strided_slice %[[MASK]] {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : vector<6x4xi1> to vector<2x2xi1>
// CHECK-NEXT:    %[[PASS0:.*]] = vector.extract_strided_slice %[[PT]] {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : vector<6x4xf32> to vector<2x2xf32>
// CHECK-NEXT:    %[[VGT0:.*]] = vector.gather {{.*}}[%[[C0]], %[[C0]], %[[C0]]] [%[[IDX0]]], %[[MASK0]], %[[PASS0]] : memref<?x?x?xf32>, vector<2x2xindex>, vector<2x2xi1>, vector<2x2xf32> into vector<2x2xf32>
// CHECK-NEXT:    %[[VEC0:.*]] = vector.insert_strided_slice %[[VGT0]], %{{.*}} {offsets = [0, 0], strides = [1, 1]} : vector<2x2xf32> into vector<6x4xf32>
// CHECK-NEXT:    %[[IDX1:.*]] = vector.extract_strided_slice %[[INDICES]] {offsets = [0, 2], sizes = [2, 2], strides = [1, 1]} : vector<6x4xindex> to vector<2x2xindex>
// CHECK-NEXT:    %[[MASK1:.*]] = vector.extract_strided_slice %[[MASK]] {offsets = [0, 2], sizes = [2, 2], strides = [1, 1]} : vector<6x4xi1> to vector<2x2xi1>
// CHECK-NEXT:    %[[PASS1:.*]] = vector.extract_strided_slice %[[PT]] {offsets = [0, 2], sizes = [2, 2], strides = [1, 1]} : vector<6x4xf32> to vector<2x2xf32>
// CHECK-NEXT:    %[[VGT1:.*]] = vector.gather {{.*}}[%[[C0]], %[[C0]], %[[C0]]] [%[[IDX1]]], %[[MASK1]], %[[PASS1]] : memref<?x?x?xf32>, vector<2x2xindex>, vector<2x2xi1>, vector<2x2xf32> into vector<2x2xf32>
// CHECK-NEXT:    %[[VEC1:.*]] = vector.insert_strided_slice %[[VGT1]], %[[VEC0]] {offsets = [0, 2], strides = [1, 1]} : vector<2x2xf32> into vector<6x4xf32>
// CHECK-NEXT:    %[[IDX2:.*]] = vector.extract_strided_slice %[[INDICES]] {offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : vector<6x4xindex> to vector<2x2xindex>
// CHECK-NEXT:    %[[MASK2:.*]] = vector.extract_strided_slice %[[MASK]] {offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : vector<6x4xi1> to vector<2x2xi1>
// CHECK-NEXT:    %[[PASS2:.*]] = vector.extract_strided_slice %[[PT]] {offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : vector<6x4xf32> to vector<2x2xf32>
// CHECK-NEXT:    %[[VGT2:.*]] = vector.gather {{.*}}[%[[C0]], %[[C0]], %[[C0]]] [%[[IDX2]]], %[[MASK2]], %[[PASS2]] : memref<?x?x?xf32>, vector<2x2xindex>, vector<2x2xi1>, vector<2x2xf32> into vector<2x2xf32>
// CHECK-NEXT:    %[[VEC2:.*]] = vector.insert_strided_slice %[[VGT2]], %[[VEC1]] {offsets = [2, 0], strides = [1, 1]} : vector<2x2xf32> into vector<6x4xf32>
// CHECK-NEXT:    %[[IDX3:.*]] = vector.extract_strided_slice %[[INDICES]] {offsets = [2, 2], sizes = [2, 2], strides = [1, 1]} : vector<6x4xindex> to vector<2x2xindex>
// CHECK-NEXT:    %[[MASK3:.*]] = vector.extract_strided_slice %[[MASK]] {offsets = [2, 2], sizes = [2, 2], strides = [1, 1]} : vector<6x4xi1> to vector<2x2xi1>
// CHECK-NEXT:    %[[PASS3:.*]] = vector.extract_strided_slice %[[PT]] {offsets = [2, 2], sizes = [2, 2], strides = [1, 1]} : vector<6x4xf32> to vector<2x2xf32>
// CHECK-NEXT:    %[[VGT3:.*]] = vector.gather {{.*}}[%[[C0]], %[[C0]], %[[C0]]] [%[[IDX3]]], %[[MASK3]], %[[PASS3]] : memref<?x?x?xf32>, vector<2x2xindex>, vector<2x2xi1>, vector<2x2xf32> into vector<2x2xf32>
// CHECK-NEXT:    %[[VEC3:.*]] = vector.insert_strided_slice %[[VGT3]], %[[VEC2]] {offsets = [2, 2], strides = [1, 1]} : vector<2x2xf32> into vector<6x4xf32>
// CHECK-NEXT:    %[[IDX4:.*]] = vector.extract_strided_slice %[[INDICES]] {offsets = [4, 0], sizes = [2, 2], strides = [1, 1]} : vector<6x4xindex> to vector<2x2xindex>
// CHECK-NEXT:    %[[MASK4:.*]] = vector.extract_strided_slice %[[MASK]] {offsets = [4, 0], sizes = [2, 2], strides = [1, 1]} : vector<6x4xi1> to vector<2x2xi1>
// CHECK-NEXT:    %[[PASS4:.*]] = vector.extract_strided_slice %[[PT]] {offsets = [4, 0], sizes = [2, 2], strides = [1, 1]} : vector<6x4xf32> to vector<2x2xf32>
// CHECK-NEXT:    %[[VGT4:.*]] = vector.gather {{.*}}[%[[C0]], %[[C0]], %[[C0]]] [%[[IDX4]]], %[[MASK4]], %[[PASS4]] : memref<?x?x?xf32>, vector<2x2xindex>, vector<2x2xi1>, vector<2x2xf32> into vector<2x2xf32>
// CHECK-NEXT:    %[[VEC4:.*]] = vector.insert_strided_slice %[[VGT4]], %[[VEC3]] {offsets = [4, 0], strides = [1, 1]} : vector<2x2xf32> into vector<6x4xf32>
// CHECK-NEXT:    %[[IDX5:.*]] = vector.extract_strided_slice %[[INDICES]] {offsets = [4, 2], sizes = [2, 2], strides = [1, 1]} : vector<6x4xindex> to vector<2x2xindex>
// CHECK-NEXT:    %[[MASK5:.*]] = vector.extract_strided_slice %[[MASK]] {offsets = [4, 2], sizes = [2, 2], strides = [1, 1]} : vector<6x4xi1> to vector<2x2xi1>
// CHECK-NEXT:    %[[PASS5:.*]] = vector.extract_strided_slice %[[PT]] {offsets = [4, 2], sizes = [2, 2], strides = [1, 1]} : vector<6x4xf32> to vector<2x2xf32>
// CHECK-NEXT:    %[[VGT5:.*]] = vector.gather {{.*}}[%[[C0]], %[[C0]], %[[C0]]] [%[[IDX5]]], %[[MASK5]], %[[PASS5]] : memref<?x?x?xf32>, vector<2x2xindex>, vector<2x2xi1>, vector<2x2xf32> into vector<2x2xf32>
// CHECK-NEXT:    %[[VEC5:.*]] = vector.insert_strided_slice %[[VGT5]], %[[VEC4]] {offsets = [4, 2], strides = [1, 1]} : vector<2x2xf32> into vector<6x4xf32>
// CHECK-NEXT:    return %[[VEC5]] : vector<6x4xf32>

// ORDER-DAG:     %[[C0:.*]] = arith.constant 0 : index
// ORDER:         %[[IDX0:.*]] = vector.extract_strided_slice %[[INDICES]] {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : vector<6x4xindex> to vector<2x2xindex>
// ORDER-NEXT:    %[[MASK0:.*]] = vector.extract_strided_slice %[[MASK]] {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : vector<6x4xi1> to vector<2x2xi1>
// ORDER-NEXT:    %[[PASS0:.*]] = vector.extract_strided_slice %[[PT]] {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : vector<6x4xf32> to vector<2x2xf32>
// ORDER-NEXT:    %[[VGT0:.*]] = vector.gather {{.*}}[%[[C0]], %[[C0]], %[[C0]]] [%[[IDX0]]], %[[MASK0]], %[[PASS0]] : memref<?x?x?xf32>, vector<2x2xindex>, vector<2x2xi1>, vector<2x2xf32> into vector<2x2xf32>
// ORDER-NEXT:    %[[VEC0:.*]] = vector.insert_strided_slice %[[VGT0]], %{{.*}} {offsets = [0, 0], strides = [1, 1]} : vector<2x2xf32> into vector<6x4xf32>
// ORDER-NEXT:    %[[IDX1:.*]] = vector.extract_strided_slice %[[INDICES]] {offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : vector<6x4xindex> to vector<2x2xindex>
// ORDER-NEXT:    %[[MASK1:.*]] = vector.extract_strided_slice %[[MASK]] {offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : vector<6x4xi1> to vector<2x2xi1>
// ORDER-NEXT:    %[[PASS1:.*]] = vector.extract_strided_slice %[[PT]] {offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : vector<6x4xf32> to vector<2x2xf32>
// ORDER-NEXT:    %[[VGT1:.*]] = vector.gather {{.*}}[%[[C0]], %[[C0]], %[[C0]]] [%[[IDX1]]], %[[MASK1]], %[[PASS1]] : memref<?x?x?xf32>, vector<2x2xindex>, vector<2x2xi1>, vector<2x2xf32> into vector<2x2xf32>
// ORDER-NEXT:    %[[VEC1:.*]] = vector.insert_strided_slice %[[VGT1]], %[[VEC0]] {offsets = [2, 0], strides = [1, 1]} : vector<2x2xf32> into vector<6x4xf32>
// ORDER-NEXT:    %[[IDX2:.*]] = vector.extract_strided_slice %[[INDICES]] {offsets = [4, 0], sizes = [2, 2], strides = [1, 1]} : vector<6x4xindex> to vector<2x2xindex>
// ORDER-NEXT:    %[[MASK2:.*]] = vector.extract_strided_slice %[[MASK]] {offsets = [4, 0], sizes = [2, 2], strides = [1, 1]} : vector<6x4xi1> to vector<2x2xi1>
// ORDER-NEXT:    %[[PASS2:.*]] = vector.extract_strided_slice %[[PT]] {offsets = [4, 0], sizes = [2, 2], strides = [1, 1]} : vector<6x4xf32> to vector<2x2xf32>
// ORDER-NEXT:    %[[VGT2:.*]] = vector.gather {{.*}}[%[[C0]], %[[C0]], %[[C0]]] [%[[IDX2]]], %[[MASK2]], %[[PASS2]] : memref<?x?x?xf32>, vector<2x2xindex>, vector<2x2xi1>, vector<2x2xf32> into vector<2x2xf32>
// ORDER-NEXT:    %[[VEC2:.*]] = vector.insert_strided_slice %[[VGT2]], %[[VEC1]] {offsets = [4, 0], strides = [1, 1]} : vector<2x2xf32> into vector<6x4xf32>
// ORDER-NEXT:    %[[IDX3:.*]] = vector.extract_strided_slice %[[INDICES]] {offsets = [0, 2], sizes = [2, 2], strides = [1, 1]} : vector<6x4xindex> to vector<2x2xindex>
// ORDER-NEXT:    %[[MASK3:.*]] = vector.extract_strided_slice %[[MASK]] {offsets = [0, 2], sizes = [2, 2], strides = [1, 1]} : vector<6x4xi1> to vector<2x2xi1>
// ORDER-NEXT:    %[[PASS3:.*]] = vector.extract_strided_slice %[[PT]] {offsets = [0, 2], sizes = [2, 2], strides = [1, 1]} : vector<6x4xf32> to vector<2x2xf32>
// ORDER-NEXT:    %[[VGT3:.*]] = vector.gather {{.*}}[%[[C0]], %[[C0]], %[[C0]]] [%[[IDX3]]], %[[MASK3]], %[[PASS3]] : memref<?x?x?xf32>, vector<2x2xindex>, vector<2x2xi1>, vector<2x2xf32> into vector<2x2xf32>
// ORDER-NEXT:    %[[VEC3:.*]] = vector.insert_strided_slice %[[VGT3]], %[[VEC2]] {offsets = [0, 2], strides = [1, 1]} : vector<2x2xf32> into vector<6x4xf32>
// ORDER-NEXT:    %[[IDX4:.*]] = vector.extract_strided_slice %[[INDICES]] {offsets = [2, 2], sizes = [2, 2], strides = [1, 1]} : vector<6x4xindex> to vector<2x2xindex>
// ORDER-NEXT:    %[[MASK4:.*]] = vector.extract_strided_slice %[[MASK]] {offsets = [2, 2], sizes = [2, 2], strides = [1, 1]} : vector<6x4xi1> to vector<2x2xi1>
// ORDER-NEXT:    %[[PASS4:.*]] = vector.extract_strided_slice %[[PT]] {offsets = [2, 2], sizes = [2, 2], strides = [1, 1]} : vector<6x4xf32> to vector<2x2xf32>
// ORDER-NEXT:    %[[VGT4:.*]] = vector.gather {{.*}}[%[[C0]], %[[C0]], %[[C0]]] [%[[IDX4]]], %[[MASK4]], %[[PASS4]] : memref<?x?x?xf32>, vector<2x2xindex>, vector<2x2xi1>, vector<2x2xf32> into vector<2x2xf32>
// ORDER-NEXT:    %[[VEC4:.*]] = vector.insert_strided_slice %[[VGT4]], %[[VEC3]] {offsets = [2, 2], strides = [1, 1]} : vector<2x2xf32> into vector<6x4xf32>
// ORDER-NEXT:    %[[IDX5:.*]] = vector.extract_strided_slice %[[INDICES]] {offsets = [4, 2], sizes = [2, 2], strides = [1, 1]} : vector<6x4xindex> to vector<2x2xindex>
// ORDER-NEXT:    %[[MASK5:.*]] = vector.extract_strided_slice %[[MASK]] {offsets = [4, 2], sizes = [2, 2], strides = [1, 1]} : vector<6x4xi1> to vector<2x2xi1>
// ORDER-NEXT:    %[[PASS5:.*]] = vector.extract_strided_slice %[[PT]] {offsets = [4, 2], sizes = [2, 2], strides = [1, 1]} : vector<6x4xf32> to vector<2x2xf32>
// ORDER-NEXT:    %[[VGT5:.*]] = vector.gather {{.*}}[%[[C0]], %[[C0]], %[[C0]]] [%[[IDX5]]], %[[MASK5]], %[[PASS5]] : memref<?x?x?xf32>, vector<2x2xindex>, vector<2x2xi1>, vector<2x2xf32> into vector<2x2xf32>
// ORDER-NEXT:    %[[VEC5:.*]] = vector.insert_strided_slice %[[VGT5]], %[[VEC4]] {offsets = [4, 2], strides = [1, 1]} : vector<2x2xf32> into vector<6x4xf32>
// ORDER-NEXT:    return %[[VEC5]] : vector<6x4xf32>

func.func @vector_gather_unroll(%mem : memref<?x?x?xf32>,
                                %indices : vector<6x4xindex>,
                                %mask : vector<6x4xi1>,
                                %pass_thru : vector<6x4xf32>) -> vector<6x4xf32> {
  %c0 = arith.constant 0 : index
  %res = vector.gather %mem[%c0, %c0, %c0] [%indices], %mask, %pass_thru : memref<?x?x?xf32>, vector<6x4xindex>, vector<6x4xi1>, vector<6x4xf32> into vector<6x4xf32>
  return %res : vector<6x4xf32>
}
