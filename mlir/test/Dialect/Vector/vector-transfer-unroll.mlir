// RUN: mlir-opt %s -test-vector-transfer-unrolling-patterns | FileCheck %s

// CHECK-LABEL: func @transfer_read_unroll
//       CHECK:   %[[C0:.*]] = constant 0 : index
//       CHECK:   %[[C2:.*]] = constant 2 : index
//       CHECK:   %[[VTR0:.*]] = vector.transfer_read {{.*}}[%[[C0]], %[[C0]]], %{{.*}} : memref<4x4xf32>, vector<2x2xf32>
//  CHECK-NEXT:   %[[VTR1:.*]] = vector.transfer_read {{.*}}[%[[C0]], %[[C2]]], %{{.*}} : memref<4x4xf32>, vector<2x2xf32>
//  CHECK-NEXT:   %[[VTR2:.*]] = vector.transfer_read {{.*}}[%[[C2]], %[[C0]]], %{{.*}} : memref<4x4xf32>, vector<2x2xf32>
//  CHECK-NEXT:   %[[VTR3:.*]] = vector.transfer_read {{.*}}[%[[C2]], %[[C2]]], %{{.*}} : memref<4x4xf32>, vector<2x2xf32>
//  CHECK-NEXT:   %[[TUPL:.*]] = vector.tuple %[[VTR0]], %[[VTR1]], %[[VTR2]], %[[VTR3]] : vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>
//  CHECK-NEXT:   %[[VEC:.*]] = vector.insert_slices %[[TUPL]], [2, 2], [1, 1] : tuple<vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>> into vector<4x4xf32>
//  CHECK-NEXT:   return %[[VEC]] : vector<4x4xf32>

func @transfer_read_unroll(%arg0 : memref<4x4xf32>) -> vector<4x4xf32> {
  %c0 = constant 0 : index
  %cf0 = constant 0.0 : f32
  %0 = vector.transfer_read %arg0[%c0, %c0], %cf0 : memref<4x4xf32>, vector<4x4xf32>
  return %0 : vector<4x4xf32>
}

// CHECK-LABEL: func @transfer_write_unroll
//       CHECK:   %[[C0:.*]] = constant 0 : index
//       CHECK:   %[[C2:.*]] = constant 2 : index
//       CHECK:   %[[TUPL:.*]] = vector.extract_slices {{.*}}, [2, 2], [1, 1] : vector<4x4xf32> into tuple<vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>>
//  CHECK-NEXT:   %[[T0:.*]] = vector.tuple_get %[[TUPL]], 0 : tuple<vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>>
//  CHECK-NEXT:   vector.transfer_write %[[T0]], {{.*}}[%[[C0]], %[[C0]]] {{.*}} : vector<2x2xf32>, memref<4x4xf32>
//  CHECK-NEXT:   %[[T1:.*]] = vector.tuple_get %[[TUPL]], 1 : tuple<vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>>
//  CHECK-NEXT:   vector.transfer_write %[[T1]], {{.*}}[%[[C0]], %[[C2]]] {{.*}} : vector<2x2xf32>, memref<4x4xf32>
//  CHECK-NEXT:   %[[T2:.*]] = vector.tuple_get %[[TUPL]], 2 : tuple<vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>>
//  CHECK-NEXT:   vector.transfer_write %[[T2]], {{.*}}[%[[C2]], %[[C0]]] {{.*}} : vector<2x2xf32>, memref<4x4xf32>
//  CHECK-NEXT:   %[[T3:.*]] = vector.tuple_get %[[TUPL]], 3 : tuple<vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>>
//  CHECK-NEXT:   vector.transfer_write %[[T3]], {{.*}}[%[[C2]], %[[C2]]] {{.*}} : vector<2x2xf32>, memref<4x4xf32>
//  CHECK-NEXT:   return

func @transfer_write_unroll(%arg0 : memref<4x4xf32>, %arg1 : vector<4x4xf32>) {
  %c0 = constant 0 : index
  vector.transfer_write %arg1, %arg0[%c0, %c0] : vector<4x4xf32>, memref<4x4xf32>
  return
}

// CHECK-LABEL: func @transfer_readwrite_unroll
//       CHECK:   %[[C0:.*]] = constant 0 : index
//       CHECK:   %[[C2:.*]] = constant 2 : index
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
