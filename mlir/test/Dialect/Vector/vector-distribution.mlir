// RUN: mlir-opt %s -test-vector-distribute-patterns=distribution-multiplicity=32,1,32 -split-input-file | FileCheck %s

// CHECK-LABEL: func @distribute_vector_add
//  CHECK-SAME: (%[[ID:.*]]: index
//  CHECK-NEXT:    %[[ADDV:.*]] = addf %{{.*}}, %{{.*}} : vector<32xf32>
//  CHECK-NEXT:    %[[EXA:.*]] = vector.extract_map %{{.*}}[%[[ID]]] : vector<32xf32> to vector<1xf32>
//  CHECK-NEXT:    %[[EXB:.*]] = vector.extract_map %{{.*}}[%[[ID]]] : vector<32xf32> to vector<1xf32>
//  CHECK-NEXT:    %[[ADD:.*]] = addf %[[EXA]], %[[EXB]] : vector<1xf32>
//  CHECK-NEXT:    %[[INS:.*]] = vector.insert_map %[[ADD]], %[[ADDV]][%[[ID]]] : vector<1xf32> into vector<32xf32>
//  CHECK-NEXT:    return %[[INS]] : vector<32xf32>
func @distribute_vector_add(%id : index, %A: vector<32xf32>, %B: vector<32xf32>) -> vector<32xf32> {
  %0 = addf %A, %B : vector<32xf32>
  return %0: vector<32xf32>
}

// -----

// CHECK-LABEL: func @vector_add_read_write
//  CHECK-SAME: (%[[ID:.*]]: index
//       CHECK:    %[[EXA:.*]] = vector.transfer_read %{{.*}}[%[[ID]]], %{{.*}} : memref<32xf32>, vector<1xf32>
//  CHECK-NEXT:    %[[EXB:.*]] = vector.transfer_read %{{.*}}[%[[ID]]], %{{.*}} : memref<32xf32>, vector<1xf32>
//  CHECK-NEXT:    %[[ADD1:.*]] = addf %[[EXA]], %[[EXB]] : vector<1xf32>
//  CHECK-NEXT:    %[[EXC:.*]] = vector.transfer_read %{{.*}}[%[[ID]]], %{{.*}} : memref<32xf32>, vector<1xf32>
//  CHECK-NEXT:    %[[ADD2:.*]] = addf %[[ADD1]], %[[EXC]] : vector<1xf32>
//  CHECK-NEXT:    vector.transfer_write %[[ADD2]], %{{.*}}[%[[ID]]] {{.*}} : vector<1xf32>, memref<32xf32>
//  CHECK-NEXT:    return
func @vector_add_read_write(%id : index, %A: memref<32xf32>, %B: memref<32xf32>, %C: memref<32xf32>, %D: memref<32xf32>) {
  %c0 = constant 0 : index
  %cf0 = constant 0.0 : f32
  %a = vector.transfer_read %A[%c0], %cf0: memref<32xf32>, vector<32xf32>
  %b = vector.transfer_read %B[%c0], %cf0: memref<32xf32>, vector<32xf32>
  %acc = addf %a, %b: vector<32xf32>
  %c = vector.transfer_read %C[%c0], %cf0: memref<32xf32>, vector<32xf32>
  %d = addf %acc, %c: vector<32xf32>
  vector.transfer_write %d, %D[%c0]: vector<32xf32>, memref<32xf32>
  return
}

// -----

// CHECK-DAG: #[[MAP0:.*]] = affine_map<()[s0] -> (s0 * 2)>

//       CHECK: func @vector_add_cycle
//  CHECK-SAME: (%[[ID:.*]]: index
//       CHECK:    %[[ID1:.*]] = affine.apply #[[MAP0]]()[%[[ID]]]
//  CHECK-NEXT:    %[[EXA:.*]] = vector.transfer_read %{{.*}}[%[[ID1]]], %{{.*}} : memref<64xf32>, vector<2xf32>
//  CHECK-NEXT:    %[[ID2:.*]] = affine.apply #[[MAP0]]()[%[[ID]]]
//  CHECK-NEXT:    %[[EXB:.*]] = vector.transfer_read %{{.*}}[%[[ID2]]], %{{.*}} : memref<64xf32>, vector<2xf32>
//  CHECK-NEXT:    %[[ADD:.*]] = addf %[[EXA]], %[[EXB]] : vector<2xf32>
//  CHECK-NEXT:    %[[ID3:.*]] = affine.apply #[[MAP0]]()[%[[ID]]]
//  CHECK-NEXT:    vector.transfer_write %[[ADD]], %{{.*}}[%[[ID3]]] {{.*}} : vector<2xf32>, memref<64xf32>
//  CHECK-NEXT:    return
func @vector_add_cycle(%id : index, %A: memref<64xf32>, %B: memref<64xf32>, %C: memref<64xf32>) {
  %c0 = constant 0 : index
  %cf0 = constant 0.0 : f32
  %a = vector.transfer_read %A[%c0], %cf0: memref<64xf32>, vector<64xf32>
  %b = vector.transfer_read %B[%c0], %cf0: memref<64xf32>, vector<64xf32>
  %acc = addf %a, %b: vector<64xf32>
  vector.transfer_write %acc, %C[%c0]: vector<64xf32>, memref<64xf32>
  return
}

// -----

// Negative test to make sure nothing is done in case the vector size is not a
// multiple of multiplicity.
// CHECK-LABEL: func @vector_negative_test
//       CHECK:    %[[C0:.*]] = constant 0 : index
//       CHECK:    %[[EXA:.*]] = vector.transfer_read %{{.*}}[%[[C0]]], %{{.*}} : memref<64xf32>, vector<16xf32>
//  CHECK-NEXT:    %[[EXB:.*]] = vector.transfer_read %{{.*}}[%[[C0]]], %{{.*}} : memref<64xf32>, vector<16xf32>
//  CHECK-NEXT:    %[[ADD:.*]] = addf %[[EXA]], %[[EXB]] : vector<16xf32>
//  CHECK-NEXT:    vector.transfer_write %[[ADD]], %{{.*}}[%[[C0]]] {{.*}} : vector<16xf32>, memref<64xf32>
//  CHECK-NEXT:    return
func @vector_negative_test(%id : index, %A: memref<64xf32>, %B: memref<64xf32>, %C: memref<64xf32>) {
  %c0 = constant 0 : index
  %cf0 = constant 0.0 : f32
  %a = vector.transfer_read %A[%c0], %cf0: memref<64xf32>, vector<16xf32>
  %b = vector.transfer_read %B[%c0], %cf0: memref<64xf32>, vector<16xf32>
  %acc = addf %a, %b: vector<16xf32>
  vector.transfer_write %acc, %C[%c0]: vector<16xf32>, memref<64xf32>
  return
}

// -----

// CHECK-LABEL: func @distribute_vector_add_3d
//  CHECK-SAME: (%[[ID0:.*]]: index, %[[ID1:.*]]: index
//  CHECK-NEXT:    %[[ADDV:.*]] = addf %{{.*}}, %{{.*}} : vector<64x4x32xf32>
//  CHECK-NEXT:    %[[EXA:.*]] = vector.extract_map %{{.*}}[%[[ID0]], %[[ID1]]] : vector<64x4x32xf32> to vector<2x4x1xf32>
//  CHECK-NEXT:    %[[EXB:.*]] = vector.extract_map %{{.*}}[%[[ID0]], %[[ID1]]] : vector<64x4x32xf32> to vector<2x4x1xf32>
//  CHECK-NEXT:    %[[ADD:.*]] = addf %[[EXA]], %[[EXB]] : vector<2x4x1xf32>
//  CHECK-NEXT:    %[[INS:.*]] = vector.insert_map %[[ADD]], %[[ADDV]][%[[ID0]], %[[ID1]]] : vector<2x4x1xf32> into vector<64x4x32xf32>
//  CHECK-NEXT:    return %[[INS]] : vector<64x4x32xf32>
func @distribute_vector_add_3d(%id0 : index, %id1 : index,
  %A: vector<64x4x32xf32>, %B: vector<64x4x32xf32>) -> vector<64x4x32xf32> {
  %0 = addf %A, %B : vector<64x4x32xf32>
  return %0: vector<64x4x32xf32>
}

// -----

// CHECK-DAG: #[[MAP0:.*]] = affine_map<()[s0] -> (s0 * 2)>

//       CHECK: func @vector_add_transfer_3d
//  CHECK-SAME: (%[[ID_0:.*]]: index, %[[ID_1:.*]]: index
//       CHECK:    %[[C0:.*]] = constant 0 : index
//       CHECK:    %[[ID1:.*]] = affine.apply #[[MAP0]]()[%[[ID_0]]]
//  CHECK-NEXT:    %[[EXA:.*]] = vector.transfer_read %{{.*}}[%[[ID1]], %[[C0]], %[[ID_1]]], %{{.*}} : memref<64x64x64xf32>, vector<2x4x1xf32>
//  CHECK-NEXT:    %[[ID2:.*]] = affine.apply #[[MAP0]]()[%[[ID_0]]]
//  CHECK-NEXT:    %[[EXB:.*]] = vector.transfer_read %{{.*}}[%[[ID2]], %[[C0]], %[[ID_1]]], %{{.*}} : memref<64x64x64xf32>, vector<2x4x1xf32>
//  CHECK-NEXT:    %[[ADD:.*]] = addf %[[EXA]], %[[EXB]] : vector<2x4x1xf32>
//  CHECK-NEXT:    %[[ID3:.*]] = affine.apply #[[MAP0]]()[%[[ID_0]]]
//  CHECK-NEXT:    vector.transfer_write %[[ADD]], %{{.*}}[%[[ID3]], %[[C0]], %[[ID_1]]] {{.*}} : vector<2x4x1xf32>, memref<64x64x64xf32>
//  CHECK-NEXT:    return
func @vector_add_transfer_3d(%id0 : index, %id1 : index, %A: memref<64x64x64xf32>,
  %B: memref<64x64x64xf32>, %C: memref<64x64x64xf32>) {
  %c0 = constant 0 : index
  %cf0 = constant 0.0 : f32
  %a = vector.transfer_read %A[%c0, %c0, %c0], %cf0: memref<64x64x64xf32>, vector<64x4x32xf32>
  %b = vector.transfer_read %B[%c0, %c0, %c0], %cf0: memref<64x64x64xf32>, vector<64x4x32xf32>
  %acc = addf %a, %b: vector<64x4x32xf32>
  vector.transfer_write %acc, %C[%c0, %c0, %c0]: vector<64x4x32xf32>, memref<64x64x64xf32>
  return
}


