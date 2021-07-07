// RUN: mlir-opt %s -test-vector-distribute-patterns=distribution-multiplicity=32,1,32 -split-input-file | FileCheck %s
// RUN: mlir-opt %s -test-vector-distribute-patterns=distribution-multiplicity=32,4 -split-input-file | FileCheck %s --check-prefix=CHECK2D

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

// CHECK-LABEL: func @distribute_vector_add_exp
//  CHECK-SAME: (%[[ID:.*]]: index
//  CHECK-NEXT:    %[[EXPV:.*]] = math.exp %{{.*}} : vector<32xf32>
//  CHECK-NEXT:    %[[ADDV:.*]] = addf %[[EXPV]], %{{.*}} : vector<32xf32>
//  CHECK-NEXT:    %[[EXA:.*]] = vector.extract_map %{{.*}}[%[[ID]]] : vector<32xf32> to vector<1xf32>
//  CHECK-NEXT:    %[[EXC:.*]] = math.exp %[[EXA]] : vector<1xf32>
//  CHECK-NEXT:    %[[EXB:.*]] = vector.extract_map %{{.*}}[%[[ID]]] : vector<32xf32> to vector<1xf32>
//  CHECK-NEXT:    %[[ADD:.*]] = addf %[[EXC]], %[[EXB]] : vector<1xf32>
//  CHECK-NEXT:    %[[INS:.*]] = vector.insert_map %[[ADD]], %[[ADDV]][%[[ID]]] : vector<1xf32> into vector<32xf32>
//  CHECK-NEXT:    return %[[INS]] : vector<32xf32>
func @distribute_vector_add_exp(%id : index, %A: vector<32xf32>, %B: vector<32xf32>) -> vector<32xf32> {
  %C = math.exp %A : vector<32xf32>
  %0 = addf %C, %B : vector<32xf32>
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

// -----

#map0 = affine_map<(d0, d1, d2, d3) -> (d3, 0, 0)>
#map1 = affine_map<(d0, d1, d2, d3) -> (0, d3, d0)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d3, d2, d1)>

// CHECK-DAG: #[[MAP0:.*]] = affine_map<()[s0] -> (s0 * 2)>
// CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d3, 0, 0)>
// CHECK-DAG: #[[MAP2:.*]] = affine_map<(d0, d1, d2, d3) -> (0, d3, d0)>
// CHECK-DAG: #[[MAP3:.*]] = affine_map<(d0, d1, d2, d3) -> (d3, d2, d1)>

//       CHECK: func @vector_add_transfer_permutation
//  CHECK-SAME: (%[[ID_0:.*]]: index, %[[ID_1:.*]]: index
//       CHECK:    %[[C0:.*]] = constant 0 : index
//       CHECK:    %[[ID2:.*]] = affine.apply #[[MAP0]]()[%[[ID_0]]]
//  CHECK-NEXT:    %[[EXA:.*]] = vector.transfer_read %{{.*}}[%[[C0]], %[[C0]], %[[C0]], %[[ID2]]], %{{.*}} {permutation_map = #[[MAP1]]} : memref<?x?x?x?xf32>, vector<2x4x1xf32>
//  CHECK-NEXT:    %[[EXB:.*]] = vector.transfer_read %{{.*}}[%[[ID_0]], %[[C0]], %[[C0]], %[[C0]]], %{{.*}} {permutation_map = #[[MAP2]]} : memref<?x?x?x?xf32>, vector<2x4x1xf32>
//  CHECK-NEXT:    %[[ADD:.*]] = addf %[[EXA]], %[[EXB]] : vector<2x4x1xf32>
//  CHECK-NEXT:    %[[ID3:.*]] = affine.apply #[[MAP0]]()[%[[ID_0]]]
//  CHECK-NEXT:    vector.transfer_write %[[ADD]], %{{.*}}[%[[C0]], %[[ID_1]], %[[C0]], %[[ID3]]] {permutation_map = #[[MAP3]]} : vector<2x4x1xf32>, memref<?x?x?x?xf32>
//  CHECK-NEXT:    return
func @vector_add_transfer_permutation(%id0 : index, %id1 : index, %A: memref<?x?x?x?xf32>,
  %B: memref<?x?x?x?xf32>, %C: memref<?x?x?x?xf32>) {
  %c0 = constant 0 : index
  %cf0 = constant 0.0 : f32
  %a = vector.transfer_read %A[%c0, %c0, %c0, %c0], %cf0 {permutation_map = #map0} : memref<?x?x?x?xf32>, vector<64x4x32xf32>
  %b = vector.transfer_read %B[%c0, %c0, %c0, %c0], %cf0 {permutation_map = #map1}: memref<?x?x?x?xf32>, vector<64x4x32xf32>
  %acc = addf %a, %b: vector<64x4x32xf32>
  vector.transfer_write %acc, %C[%c0, %c0, %c0, %c0] {permutation_map = #map2}: vector<64x4x32xf32>, memref<?x?x?x?xf32>
  return
}

// -----

// CHECK2D-LABEL: vector_add_contract
//       CHECK2D:   %[[A:.+]] = vector.transfer_read %arg2[%0, %c0], %cst : memref<?x?xf32>, vector<2x4xf32>
//       CHECK2D:   %[[B:.+]] = vector.transfer_read %arg3[%2, %c0], %cst : memref<?x?xf32>, vector<16x4xf32>
//       CHECK2D:   %[[C:.+]] = vector.transfer_read %arg4[%4, %5], %cst : memref<?x?xf32>, vector<2x16xf32>
//       CHECK2D:   %[[E:.+]] = vector.transfer_read %arg5[%7, %8], %cst : memref<?x?xf32>, vector<2x16xf32>
//       CHECK2D:   %[[D:.+]] = vector.contract {{.*}} %[[A]], %[[B]], %[[C]] : vector<2x4xf32>, vector<16x4xf32> into vector<2x16xf32>
//       CHECK2D:   %[[R:.+]] = addf %[[D]], %[[E]] : vector<2x16xf32>
//       CHECK2D:   vector.transfer_write %[[R]], {{.*}} : vector<2x16xf32>, memref<?x?xf32>
func @vector_add_contract(%id0 : index, %id1 : index, %A: memref<?x?xf32>,
  %B: memref<?x?xf32>, %C: memref<?x?xf32>, %D: memref<?x?xf32>) {
  %c0 = constant 0 : index
  %cf0 = constant 0.0 : f32
  %a = vector.transfer_read %A[%c0, %c0], %cf0 : memref<?x?xf32>, vector<64x4xf32>
  %b = vector.transfer_read %B[%c0, %c0], %cf0 : memref<?x?xf32>, vector<64x4xf32>
  %c = vector.transfer_read %C[%c0, %c0], %cf0 : memref<?x?xf32>, vector<64x64xf32>
  %d = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                                         affine_map<(d0, d1, d2) -> (d1, d2)>,
                                         affine_map<(d0, d1, d2) -> (d0, d1)>],
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>}
    %a, %b, %c : vector<64x4xf32>, vector<64x4xf32> into vector<64x64xf32>
  %e = vector.transfer_read %D[%c0, %c0], %cf0 : memref<?x?xf32>, vector<64x64xf32>
  %r = addf %d, %e : vector<64x64xf32>
  vector.transfer_write %r, %C[%c0, %c0] : vector<64x64xf32>, memref<?x?xf32>
  return
}
