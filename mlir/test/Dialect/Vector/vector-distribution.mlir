// RUN: mlir-opt %s -test-vector-distribute-patterns=distribution-multiplicity=32 | FileCheck %s

// CHECK-LABEL: func @distribute_vector_add
//  CHECK-SAME: (%[[ID:.*]]: index
//  CHECK-NEXT:    %[[EXA:.*]] = vector.extract_map %{{.*}}[%[[ID]] : 32] : vector<32xf32> to vector<1xf32>
//  CHECK-NEXT:    %[[EXB:.*]] = vector.extract_map %{{.*}}[%[[ID]] : 32] : vector<32xf32> to vector<1xf32>
//  CHECK-NEXT:    %[[ADD:.*]] = addf %[[EXA]], %[[EXB]] : vector<1xf32>
//  CHECK-NEXT:    %[[INS:.*]] = vector.insert_map %[[ADD]], %[[ID]], 32 : vector<1xf32> to vector<32xf32>
//  CHECK-NEXT:    return %[[INS]] : vector<32xf32>
func @distribute_vector_add(%id : index, %A: vector<32xf32>, %B: vector<32xf32>) -> vector<32xf32> {
  %0 = addf %A, %B : vector<32xf32>
  return %0: vector<32xf32>
}

// CHECK-LABEL: func @vector_add_read_write
//  CHECK-SAME: (%[[ID:.*]]: index
//       CHECK:    %[[EXA:.*]] = vector.transfer_read %{{.*}}[%[[ID]]], %{{.*}} : memref<32xf32>, vector<1xf32>
//  CHECK-NEXT:    %[[EXB:.*]] = vector.transfer_read %{{.*}}[%[[ID]]], %{{.*}} : memref<32xf32>, vector<1xf32>
//  CHECK-NEXT:    %[[ADD1:.*]] = addf %[[EXA]], %[[EXB]] : vector<1xf32>
//  CHECK-NEXT:    %[[EXC:.*]] = vector.transfer_read %{{.*}}[%[[ID]]], %{{.*}} : memref<32xf32>, vector<1xf32>
//  CHECK-NEXT:    %[[ADD2:.*]] = addf %[[ADD1]], %[[EXC]] : vector<1xf32>
//  CHECK-NEXT:    vector.transfer_write %[[ADD2]], %{{.*}}[%[[ID]]] : vector<1xf32>, memref<32xf32>
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

// CHECK-LABEL: func @vector_add_cycle
//  CHECK-SAME: (%[[ID:.*]]: index
//       CHECK:    %[[EXA:.*]] = vector.transfer_read %{{.*}}[%[[ID]]], %{{.*}} : memref<64xf32>, vector<2xf32>
//  CHECK-NEXT:    %[[EXB:.*]] = vector.transfer_read %{{.*}}[%[[ID]]], %{{.*}} : memref<64xf32>, vector<2xf32>
//  CHECK-NEXT:    %[[ADD:.*]] = addf %[[EXA]], %[[EXB]] : vector<2xf32>
//  CHECK-NEXT:    vector.transfer_write %[[ADD]], %{{.*}}[%[[ID]]] : vector<2xf32>, memref<64xf32>
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


