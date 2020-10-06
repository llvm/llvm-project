// RUN: mlir-opt %s -test-vector-distribute-patterns | FileCheck %s

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
//       CHECK:    %[[EXA:.*]] = vector.transfer_read %{{.*}}[%{{.*}}], %{{.*}} : memref<32xf32>, vector<1xf32>
//  CHECK-NEXT:    %[[EXB:.*]] = vector.transfer_read %{{.*}}[%{{.*}}], %{{.*}} : memref<32xf32>, vector<1xf32>
//  CHECK-NEXT:    %[[ADD1:.*]] = addf %[[EXA]], %[[EXB]] : vector<1xf32>
//  CHECK-NEXT:    %[[EXC:.*]] = vector.transfer_read %{{.*}}[%{{.*}}], %{{.*}} : memref<32xf32>, vector<1xf32>
//  CHECK-NEXT:    %[[ADD2:.*]] = addf %[[ADD1]], %[[EXC]] : vector<1xf32>
//  CHECK-NEXT:    vector.transfer_write %[[ADD2]], %{{.*}}[%{{.*}}] : vector<1xf32>, memref<32xf32>
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
