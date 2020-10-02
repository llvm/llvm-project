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
