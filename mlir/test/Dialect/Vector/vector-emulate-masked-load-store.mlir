// RUN: mlir-opt %s --test-vector-emulate-masked-load-store | FileCheck %s

// CHECK-LABEL:  @vector_maskedload
//  CHECK-SAME:  (%[[ARG0:.*]]: memref<4x5xf32>) -> vector<4xf32> {
//   CHECK-DAG:  %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<4xf32>
//   CHECK-DAG:  %[[C7:.*]] = arith.constant 7 : index
//   CHECK-DAG:  %[[C6:.*]] = arith.constant 6 : index
//   CHECK-DAG:  %[[C5:.*]] = arith.constant 5 : index
//   CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:  %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG:  %[[C4:.*]] = arith.constant 4 : index
//   CHECK-DAG:  %[[S0:.*]] = vector.create_mask %[[C1]] : vector<4xi1>
//       CHECK:  %[[S1:.*]] = vector.extract %[[S0]][0] : i1 from vector<4xi1>
//       CHECK:  %[[S2:.*]] = scf.if %[[S1]] -> (vector<4xf32>) {
//       CHECK:    %[[S9:.*]] = memref.load %[[ARG0]][%[[C0]], %[[C4]]] : memref<4x5xf32>
//       CHECK:    %[[S10:.*]] = vector.insert %[[S9]], %[[CST]] [0] : f32 into vector<4xf32>
//       CHECK:    scf.yield %[[S10]] : vector<4xf32>
//       CHECK:  } else {
//       CHECK:    scf.yield %[[CST]] : vector<4xf32>
//       CHECK:  }
//       CHECK:  %[[S3:.*]] = vector.extract %[[S0]][1] : i1 from vector<4xi1>
//       CHECK:  %[[S4:.*]] = scf.if %[[S3]] -> (vector<4xf32>) {
//       CHECK:    %[[S9:.*]] = memref.load %[[ARG0]][%[[C0]], %[[C5]]] : memref<4x5xf32>
//       CHECK:    %[[S10:.*]] = vector.insert %[[S9]], %[[S2]] [1] : f32 into vector<4xf32>
//       CHECK:    scf.yield %[[S10]] : vector<4xf32>
//       CHECK:  } else {
//       CHECK:    scf.yield %[[S2]] : vector<4xf32>
//       CHECK:  }
//       CHECK:  %[[S5:.*]] = vector.extract %[[S0]][2] : i1 from vector<4xi1>
//       CHECK:  %[[S6:.*]] = scf.if %[[S5]] -> (vector<4xf32>) {
//       CHECK:    %[[S9:.*]] = memref.load %[[ARG0]][%[[C0]], %[[C6]]] : memref<4x5xf32>
//       CHECK:    %[[S10:.*]] = vector.insert %[[S9]], %[[S4]] [2] : f32 into vector<4xf32>
//       CHECK:    scf.yield %[[S10]] : vector<4xf32>
//       CHECK:  } else {
//       CHECK:    scf.yield %[[S4]] : vector<4xf32>
//       CHECK:  }
//       CHECK:  %[[S7:.*]] = vector.extract %[[S0]][3] : i1 from vector<4xi1>
//       CHECK:  %[[S8:.*]] = scf.if %[[S7]] -> (vector<4xf32>) {
//       CHECK:    %[[S9:.*]] = memref.load %[[ARG0]][%[[C0]], %[[C7]]] : memref<4x5xf32>
//       CHECK:    %[[S10:.*]] = vector.insert %[[S9]], %[[S6]] [3] : f32 into vector<4xf32>
//       CHECK:    scf.yield %[[S10]] : vector<4xf32>
//       CHECK:  } else {
//       CHECK:    scf.yield %[[S6]] : vector<4xf32>
//       CHECK:  }
//       CHECK:  return %[[S8]] : vector<4xf32>
func.func @vector_maskedload(%arg0 : memref<4x5xf32>) -> vector<4xf32> {
  %idx_0 = arith.constant 0 : index
  %idx_1 = arith.constant 1 : index
  %idx_4 = arith.constant 4 : index
  %mask = vector.create_mask %idx_1 : vector<4xi1>
  %s = arith.constant 0.0 : f32
  %pass_thru = vector.broadcast %s : f32 to vector<4xf32>
  %0 = vector.maskedload %arg0[%idx_0, %idx_4], %mask, %pass_thru : memref<4x5xf32>, vector<4xi1>, vector<4xf32> into vector<4xf32>
  return %0: vector<4xf32>
}

// CHECK-LABEL:  @vector_maskedload_with_alignment
//       CHECK:       memref.load
//       CHECK-SAME:  {alignment = 8 : i64}
//       CHECK:       memref.load
//       CHECK-SAME:  {alignment = 8 : i64}
func.func @vector_maskedload_with_alignment(%arg0 : memref<4x5xf32>) -> vector<4xf32> {
  %idx_0 = arith.constant 0 : index
  %idx_1 = arith.constant 1 : index
  %idx_4 = arith.constant 4 : index
  %mask = vector.create_mask %idx_1 : vector<4xi1>
  %s = arith.constant 0.0 : f32
  %pass_thru = vector.broadcast %s : f32 to vector<4xf32>
  %0 = vector.maskedload %arg0[%idx_0, %idx_4], %mask, %pass_thru {alignment = 8}: memref<4x5xf32>, vector<4xi1>, vector<4xf32> into vector<4xf32>
  return %0: vector<4xf32>
}

// CHECK-LABEL:  @vector_maskedstore
//  CHECK-SAME:  (%[[ARG0:.*]]: memref<4x5xf32>, %[[ARG1:.*]]: vector<4xf32>) {
//   CHECK-DAG:  %[[C7:.*]] = arith.constant 7 : index
//   CHECK-DAG:  %[[C6:.*]] = arith.constant 6 : index
//   CHECK-DAG:  %[[C5:.*]] = arith.constant 5 : index
//   CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:  %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG:  %[[C4:.*]] = arith.constant 4 : index
//   CHECK-DAG:  %[[S0:.*]] = vector.create_mask %[[C1]] : vector<4xi1>
//       CHECK:  %[[S1:.*]] = vector.extract %[[S0]][0] : i1 from vector<4xi1>
//       CHECK:  scf.if %[[S1]] {
//       CHECK:    %[[S5:.*]] = vector.extract %[[ARG1]][0] : f32 from vector<4xf32>
//       CHECK:    memref.store %[[S5]], %[[ARG0]][%[[C0]], %[[C4]]] : memref<4x5xf32>
//       CHECK:  }
//       CHECK:  %[[S2:.*]] = vector.extract %[[S0]][1] : i1 from vector<4xi1>
//       CHECK:  scf.if %[[S2]] {
//       CHECK:    %[[S5:.*]] = vector.extract %[[ARG1]][1] : f32 from vector<4xf32>
//       CHECK:    memref.store %[[S5]], %[[ARG0]][%[[C0]], %[[C5]]] : memref<4x5xf32>
//       CHECK:  }
//       CHECK:  %[[S3:.*]] = vector.extract %[[S0]][2] : i1 from vector<4xi1>
//       CHECK:  scf.if %[[S3]] {
//       CHECK:    %[[S5:.*]] = vector.extract %[[ARG1]][2] : f32 from vector<4xf32>
//       CHECK:    memref.store %[[S5]], %[[ARG0]][%[[C0]], %[[C6]]] : memref<4x5xf32>
//       CHECK:  }
//       CHECK:  %[[S4:.*]] = vector.extract %[[S0]][3] : i1 from vector<4xi1>
//       CHECK:  scf.if %[[S4]] {
//       CHECK:    %[[S5:.*]] = vector.extract %[[ARG1]][3] : f32 from vector<4xf32>
//       CHECK:    memref.store %[[S5]], %[[ARG0]][%[[C0]], %[[C7]]] : memref<4x5xf32>
//       CHECK:  }
//       CHECK:  return
//       CHECK:}
func.func @vector_maskedstore(%arg0 : memref<4x5xf32>, %arg1 : vector<4xf32>) {
  %idx_0 = arith.constant 0 : index
  %idx_1 = arith.constant 1 : index
  %idx_4 = arith.constant 4 : index
  %mask = vector.create_mask %idx_1 : vector<4xi1>
  vector.maskedstore %arg0[%idx_0, %idx_4], %mask, %arg1 : memref<4x5xf32>, vector<4xi1>, vector<4xf32>
  return
}

// CHECK-LABEL:  @vector_maskedstore_with_alignment
//       CHECK:       memref.store
//       CHECK-SAME:  {alignment = 8 : i64}
//       CHECK:       memref.store
//       CHECK-SAME:  {alignment = 8 : i64}
func.func @vector_maskedstore_with_alignment(%arg0 : memref<4x5xf32>, %arg1 : vector<4xf32>) {
  %idx_0 = arith.constant 0 : index
  %idx_1 = arith.constant 1 : index
  %idx_4 = arith.constant 4 : index
  %mask = vector.create_mask %idx_1 : vector<4xi1>
  vector.maskedstore %arg0[%idx_0, %idx_4], %mask, %arg1 { alignment = 8 } : memref<4x5xf32>, vector<4xi1>, vector<4xf32>
  return
}
