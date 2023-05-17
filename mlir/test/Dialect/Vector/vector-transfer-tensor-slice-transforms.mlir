// RUN: mlir-opt -split-input-file -test-vector-transfer-tensor-slice-patterns %s | FileCheck %s

// CHECK-LABEL: func @transfer_read_of_extract_slice(
//  CHECK-SAME:     %[[t:.*]]: tensor<?x?xf32>, %[[s1:.*]]: index, %[[s2:.*]]: index
//   CHECK-DAG:   %[[c4:.*]] = arith.constant 4 : index
//   CHECK-DAG:   %[[c8:.*]] = arith.constant 8 : index
//       CHECK:   %[[add:.*]] = arith.addi %[[s1]], %[[c4]]
//       CHECK:   %[[r:.*]] = vector.transfer_read %[[t]][%[[c8]], %[[add]]], %{{.*}} {in_bounds = [true, true]} : tensor<?x?xf32>, vector<5x6xf32>
//       CHECK:   return %[[r]]
func.func @transfer_read_of_extract_slice(%t : tensor<?x?xf32>, %s1 : index, %s2 : index) -> vector<5x6xf32> {
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %cst = arith.constant 0.0 : f32
  %0 = tensor.extract_slice %t[5, %s1] [10, %s2] [1, 1] : tensor<?x?xf32> to tensor<10x?xf32>
  %1 = vector.transfer_read %0[%c3, %c4], %cst {in_bounds = [true, true]} : tensor<10x?xf32>, vector<5x6xf32>
  return %1 : vector<5x6xf32>
}

// -----

// CHECK-LABEL: func @transfer_read_of_extract_slice(
//  CHECK-SAME:     %[[t:.*]]: tensor<?x?xf32>, %[[s1:.*]]: index, %[[s2:.*]]: index
//   CHECK-DAG:   %[[c4:.*]] = arith.constant 4 : index
//   CHECK-DAG:   %[[c8:.*]] = arith.constant 8 : index
//       CHECK:   %[[add:.*]] = arith.addi %[[s1]], %[[c4]]
//       CHECK:   %[[r:.*]] = vector.transfer_read %[[t]][%[[c8]], %[[add]]], %{{.*}} {in_bounds = [true]} : tensor<?x?xf32>, vector<6xf32>
//       CHECK:   return %[[r]]
func.func @transfer_read_of_extract_slice(%t : tensor<?x?xf32>, %s1 : index, %s2 : index) -> vector<6xf32> {
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %cst = arith.constant 0.0 : f32
  %0 = tensor.extract_slice %t[5, %s1] [10, %s2] [1, 1] : tensor<?x?xf32> to tensor<10x?xf32>
  %1 = vector.transfer_read %0[%c3, %c4], %cst {in_bounds = [true]} : tensor<10x?xf32>, vector<6xf32>
  return %1 : vector<6xf32>
}

// -----

// CHECK-LABEL: func @transfer_read_of_extract_slice_rank_reducing(
//  CHECK-SAME:     %[[t:.*]]: tensor<?x?x?xf32>, %[[s1:.*]]: index, %[[s2:.*]]: index
//   CHECK-DAG:   %[[c3:.*]] = arith.constant 3 : index
//   CHECK-DAG:   %[[c5:.*]] = arith.constant 5 : index
//   CHECK-DAG:   %[[c10:.*]] = arith.constant 10 : index
//       CHECK:   %[[add:.*]] = arith.addi %[[s1]], %[[c3]]
//       CHECK:   %[[r:.*]] = vector.transfer_read %[[t]][%[[c5]], %[[add]], %[[c10]]], %{{.*}} {in_bounds = [true, true]} : tensor<?x?x?xf32>, vector<5x6xf32>
//       CHECK:   return %[[r]]
func.func @transfer_read_of_extract_slice_rank_reducing(%t : tensor<?x?x?xf32>, %s1 : index, %s2 : index) -> vector<5x6xf32> {
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %cst = arith.constant 0.0 : f32
  %0 = tensor.extract_slice %t[5, %s1, 6] [1, %s2, 12] [1, 1, 1] : tensor<?x?x?xf32> to tensor<?x12xf32>
  %1 = vector.transfer_read %0[%c3, %c4], %cst {in_bounds = [true, true]} : tensor<?x12xf32>, vector<5x6xf32>
  return %1 : vector<5x6xf32>
}

// -----

// CHECK-LABEL: func @transfer_read_of_extract_slice_illegal_rank_reducing(
//       CHECK:   extract_slice
//       CHECK:   vector.transfer_read
func.func @transfer_read_of_extract_slice_illegal_rank_reducing(%t : tensor<?x?x?xf32>, %s1 : index, %s2 : index) -> vector<5x6xf32> {
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %cst = arith.constant 0.0 : f32
  %0 = tensor.extract_slice %t[5, %s1, 6] [%s2, 1, 12] [1, 1, 1] : tensor<?x?x?xf32> to tensor<?x12xf32>
  %1 = vector.transfer_read %0[%c3, %c4], %cst {in_bounds = [true, true]} : tensor<?x12xf32>, vector<5x6xf32>
  return %1 : vector<5x6xf32>
}

// -----

// CHECK-LABEL: func @insert_slice_of_transfer_write(
//  CHECK-SAME:     %[[t1:.*]]: tensor<?x12xf32>, %[[v:.*]]: vector<5x6xf32>, %[[s:.*]]: index
//       CHECK:   %[[c3:.*]] = arith.constant 3 : index
//       CHECK:   %[[r:.*]] = vector.transfer_write %[[v]], %[[t1]][%[[c3]], %[[s]]] {in_bounds = [true, true]} : vector<5x6xf32>, tensor<?x12xf32>
//       CHECK:   return %[[r]]
func.func @insert_slice_of_transfer_write(%t1 : tensor<?x12xf32>, %v : vector<5x6xf32>, %s : index, %t2 : tensor<5x6xf32>) -> tensor<?x12xf32> {
  %c0 = arith.constant 0 : index
  %0 = vector.transfer_write %v, %t2[%c0, %c0] {in_bounds = [true, true]} : vector<5x6xf32>, tensor<5x6xf32>
  %1 = tensor.insert_slice %0 into %t1[3, %s] [5, 6] [1, 1] : tensor<5x6xf32> into tensor<?x12xf32>
  return %1 : tensor<?x12xf32>
}

// -----

// CHECK-LABEL: func @insert_slice_of_transfer_write_illegal_rank_extending(
//       CHECK:   vector.transfer_write
//       CHECK:   insert_slice
func.func @insert_slice_of_transfer_write_illegal_rank_extending(%t1 : tensor<?x?x12xf32>, %v : vector<5x6xf32>, %s : index, %t2 : tensor<5x6xf32>) -> tensor<?x?x12xf32> {
  %c0 = arith.constant 0 : index
  %0 = vector.transfer_write %v, %t2[%c0, %c0] {in_bounds = [true, true]} : vector<5x6xf32>, tensor<5x6xf32>
  %1 = tensor.insert_slice %0 into %t1[4, 3, %s] [5, 1, 6] [1, 1, 1] : tensor<5x6xf32> into tensor<?x?x12xf32>
  return %1 : tensor<?x?x12xf32>
}

// -----

// CHECK-LABEL: func @insert_slice_of_transfer_write_rank_extending(
//  CHECK-SAME:     %[[t1:.*]]: tensor<?x?x12xf32>, %[[v:.*]]: vector<5x6xf32>, %[[s:.*]]: index
//   CHECK-DAG:   %[[c3:.*]] = arith.constant 3 : index
//   CHECK-DAG:   %[[c4:.*]] = arith.constant 4 : index
//       CHECK:   %[[r:.*]] = vector.transfer_write %[[v]], %[[t1]][%[[c4]], %[[c3]], %[[s]]] {in_bounds = [true, true]} : vector<5x6xf32>, tensor<?x?x12xf32>
//       CHECK:   return %[[r]]
func.func @insert_slice_of_transfer_write_rank_extending(%t1 : tensor<?x?x12xf32>, %v : vector<5x6xf32>, %s : index, %t2 : tensor<5x6xf32>) -> tensor<?x?x12xf32> {
  %c0 = arith.constant 0 : index
  %0 = vector.transfer_write %v, %t2[%c0, %c0] {in_bounds = [true, true]} : vector<5x6xf32>, tensor<5x6xf32>
  %1 = tensor.insert_slice %0 into %t1[4, 3, %s] [1, 5, 6] [1, 1, 1] : tensor<5x6xf32> into tensor<?x?x12xf32>
  return %1 : tensor<?x?x12xf32>
}
