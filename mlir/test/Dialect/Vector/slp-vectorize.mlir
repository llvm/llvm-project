// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(greedy-slp-vectorizer{max-vector-bitwidth=256}))' | FileCheck %s


// CHECK-LABEL: func @negative_single_op
func.func @negative_single_op(%arg0: memref<8xi32>, %arg1: memref<8xi32>) {
  // CHECK-NOT: vector
  %c0 = arith.constant 0 : index

  %0 = memref.load %arg0[%c0] : memref<8xi32>
  %4 = memref.load %arg1[%c0] : memref<8xi32>
  %8 = arith.addi %0, %4 : i32
  memref.store %8, %arg0[%c0] : memref<8xi32>

  return
}


// CHECK-LABEL: func @read_write
//  CHECK-SAME: (%[[ARG0:.*]]: memref<8xi32>, %[[ARG1:.*]]: memref<8xi32>)
func.func @read_write(%arg0: memref<8xi32>, %arg1: memref<8xi32>) {
  // CHECK:     %[[C0:.*]] = arith.constant 0 : index
  // CHECK:     %[[RES:.*]] = vector.load %[[ARG0]][%[[C0]]] : memref<8xi32>, vector<4xi32>
  // CHECK:     vector.store %[[RES]], %[[ARG0]][%[[C0]]] : memref<8xi32>, vector<4xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index

  %0 = memref.load %arg0[%c0] : memref<8xi32>
  %1 = memref.load %arg0[%c1] : memref<8xi32>
  %2 = memref.load %arg0[%c2] : memref<8xi32>
  %3 = memref.load %arg0[%c3] : memref<8xi32>

  memref.store %0, %arg0[%c0] : memref<8xi32>
  memref.store %1, %arg0[%c1] : memref<8xi32>
  memref.store %2, %arg0[%c2] : memref<8xi32>
  memref.store %3, %arg0[%c3] : memref<8xi32>

  return
}


// CHECK-LABEL: func @read_write_size_mistamtch
//  CHECK-SAME: (%[[ARG0:.*]]: memref<8xi32>, %[[ARG1:.*]]: memref<8xi32>)
func.func @read_write_size_mistamtch(%arg0: memref<8xi32>, %arg1: memref<8xi32>) {
  // CHECK:     %[[C0:.*]] = arith.constant 0 : index
  // CHECK:     %[[RES:.*]] = vector.load %[[ARG0]][%[[C0]]] : memref<8xi32>, vector<4xi32>
  // CHECK:     %[[RES1:.*]] = vector.extract_strided_slice %[[RES]] {offsets = [0], sizes = [2], strides = [1]} : vector<4xi32> to vector<2xi32>
  // CHECK:     vector.store %[[RES1]], %[[ARG0]][%[[C0]]] : memref<8xi32>, vector<2xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index

  %0 = memref.load %arg0[%c0] : memref<8xi32>
  %1 = memref.load %arg0[%c1] : memref<8xi32>
  %2 = memref.load %arg0[%c2] : memref<8xi32>
  %3 = memref.load %arg0[%c3] : memref<8xi32>

  memref.store %0, %arg0[%c0] : memref<8xi32>
  memref.store %1, %arg0[%c1] : memref<8xi32>

  return
}


// CHECK-LABEL: func @read_write_interleaved
//  CHECK-SAME: (%[[ARG0:.*]]: memref<8xi32>, %[[ARG1:.*]]: memref<8xi32>)
func.func @read_write_interleaved(%arg0: memref<8xi32>, %arg1: memref<8xi32>) {
  // CHECK:     %[[C0:.*]] = arith.constant 0 : index
  // CHECK:     %[[RES:.*]] = vector.load %[[ARG0]][%[[C0]]] : memref<8xi32>, vector<4xi32>
  // CHECK:     vector.store %[[RES]], %[[ARG0]][%[[C0]]] : memref<8xi32>, vector<4xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index

  %3 = memref.load %arg0[%c3] : memref<8xi32>
  %0 = memref.load %arg0[%c0] : memref<8xi32>
  %2 = memref.load %arg0[%c2] : memref<8xi32>
  %1 = memref.load %arg0[%c1] : memref<8xi32>

  memref.store %1, %arg0[%c1] : memref<8xi32>
  memref.store %0, %arg0[%c0] : memref<8xi32>
  memref.store %3, %arg0[%c3] : memref<8xi32>
  memref.store %2, %arg0[%c2] : memref<8xi32>

  return
}


// CHECK-LABEL: func @read_write_add_index
//  CHECK-SAME: (%[[ARG0:.*]]: memref<8xi32>, %[[ARG1:.*]]: memref<8xi32>, %[[ARG2:.*]]: index)
func.func @read_write_add_index(%arg0: memref<8xi32>, %arg1: memref<8xi32>, %arg2: index) {
  // CHECK:     %[[RES:.*]] = vector.load %[[ARG0]][%[[ARG2]]] : memref<8xi32>, vector<4xi32>
  // CHECK:     vector.store %[[RES]], %[[ARG0]][%[[ARG2]]] : memref<8xi32>, vector<4xi32>
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index

  %ind1 = arith.addi %arg2, %c1 : index
  %ind2 = arith.addi %arg2, %c2 : index
  %ind3 = arith.addi %arg2, %c3 : index

  %0 = memref.load %arg0[%arg2] : memref<8xi32>
  %1 = memref.load %arg0[%ind1] : memref<8xi32>
  %2 = memref.load %arg0[%ind2] : memref<8xi32>
  %3 = memref.load %arg0[%ind3] : memref<8xi32>

  memref.store %0, %arg0[%arg2] : memref<8xi32>
  memref.store %1, %arg0[%ind1] : memref<8xi32>
  memref.store %2, %arg0[%ind2] : memref<8xi32>
  memref.store %3, %arg0[%ind3] : memref<8xi32>

  return
}


// CHECK-LABEL: func @read_write_add_index_interleaved
//  CHECK-SAME: (%[[ARG0:.*]]: memref<8xi32>, %[[ARG1:.*]]: memref<8xi32>, %[[ARG2:.*]]: index)
func.func @read_write_add_index_interleaved(%arg0: memref<8xi32>, %arg1: memref<8xi32>, %arg2: index) {
  // CHECK:     %[[RES:.*]] = vector.load %[[ARG0]][%[[ARG2]]] : memref<8xi32>, vector<4xi32>
  // CHECK:     vector.store %[[RES]], %[[ARG0]][%[[ARG2]]] : memref<8xi32>, vector<4xi32>
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index

  %ind1 = arith.addi %arg2, %c1 : index
  %ind2 = arith.addi %arg2, %c2 : index
  %ind3 = arith.addi %arg2, %c3 : index

  %0 = memref.load %arg0[%arg2] : memref<8xi32>
  %1 = memref.load %arg0[%ind1] : memref<8xi32>
  %3 = memref.load %arg0[%ind3] : memref<8xi32>
  %2 = memref.load %arg0[%ind2] : memref<8xi32>

  memref.store %3, %arg0[%ind3] : memref<8xi32>
  memref.store %0, %arg0[%arg2] : memref<8xi32>
  memref.store %1, %arg0[%ind1] : memref<8xi32>
  memref.store %2, %arg0[%ind2] : memref<8xi32>

  return
}


#map0 = affine_map<()[s0, s1] -> (s1 * s0)>
#map1 = affine_map<()[s0, s1] -> (s1 * s0 + 1)>
#map2 = affine_map<()[s0, s1] -> (s1 * s0 + 2)>
#map3 = affine_map<()[s0, s1] -> (s1 * s0 + 3)>

// CHECK-LABEL: func @read_write_affine_apply
//  CHECK-SAME: (%[[ARG0:.*]]: memref<8xi32>, %[[ARG1:.*]]: memref<8xi32>, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index)
func.func @read_write_affine_apply(%arg0: memref<8xi32>, %arg1: memref<8xi32>, %arg2: index, %arg3: index) {
  // CHECK:     %[[IDX:.*]] = affine.apply #{{.*}}()[%[[ARG2]], %[[ARG3]]]
  // CHECK:     %[[RES:.*]] = vector.load %[[ARG0]][%[[IDX]]] : memref<8xi32>, vector<4xi32>
  // CHECK:     vector.store %[[RES]], %[[ARG0]][%[[IDX]]] : memref<8xi32>, vector<4xi32>

  %ind0 = affine.apply #map0()[%arg2, %arg3]
  %ind1 = affine.apply #map1()[%arg2, %arg3]
  %ind2 = affine.apply #map2()[%arg2, %arg3]
  %ind3 = affine.apply #map3()[%arg2, %arg3]

  %0 = memref.load %arg0[%ind0] : memref<8xi32>
  %1 = memref.load %arg0[%ind1] : memref<8xi32>
  %2 = memref.load %arg0[%ind2] : memref<8xi32>
  %3 = memref.load %arg0[%ind3] : memref<8xi32>

  memref.store %0, %arg0[%ind0] : memref<8xi32>
  memref.store %1, %arg0[%ind1] : memref<8xi32>
  memref.store %2, %arg0[%ind2] : memref<8xi32>
  memref.store %3, %arg0[%ind3] : memref<8xi32>

  return
}


// CHECK-LABEL: func @read_read_add
//  CHECK-SAME: (%[[ARG0:.*]]: memref<8xi32>, %[[ARG1:.*]]: memref<8xi32>)
func.func @read_read_add(%arg0: memref<8xi32>, %arg1: memref<8xi32>) -> (i32, i32, i32, i32) {
  // CHECK:     %[[C0:.*]] = arith.constant 0 : index
  // CHECK:     %[[A:.*]] = vector.load %[[ARG0]][%[[C0]]] : memref<8xi32>, vector<4xi32>
  // CHECK:     %[[B:.*]] = vector.load %[[ARG1]][%[[C0]]] : memref<8xi32>, vector<4xi32>
  // CHECK:     %[[RES:.*]] = arith.addi %[[A]], %[[B]] : vector<4xi32>
  // CHECK:     %[[R0:.*]] = vector.extract %[[RES]][0] : i32 from vector<4xi32>
  // CHECK:     %[[R1:.*]] = vector.extract %[[RES]][1] : i32 from vector<4xi32>
  // CHECK:     %[[R2:.*]] = vector.extract %[[RES]][2] : i32 from vector<4xi32>
  // CHECK:     %[[R3:.*]] = vector.extract %[[RES]][3] : i32 from vector<4xi32>
  // CHECK:     return %[[R0]], %[[R1]], %[[R2]], %[[R3]] : i32, i32, i32, i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index

  %0 = memref.load %arg0[%c0] : memref<8xi32>
  %1 = memref.load %arg0[%c1] : memref<8xi32>
  %2 = memref.load %arg0[%c2] : memref<8xi32>
  %3 = memref.load %arg0[%c3] : memref<8xi32>

  %4 = memref.load %arg1[%c0] : memref<8xi32>
  %5 = memref.load %arg1[%c1] : memref<8xi32>
  %6 = memref.load %arg1[%c2] : memref<8xi32>
  %7 = memref.load %arg1[%c3] : memref<8xi32>

  %8 = arith.addi %0, %4 : i32
  %9 = arith.addi %1, %5 : i32
  %10 = arith.addi %2, %6 : i32
  %11 = arith.addi %3, %7 : i32

  return %8, %9, %10, %11 : i32, i32, i32, i32
}


// CHECK-LABEL: func @add_write
//  CHECK-SAME: (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32, %[[ARG2:.*]]: i32, %[[ARG3:.*]]: i32, %[[ARG4:.*]]: i32, %[[ARG5:.*]]: i32, %[[ARG6:.*]]: i32, %[[ARG7:.*]]: i32, %[[ARG8:.*]]: memref<8xi32>)
func.func @add_write(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32,
                     %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32,
                     %arg8: memref<8xi32>) {
  // CHECK:     %[[C0:.*]] = arith.constant 0 : index
  // CHECK:     %[[A:.*]] = vector.from_elements %[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]] : vector<4xi32>
  // CHECK:     %[[B:.*]] = vector.from_elements %[[ARG4]], %[[ARG5]], %[[ARG6]], %[[ARG7]] : vector<4xi32>
  // CHECK:     %[[RES:.*]] = arith.addi %0, %1 : vector<4xi32>
  // CHECK:     vector.store %[[RES]], %[[ARG8]][%[[C0]]] : memref<8xi32>, vector<4xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index

  %8 = arith.addi %arg0, %arg4 : i32
  %9 = arith.addi %arg1, %arg5 : i32
  %10 = arith.addi %arg2, %arg6 : i32
  %11 = arith.addi %arg3, %arg7 : i32

  memref.store %8, %arg8[%c0] : memref<8xi32>
  memref.store %9, %arg8[%c1] : memref<8xi32>
  memref.store %10, %arg8[%c2] : memref<8xi32>
  memref.store %11, %arg8[%c3] : memref<8xi32>

  return
}


// CHECK-LABEL: func @read_read_add_write
//  CHECK-SAME: (%[[ARG0:.*]]: memref<8xi32>, %[[ARG1:.*]]: memref<8xi32>)
func.func @read_read_add_write(%arg0: memref<8xi32>, %arg1: memref<8xi32>) {
  // CHECK:     %[[C0:.*]] = arith.constant 0 : index
  // CHECK:     %[[A:.*]] = vector.load %[[ARG0]][%[[C0]]] : memref<8xi32>, vector<4xi32>
  // CHECK:     %[[B:.*]] = vector.load %[[ARG1]][%[[C0]]] : memref<8xi32>, vector<4xi32>
  // CHECK:     %[[RES:.*]] = arith.addi %[[A]], %[[B]] : vector<4xi32>
  // CHECK:     vector.store %[[RES]], %[[ARG0]][%[[C0]]] : memref<8xi32>, vector<4xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index

  %0 = memref.load %arg0[%c0] : memref<8xi32>
  %1 = memref.load %arg0[%c1] : memref<8xi32>
  %2 = memref.load %arg0[%c2] : memref<8xi32>
  %3 = memref.load %arg0[%c3] : memref<8xi32>

  %4 = memref.load %arg1[%c0] : memref<8xi32>
  %5 = memref.load %arg1[%c1] : memref<8xi32>
  %6 = memref.load %arg1[%c2] : memref<8xi32>
  %7 = memref.load %arg1[%c3] : memref<8xi32>

  %8 = arith.addi %0, %4 : i32
  %9 = arith.addi %1, %5 : i32
  %10 = arith.addi %2, %6 : i32
  %11 = arith.addi %3, %7 : i32

  memref.store %8, %arg0[%c0] : memref<8xi32>
  memref.store %9, %arg0[%c1] : memref<8xi32>
  memref.store %10, %arg0[%c2] : memref<8xi32>
  memref.store %11, %arg0[%c3] : memref<8xi32>

  return
}


// CHECK-LABEL: func @read_read_add_write_vec_0d
//  CHECK-SAME: (%[[ARG0:.*]]: memref<8xi32>, %[[ARG1:.*]]: memref<8xi32>)
func.func @read_read_add_write_vec_0d(%arg0: memref<8xi32>, %arg1: memref<8xi32>) {
  // CHECK:     %[[C0:.*]] = arith.constant 0 : index
  // CHECK:     %[[A:.*]] = vector.load %[[ARG0]][%[[C0]]] : memref<8xi32>, vector<4xi32>
  // CHECK:     %[[B:.*]] = vector.load %[[ARG1]][%[[C0]]] : memref<8xi32>, vector<4xi32>
  // CHECK:     %[[RES:.*]] = arith.addi %[[A]], %[[B]] : vector<4xi32>
  // CHECK:     vector.store %[[RES]], %[[ARG0]][%[[C0]]] : memref<8xi32>, vector<4xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index

  %0 = vector.load %arg0[%c0] : memref<8xi32>, vector<i32>
  %1 = vector.load %arg0[%c1] : memref<8xi32>, vector<i32>
  %2 = vector.load %arg0[%c2] : memref<8xi32>, vector<i32>
  %3 = vector.load %arg0[%c3] : memref<8xi32>, vector<i32>

  %4 = vector.load %arg1[%c0] : memref<8xi32>, vector<i32>
  %5 = vector.load %arg1[%c1] : memref<8xi32>, vector<i32>
  %6 = vector.load %arg1[%c2] : memref<8xi32>, vector<i32>
  %7 = vector.load %arg1[%c3] : memref<8xi32>, vector<i32>

  %8 = arith.addi %0, %4 : vector<i32>
  %9 = arith.addi %1, %5 : vector<i32>
  %10 = arith.addi %2, %6 : vector<i32>
  %11 = arith.addi %3, %7 : vector<i32>

  vector.store %8, %arg0[%c0] : memref<8xi32>, vector<i32>
  vector.store %9, %arg0[%c1] : memref<8xi32>, vector<i32>
  vector.store %10, %arg0[%c2] : memref<8xi32>, vector<i32>
  vector.store %11, %arg0[%c3] : memref<8xi32>, vector<i32>

  return
}


// CHECK-LABEL: func @read_read_add_write_vec_1d
//  CHECK-SAME: (%[[ARG0:.*]]: memref<8xi32>, %[[ARG1:.*]]: memref<8xi32>)
func.func @read_read_add_write_vec_1d(%arg0: memref<8xi32>, %arg1: memref<8xi32>) {
  // CHECK:     %[[C0:.*]] = arith.constant 0 : index
  // CHECK:     %[[A:.*]] = vector.load %[[ARG0]][%[[C0]]] : memref<8xi32>, vector<4xi32>
  // CHECK:     %[[B:.*]] = vector.load %[[ARG1]][%[[C0]]] : memref<8xi32>, vector<4xi32>
  // CHECK:     %[[RES:.*]] = arith.addi %[[A]], %[[B]] : vector<4xi32>
  // CHECK:     vector.store %[[RES]], %[[ARG0]][%[[C0]]] : memref<8xi32>, vector<4xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index

  %0 = vector.load %arg0[%c0] : memref<8xi32>, vector<1xi32>
  %1 = vector.load %arg0[%c1] : memref<8xi32>, vector<1xi32>
  %2 = vector.load %arg0[%c2] : memref<8xi32>, vector<1xi32>
  %3 = vector.load %arg0[%c3] : memref<8xi32>, vector<1xi32>

  %4 = vector.load %arg1[%c0] : memref<8xi32>, vector<1xi32>
  %5 = vector.load %arg1[%c1] : memref<8xi32>, vector<1xi32>
  %6 = vector.load %arg1[%c2] : memref<8xi32>, vector<1xi32>
  %7 = vector.load %arg1[%c3] : memref<8xi32>, vector<1xi32>

  %8 = arith.addi %0, %4 : vector<1xi32>
  %9 = arith.addi %1, %5 : vector<1xi32>
  %10 = arith.addi %2, %6 : vector<1xi32>
  %11 = arith.addi %3, %7 : vector<1xi32>

  vector.store %8, %arg0[%c0] : memref<8xi32>, vector<1xi32>
  vector.store %9, %arg0[%c1] : memref<8xi32>, vector<1xi32>
  vector.store %10, %arg0[%c2] : memref<8xi32>, vector<1xi32>
  vector.store %11, %arg0[%c3] : memref<8xi32>, vector<1xi32>

  return
}


// CHECK-LABEL: func @read_read_add_write_mixed_vecs
//  CHECK-SAME: (%[[ARG0:.*]]: memref<8xi32>, %[[ARG1:.*]]: memref<8xi32>)
func.func @read_read_add_write_mixed_vecs(%arg0: memref<8xi32>, %arg1: memref<8xi32>) {
  // CHECK:     %[[C0:.*]] = arith.constant 0 : index
  // CHECK:     %[[A:.*]] = vector.load %[[ARG0]][%[[C0]]] : memref<8xi32>, vector<4xi32>
  // CHECK:     %[[B:.*]] = vector.load %[[ARG1]][%[[C0]]] : memref<8xi32>, vector<4xi32>
  // CHECK:     %[[RES:.*]] = arith.addi %[[A]], %[[B]] : vector<4xi32>
  // CHECK:     vector.store %[[RES]], %[[ARG0]][%[[C0]]] : memref<8xi32>, vector<4xi32>
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index

  %0 = vector.load %arg0[%c0] : memref<8xi32>, vector<2xi32>
  %2 = memref.load %arg0[%c2] : memref<8xi32>
  %3 = vector.load %arg0[%c3] : memref<8xi32>, vector<1xi32>

  %4 = vector.load %arg1[%c0] : memref<8xi32>, vector<2xi32>
  %6 = memref.load %arg1[%c2] : memref<8xi32>
  %7 = vector.load %arg1[%c3] : memref<8xi32>, vector<1xi32>

  %8 = arith.addi %0, %4 : vector<2xi32>
  %10 = arith.addi %2, %6 : i32
  %11 = arith.addi %3, %7 : vector<1xi32>

  vector.store %8, %arg0[%c0] : memref<8xi32>, vector<2xi32>
  memref.store %10, %arg0[%c2] : memref<8xi32>
  vector.store %11, %arg0[%c3] : memref<8xi32>, vector<1xi32>

  return
}


// CHECK-LABEL: func @read_read_add_write_seven
//  CHECK-SAME: (%[[ARG0:.*]]: memref<8xindex>, %[[ARG1:.*]]: memref<8xindex>)
func.func @read_read_add_write_seven(%arg0: memref<8xindex>, %arg1: memref<8xindex>) {
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
  // CHECK-DAG: %[[C6:.*]] = arith.constant 6 : index
  // CHECK:     %[[A0:.*]] = vector.load %[[ARG0]][%[[C0]]] : memref<8xindex>, vector<4xindex>
  // CHECK:     %[[A1:.*]] = vector.load %[[ARG0]][%[[C4]]] : memref<8xindex>, vector<2xindex>
  // CHECK:     %[[A2:.*]] = memref.load %[[ARG0]][%[[C6]]] : memref<8xindex>
  // CHECK:     %[[B0:.*]] = vector.load %[[ARG1]][%[[C0]]] : memref<8xindex>, vector<4xindex>
  // CHECK:     %[[B1:.*]] = vector.load %[[ARG1]][%[[C4]]] : memref<8xindex>, vector<2xindex>
  // CHECK:     %[[B2:.*]] = memref.load %[[ARG1]][%[[C6]]] : memref<8xindex>
  // CHECK:     %[[RES0:.*]] = arith.addi %[[A0]], %[[B0]] : vector<4xindex>
  // CHECK:     %[[RES1:.*]] = arith.addi %[[A1]], %[[B1]] : vector<2xindex>
  // CHECK:     %[[RES2:.*]] = arith.addi %[[A2]], %[[B2]] : index
  // CHECK:     vector.store %[[RES0]], %[[ARG0]][%[[C0]]] : memref<8xindex>, vector<4xindex>
  // CHECK:     vector.store %[[RES1]], %[[ARG0]][%[[C4]]] : memref<8xindex>, vector<2xindex>
  // CHECK:     memref.store %[[RES2]], %[[ARG0]][%[[C6]]] : memref<8xindex>

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index

  %0 = memref.load %arg0[%c0] : memref<8xindex>
  %1 = memref.load %arg0[%c1] : memref<8xindex>
  %2 = memref.load %arg0[%c2] : memref<8xindex>
  %3 = memref.load %arg0[%c3] : memref<8xindex>
  %4 = memref.load %arg0[%c4] : memref<8xindex>
  %5 = memref.load %arg0[%c5] : memref<8xindex>
  %6 = memref.load %arg0[%c6] : memref<8xindex>

  %7 = memref.load %arg1[%c0] : memref<8xindex>
  %8 = memref.load %arg1[%c1] : memref<8xindex>
  %9 = memref.load %arg1[%c2] : memref<8xindex>
  %10 = memref.load %arg1[%c3] : memref<8xindex>
  %11 = memref.load %arg1[%c4] : memref<8xindex>
  %12 = memref.load %arg1[%c5] : memref<8xindex>
  %13 = memref.load %arg1[%c6] : memref<8xindex>

  %14 = arith.addi %0, %7 : index
  %15 = arith.addi %1, %8 : index
  %16 = arith.addi %2, %9 : index
  %17 = arith.addi %3, %10 : index
  %18 = arith.addi %4, %11 : index
  %19 = arith.addi %5, %12 : index
  %20 = arith.addi %6, %13 : index

  memref.store %14, %arg0[%c0] : memref<8xindex>
  memref.store %15, %arg0[%c1] : memref<8xindex>
  memref.store %16, %arg0[%c2] : memref<8xindex>
  memref.store %17, %arg0[%c3] : memref<8xindex>
  memref.store %18, %arg0[%c4] : memref<8xindex>
  memref.store %19, %arg0[%c5] : memref<8xindex>
  memref.store %20, %arg0[%c6] : memref<8xindex>

  return
}


// CHECK-LABEL: func @read_read_add_write_size_mismatch
//  CHECK-SAME: (%[[ARG0:.*]]: memref<8xi32>, %[[ARG1:.*]]: memref<8xi32>)
func.func @read_read_add_write_size_mismatch(%arg0: memref<8xi32>, %arg1: memref<8xi32>) {
  // CHECK:     %[[C0:.*]] = arith.constant 0 : index
  // CHECK:     %[[A:.*]] = vector.load %[[ARG0]][%[[C0]]] : memref<8xi32>, vector<4xi32>
  // CHECK:     %[[B:.*]] = vector.load %[[ARG1]][%[[C0]]] : memref<8xi32>, vector<4xi32>
  // CHECK:     %[[A1:.*]] = vector.extract_strided_slice %[[A]] {offsets = [0], sizes = [2], strides = [1]} : vector<4xi32> to vector<2xi32>
  // CHECK:     %[[B1:.*]] = vector.extract_strided_slice %[[B]] {offsets = [0], sizes = [2], strides = [1]} : vector<4xi32> to vector<2xi32>
  // CHECK:     %[[RES:.*]] = arith.addi %[[A1]], %[[B1]] : vector<2xi32>
  // CHECK:     vector.store %[[RES]], %[[ARG0]][%[[C0]]] : memref<8xi32>, vector<2xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index

  %0 = memref.load %arg0[%c0] : memref<8xi32>
  %1 = memref.load %arg0[%c1] : memref<8xi32>
  %2 = memref.load %arg0[%c2] : memref<8xi32>
  %3 = memref.load %arg0[%c3] : memref<8xi32>

  %4 = memref.load %arg1[%c0] : memref<8xi32>
  %5 = memref.load %arg1[%c1] : memref<8xi32>
  %6 = memref.load %arg1[%c2] : memref<8xi32>
  %7 = memref.load %arg1[%c3] : memref<8xi32>

  %8 = arith.addi %0, %4 : i32
  %9 = arith.addi %1, %5 : i32

  memref.store %8, %arg0[%c0] : memref<8xi32>
  memref.store %9, %arg0[%c1] : memref<8xi32>

  return
}


// CHECK-LABEL: func @read_read_add_write_attrs_mismatch
//  CHECK-SAME: (%[[ARG0:.*]]: memref<8xi32>, %[[ARG1:.*]]: memref<8xi32>)
func.func @read_read_add_write_attrs_mismatch(%arg0: memref<8xi32>, %arg1: memref<8xi32>) {
    // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
    // CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
    // CHECK:     %[[V0:.*]] = vector.load %[[ARG0]][%[[C0]]] : memref<8xi32>, vector<4xi32>
    // CHECK:     %[[V1:.*]] = vector.extract_strided_slice %[[V0]] {offsets = [2], sizes = [2], strides = [1]} : vector<4xi32> to vector<2xi32>
    // CHECK:     %[[V2:.*]] = vector.load %[[ARG1]][%[[C0]]] : memref<8xi32>, vector<4xi32>
    // CHECK:     %[[V3:.*]] = vector.extract_strided_slice %[[V2]] {offsets = [2], sizes = [2], strides = [1]} : vector<4xi32> to vector<2xi32>
    // CHECK:     %[[V4:.*]] = vector.extract_strided_slice %[[V0]] {offsets = [0], sizes = [2], strides = [1]} : vector<4xi32> to vector<2xi32>
    // CHECK:     %[[V5:.*]] = vector.extract_strided_slice %[[V2]] {offsets = [0], sizes = [2], strides = [1]} : vector<4xi32> to vector<2xi32>
    // CHECK:     %[[V6:.*]] = arith.addi %[[V4]], %[[V5]] overflow<nsw> : vector<2xi32>
    // CHECK:     %[[V7:.*]] = arith.addi %[[V1]], %[[V3]] overflow<nuw> : vector<2xi32>
    // CHECK:     vector.store %[[V6]], %[[ARG0]][%[[C0]]] : memref<8xi32>, vector<2xi32>
    // CHECK:     vector.store %[[V7]], %[[ARG0]][%[[C2]]] : memref<8xi32>, vector<2xi32>

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index

  %0 = memref.load %arg0[%c0] : memref<8xi32>
  %1 = memref.load %arg0[%c1] : memref<8xi32>
  %2 = memref.load %arg0[%c2] : memref<8xi32>
  %3 = memref.load %arg0[%c3] : memref<8xi32>

  %4 = memref.load %arg1[%c0] : memref<8xi32>
  %5 = memref.load %arg1[%c1] : memref<8xi32>
  %6 = memref.load %arg1[%c2] : memref<8xi32>
  %7 = memref.load %arg1[%c3] : memref<8xi32>

  %8 = arith.addi %0, %4 overflow<nsw> : i32
  %9 = arith.addi %1, %5 overflow<nsw> : i32
  %10 = arith.addi %2, %6 overflow<nuw> : i32
  %11 = arith.addi %3, %7 overflow<nuw> : i32

  memref.store %8, %arg0[%c0] : memref<8xi32>
  memref.store %9, %arg0[%c1] : memref<8xi32>
  memref.store %10, %arg0[%c2] : memref<8xi32>
  memref.store %11, %arg0[%c3] : memref<8xi32>

  return
}


// CHECK-LABEL: func @read_read_add_write_interleaved
//  CHECK-SAME: (%[[ARG0:.*]]: memref<8xi32>, %[[ARG1:.*]]: memref<8xi32>)
func.func @read_read_add_write_interleaved(%arg0: memref<8xi32>, %arg1: memref<8xi32>) {
  // CHECK:     %[[C0:.*]] = arith.constant 0 : index
  // CHECK:     %[[A:.*]] = vector.load %[[ARG0]][%[[C0]]] : memref<8xi32>, vector<4xi32>
  // CHECK:     %[[B:.*]] = vector.load %[[ARG1]][%[[C0]]] : memref<8xi32>, vector<4xi32>
  // CHECK:     %[[RES:.*]] = arith.addi %[[A]], %[[B]] : vector<4xi32>
  // CHECK:     vector.store %[[RES]], %[[ARG0]][%[[C0]]] : memref<8xi32>, vector<4xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index

  %3 = memref.load %arg0[%c3] : memref<8xi32>
  %7 = memref.load %arg1[%c3] : memref<8xi32>
  %11 = arith.addi %3, %7 : i32

  %0 = memref.load %arg0[%c0] : memref<8xi32>
  %4 = memref.load %arg1[%c0] : memref<8xi32>
  %8 = arith.addi %0, %4 : i32

  %2 = memref.load %arg0[%c2] : memref<8xi32>
  %6 = memref.load %arg1[%c2] : memref<8xi32>
  %10 = arith.addi %2, %6 : i32

  %1 = memref.load %arg0[%c1] : memref<8xi32>
  %5 = memref.load %arg1[%c1] : memref<8xi32>
  %9 = arith.addi %1, %5 : i32

  memref.store %8, %arg0[%c0] : memref<8xi32>
  memref.store %11, %arg0[%c3] : memref<8xi32>
  memref.store %10, %arg0[%c2] : memref<8xi32>
  memref.store %9, %arg0[%c1] : memref<8xi32>

  return
}


// CHECK-LABEL: func @read_read_add_add_write
//  CHECK-SAME: (%[[ARG0:.*]]: memref<8xi32>, %[[ARG1:.*]]: memref<8xi32>
//  CHECK-SAME: , %[[ARG2:.*]]: i32, %[[ARG3:.*]]: i32, %[[ARG4:.*]]: i32, %[[ARG5:.*]]: i32)
func.func @read_read_add_add_write(%arg0: memref<8xi32>, %arg1: memref<8xi32>,
                                   %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) {
  // Each load group have multiple 2 uses (in potentially different order)
  // make sure we the both were vectorized.
  // CHECK:     %[[C0:.*]] = arith.constant 0 : index
  // CHECK:     %[[A:.*]] = vector.load %[[ARG0]][%[[C0]]] : memref<8xi32>, vector<4xi32>
  // CHECK:     %[[B:.*]] = vector.load %[[ARG1]][%[[C0]]] : memref<8xi32>, vector<4xi32>
  // CHECK:     %[[ADD1:.*]] = arith.addi %[[A]], %[[B]] : vector<4xi32>
  // CHECK:     %[[C:.*]] = vector.from_elements %[[ARG2]], %[[ARG3]], %[[ARG4]], %[[ARG5]] : vector<4xi32>
  // CHECK:     %[[ADD2:.*]] = arith.addi %[[A]], %[[C]] : vector<4xi32>
  // CHECK:     vector.store %[[ADD1]], %[[ARG0]][%[[C0]]] : memref<8xi32>, vector<4xi32>
  // CHECK:     vector.store %[[ADD2]], %[[ARG1]][%[[C0]]] : memref<8xi32>, vector<4xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index

  %0 = memref.load %arg0[%c0] : memref<8xi32>
  %1 = memref.load %arg0[%c1] : memref<8xi32>
  %2 = memref.load %arg0[%c2] : memref<8xi32>
  %3 = memref.load %arg0[%c3] : memref<8xi32>

  %4 = memref.load %arg1[%c0] : memref<8xi32>
  %5 = memref.load %arg1[%c1] : memref<8xi32>
  %6 = memref.load %arg1[%c2] : memref<8xi32>
  %7 = memref.load %arg1[%c3] : memref<8xi32>

  %8 = arith.addi %0, %4 : i32
  %12 = arith.addi %0, %arg2 : i32

  %13 = arith.addi %1, %arg3 : i32
  %9 = arith.addi %1, %5 : i32

  %10 = arith.addi %2, %6 : i32
  %14 = arith.addi %2, %arg4 : i32

  %15 = arith.addi %3, %arg5 : i32
  %11 = arith.addi %3, %7 : i32

  memref.store %8, %arg0[%c0] : memref<8xi32>
  memref.store %9, %arg0[%c1] : memref<8xi32>
  memref.store %10, %arg0[%c2] : memref<8xi32>
  memref.store %11, %arg0[%c3] : memref<8xi32>

  memref.store %12, %arg1[%c0] : memref<8xi32>
  memref.store %13, %arg1[%c1] : memref<8xi32>
  memref.store %14, %arg1[%c2] : memref<8xi32>
  memref.store %15, %arg1[%c3] : memref<8xi32>

  return
}

// CHECK-LABEL: func @read_read_add_add
//  CHECK-SAME: (%[[ARG0:.*]]: memref<8xi32>, %[[ARG1:.*]]: memref<8xi32>
//  CHECK-SAME: , %[[ARG2:.*]]: i32, %[[ARG3:.*]]: i32, %[[ARG4:.*]]: i32, %[[ARG5:.*]]: i32)
func.func @read_read_add_add(%arg0: memref<8xi32>, %arg1: memref<8xi32>,
                                   %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) ->
                                   (i32, i32, i32, i32, i32, i32, i32, i32){
  // Each load group have multiple 2 uses (in potentially different order)
  // make sure we the both were vectorized.
  // CHECK:     %[[C0:.*]] = arith.constant 0 : index
  // CHECK:     %[[A:.*]] = vector.load %[[ARG0]][%[[C0]]] : memref<8xi32>, vector<4xi32>
  // CHECK:     %[[B:.*]] = vector.load %[[ARG1]][%[[C0]]] : memref<8xi32>, vector<4xi32>
  // CHECK:     %[[ADD1:.*]] = arith.addi %[[A]], %[[B]] : vector<4xi32>
  // CHECK:     %[[R0:.*]] = vector.extract %[[ADD1]][0] : i32 from vector<4xi32>
  // CHECK:     %[[R1:.*]] = vector.extract %[[ADD1]][1] : i32 from vector<4xi32>
  // CHECK:     %[[R2:.*]] = vector.extract %[[ADD1]][2] : i32 from vector<4xi32>
  // CHECK:     %[[R3:.*]] = vector.extract %[[ADD1]][3] : i32 from vector<4xi32>
  // CHECK:     %[[C:.*]] = vector.from_elements %[[ARG2]], %[[ARG3]], %[[ARG4]], %[[ARG5]] : vector<4xi32>
  // CHECK:     %[[ADD2:.*]] = arith.addi %[[A]], %[[C]] : vector<4xi32>
  // CHECK:     %[[R4:.*]] = vector.extract %[[ADD2]][0] : i32 from vector<4xi32>
  // CHECK:     %[[R5:.*]] = vector.extract %[[ADD2]][1] : i32 from vector<4xi32>
  // CHECK:     %[[R6:.*]] = vector.extract %[[ADD2]][2] : i32 from vector<4xi32>
  // CHECK:     %[[R7:.*]] = vector.extract %[[ADD2]][3] : i32 from vector<4xi32>
  // CHECK:     return %[[R0]], %[[R1]], %[[R2]], %[[R3]], %[[R4]], %[[R5]], %[[R6]], %[[R7]] : i32, i32, i32, i32, i32, i32, i32, i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index

  %0 = memref.load %arg0[%c0] : memref<8xi32>
  %1 = memref.load %arg0[%c1] : memref<8xi32>
  %2 = memref.load %arg0[%c2] : memref<8xi32>
  %3 = memref.load %arg0[%c3] : memref<8xi32>

  %4 = memref.load %arg1[%c0] : memref<8xi32>
  %5 = memref.load %arg1[%c1] : memref<8xi32>
  %6 = memref.load %arg1[%c2] : memref<8xi32>
  %7 = memref.load %arg1[%c3] : memref<8xi32>

  %8 = arith.addi %0, %4 : i32
  %12 = arith.addi %0, %arg2 : i32

  %13 = arith.addi %1, %arg3 : i32
  %9 = arith.addi %1, %5 : i32

  %10 = arith.addi %2, %6 : i32
  %14 = arith.addi %2, %arg4 : i32

  %15 = arith.addi %3, %arg5 : i32
  %11 = arith.addi %3, %7 : i32

  return %8, %9, %10, %11, %12, %13, %14, %15 : i32, i32, i32, i32, i32, i32, i32, i32
}


// CHECK-LABEL: func @read_read_add_add_vec
//  CHECK-SAME: (%[[ARG0:.*]]: memref<8xi32>, %[[ARG1:.*]]: memref<8xi32>)
func.func @read_read_add_add_vec(%arg0: memref<8xi32>, %arg1: memref<8xi32>) ->
                                   (vector<2xi32>, vector<2xi32>){
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[V0:.*]] = vector.load %[[ARG0]][%[[C0]]] : memref<8xi32>, vector<4xi32>
  // CHECK: %[[V1:.*]] = vector.load %[[ARG1]][%[[C0]]] : memref<8xi32>, vector<4xi32>
  // CHECK: %[[V2:.*]] = arith.addi %[[V0]], %[[V1]] : vector<4xi32>
  // CHECK: %[[V3:.*]] = vector.extract_strided_slice %[[V2]] {offsets = [0], sizes = [2], strides = [1]} : vector<4xi32> to vector<2xi32>
  // CHECK: %[[V4:.*]] = vector.extract_strided_slice %[[V2]] {offsets = [2], sizes = [2], strides = [1]} : vector<4xi32> to vector<2xi32>
  // CHECK: return %[[V3]], %[[V4]] : vector<2xi32>, vector<2xi32>
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index

  %0 = vector.load %arg0[%c0] : memref<8xi32>, vector<2xi32>
  %2 = vector.load %arg0[%c2] : memref<8xi32>, vector<2xi32>

  %4 = vector.load %arg1[%c0] : memref<8xi32>, vector<2xi32>
  %6 = vector.load %arg1[%c2] : memref<8xi32>, vector<2xi32>

  %8 = arith.addi %0, %4 : vector<2xi32>
  %10 = arith.addi %2, %6 : vector<2xi32>

  return %8, %10 : vector<2xi32>, vector<2xi32>
}


// CHECK-LABEL: func @read_read_add_add_vec1
//  CHECK-SAME: (%[[ARG0:.*]]: memref<8xi32>, %[[ARG1:.*]]: memref<8xi32>)
func.func @read_read_add_add_vec1(%arg0: memref<8xi32>, %arg1: memref<8xi32>) ->
                                   (vector<1xi32>, vector<1xi32>){
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[V0:.*]] = vector.load %[[ARG0]][%[[C0]]] : memref<8xi32>, vector<2xi32>
  // CHECK: %[[V1:.*]] = vector.load %[[ARG1]][%[[C0]]] : memref<8xi32>, vector<2xi32>
  // CHECK: %[[V2:.*]] = arith.addi %[[V0]], %[[V1]] : vector<2xi32>
  // CHECK: %[[V3:.*]] = vector.extract_strided_slice %[[V2]] {offsets = [0], sizes = [1], strides = [1]} : vector<2xi32> to vector<1xi32>
  // CHECK: %[[V4:.*]] = vector.extract_strided_slice %[[V2]] {offsets = [1], sizes = [1], strides = [1]} : vector<2xi32> to vector<1xi32>
  // CHECK: return %[[V3]], %[[V4]] : vector<1xi32>, vector<1xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  %0 = vector.load %arg0[%c0] : memref<8xi32>, vector<1xi32>
  %2 = vector.load %arg0[%c1] : memref<8xi32>, vector<1xi32>

  %4 = vector.load %arg1[%c0] : memref<8xi32>, vector<1xi32>
  %6 = vector.load %arg1[%c1] : memref<8xi32>, vector<1xi32>

  %8 = arith.addi %0, %4 : vector<1xi32>
  %10 = arith.addi %2, %6 : vector<1xi32>

  return %8, %10 : vector<1xi32>, vector<1xi32>
}


// CHECK-LABEL: func @read_read_add_add_vec0d
//  CHECK-SAME: (%[[ARG0:.*]]: memref<8xi32>, %[[ARG1:.*]]: memref<8xi32>)
func.func @read_read_add_add_vec0d(%arg0: memref<8xi32>, %arg1: memref<8xi32>) ->
                                   (vector<i32>, vector<i32>){
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[V0:.*]] = vector.load %[[ARG0]][%[[C0]]] : memref<8xi32>, vector<2xi32>
  // CHECK: %[[V1:.*]] = vector.load %[[ARG1]][%[[C0]]] : memref<8xi32>, vector<2xi32>
  // CHECK: %[[V2:.*]] = arith.addi %[[V0]], %[[V1]] : vector<2xi32>
  // CHECK: %[[V3:.*]] = vector.extract %[[V2]][0] : i32 from vector<2xi32>
  // CHECK: %[[V4:.*]] = vector.splat %[[V3]] : vector<i32>
  // CHECK: %[[V5:.*]] = vector.extract %[[V2]][1] : i32 from vector<2xi32>
  // CHECK: %[[V6:.*]] = vector.splat %[[V5]] : vector<i32>
  // CHECK: return %[[V4]], %[[V6]] : vector<i32>, vector<i32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  %0 = vector.load %arg0[%c0] : memref<8xi32>, vector<i32>
  %2 = vector.load %arg0[%c1] : memref<8xi32>, vector<i32>

  %4 = vector.load %arg1[%c0] : memref<8xi32>, vector<i32>
  %6 = vector.load %arg1[%c1] : memref<8xi32>, vector<i32>

  %8 = arith.addi %0, %4 : vector<i32>
  %10 = arith.addi %2, %6 : vector<i32>

  return %8, %10 : vector<i32>, vector<i32>
}


func.func private @use(i32)

// CHECK-LABEL: func @read_read_add_write_interleaved_use
//  CHECK-SAME: (%[[ARG0:.*]]: memref<8xi32>, %[[ARG1:.*]]: memref<8xi32>)
func.func @read_read_add_write_interleaved_use(%arg0: memref<8xi32>, %arg1: memref<8xi32>) {
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
  // CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
  // CHECK: %[[V0:.*]] = memref.load %[[ARG0]][%[[C3]]] : memref<8xi32>
  // CHECK: %[[V1:.*]] = memref.load %[[ARG1]][%[[C3]]] : memref<8xi32>
  // CHECK: call @use(%[[V0]]) : (i32) -> ()
  // CHECK: %[[V2:.*]] = vector.load %[[ARG0]][%[[C0]]] : memref<8xi32>, vector<2xi32>
  // CHECK: %[[V3:.*]] = vector.load %[[ARG1]][%[[C0]]] : memref<8xi32>, vector<2xi32>
  // CHECK: %[[V4:.*]] = memref.load %[[ARG0]][%[[C2]]] : memref<8xi32>
  // CHECK: %[[V5:.*]] = memref.load %[[ARG1]][%[[C2]]] : memref<8xi32>
  // CHECK: %[[V6:.*]] = vector.extract %[[V2]][0] : i32 from vector<2xi32>
  // CHECK: %[[V7:.*]] = vector.extract %[[V2]][1] : i32 from vector<2xi32>
  // CHECK: %[[V8:.*]] = vector.from_elements %[[V6]], %[[V7]], %[[V4]], %[[V0]] : vector<4xi32>
  // CHECK: %[[V9:.*]] = vector.extract %[[V3]][0] : i32 from vector<2xi32>
  // CHECK: %[[V10:.*]] = vector.extract %[[V3]][1] : i32 from vector<2xi32>
  // CHECK: %[[V11:.*]] = vector.from_elements %[[V9]], %[[V10]], %[[V5]], %[[V1]] : vector<4xi32>
  // CHECK: %[[V12:.*]] = arith.addi %[[V8]], %[[V11]] : vector<4xi32>
  // CHECK: vector.store %[[V12]], %[[ARG0]][%[[C0]]] : memref<8xi32>, vector<4xi32>

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index

  %3 = memref.load %arg0[%c3] : memref<8xi32>
  %7 = memref.load %arg1[%c3] : memref<8xi32>
  call @use(%3) : (i32) -> ()
  %11 = arith.addi %3, %7 : i32

  %0 = memref.load %arg0[%c0] : memref<8xi32>
  %4 = memref.load %arg1[%c0] : memref<8xi32>
  %8 = arith.addi %0, %4 : i32

  %2 = memref.load %arg0[%c2] : memref<8xi32>
  %6 = memref.load %arg1[%c2] : memref<8xi32>
  %10 = arith.addi %2, %6 : i32

  %1 = memref.load %arg0[%c1] : memref<8xi32>
  %5 = memref.load %arg1[%c1] : memref<8xi32>
  %9 = arith.addi %1, %5 : i32

  memref.store %8, %arg0[%c0] : memref<8xi32>
  memref.store %11, %arg0[%c3] : memref<8xi32>
  memref.store %10, %arg0[%c2] : memref<8xi32>
  memref.store %9, %arg0[%c1] : memref<8xi32>

  return
}


// CHECK-LABEL: func @read_read_add_write_interleaved_use_add
//  CHECK-SAME: (%[[ARG0:.*]]: memref<8xi32>, %[[ARG1:.*]]: memref<8xi32>)
func.func @read_read_add_write_interleaved_use_add(%arg0: memref<8xi32>, %arg1: memref<8xi32>) {
  // CHECK:     %[[C0:.*]] = arith.constant 0 : index
  // CHECK:     %[[V0:.*]] = vector.load %[[ARG0]][%[[C0]]] : memref<8xi32>, vector<4xi32>
  // CHECK:     %[[V1:.*]] = vector.extract %[[V0]][3] : i32 from vector<4xi32>
  // CHECK:     %[[V2:.*]] = vector.load %[[ARG1]][%[[C0]]] : memref<8xi32>, vector<4xi32>
  // CHECK:     %[[V3:.*]] = vector.extract %[[V2]][3] : i32 from vector<4xi32>
  // CHECK:     %[[V4:.*]] = arith.subi %[[V1]], %[[V3]] : i32
  // CHECK:     %[[V5:.*]] = arith.addi %[[V0]], %[[V2]] : vector<4xi32>
  // CHECK:     vector.store %[[V5]], %[[ARG0]][%[[C0]]] : memref<8xi32>, vector<4xi32>
  // CHECK:     call @use(%[[V4]]) : (i32) -> ()
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index

  %3 = memref.load %arg0[%c3] : memref<8xi32>
  %7 = memref.load %arg1[%c3] : memref<8xi32>
  %12 = arith.subi %3, %7 : i32
  %11 = arith.addi %3, %7 : i32

  %0 = memref.load %arg0[%c0] : memref<8xi32>
  %4 = memref.load %arg1[%c0] : memref<8xi32>
  %8 = arith.addi %0, %4 : i32

  %2 = memref.load %arg0[%c2] : memref<8xi32>
  %6 = memref.load %arg1[%c2] : memref<8xi32>
  %10 = arith.addi %2, %6 : i32

  %1 = memref.load %arg0[%c1] : memref<8xi32>
  %5 = memref.load %arg1[%c1] : memref<8xi32>
  %9 = arith.addi %1, %5 : i32

  memref.store %8, %arg0[%c0] : memref<8xi32>
  memref.store %11, %arg0[%c3] : memref<8xi32>
  memref.store %10, %arg0[%c2] : memref<8xi32>
  memref.store %9, %arg0[%c1] : memref<8xi32>

  call @use(%12) : (i32) -> ()
  return
}


// CHECK-LABEL: func @different_blocks
//  CHECK-SAME: (%[[ARG0:.*]]: memref<8xi32>, %[[ARG1:.*]]: memref<8xi32>)
func.func @different_blocks(%arg0: memref<8xi32>, %arg1: memref<8xi32>) {
    // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
    // CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
    // CHECK:     %[[V0:.*]] = vector.load %[[ARG0]][%[[C0]]] : memref<8xi32>, vector<4xi32>
    // CHECK:     %[[V1:.*]] = vector.extract_strided_slice %[[V0]] {offsets = [2], sizes = [2], strides = [1]} : vector<4xi32> to vector<2xi32>
    // CHECK:     %[[V2:.*]] = vector.load %[[ARG1]][%[[C0]]] : memref<8xi32>, vector<4xi32>
    // CHECK:     %[[V3:.*]] = vector.extract_strided_slice %[[V2]] {offsets = [2], sizes = [2], strides = [1]} : vector<4xi32> to vector<2xi32>
    // CHECK:     cf.br ^bb1
    // CHECK:   ^bb1:
    // CHECK:     %[[V4:.*]] = vector.extract_strided_slice %[[V0]] {offsets = [0], sizes = [2], strides = [1]} : vector<4xi32> to vector<2xi32>
    // CHECK:     %[[V5:.*]] = vector.extract_strided_slice %[[V2]] {offsets = [0], sizes = [2], strides = [1]} : vector<4xi32> to vector<2xi32>
    // CHECK:     %[[V6:.*]] = arith.addi %[[V4]], %[[V5]] : vector<2xi32>
    // CHECK:     cf.br ^bb2
    // CHECK:   ^bb2:
    // CHECK:     %[[V7:.*]] = arith.addi %[[V1]], %[[V3]] : vector<2xi32>
    // CHECK:     cf.br ^bb3
    // CHECK:   ^bb3:
    // CHECK:     vector.store %[[V6]], %[[ARG0]][%[[C0]]] : memref<8xi32>, vector<2xi32>
    // CHECK:     vector.store %[[V7]], %[[ARG0]][%[[C2]]] : memref<8xi32>, vector<2xi32>

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index

  %0 = memref.load %arg0[%c0] : memref<8xi32>
  %1 = memref.load %arg0[%c1] : memref<8xi32>
  %2 = memref.load %arg0[%c2] : memref<8xi32>
  %3 = memref.load %arg0[%c3] : memref<8xi32>

  %4 = memref.load %arg1[%c0] : memref<8xi32>
  %5 = memref.load %arg1[%c1] : memref<8xi32>
  %6 = memref.load %arg1[%c2] : memref<8xi32>
  %7 = memref.load %arg1[%c3] : memref<8xi32>

  cf.br ^bb0

^bb0:
  %8 = arith.addi %0, %4 : i32
  %9 = arith.addi %1, %5 : i32
  cf.br ^bb1

^bb1:
  %10 = arith.addi %2, %6 : i32
  %11 = arith.addi %3, %7 : i32
  cf.br ^bb2

^bb2:
  memref.store %8, %arg0[%c0] : memref<8xi32>
  memref.store %9, %arg0[%c1] : memref<8xi32>
  memref.store %10, %arg0[%c2] : memref<8xi32>
  memref.store %11, %arg0[%c3] : memref<8xi32>

  return
}
