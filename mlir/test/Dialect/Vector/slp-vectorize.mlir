// RUN: mlir-opt %s --slp-vectorizer | FileCheck %s


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


// CHECK-LABEL: func @read_read_add
//  CHECK-SAME: (%[[ARG0:.*]]: memref<8xi32>, %[[ARG1:.*]]: memref<8xi32>)
func.func @read_read_add(%arg0: memref<8xi32>, %arg1: memref<8xi32>) -> (i32, i32, i32, i32){
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
