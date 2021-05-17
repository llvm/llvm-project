// RUN: mlir-opt %s -test-vector-to-vector-conversion | FileCheck %s

// CHECK-LABEL:   func @maskedload0(
// CHECK-SAME:                      %[[A0:.*]]: memref<?xf32>,
// CHECK-SAME:                      %[[A1:.*]]: vector<16xf32>) -> vector<16xf32> {
// CHECK-DAG:       %[[C:.*]] = constant 0 : index
// CHECK-NEXT:      %[[T:.*]] = vector.load %[[A0]][%[[C]]] : memref<?xf32>, vector<16xf32>
// CHECK-NEXT:      return %[[T]] : vector<16xf32>
func @maskedload0(%base: memref<?xf32>, %pass_thru: vector<16xf32>) -> vector<16xf32> {
  %c0 = constant 0 : index
  %mask = vector.constant_mask [16] : vector<16xi1>
  %ld = vector.maskedload %base[%c0], %mask, %pass_thru
    : memref<?xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
  return %ld : vector<16xf32>
}

// CHECK-LABEL:   func @maskedload1(
// CHECK-SAME:                      %[[A0:.*]]: memref<16xf32>,
// CHECK-SAME:                      %[[A1:.*]]: vector<16xf32>) -> vector<16xf32> {
// CHECK-DAG:       %[[C:.*]] = constant 0 : index
// CHECK-NEXT:      %[[T:.*]] = vector.load %[[A0]][%[[C]]] : memref<16xf32>, vector<16xf32>
// CHECK-NEXT:      return %[[T]] : vector<16xf32>
func @maskedload1(%base: memref<16xf32>, %pass_thru: vector<16xf32>) -> vector<16xf32> {
  %c0 = constant 0 : index
  %mask = vector.constant_mask [16] : vector<16xi1>
  %ld = vector.maskedload %base[%c0], %mask, %pass_thru
    : memref<16xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
  return %ld : vector<16xf32>
}

// CHECK-LABEL:   func @maskedload2(
// CHECK-SAME:                      %[[A0:.*]]: memref<16xf32>,
// CHECK-SAME:                      %[[A1:.*]]: vector<16xf32>) -> vector<16xf32> {
// CHECK-NEXT:      return %[[A1]] : vector<16xf32>
func @maskedload2(%base: memref<16xf32>, %pass_thru: vector<16xf32>) -> vector<16xf32> {
  %c0 = constant 0 : index
  %mask = vector.constant_mask [0] : vector<16xi1>
  %ld = vector.maskedload %base[%c0], %mask, %pass_thru
    : memref<16xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
  return %ld : vector<16xf32>
}

// CHECK-LABEL:   func @maskedload3(
// CHECK-SAME:                      %[[A0:.*]]: memref<?xf32>,
// CHECK-SAME:                      %[[A1:.*]]: vector<16xf32>) -> vector<16xf32> {
// CHECK-DAG:       %[[C:.*]] = constant 8 : index
// CHECK-NEXT:      %[[T:.*]] = vector.load %[[A0]][%[[C]]] : memref<?xf32>, vector<16xf32>
// CHECK-NEXT:      return %[[T]] : vector<16xf32>
func @maskedload3(%base: memref<?xf32>, %pass_thru: vector<16xf32>) -> vector<16xf32> {
  %c8 = constant 8 : index
  %mask = vector.constant_mask [16] : vector<16xi1>
  %ld = vector.maskedload %base[%c8], %mask, %pass_thru
    : memref<?xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
  return %ld : vector<16xf32>
}

// CHECK-LABEL:   func @maskedstore1(
// CHECK-SAME:                       %[[A0:.*]]: memref<16xf32>,
// CHECK-SAME:                       %[[A1:.*]]: vector<16xf32>) {
// CHECK-NEXT:      %[[C:.*]] = constant 0 : index
// CHECK-NEXT:      vector.store %[[A1]], %[[A0]][%[[C]]] : memref<16xf32>, vector<16xf32>
// CHECK-NEXT:      return
func @maskedstore1(%base: memref<16xf32>, %value: vector<16xf32>) {
  %c0 = constant 0 : index
  %mask = vector.constant_mask [16] : vector<16xi1>
  vector.maskedstore %base[%c0], %mask, %value : memref<16xf32>, vector<16xi1>, vector<16xf32>
  return
}

// CHECK-LABEL:   func @maskedstore2(
// CHECK-SAME:                       %[[A0:.*]]: memref<16xf32>,
// CHECK-SAME:                       %[[A1:.*]]: vector<16xf32>) {
// CHECK-NEXT:      return
func @maskedstore2(%base: memref<16xf32>, %value: vector<16xf32>)  {
  %c0 = constant 0 : index
  %mask = vector.constant_mask [0] : vector<16xi1>
  vector.maskedstore %base[%c0], %mask, %value : memref<16xf32>, vector<16xi1>, vector<16xf32>
  return
}

// CHECK-LABEL:   func @gather1(
// CHECK-SAME:                  %[[A0:.*]]: memref<16xf32>,
// CHECK-SAME:                  %[[A1:.*]]: vector<16xi32>,
// CHECK-SAME:                  %[[A2:.*]]: vector<16xf32>) -> vector<16xf32> {
// CHECK-NEXT:      %[[C:.*]] = constant 0 : index
// CHECK-NEXT:      %[[M:.*]] = vector.constant_mask [16] : vector<16xi1>
// CHECK-NEXT:      %[[G:.*]] = vector.gather %[[A0]][%[[C]]] [%[[A1]]], %[[M]], %[[A2]] : memref<16xf32>, vector<16xi32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
// CHECK-NEXT:      return %[[G]] : vector<16xf32>
func @gather1(%base: memref<16xf32>, %indices: vector<16xi32>, %pass_thru: vector<16xf32>) -> vector<16xf32> {
  %c0 = constant 0 : index
  %mask = vector.constant_mask [16] : vector<16xi1>
  %ld = vector.gather %base[%c0][%indices], %mask, %pass_thru
    : memref<16xf32>, vector<16xi32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
  return %ld : vector<16xf32>
}

// CHECK-LABEL:   func @gather2(
// CHECK-SAME:                  %[[A0:.*]]: memref<16xf32>,
// CHECK-SAME:                  %[[A1:.*]]: vector<16xi32>,
// CHECK-SAME:                  %[[A2:.*]]: vector<16xf32>) -> vector<16xf32> {
// CHECK-NEXT:      return %[[A2]] : vector<16xf32>
func @gather2(%base: memref<16xf32>, %indices: vector<16xi32>, %pass_thru: vector<16xf32>) -> vector<16xf32> {
  %c0 = constant 0 : index
  %mask = vector.constant_mask [0] : vector<16xi1>
  %ld = vector.gather %base[%c0][%indices], %mask, %pass_thru
    : memref<16xf32>, vector<16xi32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
  return %ld : vector<16xf32>
}

// CHECK-LABEL:   func @scatter1(
// CHECK-SAME:                   %[[A0:.*]]: memref<16xf32>,
// CHECK-SAME:                   %[[A1:.*]]: vector<16xi32>,
// CHECK-SAME:                   %[[A2:.*]]: vector<16xf32>) {
// CHECK-NEXT:      %[[C:.*]] = constant 0 : index
// CHECK-NEXT:      %[[M:.*]] = vector.constant_mask [16] : vector<16xi1>
// CHECK-NEXT:      vector.scatter %[[A0]][%[[C]]] [%[[A1]]], %[[M]], %[[A2]] : memref<16xf32>, vector<16xi32>, vector<16xi1>, vector<16xf32>
// CHECK-NEXT:      return
func @scatter1(%base: memref<16xf32>, %indices: vector<16xi32>, %value: vector<16xf32>) {
  %c0 = constant 0 : index
  %mask = vector.constant_mask [16] : vector<16xi1>
  vector.scatter %base[%c0][%indices], %mask, %value
    : memref<16xf32>, vector<16xi32>, vector<16xi1>, vector<16xf32>
  return
}

// CHECK-LABEL:   func @scatter2(
// CHECK-SAME:                   %[[A0:.*]]: memref<16xf32>,
// CHECK-SAME:                   %[[A1:.*]]: vector<16xi32>,
// CHECK-SAME:                   %[[A2:.*]]: vector<16xf32>) {
// CHECK-NEXT:      return
func @scatter2(%base: memref<16xf32>, %indices: vector<16xi32>, %value: vector<16xf32>) {
  %c0 = constant 0 : index
  %0 = vector.type_cast %base : memref<16xf32> to memref<vector<16xf32>>
  %mask = vector.constant_mask [0] : vector<16xi1>
  vector.scatter %base[%c0][%indices], %mask, %value
    : memref<16xf32>, vector<16xi32>, vector<16xi1>, vector<16xf32>
  return
}

// CHECK-LABEL:   func @expand1(
// CHECK-SAME:                  %[[A0:.*]]: memref<16xf32>,
// CHECK-SAME:                  %[[A1:.*]]: vector<16xf32>) -> vector<16xf32> {
// CHECK-DAG:       %[[C:.*]] = constant 0 : index
// CHECK-NEXT:      %[[T:.*]] = vector.load %[[A0]][%[[C]]] : memref<16xf32>, vector<16xf32>
// CHECK-NEXT:      return %[[T]] : vector<16xf32>
func @expand1(%base: memref<16xf32>, %pass_thru: vector<16xf32>) -> vector<16xf32> {
  %c0 = constant 0 : index
  %mask = vector.constant_mask [16] : vector<16xi1>
  %ld = vector.expandload %base[%c0], %mask, %pass_thru
    : memref<16xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
  return %ld : vector<16xf32>
}

// CHECK-LABEL:   func @expand2(
// CHECK-SAME:                  %[[A0:.*]]: memref<16xf32>,
// CHECK-SAME:                  %[[A1:.*]]: vector<16xf32>) -> vector<16xf32> {
// CHECK-NEXT:      return %[[A1]] : vector<16xf32>
func @expand2(%base: memref<16xf32>, %pass_thru: vector<16xf32>) -> vector<16xf32> {
  %c0 = constant 0 : index
  %mask = vector.constant_mask [0] : vector<16xi1>
  %ld = vector.expandload %base[%c0], %mask, %pass_thru
    : memref<16xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
  return %ld : vector<16xf32>
}

// CHECK-LABEL:   func @compress1(
// CHECK-SAME:                    %[[A0:.*]]: memref<16xf32>,
// CHECK-SAME:                    %[[A1:.*]]: vector<16xf32>) {
// CHECK-NEXT:      %[[C:.*]] = constant 0 : index
// CHECK-NEXT:      vector.store %[[A1]], %[[A0]][%[[C]]] : memref<16xf32>, vector<16xf32>
// CHECK-NEXT:      return
func @compress1(%base: memref<16xf32>, %value: vector<16xf32>) {
  %c0 = constant 0 : index
  %mask = vector.constant_mask [16] : vector<16xi1>
  vector.compressstore %base[%c0], %mask, %value  : memref<16xf32>, vector<16xi1>, vector<16xf32>
  return
}

// CHECK-LABEL:   func @compress2(
// CHECK-SAME:                    %[[A0:.*]]: memref<16xf32>,
// CHECK-SAME:                    %[[A1:.*]]: vector<16xf32>) {
// CHECK-NEXT:      return
func @compress2(%base: memref<16xf32>, %value: vector<16xf32>) {
  %c0 = constant 0 : index
  %mask = vector.constant_mask [0] : vector<16xi1>
  vector.compressstore %base[%c0], %mask, %value : memref<16xf32>, vector<16xi1>, vector<16xf32>
  return
}
