// RUN: mlir-opt %s -test-vector-to-vector-conversion | FileCheck %s

//
// TODO: optimize this one too!
//
// CHECK-LABEL: func @maskedload0(
// CHECK-SAME: %[[A0:.*]]: memref<?xf32>,
// CHECK-SAME: %[[A1:.*]]: vector<16xf32>)
// CHECK-NEXT: %[[M:.*]] = vector.constant_mask
// CHECK-NEXT: %[[T:.*]] = vector.maskedload %[[A0]], %[[M]], %[[A1]] : memref<?xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
// CHECK-NEXT: return %[[T]] : vector<16xf32>

func @maskedload0(%base: memref<?xf32>, %pass_thru: vector<16xf32>) -> vector<16xf32> {
  %mask = vector.constant_mask [16] : vector<16xi1>
  %ld = vector.maskedload %base, %mask, %pass_thru
    : memref<?xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
  return %ld : vector<16xf32>
}

// CHECK-LABEL: func @maskedload1(
// CHECK-SAME: %[[A0:.*]]: memref<16xf32>,
// CHECK-SAME: %[[A1:.*]]: vector<16xf32>)
// CHECK-NEXT: %[[T0:.*]] = vector.type_cast %[[A0]] : memref<16xf32> to memref<vector<16xf32>>
// CHECK-NEXT: %[[T1:.*]] = load %[[T0]][] : memref<vector<16xf32>>
// CHECK-NEXT: return %[[T1]] : vector<16xf32>

func @maskedload1(%base: memref<16xf32>, %pass_thru: vector<16xf32>) -> vector<16xf32> {
  %mask = vector.constant_mask [16] : vector<16xi1>
  %ld = vector.maskedload %base, %mask, %pass_thru
    : memref<16xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
  return %ld : vector<16xf32>
}

// CHECK-LABEL: func @maskedload2(
// CHECK-SAME: %[[A0:.*]]: memref<16xf32>,
// CHECK-SAME: %[[A1:.*]]: vector<16xf32>)
// CHECK-NEXT: return %[[A1]] : vector<16xf32>

func @maskedload2(%base: memref<16xf32>, %pass_thru: vector<16xf32>) -> vector<16xf32> {
  %mask = vector.constant_mask [0] : vector<16xi1>
  %ld = vector.maskedload %base, %mask, %pass_thru
    : memref<16xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
  return %ld : vector<16xf32>
}

// CHECK-LABEL: func @maskedstore1(
// CHECK-SAME: %[[A0:.*]]: memref<16xf32>,
// CHECK-SAME: %[[A1:.*]]: vector<16xf32>)
// CHECK-NEXT: %[[T0:.*]] = vector.type_cast %[[A0]] : memref<16xf32> to memref<vector<16xf32>>
// CHECK-NEXT: store %[[A1]], %[[T0]][] : memref<vector<16xf32>>
// CHECK-NEXT: return

func @maskedstore1(%base: memref<16xf32>, %value: vector<16xf32>) {
  %mask = vector.constant_mask [16] : vector<16xi1>
  vector.maskedstore %base, %mask, %value
    : vector<16xi1>, vector<16xf32> into memref<16xf32>
  return
}

// CHECK-LABEL: func @maskedstore2(
// CHECK-SAME: %[[A0:.*]]: memref<16xf32>,
// CHECK-SAME: %[[A1:.*]]: vector<16xf32>)
// CHECK-NEXT: return

func @maskedstore2(%base: memref<16xf32>, %value: vector<16xf32>)  {
  %mask = vector.constant_mask [0] : vector<16xi1>
  vector.maskedstore %base, %mask, %value
    : vector<16xi1>, vector<16xf32> into memref<16xf32>
  return
}

// CHECK-LABEL: func @gather1(
// CHECK-SAME: %[[A0:.*]]: memref<16xf32>,
// CHECK-SAME: %[[A1:.*]]: vector<16xi32>,
// CHECK-SAME: %[[A2:.*]]: vector<16xf32>)
// CHECK-NEXT: %[[T0:.*]] = vector.constant_mask [16] : vector<16xi1>
// CHECK-NEXT: %[[T1:.*]] = vector.gather %[[A0]], %[[A1]], %[[T0]], %[[A2]] : (memref<16xf32>, vector<16xi32>, vector<16xi1>, vector<16xf32>) -> vector<16xf32>
// CHECK-NEXT: return %1 : vector<16xf32>

func @gather1(%base: memref<16xf32>, %indices: vector<16xi32>, %pass_thru: vector<16xf32>) -> vector<16xf32> {
  %mask = vector.constant_mask [16] : vector<16xi1>
  %ld = vector.gather %base, %indices, %mask, %pass_thru
    : (memref<16xf32>, vector<16xi32>, vector<16xi1>, vector<16xf32>) -> vector<16xf32>
  return %ld : vector<16xf32>
}

// CHECK-LABEL: func @gather2(
// CHECK-SAME: %[[A0:.*]]: memref<16xf32>,
// CHECK-SAME: %[[A1:.*]]: vector<16xi32>,
// CHECK-SAME: %[[A2:.*]]: vector<16xf32>)
// CHECK-NEXT: return %[[A2]] : vector<16xf32>

func @gather2(%base: memref<16xf32>, %indices: vector<16xi32>, %pass_thru: vector<16xf32>) -> vector<16xf32> {
  %mask = vector.constant_mask [0] : vector<16xi1>
  %ld = vector.gather %base, %indices, %mask, %pass_thru
    : (memref<16xf32>, vector<16xi32>, vector<16xi1>, vector<16xf32>) -> vector<16xf32>
  return %ld : vector<16xf32>
}

// CHECK-LABEL: func @scatter1(
// CHECK-SAME: %[[A0:.*]]: memref<16xf32>,
// CHECK-SAME: %[[A1:.*]]: vector<16xi32>,
// CHECK-SAME: %[[A2:.*]]: vector<16xf32>)
// CHECK-NEXT: %[[T0:.*]] = vector.constant_mask [16] : vector<16xi1>
// CHECK-NEXT: vector.scatter %[[A0]], %[[A1]], %[[T0]], %[[A2]] : vector<16xi32>, vector<16xi1>, vector<16xf32> into memref<16xf32>
// CHECK-NEXT: return

func @scatter1(%base: memref<16xf32>, %indices: vector<16xi32>, %value: vector<16xf32>) {
  %mask = vector.constant_mask [16] : vector<16xi1>
  vector.scatter %base, %indices, %mask, %value
    : vector<16xi32>, vector<16xi1>, vector<16xf32> into memref<16xf32>
  return
}

// CHECK-LABEL: func @scatter2(
// CHECK-SAME: %[[A0:.*]]: memref<16xf32>,
// CHECK-SAME: %[[A1:.*]]: vector<16xi32>,
// CHECK-SAME: %[[A2:.*]]: vector<16xf32>)
// CHECK-NEXT: return

func @scatter2(%base: memref<16xf32>, %indices: vector<16xi32>, %value: vector<16xf32>) {
  %0 = vector.type_cast %base : memref<16xf32> to memref<vector<16xf32>>
  %mask = vector.constant_mask [0] : vector<16xi1>
  vector.scatter %base, %indices, %mask, %value
    : vector<16xi32>, vector<16xi1>, vector<16xf32> into memref<16xf32>
  return
}

// CHECK-LABEL: func @expand1(
// CHECK-SAME: %[[A0:.*]]: memref<16xf32>,
// CHECK-SAME: %[[A1:.*]]: vector<16xf32>)
// CHECK-NEXT: %[[T0:.*]] = vector.type_cast %[[A0]] : memref<16xf32> to memref<vector<16xf32>>
// CHECK-NEXT: %[[T1:.*]] = load %[[T0]][] : memref<vector<16xf32>>
// CHECK-NEXT: return %[[T1]] : vector<16xf32>

func @expand1(%base: memref<16xf32>, %pass_thru: vector<16xf32>) -> vector<16xf32> {
  %mask = vector.constant_mask [16] : vector<16xi1>
  %ld = vector.expandload %base, %mask, %pass_thru
    : memref<16xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
  return %ld : vector<16xf32>
}

// CHECK-LABEL: func @expand2(
// CHECK-SAME: %[[A0:.*]]: memref<16xf32>,
// CHECK-SAME: %[[A1:.*]]: vector<16xf32>)
// CHECK-NEXT: return %[[A1]] : vector<16xf32>

func @expand2(%base: memref<16xf32>, %pass_thru: vector<16xf32>) -> vector<16xf32> {
  %mask = vector.constant_mask [0] : vector<16xi1>
  %ld = vector.expandload %base, %mask, %pass_thru
    : memref<16xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
  return %ld : vector<16xf32>
}

// CHECK-LABEL: func @compress1(
// CHECK-SAME: %[[A0:.*]]: memref<16xf32>,
// CHECK-SAME: %[[A1:.*]]: vector<16xf32>)
// CHECK-NEXT: %[[T0:.*]] = vector.type_cast %[[A0]] : memref<16xf32> to memref<vector<16xf32>>
// CHECK-NEXT: store %[[A1]], %[[T0]][] : memref<vector<16xf32>>
// CHECK-NEXT: return

func @compress1(%base: memref<16xf32>, %value: vector<16xf32>) {
  %mask = vector.constant_mask [16] : vector<16xi1>
  vector.compressstore %base, %mask, %value  : memref<16xf32>, vector<16xi1>, vector<16xf32>
  return
}

// CHECK-LABEL: func @compress2(
// CHECK-SAME: %[[A0:.*]]: memref<16xf32>,
// CHECK-SAME: %[[A1:.*]]: vector<16xf32>)
// CHECK-NEXT: return

func @compress2(%base: memref<16xf32>, %value: vector<16xf32>) {
  %mask = vector.constant_mask [0] : vector<16xi1>
  vector.compressstore %base, %mask, %value : memref<16xf32>, vector<16xi1>, vector<16xf32>
  return
}
