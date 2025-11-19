// RUN: mlir-opt --flatten-memref %s --split-input-file --verify-diagnostics | FileCheck %s

func.func @load_scalar_from_memref(%input: memref<4x8xf32, strided<[8, 1], offset: 100>>) -> f32 {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %value = memref.load %input[%c1, %c2] : memref<4x8xf32, strided<[8, 1], offset: 100>>
  return %value : f32
}
// CHECK-LABEL: func @load_scalar_from_memref
// CHECK-NEXT: %[[C10:.*]] = arith.constant 10 : index
// CHECK-NEXT: %[[REINT:.*]] = memref.reinterpret_cast %arg0 to offset: [100], sizes: [32], strides: [1]
// CHECK-SAME: memref<4x8xf32, strided<[8, 1], offset: 100>> to memref<32xf32, strided<[1], offset: 100>>
// CHECK-NEXT: memref.load %[[REINT]][%[[C10]]] : memref<32xf32, strided<[1], offset: 100>>


// -----

func.func @load_scalar_from_memref_dynamic_dim(%input: memref<?x?xf32, strided<[?, ?], offset: ?>>, %row: index, %col: index) -> f32 {
  %value = memref.load %input[%col, %row] : memref<?x?xf32, strided<[?, ?], offset: ?>>
  return %value : f32
}

// CHECK: #[[MAP:.*]] = affine_map<()[s0, s1, s2, s3] -> (s0 * s1 + s2 * s3)>
// CHECK: #[[MAP1:.*]] = affine_map<()[s0, s1, s2, s3] -> (s0 * s1, s2 * s3)>
// CHECK: func @load_scalar_from_memref_dynamic_dim
// CHECK-SAME: (%[[ARG0:.*]]: memref<?x?xf32, strided<[?, ?], offset: ?>>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index)
// CHECK: %[[BASE:.*]], %[[OFFSET:.*]], %[[SIZES:.*]]:2, %[[STRIDES:.*]]:2 = memref.extract_strided_metadata %[[ARG0]]
// CHECK: %[[IDX:.*]] = affine.apply #[[MAP]]()[%[[ARG2]], %[[STRIDES]]#0, %[[ARG1]], %[[STRIDES]]#1]
// CHECK: %[[SIZE:.*]] = affine.max #[[MAP1]]()[%[[STRIDES]]#0, %[[SIZES]]#0, %[[STRIDES]]#1, %[[SIZES]]#1]
// CHECK: %[[REINT:.*]] = memref.reinterpret_cast %arg0 to offset: [%[[OFFSET]]], sizes: [%[[SIZE]]], strides: [1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?xf32, strided<[1], offset: ?>> 
// CHECK: memref.load %[[REINT]][%[[IDX]]]

// -----

func.func @load_scalar_from_memref_static_dim(%input: memref<8x12xf32, strided<[24, 2], offset: 100>>) -> f32 {
   %c7 = arith.constant 7 : index
   %c10 = arith.constant 10 : index
  %value = memref.load %input[%c7, %c10] : memref<8x12xf32, strided<[24, 2], offset: 100>>
  return %value : f32
}

// CHECK-LABEL: func @load_scalar_from_memref_static_dim
// CHECK-SAME: (%[[ARG0:.*]]: memref<8x12xf32, strided<[24, 2], offset: 100>>)
// CHECK: %[[C188:.*]] = arith.constant 188 : index
// CHECK: %[[REINT:.*]] = memref.reinterpret_cast %[[ARG0]] to offset: [100], sizes: [192], strides: [1] : memref<8x12xf32, strided<[24, 2], offset: 100>> to memref<192xf32, strided<[1], offset: 100>>
// CHECK: memref.load %[[REINT]][%[[C188]]] : memref<192xf32, strided<[1], offset: 100>>

// -----

func.func @store_scalar_from_memref_padded(%input: memref<4x8xf32, strided<[18, 2], offset: 100>>, %row: index, %col: index, %value: f32) {
  memref.store %value, %input[%col, %row] : memref<4x8xf32, strided<[18, 2], offset: 100>>
  return
}
// CHECK: #[[MAP:.*]] = affine_map<()[s0, s1] -> (s0 * 18 + s1 * 2)>
// CHECK: func @store_scalar_from_memref_padded
// CHECK-SAME: (%[[ARG0:.*]]: memref<4x8xf32, strided<[18, 2], offset: 100>>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: f32)
// CHECK: %[[IDX:.*]] = affine.apply #[[MAP]]()[%[[ARG2]], %[[ARG1]]]
// CHECK: %[[REINT:.*]] = memref.reinterpret_cast %[[ARG0]]
// CHECK: memref.store %[[ARG3]], %[[REINT]][%[[IDX]]] : memref<72xf32, strided<[1], offset: 100>>

// -----

func.func @store_scalar_from_memref_dynamic_dim(%input: memref<?x?xf32, strided<[?, ?], offset: ?>>, %row: index, %col: index, %value: f32) {
  memref.store %value, %input[%col, %row] : memref<?x?xf32, strided<[?, ?], offset: ?>>
  return
}
// CHECK: #[[MAP:.*]] = affine_map<()[s0, s1, s2, s3] -> (s0 * s1 + s2 * s3)>
// CHECK: #[[MAP1:.*]] = affine_map<()[s0, s1, s2, s3] -> (s0 * s1, s2 * s3)>
// CHECK: func @store_scalar_from_memref_dynamic_dim
// CHECK-SAME: (%[[ARG0:.*]]: memref<?x?xf32, strided<[?, ?], offset: ?>>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: f32)
// CHECK: %[[BASE:.*]], %[[OFFSET:.*]], %[[SIZES:.*]]:2, %[[STRIDES:.*]]:2 = memref.extract_strided_metadata %[[ARG0]]
// CHECK: %[[IDX:.*]] = affine.apply #[[MAP]]()[%[[ARG2]], %[[STRIDES]]#0, %[[ARG1]], %[[STRIDES]]#1]
// CHECK: %[[SIZE:.*]] = affine.max #[[MAP1]]()[%[[STRIDES]]#0, %[[SIZES]]#0, %[[STRIDES]]#1, %[[SIZES]]#1]
// CHECK: %[[REINT:.*]] = memref.reinterpret_cast %[[ARG0]] to offset: [%[[OFFSET]]], sizes: [%[[SIZE]]], strides: [1]
// CHECK: memref.store %[[ARG3]], %[[REINT]][%[[IDX]]]

// -----

func.func @load_vector_from_memref(%input: memref<4x8xf32>) -> vector<8xf32> {
  %c3 = arith.constant 3 : index
  %c6 = arith.constant 6 : index
  %value = vector.load %input[%c3, %c6] : memref<4x8xf32>, vector<8xf32>
  return %value : vector<8xf32>
}
// CHECK-LABEL: func @load_vector_from_memref
// CHECK: %[[C30:.*]] = arith.constant 30
// CHECK-NEXT: %[[REINT:.*]] = memref.reinterpret_cast %arg0 to offset: [0], sizes: [32], strides: [1]
// CHECK-NEXT: vector.load %[[REINT]][%[[C30]]]

// -----

func.func @load_vector_from_memref_odd(%input: memref<3x7xi2>) -> vector<3xi2> {
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %value = vector.load %input[%c1, %c3] : memref<3x7xi2>, vector<3xi2>
  return %value : vector<3xi2>
}
// CHECK-LABEL: func @load_vector_from_memref_odd
// CHECK: %[[C10:.*]] = arith.constant 10 : index
// CHECK-NEXT: %[[REINT:.*]] = memref.reinterpret_cast
// CHECK-NEXT: vector.load %[[REINT]][%[[C10]]]

// -----

func.func @load_vector_from_memref_dynamic(%input: memref<3x7xi2>, %row: index, %col: index) -> vector<3xi2> {
  %value = vector.load %input[%col, %row] : memref<3x7xi2>, vector<3xi2>
  return %value : vector<3xi2>
}
// CHECK: #[[MAP:.*]] = affine_map<()[s0, s1] -> (s0 * 7 + s1)>
// CHECK: func @load_vector_from_memref_dynamic
// CHECK: %[[IDX:.*]] = affine.apply #[[MAP]]()
// CHECK: %[[REINT:.*]] = memref.reinterpret_cast
// CHECK: vector.load %[[REINT]][%[[IDX]]] : memref<21xi2, strided<[1]>>, vector<3xi2>

// -----

func.func @store_vector_to_memref_odd(%input: memref<3x7xi2>, %value: vector<3xi2>) {
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  vector.store %value, %input[%c1, %c3] : memref<3x7xi2>, vector<3xi2>
  return
}
// CHECK-LABEL: func @store_vector_to_memref_odd
// CHECK-SAME: (%[[ARG0:.*]]: memref<3x7xi2>, %[[ARG1:.*]]: vector<3xi2>)
// CHECK: %[[C10:.*]] = arith.constant 10 : index
// CHECK-NEXT: %[[REINT:.*]] = memref.reinterpret_cast
// CHECK-NEXT: vector.store %[[ARG1]], %[[REINT]][%[[C10]]] : memref<21xi2, strided<[1]>

// -----

func.func @store_vector_to_memref_dynamic(%input: memref<3x7xi2>, %value: vector<3xi2>, %row: index, %col: index) {
  vector.store %value, %input[%col, %row] : memref<3x7xi2>, vector<3xi2>
  return
}
// CHECK: #[[MAP:.*]] = affine_map<()[s0, s1] -> (s0 * 7 + s1)>
// CHECK: func @store_vector_to_memref_dynamic
// CHECK-SAME: (%[[ARG0:.*]]: memref<3x7xi2>, %[[ARG1:.*]]: vector<3xi2>, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index)
// CHECK: %[[IDX:.*]] = affine.apply #[[MAP]]()[%[[ARG3]], %[[ARG2]]]
// CHECK: %[[REINT:.*]] = memref.reinterpret_cast %[[ARG0]] to offset: [0], sizes: [21], strides: [1]
// CHECK: vector.store %[[ARG1]], %[[REINT]][%[[IDX]]]

// -----

func.func @mask_store_vector_to_memref_odd(%input: memref<3x7xi2>, %value: vector<3xi2>, %mask: vector<3xi1>) {
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  vector.maskedstore %input[%c1, %c3], %mask, %value  : memref<3x7xi2>, vector<3xi1>, vector<3xi2>
  return
}
// CHECK-LABEL: func @mask_store_vector_to_memref_odd
// CHECK-SAME: (%[[ARG0:.*]]: memref<3x7xi2>, %[[ARG1:.*]]: vector<3xi2>, %[[ARG2:.*]]: vector<3xi1>)
// CHECK: %[[C10:.*]] = arith.constant 10 : index
// CHECK-NEXT: %[[REINT:.*]] = memref.reinterpret_cast
// CHECK: vector.maskedstore %[[REINT]][%[[C10]]], %[[ARG2]], %[[ARG1]]

// -----

func.func @mask_store_vector_to_memref_dynamic(%input: memref<3x7xi2>, %value: vector<3xi2>, %row: index, %col: index, %mask: vector<3xi1>) {
  vector.maskedstore %input[%col, %row], %mask, %value : memref<3x7xi2>, vector<3xi1>, vector<3xi2>
  return
}
// CHECK: #map = affine_map<()[s0, s1] -> (s0 * 7 + s1)>
// CHECK: func @mask_store_vector_to_memref_dynamic
// CHECK-SAME: (%[[ARG0:.*]]: memref<3x7xi2>, %[[ARG1:.*]]: vector<3xi2>, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index, %[[ARG4:.*]]: vector<3xi1>)
// CHECK: %[[IDX:.*]] = affine.apply #map()[%[[ARG3]], %[[ARG2]]]
// CHECK: %[[REINT:.*]] = memref.reinterpret_cast %[[ARG0]]
// CHECK: vector.maskedstore %[[REINT]][%[[IDX]]], %[[ARG4]], %[[ARG1]]

// -----
func.func @mask_load_vector_from_memref_odd(%input: memref<3x7xi2>, %mask: vector<3xi1>, %passthru: vector<3xi2>) -> vector<3xi2> {
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %result = vector.maskedload %input[%c1, %c3], %mask, %passthru : memref<3x7xi2>, vector<3xi1>, vector<3xi2> into vector<3xi2>
  return %result : vector<3xi2>
}
// CHECK-LABEL: func @mask_load_vector_from_memref_odd
// CHECK-SAME: (%[[ARG0:.*]]: memref<3x7xi2>, %[[MASK:.*]]: vector<3xi1>, %[[PASSTHRU:.*]]: vector<3xi2>)
// CHECK: %[[C10:.*]] = arith.constant 10 : index
// CHECK: %[[REINT:.*]] = memref.reinterpret_cast %[[ARG0]] to offset: [0], sizes: [21], strides: [1]
// CHECK: vector.maskedload %[[REINT]][%[[C10]]], %[[MASK]], %[[PASSTHRU]]

// -----

func.func @mask_load_vector_from_memref_dynamic(%input: memref<3x7xi2>, %row: index, %col: index, %mask: vector<3xi1>, %passthru: vector<3xi2>) -> vector<3xi2> {
  %result = vector.maskedload %input[%col, %row], %mask, %passthru : memref<3x7xi2>, vector<3xi1>, vector<3xi2> into vector<3xi2>
  return %result : vector<3xi2>
}
// CHECK: #[[MAP:.*]] = affine_map<()[s0, s1] -> (s0 * 7 + s1)>
// CHECK: func @mask_load_vector_from_memref_dynamic
// CHECK-SAME: (%[[ARG0:.*]]: memref<3x7xi2>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: vector<3xi1>, %[[ARG4:.*]]: vector<3xi2>)
// CHECK: %[[IDX:.*]] = affine.apply #[[MAP]]()[%[[ARG2]], %[[ARG1]]]
// CHECK: %[[REINT:.*]] = memref.reinterpret_cast %[[ARG0]]
// CHECK: vector.maskedload %[[REINT]][%[[IDX]]], %[[ARG3]]

// -----

func.func @transfer_read_memref(%input: memref<4x8xi2>, %value: vector<8xi2>, %row: index, %col: index) -> vector<8xi2> {
   %c0 = arith.constant 0 : i2
   %0 = vector.transfer_read %input[%col, %row], %c0 {in_bounds = [true]} : memref<4x8xi2>, vector<8xi2>
   return %0 : vector<8xi2>
}

// CHECK: #[[MAP:.*]] = affine_map<()[s0, s1] -> (s0 * 8 + s1)>
// CHECK: func @transfer_read_memref
// CHECK-SAME: (%[[ARG0:.*]]: memref<4x8xi2>, %[[ARG1:.*]]: vector<8xi2>, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index)
// CHECK: %[[C0:.*]] = arith.constant 0 : i2
// CHECK: %[[IDX:.*]] = affine.apply #[[MAP]]()[%[[ARG3]], %[[ARG2]]]
// CHECK-NEXT: %[[REINT:.*]] = memref.reinterpret_cast %[[ARG0]]
// CHECK-NEXT: vector.transfer_read %[[REINT]][%[[IDX]]], %[[C0]]

// -----

func.func @transfer_read_memref_not_inbound(%input: memref<4x8xi2>, %value: vector<8xi2>, %row: index, %col: index) -> vector<8xi2> {
   %c0 = arith.constant 0 : i2
   %0 = vector.transfer_read %input[%col, %row], %c0 {in_bounds = [false]} : memref<4x8xi2>, vector<8xi2>
   return %0 : vector<8xi2>
}

// CHECK-LABEL: func @transfer_read_memref_not_inbound
// CHECK-SAME: (%[[ARG0:.*]]: memref<4x8xi2>, %[[ARG1:.*]]: vector<8xi2>, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index)
// CHECK: vector.transfer_read %[[ARG0]][%[[ARG3]], %[[ARG2]]]

// -----

func.func @transfer_read_memref_non_id(%input: memref<4x8xi2>, %value: vector<8xi2>, %row: index, %col: index) -> vector<8xi2> {
   %c0 = arith.constant 0 : i2
   %0 = vector.transfer_read %input[%col, %row], %c0 {permutation_map = affine_map<(d0, d1) -> (d0)>, in_bounds = [true]} : memref<4x8xi2>, vector<8xi2>
   return %0 : vector<8xi2>
}

// CHECK-LABEL: func @transfer_read_memref_non_id
// CHECK-SAME: (%[[ARG0:.*]]: memref<4x8xi2>, %[[ARG1:.*]]: vector<8xi2>, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index)
// CHECK: vector.transfer_read %[[ARG0]][%[[ARG3]], %[[ARG2]]]

// -----

func.func @transfer_write_memref(%input: memref<4x8xi2>, %value: vector<8xi2>, %row: index, %col: index) {
   vector.transfer_write %value, %input[%col, %row] {in_bounds = [true]} : vector<8xi2>, memref<4x8xi2>
   return
}

// CHECK: #[[MAP:.*]] = affine_map<()[s0, s1] -> (s0 * 8 + s1)>
// CHECK: func @transfer_write_memref
// CHECK-SAME: (%[[ARG0:.*]]: memref<4x8xi2>, %[[ARG1:.*]]: vector<8xi2>, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index)
// CHECK: %[[IDX:.*]] = affine.apply #[[MAP]]()[%[[ARG3]], %[[ARG2]]]
// CHECK: %[[REINT:.*]] = memref.reinterpret_cast %[[ARG0]]
// CHECK: vector.transfer_write %[[ARG1]], %[[REINT]][%[[IDX]]]

// -----

func.func @alloc() -> memref<4x8xf32> {
  %0 = memref.alloc() : memref<4x8xf32>
  return %0 : memref<4x8xf32>
}

// CHECK-LABEL: func @alloc
// CHECK-SAME: () -> memref<4x8xf32>
// CHECK-NEXT: %[[ALLOC:.*]] = memref.alloc() : memref<32xf32, strided<[1]>>
// CHECK-NEXT: %[[REINT:.*]] = memref.reinterpret_cast %[[ALLOC]] to offset: [0], sizes: [4, 8], strides: [8, 1] : memref<32xf32, strided<[1]>> to memref<4x8xf32>

// -----

func.func @alloca() -> memref<4x8xf32> {
  %0 = memref.alloca() : memref<4x8xf32>
  return %0 : memref<4x8xf32>
}

// CHECK-LABEL: func.func @alloca() -> memref<4x8xf32>
// CHECK: %[[ALLOC:.*]] = memref.alloca() : memref<32xf32, strided<[1]>>
// CHECK: %[[REINT:.*]] = memref.reinterpret_cast %[[ALLOC]] to offset: [0], sizes: [4, 8], strides: [8, 1] : memref<32xf32, strided<[1]>> to memref<4x8xf32>

// -----

func.func @chained_alloc_load() -> vector<8xf32> {
  %c3 = arith.constant 3 : index
  %c6 = arith.constant 6 : index
  %0 = memref.alloc() : memref<4x8xf32>
  %value = vector.load %0[%c3, %c6] : memref<4x8xf32>, vector<8xf32>
  return %value : vector<8xf32>
}

// CHECK-LABEL: func @chained_alloc_load
// CHECK-SAME: () -> vector<8xf32>
// CHECK-NEXT: %[[C30:.*]] = arith.constant 30 : index
// CHECK-NEXT: %[[ALLOC:.*]] = memref.alloc() : memref<32xf32, strided<[1]>>
// CHECK-NEXT: vector.load %[[ALLOC]][%[[C30]]] : memref<32xf32, strided<[1]>>, vector<8xf32>

// -----

func.func @load_scalar_from_memref_static_dim_col_major(%input: memref<4x8xf32, strided<[1, 4], offset: 100>>, %row: index, %col: index) -> f32 {
  %value = memref.load %input[%col, %row] : memref<4x8xf32, strided<[1, 4], offset: 100>>
  return %value : f32
}

// CHECK: #[[MAP:.*]] = affine_map<()[s0, s1] -> (s0 + s1 * 4)>
// CHECK: func @load_scalar_from_memref_static_dim_col_major
// CHECK-SAME: (%[[ARG0:.*]]: memref<4x8xf32, strided<[1, 4], offset: 100>>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index)
// CHECK: %[[IDX:.*]] = affine.apply #[[MAP]]()[%[[ARG2]], %[[ARG1]]]
// CHECK: %[[REINT:.*]] = memref.reinterpret_cast %[[ARG0]] to offset: [100], sizes: [32], strides: [1] : memref<4x8xf32, strided<[1, 4], offset: 100>> to memref<32xf32, strided<[1], offset: 100>>
// CHECK: memref.load %[[REINT]][%[[IDX]]] : memref<32xf32, strided<[1], offset: 100>>
