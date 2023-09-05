// RUN: mlir-opt -gpu-decompose-memrefs -allow-unregistered-dialect -split-input-file %s | FileCheck %s

//       CHECK: #[[MAP:.*]] = affine_map<()[s0, s1, s2, s3, s4] -> (s0 * s1 + s2 * s3 + s4)>
//       CHECK: @decompose_store
//  CHECK-SAME: (%[[VAL:.*]]: f32, %[[MEM:.*]]: memref<?x?x?xf32>)
//       CHECK:  %[[BASE:.*]], %[[OFFSET:.*]], %[[SIZES:.*]]:3, %[[STRIDES:.*]]:3 = memref.extract_strided_metadata %[[MEM]]
//       CHECK:  gpu.launch
//  CHECK-SAME:  threads(%[[TX:.*]], %[[TY:.*]], %[[TZ:.*]]) in
//       CHECK:  %[[IDX:.*]] = affine.apply #[[MAP]]()[%[[TX]], %[[STRIDES]]#0, %[[TY]], %[[STRIDES]]#1, %[[TZ]]]
//       CHECK:  %[[PTR:.*]] = memref.reinterpret_cast %[[BASE]] to offset: [%[[IDX]]], sizes: [], strides: [] : memref<f32> to memref<f32>
//       CHECK:  memref.store %[[VAL]], %[[PTR]][] : memref<f32>
func.func @decompose_store(%arg0 : f32, %arg1 : memref<?x?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %block_dim0 = memref.dim %arg1, %c0 : memref<?x?x?xf32>
  %block_dim1 = memref.dim %arg1, %c1 : memref<?x?x?xf32>
  %block_dim2 = memref.dim %arg1, %c2 : memref<?x?x?xf32>
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1, %grid_z = %c1)
             threads(%tx, %ty, %tz) in (%block_x = %block_dim0, %block_y = %block_dim1, %block_z = %block_dim2) {
    memref.store %arg0, %arg1[%tx, %ty, %tz] : memref<?x?x?xf32>
    gpu.terminator
  }
  return
}

// -----

//       CHECK: #[[MAP:.*]] = affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> (s0 + s1 * s2 + s3 * s4 + s5 * s6)>
//       CHECK: @decompose_store_strided
//  CHECK-SAME: (%[[VAL:.*]]: f32, %[[MEM:.*]]: memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>)
//       CHECK:  %[[BASE:.*]], %[[OFFSET:.*]], %[[SIZES:.*]]:3, %[[STRIDES:.*]]:3 = memref.extract_strided_metadata %[[MEM]]
//       CHECK:  gpu.launch
//  CHECK-SAME:  threads(%[[TX:.*]], %[[TY:.*]], %[[TZ:.*]]) in
//       CHECK:  %[[IDX:.*]] = affine.apply #[[MAP]]()[%[[OFFSET]], %[[TX]], %[[STRIDES]]#0, %[[TY]], %[[STRIDES]]#1, %[[TZ]], %[[STRIDES]]#2]
//       CHECK:  %[[PTR:.*]] = memref.reinterpret_cast %[[BASE]] to offset: [%[[IDX]]], sizes: [], strides: [] : memref<f32> to memref<f32>
//       CHECK:  memref.store %[[VAL]], %[[PTR]][] : memref<f32>
func.func @decompose_store_strided(%arg0 : f32, %arg1 : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %block_dim0 = memref.dim %arg1, %c0 : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
  %block_dim1 = memref.dim %arg1, %c1 : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
  %block_dim2 = memref.dim %arg1, %c2 : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1, %grid_z = %c1)
             threads(%tx, %ty, %tz) in (%block_x = %block_dim0, %block_y = %block_dim1, %block_z = %block_dim2) {
    memref.store %arg0, %arg1[%tx, %ty, %tz] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
    gpu.terminator
  }
  return
}

// -----

//       CHECK: #[[MAP:.*]] = affine_map<()[s0, s1, s2, s3, s4] -> (s0 * s1 + s2 * s3 + s4)>
//       CHECK: @decompose_load
//  CHECK-SAME: (%[[MEM:.*]]: memref<?x?x?xf32>)
//       CHECK:  %[[BASE:.*]], %[[OFFSET:.*]], %[[SIZES:.*]]:3, %[[STRIDES:.*]]:3 = memref.extract_strided_metadata %[[MEM]]
//       CHECK:  gpu.launch
//  CHECK-SAME:  threads(%[[TX:.*]], %[[TY:.*]], %[[TZ:.*]]) in
//       CHECK:  %[[IDX:.*]] = affine.apply #[[MAP]]()[%[[TX]], %[[STRIDES]]#0, %[[TY]], %[[STRIDES]]#1, %[[TZ]]]
//       CHECK:  %[[PTR:.*]] = memref.reinterpret_cast %[[BASE]] to offset: [%[[IDX]]], sizes: [], strides: [] : memref<f32> to memref<f32>
//       CHECK:  %[[RES:.*]] = memref.load %[[PTR]][] : memref<f32>
//       CHECK:  "test.test"(%[[RES]]) : (f32) -> ()
func.func @decompose_load(%arg0 : memref<?x?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %block_dim0 = memref.dim %arg0, %c0 : memref<?x?x?xf32>
  %block_dim1 = memref.dim %arg0, %c1 : memref<?x?x?xf32>
  %block_dim2 = memref.dim %arg0, %c2 : memref<?x?x?xf32>
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1, %grid_z = %c1)
             threads(%tx, %ty, %tz) in (%block_x = %block_dim0, %block_y = %block_dim1, %block_z = %block_dim2) {
    %res = memref.load %arg0[%tx, %ty, %tz] : memref<?x?x?xf32>
    "test.test"(%res) : (f32) -> ()
    gpu.terminator
  }
  return
}

// -----

//       CHECK: #[[MAP:.*]] = affine_map<()[s0, s1, s2, s3, s4] -> (s0 * s1 + s2 * s3 + s4)>
//       CHECK: @decompose_subview
//  CHECK-SAME: (%[[MEM:.*]]: memref<?x?x?xf32>)
//       CHECK:  %[[BASE:.*]], %[[OFFSET:.*]], %[[SIZES:.*]]:3, %[[STRIDES:.*]]:3 = memref.extract_strided_metadata %[[MEM]]
//       CHECK:  gpu.launch
//  CHECK-SAME:  threads(%[[TX:.*]], %[[TY:.*]], %[[TZ:.*]]) in
//       CHECK:  %[[IDX:.*]] = affine.apply #[[MAP]]()[%[[TX]], %[[STRIDES]]#0, %[[TY]], %[[STRIDES]]#1, %[[TZ]]]
//       CHECK:  %[[PTR:.*]] = memref.reinterpret_cast %[[BASE]] to offset: [%[[IDX]]], sizes: [%{{.*}}, %{{.*}}, %{{.*}}], strides: [%[[STRIDES]]#0, %[[STRIDES]]#1, 1]
//       CHECK:  "test.test"(%[[PTR]]) : (memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>) -> ()
func.func @decompose_subview(%arg0 : memref<?x?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %block_dim0 = memref.dim %arg0, %c0 : memref<?x?x?xf32>
  %block_dim1 = memref.dim %arg0, %c1 : memref<?x?x?xf32>
  %block_dim2 = memref.dim %arg0, %c2 : memref<?x?x?xf32>
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1, %grid_z = %c1)
             threads(%tx, %ty, %tz) in (%block_x = %block_dim0, %block_y = %block_dim1, %block_z = %block_dim2) {
    %res = memref.subview %arg0[%tx, %ty, %tz] [%c2, %c2, %c2] [%c1, %c1, %c1] : memref<?x?x?xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
    "test.test"(%res) : (memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>) -> ()
    gpu.terminator
  }
  return
}

// -----

//       CHECK: #[[MAP:.*]] = affine_map<()[s0] -> (s0 * 2)>
//       CHECK: #[[MAP1:.*]] = affine_map<()[s0] -> (s0 * 3)>
//       CHECK: #[[MAP2:.*]] = affine_map<()[s0, s1, s2, s3, s4] -> (s0 * s1 + s2 * s3 + s4)>
//       CHECK: @decompose_subview_strided
//  CHECK-SAME: (%[[MEM:.*]]: memref<?x?x?xf32>)
//       CHECK:  %[[BASE:.*]], %[[OFFSET:.*]], %[[SIZES:.*]]:3, %[[STRIDES:.*]]:3 = memref.extract_strided_metadata %[[MEM]]
//       CHECK:  gpu.launch
//  CHECK-SAME:  threads(%[[TX:.*]], %[[TY:.*]], %[[TZ:.*]]) in
//       CHECK:  %[[IDX:.*]] = affine.apply #[[MAP]]()[%[[STRIDES]]#0]
//       CHECK:  %[[IDX1:.*]] = affine.apply #[[MAP1]]()[%[[STRIDES]]#1]
//       CHECK:  %[[IDX2:.*]] = affine.apply #[[MAP2]]()[%[[TX]], %[[STRIDES]]#0, %[[TY]], %[[STRIDES]]#1, %[[TZ]]]
//       CHECK:  %[[PTR:.*]] = memref.reinterpret_cast %[[BASE]] to offset: [%[[IDX2]]], sizes: [%{{.*}}, %{{.*}}, %{{.*}}], strides: [%[[IDX]], %[[IDX1]], 4]
//       CHECK:  "test.test"(%[[PTR]]) : (memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>) -> ()
func.func @decompose_subview_strided(%arg0 : memref<?x?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %block_dim0 = memref.dim %arg0, %c0 : memref<?x?x?xf32>
  %block_dim1 = memref.dim %arg0, %c1 : memref<?x?x?xf32>
  %block_dim2 = memref.dim %arg0, %c2 : memref<?x?x?xf32>
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1, %grid_z = %c1)
             threads(%tx, %ty, %tz) in (%block_x = %block_dim0, %block_y = %block_dim1, %block_z = %block_dim2) {
    %res = memref.subview %arg0[%tx, %ty, %tz] [%c2, %c2, %c2] [2, 3, 4] : memref<?x?x?xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
    "test.test"(%res) : (memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>) -> ()
    gpu.terminator
  }
  return
}
