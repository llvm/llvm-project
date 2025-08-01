// RUN: mlir-opt %s -convert-vector-to-llvm -split-input-file | FileCheck '-D$IDX_TYPE=i32' %s
// RUN: mlir-opt %s --convert-vector-to-llvm='force-32bit-vector-indices=0' | FileCheck '-D$IDX_TYPE=i64' %s

func.func @transfer_read_write_1d(%A : memref<?xf32>, %base: index) -> vector<17xf32> {
  %f7 = arith.constant 7.0: f32
  %f = vector.transfer_read %A[%base], %f7
      {permutation_map = affine_map<(d0) -> (d0)>} :
    memref<?xf32>, vector<17xf32>
  vector.transfer_write %f, %A[%base]
      {permutation_map = affine_map<(d0) -> (d0)>} :
    vector<17xf32>, memref<?xf32>
  return %f: vector<17xf32>
}
// CHECK-LABEL: func @transfer_read_write_1d
//  CHECK-SAME: %[[MEM:.*]]: memref<?xf32>,
//  CHECK-SAME: %[[BASE:.*]]: index) -> vector<17xf32>
// 1. Create pass-through vector.
//   CHECK-DAG: %[[PASS_THROUGH:.*]] = arith.constant dense<7.000000e+00> : vector<17xf32>
//
// 2. Create a vector with linear indices [ 0 .. vector_length - 1 ].
//   CHECK-DAG: %[[linearIndex:.*]] = arith.constant dense
//  CHECK-SAME: <[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]> : vector<17x[[$IDX_TYPE]]>
//
// 3. Let dim be the memref dimension, compute the in-bound index (dim - offset)
//   CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
//       CHECK: %[[DIM:.*]] = memref.dim %[[MEM]], %[[C0]] : memref<?xf32>
//       CHECK: %[[BOUND:.*]] = arith.subi %[[DIM]],  %[[BASE]] : index
//
// 4. Create bound vector to compute in-bound mask:
//    [ 0 .. vector_length - 1 ] < [ dim - offset .. dim - offset ]
//       CHECK: %[[btrunc:.*]] = arith.index_cast %[[BOUND]] :
//  CMP32-SAME: index to i32
//  CMP64-SAME: index to i64
//       CHECK: %[[boundVecInsert:.*]] = llvm.insertelement %[[btrunc]]
//       CHECK: %[[boundVect:.*]] = llvm.shufflevector %[[boundVecInsert]]
//       CHECK: %[[mask:.*]] = arith.cmpi sgt, %[[boundVect]], %[[linearIndex]] : vector<17x[[$IDX_TYPE]]>
//  CMP64-SAME: : vector<17xi64>
//
// 5. Bitcast to vector form.
//       CHECK: %[[gep:.*]] = llvm.getelementptr %{{.*}} :
//  CHECK-SAME: (!llvm.ptr, i64) -> !llvm.ptr, f32
//
// 6. Rewrite as a masked read.
//       CHECK: %[[loaded:.*]] = llvm.intr.masked.load %[[gep]], %[[mask]],
//  CHECK-SAME: %[[PASS_THROUGH]] {alignment = 4 : i32} :
//  CHECK-SAME: -> vector<17xf32>
//
// 1. Let dim be the memref dimension, compute the in-bound index (dim - offset)
//       CHECK: %[[DIM_b:.*]] = memref.dim %[[MEM]], %[[C0]] : memref<?xf32>
//       CHECK: %[[BOUND_b:.*]] = arith.subi %[[DIM_b]], %[[BASE]] : index
//
// 2. Create bound vector to compute in-bound mask:
//    [ 0 .. vector_length - 1 ] < [ dim - offset .. dim - offset ]
//       CHECK: %[[btrunc_b:.*]] = arith.index_cast %[[BOUND_b]]
//  CMP32-SAME: index to i32
//       CHECK: %[[boundVecInsert_b:.*]] = llvm.insertelement %[[btrunc_b]]
//       CHECK: %[[boundVect_b:.*]] = llvm.shufflevector %[[boundVecInsert_b]]
//       CHECK: %[[mask_b:.*]] = arith.cmpi sgt, %[[boundVect_b]],
//  CHECK-SAME: %[[linearIndex]] : vector<17x[[$IDX_TYPE]]>
//
// 3. Bitcast to vector form.
//       CHECK: %[[gep_b:.*]] = llvm.getelementptr {{.*}} :
//  CHECK-SAME: (!llvm.ptr, i64) -> !llvm.ptr, f32
//
// 4. Rewrite as a masked write.
//       CHECK: llvm.intr.masked.store %[[loaded]], %[[gep_b]], %[[mask_b]]
//  CHECK-SAME: {alignment = 4 : i32} :
//  CHECK-SAME: vector<17xf32>, vector<17xi1> into !llvm.ptr

func.func @transfer_read_write_1d_scalable(%A : memref<?xf32>, %base: index) -> vector<[17]xf32> {
  %f7 = arith.constant 7.0: f32
  %f = vector.transfer_read %A[%base], %f7
      {permutation_map = affine_map<(d0) -> (d0)>} :
    memref<?xf32>, vector<[17]xf32>
  vector.transfer_write %f, %A[%base]
      {permutation_map = affine_map<(d0) -> (d0)>} :
    vector<[17]xf32>, memref<?xf32>
  return %f: vector<[17]xf32>
}
// CHECK-LABEL: func @transfer_read_write_1d_scalable
//  CHECK-SAME: %[[MEM:.*]]: memref<?xf32>,
//  CHECK-SAME: %[[BASE:.*]]: index) -> vector<[17]xf32>
// 1. Create pass-through vector.
//   CHECK-DAG: %[[PASS_THROUGH:.*]] = arith.constant dense<7.000000e+00> : vector<[17]xf32>
//
// 2. Let dim be the memref dimension, compute the in-bound index (dim - offset)
//   CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
//       CHECK: %[[DIM:.*]] = memref.dim %[[MEM]], %[[C0]] : memref<?xf32>
//       CHECK: %[[BOUND:.*]] = arith.subi %[[DIM]],  %[[BASE]] : index
//
// 3. Create a vector with linear indices [ 0 .. vector_length - 1 ].
//       CHECK: %[[linearIndex:.*]] = llvm.intr.stepvector : vector<[17]x[[$IDX_TYPE]]>
//
// 4. Create bound vector to compute in-bound mask:
//    [ 0 .. vector_length - 1 ] < [ dim - offset .. dim - offset ]
//       CHECK: %[[btrunc:.*]] = arith.index_cast %[[BOUND]] : index to [[$IDX_TYPE]]
//       CHECK: %[[boundVecInsert:.*]] = llvm.insertelement %[[btrunc]]
//       CHECK: %[[boundVect:.*]] = llvm.shufflevector %[[boundVecInsert]]
//       CHECK: %[[mask:.*]] = arith.cmpi slt, %[[linearIndex]], %[[boundVect]]
//  CHECK-SAME: : vector<[17]x[[$IDX_TYPE]]>
//
// 5. Bitcast to vector form.
//       CHECK: %[[gep:.*]] = llvm.getelementptr %{{.*}} :
//  CHECK-SAME: (!llvm.ptr, i64) -> !llvm.ptr, f32
//
// 6. Rewrite as a masked read.
//       CHECK: %[[loaded:.*]] = llvm.intr.masked.load %[[gep]], %[[mask]],
//  CHECK-SAME: %[[PASS_THROUGH]] {alignment = 4 : i32} :
//  CHECK-SAME: -> vector<[17]xf32>
//
// 1. Let dim be the memref dimension, compute the in-bound index (dim - offset)
//       CHECK: %[[DIM_b:.*]] = memref.dim %[[MEM]], %[[C0]] : memref<?xf32>
//       CHECK: %[[BOUND_b:.*]] = arith.subi %[[DIM_b]], %[[BASE]] : index
//
// 2. Create a vector with linear indices [ 0 .. vector_length - 1 ].
//       CHECK: %[[linearIndex_b:.*]] = llvm.intr.stepvector : vector<[17]x[[$IDX_TYPE]]>
//
// 3. Create bound vector to compute in-bound mask:
//    [ 0 .. vector_length - 1 ] < [ dim - offset .. dim - offset ]
//       CHECK: %[[btrunc_b:.*]] = arith.index_cast %[[BOUND_b]] : index to [[$IDX_TYPE]]
//       CHECK: %[[boundVecInsert_b:.*]] = llvm.insertelement %[[btrunc_b]]
//       CHECK: %[[boundVect_b:.*]] = llvm.shufflevector %[[boundVecInsert_b]]
//       CHECK: %[[mask_b:.*]] = arith.cmpi slt, %[[linearIndex_b]],
//  CHECK-SAME: %[[boundVect_b]] : vector<[17]x[[$IDX_TYPE]]>
//
// 4. Bitcast to vector form.
//       CHECK: %[[gep_b:.*]] = llvm.getelementptr {{.*}} :
//  CHECK-SAME: (!llvm.ptr, i64) -> !llvm.ptr, f32
//
// 5. Rewrite as a masked write.
//       CHECK: llvm.intr.masked.store %[[loaded]], %[[gep_b]], %[[mask_b]]
//  CHECK-SAME: {alignment = 4 : i32} :
//  CHECK-SAME: vector<[17]xf32>, vector<[17]xi1> into !llvm.ptr

// -----

func.func @transfer_read_write_index_1d(%A : memref<?xindex>, %base: index) -> vector<17xindex> {
  %f7 = arith.constant 7: index
  %f = vector.transfer_read %A[%base], %f7
      {permutation_map = affine_map<(d0) -> (d0)>} :
    memref<?xindex>, vector<17xindex>
  vector.transfer_write %f, %A[%base]
      {permutation_map = affine_map<(d0) -> (d0)>} :
    vector<17xindex>, memref<?xindex>
  return %f: vector<17xindex>
}
// CHECK-LABEL: func @transfer_read_write_index_1d
//  CHECK-SAME: %[[BASE:[a-zA-Z0-9]*]]: index) -> vector<17xindex>
//       CHECK: %[[SPLAT:.*]] = arith.constant dense<7> : vector<17xindex>
//       CHECK: %{{.*}} = builtin.unrealized_conversion_cast %[[SPLAT]] : vector<17xindex> to vector<17xi64>

//       CHECK: %[[loaded:.*]] = llvm.intr.masked.load %{{.*}}, %{{.*}}, %{{.*}} {alignment = 8 : i32} :
//  CHECK-SAME: (!llvm.ptr, vector<17xi1>, vector<17xi64>) -> vector<17xi64>

//       CHECK: llvm.intr.masked.store %[[loaded]], %{{.*}}, %{{.*}} {alignment = 8 : i32} :
//  CHECK-SAME: vector<17xi64>, vector<17xi1> into !llvm.ptr

func.func @transfer_read_write_index_1d_scalable(%A : memref<?xindex>, %base: index) -> vector<[17]xindex> {
  %f7 = arith.constant 7: index
  %f = vector.transfer_read %A[%base], %f7
      {permutation_map = affine_map<(d0) -> (d0)>} :
    memref<?xindex>, vector<[17]xindex>
  vector.transfer_write %f, %A[%base]
      {permutation_map = affine_map<(d0) -> (d0)>} :
    vector<[17]xindex>, memref<?xindex>
  return %f: vector<[17]xindex>
}
// CHECK-LABEL: func @transfer_read_write_index_1d
//  CHECK-SAME: %[[BASE:[a-zA-Z0-9]*]]: index) -> vector<[17]xindex>
//       CHECK: %[[SPLAT:.*]] = arith.constant dense<7> : vector<[17]xindex>
//       CHECK: %{{.*}} = builtin.unrealized_conversion_cast %[[SPLAT]] : vector<[17]xindex> to vector<[17]xi64>

//       CHECK: %[[loaded:.*]] = llvm.intr.masked.load %{{.*}}, %{{.*}}, %{{.*}} {alignment = 8 : i32} :
//  CHECK-SAME: (!llvm.ptr, vector<[17]xi1>, vector<[17]xi64>) -> vector<[17]xi64>

//       CHECK: llvm.intr.masked.store %[[loaded]], %{{.*}}, %{{.*}} {alignment = 8 : i32} :
//  CHECK-SAME: vector<[17]xi64>, vector<[17]xi1> into !llvm.ptr

// -----

func.func @transfer_read_2d_to_1d(%A : memref<?x?xf32>, %base0: index, %base1: index) -> vector<17xf32> {
  %f7 = arith.constant 7.0: f32
  %f = vector.transfer_read %A[%base0, %base1], %f7
      {permutation_map = affine_map<(d0, d1) -> (d1)>} :
    memref<?x?xf32>, vector<17xf32>
  return %f: vector<17xf32>
}
// CHECK-LABEL: func @transfer_read_2d_to_1d
//  CHECK-SAME: %[[BASE_0:[a-zA-Z0-9]*]]: index, %[[BASE_1:[a-zA-Z0-9]*]]: index) -> vector<17xf32>
//
// Create a vector with linear indices [ 0 .. vector_length - 1 ].
//   CHECK-DAG: %[[linearIndex:.*]] = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]> :
//  CHECK-SAME: vector<17x[[$IDX_TYPE]]>
//
//   CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
//       CHECK: %[[DIM:.*]] = memref.dim %{{.*}}, %[[c1]] : memref<?x?xf32>
//
// Compute the in-bound index (dim - offset)
//       CHECK: %[[BOUND:.*]] = arith.subi %[[DIM]], %[[BASE_1]] : index
//
// Create bound vector to compute in-bound mask:
//    [ 0 .. vector_length - 1 ] < [ dim - offset .. dim - offset ]
//       CHECK: %[[btrunc:.*]] = arith.index_cast %[[BOUND]] : index to [[$IDX_TYPE]]
//       CHECK: %[[boundVecInsert:.*]] = llvm.insertelement %[[btrunc]]
//       CHECK: %[[boundVect:.*]] = llvm.shufflevector %[[boundVecInsert]]
//       CHECK: %[[mask:.*]] = arith.cmpi sgt, %[[boundVect]], %[[linearIndex]]

func.func @transfer_read_2d_to_1d_scalable(%A : memref<?x?xf32>, %base0: index, %base1: index) -> vector<[17]xf32> {
  %f7 = arith.constant 7.0: f32
  %f = vector.transfer_read %A[%base0, %base1], %f7
      {permutation_map = affine_map<(d0, d1) -> (d1)>} :
    memref<?x?xf32>, vector<[17]xf32>
  return %f: vector<[17]xf32>
}
// CHECK-LABEL: func @transfer_read_2d_to_1d
//  CHECK-SAME: %[[BASE_0:[a-zA-Z0-9]*]]: index, %[[BASE_1:[a-zA-Z0-9]*]]: index) -> vector<[17]xf32>
//       CHECK: %[[c1:.*]] = arith.constant 1 : index
//       CHECK: %[[DIM:.*]] = memref.dim %{{.*}}, %[[c1]] : memref<?x?xf32>
//
// Compute the in-bound index (dim - offset)
//       CHECK: %[[BOUND:.*]] = arith.subi %[[DIM]], %[[BASE_1]] : index
//
// Create a vector with linear indices [ 0 .. vector_length - 1 ].
//       CHECK: %[[linearIndex:.*]] = llvm.intr.stepvector : vector<[17]x[[$IDX_TYPE]]>
//
// Create bound vector to compute in-bound mask:
//    [ 0 .. vector_length - 1 ] < [ dim - offset .. dim - offset ]
//       CHECK: %[[btrunc:.*]] = arith.index_cast %[[BOUND]] : index to [[$IDX_TYPE]]
//       CHECK: %[[boundVecInsert:.*]] = llvm.insertelement %[[btrunc]]
//       CHECK: %[[boundVect:.*]] = llvm.shufflevector %[[boundVecInsert]]
//       CHECK: %[[mask:.*]] = arith.cmpi slt, %[[linearIndex]], %[[boundVect]]

// -----

func.func @transfer_read_write_1d_non_zero_addrspace(%A : memref<?xf32, 3>, %base: index) -> vector<17xf32> {
  %f7 = arith.constant 7.0: f32
  %f = vector.transfer_read %A[%base], %f7
      {permutation_map = affine_map<(d0) -> (d0)>} :
    memref<?xf32, 3>, vector<17xf32>
  vector.transfer_write %f, %A[%base]
      {permutation_map = affine_map<(d0) -> (d0)>} :
    vector<17xf32>, memref<?xf32, 3>
  return %f: vector<17xf32>
}
// CHECK-LABEL: func @transfer_read_write_1d_non_zero_addrspace
//  CHECK-SAME: %[[BASE:[a-zA-Z0-9]*]]: index) -> vector<17xf32>
//
//       CHECK: %[[c0:.*]] = arith.constant 0 : index
//
// 1. Check address space for GEP is correct.
//       CHECK: %[[gep:.*]] = llvm.getelementptr {{.*}} :
//  CHECK-SAME: (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
//
// 2. Check address space of the memref is correct.
//       CHECK: %[[DIM:.*]] = memref.dim %{{.*}}, %[[c0]] : memref<?xf32, 3>
//
// 3. Check address space for GEP is correct.
//       CHECK: %[[gep_b:.*]] = llvm.getelementptr {{.*}} :
//  CHECK-SAME: (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32

func.func @transfer_read_write_1d_non_zero_addrspace_scalable(%A : memref<?xf32, 3>, %base: index) -> vector<[17]xf32> {
  %f7 = arith.constant 7.0: f32
  %f = vector.transfer_read %A[%base], %f7
      {permutation_map = affine_map<(d0) -> (d0)>} :
    memref<?xf32, 3>, vector<[17]xf32>
  vector.transfer_write %f, %A[%base]
      {permutation_map = affine_map<(d0) -> (d0)>} :
    vector<[17]xf32>, memref<?xf32, 3>
  return %f: vector<[17]xf32>
}
// CHECK-LABEL: func @transfer_read_write_1d_non_zero_addrspace_scalable
//  CHECK-SAME: %[[BASE:[a-zA-Z0-9]*]]: index) -> vector<[17]xf32>
//
//       CHECK: %[[c0:.*]] = arith.constant 0 : index
//
// 1. Check address space for GEP is correct.
//       CHECK: %[[gep:.*]] = llvm.getelementptr {{.*}} :
//  CHECK-SAME: (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
//
// 2. Check address space of the memref is correct.
//       CHECK: %[[DIM:.*]] = memref.dim %{{.*}}, %[[c0]] : memref<?xf32, 3>
//
// 3. Check address space for GEP is correct.
//       CHECK: %[[gep_b:.*]] = llvm.getelementptr {{.*}} :
//  CHECK-SAME: (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32

// -----

func.func @transfer_read_1d_inbounds(%A : memref<?xf32>, %base: index) -> vector<17xf32> {
  %f7 = arith.constant 7.0: f32
  %f = vector.transfer_read %A[%base], %f7 {in_bounds = [true]} :
    memref<?xf32>, vector<17xf32>
  return %f: vector<17xf32>
}
// CHECK-LABEL: func @transfer_read_1d_inbounds
//  CHECK-SAME: %[[BASE:[a-zA-Z0-9]*]]: index) -> vector<17xf32>
//
// 1. Bitcast to vector form.
//       CHECK: %[[gep:.*]] = llvm.getelementptr {{.*}} :
//  CHECK-SAME: (!llvm.ptr, i64) -> !llvm.ptr, f32
//
// 2. Rewrite as a load.
//       CHECK: %[[loaded:.*]] = llvm.load %[[gep]] {alignment = 4 : i64} : !llvm.ptr -> vector<17xf32>

func.func @transfer_read_1d_inbounds_scalable(%A : memref<?xf32>, %base: index) -> vector<[17]xf32> {
  %f7 = arith.constant 7.0: f32
  %f = vector.transfer_read %A[%base], %f7 {in_bounds = [true]} :
    memref<?xf32>, vector<[17]xf32>
  return %f: vector<[17]xf32>
}
// CHECK-LABEL: func @transfer_read_1d_inbounds_scalable
//  CHECK-SAME: %[[BASE:[a-zA-Z0-9]*]]: index) -> vector<[17]xf32>
//
// 1. Bitcast to vector form.
//       CHECK: %[[gep:.*]] = llvm.getelementptr {{.*}} :
//  CHECK-SAME: (!llvm.ptr, i64) -> !llvm.ptr, f32
//
// 2. Rewrite as a load.
//       CHECK: %[[loaded:.*]] = llvm.load %[[gep]] {alignment = 4 : i64} : !llvm.ptr -> vector<[17]xf32>

// -----

// CHECK-LABEL: func @transfer_read_write_1d_mask
// CHECK: %[[mask1:.*]] = arith.constant dense<[false, false, true, false, true]>
// CHECK: %[[cmpi:.*]] = arith.cmpi sgt
// CHECK: %[[mask2:.*]] = arith.andi %[[cmpi]], %[[mask1]]
// CHECK: %[[r:.*]] = llvm.intr.masked.load %{{.*}}, %[[mask2]]
// CHECK: %[[cmpi_1:.*]] = arith.cmpi sgt
// CHECK: %[[mask3:.*]] = arith.andi %[[cmpi_1]], %[[mask1]]
// CHECK: llvm.intr.masked.store %[[r]], %{{.*}}, %[[mask3]]
// CHECK: return %[[r]]
func.func @transfer_read_write_1d_mask(%A : memref<?xf32>, %base : index) -> vector<5xf32> {
  %m = arith.constant dense<[0, 0, 1, 0, 1]> : vector<5xi1>
  %f7 = arith.constant 7.0: f32
  %f = vector.transfer_read %A[%base], %f7, %m : memref<?xf32>, vector<5xf32>
  vector.transfer_write %f, %A[%base], %m : vector<5xf32>, memref<?xf32>
  return %f: vector<5xf32>
}

// CHECK-LABEL: func @transfer_read_write_1d_mask_scalable
// CHECK-SAME: %[[mask:[a-zA-Z0-9]*]]: vector<[5]xi1>
// CHECK: %[[cmpi:.*]] = arith.cmpi slt
// CHECK: %[[mask1:.*]] = arith.andi %[[cmpi]], %[[mask]]
// CHECK: %[[r:.*]] = llvm.intr.masked.load %{{.*}}, %[[mask1]]
// CHECK: %[[cmpi_1:.*]] = arith.cmpi slt
// CHECK: %[[mask2:.*]] = arith.andi %[[cmpi_1]], %[[mask]]
// CHECK: llvm.intr.masked.store %[[r]], %{{.*}}, %[[mask2]]
// CHECK: return %[[r]]
func.func @transfer_read_write_1d_mask_scalable(%A : memref<?xf32>, %base : index, %m : vector<[5]xi1>) -> vector<[5]xf32> {
  %f7 = arith.constant 7.0: f32
  %f = vector.transfer_read %A[%base], %f7, %m : memref<?xf32>, vector<[5]xf32>
  vector.transfer_write %f, %A[%base], %m : vector<[5]xf32>, memref<?xf32>
  return %f: vector<[5]xf32>
}

// -----

// Can't lower xfer_read/xfer_write on tensors, but this shouldn't crash

// CHECK-LABEL: func @transfer_read_write_tensor
//       CHECK:   vector.transfer_read
//       CHECK:   vector.transfer_write
func.func @transfer_read_write_tensor(%A: tensor<?xf32>, %base : index) -> vector<4xf32> {
  %f7 = arith.constant 7.0: f32
  %c0 = arith.constant 0: index
  %f = vector.transfer_read %A[%base], %f7 : tensor<?xf32>, vector<4xf32>
  %w = vector.transfer_write %f, %A[%c0] : vector<4xf32>, tensor<?xf32>
  "test.some_use"(%w) : (tensor<?xf32>) -> ()
  return %f : vector<4xf32>
}
