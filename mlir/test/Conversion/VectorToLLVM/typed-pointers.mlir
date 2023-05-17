// RUN: mlir-opt %s -convert-vector-to-llvm='use-opaque-pointers=0' -split-input-file | FileCheck %s

func.func @vector_type_cast(%arg0: memref<8x8x8xf32>) -> memref<vector<8x8x8xf32>> {
  %0 = vector.type_cast %arg0: memref<8x8x8xf32> to memref<vector<8x8x8xf32>>
  return %0 : memref<vector<8x8x8xf32>>
}
// CHECK-LABEL: @vector_type_cast
//       CHECK:   llvm.mlir.undef : !llvm.struct<(ptr<array<8 x array<8 x vector<8xf32>>>>, ptr<array<8 x array<8 x vector<8xf32>>>>, i64)>
//       CHECK:   %[[allocated:.*]] = llvm.extractvalue {{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:   %[[allocatedBit:.*]] = llvm.bitcast %[[allocated]] : !llvm.ptr<f32> to !llvm.ptr<array<8 x array<8 x vector<8xf32>>>>
//       CHECK:   llvm.insertvalue %[[allocatedBit]], {{.*}}[0] : !llvm.struct<(ptr<array<8 x array<8 x vector<8xf32>>>>, ptr<array<8 x array<8 x vector<8xf32>>>>, i64)>
//       CHECK:   %[[aligned:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:   %[[alignedBit:.*]] = llvm.bitcast %[[aligned]] : !llvm.ptr<f32> to !llvm.ptr<array<8 x array<8 x vector<8xf32>>>>
//       CHECK:   llvm.insertvalue %[[alignedBit]], {{.*}}[1] : !llvm.struct<(ptr<array<8 x array<8 x vector<8xf32>>>>, ptr<array<8 x array<8 x vector<8xf32>>>>, i64)>
//       CHECK:   llvm.mlir.constant(0 : index
//       CHECK:   llvm.insertvalue {{.*}}[2] : !llvm.struct<(ptr<array<8 x array<8 x vector<8xf32>>>>, ptr<array<8 x array<8 x vector<8xf32>>>>, i64)>

// -----

func.func @vector_type_cast_non_zero_addrspace(%arg0: memref<8x8x8xf32, 3>) -> memref<vector<8x8x8xf32>, 3> {
  %0 = vector.type_cast %arg0: memref<8x8x8xf32, 3> to memref<vector<8x8x8xf32>, 3>
  return %0 : memref<vector<8x8x8xf32>, 3>
}
// CHECK-LABEL: @vector_type_cast_non_zero_addrspace
//       CHECK:   llvm.mlir.undef : !llvm.struct<(ptr<array<8 x array<8 x vector<8xf32>>>, 3>, ptr<array<8 x array<8 x vector<8xf32>>>, 3>, i64)>
//       CHECK:   %[[allocated:.*]] = llvm.extractvalue {{.*}}[0] : !llvm.struct<(ptr<f32, 3>, ptr<f32, 3>, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:   %[[allocatedBit:.*]] = llvm.bitcast %[[allocated]] : !llvm.ptr<f32, 3> to !llvm.ptr<array<8 x array<8 x vector<8xf32>>>, 3>
//       CHECK:   llvm.insertvalue %[[allocatedBit]], {{.*}}[0] : !llvm.struct<(ptr<array<8 x array<8 x vector<8xf32>>>, 3>, ptr<array<8 x array<8 x vector<8xf32>>>, 3>, i64)>
//       CHECK:   %[[aligned:.*]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<f32, 3>, ptr<f32, 3>, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:   %[[alignedBit:.*]] = llvm.bitcast %[[aligned]] : !llvm.ptr<f32, 3> to !llvm.ptr<array<8 x array<8 x vector<8xf32>>>, 3>
//       CHECK:   llvm.insertvalue %[[alignedBit]], {{.*}}[1] : !llvm.struct<(ptr<array<8 x array<8 x vector<8xf32>>>, 3>, ptr<array<8 x array<8 x vector<8xf32>>>, 3>, i64)>
//       CHECK:   llvm.mlir.constant(0 : index
//       CHECK:   llvm.insertvalue {{.*}}[2] : !llvm.struct<(ptr<array<8 x array<8 x vector<8xf32>>>, 3>, ptr<array<8 x array<8 x vector<8xf32>>>, 3>, i64)>

// -----

func.func @transfer_read_1d(%A : memref<?xf32>, %base: index) -> vector<17xf32> {
  %f7 = arith.constant 7.0: f32
  %f = vector.transfer_read %A[%base], %f7
      {permutation_map = affine_map<(d0) -> (d0)>} :
    memref<?xf32>, vector<17xf32>
  vector.transfer_write %f, %A[%base]
      {permutation_map = affine_map<(d0) -> (d0)>} :
    vector<17xf32>, memref<?xf32>
  return %f: vector<17xf32>
}
// CHECK-LABEL: func @transfer_read_1d
//  CHECK-SAME: %[[MEM:.*]]: memref<?xf32>,
//  CHECK-SAME: %[[BASE:.*]]: index) -> vector<17xf32>
//       CHECK: %[[C7:.*]] = arith.constant 7.0
//
// 1. Let dim be the memref dimension, compute the in-bound index (dim - offset)
//       CHECK: %[[C0:.*]] = arith.constant 0 : index
//       CHECK: %[[DIM:.*]] = memref.dim %[[MEM]], %[[C0]] : memref<?xf32>
//       CHECK: %[[BOUND:.*]] = arith.subi %[[DIM]],  %[[BASE]] : index
//
// 2. Create a vector with linear indices [ 0 .. vector_length - 1 ].
//       CHECK: %[[linearIndex:.*]] = arith.constant dense
//  CHECK-SAME: <[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]> :
//  CHECK-SAME: vector<17xi32>
//
// 3. Create bound vector to compute in-bound mask:
//    [ 0 .. vector_length - 1 ] < [ dim - offset .. dim - offset ]
//       CHECK: %[[btrunc:.*]] = arith.index_cast %[[BOUND]] : index to i32
//       CHECK: %[[boundVecInsert:.*]] = llvm.insertelement %[[btrunc]]
//       CHECK: %[[boundVect:.*]] = llvm.shufflevector %[[boundVecInsert]]
//       CHECK: %[[mask:.*]] = arith.cmpi slt, %[[linearIndex]], %[[boundVect]]
//  CHECK-SAME: : vector<17xi32>
//
// 4. Create pass-through vector.
//       CHECK: %[[PASS_THROUGH:.*]] = arith.constant dense<7.{{.*}}> : vector<17xf32>
//
// 5. Bitcast to vector form.
//       CHECK: %[[gep:.*]] = llvm.getelementptr %{{.*}} :
//  CHECK-SAME: (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
//       CHECK: %[[vecPtr:.*]] = llvm.bitcast %[[gep]] :
//  CHECK-SAME: !llvm.ptr<f32> to !llvm.ptr<vector<17xf32>>
//
// 6. Rewrite as a masked read.
//       CHECK: %[[loaded:.*]] = llvm.intr.masked.load %[[vecPtr]], %[[mask]],
//  CHECK-SAME: %[[PASS_THROUGH]] {alignment = 4 : i32} :
//
// 1. Let dim be the memref dimension, compute the in-bound index (dim - offset)
//       CHECK: %[[C0_b:.*]] = arith.constant 0 : index
//       CHECK: %[[DIM_b:.*]] = memref.dim %[[MEM]], %[[C0_b]] : memref<?xf32>
//       CHECK: %[[BOUND_b:.*]] = arith.subi %[[DIM_b]], %[[BASE]] : index
//
// 2. Create a vector with linear indices [ 0 .. vector_length - 1 ].
//       CHECK: %[[linearIndex_b:.*]] = arith.constant dense
//  CHECK-SAME: <[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]> :
//  CHECK-SAME: vector<17xi32>
//
// 3. Create bound vector to compute in-bound mask:
//    [ 0 .. vector_length - 1 ] < [ dim - offset .. dim - offset ]
//       CHECK: %[[btrunc_b:.*]] = arith.index_cast %[[BOUND_b]] : index to i32
//       CHECK: %[[boundVecInsert_b:.*]] = llvm.insertelement %[[btrunc_b]]
//       CHECK: %[[boundVect_b:.*]] = llvm.shufflevector %[[boundVecInsert_b]]
//       CHECK: %[[mask_b:.*]] = arith.cmpi slt, %[[linearIndex_b]],
//  CHECK-SAME: %[[boundVect_b]] : vector<17xi32>
//
// 4. Bitcast to vector form.
//       CHECK: %[[gep_b:.*]] = llvm.getelementptr {{.*}} :
//  CHECK-SAME: (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
//       CHECK: %[[vecPtr_b:.*]] = llvm.bitcast %[[gep_b]] :
//  CHECK-SAME: !llvm.ptr<f32> to !llvm.ptr<vector<17xf32>>
//
// 5. Rewrite as a masked write.
//       CHECK: llvm.intr.masked.store %[[loaded]], %[[vecPtr_b]], %[[mask_b]]
//  CHECK-SAME: {alignment = 4 : i32} :
//  CHECK-SAME: vector<17xf32>, vector<17xi1> into !llvm.ptr<vector<17xf32>>

// -----

func.func @vector_load_op(%memref : memref<200x100xf32>, %i : index, %j : index) -> vector<8xf32> {
  %0 = vector.load %memref[%i, %j] : memref<200x100xf32>, vector<8xf32>
  return %0 : vector<8xf32>
}

// CHECK-LABEL: func @vector_load_op
// CHECK: %[[c100:.*]] = llvm.mlir.constant(100 : index) : i64
// CHECK: %[[mul:.*]] = llvm.mul %{{.*}}, %[[c100]]  : i64
// CHECK: %[[add:.*]] = llvm.add %[[mul]], %{{.*}}  : i64
// CHECK: %[[gep:.*]] = llvm.getelementptr %{{.*}}[%[[add]]] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
// CHECK: %[[bcast:.*]] = llvm.bitcast %[[gep]] : !llvm.ptr<f32> to !llvm.ptr<vector<8xf32>>
// CHECK: llvm.load %[[bcast]] {alignment = 4 : i64} : !llvm.ptr<vector<8xf32>>

// -----

func.func @vector_store_op(%memref : memref<200x100xf32>, %i : index, %j : index) {
  %val = arith.constant dense<11.0> : vector<4xf32>
  vector.store %val, %memref[%i, %j] : memref<200x100xf32>, vector<4xf32>
  return
}

// CHECK-LABEL: func @vector_store_op
// CHECK: %[[c100:.*]] = llvm.mlir.constant(100 : index) : i64
// CHECK: %[[mul:.*]] = llvm.mul %{{.*}}, %[[c100]]  : i64
// CHECK: %[[add:.*]] = llvm.add %[[mul]], %{{.*}}  : i64
// CHECK: %[[gep:.*]] = llvm.getelementptr %{{.*}}[%[[add]]] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
// CHECK: %[[bcast:.*]] = llvm.bitcast %[[gep]] : !llvm.ptr<f32> to !llvm.ptr<vector<4xf32>>
// CHECK: llvm.store %{{.*}}, %[[bcast]] {alignment = 4 : i64} : !llvm.ptr<vector<4xf32>>

// -----

func.func @masked_load_op(%arg0: memref<?xf32>, %arg1: vector<16xi1>, %arg2: vector<16xf32>) -> vector<16xf32> {
  %c0 = arith.constant 0: index
  %0 = vector.maskedload %arg0[%c0], %arg1, %arg2 : memref<?xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
  return %0 : vector<16xf32>
}

// CHECK-LABEL: func @masked_load_op
// CHECK: %[[CO:.*]] = arith.constant 0 : index
// CHECK: %[[C:.*]] = builtin.unrealized_conversion_cast %[[CO]] : index to i64
// CHECK: %[[P:.*]] = llvm.getelementptr %{{.*}}[%[[C]]] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
// CHECK: %[[B:.*]] = llvm.bitcast %[[P]] : !llvm.ptr<f32> to !llvm.ptr<vector<16xf32>>
// CHECK: %[[L:.*]] = llvm.intr.masked.load %[[B]], %{{.*}}, %{{.*}} {alignment = 4 : i32} : (!llvm.ptr<vector<16xf32>>, vector<16xi1>, vector<16xf32>) -> vector<16xf32>
// CHECK: return %[[L]] : vector<16xf32>

// -----

func.func @masked_store_op(%arg0: memref<?xf32>, %arg1: vector<16xi1>, %arg2: vector<16xf32>) {
  %c0 = arith.constant 0: index
  vector.maskedstore %arg0[%c0], %arg1, %arg2 : memref<?xf32>, vector<16xi1>, vector<16xf32>
  return
}

// CHECK-LABEL: func @masked_store_op
// CHECK: %[[CO:.*]] = arith.constant 0 : index
// CHECK: %[[C:.*]] = builtin.unrealized_conversion_cast %[[CO]] : index to i64
// CHECK: %[[P:.*]] = llvm.getelementptr %{{.*}}[%[[C]]] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
// CHECK: %[[B:.*]] = llvm.bitcast %[[P]] : !llvm.ptr<f32> to !llvm.ptr<vector<16xf32>>
// CHECK: llvm.intr.masked.store %{{.*}}, %[[B]], %{{.*}} {alignment = 4 : i32} : vector<16xf32>, vector<16xi1> into !llvm.ptr<vector<16xf32>>
