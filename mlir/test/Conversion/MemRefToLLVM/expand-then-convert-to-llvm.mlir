// RUN: mlir-opt -expand-strided-metadata -convert-memref-to-llvm -lower-affine -convert-arith-to-llvm -cse %s -split-input-file | FileCheck %s
//
// This test demonstrates a full "memref to llvm" pipeline where
// we first expand some of the memref operations (using affine,
// arith, and memref operations) and convert each individual
// dialect to llvm.
//
// Note: We run CSE in that test to get rid of duplicated
// unrealized_conversion_cast that are inserted with
// convert-memref-to-llvm and then convert-arith-to-llvm.
// The final code is still not perfect, because we have
// noop unrealized_conversion_cast from i64 to index
// and back.

// -----

// Final offset = baseOffset + subOffset0 * baseStride0 + subOffset1 * baseStride1
//              = 0 + %arg0 * 4 + %arg1 * 1
//              = %arg0 * 4 + %arg1
// Because of how subviews are lowered (i.e., using
// reintpret_cast(extract_strided_metadata(subSrc))),
// we end up with two casts of the base.
// More specifically, when we lower this sequence:
// ```
// base, ... = extract_strided_metadata subSrc
// dst = reinterpret_cast base, ...
// ```
//
// extract_strided_metadata gets lowered into:
// ```
// castedSrc = unrealized_conversion_cast subSrc
// base = extractvalue castedSrc[0]
// ```
//
// And reinterpret_cast gets lowered into:
// ```
// castedBase = unrealized_conversion_cast base
// dst = extractvalue %castedBase[0]
// ```
//
// Which give us:
// ```
// castedSrc = unrealized_conversion_cast src
// base = extractvalue castedSrc[0]
// castedBase = unrealized_conversion_cast base
// dst = extractvalue %castedBase[0] <-- dst and base are effectively equal.
// ```
// CHECK-LABEL: func @subview(
// CHECK:         %[[MEM:.*]]: memref<{{.*}}>,
// CHECK:         %[[ARG0f:[a-zA-Z0-9]*]]: index,
// CHECK:         %[[ARG1f:[a-zA-Z0-9]*]]: index,
// CHECK:         %[[ARG2f:.*]]: index)
func.func @subview(%0 : memref<64x4xf32, strided<[4, 1], offset: 0>>, %arg0 : index, %arg1 : index, %arg2 : index)
-> memref<?x?xf32, strided<[?, ?], offset: ?>> {
  // CHECK-DAG: %[[MEMREF:.*]] = builtin.unrealized_conversion_cast %[[MEM]]
  // CHECK-DAG: %[[ARG0:.*]] = builtin.unrealized_conversion_cast %[[ARG0f]]
  // CHECK-DAG: %[[ARG1:.*]] = builtin.unrealized_conversion_cast %[[ARG1f]]

  // CHECK: %[[BASE:.*]] = llvm.extractvalue %[[MEMREF]][0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64
  // CHECK: %[[BASE_ALIGNED:.*]] = llvm.extractvalue %[[MEMREF]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64
  // CHECK: %[[STRIDE0:.*]] = llvm.mlir.constant(4 : index) : i64
  // CHECK: %[[DESCSTRIDE0:.*]] = llvm.mul %[[ARG0]], %[[STRIDE0]] : i64
  // CHECK: %[[TMP:.*]] = builtin.unrealized_conversion_cast %[[DESCSTRIDE0]] : i64 to index
  // CHECK: %[[DESCSTRIDE0_V2:.*]] = builtin.unrealized_conversion_cast %[[TMP]] : index to i64
  // CHECK: %[[OFF2:.*]] = llvm.add %[[DESCSTRIDE0]], %[[ARG1]] : i64
  // CHECK: %[[TMP:.*]] = builtin.unrealized_conversion_cast %[[OFF2]] : i64 to index
  // CHECK: %[[OFF2:.*]] = builtin.unrealized_conversion_cast %[[TMP]] : index to i64
  // CHECK: %[[DESC:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[DESC0:.*]] = llvm.insertvalue %[[BASE]], %[[DESC]][0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[DESC1:.*]] = llvm.insertvalue %[[BASE_ALIGNED]], %[[DESC0]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[DESC2:.*]] = llvm.insertvalue %[[OFF2]], %[[DESC1]][2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[DESC3:.*]] = llvm.insertvalue %[[ARG0]], %[[DESC2]][3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[DESC4:.*]] = llvm.insertvalue %[[DESCSTRIDE0_V2]], %[[DESC3]][4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[DESC5:.*]] = llvm.insertvalue %[[ARG1]], %[[DESC4]][3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[DESC6:.*]] = llvm.insertvalue %[[ARG1]], %[[DESC5]][4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>

  %1 = memref.subview %0[%arg0, %arg1][%arg0, %arg1][%arg0, %arg1] :
    memref<64x4xf32, strided<[4, 1], offset: 0>>
  to memref<?x?xf32, strided<[?, ?], offset: ?>>
  return %1 : memref<?x?xf32, strided<[?, ?], offset: ?>>
}

// -----

// CHECK-LABEL: func @subview_non_zero_addrspace(
// CHECK:         %[[MEM:[a-zA-Z0-9]*]]: memref<{{[^%]*}}>,
// CHECK:         %[[ARG0f:[a-zA-Z0-9]*]]: index,
// CHECK:         %[[ARG1f:[a-zA-Z0-9]*]]: index,
// CHECK:         %[[ARG2f:.*]]: index)
func.func @subview_non_zero_addrspace(%0 : memref<64x4xf32, strided<[4, 1], offset: 0>, 3>, %arg0 : index, %arg1 : index, %arg2 : index) -> memref<?x?xf32, strided<[?, ?], offset: ?>, 3> {
  // CHECK-DAG: %[[MEMREF:.*]] = builtin.unrealized_conversion_cast %[[MEM]]
  // CHECK-DAG: %[[ARG0:.*]] = builtin.unrealized_conversion_cast %[[ARG0f]]
  // CHECK-DAG: %[[ARG1:.*]] = builtin.unrealized_conversion_cast %[[ARG1f]]

  // CHECK: %[[BASE:.*]] = llvm.extractvalue %[[MEMREF]][0] : !llvm.struct<(ptr<f32, 3>, ptr<f32, 3>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[BASE_ALIGNED:.*]] = llvm.extractvalue %[[MEMREF]][1] : !llvm.struct<(ptr<f32, 3>, ptr<f32, 3>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[STRIDE0:.*]] = llvm.mlir.constant(4 : index) : i64
  // CHECK: %[[DESCSTRIDE0:.*]] = llvm.mul %[[ARG0]], %[[STRIDE0]]  : i64
  // CHECK: %[[TMP:.*]] = builtin.unrealized_conversion_cast %[[DESCSTRIDE0]] : i64 to index
  // CHECK: %[[DESCSTRIDE0_V2:.*]] = builtin.unrealized_conversion_cast %[[TMP]] : index to i64
  // CHECK: %[[OFF2:.*]] = llvm.add %[[DESCSTRIDE0]], %[[ARG1]]  : i64
  // CHECK: %[[TMP:.*]] = builtin.unrealized_conversion_cast %[[OFF2]] : i64 to index
  // CHECK: %[[OFF2:.*]] = builtin.unrealized_conversion_cast %[[TMP]] : index to i64
  // CHECK: %[[DESC:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<f32, 3>, ptr<f32, 3>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[DESC0:.*]] = llvm.insertvalue %[[BASE]], %[[DESC]][0] : !llvm.struct<(ptr<f32, 3>, ptr<f32, 3>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[DESC1:.*]] = llvm.insertvalue %[[BASE_ALIGNED]], %[[DESC0]][1] : !llvm.struct<(ptr<f32, 3>, ptr<f32, 3>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[DESC2:.*]] = llvm.insertvalue %[[OFF2]], %[[DESC1]][2] : !llvm.struct<(ptr<f32, 3>, ptr<f32, 3>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[DESC3:.*]] = llvm.insertvalue %[[ARG0]], %[[DESC2]][3, 0] : !llvm.struct<(ptr<f32, 3>, ptr<f32, 3>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[DESC4:.*]] = llvm.insertvalue %[[DESCSTRIDE0_V2]], %[[DESC3]][4, 0] : !llvm.struct<(ptr<f32, 3>, ptr<f32, 3>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[DESC5:.*]] = llvm.insertvalue %[[ARG1]], %[[DESC4]][3, 1] : !llvm.struct<(ptr<f32, 3>, ptr<f32, 3>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[DESC6:.*]] = llvm.insertvalue %[[ARG1]], %[[DESC5]][4, 1] : !llvm.struct<(ptr<f32, 3>, ptr<f32, 3>, i64, array<2 x i64>, array<2 x i64>)>

  %1 = memref.subview %0[%arg0, %arg1][%arg0, %arg1][%arg0, %arg1] :
    memref<64x4xf32, strided<[4, 1], offset: 0>, 3>
    to memref<?x?xf32, strided<[?, ?], offset: ?>, 3>
  return %1 : memref<?x?xf32, strided<[?, ?], offset: ?>, 3>
}

// -----

// CHECK-LABEL: func @subview_const_size(
// CHECK-SAME:         %[[MEM:[a-zA-Z0-9]*]]: memref<{{[^%]*}}>,
// CHECK-SAME:         %[[ARG0f:[a-zA-Z0-9]*]]: index
// CHECK-SAME:         %[[ARG1f:[a-zA-Z0-9]*]]: index
// CHECK-SAME:         %[[ARG2f:[a-zA-Z0-9]*]]: index
func.func @subview_const_size(%0 : memref<64x4xf32, strided<[4, 1], offset: 0>>, %arg0 : index, %arg1 : index, %arg2 : index) -> memref<4x2xf32, strided<[?, ?], offset: ?>> {
  // CHECK-DAG: %[[MEMREF:.*]] = builtin.unrealized_conversion_cast %[[MEM]]
  // CHECK-DAG: %[[ARG0:.*]] = builtin.unrealized_conversion_cast %[[ARG0f]]
  // CHECK-DAG: %[[ARG1:.*]] = builtin.unrealized_conversion_cast %[[ARG1f]]

  // CHECK: %[[BASE:.*]] = llvm.extractvalue %[[MEMREF]][0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[BASE_ALIGNED:.*]] = llvm.extractvalue %[[MEMREF]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[C4:.*]] = llvm.mlir.constant(4 : index) : i64
  // CHECK: %[[DESCSTRIDE0:.*]] = llvm.mul %[[ARG0]], %[[C4]]  : i64
  // CHECK: %[[TMP:.*]] = builtin.unrealized_conversion_cast %[[DESCSTRIDE0]] : i64 to index
  // CHECK: %[[DESCSTRIDE0_V2:.*]] = builtin.unrealized_conversion_cast %[[TMP]] : index to i64
  // CHECK: %[[OFF2:.*]] = llvm.add %[[DESCSTRIDE0]], %[[ARG1]]  : i64
  // CHECK: %[[TMP:.*]] = builtin.unrealized_conversion_cast %[[OFF2]] : i64 to index
  // CHECK: %[[OFF2:.*]] = builtin.unrealized_conversion_cast %[[TMP]] : index to i64
  // CHECK: %[[DESC:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[DESC0:.*]] = llvm.insertvalue %[[BASE]], %[[DESC]][0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[DESC1:.*]] = llvm.insertvalue %[[BASE_ALIGNED]], %[[DESC0]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[DESC2:.*]] = llvm.insertvalue %[[OFF2]], %[[DESC1]][2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[DESC3:.*]] = llvm.insertvalue %[[C4]], %[[DESC2]][3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[DESC4:.*]] = llvm.insertvalue %[[DESCSTRIDE0_V2]], %[[DESC3]][4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[CST_SIZE1:.*]] = llvm.mlir.constant(2 : index) : i64
  // CHECK: %[[DESC5:.*]] = llvm.insertvalue %[[CST_SIZE1]], %[[DESC4]][3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[DESC6:.*]] = llvm.insertvalue %[[ARG1]], %[[DESC5]][4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>

  %1 = memref.subview %0[%arg0, %arg1][4, 2][%arg0, %arg1] :
    memref<64x4xf32, strided<[4, 1], offset: 0>>
    to memref<4x2xf32, strided<[?, ?], offset: ?>>
  return %1 : memref<4x2xf32, strided<[?, ?], offset: ?>>
}

// -----

// CHECK-LABEL: func @subview_const_stride(
// CHECK-SAME:         %[[MEM:[a-zA-Z0-9]*]]: memref<{{[^%]*}}>,
// CHECK-SAME:         %[[ARG0f:[a-zA-Z0-9]*]]: index
// CHECK-SAME:         %[[ARG1f:[a-zA-Z0-9]*]]: index
// CHECK-SAME:         %[[ARG2f:[a-zA-Z0-9]*]]: index
func.func @subview_const_stride(%0 : memref<64x4xf32, strided<[4, 1], offset: 0>>, %arg0 : index, %arg1 : index, %arg2 : index) -> memref<?x?xf32, strided<[4, 2], offset: ?>> {
  // CHECK-DAG: %[[MEMREF:.*]] = builtin.unrealized_conversion_cast %[[MEM]]
  // CHECK-DAG: %[[ARG0:.*]] = builtin.unrealized_conversion_cast %[[ARG0f]]
  // CHECK-DAG: %[[ARG1:.*]] = builtin.unrealized_conversion_cast %[[ARG1f]]

  // CHECK: %[[BASE:.*]] = llvm.extractvalue %[[MEMREF]][0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[BASE_ALIGNED:.*]] = llvm.extractvalue %[[MEMREF]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[C4:.*]] = llvm.mlir.constant(4 : index) : i64
  // CHECK: %[[OFF0:.*]] = llvm.mul %[[ARG0]], %[[C4]]  : i64
  // CHECK: %[[OFF2:.*]] = llvm.add %[[OFF0]], %[[ARG1]]  : i64
  // CHECK: %[[TMP:.*]] = builtin.unrealized_conversion_cast %[[OFF2]] : i64 to index
  // CHECK: %[[OFF2:.*]] = builtin.unrealized_conversion_cast %[[TMP]] : index to i64
  // CHECK: %[[DESC:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[DESC0:.*]] = llvm.insertvalue %[[BASE]], %[[DESC]][0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[DESC1:.*]] = llvm.insertvalue %[[BASE_ALIGNED]], %[[DESC0]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[DESC2:.*]] = llvm.insertvalue %[[OFF2]], %[[DESC1]][2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[DESC3:.*]] = llvm.insertvalue %[[ARG0]], %[[DESC2]][3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[DESC4:.*]] = llvm.insertvalue %[[C4]], %[[DESC3]][4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[DESC5:.*]] = llvm.insertvalue %[[ARG1]], %[[DESC4]][3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[CST_STRIDE1:.*]] = llvm.mlir.constant(2 : index) : i64
  // CHECK: %[[DESC6:.*]] = llvm.insertvalue %[[CST_STRIDE1]], %[[DESC5]][4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>

  %1 = memref.subview %0[%arg0, %arg1][%arg0, %arg1][1, 2] :
    memref<64x4xf32, strided<[4, 1], offset: 0>>
    to memref<?x?xf32, strided<[4, 2], offset: ?>>
  return %1 : memref<?x?xf32, strided<[4, 2], offset: ?>>
}

// -----

// CHECK-LABEL: func @subview_const_stride_and_offset(
// CHECK-SAME:         %[[MEM:.*]]: memref<{{.*}}>
func.func @subview_const_stride_and_offset(%0 : memref<64x4xf32, strided<[4, 1], offset: 0>>) -> memref<62x3xf32, strided<[4, 1], offset: 8>> {
  // The last "insertvalue" that populates the memref descriptor from the function arguments.
  // CHECK: %[[MEMREF:.*]] = builtin.unrealized_conversion_cast %[[MEM]]

  // CHECK: %[[BASE:.*]] = llvm.extractvalue %[[MEMREF]][0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[BASE_ALIGNED:.*]] = llvm.extractvalue %[[MEMREF]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[DESC:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[DESC0:.*]] = llvm.insertvalue %[[BASE]], %[[DESC]][0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[DESC1:.*]] = llvm.insertvalue %[[BASE_ALIGNED]], %[[DESC0]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[CST_OFF:.*]] = llvm.mlir.constant(8 : index) : i64
  // CHECK: %[[DESC2:.*]] = llvm.insertvalue %[[CST_OFF]], %[[DESC1]][2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[CST_SIZE0:.*]] = llvm.mlir.constant(62 : index) : i64
  // CHECK: %[[DESC3:.*]] = llvm.insertvalue %[[CST_SIZE0]], %[[DESC2]][3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[CST_STRIDE0:.*]] = llvm.mlir.constant(4 : index) : i64
  // CHECK: %[[DESC4:.*]] = llvm.insertvalue %[[CST_STRIDE0]], %[[DESC3]][4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[CST_SIZE1:.*]] = llvm.mlir.constant(3 : index) : i64
  // CHECK: %[[DESC5:.*]] = llvm.insertvalue %[[CST_SIZE1]], %[[DESC4]][3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[CST_STRIDE1:.*]] = llvm.mlir.constant(1 : index) : i64
  // CHECK: %[[DESC6:.*]] = llvm.insertvalue %[[CST_STRIDE1]], %[[DESC5]][4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>

  %1 = memref.subview %0[0, 8][62, 3][1, 1] :
    memref<64x4xf32, strided<[4, 1], offset: 0>>
    to memref<62x3xf32, strided<[4, 1], offset: 8>>
  return %1 : memref<62x3xf32, strided<[4, 1], offset: 8>>
}

// -----

// CHECK-LABEL: func @subview_mixed_static_dynamic(
// CHECK:         %[[MEM:[a-zA-Z0-9]*]]: memref<{{[^%]*}}>,
// CHECK:         %[[ARG0f:[a-zA-Z0-9]*]]: index,
// CHECK:         %[[ARG1f:[a-zA-Z0-9]*]]: index,
// CHECK:         %[[ARG2f:.*]]: index)
func.func @subview_mixed_static_dynamic(%0 : memref<64x4xf32, strided<[4, 1], offset: 0>>, %arg0 : index, %arg1 : index, %arg2 : index) -> memref<62x?xf32, strided<[?, 1], offset: ?>> {
  // CHECK-DAG: %[[MEMREF:.*]] = builtin.unrealized_conversion_cast %[[MEM]]
  // CHECK-DAG: %[[ARG0:.*]] = builtin.unrealized_conversion_cast %[[ARG0f]]
  // CHECK-DAG: %[[ARG1:.*]] = builtin.unrealized_conversion_cast %[[ARG1f]]
  // CHECK-DAG: %[[ARG2:.*]] = builtin.unrealized_conversion_cast %[[ARG2f]]

  // CHECK: %[[BASE:.*]] = llvm.extractvalue %[[MEMREF]][0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[BASE_ALIGNED:.*]] = llvm.extractvalue %[[MEMREF]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[STRIDE0:.*]] = llvm.mlir.constant(4 : index) : i64
  // CHECK: %[[DESCSTRIDE0:.*]] = llvm.mul %[[ARG0]], %[[STRIDE0]]  : i64
  // CHECK: %[[TMP:.*]] = builtin.unrealized_conversion_cast %[[DESCSTRIDE0]] : i64 to index
  // CHECK: %[[DESCSTRIDE0_V2:.*]] = builtin.unrealized_conversion_cast %[[TMP]] : index to i64
  // CHECK: %[[OFF0:.*]] = llvm.mul %[[ARG1]], %[[STRIDE0]]  : i64
  // CHECK: %[[BASE_OFF:.*]] = llvm.mlir.constant(8 : index)  : i64
  // CHECK: %[[OFF2:.*]] = llvm.add %[[OFF0]], %[[BASE_OFF]]  : i64
  // CHECK: %[[TMP:.*]] = builtin.unrealized_conversion_cast %[[OFF2]] : i64 to index
  // CHECK: %[[OFF2:.*]] = builtin.unrealized_conversion_cast %[[TMP]] : index to i64
  // CHECK: %[[DESC:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[DESC0:.*]] = llvm.insertvalue %[[BASE]], %[[DESC]][0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[DESC1:.*]] = llvm.insertvalue %[[BASE_ALIGNED]], %[[DESC0]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[DESC2:.*]] = llvm.insertvalue %[[OFF2]], %[[DESC1]][2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[CST_SIZE0:.*]] = llvm.mlir.constant(62 : index)  : i64
  // CHECK: %[[DESC3:.*]] = llvm.insertvalue %[[CST_SIZE0]], %[[DESC2]][3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[DESC4:.*]] = llvm.insertvalue %[[DESCSTRIDE0_V2]], %[[DESC3]][4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[DESC5:.*]] = llvm.insertvalue %[[ARG2]], %[[DESC4]][3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[CST_STRIDE1:.*]] = llvm.mlir.constant(1 : index) : i64
  // CHECK: %[[DESC6:.*]] = llvm.insertvalue %[[CST_STRIDE1]], %[[DESC5]][4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>

  %1 = memref.subview %0[%arg1, 8][62, %arg2][%arg0, 1] :
    memref<64x4xf32, strided<[4, 1], offset: 0>>
    to memref<62x?xf32, strided<[?, 1], offset: ?>>
  return %1 : memref<62x?xf32, strided<[?, 1], offset: ?>>
}

// -----

// CHECK-LABEL: func @subview_leading_operands(
// CHECK:         %[[MEM:.*]]: memref<{{.*}}>,
func.func @subview_leading_operands(%0 : memref<5x3xf32>, %1: memref<5x?xf32>) -> memref<3x3xf32, strided<[3, 1], offset: 6>> {
  // CHECK: %[[MEMREF:.*]] = builtin.unrealized_conversion_cast %[[MEM]]
  // Alloc ptr
  // CHECK: %[[BASE:.*]] = llvm.extractvalue %[[MEMREF]][0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64
  // Aligned ptr
  // CHECK: %[[BASE_ALIGNED:.*]] = llvm.extractvalue %[[MEMREF]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64
  // CHECK: %[[DESC:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[DESC0:.*]] = llvm.insertvalue %[[BASE]], %[[DESC]][0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[DESC1:.*]] = llvm.insertvalue %[[BASE_ALIGNED]], %[[DESC0]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // Offset
  // CHECK: %[[CST_OFF:.*]] = llvm.mlir.constant(6 : index) : i64
  // CHECK: %[[DESC2:.*]] = llvm.insertvalue %[[CST_OFF]], %[[DESC1]][2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // Sizes and strides @rank 0: both static extracted from type.
  // CHECK: %[[C3:.*]] = llvm.mlir.constant(3 : index) : i64
  // CHECK: %[[DESC3:.*]] = llvm.insertvalue %[[C3]], %[[DESC2]][3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[DESC4:.*]] = llvm.insertvalue %[[C3]], %[[DESC3]][4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // Sizes and strides @rank 1: both static.
  // CHECK: %[[DESC5:.*]] = llvm.insertvalue %[[C3]], %[[DESC4]][3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[CST_STRIDE1:.*]] = llvm.mlir.constant(1 : index) : i64
  // CHECK: %[[DESC6:.*]] = llvm.insertvalue %[[CST_STRIDE1]], %[[DESC5]][4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  %2 = memref.subview %0[2, 0][3, 3][1, 1]: memref<5x3xf32> to memref<3x3xf32, strided<[3, 1], offset: 6>>

  return %2 : memref<3x3xf32, strided<[3, 1], offset: 6>>
}

// -----

// CHECK-LABEL: func @subview_leading_operands_dynamic(
// CHECK:         %[[MEM:[a-zA-Z0-9]*]]: memref
func.func @subview_leading_operands_dynamic(%0 : memref<5x?xf32>) -> memref<3x?xf32, strided<[?, 1], offset: ?>> {
  // CHECK: %[[MEMREF:.*]] = builtin.unrealized_conversion_cast %[[MEM]]
  // CHECK: %[[SIZE1:.*]] = llvm.extractvalue %[[MEMREF]][3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[BASE:.*]] = llvm.extractvalue %[[MEMREF]][0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64
  // CHECK: %[[BASE_ALIGNED:.*]] = llvm.extractvalue %[[MEMREF]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64
  // Extract strides
  // CHECK: %[[STRIDE0:.*]] = llvm.extractvalue %[[MEMREF]][4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // Compute and insert offset from 2 + dynamic value.
  // CHECK: %[[CST_OFF0:.*]] = llvm.mlir.constant(2 : index) : i64
  // CHECK: %[[OFF0:.*]] = llvm.mul %[[STRIDE0]], %[[CST_OFF0]] : i64
  // CHECK: %[[TMP:.*]] = builtin.unrealized_conversion_cast %[[OFF0]] : i64 to index
  // CHECK: %[[OFF0:.*]] = builtin.unrealized_conversion_cast %[[TMP]] : index to i64
  // CHECK: %[[DESC:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // Alloc ptr
  // CHECK: %[[DESC0:.*]] = llvm.insertvalue %[[BASE]], %[[DESC]][0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // Aligned ptr
  // CHECK: %[[DESC1:.*]] = llvm.insertvalue %[[BASE_ALIGNED]], %[[DESC0]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[DESC2:.*]] = llvm.insertvalue %[[OFF0]], %[[DESC1]][2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // Sizes and strides @rank 0: both static.
  // CHECK: %[[CST_SIZE0:.*]] = llvm.mlir.constant(3 : index) : i64
  // CHECK: %[[DESC3:.*]] = llvm.insertvalue %[[CST_SIZE0]], %[[DESC2]][3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[DESC4:.*]] = llvm.insertvalue %[[STRIDE0]], %[[DESC3]][4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // Sizes and strides @rank 1: static stride 1, dynamic size unchanged from source memref.
  // CHECK: %[[DESC5:.*]] = llvm.insertvalue %[[SIZE1]], %[[DESC4]][3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[CST_STRIDE1:.*]] = llvm.mlir.constant(1 : index) : i64
  // CHECK: %[[DESC6:.*]] = llvm.insertvalue %[[CST_STRIDE1]], %[[DESC5]][4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>

  %c0 = arith.constant 1 : index
  %d0 = memref.dim %0, %c0 : memref<5x?xf32>
  %1 = memref.subview %0[2, 0][3, %d0][1, 1]: memref<5x?xf32> to memref<3x?xf32, strided<[?, 1], offset: ?>>
  return %1 : memref<3x?xf32, strided<[?, 1], offset: ?>>
}

// -----

// CHECK-LABEL: func @subview_rank_reducing_leading_operands(
// CHECK:         %[[MEM:.*]]: memref
func.func @subview_rank_reducing_leading_operands(%0 : memref<5x3xf32>) -> memref<3xf32, strided<[1], offset: 3>> {
  // CHECK: %[[MEMREF:.*]] = builtin.unrealized_conversion_cast %[[MEM]]
  // CHECK: %[[BASE:.*]] = llvm.extractvalue %[[MEMREF]][0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64
  // CHECK: %[[BASE_ALIGNED:.*]] = llvm.extractvalue %[[MEMREF]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64
  // CHECK: %[[DESC:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  // Alloc ptr
  // CHECK: %[[DESC0:.*]] = llvm.insertvalue %[[BASE]], %[[DESC]][0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  // Aligned ptr
  // CHECK: %[[DESC1:.*]] = llvm.insertvalue %[[BASE_ALIGNED]], %[[DESC0]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: %[[C3:.*]] = llvm.mlir.constant(3 : index) : i64
  // CHECK: %[[DESC2:.*]] = llvm.insertvalue %[[C3]], %[[DESC1]][2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  // Sizes and strides @rank 0: both static.
  // CHECK: %[[DESC3:.*]] = llvm.insertvalue %[[C3]], %[[DESC2]][3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: %[[CST_STRIDE0:.*]] = llvm.mlir.constant(1 : index) : i64
  // CHECK: %[[DESC4:.*]] = llvm.insertvalue %[[CST_STRIDE0]], %[[DESC3]][4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>

  %1 = memref.subview %0[1, 0][1, 3][1, 1]: memref<5x3xf32> to memref<3xf32, strided<[1], offset: 3>>

  return %1 : memref<3xf32, strided<[1], offset: 3>>
}

// -----

// CHECK-LABEL: func @subview_negative_stride
// CHECK-SAME: (%[[MEM:.*]]: memref<7xf32>)
func.func @subview_negative_stride(%arg0 : memref<7xf32>) -> memref<7xf32, strided<[-1], offset: 6>> {
  // CHECK: %[[MEMREF:.*]] = builtin.unrealized_conversion_cast %[[MEM]]
  // CHECK: %[[BASE:.*]] = llvm.extractvalue %[[MEMREF]][0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64
  // CHECK: %[[BASE_ALIGNED:.*]] = llvm.extractvalue %[[MEMREF]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64
  // CHECK: %[[DESC:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: %[[DESC0:.*]] = llvm.insertvalue %[[BASE]], %[[DESC]][0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: %[[DESC1:.*]] = llvm.insertvalue %[[BASE_ALIGNED]], %[[DESC0]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: %[[CST_OFF0:.*]] = llvm.mlir.constant(6 : index) : i64
  // CHECK: %[[DESC2:.*]] = llvm.insertvalue %[[CST_OFF0]], %[[DESC1]][2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: %[[CST_SIZE0:.*]] = llvm.mlir.constant(7 : index) : i64
  // CHECK: %[[DESC3:.*]] = llvm.insertvalue %[[CST_SIZE0]], %[[DESC2]][3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: %[[CST_STRIDE0:.*]] = llvm.mlir.constant(-1 : index) : i64
  // CHECK: %[[DESC4:.*]] = llvm.insertvalue %[[CST_STRIDE0]], %[[DESC3]][4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[DESC4]] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> to memref<7xf32, strided<[-1], offset: 6>>
  // CHECK: return %[[RES]] : memref<7xf32, strided<[-1], offset: 6>>

  %0 = memref.subview %arg0[6] [7] [-1] : memref<7xf32> to memref<7xf32, strided<[-1], offset: 6>>
  return %0 : memref<7xf32, strided<[-1], offset: 6>>
}
