// RUN: mlir-opt -convert-memref-to-llvm %s -split-input-file | FileCheck %s
// RUN: mlir-opt -convert-memref-to-llvm='index-bitwidth=32' %s -split-input-file | FileCheck --check-prefix=CHECK32 %s

// CHECK-LABEL: func @view(
// CHECK: %[[ARG0F:.*]]: index, %[[ARG1F:.*]]: index, %[[ARG2F:.*]]: index
func.func @view(%arg0 : index, %arg1 : index, %arg2 : index) {
  // CHECK: %[[ARG2:.*]] = builtin.unrealized_conversion_cast %[[ARG2F:.*]]
  // CHECK: %[[ARG0:.*]] = builtin.unrealized_conversion_cast %[[ARG0F:.*]]
  // CHECK: %[[ARG1:.*]] = builtin.unrealized_conversion_cast %[[ARG1F:.*]]
  // CHECK: llvm.mlir.constant(2048 : index) : i64
  // CHECK: llvm.mlir.undef : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
  %0 = memref.alloc() : memref<2048xi8>

  // Test two dynamic sizes.
  // CHECK: llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[BASE_PTR:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: %[[SHIFTED_BASE_PTR:.*]] = llvm.getelementptr %[[BASE_PTR]][%[[ARG2]]] : (!llvm.ptr<i8>, i64) -> !llvm.ptr<i8>
  // CHECK: %[[CAST_SHIFTED_BASE_PTR:.*]] = llvm.bitcast %[[SHIFTED_BASE_PTR]] : !llvm.ptr<i8> to !llvm.ptr<f32>
  // CHECK: llvm.insertvalue %[[CAST_SHIFTED_BASE_PTR]], %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[C0:.*]] = llvm.mlir.constant(0 : index) : i64
  // CHECK: llvm.insertvalue %[[C0]], %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.insertvalue %[[ARG1]], %{{.*}}[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mlir.constant(1 : index) : i64
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.insertvalue %[[ARG0]], %{{.*}}[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mul %{{.*}}, %[[ARG1]]
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  %1 = memref.view %0[%arg2][%arg0, %arg1] : memref<2048xi8> to memref<?x?xf32>

  // Test one dynamic size.
  // CHECK: llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[BASE_PTR_2:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: %[[SHIFTED_BASE_PTR_2:.*]] = llvm.getelementptr %[[BASE_PTR_2]][%[[ARG2]]] : (!llvm.ptr<i8>, i64) -> !llvm.ptr<i8>
  // CHECK: %[[CAST_SHIFTED_BASE_PTR_2:.*]] = llvm.bitcast %[[SHIFTED_BASE_PTR_2]] : !llvm.ptr<i8> to !llvm.ptr<f32>
  // CHECK: llvm.insertvalue %[[CAST_SHIFTED_BASE_PTR_2]], %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[C0_2:.*]] = llvm.mlir.constant(0 : index) : i64
  // CHECK: llvm.insertvalue %[[C0_2]], %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.insertvalue %[[ARG1]], %{{.*}}[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mlir.constant(1 : index) : i64
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mlir.constant(4 : index) : i64
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mul %{{.*}}, %[[ARG1]]
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  %3 = memref.view %0[%arg2][%arg1] : memref<2048xi8> to memref<4x?xf32>

  // Test static sizes.
  // CHECK: llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[BASE_PTR_3:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: %[[SHIFTED_BASE_PTR_3:.*]] = llvm.getelementptr %[[BASE_PTR_3]][%[[ARG2]]] : (!llvm.ptr<i8>, i64) -> !llvm.ptr<i8>
  // CHECK: %[[CAST_SHIFTED_BASE_PTR_3:.*]] = llvm.bitcast %[[SHIFTED_BASE_PTR_3]] : !llvm.ptr<i8> to !llvm.ptr<f32>
  // CHECK: llvm.insertvalue %[[CAST_SHIFTED_BASE_PTR_3]], %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[C0_3:.*]] = llvm.mlir.constant(0 : index) : i64
  // CHECK: llvm.insertvalue %[[C0_3]], %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mlir.constant(4 : index) : i64
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mlir.constant(1 : index) : i64
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mlir.constant(64 : index) : i64
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mlir.constant(4 : index) : i64
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  %5 = memref.view %0[%arg2][] : memref<2048xi8> to memref<64x4xf32>

  // Test view memory space.
  // CHECK: llvm.mlir.constant(2048 : index) : i64
  // CHECK: llvm.mlir.undef : !llvm.struct<(ptr<i8, 4>, ptr<i8, 4>, i64, array<1 x i64>, array<1 x i64>)>
  %6 = memref.alloc() : memref<2048xi8, 4>

  // CHECK: llvm.mlir.undef : !llvm.struct<(ptr<f32, 4>, ptr<f32, 4>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[BASE_PTR_4:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<i8, 4>, ptr<i8, 4>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: %[[SHIFTED_BASE_PTR_4:.*]] = llvm.getelementptr %[[BASE_PTR_4]][%[[ARG2]]] : (!llvm.ptr<i8, 4>, i64) -> !llvm.ptr<i8, 4>
  // CHECK: %[[CAST_SHIFTED_BASE_PTR_4:.*]] = llvm.bitcast %[[SHIFTED_BASE_PTR_4]] : !llvm.ptr<i8, 4> to !llvm.ptr<f32, 4>
  // CHECK: llvm.insertvalue %[[CAST_SHIFTED_BASE_PTR_4]], %{{.*}}[1] : !llvm.struct<(ptr<f32, 4>, ptr<f32, 4>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[C0_4:.*]] = llvm.mlir.constant(0 : index) : i64
  // CHECK: llvm.insertvalue %[[C0_4]], %{{.*}}[2] : !llvm.struct<(ptr<f32, 4>, ptr<f32, 4>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mlir.constant(4 : index) : i64
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[3, 1] : !llvm.struct<(ptr<f32, 4>, ptr<f32, 4>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mlir.constant(1 : index) : i64
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[4, 1] : !llvm.struct<(ptr<f32, 4>, ptr<f32, 4>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mlir.constant(64 : index) : i64
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[3, 0] : !llvm.struct<(ptr<f32, 4>, ptr<f32, 4>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mlir.constant(4 : index) : i64
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[4, 0] : !llvm.struct<(ptr<f32, 4>, ptr<f32, 4>, i64, array<2 x i64>, array<2 x i64>)>
  %7 = memref.view %6[%arg2][] : memref<2048xi8, 4> to memref<64x4xf32, 4>

  return
}

// -----

// CHECK-LABL: func @view_empty_memref(
// CHECK:        %[[ARG0:.*]]: index,
// CHECK:        %[[ARG1:.*]]: memref<0xi8>)
func.func @view_empty_memref(%offset: index, %mem: memref<0xi8>) {

  // CHECK: llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mlir.constant(0 : index) : i64
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mlir.constant(4 : index) : i64
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mlir.constant(0 : index) : i64
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mlir.constant(0 : index) : i64
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mlir.constant(0 : index) : i64
  // CHECK: = llvm.insertvalue %{{.*}}, %{{.*}}[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  %0 = memref.view %mem[%offset][] : memref<0xi8> to memref<0x4xf32>

  return
}

// -----

// CHECK-LABEL: func @subview(
// CHECK:         %[[MEM:.*]]: memref<{{.*}}>,
// CHECK:         %[[ARG0f:[a-zA-Z0-9]*]]: index,
// CHECK:         %[[ARG1f:[a-zA-Z0-9]*]]: index,
// CHECK:         %[[ARG2f:.*]]: index)
// CHECK32-LABEL: func @subview(
// CHECK32:         %[[MEM:.*]]: memref<{{.*}}>,
// CHECK32:         %[[ARG0f:[a-zA-Z0-9]*]]: index,
// CHECK32:         %[[ARG1f:[a-zA-Z0-9]*]]: index,
// CHECK32:         %[[ARG2f:.*]]: index)
func.func @subview(%0 : memref<64x4xf32, offset: 0, strides: [4, 1]>, %arg0 : index, %arg1 : index, %arg2 : index) {
  // CHECK-DAG: %[[MEMREF:.*]] = builtin.unrealized_conversion_cast %[[MEM]]
  // CHECK-DAG: %[[ARG0:.*]] = builtin.unrealized_conversion_cast %[[ARG0f]]
  // CHECK-DAG: %[[ARG1:.*]] = builtin.unrealized_conversion_cast %[[ARG1f]]

  // CHECK32-DAG: %[[MEMREF:.*]] = builtin.unrealized_conversion_cast %[[MEM]]
  // CHECK32-DAG: %[[ARG0:.*]] = builtin.unrealized_conversion_cast %[[ARG0f]]
  // CHECK32-DAG: %[[ARG1:.*]] = builtin.unrealized_conversion_cast %[[ARG1f]]

  // CHECK: %[[DESC:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[BITCAST0:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<f32> to !llvm.ptr<f32>
  // CHECK: %[[DESC0:.*]] = llvm.insertvalue %[[BITCAST0]], %[[DESC]][0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[BITCAST1:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<f32> to !llvm.ptr<f32>
  // CHECK: %[[DESC1:.*]] = llvm.insertvalue %[[BITCAST1]], %[[DESC0]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[STRIDE0:.*]] = llvm.extractvalue %[[MEMREF]][4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[STRIDE1:.*]] = llvm.extractvalue %[[MEMREF]][4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[OFF:.*]] = llvm.extractvalue %[[MEMREF]][2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[OFFINC:.*]] = llvm.mul %[[ARG0]], %[[STRIDE0]] : i64
  // CHECK: %[[OFF1:.*]] = llvm.add %[[OFF]], %[[OFFINC]] : i64
  // CHECK: %[[OFFINC1:.*]] = llvm.mul %[[ARG1]], %[[STRIDE1]] : i64
  // CHECK: %[[OFF2:.*]] = llvm.add %[[OFF1]], %[[OFFINC1]] : i64
  // CHECK: %[[DESC2:.*]] = llvm.insertvalue %[[OFF2]], %[[DESC1]][2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[DESCSTRIDE1:.*]] = llvm.mul %[[ARG1]], %[[STRIDE1]] : i64
  // CHECK: %[[DESC3:.*]] = llvm.insertvalue %[[ARG1]], %[[DESC2]][3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[DESC4:.*]] = llvm.insertvalue %[[DESCSTRIDE1]], %[[DESC3]][4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[DESCSTRIDE0:.*]] = llvm.mul %[[ARG0]], %[[STRIDE0]] : i64
  // CHECK: %[[DESC5:.*]] = llvm.insertvalue %[[ARG0]], %[[DESC4]][3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.insertvalue %[[DESCSTRIDE0]], %[[DESC5]][4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK32: %[[DESC:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[BITCAST0:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<f32> to !llvm.ptr<f32>
  // CHECK32: %[[DESC0:.*]] = llvm.insertvalue %[[BITCAST0]], %[[DESC]][0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[BITCAST1:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<f32> to !llvm.ptr<f32>
  // CHECK32: %[[DESC1:.*]] = llvm.insertvalue %[[BITCAST1]], %[[DESC0]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[STRIDE0:.*]] = llvm.extractvalue %[[MEMREF]][4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[STRIDE1:.*]] = llvm.extractvalue %[[MEMREF]][4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[OFF:.*]] = llvm.extractvalue %[[MEMREF]][2] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[OFFINC:.*]] = llvm.mul %[[ARG0]], %[[STRIDE0]] : i32
  // CHECK32: %[[OFF1:.*]] = llvm.add %[[OFF]], %[[OFFINC]] : i32
  // CHECK32: %[[OFFINC1:.*]] = llvm.mul %[[ARG1]], %[[STRIDE1]] : i32
  // CHECK32: %[[OFF2:.*]] = llvm.add %[[OFF1]], %[[OFFINC1]] : i32
  // CHECK32: %[[DESC2:.*]] = llvm.insertvalue %[[OFF2]], %[[DESC1]][2] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[DESCSTRIDE1:.*]] = llvm.mul %[[ARG1]], %[[STRIDE1]] : i32
  // CHECK32: %[[DESC3:.*]] = llvm.insertvalue %[[ARG1]], %[[DESC2]][3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[DESC4:.*]] = llvm.insertvalue %[[DESCSTRIDE1]], %[[DESC3]][4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[DESCSTRIDE0:.*]] = llvm.mul %[[ARG0]], %[[STRIDE0]] : i32
  // CHECK32: %[[DESC5:.*]] = llvm.insertvalue %[[ARG0]], %[[DESC4]][3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>

  %1 = memref.subview %0[%arg0, %arg1][%arg0, %arg1][%arg0, %arg1] :
    memref<64x4xf32, offset: 0, strides: [4, 1]>
  to memref<?x?xf32, offset: ?, strides: [?, ?]>
  return
}

// -----

// CHECK-LABEL: func @subview_non_zero_addrspace(
// CHECK:         %[[MEM:.*]]: memref<{{.*}}>,
// CHECK:         %[[ARG0f:[a-zA-Z0-9]*]]: index,
// CHECK:         %[[ARG1f:[a-zA-Z0-9]*]]: index,
// CHECK:         %[[ARG2f:.*]]: index)
// CHECK32-LABEL: func @subview_non_zero_addrspace(
// CHECK32:         %[[MEM:.*]]: memref<{{.*}}>,
// CHECK32:         %[[ARG0f:[a-zA-Z0-9]*]]: index,
// CHECK32:         %[[ARG1f:[a-zA-Z0-9]*]]: index,
// CHECK32:         %[[ARG2f:.*]]: index)
func.func @subview_non_zero_addrspace(%0 : memref<64x4xf32, offset: 0, strides: [4, 1], 3>, %arg0 : index, %arg1 : index, %arg2 : index) {
  // CHECK-DAG: %[[MEMREF:.*]] = builtin.unrealized_conversion_cast %[[MEM]]
  // CHECK-DAG: %[[ARG0:.*]] = builtin.unrealized_conversion_cast %[[ARG0f]]
  // CHECK-DAG: %[[ARG1:.*]] = builtin.unrealized_conversion_cast %[[ARG1f]]
  // CHECK32-DAG: %[[MEMREF:.*]] = builtin.unrealized_conversion_cast %[[MEM]]
  // CHECK32-DAG: %[[ARG0:.*]] = builtin.unrealized_conversion_cast %[[ARG0f]]
  // CHECK32-DAG: %[[ARG1:.*]] = builtin.unrealized_conversion_cast %[[ARG1f]]

  // CHECK: %[[DESC:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<f32, 3>, ptr<f32, 3>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[BITCAST0:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<f32, 3> to !llvm.ptr<f32, 3>
  // CHECK: %[[DESC0:.*]] = llvm.insertvalue %[[BITCAST0]], %[[DESC]][0] : !llvm.struct<(ptr<f32, 3>, ptr<f32, 3>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[BITCAST1:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<f32, 3> to !llvm.ptr<f32, 3>
  // CHECK: %[[DESC1:.*]] = llvm.insertvalue %[[BITCAST1]], %[[DESC0]][1] : !llvm.struct<(ptr<f32, 3>, ptr<f32, 3>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[STRIDE0:.*]] = llvm.extractvalue %[[MEMREF]][4, 0] : !llvm.struct<(ptr<f32, 3>, ptr<f32, 3>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[STRIDE1:.*]] = llvm.extractvalue %[[MEMREF]][4, 1] : !llvm.struct<(ptr<f32, 3>, ptr<f32, 3>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[OFF:.*]] = llvm.extractvalue %[[MEMREF]][2] : !llvm.struct<(ptr<f32, 3>, ptr<f32, 3>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[OFFINC:.*]] = llvm.mul %[[ARG0]], %[[STRIDE0]] : i64
  // CHECK: %[[OFF1:.*]] = llvm.add %[[OFF]], %[[OFFINC]] : i64
  // CHECK: %[[OFFINC1:.*]] = llvm.mul %[[ARG1]], %[[STRIDE1]] : i64
  // CHECK: %[[OFF2:.*]] = llvm.add %[[OFF1]], %[[OFFINC1]] : i64
  // CHECK: %[[DESC2:.*]] = llvm.insertvalue %[[OFF2]], %[[DESC1]][2] : !llvm.struct<(ptr<f32, 3>, ptr<f32, 3>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[DESCSTRIDE1:.*]] = llvm.mul %[[ARG1]], %[[STRIDE1]] : i64
  // CHECK: %[[DESC3:.*]] = llvm.insertvalue %[[ARG1]], %[[DESC2]][3, 1] : !llvm.struct<(ptr<f32, 3>, ptr<f32, 3>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[DESC4:.*]] = llvm.insertvalue %[[DESCSTRIDE1]], %[[DESC3]][4, 1] : !llvm.struct<(ptr<f32, 3>, ptr<f32, 3>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[DESCSTRIDE0:.*]] = llvm.mul %[[ARG0]], %[[STRIDE0]] : i64
  // CHECK: %[[DESC5:.*]] = llvm.insertvalue %[[ARG0]], %[[DESC4]][3, 0] : !llvm.struct<(ptr<f32, 3>, ptr<f32, 3>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.insertvalue %[[DESCSTRIDE0]], %[[DESC5]][4, 0] : !llvm.struct<(ptr<f32, 3>, ptr<f32, 3>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK32: %[[DESC:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<f32, 3>, ptr<f32, 3>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[BITCAST0:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<f32, 3> to !llvm.ptr<f32, 3>
  // CHECK32: %[[DESC0:.*]] = llvm.insertvalue %[[BITCAST0]], %[[DESC]][0] : !llvm.struct<(ptr<f32, 3>, ptr<f32, 3>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[BITCAST1:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<f32, 3> to !llvm.ptr<f32, 3>
  // CHECK32: %[[DESC1:.*]] = llvm.insertvalue %[[BITCAST1]], %[[DESC0]][1] : !llvm.struct<(ptr<f32, 3>, ptr<f32, 3>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[STRIDE0:.*]] = llvm.extractvalue %[[MEMREF]][4, 0] : !llvm.struct<(ptr<f32, 3>, ptr<f32, 3>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[STRIDE1:.*]] = llvm.extractvalue %[[MEMREF]][4, 1] : !llvm.struct<(ptr<f32, 3>, ptr<f32, 3>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[OFF:.*]] = llvm.extractvalue %[[MEMREF]][2] : !llvm.struct<(ptr<f32, 3>, ptr<f32, 3>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[OFFINC:.*]] = llvm.mul %[[ARG0]], %[[STRIDE0]] : i32
  // CHECK32: %[[OFF1:.*]] = llvm.add %[[OFF]], %[[OFFINC]] : i32
  // CHECK32: %[[OFFINC1:.*]] = llvm.mul %[[ARG1]], %[[STRIDE1]] : i32
  // CHECK32: %[[OFF2:.*]] = llvm.add %[[OFF1]], %[[OFFINC1]] : i32
  // CHECK32: %[[DESC2:.*]] = llvm.insertvalue %[[OFF2]], %[[DESC1]][2] : !llvm.struct<(ptr<f32, 3>, ptr<f32, 3>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[DESCSTRIDE1:.*]] = llvm.mul %[[ARG1]], %[[STRIDE1]] : i32
  // CHECK32: %[[DESC3:.*]] = llvm.insertvalue %[[ARG1]], %[[DESC2]][3, 1] : !llvm.struct<(ptr<f32, 3>, ptr<f32, 3>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[DESC4:.*]] = llvm.insertvalue %[[DESCSTRIDE1]], %[[DESC3]][4, 1] : !llvm.struct<(ptr<f32, 3>, ptr<f32, 3>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[DESCSTRIDE0:.*]] = llvm.mul %[[ARG0]], %[[STRIDE0]] : i32
  // CHECK32: %[[DESC5:.*]] = llvm.insertvalue %[[ARG0]], %[[DESC4]][3, 0] : !llvm.struct<(ptr<f32, 3>, ptr<f32, 3>, i32, array<2 x i32>, array<2 x i32>)>

  %1 = memref.subview %0[%arg0, %arg1][%arg0, %arg1][%arg0, %arg1] :
    memref<64x4xf32, offset: 0, strides: [4, 1], 3>
    to memref<?x?xf32, offset: ?, strides: [?, ?], 3>
  return
}

// -----

// CHECK-LABEL: func @subview_const_size(
// CHECK-SAME:         %[[MEM:.*]]: memref<{{.*}}>,
// CHECK-SAME:         %[[ARG0f:[a-zA-Z0-9]*]]: index
// CHECK-SAME:         %[[ARG1f:[a-zA-Z0-9]*]]: index
// CHECK-SAME:         %[[ARG2f:[a-zA-Z0-9]*]]: index
// CHECK32-LABEL: func @subview_const_size(
// CHECK32-SAME:         %[[MEM:.*]]: memref<{{.*}}>,
// CHECK32-SAME:         %[[ARG0f:[a-zA-Z0-9]*]]: index
// CHECK32-SAME:         %[[ARG1f:[a-zA-Z0-9]*]]: index
// CHECK32-SAME:         %[[ARG2f:[a-zA-Z0-9]*]]: index
func.func @subview_const_size(%0 : memref<64x4xf32, offset: 0, strides: [4, 1]>, %arg0 : index, %arg1 : index, %arg2 : index) {
  // CHECK-DAG: %[[MEMREF:.*]] = builtin.unrealized_conversion_cast %[[MEM]]
  // CHECK-DAG: %[[ARG0:.*]] = builtin.unrealized_conversion_cast %[[ARG0f]]
  // CHECK-DAG: %[[ARG1:.*]] = builtin.unrealized_conversion_cast %[[ARG1f]]
  // CHECK32-DAG: %[[MEMREF:.*]] = builtin.unrealized_conversion_cast %[[MEM]]
  // CHECK32-DAG: %[[ARG0:.*]] = builtin.unrealized_conversion_cast %[[ARG0f]]
  // CHECK32-DAG: %[[ARG1:.*]] = builtin.unrealized_conversion_cast %[[ARG1f]]

  // CHECK: %[[DESC:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[BITCAST0:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<f32> to !llvm.ptr<f32>
  // CHECK: %[[DESC0:.*]] = llvm.insertvalue %[[BITCAST0]], %[[DESC]][0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[BITCAST1:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<f32> to !llvm.ptr<f32>
  // CHECK: %[[DESC1:.*]] = llvm.insertvalue %[[BITCAST1]], %[[DESC0]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[STRIDE0:.*]] = llvm.extractvalue %[[MEMREF]][4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[STRIDE1:.*]] = llvm.extractvalue %[[MEMREF]][4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[OFF:.*]] = llvm.extractvalue %[[MEMREF]][2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[OFFINC:.*]] = llvm.mul %[[ARG0]], %[[STRIDE0]] : i64
  // CHECK: %[[OFF1:.*]] = llvm.add %[[OFF]], %[[OFFINC]] : i64
  // CHECK: %[[OFFINC1:.*]] = llvm.mul %[[ARG1]], %[[STRIDE1]] : i64
  // CHECK: %[[OFF2:.*]] = llvm.add %[[OFF1]], %[[OFFINC1]] : i64
  // CHECK: %[[DESC2:.*]] = llvm.insertvalue %[[OFF2]], %[[DESC1]][2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[CST2:.*]] = llvm.mlir.constant(2 : i64)
  // CHECK: %[[DESCSTRIDE1:.*]] = llvm.mul %[[ARG1]], %[[STRIDE1]] : i64
  // CHECK: %[[DESC3:.*]] = llvm.insertvalue %[[CST2]], %[[DESC2]][3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[DESC4:.*]] = llvm.insertvalue %[[DESCSTRIDE1]], %[[DESC3]][4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[CST4:.*]] = llvm.mlir.constant(4 : i64)
  // CHECK: %[[DESCSTRIDE0:.*]] = llvm.mul %[[ARG0]], %[[STRIDE0]] : i64
  // CHECK: %[[DESC5:.*]] = llvm.insertvalue %[[CST4]], %[[DESC4]][3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.insertvalue %[[DESCSTRIDE0]], %[[DESC5]][4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK32: %[[DESC:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[BITCAST0:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<f32> to !llvm.ptr<f32>
  // CHECK32: %[[DESC0:.*]] = llvm.insertvalue %[[BITCAST0]], %[[DESC]][0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[BITCAST1:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<f32> to !llvm.ptr<f32>
  // CHECK32: %[[DESC1:.*]] = llvm.insertvalue %[[BITCAST1]], %[[DESC0]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[STRIDE0:.*]] = llvm.extractvalue %[[MEMREF]][4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[STRIDE1:.*]] = llvm.extractvalue %[[MEMREF]][4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[OFF:.*]] = llvm.extractvalue %[[MEMREF]][2] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[OFFINC:.*]] = llvm.mul %[[ARG0]], %[[STRIDE0]] : i32
  // CHECK32: %[[OFF1:.*]] = llvm.add %[[OFF]], %[[OFFINC]] : i32
  // CHECK32: %[[OFFINC1:.*]] = llvm.mul %[[ARG1]], %[[STRIDE1]] : i32
  // CHECK32: %[[OFF2:.*]] = llvm.add %[[OFF1]], %[[OFFINC1]] : i32
  // CHECK32: %[[DESC2:.*]] = llvm.insertvalue %[[OFF2]], %[[DESC1]][2] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[CST2:.*]] = llvm.mlir.constant(2 : i64)
  // CHECK32: %[[DESCSTRIDE1:.*]] = llvm.mul %[[ARG1]], %[[STRIDE1]] : i32
  // CHECK32: %[[DESC3:.*]] = llvm.insertvalue %[[CST2]], %[[DESC2]][3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[DESC4:.*]] = llvm.insertvalue %[[DESCSTRIDE1]], %[[DESC3]][4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[CST4:.*]] = llvm.mlir.constant(4 : i64)
  // CHECK32: %[[DESCSTRIDE0:.*]] = llvm.mul %[[ARG0]], %[[STRIDE0]] : i32
  // CHECK32: %[[DESC5:.*]] = llvm.insertvalue %[[CST4]], %[[DESC4]][3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: llvm.insertvalue %[[DESCSTRIDE0]], %[[DESC5]][4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  %1 = memref.subview %0[%arg0, %arg1][4, 2][%arg0, %arg1] :
    memref<64x4xf32, offset: 0, strides: [4, 1]>
    to memref<4x2xf32, offset: ?, strides: [?, ?]>
  return
}

// -----

// CHECK-LABEL: func @subview_const_stride(
// CHECK-SAME:         %[[MEM:.*]]: memref<{{.*}}>,
// CHECK-SAME:         %[[ARG0f:[a-zA-Z0-9]*]]: index
// CHECK-SAME:         %[[ARG1f:[a-zA-Z0-9]*]]: index
// CHECK-SAME:         %[[ARG2f:[a-zA-Z0-9]*]]: index
// CHECK32-LABEL: func @subview_const_stride(
// CHECK32-SAME:         %[[MEM:.*]]: memref<{{.*}}>,
// CHECK32-SAME:         %[[ARG0f:[a-zA-Z0-9]*]]: index
// CHECK32-SAME:         %[[ARG1f:[a-zA-Z0-9]*]]: index
// CHECK32-SAME:         %[[ARG2f:[a-zA-Z0-9]*]]: index
func.func @subview_const_stride(%0 : memref<64x4xf32, offset: 0, strides: [4, 1]>, %arg0 : index, %arg1 : index, %arg2 : index) {
  // CHECK-DAG: %[[MEMREF:.*]] = builtin.unrealized_conversion_cast %[[MEM]]
  // CHECK-DAG: %[[ARG0:.*]] = builtin.unrealized_conversion_cast %[[ARG0f]]
  // CHECK-DAG: %[[ARG1:.*]] = builtin.unrealized_conversion_cast %[[ARG1f]]
  // CHECK32-DAG: %[[MEMREF:.*]] = builtin.unrealized_conversion_cast %[[MEM]]
  // CHECK32-DAG: %[[ARG0:.*]] = builtin.unrealized_conversion_cast %[[ARG0f]]
  // CHECK32-DAG: %[[ARG1:.*]] = builtin.unrealized_conversion_cast %[[ARG1f]]

  // CHECK: %[[DESC:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[BITCAST0:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<f32> to !llvm.ptr<f32>
  // CHECK: %[[DESC0:.*]] = llvm.insertvalue %[[BITCAST0]], %[[DESC]][0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[BITCAST1:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<f32> to !llvm.ptr<f32>
  // CHECK: %[[DESC1:.*]] = llvm.insertvalue %[[BITCAST1]], %[[DESC0]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[STRIDE0:.*]] = llvm.extractvalue %[[MEMREF]][4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[STRIDE1:.*]] = llvm.extractvalue %[[MEMREF]][4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[OFF:.*]] = llvm.extractvalue %[[MEMREF]][2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[OFFINC:.*]] = llvm.mul %[[ARG0]], %[[STRIDE0]] : i64
  // CHECK: %[[OFF1:.*]] = llvm.add %[[OFF]], %[[OFFINC]] : i64
  // CHECK: %[[OFFINC1:.*]] = llvm.mul %[[ARG1]], %[[STRIDE1]] : i64
  // CHECK: %[[OFF2:.*]] = llvm.add %[[OFF1]], %[[OFFINC1]] : i64
  // CHECK: %[[DESC2:.*]] = llvm.insertvalue %[[OFF2]], %[[DESC1]][2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[CST2:.*]] = llvm.mlir.constant(2 : i64)
  // CHECK: %[[DESC3:.*]] = llvm.insertvalue %[[ARG1]], %[[DESC2]][3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[DESC4:.*]] = llvm.insertvalue %[[CST2]], %[[DESC3]][4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[CST4:.*]] = llvm.mlir.constant(4 : i64)
  // CHECK: %[[DESC5:.*]] = llvm.insertvalue %[[ARG0]], %[[DESC4]][3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.insertvalue %[[CST4]], %[[DESC5]][4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK32: %[[DESC:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[BITCAST0:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<f32> to !llvm.ptr<f32>
  // CHECK32: %[[DESC0:.*]] = llvm.insertvalue %[[BITCAST0]], %[[DESC]][0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[BITCAST1:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<f32> to !llvm.ptr<f32>
  // CHECK32: %[[DESC1:.*]] = llvm.insertvalue %[[BITCAST1]], %[[DESC0]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[STRIDE0:.*]] = llvm.extractvalue %[[MEMREF]][4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[STRIDE1:.*]] = llvm.extractvalue %[[MEMREF]][4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[OFF:.*]] = llvm.extractvalue %[[MEMREF]][2] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[OFFINC:.*]] = llvm.mul %[[ARG0]], %[[STRIDE0]] : i32
  // CHECK32: %[[OFF1:.*]] = llvm.add %[[OFF]], %[[OFFINC]] : i32
  // CHECK32: %[[OFFINC1:.*]] = llvm.mul %[[ARG1]], %[[STRIDE1]] : i32
  // CHECK32: %[[OFF2:.*]] = llvm.add %[[OFF1]], %[[OFFINC1]] : i32
  // CHECK32: %[[DESC2:.*]] = llvm.insertvalue %[[OFF2]], %[[DESC1]][2] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[CST2:.*]] = llvm.mlir.constant(2 : i64)
  // CHECK32: %[[DESC3:.*]] = llvm.insertvalue %[[ARG1]], %[[DESC2]][3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[DESC4:.*]] = llvm.insertvalue %[[CST2]], %[[DESC3]][4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[CST4:.*]] = llvm.mlir.constant(4 : i64)
  // CHECK32: %[[DESC5:.*]] = llvm.insertvalue %[[ARG0]], %[[DESC4]][3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: llvm.insertvalue %[[CST4]], %[[DESC5]][4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  %1 = memref.subview %0[%arg0, %arg1][%arg0, %arg1][1, 2] :
    memref<64x4xf32, offset: 0, strides: [4, 1]>
    to memref<?x?xf32, offset: ?, strides: [4, 2]>
  return
}

// -----

// CHECK-LABEL: func @subview_const_stride_and_offset(
// CHECK32-LABEL: func @subview_const_stride_and_offset(
func.func @subview_const_stride_and_offset(%0 : memref<64x4xf32, offset: 0, strides: [4, 1]>) {
  // The last "insertvalue" that populates the memref descriptor from the function arguments.
  // CHECK: %[[MEMREF:.*]] = builtin.unrealized_conversion_cast
  // CHECK32: %[[MEMREF:.*]] = builtin.unrealized_conversion_cast

  // CHECK32: %[[DESC:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[BITCAST0:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<f32> to !llvm.ptr<f32>
  // CHECK32: %[[DESC0:.*]] = llvm.insertvalue %[[BITCAST0]], %[[DESC]][0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[BITCAST1:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<f32> to !llvm.ptr<f32>
  // CHECK32: %[[DESC1:.*]] = llvm.insertvalue %[[BITCAST1]], %[[DESC0]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[STRIDE0:.*]] = llvm.extractvalue %[[MEMREF]][4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[STRIDE1:.*]] = llvm.extractvalue %[[MEMREF]][4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[CST8:.*]] = llvm.mlir.constant(8 : index)
  // CHECK32: %[[DESC2:.*]] = llvm.insertvalue %[[CST8]], %[[DESC1]][2] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[CST3:.*]] = llvm.mlir.constant(3 : i64)
  // CHECK32: %[[CST1:.*]] = llvm.mlir.constant(1 : i64)
  // CHECK32: %[[DESC3:.*]] = llvm.insertvalue %[[CST3]], %[[DESC2]][3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[DESC4:.*]] = llvm.insertvalue %[[CST1]], %[[DESC3]][4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[CST62:.*]] = llvm.mlir.constant(62 : i64)
  // CHECK32: %[[CST4:.*]] = llvm.mlir.constant(4 : i64)
  // CHECK32: %[[DESC5:.*]] = llvm.insertvalue %[[CST62]], %[[DESC4]][3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: llvm.insertvalue %[[CST4]], %[[DESC5]][4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  %1 = memref.subview %0[0, 8][62, 3][1, 1] :
    memref<64x4xf32, offset: 0, strides: [4, 1]>
    to memref<62x3xf32, offset: 8, strides: [4, 1]>
  return
}

// -----

// CHECK-LABEL: func @subview_mixed_static_dynamic(
// CHECK:         %[[MEM:.*]]: memref<{{.*}}>,
// CHECK:         %[[ARG0f:[a-zA-Z0-9]*]]: index,
// CHECK:         %[[ARG1f:[a-zA-Z0-9]*]]: index,
// CHECK:         %[[ARG2f:.*]]: index)
// CHECK32-LABEL: func @subview_mixed_static_dynamic(
// CHECK32:         %[[MEM:.*]]: memref<{{.*}}>,
// CHECK32:         %[[ARG0f:[a-zA-Z0-9]*]]: index,
// CHECK32:         %[[ARG1f:[a-zA-Z0-9]*]]: index,
// CHECK32:         %[[ARG2f:.*]]: index)
func.func @subview_mixed_static_dynamic(%0 : memref<64x4xf32, offset: 0, strides: [4, 1]>, %arg0 : index, %arg1 : index, %arg2 : index) {
  // CHECK32-DAG: %[[MEMREF:.*]] = builtin.unrealized_conversion_cast %[[MEM]]
  // CHECK32-DAG: %[[ARG1:.*]] = builtin.unrealized_conversion_cast %[[ARG1f]]
  // CHECK32-DAG: %[[ARG2:.*]] = builtin.unrealized_conversion_cast %[[ARG2f]]
  // CHECK32-DAG: %[[ARG0:.*]] = builtin.unrealized_conversion_cast %[[ARG0f]]

  // CHECK32: %[[DESC:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[BITCAST0:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<f32> to !llvm.ptr<f32>
  // CHECK32: %[[DESC0:.*]] = llvm.insertvalue %[[BITCAST0]], %[[DESC]][0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[BITCAST1:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<f32> to !llvm.ptr<f32>
  // CHECK32: %[[DESC1:.*]] = llvm.insertvalue %[[BITCAST1]], %[[DESC0]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[STRIDE0:.*]] = llvm.extractvalue %[[MEMREF]][4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[STRIDE1:.*]] = llvm.extractvalue %[[MEMREF]][4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[OFF:.*]] = llvm.extractvalue %[[MEMREF]][2] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[OFFM1:.*]] = llvm.mul %[[ARG1]], %[[STRIDE0]] : i32
  // CHECK32: %[[OFFA1:.*]] = llvm.add %[[OFF]], %[[OFFM1]] : i32
  // CHECK32: %[[CST8:.*]] = llvm.mlir.constant(8 : i64) : i32
  // CHECK32: %[[OFFM2:.*]] = llvm.mul %[[CST8]], %[[STRIDE1]] : i32
  // CHECK32: %[[OFFA2:.*]] = llvm.add %[[OFFA1]], %[[OFFM2]] : i32
  // CHECK32: %[[DESC2:.*]] = llvm.insertvalue %[[OFFA2]], %[[DESC1]][2] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>

  // CHECK32: %[[CST1:.*]] = llvm.mlir.constant(1 : i64) : i32
  // CHECK32: %[[DESC3:.*]] = llvm.insertvalue %[[ARG2]], %[[DESC2]][3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[DESC4:.*]] = llvm.insertvalue %[[CST1]], %[[DESC3]][4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: %[[CST62:.*]] = llvm.mlir.constant(62 : i64) : i32
  // CHECK32: %[[DESCSTRIDE0:.*]] = llvm.mul %[[ARG0]], %[[STRIDE0]] : i32
  // CHECK32: %[[DESC5:.*]] = llvm.insertvalue %[[CST62]], %[[DESC4]][3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  // CHECK32: llvm.insertvalue %[[DESCSTRIDE0]], %[[DESC5]][4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<2 x i32>, array<2 x i32>)>
  %1 = memref.subview %0[%arg1, 8][62, %arg2][%arg0, 1] :
    memref<64x4xf32, offset: 0, strides: [4, 1]>
    to memref<62x?xf32, offset: ?, strides: [?, 1]>
  return
}

// -----

// CHECK-LABEL: func @subview_leading_operands(
func.func @subview_leading_operands(%0 : memref<5x3xf32>, %1: memref<5x?xf32>) {
  // CHECK: llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // Alloc ptr
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // Aligned ptr
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // Offset
  // CHECK: %[[C6:.*]] = llvm.mlir.constant(6 : index) : i64
  // CHECK: llvm.insertvalue %[[C6:.*]], %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // Sizes and strides @rank 1: both static.
  // CHECK: %[[C3:.*]] = llvm.mlir.constant(3 : i64) : i64
  // CHECK: %[[C1:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK: llvm.insertvalue %[[C3]], %{{.*}}[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.insertvalue %[[C1]], %{{.*}}[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // Sizes and strides @rank 0: both static extracted from type.
  // CHECK: %[[C3_2:.*]] = llvm.mlir.constant(3 : i64) : i64
  // CHECK: %[[C3_3:.*]] = llvm.mlir.constant(3 : i64) : i64
  // CHECK: llvm.insertvalue %[[C3_2]], %{{.*}}[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.insertvalue %[[C3_3]], %{{.*}}[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  %2 = memref.subview %0[2, 0][3, 3][1, 1]: memref<5x3xf32> to memref<3x3xf32, offset: 6, strides: [3, 1]>

  return
}

// -----

// CHECK-LABEL: func @subview_leading_operands_dynamic(
func.func @subview_leading_operands_dynamic(%0 : memref<5x?xf32>) {
  // CHECK: llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // Alloc ptr
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // Aligned ptr
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // Extract strides
  // CHECK: %[[ST0:.*]] = llvm.extractvalue %{{.*}}[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[ST1:.*]] = llvm.extractvalue %{{.*}}[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // Compute and insert offset from 2 + dynamic value.
  // CHECK: %[[OFF0:.*]] = llvm.extractvalue %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[C2:.*]] = llvm.mlir.constant(2 : i64) : i64
  // CHECK: %[[MUL0:.*]] = llvm.mul %[[C2]], %[[ST0]] : i64
  // CHECK: %[[OFF1:.*]] = llvm.add %[[OFF0]], %[[MUL0]] : i64
  // CHECK: %[[C0:.*]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK: %[[MUL1:.*]] = llvm.mul %[[C0]], %[[ST1]] : i64
  // CHECK: %[[NEW_OFF:.*]] = llvm.add %[[OFF1]], %[[MUL1]]  : i64
  // CHECK: llvm.insertvalue %[[NEW_OFF]], %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // Sizes and strides @rank 1: static stride 1, dynamic size unchanged from source memref.
  // CHECK: %[[C1:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.insertvalue %[[C1]], %{{.*}}[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // Sizes and strides @rank 0: both static.
  // CHECK: %[[C3:.*]] = llvm.mlir.constant(3 : i64) : i64
  // CHECK: %[[C1_2:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK: %[[MUL:.*]] = llvm.mul %[[C1_2]], %[[ST0]]  : i64
  // CHECK: llvm.insertvalue %[[C3]], %{{.*}}[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.insertvalue %[[MUL]], %{{.*}}[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  %c0 = arith.constant 1 : index
  %d0 = memref.dim %0, %c0 : memref<5x?xf32>
  %1 = memref.subview %0[2, 0][3, %d0][1, 1]: memref<5x?xf32> to memref<3x?xf32, offset: ?, strides: [?, 1]>
  return
}

// -----

// CHECK-LABEL: func @subview_rank_reducing_leading_operands(
func.func @subview_rank_reducing_leading_operands(%0 : memref<5x3xf32>) {
  // CHECK: llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  // Alloc ptr
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  // Aligned ptr
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  // Extract strides
  // CHECK: %[[ST0:.*]] = llvm.extractvalue %{{.*}}[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[ST1:.*]] = llvm.extractvalue %{{.*}}[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // Offset
  // CHECK: %[[C3:.*]] = llvm.mlir.constant(3 : index) : i64
  // CHECK: llvm.insertvalue %[[C3:.*]], %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  // Sizes and strides @rank 0: both static.
  // CHECK: %[[C3:.*]] = llvm.mlir.constant(3 : i64) : i64
  // CHECK: %[[C1:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK: llvm.insertvalue %[[C3]], %{{.*}}[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: llvm.insertvalue %[[C1]], %{{.*}}[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  %1 = memref.subview %0[1, 0][1, 3][1, 1]: memref<5x3xf32> to memref<3xf32, offset: 3, strides: [1]>

  return
}

// -----

// CHECK-LABEL: func @assume_alignment
func.func @assume_alignment(%0 : memref<4x4xf16>) {
  // CHECK: %[[PTR:.*]] = llvm.extractvalue %[[MEMREF:.*]][1] : !llvm.struct<(ptr<f16>, ptr<f16>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : index) : i64
  // CHECK-NEXT: %[[MASK:.*]] = llvm.mlir.constant(15 : index) : i64
  // CHECK-NEXT: %[[INT:.*]] = llvm.ptrtoint %[[PTR]] : !llvm.ptr<f16> to i64
  // CHECK-NEXT: %[[MASKED_PTR:.*]] = llvm.and %[[INT]], %[[MASK:.*]] : i64
  // CHECK-NEXT: %[[CONDITION:.*]] = llvm.icmp "eq" %[[MASKED_PTR]], %[[ZERO]] : i64
  // CHECK-NEXT: "llvm.intr.assume"(%[[CONDITION]]) : (i1) -> ()
  memref.assume_alignment %0, 16 : memref<4x4xf16>
  return
}

// -----

// CHECK-LABEL: func @dim_of_unranked
// CHECK32-LABEL: func @dim_of_unranked
func.func @dim_of_unranked(%unranked: memref<*xi32>) -> index {
  %c0 = arith.constant 0 : index
  %dim = memref.dim %unranked, %c0 : memref<*xi32>
  return %dim : index
}
// CHECK: %[[UNRANKED_DESC:.*]] = builtin.unrealized_conversion_cast

// CHECK: %[[RANKED_DESC:.*]] = llvm.extractvalue %[[UNRANKED_DESC]][1]
// CHECK-SAME:   : !llvm.struct<(i64, ptr<i8>)>

// CHECK: %[[ZERO_D_DESC:.*]] = llvm.bitcast %[[RANKED_DESC]]
// CHECK-SAME:   : !llvm.ptr<i8> to !llvm.ptr<struct<(ptr<i32>, ptr<i32>, i64)>>

// CHECK: %[[C0_:.*]] = llvm.mlir.constant(0 : index) : i64

// CHECK: %[[OFFSET_PTR:.*]] = llvm.getelementptr %[[ZERO_D_DESC]]{{\[}}
// CHECK-SAME:   %[[C0_]], 2] : (!llvm.ptr<struct<(ptr<i32>, ptr<i32>,
// CHECK-SAME:   i64)>>, i64) -> !llvm.ptr<i64>

// CHECK: %[[C1:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK: %[[INDEX_INC:.*]] = llvm.add %[[C1]], %{{.*}} : i64

// CHECK: %[[SIZE_PTR:.*]] = llvm.getelementptr %[[OFFSET_PTR]]{{\[}}
// CHECK-SAME:   %[[INDEX_INC]]] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>

// CHECK: %[[SIZE:.*]] = llvm.load %[[SIZE_PTR]] : !llvm.ptr<i64>

// CHECK32: %[[SIZE:.*]] = llvm.load %{{.*}} : !llvm.ptr<i32>

// -----

// CHECK-LABEL: func @address_space(
func.func @address_space(%arg0 : memref<32xf32, affine_map<(d0) -> (d0)>, 7>) {
  %0 = memref.alloc() : memref<32xf32, affine_map<(d0) -> (d0)>, 5>
  %1 = arith.constant 7 : index
  // CHECK: llvm.load %{{.*}} : !llvm.ptr<f32, 5>
  %2 = memref.load %0[%1] : memref<32xf32, affine_map<(d0) -> (d0)>, 5>
  func.return
}

// -----

// CHECK-LABEL: func @transpose
//       CHECK:   llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:   llvm.insertvalue {{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:    llvm.insertvalue {{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:    llvm.insertvalue {{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:   llvm.extractvalue {{.*}}[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:    llvm.insertvalue {{.*}}[3, 2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:   llvm.extractvalue {{.*}}[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:    llvm.insertvalue {{.*}}[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:   llvm.extractvalue {{.*}}[3, 2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:    llvm.insertvalue {{.*}}[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
func.func @transpose(%arg0: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>) {
  %0 = memref.transpose %arg0 (i, j, k) -> (k, i, j) : memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]> to memref<?x?x?xf32, affine_map<(d0, d1, d2)[s0, s1, s2] -> (d2 * s1 + s0 + d0 * s2 + d1)>>
  return
}

// -----

// CHECK:   llvm.mlir.global external @gv0() : !llvm.array<2 x f32> {
// CHECK-NEXT:     %0 = llvm.mlir.undef : !llvm.array<2 x f32>
// CHECK-NEXT:     llvm.return %0 : !llvm.array<2 x f32>
// CHECK-NEXT:   }
memref.global @gv0 : memref<2xf32> = uninitialized

// CHECK: llvm.mlir.global private @gv1() : !llvm.array<2 x f32>
memref.global "private" @gv1 : memref<2xf32>

// CHECK: llvm.mlir.global external @gv2(dense<{{\[\[}}0.000000e+00, 1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00, 5.000000e+00]]> : tensor<2x3xf32>) : !llvm.array<2 x array<3 x f32>>
memref.global @gv2 : memref<2x3xf32> = dense<[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]>

// Test 1D memref.
// CHECK-LABEL: func @get_gv0_memref
func.func @get_gv0_memref() {
  %0 = memref.get_global @gv0 : memref<2xf32>
  // CHECK: %[[DIM:.*]] = llvm.mlir.constant(2 : index) : i64
  // CHECK: %[[STRIDE:.*]] = llvm.mlir.constant(1 : index) : i64
  // CHECK: %[[ADDR:.*]] = llvm.mlir.addressof @gv0 : !llvm.ptr<array<2 x f32>>
  // CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : index) : i64
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[ADDR]][%[[ZERO]], %[[ZERO]]] : (!llvm.ptr<array<2 x f32>>, i64, i64) -> !llvm.ptr<f32>
  // CHECK: %[[DEADBEEF:.*]] = llvm.mlir.constant(3735928559 : index) : i64
  // CHECK: %[[DEADBEEFPTR:.*]] = llvm.inttoptr %[[DEADBEEF]] : i64 to !llvm.ptr<f32>
  // CHECK: llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: llvm.insertvalue %[[DEADBEEFPTR]], {{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: llvm.insertvalue %[[GEP]], {{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: %[[OFFSET:.*]] = llvm.mlir.constant(0 : index) : i64
  // CHECK: llvm.insertvalue %[[OFFSET]], {{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: llvm.insertvalue %[[DIM]], {{.*}}[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: llvm.insertvalue %[[STRIDE]], {{.*}}[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  return
}

// Test 2D memref.
// CHECK-LABEL: func @get_gv2_memref
func.func @get_gv2_memref() {
  // CHECK: %[[DIM0:.*]] = llvm.mlir.constant(2 : index) : i64
  // CHECK: %[[DIM1:.*]] = llvm.mlir.constant(3 : index) : i64
  // CHECK: %[[STRIDE1:.*]] = llvm.mlir.constant(1 : index) : i64
  // CHECK: %[[ADDR:.*]] = llvm.mlir.addressof @gv2 : !llvm.ptr<array<2 x array<3 x f32>>>
  // CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : index) : i64
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[ADDR]][%[[ZERO]], %[[ZERO]], %[[ZERO]]] : (!llvm.ptr<array<2 x array<3 x f32>>>, i64, i64, i64) -> !llvm.ptr<f32>
  // CHECK: %[[DEADBEEF:.*]] = llvm.mlir.constant(3735928559 : index) : i64
  // CHECK: %[[DEADBEEFPTR:.*]] = llvm.inttoptr %[[DEADBEEF]] : i64 to !llvm.ptr<f32>
  // CHECK: llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.insertvalue %[[DEADBEEFPTR]], {{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.insertvalue %[[GEP]], {{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[OFFSET:.*]] = llvm.mlir.constant(0 : index) : i64
  // CHECK: llvm.insertvalue %[[OFFSET]], {{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.insertvalue %[[DIM0]], {{.*}}[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.insertvalue %[[DIM1]], {{.*}}[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.insertvalue %[[DIM1]], {{.*}}[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.insertvalue %[[STRIDE1]], {{.*}}[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>

  %0 = memref.get_global @gv2 : memref<2x3xf32>
  return
}

// Test scalar memref.
// CHECK: llvm.mlir.global external @gv3(1.000000e+00 : f32) : f32
memref.global @gv3 : memref<f32> = dense<1.0>

// CHECK-LABEL: func @get_gv3_memref
func.func @get_gv3_memref() {
  // CHECK: %[[ADDR:.*]] = llvm.mlir.addressof @gv3 : !llvm.ptr<f32>
  // CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : index) : i64
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[ADDR]][%[[ZERO]]] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
  // CHECK: %[[DEADBEEF:.*]] = llvm.mlir.constant(3735928559 : index) : i64
  // CHECK: %[[DEADBEEFPTR:.*]] = llvm.inttoptr %[[DEADBEEF]] : i64 to !llvm.ptr<f32>
  // CHECK: llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64)>
  // CHECK: llvm.insertvalue %[[DEADBEEFPTR]], {{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64)>
  // CHECK: llvm.insertvalue %[[GEP]], {{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64)>
  // CHECK: %[[OFFSET:.*]] = llvm.mlir.constant(0 : index) : i64
  // CHECK: llvm.insertvalue %[[OFFSET]], {{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64)>
  %0 = memref.get_global @gv3 : memref<f32>
  return
}

// Test scalar memref with an alignment.
// CHECK: llvm.mlir.global private @gv4(1.000000e+00 : f32) {alignment = 64 : i64} : f32
memref.global "private" @gv4 : memref<f32> = dense<1.0> {alignment = 64}

// -----

func.func @collapse_shape_static(%arg0: memref<1x3x4x1x5xf32>) -> memref<3x4x5xf32> {
  %0 = memref.collapse_shape %arg0 [[0, 1], [2], [3, 4]] :
    memref<1x3x4x1x5xf32> into memref<3x4x5xf32>
  return %0 : memref<3x4x5xf32>
}
// CHECK-LABEL: func @collapse_shape_static
//       CHECK:    llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:    llvm.extractvalue %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<5 x i64>, array<5 x i64>)>
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:    llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<5 x i64>, array<5 x i64>)>
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:    llvm.extractvalue %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<5 x i64>, array<5 x i64>)>
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:    llvm.mlir.constant(3 : index) : i64
//       CHECK:    llvm.mlir.constant(4 : index) : i64
//       CHECK:    llvm.mlir.constant(5 : index) : i64
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[3, 2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:    llvm.mlir.constant(20 : index) : i64
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:    llvm.mlir.constant(5 : index) : i64
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:    llvm.mlir.constant(1 : index) : i64
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[4, 2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>

// -----

func.func @collapse_shape_dynamic_with_non_identity_layout(
        %arg0 : memref<4x?x?xf32, affine_map<(d0, d1, d2)[s0, s1] -> (d0 * s1 + s0 + d1 * 4 + d2)>>) ->
        memref<4x?xf32, affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2)>> {
  %0 = memref.collapse_shape %arg0 [[0], [1, 2]]:
    memref<4x?x?xf32, affine_map<(d0, d1, d2)[s0, s1] -> (d0 * s1 + s0 + d1 * 4 + d2)>> into
    memref<4x?xf32, affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2)>>
  return %0 : memref<4x?xf32, affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2)>>
}
// CHECK-LABEL:   func @collapse_shape_dynamic_with_non_identity_layout(
//       CHECK:      llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
//       CHECK:      llvm.extractvalue %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:      llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
//       CHECK:      llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:      llvm.insertvalue %{{.*}}, %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
//       CHECK:      llvm.extractvalue %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:      llvm.insertvalue %{{.*}}, %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
//       CHECK:      llvm.mlir.constant(1 : index) : i64
//       CHECK:      llvm.extractvalue %{{.*}}[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:      llvm.mul %{{.*}}, %{{.*}}  : i64
//       CHECK:      llvm.extractvalue %{{.*}}[3, 2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:      llvm.mul %{{.*}}, %{{.*}}  : i64
//       CHECK:      llvm.mlir.constant(4 : index) : i64
//       CHECK:      llvm.insertvalue %{{.*}}, %{{.*}}[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
//       CHECK:      llvm.insertvalue %{{.*}}, %{{.*}}[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
//       CHECK:      llvm.extractvalue %{{.*}}[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:      llvm.insertvalue %{{.*}}, %{{.*}}[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
//       CHECK:      llvm.extractvalue %{{.*}}[4, 2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:      llvm.mlir.constant(1 : i32) : i64
//       CHECK:      llvm.extractvalue %{{.*}}[3, 2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:      llvm.icmp "ne" %{{.*}}, %{{.*}} : i64
//       CHECK:      llvm.cond_br %{{.*}}, ^bb2(%{{.*}} : i64), ^bb1
//       CHECK:      ^bb1:
//       CHECK:      llvm.extractvalue %{{.*}}[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:      llvm.br ^bb2(%{{.*}} : i64)
//       CHECK:      ^bb2(%[[STRIDE:.*]]: i64):
//       CHECK:      llvm.insertvalue %[[STRIDE]], %{{.*}}[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>

// -----

func.func @expand_shape_static(%arg0: memref<3x4x5xf32>) -> memref<1x3x4x1x5xf32> {
  // Reshapes that expand a contiguous tensor with some 1's.
  %0 = memref.expand_shape %arg0 [[0, 1], [2], [3, 4]]
      : memref<3x4x5xf32> into memref<1x3x4x1x5xf32>
  return %0 : memref<1x3x4x1x5xf32>
}
// CHECK-LABEL: func @expand_shape_static
//       CHECK:    llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<5 x i64>, array<5 x i64>)>
//       CHECK:    llvm.extractvalue %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<5 x i64>, array<5 x i64>)>
//       CHECK:    llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<5 x i64>, array<5 x i64>)>
//       CHECK:    llvm.extractvalue %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<5 x i64>, array<5 x i64>)>
//       CHECK:    llvm.mlir.constant(1 : index) : i64
//       CHECK:    llvm.mlir.constant(3 : index) : i64
//       CHECK:    llvm.mlir.constant(4 : index) : i64
//       CHECK:    llvm.mlir.constant(1 : index) : i64
//       CHECK:    llvm.mlir.constant(5 : index) : i64
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<5 x i64>, array<5 x i64>)>
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<5 x i64>, array<5 x i64>)>
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[3, 2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<5 x i64>, array<5 x i64>)>
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[3, 3] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<5 x i64>, array<5 x i64>)>
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[3, 4] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<5 x i64>, array<5 x i64>)>
//       CHECK:    llvm.mlir.constant(60 : index) : i64
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<5 x i64>, array<5 x i64>)>
//       CHECK:    llvm.mlir.constant(20 : index) : i64
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<5 x i64>, array<5 x i64>)>
//       CHECK:    llvm.mlir.constant(5 : index) : i64
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[4, 2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<5 x i64>, array<5 x i64>)>
//       CHECK:    llvm.mlir.constant(5 : index) : i64
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[4, 3] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<5 x i64>, array<5 x i64>)>
//       CHECK:    llvm.mlir.constant(1 : index) : i64
//       CHECK:    llvm.insertvalue %{{.*}}, %{{.*}}[4, 4] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<5 x i64>, array<5 x i64>)>


// -----

func.func @collapse_shape_fold_zero_dim(%arg0 : memref<1x1xf32>) -> memref<f32> {
  %0 = memref.collapse_shape %arg0 [] : memref<1x1xf32> into memref<f32>
  return %0 : memref<f32>
}
// CHECK-LABEL: func @collapse_shape_fold_zero_dim
//       CHECK:   llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64)>
//       CHECK:   llvm.extractvalue %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
//       CHECK:   llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64)>
//       CHECK:   llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
//       CHECK:   llvm.insertvalue %{{.*}}, %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64)>
//       CHECK:   llvm.extractvalue %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
//       CHECK:   llvm.insertvalue %{{.*}}, %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64)>

// -----

func.func @expand_shape_zero_dim(%arg0 : memref<f32>) -> memref<1x1xf32> {
  %0 = memref.expand_shape %arg0 [] : memref<f32> into memref<1x1xf32>
  return %0 : memref<1x1xf32>
}
// CHECK-LABEL: func @expand_shape_zero_dim
//       CHECK:   llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
//       CHECK:   llvm.extractvalue %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64)>
//       CHECK:   llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
//       CHECK:   llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64)>
//       CHECK:   llvm.insertvalue %{{.*}}, %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
//       CHECK:   llvm.extractvalue %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64)>
//       CHECK:   llvm.insertvalue %{{.*}}, %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
//       CHECK:   llvm.mlir.constant(1 : index) : i64
//       CHECK:   llvm.mlir.constant(1 : index) : i64
//       CHECK:   llvm.insertvalue %{{.*}}, %{{.*}}[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
//       CHECK:   llvm.insertvalue %{{.*}}, %{{.*}}[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
//       CHECK:   llvm.mlir.constant(1 : index) : i64
//       CHECK:   llvm.insertvalue %{{.*}}, %{{.*}}[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
//       CHECK:   llvm.mlir.constant(1 : index) : i64
//       CHECK:   llvm.insertvalue %{{.*}}, %{{.*}}[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>

// -----

func.func @collapse_shape_dynamic(%arg0 : memref<1x2x?xf32>) -> memref<1x?xf32> {
  %0 = memref.collapse_shape %arg0 [[0], [1, 2]]:  memref<1x2x?xf32> into memref<1x?xf32>
  return %0 : memref<1x?xf32>
}
// CHECK-LABEL:   func @collapse_shape_dynamic(
// CHECK:           llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           llvm.extractvalue %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           llvm.insertvalue %{{.*}}, %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           llvm.extractvalue %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           llvm.insertvalue %{{.*}}, %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           llvm.mlir.constant(1 : index) : i64
// CHECK:           llvm.mlir.constant(2 : index) : i64
// CHECK:           llvm.mul %{{.*}}, %{{.*}}  : i64
// CHECK:           llvm.extractvalue %{{.*}}[3, 2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           llvm.mul %{{.*}}, %{{.*}}  : i64
// CHECK:           llvm.mlir.constant(1 : index) : i64
// CHECK:           llvm.insertvalue %{{.*}}, %{{.*}}[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           llvm.insertvalue %{{.*}}, %{{.*}}[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           llvm.mlir.constant(1 : index) : i64
// CHECK:           llvm.insertvalue %{{.*}}, %{{.*}}[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           llvm.mul %{{.*}}, %{{.*}}  : i64
// CHECK:           llvm.insertvalue %{{.*}}, %{{.*}}[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           llvm.mul %{{.*}}, %{{.*}}  : i64

// -----

func.func @expand_shape_dynamic(%arg0 : memref<1x?xf32>) -> memref<1x2x?xf32> {
  %0 = memref.expand_shape %arg0 [[0], [1, 2]]: memref<1x?xf32> into memref<1x2x?xf32>
  return %0 : memref<1x2x?xf32>
}
// CHECK-LABEL:   func @expand_shape_dynamic(
// CHECK:           llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           llvm.extractvalue %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           llvm.insertvalue %{{.*}}, %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           llvm.extractvalue %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           llvm.insertvalue %{{.*}}, %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           llvm.extractvalue %{{.*}}[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           llvm.mlir.constant(2 : index) : i64
// CHECK:           llvm.sdiv %{{.*}}, %{{.*}}  : i64
// CHECK:           llvm.mlir.constant(1 : index) : i64
// CHECK:           llvm.mlir.constant(2 : index) : i64
// CHECK:           llvm.insertvalue %{{.*}}, %{{.*}}[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           llvm.insertvalue %{{.*}}, %{{.*}}[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           llvm.insertvalue %{{.*}}, %{{.*}}[3, 2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           llvm.mlir.constant(1 : index) : i64
// CHECK:           llvm.insertvalue %{{.*}}, %{{.*}}[4, 2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           llvm.mul %{{.*}}, %{{.*}}  : i64
// CHECK:           llvm.insertvalue %{{.*}}, %{{.*}}[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           llvm.mul %{{.*}}, %{{.*}}  : i64
// CHECK:           llvm.insertvalue %{{.*}}, %{{.*}}[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           llvm.mul %{{.*}}, %{{.*}}  : i64

// -----

func.func @expand_shape_dynamic_with_non_identity_layout(
            %arg0 : memref<1x?xf32, affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2)>>) ->
            memref<1x2x?xf32, affine_map<(d0, d1, d2)[s0, s1, s2, s3] -> (d0 * s1 + s0 + d1 * s2 + d2 * s3)>> {
  %0 = memref.expand_shape %arg0 [[0], [1, 2]]:
    memref<1x?xf32, affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2)>> into
    memref<1x2x?xf32, affine_map<(d0, d1, d2)[s0, s1, s2, s3] -> (d0 * s1 + s0 + d1 * s2 + d2 * s3)>>
  return %0 : memref<1x2x?xf32, affine_map<(d0, d1, d2)[s0, s1, s2, s3] -> (d0 * s1 + s0 + d1 * s2 + d2 * s3)>>
}
// CHECK-LABEL:   func @expand_shape_dynamic_with_non_identity_layout(
// CHECK:           llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           llvm.extractvalue %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           llvm.insertvalue %{{.*}}, %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           llvm.extractvalue %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           llvm.insertvalue %{{.*}}, %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           llvm.extractvalue %{{.*}}[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           llvm.mlir.constant(2 : index) : i64
// CHECK:           llvm.sdiv %{{.*}}, %{{.*}}  : i64
// CHECK:           llvm.mlir.constant(1 : index) : i64
// CHECK:           llvm.mlir.constant(2 : index) : i64
// CHECK:           llvm.insertvalue %{{.*}}, %{{.*}}[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           llvm.insertvalue %{{.*}}, %{{.*}}[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           llvm.insertvalue %{{.*}}, %{{.*}}[3, 2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           llvm.extractvalue %{{.*}}[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           llvm.insertvalue %{{.*}}, %{{.*}}[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           llvm.extractvalue %{{.*}}[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           llvm.mul %{{.*}}, %{{.*}}  : i64
// CHECK:           llvm.extractvalue %{{.*}}[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           llvm.insertvalue %{{.*}}, %{{.*}}[4, 2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           llvm.extractvalue %{{.*}}[3, 2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           llvm.mul %{{.*}}, %{{.*}}  : i64
// CHECK:           llvm.insertvalue %{{.*}}, %{{.*}}[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           llvm.extractvalue %{{.*}}[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           llvm.mul %{{.*}}, %{{.*}}  : i64

// -----

// CHECK-LABEL: func @rank_of_unranked
// CHECK32-LABEL: func @rank_of_unranked
func.func @rank_of_unranked(%unranked: memref<*xi32>) {
  %rank = memref.rank %unranked : memref<*xi32>
  return
}
// CHECK: %[[UNRANKED_DESC:.*]] = builtin.unrealized_conversion_cast
// CHECK-NEXT: llvm.extractvalue %[[UNRANKED_DESC]][0] : !llvm.struct<(i64, ptr<i8>)>
// CHECK32: llvm.extractvalue %{{.*}}[0] : !llvm.struct<(i32, ptr<i8>)>

// CHECK-LABEL: func @rank_of_ranked
// CHECK32-LABEL: func @rank_of_ranked
func.func @rank_of_ranked(%ranked: memref<?xi32>) {
  %rank = memref.rank %ranked : memref<?xi32>
  return
}
// CHECK: llvm.mlir.constant(1 : index) : i64
// CHECK32: llvm.mlir.constant(1 : index) : i32

// -----

// CHECK-LABEL: func @atomic_rmw
func.func @atomic_rmw(%I : memref<10xi32>, %ival : i32, %F : memref<10xf32>, %fval : f32, %i : index) {
  memref.atomic_rmw assign %fval, %F[%i] : (f32, memref<10xf32>) -> f32
  // CHECK: llvm.atomicrmw xchg %{{.*}}, %{{.*}} acq_rel
  memref.atomic_rmw addi %ival, %I[%i] : (i32, memref<10xi32>) -> i32
  // CHECK: llvm.atomicrmw add %{{.*}}, %{{.*}} acq_rel
  memref.atomic_rmw maxs %ival, %I[%i] : (i32, memref<10xi32>) -> i32
  // CHECK: llvm.atomicrmw max %{{.*}}, %{{.*}} acq_rel
  memref.atomic_rmw mins %ival, %I[%i] : (i32, memref<10xi32>) -> i32
  // CHECK: llvm.atomicrmw min %{{.*}}, %{{.*}} acq_rel
  memref.atomic_rmw maxu %ival, %I[%i] : (i32, memref<10xi32>) -> i32
  // CHECK: llvm.atomicrmw umax %{{.*}}, %{{.*}} acq_rel
  memref.atomic_rmw minu %ival, %I[%i] : (i32, memref<10xi32>) -> i32
  // CHECK: llvm.atomicrmw umin %{{.*}}, %{{.*}} acq_rel
  memref.atomic_rmw addf %fval, %F[%i] : (f32, memref<10xf32>) -> f32
  // CHECK: llvm.atomicrmw fadd %{{.*}}, %{{.*}} acq_rel
  memref.atomic_rmw ori %ival, %I[%i] : (i32, memref<10xi32>) -> i32
  // CHECK: llvm.atomicrmw _or %{{.*}}, %{{.*}} acq_rel
  memref.atomic_rmw andi %ival, %I[%i] : (i32, memref<10xi32>) -> i32
  // CHECK: llvm.atomicrmw _and %{{.*}}, %{{.*}} acq_rel
  return
}

// -----

// CHECK-LABEL: func @collapse_static_shape_with_non_identity_layout
func.func @collapse_static_shape_with_non_identity_layout(%arg: memref<1x1x8x8xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 64 + s0 + d1 * 64 + d2 * 8 + d3)>>) -> memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>> {
// CHECK-NOT: memref.collapse_shape
  %1 = memref.collapse_shape %arg [[0, 1, 2, 3]] : memref<1x1x8x8xf32, affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 64 + s0 + d1 * 64 + d2 * 8 + d3)>> into memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>>
  return %1 : memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>>
}

// -----

// CHECK-LABEL: func @generic_atomic_rmw
func.func @generic_atomic_rmw(%I : memref<10xi32>, %i : index) {
  %x = memref.generic_atomic_rmw %I[%i] : memref<10xi32> {
    ^bb0(%old_value : i32):
      memref.atomic_yield %old_value : i32
  }
  // CHECK: [[init:%.*]] = llvm.load %{{.*}} : !llvm.ptr<i32>
  // CHECK-NEXT: llvm.br ^bb1([[init]] : i32)
  // CHECK-NEXT: ^bb1([[loaded:%.*]]: i32):
  // CHECK-NEXT: [[pair:%.*]] = llvm.cmpxchg %{{.*}}, [[loaded]], [[loaded]]
  // CHECK-SAME:                    acq_rel monotonic : i32
  // CHECK-NEXT: [[new:%.*]] = llvm.extractvalue [[pair]][0]
  // CHECK-NEXT: [[ok:%.*]] = llvm.extractvalue [[pair]][1]
  // CHECK-NEXT: llvm.cond_br [[ok]], ^bb2, ^bb1([[new]] : i32)
  llvm.return
}

// -----

// CHECK-LABEL: func @memref_copy_ranked
func.func @memref_copy_ranked() {
  %0 = memref.alloc() : memref<2xf32>
  // CHECK: llvm.mlir.constant(2 : index) : i64
  // CHECK: llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  %1 = memref.cast %0 : memref<2xf32> to memref<?xf32>
  %2 = memref.alloc() : memref<2xf32>
  // CHECK: llvm.mlir.constant(2 : index) : i64
  // CHECK: llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  %3 = memref.cast %2 : memref<2xf32> to memref<?xf32>
  memref.copy %1, %3 : memref<?xf32> to memref<?xf32>
  // CHECK: [[ONE:%.*]] = llvm.mlir.constant(1 : index) : i64
  // CHECK: [[EXTRACT0:%.*]] = llvm.extractvalue {{%.*}}[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: [[MUL:%.*]] = llvm.mul [[ONE]], [[EXTRACT0]] : i64
  // CHECK: [[NULL:%.*]] = llvm.mlir.null : !llvm.ptr<f32>
  // CHECK: [[ONE2:%.*]] = llvm.mlir.constant(1 : index) : i64
  // CHECK: [[GEP:%.*]] = llvm.getelementptr [[NULL]][[[ONE2]]] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
  // CHECK: [[PTRTOINT:%.*]] = llvm.ptrtoint [[GEP]] : !llvm.ptr<f32> to i64
  // CHECK: [[SIZE:%.*]] = llvm.mul [[MUL]], [[PTRTOINT]] : i64
  // CHECK: [[EXTRACT1P:%.*]] = llvm.extractvalue {{%.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: [[EXTRACT1O:%.*]] = llvm.extractvalue {{%.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: [[GEP1:%.*]] = llvm.getelementptr [[EXTRACT1P]][[[EXTRACT1O]]] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
  // CHECK: [[EXTRACT2P:%.*]] = llvm.extractvalue {{%.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: [[EXTRACT2O:%.*]] = llvm.extractvalue {{%.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: [[GEP2:%.*]] = llvm.getelementptr [[EXTRACT2P]][[[EXTRACT2O]]] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
  // CHECK: [[VOLATILE:%.*]] = llvm.mlir.constant(false) : i1
  // CHECK: "llvm.intr.memcpy"([[GEP2]], [[GEP1]], [[SIZE]], [[VOLATILE]]) : (!llvm.ptr<f32>, !llvm.ptr<f32>, i64, i1) -> ()
  return
}


// -----

// CHECK-LABEL: func @memref_copy_contiguous
#map = affine_map<(d0, d1)[s0] -> (d0 * 2 + s0 + d1)>
func.func @memref_copy_contiguous(%in: memref<16x2xi32>, %offset: index) {
  %buf = memref.alloc() : memref<1x2xi32>
  %sub = memref.subview %in[%offset, 0] [1, 2] [1, 1] : memref<16x2xi32> to memref<1x2xi32, #map>
  memref.copy %sub, %buf : memref<1x2xi32, #map> to memref<1x2xi32>
  // CHECK: [[EXTRACT0:%.*]] = llvm.extractvalue {{%.*}}[3, 0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[MUL1:%.*]] = llvm.mul {{.*}}, [[EXTRACT0]] : i64
  // CHECK: [[EXTRACT1:%.*]] = llvm.extractvalue {{%.*}}[3, 1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[MUL2:%.*]] = llvm.mul [[MUL1]], [[EXTRACT1]] : i64
  // CHECK: [[NULL:%.*]] = llvm.mlir.null : !llvm.ptr<i32>
  // CHECK: [[ONE2:%.*]] = llvm.mlir.constant(1 : index) : i64
  // CHECK: [[GEP:%.*]] = llvm.getelementptr [[NULL]][[[ONE2]]] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
  // CHECK: [[PTRTOINT:%.*]] = llvm.ptrtoint [[GEP]] : !llvm.ptr<i32> to i64
  // CHECK: [[SIZE:%.*]] = llvm.mul [[MUL2]], [[PTRTOINT]] : i64
  // CHECK: [[EXTRACT1P:%.*]] = llvm.extractvalue {{%.*}}[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[EXTRACT1O:%.*]] = llvm.extractvalue {{%.*}}[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[GEP1:%.*]] = llvm.getelementptr [[EXTRACT1P]][[[EXTRACT1O]]] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
  // CHECK: [[EXTRACT2P:%.*]] = llvm.extractvalue {{%.*}}[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[EXTRACT2O:%.*]] = llvm.extractvalue {{%.*}}[2] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[GEP2:%.*]] = llvm.getelementptr [[EXTRACT2P]][[[EXTRACT2O]]] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
  // CHECK: [[VOLATILE:%.*]] = llvm.mlir.constant(false) : i1
  // CHECK: "llvm.intr.memcpy"([[GEP2]], [[GEP1]], [[SIZE]], [[VOLATILE]]) : (!llvm.ptr<i32>, !llvm.ptr<i32>, i64, i1) -> ()
  return
}

// -----

// CHECK-LABEL: func @memref_copy_0d_offset
#map0 = affine_map<(d0) -> (d0 + 1)>
#map1 = affine_map<() -> (1)>
func.func @memref_copy_0d_offset(%in: memref<2xi32>) {
  %buf = memref.alloc() : memref<i32>
  %sub = memref.subview %in[1] [1] [1] : memref<2xi32> to memref<1xi32, #map0>
  %scalar = memref.collapse_shape %sub [] : memref<1xi32, #map0> into memref<i32, #map1>
  memref.copy %scalar, %buf : memref<i32, #map1> to memref<i32>
  // CHECK: llvm.intr.memcpy
  return
}

// -----

// CHECK-LABEL: func @memref_copy_noncontiguous
#map = affine_map<(d0, d1)[s0] -> (d0 * 2 + s0 + d1)>
func.func @memref_copy_noncontiguous(%in: memref<16x2xi32>, %offset: index) {
  %buf = memref.alloc() : memref<2x1xi32>
  %sub = memref.subview %in[%offset, 0] [2, 1] [1, 1] : memref<16x2xi32> to memref<2x1xi32, #map>
  memref.copy %sub, %buf : memref<2x1xi32, #map> to memref<2x1xi32>
  // CHECK: llvm.call @memrefCopy
  return
}

// -----

// CHECK-LABEL: func @memref_copy_unranked
func.func @memref_copy_unranked() {
  %0 = memref.alloc() : memref<2xi1>
  // CHECK: llvm.mlir.constant(2 : index) : i64
  // CHECK: llvm.mlir.undef : !llvm.struct<(ptr<i1>, ptr<i1>, i64, array<1 x i64>, array<1 x i64>)>
  %1 = memref.cast %0 : memref<2xi1> to memref<*xi1>
  %2 = memref.alloc() : memref<2xi1>
  // CHECK: llvm.mlir.constant(2 : index) : i64
  // CHECK: llvm.mlir.undef : !llvm.struct<(ptr<i1>, ptr<i1>, i64, array<1 x i64>, array<1 x i64>)>
  %3 = memref.cast %2 : memref<2xi1> to memref<*xi1>
  memref.copy %1, %3 : memref<*xi1> to memref<*xi1>
  // CHECK: [[ONE:%.*]] = llvm.mlir.constant(1 : index) : i64
  // CHECK: [[ALLOCA:%.*]] = llvm.alloca %35 x !llvm.struct<(ptr<i1>, ptr<i1>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i1>, ptr<i1>, i64, array<1 x i64>, array<1 x i64>)>>
  // CHECK: llvm.store {{%.*}}, [[ALLOCA]] : !llvm.ptr<struct<(ptr<i1>, ptr<i1>, i64, array<1 x i64>, array<1 x i64>)>>
  // CHECK: [[BITCAST:%.*]] = llvm.bitcast [[ALLOCA]] : !llvm.ptr<struct<(ptr<i1>, ptr<i1>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
  // CHECK: [[RANK:%.*]] = llvm.mlir.constant(1 : index) : i64
  // CHECK: [[UNDEF:%.*]] = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
  // CHECK: [[INSERT:%.*]] = llvm.insertvalue [[RANK]], [[UNDEF]][0] : !llvm.struct<(i64, ptr<i8>)>
  // CHECK: [[INSERT2:%.*]] = llvm.insertvalue [[BITCAST]], [[INSERT]][1] : !llvm.struct<(i64, ptr<i8>)>
  // CHECK: [[RANK2:%.*]] = llvm.mlir.constant(1 : index) : i64
  // CHECK: [[ALLOCA2:%.*]] = llvm.alloca [[RANK2]] x !llvm.struct<(i64, ptr<i8>)> : (i64) -> !llvm.ptr<struct<(i64, ptr<i8>)>>
  // CHECK: llvm.store {{%.*}}, [[ALLOCA2]] : !llvm.ptr<struct<(i64, ptr<i8>)>>
  // CHECK: [[ALLOCA3:%.*]] = llvm.alloca [[RANK2]] x !llvm.struct<(i64, ptr<i8>)> : (i64) -> !llvm.ptr<struct<(i64, ptr<i8>)>>
  // CHECK: llvm.store [[INSERT2]], [[ALLOCA3]] : !llvm.ptr<struct<(i64, ptr<i8>)>>
  // CHECK: [[SIZE:%.*]] = llvm.mlir.constant(1 : index) : i64
  // CHECK: llvm.call @memrefCopy([[SIZE]], [[ALLOCA2]], [[ALLOCA3]]) : (i64, !llvm.ptr<struct<(i64, ptr<i8>)>>, !llvm.ptr<struct<(i64, ptr<i8>)>>) -> ()
  return
}
