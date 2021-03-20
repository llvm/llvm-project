// RUN: mlir-opt -legalize-std-for-spirv -verify-diagnostics %s -o - | FileCheck %s

// CHECK-LABEL: @fold_static_stride_subview_with_load
// CHECK-SAME: [[ARG0:%.*]]: memref<12x32xf32>, [[ARG1:%.*]]: index, [[ARG2:%.*]]: index, [[ARG3:%.*]]: index, [[ARG4:%.*]]: index
func @fold_static_stride_subview_with_load(%arg0 : memref<12x32xf32>, %arg1 : index, %arg2 : index, %arg3 : index, %arg4 : index) -> f32 {
  // CHECK-NOT: memref.subview
  // CHECK: [[C2:%.*]] = constant 2 : index
  // CHECK: [[C3:%.*]] = constant 3 : index
  // CHECK: [[STRIDE1:%.*]] = muli [[ARG3]], [[C2]] : index
  // CHECK: [[INDEX1:%.*]] = addi [[ARG1]], [[STRIDE1]] : index
  // CHECK: [[STRIDE2:%.*]] = muli [[ARG4]], [[C3]] : index
  // CHECK: [[INDEX2:%.*]] = addi [[ARG2]], [[STRIDE2]] : index
  // CHECK: memref.load [[ARG0]]{{\[}}[[INDEX1]], [[INDEX2]]{{\]}}
  %0 = memref.subview %arg0[%arg1, %arg2][4, 4][2, 3] : memref<12x32xf32> to memref<4x4xf32, offset:?, strides: [64, 3]>
  %1 = memref.load %0[%arg3, %arg4] : memref<4x4xf32, offset:?, strides: [64, 3]>
  return %1 : f32
}

// CHECK-LABEL: @fold_dynamic_stride_subview_with_load
// CHECK-SAME: [[ARG0:%.*]]: memref<12x32xf32>, [[ARG1:%.*]]: index, [[ARG2:%.*]]: index, [[ARG3:%.*]]: index, [[ARG4:%.*]]: index, [[ARG5:%.*]]: index, [[ARG6:%.*]]: index
func @fold_dynamic_stride_subview_with_load(%arg0 : memref<12x32xf32>, %arg1 : index, %arg2 : index, %arg3 : index, %arg4 : index, %arg5 : index, %arg6 : index) -> f32 {
  // CHECK-NOT: memref.subview
  // CHECK: [[STRIDE1:%.*]] = muli [[ARG3]], [[ARG5]] : index
  // CHECK: [[INDEX1:%.*]] = addi [[ARG1]], [[STRIDE1]] : index
  // CHECK: [[STRIDE2:%.*]] = muli [[ARG4]], [[ARG6]] : index
  // CHECK: [[INDEX2:%.*]] = addi [[ARG2]], [[STRIDE2]] : index
  // CHECK: memref.load [[ARG0]]{{\[}}[[INDEX1]], [[INDEX2]]{{\]}}
  %0 = memref.subview %arg0[%arg1, %arg2][4, 4][%arg5, %arg6] :
    memref<12x32xf32> to memref<4x4xf32, offset:?, strides: [?, ?]>
  %1 = memref.load %0[%arg3, %arg4] : memref<4x4xf32, offset:?, strides: [?, ?]>
  return %1 : f32
}

// CHECK-LABEL: @fold_static_stride_subview_with_store
// CHECK-SAME: [[ARG0:%.*]]: memref<12x32xf32>, [[ARG1:%.*]]: index, [[ARG2:%.*]]: index, [[ARG3:%.*]]: index, [[ARG4:%.*]]: index, [[ARG5:%.*]]: f32
func @fold_static_stride_subview_with_store(%arg0 : memref<12x32xf32>, %arg1 : index, %arg2 : index, %arg3 : index, %arg4 : index, %arg5 : f32) {
  // CHECK-NOT: memref.subview
  // CHECK: [[C2:%.*]] = constant 2 : index
  // CHECK: [[C3:%.*]] = constant 3 : index
  // CHECK: [[STRIDE1:%.*]] = muli [[ARG3]], [[C2]] : index
  // CHECK: [[INDEX1:%.*]] = addi [[ARG1]], [[STRIDE1]] : index
  // CHECK: [[STRIDE2:%.*]] = muli [[ARG4]], [[C3]] : index
  // CHECK: [[INDEX2:%.*]] = addi [[ARG2]], [[STRIDE2]] : index
  // CHECK: memref.store [[ARG5]], [[ARG0]]{{\[}}[[INDEX1]], [[INDEX2]]{{\]}}
  %0 = memref.subview %arg0[%arg1, %arg2][4, 4][2, 3] :
    memref<12x32xf32> to memref<4x4xf32, offset:?, strides: [64, 3]>
  memref.store %arg5, %0[%arg3, %arg4] : memref<4x4xf32, offset:?, strides: [64, 3]>
  return
}

// CHECK-LABEL: @fold_dynamic_stride_subview_with_store
// CHECK-SAME: [[ARG0:%.*]]: memref<12x32xf32>, [[ARG1:%.*]]: index, [[ARG2:%.*]]: index, [[ARG3:%.*]]: index, [[ARG4:%.*]]: index, [[ARG5:%.*]]: index, [[ARG6:%.*]]: index, [[ARG7:%.*]]: f32
func @fold_dynamic_stride_subview_with_store(%arg0 : memref<12x32xf32>, %arg1 : index, %arg2 : index, %arg3 : index, %arg4 : index, %arg5 : index, %arg6 : index, %arg7 : f32) {
  // CHECK-NOT: memref.subview
  // CHECK: [[STRIDE1:%.*]] = muli [[ARG3]], [[ARG5]] : index
  // CHECK: [[INDEX1:%.*]] = addi [[ARG1]], [[STRIDE1]] : index
  // CHECK: [[STRIDE2:%.*]] = muli [[ARG4]], [[ARG6]] : index
  // CHECK: [[INDEX2:%.*]] = addi [[ARG2]], [[STRIDE2]] : index
  // CHECK: memref.store [[ARG7]], [[ARG0]]{{\[}}[[INDEX1]], [[INDEX2]]{{\]}}
  %0 = memref.subview %arg0[%arg1, %arg2][4, 4][%arg5, %arg6] :
    memref<12x32xf32> to memref<4x4xf32, offset:?, strides: [?, ?]>
  memref.store %arg7, %0[%arg3, %arg4] : memref<4x4xf32, offset:?, strides: [?, ?]>
  return
}

// CHECK-LABEL: @fold_static_stride_subview_with_transfer_read
// CHECK-SAME: [[ARG0:%.*]]: memref<12x32xf32>, [[ARG1:%.*]]: index, [[ARG2:%.*]]: index, [[ARG3:%.*]]: index, [[ARG4:%.*]]: index
func @fold_static_stride_subview_with_transfer_read(%arg0 : memref<12x32xf32>, %arg1 : index, %arg2 : index, %arg3 : index, %arg4 : index) -> vector<4xf32> {
  // CHECK-NOT: memref.subview
  // CHECK-DAG: [[F1:%.*]] = constant 1.000000e+00 : f32
  // CHECK-DAG: [[C2:%.*]] = constant 2 : index
  // CHECK-DAG: [[C3:%.*]] = constant 3 : index
  // CHECK: [[STRIDE1:%.*]] = muli [[ARG3]], [[C2]] : index
  // CHECK: [[INDEX1:%.*]] = addi [[ARG1]], [[STRIDE1]] : index
  // CHECK: [[STRIDE2:%.*]] = muli [[ARG4]], [[C3]] : index
  // CHECK: [[INDEX2:%.*]] = addi [[ARG2]], [[STRIDE2]] : index
  // CHECK: vector.transfer_read [[ARG0]]{{\[}}[[INDEX1]], [[INDEX2]]{{\]}}, [[F1]] {masked = [false]}
  %f1 = constant 1.0 : f32
  %0 = memref.subview %arg0[%arg1, %arg2][4, 4][2, 3] : memref<12x32xf32> to memref<4x4xf32, offset:?, strides: [64, 3]>
  %1 = vector.transfer_read %0[%arg3, %arg4], %f1 {masked = [false]} : memref<4x4xf32, offset:?, strides: [64, 3]>, vector<4xf32>
  return %1 : vector<4xf32>
}

// CHECK-LABEL: @fold_static_stride_subview_with_transfer_write
// CHECK-SAME: [[ARG0:%.*]]: memref<12x32xf32>, [[ARG1:%.*]]: index, [[ARG2:%.*]]: index, [[ARG3:%.*]]: index, [[ARG4:%.*]]: index, [[ARG5:%.*]]: vector<4xf32>
func @fold_static_stride_subview_with_transfer_write(%arg0 : memref<12x32xf32>, %arg1 : index, %arg2 : index, %arg3 : index, %arg4 : index, %arg5 : vector<4xf32>) {
  // CHECK-NOT: memref.subview
  // CHECK: [[C2:%.*]] = constant 2 : index
  // CHECK: [[C3:%.*]] = constant 3 : index
  // CHECK: [[STRIDE1:%.*]] = muli [[ARG3]], [[C2]] : index
  // CHECK: [[INDEX1:%.*]] = addi [[ARG1]], [[STRIDE1]] : index
  // CHECK: [[STRIDE2:%.*]] = muli [[ARG4]], [[C3]] : index
  // CHECK: [[INDEX2:%.*]] = addi [[ARG2]], [[STRIDE2]] : index
  // CHECK: vector.transfer_write [[ARG5]], [[ARG0]]{{\[}}[[INDEX1]], [[INDEX2]]{{\]}} {masked = [false]}
  %0 = memref.subview %arg0[%arg1, %arg2][4, 4][2, 3] :
    memref<12x32xf32> to memref<4x4xf32, offset:?, strides: [64, 3]>
  vector.transfer_write %arg5, %0[%arg3, %arg4] {masked = [false]} : vector<4xf32>, memref<4x4xf32, offset:?, strides: [64, 3]>
  return
}
