// RUN: mlir-opt -legalize-std-for-spirv -convert-std-to-spirv %s -o - | FileCheck %s

// TODO: For these examples running these passes separately produces
// the desired output. Adding all of patterns within a single pass does
// not seem to work.

//===----------------------------------------------------------------------===//
// std.subview
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fold_static_stride_subview_with_load
// CHECK-SAME: [[ARG0:%.*]]: !spv.ptr<!spv.struct<!spv.array<384 x f32 [4]> [0]>, StorageBuffer>, [[ARG1:%.*]]: i32, [[ARG2:%.*]]: i32, [[ARG3:%.*]]: i32, [[ARG4:%.*]]: i32
func @fold_static_stride_subview_with_load(%arg0 : memref<12x32xf32>, %arg1 : index, %arg2 : index, %arg3 : index, %arg4 : index) {
  // CHECK: [[C2:%.*]] = spv.constant 2
  // CHECK: [[C3:%.*]] = spv.constant 3
  // CHECK: [[T2:%.*]] = spv.IMul [[ARG3]], [[C2]]
  // CHECK: [[T3:%.*]] = spv.IAdd [[ARG1]], [[T2]]
  // CHECK: [[T4:%.*]] = spv.IMul [[ARG4]], [[C3]]
  // CHECK: [[T5:%.*]] = spv.IAdd [[ARG2]], [[T4]]
  // CHECK: [[C32:%.*]] = spv.constant 32
  // CHECK: [[T7:%.*]] = spv.IMul [[C32]], [[T3]]
  // CHECK: [[C1:%.*]] = spv.constant 1
  // CHECK: [[T9:%.*]] = spv.IMul [[C1]], [[T5]]
  // CHECK: [[T10:%.*]] = spv.IAdd [[T7]], [[T9]]
  // CHECK: [[C0:%.*]] = spv.constant 0
  // CHECK: [[T12:%.*]] = spv.AccessChain [[ARG0]]{{\[}}[[C0]], [[T10]]
  // CHECK: spv.Load "StorageBuffer" [[T12]] : f32
  %0 = subview %arg0[%arg1, %arg2][][] : memref<12x32xf32> to memref<4x4xf32, offset:?, strides: [64, 3]>
  %1 = load %0[%arg3, %arg4] : memref<4x4xf32, offset:?, strides: [64, 3]>
  return
}

// CHECK-LABEL: @fold_static_stride_subview_with_store
// CHECK-SAME: [[ARG0:%.*]]: !spv.ptr<!spv.struct<!spv.array<384 x f32 [4]> [0]>, StorageBuffer>, [[ARG1:%.*]]: i32, [[ARG2:%.*]]: i32, [[ARG3:%.*]]: i32, [[ARG4:%.*]]: i32, [[ARG5:%.*]]: f32
func @fold_static_stride_subview_with_store(%arg0 : memref<12x32xf32>, %arg1 : index, %arg2 : index, %arg3 : index, %arg4 : index, %arg5 : f32) {
  // CHECK: [[C2:%.*]] = spv.constant 2
  // CHECK: [[C3:%.*]] = spv.constant 3
  // CHECK: [[T2:%.*]] = spv.IMul [[ARG3]], [[C2]]
  // CHECK: [[T3:%.*]] = spv.IAdd [[ARG1]], [[T2]]
  // CHECK: [[T4:%.*]] = spv.IMul [[ARG4]], [[C3]]
  // CHECK: [[T5:%.*]] = spv.IAdd [[ARG2]], [[T4]]
  // CHECK: [[C32:%.*]] = spv.constant 32
  // CHECK: [[T7:%.*]] = spv.IMul [[C32]], [[T3]]
  // CHECK: [[C1:%.*]] = spv.constant 1
  // CHECK: [[T9:%.*]] = spv.IMul [[C1]], [[T5]]
  // CHECK: [[T10:%.*]] = spv.IAdd [[T7]], [[T9]]
  // CHECK: [[C0:%.*]] = spv.constant 0
  // CHECK: [[T12:%.*]] = spv.AccessChain [[ARG0]]{{\[}}[[C0]], [[T10]]
  // CHECK: spv.Store "StorageBuffer" [[T12]], [[ARG5]] : f32
  %0 = subview %arg0[%arg1, %arg2][][] : memref<12x32xf32> to memref<4x4xf32, offset:?, strides: [64, 3]>
  store %arg5, %0[%arg3, %arg4] : memref<4x4xf32, offset:?, strides: [64, 3]>
  return
}
