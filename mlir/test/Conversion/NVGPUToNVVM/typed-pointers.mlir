// RUN: mlir-opt --convert-nvgpu-to-nvvm='use-opaque-pointers=0' --split-input-file %s | FileCheck %s

// CHECK-LABEL: @async_cp(
// CHECK-SAME: %[[IDX:[a-zA-Z0-9_]+]]: index)
func.func @async_cp(
  %src: memref<128x128xf32>, %dst: memref<3x16x128xf32, 3>, %i : index) {
  // CHECK: %[[IDX1:.*]] = builtin.unrealized_conversion_cast %[[IDX]] : index to i64
  // CHECK-DAG: %[[BASEDST:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<f32, 3>, ptr<f32, 3>, i64, array<3 x i64>, array<3 x i64>)>
  // CHECK-DAG: %[[S0:.*]] = llvm.mlir.constant(2048 : index) : i64
  // CHECK-DAG: %[[LI:.*]] = llvm.mul %[[IDX1]], %[[S0]] : i64
  // CHECK-DAG: %[[S1:.*]] = llvm.mlir.constant(128 : index) : i64
  // CHECK-DAG: %[[FI0:.*]] = llvm.mul %[[IDX1]], %[[S1]] : i64
  // CHECK-DAG: %[[FI1:.*]] = llvm.add %[[LI]], %[[FI0]] : i64
  // CHECK-DAG: %[[FI2:.*]] = llvm.add %[[FI1]], %[[IDX1]] : i64
  // CHECK-DAG: %[[ADDRESSDST:.*]] = llvm.getelementptr %[[BASEDST]][%[[FI2]]] : (!llvm.ptr<f32, 3>, i64) -> !llvm.ptr<f32, 3>
  // CHECK-DAG: %[[CAST0:.*]] = llvm.bitcast %[[ADDRESSDST]] : !llvm.ptr<f32, 3> to !llvm.ptr<i8, 3>
  // CHECK-DAG: %[[BASESRC:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK-DAG: %[[S3:.*]] = llvm.mlir.constant(128 : index) : i64
  // CHECK-DAG: %[[FI3:.*]] = llvm.mul %[[IDX1]], %[[S3]]  : i64
  // CHECK-DAG: %[[FI4:.*]] = llvm.add %[[FI3]], %[[IDX1]]  : i64
  // CHECK-DAG: %[[ADDRESSSRC:.*]] = llvm.getelementptr %[[BASESRC]][%[[FI4]]] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
  // CHECK-DAG: %[[CAST1:.*]] = llvm.bitcast %[[ADDRESSSRC]] : !llvm.ptr<f32> to !llvm.ptr<i8>
  // CHECK-DAG: %[[CAST2:.*]] = llvm.addrspacecast %[[CAST1]] : !llvm.ptr<i8> to !llvm.ptr<i8, 1>
  // CHECK-DAG: nvvm.cp.async.shared.global %[[CAST0]], %[[CAST2]], 16, cache = ca
  %0 = nvgpu.device_async_copy %src[%i, %i], %dst[%i, %i, %i], 4 : memref<128x128xf32> to memref<3x16x128xf32, 3>
  // CHECK: nvvm.cp.async.commit.group
  %1 = nvgpu.device_async_create_group %0
  // CHECK: nvvm.cp.async.wait.group 1
  nvgpu.device_async_wait %1 { numGroups = 1 : i32 }

  // CHECK: nvvm.cp.async.shared.global %{{.*}}, %{{.*}}, 16, cache = cg
  %2 = nvgpu.device_async_copy %src[%i, %i], %dst[%i, %i, %i], 4 {bypassL1}: memref<128x128xf32> to memref<3x16x128xf32, 3>
  return
}

// -----

// CHECK-LABEL: @async_cp_i4(
// CHECK-SAME: %[[IDX:[a-zA-Z0-9_]+]]: index)
func.func @async_cp_i4(
  %src: memref<128x64xi4>, %dst: memref<128x128xi4, 3>, %i : index) -> !nvgpu.device.async.token {
  // CHECK: %[[IDX1:.*]] = builtin.unrealized_conversion_cast %[[IDX]] : index to i64
  // CHECK-DAG: %[[BASEDST:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<i4, 3>, ptr<i4, 3>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK-DAG: %[[S0:.*]] = llvm.mlir.constant(128 : index) : i64
  // CHECK-DAG: %[[LI:.*]] = llvm.mul %[[IDX1]], %[[S0]] : i64
  // CHECK-DAG: %[[FI1:.*]] = llvm.add %[[LI]], %[[IDX1]] : i64
  // CHECK-DAG: %[[ADDRESSDST:.*]] = llvm.getelementptr %[[BASEDST]][%[[FI1]]] : (!llvm.ptr<i4, 3>, i64) -> !llvm.ptr<i4, 3>
  // CHECK-DAG: %[[CAST0:.*]] = llvm.bitcast %[[ADDRESSDST]] : !llvm.ptr<i4, 3> to !llvm.ptr<i8, 3>
  // CHECK-DAG: %[[BASESRC:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<i4>, ptr<i4>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK-DAG: %[[S2:.*]] = llvm.mlir.constant(64 : index) : i64
  // CHECK-DAG: %[[FI2:.*]] = llvm.mul %[[IDX1]], %[[S2]]  : i64
  // CHECK-DAG: %[[FI3:.*]] = llvm.add %[[FI2]], %[[IDX1]]  : i64
  // CHECK-DAG: %[[ADDRESSSRC:.*]] = llvm.getelementptr %[[BASESRC]][%[[FI3]]] : (!llvm.ptr<i4>, i64) -> !llvm.ptr<i4>
  // CHECK-DAG: %[[CAST1:.*]] = llvm.bitcast %[[ADDRESSSRC]] : !llvm.ptr<i4> to !llvm.ptr<i8>
  // CHECK-DAG: %[[CAST2:.*]] = llvm.addrspacecast %[[CAST1]] : !llvm.ptr<i8> to !llvm.ptr<i8, 1>
  // CHECK-DAG: nvvm.cp.async.shared.global %[[CAST0]], %[[CAST2]], 16, cache = ca
  %0 = nvgpu.device_async_copy %src[%i, %i], %dst[%i, %i], 32 : memref<128x64xi4> to memref<128x128xi4, 3>
  return %0 : !nvgpu.device.async.token
}
