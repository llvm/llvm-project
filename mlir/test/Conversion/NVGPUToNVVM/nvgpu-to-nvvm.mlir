// RUN: mlir-opt --convert-nvgpu-to-nvvm='use-opaque-pointers=1' --split-input-file %s | FileCheck %s

// CHECK-LABEL: @m16n8k16_fp16
func.func @m16n8k16_fp16(%arg0: vector<4x2xf16>, %arg1: vector<2x2xf16>, %arg2: vector<2x2xf16>) -> vector<2x2xf16> {
  // CHECK: llvm.extractvalue %{{.*}}[0] : !llvm.array<4 x vector<2xf16>>
  // CHECK: llvm.extractvalue %{{.*}}[1] : !llvm.array<4 x vector<2xf16>>
  // CHECK: llvm.extractvalue %{{.*}}[2] : !llvm.array<4 x vector<2xf16>>
  // CHECK: llvm.extractvalue %{{.*}}[3] : !llvm.array<4 x vector<2xf16>>
  // CHECK: llvm.extractvalue %{{.*}}[0] : !llvm.array<2 x vector<2xf16>>
  // CHECK: llvm.extractvalue %{{.*}}[1] : !llvm.array<2 x vector<2xf16>>
  // CHECK: llvm.extractvalue %{{.*}}[0] : !llvm.array<2 x vector<2xf16>>
  // CHECK: llvm.extractvalue %{{.*}}[1] : !llvm.array<2 x vector<2xf16>>
  // CHECK-NOT llvm.extractvalue
  // CHECK: [[d:%.+]] = nvvm.mma.sync
  // CHECK-SAME: shape = #nvvm.shape<m = 16, n = 8, k = 16>
  %d = nvgpu.mma.sync (%arg0, %arg1, %arg2) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
  // CHECK-DAG: llvm.extractvalue [[d]][0] : !llvm.struct<(vector<2xf16>, vector<2xf16>)>
  // CHECK-DAG: llvm.extractvalue [[d]][1] : !llvm.struct<(vector<2xf16>, vector<2xf16>)>
  // CHECK: llvm.mlir.undef : !llvm.array<2 x vector<2xf16>>
  // CHECK-DAG: llvm.insertvalue {{%.+}}, {{%.+}}[0] : !llvm.array<2 x vector<2xf16>>
  // CHECK-DAG: llvm.insertvalue {{%.+}}, {{%.+}}[1] : !llvm.array<2 x vector<2xf16>>
  return %d : vector<2x2xf16>
}

// -----

// Same as above but with fp32 acumulation type.

// CHECK-LABEL: @m16n8k16_fp16_fp32
func.func @m16n8k16_fp16_fp32(%arg0: vector<4x2xf16>, %arg1: vector<2x2xf16>, %arg2: vector<2x2xf32>) -> vector<2x2xf32> {
  // We just need to check the mma instruction and the manipulatin of the result.
  // CHECK: [[d:%.+]] = nvvm.mma.sync
  // CHECK-SAME: shape = #nvvm.shape<m = 16, n = 8, k = 16>
  // CHECK-SAME: (vector<2xf16>, vector<2xf16>, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  %d = nvgpu.mma.sync (%arg0, %arg1, %arg2) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
  // CHECK: [[undef:%.+]] = llvm.mlir.undef : vector<2xf32>
  // CHECK-DAG: llvm.extractvalue [[d]][0] : !llvm.struct<(f32, f32, f32, f32)>
  // CHECK-DAG: llvm.extractvalue [[d]][1] : !llvm.struct<(f32, f32, f32, f32)>
  // CHECK: [[d00:%.+]] = llvm.insertelement {{%.+}}, [[undef]][{{.*}}] : vector<2xf32>
  // CHECK: [[d01:%.+]] = llvm.insertelement {{%.+}}, [[d00]][{{.*}}] : vector<2xf32>

  // CHECK: [[undef:%.+]] = llvm.mlir.undef : vector<2xf32>
  // CHECK-DAG: llvm.extractvalue [[d]][2] : !llvm.struct<(f32, f32, f32, f32)>
  // CHECK-DAG: llvm.extractvalue [[d]][3] : !llvm.struct<(f32, f32, f32, f32)>
  // CHECK: [[d10:%.+]] = llvm.insertelement {{%.+}}, [[undef]][{{.*}}] : vector<2xf32>
  // CHECK: [[d11:%.+]] = llvm.insertelement {{%.+}}, [[d10]][{{.*}}] : vector<2xf32>

  // CHECK-DAG: llvm.insertvalue [[d01]], {{%.+}}[0] : !llvm.array<2 x vector<2xf32>>
  // CHECK-DAG: llvm.insertvalue [[d11]], {{%.+}}[1] : !llvm.array<2 x vector<2xf32>>
  return %d : vector<2x2xf32>
}

// -----

// CHECK-LABEL: @m16n8k8_fp16
func.func @m16n8k8_fp16(%arg0: vector<2x2xf16>, %arg1: vector<1x2xf16>, %arg2: vector<2x2xf16>) -> vector<2x2xf16> {
  // CHECK: llvm.extractvalue %{{.*}}[0] : !llvm.array<2 x vector<2xf16>>
  // CHECK: llvm.extractvalue %{{.*}}[1] : !llvm.array<2 x vector<2xf16>>
  // CHECK: llvm.extractvalue %{{.*}}[0] : !llvm.array<1 x vector<2xf16>>
  // CHECK: llvm.extractvalue %{{.*}}[0] : !llvm.array<2 x vector<2xf16>>
  // CHECK: llvm.extractvalue %{{.*}}[1] : !llvm.array<2 x vector<2xf16>>
  // CHECK-NOT llvm.extractvalue
  // CHECK: [[d:%.+]] = nvvm.mma.sync
  // CHECK-SAME: shape = #nvvm.shape<m = 16, n = 8, k = 8>
  %d = nvgpu.mma.sync (%arg0, %arg1, %arg2) {mmaShape = [16, 8, 8]} : (vector<2x2xf16>, vector<1x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
  // CHECK-DAG: llvm.extractvalue [[d]][0] : !llvm.struct<(vector<2xf16>, vector<2xf16>)>
  // CHECK-DAG: llvm.extractvalue [[d]][1] : !llvm.struct<(vector<2xf16>, vector<2xf16>)>
  // CHECK: llvm.mlir.undef : !llvm.array<2 x vector<2xf16>>
  // CHECK-DAG: llvm.insertvalue {{%.+}}, {{%.+}}[0] : !llvm.array<2 x vector<2xf16>>
  // CHECK-DAG: llvm.insertvalue {{%.+}}, {{%.+}}[1] : !llvm.array<2 x vector<2xf16>>
  // CHECK: return
  return %d : vector<2x2xf16>
}

// -----


// CHECK-LABEL: @m16n8k32_int8
func.func @m16n8k32_int8(%arg0: vector<4x4xi8>, %arg1: vector<2x4xi8>, %arg2: vector<2x2xi32>) -> vector<2x2xi32> {
  // CHECK: [[el:%.+]] = llvm.extractvalue %{{.*}}[{{.*}}] : !llvm.array<4 x vector<4xi8>>
  // CHECK: llvm.bitcast [[el]] : vector<4xi8> to i32
  // CHECK: [[el:%.+]] = llvm.extractvalue %{{.*}}[{{.*}}] : !llvm.array<4 x vector<4xi8>>
  // CHECK: llvm.bitcast [[el]] : vector<4xi8> to i32
  // CHECK: [[el:%.+]] = llvm.extractvalue %{{.*}}[{{.*}}] : !llvm.array<4 x vector<4xi8>>
  // CHECK: llvm.bitcast [[el]] : vector<4xi8> to i32
  // CHECK: [[el:%.+]] = llvm.extractvalue %{{.*}}[{{.*}}] : !llvm.array<4 x vector<4xi8>>
  // CHECK: llvm.bitcast [[el]] : vector<4xi8> to i32
  // CHECK: [[el:%.+]] = llvm.extractvalue %{{.*}}[{{.*}}] : !llvm.array<2 x vector<4xi8>>
  // CHECK: llvm.bitcast [[el]] : vector<4xi8> to i32
  // CHECK: [[el:%.+]] = llvm.extractvalue %{{.*}}[{{.*}}] : !llvm.array<2 x vector<4xi8>>
  // CHECK: llvm.bitcast [[el]] : vector<4xi8> to i32
  // CHECK: [[el:%.+]] = llvm.extractvalue %{{.*}}[{{.*}}] : !llvm.array<2 x vector<2xi32>>
  // CHECK: [[el:%.+]] = llvm.extractvalue %{{.*}}[{{.*}}] : !llvm.array<2 x vector<2xi32>>
  // CHECK: [[d:%.+]] = nvvm.mma.sync
  // CHECK-SAME: intOverflowBehavior = #nvvm.mma_int_overflow<satfinite>
  // CHECK-SAME: multiplicandAPtxType = #nvvm.mma_type<s8>
  // CHECK-SAME: multiplicandBPtxType = #nvvm.mma_type<s8>
  // CHECK-SAME: shape = #nvvm.shape<m = 16, n = 8, k = 32>
  %d = nvgpu.mma.sync (%arg0, %arg1, %arg2) {mmaShape = [16, 8, 32]} : (vector<4x4xi8>, vector<2x4xi8>, vector<2x2xi32>) -> vector<2x2xi32>
  return %d : vector<2x2xi32>
}

// -----

// CHECK-LABEL: @m16n8k32_i4
func.func @m16n8k32_i4(%arg0: vector<2x8xi4>, %arg1: vector<1x8xi4>, %arg2: vector<2x2xi32>) -> vector<2x2xi32> {
  // CHECK: [[el:%.+]] = llvm.extractvalue %{{.*}}[{{.*}}] : !llvm.array<2 x vector<8xi4>>
  // CHECK: llvm.bitcast [[el]] : vector<8xi4> to i32
  // CHECK: [[el:%.+]] = llvm.extractvalue %{{.*}}[{{.*}}] : !llvm.array<2 x vector<8xi4>>
  // CHECK: llvm.bitcast [[el]] : vector<8xi4> to i32
  // CHECK: [[el:%.+]] = llvm.extractvalue %{{.*}}[{{.*}}] : !llvm.array<1 x vector<8xi4>>
  // CHECK: llvm.bitcast [[el]] : vector<8xi4> to i32
  // CHECK: [[el:%.+]] = llvm.extractvalue %{{.*}}[{{.*}}] : !llvm.array<2 x vector<2xi32>>
  // CHECK: [[el:%.+]] = llvm.extractvalue %{{.*}}[{{.*}}] : !llvm.array<2 x vector<2xi32>>
  // CHECK: [[d:%.+]] = nvvm.mma.sync
  // CHECK-SAME: intOverflowBehavior = #nvvm.mma_int_overflow<satfinite>
  // CHECK-SAME: multiplicandAPtxType = #nvvm.mma_type<s4>
  // CHECK-SAME: multiplicandBPtxType = #nvvm.mma_type<s4>
  // CHECK-SAME: shape = #nvvm.shape<m = 16, n = 8, k = 32>
  %d = nvgpu.mma.sync (%arg0, %arg1, %arg2) {mmaShape = [16, 8, 32]} : (vector<2x8xi4>, vector<1x8xi4>, vector<2x2xi32>) -> vector<2x2xi32>
  return %d : vector<2x2xi32>
}

// -----

// CHECK-LABEL: @m16n8k64_i4
func.func @m16n8k64_i4(%arg0: vector<4x8xi4>, %arg1: vector<2x8xi4>, %arg2: vector<2x2xi32>) -> vector<2x2xi32> {
  // CHECK: [[el:%.+]] = llvm.extractvalue %{{.*}}[{{.*}}] : !llvm.array<4 x vector<8xi4>>
  // CHECK: llvm.bitcast [[el]] : vector<8xi4> to i32
  // CHECK: [[el:%.+]] = llvm.extractvalue %{{.*}}[{{.*}}] : !llvm.array<4 x vector<8xi4>>
  // CHECK: llvm.bitcast [[el]] : vector<8xi4> to i32
  // CHECK: [[el:%.+]] = llvm.extractvalue %{{.*}}[{{.*}}] : !llvm.array<4 x vector<8xi4>>
  // CHECK: llvm.bitcast [[el]] : vector<8xi4> to i32
  // CHECK: [[el:%.+]] = llvm.extractvalue %{{.*}}[{{.*}}] : !llvm.array<4 x vector<8xi4>>
  // CHECK: llvm.bitcast [[el]] : vector<8xi4> to i32
  // CHECK: [[el:%.+]] = llvm.extractvalue %{{.*}}[{{.*}}] : !llvm.array<2 x vector<8xi4>>
  // CHECK: llvm.bitcast [[el]] : vector<8xi4> to i32
  // CHECK: [[el:%.+]] = llvm.extractvalue %{{.*}}[{{.*}}] : !llvm.array<2 x vector<8xi4>>
  // CHECK: llvm.bitcast [[el]] : vector<8xi4> to i32
  // CHECK: [[el:%.+]] = llvm.extractvalue %{{.*}}[{{.*}}] : !llvm.array<2 x vector<2xi32>>
  // CHECK: [[el:%.+]] = llvm.extractvalue %{{.*}}[{{.*}}] : !llvm.array<2 x vector<2xi32>>
  // CHECK: [[d:%.+]] = nvvm.mma.sync
  // CHECK-SAME: intOverflowBehavior = #nvvm.mma_int_overflow<satfinite>
  // CHECK-SAME: multiplicandAPtxType = #nvvm.mma_type<s4>
  // CHECK-SAME: multiplicandBPtxType = #nvvm.mma_type<s4>
  // CHECK-SAME: shape = #nvvm.shape<m = 16, n = 8, k = 64>
  %d = nvgpu.mma.sync (%arg0, %arg1, %arg2) {mmaShape = [16, 8, 64]} : (vector<4x8xi4>, vector<2x8xi4>, vector<2x2xi32>) -> vector<2x2xi32>
  return %d : vector<2x2xi32>
}

// -----

// CHECK-LABEL: @m8n8k4_f64
func.func @m8n8k4_f64(%arg0: vector<1x1xf64>, %arg1: vector<1x1xf64>, %arg2: vector<1x2xf64>) -> vector<1x2xf64> {
  // CHECK: llvm.extractvalue
  // CHECK: llvm.extractvalue
  // CHECK: llvm.extractvalue
  // CHECK: [[d:%.+]] = nvvm.mma.sync A[{{%.+}}] B[{{%.+}}] C[{{%.+}}, {{%.+}}]
  // CHECK-SAME: shape = #nvvm.shape<m = 8, n = 8, k = 4>
  %d = nvgpu.mma.sync (%arg0, %arg1, %arg2) {mmaShape = [8, 8, 4]} : (vector<1x1xf64>, vector<1x1xf64>, vector<1x2xf64>) -> vector<1x2xf64>
  // CHECK: llvm.mlir.undef : vector<2xf64>
  // CHECK-DAG: llvm.extractvalue [[d]][0] : !llvm.struct<(f64, f64)>
  // CHECK-DAG: llvm.extractvalue [[d]][1] : !llvm.struct<(f64, f64)>
  // CHECK-COUNT-2: llvm.insertelement {{.*}} : vector<2xf64>
  // CHECK-DAG: llvm.insertvalue {{%.+}}, {{%.+}}[0] : !llvm.array<1 x vector<2xf64>>
  // CHECK: return
  return %d : vector<1x2xf64>
}

// -----


// CHECK-LABEL: @ldmatrix_x4
func.func @ldmatrix_x4(%arg0: memref<128x128xf16, 3>) ->  vector<4x2xf16> {
  %c0  = arith.constant 0 : index
  // CHECK: nvvm.ldmatrix {{%.+}} {layout = #nvvm.mma_layout<row>, num = 4 : i32} {{.*}} -> !llvm.struct<(i32, i32, i32, i32)
  %a = nvgpu.ldmatrix %arg0[%c0, %c0] {transpose = false, numTiles = 4 : i32} : memref<128x128xf16, 3> -> vector<4x2xf16>
  // CHECK: llvm.extractvalue
  // CHECK: llvm.bitcast
  // CHECK: llvm.insertvalue
  // CHECK: llvm.extractvalue
  // CHECK: llvm.bitcast
  // CHECK: llvm.insertvalue
  // CHECK: llvm.extractvalue
  // CHECK: llvm.bitcast
  // CHECK: llvm.insertvalue
  // CHECK: llvm.extractvalue
  // CHECK: llvm.bitcast
  // CHECK: llvm.insertvalue
  return %a : vector<4x2xf16>
}

// -----

// CHECK-LABEL: @ldmatrix_x1
func.func @ldmatrix_x1(%arg0: memref<128x128xf16, 3>) ->  vector<1x2xf16> {
  %c0  = arith.constant 0 : index
  // CHECK: nvvm.ldmatrix {{%.+}} {layout = #nvvm.mma_layout<row>, num = 1 : i32} {{.*}} -> i32
  %a = nvgpu.ldmatrix %arg0[%c0, %c0] {transpose = false, numTiles = 1 : i32} : memref<128x128xf16, 3> -> vector<1x2xf16>
  // CHECK: llvm.bitcast
  // CHECK: llvm.insertvalue
  return %a : vector<1x2xf16>
}

// -----

// CHECK-LABEL: @m16n8k4_tf32
func.func @m16n8k4_tf32(%arg0: vector<2x1xf32>, %arg1: vector<1x1xf32>, %arg2: vector<2x2xf32>) -> vector<2x2xf32> {
  // The A, B operand should be bitcast to i32
  // CHECK: llvm.extractvalue
  // CHECK: llvm.bitcast {{.*}} : vector<1xf32> to i32
  // CHECK: llvm.extractvalue
  // CHECK: llvm.bitcast {{.*}} : vector<1xf32> to i32
  // CHECK: llvm.extractvalue
  // CHECK: llvm.bitcast {{.*}} : vector<1xf32> to i32

  // CHECK: [[d:%.+]] = nvvm.mma.sync A[{{%.+}}, {{%.+}}] B[{{%.+}}] C[{{%.+}}, {{%.+}}, {{%.+}}, {{%.+}}]
  // CHECK-SAME: multiplicandAPtxType = #nvvm.mma_type<tf32>
  // CHECK-SAME: multiplicandBPtxType = #nvvm.mma_type<tf32>
  // CHECK-SAME: shape = #nvvm.shape<m = 16, n = 8, k = 4>
  // CHECK-SAME: -> !llvm.struct<(f32, f32, f32, f32)>
  %d = nvgpu.mma.sync (%arg0, %arg1, %arg2) {mmaShape = [16, 8, 4], tf32Enabled} : (vector<2x1xf32>, vector<1x1xf32>, vector<2x2xf32>) -> vector<2x2xf32>
  // CHECK: [[undef:%.+]] = llvm.mlir.undef : vector<2xf32>
  // CHECK-DAG: llvm.extractvalue [[d]][0] : !llvm.struct<(f32, f32, f32, f32)>
  // CHECK-DAG: llvm.extractvalue [[d]][1] : !llvm.struct<(f32, f32, f32, f32)>
  // CHECK: [[d00:%.+]] = llvm.insertelement {{%.+}}, [[undef]][{{.*}}] : vector<2xf32>
  // CHECK: [[d01:%.+]] = llvm.insertelement {{%.+}}, [[d00]][{{.*}}] : vector<2xf32>

  // CHECK: [[undef:%.+]] = llvm.mlir.undef : vector<2xf32>
  // CHECK-DAG: llvm.extractvalue [[d]][2] : !llvm.struct<(f32, f32, f32, f32)>
  // CHECK-DAG: llvm.extractvalue [[d]][3] : !llvm.struct<(f32, f32, f32, f32)>
  // CHECK: [[d10:%.+]] = llvm.insertelement {{%.+}}, [[undef]][{{.*}}] : vector<2xf32>
  // CHECK: [[d11:%.+]] = llvm.insertelement {{%.+}}, [[d10]][{{.*}}] : vector<2xf32>

  // CHECK-DAG: llvm.insertvalue [[d01]], {{%.+}}[0] : !llvm.array<2 x vector<2xf32>>
  // CHECK-DAG: llvm.insertvalue [[d11]], {{%.+}}[1] : !llvm.array<2 x vector<2xf32>>
  return %d : vector<2x2xf32>
}

// -----

// CHECK-LABEL: @async_cp(
// CHECK-SAME: %[[IDX:[a-zA-Z0-9_]+]]: index)
func.func @async_cp(
  %src: memref<128x128xf32>, %dst: memref<3x16x128xf32, 3>, %i : index) {
  // CHECK: %[[IDX1:.*]] = builtin.unrealized_conversion_cast %[[IDX]] : index to i64
  // CHECK-DAG: %[[BASEDST:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<3 x i64>, array<3 x i64>)>
  // CHECK-DAG: %[[S0:.*]] = llvm.mlir.constant(2048 : index) : i64
  // CHECK-DAG: %[[LI:.*]] = llvm.mul %[[IDX1]], %[[S0]] : i64
  // CHECK-DAG: %[[S1:.*]] = llvm.mlir.constant(128 : index) : i64
  // CHECK-DAG: %[[FI0:.*]] = llvm.mul %[[IDX1]], %[[S1]] : i64
  // CHECK-DAG: %[[FI1:.*]] = llvm.add %[[LI]], %[[FI0]] : i64
  // CHECK-DAG: %[[FI2:.*]] = llvm.add %[[FI1]], %[[IDX1]] : i64
  // CHECK-DAG: %[[ADDRESSDST:.*]] = llvm.getelementptr %[[BASEDST]][%[[FI2]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>
  // CHECK-DAG: %[[BASESRC:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK-DAG: %[[S3:.*]] = llvm.mlir.constant(128 : index) : i64
  // CHECK-DAG: %[[FI3:.*]] = llvm.mul %[[IDX1]], %[[S3]]  : i64
  // CHECK-DAG: %[[FI4:.*]] = llvm.add %[[FI3]], %[[IDX1]]  : i64
  // CHECK-DAG: %[[ADDRESSSRC:.*]] = llvm.getelementptr %[[BASESRC]][%[[FI4]]] : (!llvm.ptr, i64) -> !llvm.ptr
  // CHECK-DAG: %[[CAST2:.*]] = llvm.addrspacecast %[[ADDRESSSRC]] : !llvm.ptr to !llvm.ptr<1>
  // CHECK-DAG: nvvm.cp.async.shared.global %[[ADDRESSDST]], %[[CAST2]], 16, cache = ca
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
  // CHECK-DAG: %[[BASEDST:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK-DAG: %[[S0:.*]] = llvm.mlir.constant(128 : index) : i64
  // CHECK-DAG: %[[LI:.*]] = llvm.mul %[[IDX1]], %[[S0]] : i64
  // CHECK-DAG: %[[FI1:.*]] = llvm.add %[[LI]], %[[IDX1]] : i64
  // CHECK-DAG: %[[ADDRESSDST:.*]] = llvm.getelementptr %[[BASEDST]][%[[FI1]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>
  // CHECK-DAG: %[[BASESRC:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK-DAG: %[[S2:.*]] = llvm.mlir.constant(64 : index) : i64
  // CHECK-DAG: %[[FI2:.*]] = llvm.mul %[[IDX1]], %[[S2]]  : i64
  // CHECK-DAG: %[[FI3:.*]] = llvm.add %[[FI2]], %[[IDX1]]  : i64
  // CHECK-DAG: %[[ADDRESSSRC:.*]] = llvm.getelementptr %[[BASESRC]][%[[FI3]]] : (!llvm.ptr, i64) -> !llvm.ptr
  // CHECK-DAG: %[[CAST2:.*]] = llvm.addrspacecast %[[ADDRESSSRC]] : !llvm.ptr to !llvm.ptr<1>
  // CHECK-DAG: nvvm.cp.async.shared.global %[[ADDRESSDST]], %[[CAST2]], 16, cache = ca
  %0 = nvgpu.device_async_copy %src[%i, %i], %dst[%i, %i], 32 : memref<128x64xi4> to memref<128x128xi4, 3>
  return %0 : !nvgpu.device.async.token
}

// -----

// CHECK-LABEL: @async_cp_zfill_f32_align4(
// CHECK-SAME: %[[IDX:[a-zA-Z0-9_]+]]: index, %[[SRCELEMENTS:[a-zA-Z0-9_]+]]: index
func.func @async_cp_zfill_f32_align4(
  %src: memref<128x128xf32>, %dst: memref<3x16x128xf32, 3>, %i : index, %srcElements : index) {
  // CHECK: %[[IDX1:.*]] = builtin.unrealized_conversion_cast %[[IDX]] : index to i64
  // CHECK: %[[SRC1:.*]] = builtin.unrealized_conversion_cast %[[SRCELEMENTS]] : index to i64
  // CHECK-DAG: %[[BASEDST:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<3 x i64>, array<3 x i64>)>                                   
  // CHECK-DAG: %[[S2048:.*]] = llvm.mlir.constant(2048 : index) : i64
  // CHECK-DAG: %[[LI1:.*]] = llvm.mul %[[IDX1]], %[[S2048]] : i64
  // CHECK-DAG: %[[S0:.*]] = llvm.mlir.constant(128 : index) : i64
  // CHECK-DAG: %[[LI:.*]] = llvm.mul %[[IDX1]], %[[S0]] : i64
  // CHECK-DAG: %[[FI1:.*]] = llvm.add %[[LI1]], %[[LI]] : i64
  // CHECK-DAG: %[[FI2:.*]] = llvm.add %[[FI1]], %[[IDX1]] : i64
  // CHECK-DAG: %[[ADDRESSDST:.*]] = llvm.getelementptr %[[BASEDST]][%[[FI2]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
  // CHECK-DAG: %[[BASESRC:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
  // CHECK-DAG: %[[S2:.*]] = llvm.mlir.constant(128 : index) : i64
  // CHECK-DAG: %[[FI2:.*]] = llvm.mul %[[IDX1]], %[[S2]]  : i64
  // CHECK-DAG: %[[FI3:.*]] = llvm.add %[[FI2]], %[[IDX1]]  : i64
  // CHECK-DAG: %[[ADDRESSSRC:.*]] = llvm.getelementptr %[[BASESRC]][%[[FI3]]] : (!llvm.ptr, i64) -> !llvm.ptr
  // CHECK-DAG: %[[CAST2:.*]] = llvm.addrspacecast %[[ADDRESSSRC]] : !llvm.ptr to !llvm.ptr<1>
  // CHECK-DAG: %[[c1:.*]] = llvm.mlir.constant(3 : i32) : i32
  // CHECK-DAG: %[[c2:.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK-DAG: %[[c3:.*]] = llvm.trunc %[[SRC1]] : i64 to i32  
  // CHECK-DAG: %[[c4:.*]] = llvm.mul %[[c2]], %[[c3]] : i32
  // CHECK-DAG: %[[c5:.*]] = llvm.lshr %[[c4]], %[[c1]] : i32
  // CHECK-DAG: nvvm.cp.async.shared.global %[[ADDRESSDST]], %[[CAST2]], 16, cache = cg, %[[c5]]
  %0 = nvgpu.device_async_copy %src[%i, %i], %dst[%i, %i, %i], 4, %srcElements {bypassL1}: memref<128x128xf32> to memref<3x16x128xf32, 3>
  // CHECK: nvvm.cp.async.commit.group
  %1 = nvgpu.device_async_create_group %0
  // CHECK: nvvm.cp.async.wait.group 1
  nvgpu.device_async_wait %1 { numGroups = 1 : i32 }

  return
}

// -----

// CHECK-LABEL: @async_cp_zfill_f32_align1(
// CHECK-SAME: %[[IDX:[a-zA-Z0-9_]+]]: index, %[[SRCELEMENTS:[a-zA-Z0-9_]+]]: index)
func.func @async_cp_zfill_f32_align1(
  %src: memref<128x128xf32>, %dst: memref<3x16x128xf32, 3>, %i : index, %srcElements : index) {
    // CHECK: %[[IDX1:.*]] = builtin.unrealized_conversion_cast %[[IDX]] : index to i64
  // CHECK: %[[SRC1:.*]] = builtin.unrealized_conversion_cast %[[SRCELEMENTS]] : index to i64
  // CHECK-DAG: %[[BASEDST:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<3 x i64>, array<3 x i64>)>                                   
  // CHECK-DAG: %[[S2048:.*]] = llvm.mlir.constant(2048 : index) : i64
  // CHECK-DAG: %[[LI1:.*]] = llvm.mul %[[IDX1]], %[[S2048]] : i64
  // CHECK-DAG: %[[S0:.*]] = llvm.mlir.constant(128 : index) : i64
  // CHECK-DAG: %[[LI:.*]] = llvm.mul %[[IDX1]], %[[S0]] : i64
  // CHECK-DAG: %[[FI1:.*]] = llvm.add %[[LI1]], %[[LI]] : i64
  // CHECK-DAG: %[[FI2:.*]] = llvm.add %[[FI1]], %[[IDX1]] : i64
  // CHECK-DAG: %[[ADDRESSDST:.*]] = llvm.getelementptr %[[BASEDST]][%[[FI2]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
  // CHECK-DAG: %[[BASESRC:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
  // CHECK-DAG: %[[S2:.*]] = llvm.mlir.constant(128 : index) : i64
  // CHECK-DAG: %[[FI2:.*]] = llvm.mul %[[IDX1]], %[[S2]]  : i64
  // CHECK-DAG: %[[FI3:.*]] = llvm.add %[[FI2]], %[[IDX1]]  : i64
  // CHECK-DAG: %[[ADDRESSSRC:.*]] = llvm.getelementptr %[[BASESRC]][%[[FI3]]] : (!llvm.ptr, i64) -> !llvm.ptr
  // CHECK-DAG: %[[CAST2:.*]] = llvm.addrspacecast %[[ADDRESSSRC]] : !llvm.ptr to !llvm.ptr<1>
  // CHECK-DAG: %[[c1:.*]] = llvm.mlir.constant(3 : i32) : i32
  // CHECK-DAG: %[[c2:.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK-DAG: %[[c3:.*]] = llvm.trunc %[[SRC1]] : i64 to i32  
  // CHECK-DAG: %[[c4:.*]] = llvm.mul %[[c2]], %[[c3]] : i32
  // CHECK-DAG: %[[c5:.*]] = llvm.lshr %[[c4]], %[[c1]] : i32
  // CHECK-DAG: nvvm.cp.async.shared.global %[[ADDRESSDST]], %[[CAST2]], 4, cache = ca, %[[c5]]
  %0 = nvgpu.device_async_copy %src[%i, %i], %dst[%i, %i, %i], 1, %srcElements : memref<128x128xf32> to memref<3x16x128xf32, 3>
  // CHECK: nvvm.cp.async.commit.group
  %1 = nvgpu.device_async_create_group %0
  // CHECK: nvvm.cp.async.wait.group 1
  nvgpu.device_async_wait %1 { numGroups = 1 : i32 }

  return
}

// -----


// CHECK-LABEL: func @mma_sp_sync_f16_16832(
func.func @mma_sp_sync_f16_16832(%arg0: vector<4x2xf16>,
                                 %arg1: vector<4x2xf16>,
                                 %arg2: vector<2x2xf16>,
                                 %arg3: vector<2xi16>) -> vector<2x2xf16> {
  // CHECK: llvm.extractvalue %{{.*}}[0] : !llvm.array<4 x vector<2xf16>>
  // CHECK: llvm.extractvalue %{{.*}}[1] : !llvm.array<4 x vector<2xf16>>
  // CHECK: llvm.extractvalue %{{.*}}[2] : !llvm.array<4 x vector<2xf16>>
  // CHECK: llvm.extractvalue %{{.*}}[3] : !llvm.array<4 x vector<2xf16>>

  // CHECK: llvm.extractvalue %{{.*}}[0] : !llvm.array<4 x vector<2xf16>>
  // CHECK: llvm.extractvalue %{{.*}}[1] : !llvm.array<4 x vector<2xf16>>
  // CHECK: llvm.extractvalue %{{.*}}[2] : !llvm.array<4 x vector<2xf16>>
  // CHECK: llvm.extractvalue %{{.*}}[3] : !llvm.array<4 x vector<2xf16>>

  // CHECK: llvm.extractvalue %{{.*}}[0] : !llvm.array<2 x vector<2xf16>>
  // CHECK: llvm.extractvalue %{{.*}}[1] : !llvm.array<2 x vector<2xf16>>

  // CHECK-NOT llvm.extractvalue

  // CHECK: %[[sparseMetadata:.+]] = llvm.bitcast %{{.+}} : vector<2xi16> to i32

  // CHECK: %[[d:.+]] = llvm.inline_asm has_side_effects asm_dialect = att
  // CHECK-SAME: "mma.sp.sync.aligned.m16n8k32.row.col.f16.f16.f16.f16 {$0,$1},{$2,$3,$4,$5},{$6,$7,$8,$9},{$10,$11},$12,0x0;"
  // CHECK-SAME: "=r,=r,r,r,r,r,r,r,r,r,r,r,r"
  // CHECK-SAME: %[[sparseMetadata]] :
  // CHECK-SAME: -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>

  %d = nvgpu.mma.sp.sync(%arg0, %arg1, %arg2) metadata(%arg3) {mmaShape = [16, 8, 32]} :
    (vector<4x2xf16>, vector<4x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>

  // CHECK-DAG: llvm.extractvalue %[[d]][0] : !llvm.struct<(vector<2xf16>, vector<2xf16>)>
  // CHECK-DAG: llvm.extractvalue %[[d]][1] : !llvm.struct<(vector<2xf16>, vector<2xf16>)>
  //     CHECK: llvm.mlir.undef : !llvm.array<2 x vector<2xf16>>
  //     CHECK: llvm.insertvalue %{{.+}}, %{{.+}}[0] : !llvm.array<2 x vector<2xf16>>
  //     CHECK: llvm.insertvalue %{{.+}}, %{{.+}}[1] : !llvm.array<2 x vector<2xf16>>
  return %d : vector<2x2xf16>
}

// -----

// CHECK-LABEL: func @mma_sp_sync_f16_16816(
func.func @mma_sp_sync_f16_16816(%arg0: vector<2x2xf16>,
                                 %arg1: vector<2x2xf16>,
                                 %arg2: vector<2x2xf16>,
                                 %arg3: vector<2xi16>) -> vector<2x2xf16> {

  // CHECK: llvm.extractvalue %{{.*}}[0] : !llvm.array<2 x vector<2xf16>>
  // CHECK: llvm.extractvalue %{{.*}}[1] : !llvm.array<2 x vector<2xf16>>

  // CHECK: llvm.extractvalue %{{.*}}[0] : !llvm.array<2 x vector<2xf16>>
  // CHECK: llvm.extractvalue %{{.*}}[1] : !llvm.array<2 x vector<2xf16>>

  // CHECK: llvm.extractvalue %{{.*}}[0] : !llvm.array<2 x vector<2xf16>>
  // CHECK: llvm.extractvalue %{{.*}}[1] : !llvm.array<2 x vector<2xf16>>

  // CHECK-NOT llvm.extractvalue

  // CHECK: %[[sparseMetadata:.+]] = llvm.bitcast %{{.+}} : vector<2xi16> to i32

  // CHECK: %[[d:.+]] = llvm.inline_asm has_side_effects asm_dialect = att
  // CHECK-SAME: "mma.sp.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {$0,$1},{$2,$3},{$4,$5},{$6,$7},$8,0x0;"
  // CHECK-SAME: "=r,=r,r,r,r,r,r,r,r"
  // CHECK-SAME: %[[sparseMetadata]] :
  // CHECK-SAME: -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>

  %d = nvgpu.mma.sp.sync(%arg0, %arg1, %arg2) metadata(%arg3) {mmaShape = [16, 8, 16]} :
    (vector<2x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
  return %d : vector<2x2xf16>
}

// -----

// CHECK-LABEL: func @mma_sp_sync_f16_16816_01(
func.func @mma_sp_sync_f16_16816_01(%arg0: vector<2x2xf16>,
                                    %arg1: vector<2x2xf16>,
                                    %arg2: vector<2x2xf16>,
                                    %arg3: vector<2xi16>) -> vector<2x2xf16> {
  //
  // As above, but with sparsity selection 0x01.
  //
  // CHECK: %[[sparseMetadata:.+]] = llvm.bitcast %{{.+}} : vector<2xi16> to i32
  // CHECK: %[[d:.+]] = llvm.inline_asm has_side_effects asm_dialect = att
  // CHECK-SAME: "mma.sp.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {$0,$1},{$2,$3},{$4,$5},{$6,$7},$8,0x1;"
  // CHECK-SAME: "=r,=r,r,r,r,r,r,r,r"
  // CHECK-SAME: %[[sparseMetadata]] :
  // CHECK-SAME: -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>

  %d = nvgpu.mma.sp.sync(%arg0, %arg1, %arg2) metadata(%arg3)
       {mmaShape = [16, 8, 16], sparsitySelector = 1 : i32} :
       (vector<2x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
  return %d : vector<2x2xf16>
}

// -----

// CHECK-LABEL: func @mma_sp_sync_i8_16864(
func.func @mma_sp_sync_i8_16864(%arg0: vector<4x4xi8>,
                                %arg1: vector<4x4xi8>,
                                %arg2: vector<2x2xi32>,
                                %arg3: vector<2xi16>) -> vector<2x2xi32> {

  // CHECK: llvm.extractvalue %{{.*}}[0] : !llvm.array<4 x vector<4xi8>>
  // CHECK: llvm.bitcast %{{.+}} : vector<4xi8> to i32
  // CHECK: llvm.extractvalue %{{.*}}[1] : !llvm.array<4 x vector<4xi8>>
  // CHECK: llvm.bitcast %{{.+}} : vector<4xi8> to i32
  // CHECK: llvm.extractvalue %{{.*}}[2] : !llvm.array<4 x vector<4xi8>>
  // CHECK: llvm.bitcast %{{.+}} : vector<4xi8> to i32
  // CHECK: llvm.extractvalue %{{.*}}[3] : !llvm.array<4 x vector<4xi8>>


  // CHECK: llvm.extractvalue %{{.*}}[0] : !llvm.array<4 x vector<4xi8>>
  // CHECK: llvm.bitcast %{{.+}} : vector<4xi8> to i32
  // CHECK: llvm.extractvalue %{{.*}}[1] : !llvm.array<4 x vector<4xi8>>
  // CHECK: llvm.bitcast %{{.+}} : vector<4xi8> to i32

  // CHECK: llvm.extractvalue %{{.*}}[{{.*}}] : !llvm.array<2 x vector<2xi32>>
  // CHECK: llvm.extractvalue %{{.*}}[{{.*}}] : !llvm.array<2 x vector<2xi32>>

  // CHECK-NOT llvm.extractvalue

  // CHECK: %[[sparseMetadata:.+]] = llvm.bitcast %{{.+}} : vector<2xi16> to i32

  // CHECK: %[[d:.+]] = llvm.inline_asm has_side_effects asm_dialect = att
  // CHECK-SAME: "mma.sp.sync.aligned.m16n8k64.row.col.satfinite.s32.s8.s8.s32 {$0,$1,$2,$3},{$4,$5,$6,$7},{$8,$9,$10,$11},{$12,$13,$14,$15},$16,0x0;"
  // CHECK-SAME: "=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r,r,r,r"
  // CHECK-SAME: %[[sparseMetadata]] :
  // CHECK-SAME: -> !llvm.struct<(i32, i32, i32, i32)

  %d = nvgpu.mma.sp.sync(%arg0, %arg1, %arg2) metadata(%arg3) {mmaShape = [16, 8, 64]} :
    (vector<4x4xi8>, vector<4x4xi8>, vector<2x2xi32>) -> vector<2x2xi32>
  return %d : vector<2x2xi32>
}

// -----
!barrierType = !nvgpu.mbarrier.barrier<memorySpace = #gpu.address_space<workgroup>>
!tokenType = !nvgpu.mbarrier.token

// CHECK-LABEL: func @mbarrier
func.func @mbarrier() {
  %num_threads = arith.constant 128 : index

  // CHECK: %[[barMemref:.+]] = memref.get_global @__mbarrier : memref<1xi64, 3>
  %barrier = nvgpu.mbarrier.create -> !barrierType

  // CHECK: %[[barStr:.+]] =  builtin.unrealized_conversion_cast %[[barMemref]] : memref<1xi64, 3> to !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: %[[barPtr:.+]] = llvm.extractvalue %[[barStr]][1] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)> 
  // CHECK: nvvm.mbarrier.init.shared %[[barPtr]]
  nvgpu.mbarrier.init %barrier, %num_threads : !barrierType
  
  // CHECK: %[[barPtr2:.+]] = llvm.extractvalue %[[barStr]][1] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)> 
  // CHECK: %[[token:.+]] = nvvm.mbarrier.arrive.shared %[[barPtr2]]
  %token = nvgpu.mbarrier.arrive %barrier : !barrierType -> !tokenType
    
  // CHECK: %[[barPtr3:.+]] = llvm.extractvalue %[[barStr]][1] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)> 
  // CHECK: nvvm.mbarrier.test.wait.shared %[[barPtr3]], %[[token]]
  %isDone = nvgpu.mbarrier.test.wait %barrier, %token : !barrierType, !tokenType

  func.return 
}

// -----
!barrierType = !nvgpu.mbarrier.barrier<memorySpace = #gpu.address_space<workgroup>>
!tokenType = !nvgpu.mbarrier.token

// CHECK-LABEL: func @mbarrier_nocomplete
func.func @mbarrier_nocomplete() {
  %num_threads = arith.constant 128 : index
  %count = arith.constant 12 : index

  // CHECK: %[[barMemref:.+]] = memref.get_global @__mbarrier : memref<1xi64, 3>
  %barrier = nvgpu.mbarrier.create -> !barrierType

  // CHECK: %[[barStr:.+]] =  builtin.unrealized_conversion_cast %[[barMemref]] : memref<1xi64, 3> to !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: %[[barPtr:.+]] = llvm.extractvalue %[[barStr]][1] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)> 
  // CHECK: nvvm.mbarrier.init.shared %[[barPtr]]
  nvgpu.mbarrier.init %barrier, %num_threads : !barrierType
  
  // CHECK: %[[barPtr2:.+]] = llvm.extractvalue %[[barStr]][1] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)> 
  // CHECK: %[[token:.+]] = nvvm.mbarrier.arrive.nocomplete.shared %[[barPtr2]]
  %token = nvgpu.mbarrier.arrive.nocomplete %barrier, %count : !barrierType -> !tokenType
    
  // CHECK: %[[barPtr3:.+]] = llvm.extractvalue %[[barStr]][1] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)> 
  // CHECK: nvvm.mbarrier.test.wait.shared %[[barPtr3]], %[[token]]
  %isDone = nvgpu.mbarrier.test.wait %barrier, %token : !barrierType, !tokenType

  func.return 
}

// -----
!barrierType = !nvgpu.mbarrier.barrier<memorySpace = #gpu.address_space<workgroup>>
!tokenType = !nvgpu.mbarrier.token

// CHECK-LABEL: func @mbarrier_txcount
func.func @mbarrier_txcount() {
      %num_threads = arith.constant 128 : index

    // CHECK: %[[barMemref:.+]] = memref.get_global @__mbarrier : memref<1xi64, 3>
    %barrier = nvgpu.mbarrier.create -> !barrierType

    // CHECK: %[[barStr:.+]] =  builtin.unrealized_conversion_cast %[[barMemref]] : memref<1xi64, 3> to !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: %[[barPtr:.+]] = llvm.extractvalue %[[barStr]][1] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)> 
    // CHECK: nvvm.mbarrier.init.shared %[[barPtr]]
    nvgpu.mbarrier.init %barrier, %num_threads : !barrierType
    
    %c0 = arith.constant 0 : index  
    %tidxreg = nvvm.read.ptx.sreg.tid.x : i32
    %tidx = arith.index_cast %tidxreg : i32 to index
    %cnd = arith.cmpi eq, %tidx, %c0 : index  

    scf.if %cnd {
      %txcount = arith.constant 256 : index
      // CHECK: %[[barPtr2:.+]] = llvm.extractvalue %[[barStr]][1] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)> 
      // CHECK: nvvm.mbarrier.arrive.expect_tx.shared %[[barPtr2]]
      nvgpu.mbarrier.arrive.expect_tx %barrier, %txcount : !barrierType
      scf.yield 
    } else {
      %txcount = arith.constant 0 : index
      // CHECK: %[[barPtr2:.+]] = llvm.extractvalue %[[barStr]][1] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)> 
      // CHECK: nvvm.mbarrier.arrive.expect_tx.shared %[[barPtr2]]
      nvgpu.mbarrier.arrive.expect_tx %barrier, %txcount : !barrierType
      scf.yield 
    }
      

    %phase = arith.constant 0 : index
    %ticks = arith.constant 10000000 : index
    // CHECK: %[[barPtr3:.+]] = llvm.extractvalue %[[barStr]][1] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)> 
    // CHECK: nvvm.mbarrier.try_wait.parity.shared %[[barPtr3]]
    nvgpu.mbarrier.try_wait.parity %barrier, %phase, %ticks : !barrierType

    func.return 
}

// -----

// CHECK-LABEL: func @async_tma_load
!tensorMap1d = !nvgpu.tensormap.descriptor<tensor = memref<128xf32,3>,         swizzle=none,        l2promo = none,        oob = nan,  interleave = none>
!tensorMap2d = !nvgpu.tensormap.descriptor<tensor = memref<32x32xf32,3>,       swizzle=swizzle_32b, l2promo = none,        oob = zero, interleave = none>
!tensorMap3d = !nvgpu.tensormap.descriptor<tensor = memref<2x32x32xf32,3>,     swizzle=swizzle_64b, l2promo = l2promo_64b, oob = zero, interleave = none>
!tensorMap4d = !nvgpu.tensormap.descriptor<tensor = memref<2x2x32x32xf32,3>,   swizzle=swizzle_128b,l2promo = l2promo_128b,oob = zero, interleave = interleave_16b>
!tensorMap5d = !nvgpu.tensormap.descriptor<tensor = memref<2x2x2x32x32xf32,3>, swizzle=none,        l2promo = none,        oob = zero, interleave = none>
!mbarrier = !nvgpu.mbarrier.barrier<memorySpace = #gpu.address_space<workgroup>>
func.func @async_tma_load(%tensorMap1d: !tensorMap1d, %tensorMap2d: !tensorMap2d, %tensorMap3d: !tensorMap3d, %tensorMap4d: !tensorMap4d, %tensorMap5d: !tensorMap5d, 
                              %buffer1d: memref<128xf32,3>,      
                              %buffer2d: memref<32x32xf32,3>,    
                              %buffer3d: memref<2x32x32xf32,3>,  
                              %buffer4d: memref<2x2x32x32xf32,3>,  
                              %buffer5d: memref<2x2x2x32x32xf32,3>,
                              %mbarrier: !mbarrier) {
  %crd0 = arith.constant 0 : index
  %crd1 = arith.constant 0 : index
  // CHECK: nvvm.cp.async.bulk.tensor.shared.cluster.global %{{.*}}, %{{.*}}, %{{.*}}, box[%{{.*}}] 
  nvgpu.tma.async.load %tensorMap1d[%crd0], %mbarrier to %buffer1d : !tensorMap1d, !mbarrier -> memref<128xf32,3>
  // CHECK: nvvm.cp.async.bulk.tensor.shared.cluster.global %{{.*}}, %{{.*}}, %{{.*}}, box[%{{.*}}, %{{.*}}] 
  nvgpu.tma.async.load %tensorMap2d[%crd0, %crd1], %mbarrier to %buffer2d : !tensorMap2d, !mbarrier -> memref<32x32xf32,3>
  // CHECK: nvvm.cp.async.bulk.tensor.shared.cluster.global %{{.*}}, %{{.*}}, %{{.*}}, box[%{{.*}}, %{{.*}}, %{{.*}}] 
  nvgpu.tma.async.load %tensorMap3d[%crd0, %crd1, %crd0], %mbarrier to %buffer3d : !tensorMap3d, !mbarrier -> memref<2x32x32xf32,3>
  // CHECK: nvvm.cp.async.bulk.tensor.shared.cluster.global %{{.*}}, %{{.*}}, %{{.*}}, box[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] 
  nvgpu.tma.async.load %tensorMap4d[%crd0, %crd1, %crd1, %crd0], %mbarrier to %buffer4d : !tensorMap4d, !mbarrier -> memref<2x2x32x32xf32,3>
  // CHECK: nvvm.cp.async.bulk.tensor.shared.cluster.global %{{.*}}, %{{.*}}, %{{.*}}, box[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] 
  nvgpu.tma.async.load %tensorMap5d[%crd0, %crd1, %crd1, %crd0, %crd0], %mbarrier to %buffer5d : !tensorMap5d, !mbarrier -> memref<2x2x2x32x32xf32,3>
  func.return 
}

func.func @create_tensor_map(%devicePtr2d : memref<64x128xf32>, %devicePtr1d : memref<128xf32>) {
  %crd0 = arith.constant 64 : index
  %crd1 = arith.constant 128 : index
  %devicePtr2d_unranked = memref.cast %devicePtr2d : memref<64x128xf32> to memref<*xf32>
  // CHECK : llvm.call @mgpuTensorMapEncodeTiledMemref
  %tensorMap2d = nvgpu.tma.create.descriptor %devicePtr2d_unranked box[%crd0, %crd1] : memref<*xf32> -> !tensorMap2d

  %devicePtr1d_unranked = memref.cast %devicePtr1d : memref<128xf32> to memref<*xf32>
  // CHECK : llvm.call @mgpuTensorMapEncodeTiledMemref
  %tensorMap1d = nvgpu.tma.create.descriptor %devicePtr1d_unranked box[%crd1] : memref<*xf32> -> !tensorMap1d
  func.return
}
