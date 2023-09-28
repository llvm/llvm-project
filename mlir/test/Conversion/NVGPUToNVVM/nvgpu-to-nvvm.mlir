// RUN: mlir-opt %s -convert-nvgpu-to-nvvm='use-opaque-pointers=1' | FileCheck %s
// RUN: mlir-opt %s -test-transform-dialect-interpreter | FileCheck %s

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

// CHECK-LABEL: @ldmatrix_x1
func.func @ldmatrix_x1(%arg0: memref<128x128xf16, 3>) ->  vector<1x2xf16> {
  %c0  = arith.constant 0 : index
  // CHECK: nvvm.ldmatrix {{%.+}} {layout = #nvvm.mma_layout<row>, num = 1 : i32} {{.*}} -> i32
  %a = nvgpu.ldmatrix %arg0[%c0, %c0] {transpose = false, numTiles = 1 : i32} : memref<128x128xf16, 3> -> vector<1x2xf16>
  // CHECK: llvm.bitcast
  // CHECK: llvm.insertvalue
  return %a : vector<1x2xf16>
}

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

!barrierType = !nvgpu.mbarrier.group<memorySpace = #gpu.address_space<workgroup>>
!tokenType = !nvgpu.mbarrier.token

// CHECK-LABEL: func @mbarrier
func.func @mbarrier() {
  %num_threads = arith.constant 128 : index
  // CHECK: %[[c0:.+]] = arith.constant 0 : index
  // CHECK: %[[mid:.+]] = builtin.unrealized_conversion_cast %[[c0]] : index to i64
  %c0 = arith.constant 0 : index

  // CHECK: %[[barMemref:.+]] = memref.get_global @__mbarrier{{.*}} : memref<1xi64, 3>
  %barrier = nvgpu.mbarrier.create -> !barrierType

  // CHECK: %[[barStr:.+]] =  builtin.unrealized_conversion_cast %[[barMemref]] : memref<1xi64, 3> to !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: %[[base:.+]] = llvm.extractvalue %[[barStr]][1] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)> 
  // CHECK: %[[barPtr:.+]] = llvm.getelementptr %[[base]][%[[mid]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i64
  // CHECK: nvvm.mbarrier.init.shared %[[barPtr]]
    nvgpu.mbarrier.init %barrier[%c0], %num_threads : !barrierType
  
  // CHECK: %[[base2:.+]] = llvm.extractvalue %[[barStr]][1] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)> 
  // CHECK: %[[barPtr2:.+]] = llvm.getelementptr %[[base2]][%[[mid]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i64
  // CHECK: %[[token:.+]] = nvvm.mbarrier.arrive.shared %[[barPtr2]]
  %token = nvgpu.mbarrier.arrive %barrier[%c0] : !barrierType -> !tokenType
    
  // CHECK: %[[base3:.+]] = llvm.extractvalue %[[barStr]][1] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)> 
  // CHECK: %[[barPtr3:.+]] = llvm.getelementptr %[[base3]][%[[mid]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i64
  // CHECK: nvvm.mbarrier.test.wait.shared %[[barPtr3]], %[[token]]
  %isDone = nvgpu.mbarrier.test.wait %barrier[%c0], %token : !barrierType, !tokenType

  func.return 
}

// CHECK-LABEL: func @mbarrier_nocomplete
func.func @mbarrier_nocomplete() {
  %num_threads = arith.constant 128 : index
  %count = arith.constant 12 : index
  // CHECK: %[[c0:.+]] = arith.constant 0 : index
  // CHECK: %[[mid:.+]] = builtin.unrealized_conversion_cast %[[c0]] : index to i64
  %c0 = arith.constant 0 : index

  // CHECK: %[[barMemref:.+]] = memref.get_global @__mbarrier{{.*}} : memref<1xi64, 3>
  %barrier = nvgpu.mbarrier.create -> !barrierType

  // CHECK: %[[barStr:.+]] =  builtin.unrealized_conversion_cast %[[barMemref]] : memref<1xi64, 3> to !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: %[[base:.+]] = llvm.extractvalue %[[barStr]][1] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)> 
  // CHECK: %[[barPtr:.+]] = llvm.getelementptr %[[base]][%[[mid]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i64
  // CHECK: nvvm.mbarrier.init.shared %[[barPtr]]
  nvgpu.mbarrier.init %barrier[%c0], %num_threads : !barrierType
  
  // CHECK: %[[base2:.+]] = llvm.extractvalue %[[barStr]][1] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)> 
  // CHECK: %[[barPtr2:.+]] = llvm.getelementptr %[[base2]][%[[mid]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i64
  // CHECK: %[[token:.+]] = nvvm.mbarrier.arrive.nocomplete.shared %[[barPtr2]]
  %token = nvgpu.mbarrier.arrive.nocomplete %barrier[%c0], %count : !barrierType -> !tokenType
    
  // CHECK: %[[base3:.+]] = llvm.extractvalue %[[barStr]][1] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)> 
  // CHECK: %[[barPtr3:.+]] = llvm.getelementptr %[[base3]][%[[mid]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i64
  // CHECK: nvvm.mbarrier.test.wait.shared %[[barPtr3]], %[[token]]
  %isDone = nvgpu.mbarrier.test.wait %barrier[%c0], %token : !barrierType, !tokenType

  func.return 
}

// CHECK-LABEL: func @mbarrier_wait
func.func @mbarrier_wait(%barriers : !nvgpu.mbarrier.group<memorySpace = #gpu.address_space<workgroup>, num_barriers = 5>, %token : !tokenType) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %n = arith.constant 100 : index

  %numBarriers = arith.constant 5 : index
  
  scf.for %i = %c0 to %n step %c1 {
// CHECK: %[[c5:.+]] = arith.constant 5 : index
// CHECK: scf.for %[[i:.*]] =
// CHECK: %[[S2:.+]] = arith.remui %[[i]], %[[c5]] : index
// CHECK: %[[S3:.+]] = builtin.unrealized_conversion_cast %[[S2]] : index to i64
// CHECK: %[[S4:.+]] = llvm.extractvalue %0[1] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)> 
// CHECK: %[[S5:.+]] = llvm.getelementptr %[[S4]][%[[S3]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i64
    %mbarId = arith.remui %i, %numBarriers : index
    %isDone = nvgpu.mbarrier.test.wait %barriers[%mbarId], %token : !nvgpu.mbarrier.group<memorySpace = #gpu.address_space<workgroup>, num_barriers = 5>, !tokenType
  }
  return
}

// CHECK-LABEL: func @mbarrier_txcount
func.func @mbarrier_txcount() {
    %num_threads = arith.constant 128 : index
    // CHECK: %[[c0:.+]] = arith.constant 0 : index
    // CHECK: %[[mid:.+]] = builtin.unrealized_conversion_cast %[[c0]] : index to i64
    %c0 = arith.constant 0 : index  

    // CHECK: %[[barMemref:.+]] = memref.get_global @__mbarrier{{.*}} : memref<1xi64, 3>
    %barrier = nvgpu.mbarrier.create -> !barrierType

    // CHECK: %[[barStr:.+]] =  builtin.unrealized_conversion_cast %[[barMemref]] : memref<1xi64, 3> to !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: %[[base:.+]] = llvm.extractvalue %[[barStr]][1] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)> 
    // CHECK: %[[barPtr:.+]] = llvm.getelementptr %[[base]][%[[mid]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i64
    // CHECK: nvvm.mbarrier.init.shared %[[barPtr]]
    nvgpu.mbarrier.init %barrier[%c0], %num_threads : !barrierType
    
    %tidxreg = nvvm.read.ptx.sreg.tid.x : i32
    %tidx = arith.index_cast %tidxreg : i32 to index
    %cnd = arith.cmpi eq, %tidx, %c0 : index  

    scf.if %cnd {
      %txcount = arith.constant 256 : index
      // CHECK: %[[base2:.+]] = llvm.extractvalue %[[barStr]][1] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)> 
      // CHECK: %[[barPtr2:.+]] = llvm.getelementptr %[[base2]][%[[mid]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i64
      // CHECK: nvvm.mbarrier.arrive.expect_tx.shared %[[barPtr2]]
      nvgpu.mbarrier.arrive.expect_tx %barrier[%c0], %txcount : !barrierType
      scf.yield 
    } else {
      %txcount = arith.constant 0 : index
      // CHECK: %[[base2:.+]] = llvm.extractvalue %[[barStr]][1] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)> 
      // CHECK: %[[barPtr2:.+]] = llvm.getelementptr %[[base2]][%[[mid]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i64
      // CHECK: nvvm.mbarrier.arrive.expect_tx.shared %[[barPtr2]]
      nvgpu.mbarrier.arrive.expect_tx %barrier[%c0], %txcount : !barrierType
      scf.yield 
    }
      

    %phase = arith.constant 0 : index
    %ticks = arith.constant 10000000 : index
    // CHECK: %[[base3:.+]] = llvm.extractvalue %[[barStr]][1] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)> 
    // CHECK: %[[barPtr3:.+]] = llvm.getelementptr %[[base3]][%[[mid]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i64
    // CHECK: nvvm.mbarrier.try_wait.parity.shared %[[barPtr3]]
    nvgpu.mbarrier.try_wait.parity %barrier[%c0], %phase, %ticks : !barrierType

    func.return 
}

// CHECK-LABEL: func @async_tma_load
!tensorMap1d = !nvgpu.tensormap.descriptor<tensor = memref<128xf32,3>,         swizzle=none,        l2promo = none,        oob = nan,  interleave = none>
!tensorMap2d = !nvgpu.tensormap.descriptor<tensor = memref<32x32xf32,3>,       swizzle=swizzle_32b, l2promo = none,        oob = zero, interleave = none>
!tensorMap3d = !nvgpu.tensormap.descriptor<tensor = memref<2x32x32xf32,3>,     swizzle=swizzle_64b, l2promo = l2promo_64b, oob = zero, interleave = none>
!tensorMap4d = !nvgpu.tensormap.descriptor<tensor = memref<2x2x32x32xf32,3>,   swizzle=swizzle_128b,l2promo = l2promo_128b,oob = zero, interleave = interleave_16b>
!tensorMap5d = !nvgpu.tensormap.descriptor<tensor = memref<2x2x2x32x32xf32,3>, swizzle=none,        l2promo = none,        oob = zero, interleave = none>
!mbarrier = !nvgpu.mbarrier.group<memorySpace = #gpu.address_space<workgroup>>
func.func @async_tma_load(%tensorMap1d: !tensorMap1d, %tensorMap2d: !tensorMap2d, %tensorMap3d: !tensorMap3d, %tensorMap4d: !tensorMap4d, %tensorMap5d: !tensorMap5d, 
                              %buffer1d: memref<128xf32,3>,      
                              %buffer2d: memref<32x32xf32,3>,    
                              %buffer3d: memref<2x32x32xf32,3>,  
                              %buffer4d: memref<2x2x32x32xf32,3>,  
                              %buffer5d: memref<2x2x2x32x32xf32,3>,
                              %mbarrier: !mbarrier) {
  %c0 = arith.constant 0 : index
  %crd0 = arith.constant 0 : index
  %crd1 = arith.constant 0 : index
  // CHECK: nvvm.cp.async.bulk.tensor.shared.cluster.global %{{.*}}, %{{.*}}, %{{.*}}, box[%{{.*}}] 
  nvgpu.tma.async.load %tensorMap1d[%crd0], %mbarrier[%c0] to %buffer1d : !tensorMap1d, !mbarrier -> memref<128xf32,3>
  // CHECK: nvvm.cp.async.bulk.tensor.shared.cluster.global %{{.*}}, %{{.*}}, %{{.*}}, box[%{{.*}}, %{{.*}}] 
  nvgpu.tma.async.load %tensorMap2d[%crd0, %crd1], %mbarrier[%c0] to %buffer2d : !tensorMap2d, !mbarrier -> memref<32x32xf32,3>
  // CHECK: nvvm.cp.async.bulk.tensor.shared.cluster.global %{{.*}}, %{{.*}}, %{{.*}}, box[%{{.*}}, %{{.*}}, %{{.*}}] 
  nvgpu.tma.async.load %tensorMap3d[%crd0, %crd1, %crd0], %mbarrier[%c0] to %buffer3d : !tensorMap3d, !mbarrier -> memref<2x32x32xf32,3>
  // CHECK: nvvm.cp.async.bulk.tensor.shared.cluster.global %{{.*}}, %{{.*}}, %{{.*}}, box[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] 
  nvgpu.tma.async.load %tensorMap4d[%crd0, %crd1, %crd1, %crd0], %mbarrier[%c0] to %buffer4d : !tensorMap4d, !mbarrier -> memref<2x2x32x32xf32,3>
  // CHECK: nvvm.cp.async.bulk.tensor.shared.cluster.global %{{.*}}, %{{.*}}, %{{.*}}, box[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] 
  nvgpu.tma.async.load %tensorMap5d[%crd0, %crd1, %crd1, %crd0, %crd0], %mbarrier[%c0] to %buffer5d : !tensorMap5d, !mbarrier -> memref<2x2x2x32x32xf32,3>
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

// CHECK-LABEL: @tma_prefetch(  
// CHECK-SAME: %[[arg0:[a-zA-Z0-9_]+]]: !nvgpu.tensormap.descriptor<tensor = memref<128xf32,3>, swizzle=none, l2promo = none, oob = nan, interleave = none>, %[[arg1:[a-zA-Z0-9_]+]]: i1
func.func @tma_prefetch(%tensorMap1d: !tensorMap1d, %p : i1) {
  // CHECK: nvvm.prefetch.tensormap %[[arg0]] : !llvm.ptr
  nvgpu.tma.prefetch.descriptor %tensorMap1d: !tensorMap1d
  // CHECK: nvvm.prefetch.tensormap %[[arg0]], predicate = %[[arg1]] : !llvm.ptr, i1
  nvgpu.tma.prefetch.descriptor %tensorMap1d, %p: !tensorMap1d
}

!lhsTensorMap = !nvgpu.tensormap.descriptor<tensor = memref<128x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>
!rhsTensorMap = !nvgpu.tensormap.descriptor<tensor = memref<64x128xf16, strided<[128, 1], offset: 8192>, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>

!shmemlhs = memref<128x64xf16,3>
!shmemrhs = memref<64x128xf16, strided<[128, 1], offset: 8192>, 3>

module @mymodule {
  // Dynamic Shared memory
  memref.global "private" @dynamicShmem : memref<0xf16,3>

  func.func @async_tma_load(%lhsTensorMap: !lhsTensorMap, %rhsTensorMap: !rhsTensorMap, %mbarrier: !barrierType) {
    %c0 = arith.constant 0 : index
    %dynamicMem = memref.get_global @dynamicShmem : memref<0xf16, 3>
    %lhsShmem = memref.reinterpret_cast %dynamicMem to offset: [0], sizes: [128,64], strides: [64,1] : memref<0xf16, 3> to !shmemlhs
    %rhsShmem2 = memref.reinterpret_cast %dynamicMem to offset: [0], sizes: [2,64,128],  strides: [8192,128,1] : memref<0xf16, 3> to memref<2x64x128xf16,3>
    %rhsShmem3 = memref.subview %rhsShmem2[1,0,0][1, 64, 128][1, 1, 1] : memref<2x64x128xf16,3> to memref<1x64x128xf16, strided<[8192, 128, 1], offset: 8192>, 3>
    %rhsShmem = memref.subview %rhsShmem3[0,0,0][1, 64, 128][1, 1, 1] : memref<1x64x128xf16, strided<[8192, 128, 1], offset: 8192>, 3> to !shmemrhs
    // CHECK: nvvm.cp.async.bulk.tensor.shared.cluster.global
    nvgpu.tma.async.load %lhsTensorMap[%c0, %c0], %mbarrier[%c0] to %lhsShmem : !lhsTensorMap, !barrierType -> !shmemlhs
    // CHECK: %[[desc:.+]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)> 
    // CHECK: %[[c8192:.+]] = llvm.mlir.constant(8192 : index) : i64
    // CHECK: %[[shmemOfset:.+]] = llvm.getelementptr %[[desc]][%[[c8192]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f16
    // CHECK: nvvm.cp.async.bulk.tensor.shared.cluster.global %[[shmemOfset]], %{{.*}}, %{{.*}}, box[%{{.*}}, %{{.*}}] : !llvm.ptr<3>, !llvm.ptr, !llvm.ptr<3>, i32, i32
    nvgpu.tma.async.load %rhsTensorMap[%c0, %c0], %mbarrier[%c0] to %rhsShmem : !rhsTensorMap, !barrierType -> !shmemrhs
    return
  }
}

!tensorMap = !nvgpu.tensormap.descriptor<tensor = memref<128x64xf16,3>, swizzle = swizzle_128b, l2promo=none, oob=zero, interleave=none>
memref.global "private" @dynamicShmem : memref<0xf16,3>
// CHECK-LABEL: func @create_wgmma_descriptor(
func.func @create_wgmma_descriptor(%tensorMap : !tensorMap) -> !nvgpu.warpgroup.descriptor<tensor=memref<128x64xf16,3>>{
  %dynamicMem = memref.get_global @dynamicShmem : memref<0xf16, 3>
  %lhsShmem = memref.reinterpret_cast %dynamicMem to offset: [0], sizes: [128,64], strides: [64,1] : memref<0xf16, 3> to memref<128x64xf16,3>
    // CHECK: %[[S0:.+]] = memref.get_global @dynamicShmem : memref<0xf16, 3>
    // CHECK: %[[Sre:.+]] = memref.reinterpret_cast %[[S0]] to offset: [0], sizes: [128, 64], strides: [64, 1] : memref<0xf16, 3> to memref<128x64xf16, 3>
    // CHECK: %[[S1:.+]] = builtin.unrealized_conversion_cast %[[Sre]] : memref<128x64xf16, 3> to !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK: %[[c64:.+]] =  llvm.mlir.constant(64 : i64) : i64
    // CHECK: %[[c1024:.+]] = llvm.mlir.constant(1024 : i64) : i64
    // CHECK: %[[S2:.+]] = llvm.extractvalue %[[S1]][1] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)> 
    // CHECK: %[[S3:.+]] = llvm.ptrtoint %[[S2]] : !llvm.ptr<3> to i64
    // CHECK: %[[S4:.+]] = llvm.mlir.constant(46 : i64) : i64
    // CHECK: %[[S5:.+]] = llvm.shl %[[S3]], %[[S4]]  : i64
    // CHECK: %[[S6:.+]] = llvm.mlir.constant(50 : i64) : i64
    // CHECK: %[[S7:.+]] = llvm.lshr %[[S5]], %[[S6]]  : i64
    // CHECK: %[[S8:.+]] = llvm.mlir.constant(0 : i64) : i64
    // CHECK: %[[S9:.+]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK: %[[S10:.+]] = llvm.mlir.constant(62 : i64) : i64
    // CHECK: %[[S11:.+]] = llvm.shl %[[S9]], %[[S10]]  : i64
    // CHECK: %[[S12:.+]] = llvm.or %[[S8]], %[[S11]]  : i64
    // CHECK: %[[S13:.+]] = llvm.mlir.constant(0 : i64) : i64
    // CHECK: %[[S14:.+]] = llvm.mlir.constant(49 : i64) : i64
    // CHECK: %[[S15:.+]] = llvm.shl %[[S13]], %[[S14]]  : i64
    // CHECK: %[[S16:.+]] = llvm.or %[[S12]], %[[S15]]  : i64
    // CHECK: %[[S18:.+]] = llvm.mlir.constant(32 : i64) : i64
    // CHECK: %[[S19:.+]] = llvm.shl %[[c64]], %[[S18]]  : i64
    // CHECK: %[[S20:.+]] = llvm.or %[[S16]], %[[S19]]  : i64
    // CHECK: %[[S22:.+]] = llvm.mlir.constant(16 : i64) : i64
    // CHECK: %[[S23:.+]] = llvm.shl %[[c1024]], %[[S22]]  : i64
    // CHECK: %[[S24:.+]] = llvm.or %[[S20]], %[[S23]]  : i64
    // CHECK: %[[S25:.+]] = llvm.mlir.constant(0 : i64) : i64
    // CHECK: %[[S26:.+]] = llvm.shl %[[S7]], %[[S25]]  : i64
    // CHECK: %[[S27:.+]] = llvm.or %[[S24]], %[[S26]]  : i64
    // CHECK: %[[ret:.+]] = builtin.unrealized_conversion_cast %[[S27]] : i64 to !nvgpu.warpgroup.descriptor<tensor = memref<128x64xf16, 3>> 
    // CHECK: return %[[ret]]
  %descA = nvgpu.warpgroup.generate.descriptor %lhsShmem, %tensorMap : memref<128x64xf16,3>, !tensorMap -> !nvgpu.warpgroup.descriptor<tensor=memref<128x64xf16,3>>
  func.return %descA : !nvgpu.warpgroup.descriptor<tensor=memref<128x64xf16,3>>
}

// CHECK-LABEL: @warpgroup_mma_128_128_64(  
// CHECK-SAME: %[[arg0:[a-zA-Z0-9_]+]]: !nvgpu.warpgroup.descriptor<tensor = memref<128x64xf16, 3>>, %[[arg1:[a-zA-Z0-9_]+]]: !nvgpu.warpgroup.descriptor<tensor = memref<64x128xf16, 3>>, %[[arg2:[a-zA-Z0-9_]+]]: !nvgpu.warpgroup.accumulator<fragmented = vector<128x128xf32>>)
func.func @warpgroup_mma_128_128_64(
      %descA: !nvgpu.warpgroup.descriptor<tensor = memref<128x64xf16, 3>>, 
      %descB: !nvgpu.warpgroup.descriptor<tensor = memref<64x128xf16, 3>>, 
      %acc: !nvgpu.warpgroup.accumulator<fragmented = vector<128x128xf32>>) 
{
// CHECK: %[[S0:.+]] = builtin.unrealized_conversion_cast %[[arg0]] : !nvgpu.warpgroup.descriptor<tensor = memref<128x64xf16, 3>> to i64
// CHECK: %[[S1:.+]] = builtin.unrealized_conversion_cast %[[arg1]] : !nvgpu.warpgroup.descriptor<tensor = memref<64x128xf16, 3>> to i64
// CHECK: %[[ARG:.+]] = builtin.unrealized_conversion_cast %[[arg2]] : !nvgpu.warpgroup.accumulator<fragmented = vector<128x128xf32>> to !llvm.struct<(struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>, struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>)>
// CHECK: nvvm.wgmma.fence.aligned
// CHECK: %[[UD:.+]] =  llvm.mlir.undef : !llvm.struct<(struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>, struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>)>
// CHECK: %[[S2:.+]] = llvm.extractvalue %[[ARG]][0] : !llvm.struct<(struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>, struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>)> 
// CHECK: %[[S4:.+]] = nvvm.wgmma.mma_async %[[S0]], %[[S1]], <m = 64, n = 128, k = 16>, D[%[[S2]], <one>, <wrapped>], A[<f16>, <one>, <row>], B[<f16>, <one>, <col>] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> -> !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>
// CHECK: %[[S5:.+]] = llvm.mlir.constant(2 : i32) : i64
// CHECK: %[[S6:.+]] = llvm.add %[[S0]], %[[S5]] : i64
// CHECK: %[[S7:.+]] = llvm.mlir.constant(128 : i32) : i64
// CHECK: %[[S8:.+]] = llvm.add %[[S1]], %[[S7]]  : i64
// CHECK: %[[S9:.+]] = nvvm.wgmma.mma_async %[[S6]], %[[S8]], <m = 64, n = 128, k = 16>, D[%[[S4]], <one>, <wrapped>], A[<f16>, <one>, <row>], B[<f16>, <one>, <col>] : !llvm.struct
// CHECK: %[[S10:.+]] = llvm.mlir.constant(4 : i32) : i64
// CHECK: %[[S11:.+]] = llvm.add %[[S0]], %[[S10]]  : i64
// CHECK: %[[S12:.+]] = llvm.mlir.constant(256 : i32) : i64
// CHECK: %[[S13:.+]] = llvm.add %[[S1]], %[[S12]]  : i64
// CHECK: %[[S14:.+]] = nvvm.wgmma.mma_async %[[S11]], %[[S13]], <m = 64, n = 128, k = 16>, D[%[[S9]], <one>, <wrapped>], A[<f16>, <one>, <row>], B[<f16>, <one>, <col>] : !llvm.struct
// CHECK: %[[S15:.+]] = llvm.mlir.constant(6 : i32) : i64
// CHECK: %[[S16:.+]] = llvm.add %[[S0]], %[[S15]]  : i64
// CHECK: %[[S17:.+]] = llvm.mlir.constant(384 : i32) : i64
// CHECK: %[[S18:.+]] = llvm.add %[[S1]], %[[S17]]  : i64
// CHECK: %[[S19:.+]] = nvvm.wgmma.mma_async %[[S16]], %[[S18]], <m = 64, n = 128, k = 16>, D[%[[S14]], <one>, <wrapped>], A[<f16>, <one>, <row>], B[<f16>, <one>, <col>] : !llvm.struct
// CHECK: %[[S3:.+]] = llvm.extractvalue %[[ARG]][1] : !llvm.struct<(struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>, struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>)> 
// CHECK: %[[S21:.+]] = llvm.mlir.constant(512 : i32) : i64
// CHECK: %[[S22:.+]] = llvm.add %[[S0]], %[[S21]]  : i64
// CHECK: %[[S23:.+]] = nvvm.wgmma.mma_async %[[S22]], %[[S1]], <m = 64, n = 128, k = 16>, D[%[[S3]], <one>, <wrapped>], A[<f16>, <one>, <row>], B[<f16>, <one>, <col>] : !llvm.struct
// CHECK: %[[S24:.+]] = llvm.mlir.constant(514 : i32) : i64
// CHECK: %[[S25:.+]] = llvm.add %[[S0]], %[[S24]]  : i64
// CHECK: %[[S26:.+]] = llvm.mlir.constant(128 : i32) : i64
// CHECK: %[[S27:.+]] = llvm.add %[[S1]], %[[S26]]  : i64
// CHECK: %[[S28:.+]] = nvvm.wgmma.mma_async %[[S25]], %[[S27]], <m = 64, n = 128, k = 16>, D[%[[S23]], <one>, <wrapped>], A[<f16>, <one>, <row>], B[<f16>, <one>, <col>] : !llvm.struct
// CHECK: %[[S29:.+]] = llvm.mlir.constant(516 : i32) : i64
// CHECK: %[[S30:.+]] = llvm.add %[[S0]], %[[S29]]  : i64
// CHECK: %[[S31:.+]] = llvm.mlir.constant(256 : i32) : i64
// CHECK: %[[S32:.+]] = llvm.add %[[S1]], %[[S31]]  : i64
// CHECK: %[[S33:.+]] = nvvm.wgmma.mma_async %[[S30]], %[[S32]], <m = 64, n = 128, k = 16>, D[%[[S28]], <one>, <wrapped>], A[<f16>, <one>, <row>], B[<f16>, <one>, <col>] : !llvm.struct
// CHECK: %[[S34:.+]] = llvm.mlir.constant(518 : i32) : i64
// CHECK: %[[S35:.+]] = llvm.add %[[S0]], %[[S34]]  : i64
// CHECK: %[[S36:.+]] = llvm.mlir.constant(384 : i32) : i64
// CHECK: %[[S37:.+]] = llvm.add %[[S1]], %[[S36]]  : i64
// CHECK: %[[S38:.+]] = nvvm.wgmma.mma_async %[[S35]], %[[S37]], <m = 64, n = 128, k = 16>, D[%[[S33]], <one>, <wrapped>], A[<f16>, <one>, <row>], B[<f16>, <one>, <col>] : !llvm.struct
// CHECK: %[[S40:.+]] = llvm.insertvalue %[[S19]], %[[UD]][0] : !llvm.struct<(struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>, struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>)> 
// CHECK: %[[S41:.+]] = llvm.insertvalue %[[S38]], %[[S40]][1] : !llvm.struct<(struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>, struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>)> 
// CHECK: nvvm.wgmma.commit.group.sync.aligned
// CHECK: nvvm.wgmma.wait.group.sync.aligned 1  
  %wgmmaResult = nvgpu.warpgroup.mma %descA, %descB, %acc {transposeB}: 
      !nvgpu.warpgroup.descriptor<tensor = memref<128x64xf16, 3>>, 
      !nvgpu.warpgroup.descriptor<tensor = memref<64x128xf16, 3>>, 
      !nvgpu.warpgroup.accumulator<fragmented = vector<128x128xf32>> 
      -> 
      !nvgpu.warpgroup.accumulator<fragmented = vector<128x128xf32>>  
  return
}

// CHECK-LABEL: @warpgroup_mma_store(  
// CHECK-SAME: %[[arg0:[a-zA-Z0-9_]+]]: !nvgpu.warpgroup.accumulator<fragmented = vector<128x128xf32>>, %[[arg2:[a-zA-Z0-9_]+]]: memref<128x128xf32, 3>)
func.func @warpgroup_mma_store(
    %result : !nvgpu.warpgroup.accumulator<fragmented = vector<128x128xf32>>, 
    %matrixD: memref<128x128xf32,3>) {
// CHECK: %[[S0:.+]] = builtin.unrealized_conversion_cast %[[arg0]] : !nvgpu.warpgroup.accumulator<fragmented = vector<128x128xf32>> to !llvm.struct<(struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>, struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>)>
// CHECK: %[[EX1:.+]] = llvm.extractvalue %[[S0]][0] : !llvm.struct<(struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>, struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>)> 
// CHECK: %[[S6:.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: %[[S5:.+]] = llvm.mlir.constant(2 : i32) : i32
// CHECK: %[[S2:.+]] = llvm.mlir.constant(4 : i32) : i32
// CHECK: %[[S4:.+]] = llvm.mlir.constant(8 : i32) : i32
// CHECK: %[[S7:.+]] = llvm.mlir.constant(16 : i32) : i32
// CHECK: %[[WarpSize:.+]] = llvm.mlir.constant(32 : i32) : i32

// ### Store {d0, d1} of each thread ###

// CHECK: %[[S8:.+]] = nvvm.read.ptx.sreg.tid.x : i32
// CHECK: %[[S9:.+]] = llvm.urem %[[S8]], %[[WarpSize]]  : i32
// CHECK: %[[S10:.+]] = llvm.udiv %[[S8]], %[[WarpSize]]  : i32
// CHECK: %[[S11:.+]] = llvm.udiv %[[S9]], %[[S2]]  : i32
// CHECK: %[[S12:.+]] = llvm.urem %[[S9]], %[[S2]]  : i32
// CHECK: %[[S13:.+]] = llvm.mul %[[S12]], %[[S5]]  : i32
// CHECK: %[[S14:.+]] = llvm.mul %[[S10]], %[[S7]]  : i32
// CHECK: %[[S15:.+]] = llvm.add %[[S11]], %[[S14]]  : i32
// CHECK: %[[S16:.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: %[[S17:.+]] = llvm.mul %[[S16]], %[[S4]]  : i32
// CHECK: %[[S18:.+]] = llvm.add %[[S15]], %[[S17]]  : i32
// CHECK: %[[S19:.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: %[[S20:.+]] = llvm.mul %[[S19]], %[[S4]]  : i32
// CHECK: %[[S21:.+]] = llvm.add %[[S13]], %[[S20]]  : i32
// CHECK: %[[S22:.+]] = arith.index_cast %[[S18]] : i32 to index
// CHECK: %[[S23:.+]] = arith.index_cast %[[S21]] : i32 to index
// CHECK: %[[S24:.+]] = llvm.add %[[S21]], %[[S6]]  : i32
// CHECK: %[[S25:.+]] = arith.index_cast %[[S24]] : i32 to index
// CHECK: %[[S26:.+]] = llvm.extractvalue %[[EX1]][0] : !llvm.struct
// CHECK: %[[S27:.+]] = llvm.extractvalue %[[EX1]][1] : !llvm.struct
// CHECK: memref.store %[[S26]], %[[arg2]][%[[S22]], %[[S23]]] : memref<128x128xf32, 3>
// CHECK: memref.store %[[S27]], %[[arg2]][%[[S22]], %[[S25]]] : memref<128x128xf32, 3>

// ### Store {d2, d3} of each thread ###

// CHECK: %[[S28:.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: %[[S29:.+]] = llvm.mul %[[S28]], %[[S4]]  : i32
// CHECK: %[[S30:.+]] = llvm.add %[[S13]], %[[S29]]  : i32
// CHECK: %[[S31:.+]] = arith.index_cast %[[S18]] : i32 to index
// CHECK: %[[S32:.+]] = arith.index_cast %[[S30]] : i32 to index
// CHECK: %[[S33:.+]] = llvm.add %[[S30]], %[[S6]]  : i32
// CHECK: %[[S34:.+]] = arith.index_cast %[[S33]] : i32 to index
// CHECK: %[[S35:.+]] = llvm.extractvalue %[[EX1]][4] : !llvm.struct<
// CHECK: %[[S36:.+]] = llvm.extractvalue %[[EX1]][5] : !llvm.struct<
// CHECK: memref.store %[[S35]], %[[arg2]][%[[S31]], %[[S32]]] : memref<128x128xf32, 3>
// CHECK: memref.store %[[S36]], %[[arg2]][%[[S31]], %[[S34]]] : memref<128x128xf32, 3>

// ### Store {d4, d5} of each thread ### 

// CHECK: %[[S37:.+]] = llvm.mlir.constant(2 : i32) : i32
// CHECK: %[[S38:.+]] = llvm.mul %[[S37]], %[[S4]]  : i32
// CHECK: %[[S39:.+]] = llvm.add %[[S13]], %[[S38]]  : i32
// CHECK: %[[S40:.+]] = arith.index_cast %[[S18]] : i32 to index
// CHECK: %[[S41:.+]] = arith.index_cast %[[S39]] : i32 to index
// CHECK: %[[S42:.+]] = llvm.add %[[S39]], %[[S6]]  : i32
// CHECK: %[[S43:.+]] = arith.index_cast %[[S42]] : i32 to index
// CHECK: %[[S44:.+]] = llvm.extractvalue %[[EX1]][8] : !llvm.struct<
// CHECK: %[[S45:.+]] = llvm.extractvalue %[[EX1]][9] : !llvm.struct<
// CHECK: memref.store %[[S44]], %[[arg2]][%[[S40]], %[[S41]]] : memref<128x128xf32, 3>
// CHECK: memref.store %[[S45]], %[[arg2]][%[[S40]], %[[S43]]] : memref<128x128xf32, 3>

// ### Store {d6, d7} of each thread ### 

// CHECK: %[[S46:.+]] = llvm.mlir.constant(3 : i32) : i32
// CHECK: %[[S47:.+]] = llvm.mul %[[S46]], %[[S4]]  : i32
// CHECK: %[[S48:.+]] = llvm.add %[[S13]], %[[S47]]  : i32
// CHECK: %[[S49:.+]] = arith.index_cast %[[S18]] : i32 to index
// CHECK: %[[S50:.+]] = arith.index_cast %[[S48]] : i32 to index
// CHECK: %[[S51:.+]] = llvm.add %[[S48]], %[[S6]]  : i32
// CHECK: %[[S52:.+]] = arith.index_cast %[[S51]] : i32 to index
// CHECK: %[[S53:.+]] = llvm.extractvalue %[[EX1]][12] : !llvm.struct<
// CHECK: %[[S54:.+]] = llvm.extractvalue %[[EX1]][13] : !llvm.struct<
// CHECK: memref.store %[[S53]], %[[arg2]][%[[S49]], %[[S50]]] : memref<128x128xf32, 3>
// CHECK: memref.store %[[S54]], %[[arg2]][%[[S49]], %[[S52]]] : memref<128x128xf32, 3>

// Pattern continues similarly 28x times until {... d62, d63}

// CHECK: %[[c1:.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: %[[c2:.+]] = llvm.mlir.constant(2 : i32) : i32

// ### Store {d64, d65} of each thread ### 
// CHECK: %[[EX2:.+]] = llvm.extractvalue %[[S0]][1] : !llvm.struct<(struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>, struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>)> 
// CHECK: %[[S315:.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: %[[S312:.+]] = llvm.mlir.constant(2 : i32) : i32
// CHECK: %[[S311:.+]] = llvm.mlir.constant(4 : i32) : i32
// CHECK: %[[S313:.+]] = llvm.mlir.constant(8 : i32) : i32
// CHECK: %[[S316:.+]] = llvm.mlir.constant(16 : i32) : i32
// CHECK: %[[WS2:.+]] = llvm.mlir.constant(32 : i32) : i32
// CHECK: %[[S317:.+]] = nvvm.read.ptx.sreg.tid.x : i32
// CHECK: %[[S318:.+]] = llvm.urem %[[S317]], %[[WS2]]  : i32
// CHECK: %[[S319:.+]] = llvm.udiv %[[S317]], %[[WS2]]  : i32
// CHECK: %[[S320:.+]] = llvm.udiv %[[S318]], %[[S311]]  : i32
// CHECK: %[[S321:.+]] = llvm.urem %[[S318]], %[[S311]]  : i32
// CHECK: %[[S322:.+]] = llvm.mul %[[S321]], %[[S312]]  : i32
// CHECK: %[[S323:.+]] = llvm.mul %[[S319]], %[[S316]]  : i32
// CHECK: %[[S324:.+]] = llvm.add %[[S320]], %[[S323]]  : i32
// CHECK: %[[S325:.+]] = llvm.mlir.constant(64 : i32) : i32
// CHECK: %[[S326:.+]] = llvm.add %[[S324]], %[[S325]]  : i32
// CHECK: %[[S327:.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: %[[S328:.+]] = llvm.mul %[[S327]], %[[S313]]  : i32
// CHECK: %[[S329:.+]] = llvm.add %[[S326]], %[[S328]]  : i32
// CHECK: %[[S330:.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: %[[S331:.+]] = llvm.mul %[[S330]], %[[S313]]  : i32
// CHECK: %[[S332:.+]] = llvm.add %[[S322]], %[[S331]]  : i32
// CHECK: %[[S333:.+]] = arith.index_cast %[[S329]] : i32 to index
// CHECK: %[[S334:.+]] = arith.index_cast %[[S332]] : i32 to index
// CHECK: %[[S335:.+]] = llvm.add %[[S332]], %[[S315]]  : i32
// CHECK: %[[S336:.+]] = arith.index_cast %[[S335]] : i32 to index
// CHECK: %[[S337:.+]] = llvm.extractvalue %[[EX2]][0] 
// CHECK: %[[S338:.+]] = llvm.extractvalue %[[EX2]][1]  
// CHECK: memref.store %[[S337]], %[[arg2]][%[[S333]], %[[S334]]] : memref<128x128xf32, 3>
// CHECK: memref.store %[[S338]], %[[arg2]][%[[S333]], %[[S336]]] : memref<128x128xf32, 3>

// Pattern continues similarly 31x times until {... d126, d127}

  nvgpu.warpgroup.mma.store %result, %matrixD : 
    !nvgpu.warpgroup.accumulator< fragmented = vector<128x128xf32>> 
    to memref<128x128xf32,3>
  return 
}

func.func @warpgroup_mma_init() {
  //CHECK: %[[S1:.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f3
  //CHECK: %[[S0:.+]] = llvm.mlir.undef : !llvm.struct<(struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>, struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>)>
  //CHECK: %[[EX:.+]] = llvm.extractvalue %[[S0]][0] : !llvm.struct<(struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>, struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>)> 
  //CHECK: %[[S2:.+]] = llvm.insertvalue %[[S1]], %[[EX]][0] : !llvm.struct
  //CHECK: %[[S3:.+]] = llvm.insertvalue %[[S1]], %[[S2]][1] : !llvm.struct
  //CHECK: %[[S4:.+]] = llvm.insertvalue %[[S1]], %[[S3]][2] : !llvm.struct
  //CHECK: %[[S5:.+]] = llvm.insertvalue %[[S1]], %[[S4]][3] : !llvm.struct
  //CHECK: %[[S6:.+]] = llvm.insertvalue %[[S1]], %[[S5]][4] : !llvm.struct
  //CHECK: %[[S7:.+]] = llvm.insertvalue %[[S1]], %[[S6]][5] : !llvm.struct
  //CHECK: %[[S8:.+]] = llvm.insertvalue %[[S1]], %[[S7]][6] : !llvm.struct
  //CHECK: %[[S9:.+]] = llvm.insertvalue %[[S1]], %[[S8]][7] : !llvm.struct
  //CHECK: %[[S10:.+]] = llvm.insertvalue %[[S1]], %[[S9]][8] : !llvm.struct
  //CHECK: %[[S11:.+]] = llvm.insertvalue %[[S1]], %[[S10]][9] : !llvm.struct
  //CHECK: %[[S12:.+]] = llvm.insertvalue %[[S1]], %[[S11]][10] : !llvm.struct
  //CHECK: %[[S13:.+]] = llvm.insertvalue %[[S1]], %[[S12]][11] : !llvm.struct
  //CHECK: %[[S14:.+]] = llvm.insertvalue %[[S1]], %[[S13]][12] : !llvm.struct
  //CHECK: %[[S15:.+]] = llvm.insertvalue %[[S1]], %[[S14]][13] : !llvm.struct
  //CHECK: %[[S16:.+]] = llvm.insertvalue %[[S1]], %[[S15]][14] : !llvm.struct
  //CHECK: %[[S17:.+]] = llvm.insertvalue %[[S1]], %[[S16]][15] : !llvm.struct
  //CHECK: %[[S18:.+]] = llvm.insertvalue %[[S1]], %[[S17]][16] : !llvm.struct
  //CHECK: %[[S19:.+]] = llvm.insertvalue %[[S1]], %[[S18]][17] : !llvm.struct
  //CHECK: %[[S20:.+]] = llvm.insertvalue %[[S1]], %[[S19]][18] : !llvm.struct
  //CHECK: %[[S21:.+]] = llvm.insertvalue %[[S1]], %[[S20]][19] : !llvm.struct
  //CHECK: %[[S22:.+]] = llvm.insertvalue %[[S1]], %[[S21]][20] : !llvm.struct
  //CHECK: %[[S23:.+]] = llvm.insertvalue %[[S1]], %[[S22]][21] : !llvm.struct
  //CHECK: %[[S24:.+]] = llvm.insertvalue %[[S1]], %[[S23]][22] : !llvm.struct
  //CHECK: %[[S25:.+]] = llvm.insertvalue %[[S1]], %[[S24]][23] : !llvm.struct
  //CHECK: %[[S26:.+]] = llvm.insertvalue %[[S1]], %[[S25]][24] : !llvm.struct
  //CHECK: %[[S27:.+]] = llvm.insertvalue %[[S1]], %[[S26]][25] : !llvm.struct
  //CHECK: %[[S28:.+]] = llvm.insertvalue %[[S1]], %[[S27]][26] : !llvm.struct
  //CHECK: %[[S29:.+]] = llvm.insertvalue %[[S1]], %[[S28]][27] : !llvm.struct
  //CHECK: %[[S30:.+]] = llvm.insertvalue %[[S1]], %[[S29]][28] : !llvm.struct
  //CHECK: %[[S31:.+]] = llvm.insertvalue %[[S1]], %[[S30]][29] : !llvm.struct
  //CHECK: %[[S32:.+]] = llvm.insertvalue %[[S1]], %[[S31]][30] : !llvm.struct
  //CHECK: %[[S33:.+]] = llvm.insertvalue %[[S1]], %[[S32]][31] : !llvm.struct
  //CHECK: %[[S34:.+]] = llvm.insertvalue %[[S1]], %[[S33]][32] : !llvm.struct
  //CHECK: %[[S35:.+]] = llvm.insertvalue %[[S1]], %[[S34]][33] : !llvm.struct
  //CHECK: %[[S36:.+]] = llvm.insertvalue %[[S1]], %[[S35]][34] : !llvm.struct
  //CHECK: %[[S37:.+]] = llvm.insertvalue %[[S1]], %[[S36]][35] : !llvm.struct
  //CHECK: %[[S38:.+]] = llvm.insertvalue %[[S1]], %[[S37]][36] : !llvm.struct
  //CHECK: %[[S39:.+]] = llvm.insertvalue %[[S1]], %[[S38]][37] : !llvm.struct
  //CHECK: %[[S40:.+]] = llvm.insertvalue %[[S1]], %[[S39]][38] : !llvm.struct
  //CHECK: %[[S41:.+]] = llvm.insertvalue %[[S1]], %[[S40]][39] : !llvm.struct
  //CHECK: %[[S42:.+]] = llvm.insertvalue %[[S1]], %[[S41]][40] : !llvm.struct
  //CHECK: %[[S43:.+]] = llvm.insertvalue %[[S1]], %[[S42]][41] : !llvm.struct
  //CHECK: %[[S44:.+]] = llvm.insertvalue %[[S1]], %[[S43]][42] : !llvm.struct
  //CHECK: %[[S45:.+]] = llvm.insertvalue %[[S1]], %[[S44]][43] : !llvm.struct
  //CHECK: %[[S46:.+]] = llvm.insertvalue %[[S1]], %[[S45]][44] : !llvm.struct
  //CHECK: %[[S47:.+]] = llvm.insertvalue %[[S1]], %[[S46]][45] : !llvm.struct
  //CHECK: %[[S48:.+]] = llvm.insertvalue %[[S1]], %[[S47]][46] : !llvm.struct
  //CHECK: %[[S49:.+]] = llvm.insertvalue %[[S1]], %[[S48]][47] : !llvm.struct
  //CHECK: %[[S50:.+]] = llvm.insertvalue %[[S1]], %[[S49]][48] : !llvm.struct
  //CHECK: %[[S51:.+]] = llvm.insertvalue %[[S1]], %[[S50]][49] : !llvm.struct
  //CHECK: %[[S52:.+]] = llvm.insertvalue %[[S1]], %[[S51]][50] : !llvm.struct
  //CHECK: %[[S53:.+]] = llvm.insertvalue %[[S1]], %[[S52]][51] : !llvm.struct
  //CHECK: %[[S54:.+]] = llvm.insertvalue %[[S1]], %[[S53]][52] : !llvm.struct
  //CHECK: %[[S55:.+]] = llvm.insertvalue %[[S1]], %[[S54]][53] : !llvm.struct
  //CHECK: %[[S56:.+]] = llvm.insertvalue %[[S1]], %[[S55]][54] : !llvm.struct
  //CHECK: %[[S57:.+]] = llvm.insertvalue %[[S1]], %[[S56]][55] : !llvm.struct
  //CHECK: %[[S58:.+]] = llvm.insertvalue %[[S1]], %[[S57]][56] : !llvm.struct
  //CHECK: %[[S59:.+]] = llvm.insertvalue %[[S1]], %[[S58]][57] : !llvm.struct
  //CHECK: %[[S60:.+]] = llvm.insertvalue %[[S1]], %[[S59]][58] : !llvm.struct
  //CHECK: %[[S61:.+]] = llvm.insertvalue %[[S1]], %[[S60]][59] : !llvm.struct
  //CHECK: %[[S62:.+]] = llvm.insertvalue %[[S1]], %[[S61]][60] : !llvm.struct
  //CHECK: %[[S63:.+]] = llvm.insertvalue %[[S1]], %[[S62]][61] : !llvm.struct
  //CHECK: %[[S64:.+]] = llvm.insertvalue %[[S1]], %[[S63]][62] : !llvm.struct
  //CHECK: %[[S65:.+]] = llvm.insertvalue %[[S1]], %[[S64]][63] : !llvm.struct
  %matrixC = nvgpu.warpgroup.mma.init.accumulator -> !nvgpu.warpgroup.accumulator< fragmented = vector<128x128xf32>>
  return 
}


transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["func.func"]} in %arg1 
    : (!transform.any_op) -> !transform.any_op
  transform.apply_conversion_patterns to %0 {
    transform.apply_conversion_patterns.nvgpu.nvgpu_to_nvvm
  } with type_converter {
    transform.apply_conversion_patterns.memref.memref_to_llvm_type_converter
      {use_opaque_pointers = true}
  } {legal_dialects = ["arith", "func", "llvm", "memref", "nvvm", "vector", "scf"], partial_conversion} : !transform.any_op
}
