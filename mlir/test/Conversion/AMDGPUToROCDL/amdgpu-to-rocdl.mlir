// RUN: mlir-opt %s -convert-amdgpu-to-rocdl=chipset=gfx908 | FileCheck %s
// RUN: mlir-opt %s -convert-amdgpu-to-rocdl=chipset=gfx1030 | FileCheck %s --check-prefix=RDNA

// CHECK-LABEL: func @gpu_gcn_raw_buffer_load_scalar_i32
func.func @gpu_gcn_raw_buffer_load_scalar_i32(%buf: memref<i32>) -> i32 {
  // CHECK: %[[ptr:.*]] = llvm.ptrtoint
  // CHECK: %[[lowHalf:.*]] = llvm.trunc %[[ptr]] : i64 to i32
  // CHECK: %[[resource_1:.*]] = llvm.insertelement %[[lowHalf]]
  // CHECK: %[[highHalfI64:.*]] = llvm.lshr %[[ptr]]
  // CHECK: %[[highHalfI32:.*]] = llvm.trunc %[[highHalfI64]] : i64 to i32
  // CHECK: %[[highHalf:.*]] = llvm.and %[[highHalfI32]], %{{.*}} : i32
  // CHECK: %[[resource_2:.*]] = llvm.insertelement %[[highHalf]], %[[resource_1]]
  // CHECK: %[[numRecords:.*]] = llvm.mlir.constant(4 : i32)
  // CHECK: %[[resource_3:.*]] = llvm.insertelement %[[numRecords]], %[[resource_2]]
  // CHECK: %[[word3:.*]] = llvm.mlir.constant(159744 : i32)
  // RDNA: %[[word3:.*]] = llvm.mlir.constant(822243328 : i32)
  // CHECK: %[[resource:.*]] = llvm.insertelement %[[word3]], %[[resource_3]]
  // CHECK: %[[ret:.*]] = rocdl.raw.buffer.load %[[resource]], %{{.*}}, %{{.*}}, %{{.*}} : i32
  // CHECK: return %[[ret]]
  %0 = amdgpu.raw_buffer_load {boundsCheck = true} %buf[] : memref<i32> -> i32
  func.return %0 : i32
}

// CHECK-LABEL: func @gpu_gcn_raw_buffer_load_i32
func.func @gpu_gcn_raw_buffer_load_i32(%buf: memref<64xi32>, %idx: i32) -> i32 {
  // CHECK: %[[ptr:.*]] = llvm.ptrtoint
  // CHECK: %[[lowHalf:.*]] = llvm.trunc %[[ptr]] : i64 to i32
  // CHECK: %[[resource_1:.*]] = llvm.insertelement %[[lowHalf]]
  // CHECK: %[[highHalfI64:.*]] = llvm.lshr %[[ptr]]
  // CHECK: %[[highHalfI32:.*]] = llvm.trunc %[[highHalfI64]] : i64 to i32
  // CHECK: %[[highHalf:.*]] = llvm.and %[[highHalfI32]], %{{.*}} : i32
  // CHECK: %[[resource_2:.*]] = llvm.insertelement %[[highHalf]], %[[resource_1]]
  // CHECK: %[[numRecords:.*]] = llvm.mlir.constant(256 : i32)
  // CHECK: %[[resource_3:.*]] = llvm.insertelement %[[numRecords]], %[[resource_2]]
  // CHECK: %[[word3:.*]] = llvm.mlir.constant(159744 : i32)
  // RDNA: %[[word3:.*]] = llvm.mlir.constant(822243328 : i32)
  // CHECK: %[[resource:.*]] = llvm.insertelement %[[word3]], %[[resource_3]]
  // CHECK: %[[ret:.*]] = rocdl.raw.buffer.load %[[resource]], %{{.*}}, %{{.*}}, %{{.*}} : i32
  // CHECK: return %[[ret]]
  %0 = amdgpu.raw_buffer_load {boundsCheck = true} %buf[%idx] : memref<64xi32>, i32 -> i32
  func.return %0 : i32
}

// CHECK-LABEL: func @gpu_gcn_raw_buffer_load_i32_oob_off
func.func @gpu_gcn_raw_buffer_load_i32_oob_off(%buf: memref<64xi32>, %idx: i32) -> i32 {
  // CHECK: %[[word3:.*]] = llvm.mlir.constant(159744 : i32)
  // RDNA: %[[word3:.*]] = llvm.mlir.constant(553807872 : i32)
  // RDNA: %[[resource:.*]] = llvm.insertelement{{.*}}%[[word3]]
  // RDNA: %[[ret:.*]] = rocdl.raw.buffer.load %[[resource]], %{{.*}}, %{{.*}}, %{{.*}} : i32
  // RDNA: return %[[ret]]
  %0 = amdgpu.raw_buffer_load {boundsCheck = false} %buf[%idx] : memref<64xi32>, i32 -> i32
  func.return %0 : i32
}

// CHECK-LABEL: func @gpu_gcn_raw_buffer_load_2xi32
func.func @gpu_gcn_raw_buffer_load_2xi32(%buf: memref<64xi32>, %idx: i32) -> vector<2xi32> {
  // CHECK: %[[ret:.*]] = rocdl.raw.buffer.load %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : vector<2xi32>
  // CHECK: return %[[ret]]
  %0 = amdgpu.raw_buffer_load {boundsCheck = true} %buf[%idx] : memref<64xi32>, i32 -> vector<2xi32>
  func.return %0 : vector<2xi32>
}

// CHECK-LABEL: func @gpu_gcn_raw_buffer_load_i8
func.func @gpu_gcn_raw_buffer_load_i8(%buf: memref<64xi8>, %idx: i32) -> i8 {
  // CHECK: %[[numRecords:.*]] = llvm.mlir.constant(64 : i32)
  // CHECK: llvm.insertelement{{.*}}%[[numRecords]]
  // CHECK: %[[ret:.*]] = rocdl.raw.buffer.load %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : i8
  // CHECK: return %[[ret]]
  %0 = amdgpu.raw_buffer_load {boundsCheck = true} %buf[%idx] : memref<64xi8>, i32 -> i8
  func.return %0 : i8
}

// CHECK-LABEL: func @gpu_gcn_raw_buffer_load_2xi8
func.func @gpu_gcn_raw_buffer_load_2xi8(%buf: memref<64xi8>, %idx: i32) -> vector<2xi8> {
  // CHECK: %[[numRecords:.*]] = llvm.mlir.constant(64 : i32)
  // CHECK: llvm.insertelement{{.*}}%[[numRecords]]
  // CHECK: %[[loaded:.*]] = rocdl.raw.buffer.load %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : i16
  // CHECK: %[[ret:.*]] = llvm.bitcast %[[loaded]] : i16 to vector<2xi8>
  // CHECK: return %[[ret]]
  %0 = amdgpu.raw_buffer_load {boundsCheck = true} %buf[%idx] : memref<64xi8>, i32 -> vector<2xi8>
  func.return %0 : vector<2xi8>
}

// CHECK-LABEL: func @gpu_gcn_raw_buffer_load_16xi8
func.func @gpu_gcn_raw_buffer_load_16xi8(%buf: memref<64xi8>, %idx: i32) -> vector<16xi8> {
  // CHECK: %[[loaded:.*]] = rocdl.raw.buffer.load %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : vector<4xi32>
  // CHECK: %[[ret:.*]] = llvm.bitcast %[[loaded]] : vector<4xi32> to vector<16xi8>
  // CHECK: return %[[ret]]
  %0 = amdgpu.raw_buffer_load {boundsCheck = true} %buf[%idx] : memref<64xi8>, i32 -> vector<16xi8>
  func.return %0 : vector<16xi8>
}

// CHECK-LABEL: func @gpu_gcn_raw_buffer_load_f8E5M2FNUZ
func.func @gpu_gcn_raw_buffer_load_f8E5M2FNUZ(%buf: memref<64xf8E5M2FNUZ>, %idx: i32) -> f8E5M2FNUZ {
  // CHECK: %[[numRecords:.*]] = llvm.mlir.constant(64 : i32)
  // CHECK: llvm.insertelement{{.*}}%[[numRecords]]
  // CHECK: %[[loaded:.*]] = rocdl.raw.buffer.load %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : i8
  // CHECK: %[[ret:.*]] = builtin.unrealized_conversion_cast %[[loaded]] : i8 to f8E5M2FNUZ
  // CHECK: return %[[ret]]
  %0 = amdgpu.raw_buffer_load {boundsCheck = true} %buf[%idx] : memref<64xf8E5M2FNUZ>, i32 -> f8E5M2FNUZ
  func.return %0 : f8E5M2FNUZ
}

// CHECK-LABEL: func @gpu_gcn_raw_buffer_load_4xf8E4M3FNUZ
func.func @gpu_gcn_raw_buffer_load_4xf8E4M3FNUZ(%buf: memref<64xf8E4M3FNUZ>, %idx: i32) -> vector<4xf8E4M3FNUZ> {
  // CHECK: %[[numRecords:.*]] = llvm.mlir.constant(64 : i32)
  // CHECK: llvm.insertelement{{.*}}%[[numRecords]]
  // CHECK: %[[loaded:.*]] = rocdl.raw.buffer.load %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : i32
  // CHECK: %[[cast:.*]] = llvm.bitcast %[[loaded]] : i32 to vector<4xi8>
  // CHECK: %[[ret:.*]] = builtin.unrealized_conversion_cast %[[cast]] : vector<4xi8> to vector<4xf8E4M3FNUZ>
  // CHECK: return %[[ret]]
  %0 = amdgpu.raw_buffer_load {boundsCheck = true} %buf[%idx] : memref<64xf8E4M3FNUZ>, i32 -> vector<4xf8E4M3FNUZ>
  func.return %0 : vector<4xf8E4M3FNUZ>
}

// Since the lowering logic is shared with loads, only bitcasts need to be rechecked
// CHECK-LABEL: func @gpu_gcn_raw_buffer_store_scalar_i32
func.func @gpu_gcn_raw_buffer_store_scalar_i32(%value: i32, %buf: memref<i32>) {
  // CHECK: %[[numRecords:.*]] = llvm.mlir.constant(4 : i32)
  // CHECK: llvm.insertelement{{.*}}%[[numRecords]]
  // CHECK: %[[word3:.*]] = llvm.mlir.constant(159744 : i32)
  // CHECK: %[[resource:.*]] = llvm.insertelement{{.*}}%[[word3]]
  // CHECK: rocdl.raw.buffer.store %{{.*}} %[[resource]], %{{.*}}, %{{.*}}, %{{.*}} : i32
  amdgpu.raw_buffer_store {boundsCheck = true} %value -> %buf[] : i32 -> memref<i32>
  func.return
}

// CHECK-LABEL: func @gpu_gcn_raw_buffer_store_i32
func.func @gpu_gcn_raw_buffer_store_i32(%value: i32, %buf: memref<64xi32>, %idx: i32) {
  // CHECK: %[[numRecords:.*]] = llvm.mlir.constant(256 : i32)
  // CHECK: llvm.insertelement{{.*}}%[[numRecords]]
  // CHECK: %[[word3:.*]] = llvm.mlir.constant(159744 : i32)
  // CHECK: %[[resource:.*]] = llvm.insertelement{{.*}}%[[word3]]
  // CHECK: rocdl.raw.buffer.store %{{.*}} %[[resource]], %{{.*}}, %{{.*}}, %{{.*}} : i32
  amdgpu.raw_buffer_store {boundsCheck = true} %value -> %buf[%idx] : i32 -> memref<64xi32>, i32
  func.return
}

// CHECK-LABEL: func @gpu_gcn_raw_buffer_store_2xi8
func.func @gpu_gcn_raw_buffer_store_2xi8(%value: vector<2xi8>, %buf: memref<64xi8>, %idx: i32) {
  // CHECK: %[[cast:.*]] = llvm.bitcast %{{.*}} : vector<2xi8> to i16
  // CHECK: rocdl.raw.buffer.store %[[cast]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : i16
  amdgpu.raw_buffer_store {boundsCheck = true} %value -> %buf[%idx] : vector<2xi8> -> memref<64xi8>, i32
  func.return
}

// CHECK-LABEL: func @gpu_gcn_raw_buffer_store_16xi8
func.func @gpu_gcn_raw_buffer_store_16xi8(%value: vector<16xi8>, %buf: memref<64xi8>, %idx: i32) {
  // CHECK: %[[cast:.*]] = llvm.bitcast %{{.*}} : vector<16xi8> to vector<4xi32>
  // CHECK: rocdl.raw.buffer.store %[[cast]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : vector<4xi32>
  amdgpu.raw_buffer_store {boundsCheck = true} %value -> %buf[%idx] : vector<16xi8> -> memref<64xi8>, i32
  func.return
}

// And more so for atomic add
// CHECK-LABEL: func @gpu_gcn_raw_buffer_atomic_fadd_f32
func.func @gpu_gcn_raw_buffer_atomic_fadd_f32(%value: f32, %buf: memref<64xf32>, %idx: i32) {
  // CHECK: %[[numRecords:.*]] = llvm.mlir.constant(256 : i32)
  // CHECK: llvm.insertelement{{.*}}%[[numRecords]]
  // CHECK: %[[word3:.*]] = llvm.mlir.constant(159744 : i32)
  // CHECK: %[[resource:.*]] = llvm.insertelement{{.*}}%[[word3]]
  // CHECK: rocdl.raw.buffer.atomic.fadd %{{.*}} %[[resource]], %{{.*}}, %{{.*}}, %{{.*}} : f32
  amdgpu.raw_buffer_atomic_fadd {boundsCheck = true} %value -> %buf[%idx] : f32 -> memref<64xf32>, i32
  func.return
}

// CHECK-LABEL: func @gpu_gcn_raw_buffer_atomic_fmax_f32
func.func @gpu_gcn_raw_buffer_atomic_fmax_f32(%value: f32, %buf: memref<64xf32>, %idx: i32) {
  // CHECK: %[[numRecords:.*]] = llvm.mlir.constant(256 : i32)
  // CHECK: llvm.insertelement{{.*}}%[[numRecords]]
  // CHECK: %[[word3:.*]] = llvm.mlir.constant(159744 : i32)
  // CHECK: %[[resource:.*]] = llvm.insertelement{{.*}}%[[word3]]
  // CHECK: rocdl.raw.buffer.atomic.fmax %{{.*}} %[[resource]], %{{.*}}, %{{.*}}, %{{.*}} : f32
  amdgpu.raw_buffer_atomic_fmax {boundsCheck = true} %value -> %buf[%idx] : f32 -> memref<64xf32>, i32
  func.return
}

// CHECK-LABEL: func @gpu_gcn_raw_buffer_atomic_smax_i32
func.func @gpu_gcn_raw_buffer_atomic_smax_i32(%value: i32, %buf: memref<64xi32>, %idx: i32) {
  // CHECK: %[[numRecords:.*]] = llvm.mlir.constant(256 : i32)
  // CHECK: llvm.insertelement{{.*}}%[[numRecords]]
  // CHECK: %[[word3:.*]] = llvm.mlir.constant(159744 : i32)
  // CHECK: %[[resource:.*]] = llvm.insertelement{{.*}}%[[word3]]
  // CHECK: rocdl.raw.buffer.atomic.smax %{{.*}} %[[resource]], %{{.*}}, %{{.*}}, %{{.*}} : i32
  amdgpu.raw_buffer_atomic_smax {boundsCheck = true} %value -> %buf[%idx] : i32 -> memref<64xi32>, i32
  func.return
}

// CHECK-LABEL: func @gpu_gcn_raw_buffer_atomic_umin_i32
func.func @gpu_gcn_raw_buffer_atomic_umin_i32(%value: i32, %buf: memref<64xi32>, %idx: i32) {
  // CHECK: %[[numRecords:.*]] = llvm.mlir.constant(256 : i32)
  // CHECK: llvm.insertelement{{.*}}%[[numRecords]]
  // CHECK: %[[word3:.*]] = llvm.mlir.constant(159744 : i32)
  // CHECK: %[[resource:.*]] = llvm.insertelement{{.*}}%[[word3]]
  // CHECK: rocdl.raw.buffer.atomic.umin %{{.*}} %[[resource]], %{{.*}}, %{{.*}}, %{{.*}} : i32
  amdgpu.raw_buffer_atomic_umin {boundsCheck = true} %value -> %buf[%idx] : i32 -> memref<64xi32>, i32
  func.return
}

// CHECK-LABEL: func @amdgpu_raw_buffer_atomic_cmpswap_f32
// CHECK-SAME: (%[[src:.*]]: f32, %[[cmp:.*]]: f32, {{.*}})
func.func @amdgpu_raw_buffer_atomic_cmpswap_f32(%src : f32, %cmp : f32, %buf : memref<64xf32>, %idx: i32) -> f32 {
  // CHECK: %[[srcCast:.*]] = llvm.bitcast %[[src]] : f32 to i32
  // CHECK: %[[cmpCast:.*]] = llvm.bitcast %[[cmp]] : f32 to i32
  // CHECK: %[[numRecords:.*]] = llvm.mlir.constant(256 : i32)
  // CHECK: llvm.insertelement{{.*}}%[[numRecords]]
  // CHECK: %[[word3:.*]] = llvm.mlir.constant(159744 : i32)
  // CHECK: %[[resource:.*]] = llvm.insertelement{{.*}}%[[word3]]
  // CHECK: %[[dst:.*]] = rocdl.raw.buffer.atomic.cmpswap(%[[srcCast]], %[[cmpCast]], %[[resource]], %{{.*}}, %{{.*}}, %{{.*}}) : i32, vector<4xi32>
  // CHECK: %[[dstCast:.*]] = llvm.bitcast %[[dst]] : i32 to f32
  // CHECK: return %[[dstCast]]
  %dst = amdgpu.raw_buffer_atomic_cmpswap {boundsCheck = true} %src, %cmp -> %buf[%idx] : f32 -> memref<64xf32>, i32
  func.return %dst : f32
}

// CHECK-LABEL: func @amdgpu_raw_buffer_atomic_cmpswap_i64
// CHECK-SAME: (%[[src:.*]]: i64, %[[cmp:.*]]: i64, {{.*}})
func.func @amdgpu_raw_buffer_atomic_cmpswap_i64(%src : i64, %cmp : i64, %buf : memref<64xi64>, %idx: i32) -> i64 {
  // CHECK: %[[numRecords:.*]] = llvm.mlir.constant(512 : i32)
  // CHECK: llvm.insertelement{{.*}}%[[numRecords]]
  // CHECK: %[[word3:.*]] = llvm.mlir.constant(159744 : i32)
  // CHECK: %[[resource:.*]] = llvm.insertelement{{.*}}%[[word3]]
  // CHECK: %[[dst:.*]] = rocdl.raw.buffer.atomic.cmpswap(%[[src]], %[[cmp]], %[[resource]], %{{.*}}, %{{.*}}, %{{.*}}) : i64, vector<4xi32>
  // CHECK: return %[[dst]]
  %dst = amdgpu.raw_buffer_atomic_cmpswap {boundsCheck = true} %src, %cmp -> %buf[%idx] : i64 -> memref<64xi64>, i32
  func.return %dst : i64
}

// CHECK-LABEL: func @lds_barrier
func.func @lds_barrier() {
  // CHECK: llvm.inline_asm has_side_effects asm_dialect = att "s_waitcnt lgkmcnt(0)\0As_barrier"
  amdgpu.lds_barrier
  func.return
}
