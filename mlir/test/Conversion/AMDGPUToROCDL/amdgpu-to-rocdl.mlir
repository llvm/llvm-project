// RUN: mlir-opt %s -convert-amdgpu-to-rocdl=chipset=gfx908 | FileCheck %s --check-prefixes=CHECK,GFX9,GFX908
// RUN: mlir-opt %s -convert-amdgpu-to-rocdl=chipset=gfx90a | FileCheck %s --check-prefixes=CHECK,GFX9,GFX90A
// RUN: mlir-opt %s -convert-amdgpu-to-rocdl=chipset=gfx1030 | FileCheck %s --check-prefixes=CHECK,GFX10,RDNA
// RUN: mlir-opt %s -convert-amdgpu-to-rocdl=chipset=gfx1100 | FileCheck %s --check-prefixes=CHECK,GFX11,RDNA
// RUN: mlir-opt %s -convert-amdgpu-to-rocdl=chipset=gfx1201 | FileCheck %s --check-prefixes=CHECK,GFX12,RDNA

// CHECK-LABEL: func @gpu_gcn_raw_buffer_load_scalar_i32
func.func @gpu_gcn_raw_buffer_load_scalar_i32(%buf: memref<i32>) -> i32 {
  // CHECK: %[[stride:.*]] = llvm.mlir.constant(0 : i16)
  // CHECK: %[[numRecords:.*]] = llvm.mlir.constant(4 : i32)
  // GFX9:  %[[flags:.*]] = llvm.mlir.constant(159744 : i32)
  // RDNA:  %[[flags:.*]] = llvm.mlir.constant(822243328 : i32)
  // CHECK: %[[resource:.*]] = rocdl.make.buffer.rsrc %{{.*}}, %[[stride]], %[[numRecords]], %[[flags]] : !llvm.ptr to <8>
  // CHECK: %[[ret:.*]] = rocdl.raw.ptr.buffer.load %[[resource]], %{{.*}}, %{{.*}}, %{{.*}} : i32
  // CHECK: return %[[ret]]
  %0 = amdgpu.raw_buffer_load {boundsCheck = true} %buf[] : memref<i32> -> i32
  func.return %0 : i32
}

// CHECK-LABEL: func @gpu_gcn_raw_buffer_load_i32
func.func @gpu_gcn_raw_buffer_load_i32(%buf: memref<64xi32>, %idx: i32) -> i32 {
  // CHECK: %[[stride:.*]] = llvm.mlir.constant(0 : i16)
  // CHECK: %[[numRecords:.*]] = llvm.mlir.constant(256 : i32)
  // GFX9:  %[[flags:.*]] = llvm.mlir.constant(159744 : i32)
  // RDNA:  %[[flags:.*]] = llvm.mlir.constant(822243328 : i32)
  // CHECK: %[[resource:.*]] = rocdl.make.buffer.rsrc %{{.*}}, %[[stride]], %[[numRecords]], %[[flags]] : !llvm.ptr to <8>
  // CHECK: %[[ret:.*]] = rocdl.raw.ptr.buffer.load %[[resource]], %{{.*}}, %{{.*}}, %{{.*}} : i32
  // CHECK: return %[[ret]]
  %0 = amdgpu.raw_buffer_load {boundsCheck = true} %buf[%idx] : memref<64xi32>, i32 -> i32
  func.return %0 : i32
}

// CHECK-LABEL: func @gpu_gcn_raw_buffer_load_i32_strided
func.func @gpu_gcn_raw_buffer_load_i32_strided(%buf: memref<16x16xi32, strided<[?, ?], offset: ?>>, %i: i32, %j: i32) -> i32 {
    // CHECK: %[[descriptor:.*]] = builtin.unrealized_conversion_cast %{{.*}} : memref<16x16xi32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK: %[[elem_size:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK: %[[algn_ptr:.*]] = llvm.extractvalue %[[descriptor]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    // CHECK: %[[offset:.*]] = llvm.extractvalue %[[descriptor]][2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    // CHECK: %[[ptr:.*]] = llvm.getelementptr %[[algn_ptr]][%[[offset]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    // CHECK: %[[stride:.*]] = llvm.mlir.constant(0 : i16) : i16
    // CHECK: %[[sz_i:.*]] = llvm.extractvalue %[[descriptor]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    // CHECK: %[[stride_i:.*]] = llvm.extractvalue %[[descriptor]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    // CHECK: %[[ext_i:.*]] = llvm.mul %[[sz_i]], %[[stride_i]] : i64
    // CHECK: %[[sz_j:.*]] = llvm.extractvalue %[[descriptor]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    // CHECK: %[[stride_j:.*]] = llvm.extractvalue %[[descriptor]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    // CHECK: %[[ext_j:.*]] = llvm.mul %[[sz_j]], %[[stride_j]] : i64
    // CHECK: %[[num_records:.*]] = llvm.intr.umax(%[[ext_i]], %[[ext_j]]) : (i64, i64) -> i64
    // CHECK: %[[num_rec_i32:.*]] = llvm.trunc %[[num_records]] : i64 to i32
    // CHECK: %[[num_rec_bytes_i32:.*]] = llvm.mul %[[num_rec_i32]], %[[elem_size]] : i32
    // CHECK: %[[rsrc:.*]] = rocdl.make.buffer.rsrc %[[ptr]], %[[stride]], %[[num_rec_bytes_i32]], %{{.*}} : !llvm.ptr to <8>
    // CHECK: %[[stride_i_1:.*]] = llvm.extractvalue %[[descriptor]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    // CHECK: %[[stride_i_i32:.*]] = llvm.trunc %[[stride_i_1]] : i64 to i32
    // CHECK: %[[t_0:.*]] = llvm.mul %{{.*}}, %[[stride_i_i32]] : i32
    // CHECK: %[[stride_j_1:.*]] = llvm.extractvalue %[[descriptor]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    // CHECK: %[[stride_j_i32:.*]] = llvm.trunc %[[stride_j_1]] : i64 to i32
    // CHECK: %[[t_1:.*]] = llvm.mul %{{.*}}, %[[stride_j_i32]] : i32
    // CHECK: %[[index:.*]] = llvm.add %[[t_0]], %[[t_1]] : i32
    // CHECK: %[[vgpr_off:.*]] = llvm.mul %[[index]], %[[elem_size]] : i32
    // CHECK: %[[zero_0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[sgpr_off:.*]] = llvm.mul %[[zero_0]], %[[elem_size]] : i32
    // CHECK: %[[zero_1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[v:.*]] = rocdl.raw.ptr.buffer.load %[[rsrc]], %[[vgpr_off]], %[[sgpr_off]], %[[zero_1]] : i32
    // CHECK: return %[[v]] : i32
  %0 = amdgpu.raw_buffer_load {boundsCheck = true} %buf[%i, %j] :  memref<16x16xi32, strided<[?, ?], offset: ?>>, i32, i32 -> i32
  func.return %0 : i32
}

// CHECK-LABEL: func @gpu_gcn_raw_buffer_load_i32_oob_off
func.func @gpu_gcn_raw_buffer_load_i32_oob_off(%buf: memref<64xi32>, %idx: i32) -> i32 {
  // GFX9:  %[[flags:.*]] = llvm.mlir.constant(159744 : i32)
  // RDNA:  %[[flags:.*]] = llvm.mlir.constant(553807872 : i32)
  // CHECK: %[[resource:.*]] = rocdl.make.buffer.rsrc %{{.*}}, %{{.*}}, %{{.*}}, %[[flags]]
  // CHECK: %[[ret:.*]] = rocdl.raw.ptr.buffer.load %[[resource]], %{{.*}}, %{{.*}}, %{{.*}} : i32
  // CHECK: return %[[ret]]
  %0 = amdgpu.raw_buffer_load {boundsCheck = false} %buf[%idx] : memref<64xi32>, i32 -> i32
  func.return %0 : i32
}

// CHECK-LABEL: func @gpu_gcn_raw_buffer_load_2xi32
func.func @gpu_gcn_raw_buffer_load_2xi32(%buf: memref<64xi32>, %idx: i32) -> vector<2xi32> {
  // CHECK: %[[ret:.*]] = rocdl.raw.ptr.buffer.load %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : vector<2xi32>
  // CHECK: return %[[ret]]
  %0 = amdgpu.raw_buffer_load {boundsCheck = true} %buf[%idx] : memref<64xi32>, i32 -> vector<2xi32>
  func.return %0 : vector<2xi32>
}

// CHECK-LABEL: func @gpu_gcn_raw_buffer_load_i8
func.func @gpu_gcn_raw_buffer_load_i8(%buf: memref<64xi8>, %idx: i32) -> i8 {
  // CHECK: %[[numRecords:.*]] = llvm.mlir.constant(64 : i32)
  // CHECK: %[[resource:.*]] = rocdl.make.buffer.rsrc %{{.*}}, %{{.*}}, %[[numRecords]], %{{.*}}
  // CHECK: %[[ret:.*]] = rocdl.raw.ptr.buffer.load %[[resource]], %{{.*}}, %{{.*}}, %{{.*}} : i8
  // CHECK: return %[[ret]]
  %0 = amdgpu.raw_buffer_load {boundsCheck = true} %buf[%idx] : memref<64xi8>, i32 -> i8
  func.return %0 : i8
}

// CHECK-LABEL: func @gpu_gcn_raw_buffer_load_2xi8
func.func @gpu_gcn_raw_buffer_load_2xi8(%buf: memref<64xi8>, %idx: i32) -> vector<2xi8> {
  // CHECK: %[[numRecords:.*]] = llvm.mlir.constant(64 : i32)
  // CHECK: %[[resource:.*]] = rocdl.make.buffer.rsrc %{{.*}}, %{{.*}}, %[[numRecords]], %{{.*}}
  // CHECK: %[[loaded:.*]] = rocdl.raw.ptr.buffer.load %[[resource]], %{{.*}}, %{{.*}}, %{{.*}} : i16
  // CHECK: %[[ret:.*]] = llvm.bitcast %[[loaded]] : i16 to vector<2xi8>
  // CHECK: return %[[ret]]
  %0 = amdgpu.raw_buffer_load {boundsCheck = true} %buf[%idx] : memref<64xi8>, i32 -> vector<2xi8>
  func.return %0 : vector<2xi8>
}

// CHECK-LABEL: func @gpu_gcn_raw_buffer_load_16xi8
func.func @gpu_gcn_raw_buffer_load_16xi8(%buf: memref<64xi8>, %idx: i32) -> vector<16xi8> {
  // CHECK: %[[loaded:.*]] = rocdl.raw.ptr.buffer.load %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : vector<4xi32>
  // CHECK: %[[ret:.*]] = llvm.bitcast %[[loaded]] : vector<4xi32> to vector<16xi8>
  // CHECK: return %[[ret]]
  %0 = amdgpu.raw_buffer_load {boundsCheck = true} %buf[%idx] : memref<64xi8>, i32 -> vector<16xi8>
  func.return %0 : vector<16xi8>
}

// CHECK-LABEL: func @gpu_gcn_raw_buffer_load_f8E5M2FNUZ
func.func @gpu_gcn_raw_buffer_load_f8E5M2FNUZ(%buf: memref<64xf8E5M2FNUZ>, %idx: i32) -> f8E5M2FNUZ {
  // CHECK: %[[numRecords:.*]] = llvm.mlir.constant(64 : i32)
  // CHECK: %[[resource:.*]] = rocdl.make.buffer.rsrc %{{.*}}, %{{.*}}, %[[numRecords]], %{{.*}}
  // CHECK: %[[loaded:.*]] = rocdl.raw.ptr.buffer.load %[[resource]], %{{.*}}, %{{.*}}, %{{.*}} : i8
  // CHECK: %[[ret:.*]] = builtin.unrealized_conversion_cast %[[loaded]] : i8 to f8E5M2FNUZ
  // CHECK: return %[[ret]]
  %0 = amdgpu.raw_buffer_load {boundsCheck = true} %buf[%idx] : memref<64xf8E5M2FNUZ>, i32 -> f8E5M2FNUZ
  func.return %0 : f8E5M2FNUZ
}

// CHECK-LABEL: func @gpu_gcn_raw_buffer_load_4xf8E4M3FNUZ
func.func @gpu_gcn_raw_buffer_load_4xf8E4M3FNUZ(%buf: memref<64xf8E4M3FNUZ>, %idx: i32) -> vector<4xf8E4M3FNUZ> {
  // CHECK: %[[numRecords:.*]] = llvm.mlir.constant(64 : i32)
  // CHECK: %[[resource:.*]] = rocdl.make.buffer.rsrc %{{.*}}, %{{.*}}, %[[numRecords]], %{{.*}}
  // CHECK: %[[loaded:.*]] = rocdl.raw.ptr.buffer.load %[[resource]], %{{.*}}, %{{.*}}, %{{.*}} : i32
  // CHECK: %[[cast:.*]] = llvm.bitcast %[[loaded]] : i32 to vector<4xi8>
  // CHECK: %[[ret:.*]] = builtin.unrealized_conversion_cast %[[cast]] : vector<4xi8> to vector<4xf8E4M3FNUZ>
  // CHECK: return %[[ret]]
  %0 = amdgpu.raw_buffer_load {boundsCheck = true} %buf[%idx] : memref<64xf8E4M3FNUZ>, i32 -> vector<4xf8E4M3FNUZ>
  func.return %0 : vector<4xf8E4M3FNUZ>
}

// Since the lowering logic is shared with loads, only bitcasts need to be rechecked
// CHECK-LABEL: func @gpu_gcn_raw_buffer_store_scalar_i32
func.func @gpu_gcn_raw_buffer_store_scalar_i32(%value: i32, %buf: memref<i32>) {
  // GFX9:  %[[flags:.*]] = llvm.mlir.constant(159744 : i32)
  // RDNA:  %[[flags:.*]] = llvm.mlir.constant(822243328 : i32)
  // CHECK: %[[resource:.*]] = rocdl.make.buffer.rsrc %{{.*}}, %{{.*}}, %{{.*}}, %[[flags]]
  // CHECK: rocdl.raw.ptr.buffer.store %{{.*}}, %[[resource]], %{{.*}}, %{{.*}}, %{{.*}} : i32
  amdgpu.raw_buffer_store {boundsCheck = true} %value -> %buf[] : i32 -> memref<i32>
  func.return
}

// CHECK-LABEL: func @gpu_gcn_raw_buffer_store_i32
func.func @gpu_gcn_raw_buffer_store_i32(%value: i32, %buf: memref<64xi32>, %idx: i32) {
  // CHECK: %[[numRecords:.*]] = llvm.mlir.constant(256 : i32)
  // GFX9:  %[[flags:.*]] = llvm.mlir.constant(159744 : i32)
  // RDNA:  %[[flags:.*]] = llvm.mlir.constant(822243328 : i32)
  // CHECK: %[[resource:.*]] = rocdl.make.buffer.rsrc %{{.*}}, %{{.*}}, %[[numRecords]], %[[flags]]
  // CHECK: rocdl.raw.ptr.buffer.store %{{.*}}, %[[resource]], %{{.*}}, %{{.*}}, %{{.*}} : i32
  amdgpu.raw_buffer_store {boundsCheck = true} %value -> %buf[%idx] : i32 -> memref<64xi32>, i32
  func.return
}

// CHECK-LABEL: func @gpu_gcn_raw_buffer_store_2xi8
func.func @gpu_gcn_raw_buffer_store_2xi8(%value: vector<2xi8>, %buf: memref<64xi8>, %idx: i32) {
  // CHECK: %[[cast:.*]] = llvm.bitcast %{{.*}} : vector<2xi8> to i16
  // CHECK: rocdl.raw.ptr.buffer.store %[[cast]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : i16
  amdgpu.raw_buffer_store {boundsCheck = true} %value -> %buf[%idx] : vector<2xi8> -> memref<64xi8>, i32
  func.return
}

// CHECK-LABEL: func @gpu_gcn_raw_buffer_store_16xi8
func.func @gpu_gcn_raw_buffer_store_16xi8(%value: vector<16xi8>, %buf: memref<64xi8>, %idx: i32) {
  // CHECK: %[[cast:.*]] = llvm.bitcast %{{.*}} : vector<16xi8> to vector<4xi32>
  // CHECK: rocdl.raw.ptr.buffer.store %[[cast]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : vector<4xi32>
  amdgpu.raw_buffer_store {boundsCheck = true} %value -> %buf[%idx] : vector<16xi8> -> memref<64xi8>, i32
  func.return
}

// And more so for atomic add
// CHECK-LABEL: func @gpu_gcn_raw_buffer_atomic_fadd_f32
func.func @gpu_gcn_raw_buffer_atomic_fadd_f32(%value: f32, %buf: memref<64xf32>, %idx: i32) {
  // CHECK: %[[numRecords:.*]] = llvm.mlir.constant(256 : i32)
  // GFX9:  %[[flags:.*]] = llvm.mlir.constant(159744 : i32)
  // RDNA:  %[[flags:.*]] = llvm.mlir.constant(822243328 : i32)
  // CHECK: %[[resource:.*]] = rocdl.make.buffer.rsrc %{{.*}}, %{{.*}}, %[[numRecords]], %[[flags]]
  // CHECK: rocdl.raw.ptr.buffer.atomic.fadd %{{.*}}, %[[resource]], %{{.*}}, %{{.*}}, %{{.*}} : f32
  amdgpu.raw_buffer_atomic_fadd {boundsCheck = true} %value -> %buf[%idx] : f32 -> memref<64xf32>, i32
  func.return
}

// CHECK-LABEL: func @gpu_gcn_raw_buffer_atomic_fadd_v2f16
func.func @gpu_gcn_raw_buffer_atomic_fadd_v2f16(%value: vector<2xf16>, %buf: memref<64xf16>, %idx: i32) {
  // CHECK: %[[numRecords:.*]] = llvm.mlir.constant(128 : i32)
  // GFX9:  %[[flags:.*]] = llvm.mlir.constant(159744 : i32)
  // RDNA:  %[[flags:.*]] = llvm.mlir.constant(822243328 : i32)
  // CHECK: %[[resource:.*]] = rocdl.make.buffer.rsrc %{{.*}}, %{{.*}}, %[[numRecords]], %[[flags]]
  // CHECK: rocdl.raw.ptr.buffer.atomic.fadd %{{.*}}, %[[resource]], %{{.*}}, %{{.*}}, %{{.*}} : vector<2xf16>
  amdgpu.raw_buffer_atomic_fadd {boundsCheck = true} %value -> %buf[%idx] : vector<2xf16> -> memref<64xf16>, i32
  func.return
}

// CHECK-LABEL: func @gpu_gcn_raw_buffer_atomic_fadd_v2bf16
func.func @gpu_gcn_raw_buffer_atomic_fadd_v2bf16(%value: vector<2xbf16>, %buf: memref<64xbf16>, %idx: i32) {
  // CHECK: %[[numRecords:.*]] = llvm.mlir.constant(128 : i32)
  // GFX9:  %[[flags:.*]] = llvm.mlir.constant(159744 : i32)
  // RDNA:  %[[flags:.*]] = llvm.mlir.constant(822243328 : i32)
  // CHECK: %[[resource:.*]] = rocdl.make.buffer.rsrc %{{.*}}, %{{.*}}, %[[numRecords]], %[[flags]]
  // CHECK: rocdl.raw.ptr.buffer.atomic.fadd %{{.*}}, %[[resource]], %{{.*}}, %{{.*}}, %{{.*}} : vector<2xbf16>
  amdgpu.raw_buffer_atomic_fadd {boundsCheck = true} %value -> %buf[%idx] : vector<2xbf16> -> memref<64xbf16>, i32
  func.return
}

// CHECK-LABEL: func @gpu_gcn_raw_buffer_atomic_fmax_f32
func.func @gpu_gcn_raw_buffer_atomic_fmax_f32(%value: f32, %buf: memref<64xf32>, %idx: i32) {
  // CHECK: %[[numRecords:.*]] = llvm.mlir.constant(256 : i32)
  // GFX9:  %[[flags:.*]] = llvm.mlir.constant(159744 : i32)
  // RDNA:  %[[flags:.*]] = llvm.mlir.constant(822243328 : i32)
  // CHECK: %[[resource:.*]] = rocdl.make.buffer.rsrc %{{.*}}, %{{.*}}, %[[numRecords]], %[[flags]]
  // CHECK: rocdl.raw.ptr.buffer.atomic.fmax %{{.*}}, %[[resource]], %{{.*}}, %{{.*}}, %{{.*}} : f32
  amdgpu.raw_buffer_atomic_fmax {boundsCheck = true} %value -> %buf[%idx] : f32 -> memref<64xf32>, i32
  func.return
}

// CHECK-LABEL: func @gpu_gcn_raw_buffer_atomic_smax_i32
func.func @gpu_gcn_raw_buffer_atomic_smax_i32(%value: i32, %buf: memref<64xi32>, %idx: i32) {
  // CHECK: %[[numRecords:.*]] = llvm.mlir.constant(256 : i32)
  // GFX9:  %[[flags:.*]] = llvm.mlir.constant(159744 : i32)
  // RDNA:  %[[flags:.*]] = llvm.mlir.constant(822243328 : i32)
  // CHECK: %[[resource:.*]] = rocdl.make.buffer.rsrc %{{.*}}, %{{.*}}, %[[numRecords]], %[[flags]]
  // CHECK: rocdl.raw.ptr.buffer.atomic.smax %{{.*}} %[[resource]], %{{.*}}, %{{.*}}, %{{.*}} : i32
  amdgpu.raw_buffer_atomic_smax {boundsCheck = true} %value -> %buf[%idx] : i32 -> memref<64xi32>, i32
  func.return
}

// CHECK-LABEL: func @gpu_gcn_raw_buffer_atomic_umin_i32
func.func @gpu_gcn_raw_buffer_atomic_umin_i32(%value: i32, %buf: memref<64xi32>, %idx: i32) {
  // CHECK: %[[numRecords:.*]] = llvm.mlir.constant(256 : i32)
  // GFX9:  %[[flags:.*]] = llvm.mlir.constant(159744 : i32)
  // RDNA:  %[[flags:.*]] = llvm.mlir.constant(822243328 : i32)
  // CHECK: %[[resource:.*]] = rocdl.make.buffer.rsrc %{{.*}}, %{{.*}}, %[[numRecords]], %[[flags]]
  // CHECK: rocdl.raw.ptr.buffer.atomic.umin %{{.*}} %[[resource]], %{{.*}}, %{{.*}}, %{{.*}} : i32
  amdgpu.raw_buffer_atomic_umin {boundsCheck = true} %value -> %buf[%idx] : i32 -> memref<64xi32>, i32
  func.return
}

// CHECK-LABEL: func @amdgpu_raw_buffer_atomic_cmpswap_f32
// CHECK-SAME: (%[[src:.*]]: f32, %[[cmp:.*]]: f32, {{.*}})
func.func @amdgpu_raw_buffer_atomic_cmpswap_f32(%src : f32, %cmp : f32, %buf : memref<64xf32>, %idx: i32) -> f32 {
  // CHECK: %[[srcCast:.*]] = llvm.bitcast %[[src]] : f32 to i32
  // CHECK: %[[cmpCast:.*]] = llvm.bitcast %[[cmp]] : f32 to i32
  // CHECK: %[[numRecords:.*]] = llvm.mlir.constant(256 : i32)
  // GFX9:  %[[flags:.*]] = llvm.mlir.constant(159744 : i32)
  // RDNA:  %[[flags:.*]] = llvm.mlir.constant(822243328 : i32)
  // CHECK: %[[resource:.*]] = rocdl.make.buffer.rsrc %{{.*}}, %{{.*}}, %[[numRecords]], %[[flags]]
  // CHECK: %[[dst:.*]] = rocdl.raw.ptr.buffer.atomic.cmpswap %[[srcCast]], %[[cmpCast]], %[[resource]], %{{.*}}, %{{.*}}, %{{.*}} : i32
  // CHECK: %[[dstCast:.*]] = llvm.bitcast %[[dst]] : i32 to f32
  // CHECK: return %[[dstCast]]
  %dst = amdgpu.raw_buffer_atomic_cmpswap {boundsCheck = true} %src, %cmp -> %buf[%idx] : f32 -> memref<64xf32>, i32
  func.return %dst : f32
}

// CHECK-LABEL: func @amdgpu_raw_buffer_atomic_cmpswap_i64
// CHECK-SAME: (%[[src:.*]]: i64, %[[cmp:.*]]: i64, {{.*}})
func.func @amdgpu_raw_buffer_atomic_cmpswap_i64(%src : i64, %cmp : i64, %buf : memref<64xi64>, %idx: i32) -> i64 {
  // CHECK: %[[numRecords:.*]] = llvm.mlir.constant(512 : i32)
  // GFX9:  %[[flags:.*]] = llvm.mlir.constant(159744 : i32)
  // RDNA:  %[[flags:.*]] = llvm.mlir.constant(822243328 : i32)
  // CHECK: %[[resource:.*]] = rocdl.make.buffer.rsrc %{{.*}}, %{{.*}}, %[[numRecords]], %[[flags]]
  // CHECK: %[[dst:.*]] = rocdl.raw.ptr.buffer.atomic.cmpswap %[[src]], %[[cmp]], %[[resource]], %{{.*}}, %{{.*}}, %{{.*}} : i64
  // CHECK: return %[[dst]]
  %dst = amdgpu.raw_buffer_atomic_cmpswap {boundsCheck = true} %src, %cmp -> %buf[%idx] : i64 -> memref<64xi64>, i32
  func.return %dst : i64
}

// CHECK-LABEL: func @amdgpu_raw_buffer_atomic_cmpswap_v2f16
// CHECK-SAME: (%[[src:.*]]: vector<2xf16>, %[[cmp:.*]]: vector<2xf16>, {{.*}})
func.func @amdgpu_raw_buffer_atomic_cmpswap_v2f16(%src : vector<2xf16>, %cmp : vector<2xf16>, %buf : memref<64xf16>, %idx: i32) -> vector<2xf16> {
  // CHECK-DAG: %[[srcBits:.+]] = llvm.bitcast %[[src]] : vector<2xf16> to i32
  // CHECK-DAG: %[[cmpBits:.+]] = llvm.bitcast %[[cmp]] : vector<2xf16> to i32
  // CHECK: %[[dstBits:.+]] = rocdl.raw.ptr.buffer.atomic.cmpswap %[[srcBits]], %[[cmpBits]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : i32
  // CHECK: %[[dst:.+]] = llvm.bitcast %[[dstBits]] : i32 to vector<2xf16>
  // CHECK: return %[[dst]]
  %dst = amdgpu.raw_buffer_atomic_cmpswap {boundsCheck = true} %src, %cmp -> %buf[%idx] : vector<2xf16> -> memref<64xf16>, i32
  func.return %dst : vector<2xf16>
}

// CHECK-LABEL: func @lds_barrier
func.func @lds_barrier() {
  // GFX908: llvm.inline_asm has_side_effects asm_dialect = att
  // GFX908-SAME: ";;;WARNING: BREAKS DEBUG WATCHES\0As_waitcnt lgkmcnt(0)\0As_barrier"
  // GFX90A: rocdl.waitcnt -7937
  // GFX90A-NEXT: rocdl.s.barrier
  // GFX10:  rocdl.waitcnt -16129
  // GFX10-NEXT: rocdl.s.barrier
  // GFX11:  llvm.inline_asm has_side_effects asm_dialect = att
  // GFX11-SAME: ";;;WARNING: BREAKS DEBUG WATCHES\0As_waitcnt lgkmcnt(0)\0As_barrier"
  // GFX12:  rocdl.s.wait.dscnt 0
  // GFX12-NEXT: rocdl.s.barrier.signal -1
  // GFX12-NEXT: rocdl.s.barrier.wait -1
  amdgpu.lds_barrier
  func.return
}

// CHECK-LABEL: func @sched_barrier
func.func @sched_barrier() {
  // CHECK: rocdl.sched.barrier 0
  amdgpu.sched_barrier allow = <none>
  // CHECK: rocdl.sched.barrier 1
  amdgpu.sched_barrier allow = <non_mem_non_sideffect>
  // CHECK: rocdl.sched.barrier 2
  amdgpu.sched_barrier allow = <valu>
  // CHECK: rocdl.sched.barrier 4
  amdgpu.sched_barrier allow = <salu>
  // CHECK: rocdl.sched.barrier 8
  amdgpu.sched_barrier allow = <mfma_wmma>
  // CHECK: rocdl.sched.barrier 16
  amdgpu.sched_barrier allow = <all_vmem>
  // CHECK: rocdl.sched.barrier 32
  amdgpu.sched_barrier allow = <vmem_read>
  // CHECK: rocdl.sched.barrier 64
  amdgpu.sched_barrier allow = <vmem_write>
  // CHECK: rocdl.sched.barrier 128
  amdgpu.sched_barrier allow = <all_ds>
  // CHECK: rocdl.sched.barrier 256
  amdgpu.sched_barrier allow = <ds_read>
  // CHECK: rocdl.sched.barrier 512
  amdgpu.sched_barrier allow = <ds_write>
  // CHECK: rocdl.sched.barrier 1024
  amdgpu.sched_barrier allow = <transcendental>
  // CHECK: rocdl.sched.barrier 18
  amdgpu.sched_barrier allow = <valu|all_vmem>
  func.return
}
