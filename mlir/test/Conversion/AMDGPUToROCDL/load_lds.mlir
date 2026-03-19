// RUN: mlir-opt %s --convert-amdgpu-to-rocdl=chipset=gfx942 | FileCheck %s
// RUN: mlir-opt %s --convert-amdgpu-to-rocdl=chipset=gfx950 | FileCheck %s

// CHECK-LABEL: func @global_load_to_rocdl_f32
// CHECK-SAME: (%[[ARG0:.*]]: memref<128x72xf32, #gpu.address_space<global>>)
func.func @global_load_to_rocdl_f32(%global : memref<128x72xf32, #gpu.address_space<global>>) {
  %c0 = arith.constant 0 : index
  %c12 = arith.constant 12 : index
  %c32 = arith.constant 32 : index
  %alloc = memref.alloc() : memref<64x64xf32, #gpu.address_space<workgroup>>
  // CHECK: %[[GLOBAL_DESC:.*]] = builtin.unrealized_conversion_cast %[[ARG0]]

  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[IC0:.*]] = builtin.unrealized_conversion_cast %c0 : index to i64
  // CHECK: %[[C12:.*]] = arith.constant 12 : index
  // CHECK: %[[IC12:.*]] = builtin.unrealized_conversion_cast %[[C12]]
  // CHECK: %[[C32:.*]] = arith.constant 32 : index
  // CHECK: %[[IC32:.*]] = builtin.unrealized_conversion_cast %[[C32]]

  // CHECK: %[[ALLOC:.*]] = memref.alloc()
  // CHECK: %[[LDS_DESC:.*]] = builtin.unrealized_conversion_cast
  // CHECK: %[[GLOBAL_BASE:.*]] = llvm.extractvalue %[[GLOBAL_DESC]][1]

  // CHECK: %[[C72:.*]] = llvm.mlir.constant(72 : index) : i64
  // CHECK: %[[MUL:.*]] = llvm.mul %[[IC12]], %[[C72]] : i64
  // CHECK: %[[SRC_OFFSET:.*]] = llvm.add %[[MUL]], %[[IC0]] : i64

  // CHECK: %[[GLOBAL_PTR:.*]] = llvm.getelementptr %[[GLOBAL_BASE]][%[[SRC_OFFSET]]]
  // CHECK: %[[LDS_BASE:.*]] = llvm.extractvalue %[[LDS_DESC]][1]

  // CHECK: %[[C64:.*]] = llvm.mlir.constant(64 : index) : i64
  // CHECK: %[[MUL_2:.*]] = llvm.mul %[[IC32]], %[[C64]] : i64
  // CHECK: %[[DST_OFFSET:.*]] = llvm.add %[[MUL_2]], %[[IC0]] : i64

  // CHECK: %[[LDS_PTR:.*]] = llvm.getelementptr %[[LDS_BASE]][%[[DST_OFFSET]]]
  // CHECK: rocdl.load.to.lds %[[GLOBAL_PTR]], %[[LDS_PTR]], 4
  amdgpu.gather_to_lds %global[%c12, %c0], %alloc[%c32, %c0]
    : f32, memref<128x72xf32, #gpu.address_space<global>>, memref<64x64xf32, #gpu.address_space<workgroup>>
  func.return
}

// CHECK-LABEL: func @global_load_to_rocdl_wg_mem
// CHECK-SAME: (%[[ARG0:.*]]: memref<128x72xf32>)
func.func @global_load_to_rocdl_wg_mem(%global : memref<128x72xf32>) {
  %c0 = arith.constant 0 : index
  %c12 = arith.constant 12 : index
  %c32 = arith.constant 32 : index
  %alloc = memref.alloc() : memref<64x64xf32, #gpu.address_space<workgroup>>
  // CHECK: %[[GLOBAL_DESC:.*]] = builtin.unrealized_conversion_cast %[[ARG0]]

  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[IC0:.*]] = builtin.unrealized_conversion_cast %c0 : index to i64
  // CHECK: %[[C12:.*]] = arith.constant 12 : index
  // CHECK: %[[IC12:.*]] = builtin.unrealized_conversion_cast %[[C12]]
  // CHECK: %[[C32:.*]] = arith.constant 32 : index
  // CHECK: %[[IC32:.*]] = builtin.unrealized_conversion_cast %[[C32]]

  // CHECK: %[[ALLOC:.*]] = memref.alloc()
  // CHECK: %[[LDS_DESC:.*]] = builtin.unrealized_conversion_cast
  // CHECK: %[[GLOBAL_BASE:.*]] = llvm.extractvalue %[[GLOBAL_DESC]][1]

  // CHECK: %[[C72:.*]] = llvm.mlir.constant(72 : index) : i64
  // CHECK: %[[MUL:.*]] = llvm.mul %[[IC12]], %[[C72]] : i64
  // CHECK: %[[SRC_OFFSET:.*]] = llvm.add %[[MUL]], %[[IC0]] : i64

  // CHECK: %[[GLOBAL_PTR:.*]] = llvm.getelementptr %[[GLOBAL_BASE]][%[[SRC_OFFSET]]]
  // CHECK: %[[LDS_BASE:.*]] = llvm.extractvalue %[[LDS_DESC]][1]

  // CHECK: %[[C64:.*]] = llvm.mlir.constant(64 : index) : i64
  // CHECK: %[[MUL_2:.*]] = llvm.mul %[[IC32]], %[[C64]] : i64
  // CHECK: %[[DST_OFFSET:.*]] = llvm.add %[[MUL_2]], %[[IC0]] : i64

  // CHECK: %[[LDS_PTR:.*]] = llvm.getelementptr %[[LDS_BASE]][%[[DST_OFFSET]]]
  // CHECK: rocdl.load.to.lds %[[GLOBAL_PTR]], %[[LDS_PTR]], 4
  amdgpu.gather_to_lds %global[%c12, %c0], %alloc[%c32, %c0]
    : f32, memref<128x72xf32>, memref<64x64xf32, #gpu.address_space<workgroup>>
  func.return
}

// CHECK-LABEL: func @global_load_to_rocdl_0d
// CHECK-SAME: (%[[ARG0:.*]]: memref<f32>)
func.func @global_load_to_rocdl_0d(%global : memref<f32>) {
  %alloc = memref.alloc() : memref<f32, #gpu.address_space<workgroup>>
  // CHECK: %[[GLOBAL_DESC:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : memref<f32> to !llvm.struct<(ptr, ptr, i64)>

  // CHECK: %[[ALLOC:.*]] = memref.alloc()
  // CHECK: %[[LDS_DESC:.*]] = builtin.unrealized_conversion_cast %[[ALLOC]] : memref<f32, #gpu.address_space<workgroup>> to !llvm.struct<(ptr<3>, ptr<3>, i64)>

  // CHECK: %[[GLOBAL_BASE:.*]] = llvm.extractvalue %[[GLOBAL_DESC]][1] : !llvm.struct<(ptr, ptr, i64)>
  // CHECK: %[[LDS_BASE:.*]] = llvm.extractvalue %[[LDS_DESC]][1] : !llvm.struct<(ptr<3>, ptr<3>, i64)>

  // CHECK: rocdl.load.to.lds %[[GLOBAL_BASE]], %[[LDS_BASE]], 4
  amdgpu.gather_to_lds %global[], %alloc[]
    : f32, memref<f32>, memref<f32, #gpu.address_space<workgroup>>
  func.return
}

// CHECK-LABEL: func @global_load_to_rocdl_i8
// CHECK-SAME: (%[[ARG0:.*]]: memref<128x72xi8, #gpu.address_space<global>>)
func.func @global_load_to_rocdl_i8(%global : memref<128x72xi8, #gpu.address_space<global>>) {
  // CHECK: %[[GLOBAL_DESC:.*]] = builtin.unrealized_conversion_cast %[[ARG0]]

  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[IC0:.*]] = builtin.unrealized_conversion_cast %c0 : index to i64
  // CHECK: %[[C12:.*]] = arith.constant 12 : index
  // CHECK: %[[IC12:.*]] = builtin.unrealized_conversion_cast %[[C12]]
  // CHECK: %[[C32:.*]] = arith.constant 32 : index
  // CHECK: %[[IC32:.*]] = builtin.unrealized_conversion_cast %[[C32]]

  // CHECK: %[[ALLOC:.*]] = memref.alloc()
  // CHECK: %[[LDS_DESC:.*]] = builtin.unrealized_conversion_cast %[[ALLOC]]
  // CHECK: %[[GLOBAL_BASE:.*]] = llvm.extractvalue %[[GLOBAL_DESC]][1]

  // CHECK: %[[C72:.*]] = llvm.mlir.constant(72 : index) : i64
  // CHECK: %[[MUL:.*]] = llvm.mul %[[IC12]], %[[C72]] : i64
  // CHECK: %[[SRC_OFFSET:.*]] = llvm.add %[[MUL]], %[[IC0]] : i64

  // CHECK: %[[GLOBAL_PTR:.*]] = llvm.getelementptr %[[GLOBAL_BASE]][%[[SRC_OFFSET]]]
  // CHECK: %[[LDS_BASE:.*]] = llvm.extractvalue %[[LDS_DESC]][1]

  // CHECK: %[[C64:.*]] = llvm.mlir.constant(64 : index) : i64
  // CHECK: %[[MUL_2:.*]] = llvm.mul %[[IC32]], %[[C64]] : i64
  // CHECK: %[[DST_OFFSET:.*]] = llvm.add %[[MUL_2]], %[[IC0]] : i64

  // CHECK: %[[LDS_PTR:.*]] = llvm.getelementptr %[[LDS_BASE]][%[[DST_OFFSET]]]
  // CHECK: rocdl.load.to.lds %[[GLOBAL_PTR]], %[[LDS_PTR]], 1, 0, 0 : <1>
  %c0 = arith.constant 0 : index
  %c12 = arith.constant 12 : index
  %c32 = arith.constant 32 : index
  %alloc = memref.alloc() : memref<64x64xi8, #gpu.address_space<workgroup>>
  amdgpu.gather_to_lds %global[%c12, %c0], %alloc[%c32, %c0]
    : i8, memref<128x72xi8, #gpu.address_space<global>>, memref<64x64xi8, #gpu.address_space<workgroup>>
  func.return
}

// CHECK-LABEL: func @global_load_to_rocdl_vec
// CHECK-SAME: (%[[ARG0:.*]]: memref<128x72xi16, #gpu.address_space<global>>)
func.func @global_load_to_rocdl_vec(%global : memref<128x72xi16, #gpu.address_space<global>>) {
  // CHECK: %[[GLOBAL_DESC:.*]] = builtin.unrealized_conversion_cast %[[ARG0]]

  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[IC0:.*]] = builtin.unrealized_conversion_cast %c0 : index to i64
  // CHECK: %[[C12:.*]] = arith.constant 12 : index
  // CHECK: %[[IC12:.*]] = builtin.unrealized_conversion_cast %[[C12]]
  // CHECK: %[[C32:.*]] = arith.constant 32 : index
  // CHECK: %[[IC32:.*]] = builtin.unrealized_conversion_cast %[[C32]]

  // CHECK: %[[ALLOC:.*]] = memref.alloc()
  // CHECK: %[[LDS_DESC:.*]] = builtin.unrealized_conversion_cast %[[ALLOC]]
  // CHECK: %[[GLOBAL_BASE:.*]] = llvm.extractvalue %[[GLOBAL_DESC]][1]

  // CHECK: %[[C72:.*]] = llvm.mlir.constant(72 : index) : i64
  // CHECK: %[[MUL:.*]] = llvm.mul %[[IC12]], %[[C72]] : i64
  // CHECK: %[[SRC_OFFSET:.*]] = llvm.add %[[MUL]], %[[IC0]] : i64

  // CHECK: %[[GLOBAL_PTR:.*]] = llvm.getelementptr %[[GLOBAL_BASE]][%[[SRC_OFFSET]]]
  // CHECK: %[[LDS_BASE:.*]] = llvm.extractvalue %[[LDS_DESC]][1]

  // CHECK: %[[C128:.*]] = llvm.mlir.constant(128 : index) : i64
  // CHECK: %[[MUL_2:.*]] = llvm.mul %[[IC32]], %[[C128]] : i64
  // CHECK: %[[DST_OFFSET:.*]] = llvm.add %[[MUL_2]], %[[IC0]] : i64

  // CHECK: %[[LDS_PTR:.*]] = llvm.getelementptr %[[LDS_BASE]][%[[DST_OFFSET]]]
  // CHECK: rocdl.load.to.lds %[[GLOBAL_PTR]], %[[LDS_PTR]], 4
  %c0 = arith.constant 0 : index
  %c12 = arith.constant 12 : index
  %c32 = arith.constant 32 : index
  %alloc = memref.alloc() : memref<64x128xi16, #gpu.address_space<workgroup>>
  amdgpu.gather_to_lds %global[%c12, %c0], %alloc[%c32, %c0]
    : vector<2 x i16>, memref<128x72xi16, #gpu.address_space<global>>, memref<64x128xi16, #gpu.address_space<workgroup>>
  func.return
}

// CHECK-LABEL: func @global_load_to_rocdl_dynamic_indices
// CHECK-SAME: (%[[ARG0:.*]]: memref<512xi32, #gpu.address_space<global>>, %[[SRC_IDX:.*]]: index, %[[DST_IDX:.*]]: index)
func.func @global_load_to_rocdl_dynamic_indices(%global : memref<512xi32, #gpu.address_space<global>>, %src_idx : index, %dst_idx : index) {
  // CHECK: %[[DSTIDX_CAST:.*]] = builtin.unrealized_conversion_cast %[[DST_IDX]]
  // CHECK: %[[SRCIDX_CAST:.*]] = builtin.unrealized_conversion_cast %[[SRC_IDX]]
  // CHECK: %[[GLOBAL_DESC:.*]] = builtin.unrealized_conversion_cast %[[ARG0]]
  // CHECK: %[[ALLOC:.*]] = memref.alloc()
  // CHECK: %[[LDS_DESC:.*]] = builtin.unrealized_conversion_cast %[[ALLOC]]
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[C0_I64:.*]] = builtin.unrealized_conversion_cast %[[C0]] : index to i64
  // CHECK: %[[GLOBAL_BASE:.*]] = llvm.extractvalue %[[GLOBAL_DESC]][1]
  // CHECK: %[[GLOBAL_PTR:.*]] = llvm.getelementptr %[[GLOBAL_BASE]][%[[SRCIDX_CAST]]]
  // CHECK: %[[LDS_BASE:.*]] = llvm.extractvalue %[[LDS_DESC]][1]
  // CHECK: %[[C64:.*]] = llvm.mlir.constant(64 : index) : i64
  // CHECK: %[[DSTIDX:.*]] = llvm.mul %[[DSTIDX_CAST]], %[[C64]] : i64
  // CHECK: %[[DSTIDX1:.*]] = llvm.add %[[DSTIDX]], %[[C0_I64]] : i64
  // CHECK: %[[LDS_PTR:.*]] = llvm.getelementptr %[[LDS_BASE]][%[[DSTIDX1]]]
  // CHECK: rocdl.load.to.lds %[[GLOBAL_PTR]], %[[LDS_PTR]], 4
  %alloc = memref.alloc() : memref<4x64xi32, #gpu.address_space<workgroup>>
  %c0 = arith.constant 0 : index
  amdgpu.gather_to_lds %global[%src_idx], %alloc[%dst_idx, %c0]
    : i32, memref<512xi32, #gpu.address_space<global>>, memref<4x64xi32, #gpu.address_space<workgroup>>
  func.return
}

// CHECK-LABEL: func @fat_buffer_load_to_rocdl_f32
// CHECK-SAME: (%[[ARG0:.*]]: memref<128x72xf32, #amdgpu.address_space<fat_raw_buffer>>)
func.func @fat_buffer_load_to_rocdl_f32(%global : memref<128x72xf32, #amdgpu.address_space<fat_raw_buffer>>) {
  %c0 = arith.constant 0 : index
  %c12 = arith.constant 12 : index
  %c32 = arith.constant 32 : index
  %alloc = memref.alloc() : memref<64x64xf32, #gpu.address_space<workgroup>>
  // CHECK: %[[BUFFER_DESC:.*]] = builtin.unrealized_conversion_cast %[[ARG0]]

  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[IC0:.*]] = builtin.unrealized_conversion_cast %[[C0]] : index to i64
  // CHECK: %[[C12:.*]] = arith.constant 12 : index
  // CHECK: %[[IC12:.*]] = builtin.unrealized_conversion_cast %[[C12]]
  // CHECK: %[[C32:.*]] = arith.constant 32 : index
  // CHECK: %[[IC32:.*]] = builtin.unrealized_conversion_cast %[[C32]]

  // CHECK: %[[ALLOC:.*]] = memref.alloc()
  // CHECK: %[[LDS_DESC:.*]] = builtin.unrealized_conversion_cast
  // CHECK: %[[GLOBAL_BASE:.*]] = llvm.extractvalue %[[BUFFER_DESC]][1]

  // CHECK: %[[C72:.*]] = llvm.mlir.constant(72 : index) : i64
  // CHECK: %[[MUL:.*]] = llvm.mul %[[IC12]], %[[C72]] : i64
  // CHECK: %[[SRC_OFFSET:.*]] = llvm.add %[[MUL]], %[[IC0]] : i64

  // CHECK: %[[GLOBAL_PTR:.*]] = llvm.getelementptr %[[GLOBAL_BASE]][%[[SRC_OFFSET]]]
  // CHECK: %[[LDS_BASE:.*]] = llvm.extractvalue %[[LDS_DESC]][1]

  // CHECK: %[[C64:.*]] = llvm.mlir.constant(64 : index) : i64
  // CHECK: %[[MUL_2:.*]] = llvm.mul %[[IC32]], %[[C64]] : i64
  // CHECK: %[[DST_OFFSET:.*]] = llvm.add %[[MUL_2]], %[[IC0]] : i64

  // CHECK: %[[LDS_PTR:.*]] = llvm.getelementptr %[[LDS_BASE]][%[[DST_OFFSET]]]
  // CHECK: rocdl.load.to.lds %[[GLOBAL_PTR]], %[[LDS_PTR]], 4
  amdgpu.gather_to_lds %global[%c12, %c0], %alloc[%c32, %c0]
    : f32, memref<128x72xf32, #amdgpu.address_space<fat_raw_buffer>>, memref<64x64xf32, #gpu.address_space<workgroup>>
  func.return
}

// CHECK-LABEL: func @global_load_to_rocdl_async_f32
// CHECK-SAME: (%[[ARG0:.*]]: memref<128x72xf32, #gpu.address_space<global>>)
func.func @global_load_to_rocdl_async_f32(%global : memref<128x72xf32, #gpu.address_space<global>>) {
  %c0 = arith.constant 0 : index
  %c12 = arith.constant 12 : index
  %c32 = arith.constant 32 : index
  %alloc = memref.alloc() : memref<64x64xf32, #gpu.address_space<workgroup>>
  // CHECK: %[[GLOBAL_DESC:.*]] = builtin.unrealized_conversion_cast %[[ARG0]]

  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[IC0:.*]] = builtin.unrealized_conversion_cast %c0 : index to i64
  // CHECK-DAG: %[[C12:.*]] = arith.constant 12 : index
  // CHECK-DAG: %[[IC12:.*]] = builtin.unrealized_conversion_cast %[[C12]]
  // CHECK-DAG: %[[C32:.*]] = arith.constant 32 : index
  // CHECK-DAG: %[[IC32:.*]] = builtin.unrealized_conversion_cast %[[C32]]

  // CHECK: %[[ALLOC:.*]] = memref.alloc()
  // CHECK: %[[LDS_DESC:.*]] = builtin.unrealized_conversion_cast
  // CHECK: %[[GLOBAL_BASE:.*]] = llvm.extractvalue %[[GLOBAL_DESC]][1]

  // CHECK: %[[C72:.*]] = llvm.mlir.constant(72 : index) : i64
  // CHECK: %[[MUL:.*]] = llvm.mul %[[IC12]], %[[C72]] : i64
  // CHECK: %[[SRC_OFFSET:.*]] = llvm.add %[[MUL]], %[[IC0]] : i64

  // CHECK: %[[GLOBAL_PTR:.*]] = llvm.getelementptr %[[GLOBAL_BASE]][%[[SRC_OFFSET]]]
  // CHECK: %[[LDS_BASE:.*]] = llvm.extractvalue %[[LDS_DESC]][1]

  // CHECK: %[[C64:.*]] = llvm.mlir.constant(64 : index) : i64
  // CHECK: %[[MUL_2:.*]] = llvm.mul %[[IC32]], %[[C64]] : i64
  // CHECK: %[[DST_OFFSET:.*]] = llvm.add %[[MUL_2]], %[[IC0]] : i64

  // CHECK: %[[LDS_PTR:.*]] = llvm.getelementptr %[[LDS_BASE]][%[[DST_OFFSET]]]
  // CHECK: rocdl.load.async.to.lds %[[GLOBAL_PTR]], %[[LDS_PTR]], 4
  amdgpu.gather_to_lds async %global[%c12, %c0], %alloc[%c32, %c0]
    : f32, memref<128x72xf32, #gpu.address_space<global>>, memref<64x64xf32, #gpu.address_space<workgroup>>
  func.return
}

// CHECK-LABEL: func @global_load_to_rocdl_async_f32_fat_raw_buffer
// CHECK-SAME: (%[[ARG0:.*]]: memref<128x72xf32, #amdgpu.address_space<fat_raw_buffer>>)
func.func @global_load_to_rocdl_async_f32_fat_raw_buffer(%global : memref<128x72xf32, #amdgpu.address_space<fat_raw_buffer>>) {
  %c0 = arith.constant 0 : index
  %c12 = arith.constant 12 : index
  %c32 = arith.constant 32 : index
  %alloc = memref.alloc() : memref<64x64xf32, #gpu.address_space<workgroup>>
  // CHECK: %[[GLOBAL_DESC:.*]] = builtin.unrealized_conversion_cast %[[ARG0]]

  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[IC0:.*]] = builtin.unrealized_conversion_cast %c0 : index to i64
  // CHECK-DAG: %[[C12:.*]] = arith.constant 12 : index
  // CHECK-DAG: %[[IC12:.*]] = builtin.unrealized_conversion_cast %[[C12]]
  // CHECK-DAG: %[[C32:.*]] = arith.constant 32 : index
  // CHECK-DAG: %[[IC32:.*]] = builtin.unrealized_conversion_cast %[[C32]]

  // CHECK: %[[ALLOC:.*]] = memref.alloc()
  // CHECK: %[[LDS_DESC:.*]] = builtin.unrealized_conversion_cast
  // CHECK: %[[GLOBAL_BASE:.*]] = llvm.extractvalue %[[GLOBAL_DESC]][1]

  // CHECK: %[[C72:.*]] = llvm.mlir.constant(72 : index) : i64
  // CHECK: %[[MUL:.*]] = llvm.mul %[[IC12]], %[[C72]] : i64
  // CHECK: %[[SRC_OFFSET:.*]] = llvm.add %[[MUL]], %[[IC0]] : i64

  // CHECK: %[[GLOBAL_PTR:.*]] = llvm.getelementptr %[[GLOBAL_BASE]][%[[SRC_OFFSET]]]
  // CHECK: %[[LDS_BASE:.*]] = llvm.extractvalue %[[LDS_DESC]][1]

  // CHECK: %[[C64:.*]] = llvm.mlir.constant(64 : index) : i64
  // CHECK: %[[MUL_2:.*]] = llvm.mul %[[IC32]], %[[C64]] : i64
  // CHECK: %[[DST_OFFSET:.*]] = llvm.add %[[MUL_2]], %[[IC0]] : i64

  // CHECK: %[[LDS_PTR:.*]] = llvm.getelementptr %[[LDS_BASE]][%[[DST_OFFSET]]]
  // CHECK: rocdl.load.async.to.lds %[[GLOBAL_PTR]], %[[LDS_PTR]], 4
  amdgpu.gather_to_lds async %global[%c12, %c0], %alloc[%c32, %c0]
    : f32, memref<128x72xf32, #amdgpu.address_space<fat_raw_buffer>>, memref<64x64xf32, #gpu.address_space<workgroup>>
  func.return
}
