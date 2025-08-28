// RUN: not mlir-opt %s --split-input-file -convert-amdgpu-to-rocdl=chipset=gfx942 2>&1 | FileCheck %s --check-prefix=GFX942
// RUN: mlir-opt %s --split-input-file -convert-amdgpu-to-rocdl=chipset=gfx950 | FileCheck %s --check-prefix=GFX950

#gpu_global_addrspace = 1
#gpu_lds_addrspace = 3
#amdgpu_fat_buffer_addrspace = 7

// GFX950-LABEL: func @fat_buffer_load_to_rocdl_f96
// GFX950-SAME: (%[[ARG0:.*]]: memref<128x72xf32, 7>)
func.func @fat_buffer_load_to_rocdl_f96(%global : memref<128x72xf32, #amdgpu_fat_buffer_addrspace>) {
  %c0 = arith.constant 0 : index
  %c12 = arith.constant 12 : index
  %c32 = arith.constant 32 : index
  %alloc = memref.alloc() : memref<64x64xf32, #gpu_lds_addrspace>
  // GFX950: %[[BUFFER_DESC:.*]] = builtin.unrealized_conversion_cast %[[ARG0]]

  // GFX950: %[[C0:.*]] = arith.constant 0 : index
  // GFX950: %[[IC0:.*]] = builtin.unrealized_conversion_cast %c0 : index to i64
  // GFX950: %[[C12:.*]] = arith.constant 12 : index
  // GFX950: %[[IC12:.*]] = builtin.unrealized_conversion_cast %[[C12]]
  // GFX950: %[[C32:.*]] = arith.constant 32 : index
  // GFX950: %[[IC32:.*]] = builtin.unrealized_conversion_cast %[[C32]]

  // GFX950: %[[ALLOC:.*]] = memref.alloc()
  // GFX950: %[[LDS_DESC:.*]] = builtin.unrealized_conversion_cast
  // GFX950: %[[GLOBAL_BASE:.*]] = llvm.extractvalue %[[BUFFER_DESC]][1]

  // GFX950: %[[C72:.*]] = llvm.mlir.constant(72 : index) : i64
  // GFX950: %[[MUL:.*]] = llvm.mul %[[IC12]], %[[C72]] : i64
  // GFX950: %[[SRC_OFFSET:.*]] = llvm.add %[[MUL]], %[[IC0]] : i64

  // GFX950: %[[GLOBAL_PTR:.*]] = llvm.getelementptr %[[GLOBAL_BASE]][%[[SRC_OFFSET]]]
  // GFX950: %[[LDS_BASE:.*]] = llvm.extractvalue %[[LDS_DESC]][1]

  // GFX950: %[[C64:.*]] = llvm.mlir.constant(64 : index) : i64
  // GFX950: %[[MUL_2:.*]] = llvm.mul %[[IC32]], %[[C64]] : i64
  // GFX950: %[[DST_OFFSET:.*]] = llvm.add %[[MUL_2]], %[[IC0]] : i64

  // GFX950: %[[LDS_PTR:.*]] = llvm.getelementptr %[[LDS_BASE]][%[[DST_OFFSET]]]
  // GFX950: rocdl.load.to.lds %[[GLOBAL_PTR]], %[[LDS_PTR]], 12
  // GFX942: error: 'amdgpu.gather_to_lds' op Gather to LDS instructions with 12-byte and 16-byte load widths are only supported on gfx950
  amdgpu.gather_to_lds %global[%c12, %c0], %alloc[%c32, %c0]
    : vector<16xf6E3M2FN>, memref<128x72xf32, #amdgpu_fat_buffer_addrspace>, memref<64x64xf32, #gpu_lds_addrspace>
  func.return
}

// -----

#gpu_global_addrspace = 1
#gpu_lds_addrspace = 3
#amdgpu_fat_buffer_addrspace = 7

// GFX950-LABEL: func @fat_buffer_load_to_rocdl_f128
// GFX950-SAME: (%[[ARG0:.*]]: memref<128x72xf32, 7>)
func.func @fat_buffer_load_to_rocdl_f128(%global : memref<128x72xf32, #amdgpu_fat_buffer_addrspace>) {
  %c0 = arith.constant 0 : index
  %c12 = arith.constant 12 : index
  %c32 = arith.constant 32 : index
  %alloc = memref.alloc() : memref<64x64xf32, #gpu_lds_addrspace>
  // GFX950: %[[BUFFER_DESC:.*]] = builtin.unrealized_conversion_cast %[[ARG0]]

  // GFX950: %[[C0:.*]] = arith.constant 0 : index
  // GFX950: %[[IC0:.*]] = builtin.unrealized_conversion_cast %c0 : index to i64
  // GFX950: %[[C12:.*]] = arith.constant 12 : index
  // GFX950: %[[IC12:.*]] = builtin.unrealized_conversion_cast %[[C12]]
  // GFX950: %[[C32:.*]] = arith.constant 32 : index
  // GFX950: %[[IC32:.*]] = builtin.unrealized_conversion_cast %[[C32]]

  // GFX950: %[[ALLOC:.*]] = memref.alloc()
  // GFX950: %[[LDS_DESC:.*]] = builtin.unrealized_conversion_cast
  // GFX950: %[[GLOBAL_BASE:.*]] = llvm.extractvalue %[[BUFFER_DESC]][1]

  // GFX950: %[[C72:.*]] = llvm.mlir.constant(72 : index) : i64
  // GFX950: %[[MUL:.*]] = llvm.mul %[[IC12]], %[[C72]] : i64
  // GFX950: %[[SRC_OFFSET:.*]] = llvm.add %[[MUL]], %[[IC0]] : i64

  // GFX950: %[[GLOBAL_PTR:.*]] = llvm.getelementptr %[[GLOBAL_BASE]][%[[SRC_OFFSET]]]
  // GFX950: %[[LDS_BASE:.*]] = llvm.extractvalue %[[LDS_DESC]][1]

  // GFX950: %[[C64:.*]] = llvm.mlir.constant(64 : index) : i64
  // GFX950: %[[MUL_2:.*]] = llvm.mul %[[IC32]], %[[C64]] : i64
  // GFX950: %[[DST_OFFSET:.*]] = llvm.add %[[MUL_2]], %[[IC0]] : i64

  // GFX950: %[[LDS_PTR:.*]] = llvm.getelementptr %[[LDS_BASE]][%[[DST_OFFSET]]]
  // GFX950: rocdl.load.to.lds %[[GLOBAL_PTR]], %[[LDS_PTR]], 16
  // GFX942: error: 'amdgpu.gather_to_lds' op Gather to LDS instructions with 12-byte and 16-byte load widths are only supported on gfx950
  amdgpu.gather_to_lds %global[%c12, %c0], %alloc[%c32, %c0]
    : f128, memref<128x72xf32, #amdgpu_fat_buffer_addrspace>, memref<64x64xf32, #gpu_lds_addrspace>
  func.return
}
