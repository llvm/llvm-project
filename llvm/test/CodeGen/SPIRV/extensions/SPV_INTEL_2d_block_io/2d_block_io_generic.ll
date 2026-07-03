; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown %s -o %t.spvt 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR
; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_2d_block_io %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s --spirv-ext=+SPV_INTEL_2d_block_io -o - -filetype=obj | spirv-val %}

; CHECK-ERROR: LLVM ERROR: OpSubgroup2DBlock[Load/LoadTranspose/LoadTransform/Prefetch/Store]INTEL
; CHECK-ERROR-SAME: instructions require the following SPIR-V extension: SPV_INTEL_2d_block_io

; CHECK: OpCapability Subgroup2DBlockIOINTEL
; CHECK: OpCapability Subgroup2DBlockTransformINTEL
; CHECK: OpCapability Subgroup2DBlockTransposeINTEL
; CHECK: OpExtension "SPV_INTEL_2d_block_io"

; CHECK-DAG: %[[Int8Ty:[0-9]+]] = OpTypeInt 8 0
; CHECK-DAG: %[[Int32Ty:[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: %[[Const42:[0-9]+]] = OpConstant %[[Int32Ty]] 42
; CHECK-DAG: %[[VoidTy:[0-9]+]] = OpTypeVoid
; CHECK-DAG: %[[GlbPtrTy:[0-9]+]] = OpTypePointer CrossWorkgroup %[[Int8Ty]]
; CHECK-DAG: %[[VectorTy:[0-9]+]] = OpTypeVector %[[Int32Ty]] 2
; CHECK-DAG: %[[PrvPtrTy:[0-9]+]] = OpTypePointer Function %[[Int8Ty]]
; CHECK: %[[BaseSrc:[0-9]+]] = OpFunctionParameter %[[GlbPtrTy]]
; CHECK: %[[BaseDst:[0-9]+]] = OpFunctionParameter %[[GlbPtrTy]]
; CHECK: %[[Width:[0-9]+]] = OpFunctionParameter %[[Int32Ty]]
; CHECK: %[[Height:[0-9]+]] = OpFunctionParameter %[[Int32Ty]]
; CHECK: %[[Pitch:[0-9]+]] = OpFunctionParameter %[[Int32Ty]]
; CHECK: %[[Coord:[0-9]+]] = OpFunctionParameter %[[VectorTy]]
; CHECK: %[[Dst:[0-9]+]] = OpFunctionParameter %[[PrvPtrTy]]
; CHECK: %[[Src:[0-9]+]] = OpFunctionParameter %[[PrvPtrTy]]
; CHECK: OpSubgroup2DBlockLoadINTEL %[[Const42]] %[[Const42]] %[[Const42]] %[[Const42]] %[[BaseSrc]] %[[Width]] %[[Height]] %[[Pitch]] %[[Coord]] %[[Dst]]
; CHECK: OpSubgroup2DBlockLoadTransformINTEL %[[Const42]] %[[Const42]] %[[Const42]] %[[Const42]] %[[BaseSrc]] %[[Width]] %[[Height]] %[[Pitch]] %[[Coord]] %[[Dst]]
; CHECK: OpSubgroup2DBlockLoadTransposeINTEL %[[Const42]] %[[Const42]] %[[Const42]] %[[Const42]] %[[BaseSrc]] %[[Width]] %[[Height]] %[[Pitch]] %[[Coord]] %[[Dst]]
; CHECK: OpSubgroup2DBlockPrefetchINTEL %[[Const42]] %[[Const42]] %[[Const42]] %[[Const42]] %[[BaseSrc]] %[[Width]] %[[Height]] %[[Pitch]] %[[Coord]]
; CHECK: OpSubgroup2DBlockStoreINTEL %[[Const42]] %[[Const42]] %[[Const42]] %[[Const42]] %[[Src]] %[[BaseDst]] %[[Width]] %[[Height]] %[[Pitch]] %[[Coord]]

define spir_func void @foo(ptr addrspace(1) %base_address, ptr addrspace(1) %dst_base_pointer, i32 %width, i32 %height, i32 %pitch, <2 x i32> %coord, ptr %dst_pointer, ptr %src_pointer) {
entry:
  call spir_func void @_Z32__spirv_Subgroup2DBlockLoadINTELiiiiPU3AS1KviiiDv2_iPv(i32 42, i32 42, i32 42, i32 42, ptr addrspace(1) %base_address, i32 %width, i32 %height, i32 %pitch, <2 x i32> %coord, ptr %dst_pointer)
  call spir_func void @_Z41__spirv_Subgroup2DBlockLoadTransformINTELiiiiPU3AS1KviiiDv2_iPv(i32 42, i32 42, i32 42, i32 42, ptr addrspace(1) %base_address, i32 %width, i32 %height, i32 %pitch, <2 x i32> %coord, ptr %dst_pointer)
  call spir_func void @_Z41__spirv_Subgroup2DBlockLoadTransposeINTELiiiiPU3AS1KviiiDv2_iPv(i32 42, i32 42, i32 42, i32 42, ptr addrspace(1) %base_address, i32 %width, i32 %height, i32 %pitch, <2 x i32> %coord, ptr %dst_pointer)
  call spir_func void @_Z36__spirv_Subgroup2DBlockPrefetchINTELiiiiPU3AS1KviiiDv2_i(i32 42, i32 42, i32 42, i32 42, ptr addrspace(1) %base_address, i32 %width, i32 %height, i32 %pitch, <2 x i32> %coord)
  call spir_func void @_Z33__spirv_Subgroup2DBlockStoreINTELiiiiPKvPU3AS1viiiDv2_i(i32 42, i32 42, i32 42, i32 42, ptr %src_pointer, ptr addrspace(1) %dst_base_pointer, i32 %width, i32 %height, i32 %pitch, <2 x i32> %coord)
  ret void
}

declare spir_func void @_Z32__spirv_Subgroup2DBlockLoadINTELiiiiPU3AS1KviiiDv2_iPv(i32, i32, i32, i32, ptr addrspace(1), i32, i32, i32, <2 x i32>, ptr)
declare spir_func void @_Z41__spirv_Subgroup2DBlockLoadTransformINTELiiiiPU3AS1KviiiDv2_iPv(i32, i32, i32, i32, ptr addrspace(1), i32, i32, i32, <2 x i32>, ptr)
declare spir_func void @_Z41__spirv_Subgroup2DBlockLoadTransposeINTELiiiiPU3AS1KviiiDv2_iPv(i32, i32, i32, i32, ptr addrspace(1), i32, i32, i32, <2 x i32>, ptr)
declare spir_func void @_Z36__spirv_Subgroup2DBlockPrefetchINTELiiiiPU3AS1KviiiDv2_i(i32, i32, i32, i32, ptr addrspace(1), i32, i32, i32, <2 x i32>)
declare spir_func void @_Z33__spirv_Subgroup2DBlockStoreINTELiiiiPKvPU3AS1viiiDv2_i(i32, i32, i32, i32, ptr, ptr addrspace(1), i32, i32, i32, <2 x i32>)
