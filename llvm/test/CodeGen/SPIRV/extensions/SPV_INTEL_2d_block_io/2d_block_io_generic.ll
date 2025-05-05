 ;Generated with:
; source.cl:
; void __spirv_Subgroup2DBlockLoadINTEL(         int element_size, int block_width, int block_height, int block_count, const __global void* src_base_pointer, int memory_width,                int memory_height, int memory_pitch,  int2 coordinate,  private void* dst_pointer);
; void __spirv_Subgroup2DBlockLoadTransposeINTEL(int element_size, int block_width, int block_height, int block_count, const __global void* src_base_pointer, int memory_width,                int memory_height, int memory_pitch,  int2 coordinate,  private void* dst_pointer);
; void __spirv_Subgroup2DBlockLoadTransformINTEL(int element_size, int block_width, int block_height, int block_count, const __global void* src_base_pointer, int memory_width,                int memory_height, int memory_pitch,  int2 coordinate,  private void* dst_pointer);
; void __spirv_Subgroup2DBlockPrefetchINTEL(     int element_size, int block_width, int block_height, int block_count, const __global void* src_base_pointer, int memory_width,                int memory_height, int memory_pitch,  int2 coordinate                            );
; void __spirv_Subgroup2DBlockStoreINTEL(        int element_size, int block_width, int block_height, int block_count, const  private void* src_pointer,      __global void* dst_base_pointer, int memory_width,  int memory_height, int memory_pitch, int2 coordinate          );
;
; void foo(const __global void* base_address, __global void* dst_base_pointer, int width, int height, int pitch, int2 coord, private void* dst_pointer, const private void* src_pointer) {
;     const int i = 42;
;     __spirv_Subgroup2DBlockLoadINTEL(i, i, i, i, base_address, width, height, pitch, coord, dst_pointer);
;     __spirv_Subgroup2DBlockLoadTransformINTEL(i, i, i, i, base_address, width, height, pitch, coord, dst_pointer);
;     __spirv_Subgroup2DBlockLoadTransposeINTEL(i, i, i, i, base_address, width, height, pitch, coord, dst_pointer);
;     __spirv_Subgroup2DBlockPrefetchINTEL(i, i, i, i, base_address, width, height, pitch, coord);
;     __spirv_Subgroup2DBlockStoreINTEL(i, i, i, i, src_pointer, dst_base_pointer, width, height, pitch, coord);
;   }
; clang -cc1 -cl-std=clc++2021 -triple spir64-unknown-unknown -emit-llvm -finclude-default-header source.cl -o tmp.ll



; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_2d_block_io %s -o %t.spt
; RUN: FileCheck %s --input-file=%t.spt
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_2d_block_io %s -o - -filetype=obj | spirv-val %}

; CHECK: OpCapability Subgroup2DBlockIOINTEL
; CHECK: OpCapability Subgroup2DBlockTransformINTEL
; CHECK: OpCapability Subgroup2DBlockTransposeINTEL
; CHECK: OpExtension "SPV_INTEL_2d_block_io"
; CHECK: %[[#Int8Ty:]] = OpTypeInt 8 0
; CHECK: %[[#GlbPtrTy:]] = OpTypePointer CrossWorkgroup %[[#Int8Ty]]
; CHECK: %[[#Int32Ty:]] = OpTypeInt 32 0
; CHECK: %[[#VectorTy:]] = OpTypeVector %[[#Int32Ty]] 2
; CHECK: %[[#PrvPtrTy:]] = OpTypePointer Function %[[#Int8Ty]]
; CHECK: %[[#VoidTy:]] = OpTypeVoid
; CHECK: %[[#Const42:]] = OpConstant %[[#Int32Ty]] 42
; CHECK: %[[#BaseSrc:]] = OpFunctionParameter %[[#GlbPtrTy]]
; CHECK: %[[#BaseDst:]] = OpFunctionParameter %[[#GlbPtrTy]]
; CHECK: %[[#Width:]] = OpFunctionParameter %[[#Int32Ty]]
; CHECK: %[[#Height:]] = OpFunctionParameter %[[#Int32Ty]]
; CHECK: %[[#Pitch:]] = OpFunctionParameter %[[#Int32Ty]]
; CHECK: %[[#Coord:]] = OpFunctionParameter %[[#VectorTy]]
; CHECK: %[[#Dst:]] = OpFunctionParameter %[[#PrvPtrTy]]
; CHECK: %[[#Src:]] = OpFunctionParameter %[[#PrvPtrTy]]
; CHECK: OpSubgroup2DBlockLoadINTEL %[[#Const42]] %[[#Const42]] %[[#Const42]] %[[#Const42]] %[[#BaseSrc]] %[[#Width]] %[[#Height]] %[[#Pitch]] %[[#Coord]] %[[#Dst]]
; CHECK: OpSubgroup2DBlockLoadTransformINTEL %[[#Const42]] %[[#Const42]] %[[#Const42]] %[[#Const42]] %[[#BaseSrc]] %[[#Width]] %[[#Height]] %[[#Pitch]] %[[#Coord]] %[[#Dst]]
; CHECK: OpSubgroup2DBlockLoadTransposeINTEL %[[#Const42]] %[[#Const42]] %[[#Const42]] %[[#Const42]] %[[#BaseSrc]] %[[#Width]] %[[#Height]] %[[#Pitch]] %[[#Coord]] %[[#Dst]]
; CHECK: OpSubgroup2DBlockPrefetchINTEL %[[#Const42]] %[[#Const42]] %[[#Const42]] %[[#Const42]] %[[#BaseSrc]] %[[#Width]] %[[#Height]] %[[#Pitch]] %[[#Coord]]
; CHECK: OpSubgroup2DBlockStoreINTEL %[[#Const42]] %[[#Const42]] %[[#Const42]] %[[#Const42]] %[[#Src]] %[[#BaseDst]] %[[#Width]] %[[#Height]] %[[#Pitch]] %[[#Coord]]



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

!opencl.spir.version = !{!0}
!spirv.Source = !{!1}
!llvm.ident = !{!2}

!0 = !{i32 1, i32 0}
!1 = !{i32 4, i32 100000}
!2 = !{!"clang version 17.0.0"}