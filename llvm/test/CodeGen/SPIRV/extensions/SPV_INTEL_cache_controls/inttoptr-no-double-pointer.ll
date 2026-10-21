; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_cache_controls,+SPV_INTEL_variable_length_array %s -o - | FileCheck %s
; TODO: %if spirv-tools %{ llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_cache_controls,+SPV_INTEL_variable_length_array %s -o - -filetype=obj | spirv-val %}

; Check that inttoptr followed by ptr.annotation does not produce a
; double-pointer type for OpConvertUToPtr (and there is no OpBitcast back to
; single-pointer type).

; CHECK-DAG: %[[#UCHAR:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#PTR:]] = OpTypePointer CrossWorkgroup %[[#UCHAR]]
; CHECK:     %[[#CONV:]] = OpConvertUToPtr %[[#PTR]]
; CHECK-NOT: OpBitcast %[[#PTR]] %[[#CONV]]
; CHECK:     OpFunctionCall %[[#]] %[[#]] %[[#CONV]]

@.str.file = private unnamed_addr addrspace(1) constant [1 x i8] zeroinitializer, section "llvm.metadata", align 1
@.str.cachecontrol = private unnamed_addr addrspace(1) constant [13 x i8] c"{6442:\220,1\22}\00", section "llvm.metadata", align 1
@.str.cachecontrol.1 = private unnamed_addr addrspace(1) constant [13 x i8] c"{6442:\221,1\22}\00", section "llvm.metadata", align 1

declare spir_func void @_Z45intel_sub_group_2d_block_prefetch_16b_8r16x2cPU3AS1viiiDv2_i(ptr addrspace(1) nonnull, i32, i32, i32, <2 x i32>) #0

define spir_kernel void @test_inttoptr_annotation(ptr addrspace(1) %base) {
entry:
  %int = ptrtoint ptr addrspace(1) %base to i64
  %vec = insertelement <4 x i64> zeroinitializer, i64 %int, i64 0
  %cast = bitcast <4 x i64> %vec to <8 x i32>
  %recast = bitcast <8 x i32> %cast to <4 x i64>
  %extracted = extractelement <4 x i64> %recast, i64 0
  %int2ptr = inttoptr i64 %extracted to ptr addrspace(1)
  %coords = insertelement <2 x i32> zeroinitializer, i32 42, i32 1
  %a1 = call ptr addrspace(1) @llvm.ptr.annotation.p1.p1(ptr addrspace(1) %int2ptr, ptr addrspace(1) @.str.cachecontrol, ptr addrspace(1) @.str.file, i32 0, ptr addrspace(1) null)
  %a2 = call ptr addrspace(1) @llvm.ptr.annotation.p1.p1(ptr addrspace(1) %a1, ptr addrspace(1) @.str.cachecontrol.1, ptr addrspace(1) @.str.file, i32 0, ptr addrspace(1) null)
  call spir_func void @_Z45intel_sub_group_2d_block_prefetch_16b_8r16x2cPU3AS1viiiDv2_i(ptr addrspace(1) nonnull %a2, i32 8192, i32 4096, i32 8192, <2 x i32> %coords) #0
  ret void
}

declare ptr addrspace(1) @llvm.ptr.annotation.p1.p1(ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), i32, ptr addrspace(1))

attributes #0 = { nounwind memory(argmem: read) }
