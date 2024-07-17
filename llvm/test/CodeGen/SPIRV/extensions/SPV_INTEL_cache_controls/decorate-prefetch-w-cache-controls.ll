; Adapted from https://github.com/KhronosGroup/SPIRV-LLVM-Translator/tree/main/test/extensions/INTEL/SPV_INTEL_cache_controls

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_cache_controls %s -o - | FileCheck %s --check-prefixes=CHECK-SPIRV
; TODO: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_cache_controls %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV: Capability CacheControlsINTEL
; CHECK-SPIRV: Extension "SPV_INTEL_cache_controls"

; CHECK-SPIRV-DAG: OpName %[[#Ptr1:]] "ptr1"
; CHECK-SPIRV-DAG: OpName %[[#Ptr2:]] "ptr2"
; CHECK-SPIRV-DAG: OpName %[[#Ptr3:]] "ptr3"
; CHECK-SPIRV-DAG: OpDecorate %[[#Ptr1]] CacheControlLoadINTEL 0 1
; CHECK-SPIRV-DAG: OpDecorate %[[#Ptr2]] CacheControlLoadINTEL 1 1
; CHECK-SPIRV-DAG: OpDecorate %[[#Ptr3]] CacheControlStoreINTEL 2 3
; CHECK-SPIRV: OpExtInst %[[#]] %[[#]] prefetch %[[#Ptr1]] %[[#]]
; CHECK-SPIRV: OpExtInst %[[#]] %[[#]] prefetch %[[#Ptr2]] %[[#]]
; CHECK-SPIRV: OpExtInst %[[#]] %[[#]] prefetch %[[#Ptr3]] %[[#]]

; 6442 stands for CacheControlLoadINTEL token
@.str.1 = private unnamed_addr addrspace(1) constant [16 x i8] c"../prefetch.hpp\00", section "llvm.metadata"
@.str.9 = private unnamed_addr addrspace(1) constant [13 x i8] c"{6442:\220,1\22}\00", section "llvm.metadata"
@.str.10 = private unnamed_addr addrspace(1) constant [13 x i8] c"{6442:\221,1\22}\00", section "llvm.metadata"
@.str.11 = private unnamed_addr addrspace(1) constant [13 x i8] c"{6443:\222,3\22}\00", section "llvm.metadata"

define weak_odr dso_local spir_kernel void @foo(ptr addrspace(1) noundef align 1 %_arg_dataPtr) {
entry:
  %r0 = addrspacecast ptr addrspace(1) %_arg_dataPtr to ptr addrspace(4)
  %ptr1 = tail call spir_func noundef ptr addrspace(1) @_Z41__spirv_GenericCastToPtrExplicit_ToGlobalPvi(ptr addrspace(4) noundef %r0, i32 noundef 5)
  %r1 = tail call ptr addrspace(1) @llvm.ptr.annotation.p1.p1(ptr addrspace(1) %ptr1, ptr addrspace(1) @.str.9, ptr addrspace(1) @.str.1, i32 76, ptr addrspace(1) null)
  tail call spir_func void @_Z20__spirv_ocl_prefetchPU3AS1Kcm(ptr addrspace(1) noundef %r1, i64 noundef 1)
  %arrayidx3.i = getelementptr inbounds i8, ptr addrspace(4) %r0, i64 1
  %ptr2 = tail call spir_func noundef ptr addrspace(1) @_Z41__spirv_GenericCastToPtrExplicit_ToGlobalPvi(ptr addrspace(4) noundef %arrayidx3.i, i32 noundef 5)
  %r2 = tail call ptr addrspace(1) @llvm.ptr.annotation.p1.p1(ptr addrspace(1) %ptr2, ptr addrspace(1) @.str.10, ptr addrspace(1) @.str.1, i32 80, ptr addrspace(1) null)
  tail call spir_func void @_Z20__spirv_ocl_prefetchPU3AS1Kcm(ptr addrspace(1) noundef %r2, i64 noundef 1)
  %arrayidx7.i = getelementptr inbounds i8, ptr addrspace(4) %r0, i64 2
  %ptr3 = tail call spir_func noundef ptr addrspace(1) @_Z41__spirv_GenericCastToPtrExplicit_ToGlobalPvi(ptr addrspace(4) noundef %arrayidx7.i, i32 noundef 5)
  %r3 = tail call ptr addrspace(1) @llvm.ptr.annotation.p1.p1(ptr addrspace(1) %ptr3, ptr addrspace(1) @.str.11, ptr addrspace(1) @.str.1, i32 80, ptr addrspace(1) null)
  tail call spir_func void @_Z20__spirv_ocl_prefetchPU3AS1Kcm(ptr addrspace(1) noundef %r3, i64 noundef 2)
  ret void
}

declare ptr addrspace(1) @llvm.ptr.annotation.p1.p1(ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), i32, ptr addrspace(1))
declare dso_local spir_func void @_Z20__spirv_ocl_prefetchPU3AS1Kcm(ptr addrspace(1) noundef, i64 noundef)
declare dso_local spir_func noundef ptr addrspace(1) @_Z41__spirv_GenericCastToPtrExplicit_ToGlobalPvi(ptr addrspace(4) noundef, i32 noundef)
