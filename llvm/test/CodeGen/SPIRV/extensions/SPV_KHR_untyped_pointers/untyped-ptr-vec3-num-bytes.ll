; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - -filetype=obj | spirv-val %}

; The Num Bytes operands of OpUntypedPrefetchKHR and OpUntypedGroupAsyncCopyKHR
; are strides, so a 3-component vector counts as 4 components. A float3 is 16
; bytes, not 12.

; CHECK-DAG: OpCapability UntypedPointersKHR
; CHECK-DAG: OpExtension "SPV_KHR_untyped_pointers"

; CHECK-DAG: %[[#I32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#I64:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#SIZE32:]] = OpConstant %[[#I32]] 16
; CHECK-DAG: %[[#SIZE64:]] = OpConstant %[[#I64]] 16

; CHECK: %[[#NUMBYTES:]] = OpIMul %[[#I64]] %[[#]] %[[#SIZE64]]
; CHECK: OpUntypedPrefetchKHR %[[#]] %[[#NUMBYTES]]
; CHECK: OpUntypedGroupAsyncCopyKHR %[[#]] %[[#]] %[[#]] %[[#]] %[[#SIZE32]] %[[#]] %[[#]] %[[#]]

define spir_kernel void @test(ptr addrspace(1) %dst, ptr addrspace(3) %src, i64 %n) {
  call spir_func void @_Z8prefetchPU3AS1KDv3_fm(ptr addrspace(1) %dst, i64 %n)
  %e = call spir_func ptr @_Z21async_work_group_copyPU3AS1Dv3_fPKU3AS3S_j9ocl_event(ptr addrspace(1) %dst, ptr addrspace(3) %src, i32 1, ptr null)
  ret void
}

declare spir_func void @_Z8prefetchPU3AS1KDv3_fm(ptr addrspace(1), i64)
declare spir_func ptr @_Z21async_work_group_copyPU3AS1Dv3_fPKU3AS3S_j9ocl_event(ptr addrspace(1), ptr addrspace(3), i32, ptr)
