; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Check a real (non-null) event_t reloaded from a stack slot is dereferenced
; to OpTypeEvent before being used as an OpGroupAsyncCopy Event operand.

; CHECK-SPIRV-DAG: %[[#EventTy:]] = OpTypeEvent
; CHECK-SPIRV-DAG: %[[#EventPtrTy:]] = OpTypePointer Function %[[#EventTy]]
; CHECK-SPIRV: %[[#EventVar:]] = OpBitcast %[[#EventPtrTy]]
; CHECK-SPIRV: %[[#FirstEvent:]] = OpGroupAsyncCopy %[[#EventTy]]
; CHECK-SPIRV: OpStore %[[#EventVar]] %[[#FirstEvent]]
; CHECK-SPIRV: %[[#ReloadedPtr:]] = OpLoad %[[#EventPtrTy]]
; CHECK-SPIRV: %[[#ReloadedEvent:]] = OpLoad %[[#EventTy]] %[[#ReloadedPtr]]
; CHECK-SPIRV: OpGroupAsyncCopy %[[#EventTy]] %[[#]] %[[#]] %[[#]] %[[#]] %[[#]] %[[#ReloadedEvent]]

%opencl.event_t = type opaque

define spir_kernel void @foo(ptr addrspace(1) %src, ptr addrspace(3) %dst) {
entry:
  %event = alloca ptr, align 4
  %call1 = call spir_func ptr @_Z21async_work_group_copyPU3AS1Dv2_cPKU3AS3S_j9ocl_event(ptr addrspace(1) %src, ptr addrspace(3) %dst, i32 4, ptr null)
  store ptr %call1, ptr %event, align 4
  %reloaded = load ptr, ptr %event, align 4
  %call2 = call spir_func ptr @_Z21async_work_group_copyPU3AS1Dv2_cPKU3AS3S_j9ocl_event(ptr addrspace(1) %src, ptr addrspace(3) %dst, i32 4, ptr %reloaded)
  ret void
}

declare spir_func ptr @_Z21async_work_group_copyPU3AS1Dv2_cPKU3AS3S_j9ocl_event(ptr addrspace(1), ptr addrspace(3), i32, ptr)
