; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - -filetype=obj | spirv-val %}

; async_work_group_copy with untyped pointers uses OpUntypedGroupAsyncCopyKHR
; with an explicit Element Num Bytes operand.

; CHECK-DAG: OpCapability UntypedPointersKHR
; CHECK-DAG: %[[#I32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#EVENT:]] = OpTypeEvent
; CHECK-DAG: %[[#ELEMBYTES:]] = OpConstant %[[#I32]] 4

; CHECK: OpUntypedGroupAsyncCopyKHR %[[#EVENT]] %[[#]] %[[#]] %[[#]] %[[#ELEMBYTES]] %[[#]] %[[#]] %[[#]]
define spir_kernel void @t(ptr addrspace(1) %dst, ptr addrspace(3) %src, i32 %n) {
  %e = call spir_func ptr @_Z21async_work_group_copyPU3AS1iPKU3AS3ij9ocl_event(ptr addrspace(1) %dst, ptr addrspace(3) %src, i32 %n, ptr null)
  ret void
}
declare spir_func ptr @_Z21async_work_group_copyPU3AS1iPKU3AS3ij9ocl_event(ptr addrspace(1), ptr addrspace(3), i32, ptr)
