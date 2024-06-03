; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#EVENT:]] = OpTypeEvent
; CHECK-DAG: %[[#EVENT_NULL:]] = OpConstantNull %[[#EVENT]]
; CHECK-DAG: %[[#]] = OpFunctionCall %[[#]] %[[#]] %[[#]] %[[#]] %[[#]] %[[#]] %[[#EVENT_NULL]]

define spir_kernel void @foo() {
  %call = call spir_func ptr @_Z29async_work_group_strided_copyPU3AS3hPU3AS1Khmm9ocl_event(ptr null, ptr null, i64 1, i64 1, ptr null)
  ret void
}

declare spir_func ptr @_Z29async_work_group_strided_copyPU3AS3hPU3AS1Khmm9ocl_event(ptr, ptr, i64, i64, ptr)
