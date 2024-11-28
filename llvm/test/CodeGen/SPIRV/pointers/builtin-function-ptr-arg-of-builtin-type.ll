; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#INT8:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#PTR_INT8:]] = OpTypePointer Function %[[#INT8]]
; CHECK-DAG: %[[#EVENT:]] = OpTypeEvent
; CHECK-DAG: %[[#FUNC_TY:]] = OpTypeFunction %[[#]] %[[#PTR_INT8]] %[[#PTR_INT8]] %[[#]] %[[#]] %[[#EVENT]]
; CHECK-DAG: %[[#]] = OpFunction %[[#]] None %[[#FUNC_TY]]

define spir_kernel void @foo(ptr %a, ptr %b) {
  %call = call spir_func ptr @_Z29async_work_group_strided_copyPU3AS3hPU3AS1Khmm9ocl_event(ptr %a, ptr %b, i64 1, i64 1, ptr null)
  ret void
}

declare spir_func ptr @_Z29async_work_group_strided_copyPU3AS3hPU3AS1Khmm9ocl_event(ptr, ptr, i64, i64, ptr)
