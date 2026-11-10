; Barriers whose execution scope is a constant other than Device do not require
; SPV_INTEL_device_barrier, even when the extension is enabled. This includes
; barriers that use Device only as the memory scope.

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_device_barrier %s -o - | FileCheck %s \
; RUN:   --implicit-check-not=DeviceBarrierINTEL --implicit-check-not=SPV_INTEL_device_barrier
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_device_barrier %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#UINT:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#SCOPE_DEVICE:]] = OpConstant %[[#UINT]] 1{{$}}
; CHECK-DAG: %[[#SCOPE_WORK_GROUP:]] = OpConstant %[[#UINT]] 2{{$}}
; CHECK-DAG: %[[#SCOPE_SUBGROUP:]] = OpConstant %[[#UINT]] 3{{$}}
; CHECK: OpControlBarrier %[[#SCOPE_WORK_GROUP]] %[[#SCOPE_DEVICE]]
; CHECK: OpControlBarrier %[[#SCOPE_SUBGROUP]] %[[#SCOPE_DEVICE]]

define spir_kernel void @test() {
  call spir_func void @_Z22__spirv_ControlBarrieriii(i32 2, i32 1, i32 520) ; Workgroup execution scope
  call spir_func void @_Z22__spirv_ControlBarrieriii(i32 3, i32 1, i32 264) ; Subgroup execution scope
  ret void
}

declare spir_func void @_Z22__spirv_ControlBarrieriii(i32, i32, i32)
