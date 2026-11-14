; Adapted from Khronos Translator test suite: test/extensions/INTEL/SPV_INTEL_device_barrier/

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_device_barrier %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_device_barrier %s -o - -filetype=obj | spirv-val %}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers,+SPV_INTEL_device_barrier %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers,+SPV_INTEL_device_barrier %s -o - -filetype=obj | spirv-val %}

; The BuiltIn cannot be used without the extension.
; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

; Using both the variable and the call form yields a single decorated variable.
; CHECK-DAG: OpCapability DeviceBarrierINTEL
; CHECK-DAG: OpExtension "SPV_INTEL_device_barrier"
; CHECK: OpDecorate %[[#VAR:]] BuiltIn DeviceBarrierValidINTEL
; CHECK-NOT: OpDecorate %[[#]] BuiltIn DeviceBarrierValidINTEL
; CHECK: %[[#BOOL:]] = OpTypeBool
; CHECK: %[[#VAR]] = {{OpVariable|OpUntypedVariableKHR}}
; CHECK: OpLoad %[[#BOOL]]

; CHECK-ERROR: LLVM ERROR: DeviceBarrierValidINTEL built-in requires the following SPIR-V extension: SPV_INTEL_device_barrier

@__spirv_BuiltInDeviceBarrierValidINTEL = external addrspace(1) global i1

define spir_kernel void @f(ptr addrspace(1) %out) {
entry:
  %v = load i1, ptr addrspace(1) @__spirv_BuiltInDeviceBarrierValidINTEL
  %z = zext i1 %v to i32
  store i32 %z, ptr addrspace(1) %out
  ret void
}

define spir_kernel void @both(ptr addrspace(1) %out) {
entry:
  %a = load i1, ptr addrspace(1) @__spirv_BuiltInDeviceBarrierValidINTEL
  %b = call spir_func i1 @_Z38__spirv_BuiltInDeviceBarrierValidINTELv()
  %v = or i1 %a, %b
  %z = zext i1 %v to i32
  store i32 %z, ptr addrspace(1) %out
  ret void
}

declare spir_func i1 @_Z38__spirv_BuiltInDeviceBarrierValidINTELv()
