; DeviceBarrierValidINTEL is the old name of DeviceBarrierExecutionValidINTEL.
; It is still accepted as an alias and lowers to the same BuiltIn.

; Test with typed pointers:
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_device_barrier %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_device_barrier %s -o - -filetype=obj | spirv-val %}

; Test with untyped pointers.
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers,+SPV_INTEL_device_barrier %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers,+SPV_INTEL_device_barrier %s -o - -filetype=obj | spirv-val %}

; The BuiltIn cannot be used without the extension.
; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

; CHECK-SPIRV-DAG: OpCapability DeviceBarrierINTEL
; CHECK-SPIRV-DAG: OpExtension "SPV_INTEL_device_barrier"
; CHECK-SPIRV: OpDecorate %[[#Id:]] BuiltIn DeviceBarrierExecutionValidINTEL
; CHECK-SPIRV: %[[#Id]] = {{OpVariable|OpUntypedVariableKHR}}

; CHECK-ERROR: LLVM ERROR: DeviceBarrierExecutionValidINTEL built-in requires the following SPIR-V extension: SPV_INTEL_device_barrier

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; Function Attrs: nounwind readnone
define spir_kernel void @f() #0 !kernel_arg_addr_space !0 !kernel_arg_access_qual !0 !kernel_arg_type !0 !kernel_arg_base_type !0 !kernel_arg_type_qual !0 {
entry:
  %0 = call spir_func i1 @_Z38__spirv_BuiltInDeviceBarrierValidINTELv()
  ret void
}

declare spir_func i1 @_Z38__spirv_BuiltInDeviceBarrierValidINTELv()

attributes #0 = { alwaysinline nounwind readonly "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!0 = !{}
