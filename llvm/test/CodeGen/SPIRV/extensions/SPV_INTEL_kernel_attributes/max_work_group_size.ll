; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_kernel_attributes %s -o - | FileCheck %s
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-amd-amdhsa %s -o - | FileCheck %s
; %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_kernel_attributes %s -o - -filetype=obj | spirv-val %}
; %if spirv-tools %{ llc -O0 -mtriple=spirv64-amd-amdhsa %s -o - -filetype=obj | spirv-val %}

; CHECK: OpCapability KernelAttributesINTEL
; CHECK: OpExtension "SPV_INTEL_kernel_attributes"
; CHECK: OpEntryPoint {{.*}} %[[DIM1:[0-9]+]] "Dim1"
; CHECK: OpEntryPoint {{.*}} %[[DIM2:[0-9]+]] "Dim2"
; CHECK: OpEntryPoint {{.*}} %[[DIM3:[0-9]+]] "Dim3"
; CHECK: OpExecutionMode %[[DIM1]] MaxWorkgroupSizeINTEL 4 1 1
; CHECK: OpExecutionMode %[[DIM2]] MaxWorkgroupSizeINTEL 8 4 1
; CHECK: OpExecutionMode %[[DIM3]] MaxWorkgroupSizeINTEL 16 8 4
; CHECK: %[[DIM1]] = OpFunction
; CHECK: %[[DIM2]] = OpFunction
; CHECK: %[[DIM3]] = OpFunction

define spir_kernel void @Dim1() !max_work_group_size !0 {
  ret void
}

define spir_kernel void @Dim2() !max_work_group_size !1 {
  ret void
}

define spir_kernel void @Dim3() !max_work_group_size !2 {
  ret void
}

!0 = !{i32 4}
!1 = !{i32 8, i32 4}
!2 = !{i32 16, i32 8, i32 4}
