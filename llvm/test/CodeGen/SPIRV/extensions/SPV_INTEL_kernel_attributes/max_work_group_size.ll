; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_kernel_attributes %s -o - | FileCheck %s
; XFAIL: *

; CHECK: OpCapability FPGAKernelAttributesINTEL
; CHECK: OpExtension "SPV_INTEL_kernel_attributes"
; CHECK: OpEntryPoint Kernel %[[DIM1:]] "Dim1"
; CHECK: OpEntryPoint Kernel %[[DIM2:]] "Dim2"
; CHECK: OpEntryPoint Kernel %[[DIM3:]] "Dim3"
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
