; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_kernel_attributes %s -o - | FileCheck %s
; XFAIL: *

; CHECK: OpCapability FPGAKernelAttributesINTEL
; CHECK: OpExtension "SPV_INTEL_kernel_attributes"
; CHECK: OpEntryPoint Kernel %[[KERNEL1:]] "test_1"
; CHECK: OpEntryPoint Kernel %[[KERNEL2:]] "test_2"
; CHECK: OpExecutionMode %[[KERNEL1]] StreamingInterfaceINTEL 0
; CHECK: OpExecutionMode %[[KERNEL2]] StreamingInterfaceINTEL 1
; CHECK: %[[KERNEL1]] = OpFunction
; CHECK: %[[KERNEL2]] = OpFunction

define spir_kernel void @test_1() !ip_interface !0
{
entry:
  ret void
}

define spir_kernel void @test_2() !ip_interface !1
{
entry:
  ret void
}

!0 = !{!"streaming"}
!1 = !{!"streaming", !"stall_free_return"}
