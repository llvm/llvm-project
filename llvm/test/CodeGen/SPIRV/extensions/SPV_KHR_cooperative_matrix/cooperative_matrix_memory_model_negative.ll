; Negative control for the cooperative-matrix memory-model upgrade: a plain Shader
; compute module that does not require CooperativeMatrixKHR must keep the default
; GLSL450 memory model, so the upgrade is conditional on the CooperativeMatrixKHR
; requirement, not a blanket change applied to every Shader-flavor module.

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan1.3-compute %s -o - | FileCheck %s

; No cooperative-matrix capability is required, so no Vulkan memory model is pulled
; in: neither the capability nor the extension is emitted, and the module keeps the
; default Logical GLSL450 memory model (not Logical VulkanKHR).
; CHECK-NOT: OpCapability VulkanMemoryModelKHR
; CHECK-NOT: OpExtension "SPV_KHR_vulkan_memory_model"
; CHECK: OpMemoryModel Logical GLSL450
; CHECK-NOT: OpMemoryModel Logical VulkanKHR

define spir_func void @nop() {
entry:
  ret void
}
