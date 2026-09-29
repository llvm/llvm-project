; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: OpMemoryModel Logical Vulkan
define void @main() {
entry:
  ret void
}

; AddressingModel=Logical (0), MemoryModel=VulkanKHR (3)
!spirv.MemoryModel = !{!0}
!0 = !{i32 0, i32 3}
