; RUN: llc -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val %}

; A function-local aggregate containing a storage buffer pointer requires
; VariablePointersStorageBuffer, even when the pointer is nested.

%struct.A = type { target("spirv.VulkanBuffer", [0 x i32], 12, 1) }

define void @main() #0 {
entry:
  %a = alloca %struct.A, align 4
  store volatile %struct.A zeroinitializer, ptr %a, align 4
  ret void
}

; CHECK: OpCapability VariablePointersStorageBuffer
; CHECK: %[[BUFFER:[0-9]+]] = OpTypePointer StorageBuffer
; CHECK: %[[A:[0-9]+]] = OpTypeStruct %[[BUFFER]]
; CHECK: %[[A_PTR:[0-9]+]] = OpTypePointer Function %[[A]]
; CHECK: OpVariable %[[A_PTR]] Function

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
