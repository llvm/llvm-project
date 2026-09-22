; RUN: llc -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val %}

; A function-local aggregate containing a storage buffer pointer requires
; VariablePointersStorageBuffer, even when the pointer is nested.

%struct.A = type { target("spirv.VulkanBuffer", [0 x i32], 12, 1) }

@.str = private unnamed_addr constant [4 x i8] c"Buf\00", align 1

define void @main() #0 {
entry:
  %handle = call target("spirv.VulkanBuffer", [0 x i32], 12, 1) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr @.str)
  %a = alloca %struct.A, align 4
  %value = insertvalue %struct.A poison, target("spirv.VulkanBuffer", [0 x i32], 12, 1) %handle, 0
  store volatile %struct.A %value, ptr %a, align 4
  ret void
}

; CHECK: OpCapability VariablePointersStorageBuffer
; CHECK: %[[BUFFER:[0-9]+]] = OpTypePointer StorageBuffer
; CHECK: %[[A:[0-9]+]] = OpTypeStruct %[[BUFFER]]
; CHECK: %[[A_PTR:[0-9]+]] = OpTypePointer Function %[[A]]
; CHECK: OpVariable %[[A_PTR]] Function

declare target("spirv.VulkanBuffer", [0 x i32], 12, 1) @llvm.spv.resource.handlefrombinding(i32, i32, i32, i32, ptr)

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
