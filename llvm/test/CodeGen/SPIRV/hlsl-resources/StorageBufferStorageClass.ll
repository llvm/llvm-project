; RUN: llc -O0 -mtriple=spirv32v1.2-unknown-vulkan-library --spirv-ext=+SPV_KHR_storage_buffer_storage_class %s -o - | FileCheck %s --check-prefixes=CHECK,EXT
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32v1.2-unknown-vulkan-library --spirv-ext=+SPV_KHR_storage_buffer_storage_class %s -o - -filetype=obj | spirv-val --target-env spv1.2 %}
; RUN: llc -O0 -mtriple=spirv32v1.3-unknown-vulkan-library %s -o - | FileCheck %s --check-prefixes=CHECK,NOEXT
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32v1.3-unknown-vulkan-library %s -o - -filetype=obj | spirv-val --target-env spv1.3 %}
; RUN: not --crash llc -O0 -mtriple=spirv32v1.2-unknown-vulkan-library %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR

; StorageBuffer requires SPIR-V 1.3 or SPV_KHR_storage_buffer_storage_class.

; ERR: LLVM ERROR: Adding SPIR-V requirements this target can't satisfy.

; EXT: OpExtension "SPV_KHR_storage_buffer_storage_class"
; NOEXT-NOT: OpExtension
; CHECK: OpDecorate %[[#BUF:]] Block
; CHECK: %[[#BUF]] = OpTypeStruct
; CHECK: %[[#PTR:]] = OpTypePointer StorageBuffer %[[#BUF]]
; CHECK: %[[#]] = OpVariable %[[#PTR]] StorageBuffer

@.str = private unnamed_addr constant [2 x i8] c"B\00", align 1

define void @main() #0 {
entry:
  %h = tail call target("spirv.VulkanBuffer", [0 x i32], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0i32_12_1t(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.str)
  %p = tail call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i32_12_1t(target("spirv.VulkanBuffer", [0 x i32], 12, 1) %h, i32 0)
  store i32 1, ptr addrspace(11) %p, align 4
  ret void
}

declare ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i32_12_1t(target("spirv.VulkanBuffer", [0 x i32], 12, 1), i32)

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
