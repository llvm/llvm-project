; RUN: llc -O0 -mtriple=spirv-unknown-vulkan1.3-pixel %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan1.3-pixel %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; CHECK-DAG: OpEntryPoint Fragment %[[#entry:]] "main" {{.*}}
; CHECK-DAG: OpExecutionMode %[[#entry]] OriginUpperLeft

@.str.b0 = private unnamed_addr constant [3 x i8] c"B0\00", align 1

define void @main() #0 {
entry:
  %0 = tail call target("spirv.VulkanBuffer", [0 x i32], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0i32_12_1t(i32 0, i32 1, i32 1, i32 0, i1 false, ptr nonnull @.str.b0)
  %1 = tail call noundef align 4 dereferenceable(4) ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i32_12_1t(target("spirv.VulkanBuffer", [0 x i32], 12, 1) %0, i32 0)
  store i32 1, ptr addrspace(11) %1, align 4

  ret void
}

attributes #0 = { "hlsl.shader"="pixel" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(none) }
