; RUN: llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - | FileCheck %s --match-full-lines
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpCapability Shader
; CHECK-DAG: OpCapability ShaderNonUniformEXT
; CHECK-DAG: OpDecorate {{%[0-9]+}} NonUniformEXT
; CHECK-DAG: OpDecorate {{%[0-9]+}} NonUniformEXT
; CHECK-DAG: OpDecorate {{%[0-9]+}} NonUniformEXT
; CHECK-DAG: OpDecorate {{%[0-9]+}} NonUniformEXT
; CHECK-DAG: OpDecorate %[[#access1:]] NonUniformEXT
@ReadWriteStructuredBuf.str = private unnamed_addr constant [23 x i8] c"ReadWriteStructuredBuf\00", align 1

define void @main() local_unnamed_addr #0 {
entry:
  %0 = tail call i32 @llvm.spv.thread.id.in.group.i32(i32 0)
  %add.i = add i32 %0, 1
  %1 = tail call noundef i32 @llvm.spv.resource.nonuniformindex(i32 %add.i)
  %2 = tail call target("spirv.VulkanBuffer", [0 x <4 x i32>], 12, 1) @llvm.spv.resource.handlefromimplicitbinding.tspirv.VulkanBuffer_a0v4i32_12_1t(i32 0, i32 0, i32 64, i32 %1, ptr nonnull @ReadWriteStructuredBuf.str)
  %3 = tail call noundef align 16 dereferenceable(16) ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0v4i32_12_1t(target("spirv.VulkanBuffer", [0 x <4 x i32>], 12, 1) %2, i32 98)
  %4 = load <4 x i32>, ptr addrspace(11) %3, align 16
  %vecins.i = insertelement <4 x i32> %4, i32 99, i64 0
; CHECK: %[[#access1]] = OpAccessChain {{.*}}
; CHECK: OpStore %[[#access1]] {{%[0-9]+}} Aligned 16
  store <4 x i32> %vecins.i, ptr addrspace(11) %3, align 16
  ret void
}
