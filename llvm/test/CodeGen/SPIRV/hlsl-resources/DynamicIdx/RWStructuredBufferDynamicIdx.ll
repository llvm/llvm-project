; RUN: llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - | FileCheck %s --match-full-lines

%"__cblayout_$Globals" = type <{ i32 }>

@i = external hidden local_unnamed_addr addrspace(12) global i32, align 4
@ReadWriteStructuredBuf.str = private unnamed_addr constant [23 x i8] c"ReadWriteStructuredBuf\00", align 1
@"$Globals.cb" = local_unnamed_addr global target("spirv.VulkanBuffer", target("spirv.Layout", %"__cblayout_$Globals", 4, 0), 2, 0) poison
@"$Globals.str" = private unnamed_addr constant [9 x i8] c"$Globals\00", align 1

; CHECK: OpCapability Shader
; CHECK: OpCapability StorageBufferArrayDynamicIndexing
define void @main() local_unnamed_addr #0 {
entry:
  %"$Globals.cb_h.i.i" = tail call target("spirv.VulkanBuffer", target("spirv.Layout", %"__cblayout_$Globals", 4, 0), 2, 0) @"llvm.spv.resource.handlefromimplicitbinding.tspirv.VulkanBuffer_tspirv.Layout_s___cblayout_$Globalss_4_0t_2_0t"(i32 2, i32 0, i32 1, i32 0, ptr nonnull @"$Globals.str")
  store target("spirv.VulkanBuffer", target("spirv.Layout", %"__cblayout_$Globals", 4, 0), 2, 0) %"$Globals.cb_h.i.i", ptr @"$Globals.cb", align 8
  %0 = load i32, ptr addrspace(12) @i, align 4
  %1 = tail call target("spirv.VulkanBuffer", [0 x i32], 12, 1) @llvm.spv.resource.handlefromimplicitbinding.tspirv.VulkanBuffer_a0i32_12_1t(i32 0, i32 0, i32 64, i32 %0, ptr nonnull @ReadWriteStructuredBuf.str)
  %2 = tail call noundef align 4 dereferenceable(4) ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0i32_12_1t(target("spirv.VulkanBuffer", [0 x i32], 12, 1) %1, i32 99)
  store i32 98, ptr addrspace(11) %2, align 4
  ret void
}
