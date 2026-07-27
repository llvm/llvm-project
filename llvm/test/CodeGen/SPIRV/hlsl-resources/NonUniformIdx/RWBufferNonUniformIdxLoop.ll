; RUN: llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val %}

; A non-uniform index carried across a loop through a phi creates an SSA cycle
; (phi <-> add) in the use walk that propagates the NonUniformEXT decoration.
; Check that codegen terminates and still decorates both the loop-carried index
; phi and the access chain derived from it.
target triple = "spirv1.6-unknown-vulkan1.3-compute"

; CHECK-DAG: OpCapability Shader
; CHECK-DAG: OpCapability ShaderNonUniformEXT
; CHECK-DAG: OpCapability StorageTexelBufferArrayNonUniformIndexingEXT
; CHECK-DAG: %[[#phi:]] = OpPhi %[[#]]
; CHECK-DAG: %[[#access:]] = OpAccessChain {{%[0-9]+}} {{%[0-9]+}} %[[#phi]]
; CHECK-DAG: OpDecorate %[[#phi]] NonUniformEXT
; CHECK-DAG: OpDecorate %[[#access]] NonUniformEXT
@ReadWriteBuf.str = private unnamed_addr constant [13 x i8] c"ReadWriteBuf\00", align 1

define void @main() local_unnamed_addr #0 {
entry:
  %ce = call token @llvm.experimental.convergence.entry()
  %0 = tail call i32 @llvm.spv.thread.id.in.group.i32(i32 0)
  %1 = tail call noundef i32 @llvm.spv.resource.nonuniformindex(i32 %0)
  br label %header

header:
  %idx = phi i32 [ %1, %entry ], [ %idx.next, %header ]
  %cl = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %ce) ]
  %idx.next = add i32 %idx, 1
  %cond = icmp slt i32 %idx.next, 10
  br i1 %cond, label %header, label %merge

merge:
  %2 = tail call target("spirv.Image", i32, 5, 2, 0, 0, 2, 33) @llvm.spv.resource.handlefromimplicitbinding.tspirv.Image_i32_5_2_0_0_2_33t(i32 0, i32 0, i32 64, i32 %idx, ptr nonnull @ReadWriteBuf.str)
  %3 = tail call noundef align 4 dereferenceable(4) ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.Image_i32_5_2_0_0_2_33t(target("spirv.Image", i32, 5, 2, 0, 0, 2, 33) %2, i32 96) [ "convergencectrl"(token %ce) ]
  store i32 95, ptr addrspace(11) %3, align 4
  ret void
}

attributes #0 = { convergent norecurse "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
