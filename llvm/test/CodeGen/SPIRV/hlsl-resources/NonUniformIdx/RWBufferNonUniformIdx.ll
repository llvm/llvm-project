; RUN: llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - | FileCheck %s --match-full-lines
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpCapability Shader
; CHECK-DAG: OpCapability ShaderNonUniformEXT
; CHECK-DAG: OpCapability StorageTexelBufferArrayNonUniformIndexingEXT
; CHECK-DAG: OpDecorate {{%[0-9]+}} NonUniformEXT
; CHECK-DAG: OpDecorate %[[#access:]] NonUniformEXT
; CHECK-DAG: OpDecorate %[[#load:]] NonUniformEXT
@ReadWriteBuf.str = private unnamed_addr constant [13 x i8] c"ReadWriteBuf\00", align 1

define void @main() local_unnamed_addr #0 {
entry:
  %0 = tail call i32 @llvm.spv.thread.id.in.group.i32(i32 0)
  %1 = tail call noundef i32 @llvm.spv.resource.nonuniformindex(i32 %0)
  %2 = tail call target("spirv.Image", i32, 5, 2, 0, 0, 2, 33) @llvm.spv.resource.handlefromimplicitbinding.tspirv.Image_i32_5_2_0_0_2_33t(i32 0, i32 0, i32 64, i32 %1, ptr nonnull @ReadWriteBuf.str)
  %3 = tail call noundef align 4 dereferenceable(4) ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.Image_i32_5_2_0_0_2_33t(target("spirv.Image", i32, 5, 2, 0, 0, 2, 33) %2, i32 96)
; CHECK: {{%[0-9]+}} = OpCompositeExtract {{.*}}
; CHECK: %[[#access]] = OpAccessChain {{.*}}
; CHECK: %[[#load]] = OpLoad {{%[0-9]+}} %[[#access]]
; CHECK: OpImageWrite %[[#load]] {{%[0-9]+}} {{%[0-9]+}}
  store i32 95, ptr addrspace(11) %3, align 4
  ret void
}
