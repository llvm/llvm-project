; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv1.6-vulkan1.3-library %s -o - | FileCheck %s --match-full-lines
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-vulkan1.3-library %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

@.str.signed = private unnamed_addr constant [7 x i8] c"signed\00", align 1
@.str.unsigned = private unnamed_addr constant [9 x i8] c"unsigned\00", align 1

; CHECK: {{.*}} OpFunction {{.*}}
define void @main_unsigned() local_unnamed_addr #0 {
entry:
  %s_h.i = tail call target("spirv.Image", i32, 5, 2, 0, 0, 2, 0) @llvm.spv.resource.handlefrombinding.tspirv.Image_i32_5_2_0_0_2_0t(i32 3, i32 5, i32 1, i32 0, ptr nonnull @.str.unsigned)

; CHECK: {{%[0-9]+}} = OpImageRead {{%[0-9]+}} {{%[0-9]+}} {{%[0-9]+}}
  %0 = tail call noundef nonnull align 4 dereferenceable(4) ptr @llvm.spv.resource.getpointer.p0.tspirv.Image_i32_5_2_0_0_2_0t(target("spirv.Image", i32, 5, 2, 0, 0, 2, 0) %s_h.i, i32 1)
  %1 = load i32, ptr %0, align 4
  %2 = tail call noundef nonnull align 4 dereferenceable(4) ptr @llvm.spv.resource.getpointer.p0.tspirv.Image_i32_5_2_0_0_2_0t(target("spirv.Image", i32, 5, 2, 0, 0, 2, 0) %s_h.i, i32 0)
; CHECK: OpImageWrite {{%[0-9]+}} {{%[0-9]+}} {{%[0-9]+}}
  store i32 %1, ptr %2, align 4
  ret void
}

; CHECK: {{.*}} OpFunction {{.*}}
define void @main_signed() local_unnamed_addr #0 {
entry:
  %s_h.i = tail call target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 0) @llvm.spv.resource.handlefrombinding.tspirv.SignedImage_i32_5_2_0_0_2_0t(i32 3, i32 5, i32 1, i32 0, ptr nonnull @.str.signed)

; CHECK: {{%[0-9]+}} = OpImageRead {{%[0-9]+}} {{%[0-9]+}} {{%[0-9]+}} SignExtend
  %0 = tail call noundef nonnull align 4 dereferenceable(4) ptr @llvm.spv.resource.getpointer.p0.tspirv.SignedImage_i32_5_2_0_0_2_0t(target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 0) %s_h.i, i32 1)
  %1 = load i32, ptr %0, align 4
  %2 = tail call noundef nonnull align 4 dereferenceable(4) ptr @llvm.spv.resource.getpointer.p0.tspirv.SignedImage_i32_5_2_0_0_2_0t(target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 0) %s_h.i, i32 0)
; CHECK: OpImageWrite {{%[0-9]+}} {{%[0-9]+}} {{%[0-9]+}} SignExtend
  store i32 %1, ptr %2, align 4
  ret void
}


attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
