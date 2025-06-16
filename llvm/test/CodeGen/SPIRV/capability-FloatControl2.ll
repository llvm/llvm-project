; RUN: llc -O0 -mtriple=spirv1.6-vulkan1.3-compute %s -o - | FileCheck %s --check-prefix=CHECK-NOEXT
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-vulkan1.3-compute %s -o - -filetype=obj | spirv-val %}

; RUN: llc -O0 -mtriple=spirv1.6-vulkan1.3-compute -spirv-ext=+SPV_KHR_float_controls2 %s -o - | FileCheck %s --check-prefix=CHECK-EXT
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-vulkan1.3-compute %s -o - -filetype=obj | spirv-val %}

; CHECK-NOEXT-NOT: OpDecorate FPFastMathMode

; CHECK-EXT: OpCapability FloatControls2
; CHECK-EXT: OpExtension "SPV_KHR_float_controls2"
; CHECK-EXT: OpDecorate {{%[0-9]+}} FPFastMathMode NotNaN|NotInf|NSZ|AllowRecip|Fast

@.str = private unnamed_addr constant [3 x i8] c"In\00", align 1
@.str.2 = private unnamed_addr constant [4 x i8] c"Out\00", align 1

define void @main() local_unnamed_addr #0 {
  %1 = tail call target("spirv.Image", float, 5, 2, 0, 0, 2, 0) @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_5_2_0_0_2_0t(i32 0, i32 0, i32 1, i32 0, i1 false, ptr nonnull @.str)
  %2 = tail call target("spirv.Image", float, 5, 2, 0, 0, 2, 0) @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_5_2_0_0_2_0t(i32 0, i32 1, i32 1, i32 0, i1 false, ptr nonnull @.str.2)
  %3 = tail call i32 @llvm.spv.thread.id.in.group(i32 0)
  %4 = tail call noundef align 4 dereferenceable(4) ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.Image_f32_5_2_0_0_2_0t(target("spirv.Image", float, 5, 2, 0, 0, 2, 0) %1, i32 %3)
  %5 = load float, ptr addrspace(11) %4, align 4
  %6 = fadd reassoc nnan ninf nsz arcp afn float %5, 0x4011333340000000
  %7 = tail call noundef align 4 dereferenceable(4) ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.Image_f32_5_2_0_0_2_0t(target("spirv.Image", float, 5, 2, 0, 0, 2, 0) %2, i32 %3)
  store float %6, ptr addrspace(11) %7, align 4
  ret void
}

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, inaccessiblemem: none) "approx-func-fp-math"="true" "frame-pointer"="all" "hlsl.numthreads"="8,1,1" "hlsl.shader"="compute" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
