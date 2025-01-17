; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-vulkan-library %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-vulkan-library %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: [[float:%[0-9]+]] = OpTypeFloat 32
; CHECK-DAG: [[v4float:%[0-9]+]] = OpTypeVector [[float]] 4
; CHECK-DAG: [[int:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[zero:%[0-9]+]] = OpConstant [[int]] 0
; CHECK-DAG: [[one:%[0-9]+]] = OpConstant [[int]] 1
; CHECK-DAG: [[twenty:%[0-9]+]] = OpConstant [[int]] 20
; CHECK-DAG: [[twenty_three:%[0-9]+]] = OpConstant [[int]] 23
; CHECK-DAG: [[ImageType:%[0-9]+]] = OpTypeImage [[float]] Buffer 2 0 0 2 Rgba32f 
; CHECK-DAG: [[ImagePtr:%[0-9]+]] = OpTypePointer UniformConstant [[ImageType]]
; CHECK: [[Var:%[0-9]+]] = OpVariable [[ImagePtr]] UniformConstant

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, inaccessiblemem: none)
define void @main() local_unnamed_addr #0 {
entry:
; CHECK: [[H:%[0-9]+]] = OpLoad [[ImageType]] [[Var]]
  %s_h.i = tail call target("spirv.Image", float, 5, 2, 0, 0, 2, 1) @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_5_2_0_0_2_0t(i32 3, i32 5, i32 1, i32 0, i1 false)

; CHECK: [[R:%[0-9]+]] = OpImageRead [[v4float]] [[H]] [[one]]
; CHECK: [[V:%[0-9]+]] = OpCompositeExtract [[float]] [[R]] 0
  %0 = tail call noundef nonnull align 4 dereferenceable(4) ptr @llvm.spv.resource.getpointer.p0.tspirv.Image_f32_5_2_0_0_2_0t(target("spirv.Image", float, 5, 2, 0, 0, 2, 1) %s_h.i, i32 1)
  %1 = load float, ptr %0, align 4
; CHECK: OpBranch [[bb_store:%[0-9]+]]
  br label %bb_store

; CHECK: [[bb_store]] = OpLabel
bb_store:

; CHECK: [[H:%[0-9]+]] = OpLoad [[ImageType]] [[Var]]
; CHECK: OpImageWrite [[H]] [[zero]] [[V]]
  %2 = tail call noundef nonnull align 4 dereferenceable(4) ptr @llvm.spv.resource.getpointer.p0.tspirv.Image_f32_5_2_0_0_2_0t(target("spirv.Image", float, 5, 2, 0, 0, 2, 1) %s_h.i, i32 0)
  store float %1, ptr %2, align 4
; CHECK: OpBranch [[bb_both:%[0-9]+]]
  br label %bb_both

; CHECK: [[bb_both]] = OpLabel
bb_both:  
; CHECK: [[H:%[0-9]+]] = OpLoad [[ImageType]] [[Var]]
; CHECK: [[R:%[0-9]+]] = OpImageRead [[v4float]] [[H]] [[twenty_three]]
; CHECK: [[V:%[0-9]+]] = OpCompositeExtract [[float]] [[R]] 0
  %3 = tail call noundef nonnull align 4 dereferenceable(4) ptr @llvm.spv.resource.getpointer.p0.tspirv.Image_f32_5_2_0_0_2_0t(target("spirv.Image", float, 5, 2, 0, 0, 2, 1) %s_h.i, i32 23)
  %4 = load float, ptr %3, align 4
  
; CHECK: [[H:%[0-9]+]] = OpLoad [[ImageType]] [[Var]]
; CHECK: OpImageWrite [[H]] [[twenty]] [[V]]
  %5 = tail call noundef nonnull align 4 dereferenceable(4) ptr @llvm.spv.resource.getpointer.p0.tspirv.Image_f32_5_2_0_0_2_0t(target("spirv.Image", float, 5, 2, 0, 0, 2, 1) %s_h.i, i32 20)
  store float %4, ptr %5, align 4
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare ptr @llvm.spv.resource.getpointer.p0.tspirv.Image_f32_5_2_0_0_2_0t(target("spirv.Image", float, 5, 2, 0, 0, 2, 1), i32) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare target("spirv.Image", float, 5, 2, 0, 0, 2, 1) @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_5_2_0_0_2_0t(i32, i32, i32, i32, i1) #1

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, inaccessiblemem: none) "frame-pointer"="all" "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(none) }
