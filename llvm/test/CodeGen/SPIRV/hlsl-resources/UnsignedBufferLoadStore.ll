; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv1.6-vulkan1.3-library %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-vulkan1.3-library %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

@.str.b0 = private unnamed_addr constant [3 x i8] c"B0\00", align 1

; CHECK-DAG: [[uint:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[v2int:%[0-9]+]] = OpTypeVector [[uint]] 2
; CHECK-DAG: [[v4int:%[0-9]+]] = OpTypeVector [[uint]] 4
; CHECK-DAG: [[zero:%[0-9]+]] = OpConstant [[uint]] 0
; CHECK-DAG: [[one:%[0-9]+]] = OpConstant [[uint]] 1
; CHECK-DAG: [[twenty:%[0-9]+]] = OpConstant [[uint]] 20
; CHECK-DAG: [[twenty_three:%[0-9]+]] = OpConstant [[uint]] 23
; CHECK-DAG: [[ImageType:%[0-9]+]] = OpTypeImage [[uint]] Buffer 2 0 0 2 Unknown 
; CHECK-DAG: [[ImagePtr:%[0-9]+]] = OpTypePointer UniformConstant [[ImageType]]
; CHECK-DAG: [[Var:%[0-9]+]] = OpVariable [[ImagePtr]] UniformConstant

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, inaccessiblemem: none)
; CHECK: OpFunction
define void @main_scalar() local_unnamed_addr #0 {
entry:
; CHECK: [[H:%[0-9]+]] = OpLoad [[ImageType]] [[Var]]
  %s_h.i = tail call target("spirv.Image", i32, 5, 2, 0, 0, 2, 0) @llvm.spv.resource.handlefrombinding.tspirv.Image_i32_5_2_0_0_2_0t(i32 3, i32 5, i32 1, i32 0, ptr nonnull @.str.b0)

; CHECK: [[R:%[0-9]+]] = OpImageRead [[v4int]] [[H]] [[one]]
; CHECK: [[V:%[0-9]+]] = OpCompositeExtract [[uint]] [[R]] 0
  %0 = tail call noundef nonnull align 4 dereferenceable(4) ptr @llvm.spv.resource.getpointer.p0.tspirv.Image_i32_5_2_0_0_2_0t(target("spirv.Image", i32, 5, 2, 0, 0, 2, 0) %s_h.i, i32 1)
  %1 = load i32, ptr %0, align 4
; CHECK: OpBranch [[bb_store:%[0-9]+]]
  br label %bb_store

; CHECK: [[bb_store]] = OpLabel
bb_store:

; CHECK: [[H:%[0-9]+]] = OpLoad [[ImageType]] [[Var]]
; CHECK: OpImageWrite [[H]] [[zero]] [[V]]
  %2 = tail call noundef nonnull align 4 dereferenceable(4) ptr @llvm.spv.resource.getpointer.p0.tspirv.Image_i32_5_2_0_0_2_0t(target("spirv.Image", i32, 5, 2, 0, 0, 2, 0) %s_h.i, i32 0)
  store i32 %1, ptr %2, align 4
; CHECK: OpBranch [[bb_both:%[0-9]+]]
  br label %bb_both

; CHECK: [[bb_both]] = OpLabel
bb_both:  
; CHECK: [[H:%[0-9]+]] = OpLoad [[ImageType]] [[Var]]
; CHECK: [[R:%[0-9]+]] = OpImageRead [[v4int]] [[H]] [[twenty_three]]
; CHECK: [[V:%[0-9]+]] = OpCompositeExtract [[uint]] [[R]] 0
  %3 = tail call noundef nonnull align 4 dereferenceable(4) ptr @llvm.spv.resource.getpointer.p0.tspirv.Image_i32_5_2_0_0_2_0t(target("spirv.Image", i32, 5, 2, 0, 0, 2, 0) %s_h.i, i32 23)
  %4 = load i32, ptr %3, align 4
  
; CHECK: [[H:%[0-9]+]] = OpLoad [[ImageType]] [[Var]]
; CHECK: OpImageWrite [[H]] [[twenty]] [[V]]
  %5 = tail call noundef nonnull align 4 dereferenceable(4) ptr @llvm.spv.resource.getpointer.p0.tspirv.Image_i32_5_2_0_0_2_0t(target("spirv.Image", i32, 5, 2, 0, 0, 2, 0) %s_h.i, i32 20)
  store i32 %4, ptr %5, align 4
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, inaccessiblemem: none)
; CHECK: OpFunction
define void @main_vector2() local_unnamed_addr #0 {
entry:
; CHECK: [[H:%[0-9]+]] = OpLoad [[ImageType]] [[Var]]
  %s_h.i = tail call target("spirv.Image", i32, 5, 2, 0, 0, 2, 0) @llvm.spv.resource.handlefrombinding.tspirv.Image_i32_5_2_0_0_2_0t(i32 3, i32 5, i32 1, i32 0, ptr nonnull @.str.b0)

; CHECK: [[R:%[0-9]+]] = OpImageRead [[v4int]] [[H]] [[one]]
; CHECK: [[E0:%[0-9]+]] = OpCompositeExtract [[uint]] [[R]] 0
; CHECK: [[E1:%[0-9]+]] = OpCompositeExtract [[uint]] [[R]] 1
; CHECK: [[V:%[0-9]+]] = OpCompositeConstruct [[v2int]] [[E0]] [[E1]]
  %0 = tail call noundef nonnull align 4 dereferenceable(4) ptr @llvm.spv.resource.getpointer.p0.tspirv.Image_i32_5_2_0_0_2_0t(target("spirv.Image", i32, 5, 2, 0, 0, 2, 0) %s_h.i, i32 1)
  %1 = load <2 x i32>, ptr %0, align 4
; CHECK: OpBranch [[bb_store:%[0-9]+]]
  br label %bb_store

; CHECK: [[bb_store]] = OpLabel
bb_store:

; CHECK: [[H:%[0-9]+]] = OpLoad [[ImageType]] [[Var]]
; CHECK: OpImageWrite [[H]] [[zero]] [[V]]
  %2 = tail call noundef nonnull align 4 dereferenceable(4) ptr @llvm.spv.resource.getpointer.p0.tspirv.Image_i32_5_2_0_0_2_0t(target("spirv.Image", i32, 5, 2, 0, 0, 2, 0) %s_h.i, i32 0)
  store <2 x i32> %1, ptr %2, align 4
; CHECK: OpBranch [[bb_both:%[0-9]+]]
  br label %bb_both

; CHECK: [[bb_both]] = OpLabel
bb_both:
; CHECK: [[H:%[0-9]+]] = OpLoad [[ImageType]] [[Var]]
; CHECK: [[R:%[0-9]+]] = OpImageRead [[v4int]] [[H]] [[twenty_three]]
; CHECK: [[E0:%[0-9]+]] = OpCompositeExtract [[uint]] [[R]] 0
; CHECK: [[E1:%[0-9]+]] = OpCompositeExtract [[uint]] [[R]] 1
; CHECK: [[V:%[0-9]+]] = OpCompositeConstruct [[v2int]] [[E0]] [[E1]]
  %3 = tail call noundef nonnull align 4 dereferenceable(4) ptr @llvm.spv.resource.getpointer.p0.tspirv.Image_i32_5_2_0_0_2_0t(target("spirv.Image", i32, 5, 2, 0, 0, 2, 0) %s_h.i, i32 23)
  %4 = load <2 x i32>, ptr %3, align 4

; CHECK: [[H:%[0-9]+]] = OpLoad [[ImageType]] [[Var]]
; CHECK: OpImageWrite [[H]] [[twenty]] [[V]]
  %5 = tail call noundef nonnull align 4 dereferenceable(4) ptr @llvm.spv.resource.getpointer.p0.tspirv.Image_i32_5_2_0_0_2_0t(target("spirv.Image", i32, 5, 2, 0, 0, 2, 0) %s_h.i, i32 20)
  store <2 x i32> %4, ptr %5, align 4
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, inaccessiblemem: none)
; CHECK: OpFunction
define void @main_vector4() local_unnamed_addr #0 {
entry:
; CHECK: [[H:%[0-9]+]] = OpLoad [[ImageType]] [[Var]]
  %s_h.i = tail call target("spirv.Image", i32, 5, 2, 0, 0, 2, 0) @llvm.spv.resource.handlefrombinding.tspirv.Image_i32_5_2_0_0_2_0t(i32 3, i32 5, i32 1, i32 0, ptr nonnull @.str.b0)

; CHECK: [[R:%[0-9]+]] = OpImageRead [[v4int]] [[H]] [[one]]
  %0 = tail call noundef nonnull align 4 dereferenceable(4) ptr @llvm.spv.resource.getpointer.p0.tspirv.Image_i32_5_2_0_0_2_0t(target("spirv.Image", i32, 5, 2, 0, 0, 2, 0) %s_h.i, i32 1)
  %1 = load <4 x i32>, ptr %0, align 4
; CHECK: OpBranch [[bb_store:%[0-9]+]]
  br label %bb_store

; CHECK: [[bb_store]] = OpLabel
bb_store:

; CHECK: [[H:%[0-9]+]] = OpLoad [[ImageType]] [[Var]]
; CHECK: OpImageWrite [[H]] [[zero]] [[R]]
  %2 = tail call noundef nonnull align 4 dereferenceable(4) ptr @llvm.spv.resource.getpointer.p0.tspirv.Image_i32_5_2_0_0_2_0t(target("spirv.Image", i32, 5, 2, 0, 0, 2, 0) %s_h.i, i32 0)
  store <4 x i32> %1, ptr %2, align 4
; CHECK: OpBranch [[bb_both:%[0-9]+]]
  br label %bb_both

; CHECK: [[bb_both]] = OpLabel
bb_both:
; CHECK: [[H:%[0-9]+]] = OpLoad [[ImageType]] [[Var]]
; CHECK: [[R:%[0-9]+]] = OpImageRead [[v4int]] [[H]] [[twenty_three]]
  %3 = tail call noundef nonnull align 4 dereferenceable(4) ptr @llvm.spv.resource.getpointer.p0.tspirv.Image_i32_5_2_0_0_2_0t(target("spirv.Image", i32, 5, 2, 0, 0, 2, 0) %s_h.i, i32 23)
  %4 = load <4 x i32>, ptr %3, align 4

; CHECK: [[H:%[0-9]+]] = OpLoad [[ImageType]] [[Var]]
; CHECK: OpImageWrite [[H]] [[twenty]] [[R]]
  %5 = tail call noundef nonnull align 4 dereferenceable(4) ptr @llvm.spv.resource.getpointer.p0.tspirv.Image_i32_5_2_0_0_2_0t(target("spirv.Image", i32, 5, 2, 0, 0, 2, 0) %s_h.i, i32 20)
  store <4 x i32> %4, ptr %5, align 4
  ret void
}

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, inaccessiblemem: none) "frame-pointer"="all" "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(none) }
