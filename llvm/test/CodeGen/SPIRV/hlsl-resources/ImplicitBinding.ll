; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv1.6-vulkan1.3-library %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-vulkan1.3-library %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

@.str = private unnamed_addr constant [2 x i8] c"b\00", align 1
@.str.2 = private unnamed_addr constant [2 x i8] c"c\00", align 1
@.str.4 = private unnamed_addr constant [2 x i8] c"d\00", align 1
@.str.6 = private unnamed_addr constant [2 x i8] c"e\00", align 1
@.str.8 = private unnamed_addr constant [2 x i8] c"f\00", align 1
@.str.10 = private unnamed_addr constant [2 x i8] c"g\00", align 1
@.str.12 = private unnamed_addr constant [2 x i8] c"h\00", align 1
@.str.14 = private unnamed_addr constant [2 x i8] c"i\00", align 1

; CHECK-DAG: OpName [[b:%[0-9]+]] "b"
; CHECK-DAG: OpName [[c:%[0-9]+]] "c"
; CHECK-DAG: OpName [[d:%[0-9]+]] "d"
; CHECK-DAG: OpName [[e:%[0-9]+]] "e"
; CHECK-DAG: OpName [[f:%[0-9]+]] "f"
; CHECK-DAG: OpName [[g:%[0-9]+]] "g"
; CHECK-DAG: OpName [[h:%[0-9]+]] "h"
; CHECK-DAG: OpName [[i:%[0-9]+]] "i"
; CHECK-DAG: OpDecorate [[b]] DescriptorSet 0
; CHECK-DAG: OpDecorate [[b]] Binding 1
; CHECK-DAG: OpDecorate [[c]] DescriptorSet 0
; CHECK-DAG: OpDecorate [[c]] Binding 0
; CHECK-DAG: OpDecorate [[d]] DescriptorSet 0
; CHECK-DAG: OpDecorate [[d]] Binding 3
; CHECK-DAG: OpDecorate [[e]] DescriptorSet 0
; CHECK-DAG: OpDecorate [[e]] Binding 2
; CHECK-DAG: OpDecorate [[f]] DescriptorSet 10
; CHECK-DAG: OpDecorate [[f]] Binding 1
; CHECK-DAG: OpDecorate [[g]] DescriptorSet 10
; CHECK-DAG: OpDecorate [[g]] Binding 0
; CHECK-DAG: OpDecorate [[h]] DescriptorSet 10
; CHECK-DAG: OpDecorate [[h]] Binding 3
; CHECK-DAG: OpDecorate [[i]] DescriptorSet 10
; CHECK-DAG: OpDecorate [[i]] Binding 2


define void @main() local_unnamed_addr #0 {
entry:
  %0 = tail call target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 0) @llvm.spv.resource.handlefromimplicitbinding.tspirv.SignedImage_i32_5_2_0_0_2_0t(i32 0, i32 0, i32 1, i32 0, i1 false, ptr nonnull @.str)
  %1 = tail call target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 0) @llvm.spv.resource.handlefrombinding.tspirv.SignedImage_i32_5_2_0_0_2_0t(i32 0, i32 0, i32 1, i32 0, i1 false, ptr nonnull @.str.2)
  %2 = tail call target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 0) @llvm.spv.resource.handlefromimplicitbinding.tspirv.SignedImage_i32_5_2_0_0_2_0t(i32 1, i32 0, i32 1, i32 0, i1 false, ptr nonnull @.str.4)
  %3 = tail call target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 0) @llvm.spv.resource.handlefrombinding.tspirv.SignedImage_i32_5_2_0_0_2_0t(i32 0, i32 2, i32 1, i32 0, i1 false, ptr nonnull @.str.6)
  %4 = tail call target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 0) @llvm.spv.resource.handlefrombinding.tspirv.SignedImage_i32_5_2_0_0_2_0t(i32 10, i32 1, i32 1, i32 0, i1 false, ptr nonnull @.str.8)
  %5 = tail call target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 0) @llvm.spv.resource.handlefromimplicitbinding.tspirv.SignedImage_i32_5_2_0_0_2_0t(i32 2, i32 10, i32 1, i32 0, i1 false, ptr nonnull @.str.10)
  %6 = tail call target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 0) @llvm.spv.resource.handlefromimplicitbinding.tspirv.SignedImage_i32_5_2_0_0_2_0t(i32 3, i32 10, i32 1, i32 0, i1 false, ptr nonnull @.str.12)
  %7 = tail call target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 0) @llvm.spv.resource.handlefrombinding.tspirv.SignedImage_i32_5_2_0_0_2_0t(i32 10, i32 2, i32 1, i32 0, i1 false, ptr nonnull @.str.14)
  %8 = tail call noundef align 4 dereferenceable(4) ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.SignedImage_i32_5_2_0_0_2_0t(target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 0) %1, i32 0)
  %9 = load i32, ptr addrspace(11) %8, align 4
  %10 = tail call noundef align 4 dereferenceable(4) ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.SignedImage_i32_5_2_0_0_2_0t(target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 0) %2, i32 0)
  %11 = load i32, ptr addrspace(11) %10, align 4
  %add.i = add nsw i32 %11, %9
  %12 = tail call noundef align 4 dereferenceable(4) ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.SignedImage_i32_5_2_0_0_2_0t(target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 0) %3, i32 0)
  %13 = load i32, ptr addrspace(11) %12, align 4
  %add4.i = add nsw i32 %add.i, %13
  %14 = tail call noundef align 4 dereferenceable(4) ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.SignedImage_i32_5_2_0_0_2_0t(target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 0) %4, i32 0)
  %15 = load i32, ptr addrspace(11) %14, align 4
  %add6.i = add nsw i32 %add4.i, %15
  %16 = tail call noundef align 4 dereferenceable(4) ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.SignedImage_i32_5_2_0_0_2_0t(target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 0) %5, i32 0)
  %17 = load i32, ptr addrspace(11) %16, align 4
  %add8.i = add nsw i32 %add6.i, %17
  %18 = tail call noundef align 4 dereferenceable(4) ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.SignedImage_i32_5_2_0_0_2_0t(target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 0) %6, i32 0)
  %19 = load i32, ptr addrspace(11) %18, align 4
  %add10.i = add nsw i32 %add8.i, %19
  %20 = tail call noundef align 4 dereferenceable(4) ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.SignedImage_i32_5_2_0_0_2_0t(target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 0) %7, i32 0)
  %21 = load i32, ptr addrspace(11) %20, align 4
  %add12.i = add nsw i32 %add10.i, %21
  %22 = tail call noundef align 4 dereferenceable(4) ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.SignedImage_i32_5_2_0_0_2_0t(target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 0) %0, i32 0)
  store i32 %add12.i, ptr addrspace(11) %22, align 4
  ret void
}


attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }