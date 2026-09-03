; RUN: opt -S -passes=spirv-legalize-implicit-binding -mtriple=spirv1.6-vulkan1.3-library < %s | FileCheck %s

@.str.b = private unnamed_addr constant [2 x i8] c"b\00", align 1
@.str.c = private unnamed_addr constant [2 x i8] c"c\00", align 1

; Verify implicit-binding intrinsic calls are rewritten to explicit
; handlefrombinding intrinsic calls; the descriptor set is preserved
; (1st operand of the original) and a new binding number is assigned.

define void @main() local_unnamed_addr #0 {
entry:
; CHECK-LABEL: define void @main(
; CHECK: call target({{.*}}) @llvm.spv.resource.handlefrombinding{{.*}}(i32 0, i32 0, i32 1, i32 0, ptr {{.*}}@.str.b)
; CHECK: call target({{.*}}) @llvm.spv.resource.handlefrombinding{{.*}}(i32 1, i32 0, i32 1, i32 0, ptr {{.*}}@.str.c)
  %0 = tail call target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 0) @llvm.spv.resource.handlefromimplicitbinding.tspirv.SignedImage_i32_5_2_0_0_2_0t(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.str.b)
  %1 = tail call target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 0) @llvm.spv.resource.handlefromimplicitbinding.tspirv.SignedImage_i32_5_2_0_0_2_0t(i32 1, i32 1, i32 1, i32 0, ptr nonnull @.str.c)
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
