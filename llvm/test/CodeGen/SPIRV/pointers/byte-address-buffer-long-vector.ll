; RUN: llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; CHECK-DAG: %[[GLSLEXT:[0-9]+]] = OpExtInstImport "GLSL.std.450"
; CHECK-DAG: %[[FLOAT:[0-9]+]] = OpTypeFloat 32
; CHECK-DAG: %[[V4FLOAT:[0-9]+]] = OpTypeVector %[[FLOAT]] 4
; CHECK-LABEL: %[[#]] = OpFunction %[[#]] DontInline %[[#]] ; -- Begin function main
; CHECK: OpBitcast %[[FLOAT]]
; CHECK-COUNT-4: OpCompositeConstruct %[[V4FLOAT]]
; CHECK-COUNT-4: OpExtInst %[[V4FLOAT]] %[[GLSLEXT]] {{Cosh}}


@.str = private unnamed_addr constant [4 x i8] c"In0\00", align 1
@.str.2 = private unnamed_addr constant [4 x i8] c"Out\00", align 1

declare token @llvm.experimental.convergence.entry()

define void @main() local_unnamed_addr #0 {
entry:
  %convergence = tail call token @llvm.experimental.convergence.entry()
  %input = tail call target("spirv.VulkanBuffer", [0 x i8], 12, 0) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.str)
  %output = tail call target("spirv.VulkanBuffer", [0 x i8], 12, 1) @llvm.spv.resource.handlefrombinding(i32 0, i32 1, i32 1, i32 0, ptr nonnull @.str.2)
  %input.ptr = call ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x i8], 12, 0) %input, i32 0) [ "convergencectrl"(token %convergence) ]
  %value = load <16 x float>, ptr addrspace(11) %input.ptr, align 4
  %result = tail call <16 x float> @llvm.cosh.v16f32(<16 x float> %value)
  %output.ptr = call ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x i8], 12, 1) %output, i32 0) [ "convergencectrl"(token %convergence) ]
  store <16 x float> %result, ptr addrspace(11) %output.ptr, align 4
  ret void
}


attributes #0 = { convergent noinline norecurse "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
