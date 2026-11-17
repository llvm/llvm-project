; RUN: opt -S -passes=spirv-remove-unused-resources %s -o - | FileCheck %s

; For reference, this test contains the IR for the following HLSL
; after selecting frag_ep1 and just after spirv-finalize-shader-linkage
;
; struct Data { float3 color; };
; ConstantBuffer<Data> data;
;
; float4 frag_ep1() : SV_TARGET {
;   return float4(data.color, 1.0);
; }
;
; float4 frag_ep2() : SV_TARGET {
;   return float4(data.color + 0.2, 1.0);
; }
;
; CHECK-NOT: @_ZL4data.0 =
; CHECK-LABEL: define void @frag_ep1()
; CHECK: %handle = call target("spirv.VulkanBuffer", %Data, 2, 0) @llvm.spv.resource.handlefromimplicitbinding
; CHECK-NOT: store target("spirv.VulkanBuffer", %Data, 2, 0) %handle, ptr @_ZL4data.0
; CHECK: %base = call ptr addrspace(12) @llvm.spv.resource.getbasepointer{{.*}}(target("spirv.VulkanBuffer", %Data, 2, 0) %handle)
; CHECK: %color = load <3 x float>, ptr addrspace(12) %base
; CHECK: %extended = shufflevector <3 x float> %color, <3 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 poison>
; CHECK: %result = insertelement <4 x float> %extended, float 1.000000e+00, i64 3
; CHECK: store <4 x float> %result, ptr addrspace(8) @SV_TARGET0
; CHECK-NOT: @_ZL4data.0

target triple = "spirv1.6-unknown-vulkan1.3-pixel"

%Data = type <{ <3 x float> }>

@_ZL4data.0 = internal unnamed_addr global target("spirv.VulkanBuffer", %Data, 2, 0) poison, align 8
@.str = private unnamed_addr constant [5 x i8] c"data\00", align 1
@SV_TARGET0 = external hidden thread_local local_unnamed_addr addrspace(8) global <4 x float>

define void @frag_ep1() #0 {
entry:
  %handle = call target("spirv.VulkanBuffer", %Data, 2, 0) @llvm.spv.resource.handlefromimplicitbinding.tspirv.VulkanBuffer_s_Datas_2_0t(i32 0, i32 0, i32 1, i32 0, ptr @.str)
  store target("spirv.VulkanBuffer", %Data, 2, 0) %handle, ptr @_ZL4data.0, align 8
  %base = call ptr addrspace(12) @llvm.spv.resource.getbasepointer.p12.tspirv.VulkanBuffer_s_Datas_2_0t(target("spirv.VulkanBuffer", %Data, 2, 0) %handle)
  %color = load <3 x float>, ptr addrspace(12) %base, align 4
  %extended = shufflevector <3 x float> %color, <3 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 poison>
  %result = insertelement <4 x float> %extended, float 1.000000e+00, i64 3
  store <4 x float> %result, ptr addrspace(8) @SV_TARGET0, align 4
  ret void
}

declare target("spirv.VulkanBuffer", %Data, 2, 0) @llvm.spv.resource.handlefromimplicitbinding.tspirv.VulkanBuffer_s_Datas_2_0t(i32, i32, i32, i32, ptr)
declare ptr addrspace(12) @llvm.spv.resource.getbasepointer.p12.tspirv.VulkanBuffer_s_Datas_2_0t(target("spirv.VulkanBuffer", %Data, 2, 0))

attributes #0 = { "hlsl.shader"="pixel" }
