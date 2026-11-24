; RUN: llc -O0 -mtriple=spirv-unknown-vulkan1.3-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val %}

; Optimizations such as VectorCombine may widen the load of a scalar into the
; load of a vector when only the first element is used. SPIR-V cannot
; access a scalar as a vector, so the LegalizePtrCast legalization loads the
; scalar and inserts it into the first element of a vector whose other elements are undef

; Tests the same functionality as load-vector-from-scalar with a more real-world
; construction that includes the splat that would be involved in such a problem
; optimization.

; CHECK-DAG: %[[#float:]] = OpTypeFloat 32
; CHECK-DAG: %[[#v4:]] = OpTypeVector %[[#float]] 4
; CHECK-DAG: %[[#fptr:]] = OpTypePointer {{.*}} %[[#float]]
; CHECK-DAG: %[[#undef:]] = OpUndef %[[#v4]]

%S = type { <4 x float>, float, float }

@.src = private unnamed_addr constant [2 x i8] c"B\00"
@.out = private unnamed_addr constant [7 x i8] c"Output\00"

define void @main() #0 {
  %tok = call token @llvm.experimental.convergence.entry()
  %h = call target("spirv.VulkanBuffer", %S, 2, 0) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_s_Ss_2_0t(i32 0, i32 0, i32 1, i32 0, ptr @.src)
  %base = call noundef align 1 dereferenceable(32) ptr addrspace(12) @llvm.spv.resource.getbasepointer.p12.tspirv.VulkanBuffer_s_Ss_2_0t(target("spirv.VulkanBuffer", %S, 2, 0) %h) [ "convergencectrl"(token %tok) ]
  %p = getelementptr inbounds nuw i8, ptr addrspace(12) %base, i64 20
; CHECK: %[[#ptr:]] = OpInBoundsAccessChain %[[#fptr]] %[[#]] %[[#]]
  %l = load <4 x float>, ptr addrspace(12) %p, align 4
; CHECK: %[[#val:]] = OpLoad %[[#float]] %[[#ptr]]
; CHECK: %[[#ins:]] = OpCompositeInsert %[[#v4]] %[[#val]] %[[#undef]] 0
  %s = shufflevector <4 x float> %l, <4 x float> poison, <4 x i32> zeroinitializer
  %img = call target("spirv.Image", float, 1, 2, 0, 0, 2, 1) @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_1_2_0_0_2_1t(i32 0, i32 1, i32 1, i32 0, ptr @.out)
  %o = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.Image_f32_1_2_0_0_2_1t.v2i32(target("spirv.Image", float, 1, 2, 0, 0, 2, 1) %img, <2 x i32> splat (i32 2)) [ "convergencectrl"(token %tok) ]
  store <4 x float> %s, ptr addrspace(11) %o, align 4
  ret void
}

attributes #0 = { convergent "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
