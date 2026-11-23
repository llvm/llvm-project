; RUN: opt -S -passes=spirv-remove-unused-resources %s -o - | FileCheck %s
;
; When using entry point main, eliminate the unused buffer handle and
; it's associated counter handle
; Original HLSL:
;
; RWStructuredBuffer<float> unused;
; RWBuffer<float> used;
;
; [numthreads(1, 1, 1)]
; void main() {
;   used[0] = 1.0;
; }
;
; [numthreads(1, 1, 1)]
; void other() {
;   unused[unused.IncrementCounter()] = 2.0;
; }

; CHECK-NOT: @_ZL6unused.0
; CHECK-NOT: @_ZL6unused.1
; CHECK-NOT: @.str =
; CHECK-LABEL: define void @main()
; CHECK-NOT: call target("spirv.VulkanBuffer",
; CHECK: %[[USED:.*]] = call target("spirv.Image", {{.*}}) @llvm.spv.resource.handlefromimplicitbinding{{.*}}(i32 2, i32 0, i32 1, i32 0, ptr @.str.2)
; CHECK: @llvm.spv.resource.getpointer{{.*}}(target("spirv.Image", {{.*}}) %[[USED]], i32 0)
; CHECK: store float 1.000000e+00
; CHECK: ret void

target datalayout = "e-ve-i64:64-n8:16:32:64-G10"
target triple = "spirv1.6-unknown-vulkan1.3-compute"

@_ZL6unused.0 = internal unnamed_addr global target("spirv.VulkanBuffer", [0 x float], 12, 1) poison, align 8
@_ZL6unused.1 = internal unnamed_addr global target("spirv.VulkanBuffer", i32, 12, 1) poison, align 8
@.str = private unnamed_addr constant [7 x i8] c"unused\00", align 1
@.str.2 = private unnamed_addr constant [5 x i8] c"used\00", align 1

define void @main() {
entry:
  %buffer = call target("spirv.VulkanBuffer", [0 x float], 12, 1) @llvm.spv.resource.handlefromimplicitbinding.tspirv.VulkanBuffer_a0f32_12_1t(i32 0, i32 0, i32 1, i32 0, ptr @.str)
  %counter = call target("spirv.VulkanBuffer", i32, 12, 1) @llvm.spv.resource.counterhandlefromimplicitbinding.tspirv.VulkanBuffer_i32_12_1t.tspirv.VulkanBuffer_a0f32_12_1t(target("spirv.VulkanBuffer", [0 x float], 12, 1) %buffer, i32 1, i32 0)
  store target("spirv.VulkanBuffer", [0 x float], 12, 1) %buffer, ptr @_ZL6unused.0, align 8
  store target("spirv.VulkanBuffer", i32, 12, 1) %counter, ptr @_ZL6unused.1, align 8
  %used = call target("spirv.Image", float, 5, 2, 0, 0, 2, 3) @llvm.spv.resource.handlefromimplicitbinding.tspirv.Image_f32_5_2_0_0_2_3t(i32 2, i32 0, i32 1, i32 0, ptr @.str.2)
  %ptr = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.Image_f32_5_2_0_0_2_3t.i32(target("spirv.Image", float, 5, 2, 0, 0, 2, 3) %used, i32 0)
  store float 1.000000e+00, ptr addrspace(11) %ptr, align 4
  ret void
}
