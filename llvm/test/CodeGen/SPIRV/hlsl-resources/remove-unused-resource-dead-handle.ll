; RUN: opt -S -passes=spirv-remove-unused-resources %s -o - | FileCheck %s

; Verify that the handle creation is removed when its only use was a store to
; an unused resource global.

; CHECK-NOT: @_ZL4data.0 =
; CHECK-LABEL: define void @main()
; CHECK-NOT: call target("spirv.VulkanBuffer", %Data, 2, 0) @llvm.spv.resource.handlefromimplicitbinding
; CHECK-NOT: store target("spirv.VulkanBuffer", %Data, 2, 0)
; CHECK: ret void

target triple = "spirv1.6-unknown-vulkan1.3-compute"

%Data = type <{ <3 x float> }>

@_ZL4data.0 = internal unnamed_addr global target("spirv.VulkanBuffer", %Data, 2, 0) poison, align 8

define void @main() #0 {
entry:
  %handle = call target("spirv.VulkanBuffer", %Data, 2, 0) @llvm.spv.resource.handlefromimplicitbinding.tspirv.VulkanBuffer_s_Datas_2_0t(i32 0, i32 0, i32 1, i32 0, ptr null)
  store target("spirv.VulkanBuffer", %Data, 2, 0) %handle, ptr @_ZL4data.0, align 8
  ret void
}

declare target("spirv.VulkanBuffer", %Data, 2, 0) @llvm.spv.resource.handlefromimplicitbinding.tspirv.VulkanBuffer_s_Datas_2_0t(i32, i32, i32, i32, ptr)

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
