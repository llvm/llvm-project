; RUN: opt -S -passes=spirv-remove-unused-resources %s -o - | FileCheck %s

; Make sure the handle/global survive the removal pass since they're actually used
; Original HLSL:

; struct Data { float value; };
; ConstantBuffer<Data> data;
;
; __attribute__((noinline)) float readData() {
;   return data.value;
; }
;
; float main() : SV_Target {
;   return readData();
; }

; CHECK: @_ZL4data.0 = internal {{.*}}global target("spirv.VulkanBuffer", %Data, 2, 0)
; CHECK: @.str = private unnamed_addr constant [5 x i8] c"data\00"
; CHECK-LABEL: define internal {{.*}}@_Z8readDatav()
; CHECK: load target("spirv.VulkanBuffer", %Data, 2, 0), ptr @_ZL4data.0
; CHECK-LABEL: define void @main()
; CHECK: %[[HANDLE:.*]] = call target("spirv.VulkanBuffer", %Data, 2, 0) @llvm.spv.resource.handlefromimplicitbinding
; CHECK-NEXT: store target("spirv.VulkanBuffer", %Data, 2, 0) %[[HANDLE]], ptr @_ZL4data.0

target datalayout = "e-ve-i64:64-n8:16:32:64-G10"
target triple = "spirv1.6-unknown-vulkan1.3-pixel"

%Data = type <{ float }>

@_ZL4data.0 = internal unnamed_addr global target("spirv.VulkanBuffer", %Data, 2, 0) poison, align 8
@.str = private unnamed_addr constant [5 x i8] c"data\00", align 1
@SV_Target0 = external hidden thread_local local_unnamed_addr addrspace(8) global float

define internal spir_func float @_Z8readDatav() noinline {
entry:
  %handle = load target("spirv.VulkanBuffer", %Data, 2, 0), ptr @_ZL4data.0, align 8
  %ptr = call ptr addrspace(12) @llvm.spv.resource.getbasepointer.p12.tspirv.VulkanBuffer_s_Datas_2_0t(target("spirv.VulkanBuffer", %Data, 2, 0) %handle)
  %value = load float, ptr addrspace(12) %ptr, align 4
  ret float %value
}

define void @main() {
entry:
  %handle = call target("spirv.VulkanBuffer", %Data, 2, 0) @llvm.spv.resource.handlefromimplicitbinding.tspirv.VulkanBuffer_s_Datas_2_0t(i32 0, i32 0, i32 1, i32 0, ptr @.str)
  store target("spirv.VulkanBuffer", %Data, 2, 0) %handle, ptr @_ZL4data.0, align 8
  %value = call spir_func float @_Z8readDatav()
  store float %value, ptr addrspace(8) @SV_Target0, align 4
  ret void
}
