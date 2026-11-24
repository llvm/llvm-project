; RUN: llc -O0 -verify-machineinstrs %s -o - | FileCheck %s --check-prefix=TYPED --implicit-check-not=Untyped --implicit-check-not=OpTypeVectorIdEXT
; RUN: llc -O0 -verify-machineinstrs -spirv-ext=+SPV_KHR_untyped_pointers %s -o - | FileCheck %s --check-prefix=SCALAR
; RUN: llc -O0 -verify-machineinstrs -spirv-ext=+SPV_KHR_untyped_pointers,+SPV_EXT_long_vector %s -o - | FileCheck %s --check-prefix=LONG
; RUN: %if spirv-tools %{ llc -O0 %s -filetype=obj -o %t.typed.spv && spirv-val --target-env vulkan1.3 %t.typed.spv %}
; RUN: %if spirv-tools %{ llc -O0 -spirv-ext=+SPV_KHR_untyped_pointers %s -filetype=obj -o %t.spv && spirv-val --target-env vulkan1.3 %t.spv %}
; RUN: %if spirv-tools %{ llc -O0 -spirv-ext=+SPV_KHR_untyped_pointers,+SPV_EXT_long_vector %s -filetype=obj -o %t.long.spv && spirv-val --target-env vulkan1.3 %t.long.spv %}

; With long vectors and untyped pointers enabled, we can emit one load/store per-access.
; With only untyped, the long vector should be split, in this case it's 8 scalar loads then stores.
; With neither, currently the long vector is copied a byte at a time, because of the i8[] resource
; definition.

target triple = "spirv1.6-unknown-vulkan1.3-compute"

; TYPED: %[[#UCHAR:]] = OpTypeInt 8 0
; TYPED-LABEL: Begin function main
; TYPED-COUNT-32: OpLoad %[[#UCHAR]]
; TYPED-COUNT-32: OpStore
; TYPED: OpFunctionEnd

; SCALAR: %[[#UINT:]] = OpTypeInt 32 0
; SCALAR-LABEL: Begin function main
; SCALAR-COUNT-8: OpLoad %[[#UINT]] {{.*}} Aligned 4
; SCALAR-COUNT-8: OpStore {{.*}} Aligned 4
; SCALAR: OpFunctionEnd

; LONG-DAG: OpCapability LongVectorEXT
; LONG-DAG: %[[#UINT:]] = OpTypeInt 32 0
; LONG-DAG: %[[#EIGHT:]] = OpConstant %[[#UINT]] 8{{$}}
; LONG-DAG: %[[#VEC8:]] = OpTypeVectorIdEXT %[[#UINT]] %[[#EIGHT]]
; LONG-LABEL: Begin function main
; LONG: %[[#VALUE:]] = OpLoad %[[#VEC8]] {{%[0-9]+}} Aligned 4
; LONG: OpStore {{%[0-9]+}} %[[#VALUE]] Aligned 4
; LONG: OpFunctionEnd

@name = private unnamed_addr constant [4 x i8] c"buf\00"

define void @main() #0 {
  %handle = call target("spirv.VulkanBuffer", [0 x i8], 12, 1) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr @name)
  %src = call ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x i8], 12, 1) %handle, i32 4)
  %value = load <8 x i32>, ptr addrspace(11) %src, align 4
  %dst = call ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x i8], 12, 1) %handle, i32 36)
  store <8 x i32> %value, ptr addrspace(11) %dst, align 4
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
