; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-vulkan-compute %s -o - | FileCheck %s --check-prefixes=CHECK,TYPED --implicit-check-not=Untyped
; RUN: llc -O2 -verify-machineinstrs -mtriple=spirv-unknown-vulkan-compute %s -o - | FileCheck %s --check-prefixes=CHECK,TYPED --implicit-check-not=Untyped
; RUN: llc -O0 -enable-new-pm -verify-machineinstrs -mtriple=spirv-unknown-vulkan-compute %s -o - | FileCheck %s --check-prefixes=CHECK,TYPED --implicit-check-not=Untyped
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -filetype=obj -o %t.typed.spv && spirv-val --target-env vulkan1.3 %t.typed.spv %}
; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-vulkan-compute -spirv-ext=+SPV_KHR_untyped_pointers %s -o - | FileCheck %s --check-prefixes=CHECK,UNTYPED --implicit-check-not=Aliased
; RUN: llc -O2 -verify-machineinstrs -mtriple=spirv-unknown-vulkan-compute -spirv-ext=+SPV_KHR_untyped_pointers %s -o - | FileCheck %s --check-prefixes=CHECK,UNTYPED --implicit-check-not=Aliased
; RUN: llc -O0 -enable-new-pm -verify-machineinstrs -mtriple=spirv-unknown-vulkan-compute -spirv-ext=+SPV_KHR_untyped_pointers %s -o - | FileCheck %s --check-prefixes=CHECK,UNTYPED --implicit-check-not=Aliased
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute -spirv-ext=+SPV_KHR_untyped_pointers %s -filetype=obj -o %t.spv && spirv-val --target-env vulkan1.3 %t.spv %}

; Check lowering of raw buffer intrinsics to SPIRV. The resource, which can be reinterpreted to load/store
; multiple types, should become multiple aliasing resources depending on which types are used.
; If untyped pointers are supported, we can skip this step and do the reinterpretation with untyped access
; chains.
; Assuming long vectors are disabled, we must break up long vector load/stores regardless of untyped pointer
; support

; UNTYPED: OpCapability UntypedPointersKHR
; CHECK-DAG: %[[#UINT:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#FLOAT:]] = OpTypeFloat 32
; TYPED-DAG: OpTypeRuntimeArray %[[#FLOAT]]
; TYPED-DAG: OpName %[[#BUF_I32:]] ".resource.alias.0.0.i32.rw.buf"
; TYPED-DAG: OpDecorate %[[#BUF_I32]] Aliased
; UNTYPED-DAG: %[[#FLOAT4:]] = OpTypeVector %[[#FLOAT]] 4

@in = private unnamed_addr constant [3 x i8] c"in\00"
@out = private unnamed_addr constant [4 x i8] c"out\00"

@name = private unnamed_addr constant [4 x i8] c"buf\00"
@array_name = private unnamed_addr constant [6 x i8] c"array\00"

; CHECK-LABEL: Begin function increment
; TYPED: OpLoad %[[#UINT]]
; TYPED: OpIAdd %[[#UINT]]
; TYPED: OpStore
; UNTYPED: %[[#VALUE:]] = OpLoad %[[#UINT]] {{%[0-9]+}} Aligned 4
; UNTYPED: %[[#NEXT:]] = OpIAdd %[[#UINT]] %[[#VALUE]]
; UNTYPED: OpStore {{%[0-9]+}} %[[#NEXT]] Aligned 4
; CHECK: OpFunctionEnd
define void @increment() #0 {
  %handle = call target("spirv.VulkanBuffer", [0 x i8], 12, 1) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr @name)
  %value = call i32 @llvm.spv.resource.load.rawbuffer(target("spirv.VulkanBuffer", [0 x i8], 12, 1) %handle, i32 0, i32 4)
  %next = add i32 %value, 1
  call void @llvm.spv.resource.store.rawbuffer(target("spirv.VulkanBuffer", [0 x i8], 12, 1) %handle, i32 0, i32 %next, i32 4)
  ret void
}

; CHECK-LABEL: Begin function vector_copy
; TYPED-COUNT-4: OpLoad %[[#FLOAT]]
; TYPED-COUNT-4: OpStore
; UNTYPED: %[[#VECTOR:]] = OpLoad %[[#FLOAT4]] {{%[0-9]+}} Aligned 4
; UNTYPED: OpStore {{%[0-9]+}} %[[#VECTOR]] Aligned 4
; CHECK: OpFunctionEnd
define void @vector_copy() #0 {
  %tid = call i32 @llvm.spv.thread.id.i32(i32 0)
  %offset = shl i32 %tid, 2
  %handle = call target("spirv.VulkanBuffer", [0 x i8], 12, 1) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr @name)
  %value = call <4 x float> @llvm.spv.resource.load.rawbuffer(target("spirv.VulkanBuffer", [0 x i8], 12, 1) %handle, i32 %offset, i32 4)
  %next = add i32 %offset, 16
  call void @llvm.spv.resource.store.rawbuffer(target("spirv.VulkanBuffer", [0 x i8], 12, 1) %handle, i32 %next, <4 x float> %value, i32 4)
  ret void
}

; CHECK-LABEL: Begin function descriptor_array
; TYPED: OpLoad %[[#UINT]]
; TYPED: OpStore
; UNTYPED: OpAccessChain
; UNTYPED: %[[#WORD:]] = OpLoad %[[#UINT]] {{%[0-9]+}} Aligned 4
; UNTYPED: %[[#UPDATED:]] = OpIAdd %[[#UINT]] %[[#WORD]]
; UNTYPED: OpStore {{%[0-9]+}} %[[#UPDATED]] Aligned 4
; CHECK: OpFunctionEnd
define void @descriptor_array() #0 {
  %tid = call i32 @llvm.spv.thread.id.i32(i32 0)
  %index = and i32 %tid, 1
  %handle = call target("spirv.VulkanBuffer", [0 x i8], 12, 1) @llvm.spv.resource.handlefrombinding(i32 0, i32 1, i32 2, i32 %index, ptr @array_name)
  %value = call i32 @llvm.spv.resource.load.rawbuffer(target("spirv.VulkanBuffer", [0 x i8], 12, 1) %handle, i32 4, i32 4)
  %next = add i32 %value, 1
  call void @llvm.spv.resource.store.rawbuffer(target("spirv.VulkanBuffer", [0 x i8], 12, 1) %handle, i32 4, i32 %next, i32 4)
  ret void
}

; CHECK-LABEL: Begin function vector_copy_five
; TYPED-COUNT-5: OpLoad %[[#FLOAT]]
; TYPED-COUNT-5: OpStore
; UNTYPED-COUNT-5: OpLoad %[[#FLOAT]] {{.*}} Aligned 4
; UNTYPED-COUNT-5: OpStore {{.*}} Aligned 4
; CHECK: OpFunctionEnd
define void @vector_copy_five() #0 {
  %handle = call target("spirv.VulkanBuffer", [0 x i8], 12, 1) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr @name)
  %value = call <5 x float> @llvm.spv.resource.load.rawbuffer(target("spirv.VulkanBuffer", [0 x i8], 12, 1) %handle, i32 0, i32 4)
  call void @llvm.spv.resource.store.rawbuffer(target("spirv.VulkanBuffer", [0 x i8], 12, 1) %handle, i32 32, <5 x float> %value, i32 4)
  ret void
}

; CHECK-LABEL: Begin function vector_copy_eight
; TYPED-COUNT-8: OpLoad %[[#UINT]]
; TYPED-COUNT-8: OpStore
; UNTYPED-COUNT-8: OpLoad %[[#UINT]] {{.*}} Aligned 4
; UNTYPED-COUNT-8: OpStore {{.*}} Aligned 4
; CHECK: OpFunctionEnd
define void @vector_copy_eight() #0 {
  %tid = call i32 @llvm.spv.thread.id.i32(i32 0)
  %offset = shl i32 %tid, 2
  %handle = call target("spirv.VulkanBuffer", [0 x i8], 12, 1) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr @name)
  %value = call <8 x i32> @llvm.spv.resource.load.rawbuffer(target("spirv.VulkanBuffer", [0 x i8], 12, 1) %handle, i32 %offset, i32 4)
  %next = add i32 %offset, 32
  call void @llvm.spv.resource.store.rawbuffer(target("spirv.VulkanBuffer", [0 x i8], 12, 1) %handle, i32 %next, <8 x i32> %value, i32 4)
  ret void
}

; CHECK-LABEL: Begin function copy_readonly
; TYPED-COUNT-4: OpLoad %[[#FLOAT]]
; TYPED-COUNT-4: OpStore
; UNTYPED: %[[#VECTOR:]] = OpLoad %[[#FLOAT4]] {{%[0-9]+}} Aligned 4
; UNTYPED: OpStore {{%[0-9]+}} %[[#VECTOR]] Aligned 4
; CHECK: OpFunctionEnd
define void @copy_readonly() #0 {
  %input = call target("spirv.VulkanBuffer", [0 x i8], 12, 0) @llvm.spv.resource.handlefrombinding(i32 0, i32 2, i32 1, i32 0, ptr @in)
  %output = call target("spirv.VulkanBuffer", [0 x i8], 12, 1) @llvm.spv.resource.handlefrombinding(i32 0, i32 3, i32 1, i32 0, ptr @out)
  %v = call <4 x float> @llvm.spv.resource.load.rawbuffer(target("spirv.VulkanBuffer", [0 x i8], 12, 0) %input, i32 4, i32 4)
  call void @llvm.spv.resource.store.rawbuffer(target("spirv.VulkanBuffer", [0 x i8], 12, 1) %output, i32 8, <4 x float> %v, i32 4)
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
