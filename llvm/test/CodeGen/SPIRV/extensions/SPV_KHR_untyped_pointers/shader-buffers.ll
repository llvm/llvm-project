; RUN: llc -O0 -verify-machineinstrs -spirv-ext=+SPV_KHR_untyped_pointers %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -spirv-ext=+SPV_KHR_untyped_pointers %s -filetype=obj -o %t.spv && spirv-val --target-env vulkan1.3 %t.spv %}

target triple = "spirv1.6-unknown-vulkan1.3-compute"

; CHECK: OpCapability UntypedPointersKHR
; CHECK: OpExtension "SPV_KHR_untyped_pointers"
; CHECK-DAG: %[[#UINT:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#FLOAT:]] = OpTypeFloat 32
; CHECK-DAG: %[[#FLOAT4:]] = OpTypeVector %[[#FLOAT]] 4
; CHECK-DAG: %[[#PTR:]] = OpTypeUntypedPointerKHR StorageBuffer

%Record = type { <4 x i32>, <4 x float> }

@raw_name = private unnamed_addr constant [4 x i8] c"raw\00"
@structured_name = private unnamed_addr constant [11 x i8] c"structured\00"

; Raw buffer accesses preserve the requested vector type and scalar alignment.
; CHECK-LABEL: Begin function raw_buffer
; CHECK: %[[#SRC:]] = OpUntypedAccessChainKHR %[[#PTR]]
; CHECK: %[[#VALUE:]] = OpLoad %[[#FLOAT4]] %[[#SRC]] Aligned 4
; CHECK: %[[#DST:]] = OpUntypedAccessChainKHR %[[#PTR]]
; CHECK: OpStore %[[#DST]] %[[#VALUE]] Aligned 4
; CHECK: OpFunctionEnd
define void @raw_buffer() #0 {
  %handle = call target("spirv.VulkanBuffer", [0 x i8], 12, 1) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr @raw_name)
  %value = call <4 x float> @llvm.spv.resource.load.rawbuffer(target("spirv.VulkanBuffer", [0 x i8], 12, 1) %handle, i32 4, i32 4)
  call void @llvm.spv.resource.store.rawbuffer(target("spirv.VulkanBuffer", [0 x i8], 12, 1) %handle, i32 20, <4 x float> %value, i32 4)
  ret void
}

; Structured buffer member GEPs also use untyped access chains.
; CHECK-LABEL: Begin function structured_buffer
; CHECK: OpUntypedAccessChainKHR %[[#PTR]]
; CHECK: %[[#SRC:]] = OpUntypedInBoundsAccessChainKHR %[[#PTR]]
; CHECK: %[[#VALUE:]] = OpLoad %[[#FLOAT4]] %[[#SRC]] Aligned 4
; CHECK: OpUntypedAccessChainKHR %[[#PTR]]
; CHECK: %[[#DST:]] = OpUntypedInBoundsAccessChainKHR %[[#PTR]]
; CHECK: OpStore %[[#DST]] %[[#VALUE]] Aligned 4
; CHECK: OpFunctionEnd
define void @structured_buffer() #0 {
  %index = call i32 @llvm.spv.thread.id.i32(i32 0)
  %handle = call target("spirv.VulkanBuffer", [0 x %Record], 12, 1) @llvm.spv.resource.handlefrombinding(i32 0, i32 1, i32 1, i32 0, ptr @structured_name)
  %src_record = call ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x %Record], 12, 1) %handle, i32 %index)
  %src = getelementptr inbounds %Record, ptr addrspace(11) %src_record, i32 0, i32 1
  %value = load <4 x float>, ptr addrspace(11) %src, align 4
  %next = add i32 %index, 1
  %dst_record = call ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x %Record], 12, 1) %handle, i32 %next)
  %dst = getelementptr inbounds %Record, ptr addrspace(11) %dst_record, i32 0, i32 1
  store <4 x float> %value, ptr addrspace(11) %dst, align 4
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
