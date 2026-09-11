; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv1.6-vulkan1.3-library %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-vulkan1.3-library %s -o - -filetype=obj | spirv-val %}

; Verify that unbounded arrays of structured buffers and their associated
; counters enable RuntimeDescriptorArrayEXT.

; CHECK-DAG: OpCapability RuntimeDescriptorArrayEXT
; CHECK-DAG: %[[FLOAT:[0-9]+]] = OpTypeFloat 32
; CHECK-DAG: %[[UINT:[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: %[[DATA_ARRAY:[0-9]+]] = OpTypeRuntimeArray %[[FLOAT]]
; CHECK-DAG: %[[BUFFER:[0-9]+]] = OpTypeStruct %[[DATA_ARRAY]]
; CHECK-DAG: OpTypeRuntimeArray %[[BUFFER]]
; CHECK-DAG: %[[COUNTER:[0-9]+]] = OpTypeStruct %[[UINT]]
; CHECK-DAG: OpTypeRuntimeArray %[[COUNTER]]

@Bufs.str = private unnamed_addr constant [5 x i8] c"Bufs\00", align 1

define void @main() #0 {
entry:
  %handle = call target("spirv.VulkanBuffer", [0 x float], 12, 1) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 0, i32 0, ptr @Bufs.str)
  %counter.handle = call target("spirv.VulkanBuffer", i32, 12, 1) @llvm.spv.resource.counterhandlefrombinding(target("spirv.VulkanBuffer", [0 x float], 12, 1) %handle, i32 0, i32 1)
  %counter = call i32 @llvm.spv.resource.updatecounter(target("spirv.VulkanBuffer", i32, 12, 1) %counter.handle, i8 1)
  %pointer = call ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x float], 12, 1) %handle, i32 %counter)
  store float 0.000000e+00, ptr addrspace(11) %pointer, align 4
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
