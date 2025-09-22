; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-vulkan1.3-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-vulkan1.3-unknown %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; CHECK-DAG:        %[[#int:]] = OpTypeInt 32 0
; CHECK-DAG:        %[[#ptr_Input_int:]] = OpTypePointer Input %[[#int]]
; CHECK-DAG:        %[[#LocalInvocationIndex:]] = OpVariable %[[#ptr_Input_int]] Input

; CHECK-DAG:        OpEntryPoint GLCompute {{.*}} %[[#LocalInvocationIndex]]
; CHECK-DAG:        OpName %[[#LocalInvocationIndex]] "__spirv_BuiltInLocalInvocationIndex"
; CHECK-DAG:        OpDecorate %[[#LocalInvocationIndex]] BuiltIn LocalInvocationIndex

target triple = "spirv-unknown-vulkan-library"

define internal void @local_index_user(i32) {
entry:
  ret void
}

; Function Attrs: convergent noinline norecurse
define void @main() #1 {
entry:

; CHECK:        %[[#load:]] = OpLoad %[[#int]] %[[#LocalInvocationIndex]]
  %1 = call i32 @llvm.spv.flattened.thread.id.in.group()

  call spir_func void @local_index_user(i32 %1)
  ret void
}

; Function Attrs: nounwind willreturn memory(none)
declare i32 @llvm.spv.flattened.thread.id.in.group() #3

attributes #1 = { convergent noinline norecurse "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #3 = { nounwind willreturn memory(none) }
