; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-vulkan-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-vulkan-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG:        %[[#int:]] = OpTypeInt 32 0
; CHECK-DAG:        %[[#v3int:]] = OpTypeVector %[[#int]] 3
; CHECK-DAG:        %[[#ptr_Input_v3int:]] = OpTypePointer Input %[[#v3int]]
; CHECK-DAG:        %[[#tempvar:]] = OpUndef %[[#v3int]]
; CHECK-DAG:        %[[#WorkgroupId:]] = OpVariable %[[#ptr_Input_v3int]] Input

; CHECK-DAG:        OpEntryPoint GLCompute {{.*}} %[[#WorkgroupId]]
; CHECK-DAG:        OpName %[[#WorkgroupId]] "__spirv_BuiltInWorkgroupId"
; CHECK-DAG:        OpDecorate %[[#WorkgroupId]] LinkageAttributes "__spirv_BuiltInWorkgroupId" Import
; CHECK-DAG:        OpDecorate %[[#WorkgroupId]] BuiltIn WorkgroupId

target triple = "spirv-unknown-vulkan-library"

declare void @group_id_user(<3 x i32>)

; Function Attrs: convergent noinline norecurse
define void @main() #1 {
entry:

; CHECK:        %[[#load:]] = OpLoad %[[#v3int]] %[[#WorkgroupId]]
; CHECK:        %[[#load0:]] = OpCompositeExtract %[[#int]] %[[#load]] 0
  %1 = call i32 @llvm.spv.group.id(i32 0)

; CHECK:        %[[#tempvar:]] = OpCompositeInsert %[[#v3int]] %[[#load0]] %[[#tempvar]]
  %2 = insertelement <3 x i32> poison, i32 %1, i64 0

; CHECK:        %[[#load:]] = OpLoad %[[#v3int]] %[[#WorkgroupId]]
; CHECK:        %[[#load1:]] = OpCompositeExtract %[[#int]] %[[#load]] 1
  %3 = call i32 @llvm.spv.group.id(i32 1)

; CHECK:        %[[#tempvar:]] = OpCompositeInsert %[[#v3int]] %[[#load1]] %[[#tempvar]] 1
  %4 = insertelement <3 x i32> %2, i32 %3, i64 1

; CHECK:        %[[#load:]] = OpLoad %[[#v3int]] %[[#WorkgroupId]]
; CHECK:        %[[#load2:]] = OpCompositeExtract %[[#int]] %[[#load]] 2
  %5 = call i32 @llvm.spv.group.id(i32 2)

; CHECK:        %[[#tempvar:]] = OpCompositeInsert %[[#v3int]] %[[#load2]] %[[#tempvar]] 2
  %6 = insertelement <3 x i32> %4, i32 %5, i64 2

  call spir_func void @group_id_user(<3 x i32> %6)
  ret void
}

; Function Attrs: nounwind willreturn memory(none)
declare i32 @llvm.spv.group.id(i32) #3

attributes #1 = { convergent noinline norecurse "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #3 = { nounwind willreturn memory(none) }
