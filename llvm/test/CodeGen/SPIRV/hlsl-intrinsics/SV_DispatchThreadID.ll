; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-vulkan-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-vulkan-unknown %s -o - -filetype=obj | spirv-val %}

; This file generated from the following command:
; clang -cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm -disable-llvm-passes -finclude-default-header - -o - <<EOF
; [shader("compute")]
; [numthreads(1,1,1)]
; void main(uint3 ID : SV_DispatchThreadID) {}
; EOF

; CHECK-DAG:        %[[#int:]] = OpTypeInt 32 0
; CHECK-DAG:        %[[#v3int:]] = OpTypeVector %[[#int]] 3
; CHECK-DAG:        %[[#ptr_Input_v3int:]] = OpTypePointer Input %[[#v3int]]
; CHECK-DAG:        %[[#tempvar:]] = OpUndef %[[#v3int]]
; CHECK-DAG:        %[[#GlobalInvocationId:]] = OpVariable %[[#ptr_Input_v3int]] Input

; CHECK-DAG:        OpEntryPoint GLCompute {{.*}} %[[#GlobalInvocationId]]
; CHECK-DAG:        OpName %[[#GlobalInvocationId]] "__spirv_BuiltInGlobalInvocationId"
; CHECK-DAG:        OpDecorate %[[#GlobalInvocationId]] LinkageAttributes "__spirv_BuiltInGlobalInvocationId" Import
; CHECK-DAG:        OpDecorate %[[#GlobalInvocationId]] BuiltIn GlobalInvocationId

; ModuleID = '-'
source_filename = "-"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spirv-unknown-vulkan-library"

; Function Attrs: noinline norecurse nounwind optnone
define internal spir_func void @main(<3 x i32> noundef %ID) #0 {
entry:
  %ID.addr = alloca <3 x i32>, align 16
  store <3 x i32> %ID, ptr %ID.addr, align 16
  ret void
}

; Function Attrs: norecurse
define void @main.1() #1 {
entry:

; CHECK:        %[[#load:]] = OpLoad %[[#v3int]] %[[#GlobalInvocationId]]
; CHECK:        %[[#load0:]] = OpCompositeExtract %[[#int]] %[[#load]] 0
  %0 = call i32 @llvm.spv.thread.id(i32 0)

; CHECK:        %[[#tempvar:]] = OpCompositeInsert %[[#v3int]] %[[#load0]] %[[#tempvar]] 0
  %1 = insertelement <3 x i32> poison, i32 %0, i64 0

; CHECK:        %[[#load:]] = OpLoad %[[#v3int]] %[[#GlobalInvocationId]]
; CHECK:        %[[#load1:]] = OpCompositeExtract %[[#int]] %[[#load]] 1
  %2 = call i32 @llvm.spv.thread.id(i32 1)

; CHECK:        %[[#tempvar:]] = OpCompositeInsert %[[#v3int]] %[[#load1]] %[[#tempvar]] 1
  %3 = insertelement <3 x i32> %1, i32 %2, i64 1

; CHECK:        %[[#load:]] = OpLoad %[[#v3int]] %[[#GlobalInvocationId]]
; CHECK:        %[[#load2:]] = OpCompositeExtract %[[#int]] %[[#load]] 2
  %4 = call i32 @llvm.spv.thread.id(i32 2)

; CHECK:        %[[#tempvar:]] = OpCompositeInsert %[[#v3int]] %[[#load2]] %[[#tempvar]] 2
  %5 = insertelement <3 x i32> %3, i32 %4, i64 2

  call void @main(<3 x i32> %5)
  ret void
}

; Function Attrs: nounwind willreturn memory(none)
declare i32 @llvm.spv.thread.id(i32) #2

attributes #0 = { noinline norecurse nounwind optnone "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { norecurse "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { nounwind willreturn memory(none) }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 4, !"dx.disable_optimizations", i32 1}
!2 = !{!"clang version 19.0.0git (git@github.com:llvm/llvm-project.git 91600507765679e92434ec7c5edb883bf01f847f)"}
