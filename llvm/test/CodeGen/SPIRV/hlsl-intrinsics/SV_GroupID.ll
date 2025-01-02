; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-vulkan-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-vulkan-unknown %s -o - -filetype=obj | spirv-val %}

; This file generated from the following command:
; clang -cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm -disable-llvm-passes -finclude-default-header - -o - <<EOF
; [shader("compute")]
; [numthreads(1,1,1)]
; void main(uint3 ID : SV_GroupID) {}
; EOF

; CHECK-DAG:        %[[#int:]] = OpTypeInt 32 0
; CHECK-DAG:        %[[#v3int:]] = OpTypeVector %[[#int]] 3
; CHECK-DAG:        %[[#ptr_Input_v3int:]] = OpTypePointer Input %[[#v3int]]
; CHECK-DAG:        %[[#tempvar:]] = OpUndef %[[#v3int]]
; CHECK-DAG:        %[[#WorkgroupId:]] = OpVariable %[[#ptr_Input_v3int]] Input

; CHECK-DAG:        OpEntryPoint GLCompute {{.*}} %[[#WorkgroupId]]
; CHECK-DAG:        OpName %[[#WorkgroupId]] "__spirv_BuiltInWorkgroupId"
; CHECK-DAG:        OpDecorate %[[#WorkgroupId]] LinkageAttributes "__spirv_BuiltInWorkgroupId" Import
; CHECK-DAG:        OpDecorate %[[#WorkgroupId]] BuiltIn WorkgroupId

; ModuleID = '-'
source_filename = "-"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv-unknown-vulkan-library"

; Function Attrs: alwaysinline convergent mustprogress norecurse nounwind
define internal spir_func void @_Z4mainDv3_j(<3 x i32> noundef %ID) #0 {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  %ID.addr = alloca <3 x i32>, align 16
  store <3 x i32> %ID, ptr %ID.addr, align 16
  ret void
}

; Function Attrs: convergent noinline norecurse
define void @main() #1 {
entry:
  %0 = call token @llvm.experimental.convergence.entry()

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

  call spir_func void @_Z4mainDv3_j(<3 x i32> %6) [ "convergencectrl"(token %0) ]
  ret void
}

; Function Attrs: convergent nocallback nofree nosync nounwind willreturn memory(none)
declare token @llvm.experimental.convergence.entry() #2

; Function Attrs: nounwind willreturn memory(none)
declare i32 @llvm.spv.group.id(i32) #3

attributes #0 = { alwaysinline convergent mustprogress norecurse nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { convergent noinline norecurse "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { convergent nocallback nofree nosync nounwind willreturn memory(none) }
attributes #3 = { nounwind willreturn memory(none) }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 4, !"dx.disable_optimizations", i32 1}
!2 = !{!"clang version 20.0.0git (https://github.com/llvm/llvm-project.git 4075ddad7183e6f0b66e2c8cc7a03b461a8038e6)"}
