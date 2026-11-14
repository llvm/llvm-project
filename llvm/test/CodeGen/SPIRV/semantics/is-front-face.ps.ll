; RUN: split-file %s %t
; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-vulkan1.3-pixel %t/bool.ll -o - | FileCheck %t/bool.ll --check-prefix=BOOL --implicit-check-not=OpExtension
; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-vulkan1.2-pixel %t/bool.ll -o - | FileCheck %t/bool.ll --check-prefix=BOOL --implicit-check-not=OpExtension
; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-vulkan1.3-pixel %t/uint.ll -o - | FileCheck %t/uint.ll --check-prefix=UINT --implicit-check-not=OpExtension
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan1.3-pixel %t/bool.ll -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan1.3-pixel %t/uint.ll -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; SV_IsFrontFace is lowered to a flat load of the FrontFacing built-in. The
; built-in is always an OpTypeBool, so a bool parameter uses the loaded value
; directly, and a uint or int parameter converts it with a select between 1 and
; 0. FrontFacing needs no SPIR-V extension.

;--- bool.ll
; BOOL-DAG: OpDecorate %[[#FF:]] BuiltIn FrontFacing
; BOOL-DAG: OpDecorate %[[#FF]] Flat
; The built-in is part of the entry point interface.
; BOOL-DAG: OpEntryPoint Fragment %[[#]] "main" {{.*}}%[[#FF]]{{( |$)}}
; BOOL-DAG: %[[#BOOL:]] = OpTypeBool
; BOOL-DAG: %[[#PTR:]] = OpTypePointer Input %[[#BOOL]]
; BOOL-DAG: %[[#FF]] = OpVariable %[[#PTR]] Input

@SV_IsFrontFace = external hidden thread_local addrspace(7) externally_initialized constant i1, !spirv.Decorations !0
@SV_Target0 = external hidden thread_local addrspace(8) global float, !spirv.Decorations !3

define void @main() #0 {
entry:
; BOOL: %[[#LOAD:]] = OpLoad %[[#BOOL]] %[[#FF]]
; BOOL: OpSelect %[[#]] %[[#LOAD]] %[[#]] %[[#]]
  %0 = load i1, ptr addrspace(7) @SV_IsFrontFace, align 1
  %1 = select i1 %0, float 1.0, float 0.0
  store float %1, ptr addrspace(8) @SV_Target0, align 4
  ret void
}

attributes #0 = { "hlsl.shader"="pixel" }

!0 = !{!1, !2}
!1 = !{i32 11, i32 17}
!2 = !{i32 14}
!3 = !{!4}
!4 = !{i32 30, i32 0}

;--- uint.ll
; UINT-DAG: OpDecorate %[[#FF:]] BuiltIn FrontFacing
; UINT-DAG: OpDecorate %[[#FF]] Flat
; The built-in is part of the entry point interface.
; UINT-DAG: OpEntryPoint Fragment %[[#]] "main" {{.*}}%[[#FF]]{{( |$)}}
; UINT-DAG: %[[#BOOL:]] = OpTypeBool
; UINT-DAG: %[[#INT:]] = OpTypeInt 32 0
; UINT-DAG: %[[#ONE:]] = OpConstant %[[#INT]] 1
; UINT-DAG: %[[#ZERO:]] = {{OpConstantNull|OpConstant}} %[[#INT]]{{( 0)?$}}
; UINT-DAG: %[[#PTR:]] = OpTypePointer Input %[[#BOOL]]
; UINT-DAG: %[[#FF]] = OpVariable %[[#PTR]] Input

@SV_IsFrontFace = external hidden thread_local addrspace(7) externally_initialized constant i1, !spirv.Decorations !0
@SV_Target0 = external hidden thread_local addrspace(8) global float, !spirv.Decorations !3

define void @main() #0 {
entry:
; The variable stays bool. The conversion to uint is a zero extension, which is
; 1 when the face is front and 0 otherwise.
; UINT: %[[#LOAD:]] = OpLoad %[[#BOOL]] %[[#FF]]
; UINT: OpSelect %[[#INT]] %[[#LOAD]] %[[#ONE]] %[[#ZERO]]
  %0 = load i1, ptr addrspace(7) @SV_IsFrontFace, align 1
  %1 = zext i1 %0 to i32
  %2 = uitofp i32 %1 to float
  store float %2, ptr addrspace(8) @SV_Target0, align 4
  ret void
}

attributes #0 = { "hlsl.shader"="pixel" }

!0 = !{!1, !2}
!1 = !{i32 11, i32 17}
!2 = !{i32 14}
!3 = !{!4}
!4 = !{i32 30, i32 0}
