; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-vulkan1.3-pixel %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan1.3-pixel %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; This test is the simplified llvm IR of flat-decoration.ps.hlsl. It exists
; because we can't do end to end testing but this gets us close & shows that
; the generated code will now pass the spirv validator by adding the Flat & 
; Location decorators. 

; CHECK-DAG: OpEntryPoint Fragment %[[#entry:]] "main"
; CHECK-DAG: OpExecutionMode %[[#entry]] OriginUpperLeft

; CHECK-DAG: OpDecorate %[[#INT_IN:]] Location 0
; CHECK-DAG: OpDecorate %[[#INT_IN]] Flat
; CHECK-DAG: OpDecorate %[[#DOUBLE_IN:]] Location 1
; CHECK-DAG: OpDecorate %[[#DOUBLE_IN]] Flat
; CHECK-DAG: OpDecorate %[[#FLOAT_IN:]] Location 2
; CHECK-NOT: OpDecorate %[[#FLOAT_IN]] Flat

@B0 = external hidden thread_local addrspace(7) externally_initialized constant i32, !spirv.Decorations !0
@C0 = external hidden thread_local addrspace(7) externally_initialized constant double, !spirv.Decorations !3
@A0 = external hidden thread_local addrspace(7) externally_initialized constant float, !spirv.Decorations !5
@SV_Target = external hidden thread_local addrspace(8) global float, !spirv.Decorations !7

define void @main() #0 {
entry:
  %0 = load i32, ptr addrspace(7) @B0, align 4
  %1 = load double, ptr addrspace(7) @C0, align 8
  %2 = load float, ptr addrspace(7) @A0, align 4
  %conv.i = sitofp i32 %0 to float
  %conv.d = fptrunc double %1 to float
  %add1 = fadd float %2, %conv.i
  %add2 = fadd float %add1, %conv.d
  store float %add2, ptr addrspace(8) @SV_Target, align 4
  ret void
}

attributes #0 = { "hlsl.shader"="pixel" }

; Integer input: Location 0 and Flat.
!0 = !{!1, !2}
!1 = !{i32 30, i32 0}
!2 = !{i32 14}
; Double input: Location 1 and Flat.
!3 = !{!4, !2}
!4 = !{i32 30, i32 1}
; Float input: Location 2 only (interpolated, no Flat).
!5 = !{!6}
!6 = !{i32 30, i32 2}
; SV_Target output: Location 0.
!7 = !{!8}
!8 = !{i32 30, i32 0}
