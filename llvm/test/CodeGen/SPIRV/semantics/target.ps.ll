; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-vulkan-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-vulkan-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG:        OpDecorate %[[#INPUT:]] BuiltIn FragCoord
; CHECK-DAG:        OpDecorate %[[#OUTPUT:]] Location 0

; CHECK-DAG:   %[[#float:]] = OpTypeFloat 32
; CHECK-DAG:      %[[#v4:]] = OpTypeVector %[[#float]] 4
; CHECK-DAG:   %[[#ptr_i:]] = OpTypePointer Input %[[#v4]]
; CHECK-DAG:   %[[#ptr_o:]] = OpTypePointer Output %[[#v4]]

; CHECK-DAG:      %[[#INPUT]] = OpVariable %[[#ptr_i]] Input
; CHECK-DAG:      %[[#OUTPUT]] = OpVariable %[[#ptr_o]] Output

@SV_Position = external hidden thread_local addrspace(7) externally_initialized constant <4 x float>, !spirv.Decorations !0
@SV_Target0 = external hidden thread_local addrspace(8) global <4 x float>, !spirv.Decorations !2

define void @main() #1 {
entry:
  %0 = load <4 x float>, ptr addrspace(7) @SV_Position, align 16
  store <4 x float> %0, ptr addrspace(8) @SV_Target0, align 16
  ret void

; CHECK: %[[#TMP:]] = OpLoad %[[#v4]] %[[#INPUT]] Aligned 16
; CHECK:              OpStore %[[#OUTPUT]] %[[#TMP]] Aligned 16
}

!0 = !{!1}
!1 = !{i32 11, i32 15}
!2 = !{!3}
!3 = !{i32 30, i32 0}


