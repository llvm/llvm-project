; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-vulkan-unknown %s -o - | FileCheck %s

; FIXME(138268): Alignment decoration is emitted.
; FIXME: %if spirv-tools %{ llc -O0 -mtriple=spirv-vulkan-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG:        OpDecorate %[[#WorkgroupId:]] BuiltIn WorkgroupId

; CHECK-DAG:             %[[#uint:]] = OpTypeInt 32 0
; CHECK-DAG:           %[[#uint_0:]] = OpConstant %[[#uint]] 0
; CHECK-DAG:           %[[#v3uint:]] = OpTypeVector %[[#uint]] 3
; CHECK-DAG:   %[[#ptr_Input_uint:]] = OpTypePointer Input %[[#uint]]
; CHECK-DAG: %[[#ptr_Input_v3uint:]] = OpTypePointer Input %[[#v3uint]]
; CHECK-DAG:      %[[#WorkgroupId:]] = OpVariable %[[#ptr_Input_v3uint]] Input
@var = external local_unnamed_addr addrspace(7) externally_initialized constant <3 x i32>, align 16, !spirv.Decorations !0

define i32 @foo() {
entry:
; CHECK: %[[#ptr:]] = OpAccessChain %[[#ptr_Input_uint]] %[[#WorkgroupId]] %[[#uint_0]]
; CHECK: %[[#res:]] = OpLoad %[[#uint]] %[[#ptr]] Aligned 16
; CHECK:              OpReturnValue %[[#res]]
  %0 = load i32, ptr addrspace(7) @var, align 16
  ret i32 %0
}

!0 = !{!1}
!1 = !{i32 11, i32 26}
