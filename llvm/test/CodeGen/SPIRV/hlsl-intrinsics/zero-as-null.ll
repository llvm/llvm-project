; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-vulkan-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-vulkan-unknown %s -o - -filetype=obj | spirv-val %}

; Verify that HLSL shader targets use OpConstantNull for zero constants,
; not OpConstant ... 0.

; CHECK-DAG: %[[#int16:]] = OpTypeInt 16 0
; CHECK-DAG: %[[#vec4_int16:]] = OpTypeVector %[[#int16]] 4
; CHECK:     %[[#zero:]] = OpConstantNull %[[#vec4_int16]]
; CHECK-NOT: OpConstant %[[#int16]] 0

define <4 x i16> @test_vec4_i16_zero() {
  ret <4 x i16> zeroinitializer
}
