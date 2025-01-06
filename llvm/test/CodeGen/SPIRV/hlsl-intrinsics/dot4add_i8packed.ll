; RUN: llc -O0 -mtriple=spirv1.5-unknown-unknown %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-EXP
; RUN: llc -O0 -mtriple=spirv1.6-unknown-unknown %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-DOT
; RUN: llc -O0 -mtriple=spirv-unknown-unknown -spirv-ext=+SPV_KHR_integer_dot_product %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-DOT,CHECK-EXT
; RUN: %if spirv-tools %{ llc -verify-machineinstrs -O0 -mtriple=spirv1.5-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: %if spirv-tools %{ llc -verify-machineinstrs -O0 -mtriple=spirv1.6-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: %if spirv-tools %{ llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-unknown -spirv-ext=+SPV_KHR_integer_dot_product %s -o - -filetype=obj | spirv-val %}

; CHECK-DOT: OpCapability DotProduct
; CHECK-DOT: OpCapability DotProductInput4x8BitPacked
; CHECK-EXT: OpExtension "SPV_KHR_integer_dot_product"

; CHECK: %[[#int_32:]] = OpTypeInt 32 0
; CHECK-EXP-DAG: %[[#int_8:]] = OpTypeInt 8 0
; CHECK-EXP-DAG: %[[#zero:]] = OpConstantNull %[[#int_8]]
; CHECK-EXP-DAG: %[[#eight:]] = OpConstant %[[#int_8]] 8
; CHECK-EXP-DAG: %[[#sixteen:]] = OpConstant %[[#int_8]] 16
; CHECK-EXP-DAG: %[[#twentyfour:]] = OpConstant %[[#int_8]] 24

; CHECK-LABEL: Begin function test_dot
define noundef i32 @test_dot(i32 noundef %a, i32 noundef %b, i32 noundef %c) {
entry:
; CHECK: %[[#A:]] = OpFunctionParameter %[[#int_32]]
; CHECK: %[[#B:]] = OpFunctionParameter %[[#int_32]]
; CHECK: %[[#C:]] = OpFunctionParameter %[[#int_32]]

; Test that we use the dot product op when capabilities allow

; CHECK-DOT: %[[#DOT:]] = OpSDot %[[#int_32]] %[[#A]] %[[#B]]
; CHECK-DOT: %[[#RES:]] = OpIAdd %[[#int_32]] %[[#DOT]] %[[#C]]

; Test expansion is used when spirv dot product capabilities aren't available:

; First element of the packed vector
; CHECK-EXP: %[[#A0:]] = OpBitFieldSExtract %[[#int_32]] %[[#A]] %[[#zero]] %[[#eight]]
; CHECK-EXP: %[[#B0:]] = OpBitFieldSExtract %[[#int_32]] %[[#B]] %[[#zero]] %[[#eight]]
; CHECK-EXP: %[[#MUL0:]] = OpIMul %[[#int_32]] %[[#A0]] %[[#B0]]
; CHECK-EXP: %[[#MASK0:]] = OpBitFieldSExtract %[[#int_32]] %[[#MUL0]] %[[#zero]] %[[#eight]]
; CHECK-EXP: %[[#ACC0:]] = OpIAdd %[[#int_32]] %[[#C]] %[[#MASK0]]

; Second element of the packed vector
; CHECK-EXP: %[[#A1:]] = OpBitFieldSExtract %[[#int_32]] %[[#A]] %[[#eight]] %[[#eight]]
; CHECK-EXP: %[[#B1:]] = OpBitFieldSExtract %[[#int_32]] %[[#B]] %[[#eight]] %[[#eight]]
; CHECK-EXP: %[[#MUL1:]] = OpIMul %[[#int_32]] %[[#A1]] %[[#B1]]
; CHECK-EXP: %[[#MASK1:]] = OpBitFieldSExtract %[[#int_32]] %[[#MUL1]] %[[#zero]] %[[#eight]]
; CHECK-EXP: %[[#ACC1:]] = OpIAdd %[[#int_32]] %[[#ACC0]] %[[#MASK1]]

; Third element of the packed vector
; CHECK-EXP: %[[#A2:]] = OpBitFieldSExtract %[[#int_32]] %[[#A]] %[[#sixteen]] %[[#eight]]
; CHECK-EXP: %[[#B2:]] = OpBitFieldSExtract %[[#int_32]] %[[#B]] %[[#sixteen]] %[[#eight]]
; CHECK-EXP: %[[#MUL2:]] = OpIMul %[[#int_32]] %[[#A2]] %[[#B2]]
; CHECK-EXP: %[[#MASK2:]] = OpBitFieldSExtract %[[#int_32]] %[[#MUL2]] %[[#zero]] %[[#eight]]
; CHECK-EXP: %[[#ACC2:]] = OpIAdd %[[#int_32]] %[[#ACC1]] %[[#MASK2]]

; Fourth element of the packed vector
; CHECK-EXP: %[[#A3:]] = OpBitFieldSExtract %[[#int_32]] %[[#A]] %[[#twentyfour]] %[[#eight]]
; CHECK-EXP: %[[#B3:]] = OpBitFieldSExtract %[[#int_32]] %[[#B]] %[[#twentyfour]] %[[#eight]]
; CHECK-EXP: %[[#MUL3:]] = OpIMul %[[#int_32]] %[[#A3]] %[[#B3]]
; CHECK-EXP: %[[#MASK3:]] = OpBitFieldSExtract %[[#int_32]] %[[#MUL3]] %[[#zero]] %[[#eight]]

; CHECK-EXP: %[[#RES:]] = OpIAdd %[[#int_32]] %[[#ACC2]] %[[#MASK3]]
; CHECK: OpReturnValue %[[#RES]]
  %spv.dot = call i32 @llvm.spv.dot4add.i8packed(i32 %a, i32 %b, i32 %c)

  ret i32 %spv.dot
}
