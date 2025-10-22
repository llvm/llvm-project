; RUN: llc -O0 -mtriple=spirv1.5-unknown-vulkan %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-EXP
; RUN: llc -O0 -mtriple=spirv1.6-unknown-vulkan %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-DOT
; RUN: llc -O0 -mtriple=spirv-unknown-vulkan -spirv-ext=+SPV_KHR_integer_dot_product %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-DOT,CHECK-EXT
; RUN: %if spirv-tools %{ llc -verify-machineinstrs -O0 -mtriple=spirv1.5-unknown-vulkan %s -o - -filetype=obj | spirv-val %}
; RUN: %if spirv-tools %{ llc -verify-machineinstrs -O0 -mtriple=spirv1.6-unknown-vulkan %s -o - -filetype=obj | spirv-val %}
; RUN: %if spirv-tools %{ llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan -spirv-ext=+SPV_KHR_integer_dot_product %s -o - -filetype=obj | spirv-val %}

; CHECK-DOT: OpCapability DotProduct
; CHECK-DOT: OpCapability DotProductInput4x8BitPacked
; CHECK-EXT: OpExtension "SPV_KHR_integer_dot_product"

; CHECK: %[[#int_32:]] = OpTypeInt 32 0
; CHECK-EXP-DAG: %[[#int_8:]] = OpTypeInt 8 0
; CHECK-EXP-DAG: %[[#zero:]] = OpConstant %[[#int_8]] 0{{$}}
; CHECK-EXP-DAG: %[[#eight:]] = OpConstant %[[#int_8]] 8
; CHECK-EXP-DAG: %[[#sixteen:]] = OpConstant %[[#int_8]] 16
; CHECK-EXP-DAG: %[[#twentyfour:]] = OpConstant %[[#int_8]] 24

; CHECK-LABEL: Begin function test_dot
define noundef i32 @test_dot(i32 noundef %acc, i32 noundef %x, i32 noundef %y) {
entry:
; CHECK: %[[#ACC:]] = OpFunctionParameter %[[#int_32]]
; CHECK: %[[#X:]] = OpFunctionParameter %[[#int_32]]
; CHECK: %[[#Y:]] = OpFunctionParameter %[[#int_32]]

; Test that we use the dot product op when capabilities allow

; CHECK-DOT: %[[#DOT:]] = OpSDot %[[#int_32]] %[[#X]] %[[#Y]]
; CHECK-DOT: %[[#RES:]] = OpIAdd %[[#int_32]] %[[#DOT]] %[[#ACC]]

; Test expansion is used when spirv dot product capabilities aren't available:

; First element of the packed vector
; CHECK-EXP: %[[#X0:]] = OpBitFieldSExtract %[[#int_32]] %[[#X]] %[[#zero]] %[[#eight]]
; CHECK-EXP: %[[#Y0:]] = OpBitFieldSExtract %[[#int_32]] %[[#Y]] %[[#zero]] %[[#eight]]
; CHECK-EXP: %[[#MUL0:]] = OpIMul %[[#int_32]] %[[#X0]] %[[#Y0]]
; CHECK-EXP: %[[#MASK0:]] = OpBitFieldSExtract %[[#int_32]] %[[#MUL0]] %[[#zero]] %[[#eight]]
; CHECK-EXP: %[[#ACC0:]] = OpIAdd %[[#int_32]] %[[#ACC]] %[[#MASK0]]

; Second element of the packed vector
; CHECK-EXP: %[[#X1:]] = OpBitFieldSExtract %[[#int_32]] %[[#X]] %[[#eight]] %[[#eight]]
; CHECK-EXP: %[[#Y1:]] = OpBitFieldSExtract %[[#int_32]] %[[#Y]] %[[#eight]] %[[#eight]]
; CHECK-EXP: %[[#MUL1:]] = OpIMul %[[#int_32]] %[[#X1]] %[[#Y1]]
; CHECK-EXP: %[[#MASK1:]] = OpBitFieldSExtract %[[#int_32]] %[[#MUL1]] %[[#zero]] %[[#eight]]
; CHECK-EXP: %[[#ACC1:]] = OpIAdd %[[#int_32]] %[[#ACC0]] %[[#MASK1]]

; Third element of the packed vector
; CHECK-EXP: %[[#X2:]] = OpBitFieldSExtract %[[#int_32]] %[[#X]] %[[#sixteen]] %[[#eight]]
; CHECK-EXP: %[[#Y2:]] = OpBitFieldSExtract %[[#int_32]] %[[#Y]] %[[#sixteen]] %[[#eight]]
; CHECK-EXP: %[[#MUL2:]] = OpIMul %[[#int_32]] %[[#X2]] %[[#Y2]]
; CHECK-EXP: %[[#MASK2:]] = OpBitFieldSExtract %[[#int_32]] %[[#MUL2]] %[[#zero]] %[[#eight]]
; CHECK-EXP: %[[#ACC2:]] = OpIAdd %[[#int_32]] %[[#ACC1]] %[[#MASK2]]

; Fourth element of the packed vector
; CHECK-EXP: %[[#X3:]] = OpBitFieldSExtract %[[#int_32]] %[[#X]] %[[#twentyfour]] %[[#eight]]
; CHECK-EXP: %[[#Y3:]] = OpBitFieldSExtract %[[#int_32]] %[[#Y]] %[[#twentyfour]] %[[#eight]]
; CHECK-EXP: %[[#MUL3:]] = OpIMul %[[#int_32]] %[[#X3]] %[[#Y3]]
; CHECK-EXP: %[[#MASK3:]] = OpBitFieldSExtract %[[#int_32]] %[[#MUL3]] %[[#zero]] %[[#eight]]

; CHECK-EXP: %[[#RES:]] = OpIAdd %[[#int_32]] %[[#ACC2]] %[[#MASK3]]
; CHECK: OpReturnValue %[[#RES]]
  %spv.dot = call i32 @llvm.spv.dot4add.i8packed(i32 %acc, i32 %x, i32 %y)

  ret i32 %spv.dot
}
