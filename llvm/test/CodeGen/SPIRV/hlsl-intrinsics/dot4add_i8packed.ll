; RUN: llc -O0 -mtriple=spirv32v1.3-vulkan-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32v1.3-vulkan-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#int_32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#int_8:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#zero:]] = OpConstantNull %[[#int_8]]
; CHECK-DAG: %[[#eight:]] = OpConstant %[[#int_8]] 8
; CHECK-DAG: %[[#sixteen:]] = OpConstant %[[#int_8]] 16
; CHECK-DAG: %[[#twentyfour:]] = OpConstant %[[#int_8]] 24
; CHECK-LABEL: Begin function test_dot
define noundef i32 @test_dot(i32 noundef %a, i32 noundef %b, i32 noundef %c) {
entry:
; CHECK: %[[#A:]] = OpFunctionParameter %[[#int_32]]
; CHECK: %[[#B:]] = OpFunctionParameter %[[#int_32]]
; CHECK: %[[#C:]] = OpFunctionParameter %[[#int_32]]

; First element of the packed vector
; CHECK: %[[#A0:]] = OpBitFieldSExtract %[[#int_32]] %[[#A]] %[[#zero]] 8
; CHECK: %[[#B0:]] = OpBitFieldSExtract %[[#int_32]] %[[#B]] %[[#zero]] 8
; CHECK: %[[#MUL0:]] = OpIMul %[[#int_32]] %[[#A0]] %[[#B0]]
; CHECK: %[[#MASK0:]] = OpBitFieldSExtract %[[#int_32]] %[[#MUL0]] %[[#zero]] 8
; CHECK: %[[#ACC0:]] = OpIAdd %[[#int_32]] %[[#C]] %[[#MASK0]]

; Second element of the packed vector
; CHECK: %[[#A1:]] = OpBitFieldSExtract %[[#int_32]] %[[#A]] %[[#eight]] 8
; CHECK: %[[#B1:]] = OpBitFieldSExtract %[[#int_32]] %[[#B]] %[[#eight]] 8
; CHECK: %[[#MUL1:]] = OpIMul %[[#int_32]] %[[#A1]] %[[#B1]]
; CHECK: %[[#MASK1:]] = OpBitFieldSExtract %[[#int_32]] %[[#MUL1]] %[[#zero]] 8
; CHECK: %[[#ACC1:]] = OpIAdd %[[#int_32]] %[[#ACC0]] %[[#MASK1]]

; Third element of the packed vector
; CHECK: %[[#A2:]] = OpBitFieldSExtract %[[#int_32]] %[[#A]] %[[#sixteen]] 8
; CHECK: %[[#B2:]] = OpBitFieldSExtract %[[#int_32]] %[[#B]] %[[#sixteen]] 8
; CHECK: %[[#MUL2:]] = OpIMul %[[#int_32]] %[[#A2]] %[[#B2]]
; CHECK: %[[#MASK2:]] = OpBitFieldSExtract %[[#int_32]] %[[#MUL2]] %[[#zero]] 8
; CHECK: %[[#ACC2:]] = OpIAdd %[[#int_32]] %[[#ACC1]] %[[#MASK2]]

; Fourth element of the packed vector
; CHECK: %[[#A3:]] = OpBitFieldSExtract %[[#int_32]] %[[#A]] %[[#twentyfour]] 8
; CHECK: %[[#B3:]] = OpBitFieldSExtract %[[#int_32]] %[[#B]] %[[#twentyfour]] 8
; CHECK: %[[#MUL3:]] = OpIMul %[[#int_32]] %[[#A3]] %[[#B3]]
; CHECK: %[[#MASK3:]] = OpBitFieldSExtract %[[#int_32]] %[[#MUL3]] %[[#zero]] 8
; CHECK: %[[#ACC3:]] = OpIAdd %[[#int_32]] %[[#ACC2]] %[[#MASK3]]

; CHECK: OpReturnValue %[[#ACC3]]
  %spv.dot = call i32 @llvm.spv.dot4add.i8packed(i32 %a, i32 %b, i32 %c)
  ret i32 %spv.dot
}
