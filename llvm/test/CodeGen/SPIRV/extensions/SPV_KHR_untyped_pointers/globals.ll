; This test validated untyped access chain and its use in SpecConstantOp.
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - | FileCheck %s
; TODO: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - -filetype=obj | spirv-val %}

; TODO: enable back once spirv-tools are updated and allow untyped access chain as OpSpecConstantOp operand.
; XFAIL: *

; CHECK-DAG: %[[#I16:]] = OpTypeInt 16 0
; CHECK-DAG: %[[#I32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#I64:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#CONST0:]] = OpConstant %[[#I16]] 0
; CHECK-DAG: %[[#CONST2:]] = OpConstant %[[#I32]] 2
; CHECK-DAG: %[[#CONST3:]] = OpConstant %[[#I32]] 3
; CHECK-DAG: %[[#CONST4:]] = OpConstant %[[#I32]] 4
; CHECK-DAG: %[[#CONST0_I64:]] = OpConstant %[[#I64]] 0
; CHECK-DAG: %[[#CONST1_I64:]] = OpConstant %[[#I64]] 1
; CHECK-DAG: %[[#CONST2_I64:]] = OpConstant %[[#I64]] 2
; CHECK-DAG: %[[#CONST3_I64:]] = OpConstant %[[#I64]] 3

; CHECK-DAG: %[[#PTRTY:]] = OpTypeUntypedPointerKHR 5
; CHECK-DAG: %[[#LOCALPTRTY:]] = OpTypeUntypedPointerKHR 4
; CHECK-DAG: %[[#ARRAYTY:]] = OpTypeArray %[[#PTRTY]] %[[#CONST2]]
; CHECK-DAG: %[[#ARRAYPTRTY:]] = OpTypePointer 5 %[[#ARRAYTY]]
; CHECK-DAG: %[[#ARRAY1TY:]] = OpTypeArray %[[#I32]] %[[#CONST4]]
; CHECK-DAG: %[[#ARRAY2TY:]] = OpTypeArray %[[#ARRAY1TY]] %[[#CONST3]]
; CHECK-DAG: %[[#ARRAY3TY:]] = OpTypeArray %[[#ARRAY2TY]] %[[#CONST2]]
; CHECK-DAG: %[[#ARRAY3PTRTY:]] = OpTypePointer 5 %[[#ARRAY3TY]]
; CHECK: %[[#VARA:]] = OpUntypedVariableKHR %[[#PTRTY]] 5 %[[#I16]] %[[#CONST0]]
; CHECK: %[[#VARB:]] = OpUntypedVariableKHR %[[#PTRTY]] 5 %[[#I32]]
; CHECK: %[[#VARC:]] = OpUntypedVariableKHR %[[#PTRTY]] 5 %[[#PTRTY]] %[[#VARA]]
; CHECK: %[[#VARD:]] = OpUntypedVariableKHR %[[#LOCALPTRTY]] 4 %[[#PTRTY]]
; CHECK: %[[#VARE:]] = OpVariable %[[#ARRAYPTRTY]] 5
; CHECK: %[[#VARF:]] = OpVariable %[[#ARRAY3PTRTY]] 5
; CHECK: %[[#SPECCONST:]] = OpSpecConstantOp %[[#PTRTY]] 4424 %[[#ARRAY3TY]] %[[#VARF]] %[[#CONST0_I64]] %[[#CONST1_I64]] %[[#CONST2_I64]] %[[#CONST3_I64]]
; CHECK: %[[#VARG:]] = OpUntypedVariableKHR %[[#PTRTY]] 5 %[[#PTRTY]] %[[#SPECCONST]]

@a = addrspace(1) global i16 0
@b = external addrspace(1) global i32
@c = addrspace(1) global ptr addrspace(1) @a
@d = external addrspace(3) global ptr addrspace(1)
@e = addrspace(1) global [2 x ptr addrspace(1)] [ptr addrspace(1) @a, ptr addrspace(1) @b]
@f = addrspace(1) global [2 x [3 x [4 x i32]]] [[3 x [4 x i32]] [[4 x i32] [i32 1, i32 2, i32 3, i32 4], [4 x i32] [i32 1, i32 2, i32 3, i32 4], [4 x i32] [i32 1, i32 2, i32 3, i32 4]], [3 x [4 x i32]] [[4 x i32] [i32 1, i32 2, i32 3, i32 4], [4 x i32] [i32 1, i32 2, i32 3, i32 4], [4 x i32] [i32 1, i32 2, i32 3, i32 4]]]
@g = addrspace(1) global ptr addrspace(1) getelementptr inbounds ([2 x [3 x [4 x i32]]], ptr addrspace(1) @f, i64 0, i64 1, i64 2, i64 3)

define spir_func void @foo() {
entry:
  ret void
}
