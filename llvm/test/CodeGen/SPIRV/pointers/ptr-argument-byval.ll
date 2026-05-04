; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#VOID:]] = OpTypeVoid
; CHECK-DAG: %[[#INT32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#STRUCT1:]] = OpTypeStruct %[[#INT32]]
; CHECK-DAG: %[[#CONST:]] = OpConstant %[[#INT32]] 7
; CHECK-DAG: %[[#ARRAY:]] = OpTypeArray %[[#STRUCT1]] %[[#CONST]]
; CHECK-DAG: %[[#STRUCT2:]] = OpTypeStruct %[[#ARRAY]]
; CHECK-DAG: %[[#PTR:]] = OpTypePointer Function %[[#STRUCT2]]

; CHECK:  %[[#FUNC:]] = OpTypeFunction %[[#VOID]] %[[#PTR]]
; CHECK:  %[[#]] = OpFunction %[[#VOID]] None %[[#FUNC]]
; CHECK:  %[[#]] = OpFunctionParameter %[[#PTR]]

%struct.S = type { i32 }
%struct.__wrapper_class = type { [7 x %struct.S] }

define spir_kernel void @foo(ptr noundef byval(%struct.__wrapper_class) align 4 %_arg_Arr) {
entry:
  ret void
}
