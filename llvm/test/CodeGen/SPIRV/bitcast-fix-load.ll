; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#TYLONG:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#TYSTRUCTLONG:]] = OpTypeStruct %[[#TYLONG]]
; CHECK-DAG: %[[#TYARRAY:]] = OpTypeArray %[[#TYSTRUCTLONG]] %[[#]]
; CHECK-DAG: %[[#TYSTRUCT:]] = OpTypeStruct %[[#TYARRAY]]
; CHECK-DAG: %[[#TYSTRUCTPTR:]] = OpTypePointer Function %[[#TYSTRUCT]]
; CHECK-DAG: %[[#TYLONGPTR:]] = OpTypePointer Function %[[#TYLONG]]
; CHECK: %[[#PTRTOSTRUCT:]] = OpFunctionParameter %[[#TYSTRUCTPTR]]
; CHECK: %[[#PTRTOLONG:]] = OpBitcast %[[#TYLONGPTR]] %[[#PTRTOSTRUCT]]
; CHECK: OpLoad %[[#TYLONG]] %[[#PTRTOLONG]]

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%struct.S = type { i32 }
%struct.__wrapper_class = type { [7 x %struct.S] }

define spir_kernel void @foo(ptr noundef byval(%struct.__wrapper_class) align 4 %_arg_Arr) {
entry:
  %val = load i32, ptr %_arg_Arr
  ret void
}
