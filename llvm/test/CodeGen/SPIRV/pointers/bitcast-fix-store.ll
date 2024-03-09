; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#TYLONG:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#TYLONGPTR:]] = OpTypePointer Function %[[#TYLONG]]
; CHECK-DAG: %[[#TYSTRUCT:]] = OpTypeStruct %[[#TYLONG]]
; CHECK-DAG: %[[#CONST:]] = OpConstant %[[#TYLONG]] 3
; CHECK-DAG: %[[#TYSTRUCTPTR:]] = OpTypePointer Function %[[#TYSTRUCT]]
; CHECK: OpFunction
; CHECK: %[[#ARGPTR1:]] = OpFunctionParameter %[[#TYLONGPTR]]
; CHECK: OpStore %[[#ARGPTR1]] %[[#CONST:]]
; CHECK: OpFunction
; CHECK: %[[#OBJ:]] = OpFunctionParameter %[[#TYSTRUCT]]
; CHECK: %[[#ARGPTR2:]] = OpFunctionParameter %[[#TYLONGPTR]]
; CHECK: %[[#PTRTOSTRUCT:]] = OpBitcast %[[#TYSTRUCTPTR]] %[[#ARGPTR2]]
; CHECK: OpStore %[[#PTRTOSTRUCT]] %[[#OBJ]]

%struct.S = type { i32 }
%struct.__wrapper_class = type { [7 x %struct.S] }

define spir_kernel void @foo(%struct.S %arg, ptr %ptr) {
entry:
  store %struct.S %arg, ptr %ptr
  ret void
}

define spir_kernel void @bar(ptr %ptr) {
entry:
  store i32 3, ptr %ptr
  ret void
}
