; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#I16:]] = OpTypeInt 16 0
; CHECK-DAG: %[[#I32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#STRUCT:]] = OpTypeStruct %[[#I32]] %[[#I16]]
; CHECK-DAG: %[[#NESTED:]] = OpTypeStruct %[[#STRUCT]] %[[#I16]]
; CHECK-DAG: %[[#PTR:]] = OpTypePointer Function %[[#NESTED]]
; CHECK-DAG: %[[#UNDEF:]] = OpUndef %[[#NESTED]]
; CHECK-DAG: %[[#C_I32_1:]] = OpConstant %[[#I32]] 1
; CHECK-DAG: %[[#C_I16_2:]] = OpConstant %[[#I16]] 2
; CHECK-DAG: %[[#C_I16_3:]] = OpConstant %[[#I16]] 3
; CHECK-DAG: %[[#INNER:]] = OpConstantComposite %[[#STRUCT]] %[[#C_I32_1]] %[[#C_I16_2]]
; CHECK-DAG: %[[#OUTER:]] = OpConstantComposite %[[#NESTED]] %[[#INNER]] %[[#C_I16_3]]

; CHECK: OpFunction
; CHECK: %[[#PARAM:]] = OpFunctionParameter %[[#PTR]]
; CHECK-NOT: OpBitcast
; CHECK: OpStore %[[#PARAM]] %[[#UNDEF]]
; CHECK: OpFunctionEnd

; CHECK: OpFunction
; CHECK: %[[#PARAM2:]] = OpFunctionParameter %[[#PTR]]
; CHECK-NOT: OpBitcast
; CHECK: OpStore %[[#PARAM2]] %[[#OUTER]]
; CHECK: OpFunctionEnd

%struct = type {
  i32,
  i16
}

%nested_struct = type {
  %struct,
  i16
}

define void @foo(ptr %ptr) {
  store %nested_struct undef, ptr %ptr
  ret void
}

define void @bar(ptr %ptr) {
  store %nested_struct { %struct { i32 1, i16 2 }, i16 3 }, ptr %ptr
  ret void
}
