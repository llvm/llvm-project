; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s

; CHECK-DAG: %[[#I32:]] = OpTypeInt 32
; CHECK-DAG: %[[#I16:]] = OpTypeInt 16
; CHECK-DAG: %[[#STRUCT:]] = OpTypeStruct %[[#I32]] %[[#I16]]
; CHECK-DAG: %[[#UNDEF:]] = OpUndef %[[#STRUCT]]

; CHECK: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK: %[[#]] = OpLabel
; CHECK: OpStore %[[#]] %[[#UNDEF]] Aligned 4
; CHECK: OpReturn
; CHECK: OpFunctionEnd

define void @foo(ptr %ptr) {
  store { i32, i16 } undef, ptr %ptr
  ret void
}
