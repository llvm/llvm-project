; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s

; CHECK-DAG: %[[#I32:]] = OpTypeInt 32
; CHECK-DAG: %[[#I16:]] = OpTypeInt 16
; CHECK-DAG: %[[#STRUCT:]] = OpTypeStruct %[[#I32]] %[[#I16]]
; CHECK-DAG: %[[#UNDEF:]] = OpUndef %[[#STRUCT]]

; CHECK: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-NEXT: %[[#PTR:]] = OpFunctionParameter %[[#]]
; CHECK-NEXT: %[[#]] = OpLabel
; CHECK-NEXT: OpStore %[[#PTR]] %[[#UNDEF]] Aligned 4
; CHECK-NEXT: OpReturn
; CHECK-NEXT: OpFunctionEnd

define void @foo(ptr %ptr) {
  store { i32, i16 } undef, ptr %ptr
  ret void
}
