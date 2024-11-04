; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s

; CHECK-DAG: %[[#CHAR:]] = OpTypeInt 8
; CHECK-DAG: %[[#INT:]] = OpTypeInt 32
; CHECK-DAG: %[[#GLOBAL_PTR_INT:]] = OpTypePointer CrossWorkgroup %[[#INT]]
; CHECK-DAG: %[[#GLOBAL_PTR_CHAR:]] = OpTypePointer CrossWorkgroup %[[#CHAR]]

define i32 @foo(i32 %a, ptr addrspace(1) %p) {
  store i32 %a, i32 addrspace(1)* %p
  %b = load i32, i32 addrspace(1)* %p
  ret i32 %b
}

; CHECK: %[[#A:]] = OpFunctionParameter %[[#INT]]
; CHECK: %[[#P:]] = OpFunctionParameter %[[#GLOBAL_PTR_CHAR]]
; CHECK: %[[#C:]] = OpBitcast %[[#GLOBAL_PTR_INT]] %[[#P]]
; CHECK: OpStore %[[#C]] %[[#A]]
; CHECK: %[[#B:]] = OpLoad %[[#INT]] %[[#C]]
; CHECK-NOT: %[[#B:]] = OpLoad %[[#INT]] %[[#P]]
