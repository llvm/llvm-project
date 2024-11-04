; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s

; CHECK-DAG: OpName %[[#BAR:]] "bar"
; CHECK-DAG: OpName %[[#FOO:]] "foo"
; CHECK-DAG: OpName %[[#GOO:]] "goo"

; CHECK-DAG: %[[#INT:]] = OpTypeInt 32
; CHECK-DAG: %[[#STACK_PTR_INT:]] = OpTypePointer Function %[[#INT]]
; CHECK-DAG: %[[#GLOBAL_PTR_INT:]] = OpTypePointer CrossWorkgroup %[[#INT]]
; CHECK-DAG: %[[#FN1:]] = OpTypeFunction %[[#INT]] %[[#INT]]
; CHECK-DAG: %[[#FN2:]] = OpTypeFunction %[[#INT]] %[[#INT]] %[[#GLOBAL_PTR_INT]]

define i32 @bar(i32 %a) {
  %p = alloca i32
  store i32 %a, i32* %p
  %b = load i32, i32* %p
  ret i32 %b
}

; CHECK: %[[#BAR]] = OpFunction %[[#INT]] None %[[#FN1]]
; CHECK: %[[#A:]] = OpFunctionParameter %[[#INT]]
; CHECK: OpLabel
; CHECK: %[[#P:]] = OpVariable %[[#STACK_PTR_INT]] Function
; CHECK: OpStore %[[#P]] %[[#A]]
; CHECK: %[[#B:]] = OpLoad %[[#INT]] %[[#P]]
; CHECK: OpReturnValue %[[#B]]
; CHECK: OpFunctionEnd


define i32 @foo(i32 %a) {
  %p = alloca i32
  store volatile i32 %a, i32* %p
  %b = load volatile i32, i32* %p
  ret i32 %b
}

; CHECK: %[[#FOO]] = OpFunction %[[#INT]] None %[[#FN1]]
; CHECK: %[[#A:]] = OpFunctionParameter %[[#INT]]
; CHECK: OpLabel
; CHECK: %[[#P:]] = OpVariable %[[#STACK_PTR_INT]] Function
; CHECK: OpStore %[[#P]] %[[#A]] Volatile
; CHECK: %[[#B:]] = OpLoad %[[#INT]] %[[#P]] Volatile
; CHECK: OpReturnValue %[[#B]]
; CHECK: OpFunctionEnd


;; Test load and store in global address space.
define i32 @goo(i32 %a, ptr addrspace(1) %p) {
  store i32 %a, i32 addrspace(1)* %p
  %b = load i32, i32 addrspace(1)* %p
  ret i32 %b
}

; CHECK: %[[#GOO]] = OpFunction %[[#INT]] None %[[#FN2]]
; CHECK: %[[#A:]] = OpFunctionParameter %[[#INT]]
; CHECK: %[[#P:]] = OpFunctionParameter %[[#GLOBAL_PTR_INT]]
; CHECK: OpLabel
; CHECK: OpStore %[[#P]] %[[#A]]
; CHECK: %[[#B:]] = OpLoad %[[#INT]] %[[#P]]
; CHECK: OpReturnValue %[[#B]]
; CHECK: OpFunctionEnd
