; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#INT:]] = OpTypeInt 32
; CHECK-DAG: %[[#GLOBAL_PTR_INT:]] = OpTypePointer CrossWorkgroup %[[#INT]]

define i32 @foo(i32 %a, ptr addrspace(1) %p) {
  store i32 %a, i32 addrspace(1)* %p
  %b = load i32, i32 addrspace(1)* %p
  ret i32 %b
}

; CHECK: %[[#A:]] = OpFunctionParameter %[[#INT]]
; CHECK: %[[#CorP:]] = {{OpBitcast|OpFunctionParameter}}{{.*}}%[[#GLOBAL_PTR_INT]]{{.*}}
; CHECK: OpStore %[[#CorP]] %[[#A]]
; CHECK: %[[#B:]] = OpLoad %[[#INT]] %[[#CorP]]
