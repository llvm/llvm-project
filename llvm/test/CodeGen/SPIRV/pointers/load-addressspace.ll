; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK:  %[[#INT8:]] = OpTypeInt 8 0
; CHECK:  %[[#PTR1:]] = OpTypePointer CrossWorkgroup %[[#INT8]]
; CHECK:  %[[#PTR2:]] = OpTypePointer UniformConstant %[[#INT8]]
; CHECK:  %[[#FNP1:]] = OpFunctionParameter %[[#PTR1]]
; CHECK:  %[[#FNP2:]] = OpFunctionParameter %[[#PTR2]]
; CHECK:  %[[#]] = OpLoad %[[#INT8]] %[[#FNP1]] Aligned 1
; CHECK:  %[[#]] = OpLoad %[[#INT8]] %[[#FNP2]] Aligned 1

@G_c = global i8 0
@G_d = global i8 0

define spir_kernel void @foo(ptr addrspace(1) %a, ptr addrspace(2) %b) {
entry:
  %c = load i8, ptr addrspace(1) %a
  store i8 %c, ptr @G_c
  %d = load i8, ptr addrspace(2) %b
  store i8 %d, ptr @G_d
  ret void
}
