; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s

; CHECK:  %[[#INT8:]] = OpTypeInt 8 0
; CHECK:  %[[#PTR1:]] = OpTypePointer CrossWorkgroup %[[#INT8]]
; CHECK:  %[[#PTR2:]] = OpTypePointer UniformConstant %[[#INT8]]
; CHECK:  %[[#FNP1:]] = OpFunctionParameter %[[#PTR1]]
; CHECK:  %[[#FNP2:]] = OpFunctionParameter %[[#PTR2]]
; CHECK:  %[[#]] = OpLoad %[[#INT8]] %[[#FNP1]] Aligned 1
; CHECK:  %[[#]] = OpLoad %[[#INT8]] %[[#FNP2]] Aligned 1

define spir_kernel void @foo(ptr addrspace(1) %a, ptr addrspace(2) %b) {
entry:
  %c = load i8, ptr addrspace(1) %a
  %d = load i8, ptr addrspace(2) %b
  ret void
}
