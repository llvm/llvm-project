; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK:  %[[#INT8:]] = OpTypeInt 8 0
; CHECK:  %[[#PTR1:]] = OpTypePointer CrossWorkgroup %[[#INT8]]
; CHECK:  %[[#PTR2:]] = OpTypePointer UniformConstant %[[#INT8]]
; CHECK:  %[[#]] = OpInBoundsPtrAccessChain %[[#PTR1]] %[[#]] %[[#]]
; CHECK:  %[[#]] = OpInBoundsPtrAccessChain %[[#PTR2]] %[[#]] %[[#]]

define spir_kernel void @foo(ptr addrspace(1) %a, ptr addrspace(2) %b) {
entry:
  %c = getelementptr inbounds i8, ptr addrspace(1) %a, i32 1
  %d = getelementptr inbounds i8, ptr addrspace(2) %b, i32 2
  ret void
}
