; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan1.3 %s -o - | FileCheck %s
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; A byte getelementptr on a pointer whose pointee is not i8 lowers to an
; OpAccessChain with no explicit index operands. The SPIR-V requirement
; analysis used to unconditionally read the (non-existent) first-index operand
; of every OpAccessChain, which indexed past the end of the instruction's
; operand list and crashed. Check that such an access chain now compiles
; without crashing and still emits the OpAccessChain.

; CHECK: OpFunction
; CHECK: %[[#PTR:]] = OpFunctionParameter
; CHECK: %[[#]] = OpAccessChain %[[#]] %[[#PTR]]
; CHECK: OpReturnValue
; CHECK: OpFunctionEnd

define ptr addrspace(2) @access_chain_no_index(ptr addrspace(2) %p) {
  %gep = getelementptr i8, ptr addrspace(2) %p, i64 8
  ret ptr addrspace(2) %gep
}
