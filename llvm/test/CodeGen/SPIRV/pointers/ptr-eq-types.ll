; This test case ensures that several references to "null" in the same function
; don't break validity of the output code from the perspective of type inference.

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: OpInBoundsPtrAccessChain
; CHECK: OpPtrCastToGeneric
; CHECK: OpGenericCastToPtr
; CHECK: OpPtrEqual
; CHECK: OpInBoundsPtrAccessChain
; CHECK: OpGenericCastToPtr
; CHECK: OpPtrEqual

define spir_kernel void @foo(ptr addrspace(3) align 4 %_arg_local, ptr addrspace(1) align 4 %_arg_global) {
entry:
  %p1 = getelementptr inbounds i32, ptr addrspace(1) %_arg_global, i64 0
  %p2 = getelementptr inbounds i32, ptr addrspace(3) %_arg_local, i64 0
  store i32 0, ptr addrspace(3) %p2, align 4
  %p3 = getelementptr inbounds i32, ptr addrspace(1) %p1, i64 0
  %p4 = addrspacecast ptr addrspace(1) %p3 to ptr addrspace(4)
  %p5 = tail call spir_func ptr addrspace(3) @_Z40__spirv_GenericCastToPtrExplicit_ToLocalPvi(ptr addrspace(4) %p4, i32 4)
  %b1 = icmp eq ptr addrspace(3) %p5, null
  %p6 = getelementptr inbounds i32, ptr addrspace(3) %p5, i64 0
  %p7 = tail call spir_func ptr addrspace(3) @_Z40__spirv_GenericCastToPtrExplicit_ToLocalPvi(ptr addrspace(4) %p4, i32 4)
  %b2 = icmp eq ptr addrspace(3) %p7, null
  ret void
}

declare dso_local spir_func ptr addrspace(3) @_Z40__spirv_GenericCastToPtrExplicit_ToLocalPvi(ptr addrspace(4), i32)
