; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; A T*& parameter stored to an alloca must remain be a pointer-to-a-pointer.

; CHECK-DAG: %[[#F32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#PtrF:]] = OpTypePointer Generic %[[#F32]]
; CHECK-DAG: %[[#PtrPtrF:]] = OpTypePointer Generic %[[#PtrF]]
; CHECK-DAG: %[[#FnPtrPtrF:]] = OpTypePointer Function %[[#PtrPtrF]]

; CHECK: OpFunction
; CHECK: %[[#Var:]] = OpVariable %[[#FnPtrPtrF]] Function
; CHECK: %[[#Load:]] = OpLoad %[[#PtrPtrF]] %[[#Var]]
; CHECK: OpStore %[[#Load]] %[[#]]

define spir_func void @set_buf(ptr addrspace(4) %in, ptr addrspace(4) %in_buf) {
entry:
  %in.addr = alloca ptr addrspace(4), align 8
  %in_buf.addr = alloca ptr addrspace(4), align 8
  store ptr addrspace(4) %in, ptr %in.addr, align 8
  store ptr addrspace(4) %in_buf, ptr %in_buf.addr, align 8
  %0 = load ptr addrspace(4), ptr %in.addr, align 8
  %g = getelementptr inbounds float, ptr addrspace(4) %0, i64 1
  %1 = load ptr addrspace(4), ptr %in_buf.addr, align 8
  store ptr addrspace(4) %0, ptr addrspace(4) %1, align 8
  ret void
}
