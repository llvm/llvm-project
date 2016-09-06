; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck %s

; Test for a bug where DAGCombiner::ReassociateOps() was creating adds
; with offset in the first operand and base pointers in the second.

; CHECK-LABEL: {{^}}store_same_base_ptr:
; CHECK: buffer_store_dword v{{[0-9]+}}, [[VADDR:v\[[0-9]+:[0-9]+\]]], [[SADDR:s\[[0-9]+:[0-9]+\]]]
; CHECK: buffer_store_dword v{{[0-9]+}}, [[VADDR]], [[SADDR]]
; CHECK: buffer_store_dword v{{[0-9]+}}, [[VADDR]], [[SADDR]]
; CHECK: buffer_store_dword v{{[0-9]+}}, [[VADDR]], [[SADDR]]

define void @store_same_base_ptr(i32 addrspace(1)* %out) {
entry:
  %id = call i32 @llvm.amdgcn.workitem.id.x() #0
  %offset = sext i32 %id to i64
  %offset0 = add i64 %offset, 1027
  %ptr0 = getelementptr i32, i32 addrspace(1)* %out, i64 %offset0
  store volatile i32 3, i32 addrspace(1)* %ptr0
  %offset1 = add i64 %offset, 1026
  %ptr1 = getelementptr i32, i32 addrspace(1)* %out, i64 %offset1
  store volatile i32 2, i32 addrspace(1)* %ptr1
  %offset2 = add i64 %offset, 1025
  %ptr2 = getelementptr i32, i32 addrspace(1)* %out, i64 %offset2
  store volatile i32 1, i32 addrspace(1)* %ptr2
  %offset3 = add i64 %offset, 1024
  %ptr3 = getelementptr i32, i32 addrspace(1)* %out, i64 %offset3
  store volatile i32 0, i32 addrspace(1)* %ptr3
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #0

attributes #0 = { nounwind readnone }
