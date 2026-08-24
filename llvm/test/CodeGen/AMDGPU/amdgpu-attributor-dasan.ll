; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -passes=amdgpu-attributor %s | FileCheck %s

; CHECK-LABEL: define amdgpu_kernel void @k0(
; CHECK-SAME: ) #[[ATTR0:[0-9]+]] {
define amdgpu_kernel void @k0() #0 {
  ret void
}

; CHECK-LABEL: define void @f0(
; CHECK-SAME: ) #[[ATTR0]] {
define void @f0(ptr addrspace(3) %p) #0 {
  store i32 0, ptr addrspace(3) %p
  ret void
}

; CHECK-LABEL: define amdgpu_kernel void @caller(
; CHECK-SAME: ) #[[CALLER:[0-9]+]] {
define amdgpu_kernel void @caller() {
  call void @f0(ptr addrspace(3) null)
  ret void
}

; CHECK-LABEL: define amdgpu_kernel void @k_preannotated(
; CHECK-SAME: ) #[[ATTR0]] {
define amdgpu_kernel void @k_preannotated() #1 {
  ret void
}

attributes #0 = { sanitize_device_address }
attributes #1 = { sanitize_device_address "amdgpu-no-dispatch-ptr" }

; CHECK: attributes #[[ATTR0]] = { sanitize_device_address "amdgpu-no-cluster-id-x" "amdgpu-no-cluster-id-y" "amdgpu-no-cluster-id-z" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-dma" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-no-wwm" }
; CHECK: attributes #[[CALLER]] = {
; CHECK-SAME: "amdgpu-no-dispatch-ptr"
