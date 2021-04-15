; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -amdgpu-annotate-kernel-features  %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: define internal void @indirect() #0 {
define internal void @indirect() {
    ret void
}

; GCN-LABEL: define internal void @direct() #1 {
define internal void @direct() {
    %fptr = alloca void()*
    store void()* @indirect, void()** %fptr
    %fp = load void()*, void()** %fptr
    call void %fp()
    ret void
}

; GCN-LABEL: define amdgpu_kernel void @test_direct_indirect_call() #2 {
define amdgpu_kernel void @test_direct_indirect_call() {
  call void @direct()
  ret void
}

; attributes #0 = { "amdgpu-dispatch-id" "amdgpu-dispatch-ptr" "amdgpu-implicitarg-ptr" "amdgpu-work-group-id-x" "amdgpu-work-group-id-y" "amdgpu-work-group-id-z" "amdgpu-work-item-id-x" "amdgpu-work-item-id-y" "amdgpu-work-item-id-z" }
; attributes #1 = { "amdgpu-dispatch-id" "amdgpu-dispatch-ptr" "amdgpu-implicitarg-ptr" "amdgpu-stack-objects" "amdgpu-work-group-id-x" "amdgpu-work-group-id-y" "amdgpu-work-group-id-z" "amdgpu-work-item-id-x" "amdgpu-work-item-id-y" "amdgpu-work-item-id-z" "uniform-work-group-size"="false" }
; attributes #2 = { "amdgpu-calls" "amdgpu-dispatch-id" "amdgpu-dispatch-ptr" "amdgpu-implicitarg-ptr" "amdgpu-work-group-id-x" "amdgpu-work-group-id-y" "amdgpu-work-group-id-z" "amdgpu-work-item-id-x" "amdgpu-work-item-id-y" "amdgpu-work-item-id-z" "uniform-work-group-size"="false" }
