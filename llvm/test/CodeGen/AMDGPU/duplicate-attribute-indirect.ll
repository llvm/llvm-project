; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -amdgpu-annotate-kernel-features  %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: define internal void @indirect() #0 {
define internal void @indirect() {
    ret void
}

; GCN-LABEL: define amdgpu_kernel void @test_simple_indirect_call() #1 {
define amdgpu_kernel void @test_simple_indirect_call() #0 {
    %fptr = alloca void()*
    store void()* @indirect, void()** %fptr
    %fp = load void()*, void()** %fptr
    call void %fp()
    ret void
}

attributes #0 = { "amdgpu-dispatch-id" }

; compiler modification to attributes
attributes #0 = { "amdgpu-dispatch-id" "amdgpu-dispatch-ptr" "amdgpu-implicitarg-ptr" "amdgpu-work-group-id-x" "amdgpu-work-group-id-y" "amdgpu-work-group-id-z" "amdgpu-work-item-id-x" "amdgpu-work-item-id-y" "amdgpu-work-item-id-z" }
attributes #1 = { "amdgpu-calls" "amdgpu-dispatch-id" "amdgpu-dispatch-ptr" "amdgpu-implicitarg-ptr" "amdgpu-stack-objects" "amdgpu-work-group-id-x" "amdgpu-work-group-id-y" "amdgpu-work-group-id-z" "amdgpu-work-item-id-x" "amdgpu-work-item-id-y" "amdgpu-work-item-id-z" }

