; NOTE: Do not autogenerate. The checks cover target-specific work-item inputs.
; RUN: opt -mtriple=amdgcn-amd-amdpal -mcpu=gfx900 -passes='amdgpu-lds-buffering<max-bytes=64>' -S < %s | FileCheck %s --check-prefix=PAL
; RUN: opt -mtriple=r600-amd-unknown -mcpu=redwood -passes='amdgpu-lds-buffering<max-bytes=64>' -S < %s | FileCheck %s --check-prefix=R600

; PAL-NOT: @llvm.amdgcn.dispatch.ptr
; PAL-LABEL: define amdgpu_kernel void @kernel(
; PAL-NOT: @llvm.amdgcn.dispatch.ptr
; PAL: call{{.*}}i32 @llvm.r600.read.local.size.y()
; PAL: call{{.*}}i32 @llvm.r600.read.local.size.z()
; PAL: call{{.*}}i32 @llvm.amdgcn.workitem.id.x()
; R600-NOT: @kernel.ldsbuf
; R600-LABEL: define amdgpu_kernel void @kernel(
; R600: %value = load <4 x i32>, ptr addrspace(1) %ptr, align 16
; R600: store <4 x i32> %value, ptr addrspace(1) %ptr, align 16
define amdgpu_kernel void @kernel(ptr addrspace(1) %ptr,
                                  ptr addrspace(1) %other) #0 {
  %value = load <4 x i32>, ptr addrspace(1) %ptr, align 16
  store i32 0, ptr addrspace(1) %other, align 4
  store <4 x i32> %value, ptr addrspace(1) %ptr, align 16
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="1,256" }
