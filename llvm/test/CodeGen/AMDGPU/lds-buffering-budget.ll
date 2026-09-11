; NOTE: Do not autogenerate. The checks cover LDS allocation rejection only.
; RUN: opt -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -passes='amdgpu-lds-buffering<max-bytes=64>' -S < %s | FileCheck %s --check-prefix=PADDING
; RUN: opt -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -passes='amdgpu-lds-buffering<max-bytes=4194304>' -S < %s | FileCheck %s --check-prefix=OVERFLOW
; RUN: opt -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -passes='amdgpu-lds-buffering<max-bytes=64>' -S < %s | FileCheck %s --check-prefix=EXISTING-OVERFLOW

target triple = "amdgcn-amd-amdhsa"

@padding.used = internal addrspace(3) global [102400 x i8] poison, align 16
@existing_overflow.a = internal addrspace(3) global [2147483700 x i8] poison, align 1
@existing_overflow.b = internal addrspace(3) global [2147483700 x i8] poison, align 1

; The store size of <9 x i32> is 36 bytes, but its allocation size is 64
; bytes. Account for the allocation size of every work-item slot.
; PADDING-NOT: @padding.ldsbuf
; PADDING-LABEL: define amdgpu_kernel void @padding(
; PADDING: %value = load <9 x i32>, ptr addrspace(1) %ptr, align 16
; PADDING: store <9 x i32> %value, ptr addrspace(1) %ptr, align 16
define amdgpu_kernel void @padding(ptr addrspace(1) %ptr) #0 {
  %used = load volatile i8, ptr addrspace(3) @padding.used, align 1
  %value = load <9 x i32>, ptr addrspace(1) %ptr, align 16
  store <9 x i32> %value, ptr addrspace(1) %ptr, align 16
  ret void
}

; The per-work-group allocation does not fit in the target's LDS limit. The
; size calculation must not wrap when max-bytes permits this candidate.
; OVERFLOW-NOT: @overflow.ldsbuf
; OVERFLOW-LABEL: define amdgpu_kernel void @overflow(
; OVERFLOW: %value = load <1048576 x i32>, ptr addrspace(1) %ptr, align 16
; OVERFLOW: store <1048576 x i32> %value, ptr addrspace(1) %ptr, align 16
define amdgpu_kernel void @overflow(ptr addrspace(1) %ptr) #0 {
  %value = load <1048576 x i32>, ptr addrspace(1) %ptr, align 16
  store <1048576 x i32> %value, ptr addrspace(1) %ptr, align 16
  ret void
}

; Accumulating existing LDS allocations must not wrap at 32 bits.
; EXISTING-OVERFLOW-NOT: @existing_overflow.ldsbuf
; EXISTING-OVERFLOW-LABEL: define amdgpu_kernel void @existing_overflow(
; EXISTING-OVERFLOW: %value = load <4 x i32>, ptr addrspace(1) %ptr, align 16
; EXISTING-OVERFLOW: store <4 x i32> %value, ptr addrspace(1) %ptr, align 16
define amdgpu_kernel void @existing_overflow(ptr addrspace(1) %ptr) #0 {
  %used.a = load volatile i8, ptr addrspace(3) @existing_overflow.a, align 1
  %used.b = load volatile i8, ptr addrspace(3) @existing_overflow.b, align 1
  %value = load <4 x i32>, ptr addrspace(1) %ptr, align 16
  store <4 x i32> %value, ptr addrspace(1) %ptr, align 16
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="1024,1024" }
