; RUN: opt -S -mtriple=amdgpu-amd-amdhsa --amdgpu-annotate-uniform < %s | FileCheck -check-prefix=OPT %s
; RUN: opt -S -mtriple=amdgpu-amd-amdhsa -passes=amdgpu-annotate-uniform < %s | FileCheck -check-prefix=OPT %s


; OPT-LABEL: @amdgpu_noclobber_global(
; OPT-NEXT: %load = load i32, ptr addrspace(1) %in, align 4, !amdgpu.noclobber !0
define amdgpu_kernel void @amdgpu_noclobber_global( ptr addrspace(1) %in,  ptr addrspace(1) %out) {
  %load = load i32, ptr addrspace(1) %in, align 4
  store i32 %load, ptr addrspace(1) %out, align 4
  ret void
}

; OPT-LABEL: @amdgpu_noclobber_local(
; OPT-NEXT: %load = load i32, ptr addrspace(3) %in, align 4
define amdgpu_kernel void @amdgpu_noclobber_local( ptr addrspace(3) %in,  ptr addrspace(1) %out) {
  %load = load i32, ptr addrspace(3) %in, align 4
  store i32 %load, ptr addrspace(1) %out, align 4
  ret void
}

; OPT-LABEL: @amdgpu_noclobber_private(
; OPT-NEXT: %load = load i32, ptr addrspace(5) %in, align 4
define amdgpu_kernel void @amdgpu_noclobber_private( ptr addrspace(5) %in,  ptr addrspace(1) %out) {
  %load = load i32, ptr addrspace(5) %in, align 4
  store i32 %load, ptr addrspace(1) %out, align 4
  ret void
}

; OPT-LABEL: @amdgpu_noclobber_flat(
; OPT-NEXT: %load = load i32, ptr addrspace(4) %in, align 4
define amdgpu_kernel void @amdgpu_noclobber_flat( ptr addrspace(4) %in,  ptr addrspace(1) %out) {
  %load = load i32, ptr addrspace(4) %in, align 4
  store i32 %load, ptr addrspace(1) %out, align 4
  ret void
}

; A prefetch does not clobber the memory it prefetches.
; OPT-LABEL: @amdgpu_noclobber_global_after_prefetch(
; OPT-NEXT: call void @llvm.amdgcn.global.prefetch(ptr addrspace(1) %in, i32 8)
; OPT-NEXT: %load = load i32, ptr addrspace(1) %in, align 4, !amdgpu.noclobber !0
define amdgpu_kernel void @amdgpu_noclobber_global_after_prefetch(ptr addrspace(1) %in, ptr addrspace(1) %out) {
  call void @llvm.amdgcn.global.prefetch(ptr addrspace(1) %in, i32 8)
  %load = load i32, ptr addrspace(1) %in, align 4
  store i32 %load, ptr addrspace(1) %out, align 4
  ret void
}
