; RUN: opt -S --amdgpu-annotate-uniform < %s | FileCheck -check-prefix=OPT %s
target datalayout = "A5"


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
