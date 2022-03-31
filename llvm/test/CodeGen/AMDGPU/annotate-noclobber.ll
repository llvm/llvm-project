; RUN: opt -S --amdgpu-annotate-uniform < %s | FileCheck -check-prefix=OPT %s
target datalayout = "A5"


; OPT-LABEL: @amdgpu_noclobber_global(
; OPT:      %addr = getelementptr i32, i32 addrspace(1)* %in, i64 0, !amdgpu.uniform !0
; OPT-NEXT: %load = load i32, i32 addrspace(1)* %addr, align 4, !amdgpu.noclobber !0
define amdgpu_kernel void @amdgpu_noclobber_global( i32 addrspace(1)* %in,  i32 addrspace(1)* %out) {
entry:
  %addr = getelementptr i32, i32 addrspace(1)* %in, i64 0
  %load = load i32, i32 addrspace(1)* %addr, align 4
  store i32 %load, i32 addrspace(1)* %out, align 4
  ret void
}

; OPT-LABEL: @amdgpu_noclobber_local(
; OPT:      %addr = getelementptr i32, i32 addrspace(3)* %in, i64 0, !amdgpu.uniform !0
; OPT-NEXT: %load = load i32, i32 addrspace(3)* %addr, align 4
define amdgpu_kernel void @amdgpu_noclobber_local( i32 addrspace(3)* %in,  i32 addrspace(1)* %out) {
entry:
  %addr = getelementptr i32, i32 addrspace(3)* %in, i64 0
  %load = load i32, i32 addrspace(3)* %addr, align 4
  store i32 %load, i32 addrspace(1)* %out, align 4
  ret void
}

; OPT-LABEL: @amdgpu_noclobber_private(
; OPT:      %addr = getelementptr i32, i32 addrspace(5)* %in, i64 0, !amdgpu.uniform !0
; OPT-NEXT: %load = load i32, i32 addrspace(5)* %addr, align 4
define amdgpu_kernel void @amdgpu_noclobber_private( i32 addrspace(5)* %in,  i32 addrspace(1)* %out) {
entry:
  %addr = getelementptr i32, i32 addrspace(5)* %in, i64 0
  %load = load i32, i32 addrspace(5)* %addr, align 4
  store i32 %load, i32 addrspace(1)* %out, align 4
  ret void
}

; OPT-LABEL: @amdgpu_noclobber_flat(
; OPT:      %addr = getelementptr i32, i32 addrspace(4)* %in, i64 0, !amdgpu.uniform !0
; OPT-NEXT: %load = load i32, i32 addrspace(4)* %addr, align 4
define amdgpu_kernel void @amdgpu_noclobber_flat( i32 addrspace(4)* %in,  i32 addrspace(1)* %out) {
entry:
  %addr = getelementptr i32, i32 addrspace(4)* %in, i64 0
  %load = load i32, i32 addrspace(4)* %addr, align 4
  store i32 %load, i32 addrspace(1)* %out, align 4
  ret void
}
