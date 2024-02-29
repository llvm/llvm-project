; RUN: not llc -global-isel=0 -mtriple=amdgcn -mcpu=bonaire -mattr=-promote-alloca < %s 2>&1 | FileCheck -check-prefix=ERROR %s
; RUN: not llc -global-isel=1 -mtriple=amdgcn -mcpu=bonaire -mattr=-promote-alloca < %s 2>&1 | FileCheck -check-prefix=ERROR %s

; ERROR: error: <unknown>:0:0: in function use_group_to_global_addrspacecast void (ptr addrspace(3)): invalid addrspacecast
define amdgpu_kernel void @use_group_to_global_addrspacecast(ptr addrspace(3) %ptr) {
  %stof = addrspacecast ptr addrspace(3) %ptr to ptr addrspace(1)
  store volatile i32 0, ptr addrspace(1) %stof
  ret void
}

; ERROR: error: <unknown>:0:0: in function use_local_to_constant32bit_addrspacecast void (ptr addrspace(3)): invalid addrspacecast
define amdgpu_kernel void @use_local_to_constant32bit_addrspacecast(ptr addrspace(3) %ptr) {
  %stof = addrspacecast ptr addrspace(3) %ptr to ptr addrspace(6)
  %load = load volatile i32, ptr addrspace(6) %stof
  ret void
}

; ERROR: error: <unknown>:0:0: in function use_constant32bit_to_local_addrspacecast void (ptr addrspace(6)): invalid addrspacecast
define amdgpu_kernel void @use_constant32bit_to_local_addrspacecast(ptr addrspace(6) %ptr) {
  %cast = addrspacecast ptr addrspace(6) %ptr to ptr addrspace(3)
  %load = load volatile i32, ptr addrspace(3) %cast
  ret void
}

; ERROR: error: <unknown>:0:0: in function use_local_to_42_addrspacecast void (ptr addrspace(3)): invalid addrspacecast
define amdgpu_kernel void @use_local_to_42_addrspacecast(ptr addrspace(3) %ptr) {
  %cast = addrspacecast ptr addrspace(3) %ptr to ptr addrspace(42)
  store volatile ptr addrspace(42) %cast, ptr addrspace(1) null
  ret void
}

; ERROR: error: <unknown>:0:0: in function use_42_to_local_addrspacecast void (ptr addrspace(42)): invalid addrspacecast
define amdgpu_kernel void @use_42_to_local_addrspacecast(ptr addrspace(42) %ptr) {
  %cast = addrspacecast ptr addrspace(42) %ptr to ptr addrspace(3)
  %load = load volatile i32, ptr addrspace(3) %cast
  ret void
}
