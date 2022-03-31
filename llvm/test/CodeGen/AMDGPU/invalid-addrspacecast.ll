; RUN: not llc -global-isel=0 -march=amdgcn -mcpu=bonaire -mattr=-promote-alloca < %s 2>&1 | FileCheck -check-prefix=ERROR %s
; RUN: not llc -global-isel=1 -march=amdgcn -mcpu=bonaire -mattr=-promote-alloca < %s 2>&1 | FileCheck -check-prefix=ERROR %s

; ERROR: error: <unknown>:0:0: in function use_group_to_global_addrspacecast void (i32 addrspace(3)*): invalid addrspacecast
define amdgpu_kernel void @use_group_to_global_addrspacecast(i32 addrspace(3)* %ptr) {
  %stof = addrspacecast i32 addrspace(3)* %ptr to i32 addrspace(1)*
  store volatile i32 0, i32 addrspace(1)* %stof
  ret void
}

; ERROR: error: <unknown>:0:0: in function use_local_to_constant32bit_addrspacecast void (i32 addrspace(3)*): invalid addrspacecast
define amdgpu_kernel void @use_local_to_constant32bit_addrspacecast(i32 addrspace(3)* %ptr) {
  %stof = addrspacecast i32 addrspace(3)* %ptr to i32 addrspace(6)*
  %load = load volatile i32, i32 addrspace(6)* %stof
  ret void
}

; ERROR: error: <unknown>:0:0: in function use_constant32bit_to_local_addrspacecast void (i32 addrspace(6)*): invalid addrspacecast
define amdgpu_kernel void @use_constant32bit_to_local_addrspacecast(i32 addrspace(6)* %ptr) {
  %cast = addrspacecast i32 addrspace(6)* %ptr to i32 addrspace(3)*
  %load = load volatile i32, i32 addrspace(3)* %cast
  ret void
}

; ERROR: error: <unknown>:0:0: in function use_local_to_42_addrspacecast void (i32 addrspace(3)*): invalid addrspacecast
define amdgpu_kernel void @use_local_to_42_addrspacecast(i32 addrspace(3)* %ptr) {
  %cast = addrspacecast i32 addrspace(3)* %ptr to i32 addrspace(42)*
  store volatile i32 addrspace(42)* %cast, i32 addrspace(42)* addrspace(1)* null
  ret void
}

; ERROR: error: <unknown>:0:0: in function use_42_to_local_addrspacecast void (i32 addrspace(42)*): invalid addrspacecast
define amdgpu_kernel void @use_42_to_local_addrspacecast(i32 addrspace(42)* %ptr) {
  %cast = addrspacecast i32 addrspace(42)* %ptr to i32 addrspace(3)*
  %load = load volatile i32, i32 addrspace(3)* %cast
  ret void
}
