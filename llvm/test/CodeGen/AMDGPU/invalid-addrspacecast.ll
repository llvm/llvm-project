; RUN: not llc -march=amdgcn -mcpu=bonaire -mattr=-promote-alloca < %s 2>&1 | FileCheck -check-prefix=ERROR %s

; ERROR: error: <unknown>:0:0: in function use_group_to_global_addrspacecast void (i32 addrspace(3)*): invalid addrspacecast
define amdgpu_kernel void @use_group_to_global_addrspacecast(i32 addrspace(3)* %ptr) {
  %stof = addrspacecast i32 addrspace(3)* %ptr to i32 addrspace(1)*
  store volatile i32 0, i32 addrspace(1)* %stof
  ret void
}

; ERROR: error: <unknown>:0:0: in function use_constant32bit_to_flat_addrspacecast void (i32 addrspace(6)*): invalid addrspacecast
define amdgpu_kernel void @use_constant32bit_to_flat_addrspacecast(i32 addrspace(6)* %ptr) #0 {
  %stof = addrspacecast i32 addrspace(6)* %ptr to i32*
  store volatile i32 7, i32* %stof
  ret void
}
