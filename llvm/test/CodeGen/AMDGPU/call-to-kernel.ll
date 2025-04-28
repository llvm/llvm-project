; RUN: llc -mtriple=amdgcn -mcpu=tahiti -verify-machineinstrs < %s 2>&1 | FileCheck -check-prefix=GCN %s

; GCN: callee_kernel:
; GCN: s_endpgm
; GCN: __amdgpu_callee_kernel_kernel_body
; GCN: s_setpc_b64
define amdgpu_kernel void @callee_kernel(i32 addrspace(1)* %out) #0 {
entry:
  store volatile i32 0, i32 addrspace(1)* %out
  ret void
}

; GCN: caller_kernel:
; GCN: s_getpc_b64 s{{\[}}[[LO1:[0-9]+]]:[[HI1:[0-9]+]]]
; GCN: s_add_u32 s[[LO2:[0-9]+]], s[[LO1]], __amdgpu_callee_kernel_kernel_body@rel32@lo+4
; GCN: s_addc_u32 s[[HI2:[0-9]+]], s[[HI1]], __amdgpu_callee_kernel_kernel_body@rel32@hi+12
; GCN: s_swappc_b64 s[{{[0-9:]+}}], s{{\[}}[[LO2]]:[[HI2]]]
; GCN: s_endpgm
define amdgpu_kernel void @caller_kernel(i32 addrspace(1)* %out) #0 {
entry:
  call amdgpu_kernel void @callee_kernel(i32 addrspace(1)* %out)
  ret void
}

attributes #0 = { nounwind noinline }
