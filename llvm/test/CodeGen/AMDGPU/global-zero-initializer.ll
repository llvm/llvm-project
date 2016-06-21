; RUN: llc -march=amdgcn -mcpu=SI < %s 2>&1 | FileCheck %s
; RUN: llc -march=amdgcn -mcpu=tonga < %s 2>&1 | FileCheck %s

; CHECK: {{^}}load_init_global_global:
; CHECK: s_getpc_b64 s{{\[}}[[PC_LO:[0-9]+]]:[[PC_HI:[0-9]+]]{{\]}}
; CHECK: s_add_u32 s[[ADDR_LO:[0-9]+]], s[[PC_LO]], global+4
; CHECK: s_addc_u32 s5, s[[PC_HI]], 0
; CHECK: buffer_load_dword v{{[0-9]+}}, off, s{{\[}}[[ADDR_LO]]:7], 0 offset:40
; CHECK: global:
; CHECK: .zero 1024
@global = addrspace(1) global [256 x i32] zeroinitializer

define void @load_init_global_global(i32 addrspace(1)* %out, i1 %p) {
 %gep = getelementptr [256 x i32], [256 x i32] addrspace(1)* @global, i32 0, i32 10
  %ld = load i32, i32 addrspace(1)* %gep
  store i32 %ld, i32 addrspace(1)* %out
  ret void
}
