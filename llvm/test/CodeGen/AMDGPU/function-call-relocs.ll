; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=fiji < %s | FileCheck %s

declare void @func(i32 addrspace(1)* %out)

declare protected void @protected_func(i32 addrspace(1)* %out)

declare hidden void @hidden_func(i32 addrspace(1)* %out)

; CHECK-LABEL: call_func:
; CHECK: s_getpc_b64 s{{\[}}[[PC_LO:[0-9]+]]:[[PC_HI:[0-9]+]]{{\]}}
; CHECK: s_add_u32 s[[GOT_ADDR_LO:[0-9]+]], s[[PC_LO]], func@gotpcrel32@lo+4
; CHECK: s_addc_u32 s[[GOT_ADDR_HI:[0-9]+]], s[[PC_HI]], func@gotpcrel32@hi+4
; CHECK: s_load_dwordx2 s{{\[}}[[ADDR_LO:[0-9]+]]:[[ADDR_HI:[0-9]+]]{{\]}}, s{{\[}}[[GOT_ADDR_LO]]:[[GOT_ADDR_HI]]{{\]}}, 0x0
; CHECK: s_swappc_b64 s{{\[}}{{[0-9]+:[0-9]+}}{{\]}}, s{{\[}}[[ADDR_LO]]:[[ADDR_HI]]{{\]}}
define amdgpu_kernel void @call_func(i32 addrspace(1)* %out) {
  call void @func(i32 addrspace(1)* %out)
  ret void
}

; CHECK-LABEL: call_protected_func:
; CHECK: s_getpc_b64 s{{\[}}[[PC_LO:[0-9]+]]:[[PC_HI:[0-9]+]]{{\]}}
; CHECK: s_add_u32 s[[ADDR_LO:[0-9]+]], s[[PC_LO]], protected_func@rel32@lo+4
; CHECK: s_addc_u32 s[[ADDR_HI:[0-9]+]], s[[PC_HI]], protected_func@rel32@hi+4
; CHECK: s_swappc_b64 s{{\[}}{{[0-9]+:[0-9]+}}{{\]}}, s{{\[}}[[ADDR_LO]]:[[ADDR_HI]]{{\]}}
define amdgpu_kernel void @call_protected_func(i32 addrspace(1)* %out) {
  call void @protected_func(i32 addrspace(1)* %out)
  ret void
}

; CHECK-LABEL: call_hidden_func:
; CHECK: s_getpc_b64 s{{\[}}[[PC_LO:[0-9]+]]:[[PC_HI:[0-9]+]]{{\]}}
; CHECK: s_add_u32 s[[ADDR_LO:[0-9]+]], s[[PC_LO]], hidden_func@rel32@lo+4
; CHECK: s_addc_u32 s[[ADDR_HI:[0-9]+]], s[[PC_HI]], hidden_func@rel32@hi+4
; CHECK: s_swappc_b64 s{{\[}}{{[0-9]+:[0-9]+}}{{\]}}, s{{\[}}[[ADDR_LO]]:[[ADDR_HI]]{{\]}}
define amdgpu_kernel void @call_hidden_func(i32 addrspace(1)* %out) {
  call void @hidden_func(i32 addrspace(1)* %out)
  ret void
}

declare i64 @funci()

; CHECK-LABEL: tail_call_func:
; CHECK: s_getpc_b64 s{{\[}}[[PC_LO:[0-9]+]]:[[PC_HI:[0-9]+]]{{\]}}
; CHECK: s_add_u32 s[[GOT_ADDR_LO:[0-9]+]], s[[PC_LO]], funci@gotpcrel32@lo+4
; CHECK: s_addc_u32 s[[GOT_ADDR_HI:[0-9]+]], s[[PC_HI]], funci@gotpcrel32@hi+4
; CHECK: s_load_dwordx2 s{{\[}}[[ADDR_LO:[0-9]+]]:[[ADDR_HI:[0-9]+]]{{\]}}, s{{\[}}[[GOT_ADDR_LO]]:[[GOT_ADDR_HI]]{{\]}}, 0x0
; CHECK: s_setpc_b64 s{{\[}}[[ADDR_LO]]:[[ADDR_HI]]{{\]}}
define i64 @tail_call_func() {
  %ret = tail call i64 @funci()
  ret i64 %ret
}
