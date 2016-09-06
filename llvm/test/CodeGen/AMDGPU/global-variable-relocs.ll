; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=fiji < %s | FileCheck %s

@private = private addrspace(1) global [256 x i32] zeroinitializer
@internal = internal addrspace(1) global [256 x i32] zeroinitializer
@available_externally = available_externally addrspace(1) global [256 x i32] zeroinitializer
@linkonce = linkonce addrspace(1) global [256 x i32] zeroinitializer
@weak= weak addrspace(1) global [256 x i32] zeroinitializer
@common = common addrspace(1) global [256 x i32] zeroinitializer
@extern_weak = extern_weak addrspace(1) global [256 x i32]
@linkonce_odr = linkonce_odr addrspace(1) global [256 x i32] zeroinitializer
@weak_odr = weak_odr addrspace(1) global [256 x i32] zeroinitializer
@external = external addrspace(1) global [256 x i32]
@external_w_init = addrspace(1) global [256 x i32] zeroinitializer

; CHECK-LABEL: private_test:
; CHECK: s_getpc_b64 s{{\[}}[[PC_LO:[0-9]+]]:[[PC_HI:[0-9]+]]{{\]}}
; CHECK: s_add_u32 s[[ADDR_LO:[0-9]+]], s[[PC_LO]], private+8
; CHECK: s_addc_u32 s[[ADDR_HI:[0-9]+]], s[[PC_HI]], 0
; CHECK-DAG: v_mov_b32_e32 v[[V_LO:[0-9]+]], s[[ADDR_LO]]
; CHECK-DAG: v_mov_b32_e32 v[[V_HI:[0-9]+]], s[[ADDR_HI]]
; CHECK: flat_load_dword v{{[0-9]+}}, v{{\[}}[[V_LO]]:[[V_HI]]{{\]}}
define void @private_test(i32 addrspace(1)* %out) {
  %ptr = getelementptr [256 x i32], [256 x i32] addrspace(1)* @private, i32 0, i32 1
  %val = load i32, i32 addrspace(1)* %ptr
  store i32 %val, i32 addrspace(1)* %out
  ret void
}

; CHECK-LABEL: internal_test:
; CHECK: s_getpc_b64 s{{\[}}[[PC_LO:[0-9]+]]:[[PC_HI:[0-9]+]]{{\]}}
; CHECK: s_add_u32 s[[ADDR_LO:[0-9]+]], s[[PC_LO]], internal+8
; CHECK: s_addc_u32 s[[ADDR_HI:[0-9]+]], s[[PC_HI]], 0
; CHECK-DAG: v_mov_b32_e32 v[[V_LO:[0-9]+]], s[[ADDR_LO]]
; CHECK-DAG: v_mov_b32_e32 v[[V_HI:[0-9]+]], s[[ADDR_HI]]
; CHECK: flat_load_dword v{{[0-9]+}}, v{{\[}}[[V_LO]]:[[V_HI]]{{\]}}
define void @internal_test(i32 addrspace(1)* %out) {
  %ptr = getelementptr [256 x i32], [256 x i32] addrspace(1)* @internal, i32 0, i32 1
  %val = load i32, i32 addrspace(1)* %ptr
  store i32 %val, i32 addrspace(1)* %out
  ret void
}

; CHECK-LABEL: available_externally_test:
; CHECK: s_getpc_b64 s{{\[}}[[PC_LO:[0-9]+]]:[[PC_HI:[0-9]+]]{{\]}}
; CHECK: s_add_u32 s[[GOTADDR_LO:[0-9]+]], s[[PC_LO]], available_externally@GOTPCREL+4
; CHECK: s_addc_u32 s[[GOTADDR_HI:[0-9]+]], s[[PC_HI]], 0
; CHECK: s_load_dwordx2 s{{\[}}[[ADDR_LO:[0-9]+]]:[[ADDR_HI:[0-9]+]]{{\]}}, s{{\[}}[[GOTADDR_LO]]:[[GOTADDR_HI]]{{\]}}, 0x0
; CHECK: s_add_u32 s[[GEP_LO:[0-9]+]], s[[ADDR_LO]], 4
; CHECK: s_addc_u32 s[[GEP_HI:[0-9]+]], s[[ADDR_HI]], 0
; CHECK-DAG: v_mov_b32_e32 v[[V_LO:[0-9]+]], s[[GEP_LO]]
; CHECK-DAG: v_mov_b32_e32 v[[V_HI:[0-9]+]], s[[GEP_HI]]
; CHECK: flat_load_dword v{{[0-9]+}}, v{{\[}}[[V_LO]]:[[V_HI]]{{\]}}
define void @available_externally_test(i32 addrspace(1)* %out) {
  %ptr = getelementptr [256 x i32], [256 x i32] addrspace(1)* @available_externally, i32 0, i32 1
  %val = load i32, i32 addrspace(1)* %ptr
  store i32 %val, i32 addrspace(1)* %out
  ret void
}

; CHECK-LABEL: linkonce_test:
; CHECK: s_getpc_b64 s{{\[}}[[PC_LO:[0-9]+]]:[[PC_HI:[0-9]+]]{{\]}}
; CHECK: s_add_u32 s[[GOTADDR_LO:[0-9]+]], s[[PC_LO]], linkonce@GOTPCREL+4
; CHECK: s_addc_u32 s[[GOTADDR_HI:[0-9]+]], s[[PC_HI]], 0
; CHECK: s_load_dwordx2 s{{\[}}[[ADDR_LO:[0-9]+]]:[[ADDR_HI:[0-9]+]]{{\]}}, s{{\[}}[[GOTADDR_LO]]:[[GOTADDR_HI]]{{\]}}, 0x0
; CHECK: s_add_u32 s[[GEP_LO:[0-9]+]], s[[ADDR_LO]], 4
; CHECK: s_addc_u32 s[[GEP_HI:[0-9]+]], s[[ADDR_HI]], 0
; CHECK-DAG: v_mov_b32_e32 v[[V_LO:[0-9]+]], s[[GEP_LO]]
; CHECK-DAG: v_mov_b32_e32 v[[V_HI:[0-9]+]], s[[GEP_HI]]
; CHECK: flat_load_dword v{{[0-9]+}}, v{{\[}}[[V_LO]]:[[V_HI]]{{\]}}
define void @linkonce_test(i32 addrspace(1)* %out) {
  %ptr = getelementptr [256 x i32], [256 x i32] addrspace(1)* @linkonce, i32 0, i32 1
  %val = load i32, i32 addrspace(1)* %ptr
  store i32 %val, i32 addrspace(1)* %out
  ret void
}

; CHECK-LABEL: weak_test:
; CHECK: s_getpc_b64 s{{\[}}[[PC_LO:[0-9]+]]:[[PC_HI:[0-9]+]]{{\]}}
; CHECK: s_add_u32 s[[GOTADDR_LO:[0-9]+]], s[[PC_LO]], weak@GOTPCREL+4
; CHECK: s_addc_u32 s[[GOTADDR_HI:[0-9]+]], s[[PC_HI]], 0
; CHECK: s_load_dwordx2 s{{\[}}[[ADDR_LO:[0-9]+]]:[[ADDR_HI:[0-9]+]]{{\]}}, s{{\[}}[[GOTADDR_LO]]:[[GOTADDR_HI]]{{\]}}, 0x0
; CHECK: s_add_u32 s[[GEP_LO:[0-9]+]], s[[ADDR_LO]], 4
; CHECK: s_addc_u32 s[[GEP_HI:[0-9]+]], s[[ADDR_HI]], 0
; CHECK-DAG: v_mov_b32_e32 v[[V_LO:[0-9]+]], s[[GEP_LO]]
; CHECK-DAG: v_mov_b32_e32 v[[V_HI:[0-9]+]], s[[GEP_HI]]
; CHECK: flat_load_dword v{{[0-9]+}}, v{{\[}}[[V_LO]]:[[V_HI]]{{\]}}
define void @weak_test(i32 addrspace(1)* %out) {
  %ptr = getelementptr [256 x i32], [256 x i32] addrspace(1)* @weak, i32 0, i32 1
  %val = load i32, i32 addrspace(1)* %ptr
  store i32 %val, i32 addrspace(1)* %out
  ret void
}

; CHECK-LABEL: common_test:
; CHECK: s_getpc_b64 s{{\[}}[[PC_LO:[0-9]+]]:[[PC_HI:[0-9]+]]{{\]}}
; CHECK: s_add_u32 s[[GOTADDR_LO:[0-9]+]], s[[PC_LO]], common@GOTPCREL+4
; CHECK: s_addc_u32 s[[GOTADDR_HI:[0-9]+]], s[[PC_HI]], 0
; CHECK: s_load_dwordx2 s{{\[}}[[ADDR_LO:[0-9]+]]:[[ADDR_HI:[0-9]+]]{{\]}}, s{{\[}}[[GOTADDR_LO]]:[[GOTADDR_HI]]{{\]}}, 0x0
; CHECK: s_add_u32 s[[GEP_LO:[0-9]+]], s[[ADDR_LO]], 4
; CHECK: s_addc_u32 s[[GEP_HI:[0-9]+]], s[[ADDR_HI]], 0
; CHECK-DAG: v_mov_b32_e32 v[[V_LO:[0-9]+]], s[[GEP_LO]]
; CHECK-DAG: v_mov_b32_e32 v[[V_HI:[0-9]+]], s[[GEP_HI]]
; CHECK: flat_load_dword v{{[0-9]+}}, v{{\[}}[[V_LO]]:[[V_HI]]{{\]}}
define void @common_test(i32 addrspace(1)* %out) {
  %ptr = getelementptr [256 x i32], [256 x i32] addrspace(1)* @common, i32 0, i32 1
  %val = load i32, i32 addrspace(1)* %ptr
  store i32 %val, i32 addrspace(1)* %out
  ret void
}

; CHECK-LABEL: extern_weak_test:
; CHECK: s_getpc_b64 s{{\[}}[[PC_LO:[0-9]+]]:[[PC_HI:[0-9]+]]{{\]}}
; CHECK: s_add_u32 s[[GOTADDR_LO:[0-9]+]], s[[PC_LO]], extern_weak@GOTPCREL+4
; CHECK: s_addc_u32 s[[GOTADDR_HI:[0-9]+]], s[[PC_HI]], 0
; CHECK: s_load_dwordx2 s{{\[}}[[ADDR_LO:[0-9]+]]:[[ADDR_HI:[0-9]+]]{{\]}}, s{{\[}}[[GOTADDR_LO]]:[[GOTADDR_HI]]{{\]}}, 0x0
; CHECK: s_add_u32 s[[GEP_LO:[0-9]+]], s[[ADDR_LO]], 4
; CHECK: s_addc_u32 s[[GEP_HI:[0-9]+]], s[[ADDR_HI]], 0
; CHECK-DAG: v_mov_b32_e32 v[[V_LO:[0-9]+]], s[[GEP_LO]]
; CHECK-DAG: v_mov_b32_e32 v[[V_HI:[0-9]+]], s[[GEP_HI]]
; CHECK: flat_load_dword v{{[0-9]+}}, v{{\[}}[[V_LO]]:[[V_HI]]{{\]}}
define void @extern_weak_test(i32 addrspace(1)* %out) {
  %ptr = getelementptr [256 x i32], [256 x i32] addrspace(1)* @extern_weak, i32 0, i32 1
  %val = load i32, i32 addrspace(1)* %ptr
  store i32 %val, i32 addrspace(1)* %out
  ret void
}

; CHECK-LABEL: linkonce_odr_test:
; CHECK: s_getpc_b64 s{{\[}}[[PC_LO:[0-9]+]]:[[PC_HI:[0-9]+]]{{\]}}
; CHECK: s_add_u32 s[[GOTADDR_LO:[0-9]+]], s[[PC_LO]], linkonce_odr@GOTPCREL+4
; CHECK: s_addc_u32 s[[GOTADDR_HI:[0-9]+]], s[[PC_HI]], 0
; CHECK: s_load_dwordx2 s{{\[}}[[ADDR_LO:[0-9]+]]:[[ADDR_HI:[0-9]+]]{{\]}}, s{{\[}}[[GOTADDR_LO]]:[[GOTADDR_HI]]{{\]}}, 0x0
; CHECK: s_add_u32 s[[GEP_LO:[0-9]+]], s[[ADDR_LO]], 4
; CHECK: s_addc_u32 s[[GEP_HI:[0-9]+]], s[[ADDR_HI]], 0
; CHECK-DAG: v_mov_b32_e32 v[[V_LO:[0-9]+]], s[[GEP_LO]]
; CHECK-DAG: v_mov_b32_e32 v[[V_HI:[0-9]+]], s[[GEP_HI]]
; CHECK: flat_load_dword v{{[0-9]+}}, v{{\[}}[[V_LO]]:[[V_HI]]{{\]}}
define void @linkonce_odr_test(i32 addrspace(1)* %out) {
  %ptr = getelementptr [256 x i32], [256 x i32] addrspace(1)* @linkonce_odr, i32 0, i32 1
  %val = load i32, i32 addrspace(1)* %ptr
  store i32 %val, i32 addrspace(1)* %out
  ret void
}

; CHECK-LABEL: weak_odr_test:
; CHECK: s_getpc_b64 s{{\[}}[[PC_LO:[0-9]+]]:[[PC_HI:[0-9]+]]{{\]}}
; CHECK: s_add_u32 s[[GOTADDR_LO:[0-9]+]], s[[PC_LO]], weak_odr@GOTPCREL+4
; CHECK: s_addc_u32 s[[GOTADDR_HI:[0-9]+]], s[[PC_HI]], 0
; CHECK: s_load_dwordx2 s{{\[}}[[ADDR_LO:[0-9]+]]:[[ADDR_HI:[0-9]+]]{{\]}}, s{{\[}}[[GOTADDR_LO]]:[[GOTADDR_HI]]{{\]}}, 0x0
; CHECK: s_add_u32 s[[GEP_LO:[0-9]+]], s[[ADDR_LO]], 4
; CHECK: s_addc_u32 s[[GEP_HI:[0-9]+]], s[[ADDR_HI]], 0
; CHECK-DAG: v_mov_b32_e32 v[[V_LO:[0-9]+]], s[[GEP_LO]]
; CHECK-DAG: v_mov_b32_e32 v[[V_HI:[0-9]+]], s[[GEP_HI]]
; CHECK: flat_load_dword v{{[0-9]+}}, v{{\[}}[[V_LO]]:[[V_HI]]{{\]}}
define void @weak_odr_test(i32 addrspace(1)* %out) {
  %ptr = getelementptr [256 x i32], [256 x i32] addrspace(1)* @weak_odr, i32 0, i32 1
  %val = load i32, i32 addrspace(1)* %ptr
  store i32 %val, i32 addrspace(1)* %out
  ret void
}

; CHECK-LABEL: external_test:
; CHECK: s_getpc_b64 s{{\[}}[[PC_LO:[0-9]+]]:[[PC_HI:[0-9]+]]{{\]}}
; CHECK: s_add_u32 s[[GOTADDR_LO:[0-9]+]], s[[PC_LO]], external@GOTPCREL+4
; CHECK: s_addc_u32 s[[GOTADDR_HI:[0-9]+]], s[[PC_HI]], 0
; CHECK: s_load_dwordx2 s{{\[}}[[ADDR_LO:[0-9]+]]:[[ADDR_HI:[0-9]+]]{{\]}}, s{{\[}}[[GOTADDR_LO]]:[[GOTADDR_HI]]{{\]}}, 0x0
; CHECK: s_add_u32 s[[GEP_LO:[0-9]+]], s[[ADDR_LO]], 4
; CHECK: s_addc_u32 s[[GEP_HI:[0-9]+]], s[[ADDR_HI]], 0
; CHECK-DAG: v_mov_b32_e32 v[[V_LO:[0-9]+]], s[[GEP_LO]]
; CHECK-DAG: v_mov_b32_e32 v[[V_HI:[0-9]+]], s[[GEP_HI]]
; CHECK: flat_load_dword v{{[0-9]+}}, v{{\[}}[[V_LO]]:[[V_HI]]{{\]}}
define void @external_test(i32 addrspace(1)* %out) {
  %ptr = getelementptr [256 x i32], [256 x i32] addrspace(1)* @external, i32 0, i32 1
  %val = load i32, i32 addrspace(1)* %ptr
  store i32 %val, i32 addrspace(1)* %out
  ret void
}

; CHECK-LABEL: external_w_init_test:
; CHECK: s_getpc_b64 s{{\[}}[[PC_LO:[0-9]+]]:[[PC_HI:[0-9]+]]{{\]}}
; CHECK: s_add_u32 s[[GOTADDR_LO:[0-9]+]], s[[PC_LO]], external_w_init@GOTPCREL+4
; CHECK: s_addc_u32 s[[GOTADDR_HI:[0-9]+]], s[[PC_HI]], 0
; CHECK: s_load_dwordx2 s{{\[}}[[ADDR_LO:[0-9]+]]:[[ADDR_HI:[0-9]+]]{{\]}}, s{{\[}}[[GOTADDR_LO]]:[[GOTADDR_HI]]{{\]}}, 0x0
; CHECK: s_add_u32 s[[GEP_LO:[0-9]+]], s[[ADDR_LO]], 4
; CHECK: s_addc_u32 s[[GEP_HI:[0-9]+]], s[[ADDR_HI]], 0
; CHECK-DAG: v_mov_b32_e32 v[[V_LO:[0-9]+]], s[[GEP_LO]]
; CHECK-DAG: v_mov_b32_e32 v[[V_HI:[0-9]+]], s[[GEP_HI]]
; CHECK: flat_load_dword v{{[0-9]+}}, v{{\[}}[[V_LO]]:[[V_HI]]{{\]}}
define void @external_w_init_test(i32 addrspace(1)* %out) {
  %ptr = getelementptr [256 x i32], [256 x i32] addrspace(1)* @external_w_init, i32 0, i32 1
  %val = load i32, i32 addrspace(1)* %ptr
  store i32 %val, i32 addrspace(1)* %out
  ret void
}

; CHECK: .local private
; CHECK: .local internal
; CHECK: .weak linkonce
; CHECK: .weak weak
; CHECK: .weak linkonce_odr
; CHECK: .weak weak_odr
; CHECK-NOT: external{{$}}
; CHECK: .globl external_w_init
