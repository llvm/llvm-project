; RUN: llc -march=amdgcn -mcpu=tahiti < %s | FileCheck -check-prefix=FAST64 -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=bonaire < %s | FileCheck -check-prefix=SLOW64 -check-prefix=GCN %s


; lshr (i64 x), c: c > 32 => reg_sequence lshr (i32 hi_32(x)), (c - 32), 0
; GCN-LABEL: {{^}}lshr_i64_35:
; GCN: buffer_load_dword [[VAL:v[0-9]+]]
; GCN: v_lshrrev_b32_e32 v[[LO:[0-9]+]], 3, [[VAL]]
; GCN: v_mov_b32_e32 v[[HI:[0-9]+]], 0{{$}}
; GCN: buffer_store_dwordx2 v{{\[}}[[LO]]:[[HI]]{{\]}}
define void @lshr_i64_35(i64 addrspace(1)* %out, i64 addrspace(1)* %in) {
  %val = load i64, i64 addrspace(1)* %in
  %shl = lshr i64 %val, 35
  store i64 %shl, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}lshr_i64_63:
; GCN: buffer_load_dword [[VAL:v[0-9]+]]
; GCN: v_lshrrev_b32_e32 v[[LO:[0-9]+]], 31, [[VAL]]
; GCN: v_mov_b32_e32 v[[HI:[0-9]+]], 0{{$}}
; GCN: buffer_store_dwordx2 v{{\[}}[[LO]]:[[HI]]{{\]}}
define void @lshr_i64_63(i64 addrspace(1)* %out, i64 addrspace(1)* %in) {
  %val = load i64, i64 addrspace(1)* %in
  %shl = lshr i64 %val, 63
  store i64 %shl, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}lshr_i64_33:
; GCN: buffer_load_dword [[VAL:v[0-9]+]]
; GCN: v_lshrrev_b32_e32 v[[LO:[0-9]+]], 1, [[VAL]]
; GCN: v_mov_b32_e32 v[[HI:[0-9]+]], 0{{$}}
; GCN: buffer_store_dwordx2 v{{\[}}[[LO]]:[[HI]]{{\]}}
define void @lshr_i64_33(i64 addrspace(1)* %out, i64 addrspace(1)* %in) {
  %val = load i64, i64 addrspace(1)* %in
  %shl = lshr i64 %val, 33
  store i64 %shl, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}lshr_i64_32:
; GCN: buffer_load_dword v[[LO:[0-9]+]]
; GCN: v_mov_b32_e32 v[[HI:[0-9]+]], 0{{$}}
; GCN: buffer_store_dwordx2 v{{\[}}[[LO]]:[[HI]]{{\]}}
define void @lshr_i64_32(i64 addrspace(1)* %out, i64 addrspace(1)* %in) {
  %val = load i64, i64 addrspace(1)* %in
  %shl = lshr i64 %val, 32
  store i64 %shl, i64 addrspace(1)* %out
  ret void
}

; Make sure the and of the constant doesn't prevent bfe from forming
; after 64-bit shift is split.

; GCN-LABEL: {{^}}lshr_and_i64_35:
; GCN: buffer_load_dwordx2 v{{\[}}[[LO:[0-9]+]]:[[HI:[0-9]+]]{{\]}}
; GCN: v_bfe_u32 v[[BFE:[0-9]+]], v[[HI]], 8, 23
; GCN: v_mov_b32_e32 v[[ZERO:[0-9]+]], 0{{$}}
; GCN: buffer_store_dwordx2 v{{\[}}[[BFE]]:[[ZERO]]{{\]}}
define void @lshr_and_i64_35(i64 addrspace(1)* %out, i64 addrspace(1)* %in) {
  %val = load i64, i64 addrspace(1)* %in
  %and = and i64 %val, 9223372036854775807 ; 0x7fffffffffffffff
  %shl = lshr i64 %and, 40
  store i64 %shl, i64 addrspace(1)* %out
  ret void
}

; lshl (i64 x), c: c > 32 => reg_sequence lshl 0, (i32 lo_32(x)), (c - 32)

; GCN-LABEL: {{^}}shl_i64_const_35:
; GCN: buffer_load_dword [[VAL:v[0-9]+]]
; GCN: v_lshlrev_b32_e32 v[[HI:[0-9]+]], 3, [[VAL]]
; GCN: v_mov_b32_e32 v[[LO:[0-9]+]], 0{{$}}
; GCN: buffer_store_dwordx2 v{{\[}}[[LO]]:[[HI]]{{\]}}
define void @shl_i64_const_35(i64 addrspace(1)* %out, i64 addrspace(1)* %in) {
  %val = load i64, i64 addrspace(1)* %in
  %shl = shl i64 %val, 35
  store i64 %shl, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}shl_i64_const_32:
; GCN: buffer_load_dword v[[HI:[0-9]+]]
; GCN: v_mov_b32_e32 v[[LO:[0-9]+]], 0{{$}}
; GCN: buffer_store_dwordx2 v{{\[}}[[LO]]:[[HI]]{{\]}}
define void @shl_i64_const_32(i64 addrspace(1)* %out, i64 addrspace(1)* %in) {
  %val = load i64, i64 addrspace(1)* %in
  %shl = shl i64 %val, 32
  store i64 %shl, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}shl_i64_const_63:
; GCN: buffer_load_dword [[VAL:v[0-9]+]]
; GCN: v_lshlrev_b32_e32 v[[HI:[0-9]+]], 31, [[VAL]]
; GCN: v_mov_b32_e32 v[[LO:[0-9]+]], 0{{$}}
; GCN: buffer_store_dwordx2 v{{\[}}[[LO]]:[[HI]]{{\]}}
define void @shl_i64_const_63(i64 addrspace(1)* %out, i64 addrspace(1)* %in) {
  %val = load i64, i64 addrspace(1)* %in
  %shl = shl i64 %val, 63
  store i64 %shl, i64 addrspace(1)* %out
  ret void
}

; ashr (i64 x), 63 => (ashr lo(x), 31), lo(x)

; GCN-LABEL: {{^}}ashr_i64_const_32:
define void @ashr_i64_const_32(i64 addrspace(1)* %out, i64 addrspace(1)* %in) {
  %val = load i64, i64 addrspace(1)* %in
  %shl = ashr i64 %val, 32
  store i64 %shl, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}ashr_i64_const_63:
define void @ashr_i64_const_63(i64 addrspace(1)* %out, i64 addrspace(1)* %in) {
  %val = load i64, i64 addrspace(1)* %in
  %shl = ashr i64 %val, 63
  store i64 %shl, i64 addrspace(1)* %out
  ret void
}
