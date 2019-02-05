; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}chain_hi_to_lo_private:
; GCN: buffer_load_ushort [[DST:v[0-9]+]], off, [[RSRC:s\[[0-9]+:[0-9]+\]]], [[SOFF:s[0-9]+]] offset:2
; GCN-NEXT: s_waitcnt vmcnt(0)
; GCN-NEXT: buffer_load_short_d16_hi [[DST]], off, [[RSRC]], [[SOFF]]
define <2 x half> @chain_hi_to_lo_private() {
bb:
  %gep_lo = getelementptr inbounds half, half addrspace(5)* null, i64 1
  %load_lo = load half, half addrspace(5)* %gep_lo
  %gep_hi = getelementptr inbounds half, half addrspace(5)* null, i64 0
  %load_hi = load half, half addrspace(5)* %gep_hi

  %temp = insertelement <2 x half> undef, half %load_lo, i32 0
  %result = insertelement <2 x half> %temp, half %load_hi, i32 1

  ret <2 x half> %result
}

; GCN-LABEL: {{^}}chain_hi_to_lo_private_different_bases:
; GCN: buffer_load_ushort [[DST:v[0-9]+]], v{{[0-9]+}}, [[RSRC:s\[[0-9]+:[0-9]+\]]], [[SOFF:s[0-9]+]] offen
; GCN-NEXT: s_waitcnt vmcnt(0)
; GCN-NEXT: buffer_load_short_d16_hi [[DST]], v{{[0-9]+}}, [[RSRC]], [[SOFF]] offen
define <2 x half> @chain_hi_to_lo_private_different_bases(half addrspace(5)* %base_lo, half addrspace(5)* %base_hi) {
bb:
  %load_lo = load half, half addrspace(5)* %base_lo
  %load_hi = load half, half addrspace(5)* %base_hi

  %temp = insertelement <2 x half> undef, half %load_lo, i32 0
  %result = insertelement <2 x half> %temp, half %load_hi, i32 1

  ret <2 x half> %result
}

; GCN-LABEL: {{^}}chain_hi_to_lo_arithmatic:
; GCN: v_add_f16_e32 [[DST:v[0-9]+]], 1.0, v{{[0-9]+}}
; GCN-NEXT: buffer_load_short_d16_hi [[DST]], v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}} offen
define <2 x half> @chain_hi_to_lo_arithmatic(half addrspace(5)* %base, half %in) {
bb:
  %arith_lo = fadd half %in, 1.0
  %load_hi = load half, half addrspace(5)* %base

  %temp = insertelement <2 x half> undef, half %arith_lo, i32 0
  %result = insertelement <2 x half> %temp, half %load_hi, i32 1

  ret <2 x half> %result
}

; GCN-LABEL: {{^}}chain_hi_to_lo_group:
; GCN: ds_read_u16 [[DST:v[0-9]+]], [[ADDR:v[0-9]+]] offset:2
; GCN-NEXT: s_waitcnt lgkmcnt(0)
; GCN-NEXT: ds_read_u16_d16_hi [[DST]], [[ADDR]]
define <2 x half> @chain_hi_to_lo_group() {
bb:
  %gep_lo = getelementptr inbounds half, half addrspace(3)* null, i64 1
  %load_lo = load half, half addrspace(3)* %gep_lo
  %gep_hi = getelementptr inbounds half, half addrspace(3)* null, i64 0
  %load_hi = load half, half addrspace(3)* %gep_hi

  %temp = insertelement <2 x half> undef, half %load_lo, i32 0
  %result = insertelement <2 x half> %temp, half %load_hi, i32 1

  ret <2 x half> %result
}

; GCN-LABEL: {{^}}chain_hi_to_lo_group_different_bases:
; GCN: ds_read_u16 [[DST:v[0-9]+]], v{{[0-9]+}}
; GCN-NEXT: s_waitcnt lgkmcnt(0)
; GCN-NEXT: ds_read_u16_d16_hi [[DST]], v{{[0-9]+}}
define <2 x half> @chain_hi_to_lo_group_different_bases(half addrspace(3)* %base_lo, half addrspace(3)* %base_hi) {
bb:
  %load_lo = load half, half addrspace(3)* %base_lo
  %load_hi = load half, half addrspace(3)* %base_hi

  %temp = insertelement <2 x half> undef, half %load_lo, i32 0
  %result = insertelement <2 x half> %temp, half %load_hi, i32 1

  ret <2 x half> %result
}

; GCN-LABEL: {{^}}chain_hi_to_lo_global:
; GCN: global_load_ushort [[DST:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, off
; GCN: global_load_short_d16_hi [[DST]], v{{\[[0-9]+:[0-9]+\]}}, off
define <2 x half> @chain_hi_to_lo_global() {
bb:
  %gep_lo = getelementptr inbounds half, half addrspace(1)* null, i64 1
  %load_lo = load half, half addrspace(1)* %gep_lo
  %gep_hi = getelementptr inbounds half, half addrspace(1)* null, i64 0
  %load_hi = load half, half addrspace(1)* %gep_hi

  %temp = insertelement <2 x half> undef, half %load_lo, i32 0
  %result = insertelement <2 x half> %temp, half %load_hi, i32 1

  ret <2 x half> %result
}

; GCN-LABEL: {{^}}chain_hi_to_lo_global_different_bases:
; GCN: global_load_ushort [[DST:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, off
; GCN-NEXT: s_waitcnt vmcnt(0)
; GCN-NEXT: global_load_short_d16_hi [[DST]], v{{\[[0-9]+:[0-9]+\]}}, off
define <2 x half> @chain_hi_to_lo_global_different_bases(half addrspace(1)* %base_lo, half addrspace(1)* %base_hi) {
bb:
  %load_lo = load half, half addrspace(1)* %base_lo
  %load_hi = load half, half addrspace(1)* %base_hi

  %temp = insertelement <2 x half> undef, half %load_lo, i32 0
  %result = insertelement <2 x half> %temp, half %load_hi, i32 1

  ret <2 x half> %result
}

; GCN-LABEL: {{^}}chain_hi_to_lo_flat:
; GCN: flat_load_ushort [[DST:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}
; GCN: flat_load_short_d16_hi [[DST]], v{{\[[0-9]+:[0-9]+\]}}
define <2 x half> @chain_hi_to_lo_flat() {
bb:
  %gep_lo = getelementptr inbounds half, half* null, i64 1
  %load_lo = load half, half* %gep_lo
  %gep_hi = getelementptr inbounds half, half* null, i64 0
  %load_hi = load half, half* %gep_hi

  %temp = insertelement <2 x half> undef, half %load_lo, i32 0
  %result = insertelement <2 x half> %temp, half %load_hi, i32 1

  ret <2 x half> %result
}

; GCN-LABEL: {{^}}chain_hi_to_lo_flat_different_bases:
; GCN: flat_load_ushort [[DST:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}
; GCN-NEXT: s_waitcnt vmcnt(0) lgkmcnt(0)
; GCN-NEXT: flat_load_short_d16_hi [[DST]], v{{\[[0-9]+:[0-9]+\]}}
define <2 x half> @chain_hi_to_lo_flat_different_bases(half* %base_lo, half* %base_hi) {
bb:
  %load_lo = load half, half* %base_lo
  %load_hi = load half, half* %base_hi

  %temp = insertelement <2 x half> undef, half %load_lo, i32 0
  %result = insertelement <2 x half> %temp, half %load_hi, i32 1

  ret <2 x half> %result
}
