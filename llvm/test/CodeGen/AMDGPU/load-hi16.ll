; RUN: llc -march=amdgcn -mcpu=gfx900 -amdgpu-sroa=0 -mattr=-promote-alloca -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX9 %s
; RUN: llc -march=amdgcn -mcpu=fiji -amdgpu-sroa=0 -mattr=-promote-alloca -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,VI %s

; GCN-LABEL: {{^}}load_local_hi_v2i16_undeflo:
; GCN: s_waitcnt
; GFX9-NEXT: ds_read_u16_d16_hi v0, v0
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: s_setpc_b64

; VI: ds_read_u16 v
define <2 x i16> @load_local_hi_v2i16_undeflo(i16 addrspace(3)* %in) #0 {
entry:
  %load = load i16, i16 addrspace(3)* %in
  %build = insertelement <2 x i16> undef, i16 %load, i32 1
  ret <2 x i16> %build
}

; GCN-LABEL: {{^}}load_local_hi_v2i16_reglo:
; GCN: s_waitcnt
; GFX9-NEXT: ds_read_u16_d16_hi v1, v0
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: v_mov_b32_e32 v0, v1
; GFX9-NEXT: s_setpc_b64

; VI: ds_read_u16 v
define <2 x i16> @load_local_hi_v2i16_reglo(i16 addrspace(3)* %in, i16 %reg) #0 {
entry:
  %load = load i16, i16 addrspace(3)* %in
  %build0 = insertelement <2 x i16> undef, i16 %reg, i32 0
  %build1 = insertelement <2 x i16> %build0, i16 %load, i32 1
  ret <2 x i16> %build1
}

; Show that we get reasonable regalloc without physreg constraints.
; GCN-LABEL: {{^}}load_local_hi_v2i16_reglo_vreg:
; GCN: s_waitcnt
; GFX9-NEXT: ds_read_u16_d16_hi v1, v0
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: global_store_dword v[0:1], v1, off{{$}}
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: s_setpc_b64

; VI: ds_read_u16 v
define void @load_local_hi_v2i16_reglo_vreg(i16 addrspace(3)* %in, i16 %reg) #0 {
entry:
  %load = load i16, i16 addrspace(3)* %in
  %build0 = insertelement <2 x i16> undef, i16 %reg, i32 0
  %build1 = insertelement <2 x i16> %build0, i16 %load, i32 1
  store <2 x i16> %build1, <2 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}load_local_hi_v2i16_zerolo:
; GCN: s_waitcnt
; GFX9-NEXT: v_mov_b32_e32 v1, 0
; GFX9-NEXT: ds_read_u16_d16_hi v1, v0
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: v_mov_b32_e32 v0, v1
; GFX9-NEXT: s_setpc_b64

; VI: ds_read_u16 v
define <2 x i16> @load_local_hi_v2i16_zerolo(i16 addrspace(3)* %in) #0 {
entry:
  %load = load i16, i16 addrspace(3)* %in
  %build = insertelement <2 x i16> zeroinitializer, i16 %load, i32 1
  ret <2 x i16> %build
}

; FIXME: Remove m0 initialization
; GCN-LABEL: {{^}}load_local_hi_v2i16_zerolo_shift:
; GCN: s_waitcnt
; GFX9-NEXT: ds_read_u16 v0, v0
; GFX9-NEXT: s_waitcnt lgkmcnt(0)
; GFX9-NEXT: v_lshlrev_b32_e32 v0, 16, v0
; GFX9-NEXT: s_setpc_b64

; VI: ds_read_u16 v
; VI: v_lshlrev_b32_e32 v0, 16, v0
define i32 @load_local_hi_v2i16_zerolo_shift(i16 addrspace(3)* %in) #0 {
entry:
  %load = load i16, i16 addrspace(3)* %in
  %zext = zext i16 %load to i32
  %shift = shl i32 %zext, 16
  ret i32 %shift
}

; GCN-LABEL: {{^}}load_local_hi_v2f16_reglo_vreg:
; GCN: s_waitcnt
; GFX9-NEXT: ds_read_u16_d16_hi v1, v0
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: global_store_dword v[0:1], v1, off{{$}}
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: s_setpc_b64

; VI: ds_read_u16 v
define void @load_local_hi_v2f16_reglo_vreg(half addrspace(3)* %in, half %reg) #0 {
entry:
  %load = load half, half addrspace(3)* %in
  %build0 = insertelement <2 x half> undef, half %reg, i32 0
  %build1 = insertelement <2 x half> %build0, half %load, i32 1
  store <2 x half> %build1, <2 x half> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}load_local_hi_v2i16_reglo_vreg_zexti8:
; GCN: s_waitcnt
; GFX9-NEXT: ds_read_u8_d16_hi v1, v0
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: global_store_dword v[0:1], v1, off{{$}}
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: s_setpc_b64

; VI: ds_read_u8 v
define void @load_local_hi_v2i16_reglo_vreg_zexti8(i8 addrspace(3)* %in, i16 %reg) #0 {
entry:
  %load = load i8, i8 addrspace(3)* %in
  %ext = zext i8 %load to i16
  %build0 = insertelement <2 x i16> undef, i16 %reg, i32 0
  %build1 = insertelement <2 x i16> %build0, i16 %ext, i32 1
  store <2 x i16> %build1, <2 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}load_local_hi_v2i16_reglo_vreg_sexti8:
; GCN: s_waitcnt
; GFX9-NEXT: ds_read_i8_d16_hi v1, v0
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: global_store_dword v[0:1], v1, off{{$}}
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: s_setpc_b64

; VI: ds_read_i8 v
define void @load_local_hi_v2i16_reglo_vreg_sexti8(i8 addrspace(3)* %in, i16 %reg) #0 {
entry:
  %load = load i8, i8 addrspace(3)* %in
  %ext = sext i8 %load to i16
  %build0 = insertelement <2 x i16> undef, i16 %reg, i32 0
  %build1 = insertelement <2 x i16> %build0, i16 %ext, i32 1
  store <2 x i16> %build1, <2 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}load_global_hi_v2i16_reglo_vreg:
; GCN: s_waitcnt
; GFX9-NEXT: global_load_short_d16_hi v2, v[0:1], off offset:-4094
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: global_store_dword
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: s_setpc_b64
define void @load_global_hi_v2i16_reglo_vreg(i16 addrspace(1)* %in, i16 %reg) #0 {
entry:
  %gep = getelementptr inbounds i16, i16 addrspace(1)* %in, i64 -2047
  %load = load i16, i16 addrspace(1)* %gep
  %build0 = insertelement <2 x i16> undef, i16 %reg, i32 0
  %build1 = insertelement <2 x i16> %build0, i16 %load, i32 1
  store <2 x i16> %build1, <2 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}load_global_hi_v2f16_reglo_vreg:
; GCN: s_waitcnt
; GFX9-NEXT: global_load_short_d16_hi v2, v[0:1], off offset:-4094
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: global_store_dword
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: s_setpc_b64
define void @load_global_hi_v2f16_reglo_vreg(half addrspace(1)* %in, half %reg) #0 {
entry:
  %gep = getelementptr inbounds half, half addrspace(1)* %in, i64 -2047
  %load = load half, half addrspace(1)* %gep
  %build0 = insertelement <2 x half> undef, half %reg, i32 0
  %build1 = insertelement <2 x half> %build0, half %load, i32 1
  store <2 x half> %build1, <2 x half> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}load_global_hi_v2i16_reglo_vreg_zexti8:
; GCN: s_waitcnt
; GFX9-NEXT: global_load_ubyte_d16_hi v2, v[0:1], off offset:-4095
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: global_store_dword
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: s_setpc_b64
define void @load_global_hi_v2i16_reglo_vreg_zexti8(i8 addrspace(1)* %in, i16 %reg) #0 {
entry:
  %gep = getelementptr inbounds i8, i8 addrspace(1)* %in, i64 -4095
  %load = load i8, i8 addrspace(1)* %gep
  %ext = zext i8 %load to i16
  %build0 = insertelement <2 x i16> undef, i16 %reg, i32 0
  %build1 = insertelement <2 x i16> %build0, i16 %ext, i32 1
  store <2 x i16> %build1, <2 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}load_global_hi_v2i16_reglo_vreg_sexti8:
; GCN: s_waitcnt
; GFX9-NEXT: global_load_sbyte_d16_hi v2, v[0:1], off offset:-4095
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: global_store_dword
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: s_setpc_b64
define void @load_global_hi_v2i16_reglo_vreg_sexti8(i8 addrspace(1)* %in, i16 %reg) #0 {
entry:
  %gep = getelementptr inbounds i8, i8 addrspace(1)* %in, i64 -4095
  %load = load i8, i8 addrspace(1)* %gep
  %ext = sext i8 %load to i16
  %build0 = insertelement <2 x i16> undef, i16 %reg, i32 0
  %build1 = insertelement <2 x i16> %build0, i16 %ext, i32 1
  store <2 x i16> %build1, <2 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: load_flat_hi_v2i16_reglo_vreg:
; GCN: s_waitcnt
; GFX9-NEXT: flat_load_short_d16_hi v2, v[0:1]
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: global_store_dword v[0:1], v2
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: s_setpc_b64

; VI: flat_load_ushort v{{[0-9]+}}
; VI: v_lshlrev_b32_e32 v{{[0-9]+}}, 16,
; VI: v_or_b32_sdwa
define void @load_flat_hi_v2i16_reglo_vreg(i16* %in, i16 %reg) #0 {
entry:
  %load = load i16, i16* %in
  %build0 = insertelement <2 x i16> undef, i16 %reg, i32 0
  %build1 = insertelement <2 x i16> %build0, i16 %load, i32 1
  store <2 x i16> %build1, <2 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}load_flat_hi_v2f16_reglo_vreg:
; GCN: s_waitcnt
; GFX9-NEXT: flat_load_short_d16_hi v2, v[0:1]
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: global_store_dword v[0:1], v2
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: s_setpc_b64

; VI: flat_load_ushort v{{[0-9]+}}
; VI: v_lshlrev_b32_e32 v{{[0-9]+}}, 16,
; VI: v_or_b32_sdwa
define void @load_flat_hi_v2f16_reglo_vreg(half* %in, half %reg) #0 {
entry:
  %load = load half, half* %in
  %build0 = insertelement <2 x half> undef, half %reg, i32 0
  %build1 = insertelement <2 x half> %build0, half %load, i32 1
  store <2 x half> %build1, <2 x half> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}load_flat_hi_v2i16_reglo_vreg_zexti8:
; GCN: s_waitcnt
; GFX9-NEXT: flat_load_ubyte_d16_hi v2, v[0:1]
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: global_store_dword v[0:1], v2
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: s_setpc_b64

; VI: flat_load_ubyte v{{[0-9]+}}
; VI: v_lshlrev_b32_e32 v{{[0-9]+}}, 16,
; VI: v_or_b32_sdwa
define void @load_flat_hi_v2i16_reglo_vreg_zexti8(i8* %in, i16 %reg) #0 {
entry:
  %load = load i8, i8* %in
  %ext = zext i8 %load to i16
  %build0 = insertelement <2 x i16> undef, i16 %reg, i32 0
  %build1 = insertelement <2 x i16> %build0, i16 %ext, i32 1
  store <2 x i16> %build1, <2 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}load_flat_hi_v2i16_reglo_vreg_sexti8:
; GCN: s_waitcnt
; GFX9-NEXT: flat_load_sbyte_d16_hi v2, v[0:1]
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: global_store_dword v[0:1], v2
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: s_setpc_b64

; VI: flat_load_sbyte v{{[0-9]+}}
; VI: v_lshlrev_b32_e32 v{{[0-9]+}}, 16,
; VI: v_or_b32_sdwa
define void @load_flat_hi_v2i16_reglo_vreg_sexti8(i8* %in, i16 %reg) #0 {
entry:
  %load = load i8, i8* %in
  %ext = sext i8 %load to i16
  %build0 = insertelement <2 x i16> undef, i16 %reg, i32 0
  %build1 = insertelement <2 x i16> %build0, i16 %ext, i32 1
  store <2 x i16> %build1, <2 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}load_private_hi_v2i16_reglo_vreg:
; GCN: s_waitcnt
; GFX9: buffer_load_short_d16_hi v0, off, s[0:3], s5 offset:4094{{$}}
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: global_store_dword v{{\[[0-9]+:[0-9]+\]}}, v0
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: s_setpc_b64

; VI: buffer_load_ushort v{{[0-9]+}}, off, s[0:3], s5 offset:4094{{$}}
define void @load_private_hi_v2i16_reglo_vreg(i16 addrspace(5)* byval %in, i16 %reg) #0 {
entry:
  %gep = getelementptr inbounds i16, i16 addrspace(5)* %in, i64 2045
  %load = load i16, i16 addrspace(5)* %gep
  %build0 = insertelement <2 x i16> undef, i16 %reg, i32 0
  %build1 = insertelement <2 x i16> %build0, i16 %load, i32 1
  store <2 x i16> %build1, <2 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}load_private_hi_v2f16_reglo_vreg:
; GCN: s_waitcnt
; GFX9: buffer_load_short_d16_hi v0, off, s[0:3], s5 offset:4094{{$}}
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: global_store_dword v{{\[[0-9]+:[0-9]+\]}}, v0
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: s_setpc_b64

; VI: buffer_load_ushort v{{[0-9]+}}, off, s[0:3], s5 offset:4094{{$}}
define void @load_private_hi_v2f16_reglo_vreg(half addrspace(5)* byval %in, half %reg) #0 {
entry:
  %gep = getelementptr inbounds half, half addrspace(5)* %in, i64 2045
  %load = load half, half addrspace(5)* %gep
  %build0 = insertelement <2 x half> undef, half %reg, i32 0
  %build1 = insertelement <2 x half> %build0, half %load, i32 1
  store <2 x half> %build1, <2 x half> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}load_private_hi_v2i16_reglo_vreg_nooff:
; GCN: s_waitcnt
; GFX9: buffer_load_short_d16_hi v0, off, s[0:3], s4 offset:4094{{$}}
; GFX9: s_waitcnt
; GFX9-NEXT: global_store_dword v{{\[[0-9]+:[0-9]+\]}}, v0
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: s_setpc_b64

; VI: buffer_load_ushort v{{[0-9]+}}, off, s[0:3], s4 offset:4094{{$}}
define void @load_private_hi_v2i16_reglo_vreg_nooff(i16 addrspace(5)* byval %in, i16 %reg) #0 {
entry:
  %load = load volatile i16, i16 addrspace(5)* inttoptr (i32 4094 to i16 addrspace(5)*)
  %build0 = insertelement <2 x i16> undef, i16 %reg, i32 0
  %build1 = insertelement <2 x i16> %build0, i16 %load, i32 1
  store <2 x i16> %build1, <2 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}load_private_hi_v2f16_reglo_vreg_nooff:
; GCN: s_waitcnt
; GFX9-NEXT: buffer_load_short_d16_hi v1, off, s[0:3], s4 offset:4094{{$}}
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: global_store_dword v{{\[[0-9]+:[0-9]+\]}}, v1
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: s_setpc_b64

; VI: buffer_load_ushort v{{[0-9]+}}, off, s[0:3], s4 offset:4094{{$}}
define void @load_private_hi_v2f16_reglo_vreg_nooff(half addrspace(5)* %in, half %reg) #0 {
entry:
  %load = load volatile half, half addrspace(5)* inttoptr (i32 4094 to half addrspace(5)*)
  %build0 = insertelement <2 x half> undef, half %reg, i32 0
  %build1 = insertelement <2 x half> %build0, half %load, i32 1
  store <2 x half> %build1, <2 x half> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}load_private_hi_v2i16_reglo_vreg_zexti8:
; GCN: s_waitcnt
; GFX9: buffer_load_ubyte_d16_hi v0, off, s[0:3], s5 offset:4095{{$}}
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: global_store_dword v{{\[[0-9]+:[0-9]+\]}}, v0
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: s_setpc_b64

; VI: buffer_load_ubyte v{{[0-9]+}}, off, s[0:3], s5 offset:4095{{$}}
define void @load_private_hi_v2i16_reglo_vreg_zexti8(i8 addrspace(5)* byval %in, i16 %reg) #0 {
entry:
  %gep = getelementptr inbounds i8, i8 addrspace(5)* %in, i64 4091
  %load = load i8, i8 addrspace(5)* %gep
  %ext = zext i8 %load to i16
  %build0 = insertelement <2 x i16> undef, i16 %reg, i32 0
  %build1 = insertelement <2 x i16> %build0, i16 %ext, i32 1
  store <2 x i16> %build1, <2 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}load_private_hi_v2i16_reglo_vreg_sexti8:
; GCN: s_waitcnt
; GFX9: buffer_load_sbyte_d16_hi v0, off, s[0:3], s5 offset:4095{{$}}
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: global_store_dword v{{\[[0-9]+:[0-9]+\]}}, v0
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: s_setpc_b64

; VI: buffer_load_sbyte v{{[0-9]+}}, off, s[0:3], s5 offset:4095{{$}}
define void @load_private_hi_v2i16_reglo_vreg_sexti8(i8 addrspace(5)* byval %in, i16 %reg) #0 {
entry:
  %gep = getelementptr inbounds i8, i8 addrspace(5)* %in, i64 4091
  %load = load i8, i8 addrspace(5)* %gep
  %ext = sext i8 %load to i16
  %build0 = insertelement <2 x i16> undef, i16 %reg, i32 0
  %build1 = insertelement <2 x i16> %build0, i16 %ext, i32 1
  store <2 x i16> %build1, <2 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}load_private_hi_v2i16_reglo_vreg_nooff_zexti8:
; GCN: s_waitcnt
; GFX9-NEXT: buffer_load_ubyte_d16_hi v1, off, s[0:3], s4 offset:4094{{$}}
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: global_store_dword v{{\[[0-9]+:[0-9]+\]}}, v1
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: s_setpc_b64

; VI: buffer_load_ubyte v0, off, s[0:3], s4 offset:4094{{$}}
define void @load_private_hi_v2i16_reglo_vreg_nooff_zexti8(i8 addrspace(5)* %in, i16 %reg) #0 {
entry:
  %load = load volatile i8, i8 addrspace(5)* inttoptr (i32 4094 to i8 addrspace(5)*)
  %ext = zext i8 %load to i16
  %build0 = insertelement <2 x i16> undef, i16 %reg, i32 0
  %build1 = insertelement <2 x i16> %build0, i16 %ext, i32 1
  store <2 x i16> %build1, <2 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}load_private_hi_v2i16_reglo_vreg_nooff_sexti8:
; GCN: s_waitcnt
; GFX9-NEXT: buffer_load_sbyte_d16_hi v1, off, s[0:3], s4 offset:4094{{$}}
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: global_store_dword v{{\[[0-9]+:[0-9]+\]}}, v1
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: s_setpc_b64

; VI: buffer_load_sbyte v0, off, s[0:3], s4 offset:4094{{$}}
define void @load_private_hi_v2i16_reglo_vreg_nooff_sexti8(i8 addrspace(5)* %in, i16 %reg) #0 {
entry:
  %load = load volatile i8, i8 addrspace(5)* inttoptr (i32 4094 to i8 addrspace(5)*)
  %ext = sext i8 %load to i16
  %build0 = insertelement <2 x i16> undef, i16 %reg, i32 0
  %build1 = insertelement <2 x i16> %build0, i16 %ext, i32 1
  store <2 x i16> %build1, <2 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}load_private_hi_v2f16_reglo_vreg_nooff_zexti8:
; GCN: s_waitcnt
; GFX9-NEXT: buffer_load_ubyte_d16_hi v1, off, s[0:3], s4 offset:4094{{$}}
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: global_store_dword v{{\[[0-9]+:[0-9]+\]}}, v1
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: s_setpc_b64

; VI: buffer_load_ubyte v0, off, s[0:3], s4 offset:4094{{$}}
define void @load_private_hi_v2f16_reglo_vreg_nooff_zexti8(i8 addrspace(5)* %in, half %reg) #0 {
entry:
  %load = load volatile i8, i8 addrspace(5)* inttoptr (i32 4094 to i8 addrspace(5)*)
  %ext = zext i8 %load to i16
  %bc.ext = bitcast i16 %ext to half
  %build0 = insertelement <2 x half> undef, half %reg, i32 0
  %build1 = insertelement <2 x half> %build0, half %bc.ext, i32 1
  store <2 x half> %build1, <2 x half> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}load_constant_hi_v2i16_reglo_vreg:
; GCN: s_waitcnt
; GFX9-NEXT: global_load_short_d16_hi v2, v[0:1], off offset:-4094
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: global_store_dword
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: s_setpc_b64

; VI: flat_load_ushort
define void @load_constant_hi_v2i16_reglo_vreg(i16 addrspace(4)* %in, i16 %reg) #0 {
entry:
  %gep = getelementptr inbounds i16, i16 addrspace(4)* %in, i64 -2047
  %load = load i16, i16 addrspace(4)* %gep
  %build0 = insertelement <2 x i16> undef, i16 %reg, i32 0
  %build1 = insertelement <2 x i16> %build0, i16 %load, i32 1
  store <2 x i16> %build1, <2 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: load_constant_hi_v2f16_reglo_vreg
; GCN: s_waitcnt
; GFX9-NEXT: global_load_short_d16_hi v2, v[0:1], off offset:-4094
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: global_store_dword
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: s_setpc_b64

; VI: flat_load_ushort
define void @load_constant_hi_v2f16_reglo_vreg(half addrspace(4)* %in, half %reg) #0 {
entry:
  %gep = getelementptr inbounds half, half addrspace(4)* %in, i64 -2047
  %load = load half, half addrspace(4)* %gep
  %build0 = insertelement <2 x half> undef, half %reg, i32 0
  %build1 = insertelement <2 x half> %build0, half %load, i32 1
  store <2 x half> %build1, <2 x half> addrspace(1)* undef
  ret void
}

; Local object gives known offset, so requires converting from offen
; to offset variant.

; GCN-LABEL: {{^}}load_private_hi_v2i16_reglo_vreg_to_offset:
; GFX9: buffer_store_dword
; GFX9-NEXT: buffer_load_short_d16_hi v{{[0-9]+}}, off, s[0:3], s5 offset:4094
define void @load_private_hi_v2i16_reglo_vreg_to_offset(i16 %reg) #0 {
entry:
  %obj0 = alloca [10 x i32], align 4, addrspace(5)
  %obj1 = alloca [4096 x i16], align 2, addrspace(5)
  %bc = bitcast [10 x i32] addrspace(5)* %obj0 to i32 addrspace(5)*
  store volatile i32 123, i32 addrspace(5)* %bc
  %gep = getelementptr inbounds [4096 x i16], [4096 x i16] addrspace(5)* %obj1, i32 0, i32 2025
  %load = load i16, i16 addrspace(5)* %gep
  %build0 = insertelement <2 x i16> undef, i16 %reg, i32 0
  %build1 = insertelement <2 x i16> %build0, i16 %load, i32 1
  store <2 x i16> %build1, <2 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}load_private_hi_v2i16_reglo_vreg_sexti8_to_offset:
; GFX9: buffer_store_dword
; GFX9-NEXT: buffer_load_sbyte_d16_hi v{{[0-9]+}}, off, s[0:3], s5 offset:4095
define void @load_private_hi_v2i16_reglo_vreg_sexti8_to_offset(i16 %reg) #0 {
entry:
  %obj0 = alloca [10 x i32], align 4, addrspace(5)
  %obj1 = alloca [4096 x i8], align 2, addrspace(5)
  %bc = bitcast [10 x i32] addrspace(5)* %obj0 to i32 addrspace(5)*
  store volatile i32 123, i32 addrspace(5)* %bc
  %gep = getelementptr inbounds [4096 x i8], [4096 x i8] addrspace(5)* %obj1, i32 0, i32 4051
  %load = load i8, i8 addrspace(5)* %gep
  %ext = sext i8 %load to i16
  %build0 = insertelement <2 x i16> undef, i16 %reg, i32 0
  %build1 = insertelement <2 x i16> %build0, i16 %ext, i32 1
  store <2 x i16> %build1, <2 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}load_private_hi_v2i16_reglo_vreg_zexti8_to_offset:
; GFX9: buffer_store_dword
; GFX9-NEXT: buffer_load_ubyte_d16_hi v{{[0-9]+}}, off, s[0:3], s5 offset:4095
define void @load_private_hi_v2i16_reglo_vreg_zexti8_to_offset(i16 %reg) #0 {
entry:
  %obj0 = alloca [10 x i32], align 4, addrspace(5)
  %obj1 = alloca [4096 x i8], align 2, addrspace(5)
  %bc = bitcast [10 x i32] addrspace(5)* %obj0 to i32 addrspace(5)*
  store volatile i32 123, i32 addrspace(5)* %bc
  %gep = getelementptr inbounds [4096 x i8], [4096 x i8] addrspace(5)* %obj1, i32 0, i32 4051
  %load = load i8, i8 addrspace(5)* %gep
  %ext = zext i8 %load to i16
  %build0 = insertelement <2 x i16> undef, i16 %reg, i32 0
  %build1 = insertelement <2 x i16> %build0, i16 %ext, i32 1
  store <2 x i16> %build1, <2 x i16> addrspace(1)* undef
  ret void
}

; FIXME: Remove m0 init and waitcnt between reads
; FIXME: Is there a cost to using the extload over not?
; GCN-LABEL: {{^}}load_local_v2i16_split:
; GCN: s_waitcnt
; GFX9-NEXT: ds_read_u16 v1, v0
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: ds_read_u16_d16_hi v1, v0 offset:2
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: v_mov_b32_e32 v0, v1
; GFX9-NEXT: s_setpc_b64
define <2 x i16> @load_local_v2i16_split(i16 addrspace(3)* %in) #0 {
entry:
  %gep = getelementptr inbounds i16, i16 addrspace(3)* %in, i32 1
  %load0 = load volatile i16, i16 addrspace(3)* %in
  %load1 = load volatile i16, i16 addrspace(3)* %gep
  %build0 = insertelement <2 x i16> undef, i16 %load0, i32 0
  %build1 = insertelement <2 x i16> %build0, i16 %load1, i32 1
  ret <2 x i16> %build1
}

; FIXME: Remove waitcnt between reads
; GCN-LABEL: {{^}}load_global_v2i16_split:
; GCN: s_waitcnt
; GFX9-NEXT: global_load_ushort v2
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: global_load_short_d16_hi v2
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: v_mov_b32_e32 v0, v2
; GFX9-NEXT: s_setpc_b64
define <2 x i16> @load_global_v2i16_split(i16 addrspace(1)* %in) #0 {
entry:
  %gep = getelementptr inbounds i16, i16 addrspace(1)* %in, i64 1
  %load0 = load volatile i16, i16 addrspace(1)* %in
  %load1 = load volatile i16, i16 addrspace(1)* %gep
  %build0 = insertelement <2 x i16> undef, i16 %load0, i32 0
  %build1 = insertelement <2 x i16> %build0, i16 %load1, i32 1
  ret <2 x i16> %build1
}

; FIXME: Remove waitcnt between reads
; GCN-LABEL: {{^}}load_flat_v2i16_split:
; GCN: s_waitcnt
; GFX9-NEXT: flat_load_ushort v2
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: flat_load_short_d16_hi v2
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: v_mov_b32_e32 v0, v2
; GFX9-NEXT: s_setpc_b64
define <2 x i16> @load_flat_v2i16_split(i16* %in) #0 {
entry:
  %gep = getelementptr inbounds i16, i16* %in, i64 1
  %load0 = load volatile i16, i16* %in
  %load1 = load volatile i16, i16* %gep
  %build0 = insertelement <2 x i16> undef, i16 %load0, i32 0
  %build1 = insertelement <2 x i16> %build0, i16 %load1, i32 1
  ret <2 x i16> %build1
}

; FIXME: Remove waitcnt between reads
; GCN-LABEL: {{^}}load_constant_v2i16_split:
; GCN: s_waitcnt
; GFX9-NEXT: global_load_ushort v2
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: global_load_short_d16_hi v2
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: v_mov_b32_e32 v0, v2
; GFX9-NEXT: s_setpc_b64
define <2 x i16> @load_constant_v2i16_split(i16 addrspace(4)* %in) #0 {
entry:
  %gep = getelementptr inbounds i16, i16 addrspace(4)* %in, i64 1
  %load0 = load volatile i16, i16 addrspace(4)* %in
  %load1 = load volatile i16, i16 addrspace(4)* %gep
  %build0 = insertelement <2 x i16> undef, i16 %load0, i32 0
  %build1 = insertelement <2 x i16> %build0, i16 %load1, i32 1
  ret <2 x i16> %build1
}

; FIXME: Remove m0 init and waitcnt between reads
; FIXME: Is there a cost to using the extload over not?
; GCN-LABEL: {{^}}load_private_v2i16_split:
; GCN: s_waitcnt
; GFX9: buffer_load_ushort v0, off, s[0:3], s5 offset:4{{$}}
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: buffer_load_short_d16_hi v0, off, s[0:3], s5 offset:6
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: s_setpc_b64
define <2 x i16> @load_private_v2i16_split(i16 addrspace(5)* byval %in) #0 {
entry:
  %gep = getelementptr inbounds i16, i16 addrspace(5)* %in, i32 1
  %load0 = load volatile i16, i16 addrspace(5)* %in
  %load1 = load volatile i16, i16 addrspace(5)* %gep
  %build0 = insertelement <2 x i16> undef, i16 %load0, i32 0
  %build1 = insertelement <2 x i16> %build0, i16 %load1, i32 1
  ret <2 x i16> %build1
}

attributes #0 = { nounwind }
