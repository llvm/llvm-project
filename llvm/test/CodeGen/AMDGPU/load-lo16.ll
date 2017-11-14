; RUN: llc -march=amdgcn -mcpu=gfx900 -amdgpu-sroa=0 -mattr=-promote-alloca -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX9 %s
; RUN: llc -march=amdgcn -mcpu=fiji -amdgpu-sroa=0 -mattr=-promote-alloca -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,VI %s

; GCN-LABEL: {{^}}load_local_lo_v2i16_undeflo:
; GCN: s_waitcnt
; GFX9-NEXT: ds_read_u16_d16 v0, v0
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: s_setpc_b64

; VI: ds_read_u16
define <2 x i16> @load_local_lo_v2i16_undeflo(i16 addrspace(3)* %in) #0 {
entry:
  %load = load i16, i16 addrspace(3)* %in
  %build = insertelement <2 x i16> undef, i16 %load, i32 0
  ret <2 x i16> %build
}

; GCN-LABEL: {{^}}load_local_lo_v2i16_reglo:
; GCN: s_waitcnt
; GFX9-NEXT: ds_read_u16_d16 v0, v0
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: s_setpc_b64

; VI: ds_read_u16
define <2 x i16> @load_local_lo_v2i16_reglo(i16 addrspace(3)* %in, i16 %reg) #0 {
entry:
  %load = load i16, i16 addrspace(3)* %in
  %build0 = insertelement <2 x i16> undef, i16 %reg, i32 0
  %build1 = insertelement <2 x i16> %build0, i16 %load, i32 0
  ret <2 x i16> %build1
}

; Show that we get reasonable regalloc without physreg constraints.
; GCN-LABEL: {{^}}load_local_lo_v2i16_reglo_vreg:
; GCN: s_waitcnt
; GFX9-NEXT: ds_read_u16_d16 v0, v0
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: global_store_dword v[0:1], v0, off{{$}}
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: s_setpc_b64

; VI: ds_read_u16
define void @load_local_lo_v2i16_reglo_vreg(i16 addrspace(3)* %in, i16 %reg) #0 {
entry:
  %load = load i16, i16 addrspace(3)* %in
  %build0 = insertelement <2 x i16> undef, i16 %reg, i32 0
  %build1 = insertelement <2 x i16> %build0, i16 %load, i32 0
  store <2 x i16> %build1, <2 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}load_local_lo_v2i16_zerolo:
; GCN: s_waitcnt
; GFX9-NEXT: v_mov_b32_e32 v1, 0
; GFX9-NEXT: ds_read_u16_d16 v1, v0
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: v_mov_b32_e32 v0, v1
; GFX9-NEXT: s_setpc_b64

; VI: ds_read_u16 v
define <2 x i16> @load_local_lo_v2i16_zerolo(i16 addrspace(3)* %in) #0 {
entry:
  %load = load i16, i16 addrspace(3)* %in
  %build = insertelement <2 x i16> zeroinitializer, i16 %load, i32 0
  ret <2 x i16> %build
}

; GCN-LABEL: {{^}}load_local_lo_v2f16_fpimm:
; GCN: s_waitcnt
; GFX9-NEXT: v_mov_b32_e32 v1, 2.0
; GFX9-NEXT: ds_read_u16_d16 v1, v0
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: v_mov_b32_e32 v0, v1
; GFX9-NEXT: s_setpc_b64

; VI: ds_read_u16 v
define <2 x half> @load_local_lo_v2f16_fpimm(half addrspace(3)* %in) #0 {
entry:
  %load = load half, half addrspace(3)* %in
  %build = insertelement <2 x half> <half 0.0, half 2.0>, half %load, i32 0
  ret <2 x half> %build
}

; GCN-LABEL: {{^}}load_local_lo_v2f16_reghi_vreg:
; GCN: s_waitcnt
; GFX9-NEXT: ds_read_u16_d16 v1, v0
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: global_store_dword v[0:1], v1, off{{$}}
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: s_setpc_b64

; VI: ds_read_u16 v
define void @load_local_lo_v2f16_reghi_vreg(half addrspace(3)* %in, i32 %reg) #0 {
entry:
  %reg.bc = bitcast i32 %reg to <2 x half>
  %load = load half, half addrspace(3)* %in
  %build1 = insertelement <2 x half> %reg.bc, half %load, i32 0
  store <2 x half> %build1, <2 x half> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}load_local_lo_v2f16_reglo_vreg:

; GFX9: ds_read_u16 v
; GFX9: v_and_b32_e32 v{{[0-9]+}}, 0xffff, v{{[0-9]+}}
; GFX9: v_lshl_or_b32 v{{[0-9]+}}, v{{[0-9]+}}, 16, v{{[0-9]+}}
; GFX9: global_store_dword

; VI: ds_read_u16 v
define void @load_local_lo_v2f16_reglo_vreg(half addrspace(3)* %in, half %reg) #0 {
entry:
  %load = load half, half addrspace(3)* %in
  %build0 = insertelement <2 x half> undef, half %reg, i32 1
  %build1 = insertelement <2 x half> %build0, half %load, i32 0
  store <2 x half> %build1, <2 x half> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}load_local_lo_v2i16_reghi_vreg_zexti8:
; GCN: s_waitcnt
; GFX9-NEXT: ds_read_u8_d16 v1, v0
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: global_store_dword v[0:1], v1, off{{$}}
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: s_setpc_b64

; VI: ds_read_u8 v
define void @load_local_lo_v2i16_reghi_vreg_zexti8(i8 addrspace(3)* %in, i32 %reg) #0 {
entry:
  %reg.bc = bitcast i32 %reg to <2 x i16>
  %load = load i8, i8 addrspace(3)* %in
  %ext = zext i8 %load to i16
  %build1 = insertelement <2 x i16> %reg.bc, i16 %ext, i32 0
  store <2 x i16> %build1, <2 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}load_local_lo_v2i16_reglo_vreg_zexti8:
; GCN: s_waitcnt
; GFX9: ds_read_u8 v
; GFX9: global_store_dword
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: s_setpc_b64

; VI: ds_read_u8 v
define void @load_local_lo_v2i16_reglo_vreg_zexti8(i8 addrspace(3)* %in, i16 %reg) #0 {
entry:
  %load = load i8, i8 addrspace(3)* %in
  %ext = zext i8 %load to i16
  %build0 = insertelement <2 x i16> undef, i16 %reg, i32 1
  %build1 = insertelement <2 x i16> %build0, i16 %ext, i32 0
  store <2 x i16> %build1, <2 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}load_local_lo_v2i16_reghi_vreg_sexti8:
; GCN: s_waitcnt
; GFX9-NEXT: ds_read_i8_d16 v1, v0
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: global_store_dword v[0:1], v1, off{{$}}
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: s_setpc_b64

; VI: ds_read_i8 v
define void @load_local_lo_v2i16_reghi_vreg_sexti8(i8 addrspace(3)* %in, i32 %reg) #0 {
entry:
  %reg.bc = bitcast i32 %reg to <2 x i16>
  %load = load i8, i8 addrspace(3)* %in
  %ext = sext i8 %load to i16
  %build1 = insertelement <2 x i16> %reg.bc, i16 %ext, i32 0
  store <2 x i16> %build1, <2 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}load_local_lo_v2i16_reglo_vreg_sexti8:
; GCN: s_waitcnt
; GFX9: ds_read_i8 v
; GFX9: v_and_b32_e32 v{{[0-9]+}}, 0xffff, v{{[0-9]+}}
; GFX9: v_lshl_or_b32 v{{[0-9]+}}, v{{[0-9]+}}, 16, v{{[0-9]+}}

; VI: ds_read_i8 v
define void @load_local_lo_v2i16_reglo_vreg_sexti8(i8 addrspace(3)* %in, i16 %reg) #0 {
entry:
  %load = load i8, i8 addrspace(3)* %in
  %ext = sext i8 %load to i16
  %build0 = insertelement <2 x i16> undef, i16 %reg, i32 1
  %build1 = insertelement <2 x i16> %build0, i16 %ext, i32 0
  store <2 x i16> %build1, <2 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}load_global_lo_v2i16_reglo_vreg:
; GCN: s_waitcnt
; GFX9-NEXT: global_load_short_d16 v2, v[0:1], off offset:-4094
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: global_store_dword
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: s_setpc_b64
define void @load_global_lo_v2i16_reglo_vreg(i16 addrspace(1)* %in, i32 %reg) #0 {
entry:
  %reg.bc = bitcast i32 %reg to <2 x i16>
  %gep = getelementptr inbounds i16, i16 addrspace(1)* %in, i64 -2047
  %load = load i16, i16 addrspace(1)* %gep
  %build1 = insertelement <2 x i16> %reg.bc, i16 %load, i32 0
  store <2 x i16> %build1, <2 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}load_global_lo_v2f16_reglo_vreg:
; GCN: s_waitcnt
; GFX9-NEXT: global_load_short_d16 v2, v[0:1], off offset:-4094
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: global_store_dword
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: s_setpc_b64
define void @load_global_lo_v2f16_reglo_vreg(half addrspace(1)* %in, i32 %reg) #0 {
entry:
  %reg.bc = bitcast i32 %reg to <2 x half>
  %gep = getelementptr inbounds half, half addrspace(1)* %in, i64 -2047
  %load = load half, half addrspace(1)* %gep
  %build1 = insertelement <2 x half> %reg.bc, half %load, i32 0
  store <2 x half> %build1, <2 x half> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}load_global_lo_v2i16_reglo_vreg_zexti8:
; GCN: s_waitcnt
; GFX9-NEXT: global_load_ubyte_d16 v2, v[0:1], off offset:-4095
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: global_store_dword
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: s_setpc_b64
define void @load_global_lo_v2i16_reglo_vreg_zexti8(i8 addrspace(1)* %in, i32 %reg) #0 {
entry:
  %reg.bc = bitcast i32 %reg to <2 x i16>
  %gep = getelementptr inbounds i8, i8 addrspace(1)* %in, i64 -4095
  %load = load i8, i8 addrspace(1)* %gep
  %ext = zext i8 %load to i16
  %build1 = insertelement <2 x i16> %reg.bc, i16 %ext, i32 0
  store <2 x i16> %build1, <2 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}load_global_lo_v2i16_reglo_vreg_sexti8:
; GCN: s_waitcnt
; GFX9-NEXT: global_load_sbyte_d16 v2, v[0:1], off offset:-4095
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: global_store_dword
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: s_setpc_b64
define void @load_global_lo_v2i16_reglo_vreg_sexti8(i8 addrspace(1)* %in, i32 %reg) #0 {
entry:
  %reg.bc = bitcast i32 %reg to <2 x i16>
  %gep = getelementptr inbounds i8, i8 addrspace(1)* %in, i64 -4095
  %load = load i8, i8 addrspace(1)* %gep
  %ext = sext i8 %load to i16
  %build1 = insertelement <2 x i16> %reg.bc, i16 %ext, i32 0
  store <2 x i16> %build1, <2 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}load_flat_lo_v2i16_reghi_vreg:
; GCN: s_waitcnt
; GFX9-NEXT: flat_load_short_d16 v2, v[0:1]
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: global_store_dword v[0:1], v2
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: s_setpc_b64

; VI: flat_load_ushort v{{[0-9]+}}
; VI: v_or_b32_e32
define void @load_flat_lo_v2i16_reghi_vreg(i16 addrspace(4)* %in, i32 %reg) #0 {
entry:
  %reg.bc = bitcast i32 %reg to <2 x i16>
  %load = load i16, i16 addrspace(4)* %in
  %build1 = insertelement <2 x i16> %reg.bc, i16 %load, i32 0
  store <2 x i16> %build1, <2 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}load_flat_lo_v2f16_reghi_vreg:
; GCN: s_waitcnt
; GFX9-NEXT: flat_load_short_d16 v2, v[0:1]
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: global_store_dword v[0:1], v2
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: s_setpc_b64

; VI: flat_load_ushort v{{[0-9]+}}
; VI: v_or_b32_e32
define void @load_flat_lo_v2f16_reghi_vreg(half addrspace(4)* %in, i32 %reg) #0 {
entry:
  %reg.bc = bitcast i32 %reg to <2 x half>
  %load = load half, half addrspace(4)* %in
  %build1 = insertelement <2 x half> %reg.bc, half %load, i32 0
  store <2 x half> %build1, <2 x half> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}load_flat_lo_v2i16_reglo_vreg_zexti8:
; GCN: s_waitcnt
; GFX9-NEXT: flat_load_ubyte_d16 v2, v[0:1]
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: global_store_dword v[0:1], v2
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: s_setpc_b64

; VI: flat_load_ubyte v{{[0-9]+}}
; VI: v_or_b32_e32
define void @load_flat_lo_v2i16_reglo_vreg_zexti8(i8 addrspace(4)* %in, i32 %reg) #0 {
entry:
  %reg.bc = bitcast i32 %reg to <2 x i16>
  %load = load i8, i8 addrspace(4)* %in
  %ext = zext i8 %load to i16
  %build1 = insertelement <2 x i16> %reg.bc, i16 %ext, i32 0
  store <2 x i16> %build1, <2 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}load_flat_lo_v2i16_reglo_vreg_sexti8:
; GCN: s_waitcnt
; GFX9-NEXT: flat_load_sbyte_d16 v2, v[0:1]
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: global_store_dword v[0:1], v2
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: s_setpc_b64

; VI: flat_load_sbyte v{{[0-9]+}}
; VI: v_or_b32_sdwa v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD

define void @load_flat_lo_v2i16_reglo_vreg_sexti8(i8 addrspace(4)* %in, i32 %reg) #0 {
entry:
  %reg.bc = bitcast i32 %reg to <2 x i16>
  %load = load i8, i8 addrspace(4)* %in
  %ext = sext i8 %load to i16
  %build1 = insertelement <2 x i16> %reg.bc, i16 %ext, i32 0
  store <2 x i16> %build1, <2 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}load_private_lo_v2i16_reglo_vreg:
; GCN: s_waitcnt
; GFX9-NEXT: buffer_load_short_d16 v1, v0, s[0:3], s4 offen offset:4094{{$}}
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: global_store_dword v{{\[[0-9]+:[0-9]+\]}}, v1
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: s_setpc_b64

; VI: buffer_load_ushort v{{[0-9]+}}, v0, s[0:3], s4 offen offset:4094{{$}}
define void @load_private_lo_v2i16_reglo_vreg(i16* %in, i32 %reg) #0 {
entry:
  %reg.bc = bitcast i32 %reg to <2 x i16>
  %gep = getelementptr inbounds i16, i16* %in, i64 2047
  %load = load i16, i16* %gep
  %build1 = insertelement <2 x i16> %reg.bc, i16 %load, i32 0
  store <2 x i16> %build1, <2 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}load_private_lo_v2i16_reghi_vreg:
; GCN: s_waitcnt
; GFX9-NEXT: buffer_load_ushort v{{[0-9]+}}, v{{[0-9]+}}, s[0:3], s4 offen offset:4094{{$}}
; GFX9-NEXT: s_waitcnt
; GFX9: v_and_b32
; GFX9: v_lshl_or_b32

; GFX9-NEXT: global_store_dword v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: s_setpc_b64

; VI: buffer_load_ushort v{{[0-9]+}}, v0, s[0:3], s4 offen offset:4094{{$}}
define void @load_private_lo_v2i16_reghi_vreg(i16* %in, i16 %reg) #0 {
entry:
  %gep = getelementptr inbounds i16, i16* %in, i64 2047
  %load = load i16, i16* %gep
  %build0 = insertelement <2 x i16> undef, i16 %reg, i32 1
  %build1 = insertelement <2 x i16> %build0, i16 %load, i32 0
  store <2 x i16> %build1, <2 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}load_private_lo_v2f16_reglo_vreg:
; GCN: s_waitcnt
; GFX9-NEXT: buffer_load_short_d16 v1, v0, s[0:3], s4 offen offset:4094{{$}}
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: global_store_dword v{{\[[0-9]+:[0-9]+\]}}, v1
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: s_setpc_b64

; VI: buffer_load_ushort v{{[0-9]+}}, v0, s[0:3], s4 offen offset:4094{{$}}
define void @load_private_lo_v2f16_reglo_vreg(half* %in, i32 %reg) #0 {
entry:
  %reg.bc = bitcast i32 %reg to <2 x half>
  %gep = getelementptr inbounds half, half* %in, i64 2047
  %load = load half, half* %gep
  %build1 = insertelement <2 x half> %reg.bc, half %load, i32 0
  store <2 x half> %build1, <2 x half> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}load_private_lo_v2i16_reglo_vreg_nooff:
; GCN: s_waitcnt
; GFX9-NEXT: buffer_load_short_d16 v1, off, s[0:3], s4 offset:4094{{$}}
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: global_store_dword v{{\[[0-9]+:[0-9]+\]}}, v1
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: s_setpc_b64

; VI: buffer_load_ushort v{{[0-9]+}}, off, s[0:3], s4 offset:4094{{$}}
define void @load_private_lo_v2i16_reglo_vreg_nooff(i16* %in, i32 %reg) #0 {
entry:
  %reg.bc = bitcast i32 %reg to <2 x i16>
  %load = load volatile i16, i16* inttoptr (i32 4094 to i16*)
  %build1 = insertelement <2 x i16> %reg.bc, i16 %load, i32 0
  store <2 x i16> %build1, <2 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}load_private_lo_v2i16_reghi_vreg_nooff:
; GCN: s_waitcnt
; GFX9-NEXT: buffer_load_short_d16 v1, off, s[0:3], s4 offset:4094{{$}}
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: global_store_dword v{{\[[0-9]+:[0-9]+\]}}, v1
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: s_setpc_b64

; VI: buffer_load_ushort v{{[0-9]+}}, off, s[0:3], s4 offset:4094{{$}}
define void @load_private_lo_v2i16_reghi_vreg_nooff(i16* %in, i32 %reg) #0 {
entry:
  %reg.bc = bitcast i32 %reg to <2 x i16>
  %load = load volatile i16, i16* inttoptr (i32 4094 to i16*)
  %build1 = insertelement <2 x i16> %reg.bc, i16 %load, i32 0
  store <2 x i16> %build1, <2 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}load_private_lo_v2f16_reglo_vreg_nooff:
; GCN: s_waitcnt
; GFX9-NEXT: buffer_load_short_d16 v1, off, s[0:3], s4 offset:4094{{$}}
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: global_store_dword v{{\[[0-9]+:[0-9]+\]}}, v1
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: s_setpc_b64

; VI: buffer_load_ushort v{{[0-9]+}}, off, s[0:3], s4 offset:4094{{$}}
define void @load_private_lo_v2f16_reglo_vreg_nooff(half* %in, i32 %reg) #0 {
entry:
  %reg.bc = bitcast i32 %reg to <2 x half>
  %load = load volatile half, half* inttoptr (i32 4094 to half*)
  %build1 = insertelement <2 x half> %reg.bc, half %load, i32 0
  store <2 x half> %build1, <2 x half> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}load_private_lo_v2i16_reglo_vreg_zexti8:
; GCN: s_waitcnt
; GFX9-NEXT: buffer_load_ubyte_d16 v1, v0, s[0:3], s4 offen offset:2047{{$}}
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: global_store_dword v{{\[[0-9]+:[0-9]+\]}}, v1
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: s_setpc_b64

; VI: buffer_load_ubyte v{{[0-9]+}}, v0, s[0:3], s4 offen offset:2047{{$}}
define void @load_private_lo_v2i16_reglo_vreg_zexti8(i8* %in, i32 %reg) #0 {
entry:
  %reg.bc = bitcast i32 %reg to <2 x i16>
  %gep = getelementptr inbounds i8, i8* %in, i64 2047
  %load = load i8, i8* %gep
  %ext = zext i8 %load to i16
  %build1 = insertelement <2 x i16> %reg.bc, i16 %ext, i32 0
  store <2 x i16> %build1, <2 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}load_private_lo_v2i16_reglo_vreg_sexti8:
; GCN: s_waitcnt
; GFX9-NEXT: buffer_load_sbyte_d16 v1, v0, s[0:3], s4 offen offset:2047{{$}}
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: global_store_dword v{{\[[0-9]+:[0-9]+\]}}, v1
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: s_setpc_b64

; VI: buffer_load_sbyte v{{[0-9]+}}, v0, s[0:3], s4 offen offset:2047{{$}}
define void @load_private_lo_v2i16_reglo_vreg_sexti8(i8* %in, i32 %reg) #0 {
entry:
  %reg.bc = bitcast i32 %reg to <2 x i16>
  %gep = getelementptr inbounds i8, i8* %in, i64 2047
  %load = load i8, i8* %gep
  %ext = sext i8 %load to i16
  %build1 = insertelement <2 x i16> %reg.bc, i16 %ext, i32 0
  store <2 x i16> %build1, <2 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}load_private_lo_v2i16_reglo_vreg_nooff_zexti8:
; GCN: s_waitcnt
; GFX9-NEXT: buffer_load_ubyte_d16 v1, off, s[0:3], s4 offset:4094{{$}}
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: global_store_dword v{{\[[0-9]+:[0-9]+\]}}, v1
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: s_setpc_b64

; VI: buffer_load_ubyte v0, off, s[0:3], s4 offset:4094{{$}}
define void @load_private_lo_v2i16_reglo_vreg_nooff_zexti8(i8* %in, i32 %reg) #0 {
entry:
  %reg.bc = bitcast i32 %reg to <2 x i16>
  %load = load volatile i8, i8* inttoptr (i32 4094 to i8*)
  %ext = zext i8 %load to i16
  %build1 = insertelement <2 x i16> %reg.bc, i16 %ext, i32 0
  store <2 x i16> %build1, <2 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}load_private_lo_v2i16_reglo_vreg_nooff_sexti8:
; GCN: s_waitcnt
; GFX9-NEXT: buffer_load_sbyte_d16 v1, off, s[0:3], s4 offset:4094{{$}}
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: global_store_dword v{{\[[0-9]+:[0-9]+\]}}, v1
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: s_setpc_b64

; VI: buffer_load_sbyte v0, off, s[0:3], s4 offset:4094{{$}}
define void @load_private_lo_v2i16_reglo_vreg_nooff_sexti8(i8* %in, i32 %reg) #0 {
entry:
  %reg.bc = bitcast i32 %reg to <2 x i16>
  %load = load volatile i8, i8* inttoptr (i32 4094 to i8*)
  %ext = sext i8 %load to i16
  %build1 = insertelement <2 x i16> %reg.bc, i16 %ext, i32 0
  store <2 x i16> %build1, <2 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}load_private_lo_v2f16_reglo_vreg_nooff_zexti8:
; GCN: s_waitcnt
; GFX9-NEXT: buffer_load_ubyte_d16 v1, off, s[0:3], s4 offset:4094{{$}}
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: global_store_dword v{{\[[0-9]+:[0-9]+\]}}, v1
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: s_setpc_b64

; VI: buffer_load_ubyte v0, off, s[0:3], s4 offset:4094{{$}}
define void @load_private_lo_v2f16_reglo_vreg_nooff_zexti8(i8* %in, i32 %reg) #0 {
entry:
  %reg.bc = bitcast i32 %reg to <2 x half>
  %load = load volatile i8, i8* inttoptr (i32 4094 to i8*)
  %ext = zext i8 %load to i16
  %bc.ext = bitcast i16 %ext to half
  %build1 = insertelement <2 x half> %reg.bc, half %bc.ext, i32 0
  store <2 x half> %build1, <2 x half> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}load_constant_lo_v2i16_reglo_vreg:
; GCN: s_waitcnt
; GFX9-NEXT: global_load_short_d16 v2, v[0:1], off offset:-4094
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: global_store_dword
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: s_setpc_b64

; VI: flat_load_ushort
define void @load_constant_lo_v2i16_reglo_vreg(i16 addrspace(2)* %in, i32 %reg) #0 {
entry:
  %reg.bc = bitcast i32 %reg to <2 x i16>
  %gep = getelementptr inbounds i16, i16 addrspace(2)* %in, i64 -2047
  %load = load i16, i16 addrspace(2)* %gep
  %build1 = insertelement <2 x i16> %reg.bc, i16 %load, i32 0
  store <2 x i16> %build1, <2 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: load_constant_lo_v2f16_reglo_vreg
; GCN: s_waitcnt
; GFX9-NEXT: global_load_short_d16 v2, v[0:1], off offset:-4094
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: global_store_dword
; GFX9-NEXT: s_waitcnt
; GFX9-NEXT: s_setpc_b64

; VI: flat_load_ushort
define void @load_constant_lo_v2f16_reglo_vreg(half addrspace(2)* %in, i32 %reg) #0 {
entry:
  %reg.bc = bitcast i32 %reg to <2 x half>
  %gep = getelementptr inbounds half, half addrspace(2)* %in, i64 -2047
  %load = load half, half addrspace(2)* %gep
  %build1 = insertelement <2 x half> %reg.bc, half %load, i32 0
  store <2 x half> %build1, <2 x half> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}load_private_lo_v2i16_reglo_vreg_to_offset:
; GFX9: buffer_store_dword
; GFX9-NEXT: buffer_load_short_d16 v0, off, s[0:3], s5 offset:4094

; VI: buffer_load_ushort v
define void @load_private_lo_v2i16_reglo_vreg_to_offset(i32 %reg) #0 {
entry:
  %obj0 = alloca [10 x i32], align 4
  %obj1 = alloca [4096 x i16], align 2
  %reg.bc = bitcast i32 %reg to <2 x i16>
  %bc = bitcast [10 x i32]* %obj0 to i32*
  store volatile i32 123, i32* %bc
  %gep = getelementptr inbounds [4096 x i16], [4096 x i16]* %obj1, i32 0, i32 2025
  %load = load volatile i16, i16* %gep
  %build1 = insertelement <2 x i16> %reg.bc, i16 %load, i32 0
  store <2 x i16> %build1, <2 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}load_private_lo_v2i16_reglo_vreg_sexti8_to_offset:
; GFX9: buffer_store_dword
; GFX9-NEXT: buffer_load_sbyte_d16 v0, off, s[0:3], s5 offset:4095

; VI: buffer_load_sbyte v
define void @load_private_lo_v2i16_reglo_vreg_sexti8_to_offset(i32 %reg) #0 {
entry:
  %obj0 = alloca [10 x i32], align 4
  %obj1 = alloca [4096 x i8], align 2
  %reg.bc = bitcast i32 %reg to <2 x i16>
  %bc = bitcast [10 x i32]* %obj0 to i32*
  store volatile i32 123, i32* %bc
  %gep = getelementptr inbounds [4096 x i8], [4096 x i8]* %obj1, i32 0, i32 4051
  %load = load volatile i8, i8* %gep
  %load.ext = sext i8 %load to i16
  %build1 = insertelement <2 x i16> %reg.bc, i16 %load.ext, i32 0
  store <2 x i16> %build1, <2 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}load_private_lo_v2i16_reglo_vreg_zexti8_to_offset:
; GFX9: buffer_store_dword
; GFX9-NEXT: buffer_load_ubyte_d16 v0, off, s[0:3], s5 offset:4095

; VI: buffer_load_ubyte v
define void @load_private_lo_v2i16_reglo_vreg_zexti8_to_offset(i32 %reg) #0 {
entry:
  %obj0 = alloca [10 x i32], align 4
  %obj1 = alloca [4096 x i8], align 2
  %reg.bc = bitcast i32 %reg to <2 x i16>
  %bc = bitcast [10 x i32]* %obj0 to i32*
  store volatile i32 123, i32* %bc
  %gep = getelementptr inbounds [4096 x i8], [4096 x i8]* %obj1, i32 0, i32 4051
  %load = load volatile i8, i8* %gep
  %load.ext = zext i8 %load to i16
  %build1 = insertelement <2 x i16> %reg.bc, i16 %load.ext, i32 0
  store <2 x i16> %build1, <2 x i16> addrspace(1)* undef
  ret void
}

attributes #0 = { nounwind }
