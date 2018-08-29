; RUN: llc -mtriple=amdgcn -mcpu=gfx700 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX789 %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx803 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX789 %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX789 %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx906 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GCN-DL %s

; GCN-LABEL: {{^}}udot4_acc32:
; GCN:       ; %bb.0: ; %entry
; GCN-NEXT:       s_load_dwordx4 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, {{0x9|0x24}}
; GCN-NEXT:       s_load_dwordx2 s{{\[}}[[SRC2_LO:[0-9]+]]:[[SRC2_HI:[0-9]+]]{{\]}}, s{{\[[0-9]+:[0-9]+\]}}, {{0xd|0x34}}

; GFX789-NEXT:    s_movk_i32 s{{[0-9]+}}, 0xff
; GFX789:         s_load_dword s{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}, 0x0
; GFX789-NEXT:    s_load_dword [[S1:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x0
; GFX789-NEXT:    s_load_dword [[S2:s[0-9]+]], s{{\[}}[[SRC2_LO]]:[[SRC2_HI]]{{\]}}, 0x0
; GFX789-NEXT:    s_waitcnt lgkmcnt(0)
; GFX789-NEXT:    s_and_b32 [[V1E1:s[0-9]+]], s{{[0-9]+}}, s{{[0-9]+}}
; GFX789-NEXT:    s_and_b32 s{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}
; GFX789-NEXT:    s_bfe_u32 s{{[0-9]+}}, s{{[0-9]+}}, 0x80008
; GFX789-NEXT:    v_mov_b32_e32 [[V2E1:v[0-9]+]], s{{[0-9]+}}
; GFX789-NEXT:    v_mov_b32_e32 [[SRC2:v[0-9]+]], s{{[0-9]+}}
; GFX789-NEXT:    s_bfe_u32 s{{[0-9]+}}, s{{[0-9]+}}, 0x80010
; GFX789-NEXT:    v_mad_u32_u24 [[MAD1:v[0-9]+]], [[V1E1]], [[V2E1]], [[SRC2]]

; GFX789-NEXT:    s_bfe_u32 [[V1E2:s[0-9]+]], s{{[0-9]+}}, 0x80008
; GFX789-NEXT:    v_mov_b32_e32 [[V2E2:v[0-9]+]], s{{[0-9]+}}
; GFX789-NEXT:    s_bfe_u32 [[V1E3:s[0-9]+]], s{{[0-9]+}}, 0x80010
; GFX789-NEXT:    v_mad_u32_u24 [[MAD2:v[0-9]+]], [[V1E2]], [[V2E2]], [[MAD1]]

; GFX789-NEXT:    v_mov_b32_e32 [[V2E3:v[0-9]+]], s{{[0-9]+}}
; GFX789-NEXT:    s_lshr_b32 s{{[0-9]+}}, s{{[0-9]+}}, 24
; GFX789-NEXT:    v_mad_u32_u24 [[MAD3:v[0-9]+]], [[V1E3]], [[V2E3]], [[MAD2]]

; GFX789-NEXT:    s_lshr_b32 [[V1E4:s[0-9]+]], s{{[0-9]+}}, 24
; GFX789-NEXT:    v_mov_b32_e32 [[V2E4:v[0-9]+]], s{{[0-9]+}}
; GFX789-NEXT:    v_mad_u32_u24 [[RES:v[0-9]+]], [[V1E4]], [[V2E4]], [[MAD3]]
; GFX789:         {{buffer|flat|global}}_store_dword
; GFX789-NEXT:    s_endpgm

; GCN-DL:         s_waitcnt lgkmcnt(0)
; GCN-DL-NEXT:    s_load_dword [[SRC0:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x0
; GCN-DL-NEXT:    s_load_dword [[S1:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x0
; GCN-DL-NEXT:    s_load_dword [[S2:s[0-9]+]], s{{\[}}[[SRC2_LO]]:[[SRC2_HI]]{{\]}}, 0x0
; GCN-DL-NEXT:    v_mov_b32_e32 v[[STLO:[0-9]+]], s[[SRC2_LO]]
; GCN-DL-NEXT:    v_mov_b32_e32 v[[STHI:[0-9]+]], s[[SRC2_HI]]
; GCN-DL-NEXT:    s_waitcnt lgkmcnt(0)
; GCN-DL-NEXT:    v_mov_b32_e32 [[SRC1:v[0-9]+]], [[S1]]
; GCN-DL-NEXT:    v_mov_b32_e32 [[SRC2:v[0-9]+]], [[S2]]
; GCN-DL-NEXT:    v_dot4_u32_u8 [[DOT:v[0-9]+]], [[SRC0]], [[SRC1]], [[SRC2]]
; GCN-DL-NEXT:    global_store_dword v{{\[}}[[STLO]]:[[STHI]]{{\]}}, [[DOT]], off
; GCN-DL-NEXT:    s_endpgm


define amdgpu_kernel void @udot4_acc32(<4 x i8> addrspace(1)* %src1,
                                       <4 x i8> addrspace(1)* %src2,
                                       i32 addrspace(1)* nocapture %dst) {
entry:
  %vec1 = load <4 x i8>, <4 x i8> addrspace(1)* %src1
  %vec2 = load <4 x i8>, <4 x i8> addrspace(1)* %src2

  %v1e0 = extractelement <4 x i8> %vec1, i64 0
  %cv1e0 = zext i8 %v1e0 to i32
  %v2e0 = extractelement <4 x i8> %vec2, i64 0
  %cv2e0 = zext i8 %v2e0 to i32
  %mul1 = mul nuw nsw i32 %cv1e0, %cv2e0

  %v1e1 = extractelement <4 x i8> %vec1, i64 1
  %cv1e1 = zext i8 %v1e1 to i32
  %v2e1 = extractelement <4 x i8> %vec2, i64 1
  %cv2e1 = zext i8 %v2e1 to i32
  %mul2 = mul nuw nsw i32 %cv1e1, %cv2e1

  %v1e2 = extractelement <4 x i8> %vec1, i64 2
  %cv1e2 = zext i8 %v1e2 to i32
  %v2e2 = extractelement <4 x i8> %vec2, i64 2
  %cv2e2 = zext i8 %v2e2 to i32
  %mul3 = mul nuw nsw i32 %cv1e2, %cv2e2

  %v1e3 = extractelement <4 x i8> %vec1, i64 3
  %cv1e3 = zext i8 %v1e3 to i32
  %v2e3 = extractelement <4 x i8> %vec2, i64 3
  %cv2e3 = zext i8 %v2e3 to i32
  %mul4 = mul nuw nsw i32 %cv1e3, %cv2e3

  %acc = load i32, i32 addrspace(1)* %dst, align 4
  %mad1 = add i32 %mul1, %acc
  %mad2 = add i32 %mad1, %mul2
  %mad3 = add i32 %mad2, %mul3
  %mad4 = add i32 %mad3, %mul4

  store i32 %mad4, i32 addrspace(1)* %dst, align 4
  ret void
}

define amdgpu_kernel void @udot4_acc16(<4 x i8> addrspace(1)* %src1,
; GCN-LABEL: udot4_acc16:
; GCN:       ; %bb.0: ; %entry
; GCN-NEXT:       s_load_dwordx4 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, {{0x9|0x24}}
; GCN-NEXT:       s_load_dwordx2 s{{\[}}[[SRC2_LO:[0-9]+]]:[[SRC2_HI:[0-9]+]]{{\]}}, s{{\[[0-9]+:[0-9]+\]}}, {{0xd|0x34}}

; GFX789:         {{buffer|flat|global}}_load_ushort [[SRC2:v[0-9]+]]
; GFX789:         s_load_dword
; GFX789:         s_waitcnt lgkmcnt(0)
; GFX789:         s_and_b32
; GFX789:         s_bfe_u32 [[V1E2:s[0-9]+]], s{{[0-9]+}}, 0x80008
; GFX789:         s_bfe_u32
; GFX789:         s_bfe_u32
; GFX789-NEXT:    v_mov_b32_e32 [[V2E2:v[0-9]+]], s{{[0-9]+}}
; GFX789-NEXT:    s_bfe_u32 [[V1E3:s[0-9]+]], s{{[0-9]+}}, 0x80010
; GFX789-NEXT:    s_lshr_b32 [[V1E4:s[0-9]+]], s{{[0-9]+}}, 24
; GFX789-NEXT:    v_mov_b32_e32 [[V2E3:v[0-9]+]]
; GFX789-NEXT:    s_lshr_b32 [[V1E4:s[0-9]+]], s{{[0-9]+}}, 24
; GFX789-NEXT:    s_waitcnt vmcnt(0)
; GFX789-NEXT:    v_mad_u32_u24 [[MAD1:v[0-9]+]], {{s[0-9]+}}, {{v[0-9]+}}, [[SRC2]]
; GFX789-NEXT:    v_mad_u32_u24 [[MAD2:v[0-9]+]], {{s[0-9]+}}, [[V2E2]], [[MAD1]]
; GFX789-NEXT:    v_mad_u32_u24 [[MAD3:v[0-9]+]], [[V1E3]], [[V2E3]], [[MAD2]]
; GFX789-NEXT:    v_mov_b32_e32 [[V2E4:v[0-9]+]], s{{[0-9]+}}
; GFX789-NEXT:    v_mad_u32_u24 [[MAD4:v[0-9]+]], [[V1E4]], [[V2E4]], [[MAD3]]
; GFX789-NEXT:    {{buffer|flat|global}}_store_short
; GFX789-NEXT:    s_endpgm

; GCN-DL:         s_waitcnt lgkmcnt(0)
; GCN-DL-NEXT:    s_load_dword [[SRC0:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x0
; GCN-DL-NEXT:    s_load_dword [[S1:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x0
; GCN-DL-NEXT:    v_mov_b32_e32 v[[STLO:[0-9]+]], s[[SRC2_LO]]
; GCN-DL-NEXT:    v_mov_b32_e32 v[[STHI:[0-9]+]], s[[SRC2_HI]]
; GCN-DL-NEXT:    global_load_ushort [[SRC2:v[0-9]+]], v{{\[}}[[STLO]]:[[STHI]]{{\]}}, off
; GCN-DL-NEXT:    s_waitcnt lgkmcnt(0)
; GCN-DL-NEXT:    v_mov_b32_e32 [[SRC1:v[0-9]+]], [[S1]]
; GCN-DL-NEXT:    s_waitcnt vmcnt(0)
; GCN-DL-NEXT:    v_dot4_u32_u8 [[DOT:v[0-9]+]], [[SRC0]], [[SRC1]], [[SRC2]]
; GCN-DL-NEXT:    global_store_short v{{\[}}[[STLO]]:[[STHI]]{{\]}}, [[DOT]], off
; GCN-DL-NEXT:    s_endpgm
                                       <4 x i8> addrspace(1)* %src2,
                                       i16 addrspace(1)* nocapture %dst) {
entry:
  %vec1 = load <4 x i8>, <4 x i8> addrspace(1)* %src1
  %vec2 = load <4 x i8>, <4 x i8> addrspace(1)* %src2

  %v1e0 = extractelement <4 x i8> %vec1, i64 0
  %cv1e0 = zext i8 %v1e0 to i16
  %v2e0 = extractelement <4 x i8> %vec2, i64 0
  %cv2e0 = zext i8 %v2e0 to i16
  %mul1 = mul nuw nsw i16 %cv1e0, %cv2e0

  %v1e1 = extractelement <4 x i8> %vec1, i64 1
  %cv1e1 = zext i8 %v1e1 to i16
  %v2e1 = extractelement <4 x i8> %vec2, i64 1
  %cv2e1 = zext i8 %v2e1 to i16
  %mul2 = mul nuw nsw i16 %cv1e1, %cv2e1

  %v1e2 = extractelement <4 x i8> %vec1, i64 2
  %cv1e2 = zext i8 %v1e2 to i16
  %v2e2 = extractelement <4 x i8> %vec2, i64 2
  %cv2e2 = zext i8 %v2e2 to i16
  %mul3 = mul nuw nsw i16 %cv1e2, %cv2e2

  %v1e3 = extractelement <4 x i8> %vec1, i64 3
  %cv1e3 = zext i8 %v1e3 to i16
  %v2e3 = extractelement <4 x i8> %vec2, i64 3
  %cv2e3 = zext i8 %v2e3 to i16
  %mul4 = mul nuw nsw i16 %cv1e3, %cv2e3

  %acc = load i16, i16 addrspace(1)* %dst, align 2
  %mad1 = add i16 %mul1, %acc
  %mad2 = add i16 %mad1, %mul2
  %mad3 = add i16 %mad2, %mul3
  %mad4 = add i16 %mad3, %mul4

  store i16 %mad4, i16 addrspace(1)* %dst, align 2
  ret void
}

define amdgpu_kernel void @udot4_acc8(<4 x i8> addrspace(1)* %src1,
; GCN-LABEL: udot4_acc8:
; GCN:       ; %bb.0: ; %entry
; GCN-NEXT:     s_load_dwordx4 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, {{0x9|0x24}}
; GCN-NEXT:     s_load_dwordx2 s{{\[}}[[SRC2_LO:[0-9]+]]:[[SRC2_HI:[0-9]+]]{{\]}}, s{{\[[0-9]+:[0-9]+\]}}, {{0xd|0x34}}
; GFX789:       s_movk_i32 s{{[0-9]+}}, 0xff
; GFX789:       s_waitcnt lgkmcnt(0)
; GFX789:       {{buffer|flat|global}}_load_ubyte [[SRC2:v[0-9]+]]
; GFX789:       s_load_dword s{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}
; GFX789:       s_waitcnt lgkmcnt(0)
; GFX789:       s_bfe_u32 [[V1E2:s[0-9]+]], s{{[0-9]+}}, 0x80008
; GFX789:       s_and_b32 s{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}
; GFX789:       s_bfe_u32 s{{[0-9]+}}, s{{[0-9]+}}, 0x80008
; GFX789:       v_mov_b32_e32 [[V2E1:v[0-9]+]]
; GFX789:       s_bfe_u32 s{{[0-9]+}}, s{{[0-9]+}}, 0x80010
; GFX789-NEXT:  v_mov_b32_e32 [[V2E2:v[0-9]+]]
; GFX789-NEXT:  s_bfe_u32 [[V1E3:s[0-9]+]], s{{[0-9]+}}, 0x80010
; GFX789-NEXT:  s_lshr_b32 s{{[0-9]+}}, s{{[0-9]+}}, 24
; GFX789-NEXT:  v_mov_b32_e32 [[V2E3:v[0-9]+]]
; GFX789-NEXT:  s_lshr_b32 [[V1E4:s[0-9]+]], s{{[0-9]+}}, 24
; GFX789-NEXT:  s_waitcnt vmcnt(0)
; GFX789-NEXT:  v_mad_u32_u24 [[MAD1:v[0-9]+]], s{{[0-9]+}}, [[V2E1]], [[SRC2]]
; GFX789-NEXT:  v_mad_u32_u24 [[MAD2:v[0-9]+]], [[V1E2]], [[V2E2]], [[MAD1]]
; GFX789-NEXT:  v_mad_u32_u24 [[MAD3:v[0-9]+]], [[V1E3]], [[V2E3]], [[MAD2]]
; GFX789-NEXT:  v_mov_b32_e32 [[V2E4:v[0-9]+]]
; GFX789-NEXT:  v_mad_u32_u24 [[MAD4:v[0-9]+]], [[V1E4]], [[V2E4]], [[MAD3]]
; GFX789-NEXT:  {{buffer|flat|global}}_store_byte
; GFX789-NEXT:  s_endpgm

; GCN-DL:       s_waitcnt lgkmcnt(0)
; GCN-DL-NEXT:  s_load_dword [[SRC0:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}
; GCN-DL-NEXT:  s_load_dword [[S1:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}
; GCN-DL-NEXT:  v_mov_b32_e32 v[[STLO:[0-9]+]], s[[SRC2_LO]]
; GCN-DL-NEXT:  v_mov_b32_e32 v[[STHI:[0-9]+]], s[[SRC2_HI]]
; GCN-DL-NEXT:  global_load_ubyte [[SRC2:v[0-9]+]], v{{\[}}[[STLO]]:[[STHI]]{{\]}}, off
; GCN-DL-NEXT:  s_waitcnt lgkmcnt(0)
; GCN-DL-NEXT:  v_mov_b32_e32 [[SRC1:v[0-9]+]], [[S1]]
; GCN-DL-NEXT:  s_waitcnt vmcnt(0)
; GCN-DL-NEXT:  v_dot4_u32_u8 [[DOT:v[0-9]+]], [[SRC0]], [[SRC1]], [[SRC2]]
; GCN-DL-NEXT:  global_store_byte v{{\[}}[[STLO]]:[[STHI]]{{\]}}, [[DOT]], off
; GCN-DL-NEXT:  s_endpgm
                                      <4 x i8> addrspace(1)* %src2,
                                      i8 addrspace(1)* nocapture %dst) {
entry:
  %vec1 = load <4 x i8>, <4 x i8> addrspace(1)* %src1
  %vec2 = load <4 x i8>, <4 x i8> addrspace(1)* %src2

  %v1e0 = extractelement <4 x i8> %vec1, i64 0
  %v2e0 = extractelement <4 x i8> %vec2, i64 0
  %mul1 = mul nuw nsw i8 %v1e0, %v2e0

  %v1e1 = extractelement <4 x i8> %vec1, i64 1
  %v2e1 = extractelement <4 x i8> %vec2, i64 1
  %mul2 = mul nuw nsw i8 %v1e1, %v2e1

  %v1e2 = extractelement <4 x i8> %vec1, i64 2
  %v2e2 = extractelement <4 x i8> %vec2, i64 2
  %mul3 = mul nuw nsw i8 %v1e2, %v2e2

  %v1e3 = extractelement <4 x i8> %vec1, i64 3
  %v2e3 = extractelement <4 x i8> %vec2, i64 3
  %mul4 = mul nuw nsw i8 %v1e3, %v2e3

  %acc = load i8, i8 addrspace(1)* %dst, align 2
  %mad1 = add i8 %mul1, %acc
  %mad2 = add i8 %mad1, %mul2
  %mad3 = add i8 %mad2, %mul3
  %mad4 = add i8 %mad3, %mul4

  store i8 %mad4, i8 addrspace(1)* %dst, align 2
  ret void
}

; TODO: Generate udot4?
define amdgpu_kernel void @udot2_8(<4 x i8> addrspace(1)* %src1,
; GCN-LABEL: udot2_8:
; GCN-NEXT:   ; %bb.0: ; %entry
; GCN-NEXT:    s_load_dwordx4 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, {{0x9|0x24}}
; GCN-NEXT:    s_load_dwordx2 s{{\[}}[[SRC2_LO:[0-9]+]]:[[SRC2_HI:[0-9]+]]{{\]}}
; GCN:         s_movk_i32 [[FF:s[0-9]+]], 0xff
; GCN-NEXT:    s_waitcnt lgkmcnt(0)
; GCN:         s_load_dword [[V1:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x0
; GCN:         s_load_dword [[V2:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x0
; GCN-NEXT:    s_waitcnt lgkmcnt(0)
; GCN:         s_and_b32 [[V1E1:s[0-9]+]], [[V1]], [[FF]]
; GCN:         s_bfe_u32 [[VE2:s[0-9]+]], {{s[0-9]+}}, 0x80008
; GCN:         s_bfe_u32 [[V1E2:s[0-9]+]], {{s[0-9]+}}, 0x80008
; GCN-NEXT:    s_waitcnt vmcnt(0)
; GCN-NEXT:    v_mad_u32_u24 [[MAD1:v[0-9]+]], [[V1E1]]
; GCN-NEXT:    v_mov_b32_e32 [[V2E2:v[0-9]+]]
; GCN-NEXT:    v_mad_u32_u24 [[MAD2:v[0-9]+]], {{s[0-9]+}}, [[V2E2]], [[MAD1]]
; GCN-NEXT:    {{buffer|flat|global}}_store_byte
; GCN-NEXT:    s_endpgm
                                 <4 x i8> addrspace(1)* %src2,
                                 i8 addrspace(1)* nocapture %dst) {
entry:
  %vec1 = load <4 x i8>, <4 x i8> addrspace(1)* %src1
  %vec2 = load <4 x i8>, <4 x i8> addrspace(1)* %src2

  %v1e0 = extractelement <4 x i8> %vec1, i64 0
  %v2e0 = extractelement <4 x i8> %vec2, i64 0
  %mul1 = mul nuw nsw i8 %v1e0, %v2e0

  %v1e1 = extractelement <4 x i8> %vec1, i64 1
  %v2e1 = extractelement <4 x i8> %vec2, i64 1
  %mul2 = mul nuw nsw i8 %v1e1, %v2e1

  %acc = load i8, i8 addrspace(1)* %dst, align 2
  %mad1 = add i8 %mul1, %acc
  %mad2 = add i8 %mad1, %mul2
  store i8 %mad2, i8 addrspace(1)* %dst, align 2
  ret void
}

define amdgpu_kernel void @udot4_CommutationInsideMAD(<4 x i8> addrspace(1)* %src1,
; GCN-LABEL: udot4_CommutationInsideMAD:
; GCN:       ; %bb.0: ; %entry
; GCN-NEXT:    s_load_dwordx4 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, {{0x9|0x24}}
; GCN-NEXT:    s_load_dwordx2 s{{\[}}[[SRC2_LO:[0-9]+]]:[[SRC2_HI:[0-9]+]]{{\]}}, s{{\[[0-9]+:[0-9]+\]}}, {{0xd|0x34}}
; GFX789:      s_waitcnt lgkmcnt(0)
; GFX789:      {{buffer|flat|global}}_load_ubyte [[SRC2:v[0-9]+]]
; GFX789:      s_load_dword s{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}
; GFX789:      s_waitcnt lgkmcnt(0)
; GFX789:      s_bfe_u32
; GFX789:      s_bfe_u32
; GFX789-NEXT: s_bfe_u32 [[V1E2:s[0-9]+]], s{{[0-9]+}}, 0x80008
; GFX789-NEXT: v_mov_b32_e32 [[V2E2:v[0-9]+]]
; GFX789-NEXT: s_bfe_u32 [[V1E3:s[0-9]+]], s{{[0-9]+}}, 0x80010
; GFX789-NEXT: s_lshr_b32 [[VE4:s[0-9]+]], s{{[0-9]+}}, 24
; GFX789-NEXT: v_mov_b32_e32 [[V2E3:v[0-9]+]]
; GFX789-NEXT: s_lshr_b32 [[V1E4:s[0-9]+]], s{{[0-9]+}}, 24
; GFX789-NEXT: s_waitcnt vmcnt(0)

; GFX789-NEXT: v_mad_u32_u24 [[MAD1:v[0-9]+]],  s{{[0-9]+}},  v{{[0-9]+}}, [[SRC2]]
; GFX789-NEXT: v_mad_u32_u24 [[MAD2:v[0-9]+]], [[V1E2]],  [[V2E2]], [[MAD1]]
; GFX789-NEXT: v_mad_u32_u24 [[MAD3:v[0-9]+]], [[V1E3]], [[V2E3]], [[MAD2]]
; GFX789-NEXT: v_mov_b32_e32 [[V2E4:v[0-9]+]], [[VE4]]
; GFX789-NEXT: v_mad_u32_u24 [[MAD4:v[0-9]+]], [[V1E4]], [[V2E4]], [[MAD3]]
; GFX789-NEXT: {{buffer|flat|global}}_store_byte
; GFX789-NEXT: s_endpgm

; GCN-DL:      s_waitcnt lgkmcnt(0)
; GCN-DL-NEXT: s_load_dword [[S1:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}
; GCN-DL-NEXT: s_load_dword [[SRC0:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}
; GCN-DL-NEXT: v_mov_b32_e32 v[[STLO:[0-9]+]], s[[SRC2_LO]]
; GCN-DL-NEXT: v_mov_b32_e32 v[[STHI:[0-9]+]], s[[SRC2_HI]]
; GCN-DL-NEXT: global_load_ubyte [[SRC2:v[0-9]+]], v{{\[}}[[STLO]]:[[STHI]]{{\]}}, off
; GCN-DL-NEXT: s_waitcnt lgkmcnt(0)
; GCN-DL-NEXT: v_mov_b32_e32 [[SRC1:v[0-9]+]], [[S1]]
; GCN-DL-NEXT: s_waitcnt vmcnt(0)
; GCN-DL-NEXT: v_dot4_u32_u8 [[DOT:v[0-9]+]], [[SRC0]], [[SRC1]], [[SRC2]]
; GCN-DL-NEXT: global_store_byte v{{\[}}[[STLO]]:[[STHI]]{{\]}}, [[DOT]], off
; GCN-DL-NEXT: s_endpgm
                                              <4 x i8> addrspace(1)* %src2,
                                              i8 addrspace(1)* nocapture %dst) {
entry:
  %vec1 = load <4 x i8>, <4 x i8> addrspace(1)* %src1
  %vec2 = load <4 x i8>, <4 x i8> addrspace(1)* %src2

  %v1e0 = extractelement <4 x i8> %vec1, i64 0
  %v2e0 = extractelement <4 x i8> %vec2, i64 0
  %mul1 = mul nuw nsw i8 %v2e0, %v1e0

  %v1e1 = extractelement <4 x i8> %vec1, i64 1
  %v2e1 = extractelement <4 x i8> %vec2, i64 1
  %mul2 = mul nuw nsw i8 %v2e1, %v1e1

  %v1e2 = extractelement <4 x i8> %vec1, i64 2
  %v2e2 = extractelement <4 x i8> %vec2, i64 2
  %mul3 = mul nuw nsw i8 %v2e2, %v1e2

  %v1e3 = extractelement <4 x i8> %vec1, i64 3
  %v2e3 = extractelement <4 x i8> %vec2, i64 3
  %mul4 = mul nuw nsw i8 %v2e3, %v1e3

  %acc = load i8, i8 addrspace(1)* %dst, align 2
  %mad1 = add i8 %acc, %mul1
  %mad2 = add i8 %mul2, %mad1
  %mad3 = add i8 %mul3, %mad2
  %mad4 = add i8 %mul4, %mad3

  store i8 %mad4, i8 addrspace(1)* %dst, align 2
  ret void
}

; TODO: Support commutation accross the adds.
define amdgpu_kernel void @udot4_CommutationAccrossMADs(<4 x i8> addrspace(1)* %src1,
; GCN-LABEL: udot4_CommutationAccrossMADs:
; GCN:       ; %bb.0: ; %entry
; GCN-NEXT:    s_load_dwordx4 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, {{0x9|0x24}}
; GCN-NEXT:    s_load_dwordx2 s{{\[}}[[SRC2_LO:[0-9]+]]:[[SRC2_HI:[0-9]+]]{{\]}}
; GCN:         s_waitcnt lgkmcnt(0)
; GCN:         {{buffer|flat|global}}_load_ubyte [[SRC2:v[0-9]+]]
; GCN:         s_load_dword [[V2:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x0
; GCN:         s_waitcnt lgkmcnt(0)
; GCN:         s_bfe_u32 {{s[0-9]+}}, {{s[0-9]+}}, 0x80008
; GCN:         s_bfe_u32 {{s[0-9]+}}, {{s[0-9]+}}, 0x80008
; GCN:         v_mov_b32_e32 [[V2E1:v[0-9]+]]
; GCN:         s_bfe_u32 [[V1E3:s[0-9]+]], {{s[0-9]+}}, 0x80010
; GCN:         s_lshr_b32 [[VE4:s[0-9]+]], {{s[0-9]+}}, 24
; GCN-NEXT:    v_mov_b32_e32 [[V2E3:v[0-9]+]], {{s[0-9]+}}
; GCN-NEXT:    s_lshr_b32 [[V1E4:s[0-9]+]], {{s[0-9]+}}, 24
; GCN-NEXT:    s_waitcnt vmcnt(0)
; GCN-NEXT:    v_mad_u32_u24 [[MAD1:v[0-9]+]], {{s[0-9]+}}, [[V2E1]], [[SRC2]]
; GCN-NEXT:    v_mad_u32_u24 [[MAD2:v[0-9]+]], {{s[0-9]+}}, {{v[0-9]+}}, [[MAD1]]
; GCN-NEXT:    v_mad_u32_u24 [[MAD3:v[0-9]+]], {{s[0-9]+}}, {{v[0-9]+}}, [[MAD2]]
; GCN-NEXT:    v_mov_b32_e32 [[V2E4:v[0-9]+]], [[VE4]]
; GCN-NEXT:    v_mad_u32_u24 [[MAD4:v[0-9]+]], [[V1E4]], [[V2E4]], [[MAD3]]
; GCN-NEXT:    {{buffer|flat|global}}_store_byte
; GCN-NEXT:    s_endpgm
                                                        <4 x i8> addrspace(1)* %src2,
                                                        i8 addrspace(1)* nocapture %dst) {
entry:
  %vec1 = load <4 x i8>, <4 x i8> addrspace(1)* %src1
  %vec2 = load <4 x i8>, <4 x i8> addrspace(1)* %src2

  %v1e0 = extractelement <4 x i8> %vec1, i64 0
  %v2e0 = extractelement <4 x i8> %vec2, i64 0
  %mul1 = mul nuw nsw i8 %v2e0, %v1e0

  %v1e1 = extractelement <4 x i8> %vec1, i64 1
  %v2e1 = extractelement <4 x i8> %vec2, i64 1
  %mul2 = mul nuw nsw i8 %v2e1, %v1e1

  %v1e2 = extractelement <4 x i8> %vec1, i64 2
  %v2e2 = extractelement <4 x i8> %vec2, i64 2
  %mul3 = mul nuw nsw i8 %v2e2, %v1e2

  %v1e3 = extractelement <4 x i8> %vec1, i64 3
  %v2e3 = extractelement <4 x i8> %vec2, i64 3
  %mul4 = mul nuw nsw i8 %v2e3, %v1e3

  %acc = load i8, i8 addrspace(1)* %dst, align 2
  %mad1 = add i8 %acc, %mul2
  %mad2 = add i8 %mad1, %mul1
  %mad3 = add i8 %mad2, %mul3
  %mad4 = add i8 %mad3, %mul4

  store i8 %mad4, i8 addrspace(1)* %dst, align 2
  ret void
}

define amdgpu_kernel void @udot4_multiuse_mul1(<4 x i8> addrspace(1)* %src1,
; GCN-LABEL: udot4_multiuse_mul1:
; GCN:        ; %bb.0: ; %entry
; GCN-NEXT:    s_load_dwordx4 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, {{0x9|0x24}}
; GCN-NEXT:    s_load_dwordx2 s{{\[}}[[SRC2_LO:[0-9]+]]:[[SRC2_HI:[0-9]+]]{{\]}}, s{{\[[0-9]+:[0-9]+\]}}, {{0xd|0x34}}
; GCN-NEXT:    s_movk_i32 [[FF:s[0-9]+]], 0xff
; GCN:         s_load_dword [[S0:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x0
; GCN-NEXT:    s_load_dword [[S1:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x0
; GCN-NEXT:    s_load_dword [[S2:s[0-9]+]], s{{\[}}[[SRC2_LO]]:[[SRC2_HI]]{{\]}}, 0x0
; GCN-NEXT:    s_waitcnt lgkmcnt(0)
; GCN-NEXT:    s_and_b32 [[V1E1:s[0-9]+]], [[S0]], [[FF]]
; GCN-NEXT:    s_and_b32 [[SV2E1:s[0-9]+]], [[S1]], [[FF]]
; GCN-NEXT:    s_bfe_u32 [[SV2E2:s[0-9]+]], [[S1]], 0x80008
; GCN-NEXT:    v_mov_b32_e32 [[V2E1:v[0-9]+]], [[SV2E1]]
; GCN-NEXT:    v_mov_b32_e32 [[SRC2:v[0-9]+]], [[S2]]
; GCN-NEXT:    s_bfe_u32 [[V1E2:s[0-9]+]], [[S0]], 0x80008
; GCN-NEXT:    v_mad_u32_u24 [[MAD1:v[0-9]+]], [[V1E1]], [[V2E1]], [[SRC2]]
; GCN-NEXT:    v_mov_b32_e32 [[V2E2:v[0-9]+]], [[SV2E2]]
; GCN-NEXT:    s_bfe_u32 [[VE4:s[0-9]+]], [[S1]], 0x80010
; GCN-NEXT:    v_mad_u32_u24 [[MAD2:v[0-9]+]], [[V1E2]], [[V2E2]], [[MAD1]]
; GCN-NEXT:    s_bfe_u32 [[V1E3:s[0-9]+]], [[S0]], 0x80010
; GCN-NEXT:    v_mad_u32_u24 [[MAD3:v[0-9]+]], [[V1E1]], [[V2E1]], [[MAD2]]
; GCN-NEXT:    v_mov_b32_e32 [[V2E3:v[0-9]+]], [[VE4]]
; GCN-NEXT:    s_lshr_b32 [[VE4:s[0-9]+]], [[S1]], 24
; GCN-NEXT:    v_mad_u32_u24 [[MAD4:v[0-9]+]], [[V1E3]], [[V2E3]], [[MAD3]]
; GCN-NEXT:    s_lshr_b32 [[V1E4:s[0-9]+]], [[S0]], 24
; GCN-NEXT:    v_mov_b32_e32 [[V2E4:v[0-9]+]]
; GCN-NEXT:    v_mad_u32_u24 [[MAD5:v[0-9]+]], [[V1E4]], [[V2E4]], [[MAD4]]
; GCN:         {{buffer|flat|global}}_store_dword
; GCN-NEXT:    s_endpgm


                                               <4 x i8> addrspace(1)* %src2,
                                               i32 addrspace(1)* nocapture %dst) {
entry:
  %vec1 = load <4 x i8>, <4 x i8> addrspace(1)* %src1
  %vec2 = load <4 x i8>, <4 x i8> addrspace(1)* %src2

  %v1e0 = extractelement <4 x i8> %vec1, i64 0
  %cv1e0 = zext i8 %v1e0 to i32
  %v2e0 = extractelement <4 x i8> %vec2, i64 0
  %cv2e0 = zext i8 %v2e0 to i32
  %mul1 = mul nuw nsw i32 %cv1e0, %cv2e0

  %v1e1 = extractelement <4 x i8> %vec1, i64 1
  %cv1e1 = zext i8 %v1e1 to i32
  %v2e1 = extractelement <4 x i8> %vec2, i64 1
  %cv2e1 = zext i8 %v2e1 to i32
  %mul2 = mul nuw nsw i32 %cv1e1, %cv2e1

  %v1e2 = extractelement <4 x i8> %vec1, i64 2
  %cv1e2 = zext i8 %v1e2 to i32
  %v2e2 = extractelement <4 x i8> %vec2, i64 2
  %cv2e2 = zext i8 %v2e2 to i32
  %mul3 = mul nuw nsw i32 %cv1e2, %cv2e2

  %v1e3 = extractelement <4 x i8> %vec1, i64 3
  %cv1e3 = zext i8 %v1e3 to i32
  %v2e3 = extractelement <4 x i8> %vec2, i64 3
  %cv2e3 = zext i8 %v2e3 to i32
  %mul4 = mul nuw nsw i32 %cv1e3, %cv2e3

  %acc = load i32, i32 addrspace(1)* %dst, align 4
  %add = add i32 %mul1, %acc
  %add1 = add i32 %mul2, %add
  %add2 = add i32 %add1, %mul1
  %add3 = add i32 %add2, %mul3
  %add4 = add i32 %add3, %mul4

  store i32 %add4, i32 addrspace(1)* %dst, align 4
  ret void
}

define amdgpu_kernel void @udot4_multiuse_add1(<4 x i8> addrspace(1)* %src1,
; GCN-LABEL: udot4_multiuse_add1:
; GCN:       ; %bb.0: ; %entry
; GCN-NEXT:    s_load_dwordx4 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, {{0x9|0x24}}
; GCN-NEXT:    s_load_dwordx2 s{{\[}}[[SRC2_LO:[0-9]+]]:[[SRC2_HI:[0-9]+]]{{\]}}, s{{\[[0-9]+:[0-9]+\]}}, {{0xd|0x34}}
; GCN-NEXT:    s_movk_i32 [[FF:s[0-9]+]], 0xff
; GCN:         s_load_dword [[S0:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x0
; GCN-NEXT:    s_load_dword [[S1:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x0
; GCN-NEXT:    s_load_dword [[S2:s[0-9]+]], s{{\[}}[[SRC2_LO]]:[[SRC2_HI]]{{\]}}, 0x0
; GCN-NEXT:    s_waitcnt lgkmcnt(0)
; GCN-NEXT:    s_and_b32 [[V1E2:s[0-9]+]], [[S0]], [[FF]]
; GCN-NEXT:    s_bfe_u32 [[SV2E1:s[0-9]+]], [[S1]], 0x80008
; GCN-NEXT:    s_and_b32 [[SV2E2:s[0-9]+]], [[S1]], [[FF]]
; GCN-NEXT:    s_bfe_u32 [[V1E1:s[0-9]+]], [[S0]], 0x80008
; GCN-NEXT:    v_mov_b32_e32 [[V2E1:v[0-9]+]], [[SV2E1]]
; GCN-NEXT:    v_mov_b32_e32 [[SRC2:v[0-9]+]], [[S2]]
; GCN-NEXT:    v_mad_u32_u24 [[MAD1:v[0-9]+]], [[V1E1]], [[V2E1]], [[SRC2]]
; GCN-NEXT:    s_bfe_u32 [[SV2E3:s[0-9]+]], [[S1]], 0x80010
; GCN-NEXT:    v_mov_b32_e32 [[V2E2:v[0-9]+]], [[SV2E2]]
; GCN-NEXT:    s_bfe_u32 [[V1E3:s[0-9]+]], [[S0]], 0x80010
; GCN-NEXT:    v_add_{{i|u}}32_e32 [[ADD1:v[0-9]+]]
; GCN-NEXT:    v_mad_u32_u24 [[MAD2:v[0-9]+]], [[V1E2]], [[V2E2]], [[MAD1]]
; GCN-NEXT:    v_mov_b32_e32 [[V2E3:v[0-9]+]], [[SV2E3]]
; GCN-NEXT:    s_lshr_b32 [[SV2E4:s[0-9]+]], [[S1]], 24
; GCN-NEXT:    v_mad_u32_u24 [[MAD3:v[0-9]+]], [[V1E3]], [[V2E3]], [[MAD2]]
; GCN-NEXT:    s_lshr_b32 [[V1E4:s[0-9]+]], [[S0]], 24
; GCN-NEXT:    v_mov_b32_e32 [[V2E4:v[0-9]+]], [[SV2E4]]
; GCN-NEXT:    v_mad_u32_u24 [[MAD4:v[0-9]+]], [[V1E4]], [[V2E4]], [[MAD3]]
; GCN-NEXT:    v_add_{{i|u}}32_e32 [[RES:v[0-9]+]]
; GCN:         {{buffer|flat|global}}_store_dword
; GCN-NEXT:    s_endpgm
                                              <4 x i8> addrspace(1)* %src2,
                                              i32 addrspace(1)* nocapture %dst) {
entry:
  %vec1 = load <4 x i8>, <4 x i8> addrspace(1)* %src1
  %vec2 = load <4 x i8>, <4 x i8> addrspace(1)* %src2

  %v1e0 = extractelement <4 x i8> %vec1, i64 0
  %cv1e0 = zext i8 %v1e0 to i32
  %v2e0 = extractelement <4 x i8> %vec2, i64 0
  %cv2e0 = zext i8 %v2e0 to i32
  %mul1 = mul nuw nsw i32 %cv1e0, %cv2e0

  %v1e1 = extractelement <4 x i8> %vec1, i64 1
  %cv1e1 = zext i8 %v1e1 to i32
  %v2e1 = extractelement <4 x i8> %vec2, i64 1
  %cv2e1 = zext i8 %v2e1 to i32
  %mul2 = mul nuw nsw i32 %cv1e1, %cv2e1

  %v1e2 = extractelement <4 x i8> %vec1, i64 2
  %cv1e2 = zext i8 %v1e2 to i32
  %v2e2 = extractelement <4 x i8> %vec2, i64 2
  %cv2e2 = zext i8 %v2e2 to i32
  %mul3 = mul nuw nsw i32 %cv1e2, %cv2e2

  %v1e3 = extractelement <4 x i8> %vec1, i64 3
  %cv1e3 = zext i8 %v1e3 to i32
  %v2e3 = extractelement <4 x i8> %vec2, i64 3
  %cv2e3 = zext i8 %v2e3 to i32
  %mul4 = mul nuw nsw i32 %cv1e3, %cv2e3

  %acc = load i32, i32 addrspace(1)* %dst, align 4
  %add1 = add i32 %mul2, %acc
  %add = add i32 %add1, %acc
  %add2 = add i32 %add1, %mul1
  %add3 = add i32 %add2, %mul3
  %add4 = add i32 %add3, %mul4
  %res = add i32 %add4, %add
  store i32 %res, i32 addrspace(1)* %dst, align 4
  ret void
}

define amdgpu_kernel void @notdot4_mixedtypes(<4 x i8> addrspace(1)* %src1,
; GCN-LABEL: notdot4_mixedtypes:
; GCN:       ; %bb.0: ; %entry
; GCN-NEXT:    s_load_dwordx4 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, {{0x9|0x24}}
; GCN-NEXT:    s_load_dwordx2 s{{\[}}[[SRC2_LO:[0-9]+]]:[[SRC2_HI:[0-9]+]]{{\]}}, s{{\[[0-9]+:[0-9]+\]}}, {{0xd|0x34}}
; GCN:         s_mov_b32 [[FFFF:s[0-9]+]], 0xffff
; GCN-NEXT:    s_waitcnt lgkmcnt(0)
; GCN:         {{buffer|flat|global}}_load_ushort [[SRC2:v[0-9]+]]
; GCN:         s_load_dword [[S1:s[0-9]+]], s[6:7], 0x0
; GCN:         s_waitcnt lgkmcnt(0)
; GCN:         s_bfe_u32 [[SV2E1:s[0-9]+]], [[S1]], 0x80008
; GCN:         v_mov_b32_e32 [[V2E1:v[0-9]+]], [[SV2E1]]
; GCN:         s_bfe_u32 [[SV2E3:s[0-9]+]], [[S1]], 0x80010
; GCN:         v_mov_b32_e32 [[V2E2:v[0-9]+]]
; GCN:         s_bfe_u32 [[V1E3:s[0-9]+]], {{s[0-9]+}}, 0x80010
; GCN-NEXT:    s_lshr_b32 [[SV2E4:s[0-9]+]], [[S1]], 24
; GCN-NEXT:    v_mov_b32_e32 [[V2E3:v[0-9]+]], [[SV2E3]]
; GCN-NEXT:    s_lshr_b32 [[V1E4:s[0-9]+]], {{s[0-9]+}}, 24
; GCN-NEXT:    s_waitcnt vmcnt(0)
; GCN-NEXT:    v_mad_u32_u24 [[MAD1:v[0-9]+]], {{s[0-9]+}}, [[V2E1]], [[SRC2]]
; GCN-NEXT:    v_mad_u32_u24 [[MAD2:v[0-9]+]], {{s[0-9]+}}, [[V2E2]], [[MAD1]]
; GCN-NEXT:    v_mad_u32_u24 [[MAD3:v[0-9]+]], [[V1E3]], [[V2E3]], [[MAD2]]
; GCN-NEXT:    v_mov_b32_e32 [[V2E4:v[0-9]+]], [[SV2E4]]
; GCN-NEXT:    v_mad_u32_u24 [[MAD4:v[0-9]+]], [[V1E4]], [[V2E4]], [[MAD3]]
; GCN-NEXT:    {{buffer|flat|global}}_store_short
; GCN-NEXT:    s_endpgm
                                              <4 x i8> addrspace(1)* %src2,
                                              i16 addrspace(1)* nocapture %dst) {
entry:
  %vec1 = load <4 x i8>, <4 x i8> addrspace(1)* %src1
  %vec2 = load <4 x i8>, <4 x i8> addrspace(1)* %src2

  %v1e0 = extractelement <4 x i8> %vec1, i64 0
  %cv1e0 = sext i8 %v1e0 to i16
  %v2e0 = extractelement <4 x i8> %vec2, i64 0
  %cv2e0 = sext i8 %v2e0 to i16
  %mul1 = mul nuw nsw i16 %cv1e0, %cv2e0

  %v1e1 = extractelement <4 x i8> %vec1, i64 1
  %cv1e1 = zext i8 %v1e1 to i16
  %v2e1 = extractelement <4 x i8> %vec2, i64 1
  %cv2e1 = zext i8 %v2e1 to i16
  %mul2 = mul nuw nsw i16 %cv1e1, %cv2e1

  %v1e2 = extractelement <4 x i8> %vec1, i64 2
  %cv1e2 = zext i8 %v1e2 to i16
  %v2e2 = extractelement <4 x i8> %vec2, i64 2
  %cv2e2 = zext i8 %v2e2 to i16
  %mul3 = mul nuw nsw i16 %cv1e2, %cv2e2

  %v1e3 = extractelement <4 x i8> %vec1, i64 3
  %cv1e3 = zext i8 %v1e3 to i16
  %v2e3 = extractelement <4 x i8> %vec2, i64 3
  %cv2e3 = zext i8 %v2e3 to i16
  %mul4 = mul nuw nsw i16 %cv1e3, %cv2e3

  %acc = load i16, i16 addrspace(1)* %dst, align 2
  %add1 = add i16 %mul2, %acc
  %add2 = add i16 %add1, %mul1
  %add3 = add i16 %add2, %mul3
  %add4 = add i16 %add3, %mul4

  store i16 %add4, i16 addrspace(1)* %dst, align 2
  ret void
}
