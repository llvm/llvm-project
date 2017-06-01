; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=cypress -verify-machineinstrs < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

; FIXME: i16 promotion pass ruins the scalar cases when legal.

; FUNC-LABEL: {{^}}sext_in_reg_i1_i32:
; GCN: s_load_dword [[ARG:s[0-9]+]],
; GCN: s_bfe_i32 [[SEXTRACT:s[0-9]+]], [[ARG]], 0x10000
; GCN: v_mov_b32_e32 [[EXTRACT:v[0-9]+]], [[SEXTRACT]]
; GCN: buffer_store_dword [[EXTRACT]],

; EG: MEM_{{.*}} STORE_{{.*}} [[RES:T[0-9]+\.[XYZW]]], [[ADDR:T[0-9]+.[XYZW]]]
; EG: LSHR * [[ADDR]]
; EG: BFE_INT * [[RES]], {{.*}}, 0.0, 1
define void @sext_in_reg_i1_i32(i32 addrspace(1)* %out, i32 %in) #0 {
  %shl = shl i32 %in, 31
  %sext = ashr i32 %shl, 31
  store i32 %sext, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}sext_in_reg_i8_to_i32:
; GCN: s_add_i32 [[VAL:s[0-9]+]],
; GCN: s_sext_i32_i8 [[EXTRACT:s[0-9]+]], [[VAL]]
; GCN: v_mov_b32_e32 [[VEXTRACT:v[0-9]+]], [[EXTRACT]]
; GCN: buffer_store_dword [[VEXTRACT]],

; EG: MEM_{{.*}} STORE_{{.*}} [[RES:T[0-9]+\.[XYZW]]], [[ADDR:T[0-9]+.[XYZW]]]
; EG: ADD_INT
; EG-NEXT: BFE_INT [[RES]], {{.*}}, 0.0, literal
; EG-NEXT: LSHR * [[ADDR]]
define void @sext_in_reg_i8_to_i32(i32 addrspace(1)* %out, i32 %a, i32 %b) #0 {
  %c = add i32 %a, %b ; add to prevent folding into extload
  %shl = shl i32 %c, 24
  %ashr = ashr i32 %shl, 24
  store i32 %ashr, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}sext_in_reg_i16_to_i32:
; GCN: s_add_i32 [[VAL:s[0-9]+]],
; GCN: s_sext_i32_i16 [[EXTRACT:s[0-9]+]], [[VAL]]
; GCN: v_mov_b32_e32 [[VEXTRACT:v[0-9]+]], [[EXTRACT]]
; GCN: buffer_store_dword [[VEXTRACT]],

; EG: MEM_{{.*}} STORE_{{.*}} [[RES:T[0-9]+\.[XYZW]]], [[ADDR:T[0-9]+.[XYZW]]]
; EG: ADD_INT
; EG-NEXT: BFE_INT [[RES]], {{.*}}, 0.0, literal
; EG-NEXT: LSHR * [[ADDR]]
define void @sext_in_reg_i16_to_i32(i32 addrspace(1)* %out, i32 %a, i32 %b) #0 {
  %c = add i32 %a, %b ; add to prevent folding into extload
  %shl = shl i32 %c, 16
  %ashr = ashr i32 %shl, 16
  store i32 %ashr, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}sext_in_reg_i8_to_v1i32:
; GCN: s_add_i32 [[VAL:s[0-9]+]],
; GCN: s_sext_i32_i8 [[EXTRACT:s[0-9]+]], [[VAL]]
; GCN: v_mov_b32_e32 [[VEXTRACT:v[0-9]+]], [[EXTRACT]]
; GCN: buffer_store_dword [[VEXTRACT]],

; EG: MEM_{{.*}} STORE_{{.*}} [[RES:T[0-9]+\.[XYZW]]], [[ADDR:T[0-9]+.[XYZW]]]
; EG: ADD_INT
; EG-NEXT: BFE_INT [[RES]], {{.*}}, 0.0, literal
; EG-NEXT: LSHR * [[ADDR]]
define void @sext_in_reg_i8_to_v1i32(<1 x i32> addrspace(1)* %out, <1 x i32> %a, <1 x i32> %b) #0 {
  %c = add <1 x i32> %a, %b ; add to prevent folding into extload
  %shl = shl <1 x i32> %c, <i32 24>
  %ashr = ashr <1 x i32> %shl, <i32 24>
  store <1 x i32> %ashr, <1 x i32> addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}sext_in_reg_i1_to_i64:
; GCN: s_lshl_b64 [[VAL:s\[[0-9]+:[0-9]+\]]]
; GCN-DAG: s_bfe_i64 s{{\[}}[[SLO:[0-9]+]]:[[SHI:[0-9]+]]{{\]}}, [[VAL]], 0x10000
; GCN-DAG: v_mov_b32_e32 v[[VLO:[0-9]+]], s[[SLO]]
; GCN-DAG: v_mov_b32_e32 v[[VHI:[0-9]+]], s[[SHI]]
; GCN: buffer_store_dwordx2 v{{\[}}[[VLO]]:[[VHI]]{{\]}}
define void @sext_in_reg_i1_to_i64(i64 addrspace(1)* %out, i64 %a, i64 %b) #0 {
  %c = shl i64 %a, %b
  %shl = shl i64 %c, 63
  %ashr = ashr i64 %shl, 63
  store i64 %ashr, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}sext_in_reg_i8_to_i64:
; GCN: s_lshl_b64 [[VAL:s\[[0-9]+:[0-9]+\]]]
; GCN-DAG: s_bfe_i64 s{{\[}}[[SLO:[0-9]+]]:[[SHI:[0-9]+]]{{\]}}, [[VAL]], 0x80000
; GCN-DAG: v_mov_b32_e32 v[[VLO:[0-9]+]], s[[SLO]]
; GCN-DAG: v_mov_b32_e32 v[[VHI:[0-9]+]], s[[SHI]]
; GCN: buffer_store_dwordx2 v{{\[}}[[VLO]]:[[VHI]]{{\]}}
define void @sext_in_reg_i8_to_i64(i64 addrspace(1)* %out, i64 %a, i64 %b) #0 {
  %c = shl i64 %a, %b
  %shl = shl i64 %c, 56
  %ashr = ashr i64 %shl, 56
  store i64 %ashr, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}sext_in_reg_i16_to_i64:
; GCN: s_lshl_b64 [[VAL:s\[[0-9]+:[0-9]+\]]]
; GCN-DAG: s_bfe_i64 s{{\[}}[[SLO:[0-9]+]]:[[SHI:[0-9]+]]{{\]}}, [[VAL]], 0x100000
; GCN-DAG: v_mov_b32_e32 v[[VLO:[0-9]+]], s[[SLO]]
; GCN-DAG: v_mov_b32_e32 v[[VHI:[0-9]+]], s[[SHI]]
; GCN: buffer_store_dwordx2 v{{\[}}[[VLO]]:[[VHI]]{{\]}}

define void @sext_in_reg_i16_to_i64(i64 addrspace(1)* %out, i64 %a, i64 %b) #0 {
  %c = shl i64 %a, %b
  %shl = shl i64 %c, 48
  %ashr = ashr i64 %shl, 48
  store i64 %ashr, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}sext_in_reg_i32_to_i64:
; GCN: s_lshl_b64 [[VAL:s\[[0-9]+:[0-9]+\]]]
; GCN-DAG: s_bfe_i64 s{{\[}}[[SLO:[0-9]+]]:[[SHI:[0-9]+]]{{\]}}, [[VAL]], 0x200000
; GCN-DAG: v_mov_b32_e32 v[[VLO:[0-9]+]], s[[SLO]]
; GCN-DAG: v_mov_b32_e32 v[[VHI:[0-9]+]], s[[SHI]]
; GCN: buffer_store_dwordx2 v{{\[}}[[VLO]]:[[VHI]]{{\]}}
define void @sext_in_reg_i32_to_i64(i64 addrspace(1)* %out, i64 %a, i64 %b) #0 {
  %c = shl i64 %a, %b
  %shl = shl i64 %c, 32
  %ashr = ashr i64 %shl, 32
  store i64 %ashr, i64 addrspace(1)* %out, align 8
  ret void
}

; This is broken on Evergreen for some reason related to the <1 x i64> kernel arguments.
; XFUNC-LABEL: {{^}}sext_in_reg_i8_to_v1i64:
; XGCN: s_bfe_i32 [[EXTRACT:s[0-9]+]], {{s[0-9]+}}, 524288
; XGCN: s_ashr_i32 {{v[0-9]+}}, [[EXTRACT]], 31
; XGCN: buffer_store_dword
; XEG: BFE_INT
; XEG: ASHR
; define void @sext_in_reg_i8_to_v1i64(<1 x i64> addrspace(1)* %out, <1 x i64> %a, <1 x i64> %b) #0 {
;   %c = add <1 x i64> %a, %b
;   %shl = shl <1 x i64> %c, <i64 56>
;   %ashr = ashr <1 x i64> %shl, <i64 56>
;   store <1 x i64> %ashr, <1 x i64> addrspace(1)* %out, align 8
;   ret void
; }

; FUNC-LABEL: {{^}}v_sext_in_reg_i1_to_i64:
; SI: buffer_load_dwordx2
; SI: v_lshl_b64 v{{\[}}[[VAL_LO:[0-9]+]]:[[VAL_HI:[0-9]+]]{{\]}}

; VI: flat_load_dwordx2
; VI: v_lshlrev_b64 v{{\[}}[[VAL_LO:[0-9]+]]:[[VAL_HI:[0-9]+]]{{\]}}

; GCN: v_bfe_i32 v[[LO:[0-9]+]], v[[VAL_LO]], 0, 1
; GCN: v_ashrrev_i32_e32 v[[HI:[0-9]+]], 31, v[[LO]]

; SI: buffer_store_dwordx2 v{{\[}}[[LO]]:[[HI]]{{\]}}
; VI: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, v{{\[}}[[LO]]:[[HI]]{{\]}}
define void @v_sext_in_reg_i1_to_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %aptr, i64 addrspace(1)* %bptr) #0 {
  %tid = call i32 @llvm.r600.read.tidig.x()
  %a.gep = getelementptr i64, i64 addrspace(1)* %aptr, i32 %tid
  %b.gep = getelementptr i64, i64 addrspace(1)* %aptr, i32 %tid
  %out.gep = getelementptr i64, i64 addrspace(1)* %out, i32 %tid
  %a = load i64, i64 addrspace(1)* %a.gep, align 8
  %b = load i64, i64 addrspace(1)* %b.gep, align 8

  %c = shl i64 %a, %b
  %shl = shl i64 %c, 63
  %ashr = ashr i64 %shl, 63
  store i64 %ashr, i64 addrspace(1)* %out.gep, align 8
  ret void
}

; FUNC-LABEL: {{^}}v_sext_in_reg_i8_to_i64:
; SI: buffer_load_dwordx2
; SI: v_lshl_b64 v{{\[}}[[VAL_LO:[0-9]+]]:[[VAL_HI:[0-9]+]]{{\]}}

; VI: flat_load_dwordx2
; VI: v_lshlrev_b64 v{{\[}}[[VAL_LO:[0-9]+]]:[[VAL_HI:[0-9]+]]{{\]}}

; GCN: v_bfe_i32 v[[LO:[0-9]+]], v[[VAL_LO]], 0, 8
; GCN: v_ashrrev_i32_e32 v[[HI:[0-9]+]], 31, v[[LO]]

; SI: buffer_store_dwordx2 v{{\[}}[[LO]]:[[HI]]{{\]}}
; VI: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, v{{\[}}[[LO]]:[[HI]]{{\]}}
define void @v_sext_in_reg_i8_to_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %aptr, i64 addrspace(1)* %bptr) #0 {
  %tid = call i32 @llvm.r600.read.tidig.x()
  %a.gep = getelementptr i64, i64 addrspace(1)* %aptr, i32 %tid
  %b.gep = getelementptr i64, i64 addrspace(1)* %aptr, i32 %tid
  %out.gep = getelementptr i64, i64 addrspace(1)* %out, i32 %tid
  %a = load i64, i64 addrspace(1)* %a.gep, align 8
  %b = load i64, i64 addrspace(1)* %b.gep, align 8

  %c = shl i64 %a, %b
  %shl = shl i64 %c, 56
  %ashr = ashr i64 %shl, 56
  store i64 %ashr, i64 addrspace(1)* %out.gep, align 8
  ret void
}

; FUNC-LABEL: {{^}}v_sext_in_reg_i16_to_i64:
; SI: buffer_load_dwordx2
; SI: v_lshl_b64 v{{\[}}[[VAL_LO:[0-9]+]]:[[VAL_HI:[0-9]+]]{{\]}}

; VI: flat_load_dwordx2
; VI: v_lshlrev_b64 v{{\[}}[[VAL_LO:[0-9]+]]:[[VAL_HI:[0-9]+]]{{\]}}

; GCN: v_bfe_i32 v[[LO:[0-9]+]], v[[VAL_LO]], 0, 16
; GCN: v_ashrrev_i32_e32 v[[HI:[0-9]+]], 31, v[[LO]]

; SI: buffer_store_dwordx2 v{{\[}}[[LO]]:[[HI]]{{\]}}
; VI: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, v{{\[}}[[LO]]:[[HI]]{{\]}}
define void @v_sext_in_reg_i16_to_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %aptr, i64 addrspace(1)* %bptr) #0 {
  %tid = call i32 @llvm.r600.read.tidig.x()
  %a.gep = getelementptr i64, i64 addrspace(1)* %aptr, i32 %tid
  %b.gep = getelementptr i64, i64 addrspace(1)* %aptr, i32 %tid
  %out.gep = getelementptr i64, i64 addrspace(1)* %out, i32 %tid
  %a = load i64, i64 addrspace(1)* %a.gep, align 8
  %b = load i64, i64 addrspace(1)* %b.gep, align 8

  %c = shl i64 %a, %b
  %shl = shl i64 %c, 48
  %ashr = ashr i64 %shl, 48
  store i64 %ashr, i64 addrspace(1)* %out.gep, align 8
  ret void
}

; FUNC-LABEL: {{^}}v_sext_in_reg_i32_to_i64:
; SI: buffer_load_dwordx2
; SI: v_lshl_b64 v{{\[}}[[LO:[0-9]+]]:[[HI:[0-9]+]]{{\]}},

; VI: flat_load_dwordx2
; VI: v_lshlrev_b64 v{{\[}}[[LO:[0-9]+]]:[[HI:[0-9]+]]{{\]}},

; GCN: v_ashrrev_i32_e32 v[[SHR:[0-9]+]], 31, v[[LO]]
; VI: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, v{{\[}}[[LO]]:[[SHR]]{{\]}}
define void @v_sext_in_reg_i32_to_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %aptr, i64 addrspace(1)* %bptr) #0 {
  %tid = call i32 @llvm.r600.read.tidig.x()
  %a.gep = getelementptr i64, i64 addrspace(1)* %aptr, i32 %tid
  %b.gep = getelementptr i64, i64 addrspace(1)* %aptr, i32 %tid
  %out.gep = getelementptr i64, i64 addrspace(1)* %out, i32 %tid
  %a = load i64, i64 addrspace(1)* %a.gep, align 8
  %b = load i64, i64 addrspace(1)* %b.gep, align 8

  %c = shl i64 %a, %b
  %shl = shl i64 %c, 32
  %ashr = ashr i64 %shl, 32
  store i64 %ashr, i64 addrspace(1)* %out.gep, align 8
  ret void
}

; FUNC-LABEL: {{^}}sext_in_reg_i1_in_i32_other_amount:
; GCN-NOT: s_lshl
; GCN-NOT: s_ashr
; GCN: s_bfe_i32 {{s[0-9]+}}, {{s[0-9]+}}, 0x190001

; EG: MEM_{{.*}} STORE_{{.*}} [[RES:T[0-9]+\.[XYZW]]], [[ADDR:T[0-9]+.[XYZW]]]
; EG-NOT: BFE
; EG: ADD_INT
; EG: LSHL
; EG: ASHR [[RES]]
; EG: LSHR {{\*?}} [[ADDR]]
define void @sext_in_reg_i1_in_i32_other_amount(i32 addrspace(1)* %out, i32 %a, i32 %b) #0 {
  %c = add i32 %a, %b
  %x = shl i32 %c, 6
  %y = ashr i32 %x, 7
  store i32 %y, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}sext_in_reg_v2i1_in_v2i32_other_amount:
; GCN-NOT: s_lshl
; GCN-NOT: s_ashr
; GCN-DAG: s_bfe_i32 {{s[0-9]+}}, {{s[0-9]+}}, 0x190001
; GCN-DAG: s_bfe_i32 {{s[0-9]+}}, {{s[0-9]+}}, 0x190001
; GCN: s_endpgm

; EG: MEM_{{.*}} STORE_{{.*}} [[RES:T[0-9]+]]{{\.[XYZW][XYZW]}}, [[ADDR:T[0-9]+.[XYZW]]]
; EG-NOT: BFE
; EG: ADD_INT
; EG: LSHL
; EG: ASHR [[RES]]
; EG: LSHL
; EG: ASHR [[RES]]
; EG: LSHR {{\*?}} [[ADDR]]
define void @sext_in_reg_v2i1_in_v2i32_other_amount(<2 x i32> addrspace(1)* %out, <2 x i32> %a, <2 x i32> %b) #0 {
  %c = add <2 x i32> %a, %b
  %x = shl <2 x i32> %c, <i32 6, i32 6>
  %y = ashr <2 x i32> %x, <i32 7, i32 7>
  store <2 x i32> %y, <2 x i32> addrspace(1)* %out
  ret void
}


; FUNC-LABEL: {{^}}sext_in_reg_v2i1_to_v2i32:
; GCN: s_bfe_i32 {{s[0-9]+}}, {{s[0-9]+}}, 0x10000
; GCN: s_bfe_i32 {{s[0-9]+}}, {{s[0-9]+}}, 0x10000
; GCN: buffer_store_dwordx2

; EG: MEM_{{.*}} STORE_{{.*}} [[RES:T[0-9]+]]{{\.[XYZW][XYZW]}}, [[ADDR:T[0-9]+.[XYZW]]]
; EG: BFE_INT [[RES]]
; EG: BFE_INT [[RES]]
; EG: LSHR {{\*?}} [[ADDR]]
define void @sext_in_reg_v2i1_to_v2i32(<2 x i32> addrspace(1)* %out, <2 x i32> %a, <2 x i32> %b) #0 {
  %c = add <2 x i32> %a, %b ; add to prevent folding into extload
  %shl = shl <2 x i32> %c, <i32 31, i32 31>
  %ashr = ashr <2 x i32> %shl, <i32 31, i32 31>
  store <2 x i32> %ashr, <2 x i32> addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}sext_in_reg_v4i1_to_v4i32:
; GCN: s_bfe_i32 {{s[0-9]+}}, {{s[0-9]+}}, 0x10000
; GCN: s_bfe_i32 {{s[0-9]+}}, {{s[0-9]+}}, 0x10000
; GCN: s_bfe_i32 {{s[0-9]+}}, {{s[0-9]+}}, 0x10000
; GCN: s_bfe_i32 {{s[0-9]+}}, {{s[0-9]+}}, 0x10000
; GCN: buffer_store_dwordx4

; EG: MEM_{{.*}} STORE_{{.*}} [[RES:T[0-9]+]]{{\.[XYZW][XYZW][XYZW][XYZW]}}, [[ADDR:T[0-9]+.[XYZW]]]
; EG: BFE_INT [[RES]]
; EG: BFE_INT [[RES]]
; EG: BFE_INT [[RES]]
; EG: BFE_INT [[RES]]
; EG: LSHR {{\*?}} [[ADDR]]
define void @sext_in_reg_v4i1_to_v4i32(<4 x i32> addrspace(1)* %out, <4 x i32> %a, <4 x i32> %b) #0 {
  %c = add <4 x i32> %a, %b ; add to prevent folding into extload
  %shl = shl <4 x i32> %c, <i32 31, i32 31, i32 31, i32 31>
  %ashr = ashr <4 x i32> %shl, <i32 31, i32 31, i32 31, i32 31>
  store <4 x i32> %ashr, <4 x i32> addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}sext_in_reg_v2i8_to_v2i32:
; GCN: s_sext_i32_i8 {{s[0-9]+}}, {{s[0-9]+}}
; GCN: s_sext_i32_i8 {{s[0-9]+}}, {{s[0-9]+}}
; GCN: buffer_store_dwordx2

; EG: MEM_{{.*}} STORE_{{.*}} [[RES:T[0-9]+]]{{\.[XYZW][XYZW]}}, [[ADDR:T[0-9]+.[XYZW]]]
; EG: BFE_INT [[RES]]
; EG: BFE_INT [[RES]]
; EG: LSHR {{\*?}} [[ADDR]]
define void @sext_in_reg_v2i8_to_v2i32(<2 x i32> addrspace(1)* %out, <2 x i32> %a, <2 x i32> %b) #0 {
  %c = add <2 x i32> %a, %b ; add to prevent folding into extload
  %shl = shl <2 x i32> %c, <i32 24, i32 24>
  %ashr = ashr <2 x i32> %shl, <i32 24, i32 24>
  store <2 x i32> %ashr, <2 x i32> addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}sext_in_reg_v4i8_to_v4i32:
; GCN: s_sext_i32_i8 {{s[0-9]+}}, {{s[0-9]+}}
; GCN: s_sext_i32_i8 {{s[0-9]+}}, {{s[0-9]+}}
; GCN: s_sext_i32_i8 {{s[0-9]+}}, {{s[0-9]+}}
; GCN: s_sext_i32_i8 {{s[0-9]+}}, {{s[0-9]+}}
; GCN: buffer_store_dwordx4

; EG: MEM_{{.*}} STORE_{{.*}} [[RES:T[0-9]+]]{{\.[XYZW][XYZW][XYZW][XYZW]}}, [[ADDR:T[0-9]+.[XYZW]]]
; EG: BFE_INT [[RES]]
; EG: BFE_INT [[RES]]
; EG: BFE_INT [[RES]]
; EG: BFE_INT [[RES]]
; EG: LSHR {{\*?}} [[ADDR]]
define void @sext_in_reg_v4i8_to_v4i32(<4 x i32> addrspace(1)* %out, <4 x i32> %a, <4 x i32> %b) #0 {
  %c = add <4 x i32> %a, %b ; add to prevent folding into extload
  %shl = shl <4 x i32> %c, <i32 24, i32 24, i32 24, i32 24>
  %ashr = ashr <4 x i32> %shl, <i32 24, i32 24, i32 24, i32 24>
  store <4 x i32> %ashr, <4 x i32> addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}sext_in_reg_v2i16_to_v2i32:
; GCN: s_sext_i32_i16 {{s[0-9]+}}, {{s[0-9]+}}
; GCN: s_sext_i32_i16 {{s[0-9]+}}, {{s[0-9]+}}
; GCN: buffer_store_dwordx2

; EG: MEM_{{.*}} STORE_{{.*}} [[RES:T[0-9]+]]{{\.[XYZW][XYZW]}}, [[ADDR:T[0-9]+.[XYZW]]]
; EG: BFE_INT [[RES]]
; EG: BFE_INT [[RES]]
; EG: LSHR {{\*?}} [[ADDR]]
define void @sext_in_reg_v2i16_to_v2i32(<2 x i32> addrspace(1)* %out, <2 x i32> %a, <2 x i32> %b) #0 {
  %c = add <2 x i32> %a, %b ; add to prevent folding into extload
  %shl = shl <2 x i32> %c, <i32 16, i32 16>
  %ashr = ashr <2 x i32> %shl, <i32 16, i32 16>
  store <2 x i32> %ashr, <2 x i32> addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}testcase:
define void @testcase(i8 addrspace(1)* %out, i8 %a) #0 {
  %and_a_1 = and i8 %a, 1
  %cmp_eq = icmp eq i8 %and_a_1, 0
  %cmp_slt = icmp slt i8 %a, 0
  %sel0 = select i1 %cmp_slt, i8 0, i8 %a
  %sel1 = select i1 %cmp_eq, i8 0, i8 %a
  %xor = xor i8 %sel0, %sel1
  store i8 %xor, i8 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}testcase_3:
define void @testcase_3(i8 addrspace(1)* %out, i8 %a) #0 {
  %and_a_1 = and i8 %a, 1
  %cmp_eq = icmp eq i8 %and_a_1, 0
  %cmp_slt = icmp slt i8 %a, 0
  %sel0 = select i1 %cmp_slt, i8 0, i8 %a
  %sel1 = select i1 %cmp_eq, i8 0, i8 %a
  %xor = xor i8 %sel0, %sel1
  store i8 %xor, i8 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}vgpr_sext_in_reg_v4i8_to_v4i32:
; GCN: v_bfe_i32 [[EXTRACT:v[0-9]+]], {{v[0-9]+}}, 0, 8
; GCN: v_bfe_i32 [[EXTRACT:v[0-9]+]], {{v[0-9]+}}, 0, 8
; GCN: v_bfe_i32 [[EXTRACT:v[0-9]+]], {{v[0-9]+}}, 0, 8
; GCN: v_bfe_i32 [[EXTRACT:v[0-9]+]], {{v[0-9]+}}, 0, 8
define void @vgpr_sext_in_reg_v4i8_to_v4i32(<4 x i32> addrspace(1)* %out, <4 x i32> addrspace(1)* %a, <4 x i32> addrspace(1)* %b) #0 {
  %loada = load <4 x i32>, <4 x i32> addrspace(1)* %a, align 16
  %loadb = load <4 x i32>, <4 x i32> addrspace(1)* %b, align 16
  %c = add <4 x i32> %loada, %loadb ; add to prevent folding into extload
  %shl = shl <4 x i32> %c, <i32 24, i32 24, i32 24, i32 24>
  %ashr = ashr <4 x i32> %shl, <i32 24, i32 24, i32 24, i32 24>
  store <4 x i32> %ashr, <4 x i32> addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}vgpr_sext_in_reg_v4i16_to_v4i32:
; GCN: v_bfe_i32 [[EXTRACT:v[0-9]+]], {{v[0-9]+}}, 0, 16
; GCN: v_bfe_i32 [[EXTRACT:v[0-9]+]], {{v[0-9]+}}, 0, 16
define void @vgpr_sext_in_reg_v4i16_to_v4i32(<4 x i32> addrspace(1)* %out, <4 x i32> addrspace(1)* %a, <4 x i32> addrspace(1)* %b) #0 {
  %loada = load <4 x i32>, <4 x i32> addrspace(1)* %a, align 16
  %loadb = load <4 x i32>, <4 x i32> addrspace(1)* %b, align 16
  %c = add <4 x i32> %loada, %loadb ; add to prevent folding into extload
  %shl = shl <4 x i32> %c, <i32 16, i32 16, i32 16, i32 16>
  %ashr = ashr <4 x i32> %shl, <i32 16, i32 16, i32 16, i32 16>
  store <4 x i32> %ashr, <4 x i32> addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}sext_in_reg_to_illegal_type:
; GCN: buffer_load_sbyte
; GCN: v_max_i32
; GCN-NOT: bfe
; GCN: buffer_store_short
define void @sext_in_reg_to_illegal_type(i16 addrspace(1)* nocapture %out, i8 addrspace(1)* nocapture %src) #0 {
  %tmp5 = load i8, i8 addrspace(1)* %src, align 1
  %tmp2 = sext i8 %tmp5 to i32
  %tmp2.5 = icmp sgt i32 %tmp2, 0
  %tmp3 = select i1 %tmp2.5, i32 %tmp2, i32 0
  %tmp4 = trunc i32 %tmp3 to i8
  %tmp6 = sext i8 %tmp4 to i16
  store i16 %tmp6, i16 addrspace(1)* %out, align 2
  ret void
}

declare i32 @llvm.AMDGPU.bfe.i32(i32, i32, i32) nounwind readnone

; FUNC-LABEL: {{^}}bfe_0_width:
; GCN-NOT: {{[^@]}}bfe
; GCN: s_endpgm
define void @bfe_0_width(i32 addrspace(1)* %out, i32 addrspace(1)* %ptr) #0 {
  %load = load i32, i32 addrspace(1)* %ptr, align 4
  %bfe = call i32 @llvm.AMDGPU.bfe.i32(i32 %load, i32 8, i32 0) nounwind readnone
  store i32 %bfe, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}bfe_8_bfe_8:
; GCN: v_bfe_i32
; GCN-NOT: {{[^@]}}bfe
; GCN: s_endpgm
define void @bfe_8_bfe_8(i32 addrspace(1)* %out, i32 addrspace(1)* %ptr) #0 {
  %load = load i32, i32 addrspace(1)* %ptr, align 4
  %bfe0 = call i32 @llvm.AMDGPU.bfe.i32(i32 %load, i32 0, i32 8) nounwind readnone
  %bfe1 = call i32 @llvm.AMDGPU.bfe.i32(i32 %bfe0, i32 0, i32 8) nounwind readnone
  store i32 %bfe1, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}bfe_8_bfe_16:
; GCN: v_bfe_i32 v{{[0-9]+}}, v{{[0-9]+}}, 0, 8
; GCN: s_endpgm
define void @bfe_8_bfe_16(i32 addrspace(1)* %out, i32 addrspace(1)* %ptr) #0 {
  %load = load i32, i32 addrspace(1)* %ptr, align 4
  %bfe0 = call i32 @llvm.AMDGPU.bfe.i32(i32 %load, i32 0, i32 8) nounwind readnone
  %bfe1 = call i32 @llvm.AMDGPU.bfe.i32(i32 %bfe0, i32 0, i32 16) nounwind readnone
  store i32 %bfe1, i32 addrspace(1)* %out, align 4
  ret void
}

; This really should be folded into 1
; FUNC-LABEL: {{^}}bfe_16_bfe_8:
; GCN: v_bfe_i32 v{{[0-9]+}}, v{{[0-9]+}}, 0, 8
; GCN-NOT: {{[^@]}}bfe
; GCN: s_endpgm
define void @bfe_16_bfe_8(i32 addrspace(1)* %out, i32 addrspace(1)* %ptr) #0 {
  %load = load i32, i32 addrspace(1)* %ptr, align 4
  %bfe0 = call i32 @llvm.AMDGPU.bfe.i32(i32 %load, i32 0, i32 16) nounwind readnone
  %bfe1 = call i32 @llvm.AMDGPU.bfe.i32(i32 %bfe0, i32 0, i32 8) nounwind readnone
  store i32 %bfe1, i32 addrspace(1)* %out, align 4
  ret void
}

; Make sure there isn't a redundant BFE
; FUNC-LABEL: {{^}}sext_in_reg_i8_to_i32_bfe:
; GCN: s_sext_i32_i8 s{{[0-9]+}}, s{{[0-9]+}}
; GCN-NOT: {{[^@]}}bfe
; GCN: s_endpgm
define void @sext_in_reg_i8_to_i32_bfe(i32 addrspace(1)* %out, i32 %a, i32 %b) #0 {
  %c = add i32 %a, %b ; add to prevent folding into extload
  %bfe = call i32 @llvm.AMDGPU.bfe.i32(i32 %c, i32 0, i32 8) nounwind readnone
  %shl = shl i32 %bfe, 24
  %ashr = ashr i32 %shl, 24
  store i32 %ashr, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}sext_in_reg_i8_to_i32_bfe_wrong:
define void @sext_in_reg_i8_to_i32_bfe_wrong(i32 addrspace(1)* %out, i32 %a, i32 %b) #0 {
  %c = add i32 %a, %b ; add to prevent folding into extload
  %bfe = call i32 @llvm.AMDGPU.bfe.i32(i32 %c, i32 8, i32 0) nounwind readnone
  %shl = shl i32 %bfe, 24
  %ashr = ashr i32 %shl, 24
  store i32 %ashr, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}sextload_i8_to_i32_bfe:
; GCN: buffer_load_sbyte
; GCN-NOT: {{[^@]}}bfe
; GCN: s_endpgm
define void @sextload_i8_to_i32_bfe(i32 addrspace(1)* %out, i8 addrspace(1)* %ptr) #0 {
  %load = load i8, i8 addrspace(1)* %ptr, align 1
  %sext = sext i8 %load to i32
  %bfe = call i32 @llvm.AMDGPU.bfe.i32(i32 %sext, i32 0, i32 8) nounwind readnone
  %shl = shl i32 %bfe, 24
  %ashr = ashr i32 %shl, 24
  store i32 %ashr, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN: .text
; FUNC-LABEL: {{^}}sextload_i8_to_i32_bfe_0:{{.*$}}
; GCN-NOT: {{[^@]}}bfe
; GCN: s_endpgm
define void @sextload_i8_to_i32_bfe_0(i32 addrspace(1)* %out, i8 addrspace(1)* %ptr) #0 {
  %load = load i8, i8 addrspace(1)* %ptr, align 1
  %sext = sext i8 %load to i32
  %bfe = call i32 @llvm.AMDGPU.bfe.i32(i32 %sext, i32 8, i32 0) nounwind readnone
  %shl = shl i32 %bfe, 24
  %ashr = ashr i32 %shl, 24
  store i32 %ashr, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}sext_in_reg_i1_bfe_offset_0:
; GCN-NOT: shr
; GCN-NOT: shl
; GCN: v_bfe_i32 v{{[0-9]+}}, v{{[0-9]+}}, 0, 1
; GCN: s_endpgm
define void @sext_in_reg_i1_bfe_offset_0(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
  %x = load i32, i32 addrspace(1)* %in, align 4
  %shl = shl i32 %x, 31
  %shr = ashr i32 %shl, 31
  %bfe = call i32 @llvm.AMDGPU.bfe.i32(i32 %shr, i32 0, i32 1)
  store i32 %bfe, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}sext_in_reg_i1_bfe_offset_1:
; GCN: buffer_load_dword
; GCN-NOT: shl
; GCN-NOT: shr
; GCN: v_bfe_i32 v{{[0-9]+}}, v{{[0-9]+}}, 1, 1
; GCN: s_endpgm
define void @sext_in_reg_i1_bfe_offset_1(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
  %x = load i32, i32 addrspace(1)* %in, align 4
  %shl = shl i32 %x, 30
  %shr = ashr i32 %shl, 30
  %bfe = call i32 @llvm.AMDGPU.bfe.i32(i32 %shr, i32 1, i32 1)
  store i32 %bfe, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}sext_in_reg_i2_bfe_offset_1:
; GCN: buffer_load_dword
; GCN-NOT: v_lshl
; GCN-NOT: v_ashr
; GCN: v_bfe_i32 v{{[0-9]+}}, v{{[0-9]+}}, 0, 2
; GCN: v_bfe_i32 v{{[0-9]+}}, v{{[0-9]+}}, 1, 2
; GCN: s_endpgm
define void @sext_in_reg_i2_bfe_offset_1(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
  %x = load i32, i32 addrspace(1)* %in, align 4
  %shl = shl i32 %x, 30
  %shr = ashr i32 %shl, 30
  %bfe = call i32 @llvm.AMDGPU.bfe.i32(i32 %shr, i32 1, i32 2)
  store i32 %bfe, i32 addrspace(1)* %out, align 4
  ret void
}

; Make sure we propagate the VALUness to users of a moved scalar BFE.

; FUNC-LABEL: {{^}}v_sext_in_reg_i1_to_i64_move_use:
; SI: buffer_load_dwordx2
; SI: v_lshl_b64 v{{\[}}[[VAL_LO:[0-9]+]]:[[VAL_HI:[0-9]+]]{{\]}}

; VI: flat_load_dwordx2
; VI: v_lshlrev_b64 v{{\[}}[[VAL_LO:[0-9]+]]:[[VAL_HI:[0-9]+]]{{\]}}

; GCN-DAG: v_bfe_i32 v[[LO:[0-9]+]], v[[VAL_LO]], 0, 1
; GCN-DAG: v_ashrrev_i32_e32 v[[HI:[0-9]+]], 31, v[[LO]]
; GCN-DAG: v_and_b32_e32 v[[RESULT_LO:[0-9]+]], s{{[0-9]+}}, v[[LO]]
; GCN-DAG: v_and_b32_e32 v[[RESULT_HI:[0-9]+]], s{{[0-9]+}}, v[[HI]]
; SI: buffer_store_dwordx2 v{{\[}}[[RESULT_LO]]:[[RESULT_HI]]{{\]}}
; VI: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, v{{\[}}[[RESULT_LO]]:[[RESULT_HI]]{{\]}}
define void @v_sext_in_reg_i1_to_i64_move_use(i64 addrspace(1)* %out, i64 addrspace(1)* %aptr, i64 addrspace(1)* %bptr, i64 %s.val) #0 {
  %tid = call i32 @llvm.r600.read.tidig.x()
  %a.gep = getelementptr i64, i64 addrspace(1)* %aptr, i32 %tid
  %b.gep = getelementptr i64, i64 addrspace(1)* %aptr, i32 %tid
  %out.gep = getelementptr i64, i64 addrspace(1)* %out, i32 %tid
  %a = load i64, i64 addrspace(1)* %a.gep, align 8
  %b = load i64, i64 addrspace(1)* %b.gep, align 8

  %c = shl i64 %a, %b
  %shl = shl i64 %c, 63
  %ashr = ashr i64 %shl, 63

  %and = and i64 %ashr, %s.val
  store i64 %and, i64 addrspace(1)* %out.gep, align 8
  ret void
}

; FUNC-LABEL: {{^}}v_sext_in_reg_i32_to_i64_move_use:
; SI: buffer_load_dwordx2
; SI: v_lshl_b64 v{{\[}}[[LO:[0-9]+]]:[[HI:[0-9]+]]{{\]}},

; VI: flat_load_dwordx2
; VI: v_lshlrev_b64 v{{\[}}[[LO:[0-9]+]]:[[HI:[0-9]+]]{{\]}},

; GCN-DAG: v_ashrrev_i32_e32 v[[SHR:[0-9]+]], 31, v[[LO]]
; GCN-DAG: v_and_b32_e32 v[[RESULT_LO:[0-9]+]], s{{[0-9]+}}, v[[LO]]
; GCN-DAG: v_and_b32_e32 v[[RESULT_HI:[0-9]+]], s{{[0-9]+}}, v[[SHR]]

; SI: buffer_store_dwordx2 v{{\[}}[[RESULT_LO]]:[[RESULT_HI]]{{\]}}
; VI: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, v{{\[}}[[RESULT_LO]]:[[RESULT_HI]]{{\]}}
define void @v_sext_in_reg_i32_to_i64_move_use(i64 addrspace(1)* %out, i64 addrspace(1)* %aptr, i64 addrspace(1)* %bptr, i64 %s.val) #0 {
  %tid = call i32 @llvm.r600.read.tidig.x()
  %a.gep = getelementptr i64, i64 addrspace(1)* %aptr, i32 %tid
  %b.gep = getelementptr i64, i64 addrspace(1)* %aptr, i32 %tid
  %out.gep = getelementptr i64, i64 addrspace(1)* %out, i32 %tid
  %a = load i64, i64 addrspace(1)* %a.gep, align 8
  %b = load i64, i64 addrspace(1)* %b.gep, align 8

  %c = shl i64 %a, %b
  %shl = shl i64 %c, 32
  %ashr = ashr i64 %shl, 32
  %and = and i64 %ashr, %s.val
  store i64 %and, i64 addrspace(1)* %out.gep, align 8
  ret void
}

; FUNC-LABEL: {{^}}s_sext_in_reg_i1_i16:
; GCN: s_load_dword [[VAL:s[0-9]+]]

; SI: s_bfe_i32 [[BFE:s[0-9]+]], [[VAL]], 0x10000
; SI: v_mov_b32_e32 [[VBFE:v[0-9]+]], [[BFE]]
; SI: buffer_store_short [[VBFE]]

; VI: s_lshl_b32 s{{[0-9]+}}, s{{[0-9]+}}, 15
; VI: s_sext_i32_i16 s{{[0-9]+}}, s{{[0-9]+}}
; VI: s_lshr_b32 s{{[0-9]+}}, s{{[0-9]+}}, 15
define void @s_sext_in_reg_i1_i16(i16 addrspace(1)* %out, i32 addrspace(2)* %ptr) #0 {
  %ld = load i32, i32 addrspace(2)* %ptr
  %in = trunc i32 %ld to i16
  %shl = shl i16 %in, 15
  %sext = ashr i16 %shl, 15
  store i16 %sext, i16 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}s_sext_in_reg_i2_i16:
; GCN: s_load_dword [[VAL:s[0-9]+]]

; SI: s_bfe_i32 [[BFE:s[0-9]+]], [[VAL]], 0x20000
; SI: v_mov_b32_e32 [[VBFE:v[0-9]+]], [[BFE]]
; SI: buffer_store_short [[VBFE]]

; VI: s_lshl_b32 s{{[0-9]+}}, s{{[0-9]+}}, 14
; VI: s_sext_i32_i16 s{{[0-9]+}}, s{{[0-9]+}}
; VI: s_lshr_b32 s{{[0-9]+}}, s{{[0-9]+}}, 14
define void @s_sext_in_reg_i2_i16(i16 addrspace(1)* %out, i32 addrspace(2)* %ptr) #0 {
  %ld = load i32, i32 addrspace(2)* %ptr
  %in = trunc i32 %ld to i16
  %shl = shl i16 %in, 14
  %sext = ashr i16 %shl, 14
  store i16 %sext, i16 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}v_sext_in_reg_i1_i16:
; GCN: {{buffer|flat}}_load_ushort [[VAL:v[0-9]+]]
; GCN: v_bfe_i32 [[BFE:v[0-9]+]], [[VAL]], 0, 1{{$}}

; GCN: ds_write_b16 v{{[0-9]+}}, [[BFE]]
define void @v_sext_in_reg_i1_i16(i16 addrspace(3)* %out, i16 addrspace(1)* %ptr) #0 {
  %tid = call i32 @llvm.r600.read.tidig.x()
  %gep = getelementptr i16, i16 addrspace(1)* %ptr, i32 %tid
  %out.gep = getelementptr i16, i16 addrspace(3)* %out, i32 %tid

  %in = load i16, i16 addrspace(1)* %gep
  %shl = shl i16 %in, 15
  %sext = ashr i16 %shl, 15
  store i16 %sext, i16 addrspace(3)* %out.gep
  ret void
}

; FUNC-LABEL: {{^}}v_sext_in_reg_i1_i16_nonload:
; GCN: {{buffer|flat}}_load_ushort [[VAL0:v[0-9]+]]
; GCN: {{buffer|flat}}_load_ushort [[VAL1:v[0-9]+]]

; SI: v_lshlrev_b32_e32 [[REG:v[0-9]+]], [[VAL1]], [[VAL0]]
; VI: v_lshlrev_b16_e32 [[REG:v[0-9]+]], [[VAL1]], [[VAL0]]

; GCN: v_bfe_i32 [[BFE:v[0-9]+]], [[REG]], 0, 1{{$}}
; GCN: ds_write_b16 v{{[0-9]+}}, [[BFE]]
define void @v_sext_in_reg_i1_i16_nonload(i16 addrspace(3)* %out, i16 addrspace(1)* %aptr, i16 addrspace(1)* %bptr, i16 %s.val) nounwind {
  %tid = call i32 @llvm.r600.read.tidig.x()
  %a.gep = getelementptr i16, i16 addrspace(1)* %aptr, i32 %tid
  %b.gep = getelementptr i16, i16 addrspace(1)* %bptr, i32 %tid
  %out.gep = getelementptr i16, i16 addrspace(3)* %out, i32 %tid
  %a = load volatile i16, i16 addrspace(1)* %a.gep, align 2
  %b = load volatile i16, i16 addrspace(1)* %b.gep, align 2

  %c = shl i16 %a, %b
  %shl = shl i16 %c, 15
  %ashr = ashr i16 %shl, 15

  store i16 %ashr, i16 addrspace(3)* %out.gep, align 2
  ret void
}

; FUNC-LABEL: {{^}}s_sext_in_reg_i2_i16_arg:
; GCN: s_load_dword [[VAL:s[0-9]+]]

; SI: s_bfe_i32 [[BFE:s[0-9]+]], [[VAL]], 0x20000
; SI: v_mov_b32_e32 [[VBFE:v[0-9]+]], [[BFE]]
; SI: buffer_store_short [[VBFE]]

; VI: s_lshl_b32 s{{[0-9]+}}, s{{[0-9]+}}, 14{{$}}
; VI: s_sext_i32_i16 s{{[0-9]+}}, s{{[0-9]+}}
; VI: s_lshr_b32 s{{[0-9]+}}, s{{[0-9]+}}, 14{{$}}
define void @s_sext_in_reg_i2_i16_arg(i16 addrspace(1)* %out, i16 %in) #0 {
  %shl = shl i16 %in, 14
  %sext = ashr i16 %shl, 14
  store i16 %sext, i16 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}s_sext_in_reg_i8_i16_arg:
; GCN: s_load_dword [[VAL:s[0-9]+]]

; SI: s_sext_i32_i8 [[SSEXT:s[0-9]+]], [[VAL]]
; SI: v_mov_b32_e32 [[VSEXT:v[0-9]+]], [[SSEXT]]
; SI: buffer_store_short [[VBFE]]

; VI: s_lshl_b32 s{{[0-9]+}}, s{{[0-9]+}}, 8{{$}}
; VI: s_sext_i32_i16 s{{[0-9]+}}, s{{[0-9]+}}
; VI: s_lshr_b32 s{{[0-9]+}}, s{{[0-9]+}}, 8{{$}}
define void @s_sext_in_reg_i8_i16_arg(i16 addrspace(1)* %out, i16 %in) #0 {
  %shl = shl i16 %in, 8
  %sext = ashr i16 %shl, 8
  store i16 %sext, i16 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}s_sext_in_reg_i15_i16_arg:
; GCN: s_load_dword [[VAL:s[0-9]+]]

; SI: s_bfe_i32 [[BFE:s[0-9]+]], [[VAL]], 0xf0000
; SI: v_mov_b32_e32 [[VBFE:v[0-9]+]], [[BFE]]
; SI: buffer_store_short [[VBFE]]

; VI: s_lshl_b32 s{{[0-9]+}}, s{{[0-9]+}}, 1{{$}}
; VI: s_sext_i32_i16 s{{[0-9]+}}, s{{[0-9]+}}
; VI: s_lshr_b32 s{{[0-9]+}}, s{{[0-9]+}}, 1{{$}}
define void @s_sext_in_reg_i15_i16_arg(i16 addrspace(1)* %out, i16 %in) #0 {
  %shl = shl i16 %in, 1
  %sext = ashr i16 %shl, 1
  store i16 %sext, i16 addrspace(1)* %out
  ret void
}

declare i32 @llvm.r600.read.tidig.x() #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
