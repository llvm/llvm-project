; RUN: llc -march=amdgcn -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=cypress -verify-machineinstrs < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

declare i32 @llvm.AMDGPU.imax(i32, i32) nounwind readnone
declare i32 @llvm.r600.read.tidig.x() nounwind readnone


; FUNC-LABEL: {{^}}sext_in_reg_i1_i32:
; SI: s_load_dword [[ARG:s[0-9]+]],
; SI: s_bfe_i32 [[SEXTRACT:s[0-9]+]], [[ARG]], 0x10000
; SI: v_mov_b32_e32 [[EXTRACT:v[0-9]+]], [[SEXTRACT]]
; SI: buffer_store_dword [[EXTRACT]],

; EG: MEM_{{.*}} STORE_{{.*}} [[RES:T[0-9]+\.[XYZW]]], [[ADDR:T[0-9]+.[XYZW]]]
; EG: LSHR * [[ADDR]]
; EG: BFE_INT * [[RES]], {{.*}}, 0.0, 1
define void @sext_in_reg_i1_i32(i32 addrspace(1)* %out, i32 %in) {
  %shl = shl i32 %in, 31
  %sext = ashr i32 %shl, 31
  store i32 %sext, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}sext_in_reg_i8_to_i32:
; SI: s_add_i32 [[VAL:s[0-9]+]],
; SI: s_sext_i32_i8 [[EXTRACT:s[0-9]+]], [[VAL]]
; SI: v_mov_b32_e32 [[VEXTRACT:v[0-9]+]], [[EXTRACT]]
; SI: buffer_store_dword [[VEXTRACT]],

; EG: MEM_{{.*}} STORE_{{.*}} [[RES:T[0-9]+\.[XYZW]]], [[ADDR:T[0-9]+.[XYZW]]]
; EG: ADD_INT
; EG-NEXT: BFE_INT [[RES]], {{.*}}, 0.0, literal
; EG-NEXT: LSHR * [[ADDR]]
define void @sext_in_reg_i8_to_i32(i32 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %c = add i32 %a, %b ; add to prevent folding into extload
  %shl = shl i32 %c, 24
  %ashr = ashr i32 %shl, 24
  store i32 %ashr, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}sext_in_reg_i16_to_i32:
; SI: s_add_i32 [[VAL:s[0-9]+]],
; SI: s_sext_i32_i16 [[EXTRACT:s[0-9]+]], [[VAL]]
; SI: v_mov_b32_e32 [[VEXTRACT:v[0-9]+]], [[EXTRACT]]
; SI: buffer_store_dword [[VEXTRACT]],

; EG: MEM_{{.*}} STORE_{{.*}} [[RES:T[0-9]+\.[XYZW]]], [[ADDR:T[0-9]+.[XYZW]]]
; EG: ADD_INT
; EG-NEXT: BFE_INT [[RES]], {{.*}}, 0.0, literal
; EG-NEXT: LSHR * [[ADDR]]
define void @sext_in_reg_i16_to_i32(i32 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %c = add i32 %a, %b ; add to prevent folding into extload
  %shl = shl i32 %c, 16
  %ashr = ashr i32 %shl, 16
  store i32 %ashr, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}sext_in_reg_i8_to_v1i32:
; SI: s_add_i32 [[VAL:s[0-9]+]],
; SI: s_sext_i32_i8 [[EXTRACT:s[0-9]+]], [[VAL]]
; SI: v_mov_b32_e32 [[VEXTRACT:v[0-9]+]], [[EXTRACT]]
; SI: buffer_store_dword [[VEXTRACT]],

; EG: MEM_{{.*}} STORE_{{.*}} [[RES:T[0-9]+\.[XYZW]]], [[ADDR:T[0-9]+.[XYZW]]]
; EG: ADD_INT
; EG-NEXT: BFE_INT [[RES]], {{.*}}, 0.0, literal
; EG-NEXT: LSHR * [[ADDR]]
define void @sext_in_reg_i8_to_v1i32(<1 x i32> addrspace(1)* %out, <1 x i32> %a, <1 x i32> %b) nounwind {
  %c = add <1 x i32> %a, %b ; add to prevent folding into extload
  %shl = shl <1 x i32> %c, <i32 24>
  %ashr = ashr <1 x i32> %shl, <i32 24>
  store <1 x i32> %ashr, <1 x i32> addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}sext_in_reg_i1_to_i64:
; SI: s_lshl_b64 [[VAL:s\[[0-9]+:[0-9]+\]]]
; SI-DAG: s_bfe_i64 s{{\[}}[[SLO:[0-9]+]]:[[SHI:[0-9]+]]{{\]}}, [[VAL]], 0x10000
; SI-DAG: v_mov_b32_e32 v[[VLO:[0-9]+]], s[[SLO]]
; SI-DAG: v_mov_b32_e32 v[[VHI:[0-9]+]], s[[SHI]]
; SI: buffer_store_dwordx2 v{{\[}}[[VLO]]:[[VHI]]{{\]}}
define void @sext_in_reg_i1_to_i64(i64 addrspace(1)* %out, i64 %a, i64 %b) nounwind {
  %c = shl i64 %a, %b
  %shl = shl i64 %c, 63
  %ashr = ashr i64 %shl, 63
  store i64 %ashr, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}sext_in_reg_i8_to_i64:
; SI: s_lshl_b64 [[VAL:s\[[0-9]+:[0-9]+\]]]
; SI-DAG: s_bfe_i64 s{{\[}}[[SLO:[0-9]+]]:[[SHI:[0-9]+]]{{\]}}, [[VAL]], 0x80000
; SI-DAG: v_mov_b32_e32 v[[VLO:[0-9]+]], s[[SLO]]
; SI-DAG: v_mov_b32_e32 v[[VHI:[0-9]+]], s[[SHI]]
; SI: buffer_store_dwordx2 v{{\[}}[[VLO]]:[[VHI]]{{\]}}

; EG: MEM_{{.*}} STORE_{{.*}} [[RES_LO:T[0-9]+\.[XYZW]]], [[ADDR_LO:T[0-9]+.[XYZW]]]
; EG: MEM_{{.*}} STORE_{{.*}} [[RES_HI:T[0-9]+\.[XYZW]]], [[ADDR_HI:T[0-9]+.[XYZW]]]
; EG: LSHL
; EG: BFE_INT {{\*?}} [[RES_LO]], {{.*}}, 0.0, literal
; EG: ASHR [[RES_HI]]
; EG-NOT: BFE_INT
; EG: LSHR
; EG: LSHR
;; TODO Check address computation, using | with variables in {{}} does not work,
;; also the _LO/_HI order might be different
define void @sext_in_reg_i8_to_i64(i64 addrspace(1)* %out, i64 %a, i64 %b) nounwind {
  %c = shl i64 %a, %b
  %shl = shl i64 %c, 56
  %ashr = ashr i64 %shl, 56
  store i64 %ashr, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}sext_in_reg_i16_to_i64:
; SI: s_lshl_b64 [[VAL:s\[[0-9]+:[0-9]+\]]]
; SI-DAG: s_bfe_i64 s{{\[}}[[SLO:[0-9]+]]:[[SHI:[0-9]+]]{{\]}}, [[VAL]], 0x100000
; SI-DAG: v_mov_b32_e32 v[[VLO:[0-9]+]], s[[SLO]]
; SI-DAG: v_mov_b32_e32 v[[VHI:[0-9]+]], s[[SHI]]
; SI: buffer_store_dwordx2 v{{\[}}[[VLO]]:[[VHI]]{{\]}}

; EG: MEM_{{.*}} STORE_{{.*}} [[RES_LO:T[0-9]+\.[XYZW]]], [[ADDR_LO:T[0-9]+.[XYZW]]]
; EG: MEM_{{.*}} STORE_{{.*}} [[RES_HI:T[0-9]+\.[XYZW]]], [[ADDR_HI:T[0-9]+.[XYZW]]]
; EG: LSHL
; EG: BFE_INT {{\*?}} [[RES_LO]], {{.*}}, 0.0, literal
; EG: ASHR [[RES_HI]]
; EG-NOT: BFE_INT
; EG: LSHR
; EG: LSHR
;; TODO Check address computation, using | with variables in {{}} does not work,
;; also the _LO/_HI order might be different
define void @sext_in_reg_i16_to_i64(i64 addrspace(1)* %out, i64 %a, i64 %b) nounwind {
  %c = shl i64 %a, %b
  %shl = shl i64 %c, 48
  %ashr = ashr i64 %shl, 48
  store i64 %ashr, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}sext_in_reg_i32_to_i64:
; SI: s_lshl_b64 [[VAL:s\[[0-9]+:[0-9]+\]]]
; SI-DAG: s_bfe_i64 s{{\[}}[[SLO:[0-9]+]]:[[SHI:[0-9]+]]{{\]}}, [[VAL]], 0x200000
; SI-DAG: v_mov_b32_e32 v[[VLO:[0-9]+]], s[[SLO]]
; SI-DAG: v_mov_b32_e32 v[[VHI:[0-9]+]], s[[SHI]]
; SI: buffer_store_dwordx2 v{{\[}}[[VLO]]:[[VHI]]{{\]}}

; EG: MEM_{{.*}} STORE_{{.*}} [[RES_LO:T[0-9]+\.[XYZW]]], [[ADDR_LO:T[0-9]+.[XYZW]]]
; EG: MEM_{{.*}} STORE_{{.*}} [[RES_HI:T[0-9]+\.[XYZW]]], [[ADDR_HI:T[0-9]+.[XYZW]]]
; EG-NOT: BFE_INT

; EG: ASHR [[RES_HI]]

; EG: LSHR
; EG: LSHR
;; TODO Check address computation, using | with variables in {{}} does not work,
;; also the _LO/_HI order might be different
define void @sext_in_reg_i32_to_i64(i64 addrspace(1)* %out, i64 %a, i64 %b) nounwind {
  %c = shl i64 %a, %b
  %shl = shl i64 %c, 32
  %ashr = ashr i64 %shl, 32
  store i64 %ashr, i64 addrspace(1)* %out, align 8
  ret void
}

; This is broken on Evergreen for some reason related to the <1 x i64> kernel arguments.
; XFUNC-LABEL: {{^}}sext_in_reg_i8_to_v1i64:
; XSI: s_bfe_i32 [[EXTRACT:s[0-9]+]], {{s[0-9]+}}, 524288
; XSI: s_ashr_i32 {{v[0-9]+}}, [[EXTRACT]], 31
; XSI: buffer_store_dword
; XEG: BFE_INT
; XEG: ASHR
; define void @sext_in_reg_i8_to_v1i64(<1 x i64> addrspace(1)* %out, <1 x i64> %a, <1 x i64> %b) nounwind {
;   %c = add <1 x i64> %a, %b
;   %shl = shl <1 x i64> %c, <i64 56>
;   %ashr = ashr <1 x i64> %shl, <i64 56>
;   store <1 x i64> %ashr, <1 x i64> addrspace(1)* %out, align 8
;   ret void
; }

; FUNC-LABEL: {{^}}v_sext_in_reg_i1_to_i64:
; SI: buffer_load_dwordx2
; SI: v_lshl_b64 v{{\[}}[[VAL_LO:[0-9]+]]:[[VAL_HI:[0-9]+]]{{\]}}
; SI: v_bfe_i32 v[[LO:[0-9]+]], v[[VAL_LO]], 0, 1
; SI: v_ashrrev_i32_e32 v[[HI:[0-9]+]], 31, v[[LO]]
; SI: buffer_store_dwordx2 v{{\[}}[[LO]]:[[HI]]{{\]}}
define void @v_sext_in_reg_i1_to_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %aptr, i64 addrspace(1)* %bptr) nounwind {
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
; SI: v_bfe_i32 v[[LO:[0-9]+]], v[[VAL_LO]], 0, 8
; SI: v_ashrrev_i32_e32 v[[HI:[0-9]+]], 31, v[[LO]]
; SI: buffer_store_dwordx2 v{{\[}}[[LO]]:[[HI]]{{\]}}
define void @v_sext_in_reg_i8_to_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %aptr, i64 addrspace(1)* %bptr) nounwind {
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
; SI: v_bfe_i32 v[[LO:[0-9]+]], v[[VAL_LO]], 0, 16
; SI: v_ashrrev_i32_e32 v[[HI:[0-9]+]], 31, v[[LO]]
; SI: buffer_store_dwordx2 v{{\[}}[[LO]]:[[HI]]{{\]}}
define void @v_sext_in_reg_i16_to_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %aptr, i64 addrspace(1)* %bptr) nounwind {
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
; SI: v_ashrrev_i32_e32 v[[SHR:[0-9]+]], 31, v[[LO]]
; SI: buffer_store_dwordx2 v{{\[}}[[LO]]:[[SHR]]{{\]}}
define void @v_sext_in_reg_i32_to_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %aptr, i64 addrspace(1)* %bptr) nounwind {
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
; SI-NOT: s_lshl
; SI-NOT: s_ashr
; SI: s_bfe_i32 {{s[0-9]+}}, {{s[0-9]+}}, 0x190001

; EG: MEM_{{.*}} STORE_{{.*}} [[RES:T[0-9]+\.[XYZW]]], [[ADDR:T[0-9]+.[XYZW]]]
; EG-NOT: BFE
; EG: ADD_INT
; EG: LSHL
; EG: ASHR [[RES]]
; EG: LSHR {{\*?}} [[ADDR]]
define void @sext_in_reg_i1_in_i32_other_amount(i32 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %c = add i32 %a, %b
  %x = shl i32 %c, 6
  %y = ashr i32 %x, 7
  store i32 %y, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}sext_in_reg_v2i1_in_v2i32_other_amount:
; SI-NOT: s_lshl
; SI-NOT: s_ashr
; SI-DAG: s_bfe_i32 {{s[0-9]+}}, {{s[0-9]+}}, 0x190001
; SI-DAG: s_bfe_i32 {{s[0-9]+}}, {{s[0-9]+}}, 0x190001
; SI: s_endpgm

; EG: MEM_{{.*}} STORE_{{.*}} [[RES:T[0-9]+]]{{\.[XYZW][XYZW]}}, [[ADDR:T[0-9]+.[XYZW]]]
; EG-NOT: BFE
; EG: ADD_INT
; EG: LSHL
; EG: ASHR [[RES]]
; EG: LSHL
; EG: ASHR [[RES]]
; EG: LSHR {{\*?}} [[ADDR]]
define void @sext_in_reg_v2i1_in_v2i32_other_amount(<2 x i32> addrspace(1)* %out, <2 x i32> %a, <2 x i32> %b) nounwind {
  %c = add <2 x i32> %a, %b
  %x = shl <2 x i32> %c, <i32 6, i32 6>
  %y = ashr <2 x i32> %x, <i32 7, i32 7>
  store <2 x i32> %y, <2 x i32> addrspace(1)* %out, align 2
  ret void
}


; FUNC-LABEL: {{^}}sext_in_reg_v2i1_to_v2i32:
; SI: s_bfe_i32 {{s[0-9]+}}, {{s[0-9]+}}, 0x10000
; SI: s_bfe_i32 {{s[0-9]+}}, {{s[0-9]+}}, 0x10000
; SI: buffer_store_dwordx2

; EG: MEM_{{.*}} STORE_{{.*}} [[RES:T[0-9]+]]{{\.[XYZW][XYZW]}}, [[ADDR:T[0-9]+.[XYZW]]]
; EG: BFE_INT [[RES]]
; EG: BFE_INT [[RES]]
; EG: LSHR {{\*?}} [[ADDR]]
define void @sext_in_reg_v2i1_to_v2i32(<2 x i32> addrspace(1)* %out, <2 x i32> %a, <2 x i32> %b) nounwind {
  %c = add <2 x i32> %a, %b ; add to prevent folding into extload
  %shl = shl <2 x i32> %c, <i32 31, i32 31>
  %ashr = ashr <2 x i32> %shl, <i32 31, i32 31>
  store <2 x i32> %ashr, <2 x i32> addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}sext_in_reg_v4i1_to_v4i32:
; SI: s_bfe_i32 {{s[0-9]+}}, {{s[0-9]+}}, 0x10000
; SI: s_bfe_i32 {{s[0-9]+}}, {{s[0-9]+}}, 0x10000
; SI: s_bfe_i32 {{s[0-9]+}}, {{s[0-9]+}}, 0x10000
; SI: s_bfe_i32 {{s[0-9]+}}, {{s[0-9]+}}, 0x10000
; SI: buffer_store_dwordx4

; EG: MEM_{{.*}} STORE_{{.*}} [[RES:T[0-9]+]]{{\.[XYZW][XYZW][XYZW][XYZW]}}, [[ADDR:T[0-9]+.[XYZW]]]
; EG: BFE_INT [[RES]]
; EG: BFE_INT [[RES]]
; EG: BFE_INT [[RES]]
; EG: BFE_INT [[RES]]
; EG: LSHR {{\*?}} [[ADDR]]
define void @sext_in_reg_v4i1_to_v4i32(<4 x i32> addrspace(1)* %out, <4 x i32> %a, <4 x i32> %b) nounwind {
  %c = add <4 x i32> %a, %b ; add to prevent folding into extload
  %shl = shl <4 x i32> %c, <i32 31, i32 31, i32 31, i32 31>
  %ashr = ashr <4 x i32> %shl, <i32 31, i32 31, i32 31, i32 31>
  store <4 x i32> %ashr, <4 x i32> addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}sext_in_reg_v2i8_to_v2i32:
; SI: s_sext_i32_i8 {{s[0-9]+}}, {{s[0-9]+}}
; SI: s_sext_i32_i8 {{s[0-9]+}}, {{s[0-9]+}}
; SI: buffer_store_dwordx2

; EG: MEM_{{.*}} STORE_{{.*}} [[RES:T[0-9]+]]{{\.[XYZW][XYZW]}}, [[ADDR:T[0-9]+.[XYZW]]]
; EG: BFE_INT [[RES]]
; EG: BFE_INT [[RES]]
; EG: LSHR {{\*?}} [[ADDR]]
define void @sext_in_reg_v2i8_to_v2i32(<2 x i32> addrspace(1)* %out, <2 x i32> %a, <2 x i32> %b) nounwind {
  %c = add <2 x i32> %a, %b ; add to prevent folding into extload
  %shl = shl <2 x i32> %c, <i32 24, i32 24>
  %ashr = ashr <2 x i32> %shl, <i32 24, i32 24>
  store <2 x i32> %ashr, <2 x i32> addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}sext_in_reg_v4i8_to_v4i32:
; SI: s_sext_i32_i8 {{s[0-9]+}}, {{s[0-9]+}}
; SI: s_sext_i32_i8 {{s[0-9]+}}, {{s[0-9]+}}
; SI: s_sext_i32_i8 {{s[0-9]+}}, {{s[0-9]+}}
; SI: s_sext_i32_i8 {{s[0-9]+}}, {{s[0-9]+}}
; SI: buffer_store_dwordx4

; EG: MEM_{{.*}} STORE_{{.*}} [[RES:T[0-9]+]]{{\.[XYZW][XYZW][XYZW][XYZW]}}, [[ADDR:T[0-9]+.[XYZW]]]
; EG: BFE_INT [[RES]]
; EG: BFE_INT [[RES]]
; EG: BFE_INT [[RES]]
; EG: BFE_INT [[RES]]
; EG: LSHR {{\*?}} [[ADDR]]
define void @sext_in_reg_v4i8_to_v4i32(<4 x i32> addrspace(1)* %out, <4 x i32> %a, <4 x i32> %b) nounwind {
  %c = add <4 x i32> %a, %b ; add to prevent folding into extload
  %shl = shl <4 x i32> %c, <i32 24, i32 24, i32 24, i32 24>
  %ashr = ashr <4 x i32> %shl, <i32 24, i32 24, i32 24, i32 24>
  store <4 x i32> %ashr, <4 x i32> addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}sext_in_reg_v2i16_to_v2i32:
; SI: s_sext_i32_i16 {{s[0-9]+}}, {{s[0-9]+}}
; SI: s_sext_i32_i16 {{s[0-9]+}}, {{s[0-9]+}}
; SI: buffer_store_dwordx2

; EG: MEM_{{.*}} STORE_{{.*}} [[RES:T[0-9]+]]{{\.[XYZW][XYZW]}}, [[ADDR:T[0-9]+.[XYZW]]]
; EG: BFE_INT [[RES]]
; EG: BFE_INT [[RES]]
; EG: LSHR {{\*?}} [[ADDR]]
define void @sext_in_reg_v2i16_to_v2i32(<2 x i32> addrspace(1)* %out, <2 x i32> %a, <2 x i32> %b) nounwind {
  %c = add <2 x i32> %a, %b ; add to prevent folding into extload
  %shl = shl <2 x i32> %c, <i32 16, i32 16>
  %ashr = ashr <2 x i32> %shl, <i32 16, i32 16>
  store <2 x i32> %ashr, <2 x i32> addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}testcase:
define void @testcase(i8 addrspace(1)* %out, i8 %a) nounwind {
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
define void @testcase_3(i8 addrspace(1)* %out, i8 %a) nounwind {
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
; SI: v_bfe_i32 [[EXTRACT:v[0-9]+]], {{v[0-9]+}}, 0, 8
; SI: v_bfe_i32 [[EXTRACT:v[0-9]+]], {{v[0-9]+}}, 0, 8
; SI: v_bfe_i32 [[EXTRACT:v[0-9]+]], {{v[0-9]+}}, 0, 8
; SI: v_bfe_i32 [[EXTRACT:v[0-9]+]], {{v[0-9]+}}, 0, 8
define void @vgpr_sext_in_reg_v4i8_to_v4i32(<4 x i32> addrspace(1)* %out, <4 x i32> addrspace(1)* %a, <4 x i32> addrspace(1)* %b) nounwind {
  %loada = load <4 x i32>, <4 x i32> addrspace(1)* %a, align 16
  %loadb = load <4 x i32>, <4 x i32> addrspace(1)* %b, align 16
  %c = add <4 x i32> %loada, %loadb ; add to prevent folding into extload
  %shl = shl <4 x i32> %c, <i32 24, i32 24, i32 24, i32 24>
  %ashr = ashr <4 x i32> %shl, <i32 24, i32 24, i32 24, i32 24>
  store <4 x i32> %ashr, <4 x i32> addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}vgpr_sext_in_reg_v4i16_to_v4i32:
; SI: v_bfe_i32 [[EXTRACT:v[0-9]+]], {{v[0-9]+}}, 0, 16
; SI: v_bfe_i32 [[EXTRACT:v[0-9]+]], {{v[0-9]+}}, 0, 16
define void @vgpr_sext_in_reg_v4i16_to_v4i32(<4 x i32> addrspace(1)* %out, <4 x i32> addrspace(1)* %a, <4 x i32> addrspace(1)* %b) nounwind {
  %loada = load <4 x i32>, <4 x i32> addrspace(1)* %a, align 16
  %loadb = load <4 x i32>, <4 x i32> addrspace(1)* %b, align 16
  %c = add <4 x i32> %loada, %loadb ; add to prevent folding into extload
  %shl = shl <4 x i32> %c, <i32 16, i32 16, i32 16, i32 16>
  %ashr = ashr <4 x i32> %shl, <i32 16, i32 16, i32 16, i32 16>
  store <4 x i32> %ashr, <4 x i32> addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}sext_in_reg_to_illegal_type:
; SI: buffer_load_sbyte
; SI: v_max_i32
; SI-NOT: bfe
; SI: buffer_store_short
define void @sext_in_reg_to_illegal_type(i16 addrspace(1)* nocapture %out, i8 addrspace(1)* nocapture %src) nounwind {
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
; SI-NOT: {{[^@]}}bfe
; SI: s_endpgm
define void @bfe_0_width(i32 addrspace(1)* %out, i32 addrspace(1)* %ptr) nounwind {
  %load = load i32, i32 addrspace(1)* %ptr, align 4
  %bfe = call i32 @llvm.AMDGPU.bfe.i32(i32 %load, i32 8, i32 0) nounwind readnone
  store i32 %bfe, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}bfe_8_bfe_8:
; SI: v_bfe_i32
; SI-NOT: {{[^@]}}bfe
; SI: s_endpgm
define void @bfe_8_bfe_8(i32 addrspace(1)* %out, i32 addrspace(1)* %ptr) nounwind {
  %load = load i32, i32 addrspace(1)* %ptr, align 4
  %bfe0 = call i32 @llvm.AMDGPU.bfe.i32(i32 %load, i32 0, i32 8) nounwind readnone
  %bfe1 = call i32 @llvm.AMDGPU.bfe.i32(i32 %bfe0, i32 0, i32 8) nounwind readnone
  store i32 %bfe1, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}bfe_8_bfe_16:
; SI: v_bfe_i32 v{{[0-9]+}}, v{{[0-9]+}}, 0, 8
; SI: s_endpgm
define void @bfe_8_bfe_16(i32 addrspace(1)* %out, i32 addrspace(1)* %ptr) nounwind {
  %load = load i32, i32 addrspace(1)* %ptr, align 4
  %bfe0 = call i32 @llvm.AMDGPU.bfe.i32(i32 %load, i32 0, i32 8) nounwind readnone
  %bfe1 = call i32 @llvm.AMDGPU.bfe.i32(i32 %bfe0, i32 0, i32 16) nounwind readnone
  store i32 %bfe1, i32 addrspace(1)* %out, align 4
  ret void
}

; This really should be folded into 1
; FUNC-LABEL: {{^}}bfe_16_bfe_8:
; SI: v_bfe_i32 v{{[0-9]+}}, v{{[0-9]+}}, 0, 8
; SI-NOT: {{[^@]}}bfe
; SI: s_endpgm
define void @bfe_16_bfe_8(i32 addrspace(1)* %out, i32 addrspace(1)* %ptr) nounwind {
  %load = load i32, i32 addrspace(1)* %ptr, align 4
  %bfe0 = call i32 @llvm.AMDGPU.bfe.i32(i32 %load, i32 0, i32 16) nounwind readnone
  %bfe1 = call i32 @llvm.AMDGPU.bfe.i32(i32 %bfe0, i32 0, i32 8) nounwind readnone
  store i32 %bfe1, i32 addrspace(1)* %out, align 4
  ret void
}

; Make sure there isn't a redundant BFE
; FUNC-LABEL: {{^}}sext_in_reg_i8_to_i32_bfe:
; SI: s_sext_i32_i8 s{{[0-9]+}}, s{{[0-9]+}}
; SI-NOT: {{[^@]}}bfe
; SI: s_endpgm
define void @sext_in_reg_i8_to_i32_bfe(i32 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %c = add i32 %a, %b ; add to prevent folding into extload
  %bfe = call i32 @llvm.AMDGPU.bfe.i32(i32 %c, i32 0, i32 8) nounwind readnone
  %shl = shl i32 %bfe, 24
  %ashr = ashr i32 %shl, 24
  store i32 %ashr, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}sext_in_reg_i8_to_i32_bfe_wrong:
define void @sext_in_reg_i8_to_i32_bfe_wrong(i32 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %c = add i32 %a, %b ; add to prevent folding into extload
  %bfe = call i32 @llvm.AMDGPU.bfe.i32(i32 %c, i32 8, i32 0) nounwind readnone
  %shl = shl i32 %bfe, 24
  %ashr = ashr i32 %shl, 24
  store i32 %ashr, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}sextload_i8_to_i32_bfe:
; SI: buffer_load_sbyte
; SI-NOT: {{[^@]}}bfe
; SI: s_endpgm
define void @sextload_i8_to_i32_bfe(i32 addrspace(1)* %out, i8 addrspace(1)* %ptr) nounwind {
  %load = load i8, i8 addrspace(1)* %ptr, align 1
  %sext = sext i8 %load to i32
  %bfe = call i32 @llvm.AMDGPU.bfe.i32(i32 %sext, i32 0, i32 8) nounwind readnone
  %shl = shl i32 %bfe, 24
  %ashr = ashr i32 %shl, 24
  store i32 %ashr, i32 addrspace(1)* %out, align 4
  ret void
}

; SI: .text
; FUNC-LABEL: {{^}}sextload_i8_to_i32_bfe_0:{{.*$}}
; SI-NOT: {{[^@]}}bfe
; SI: s_endpgm
define void @sextload_i8_to_i32_bfe_0(i32 addrspace(1)* %out, i8 addrspace(1)* %ptr) nounwind {
  %load = load i8, i8 addrspace(1)* %ptr, align 1
  %sext = sext i8 %load to i32
  %bfe = call i32 @llvm.AMDGPU.bfe.i32(i32 %sext, i32 8, i32 0) nounwind readnone
  %shl = shl i32 %bfe, 24
  %ashr = ashr i32 %shl, 24
  store i32 %ashr, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}sext_in_reg_i1_bfe_offset_0:
; SI-NOT: shr
; SI-NOT: shl
; SI: v_bfe_i32 v{{[0-9]+}}, v{{[0-9]+}}, 0, 1
; SI: s_endpgm
define void @sext_in_reg_i1_bfe_offset_0(i32 addrspace(1)* %out, i32 addrspace(1)* %in) nounwind {
  %x = load i32, i32 addrspace(1)* %in, align 4
  %shl = shl i32 %x, 31
  %shr = ashr i32 %shl, 31
  %bfe = call i32 @llvm.AMDGPU.bfe.i32(i32 %shr, i32 0, i32 1)
  store i32 %bfe, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}sext_in_reg_i1_bfe_offset_1:
; SI: buffer_load_dword
; SI-NOT: shl
; SI-NOT: shr
; SI: v_bfe_i32 v{{[0-9]+}}, v{{[0-9]+}}, 1, 1
; SI: s_endpgm
define void @sext_in_reg_i1_bfe_offset_1(i32 addrspace(1)* %out, i32 addrspace(1)* %in) nounwind {
  %x = load i32, i32 addrspace(1)* %in, align 4
  %shl = shl i32 %x, 30
  %shr = ashr i32 %shl, 30
  %bfe = call i32 @llvm.AMDGPU.bfe.i32(i32 %shr, i32 1, i32 1)
  store i32 %bfe, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}sext_in_reg_i2_bfe_offset_1:
; SI: buffer_load_dword
; SI-NOT: v_lshl
; SI-NOT: v_ashr
; SI: v_bfe_i32 v{{[0-9]+}}, v{{[0-9]+}}, 0, 2
; SI: v_bfe_i32 v{{[0-9]+}}, v{{[0-9]+}}, 1, 2
; SI: s_endpgm
define void @sext_in_reg_i2_bfe_offset_1(i32 addrspace(1)* %out, i32 addrspace(1)* %in) nounwind {
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
; SI-DAG: v_bfe_i32 v[[LO:[0-9]+]], v[[VAL_LO]], 0, 1
; SI-DAG: v_ashrrev_i32_e32 v[[HI:[0-9]+]], 31, v[[LO]]
; SI-DAG: v_and_b32_e32 v[[RESULT_LO:[0-9]+]], s{{[0-9]+}}, v[[LO]]
; SI-DAG: v_and_b32_e32 v[[RESULT_HI:[0-9]+]], s{{[0-9]+}}, v[[HI]]
; SI: buffer_store_dwordx2 v{{\[}}[[RESULT_LO]]:[[RESULT_HI]]{{\]}}
define void @v_sext_in_reg_i1_to_i64_move_use(i64 addrspace(1)* %out, i64 addrspace(1)* %aptr, i64 addrspace(1)* %bptr, i64 %s.val) nounwind {
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
; SI-DAG: v_ashrrev_i32_e32 v[[SHR:[0-9]+]], 31, v[[LO]]
; SI-DAG: v_and_b32_e32 v[[RESULT_LO:[0-9]+]], s{{[0-9]+}}, v[[LO]]
; SI-DAG: v_and_b32_e32 v[[RESULT_HI:[0-9]+]], s{{[0-9]+}}, v[[SHR]]
; SI: buffer_store_dwordx2 v{{\[}}[[RESULT_LO]]:[[RESULT_HI]]{{\]}}
define void @v_sext_in_reg_i32_to_i64_move_use(i64 addrspace(1)* %out, i64 addrspace(1)* %aptr, i64 addrspace(1)* %bptr, i64 %s.val) nounwind {
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
