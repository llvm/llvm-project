; RUN: llc -march=amdgcn -mcpu=SI < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

declare i32 @llvm.r600.read.tidig.x() nounwind readnone

; FUNC-LABEL: @v_test_imin_sle_i32
; SI: v_min_i32_e32
define void @v_test_imin_sle_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %aptr, i32 addrspace(1)* %bptr) nounwind {
  %tid = call i32 @llvm.r600.read.tidig.x() nounwind readnone
  %gep0 = getelementptr i32, i32 addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr i32, i32 addrspace(1)* %bptr, i32 %tid
  %outgep = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %a = load i32, i32 addrspace(1)* %gep0, align 4
  %b = load i32, i32 addrspace(1)* %gep1, align 4
  %cmp = icmp sle i32 %a, %b
  %val = select i1 %cmp, i32 %a, i32 %b
  store i32 %val, i32 addrspace(1)* %outgep, align 4
  ret void
}

; FUNC-LABEL: @s_test_imin_sle_i32
; SI: s_min_i32
define void @s_test_imin_sle_i32(i32 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %cmp = icmp sle i32 %a, %b
  %val = select i1 %cmp, i32 %a, i32 %b
  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @v_test_imin_slt_i32
; SI: v_min_i32_e32
define void @v_test_imin_slt_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %aptr, i32 addrspace(1)* %bptr) nounwind {
  %tid = call i32 @llvm.r600.read.tidig.x() nounwind readnone
  %gep0 = getelementptr i32, i32 addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr i32, i32 addrspace(1)* %bptr, i32 %tid
  %outgep = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %a = load i32, i32 addrspace(1)* %gep0, align 4
  %b = load i32, i32 addrspace(1)* %gep1, align 4
  %cmp = icmp slt i32 %a, %b
  %val = select i1 %cmp, i32 %a, i32 %b
  store i32 %val, i32 addrspace(1)* %outgep, align 4
  ret void
}

; FUNC-LABEL: @s_test_imin_slt_i32
; SI: s_min_i32
define void @s_test_imin_slt_i32(i32 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %cmp = icmp slt i32 %a, %b
  %val = select i1 %cmp, i32 %a, i32 %b
  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}s_test_imin_slt_imm_i32:
; SI: s_min_i32 {{s[0-9]+}}, {{s[0-9]+}}, 8
define void @s_test_imin_slt_imm_i32(i32 addrspace(1)* %out, i32 %a) nounwind {
  %cmp = icmp slt i32 %a, 8
  %val = select i1 %cmp, i32 %a, i32 8
  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}s_test_imin_sle_imm_i32:
; SI: s_min_i32 {{s[0-9]+}}, {{s[0-9]+}}, 8
define void @s_test_imin_sle_imm_i32(i32 addrspace(1)* %out, i32 %a) nounwind {
  %cmp = icmp sle i32 %a, 8
  %val = select i1 %cmp, i32 %a, i32 8
  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @v_test_umin_ule_i32
; SI: v_min_u32_e32
define void @v_test_umin_ule_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %aptr, i32 addrspace(1)* %bptr) nounwind {
  %tid = call i32 @llvm.r600.read.tidig.x() nounwind readnone
  %gep0 = getelementptr i32, i32 addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr i32, i32 addrspace(1)* %bptr, i32 %tid
  %outgep = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %a = load i32, i32 addrspace(1)* %gep0, align 4
  %b = load i32, i32 addrspace(1)* %gep1, align 4
  %cmp = icmp ule i32 %a, %b
  %val = select i1 %cmp, i32 %a, i32 %b
  store i32 %val, i32 addrspace(1)* %outgep, align 4
  ret void
}

; FUNC-LABEL: @s_test_umin_ule_i32
; SI: s_min_u32
define void @s_test_umin_ule_i32(i32 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %cmp = icmp ule i32 %a, %b
  %val = select i1 %cmp, i32 %a, i32 %b
  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @v_test_umin_ult_i32
; SI: v_min_u32_e32
define void @v_test_umin_ult_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %aptr, i32 addrspace(1)* %bptr) nounwind {
  %tid = call i32 @llvm.r600.read.tidig.x() nounwind readnone
  %gep0 = getelementptr i32, i32 addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr i32, i32 addrspace(1)* %bptr, i32 %tid
  %outgep = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %a = load i32, i32 addrspace(1)* %gep0, align 4
  %b = load i32, i32 addrspace(1)* %gep1, align 4
  %cmp = icmp ult i32 %a, %b
  %val = select i1 %cmp, i32 %a, i32 %b
  store i32 %val, i32 addrspace(1)* %outgep, align 4
  ret void
}

; FUNC-LABEL: @s_test_umin_ult_i32
; SI: s_min_u32
define void @s_test_umin_ult_i32(i32 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %cmp = icmp ult i32 %a, %b
  %val = select i1 %cmp, i32 %a, i32 %b
  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @v_test_umin_ult_i32_multi_use
; SI-NOT: v_min
; SI: v_cmp_lt_u32
; SI-NEXT: v_cndmask_b32
; SI-NOT: v_min
; SI: s_endpgm
define void @v_test_umin_ult_i32_multi_use(i32 addrspace(1)* %out0, i1 addrspace(1)* %out1, i32 addrspace(1)* %aptr, i32 addrspace(1)* %bptr) nounwind {
  %tid = call i32 @llvm.r600.read.tidig.x() nounwind readnone
  %gep0 = getelementptr i32, i32 addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr i32, i32 addrspace(1)* %bptr, i32 %tid
  %outgep0 = getelementptr i32, i32 addrspace(1)* %out0, i32 %tid
  %outgep1 = getelementptr i1, i1 addrspace(1)* %out1, i32 %tid
  %a = load i32, i32 addrspace(1)* %gep0, align 4
  %b = load i32, i32 addrspace(1)* %gep1, align 4
  %cmp = icmp ult i32 %a, %b
  %val = select i1 %cmp, i32 %a, i32 %b
  store i32 %val, i32 addrspace(1)* %outgep0, align 4
  store i1 %cmp, i1 addrspace(1)* %outgep1
  ret void
}

; Make sure redundant and removed
; FUNC-LABEL: {{^}}simplify_demanded_bits_test_umin_ult_i16:
; SI-DAG: s_load_dword [[A:s[0-9]+]], {{s\[[0-9]+:[0-9]+\]}}, 0xb
; SI-DAG: s_load_dword [[B:s[0-9]+]], {{s\[[0-9]+:[0-9]+\]}}, 0xc
; SI: s_min_u32 [[MIN:s[0-9]+]], [[A]], [[B]]
; SI-NEXT: v_mov_b32_e32 [[VMIN:v[0-9]+]], [[MIN]]
; SI-NEXT: buffer_store_dword [[VMIN]]
define void @simplify_demanded_bits_test_umin_ult_i16(i32 addrspace(1)* %out, i16 zeroext %a, i16 zeroext %b) nounwind {
  %a.ext = zext i16 %a to i32
  %b.ext = zext i16 %b to i32
  %cmp = icmp ult i32 %a.ext, %b.ext
  %val = select i1 %cmp, i32 %a.ext, i32 %b.ext
  %mask = and i32 %val, 65535
  store i32 %mask, i32 addrspace(1)* %out
  ret void
}

; Make sure redundant sign_extend_inreg removed.

; FUNC-LABEL: {{^}}simplify_demanded_bits_test_min_slt_i16:
; SI-DAG: s_load_dword [[A:s[0-9]+]], {{s\[[0-9]+:[0-9]+\]}}, 0xb
; SI-DAG: s_load_dword [[B:s[0-9]+]], {{s\[[0-9]+:[0-9]+\]}}, 0xc
; SI: s_min_i32 [[MIN:s[0-9]+]], [[A]], [[B]]
; SI-NEXT: v_mov_b32_e32 [[VMIN:v[0-9]+]], [[MIN]]
; SI-NEXT: buffer_store_dword [[VMIN]]
define void @simplify_demanded_bits_test_min_slt_i16(i32 addrspace(1)* %out, i16 signext %a, i16 signext %b) nounwind {
  %a.ext = sext i16 %a to i32
  %b.ext = sext i16 %b to i32
  %cmp = icmp slt i32 %a.ext, %b.ext
  %val = select i1 %cmp, i32 %a.ext, i32 %b.ext
  %shl = shl i32 %val, 16
  %sextinreg = ashr i32 %shl, 16
  store i32 %sextinreg, i32 addrspace(1)* %out
  ret void
}

; FIXME: Should get match min/max through extends inserted by
; legalization.

; FUNC-LABEL: {{^}}s_test_imin_sle_i16:
; SI: s_sext_i32_i16
; SI: s_sext_i32_i16
; SI: v_cmp_le_i32_e32
; SI: v_cndmask_b32
define void @s_test_imin_sle_i16(i16 addrspace(1)* %out, i16 %a, i16 %b) nounwind {
  %cmp = icmp sle i16 %a, %b
  %val = select i1 %cmp, i16 %a, i16 %b
  store i16 %val, i16 addrspace(1)* %out
  ret void
}
