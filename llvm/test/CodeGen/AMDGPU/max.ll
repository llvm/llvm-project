; RUN: llc -march=amdgcn -mcpu=SI < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

declare i32 @llvm.r600.read.tidig.x() nounwind readnone

; FUNC-LABEL: {{^}}v_test_imax_sge_i32:
; SI: v_max_i32_e32
define void @v_test_imax_sge_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %aptr, i32 addrspace(1)* %bptr) nounwind {
  %tid = call i32 @llvm.r600.read.tidig.x() nounwind readnone
  %gep0 = getelementptr i32, i32 addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr i32, i32 addrspace(1)* %bptr, i32 %tid
  %outgep = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %a = load i32, i32 addrspace(1)* %gep0, align 4
  %b = load i32, i32 addrspace(1)* %gep1, align 4
  %cmp = icmp sge i32 %a, %b
  %val = select i1 %cmp, i32 %a, i32 %b
  store i32 %val, i32 addrspace(1)* %outgep, align 4
  ret void
}

; FUNC-LABEL: {{^}}v_test_imax_sge_v4i32:
; SI: v_max_i32_e32
; SI: v_max_i32_e32
; SI: v_max_i32_e32
; SI: v_max_i32_e32
define void @v_test_imax_sge_v4i32(<4 x i32> addrspace(1)* %out, <4 x i32> addrspace(1)* %aptr, <4 x i32> addrspace(1)* %bptr) nounwind {
  %tid = call i32 @llvm.r600.read.tidig.x() nounwind readnone
  %gep0 = getelementptr <4 x i32>, <4 x i32> addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr <4 x i32>, <4 x i32> addrspace(1)* %bptr, i32 %tid
  %outgep = getelementptr <4 x i32>, <4 x i32> addrspace(1)* %out, i32 %tid
  %a = load <4 x i32>, <4 x i32> addrspace(1)* %gep0, align 4
  %b = load <4 x i32>, <4 x i32> addrspace(1)* %gep1, align 4
  %cmp = icmp sge <4 x i32> %a, %b
  %val = select <4 x i1> %cmp, <4 x i32> %a, <4 x i32> %b
  store <4 x i32> %val, <4 x i32> addrspace(1)* %outgep, align 4
  ret void
}

; FUNC-LABEL: @s_test_imax_sge_i32
; SI: s_max_i32
define void @s_test_imax_sge_i32(i32 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %cmp = icmp sge i32 %a, %b
  %val = select i1 %cmp, i32 %a, i32 %b
  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}s_test_imax_sge_imm_i32:
; SI: s_max_i32 {{s[0-9]+}}, {{s[0-9]+}}, 9
define void @s_test_imax_sge_imm_i32(i32 addrspace(1)* %out, i32 %a) nounwind {
  %cmp = icmp sge i32 %a, 9
  %val = select i1 %cmp, i32 %a, i32 9
  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}s_test_imax_sgt_imm_i32:
; SI: s_max_i32 {{s[0-9]+}}, {{s[0-9]+}}, 9
define void @s_test_imax_sgt_imm_i32(i32 addrspace(1)* %out, i32 %a) nounwind {
  %cmp = icmp sgt i32 %a, 9
  %val = select i1 %cmp, i32 %a, i32 9
  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}s_test_imax_sgt_imm_v2i32:
; SI: s_max_i32 {{s[0-9]+}}, {{s[0-9]+}}, 9
; SI: s_max_i32 {{s[0-9]+}}, {{s[0-9]+}}, 9
define void @s_test_imax_sgt_imm_v2i32(<2 x i32> addrspace(1)* %out, <2 x i32> %a) nounwind {
  %cmp = icmp sgt <2 x i32> %a, <i32 9, i32 9>
  %val = select <2 x i1> %cmp, <2 x i32> %a, <2 x i32> <i32 9, i32 9>
  store <2 x i32> %val, <2 x i32> addrspace(1)* %out, align 4
  ret void
}
; FUNC-LABEL: @v_test_imax_sgt_i32
; SI: v_max_i32_e32
define void @v_test_imax_sgt_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %aptr, i32 addrspace(1)* %bptr) nounwind {
  %tid = call i32 @llvm.r600.read.tidig.x() nounwind readnone
  %gep0 = getelementptr i32, i32 addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr i32, i32 addrspace(1)* %bptr, i32 %tid
  %outgep = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %a = load i32, i32 addrspace(1)* %gep0, align 4
  %b = load i32, i32 addrspace(1)* %gep1, align 4
  %cmp = icmp sgt i32 %a, %b
  %val = select i1 %cmp, i32 %a, i32 %b
  store i32 %val, i32 addrspace(1)* %outgep, align 4
  ret void
}

; FUNC-LABEL: @s_test_imax_sgt_i32
; SI: s_max_i32
define void @s_test_imax_sgt_i32(i32 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %cmp = icmp sgt i32 %a, %b
  %val = select i1 %cmp, i32 %a, i32 %b
  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @v_test_umax_uge_i32
; SI: v_max_u32_e32
define void @v_test_umax_uge_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %aptr, i32 addrspace(1)* %bptr) nounwind {
  %tid = call i32 @llvm.r600.read.tidig.x() nounwind readnone
  %gep0 = getelementptr i32, i32 addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr i32, i32 addrspace(1)* %bptr, i32 %tid
  %outgep = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %a = load i32, i32 addrspace(1)* %gep0, align 4
  %b = load i32, i32 addrspace(1)* %gep1, align 4
  %cmp = icmp uge i32 %a, %b
  %val = select i1 %cmp, i32 %a, i32 %b
  store i32 %val, i32 addrspace(1)* %outgep, align 4
  ret void
}

; FUNC-LABEL: @s_test_umax_uge_i32
; SI: s_max_u32
define void @s_test_umax_uge_i32(i32 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %cmp = icmp uge i32 %a, %b
  %val = select i1 %cmp, i32 %a, i32 %b
  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}s_test_umax_uge_v3i32:
; SI: s_max_u32
; SI: s_max_u32
; SI: s_max_u32
; SI-NOT: s_max_u32
; SI: s_endpgm
define void @s_test_umax_uge_v3i32(<3 x i32> addrspace(1)* %out, <3 x i32> %a, <3 x i32> %b) nounwind {
  %cmp = icmp uge <3 x i32> %a, %b
  %val = select <3 x i1> %cmp, <3 x i32> %a, <3 x i32> %b
  store <3 x i32> %val, <3 x i32> addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @v_test_umax_ugt_i32
; SI: v_max_u32_e32
define void @v_test_umax_ugt_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %aptr, i32 addrspace(1)* %bptr) nounwind {
  %tid = call i32 @llvm.r600.read.tidig.x() nounwind readnone
  %gep0 = getelementptr i32, i32 addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr i32, i32 addrspace(1)* %bptr, i32 %tid
  %outgep = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %a = load i32, i32 addrspace(1)* %gep0, align 4
  %b = load i32, i32 addrspace(1)* %gep1, align 4
  %cmp = icmp ugt i32 %a, %b
  %val = select i1 %cmp, i32 %a, i32 %b
  store i32 %val, i32 addrspace(1)* %outgep, align 4
  ret void
}

; FUNC-LABEL: {{^}}s_test_umax_ugt_i32:
; SI: s_max_u32
define void @s_test_umax_ugt_i32(i32 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %cmp = icmp ugt i32 %a, %b
  %val = select i1 %cmp, i32 %a, i32 %b
  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}s_test_umax_ugt_imm_v2i32:
; SI: s_max_u32 {{s[0-9]+}}, {{s[0-9]+}}, 15
; SI: s_max_u32 {{s[0-9]+}}, {{s[0-9]+}}, 23
define void @s_test_umax_ugt_imm_v2i32(<2 x i32> addrspace(1)* %out, <2 x i32> %a) nounwind {
  %cmp = icmp ugt <2 x i32> %a, <i32 15, i32 23>
  %val = select <2 x i1> %cmp, <2 x i32> %a, <2 x i32> <i32 15, i32 23>
  store <2 x i32> %val, <2 x i32> addrspace(1)* %out, align 4
  ret void
}

; Make sure redundant and removed
; FUNC-LABEL: {{^}}simplify_demanded_bits_test_umax_ugt_i16:
; SI-DAG: s_load_dword [[A:s[0-9]+]], {{s\[[0-9]+:[0-9]+\]}}, 0xb
; SI-DAG: s_load_dword [[B:s[0-9]+]], {{s\[[0-9]+:[0-9]+\]}}, 0xc
; SI: s_max_u32 [[MAX:s[0-9]+]], [[A]], [[B]]
; SI-NEXT: v_mov_b32_e32 [[VMAX:v[0-9]+]], [[MAX]]
; SI-NEXT: buffer_store_dword [[VMAX]]
define void @simplify_demanded_bits_test_umax_ugt_i16(i32 addrspace(1)* %out, i16 zeroext %a, i16 zeroext %b) nounwind {
  %a.ext = zext i16 %a to i32
  %b.ext = zext i16 %b to i32
  %cmp = icmp ugt i32 %a.ext, %b.ext
  %val = select i1 %cmp, i32 %a.ext, i32 %b.ext
  %mask = and i32 %val, 65535
  store i32 %mask, i32 addrspace(1)* %out
  ret void
}

; Make sure redundant sign_extend_inreg removed.

; FUNC-LABEL: {{^}}simplify_demanded_bits_test_max_slt_i16:
; SI-DAG: s_load_dword [[A:s[0-9]+]], {{s\[[0-9]+:[0-9]+\]}}, 0xb
; SI-DAG: s_load_dword [[B:s[0-9]+]], {{s\[[0-9]+:[0-9]+\]}}, 0xc
; SI: s_max_i32 [[MAX:s[0-9]+]], [[A]], [[B]]
; SI-NEXT: v_mov_b32_e32 [[VMAX:v[0-9]+]], [[MAX]]
; SI-NEXT: buffer_store_dword [[VMAX]]
define void @simplify_demanded_bits_test_max_slt_i16(i32 addrspace(1)* %out, i16 signext %a, i16 signext %b) nounwind {
  %a.ext = sext i16 %a to i32
  %b.ext = sext i16 %b to i32
  %cmp = icmp sgt i32 %a.ext, %b.ext
  %val = select i1 %cmp, i32 %a.ext, i32 %b.ext
  %shl = shl i32 %val, 16
  %sextinreg = ashr i32 %shl, 16
  store i32 %sextinreg, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}s_test_imax_sge_i16:
; SI: s_max_i32
define void @s_test_imax_sge_i16(i16 addrspace(1)* %out, i16 %a, i16 %b) nounwind {
  %cmp = icmp sge i16 %a, %b
  %val = select i1 %cmp, i16 %a, i16 %b
  store i16 %val, i16 addrspace(1)* %out
  ret void
}
