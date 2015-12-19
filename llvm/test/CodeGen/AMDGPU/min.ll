; RUN: llc -march=amdgcn -mcpu=SI < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

declare i32 @llvm.r600.read.tidig.x() nounwind readnone

; FUNC-LABEL: {{^}}v_test_imin_sle_i32:
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

; FUNC-LABEL: {{^}}s_test_imin_sle_i32:
; SI: s_min_i32
define void @s_test_imin_sle_i32(i32 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %cmp = icmp sle i32 %a, %b
  %val = select i1 %cmp, i32 %a, i32 %b
  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}s_test_imin_sle_v1i32:
; SI: s_min_i32
define void @s_test_imin_sle_v1i32(<1 x i32> addrspace(1)* %out, <1 x i32> %a, <1 x i32> %b) nounwind {
  %cmp = icmp sle <1 x i32> %a, %b
  %val = select <1 x i1> %cmp, <1 x i32> %a, <1 x i32> %b
  store <1 x i32> %val, <1 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}s_test_imin_sle_v4i32:
; SI: s_min_i32
; SI: s_min_i32
; SI: s_min_i32
; SI: s_min_i32
define void @s_test_imin_sle_v4i32(<4 x i32> addrspace(1)* %out, <4 x i32> %a, <4 x i32> %b) nounwind {
  %cmp = icmp sle <4 x i32> %a, %b
  %val = select <4 x i1> %cmp, <4 x i32> %a, <4 x i32> %b
  store <4 x i32> %val, <4 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}s_test_imin_sle_i8:
; SI: s_load_dword
; SI: s_load_dword
; SI: s_sext_i32_i8
; SI: s_sext_i32_i8
; SI: s_min_i32
define void @s_test_imin_sle_i8(i8 addrspace(1)* %out, i8 %a, i8 %b) nounwind {
  %cmp = icmp sle i8 %a, %b
  %val = select i1 %cmp, i8 %a, i8 %b
  store i8 %val, i8 addrspace(1)* %out
  ret void
}

; XXX - should be able to use s_min if we stop unnecessarily doing
; extloads with mubuf instructions.

; FUNC-LABEL: {{^}}s_test_imin_sle_v4i8:
; SI: buffer_load_sbyte
; SI: buffer_load_sbyte
; SI: buffer_load_sbyte
; SI: buffer_load_sbyte
; SI: buffer_load_sbyte
; SI: buffer_load_sbyte
; SI: buffer_load_sbyte
; SI: buffer_load_sbyte

; SI: v_min_i32
; SI: v_min_i32
; SI: v_min_i32
; SI: v_min_i32

; SI: s_endpgm
define void @s_test_imin_sle_v4i8(<4 x i8> addrspace(1)* %out, <4 x i8> %a, <4 x i8> %b) nounwind {
  %cmp = icmp sle <4 x i8> %a, %b
  %val = select <4 x i1> %cmp, <4 x i8> %a, <4 x i8> %b
  store <4 x i8> %val, <4 x i8> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}s_test_imin_sle_v4i16:
; SI: v_min_i32
; SI: v_min_i32
; SI: v_min_i32
; SI: v_min_i32
define void @s_test_imin_sle_v4i16(<4 x i16> addrspace(1)* %out, <4 x i16> %a, <4 x i16> %b) nounwind {
  %cmp = icmp sle <4 x i16> %a, %b
  %val = select <4 x i1> %cmp, <4 x i16> %a, <4 x i16> %b
  store <4 x i16> %val, <4 x i16> addrspace(1)* %out
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

; FUNC-LABEL: {{^}}s_test_imin_slt_v2i32:
; SI: s_min_i32
; SI: s_min_i32
define void @s_test_imin_slt_v2i32(<2 x i32> addrspace(1)* %out, <2 x i32> %a, <2 x i32> %b) nounwind {
  %cmp = icmp slt <2 x i32> %a, %b
  %val = select <2 x i1> %cmp, <2 x i32> %a, <2 x i32> %b
  store <2 x i32> %val, <2 x i32> addrspace(1)* %out
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

; FUNC-LABEL: @v_test_umin_ule_v3i32
; SI: v_min_u32_e32
; SI: v_min_u32_e32
; SI: v_min_u32_e32
; SI-NOT: v_min_u32_e32
; SI: s_endpgm
define void @v_test_umin_ule_v3i32(<3 x i32> addrspace(1)* %out, <3 x i32> addrspace(1)* %aptr, <3 x i32> addrspace(1)* %bptr) nounwind {
  %tid = call i32 @llvm.r600.read.tidig.x() nounwind readnone
  %gep0 = getelementptr <3 x i32>, <3 x i32> addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr <3 x i32>, <3 x i32> addrspace(1)* %bptr, i32 %tid
  %outgep = getelementptr <3 x i32>, <3 x i32> addrspace(1)* %out, i32 %tid
  %a = load <3 x i32>, <3 x i32> addrspace(1)* %gep0
  %b = load <3 x i32>, <3 x i32> addrspace(1)* %gep1
  %cmp = icmp ule <3 x i32> %a, %b
  %val = select <3 x i1> %cmp, <3 x i32> %a, <3 x i32> %b
  store <3 x i32> %val, <3 x i32> addrspace(1)* %outgep
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

; FUNC-LABEL: {{^}}v_test_umin_ult_i8:
; SI: buffer_load_ubyte
; SI: buffer_load_ubyte
; SI: v_min_u32_e32
define void @v_test_umin_ult_i8(i8 addrspace(1)* %out, i8 addrspace(1)* %aptr, i8 addrspace(1)* %bptr) nounwind {
  %tid = call i32 @llvm.r600.read.tidig.x() nounwind readnone
  %gep0 = getelementptr i8, i8 addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr i8, i8 addrspace(1)* %bptr, i32 %tid
  %outgep = getelementptr i8, i8 addrspace(1)* %out, i32 %tid
  %a = load i8, i8 addrspace(1)* %gep0, align 1
  %b = load i8, i8 addrspace(1)* %gep1, align 1
  %cmp = icmp ult i8 %a, %b
  %val = select i1 %cmp, i8 %a, i8 %b
  store i8 %val, i8 addrspace(1)* %outgep, align 1
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


; FUNC-LABEL: @s_test_umin_ult_v1i32
; SI: s_min_u32
define void @s_test_umin_ult_v1i32(<1 x i32> addrspace(1)* %out, <1 x i32> %a, <1 x i32> %b) nounwind {
  %cmp = icmp ult <1 x i32> %a, %b
  %val = select <1 x i1> %cmp, <1 x i32> %a, <1 x i32> %b
  store <1 x i32> %val, <1 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}s_test_umin_ult_v8i32:
; SI: s_min_u32
; SI: s_min_u32
; SI: s_min_u32
; SI: s_min_u32
; SI: s_min_u32
; SI: s_min_u32
; SI: s_min_u32
; SI: s_min_u32
define void @s_test_umin_ult_v8i32(<8 x i32> addrspace(1)* %out, <8 x i32> %a, <8 x i32> %b) nounwind {
  %cmp = icmp ult <8 x i32> %a, %b
  %val = select <8 x i1> %cmp, <8 x i32> %a, <8 x i32> %b
  store <8 x i32> %val, <8 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}s_test_umin_ult_v8i16:
; SI: v_min_u32
; SI: v_min_u32
; SI: v_min_u32
; SI: v_min_u32
; SI: v_min_u32
; SI: v_min_u32
; SI: v_min_u32
; SI: v_min_u32
define void @s_test_umin_ult_v8i16(<8 x i16> addrspace(1)* %out, <8 x i16> %a, <8 x i16> %b) nounwind {
  %cmp = icmp ult <8 x i16> %a, %b
  %val = select <8 x i1> %cmp, <8 x i16> %a, <8 x i16> %b
  store <8 x i16> %val, <8 x i16> addrspace(1)* %out
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

; FUNC-LABEL: {{^}}s_test_imin_sle_i16:
; SI: s_min_i32
define void @s_test_imin_sle_i16(i16 addrspace(1)* %out, i16 %a, i16 %b) nounwind {
  %cmp = icmp sle i16 %a, %b
  %val = select i1 %cmp, i16 %a, i16 %b
  store i16 %val, i16 addrspace(1)* %out
  ret void
}
