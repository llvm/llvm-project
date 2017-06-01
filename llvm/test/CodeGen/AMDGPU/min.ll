; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=cypress < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}v_test_imin_sle_i32:
; GCN: v_min_i32_e32

; EG: MIN_INT
define void @v_test_imin_sle_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %aptr, i32 addrspace(1)* %bptr) nounwind {
  %a = load i32, i32 addrspace(1)* %aptr, align 4
  %b = load i32, i32 addrspace(1)* %bptr, align 4
  %cmp = icmp sle i32 %a, %b
  %val = select i1 %cmp, i32 %a, i32 %b
  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}s_test_imin_sle_i32:
; GCN: s_min_i32

; EG: MIN_INT
define void @s_test_imin_sle_i32(i32 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %cmp = icmp sle i32 %a, %b
  %val = select i1 %cmp, i32 %a, i32 %b
  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}s_test_imin_sle_v1i32:
; GCN: s_min_i32

; EG: MIN_INT
define void @s_test_imin_sle_v1i32(<1 x i32> addrspace(1)* %out, <1 x i32> %a, <1 x i32> %b) nounwind {
  %cmp = icmp sle <1 x i32> %a, %b
  %val = select <1 x i1> %cmp, <1 x i32> %a, <1 x i32> %b
  store <1 x i32> %val, <1 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}s_test_imin_sle_v4i32:
; GCN: s_min_i32
; GCN: s_min_i32
; GCN: s_min_i32
; GCN: s_min_i32

; EG: MIN_INT
; EG: MIN_INT
; EG: MIN_INT
; EG: MIN_INT
define void @s_test_imin_sle_v4i32(<4 x i32> addrspace(1)* %out, <4 x i32> %a, <4 x i32> %b) nounwind {
  %cmp = icmp sle <4 x i32> %a, %b
  %val = select <4 x i1> %cmp, <4 x i32> %a, <4 x i32> %b
  store <4 x i32> %val, <4 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}s_test_imin_sle_i8:
; GCN: s_load_dword
; GCN: s_load_dword
; GCN: s_sext_i32_i8
; GCN: s_sext_i32_i8
; GCN: s_min_i32
define void @s_test_imin_sle_i8(i8 addrspace(1)* %out, i8 %a, i8 %b) nounwind {
  %cmp = icmp sle i8 %a, %b
  %val = select i1 %cmp, i8 %a, i8 %b
  store i8 %val, i8 addrspace(1)* %out
  ret void
}

; XXX - should be able to use s_min if we stop unnecessarily doing
; extloads with mubuf instructions.

; FUNC-LABEL: {{^}}s_test_imin_sle_v4i8:
; GCN: buffer_load_sbyte
; GCN: buffer_load_sbyte
; GCN: buffer_load_sbyte
; GCN: buffer_load_sbyte
; GCN: buffer_load_sbyte
; GCN: buffer_load_sbyte
; GCN: buffer_load_sbyte
; GCN: buffer_load_sbyte

; SI: v_min_i32
; SI: v_min_i32
; SI: v_min_i32
; SI: v_min_i32

; VI: v_min_i32
; VI: v_min_i32
; VI: v_min_i32
; VI: v_min_i32

; GCN: s_endpgm

; EG: MIN_INT
; EG: MIN_INT
; EG: MIN_INT
; EG: MIN_INT
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

; EG: MIN_INT
; EG: MIN_INT
; EG: MIN_INT
; EG: MIN_INT
define void @s_test_imin_sle_v4i16(<4 x i16> addrspace(1)* %out, <4 x i16> %a, <4 x i16> %b) nounwind {
  %cmp = icmp sle <4 x i16> %a, %b
  %val = select <4 x i1> %cmp, <4 x i16> %a, <4 x i16> %b
  store <4 x i16> %val, <4 x i16> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: @v_test_imin_slt_i32
; GCN: v_min_i32_e32

; EG: MIN_INT
define void @v_test_imin_slt_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %aptr, i32 addrspace(1)* %bptr) nounwind {
  %a = load i32, i32 addrspace(1)* %aptr, align 4
  %b = load i32, i32 addrspace(1)* %bptr, align 4
  %cmp = icmp slt i32 %a, %b
  %val = select i1 %cmp, i32 %a, i32 %b
  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @s_test_imin_slt_i32
; GCN: s_min_i32

; EG: MIN_INT
define void @s_test_imin_slt_i32(i32 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %cmp = icmp slt i32 %a, %b
  %val = select i1 %cmp, i32 %a, i32 %b
  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}s_test_imin_slt_v2i32:
; GCN: s_min_i32
; GCN: s_min_i32

; EG: MIN_INT
; EG: MIN_INT
define void @s_test_imin_slt_v2i32(<2 x i32> addrspace(1)* %out, <2 x i32> %a, <2 x i32> %b) nounwind {
  %cmp = icmp slt <2 x i32> %a, %b
  %val = select <2 x i1> %cmp, <2 x i32> %a, <2 x i32> %b
  store <2 x i32> %val, <2 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}s_test_imin_slt_imm_i32:
; GCN: s_min_i32 {{s[0-9]+}}, {{s[0-9]+}}, 8

; EG: MIN_INT {{.*}}literal.{{[xyzw]}}
define void @s_test_imin_slt_imm_i32(i32 addrspace(1)* %out, i32 %a) nounwind {
  %cmp = icmp slt i32 %a, 8
  %val = select i1 %cmp, i32 %a, i32 8
  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}s_test_imin_sle_imm_i32:
; GCN: s_min_i32 {{s[0-9]+}}, {{s[0-9]+}}, 8

; EG: MIN_INT {{.*}}literal.{{[xyzw]}}
define void @s_test_imin_sle_imm_i32(i32 addrspace(1)* %out, i32 %a) nounwind {
  %cmp = icmp sle i32 %a, 8
  %val = select i1 %cmp, i32 %a, i32 8
  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @v_test_umin_ule_i32
; GCN: v_min_u32_e32

; EG: MIN_UINT
define void @v_test_umin_ule_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %aptr, i32 addrspace(1)* %bptr) nounwind {
  %a = load i32, i32 addrspace(1)* %aptr, align 4
  %b = load i32, i32 addrspace(1)* %bptr, align 4
  %cmp = icmp ule i32 %a, %b
  %val = select i1 %cmp, i32 %a, i32 %b
  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @v_test_umin_ule_v3i32
; GCN: v_min_u32_e32
; GCN: v_min_u32_e32
; GCN: v_min_u32_e32
; SI-NOT: v_min_u32_e32
; GCN: s_endpgm

; EG: MIN_UINT
; EG: MIN_UINT
; EG: MIN_UINT
define void @v_test_umin_ule_v3i32(<3 x i32> addrspace(1)* %out, <3 x i32> addrspace(1)* %aptr, <3 x i32> addrspace(1)* %bptr) nounwind {
  %a = load <3 x i32>, <3 x i32> addrspace(1)* %aptr
  %b = load <3 x i32>, <3 x i32> addrspace(1)* %bptr
  %cmp = icmp ule <3 x i32> %a, %b
  %val = select <3 x i1> %cmp, <3 x i32> %a, <3 x i32> %b
  store <3 x i32> %val, <3 x i32> addrspace(1)* %out
  ret void
}
; FUNC-LABEL: @s_test_umin_ule_i32
; GCN: s_min_u32

; EG: MIN_UINT
define void @s_test_umin_ule_i32(i32 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %cmp = icmp ule i32 %a, %b
  %val = select i1 %cmp, i32 %a, i32 %b
  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @v_test_umin_ult_i32
; GCN: v_min_u32_e32

; EG: MIN_UINT
define void @v_test_umin_ult_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %aptr, i32 addrspace(1)* %bptr) nounwind {
  %a = load i32, i32 addrspace(1)* %aptr, align 4
  %b = load i32, i32 addrspace(1)* %bptr, align 4
  %cmp = icmp ult i32 %a, %b
  %val = select i1 %cmp, i32 %a, i32 %b
  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v_test_umin_ult_i8:
; GCN: buffer_load_ubyte
; GCN: buffer_load_ubyte
; GCN: v_min_u32_e32

; EG: MIN_UINT
define void @v_test_umin_ult_i8(i8 addrspace(1)* %out, i8 addrspace(1)* %aptr, i8 addrspace(1)* %bptr) nounwind {
  %a = load i8, i8 addrspace(1)* %aptr, align 1
  %b = load i8, i8 addrspace(1)* %bptr, align 1
  %cmp = icmp ult i8 %a, %b
  %val = select i1 %cmp, i8 %a, i8 %b
  store i8 %val, i8 addrspace(1)* %out, align 1
  ret void
}

; FUNC-LABEL: @s_test_umin_ult_i32
; GCN: s_min_u32

; EG: MIN_UINT
define void @s_test_umin_ult_i32(i32 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %cmp = icmp ult i32 %a, %b
  %val = select i1 %cmp, i32 %a, i32 %b
  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @v_test_umin_ult_i32_multi_use
; SI-NOT: v_min
; GCN: v_cmp_lt_u32
; SI-NEXT: v_cndmask_b32
; SI-NOT: v_min
; GCN: s_endpgm

; EG-NOT: MIN_UINT
define void @v_test_umin_ult_i32_multi_use(i32 addrspace(1)* %out0, i1 addrspace(1)* %out1, i32 addrspace(1)* %aptr, i32 addrspace(1)* %bptr) nounwind {
  %a = load i32, i32 addrspace(1)* %aptr, align 4
  %b = load i32, i32 addrspace(1)* %bptr, align 4
  %cmp = icmp ult i32 %a, %b
  %val = select i1 %cmp, i32 %a, i32 %b
  store i32 %val, i32 addrspace(1)* %out0, align 4
  store i1 %cmp, i1 addrspace(1)* %out1
  ret void
}

; FUNC-LABEL: @v_test_umin_ult_i16_multi_use
; GCN-NOT: v_min
; GCN: v_cmp_lt_u32
; GCN-NEXT: v_cndmask_b32
; GCN-NOT: v_min
; GCN: s_endpgm

; EG-NOT: MIN_UINT
define void @v_test_umin_ult_i16_multi_use(i16 addrspace(1)* %out0, i1 addrspace(1)* %out1, i16 addrspace(1)* %aptr, i16 addrspace(1)* %bptr) nounwind {
  %a = load i16, i16 addrspace(1)* %aptr, align 2
  %b = load i16, i16 addrspace(1)* %bptr, align 2
  %cmp = icmp ult i16 %a, %b
  %val = select i1 %cmp, i16 %a, i16 %b
  store i16 %val, i16 addrspace(1)* %out0, align 2
  store i1 %cmp, i1 addrspace(1)* %out1
  ret void
}


; FUNC-LABEL: @s_test_umin_ult_v1i32
; GCN: s_min_u32

; EG: MIN_UINT
define void @s_test_umin_ult_v1i32(<1 x i32> addrspace(1)* %out, <1 x i32> %a, <1 x i32> %b) nounwind {
  %cmp = icmp ult <1 x i32> %a, %b
  %val = select <1 x i1> %cmp, <1 x i32> %a, <1 x i32> %b
  store <1 x i32> %val, <1 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}s_test_umin_ult_v8i32:
; GCN: s_min_u32
; GCN: s_min_u32
; GCN: s_min_u32
; GCN: s_min_u32
; GCN: s_min_u32
; GCN: s_min_u32
; GCN: s_min_u32
; GCN: s_min_u32

; EG: MIN_UINT
; EG: MIN_UINT
; EG: MIN_UINT
; EG: MIN_UINT
; EG: MIN_UINT
; EG: MIN_UINT
; EG: MIN_UINT
; EG: MIN_UINT
define void @s_test_umin_ult_v8i32(<8 x i32> addrspace(1)* %out, <8 x i32> %a, <8 x i32> %b) nounwind {
  %cmp = icmp ult <8 x i32> %a, %b
  %val = select <8 x i1> %cmp, <8 x i32> %a, <8 x i32> %b
  store <8 x i32> %val, <8 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}s_test_umin_ult_v8i16:
; GCN: v_min_u32
; GCN: v_min_u32
; GCN: v_min_u32
; GCN: v_min_u32
; GCN: v_min_u32
; GCN: v_min_u32
; GCN: v_min_u32
; GCN: v_min_u32

; EG: MIN_UINT
; EG: MIN_UINT
; EG: MIN_UINT
; EG: MIN_UINT
; EG: MIN_UINT
; EG: MIN_UINT
; EG: MIN_UINT
; EG: MIN_UINT
define void @s_test_umin_ult_v8i16(<8 x i16> addrspace(1)* %out, <8 x i16> %a, <8 x i16> %b) nounwind {
  %cmp = icmp ult <8 x i16> %a, %b
  %val = select <8 x i1> %cmp, <8 x i16> %a, <8 x i16> %b
  store <8 x i16> %val, <8 x i16> addrspace(1)* %out
  ret void
}

; Make sure redundant and removed
; FUNC-LABEL: {{^}}simplify_demanded_bits_test_umin_ult_i16:
; GCN-DAG: s_load_dword [[A:s[0-9]+]], {{s\[[0-9]+:[0-9]+\]}}, {{0xb|0x2c}}
; GCN-DAG: s_load_dword [[B:s[0-9]+]], {{s\[[0-9]+:[0-9]+\]}}, {{0xc|0x30}}
; GCN: s_min_u32 [[MIN:s[0-9]+]], [[A]], [[B]]
; GCN: v_mov_b32_e32 [[VMIN:v[0-9]+]], [[MIN]]
; GCN: buffer_store_dword [[VMIN]]

; EG: MIN_UINT
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
; GCN-DAG: s_load_dword [[A:s[0-9]+]], {{s\[[0-9]+:[0-9]+\]}}, {{0xb|0x2c}}
; GCN-DAG: s_load_dword [[B:s[0-9]+]], {{s\[[0-9]+:[0-9]+\]}}, {{0xc|0x30}}
; GCN: s_min_i32 [[MIN:s[0-9]+]], [[A]], [[B]]
; GCN: v_mov_b32_e32 [[VMIN:v[0-9]+]], [[MIN]]
; GCN: buffer_store_dword [[VMIN]]

; EG: MIN_INT
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
; GCN: s_min_i32

; EG: MIN_INT
define void @s_test_imin_sle_i16(i16 addrspace(1)* %out, i16 %a, i16 %b) nounwind {
  %cmp = icmp sle i16 %a, %b
  %val = select i1 %cmp, i16 %a, i16 %b
  store i16 %val, i16 addrspace(1)* %out
  ret void
}

; 64 bit
; FUNC-LABEL: {{^}}test_umin_ult_i64
; GCN: s_endpgm

; EG: MIN_UINT
; EG: MIN_UINT
define void @test_umin_ult_i64(i64 addrspace(1)* %out, i64 %a, i64 %b) nounwind {
  %tmp = icmp ult i64 %a, %b
  %val = select i1 %tmp, i64 %a, i64 %b
  store i64 %val, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}test_umin_ule_i64
; GCN: s_endpgm

; EG: MIN_UINT
; EG: MIN_UINT
define void @test_umin_ule_i64(i64 addrspace(1)* %out, i64 %a, i64 %b) nounwind {
  %tmp = icmp ule i64 %a, %b
  %val = select i1 %tmp, i64 %a, i64 %b
  store i64 %val, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}test_imin_slt_i64
; GCN: s_endpgm

; EG-DAG: MIN_UINT
; EG-DAG: MIN_INT
define void @test_imin_slt_i64(i64 addrspace(1)* %out, i64 %a, i64 %b) nounwind {
  %tmp = icmp slt i64 %a, %b
  %val = select i1 %tmp, i64 %a, i64 %b
  store i64 %val, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}test_imin_sle_i64
; GCN: s_endpgm

; EG-DAG: MIN_UINT
; EG-DAG: MIN_INT
define void @test_imin_sle_i64(i64 addrspace(1)* %out, i64 %a, i64 %b) nounwind {
  %tmp = icmp sle i64 %a, %b
  %val = select i1 %tmp, i64 %a, i64 %b
  store i64 %val, i64 addrspace(1)* %out, align 8
  ret void
}
