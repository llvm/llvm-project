; RUN: llc -march=amdgcn -mcpu=pitcairn < %s | FileCheck -enable-var-scope -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=cypress < %s | FileCheck -enable-var-scope -check-prefix=EG -check-prefix=FUNC %s


; FUNC-LABEL: {{^}}v_test_imax_sge_i32:
; SI: v_max_i32_e32

; EG: MAX_INT
define amdgpu_kernel void @v_test_imax_sge_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %aptr, i32 addrspace(1)* %bptr) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.in = getelementptr inbounds i32, i32 addrspace(1)* %bptr, i32 %tid
  %a = load i32, i32 addrspace(1)* %aptr, align 4
  %b = load i32, i32 addrspace(1)* %gep.in, align 4
  %cmp = icmp sge i32 %a, %b
  %val = select i1 %cmp, i32 %a, i32 %b
  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v_test_imax_sge_v4i32:
; SI: v_max_i32_e32
; SI: v_max_i32_e32
; SI: v_max_i32_e32
; SI: v_max_i32_e32

; These could be merged into one
; EG: MAX_INT
; EG: MAX_INT
; EG: MAX_INT
; EG: MAX_INT
define amdgpu_kernel void @v_test_imax_sge_v4i32(<4 x i32> addrspace(1)* %out, <4 x i32> addrspace(1)* %aptr, <4 x i32> addrspace(1)* %bptr) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.in = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(1)* %bptr, i32 %tid
  %a = load <4 x i32>, <4 x i32> addrspace(1)* %aptr, align 4
  %b = load <4 x i32>, <4 x i32> addrspace(1)* %gep.in, align 4
  %cmp = icmp sge <4 x i32> %a, %b
  %val = select <4 x i1> %cmp, <4 x i32> %a, <4 x i32> %b
  store <4 x i32> %val, <4 x i32> addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @s_test_imax_sge_i32
; SI: s_max_i32

; EG: MAX_INT
define amdgpu_kernel void @s_test_imax_sge_i32(i32 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %cmp = icmp sge i32 %a, %b
  %val = select i1 %cmp, i32 %a, i32 %b
  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}s_test_imax_sge_imm_i32:
; SI: s_max_i32 {{s[0-9]+}}, {{s[0-9]+}}, 9

; EG: MAX_INT {{.*}}literal.{{[xyzw]}}
define amdgpu_kernel void @s_test_imax_sge_imm_i32(i32 addrspace(1)* %out, i32 %a) nounwind {
  %cmp = icmp sge i32 %a, 9
  %val = select i1 %cmp, i32 %a, i32 9
  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v_test_imax_sge_i8:
; SI: buffer_load_sbyte
; SI: buffer_load_sbyte
; SI: v_max_i32_e32

; EG: MAX_INT
define amdgpu_kernel void @v_test_imax_sge_i8(i8 addrspace(1)* %out, i8 addrspace(1)* %aptr, i8 addrspace(1)* %bptr) nounwind {
  %a = load i8, i8 addrspace(1)* %aptr, align 1
  %b = load i8, i8 addrspace(1)* %bptr, align 1
  %cmp = icmp sge i8 %a, %b
  %val = select i1 %cmp, i8 %a, i8 %b
  store i8 %val, i8 addrspace(1)* %out, align 1
  ret void
}

; FUNC-LABEL: {{^}}s_test_imax_sgt_imm_i32:
; SI: s_max_i32 {{s[0-9]+}}, {{s[0-9]+}}, 9

; EG: MAX_INT {{.*}}literal.{{[xyzw]}}
define amdgpu_kernel void @s_test_imax_sgt_imm_i32(i32 addrspace(1)* %out, i32 %a) nounwind {
  %cmp = icmp sgt i32 %a, 9
  %val = select i1 %cmp, i32 %a, i32 9
  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}s_test_imax_sgt_imm_v2i32:
; SI: s_max_i32 {{s[0-9]+}}, {{s[0-9]+}}, 9
; SI: s_max_i32 {{s[0-9]+}}, {{s[0-9]+}}, 9

; EG: MAX_INT {{.*}}literal.{{[xyzw]}}
; EG: MAX_INT {{.*}}literal.{{[xyzw]}}
define amdgpu_kernel void @s_test_imax_sgt_imm_v2i32(<2 x i32> addrspace(1)* %out, <2 x i32> %a) nounwind {
  %cmp = icmp sgt <2 x i32> %a, <i32 9, i32 9>
  %val = select <2 x i1> %cmp, <2 x i32> %a, <2 x i32> <i32 9, i32 9>
  store <2 x i32> %val, <2 x i32> addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @v_test_imax_sgt_i32
; SI: v_max_i32_e32

; EG: MAX_INT
define amdgpu_kernel void @v_test_imax_sgt_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %aptr, i32 addrspace(1)* %bptr) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.in = getelementptr inbounds i32, i32 addrspace(1)* %bptr, i32 %tid
  %a = load i32, i32 addrspace(1)* %aptr, align 4
  %b = load i32, i32 addrspace(1)* %gep.in, align 4
  %cmp = icmp sgt i32 %a, %b
  %val = select i1 %cmp, i32 %a, i32 %b
  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @s_test_imax_sgt_i32
; SI: s_max_i32

; EG: MAX_INT
define amdgpu_kernel void @s_test_imax_sgt_i32(i32 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %cmp = icmp sgt i32 %a, %b
  %val = select i1 %cmp, i32 %a, i32 %b
  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @v_test_umax_uge_i32
; SI: v_max_u32_e32

; EG: MAX_UINT
define amdgpu_kernel void @v_test_umax_uge_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %aptr, i32 addrspace(1)* %bptr) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.in = getelementptr inbounds i32, i32 addrspace(1)* %bptr, i32 %tid
  %a = load i32, i32 addrspace(1)* %aptr, align 4
  %b = load i32, i32 addrspace(1)* %gep.in, align 4
  %cmp = icmp uge i32 %a, %b
  %val = select i1 %cmp, i32 %a, i32 %b
  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @s_test_umax_uge_i32
; SI: s_max_u32

; EG: MAX_UINT
define amdgpu_kernel void @s_test_umax_uge_i32(i32 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
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

; EG: MAX_UINT
; EG: MAX_UINT
; EG: MAX_UINT
; EG-NOT: MAX_UINT
define amdgpu_kernel void @s_test_umax_uge_v3i32(<3 x i32> addrspace(1)* %out, <3 x i32> %a, <3 x i32> %b) nounwind {
  %cmp = icmp uge <3 x i32> %a, %b
  %val = select <3 x i1> %cmp, <3 x i32> %a, <3 x i32> %b
  store <3 x i32> %val, <3 x i32> addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v_test_umax_uge_i8:
; SI: buffer_load_ubyte
; SI: buffer_load_ubyte
; SI: v_max_u32_e32

; EG: MAX_UINT
define amdgpu_kernel void @v_test_umax_uge_i8(i8 addrspace(1)* %out, i8 addrspace(1)* %aptr, i8 addrspace(1)* %bptr) nounwind {
  %a = load i8, i8 addrspace(1)* %aptr, align 1
  %b = load i8, i8 addrspace(1)* %bptr, align 1
  %cmp = icmp uge i8 %a, %b
  %val = select i1 %cmp, i8 %a, i8 %b
  store i8 %val, i8 addrspace(1)* %out, align 1
  ret void
}

; FUNC-LABEL: @v_test_umax_ugt_i32
; SI: v_max_u32_e32

; EG: MAX_UINT
define amdgpu_kernel void @v_test_umax_ugt_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %aptr, i32 addrspace(1)* %bptr) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.in = getelementptr inbounds i32, i32 addrspace(1)* %bptr, i32 %tid
  %a = load i32, i32 addrspace(1)* %gep.in, align 4
  %b = load i32, i32 addrspace(1)* %bptr, align 4
  %cmp = icmp ugt i32 %a, %b
  %val = select i1 %cmp, i32 %a, i32 %b
  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}s_test_umax_ugt_i32:
; SI: s_max_u32

; EG: MAX_UINT
define amdgpu_kernel void @s_test_umax_ugt_i32(i32 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %cmp = icmp ugt i32 %a, %b
  %val = select i1 %cmp, i32 %a, i32 %b
  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}s_test_umax_ugt_imm_v2i32:
; SI-DAG: s_max_u32 {{s[0-9]+}}, {{s[0-9]+}}, 15
; SI-DAG: s_max_u32 {{s[0-9]+}}, {{s[0-9]+}}, 23

; EG: MAX_UINT {{.*}}literal.{{[xyzw]}}
; EG: MAX_UINT {{.*}}literal.{{[xyzw]}}
define amdgpu_kernel void @s_test_umax_ugt_imm_v2i32(<2 x i32> addrspace(1)* %out, <2 x i32> %a) nounwind {
  %cmp = icmp ugt <2 x i32> %a, <i32 15, i32 23>
  %val = select <2 x i1> %cmp, <2 x i32> %a, <2 x i32> <i32 15, i32 23>
  store <2 x i32> %val, <2 x i32> addrspace(1)* %out, align 4
  ret void
}

; Make sure redundant and removed
; FUNC-LABEL: {{^}}simplify_demanded_bits_test_umax_ugt_i16:
; SI-DAG: s_load_dword [[A:s[0-9]+]], {{s\[[0-9]+:[0-9]+\]}}, 0x13
; SI-DAG: s_load_dword [[B:s[0-9]+]], {{s\[[0-9]+:[0-9]+\]}}, 0x1c
; SI-DAG: s_and_b32 [[A16:s[0-9]+]], [[A]], 0xffff
; SI-DAG: s_and_b32 [[B16:s[0-9]+]], [[B]], 0xffff
; SI: s_max_u32 [[MAX:s[0-9]+]], [[A16]], [[B16]]
; SI: v_mov_b32_e32 [[VMAX:v[0-9]+]], [[MAX]]
; SI: buffer_store_dword [[VMAX]]

; EG: MAX_UINT
define amdgpu_kernel void @simplify_demanded_bits_test_umax_ugt_i16(i32 addrspace(1)* %out, [8 x i32], i16 zeroext %a, [8 x i32], i16 zeroext %b) nounwind {
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
; SI-DAG: s_load_dword [[A:s[0-9]+]], {{s\[[0-9]+:[0-9]+\]}}, 0x13
; SI-DAG: s_load_dword [[B:s[0-9]+]], {{s\[[0-9]+:[0-9]+\]}}, 0x1c
; SI-DAG: s_sext_i32_i16 [[EXT_A:s[0-9]+]], [[A]]
; SI-DAG: s_sext_i32_i16 [[EXT_B:s[0-9]+]], [[B]]

; SI: s_max_i32 [[MAX:s[0-9]+]], [[EXT_A]], [[EXT_B]]
; SI: v_mov_b32_e32 [[VMAX:v[0-9]+]], [[MAX]]
; SI: buffer_store_dword [[VMAX]]

; EG: MAX_INT
define amdgpu_kernel void @simplify_demanded_bits_test_max_slt_i16(i32 addrspace(1)* %out, [8 x i32], i16 signext %a, [8 x i32], i16 signext %b) nounwind {
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
; SI: s_load_dword
; SI: s_load_dword
; SI: s_sext_i32_i16
; SI: s_sext_i32_i16
; SI: s_max_i32

; EG: MAX_INT
define amdgpu_kernel void @s_test_imax_sge_i16(i16 addrspace(1)* %out, [8 x i32], i16 %a, [8 x i32], i16 %b) nounwind {
  %cmp = icmp sge i16 %a, %b
  %val = select i1 %cmp, i16 %a, i16 %b
  store i16 %val, i16 addrspace(1)* %out
  ret void
}

; 64 bit
; FUNC-LABEL: {{^}}test_umax_ugt_i64
; SI: s_endpgm

; EG: MAX_UINT
; EG: MAX_UINT
define amdgpu_kernel void @test_umax_ugt_i64(i64 addrspace(1)* %out, i64 %a, i64 %b) nounwind {
  %tmp = icmp ugt i64 %a, %b
  %val = select i1 %tmp, i64 %a, i64 %b
  store i64 %val, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}test_umax_uge_i64
; SI: s_endpgm

; EG: MAX_UINT
; EG: MAX_UINT
define amdgpu_kernel void @test_umax_uge_i64(i64 addrspace(1)* %out, i64 %a, i64 %b) nounwind {
  %tmp = icmp uge i64 %a, %b
  %val = select i1 %tmp, i64 %a, i64 %b
  store i64 %val, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}test_imax_sgt_i64
; SI: s_endpgm

; EG-DAG: MAX_UINT
; EG-DAG: MAX_INT
define amdgpu_kernel void @test_imax_sgt_i64(i64 addrspace(1)* %out, i64 %a, i64 %b) nounwind {
  %tmp = icmp sgt i64 %a, %b
  %val = select i1 %tmp, i64 %a, i64 %b
  store i64 %val, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}test_imax_sge_i64
; SI: s_endpgm

; EG-DAG: MAX_UINT
; EG-DAG: MAX_INT
define amdgpu_kernel void @test_imax_sge_i64(i64 addrspace(1)* %out, i64 %a, i64 %b) nounwind {
  %tmp = icmp sge i64 %a, %b
  %val = select i1 %tmp, i64 %a, i64 %b
  store i64 %val, i64 addrspace(1)* %out, align 8
  ret void
}


declare i32 @llvm.amdgcn.workitem.id.x() #0

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
