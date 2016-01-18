; RUN: llc -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=bonaire -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=CI %s

declare i32 @llvm.r600.read.tidig.x() #0
declare i32 @llvm.r600.read.tidig.y() #0

; In this test both the pointer and the offset operands to the
; BUFFER_LOAD instructions end up being stored in vgprs.  This
; requires us to add the pointer and offset together, store the
; result in the offset operand (vaddr), and then store 0 in an
; sgpr register pair and use that for the pointer operand
; (low 64-bits of srsrc).

; GCN-LABEL: {{^}}mubuf:

; Make sure we aren't using VGPRs for the source operand of s_mov_b64
; GCN-NOT: s_mov_b64 s[{{[0-9]+:[0-9]+}}], v

; Make sure we aren't using VGPR's for the srsrc operand of BUFFER_LOAD_*
; instructions
; GCN: buffer_load_ubyte v{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], 0 addr64
; GCN: buffer_load_ubyte v{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], 0 addr64

define void @mubuf(i32 addrspace(1)* %out, i8 addrspace(1)* %in) #1 {
entry:
  %tmp = call i32 @llvm.r600.read.tidig.x()
  %tmp1 = call i32 @llvm.r600.read.tidig.y()
  %tmp2 = sext i32 %tmp to i64
  %tmp3 = sext i32 %tmp1 to i64
  br label %loop

loop:                                             ; preds = %loop, %entry
  %tmp4 = phi i64 [ 0, %entry ], [ %tmp5, %loop ]
  %tmp5 = add i64 %tmp2, %tmp4
  %tmp6 = getelementptr i8, i8 addrspace(1)* %in, i64 %tmp5
  %tmp7 = load i8, i8 addrspace(1)* %tmp6, align 1
  %tmp8 = or i64 %tmp5, 1
  %tmp9 = getelementptr i8, i8 addrspace(1)* %in, i64 %tmp8
  %tmp10 = load i8, i8 addrspace(1)* %tmp9, align 1
  %tmp11 = add i8 %tmp7, %tmp10
  %tmp12 = sext i8 %tmp11 to i32
  store i32 %tmp12, i32 addrspace(1)* %out
  %tmp13 = icmp slt i64 %tmp5, 10
  br i1 %tmp13, label %loop, label %done

done:                                             ; preds = %loop
  ret void
}

; Test moving an SMRD instruction to the VALU

; GCN-LABEL: {{^}}smrd_valu:
; GCN: buffer_load_dword [[OUT:v[0-9]+]]
; GCN: buffer_store_dword [[OUT]]
define void @smrd_valu(i32 addrspace(2)* addrspace(1)* %in, i32 %a, i32 %b, i32 addrspace(1)* %out) #1 {
entry:
  %tmp = icmp ne i32 %a, 0
  br i1 %tmp, label %if, label %else

if:                                               ; preds = %entry
  %tmp1 = load i32 addrspace(2)*, i32 addrspace(2)* addrspace(1)* %in
  br label %endif

else:                                             ; preds = %entry
  %tmp2 = getelementptr i32 addrspace(2)*, i32 addrspace(2)* addrspace(1)* %in
  %tmp3 = load i32 addrspace(2)*, i32 addrspace(2)* addrspace(1)* %tmp2
  br label %endif

endif:                                            ; preds = %else, %if
  %tmp4 = phi i32 addrspace(2)* [ %tmp1, %if ], [ %tmp3, %else ]
  %tmp5 = getelementptr i32, i32 addrspace(2)* %tmp4, i32 3000
  %tmp6 = load i32, i32 addrspace(2)* %tmp5
  store i32 %tmp6, i32 addrspace(1)* %out
  ret void
}

; Test moving an SMRD with an immediate offset to the VALU

; GCN-LABEL: {{^}}smrd_valu2:
; GCN: buffer_load_dword v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:16{{$}}
define void @smrd_valu2(i32 addrspace(1)* %out, [8 x i32] addrspace(2)* %in) #1 {
entry:
  %tmp = call i32 @llvm.r600.read.tidig.x() #0
  %tmp1 = add i32 %tmp, 4
  %tmp2 = getelementptr [8 x i32], [8 x i32] addrspace(2)* %in, i32 %tmp, i32 4
  %tmp3 = load i32, i32 addrspace(2)* %tmp2
  store i32 %tmp3, i32 addrspace(1)* %out
  ret void
}

; Use a big offset that will use the SMRD literal offset on CI
; GCN-LABEL: {{^}}smrd_valu_ci_offset:
; GCN: s_movk_i32 s[[OFFSET:[0-9]+]], 0x4e20{{$}}
; GCN: buffer_load_dword v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[}}[[OFFSET]]:{{[0-9]+}}], 0 addr64{{$}}
; GCN: v_add_i32_e32
; GCN: buffer_store_dword
define void @smrd_valu_ci_offset(i32 addrspace(1)* %out, i32 addrspace(2)* %in, i32 %c) #1 {
entry:
  %tmp = call i32 @llvm.r600.read.tidig.x() #0
  %tmp2 = getelementptr i32, i32 addrspace(2)* %in, i32 %tmp
  %tmp3 = getelementptr i32, i32 addrspace(2)* %tmp2, i32 5000
  %tmp4 = load i32, i32 addrspace(2)* %tmp3
  %tmp5 = add i32 %tmp4, %c
  store i32 %tmp5, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}smrd_valu_ci_offset_x2:
; GCN: s_mov_b32 s[[OFFSET:[0-9]+]], 0x9c40{{$}}
; GCN: buffer_load_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[}}[[OFFSET]]:{{[0-9]+}}], 0 addr64{{$}}
; GCN: v_or_b32_e32 {{v[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}
; GCN: v_or_b32_e32 {{v[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}
; GCN: buffer_store_dwordx2
define void @smrd_valu_ci_offset_x2(i64 addrspace(1)* %out, i64 addrspace(2)* %in, i64 %c) #1 {
entry:
  %tmp = call i32 @llvm.r600.read.tidig.x() #0
  %tmp2 = getelementptr i64, i64 addrspace(2)* %in, i32 %tmp
  %tmp3 = getelementptr i64, i64 addrspace(2)* %tmp2, i32 5000
  %tmp4 = load i64, i64 addrspace(2)* %tmp3
  %tmp5 = or i64 %tmp4, %c
  store i64 %tmp5, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}smrd_valu_ci_offset_x4:
; GCN: s_movk_i32 s[[OFFSET:[0-9]+]], 0x4d20{{$}}
; GCN: buffer_load_dwordx4 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[}}[[OFFSET]]:{{[0-9]+}}], 0 addr64{{$}}
; GCN: v_or_b32_e32 {{v[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}
; GCN: v_or_b32_e32 {{v[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}
; GCN: v_or_b32_e32 {{v[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}
; GCN: v_or_b32_e32 {{v[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}
; GCN: buffer_store_dwordx4
define void @smrd_valu_ci_offset_x4(<4 x i32> addrspace(1)* %out, <4 x i32> addrspace(2)* %in, <4 x i32> %c) #1 {
entry:
  %tmp = call i32 @llvm.r600.read.tidig.x() #0
  %tmp2 = getelementptr <4 x i32>, <4 x i32> addrspace(2)* %in, i32 %tmp
  %tmp3 = getelementptr <4 x i32>, <4 x i32> addrspace(2)* %tmp2, i32 1234
  %tmp4 = load <4 x i32>, <4 x i32> addrspace(2)* %tmp3
  %tmp5 = or <4 x i32> %tmp4, %c
  store <4 x i32> %tmp5, <4 x i32> addrspace(1)* %out
  ret void
}

; Original scalar load uses SGPR offset on SI and 32-bit literal on
; CI.

; GCN-LABEL: {{^}}smrd_valu_ci_offset_x8:
; GCN: s_mov_b32 s[[OFFSET0:[0-9]+]], 0x9a40{{$}}
; GCN: buffer_load_dwordx4 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[}}[[OFFSET0]]:{{[0-9]+}}], 0 addr64{{$}}

; SI: s_add_i32 s[[OFFSET1:[0-9]+]], s[[OFFSET0]], 16
; SI: buffer_load_dwordx4 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[}}[[OFFSET1]]:{{[0-9]+}}], 0 addr64{{$}}

; CI: s_mov_b32 s[[OFFSET1:[0-9]+]], 0x9a50{{$}}
; CI: buffer_load_dwordx4 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[}}[[OFFSET1]]:{{[0-9]+}}], 0 addr64{{$}}

; GCN: v_or_b32_e32 {{v[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}
; GCN: v_or_b32_e32 {{v[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}
; GCN: v_or_b32_e32 {{v[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}
; GCN: v_or_b32_e32 {{v[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}
; GCN: v_or_b32_e32 {{v[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}
; GCN: v_or_b32_e32 {{v[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}
; GCN: v_or_b32_e32 {{v[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}
; GCN: v_or_b32_e32 {{v[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}
; GCN: buffer_store_dword
; GCN: buffer_store_dword
; GCN: buffer_store_dword
; GCN: buffer_store_dword
; GCN: buffer_store_dword
; GCN: buffer_store_dword
; GCN: buffer_store_dword
; GCN: buffer_store_dword
define void @smrd_valu_ci_offset_x8(<8 x i32> addrspace(1)* %out, <8 x i32> addrspace(2)* %in, <8 x i32> %c) #1 {
entry:
  %tmp = call i32 @llvm.r600.read.tidig.x() #0
  %tmp2 = getelementptr <8 x i32>, <8 x i32> addrspace(2)* %in, i32 %tmp
  %tmp3 = getelementptr <8 x i32>, <8 x i32> addrspace(2)* %tmp2, i32 1234
  %tmp4 = load <8 x i32>, <8 x i32> addrspace(2)* %tmp3
  %tmp5 = or <8 x i32> %tmp4, %c
  store <8 x i32> %tmp5, <8 x i32> addrspace(1)* %out
  ret void
}

; FIXME: should use immediate offset instead of using s_add_i32 for adding to constant.
; GCN-LABEL: {{^}}smrd_valu_ci_offset_x16:

; GCN: s_mov_b32 s[[OFFSET0:[0-9]+]], 0x13480{{$}}
; SI: s_add_i32 s[[OFFSET1:[0-9]+]], s[[OFFSET0]], 16
; GCN: buffer_load_dwordx4 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[}}[[OFFSET0]]:{{[0-9]+}}], 0 addr64{{$}}

; CI: s_mov_b32 s[[OFFSET1:[0-9]+]], 0x13490{{$}}
; GCN: buffer_load_dwordx4 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[}}[[OFFSET1]]:{{[0-9]+}}], 0 addr64{{$}}

; SI: s_add_i32 s[[OFFSET2:[0-9]+]], s[[OFFSET0]], 32
; CI: s_mov_b32 s[[OFFSET2:[0-9]+]], 0x134a0

; GCN: buffer_load_dwordx4 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[}}[[OFFSET2]]:{{[0-9]+}}], 0 addr64{{$}}
; GCN: s_add_i32 s[[OFFSET3:[0-9]+]], s[[OFFSET2]], 16
; GCN: buffer_load_dwordx4 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[}}[[OFFSET3]]:{{[0-9]+}}], 0 addr64{{$}}
; GCN: v_or_b32_e32 {{v[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}
; GCN: v_or_b32_e32 {{v[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}
; GCN: v_or_b32_e32 {{v[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}
; GCN: v_or_b32_e32 {{v[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}
; GCN: v_or_b32_e32 {{v[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}
; GCN: v_or_b32_e32 {{v[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}
; GCN: v_or_b32_e32 {{v[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}
; GCN: v_or_b32_e32 {{v[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}
; GCN: buffer_store_dword
; GCN: buffer_store_dword
; GCN: buffer_store_dword
; GCN: buffer_store_dword
; GCN: buffer_store_dword
; GCN: buffer_store_dword
; GCN: buffer_store_dword
; GCN: buffer_store_dword
define void @smrd_valu_ci_offset_x16(<16 x i32> addrspace(1)* %out, <16 x i32> addrspace(2)* %in, <16 x i32> %c) #1 {
entry:
  %tmp = call i32 @llvm.r600.read.tidig.x() #0
  %tmp2 = getelementptr <16 x i32>, <16 x i32> addrspace(2)* %in, i32 %tmp
  %tmp3 = getelementptr <16 x i32>, <16 x i32> addrspace(2)* %tmp2, i32 1234
  %tmp4 = load <16 x i32>, <16 x i32> addrspace(2)* %tmp3
  %tmp5 = or <16 x i32> %tmp4, %c
  store <16 x i32> %tmp5, <16 x i32> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}smrd_valu2_salu_user:
; GCN: buffer_load_dword [[MOVED:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:16{{$}}
; GCN: v_add_i32_e32 [[ADD:v[0-9]+]], vcc, s{{[0-9]+}}, [[MOVED]]
; GCN: buffer_store_dword [[ADD]]
define void @smrd_valu2_salu_user(i32 addrspace(1)* %out, [8 x i32] addrspace(2)* %in, i32 %a) #1 {
entry:
  %tmp = call i32 @llvm.r600.read.tidig.x() #0
  %tmp1 = add i32 %tmp, 4
  %tmp2 = getelementptr [8 x i32], [8 x i32] addrspace(2)* %in, i32 %tmp, i32 4
  %tmp3 = load i32, i32 addrspace(2)* %tmp2
  %tmp4 = add i32 %tmp3, %a
  store i32 %tmp4, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}smrd_valu2_max_smrd_offset:
; GCN: buffer_load_dword v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:1020{{$}}
define void @smrd_valu2_max_smrd_offset(i32 addrspace(1)* %out, [1024 x i32] addrspace(2)* %in) #1 {
entry:
  %tmp = call i32 @llvm.r600.read.tidig.x() #0
  %tmp1 = add i32 %tmp, 4
  %tmp2 = getelementptr [1024 x i32], [1024 x i32] addrspace(2)* %in, i32 %tmp, i32 255
  %tmp3 = load i32, i32 addrspace(2)* %tmp2
  store i32 %tmp3, i32 addrspace(1)* %out
  ret void
}

; Offset is too big to fit in SMRD 8-bit offset, but small enough to
; fit in MUBUF offset.
; FIXME: We should be using the offset but we don't

; GCN-LABEL: {{^}}smrd_valu2_mubuf_offset:
; SI: s_movk_i32 s[[OFFSET:[0-9]+]], 0x400{{$}}
; SI: buffer_load_dword v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[}}[[OFFSET]]:{{[0-9]+\]}}, 0 addr64{{$}}

; CI: buffer_load_dword v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:1024{{$}}
define void @smrd_valu2_mubuf_offset(i32 addrspace(1)* %out, [1024 x i32] addrspace(2)* %in) #1 {
entry:
  %tmp = call i32 @llvm.r600.read.tidig.x() #0
  %tmp1 = add i32 %tmp, 4
  %tmp2 = getelementptr [1024 x i32], [1024 x i32] addrspace(2)* %in, i32 %tmp, i32 256
  %tmp3 = load i32, i32 addrspace(2)* %tmp2
  store i32 %tmp3, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}s_load_imm_v8i32:
; GCN: buffer_load_dwordx4
; GCN: buffer_load_dwordx4
define void @s_load_imm_v8i32(<8 x i32> addrspace(1)* %out, i32 addrspace(2)* nocapture readonly %in) #1 {
entry:
  %tmp0 = tail call i32 @llvm.r600.read.tidig.x()
  %tmp1 = getelementptr inbounds i32, i32 addrspace(2)* %in, i32 %tmp0
  %tmp2 = bitcast i32 addrspace(2)* %tmp1 to <8 x i32> addrspace(2)*
  %tmp3 = load <8 x i32>, <8 x i32> addrspace(2)* %tmp2, align 4
  store <8 x i32> %tmp3, <8 x i32> addrspace(1)* %out, align 32
  ret void
}

; GCN-LABEL: {{^}}s_load_imm_v8i32_salu_user:
; GCN: buffer_load_dwordx4
; GCN: buffer_load_dwordx4
; GCN: v_add_i32_e32
; GCN: v_add_i32_e32
; GCN: v_add_i32_e32
; GCN: v_add_i32_e32
; GCN: v_add_i32_e32
; GCN: v_add_i32_e32
; GCN: v_add_i32_e32
; GCN: buffer_store_dword
define void @s_load_imm_v8i32_salu_user(i32 addrspace(1)* %out, i32 addrspace(2)* nocapture readonly %in) #1 {
entry:
  %tmp0 = tail call i32 @llvm.r600.read.tidig.x()
  %tmp1 = getelementptr inbounds i32, i32 addrspace(2)* %in, i32 %tmp0
  %tmp2 = bitcast i32 addrspace(2)* %tmp1 to <8 x i32> addrspace(2)*
  %tmp3 = load <8 x i32>, <8 x i32> addrspace(2)* %tmp2, align 4

  %elt0 = extractelement <8 x i32> %tmp3, i32 0
  %elt1 = extractelement <8 x i32> %tmp3, i32 1
  %elt2 = extractelement <8 x i32> %tmp3, i32 2
  %elt3 = extractelement <8 x i32> %tmp3, i32 3
  %elt4 = extractelement <8 x i32> %tmp3, i32 4
  %elt5 = extractelement <8 x i32> %tmp3, i32 5
  %elt6 = extractelement <8 x i32> %tmp3, i32 6
  %elt7 = extractelement <8 x i32> %tmp3, i32 7

  %add0 = add i32 %elt0, %elt1
  %add1 = add i32 %add0, %elt2
  %add2 = add i32 %add1, %elt3
  %add3 = add i32 %add2, %elt4
  %add4 = add i32 %add3, %elt5
  %add5 = add i32 %add4, %elt6
  %add6 = add i32 %add5, %elt7

  store i32 %add6, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}s_load_imm_v16i32:
; GCN: buffer_load_dwordx4
; GCN: buffer_load_dwordx4
; GCN: buffer_load_dwordx4
; GCN: buffer_load_dwordx4
define void @s_load_imm_v16i32(<16 x i32> addrspace(1)* %out, i32 addrspace(2)* nocapture readonly %in) #1 {
entry:
  %tmp0 = tail call i32 @llvm.r600.read.tidig.x() #1
  %tmp1 = getelementptr inbounds i32, i32 addrspace(2)* %in, i32 %tmp0
  %tmp2 = bitcast i32 addrspace(2)* %tmp1 to <16 x i32> addrspace(2)*
  %tmp3 = load <16 x i32>, <16 x i32> addrspace(2)* %tmp2, align 4
  store <16 x i32> %tmp3, <16 x i32> addrspace(1)* %out, align 32
  ret void
}

; GCN-LABEL: {{^}}s_load_imm_v16i32_salu_user:
; GCN: buffer_load_dwordx4
; GCN: buffer_load_dwordx4
; GCN: buffer_load_dwordx4
; GCN: buffer_load_dwordx4
; GCN: v_add_i32_e32
; GCN: v_add_i32_e32
; GCN: v_add_i32_e32
; GCN: v_add_i32_e32
; GCN: v_add_i32_e32
; GCN: v_add_i32_e32
; GCN: v_add_i32_e32
; GCN: v_add_i32_e32
; GCN: v_add_i32_e32
; GCN: v_add_i32_e32
; GCN: v_add_i32_e32
; GCN: v_add_i32_e32
; GCN: v_add_i32_e32
; GCN: v_add_i32_e32
; GCN: v_add_i32_e32
; GCN: buffer_store_dword
define void @s_load_imm_v16i32_salu_user(i32 addrspace(1)* %out, i32 addrspace(2)* nocapture readonly %in) #1 {
entry:
  %tmp0 = tail call i32 @llvm.r600.read.tidig.x() #1
  %tmp1 = getelementptr inbounds i32, i32 addrspace(2)* %in, i32 %tmp0
  %tmp2 = bitcast i32 addrspace(2)* %tmp1 to <16 x i32> addrspace(2)*
  %tmp3 = load <16 x i32>, <16 x i32> addrspace(2)* %tmp2, align 4

  %elt0 = extractelement <16 x i32> %tmp3, i32 0
  %elt1 = extractelement <16 x i32> %tmp3, i32 1
  %elt2 = extractelement <16 x i32> %tmp3, i32 2
  %elt3 = extractelement <16 x i32> %tmp3, i32 3
  %elt4 = extractelement <16 x i32> %tmp3, i32 4
  %elt5 = extractelement <16 x i32> %tmp3, i32 5
  %elt6 = extractelement <16 x i32> %tmp3, i32 6
  %elt7 = extractelement <16 x i32> %tmp3, i32 7
  %elt8 = extractelement <16 x i32> %tmp3, i32 8
  %elt9 = extractelement <16 x i32> %tmp3, i32 9
  %elt10 = extractelement <16 x i32> %tmp3, i32 10
  %elt11 = extractelement <16 x i32> %tmp3, i32 11
  %elt12 = extractelement <16 x i32> %tmp3, i32 12
  %elt13 = extractelement <16 x i32> %tmp3, i32 13
  %elt14 = extractelement <16 x i32> %tmp3, i32 14
  %elt15 = extractelement <16 x i32> %tmp3, i32 15

  %add0 = add i32 %elt0, %elt1
  %add1 = add i32 %add0, %elt2
  %add2 = add i32 %add1, %elt3
  %add3 = add i32 %add2, %elt4
  %add4 = add i32 %add3, %elt5
  %add5 = add i32 %add4, %elt6
  %add6 = add i32 %add5, %elt7
  %add7 = add i32 %add6, %elt8
  %add8 = add i32 %add7, %elt9
  %add9 = add i32 %add8, %elt10
  %add10 = add i32 %add9, %elt11
  %add11 = add i32 %add10, %elt12
  %add12 = add i32 %add11, %elt13
  %add13 = add i32 %add12, %elt14
  %add14 = add i32 %add13, %elt15

  store i32 %add14, i32 addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
