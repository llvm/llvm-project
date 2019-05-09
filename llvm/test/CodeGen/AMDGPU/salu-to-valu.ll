; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=GCN-NOHSA -check-prefix=SI %s
; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=amdgcn -mcpu=bonaire -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=GCN-NOHSA -check-prefix=CI %s
; RUN:  llc -amdgpu-scalarize-global-loads=false  -mtriple=amdgcn--amdhsa -mcpu=bonaire -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=CI --check-prefix=GCN-HSA %s

declare i32 @llvm.amdgcn.workitem.id.x() #0
declare i32 @llvm.amdgcn.workitem.id.y() #0

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
; GCN-NOHSA: buffer_load_ubyte v{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], 0 addr64
; GCN-NOHSA: buffer_load_ubyte v{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], 0 addr64
; GCN-HSA: flat_load_ubyte v{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}
; GCN-HSA: flat_load_ubyte v{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}

define amdgpu_kernel void @mubuf(i32 addrspace(1)* %out, i8 addrspace(1)* %in) #1 {
entry:
  %tmp = call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = call i32 @llvm.amdgcn.workitem.id.y()
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
; FIXME: movs can be moved before nop to reduce count

; GCN-LABEL: {{^}}smrd_valu:
; SI: s_movk_i32 [[OFFSET:s[0-9]+]], 0x2ee0
; SI: s_mov_b32
; GCN: v_readfirstlane_b32 s[[PTR_LO:[0-9]+]], v{{[0-9]+}}
; GCN: v_readfirstlane_b32 s[[PTR_HI:[0-9]+]], v{{[0-9]+}}
; SI: s_nop 3
; SI: s_load_dword [[OUT:s[0-9]+]], s{{\[}}[[PTR_LO]]:[[PTR_HI]]{{\]}}, [[OFFSET]]

; CI: s_load_dword [[OUT:s[0-9]+]], s{{\[}}[[PTR_LO]]:[[PTR_HI]]{{\]}}, 0xbb8
; GCN: v_mov_b32_e32 [[V_OUT:v[0-9]+]], [[OUT]]
; GCN-NOHSA: buffer_store_dword [[V_OUT]]
; GCN-HSA: flat_store_dword {{.*}}, [[V_OUT]]
define amdgpu_kernel void @smrd_valu(i32 addrspace(4)* addrspace(1)* %in, i32 %a, i32 %b, i32 addrspace(1)* %out) #1 {
entry:
  %tmp = icmp ne i32 %a, 0
  br i1 %tmp, label %if, label %else

if:                                               ; preds = %entry
  %tmp1 = load i32 addrspace(4)*, i32 addrspace(4)* addrspace(1)* %in
  br label %endif

else:                                             ; preds = %entry
  %tmp2 = getelementptr i32 addrspace(4)*, i32 addrspace(4)* addrspace(1)* %in
  %tmp3 = load i32 addrspace(4)*, i32 addrspace(4)* addrspace(1)* %tmp2
  br label %endif

endif:                                            ; preds = %else, %if
  %tmp4 = phi i32 addrspace(4)* [ %tmp1, %if ], [ %tmp3, %else ]
  %tmp5 = getelementptr i32, i32 addrspace(4)* %tmp4, i32 3000
  %tmp6 = load i32, i32 addrspace(4)* %tmp5
  store i32 %tmp6, i32 addrspace(1)* %out
  ret void
}

; Test moving an SMRD with an immediate offset to the VALU

; GCN-LABEL: {{^}}smrd_valu2:
; GCN-NOHSA-NOT: v_add
; GCN-NOHSA: buffer_load_dword v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, s[{{[0-9]+:[0-9]+}}], 0 addr64 offset:16{{$}}
; GCN-HSA: flat_load_dword v{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}]
define amdgpu_kernel void @smrd_valu2(i32 addrspace(1)* %out, [8 x i32] addrspace(4)* %in) #1 {
entry:
  %tmp = call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = add i32 %tmp, 4
  %tmp2 = getelementptr [8 x i32], [8 x i32] addrspace(4)* %in, i32 %tmp, i32 4
  %tmp3 = load i32, i32 addrspace(4)* %tmp2
  store i32 %tmp3, i32 addrspace(1)* %out
  ret void
}

; Use a big offset that will use the SMRD literal offset on CI
; GCN-LABEL: {{^}}smrd_valu_ci_offset:
; GCN-NOHSA-NOT: v_add
; GCN-NOHSA: s_movk_i32 [[OFFSET:s[0-9]+]], 0x4e20{{$}}
; GCN-NOHSA-NOT: v_add
; GCN-NOHSA: buffer_load_dword v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, s[{{[0-9]+:[0-9]+}}], [[OFFSET]] addr64{{$}}
; GCN-NOHSA: v_add_i32_e32
; GCN-NOHSA: buffer_store_dword
; GCN-HSA: flat_load_dword v{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}]
; GCN-HSA: flat_store_dword v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}
define amdgpu_kernel void @smrd_valu_ci_offset(i32 addrspace(1)* %out, i32 addrspace(4)* %in, i32 %c) #1 {
entry:
  %tmp = call i32 @llvm.amdgcn.workitem.id.x()
  %tmp2 = getelementptr i32, i32 addrspace(4)* %in, i32 %tmp
  %tmp3 = getelementptr i32, i32 addrspace(4)* %tmp2, i32 5000
  %tmp4 = load i32, i32 addrspace(4)* %tmp3
  %tmp5 = add i32 %tmp4, %c
  store i32 %tmp5, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}smrd_valu_ci_offset_x2:
; GCN-NOHSA-NOT: v_add
; GCN-NOHSA: s_mov_b32 [[OFFSET:s[0-9]+]], 0x9c40{{$}}
; GCN-NOHSA-NOT: v_add
; GCN-NOHSA: buffer_load_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, s[{{[0-9]+:[0-9]+}}], [[OFFSET]] addr64{{$}}
; GCN-NOHSA: v_or_b32_e32 {{v[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}
; GCN-NOHSA: v_or_b32_e32 {{v[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}
; GCN-NOHSA: buffer_store_dwordx2
; GCN-HSA: flat_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
define amdgpu_kernel void @smrd_valu_ci_offset_x2(i64 addrspace(1)* %out, i64 addrspace(4)* %in, i64 %c) #1 {
entry:
  %tmp = call i32 @llvm.amdgcn.workitem.id.x()
  %tmp2 = getelementptr i64, i64 addrspace(4)* %in, i32 %tmp
  %tmp3 = getelementptr i64, i64 addrspace(4)* %tmp2, i32 5000
  %tmp4 = load i64, i64 addrspace(4)* %tmp3
  %tmp5 = or i64 %tmp4, %c
  store i64 %tmp5, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}smrd_valu_ci_offset_x4:
; GCN-NOHSA-NOT: v_add
; GCN-NOHSA: s_movk_i32 [[OFFSET:s[0-9]+]], 0x4d20{{$}}
; GCN-NOHSA-NOT: v_add
; GCN-NOHSA: buffer_load_dwordx4 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, s[{{[0-9]+:[0-9]+}}], [[OFFSET]] addr64{{$}}
; GCN-NOHSA: v_or_b32_e32 {{v[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}
; GCN-NOHSA: v_or_b32_e32 {{v[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}
; GCN-NOHSA: v_or_b32_e32 {{v[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}
; GCN-NOHSA: v_or_b32_e32 {{v[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}
; GCN-NOHSA: buffer_store_dwordx4
; GCN-HSA: flat_load_dwordx4 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
define amdgpu_kernel void @smrd_valu_ci_offset_x4(<4 x i32> addrspace(1)* %out, <4 x i32> addrspace(4)* %in, <4 x i32> %c) #1 {
entry:
  %tmp = call i32 @llvm.amdgcn.workitem.id.x()
  %tmp2 = getelementptr <4 x i32>, <4 x i32> addrspace(4)* %in, i32 %tmp
  %tmp3 = getelementptr <4 x i32>, <4 x i32> addrspace(4)* %tmp2, i32 1234
  %tmp4 = load <4 x i32>, <4 x i32> addrspace(4)* %tmp3
  %tmp5 = or <4 x i32> %tmp4, %c
  store <4 x i32> %tmp5, <4 x i32> addrspace(1)* %out
  ret void
}

; Original scalar load uses SGPR offset on SI and 32-bit literal on
; CI.

; GCN-LABEL: {{^}}smrd_valu_ci_offset_x8:
; GCN-NOHSA: s_mov_b32 [[OFFSET0:s[0-9]+]], 0x9a40{{$}}
; GCN-NOHSA-NOT: v_add
; GCN-NOHSA: s_mov_b32 [[OFFSET1:s[0-9]+]], 0x9a50{{$}}
; GCN-NOHSA-NOT: v_add
; GCN-NOHSA: buffer_load_dwordx4 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, s[{{[0-9]+:[0-9]+}}], [[OFFSET1]] addr64{{$}}
; GCN-NOHSA: buffer_load_dwordx4 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, s[{{[0-9]+:[0-9]+}}], [[OFFSET0]] addr64{{$}}

; GCN-NOHSA: v_or_b32_e32 {{v[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}
; GCN-NOHSA: v_or_b32_e32 {{v[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}
; GCN-NOHSA: v_or_b32_e32 {{v[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}
; GCN-NOHSA: v_or_b32_e32 {{v[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}
; GCN-NOHSA: v_or_b32_e32 {{v[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}
; GCN-NOHSA: v_or_b32_e32 {{v[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}
; GCN-NOHSA: v_or_b32_e32 {{v[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}
; GCN-NOHSA: v_or_b32_e32 {{v[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}
; GCN-NOHSA: buffer_store_dwordx4
; GCN-NOHSA: buffer_store_dwordx4
; GCN-HSA: flat_load_dwordx4
; GCN-HSA: flat_load_dwordx4
define amdgpu_kernel void @smrd_valu_ci_offset_x8(<8 x i32> addrspace(1)* %out, <8 x i32> addrspace(4)* %in, <8 x i32> %c) #1 {
entry:
  %tmp = call i32 @llvm.amdgcn.workitem.id.x()
  %tmp2 = getelementptr <8 x i32>, <8 x i32> addrspace(4)* %in, i32 %tmp
  %tmp3 = getelementptr <8 x i32>, <8 x i32> addrspace(4)* %tmp2, i32 1234
  %tmp4 = load <8 x i32>, <8 x i32> addrspace(4)* %tmp3
  %tmp5 = or <8 x i32> %tmp4, %c
  store <8 x i32> %tmp5, <8 x i32> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}smrd_valu_ci_offset_x16:

; GCN-NOHSA-DAG: s_mov_b32 [[OFFSET0:s[0-9]+]], 0x13480{{$}}
; GCN-NOHSA-DAG: buffer_load_dwordx4 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, s[{{[0-9]+:[0-9]+}}], [[OFFSET0]] addr64{{$}}
; GCN-NOHSA-DAG: s_mov_b32 [[OFFSET1:s[0-9]+]], 0x13490{{$}}
; GCN-NOHSA-DAG: buffer_load_dwordx4 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, s[{{[0-9]+:[0-9]+}}], [[OFFSET1]] addr64{{$}}
; GCN-NOHSA-DAG: s_mov_b32 [[OFFSET2:s[0-9]+]], 0x134a0{{$}}
; GCN-NOHSA-DAG: buffer_load_dwordx4 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, s[{{[0-9]+:[0-9]+}}], [[OFFSET2]] addr64{{$}}
; GCN-NOHSA-DAG: s_mov_b32 [[OFFSET3:s[0-9]+]], 0x134b0{{$}}
; GCN-NOHSA-DAG: buffer_load_dwordx4 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, s[{{[0-9]+:[0-9]+}}], [[OFFSET3]] addr64{{$}}

; GCN-NOHSA: v_or_b32_e32 {{v[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}
; GCN-NOHSA: v_or_b32_e32 {{v[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}
; GCN-NOHSA: v_or_b32_e32 {{v[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}
; GCN-NOHSA: v_or_b32_e32 {{v[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}
; GCN-NOHSA: v_or_b32_e32 {{v[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}
; GCN-NOHSA: v_or_b32_e32 {{v[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}
; GCN-NOHSA: v_or_b32_e32 {{v[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}
; GCN-NOHSA: v_or_b32_e32 {{v[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}
; GCN-NOHSA: buffer_store_dwordx4
; GCN-NOHSA: buffer_store_dwordx4
; GCN-NOHSA: buffer_store_dwordx4
; GCN-NOHSA: buffer_store_dwordx4

; GCN-HSA: flat_load_dwordx4
; GCN-HSA: flat_load_dwordx4
; GCN-HSA: flat_load_dwordx4
; GCN-HSA: flat_load_dwordx4

; GCN: s_endpgm
define amdgpu_kernel void @smrd_valu_ci_offset_x16(<16 x i32> addrspace(1)* %out, <16 x i32> addrspace(4)* %in, <16 x i32> %c) #1 {
entry:
  %tmp = call i32 @llvm.amdgcn.workitem.id.x()
  %tmp2 = getelementptr <16 x i32>, <16 x i32> addrspace(4)* %in, i32 %tmp
  %tmp3 = getelementptr <16 x i32>, <16 x i32> addrspace(4)* %tmp2, i32 1234
  %tmp4 = load <16 x i32>, <16 x i32> addrspace(4)* %tmp3
  %tmp5 = or <16 x i32> %tmp4, %c
  store <16 x i32> %tmp5, <16 x i32> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}smrd_valu2_salu_user:
; GCN-NOHSA: buffer_load_dword [[MOVED:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:16{{$}}
; GCN-HSA: flat_load_dword [[MOVED:v[0-9]+]], v[{{[0-9+:[0-9]+}}]
; GCN: v_add_i32_e32 [[ADD:v[0-9]+]], vcc, s{{[0-9]+}}, [[MOVED]]
; GCN-NOHSA: buffer_store_dword [[ADD]]
; GCN-HSA: flat_store_dword {{.*}}, [[ADD]]
define amdgpu_kernel void @smrd_valu2_salu_user(i32 addrspace(1)* %out, [8 x i32] addrspace(4)* %in, i32 %a) #1 {
entry:
  %tmp = call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = add i32 %tmp, 4
  %tmp2 = getelementptr [8 x i32], [8 x i32] addrspace(4)* %in, i32 %tmp, i32 4
  %tmp3 = load i32, i32 addrspace(4)* %tmp2
  %tmp4 = add i32 %tmp3, %a
  store i32 %tmp4, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}smrd_valu2_max_smrd_offset:
; GCN-NOHSA: buffer_load_dword v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:1020{{$}}
; GCN-HSA flat_load_dword v{{[0-9]}}, v{{[0-9]+:[0-9]+}}
define amdgpu_kernel void @smrd_valu2_max_smrd_offset(i32 addrspace(1)* %out, [1024 x i32] addrspace(4)* %in) #1 {
entry:
  %tmp = call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = add i32 %tmp, 4
  %tmp2 = getelementptr [1024 x i32], [1024 x i32] addrspace(4)* %in, i32 %tmp, i32 255
  %tmp3 = load i32, i32 addrspace(4)* %tmp2
  store i32 %tmp3, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}smrd_valu2_mubuf_offset:
; GCN-NOHSA-NOT: v_add
; GCN-NOHSA: buffer_load_dword v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:1024{{$}}
; GCN-HSA: flat_load_dword v{{[0-9]}}, v[{{[0-9]+:[0-9]+}}]
define amdgpu_kernel void @smrd_valu2_mubuf_offset(i32 addrspace(1)* %out, [1024 x i32] addrspace(4)* %in) #1 {
entry:
  %tmp = call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = add i32 %tmp, 4
  %tmp2 = getelementptr [1024 x i32], [1024 x i32] addrspace(4)* %in, i32 %tmp, i32 256
  %tmp3 = load i32, i32 addrspace(4)* %tmp2
  store i32 %tmp3, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}s_load_imm_v8i32:
; GCN-NOHSA: buffer_load_dwordx4
; GCN-NOHSA: buffer_load_dwordx4
; GCN-HSA: flat_load_dwordx4
; GCN-HSA: flat_load_dwordx4
define amdgpu_kernel void @s_load_imm_v8i32(<8 x i32> addrspace(1)* %out, i32 addrspace(4)* nocapture readonly %in) #1 {
entry:
  %tmp0 = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = getelementptr inbounds i32, i32 addrspace(4)* %in, i32 %tmp0
  %tmp2 = bitcast i32 addrspace(4)* %tmp1 to <8 x i32> addrspace(4)*
  %tmp3 = load <8 x i32>, <8 x i32> addrspace(4)* %tmp2, align 4
  store <8 x i32> %tmp3, <8 x i32> addrspace(1)* %out, align 32
  ret void
}

; GCN-LABEL: {{^}}s_load_imm_v8i32_salu_user:
; GCN-NOHSA: buffer_load_dwordx4
; GCN-NOHSA: buffer_load_dwordx4
; GCN-NOHSA: v_add_i32_e32
; GCN-NOHSA: v_add_i32_e32
; GCN-NOHSA: v_add_i32_e32
; GCN-NOHSA: v_add_i32_e32
; GCN-NOHSA: v_add_i32_e32
; GCN-NOHSA: v_add_i32_e32
; GCN-NOHSA: v_add_i32_e32
; GCN-NOHSA: buffer_store_dword
; GCN-HSA: flat_load_dwordx4
; GCN-HSA: flat_load_dwordx4
define amdgpu_kernel void @s_load_imm_v8i32_salu_user(i32 addrspace(1)* %out, i32 addrspace(4)* nocapture readonly %in) #1 {
entry:
  %tmp0 = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = getelementptr inbounds i32, i32 addrspace(4)* %in, i32 %tmp0
  %tmp2 = bitcast i32 addrspace(4)* %tmp1 to <8 x i32> addrspace(4)*
  %tmp3 = load <8 x i32>, <8 x i32> addrspace(4)* %tmp2, align 4

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
; GCN-NOHSA: buffer_load_dwordx4
; GCN-NOHSA: buffer_load_dwordx4
; GCN-NOHSA: buffer_load_dwordx4
; GCN-NOHSA: buffer_load_dwordx4
; GCN-HSA: flat_load_dwordx4
; GCN-HSA: flat_load_dwordx4
; GCN-HSA: flat_load_dwordx4
; GCN-HSA: flat_load_dwordx4
define amdgpu_kernel void @s_load_imm_v16i32(<16 x i32> addrspace(1)* %out, i32 addrspace(4)* nocapture readonly %in) #1 {
entry:
  %tmp0 = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = getelementptr inbounds i32, i32 addrspace(4)* %in, i32 %tmp0
  %tmp2 = bitcast i32 addrspace(4)* %tmp1 to <16 x i32> addrspace(4)*
  %tmp3 = load <16 x i32>, <16 x i32> addrspace(4)* %tmp2, align 4
  store <16 x i32> %tmp3, <16 x i32> addrspace(1)* %out, align 32
  ret void
}

; GCN-LABEL: {{^}}s_load_imm_v16i32_salu_user:
; GCN-NOHSA: buffer_load_dwordx4
; GCN-NOHSA: buffer_load_dwordx4
; GCN-NOHSA: buffer_load_dwordx4
; GCN-NOHSA: buffer_load_dwordx4
; GCN-NOHSA: v_add_i32_e32
; GCN-NOHSA: v_add_i32_e32
; GCN-NOHSA: v_add_i32_e32
; GCN-NOHSA: v_add_i32_e32
; GCN-NOHSA: v_add_i32_e32
; GCN-NOHSA: v_add_i32_e32
; GCN-NOHSA: v_add_i32_e32
; GCN-NOHSA: v_add_i32_e32
; GCN-NOHSA: v_add_i32_e32
; GCN-NOHSA: v_add_i32_e32
; GCN-NOHSA: v_add_i32_e32
; GCN-NOHSA: v_add_i32_e32
; GCN-NOHSA: v_add_i32_e32
; GCN-NOHSA: v_add_i32_e32
; GCN-NOHSA: v_add_i32_e32
; GCN-NOHSA: buffer_store_dword
; GCN-HSA: flat_load_dwordx4
; GCN-HSA: flat_load_dwordx4
; GCN-HSA: flat_load_dwordx4
; GCN-HSA: flat_load_dwordx4
define amdgpu_kernel void @s_load_imm_v16i32_salu_user(i32 addrspace(1)* %out, i32 addrspace(4)* nocapture readonly %in) #1 {
entry:
  %tmp0 = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = getelementptr inbounds i32, i32 addrspace(4)* %in, i32 %tmp0
  %tmp2 = bitcast i32 addrspace(4)* %tmp1 to <16 x i32> addrspace(4)*
  %tmp3 = load <16 x i32>, <16 x i32> addrspace(4)* %tmp2, align 4

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

; Make sure we legalize vopc operands after moving an sopc to the value.

; {{^}}sopc_vopc_legalize_bug:
; GCN: s_load_dword [[SGPR:s[0-9]+]]
; GCN: v_cmp_le_u32_e32 vcc, [[SGPR]], v{{[0-9]+}}
; GCN: s_and_b64 vcc, exec, vcc
; GCN: s_cbranch_vccnz [[EXIT:[A-Z0-9_]+]]
; GCN: v_mov_b32_e32 [[ONE:v[0-9]+]], 1
; GCN-NOHSA: buffer_store_dword [[ONE]]
; GCN-HSA: flat_store_dword v[{{[0-9]+:[0-9]+}}], [[ONE]]
; GCN: {{^}}[[EXIT]]:
; GCN: s_endpgm
define amdgpu_kernel void @sopc_vopc_legalize_bug(i32 %cond, i32 addrspace(1)* %out, i32 addrspace(1)* %in) {
bb3:                                              ; preds = %bb2
  %tmp0 = bitcast i32 %cond to float
  %tmp1 = fadd float %tmp0, 2.500000e-01
  %tmp2 = bitcast float %tmp1 to i32
  %tmp3 = icmp ult i32 %tmp2, %cond
  br i1 %tmp3, label %bb6, label %bb7

bb6:
  store i32 1, i32 addrspace(1)* %out
  br label %bb7

bb7:                                              ; preds = %bb3
  ret void
}

; GCN-LABEL: {{^}}phi_visit_order:
; GCN: v_add_i32_e64 v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}, 1, v{{[0-9]+}}
define amdgpu_kernel void @phi_visit_order() {
bb:
  br label %bb1

bb1:
  %tmp = phi i32 [ 0, %bb ], [ %tmp5, %bb4 ]
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %cnd = icmp eq i32 %tid, 0
  br i1 %cnd, label %bb4, label %bb2

bb2:
  %tmp3 = add nsw i32 %tmp, 1
  br label %bb4

bb4:
  %tmp5 = phi i32 [ %tmp3, %bb2 ], [ %tmp, %bb1 ]
  store volatile i32 %tmp5, i32 addrspace(1)* undef
  br label %bb1
}

; GCN-LABEL: {{^}}phi_imm_in_sgprs
; GCN: s_movk_i32 [[A:s[0-9]+]], 0x400
; GCN: s_movk_i32 [[B:s[0-9]+]], 0x400
; GCN: [[LOOP_LABEL:[0-9a-zA-Z_]+]]:
; GCN: s_xor_b32 [[B]], [[B]], [[A]]
; GCN: s_cbranch_scc{{[01]}} [[LOOP_LABEL]]
define amdgpu_kernel void @phi_imm_in_sgprs(i32 addrspace(3)* %out, i32 %cond) {
entry:
  br label %loop

loop:
  %i = phi i32 [0, %entry], [%i.add, %loop]
  %offset = phi i32 [1024, %entry], [%offset.xor, %loop]
  %offset.xor = xor i32 %offset, 1024
  %offset.i = add i32 %offset.xor, %i
  %ptr = getelementptr i32, i32 addrspace(3)* %out, i32 %offset.i
  store i32 0, i32 addrspace(3)* %ptr
  %i.add = add i32 %i, 1
  %cmp = icmp ult i32 %i.add, %cond
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
