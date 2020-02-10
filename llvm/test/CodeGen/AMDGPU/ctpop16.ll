; RUN: llc -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN -check-prefix=FUNC -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN -check-prefix=FUNC -check-prefix=VI %s
; RUN: llc -march=r600 -mcpu=cypress -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=EG -check-prefix=FUNC %s

declare i16 @llvm.ctpop.i16(i16) nounwind readnone
declare <2 x i16> @llvm.ctpop.v2i16(<2 x i16>) nounwind readnone
declare <4 x i16> @llvm.ctpop.v4i16(<4 x i16>) nounwind readnone
declare <8 x i16> @llvm.ctpop.v8i16(<8 x i16>) nounwind readnone
declare <16 x i16> @llvm.ctpop.v16i16(<16 x i16>) nounwind readnone

declare i32 @llvm.amdgcn.workitem.id.x() nounwind readnone

; FUNC-LABEL: {{^}}s_ctpop_i16:
; GCN: s_load_dword [[SVAL:s[0-9]+]],
; GCN: s_bcnt1_i32_b32 [[SRESULT:s[0-9]+]], [[SVAL]]
; GCN: v_mov_b32_e32 [[VRESULT:v[0-9]+]], [[SRESULT]]
; GCN: buffer_store_short [[VRESULT]],
; GCN: s_endpgm

; EG: BCNT_INT
define amdgpu_kernel void @s_ctpop_i16(i16 addrspace(1)* noalias %out, i16 %val) nounwind {
  %ctpop = call i16 @llvm.ctpop.i16(i16 %val) nounwind readnone
  store i16 %ctpop, i16 addrspace(1)* %out, align 4
  ret void
}

; XXX - Why 0 in register?
; FUNC-LABEL: {{^}}v_ctpop_i16:
; GCN: {{buffer|flat}}_load_ushort [[VAL:v[0-9]+]],
; GCN: v_bcnt_u32_b32{{(_e64)*}} [[RESULT:v[0-9]+]], [[VAL]], 0
; GCN: buffer_store_short [[RESULT]],
; GCN: s_endpgm

; EG: BCNT_INT
define amdgpu_kernel void @v_ctpop_i16(i16 addrspace(1)* noalias %out, i16 addrspace(1)* noalias %in) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %in.gep = getelementptr i16, i16 addrspace(1)* %in, i32 %tid
  %val = load i16, i16 addrspace(1)* %in.gep, align 4
  %ctpop = call i16 @llvm.ctpop.i16(i16 %val) nounwind readnone
  store i16 %ctpop, i16 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v_ctpop_add_chain_i16:
; SI: buffer_load_ushort [[VAL0:v[0-9]+]],
; SI: buffer_load_ushort [[VAL1:v[0-9]+]],
; VI: flat_load_ushort [[VAL0:v[0-9]+]],
; VI: flat_load_ushort [[VAL1:v[0-9]+]],
; GCN: v_bcnt_u32_b32{{(_e64)*}} [[MIDRESULT:v[0-9]+]], [[VAL1]], 0
; SI: v_bcnt_u32_b32_e32 [[RESULT:v[0-9]+]], [[VAL0]], [[MIDRESULT]]
; VI: v_bcnt_u32_b32 [[RESULT:v[0-9]+]], [[VAL0]], [[MIDRESULT]]
; GCN: buffer_store_short [[RESULT]],
; GCN: s_endpgm

; EG: BCNT_INT
; EG: BCNT_INT
define amdgpu_kernel void @v_ctpop_add_chain_i16(i16 addrspace(1)* noalias %out, i16 addrspace(1)* noalias %in0, i16 addrspace(1)* noalias %in1) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %in0.gep = getelementptr i16, i16 addrspace(1)* %in0, i32 %tid
  %in1.gep = getelementptr i16, i16 addrspace(1)* %in1, i32 %tid
  %val0 = load volatile i16, i16 addrspace(1)* %in0.gep, align 4
  %val1 = load volatile i16, i16 addrspace(1)* %in1.gep, align 4
  %ctpop0 = call i16 @llvm.ctpop.i16(i16 %val0) nounwind readnone
  %ctpop1 = call i16 @llvm.ctpop.i16(i16 %val1) nounwind readnone
  %add = add i16 %ctpop0, %ctpop1
  store i16 %add, i16 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v_ctpop_add_sgpr_i16:
; GCN: {{buffer|flat}}_load_ushort [[VAL0:v[0-9]+]],
; GCN: s_waitcnt
; GCN-NEXT: v_bcnt_u32_b32{{(_e64)*}} [[RESULT:v[0-9]+]], [[VAL0]], s{{[0-9]+}}
; GCN: buffer_store_short [[RESULT]],
; GCN: s_endpgm
define amdgpu_kernel void @v_ctpop_add_sgpr_i16(i16 addrspace(1)* noalias %out, i16 addrspace(1)* noalias %in, i16 %sval) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %in.gep = getelementptr i16, i16 addrspace(1)* %in, i32 %tid
  %val = load i16, i16 addrspace(1)* %in.gep, align 4
  %ctpop = call i16 @llvm.ctpop.i16(i16 %val) nounwind readnone
  %add = add i16 %ctpop, %sval
  store i16 %add, i16 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v_ctpop_v2i16:
; GCN: v_bcnt_u32_b32{{(_e64)*}}
; GCN: v_bcnt_u32_b32{{(_e64)*}}
; GCN: s_endpgm

; EG: BCNT_INT
; EG: BCNT_INT
define amdgpu_kernel void @v_ctpop_v2i16(<2 x i16> addrspace(1)* noalias %out, <2 x i16> addrspace(1)* noalias %in) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %in.gep = getelementptr <2 x i16>, <2 x i16> addrspace(1)* %in, i32 %tid
  %val = load <2 x i16>, <2 x i16> addrspace(1)* %in.gep, align 8
  %ctpop = call <2 x i16> @llvm.ctpop.v2i16(<2 x i16> %val) nounwind readnone
  store <2 x i16> %ctpop, <2 x i16> addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}v_ctpop_v4i16:
; GCN: v_bcnt_u32_b32{{(_e64)*}}
; GCN: v_bcnt_u32_b32{{(_e64)*}}
; GCN: v_bcnt_u32_b32{{(_e64)*}}
; GCN: v_bcnt_u32_b32{{(_e64)*}}
; GCN: s_endpgm

; EG: BCNT_INT
; EG: BCNT_INT
; EG: BCNT_INT
; EG: BCNT_INT
define amdgpu_kernel void @v_ctpop_v4i16(<4 x i16> addrspace(1)* noalias %out, <4 x i16> addrspace(1)* noalias %in) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %in.gep = getelementptr <4 x i16>, <4 x i16> addrspace(1)* %in, i32 %tid
  %val = load <4 x i16>, <4 x i16> addrspace(1)* %in.gep, align 16
  %ctpop = call <4 x i16> @llvm.ctpop.v4i16(<4 x i16> %val) nounwind readnone
  store <4 x i16> %ctpop, <4 x i16> addrspace(1)* %out, align 16
  ret void
}

; FUNC-LABEL: {{^}}v_ctpop_v8i16:
; GCN: v_bcnt_u32_b32{{(_e64)*}}
; GCN: v_bcnt_u32_b32{{(_e64)*}}
; GCN: v_bcnt_u32_b32{{(_e64)*}}
; GCN: v_bcnt_u32_b32{{(_e64)*}}
; GCN: v_bcnt_u32_b32{{(_e64)*}}
; GCN: v_bcnt_u32_b32{{(_e64)*}}
; GCN: v_bcnt_u32_b32{{(_e64)*}}
; GCN: v_bcnt_u32_b32{{(_e64)*}}
; GCN: s_endpgm

; EG: BCNT_INT
; EG: BCNT_INT
; EG: BCNT_INT
; EG: BCNT_INT
; EG: BCNT_INT
; EG: BCNT_INT
; EG: BCNT_INT
; EG: BCNT_INT
define amdgpu_kernel void @v_ctpop_v8i16(<8 x i16> addrspace(1)* noalias %out, <8 x i16> addrspace(1)* noalias %in) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %in.gep = getelementptr <8 x i16>, <8 x i16> addrspace(1)* %in, i32 %tid
  %val = load <8 x i16>, <8 x i16> addrspace(1)* %in.gep, align 32
  %ctpop = call <8 x i16> @llvm.ctpop.v8i16(<8 x i16> %val) nounwind readnone
  store <8 x i16> %ctpop, <8 x i16> addrspace(1)* %out, align 32
  ret void
}

; FUNC-LABEL: {{^}}v_ctpop_v16i16:
; GCN: v_bcnt_u32_b32{{(_e64)*}}
; GCN: v_bcnt_u32_b32{{(_e64)*}}
; GCN: v_bcnt_u32_b32{{(_e64)*}}
; GCN: v_bcnt_u32_b32{{(_e64)*}}
; GCN: v_bcnt_u32_b32{{(_e64)*}}
; GCN: v_bcnt_u32_b32{{(_e64)*}}
; GCN: v_bcnt_u32_b32{{(_e64)*}}
; GCN: v_bcnt_u32_b32{{(_e64)*}}
; GCN: v_bcnt_u32_b32{{(_e64)*}}
; GCN: v_bcnt_u32_b32{{(_e64)*}}
; GCN: v_bcnt_u32_b32{{(_e64)*}}
; GCN: v_bcnt_u32_b32{{(_e64)*}}
; GCN: v_bcnt_u32_b32{{(_e64)*}}
; GCN: v_bcnt_u32_b32{{(_e64)*}}
; GCN: v_bcnt_u32_b32{{(_e64)*}}
; GCN: v_bcnt_u32_b32{{(_e64)*}}
; GCN: s_endpgm

; EG: BCNT_INT
; EG: BCNT_INT
; EG: BCNT_INT
; EG: BCNT_INT
; EG: BCNT_INT
; EG: BCNT_INT
; EG: BCNT_INT
; EG: BCNT_INT
; EG: BCNT_INT
; EG: BCNT_INT
; EG: BCNT_INT
; EG: BCNT_INT
; EG: BCNT_INT
; EG: BCNT_INT
; EG: BCNT_INT
; EG: BCNT_INT
define amdgpu_kernel void @v_ctpop_v16i16(<16 x i16> addrspace(1)* noalias %out, <16 x i16> addrspace(1)* noalias %in) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %in.gep = getelementptr <16 x i16>, <16 x i16> addrspace(1)* %in, i32 %tid
  %val = load <16 x i16>, <16 x i16> addrspace(1)* %in.gep, align 32
  %ctpop = call <16 x i16> @llvm.ctpop.v16i16(<16 x i16> %val) nounwind readnone
  store <16 x i16> %ctpop, <16 x i16> addrspace(1)* %out, align 32
  ret void
}

; FUNC-LABEL: {{^}}v_ctpop_i16_add_inline_constant:
; GCN: {{buffer|flat}}_load_ushort [[VAL:v[0-9]+]],
; GCN: v_bcnt_u32_b32{{(_e64)*}} [[RESULT:v[0-9]+]], [[VAL]], 4
; GCN: buffer_store_short [[RESULT]],
; GCN: s_endpgm

; EG: BCNT_INT
define amdgpu_kernel void @v_ctpop_i16_add_inline_constant(i16 addrspace(1)* noalias %out, i16 addrspace(1)* noalias %in) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %in.gep = getelementptr i16, i16 addrspace(1)* %in, i32 %tid
  %val = load i16, i16 addrspace(1)* %in.gep, align 4
  %ctpop = call i16 @llvm.ctpop.i16(i16 %val) nounwind readnone
  %add = add i16 %ctpop, 4
  store i16 %add, i16 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v_ctpop_i16_add_inline_constant_inv:
; GCN: {{buffer|flat}}_load_ushort [[VAL:v[0-9]+]],
; GCN: v_bcnt_u32_b32{{(_e64)*}} [[RESULT:v[0-9]+]], [[VAL]], 4
; GCN: buffer_store_short [[RESULT]],
; GCN: s_endpgm

; EG: BCNT_INT
define amdgpu_kernel void @v_ctpop_i16_add_inline_constant_inv(i16 addrspace(1)* noalias %out, i16 addrspace(1)* noalias %in) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %in.gep = getelementptr i16, i16 addrspace(1)* %in, i32 %tid
  %val = load i16, i16 addrspace(1)* %in.gep, align 4
  %ctpop = call i16 @llvm.ctpop.i16(i16 %val) nounwind readnone
  %add = add i16 4, %ctpop
  store i16 %add, i16 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v_ctpop_i16_add_literal:
; GCN-DAG: {{buffer|flat}}_load_ushort [[VAL:v[0-9]+]],
; SI-DAG: s_movk_i32 [[LIT:s[0-9]+]], 0x3e7
; VI-DAG: s_movk_i32 [[LIT:s[0-9]+]], 0x3e7
; SI: v_bcnt_u32_b32_e64 [[RESULT:v[0-9]+]], [[VAL]], [[LIT]]
; VI: v_bcnt_u32_b32 [[RESULT:v[0-9]+]], [[VAL]], [[LIT]]
; GCN: buffer_store_short [[RESULT]],
; GCN: s_endpgm
define amdgpu_kernel void @v_ctpop_i16_add_literal(i16 addrspace(1)* noalias %out, i16 addrspace(1)* noalias %in) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %in.gep = getelementptr i16, i16 addrspace(1)* %in, i32 %tid
  %val = load i16, i16 addrspace(1)* %in.gep, align 4
  %ctpop = call i16 @llvm.ctpop.i16(i16 %val) nounwind readnone
  %add = add i16 %ctpop, 999
  store i16 %add, i16 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v_ctpop_i16_add_var:
; GCN-DAG: {{buffer|flat}}_load_ushort [[VAL:v[0-9]+]],
; GCN-DAG: s_load_dword [[VAR:s[0-9]+]],
; GCN: v_bcnt_u32_b32{{(_e64)*}} [[RESULT:v[0-9]+]], [[VAL]], [[VAR]]
; GCN: buffer_store_short [[RESULT]],
; GCN: s_endpgm

; EG: BCNT_INT
define amdgpu_kernel void @v_ctpop_i16_add_var(i16 addrspace(1)* noalias %out, i16 addrspace(1)* noalias %in, i16 %const) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %in.gep = getelementptr i16, i16 addrspace(1)* %in, i32 %tid
  %val = load i16, i16 addrspace(1)* %in.gep, align 4
  %ctpop = call i16 @llvm.ctpop.i16(i16 %val) nounwind readnone
  %add = add i16 %ctpop, %const
  store i16 %add, i16 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v_ctpop_i16_add_var_inv:
; GCN-DAG: {{buffer|flat}}_load_ushort [[VAL:v[0-9]+]],
; GCN-DAG: s_load_dword [[VAR:s[0-9]+]],
; GCN: v_bcnt_u32_b32{{(_e64)*}} [[RESULT:v[0-9]+]], [[VAL]], [[VAR]]
; GCN: buffer_store_short [[RESULT]],
; GCN: s_endpgm

; EG: BCNT_INT
define amdgpu_kernel void @v_ctpop_i16_add_var_inv(i16 addrspace(1)* noalias %out, i16 addrspace(1)* noalias %in, i16 %const) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %in.gep = getelementptr i16, i16 addrspace(1)* %in, i32 %tid
  %val = load i16, i16 addrspace(1)* %in.gep, align 4
  %ctpop = call i16 @llvm.ctpop.i16(i16 %val) nounwind readnone
  %add = add i16 %const, %ctpop
  store i16 %add, i16 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v_ctpop_i16_add_vvar_inv:
; SI: buffer_load_ushort [[VAR:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], 0 addr64
; SI: buffer_load_ushort [[VAL:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], 0 addr64
; SI: v_bcnt_u32_b32_e32 [[RESULT:v[0-9]+]], [[VAR]], [[VAL]]
; VI: flat_load_ushort [[VAR:v[0-9]+]], v[{{[0-9]+:[0-9]+}}]
; VI: flat_load_ushort [[VAL:v[0-9]+]], v[{{[0-9]+:[0-9]+}}]
; VI: v_bcnt_u32_b32 [[RESULT:v[0-9]+]], [[VAR]], [[VAL]]
; GCN: buffer_store_short [[RESULT]],
; GCN: s_endpgm

; EG: BCNT_INT
define amdgpu_kernel void @v_ctpop_i16_add_vvar_inv(i16 addrspace(1)* noalias %out, i16 addrspace(1)* noalias %in, i16 addrspace(1)* noalias %constptr) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %in.gep = getelementptr i16, i16 addrspace(1)* %in, i32 %tid
  %val = load i16, i16 addrspace(1)* %in.gep, align 4
  %ctpop = call i16 @llvm.ctpop.i16(i16 %val) nounwind readnone
  %gep = getelementptr i16, i16 addrspace(1)* %constptr, i32 %tid
  %const = load i16, i16 addrspace(1)* %gep, align 4
  %add = add i16 %const, %ctpop
  store i16 %add, i16 addrspace(1)* %out, align 4
  ret void
}

; FIXME: We currently disallow SALU instructions in all branches,
; but there are some cases when the should be allowed.

; FUNC-LABEL: {{^}}ctpop_i16_in_br:
; SI: s_load_dword [[VAL:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0xd
; VI: s_load_dword [[VAL:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0x34

; GCN: s_and_b32 [[CTPOP_ARG:s[0-9]+]], [[VAL]], 0xffff
; GCN: s_bcnt1_i32_b32  [[SRESULT:s[0-9]+]], [[CTPOP_ARG]]
; GCN: v_mov_b32_e32 [[RESULT:v[0-9]+]], [[SRESULT]]
; GCN: buffer_store_short [[RESULT]],
; GCN: s_endpgm
; EG: BCNT_INT
define amdgpu_kernel void @ctpop_i16_in_br(i16 addrspace(1)* %out, i16 addrspace(1)* %in, i16 %ctpop_arg, i16 %cond) {
entry:
  %tmp0 = icmp eq i16 %cond, 0
  br i1 %tmp0, label %if, label %else

if:
  %tmp2 = call i16 @llvm.ctpop.i16(i16 %ctpop_arg)
  br label %endif

else:
  %tmp3 = getelementptr i16, i16 addrspace(1)* %in, i16 1
  %tmp4 = load i16, i16 addrspace(1)* %tmp3
  br label %endif

endif:
  %tmp5 = phi i16 [%tmp2, %if], [%tmp4, %else]
  store i16 %tmp5, i16 addrspace(1)* %out
  ret void
}
