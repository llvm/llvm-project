; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=FUNC -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=FUNC -check-prefix=VI %s
; RUN: llc -march=r600 -mcpu=cypress -verify-machineinstrs < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

declare i32 @llvm.ctpop.i32(i32) nounwind readnone
declare <2 x i32> @llvm.ctpop.v2i32(<2 x i32>) nounwind readnone
declare <4 x i32> @llvm.ctpop.v4i32(<4 x i32>) nounwind readnone
declare <8 x i32> @llvm.ctpop.v8i32(<8 x i32>) nounwind readnone
declare <16 x i32> @llvm.ctpop.v16i32(<16 x i32>) nounwind readnone

; FUNC-LABEL: {{^}}s_ctpop_i32:
; GCN: s_load_dword [[SVAL:s[0-9]+]],
; GCN: s_bcnt1_i32_b32 [[SRESULT:s[0-9]+]], [[SVAL]]
; GCN: v_mov_b32_e32 [[VRESULT:v[0-9]+]], [[SRESULT]]
; GCN: buffer_store_dword [[VRESULT]],
; GCN: s_endpgm

; EG: BCNT_INT
define void @s_ctpop_i32(i32 addrspace(1)* noalias %out, i32 %val) nounwind {
  %ctpop = call i32 @llvm.ctpop.i32(i32 %val) nounwind readnone
  store i32 %ctpop, i32 addrspace(1)* %out, align 4
  ret void
}

; XXX - Why 0 in register?
; FUNC-LABEL: {{^}}v_ctpop_i32:
; GCN: buffer_load_dword [[VAL:v[0-9]+]],
; GCN: v_bcnt_u32_b32_e64 [[RESULT:v[0-9]+]], [[VAL]], 0
; GCN: buffer_store_dword [[RESULT]],
; GCN: s_endpgm

; EG: BCNT_INT
define void @v_ctpop_i32(i32 addrspace(1)* noalias %out, i32 addrspace(1)* noalias %in) nounwind {
  %val = load i32, i32 addrspace(1)* %in, align 4
  %ctpop = call i32 @llvm.ctpop.i32(i32 %val) nounwind readnone
  store i32 %ctpop, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v_ctpop_add_chain_i32:
; GCN: buffer_load_dword [[VAL1:v[0-9]+]],
; GCN: buffer_load_dword [[VAL0:v[0-9]+]],
; GCN: v_bcnt_u32_b32_e64 [[MIDRESULT:v[0-9]+]], [[VAL1]], 0
; SI: v_bcnt_u32_b32_e32 [[RESULT:v[0-9]+]], [[VAL0]], [[MIDRESULT]]
; VI: v_bcnt_u32_b32_e64 [[RESULT:v[0-9]+]], [[VAL0]], [[MIDRESULT]]
; GCN: buffer_store_dword [[RESULT]],
; GCN: s_endpgm

; EG: BCNT_INT
; EG: BCNT_INT
define void @v_ctpop_add_chain_i32(i32 addrspace(1)* noalias %out, i32 addrspace(1)* noalias %in0, i32 addrspace(1)* noalias %in1) nounwind {
  %val0 = load i32, i32 addrspace(1)* %in0, align 4
  %val1 = load i32, i32 addrspace(1)* %in1, align 4
  %ctpop0 = call i32 @llvm.ctpop.i32(i32 %val0) nounwind readnone
  %ctpop1 = call i32 @llvm.ctpop.i32(i32 %val1) nounwind readnone
  %add = add i32 %ctpop0, %ctpop1
  store i32 %add, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v_ctpop_add_sgpr_i32:
; GCN: buffer_load_dword [[VAL0:v[0-9]+]],
; GCN: s_waitcnt
; GCN-NEXT: v_bcnt_u32_b32_e64 [[RESULT:v[0-9]+]], [[VAL0]], s{{[0-9]+}}
; GCN: buffer_store_dword [[RESULT]],
; GCN: s_endpgm
define void @v_ctpop_add_sgpr_i32(i32 addrspace(1)* noalias %out, i32 addrspace(1)* noalias %in0, i32 addrspace(1)* noalias %in1, i32 %sval) nounwind {
  %val0 = load i32, i32 addrspace(1)* %in0, align 4
  %ctpop0 = call i32 @llvm.ctpop.i32(i32 %val0) nounwind readnone
  %add = add i32 %ctpop0, %sval
  store i32 %add, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v_ctpop_v2i32:
; GCN: v_bcnt_u32_b32_e64
; GCN: v_bcnt_u32_b32_e64
; GCN: s_endpgm

; EG: BCNT_INT
; EG: BCNT_INT
define void @v_ctpop_v2i32(<2 x i32> addrspace(1)* noalias %out, <2 x i32> addrspace(1)* noalias %in) nounwind {
  %val = load <2 x i32>, <2 x i32> addrspace(1)* %in, align 8
  %ctpop = call <2 x i32> @llvm.ctpop.v2i32(<2 x i32> %val) nounwind readnone
  store <2 x i32> %ctpop, <2 x i32> addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}v_ctpop_v4i32:
; GCN: v_bcnt_u32_b32_e64
; GCN: v_bcnt_u32_b32_e64
; GCN: v_bcnt_u32_b32_e64
; GCN: v_bcnt_u32_b32_e64
; GCN: s_endpgm

; EG: BCNT_INT
; EG: BCNT_INT
; EG: BCNT_INT
; EG: BCNT_INT
define void @v_ctpop_v4i32(<4 x i32> addrspace(1)* noalias %out, <4 x i32> addrspace(1)* noalias %in) nounwind {
  %val = load <4 x i32>, <4 x i32> addrspace(1)* %in, align 16
  %ctpop = call <4 x i32> @llvm.ctpop.v4i32(<4 x i32> %val) nounwind readnone
  store <4 x i32> %ctpop, <4 x i32> addrspace(1)* %out, align 16
  ret void
}

; FUNC-LABEL: {{^}}v_ctpop_v8i32:
; GCN: v_bcnt_u32_b32_e64
; GCN: v_bcnt_u32_b32_e64
; GCN: v_bcnt_u32_b32_e64
; GCN: v_bcnt_u32_b32_e64
; GCN: v_bcnt_u32_b32_e64
; GCN: v_bcnt_u32_b32_e64
; GCN: v_bcnt_u32_b32_e64
; GCN: v_bcnt_u32_b32_e64
; GCN: s_endpgm

; EG: BCNT_INT
; EG: BCNT_INT
; EG: BCNT_INT
; EG: BCNT_INT
; EG: BCNT_INT
; EG: BCNT_INT
; EG: BCNT_INT
; EG: BCNT_INT
define void @v_ctpop_v8i32(<8 x i32> addrspace(1)* noalias %out, <8 x i32> addrspace(1)* noalias %in) nounwind {
  %val = load <8 x i32>, <8 x i32> addrspace(1)* %in, align 32
  %ctpop = call <8 x i32> @llvm.ctpop.v8i32(<8 x i32> %val) nounwind readnone
  store <8 x i32> %ctpop, <8 x i32> addrspace(1)* %out, align 32
  ret void
}

; FUNC-LABEL: {{^}}v_ctpop_v16i32:
; GCN: v_bcnt_u32_b32_e64
; GCN: v_bcnt_u32_b32_e64
; GCN: v_bcnt_u32_b32_e64
; GCN: v_bcnt_u32_b32_e64
; GCN: v_bcnt_u32_b32_e64
; GCN: v_bcnt_u32_b32_e64
; GCN: v_bcnt_u32_b32_e64
; GCN: v_bcnt_u32_b32_e64
; GCN: v_bcnt_u32_b32_e64
; GCN: v_bcnt_u32_b32_e64
; GCN: v_bcnt_u32_b32_e64
; GCN: v_bcnt_u32_b32_e64
; GCN: v_bcnt_u32_b32_e64
; GCN: v_bcnt_u32_b32_e64
; GCN: v_bcnt_u32_b32_e64
; GCN: v_bcnt_u32_b32_e64
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
define void @v_ctpop_v16i32(<16 x i32> addrspace(1)* noalias %out, <16 x i32> addrspace(1)* noalias %in) nounwind {
  %val = load <16 x i32>, <16 x i32> addrspace(1)* %in, align 32
  %ctpop = call <16 x i32> @llvm.ctpop.v16i32(<16 x i32> %val) nounwind readnone
  store <16 x i32> %ctpop, <16 x i32> addrspace(1)* %out, align 32
  ret void
}

; FUNC-LABEL: {{^}}v_ctpop_i32_add_inline_constant:
; GCN: buffer_load_dword [[VAL:v[0-9]+]],
; GCN: v_bcnt_u32_b32_e64 [[RESULT:v[0-9]+]], [[VAL]], 4
; GCN: buffer_store_dword [[RESULT]],
; GCN: s_endpgm

; EG: BCNT_INT
define void @v_ctpop_i32_add_inline_constant(i32 addrspace(1)* noalias %out, i32 addrspace(1)* noalias %in) nounwind {
  %val = load i32, i32 addrspace(1)* %in, align 4
  %ctpop = call i32 @llvm.ctpop.i32(i32 %val) nounwind readnone
  %add = add i32 %ctpop, 4
  store i32 %add, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v_ctpop_i32_add_inline_constant_inv:
; GCN: buffer_load_dword [[VAL:v[0-9]+]],
; GCN: v_bcnt_u32_b32_e64 [[RESULT:v[0-9]+]], [[VAL]], 4
; GCN: buffer_store_dword [[RESULT]],
; GCN: s_endpgm

; EG: BCNT_INT
define void @v_ctpop_i32_add_inline_constant_inv(i32 addrspace(1)* noalias %out, i32 addrspace(1)* noalias %in) nounwind {
  %val = load i32, i32 addrspace(1)* %in, align 4
  %ctpop = call i32 @llvm.ctpop.i32(i32 %val) nounwind readnone
  %add = add i32 4, %ctpop
  store i32 %add, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v_ctpop_i32_add_literal:
; GCN-DAG: buffer_load_dword [[VAL:v[0-9]+]],
; GCN-DAG: v_mov_b32_e32 [[LIT:v[0-9]+]], 0x1869f
; SI: v_bcnt_u32_b32_e32 [[RESULT:v[0-9]+]], [[VAL]], [[LIT]]
; VI: v_bcnt_u32_b32_e64 [[RESULT:v[0-9]+]], [[VAL]], [[LIT]]
; GCN: buffer_store_dword [[RESULT]],
; GCN: s_endpgm
define void @v_ctpop_i32_add_literal(i32 addrspace(1)* noalias %out, i32 addrspace(1)* noalias %in) nounwind {
  %val = load i32, i32 addrspace(1)* %in, align 4
  %ctpop = call i32 @llvm.ctpop.i32(i32 %val) nounwind readnone
  %add = add i32 %ctpop, 99999
  store i32 %add, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v_ctpop_i32_add_var:
; GCN-DAG: buffer_load_dword [[VAL:v[0-9]+]],
; GCN-DAG: s_load_dword [[VAR:s[0-9]+]],
; GCN: v_bcnt_u32_b32_e64 [[RESULT:v[0-9]+]], [[VAL]], [[VAR]]
; GCN: buffer_store_dword [[RESULT]],
; GCN: s_endpgm

; EG: BCNT_INT
define void @v_ctpop_i32_add_var(i32 addrspace(1)* noalias %out, i32 addrspace(1)* noalias %in, i32 %const) nounwind {
  %val = load i32, i32 addrspace(1)* %in, align 4
  %ctpop = call i32 @llvm.ctpop.i32(i32 %val) nounwind readnone
  %add = add i32 %ctpop, %const
  store i32 %add, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v_ctpop_i32_add_var_inv:
; GCN-DAG: buffer_load_dword [[VAL:v[0-9]+]],
; GCN-DAG: s_load_dword [[VAR:s[0-9]+]],
; GCN: v_bcnt_u32_b32_e64 [[RESULT:v[0-9]+]], [[VAL]], [[VAR]]
; GCN: buffer_store_dword [[RESULT]],
; GCN: s_endpgm

; EG: BCNT_INT
define void @v_ctpop_i32_add_var_inv(i32 addrspace(1)* noalias %out, i32 addrspace(1)* noalias %in, i32 %const) nounwind {
  %val = load i32, i32 addrspace(1)* %in, align 4
  %ctpop = call i32 @llvm.ctpop.i32(i32 %val) nounwind readnone
  %add = add i32 %const, %ctpop
  store i32 %add, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v_ctpop_i32_add_vvar_inv:
; GCN-DAG: buffer_load_dword [[VAL:v[0-9]+]], off, s[{{[0-9]+:[0-9]+}}], {{0$}}
; GCN-DAG: buffer_load_dword [[VAR:v[0-9]+]], off, s[{{[0-9]+:[0-9]+}}], 0 offset:16
; SI: v_bcnt_u32_b32_e32 [[RESULT:v[0-9]+]], [[VAL]], [[VAR]]
; VI: v_bcnt_u32_b32_e64 [[RESULT:v[0-9]+]], [[VAL]], [[VAR]]
; GCN: buffer_store_dword [[RESULT]],
; GCN: s_endpgm

; EG: BCNT_INT
define void @v_ctpop_i32_add_vvar_inv(i32 addrspace(1)* noalias %out, i32 addrspace(1)* noalias %in, i32 addrspace(1)* noalias %constptr) nounwind {
  %val = load i32, i32 addrspace(1)* %in, align 4
  %ctpop = call i32 @llvm.ctpop.i32(i32 %val) nounwind readnone
  %gep = getelementptr i32, i32 addrspace(1)* %constptr, i32 4
  %const = load i32, i32 addrspace(1)* %gep, align 4
  %add = add i32 %const, %ctpop
  store i32 %add, i32 addrspace(1)* %out, align 4
  ret void
}

; FIXME: We currently disallow SALU instructions in all branches,
; but there are some cases when the should be allowed.

; FUNC-LABEL: {{^}}ctpop_i32_in_br:
; SI: s_load_dword [[VAL:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0xd
; VI: s_load_dword [[VAL:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0x34
; GCN: s_bcnt1_i32_b32  [[SRESULT:s[0-9]+]], [[VAL]]
; GCN: v_mov_b32_e32 [[RESULT]], [[SRESULT]]
; GCN: buffer_store_dword [[RESULT]],
; GCN: s_endpgm
; EG: BCNT_INT
define void @ctpop_i32_in_br(i32 addrspace(1)* %out, i32 addrspace(1)* %in, i32 %ctpop_arg, i32 %cond) {
entry:
  %tmp0 = icmp eq i32 %cond, 0
  br i1 %tmp0, label %if, label %else

if:
  %tmp2 = call i32 @llvm.ctpop.i32(i32 %ctpop_arg)
  br label %endif

else:
  %tmp3 = getelementptr i32, i32 addrspace(1)* %in, i32 1
  %tmp4 = load i32, i32 addrspace(1)* %tmp3
  br label %endif

endif:
  %tmp5 = phi i32 [%tmp2, %if], [%tmp4, %else]
  store i32 %tmp5, i32 addrspace(1)* %out
  ret void
}
