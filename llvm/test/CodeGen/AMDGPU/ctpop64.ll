; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=GCN -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=VI -check-prefix=GCN -check-prefix=FUNC %s

declare i64 @llvm.ctpop.i64(i64) nounwind readnone
declare <2 x i64> @llvm.ctpop.v2i64(<2 x i64>) nounwind readnone
declare <4 x i64> @llvm.ctpop.v4i64(<4 x i64>) nounwind readnone
declare <8 x i64> @llvm.ctpop.v8i64(<8 x i64>) nounwind readnone
declare <16 x i64> @llvm.ctpop.v16i64(<16 x i64>) nounwind readnone

declare i65 @llvm.ctpop.i65(i65) nounwind readnone
declare i128 @llvm.ctpop.i128(i128) nounwind readnone

; FUNC-LABEL: {{^}}s_ctpop_i64:
; SI: s_load_dwordx2 [[SVAL:s\[[0-9]+:[0-9]+\]]], s{{\[[0-9]+:[0-9]+\]}}, 0xb
; VI: s_load_dwordx2 [[SVAL:s\[[0-9]+:[0-9]+\]]], s{{\[[0-9]+:[0-9]+\]}}, 0x2c
; GCN: s_bcnt1_i32_b64 [[SRESULT:s[0-9]+]], [[SVAL]]
; GCN: v_mov_b32_e32 [[VRESULT:v[0-9]+]], [[SRESULT]]
; GCN: buffer_store_dword [[VRESULT]],
; GCN: s_endpgm
define void @s_ctpop_i64(i32 addrspace(1)* noalias %out, i64 %val) nounwind {
  %ctpop = call i64 @llvm.ctpop.i64(i64 %val) nounwind readnone
  %truncctpop = trunc i64 %ctpop to i32
  store i32 %truncctpop, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v_ctpop_i64:
; GCN: buffer_load_dwordx2 v{{\[}}[[LOVAL:[0-9]+]]:[[HIVAL:[0-9]+]]{{\]}},
; GCN: v_bcnt_u32_b32_e64 [[MIDRESULT:v[0-9]+]], v[[LOVAL]], 0
; SI-NEXT: v_bcnt_u32_b32_e32 [[RESULT:v[0-9]+]], v[[HIVAL]], [[MIDRESULT]]
; VI-NEXT: v_bcnt_u32_b32_e64 [[RESULT:v[0-9]+]], v[[HIVAL]], [[MIDRESULT]]
; GCN: buffer_store_dword [[RESULT]],
; GCN: s_endpgm
define void @v_ctpop_i64(i32 addrspace(1)* noalias %out, i64 addrspace(1)* noalias %in) nounwind {
  %val = load i64, i64 addrspace(1)* %in, align 8
  %ctpop = call i64 @llvm.ctpop.i64(i64 %val) nounwind readnone
  %truncctpop = trunc i64 %ctpop to i32
  store i32 %truncctpop, i32 addrspace(1)* %out, align 4
  ret void
}

; FIXME: or 0 should be replaxed with copy
; FUNC-LABEL: {{^}}v_ctpop_i64_user:
; GCN: buffer_load_dwordx2 v{{\[}}[[LOVAL:[0-9]+]]:[[HIVAL:[0-9]+]]{{\]}},
; GCN: v_bcnt_u32_b32_e64 [[MIDRESULT:v[0-9]+]], v[[LOVAL]], 0
; SI-NEXT: v_bcnt_u32_b32_e32 [[RESULT:v[0-9]+]], v[[HIVAL]], [[MIDRESULT]]
; VI-NEXT: v_bcnt_u32_b32_e64 [[RESULT:v[0-9]+]], v[[HIVAL]], [[MIDRESULT]]
; GCN-DAG: v_or_b32_e32 v[[RESULT_LO:[0-9]+]], s{{[0-9]+}}, [[RESULT]]
; GCN-DAG: v_or_b32_e64 v[[RESULT_HI:[0-9]+]], 0, s{{[0-9]+}}
; GCN: buffer_store_dwordx2 v{{\[}}[[RESULT_LO]]:[[RESULT_HI]]{{\]}}
; GCN: s_endpgm
define void @v_ctpop_i64_user(i64 addrspace(1)* noalias %out, i64 addrspace(1)* noalias %in, i64 %s.val) nounwind {
  %val = load i64, i64 addrspace(1)* %in, align 8
  %ctpop = call i64 @llvm.ctpop.i64(i64 %val) nounwind readnone
  %or = or i64 %ctpop, %s.val
  store i64 %or, i64 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}s_ctpop_v2i64:
; GCN: s_bcnt1_i32_b64
; GCN: s_bcnt1_i32_b64
; GCN: s_endpgm
define void @s_ctpop_v2i64(<2 x i32> addrspace(1)* noalias %out, <2 x i64> %val) nounwind {
  %ctpop = call <2 x i64> @llvm.ctpop.v2i64(<2 x i64> %val) nounwind readnone
  %truncctpop = trunc <2 x i64> %ctpop to <2 x i32>
  store <2 x i32> %truncctpop, <2 x i32> addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}s_ctpop_v4i64:
; GCN: s_bcnt1_i32_b64
; GCN: s_bcnt1_i32_b64
; GCN: s_bcnt1_i32_b64
; GCN: s_bcnt1_i32_b64
; GCN: s_endpgm
define void @s_ctpop_v4i64(<4 x i32> addrspace(1)* noalias %out, <4 x i64> %val) nounwind {
  %ctpop = call <4 x i64> @llvm.ctpop.v4i64(<4 x i64> %val) nounwind readnone
  %truncctpop = trunc <4 x i64> %ctpop to <4 x i32>
  store <4 x i32> %truncctpop, <4 x i32> addrspace(1)* %out, align 16
  ret void
}

; FUNC-LABEL: {{^}}v_ctpop_v2i64:
; GCN: v_bcnt_u32_b32
; GCN: v_bcnt_u32_b32
; GCN: v_bcnt_u32_b32
; GCN: v_bcnt_u32_b32
; GCN: s_endpgm
define void @v_ctpop_v2i64(<2 x i32> addrspace(1)* noalias %out, <2 x i64> addrspace(1)* noalias %in) nounwind {
  %val = load <2 x i64>, <2 x i64> addrspace(1)* %in, align 16
  %ctpop = call <2 x i64> @llvm.ctpop.v2i64(<2 x i64> %val) nounwind readnone
  %truncctpop = trunc <2 x i64> %ctpop to <2 x i32>
  store <2 x i32> %truncctpop, <2 x i32> addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}v_ctpop_v4i64:
; GCN: v_bcnt_u32_b32
; GCN: v_bcnt_u32_b32
; GCN: v_bcnt_u32_b32
; GCN: v_bcnt_u32_b32
; GCN: v_bcnt_u32_b32
; GCN: v_bcnt_u32_b32
; GCN: v_bcnt_u32_b32
; GCN: v_bcnt_u32_b32
; GCN: s_endpgm
define void @v_ctpop_v4i64(<4 x i32> addrspace(1)* noalias %out, <4 x i64> addrspace(1)* noalias %in) nounwind {
  %val = load <4 x i64>, <4 x i64> addrspace(1)* %in, align 32
  %ctpop = call <4 x i64> @llvm.ctpop.v4i64(<4 x i64> %val) nounwind readnone
  %truncctpop = trunc <4 x i64> %ctpop to <4 x i32>
  store <4 x i32> %truncctpop, <4 x i32> addrspace(1)* %out, align 16
  ret void
}

; FUNC-LABEL: {{^}}ctpop_i64_in_br:
; SI-DAG: s_load_dwordx2 s{{\[}}[[LOVAL:[0-9]+]]:[[HIVAL:[0-9]+]]{{\]}}, s[{{[0-9]+:[0-9]+}}], 0xd
; VI-DAG: s_load_dwordx2 s{{\[}}[[LOVAL:[0-9]+]]:[[HIVAL:[0-9]+]]{{\]}}, s[{{[0-9]+:[0-9]+}}], 0x34
; GCN-DAG: s_bcnt1_i32_b64 [[RESULT:s[0-9]+]], {{s\[}}[[LOVAL]]:[[HIVAL]]{{\]}}
; GCN-DAG: s_mov_b32 [[ZERO:s[0-9]+]], 0
; GCN-DAG: v_mov_b32_e32 v[[VLO:[0-9]+]], [[RESULT]]
; GCN-DAG: v_mov_b32_e32 v[[VHI:[0-9]+]], [[ZERO]]
; GCN: buffer_store_dwordx2 {{v\[}}[[VLO]]:[[VHI]]{{\]}}
; GCN: s_endpgm
define void @ctpop_i64_in_br(i64 addrspace(1)* %out, i64 addrspace(1)* %in, i64 %ctpop_arg, i32 %cond) {
entry:
  %tmp0 = icmp eq i32 %cond, 0
  br i1 %tmp0, label %if, label %else

if:
  %tmp2 = call i64 @llvm.ctpop.i64(i64 %ctpop_arg)
  br label %endif

else:
  %tmp3 = getelementptr i64, i64 addrspace(1)* %in, i32 1
  %tmp4 = load i64, i64 addrspace(1)* %tmp3
  br label %endif

endif:
  %tmp5 = phi i64 [%tmp2, %if], [%tmp4, %else]
  store i64 %tmp5, i64 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}s_ctpop_i128:
; GCN: s_bcnt1_i32_b64 [[SRESULT0:s[0-9]+]],
; GCN: s_bcnt1_i32_b64 [[SRESULT1:s[0-9]+]],
; GCN: s_add_i32 s{{[0-9]+}}, [[SRESULT1]], [[SRESULT0]]
; GCN: s_endpgm
define void @s_ctpop_i128(i32 addrspace(1)* noalias %out, i128 %val) nounwind {
  %ctpop = call i128 @llvm.ctpop.i128(i128 %val) nounwind readnone
  %truncctpop = trunc i128 %ctpop to i32
  store i32 %truncctpop, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}s_ctpop_i65:
; GCN: s_bcnt1_i32_b64 [[REG0:s[0-9]+]],
; GCN: s_and_b32
; GCN: s_bcnt1_i32_b64 [[REG1:s[0-9]+]],
; GCN: s_add_i32 {{s[0-9]+}}, [[REG0]], [[REG1]]
; GCN: s_endpgm
define void @s_ctpop_i65(i32 addrspace(1)* noalias %out, i65 %val) nounwind {
  %ctpop = call i65 @llvm.ctpop.i65(i65 %val) nounwind readnone
  %truncctpop = trunc i65 %ctpop to i32
  store i32 %truncctpop, i32 addrspace(1)* %out, align 4
  ret void
}

; FIXME: Should not have extra add

; FUNC-LABEL: {{^}}v_ctpop_i128:
; GCN: buffer_load_dwordx4 v{{\[}}[[VAL0:[0-9]+]]:[[VAL3:[0-9]+]]{{\]}}, off, s{{\[[0-9]+:[0-9]+\]}}, 0{{$}}

; GCN-DAG: v_bcnt_u32_b32_e64 [[MIDRESULT0:v[0-9]+]], v{{[0-9]+}}, 0
; GCN-DAG: v_bcnt_u32_b32{{_e32|_e64}} [[MIDRESULT1:v[0-9]+]], v[[VAL3]], [[MIDRESULT0]]

; GCN-DAG: v_bcnt_u32_b32_e64 [[MIDRESULT2:v[0-9]+]], v[[VAL0]], 0
; GCN-DAG: v_bcnt_u32_b32{{_e32|_e64}} [[MIDRESULT3:v[0-9]+]], v{{[0-9]+}}, [[MIDRESULT2]]

; GCN: v_add_i32_e32 [[RESULT:v[0-9]+]], vcc, [[MIDRESULT1]], [[MIDRESULT2]]

; GCN: buffer_store_dword [[RESULT]],
; GCN: s_endpgm
define void @v_ctpop_i128(i32 addrspace(1)* noalias %out, i128 addrspace(1)* noalias %in) nounwind {
  %val = load i128, i128 addrspace(1)* %in, align 8
  %ctpop = call i128 @llvm.ctpop.i128(i128 %val) nounwind readnone
  %truncctpop = trunc i128 %ctpop to i32
  store i32 %truncctpop, i32 addrspace(1)* %out, align 4
  ret void
}
