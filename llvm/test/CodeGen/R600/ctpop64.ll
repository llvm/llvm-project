; RUN: llc -march=amdgcn -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

declare i64 @llvm.ctpop.i64(i64) nounwind readnone
declare <2 x i64> @llvm.ctpop.v2i64(<2 x i64>) nounwind readnone
declare <4 x i64> @llvm.ctpop.v4i64(<4 x i64>) nounwind readnone
declare <8 x i64> @llvm.ctpop.v8i64(<8 x i64>) nounwind readnone
declare <16 x i64> @llvm.ctpop.v16i64(<16 x i64>) nounwind readnone

; FUNC-LABEL: {{^}}s_ctpop_i64:
; SI: s_load_dwordx2 [[SVAL:s\[[0-9]+:[0-9]+\]]], s{{\[[0-9]+:[0-9]+\]}}, 0xb
; SI: s_bcnt1_i32_b64 [[SRESULT:s[0-9]+]], [[SVAL]]
; SI: v_mov_b32_e32 [[VRESULT:v[0-9]+]], [[SRESULT]]
; SI: buffer_store_dword [[VRESULT]],
; SI: s_endpgm
define void @s_ctpop_i64(i32 addrspace(1)* noalias %out, i64 %val) nounwind {
  %ctpop = call i64 @llvm.ctpop.i64(i64 %val) nounwind readnone
  %truncctpop = trunc i64 %ctpop to i32
  store i32 %truncctpop, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v_ctpop_i64:
; SI: buffer_load_dwordx2 v{{\[}}[[LOVAL:[0-9]+]]:[[HIVAL:[0-9]+]]{{\]}},
; SI: v_bcnt_u32_b32_e64 [[MIDRESULT:v[0-9]+]], v[[LOVAL]], 0
; SI-NEXT: v_bcnt_u32_b32_e32 [[RESULT:v[0-9]+]], v[[HIVAL]], [[MIDRESULT]]
; SI: buffer_store_dword [[RESULT]],
; SI: s_endpgm
define void @v_ctpop_i64(i32 addrspace(1)* noalias %out, i64 addrspace(1)* noalias %in) nounwind {
  %val = load i64 addrspace(1)* %in, align 8
  %ctpop = call i64 @llvm.ctpop.i64(i64 %val) nounwind readnone
  %truncctpop = trunc i64 %ctpop to i32
  store i32 %truncctpop, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}s_ctpop_v2i64:
; SI: s_bcnt1_i32_b64
; SI: s_bcnt1_i32_b64
; SI: s_endpgm
define void @s_ctpop_v2i64(<2 x i32> addrspace(1)* noalias %out, <2 x i64> %val) nounwind {
  %ctpop = call <2 x i64> @llvm.ctpop.v2i64(<2 x i64> %val) nounwind readnone
  %truncctpop = trunc <2 x i64> %ctpop to <2 x i32>
  store <2 x i32> %truncctpop, <2 x i32> addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}s_ctpop_v4i64:
; SI: s_bcnt1_i32_b64
; SI: s_bcnt1_i32_b64
; SI: s_bcnt1_i32_b64
; SI: s_bcnt1_i32_b64
; SI: s_endpgm
define void @s_ctpop_v4i64(<4 x i32> addrspace(1)* noalias %out, <4 x i64> %val) nounwind {
  %ctpop = call <4 x i64> @llvm.ctpop.v4i64(<4 x i64> %val) nounwind readnone
  %truncctpop = trunc <4 x i64> %ctpop to <4 x i32>
  store <4 x i32> %truncctpop, <4 x i32> addrspace(1)* %out, align 16
  ret void
}

; FUNC-LABEL: {{^}}v_ctpop_v2i64:
; SI: v_bcnt_u32_b32
; SI: v_bcnt_u32_b32
; SI: v_bcnt_u32_b32
; SI: v_bcnt_u32_b32
; SI: s_endpgm
define void @v_ctpop_v2i64(<2 x i32> addrspace(1)* noalias %out, <2 x i64> addrspace(1)* noalias %in) nounwind {
  %val = load <2 x i64> addrspace(1)* %in, align 16
  %ctpop = call <2 x i64> @llvm.ctpop.v2i64(<2 x i64> %val) nounwind readnone
  %truncctpop = trunc <2 x i64> %ctpop to <2 x i32>
  store <2 x i32> %truncctpop, <2 x i32> addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}v_ctpop_v4i64:
; SI: v_bcnt_u32_b32
; SI: v_bcnt_u32_b32
; SI: v_bcnt_u32_b32
; SI: v_bcnt_u32_b32
; SI: v_bcnt_u32_b32
; SI: v_bcnt_u32_b32
; SI: v_bcnt_u32_b32
; SI: v_bcnt_u32_b32
; SI: s_endpgm
define void @v_ctpop_v4i64(<4 x i32> addrspace(1)* noalias %out, <4 x i64> addrspace(1)* noalias %in) nounwind {
  %val = load <4 x i64> addrspace(1)* %in, align 32
  %ctpop = call <4 x i64> @llvm.ctpop.v4i64(<4 x i64> %val) nounwind readnone
  %truncctpop = trunc <4 x i64> %ctpop to <4 x i32>
  store <4 x i32> %truncctpop, <4 x i32> addrspace(1)* %out, align 16
  ret void
}

; FIXME: We currently disallow SALU instructions in all branches,
; but there are some cases when the should be allowed.

; FUNC-LABEL: {{^}}ctpop_i64_in_br:
; SI: s_load_dwordx2 s{{\[}}[[LOVAL:[0-9]+]]:[[HIVAL:[0-9]+]]{{\]}}, s[{{[0-9]+:[0-9]+}}], 0xd
; SI: s_bcnt1_i32_b64 [[RESULT:s[0-9]+]], {{s\[}}[[LOVAL]]:[[HIVAL]]{{\]}}
; SI: v_mov_b32_e32 v[[VLO:[0-9]+]], [[RESULT]]
; SI: v_mov_b32_e32 v[[VHI:[0-9]+]], s[[HIVAL]]
; SI: buffer_store_dwordx2 {{v\[}}[[VLO]]:[[VHI]]{{\]}}
; SI: s_endpgm
define void @ctpop_i64_in_br(i64 addrspace(1)* %out, i64 addrspace(1)* %in, i64 %ctpop_arg, i32 %cond) {
entry:
  %tmp0 = icmp eq i32 %cond, 0
  br i1 %tmp0, label %if, label %else

if:
  %tmp2 = call i64 @llvm.ctpop.i64(i64 %ctpop_arg)
  br label %endif

else:
  %tmp3 = getelementptr i64 addrspace(1)* %in, i32 1
  %tmp4 = load i64 addrspace(1)* %tmp3
  br label %endif

endif:
  %tmp5 = phi i64 [%tmp2, %if], [%tmp4, %else]
  store i64 %tmp5, i64 addrspace(1)* %out
  ret void
}
