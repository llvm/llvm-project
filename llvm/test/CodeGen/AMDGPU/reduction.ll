; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX9 %s
; RUN: llc -march=amdgcn -mcpu=fiji  -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,VI %s

; GCN-LABEL: {{^}}reduction_fadd_v4f16:
; GFX9:      v_pk_add_f16 [[ADD:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}{{$}}
; GFX9-NEXT: v_add_f16_sdwa v{{[0-9]+}}, [[ADD]], [[ADD]] dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1

; VI:      v_add_f16_sdwa
; VI-NEXT: v_add_f16_e32
; VI-NEXT: v_add_f16_e32
define half @reduction_fadd_v4f16(<4 x half> %vec4) {
entry:
  %rdx.shuf = shufflevector <4 x half> %vec4, <4 x half> undef, <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
  %bin.rdx = fadd <4 x half> %vec4, %rdx.shuf
  %rdx.shuf1 = shufflevector <4 x half> %bin.rdx, <4 x half> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  %bin.rdx2 = fadd <4 x half> %bin.rdx, %rdx.shuf1
  %res = extractelement <4 x half> %bin.rdx2, i32 0
  ret half %res
}

; GCN-LABEL: {{^}}reduction_fsub_v4f16:
; GFX9: s_waitcnt
; GFX9-NEXT: v_pk_add_f16 [[ADD:v[0-9]+]], v0, v1 neg_lo:[0,1] neg_hi:[0,1]{{$}}
; GFX9-NEXT: v_sub_f16_sdwa v0, [[ADD]], [[ADD]] dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
; GFX9-NEXT: s_setpc_b64

; VI:      v_sub_f16_sdwa
; VI-NEXT: v_sub_f16_e32
; VI-NEXT: v_sub_f16_e32
; VI-NEXT: s_setpc_b64
define half @reduction_fsub_v4f16(<4 x half> %vec4) {
entry:
  %rdx.shuf = shufflevector <4 x half> %vec4, <4 x half> undef, <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
  %bin.rdx = fsub <4 x half> %vec4, %rdx.shuf
  %rdx.shuf1 = shufflevector <4 x half> %bin.rdx, <4 x half> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  %bin.rdx2 = fsub <4 x half> %bin.rdx, %rdx.shuf1
  %res = extractelement <4 x half> %bin.rdx2, i32 0
  ret half %res
}

; Make sure nsz is preserved when the operations are split.
; GCN-LABEL: {{^}}reduction_fsub_v4f16_preserve_fmf:
; GFX9: s_waitcnt
; GFX9-NEXT: v_pk_add_f16 v0, v0, v1 neg_lo:[0,1] neg_hi:[0,1]{{$}}
; GFX9-NEXT: v_sub_f16_sdwa v0, v0, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
; GFX9-NEXT: s_setpc_b64

; VI: s_waitcnt
; VI-NEXT: v_sub_f16_sdwa v2, v0, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; VI-NEXT: v_sub_f16_e32 v0, v1, v0
; VI-NEXT: v_add_f16_e32 v0, v2, v0
; VI-NEXT: s_setpc_b64
define half @reduction_fsub_v4f16_preserve_fmf(<4 x half> %vec4) {
entry:
  %rdx.shuf = shufflevector <4 x half> %vec4, <4 x half> undef, <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
  %bin.rdx = fsub nsz <4 x half> %vec4, %rdx.shuf
  %rdx.shuf1 = shufflevector <4 x half> %bin.rdx, <4 x half> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  %bin.rdx2 = fsub nsz <4 x half> %bin.rdx, %rdx.shuf1
  %res = extractelement <4 x half> %bin.rdx2, i32 0
  %neg.res = fsub half -0.0, %res
  ret half %neg.res
}

; GCN-LABEL: {{^}}reduction_fmul_half4:
; GFX9:      v_pk_mul_f16 [[MUL:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}{{$}}
; GFX9-NEXT: v_mul_f16_sdwa v{{[0-9]+}}, [[MUL]], [[MUL]] dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1

; VI:      v_mul_f16_sdwa
; VI-NEXT: v_mul_f16_e32
; VI-NEXT: v_mul_f16_e32
define half @reduction_fmul_half4(<4 x half> %vec4) {
entry:
  %rdx.shuf = shufflevector <4 x half> %vec4, <4 x half> undef, <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
  %bin.rdx = fmul <4 x half> %vec4, %rdx.shuf
  %rdx.shuf1 = shufflevector <4 x half> %bin.rdx, <4 x half> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  %bin.rdx2 = fmul <4 x half> %bin.rdx, %rdx.shuf1
  %res = extractelement <4 x half> %bin.rdx2, i32 0
  ret half %res
}

; GCN-LABEL: {{^}}reduction_v4i16:
; GFX9:      v_pk_add_u16 [[ADD:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}{{$}}
; GFX9-NEXT: v_add_u16_sdwa v{{[0-9]+}}, [[ADD]], [[ADD]] dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1

; VI:      v_add_u16_sdwa
; VI-NEXT: v_add_u16_e32
; VI-NEXT: v_add_u16_e32
define i16 @reduction_v4i16(<4 x i16> %vec4) {
entry:
  %rdx.shuf = shufflevector <4 x i16> %vec4, <4 x i16> undef, <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
  %bin.rdx = add <4 x i16> %vec4, %rdx.shuf
  %rdx.shuf1 = shufflevector <4 x i16> %bin.rdx, <4 x i16> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  %bin.rdx2 = add <4 x i16> %bin.rdx, %rdx.shuf1
  %res = extractelement <4 x i16> %bin.rdx2, i32 0
  ret i16 %res
}

; GCN-LABEL: {{^}}reduction_half8:
; GFX9:      v_pk_add_f16 [[ADD1:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}{{$}}
; GFX9-NEXT: v_pk_add_f16 [[ADD2:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}{{$}}
; GFX9-NEXT: v_pk_add_f16 [[ADD3:v[0-9]+]], [[ADD2]], [[ADD1]]{{$}}
; GFX9-NEXT: v_add_f16_sdwa v{{[0-9]+}}, [[ADD3]], [[ADD3]] dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1

; VI:      v_add_f16_sdwa
; VI-NEXT: v_add_f16_sdwa
; VI-NEXT: v_add_f16_e32
; VI-NEXT: v_add_f16_e32
; VI-NEXT: v_add_f16_e32
; VI-NEXT: v_add_f16_e32
; VI-NEXT: v_add_f16_e32

define half @reduction_half8(<8 x half> %vec8) {
entry:
  %rdx.shuf = shufflevector <8 x half> %vec8, <8 x half> undef, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx = fadd <8 x half> %vec8, %rdx.shuf
  %rdx.shuf1 = shufflevector <8 x half> %bin.rdx, <8 x half> undef, <8 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx2 = fadd <8 x half> %bin.rdx, %rdx.shuf1
  %rdx.shuf3 = shufflevector <8 x half> %bin.rdx2, <8 x half> undef, <8 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx4 = fadd <8 x half> %bin.rdx2, %rdx.shuf3
  %res = extractelement <8 x half> %bin.rdx4, i32 0
  ret half %res
}

; GCN-LABEL: {{^}}reduction_v8i16:
; GFX9:      v_pk_add_u16 [[ADD1:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}{{$}}
; GFX9-NEXT: v_pk_add_u16 [[ADD2:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}{{$}}
; GFX9-NEXT: v_pk_add_u16 [[ADD3:v[0-9]+]], [[ADD2]], [[ADD1]]{{$}}
; GFX9-NEXT: v_add_u16_sdwa v{{[0-9]+}}, [[ADD3]], [[ADD3]] dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1

; VI:      v_add_u16_sdwa
; VI-NEXT: v_add_u16_sdwa
; VI-NEXT: v_add_u16_e32
; VI-NEXT: v_add_u16_e32
; VI-NEXT: v_add_u16_e32
; VI-NEXT: v_add_u16_e32
; VI-NEXT: v_add_u16_e32

define i16 @reduction_v8i16(<8 x i16> %vec8) {
entry:
  %rdx.shuf = shufflevector <8 x i16> %vec8, <8 x i16> undef, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx = add <8 x i16> %vec8, %rdx.shuf
  %rdx.shuf1 = shufflevector <8 x i16> %bin.rdx, <8 x i16> undef, <8 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx2 = add <8 x i16> %bin.rdx, %rdx.shuf1
  %rdx.shuf3 = shufflevector <8 x i16> %bin.rdx2, <8 x i16> undef, <8 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx4 = add <8 x i16> %bin.rdx2, %rdx.shuf3
  %res = extractelement <8 x i16> %bin.rdx4, i32 0
  ret i16 %res
}

; GCN-LABEL: {{^}}reduction_half16:
; GFX9:      v_pk_add_f16 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}{{$}}
; GFX9-NEXT: v_pk_add_f16 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}{{$}}
; GFX9-NEXT: v_pk_add_f16 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}{{$}}
; GFX9:      v_pk_add_f16 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}{{$}}
; GFX9-NEXT: v_pk_add_f16 [[ADD1:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}{{$}}
; GFX9-NEXT: v_pk_add_f16 [[ADD2:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}{{$}}
; GFX9-NEXT: v_pk_add_f16 [[ADD3:v[0-9]+]], [[ADD2]], [[ADD1]]{{$}}
; GFX9-NEXT: v_add_f16_sdwa v{{[0-9]+}}, [[ADD3]], [[ADD3]] dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1

; VI:      v_add_f16_sdwa
; VI-NEXT: v_add_f16_sdwa
; VI-NEXT: v_add_f16_sdwa
; VI-NEXT: v_add_f16_sdwa
; VI-NEXT: v_add_f16_e32
; VI-NEXT: v_add_f16_e32
; VI-NEXT: v_add_f16_e32
; VI-NEXT: v_add_f16_e32
; VI-NEXT: v_add_f16_e32
; VI-NEXT: v_add_f16_e32
; VI-NEXT: v_add_f16_e32
; VI-NEXT: v_add_f16_e32
; VI-NEXT: v_add_f16_e32
; VI-NEXT: v_add_f16_e32
; VI-NEXT: v_add_f16_e32

define half @reduction_half16(<16 x half> %vec16) {
entry:
  %rdx.shuf = shufflevector <16 x half> %vec16, <16 x half> undef, <16 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx = fadd <16 x half> %vec16, %rdx.shuf
  %rdx.shuf1 = shufflevector <16 x half> %bin.rdx, <16 x half> undef, <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx2 = fadd <16 x half> %bin.rdx, %rdx.shuf1
  %rdx.shuf3 = shufflevector <16 x half> %bin.rdx2, <16 x half> undef, <16 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx4 = fadd <16 x half> %bin.rdx2, %rdx.shuf3
  %rdx.shuf5 = shufflevector <16 x half> %bin.rdx4, <16 x half> undef, <16 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx6 = fadd <16 x half> %bin.rdx4, %rdx.shuf5
  %res = extractelement <16 x half> %bin.rdx6, i32 0
  ret half %res
}

; GCN-LABEL: {{^}}reduction_min_v4i16:
; GFX9:      v_pk_min_u16 [[MIN:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}{{$}}
; GFX9-NEXT: v_min_u16_sdwa v{{[0-9]+}}, [[MIN]], [[MIN]] dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1

; VI:      v_min_u16_sdwa
; VI-NEXT: v_min_u16_e32
; VI-NEXT: v_min_u16_e32
define i16 @reduction_min_v4i16(<4 x i16> %vec4) {
entry:
  %rdx.shuf = shufflevector <4 x i16> %vec4, <4 x i16> undef, <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
  %rdx.minmax.cmp = icmp ult <4 x i16> %vec4, %rdx.shuf
  %rdx.minmax.select = select <4 x i1> %rdx.minmax.cmp, <4 x i16> %vec4, <4 x i16> %rdx.shuf
  %rdx.shuf1 = shufflevector <4 x i16> %rdx.minmax.select, <4 x i16> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp2 = icmp ult <4 x i16> %rdx.minmax.select, %rdx.shuf1
  %rdx.minmax.select3 = select <4 x i1> %rdx.minmax.cmp2, <4 x i16> %rdx.minmax.select, <4 x i16> %rdx.shuf1
  %res = extractelement <4 x i16> %rdx.minmax.select3, i32 0
  ret i16 %res
}

; GCN-LABEL: {{^}}reduction_umin_v8i16:
; GFX9:      v_pk_min_u16 [[MIN1:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}{{$}}
; GFX9-NEXT: v_pk_min_u16 [[MIN2:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}{{$}}
; GFX9-NEXT: v_pk_min_u16 [[MIN3:v[0-9]+]], [[MIN2]], [[MIN1]]{{$}}
; GFX9-NEXT: v_min_u16_sdwa v{{[0-9]+}}, [[MIN3]], [[MIN3]] dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1

; VI:      v_min_u16_sdwa
; VI-NEXT: v_min_u16_sdwa
; VI-NEXT: v_min_u16_e32
; VI-NEXT: v_min_u16_e32
; VI-NEXT: v_min_u16_e32
; VI-NEXT: v_min_u16_e32
; VI-NEXT: v_min_u16_e32
define i16 @reduction_umin_v8i16(<8 x i16> %vec8) {
entry:
  %rdx.shuf = shufflevector <8 x i16> %vec8, <8 x i16> undef, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp = icmp ult <8 x i16> %vec8, %rdx.shuf
  %rdx.minmax.select = select <8 x i1> %rdx.minmax.cmp, <8 x i16> %vec8, <8 x i16> %rdx.shuf
  %rdx.shuf1 = shufflevector <8 x i16> %rdx.minmax.select, <8 x i16> undef, <8 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp2 = icmp ult <8 x i16> %rdx.minmax.select, %rdx.shuf1
  %rdx.minmax.select3 = select <8 x i1> %rdx.minmax.cmp2, <8 x i16> %rdx.minmax.select, <8 x i16> %rdx.shuf1
  %rdx.shuf4 = shufflevector <8 x i16> %rdx.minmax.select3, <8 x i16> undef, <8 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp5 = icmp ult <8 x i16> %rdx.minmax.select3, %rdx.shuf4
  %rdx.minmax.select6 = select <8 x i1> %rdx.minmax.cmp5, <8 x i16> %rdx.minmax.select3, <8 x i16> %rdx.shuf4
  %res = extractelement <8 x i16> %rdx.minmax.select6, i32 0
  ret i16 %res
}

; Tests to make sure without slp the number of instructions are more.
; GCN-LABEL: {{^}}reduction_umin_v8i16_woslp:
; GFX9:      v_lshrrev_b32_e32
; GFX9-NEXT: v_min_u16_sdwa v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
; GFX9-NEXT: v_lshrrev_b32_e32
; GFX9-NEXT: v_min3_u16
; GFX9-NEXT: v_lshrrev_b32_e32
; GFX9-NEXT: v_min3_u16
; GFX9-NEXT: v_min3_u16
define i16 @reduction_umin_v8i16_woslp(<8 x i16> %vec8) {
entry:
  %elt0 = extractelement <8 x i16> %vec8, i64 0
  %elt1 = extractelement <8 x i16> %vec8, i64 1
  %elt2 = extractelement <8 x i16> %vec8, i64 2
  %elt3 = extractelement <8 x i16> %vec8, i64 3
  %elt4 = extractelement <8 x i16> %vec8, i64 4
  %elt5 = extractelement <8 x i16> %vec8, i64 5
  %elt6 = extractelement <8 x i16> %vec8, i64 6
  %elt7 = extractelement <8 x i16> %vec8, i64 7

  %cmp0 = icmp ult i16 %elt1, %elt0
  %min1 = select i1 %cmp0, i16 %elt1, i16 %elt0
  %cmp1 = icmp ult i16 %elt2, %min1
  %min2 = select i1 %cmp1, i16 %elt2, i16 %min1
  %cmp2 = icmp ult i16 %elt3, %min2
  %min3 = select i1 %cmp2, i16 %elt3, i16 %min2

  %cmp3 = icmp ult i16 %elt4, %min3
  %min4 = select i1 %cmp3, i16 %elt4, i16 %min3
  %cmp4 = icmp ult i16 %elt5, %min4
  %min5 = select i1 %cmp4, i16 %elt5, i16 %min4

  %cmp5 = icmp ult i16 %elt6, %min5
  %min6 = select i1 %cmp5, i16 %elt6, i16 %min5
  %cmp6 = icmp ult i16 %elt7, %min6
  %min7 = select i1 %cmp6, i16 %elt7, i16 %min6

  ret i16 %min7
}

; GCN-LABEL: {{^}}reduction_smin_v16i16:
; GFX9:        v_pk_min_i16 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}{{$}}
; GFX9-NEXT:   v_pk_min_i16 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}{{$}}
; GFX9-NEXT:   v_pk_min_i16 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}{{$}}
; GFX9-NEXT:   v_pk_min_i16 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}{{$}}
; GFX9-NEXT:   v_pk_min_i16 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}{{$}}
; GFX9-NEXT:   v_pk_min_i16 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}{{$}}
; GFX9-NEXT:   v_pk_min_i16 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}{{$}}
; GFX9-NEXT:   v_min_i16_sdwa v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1

; VI:      v_min_i16_sdwa
; VI-NEXT: v_min_i16_sdwa
; VI-NEXT: v_min_i16_sdwa
; VI-NEXT: v_min_i16_sdwa
; VI-NEXT: v_min_i16_e32
; VI-NEXT: v_min_i16_e32
; VI-NEXT: v_min_i16_e32
; VI-NEXT: v_min_i16_e32
; VI-NEXT: v_min_i16_e32
; VI-NEXT: v_min_i16_e32
; VI-NEXT: v_min_i16_e32
; VI-NEXT: v_min_i16_e32
; VI-NEXT: v_min_i16_e32
; VI-NEXT: v_min_i16_e32
; VI-NEXT: v_min_i16_e32
define i16 @reduction_smin_v16i16(<16 x i16> %vec16) {
entry:
  %rdx.shuf = shufflevector <16 x i16> %vec16, <16 x i16> undef, <16 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp = icmp slt <16 x i16> %vec16, %rdx.shuf
  %rdx.minmax.select = select <16 x i1> %rdx.minmax.cmp, <16 x i16> %vec16, <16 x i16> %rdx.shuf
  %rdx.shuf1 = shufflevector <16 x i16> %rdx.minmax.select, <16 x i16> undef, <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp2 = icmp slt <16 x i16> %rdx.minmax.select, %rdx.shuf1
  %rdx.minmax.select3 = select <16 x i1> %rdx.minmax.cmp2, <16 x i16> %rdx.minmax.select, <16 x i16> %rdx.shuf1
  %rdx.shuf4 = shufflevector <16 x i16> %rdx.minmax.select3, <16 x i16> undef, <16 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp5 = icmp slt <16 x i16> %rdx.minmax.select3, %rdx.shuf4
  %rdx.minmax.select6 = select <16 x i1> %rdx.minmax.cmp5, <16 x i16> %rdx.minmax.select3, <16 x i16> %rdx.shuf4
  %rdx.shuf7 = shufflevector <16 x i16> %rdx.minmax.select6, <16 x i16> undef, <16 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp8 = icmp slt <16 x i16> %rdx.minmax.select6, %rdx.shuf7
  %rdx.minmax.select9 = select <16 x i1> %rdx.minmax.cmp8, <16 x i16> %rdx.minmax.select6, <16 x i16> %rdx.shuf7
  %res = extractelement <16 x i16> %rdx.minmax.select9, i32 0
  ret i16 %res
}

; Tests to make sure without slp the number of instructions are more.
; GCN-LABEL: {{^}}reduction_smin_v16i16_woslp:
; GFX9:      v_lshrrev_b32_e32
; GFX9-NEXT: v_min_i16_sdwa v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
; GFX9-NEXT: v_lshrrev_b32_e32
; GFX9-NEXT: v_min3_i16
; GFX9-NEXT: v_lshrrev_b32_e32
; GFX9-NEXT: v_min3_i16
; GFX9-NEXT: v_lshrrev_b32_e32
; GFX9-NEXT: v_min3_i16
; GFX9-NEXT: v_lshrrev_b32_e32
; GFX9-NEXT: v_min3_i16
; GFX9-NEXT: v_lshrrev_b32_e32
; GFX9-NEXT: v_min3_i16
; GFX9-NEXT: v_lshrrev_b32_e32
; GFX9-NEXT: v_min3_i16
; GFX9-NEXT: v_min3_i16
define i16 @reduction_smin_v16i16_woslp(<16 x i16> %vec16) {
entry:
  %elt0 = extractelement <16 x i16> %vec16, i64 0
  %elt1 = extractelement <16 x i16> %vec16, i64 1
  %elt2 = extractelement <16 x i16> %vec16, i64 2
  %elt3 = extractelement <16 x i16> %vec16, i64 3
  %elt4 = extractelement <16 x i16> %vec16, i64 4
  %elt5 = extractelement <16 x i16> %vec16, i64 5
  %elt6 = extractelement <16 x i16> %vec16, i64 6
  %elt7 = extractelement <16 x i16> %vec16, i64 7

  %elt8 = extractelement <16 x i16> %vec16, i64 8
  %elt9 = extractelement <16 x i16> %vec16, i64 9
  %elt10 = extractelement <16 x i16> %vec16, i64 10
  %elt11 = extractelement <16 x i16> %vec16, i64 11
  %elt12 = extractelement <16 x i16> %vec16, i64 12
  %elt13 = extractelement <16 x i16> %vec16, i64 13
  %elt14 = extractelement <16 x i16> %vec16, i64 14
  %elt15 = extractelement <16 x i16> %vec16, i64 15

  %cmp0 = icmp slt i16 %elt1, %elt0
  %min1 = select i1 %cmp0, i16 %elt1, i16 %elt0
  %cmp1 = icmp slt i16 %elt2, %min1
  %min2 = select i1 %cmp1, i16 %elt2, i16 %min1
  %cmp2 = icmp slt i16 %elt3, %min2
  %min3 = select i1 %cmp2, i16 %elt3, i16 %min2

  %cmp3 = icmp slt i16 %elt4, %min3
  %min4 = select i1 %cmp3, i16 %elt4, i16 %min3
  %cmp4 = icmp slt i16 %elt5, %min4
  %min5 = select i1 %cmp4, i16 %elt5, i16 %min4

  %cmp5 = icmp slt i16 %elt6, %min5
  %min6 = select i1 %cmp5, i16 %elt6, i16 %min5
  %cmp6 = icmp slt i16 %elt7, %min6
  %min7 = select i1 %cmp6, i16 %elt7, i16 %min6

  %cmp7 = icmp slt i16 %elt8, %min7
  %min8 = select i1 %cmp7, i16 %elt8, i16 %min7
  %cmp8 = icmp slt i16 %elt9, %min8
  %min9 = select i1 %cmp8, i16 %elt9, i16 %min8

  %cmp9 = icmp slt i16 %elt10, %min9
  %min10 = select i1 %cmp9, i16 %elt10, i16 %min9
  %cmp10 = icmp slt i16 %elt11, %min10
  %min11 = select i1 %cmp10, i16 %elt11, i16 %min10

  %cmp11 = icmp slt i16 %elt12, %min11
  %min12 = select i1 %cmp11, i16 %elt12, i16 %min11
  %cmp12 = icmp slt i16 %elt13, %min12
  %min13 = select i1 %cmp12, i16 %elt13, i16 %min12

  %cmp13 = icmp slt i16 %elt14, %min13
  %min14 = select i1 %cmp13, i16 %elt14, i16 %min13
  %cmp14 = icmp slt i16 %elt15, %min14
  %min15 = select i1 %cmp14, i16 %elt15, i16 %min14


  ret i16 %min15
}

; GCN-LABEL: {{^}}reduction_umax_v4i16:
; GFX9:      v_pk_max_u16 [[MAX:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}{{$}}
; GFX9-NEXT: v_max_u16_sdwa v{{[0-9]+}}, [[MAX]], [[MAX]] dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1

; VI:      v_max_u16_sdwa
; VI-NEXT: v_max_u16_e32
; VI-NEXT: v_max_u16_e32
define i16 @reduction_umax_v4i16(<4 x i16> %vec4) {
entry:
  %rdx.shuf = shufflevector <4 x i16> %vec4, <4 x i16> undef, <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
  %rdx.minmax.cmp = icmp ugt <4 x i16> %vec4, %rdx.shuf
  %rdx.minmax.select = select <4 x i1> %rdx.minmax.cmp, <4 x i16> %vec4, <4 x i16> %rdx.shuf
  %rdx.shuf1 = shufflevector <4 x i16> %rdx.minmax.select, <4 x i16> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp2 = icmp ugt <4 x i16> %rdx.minmax.select, %rdx.shuf1
  %rdx.minmax.select3 = select <4 x i1> %rdx.minmax.cmp2, <4 x i16> %rdx.minmax.select, <4 x i16> %rdx.shuf1
  %res = extractelement <4 x i16> %rdx.minmax.select3, i32 0
  ret i16 %res
}

; GCN-LABEL: {{^}}reduction_smax_v4i16:
; GFX9:      v_pk_max_i16 [[MAX:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}{{$}}
; GFX9-NEXT: v_max_i16_sdwa v{{[0-9]+}}, [[MAX]], [[MAX]] dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1

; VI:      v_max_i16_sdwa
; VI-NEXT: v_max_i16_e32
; VI-NEXT: v_max_i16_e32
define i16 @reduction_smax_v4i16(<4 x i16> %vec4) #0 {
entry:
  %rdx.shuf = shufflevector <4 x i16> %vec4, <4 x i16> undef, <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
  %rdx.minmax.cmp = icmp sgt <4 x i16> %vec4, %rdx.shuf
  %rdx.minmax.select = select <4 x i1> %rdx.minmax.cmp, <4 x i16> %vec4, <4 x i16> %rdx.shuf
  %rdx.shuf1 = shufflevector <4 x i16> %rdx.minmax.select, <4 x i16> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp2 = icmp sgt <4 x i16> %rdx.minmax.select, %rdx.shuf1
  %rdx.minmax.select3 = select <4 x i1> %rdx.minmax.cmp2, <4 x i16> %rdx.minmax.select, <4 x i16> %rdx.shuf1
  %res = extractelement <4 x i16> %rdx.minmax.select3, i32 0
  ret i16 %res
}

; GCN-LABEL: {{^}}reduction_maxnum_v4f16:
; GFX9: s_waitcnt
; GFX9-NEXT: v_pk_max_f16 [[CANON1:v[0-9]+]], v1, v1
; GFX9-NEXT: v_pk_max_f16 [[CANON0:v[0-9]+]], v0, v0
; GFX9-NEXT: v_pk_max_f16 [[MAX:v[0-9]+]], [[CANON0]], [[CANON1]]{{$}}
; GFX9-NEXT: v_max_f16_sdwa v{{[0-9]+}}, [[MAX]], [[MAX]] dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1


; VI-DAG: v_max_f16_sdwa [[CANON1:v[0-9]+]], v0, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; VI-DAG: v_max_f16_sdwa [[CANON3:v[0-9]+]], v1, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; VI-DAG: v_max_f16_e32 [[CANON0:v[0-9]+]], v0, v0
; VI-DAG: v_max_f16_e32 [[CANON2:v[0-9]+]], v1, v1

; VI-DAG: v_max_f16_e32 [[MAX0:v[0-9]+]], [[CANON1]], [[CANON3]]
; VI-DAG: v_max_f16_e32 [[MAX1:v[0-9]+]], [[CANON0]], [[CANON2]]
; VI: v_max_f16_e32 v0, [[MAX1]], [[MAX0]]
define half @reduction_maxnum_v4f16(<4 x half> %vec4) {
entry:
  %rdx.shuf = shufflevector <4 x half> %vec4, <4 x half> undef, <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
  %rdx.minmax = call <4 x half> @llvm.maxnum.v4f16(<4 x half> %vec4, <4 x half> %rdx.shuf)
  %rdx.shuf1 = shufflevector <4 x half> %rdx.minmax, <4 x half> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  %rdx.minmax3 = call <4 x half> @llvm.maxnum.v4f16(<4 x half> %rdx.minmax, <4 x half> %rdx.shuf1)
  %res = extractelement <4 x half> %rdx.minmax3, i32 0
  ret half %res
}

; GCN-LABEL: {{^}}reduction_minnum_v4f16:
; GFX9: s_waitcnt
; GFX9-NEXT: v_pk_max_f16 [[CANON1:v[0-9]+]], v1, v1
; GFX9-NEXT: v_pk_max_f16 [[CANON0:v[0-9]+]], v0, v0
; GFX9-NEXT: v_pk_min_f16 [[MIN:v[0-9]+]], [[CANON0]], [[CANON1]]{{$}}
; GFX9-NEXT: v_min_f16_sdwa v{{[0-9]+}}, [[MIN]], [[MIN]] dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1

; VI-DAG: v_max_f16_sdwa [[CANON1:v[0-9]+]], v0, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; VI-DAG: v_max_f16_sdwa [[CANON3:v[0-9]+]], v1, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; VI-DAG: v_max_f16_e32 [[CANON0:v[0-9]+]], v0, v0
; VI-DAG: v_max_f16_e32 [[CANON2:v[0-9]+]], v1, v1

; VI-DAG: v_min_f16_e32 [[MAX0:v[0-9]+]], [[CANON1]], [[CANON3]]
; VI-DAG: v_min_f16_e32 [[MAX1:v[0-9]+]], [[CANON0]], [[CANON2]]
; VI: v_min_f16_e32 v0, [[MAX1]], [[MAX0]]
define half @reduction_minnum_v4f16(<4 x half> %vec4) {
entry:
  %rdx.shuf = shufflevector <4 x half> %vec4, <4 x half> undef, <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
  %rdx.minmax = call <4 x half> @llvm.minnum.v4f16(<4 x half> %vec4, <4 x half> %rdx.shuf)
  %rdx.shuf1 = shufflevector <4 x half> %rdx.minmax, <4 x half> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  %rdx.minmax3 = call <4 x half> @llvm.minnum.v4f16(<4 x half> %rdx.minmax, <4 x half> %rdx.shuf1)
  %res = extractelement <4 x half> %rdx.minmax3, i32 0
  ret half %res
}

; FIXME: Need to preserve fast math flags when fmaxnum matched
; directly from the IR to avoid unnecessary quieting.

; GCN-LABEL: {{^}}reduction_fast_max_pattern_v4f16:
; XGFX9:      v_pk_max_f16 [[MAX:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}{{$}}
; XGFX9-NEXT: v_max_f16_sdwa v{{[0-9]+}}, [[MAX]], [[MAX]] dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1

; XVI: s_waitcnt
; XVI-NEXT: v_max_f16_sdwa v2, v0, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; XVI-NEXT: v_max_f16_e32 v0, v0, v1
; XVI-NEXT: v_max_f16_e32 v0, v0, v2
; XVI-NEXT: s_setpc_b64

; GFX9: s_waitcnt
; GFX9-NEXT: v_pk_max_f16 [[CANON1:v[0-9]+]], v1, v1
; GFX9-NEXT: v_pk_max_f16 [[CANON0:v[0-9]+]], v0, v0
; GFX9-NEXT: v_pk_max_f16 [[MAX:v[0-9]+]], [[CANON0]], [[CANON1]]{{$}}
; GFX9-NEXT: v_max_f16_sdwa v{{[0-9]+}}, [[MAX]], [[MAX]] dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1

; VI-DAG: v_max_f16_sdwa [[CANON1:v[0-9]+]], v0, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; VI-DAG: v_max_f16_sdwa [[CANON3:v[0-9]+]], v1, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; VI-DAG: v_max_f16_e32 [[CANON0:v[0-9]+]], v0, v0
; VI-DAG: v_max_f16_e32 [[CANON2:v[0-9]+]], v1, v1

; VI-DAG: v_max_f16_e32 [[MAX0:v[0-9]+]], [[CANON1]], [[CANON3]]
; VI-DAG: v_max_f16_e32 [[MAX1:v[0-9]+]], [[CANON0]], [[CANON2]]
; VI: v_max_f16_e32 v0, [[MAX1]], [[MAX0]]
define half @reduction_fast_max_pattern_v4f16(<4 x half> %vec4) {
entry:
  %rdx.shuf = shufflevector <4 x half> %vec4, <4 x half> undef, <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
  %rdx.minmax.cmp = fcmp nnan nsz ogt <4 x half> %vec4, %rdx.shuf
  %rdx.minmax.select = select <4 x i1> %rdx.minmax.cmp, <4 x half> %vec4, <4 x half> %rdx.shuf
  %rdx.shuf1 = shufflevector <4 x half> %rdx.minmax.select, <4 x half> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp2 = fcmp nnan nsz ogt <4 x half> %rdx.minmax.select, %rdx.shuf1
  %rdx.minmax.select3 = select <4 x i1> %rdx.minmax.cmp2, <4 x half> %rdx.minmax.select, <4 x half> %rdx.shuf1
  %res = extractelement <4 x half> %rdx.minmax.select3, i32 0
  ret half %res
}

; FIXME: Need to preserve fast math flags when fmaxnum matched
; directly from the IR to avoid unnecessary quieting.

; GCN-LABEL: {{^}}reduction_fast_min_pattern_v4f16:
; XGFX9:      v_pk_min_f16 [[MIN:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}{{$}}
; XGFX9-NEXT: v_min_f16_sdwa v{{[0-9]+}}, [[MIN]], [[MIN]] dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1

; XVI: s_waitcnt
; XVI-NEXT: v_min_f16_sdwa v2, v0, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; XVI-NEXT: v_min_f16_e32 v0, v0, v1
; XVI-NEXT: v_min_f16_e32 v0, v0, v2
; XVI-NEXT: s_setpc_b64

; GFX9: s_waitcnt
; GFX9-NEXT: v_pk_max_f16 [[CANON1:v[0-9]+]], v1, v1
; GFX9-NEXT: v_pk_max_f16 [[CANON0:v[0-9]+]], v0, v0
; GFX9-NEXT: v_pk_min_f16 [[MIN:v[0-9]+]], [[CANON0]], [[CANON1]]{{$}}
; GFX9-NEXT: v_min_f16_sdwa v{{[0-9]+}}, [[MIN]], [[MIN]] dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1

; VI-DAG: v_max_f16_sdwa [[CANON1:v[0-9]+]], v0, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; VI-DAG: v_max_f16_sdwa [[CANON3:v[0-9]+]], v1, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; VI-DAG: v_max_f16_e32 [[CANON0:v[0-9]+]], v0, v0
; VI-DAG: v_max_f16_e32 [[CANON2:v[0-9]+]], v1, v1

; VI-DAG: v_min_f16_e32 [[MAX0:v[0-9]+]], [[CANON1]], [[CANON3]]
; VI-DAG: v_min_f16_e32 [[MAX1:v[0-9]+]], [[CANON0]], [[CANON2]]
; VI: v_min_f16_e32 v0, [[MAX1]], [[MAX0]]
define half @reduction_fast_min_pattern_v4f16(<4 x half> %vec4) {
entry:
  %rdx.shuf = shufflevector <4 x half> %vec4, <4 x half> undef, <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
  %rdx.minmax.cmp = fcmp nnan nsz olt <4 x half> %vec4, %rdx.shuf
  %rdx.minmax.select = select <4 x i1> %rdx.minmax.cmp, <4 x half> %vec4, <4 x half> %rdx.shuf
  %rdx.shuf1 = shufflevector <4 x half> %rdx.minmax.select, <4 x half> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp2 = fcmp nnan nsz olt <4 x half> %rdx.minmax.select, %rdx.shuf1
  %rdx.minmax.select3 = select <4 x i1> %rdx.minmax.cmp2, <4 x half> %rdx.minmax.select, <4 x half> %rdx.shuf1
  %res = extractelement <4 x half> %rdx.minmax.select3, i32 0
  ret half %res
}

declare <4 x half> @llvm.minnum.v4f16(<4 x half>, <4 x half>)
declare <4 x half> @llvm.maxnum.v4f16(<4 x half>, <4 x half>)
