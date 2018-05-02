; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX9 %s
; RUN: llc -march=amdgcn -mcpu=fiji  -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,VI %s

; GCN-LABEL: {{^}}reduction_half4:
; GFX9:      v_pk_add_f16 [[ADD:v[0-9]+]], [[ADD:v[0-9]+]], v{{[0-9]+}}{{$}}
; GFX9-NEXT: v_add_f16_sdwa [[ADD]], [[ADD]], [[ADD]] dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1

; VI:      v_add_f16_e32
; VI-NEXT: v_add_f16_e32
; VI-NEXT: v_add_f16_e32
define half @reduction_half4(<4 x half> %vec4) {
entry:
  %rdx.shuf = shufflevector <4 x half> %vec4, <4 x half> undef, <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
  %bin.rdx = fadd fast <4 x half> %vec4, %rdx.shuf
  %rdx.shuf1 = shufflevector <4 x half> %bin.rdx, <4 x half> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  %bin.rdx2 = fadd fast <4 x half> %bin.rdx, %rdx.shuf1
  %res = extractelement <4 x half> %bin.rdx2, i32 0
  ret half %res
}

; GCN-LABEL: {{^}}reduction_v4i16:
; GFX9:      v_pk_add_u16 [[ADD:v[0-9]+]], [[ADD:v[0-9]+]], v{{[0-9]+}}{{$}}
; GFX9-NEXT: v_add_u16_sdwa [[ADD]], [[ADD]], [[ADD]] dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1

; VI:      v_add_u16_e32
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
; GFX9:      v_pk_add_f16 [[ADD1:v[0-9]+]], [[ADD1:v[0-9]+]], v{{[0-9]+}}{{$}}
; GFX9-NEXT: v_pk_add_f16 [[ADD:v[0-9]+]], [[ADD:v[0-9]+]], v{{[0-9]+}}{{$}}
; GFX9-NEXT: v_pk_add_f16 [[ADD:v[0-9]+]], [[ADD]], [[ADD1]]{{$}}
; GFX9-NEXT: v_add_f16_sdwa [[ADD]], [[ADD]], [[ADD]] dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1

; VI:      v_add_f16_e32
; VI-NEXT: v_add_f16_e32
; VI-NEXT: v_add_f16_e32
; VI-NEXT: v_add_f16_e32
; VI-NEXT: v_add_f16_e32
; VI-NEXT: v_add_f16_e32
; VI-NEXT: v_add_f16_e32

define half @reduction_half8(<8 x half> %vec8) {
entry:
  %rdx.shuf = shufflevector <8 x half> %vec8, <8 x half> undef, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx = fadd fast <8 x half> %vec8, %rdx.shuf
  %rdx.shuf1 = shufflevector <8 x half> %bin.rdx, <8 x half> undef, <8 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx2 = fadd fast <8 x half> %bin.rdx, %rdx.shuf1
  %rdx.shuf3 = shufflevector <8 x half> %bin.rdx2, <8 x half> undef, <8 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx4 = fadd fast <8 x half> %bin.rdx2, %rdx.shuf3
  %res = extractelement <8 x half> %bin.rdx4, i32 0
  ret half %res
}

; GCN-LABEL: {{^}}reduction_v8i16:
; GFX9:      v_pk_add_u16 [[ADD1]], [[ADD1:v[0-9]+]], v{{[0-9]+}}{{$}}
; GFX9-NEXT: v_pk_add_u16 [[ADD]], [[ADD]], v{{[0-9]+}}{{$}}
; GFX9-NEXT: v_pk_add_u16 [[ADD]], [[ADD]], [[ADD1]]{{$}}
; GFX9-NEXT: v_add_u16_sdwa [[ADD]], [[ADD]], [[ADD]] dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1

; VI:      v_add_u16_e32
; VI-NEXT: v_add_u16_e32
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
; GFX9-NEXT: v_pk_add_f16 [[ADD1]], [[ADD1]], v{{[0-9]+}}{{$}}
; GFX9-NEXT: v_pk_add_f16 [[ADD]], [[ADD]], v{{[0-9]+}}{{$}}
; GFX9-NEXT: v_pk_add_f16 [[ADD]], [[ADD]], [[ADD1]]{{$}}
; GFX9-NEXT: v_add_f16_sdwa [[ADD]], [[ADD]], [[ADD]] dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1

; VI:      v_add_f16_e32
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
; VI-NEXT: v_add_f16_e32
; VI-NEXT: v_add_f16_e32
; VI-NEXT: v_add_f16_e32

define half @reduction_half16(<16 x half> %vec16) {
entry:
  %rdx.shuf = shufflevector <16 x half> %vec16, <16 x half> undef, <16 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx = fadd fast <16 x half> %vec16, %rdx.shuf
  %rdx.shuf1 = shufflevector <16 x half> %bin.rdx, <16 x half> undef, <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx2 = fadd fast <16 x half> %bin.rdx, %rdx.shuf1
  %rdx.shuf3 = shufflevector <16 x half> %bin.rdx2, <16 x half> undef, <16 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx4 = fadd fast <16 x half> %bin.rdx2, %rdx.shuf3
  %rdx.shuf5 = shufflevector <16 x half> %bin.rdx4, <16 x half> undef, <16 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx6 = fadd fast <16 x half> %bin.rdx4, %rdx.shuf5
  %res = extractelement <16 x half> %bin.rdx6, i32 0
  ret half %res
}