; RUN: llc -march=amdgcn -mcpu=gfx940 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

declare float @llvm.amdgcn.cvt.f32.bf8(i32, i32)
declare float @llvm.amdgcn.cvt.f32.fp8(i32, i32)
declare <2 x float> @llvm.amdgcn.cvt.pk.f32.bf8(i32, i1)
declare <2 x float> @llvm.amdgcn.cvt.pk.f32.fp8(i32, i1)
declare i32 @llvm.amdgcn.cvt.pk.bf8.f32(float, float, i32, i1)
declare i32 @llvm.amdgcn.cvt.pk.fp8.f32(float, float, i32, i1)
declare i32 @llvm.amdgcn.cvt.sr.bf8.f32(float, i32, i32, i32)
declare i32 @llvm.amdgcn.cvt.sr.fp8.f32(float, i32, i32, i32)

; GCN-LABEL: {{^}}test_cvt_f32_bf8_byte0:
; GCN: v_cvt_f32_bf8_e32 v0, v0{{$}}
define float @test_cvt_f32_bf8_byte0(i32 %a) {
  %ret = tail call float @llvm.amdgcn.cvt.f32.bf8(i32 %a, i32 0)
  ret float %ret
}

; GCN-LABEL: {{^}}test_cvt_f32_bf8_byte1:
; GCN: v_cvt_f32_bf8_sdwa v0, v0 src0_sel:BYTE_1
define float @test_cvt_f32_bf8_byte1(i32 %a) {
  %ret = tail call float @llvm.amdgcn.cvt.f32.bf8(i32 %a, i32 1)
  ret float %ret
}

; GCN-LABEL: {{^}}test_cvt_f32_bf8_byte2:
; GCN: v_cvt_f32_bf8_sdwa v0, v0 src0_sel:BYTE_2
define float @test_cvt_f32_bf8_byte2(i32 %a) {
  %ret = tail call float @llvm.amdgcn.cvt.f32.bf8(i32 %a, i32 2)
  ret float %ret
}

; GCN-LABEL: {{^}}test_cvt_f32_bf8_byte3:
; GCN: v_cvt_f32_bf8_sdwa v0, v0 src0_sel:BYTE_3
define float @test_cvt_f32_bf8_byte3(i32 %a) {
  %ret = tail call float @llvm.amdgcn.cvt.f32.bf8(i32 %a, i32 3)
  ret float %ret
}

; GCN-LABEL: {{^}}test_cvt_f32_fp8_byte0:
; GCN: v_cvt_f32_fp8_e32 v0, v0{{$}}
define float @test_cvt_f32_fp8_byte0(i32 %a) {
  %ret = tail call float @llvm.amdgcn.cvt.f32.fp8(i32 %a, i32 0)
  ret float %ret
}

; GCN-LABEL: {{^}}test_cvt_f32_fp8_byte1:
; GCN: v_cvt_f32_fp8_sdwa v0, v0 src0_sel:BYTE_1
define float @test_cvt_f32_fp8_byte1(i32 %a) {
  %ret = tail call float @llvm.amdgcn.cvt.f32.fp8(i32 %a, i32 1)
  ret float %ret
}

; GCN-LABEL: {{^}}test_cvt_f32_fp8_byte2:
; GCN: v_cvt_f32_fp8_sdwa v0, v0 src0_sel:BYTE_2
define float @test_cvt_f32_fp8_byte2(i32 %a) {
  %ret = tail call float @llvm.amdgcn.cvt.f32.fp8(i32 %a, i32 2)
  ret float %ret
}

; GCN-LABEL: {{^}}test_cvt_f32_fp8_byte3:
; GCN: v_cvt_f32_fp8_sdwa v0, v0 src0_sel:BYTE_3
define float @test_cvt_f32_fp8_byte3(i32 %a) {
  %ret = tail call float @llvm.amdgcn.cvt.f32.fp8(i32 %a, i32 3)
  ret float %ret
}

; GCN-LABEL: {{^}}test_cvt_pk_f32_bf8_word0:
; GCN: v_cvt_pk_f32_bf8_e32 v[0:1], v0{{$}}
define <2 x float> @test_cvt_pk_f32_bf8_word0(i32 %a) {
  %ret = tail call <2 x float> @llvm.amdgcn.cvt.pk.f32.bf8(i32 %a, i1 false)
  ret <2 x float> %ret
}

; GCN-LABEL: {{^}}test_cvt_pk_f32_bf8_word1:
; GCN: v_cvt_pk_f32_bf8_sdwa v[0:1], v0 src0_sel:WORD_1
define <2 x float> @test_cvt_pk_f32_bf8_word1(i32 %a) {
  %ret = tail call <2 x float> @llvm.amdgcn.cvt.pk.f32.bf8(i32 %a, i1 true)
  ret <2 x float> %ret
}

; GCN-LABEL: {{^}}test_cvt_pk_f32_fp8_word0:
; GCN: v_cvt_pk_f32_fp8_e32 v[0:1], v0{{$}}
define <2 x float> @test_cvt_pk_f32_fp8_word0(i32 %a) {
  %ret = tail call <2 x float> @llvm.amdgcn.cvt.pk.f32.fp8(i32 %a, i1 false)
  ret <2 x float> %ret
}

; GCN-LABEL: {{^}}test_cvt_pk_f32_fp8_word1:
; GCN: v_cvt_pk_f32_fp8_sdwa v[0:1], v0 src0_sel:WORD_1
define <2 x float> @test_cvt_pk_f32_fp8_word1(i32 %a) {
  %ret = tail call <2 x float> @llvm.amdgcn.cvt.pk.f32.fp8(i32 %a, i1 true)
  ret <2 x float> %ret
}

; GCN-LABEL: {{^}}test_cvt_pk_bf8_f32_word0:
; GCN: v_cvt_pk_bf8_f32 v2, v0, v1{{$}}
; GCN: v_mov_b32_e32 v0, v2
define i32 @test_cvt_pk_bf8_f32_word0(float %x, float %y, i32 %old) {
  %ret = tail call i32 @llvm.amdgcn.cvt.pk.bf8.f32(float %x, float %y, i32 %old, i1 false)
  ret i32 %ret
}

; GCN-LABEL: {{^}}test_cvt_pk_bf8_f32_word1:
; GCN: v_cvt_pk_bf8_f32 v2, v0, v1 op_sel:[0,0,1]
; GCN: v_mov_b32_e32 v0, v2
define i32 @test_cvt_pk_bf8_f32_word1(float %x, float %y, i32 %old) {
  %ret = tail call i32 @llvm.amdgcn.cvt.pk.bf8.f32(float %x, float %y, i32 %old, i1 true)
  ret i32 %ret
}

; GCN-LABEL: {{^}}test_cvt_pk_fp8_f32_word0:
; GCN: v_cvt_pk_fp8_f32 v2, v0, v1{{$}}
; GCN: v_mov_b32_e32 v0, v2
define i32 @test_cvt_pk_fp8_f32_word0(float %x, float %y, i32 %old) {
  %ret = tail call i32 @llvm.amdgcn.cvt.pk.fp8.f32(float %x, float %y, i32 %old, i1 false)
  ret i32 %ret
}

; GCN-LABEL: {{^}}test_cvt_pk_fp8_f32_word1:
; GCN: v_cvt_pk_fp8_f32 v2, v0, v1 op_sel:[0,0,1]
; GCN: v_mov_b32_e32 v0, v2
define i32 @test_cvt_pk_fp8_f32_word1(float %x, float %y, i32 %old) {
  %ret = tail call i32 @llvm.amdgcn.cvt.pk.fp8.f32(float %x, float %y, i32 %old, i1 true)
  ret i32 %ret
}

; GCN-LABEL: {{^}}test_cvt_sr_bf8_f32_byte0:
; GCN: v_cvt_sr_bf8_f32 v2, v0, v1{{$}}
; GCN: v_mov_b32_e32 v0, v2
define i32 @test_cvt_sr_bf8_f32_byte0(float %x, i32 %r, i32 %old) {
  %ret = tail call i32 @llvm.amdgcn.cvt.sr.bf8.f32(float %x, i32 %r, i32 %old, i32 0)
  ret i32 %ret
}

; GCN-LABEL: {{^}}test_cvt_sr_bf8_f32_byte1:
; GCN: v_cvt_sr_bf8_f32 v2, v0, v1 op_sel:[0,0,1,0]
; GCN: v_mov_b32_e32 v0, v2
define i32 @test_cvt_sr_bf8_f32_byte1(float %x, i32 %r, i32 %old) {
  %ret = tail call i32 @llvm.amdgcn.cvt.sr.bf8.f32(float %x, i32 %r, i32 %old, i32 1)
  ret i32 %ret
}

; GCN-LABEL: {{^}}test_cvt_sr_bf8_f32_byte2:
; GCN: v_cvt_sr_bf8_f32 v2, v0, v1 op_sel:[0,0,0,1]
; GCN: v_mov_b32_e32 v0, v2
define i32 @test_cvt_sr_bf8_f32_byte2(float %x, i32 %r, i32 %old) {
  %ret = tail call i32 @llvm.amdgcn.cvt.sr.bf8.f32(float %x, i32 %r, i32 %old, i32 2)
  ret i32 %ret
}

; GCN-LABEL: {{^}}test_cvt_sr_bf8_f32_byte3:
; GCN: v_cvt_sr_bf8_f32 v2, v0, v1 op_sel:[0,0,1,1]
; GCN: v_mov_b32_e32 v0, v2
define i32 @test_cvt_sr_bf8_f32_byte3(float %x, i32 %r, i32 %old) {
  %ret = tail call i32 @llvm.amdgcn.cvt.sr.bf8.f32(float %x, i32 %r, i32 %old, i32 3)
  ret i32 %ret
}

; GCN-LABEL: {{^}}test_cvt_sr_fp8_f32_byte0:
; GCN: v_cvt_sr_fp8_f32 v2, v0, v1{{$}}
; GCN: v_mov_b32_e32 v0, v2
define i32 @test_cvt_sr_fp8_f32_byte0(float %x, i32 %r, i32 %old) {
  %ret = tail call i32 @llvm.amdgcn.cvt.sr.fp8.f32(float %x, i32 %r, i32 %old, i32 0)
  ret i32 %ret
}

; GCN-LABEL: {{^}}test_cvt_sr_fp8_f32_byte1:
; GCN: v_cvt_sr_fp8_f32 v2, v0, v1 op_sel:[0,0,1,0]
; GCN: v_mov_b32_e32 v0, v2
define i32 @test_cvt_sr_fp8_f32_byte1(float %x, i32 %r, i32 %old) {
  %ret = tail call i32 @llvm.amdgcn.cvt.sr.fp8.f32(float %x, i32 %r, i32 %old, i32 1)
  ret i32 %ret
}

; GCN-LABEL: {{^}}test_cvt_sr_fp8_f32_byte2:
; GCN: v_cvt_sr_fp8_f32 v2, v0, v1 op_sel:[0,0,0,1]
; GCN: v_mov_b32_e32 v0, v2
define i32 @test_cvt_sr_fp8_f32_byte2(float %x, i32 %r, i32 %old) {
  %ret = tail call i32 @llvm.amdgcn.cvt.sr.fp8.f32(float %x, i32 %r, i32 %old, i32 2)
  ret i32 %ret
}

; GCN-LABEL: {{^}}test_cvt_sr_fp8_f32_byte3:
; GCN: v_cvt_sr_fp8_f32 v2, v0, v1 op_sel:[0,0,1,1]
; GCN: v_mov_b32_e32 v0, v2
define i32 @test_cvt_sr_fp8_f32_byte3(float %x, i32 %r, i32 %old) {
  %ret = tail call i32 @llvm.amdgcn.cvt.sr.fp8.f32(float %x, i32 %r, i32 %old, i32 3)
  ret i32 %ret
}
