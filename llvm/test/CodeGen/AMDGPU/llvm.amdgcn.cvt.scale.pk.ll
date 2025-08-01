; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py UTC_ARGS: --version 4
; RUN: llc -global-isel=0 -mtriple=amdgcn -mcpu=gfx1250 < %s | FileCheck -check-prefixes=GFX1250,GFX1250-SDAG %s
; RUN: llc -global-isel=1 -global-isel-abort=2 -mtriple=amdgcn -mcpu=gfx1250 < %s | FileCheck -check-prefixes=GFX1250,GFX1250-GISEL %s

declare <8 x half> @llvm.amdgcn.cvt.scale.pk8.f16.fp8(<2 x i32> %src, i32 %scale, i32 %scale_sel)
declare <8 x bfloat> @llvm.amdgcn.cvt.scale.pk8.bf16.fp8(<2 x i32> %src, i32 %scale, i32 %scale_sel)
declare <8 x half> @llvm.amdgcn.cvt.scale.pk8.f16.bf8(<2 x i32> %src, i32 %scale, i32 %scale_sel)
declare <8 x bfloat> @llvm.amdgcn.cvt.scale.pk8.bf16.bf8(<2 x i32> %src, i32 %scale, i32 %scale_sel)
declare <8 x half> @llvm.amdgcn.cvt.scale.pk8.f16.fp4(i32 %src, i32 %scale, i32 %scale_sel)
declare <8 x bfloat> @llvm.amdgcn.cvt.scale.pk8.bf16.fp4(i32 %src, i32 %scale, i32 %scale_sel)
declare <8 x float> @llvm.amdgcn.cvt.scale.pk8.f32.fp8(<2 x i32> %src, i32 %scale, i32 %scale_sel)
declare <8 x float> @llvm.amdgcn.cvt.scale.pk8.f32.bf8(<2 x i32> %src, i32 %scale, i32 %scale_sel)
declare <8 x float> @llvm.amdgcn.cvt.scale.pk8.f32.fp4(i32 %src, i32 %scale, i32 %scale_sel)
declare <16 x half> @llvm.amdgcn.cvt.scale.pk16.f16.fp6(<3 x i32> %src, i32 %scale, i32 %scale_sel)
declare <16 x bfloat> @llvm.amdgcn.cvt.scale.pk16.bf16.fp6(<3 x i32> %src, i32 %scale, i32 %scale_sel)
declare <16 x half> @llvm.amdgcn.cvt.scale.pk16.f16.bf6(<3 x i32> %src, i32 %scale, i32 %scale_sel)
declare <16 x bfloat> @llvm.amdgcn.cvt.scale.pk16.bf16.bf6(<3 x i32> %src, i32 %scale, i32 %scale_sel)
declare <16 x float> @llvm.amdgcn.cvt.scale.pk16.f32.fp6(<3 x i32> %src, i32 %scale, i32 %scale_sel)
declare <16 x float> @llvm.amdgcn.cvt.scale.pk16.f32.bf6(<3 x i32> %src, i32 %scale, i32 %scale_sel)

define amdgpu_ps void @test_cvt_scale_pk8_f16_fp8_vv(<2 x i32> %src, i32 %scale, ptr addrspace(1) %out) {
; GFX1250-SDAG-LABEL: test_cvt_scale_pk8_f16_fp8_vv:
; GFX1250-SDAG:       ; %bb.0:
; GFX1250-SDAG-NEXT:    v_dual_mov_b32 v9, v4 :: v_dual_mov_b32 v8, v3
; GFX1250-SDAG-NEXT:    v_cvt_scale_pk8_f16_fp8 v[4:7], v[0:1], v2 scale_sel:1
; GFX1250-SDAG-NEXT:    global_store_b128 v[8:9], v[4:7], off
; GFX1250-SDAG-NEXT:    s_endpgm
;
; GFX1250-GISEL-LABEL: test_cvt_scale_pk8_f16_fp8_vv:
; GFX1250-GISEL:       ; %bb.0:
; GFX1250-GISEL-NEXT:    v_dual_mov_b32 v8, v3 :: v_dual_mov_b32 v9, v4
; GFX1250-GISEL-NEXT:    v_cvt_scale_pk8_f16_fp8 v[4:7], v[0:1], v2 scale_sel:1
; GFX1250-GISEL-NEXT:    global_store_b128 v[8:9], v[4:7], off
; GFX1250-GISEL-NEXT:    s_endpgm
  %cvt = tail call <8 x half> @llvm.amdgcn.cvt.scale.pk8.f16.fp8(<2 x i32> %src, i32 %scale, i32 1)
  store <8 x half> %cvt, ptr addrspace(1) %out, align 8
  ret void
}

define amdgpu_ps void @test_cvt_scale_pk8_f16_bf8_vv(<2 x i32> %src, i32 %scale, ptr addrspace(1) %out) {
; GFX1250-SDAG-LABEL: test_cvt_scale_pk8_f16_bf8_vv:
; GFX1250-SDAG:       ; %bb.0:
; GFX1250-SDAG-NEXT:    v_dual_mov_b32 v9, v4 :: v_dual_mov_b32 v8, v3
; GFX1250-SDAG-NEXT:    v_cvt_scale_pk8_f16_bf8 v[4:7], v[0:1], v2
; GFX1250-SDAG-NEXT:    global_store_b128 v[8:9], v[4:7], off
; GFX1250-SDAG-NEXT:    s_endpgm
;
; GFX1250-GISEL-LABEL: test_cvt_scale_pk8_f16_bf8_vv:
; GFX1250-GISEL:       ; %bb.0:
; GFX1250-GISEL-NEXT:    v_dual_mov_b32 v8, v3 :: v_dual_mov_b32 v9, v4
; GFX1250-GISEL-NEXT:    v_cvt_scale_pk8_f16_bf8 v[4:7], v[0:1], v2
; GFX1250-GISEL-NEXT:    global_store_b128 v[8:9], v[4:7], off
; GFX1250-GISEL-NEXT:    s_endpgm
  %cvt = tail call <8 x half> @llvm.amdgcn.cvt.scale.pk8.f16.bf8(<2 x i32> %src, i32 %scale, i32 0)
  store <8 x half> %cvt, ptr addrspace(1) %out, align 8
  ret void
}

define amdgpu_ps void @test_cvt_scale_pk8_bf16_fp8_vv(<2 x i32> %src, i32 %scale, ptr addrspace(1) %out) {
; GFX1250-LABEL: test_cvt_scale_pk8_bf16_fp8_vv:
; GFX1250:       ; %bb.0:
; GFX1250-NEXT:    v_dual_mov_b32 v9, v4 :: v_dual_mov_b32 v8, v3
; GFX1250-NEXT:    v_cvt_scale_pk8_bf16_fp8 v[4:7], v[0:1], v2 scale_sel:1
; GFX1250-NEXT:    global_store_b128 v[8:9], v[4:7], off
; GFX1250-NEXT:    s_endpgm
  %cvt = tail call <8 x bfloat> @llvm.amdgcn.cvt.scale.pk8.bf16.fp8(<2 x i32> %src, i32 %scale, i32 1)
  store <8 x bfloat> %cvt, ptr addrspace(1) %out, align 8
  ret void
}

define amdgpu_ps void @test_cvt_scale_pk8_bf16_bf8_vv(<2 x i32> %src, i32 %scale, ptr addrspace(1) %out) {
; GFX1250-LABEL: test_cvt_scale_pk8_bf16_bf8_vv:
; GFX1250:       ; %bb.0:
; GFX1250-NEXT:    v_dual_mov_b32 v9, v4 :: v_dual_mov_b32 v8, v3
; GFX1250-NEXT:    v_cvt_scale_pk8_bf16_bf8 v[4:7], v[0:1], v2 scale_sel:2
; GFX1250-NEXT:    global_store_b128 v[8:9], v[4:7], off
; GFX1250-NEXT:    s_endpgm
  %cvt = tail call <8 x bfloat> @llvm.amdgcn.cvt.scale.pk8.bf16.bf8(<2 x i32> %src, i32 %scale, i32 2)
  store <8 x bfloat> %cvt, ptr addrspace(1) %out, align 8
  ret void
}

define amdgpu_ps void @test_cvt_scale_pk8_f16_fp4_vv(i32 %src, i32 %scale, ptr addrspace(1) %out) {
; GFX1250-LABEL: test_cvt_scale_pk8_f16_fp4_vv:
; GFX1250:       ; %bb.0:
; GFX1250-NEXT:    v_cvt_scale_pk8_f16_fp4 v[4:7], v0, v1 scale_sel:3
; GFX1250-NEXT:    global_store_b128 v[2:3], v[4:7], off
; GFX1250-NEXT:    s_endpgm
  %cvt = tail call <8 x half> @llvm.amdgcn.cvt.scale.pk8.f16.fp4(i32 %src, i32 %scale, i32 3)
  store <8 x half> %cvt, ptr addrspace(1) %out, align 16
  ret void
}

define amdgpu_ps void @test_cvt_scale_pk8_bf16_fp4_vv(i32 %src, i32 %scale, ptr addrspace(1) %out) {
; GFX1250-LABEL: test_cvt_scale_pk8_bf16_fp4_vv:
; GFX1250:       ; %bb.0:
; GFX1250-NEXT:    v_cvt_scale_pk8_bf16_fp4 v[4:7], v0, v1 scale_sel:4
; GFX1250-NEXT:    global_store_b128 v[2:3], v[4:7], off
; GFX1250-NEXT:    s_endpgm
  %cvt = tail call <8 x bfloat> @llvm.amdgcn.cvt.scale.pk8.bf16.fp4(i32 %src, i32 %scale, i32 4)
  store <8 x bfloat> %cvt, ptr addrspace(1) %out, align 16
  ret void
}

define amdgpu_ps void @test_cvt_scale_pk8_f32_fp8_vv(<2 x i32> %src, i32 %scale, ptr addrspace(1) %out) {
; GFX1250-SDAG-LABEL: test_cvt_scale_pk8_f32_fp8_vv:
; GFX1250-SDAG:       ; %bb.0:
; GFX1250-SDAG-NEXT:    v_dual_mov_b32 v13, v4 :: v_dual_mov_b32 v12, v3
; GFX1250-SDAG-NEXT:    v_cvt_scale_pk8_f32_fp8 v[4:11], v[0:1], v2 scale_sel:7
; GFX1250-SDAG-NEXT:    s_clause 0x1
; GFX1250-SDAG-NEXT:    global_store_b128 v[12:13], v[8:11], off offset:16
; GFX1250-SDAG-NEXT:    global_store_b128 v[12:13], v[4:7], off
; GFX1250-SDAG-NEXT:    s_endpgm
;
; GFX1250-GISEL-LABEL: test_cvt_scale_pk8_f32_fp8_vv:
; GFX1250-GISEL:       ; %bb.0:
; GFX1250-GISEL-NEXT:    v_dual_mov_b32 v12, v3 :: v_dual_mov_b32 v13, v4
; GFX1250-GISEL-NEXT:    v_cvt_scale_pk8_f32_fp8 v[4:11], v[0:1], v2 scale_sel:7
; GFX1250-GISEL-NEXT:    s_clause 0x1
; GFX1250-GISEL-NEXT:    global_store_b128 v[12:13], v[4:7], off
; GFX1250-GISEL-NEXT:    global_store_b128 v[12:13], v[8:11], off offset:16
; GFX1250-GISEL-NEXT:    s_endpgm
  %cvt = tail call <8 x float> @llvm.amdgcn.cvt.scale.pk8.f32.fp8(<2 x i32> %src, i32 %scale, i32 7)
  store <8 x float> %cvt, ptr addrspace(1) %out, align 16
  ret void
}

define amdgpu_ps void @test_cvt_scale_pk8_f32_bf8_vv(<2 x i32> %src, i32 %scale, ptr addrspace(1) %out) {
; GFX1250-SDAG-LABEL: test_cvt_scale_pk8_f32_bf8_vv:
; GFX1250-SDAG:       ; %bb.0:
; GFX1250-SDAG-NEXT:    v_dual_mov_b32 v13, v4 :: v_dual_mov_b32 v12, v3
; GFX1250-SDAG-NEXT:    v_cvt_scale_pk8_f32_bf8 v[4:11], v[0:1], v2
; GFX1250-SDAG-NEXT:    s_clause 0x1
; GFX1250-SDAG-NEXT:    global_store_b128 v[12:13], v[8:11], off offset:16
; GFX1250-SDAG-NEXT:    global_store_b128 v[12:13], v[4:7], off
; GFX1250-SDAG-NEXT:    s_endpgm
;
; GFX1250-GISEL-LABEL: test_cvt_scale_pk8_f32_bf8_vv:
; GFX1250-GISEL:       ; %bb.0:
; GFX1250-GISEL-NEXT:    v_dual_mov_b32 v12, v3 :: v_dual_mov_b32 v13, v4
; GFX1250-GISEL-NEXT:    v_cvt_scale_pk8_f32_bf8 v[4:11], v[0:1], v2
; GFX1250-GISEL-NEXT:    s_clause 0x1
; GFX1250-GISEL-NEXT:    global_store_b128 v[12:13], v[4:7], off
; GFX1250-GISEL-NEXT:    global_store_b128 v[12:13], v[8:11], off offset:16
; GFX1250-GISEL-NEXT:    s_endpgm
  %cvt = tail call <8 x float> @llvm.amdgcn.cvt.scale.pk8.f32.bf8(<2 x i32> %src, i32 %scale, i32 0)
  store <8 x float> %cvt, ptr addrspace(1) %out, align 16
  ret void
}

define amdgpu_ps void @test_cvt_scale_pk8_f32_fp4_vv(i32 %src, i32 %scale, ptr addrspace(1) %out) {
; GFX1250-SDAG-LABEL: test_cvt_scale_pk8_f32_fp4_vv:
; GFX1250-SDAG:       ; %bb.0:
; GFX1250-SDAG-NEXT:    v_cvt_scale_pk8_f32_fp4 v[4:11], v0, v1 scale_sel:1
; GFX1250-SDAG-NEXT:    s_clause 0x1
; GFX1250-SDAG-NEXT:    global_store_b128 v[2:3], v[8:11], off offset:16
; GFX1250-SDAG-NEXT:    global_store_b128 v[2:3], v[4:7], off
; GFX1250-SDAG-NEXT:    s_endpgm
;
; GFX1250-GISEL-LABEL: test_cvt_scale_pk8_f32_fp4_vv:
; GFX1250-GISEL:       ; %bb.0:
; GFX1250-GISEL-NEXT:    v_cvt_scale_pk8_f32_fp4 v[4:11], v0, v1 scale_sel:1
; GFX1250-GISEL-NEXT:    s_clause 0x1
; GFX1250-GISEL-NEXT:    global_store_b128 v[2:3], v[4:7], off
; GFX1250-GISEL-NEXT:    global_store_b128 v[2:3], v[8:11], off offset:16
; GFX1250-GISEL-NEXT:    s_endpgm
  %cvt = tail call <8 x float> @llvm.amdgcn.cvt.scale.pk8.f32.fp4(i32 %src, i32 %scale, i32 1)
  store <8 x float> %cvt, ptr addrspace(1) %out, align 32
  ret void
}

define amdgpu_ps void @test_cvt_scale_pk16_f16_fp6_vv(<3 x i32> %src, i32 %scale, ptr addrspace(1) %out) {
; GFX1250-SDAG-LABEL: test_cvt_scale_pk16_f16_fp6_vv:
; GFX1250-SDAG:       ; %bb.0:
; GFX1250-SDAG-NEXT:    v_cvt_scale_pk16_f16_fp6 v[6:13], v[0:2], v3
; GFX1250-SDAG-NEXT:    s_clause 0x1
; GFX1250-SDAG-NEXT:    global_store_b128 v[4:5], v[10:13], off offset:16
; GFX1250-SDAG-NEXT:    global_store_b128 v[4:5], v[6:9], off
; GFX1250-SDAG-NEXT:    s_endpgm
;
; GFX1250-GISEL-LABEL: test_cvt_scale_pk16_f16_fp6_vv:
; GFX1250-GISEL:       ; %bb.0:
; GFX1250-GISEL-NEXT:    v_cvt_scale_pk16_f16_fp6 v[6:13], v[0:2], v3
; GFX1250-GISEL-NEXT:    s_clause 0x1
; GFX1250-GISEL-NEXT:    global_store_b128 v[4:5], v[6:9], off
; GFX1250-GISEL-NEXT:    global_store_b128 v[4:5], v[10:13], off offset:16
; GFX1250-GISEL-NEXT:    s_endpgm
  %cvt = tail call <16 x half> @llvm.amdgcn.cvt.scale.pk16.f16.fp6(<3 x i32> %src, i32 %scale, i32 0)
  store <16 x half> %cvt, ptr addrspace(1) %out, align 8
  ret void
}

define amdgpu_ps void @test_cvt_scale_pk16_f16_fp6_sl(<3 x i32> inreg %src, ptr addrspace(1) %out) {
; GFX1250-SDAG-LABEL: test_cvt_scale_pk16_f16_fp6_sl:
; GFX1250-SDAG:       ; %bb.0:
; GFX1250-SDAG-NEXT:    v_dual_mov_b32 v10, s0 :: v_dual_mov_b32 v11, s1
; GFX1250-SDAG-NEXT:    v_mov_b32_e32 v12, s2
; GFX1250-SDAG-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX1250-SDAG-NEXT:    v_cvt_scale_pk16_f16_fp6 v[2:9], v[10:12], 0x64 scale_sel:1
; GFX1250-SDAG-NEXT:    s_clause 0x1
; GFX1250-SDAG-NEXT:    global_store_b128 v[0:1], v[6:9], off offset:16
; GFX1250-SDAG-NEXT:    global_store_b128 v[0:1], v[2:5], off
; GFX1250-SDAG-NEXT:    s_endpgm
;
; GFX1250-GISEL-LABEL: test_cvt_scale_pk16_f16_fp6_sl:
; GFX1250-GISEL:       ; %bb.0:
; GFX1250-GISEL-NEXT:    v_dual_mov_b32 v12, s2 :: v_dual_mov_b32 v11, s1
; GFX1250-GISEL-NEXT:    v_mov_b32_e32 v10, s0
; GFX1250-GISEL-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX1250-GISEL-NEXT:    v_cvt_scale_pk16_f16_fp6 v[2:9], v[10:12], 0x64 scale_sel:1
; GFX1250-GISEL-NEXT:    s_clause 0x1
; GFX1250-GISEL-NEXT:    global_store_b128 v[0:1], v[2:5], off
; GFX1250-GISEL-NEXT:    global_store_b128 v[0:1], v[6:9], off offset:16
; GFX1250-GISEL-NEXT:    s_endpgm
  %cvt = tail call <16 x half> @llvm.amdgcn.cvt.scale.pk16.f16.fp6(<3 x i32> %src, i32 100, i32 1)
  store <16 x half> %cvt, ptr addrspace(1) %out, align 8
  ret void
}

define amdgpu_ps void @test_cvt_scale_pk16_bf16_fp6_vv(<3 x i32> %src, i32 %scale, ptr addrspace(1) %out) {
; GFX1250-LABEL: test_cvt_scale_pk16_bf16_fp6_vv:
; GFX1250:       ; %bb.0:
; GFX1250-NEXT:    v_cvt_scale_pk16_bf16_fp6 v[6:13], v[0:2], v3 scale_sel:2
; GFX1250-NEXT:    s_clause 0x1
; GFX1250-NEXT:    global_store_b128 v[4:5], v[10:13], off offset:16
; GFX1250-NEXT:    global_store_b128 v[4:5], v[6:9], off
; GFX1250-NEXT:    s_endpgm
  %cvt = tail call <16 x bfloat> @llvm.amdgcn.cvt.scale.pk16.bf16.fp6(<3 x i32> %src, i32 %scale, i32 2)
  store <16 x bfloat> %cvt, ptr addrspace(1) %out, align 8
  ret void
}

define amdgpu_ps void @test_cvt_scale_pk16_bf16_fp6_sl(<3 x i32> inreg %src, ptr addrspace(1) %out) {
; GFX1250-LABEL: test_cvt_scale_pk16_bf16_fp6_sl:
; GFX1250:       ; %bb.0:
; GFX1250-NEXT:    v_dual_mov_b32 v10, s0 :: v_dual_mov_b32 v11, s1
; GFX1250-NEXT:    v_mov_b32_e32 v12, s2
; GFX1250-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX1250-NEXT:    v_cvt_scale_pk16_bf16_fp6 v[2:9], v[10:12], 0x64 scale_sel:3
; GFX1250-NEXT:    s_clause 0x1
; GFX1250-NEXT:    global_store_b128 v[0:1], v[6:9], off offset:16
; GFX1250-NEXT:    global_store_b128 v[0:1], v[2:5], off
; GFX1250-NEXT:    s_endpgm
  %cvt = tail call <16 x bfloat> @llvm.amdgcn.cvt.scale.pk16.bf16.fp6(<3 x i32> %src, i32 100, i32 3)
  store <16 x bfloat> %cvt, ptr addrspace(1) %out, align 8
  ret void
}

define amdgpu_ps void @test_cvt_scale_pk16_f16_bf6_vv(<3 x i32> %src, i32 %scale, ptr addrspace(1) %out) {
; GFX1250-SDAG-LABEL: test_cvt_scale_pk16_f16_bf6_vv:
; GFX1250-SDAG:       ; %bb.0:
; GFX1250-SDAG-NEXT:    v_cvt_scale_pk16_f16_bf6 v[6:13], v[0:2], v3 scale_sel:4
; GFX1250-SDAG-NEXT:    s_clause 0x1
; GFX1250-SDAG-NEXT:    global_store_b128 v[4:5], v[10:13], off offset:16
; GFX1250-SDAG-NEXT:    global_store_b128 v[4:5], v[6:9], off
; GFX1250-SDAG-NEXT:    s_endpgm
;
; GFX1250-GISEL-LABEL: test_cvt_scale_pk16_f16_bf6_vv:
; GFX1250-GISEL:       ; %bb.0:
; GFX1250-GISEL-NEXT:    v_cvt_scale_pk16_f16_bf6 v[6:13], v[0:2], v3 scale_sel:4
; GFX1250-GISEL-NEXT:    s_clause 0x1
; GFX1250-GISEL-NEXT:    global_store_b128 v[4:5], v[6:9], off
; GFX1250-GISEL-NEXT:    global_store_b128 v[4:5], v[10:13], off offset:16
; GFX1250-GISEL-NEXT:    s_endpgm
  %cvt = tail call <16 x half> @llvm.amdgcn.cvt.scale.pk16.f16.bf6(<3 x i32> %src, i32 %scale, i32 4)
  store <16 x half> %cvt, ptr addrspace(1) %out, align 8
  ret void
}

define amdgpu_ps void @test_cvt_scale_pk16_f16_bf6_sl(<3 x i32> inreg %src, ptr addrspace(1) %out) {
; GFX1250-SDAG-LABEL: test_cvt_scale_pk16_f16_bf6_sl:
; GFX1250-SDAG:       ; %bb.0:
; GFX1250-SDAG-NEXT:    v_dual_mov_b32 v10, s0 :: v_dual_mov_b32 v11, s1
; GFX1250-SDAG-NEXT:    v_mov_b32_e32 v12, s2
; GFX1250-SDAG-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX1250-SDAG-NEXT:    v_cvt_scale_pk16_f16_bf6 v[2:9], v[10:12], 0x64 scale_sel:5
; GFX1250-SDAG-NEXT:    s_clause 0x1
; GFX1250-SDAG-NEXT:    global_store_b128 v[0:1], v[6:9], off offset:16
; GFX1250-SDAG-NEXT:    global_store_b128 v[0:1], v[2:5], off
; GFX1250-SDAG-NEXT:    s_endpgm
;
; GFX1250-GISEL-LABEL: test_cvt_scale_pk16_f16_bf6_sl:
; GFX1250-GISEL:       ; %bb.0:
; GFX1250-GISEL-NEXT:    v_dual_mov_b32 v12, s2 :: v_dual_mov_b32 v11, s1
; GFX1250-GISEL-NEXT:    v_mov_b32_e32 v10, s0
; GFX1250-GISEL-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX1250-GISEL-NEXT:    v_cvt_scale_pk16_f16_bf6 v[2:9], v[10:12], 0x64 scale_sel:5
; GFX1250-GISEL-NEXT:    s_clause 0x1
; GFX1250-GISEL-NEXT:    global_store_b128 v[0:1], v[2:5], off
; GFX1250-GISEL-NEXT:    global_store_b128 v[0:1], v[6:9], off offset:16
; GFX1250-GISEL-NEXT:    s_endpgm
  %cvt = tail call <16 x half> @llvm.amdgcn.cvt.scale.pk16.f16.bf6(<3 x i32> %src, i32 100, i32 5)
  store <16 x half> %cvt, ptr addrspace(1) %out, align 8
  ret void
}

define amdgpu_ps void @test_cvt_scale_pk16_bf16_bf6_vv(<3 x i32> %src, i32 %scale, ptr addrspace(1) %out) {
; GFX1250-LABEL: test_cvt_scale_pk16_bf16_bf6_vv:
; GFX1250:       ; %bb.0:
; GFX1250-NEXT:    v_cvt_scale_pk16_bf16_bf6 v[6:13], v[0:2], v3 scale_sel:6
; GFX1250-NEXT:    s_clause 0x1
; GFX1250-NEXT:    global_store_b128 v[4:5], v[10:13], off offset:16
; GFX1250-NEXT:    global_store_b128 v[4:5], v[6:9], off
; GFX1250-NEXT:    s_endpgm
  %cvt = tail call <16 x bfloat> @llvm.amdgcn.cvt.scale.pk16.bf16.bf6(<3 x i32> %src, i32 %scale, i32 6)
  store <16 x bfloat> %cvt, ptr addrspace(1) %out, align 8
  ret void
}

define amdgpu_ps void @test_cvt_scale_pk16_bf16_bf6_sl(<3 x i32> inreg %src, ptr addrspace(1) %out) {
; GFX1250-LABEL: test_cvt_scale_pk16_bf16_bf6_sl:
; GFX1250:       ; %bb.0:
; GFX1250-NEXT:    v_dual_mov_b32 v10, s0 :: v_dual_mov_b32 v11, s1
; GFX1250-NEXT:    v_mov_b32_e32 v12, s2
; GFX1250-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX1250-NEXT:    v_cvt_scale_pk16_bf16_bf6 v[2:9], v[10:12], 0x64 scale_sel:7
; GFX1250-NEXT:    s_clause 0x1
; GFX1250-NEXT:    global_store_b128 v[0:1], v[6:9], off offset:16
; GFX1250-NEXT:    global_store_b128 v[0:1], v[2:5], off
; GFX1250-NEXT:    s_endpgm
  %cvt = tail call <16 x bfloat> @llvm.amdgcn.cvt.scale.pk16.bf16.bf6(<3 x i32> %src, i32 100, i32 7)
  store <16 x bfloat> %cvt, ptr addrspace(1) %out, align 8
  ret void
}

define amdgpu_ps void @test_cvt_scale_pk16_f32_fp6_vv(<3 x i32> %src, i32 %scale, ptr addrspace(1) %out) {
; GFX1250-SDAG-LABEL: test_cvt_scale_pk16_f32_fp6_vv:
; GFX1250-SDAG:       ; %bb.0:
; GFX1250-SDAG-NEXT:    v_cvt_scale_pk16_f32_fp6 v[6:21], v[0:2], v3 scale_sel:5
; GFX1250-SDAG-NEXT:    s_clause 0x3
; GFX1250-SDAG-NEXT:    global_store_b128 v[4:5], v[18:21], off offset:48
; GFX1250-SDAG-NEXT:    global_store_b128 v[4:5], v[14:17], off offset:32
; GFX1250-SDAG-NEXT:    global_store_b128 v[4:5], v[10:13], off offset:16
; GFX1250-SDAG-NEXT:    global_store_b128 v[4:5], v[6:9], off
; GFX1250-SDAG-NEXT:    s_endpgm
;
; GFX1250-GISEL-LABEL: test_cvt_scale_pk16_f32_fp6_vv:
; GFX1250-GISEL:       ; %bb.0:
; GFX1250-GISEL-NEXT:    v_cvt_scale_pk16_f32_fp6 v[6:21], v[0:2], v3 scale_sel:5
; GFX1250-GISEL-NEXT:    s_clause 0x3
; GFX1250-GISEL-NEXT:    global_store_b128 v[4:5], v[6:9], off
; GFX1250-GISEL-NEXT:    global_store_b128 v[4:5], v[10:13], off offset:16
; GFX1250-GISEL-NEXT:    global_store_b128 v[4:5], v[14:17], off offset:32
; GFX1250-GISEL-NEXT:    global_store_b128 v[4:5], v[18:21], off offset:48
; GFX1250-GISEL-NEXT:    s_endpgm
  %cvt = tail call <16 x float> @llvm.amdgcn.cvt.scale.pk16.f32.fp6(<3 x i32> %src, i32 %scale, i32 5)
  store <16 x float> %cvt, ptr addrspace(1) %out, align 16
  ret void
}

define amdgpu_ps void @test_cvt_scale_pk16_f32_bf6_vv(<3 x i32> %src, i32 %scale, ptr addrspace(1) %out) {
; GFX1250-SDAG-LABEL: test_cvt_scale_pk16_f32_bf6_vv:
; GFX1250-SDAG:       ; %bb.0:
; GFX1250-SDAG-NEXT:    v_cvt_scale_pk16_f32_bf6 v[6:21], v[0:2], v3 scale_sel:6
; GFX1250-SDAG-NEXT:    s_clause 0x3
; GFX1250-SDAG-NEXT:    global_store_b128 v[4:5], v[18:21], off offset:48
; GFX1250-SDAG-NEXT:    global_store_b128 v[4:5], v[14:17], off offset:32
; GFX1250-SDAG-NEXT:    global_store_b128 v[4:5], v[10:13], off offset:16
; GFX1250-SDAG-NEXT:    global_store_b128 v[4:5], v[6:9], off
; GFX1250-SDAG-NEXT:    s_endpgm
;
; GFX1250-GISEL-LABEL: test_cvt_scale_pk16_f32_bf6_vv:
; GFX1250-GISEL:       ; %bb.0:
; GFX1250-GISEL-NEXT:    v_cvt_scale_pk16_f32_bf6 v[6:21], v[0:2], v3 scale_sel:6
; GFX1250-GISEL-NEXT:    s_clause 0x3
; GFX1250-GISEL-NEXT:    global_store_b128 v[4:5], v[6:9], off
; GFX1250-GISEL-NEXT:    global_store_b128 v[4:5], v[10:13], off offset:16
; GFX1250-GISEL-NEXT:    global_store_b128 v[4:5], v[14:17], off offset:32
; GFX1250-GISEL-NEXT:    global_store_b128 v[4:5], v[18:21], off offset:48
; GFX1250-GISEL-NEXT:    s_endpgm
  %cvt = tail call <16 x float> @llvm.amdgcn.cvt.scale.pk16.f32.bf6(<3 x i32> %src, i32 %scale, i32 6)
  store <16 x float> %cvt, ptr addrspace(1) %out, align 16
  ret void
}
