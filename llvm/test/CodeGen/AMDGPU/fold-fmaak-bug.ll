; RUN: llc -march=amdgcn -mcpu=gfx1010 -verify-machineinstrs < %s | FileCheck %s -check-prefix GFX10

declare <2 x half> @llvm.amdgcn.cvt.pkrtz(float, float)
declare void @llvm.amdgcn.exp.compr.v2f16(i32 immarg, i32 immarg, <2 x half>, <2 x half>, i1 immarg, i1 immarg)

; FIXME: This instruction uses two different literal constants which is not
; allowed.
; GFX10-LABEL: _amdgpu_ps_main:
; GFX10: v_fmaak_f32 {{v[0-9]+}}, 0x40490fdb, {{v[0-9]+}}, 0xbfc90fdb
define amdgpu_ps void @_amdgpu_ps_main(float %arg) {
bb:
  %i = fmul reassoc nnan nsz arcp contract afn float %arg, 0x400921FB60000000
  %i1 = fadd reassoc nnan nsz arcp contract afn float %i, 0xBFF921FB60000000
  %i2 = fmul reassoc nnan nsz arcp contract afn float %i1, %arg
  br label %bb3

bb3:
  br label %bb4

bb4:
  %i5 = fadd reassoc nnan nsz arcp contract afn float 0x400921FB60000000, %i2
  br label %bb6

bb6:
  %i7 = call <2 x half> @llvm.amdgcn.cvt.pkrtz(float %i5, float 0.000000e+00)
  call void @llvm.amdgcn.exp.compr.v2f16(i32 0, i32 0, <2 x half> zeroinitializer, <2 x half> %i7, i1 false, i1 false)
  ret void
}
