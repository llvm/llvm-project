; RUN: not --crash llc -march=amdgcn -mcpu=gfx1010 -verify-machineinstrs < %s 2>&1 | FileCheck %s -check-prefix GFX10

declare <2 x half> @llvm.amdgcn.cvt.pkrtz(float, float)
declare void @llvm.amdgcn.exp.compr.v2f16(i32 immarg, i32 immarg, <2 x half>, <2 x half>, i1 immarg, i1 immarg)

; FIXME: This instruction uses two different literal constants which is not
; allowed.
; GFX10-LABEL: _amdgpu_ps_main:
; GFX10: Bad machine code: VOP2/VOP3 instruction uses more than one literal
; GFX10: instruction: %4:vgpr_32 = nnan nsz arcp contract afn reassoc nofpexcept V_FMAAK_F32 1078530011, %0:vgpr_32, -1077342245, implicit $mode, implicit $exec
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
