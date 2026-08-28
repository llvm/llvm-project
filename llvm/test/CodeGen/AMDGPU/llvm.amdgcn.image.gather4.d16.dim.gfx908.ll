; RUN: llc < %s -mtriple=amdgpu9.08 | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}image_gather4_b_2d_v4f16_tfe_agpr:
; GCN: image_gather4_b v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], s[0:7], s[8:11] dmask:0x4 tfe d16{{$}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_ps void @image_gather4_b_2d_v4f16_tfe_agpr(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %bias, float %s, float %t) {
main_body:
  %r = call { <4 x half>, i32 } @llvm.amdgcn.image.gather4.b.2d.sl_v4f16i32s.f32.f32(i32 4, float %bias, float %s, float %t, <8 x i32> %rsrc, <4 x i32> %samp, i1 false, i32 1, i32 0)
  %tex = extractvalue { <4 x half>, i32 } %r, 0
  %tfe = extractvalue { <4 x half>, i32 } %r, 1
  call void asm sideeffect "; use $0 $1", "a,a"(<4 x half> %tex, i32 %tfe)
  ret void
}

declare { <4 x half>, i32 } @llvm.amdgcn.image.gather4.b.2d.sl_v4f16i32s.f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #0

attributes #0 = { nounwind readonly }
