; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -enable-fp32-recip-newton-raphson < %s | FileCheck -check-prefix=GFX9 %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx1030 -enable-fp32-recip-newton-raphson < %s | FileCheck -check-prefix=GFX10 %s

; Test that -enable-fp32-recip-newton-raphson lowers 1.0f / x to a
; Newton-Raphson fast path (v_rcp_f32 + 2x v_fma_f32) guarded by an
; is-normal class check, falling back to the full division for
; denormals / inf / nan / zero.
;
; The instruction scheduler interleaves the fast and slow paths, so we
; use CHECK-DAG for most instructions and only enforce that the final
; v_cndmask_b32 (the SELECT) comes after all of them.

; GFX9-LABEL: {{^}}test_fdiv_recip_f32:
; Fast path (Newton-Raphson):
; GFX9-DAG: v_rcp_f32_e32
; GFX9-DAG: v_fma_f32 {{.*}}, -1.0
; Normal-class check and select:
; GFX9-DAG: v_cmp_class_f32
; Slow path (full division sequence):
; GFX9-DAG: v_div_scale_f32
; GFX9-DAG: v_div_fixup_f32
; Final select between fast and slow results:
; GFX9: v_cndmask_b32
;
; GFX10-LABEL: {{^}}test_fdiv_recip_f32:
; GFX10-DAG: v_rcp_f32_e32
; GFX10-DAG: v_fma_f32 {{.*}}, -1.0
; GFX10-DAG: v_cmp_class_f32
; GFX10-DAG: v_div_scale_f32
; GFX10-DAG: v_div_fixup_f32
; GFX10: v_cndmask_b32
define amdgpu_kernel void @test_fdiv_recip_f32(ptr addrspace(1) %out, float %x) #0 {
  %fdiv = fdiv float 1.0, %x
  store float %fdiv, ptr addrspace(1) %out, align 4
  ret void
}

; Non-reciprocal division (y / x where y != 1.0) should NOT get the
; Newton-Raphson treatment even with the flag enabled.

; GFX9-LABEL: {{^}}test_fdiv_non_recip_f32:
; GFX9-NOT: v_cmp_class_f32
; GFX9-NOT: v_cndmask_b32
; GFX9: v_div_scale_f32
;
; GFX10-LABEL: {{^}}test_fdiv_non_recip_f32:
; GFX10-NOT: v_cmp_class_f32
; GFX10-NOT: v_cndmask_b32
; GFX10: v_div_scale_f32
define amdgpu_kernel void @test_fdiv_non_recip_f32(ptr addrspace(1) %out, float %x, float %y) #0 {
  %fdiv = fdiv float %y, %x
  store float %fdiv, ptr addrspace(1) %out, align 4
  ret void
}

attributes #0 = { nounwind }
