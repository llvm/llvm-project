; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -global-isel %s -o - | FileCheck --check-prefix=GCN %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -global-isel \
; RUN:     --amdgpuprelegalizercombiner-disable-rule=ctlz_to_ctlz_zero_poison \
; RUN:     --amdgpupostlegalizercombiner-disable-rule=ctlz_to_ctlz_zero_poison \
; RUN:     %s -o - | FileCheck --check-prefix=GCN-NOCOMBINE %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -global-isel \
; RUN:     -stop-after=amdgpu-prelegalizer-combiner \
; RUN:     %s -o - | FileCheck --check-prefix=GISEL %s
; REQUIRES: asserts

; Verify that isKnownNeverZero looks through a variable
; G_EXTRACT_VECTOR_ELT. The selected vectors use 1 and 2, so no common
; known-one bit exists. The enabled combine emits 6 target instructions, while
; disabling it emits 7.

declare i32 @llvm.ctlz.i32(i32, i1 immarg)

define i32 @ctlz_extract_vector_elt(i1 %c, i32 %idx) {
; GISEL-LABEL: name: ctlz_extract_vector_elt
; GISEL:       [[SELECT:%[0-9]+]]:_(<2 x i32>) = G_SELECT
; GISEL-NEXT:  [[EXTRACT:%[0-9]+]]:_(i32) = G_EXTRACT_VECTOR_ELT [[SELECT]](<2 x i32>), {{%[0-9]+}}(i32)
; GISEL-NEXT:  {{%[0-9]+}}:_(i32) = G_CTLZ_ZERO_POISON [[EXTRACT]](i32)
;
; GCN-LABEL: ctlz_extract_vector_elt:
; GCN-NEXT:  ; %bb.0:
; GCN-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN-NEXT:    v_and_b32_e32 v0, 1, v0
; GCN-NEXT:    v_cmp_ne_u32_e32 vcc, 0, v0
; GCN-NEXT:    v_cndmask_b32_e64 v0, 2, 1, vcc
; GCN-NEXT:    v_ffbh_u32_e32 v0, v0
; GCN-NEXT:    s_setpc_b64 s[30:31]
;
; GCN-NOCOMBINE-LABEL: ctlz_extract_vector_elt:
; GCN-NOCOMBINE-NEXT:  ; %bb.0:
; GCN-NOCOMBINE-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN-NOCOMBINE-NEXT:    v_and_b32_e32 v0, 1, v0
; GCN-NOCOMBINE-NEXT:    v_cmp_ne_u32_e32 vcc, 0, v0
; GCN-NOCOMBINE-NEXT:    v_cndmask_b32_e64 v0, 2, 1, vcc
; GCN-NOCOMBINE-NEXT:    v_ffbh_u32_e32 v0, v0
; GCN-NOCOMBINE-NEXT:    v_min_u32_e32 v0, 32, v0
; GCN-NOCOMBINE-NEXT:    s_setpc_b64 s[30:31]
  %v = select i1 %c, <2 x i32> <i32 1, i32 1>,
                      <2 x i32> <i32 2, i32 2>
  %x = extractelement <2 x i32> %v, i32 %idx
  %r = call i32 @llvm.ctlz.i32(i32 %x, i1 false)
  ret i32 %r
}
