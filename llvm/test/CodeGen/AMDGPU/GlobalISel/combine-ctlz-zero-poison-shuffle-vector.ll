; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -global-isel \
; RUN:     --amdgpuprelegalizercombiner-disable-rule=combine_shuffle_vector_to_build_vector \
; RUN:     %s -o - | FileCheck -check-prefix=GCN %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -global-isel \
; RUN:     --amdgpuprelegalizercombiner-disable-rule=combine_shuffle_vector_to_build_vector,ctlz_to_ctlz_zero_poison \
; RUN:     --amdgpupostlegalizercombiner-disable-rule=ctlz_to_ctlz_zero_poison \
; RUN:     %s -o - | FileCheck -check-prefix=GCN-NOCOMBINE %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -global-isel \
; RUN:     -stop-after=amdgpu-prelegalizer-combiner \
; RUN:     --amdgpuprelegalizercombiner-disable-rule=combine_shuffle_vector_to_build_vector \
; RUN:     %s -o - | FileCheck -check-prefix=GISEL %s
; REQUIRES: asserts

; The enabled path has 10 target instructions. Disabling the
; ctlz_to_ctlz_zero_poison rule adds two v_min_u32 clamps, for 12 target
; instructions. Proving the shuffled lanes nonzero therefore reduces the
; instruction count from 12 to 10.

declare <2 x i32> @llvm.ctlz.v2i32(<2 x i32>, i1)

define <2 x i32> @ctlz_shuffle_vector(i1 %cond0, i1 %cond1) {
; GISEL-LABEL: name: ctlz_shuffle_vector
; GISEL:       [[SELECT:%[0-9]+]]:_(<2 x i32>) = G_SELECT
; GISEL-NEXT:  [[SHUFFLE:%[0-9]+]]:_(<2 x i32>) = G_SHUFFLE_VECTOR [[SELECT]](<2 x i32>), {{%[0-9]+}}, shufflemask(1, 0)
; GISEL-NEXT:  {{%[0-9]+}}:_(<2 x i32>) = G_CTLZ_ZERO_POISON [[SHUFFLE]](<2 x i32>)
;
; GCN-LABEL: ctlz_shuffle_vector:
; GCN-NEXT:  ; %bb.0:
; GCN-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN-NEXT:    v_and_b32_e32 v0, 1, v0
; GCN-NEXT:    v_and_b32_e32 v1, 1, v1
; GCN-NEXT:    v_cmp_ne_u32_e32 vcc, 0, v0
; GCN-NEXT:    v_cndmask_b32_e64 v2, 6, 1, vcc
; GCN-NEXT:    v_cmp_ne_u32_e32 vcc, 0, v1
; GCN-NEXT:    v_cndmask_b32_e64 v0, 1, 6, vcc
; GCN-NEXT:    v_ffbh_u32_e32 v0, v0
; GCN-NEXT:    v_ffbh_u32_e32 v1, v2
; GCN-NEXT:    s_setpc_b64 s[30:31]
;
; GCN-NOCOMBINE-LABEL: ctlz_shuffle_vector:
; GCN-NOCOMBINE-NEXT:  ; %bb.0:
; GCN-NOCOMBINE-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN-NOCOMBINE-NEXT:    v_and_b32_e32 v0, 1, v0
; GCN-NOCOMBINE-NEXT:    v_and_b32_e32 v1, 1, v1
; GCN-NOCOMBINE-NEXT:    v_cmp_ne_u32_e32 vcc, 0, v0
; GCN-NOCOMBINE-NEXT:    v_cndmask_b32_e64 v2, 6, 1, vcc
; GCN-NOCOMBINE-NEXT:    v_cmp_ne_u32_e32 vcc, 0, v1
; GCN-NOCOMBINE-NEXT:    v_cndmask_b32_e64 v0, 1, 6, vcc
; GCN-NOCOMBINE-NEXT:    v_ffbh_u32_e32 v0, v0
; GCN-NOCOMBINE-NEXT:    v_ffbh_u32_e32 v1, v2
; GCN-NOCOMBINE-NEXT:    v_min_u32_e32 v0, 32, v0
; GCN-NOCOMBINE-NEXT:    v_min_u32_e32 v1, 32, v1
; GCN-NOCOMBINE-NEXT:    s_setpc_b64 s[30:31]
  %conds0 = insertelement <2 x i1> poison, i1 %cond0, i64 0
  %conds1 = insertelement <2 x i1> %conds0, i1 %cond1, i64 1
  %lanes = select <2 x i1> %conds1, <2 x i32> <i32 1, i32 6>,
                                      <2 x i32> <i32 6, i32 1>
  %reverse = shufflevector <2 x i32> %lanes, <2 x i32> poison,
                           <2 x i32> <i32 1, i32 0>
  %result = call <2 x i32> @llvm.ctlz.v2i32(<2 x i32> %reverse, i1 false)
  ret <2 x i32> %result
}
