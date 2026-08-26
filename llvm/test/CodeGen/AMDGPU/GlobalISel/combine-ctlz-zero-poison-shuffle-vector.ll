; RUN: llc -global-isel -mtriple=amdgpu9.00-amd-amdhsa %s -o - | FileCheck --check-prefix=GCN %s

; The nonzero proof eliminates two v_min_u32 clamps, reducing the output from
; 12 to 10 target instructions. Direct G_SHUFFLE_VECTOR value-tracking coverage
; is in known-never-zero-vector.mir.

declare <2 x i32> @llvm.ctlz.v2i32(<2 x i32>, i1)

define <2 x i32> @ctlz_shuffle_vector(i1 %cond0, i1 %cond1) {
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
; GCN-NOT:     v_min_u32
; GCN-NEXT:    s_setpc_b64 s[30:31]
  %lane0 = select i1 %cond0, i32 1, i32 6
  %lane1 = select i1 %cond1, i32 6, i32 1
  %v0 = insertelement <1 x i32> poison, i32 %lane0, i64 0
  %v1 = insertelement <1 x i32> poison, i32 %lane1, i64 0
  %lanes = shufflevector <1 x i32> %v0, <1 x i32> %v1,
                         <2 x i32> <i32 0, i32 1>
  %reverse = shufflevector <2 x i32> %lanes, <2 x i32> poison,
                           <2 x i32> <i32 1, i32 0>
  %result = call <2 x i32> @llvm.ctlz.v2i32(<2 x i32> %reverse, i1 false)
  ret <2 x i32> %result
}
