; RUN: llc -global-isel -mtriple=amdgpu9.00-amd-amdhsa %s -o - | FileCheck --check-prefix=GCN %s

; The nonzero proof eliminates two v_min_u32 clamps, reducing the output from
; 12 to 10 target instructions. Direct G_BUILD_VECTOR value-tracking coverage
; is in known-never-zero-vector.mir.

declare <2 x i32> @llvm.ctlz.v2i32(<2 x i32>, i1 immarg)

define <2 x i32> @ctlz_build_vector(i1 %c0, i1 %c1) {
; GCN-LABEL: ctlz_build_vector:
; GCN-NEXT:  ; %bb.0:
; GCN-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN-NEXT:    v_and_b32_e32 v0, 1, v0
; GCN-NEXT:    v_and_b32_e32 v1, 1, v1
; GCN-NEXT:    v_cmp_ne_u32_e32 vcc, 0, v0
; GCN-NEXT:    v_cndmask_b32_e64 v0, 6, 1, vcc
; GCN-NEXT:    v_cmp_ne_u32_e32 vcc, 0, v1
; GCN-NEXT:    v_cndmask_b32_e64 v1, 6, 1, vcc
; GCN-NEXT:    v_ffbh_u32_e32 v0, v0
; GCN-NEXT:    v_ffbh_u32_e32 v1, v1
; GCN-NOT:     v_min_u32
; GCN-NEXT:    s_setpc_b64 s[30:31]
  %x0 = select i1 %c0, i32 1, i32 6
  %x1 = select i1 %c1, i32 1, i32 6
  %v0 = insertelement <1 x i32> poison, i32 %x0, i64 0
  %v1 = insertelement <1 x i32> poison, i32 %x1, i64 0
  %built = shufflevector <1 x i32> %v0, <1 x i32> %v1,
                         <2 x i32> <i32 0, i32 1>
  %r = call <2 x i32> @llvm.ctlz.v2i32(<2 x i32> %built, i1 false)
  ret <2 x i32> %r
}
