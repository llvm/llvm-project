; RUN: llc -global-isel -mtriple=amdgpu9.00-amd-amdhsa %s -o - | FileCheck --check-prefix=GCN %s

; The nonzero proof eliminates a v_min_u32 clamp, reducing the output from 7 to
; 6 target instructions. Direct G_EXTRACT_VECTOR_ELT value-tracking coverage is
; in known-never-zero-vector.mir.

declare i32 @llvm.ctlz.i32(i32, i1 immarg)

define i32 @ctlz_extract_vector_elt(i1 %c, i32 %idx) {
; GCN-LABEL: ctlz_extract_vector_elt:
; GCN-NEXT:  ; %bb.0:
; GCN-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN-NEXT:    v_and_b32_e32 v0, 1, v0
; GCN-NEXT:    v_cmp_ne_u32_e32 vcc, 0, v0
; GCN-NEXT:    v_cndmask_b32_e64 v0, 2, 1, vcc
; GCN-NEXT:    v_ffbh_u32_e32 v0, v0
; GCN-NOT:     v_min_u32
; GCN-NEXT:    s_setpc_b64 s[30:31]
  %v = select i1 %c, <2 x i32> <i32 1, i32 1>,
                      <2 x i32> <i32 2, i32 2>
  %x = extractelement <2 x i32> %v, i32 %idx
  %r = call i32 @llvm.ctlz.i32(i32 %x, i1 false)
  ret i32 %r
}
