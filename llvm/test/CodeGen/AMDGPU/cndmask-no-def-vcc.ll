; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

declare i1 @llvm.amdgcn.class.f32(float, i32)

; Produces error after adding an implicit def to v_cndmask_b32

; GCN-LABEL: {{^}}vcc_shrink_vcc_def:
; GCN: v_cmp_eq_i32_e64 vcc, s{{[0-9]+}}, 0{{$}}
; GCN: v_cndmask_b32_e32 v{{[0-9]+}}, 1.0, v{{[0-9]+}}, vcc
; GCN: v_cndmask_b32_e64 v1, 0, 1, s{{\[[0-9]+:[0-9]+\]}}
define void @vcc_shrink_vcc_def(float %arg, i32 %arg1, float %arg2, i32 %arg3) {
bb0:
  %tmp = icmp sgt i32 %arg1, 4
  %c = icmp eq i32 %arg3, 0
  %tmp4 = select i1 %c, float %arg, float 1.000000e+00
  %tmp5 = fcmp ogt float %arg2, 0.000000e+00
  %tmp6 = fcmp olt float %arg2, 1.000000e+00
  %tmp7 = fcmp olt float %arg, %tmp4
  %tmp8 = and i1 %tmp5, %tmp6
  %tmp9 = and i1 %tmp8, %tmp7
  br i1 %tmp9, label %bb1, label %bb2

bb1:
  store volatile i32 0, i32 addrspace(1)* undef
  br label %bb2

bb2:
  ret void
}

; The undef flag on the condition src must be preserved on the
; implicit vcc use to avoid verifier errors.

; GCN-LABEL: {{^}}preserve_condition_undef_flag:
; GCN-NOT: vcc
; GCN: v_cndmask_b32_e32 v{{[0-9]+}}, 1.0, v{{[0-9]+}}, vcc
; GCN: v_cndmask_b32_e64 v1, 0, 1, s{{\[[0-9]+:[0-9]+\]}}
define void @preserve_condition_undef_flag(float %arg, i32 %arg1, float %arg2) {
bb0:
  %tmp = icmp sgt i32 %arg1, 4
  %undef = call i1 @llvm.amdgcn.class.f32(float undef, i32 undef)
  %tmp4 = select i1 %undef, float %arg, float 1.000000e+00
  %tmp5 = fcmp ogt float %arg2, 0.000000e+00
  %tmp6 = fcmp olt float %arg2, 1.000000e+00
  %tmp7 = fcmp olt float %arg, %tmp4
  %tmp8 = and i1 %tmp5, %tmp6
  %tmp9 = and i1 %tmp8, %tmp7
  br i1 %tmp9, label %bb1, label %bb2

bb1:
  store volatile i32 0, i32 addrspace(1)* undef
  br label %bb2

bb2:
  ret void
}
