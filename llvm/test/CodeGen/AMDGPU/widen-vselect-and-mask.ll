; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN %s

; Check that DAGTypeLegalizer::WidenVSELECTAndMask doesn't try to
; create vselects with i64 condition masks.

; FIXME: Should be able to avoid intermediate vselect
; GCN-LABEL: {{^}}widen_vselect_and_mask_v4f64:
; GCN: v_cmp_u_f64_e64 [[CMP:s\[[0-9]+:[0-9]+\]]],
; GCN: v_cndmask_b32_e64 v[[VSEL:[0-9]+]], 0, -1, [[CMP]]
; GCN: v_mov_b32_e32 v[[VSEL_EXT:[0-9]+]], v[[VSEL]]
; GCN: v_cmp_lt_i64_e32 vcc, -1, v[[[VSEL]]:[[VSEL_EXT]]]
define amdgpu_kernel void @widen_vselect_and_mask_v4f64(<4 x double> %arg) #0 {
bb:
  %tmp = extractelement <4 x double> %arg, i64 0
  %tmp1 = fcmp uno double %tmp, 0.000000e+00
  %tmp2 = sext i1 %tmp1 to i64
  %tmp3 = insertelement <4 x i64> undef, i64 %tmp2, i32 0
  %tmp4 = insertelement <4 x i64> %tmp3, i64 undef, i32 1
  %tmp5 = insertelement <4 x i64> %tmp4, i64 undef, i32 2
  %tmp6 = insertelement <4 x i64> %tmp5, i64 undef, i32 3
  %tmp7 = fcmp une <4 x double> %arg, zeroinitializer
  %tmp8 = icmp sgt <4 x i64> %tmp6, <i64 -1, i64 -1, i64 -1, i64 -1>
  %tmp9 = and <4 x i1> %tmp8, %tmp7
  %tmp10 = select <4 x i1> %tmp9, <4 x double> <double 1.0, double 1.0, double 1.0, double 1.0>, <4 x double> zeroinitializer
  store <4 x double> %tmp10, ptr addrspace(1) null, align 32
  ret void
}

; GCN-LABEL: {{^}}widen_vselect_and_mask_v4i64:
; GCN: v_cmp_eq_u64_e64 [[CMP:s\[[0-9]+:[0-9]+\]]],
; GCN: v_cndmask_b32_e64 v[[VSEL:[0-9]+]], 0, -1, [[CMP]]
; GCN: v_mov_b32_e32 v[[VSEL_EXT:[0-9]+]], v[[VSEL]]
; GCN: v_cmp_lt_i64_e32 vcc, -1, v[[[VSEL]]:[[VSEL_EXT]]]
define amdgpu_kernel void @widen_vselect_and_mask_v4i64(<4 x i64> %arg) #0 {
bb:
  %tmp = extractelement <4 x i64> %arg, i64 0
  %tmp1 = icmp eq i64 %tmp, 0
  %tmp2 = sext i1 %tmp1 to i64
  %tmp3 = insertelement <4 x i64> undef, i64 %tmp2, i32 0
  %tmp4 = insertelement <4 x i64> %tmp3, i64 undef, i32 1
  %tmp5 = insertelement <4 x i64> %tmp4, i64 undef, i32 2
  %tmp6 = insertelement <4 x i64> %tmp5, i64 undef, i32 3
  %tmp7 = icmp ne <4 x i64> %arg, zeroinitializer
  %tmp8 = icmp sgt <4 x i64> %tmp6, <i64 -1, i64 -1, i64 -1, i64 -1>
  %tmp9 = and <4 x i1> %tmp8, %tmp7
  %tmp10 = select <4 x i1> %tmp9, <4 x i64> <i64 1, i64 1, i64 1, i64 1>, <4 x i64> zeroinitializer
  store <4 x i64> %tmp10, ptr addrspace(1) null, align 32
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone speculatable }
