; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN %s

; GCN-LABEL: {{^}}udiv32_invariant_denom:
; GCN:     v_cvt_f32_u32
; GCN:     v_rcp_iflag_f32
; GCN:     v_mul_f32_e32 v{{[0-9]+}}, 0x4f800000,
; GCN:     v_cvt_u32_f32_e32
; GCN-DAG: v_mul_hi_u32
; GCN-DAG: v_mul_lo_i32
; GCN-DAG: v_sub_i32_e32
; GCN-DAG: v_cmp_eq_u32_e64
; GCN-DAG: v_cndmask_b32_e64
; GCN-DAG: v_mul_hi_u32
; GCN-DAG: v_add_i32_e32
; GCN-DAG: v_subrev_i32_e32
; GCN-DAG: v_cndmask_b32_e64
; GCN:     [[LOOP:BB[0-9_]+]]:
; GCN-NOT: v_rcp
; GCN:     s_cbranch_scc0 [[LOOP]]
; GCN:     s_endpgm
define amdgpu_kernel void @udiv32_invariant_denom(i32 addrspace(1)* nocapture %arg, i32 %arg1) {
bb:
  br label %bb3

bb2:                                              ; preds = %bb3
  ret void

bb3:                                              ; preds = %bb3, %bb
  %tmp = phi i32 [ 0, %bb ], [ %tmp7, %bb3 ]
  %tmp4 = udiv i32 %tmp, %arg1
  %tmp5 = zext i32 %tmp to i64
  %tmp6 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i64 %tmp5
  store i32 %tmp4, i32 addrspace(1)* %tmp6, align 4
  %tmp7 = add nuw nsw i32 %tmp, 1
  %tmp8 = icmp eq i32 %tmp7, 1024
  br i1 %tmp8, label %bb2, label %bb3
}

; GCN-LABEL: {{^}}urem32_invariant_denom:
; GCN:     v_cvt_f32_u32
; GCN:     v_rcp_iflag_f32
; GCN:     v_mul_f32_e32 v{{[0-9]+}}, 0x4f800000,
; GCN:     v_cvt_u32_f32_e32
; GCN-DAG: v_mul_hi_u32
; GCN-DAG: v_mul_lo_i32
; GCN-DAG: v_sub_i32_e32
; GCN-DAG: v_cmp_eq_u32_e64
; GCN-DAG: v_cndmask_b32_e64
; GCN-DAG: v_mul_hi_u32
; GCN-DAG: v_add_i32_e32
; GCN-DAG: v_subrev_i32_e32
; GCN-DAG: v_cndmask_b32_e64
; GCN:     [[LOOP:BB[0-9_]+]]:
; GCN-NOT: v_rcp
; GCN:     s_cbranch_scc0 [[LOOP]]
; GCN:     s_endpgm
define amdgpu_kernel void @urem32_invariant_denom(i32 addrspace(1)* nocapture %arg, i32 %arg1) {
bb:
  br label %bb3

bb2:                                              ; preds = %bb3
  ret void

bb3:                                              ; preds = %bb3, %bb
  %tmp = phi i32 [ 0, %bb ], [ %tmp7, %bb3 ]
  %tmp4 = urem i32 %tmp, %arg1
  %tmp5 = zext i32 %tmp to i64
  %tmp6 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i64 %tmp5
  store i32 %tmp4, i32 addrspace(1)* %tmp6, align 4
  %tmp7 = add nuw nsw i32 %tmp, 1
  %tmp8 = icmp eq i32 %tmp7, 1024
  br i1 %tmp8, label %bb2, label %bb3
}

; GCN-LABEL: {{^}}sdiv32_invariant_denom:
; GCN:     v_cvt_f32_u32
; GCN:     v_rcp_iflag_f32
; GCN:     v_mul_f32_e32 v{{[0-9]+}}, 0x4f800000,
; GCN:     v_cvt_u32_f32_e32
; GCN-DAG: v_mul_hi_u32
; GCN-DAG: v_mul_lo_i32
; GCN-DAG: v_sub_i32_e32
; GCN-DAG: v_cmp_eq_u32_e64
; GCN-DAG: v_cndmask_b32_e64
; GCN-DAG: v_mul_hi_u32
; GCN-DAG: v_add_i32_e32
; GCN-DAG: v_subrev_i32_e32
; GCN-DAG: v_cndmask_b32_e64
; GCN:     [[LOOP:BB[0-9_]+]]:
; GCN-NOT: v_rcp
; GCN:     s_cbranch_scc0 [[LOOP]]
; GCN:     s_endpgm
define amdgpu_kernel void @sdiv32_invariant_denom(i32 addrspace(1)* nocapture %arg, i32 %arg1) {
bb:
  br label %bb3

bb2:                                              ; preds = %bb3
  ret void

bb3:                                              ; preds = %bb3, %bb
  %tmp = phi i32 [ 0, %bb ], [ %tmp7, %bb3 ]
  %tmp4 = sdiv i32 %tmp, %arg1
  %tmp5 = zext i32 %tmp to i64
  %tmp6 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i64 %tmp5
  store i32 %tmp4, i32 addrspace(1)* %tmp6, align 4
  %tmp7 = add nuw nsw i32 %tmp, 1
  %tmp8 = icmp eq i32 %tmp7, 1024
  br i1 %tmp8, label %bb2, label %bb3
}

; GCN-LABEL: {{^}}srem32_invariant_denom:
; GCN:     v_cvt_f32_u32
; GCN:     v_rcp_iflag_f32
; GCN:     v_mul_f32_e32 v{{[0-9]+}}, 0x4f800000,
; GCN:     v_cvt_u32_f32_e32
; GCN-DAG: v_mul_hi_u32
; GCN-DAG: v_mul_lo_i32
; GCN-DAG: v_sub_i32_e32
; GCN-DAG: v_cmp_eq_u32_e64
; GCN-DAG: v_cndmask_b32_e64
; GCN-DAG: v_mul_hi_u32
; GCN-DAG: v_add_i32_e32
; GCN-DAG: v_subrev_i32_e32
; GCN-DAG: v_cndmask_b32_e64
; GCN:     [[LOOP:BB[0-9_]+]]:
; GCN-NOT: v_rcp
; GCN:     s_cbranch_scc0 [[LOOP]]
; GCN:     s_endpgm
define amdgpu_kernel void @srem32_invariant_denom(i32 addrspace(1)* nocapture %arg, i32 %arg1) {
bb:
  br label %bb3

bb2:                                              ; preds = %bb3
  ret void

bb3:                                              ; preds = %bb3, %bb
  %tmp = phi i32 [ 0, %bb ], [ %tmp7, %bb3 ]
  %tmp4 = srem i32 %tmp, %arg1
  %tmp5 = zext i32 %tmp to i64
  %tmp6 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i64 %tmp5
  store i32 %tmp4, i32 addrspace(1)* %tmp6, align 4
  %tmp7 = add nuw nsw i32 %tmp, 1
  %tmp8 = icmp eq i32 %tmp7, 1024
  br i1 %tmp8, label %bb2, label %bb3
}

; GCN-LABEL: {{^}}udiv16_invariant_denom:
; GCN:     v_cvt_f32_u32
; GCN:     v_rcp_iflag_f32
; GCN:     [[LOOP:BB[0-9_]+]]:
; GCN-NOT: v_rcp
; GCN:     s_cbranch_scc0 [[LOOP]]
; GCN:     s_endpgm
define amdgpu_kernel void @udiv16_invariant_denom(i16 addrspace(1)* nocapture %arg, i16 %arg1) {
bb:
  br label %bb3

bb2:                                              ; preds = %bb3
  ret void

bb3:                                              ; preds = %bb3, %bb
  %tmp = phi i16 [ 0, %bb ], [ %tmp7, %bb3 ]
  %tmp4 = udiv i16 %tmp, %arg1
  %tmp5 = zext i16 %tmp to i64
  %tmp6 = getelementptr inbounds i16, i16 addrspace(1)* %arg, i64 %tmp5
  store i16 %tmp4, i16 addrspace(1)* %tmp6, align 2
  %tmp7 = add nuw nsw i16 %tmp, 1
  %tmp8 = icmp eq i16 %tmp7, 1024
  br i1 %tmp8, label %bb2, label %bb3
}

; GCN-LABEL: {{^}}urem16_invariant_denom:
; GCN:     v_cvt_f32_u32
; GCN:     v_rcp_iflag_f32
; GCN:     [[LOOP:BB[0-9_]+]]:
; GCN-NOT: v_rcp
; GCN:     s_cbranch_scc0 [[LOOP]]
; GCN:     s_endpgm
define amdgpu_kernel void @urem16_invariant_denom(i16 addrspace(1)* nocapture %arg, i16 %arg1) {
bb:
  br label %bb3

bb2:                                              ; preds = %bb3
  ret void

bb3:                                              ; preds = %bb3, %bb
  %tmp = phi i16 [ 0, %bb ], [ %tmp7, %bb3 ]
  %tmp4 = urem i16 %tmp, %arg1
  %tmp5 = zext i16 %tmp to i64
  %tmp6 = getelementptr inbounds i16, i16 addrspace(1)* %arg, i64 %tmp5
  store i16 %tmp4, i16 addrspace(1)* %tmp6, align 2
  %tmp7 = add nuw nsw i16 %tmp, 1
  %tmp8 = icmp eq i16 %tmp7, 1024
  br i1 %tmp8, label %bb2, label %bb3
}

; GCN-LABEL: {{^}}sdiv16_invariant_denom:
; GCN-DAG: s_sext_i32_i16
; GCN-DAG: v_and_b32_e32 v{{[0-9]+}}, 0x7fffffff
; GCN-DAG: v_cvt_f32_i32
; GCN-DAG: v_rcp_iflag_f32
; GCN:     [[LOOP:BB[0-9_]+]]:
; GCN-NOT: v_rcp
; GCN:     s_cbranch_scc0 [[LOOP]]
; GCN:     s_endpgm
define amdgpu_kernel void @sdiv16_invariant_denom(i16 addrspace(1)* nocapture %arg, i16 %arg1) {
bb:
  br label %bb3

bb2:                                              ; preds = %bb3
  ret void

bb3:                                              ; preds = %bb3, %bb
  %tmp = phi i16 [ 0, %bb ], [ %tmp7, %bb3 ]
  %tmp4 = sdiv i16 %tmp, %arg1
  %tmp5 = zext i16 %tmp to i64
  %tmp6 = getelementptr inbounds i16, i16 addrspace(1)* %arg, i64 %tmp5
  store i16 %tmp4, i16 addrspace(1)* %tmp6, align 2
  %tmp7 = add nuw nsw i16 %tmp, 1
  %tmp8 = icmp eq i16 %tmp7, 1024
  br i1 %tmp8, label %bb2, label %bb3
}

; GCN-LABEL: {{^}}srem16_invariant_denom:
; GCN-DAG: s_sext_i32_i16
; GCN-DAG: v_and_b32_e32 v{{[0-9]+}}, 0x7fffffff
; GCN-DAG: v_cvt_f32_i32
; GCN-DAG: v_rcp_iflag_f32
; GCN:     [[LOOP:BB[0-9_]+]]:
; GCN-NOT: v_rcp
; GCN:     s_cbranch_scc0 [[LOOP]]
; GCN:     s_endpgm
define amdgpu_kernel void @srem16_invariant_denom(i16 addrspace(1)* nocapture %arg, i16 %arg1) {
bb:
  br label %bb3

bb2:                                              ; preds = %bb3
  ret void

bb3:                                              ; preds = %bb3, %bb
  %tmp = phi i16 [ 0, %bb ], [ %tmp7, %bb3 ]
  %tmp4 = srem i16 %tmp, %arg1
  %tmp5 = zext i16 %tmp to i64
  %tmp6 = getelementptr inbounds i16, i16 addrspace(1)* %arg, i64 %tmp5
  store i16 %tmp4, i16 addrspace(1)* %tmp6, align 2
  %tmp7 = add nuw nsw i16 %tmp, 1
  %tmp8 = icmp eq i16 %tmp7, 1024
  br i1 %tmp8, label %bb2, label %bb3
}
