; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; Make sure that m0 is not reinitialized in the loop.

; GCN-LABEL: {{^}}copy_local_to_global_loop_m0_init:
; GCN: s_cbranch_scc1 .LBB0_3

; Initialize in preheader
; GCN: s_mov_b32 m0, -1

; GCN: .LBB0_2:
; GCN-NOT: m0
; GCN: ds_read_b32
; GCN-NOT: m0
; GCN: buffer_store_dword

; GCN: s_cbranch_scc0 .LBB0_2

; GCN: .LBB0_3:
; GCN-NEXT: s_endpgm
define amdgpu_kernel void @copy_local_to_global_loop_m0_init(i32 addrspace(1)* noalias nocapture %out, i32 addrspace(3)* noalias nocapture readonly %in, i32 %n) #0 {
bb:
  %tmp = icmp sgt i32 %n, 0
  br i1 %tmp, label %.lr.ph.preheader, label %._crit_edge

.lr.ph.preheader:                                 ; preds = %bb
  br label %.lr.ph

._crit_edge.loopexit:                             ; preds = %.lr.ph
  br label %._crit_edge

._crit_edge:                                      ; preds = %._crit_edge.loopexit, %bb
  ret void

.lr.ph:                                           ; preds = %.lr.ph, %.lr.ph.preheader
  %indvars.iv = phi i64 [ %indvars.iv.next, %.lr.ph ], [ 0, %.lr.ph.preheader ]
  %i.01 = phi i32 [ %tmp4, %.lr.ph ], [ 0, %.lr.ph.preheader ]
  %tmp1 = getelementptr inbounds i32, i32 addrspace(3)* %in, i32 %i.01
  %tmp2 = load i32, i32 addrspace(3)* %tmp1, align 4
  %tmp3 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %indvars.iv
  store i32 %tmp2, i32 addrspace(1)* %tmp3, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %tmp4 = add nuw nsw i32 %i.01, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %._crit_edge.loopexit, label %.lr.ph
}

attributes #0 = { nounwind }
