; RUN: opt -S -mtriple=amdgcn-- -mcpu=bonaire -loop-reduce < %s | FileCheck -check-prefix=OPT %s

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5"

; Make sure the pointer / address space of AtomicRMW is considered

; OPT-LABEL: @test_local_atomicrmw_addressing_loop_uniform_index_max_offset_i32(

; OPT: .lr.ph.preheader:
; OPT: %scevgep2 = getelementptr i8, ptr addrspace(3) %arg1, i32 65532
; OPT: br label %.lr.ph
; OPT: .lr.ph:
; OPT: %lsr.iv3 = phi ptr addrspace(3) [ %scevgep4, %.lr.ph ], [ %scevgep2, %.lr.ph.preheader ]
; OPT: %lsr.iv1 = phi ptr addrspace(3) [ %scevgep, %.lr.ph ], [ %arg0, %.lr.ph.preheader ]
; OPT: %lsr.iv = phi i32 [ %lsr.iv.next, %.lr.ph ], [ %n, %.lr.ph.preheader ]
; OPT: %tmp4 = atomicrmw add ptr addrspace(3) %lsr.iv3, i32 undef seq_cst, align 4
; OPT: %tmp7 = atomicrmw add ptr addrspace(3) %lsr.iv1, i32 undef seq_cst, align 4
; OPT: %0 = atomicrmw add ptr addrspace(3) %lsr.iv1, i32 %tmp8 seq_cst, align 4
; OPT: %scevgep4 = getelementptr i8, ptr addrspace(3) %lsr.iv3, i32 4
; OPT: br i1 %exitcond
define amdgpu_kernel void @test_local_atomicrmw_addressing_loop_uniform_index_max_offset_i32(ptr addrspace(3) noalias nocapture %arg0, ptr addrspace(3) noalias nocapture readonly %arg1, i32 %n) #0 {
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
  %indvars.iv = phi i32 [ %indvars.iv.next, %.lr.ph ], [ 0, %.lr.ph.preheader ]
  %tmp1 = add nuw nsw i32 %indvars.iv, 16383
  %tmp3 = getelementptr inbounds i32, ptr addrspace(3) %arg1, i32 %tmp1
  %tmp4 = atomicrmw add ptr addrspace(3) %tmp3, i32 undef seq_cst
  %tmp6 = getelementptr inbounds i32, ptr addrspace(3) %arg0, i32 %indvars.iv
  %tmp7 = atomicrmw add ptr addrspace(3) %tmp6, i32 undef seq_cst
  %tmp8 = add nsw i32 %tmp7, %tmp4
  atomicrmw add ptr addrspace(3) %tmp6, i32 %tmp8 seq_cst
  %indvars.iv.next = add nuw nsw i32 %indvars.iv, 1
  %exitcond = icmp eq i32 %indvars.iv.next, %n
  br i1 %exitcond, label %._crit_edge.loopexit, label %.lr.ph
}

; OPT-LABEL: test_local_cmpxchg_addressing_loop_uniform_index_max_offset_i32(

; OPT: .lr.ph.preheader:
; OPT: %scevgep2 = getelementptr i8, ptr addrspace(3)  %arg1, i32 65532
; OPT: br label %.lr.ph
; OPT: .lr.ph:
; OPT: %lsr.iv3 = phi ptr addrspace(3) [ %scevgep4, %.lr.ph ], [ %scevgep2, %.lr.ph.preheader ]
; OPT: %lsr.iv1 = phi ptr addrspace(3) [ %scevgep, %.lr.ph ], [ %arg0, %.lr.ph.preheader ]
; OPT: %lsr.iv = phi i32 [ %lsr.iv.next, %.lr.ph ], [ %n, %.lr.ph.preheader ]
; OPT: %tmp4 =  cmpxchg ptr addrspace(3) %lsr.iv3, i32 undef, i32 undef seq_cst monotonic, align 4
; OPT: %scevgep4 = getelementptr i8, ptr addrspace(3) %lsr.iv3, i32 4
define amdgpu_kernel void @test_local_cmpxchg_addressing_loop_uniform_index_max_offset_i32(ptr addrspace(3) noalias nocapture %arg0, ptr addrspace(3) noalias nocapture readonly %arg1, i32 %n) #0 {
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
  %indvars.iv = phi i32 [ %indvars.iv.next, %.lr.ph ], [ 0, %.lr.ph.preheader ]
  %tmp1 = add nuw nsw i32 %indvars.iv, 16383
  %tmp3 = getelementptr inbounds i32, ptr addrspace(3) %arg1, i32 %tmp1
  %tmp4 = cmpxchg ptr addrspace(3) %tmp3, i32 undef, i32 undef seq_cst monotonic
  %tmp4.0 = extractvalue { i32, i1 } %tmp4, 0
  %tmp6 = getelementptr inbounds i32, ptr addrspace(3) %arg0, i32 %indvars.iv
  %tmp7 = cmpxchg ptr addrspace(3) %tmp6, i32 undef, i32 undef seq_cst monotonic
  %tmp7.0 = extractvalue { i32, i1 } %tmp7, 0
  %tmp8 = add nsw i32 %tmp7.0, %tmp4.0
  atomicrmw add ptr addrspace(3) %tmp6, i32 %tmp8 seq_cst
  %indvars.iv.next = add nuw nsw i32 %indvars.iv, 1
  %exitcond = icmp eq i32 %indvars.iv.next, %n
  br i1 %exitcond, label %._crit_edge.loopexit, label %.lr.ph
}

attributes #0 = { nounwind }
attributes #1 = { nounwind argmemonly }
