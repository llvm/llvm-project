; RUN: opt -S -mtriple=amdgcn-- -mcpu=bonaire -loop-reduce < %s | FileCheck -check-prefix=OPT %s

; Test that loops with different maximum offsets for different address
; spaces are correctly handled.

target datalayout = "e-p:32:32-p1:64:64-p2:64:64-p3:32:32-p4:64:64-p5:32:32-p24:64:64-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64"

; OPT-LABEL: @test_global_addressing_loop_uniform_index_max_offset_i32(
; OPT: {{^}}.lr.ph:
; OPT: %lsr.iv2 = phi i8 addrspace(1)* [ %scevgep3, %.lr.ph ], [ %arg1, %.lr.ph.preheader ]
; OPT: %scevgep4 = getelementptr i8, i8 addrspace(1)* %lsr.iv2, i64 4095
; OPT: load i8, i8 addrspace(1)* %scevgep4, align 1
define amdgpu_kernel void @test_global_addressing_loop_uniform_index_max_offset_i32(i32 addrspace(1)* noalias nocapture %arg0, i8 addrspace(1)* noalias nocapture readonly %arg1, i32 %n) #0 {
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
  %tmp1 = add nuw nsw i64 %indvars.iv, 4095
  %tmp2 = getelementptr inbounds i8, i8 addrspace(1)* %arg1, i64 %tmp1
  %tmp3 = load i8, i8 addrspace(1)* %tmp2, align 1
  %tmp4 = sext i8 %tmp3 to i32
  %tmp5 = getelementptr inbounds i32, i32 addrspace(1)* %arg0, i64 %indvars.iv
  %tmp6 = load i32, i32 addrspace(1)* %tmp5, align 4
  %tmp7 = add nsw i32 %tmp6, %tmp4
  store i32 %tmp7, i32 addrspace(1)* %tmp5, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %._crit_edge.loopexit, label %.lr.ph
}

; OPT-LABEL: @test_global_addressing_loop_uniform_index_max_offset_p1_i32(
; OPT: {{^}}.lr.ph.preheader:
; OPT: %scevgep2 = getelementptr i8, i8 addrspace(1)* %arg1, i64 4096
; OPT: br label %.lr.ph

; OPT: {{^}}.lr.ph:
; OPT: %lsr.iv3 = phi i8 addrspace(1)* [ %scevgep4, %.lr.ph ], [ %scevgep2, %.lr.ph.preheader ]
; OPT: %scevgep4 = getelementptr i8, i8 addrspace(1)* %lsr.iv3, i64 1
define amdgpu_kernel void @test_global_addressing_loop_uniform_index_max_offset_p1_i32(i32 addrspace(1)* noalias nocapture %arg0, i8 addrspace(1)* noalias nocapture readonly %arg1, i32 %n) #0 {
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
  %tmp1 = add nuw nsw i64 %indvars.iv, 4096
  %tmp2 = getelementptr inbounds i8, i8 addrspace(1)* %arg1, i64 %tmp1
  %tmp3 = load i8, i8 addrspace(1)* %tmp2, align 1
  %tmp4 = sext i8 %tmp3 to i32
  %tmp5 = getelementptr inbounds i32, i32 addrspace(1)* %arg0, i64 %indvars.iv
  %tmp6 = load i32, i32 addrspace(1)* %tmp5, align 4
  %tmp7 = add nsw i32 %tmp6, %tmp4
  store i32 %tmp7, i32 addrspace(1)* %tmp5, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %._crit_edge.loopexit, label %.lr.ph
}

; OPT-LABEL: @test_local_addressing_loop_uniform_index_max_offset_i32(
; OPT: {{^}}.lr.ph
; OPT: %lsr.iv2 = phi i8 addrspace(3)* [ %scevgep3, %.lr.ph ], [ %arg1, %.lr.ph.preheader ]
; OPT: %scevgep4 = getelementptr i8, i8 addrspace(3)* %lsr.iv2, i32 65535
; OPT: %tmp4 = load i8, i8 addrspace(3)* %scevgep4, align 1
define amdgpu_kernel void @test_local_addressing_loop_uniform_index_max_offset_i32(i32 addrspace(1)* noalias nocapture %arg0, i8 addrspace(3)* noalias nocapture readonly %arg1, i32 %n) #0 {
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
  %tmp1 = add nuw nsw i64 %indvars.iv, 65535
  %tmp2 = trunc i64 %tmp1 to i32
  %tmp3 = getelementptr inbounds i8, i8 addrspace(3)* %arg1, i32 %tmp2
  %tmp4 = load i8, i8 addrspace(3)* %tmp3, align 1
  %tmp5 = sext i8 %tmp4 to i32
  %tmp6 = getelementptr inbounds i32, i32 addrspace(1)* %arg0, i64 %indvars.iv
  %tmp7 = load i32, i32 addrspace(1)* %tmp6, align 4
  %tmp8 = add nsw i32 %tmp7, %tmp5
  store i32 %tmp8, i32 addrspace(1)* %tmp6, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %._crit_edge.loopexit, label %.lr.ph
}

; OPT-LABEL: @test_local_addressing_loop_uniform_index_max_offset_p1_i32(
; OPT: {{^}}.lr.ph.preheader:
; OPT: %scevgep2 = getelementptr i8, i8 addrspace(3)* %arg1, i32 65536
; OPT: br label %.lr.ph

; OPT: {{^}}.lr.ph:
; OPT: %lsr.iv3 = phi i8 addrspace(3)* [ %scevgep4, %.lr.ph ], [ %scevgep2, %.lr.ph.preheader ]
; OPT: %scevgep4 = getelementptr i8, i8 addrspace(3)* %lsr.iv3, i32 1
define amdgpu_kernel void @test_local_addressing_loop_uniform_index_max_offset_p1_i32(i32 addrspace(1)* noalias nocapture %arg0, i8 addrspace(3)* noalias nocapture readonly %arg1, i32 %n) #0 {
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
  %tmp1 = add nuw nsw i64 %indvars.iv, 65536
  %tmp2 = trunc i64 %tmp1 to i32
  %tmp3 = getelementptr inbounds i8, i8 addrspace(3)* %arg1, i32 %tmp2
  %tmp4 = load i8, i8 addrspace(3)* %tmp3, align 1
  %tmp5 = sext i8 %tmp4 to i32
  %tmp6 = getelementptr inbounds i32, i32 addrspace(1)* %arg0, i64 %indvars.iv
  %tmp7 = load i32, i32 addrspace(1)* %tmp6, align 4
  %tmp8 = add nsw i32 %tmp7, %tmp5
  store i32 %tmp8, i32 addrspace(1)* %tmp6, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %._crit_edge.loopexit, label %.lr.ph
}

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="hawaii" "unsafe-fp-math"="false" "use-soft-float"="false" }
