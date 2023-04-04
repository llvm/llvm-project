; RUN: llc -march=amdgcn < %s | FileCheck %s

; Check we can compile this bugpoint-reduced test without an
; infinite loop in TLI.SimplifyDemandedBits() due to failure
; to use return value of TLO.DAG.UpdateNodeOperands()

; Check that code was generated; we know there will be
; a s_endpgm, so check for it.

@0 = external unnamed_addr addrspace(3) global [462 x float], align 4

; Function Attrs: nounwind readnone speculatable
declare i32 @llvm.amdgcn.workitem.id.y() #0

; Function Attrs: nounwind readnone speculatable
declare i32 @llvm.amdgcn.workitem.id.x() #0

; Function Attrs: nounwind readnone speculatable
declare float @llvm.fmuladd.f32(float, float, float) #0

; CHECK: s_endpgm
define amdgpu_kernel void @foo(ptr addrspace(1) noalias nocapture readonly %arg, ptr addrspace(1) noalias nocapture readonly %arg1, ptr addrspace(1) noalias nocapture %arg2, float %arg3, i1 %c0, i1 %c1, i1 %c2, i1 %c3, i1 %c4, i1 %c5) local_unnamed_addr !reqd_work_group_size !0 {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.y()
  %tmp4 = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp5 = and i32 %tmp, 15
  %tmp6 = mul nuw nsw i32 %tmp5, 21
  %tmp7 = sub i32 %tmp6, 0
  %tmp8 = add i32 %tmp7, 0
  %tmp9 = add i32 %tmp8, 0
  br label %bb12

bb11:                                             ; preds = %bb30
  br i1 %c0, label %bb37, label %bb38

bb12:                                             ; preds = %bb30, %bb
  br i1 false, label %.preheader, label %.loopexit145

.loopexit145:                                     ; preds = %.preheader, %bb12
  br label %bb13

bb13:                                             ; preds = %.loopexit, %.loopexit145
  %tmp14 = phi i32 [ %tmp5, %.loopexit145 ], [ %tmp20, %.loopexit ]
  %tmp15 = add nsw i32 %tmp14, -3
  %tmp16 = mul i32 %tmp14, 21
  br i1 %c1, label %bb17, label %.loopexit

bb17:                                             ; preds = %bb13
  %tmp18 = mul i32 %tmp15, 224
  %tmp19 = add i32 undef, %tmp18
  br label %bb21

.loopexit:                                        ; preds = %bb21, %bb13
  %tmp20 = add nuw nsw i32 %tmp14, 16
  br i1 %c2, label %bb13, label %bb26

bb21:                                             ; preds = %bb21, %bb17
  %tmp22 = phi i32 [ %tmp4, %bb17 ], [ %tmp25, %bb21 ]
  %tmp23 = add i32 %tmp22, %tmp16
  %tmp24 = getelementptr inbounds float, ptr addrspace(3) @0, i32 %tmp23
  store float undef, ptr addrspace(3) %tmp24, align 4
  %tmp25 = add nuw i32 %tmp22, 8
  br i1 %c3, label %bb21, label %.loopexit

bb26:                                             ; preds = %.loopexit
  br label %bb31

.preheader:                                       ; preds = %.preheader, %bb12
  %tmp27 = phi i32 [ %tmp28, %.preheader ], [ undef, %bb12 ]
  %tmp28 = add nuw i32 %tmp27, 128
  %tmp29 = icmp ult i32 %tmp28, 1568
  br i1 %tmp29, label %.preheader, label %.loopexit145

bb30:                                             ; preds = %bb31
  br i1 %c4, label %bb11, label %bb12

bb31:                                             ; preds = %bb31, %bb26
  %tmp32 = phi i32 [ %tmp9, %bb26 ], [ undef, %bb31 ]
  %tmp33 = getelementptr inbounds [462 x float], ptr addrspace(3) @0, i32 0, i32 %tmp32
  %tmp34 = load float, ptr addrspace(3) %tmp33, align 4
  %tmp35 = tail call float @llvm.fmuladd.f32(float %tmp34, float undef, float undef)
  %tmp36 = tail call float @llvm.fmuladd.f32(float undef, float undef, float %tmp35)
  br i1 %c5, label %bb30, label %bb31

bb37:                                             ; preds = %bb11
  br label %bb38

bb38:                                             ; preds = %bb37, %bb11
  ret void
}

attributes #0 = { nounwind readnone speculatable }

!0 = !{i32 8, i32 16, i32 1}
