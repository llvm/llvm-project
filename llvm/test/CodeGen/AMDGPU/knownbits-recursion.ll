; RUN: llc -march=amdgcn < %s | FileCheck -check-prefix=GCN %s

; Check we can compile this test without infinite loop in the
; DAG.computeKnownBits() due to missing (Depth + 1) argument in
; call to it from computeKnownBitsForTargetNode().

; Check that we actually have that target 24 bit multiplication
; node produced.

; GCN: v_mul_u32_u24
define amdgpu_kernel void @test(i32 addrspace(1)* nocapture %arg) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  br label %bb4

bb1:                                              ; preds = %bb4
  %tmp3 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i32 %tmp46
  store i32 %tmp46, i32 addrspace(1)* %tmp3, align 4
  ret void

bb4:                                              ; preds = %bb4, %bb
  %tmp5 = phi i32 [ 0, %bb ], [ %tmp87, %bb4 ]
  %tmp6 = phi i32 [ %tmp, %bb ], [ %tmp46, %bb4 ]
  %tmp7 = ashr i32 %tmp6, 16
  %tmp8 = mul nsw i32 %tmp7, %tmp7
  %tmp9 = lshr i32 %tmp8, 16
  %tmp10 = mul nuw nsw i32 %tmp9, %tmp9
  %tmp11 = lshr i32 %tmp10, 16
  %tmp12 = mul nuw nsw i32 %tmp11, %tmp11
  %tmp13 = lshr i32 %tmp12, 16
  %tmp14 = mul nuw nsw i32 %tmp13, %tmp13
  %tmp15 = lshr i32 %tmp14, 16
  %tmp16 = mul nuw nsw i32 %tmp15, %tmp15
  %tmp17 = lshr i32 %tmp16, 16
  %tmp18 = mul nuw nsw i32 %tmp17, %tmp17
  %tmp19 = lshr i32 %tmp18, 16
  %tmp20 = mul nuw nsw i32 %tmp19, %tmp19
  %tmp21 = lshr i32 %tmp20, 16
  %tmp22 = mul nuw nsw i32 %tmp21, %tmp21
  %tmp23 = lshr i32 %tmp22, 16
  %tmp24 = mul nuw nsw i32 %tmp23, %tmp23
  %tmp25 = lshr i32 %tmp24, 16
  %tmp26 = mul nuw nsw i32 %tmp25, %tmp25
  %tmp27 = lshr i32 %tmp26, 16
  %tmp28 = mul nuw nsw i32 %tmp27, %tmp27
  %tmp29 = lshr i32 %tmp28, 16
  %tmp30 = mul nuw nsw i32 %tmp29, %tmp29
  %tmp31 = lshr i32 %tmp30, 16
  %tmp32 = mul nuw nsw i32 %tmp31, %tmp31
  %tmp33 = lshr i32 %tmp32, 16
  %tmp34 = mul nuw nsw i32 %tmp33, %tmp33
  %tmp35 = lshr i32 %tmp34, 16
  %tmp36 = mul nuw nsw i32 %tmp35, %tmp35
  %tmp37 = lshr i32 %tmp36, 16
  %tmp38 = mul nuw nsw i32 %tmp37, %tmp37
  %tmp39 = lshr i32 %tmp38, 16
  %tmp40 = mul nuw nsw i32 %tmp39, %tmp39
  %tmp41 = lshr i32 %tmp40, 16
  %tmp42 = mul nuw nsw i32 %tmp41, %tmp41
  %tmp43 = lshr i32 %tmp42, 16
  %tmp44 = mul nuw nsw i32 %tmp43, %tmp43
  %tmp45 = lshr i32 %tmp44, 16
  %tmp46 = mul nuw nsw i32 %tmp45, %tmp45
  %tmp87 = add nuw nsw i32 %tmp5, 1
  %tmp88 = icmp eq i32 %tmp87, 1000
  br i1 %tmp88, label %bb1, label %bb4
}

declare i32 @llvm.amdgcn.workitem.id.x()
