; RUN: opt -S -indvars < %s | FileCheck %s

; Do not rewrite the user outside the loop because we must keep the instruction
; inside the loop due to store. Rewrite doesn't give us any profit.
define void @f(i32 %length.i.88, i32 %length.i, i8* %tmp12, i32 %tmp10, i8* %tmp8) {
; CHECK-LABEL: @f(
not_zero11.preheader:
  %tmp13 = icmp ugt i32 %length.i, %length.i.88
  %tmp14 = select i1 %tmp13, i32 %length.i.88, i32 %length.i
  %tmp15 = icmp sgt i32 %tmp14, 0
  br i1 %tmp15, label %not_zero11, label %not_zero11.postloop

not_zero11:
  %v_1 = phi i32 [ %tmp22, %not_zero11 ], [ 0, %not_zero11.preheader ]
  %tmp16 = zext i32 %v_1 to i64
  %tmp17 = getelementptr inbounds i8, i8* %tmp8, i64 %tmp16
  %tmp18 = load i8, i8* %tmp17, align 1
  %tmp19 = zext i8 %tmp18 to i32
  %tmp20 = or i32 %tmp19, %tmp10
  %tmp21 = trunc i32 %tmp20 to i8
  %addr22 = getelementptr inbounds i8, i8* %tmp12, i64 %tmp16
  store i8 %tmp21, i8* %addr22, align 1
  %tmp22 = add nuw nsw i32 %v_1, 1
  %tmp23 = icmp slt i32 %tmp22, %tmp14
  br i1 %tmp23, label %not_zero11, label %main.exit.selector

main.exit.selector:
; CHECK-LABEL: main.exit.selector:
; CHECK:   %tmp22.lcssa = phi i32 [ %tmp22, %not_zero11 ]
; CHECK:   %tmp24 = icmp slt i32 %tmp22.lcssa, %length.
  %tmp24 = icmp slt i32 %tmp22, %length.i
  br i1 %tmp24, label %not_zero11.postloop, label %leave

leave:
  ret void

not_zero11.postloop:
  ret void
}

; Rewrite the user outside the loop because there is no hard users inside the loop.
define void @f1(i32 %length.i.88, i32 %length.i, i8* %tmp12, i32 %tmp10, i8* %tmp8) {
; CHECK-LABEL: @f1(
not_zero11.preheader:
  %tmp13 = icmp ugt i32 %length.i, %length.i.88
  %tmp14 = select i1 %tmp13, i32 %length.i.88, i32 %length.i
  %tmp15 = icmp sgt i32 %tmp14, 0
  br i1 %tmp15, label %not_zero11, label %not_zero11.postloop

not_zero11:
  %v_1 = phi i32 [ %tmp22, %not_zero11 ], [ 0, %not_zero11.preheader ]
  %tmp16 = zext i32 %v_1 to i64
  %tmp17 = getelementptr inbounds i8, i8* %tmp8, i64 %tmp16
  %tmp18 = load i8, i8* %tmp17, align 1
  %tmp19 = zext i8 %tmp18 to i32
  %tmp20 = or i32 %tmp19, %tmp10
  %tmp21 = trunc i32 %tmp20 to i8
  %addr22 = getelementptr inbounds i8, i8* %tmp12, i64 %tmp16
  %tmp22 = add nuw nsw i32 %v_1, 1
  %tmp23 = icmp slt i32 %tmp22, %tmp14
  br i1 %tmp23, label %not_zero11, label %main.exit.selector

main.exit.selector:
; CHECK-LABEL: main.exit.selector:
; CHECK: %tmp24 = icmp slt i32 %tmp14, %length.i
  %tmp24 = icmp slt i32 %tmp22, %length.i
  br i1 %tmp24, label %not_zero11.postloop, label %leave

leave:
  ret void

not_zero11.postloop:
  ret void
}
