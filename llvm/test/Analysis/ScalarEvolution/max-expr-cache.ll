; RUN: opt < %s -analyze -scalar-evolution | FileCheck %s

; SCEV would take a long time to compute SCEV expressions for this IR.  If SCEV
; finishes in < 1 second then the bug is fixed.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64--linux-gnu"

define void @smax(i32 %tmp3) {
 ; CHECK-LABEL: Printing analysis 'Scalar Evolution Analysis' for function 'smax'
entry:
  br label %bb4

bb4:
  %tmp5 = phi i64 [ %tmp62, %bb61 ], [ 0, %entry ]
  %tmp6 = trunc i64 %tmp5 to i32
  %tmp7 = shl nsw i32 %tmp6, 8
  %tmp8 = sub nsw i32 %tmp3, %tmp7
  %tmp9 = icmp slt i32 %tmp8, 256
  %tmp10 = select i1 %tmp9, i32 %tmp8, i32 256
  %tmp11 = add nsw i32 %tmp10, 1
  %tmp12 = icmp sgt i32 %tmp8, %tmp11
  %tmp13 = select i1 %tmp12, i32 %tmp11, i32 %tmp8
  %tmp14 = icmp slt i32 %tmp13, 256
  %tmp15 = select i1 %tmp14, i32 %tmp13, i32 256
  %tmp16 = add nsw i32 %tmp15, 1
  %tmp17 = icmp sgt i32 %tmp8, %tmp16
  %tmp18 = select i1 %tmp17, i32 %tmp16, i32 %tmp8
  %tmp19 = icmp slt i32 %tmp18, 256
  %tmp20 = select i1 %tmp19, i32 %tmp18, i32 256
  %tmp21 = add nsw i32 %tmp20, 1
  %tmp22 = icmp sgt i32 %tmp8, %tmp21
  %tmp23 = select i1 %tmp22, i32 %tmp21, i32 %tmp8
  %tmp24 = icmp slt i32 %tmp23, 256
  %tmp25 = select i1 %tmp24, i32 %tmp23, i32 256
  %tmp26 = add nsw i32 %tmp25, 1
  %tmp27 = icmp sgt i32 %tmp8, %tmp26
  %tmp28 = select i1 %tmp27, i32 %tmp26, i32 %tmp8
  %tmp29 = icmp slt i32 %tmp28, 256
  %tmp30 = select i1 %tmp29, i32 %tmp28, i32 256
  %tmp31 = add nsw i32 %tmp30, 1
  %tmp32 = icmp sgt i32 %tmp8, %tmp31
  %tmp33 = select i1 %tmp32, i32 %tmp31, i32 %tmp8
  %tmp34 = icmp slt i32 %tmp33, 256
  %tmp35 = select i1 %tmp34, i32 %tmp33, i32 256
  %tmp36 = add nsw i32 %tmp35, 1
  %tmp37 = icmp sgt i32 %tmp8, %tmp36
  %tmp38 = select i1 %tmp37, i32 %tmp36, i32 %tmp8
  %tmp39 = icmp slt i32 %tmp38, 256
  %tmp40 = select i1 %tmp39, i32 %tmp38, i32 256
  %tmp41 = add nsw i32 %tmp40, 1
  %tmp42 = icmp sgt i32 %tmp8, %tmp41
  %tmp43 = select i1 %tmp42, i32 %tmp41, i32 %tmp8
  %tmp44 = add nsw i32 %tmp10, 7
  %tmp45 = icmp slt i32 %tmp43, 256
  %tmp46 = select i1 %tmp45, i32 %tmp43, i32 256
; CHECK:  %tmp46 = select i1 %tmp45, i32 %tmp43, i32 256
; CHECK-NEXT:  -->  (256 smin (1 + (256 smin (1 + (256 smin (1 + (256 smin (1 + (256 smin (1 + (256 smin (1 + (256 smin (1 + (256 smin {%tmp3,+,-256}<%bb4>))<nsw> smin {%tmp3,+,-256}<%bb4>))<nsw> smin {%tmp3,+,-256}<%bb4>))<nsw> smin {%tmp3,+,-256}<%bb4>))<nsw> smin {%tmp3,+,-256}<%bb4>))<nsw> smin {%tmp3,+,-256}<%bb4>))<nsw> smin {%tmp3,+,-256}<%bb4>))<nsw> smin {%tmp3,+,-256}<%bb4>) U: [-2147483648,257) S: [-2147483648,257)
  %tmp47 = icmp sgt i32 %tmp44, %tmp46
  %tmp48 = select i1 %tmp47, i32 %tmp44, i32 %tmp46
  %tmp49 = ashr i32 %tmp48, 3
  %tmp50 = icmp sgt i32 %tmp49, 0
  %tmp51 = select i1 %tmp50, i32 %tmp49, i32 0
  %tmp52 = zext i32 %tmp51 to i64
  br label %bb53

bb53:
  %tmp54 = phi i64 [ undef, %bb4 ], [ %tmp59, %bb53 ]
  %tmp55 = trunc i64 %tmp54 to i32
  %tmp56 = shl nsw i32 %tmp55, 3
  %tmp57 = sext i32 %tmp56 to i64
  %tmp58 = getelementptr inbounds i8, i8* null, i64 %tmp57
  store i8 undef, i8* %tmp58, align 8
  %tmp59 = add nsw i64 %tmp54, 1
  %tmp60 = icmp eq i64 %tmp59, %tmp52
  br i1 %tmp60, label %bb61, label %bb53

bb61:
  %tmp62 = add nuw nsw i64 %tmp5, 1
  br label %bb4
}


define void @umax(i32 %tmp3) {
; CHECK-LABEL: Printing analysis 'Scalar Evolution Analysis' for function 'umax'
entry:
  br label %bb4

bb4:
  %tmp5 = phi i64 [ %tmp62, %bb61 ], [ 0, %entry ]
  %tmp6 = trunc i64 %tmp5 to i32
  %tmp7 = shl nsw i32 %tmp6, 8
  %tmp8 = sub nsw i32 %tmp3, %tmp7
  %tmp9 = icmp ult i32 %tmp8, 256
  %tmp10 = select i1 %tmp9, i32 %tmp8, i32 256
  %tmp11 = add nsw i32 %tmp10, 1
  %tmp12 = icmp ugt i32 %tmp8, %tmp11
  %tmp13 = select i1 %tmp12, i32 %tmp11, i32 %tmp8
  %tmp14 = icmp ult i32 %tmp13, 256
  %tmp15 = select i1 %tmp14, i32 %tmp13, i32 256
  %tmp16 = add nsw i32 %tmp15, 1
  %tmp17 = icmp ugt i32 %tmp8, %tmp16
  %tmp18 = select i1 %tmp17, i32 %tmp16, i32 %tmp8
  %tmp19 = icmp ult i32 %tmp18, 256
  %tmp20 = select i1 %tmp19, i32 %tmp18, i32 256
  %tmp21 = add nsw i32 %tmp20, 1
  %tmp22 = icmp ugt i32 %tmp8, %tmp21
  %tmp23 = select i1 %tmp22, i32 %tmp21, i32 %tmp8
  %tmp24 = icmp ult i32 %tmp23, 256
  %tmp25 = select i1 %tmp24, i32 %tmp23, i32 256
  %tmp26 = add nsw i32 %tmp25, 1
  %tmp27 = icmp ugt i32 %tmp8, %tmp26
  %tmp28 = select i1 %tmp27, i32 %tmp26, i32 %tmp8
  %tmp29 = icmp ult i32 %tmp28, 256
  %tmp30 = select i1 %tmp29, i32 %tmp28, i32 256
  %tmp31 = add nsw i32 %tmp30, 1
  %tmp32 = icmp ugt i32 %tmp8, %tmp31
  %tmp33 = select i1 %tmp32, i32 %tmp31, i32 %tmp8
  %tmp34 = icmp ult i32 %tmp33, 256
  %tmp35 = select i1 %tmp34, i32 %tmp33, i32 256
  %tmp36 = add nsw i32 %tmp35, 1
  %tmp37 = icmp ugt i32 %tmp8, %tmp36
  %tmp38 = select i1 %tmp37, i32 %tmp36, i32 %tmp8
  %tmp39 = icmp ult i32 %tmp38, 256
  %tmp40 = select i1 %tmp39, i32 %tmp38, i32 256
  %tmp41 = add nsw i32 %tmp40, 1
  %tmp42 = icmp ugt i32 %tmp8, %tmp41
  %tmp43 = select i1 %tmp42, i32 %tmp41, i32 %tmp8
  %tmp44 = add nsw i32 %tmp10, 7
  %tmp45 = icmp ult i32 %tmp43, 256
  %tmp46 = select i1 %tmp45, i32 %tmp43, i32 256
; CHECK:  %tmp46 = select i1 %tmp45, i32 %tmp43, i32 256
; CHECK-NEXT:  --> (256 umin (1 + (256 umin (1 + (256 umin (1 + (256 umin (1 + (256 umin (1 + (256 umin (1 + (256 umin (1 + (256 umin {%tmp3,+,-256}<%bb4>))<nuw><nsw> umin {%tmp3,+,-256}<%bb4>))<nuw><nsw> umin {%tmp3,+,-256}<%bb4>))<nuw><nsw> umin {%tmp3,+,-256}<%bb4>))<nuw><nsw> umin {%tmp3,+,-256}<%bb4>))<nuw><nsw> umin {%tmp3,+,-256}<%bb4>))<nuw><nsw> umin {%tmp3,+,-256}<%bb4>))<nuw><nsw> umin {%tmp3,+,-256}<%bb4>) U: [0,257) S: [0,257)
  %tmp47 = icmp ugt i32 %tmp44, %tmp46
  %tmp48 = select i1 %tmp47, i32 %tmp44, i32 %tmp46
  %tmp49 = ashr i32 %tmp48, 3
  %tmp50 = icmp ugt i32 %tmp49, 0
  %tmp51 = select i1 %tmp50, i32 %tmp49, i32 0
  %tmp52 = zext i32 %tmp51 to i64
  br label %bb53

bb53:
  %tmp54 = phi i64 [ undef, %bb4 ], [ %tmp59, %bb53 ]
  %tmp55 = trunc i64 %tmp54 to i32
  %tmp56 = shl nsw i32 %tmp55, 3
  %tmp57 = sext i32 %tmp56 to i64
  %tmp58 = getelementptr inbounds i8, i8* null, i64 %tmp57
  store i8 undef, i8* %tmp58, align 8
  %tmp59 = add nsw i64 %tmp54, 1
  %tmp60 = icmp eq i64 %tmp59, %tmp52
  br i1 %tmp60, label %bb61, label %bb53

bb61:
  %tmp62 = add nuw nsw i64 %tmp5, 1
  br label %bb4
}
