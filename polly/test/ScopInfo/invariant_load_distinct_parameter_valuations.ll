; RUN: opt %loadNPMPolly '-passes=print<polly-function-scops>' -polly-invariant-load-hoisting=true -disable-output < %s 2>&1 | FileCheck %s
;
; Check that we do not consolidate the invariant loads to smp[order - 1] and
; smp[order - 2] in the blocks %0 and %16. While they have the same pointer
; operand (SCEV) they do not have the same access relation due to the
; instantiation of "order" from their domain.
;
; CHECK:         Invariant Accesses: {
; CHECK-NEXT:            ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                [order, n] -> { Stmt_bb1[] -> MemRef_smp[1] };
; CHECK-NEXT:            Execution Context: [order, n] -> {  : order = 2 }
; CHECK-NEXT:            ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                [order, n] -> { Stmt_bb1[] -> MemRef_smp[0] };
; CHECK-NEXT:            Execution Context: [order, n] -> {  : order = 2 }
; CHECK-NEXT:            ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                [order, n] -> { Stmt_bb16[] -> MemRef_smp[2] };
; CHECK-NEXT:            Execution Context: [order, n] -> {  : order = 3 }
; CHECK-NEXT:            ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                [order, n] -> { Stmt_bb16[] -> MemRef_smp[1] };
; CHECK-NEXT:            Execution Context: [order, n] -> {  : order = 3 }
; CHECK-NEXT:    }
;
; ModuleID = '/home/johannes/Downloads/test_case.ll'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Function Attrs: nounwind uwtable
define void @encode_residual_fixed(ptr %res, ptr %smp, i32 %n, i32 %order) {
bb:
  br label %.split

.split:                                           ; preds = %bb
  switch i32 %order, label %bb32 [
    i32 2, label %bb1
    i32 3, label %bb16
  ]

bb1:                                              ; preds = %.split
  %tmp = add nsw i32 %order, -1
  %tmp2 = sext i32 %tmp to i64
  %tmp3 = getelementptr inbounds i32, ptr %smp, i64 %tmp2
  %tmp4 = load i32, ptr %tmp3, align 4
  %tmp5 = add nsw i32 %order, -2
  %tmp6 = sext i32 %tmp5 to i64
  %tmp7 = getelementptr inbounds i32, ptr %smp, i64 %tmp6
  %tmp8 = load i32, ptr %tmp7, align 4
  %tmp9 = sub nsw i32 %tmp4, %tmp8
  %tmp10 = icmp slt i32 %order, %n
  br i1 %tmp10, label %.lr.ph, label %.loopexit

.lr.ph:                                           ; preds = %bb1
  %tmp11 = sext i32 %order to i64
  br label %bb12

bb12:                                             ; preds = %bb12, %.lr.ph
  %indvars.iv = phi i64 [ %tmp11, %.lr.ph ], [ %indvars.iv.next, %bb12 ]
  %i.03 = phi i32 [ %order, %.lr.ph ], [ %tmp14, %bb12 ]
  %tmp13 = getelementptr inbounds i32, ptr %res, i64 %indvars.iv
  store i32 %tmp9, ptr %tmp13, align 4
  %tmp14 = add nsw i32 %i.03, 2
  %tmp15 = icmp slt i32 %tmp14, %n
  %indvars.iv.next = add nsw i64 %indvars.iv, 2
  br i1 %tmp15, label %bb12, label %..loopexit_crit_edge

bb16:                                             ; preds = %.split
  %tmp17 = add nsw i32 %order, -1
  %tmp18 = sext i32 %tmp17 to i64
  %tmp19 = getelementptr inbounds i32, ptr %smp, i64 %tmp18
  %tmp20 = load i32, ptr %tmp19, align 4
  %tmp21 = add nsw i32 %order, -2
  %tmp22 = sext i32 %tmp21 to i64
  %tmp23 = getelementptr inbounds i32, ptr %smp, i64 %tmp22
  %tmp24 = load i32, ptr %tmp23, align 4
  %tmp25 = sub nsw i32 %tmp20, %tmp24
  %tmp26 = icmp slt i32 %order, %n
  br i1 %tmp26, label %.lr.ph5, label %.loopexit2

.lr.ph5:                                          ; preds = %bb16
  %tmp27 = sext i32 %order to i64
  br label %bb28

bb28:                                             ; preds = %bb28, %.lr.ph5
  %indvars.iv6 = phi i64 [ %tmp27, %.lr.ph5 ], [ %indvars.iv.next7, %bb28 ]
  %i.14 = phi i32 [ %order, %.lr.ph5 ], [ %tmp30, %bb28 ]
  %tmp29 = getelementptr inbounds i32, ptr %res, i64 %indvars.iv6
  store i32 %tmp25, ptr %tmp29, align 4
  %tmp30 = add nsw i32 %i.14, 2
  %tmp31 = icmp slt i32 %tmp30, %n
  %indvars.iv.next7 = add nsw i64 %indvars.iv6, 2
  br i1 %tmp31, label %bb28, label %..loopexit2_crit_edge

..loopexit_crit_edge:                             ; preds = %bb12
  br label %.loopexit

.loopexit:                                        ; preds = %..loopexit_crit_edge, %bb1
  br label %bb32

..loopexit2_crit_edge:                            ; preds = %bb28
  br label %.loopexit2

.loopexit2:                                       ; preds = %..loopexit2_crit_edge, %bb16
  br label %bb32

bb32:                                             ; preds = %.loopexit2, %.loopexit, %.split
  %tmp33 = getelementptr inbounds i32, ptr %res, i64 2
  %tmp34 = load i32, ptr %tmp33, align 4
  %tmp35 = icmp eq i32 %tmp34, 5
  br i1 %tmp35, label %bb37, label %bb36

bb36:                                             ; preds = %bb32
  ret void

bb37:                                             ; preds = %bb32
  ret void
}
