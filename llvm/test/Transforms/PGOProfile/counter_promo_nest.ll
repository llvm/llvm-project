; TEST that counter updates are promoted outside the whole loop nest
; RUN: opt < %s -pgo-instr-gen -instrprof -do-counter-promotion=true -S | FileCheck --check-prefix=PROMO  %s
; RUN: opt < %s --passes=pgo-instr-gen,instrprof -do-counter-promotion=true -S | FileCheck --check-prefix=PROMO  %s 

@g = common local_unnamed_addr global i32 0, align 4
@c = local_unnamed_addr global i32 10, align 4

; Function Attrs: noinline norecurse nounwind uwtable
define void @bar() local_unnamed_addr #0 {
bb:
  %tmp2 = load i32, i32* @g, align 4, !tbaa !2
  %tmp3 = add nsw i32 %tmp2, 1
  store i32 %tmp3, i32* @g, align 4, !tbaa !2
  ret void
}

; Function Attrs: norecurse nounwind uwtable
define i32 @main() local_unnamed_addr #1 {
bb:
  store i32 0, i32* @g, align 4, !tbaa !2
  %tmp = load i32, i32* @c, align 4, !tbaa !2
  %tmp1 = icmp sgt i32 %tmp, 0
  br i1 %tmp1, label %bb2_1, label %bb84

bb2_1:
  br label %bb2

bb2:                                              ; preds = %bb39, %bb
  %tmp3 = phi i32 [ %tmp40, %bb39 ], [ %tmp, %bb2_1 ]
  %tmp5 = phi i32 [ %tmp43, %bb39 ], [ 0, %bb2_1 ]
  %tmp7 = icmp sgt i32 %tmp3, 0
  br i1 %tmp7, label %bb14_1, label %bb39

bb8:                                              ; preds = %bb39
; PROMO-LABEL: bb8
; PROMO: load {{.*}} @__profc_main{{.*}}
; PROMO-NEXT: add
; PROMO-NEXT: store {{.*}}@__profc_main{{.*}}
; PROMO-NEXT: load {{.*}} @__profc_main{{.*}}
; PROMO-NEXT: add
; PROMO-NEXT: store {{.*}}@__profc_main{{.*}}
; PROMO-NEXT: load {{.*}} @__profc_main{{.*}}
; PROMO-NEXT: add
; PROMO-NEXT: store {{.*}}@__profc_main{{.*}}
; PROMO-NEXT: load {{.*}} @__profc_main{{.*}}
; PROMO-NEXT: add
; PROMO-NEXT: store {{.*}}@__profc_main{{.*}}
; PROMO-NEXT: load {{.*}} @__profc_main{{.*}}
; PROMO-NEXT: add
; PROMO-NEXT: store {{.*}}@__profc_main{{.*}}

  %tmp13 = icmp sgt i32 %tmp40, 0
  br i1 %tmp13, label %bb45, label %bb84

bb14_1:
  br label %bb14

bb14:                                             ; preds = %bb29, %bb2
  %tmp15 = phi i32 [ %tmp30, %bb29 ], [ %tmp3, %bb14_1 ]
  %tmp16 = phi i64 [ %tmp31, %bb29 ], [ 0, %bb14_1 ]
  %tmp17 = phi i64 [ %tmp32, %bb29 ], [ 0, %bb14_1 ]
  %tmp18 = phi i32 [ %tmp33, %bb29 ], [ 0, %bb14_1 ]
  %tmp19 = icmp sgt i32 %tmp15, 0
  br i1 %tmp19, label %bb20_split, label %bb29

bb20_split:                                             
 br label %bb20

bb20:                                             ; preds = %bb20, %bb14
  %tmp21 = phi i64 [ %tmp23, %bb20 ], [ 0, %bb20_split ]
  %tmp22 = phi i32 [ %tmp24, %bb20 ], [ 0, %bb20_split ]
  %tmp23 = add nuw i64 %tmp21, 1
  tail call void @bar()
  %tmp24 = add nuw nsw i32 %tmp22, 1
  %tmp25 = load i32, i32* @c, align 4, !tbaa !2
  %tmp26 = icmp slt i32 %tmp24, %tmp25
  br i1 %tmp26, label %bb20, label %bb27

bb27:                                             ; preds = %bb20
  %tmp28 = add i64 %tmp23, %tmp16
  br label %bb29

bb29:                                             ; preds = %bb27, %bb14
  %tmp30 = phi i32 [ %tmp25, %bb27 ], [ %tmp15, %bb14 ]
  %tmp31 = phi i64 [ %tmp28, %bb27 ], [ %tmp16, %bb14 ]
  %tmp32 = add nuw i64 %tmp17, 1
  %tmp33 = add nuw nsw i32 %tmp18, 1
  %tmp34 = icmp slt i32 %tmp33, %tmp30
  br i1 %tmp34, label %bb14, label %bb35

bb35:                                             ; preds = %bb29
  %tmp36 = insertelement <2 x i64> undef, i64 %tmp31, i32 0
  br label %bb39

bb39:                                             ; preds = %bb35, %bb2
  %tmp40 = phi i32 [ %tmp30, %bb35 ], [ %tmp3, %bb2 ]
  %tmp43 = add nuw nsw i32 %tmp5, 1
  %tmp44 = icmp slt i32 %tmp43, %tmp40
  br i1 %tmp44, label %bb2, label %bb8

bb45:                                             ; preds = %bb67, %bb8
  %tmp46 = phi i32 [ %tmp68, %bb67 ], [ %tmp40, %bb8 ]
  %tmp47 = phi i64 [ %tmp69, %bb67 ], [ 0, %bb8 ]
  %tmp48 = phi i64 [ %tmp70, %bb67 ], [ 0, %bb8 ]
  %tmp49 = phi i32 [ %tmp71, %bb67 ], [ 0, %bb8 ]
  %tmp50 = icmp sgt i32 %tmp46, 0
  br i1 %tmp50, label %bb57, label %bb67

bb51:                                             ; preds = %bb67
  %tmp56 = icmp sgt i32 %tmp68, 0
  br i1 %tmp56, label %bb73, label %bb84

bb57:                                             ; preds = %bb57, %bb45
  %tmp58 = phi i64 [ %tmp60, %bb57 ], [ 0, %bb45 ]
  %tmp59 = phi i32 [ %tmp61, %bb57 ], [ 0, %bb45 ]
  %tmp60 = add nuw i64 %tmp58, 1
  tail call void @bar()
  %tmp61 = add nuw nsw i32 %tmp59, 1
  %tmp62 = load i32, i32* @c, align 4, !tbaa !2
  %tmp63 = mul nsw i32 %tmp62, 10
  %tmp64 = icmp slt i32 %tmp61, %tmp63
  br i1 %tmp64, label %bb57, label %bb65

bb65:                                             ; preds = %bb57
  %tmp66 = add i64 %tmp60, %tmp47
  br label %bb67

bb67:                                             ; preds = %bb65, %bb45
  %tmp68 = phi i32 [ %tmp62, %bb65 ], [ %tmp46, %bb45 ]
  %tmp69 = phi i64 [ %tmp66, %bb65 ], [ %tmp47, %bb45 ]
  %tmp70 = add nuw i64 %tmp48, 1
  %tmp71 = add nuw nsw i32 %tmp49, 1
  %tmp72 = icmp slt i32 %tmp71, %tmp68
  br i1 %tmp72, label %bb45, label %bb51

bb73:                                             ; preds = %bb73, %bb51
  %tmp74 = phi i64 [ %tmp76, %bb73 ], [ 0, %bb51 ]
  %tmp75 = phi i32 [ %tmp77, %bb73 ], [ 0, %bb51 ]
  %tmp76 = add nuw i64 %tmp74, 1
  tail call void @bar()
  %tmp77 = add nuw nsw i32 %tmp75, 1
  %tmp78 = load i32, i32* @c, align 4, !tbaa !2
  %tmp79 = mul nsw i32 %tmp78, 100
  %tmp80 = icmp slt i32 %tmp77, %tmp79
  br i1 %tmp80, label %bb73, label %bb81

bb81:                                             ; preds = %bb73
  br label %bb84

bb84:                                             ; preds = %bb81, %bb51, %bb8, %bb
  ret i32 0
}

attributes #0 = { noinline }
attributes #1 = { norecurse nounwind uwtable } 

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 5.0.0 (trunk 307355)"}
!2 = !{!3, !3, i64 0}
!3 = !{!"int", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
