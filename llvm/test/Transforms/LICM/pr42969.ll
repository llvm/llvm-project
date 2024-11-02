; RUN: opt %s -S -licm | FileCheck %s

define i16 @main(i1 %a_b_mayalias, ptr %a, ptr %b) {
; CHECK:       scalar.body:
; CHECK-NEXT:    [[J:%.*]] = phi i64
; CHECK-NEXT:    [[TMP3:%.*]] = load i16
; CHECK-NEXT:    [[RESULT:%.*]] = add i16 [[TMP3]], 1
; CHECK-NEXT:    store i16 [[RESULT]]

entry:
  br label %outer

outer:                                            ; preds = %scalar.cleanup, %entry
; 4 = MemoryPhi({entry,liveOnEntry},{scalar.cleanup,2})
  %i = phi i16 [ 0, %entry ], [ %i.next, %scalar.cleanup ]
  br i1 %a_b_mayalias, label %scalar.ph, label %vector.ph

vector.ph:                                        ; preds = %outer
; MemoryUse(4) MayAlias
  %tmp1 = load i16, ptr %a, align 1, !alias.scope !0, !tbaa !7
  %tmp2 = add i16 %tmp1, 1
; 1 = MemoryDef(4)
  store i16 %tmp2, ptr %b, align 1, !alias.scope !3, !noalias !0, !tbaa !7
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ %index.next, %vector.body ], [ 0, %vector.ph ]
  %index.next = add i64 %index, 1
  %cmp1 = icmp eq i64 %index.next, 16
  br i1 %cmp1, label %middle.block, label %vector.body

middle.block:                                     ; preds = %vector.body
  br label %scalar.ph

scalar.ph:                                        ; preds = %middle.block, %outer
; 5 = MemoryPhi({outer,4},{middle.block,1})
  %j.start = phi i64 [ 0, %outer ], [ 16, %middle.block ]
  br label %scalar.body

scalar.body:                                      ; preds = %scalar.body, %scalar.ph
; 3 = MemoryPhi({scalar.ph,5},{scalar.body,2})
  %j = phi i64 [ %j.next, %scalar.body ], [ %j.start, %scalar.ph ]
; MemoryUse(3) MayAlias
  %tmp3 = load i16, ptr %a, align 1, !tbaa !7
  %result = add i16 %tmp3, 1
; 2 = MemoryDef(3)
  store i16 %result, ptr %b, align 1, !tbaa !7
  %j.next = add nuw nsw i64 %j, 1
  %cmp2 = icmp ult i64 %j.next, 20
  br i1 %cmp2, label %scalar.body, label %scalar.cleanup

scalar.cleanup:                                   ; preds = %scalar.body
  %result.lcssa = phi i16 [ %result, %scalar.body ]
  %i.next = add nuw nsw i16 %i, 1
  %exitcond = icmp eq i16 %i.next, 10
  br i1 %exitcond, label %exit.block, label %outer

exit.block:                                       ; preds = %scalar.cleanup
  %result.lcssa.lcssa = phi i16 [ %result.lcssa, %scalar.cleanup ]
  ret i16 %result.lcssa.lcssa
}

!0 = !{!1}
!1 = distinct !{!1, !2}
!2 = distinct !{!2, !"LVerDomain"}
!3 = !{!4}
!4 = distinct !{!4, !2}
!5 = !{!"Simple C++ TBAA"}
!6 = !{!"omnipotent char", !5, i64 0}
!7 = !{!6, !6, i64 0}
