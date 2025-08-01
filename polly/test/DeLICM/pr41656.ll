; RUN: opt %loadNPMPolly '-passes=print<polly-function-scops>,scop(print<polly-delicm>)' -disable-output < %s 2>&1 | FileCheck %s
;
; llvm.org/PR41656
;
; This test case has an InvalidContext such that part of the predecessors
; of for.body.us.i lie within the invalid context. This causes a
; consistency check within the invalid context of PR41656 to fail.
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define dso_local void @main() local_unnamed_addr #0 {
entry:
  %call24 = tail call i32 @av_get_channel_layout_nb_channels() #2
  br label %if.end30

if.end30:                                         ; preds = %entry
  br i1 undef, label %if.then40, label %do.body.preheader

do.body.preheader:                                ; preds = %if.end30
  %idx.ext.i = sext i32 %call24 to i64
  %wide.trip.count.i = zext i32 %call24 to i64
  %0 = load ptr, ptr undef, align 8, !tbaa !1
  br label %for.body.us.preheader.i

if.then40:                                        ; preds = %if.end30
  unreachable

for.body.us.preheader.i:                          ; preds = %do.body.preheader
  br i1 false, label %for.body.us.i.us, label %for.body.us.i

for.body.us.i.us:                                 ; preds = %for.body.us.preheader.i
  br label %fill_samples.exit

for.body.us.i:                                    ; preds = %for.cond2.for.end_crit_edge.us.i, %for.body.us.preheader.i
  %t.1 = phi double [ undef, %for.cond2.for.end_crit_edge.us.i ], [ 0.000000e+00, %for.body.us.preheader.i ]
  %i.05.us.i = phi i32 [ %inc8.us.i, %for.cond2.for.end_crit_edge.us.i ], [ 0, %for.body.us.preheader.i ]
  %dstp.03.us.i = phi ptr [ %add.ptr.us.i, %for.cond2.for.end_crit_edge.us.i ], [ %0, %for.body.us.preheader.i ]
  %mul.us.i = fmul nsz double %t.1, 0x40A59933FC6A96C1
  %1 = call nsz double @llvm.sin.f64(double %mul.us.i) #2
  store double %1, ptr %dstp.03.us.i, align 8, !tbaa !5
  br label %for.body5.us.for.body5.us_crit_edge.i

for.body5.us.for.body5.us_crit_edge.i:            ; preds = %for.body5.us.for.body5.us_crit_edge.i.for.body5.us.for.body5.us_crit_edge.i_crit_edge, %for.body.us.i
  %indvars.iv.next.i66 = phi i64 [ 2, %for.body.us.i ], [ %indvars.iv.next.i, %for.body5.us.for.body5.us_crit_edge.i.for.body5.us.for.body5.us_crit_edge.i_crit_edge ]
  %indvars.iv.next.i = add nuw nsw i64 %indvars.iv.next.i66, 1
  %exitcond.i = icmp eq i64 %indvars.iv.next.i, %wide.trip.count.i
  br i1 %exitcond.i, label %for.cond2.for.end_crit_edge.us.i, label %for.body5.us.for.body5.us_crit_edge.i.for.body5.us.for.body5.us_crit_edge.i_crit_edge

for.body5.us.for.body5.us_crit_edge.i.for.body5.us.for.body5.us_crit_edge.i_crit_edge: ; preds = %for.body5.us.for.body5.us_crit_edge.i
  %.pre10.i.pre = load i64, ptr %dstp.03.us.i, align 8, !tbaa !5
  br label %for.body5.us.for.body5.us_crit_edge.i

for.cond2.for.end_crit_edge.us.i:                 ; preds = %for.body5.us.for.body5.us_crit_edge.i
  %add.ptr.us.i = getelementptr inbounds double, ptr %dstp.03.us.i, i64 %idx.ext.i
  %inc8.us.i = add nuw nsw i32 %i.05.us.i, 1
  %exitcond7.i = icmp eq i32 %inc8.us.i, 1024
  br i1 %exitcond7.i, label %fill_samples.exit, label %for.body.us.i

fill_samples.exit:                                ; preds = %for.cond2.for.end_crit_edge.us.i, %for.body.us.i.us
  ret void
}

declare dso_local i32 @av_get_channel_layout_nb_channels() local_unnamed_addr #0

; Function Attrs: nounwind readnone speculatable
declare double @llvm.sin.f64(double) #1

attributes #0 = { "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { nounwind }

!llvm.ident = !{!0}

!0 = !{!"clang version 9.0.0 (https://github.com/llvm/llvm-project.git 2436237895b70ed44cf256f67eb2f74e147eb559)"}
!1 = !{!2, !2, i64 0}
!2 = !{!"any pointer", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
!5 = !{!6, !6, i64 0}
!6 = !{!"double", !3, i64 0}


; CHECK:      Invalid Context:
; CHECK-NEXT: [call24] -> {  : false }
; CHECK:      Defined Behavior Context:
; CHECK-NEXT: [call24] -> {  : 3 <= call24 <= 2147483647 }

; Only write to scalar if call24 >= 3 (i.e. not in invalid context)
; Since it should be never executed otherwise, the condition is not strictly necessary.
; CHECK-LABEL: DeLICM result:
; CHECK:          Stmt_for_body_us_preheader_i
; CHECK-NEXT:            MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 [call24] -> { Stmt_for_body_us_preheader_i[] -> MemRef_t_1__phi[] };
; CHECK-NEXT:            new: [call24] -> { Stmt_for_body_us_preheader_i[] -> MemRef1[0, 0] : call24 >= 3 };
