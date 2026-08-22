; test for https://github.com/llvm/llvm-project/issues/217740

; REQUIRES: asserts

; RUN: opt < %s '-passes=require<block-freq>' -disable-output 2>&1 | count 0
; RUN: llc -O3 -o /dev/null %s

@g1 = dso_local local_unnamed_addr global i8 0, align 1
@g5 = dso_local local_unnamed_addr global i32 0, align 4
@g19 = dso_local local_unnamed_addr global [1 x ptr] zeroinitializer, align 8

; Function Attrs: nofree noreturn nounwind memory(readwrite, argmem: none, target_mem: none) uwtable
define dso_local void @f20(i64 noundef %a4) local_unnamed_addr #0 {
entry:
  %tobool.not = icmp eq i64 %a4, 0
  br i1 %tobool.not, label %if.end, label %lbl_bf2

lbl_bf2.loopexit.split:                           ; preds = %lbl_br52.lr.ph.split
  store ptr %call65, ptr %arrayidx, align 8, !tbaa !9
  br label %lbl_bf2

lbl_bf2:                                          ; preds = %lbl_br52.lr.ph.split.us, %lbl_br52.us, %lbl_br52.us.peel, %lbl_bf2.loopexit.split, %entry
  %a4.addr.0 = phi i64 [ %a4, %entry ], [ %conv12.us110, %lbl_bf2.loopexit.split ], [ 189, %lbl_br52.us ], [ 189, %lbl_br52.us.peel ], [ %conv12.us110, %lbl_br52.lr.ph.split.us ]
  %c11.0 = phi i1 [ undef, %entry ], [ false, %lbl_bf2.loopexit.split ], [ true, %lbl_br52.us ], [ true, %lbl_br52.us.peel ], [ true, %lbl_br52.lr.ph.split.us ]
  %v15.0 = phi i8 [ undef, %entry ], [ %v15.4.ph.fr, %lbl_bf2.loopexit.split ], [ 5, %lbl_br52.us ], [ 5, %lbl_br52.us.peel ], [ 5, %lbl_br52.lr.ph.split.us ]
  %0 = load i8, ptr @g1, align 1, !tbaa !11
  %tobool1 = icmp ne i8 %0, 0
  br label %if.end

if.end:                                           ; preds = %lbl_bf2, %entry
  %a4.addr.1 = phi i64 [ %a4.addr.0, %lbl_bf2 ], [ 0, %entry ]
  %c10.0 = phi i1 [ %tobool1, %lbl_bf2 ], [ undef, %entry ]
  %c11.1 = phi i1 [ %c11.0, %lbl_bf2 ], [ undef, %entry ]
  %v15.1 = phi i8 [ %v15.0, %lbl_bf2 ], [ undef, %entry ]
  br label %lbl_br28

lbl_bf27:                                         ; preds = %if.end25, %lbl_br28
  %a4.addr.2 = phi i64 [ %a4.addr.3, %lbl_br28 ], [ %.us-phi56, %if.end25 ]
  %c5.2 = phi i1 [ %c5.3, %lbl_br28 ], [ %.us-phi55, %if.end25 ]
  %v13.2 = phi i32 [ %v13.3, %lbl_br28 ], [ %.us-phi57, %if.end25 ]
  %v15.2 = phi i8 [ %v15.3, %lbl_br28 ], [ %.us-phi58, %if.end25 ]
  %tobool3 = icmp ne i64 %a4.addr.2, 0
  br label %lbl_b51.preheader

lbl_br28:                                         ; preds = %lbl_br28.backedge, %if.end
  %a4.addr.3 = phi i64 [ %a4.addr.1, %if.end ], [ %a4.addr.3.be, %lbl_br28.backedge ]
  %c5.3 = phi i1 [ true, %if.end ], [ %c5.3.be, %lbl_br28.backedge ]
  %c11.2 = phi i1 [ %c11.1, %if.end ], [ %c11.2.be, %lbl_br28.backedge ]
  %v13.3 = phi i32 [ 5, %if.end ], [ %v13.3.be, %lbl_br28.backedge ]
  %v15.3 = phi i8 [ %v15.1, %if.end ], [ %v15.3.be, %lbl_br28.backedge ]
  br i1 %tobool.not, label %lbl_b51.preheader, label %lbl_bf27

lbl_b51.preheader:                                ; preds = %lbl_bf27, %lbl_br28
  %a4.addr.4.ph = phi i64 [ %a4.addr.3, %lbl_br28 ], [ %a4.addr.2, %lbl_bf27 ]
  %c5.4.ph = phi i1 [ %c5.3, %lbl_br28 ], [ %c5.2, %lbl_bf27 ]
  %c11.3.ph = phi i1 [ %c11.2, %lbl_br28 ], [ %tobool3, %lbl_bf27 ]
  %v13.4.ph = phi i32 [ %v13.3, %lbl_br28 ], [ %v13.2, %lbl_bf27 ]
  %v15.4.ph = phi i8 [ %v15.3, %lbl_br28 ], [ %v15.2, %lbl_bf27 ]
  %v15.4.ph.fr = freeze i8 %v15.4.ph
  %c11.3.ph.fr = freeze i1 %c11.3.ph
  br i1 %c5.4.ph, label %lbl_br52.lr.ph, label %lbl_br28.backedge, !prof !12

lbl_br52.lr.ph:                                   ; preds = %lbl_b51.preheader
  %conv12.us110 = zext i8 %v15.4.ph.fr to i64
  br i1 %c11.3.ph.fr, label %lbl_br52.lr.ph.split.us, label %lbl_br52.lr.ph.split

lbl_br52.lr.ph.split.us:                          ; preds = %lbl_br52.lr.ph
  %call.us111 = tail call align 4 dereferenceable_or_null(4) ptr @aligned_alloc(i64 noundef 4, i64 noundef 4) #2
  %arrayidx.us112 = getelementptr inbounds nuw [8 x i8], ptr @g19, i64 %conv12.us110
  store ptr %call.us111, ptr %arrayidx.us112, align 8, !tbaa !9
  %1 = load i32, ptr @g5, align 4
  %cmp.us114 = icmp eq i32 %1, 5
  br i1 %cmp.us114, label %lbl_bf2, label %if.end19.us.peel

if.end19.us.peel:                                 ; preds = %lbl_br52.lr.ph.split.us
  %conv15.us116.peel = trunc i32 %1 to i8
  %tobool20.us.peel = icmp ne i8 %conv15.us116.peel, 0
  %cmp22.us.peel = icmp eq i32 %1, 4029
  br i1 %cmp22.us.peel, label %lbl_b51.us.peel, label %if.end25

lbl_b51.us.peel:                                  ; preds = %if.end19.us.peel
  br i1 %tobool20.us.peel, label %lbl_br52.us.peel, label %lbl_br28.backedge, !prof !13

lbl_br52.us.peel:                                 ; preds = %lbl_b51.us.peel
  %call.us.peel = tail call align 4 dereferenceable_or_null(4) ptr @aligned_alloc(i64 noundef 4, i64 noundef 4) #2
  store ptr %call.us.peel, ptr getelementptr inbounds nuw (i8, ptr @g19, i64 1512), align 8, !tbaa !9
  %2 = load i32, ptr @g5, align 4
  %cmp.us.peel = icmp eq i32 %2, 5
  br i1 %cmp.us.peel, label %lbl_bf2, label %if.end19.us

lbl_br52.us:                                      ; preds = %lbl_b51.us
  %call.us = tail call align 4 dereferenceable_or_null(4) ptr @aligned_alloc(i64 noundef 4, i64 noundef 4) #2
  store ptr %call.us, ptr getelementptr inbounds nuw (i8, ptr @g19, i64 1512), align 8, !tbaa !9
  %3 = load i32, ptr @g5, align 4
  %cmp.us = icmp eq i32 %3, 5
  br i1 %cmp.us, label %lbl_bf2, label %if.end19.us, !llvm.loop !14

lbl_b51.us:                                       ; preds = %if.end19.us
  br i1 %tobool20.us, label %lbl_br52.us, label %lbl_br28.backedge, !prof !13

if.end19.us:                                      ; preds = %lbl_br52.us.peel, %lbl_br52.us
  %4 = phi i32 [ %3, %lbl_br52.us ], [ %2, %lbl_br52.us.peel ]
  %conv15.us116 = trunc i32 %4 to i8
  %tobool20.us = icmp ne i8 %conv15.us116, 0
  %cmp22.us = icmp eq i32 %4, 4029
  br i1 %cmp22.us, label %lbl_b51.us, label %if.end25

lbl_br52.lr.ph.split:                             ; preds = %lbl_br52.lr.ph
  %arrayidx = getelementptr inbounds nuw [8 x i8], ptr @g19, i64 %conv12.us110
  %call65 = tail call align 4 dereferenceable_or_null(4) ptr @aligned_alloc(i64 noundef 4, i64 noundef 4) #2
  %cmp66 = icmp eq i32 %v13.4.ph, 5
  br i1 %cmp66, label %lbl_bf2.loopexit.split, label %if.end19.lr.ph

if.end19.lr.ph:                                   ; preds = %lbl_br52.lr.ph.split
  %tobool20.not = icmp ne i8 %v15.4.ph.fr, 0
  %cmp22.us82104 = icmp eq i32 %v13.4.ph, 4029
  br i1 %tobool20.not, label %if.end19.lr.ph.split.us.split, label %if.end19.lr.ph.split, !prof !13

if.end19.lr.ph.split.us.split:                    ; preds = %if.end19.lr.ph
  br i1 %cmp22.us82104, label %lbl_b51.us83.us, label %if.end25.split

lbl_b51.us83.us:                                  ; preds = %if.end19.lr.ph.split.us.split, %lbl_b51.us83.us
  br label %lbl_b51.us83.us

if.end19.lr.ph.split:                             ; preds = %if.end19.lr.ph
  br i1 %cmp22.us82104, label %lbl_b51, label %if.end25.split

lbl_b51:                                          ; preds = %if.end19.lr.ph.split
  store ptr %call65, ptr %arrayidx, align 8, !tbaa !9
  br label %lbl_br28.backedge

lbl_br28.backedge:                                ; preds = %lbl_b51.us.peel, %lbl_b51.us, %lbl_b51, %lbl_b51.preheader, %if.end25
  %a4.addr.3.be = phi i64 [ %.us-phi56, %if.end25 ], [ %a4.addr.4.ph, %lbl_b51.preheader ], [ 0, %lbl_b51 ], [ %conv12.us110, %lbl_b51.us.peel ], [ 189, %lbl_b51.us ]
  %c5.3.be = phi i1 [ %.us-phi55, %if.end25 ], [ false, %lbl_b51.preheader ], [ false, %lbl_b51 ], [ false, %lbl_b51.us ], [ false, %lbl_b51.us.peel ]
  %c11.2.be = phi i1 [ %c11.3.ph.fr, %if.end25 ], [ %c11.3.ph.fr, %lbl_b51.preheader ], [ false, %lbl_b51 ], [ true, %lbl_b51.us ], [ true, %lbl_b51.us.peel ]
  %v13.3.be = phi i32 [ %.us-phi57, %if.end25 ], [ %v13.4.ph, %lbl_b51.preheader ], [ 4029, %lbl_b51 ], [ 4029, %lbl_b51.us ], [ 4029, %lbl_b51.us.peel ]
  %v15.3.be = phi i8 [ %.us-phi58, %if.end25 ], [ %v15.4.ph.fr, %lbl_b51.preheader ], [ 0, %lbl_b51 ], [ 0, %lbl_b51.us ], [ 0, %lbl_b51.us.peel ]
  br label %lbl_br28

if.end25.split:                                   ; preds = %if.end19.lr.ph.split, %if.end19.lr.ph.split.us.split
  store ptr %call65, ptr %arrayidx, align 8, !tbaa !9
  br label %if.end25

if.end25:                                         ; preds = %if.end19.us.peel, %if.end19.us, %if.end25.split
  %.us-phi55 = phi i1 [ %tobool20.not, %if.end25.split ], [ %tobool20.us.peel, %if.end19.us.peel ], [ %tobool20.us, %if.end19.us ]
  %.us-phi56 = phi i64 [ %conv12.us110, %if.end25.split ], [ %conv12.us110, %if.end19.us.peel ], [ 189, %if.end19.us ]
  %.us-phi57 = phi i32 [ %v13.4.ph, %if.end25.split ], [ %1, %if.end19.us.peel ], [ %4, %if.end19.us ]
  %.us-phi58 = phi i8 [ %v15.4.ph.fr, %if.end25.split ], [ %conv15.us116.peel, %if.end19.us.peel ], [ %conv15.us116, %if.end19.us ]
  br i1 %c10.0, label %lbl_bf27, label %lbl_br28.backedge
}

; Function Attrs: mustprogress nofree nounwind willreturn allockind("alloc,uninitialized,aligned") allocsize(1) memory(inaccessiblemem: readwrite, errnomem: write)
declare noalias noundef ptr @aligned_alloc(i64 allocalign noundef, i64 noundef) local_unnamed_addr #1

attributes #0 = { nofree noreturn nounwind memory(readwrite, argmem: none, target_mem: none) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { mustprogress nofree nounwind willreturn allockind("alloc,uninitialized,aligned") allocsize(1) memory(inaccessiblemem: readwrite, errnomem: write) "alloc-family"="malloc" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { allocsize(1) }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}
!llvm.errno.tbaa = !{!4}

!0 = !{i32 8, !"PIC Level", i32 2}
!1 = !{i32 7, !"PIE Level", i32 2}
!2 = !{i32 7, !"uwtable", i32 2}
!3 = !{!"clang version 24.0.0git (https://github.com/im-lunex/llvm-project 6978738e1efe0c33a9540b70697ce0c83852d9c7)"}
!4 = !{!5, !6, i64 0}
!5 = !{!"__libc_errno", !6, i64 0}
!6 = !{!"int", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C/C++ TBAA"}
!9 = !{!10, !10, i64 0}
!10 = !{!"any pointer", !7, i64 0}
!11 = !{!7, !7, i64 0}
!12 = !{!"branch_weights", i32 128849020, i32 1889785609}
!13 = !{!"branch_weights", i32 0, i32 128849020}
!14 = distinct !{!14, !15}
!15 = !{!"llvm.loop.peeled.count", i32 1}
