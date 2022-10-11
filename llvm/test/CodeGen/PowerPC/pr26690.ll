; RUN: llc -verify-machineinstrs -mcpu=pwr8 -mtriple=powerpc64le-unknown-linux-gnu < %s | FileCheck %s

%struct.anon = type { %struct.anon.0, %struct.anon.1 }
%struct.anon.0 = type { i32 }
%struct.anon.1 = type { i32 }

@i = common global i32 0, align 4
@b = common global ptr null, align 8
@c = common global i32 0, align 4
@a = common global i32 0, align 4
@h = common global i32 0, align 4
@g = common global i32 0, align 4
@j = common global i32 0, align 4
@f = common global %struct.anon zeroinitializer, align 4
@d = common global i32 0, align 4
@e = common global i32 0, align 4

; Function Attrs: norecurse nounwind
define signext i32 @fn1(ptr nocapture %p1, i32 signext %p2, ptr nocapture %p3) {
entry:
  %0 = load i32, ptr @i, align 4, !tbaa !1
  %cond = icmp eq i32 %0, 8
  br i1 %cond, label %if.end16, label %while.cond.preheader

while.cond.preheader:                             ; preds = %entry
  %1 = load ptr, ptr @b, align 8, !tbaa !5
  %2 = load i32, ptr %1, align 4, !tbaa !1
  %tobool18 = icmp eq i32 %2, 0
  br i1 %tobool18, label %while.end, label %while.body.lr.ph

while.body.lr.ph:                                 ; preds = %while.cond.preheader
  %.pre = load i32, ptr @c, align 4, !tbaa !1
  br label %while.body

while.body:                                       ; preds = %while.body.backedge, %while.body.lr.ph
  switch i32 %.pre, label %while.body.backedge [
    i32 0, label %sw.bb1
    i32 80, label %sw.bb1
    i32 60, label %sw.bb1
    i32 240, label %while.cond.backedge
  ]

while.body.backedge:                              ; preds = %while.body, %while.cond.backedge
  br label %while.body

sw.bb1:                                           ; preds = %while.body, %while.body, %while.body
  store i32 2, ptr @a, align 4, !tbaa !1
  br label %while.cond.backedge

while.cond.backedge:                              ; preds = %while.body, %sw.bb1
  store i32 4, ptr @a, align 4, !tbaa !1
  %.pre19 = load i32, ptr %1, align 4, !tbaa !1
  %tobool = icmp eq i32 %.pre19, 0
  br i1 %tobool, label %while.end.loopexit, label %while.body.backedge

while.end.loopexit:                               ; preds = %while.cond.backedge
  br label %while.end

while.end:                                        ; preds = %while.end.loopexit, %while.cond.preheader
  %3 = load i32, ptr @h, align 4, !tbaa !1
  %mul = mul nsw i32 %0, %3
  %4 = load i32, ptr @g, align 4, !tbaa !1
  %mul4 = mul nsw i32 %mul, %4
  store i32 %mul4, ptr @j, align 4, !tbaa !1
  %5 = load i32, ptr @f, align 4, !tbaa !7
  %tobool5 = icmp eq i32 %5, 0
  br i1 %tobool5, label %if.end, label %if.then

if.then:                                          ; preds = %while.end
  %div = sdiv i32 %5, %mul
  store i32 %div, ptr @g, align 4, !tbaa !1
  br label %if.end

if.end:                                           ; preds = %while.end, %if.then
  %6 = phi i32 [ %4, %while.end ], [ %div, %if.then ]
  %7 = load i32, ptr getelementptr inbounds (%struct.anon, ptr @f, i64 0, i32 1, i32 0), align 4, !tbaa !10
  %tobool7 = icmp ne i32 %7, 0
  %tobool8 = icmp ne i32 %mul4, 0
  %or.cond = and i1 %tobool7, %tobool8
  %tobool10 = icmp ne i32 %0, 0
  %or.cond17 = and i1 %or.cond, %tobool10
  br i1 %or.cond17, label %if.then11, label %if.end13

if.then11:                                        ; preds = %if.end
  store i32 %3, ptr @d, align 4, !tbaa !1
  %8 = load i32, ptr @e, align 4, !tbaa !1
  store i32 %8, ptr %p3, align 4, !tbaa !1
  %.pre20 = load i32, ptr @g, align 4, !tbaa !1
  br label %if.end13

if.end13:                                         ; preds = %if.then11, %if.end
  %9 = phi i32 [ %.pre20, %if.then11 ], [ %6, %if.end ]
  %tobool14 = icmp eq i32 %9, 0
  br i1 %tobool14, label %if.end16, label %if.then15

if.then15:                                        ; preds = %if.end13
  store i32 %p2, ptr %p1, align 4, !tbaa !1
  br label %if.end16

if.end16:                                         ; preds = %entry, %if.end13, %if.then15
  ret i32 2
}

; CHECK: mfocrf {{[0-9]+}}

!llvm.ident = !{!0}

!0 = !{!"clang version 3.9.0 (trunk 261520)"}
!1 = !{!2, !2, i64 0}
!2 = !{!"int", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
!5 = !{!6, !6, i64 0}
!6 = !{!"any pointer", !3, i64 0}
!7 = !{!8, !2, i64 0}
!8 = !{!"", !9, i64 0, !9, i64 4}
!9 = !{!"", !2, i64 0}
!10 = !{!8, !2, i64 4}
