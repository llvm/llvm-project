; RUN: opt %loadPolly -polly-stmt-granularity=scalar-indep -polly-print-instructions -polly-print-scops -disable-output < %s | FileCheck %s -match-full-lines

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"

@b = dso_local local_unnamed_addr global i32 1, align 4
@e = dso_local local_unnamed_addr global i32 3, align 4
@a = common dso_local local_unnamed_addr global [56 x i32] zeroinitializer, align 16
@f = common dso_local local_unnamed_addr global i16 0, align 2
@d = common dso_local local_unnamed_addr global i8 0, align 1

; Function Attrs: nounwind uwtable
define dso_local i32 @func() {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body, %entry.split
  %indvars.iv = phi i64 [ 0, %entry.split ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds [56 x i32], ptr @a, i64 0, i64 %indvars.iv
  %0 = trunc i64 %indvars.iv to i32
  store i32 %0, ptr %arrayidx, align 4, !tbaa !0
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 56
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  %1 = load i32, ptr @e, align 4, !tbaa !0
  store i32 2, ptr @e, align 4, !tbaa !0
  %2 = trunc i32 %1 to i16
  %conv = and i16 %2, 1
  %tobool = icmp eq i16 %conv, 0
  br label %for.body3

for.body3:                                        ; preds = %for.end, %for.inc11
  %storemerge20 = phi i32 [ 2, %for.end ], [ %dec12, %for.inc11 ]
  %3 = load i8, ptr @d, align 1
  %cmp6 = icmp eq i8 %3, 8
  %or.cond = or i1 %tobool, %cmp6
  br i1 %or.cond, label %for.inc11, label %for.inc11.loopexit

for.inc11.loopexit:                               ; preds = %for.body3
  store i32 0, ptr @b, align 4, !tbaa !0
  store i8 8, ptr @d, align 1, !tbaa !4
  br label %for.inc11

for.inc11:                                        ; preds = %for.inc11.loopexit, %for.body3
  %dec12 = add nsw i32 %storemerge20, -1
  %cmp2 = icmp sgt i32 %storemerge20, -18
  br i1 %cmp2, label %for.body3, label %for.end13

for.end13:                                        ; preds = %for.inc11
  store i16 %conv, ptr @f, align 2, !tbaa !5
  store i32 -19, ptr @e, align 4, !tbaa !0
  %4 = load i32, ptr @b, align 4, !tbaa !0
  ret i32 %4
}

!0 = !{!1, !1, i64 0}
!1 = !{!"int", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
!4 = !{!2, !2, i64 0}
!5 = !{!6, !6, i64 0}
!6 = !{!"short", !2, i64 0}

; CHECK:    	Stmt_for_end_a
; CHECK-NEXT:            Domain :=
; CHECK-NEXT:                { Stmt_for_end_a[] };
; CHECK-NEXT:            Schedule :=
; CHECK-NEXT:                { Stmt_for_end_a[] -> [1, 0] };
; CHECK-NEXT:            ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                { Stmt_for_end_a[] -> MemRef_e[0] };
; CHECK-NEXT:            MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                { Stmt_for_end_a[] -> MemRef_tobool[] };
; CHECK-NEXT:            MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                { Stmt_for_end_a[] -> MemRef_conv[] };
; CHECK-NEXT:            Instructions {
; CHECK-NEXT:                  %1 = load i32, ptr @e, align 4, !tbaa !0
; CHECK-NEXT:                  %2 = trunc i32 %1 to i16
; CHECK-NEXT:                  %conv = and i16 %2, 1
; CHECK-NEXT:                  %tobool = icmp eq i16 %conv, 0
; CHECK-NEXT:            }
; CHECK-NEXT:    	Stmt_for_end
; CHECK-NEXT:            Domain :=
; CHECK-NEXT:                { Stmt_for_end[] };
; CHECK-NEXT:            Schedule :=
; CHECK-NEXT:                { Stmt_for_end[] -> [2, 0] };
; CHECK-NEXT:            MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                { Stmt_for_end[] -> MemRef_e[0] };
; CHECK-NEXT:            Instructions {
; CHECK-NEXT:                  store i32 2, ptr @e, align 4, !tbaa !0
; CHECK-NEXT:            }
