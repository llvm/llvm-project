; RUN: opt %loadNPMPolly '-passes=polly<no-default-opts>' -S %s | FileCheck %s
;
; https://github.com/llvm/llvm-project/issues/205137
; visitAddRecExpr cause infinite recursion when visitUnknown follow VMa
; and GenSE.getSCEV() back to AddRec.
;
; CHECK: polly.split_new_and_old

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@a = dso_local local_unnamed_addr global i64 0, align 8
@b = dso_local local_unnamed_addr global i16 0, align 2

define dso_local void @_Z1cisPiPA5_A5_xPS0_Pc(i32 noundef %d, i16 noundef signext %e, ptr nofree noundef readonly captures(none) %f, ptr nofree noundef readonly captures(none) %g, ptr nofree noundef readonly captures(none) %h, ptr nofree noundef readnone captures(none) %i) local_unnamed_addr {
entry:
  %a.promoted39 = load i64, ptr @a, align 8
  %tobool38.not = icmp eq i16 %e, 0
  br label %for.cond1.preheader

for.cond1.preheader:
  %conv45 = phi i64 [ 0, %entry ], [ %sext, %for.cond56.preheader.1 ]
  %a.promoted4042 = phi i64 [ %a.promoted39, %entry ], [ %.sroa.speculated.14.1, %for.cond56.preheader.1 ]
  %invariant.gep = getelementptr [8 x i8], ptr %h, i64 %conv45
  %arrayidx25 = getelementptr inbounds nuw [200 x i8], ptr %g, i64 %conv45
  %arrayidx27 = getelementptr inbounds nuw [40 x i8], ptr %arrayidx25, i64 %conv45
  %arrayidx29 = getelementptr inbounds nuw [8 x i8], ptr %arrayidx27, i64 %conv45
  br label %for.cond19.preheader

for.cond.cleanup:
  store i16 %conv47.1, ptr @b, align 2, !tbaa !8
  ret void

for.cond19.preheader:
  %conv1532 = phi i64 [ 0, %for.cond1.preheader ], [ %sext17, %for.cond19.preheader ]
  %a.promoted3031 = phi i64 [ %a.promoted4042, %for.cond1.preheader ], [ %.sroa.speculated.14, %for.cond19.preheader ]
  %0 = load i64, ptr %arrayidx29, align 8, !tbaa !10
  %.sroa.speculated = tail call i64 @llvm.umax.i64(i64 %a.promoted3031, i64 %0)
  store i64 %.sroa.speculated, ptr @a, align 8, !tbaa !10
  %1 = load i64, ptr %arrayidx29, align 8, !tbaa !10
  %.sroa.speculated.1 = tail call i64 @llvm.umax.i64(i64 %.sroa.speculated, i64 %1)
  store i64 %.sroa.speculated.1, ptr @a, align 8, !tbaa !10
  %2 = load i64, ptr %arrayidx29, align 8, !tbaa !10
  %.sroa.speculated.2 = tail call i64 @llvm.umax.i64(i64 %.sroa.speculated.1, i64 %2)
  store i64 %.sroa.speculated.2, ptr @a, align 8, !tbaa !10
  %3 = load i64, ptr %arrayidx29, align 8, !tbaa !10
  %.sroa.speculated.3 = tail call i64 @llvm.umax.i64(i64 %.sroa.speculated.2, i64 %3)
  store i64 %.sroa.speculated.3, ptr @a, align 8, !tbaa !10
  %4 = load i64, ptr %arrayidx29, align 8, !tbaa !10
  %.sroa.speculated.4 = tail call i64 @llvm.umax.i64(i64 %.sroa.speculated.3, i64 %4)
  store i64 %.sroa.speculated.4, ptr @a, align 8, !tbaa !10
  %5 = load i64, ptr %arrayidx29, align 8, !tbaa !10
  %.sroa.speculated.5 = tail call i64 @llvm.umax.i64(i64 %.sroa.speculated.4, i64 %5)
  store i64 %.sroa.speculated.5, ptr @a, align 8, !tbaa !10
  %6 = load i64, ptr %arrayidx29, align 8, !tbaa !10
  %.sroa.speculated.6 = tail call i64 @llvm.umax.i64(i64 %.sroa.speculated.5, i64 %6)
  store i64 %.sroa.speculated.6, ptr @a, align 8, !tbaa !10
  %7 = load i64, ptr %arrayidx29, align 8, !tbaa !10
  %.sroa.speculated.7 = tail call i64 @llvm.umax.i64(i64 %.sroa.speculated.6, i64 %7)
  store i64 %.sroa.speculated.7, ptr @a, align 8, !tbaa !10
  %8 = load i64, ptr %arrayidx29, align 8, !tbaa !10
  %.sroa.speculated.8 = tail call i64 @llvm.umax.i64(i64 %.sroa.speculated.7, i64 %8)
  store i64 %.sroa.speculated.8, ptr @a, align 8, !tbaa !10
  %9 = load i64, ptr %arrayidx29, align 8, !tbaa !10
  %.sroa.speculated.9 = tail call i64 @llvm.umax.i64(i64 %.sroa.speculated.8, i64 %9)
  store i64 %.sroa.speculated.9, ptr @a, align 8, !tbaa !10
  %10 = load i64, ptr %arrayidx29, align 8, !tbaa !10
  %.sroa.speculated.10 = tail call i64 @llvm.umax.i64(i64 %.sroa.speculated.9, i64 %10)
  store i64 %.sroa.speculated.10, ptr @a, align 8, !tbaa !10
  %11 = load i64, ptr %arrayidx29, align 8, !tbaa !10
  %.sroa.speculated.11 = tail call i64 @llvm.umax.i64(i64 %.sroa.speculated.10, i64 %11)
  store i64 %.sroa.speculated.11, ptr @a, align 8, !tbaa !10
  %12 = load i64, ptr %arrayidx29, align 8, !tbaa !10
  %.sroa.speculated.12 = tail call i64 @llvm.umax.i64(i64 %.sroa.speculated.11, i64 %12)
  store i64 %.sroa.speculated.12, ptr @a, align 8, !tbaa !10
  %13 = load i64, ptr %arrayidx29, align 8, !tbaa !10
  %.sroa.speculated.13 = tail call i64 @llvm.umax.i64(i64 %.sroa.speculated.12, i64 %13)
  store i64 %.sroa.speculated.13, ptr @a, align 8, !tbaa !10
  %14 = load i64, ptr %arrayidx29, align 8, !tbaa !10
  %.sroa.speculated.14 = tail call i64 @llvm.umax.i64(i64 %.sroa.speculated.13, i64 %14)
  store i64 %.sroa.speculated.14, ptr @a, align 8, !tbaa !10
  %sext17 = add nuw nsw i64 %conv1532, 12884901888
  %cmp16 = icmp samesign ult i64 %conv1532, 51539607552
  br i1 %cmp16, label %for.cond19.preheader, label %for.cond14.preheader.1, !llvm.loop !12

for.cond14.preheader.1:
  %conv.pn.1 = select i1 %tobool38.not, i64 4, i64 %conv45
  %gep.1 = getelementptr [40 x i8], ptr %invariant.gep, i64 %conv.pn.1
  br label %for.cond19.preheader.1

for.cond19.preheader.1:
  %conv1532.1 = phi i64 [ 0, %for.cond14.preheader.1 ], [ %sext17.1, %for.cond19.preheader.1 ]
  %a.promoted3031.1 = phi i64 [ %.sroa.speculated.14, %for.cond14.preheader.1 ], [ %.sroa.speculated.14.1, %for.cond19.preheader.1 ]
  %15 = load i64, ptr %arrayidx29, align 8, !tbaa !10
  %.sroa.speculated.147 = tail call i64 @llvm.umax.i64(i64 %a.promoted3031.1, i64 %15)
  store i64 %.sroa.speculated.147, ptr @a, align 8, !tbaa !10
  %16 = load i64, ptr %arrayidx29, align 8, !tbaa !10
  %.sroa.speculated.1.1 = tail call i64 @llvm.umax.i64(i64 %.sroa.speculated.147, i64 %16)
  store i64 %.sroa.speculated.1.1, ptr @a, align 8, !tbaa !10
  %17 = load i64, ptr %arrayidx29, align 8, !tbaa !10
  %.sroa.speculated.2.1 = tail call i64 @llvm.umax.i64(i64 %.sroa.speculated.1.1, i64 %17)
  store i64 %.sroa.speculated.2.1, ptr @a, align 8, !tbaa !10
  %18 = load i64, ptr %arrayidx29, align 8, !tbaa !10
  %.sroa.speculated.3.1 = tail call i64 @llvm.umax.i64(i64 %.sroa.speculated.2.1, i64 %18)
  store i64 %.sroa.speculated.3.1, ptr @a, align 8, !tbaa !10
  %19 = load i64, ptr %arrayidx29, align 8, !tbaa !10
  %.sroa.speculated.4.1 = tail call i64 @llvm.umax.i64(i64 %.sroa.speculated.3.1, i64 %19)
  store i64 %.sroa.speculated.4.1, ptr @a, align 8, !tbaa !10
  %20 = load i64, ptr %arrayidx29, align 8, !tbaa !10
  %.sroa.speculated.5.1 = tail call i64 @llvm.umax.i64(i64 %.sroa.speculated.4.1, i64 %20)
  store i64 %.sroa.speculated.5.1, ptr @a, align 8, !tbaa !10
  %21 = load i64, ptr %arrayidx29, align 8, !tbaa !10
  %.sroa.speculated.6.1 = tail call i64 @llvm.umax.i64(i64 %.sroa.speculated.5.1, i64 %21)
  store i64 %.sroa.speculated.6.1, ptr @a, align 8, !tbaa !10
  %22 = load i64, ptr %arrayidx29, align 8, !tbaa !10
  %.sroa.speculated.7.1 = tail call i64 @llvm.umax.i64(i64 %.sroa.speculated.6.1, i64 %22)
  store i64 %.sroa.speculated.7.1, ptr @a, align 8, !tbaa !10
  %23 = load i64, ptr %arrayidx29, align 8, !tbaa !10
  %.sroa.speculated.8.1 = tail call i64 @llvm.umax.i64(i64 %.sroa.speculated.7.1, i64 %23)
  store i64 %.sroa.speculated.8.1, ptr @a, align 8, !tbaa !10
  %24 = load i64, ptr %arrayidx29, align 8, !tbaa !10
  %.sroa.speculated.9.1 = tail call i64 @llvm.umax.i64(i64 %.sroa.speculated.8.1, i64 %24)
  store i64 %.sroa.speculated.9.1, ptr @a, align 8, !tbaa !10
  %25 = load i64, ptr %arrayidx29, align 8, !tbaa !10
  %.sroa.speculated.10.1 = tail call i64 @llvm.umax.i64(i64 %.sroa.speculated.9.1, i64 %25)
  store i64 %.sroa.speculated.10.1, ptr @a, align 8, !tbaa !10
  %26 = load i64, ptr %arrayidx29, align 8, !tbaa !10
  %.sroa.speculated.11.1 = tail call i64 @llvm.umax.i64(i64 %.sroa.speculated.10.1, i64 %26)
  store i64 %.sroa.speculated.11.1, ptr @a, align 8, !tbaa !10
  %27 = load i64, ptr %arrayidx29, align 8, !tbaa !10
  %.sroa.speculated.12.1 = tail call i64 @llvm.umax.i64(i64 %.sroa.speculated.11.1, i64 %27)
  store i64 %.sroa.speculated.12.1, ptr @a, align 8, !tbaa !10
  %28 = load i64, ptr %arrayidx29, align 8, !tbaa !10
  %.sroa.speculated.13.1 = tail call i64 @llvm.umax.i64(i64 %.sroa.speculated.12.1, i64 %28)
  store i64 %.sroa.speculated.13.1, ptr @a, align 8, !tbaa !10
  %29 = load i64, ptr %arrayidx29, align 8, !tbaa !10
  %.sroa.speculated.14.1 = tail call i64 @llvm.umax.i64(i64 %.sroa.speculated.13.1, i64 %29)
  store i64 %.sroa.speculated.14.1, ptr @a, align 8, !tbaa !10
  %sext17.1 = add nuw nsw i64 %conv1532.1, 12884901888
  %cmp16.1 = icmp samesign ult i64 %conv1532.1, 51539607552
  br i1 %cmp16.1, label %for.cond19.preheader.1, label %for.cond56.preheader.1, !llvm.loop !12

for.cond56.preheader.1:
  %cond.1 = load i64, ptr %gep.1, align 8, !tbaa !10
  %conv47.1 = trunc i64 %cond.1 to i16
  %sext = add nuw nsw i64 %conv45, 1
  %exitcond.not = icmp eq i64 %sext, 5
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.cond1.preheader, !llvm.loop !14
}

declare i64 @llvm.umax.i64(i64, i64)

!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C++ TBAA"}
!8 = !{!9, !9, i64 0}
!9 = !{!"short", !6, i64 0}
!10 = !{!11, !11, i64 0}
!11 = !{!"long long", !6, i64 0}
!12 = distinct !{!12, !13}
!13 = !{!"llvm.loop.mustprogress"}
!14 = distinct !{!14, !13}
