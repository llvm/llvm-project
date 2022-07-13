; ModuleID = 'loop_fuse.ll'
source_filename = "loop_fuse.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind uwtable
define dso_local void @init(i32* noundef %a, i32* noundef %b, i32* noundef %c, i32 noundef %n) #0 {
entry:
  %smax = call i32 @llvm.smax.i32(i32 %n, i32 0)
  %wide.trip.count = zext i32 %smax to i64
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %exitcond = icmp ne i64 %indvars.iv, %wide.trip.count
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %0 = shl nuw nsw i64 %indvars.iv, 1
  %arrayidx = getelementptr inbounds i32, i32* %c, i64 %indvars.iv
  %1 = trunc i64 %0 to i32
  store i32 %1, i32* %arrayidx, align 4
  %2 = trunc i64 %indvars.iv to i32
  %mul = mul nsw i32 %2, %2
  %arrayidx2 = getelementptr inbounds i32, i32* %b, i64 %indvars.iv
  store i32 %mul, i32* %arrayidx2, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond, !llvm.loop !4

for.end:                                          ; preds = %for.cond
  br label %for.cond4

for.cond4:                                        ; preds = %for.inc14, %for.end
  %indvars.iv8 = phi i64 [ %indvars.iv.next9, %for.inc14 ], [ 0, %for.end ]
  %exitcond11 = icmp ne i64 %indvars.iv8, 10
  br i1 %exitcond11, label %for.body6, label %for.end16

for.body6:                                        ; preds = %for.cond4
  %arrayidx8 = getelementptr inbounds i32, i32* %b, i64 %indvars.iv8
  %i = load i32, i32* %arrayidx8, align 4
  %arrayidx10 = getelementptr inbounds i32, i32* %c, i64 %indvars.iv8
  %i1 = load i32, i32* %arrayidx10, align 4
  %add11 = add nsw i32 %i, %i1
  %arrayidx13 = getelementptr inbounds i32, i32* %a, i64 %indvars.iv8
  store i32 %add11, i32* %arrayidx13, align 4
  br label %for.inc14

for.inc14:                                        ; preds = %for.body6
  %indvars.iv.next9 = add nuw nsw i64 %indvars.iv8, 1
  br label %for.cond4, !llvm.loop !6

for.end16:                                        ; preds = %for.cond4
  br label %for.cond18

for.cond18:                                       ; preds = %for.inc28, %for.end16
  %indvars.iv12 = phi i64 [ %indvars.iv.next13, %for.inc28 ], [ 0, %for.end16 ]
  %exitcond15 = icmp ne i64 %indvars.iv12, 10
  br i1 %exitcond15, label %for.body20, label %for.end30

for.body20:                                       ; preds = %for.cond18
  %arrayidx22 = getelementptr inbounds i32, i32* %b, i64 %indvars.iv12
  %i2 = load i32, i32* %arrayidx22, align 4
  %arrayidx24 = getelementptr inbounds i32, i32* %c, i64 %indvars.iv12
  %i3 = load i32, i32* %arrayidx24, align 4
  %add25 = add nsw i32 %i2, %i3
  %arrayidx27 = getelementptr inbounds i32, i32* %a, i64 %indvars.iv12
  store i32 %add25, i32* %arrayidx27, align 4
  br label %for.inc28

for.inc28:                                        ; preds = %for.body20
  %indvars.iv.next13 = add nuw nsw i64 %indvars.iv12, 1
  br label %for.cond18, !llvm.loop !7

for.end30:                                        ; preds = %for.cond18
  %wide.trip.count19 = zext i32 %smax to i64
  br label %for.cond32

for.cond32:                                       ; preds = %for.inc42, %for.end30
  %indvars.iv16 = phi i64 [ %indvars.iv.next17, %for.inc42 ], [ 0, %for.end30 ]
  %exitcond20 = icmp ne i64 %indvars.iv16, %wide.trip.count19
  br i1 %exitcond20, label %for.body34, label %for.end44

for.body34:                                       ; preds = %for.cond32
  %arrayidx36 = getelementptr inbounds i32, i32* %b, i64 %indvars.iv16
  %i4 = load i32, i32* %arrayidx36, align 4
  %arrayidx38 = getelementptr inbounds i32, i32* %c, i64 %indvars.iv16
  %i5 = load i32, i32* %arrayidx38, align 4
  %add39 = add nsw i32 %i4, %i5
  %arrayidx41 = getelementptr inbounds i32, i32* %a, i64 %indvars.iv16
  store i32 %add39, i32* %arrayidx41, align 4
  br label %for.inc42

for.inc42:                                        ; preds = %for.body34
  %indvars.iv.next17 = add nuw nsw i64 %indvars.iv16, 1
  br label %for.cond32, !llvm.loop !8

for.end44:                                        ; preds = %for.cond32
  ret void
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare i32 @llvm.smax.i32(i32, i32) #1

attributes #0 = { noinline nounwind uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nofree nosync nounwind readnone speculatable willreturn }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"uwtable", i32 1}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{!"clang version 14.0.6 (https://github.com/shravankumar0811/llvm-project.git 47ee914ea16086c1958b93540ed2351bcdae7cdb)"}
!4 = distinct !{!4, !5}
!5 = !{!"llvm.loop.mustprogress"}
!6 = distinct !{!6, !5}
!7 = distinct !{!7, !5}
!8 = distinct !{!8, !5}
