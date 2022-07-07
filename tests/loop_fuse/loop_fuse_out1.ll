; ModuleID = 'loop_fuse_out.ll'
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
  br i1 %exitcond, label %for.body, label %for.end16

for.body:                                         ; preds = %for.cond
  %0 = shl nuw nsw i64 %indvars.iv, 1
  %arrayidx = getelementptr inbounds i32, i32* %c, i64 %indvars.iv
  %1 = trunc i64 %0 to i32
  store i32 %1, i32* %arrayidx, align 4
  %2 = trunc i64 %indvars.iv to i32
  %mul = mul nsw i32 %2, %2
  %arrayidx2 = getelementptr inbounds i32, i32* %b, i64 %indvars.iv
  store i32 %mul, i32* %arrayidx2, align 4
  br label %for.body6

for.inc:                                          ; preds = %for.body6
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond, !llvm.loop !4

for.body6:                                        ; preds = %for.body
  %arrayidx8 = getelementptr inbounds i32, i32* %b, i64 %indvars.iv
  %i = load i32, i32* %arrayidx8, align 4
  %arrayidx10 = getelementptr inbounds i32, i32* %c, i64 %indvars.iv
  %i1 = load i32, i32* %arrayidx10, align 4
  %add11 = add nsw i32 %i, %i1
  %arrayidx13 = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  store i32 %add11, i32* %arrayidx13, align 4
  br label %for.inc

for.end16:                                        ; preds = %for.cond
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
!3 = !{!"clang version 14.0.6 (https://github.com/shravankumar0811/llvm-project.git 7f049514ee22563de5f8817412efd6d7d83109cf)"}
!4 = distinct !{!4, !5}
!5 = !{!"llvm.loop.mustprogress"}
