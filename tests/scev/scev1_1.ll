; ModuleID = 'scev1.ll'
source_filename = "scev1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@Z = dso_local global i32 5, align 4
@.str = private unnamed_addr constant [5 x i8] c"%d%d\00", align 1

; Function Attrs: noinline nounwind uwtable
define dso_local i32 @main() #0 {
entry:
  %A = alloca [100 x i32], align 16
  %B = alloca [100 x i32], align 16
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %cmp = icmp ult i64 %indvars.iv, 20
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %0 = add nuw nsw i64 %indvars.iv, 12
  %1 = or i64 %indvars.iv, 2
  %arrayidx = getelementptr inbounds [100 x i32], [100 x i32]* %A, i64 0, i64 %1
  %2 = trunc i64 %0 to i32
  store i32 %2, i32* %arrayidx, align 8
  %3 = add nuw nsw i64 %indvars.iv, 11
  %4 = or i64 %indvars.iv, 1
  %arrayidx7 = getelementptr inbounds [100 x i32], [100 x i32]* %B, i64 0, i64 %4
  %5 = trunc i64 %3 to i32
  store i32 %5, i32* %arrayidx7, align 4
  %6 = add nuw nsw i64 %indvars.iv, 13
  %7 = or i64 %indvars.iv, 3
  %arrayidx12 = getelementptr inbounds [100 x i32], [100 x i32]* %A, i64 0, i64 %7
  %8 = trunc i64 %6 to i32
  store i32 %8, i32* %arrayidx12, align 4
  %9 = add nuw nsw i64 %indvars.iv, 10
  %arrayidx15 = getelementptr inbounds [100 x i32], [100 x i32]* %A, i64 0, i64 %indvars.iv
  %10 = trunc i64 %9 to i32
  store i32 %10, i32* %arrayidx15, align 16
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 4
  br label %for.cond, !llvm.loop !4

for.end:                                          ; preds = %for.cond
  br label %for.cond18

for.cond18:                                       ; preds = %for.inc25, %for.end
  %indvars.iv11 = phi i64 [ %indvars.iv.next12, %for.inc25 ], [ 0, %for.end ]
  %exitcond = icmp ne i64 %indvars.iv11, 20
  br i1 %exitcond, label %for.body20, label %for.end26

for.body20:                                       ; preds = %for.cond18
  %arrayidx22 = getelementptr inbounds [100 x i32], [100 x i32]* %A, i64 0, i64 %indvars.iv11
  %i = load i32, i32* %arrayidx22, align 4
  %arrayidx24 = getelementptr inbounds [100 x i32], [100 x i32]* %B, i64 0, i64 %indvars.iv11
  %i1 = load i32, i32* %arrayidx24, align 4
  %call = call i32 (i8*, ...) @printf(i8* noundef nonnull dereferenceable(1) getelementptr inbounds ([5 x i8], [5 x i8]* @.str, i64 0, i64 0), i32 noundef %i, i32 noundef %i1) #2
  br label %for.inc25

for.inc25:                                        ; preds = %for.body20
  %indvars.iv.next12 = add nuw nsw i64 %indvars.iv11, 1
  br label %for.cond18, !llvm.loop !6

for.end26:                                        ; preds = %for.cond18
  ret i32 0
}

declare dso_local i32 @printf(i8* noundef, ...) #1

attributes #0 = { noinline nounwind uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { nounwind }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"uwtable", i32 1}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{!"clang version 14.0.6 (https://github.com/shravankumar0811/llvm-project.git 973cbf86551cd15a9cb85bbb0ad01f2406ccb62d)"}
!4 = distinct !{!4, !5}
!5 = !{!"llvm.loop.mustprogress"}
!6 = distinct !{!6, !5}
