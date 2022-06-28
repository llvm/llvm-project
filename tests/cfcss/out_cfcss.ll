; ModuleID = 'cfcss.ll'
source_filename = "cfcss.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.str = private unnamed_addr constant [25 x i8] c" Signatures do not match\00", align 1
@.str.1 = private unnamed_addr constant [13 x i8] c" Value is %d\00", align 1
@G = internal global i32 0
@D = internal global i32 0

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @__cfcss_error() #0 {
entry:
  %call = call i32 (i8*, ...) @printf(i8* noundef getelementptr inbounds ([25 x i8], [25 x i8]* @.str, i64 0, i64 0))
  call void @exit(i32 noundef 0) #3
  unreachable
}

declare dso_local i32 @printf(i8* noundef, ...) #1

; Function Attrs: noreturn nounwind
declare dso_local void @exit(i32 noundef) #2

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 {
entry:
  store i32 5, i32* @D, align 4
  store i32 1, i32* @G, align 4
  %retval = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 0, i32* %retval, align 4
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %split4, %entry
  %0 = load i32, i32* @G, align 4
  %1 = load i32, i32* @D, align 4
  %2 = xor i32 %0, 6
  %3 = xor i32 %2, %1
  store i32 %3, i32* @G, align 4
  %failure = icmp ne i32 %3, 2
  br i1 %failure, label %ErrorBlock, label %split

split:                                            ; preds = %for.cond
  %4 = load i32, i32* %i, align 4
  %cmp = icmp slt i32 %4, 10
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %split
  %5 = load i32, i32* @G, align 4
  %6 = xor i32 %5, 1
  store i32 %6, i32* @G, align 4
  %failure1 = icmp ne i32 %6, 3
  br i1 %failure1, label %ErrorBlock, label %split2

split2:                                           ; preds = %for.body
  %7 = load i32, i32* %i, align 4
  %call = call i32 (i8*, ...) @printf(i8* noundef getelementptr inbounds ([13 x i8], [13 x i8]* @.str.1, i64 0, i64 0), i32 noundef %7)
  br label %for.inc

for.inc:                                          ; preds = %split2
  %8 = load i32, i32* @G, align 4
  %9 = xor i32 %8, 7
  store i32 %9, i32* @G, align 4
  %failure3 = icmp ne i32 %9, 4
  br i1 %failure3, label %ErrorBlock, label %split4

split4:                                           ; preds = %for.inc
  store i32 0, i32* @D, align 4
  %10 = load i32, i32* %i, align 4
  %inc = add nsw i32 %10, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond, !llvm.loop !4

for.end:                                          ; preds = %split
  %11 = load i32, i32* @G, align 4
  %12 = xor i32 %11, 7
  store i32 %12, i32* @G, align 4
  %failure5 = icmp ne i32 %12, 5
  br i1 %failure5, label %ErrorBlock, label %split6

split6:                                           ; preds = %for.end
  %13 = load i32, i32* %retval, align 4
  ret i32 %13

ErrorBlock:                                       ; preds = %for.end, %for.inc, %for.body, %for.cond
  call void @__cfcss_error()
  ret i32 0
}

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { noreturn nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"uwtable", i32 1}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{!"clang version 14.0.6 (https://github.com/shravankumar0811/llvm-project.git f28c006a5895fc0e329fe15fead81e37457cb1d1)"}
!4 = distinct !{!4, !5}
!5 = !{!"llvm.loop.mustprogress"}
