; ModuleID = 'out_cfcss.ll'
source_filename = "cfcss.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.str = private unnamed_addr constant [25 x i8] c" Signatures do not match\00", align 1
@.str.1 = private unnamed_addr constant [13 x i8] c" Value is %d\00", align 1
@G = internal global i32 0
@D = internal global i32 0
@G.1 = internal global i32 0
@D.2 = internal global i32 0

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @__cfcss_error() #0 {
entry:
  %call = call i32 (ptr, ...) @printf(ptr noundef @.str)
  call void @exit(i32 noundef 0) #3
  unreachable
}

declare i32 @printf(ptr noundef, ...) #1

; Function Attrs: noreturn nounwind
declare void @exit(i32 noundef) #2

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 {
entry:
  store i32 6, ptr @D.2, align 4
  store i32 1, ptr @G.1, align 4
  store i32 5, ptr @D, align 4
  store i32 1, ptr @G, align 4
  %retval = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 0, ptr %retval, align 4
  store i32 0, ptr %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %split13, %entry
  %0 = load i32, ptr @G.1, align 4
  %1 = load i32, ptr @D.2, align 4
  %2 = xor i32 %0, 5
  %3 = xor i32 %2, %1
  store i32 %3, ptr @G.1, align 4
  %failure2 = icmp ne i32 %3, 2
  br i1 %failure2, label %ErrorBlock1, label %split3

split3:                                           ; preds = %for.cond
  store i32 10, ptr @D.2, align 4
  %4 = load i32, ptr @G, align 4
  %5 = load i32, ptr @D, align 4
  %6 = xor i32 %4, 6
  %7 = xor i32 %6, %5
  store i32 %7, ptr @G, align 4
  %failure = icmp ne i32 %7, 2
  br i1 %failure, label %ErrorBlock, label %split

split:                                            ; preds = %split3
  %8 = load i32, ptr @G.1, align 4
  %9 = xor i32 %8, 1
  store i32 %9, ptr @G.1, align 4
  %failure4 = icmp ne i32 %9, 3
  br i1 %failure4, label %ErrorBlock1, label %split5

split5:                                           ; preds = %split
  %10 = load i32, ptr %i, align 4
  %cmp = icmp slt i32 %10, 10
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %split5
  %11 = load i32, ptr @G.1, align 4
  %12 = xor i32 %11, 7
  store i32 %12, ptr @G.1, align 4
  %failure6 = icmp ne i32 %12, 4
  br i1 %failure6, label %ErrorBlock1, label %split7

split7:                                           ; preds = %for.body
  store i32 12, ptr @D.2, align 4
  %13 = load i32, ptr @G, align 4
  %14 = xor i32 %13, 1
  store i32 %14, ptr @G, align 4
  %failure1 = icmp ne i32 %14, 3
  br i1 %failure1, label %ErrorBlock, label %split2

split2:                                           ; preds = %split7
  %15 = load i32, ptr @G.1, align 4
  %16 = xor i32 %15, 1
  store i32 %16, ptr @G.1, align 4
  %failure8 = icmp ne i32 %16, 5
  br i1 %failure8, label %ErrorBlock1, label %split9

split9:                                           ; preds = %split2
  %17 = load i32, ptr %i, align 4
  %call = call i32 (ptr, ...) @printf(ptr noundef @.str.1, i32 noundef %17)
  br label %for.inc

for.inc:                                          ; preds = %split9
  %18 = load i32, ptr @G.1, align 4
  %19 = xor i32 %18, 3
  store i32 %19, ptr @G.1, align 4
  %failure10 = icmp ne i32 %19, 6
  br i1 %failure10, label %ErrorBlock1, label %split11

split11:                                          ; preds = %for.inc
  store i32 14, ptr @D.2, align 4
  %20 = load i32, ptr @G, align 4
  %21 = xor i32 %20, 10
  store i32 %21, ptr @G, align 4
  %failure3 = icmp ne i32 %21, 4
  br i1 %failure3, label %ErrorBlock, label %split4

split4:                                           ; preds = %split11
  %22 = load i32, ptr @G.1, align 4
  %23 = xor i32 %22, 1
  store i32 %23, ptr @G.1, align 4
  %failure12 = icmp ne i32 %23, 7
  br i1 %failure12, label %ErrorBlock1, label %split13

split13:                                          ; preds = %split4
  store i32 0, ptr @D.2, align 4
  store i32 0, ptr @D, align 4
  %24 = load i32, ptr %i, align 4
  %inc = add nsw i32 %24, 1
  store i32 %inc, ptr %i, align 4
  br label %for.cond, !llvm.loop !6

for.end:                                          ; preds = %split5
  %25 = load i32, ptr @G.1, align 4
  %26 = xor i32 %25, 11
  store i32 %26, ptr @G.1, align 4
  %failure14 = icmp ne i32 %26, 8
  br i1 %failure14, label %ErrorBlock1, label %split15

split15:                                          ; preds = %for.end
  store i32 0, ptr @D.2, align 4
  %27 = load i32, ptr @G, align 4
  %28 = xor i32 %27, 7
  store i32 %28, ptr @G, align 4
  %failure5 = icmp ne i32 %28, 5
  br i1 %failure5, label %ErrorBlock, label %split6

split6:                                           ; preds = %split15
  %29 = load i32, ptr @G.1, align 4
  %30 = xor i32 %29, 1
  store i32 %30, ptr @G.1, align 4
  %failure16 = icmp ne i32 %30, 9
  br i1 %failure16, label %ErrorBlock1, label %split17

split17:                                          ; preds = %split6
  %31 = load i32, ptr %retval, align 4
  ret i32 %31

ErrorBlock:                                       ; preds = %split15, %split11, %split7, %split3
  %32 = load i32, ptr @G.1, align 4
  %33 = load i32, ptr @D.2, align 4
  %34 = xor i32 %32, 2
  %35 = xor i32 %34, %33
  store i32 %35, ptr @G.1, align 4
  %failure18 = icmp ne i32 %35, 10
  br i1 %failure18, label %ErrorBlock1, label %split19

split19:                                          ; preds = %ErrorBlock
  call void @__cfcss_error()
  ret i32 0

ErrorBlock1:                                      ; preds = %ErrorBlock, %split6, %for.end, %split4, %for.inc, %split2, %for.body, %split, %for.cond
  call void @__cfcss_error()
  ret i32 0
}

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { noreturn nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 2}
!5 = !{!"clang version 15.0.0 (https://shravan_kumar0826@bitbucket.org/shravan_kumar0826/llvm-project.git 00bb96a3bfe1901661abfdb27177c1ba6c6920c6)"}
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.mustprogress"}
