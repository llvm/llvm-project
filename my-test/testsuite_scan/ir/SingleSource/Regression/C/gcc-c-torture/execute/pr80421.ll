; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr80421.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr80421.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@.str = private unnamed_addr constant [6 x i8] c"x %c\0A\00", align 1
@.str.1 = private unnamed_addr constant [14 x i8] c"case default\0A\00", align 1
@.str.3 = private unnamed_addr constant [10 x i8] c"case 'D'\0A\00", align 1
@.str.4 = private unnamed_addr constant [10 x i8] c"case 'I'\0A\00", align 1
@__const.bar.c = private unnamed_addr constant <{ [402 x i8], [18 x i8] }> <{ [402 x i8] c"\02\04\01\02\05\05\02\04\04\00\00\00\00\00\00\03\04\04\02\04\01\02\05\05\02\04\01\00\00\00\02\04\04\03\04\03\03\05\01\03\05\05\02\04\04\02\04\01\03\05\03\03\05\01\03\05\01\02\04\04\02\04\02\03\05\01\03\05\01\03\05\05\02\04\01\02\04\02\03\05\03\03\05\01\03\05\05\02\04\01\02\04\01\03\05\03\03\05\01\03\05\05\02\04\04\02\04\01\03\05\03\03\05\01\03\05\01\02\04\01\02\04\02\03\05\01\03\05\01\03\05\01\02\04\01\02\04\01\03\05\01\03\05\01\03\05\01\02\04\04\02\04\01\03\05\01\03\05\01\03\05\05\02\04\04\02\04\02\03\05\03\03\05\01\03\05\05\02\04\04\02\04\01\03\05\03\03\05\01\03\05\01\02\05\05\02\04\02\03\05\01\03\04\01\03\05\01\02\05\05\02\04\01\02\05\01\03\05\03\03\05\01\02\05\05\02\04\02\02\05\01\03\05\03\03\05\01\02\05\01\02\04\01\02\05\02\03\05\01\03\05\01\02\05\01\02\04\02\02\05\01\03\05\01\03\05\01\02\05\05\02\04\02\02\05\02\03\05\03\03\05\01\02\05\05\02\04\02\02\05\02\03\05\03\03\05\01\02\05\05\02\04\02\02\05\01\03\05\03\03\05\01\02\05\05\02\04\02\02\05\01\03\05\03\03\05\01\02\05\01\02\04\01\02\05\02\03\05\01\03\05\01\02\05\05\02\04\02\02\05\02\03\05\03\03\05\01\02\05\05\02\04\01\02\05\01\03\05\03\03\05\01\02\05\05\02\04\02\02\05\01\03\05\03\03\05\01\02\05\05\02\04\02\02\05\01\03\05\03\03\05\01", [18 x i8] zeroinitializer }>, align 1

; Function Attrs: noinline nounwind uwtable
define dso_local void @baz(ptr noundef %0, ...) local_unnamed_addr #0 {
  tail call void asm sideeffect "", "r,~{memory}"(ptr %0) #4, !srcloc !6
  %2 = load i8, ptr %0, align 1, !tbaa !7
  %3 = icmp eq i8 %2, 84
  br i1 %3, label %4, label %5

4:                                                ; preds = %1
  tail call void @abort() #5
  unreachable

5:                                                ; preds = %1
  ret void
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: nounwind uwtable
define dso_local noundef i32 @foo(i8 noundef %0) local_unnamed_addr #2 {
  %2 = zext i8 %0 to i32
  tail call void (ptr, ...) @baz(ptr noundef nonnull @.str, i32 noundef %2)
  %3 = icmp eq i8 %0, 73
  %4 = select i1 %3, ptr @.str.4, ptr @.str.1
  %5 = icmp eq i8 %0, 68
  %6 = select i1 %5, ptr @.str.3, ptr %4
  tail call void (ptr, ...) @baz(ptr noundef nonnull %6)
  ret i32 0
}

; Function Attrs: nounwind uwtable
define dso_local void @bar() local_unnamed_addr #2 {
  br label %1

1:                                                ; preds = %0, %42
  %2 = phi ptr [ getelementptr inbounds nuw (i8, ptr @__const.bar.c, i64 390), %0 ], [ %34, %42 ]
  %3 = phi i8 [ 77, %0 ], [ %43, %42 ]
  %4 = phi i8 [ 77, %0 ], [ %33, %42 ]
  %5 = phi i32 [ 2, %0 ], [ %32, %42 ]
  %6 = phi i32 [ 26, %0 ], [ %31, %42 ]
  %7 = phi i32 [ 25, %0 ], [ %30, %42 ]
  %8 = tail call i32 @llvm.usub.sat.i32(i32 %6, i32 2)
  %9 = sub nsw i32 %7, %8
  %10 = mul nsw i32 %9, 3
  %11 = add nsw i32 %10, %5
  %12 = sext i32 %11 to i64
  %13 = getelementptr inbounds i8, ptr %2, i64 %12
  %14 = load i8, ptr %13, align 1, !tbaa !7
  switch i8 %14, label %29 [
    i8 1, label %15
    i8 2, label %19
    i8 3, label %22
    i8 4, label %25
    i8 5, label %27
  ]

15:                                               ; preds = %1
  %16 = add nsw i32 %6, -1
  %17 = add nsw i32 %7, -1
  %18 = getelementptr inbounds i8, ptr %2, i64 -15
  br label %29

19:                                               ; preds = %1
  %20 = add nsw i32 %6, -1
  %21 = getelementptr inbounds i8, ptr %2, i64 -15
  br label %29

22:                                               ; preds = %1
  %23 = add nsw i32 %6, -1
  %24 = getelementptr inbounds i8, ptr %2, i64 -15
  br label %29

25:                                               ; preds = %1
  %26 = add nsw i32 %7, -1
  br label %29

27:                                               ; preds = %1
  %28 = add nsw i32 %7, -1
  br label %29

29:                                               ; preds = %1, %27, %25, %22, %19, %15
  %30 = phi i32 [ %7, %1 ], [ %17, %15 ], [ %7, %19 ], [ %7, %22 ], [ %26, %25 ], [ %28, %27 ]
  %31 = phi i32 [ %6, %1 ], [ %16, %15 ], [ %20, %19 ], [ %23, %22 ], [ %6, %25 ], [ %6, %27 ]
  %32 = phi i32 [ %5, %1 ], [ 2, %15 ], [ 0, %19 ], [ 2, %22 ], [ 1, %25 ], [ 2, %27 ]
  %33 = phi i8 [ %4, %1 ], [ 77, %15 ], [ 73, %19 ], [ 73, %22 ], [ 68, %25 ], [ 68, %27 ]
  %34 = phi ptr [ %2, %1 ], [ %18, %15 ], [ %21, %19 ], [ %24, %22 ], [ %2, %25 ], [ %2, %27 ]
  %35 = icmp eq i8 %33, %3
  br i1 %35, label %42, label %36

36:                                               ; preds = %29
  %37 = zext nneg i8 %3 to i32
  tail call void (ptr, ...) @baz(ptr noundef nonnull @.str, i32 noundef %37)
  %38 = icmp eq i8 %3, 73
  %39 = select i1 %38, ptr @.str.4, ptr @.str.1
  %40 = icmp eq i8 %3, 68
  %41 = select i1 %40, ptr @.str.3, ptr %39
  tail call void (ptr, ...) @baz(ptr noundef nonnull %41)
  br label %42

42:                                               ; preds = %29, %36
  %43 = phi i8 [ %33, %36 ], [ %3, %29 ]
  %44 = icmp sgt i32 %31, 0
  br i1 %44, label %1, label %45, !llvm.loop !10

45:                                               ; preds = %42
  ret void
}

; Function Attrs: nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 {
  tail call void (ptr, ...) @baz(ptr noundef nonnull @.str, i32 noundef 68)
  tail call void (ptr, ...) @baz(ptr noundef nonnull @.str.3)
  br label %1

1:                                                ; preds = %42, %0
  %2 = phi ptr [ getelementptr inbounds nuw (i8, ptr @__const.bar.c, i64 390), %0 ], [ %34, %42 ]
  %3 = phi i8 [ 77, %0 ], [ %43, %42 ]
  %4 = phi i8 [ 77, %0 ], [ %33, %42 ]
  %5 = phi i32 [ 2, %0 ], [ %32, %42 ]
  %6 = phi i32 [ 26, %0 ], [ %31, %42 ]
  %7 = phi i32 [ 25, %0 ], [ %30, %42 ]
  %8 = tail call i32 @llvm.usub.sat.i32(i32 %6, i32 2)
  %9 = sub nsw i32 %7, %8
  %10 = mul nsw i32 %9, 3
  %11 = add nsw i32 %10, %5
  %12 = sext i32 %11 to i64
  %13 = getelementptr inbounds i8, ptr %2, i64 %12
  %14 = load i8, ptr %13, align 1, !tbaa !7
  switch i8 %14, label %29 [
    i8 1, label %15
    i8 2, label %19
    i8 3, label %22
    i8 4, label %25
    i8 5, label %27
  ]

15:                                               ; preds = %1
  %16 = add nsw i32 %6, -1
  %17 = add nsw i32 %7, -1
  %18 = getelementptr inbounds i8, ptr %2, i64 -15
  br label %29

19:                                               ; preds = %1
  %20 = add nsw i32 %6, -1
  %21 = getelementptr inbounds i8, ptr %2, i64 -15
  br label %29

22:                                               ; preds = %1
  %23 = add nsw i32 %6, -1
  %24 = getelementptr inbounds i8, ptr %2, i64 -15
  br label %29

25:                                               ; preds = %1
  %26 = add nsw i32 %7, -1
  br label %29

27:                                               ; preds = %1
  %28 = add nsw i32 %7, -1
  br label %29

29:                                               ; preds = %27, %25, %22, %19, %15, %1
  %30 = phi i32 [ %7, %1 ], [ %17, %15 ], [ %7, %19 ], [ %7, %22 ], [ %26, %25 ], [ %28, %27 ]
  %31 = phi i32 [ %6, %1 ], [ %16, %15 ], [ %20, %19 ], [ %23, %22 ], [ %6, %25 ], [ %6, %27 ]
  %32 = phi i32 [ %5, %1 ], [ 2, %15 ], [ 0, %19 ], [ 2, %22 ], [ 1, %25 ], [ 2, %27 ]
  %33 = phi i8 [ %4, %1 ], [ 77, %15 ], [ 73, %19 ], [ 73, %22 ], [ 68, %25 ], [ 68, %27 ]
  %34 = phi ptr [ %2, %1 ], [ %18, %15 ], [ %21, %19 ], [ %24, %22 ], [ %2, %25 ], [ %2, %27 ]
  %35 = icmp eq i8 %33, %3
  br i1 %35, label %42, label %36

36:                                               ; preds = %29
  %37 = zext nneg i8 %3 to i32
  tail call void (ptr, ...) @baz(ptr noundef nonnull @.str, i32 noundef %37)
  %38 = icmp eq i8 %3, 73
  %39 = select i1 %38, ptr @.str.4, ptr @.str.1
  %40 = icmp eq i8 %3, 68
  %41 = select i1 %40, ptr @.str.3, ptr %39
  tail call void (ptr, ...) @baz(ptr noundef nonnull %41)
  br label %42

42:                                               ; preds = %36, %29
  %43 = phi i8 [ %33, %36 ], [ %3, %29 ]
  %44 = icmp sgt i32 %31, 0
  br i1 %44, label %1, label %45, !llvm.loop !10

45:                                               ; preds = %42
  ret i32 0
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.usub.sat.i32(i32, i32) #3

attributes #0 = { noinline nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { nounwind }
attributes #5 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{i64 113}
!7 = !{!8, !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = distinct !{!10, !11}
!11 = !{!"llvm.loop.mustprogress"}
