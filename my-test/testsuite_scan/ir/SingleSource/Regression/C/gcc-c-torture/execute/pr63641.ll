; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr63641.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr63641.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@tab1 = dso_local local_unnamed_addr global [32 x i8] c"\01\01\01\01\01\01\01\01\01\00\00\01\00\00\01\01\01\01\01\01\01\01\01\01\01\01\01\00\01\01\01\01", align 1
@tab2 = dso_local local_unnamed_addr global [64 x i8] c"\01\01\01\01\01\01\01\01\01\00\00\01\00\00\01\01\01\01\01\01\01\01\01\01\01\01\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\01\01\01", align 1

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local range(i32 0, 2) i32 @foo(i8 noundef %0) local_unnamed_addr #0 {
  %2 = icmp ult i8 %0, 9
  %3 = icmp eq i8 %0, 11
  %4 = or i1 %2, %3
  %5 = add i8 %0, -14
  %6 = icmp ult i8 %5, 13
  %7 = or i1 %4, %6
  %8 = and i8 %0, -4
  %9 = icmp eq i8 %8, 28
  %10 = or i1 %9, %7
  %11 = zext i1 %10 to i32
  ret i32 %11
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local range(i32 0, 2) i32 @bar(i8 noundef %0) local_unnamed_addr #0 {
  %2 = icmp ult i8 %0, 9
  %3 = icmp eq i8 %0, 11
  %4 = or i1 %2, %3
  %5 = add i8 %0, -14
  %6 = icmp ult i8 %5, 13
  %7 = or i1 %4, %6
  %8 = and i8 %0, -4
  %9 = icmp eq i8 %8, 60
  %10 = or i1 %9, %7
  %11 = zext i1 %10 to i32
  ret i32 %11
}

; Function Attrs: nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  tail call void asm sideeffect "", "~{memory}"() #3, !srcloc !6
  br label %4

1:                                                ; preds = %13
  %2 = add nuw nsw i64 %5, 1
  %3 = icmp eq i64 %2, 256
  br i1 %3, label %20, label %4, !llvm.loop !7

4:                                                ; preds = %0, %1
  %5 = phi i64 [ 0, %0 ], [ %2, %1 ]
  %6 = trunc i64 %5 to i8
  %7 = tail call i32 @foo(i8 noundef %6)
  %8 = icmp samesign ult i64 %5, 32
  br i1 %8, label %9, label %13

9:                                                ; preds = %4
  %10 = getelementptr inbounds nuw i8, ptr @tab1, i64 %5
  %11 = load i8, ptr %10, align 1, !tbaa !9
  %12 = zext i8 %11 to i32
  br label %13

13:                                               ; preds = %4, %9
  %14 = phi i32 [ %12, %9 ], [ 0, %4 ]
  %15 = icmp eq i32 %7, %14
  br i1 %15, label %1, label %16

16:                                               ; preds = %13
  tail call void @abort() #4
  unreachable

17:                                               ; preds = %29
  %18 = add nuw nsw i64 %21, 1
  %19 = icmp eq i64 %18, 256
  br i1 %19, label %33, label %20, !llvm.loop !12

20:                                               ; preds = %1, %17
  %21 = phi i64 [ %18, %17 ], [ 0, %1 ]
  %22 = trunc i64 %21 to i8
  %23 = tail call i32 @bar(i8 noundef %22)
  %24 = icmp samesign ult i64 %21, 64
  br i1 %24, label %25, label %29

25:                                               ; preds = %20
  %26 = getelementptr inbounds nuw i8, ptr @tab2, i64 %21
  %27 = load i8, ptr %26, align 1, !tbaa !9
  %28 = zext i8 %27 to i32
  br label %29

29:                                               ; preds = %20, %25
  %30 = phi i32 [ %28, %25 ], [ 0, %20 ]
  %31 = icmp eq i32 %23, %30
  br i1 %31, label %17, label %32

32:                                               ; preds = %29
  tail call void @abort() #4
  unreachable

33:                                               ; preds = %17
  ret i32 0
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #2

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nounwind }
attributes #4 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{i64 922}
!7 = distinct !{!7, !8}
!8 = !{!"llvm.loop.mustprogress"}
!9 = !{!10, !10, i64 0}
!10 = !{!"omnipotent char", !11, i64 0}
!11 = !{!"Simple C/C++ TBAA"}
!12 = distinct !{!12, !8}
