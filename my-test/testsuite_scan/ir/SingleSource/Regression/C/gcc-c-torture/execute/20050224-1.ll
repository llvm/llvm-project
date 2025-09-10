; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20050224-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20050224-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@a = dso_local local_unnamed_addr global i64 3221225472, align 8
@b = dso_local local_unnamed_addr global i64 3489660928, align 8
@c = dso_local local_unnamed_addr global i64 3223042392, align 8
@d = dso_local local_unnamed_addr global i64 3223732224, align 8
@e = dso_local local_unnamed_addr global i64 3223879680, align 8
@f = dso_local local_unnamed_addr global i64 3224191864, align 8

; Function Attrs: nofree nounwind uwtable
define dso_local void @foo(i32 noundef %0, i32 noundef %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = icmp ne i32 %0, 245
  %5 = icmp ne i32 %1, 36
  %6 = or i1 %4, %5
  %7 = icmp ne i32 %2, 444
  %8 = or i1 %6, %7
  br i1 %8, label %9, label %10

9:                                                ; preds = %3
  tail call void @abort() #2
  unreachable

10:                                               ; preds = %3
  ret void
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = load i64, ptr @a, align 8, !tbaa !6
  %2 = load i64, ptr @b, align 8, !tbaa !6
  %3 = icmp ult i64 %1, %2
  br i1 %3, label %4, label %39

4:                                                ; preds = %0
  %5 = load i64, ptr @c, align 8, !tbaa !6
  %6 = load i64, ptr @d, align 8
  %7 = load i64, ptr @e, align 8
  %8 = load i64, ptr @f, align 8
  br label %9

9:                                                ; preds = %4, %27
  %10 = phi i32 [ 0, %4 ], [ %30, %27 ]
  %11 = phi i32 [ 0, %4 ], [ %29, %27 ]
  %12 = phi i32 [ 0, %4 ], [ %28, %27 ]
  %13 = phi i64 [ %1, %4 ], [ %31, %27 ]
  %14 = icmp ult i64 %13, %5
  br i1 %14, label %15, label %17

15:                                               ; preds = %9
  %16 = add nsw i32 %12, 1
  br label %27

17:                                               ; preds = %9
  %18 = icmp uge i64 %13, %6
  %19 = icmp ult i64 %13, %7
  %20 = select i1 %18, i1 %19, i1 false
  br i1 %20, label %21, label %23

21:                                               ; preds = %17
  %22 = add nsw i32 %10, 1
  br label %27

23:                                               ; preds = %17
  %24 = icmp ult i64 %13, %8
  %25 = zext i1 %24 to i32
  %26 = add nsw i32 %11, %25
  br label %27

27:                                               ; preds = %23, %15, %21
  %28 = phi i32 [ %16, %15 ], [ %12, %21 ], [ %12, %23 ]
  %29 = phi i32 [ %11, %15 ], [ %11, %21 ], [ %26, %23 ]
  %30 = phi i32 [ %10, %15 ], [ %22, %21 ], [ %10, %23 ]
  %31 = add i64 %13, 4096
  %32 = icmp ult i64 %31, %2
  br i1 %32, label %9, label %33, !llvm.loop !10

33:                                               ; preds = %27
  %34 = icmp ne i32 %29, 245
  %35 = icmp ne i32 %30, 36
  %36 = or i1 %34, %35
  %37 = icmp ne i32 %28, 444
  %38 = or i1 %37, %36
  br i1 %38, label %39, label %40

39:                                               ; preds = %0, %33
  tail call void @abort() #2
  unreachable

40:                                               ; preds = %33
  ret i32 0
}

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"long", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = distinct !{!10, !11}
!11 = !{!"llvm.loop.mustprogress"}
