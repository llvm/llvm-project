; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr68911.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr68911.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@c = dso_local local_unnamed_addr global i32 0, align 4
@b = dso_local local_unnamed_addr global i32 0, align 4
@d = dso_local local_unnamed_addr global i16 0, align 4
@a = dso_local local_unnamed_addr global i8 0, align 4

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = load i32, ptr @c, align 4, !tbaa !6
  %2 = icmp slt i32 %1, 2
  br i1 %2, label %3, label %51

3:                                                ; preds = %0
  %4 = load i32, ptr @b, align 4
  %5 = icmp ne i32 %4, 0
  %6 = load i16, ptr @d, align 4
  %7 = icmp ne i16 %6, 0
  %8 = select i1 %5, i1 %7, i1 false
  %9 = zext i1 %8 to i32
  %10 = xor i32 %9, -1
  %11 = trunc nsw i32 %10 to i8
  br label %12

12:                                               ; preds = %3, %47
  %13 = phi i32 [ 0, %3 ], [ %41, %47 ]
  %14 = phi i32 [ 2, %3 ], [ %21, %47 ]
  %15 = phi i32 [ %1, %3 ], [ %48, %47 ]
  %16 = freeze i32 %13
  %17 = icmp ugt i32 %14, -8
  br i1 %17, label %19, label %18

18:                                               ; preds = %12
  store i8 %11, ptr @a, align 4, !tbaa !10
  br label %19

19:                                               ; preds = %18, %12
  %20 = phi i32 [ %10, %18 ], [ %14, %12 ]
  %21 = tail call i32 @llvm.umax.i32(i32 %20, i32 94)
  %22 = tail call i32 @llvm.umax.i32(i32 %16, i32 100)
  %23 = sub i32 %22, %16
  %24 = sub i32 %21, %20
  %25 = tail call i32 @llvm.umin.i32(i32 %23, i32 %24)
  %26 = add i32 %25, 1
  %27 = icmp ult i32 %26, 3
  br i1 %27, label %36, label %28

28:                                               ; preds = %19
  %29 = and i32 %25, -2
  %30 = add i32 %20, %29
  %31 = add i32 %16, %29
  br label %32

32:                                               ; preds = %32, %28
  %33 = phi i32 [ 0, %28 ], [ %34, %32 ]
  %34 = add nuw i32 %33, 2
  %35 = icmp eq i32 %34, %29
  br i1 %35, label %36, label %32, !llvm.loop !11

36:                                               ; preds = %32, %19
  %37 = phi i32 [ %20, %19 ], [ %30, %32 ]
  %38 = phi i32 [ %16, %19 ], [ %31, %32 ]
  br label %39

39:                                               ; preds = %36, %43
  %40 = phi i32 [ %44, %43 ], [ %37, %36 ]
  %41 = phi i32 [ %45, %43 ], [ %38, %36 ]
  %42 = icmp eq i32 %40, %21
  br i1 %42, label %47, label %43

43:                                               ; preds = %39
  %44 = add i32 %40, 1
  %45 = add i32 %41, 1
  %46 = icmp eq i32 %41, %22
  br i1 %46, label %50, label %39, !llvm.loop !15

47:                                               ; preds = %39
  %48 = add nsw i32 %15, 1
  store i32 %48, ptr @c, align 4, !tbaa !6
  %49 = icmp eq i32 %48, 2
  br i1 %49, label %51, label %12, !llvm.loop !16

50:                                               ; preds = %43
  tail call void @abort() #3
  unreachable

51:                                               ; preds = %47, %0
  ret i32 0
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.umax.i32(i32, i32) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.umin.i32(i32, i32) #2

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"int", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!8, !8, i64 0}
!11 = distinct !{!11, !12, !13, !14}
!12 = !{!"llvm.loop.mustprogress"}
!13 = !{!"llvm.loop.isvectorized", i32 1}
!14 = !{!"llvm.loop.unroll.runtime.disable"}
!15 = distinct !{!15, !12, !13}
!16 = distinct !{!16, !12}
