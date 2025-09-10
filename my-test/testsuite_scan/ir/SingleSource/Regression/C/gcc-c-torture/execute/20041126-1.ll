; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20041126-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20041126-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

; Function Attrs: nofree nounwind uwtable
define dso_local void @check(ptr noundef readonly captures(none) %0) local_unnamed_addr #0 {
  %2 = load i32, ptr %0, align 4, !tbaa !6
  %3 = icmp eq i32 %2, 0
  br i1 %3, label %5, label %4

4:                                                ; preds = %17, %13, %9, %5, %1
  tail call void @abort() #3
  unreachable

5:                                                ; preds = %1
  %6 = getelementptr inbounds nuw i8, ptr %0, i64 4
  %7 = load i32, ptr %6, align 4, !tbaa !6
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %9, label %4

9:                                                ; preds = %5
  %10 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %11 = load i32, ptr %10, align 4, !tbaa !6
  %12 = icmp eq i32 %11, 0
  br i1 %12, label %13, label %4

13:                                               ; preds = %9
  %14 = getelementptr inbounds nuw i8, ptr %0, i64 12
  %15 = load i32, ptr %14, align 4, !tbaa !6
  %16 = icmp eq i32 %15, 0
  br i1 %16, label %17, label %4

17:                                               ; preds = %13
  %18 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %19 = load i32, ptr %18, align 4, !tbaa !6
  %20 = icmp eq i32 %19, 0
  br i1 %20, label %21, label %4

21:                                               ; preds = %17
  %22 = getelementptr inbounds nuw i8, ptr %0, i64 20
  %23 = load i32, ptr %22, align 4, !tbaa !6
  %24 = icmp eq i32 %23, 6
  br i1 %24, label %25, label %41, !llvm.loop !10

25:                                               ; preds = %21
  %26 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %27 = load i32, ptr %26, align 4, !tbaa !6
  %28 = icmp eq i32 %27, 7
  br i1 %28, label %29, label %41, !llvm.loop !10

29:                                               ; preds = %25
  %30 = getelementptr inbounds nuw i8, ptr %0, i64 28
  %31 = load i32, ptr %30, align 4, !tbaa !6
  %32 = icmp eq i32 %31, 8
  br i1 %32, label %33, label %41, !llvm.loop !10

33:                                               ; preds = %29
  %34 = getelementptr inbounds nuw i8, ptr %0, i64 32
  %35 = load i32, ptr %34, align 4, !tbaa !6
  %36 = icmp eq i32 %35, 9
  br i1 %36, label %37, label %41, !llvm.loop !10

37:                                               ; preds = %33
  %38 = getelementptr inbounds nuw i8, ptr %0, i64 36
  %39 = load i32, ptr %38, align 4, !tbaa !6
  %40 = icmp eq i32 %39, 10
  br i1 %40, label %42, label %41, !llvm.loop !10

41:                                               ; preds = %37, %33, %29, %25, %21
  tail call void @abort() #3
  unreachable

42:                                               ; preds = %37
  ret void
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 {
  ret i32 0
}

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
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
!10 = distinct !{!10, !11}
!11 = !{!"llvm.loop.mustprogress"}
