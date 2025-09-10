; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr57281.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr57281.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@a = dso_local local_unnamed_addr global i32 1, align 4
@d = dso_local global i32 0, align 4
@e = dso_local local_unnamed_addr global ptr @d, align 8
@c = dso_local global i64 0, align 8
@g = dso_local local_unnamed_addr global ptr @c, align 8
@b = dso_local local_unnamed_addr global i32 0, align 4
@f = dso_local global i64 0, align 8

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: write, inaccessiblemem: none) uwtable
define dso_local i32 @foo(i32 noundef %0) local_unnamed_addr #0 {
  %2 = load i32, ptr @b, align 4, !tbaa !6
  %3 = sext i32 %2 to i64
  %4 = load ptr, ptr @g, align 8, !tbaa !10
  store i64 %3, ptr %4, align 8, !tbaa !13
  %5 = icmp eq i32 %0, 0
  %6 = select i1 %5, i32 %2, i32 0
  ret i32 %6
}

; Function Attrs: nofree norecurse nounwind memory(readwrite, argmem: write) uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  %1 = load i32, ptr @b, align 4, !tbaa !6
  %2 = icmp eq i32 %1, -20
  br i1 %2, label %22, label %3

3:                                                ; preds = %0
  %4 = load i32, ptr @a, align 4, !tbaa !6
  %5 = load ptr, ptr @e, align 8, !tbaa !15
  %6 = load ptr, ptr @g, align 8, !tbaa !10
  %7 = freeze i32 %4
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %9, label %15

9:                                                ; preds = %3, %9
  %10 = load volatile i64, ptr @f, align 8, !tbaa !13
  store i32 0, ptr %5, align 4, !tbaa !6
  %11 = load i32, ptr @b, align 4, !tbaa !6
  %12 = sext i32 %11 to i64
  store i64 %12, ptr %6, align 8, !tbaa !13
  store i32 %11, ptr %5, align 4, !tbaa !6
  %13 = add nsw i32 %11, -1
  store i32 %13, ptr @b, align 4, !tbaa !6
  %14 = icmp eq i32 %13, -20
  br i1 %14, label %22, label %9, !llvm.loop !17

15:                                               ; preds = %3, %15
  %16 = load volatile i64, ptr @f, align 8, !tbaa !13
  store i32 0, ptr %5, align 4, !tbaa !6
  %17 = load i32, ptr @b, align 4, !tbaa !6
  %18 = sext i32 %17 to i64
  store i64 %18, ptr %6, align 8, !tbaa !13
  store i32 0, ptr %5, align 4, !tbaa !6
  %19 = load i32, ptr @b, align 4, !tbaa !6
  %20 = add nsw i32 %19, -1
  store i32 %20, ptr @b, align 4, !tbaa !6
  %21 = icmp eq i32 %20, -20
  br i1 %21, label %22, label %15, !llvm.loop !17

22:                                               ; preds = %15, %9, %0
  ret i32 0
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: write, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree norecurse nounwind memory(readwrite, argmem: write) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }

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
!10 = !{!11, !11, i64 0}
!11 = !{!"p1 long long", !12, i64 0}
!12 = !{!"any pointer", !8, i64 0}
!13 = !{!14, !14, i64 0}
!14 = !{!"long long", !8, i64 0}
!15 = !{!16, !16, i64 0}
!16 = !{!"p1 int", !12, i64 0}
!17 = distinct !{!17, !18}
!18 = !{!"llvm.loop.mustprogress"}
