; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr20100-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr20100-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@g = internal unnamed_addr global i16 0, align 4
@p = internal unnamed_addr global i16 0, align 4
@e = dso_local local_unnamed_addr global i8 0, align 4

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i8 0, 2) i8 @frob(i16 noundef %0, i16 noundef %1) local_unnamed_addr #0 {
  store i16 %1, ptr @p, align 4, !tbaa !6
  %3 = zext i16 %0 to i32
  %4 = load i8, ptr @e, align 4, !tbaa !10
  %5 = zext i8 %4 to i32
  %6 = add nsw i32 %5, -1
  %7 = icmp eq i32 %6, %3
  %8 = add i16 %0, 1
  %9 = select i1 %7, i16 0, i16 %8
  store i16 %9, ptr @g, align 4, !tbaa !6
  %10 = icmp eq i16 %1, %9
  %11 = zext i1 %10 to i8
  ret i8 %11
}

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i16 0, 6) i16 @get_n() local_unnamed_addr #1 {
  %1 = load i16, ptr @p, align 4, !tbaa !6
  %2 = load i16, ptr @g, align 4, !tbaa !6
  %3 = icmp eq i16 %1, %2
  br i1 %3, label %39, label %4

4:                                                ; preds = %0
  %5 = load i8, ptr @e, align 4, !tbaa !10
  %6 = zext i8 %5 to i32
  %7 = add nsw i32 %6, -1
  %8 = zext i16 %2 to i32
  %9 = icmp eq i32 %7, %8
  %10 = add i16 %2, 1
  %11 = select i1 %9, i16 0, i16 %10
  %12 = icmp eq i16 %1, %11
  br i1 %12, label %36, label %13, !llvm.loop !11

13:                                               ; preds = %4
  %14 = zext i16 %11 to i32
  %15 = icmp eq i32 %7, %14
  %16 = add i16 %11, 1
  %17 = select i1 %15, i16 0, i16 %16
  %18 = icmp eq i16 %1, %17
  br i1 %18, label %36, label %19, !llvm.loop !11

19:                                               ; preds = %13
  %20 = zext i16 %17 to i32
  %21 = icmp eq i32 %7, %20
  %22 = add i16 %17, 1
  %23 = select i1 %21, i16 0, i16 %22
  %24 = icmp eq i16 %1, %23
  br i1 %24, label %36, label %25, !llvm.loop !11

25:                                               ; preds = %19
  %26 = zext i16 %23 to i32
  %27 = icmp eq i32 %7, %26
  %28 = add i16 %23, 1
  %29 = select i1 %27, i16 0, i16 %28
  %30 = icmp eq i16 %1, %29
  br i1 %30, label %36, label %31, !llvm.loop !11

31:                                               ; preds = %25
  %32 = zext i16 %29 to i32
  %33 = icmp eq i32 %7, %32
  %34 = add i16 %29, 1
  %35 = select i1 %33, i16 0, i16 %34
  br label %36

36:                                               ; preds = %31, %25, %19, %13, %4
  %37 = phi i16 [ %11, %4 ], [ %17, %13 ], [ %23, %19 ], [ %29, %25 ], [ %35, %31 ]
  %38 = phi i16 [ 1, %4 ], [ 2, %13 ], [ 3, %19 ], [ 4, %25 ], [ 5, %31 ]
  store i16 %37, ptr @g, align 4, !tbaa !6
  br label %39

39:                                               ; preds = %36, %0
  %40 = phi i16 [ %38, %36 ], [ 0, %0 ]
  ret i16 %40
}

; Function Attrs: nofree noreturn nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 {
  store i8 3, ptr @e, align 4, !tbaa !10
  store i16 2, ptr @p, align 4, !tbaa !6
  store i16 2, ptr @g, align 4, !tbaa !6
  tail call void @exit(i32 noundef 0) #4
  unreachable
}

; Function Attrs: nofree noreturn
declare void @exit(i32 noundef) local_unnamed_addr #3

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nofree noreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"short", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!8, !8, i64 0}
!11 = distinct !{!11, !12}
!12 = !{!"llvm.loop.mustprogress"}
