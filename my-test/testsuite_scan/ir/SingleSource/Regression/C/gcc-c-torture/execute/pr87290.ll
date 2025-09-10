; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr87290.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr87290.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@c = dso_local local_unnamed_addr global i32 0, align 4

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @f0() local_unnamed_addr #0 {
  %1 = load i32, ptr @c, align 4, !tbaa !6
  %2 = add nsw i32 %1, 1
  store i32 %2, ptr @c, align 4, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local range(i32 0, 2) i32 @f1(i32 noundef %0) local_unnamed_addr #1 {
  %2 = and i32 %0, -2147483633
  %3 = icmp eq i32 %2, 13
  %4 = zext i1 %3 to i32
  ret i32 %4
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local range(i32 0, 2) i32 @f2(i32 noundef %0) local_unnamed_addr #1 {
  %2 = srem i32 %0, 16
  %3 = icmp eq i32 %2, -13
  %4 = zext i1 %3 to i32
  ret i32 %4
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @f3(i32 noundef %0) local_unnamed_addr #0 {
  %2 = and i32 %0, -2147483633
  %3 = icmp eq i32 %2, 13
  br i1 %3, label %4, label %7

4:                                                ; preds = %1
  %5 = load i32, ptr @c, align 4, !tbaa !6
  %6 = add nsw i32 %5, 1
  store i32 %6, ptr @c, align 4, !tbaa !6
  br label %7

7:                                                ; preds = %4, %1
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @f4(i32 noundef %0) local_unnamed_addr #0 {
  %2 = srem i32 %0, 16
  %3 = icmp eq i32 %2, -13
  br i1 %3, label %4, label %7

4:                                                ; preds = %1
  %5 = load i32, ptr @c, align 4, !tbaa !6
  %6 = add nsw i32 %5, 1
  store i32 %6, ptr @c, align 4, !tbaa !6
  br label %7

7:                                                ; preds = %4, %1
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 {
  %1 = load i32, ptr @c, align 4
  %2 = icmp eq i32 %1, 0
  br i1 %2, label %3, label %20

3:                                                ; preds = %0, %30
  %4 = phi i32 [ %31, %30 ], [ -29, %0 ]
  %5 = icmp sgt i32 %4, -1
  %6 = zext i1 %5 to i32
  %7 = shl nsw i32 %4, 4
  %8 = add nsw i32 %7, -13
  %9 = srem i32 %8, 16
  %10 = icmp eq i32 %9, -13
  %11 = icmp sgt i32 %4, 0
  %12 = xor i1 %11, %10
  br i1 %12, label %14, label %13

13:                                               ; preds = %3
  tail call void @abort() #4
  unreachable

14:                                               ; preds = %3
  %15 = icmp sgt i32 %4, -1
  br i1 %15, label %16, label %17

16:                                               ; preds = %14
  store i32 1, ptr @c, align 4, !tbaa !6
  br label %17

17:                                               ; preds = %14, %16
  %18 = phi i32 [ 0, %14 ], [ 1, %16 ]
  %19 = icmp eq i32 %18, %6
  br i1 %19, label %21, label %20

20:                                               ; preds = %17, %0
  tail call void @abort() #4
  unreachable

21:                                               ; preds = %17
  br i1 %10, label %22, label %24

22:                                               ; preds = %21
  %23 = select i1 %5, i32 2, i32 1
  store i32 %23, ptr @c, align 4, !tbaa !6
  br label %24

24:                                               ; preds = %21, %22
  %25 = phi i32 [ %6, %21 ], [ %23, %22 ]
  %26 = icmp eq i32 %4, 0
  %27 = select i1 %26, i32 2, i32 1
  %28 = icmp eq i32 %25, %27
  br i1 %28, label %30, label %29

29:                                               ; preds = %24
  tail call void @abort() #4
  unreachable

30:                                               ; preds = %24
  store i32 0, ptr @c, align 4, !tbaa !6
  %31 = add nsw i32 %4, 1
  %32 = icmp eq i32 %31, 30
  br i1 %32, label %33, label %3, !llvm.loop !10

33:                                               ; preds = %30
  ret i32 0
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #3

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
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
!7 = !{!"int", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = distinct !{!10, !11, !12}
!11 = !{!"llvm.loop.mustprogress"}
!12 = !{!"llvm.loop.peeled.count", i32 1}
