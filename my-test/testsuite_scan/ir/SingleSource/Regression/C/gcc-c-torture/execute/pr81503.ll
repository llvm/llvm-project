; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr81503.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr81503.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@a = dso_local local_unnamed_addr global i16 -24075, align 4
@b = dso_local local_unnamed_addr global i16 3419, align 4
@c = dso_local local_unnamed_addr global i32 0, align 4

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @foo() local_unnamed_addr #0 {
  %1 = load i16, ptr @a, align 4, !tbaa !6
  %2 = zext i16 %1 to i32
  %3 = load i16, ptr @b, align 4, !tbaa !6
  %4 = zext i16 %3 to i32
  %5 = mul nsw i32 %4, -2
  %6 = sub nsw i32 0, %2
  %7 = icmp eq i32 %5, %6
  br i1 %7, label %10, label %8

8:                                                ; preds = %0
  %9 = xor i32 %5, -2147483648
  store i32 %9, ptr @c, align 4, !tbaa !10
  br label %10

10:                                               ; preds = %8, %0
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 -1, 1) i32 @main() local_unnamed_addr #0 {
  %1 = load i16, ptr @a, align 4, !tbaa !6
  %2 = zext i16 %1 to i32
  %3 = load i16, ptr @b, align 4, !tbaa !6
  %4 = zext i16 %3 to i32
  %5 = mul nsw i32 %4, -2
  %6 = sub nsw i32 0, %2
  %7 = icmp eq i32 %5, %6
  br i1 %7, label %8, label %10

8:                                                ; preds = %0
  %9 = load i32, ptr @c, align 4, !tbaa !10
  br label %12

10:                                               ; preds = %0
  %11 = xor i32 %5, -2147483648
  store i32 %11, ptr @c, align 4, !tbaa !10
  br label %12

12:                                               ; preds = %8, %10
  %13 = phi i32 [ %9, %8 ], [ %11, %10 ]
  %14 = icmp ne i32 %13, 2147476810
  %15 = sext i1 %14 to i32
  ret i32 %15
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }

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
!10 = !{!11, !11, i64 0}
!11 = !{!"int", !8, i64 0}
