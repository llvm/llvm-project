; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr85582-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr85582-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@d = dso_local local_unnamed_addr global i32 2, align 4
@c = dso_local local_unnamed_addr global i64 1, align 8
@e = dso_local local_unnamed_addr global i32 0, align 4
@a = dso_local local_unnamed_addr global i32 0, align 4
@b = dso_local local_unnamed_addr global i32 0, align 4

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = load i32, ptr @a, align 4, !tbaa !6
  %2 = load i32, ptr @d, align 4, !tbaa !6
  store i32 %2, ptr @e, align 4, !tbaa !6
  %3 = icmp eq i32 %1, 0
  br i1 %3, label %5, label %4

4:                                                ; preds = %0, %4
  br label %4

5:                                                ; preds = %0
  %6 = icmp eq i32 %2, 0
  br i1 %6, label %7, label %8

7:                                                ; preds = %5, %7
  br label %7

8:                                                ; preds = %5
  %9 = load i64, ptr @c, align 8, !tbaa !10
  %10 = and i32 %2, 5
  %11 = zext nneg i32 %10 to i64
  %12 = ashr i64 %9, %11
  %13 = shl i64 %12, 32
  %14 = ashr exact i64 %13, 32
  %15 = load i32, ptr @b, align 4, !tbaa !6
  %16 = sext i32 %15 to i64
  %17 = srem i64 %16, %9
  %18 = mul nsw i64 %17, %14
  store i64 %18, ptr @c, align 8, !tbaa !10
  ret i32 0
}

attributes #0 = { nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }

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
!11 = !{!"long long", !8, i64 0}
