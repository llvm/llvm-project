; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr31169.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr31169.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read) uwtable
define dso_local range(i32 0, 2) i32 @sign_bit_p(ptr noundef readonly captures(none) %0, i64 noundef %1, i64 noundef %2) local_unnamed_addr #0 {
  %4 = load i16, ptr %0, align 4
  %5 = and i16 %4, 511
  %6 = zext nneg i16 %5 to i64
  %7 = icmp samesign ugt i16 %5, 64
  br i1 %7, label %8, label %13

8:                                                ; preds = %3
  %9 = add nsw i64 %6, -65
  %10 = shl nuw i64 1, %9
  %11 = sub nsw i64 128, %6
  %12 = lshr i64 -1, %11
  br label %19

13:                                               ; preds = %3
  %14 = add nuw nsw i64 %6, 4294967295
  %15 = and i64 %14, 4294967295
  %16 = shl nuw i64 1, %15
  %17 = sub nuw nsw i64 64, %6
  %18 = lshr i64 -1, %17
  br label %19

19:                                               ; preds = %13, %8
  %20 = phi i64 [ -1, %8 ], [ %18, %13 ]
  %21 = phi i64 [ 0, %8 ], [ %16, %13 ]
  %22 = phi i64 [ %12, %8 ], [ 0, %13 ]
  %23 = phi i64 [ %10, %8 ], [ 0, %13 ]
  %24 = and i64 %22, %1
  %25 = icmp eq i64 %24, %23
  %26 = and i64 %20, %2
  %27 = icmp eq i64 %26, %21
  %28 = select i1 %25, i1 %27, i1 false
  %29 = zext i1 %28 to i32
  ret i32 %29
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  ret i32 0
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
