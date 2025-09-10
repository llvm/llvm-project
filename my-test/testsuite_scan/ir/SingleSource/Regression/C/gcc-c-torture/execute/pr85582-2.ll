; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr85582-2.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr85582-2.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local i128 @f1(i128 noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = and i32 %1, 5
  %4 = zext nneg i32 %3 to i128
  %5 = shl i128 %0, %4
  %6 = sext i32 %1 to i128
  %7 = add nsw i128 %5, %6
  ret i128 %7
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local i128 @f2(i128 noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = and i32 %1, 5
  %4 = zext nneg i32 %3 to i128
  %5 = ashr i128 %0, %4
  %6 = sext i32 %1 to i128
  %7 = add nsw i128 %5, %6
  ret i128 %7
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local i128 @f3(i128 noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = and i32 %1, 5
  %4 = zext nneg i32 %3 to i128
  %5 = lshr i128 %0, %4
  %6 = sext i32 %1 to i128
  %7 = add i128 %5, %6
  ret i128 %7
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  ret i32 0
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
