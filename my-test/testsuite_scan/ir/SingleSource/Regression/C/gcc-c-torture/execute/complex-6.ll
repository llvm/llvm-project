; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/complex-6.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/complex-6.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@err = dso_local local_unnamed_addr global i32 0, align 4

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local { float, float } @ctest_float([2 x float] noundef alignstack(8) %0) local_unnamed_addr #0 {
  %2 = extractvalue [2 x float] %0, 0
  %3 = extractvalue [2 x float] %0, 1
  %4 = fneg float %3
  %5 = insertvalue { float, float } poison, float %2, 0
  %6 = insertvalue { float, float } %5, float %4, 1
  ret { float, float } %6
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local void @test_float() local_unnamed_addr #0 {
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local { double, double } @ctest_double([2 x double] noundef alignstack(8) %0) local_unnamed_addr #0 {
  %2 = extractvalue [2 x double] %0, 0
  %3 = extractvalue [2 x double] %0, 1
  %4 = fneg double %3
  %5 = insertvalue { double, double } poison, double %2, 0
  %6 = insertvalue { double, double } %5, double %4, 1
  ret { double, double } %6
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local void @test_double() local_unnamed_addr #0 {
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local { fp128, fp128 } @ctest_long_double([2 x fp128] noundef alignstack(16) %0) local_unnamed_addr #0 {
  %2 = extractvalue [2 x fp128] %0, 0
  %3 = extractvalue [2 x fp128] %0, 1
  %4 = fneg fp128 %3
  %5 = insertvalue { fp128, fp128 } poison, fp128 %2, 0
  %6 = insertvalue { fp128, fp128 } %5, fp128 %4, 1
  ret { fp128, fp128 } %6
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local void @test_long_double() local_unnamed_addr #0 {
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i64 @ctest_int(i64 noundef %0) local_unnamed_addr #0 {
  %2 = and i64 %0, -4294967296
  %3 = and i64 %0, 4294967295
  %4 = sub i64 %3, %2
  ret i64 %4
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local void @test_int() local_unnamed_addr #0 {
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef [2 x i64] @ctest_long_int([2 x i64] noundef %0) local_unnamed_addr #0 {
  %2 = extractvalue [2 x i64] %0, 1
  %3 = sub i64 0, %2
  %4 = insertvalue [2 x i64] %0, i64 %3, 1
  ret [2 x i64] %4
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local void @test_long_int() local_unnamed_addr #0 {
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  store i32 0, ptr @err, align 4, !tbaa !6
  ret i32 0
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nofree norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }

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
