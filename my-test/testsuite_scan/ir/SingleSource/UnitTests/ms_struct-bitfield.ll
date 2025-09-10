; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/ms_struct-bitfield.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/ms_struct-bitfield.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct._struct_0 = type { i8 }
%struct._struct_7 = type { double }

@test_struct_0 = dso_local local_unnamed_addr global %struct._struct_0 { i8 123 }, align 1
@test_struct_1 = dso_local local_unnamed_addr global { i8, i8, i8, i8 } { i8 82, i8 0, i8 57, i8 4 }, align 2
@test_struct_2 = dso_local local_unnamed_addr global { double, i8, [3 x i8], i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, [2 x i8] } { double 2.000000e+01, i8 31, [3 x i8] zeroinitializer, i8 -48, i8 56, i8 6, i8 0, i8 1, i8 0, i8 68, i8 58, i8 56, i8 26, [2 x i8] zeroinitializer }, align 8
@test_struct_3 = dso_local local_unnamed_addr global { i8, i8, i8, i8, i8, [3 x i8] } { i8 39, i8 -6, i8 -39, i8 3, i8 1, [3 x i8] zeroinitializer }, align 4
@test_struct_4 = dso_local local_unnamed_addr global { i8, [7 x i8], double, double, i8, i8, i8, [5 x i8] } { i8 61, [7 x i8] zeroinitializer, double 2.000000e+01, double 2.000000e+01, i8 12, i8 0, i8 0, [5 x i8] zeroinitializer }, align 8
@test_struct_5 = dso_local local_unnamed_addr global { i8, i8, [2 x i8], i8, [3 x i8], i8, [3 x i8] } { i8 -115, i8 3, [2 x i8] zeroinitializer, i8 1, [3 x i8] zeroinitializer, i8 57, [3 x i8] zeroinitializer }, align 4
@test_struct_6 = dso_local local_unnamed_addr global { i8, [3 x i8], i8, i8, i8, i8, i8, [7 x i8], double, i8, i8, [6 x i8], double } { i8 12, [3 x i8] zeroinitializer, i8 20, i8 -35, i8 69, i8 1, i8 0, [7 x i8] zeroinitializer, double 2.000000e+01, i8 -45, i8 1, [6 x i8] zeroinitializer, double 2.000000e+01 }, align 8
@test_struct_7 = dso_local local_unnamed_addr global %struct._struct_7 { double 2.000000e+01 }, align 8
@test_struct_8 = dso_local local_unnamed_addr global { i8, [3 x i8], i8, i8, [2 x i8], i8, i8, i8, i8, i8, [3 x i8], i8, i8, i8, i8 } { i8 126, [3 x i8] zeroinitializer, i8 29, i8 -73, [2 x i8] zeroinitializer, i8 125, i8 0, i8 6, i8 0, i8 0, [3 x i8] zeroinitializer, i8 126, i8 30, i8 37, i8 0 }, align 4
@test_struct_9 = dso_local local_unnamed_addr global { double, i8, i8, i8, [5 x i8], double, i8, i8, i8, [5 x i8] } { double 2.000000e+01, i8 67, i8 101, i8 23, [5 x i8] zeroinitializer, double 2.000000e+01, i8 -97, i8 72, i8 15, [5 x i8] zeroinitializer }, align 8

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
