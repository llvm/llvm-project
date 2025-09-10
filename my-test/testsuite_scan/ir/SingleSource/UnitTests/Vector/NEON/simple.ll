; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/Vector/NEON/simple.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/Vector/NEON/simple.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@.str = private unnamed_addr constant [65 x i8] c"(%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d)\00", align 1
@.str.1 = private unnamed_addr constant [33 x i8] c"(%d, %d, %d, %d, %d, %d, %d, %d)\00", align 1
@.str.2 = private unnamed_addr constant [17 x i8] c"(%d, %d, %d, %d)\00", align 1
@__FUNCTION__.test_basic = private unnamed_addr constant [11 x i8] c"test_basic\00", align 1
@.str.4 = private unnamed_addr constant [7 x i8] c"a0_0: \00", align 1
@.str.6 = private unnamed_addr constant [7 x i8] c"a0_1: \00", align 1
@.str.7 = private unnamed_addr constant [7 x i8] c"a0_2: \00", align 1
@.str.8 = private unnamed_addr constant [7 x i8] c"a1_0: \00", align 1
@.str.9 = private unnamed_addr constant [7 x i8] c"a1_1: \00", align 1
@.str.10 = private unnamed_addr constant [7 x i8] c"a1_2: \00", align 1
@.str.11 = private unnamed_addr constant [7 x i8] c"a2_0: \00", align 1
@.str.12 = private unnamed_addr constant [7 x i8] c"a2_1: \00", align 1
@.str.13 = private unnamed_addr constant [7 x i8] c"a2_2: \00", align 1
@__FUNCTION__.test_zip = private unnamed_addr constant [9 x i8] c"test_zip\00", align 1
@.str.14 = private unnamed_addr constant [14 x i8] c"a0_2.val[0]: \00", align 1
@.str.15 = private unnamed_addr constant [14 x i8] c"a0_2.val[1]: \00", align 1
@.str.16 = private unnamed_addr constant [14 x i8] c"a0_3.val[0]: \00", align 1
@.str.17 = private unnamed_addr constant [14 x i8] c"a0_3.val[1]: \00", align 1
@.str.18 = private unnamed_addr constant [14 x i8] c"a1_2.val[0]: \00", align 1
@.str.19 = private unnamed_addr constant [14 x i8] c"a1_2.val[1]: \00", align 1
@.str.20 = private unnamed_addr constant [14 x i8] c"a1_3.val[0]: \00", align 1
@.str.21 = private unnamed_addr constant [14 x i8] c"a1_3.val[1]: \00", align 1
@.str.22 = private unnamed_addr constant [14 x i8] c"a2_2.val[0]: \00", align 1
@.str.23 = private unnamed_addr constant [14 x i8] c"a2_2.val[1]: \00", align 1
@.str.24 = private unnamed_addr constant [14 x i8] c"a2_3.val[0]: \00", align 1
@.str.25 = private unnamed_addr constant [14 x i8] c"a2_3.val[1]: \00", align 1

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local <16 x i8> @init_v16i8(i8 noundef %0, i8 noundef %1, i8 noundef %2, i8 noundef %3, i8 noundef %4, i8 noundef %5, i8 noundef %6, i8 noundef %7, i8 noundef %8, i8 noundef %9, i8 noundef %10, i8 noundef %11, i8 noundef %12, i8 noundef %13, i8 noundef %14, i8 noundef %15) local_unnamed_addr #0 {
  %17 = insertelement <16 x i8> <i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef>, i8 %0, i64 0
  %18 = insertelement <16 x i8> %17, i8 %1, i64 1
  %19 = insertelement <16 x i8> %18, i8 %2, i64 2
  %20 = insertelement <16 x i8> %19, i8 %3, i64 3
  %21 = insertelement <16 x i8> %20, i8 %4, i64 4
  %22 = insertelement <16 x i8> %21, i8 %5, i64 5
  %23 = insertelement <16 x i8> %22, i8 %6, i64 6
  %24 = insertelement <16 x i8> %23, i8 %7, i64 7
  %25 = insertelement <16 x i8> %24, i8 %8, i64 8
  %26 = insertelement <16 x i8> %25, i8 %9, i64 9
  %27 = insertelement <16 x i8> %26, i8 %10, i64 10
  %28 = insertelement <16 x i8> %27, i8 %11, i64 11
  %29 = insertelement <16 x i8> %28, i8 %12, i64 12
  %30 = insertelement <16 x i8> %29, i8 %13, i64 13
  %31 = insertelement <16 x i8> %30, i8 %14, i64 14
  %32 = insertelement <16 x i8> %31, i8 %15, i64 15
  ret <16 x i8> %32
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local <8 x i16> @init_v8i16(i16 noundef %0, i16 noundef %1, i16 noundef %2, i16 noundef %3, i16 noundef %4, i16 noundef %5, i16 noundef %6, i16 noundef %7) local_unnamed_addr #0 {
  %9 = insertelement <8 x i16> poison, i16 %0, i64 0
  %10 = insertelement <8 x i16> %9, i16 %1, i64 1
  %11 = insertelement <8 x i16> %10, i16 %2, i64 2
  %12 = insertelement <8 x i16> %11, i16 %3, i64 3
  %13 = insertelement <8 x i16> %12, i16 %4, i64 4
  %14 = insertelement <8 x i16> %13, i16 %5, i64 5
  %15 = insertelement <8 x i16> %14, i16 %6, i64 6
  %16 = insertelement <8 x i16> %15, i16 %7, i64 7
  ret <8 x i16> %16
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local <4 x i32> @init_v4i32(i32 noundef %0, i32 noundef %1, i32 noundef %2, i32 noundef %3) local_unnamed_addr #0 {
  %5 = insertelement <4 x i32> poison, i32 %0, i64 0
  %6 = insertelement <4 x i32> %5, i32 %1, i64 1
  %7 = insertelement <4 x i32> %6, i32 %2, i64 2
  %8 = insertelement <4 x i32> %7, i32 %3, i64 3
  ret <4 x i32> %8
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @print_v16i8(<16 x i8> noundef %0) local_unnamed_addr #1 {
  %2 = extractelement <16 x i8> %0, i64 0
  %3 = sext i8 %2 to i32
  %4 = extractelement <16 x i8> %0, i64 1
  %5 = sext i8 %4 to i32
  %6 = extractelement <16 x i8> %0, i64 2
  %7 = sext i8 %6 to i32
  %8 = extractelement <16 x i8> %0, i64 3
  %9 = sext i8 %8 to i32
  %10 = extractelement <16 x i8> %0, i64 4
  %11 = sext i8 %10 to i32
  %12 = extractelement <16 x i8> %0, i64 5
  %13 = sext i8 %12 to i32
  %14 = extractelement <16 x i8> %0, i64 6
  %15 = sext i8 %14 to i32
  %16 = extractelement <16 x i8> %0, i64 7
  %17 = sext i8 %16 to i32
  %18 = extractelement <16 x i8> %0, i64 8
  %19 = sext i8 %18 to i32
  %20 = extractelement <16 x i8> %0, i64 9
  %21 = sext i8 %20 to i32
  %22 = extractelement <16 x i8> %0, i64 10
  %23 = sext i8 %22 to i32
  %24 = extractelement <16 x i8> %0, i64 11
  %25 = sext i8 %24 to i32
  %26 = extractelement <16 x i8> %0, i64 12
  %27 = sext i8 %26 to i32
  %28 = extractelement <16 x i8> %0, i64 13
  %29 = sext i8 %28 to i32
  %30 = extractelement <16 x i8> %0, i64 14
  %31 = sext i8 %30 to i32
  %32 = extractelement <16 x i8> %0, i64 15
  %33 = sext i8 %32 to i32
  %34 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %3, i32 noundef %5, i32 noundef %7, i32 noundef %9, i32 noundef %11, i32 noundef %13, i32 noundef %15, i32 noundef %17, i32 noundef %19, i32 noundef %21, i32 noundef %23, i32 noundef %25, i32 noundef %27, i32 noundef %29, i32 noundef %31, i32 noundef %33)
  ret void
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #2

; Function Attrs: nofree nounwind uwtable
define dso_local void @print_v8i16(<8 x i16> noundef %0) local_unnamed_addr #1 {
  %2 = extractelement <8 x i16> %0, i64 0
  %3 = sext i16 %2 to i32
  %4 = extractelement <8 x i16> %0, i64 1
  %5 = sext i16 %4 to i32
  %6 = extractelement <8 x i16> %0, i64 2
  %7 = sext i16 %6 to i32
  %8 = extractelement <8 x i16> %0, i64 3
  %9 = sext i16 %8 to i32
  %10 = extractelement <8 x i16> %0, i64 4
  %11 = sext i16 %10 to i32
  %12 = extractelement <8 x i16> %0, i64 5
  %13 = sext i16 %12 to i32
  %14 = extractelement <8 x i16> %0, i64 6
  %15 = sext i16 %14 to i32
  %16 = extractelement <8 x i16> %0, i64 7
  %17 = sext i16 %16 to i32
  %18 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %3, i32 noundef %5, i32 noundef %7, i32 noundef %9, i32 noundef %11, i32 noundef %13, i32 noundef %15, i32 noundef %17)
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @print_v4i32(<4 x i32> noundef %0) local_unnamed_addr #1 {
  %2 = extractelement <4 x i32> %0, i64 0
  %3 = extractelement <4 x i32> %0, i64 1
  %4 = extractelement <4 x i32> %0, i64 2
  %5 = extractelement <4 x i32> %0, i64 3
  %6 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef %2, i32 noundef %3, i32 noundef %4, i32 noundef %5)
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @test_basic() local_unnamed_addr #1 {
  %1 = tail call i32 @puts(ptr nonnull dereferenceable(1) @__FUNCTION__.test_basic)
  %2 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.4)
  %3 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 1, i32 noundef 1, i32 noundef 1, i32 noundef 1, i32 noundef 1, i32 noundef 1, i32 noundef 1, i32 noundef 1, i32 noundef 1, i32 noundef 1, i32 noundef 1, i32 noundef 1, i32 noundef 1, i32 noundef 1, i32 noundef 1, i32 noundef 1)
  %4 = tail call i32 @putchar(i32 10)
  %5 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6)
  %6 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 0, i32 noundef 1, i32 noundef 2, i32 noundef 3, i32 noundef 4, i32 noundef 5, i32 noundef 6, i32 noundef 7, i32 noundef 8, i32 noundef 9, i32 noundef 10, i32 noundef 11, i32 noundef 12, i32 noundef 13, i32 noundef 14, i32 noundef 15)
  %7 = tail call i32 @putchar(i32 10)
  %8 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.7)
  %9 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 1, i32 noundef 2, i32 noundef 3, i32 noundef 4, i32 noundef 5, i32 noundef 6, i32 noundef 7, i32 noundef 8, i32 noundef 9, i32 noundef 10, i32 noundef 11, i32 noundef 12, i32 noundef 13, i32 noundef 14, i32 noundef 15, i32 noundef 16)
  %10 = tail call i32 @putchar(i32 10)
  %11 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.8)
  %12 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef 1, i32 noundef 1, i32 noundef 1, i32 noundef 1, i32 noundef 1, i32 noundef 1, i32 noundef 1, i32 noundef 1)
  %13 = tail call i32 @putchar(i32 10)
  %14 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.9)
  %15 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef 0, i32 noundef 1, i32 noundef 2, i32 noundef 3, i32 noundef 4, i32 noundef 5, i32 noundef 6, i32 noundef 7)
  %16 = tail call i32 @putchar(i32 10)
  %17 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.10)
  %18 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef 1, i32 noundef 2, i32 noundef 3, i32 noundef 4, i32 noundef 5, i32 noundef 6, i32 noundef 7, i32 noundef 8)
  %19 = tail call i32 @putchar(i32 10)
  %20 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.11)
  %21 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef 1, i32 noundef 1, i32 noundef 1, i32 noundef 1)
  %22 = tail call i32 @putchar(i32 10)
  %23 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.12)
  %24 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef 0, i32 noundef 1, i32 noundef 2, i32 noundef 3)
  %25 = tail call i32 @putchar(i32 10)
  %26 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.13)
  %27 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef 1, i32 noundef 2, i32 noundef 3, i32 noundef 4)
  %28 = tail call i32 @putchar(i32 10)
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @test_zip() local_unnamed_addr #1 {
  %1 = tail call i32 @puts(ptr nonnull dereferenceable(1) @__FUNCTION__.test_zip)
  %2 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.14)
  %3 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 0, i32 noundef 15, i32 noundef 1, i32 noundef 14, i32 noundef 2, i32 noundef 13, i32 noundef 3, i32 noundef 12, i32 noundef 4, i32 noundef 11, i32 noundef 5, i32 noundef 10, i32 noundef 6, i32 noundef 9, i32 noundef 7, i32 noundef 8)
  %4 = tail call i32 @putchar(i32 10)
  %5 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.15)
  %6 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 8, i32 noundef 7, i32 noundef 9, i32 noundef 6, i32 noundef 10, i32 noundef 5, i32 noundef 11, i32 noundef 4, i32 noundef 12, i32 noundef 3, i32 noundef 13, i32 noundef 2, i32 noundef 14, i32 noundef 1, i32 noundef 15, i32 noundef 0)
  %7 = tail call i32 @putchar(i32 10)
  %8 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.16)
  %9 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 0, i32 noundef 1, i32 noundef 2, i32 noundef 3, i32 noundef 4, i32 noundef 5, i32 noundef 6, i32 noundef 7, i32 noundef 8, i32 noundef 9, i32 noundef 10, i32 noundef 11, i32 noundef 12, i32 noundef 13, i32 noundef 14, i32 noundef 15)
  %10 = tail call i32 @putchar(i32 10)
  %11 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.17)
  %12 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 15, i32 noundef 14, i32 noundef 13, i32 noundef 12, i32 noundef 11, i32 noundef 10, i32 noundef 9, i32 noundef 8, i32 noundef 7, i32 noundef 6, i32 noundef 5, i32 noundef 4, i32 noundef 3, i32 noundef 2, i32 noundef 1, i32 noundef 0)
  %13 = tail call i32 @putchar(i32 10)
  %14 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.18)
  %15 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef 0, i32 noundef 7, i32 noundef 1, i32 noundef 6, i32 noundef 2, i32 noundef 5, i32 noundef 3, i32 noundef 4)
  %16 = tail call i32 @putchar(i32 10)
  %17 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.19)
  %18 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef 4, i32 noundef 3, i32 noundef 5, i32 noundef 2, i32 noundef 6, i32 noundef 1, i32 noundef 7, i32 noundef 0)
  %19 = tail call i32 @putchar(i32 10)
  %20 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.20)
  %21 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef 0, i32 noundef 1, i32 noundef 2, i32 noundef 3, i32 noundef 4, i32 noundef 5, i32 noundef 6, i32 noundef 7)
  %22 = tail call i32 @putchar(i32 10)
  %23 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.21)
  %24 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef 7, i32 noundef 6, i32 noundef 5, i32 noundef 4, i32 noundef 3, i32 noundef 2, i32 noundef 1, i32 noundef 0)
  %25 = tail call i32 @putchar(i32 10)
  %26 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.22)
  %27 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef 0, i32 noundef 3, i32 noundef 1, i32 noundef 2)
  %28 = tail call i32 @putchar(i32 10)
  %29 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.23)
  %30 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef 2, i32 noundef 1, i32 noundef 3, i32 noundef 0)
  %31 = tail call i32 @putchar(i32 10)
  %32 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.24)
  %33 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef 0, i32 noundef 1, i32 noundef 2, i32 noundef 3)
  %34 = tail call i32 @putchar(i32 10)
  %35 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.25)
  %36 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef 3, i32 noundef 2, i32 noundef 1, i32 noundef 0)
  %37 = tail call i32 @putchar(i32 10)
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  tail call void @test_basic()
  tail call void @test_zip()
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr noundef readonly captures(none)) local_unnamed_addr #3

; Function Attrs: nofree nounwind
declare noundef i32 @putchar(i32 noundef) local_unnamed_addr #3

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nofree nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
