; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/AArch64/acle-fmv-features.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/AArch64/acle-fmv-features.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@.str.1 = private unnamed_addr constant [6 x i8] c"flagm\00", align 1
@.str.2 = private unnamed_addr constant [7 x i8] c"flagm2\00", align 1
@.str.3 = private unnamed_addr constant [8 x i8] c"dotprod\00", align 1
@.str.4 = private unnamed_addr constant [5 x i8] c"sha3\00", align 1
@.str.5 = private unnamed_addr constant [4 x i8] c"rdm\00", align 1
@.str.6 = private unnamed_addr constant [4 x i8] c"lse\00", align 1
@.str.7 = private unnamed_addr constant [5 x i8] c"sha2\00", align 1
@.str.8 = private unnamed_addr constant [4 x i8] c"aes\00", align 1
@.str.9 = private unnamed_addr constant [5 x i8] c"rcpc\00", align 1
@.str.10 = private unnamed_addr constant [6 x i8] c"rcpc2\00", align 1
@.str.11 = private unnamed_addr constant [5 x i8] c"fcma\00", align 1
@.str.12 = private unnamed_addr constant [6 x i8] c"jscvt\00", align 1
@.str.13 = private unnamed_addr constant [4 x i8] c"dpb\00", align 1
@.str.14 = private unnamed_addr constant [5 x i8] c"dpb2\00", align 1
@.str.15 = private unnamed_addr constant [5 x i8] c"bf16\00", align 1
@.str.16 = private unnamed_addr constant [5 x i8] c"i8mm\00", align 1
@.str.17 = private unnamed_addr constant [4 x i8] c"dit\00", align 1
@.str.18 = private unnamed_addr constant [5 x i8] c"fp16\00", align 1
@.str.19 = private unnamed_addr constant [5 x i8] c"ssbs\00", align 1
@.str.20 = private unnamed_addr constant [4 x i8] c"bti\00", align 1
@.str.21 = private unnamed_addr constant [5 x i8] c"simd\00", align 1
@.str.22 = private unnamed_addr constant [3 x i8] c"fp\00", align 1
@.str.23 = private unnamed_addr constant [4 x i8] c"crc\00", align 1
@.str.24 = private unnamed_addr constant [4 x i8] c"sme\00", align 1
@.str.25 = private unnamed_addr constant [5 x i8] c"sme2\00", align 1
@.str.26 = private unnamed_addr constant [6 x i8] c"f32mm\00", align 1
@.str.27 = private unnamed_addr constant [6 x i8] c"f64mm\00", align 1
@.str.28 = private unnamed_addr constant [8 x i8] c"fp16fml\00", align 1
@.str.29 = private unnamed_addr constant [8 x i8] c"frintts\00", align 1
@.str.30 = private unnamed_addr constant [6 x i8] c"rcpc3\00", align 1
@.str.31 = private unnamed_addr constant [4 x i8] c"rng\00", align 1
@.str.32 = private unnamed_addr constant [4 x i8] c"sve\00", align 1
@.str.33 = private unnamed_addr constant [5 x i8] c"sve2\00", align 1
@.str.34 = private unnamed_addr constant [9 x i8] c"sve2-aes\00", align 1
@.str.35 = private unnamed_addr constant [13 x i8] c"sve2-bitperm\00", align 1
@.str.36 = private unnamed_addr constant [10 x i8] c"sve2-sha3\00", align 1
@.str.37 = private unnamed_addr constant [9 x i8] c"sve2-sm4\00", align 1
@.str.38 = private unnamed_addr constant [5 x i8] c"wfxt\00", align 1
@.str.39 = private unnamed_addr constant [3 x i8] c"sb\00", align 1
@.str.40 = private unnamed_addr constant [4 x i8] c"sm4\00", align 1
@.str.41 = private unnamed_addr constant [11 x i8] c"sme-f64f64\00", align 1
@.str.42 = private unnamed_addr constant [11 x i8] c"sme-i16i64\00", align 1
@.str.43 = private unnamed_addr constant [5 x i8] c"mops\00", align 1
@.str.44 = private unnamed_addr constant [7 x i8] c"memtag\00", align 1
@.str.45 = private unnamed_addr constant [5 x i8] c"cssc\00", align 1

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readnone captures(none) %1) local_unnamed_addr #0 {
  %3 = tail call i32 @puts(ptr nonnull dereferenceable(1) @.str.1)
  %4 = tail call i32 @puts(ptr nonnull dereferenceable(1) @.str.2)
  %5 = tail call i32 @puts(ptr nonnull dereferenceable(1) @.str.3)
  %6 = tail call i32 @puts(ptr nonnull dereferenceable(1) @.str.4)
  %7 = tail call i32 @puts(ptr nonnull dereferenceable(1) @.str.5)
  %8 = tail call i32 @puts(ptr nonnull dereferenceable(1) @.str.6)
  %9 = tail call i32 @puts(ptr nonnull dereferenceable(1) @.str.7)
  %10 = tail call i32 @puts(ptr nonnull dereferenceable(1) @.str.8)
  %11 = tail call i32 @puts(ptr nonnull dereferenceable(1) @.str.9)
  %12 = tail call i32 @puts(ptr nonnull dereferenceable(1) @.str.10)
  %13 = tail call i32 @puts(ptr nonnull dereferenceable(1) @.str.11)
  %14 = tail call i32 @puts(ptr nonnull dereferenceable(1) @.str.12)
  %15 = tail call i32 @puts(ptr nonnull dereferenceable(1) @.str.13)
  %16 = tail call i32 @puts(ptr nonnull dereferenceable(1) @.str.14)
  %17 = tail call i32 @puts(ptr nonnull dereferenceable(1) @.str.15)
  %18 = tail call i32 @puts(ptr nonnull dereferenceable(1) @.str.16)
  %19 = tail call i32 @puts(ptr nonnull dereferenceable(1) @.str.17)
  %20 = tail call i32 @puts(ptr nonnull dereferenceable(1) @.str.18)
  %21 = tail call i32 @puts(ptr nonnull dereferenceable(1) @.str.19)
  %22 = tail call i32 @puts(ptr nonnull dereferenceable(1) @.str.20)
  %23 = tail call i32 @puts(ptr nonnull dereferenceable(1) @.str.21)
  %24 = tail call i32 @puts(ptr nonnull dereferenceable(1) @.str.22)
  %25 = tail call i32 @puts(ptr nonnull dereferenceable(1) @.str.23)
  %26 = tail call i32 @puts(ptr nonnull dereferenceable(1) @.str.24)
  %27 = tail call i32 @puts(ptr nonnull dereferenceable(1) @.str.25)
  %28 = tail call i32 @puts(ptr nonnull dereferenceable(1) @.str.26)
  %29 = tail call i32 @puts(ptr nonnull dereferenceable(1) @.str.27)
  %30 = tail call i32 @puts(ptr nonnull dereferenceable(1) @.str.28)
  %31 = tail call i32 @puts(ptr nonnull dereferenceable(1) @.str.29)
  %32 = tail call i32 @puts(ptr nonnull dereferenceable(1) @.str.30)
  %33 = tail call i32 @puts(ptr nonnull dereferenceable(1) @.str.31)
  %34 = tail call i32 @puts(ptr nonnull dereferenceable(1) @.str.32)
  %35 = tail call i32 @puts(ptr nonnull dereferenceable(1) @.str.33)
  %36 = tail call i32 @puts(ptr nonnull dereferenceable(1) @.str.34)
  %37 = tail call i32 @puts(ptr nonnull dereferenceable(1) @.str.35)
  %38 = tail call i32 @puts(ptr nonnull dereferenceable(1) @.str.36)
  %39 = tail call i32 @puts(ptr nonnull dereferenceable(1) @.str.37)
  %40 = tail call i32 @puts(ptr nonnull dereferenceable(1) @.str.38)
  %41 = tail call i32 @puts(ptr nonnull dereferenceable(1) @.str.39)
  %42 = tail call i32 @puts(ptr nonnull dereferenceable(1) @.str.40)
  %43 = tail call i32 @puts(ptr nonnull dereferenceable(1) @.str.41)
  %44 = tail call i32 @puts(ptr nonnull dereferenceable(1) @.str.42)
  %45 = tail call i32 @puts(ptr nonnull dereferenceable(1) @.str.43)
  %46 = tail call i32 @puts(ptr nonnull dereferenceable(1) @.str.44)
  %47 = tail call i32 @puts(ptr nonnull dereferenceable(1) @.str.45)
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr noundef readonly captures(none)) local_unnamed_addr #1

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
