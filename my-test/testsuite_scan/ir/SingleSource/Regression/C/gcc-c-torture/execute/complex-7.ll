; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/complex-7.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/complex-7.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@f1 = dso_local global { float, float } { float 0x3FF19999A0000000, float 0x40019999A0000000 }, align 4
@f2 = dso_local global { float, float } { float 0x400A666660000000, float 0x40119999A0000000 }, align 4
@f3 = dso_local global { float, float } { float 5.500000e+00, float 0x401A666660000000 }, align 4
@f4 = dso_local global { float, float } { float 0x401ECCCCC0000000, float 0x40219999A0000000 }, align 4
@f5 = dso_local global { float, float } { float 0x4023CCCCC0000000, float 0x4024333340000000 }, align 4
@d1 = dso_local global { double, double } { double 1.100000e+00, double 2.200000e+00 }, align 8
@d2 = dso_local global { double, double } { double 3.300000e+00, double 4.400000e+00 }, align 8
@d3 = dso_local global { double, double } { double 5.500000e+00, double 6.600000e+00 }, align 8
@d4 = dso_local global { double, double } { double 7.700000e+00, double 8.800000e+00 }, align 8
@d5 = dso_local global { double, double } { double 9.900000e+00, double 1.010000e+01 }, align 8
@ld1 = dso_local global { fp128, fp128 } { fp128 0xL999999999999999A3FFF199999999999, fp128 0xL999999999999999A4000199999999999 }, align 16
@ld2 = dso_local global { fp128, fp128 } { fp128 0xL66666666666666664000A66666666666, fp128 0xL999999999999999A4001199999999999 }, align 16
@ld3 = dso_local global { fp128, fp128 } { fp128 0xL00000000000000004001600000000000, fp128 0xL66666666666666664001A66666666666 }, align 16
@ld4 = dso_local global { fp128, fp128 } { fp128 0xLCCCCCCCCCCCCCCCD4001ECCCCCCCCCCC, fp128 0xL999999999999999A4002199999999999 }, align 16
@ld5 = dso_local global { fp128, fp128 } { fp128 0xLCCCCCCCCCCCCCCCD40023CCCCCCCCCCC, fp128 0xL33333333333333334002433333333333 }, align 16

; Function Attrs: nofree noinline nounwind uwtable
define dso_local void @check_float(i32 %0, [2 x float] noundef alignstack(8) %1, [2 x float] noundef alignstack(8) %2, [2 x float] noundef alignstack(8) %3, [2 x float] noundef alignstack(8) %4, [2 x float] noundef alignstack(8) %5) local_unnamed_addr #0 {
  %7 = extractvalue [2 x float] %1, 0
  %8 = extractvalue [2 x float] %1, 1
  %9 = extractvalue [2 x float] %3, 0
  %10 = extractvalue [2 x float] %3, 1
  %11 = extractvalue [2 x float] %4, 0
  %12 = extractvalue [2 x float] %4, 1
  %13 = extractvalue [2 x float] %5, 0
  %14 = extractvalue [2 x float] %5, 1
  %15 = load volatile float, ptr @f1, align 4
  %16 = load volatile float, ptr getelementptr inbounds nuw (i8, ptr @f1, i64 4), align 4
  %17 = fcmp une float %7, %15
  %18 = fcmp une float %8, %16
  %19 = or i1 %17, %18
  br i1 %19, label %46, label %20

20:                                               ; preds = %6
  %21 = extractvalue [2 x float] %2, 1
  %22 = extractvalue [2 x float] %2, 0
  %23 = load volatile float, ptr @f2, align 4
  %24 = load volatile float, ptr getelementptr inbounds nuw (i8, ptr @f2, i64 4), align 4
  %25 = fcmp une float %22, %23
  %26 = fcmp une float %21, %24
  %27 = or i1 %25, %26
  br i1 %27, label %46, label %28

28:                                               ; preds = %20
  %29 = load volatile float, ptr @f3, align 4
  %30 = load volatile float, ptr getelementptr inbounds nuw (i8, ptr @f3, i64 4), align 4
  %31 = fcmp une float %9, %29
  %32 = fcmp une float %10, %30
  %33 = or i1 %31, %32
  br i1 %33, label %46, label %34

34:                                               ; preds = %28
  %35 = load volatile float, ptr @f4, align 4
  %36 = load volatile float, ptr getelementptr inbounds nuw (i8, ptr @f4, i64 4), align 4
  %37 = fcmp une float %11, %35
  %38 = fcmp une float %12, %36
  %39 = or i1 %37, %38
  br i1 %39, label %46, label %40

40:                                               ; preds = %34
  %41 = load volatile float, ptr @f5, align 4
  %42 = load volatile float, ptr getelementptr inbounds nuw (i8, ptr @f5, i64 4), align 4
  %43 = fcmp une float %13, %41
  %44 = fcmp une float %14, %42
  %45 = or i1 %43, %44
  br i1 %45, label %46, label %47

46:                                               ; preds = %40, %34, %28, %20, %6
  tail call void @abort() #4
  unreachable

47:                                               ; preds = %40
  ret void
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: nofree noinline nounwind uwtable
define dso_local void @check_double(i32 %0, [2 x double] noundef alignstack(8) %1, [2 x double] noundef alignstack(8) %2, [2 x double] noundef alignstack(8) %3, [2 x double] noundef alignstack(8) %4, [2 x double] noundef alignstack(8) %5) local_unnamed_addr #0 {
  %7 = extractvalue [2 x double] %1, 0
  %8 = extractvalue [2 x double] %1, 1
  %9 = extractvalue [2 x double] %3, 0
  %10 = extractvalue [2 x double] %3, 1
  %11 = extractvalue [2 x double] %4, 0
  %12 = extractvalue [2 x double] %4, 1
  %13 = extractvalue [2 x double] %5, 0
  %14 = extractvalue [2 x double] %5, 1
  %15 = load volatile double, ptr @d1, align 8
  %16 = load volatile double, ptr getelementptr inbounds nuw (i8, ptr @d1, i64 8), align 8
  %17 = fcmp une double %7, %15
  %18 = fcmp une double %8, %16
  %19 = or i1 %17, %18
  br i1 %19, label %46, label %20

20:                                               ; preds = %6
  %21 = extractvalue [2 x double] %2, 1
  %22 = extractvalue [2 x double] %2, 0
  %23 = load volatile double, ptr @d2, align 8
  %24 = load volatile double, ptr getelementptr inbounds nuw (i8, ptr @d2, i64 8), align 8
  %25 = fcmp une double %22, %23
  %26 = fcmp une double %21, %24
  %27 = or i1 %25, %26
  br i1 %27, label %46, label %28

28:                                               ; preds = %20
  %29 = load volatile double, ptr @d3, align 8
  %30 = load volatile double, ptr getelementptr inbounds nuw (i8, ptr @d3, i64 8), align 8
  %31 = fcmp une double %9, %29
  %32 = fcmp une double %10, %30
  %33 = or i1 %31, %32
  br i1 %33, label %46, label %34

34:                                               ; preds = %28
  %35 = load volatile double, ptr @d4, align 8
  %36 = load volatile double, ptr getelementptr inbounds nuw (i8, ptr @d4, i64 8), align 8
  %37 = fcmp une double %11, %35
  %38 = fcmp une double %12, %36
  %39 = or i1 %37, %38
  br i1 %39, label %46, label %40

40:                                               ; preds = %34
  %41 = load volatile double, ptr @d5, align 8
  %42 = load volatile double, ptr getelementptr inbounds nuw (i8, ptr @d5, i64 8), align 8
  %43 = fcmp une double %13, %41
  %44 = fcmp une double %14, %42
  %45 = or i1 %43, %44
  br i1 %45, label %46, label %47

46:                                               ; preds = %40, %34, %28, %20, %6
  tail call void @abort() #4
  unreachable

47:                                               ; preds = %40
  ret void
}

; Function Attrs: nofree noinline nounwind uwtable
define dso_local void @check_long_double(i32 %0, [2 x fp128] noundef alignstack(16) %1, [2 x fp128] noundef alignstack(16) %2, [2 x fp128] noundef alignstack(16) %3, [2 x fp128] noundef alignstack(16) %4, [2 x fp128] noundef alignstack(16) %5) local_unnamed_addr #0 {
  %7 = extractvalue [2 x fp128] %1, 0
  %8 = extractvalue [2 x fp128] %1, 1
  %9 = extractvalue [2 x fp128] %3, 0
  %10 = extractvalue [2 x fp128] %3, 1
  %11 = extractvalue [2 x fp128] %4, 0
  %12 = extractvalue [2 x fp128] %4, 1
  %13 = extractvalue [2 x fp128] %5, 0
  %14 = extractvalue [2 x fp128] %5, 1
  %15 = load volatile fp128, ptr @ld1, align 16
  %16 = load volatile fp128, ptr getelementptr inbounds nuw (i8, ptr @ld1, i64 16), align 16
  %17 = fcmp une fp128 %7, %15
  %18 = fcmp une fp128 %8, %16
  %19 = or i1 %17, %18
  br i1 %19, label %46, label %20

20:                                               ; preds = %6
  %21 = extractvalue [2 x fp128] %2, 1
  %22 = extractvalue [2 x fp128] %2, 0
  %23 = load volatile fp128, ptr @ld2, align 16
  %24 = load volatile fp128, ptr getelementptr inbounds nuw (i8, ptr @ld2, i64 16), align 16
  %25 = fcmp une fp128 %22, %23
  %26 = fcmp une fp128 %21, %24
  %27 = or i1 %25, %26
  br i1 %27, label %46, label %28

28:                                               ; preds = %20
  %29 = load volatile fp128, ptr @ld3, align 16
  %30 = load volatile fp128, ptr getelementptr inbounds nuw (i8, ptr @ld3, i64 16), align 16
  %31 = fcmp une fp128 %9, %29
  %32 = fcmp une fp128 %10, %30
  %33 = or i1 %31, %32
  br i1 %33, label %46, label %34

34:                                               ; preds = %28
  %35 = load volatile fp128, ptr @ld4, align 16
  %36 = load volatile fp128, ptr getelementptr inbounds nuw (i8, ptr @ld4, i64 16), align 16
  %37 = fcmp une fp128 %11, %35
  %38 = fcmp une fp128 %12, %36
  %39 = or i1 %37, %38
  br i1 %39, label %46, label %40

40:                                               ; preds = %34
  %41 = load volatile fp128, ptr @ld5, align 16
  %42 = load volatile fp128, ptr getelementptr inbounds nuw (i8, ptr @ld5, i64 16), align 16
  %43 = fcmp une fp128 %13, %41
  %44 = fcmp une fp128 %14, %42
  %45 = or i1 %43, %44
  br i1 %45, label %46, label %47

46:                                               ; preds = %40, %34, %28, %20, %6
  tail call void @abort() #4
  unreachable

47:                                               ; preds = %40
  ret void
}

; Function Attrs: nofree noreturn nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 {
  %1 = load volatile float, ptr @f1, align 4
  %2 = load volatile float, ptr getelementptr inbounds nuw (i8, ptr @f1, i64 4), align 4
  %3 = load volatile float, ptr @f2, align 4
  %4 = load volatile float, ptr getelementptr inbounds nuw (i8, ptr @f2, i64 4), align 4
  %5 = load volatile float, ptr @f3, align 4
  %6 = load volatile float, ptr getelementptr inbounds nuw (i8, ptr @f3, i64 4), align 4
  %7 = load volatile float, ptr @f4, align 4
  %8 = load volatile float, ptr getelementptr inbounds nuw (i8, ptr @f4, i64 4), align 4
  %9 = load volatile float, ptr @f5, align 4
  %10 = load volatile float, ptr getelementptr inbounds nuw (i8, ptr @f5, i64 4), align 4
  %11 = insertvalue [2 x float] poison, float %1, 0
  %12 = insertvalue [2 x float] %11, float %2, 1
  %13 = insertvalue [2 x float] poison, float %3, 0
  %14 = insertvalue [2 x float] %13, float %4, 1
  %15 = insertvalue [2 x float] poison, float %5, 0
  %16 = insertvalue [2 x float] %15, float %6, 1
  %17 = insertvalue [2 x float] poison, float %7, 0
  %18 = insertvalue [2 x float] %17, float %8, 1
  %19 = insertvalue [2 x float] poison, float %9, 0
  %20 = insertvalue [2 x float] %19, float %10, 1
  tail call void @check_float(i32 poison, [2 x float] noundef alignstack(8) %12, [2 x float] noundef alignstack(8) %14, [2 x float] noundef alignstack(8) %16, [2 x float] noundef alignstack(8) %18, [2 x float] noundef alignstack(8) %20)
  %21 = load volatile double, ptr @d1, align 8
  %22 = load volatile double, ptr getelementptr inbounds nuw (i8, ptr @d1, i64 8), align 8
  %23 = load volatile double, ptr @d2, align 8
  %24 = load volatile double, ptr getelementptr inbounds nuw (i8, ptr @d2, i64 8), align 8
  %25 = load volatile double, ptr @d3, align 8
  %26 = load volatile double, ptr getelementptr inbounds nuw (i8, ptr @d3, i64 8), align 8
  %27 = load volatile double, ptr @d4, align 8
  %28 = load volatile double, ptr getelementptr inbounds nuw (i8, ptr @d4, i64 8), align 8
  %29 = load volatile double, ptr @d5, align 8
  %30 = load volatile double, ptr getelementptr inbounds nuw (i8, ptr @d5, i64 8), align 8
  %31 = insertvalue [2 x double] poison, double %21, 0
  %32 = insertvalue [2 x double] %31, double %22, 1
  %33 = insertvalue [2 x double] poison, double %23, 0
  %34 = insertvalue [2 x double] %33, double %24, 1
  %35 = insertvalue [2 x double] poison, double %25, 0
  %36 = insertvalue [2 x double] %35, double %26, 1
  %37 = insertvalue [2 x double] poison, double %27, 0
  %38 = insertvalue [2 x double] %37, double %28, 1
  %39 = insertvalue [2 x double] poison, double %29, 0
  %40 = insertvalue [2 x double] %39, double %30, 1
  tail call void @check_double(i32 poison, [2 x double] noundef alignstack(8) %32, [2 x double] noundef alignstack(8) %34, [2 x double] noundef alignstack(8) %36, [2 x double] noundef alignstack(8) %38, [2 x double] noundef alignstack(8) %40)
  %41 = load volatile fp128, ptr @ld1, align 16
  %42 = load volatile fp128, ptr getelementptr inbounds nuw (i8, ptr @ld1, i64 16), align 16
  %43 = load volatile fp128, ptr @ld2, align 16
  %44 = load volatile fp128, ptr getelementptr inbounds nuw (i8, ptr @ld2, i64 16), align 16
  %45 = load volatile fp128, ptr @ld3, align 16
  %46 = load volatile fp128, ptr getelementptr inbounds nuw (i8, ptr @ld3, i64 16), align 16
  %47 = load volatile fp128, ptr @ld4, align 16
  %48 = load volatile fp128, ptr getelementptr inbounds nuw (i8, ptr @ld4, i64 16), align 16
  %49 = load volatile fp128, ptr @ld5, align 16
  %50 = load volatile fp128, ptr getelementptr inbounds nuw (i8, ptr @ld5, i64 16), align 16
  %51 = insertvalue [2 x fp128] poison, fp128 %41, 0
  %52 = insertvalue [2 x fp128] %51, fp128 %42, 1
  %53 = insertvalue [2 x fp128] poison, fp128 %43, 0
  %54 = insertvalue [2 x fp128] %53, fp128 %44, 1
  %55 = insertvalue [2 x fp128] poison, fp128 %45, 0
  %56 = insertvalue [2 x fp128] %55, fp128 %46, 1
  %57 = insertvalue [2 x fp128] poison, fp128 %47, 0
  %58 = insertvalue [2 x fp128] %57, fp128 %48, 1
  %59 = insertvalue [2 x fp128] poison, fp128 %49, 0
  %60 = insertvalue [2 x fp128] %59, fp128 %50, 1
  tail call void @check_long_double(i32 poison, [2 x fp128] noundef alignstack(16) %52, [2 x fp128] noundef alignstack(16) %54, [2 x fp128] noundef alignstack(16) %56, [2 x fp128] noundef alignstack(16) %58, [2 x fp128] noundef alignstack(16) %60)
  tail call void @exit(i32 noundef 0) #4
  unreachable
}

; Function Attrs: nofree noreturn
declare void @exit(i32 noundef) local_unnamed_addr #3

attributes #0 = { nofree noinline nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
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
