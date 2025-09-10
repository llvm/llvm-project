; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/Vector/simple.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/Vector/simple.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@.str = private unnamed_addr constant [13 x i8] c"%f %f %f %f\0A\00", align 1
@.str.1 = private unnamed_addr constant [7 x i8] c"%g %g\0A\00", align 1

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readnone captures(none) %1) local_unnamed_addr #0 {
  %3 = insertelement <2 x i32> poison, i32 %0, i64 0
  %4 = shufflevector <2 x i32> %3, <2 x i32> poison, <2 x i32> zeroinitializer
  %5 = icmp eq <2 x i32> %4, <i32 2123, i32 5123>
  %6 = icmp eq i32 %0, 2123
  %7 = select <2 x i1> %5, <2 x double> <double 0x4050693404EA4A8C, double 0x4063732FEC56D5D0>, <2 x double> <double 0x409B49779A6B50B1, double 0x40ACCB9C779A6B51>
  %8 = select i1 %6, double 0x4050693404EA4A8C, double 0x409B49779A6B50B1
  %9 = icmp eq i32 %0, 1432
  %10 = select i1 %9, float 0x401EE0B780000000, float 0x4023C08320000000
  %11 = insertelement <4 x float> <float poison, float poison, float 0x3FF1C6A7E0000000, float 0x3FF1C6A7E0000000>, float %10, i64 0
  %12 = insertelement <4 x float> %11, float %10, i64 1
  %13 = fadd <4 x float> %12, %12
  %14 = icmp eq i32 %0, 1123
  %15 = select i1 %14, float 0x40030E9A20000000, float 0x3FF3BE76C0000000
  %16 = insertelement <4 x float> poison, float %15, i64 0
  %17 = shufflevector <4 x float> %16, <4 x float> poison, <4 x i32> <i32 0, i32 0, i32 poison, i32 poison>
  %18 = insertelement <4 x float> %17, float %10, i64 2
  %19 = insertelement <4 x float> %18, float %10, i64 3
  %20 = fadd <4 x float> %19, %19
  %21 = shufflevector <4 x float> %17, <4 x float> <float poison, float poison, float 0.000000e+00, float 0.000000e+00>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %22 = fadd <4 x float> %21, %21
  %23 = select i1 %14, double 0x4016B2BB60000000, double 0x3FF85D3540000000
  %24 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, double noundef %23, double noundef %23, double noundef %23, double noundef %23)
  %25 = extractelement <4 x float> %22, i64 0
  %26 = fpext float %25 to double
  %27 = extractelement <4 x float> %22, i64 1
  %28 = fpext float %27 to double
  %29 = extractelement <4 x float> %22, i64 2
  %30 = fpext float %29 to double
  %31 = extractelement <4 x float> %22, i64 3
  %32 = fpext float %31 to double
  %33 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, double noundef %26, double noundef %28, double noundef %30, double noundef %32)
  %34 = extractelement <4 x float> %20, i64 0
  %35 = fpext float %34 to double
  %36 = extractelement <4 x float> %20, i64 1
  %37 = fpext float %36 to double
  %38 = extractelement <4 x float> %20, i64 2
  %39 = fpext float %38 to double
  %40 = extractelement <4 x float> %20, i64 3
  %41 = fpext float %40 to double
  %42 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, double noundef %35, double noundef %37, double noundef %39, double noundef %41)
  %43 = extractelement <4 x float> %13, i64 0
  %44 = fpext float %43 to double
  %45 = extractelement <4 x float> %13, i64 1
  %46 = fpext float %45 to double
  %47 = extractelement <4 x float> %13, i64 2
  %48 = fpext float %47 to double
  %49 = extractelement <4 x float> %13, i64 3
  %50 = fpext float %49 to double
  %51 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, double noundef %44, double noundef %46, double noundef %48, double noundef %50)
  %52 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, double noundef %8, double noundef %8)
  %53 = extractelement <2 x double> %7, i64 0
  %54 = extractelement <2 x double> %7, i64 1
  %55 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, double noundef %53, double noundef %54)
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #1

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
