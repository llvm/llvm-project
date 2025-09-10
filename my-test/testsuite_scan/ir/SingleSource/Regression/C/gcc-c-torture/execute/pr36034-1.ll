; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr36034-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr36034-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@x = dso_local local_unnamed_addr global [5 x [10 x double]] [[10 x double] [double 1.000000e+01, double 1.100000e+01, double 1.200000e+01, double 1.300000e+01, double 1.400000e+01, double 1.500000e+01, double -1.000000e+00, double -1.000000e+00, double -1.000000e+00, double -1.000000e+00], [10 x double] [double 2.100000e+01, double 2.200000e+01, double 2.300000e+01, double 2.400000e+01, double 2.500000e+01, double 2.600000e+01, double -1.000000e+00, double -1.000000e+00, double -1.000000e+00, double -1.000000e+00], [10 x double] [double 3.200000e+01, double 3.300000e+01, double 3.400000e+01, double 3.500000e+01, double 3.600000e+01, double 3.700000e+01, double -1.000000e+00, double -1.000000e+00, double -1.000000e+00, double -1.000000e+00], [10 x double] [double 4.300000e+01, double 4.400000e+01, double 4.500000e+01, double 4.600000e+01, double 4.700000e+01, double 4.800000e+01, double -1.000000e+00, double -1.000000e+00, double -1.000000e+00, double -1.000000e+00], [10 x double] [double 5.400000e+01, double 5.500000e+01, double 5.600000e+01, double 5.700000e+01, double 5.800000e+01, double 5.900000e+01, double -1.000000e+00, double -1.000000e+00, double -1.000000e+00, double -1.000000e+00]], align 16
@tmp = dso_local local_unnamed_addr global [5 x [6 x double]] zeroinitializer, align 128

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @test() local_unnamed_addr #0 {
  %1 = load <2 x double>, ptr @x, align 16, !tbaa !6
  store <2 x double> %1, ptr @tmp, align 16, !tbaa !6
  %2 = load <2 x double>, ptr getelementptr inbounds nuw (i8, ptr @x, i64 16), align 16, !tbaa !6
  store <2 x double> %2, ptr getelementptr inbounds nuw (i8, ptr @tmp, i64 16), align 16, !tbaa !6
  %3 = load <2 x double>, ptr getelementptr inbounds nuw (i8, ptr @x, i64 32), align 16, !tbaa !6
  store <2 x double> %3, ptr getelementptr inbounds nuw (i8, ptr @tmp, i64 32), align 16, !tbaa !6
  %4 = load <2 x double>, ptr getelementptr inbounds nuw (i8, ptr @x, i64 80), align 16, !tbaa !6
  store <2 x double> %4, ptr getelementptr inbounds nuw (i8, ptr @tmp, i64 48), align 16, !tbaa !6
  %5 = load <2 x double>, ptr getelementptr inbounds nuw (i8, ptr @x, i64 96), align 16, !tbaa !6
  store <2 x double> %5, ptr getelementptr inbounds nuw (i8, ptr @tmp, i64 64), align 16, !tbaa !6
  %6 = load <2 x double>, ptr getelementptr inbounds nuw (i8, ptr @x, i64 112), align 16, !tbaa !6
  store <2 x double> %6, ptr getelementptr inbounds nuw (i8, ptr @tmp, i64 80), align 16, !tbaa !6
  %7 = load <2 x double>, ptr getelementptr inbounds nuw (i8, ptr @x, i64 160), align 16, !tbaa !6
  store <2 x double> %7, ptr getelementptr inbounds nuw (i8, ptr @tmp, i64 96), align 16, !tbaa !6
  %8 = load <2 x double>, ptr getelementptr inbounds nuw (i8, ptr @x, i64 176), align 16, !tbaa !6
  store <2 x double> %8, ptr getelementptr inbounds nuw (i8, ptr @tmp, i64 112), align 16, !tbaa !6
  %9 = load <2 x double>, ptr getelementptr inbounds nuw (i8, ptr @x, i64 192), align 16, !tbaa !6
  store <2 x double> %9, ptr getelementptr inbounds nuw (i8, ptr @tmp, i64 128), align 16, !tbaa !6
  %10 = load <2 x double>, ptr getelementptr inbounds nuw (i8, ptr @x, i64 240), align 16, !tbaa !6
  store <2 x double> %10, ptr getelementptr inbounds nuw (i8, ptr @tmp, i64 144), align 16, !tbaa !6
  %11 = load <2 x double>, ptr getelementptr inbounds nuw (i8, ptr @x, i64 256), align 16, !tbaa !6
  store <2 x double> %11, ptr getelementptr inbounds nuw (i8, ptr @tmp, i64 160), align 16, !tbaa !6
  %12 = load <2 x double>, ptr getelementptr inbounds nuw (i8, ptr @x, i64 272), align 16, !tbaa !6
  store <2 x double> %12, ptr getelementptr inbounds nuw (i8, ptr @tmp, i64 176), align 16, !tbaa !6
  %13 = load <2 x double>, ptr getelementptr inbounds nuw (i8, ptr @x, i64 320), align 16, !tbaa !6
  store <2 x double> %13, ptr getelementptr inbounds nuw (i8, ptr @tmp, i64 192), align 16, !tbaa !6
  %14 = load <2 x double>, ptr getelementptr inbounds nuw (i8, ptr @x, i64 336), align 16, !tbaa !6
  store <2 x double> %14, ptr getelementptr inbounds nuw (i8, ptr @tmp, i64 208), align 16, !tbaa !6
  %15 = load <2 x double>, ptr getelementptr inbounds nuw (i8, ptr @x, i64 352), align 16, !tbaa !6
  store <2 x double> %15, ptr getelementptr inbounds nuw (i8, ptr @tmp, i64 224), align 16, !tbaa !6
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  tail call void @test()
  %1 = load <16 x double>, ptr @tmp, align 128
  %2 = freeze <16 x double> %1
  %3 = fcmp oeq <16 x double> %2, splat (double -1.000000e+00)
  %4 = load <8 x double>, ptr getelementptr inbounds nuw (i8, ptr @tmp, i64 128), align 128
  %5 = freeze <8 x double> %4
  %6 = fcmp oeq <8 x double> %5, splat (double -1.000000e+00)
  %7 = load <4 x double>, ptr getelementptr inbounds nuw (i8, ptr @tmp, i64 192), align 64
  %8 = freeze <4 x double> %7
  %9 = fcmp oeq <4 x double> %8, splat (double -1.000000e+00)
  %10 = load double, ptr getelementptr inbounds nuw (i8, ptr @tmp, i64 224), align 32
  %11 = freeze double %10
  %12 = fcmp oeq double %11, -1.000000e+00
  %13 = load double, ptr getelementptr inbounds nuw (i8, ptr @tmp, i64 232), align 8
  %14 = fcmp oeq double %13, -1.000000e+00
  %15 = shufflevector <16 x i1> %3, <16 x i1> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %16 = or <8 x i1> %15, %6
  %17 = shufflevector <8 x i1> %16, <8 x i1> poison, <16 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %18 = shufflevector <16 x i1> %17, <16 x i1> %3, <16 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 4, i32 5, i32 6, i32 7, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %19 = shufflevector <8 x i1> %16, <8 x i1> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %20 = or <4 x i1> %19, %9
  %21 = shufflevector <4 x i1> %20, <4 x i1> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %22 = freeze <16 x i1> %21
  %23 = freeze <16 x i1> %18
  %24 = shufflevector <16 x i1> %22, <16 x i1> %23, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %25 = bitcast <16 x i1> %24 to i16
  %26 = icmp ne i16 %25, 0
  %27 = or i1 %26, %12
  %28 = select i1 %27, i1 true, i1 %14
  br i1 %28, label %30, label %29

29:                                               ; preds = %0
  ret i32 0

30:                                               ; preds = %0
  tail call void @abort() #3
  unreachable
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #2

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"double", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
