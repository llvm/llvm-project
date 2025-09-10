; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/BenchmarkGame/n-body.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/BenchmarkGame/n-body.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.planet = type { double, double, double, double, double, double, double }

@bodies = dso_local global [5 x %struct.planet] [%struct.planet { double 0.000000e+00, double 0.000000e+00, double 0.000000e+00, double 0.000000e+00, double 0.000000e+00, double 0.000000e+00, double 0x4043BD3CC9BE45DE }, %struct.planet { double 0x40135DA0343CD92C, double 0xBFF290ABC01FDB7C, double 0xBFBA86F96C25EBF0, double 0x3FE367069B93CCBC, double 0x40067EF2F57D949B, double 0xBF99D2D79A5A0715, double 0x3FA34C95D9AB33D8 }, %struct.planet { double 0x4020AFCDC332CA67, double 0x40107FCB31DE01B0, double 0xBFD9D353E1EB467C, double 0xBFF02C21B8879442, double 0x3FFD35E9BF1F8F13, double 0x3F813C485F1123B4, double 0x3F871D490D07C637 }, %struct.planet { double 0x4029C9EACEA7D9CF, double 0xC02E38E8D626667E, double 0xBFCC9557BE257DA0, double 0x3FF1531CA9911BEF, double 0x3FEBCC7F3E54BBC5, double 0xBF862F6BFAF23E7C, double 0x3F5C3DD29CF41EB3 }, %struct.planet { double 0x402EC267A905572A, double 0xC039EB5833C8A220, double 0x3FC6F1F393ABE540, double 0x3FEF54B61659BC4A, double 0x3FE307C631C4FBA3, double 0xBFA1CB88587665F6, double 0x3F60A8F3531799AC }], align 8
@.str = private unnamed_addr constant [6 x i8] c"%.9f\0A\00", align 1

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable
define dso_local void @advance(i32 noundef %0, ptr noundef captures(none) %1, double noundef %2) local_unnamed_addr #0 {
  %4 = icmp sgt i32 %0, 0
  br i1 %4, label %5, label %85

5:                                                ; preds = %3
  %6 = zext nneg i32 %0 to i64
  %7 = zext nneg i32 %0 to i64
  br label %15

8:                                                ; preds = %31, %15
  %9 = add nuw nsw i64 %17, 1
  %10 = icmp eq i64 %18, %7
  br i1 %10, label %11, label %15, !llvm.loop !6

11:                                               ; preds = %8
  %12 = zext nneg i32 %0 to i64
  %13 = insertelement <2 x double> poison, double %2, i64 0
  %14 = shufflevector <2 x double> %13, <2 x double> poison, <2 x i32> zeroinitializer
  br label %71

15:                                               ; preds = %5, %8
  %16 = phi i64 [ 0, %5 ], [ %18, %8 ]
  %17 = phi i64 [ 1, %5 ], [ %9, %8 ]
  %18 = add nuw nsw i64 %16, 1
  %19 = icmp samesign ult i64 %18, %6
  br i1 %19, label %20, label %8

20:                                               ; preds = %15
  %21 = getelementptr inbounds nuw %struct.planet, ptr %1, i64 %16
  %22 = load <2 x double>, ptr %21, align 8, !tbaa !8
  %23 = getelementptr inbounds nuw i8, ptr %21, i64 16
  %24 = load double, ptr %23, align 8, !tbaa !12
  %25 = getelementptr inbounds nuw i8, ptr %21, i64 24
  %26 = getelementptr inbounds nuw i8, ptr %21, i64 40
  %27 = getelementptr inbounds nuw i8, ptr %21, i64 48
  %28 = load double, ptr %27, align 8, !tbaa !14
  %29 = insertelement <2 x double> poison, double %28, i64 0
  %30 = shufflevector <2 x double> %29, <2 x double> poison, <2 x i32> zeroinitializer
  br label %31

31:                                               ; preds = %20, %31
  %32 = phi i64 [ %17, %20 ], [ %69, %31 ]
  %33 = getelementptr inbounds nuw %struct.planet, ptr %1, i64 %32
  %34 = getelementptr inbounds nuw i8, ptr %33, i64 16
  %35 = load double, ptr %34, align 8, !tbaa !12
  %36 = fsub double %24, %35
  %37 = getelementptr inbounds nuw i8, ptr %33, i64 48
  %38 = load double, ptr %37, align 8, !tbaa !14
  %39 = fneg double %38
  %40 = load double, ptr %26, align 8, !tbaa !15
  %41 = fmul double %36, %39
  %42 = getelementptr inbounds nuw i8, ptr %33, i64 24
  %43 = load <2 x double>, ptr %33, align 8, !tbaa !8
  %44 = fsub <2 x double> %22, %43
  %45 = fmul <2 x double> %44, %44
  %46 = extractelement <2 x double> %45, i64 1
  %47 = extractelement <2 x double> %44, i64 0
  %48 = tail call double @llvm.fmuladd.f64(double %47, double %47, double %46)
  %49 = tail call double @llvm.fmuladd.f64(double %36, double %36, double %48)
  %50 = tail call double @llvm.sqrt.f64(double %49)
  %51 = fmul double %50, %50
  %52 = fmul double %50, %51
  %53 = fdiv double %2, %52
  %54 = load <2 x double>, ptr %25, align 8, !tbaa !8
  %55 = insertelement <2 x double> poison, double %39, i64 0
  %56 = shufflevector <2 x double> %55, <2 x double> poison, <2 x i32> zeroinitializer
  %57 = fmul <2 x double> %44, %56
  %58 = insertelement <2 x double> poison, double %53, i64 0
  %59 = shufflevector <2 x double> %58, <2 x double> poison, <2 x i32> zeroinitializer
  %60 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %57, <2 x double> %59, <2 x double> %54)
  store <2 x double> %60, ptr %25, align 8, !tbaa !8
  %61 = tail call double @llvm.fmuladd.f64(double %41, double %53, double %40)
  store double %61, ptr %26, align 8, !tbaa !15
  %62 = fmul <2 x double> %44, %30
  %63 = load <2 x double>, ptr %42, align 8, !tbaa !8
  %64 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %62, <2 x double> %59, <2 x double> %63)
  store <2 x double> %64, ptr %42, align 8, !tbaa !8
  %65 = fmul double %36, %28
  %66 = getelementptr inbounds nuw i8, ptr %33, i64 40
  %67 = load double, ptr %66, align 8, !tbaa !15
  %68 = tail call double @llvm.fmuladd.f64(double %65, double %53, double %67)
  store double %68, ptr %66, align 8, !tbaa !15
  %69 = add nuw nsw i64 %32, 1
  %70 = icmp eq i64 %69, %7
  br i1 %70, label %8, label %31, !llvm.loop !16

71:                                               ; preds = %11, %71
  %72 = phi i64 [ 0, %11 ], [ %83, %71 ]
  %73 = getelementptr inbounds nuw %struct.planet, ptr %1, i64 %72
  %74 = getelementptr inbounds nuw i8, ptr %73, i64 24
  %75 = load <2 x double>, ptr %74, align 8, !tbaa !8
  %76 = load <2 x double>, ptr %73, align 8, !tbaa !8
  %77 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %14, <2 x double> %75, <2 x double> %76)
  store <2 x double> %77, ptr %73, align 8, !tbaa !8
  %78 = getelementptr inbounds nuw i8, ptr %73, i64 40
  %79 = load double, ptr %78, align 8, !tbaa !15
  %80 = getelementptr inbounds nuw i8, ptr %73, i64 16
  %81 = load double, ptr %80, align 8, !tbaa !12
  %82 = tail call double @llvm.fmuladd.f64(double %2, double %79, double %81)
  store double %82, ptr %80, align 8, !tbaa !12
  %83 = add nuw nsw i64 %72, 1
  %84 = icmp eq i64 %83, %12
  br i1 %84, label %85, label %71, !llvm.loop !17

85:                                               ; preds = %71, %3
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fmuladd.f64(double, double, double) #1

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: read) uwtable
define dso_local double @energy(i32 noundef %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #2 {
  %3 = icmp sgt i32 %0, 0
  br i1 %3, label %4, label %60

4:                                                ; preds = %2
  %5 = zext nneg i32 %0 to i64
  %6 = zext nneg i32 %0 to i64
  br label %11

7:                                                ; preds = %37, %11
  %8 = phi double [ %28, %11 ], [ %57, %37 ]
  %9 = add nuw nsw i64 %13, 1
  %10 = icmp eq i64 %29, %6
  br i1 %10, label %60, label %11, !llvm.loop !18

11:                                               ; preds = %4, %7
  %12 = phi i64 [ 0, %4 ], [ %29, %7 ]
  %13 = phi i64 [ 1, %4 ], [ %9, %7 ]
  %14 = phi double [ 0.000000e+00, %4 ], [ %8, %7 ]
  %15 = getelementptr inbounds nuw %struct.planet, ptr %1, i64 %12
  %16 = getelementptr inbounds nuw i8, ptr %15, i64 48
  %17 = load double, ptr %16, align 8, !tbaa !14
  %18 = fmul double %17, 5.000000e-01
  %19 = getelementptr inbounds nuw i8, ptr %15, i64 24
  %20 = load double, ptr %19, align 8, !tbaa !19
  %21 = getelementptr inbounds nuw i8, ptr %15, i64 32
  %22 = load double, ptr %21, align 8, !tbaa !20
  %23 = fmul double %22, %22
  %24 = tail call double @llvm.fmuladd.f64(double %20, double %20, double %23)
  %25 = getelementptr inbounds nuw i8, ptr %15, i64 40
  %26 = load double, ptr %25, align 8, !tbaa !15
  %27 = tail call double @llvm.fmuladd.f64(double %26, double %26, double %24)
  %28 = tail call double @llvm.fmuladd.f64(double %18, double %27, double %14)
  %29 = add nuw nsw i64 %12, 1
  %30 = icmp samesign ult i64 %29, %5
  br i1 %30, label %31, label %7

31:                                               ; preds = %11
  %32 = load double, ptr %15, align 8, !tbaa !21
  %33 = getelementptr inbounds nuw i8, ptr %15, i64 8
  %34 = load double, ptr %33, align 8, !tbaa !22
  %35 = getelementptr inbounds nuw i8, ptr %15, i64 16
  %36 = load double, ptr %35, align 8, !tbaa !12
  br label %37

37:                                               ; preds = %31, %37
  %38 = phi i64 [ %13, %31 ], [ %58, %37 ]
  %39 = phi double [ %28, %31 ], [ %57, %37 ]
  %40 = getelementptr inbounds nuw %struct.planet, ptr %1, i64 %38
  %41 = load double, ptr %40, align 8, !tbaa !21
  %42 = fsub double %32, %41
  %43 = getelementptr inbounds nuw i8, ptr %40, i64 8
  %44 = load double, ptr %43, align 8, !tbaa !22
  %45 = fsub double %34, %44
  %46 = getelementptr inbounds nuw i8, ptr %40, i64 16
  %47 = load double, ptr %46, align 8, !tbaa !12
  %48 = fsub double %36, %47
  %49 = fmul double %45, %45
  %50 = tail call double @llvm.fmuladd.f64(double %42, double %42, double %49)
  %51 = tail call double @llvm.fmuladd.f64(double %48, double %48, double %50)
  %52 = tail call double @llvm.sqrt.f64(double %51)
  %53 = getelementptr inbounds nuw i8, ptr %40, i64 48
  %54 = load double, ptr %53, align 8, !tbaa !14
  %55 = fmul double %17, %54
  %56 = fdiv double %55, %52
  %57 = fsub double %39, %56
  %58 = add nuw nsw i64 %38, 1
  %59 = icmp eq i64 %58, %6
  br i1 %59, label %7, label %37, !llvm.loop !23

60:                                               ; preds = %7, %2
  %61 = phi double [ 0.000000e+00, %2 ], [ %8, %7 ]
  ret double %61
}

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable
define dso_local void @offset_momentum(i32 noundef %0, ptr noundef captures(none) %1) local_unnamed_addr #0 {
  %3 = icmp sgt i32 %0, 0
  br i1 %3, label %4, label %64

4:                                                ; preds = %2
  %5 = zext nneg i32 %0 to i64
  %6 = icmp eq i32 %0, 1
  br i1 %6, label %43, label %7

7:                                                ; preds = %4
  %8 = and i64 %5, 2147483646
  br label %9

9:                                                ; preds = %9, %7
  %10 = phi i64 [ 0, %7 ], [ %39, %9 ]
  %11 = phi double [ 0.000000e+00, %7 ], [ %28, %9 ]
  %12 = phi <2 x double> [ zeroinitializer, %7 ], [ %38, %9 ]
  %13 = getelementptr inbounds nuw %struct.planet, ptr %1, i64 %10
  %14 = getelementptr inbounds nuw %struct.planet, ptr %1, i64 %10
  %15 = getelementptr inbounds nuw i8, ptr %13, i64 24
  %16 = getelementptr inbounds nuw i8, ptr %14, i64 80
  %17 = getelementptr inbounds nuw i8, ptr %13, i64 48
  %18 = getelementptr inbounds nuw i8, ptr %14, i64 104
  %19 = load double, ptr %17, align 8, !tbaa !14
  %20 = load double, ptr %18, align 8, !tbaa !14
  %21 = getelementptr inbounds nuw i8, ptr %13, i64 40
  %22 = getelementptr inbounds nuw i8, ptr %14, i64 96
  %23 = load double, ptr %21, align 8, !tbaa !15
  %24 = load double, ptr %22, align 8, !tbaa !15
  %25 = fmul double %23, %19
  %26 = fmul double %24, %20
  %27 = fadd double %11, %25
  %28 = fadd double %27, %26
  %29 = load <2 x double>, ptr %15, align 8, !tbaa !8
  %30 = load <2 x double>, ptr %16, align 8, !tbaa !8
  %31 = insertelement <2 x double> poison, double %19, i64 0
  %32 = shufflevector <2 x double> %31, <2 x double> poison, <2 x i32> zeroinitializer
  %33 = fmul <2 x double> %29, %32
  %34 = insertelement <2 x double> poison, double %20, i64 0
  %35 = shufflevector <2 x double> %34, <2 x double> poison, <2 x i32> zeroinitializer
  %36 = fmul <2 x double> %30, %35
  %37 = fadd <2 x double> %12, %33
  %38 = fadd <2 x double> %37, %36
  %39 = add nuw i64 %10, 2
  %40 = icmp eq i64 %39, %8
  br i1 %40, label %41, label %9, !llvm.loop !24

41:                                               ; preds = %9
  %42 = icmp eq i64 %8, %5
  br i1 %42, label %64, label %43

43:                                               ; preds = %4, %41
  %44 = phi i64 [ 0, %4 ], [ %8, %41 ]
  %45 = phi double [ 0.000000e+00, %4 ], [ %28, %41 ]
  %46 = phi <2 x double> [ zeroinitializer, %4 ], [ %38, %41 ]
  br label %47

47:                                               ; preds = %43, %47
  %48 = phi i64 [ %62, %47 ], [ %44, %43 ]
  %49 = phi double [ %61, %47 ], [ %45, %43 ]
  %50 = phi <2 x double> [ %58, %47 ], [ %46, %43 ]
  %51 = getelementptr inbounds nuw %struct.planet, ptr %1, i64 %48
  %52 = getelementptr inbounds nuw i8, ptr %51, i64 24
  %53 = getelementptr inbounds nuw i8, ptr %51, i64 48
  %54 = load double, ptr %53, align 8, !tbaa !14
  %55 = load <2 x double>, ptr %52, align 8, !tbaa !8
  %56 = insertelement <2 x double> poison, double %54, i64 0
  %57 = shufflevector <2 x double> %56, <2 x double> poison, <2 x i32> zeroinitializer
  %58 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %55, <2 x double> %57, <2 x double> %50)
  %59 = getelementptr inbounds nuw i8, ptr %51, i64 40
  %60 = load double, ptr %59, align 8, !tbaa !15
  %61 = tail call double @llvm.fmuladd.f64(double %60, double %54, double %49)
  %62 = add nuw nsw i64 %48, 1
  %63 = icmp eq i64 %62, %5
  br i1 %63, label %64, label %47, !llvm.loop !27

64:                                               ; preds = %47, %41, %2
  %65 = phi double [ 0.000000e+00, %2 ], [ %28, %41 ], [ %61, %47 ]
  %66 = phi <2 x double> [ zeroinitializer, %2 ], [ %38, %41 ], [ %58, %47 ]
  %67 = fdiv <2 x double> %66, splat (double 0xC043BD3CC9BE45DE)
  %68 = getelementptr inbounds nuw i8, ptr %1, i64 24
  store <2 x double> %67, ptr %68, align 8, !tbaa !8
  %69 = fdiv double %65, 0xC043BD3CC9BE45DE
  %70 = getelementptr inbounds nuw i8, ptr %1, i64 40
  store double %69, ptr %70, align 8, !tbaa !15
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readnone captures(none) %1) local_unnamed_addr #3 {
  %3 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 48), align 8, !tbaa !14
  %4 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 40), align 8, !tbaa !15
  %5 = tail call double @llvm.fmuladd.f64(double %4, double %3, double 0.000000e+00)
  %6 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 104), align 8, !tbaa !14
  %7 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 96), align 8, !tbaa !15
  %8 = tail call double @llvm.fmuladd.f64(double %7, double %6, double %5)
  %9 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 160), align 8, !tbaa !14
  %10 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 152), align 8, !tbaa !15
  %11 = tail call double @llvm.fmuladd.f64(double %10, double %9, double %8)
  %12 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 216), align 8, !tbaa !14
  %13 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 208), align 8, !tbaa !15
  %14 = tail call double @llvm.fmuladd.f64(double %13, double %12, double %11)
  %15 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 272), align 8, !tbaa !14
  %16 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 264), align 8, !tbaa !15
  %17 = tail call double @llvm.fmuladd.f64(double %16, double %15, double %14)
  %18 = load <2 x double>, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 24), align 8, !tbaa !8
  %19 = insertelement <2 x double> poison, double %3, i64 0
  %20 = shufflevector <2 x double> %19, <2 x double> poison, <2 x i32> zeroinitializer
  %21 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %18, <2 x double> %20, <2 x double> zeroinitializer)
  %22 = load <2 x double>, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 80), align 8, !tbaa !8
  %23 = insertelement <2 x double> poison, double %6, i64 0
  %24 = shufflevector <2 x double> %23, <2 x double> poison, <2 x i32> zeroinitializer
  %25 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %22, <2 x double> %24, <2 x double> %21)
  %26 = load <2 x double>, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 136), align 8, !tbaa !8
  %27 = insertelement <2 x double> poison, double %9, i64 0
  %28 = shufflevector <2 x double> %27, <2 x double> poison, <2 x i32> zeroinitializer
  %29 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %26, <2 x double> %28, <2 x double> %25)
  %30 = load <2 x double>, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 192), align 8, !tbaa !8
  %31 = insertelement <2 x double> poison, double %12, i64 0
  %32 = shufflevector <2 x double> %31, <2 x double> poison, <2 x i32> zeroinitializer
  %33 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %30, <2 x double> %32, <2 x double> %29)
  %34 = load <2 x double>, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 248), align 8, !tbaa !8
  %35 = insertelement <2 x double> poison, double %15, i64 0
  %36 = shufflevector <2 x double> %35, <2 x double> poison, <2 x i32> zeroinitializer
  %37 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %34, <2 x double> %36, <2 x double> %33)
  %38 = fdiv <2 x double> %37, splat (double 0xC043BD3CC9BE45DE)
  store <2 x double> %38, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 24), align 8, !tbaa !8
  %39 = fdiv double %17, 0xC043BD3CC9BE45DE
  store double %39, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 40), align 8, !tbaa !15
  %40 = fmul double %15, 5.000000e-01
  %41 = fmul <2 x double> %34, %34
  %42 = extractelement <2 x double> %41, i64 1
  %43 = extractelement <2 x double> %34, i64 0
  %44 = tail call double @llvm.fmuladd.f64(double %43, double %43, double %42)
  %45 = tail call double @llvm.fmuladd.f64(double %16, double %16, double %44)
  %46 = fmul double %12, 5.000000e-01
  %47 = fmul <2 x double> %30, %30
  %48 = extractelement <2 x double> %47, i64 1
  %49 = extractelement <2 x double> %30, i64 0
  %50 = tail call double @llvm.fmuladd.f64(double %49, double %49, double %48)
  %51 = tail call double @llvm.fmuladd.f64(double %13, double %13, double %50)
  %52 = fmul double %9, 5.000000e-01
  %53 = fmul <2 x double> %26, %26
  %54 = extractelement <2 x double> %53, i64 1
  %55 = extractelement <2 x double> %26, i64 0
  %56 = tail call double @llvm.fmuladd.f64(double %55, double %55, double %54)
  %57 = tail call double @llvm.fmuladd.f64(double %10, double %10, double %56)
  %58 = fmul double %6, 5.000000e-01
  %59 = fmul <2 x double> %22, %22
  %60 = extractelement <2 x double> %59, i64 1
  %61 = extractelement <2 x double> %22, i64 0
  %62 = tail call double @llvm.fmuladd.f64(double %61, double %61, double %60)
  %63 = tail call double @llvm.fmuladd.f64(double %7, double %7, double %62)
  %64 = fmul double %3, 5.000000e-01
  %65 = fmul <2 x double> %38, %38
  %66 = extractelement <2 x double> %65, i64 1
  %67 = extractelement <2 x double> %38, i64 0
  %68 = tail call double @llvm.fmuladd.f64(double %67, double %67, double %66)
  %69 = tail call double @llvm.fmuladd.f64(double %39, double %39, double %68)
  %70 = tail call double @llvm.fmuladd.f64(double %64, double %69, double 0.000000e+00)
  %71 = fmul double %3, %6
  %72 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 16), align 8, !tbaa !12
  %73 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 72), align 8, !tbaa !12
  %74 = fsub double %72, %73
  %75 = load double, ptr @bodies, align 8, !tbaa !21
  %76 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 56), align 8, !tbaa !21
  %77 = fsub double %75, %76
  %78 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 8), align 8, !tbaa !22
  %79 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 64), align 8, !tbaa !22
  %80 = fsub double %78, %79
  %81 = fmul double %80, %80
  %82 = tail call double @llvm.fmuladd.f64(double %77, double %77, double %81)
  %83 = tail call double @llvm.fmuladd.f64(double %74, double %74, double %82)
  %84 = tail call double @llvm.sqrt.f64(double %83)
  %85 = fdiv double %71, %84
  %86 = fsub double %70, %85
  %87 = fmul double %3, %9
  %88 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 128), align 8, !tbaa !12
  %89 = fsub double %72, %88
  %90 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 112), align 8, !tbaa !21
  %91 = fsub double %75, %90
  %92 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 120), align 8, !tbaa !22
  %93 = fsub double %78, %92
  %94 = fmul double %93, %93
  %95 = tail call double @llvm.fmuladd.f64(double %91, double %91, double %94)
  %96 = tail call double @llvm.fmuladd.f64(double %89, double %89, double %95)
  %97 = tail call double @llvm.sqrt.f64(double %96)
  %98 = fdiv double %87, %97
  %99 = fsub double %86, %98
  %100 = load <2 x double>, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 184), align 8
  %101 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 240), align 8, !tbaa !12
  %102 = load <2 x double>, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 168), align 8, !tbaa !8
  %103 = insertelement <2 x double> %31, double %15, i64 1
  %104 = fmul <2 x double> %20, %103
  %105 = insertelement <2 x double> poison, double %72, i64 0
  %106 = shufflevector <2 x double> %105, <2 x double> poison, <2 x i32> zeroinitializer
  %107 = insertelement <2 x double> %100, double %101, i64 1
  %108 = fsub <2 x double> %106, %107
  %109 = load <2 x double>, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 224), align 8, !tbaa !8
  %110 = insertelement <2 x double> poison, double %75, i64 0
  %111 = shufflevector <2 x double> %110, <2 x double> poison, <2 x i32> zeroinitializer
  %112 = shufflevector <2 x double> %102, <2 x double> %109, <2 x i32> <i32 0, i32 2>
  %113 = fsub <2 x double> %111, %112
  %114 = insertelement <2 x double> poison, double %78, i64 0
  %115 = shufflevector <2 x double> %114, <2 x double> poison, <2 x i32> zeroinitializer
  %116 = shufflevector <2 x double> %102, <2 x double> %109, <2 x i32> <i32 1, i32 3>
  %117 = fsub <2 x double> %115, %116
  %118 = fmul <2 x double> %117, %117
  %119 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %113, <2 x double> %113, <2 x double> %118)
  %120 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %108, <2 x double> %108, <2 x double> %119)
  %121 = tail call <2 x double> @llvm.sqrt.v2f64(<2 x double> %120)
  %122 = fdiv <2 x double> %104, %121
  %123 = extractelement <2 x double> %122, i64 0
  %124 = fsub double %99, %123
  %125 = extractelement <2 x double> %122, i64 1
  %126 = fsub double %124, %125
  %127 = tail call double @llvm.fmuladd.f64(double %58, double %63, double %126)
  %128 = fmul double %6, %9
  %129 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 72), align 8, !tbaa !12
  %130 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 128), align 8, !tbaa !12
  %131 = fsub double %129, %130
  %132 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 56), align 8, !tbaa !21
  %133 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 112), align 8, !tbaa !21
  %134 = fsub double %132, %133
  %135 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 64), align 8, !tbaa !22
  %136 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 120), align 8, !tbaa !22
  %137 = fsub double %135, %136
  %138 = fmul double %137, %137
  %139 = tail call double @llvm.fmuladd.f64(double %134, double %134, double %138)
  %140 = tail call double @llvm.fmuladd.f64(double %131, double %131, double %139)
  %141 = tail call double @llvm.sqrt.f64(double %140)
  %142 = fdiv double %128, %141
  %143 = fsub double %127, %142
  %144 = load <2 x double>, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 184), align 8
  %145 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 240), align 8, !tbaa !12
  %146 = load <2 x double>, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 168), align 8, !tbaa !8
  %147 = fmul <2 x double> %24, %103
  %148 = insertelement <2 x double> poison, double %129, i64 0
  %149 = shufflevector <2 x double> %148, <2 x double> poison, <2 x i32> zeroinitializer
  %150 = insertelement <2 x double> %144, double %145, i64 1
  %151 = fsub <2 x double> %149, %150
  %152 = load <2 x double>, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 224), align 8, !tbaa !8
  %153 = insertelement <2 x double> poison, double %132, i64 0
  %154 = shufflevector <2 x double> %153, <2 x double> poison, <2 x i32> zeroinitializer
  %155 = shufflevector <2 x double> %146, <2 x double> %152, <2 x i32> <i32 0, i32 2>
  %156 = fsub <2 x double> %154, %155
  %157 = insertelement <2 x double> poison, double %135, i64 0
  %158 = shufflevector <2 x double> %157, <2 x double> poison, <2 x i32> zeroinitializer
  %159 = shufflevector <2 x double> %146, <2 x double> %152, <2 x i32> <i32 1, i32 3>
  %160 = fsub <2 x double> %158, %159
  %161 = fmul <2 x double> %160, %160
  %162 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %156, <2 x double> %156, <2 x double> %161)
  %163 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %151, <2 x double> %151, <2 x double> %162)
  %164 = tail call <2 x double> @llvm.sqrt.v2f64(<2 x double> %163)
  %165 = fdiv <2 x double> %147, %164
  %166 = extractelement <2 x double> %165, i64 0
  %167 = fsub double %143, %166
  %168 = extractelement <2 x double> %165, i64 1
  %169 = fsub double %167, %168
  %170 = tail call double @llvm.fmuladd.f64(double %52, double %57, double %169)
  %171 = load <1 x double>, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 128), align 8
  %172 = load <2 x double>, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 184), align 8
  %173 = load <1 x double>, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 112), align 8
  %174 = load <1 x double>, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 120), align 8
  %175 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 240), align 8, !tbaa !12
  %176 = load <2 x double>, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 168), align 8, !tbaa !8
  %177 = fmul <2 x double> %28, %103
  %178 = shufflevector <1 x double> %171, <1 x double> poison, <2 x i32> zeroinitializer
  %179 = insertelement <2 x double> %172, double %175, i64 1
  %180 = fsub <2 x double> %178, %179
  %181 = load <2 x double>, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 224), align 8, !tbaa !8
  %182 = shufflevector <1 x double> %173, <1 x double> poison, <2 x i32> zeroinitializer
  %183 = shufflevector <2 x double> %176, <2 x double> %181, <2 x i32> <i32 0, i32 2>
  %184 = fsub <2 x double> %182, %183
  %185 = shufflevector <1 x double> %174, <1 x double> poison, <2 x i32> zeroinitializer
  %186 = shufflevector <2 x double> %176, <2 x double> %181, <2 x i32> <i32 1, i32 3>
  %187 = fsub <2 x double> %185, %186
  %188 = fmul <2 x double> %187, %187
  %189 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %184, <2 x double> %184, <2 x double> %188)
  %190 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %180, <2 x double> %180, <2 x double> %189)
  %191 = tail call <2 x double> @llvm.sqrt.v2f64(<2 x double> %190)
  %192 = fdiv <2 x double> %177, %191
  %193 = extractelement <2 x double> %192, i64 0
  %194 = fsub double %170, %193
  %195 = extractelement <2 x double> %192, i64 1
  %196 = fsub double %194, %195
  %197 = tail call double @llvm.fmuladd.f64(double %46, double %51, double %196)
  %198 = fmul double %12, %15
  %199 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 184), align 8, !tbaa !12
  %200 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 240), align 8, !tbaa !12
  %201 = fsub double %199, %200
  %202 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 168), align 8, !tbaa !21
  %203 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 224), align 8, !tbaa !21
  %204 = fsub double %202, %203
  %205 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 176), align 8, !tbaa !22
  %206 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 232), align 8, !tbaa !22
  %207 = fsub double %205, %206
  %208 = fmul double %207, %207
  %209 = tail call double @llvm.fmuladd.f64(double %204, double %204, double %208)
  %210 = tail call double @llvm.fmuladd.f64(double %201, double %201, double %209)
  %211 = tail call double @llvm.sqrt.f64(double %210)
  %212 = fdiv double %198, %211
  %213 = fsub double %197, %212
  %214 = tail call double @llvm.fmuladd.f64(double %40, double %45, double %213)
  %215 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, double noundef %214)
  br label %216

216:                                              ; preds = %2, %216
  %217 = phi i32 [ 1, %2 ], [ %218, %216 ]
  tail call void @advance(i32 noundef 5, ptr noundef nonnull @bodies, double noundef 1.000000e-02)
  %218 = add nuw nsw i32 %217, 1
  %219 = icmp eq i32 %218, 5000001
  br i1 %219, label %220, label %216, !llvm.loop !28

220:                                              ; preds = %216
  %221 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 104), align 8, !tbaa !14
  %222 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 160), align 8, !tbaa !14
  %223 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 216), align 8, !tbaa !14
  %224 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 272), align 8, !tbaa !14
  %225 = fmul double %224, 5.000000e-01
  %226 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 264), align 8, !tbaa !15
  %227 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 248), align 8, !tbaa !19
  %228 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 256), align 8, !tbaa !20
  %229 = fmul double %228, %228
  %230 = tail call double @llvm.fmuladd.f64(double %227, double %227, double %229)
  %231 = tail call double @llvm.fmuladd.f64(double %226, double %226, double %230)
  %232 = fmul double %223, 5.000000e-01
  %233 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 208), align 8, !tbaa !15
  %234 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 192), align 8, !tbaa !19
  %235 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 200), align 8, !tbaa !20
  %236 = fmul double %235, %235
  %237 = tail call double @llvm.fmuladd.f64(double %234, double %234, double %236)
  %238 = tail call double @llvm.fmuladd.f64(double %233, double %233, double %237)
  %239 = fmul double %222, 5.000000e-01
  %240 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 152), align 8, !tbaa !15
  %241 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 136), align 8, !tbaa !19
  %242 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 144), align 8, !tbaa !20
  %243 = fmul double %242, %242
  %244 = tail call double @llvm.fmuladd.f64(double %241, double %241, double %243)
  %245 = tail call double @llvm.fmuladd.f64(double %240, double %240, double %244)
  %246 = fmul double %221, 5.000000e-01
  %247 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 96), align 8, !tbaa !15
  %248 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 80), align 8, !tbaa !19
  %249 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 88), align 8, !tbaa !20
  %250 = fmul double %249, %249
  %251 = tail call double @llvm.fmuladd.f64(double %248, double %248, double %250)
  %252 = tail call double @llvm.fmuladd.f64(double %247, double %247, double %251)
  %253 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 48), align 8, !tbaa !14
  %254 = fmul double %253, 5.000000e-01
  %255 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 40), align 8, !tbaa !15
  %256 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 24), align 8, !tbaa !19
  %257 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 32), align 8, !tbaa !20
  %258 = fmul double %257, %257
  %259 = tail call double @llvm.fmuladd.f64(double %256, double %256, double %258)
  %260 = tail call double @llvm.fmuladd.f64(double %255, double %255, double %259)
  %261 = tail call double @llvm.fmuladd.f64(double %254, double %260, double 0.000000e+00)
  %262 = fmul double %253, %221
  %263 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 16), align 8, !tbaa !12
  %264 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 72), align 8, !tbaa !12
  %265 = fsub double %263, %264
  %266 = load double, ptr @bodies, align 8, !tbaa !21
  %267 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 56), align 8, !tbaa !21
  %268 = fsub double %266, %267
  %269 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 8), align 8, !tbaa !22
  %270 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 64), align 8, !tbaa !22
  %271 = fsub double %269, %270
  %272 = fmul double %271, %271
  %273 = tail call double @llvm.fmuladd.f64(double %268, double %268, double %272)
  %274 = tail call double @llvm.fmuladd.f64(double %265, double %265, double %273)
  %275 = tail call double @llvm.sqrt.f64(double %274)
  %276 = fdiv double %262, %275
  %277 = fsub double %261, %276
  %278 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 160), align 8, !tbaa !14
  %279 = fmul double %253, %278
  %280 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 128), align 8, !tbaa !12
  %281 = fsub double %263, %280
  %282 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 112), align 8, !tbaa !21
  %283 = fsub double %266, %282
  %284 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 120), align 8, !tbaa !22
  %285 = fsub double %269, %284
  %286 = fmul double %285, %285
  %287 = tail call double @llvm.fmuladd.f64(double %283, double %283, double %286)
  %288 = tail call double @llvm.fmuladd.f64(double %281, double %281, double %287)
  %289 = tail call double @llvm.sqrt.f64(double %288)
  %290 = fdiv double %279, %289
  %291 = fsub double %277, %290
  %292 = load <2 x double>, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 216), align 8
  %293 = load <2 x double>, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 184), align 8
  %294 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 272), align 8, !tbaa !14
  %295 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 240), align 8, !tbaa !12
  %296 = load <2 x double>, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 168), align 8, !tbaa !8
  %297 = insertelement <2 x double> %292, double %253, i64 1
  %298 = insertelement <2 x double> poison, double %253, i64 0
  %299 = insertelement <2 x double> %298, double %294, i64 1
  %300 = fmul <2 x double> %297, %299
  %301 = insertelement <2 x double> poison, double %263, i64 0
  %302 = shufflevector <2 x double> %301, <2 x double> poison, <2 x i32> zeroinitializer
  %303 = insertelement <2 x double> %293, double %295, i64 1
  %304 = fsub <2 x double> %302, %303
  %305 = load <2 x double>, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 224), align 8, !tbaa !8
  %306 = insertelement <2 x double> poison, double %266, i64 0
  %307 = shufflevector <2 x double> %306, <2 x double> poison, <2 x i32> zeroinitializer
  %308 = shufflevector <2 x double> %296, <2 x double> %305, <2 x i32> <i32 0, i32 2>
  %309 = fsub <2 x double> %307, %308
  %310 = insertelement <2 x double> poison, double %269, i64 0
  %311 = shufflevector <2 x double> %310, <2 x double> poison, <2 x i32> zeroinitializer
  %312 = shufflevector <2 x double> %296, <2 x double> %305, <2 x i32> <i32 1, i32 3>
  %313 = fsub <2 x double> %311, %312
  %314 = fmul <2 x double> %313, %313
  %315 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %309, <2 x double> %309, <2 x double> %314)
  %316 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %304, <2 x double> %304, <2 x double> %315)
  %317 = tail call <2 x double> @llvm.sqrt.v2f64(<2 x double> %316)
  %318 = fdiv <2 x double> %300, %317
  %319 = extractelement <2 x double> %318, i64 0
  %320 = fsub double %291, %319
  %321 = extractelement <2 x double> %318, i64 1
  %322 = fsub double %320, %321
  %323 = tail call double @llvm.fmuladd.f64(double %246, double %252, double %322)
  %324 = fmul double %221, %222
  %325 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 72), align 8, !tbaa !12
  %326 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 128), align 8, !tbaa !12
  %327 = fsub double %325, %326
  %328 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 56), align 8, !tbaa !21
  %329 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 112), align 8, !tbaa !21
  %330 = fsub double %328, %329
  %331 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 64), align 8, !tbaa !22
  %332 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 120), align 8, !tbaa !22
  %333 = fsub double %331, %332
  %334 = fmul double %333, %333
  %335 = tail call double @llvm.fmuladd.f64(double %330, double %330, double %334)
  %336 = tail call double @llvm.fmuladd.f64(double %327, double %327, double %335)
  %337 = tail call double @llvm.sqrt.f64(double %336)
  %338 = fdiv double %324, %337
  %339 = fsub double %323, %338
  %340 = load <2 x double>, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 216), align 8
  %341 = load <2 x double>, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 184), align 8
  %342 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 272), align 8, !tbaa !14
  %343 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 240), align 8, !tbaa !12
  %344 = load <2 x double>, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 168), align 8, !tbaa !8
  %345 = insertelement <2 x double> %340, double %221, i64 1
  %346 = insertelement <2 x double> poison, double %221, i64 0
  %347 = insertelement <2 x double> %346, double %342, i64 1
  %348 = fmul <2 x double> %345, %347
  %349 = insertelement <2 x double> poison, double %325, i64 0
  %350 = shufflevector <2 x double> %349, <2 x double> poison, <2 x i32> zeroinitializer
  %351 = insertelement <2 x double> %341, double %343, i64 1
  %352 = fsub <2 x double> %350, %351
  %353 = load <2 x double>, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 224), align 8, !tbaa !8
  %354 = insertelement <2 x double> poison, double %328, i64 0
  %355 = shufflevector <2 x double> %354, <2 x double> poison, <2 x i32> zeroinitializer
  %356 = shufflevector <2 x double> %344, <2 x double> %353, <2 x i32> <i32 0, i32 2>
  %357 = fsub <2 x double> %355, %356
  %358 = insertelement <2 x double> poison, double %331, i64 0
  %359 = shufflevector <2 x double> %358, <2 x double> poison, <2 x i32> zeroinitializer
  %360 = shufflevector <2 x double> %344, <2 x double> %353, <2 x i32> <i32 1, i32 3>
  %361 = fsub <2 x double> %359, %360
  %362 = fmul <2 x double> %361, %361
  %363 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %357, <2 x double> %357, <2 x double> %362)
  %364 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %352, <2 x double> %352, <2 x double> %363)
  %365 = tail call <2 x double> @llvm.sqrt.v2f64(<2 x double> %364)
  %366 = fdiv <2 x double> %348, %365
  %367 = extractelement <2 x double> %366, i64 0
  %368 = fsub double %339, %367
  %369 = extractelement <2 x double> %366, i64 1
  %370 = fsub double %368, %369
  %371 = tail call double @llvm.fmuladd.f64(double %239, double %245, double %370)
  %372 = load <1 x double>, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 128), align 8
  %373 = load <2 x double>, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 184), align 8
  %374 = load <1 x double>, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 112), align 8
  %375 = load <1 x double>, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 120), align 8
  %376 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 272), align 8, !tbaa !14
  %377 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 240), align 8, !tbaa !12
  %378 = load <2 x double>, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 168), align 8, !tbaa !8
  %379 = insertelement <2 x double> poison, double %222, i64 0
  %380 = shufflevector <2 x double> %379, <2 x double> poison, <2 x i32> zeroinitializer
  %381 = insertelement <2 x double> poison, double %223, i64 0
  %382 = insertelement <2 x double> %381, double %376, i64 1
  %383 = fmul <2 x double> %380, %382
  %384 = shufflevector <1 x double> %372, <1 x double> poison, <2 x i32> zeroinitializer
  %385 = insertelement <2 x double> %373, double %377, i64 1
  %386 = fsub <2 x double> %384, %385
  %387 = load <2 x double>, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 224), align 8, !tbaa !8
  %388 = shufflevector <1 x double> %374, <1 x double> poison, <2 x i32> zeroinitializer
  %389 = shufflevector <2 x double> %378, <2 x double> %387, <2 x i32> <i32 0, i32 2>
  %390 = fsub <2 x double> %388, %389
  %391 = shufflevector <1 x double> %375, <1 x double> poison, <2 x i32> zeroinitializer
  %392 = shufflevector <2 x double> %378, <2 x double> %387, <2 x i32> <i32 1, i32 3>
  %393 = fsub <2 x double> %391, %392
  %394 = fmul <2 x double> %393, %393
  %395 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %390, <2 x double> %390, <2 x double> %394)
  %396 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %386, <2 x double> %386, <2 x double> %395)
  %397 = tail call <2 x double> @llvm.sqrt.v2f64(<2 x double> %396)
  %398 = fdiv <2 x double> %383, %397
  %399 = extractelement <2 x double> %398, i64 0
  %400 = fsub double %371, %399
  %401 = extractelement <2 x double> %398, i64 1
  %402 = fsub double %400, %401
  %403 = tail call double @llvm.fmuladd.f64(double %232, double %238, double %402)
  %404 = fmul double %223, %224
  %405 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 184), align 8, !tbaa !12
  %406 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 240), align 8, !tbaa !12
  %407 = fsub double %405, %406
  %408 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 168), align 8, !tbaa !21
  %409 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 224), align 8, !tbaa !21
  %410 = fsub double %408, %409
  %411 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 176), align 8, !tbaa !22
  %412 = load double, ptr getelementptr inbounds nuw (i8, ptr @bodies, i64 232), align 8, !tbaa !22
  %413 = fsub double %411, %412
  %414 = fmul double %413, %413
  %415 = tail call double @llvm.fmuladd.f64(double %410, double %410, double %414)
  %416 = tail call double @llvm.fmuladd.f64(double %407, double %407, double %415)
  %417 = tail call double @llvm.sqrt.f64(double %416)
  %418 = fdiv double %404, %417
  %419 = fsub double %403, %418
  %420 = tail call double @llvm.fmuladd.f64(double %225, double %231, double %419)
  %421 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, double noundef %420)
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #4

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.sqrt.f64(double) #5

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <2 x double> @llvm.fmuladd.v2f64(<2 x double>, <2 x double>, <2 x double>) #5

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <2 x double> @llvm.sqrt.v2f64(<2 x double>) #5

attributes #0 = { nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nofree norecurse nosync nounwind memory(argmem: read) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.mustprogress"}
!8 = !{!9, !9, i64 0}
!9 = !{!"double", !10, i64 0}
!10 = !{!"omnipotent char", !11, i64 0}
!11 = !{!"Simple C/C++ TBAA"}
!12 = !{!13, !9, i64 16}
!13 = !{!"planet", !9, i64 0, !9, i64 8, !9, i64 16, !9, i64 24, !9, i64 32, !9, i64 40, !9, i64 48}
!14 = !{!13, !9, i64 48}
!15 = !{!13, !9, i64 40}
!16 = distinct !{!16, !7}
!17 = distinct !{!17, !7}
!18 = distinct !{!18, !7}
!19 = !{!13, !9, i64 24}
!20 = !{!13, !9, i64 32}
!21 = !{!13, !9, i64 0}
!22 = !{!13, !9, i64 8}
!23 = distinct !{!23, !7}
!24 = distinct !{!24, !7, !25, !26}
!25 = !{!"llvm.loop.isvectorized", i32 1}
!26 = !{!"llvm.loop.unroll.runtime.disable"}
!27 = distinct !{!27, !7, !25}
!28 = distinct !{!28, !7}
