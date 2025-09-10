; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/CoyoteBench/lpbench.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/CoyoteBench/lpbench.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@.str = private unnamed_addr constant [4 x i8] c"-ga\00", align 1
@stdout = external local_unnamed_addr global ptr, align 8
@.str.1 = private unnamed_addr constant [3 x i8] c"%f\00", align 1
@.str.2 = private unnamed_addr constant [33 x i8] c"\0Alpbench (Std. C) run time: %f\0A\0A\00", align 1
@seed = internal unnamed_addr global i64 1325, align 8

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define dso_local void @matgen(ptr noundef readonly captures(none) %0, ptr noundef captures(none) %1) local_unnamed_addr #0 {
  %3 = load i64, ptr @seed, align 8, !tbaa !6
  %4 = xor i64 %3, 123459876
  br label %5

5:                                                ; preds = %2, %30
  %6 = phi i64 [ 0, %2 ], [ %31, %30 ]
  %7 = phi i64 [ %4, %2 ], [ %22, %30 ]
  br label %11

8:                                                ; preds = %30
  %9 = xor i64 %22, 123459876
  store i64 %9, ptr @seed, align 8, !tbaa !6
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16000) %1, i8 0, i64 16000, i1 false), !tbaa !10
  %10 = getelementptr i8, ptr %1, i64 16000
  br label %33

11:                                               ; preds = %5, %11
  %12 = phi i64 [ 0, %5 ], [ %28, %11 ]
  %13 = phi i64 [ %7, %5 ], [ %22, %11 ]
  %14 = sdiv i64 %13, 127773
  %15 = mul nsw i64 %14, -127773
  %16 = add i64 %15, %13
  %17 = mul nsw i64 %16, 16807
  %18 = mul nsw i64 %14, -2836
  %19 = add i64 %17, %18
  %20 = icmp slt i64 %19, 0
  %21 = add nsw i64 %19, 2147483647
  %22 = select i1 %20, i64 %21, i64 %19
  %23 = sitofp i64 %22 to double
  %24 = fmul double %23, 0x3E00000000200FE1
  %25 = getelementptr inbounds nuw ptr, ptr %0, i64 %12
  %26 = load ptr, ptr %25, align 8, !tbaa !12
  %27 = getelementptr inbounds nuw double, ptr %26, i64 %6
  store double %24, ptr %27, align 8, !tbaa !10
  %28 = add nuw nsw i64 %12, 1
  %29 = icmp eq i64 %28, 2000
  br i1 %29, label %30, label %11, !llvm.loop !15

30:                                               ; preds = %11
  %31 = add nuw nsw i64 %6, 1
  %32 = icmp eq i64 %31, 2000
  br i1 %32, label %8, label %5, !llvm.loop !17

33:                                               ; preds = %8, %64
  %34 = phi i64 [ 0, %8 ], [ %65, %64 ]
  %35 = getelementptr inbounds nuw ptr, ptr %0, i64 %34
  %36 = load ptr, ptr %35, align 8, !tbaa !12
  %37 = getelementptr i8, ptr %36, i64 16000
  %38 = icmp ult ptr %1, %37
  %39 = icmp ult ptr %36, %10
  %40 = and i1 %38, %39
  br i1 %40, label %55, label %41

41:                                               ; preds = %33, %41
  %42 = phi i64 [ %53, %41 ], [ 0, %33 ]
  %43 = getelementptr inbounds nuw double, ptr %36, i64 %42
  %44 = getelementptr inbounds nuw i8, ptr %43, i64 16
  %45 = load <2 x double>, ptr %43, align 8, !tbaa !10, !alias.scope !18
  %46 = load <2 x double>, ptr %44, align 8, !tbaa !10, !alias.scope !18
  %47 = getelementptr inbounds nuw double, ptr %1, i64 %42
  %48 = getelementptr inbounds nuw i8, ptr %47, i64 16
  %49 = load <2 x double>, ptr %47, align 8, !tbaa !10, !alias.scope !21, !noalias !18
  %50 = load <2 x double>, ptr %48, align 8, !tbaa !10, !alias.scope !21, !noalias !18
  %51 = fadd <2 x double> %45, %49
  %52 = fadd <2 x double> %46, %50
  store <2 x double> %51, ptr %47, align 8, !tbaa !10, !alias.scope !21, !noalias !18
  store <2 x double> %52, ptr %48, align 8, !tbaa !10, !alias.scope !21, !noalias !18
  %53 = add nuw i64 %42, 4
  %54 = icmp eq i64 %53, 2000
  br i1 %54, label %64, label %41, !llvm.loop !23

55:                                               ; preds = %33, %55
  %56 = phi i64 [ %62, %55 ], [ 0, %33 ]
  %57 = getelementptr inbounds nuw double, ptr %36, i64 %56
  %58 = load double, ptr %57, align 8, !tbaa !10
  %59 = getelementptr inbounds nuw double, ptr %1, i64 %56
  %60 = load double, ptr %59, align 8, !tbaa !10
  %61 = fadd double %58, %60
  store double %61, ptr %59, align 8, !tbaa !10
  %62 = add nuw nsw i64 %56, 1
  %63 = icmp eq i64 %62, 2000
  br i1 %63, label %64, label %55, !llvm.loop !26

64:                                               ; preds = %41, %55
  %65 = add nuw nsw i64 %34, 1
  %66 = icmp eq i64 %65, 2000
  br i1 %66, label %67, label %33, !llvm.loop !27

67:                                               ; preds = %64
  ret void
}

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: read) uwtable
define dso_local i32 @idamax(i32 noundef %0, ptr noundef readonly captures(none) %1, i32 noundef %2, i32 noundef %3) local_unnamed_addr #1 {
  %5 = icmp slt i32 %0, 1
  br i1 %5, label %52, label %6

6:                                                ; preds = %4
  %7 = icmp eq i32 %0, 1
  br i1 %7, label %52, label %8

8:                                                ; preds = %6
  %9 = icmp eq i32 %3, 1
  %10 = sext i32 %2 to i64
  br i1 %9, label %33, label %11

11:                                               ; preds = %8
  %12 = add i32 %3, 1
  %13 = getelementptr inbounds double, ptr %1, i64 %10
  %14 = load double, ptr %13, align 8, !tbaa !10
  %15 = tail call double @llvm.fabs.f64(double %14)
  %16 = sext i32 %12 to i64
  %17 = sext i32 %3 to i64
  %18 = getelementptr double, ptr %1, i64 %10
  br label %19

19:                                               ; preds = %11, %19
  %20 = phi i64 [ %16, %11 ], [ %30, %19 ]
  %21 = phi i32 [ 0, %11 ], [ %29, %19 ]
  %22 = phi i32 [ 1, %11 ], [ %31, %19 ]
  %23 = phi double [ %15, %11 ], [ %28, %19 ]
  %24 = getelementptr double, ptr %18, i64 %20
  %25 = load double, ptr %24, align 8, !tbaa !10
  %26 = tail call double @llvm.fabs.f64(double %25)
  %27 = fcmp ogt double %26, %23
  %28 = select i1 %27, double %26, double %23
  %29 = select i1 %27, i32 %22, i32 %21
  %30 = add nsw i64 %20, %17
  %31 = add nuw nsw i32 %22, 1
  %32 = icmp eq i32 %31, %0
  br i1 %32, label %52, label %19, !llvm.loop !28

33:                                               ; preds = %8
  %34 = getelementptr inbounds double, ptr %1, i64 %10
  %35 = load double, ptr %34, align 8, !tbaa !10
  %36 = tail call double @llvm.fabs.f64(double %35)
  %37 = zext nneg i32 %0 to i64
  %38 = getelementptr double, ptr %1, i64 %10
  br label %39

39:                                               ; preds = %33, %39
  %40 = phi i64 [ 1, %33 ], [ %50, %39 ]
  %41 = phi i32 [ 0, %33 ], [ %49, %39 ]
  %42 = phi double [ %36, %33 ], [ %47, %39 ]
  %43 = getelementptr double, ptr %38, i64 %40
  %44 = load double, ptr %43, align 8, !tbaa !10
  %45 = tail call double @llvm.fabs.f64(double %44)
  %46 = fcmp ogt double %45, %42
  %47 = select i1 %46, double %45, double %42
  %48 = trunc nuw nsw i64 %40 to i32
  %49 = select i1 %46, i32 %48, i32 %41
  %50 = add nuw nsw i64 %40, 1
  %51 = icmp eq i64 %50, %37
  br i1 %51, label %52, label %39, !llvm.loop !29

52:                                               ; preds = %19, %39, %6, %4
  %53 = phi i32 [ -1, %4 ], [ 0, %6 ], [ %49, %39 ], [ %29, %19 ]
  ret i32 %53
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fabs.f64(double) #2

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable
define dso_local void @dscal(i32 noundef %0, double noundef %1, ptr noundef captures(none) %2, i32 noundef %3, i32 noundef %4) local_unnamed_addr #3 {
  %6 = icmp sgt i32 %0, 0
  br i1 %6, label %7, label %54

7:                                                ; preds = %5
  %8 = icmp eq i32 %4, 1
  br i1 %8, label %9, label %32

9:                                                ; preds = %7
  %10 = sext i32 %3 to i64
  %11 = zext nneg i32 %0 to i64
  %12 = getelementptr double, ptr %2, i64 %10
  %13 = icmp ult i32 %0, 4
  br i1 %13, label %30, label %14

14:                                               ; preds = %9
  %15 = and i64 %11, 2147483644
  %16 = insertelement <2 x double> poison, double %1, i64 0
  %17 = shufflevector <2 x double> %16, <2 x double> poison, <2 x i32> zeroinitializer
  br label %18

18:                                               ; preds = %18, %14
  %19 = phi i64 [ 0, %14 ], [ %26, %18 ]
  %20 = getelementptr double, ptr %12, i64 %19
  %21 = getelementptr i8, ptr %20, i64 16
  %22 = load <2 x double>, ptr %20, align 8, !tbaa !10
  %23 = load <2 x double>, ptr %21, align 8, !tbaa !10
  %24 = fmul <2 x double> %17, %22
  %25 = fmul <2 x double> %17, %23
  store <2 x double> %24, ptr %20, align 8, !tbaa !10
  store <2 x double> %25, ptr %21, align 8, !tbaa !10
  %26 = add nuw i64 %19, 4
  %27 = icmp eq i64 %26, %15
  br i1 %27, label %28, label %18, !llvm.loop !30

28:                                               ; preds = %18
  %29 = icmp eq i64 %15, %11
  br i1 %29, label %54, label %30

30:                                               ; preds = %9, %28
  %31 = phi i64 [ 0, %9 ], [ %15, %28 ]
  br label %47

32:                                               ; preds = %7
  %33 = mul nsw i32 %4, %0
  %34 = icmp sgt i32 %33, 0
  br i1 %34, label %35, label %54

35:                                               ; preds = %32
  %36 = sext i32 %4 to i64
  %37 = sext i32 %3 to i64
  %38 = zext nneg i32 %33 to i64
  %39 = getelementptr double, ptr %2, i64 %37
  br label %40

40:                                               ; preds = %35, %40
  %41 = phi i64 [ 0, %35 ], [ %45, %40 ]
  %42 = getelementptr double, ptr %39, i64 %41
  %43 = load double, ptr %42, align 8, !tbaa !10
  %44 = fmul double %1, %43
  store double %44, ptr %42, align 8, !tbaa !10
  %45 = add nsw i64 %41, %36
  %46 = icmp slt i64 %45, %38
  br i1 %46, label %40, label %54, !llvm.loop !31

47:                                               ; preds = %30, %47
  %48 = phi i64 [ %52, %47 ], [ %31, %30 ]
  %49 = getelementptr double, ptr %12, i64 %48
  %50 = load double, ptr %49, align 8, !tbaa !10
  %51 = fmul double %1, %50
  store double %51, ptr %49, align 8, !tbaa !10
  %52 = add nuw nsw i64 %48, 1
  %53 = icmp eq i64 %52, %11
  br i1 %53, label %54, label %47, !llvm.loop !32

54:                                               ; preds = %40, %47, %28, %32, %5
  ret void
}

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable
define dso_local void @daxpy(i32 noundef %0, double noundef %1, ptr noundef readonly captures(none) %2, i32 noundef %3, i32 noundef %4, ptr noundef captures(none) %5, i32 noundef %6, i32 noundef %7) local_unnamed_addr #3 {
  %9 = icmp sgt i32 %0, 0
  %10 = fcmp une double %1, 0.000000e+00
  %11 = and i1 %9, %10
  br i1 %11, label %12, label %151

12:                                               ; preds = %8
  %13 = icmp ne i32 %4, 1
  %14 = icmp ne i32 %7, 1
  %15 = or i1 %13, %14
  br i1 %15, label %55, label %16

16:                                               ; preds = %12
  %17 = sext i32 %3 to i64
  %18 = sext i32 %6 to i64
  %19 = zext nneg i32 %0 to i64
  %20 = getelementptr double, ptr %2, i64 %17
  %21 = getelementptr double, ptr %5, i64 %18
  %22 = icmp ult i32 %0, 8
  br i1 %22, label %53, label %23

23:                                               ; preds = %16
  %24 = add nsw i64 %18, %19
  %25 = shl nsw i64 %24, 3
  %26 = getelementptr i8, ptr %5, i64 %25
  %27 = add nsw i64 %17, %19
  %28 = shl nsw i64 %27, 3
  %29 = getelementptr i8, ptr %2, i64 %28
  %30 = icmp ult ptr %21, %29
  %31 = icmp ult ptr %20, %26
  %32 = and i1 %30, %31
  br i1 %32, label %53, label %33

33:                                               ; preds = %23
  %34 = and i64 %19, 2147483644
  %35 = insertelement <2 x double> poison, double %1, i64 0
  %36 = shufflevector <2 x double> %35, <2 x double> poison, <2 x i32> zeroinitializer
  br label %37

37:                                               ; preds = %37, %33
  %38 = phi i64 [ 0, %33 ], [ %49, %37 ]
  %39 = getelementptr double, ptr %20, i64 %38
  %40 = getelementptr i8, ptr %39, i64 16
  %41 = load <2 x double>, ptr %39, align 8, !tbaa !10, !alias.scope !33
  %42 = load <2 x double>, ptr %40, align 8, !tbaa !10, !alias.scope !33
  %43 = getelementptr double, ptr %21, i64 %38
  %44 = getelementptr i8, ptr %43, i64 16
  %45 = load <2 x double>, ptr %43, align 8, !tbaa !10, !alias.scope !36, !noalias !33
  %46 = load <2 x double>, ptr %44, align 8, !tbaa !10, !alias.scope !36, !noalias !33
  %47 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %36, <2 x double> %41, <2 x double> %45)
  %48 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %36, <2 x double> %42, <2 x double> %46)
  store <2 x double> %47, ptr %43, align 8, !tbaa !10, !alias.scope !36, !noalias !33
  store <2 x double> %48, ptr %44, align 8, !tbaa !10, !alias.scope !36, !noalias !33
  %49 = add nuw i64 %38, 4
  %50 = icmp eq i64 %49, %34
  br i1 %50, label %51, label %37, !llvm.loop !38

51:                                               ; preds = %37
  %52 = icmp eq i64 %34, %19
  br i1 %52, label %151, label %53

53:                                               ; preds = %23, %16, %51
  %54 = phi i64 [ 0, %23 ], [ 0, %16 ], [ %34, %51 ]
  br label %142

55:                                               ; preds = %12
  %56 = icmp slt i32 %7, 0
  %57 = sub nsw i32 1, %0
  %58 = mul nsw i32 %7, %57
  %59 = select i1 %56, i32 %58, i32 0
  %60 = icmp slt i32 %4, 0
  %61 = mul nsw i32 %4, %57
  %62 = select i1 %60, i32 %61, i32 0
  %63 = sext i32 %59 to i64
  %64 = sext i32 %7 to i64
  %65 = sext i32 %6 to i64
  %66 = sext i32 %62 to i64
  %67 = sext i32 %4 to i64
  %68 = sext i32 %3 to i64
  %69 = getelementptr double, ptr %2, i64 %68
  %70 = getelementptr double, ptr %5, i64 %65
  %71 = zext nneg i32 %0 to i64
  %72 = icmp ult i32 %0, 16
  br i1 %72, label %125, label %73

73:                                               ; preds = %55
  %74 = icmp ne i32 %7, 1
  %75 = icmp ne i32 %4, 1
  %76 = or i1 %74, %75
  br i1 %76, label %125, label %77

77:                                               ; preds = %73
  %78 = add nsw i64 %65, %63
  %79 = shl nsw i64 %78, 3
  %80 = getelementptr i8, ptr %5, i64 %79
  %81 = add nsw i32 %0, -1
  %82 = zext i32 %81 to i64
  %83 = shl nuw nsw i64 %82, 3
  %84 = getelementptr i8, ptr %5, i64 %79
  %85 = getelementptr i8, ptr %84, i64 %83
  %86 = getelementptr i8, ptr %85, i64 8
  %87 = add nsw i64 %68, %66
  %88 = shl nsw i64 %87, 3
  %89 = getelementptr i8, ptr %2, i64 %88
  %90 = getelementptr i8, ptr %2, i64 %88
  %91 = getelementptr i8, ptr %90, i64 %83
  %92 = getelementptr i8, ptr %91, i64 8
  %93 = icmp ult ptr %80, %92
  %94 = icmp ult ptr %89, %86
  %95 = and i1 %93, %94
  br i1 %95, label %125, label %96

96:                                               ; preds = %77
  %97 = and i64 %71, 2147483644
  %98 = mul nuw nsw i64 %97, %67
  %99 = add nsw i64 %98, %66
  %100 = mul nuw nsw i64 %97, %64
  %101 = add nsw i64 %100, %63
  %102 = trunc nuw nsw i64 %97 to i32
  %103 = insertelement <2 x double> poison, double %1, i64 0
  %104 = shufflevector <2 x double> %103, <2 x double> poison, <2 x i32> zeroinitializer
  %105 = getelementptr double, ptr %69, i64 %66
  %106 = getelementptr double, ptr %70, i64 %63
  br label %107

107:                                              ; preds = %107, %96
  %108 = phi i64 [ 0, %96 ], [ %121, %107 ]
  %109 = mul nuw i64 %108, %67
  %110 = mul nuw i64 %108, %64
  %111 = getelementptr double, ptr %105, i64 %109
  %112 = getelementptr i8, ptr %111, i64 16
  %113 = load <2 x double>, ptr %111, align 8, !tbaa !10, !alias.scope !39
  %114 = load <2 x double>, ptr %112, align 8, !tbaa !10, !alias.scope !39
  %115 = getelementptr double, ptr %106, i64 %110
  %116 = getelementptr i8, ptr %115, i64 16
  %117 = load <2 x double>, ptr %115, align 8, !tbaa !10, !alias.scope !42, !noalias !39
  %118 = load <2 x double>, ptr %116, align 8, !tbaa !10, !alias.scope !42, !noalias !39
  %119 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %104, <2 x double> %113, <2 x double> %117)
  %120 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %104, <2 x double> %114, <2 x double> %118)
  store <2 x double> %119, ptr %115, align 8, !tbaa !10, !alias.scope !42, !noalias !39
  store <2 x double> %120, ptr %116, align 8, !tbaa !10, !alias.scope !42, !noalias !39
  %121 = add nuw i64 %108, 4
  %122 = icmp eq i64 %121, %97
  br i1 %122, label %123, label %107, !llvm.loop !44

123:                                              ; preds = %107
  %124 = icmp eq i64 %97, %71
  br i1 %124, label %151, label %125

125:                                              ; preds = %77, %73, %55, %123
  %126 = phi i64 [ %66, %77 ], [ %66, %73 ], [ %66, %55 ], [ %99, %123 ]
  %127 = phi i64 [ %63, %77 ], [ %63, %73 ], [ %63, %55 ], [ %101, %123 ]
  %128 = phi i32 [ 0, %77 ], [ 0, %73 ], [ 0, %55 ], [ %102, %123 ]
  br label %129

129:                                              ; preds = %125, %129
  %130 = phi i64 [ %138, %129 ], [ %126, %125 ]
  %131 = phi i64 [ %139, %129 ], [ %127, %125 ]
  %132 = phi i32 [ %140, %129 ], [ %128, %125 ]
  %133 = getelementptr double, ptr %69, i64 %130
  %134 = load double, ptr %133, align 8, !tbaa !10
  %135 = getelementptr double, ptr %70, i64 %131
  %136 = load double, ptr %135, align 8, !tbaa !10
  %137 = tail call double @llvm.fmuladd.f64(double %1, double %134, double %136)
  store double %137, ptr %135, align 8, !tbaa !10
  %138 = add nsw i64 %130, %67
  %139 = add nsw i64 %131, %64
  %140 = add nuw nsw i32 %132, 1
  %141 = icmp eq i32 %140, %0
  br i1 %141, label %151, label %129, !llvm.loop !45

142:                                              ; preds = %53, %142
  %143 = phi i64 [ %149, %142 ], [ %54, %53 ]
  %144 = getelementptr double, ptr %20, i64 %143
  %145 = load double, ptr %144, align 8, !tbaa !10
  %146 = getelementptr double, ptr %21, i64 %143
  %147 = load double, ptr %146, align 8, !tbaa !10
  %148 = tail call double @llvm.fmuladd.f64(double %1, double %145, double %147)
  store double %148, ptr %146, align 8, !tbaa !10
  %149 = add nuw nsw i64 %143, 1
  %150 = icmp eq i64 %149, %19
  br i1 %150, label %151, label %142, !llvm.loop !46

151:                                              ; preds = %142, %129, %51, %123, %8
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fmuladd.f64(double, double, double) #2

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define dso_local void @dgefa(ptr noundef readonly captures(none) %0, ptr noundef writeonly captures(none) %1) local_unnamed_addr #0 {
  br label %3

3:                                                ; preds = %2, %125
  %4 = phi i64 [ 0, %2 ], [ %28, %125 ]
  %5 = phi i64 [ 1, %2 ], [ %126, %125 ]
  %6 = sub nsw i64 1999, %4
  %7 = sub nsw i64 1999, %4
  %8 = getelementptr inbounds nuw ptr, ptr %0, i64 %4
  %9 = load ptr, ptr %8, align 8, !tbaa !12
  %10 = sub nuw nsw i64 2000, %4
  %11 = getelementptr inbounds nuw double, ptr %9, i64 %4
  %12 = load double, ptr %11, align 8, !tbaa !10
  %13 = tail call double @llvm.fabs.f64(double %12)
  br label %14

14:                                               ; preds = %14, %3
  %15 = phi i64 [ 1, %3 ], [ %25, %14 ]
  %16 = phi i32 [ 0, %3 ], [ %24, %14 ]
  %17 = phi double [ %13, %3 ], [ %22, %14 ]
  %18 = getelementptr double, ptr %11, i64 %15
  %19 = load double, ptr %18, align 8, !tbaa !10
  %20 = tail call double @llvm.fabs.f64(double %19)
  %21 = fcmp ogt double %20, %17
  %22 = select i1 %21, double %20, double %17
  %23 = trunc nuw nsw i64 %15 to i32
  %24 = select i1 %21, i32 %23, i32 %16
  %25 = add nuw nsw i64 %15, 1
  %26 = icmp eq i64 %25, %10
  br i1 %26, label %27, label %14, !llvm.loop !29

27:                                               ; preds = %14
  %28 = add nuw nsw i64 %4, 1
  %29 = trunc nuw nsw i64 %4 to i32
  %30 = add nsw i32 %24, %29
  %31 = getelementptr inbounds nuw i32, ptr %1, i64 %4
  store i32 %30, ptr %31, align 4, !tbaa !47
  %32 = sext i32 %30 to i64
  %33 = getelementptr inbounds double, ptr %9, i64 %32
  %34 = load double, ptr %33, align 8, !tbaa !10
  %35 = fcmp une double %34, 0.000000e+00
  br i1 %35, label %36, label %125

36:                                               ; preds = %27
  %37 = icmp eq i32 %24, 0
  br i1 %37, label %39, label %38

38:                                               ; preds = %36
  store double %12, ptr %33, align 8, !tbaa !10
  store double %34, ptr %11, align 8, !tbaa !10
  br label %39

39:                                               ; preds = %38, %36
  %40 = phi double [ %34, %38 ], [ %12, %36 ]
  %41 = fdiv double -1.000000e+00, %40
  %42 = sub nuw nsw i64 1999, %4
  %43 = getelementptr double, ptr %9, i64 %28
  %44 = icmp ult i64 %6, 4
  br i1 %44, label %61, label %45

45:                                               ; preds = %39
  %46 = and i64 %6, -4
  %47 = insertelement <2 x double> poison, double %41, i64 0
  %48 = shufflevector <2 x double> %47, <2 x double> poison, <2 x i32> zeroinitializer
  br label %49

49:                                               ; preds = %49, %45
  %50 = phi i64 [ 0, %45 ], [ %57, %49 ]
  %51 = getelementptr double, ptr %43, i64 %50
  %52 = getelementptr i8, ptr %51, i64 16
  %53 = load <2 x double>, ptr %51, align 8, !tbaa !10
  %54 = load <2 x double>, ptr %52, align 8, !tbaa !10
  %55 = fmul <2 x double> %48, %53
  %56 = fmul <2 x double> %48, %54
  store <2 x double> %55, ptr %51, align 8, !tbaa !10
  store <2 x double> %56, ptr %52, align 8, !tbaa !10
  %57 = add nuw i64 %50, 4
  %58 = icmp eq i64 %57, %46
  br i1 %58, label %59, label %49, !llvm.loop !49

59:                                               ; preds = %49
  %60 = icmp eq i64 %6, %46
  br i1 %60, label %70, label %61

61:                                               ; preds = %39, %59
  %62 = phi i64 [ 0, %39 ], [ %46, %59 ]
  br label %63

63:                                               ; preds = %61, %63
  %64 = phi i64 [ %68, %63 ], [ %62, %61 ]
  %65 = getelementptr double, ptr %43, i64 %64
  %66 = load double, ptr %65, align 8, !tbaa !10
  %67 = fmul double %41, %66
  store double %67, ptr %65, align 8, !tbaa !10
  %68 = add nuw nsw i64 %64, 1
  %69 = icmp eq i64 %68, %42
  br i1 %69, label %70, label %63, !llvm.loop !50

70:                                               ; preds = %63, %59
  %71 = getelementptr i8, ptr %9, i64 16000
  %72 = icmp ult i64 %7, 4
  %73 = and i64 %7, -4
  %74 = icmp eq i64 %7, %73
  br label %75

75:                                               ; preds = %70, %122
  %76 = phi i64 [ %123, %122 ], [ %5, %70 ]
  %77 = getelementptr inbounds nuw ptr, ptr %0, i64 %76
  %78 = load ptr, ptr %77, align 8, !tbaa !12
  %79 = getelementptr inbounds double, ptr %78, i64 %32
  %80 = load double, ptr %79, align 8, !tbaa !10
  br i1 %37, label %84, label %81

81:                                               ; preds = %75
  %82 = getelementptr inbounds nuw double, ptr %78, i64 %4
  %83 = load double, ptr %82, align 8, !tbaa !10
  store double %83, ptr %79, align 8, !tbaa !10
  store double %80, ptr %82, align 8, !tbaa !10
  br label %84

84:                                               ; preds = %81, %75
  %85 = fcmp une double %80, 0.000000e+00
  br i1 %85, label %86, label %122

86:                                               ; preds = %84
  %87 = getelementptr double, ptr %78, i64 %28
  br i1 %72, label %111, label %88

88:                                               ; preds = %86
  %89 = getelementptr i8, ptr %78, i64 16000
  %90 = icmp ult ptr %87, %71
  %91 = icmp ult ptr %43, %89
  %92 = and i1 %90, %91
  br i1 %92, label %111, label %93

93:                                               ; preds = %88
  %94 = insertelement <2 x double> poison, double %80, i64 0
  %95 = shufflevector <2 x double> %94, <2 x double> poison, <2 x i32> zeroinitializer
  br label %96

96:                                               ; preds = %96, %93
  %97 = phi i64 [ 0, %93 ], [ %108, %96 ]
  %98 = getelementptr double, ptr %43, i64 %97
  %99 = getelementptr i8, ptr %98, i64 16
  %100 = load <2 x double>, ptr %98, align 8, !tbaa !10, !alias.scope !51
  %101 = load <2 x double>, ptr %99, align 8, !tbaa !10, !alias.scope !51
  %102 = getelementptr double, ptr %87, i64 %97
  %103 = getelementptr i8, ptr %102, i64 16
  %104 = load <2 x double>, ptr %102, align 8, !tbaa !10, !alias.scope !54, !noalias !51
  %105 = load <2 x double>, ptr %103, align 8, !tbaa !10, !alias.scope !54, !noalias !51
  %106 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %95, <2 x double> %100, <2 x double> %104)
  %107 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %95, <2 x double> %101, <2 x double> %105)
  store <2 x double> %106, ptr %102, align 8, !tbaa !10, !alias.scope !54, !noalias !51
  store <2 x double> %107, ptr %103, align 8, !tbaa !10, !alias.scope !54, !noalias !51
  %108 = add nuw i64 %97, 4
  %109 = icmp eq i64 %108, %73
  br i1 %109, label %110, label %96, !llvm.loop !56

110:                                              ; preds = %96
  br i1 %74, label %122, label %111

111:                                              ; preds = %88, %86, %110
  %112 = phi i64 [ 0, %88 ], [ 0, %86 ], [ %73, %110 ]
  br label %113

113:                                              ; preds = %111, %113
  %114 = phi i64 [ %120, %113 ], [ %112, %111 ]
  %115 = getelementptr double, ptr %43, i64 %114
  %116 = load double, ptr %115, align 8, !tbaa !10
  %117 = getelementptr double, ptr %87, i64 %114
  %118 = load double, ptr %117, align 8, !tbaa !10
  %119 = tail call double @llvm.fmuladd.f64(double %80, double %116, double %118)
  store double %119, ptr %117, align 8, !tbaa !10
  %120 = add nuw nsw i64 %114, 1
  %121 = icmp eq i64 %120, %42
  br i1 %121, label %122, label %113, !llvm.loop !57

122:                                              ; preds = %113, %110, %84
  %123 = add nuw nsw i64 %76, 1
  %124 = icmp eq i64 %123, 2000
  br i1 %124, label %125, label %75, !llvm.loop !58

125:                                              ; preds = %122, %27
  %126 = add nuw nsw i64 %5, 1
  %127 = icmp eq i64 %28, 1999
  br i1 %127, label %128, label %3, !llvm.loop !59

128:                                              ; preds = %125
  %129 = getelementptr inbounds nuw i8, ptr %1, i64 7996
  store i32 1999, ptr %129, align 4, !tbaa !47
  ret void
}

; Function Attrs: nofree norecurse nosync nounwind memory(read, argmem: readwrite, inaccessiblemem: none) uwtable
define dso_local void @dgesl(ptr noundef readonly captures(none) %0, ptr noundef readonly captures(none) %1, ptr noundef captures(none) %2) local_unnamed_addr #4 {
  %4 = getelementptr i8, ptr %2, i64 16000
  br label %5

5:                                                ; preds = %3, %67
  %6 = phi i64 [ 0, %3 ], [ %22, %67 ]
  %7 = sub nsw i64 1999, %6
  %8 = shl nuw nsw i64 %6, 3
  %9 = getelementptr i8, ptr %2, i64 %8
  %10 = getelementptr i8, ptr %9, i64 8
  %11 = getelementptr inbounds nuw i32, ptr %1, i64 %6
  %12 = load i32, ptr %11, align 4, !tbaa !47
  %13 = sext i32 %12 to i64
  %14 = getelementptr inbounds double, ptr %2, i64 %13
  %15 = load double, ptr %14, align 8, !tbaa !10
  %16 = zext i32 %12 to i64
  %17 = icmp eq i64 %6, %16
  br i1 %17, label %21, label %18

18:                                               ; preds = %5
  %19 = getelementptr inbounds nuw double, ptr %2, i64 %6
  %20 = load double, ptr %19, align 8, !tbaa !10
  store double %20, ptr %14, align 8, !tbaa !10
  store double %15, ptr %19, align 8, !tbaa !10
  br label %21

21:                                               ; preds = %18, %5
  %22 = add nuw nsw i64 %6, 1
  %23 = fcmp une double %15, 0.000000e+00
  br i1 %23, label %24, label %67

24:                                               ; preds = %21
  %25 = getelementptr inbounds nuw ptr, ptr %0, i64 %6
  %26 = load ptr, ptr %25, align 8, !tbaa !12
  %27 = sub nuw nsw i64 1999, %6
  %28 = getelementptr double, ptr %26, i64 %22
  %29 = getelementptr double, ptr %2, i64 %22
  %30 = icmp ult i64 %7, 4
  br i1 %30, label %56, label %31

31:                                               ; preds = %24
  %32 = getelementptr i8, ptr %26, i64 16000
  %33 = icmp ult ptr %10, %32
  %34 = icmp ult ptr %28, %4
  %35 = and i1 %33, %34
  br i1 %35, label %56, label %36

36:                                               ; preds = %31
  %37 = and i64 %7, -4
  %38 = insertelement <2 x double> poison, double %15, i64 0
  %39 = shufflevector <2 x double> %38, <2 x double> poison, <2 x i32> zeroinitializer
  br label %40

40:                                               ; preds = %40, %36
  %41 = phi i64 [ 0, %36 ], [ %52, %40 ]
  %42 = getelementptr double, ptr %28, i64 %41
  %43 = getelementptr i8, ptr %42, i64 16
  %44 = load <2 x double>, ptr %42, align 8, !tbaa !10, !alias.scope !60
  %45 = load <2 x double>, ptr %43, align 8, !tbaa !10, !alias.scope !60
  %46 = getelementptr double, ptr %29, i64 %41
  %47 = getelementptr i8, ptr %46, i64 16
  %48 = load <2 x double>, ptr %46, align 8, !tbaa !10, !alias.scope !63, !noalias !60
  %49 = load <2 x double>, ptr %47, align 8, !tbaa !10, !alias.scope !63, !noalias !60
  %50 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %39, <2 x double> %44, <2 x double> %48)
  %51 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %39, <2 x double> %45, <2 x double> %49)
  store <2 x double> %50, ptr %46, align 8, !tbaa !10, !alias.scope !63, !noalias !60
  store <2 x double> %51, ptr %47, align 8, !tbaa !10, !alias.scope !63, !noalias !60
  %52 = add nuw i64 %41, 4
  %53 = icmp eq i64 %52, %37
  br i1 %53, label %54, label %40, !llvm.loop !65

54:                                               ; preds = %40
  %55 = icmp eq i64 %7, %37
  br i1 %55, label %67, label %56

56:                                               ; preds = %31, %24, %54
  %57 = phi i64 [ 0, %31 ], [ 0, %24 ], [ %37, %54 ]
  br label %58

58:                                               ; preds = %56, %58
  %59 = phi i64 [ %65, %58 ], [ %57, %56 ]
  %60 = getelementptr double, ptr %28, i64 %59
  %61 = load double, ptr %60, align 8, !tbaa !10
  %62 = getelementptr double, ptr %29, i64 %59
  %63 = load double, ptr %62, align 8, !tbaa !10
  %64 = tail call double @llvm.fmuladd.f64(double %15, double %61, double %63)
  store double %64, ptr %62, align 8, !tbaa !10
  %65 = add nuw nsw i64 %59, 1
  %66 = icmp eq i64 %65, %27
  br i1 %66, label %67, label %58, !llvm.loop !66

67:                                               ; preds = %58, %54, %21
  %68 = icmp eq i64 %22, 1999
  br i1 %68, label %69, label %5, !llvm.loop !67

69:                                               ; preds = %67, %126
  %70 = phi i64 [ %75, %126 ], [ 0, %67 ]
  %71 = sub nsw i64 1999, %70
  %72 = shl i64 %70, 3
  %73 = sub i64 15992, %72
  %74 = getelementptr i8, ptr %2, i64 %73
  %75 = add nuw nsw i64 %70, 1
  %76 = sub nuw nsw i64 1999, %70
  %77 = getelementptr inbounds nuw ptr, ptr %0, i64 %76
  %78 = load ptr, ptr %77, align 8, !tbaa !12
  %79 = getelementptr inbounds nuw double, ptr %78, i64 %76
  %80 = load double, ptr %79, align 8, !tbaa !10
  %81 = getelementptr inbounds nuw double, ptr %2, i64 %76
  %82 = load double, ptr %81, align 8, !tbaa !10
  %83 = fdiv double %82, %80
  store double %83, ptr %81, align 8, !tbaa !10
  %84 = fneg double %83
  %85 = icmp ne i64 %70, 1999
  %86 = fcmp une double %83, 0.000000e+00
  %87 = and i1 %85, %86
  br i1 %87, label %88, label %126

88:                                               ; preds = %69
  %89 = icmp ult i64 %71, 4
  br i1 %89, label %115, label %90

90:                                               ; preds = %88
  %91 = getelementptr i8, ptr %78, i64 %73
  %92 = icmp ult ptr %2, %91
  %93 = icmp ult ptr %78, %74
  %94 = and i1 %92, %93
  br i1 %94, label %115, label %95

95:                                               ; preds = %90
  %96 = and i64 %71, -4
  %97 = insertelement <2 x double> poison, double %84, i64 0
  %98 = shufflevector <2 x double> %97, <2 x double> poison, <2 x i32> zeroinitializer
  br label %99

99:                                               ; preds = %99, %95
  %100 = phi i64 [ 0, %95 ], [ %111, %99 ]
  %101 = getelementptr double, ptr %78, i64 %100
  %102 = getelementptr i8, ptr %101, i64 16
  %103 = load <2 x double>, ptr %101, align 8, !tbaa !10, !alias.scope !68
  %104 = load <2 x double>, ptr %102, align 8, !tbaa !10, !alias.scope !68
  %105 = getelementptr double, ptr %2, i64 %100
  %106 = getelementptr i8, ptr %105, i64 16
  %107 = load <2 x double>, ptr %105, align 8, !tbaa !10, !alias.scope !71, !noalias !68
  %108 = load <2 x double>, ptr %106, align 8, !tbaa !10, !alias.scope !71, !noalias !68
  %109 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %98, <2 x double> %103, <2 x double> %107)
  %110 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %98, <2 x double> %104, <2 x double> %108)
  store <2 x double> %109, ptr %105, align 8, !tbaa !10, !alias.scope !71, !noalias !68
  store <2 x double> %110, ptr %106, align 8, !tbaa !10, !alias.scope !71, !noalias !68
  %111 = add nuw i64 %100, 4
  %112 = icmp eq i64 %111, %96
  br i1 %112, label %113, label %99, !llvm.loop !73

113:                                              ; preds = %99
  %114 = icmp eq i64 %71, %96
  br i1 %114, label %126, label %115

115:                                              ; preds = %90, %88, %113
  %116 = phi i64 [ 0, %90 ], [ 0, %88 ], [ %96, %113 ]
  br label %117

117:                                              ; preds = %115, %117
  %118 = phi i64 [ %124, %117 ], [ %116, %115 ]
  %119 = getelementptr double, ptr %78, i64 %118
  %120 = load double, ptr %119, align 8, !tbaa !10
  %121 = getelementptr double, ptr %2, i64 %118
  %122 = load double, ptr %121, align 8, !tbaa !10
  %123 = tail call double @llvm.fmuladd.f64(double %84, double %120, double %122)
  store double %123, ptr %121, align 8, !tbaa !10
  %124 = add nuw nsw i64 %118, 1
  %125 = icmp eq i64 %124, %76
  br i1 %125, label %126, label %117, !llvm.loop !74

126:                                              ; preds = %117, %113, %69
  %127 = icmp eq i64 %75, 2000
  br i1 %127, label %128, label %69, !llvm.loop !75

128:                                              ; preds = %126
  ret void
}

; Function Attrs: nounwind uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #5 {
  %3 = icmp sgt i32 %0, 1
  br i1 %3, label %4, label %10

4:                                                ; preds = %2
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %6 = load ptr, ptr %5, align 8, !tbaa !76
  %7 = tail call i32 @strcmp(ptr noundef nonnull dereferenceable(1) %6, ptr noundef nonnull dereferenceable(4) @.str) #12
  %8 = icmp eq i32 %7, 0
  %9 = select i1 %8, ptr @.str.1, ptr @.str.2
  br label %10

10:                                               ; preds = %4, %2
  %11 = phi ptr [ @.str.2, %2 ], [ %9, %4 ]
  %12 = tail call noalias dereferenceable_or_null(16000) ptr @malloc(i64 noundef 16000) #13
  br label %13

13:                                               ; preds = %10, %13
  %14 = phi i64 [ 0, %10 ], [ %17, %13 ]
  %15 = tail call noalias dereferenceable_or_null(16008) ptr @malloc(i64 noundef 16008) #13
  %16 = getelementptr inbounds nuw ptr, ptr %12, i64 %14
  store ptr %15, ptr %16, align 8, !tbaa !12
  %17 = add nuw nsw i64 %14, 1
  %18 = icmp eq i64 %17, 2000
  br i1 %18, label %19, label %13, !llvm.loop !78

19:                                               ; preds = %13
  %20 = tail call noalias dereferenceable_or_null(16000) ptr @malloc(i64 noundef 16000) #13
  %21 = tail call noalias dereferenceable_or_null(8000) ptr @malloc(i64 noundef 8000) #13
  %22 = load i64, ptr @seed, align 8, !tbaa !6
  %23 = xor i64 %22, 123459876
  br label %24

24:                                               ; preds = %48, %19
  %25 = phi i64 [ 0, %19 ], [ %49, %48 ]
  %26 = phi i64 [ %23, %19 ], [ %40, %48 ]
  br label %29

27:                                               ; preds = %48
  %28 = xor i64 %40, 123459876
  store i64 %28, ptr @seed, align 8, !tbaa !6
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16000) %20, i8 0, i64 16000, i1 false), !tbaa !10
  br label %51

29:                                               ; preds = %29, %24
  %30 = phi i64 [ 0, %24 ], [ %46, %29 ]
  %31 = phi i64 [ %26, %24 ], [ %40, %29 ]
  %32 = sdiv i64 %31, 127773
  %33 = mul nsw i64 %32, -127773
  %34 = add i64 %33, %31
  %35 = mul nsw i64 %34, 16807
  %36 = mul nsw i64 %32, -2836
  %37 = add i64 %35, %36
  %38 = icmp slt i64 %37, 0
  %39 = add nsw i64 %37, 2147483647
  %40 = select i1 %38, i64 %39, i64 %37
  %41 = sitofp i64 %40 to double
  %42 = fmul double %41, 0x3E00000000200FE1
  %43 = getelementptr inbounds nuw ptr, ptr %12, i64 %30
  %44 = load ptr, ptr %43, align 8, !tbaa !12
  %45 = getelementptr inbounds nuw double, ptr %44, i64 %25
  store double %42, ptr %45, align 8, !tbaa !10
  %46 = add nuw nsw i64 %30, 1
  %47 = icmp eq i64 %46, 2000
  br i1 %47, label %48, label %29, !llvm.loop !15

48:                                               ; preds = %29
  %49 = add nuw nsw i64 %25, 1
  %50 = icmp eq i64 %49, 2000
  br i1 %50, label %27, label %24, !llvm.loop !17

51:                                               ; preds = %69, %27
  %52 = phi i64 [ 0, %27 ], [ %70, %69 ]
  %53 = getelementptr inbounds nuw ptr, ptr %12, i64 %52
  %54 = load ptr, ptr %53, align 8, !tbaa !12
  br label %55

55:                                               ; preds = %55, %51
  %56 = phi i64 [ 0, %51 ], [ %67, %55 ]
  %57 = getelementptr inbounds nuw double, ptr %54, i64 %56
  %58 = getelementptr inbounds nuw i8, ptr %57, i64 16
  %59 = load <2 x double>, ptr %57, align 8, !tbaa !10
  %60 = load <2 x double>, ptr %58, align 8, !tbaa !10
  %61 = getelementptr inbounds nuw double, ptr %20, i64 %56
  %62 = getelementptr inbounds nuw i8, ptr %61, i64 16
  %63 = load <2 x double>, ptr %61, align 8, !tbaa !10
  %64 = load <2 x double>, ptr %62, align 8, !tbaa !10
  %65 = fadd <2 x double> %59, %63
  %66 = fadd <2 x double> %60, %64
  store <2 x double> %65, ptr %61, align 8, !tbaa !10
  store <2 x double> %66, ptr %62, align 8, !tbaa !10
  %67 = add nuw i64 %56, 4
  %68 = icmp eq i64 %67, 2000
  br i1 %68, label %69, label %55, !llvm.loop !79

69:                                               ; preds = %55
  %70 = add nuw nsw i64 %52, 1
  %71 = icmp eq i64 %70, 2000
  br i1 %71, label %72, label %51, !llvm.loop !27

72:                                               ; preds = %69, %194
  %73 = phi i64 [ %97, %194 ], [ 0, %69 ]
  %74 = phi i64 [ %195, %194 ], [ 1, %69 ]
  %75 = sub nsw i64 1999, %73
  %76 = sub nsw i64 1999, %73
  %77 = getelementptr inbounds nuw ptr, ptr %12, i64 %73
  %78 = load ptr, ptr %77, align 8, !tbaa !12
  %79 = sub nuw nsw i64 2000, %73
  %80 = getelementptr inbounds nuw double, ptr %78, i64 %73
  %81 = load double, ptr %80, align 8, !tbaa !10
  %82 = tail call double @llvm.fabs.f64(double %81)
  br label %83

83:                                               ; preds = %83, %72
  %84 = phi i64 [ 1, %72 ], [ %94, %83 ]
  %85 = phi i32 [ 0, %72 ], [ %93, %83 ]
  %86 = phi double [ %82, %72 ], [ %91, %83 ]
  %87 = getelementptr double, ptr %80, i64 %84
  %88 = load double, ptr %87, align 8, !tbaa !10
  %89 = tail call double @llvm.fabs.f64(double %88)
  %90 = fcmp ogt double %89, %86
  %91 = select i1 %90, double %89, double %86
  %92 = trunc nuw nsw i64 %84 to i32
  %93 = select i1 %90, i32 %92, i32 %85
  %94 = add nuw nsw i64 %84, 1
  %95 = icmp eq i64 %94, %79
  br i1 %95, label %96, label %83, !llvm.loop !29

96:                                               ; preds = %83
  %97 = add nuw nsw i64 %73, 1
  %98 = trunc nuw nsw i64 %73 to i32
  %99 = add nsw i32 %93, %98
  %100 = getelementptr inbounds nuw i32, ptr %21, i64 %73
  store i32 %99, ptr %100, align 4, !tbaa !47
  %101 = sext i32 %99 to i64
  %102 = getelementptr inbounds double, ptr %78, i64 %101
  %103 = load double, ptr %102, align 8, !tbaa !10
  %104 = fcmp une double %103, 0.000000e+00
  br i1 %104, label %105, label %194

105:                                              ; preds = %96
  %106 = icmp eq i32 %93, 0
  br i1 %106, label %108, label %107

107:                                              ; preds = %105
  store double %81, ptr %102, align 8, !tbaa !10
  store double %103, ptr %80, align 8, !tbaa !10
  br label %108

108:                                              ; preds = %107, %105
  %109 = phi double [ %103, %107 ], [ %81, %105 ]
  %110 = fdiv double -1.000000e+00, %109
  %111 = sub nuw nsw i64 1999, %73
  %112 = getelementptr double, ptr %78, i64 %97
  %113 = icmp ult i64 %75, 4
  br i1 %113, label %130, label %114

114:                                              ; preds = %108
  %115 = and i64 %75, -4
  %116 = insertelement <2 x double> poison, double %110, i64 0
  %117 = shufflevector <2 x double> %116, <2 x double> poison, <2 x i32> zeroinitializer
  br label %118

118:                                              ; preds = %118, %114
  %119 = phi i64 [ 0, %114 ], [ %126, %118 ]
  %120 = getelementptr double, ptr %112, i64 %119
  %121 = getelementptr i8, ptr %120, i64 16
  %122 = load <2 x double>, ptr %120, align 8, !tbaa !10
  %123 = load <2 x double>, ptr %121, align 8, !tbaa !10
  %124 = fmul <2 x double> %117, %122
  %125 = fmul <2 x double> %117, %123
  store <2 x double> %124, ptr %120, align 8, !tbaa !10
  store <2 x double> %125, ptr %121, align 8, !tbaa !10
  %126 = add nuw i64 %119, 4
  %127 = icmp eq i64 %126, %115
  br i1 %127, label %128, label %118, !llvm.loop !80

128:                                              ; preds = %118
  %129 = icmp eq i64 %75, %115
  br i1 %129, label %139, label %130

130:                                              ; preds = %108, %128
  %131 = phi i64 [ 0, %108 ], [ %115, %128 ]
  br label %132

132:                                              ; preds = %130, %132
  %133 = phi i64 [ %137, %132 ], [ %131, %130 ]
  %134 = getelementptr double, ptr %112, i64 %133
  %135 = load double, ptr %134, align 8, !tbaa !10
  %136 = fmul double %110, %135
  store double %136, ptr %134, align 8, !tbaa !10
  %137 = add nuw nsw i64 %133, 1
  %138 = icmp eq i64 %137, %111
  br i1 %138, label %139, label %132, !llvm.loop !81

139:                                              ; preds = %132, %128
  %140 = getelementptr i8, ptr %78, i64 16000
  %141 = icmp ult i64 %76, 4
  %142 = and i64 %76, -4
  %143 = icmp eq i64 %76, %142
  br label %144

144:                                              ; preds = %139, %191
  %145 = phi i64 [ %192, %191 ], [ %74, %139 ]
  %146 = getelementptr inbounds nuw ptr, ptr %12, i64 %145
  %147 = load ptr, ptr %146, align 8, !tbaa !12
  %148 = getelementptr inbounds double, ptr %147, i64 %101
  %149 = load double, ptr %148, align 8, !tbaa !10
  br i1 %106, label %153, label %150

150:                                              ; preds = %144
  %151 = getelementptr inbounds nuw double, ptr %147, i64 %73
  %152 = load double, ptr %151, align 8, !tbaa !10
  store double %152, ptr %148, align 8, !tbaa !10
  store double %149, ptr %151, align 8, !tbaa !10
  br label %153

153:                                              ; preds = %150, %144
  %154 = fcmp une double %149, 0.000000e+00
  br i1 %154, label %155, label %191

155:                                              ; preds = %153
  %156 = getelementptr double, ptr %147, i64 %97
  br i1 %141, label %180, label %157

157:                                              ; preds = %155
  %158 = getelementptr i8, ptr %147, i64 16000
  %159 = icmp ult ptr %156, %140
  %160 = icmp ult ptr %112, %158
  %161 = and i1 %159, %160
  br i1 %161, label %180, label %162

162:                                              ; preds = %157
  %163 = insertelement <2 x double> poison, double %149, i64 0
  %164 = shufflevector <2 x double> %163, <2 x double> poison, <2 x i32> zeroinitializer
  br label %165

165:                                              ; preds = %165, %162
  %166 = phi i64 [ 0, %162 ], [ %177, %165 ]
  %167 = getelementptr double, ptr %112, i64 %166
  %168 = getelementptr i8, ptr %167, i64 16
  %169 = load <2 x double>, ptr %167, align 8, !tbaa !10, !alias.scope !82
  %170 = load <2 x double>, ptr %168, align 8, !tbaa !10, !alias.scope !82
  %171 = getelementptr double, ptr %156, i64 %166
  %172 = getelementptr i8, ptr %171, i64 16
  %173 = load <2 x double>, ptr %171, align 8, !tbaa !10, !alias.scope !85, !noalias !82
  %174 = load <2 x double>, ptr %172, align 8, !tbaa !10, !alias.scope !85, !noalias !82
  %175 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %164, <2 x double> %169, <2 x double> %173)
  %176 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %164, <2 x double> %170, <2 x double> %174)
  store <2 x double> %175, ptr %171, align 8, !tbaa !10, !alias.scope !85, !noalias !82
  store <2 x double> %176, ptr %172, align 8, !tbaa !10, !alias.scope !85, !noalias !82
  %177 = add nuw i64 %166, 4
  %178 = icmp eq i64 %177, %142
  br i1 %178, label %179, label %165, !llvm.loop !87

179:                                              ; preds = %165
  br i1 %143, label %191, label %180

180:                                              ; preds = %157, %155, %179
  %181 = phi i64 [ 0, %157 ], [ 0, %155 ], [ %142, %179 ]
  br label %182

182:                                              ; preds = %180, %182
  %183 = phi i64 [ %189, %182 ], [ %181, %180 ]
  %184 = getelementptr double, ptr %112, i64 %183
  %185 = load double, ptr %184, align 8, !tbaa !10
  %186 = getelementptr double, ptr %156, i64 %183
  %187 = load double, ptr %186, align 8, !tbaa !10
  %188 = tail call double @llvm.fmuladd.f64(double %149, double %185, double %187)
  store double %188, ptr %186, align 8, !tbaa !10
  %189 = add nuw nsw i64 %183, 1
  %190 = icmp eq i64 %189, %111
  br i1 %190, label %191, label %182, !llvm.loop !88

191:                                              ; preds = %182, %179, %153
  %192 = add nuw nsw i64 %145, 1
  %193 = icmp eq i64 %192, 2000
  br i1 %193, label %194, label %144, !llvm.loop !58

194:                                              ; preds = %191, %96
  %195 = add nuw nsw i64 %74, 1
  %196 = icmp eq i64 %97, 1999
  br i1 %196, label %197, label %72, !llvm.loop !59

197:                                              ; preds = %194
  %198 = getelementptr inbounds nuw i8, ptr %21, i64 7996
  store i32 1999, ptr %198, align 4, !tbaa !47
  br label %199

199:                                              ; preds = %253, %197
  %200 = phi i64 [ 0, %197 ], [ %213, %253 ]
  %201 = sub nsw i64 1999, %200
  %202 = getelementptr inbounds nuw i32, ptr %21, i64 %200
  %203 = load i32, ptr %202, align 4, !tbaa !47
  %204 = sext i32 %203 to i64
  %205 = getelementptr inbounds double, ptr %20, i64 %204
  %206 = load double, ptr %205, align 8, !tbaa !10
  %207 = zext i32 %203 to i64
  %208 = icmp eq i64 %200, %207
  br i1 %208, label %212, label %209

209:                                              ; preds = %199
  %210 = getelementptr inbounds nuw double, ptr %20, i64 %200
  %211 = load double, ptr %210, align 8, !tbaa !10
  store double %211, ptr %205, align 8, !tbaa !10
  store double %206, ptr %210, align 8, !tbaa !10
  br label %212

212:                                              ; preds = %209, %199
  %213 = add nuw nsw i64 %200, 1
  %214 = fcmp une double %206, 0.000000e+00
  br i1 %214, label %215, label %253

215:                                              ; preds = %212
  %216 = getelementptr inbounds nuw ptr, ptr %12, i64 %200
  %217 = load ptr, ptr %216, align 8, !tbaa !12
  %218 = sub nuw nsw i64 1999, %200
  %219 = getelementptr double, ptr %217, i64 %213
  %220 = getelementptr double, ptr %20, i64 %213
  %221 = icmp ult i64 %201, 4
  br i1 %221, label %242, label %222

222:                                              ; preds = %215
  %223 = and i64 %201, -4
  %224 = insertelement <2 x double> poison, double %206, i64 0
  %225 = shufflevector <2 x double> %224, <2 x double> poison, <2 x i32> zeroinitializer
  br label %226

226:                                              ; preds = %226, %222
  %227 = phi i64 [ 0, %222 ], [ %238, %226 ]
  %228 = getelementptr double, ptr %219, i64 %227
  %229 = getelementptr i8, ptr %228, i64 16
  %230 = load <2 x double>, ptr %228, align 8, !tbaa !10
  %231 = load <2 x double>, ptr %229, align 8, !tbaa !10
  %232 = getelementptr double, ptr %220, i64 %227
  %233 = getelementptr i8, ptr %232, i64 16
  %234 = load <2 x double>, ptr %232, align 8, !tbaa !10
  %235 = load <2 x double>, ptr %233, align 8, !tbaa !10
  %236 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %225, <2 x double> %230, <2 x double> %234)
  %237 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %225, <2 x double> %231, <2 x double> %235)
  store <2 x double> %236, ptr %232, align 8, !tbaa !10
  store <2 x double> %237, ptr %233, align 8, !tbaa !10
  %238 = add nuw i64 %227, 4
  %239 = icmp eq i64 %238, %223
  br i1 %239, label %240, label %226, !llvm.loop !89

240:                                              ; preds = %226
  %241 = icmp eq i64 %201, %223
  br i1 %241, label %253, label %242

242:                                              ; preds = %215, %240
  %243 = phi i64 [ 0, %215 ], [ %223, %240 ]
  br label %244

244:                                              ; preds = %242, %244
  %245 = phi i64 [ %251, %244 ], [ %243, %242 ]
  %246 = getelementptr double, ptr %219, i64 %245
  %247 = load double, ptr %246, align 8, !tbaa !10
  %248 = getelementptr double, ptr %220, i64 %245
  %249 = load double, ptr %248, align 8, !tbaa !10
  %250 = tail call double @llvm.fmuladd.f64(double %206, double %247, double %249)
  store double %250, ptr %248, align 8, !tbaa !10
  %251 = add nuw nsw i64 %245, 1
  %252 = icmp eq i64 %251, %218
  br i1 %252, label %253, label %244, !llvm.loop !90

253:                                              ; preds = %244, %240, %212
  %254 = icmp eq i64 %213, 1999
  br i1 %254, label %255, label %199, !llvm.loop !67

255:                                              ; preds = %253, %304
  %256 = phi i64 [ %258, %304 ], [ 0, %253 ]
  %257 = sub nsw i64 1999, %256
  %258 = add nuw nsw i64 %256, 1
  %259 = sub nuw nsw i64 1999, %256
  %260 = getelementptr inbounds nuw ptr, ptr %12, i64 %259
  %261 = load ptr, ptr %260, align 8, !tbaa !12
  %262 = getelementptr inbounds nuw double, ptr %261, i64 %259
  %263 = load double, ptr %262, align 8, !tbaa !10
  %264 = getelementptr inbounds nuw double, ptr %20, i64 %259
  %265 = load double, ptr %264, align 8, !tbaa !10
  %266 = fdiv double %265, %263
  store double %266, ptr %264, align 8, !tbaa !10
  %267 = fneg double %266
  %268 = icmp ne i64 %256, 1999
  %269 = fcmp une double %266, 0.000000e+00
  %270 = and i1 %268, %269
  br i1 %270, label %271, label %304

271:                                              ; preds = %255
  %272 = icmp ult i64 %257, 4
  br i1 %272, label %293, label %273

273:                                              ; preds = %271
  %274 = and i64 %257, -4
  %275 = insertelement <2 x double> poison, double %267, i64 0
  %276 = shufflevector <2 x double> %275, <2 x double> poison, <2 x i32> zeroinitializer
  br label %277

277:                                              ; preds = %277, %273
  %278 = phi i64 [ 0, %273 ], [ %289, %277 ]
  %279 = getelementptr double, ptr %261, i64 %278
  %280 = getelementptr i8, ptr %279, i64 16
  %281 = load <2 x double>, ptr %279, align 8, !tbaa !10
  %282 = load <2 x double>, ptr %280, align 8, !tbaa !10
  %283 = getelementptr double, ptr %20, i64 %278
  %284 = getelementptr i8, ptr %283, i64 16
  %285 = load <2 x double>, ptr %283, align 8, !tbaa !10
  %286 = load <2 x double>, ptr %284, align 8, !tbaa !10
  %287 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %276, <2 x double> %281, <2 x double> %285)
  %288 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %276, <2 x double> %282, <2 x double> %286)
  store <2 x double> %287, ptr %283, align 8, !tbaa !10
  store <2 x double> %288, ptr %284, align 8, !tbaa !10
  %289 = add nuw i64 %278, 4
  %290 = icmp eq i64 %289, %274
  br i1 %290, label %291, label %277, !llvm.loop !91

291:                                              ; preds = %277
  %292 = icmp eq i64 %257, %274
  br i1 %292, label %304, label %293

293:                                              ; preds = %271, %291
  %294 = phi i64 [ 0, %271 ], [ %274, %291 ]
  br label %295

295:                                              ; preds = %293, %295
  %296 = phi i64 [ %302, %295 ], [ %294, %293 ]
  %297 = getelementptr double, ptr %261, i64 %296
  %298 = load double, ptr %297, align 8, !tbaa !10
  %299 = getelementptr double, ptr %20, i64 %296
  %300 = load double, ptr %299, align 8, !tbaa !10
  %301 = tail call double @llvm.fmuladd.f64(double %267, double %298, double %300)
  store double %301, ptr %299, align 8, !tbaa !10
  %302 = add nuw nsw i64 %296, 1
  %303 = icmp eq i64 %302, %259
  br i1 %303, label %304, label %295, !llvm.loop !92

304:                                              ; preds = %295, %291, %255
  %305 = icmp eq i64 %258, 2000
  br i1 %305, label %306, label %255, !llvm.loop !75

306:                                              ; preds = %304
  tail call void @free(ptr noundef %21) #14
  tail call void @free(ptr noundef nonnull %20) #14
  br label %307

307:                                              ; preds = %306, %307
  %308 = phi i64 [ 0, %306 ], [ %311, %307 ]
  %309 = getelementptr inbounds nuw ptr, ptr %12, i64 %308
  %310 = load ptr, ptr %309, align 8, !tbaa !12
  tail call void @free(ptr noundef %310) #14
  %311 = add nuw nsw i64 %308, 1
  %312 = icmp eq i64 %311, 2000
  br i1 %312, label %313, label %307, !llvm.loop !93

313:                                              ; preds = %307
  tail call void @free(ptr noundef nonnull %12) #14
  %314 = load ptr, ptr @stdout, align 8, !tbaa !94
  %315 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %314, ptr noundef nonnull %11, double noundef 0.000000e+00) #14
  %316 = load ptr, ptr @stdout, align 8, !tbaa !94
  %317 = tail call i32 @fflush(ptr noundef %316)
  ret i32 0
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: read)
declare i32 @strcmp(ptr noundef captures(none), ptr noundef captures(none)) local_unnamed_addr #6

; Function Attrs: mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite)
declare noalias noundef ptr @malloc(i64 noundef) local_unnamed_addr #7

; Function Attrs: mustprogress nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite)
declare void @free(ptr allocptr noundef captures(none)) local_unnamed_addr #8

; Function Attrs: nofree nounwind
declare noundef i32 @fprintf(ptr noundef captures(none), ptr noundef readonly captures(none), ...) local_unnamed_addr #9

; Function Attrs: nofree nounwind
declare noundef i32 @fflush(ptr noundef captures(none)) local_unnamed_addr #9

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #10

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <2 x double> @llvm.fmuladd.v2f64(<2 x double>, <2 x double>, <2 x double>) #11

attributes #0 = { nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree norecurse nosync nounwind memory(argmem: read) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nofree norecurse nosync nounwind memory(read, argmem: readwrite, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: read) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite) "alloc-family"="malloc" "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #8 = { mustprogress nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite) "alloc-family"="malloc" "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #9 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #10 = { nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #11 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #12 = { nounwind willreturn memory(read) }
attributes #13 = { nounwind allocsize(0) }
attributes #14 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"long", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!11, !11, i64 0}
!11 = !{!"double", !8, i64 0}
!12 = !{!13, !13, i64 0}
!13 = !{!"p1 double", !14, i64 0}
!14 = !{!"any pointer", !8, i64 0}
!15 = distinct !{!15, !16}
!16 = !{!"llvm.loop.mustprogress"}
!17 = distinct !{!17, !16}
!18 = !{!19}
!19 = distinct !{!19, !20}
!20 = distinct !{!20, !"LVerDomain"}
!21 = !{!22}
!22 = distinct !{!22, !20}
!23 = distinct !{!23, !16, !24, !25}
!24 = !{!"llvm.loop.isvectorized", i32 1}
!25 = !{!"llvm.loop.unroll.runtime.disable"}
!26 = distinct !{!26, !16, !24}
!27 = distinct !{!27, !16}
!28 = distinct !{!28, !16}
!29 = distinct !{!29, !16}
!30 = distinct !{!30, !16, !24, !25}
!31 = distinct !{!31, !16, !24}
!32 = distinct !{!32, !16, !25, !24}
!33 = !{!34}
!34 = distinct !{!34, !35}
!35 = distinct !{!35, !"LVerDomain"}
!36 = !{!37}
!37 = distinct !{!37, !35}
!38 = distinct !{!38, !16, !24, !25}
!39 = !{!40}
!40 = distinct !{!40, !41}
!41 = distinct !{!41, !"LVerDomain"}
!42 = !{!43}
!43 = distinct !{!43, !41}
!44 = distinct !{!44, !16, !24, !25}
!45 = distinct !{!45, !16, !24}
!46 = distinct !{!46, !16, !24}
!47 = !{!48, !48, i64 0}
!48 = !{!"int", !8, i64 0}
!49 = distinct !{!49, !16, !24, !25}
!50 = distinct !{!50, !16, !25, !24}
!51 = !{!52}
!52 = distinct !{!52, !53}
!53 = distinct !{!53, !"LVerDomain"}
!54 = !{!55}
!55 = distinct !{!55, !53}
!56 = distinct !{!56, !16, !24, !25}
!57 = distinct !{!57, !16, !24}
!58 = distinct !{!58, !16}
!59 = distinct !{!59, !16}
!60 = !{!61}
!61 = distinct !{!61, !62}
!62 = distinct !{!62, !"LVerDomain"}
!63 = !{!64}
!64 = distinct !{!64, !62}
!65 = distinct !{!65, !16, !24, !25}
!66 = distinct !{!66, !16, !24}
!67 = distinct !{!67, !16}
!68 = !{!69}
!69 = distinct !{!69, !70}
!70 = distinct !{!70, !"LVerDomain"}
!71 = !{!72}
!72 = distinct !{!72, !70}
!73 = distinct !{!73, !16, !24, !25}
!74 = distinct !{!74, !16, !24}
!75 = distinct !{!75, !16}
!76 = !{!77, !77, i64 0}
!77 = !{!"p1 omnipotent char", !14, i64 0}
!78 = distinct !{!78, !16}
!79 = distinct !{!79, !16, !24, !25}
!80 = distinct !{!80, !16, !24, !25}
!81 = distinct !{!81, !16, !25, !24}
!82 = !{!83}
!83 = distinct !{!83, !84}
!84 = distinct !{!84, !"LVerDomain"}
!85 = !{!86}
!86 = distinct !{!86, !84}
!87 = distinct !{!87, !16, !24, !25}
!88 = distinct !{!88, !16, !24}
!89 = distinct !{!89, !16, !24, !25}
!90 = distinct !{!90, !16, !25, !24}
!91 = distinct !{!91, !16, !24, !25}
!92 = distinct !{!92, !16, !25, !24}
!93 = distinct !{!93, !16}
!94 = !{!95, !95, i64 0}
!95 = !{!"p1 _ZTS8_IO_FILE", !14, i64 0}
