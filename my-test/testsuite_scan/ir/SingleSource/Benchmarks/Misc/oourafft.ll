; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Misc/oourafft.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Misc/oourafft.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.timeval = type { i64, i64 }

@.str = private unnamed_addr constant [45 x i8] c"FFT sanity check failed! Difference is: %le\0A\00", align 1
@.str.1 = private unnamed_addr constant [7 x i8] c"%e %e\0A\00", align 1

; Function Attrs: nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = alloca %struct.timeval, align 8
  %2 = alloca %struct.timeval, align 8
  %3 = alloca %struct.timeval, align 8
  %4 = alloca %struct.timeval, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #17
  %5 = call i32 @gettimeofday(ptr noundef nonnull %4, ptr noundef null) #17
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #17
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #17
  %6 = call i32 @gettimeofday(ptr noundef nonnull %3, ptr noundef null) #17
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #17
  %7 = tail call noalias align 16 dereferenceable_or_null(128) ptr @memalign(i64 noundef 16, i64 noundef 128) #18
  %8 = tail call noalias align 16 dereferenceable_or_null(20480) ptr @memalign(i64 noundef 16, i64 noundef 20480) #18
  store <2 x double> <double 1.000000e+00, double 0.000000e+00>, ptr %8, align 16, !tbaa !6
  %9 = getelementptr inbounds nuw i8, ptr %8, i64 2048
  store <2 x double> splat (double 0x3FE6A09E667F3BCD), ptr %9, align 16, !tbaa !6
  br label %10

10:                                               ; preds = %10, %0
  %11 = phi i64 [ 2, %0 ], [ %22, %10 ]
  %12 = trunc nuw nsw i64 %11 to i32
  %13 = uitofp nneg i32 %12 to double
  %14 = fmul double %13, 0x3F6921FB54442D18
  %15 = tail call double @cos(double noundef %14) #17, !tbaa !10
  %16 = tail call double @sin(double noundef %14) #17, !tbaa !10
  %17 = getelementptr inbounds nuw double, ptr %8, i64 %11
  store double %15, ptr %17, align 16, !tbaa !6
  %18 = getelementptr inbounds nuw i8, ptr %17, i64 8
  store double %16, ptr %18, align 8, !tbaa !6
  %19 = sub nuw nsw i64 512, %11
  %20 = getelementptr inbounds nuw double, ptr %8, i64 %19
  store double %16, ptr %20, align 16, !tbaa !6
  %21 = getelementptr i8, ptr %20, i64 8
  store double %15, ptr %21, align 8, !tbaa !6
  %22 = add nuw nsw i64 %11, 2
  %23 = icmp samesign ult i64 %11, 254
  br i1 %23, label %10, label %24, !llvm.loop !12

24:                                               ; preds = %10
  tail call fastcc void @bitrv2(i32 noundef 512, ptr noundef %7, ptr noundef nonnull %8)
  %25 = tail call noalias align 16 dereferenceable_or_null(16384) ptr @memalign(i64 noundef 16, i64 noundef 16384) #18
  %26 = tail call noalias align 16 dereferenceable_or_null(16384) ptr @memalign(i64 noundef 16, i64 noundef 16384) #18
  %27 = tail call noalias align 16 dereferenceable_or_null(16384) ptr @memalign(i64 noundef 16, i64 noundef 16384) #18
  br label %28

28:                                               ; preds = %28, %24
  %29 = phi i64 [ 0, %24 ], [ %37, %28 ]
  %30 = phi i32 [ 0, %24 ], [ %33, %28 ]
  %31 = mul nuw nsw i32 %30, 7141
  %32 = add nuw nsw i32 %31, 54773
  %33 = urem i32 %32, 259200
  %34 = uitofp nneg i32 %33 to double
  %35 = fmul double %34, 0x3ED02E85C0898B71
  %36 = getelementptr inbounds nuw double, ptr %25, i64 %29
  store double %35, ptr %36, align 8, !tbaa !6
  %37 = add nuw nsw i64 %29, 1
  %38 = and i64 %37, 4294967295
  %39 = icmp eq i64 %38, 2048
  br i1 %39, label %40, label %28, !llvm.loop !14

40:                                               ; preds = %28
  tail call fastcc void @bitrv2(i32 noundef 2048, ptr noundef %7, ptr noundef nonnull %25)
  tail call fastcc void @cftfsub(i32 noundef 2048, ptr noundef nonnull %25, ptr noundef nonnull readonly %8)
  tail call void @cdft(i32 noundef 2048, i32 noundef -1, ptr noundef nonnull %25, ptr noundef %7, ptr noundef nonnull %8)
  br label %41

41:                                               ; preds = %41, %40
  %42 = phi i64 [ 0, %40 ], [ %56, %41 ]
  %43 = phi double [ 0.000000e+00, %40 ], [ %55, %41 ]
  %44 = phi i32 [ 0, %40 ], [ %47, %41 ]
  %45 = mul nuw nsw i32 %44, 7141
  %46 = add nuw nsw i32 %45, 54773
  %47 = urem i32 %46, 259200
  %48 = uitofp nneg i32 %47 to double
  %49 = getelementptr inbounds nuw double, ptr %25, i64 %42
  %50 = load double, ptr %49, align 8, !tbaa !6
  %51 = fmul double %50, 0xBF50000000000000
  %52 = tail call double @llvm.fmuladd.f64(double %48, double 0x3ED02E85C0898B71, double %51)
  %53 = tail call double @llvm.fabs.f64(double %52)
  %54 = fcmp ogt double %43, %53
  %55 = select i1 %54, double %43, double %53
  %56 = add nuw nsw i64 %42, 1
  %57 = and i64 %56, 4294967295
  %58 = icmp eq i64 %57, 2048
  br i1 %58, label %59, label %41, !llvm.loop !15

59:                                               ; preds = %41
  %60 = fcmp ogt double %55, 1.000000e-10
  br i1 %60, label %61, label %63

61:                                               ; preds = %59
  %62 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, double noundef %55)
  tail call void @abort() #19
  unreachable

63:                                               ; preds = %59
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 16 dereferenceable(16384) %25, i8 0, i64 16384, i1 false)
  br label %64

64:                                               ; preds = %64, %63
  %65 = phi i64 [ 0, %63 ], [ %73, %64 ]
  %66 = phi i32 [ 0, %63 ], [ %69, %64 ]
  %67 = mul nuw nsw i32 %66, 7141
  %68 = add nuw nsw i32 %67, 54773
  %69 = urem i32 %68, 259200
  %70 = uitofp nneg i32 %69 to double
  %71 = fmul double %70, 0x3ED02E85C0898B71
  %72 = getelementptr inbounds nuw double, ptr %25, i64 %65
  store double %71, ptr %72, align 8, !tbaa !6
  %73 = add nuw nsw i64 %65, 1
  %74 = and i64 %73, 4294967295
  %75 = icmp eq i64 %74, 1024
  br i1 %75, label %76, label %64, !llvm.loop !14

76:                                               ; preds = %64
  tail call fastcc void @bitrv2(i32 noundef 2048, ptr noundef %7, ptr noundef nonnull %25)
  tail call fastcc void @cftfsub(i32 noundef 2048, ptr noundef nonnull %25, ptr noundef nonnull readonly %8)
  br label %77

77:                                               ; preds = %77, %76
  %78 = phi i64 [ 0, %76 ], [ %89, %77 ]
  %79 = shl nuw nsw i64 %78, 4
  %80 = shl i64 %78, 4
  %81 = getelementptr inbounds nuw i8, ptr %25, i64 %79
  %82 = getelementptr inbounds nuw i8, ptr %25, i64 %80
  %83 = getelementptr inbounds nuw i8, ptr %81, i64 8
  %84 = getelementptr inbounds nuw i8, ptr %82, i64 24
  %85 = load double, ptr %83, align 8, !tbaa !6
  %86 = load double, ptr %84, align 8, !tbaa !6
  %87 = fneg double %85
  %88 = fneg double %86
  store double %87, ptr %83, align 8, !tbaa !6
  store double %88, ptr %84, align 8, !tbaa !6
  %89 = add nuw i64 %78, 2
  %90 = icmp eq i64 %89, 1024
  br i1 %90, label %91, label %77, !llvm.loop !16

91:                                               ; preds = %77
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 16 dereferenceable(16384) %27, i8 0, i64 16384, i1 false)
  br label %92

92:                                               ; preds = %92, %91
  %93 = phi i64 [ 0, %91 ], [ %101, %92 ]
  %94 = phi i32 [ 0, %91 ], [ %97, %92 ]
  %95 = mul nuw nsw i32 %94, 7141
  %96 = add nuw nsw i32 %95, 54773
  %97 = urem i32 %96, 259200
  %98 = uitofp nneg i32 %97 to double
  %99 = fmul double %98, 0x3ED02E85C0898B71
  %100 = getelementptr inbounds nuw double, ptr %27, i64 %93
  store double %99, ptr %100, align 8, !tbaa !6
  %101 = add nuw nsw i64 %93, 1
  %102 = and i64 %101, 4294967295
  %103 = icmp eq i64 %102, 1024
  br i1 %103, label %104, label %92, !llvm.loop !14

104:                                              ; preds = %92
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #17
  %105 = call i32 @gettimeofday(ptr noundef nonnull %2, ptr noundef null) #17
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #17
  br label %106

106:                                              ; preds = %104, %127
  %107 = phi i32 [ 0, %104 ], [ %128, %127 ]
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 16 dereferenceable(16384) %26, ptr noundef nonnull align 16 dereferenceable(16384) %27, i64 16384, i1 false)
  tail call fastcc void @bitrv2(i32 noundef 2048, ptr noundef %7, ptr noundef nonnull %26)
  tail call fastcc void @cftfsub(i32 noundef 2048, ptr noundef nonnull %26, ptr noundef nonnull readonly %8)
  br label %108

108:                                              ; preds = %108, %106
  %109 = phi i64 [ 0, %106 ], [ %125, %108 ]
  %110 = shl nuw nsw i64 %109, 1
  %111 = getelementptr inbounds nuw double, ptr %26, i64 %110
  %112 = load <4 x double>, ptr %111, align 16, !tbaa !6
  %113 = shufflevector <4 x double> %112, <4 x double> poison, <2 x i32> <i32 0, i32 2>
  %114 = shufflevector <4 x double> %112, <4 x double> poison, <2 x i32> <i32 1, i32 3>
  %115 = getelementptr inbounds nuw double, ptr %25, i64 %110
  %116 = load <4 x double>, ptr %115, align 16, !tbaa !6
  %117 = shufflevector <4 x double> %116, <4 x double> poison, <2 x i32> <i32 0, i32 2>
  %118 = shufflevector <4 x double> %116, <4 x double> poison, <2 x i32> <i32 1, i32 3>
  %119 = fneg <2 x double> %118
  %120 = fmul <2 x double> %114, %119
  %121 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %113, <2 x double> %117, <2 x double> %120)
  %122 = fmul <2 x double> %117, %114
  %123 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %113, <2 x double> %118, <2 x double> %122)
  %124 = shufflevector <2 x double> %121, <2 x double> %123, <4 x i32> <i32 0, i32 2, i32 1, i32 3>
  store <4 x double> %124, ptr %111, align 16, !tbaa !6
  %125 = add nuw i64 %109, 2
  %126 = icmp eq i64 %125, 1024
  br i1 %126, label %127, label %108, !llvm.loop !19

127:                                              ; preds = %108
  tail call void @cdft(i32 noundef 2048, i32 noundef -1, ptr noundef nonnull %26, ptr noundef %7, ptr noundef nonnull %8)
  %128 = add nuw nsw i32 %107, 1
  %129 = icmp eq i32 %128, 150000
  br i1 %129, label %130, label %106, !llvm.loop !20

130:                                              ; preds = %127
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #17
  %131 = call i32 @gettimeofday(ptr noundef nonnull %1, ptr noundef null) #17
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #17
  br label %132

132:                                              ; preds = %130, %132
  %133 = phi i64 [ 0, %130 ], [ %146, %132 ]
  %134 = shl nuw nsw i64 %133, 4
  %135 = getelementptr inbounds nuw i8, ptr %26, i64 %134
  %136 = load double, ptr %135, align 16, !tbaa !6
  %137 = tail call double @llvm.fabs.f64(double %136)
  %138 = fcmp ogt double %137, 1.000000e-09
  %139 = select i1 %138, double %136, double 0.000000e+00
  %140 = getelementptr inbounds nuw i8, ptr %135, i64 8
  %141 = load double, ptr %140, align 8, !tbaa !6
  %142 = tail call double @llvm.fabs.f64(double %141)
  %143 = fcmp ogt double %142, 1.000000e-09
  %144 = select i1 %143, double %141, double 0.000000e+00
  %145 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, double noundef %139, double noundef %144)
  %146 = add nuw nsw i64 %133, 1
  %147 = icmp eq i64 %146, 1024
  br i1 %147, label %148, label %132, !llvm.loop !21

148:                                              ; preds = %132
  tail call void @free(ptr noundef nonnull %25) #17
  tail call void @free(ptr noundef nonnull %8) #17
  tail call void @free(ptr noundef %7) #17
  tail call void @free(ptr noundef nonnull %26) #17
  tail call void @free(ptr noundef nonnull %27) #17
  ret i32 0
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: nofree nounwind uwtable
define dso_local double @get_time() local_unnamed_addr #2 {
  %1 = alloca %struct.timeval, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #17
  %2 = call i32 @gettimeofday(ptr noundef nonnull %1, ptr noundef null) #17
  %3 = load i64, ptr %1, align 8, !tbaa !22
  %4 = sitofp i64 %3 to double
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %6 = load i64, ptr %5, align 8, !tbaa !25
  %7 = sitofp i64 %6 to double
  %8 = tail call double @llvm.fmuladd.f64(double %7, double 0x3EB0C6F7A0B5ED8D, double %4)
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #17
  ret double %8
}

; Function Attrs: mustprogress nofree nounwind willreturn allockind("alloc,uninitialized,aligned") allocsize(1) memory(inaccessiblemem: readwrite)
declare noalias noundef ptr @memalign(i64 allocalign noundef, i64 noundef) local_unnamed_addr #3

; Function Attrs: nofree norecurse nounwind memory(argmem: readwrite, errnomem: write) uwtable
define dso_local void @makewt(i32 noundef %0, ptr noundef captures(none) %1, ptr noundef captures(none) %2) local_unnamed_addr #4 {
  %4 = icmp sgt i32 %0, 2
  br i1 %4, label %5, label %33

5:                                                ; preds = %3
  %6 = lshr i32 %0, 1
  %7 = uitofp nneg i32 %6 to double
  %8 = fdiv double 0x3FE921FB54442D18, %7
  store <2 x double> <double 1.000000e+00, double 0.000000e+00>, ptr %2, align 8, !tbaa !6
  %9 = fmul double %8, %7
  %10 = tail call double @cos(double noundef %9) #17, !tbaa !10
  %11 = zext nneg i32 %6 to i64
  %12 = getelementptr inbounds nuw double, ptr %2, i64 %11
  store double %10, ptr %12, align 8, !tbaa !6
  %13 = getelementptr inbounds nuw i8, ptr %12, i64 8
  store double %10, ptr %13, align 8, !tbaa !6
  %14 = icmp samesign ugt i32 %0, 5
  br i1 %14, label %15, label %33

15:                                               ; preds = %5
  %16 = zext nneg i32 %0 to i64
  %17 = zext nneg i32 %6 to i64
  br label %18

18:                                               ; preds = %15, %18
  %19 = phi i64 [ 2, %15 ], [ %30, %18 ]
  %20 = trunc nuw nsw i64 %19 to i32
  %21 = uitofp nneg i32 %20 to double
  %22 = fmul double %8, %21
  %23 = tail call double @cos(double noundef %22) #17, !tbaa !10
  %24 = tail call double @sin(double noundef %22) #17, !tbaa !10
  %25 = getelementptr inbounds nuw double, ptr %2, i64 %19
  store double %23, ptr %25, align 8, !tbaa !6
  %26 = getelementptr inbounds nuw i8, ptr %25, i64 8
  store double %24, ptr %26, align 8, !tbaa !6
  %27 = sub nsw i64 %16, %19
  %28 = getelementptr inbounds double, ptr %2, i64 %27
  store double %24, ptr %28, align 8, !tbaa !6
  %29 = getelementptr i8, ptr %28, i64 8
  store double %23, ptr %29, align 8, !tbaa !6
  %30 = add nuw nsw i64 %19, 2
  %31 = icmp samesign ult i64 %30, %17
  br i1 %31, label %18, label %32, !llvm.loop !12

32:                                               ; preds = %18
  tail call fastcc void @bitrv2(i32 noundef %0, ptr noundef %1, ptr noundef nonnull %2)
  br label %33

33:                                               ; preds = %5, %32, %3
  ret void
}

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: write) uwtable
define dso_local void @putdata(i32 noundef %0, i32 noundef %1, ptr noundef writeonly captures(none) %2) local_unnamed_addr #5 {
  %4 = icmp sgt i32 %0, %1
  br i1 %4, label %20, label %5

5:                                                ; preds = %3
  %6 = sext i32 %0 to i64
  %7 = add i32 %1, 1
  br label %8

8:                                                ; preds = %5, %8
  %9 = phi i64 [ %6, %5 ], [ %17, %8 ]
  %10 = phi i32 [ 0, %5 ], [ %13, %8 ]
  %11 = mul nuw nsw i32 %10, 7141
  %12 = add nuw nsw i32 %11, 54773
  %13 = urem i32 %12, 259200
  %14 = uitofp nneg i32 %13 to double
  %15 = fmul double %14, 0x3ED02E85C0898B71
  %16 = getelementptr inbounds double, ptr %2, i64 %9
  store double %15, ptr %16, align 8, !tbaa !6
  %17 = add nsw i64 %9, 1
  %18 = trunc i64 %17 to i32
  %19 = icmp eq i32 %7, %18
  br i1 %19, label %20, label %8, !llvm.loop !14

20:                                               ; preds = %8, %3
  ret void
}

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable
define dso_local void @cdft(i32 noundef %0, i32 noundef %1, ptr noundef %2, ptr noundef captures(none) %3, ptr noundef readonly captures(none) %4) local_unnamed_addr #6 {
  %6 = icmp sgt i32 %0, 4
  br i1 %6, label %7, label %580

7:                                                ; preds = %5
  %8 = icmp sgt i32 %1, -1
  br i1 %8, label %9, label %10

9:                                                ; preds = %7
  tail call fastcc void @bitrv2(i32 noundef %0, ptr noundef %3, ptr noundef %2)
  tail call fastcc void @cftfsub(i32 noundef %0, ptr noundef %2, ptr noundef %4)
  br label %591

10:                                               ; preds = %7
  store i32 0, ptr %3, align 4, !tbaa !10
  %11 = icmp samesign ugt i32 %0, 8
  br i1 %11, label %12, label %56

12:                                               ; preds = %10, %49
  %13 = phi i32 [ %50, %49 ], [ 1, %10 ]
  %14 = phi i32 [ %15, %49 ], [ %0, %10 ]
  %15 = lshr i32 %14, 1
  %16 = icmp sgt i32 %13, 0
  br i1 %16, label %17, label %49

17:                                               ; preds = %12
  %18 = zext nneg i32 %13 to i64
  %19 = getelementptr inbounds nuw i32, ptr %3, i64 %18
  %20 = icmp ult i32 %13, 8
  br i1 %20, label %39, label %21

21:                                               ; preds = %17
  %22 = and i64 %18, 2147483640
  %23 = insertelement <4 x i32> poison, i32 %15, i64 0
  %24 = shufflevector <4 x i32> %23, <4 x i32> poison, <4 x i32> zeroinitializer
  br label %25

25:                                               ; preds = %25, %21
  %26 = phi i64 [ 0, %21 ], [ %35, %25 ]
  %27 = getelementptr inbounds nuw i32, ptr %3, i64 %26
  %28 = getelementptr inbounds nuw i8, ptr %27, i64 16
  %29 = load <4 x i32>, ptr %27, align 4, !tbaa !10
  %30 = load <4 x i32>, ptr %28, align 4, !tbaa !10
  %31 = add nsw <4 x i32> %29, %24
  %32 = add nsw <4 x i32> %30, %24
  %33 = getelementptr inbounds nuw i32, ptr %19, i64 %26
  %34 = getelementptr inbounds nuw i8, ptr %33, i64 16
  store <4 x i32> %31, ptr %33, align 4, !tbaa !10
  store <4 x i32> %32, ptr %34, align 4, !tbaa !10
  %35 = add nuw i64 %26, 8
  %36 = icmp eq i64 %35, %22
  br i1 %36, label %37, label %25, !llvm.loop !26

37:                                               ; preds = %25
  %38 = icmp eq i64 %22, %18
  br i1 %38, label %49, label %39

39:                                               ; preds = %17, %37
  %40 = phi i64 [ 0, %17 ], [ %22, %37 ]
  br label %41

41:                                               ; preds = %39, %41
  %42 = phi i64 [ %47, %41 ], [ %40, %39 ]
  %43 = getelementptr inbounds nuw i32, ptr %3, i64 %42
  %44 = load i32, ptr %43, align 4, !tbaa !10
  %45 = add nsw i32 %44, %15
  %46 = getelementptr inbounds nuw i32, ptr %19, i64 %42
  store i32 %45, ptr %46, align 4, !tbaa !10
  %47 = add nuw nsw i64 %42, 1
  %48 = icmp eq i64 %47, %18
  br i1 %48, label %49, label %41, !llvm.loop !27

49:                                               ; preds = %41, %37, %12
  %50 = shl i32 %13, 1
  %51 = shl i32 %13, 4
  %52 = icmp slt i32 %51, %15
  br i1 %52, label %12, label %53, !llvm.loop !28

53:                                               ; preds = %49
  %54 = shl i32 %13, 2
  %55 = icmp eq i32 %51, %15
  br i1 %55, label %58, label %165

56:                                               ; preds = %10
  %57 = icmp eq i32 %0, 8
  br i1 %57, label %60, label %165

58:                                               ; preds = %53
  %59 = icmp sgt i32 %50, 0
  br i1 %59, label %60, label %233

60:                                               ; preds = %58, %56
  %61 = phi i32 [ %50, %58 ], [ 1, %56 ]
  %62 = phi i32 [ %54, %58 ], [ 2, %56 ]
  %63 = shl nsw i32 %61, 2
  %64 = zext nneg i32 %62 to i64
  %65 = zext nneg i32 %61 to i64
  %66 = getelementptr double, ptr %2, i64 %64
  br label %67

67:                                               ; preds = %136, %60
  %68 = phi i64 [ 0, %60 ], [ %163, %136 ]
  %69 = icmp eq i64 %68, 0
  br i1 %69, label %136, label %70

70:                                               ; preds = %67
  %71 = getelementptr inbounds nuw i32, ptr %3, i64 %68
  %72 = load i32, ptr %71, align 4, !tbaa !10
  %73 = sext i32 %72 to i64
  %74 = trunc i64 %68 to i32
  %75 = shl i32 %74, 1
  br label %76

76:                                               ; preds = %76, %70
  %77 = phi i64 [ 0, %70 ], [ %132, %76 ]
  %78 = shl nuw nsw i64 %77, 1
  %79 = add nsw i64 %78, %73
  %80 = getelementptr inbounds nuw i32, ptr %3, i64 %77
  %81 = load i32, ptr %80, align 4, !tbaa !10
  %82 = add nsw i32 %81, %75
  %83 = getelementptr inbounds double, ptr %2, i64 %79
  %84 = load double, ptr %83, align 8, !tbaa !6
  %85 = getelementptr i8, ptr %83, i64 8
  %86 = load double, ptr %85, align 8, !tbaa !6
  %87 = fneg double %86
  %88 = sext i32 %82 to i64
  %89 = getelementptr inbounds double, ptr %2, i64 %88
  %90 = load double, ptr %89, align 8, !tbaa !6
  %91 = getelementptr i8, ptr %89, i64 8
  %92 = load double, ptr %91, align 8, !tbaa !6
  %93 = fneg double %92
  store double %90, ptr %83, align 8, !tbaa !6
  store double %93, ptr %85, align 8, !tbaa !6
  store double %84, ptr %89, align 8, !tbaa !6
  store double %87, ptr %91, align 8, !tbaa !6
  %94 = add nsw i64 %79, %64
  %95 = add nsw i32 %82, %63
  %96 = getelementptr inbounds double, ptr %2, i64 %94
  %97 = load double, ptr %96, align 8, !tbaa !6
  %98 = getelementptr i8, ptr %96, i64 8
  %99 = load double, ptr %98, align 8, !tbaa !6
  %100 = fneg double %99
  %101 = sext i32 %95 to i64
  %102 = getelementptr inbounds double, ptr %2, i64 %101
  %103 = load double, ptr %102, align 8, !tbaa !6
  %104 = getelementptr i8, ptr %102, i64 8
  %105 = load double, ptr %104, align 8, !tbaa !6
  %106 = fneg double %105
  store double %103, ptr %96, align 8, !tbaa !6
  store double %106, ptr %98, align 8, !tbaa !6
  store double %97, ptr %102, align 8, !tbaa !6
  store double %100, ptr %104, align 8, !tbaa !6
  %107 = add nsw i64 %94, %64
  %108 = sub nsw i32 %95, %62
  %109 = getelementptr inbounds double, ptr %2, i64 %107
  %110 = load double, ptr %109, align 8, !tbaa !6
  %111 = getelementptr i8, ptr %109, i64 8
  %112 = load double, ptr %111, align 8, !tbaa !6
  %113 = fneg double %112
  %114 = sext i32 %108 to i64
  %115 = getelementptr inbounds double, ptr %2, i64 %114
  %116 = load double, ptr %115, align 8, !tbaa !6
  %117 = getelementptr i8, ptr %115, i64 8
  %118 = load double, ptr %117, align 8, !tbaa !6
  %119 = fneg double %118
  store double %116, ptr %109, align 8, !tbaa !6
  store double %119, ptr %111, align 8, !tbaa !6
  store double %110, ptr %115, align 8, !tbaa !6
  store double %113, ptr %117, align 8, !tbaa !6
  %120 = add nsw i32 %108, %63
  %121 = getelementptr double, ptr %66, i64 %107
  %122 = load double, ptr %121, align 8, !tbaa !6
  %123 = getelementptr i8, ptr %121, i64 8
  %124 = load double, ptr %123, align 8, !tbaa !6
  %125 = fneg double %124
  %126 = sext i32 %120 to i64
  %127 = getelementptr inbounds double, ptr %2, i64 %126
  %128 = load double, ptr %127, align 8, !tbaa !6
  %129 = getelementptr i8, ptr %127, i64 8
  %130 = load double, ptr %129, align 8, !tbaa !6
  %131 = fneg double %130
  store double %128, ptr %121, align 8, !tbaa !6
  store double %131, ptr %123, align 8, !tbaa !6
  store double %122, ptr %127, align 8, !tbaa !6
  store double %125, ptr %129, align 8, !tbaa !6
  %132 = add nuw nsw i64 %77, 1
  %133 = icmp eq i64 %132, %68
  br i1 %133, label %134, label %76, !llvm.loop !29

134:                                              ; preds = %76
  %135 = add nsw i32 %72, %75
  br label %136

136:                                              ; preds = %134, %67
  %137 = phi i32 [ %135, %134 ], [ 0, %67 ]
  %138 = sext i32 %137 to i64
  %139 = getelementptr double, ptr %2, i64 %138
  %140 = getelementptr i8, ptr %139, i64 8
  %141 = load double, ptr %140, align 8, !tbaa !6
  %142 = fneg double %141
  store double %142, ptr %140, align 8, !tbaa !6
  %143 = add nsw i32 %137, %62
  %144 = add nsw i32 %143, %62
  %145 = sext i32 %143 to i64
  %146 = getelementptr inbounds double, ptr %2, i64 %145
  %147 = load double, ptr %146, align 8, !tbaa !6
  %148 = getelementptr i8, ptr %146, i64 8
  %149 = load double, ptr %148, align 8, !tbaa !6
  %150 = fneg double %149
  %151 = sext i32 %144 to i64
  %152 = getelementptr inbounds double, ptr %2, i64 %151
  %153 = load double, ptr %152, align 8, !tbaa !6
  %154 = getelementptr i8, ptr %152, i64 8
  %155 = load double, ptr %154, align 8, !tbaa !6
  %156 = fneg double %155
  store double %153, ptr %146, align 8, !tbaa !6
  store double %156, ptr %148, align 8, !tbaa !6
  store double %147, ptr %152, align 8, !tbaa !6
  store double %150, ptr %154, align 8, !tbaa !6
  %157 = add nsw i32 %144, %62
  %158 = sext i32 %157 to i64
  %159 = getelementptr double, ptr %2, i64 %158
  %160 = getelementptr i8, ptr %159, i64 8
  %161 = load double, ptr %160, align 8, !tbaa !6
  %162 = fneg double %161
  store double %162, ptr %160, align 8, !tbaa !6
  %163 = add nuw nsw i64 %68, 1
  %164 = icmp eq i64 %163, %65
  br i1 %164, label %233, label %67, !llvm.loop !30

165:                                              ; preds = %56, %53
  %166 = phi i32 [ 2, %56 ], [ %54, %53 ]
  %167 = phi i32 [ 1, %56 ], [ %50, %53 ]
  %168 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %169 = load double, ptr %168, align 8, !tbaa !6
  %170 = fneg double %169
  store double %170, ptr %168, align 8, !tbaa !6
  %171 = sext i32 %166 to i64
  %172 = getelementptr double, ptr %2, i64 %171
  %173 = getelementptr i8, ptr %172, i64 8
  %174 = load double, ptr %173, align 8, !tbaa !6
  %175 = fneg double %174
  store double %175, ptr %173, align 8, !tbaa !6
  %176 = icmp sgt i32 %167, 1
  br i1 %176, label %177, label %233

177:                                              ; preds = %165
  %178 = zext nneg i32 %167 to i64
  br label %179

179:                                              ; preds = %218, %177
  %180 = phi i64 [ 1, %177 ], [ %231, %218 ]
  %181 = getelementptr inbounds nuw i32, ptr %3, i64 %180
  %182 = load i32, ptr %181, align 4, !tbaa !10
  %183 = sext i32 %182 to i64
  %184 = trunc i64 %180 to i32
  %185 = shl i32 %184, 1
  br label %186

186:                                              ; preds = %186, %179
  %187 = phi i64 [ 0, %179 ], [ %216, %186 ]
  %188 = shl nuw nsw i64 %187, 1
  %189 = add nsw i64 %188, %183
  %190 = getelementptr inbounds nuw i32, ptr %3, i64 %187
  %191 = load i32, ptr %190, align 4, !tbaa !10
  %192 = add nsw i32 %191, %185
  %193 = getelementptr inbounds double, ptr %2, i64 %189
  %194 = load double, ptr %193, align 8, !tbaa !6
  %195 = getelementptr i8, ptr %193, i64 8
  %196 = load double, ptr %195, align 8, !tbaa !6
  %197 = fneg double %196
  %198 = sext i32 %192 to i64
  %199 = getelementptr inbounds double, ptr %2, i64 %198
  %200 = load double, ptr %199, align 8, !tbaa !6
  %201 = getelementptr i8, ptr %199, i64 8
  %202 = load double, ptr %201, align 8, !tbaa !6
  %203 = fneg double %202
  store double %200, ptr %193, align 8, !tbaa !6
  store double %203, ptr %195, align 8, !tbaa !6
  store double %194, ptr %199, align 8, !tbaa !6
  store double %197, ptr %201, align 8, !tbaa !6
  %204 = add nsw i32 %192, %166
  %205 = getelementptr double, ptr %172, i64 %189
  %206 = load double, ptr %205, align 8, !tbaa !6
  %207 = getelementptr i8, ptr %205, i64 8
  %208 = load double, ptr %207, align 8, !tbaa !6
  %209 = fneg double %208
  %210 = sext i32 %204 to i64
  %211 = getelementptr inbounds double, ptr %2, i64 %210
  %212 = load double, ptr %211, align 8, !tbaa !6
  %213 = getelementptr i8, ptr %211, i64 8
  %214 = load double, ptr %213, align 8, !tbaa !6
  %215 = fneg double %214
  store double %212, ptr %205, align 8, !tbaa !6
  store double %215, ptr %207, align 8, !tbaa !6
  store double %206, ptr %211, align 8, !tbaa !6
  store double %209, ptr %213, align 8, !tbaa !6
  %216 = add nuw nsw i64 %187, 1
  %217 = icmp eq i64 %216, %180
  br i1 %217, label %218, label %186, !llvm.loop !31

218:                                              ; preds = %186
  %219 = add nsw i32 %185, %182
  %220 = sext i32 %219 to i64
  %221 = getelementptr double, ptr %2, i64 %220
  %222 = getelementptr i8, ptr %221, i64 8
  %223 = load double, ptr %222, align 8, !tbaa !6
  %224 = fneg double %223
  store double %224, ptr %222, align 8, !tbaa !6
  %225 = add nsw i32 %219, %166
  %226 = sext i32 %225 to i64
  %227 = getelementptr double, ptr %2, i64 %226
  %228 = getelementptr i8, ptr %227, i64 8
  %229 = load double, ptr %228, align 8, !tbaa !6
  %230 = fneg double %229
  store double %230, ptr %228, align 8, !tbaa !6
  %231 = add nuw nsw i64 %180, 1
  %232 = icmp eq i64 %231, %178
  br i1 %232, label %233, label %179, !llvm.loop !32

233:                                              ; preds = %218, %136, %58, %165
  br i1 %11, label %234, label %241

234:                                              ; preds = %233
  tail call fastcc void @cft1st(i32 noundef range(i32 5, -2147483648) %0, ptr noundef %2, ptr noundef readonly %4)
  %235 = icmp samesign ugt i32 %0, 32
  br i1 %235, label %236, label %241

236:                                              ; preds = %234, %236
  %237 = phi i32 [ %239, %236 ], [ 32, %234 ]
  %238 = phi i32 [ %237, %236 ], [ 8, %234 ]
  tail call fastcc void @cftmdl(i32 noundef range(i32 5, -2147483648) %0, i32 noundef %238, ptr noundef %2, ptr noundef readonly %4)
  %239 = shl i32 %237, 2
  %240 = icmp slt i32 %239, %0
  br i1 %240, label %236, label %241, !llvm.loop !33

241:                                              ; preds = %236, %234, %233
  %242 = phi i32 [ 2, %233 ], [ 8, %234 ], [ %237, %236 ]
  %243 = shl i32 %242, 2
  %244 = icmp eq i32 %243, %0
  %245 = icmp sgt i32 %242, 0
  br i1 %244, label %320, label %246

246:                                              ; preds = %241
  br i1 %245, label %247, label %591

247:                                              ; preds = %246
  %248 = zext nneg i32 %242 to i64
  %249 = getelementptr inbounds nuw double, ptr %2, i64 %248
  %250 = add nsw i64 %248, -1
  %251 = lshr i64 %250, 1
  %252 = add nuw i64 %251, 1
  %253 = icmp ult i32 %242, 31
  br i1 %253, label %318, label %254

254:                                              ; preds = %247
  %255 = shl nuw nsw i64 %248, 3
  %256 = add nsw i64 %255, -8
  %257 = and i64 %256, -16
  %258 = getelementptr i8, ptr %2, i64 %256
  %259 = getelementptr i8, ptr %2, i64 8
  %260 = getelementptr i8, ptr %2, i64 %257
  %261 = getelementptr i8, ptr %260, i64 16
  %262 = shl nuw nsw i64 %248, 3
  %263 = add nsw i64 %257, %262
  %264 = getelementptr i8, ptr %2, i64 %263
  %265 = getelementptr i8, ptr %264, i64 8
  %266 = getelementptr i8, ptr %2, i64 %262
  %267 = getelementptr i8, ptr %266, i64 8
  %268 = getelementptr i8, ptr %2, i64 %263
  %269 = getelementptr i8, ptr %268, i64 16
  %270 = icmp ult ptr %2, %261
  %271 = icmp ult ptr %259, %258
  %272 = and i1 %270, %271
  %273 = icmp ult ptr %2, %265
  %274 = icmp ult ptr %249, %258
  %275 = and i1 %273, %274
  %276 = or i1 %272, %275
  %277 = icmp ult ptr %2, %269
  %278 = icmp ult ptr %267, %258
  %279 = and i1 %277, %278
  %280 = or i1 %276, %279
  %281 = icmp ult ptr %259, %265
  %282 = icmp ult ptr %249, %261
  %283 = and i1 %281, %282
  %284 = or i1 %280, %283
  %285 = icmp ult ptr %259, %269
  %286 = icmp ult ptr %267, %261
  %287 = and i1 %285, %286
  %288 = or i1 %284, %287
  %289 = icmp ult ptr %249, %269
  %290 = icmp ult ptr %267, %265
  %291 = and i1 %289, %290
  %292 = or i1 %288, %291
  br i1 %292, label %318, label %293

293:                                              ; preds = %254
  %294 = and i64 %252, -2
  %295 = shl i64 %294, 1
  br label %296

296:                                              ; preds = %296, %293
  %297 = phi i64 [ 0, %293 ], [ %314, %296 ]
  %298 = shl i64 %297, 1
  %299 = getelementptr inbounds nuw double, ptr %2, i64 %298
  %300 = load <4 x double>, ptr %299, align 8, !tbaa !6
  %301 = shufflevector <4 x double> %300, <4 x double> poison, <2 x i32> <i32 0, i32 2>
  %302 = shufflevector <4 x double> %300, <4 x double> poison, <2 x i32> <i32 1, i32 3>
  %303 = getelementptr inbounds nuw double, ptr %249, i64 %298
  %304 = load <4 x double>, ptr %303, align 8, !tbaa !6
  %305 = shufflevector <4 x double> %304, <4 x double> poison, <2 x i32> <i32 0, i32 2>
  %306 = shufflevector <4 x double> %304, <4 x double> poison, <2 x i32> <i32 1, i32 3>
  %307 = fsub <2 x double> %301, %305
  %308 = fsub <2 x double> %306, %302
  %309 = fadd <2 x double> %301, %305
  %310 = fneg <2 x double> %302
  %311 = fsub <2 x double> %310, %306
  %312 = shufflevector <2 x double> %309, <2 x double> %311, <4 x i32> <i32 0, i32 2, i32 1, i32 3>
  store <4 x double> %312, ptr %299, align 8, !tbaa !6
  %313 = shufflevector <2 x double> %307, <2 x double> %308, <4 x i32> <i32 0, i32 2, i32 1, i32 3>
  store <4 x double> %313, ptr %303, align 8, !tbaa !6
  %314 = add nuw i64 %297, 2
  %315 = icmp eq i64 %314, %294
  br i1 %315, label %316, label %296, !llvm.loop !34

316:                                              ; preds = %296
  %317 = icmp eq i64 %252, %294
  br i1 %317, label %591, label %318

318:                                              ; preds = %254, %247, %316
  %319 = phi i64 [ 0, %254 ], [ 0, %247 ], [ %295, %316 ]
  br label %563

320:                                              ; preds = %241
  br i1 %245, label %321, label %591

321:                                              ; preds = %320
  %322 = zext nneg i32 %242 to i64
  %323 = getelementptr double, ptr %2, i64 %322
  %324 = add nsw i64 %322, -1
  %325 = lshr i64 %324, 1
  %326 = add nuw i64 %325, 1
  %327 = icmp ult i32 %242, 51
  br i1 %327, label %522, label %328

328:                                              ; preds = %321
  %329 = shl nuw nsw i64 %322, 3
  %330 = add nsw i64 %329, -8
  %331 = and i64 %330, -16
  %332 = getelementptr i8, ptr %2, i64 %330
  %333 = getelementptr i8, ptr %2, i64 8
  %334 = getelementptr i8, ptr %2, i64 %331
  %335 = getelementptr i8, ptr %334, i64 16
  %336 = shl nuw nsw i64 %322, 4
  %337 = getelementptr i8, ptr %2, i64 %336
  %338 = add nsw i64 %331, %336
  %339 = getelementptr i8, ptr %2, i64 %338
  %340 = getelementptr i8, ptr %339, i64 8
  %341 = getelementptr i8, ptr %2, i64 %336
  %342 = getelementptr i8, ptr %341, i64 8
  %343 = getelementptr i8, ptr %2, i64 %338
  %344 = getelementptr i8, ptr %343, i64 16
  %345 = shl nuw nsw i64 %322, 3
  %346 = add nsw i64 %331, %345
  %347 = getelementptr i8, ptr %2, i64 %346
  %348 = getelementptr i8, ptr %347, i64 8
  %349 = getelementptr i8, ptr %2, i64 %345
  %350 = getelementptr i8, ptr %349, i64 8
  %351 = getelementptr i8, ptr %2, i64 %346
  %352 = getelementptr i8, ptr %351, i64 16
  %353 = mul nuw nsw i64 %322, 24
  %354 = getelementptr i8, ptr %2, i64 %353
  %355 = add nsw i64 %353, %331
  %356 = getelementptr i8, ptr %2, i64 %355
  %357 = getelementptr i8, ptr %356, i64 8
  %358 = getelementptr i8, ptr %2, i64 %353
  %359 = getelementptr i8, ptr %358, i64 8
  %360 = getelementptr i8, ptr %2, i64 %355
  %361 = getelementptr i8, ptr %360, i64 16
  %362 = icmp ult ptr %2, %335
  %363 = icmp ult ptr %333, %332
  %364 = and i1 %362, %363
  %365 = icmp ult ptr %2, %340
  %366 = icmp ult ptr %337, %332
  %367 = and i1 %365, %366
  %368 = or i1 %364, %367
  %369 = icmp ult ptr %2, %344
  %370 = icmp ult ptr %342, %332
  %371 = and i1 %369, %370
  %372 = or i1 %368, %371
  %373 = icmp ult ptr %2, %348
  %374 = icmp ult ptr %323, %332
  %375 = and i1 %373, %374
  %376 = or i1 %372, %375
  %377 = icmp ult ptr %2, %352
  %378 = icmp ult ptr %350, %332
  %379 = and i1 %377, %378
  %380 = or i1 %376, %379
  %381 = icmp ult ptr %2, %357
  %382 = icmp ult ptr %354, %332
  %383 = and i1 %381, %382
  %384 = or i1 %380, %383
  %385 = icmp ult ptr %2, %361
  %386 = icmp ult ptr %359, %332
  %387 = and i1 %385, %386
  %388 = or i1 %384, %387
  %389 = icmp ult ptr %333, %340
  %390 = icmp ult ptr %337, %335
  %391 = and i1 %389, %390
  %392 = or i1 %388, %391
  %393 = icmp ult ptr %333, %344
  %394 = icmp ult ptr %342, %335
  %395 = and i1 %393, %394
  %396 = or i1 %392, %395
  %397 = icmp ult ptr %333, %348
  %398 = icmp ult ptr %323, %335
  %399 = and i1 %397, %398
  %400 = or i1 %396, %399
  %401 = icmp ult ptr %333, %352
  %402 = icmp ult ptr %350, %335
  %403 = and i1 %401, %402
  %404 = or i1 %400, %403
  %405 = icmp ult ptr %333, %357
  %406 = icmp ult ptr %354, %335
  %407 = and i1 %405, %406
  %408 = or i1 %404, %407
  %409 = icmp ult ptr %333, %361
  %410 = icmp ult ptr %359, %335
  %411 = and i1 %409, %410
  %412 = or i1 %408, %411
  %413 = icmp ult ptr %337, %344
  %414 = icmp ult ptr %342, %340
  %415 = and i1 %413, %414
  %416 = or i1 %412, %415
  %417 = icmp ult ptr %337, %348
  %418 = icmp ult ptr %323, %340
  %419 = and i1 %417, %418
  %420 = or i1 %416, %419
  %421 = icmp ult ptr %337, %352
  %422 = icmp ult ptr %350, %340
  %423 = and i1 %421, %422
  %424 = or i1 %420, %423
  %425 = icmp ult ptr %337, %357
  %426 = icmp ult ptr %354, %340
  %427 = and i1 %425, %426
  %428 = or i1 %424, %427
  %429 = icmp ult ptr %337, %361
  %430 = icmp ult ptr %359, %340
  %431 = and i1 %429, %430
  %432 = or i1 %428, %431
  %433 = icmp ult ptr %342, %348
  %434 = icmp ult ptr %323, %344
  %435 = and i1 %433, %434
  %436 = or i1 %432, %435
  %437 = icmp ult ptr %342, %352
  %438 = icmp ult ptr %350, %344
  %439 = and i1 %437, %438
  %440 = or i1 %436, %439
  %441 = icmp ult ptr %342, %357
  %442 = icmp ult ptr %354, %344
  %443 = and i1 %441, %442
  %444 = or i1 %440, %443
  %445 = icmp ult ptr %342, %361
  %446 = icmp ult ptr %359, %344
  %447 = and i1 %445, %446
  %448 = or i1 %444, %447
  %449 = icmp ult ptr %323, %352
  %450 = icmp ult ptr %350, %348
  %451 = and i1 %449, %450
  %452 = or i1 %448, %451
  %453 = icmp ult ptr %323, %357
  %454 = icmp ult ptr %354, %348
  %455 = and i1 %453, %454
  %456 = or i1 %452, %455
  %457 = icmp ult ptr %323, %361
  %458 = icmp ult ptr %359, %348
  %459 = and i1 %457, %458
  %460 = or i1 %456, %459
  %461 = icmp ult ptr %350, %357
  %462 = icmp ult ptr %354, %352
  %463 = and i1 %461, %462
  %464 = or i1 %460, %463
  %465 = icmp ult ptr %350, %361
  %466 = icmp ult ptr %359, %352
  %467 = and i1 %465, %466
  %468 = or i1 %464, %467
  %469 = icmp ult ptr %354, %361
  %470 = icmp ult ptr %359, %357
  %471 = and i1 %469, %470
  %472 = or i1 %468, %471
  br i1 %472, label %522, label %473

473:                                              ; preds = %328
  %474 = and i64 %326, -2
  %475 = shl i64 %474, 1
  br label %476

476:                                              ; preds = %476, %473
  %477 = phi i64 [ 0, %473 ], [ %518, %476 ]
  %478 = shl i64 %477, 1
  %479 = add nuw nsw i64 %478, %322
  %480 = add nuw nsw i64 %479, %322
  %481 = getelementptr inbounds nuw double, ptr %2, i64 %478
  %482 = load <4 x double>, ptr %481, align 8, !tbaa !6
  %483 = shufflevector <4 x double> %482, <4 x double> poison, <2 x i32> <i32 0, i32 2>
  %484 = shufflevector <4 x double> %482, <4 x double> poison, <2 x i32> <i32 1, i32 3>
  %485 = getelementptr inbounds nuw double, ptr %2, i64 %479
  %486 = load <4 x double>, ptr %485, align 8, !tbaa !6
  %487 = shufflevector <4 x double> %486, <4 x double> poison, <2 x i32> <i32 0, i32 2>
  %488 = shufflevector <4 x double> %486, <4 x double> poison, <2 x i32> <i32 1, i32 3>
  %489 = fadd <2 x double> %483, %487
  %490 = fneg <2 x double> %484
  %491 = fsub <2 x double> %490, %488
  %492 = fsub <2 x double> %483, %487
  %493 = fsub <2 x double> %488, %484
  %494 = getelementptr inbounds nuw double, ptr %2, i64 %480
  %495 = load <4 x double>, ptr %494, align 8, !tbaa !6
  %496 = shufflevector <4 x double> %495, <4 x double> poison, <2 x i32> <i32 0, i32 2>
  %497 = shufflevector <4 x double> %495, <4 x double> poison, <2 x i32> <i32 1, i32 3>
  %498 = getelementptr inbounds nuw double, ptr %323, i64 %480
  %499 = load <4 x double>, ptr %498, align 8, !tbaa !6
  %500 = shufflevector <4 x double> %499, <4 x double> poison, <2 x i32> <i32 0, i32 2>
  %501 = shufflevector <4 x double> %499, <4 x double> poison, <2 x i32> <i32 1, i32 3>
  %502 = fadd <2 x double> %496, %500
  %503 = fadd <2 x double> %497, %501
  %504 = fsub <2 x double> %496, %500
  %505 = fsub <2 x double> %497, %501
  %506 = fadd <2 x double> %489, %502
  %507 = fsub <2 x double> %491, %503
  %508 = shufflevector <2 x double> %506, <2 x double> %507, <4 x i32> <i32 0, i32 2, i32 1, i32 3>
  store <4 x double> %508, ptr %481, align 8, !tbaa !6
  %509 = fsub <2 x double> %489, %502
  %510 = fadd <2 x double> %491, %503
  %511 = shufflevector <2 x double> %509, <2 x double> %510, <4 x i32> <i32 0, i32 2, i32 1, i32 3>
  store <4 x double> %511, ptr %494, align 8, !tbaa !6
  %512 = fsub <2 x double> %492, %505
  %513 = fsub <2 x double> %493, %504
  %514 = shufflevector <2 x double> %512, <2 x double> %513, <4 x i32> <i32 0, i32 2, i32 1, i32 3>
  store <4 x double> %514, ptr %485, align 8, !tbaa !6
  %515 = fadd <2 x double> %492, %505
  %516 = fadd <2 x double> %493, %504
  %517 = shufflevector <2 x double> %515, <2 x double> %516, <4 x i32> <i32 0, i32 2, i32 1, i32 3>
  store <4 x double> %517, ptr %498, align 8, !tbaa !6
  %518 = add nuw i64 %477, 2
  %519 = icmp eq i64 %518, %474
  br i1 %519, label %520, label %476, !llvm.loop !35

520:                                              ; preds = %476
  %521 = icmp eq i64 %326, %474
  br i1 %521, label %591, label %522

522:                                              ; preds = %328, %321, %520
  %523 = phi i64 [ 0, %328 ], [ 0, %321 ], [ %475, %520 ]
  br label %524

524:                                              ; preds = %522, %524
  %525 = phi i64 [ %561, %524 ], [ %523, %522 ]
  %526 = add nuw nsw i64 %525, %322
  %527 = add nuw nsw i64 %526, %322
  %528 = getelementptr inbounds nuw double, ptr %2, i64 %525
  %529 = load double, ptr %528, align 8, !tbaa !6
  %530 = getelementptr inbounds nuw double, ptr %2, i64 %526
  %531 = load double, ptr %530, align 8, !tbaa !6
  %532 = fadd double %529, %531
  %533 = getelementptr inbounds nuw i8, ptr %528, i64 8
  %534 = load double, ptr %533, align 8, !tbaa !6
  %535 = fneg double %534
  %536 = getelementptr i8, ptr %530, i64 8
  %537 = load double, ptr %536, align 8, !tbaa !6
  %538 = fsub double %535, %537
  %539 = fsub double %529, %531
  %540 = fsub double %537, %534
  %541 = getelementptr inbounds nuw double, ptr %2, i64 %527
  %542 = load double, ptr %541, align 8, !tbaa !6
  %543 = getelementptr inbounds nuw double, ptr %323, i64 %527
  %544 = load double, ptr %543, align 8, !tbaa !6
  %545 = fadd double %542, %544
  %546 = getelementptr i8, ptr %541, i64 8
  %547 = load double, ptr %546, align 8, !tbaa !6
  %548 = getelementptr i8, ptr %543, i64 8
  %549 = load double, ptr %548, align 8, !tbaa !6
  %550 = fadd double %547, %549
  %551 = fsub double %542, %544
  %552 = fsub double %547, %549
  %553 = fadd double %532, %545
  store double %553, ptr %528, align 8, !tbaa !6
  %554 = fsub double %538, %550
  store double %554, ptr %533, align 8, !tbaa !6
  %555 = fsub double %532, %545
  store double %555, ptr %541, align 8, !tbaa !6
  %556 = fadd double %538, %550
  store double %556, ptr %546, align 8, !tbaa !6
  %557 = fsub double %539, %552
  store double %557, ptr %530, align 8, !tbaa !6
  %558 = fsub double %540, %551
  store double %558, ptr %536, align 8, !tbaa !6
  %559 = fadd double %539, %552
  store double %559, ptr %543, align 8, !tbaa !6
  %560 = fadd double %540, %551
  store double %560, ptr %548, align 8, !tbaa !6
  %561 = add nuw nsw i64 %525, 2
  %562 = icmp samesign ult i64 %561, %322
  br i1 %562, label %524, label %591, !llvm.loop !36

563:                                              ; preds = %318, %563
  %564 = phi i64 [ %578, %563 ], [ %319, %318 ]
  %565 = getelementptr inbounds nuw double, ptr %2, i64 %564
  %566 = load double, ptr %565, align 8, !tbaa !6
  %567 = getelementptr inbounds nuw double, ptr %249, i64 %564
  %568 = load double, ptr %567, align 8, !tbaa !6
  %569 = fsub double %566, %568
  %570 = getelementptr inbounds nuw i8, ptr %565, i64 8
  %571 = load double, ptr %570, align 8, !tbaa !6
  %572 = getelementptr i8, ptr %567, i64 8
  %573 = load double, ptr %572, align 8, !tbaa !6
  %574 = fsub double %573, %571
  %575 = fadd double %566, %568
  store double %575, ptr %565, align 8, !tbaa !6
  %576 = fneg double %571
  %577 = fsub double %576, %573
  store double %577, ptr %570, align 8, !tbaa !6
  store double %569, ptr %567, align 8, !tbaa !6
  store double %574, ptr %572, align 8, !tbaa !6
  %578 = add nuw nsw i64 %564, 2
  %579 = icmp samesign ult i64 %578, %248
  br i1 %579, label %563, label %591, !llvm.loop !37

580:                                              ; preds = %5
  %581 = icmp eq i32 %0, 4
  br i1 %581, label %582, label %591

582:                                              ; preds = %580
  %583 = getelementptr inbounds nuw i8, ptr %2, i64 16
  %584 = load <2 x double>, ptr %2, align 8, !tbaa !6
  %585 = shufflevector <2 x double> %584, <2 x double> poison, <4 x i32> <i32 0, i32 1, i32 0, i32 1>
  %586 = load <2 x double>, ptr %583, align 8, !tbaa !6
  %587 = shufflevector <2 x double> %586, <2 x double> poison, <4 x i32> <i32 0, i32 1, i32 0, i32 1>
  %588 = fadd <4 x double> %585, %587
  %589 = fsub <4 x double> %585, %587
  %590 = shufflevector <4 x double> %588, <4 x double> %589, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  store <4 x double> %590, ptr %2, align 8, !tbaa !6
  br label %591

591:                                              ; preds = %563, %524, %316, %520, %320, %246, %580, %582, %9
  ret void
}

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: read) uwtable
define dso_local double @errorcheck(i32 noundef %0, i32 noundef %1, double noundef %2, ptr noundef readonly captures(none) %3) local_unnamed_addr #7 {
  %5 = icmp sgt i32 %0, %1
  br i1 %5, label %28, label %6

6:                                                ; preds = %4
  %7 = fneg double %2
  %8 = sext i32 %0 to i64
  %9 = add i32 %1, 1
  br label %10

10:                                               ; preds = %6, %10
  %11 = phi i64 [ %8, %6 ], [ %25, %10 ]
  %12 = phi double [ 0.000000e+00, %6 ], [ %24, %10 ]
  %13 = phi i32 [ 0, %6 ], [ %16, %10 ]
  %14 = mul nuw nsw i32 %13, 7141
  %15 = add nuw nsw i32 %14, 54773
  %16 = urem i32 %15, 259200
  %17 = uitofp nneg i32 %16 to double
  %18 = getelementptr inbounds double, ptr %3, i64 %11
  %19 = load double, ptr %18, align 8, !tbaa !6
  %20 = fmul double %19, %7
  %21 = tail call double @llvm.fmuladd.f64(double %17, double 0x3ED02E85C0898B71, double %20)
  %22 = tail call double @llvm.fabs.f64(double %21)
  %23 = fcmp ogt double %12, %22
  %24 = select i1 %23, double %12, double %22
  %25 = add nsw i64 %11, 1
  %26 = trunc i64 %25 to i32
  %27 = icmp eq i32 %9, %26
  br i1 %27, label %28, label %10, !llvm.loop !15

28:                                               ; preds = %10, %4
  %29 = phi double [ 0.000000e+00, %4 ], [ %24, %10 ]
  ret double %29
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fabs.f64(double) #8

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #9

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #10

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #11

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #12

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fmuladd.f64(double, double, double) #8

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite)
declare void @free(ptr allocptr noundef captures(none)) local_unnamed_addr #13

; Function Attrs: nofree nounwind
declare noundef i32 @gettimeofday(ptr noundef captures(none), ptr noundef captures(none)) local_unnamed_addr #9

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable
define internal fastcc void @bitrv2(i32 noundef range(i32 3, -2147483648) %0, ptr noundef captures(none) initializes((0, 4)) %1, ptr noundef captures(none) %2) unnamed_addr #6 {
  store i32 0, ptr %1, align 4, !tbaa !10
  %4 = icmp samesign ugt i32 %0, 8
  br i1 %4, label %5, label %49

5:                                                ; preds = %3, %42
  %6 = phi i32 [ %43, %42 ], [ 1, %3 ]
  %7 = phi i32 [ %8, %42 ], [ %0, %3 ]
  %8 = lshr i32 %7, 1
  %9 = icmp sgt i32 %6, 0
  br i1 %9, label %10, label %42

10:                                               ; preds = %5
  %11 = zext nneg i32 %6 to i64
  %12 = getelementptr inbounds nuw i32, ptr %1, i64 %11
  %13 = icmp ult i32 %6, 8
  br i1 %13, label %32, label %14

14:                                               ; preds = %10
  %15 = and i64 %11, 2147483640
  %16 = insertelement <4 x i32> poison, i32 %8, i64 0
  %17 = shufflevector <4 x i32> %16, <4 x i32> poison, <4 x i32> zeroinitializer
  br label %18

18:                                               ; preds = %18, %14
  %19 = phi i64 [ 0, %14 ], [ %28, %18 ]
  %20 = getelementptr inbounds nuw i32, ptr %1, i64 %19
  %21 = getelementptr inbounds nuw i8, ptr %20, i64 16
  %22 = load <4 x i32>, ptr %20, align 4, !tbaa !10
  %23 = load <4 x i32>, ptr %21, align 4, !tbaa !10
  %24 = add nsw <4 x i32> %22, %17
  %25 = add nsw <4 x i32> %23, %17
  %26 = getelementptr inbounds nuw i32, ptr %12, i64 %19
  %27 = getelementptr inbounds nuw i8, ptr %26, i64 16
  store <4 x i32> %24, ptr %26, align 4, !tbaa !10
  store <4 x i32> %25, ptr %27, align 4, !tbaa !10
  %28 = add nuw i64 %19, 8
  %29 = icmp eq i64 %28, %15
  br i1 %29, label %30, label %18, !llvm.loop !38

30:                                               ; preds = %18
  %31 = icmp eq i64 %15, %11
  br i1 %31, label %42, label %32

32:                                               ; preds = %10, %30
  %33 = phi i64 [ 0, %10 ], [ %15, %30 ]
  br label %34

34:                                               ; preds = %32, %34
  %35 = phi i64 [ %40, %34 ], [ %33, %32 ]
  %36 = getelementptr inbounds nuw i32, ptr %1, i64 %35
  %37 = load i32, ptr %36, align 4, !tbaa !10
  %38 = add nsw i32 %37, %8
  %39 = getelementptr inbounds nuw i32, ptr %12, i64 %35
  store i32 %38, ptr %39, align 4, !tbaa !10
  %40 = add nuw nsw i64 %35, 1
  %41 = icmp eq i64 %40, %11
  br i1 %41, label %42, label %34, !llvm.loop !39

42:                                               ; preds = %34, %30, %5
  %43 = shl i32 %6, 1
  %44 = shl i32 %6, 4
  %45 = icmp slt i32 %44, %8
  br i1 %45, label %5, label %46, !llvm.loop !40

46:                                               ; preds = %42
  %47 = shl i32 %6, 2
  %48 = icmp eq i32 %44, %8
  br i1 %48, label %57, label %51

49:                                               ; preds = %3
  %50 = icmp eq i32 %0, 8
  br i1 %50, label %59, label %175

51:                                               ; preds = %46
  %52 = icmp sgt i32 %43, 1
  br i1 %52, label %53, label %175

53:                                               ; preds = %51
  %54 = zext nneg i32 %47 to i64
  %55 = zext nneg i32 %43 to i64
  %56 = getelementptr double, ptr %2, i64 %54
  br label %139

57:                                               ; preds = %46
  %58 = icmp sgt i32 %43, 0
  br i1 %58, label %59, label %175

59:                                               ; preds = %49, %57
  %60 = phi i32 [ %43, %57 ], [ 1, %49 ]
  %61 = phi i32 [ %47, %57 ], [ 2, %49 ]
  %62 = shl nsw i32 %60, 2
  %63 = zext nneg i32 %61 to i64
  %64 = zext nneg i32 %60 to i64
  %65 = getelementptr double, ptr %2, i64 %63
  br label %66

66:                                               ; preds = %59, %121
  %67 = phi i64 [ 0, %59 ], [ %137, %121 ]
  %68 = icmp eq i64 %67, 0
  br i1 %68, label %121, label %69

69:                                               ; preds = %66
  %70 = getelementptr inbounds nuw i32, ptr %1, i64 %67
  %71 = load i32, ptr %70, align 4, !tbaa !10
  %72 = sext i32 %71 to i64
  %73 = trunc i64 %67 to i32
  %74 = shl i32 %73, 1
  br label %75

75:                                               ; preds = %69, %75
  %76 = phi i64 [ 0, %69 ], [ %119, %75 ]
  %77 = shl nuw nsw i64 %76, 1
  %78 = add nsw i64 %77, %72
  %79 = getelementptr inbounds nuw i32, ptr %1, i64 %76
  %80 = load i32, ptr %79, align 4, !tbaa !10
  %81 = add nsw i32 %80, %74
  %82 = getelementptr inbounds double, ptr %2, i64 %78
  %83 = getelementptr i8, ptr %82, i64 8
  %84 = sext i32 %81 to i64
  %85 = getelementptr inbounds double, ptr %2, i64 %84
  %86 = load double, ptr %85, align 8, !tbaa !6
  %87 = getelementptr i8, ptr %85, i64 8
  %88 = load double, ptr %87, align 8, !tbaa !6
  %89 = load <2 x double>, ptr %82, align 8, !tbaa !6
  store double %86, ptr %82, align 8, !tbaa !6
  store double %88, ptr %83, align 8, !tbaa !6
  store <2 x double> %89, ptr %85, align 8, !tbaa !6
  %90 = add nsw i64 %78, %63
  %91 = add nsw i32 %81, %62
  %92 = getelementptr inbounds double, ptr %2, i64 %90
  %93 = getelementptr i8, ptr %92, i64 8
  %94 = sext i32 %91 to i64
  %95 = getelementptr inbounds double, ptr %2, i64 %94
  %96 = load double, ptr %95, align 8, !tbaa !6
  %97 = getelementptr i8, ptr %95, i64 8
  %98 = load double, ptr %97, align 8, !tbaa !6
  %99 = load <2 x double>, ptr %92, align 8, !tbaa !6
  store double %96, ptr %92, align 8, !tbaa !6
  store double %98, ptr %93, align 8, !tbaa !6
  store <2 x double> %99, ptr %95, align 8, !tbaa !6
  %100 = add nsw i64 %90, %63
  %101 = sub nsw i32 %91, %61
  %102 = getelementptr inbounds double, ptr %2, i64 %100
  %103 = getelementptr i8, ptr %102, i64 8
  %104 = sext i32 %101 to i64
  %105 = getelementptr inbounds double, ptr %2, i64 %104
  %106 = load double, ptr %105, align 8, !tbaa !6
  %107 = getelementptr i8, ptr %105, i64 8
  %108 = load double, ptr %107, align 8, !tbaa !6
  %109 = load <2 x double>, ptr %102, align 8, !tbaa !6
  store double %106, ptr %102, align 8, !tbaa !6
  store double %108, ptr %103, align 8, !tbaa !6
  store <2 x double> %109, ptr %105, align 8, !tbaa !6
  %110 = add nsw i32 %101, %62
  %111 = getelementptr double, ptr %65, i64 %100
  %112 = getelementptr i8, ptr %111, i64 8
  %113 = sext i32 %110 to i64
  %114 = getelementptr inbounds double, ptr %2, i64 %113
  %115 = load double, ptr %114, align 8, !tbaa !6
  %116 = getelementptr i8, ptr %114, i64 8
  %117 = load double, ptr %116, align 8, !tbaa !6
  %118 = load <2 x double>, ptr %111, align 8, !tbaa !6
  store double %115, ptr %111, align 8, !tbaa !6
  store double %117, ptr %112, align 8, !tbaa !6
  store <2 x double> %118, ptr %114, align 8, !tbaa !6
  %119 = add nuw nsw i64 %76, 1
  %120 = icmp eq i64 %119, %67
  br i1 %120, label %121, label %75, !llvm.loop !41

121:                                              ; preds = %75, %66
  %122 = phi i32 [ 0, %66 ], [ %71, %75 ]
  %123 = trunc i64 %67 to i32
  %124 = add i32 %60, %123
  %125 = shl i32 %124, 1
  %126 = add nsw i32 %122, %125
  %127 = add nsw i32 %126, %61
  %128 = sext i32 %126 to i64
  %129 = getelementptr inbounds double, ptr %2, i64 %128
  %130 = getelementptr i8, ptr %129, i64 8
  %131 = sext i32 %127 to i64
  %132 = getelementptr inbounds double, ptr %2, i64 %131
  %133 = load double, ptr %132, align 8, !tbaa !6
  %134 = getelementptr i8, ptr %132, i64 8
  %135 = load double, ptr %134, align 8, !tbaa !6
  %136 = load <2 x double>, ptr %129, align 8, !tbaa !6
  store double %133, ptr %129, align 8, !tbaa !6
  store double %135, ptr %130, align 8, !tbaa !6
  store <2 x double> %136, ptr %132, align 8, !tbaa !6
  %137 = add nuw nsw i64 %67, 1
  %138 = icmp eq i64 %137, %64
  br i1 %138, label %175, label %66, !llvm.loop !42

139:                                              ; preds = %53, %172
  %140 = phi i64 [ 1, %53 ], [ %173, %172 ]
  %141 = getelementptr inbounds nuw i32, ptr %1, i64 %140
  %142 = load i32, ptr %141, align 4, !tbaa !10
  %143 = sext i32 %142 to i64
  %144 = trunc i64 %140 to i32
  %145 = shl i32 %144, 1
  br label %146

146:                                              ; preds = %139, %146
  %147 = phi i64 [ 0, %139 ], [ %170, %146 ]
  %148 = shl nuw nsw i64 %147, 1
  %149 = add nsw i64 %148, %143
  %150 = getelementptr inbounds nuw i32, ptr %1, i64 %147
  %151 = load i32, ptr %150, align 4, !tbaa !10
  %152 = add nsw i32 %151, %145
  %153 = getelementptr inbounds double, ptr %2, i64 %149
  %154 = getelementptr i8, ptr %153, i64 8
  %155 = sext i32 %152 to i64
  %156 = getelementptr inbounds double, ptr %2, i64 %155
  %157 = load double, ptr %156, align 8, !tbaa !6
  %158 = getelementptr i8, ptr %156, i64 8
  %159 = load double, ptr %158, align 8, !tbaa !6
  %160 = load <2 x double>, ptr %153, align 8, !tbaa !6
  store double %157, ptr %153, align 8, !tbaa !6
  store double %159, ptr %154, align 8, !tbaa !6
  store <2 x double> %160, ptr %156, align 8, !tbaa !6
  %161 = add nsw i32 %152, %47
  %162 = getelementptr double, ptr %56, i64 %149
  %163 = getelementptr i8, ptr %162, i64 8
  %164 = sext i32 %161 to i64
  %165 = getelementptr inbounds double, ptr %2, i64 %164
  %166 = load double, ptr %165, align 8, !tbaa !6
  %167 = getelementptr i8, ptr %165, i64 8
  %168 = load double, ptr %167, align 8, !tbaa !6
  %169 = load <2 x double>, ptr %162, align 8, !tbaa !6
  store double %166, ptr %162, align 8, !tbaa !6
  store double %168, ptr %163, align 8, !tbaa !6
  store <2 x double> %169, ptr %165, align 8, !tbaa !6
  %170 = add nuw nsw i64 %147, 1
  %171 = icmp eq i64 %170, %140
  br i1 %171, label %172, label %146, !llvm.loop !43

172:                                              ; preds = %146
  %173 = add nuw nsw i64 %140, 1
  %174 = icmp eq i64 %173, %55
  br i1 %174, label %175, label %139, !llvm.loop !44

175:                                              ; preds = %172, %121, %49, %51, %57
  ret void
}

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable
define internal fastcc void @cftfsub(i32 noundef range(i32 4, -2147483648) %0, ptr noundef captures(none) %1, ptr noundef readonly captures(none) %2) unnamed_addr #6 {
  %4 = icmp samesign ugt i32 %0, 8
  br i1 %4, label %5, label %12

5:                                                ; preds = %3
  tail call fastcc void @cft1st(i32 noundef %0, ptr noundef %1, ptr noundef %2)
  %6 = icmp samesign ugt i32 %0, 32
  br i1 %6, label %7, label %12

7:                                                ; preds = %5, %7
  %8 = phi i32 [ %10, %7 ], [ 32, %5 ]
  %9 = phi i32 [ %8, %7 ], [ 8, %5 ]
  tail call fastcc void @cftmdl(i32 noundef %0, i32 noundef %9, ptr noundef %1, ptr noundef %2)
  %10 = shl i32 %8, 2
  %11 = icmp slt i32 %10, %0
  br i1 %11, label %7, label %12, !llvm.loop !45

12:                                               ; preds = %7, %5, %3
  %13 = phi i32 [ 2, %3 ], [ 8, %5 ], [ %8, %7 ]
  %14 = shl i32 %13, 2
  %15 = icmp eq i32 %14, %0
  %16 = icmp sgt i32 %13, 0
  br i1 %15, label %83, label %17

17:                                               ; preds = %12
  br i1 %16, label %18, label %341

18:                                               ; preds = %17
  %19 = zext nneg i32 %13 to i64
  %20 = zext nneg i32 %13 to i64
  %21 = getelementptr inbounds nuw double, ptr %1, i64 %19
  %22 = add nsw i64 %20, -1
  %23 = lshr i64 %22, 1
  %24 = add nuw i64 %23, 1
  %25 = icmp ult i32 %13, 31
  br i1 %25, label %81, label %26

26:                                               ; preds = %18
  %27 = shl nuw nsw i64 %20, 3
  %28 = add nsw i64 %27, -8
  %29 = and i64 %28, -16
  %30 = getelementptr i8, ptr %1, i64 %28
  %31 = getelementptr i8, ptr %1, i64 8
  %32 = getelementptr i8, ptr %1, i64 %29
  %33 = getelementptr i8, ptr %32, i64 16
  %34 = shl nuw nsw i64 %20, 3
  %35 = add nsw i64 %29, %34
  %36 = getelementptr i8, ptr %1, i64 %35
  %37 = getelementptr i8, ptr %36, i64 8
  %38 = getelementptr i8, ptr %1, i64 %34
  %39 = getelementptr i8, ptr %38, i64 8
  %40 = getelementptr i8, ptr %1, i64 %35
  %41 = getelementptr i8, ptr %40, i64 16
  %42 = icmp ult ptr %1, %33
  %43 = icmp ult ptr %31, %30
  %44 = and i1 %42, %43
  %45 = icmp ult ptr %1, %37
  %46 = icmp ult ptr %21, %30
  %47 = and i1 %45, %46
  %48 = or i1 %44, %47
  %49 = icmp ult ptr %1, %41
  %50 = icmp ult ptr %39, %30
  %51 = and i1 %49, %50
  %52 = or i1 %48, %51
  %53 = icmp ult ptr %31, %37
  %54 = icmp ult ptr %21, %33
  %55 = and i1 %53, %54
  %56 = or i1 %52, %55
  %57 = icmp ult ptr %31, %41
  %58 = icmp ult ptr %39, %33
  %59 = and i1 %57, %58
  %60 = or i1 %56, %59
  %61 = icmp ult ptr %21, %41
  %62 = icmp ult ptr %39, %37
  %63 = and i1 %61, %62
  %64 = or i1 %60, %63
  br i1 %64, label %81, label %65

65:                                               ; preds = %26
  %66 = and i64 %24, -2
  %67 = shl i64 %66, 1
  br label %68

68:                                               ; preds = %68, %65
  %69 = phi i64 [ 0, %65 ], [ %77, %68 ]
  %70 = shl i64 %69, 1
  %71 = getelementptr inbounds nuw double, ptr %1, i64 %70
  %72 = load <2 x double>, ptr %71, align 8
  %73 = getelementptr inbounds nuw double, ptr %21, i64 %70
  %74 = load <2 x double>, ptr %73, align 8
  %75 = fsub <2 x double> %72, %74
  %76 = fadd <2 x double> %72, %74
  store <2 x double> %76, ptr %71, align 8
  store <2 x double> %75, ptr %73, align 8
  %77 = add nuw i64 %69, 1
  %78 = icmp eq i64 %77, %66
  br i1 %78, label %79, label %68, !llvm.loop !46

79:                                               ; preds = %68
  %80 = icmp eq i64 %24, %66
  br i1 %80, label %341, label %81

81:                                               ; preds = %26, %18, %79
  %82 = phi i64 [ 0, %26 ], [ 0, %18 ], [ %67, %79 ]
  br label %325

83:                                               ; preds = %12
  br i1 %16, label %84, label %341

84:                                               ; preds = %83
  %85 = zext nneg i32 %13 to i64
  %86 = zext nneg i32 %13 to i64
  %87 = getelementptr double, ptr %1, i64 %85
  %88 = add nsw i64 %86, -1
  %89 = lshr i64 %88, 1
  %90 = add nuw i64 %89, 1
  %91 = icmp ult i32 %13, 51
  br i1 %91, label %285, label %92

92:                                               ; preds = %84
  %93 = shl nuw nsw i64 %86, 3
  %94 = add nsw i64 %93, -8
  %95 = and i64 %94, -16
  %96 = getelementptr i8, ptr %1, i64 %94
  %97 = getelementptr i8, ptr %1, i64 8
  %98 = getelementptr i8, ptr %1, i64 %95
  %99 = getelementptr i8, ptr %98, i64 16
  %100 = shl nuw nsw i64 %86, 4
  %101 = getelementptr i8, ptr %1, i64 %100
  %102 = add nsw i64 %95, %100
  %103 = getelementptr i8, ptr %1, i64 %102
  %104 = getelementptr i8, ptr %103, i64 8
  %105 = getelementptr i8, ptr %1, i64 %100
  %106 = getelementptr i8, ptr %105, i64 8
  %107 = getelementptr i8, ptr %1, i64 %102
  %108 = getelementptr i8, ptr %107, i64 16
  %109 = shl nuw nsw i64 %86, 3
  %110 = add nsw i64 %95, %109
  %111 = getelementptr i8, ptr %1, i64 %110
  %112 = getelementptr i8, ptr %111, i64 8
  %113 = getelementptr i8, ptr %1, i64 %109
  %114 = getelementptr i8, ptr %113, i64 8
  %115 = getelementptr i8, ptr %1, i64 %110
  %116 = getelementptr i8, ptr %115, i64 16
  %117 = mul nuw nsw i64 %86, 24
  %118 = getelementptr i8, ptr %1, i64 %117
  %119 = add nsw i64 %117, %95
  %120 = getelementptr i8, ptr %1, i64 %119
  %121 = getelementptr i8, ptr %120, i64 8
  %122 = getelementptr i8, ptr %1, i64 %117
  %123 = getelementptr i8, ptr %122, i64 8
  %124 = getelementptr i8, ptr %1, i64 %119
  %125 = getelementptr i8, ptr %124, i64 16
  %126 = icmp ult ptr %1, %99
  %127 = icmp ult ptr %97, %96
  %128 = and i1 %126, %127
  %129 = icmp ult ptr %1, %104
  %130 = icmp ult ptr %101, %96
  %131 = and i1 %129, %130
  %132 = or i1 %128, %131
  %133 = icmp ult ptr %1, %108
  %134 = icmp ult ptr %106, %96
  %135 = and i1 %133, %134
  %136 = or i1 %132, %135
  %137 = icmp ult ptr %1, %112
  %138 = icmp ult ptr %87, %96
  %139 = and i1 %137, %138
  %140 = or i1 %136, %139
  %141 = icmp ult ptr %1, %116
  %142 = icmp ult ptr %114, %96
  %143 = and i1 %141, %142
  %144 = or i1 %140, %143
  %145 = icmp ult ptr %1, %121
  %146 = icmp ult ptr %118, %96
  %147 = and i1 %145, %146
  %148 = or i1 %144, %147
  %149 = icmp ult ptr %1, %125
  %150 = icmp ult ptr %123, %96
  %151 = and i1 %149, %150
  %152 = or i1 %148, %151
  %153 = icmp ult ptr %97, %104
  %154 = icmp ult ptr %101, %99
  %155 = and i1 %153, %154
  %156 = or i1 %152, %155
  %157 = icmp ult ptr %97, %108
  %158 = icmp ult ptr %106, %99
  %159 = and i1 %157, %158
  %160 = or i1 %156, %159
  %161 = icmp ult ptr %97, %112
  %162 = icmp ult ptr %87, %99
  %163 = and i1 %161, %162
  %164 = or i1 %160, %163
  %165 = icmp ult ptr %97, %116
  %166 = icmp ult ptr %114, %99
  %167 = and i1 %165, %166
  %168 = or i1 %164, %167
  %169 = icmp ult ptr %97, %121
  %170 = icmp ult ptr %118, %99
  %171 = and i1 %169, %170
  %172 = or i1 %168, %171
  %173 = icmp ult ptr %97, %125
  %174 = icmp ult ptr %123, %99
  %175 = and i1 %173, %174
  %176 = or i1 %172, %175
  %177 = icmp ult ptr %101, %108
  %178 = icmp ult ptr %106, %104
  %179 = and i1 %177, %178
  %180 = or i1 %176, %179
  %181 = icmp ult ptr %101, %112
  %182 = icmp ult ptr %87, %104
  %183 = and i1 %181, %182
  %184 = or i1 %180, %183
  %185 = icmp ult ptr %101, %116
  %186 = icmp ult ptr %114, %104
  %187 = and i1 %185, %186
  %188 = or i1 %184, %187
  %189 = icmp ult ptr %101, %121
  %190 = icmp ult ptr %118, %104
  %191 = and i1 %189, %190
  %192 = or i1 %188, %191
  %193 = icmp ult ptr %101, %125
  %194 = icmp ult ptr %123, %104
  %195 = and i1 %193, %194
  %196 = or i1 %192, %195
  %197 = icmp ult ptr %106, %112
  %198 = icmp ult ptr %87, %108
  %199 = and i1 %197, %198
  %200 = or i1 %196, %199
  %201 = icmp ult ptr %106, %116
  %202 = icmp ult ptr %114, %108
  %203 = and i1 %201, %202
  %204 = or i1 %200, %203
  %205 = icmp ult ptr %106, %121
  %206 = icmp ult ptr %118, %108
  %207 = and i1 %205, %206
  %208 = or i1 %204, %207
  %209 = icmp ult ptr %106, %125
  %210 = icmp ult ptr %123, %108
  %211 = and i1 %209, %210
  %212 = or i1 %208, %211
  %213 = icmp ult ptr %87, %116
  %214 = icmp ult ptr %114, %112
  %215 = and i1 %213, %214
  %216 = or i1 %212, %215
  %217 = icmp ult ptr %87, %121
  %218 = icmp ult ptr %118, %112
  %219 = and i1 %217, %218
  %220 = or i1 %216, %219
  %221 = icmp ult ptr %87, %125
  %222 = icmp ult ptr %123, %112
  %223 = and i1 %221, %222
  %224 = or i1 %220, %223
  %225 = icmp ult ptr %114, %121
  %226 = icmp ult ptr %118, %116
  %227 = and i1 %225, %226
  %228 = or i1 %224, %227
  %229 = icmp ult ptr %114, %125
  %230 = icmp ult ptr %123, %116
  %231 = and i1 %229, %230
  %232 = or i1 %228, %231
  %233 = icmp ult ptr %118, %125
  %234 = icmp ult ptr %123, %121
  %235 = and i1 %233, %234
  %236 = or i1 %232, %235
  br i1 %236, label %285, label %237

237:                                              ; preds = %92
  %238 = and i64 %90, -2
  %239 = shl i64 %238, 1
  br label %240

240:                                              ; preds = %240, %237
  %241 = phi i64 [ 0, %237 ], [ %281, %240 ]
  %242 = shl i64 %241, 1
  %243 = add nuw nsw i64 %242, %85
  %244 = add nuw nsw i64 %243, %85
  %245 = getelementptr inbounds nuw double, ptr %1, i64 %242
  %246 = load <4 x double>, ptr %245, align 8, !tbaa !6
  %247 = shufflevector <4 x double> %246, <4 x double> poison, <2 x i32> <i32 0, i32 2>
  %248 = shufflevector <4 x double> %246, <4 x double> poison, <2 x i32> <i32 1, i32 3>
  %249 = getelementptr inbounds nuw double, ptr %1, i64 %243
  %250 = load <4 x double>, ptr %249, align 8, !tbaa !6
  %251 = shufflevector <4 x double> %250, <4 x double> poison, <2 x i32> <i32 0, i32 2>
  %252 = shufflevector <4 x double> %250, <4 x double> poison, <2 x i32> <i32 1, i32 3>
  %253 = fadd <2 x double> %247, %251
  %254 = fadd <2 x double> %248, %252
  %255 = fsub <2 x double> %247, %251
  %256 = fsub <2 x double> %248, %252
  %257 = getelementptr inbounds nuw double, ptr %1, i64 %244
  %258 = load <4 x double>, ptr %257, align 8, !tbaa !6
  %259 = shufflevector <4 x double> %258, <4 x double> poison, <2 x i32> <i32 0, i32 2>
  %260 = shufflevector <4 x double> %258, <4 x double> poison, <2 x i32> <i32 1, i32 3>
  %261 = getelementptr inbounds nuw double, ptr %87, i64 %244
  %262 = load <4 x double>, ptr %261, align 8, !tbaa !6
  %263 = shufflevector <4 x double> %262, <4 x double> poison, <2 x i32> <i32 0, i32 2>
  %264 = shufflevector <4 x double> %262, <4 x double> poison, <2 x i32> <i32 1, i32 3>
  %265 = fadd <2 x double> %259, %263
  %266 = fadd <2 x double> %260, %264
  %267 = fsub <2 x double> %259, %263
  %268 = fsub <2 x double> %260, %264
  %269 = fadd <2 x double> %253, %265
  %270 = fadd <2 x double> %254, %266
  %271 = shufflevector <2 x double> %269, <2 x double> %270, <4 x i32> <i32 0, i32 2, i32 1, i32 3>
  store <4 x double> %271, ptr %245, align 8, !tbaa !6
  %272 = fsub <2 x double> %253, %265
  %273 = fsub <2 x double> %254, %266
  %274 = shufflevector <2 x double> %272, <2 x double> %273, <4 x i32> <i32 0, i32 2, i32 1, i32 3>
  store <4 x double> %274, ptr %257, align 8, !tbaa !6
  %275 = fsub <2 x double> %255, %268
  %276 = fadd <2 x double> %256, %267
  %277 = shufflevector <2 x double> %275, <2 x double> %276, <4 x i32> <i32 0, i32 2, i32 1, i32 3>
  store <4 x double> %277, ptr %249, align 8, !tbaa !6
  %278 = fadd <2 x double> %255, %268
  %279 = fsub <2 x double> %256, %267
  %280 = shufflevector <2 x double> %278, <2 x double> %279, <4 x i32> <i32 0, i32 2, i32 1, i32 3>
  store <4 x double> %280, ptr %261, align 8, !tbaa !6
  %281 = add nuw i64 %241, 2
  %282 = icmp eq i64 %281, %238
  br i1 %282, label %283, label %240, !llvm.loop !47

283:                                              ; preds = %240
  %284 = icmp eq i64 %90, %238
  br i1 %284, label %341, label %285

285:                                              ; preds = %92, %84, %283
  %286 = phi i64 [ 0, %92 ], [ 0, %84 ], [ %239, %283 ]
  br label %287

287:                                              ; preds = %285, %287
  %288 = phi i64 [ %323, %287 ], [ %286, %285 ]
  %289 = add nuw nsw i64 %288, %85
  %290 = add nuw nsw i64 %289, %85
  %291 = getelementptr inbounds nuw double, ptr %1, i64 %288
  %292 = load double, ptr %291, align 8, !tbaa !6
  %293 = getelementptr inbounds nuw double, ptr %1, i64 %289
  %294 = load double, ptr %293, align 8, !tbaa !6
  %295 = fadd double %292, %294
  %296 = getelementptr inbounds nuw i8, ptr %291, i64 8
  %297 = load double, ptr %296, align 8, !tbaa !6
  %298 = getelementptr i8, ptr %293, i64 8
  %299 = load double, ptr %298, align 8, !tbaa !6
  %300 = fadd double %297, %299
  %301 = fsub double %292, %294
  %302 = fsub double %297, %299
  %303 = getelementptr inbounds nuw double, ptr %1, i64 %290
  %304 = load double, ptr %303, align 8, !tbaa !6
  %305 = getelementptr inbounds nuw double, ptr %87, i64 %290
  %306 = load double, ptr %305, align 8, !tbaa !6
  %307 = fadd double %304, %306
  %308 = getelementptr i8, ptr %303, i64 8
  %309 = load double, ptr %308, align 8, !tbaa !6
  %310 = getelementptr i8, ptr %305, i64 8
  %311 = load double, ptr %310, align 8, !tbaa !6
  %312 = fadd double %309, %311
  %313 = fsub double %304, %306
  %314 = fsub double %309, %311
  %315 = fadd double %295, %307
  store double %315, ptr %291, align 8, !tbaa !6
  %316 = fadd double %300, %312
  store double %316, ptr %296, align 8, !tbaa !6
  %317 = fsub double %295, %307
  store double %317, ptr %303, align 8, !tbaa !6
  %318 = fsub double %300, %312
  store double %318, ptr %308, align 8, !tbaa !6
  %319 = fsub double %301, %314
  store double %319, ptr %293, align 8, !tbaa !6
  %320 = fadd double %302, %313
  store double %320, ptr %298, align 8, !tbaa !6
  %321 = fadd double %301, %314
  store double %321, ptr %305, align 8, !tbaa !6
  %322 = fsub double %302, %313
  store double %322, ptr %310, align 8, !tbaa !6
  %323 = add nuw nsw i64 %288, 2
  %324 = icmp samesign ult i64 %323, %86
  br i1 %324, label %287, label %341, !llvm.loop !48

325:                                              ; preds = %81, %325
  %326 = phi i64 [ %339, %325 ], [ %82, %81 ]
  %327 = getelementptr inbounds nuw double, ptr %1, i64 %326
  %328 = load double, ptr %327, align 8, !tbaa !6
  %329 = getelementptr inbounds nuw double, ptr %21, i64 %326
  %330 = load double, ptr %329, align 8, !tbaa !6
  %331 = fsub double %328, %330
  %332 = getelementptr inbounds nuw i8, ptr %327, i64 8
  %333 = load double, ptr %332, align 8, !tbaa !6
  %334 = getelementptr i8, ptr %329, i64 8
  %335 = load double, ptr %334, align 8, !tbaa !6
  %336 = fsub double %333, %335
  %337 = fadd double %328, %330
  store double %337, ptr %327, align 8, !tbaa !6
  %338 = fadd double %333, %335
  store double %338, ptr %332, align 8, !tbaa !6
  store double %331, ptr %329, align 8, !tbaa !6
  store double %336, ptr %334, align 8, !tbaa !6
  %339 = add nuw nsw i64 %326, 2
  %340 = icmp samesign ult i64 %339, %20
  br i1 %340, label %325, label %341, !llvm.loop !49

341:                                              ; preds = %325, %287, %79, %283, %17, %83
  ret void
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(errnomem: write)
declare double @cos(double noundef) local_unnamed_addr #14

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(errnomem: write)
declare double @sin(double noundef) local_unnamed_addr #14

; Function Attrs: inlinehint nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable
define internal fastcc void @cft1st(i32 noundef range(i32 9, -2147483648) %0, ptr noundef captures(none) %1, ptr noundef readonly captures(none) %2) unnamed_addr #15 {
  %4 = load double, ptr %1, align 8, !tbaa !6
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %6 = load double, ptr %5, align 8, !tbaa !6
  %7 = fadd double %4, %6
  %8 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %9 = load double, ptr %8, align 8, !tbaa !6
  %10 = getelementptr inbounds nuw i8, ptr %1, i64 24
  %11 = load double, ptr %10, align 8, !tbaa !6
  %12 = fadd double %9, %11
  %13 = fsub double %4, %6
  %14 = fsub double %9, %11
  %15 = getelementptr inbounds nuw i8, ptr %1, i64 32
  %16 = load double, ptr %15, align 8, !tbaa !6
  %17 = getelementptr inbounds nuw i8, ptr %1, i64 48
  %18 = load double, ptr %17, align 8, !tbaa !6
  %19 = fadd double %16, %18
  %20 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %21 = load double, ptr %20, align 8, !tbaa !6
  %22 = getelementptr inbounds nuw i8, ptr %1, i64 56
  %23 = load double, ptr %22, align 8, !tbaa !6
  %24 = fadd double %21, %23
  %25 = fsub double %16, %18
  %26 = fsub double %21, %23
  %27 = fadd double %7, %19
  store double %27, ptr %1, align 8, !tbaa !6
  %28 = fadd double %12, %24
  store double %28, ptr %8, align 8, !tbaa !6
  %29 = fsub double %7, %19
  store double %29, ptr %15, align 8, !tbaa !6
  %30 = fsub double %12, %24
  store double %30, ptr %20, align 8, !tbaa !6
  %31 = fsub double %13, %26
  store double %31, ptr %5, align 8, !tbaa !6
  %32 = fadd double %14, %25
  store double %32, ptr %10, align 8, !tbaa !6
  %33 = fadd double %13, %26
  store double %33, ptr %17, align 8, !tbaa !6
  %34 = fsub double %14, %25
  store double %34, ptr %22, align 8, !tbaa !6
  %35 = getelementptr inbounds nuw i8, ptr %2, i64 16
  %36 = load double, ptr %35, align 8, !tbaa !6
  %37 = getelementptr inbounds nuw i8, ptr %1, i64 64
  %38 = load double, ptr %37, align 8, !tbaa !6
  %39 = getelementptr inbounds nuw i8, ptr %1, i64 80
  %40 = load double, ptr %39, align 8, !tbaa !6
  %41 = fadd double %38, %40
  %42 = getelementptr inbounds nuw i8, ptr %1, i64 72
  %43 = load double, ptr %42, align 8, !tbaa !6
  %44 = getelementptr inbounds nuw i8, ptr %1, i64 88
  %45 = load double, ptr %44, align 8, !tbaa !6
  %46 = fadd double %43, %45
  %47 = fsub double %38, %40
  %48 = fsub double %43, %45
  %49 = getelementptr inbounds nuw i8, ptr %1, i64 96
  %50 = load double, ptr %49, align 8, !tbaa !6
  %51 = getelementptr inbounds nuw i8, ptr %1, i64 112
  %52 = load double, ptr %51, align 8, !tbaa !6
  %53 = fadd double %50, %52
  %54 = getelementptr inbounds nuw i8, ptr %1, i64 104
  %55 = load double, ptr %54, align 8, !tbaa !6
  %56 = getelementptr inbounds nuw i8, ptr %1, i64 120
  %57 = load double, ptr %56, align 8, !tbaa !6
  %58 = fadd double %55, %57
  %59 = fsub double %50, %52
  %60 = fsub double %55, %57
  %61 = fadd double %41, %53
  store double %61, ptr %37, align 8, !tbaa !6
  %62 = fadd double %46, %58
  store double %62, ptr %42, align 8, !tbaa !6
  %63 = fsub double %58, %46
  store double %63, ptr %49, align 8, !tbaa !6
  %64 = fsub double %41, %53
  store double %64, ptr %54, align 8, !tbaa !6
  %65 = fsub double %47, %60
  %66 = fadd double %48, %59
  %67 = fsub double %65, %66
  %68 = fmul double %36, %67
  store double %68, ptr %39, align 8, !tbaa !6
  %69 = fadd double %66, %65
  %70 = fmul double %36, %69
  store double %70, ptr %44, align 8, !tbaa !6
  %71 = fadd double %47, %60
  %72 = fsub double %59, %48
  %73 = fsub double %72, %71
  %74 = fmul double %36, %73
  store double %74, ptr %51, align 8, !tbaa !6
  %75 = fadd double %72, %71
  %76 = fmul double %36, %75
  store double %76, ptr %56, align 8, !tbaa !6
  %77 = icmp samesign ugt i32 %0, 16
  br i1 %77, label %78, label %205

78:                                               ; preds = %3
  %79 = zext nneg i32 %0 to i64
  br label %80

80:                                               ; preds = %78, %80
  %81 = phi i64 [ 0, %78 ], [ %83, %80 ]
  %82 = phi i64 [ 16, %78 ], [ %203, %80 ]
  %83 = add nuw nsw i64 %81, 2
  %84 = getelementptr inbounds nuw double, ptr %2, i64 %83
  %85 = load double, ptr %84, align 8, !tbaa !6
  %86 = getelementptr inbounds nuw double, ptr %2, i64 %81
  %87 = getelementptr inbounds nuw i8, ptr %86, i64 24
  %88 = load double, ptr %87, align 8, !tbaa !6
  %89 = shl nuw nsw i64 %83, 4
  %90 = getelementptr inbounds nuw i8, ptr %2, i64 %89
  %91 = load double, ptr %90, align 8, !tbaa !6
  %92 = getelementptr inbounds nuw i8, ptr %90, i64 8
  %93 = load double, ptr %92, align 8, !tbaa !6
  %94 = fmul double %88, 2.000000e+00
  %95 = fneg double %94
  %96 = tail call double @llvm.fmuladd.f64(double %95, double %93, double %91)
  %97 = fneg double %93
  %98 = tail call double @llvm.fmuladd.f64(double %94, double %91, double %97)
  %99 = getelementptr inbounds nuw double, ptr %1, i64 %82
  %100 = load double, ptr %99, align 8, !tbaa !6
  %101 = getelementptr inbounds nuw i8, ptr %99, i64 16
  %102 = load double, ptr %101, align 8, !tbaa !6
  %103 = fadd double %100, %102
  %104 = getelementptr inbounds nuw i8, ptr %99, i64 8
  %105 = load double, ptr %104, align 8, !tbaa !6
  %106 = getelementptr inbounds nuw i8, ptr %99, i64 24
  %107 = load double, ptr %106, align 8, !tbaa !6
  %108 = fadd double %105, %107
  %109 = fsub double %100, %102
  %110 = fsub double %105, %107
  %111 = getelementptr inbounds nuw i8, ptr %99, i64 32
  %112 = load double, ptr %111, align 8, !tbaa !6
  %113 = getelementptr inbounds nuw i8, ptr %99, i64 48
  %114 = load double, ptr %113, align 8, !tbaa !6
  %115 = fadd double %112, %114
  %116 = getelementptr inbounds nuw i8, ptr %99, i64 40
  %117 = load double, ptr %116, align 8, !tbaa !6
  %118 = getelementptr inbounds nuw i8, ptr %99, i64 56
  %119 = load double, ptr %118, align 8, !tbaa !6
  %120 = fadd double %117, %119
  %121 = fsub double %112, %114
  %122 = fsub double %117, %119
  %123 = fadd double %103, %115
  store double %123, ptr %99, align 8, !tbaa !6
  %124 = fadd double %108, %120
  store double %124, ptr %104, align 8, !tbaa !6
  %125 = fsub double %103, %115
  %126 = fsub double %108, %120
  %127 = fneg double %126
  %128 = fmul double %88, %127
  %129 = tail call double @llvm.fmuladd.f64(double %85, double %125, double %128)
  store double %129, ptr %111, align 8, !tbaa !6
  %130 = fmul double %88, %125
  %131 = tail call double @llvm.fmuladd.f64(double %85, double %126, double %130)
  store double %131, ptr %116, align 8, !tbaa !6
  %132 = fsub double %109, %122
  %133 = fadd double %110, %121
  %134 = fneg double %133
  %135 = fmul double %93, %134
  %136 = tail call double @llvm.fmuladd.f64(double %91, double %132, double %135)
  store double %136, ptr %101, align 8, !tbaa !6
  %137 = fmul double %93, %132
  %138 = tail call double @llvm.fmuladd.f64(double %91, double %133, double %137)
  store double %138, ptr %106, align 8, !tbaa !6
  %139 = fadd double %109, %122
  %140 = fsub double %110, %121
  %141 = fneg double %140
  %142 = fmul double %98, %141
  %143 = tail call double @llvm.fmuladd.f64(double %96, double %139, double %142)
  store double %143, ptr %113, align 8, !tbaa !6
  %144 = fmul double %98, %139
  %145 = tail call double @llvm.fmuladd.f64(double %96, double %140, double %144)
  store double %145, ptr %118, align 8, !tbaa !6
  %146 = getelementptr inbounds nuw i8, ptr %90, i64 16
  %147 = load double, ptr %146, align 8, !tbaa !6
  %148 = getelementptr inbounds nuw i8, ptr %90, i64 24
  %149 = load double, ptr %148, align 8, !tbaa !6
  %150 = fmul double %85, 2.000000e+00
  %151 = fneg double %150
  %152 = tail call double @llvm.fmuladd.f64(double %151, double %149, double %147)
  %153 = fneg double %149
  %154 = tail call double @llvm.fmuladd.f64(double %150, double %147, double %153)
  %155 = getelementptr inbounds nuw i8, ptr %99, i64 64
  %156 = load double, ptr %155, align 8, !tbaa !6
  %157 = getelementptr inbounds nuw i8, ptr %99, i64 80
  %158 = load double, ptr %157, align 8, !tbaa !6
  %159 = fadd double %156, %158
  %160 = getelementptr inbounds nuw i8, ptr %99, i64 72
  %161 = load double, ptr %160, align 8, !tbaa !6
  %162 = getelementptr inbounds nuw i8, ptr %99, i64 88
  %163 = load double, ptr %162, align 8, !tbaa !6
  %164 = fadd double %161, %163
  %165 = fsub double %156, %158
  %166 = fsub double %161, %163
  %167 = getelementptr inbounds nuw i8, ptr %99, i64 96
  %168 = load double, ptr %167, align 8, !tbaa !6
  %169 = getelementptr inbounds nuw i8, ptr %99, i64 112
  %170 = load double, ptr %169, align 8, !tbaa !6
  %171 = fadd double %168, %170
  %172 = getelementptr inbounds nuw i8, ptr %99, i64 104
  %173 = load double, ptr %172, align 8, !tbaa !6
  %174 = getelementptr inbounds nuw i8, ptr %99, i64 120
  %175 = load double, ptr %174, align 8, !tbaa !6
  %176 = fadd double %173, %175
  %177 = fsub double %168, %170
  %178 = fsub double %173, %175
  %179 = fadd double %159, %171
  store double %179, ptr %155, align 8, !tbaa !6
  %180 = fadd double %164, %176
  store double %180, ptr %160, align 8, !tbaa !6
  %181 = fsub double %159, %171
  %182 = fsub double %164, %176
  %183 = fneg double %88
  %184 = fneg double %182
  %185 = fmul double %85, %184
  %186 = tail call double @llvm.fmuladd.f64(double %183, double %181, double %185)
  store double %186, ptr %167, align 8, !tbaa !6
  %187 = fmul double %85, %181
  %188 = tail call double @llvm.fmuladd.f64(double %183, double %182, double %187)
  store double %188, ptr %172, align 8, !tbaa !6
  %189 = fsub double %165, %178
  %190 = fadd double %166, %177
  %191 = fneg double %190
  %192 = fmul double %149, %191
  %193 = tail call double @llvm.fmuladd.f64(double %147, double %189, double %192)
  store double %193, ptr %157, align 8, !tbaa !6
  %194 = fmul double %149, %189
  %195 = tail call double @llvm.fmuladd.f64(double %147, double %190, double %194)
  store double %195, ptr %162, align 8, !tbaa !6
  %196 = fadd double %165, %178
  %197 = fsub double %166, %177
  %198 = fneg double %197
  %199 = fmul double %154, %198
  %200 = tail call double @llvm.fmuladd.f64(double %152, double %196, double %199)
  store double %200, ptr %169, align 8, !tbaa !6
  %201 = fmul double %154, %196
  %202 = tail call double @llvm.fmuladd.f64(double %152, double %197, double %201)
  store double %202, ptr %174, align 8, !tbaa !6
  %203 = add nuw nsw i64 %82, 16
  %204 = icmp samesign ult i64 %203, %79
  br i1 %204, label %80, label %205, !llvm.loop !50

205:                                              ; preds = %80, %3
  ret void
}

; Function Attrs: inlinehint nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable
define internal fastcc void @cftmdl(i32 noundef range(i32 9, -2147483648) %0, i32 noundef %1, ptr noundef captures(none) %2, ptr noundef readonly captures(none) %3) unnamed_addr #15 {
  %5 = shl i32 %1, 2
  %6 = icmp sgt i32 %1, 0
  br i1 %6, label %7, label %249

7:                                                ; preds = %4
  %8 = zext nneg i32 %1 to i64
  %9 = zext nneg i32 %1 to i64
  %10 = getelementptr double, ptr %2, i64 %8
  %11 = add nsw i64 %9, -1
  %12 = lshr i64 %11, 1
  %13 = add nuw i64 %12, 1
  %14 = icmp ult i32 %1, 51
  br i1 %14, label %209, label %15

15:                                               ; preds = %7
  %16 = shl nuw nsw i64 %9, 3
  %17 = add nsw i64 %16, -8
  %18 = and i64 %17, -16
  %19 = or i64 %17, 8
  %20 = getelementptr i8, ptr %2, i64 %19
  %21 = getelementptr i8, ptr %2, i64 8
  %22 = getelementptr i8, ptr %2, i64 %18
  %23 = getelementptr i8, ptr %22, i64 16
  %24 = shl nuw nsw i64 %9, 4
  %25 = getelementptr i8, ptr %2, i64 %24
  %26 = add nsw i64 %18, %24
  %27 = getelementptr i8, ptr %2, i64 %26
  %28 = getelementptr i8, ptr %27, i64 8
  %29 = getelementptr i8, ptr %2, i64 %24
  %30 = getelementptr i8, ptr %29, i64 8
  %31 = getelementptr i8, ptr %2, i64 %26
  %32 = getelementptr i8, ptr %31, i64 16
  %33 = shl nuw nsw i64 %9, 3
  %34 = add nsw i64 %18, %33
  %35 = getelementptr i8, ptr %2, i64 %34
  %36 = getelementptr i8, ptr %35, i64 8
  %37 = getelementptr i8, ptr %2, i64 %33
  %38 = getelementptr i8, ptr %37, i64 8
  %39 = getelementptr i8, ptr %2, i64 %34
  %40 = getelementptr i8, ptr %39, i64 16
  %41 = mul nuw nsw i64 %9, 24
  %42 = getelementptr i8, ptr %2, i64 %41
  %43 = add nsw i64 %41, %18
  %44 = getelementptr i8, ptr %2, i64 %43
  %45 = getelementptr i8, ptr %44, i64 8
  %46 = getelementptr i8, ptr %2, i64 %41
  %47 = getelementptr i8, ptr %46, i64 8
  %48 = getelementptr i8, ptr %2, i64 %43
  %49 = getelementptr i8, ptr %48, i64 16
  %50 = icmp ult ptr %2, %23
  %51 = icmp ult ptr %21, %20
  %52 = and i1 %50, %51
  %53 = icmp ult ptr %2, %28
  %54 = icmp ult ptr %25, %20
  %55 = and i1 %53, %54
  %56 = or i1 %52, %55
  %57 = icmp ult ptr %2, %32
  %58 = icmp ult ptr %30, %20
  %59 = and i1 %57, %58
  %60 = or i1 %56, %59
  %61 = icmp ult ptr %2, %36
  %62 = icmp ult ptr %10, %20
  %63 = and i1 %61, %62
  %64 = or i1 %60, %63
  %65 = icmp ult ptr %2, %40
  %66 = icmp ult ptr %38, %20
  %67 = and i1 %65, %66
  %68 = or i1 %64, %67
  %69 = icmp ult ptr %2, %45
  %70 = icmp ult ptr %42, %20
  %71 = and i1 %69, %70
  %72 = or i1 %68, %71
  %73 = icmp ult ptr %2, %49
  %74 = icmp ult ptr %47, %20
  %75 = and i1 %73, %74
  %76 = or i1 %72, %75
  %77 = icmp ult ptr %21, %28
  %78 = icmp ult ptr %25, %23
  %79 = and i1 %77, %78
  %80 = or i1 %76, %79
  %81 = icmp ult ptr %21, %32
  %82 = icmp ult ptr %30, %23
  %83 = and i1 %81, %82
  %84 = or i1 %80, %83
  %85 = icmp ult ptr %21, %36
  %86 = icmp ult ptr %10, %23
  %87 = and i1 %85, %86
  %88 = or i1 %84, %87
  %89 = icmp ult ptr %21, %40
  %90 = icmp ult ptr %38, %23
  %91 = and i1 %89, %90
  %92 = or i1 %88, %91
  %93 = icmp ult ptr %21, %45
  %94 = icmp ult ptr %42, %23
  %95 = and i1 %93, %94
  %96 = or i1 %92, %95
  %97 = icmp ult ptr %21, %49
  %98 = icmp ult ptr %47, %23
  %99 = and i1 %97, %98
  %100 = or i1 %96, %99
  %101 = icmp ult ptr %25, %32
  %102 = icmp ult ptr %30, %28
  %103 = and i1 %101, %102
  %104 = or i1 %100, %103
  %105 = icmp ult ptr %25, %36
  %106 = icmp ult ptr %10, %28
  %107 = and i1 %105, %106
  %108 = or i1 %104, %107
  %109 = icmp ult ptr %25, %40
  %110 = icmp ult ptr %38, %28
  %111 = and i1 %109, %110
  %112 = or i1 %108, %111
  %113 = icmp ult ptr %25, %45
  %114 = icmp ult ptr %42, %28
  %115 = and i1 %113, %114
  %116 = or i1 %112, %115
  %117 = icmp ult ptr %25, %49
  %118 = icmp ult ptr %47, %28
  %119 = and i1 %117, %118
  %120 = or i1 %116, %119
  %121 = icmp ult ptr %30, %36
  %122 = icmp ult ptr %10, %32
  %123 = and i1 %121, %122
  %124 = or i1 %120, %123
  %125 = icmp ult ptr %30, %40
  %126 = icmp ult ptr %38, %32
  %127 = and i1 %125, %126
  %128 = or i1 %124, %127
  %129 = icmp ult ptr %30, %45
  %130 = icmp ult ptr %42, %32
  %131 = and i1 %129, %130
  %132 = or i1 %128, %131
  %133 = icmp ult ptr %30, %49
  %134 = icmp ult ptr %47, %32
  %135 = and i1 %133, %134
  %136 = or i1 %132, %135
  %137 = icmp ult ptr %10, %40
  %138 = icmp ult ptr %38, %36
  %139 = and i1 %137, %138
  %140 = or i1 %136, %139
  %141 = icmp ult ptr %10, %45
  %142 = icmp ult ptr %42, %36
  %143 = and i1 %141, %142
  %144 = or i1 %140, %143
  %145 = icmp ult ptr %10, %49
  %146 = icmp ult ptr %47, %36
  %147 = and i1 %145, %146
  %148 = or i1 %144, %147
  %149 = icmp ult ptr %38, %45
  %150 = icmp ult ptr %42, %40
  %151 = and i1 %149, %150
  %152 = or i1 %148, %151
  %153 = icmp ult ptr %38, %49
  %154 = icmp ult ptr %47, %40
  %155 = and i1 %153, %154
  %156 = or i1 %152, %155
  %157 = icmp ult ptr %42, %49
  %158 = icmp ult ptr %47, %45
  %159 = and i1 %157, %158
  %160 = or i1 %156, %159
  br i1 %160, label %209, label %161

161:                                              ; preds = %15
  %162 = and i64 %13, -2
  %163 = shl i64 %162, 1
  br label %164

164:                                              ; preds = %164, %161
  %165 = phi i64 [ 0, %161 ], [ %205, %164 ]
  %166 = shl i64 %165, 1
  %167 = add nuw nsw i64 %166, %8
  %168 = add nuw nsw i64 %167, %8
  %169 = getelementptr inbounds nuw double, ptr %2, i64 %166
  %170 = load <4 x double>, ptr %169, align 8, !tbaa !6
  %171 = shufflevector <4 x double> %170, <4 x double> poison, <2 x i32> <i32 0, i32 2>
  %172 = shufflevector <4 x double> %170, <4 x double> poison, <2 x i32> <i32 1, i32 3>
  %173 = getelementptr inbounds nuw double, ptr %2, i64 %167
  %174 = load <4 x double>, ptr %173, align 8, !tbaa !6
  %175 = shufflevector <4 x double> %174, <4 x double> poison, <2 x i32> <i32 0, i32 2>
  %176 = shufflevector <4 x double> %174, <4 x double> poison, <2 x i32> <i32 1, i32 3>
  %177 = fadd <2 x double> %171, %175
  %178 = fadd <2 x double> %172, %176
  %179 = fsub <2 x double> %171, %175
  %180 = fsub <2 x double> %172, %176
  %181 = getelementptr inbounds nuw double, ptr %2, i64 %168
  %182 = load <4 x double>, ptr %181, align 8, !tbaa !6
  %183 = shufflevector <4 x double> %182, <4 x double> poison, <2 x i32> <i32 0, i32 2>
  %184 = shufflevector <4 x double> %182, <4 x double> poison, <2 x i32> <i32 1, i32 3>
  %185 = getelementptr inbounds nuw double, ptr %10, i64 %168
  %186 = load <4 x double>, ptr %185, align 8, !tbaa !6
  %187 = shufflevector <4 x double> %186, <4 x double> poison, <2 x i32> <i32 0, i32 2>
  %188 = shufflevector <4 x double> %186, <4 x double> poison, <2 x i32> <i32 1, i32 3>
  %189 = fadd <2 x double> %183, %187
  %190 = fadd <2 x double> %184, %188
  %191 = fsub <2 x double> %183, %187
  %192 = fsub <2 x double> %184, %188
  %193 = fadd <2 x double> %177, %189
  %194 = fadd <2 x double> %178, %190
  %195 = shufflevector <2 x double> %193, <2 x double> %194, <4 x i32> <i32 0, i32 2, i32 1, i32 3>
  store <4 x double> %195, ptr %169, align 8, !tbaa !6
  %196 = fsub <2 x double> %177, %189
  %197 = fsub <2 x double> %178, %190
  %198 = shufflevector <2 x double> %196, <2 x double> %197, <4 x i32> <i32 0, i32 2, i32 1, i32 3>
  store <4 x double> %198, ptr %181, align 8, !tbaa !6
  %199 = fsub <2 x double> %179, %192
  %200 = fadd <2 x double> %180, %191
  %201 = shufflevector <2 x double> %199, <2 x double> %200, <4 x i32> <i32 0, i32 2, i32 1, i32 3>
  store <4 x double> %201, ptr %173, align 8, !tbaa !6
  %202 = fadd <2 x double> %179, %192
  %203 = fsub <2 x double> %180, %191
  %204 = shufflevector <2 x double> %202, <2 x double> %203, <4 x i32> <i32 0, i32 2, i32 1, i32 3>
  store <4 x double> %204, ptr %185, align 8, !tbaa !6
  %205 = add nuw i64 %165, 2
  %206 = icmp eq i64 %205, %162
  br i1 %206, label %207, label %164, !llvm.loop !51

207:                                              ; preds = %164
  %208 = icmp eq i64 %13, %162
  br i1 %208, label %249, label %209

209:                                              ; preds = %15, %7, %207
  %210 = phi i64 [ 0, %15 ], [ 0, %7 ], [ %163, %207 ]
  br label %211

211:                                              ; preds = %209, %211
  %212 = phi i64 [ %247, %211 ], [ %210, %209 ]
  %213 = add nuw nsw i64 %212, %8
  %214 = add nuw nsw i64 %213, %8
  %215 = getelementptr inbounds nuw double, ptr %2, i64 %212
  %216 = load double, ptr %215, align 8, !tbaa !6
  %217 = getelementptr inbounds nuw double, ptr %2, i64 %213
  %218 = load double, ptr %217, align 8, !tbaa !6
  %219 = fadd double %216, %218
  %220 = getelementptr inbounds nuw i8, ptr %215, i64 8
  %221 = load double, ptr %220, align 8, !tbaa !6
  %222 = getelementptr i8, ptr %217, i64 8
  %223 = load double, ptr %222, align 8, !tbaa !6
  %224 = fadd double %221, %223
  %225 = fsub double %216, %218
  %226 = fsub double %221, %223
  %227 = getelementptr inbounds nuw double, ptr %2, i64 %214
  %228 = load double, ptr %227, align 8, !tbaa !6
  %229 = getelementptr inbounds nuw double, ptr %10, i64 %214
  %230 = load double, ptr %229, align 8, !tbaa !6
  %231 = fadd double %228, %230
  %232 = getelementptr i8, ptr %227, i64 8
  %233 = load double, ptr %232, align 8, !tbaa !6
  %234 = getelementptr i8, ptr %229, i64 8
  %235 = load double, ptr %234, align 8, !tbaa !6
  %236 = fadd double %233, %235
  %237 = fsub double %228, %230
  %238 = fsub double %233, %235
  %239 = fadd double %219, %231
  store double %239, ptr %215, align 8, !tbaa !6
  %240 = fadd double %224, %236
  store double %240, ptr %220, align 8, !tbaa !6
  %241 = fsub double %219, %231
  store double %241, ptr %227, align 8, !tbaa !6
  %242 = fsub double %224, %236
  store double %242, ptr %232, align 8, !tbaa !6
  %243 = fsub double %225, %238
  store double %243, ptr %217, align 8, !tbaa !6
  %244 = fadd double %226, %237
  store double %244, ptr %222, align 8, !tbaa !6
  %245 = fadd double %225, %238
  store double %245, ptr %229, align 8, !tbaa !6
  %246 = fsub double %226, %237
  store double %246, ptr %234, align 8, !tbaa !6
  %247 = add nuw nsw i64 %212, 2
  %248 = icmp samesign ult i64 %247, %9
  br i1 %248, label %211, label %249, !llvm.loop !52

249:                                              ; preds = %211, %207, %4
  %250 = getelementptr inbounds nuw i8, ptr %3, i64 16
  %251 = load double, ptr %250, align 8, !tbaa !6
  %252 = mul i32 %1, 5
  %253 = icmp slt i32 %5, %252
  br i1 %253, label %254, label %529

254:                                              ; preds = %249
  %255 = sext i32 %5 to i64
  %256 = sext i32 %1 to i64
  %257 = sext i32 %252 to i64
  %258 = getelementptr double, ptr %2, i64 %256
  %259 = xor i64 %255, -1
  %260 = add nsw i64 %259, %257
  %261 = lshr i64 %260, 1
  %262 = add nuw i64 %261, 1
  %263 = icmp ult i64 %260, 42
  br i1 %263, label %481, label %264

264:                                              ; preds = %254
  %265 = shl nsw i64 %255, 3
  %266 = getelementptr i8, ptr %2, i64 %265
  %267 = xor i64 %255, -1
  %268 = add nsw i64 %267, %257
  %269 = shl nsw i64 %268, 3
  %270 = and i64 %269, -16
  %271 = add nsw i64 %270, %265
  %272 = getelementptr i8, ptr %2, i64 %271
  %273 = getelementptr i8, ptr %272, i64 8
  %274 = getelementptr i8, ptr %2, i64 %265
  %275 = getelementptr i8, ptr %274, i64 8
  %276 = getelementptr i8, ptr %2, i64 %271
  %277 = getelementptr i8, ptr %276, i64 16
  %278 = shl nsw i64 %256, 4
  %279 = add nsw i64 %278, %265
  %280 = getelementptr i8, ptr %2, i64 %279
  %281 = add nsw i64 %270, %278
  %282 = add nsw i64 %281, %265
  %283 = getelementptr i8, ptr %2, i64 %282
  %284 = getelementptr i8, ptr %283, i64 8
  %285 = getelementptr i8, ptr %2, i64 %279
  %286 = getelementptr i8, ptr %285, i64 8
  %287 = getelementptr i8, ptr %2, i64 %282
  %288 = getelementptr i8, ptr %287, i64 16
  %289 = add nsw i64 %256, %255
  %290 = shl nsw i64 %289, 3
  %291 = getelementptr i8, ptr %2, i64 %290
  %292 = add nsw i64 %270, %290
  %293 = getelementptr i8, ptr %2, i64 %292
  %294 = getelementptr i8, ptr %293, i64 8
  %295 = getelementptr i8, ptr %2, i64 %290
  %296 = getelementptr i8, ptr %295, i64 8
  %297 = getelementptr i8, ptr %2, i64 %292
  %298 = getelementptr i8, ptr %297, i64 16
  %299 = mul nsw i64 %256, 24
  %300 = add nsw i64 %299, %265
  %301 = getelementptr i8, ptr %2, i64 %300
  %302 = add nsw i64 %299, %270
  %303 = add nsw i64 %302, %265
  %304 = getelementptr i8, ptr %2, i64 %303
  %305 = getelementptr i8, ptr %304, i64 8
  %306 = getelementptr i8, ptr %2, i64 %300
  %307 = getelementptr i8, ptr %306, i64 8
  %308 = getelementptr i8, ptr %2, i64 %303
  %309 = getelementptr i8, ptr %308, i64 16
  %310 = icmp ult ptr %266, %277
  %311 = icmp ult ptr %275, %273
  %312 = and i1 %310, %311
  %313 = icmp ult ptr %266, %284
  %314 = icmp ult ptr %280, %273
  %315 = and i1 %313, %314
  %316 = or i1 %312, %315
  %317 = icmp ult ptr %266, %288
  %318 = icmp ult ptr %286, %273
  %319 = and i1 %317, %318
  %320 = or i1 %316, %319
  %321 = icmp ult ptr %266, %294
  %322 = icmp ult ptr %291, %273
  %323 = and i1 %321, %322
  %324 = or i1 %320, %323
  %325 = icmp ult ptr %266, %298
  %326 = icmp ult ptr %296, %273
  %327 = and i1 %325, %326
  %328 = or i1 %324, %327
  %329 = icmp ult ptr %266, %305
  %330 = icmp ult ptr %301, %273
  %331 = and i1 %329, %330
  %332 = or i1 %328, %331
  %333 = icmp ult ptr %266, %309
  %334 = icmp ult ptr %307, %273
  %335 = and i1 %333, %334
  %336 = or i1 %332, %335
  %337 = icmp ult ptr %275, %284
  %338 = icmp ult ptr %280, %277
  %339 = and i1 %337, %338
  %340 = or i1 %336, %339
  %341 = icmp ult ptr %275, %288
  %342 = icmp ult ptr %286, %277
  %343 = and i1 %341, %342
  %344 = or i1 %340, %343
  %345 = icmp ult ptr %275, %294
  %346 = icmp ult ptr %291, %277
  %347 = and i1 %345, %346
  %348 = or i1 %344, %347
  %349 = icmp ult ptr %275, %298
  %350 = icmp ult ptr %296, %277
  %351 = and i1 %349, %350
  %352 = or i1 %348, %351
  %353 = icmp ult ptr %275, %305
  %354 = icmp ult ptr %301, %277
  %355 = and i1 %353, %354
  %356 = or i1 %352, %355
  %357 = icmp ult ptr %275, %309
  %358 = icmp ult ptr %307, %277
  %359 = and i1 %357, %358
  %360 = or i1 %356, %359
  %361 = icmp ult ptr %280, %288
  %362 = icmp ult ptr %286, %284
  %363 = and i1 %361, %362
  %364 = or i1 %360, %363
  %365 = icmp ult ptr %280, %294
  %366 = icmp ult ptr %291, %284
  %367 = and i1 %365, %366
  %368 = or i1 %364, %367
  %369 = icmp ult ptr %280, %298
  %370 = icmp ult ptr %296, %284
  %371 = and i1 %369, %370
  %372 = or i1 %368, %371
  %373 = icmp ult ptr %280, %305
  %374 = icmp ult ptr %301, %284
  %375 = and i1 %373, %374
  %376 = or i1 %372, %375
  %377 = icmp ult ptr %280, %309
  %378 = icmp ult ptr %307, %284
  %379 = and i1 %377, %378
  %380 = or i1 %376, %379
  %381 = icmp ult ptr %286, %294
  %382 = icmp ult ptr %291, %288
  %383 = and i1 %381, %382
  %384 = or i1 %380, %383
  %385 = icmp ult ptr %286, %298
  %386 = icmp ult ptr %296, %288
  %387 = and i1 %385, %386
  %388 = or i1 %384, %387
  %389 = icmp ult ptr %286, %305
  %390 = icmp ult ptr %301, %288
  %391 = and i1 %389, %390
  %392 = or i1 %388, %391
  %393 = icmp ult ptr %286, %309
  %394 = icmp ult ptr %307, %288
  %395 = and i1 %393, %394
  %396 = or i1 %392, %395
  %397 = icmp ult ptr %291, %298
  %398 = icmp ult ptr %296, %294
  %399 = and i1 %397, %398
  %400 = or i1 %396, %399
  %401 = icmp ult ptr %291, %305
  %402 = icmp ult ptr %301, %294
  %403 = and i1 %401, %402
  %404 = or i1 %400, %403
  %405 = icmp ult ptr %291, %309
  %406 = icmp ult ptr %307, %294
  %407 = and i1 %405, %406
  %408 = or i1 %404, %407
  %409 = icmp ult ptr %296, %305
  %410 = icmp ult ptr %301, %298
  %411 = and i1 %409, %410
  %412 = or i1 %408, %411
  %413 = icmp ult ptr %296, %309
  %414 = icmp ult ptr %307, %298
  %415 = and i1 %413, %414
  %416 = or i1 %412, %415
  %417 = icmp ult ptr %301, %309
  %418 = icmp ult ptr %307, %305
  %419 = and i1 %417, %418
  %420 = or i1 %416, %419
  br i1 %420, label %481, label %421

421:                                              ; preds = %264
  %422 = and i64 %262, -2
  %423 = shl i64 %422, 1
  %424 = add i64 %423, %255
  %425 = insertelement <2 x double> poison, double %251, i64 0
  %426 = shufflevector <2 x double> %425, <2 x double> poison, <2 x i32> zeroinitializer
  br label %427

427:                                              ; preds = %427, %421
  %428 = phi i64 [ 0, %421 ], [ %477, %427 ]
  %429 = shl i64 %428, 1
  %430 = add i64 %429, %255
  %431 = add nsw i64 %430, %256
  %432 = add nsw i64 %431, %256
  %433 = getelementptr inbounds double, ptr %2, i64 %430
  %434 = load <4 x double>, ptr %433, align 8, !tbaa !6
  %435 = shufflevector <4 x double> %434, <4 x double> poison, <2 x i32> <i32 0, i32 2>
  %436 = shufflevector <4 x double> %434, <4 x double> poison, <2 x i32> <i32 1, i32 3>
  %437 = getelementptr inbounds double, ptr %2, i64 %431
  %438 = load <4 x double>, ptr %437, align 8, !tbaa !6
  %439 = shufflevector <4 x double> %438, <4 x double> poison, <2 x i32> <i32 0, i32 2>
  %440 = shufflevector <4 x double> %438, <4 x double> poison, <2 x i32> <i32 1, i32 3>
  %441 = fadd <2 x double> %435, %439
  %442 = fadd <2 x double> %436, %440
  %443 = fsub <2 x double> %435, %439
  %444 = fsub <2 x double> %436, %440
  %445 = getelementptr inbounds double, ptr %2, i64 %432
  %446 = load <4 x double>, ptr %445, align 8, !tbaa !6
  %447 = shufflevector <4 x double> %446, <4 x double> poison, <2 x i32> <i32 0, i32 2>
  %448 = shufflevector <4 x double> %446, <4 x double> poison, <2 x i32> <i32 1, i32 3>
  %449 = getelementptr double, ptr %258, i64 %432
  %450 = load <4 x double>, ptr %449, align 8, !tbaa !6
  %451 = shufflevector <4 x double> %450, <4 x double> poison, <2 x i32> <i32 0, i32 2>
  %452 = shufflevector <4 x double> %450, <4 x double> poison, <2 x i32> <i32 1, i32 3>
  %453 = fadd <2 x double> %447, %451
  %454 = fadd <2 x double> %448, %452
  %455 = fsub <2 x double> %447, %451
  %456 = fsub <2 x double> %448, %452
  %457 = fadd <2 x double> %441, %453
  %458 = fadd <2 x double> %442, %454
  %459 = shufflevector <2 x double> %457, <2 x double> %458, <4 x i32> <i32 0, i32 2, i32 1, i32 3>
  store <4 x double> %459, ptr %433, align 8, !tbaa !6
  %460 = fsub <2 x double> %454, %442
  %461 = fsub <2 x double> %441, %453
  %462 = shufflevector <2 x double> %460, <2 x double> %461, <4 x i32> <i32 0, i32 2, i32 1, i32 3>
  store <4 x double> %462, ptr %445, align 8, !tbaa !6
  %463 = fsub <2 x double> %443, %456
  %464 = fadd <2 x double> %444, %455
  %465 = fsub <2 x double> %463, %464
  %466 = fmul <2 x double> %426, %465
  %467 = fadd <2 x double> %464, %463
  %468 = fmul <2 x double> %426, %467
  %469 = shufflevector <2 x double> %466, <2 x double> %468, <4 x i32> <i32 0, i32 2, i32 1, i32 3>
  store <4 x double> %469, ptr %437, align 8, !tbaa !6
  %470 = fadd <2 x double> %443, %456
  %471 = fsub <2 x double> %455, %444
  %472 = fsub <2 x double> %471, %470
  %473 = fmul <2 x double> %426, %472
  %474 = fadd <2 x double> %471, %470
  %475 = fmul <2 x double> %426, %474
  %476 = shufflevector <2 x double> %473, <2 x double> %475, <4 x i32> <i32 0, i32 2, i32 1, i32 3>
  store <4 x double> %476, ptr %449, align 8, !tbaa !6
  %477 = add nuw i64 %428, 2
  %478 = icmp eq i64 %477, %422
  br i1 %478, label %479, label %427, !llvm.loop !53

479:                                              ; preds = %427
  %480 = icmp eq i64 %262, %422
  br i1 %480, label %529, label %481

481:                                              ; preds = %264, %254, %479
  %482 = phi i64 [ %255, %264 ], [ %255, %254 ], [ %424, %479 ]
  br label %483

483:                                              ; preds = %481, %483
  %484 = phi i64 [ %527, %483 ], [ %482, %481 ]
  %485 = add nsw i64 %484, %256
  %486 = add nsw i64 %485, %256
  %487 = getelementptr inbounds double, ptr %2, i64 %484
  %488 = load double, ptr %487, align 8, !tbaa !6
  %489 = getelementptr inbounds double, ptr %2, i64 %485
  %490 = load double, ptr %489, align 8, !tbaa !6
  %491 = fadd double %488, %490
  %492 = getelementptr i8, ptr %487, i64 8
  %493 = load double, ptr %492, align 8, !tbaa !6
  %494 = getelementptr i8, ptr %489, i64 8
  %495 = load double, ptr %494, align 8, !tbaa !6
  %496 = fadd double %493, %495
  %497 = fsub double %488, %490
  %498 = fsub double %493, %495
  %499 = getelementptr inbounds double, ptr %2, i64 %486
  %500 = load double, ptr %499, align 8, !tbaa !6
  %501 = getelementptr double, ptr %258, i64 %486
  %502 = load double, ptr %501, align 8, !tbaa !6
  %503 = fadd double %500, %502
  %504 = getelementptr i8, ptr %499, i64 8
  %505 = load double, ptr %504, align 8, !tbaa !6
  %506 = getelementptr i8, ptr %501, i64 8
  %507 = load double, ptr %506, align 8, !tbaa !6
  %508 = fadd double %505, %507
  %509 = fsub double %500, %502
  %510 = fsub double %505, %507
  %511 = fadd double %491, %503
  store double %511, ptr %487, align 8, !tbaa !6
  %512 = fadd double %496, %508
  store double %512, ptr %492, align 8, !tbaa !6
  %513 = fsub double %508, %496
  store double %513, ptr %499, align 8, !tbaa !6
  %514 = fsub double %491, %503
  store double %514, ptr %504, align 8, !tbaa !6
  %515 = fsub double %497, %510
  %516 = fadd double %498, %509
  %517 = fsub double %515, %516
  %518 = fmul double %251, %517
  store double %518, ptr %489, align 8, !tbaa !6
  %519 = fadd double %516, %515
  %520 = fmul double %251, %519
  store double %520, ptr %494, align 8, !tbaa !6
  %521 = fadd double %497, %510
  %522 = fsub double %509, %498
  %523 = fsub double %522, %521
  %524 = fmul double %251, %523
  store double %524, ptr %501, align 8, !tbaa !6
  %525 = fadd double %522, %521
  %526 = fmul double %251, %525
  store double %526, ptr %506, align 8, !tbaa !6
  %527 = add nsw i64 %484, 2
  %528 = icmp slt i64 %527, %257
  br i1 %528, label %483, label %529, !llvm.loop !54

529:                                              ; preds = %483, %479, %249
  %530 = shl i32 %1, 3
  %531 = icmp slt i32 %530, %0
  br i1 %531, label %532, label %1261

532:                                              ; preds = %529
  %533 = sext i32 %530 to i64
  %534 = sext i32 %1 to i64
  %535 = mul i32 %1, 12
  %536 = sext i32 %5 to i64
  %537 = zext nneg i32 %0 to i64
  %538 = getelementptr double, ptr %2, i64 %534
  %539 = add nsw i64 %536, %534
  %540 = getelementptr double, ptr %2, i64 %534
  %541 = getelementptr i8, ptr %2, i64 8
  %542 = add nsw i64 %534, %533
  %543 = add nsw i64 %542, %536
  %544 = getelementptr i8, ptr %2, i64 8
  %545 = getelementptr i8, ptr %2, i64 16
  %546 = shl nsw i64 %534, 4
  %547 = getelementptr i8, ptr %2, i64 %546
  %548 = or disjoint i64 %546, 8
  %549 = getelementptr i8, ptr %2, i64 %548
  %550 = getelementptr i8, ptr %2, i64 %548
  %551 = getelementptr i8, ptr %2, i64 %546
  %552 = getelementptr i8, ptr %551, i64 16
  %553 = getelementptr i8, ptr %2, i64 8
  %554 = getelementptr i8, ptr %2, i64 8
  %555 = getelementptr i8, ptr %2, i64 16
  %556 = mul nsw i64 %534, 24
  %557 = getelementptr i8, ptr %2, i64 %556
  %558 = add nsw i64 %556, 8
  %559 = getelementptr i8, ptr %2, i64 %558
  %560 = getelementptr i8, ptr %2, i64 %558
  %561 = getelementptr i8, ptr %2, i64 %556
  %562 = getelementptr i8, ptr %561, i64 16
  %563 = add nsw i64 %534, %533
  %564 = add nsw i64 %563, %536
  %565 = or disjoint i64 %533, 2
  %566 = xor i64 %533, -1
  %567 = shl nsw i64 %533, 3
  %568 = shl nsw i64 %533, 3
  %569 = shl nsw i64 %534, 4
  %570 = shl nsw i64 %563, 3
  %571 = mul nsw i64 %534, 24
  %572 = add nsw i64 %571, %567
  %573 = shl nsw i64 %533, 3
  %574 = shl nsw i64 %533, 3
  %575 = add nsw i64 %534, %533
  %576 = or disjoint i64 %533, 2
  %577 = xor i64 %533, -1
  %578 = shl nsw i64 %534, 4
  %579 = add nsw i64 %578, %573
  %580 = shl nsw i64 %575, 3
  %581 = mul nsw i64 %534, 24
  %582 = add nsw i64 %581, %573
  %583 = add nsw i64 %534, %533
  %584 = or disjoint i64 %533, 2
  %585 = xor i64 %533, -1
  %586 = getelementptr i8, ptr %2, i64 %573
  %587 = getelementptr i8, ptr %2, i64 %573
  %588 = getelementptr i8, ptr %587, i64 8
  %589 = getelementptr i8, ptr %2, i64 %573
  %590 = getelementptr i8, ptr %589, i64 16
  %591 = getelementptr i8, ptr %2, i64 %579
  %592 = getelementptr i8, ptr %2, i64 %579
  %593 = getelementptr i8, ptr %592, i64 8
  %594 = getelementptr i8, ptr %2, i64 %579
  %595 = getelementptr i8, ptr %594, i64 16
  %596 = getelementptr i8, ptr %2, i64 %580
  %597 = getelementptr i8, ptr %2, i64 %580
  %598 = getelementptr i8, ptr %597, i64 8
  %599 = getelementptr i8, ptr %2, i64 %580
  %600 = getelementptr i8, ptr %599, i64 16
  %601 = getelementptr i8, ptr %2, i64 %582
  %602 = getelementptr i8, ptr %2, i64 %582
  %603 = getelementptr i8, ptr %602, i64 8
  %604 = getelementptr i8, ptr %2, i64 %582
  %605 = getelementptr i8, ptr %604, i64 16
  %606 = getelementptr i8, ptr %2, i64 %567
  %607 = getelementptr i8, ptr %606, i64 8
  %608 = getelementptr i8, ptr %2, i64 %569
  %609 = getelementptr i8, ptr %608, i64 %567
  %610 = getelementptr i8, ptr %609, i64 8
  %611 = getelementptr i8, ptr %2, i64 %570
  %612 = getelementptr i8, ptr %611, i64 8
  %613 = getelementptr i8, ptr %2, i64 %572
  %614 = getelementptr i8, ptr %2, i64 %572
  %615 = getelementptr i8, ptr %614, i64 8
  br label %616

616:                                              ; preds = %532, %1256
  %617 = phi i64 [ 0, %532 ], [ %1260, %1256 ]
  %618 = phi i64 [ 0, %532 ], [ %714, %1256 ]
  %619 = phi i32 [ %535, %532 ], [ %1259, %1256 ]
  %620 = phi i64 [ %533, %532 ], [ %1257, %1256 ]
  %621 = mul i64 %617, %533
  %622 = add i64 %583, %621
  %623 = add i64 %584, %621
  %624 = tail call i64 @llvm.smax.i64(i64 %622, i64 %623)
  %625 = mul i64 %617, %533
  %626 = sub i64 %585, %625
  %627 = add i64 %624, %626
  %628 = lshr i64 %627, 1
  %629 = add nuw i64 %628, 1
  %630 = mul i64 %574, %617
  %631 = getelementptr i8, ptr %586, i64 %630
  %632 = getelementptr i8, ptr %588, i64 %630
  %633 = mul i64 %617, %533
  %634 = add i64 %575, %633
  %635 = add i64 %576, %633
  %636 = tail call i64 @llvm.smax.i64(i64 %634, i64 %635)
  %637 = mul i64 %617, %533
  %638 = sub i64 %577, %637
  %639 = add i64 %636, %638
  %640 = shl i64 %639, 3
  %641 = and i64 %640, -16
  %642 = getelementptr i8, ptr %632, i64 %641
  %643 = getelementptr i8, ptr %590, i64 %630
  %644 = getelementptr i8, ptr %643, i64 %641
  %645 = getelementptr i8, ptr %591, i64 %630
  %646 = getelementptr i8, ptr %593, i64 %630
  %647 = getelementptr i8, ptr %646, i64 %641
  %648 = getelementptr i8, ptr %595, i64 %630
  %649 = getelementptr i8, ptr %648, i64 %641
  %650 = getelementptr i8, ptr %596, i64 %630
  %651 = getelementptr i8, ptr %598, i64 %630
  %652 = getelementptr i8, ptr %651, i64 %641
  %653 = getelementptr i8, ptr %600, i64 %630
  %654 = getelementptr i8, ptr %653, i64 %641
  %655 = getelementptr i8, ptr %601, i64 %630
  %656 = getelementptr i8, ptr %603, i64 %630
  %657 = getelementptr i8, ptr %656, i64 %641
  %658 = getelementptr i8, ptr %605, i64 %630
  %659 = getelementptr i8, ptr %658, i64 %641
  %660 = mul i64 %617, %533
  %661 = add i64 %563, %660
  %662 = add i64 %565, %660
  %663 = tail call i64 @llvm.smax.i64(i64 %661, i64 %662)
  %664 = mul i64 %617, %533
  %665 = sub i64 %566, %664
  %666 = add i64 %663, %665
  %667 = lshr i64 %666, 1
  %668 = mul i64 %568, %617
  %669 = getelementptr i8, ptr %607, i64 %668
  %670 = getelementptr i8, ptr %610, i64 %668
  %671 = getelementptr i8, ptr %612, i64 %668
  %672 = getelementptr i8, ptr %613, i64 %668
  %673 = getelementptr i8, ptr %615, i64 %668
  %674 = mul i64 %617, %533
  %675 = add i64 %564, %674
  %676 = sext i32 %619 to i64
  %677 = or disjoint i64 %676, 2
  %678 = tail call i64 @llvm.smax.i64(i64 %675, i64 %677)
  %679 = xor i64 %676, -1
  %680 = add i64 %678, %679
  %681 = lshr i64 %680, 1
  %682 = add nuw i64 %681, 1
  %683 = sext i32 %619 to i64
  %684 = shl nsw i64 %683, 3
  %685 = getelementptr i8, ptr %2, i64 %684
  %686 = mul i64 %617, %533
  %687 = add i64 %543, %686
  %688 = or disjoint i64 %683, 2
  %689 = tail call i64 @llvm.smax.i64(i64 %687, i64 %688)
  %690 = xor i64 %683, -1
  %691 = add i64 %689, %690
  %692 = shl i64 %691, 3
  %693 = and i64 %692, -16
  %694 = add i64 %693, %684
  %695 = getelementptr i8, ptr %541, i64 %694
  %696 = getelementptr i8, ptr %544, i64 %684
  %697 = getelementptr i8, ptr %545, i64 %694
  %698 = getelementptr i8, ptr %547, i64 %684
  %699 = getelementptr i8, ptr %549, i64 %694
  %700 = getelementptr i8, ptr %550, i64 %684
  %701 = getelementptr i8, ptr %552, i64 %694
  %702 = add nsw i64 %534, %683
  %703 = shl nsw i64 %702, 3
  %704 = getelementptr i8, ptr %2, i64 %703
  %705 = add i64 %693, %703
  %706 = getelementptr i8, ptr %553, i64 %705
  %707 = getelementptr i8, ptr %554, i64 %703
  %708 = getelementptr i8, ptr %555, i64 %705
  %709 = getelementptr i8, ptr %557, i64 %684
  %710 = getelementptr i8, ptr %559, i64 %694
  %711 = getelementptr i8, ptr %560, i64 %684
  %712 = getelementptr i8, ptr %562, i64 %694
  %713 = sext i32 %619 to i64
  %714 = add nuw nsw i64 %618, 2
  %715 = getelementptr inbounds nuw double, ptr %3, i64 %714
  %716 = load double, ptr %715, align 8, !tbaa !6
  %717 = getelementptr inbounds nuw double, ptr %3, i64 %618
  %718 = getelementptr inbounds nuw i8, ptr %717, i64 24
  %719 = load double, ptr %718, align 8, !tbaa !6
  %720 = shl nuw nsw i64 %714, 4
  %721 = getelementptr inbounds nuw i8, ptr %3, i64 %720
  %722 = load double, ptr %721, align 8, !tbaa !6
  %723 = getelementptr inbounds nuw i8, ptr %721, i64 8
  %724 = load double, ptr %723, align 8, !tbaa !6
  %725 = fmul double %719, 2.000000e+00
  %726 = fneg double %725
  %727 = tail call double @llvm.fmuladd.f64(double %726, double %724, double %722)
  %728 = fneg double %724
  %729 = tail call double @llvm.fmuladd.f64(double %725, double %722, double %728)
  %730 = add nsw i64 %620, %534
  br i1 %6, label %731, label %1256

731:                                              ; preds = %616
  %732 = icmp ult i64 %627, 22
  br i1 %732, label %944, label %733

733:                                              ; preds = %731
  %734 = shl i64 %667, 4
  %735 = getelementptr i8, ptr %669, i64 %734
  %736 = icmp ult ptr %735, %669
  %737 = shl i64 %667, 4
  %738 = icmp ugt i64 %666, 2305843009213693951
  %739 = getelementptr i8, ptr %670, i64 %737
  %740 = icmp ult ptr %739, %670
  %741 = or i1 %740, %738
  %742 = shl i64 %667, 4
  %743 = getelementptr i8, ptr %671, i64 %742
  %744 = icmp ult ptr %743, %671
  %745 = shl i64 %667, 4
  %746 = getelementptr i8, ptr %672, i64 %745
  %747 = icmp ult ptr %746, %672
  %748 = shl i64 %667, 4
  %749 = getelementptr i8, ptr %673, i64 %748
  %750 = icmp ult ptr %749, %673
  %751 = or i1 %736, %741
  %752 = or i1 %744, %751
  %753 = or i1 %747, %752
  %754 = or i1 %750, %753
  br i1 %754, label %944, label %755

755:                                              ; preds = %733
  %756 = icmp ult ptr %631, %644
  %757 = icmp ult ptr %632, %642
  %758 = and i1 %756, %757
  %759 = icmp ult ptr %631, %647
  %760 = icmp ult ptr %645, %642
  %761 = and i1 %759, %760
  %762 = or i1 %758, %761
  %763 = icmp ult ptr %631, %649
  %764 = icmp ult ptr %646, %642
  %765 = and i1 %763, %764
  %766 = or i1 %762, %765
  %767 = icmp ult ptr %631, %652
  %768 = icmp ult ptr %650, %642
  %769 = and i1 %767, %768
  %770 = or i1 %766, %769
  %771 = icmp ult ptr %631, %654
  %772 = icmp ult ptr %651, %642
  %773 = and i1 %771, %772
  %774 = or i1 %770, %773
  %775 = icmp ult ptr %631, %657
  %776 = icmp ult ptr %655, %642
  %777 = and i1 %775, %776
  %778 = or i1 %774, %777
  %779 = icmp ult ptr %631, %659
  %780 = icmp ult ptr %656, %642
  %781 = and i1 %779, %780
  %782 = or i1 %778, %781
  %783 = icmp ult ptr %632, %647
  %784 = icmp ult ptr %645, %644
  %785 = and i1 %783, %784
  %786 = or i1 %782, %785
  %787 = icmp ult ptr %632, %649
  %788 = icmp ult ptr %646, %644
  %789 = and i1 %787, %788
  %790 = or i1 %786, %789
  %791 = icmp ult ptr %632, %652
  %792 = icmp ult ptr %650, %644
  %793 = and i1 %791, %792
  %794 = or i1 %790, %793
  %795 = icmp ult ptr %632, %654
  %796 = icmp ult ptr %651, %644
  %797 = and i1 %795, %796
  %798 = or i1 %794, %797
  %799 = icmp ult ptr %632, %657
  %800 = icmp ult ptr %655, %644
  %801 = and i1 %799, %800
  %802 = or i1 %798, %801
  %803 = icmp ult ptr %632, %659
  %804 = icmp ult ptr %656, %644
  %805 = and i1 %803, %804
  %806 = or i1 %802, %805
  %807 = icmp ult ptr %645, %649
  %808 = icmp ult ptr %646, %647
  %809 = and i1 %807, %808
  %810 = or i1 %806, %809
  %811 = icmp ult ptr %645, %652
  %812 = icmp ult ptr %650, %647
  %813 = and i1 %811, %812
  %814 = or i1 %810, %813
  %815 = icmp ult ptr %645, %654
  %816 = icmp ult ptr %651, %647
  %817 = and i1 %815, %816
  %818 = or i1 %814, %817
  %819 = icmp ult ptr %645, %657
  %820 = icmp ult ptr %655, %647
  %821 = and i1 %819, %820
  %822 = or i1 %818, %821
  %823 = icmp ult ptr %645, %659
  %824 = icmp ult ptr %656, %647
  %825 = and i1 %823, %824
  %826 = or i1 %822, %825
  %827 = icmp ult ptr %646, %652
  %828 = icmp ult ptr %650, %649
  %829 = and i1 %827, %828
  %830 = or i1 %826, %829
  %831 = icmp ult ptr %646, %654
  %832 = icmp ult ptr %651, %649
  %833 = and i1 %831, %832
  %834 = or i1 %830, %833
  %835 = icmp ult ptr %646, %657
  %836 = icmp ult ptr %655, %649
  %837 = and i1 %835, %836
  %838 = or i1 %834, %837
  %839 = icmp ult ptr %646, %659
  %840 = icmp ult ptr %656, %649
  %841 = and i1 %839, %840
  %842 = or i1 %838, %841
  %843 = icmp ult ptr %650, %654
  %844 = icmp ult ptr %651, %652
  %845 = and i1 %843, %844
  %846 = or i1 %842, %845
  %847 = icmp ult ptr %650, %657
  %848 = icmp ult ptr %655, %652
  %849 = and i1 %847, %848
  %850 = or i1 %846, %849
  %851 = icmp ult ptr %650, %659
  %852 = icmp ult ptr %656, %652
  %853 = and i1 %851, %852
  %854 = or i1 %850, %853
  %855 = icmp ult ptr %651, %657
  %856 = icmp ult ptr %655, %654
  %857 = and i1 %855, %856
  %858 = or i1 %854, %857
  %859 = icmp ult ptr %651, %659
  %860 = icmp ult ptr %656, %654
  %861 = and i1 %859, %860
  %862 = or i1 %858, %861
  %863 = icmp ult ptr %655, %659
  %864 = icmp ult ptr %656, %657
  %865 = and i1 %863, %864
  %866 = or i1 %862, %865
  br i1 %866, label %944, label %867

867:                                              ; preds = %755
  %868 = and i64 %629, -2
  %869 = shl i64 %868, 1
  %870 = add i64 %620, %869
  %871 = insertelement <2 x double> poison, double %719, i64 0
  %872 = shufflevector <2 x double> %871, <2 x double> poison, <2 x i32> zeroinitializer
  %873 = insertelement <2 x double> poison, double %716, i64 0
  %874 = shufflevector <2 x double> %873, <2 x double> poison, <2 x i32> zeroinitializer
  %875 = insertelement <2 x double> poison, double %724, i64 0
  %876 = shufflevector <2 x double> %875, <2 x double> poison, <2 x i32> zeroinitializer
  %877 = insertelement <2 x double> poison, double %722, i64 0
  %878 = shufflevector <2 x double> %877, <2 x double> poison, <2 x i32> zeroinitializer
  %879 = insertelement <2 x double> poison, double %729, i64 0
  %880 = shufflevector <2 x double> %879, <2 x double> poison, <2 x i32> zeroinitializer
  %881 = insertelement <2 x double> poison, double %727, i64 0
  %882 = shufflevector <2 x double> %881, <2 x double> poison, <2 x i32> zeroinitializer
  br label %883

883:                                              ; preds = %883, %867
  %884 = phi i64 [ 0, %867 ], [ %940, %883 ]
  %885 = shl i64 %884, 1
  %886 = add i64 %620, %885
  %887 = add nsw i64 %886, %534
  %888 = add nsw i64 %887, %534
  %889 = getelementptr inbounds double, ptr %2, i64 %886
  %890 = load <4 x double>, ptr %889, align 8, !tbaa !6
  %891 = shufflevector <4 x double> %890, <4 x double> poison, <2 x i32> <i32 0, i32 2>
  %892 = shufflevector <4 x double> %890, <4 x double> poison, <2 x i32> <i32 1, i32 3>
  %893 = getelementptr inbounds double, ptr %2, i64 %887
  %894 = load <4 x double>, ptr %893, align 8, !tbaa !6
  %895 = shufflevector <4 x double> %894, <4 x double> poison, <2 x i32> <i32 0, i32 2>
  %896 = shufflevector <4 x double> %894, <4 x double> poison, <2 x i32> <i32 1, i32 3>
  %897 = fadd <2 x double> %891, %895
  %898 = fadd <2 x double> %892, %896
  %899 = fsub <2 x double> %891, %895
  %900 = fsub <2 x double> %892, %896
  %901 = getelementptr inbounds double, ptr %2, i64 %888
  %902 = load <4 x double>, ptr %901, align 8, !tbaa !6
  %903 = shufflevector <4 x double> %902, <4 x double> poison, <2 x i32> <i32 0, i32 2>
  %904 = shufflevector <4 x double> %902, <4 x double> poison, <2 x i32> <i32 1, i32 3>
  %905 = getelementptr double, ptr %538, i64 %888
  %906 = load <4 x double>, ptr %905, align 8, !tbaa !6
  %907 = shufflevector <4 x double> %906, <4 x double> poison, <2 x i32> <i32 0, i32 2>
  %908 = shufflevector <4 x double> %906, <4 x double> poison, <2 x i32> <i32 1, i32 3>
  %909 = fadd <2 x double> %903, %907
  %910 = fadd <2 x double> %904, %908
  %911 = fsub <2 x double> %903, %907
  %912 = fsub <2 x double> %904, %908
  %913 = fadd <2 x double> %897, %909
  %914 = fadd <2 x double> %898, %910
  %915 = shufflevector <2 x double> %913, <2 x double> %914, <4 x i32> <i32 0, i32 2, i32 1, i32 3>
  store <4 x double> %915, ptr %889, align 8, !tbaa !6
  %916 = fsub <2 x double> %897, %909
  %917 = fsub <2 x double> %898, %910
  %918 = fneg <2 x double> %917
  %919 = fmul <2 x double> %872, %918
  %920 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %874, <2 x double> %916, <2 x double> %919)
  %921 = fmul <2 x double> %872, %916
  %922 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %874, <2 x double> %917, <2 x double> %921)
  %923 = shufflevector <2 x double> %920, <2 x double> %922, <4 x i32> <i32 0, i32 2, i32 1, i32 3>
  store <4 x double> %923, ptr %901, align 8, !tbaa !6
  %924 = fsub <2 x double> %899, %912
  %925 = fadd <2 x double> %900, %911
  %926 = fneg <2 x double> %925
  %927 = fmul <2 x double> %876, %926
  %928 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %878, <2 x double> %924, <2 x double> %927)
  %929 = fmul <2 x double> %876, %924
  %930 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %878, <2 x double> %925, <2 x double> %929)
  %931 = shufflevector <2 x double> %928, <2 x double> %930, <4 x i32> <i32 0, i32 2, i32 1, i32 3>
  store <4 x double> %931, ptr %893, align 8, !tbaa !6
  %932 = fadd <2 x double> %899, %912
  %933 = fsub <2 x double> %900, %911
  %934 = fneg <2 x double> %933
  %935 = fmul <2 x double> %880, %934
  %936 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %882, <2 x double> %932, <2 x double> %935)
  %937 = fmul <2 x double> %880, %932
  %938 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %882, <2 x double> %933, <2 x double> %937)
  %939 = shufflevector <2 x double> %936, <2 x double> %938, <4 x i32> <i32 0, i32 2, i32 1, i32 3>
  store <4 x double> %939, ptr %905, align 8, !tbaa !6
  %940 = add nuw i64 %884, 2
  %941 = icmp eq i64 %940, %868
  br i1 %941, label %942, label %883, !llvm.loop !55

942:                                              ; preds = %883
  %943 = icmp eq i64 %629, %868
  br i1 %943, label %999, label %944

944:                                              ; preds = %755, %733, %731, %942
  %945 = phi i64 [ %620, %755 ], [ %620, %733 ], [ %620, %731 ], [ %870, %942 ]
  br label %946

946:                                              ; preds = %944, %946
  %947 = phi i64 [ %997, %946 ], [ %945, %944 ]
  %948 = add nsw i64 %947, %534
  %949 = add nsw i64 %948, %534
  %950 = getelementptr inbounds double, ptr %2, i64 %947
  %951 = load double, ptr %950, align 8, !tbaa !6
  %952 = getelementptr inbounds double, ptr %2, i64 %948
  %953 = load double, ptr %952, align 8, !tbaa !6
  %954 = fadd double %951, %953
  %955 = getelementptr i8, ptr %950, i64 8
  %956 = load double, ptr %955, align 8, !tbaa !6
  %957 = getelementptr i8, ptr %952, i64 8
  %958 = load double, ptr %957, align 8, !tbaa !6
  %959 = fadd double %956, %958
  %960 = fsub double %951, %953
  %961 = fsub double %956, %958
  %962 = getelementptr inbounds double, ptr %2, i64 %949
  %963 = load double, ptr %962, align 8, !tbaa !6
  %964 = getelementptr double, ptr %538, i64 %949
  %965 = load double, ptr %964, align 8, !tbaa !6
  %966 = fadd double %963, %965
  %967 = getelementptr i8, ptr %962, i64 8
  %968 = load double, ptr %967, align 8, !tbaa !6
  %969 = getelementptr i8, ptr %964, i64 8
  %970 = load double, ptr %969, align 8, !tbaa !6
  %971 = fadd double %968, %970
  %972 = fsub double %963, %965
  %973 = fsub double %968, %970
  %974 = fadd double %954, %966
  store double %974, ptr %950, align 8, !tbaa !6
  %975 = fadd double %959, %971
  store double %975, ptr %955, align 8, !tbaa !6
  %976 = fsub double %954, %966
  %977 = fsub double %959, %971
  %978 = fneg double %977
  %979 = fmul double %719, %978
  %980 = tail call double @llvm.fmuladd.f64(double %716, double %976, double %979)
  store double %980, ptr %962, align 8, !tbaa !6
  %981 = fmul double %719, %976
  %982 = tail call double @llvm.fmuladd.f64(double %716, double %977, double %981)
  store double %982, ptr %967, align 8, !tbaa !6
  %983 = fsub double %960, %973
  %984 = fadd double %961, %972
  %985 = fneg double %984
  %986 = fmul double %724, %985
  %987 = tail call double @llvm.fmuladd.f64(double %722, double %983, double %986)
  store double %987, ptr %952, align 8, !tbaa !6
  %988 = fmul double %724, %983
  %989 = tail call double @llvm.fmuladd.f64(double %722, double %984, double %988)
  store double %989, ptr %957, align 8, !tbaa !6
  %990 = fadd double %960, %973
  %991 = fsub double %961, %972
  %992 = fneg double %991
  %993 = fmul double %729, %992
  %994 = tail call double @llvm.fmuladd.f64(double %727, double %990, double %993)
  store double %994, ptr %964, align 8, !tbaa !6
  %995 = fmul double %729, %990
  %996 = tail call double @llvm.fmuladd.f64(double %727, double %991, double %995)
  store double %996, ptr %969, align 8, !tbaa !6
  %997 = add nsw i64 %947, 2
  %998 = icmp slt i64 %997, %730
  br i1 %998, label %946, label %999, !llvm.loop !56

999:                                              ; preds = %946, %942
  %1000 = getelementptr inbounds nuw i8, ptr %721, i64 16
  %1001 = load double, ptr %1000, align 8, !tbaa !6
  %1002 = getelementptr inbounds nuw i8, ptr %721, i64 24
  %1003 = load double, ptr %1002, align 8, !tbaa !6
  %1004 = fmul double %716, 2.000000e+00
  %1005 = fneg double %1004
  %1006 = tail call double @llvm.fmuladd.f64(double %1005, double %1003, double %1001)
  %1007 = fneg double %1003
  %1008 = tail call double @llvm.fmuladd.f64(double %1004, double %1001, double %1007)
  %1009 = add i64 %620, %539
  %1010 = fneg double %719
  %1011 = icmp ult i64 %680, 14
  br i1 %1011, label %1201, label %1012

1012:                                             ; preds = %999
  %1013 = icmp ult ptr %685, %697
  %1014 = icmp ult ptr %696, %695
  %1015 = and i1 %1013, %1014
  %1016 = icmp ult ptr %685, %699
  %1017 = icmp ult ptr %698, %695
  %1018 = and i1 %1016, %1017
  %1019 = or i1 %1015, %1018
  %1020 = icmp ult ptr %685, %701
  %1021 = icmp ult ptr %700, %695
  %1022 = and i1 %1020, %1021
  %1023 = or i1 %1019, %1022
  %1024 = icmp ult ptr %685, %706
  %1025 = icmp ult ptr %704, %695
  %1026 = and i1 %1024, %1025
  %1027 = or i1 %1023, %1026
  %1028 = icmp ult ptr %685, %708
  %1029 = icmp ult ptr %707, %695
  %1030 = and i1 %1028, %1029
  %1031 = or i1 %1027, %1030
  %1032 = icmp ult ptr %685, %710
  %1033 = icmp ult ptr %709, %695
  %1034 = and i1 %1032, %1033
  %1035 = or i1 %1031, %1034
  %1036 = icmp ult ptr %685, %712
  %1037 = icmp ult ptr %711, %695
  %1038 = and i1 %1036, %1037
  %1039 = or i1 %1035, %1038
  %1040 = icmp ult ptr %696, %699
  %1041 = icmp ult ptr %698, %697
  %1042 = and i1 %1040, %1041
  %1043 = or i1 %1039, %1042
  %1044 = icmp ult ptr %696, %701
  %1045 = icmp ult ptr %700, %697
  %1046 = and i1 %1044, %1045
  %1047 = or i1 %1043, %1046
  %1048 = icmp ult ptr %696, %706
  %1049 = icmp ult ptr %704, %697
  %1050 = and i1 %1048, %1049
  %1051 = or i1 %1047, %1050
  %1052 = icmp ult ptr %696, %708
  %1053 = icmp ult ptr %707, %697
  %1054 = and i1 %1052, %1053
  %1055 = or i1 %1051, %1054
  %1056 = icmp ult ptr %696, %710
  %1057 = icmp ult ptr %709, %697
  %1058 = and i1 %1056, %1057
  %1059 = or i1 %1055, %1058
  %1060 = icmp ult ptr %696, %712
  %1061 = icmp ult ptr %711, %697
  %1062 = and i1 %1060, %1061
  %1063 = or i1 %1059, %1062
  %1064 = icmp ult ptr %698, %701
  %1065 = icmp ult ptr %700, %699
  %1066 = and i1 %1064, %1065
  %1067 = or i1 %1063, %1066
  %1068 = icmp ult ptr %698, %706
  %1069 = icmp ult ptr %704, %699
  %1070 = and i1 %1068, %1069
  %1071 = or i1 %1067, %1070
  %1072 = icmp ult ptr %698, %708
  %1073 = icmp ult ptr %707, %699
  %1074 = and i1 %1072, %1073
  %1075 = or i1 %1071, %1074
  %1076 = icmp ult ptr %698, %710
  %1077 = icmp ult ptr %709, %699
  %1078 = and i1 %1076, %1077
  %1079 = or i1 %1075, %1078
  %1080 = icmp ult ptr %698, %712
  %1081 = icmp ult ptr %711, %699
  %1082 = and i1 %1080, %1081
  %1083 = or i1 %1079, %1082
  %1084 = icmp ult ptr %700, %706
  %1085 = icmp ult ptr %704, %701
  %1086 = and i1 %1084, %1085
  %1087 = or i1 %1083, %1086
  %1088 = icmp ult ptr %700, %708
  %1089 = icmp ult ptr %707, %701
  %1090 = and i1 %1088, %1089
  %1091 = or i1 %1087, %1090
  %1092 = icmp ult ptr %700, %710
  %1093 = icmp ult ptr %709, %701
  %1094 = and i1 %1092, %1093
  %1095 = or i1 %1091, %1094
  %1096 = icmp ult ptr %700, %712
  %1097 = icmp ult ptr %711, %701
  %1098 = and i1 %1096, %1097
  %1099 = or i1 %1095, %1098
  %1100 = icmp ult ptr %704, %708
  %1101 = icmp ult ptr %707, %706
  %1102 = and i1 %1100, %1101
  %1103 = or i1 %1099, %1102
  %1104 = icmp ult ptr %704, %710
  %1105 = icmp ult ptr %709, %706
  %1106 = and i1 %1104, %1105
  %1107 = or i1 %1103, %1106
  %1108 = icmp ult ptr %704, %712
  %1109 = icmp ult ptr %711, %706
  %1110 = and i1 %1108, %1109
  %1111 = or i1 %1107, %1110
  %1112 = icmp ult ptr %707, %710
  %1113 = icmp ult ptr %709, %708
  %1114 = and i1 %1112, %1113
  %1115 = or i1 %1111, %1114
  %1116 = icmp ult ptr %707, %712
  %1117 = icmp ult ptr %711, %708
  %1118 = and i1 %1116, %1117
  %1119 = or i1 %1115, %1118
  %1120 = icmp ult ptr %709, %712
  %1121 = icmp ult ptr %711, %710
  %1122 = and i1 %1120, %1121
  %1123 = or i1 %1119, %1122
  br i1 %1123, label %1201, label %1124

1124:                                             ; preds = %1012
  %1125 = and i64 %682, -2
  %1126 = shl i64 %1125, 1
  %1127 = add i64 %1126, %713
  %1128 = insertelement <2 x double> poison, double %1010, i64 0
  %1129 = shufflevector <2 x double> %1128, <2 x double> poison, <2 x i32> zeroinitializer
  %1130 = insertelement <2 x double> poison, double %716, i64 0
  %1131 = shufflevector <2 x double> %1130, <2 x double> poison, <2 x i32> zeroinitializer
  %1132 = insertelement <2 x double> poison, double %1003, i64 0
  %1133 = shufflevector <2 x double> %1132, <2 x double> poison, <2 x i32> zeroinitializer
  %1134 = insertelement <2 x double> poison, double %1001, i64 0
  %1135 = shufflevector <2 x double> %1134, <2 x double> poison, <2 x i32> zeroinitializer
  %1136 = insertelement <2 x double> poison, double %1008, i64 0
  %1137 = shufflevector <2 x double> %1136, <2 x double> poison, <2 x i32> zeroinitializer
  %1138 = insertelement <2 x double> poison, double %1006, i64 0
  %1139 = shufflevector <2 x double> %1138, <2 x double> poison, <2 x i32> zeroinitializer
  br label %1140

1140:                                             ; preds = %1140, %1124
  %1141 = phi i64 [ 0, %1124 ], [ %1197, %1140 ]
  %1142 = shl i64 %1141, 1
  %1143 = add i64 %1142, %713
  %1144 = add nsw i64 %1143, %534
  %1145 = add nsw i64 %1144, %534
  %1146 = getelementptr inbounds double, ptr %2, i64 %1143
  %1147 = load <4 x double>, ptr %1146, align 8, !tbaa !6
  %1148 = shufflevector <4 x double> %1147, <4 x double> poison, <2 x i32> <i32 0, i32 2>
  %1149 = shufflevector <4 x double> %1147, <4 x double> poison, <2 x i32> <i32 1, i32 3>
  %1150 = getelementptr inbounds double, ptr %2, i64 %1144
  %1151 = load <4 x double>, ptr %1150, align 8, !tbaa !6
  %1152 = shufflevector <4 x double> %1151, <4 x double> poison, <2 x i32> <i32 0, i32 2>
  %1153 = shufflevector <4 x double> %1151, <4 x double> poison, <2 x i32> <i32 1, i32 3>
  %1154 = fadd <2 x double> %1148, %1152
  %1155 = fadd <2 x double> %1149, %1153
  %1156 = fsub <2 x double> %1148, %1152
  %1157 = fsub <2 x double> %1149, %1153
  %1158 = getelementptr inbounds double, ptr %2, i64 %1145
  %1159 = load <4 x double>, ptr %1158, align 8, !tbaa !6
  %1160 = shufflevector <4 x double> %1159, <4 x double> poison, <2 x i32> <i32 0, i32 2>
  %1161 = shufflevector <4 x double> %1159, <4 x double> poison, <2 x i32> <i32 1, i32 3>
  %1162 = getelementptr double, ptr %540, i64 %1145
  %1163 = load <4 x double>, ptr %1162, align 8, !tbaa !6
  %1164 = shufflevector <4 x double> %1163, <4 x double> poison, <2 x i32> <i32 0, i32 2>
  %1165 = shufflevector <4 x double> %1163, <4 x double> poison, <2 x i32> <i32 1, i32 3>
  %1166 = fadd <2 x double> %1160, %1164
  %1167 = fadd <2 x double> %1161, %1165
  %1168 = fsub <2 x double> %1160, %1164
  %1169 = fsub <2 x double> %1161, %1165
  %1170 = fadd <2 x double> %1154, %1166
  %1171 = fadd <2 x double> %1155, %1167
  %1172 = shufflevector <2 x double> %1170, <2 x double> %1171, <4 x i32> <i32 0, i32 2, i32 1, i32 3>
  store <4 x double> %1172, ptr %1146, align 8, !tbaa !6
  %1173 = fsub <2 x double> %1154, %1166
  %1174 = fsub <2 x double> %1155, %1167
  %1175 = fneg <2 x double> %1174
  %1176 = fmul <2 x double> %1131, %1175
  %1177 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %1129, <2 x double> %1173, <2 x double> %1176)
  %1178 = fmul <2 x double> %1131, %1173
  %1179 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %1129, <2 x double> %1174, <2 x double> %1178)
  %1180 = shufflevector <2 x double> %1177, <2 x double> %1179, <4 x i32> <i32 0, i32 2, i32 1, i32 3>
  store <4 x double> %1180, ptr %1158, align 8, !tbaa !6
  %1181 = fsub <2 x double> %1156, %1169
  %1182 = fadd <2 x double> %1157, %1168
  %1183 = fneg <2 x double> %1182
  %1184 = fmul <2 x double> %1133, %1183
  %1185 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %1135, <2 x double> %1181, <2 x double> %1184)
  %1186 = fmul <2 x double> %1133, %1181
  %1187 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %1135, <2 x double> %1182, <2 x double> %1186)
  %1188 = shufflevector <2 x double> %1185, <2 x double> %1187, <4 x i32> <i32 0, i32 2, i32 1, i32 3>
  store <4 x double> %1188, ptr %1150, align 8, !tbaa !6
  %1189 = fadd <2 x double> %1156, %1169
  %1190 = fsub <2 x double> %1157, %1168
  %1191 = fneg <2 x double> %1190
  %1192 = fmul <2 x double> %1137, %1191
  %1193 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %1139, <2 x double> %1189, <2 x double> %1192)
  %1194 = fmul <2 x double> %1137, %1189
  %1195 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %1139, <2 x double> %1190, <2 x double> %1194)
  %1196 = shufflevector <2 x double> %1193, <2 x double> %1195, <4 x i32> <i32 0, i32 2, i32 1, i32 3>
  store <4 x double> %1196, ptr %1162, align 8, !tbaa !6
  %1197 = add nuw i64 %1141, 2
  %1198 = icmp eq i64 %1197, %1125
  br i1 %1198, label %1199, label %1140, !llvm.loop !57

1199:                                             ; preds = %1140
  %1200 = icmp eq i64 %682, %1125
  br i1 %1200, label %1256, label %1201

1201:                                             ; preds = %1012, %999, %1199
  %1202 = phi i64 [ %713, %1012 ], [ %713, %999 ], [ %1127, %1199 ]
  br label %1203

1203:                                             ; preds = %1201, %1203
  %1204 = phi i64 [ %1254, %1203 ], [ %1202, %1201 ]
  %1205 = add nsw i64 %1204, %534
  %1206 = add nsw i64 %1205, %534
  %1207 = getelementptr inbounds double, ptr %2, i64 %1204
  %1208 = load double, ptr %1207, align 8, !tbaa !6
  %1209 = getelementptr inbounds double, ptr %2, i64 %1205
  %1210 = load double, ptr %1209, align 8, !tbaa !6
  %1211 = fadd double %1208, %1210
  %1212 = getelementptr i8, ptr %1207, i64 8
  %1213 = load double, ptr %1212, align 8, !tbaa !6
  %1214 = getelementptr i8, ptr %1209, i64 8
  %1215 = load double, ptr %1214, align 8, !tbaa !6
  %1216 = fadd double %1213, %1215
  %1217 = fsub double %1208, %1210
  %1218 = fsub double %1213, %1215
  %1219 = getelementptr inbounds double, ptr %2, i64 %1206
  %1220 = load double, ptr %1219, align 8, !tbaa !6
  %1221 = getelementptr double, ptr %540, i64 %1206
  %1222 = load double, ptr %1221, align 8, !tbaa !6
  %1223 = fadd double %1220, %1222
  %1224 = getelementptr i8, ptr %1219, i64 8
  %1225 = load double, ptr %1224, align 8, !tbaa !6
  %1226 = getelementptr i8, ptr %1221, i64 8
  %1227 = load double, ptr %1226, align 8, !tbaa !6
  %1228 = fadd double %1225, %1227
  %1229 = fsub double %1220, %1222
  %1230 = fsub double %1225, %1227
  %1231 = fadd double %1211, %1223
  store double %1231, ptr %1207, align 8, !tbaa !6
  %1232 = fadd double %1216, %1228
  store double %1232, ptr %1212, align 8, !tbaa !6
  %1233 = fsub double %1211, %1223
  %1234 = fsub double %1216, %1228
  %1235 = fneg double %1234
  %1236 = fmul double %716, %1235
  %1237 = tail call double @llvm.fmuladd.f64(double %1010, double %1233, double %1236)
  store double %1237, ptr %1219, align 8, !tbaa !6
  %1238 = fmul double %716, %1233
  %1239 = tail call double @llvm.fmuladd.f64(double %1010, double %1234, double %1238)
  store double %1239, ptr %1224, align 8, !tbaa !6
  %1240 = fsub double %1217, %1230
  %1241 = fadd double %1218, %1229
  %1242 = fneg double %1241
  %1243 = fmul double %1003, %1242
  %1244 = tail call double @llvm.fmuladd.f64(double %1001, double %1240, double %1243)
  store double %1244, ptr %1209, align 8, !tbaa !6
  %1245 = fmul double %1003, %1240
  %1246 = tail call double @llvm.fmuladd.f64(double %1001, double %1241, double %1245)
  store double %1246, ptr %1214, align 8, !tbaa !6
  %1247 = fadd double %1217, %1230
  %1248 = fsub double %1218, %1229
  %1249 = fneg double %1248
  %1250 = fmul double %1008, %1249
  %1251 = tail call double @llvm.fmuladd.f64(double %1006, double %1247, double %1250)
  store double %1251, ptr %1221, align 8, !tbaa !6
  %1252 = fmul double %1008, %1247
  %1253 = tail call double @llvm.fmuladd.f64(double %1006, double %1248, double %1252)
  store double %1253, ptr %1226, align 8, !tbaa !6
  %1254 = add nsw i64 %1204, 2
  %1255 = icmp slt i64 %1254, %1009
  br i1 %1255, label %1203, label %1256, !llvm.loop !58

1256:                                             ; preds = %1203, %1199, %616
  %1257 = add nsw i64 %620, %533
  %1258 = icmp slt i64 %1257, %537
  %1259 = add i32 %619, %530
  %1260 = add i64 %617, 1
  br i1 %1258, label %616, label %1261, !llvm.loop !59

1261:                                             ; preds = %1256, %529
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <2 x double> @llvm.fmuladd.v2f64(<2 x double>, <2 x double>, <2 x double>) #16

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smax.i64(i64, i64) #16

attributes #0 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress nofree nounwind willreturn allockind("alloc,uninitialized,aligned") allocsize(1) memory(inaccessiblemem: readwrite) "alloc-family"="malloc" "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nofree norecurse nounwind memory(argmem: readwrite, errnomem: write) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { nofree norecurse nosync nounwind memory(argmem: write) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { nofree norecurse nosync nounwind memory(argmem: read) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #8 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #9 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #10 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #11 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #12 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #13 = { mustprogress nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite) "alloc-family"="malloc" "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #14 = { mustprogress nocallback nofree nounwind willreturn memory(errnomem: write) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #15 = { inlinehint nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #16 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #17 = { nounwind }
attributes #18 = { nounwind allocsize(1) }
attributes #19 = { cold noreturn nounwind }

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
!10 = !{!11, !11, i64 0}
!11 = !{!"int", !8, i64 0}
!12 = distinct !{!12, !13}
!13 = !{!"llvm.loop.mustprogress"}
!14 = distinct !{!14, !13}
!15 = distinct !{!15, !13}
!16 = distinct !{!16, !13, !17, !18}
!17 = !{!"llvm.loop.isvectorized", i32 1}
!18 = !{!"llvm.loop.unroll.runtime.disable"}
!19 = distinct !{!19, !13, !17, !18}
!20 = distinct !{!20, !13}
!21 = distinct !{!21, !13}
!22 = !{!23, !24, i64 0}
!23 = !{!"timeval", !24, i64 0, !24, i64 8}
!24 = !{!"long", !8, i64 0}
!25 = !{!23, !24, i64 8}
!26 = distinct !{!26, !13, !17, !18}
!27 = distinct !{!27, !13, !18, !17}
!28 = distinct !{!28, !13}
!29 = distinct !{!29, !13}
!30 = distinct !{!30, !13}
!31 = distinct !{!31, !13}
!32 = distinct !{!32, !13}
!33 = distinct !{!33, !13}
!34 = distinct !{!34, !13, !17, !18}
!35 = distinct !{!35, !13, !17, !18}
!36 = distinct !{!36, !13, !17}
!37 = distinct !{!37, !13, !17}
!38 = distinct !{!38, !13, !17, !18}
!39 = distinct !{!39, !13, !18, !17}
!40 = distinct !{!40, !13}
!41 = distinct !{!41, !13}
!42 = distinct !{!42, !13}
!43 = distinct !{!43, !13}
!44 = distinct !{!44, !13}
!45 = distinct !{!45, !13}
!46 = distinct !{!46, !13, !17, !18}
!47 = distinct !{!47, !13, !17, !18}
!48 = distinct !{!48, !13, !17}
!49 = distinct !{!49, !13, !17}
!50 = distinct !{!50, !13}
!51 = distinct !{!51, !13, !17, !18}
!52 = distinct !{!52, !13, !17}
!53 = distinct !{!53, !13, !17, !18}
!54 = distinct !{!54, !13, !17}
!55 = distinct !{!55, !13, !17, !18}
!56 = distinct !{!56, !13, !17}
!57 = distinct !{!57, !13, !17, !18}
!58 = distinct !{!58, !13, !17}
!59 = distinct !{!59, !13}
