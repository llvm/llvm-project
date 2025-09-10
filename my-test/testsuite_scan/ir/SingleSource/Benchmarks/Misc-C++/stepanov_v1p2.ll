; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Misc-C++/stepanov_v1p2.cpp'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Misc-C++/stepanov_v1p2.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.Double = type { double }
%struct.double_pointer = type { ptr }
%struct.Double_pointer = type { ptr }
%struct.reverse_iterator = type { ptr }
%struct.reverse_iterator.0 = type { ptr }
%struct.reverse_iterator.1 = type { %struct.double_pointer }
%struct.reverse_iterator.2 = type { %struct.Double_pointer }
%struct.reverse_iterator.3 = type { %struct.reverse_iterator }
%struct.reverse_iterator.4 = type { %struct.reverse_iterator.0 }
%struct.reverse_iterator.5 = type { %struct.reverse_iterator.1 }
%struct.reverse_iterator.6 = type { %struct.reverse_iterator.2 }

$_Z4testIPddEvT_S1_T0_ = comdat any

$_Z4testIP6DoubleS0_EvT_S2_T0_ = comdat any

$_Z4testI14double_pointerdEvT_S1_T0_ = comdat any

$_Z4testI14Double_pointer6DoubleEvT_S2_T0_ = comdat any

$_Z4testI16reverse_iteratorIPddEdEvT_S3_T0_ = comdat any

$_Z4testI16reverse_iteratorIP6DoubleS1_ES1_EvT_S4_T0_ = comdat any

$_Z4testI16reverse_iteratorI14double_pointerdEdEvT_S3_T0_ = comdat any

$_Z4testI16reverse_iteratorI14Double_pointer6DoubleES2_EvT_S4_T0_ = comdat any

$_Z4testI16reverse_iteratorIS0_IPddEdEdEvT_S4_T0_ = comdat any

$_Z4testI16reverse_iteratorIS0_IP6DoubleS1_ES1_ES1_EvT_S5_T0_ = comdat any

$_Z4testI16reverse_iteratorIS0_I14double_pointerdEdEdEvT_S4_T0_ = comdat any

$_Z4testI16reverse_iteratorIS0_I14Double_pointer6DoubleES2_ES2_EvT_S5_T0_ = comdat any

@iterations = dso_local local_unnamed_addr global i32 250000, align 4
@current_test = dso_local local_unnamed_addr global i32 0, align 4
@result_times = dso_local local_unnamed_addr global [20 x double] zeroinitializer, align 8
@.str.2 = private unnamed_addr constant [43 x i8] c"%2i       %5.2fsec    %5.2fM         %.2f\0A\00", align 1
@.str.3 = private unnamed_addr constant [42 x i8] c"mean:    %5.2fsec    %5.2fM         %.2f\0A\00", align 1
@.str.4 = private unnamed_addr constant [32 x i8] c"\0ATotal absolute time: %.2f sec\0A\00", align 1
@.str.5 = private unnamed_addr constant [29 x i8] c"\0AAbstraction Penalty: %.2f\0A\0A\00", align 1
@start_time = dso_local local_unnamed_addr global i64 0, align 8
@end_time = dso_local local_unnamed_addr global i64 0, align 8
@data = dso_local global [2000 x double] zeroinitializer, align 8
@Data = dso_local global [2000 x %struct.Double] zeroinitializer, align 8
@d = dso_local local_unnamed_addr global double 0.000000e+00, align 8
@D = dso_local local_unnamed_addr global %struct.Double zeroinitializer, align 8
@dpb = dso_local local_unnamed_addr global ptr @data, align 8
@dpe = dso_local local_unnamed_addr global ptr getelementptr inbounds nuw (i8, ptr @data, i64 16000), align 8
@Dpb = dso_local local_unnamed_addr global ptr @Data, align 8
@Dpe = dso_local local_unnamed_addr global ptr getelementptr inbounds nuw (i8, ptr @Data, i64 16000), align 8
@dPb = dso_local local_unnamed_addr global %struct.double_pointer { ptr @data }, align 8
@dPe = dso_local local_unnamed_addr global %struct.double_pointer { ptr getelementptr inbounds nuw (i8, ptr @data, i64 16000) }, align 8
@DPb = dso_local local_unnamed_addr global %struct.Double_pointer { ptr @Data }, align 8
@DPe = dso_local local_unnamed_addr global %struct.Double_pointer { ptr getelementptr inbounds nuw (i8, ptr @Data, i64 16000) }, align 8
@rdpb = dso_local local_unnamed_addr global %struct.reverse_iterator { ptr getelementptr inbounds nuw (i8, ptr @data, i64 16000) }, align 8
@rdpe = dso_local local_unnamed_addr global %struct.reverse_iterator { ptr @data }, align 8
@rDpb = dso_local local_unnamed_addr global %struct.reverse_iterator.0 { ptr getelementptr inbounds nuw (i8, ptr @Data, i64 16000) }, align 8
@rDpe = dso_local local_unnamed_addr global %struct.reverse_iterator.0 { ptr @Data }, align 8
@rdPb = dso_local local_unnamed_addr global %struct.reverse_iterator.1 { %struct.double_pointer { ptr getelementptr inbounds nuw (i8, ptr @data, i64 16000) } }, align 8
@rdPe = dso_local local_unnamed_addr global %struct.reverse_iterator.1 { %struct.double_pointer { ptr @data } }, align 8
@rDPb = dso_local local_unnamed_addr global %struct.reverse_iterator.2 { %struct.Double_pointer { ptr getelementptr inbounds nuw (i8, ptr @Data, i64 16000) } }, align 8
@rDPe = dso_local local_unnamed_addr global %struct.reverse_iterator.2 { %struct.Double_pointer { ptr @Data } }, align 8
@rrdpb = dso_local local_unnamed_addr global %struct.reverse_iterator.3 { %struct.reverse_iterator { ptr @data } }, align 8
@rrdpe = dso_local local_unnamed_addr global %struct.reverse_iterator.3 { %struct.reverse_iterator { ptr getelementptr inbounds nuw (i8, ptr @data, i64 16000) } }, align 8
@rrDpb = dso_local local_unnamed_addr global %struct.reverse_iterator.4 { %struct.reverse_iterator.0 { ptr @Data } }, align 8
@rrDpe = dso_local local_unnamed_addr global %struct.reverse_iterator.4 { %struct.reverse_iterator.0 { ptr getelementptr inbounds nuw (i8, ptr @Data, i64 16000) } }, align 8
@rrdPb = dso_local local_unnamed_addr global %struct.reverse_iterator.5 { %struct.reverse_iterator.1 { %struct.double_pointer { ptr @data } } }, align 8
@rrdPe = dso_local local_unnamed_addr global %struct.reverse_iterator.5 { %struct.reverse_iterator.1 { %struct.double_pointer { ptr getelementptr inbounds nuw (i8, ptr @data, i64 16000) } } }, align 8
@rrDPb = dso_local local_unnamed_addr global %struct.reverse_iterator.6 { %struct.reverse_iterator.2 { %struct.Double_pointer { ptr @Data } } }, align 8
@rrDPe = dso_local local_unnamed_addr global %struct.reverse_iterator.6 { %struct.reverse_iterator.2 { %struct.Double_pointer { ptr getelementptr inbounds nuw (i8, ptr @Data, i64 16000) } } }, align 8
@.str.27 = private unnamed_addr constant [16 x i8] c"test %i failed\0A\00", align 1
@llvm.global_ctors = appending global [0 x { i32, ptr, ptr }] zeroinitializer
@str = private unnamed_addr constant [48 x i8] c"\0Atest      absolute   additions      ratio with\00", align 4
@str.28 = private unnamed_addr constant [43 x i8] c"number    time       per second     test0\0A\00", align 4

; Function Attrs: mustprogress nofree nounwind uwtable
define dso_local void @_Z9summarizev() local_unnamed_addr #0 {
  %1 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str)
  %2 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.28)
  %3 = load i32, ptr @iterations, align 4, !tbaa !6
  %4 = sitofp i32 %3 to double
  %5 = fmul double %4, 2.000000e+03
  %6 = fdiv double %5, 1.000000e+06
  %7 = load i32, ptr @current_test, align 4, !tbaa !6
  %8 = icmp sgt i32 %7, 0
  br i1 %8, label %13, label %52

9:                                                ; preds = %13
  %10 = icmp sgt i32 %26, 0
  br i1 %10, label %11, label %52

11:                                               ; preds = %9
  %12 = load double, ptr @result_times, align 8, !tbaa !10
  br label %29

13:                                               ; preds = %0, %13
  %14 = phi i64 [ %25, %13 ], [ 0, %0 ]
  %15 = getelementptr inbounds nuw double, ptr @result_times, i64 %14
  %16 = load double, ptr %15, align 8, !tbaa !10
  %17 = fmul double %16, 0x3E7AD7F29ABCAF48
  %18 = fdiv double %6, %16
  %19 = fmul double %18, 0x3E7AD7F29ABCAF48
  %20 = load double, ptr @result_times, align 8, !tbaa !10
  %21 = fdiv double %16, %20
  %22 = fmul double %21, 0x3E7AD7F29ABCAF48
  %23 = trunc nuw nsw i64 %14 to i32
  %24 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef %23, double noundef %17, double noundef %19, double noundef %22)
  %25 = add nuw nsw i64 %14, 1
  %26 = load i32, ptr @current_test, align 4, !tbaa !6
  %27 = sext i32 %26 to i64
  %28 = icmp slt i64 %25, %27
  br i1 %28, label %13, label %9, !llvm.loop !12

29:                                               ; preds = %11, %29
  %30 = phi i64 [ 0, %11 ], [ %46, %29 ]
  %31 = phi double [ 0.000000e+00, %11 ], [ %45, %29 ]
  %32 = phi double [ 0.000000e+00, %11 ], [ %42, %29 ]
  %33 = phi double [ 0.000000e+00, %11 ], [ %37, %29 ]
  %34 = phi double [ 0.000000e+00, %11 ], [ %39, %29 ]
  %35 = getelementptr inbounds nuw double, ptr @result_times, i64 %30
  %36 = load double, ptr %35, align 8, !tbaa !10
  %37 = fadd double %33, %36
  %38 = tail call double @log(double noundef %36) #9, !tbaa !6
  %39 = fadd double %34, %38
  %40 = fdiv double %6, %36
  %41 = tail call double @log(double noundef %40) #9, !tbaa !6
  %42 = fadd double %32, %41
  %43 = fdiv double %36, %12
  %44 = tail call double @log(double noundef %43) #9, !tbaa !6
  %45 = fadd double %31, %44
  %46 = add nuw nsw i64 %30, 1
  %47 = load i32, ptr @current_test, align 4, !tbaa !6
  %48 = sext i32 %47 to i64
  %49 = icmp slt i64 %46, %48
  br i1 %49, label %29, label %50, !llvm.loop !14

50:                                               ; preds = %29
  %51 = fmul double %37, 0x3E7AD7F29ABCAF48
  br label %52

52:                                               ; preds = %0, %50, %9
  %53 = phi double [ 0.000000e+00, %9 ], [ %39, %50 ], [ 0.000000e+00, %0 ]
  %54 = phi double [ 0.000000e+00, %9 ], [ %51, %50 ], [ 0.000000e+00, %0 ]
  %55 = phi double [ 0.000000e+00, %9 ], [ %42, %50 ], [ 0.000000e+00, %0 ]
  %56 = phi double [ 0.000000e+00, %9 ], [ %45, %50 ], [ 0.000000e+00, %0 ]
  %57 = phi i32 [ %26, %9 ], [ %47, %50 ], [ %7, %0 ]
  %58 = sitofp i32 %57 to double
  %59 = fdiv double %53, %58
  %60 = tail call double @exp(double noundef %59) #9, !tbaa !6
  %61 = fmul double %60, 0x3E7AD7F29ABCAF48
  %62 = load i32, ptr @current_test, align 4, !tbaa !6
  %63 = sitofp i32 %62 to double
  %64 = fdiv double %55, %63
  %65 = tail call double @exp(double noundef %64) #9, !tbaa !6
  %66 = fmul double %65, 0x3E7AD7F29ABCAF48
  %67 = load i32, ptr @current_test, align 4, !tbaa !6
  %68 = sitofp i32 %67 to double
  %69 = fdiv double %56, %68
  %70 = tail call double @exp(double noundef %69) #9, !tbaa !6
  %71 = fmul double %70, 0x3E7AD7F29ABCAF48
  %72 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, double noundef %61, double noundef %66, double noundef %71)
  %73 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.4, double noundef %54)
  %74 = load i32, ptr @current_test, align 4, !tbaa !6
  %75 = sitofp i32 %74 to double
  %76 = fdiv double %56, %75
  %77 = tail call double @exp(double noundef %76) #9, !tbaa !6
  %78 = fmul double %77, 0x3E7AD7F29ABCAF48
  %79 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.5, double noundef %78)
  ret void
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #1

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(errnomem: write)
declare double @log(double noundef) local_unnamed_addr #2

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(errnomem: write)
declare double @exp(double noundef) local_unnamed_addr #2

; Function Attrs: mustprogress nounwind uwtable
define dso_local void @_Z5test0PdS_(ptr noundef %0, ptr noundef %1) local_unnamed_addr #3 {
  %3 = tail call i64 @clock() #9
  store i64 %3, ptr @start_time, align 8, !tbaa !15
  %4 = load i32, ptr @iterations, align 4, !tbaa !6
  %5 = icmp sgt i32 %4, 0
  br i1 %5, label %6, label %60

6:                                                ; preds = %2
  %7 = ptrtoint ptr %1 to i64
  %8 = ptrtoint ptr %0 to i64
  %9 = sub i64 %7, %8
  %10 = ashr exact i64 %9, 3
  %11 = icmp sgt i64 %10, 0
  br i1 %11, label %12, label %53

12:                                               ; preds = %6
  %13 = icmp ult i64 %10, 4
  %14 = and i64 %10, 9223372036854775804
  %15 = icmp eq i64 %10, %14
  br label %16

16:                                               ; preds = %12, %38
  %17 = phi i32 [ %39, %38 ], [ %4, %12 ]
  %18 = phi i32 [ %40, %38 ], [ 0, %12 ]
  br i1 %13, label %31, label %19

19:                                               ; preds = %16, %19
  %20 = phi i64 [ %28, %19 ], [ 0, %16 ]
  %21 = phi double [ %27, %19 ], [ 0.000000e+00, %16 ]
  %22 = getelementptr inbounds nuw double, ptr %0, i64 %20
  %23 = getelementptr inbounds nuw i8, ptr %22, i64 16
  %24 = load <2 x double>, ptr %22, align 8, !tbaa !10
  %25 = load <2 x double>, ptr %23, align 8, !tbaa !10
  %26 = tail call double @llvm.vector.reduce.fadd.v2f64(double %21, <2 x double> %24)
  %27 = tail call double @llvm.vector.reduce.fadd.v2f64(double %26, <2 x double> %25)
  %28 = add nuw i64 %20, 4
  %29 = icmp eq i64 %28, %14
  br i1 %29, label %30, label %19, !llvm.loop !17

30:                                               ; preds = %19
  br i1 %15, label %50, label %31

31:                                               ; preds = %16, %30
  %32 = phi i64 [ 0, %16 ], [ %14, %30 ]
  %33 = phi double [ 0.000000e+00, %16 ], [ %27, %30 ]
  br label %42

34:                                               ; preds = %50
  %35 = load i32, ptr @current_test, align 4, !tbaa !6
  %36 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.27, i32 noundef %35)
  %37 = load i32, ptr @iterations, align 4, !tbaa !6
  br label %38

38:                                               ; preds = %34, %50
  %39 = phi i32 [ %37, %34 ], [ %17, %50 ]
  %40 = add nuw nsw i32 %18, 1
  %41 = icmp slt i32 %40, %39
  br i1 %41, label %16, label %60, !llvm.loop !20

42:                                               ; preds = %31, %42
  %43 = phi i64 [ %48, %42 ], [ %32, %31 ]
  %44 = phi double [ %47, %42 ], [ %33, %31 ]
  %45 = getelementptr inbounds nuw double, ptr %0, i64 %43
  %46 = load double, ptr %45, align 8, !tbaa !10
  %47 = fadd double %44, %46
  %48 = add nuw nsw i64 %43, 1
  %49 = icmp eq i64 %48, %10
  br i1 %49, label %50, label %42, !llvm.loop !21

50:                                               ; preds = %42, %30
  %51 = phi double [ %27, %30 ], [ %47, %42 ]
  %52 = fcmp une double %51, 6.000000e+03
  br i1 %52, label %34, label %38

53:                                               ; preds = %6, %53
  %54 = phi i32 [ %57, %53 ], [ 0, %6 ]
  %55 = load i32, ptr @current_test, align 4, !tbaa !6
  %56 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.27, i32 noundef %55)
  %57 = add nuw nsw i32 %54, 1
  %58 = load i32, ptr @iterations, align 4, !tbaa !6
  %59 = icmp slt i32 %57, %58
  br i1 %59, label %53, label %60, !llvm.loop !20

60:                                               ; preds = %53, %38, %2
  %61 = tail call i64 @clock() #9
  store i64 %61, ptr @end_time, align 8, !tbaa !15
  %62 = load i64, ptr @start_time, align 8, !tbaa !15
  %63 = sub nsw i64 %61, %62
  %64 = sitofp i64 %63 to double
  %65 = fdiv double %64, 1.000000e+06
  %66 = fadd double %65, 0x3E80000000000000
  %67 = load i32, ptr @current_test, align 4, !tbaa !6
  %68 = add nsw i32 %67, 1
  store i32 %68, ptr @current_test, align 4, !tbaa !6
  %69 = sext i32 %67 to i64
  %70 = getelementptr inbounds double, ptr @result_times, i64 %69
  store double %66, ptr %70, align 8, !tbaa !10
  ret void
}

; Function Attrs: mustprogress norecurse uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #4 {
  %3 = icmp sgt i32 %0, 1
  br i1 %3, label %4, label %9

4:                                                ; preds = %2
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %6 = load ptr, ptr %5, align 8, !tbaa !22
  %7 = tail call i64 @__isoc23_strtol(ptr noundef nonnull %6, ptr noundef null, i32 noundef 10) #9
  %8 = trunc i64 %7 to i32
  store i32 %8, ptr @iterations, align 4, !tbaa !6
  br label %9

9:                                                ; preds = %4, %2
  %10 = load ptr, ptr @dpb, align 8, !tbaa !25
  %11 = load ptr, ptr @dpe, align 8, !tbaa !25
  %12 = icmp eq ptr %10, %11
  br i1 %12, label %40, label %13

13:                                               ; preds = %9
  %14 = ptrtoint ptr %11 to i64
  %15 = ptrtoint ptr %10 to i64
  %16 = add i64 %14, -8
  %17 = sub i64 %16, %15
  %18 = lshr i64 %17, 3
  %19 = add nuw nsw i64 %18, 1
  %20 = icmp ult i64 %17, 24
  br i1 %20, label %34, label %21

21:                                               ; preds = %13
  %22 = and i64 %19, 4611686018427387900
  %23 = shl i64 %22, 3
  %24 = getelementptr i8, ptr %10, i64 %23
  br label %25

25:                                               ; preds = %25, %21
  %26 = phi i64 [ 0, %21 ], [ %30, %25 ]
  %27 = shl i64 %26, 3
  %28 = getelementptr i8, ptr %10, i64 %27
  %29 = getelementptr i8, ptr %28, i64 16
  store <2 x double> splat (double 3.000000e+00), ptr %28, align 8, !tbaa !10
  store <2 x double> splat (double 3.000000e+00), ptr %29, align 8, !tbaa !10
  %30 = add nuw i64 %26, 4
  %31 = icmp eq i64 %30, %22
  br i1 %31, label %32, label %25, !llvm.loop !27

32:                                               ; preds = %25
  %33 = icmp eq i64 %19, %22
  br i1 %33, label %40, label %34

34:                                               ; preds = %13, %32
  %35 = phi ptr [ %10, %13 ], [ %24, %32 ]
  br label %36

36:                                               ; preds = %34, %36
  %37 = phi ptr [ %38, %36 ], [ %35, %34 ]
  %38 = getelementptr inbounds nuw i8, ptr %37, i64 8
  store double 3.000000e+00, ptr %37, align 8, !tbaa !10
  %39 = icmp eq ptr %38, %11
  br i1 %39, label %40, label %36, !llvm.loop !28

40:                                               ; preds = %36, %32, %9
  %41 = load ptr, ptr @Dpb, align 8, !tbaa !29
  %42 = load ptr, ptr @Dpe, align 8, !tbaa !29
  %43 = icmp eq ptr %41, %42
  br i1 %43, label %71, label %44

44:                                               ; preds = %40
  %45 = ptrtoint ptr %42 to i64
  %46 = ptrtoint ptr %41 to i64
  %47 = add i64 %45, -8
  %48 = sub i64 %47, %46
  %49 = lshr i64 %48, 3
  %50 = add nuw nsw i64 %49, 1
  %51 = icmp ult i64 %48, 24
  br i1 %51, label %65, label %52

52:                                               ; preds = %44
  %53 = and i64 %50, 4611686018427387900
  %54 = shl i64 %53, 3
  %55 = getelementptr i8, ptr %41, i64 %54
  br label %56

56:                                               ; preds = %56, %52
  %57 = phi i64 [ 0, %52 ], [ %61, %56 ]
  %58 = shl i64 %57, 3
  %59 = getelementptr i8, ptr %41, i64 %58
  %60 = getelementptr i8, ptr %59, i64 16
  store <2 x double> splat (double 3.000000e+00), ptr %59, align 8, !tbaa !10
  store <2 x double> splat (double 3.000000e+00), ptr %60, align 8, !tbaa !10
  %61 = add nuw i64 %57, 4
  %62 = icmp eq i64 %61, %53
  br i1 %62, label %63, label %56, !llvm.loop !31

63:                                               ; preds = %56
  %64 = icmp eq i64 %50, %53
  br i1 %64, label %71, label %65

65:                                               ; preds = %44, %63
  %66 = phi ptr [ %41, %44 ], [ %55, %63 ]
  br label %67

67:                                               ; preds = %65, %67
  %68 = phi ptr [ %69, %67 ], [ %66, %65 ]
  %69 = getelementptr inbounds nuw i8, ptr %68, i64 8
  store double 3.000000e+00, ptr %68, align 8, !tbaa !10
  %70 = icmp eq ptr %69, %42
  br i1 %70, label %71, label %67, !llvm.loop !32

71:                                               ; preds = %67, %63, %40
  tail call void @_Z5test0PdS_(ptr noundef %10, ptr noundef %11)
  %72 = load ptr, ptr @dpb, align 8, !tbaa !25
  %73 = load ptr, ptr @dpe, align 8, !tbaa !25
  %74 = load double, ptr @d, align 8, !tbaa !10
  tail call void @_Z4testIPddEvT_S1_T0_(ptr noundef %72, ptr noundef %73, double noundef %74)
  %75 = load ptr, ptr @Dpb, align 8, !tbaa !29
  %76 = load ptr, ptr @Dpe, align 8, !tbaa !29
  %77 = load double, ptr @D, align 8, !tbaa !10
  %78 = insertvalue [1 x double] poison, double %77, 0
  tail call void @_Z4testIP6DoubleS0_EvT_S2_T0_(ptr noundef %75, ptr noundef %76, [1 x double] alignstack(8) %78)
  %79 = load ptr, ptr @dPb, align 8, !tbaa !25
  %80 = load ptr, ptr @dPe, align 8, !tbaa !25
  %81 = load double, ptr @d, align 8, !tbaa !10
  tail call void @_Z4testI14double_pointerdEvT_S1_T0_(ptr %79, ptr %80, double noundef %81)
  %82 = load ptr, ptr @DPb, align 8, !tbaa !29
  %83 = load ptr, ptr @DPe, align 8, !tbaa !29
  %84 = load double, ptr @D, align 8, !tbaa !10
  %85 = insertvalue [1 x double] poison, double %84, 0
  tail call void @_Z4testI14Double_pointer6DoubleEvT_S2_T0_(ptr %82, ptr %83, [1 x double] alignstack(8) %85)
  %86 = load ptr, ptr @rdpb, align 8, !tbaa !25
  %87 = load ptr, ptr @rdpe, align 8, !tbaa !25
  %88 = load double, ptr @d, align 8, !tbaa !10
  tail call void @_Z4testI16reverse_iteratorIPddEdEvT_S3_T0_(ptr %86, ptr %87, double noundef %88)
  %89 = load ptr, ptr @rDpb, align 8, !tbaa !29
  %90 = load ptr, ptr @rDpe, align 8, !tbaa !29
  %91 = load double, ptr @D, align 8, !tbaa !10
  %92 = insertvalue [1 x double] poison, double %91, 0
  tail call void @_Z4testI16reverse_iteratorIP6DoubleS1_ES1_EvT_S4_T0_(ptr %89, ptr %90, [1 x double] alignstack(8) %92)
  %93 = load ptr, ptr @rdPb, align 8, !tbaa !25
  %94 = load ptr, ptr @rdPe, align 8, !tbaa !25
  %95 = load double, ptr @d, align 8, !tbaa !10
  tail call void @_Z4testI16reverse_iteratorI14double_pointerdEdEvT_S3_T0_(ptr %93, ptr %94, double noundef %95)
  %96 = load ptr, ptr @rDPb, align 8, !tbaa !29
  %97 = load ptr, ptr @rDPe, align 8, !tbaa !29
  %98 = load double, ptr @D, align 8, !tbaa !10
  %99 = insertvalue [1 x double] poison, double %98, 0
  tail call void @_Z4testI16reverse_iteratorI14Double_pointer6DoubleES2_EvT_S4_T0_(ptr %96, ptr %97, [1 x double] alignstack(8) %99)
  %100 = load ptr, ptr @rrdpb, align 8, !tbaa !25
  %101 = load ptr, ptr @rrdpe, align 8, !tbaa !25
  %102 = load double, ptr @d, align 8, !tbaa !10
  tail call void @_Z4testI16reverse_iteratorIS0_IPddEdEdEvT_S4_T0_(ptr %100, ptr %101, double noundef %102)
  %103 = load ptr, ptr @rrDpb, align 8, !tbaa !29
  %104 = load ptr, ptr @rrDpe, align 8, !tbaa !29
  %105 = load double, ptr @D, align 8, !tbaa !10
  %106 = insertvalue [1 x double] poison, double %105, 0
  tail call void @_Z4testI16reverse_iteratorIS0_IP6DoubleS1_ES1_ES1_EvT_S5_T0_(ptr %103, ptr %104, [1 x double] alignstack(8) %106)
  %107 = load ptr, ptr @rrdPb, align 8, !tbaa !25
  %108 = load ptr, ptr @rrdPe, align 8, !tbaa !25
  %109 = load double, ptr @d, align 8, !tbaa !10
  tail call void @_Z4testI16reverse_iteratorIS0_I14double_pointerdEdEdEvT_S4_T0_(ptr %107, ptr %108, double noundef %109)
  %110 = load ptr, ptr @rrDPb, align 8, !tbaa !29
  %111 = load ptr, ptr @rrDPe, align 8, !tbaa !29
  %112 = load double, ptr @D, align 8, !tbaa !10
  %113 = insertvalue [1 x double] poison, double %112, 0
  tail call void @_Z4testI16reverse_iteratorIS0_I14Double_pointer6DoubleES2_ES2_EvT_S5_T0_(ptr %110, ptr %111, [1 x double] alignstack(8) %113)
  tail call void @_Z9summarizev()
  ret i32 0
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_Z4testIPddEvT_S1_T0_(ptr noundef %0, ptr noundef %1, double noundef %2) local_unnamed_addr #5 comdat {
  %4 = tail call i64 @clock() #9
  store i64 %4, ptr @start_time, align 8, !tbaa !15
  %5 = load i32, ptr @iterations, align 4, !tbaa !6
  %6 = icmp sgt i32 %5, 0
  br i1 %6, label %7, label %38

7:                                                ; preds = %3
  %8 = icmp eq ptr %0, %1
  br i1 %8, label %9, label %18

9:                                                ; preds = %7
  %10 = fcmp une double %2, 6.000000e+03
  br i1 %10, label %11, label %38

11:                                               ; preds = %9, %11
  %12 = phi i32 [ %15, %11 ], [ 0, %9 ]
  %13 = load i32, ptr @current_test, align 4, !tbaa !6
  %14 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.27, i32 noundef %13)
  %15 = add nuw nsw i32 %12, 1
  %16 = load i32, ptr @iterations, align 4, !tbaa !6
  %17 = icmp slt i32 %15, %16
  br i1 %17, label %11, label %38, !llvm.loop !33

18:                                               ; preds = %7, %34
  %19 = phi i32 [ %35, %34 ], [ %5, %7 ]
  %20 = phi i32 [ %36, %34 ], [ 0, %7 ]
  br label %21

21:                                               ; preds = %18, %21
  %22 = phi ptr [ %24, %21 ], [ %0, %18 ]
  %23 = phi double [ %26, %21 ], [ %2, %18 ]
  %24 = getelementptr inbounds nuw i8, ptr %22, i64 8
  %25 = load double, ptr %22, align 8, !tbaa !10
  %26 = fadd double %23, %25
  %27 = icmp eq ptr %24, %1
  br i1 %27, label %28, label %21, !llvm.loop !34

28:                                               ; preds = %21
  %29 = fcmp une double %26, 6.000000e+03
  br i1 %29, label %30, label %34

30:                                               ; preds = %28
  %31 = load i32, ptr @current_test, align 4, !tbaa !6
  %32 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.27, i32 noundef %31)
  %33 = load i32, ptr @iterations, align 4, !tbaa !6
  br label %34

34:                                               ; preds = %28, %30
  %35 = phi i32 [ %19, %28 ], [ %33, %30 ]
  %36 = add nuw nsw i32 %20, 1
  %37 = icmp slt i32 %36, %35
  br i1 %37, label %18, label %38, !llvm.loop !33

38:                                               ; preds = %34, %11, %9, %3
  %39 = tail call i64 @clock() #9
  store i64 %39, ptr @end_time, align 8, !tbaa !15
  %40 = load i64, ptr @start_time, align 8, !tbaa !15
  %41 = sub nsw i64 %39, %40
  %42 = sitofp i64 %41 to double
  %43 = fdiv double %42, 1.000000e+06
  %44 = fadd double %43, 0x3E80000000000000
  %45 = load i32, ptr @current_test, align 4, !tbaa !6
  %46 = add nsw i32 %45, 1
  store i32 %46, ptr @current_test, align 4, !tbaa !6
  %47 = sext i32 %45 to i64
  %48 = getelementptr inbounds double, ptr @result_times, i64 %47
  store double %44, ptr %48, align 8, !tbaa !10
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_Z4testIP6DoubleS0_EvT_S2_T0_(ptr noundef %0, ptr noundef %1, [1 x double] alignstack(8) %2) local_unnamed_addr #5 comdat {
  %4 = tail call i64 @clock() #9
  store i64 %4, ptr @start_time, align 8, !tbaa !15
  %5 = load i32, ptr @iterations, align 4, !tbaa !6
  %6 = icmp sgt i32 %5, 0
  br i1 %6, label %7, label %39

7:                                                ; preds = %3
  %8 = extractvalue [1 x double] %2, 0
  %9 = icmp eq ptr %0, %1
  br i1 %9, label %10, label %19

10:                                               ; preds = %7
  %11 = fcmp une double %8, 6.000000e+03
  br i1 %11, label %12, label %39

12:                                               ; preds = %10, %12
  %13 = phi i32 [ %16, %12 ], [ 0, %10 ]
  %14 = load i32, ptr @current_test, align 4, !tbaa !6
  %15 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.27, i32 noundef %14)
  %16 = add nuw nsw i32 %13, 1
  %17 = load i32, ptr @iterations, align 4, !tbaa !6
  %18 = icmp slt i32 %16, %17
  br i1 %18, label %12, label %39, !llvm.loop !35

19:                                               ; preds = %7, %35
  %20 = phi i32 [ %36, %35 ], [ %5, %7 ]
  %21 = phi i32 [ %37, %35 ], [ 0, %7 ]
  br label %22

22:                                               ; preds = %19, %22
  %23 = phi ptr [ %25, %22 ], [ %0, %19 ]
  %24 = phi double [ %27, %22 ], [ %8, %19 ]
  %25 = getelementptr inbounds nuw i8, ptr %23, i64 8
  %26 = load double, ptr %23, align 8, !tbaa !36
  %27 = fadd double %24, %26
  %28 = icmp eq ptr %25, %1
  br i1 %28, label %29, label %22, !llvm.loop !38

29:                                               ; preds = %22
  %30 = fcmp une double %27, 6.000000e+03
  br i1 %30, label %31, label %35

31:                                               ; preds = %29
  %32 = load i32, ptr @current_test, align 4, !tbaa !6
  %33 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.27, i32 noundef %32)
  %34 = load i32, ptr @iterations, align 4, !tbaa !6
  br label %35

35:                                               ; preds = %29, %31
  %36 = phi i32 [ %20, %29 ], [ %34, %31 ]
  %37 = add nuw nsw i32 %21, 1
  %38 = icmp slt i32 %37, %36
  br i1 %38, label %19, label %39, !llvm.loop !35

39:                                               ; preds = %35, %12, %10, %3
  %40 = tail call i64 @clock() #9
  store i64 %40, ptr @end_time, align 8, !tbaa !15
  %41 = load i64, ptr @start_time, align 8, !tbaa !15
  %42 = sub nsw i64 %40, %41
  %43 = sitofp i64 %42 to double
  %44 = fdiv double %43, 1.000000e+06
  %45 = fadd double %44, 0x3E80000000000000
  %46 = load i32, ptr @current_test, align 4, !tbaa !6
  %47 = add nsw i32 %46, 1
  store i32 %47, ptr @current_test, align 4, !tbaa !6
  %48 = sext i32 %46 to i64
  %49 = getelementptr inbounds double, ptr @result_times, i64 %48
  store double %45, ptr %49, align 8, !tbaa !10
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_Z4testI14double_pointerdEvT_S1_T0_(ptr %0, ptr %1, double noundef %2) local_unnamed_addr #5 comdat {
  %4 = tail call i64 @clock() #9
  store i64 %4, ptr @start_time, align 8, !tbaa !15
  %5 = load i32, ptr @iterations, align 4, !tbaa !6
  %6 = icmp sgt i32 %5, 0
  br i1 %6, label %7, label %38

7:                                                ; preds = %3
  %8 = icmp eq ptr %0, %1
  br i1 %8, label %9, label %18

9:                                                ; preds = %7
  %10 = fcmp une double %2, 6.000000e+03
  br i1 %10, label %11, label %38

11:                                               ; preds = %9, %11
  %12 = phi i32 [ %15, %11 ], [ 0, %9 ]
  %13 = load i32, ptr @current_test, align 4, !tbaa !6
  %14 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.27, i32 noundef %13)
  %15 = add nuw nsw i32 %12, 1
  %16 = load i32, ptr @iterations, align 4, !tbaa !6
  %17 = icmp slt i32 %15, %16
  br i1 %17, label %11, label %38, !llvm.loop !39

18:                                               ; preds = %7, %34
  %19 = phi i32 [ %35, %34 ], [ %5, %7 ]
  %20 = phi i32 [ %36, %34 ], [ 0, %7 ]
  br label %21

21:                                               ; preds = %18, %21
  %22 = phi ptr [ %24, %21 ], [ %0, %18 ]
  %23 = phi double [ %26, %21 ], [ %2, %18 ]
  %24 = getelementptr inbounds nuw i8, ptr %22, i64 8
  %25 = load double, ptr %22, align 8, !tbaa !10
  %26 = fadd double %23, %25
  %27 = icmp eq ptr %24, %1
  br i1 %27, label %28, label %21, !llvm.loop !40

28:                                               ; preds = %21
  %29 = fcmp une double %26, 6.000000e+03
  br i1 %29, label %30, label %34

30:                                               ; preds = %28
  %31 = load i32, ptr @current_test, align 4, !tbaa !6
  %32 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.27, i32 noundef %31)
  %33 = load i32, ptr @iterations, align 4, !tbaa !6
  br label %34

34:                                               ; preds = %28, %30
  %35 = phi i32 [ %19, %28 ], [ %33, %30 ]
  %36 = add nuw nsw i32 %20, 1
  %37 = icmp slt i32 %36, %35
  br i1 %37, label %18, label %38, !llvm.loop !39

38:                                               ; preds = %34, %11, %9, %3
  %39 = tail call i64 @clock() #9
  store i64 %39, ptr @end_time, align 8, !tbaa !15
  %40 = load i64, ptr @start_time, align 8, !tbaa !15
  %41 = sub nsw i64 %39, %40
  %42 = sitofp i64 %41 to double
  %43 = fdiv double %42, 1.000000e+06
  %44 = fadd double %43, 0x3E80000000000000
  %45 = load i32, ptr @current_test, align 4, !tbaa !6
  %46 = add nsw i32 %45, 1
  store i32 %46, ptr @current_test, align 4, !tbaa !6
  %47 = sext i32 %45 to i64
  %48 = getelementptr inbounds double, ptr @result_times, i64 %47
  store double %44, ptr %48, align 8, !tbaa !10
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_Z4testI14Double_pointer6DoubleEvT_S2_T0_(ptr %0, ptr %1, [1 x double] alignstack(8) %2) local_unnamed_addr #5 comdat {
  %4 = tail call i64 @clock() #9
  store i64 %4, ptr @start_time, align 8, !tbaa !15
  %5 = load i32, ptr @iterations, align 4, !tbaa !6
  %6 = icmp sgt i32 %5, 0
  br i1 %6, label %7, label %39

7:                                                ; preds = %3
  %8 = extractvalue [1 x double] %2, 0
  %9 = icmp eq ptr %0, %1
  br i1 %9, label %10, label %19

10:                                               ; preds = %7
  %11 = fcmp une double %8, 6.000000e+03
  br i1 %11, label %12, label %39

12:                                               ; preds = %10, %12
  %13 = phi i32 [ %16, %12 ], [ 0, %10 ]
  %14 = load i32, ptr @current_test, align 4, !tbaa !6
  %15 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.27, i32 noundef %14)
  %16 = add nuw nsw i32 %13, 1
  %17 = load i32, ptr @iterations, align 4, !tbaa !6
  %18 = icmp slt i32 %16, %17
  br i1 %18, label %12, label %39, !llvm.loop !41

19:                                               ; preds = %7, %35
  %20 = phi i32 [ %36, %35 ], [ %5, %7 ]
  %21 = phi i32 [ %37, %35 ], [ 0, %7 ]
  br label %22

22:                                               ; preds = %19, %22
  %23 = phi ptr [ %25, %22 ], [ %0, %19 ]
  %24 = phi double [ %27, %22 ], [ %8, %19 ]
  %25 = getelementptr inbounds nuw i8, ptr %23, i64 8
  %26 = load double, ptr %23, align 8, !tbaa !36
  %27 = fadd double %24, %26
  %28 = icmp eq ptr %25, %1
  br i1 %28, label %29, label %22, !llvm.loop !42

29:                                               ; preds = %22
  %30 = fcmp une double %27, 6.000000e+03
  br i1 %30, label %31, label %35

31:                                               ; preds = %29
  %32 = load i32, ptr @current_test, align 4, !tbaa !6
  %33 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.27, i32 noundef %32)
  %34 = load i32, ptr @iterations, align 4, !tbaa !6
  br label %35

35:                                               ; preds = %29, %31
  %36 = phi i32 [ %20, %29 ], [ %34, %31 ]
  %37 = add nuw nsw i32 %21, 1
  %38 = icmp slt i32 %37, %36
  br i1 %38, label %19, label %39, !llvm.loop !41

39:                                               ; preds = %35, %12, %10, %3
  %40 = tail call i64 @clock() #9
  store i64 %40, ptr @end_time, align 8, !tbaa !15
  %41 = load i64, ptr @start_time, align 8, !tbaa !15
  %42 = sub nsw i64 %40, %41
  %43 = sitofp i64 %42 to double
  %44 = fdiv double %43, 1.000000e+06
  %45 = fadd double %44, 0x3E80000000000000
  %46 = load i32, ptr @current_test, align 4, !tbaa !6
  %47 = add nsw i32 %46, 1
  store i32 %47, ptr @current_test, align 4, !tbaa !6
  %48 = sext i32 %46 to i64
  %49 = getelementptr inbounds double, ptr @result_times, i64 %48
  store double %45, ptr %49, align 8, !tbaa !10
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_Z4testI16reverse_iteratorIPddEdEvT_S3_T0_(ptr %0, ptr %1, double noundef %2) local_unnamed_addr #5 comdat {
  %4 = tail call i64 @clock() #9
  store i64 %4, ptr @start_time, align 8, !tbaa !15
  %5 = load i32, ptr @iterations, align 4, !tbaa !6
  %6 = icmp sgt i32 %5, 0
  br i1 %6, label %7, label %38

7:                                                ; preds = %3
  %8 = icmp eq ptr %0, %1
  br i1 %8, label %9, label %18

9:                                                ; preds = %7
  %10 = fcmp une double %2, 6.000000e+03
  br i1 %10, label %11, label %38

11:                                               ; preds = %9, %11
  %12 = phi i32 [ %15, %11 ], [ 0, %9 ]
  %13 = load i32, ptr @current_test, align 4, !tbaa !6
  %14 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.27, i32 noundef %13)
  %15 = add nuw nsw i32 %12, 1
  %16 = load i32, ptr @iterations, align 4, !tbaa !6
  %17 = icmp slt i32 %15, %16
  br i1 %17, label %11, label %38, !llvm.loop !43

18:                                               ; preds = %7, %34
  %19 = phi i32 [ %35, %34 ], [ %5, %7 ]
  %20 = phi i32 [ %36, %34 ], [ 0, %7 ]
  br label %21

21:                                               ; preds = %18, %21
  %22 = phi ptr [ %24, %21 ], [ %0, %18 ]
  %23 = phi double [ %26, %21 ], [ %2, %18 ]
  %24 = getelementptr inbounds i8, ptr %22, i64 -8
  %25 = load double, ptr %24, align 8, !tbaa !10
  %26 = fadd double %23, %25
  %27 = icmp eq ptr %24, %1
  br i1 %27, label %28, label %21, !llvm.loop !44

28:                                               ; preds = %21
  %29 = fcmp une double %26, 6.000000e+03
  br i1 %29, label %30, label %34

30:                                               ; preds = %28
  %31 = load i32, ptr @current_test, align 4, !tbaa !6
  %32 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.27, i32 noundef %31)
  %33 = load i32, ptr @iterations, align 4, !tbaa !6
  br label %34

34:                                               ; preds = %28, %30
  %35 = phi i32 [ %19, %28 ], [ %33, %30 ]
  %36 = add nuw nsw i32 %20, 1
  %37 = icmp slt i32 %36, %35
  br i1 %37, label %18, label %38, !llvm.loop !43

38:                                               ; preds = %34, %11, %9, %3
  %39 = tail call i64 @clock() #9
  store i64 %39, ptr @end_time, align 8, !tbaa !15
  %40 = load i64, ptr @start_time, align 8, !tbaa !15
  %41 = sub nsw i64 %39, %40
  %42 = sitofp i64 %41 to double
  %43 = fdiv double %42, 1.000000e+06
  %44 = fadd double %43, 0x3E80000000000000
  %45 = load i32, ptr @current_test, align 4, !tbaa !6
  %46 = add nsw i32 %45, 1
  store i32 %46, ptr @current_test, align 4, !tbaa !6
  %47 = sext i32 %45 to i64
  %48 = getelementptr inbounds double, ptr @result_times, i64 %47
  store double %44, ptr %48, align 8, !tbaa !10
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_Z4testI16reverse_iteratorIP6DoubleS1_ES1_EvT_S4_T0_(ptr %0, ptr %1, [1 x double] alignstack(8) %2) local_unnamed_addr #5 comdat {
  %4 = tail call i64 @clock() #9
  store i64 %4, ptr @start_time, align 8, !tbaa !15
  %5 = load i32, ptr @iterations, align 4, !tbaa !6
  %6 = icmp sgt i32 %5, 0
  br i1 %6, label %7, label %39

7:                                                ; preds = %3
  %8 = extractvalue [1 x double] %2, 0
  %9 = icmp eq ptr %0, %1
  br i1 %9, label %10, label %19

10:                                               ; preds = %7
  %11 = fcmp une double %8, 6.000000e+03
  br i1 %11, label %12, label %39

12:                                               ; preds = %10, %12
  %13 = phi i32 [ %16, %12 ], [ 0, %10 ]
  %14 = load i32, ptr @current_test, align 4, !tbaa !6
  %15 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.27, i32 noundef %14)
  %16 = add nuw nsw i32 %13, 1
  %17 = load i32, ptr @iterations, align 4, !tbaa !6
  %18 = icmp slt i32 %16, %17
  br i1 %18, label %12, label %39, !llvm.loop !45

19:                                               ; preds = %7, %35
  %20 = phi i32 [ %36, %35 ], [ %5, %7 ]
  %21 = phi i32 [ %37, %35 ], [ 0, %7 ]
  br label %22

22:                                               ; preds = %19, %22
  %23 = phi ptr [ %25, %22 ], [ %0, %19 ]
  %24 = phi double [ %27, %22 ], [ %8, %19 ]
  %25 = getelementptr inbounds i8, ptr %23, i64 -8
  %26 = load double, ptr %25, align 8, !tbaa !36
  %27 = fadd double %24, %26
  %28 = icmp eq ptr %25, %1
  br i1 %28, label %29, label %22, !llvm.loop !46

29:                                               ; preds = %22
  %30 = fcmp une double %27, 6.000000e+03
  br i1 %30, label %31, label %35

31:                                               ; preds = %29
  %32 = load i32, ptr @current_test, align 4, !tbaa !6
  %33 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.27, i32 noundef %32)
  %34 = load i32, ptr @iterations, align 4, !tbaa !6
  br label %35

35:                                               ; preds = %29, %31
  %36 = phi i32 [ %20, %29 ], [ %34, %31 ]
  %37 = add nuw nsw i32 %21, 1
  %38 = icmp slt i32 %37, %36
  br i1 %38, label %19, label %39, !llvm.loop !45

39:                                               ; preds = %35, %12, %10, %3
  %40 = tail call i64 @clock() #9
  store i64 %40, ptr @end_time, align 8, !tbaa !15
  %41 = load i64, ptr @start_time, align 8, !tbaa !15
  %42 = sub nsw i64 %40, %41
  %43 = sitofp i64 %42 to double
  %44 = fdiv double %43, 1.000000e+06
  %45 = fadd double %44, 0x3E80000000000000
  %46 = load i32, ptr @current_test, align 4, !tbaa !6
  %47 = add nsw i32 %46, 1
  store i32 %47, ptr @current_test, align 4, !tbaa !6
  %48 = sext i32 %46 to i64
  %49 = getelementptr inbounds double, ptr @result_times, i64 %48
  store double %45, ptr %49, align 8, !tbaa !10
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_Z4testI16reverse_iteratorI14double_pointerdEdEvT_S3_T0_(ptr %0, ptr %1, double noundef %2) local_unnamed_addr #5 comdat {
  %4 = tail call i64 @clock() #9
  store i64 %4, ptr @start_time, align 8, !tbaa !15
  %5 = load i32, ptr @iterations, align 4, !tbaa !6
  %6 = icmp sgt i32 %5, 0
  br i1 %6, label %7, label %38

7:                                                ; preds = %3
  %8 = icmp eq ptr %0, %1
  br i1 %8, label %9, label %18

9:                                                ; preds = %7
  %10 = fcmp une double %2, 6.000000e+03
  br i1 %10, label %11, label %38

11:                                               ; preds = %9, %11
  %12 = phi i32 [ %15, %11 ], [ 0, %9 ]
  %13 = load i32, ptr @current_test, align 4, !tbaa !6
  %14 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.27, i32 noundef %13)
  %15 = add nuw nsw i32 %12, 1
  %16 = load i32, ptr @iterations, align 4, !tbaa !6
  %17 = icmp slt i32 %15, %16
  br i1 %17, label %11, label %38, !llvm.loop !47

18:                                               ; preds = %7, %34
  %19 = phi i32 [ %35, %34 ], [ %5, %7 ]
  %20 = phi i32 [ %36, %34 ], [ 0, %7 ]
  br label %21

21:                                               ; preds = %18, %21
  %22 = phi ptr [ %24, %21 ], [ %0, %18 ]
  %23 = phi double [ %26, %21 ], [ %2, %18 ]
  %24 = getelementptr inbounds i8, ptr %22, i64 -8
  %25 = load double, ptr %24, align 8, !tbaa !10
  %26 = fadd double %23, %25
  %27 = icmp eq ptr %24, %1
  br i1 %27, label %28, label %21, !llvm.loop !48

28:                                               ; preds = %21
  %29 = fcmp une double %26, 6.000000e+03
  br i1 %29, label %30, label %34

30:                                               ; preds = %28
  %31 = load i32, ptr @current_test, align 4, !tbaa !6
  %32 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.27, i32 noundef %31)
  %33 = load i32, ptr @iterations, align 4, !tbaa !6
  br label %34

34:                                               ; preds = %28, %30
  %35 = phi i32 [ %19, %28 ], [ %33, %30 ]
  %36 = add nuw nsw i32 %20, 1
  %37 = icmp slt i32 %36, %35
  br i1 %37, label %18, label %38, !llvm.loop !47

38:                                               ; preds = %34, %11, %9, %3
  %39 = tail call i64 @clock() #9
  store i64 %39, ptr @end_time, align 8, !tbaa !15
  %40 = load i64, ptr @start_time, align 8, !tbaa !15
  %41 = sub nsw i64 %39, %40
  %42 = sitofp i64 %41 to double
  %43 = fdiv double %42, 1.000000e+06
  %44 = fadd double %43, 0x3E80000000000000
  %45 = load i32, ptr @current_test, align 4, !tbaa !6
  %46 = add nsw i32 %45, 1
  store i32 %46, ptr @current_test, align 4, !tbaa !6
  %47 = sext i32 %45 to i64
  %48 = getelementptr inbounds double, ptr @result_times, i64 %47
  store double %44, ptr %48, align 8, !tbaa !10
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_Z4testI16reverse_iteratorI14Double_pointer6DoubleES2_EvT_S4_T0_(ptr %0, ptr %1, [1 x double] alignstack(8) %2) local_unnamed_addr #5 comdat {
  %4 = tail call i64 @clock() #9
  store i64 %4, ptr @start_time, align 8, !tbaa !15
  %5 = load i32, ptr @iterations, align 4, !tbaa !6
  %6 = icmp sgt i32 %5, 0
  br i1 %6, label %7, label %39

7:                                                ; preds = %3
  %8 = extractvalue [1 x double] %2, 0
  %9 = icmp eq ptr %0, %1
  br i1 %9, label %10, label %19

10:                                               ; preds = %7
  %11 = fcmp une double %8, 6.000000e+03
  br i1 %11, label %12, label %39

12:                                               ; preds = %10, %12
  %13 = phi i32 [ %16, %12 ], [ 0, %10 ]
  %14 = load i32, ptr @current_test, align 4, !tbaa !6
  %15 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.27, i32 noundef %14)
  %16 = add nuw nsw i32 %13, 1
  %17 = load i32, ptr @iterations, align 4, !tbaa !6
  %18 = icmp slt i32 %16, %17
  br i1 %18, label %12, label %39, !llvm.loop !49

19:                                               ; preds = %7, %35
  %20 = phi i32 [ %36, %35 ], [ %5, %7 ]
  %21 = phi i32 [ %37, %35 ], [ 0, %7 ]
  br label %22

22:                                               ; preds = %19, %22
  %23 = phi ptr [ %25, %22 ], [ %0, %19 ]
  %24 = phi double [ %27, %22 ], [ %8, %19 ]
  %25 = getelementptr inbounds i8, ptr %23, i64 -8
  %26 = load double, ptr %25, align 8, !tbaa !36
  %27 = fadd double %24, %26
  %28 = icmp eq ptr %25, %1
  br i1 %28, label %29, label %22, !llvm.loop !50

29:                                               ; preds = %22
  %30 = fcmp une double %27, 6.000000e+03
  br i1 %30, label %31, label %35

31:                                               ; preds = %29
  %32 = load i32, ptr @current_test, align 4, !tbaa !6
  %33 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.27, i32 noundef %32)
  %34 = load i32, ptr @iterations, align 4, !tbaa !6
  br label %35

35:                                               ; preds = %29, %31
  %36 = phi i32 [ %20, %29 ], [ %34, %31 ]
  %37 = add nuw nsw i32 %21, 1
  %38 = icmp slt i32 %37, %36
  br i1 %38, label %19, label %39, !llvm.loop !49

39:                                               ; preds = %35, %12, %10, %3
  %40 = tail call i64 @clock() #9
  store i64 %40, ptr @end_time, align 8, !tbaa !15
  %41 = load i64, ptr @start_time, align 8, !tbaa !15
  %42 = sub nsw i64 %40, %41
  %43 = sitofp i64 %42 to double
  %44 = fdiv double %43, 1.000000e+06
  %45 = fadd double %44, 0x3E80000000000000
  %46 = load i32, ptr @current_test, align 4, !tbaa !6
  %47 = add nsw i32 %46, 1
  store i32 %47, ptr @current_test, align 4, !tbaa !6
  %48 = sext i32 %46 to i64
  %49 = getelementptr inbounds double, ptr @result_times, i64 %48
  store double %45, ptr %49, align 8, !tbaa !10
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_Z4testI16reverse_iteratorIS0_IPddEdEdEvT_S4_T0_(ptr %0, ptr %1, double noundef %2) local_unnamed_addr #5 comdat {
  %4 = tail call i64 @clock() #9
  store i64 %4, ptr @start_time, align 8, !tbaa !15
  %5 = load i32, ptr @iterations, align 4, !tbaa !6
  %6 = icmp sgt i32 %5, 0
  br i1 %6, label %7, label %38

7:                                                ; preds = %3
  %8 = icmp eq ptr %0, %1
  br i1 %8, label %9, label %18

9:                                                ; preds = %7
  %10 = fcmp une double %2, 6.000000e+03
  br i1 %10, label %11, label %38

11:                                               ; preds = %9, %11
  %12 = phi i32 [ %15, %11 ], [ 0, %9 ]
  %13 = load i32, ptr @current_test, align 4, !tbaa !6
  %14 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.27, i32 noundef %13)
  %15 = add nuw nsw i32 %12, 1
  %16 = load i32, ptr @iterations, align 4, !tbaa !6
  %17 = icmp slt i32 %15, %16
  br i1 %17, label %11, label %38, !llvm.loop !51

18:                                               ; preds = %7, %34
  %19 = phi i32 [ %35, %34 ], [ %5, %7 ]
  %20 = phi i32 [ %36, %34 ], [ 0, %7 ]
  br label %21

21:                                               ; preds = %18, %21
  %22 = phi ptr [ %24, %21 ], [ %0, %18 ]
  %23 = phi double [ %26, %21 ], [ %2, %18 ]
  %24 = getelementptr inbounds nuw i8, ptr %22, i64 8
  %25 = load double, ptr %22, align 8, !tbaa !10
  %26 = fadd double %23, %25
  %27 = icmp eq ptr %24, %1
  br i1 %27, label %28, label %21, !llvm.loop !52

28:                                               ; preds = %21
  %29 = fcmp une double %26, 6.000000e+03
  br i1 %29, label %30, label %34

30:                                               ; preds = %28
  %31 = load i32, ptr @current_test, align 4, !tbaa !6
  %32 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.27, i32 noundef %31)
  %33 = load i32, ptr @iterations, align 4, !tbaa !6
  br label %34

34:                                               ; preds = %28, %30
  %35 = phi i32 [ %19, %28 ], [ %33, %30 ]
  %36 = add nuw nsw i32 %20, 1
  %37 = icmp slt i32 %36, %35
  br i1 %37, label %18, label %38, !llvm.loop !51

38:                                               ; preds = %34, %11, %9, %3
  %39 = tail call i64 @clock() #9
  store i64 %39, ptr @end_time, align 8, !tbaa !15
  %40 = load i64, ptr @start_time, align 8, !tbaa !15
  %41 = sub nsw i64 %39, %40
  %42 = sitofp i64 %41 to double
  %43 = fdiv double %42, 1.000000e+06
  %44 = fadd double %43, 0x3E80000000000000
  %45 = load i32, ptr @current_test, align 4, !tbaa !6
  %46 = add nsw i32 %45, 1
  store i32 %46, ptr @current_test, align 4, !tbaa !6
  %47 = sext i32 %45 to i64
  %48 = getelementptr inbounds double, ptr @result_times, i64 %47
  store double %44, ptr %48, align 8, !tbaa !10
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_Z4testI16reverse_iteratorIS0_IP6DoubleS1_ES1_ES1_EvT_S5_T0_(ptr %0, ptr %1, [1 x double] alignstack(8) %2) local_unnamed_addr #5 comdat {
  %4 = tail call i64 @clock() #9
  store i64 %4, ptr @start_time, align 8, !tbaa !15
  %5 = load i32, ptr @iterations, align 4, !tbaa !6
  %6 = icmp sgt i32 %5, 0
  br i1 %6, label %7, label %39

7:                                                ; preds = %3
  %8 = extractvalue [1 x double] %2, 0
  %9 = icmp eq ptr %0, %1
  br i1 %9, label %10, label %19

10:                                               ; preds = %7
  %11 = fcmp une double %8, 6.000000e+03
  br i1 %11, label %12, label %39

12:                                               ; preds = %10, %12
  %13 = phi i32 [ %16, %12 ], [ 0, %10 ]
  %14 = load i32, ptr @current_test, align 4, !tbaa !6
  %15 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.27, i32 noundef %14)
  %16 = add nuw nsw i32 %13, 1
  %17 = load i32, ptr @iterations, align 4, !tbaa !6
  %18 = icmp slt i32 %16, %17
  br i1 %18, label %12, label %39, !llvm.loop !53

19:                                               ; preds = %7, %35
  %20 = phi i32 [ %36, %35 ], [ %5, %7 ]
  %21 = phi i32 [ %37, %35 ], [ 0, %7 ]
  br label %22

22:                                               ; preds = %19, %22
  %23 = phi ptr [ %25, %22 ], [ %0, %19 ]
  %24 = phi double [ %27, %22 ], [ %8, %19 ]
  %25 = getelementptr inbounds nuw i8, ptr %23, i64 8
  %26 = load double, ptr %23, align 8, !tbaa !36
  %27 = fadd double %24, %26
  %28 = icmp eq ptr %25, %1
  br i1 %28, label %29, label %22, !llvm.loop !54

29:                                               ; preds = %22
  %30 = fcmp une double %27, 6.000000e+03
  br i1 %30, label %31, label %35

31:                                               ; preds = %29
  %32 = load i32, ptr @current_test, align 4, !tbaa !6
  %33 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.27, i32 noundef %32)
  %34 = load i32, ptr @iterations, align 4, !tbaa !6
  br label %35

35:                                               ; preds = %29, %31
  %36 = phi i32 [ %20, %29 ], [ %34, %31 ]
  %37 = add nuw nsw i32 %21, 1
  %38 = icmp slt i32 %37, %36
  br i1 %38, label %19, label %39, !llvm.loop !53

39:                                               ; preds = %35, %12, %10, %3
  %40 = tail call i64 @clock() #9
  store i64 %40, ptr @end_time, align 8, !tbaa !15
  %41 = load i64, ptr @start_time, align 8, !tbaa !15
  %42 = sub nsw i64 %40, %41
  %43 = sitofp i64 %42 to double
  %44 = fdiv double %43, 1.000000e+06
  %45 = fadd double %44, 0x3E80000000000000
  %46 = load i32, ptr @current_test, align 4, !tbaa !6
  %47 = add nsw i32 %46, 1
  store i32 %47, ptr @current_test, align 4, !tbaa !6
  %48 = sext i32 %46 to i64
  %49 = getelementptr inbounds double, ptr @result_times, i64 %48
  store double %45, ptr %49, align 8, !tbaa !10
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_Z4testI16reverse_iteratorIS0_I14double_pointerdEdEdEvT_S4_T0_(ptr %0, ptr %1, double noundef %2) local_unnamed_addr #5 comdat {
  %4 = tail call i64 @clock() #9
  store i64 %4, ptr @start_time, align 8, !tbaa !15
  %5 = load i32, ptr @iterations, align 4, !tbaa !6
  %6 = icmp sgt i32 %5, 0
  br i1 %6, label %7, label %38

7:                                                ; preds = %3
  %8 = icmp eq ptr %0, %1
  br i1 %8, label %9, label %18

9:                                                ; preds = %7
  %10 = fcmp une double %2, 6.000000e+03
  br i1 %10, label %11, label %38

11:                                               ; preds = %9, %11
  %12 = phi i32 [ %15, %11 ], [ 0, %9 ]
  %13 = load i32, ptr @current_test, align 4, !tbaa !6
  %14 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.27, i32 noundef %13)
  %15 = add nuw nsw i32 %12, 1
  %16 = load i32, ptr @iterations, align 4, !tbaa !6
  %17 = icmp slt i32 %15, %16
  br i1 %17, label %11, label %38, !llvm.loop !55

18:                                               ; preds = %7, %34
  %19 = phi i32 [ %35, %34 ], [ %5, %7 ]
  %20 = phi i32 [ %36, %34 ], [ 0, %7 ]
  br label %21

21:                                               ; preds = %18, %21
  %22 = phi ptr [ %24, %21 ], [ %0, %18 ]
  %23 = phi double [ %26, %21 ], [ %2, %18 ]
  %24 = getelementptr inbounds nuw i8, ptr %22, i64 8
  %25 = load double, ptr %22, align 8, !tbaa !10
  %26 = fadd double %23, %25
  %27 = icmp eq ptr %24, %1
  br i1 %27, label %28, label %21, !llvm.loop !56

28:                                               ; preds = %21
  %29 = fcmp une double %26, 6.000000e+03
  br i1 %29, label %30, label %34

30:                                               ; preds = %28
  %31 = load i32, ptr @current_test, align 4, !tbaa !6
  %32 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.27, i32 noundef %31)
  %33 = load i32, ptr @iterations, align 4, !tbaa !6
  br label %34

34:                                               ; preds = %28, %30
  %35 = phi i32 [ %19, %28 ], [ %33, %30 ]
  %36 = add nuw nsw i32 %20, 1
  %37 = icmp slt i32 %36, %35
  br i1 %37, label %18, label %38, !llvm.loop !55

38:                                               ; preds = %34, %11, %9, %3
  %39 = tail call i64 @clock() #9
  store i64 %39, ptr @end_time, align 8, !tbaa !15
  %40 = load i64, ptr @start_time, align 8, !tbaa !15
  %41 = sub nsw i64 %39, %40
  %42 = sitofp i64 %41 to double
  %43 = fdiv double %42, 1.000000e+06
  %44 = fadd double %43, 0x3E80000000000000
  %45 = load i32, ptr @current_test, align 4, !tbaa !6
  %46 = add nsw i32 %45, 1
  store i32 %46, ptr @current_test, align 4, !tbaa !6
  %47 = sext i32 %45 to i64
  %48 = getelementptr inbounds double, ptr @result_times, i64 %47
  store double %44, ptr %48, align 8, !tbaa !10
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_Z4testI16reverse_iteratorIS0_I14Double_pointer6DoubleES2_ES2_EvT_S5_T0_(ptr %0, ptr %1, [1 x double] alignstack(8) %2) local_unnamed_addr #5 comdat {
  %4 = tail call i64 @clock() #9
  store i64 %4, ptr @start_time, align 8, !tbaa !15
  %5 = load i32, ptr @iterations, align 4, !tbaa !6
  %6 = icmp sgt i32 %5, 0
  br i1 %6, label %7, label %39

7:                                                ; preds = %3
  %8 = extractvalue [1 x double] %2, 0
  %9 = icmp eq ptr %0, %1
  br i1 %9, label %10, label %19

10:                                               ; preds = %7
  %11 = fcmp une double %8, 6.000000e+03
  br i1 %11, label %12, label %39

12:                                               ; preds = %10, %12
  %13 = phi i32 [ %16, %12 ], [ 0, %10 ]
  %14 = load i32, ptr @current_test, align 4, !tbaa !6
  %15 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.27, i32 noundef %14)
  %16 = add nuw nsw i32 %13, 1
  %17 = load i32, ptr @iterations, align 4, !tbaa !6
  %18 = icmp slt i32 %16, %17
  br i1 %18, label %12, label %39, !llvm.loop !57

19:                                               ; preds = %7, %35
  %20 = phi i32 [ %36, %35 ], [ %5, %7 ]
  %21 = phi i32 [ %37, %35 ], [ 0, %7 ]
  br label %22

22:                                               ; preds = %19, %22
  %23 = phi ptr [ %25, %22 ], [ %0, %19 ]
  %24 = phi double [ %27, %22 ], [ %8, %19 ]
  %25 = getelementptr inbounds nuw i8, ptr %23, i64 8
  %26 = load double, ptr %23, align 8, !tbaa !36
  %27 = fadd double %24, %26
  %28 = icmp eq ptr %25, %1
  br i1 %28, label %29, label %22, !llvm.loop !58

29:                                               ; preds = %22
  %30 = fcmp une double %27, 6.000000e+03
  br i1 %30, label %31, label %35

31:                                               ; preds = %29
  %32 = load i32, ptr @current_test, align 4, !tbaa !6
  %33 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.27, i32 noundef %32)
  %34 = load i32, ptr @iterations, align 4, !tbaa !6
  br label %35

35:                                               ; preds = %29, %31
  %36 = phi i32 [ %20, %29 ], [ %34, %31 ]
  %37 = add nuw nsw i32 %21, 1
  %38 = icmp slt i32 %37, %36
  br i1 %38, label %19, label %39, !llvm.loop !57

39:                                               ; preds = %35, %12, %10, %3
  %40 = tail call i64 @clock() #9
  store i64 %40, ptr @end_time, align 8, !tbaa !15
  %41 = load i64, ptr @start_time, align 8, !tbaa !15
  %42 = sub nsw i64 %40, %41
  %43 = sitofp i64 %42 to double
  %44 = fdiv double %43, 1.000000e+06
  %45 = fadd double %44, 0x3E80000000000000
  %46 = load i32, ptr @current_test, align 4, !tbaa !6
  %47 = add nsw i32 %46, 1
  store i32 %47, ptr @current_test, align 4, !tbaa !6
  %48 = sext i32 %46 to i64
  %49 = getelementptr inbounds double, ptr @result_times, i64 %48
  store double %45, ptr %49, align 8, !tbaa !10
  ret void
}

; Function Attrs: nounwind
declare i64 @clock() local_unnamed_addr #6

; Function Attrs: nounwind
declare i64 @__isoc23_strtol(ptr noundef, ptr noundef, i32 noundef) local_unnamed_addr #6

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr noundef readonly captures(none)) local_unnamed_addr #7

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.vector.reduce.fadd.v2f64(double, <2 x double>) #8

attributes #0 = { mustprogress nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nocallback nofree nounwind willreturn memory(errnomem: write) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { mustprogress norecurse uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { mustprogress uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { nofree nounwind }
attributes #8 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #9 = { nounwind }

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
!9 = !{!"Simple C++ TBAA"}
!10 = !{!11, !11, i64 0}
!11 = !{!"double", !8, i64 0}
!12 = distinct !{!12, !13}
!13 = !{!"llvm.loop.mustprogress"}
!14 = distinct !{!14, !13}
!15 = !{!16, !16, i64 0}
!16 = !{!"long", !8, i64 0}
!17 = distinct !{!17, !13, !18, !19}
!18 = !{!"llvm.loop.isvectorized", i32 1}
!19 = !{!"llvm.loop.unroll.runtime.disable"}
!20 = distinct !{!20, !13}
!21 = distinct !{!21, !13, !19, !18}
!22 = !{!23, !23, i64 0}
!23 = !{!"p1 omnipotent char", !24, i64 0}
!24 = !{!"any pointer", !8, i64 0}
!25 = !{!26, !26, i64 0}
!26 = !{!"p1 double", !24, i64 0}
!27 = distinct !{!27, !13, !18, !19}
!28 = distinct !{!28, !13, !19, !18}
!29 = !{!30, !30, i64 0}
!30 = !{!"p1 _ZTS6Double", !24, i64 0}
!31 = distinct !{!31, !13, !18, !19}
!32 = distinct !{!32, !13, !19, !18}
!33 = distinct !{!33, !13}
!34 = distinct !{!34, !13}
!35 = distinct !{!35, !13}
!36 = !{!37, !11, i64 0}
!37 = !{!"_ZTS6Double", !11, i64 0}
!38 = distinct !{!38, !13}
!39 = distinct !{!39, !13}
!40 = distinct !{!40, !13}
!41 = distinct !{!41, !13}
!42 = distinct !{!42, !13}
!43 = distinct !{!43, !13}
!44 = distinct !{!44, !13}
!45 = distinct !{!45, !13}
!46 = distinct !{!46, !13}
!47 = distinct !{!47, !13}
!48 = distinct !{!48, !13}
!49 = distinct !{!49, !13}
!50 = distinct !{!50, !13}
!51 = distinct !{!51, !13}
!52 = distinct !{!52, !13}
!53 = distinct !{!53, !13}
!54 = distinct !{!54, !13}
!55 = distinct !{!55, !13}
!56 = distinct !{!56, !13}
!57 = distinct !{!57, !13}
!58 = distinct !{!58, !13}
