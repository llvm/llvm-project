; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Adobe-C++/stepanov_abstraction.cpp'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Adobe-C++/stepanov_abstraction.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.ValueWrapper = type { double }
%struct.ValueWrapper.0 = type { %struct.ValueWrapper.1 }
%struct.ValueWrapper.1 = type { %struct.ValueWrapper.2 }
%struct.ValueWrapper.2 = type { %struct.ValueWrapper.3 }
%struct.ValueWrapper.3 = type { %struct.ValueWrapper.4 }
%struct.ValueWrapper.4 = type { %struct.ValueWrapper.5 }
%struct.ValueWrapper.5 = type { %struct.ValueWrapper.6 }
%struct.ValueWrapper.6 = type { %struct.ValueWrapper.7 }
%struct.ValueWrapper.7 = type { %struct.ValueWrapper.8 }
%struct.ValueWrapper.8 = type { %struct.ValueWrapper }
%struct.PointerWrapper = type { ptr }
%struct.PointerWrapper.9 = type { ptr }
%struct.PointerWrapper.10 = type { ptr }
%struct.one_result = type { double, ptr }

$_Z19test_insertion_sortIPddEvT_S1_S1_S1_T0_PKc = comdat any

$_Z19test_insertion_sortI14PointerWrapperIdEdEvT_S2_S2_S2_T0_PKc = comdat any

$_Z19test_insertion_sortIP12ValueWrapperIdES1_EvT_S3_S3_S3_T0_PKc = comdat any

$_Z19test_insertion_sortI14PointerWrapperI12ValueWrapperIdEES2_EvT_S4_S4_S4_T0_PKc = comdat any

$_Z19test_insertion_sortIP12ValueWrapperIS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IdEEEEEEEEEESA_EvT_SC_SC_SC_T0_PKc = comdat any

$_Z19test_insertion_sortI14PointerWrapperI12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEEESB_EvT_SD_SD_SD_T0_PKc = comdat any

$_Z14test_quicksortIPddEvT_S1_S1_S1_T0_PKc = comdat any

$_Z14test_quicksortI14PointerWrapperIdEdEvT_S2_S2_S2_T0_PKc = comdat any

$_Z14test_quicksortIP12ValueWrapperIdES1_EvT_S3_S3_S3_T0_PKc = comdat any

$_Z14test_quicksortI14PointerWrapperI12ValueWrapperIdEES2_EvT_S4_S4_S4_T0_PKc = comdat any

$_Z14test_quicksortIP12ValueWrapperIS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IdEEEEEEEEEESA_EvT_SC_SC_SC_T0_PKc = comdat any

$_Z14test_quicksortI14PointerWrapperI12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEEESB_EvT_SD_SD_SD_T0_PKc = comdat any

$_Z14test_heap_sortIPddEvT_S1_S1_S1_T0_PKc = comdat any

$_Z14test_heap_sortI14PointerWrapperIdEdEvT_S2_S2_S2_T0_PKc = comdat any

$_Z14test_heap_sortIP12ValueWrapperIdES1_EvT_S3_S3_S3_T0_PKc = comdat any

$_Z14test_heap_sortI14PointerWrapperI12ValueWrapperIdEES2_EvT_S4_S4_S4_T0_PKc = comdat any

$_Z14test_heap_sortIP12ValueWrapperIS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IdEEEEEEEEEESA_EvT_SC_SC_SC_T0_PKc = comdat any

$_Z14test_heap_sortI14PointerWrapperI12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEEESB_EvT_SD_SD_SD_T0_PKc = comdat any

$_ZN9benchmark9quicksortIPddEEvT_S2_ = comdat any

$_ZN9benchmark9quicksortI14PointerWrapperIdEdEEvT_S3_ = comdat any

$_ZN9benchmark9quicksortIP12ValueWrapperIdES2_EEvT_S4_ = comdat any

$_ZN9benchmark9quicksortI14PointerWrapperI12ValueWrapperIdEES3_EEvT_S5_ = comdat any

$_ZN9benchmark9quicksortIP12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEESB_EEvT_SD_ = comdat any

$_ZN9benchmark9quicksortI14PointerWrapperI12ValueWrapperIS2_IS2_IS2_IS2_IS2_IS2_IS2_IS2_IS2_IdEEEEEEEEEEESC_EEvT_SE_ = comdat any

$_ZN9benchmark8heapsortIPddEEvT_S2_ = comdat any

$_ZN9benchmark8heapsortI14PointerWrapperIdEdEEvT_S3_ = comdat any

$_ZN9benchmark8heapsortIP12ValueWrapperIdES2_EEvT_S4_ = comdat any

$_ZN9benchmark8heapsortI14PointerWrapperI12ValueWrapperIdEES3_EEvT_S5_ = comdat any

$_ZN9benchmark8heapsortIP12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEESB_EEvT_SD_ = comdat any

$_ZN9benchmark8heapsortI14PointerWrapperI12ValueWrapperIS2_IS2_IS2_IS2_IS2_IS2_IS2_IS2_IS2_IdEEEEEEEEEEESC_EEvT_SE_ = comdat any

@results = dso_local local_unnamed_addr global ptr null, align 8
@current_test = dso_local local_unnamed_addr global i32 0, align 4
@allocated_results = dso_local local_unnamed_addr global i32 0, align 4
@.str = private unnamed_addr constant [31 x i8] c"Could not allocate %d results\0A\00", align 1
@.str.1 = private unnamed_addr constant [60 x i8] c"\0Atest %*s description   absolute   operations   ratio with\0A\00", align 1
@.str.2 = private unnamed_addr constant [2 x i8] c" \00", align 1
@.str.3 = private unnamed_addr constant [43 x i8] c"number %*s time       per second   test0\0A\0A\00", align 1
@.str.4 = private unnamed_addr constant [43 x i8] c"%2i %*s\22%s\22  %5.2f sec   %5.2f M     %.2f\0A\00", align 1
@.str.5 = private unnamed_addr constant [1 x i8] zeroinitializer, align 1
@.str.6 = private unnamed_addr constant [39 x i8] c"\0ATotal absolute time for %s: %.2f sec\0A\00", align 1
@.str.7 = private unnamed_addr constant [20 x i8] c"\0A%s Penalty: %.2f\0A\0A\00", align 1
@.str.8 = private unnamed_addr constant [34 x i8] c"\0Atest %*s description   absolute\0A\00", align 1
@.str.9 = private unnamed_addr constant [18 x i8] c"number %*s time\0A\0A\00", align 1
@.str.10 = private unnamed_addr constant [24 x i8] c"%2i %*s\22%s\22  %5.2f sec\0A\00", align 1
@start_time = dso_local local_unnamed_addr global i64 0, align 8
@end_time = dso_local local_unnamed_addr global i64 0, align 8
@iterations = dso_local local_unnamed_addr global i32 200000, align 4
@init_value = dso_local local_unnamed_addr global double 3.000000e+00, align 8
@data = dso_local global [2000 x double] zeroinitializer, align 8
@VData = dso_local global [2000 x %struct.ValueWrapper] zeroinitializer, align 8
@V10Data = dso_local global [2000 x %struct.ValueWrapper.0] zeroinitializer, align 8
@dataMaster = dso_local global [2000 x double] zeroinitializer, align 8
@VDataMaster = dso_local global [2000 x %struct.ValueWrapper] zeroinitializer, align 8
@V10DataMaster = dso_local global [2000 x %struct.ValueWrapper.0] zeroinitializer, align 8
@dpb = dso_local local_unnamed_addr global ptr @data, align 8
@dpe = dso_local local_unnamed_addr global ptr getelementptr inbounds nuw (i8, ptr @data, i64 16000), align 8
@dMpb = dso_local local_unnamed_addr global ptr @dataMaster, align 8
@dMpe = dso_local local_unnamed_addr global ptr getelementptr inbounds nuw (i8, ptr @dataMaster, i64 16000), align 8
@DVpb = dso_local local_unnamed_addr global ptr @VData, align 8
@DVpe = dso_local local_unnamed_addr global ptr getelementptr inbounds nuw (i8, ptr @VData, i64 16000), align 8
@DVMpb = dso_local local_unnamed_addr global ptr @VDataMaster, align 8
@DVMpe = dso_local local_unnamed_addr global ptr getelementptr inbounds nuw (i8, ptr @VDataMaster, i64 16000), align 8
@DV10pb = dso_local local_unnamed_addr global ptr @V10Data, align 8
@DV10pe = dso_local local_unnamed_addr global ptr getelementptr inbounds nuw (i8, ptr @V10Data, i64 16000), align 8
@DV10Mpb = dso_local local_unnamed_addr global ptr @V10DataMaster, align 8
@DV10Mpe = dso_local local_unnamed_addr global ptr getelementptr inbounds nuw (i8, ptr @V10DataMaster, i64 16000), align 8
@dPb = dso_local local_unnamed_addr global %struct.PointerWrapper { ptr @data }, align 8
@dPe = dso_local local_unnamed_addr global %struct.PointerWrapper { ptr getelementptr inbounds nuw (i8, ptr @data, i64 16000) }, align 8
@dMPb = dso_local local_unnamed_addr global %struct.PointerWrapper { ptr @dataMaster }, align 8
@dMPe = dso_local local_unnamed_addr global %struct.PointerWrapper { ptr getelementptr inbounds nuw (i8, ptr @dataMaster, i64 16000) }, align 8
@DVPb = dso_local local_unnamed_addr global %struct.PointerWrapper.9 { ptr @VData }, align 8
@DVPe = dso_local local_unnamed_addr global %struct.PointerWrapper.9 { ptr getelementptr inbounds nuw (i8, ptr @VData, i64 16000) }, align 8
@DVMPb = dso_local local_unnamed_addr global %struct.PointerWrapper.9 { ptr @VDataMaster }, align 8
@DVMPe = dso_local local_unnamed_addr global %struct.PointerWrapper.9 { ptr getelementptr inbounds nuw (i8, ptr @VDataMaster, i64 16000) }, align 8
@DV10Pb = dso_local local_unnamed_addr global %struct.PointerWrapper.10 { ptr @V10Data }, align 8
@DV10Pe = dso_local local_unnamed_addr global %struct.PointerWrapper.10 { ptr getelementptr inbounds nuw (i8, ptr @V10Data, i64 16000) }, align 8
@DV10MPb = dso_local local_unnamed_addr global %struct.PointerWrapper.10 { ptr @V10DataMaster }, align 8
@DV10MPe = dso_local local_unnamed_addr global %struct.PointerWrapper.10 { ptr getelementptr inbounds nuw (i8, ptr @V10DataMaster, i64 16000) }, align 8
@.str.32 = private unnamed_addr constant [30 x i8] c"insertion_sort double pointer\00", align 1
@.str.33 = private unnamed_addr constant [36 x i8] c"insertion_sort double pointer_class\00", align 1
@.str.34 = private unnamed_addr constant [42 x i8] c"insertion_sort DoubleValueWrapper pointer\00", align 1
@.str.35 = private unnamed_addr constant [48 x i8] c"insertion_sort DoubleValueWrapper pointer_class\00", align 1
@.str.36 = private unnamed_addr constant [44 x i8] c"insertion_sort DoubleValueWrapper10 pointer\00", align 1
@.str.37 = private unnamed_addr constant [50 x i8] c"insertion_sort DoubleValueWrapper10 pointer_class\00", align 1
@.str.38 = private unnamed_addr constant [25 x i8] c"quicksort double pointer\00", align 1
@.str.39 = private unnamed_addr constant [31 x i8] c"quicksort double pointer_class\00", align 1
@.str.40 = private unnamed_addr constant [37 x i8] c"quicksort DoubleValueWrapper pointer\00", align 1
@.str.41 = private unnamed_addr constant [43 x i8] c"quicksort DoubleValueWrapper pointer_class\00", align 1
@.str.42 = private unnamed_addr constant [39 x i8] c"quicksort DoubleValueWrapper10 pointer\00", align 1
@.str.43 = private unnamed_addr constant [45 x i8] c"quicksort DoubleValueWrapper10 pointer_class\00", align 1
@.str.44 = private unnamed_addr constant [25 x i8] c"heap_sort double pointer\00", align 1
@.str.45 = private unnamed_addr constant [31 x i8] c"heap_sort double pointer_class\00", align 1
@.str.46 = private unnamed_addr constant [37 x i8] c"heap_sort DoubleValueWrapper pointer\00", align 1
@.str.47 = private unnamed_addr constant [43 x i8] c"heap_sort DoubleValueWrapper pointer_class\00", align 1
@.str.48 = private unnamed_addr constant [39 x i8] c"heap_sort DoubleValueWrapper10 pointer\00", align 1
@.str.49 = private unnamed_addr constant [45 x i8] c"heap_sort DoubleValueWrapper10 pointer_class\00", align 1
@.str.50 = private unnamed_addr constant [16 x i8] c"test %i failed\0A\00", align 1
@.str.51 = private unnamed_addr constant [21 x i8] c"sort test %i failed\0A\00", align 1
@llvm.global_ctors = appending global [0 x { i32, ptr, ptr }] zeroinitializer

; Function Attrs: mustprogress nounwind uwtable
define dso_local void @_Z13record_resultdPKc(double noundef %0, ptr noundef %1) local_unnamed_addr #0 {
  %3 = load ptr, ptr @results, align 8, !tbaa !6
  %4 = icmp ne ptr %3, null
  %5 = load i32, ptr @allocated_results, align 4, !tbaa !11
  %6 = load i32, ptr @current_test, align 4
  %7 = icmp slt i32 %6, %5
  %8 = select i1 %4, i1 %7, i1 false
  br i1 %8, label %20, label %9

9:                                                ; preds = %2
  %10 = add nsw i32 %5, 10
  store i32 %10, ptr @allocated_results, align 4, !tbaa !11
  %11 = sext i32 %10 to i64
  %12 = shl nsw i64 %11, 4
  %13 = tail call ptr @realloc(ptr noundef %3, i64 noundef %12) #12
  store ptr %13, ptr @results, align 8, !tbaa !6
  %14 = icmp eq ptr %13, null
  br i1 %14, label %17, label %15

15:                                               ; preds = %9
  %16 = load i32, ptr @current_test, align 4, !tbaa !11
  br label %20

17:                                               ; preds = %9
  %18 = load i32, ptr @allocated_results, align 4, !tbaa !11
  %19 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %18)
  tail call void @exit(i32 noundef -1) #13
  unreachable

20:                                               ; preds = %2, %15
  %21 = phi i32 [ %16, %15 ], [ %6, %2 ]
  %22 = phi ptr [ %13, %15 ], [ %3, %2 ]
  %23 = sext i32 %21 to i64
  %24 = getelementptr inbounds %struct.one_result, ptr %22, i64 %23
  store double %0, ptr %24, align 8, !tbaa !13
  %25 = getelementptr inbounds %struct.one_result, ptr %22, i64 %23, i32 1
  store ptr %1, ptr %25, align 8, !tbaa !17
  %26 = add nsw i32 %21, 1
  store i32 %26, ptr @current_test, align 4, !tbaa !11
  ret void
}

; Function Attrs: mustprogress nounwind willreturn allockind("realloc") allocsize(1) memory(argmem: readwrite, inaccessiblemem: readwrite)
declare noalias noundef ptr @realloc(ptr allocptr noundef captures(none), i64 noundef) local_unnamed_addr #1

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #2

; Function Attrs: nofree noreturn nounwind
declare void @exit(i32 noundef) local_unnamed_addr #3

; Function Attrs: mustprogress nofree nounwind uwtable
define dso_local void @_Z9summarizePKciiii(ptr noundef %0, i32 noundef %1, i32 noundef %2, i32 noundef %3, i32 noundef %4) local_unnamed_addr #4 {
  %6 = sitofp i32 %1 to double
  %7 = sitofp i32 %2 to double
  %8 = fmul double %6, %7
  %9 = fdiv double %8, 1.000000e+06
  %10 = load i32, ptr @current_test, align 4, !tbaa !11
  %11 = icmp sgt i32 %10, 0
  br i1 %11, label %12, label %25

12:                                               ; preds = %5
  %13 = load ptr, ptr @results, align 8, !tbaa !6
  %14 = zext nneg i32 %10 to i64
  br label %15

15:                                               ; preds = %12, %15
  %16 = phi i64 [ 0, %12 ], [ %23, %15 ]
  %17 = phi i32 [ 12, %12 ], [ %22, %15 ]
  %18 = getelementptr inbounds nuw %struct.one_result, ptr %13, i64 %16, i32 1
  %19 = load ptr, ptr %18, align 8, !tbaa !17
  %20 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %19) #14
  %21 = trunc i64 %20 to i32
  %22 = tail call i32 @llvm.smax.i32(i32 %17, i32 %21)
  %23 = add nuw nsw i64 %16, 1
  %24 = icmp eq i64 %23, %14
  br i1 %24, label %25, label %15, !llvm.loop !18

25:                                               ; preds = %15, %5
  %26 = phi i32 [ 12, %5 ], [ %22, %15 ]
  %27 = add nsw i32 %26, -12
  %28 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %27, ptr noundef nonnull @.str.2)
  %29 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i32 noundef %26, ptr noundef nonnull @.str.2)
  %30 = load i32, ptr @current_test, align 4, !tbaa !11
  %31 = icmp sgt i32 %30, 0
  br i1 %31, label %57, label %84

32:                                               ; preds = %57
  %33 = icmp sgt i32 %73, 0
  br i1 %33, label %34, label %84

34:                                               ; preds = %32
  %35 = load ptr, ptr @results, align 8, !tbaa !6
  %36 = zext nneg i32 %73 to i64
  %37 = icmp eq i32 %73, 1
  br i1 %37, label %54, label %38

38:                                               ; preds = %34
  %39 = and i64 %36, 2147483646
  br label %40

40:                                               ; preds = %40, %38
  %41 = phi i64 [ 0, %38 ], [ %50, %40 ]
  %42 = phi double [ 0.000000e+00, %38 ], [ %49, %40 ]
  %43 = getelementptr inbounds nuw %struct.one_result, ptr %35, i64 %41
  %44 = getelementptr inbounds nuw %struct.one_result, ptr %35, i64 %41
  %45 = getelementptr inbounds nuw i8, ptr %44, i64 16
  %46 = load double, ptr %43, align 8, !tbaa !13
  %47 = load double, ptr %45, align 8, !tbaa !13
  %48 = fadd double %42, %46
  %49 = fadd double %48, %47
  %50 = add nuw i64 %41, 2
  %51 = icmp eq i64 %50, %39
  br i1 %51, label %52, label %40, !llvm.loop !20

52:                                               ; preds = %40
  %53 = icmp eq i64 %39, %36
  br i1 %53, label %84, label %54

54:                                               ; preds = %34, %52
  %55 = phi i64 [ 0, %34 ], [ %39, %52 ]
  %56 = phi double [ 0.000000e+00, %34 ], [ %49, %52 ]
  br label %76

57:                                               ; preds = %25, %57
  %58 = phi i64 [ %72, %57 ], [ 0, %25 ]
  %59 = load ptr, ptr @results, align 8, !tbaa !6
  %60 = getelementptr inbounds nuw %struct.one_result, ptr %59, i64 %58
  %61 = getelementptr inbounds nuw i8, ptr %60, i64 8
  %62 = load ptr, ptr %61, align 8, !tbaa !17
  %63 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %62) #14
  %64 = trunc i64 %63 to i32
  %65 = sub i32 %26, %64
  %66 = load double, ptr %60, align 8, !tbaa !13
  %67 = fdiv double %9, %66
  %68 = load double, ptr %59, align 8, !tbaa !13
  %69 = fdiv double %66, %68
  %70 = trunc nuw nsw i64 %58 to i32
  %71 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.4, i32 noundef %70, i32 noundef %65, ptr noundef nonnull @.str.5, ptr noundef nonnull %62, double noundef %66, double noundef %67, double noundef %69)
  %72 = add nuw nsw i64 %58, 1
  %73 = load i32, ptr @current_test, align 4, !tbaa !11
  %74 = sext i32 %73 to i64
  %75 = icmp slt i64 %72, %74
  br i1 %75, label %57, label %32, !llvm.loop !23

76:                                               ; preds = %54, %76
  %77 = phi i64 [ %82, %76 ], [ %55, %54 ]
  %78 = phi double [ %81, %76 ], [ %56, %54 ]
  %79 = getelementptr inbounds nuw %struct.one_result, ptr %35, i64 %77
  %80 = load double, ptr %79, align 8, !tbaa !13
  %81 = fadd double %78, %80
  %82 = add nuw nsw i64 %77, 1
  %83 = icmp eq i64 %82, %36
  br i1 %83, label %84, label %76, !llvm.loop !24

84:                                               ; preds = %76, %52, %25, %32
  %85 = phi double [ 0.000000e+00, %32 ], [ 0.000000e+00, %25 ], [ %49, %52 ], [ %81, %76 ]
  %86 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, ptr noundef %0, double noundef %85)
  %87 = load i32, ptr @current_test, align 4, !tbaa !11
  %88 = icmp sgt i32 %87, 1
  %89 = icmp ne i32 %4, 0
  %90 = and i1 %89, %88
  br i1 %90, label %91, label %112

91:                                               ; preds = %84
  %92 = load ptr, ptr @results, align 8, !tbaa !6
  %93 = load double, ptr %92, align 8, !tbaa !13
  br label %94

94:                                               ; preds = %91, %94
  %95 = phi i64 [ 1, %91 ], [ %102, %94 ]
  %96 = phi double [ 0.000000e+00, %91 ], [ %101, %94 ]
  %97 = getelementptr inbounds nuw %struct.one_result, ptr %92, i64 %95
  %98 = load double, ptr %97, align 8, !tbaa !13
  %99 = fdiv double %98, %93
  %100 = tail call double @log(double noundef %99) #15, !tbaa !11
  %101 = fadd double %96, %100
  %102 = add nuw nsw i64 %95, 1
  %103 = load i32, ptr @current_test, align 4, !tbaa !11
  %104 = sext i32 %103 to i64
  %105 = icmp slt i64 %102, %104
  br i1 %105, label %94, label %106, !llvm.loop !25

106:                                              ; preds = %94
  %107 = add nsw i32 %103, -1
  %108 = sitofp i32 %107 to double
  %109 = fdiv double %101, %108
  %110 = tail call double @exp(double noundef %109) #15, !tbaa !11
  %111 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.7, ptr noundef %0, double noundef %110)
  br label %112

112:                                              ; preds = %106, %84
  store i32 0, ptr @current_test, align 4, !tbaa !11
  ret void
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: read)
declare i64 @strlen(ptr noundef captures(none)) local_unnamed_addr #5

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(errnomem: write)
declare double @log(double noundef) local_unnamed_addr #6

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(errnomem: write)
declare double @exp(double noundef) local_unnamed_addr #6

; Function Attrs: mustprogress nofree nounwind uwtable
define dso_local void @_Z17summarize_simplefP8_IO_FILEPKc(ptr noundef captures(none) %0, ptr noundef %1) local_unnamed_addr #4 {
  %3 = load i32, ptr @current_test, align 4, !tbaa !11
  %4 = icmp sgt i32 %3, 0
  br i1 %4, label %5, label %18

5:                                                ; preds = %2
  %6 = load ptr, ptr @results, align 8, !tbaa !6
  %7 = zext nneg i32 %3 to i64
  br label %8

8:                                                ; preds = %5, %8
  %9 = phi i64 [ 0, %5 ], [ %16, %8 ]
  %10 = phi i32 [ 12, %5 ], [ %15, %8 ]
  %11 = getelementptr inbounds nuw %struct.one_result, ptr %6, i64 %9, i32 1
  %12 = load ptr, ptr %11, align 8, !tbaa !17
  %13 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %12) #14
  %14 = trunc i64 %13 to i32
  %15 = tail call i32 @llvm.smax.i32(i32 %10, i32 %14)
  %16 = add nuw nsw i64 %9, 1
  %17 = icmp eq i64 %16, %7
  br i1 %17, label %18, label %8, !llvm.loop !26

18:                                               ; preds = %8, %2
  %19 = phi i32 [ 12, %2 ], [ %15, %8 ]
  %20 = add nsw i32 %19, -12
  %21 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %0, ptr noundef nonnull @.str.8, i32 noundef %20, ptr noundef nonnull @.str.2) #15
  %22 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %0, ptr noundef nonnull @.str.9, i32 noundef %19, ptr noundef nonnull @.str.2) #15
  %23 = load i32, ptr @current_test, align 4, !tbaa !11
  %24 = icmp sgt i32 %23, 0
  br i1 %24, label %50, label %74

25:                                               ; preds = %50
  %26 = icmp sgt i32 %63, 0
  br i1 %26, label %27, label %74

27:                                               ; preds = %25
  %28 = load ptr, ptr @results, align 8, !tbaa !6
  %29 = zext nneg i32 %63 to i64
  %30 = icmp eq i32 %63, 1
  br i1 %30, label %47, label %31

31:                                               ; preds = %27
  %32 = and i64 %29, 2147483646
  br label %33

33:                                               ; preds = %33, %31
  %34 = phi i64 [ 0, %31 ], [ %43, %33 ]
  %35 = phi double [ 0.000000e+00, %31 ], [ %42, %33 ]
  %36 = getelementptr inbounds nuw %struct.one_result, ptr %28, i64 %34
  %37 = getelementptr inbounds nuw %struct.one_result, ptr %28, i64 %34
  %38 = getelementptr inbounds nuw i8, ptr %37, i64 16
  %39 = load double, ptr %36, align 8, !tbaa !13
  %40 = load double, ptr %38, align 8, !tbaa !13
  %41 = fadd double %35, %39
  %42 = fadd double %41, %40
  %43 = add nuw i64 %34, 2
  %44 = icmp eq i64 %43, %32
  br i1 %44, label %45, label %33, !llvm.loop !27

45:                                               ; preds = %33
  %46 = icmp eq i64 %32, %29
  br i1 %46, label %74, label %47

47:                                               ; preds = %27, %45
  %48 = phi i64 [ 0, %27 ], [ %32, %45 ]
  %49 = phi double [ 0.000000e+00, %27 ], [ %42, %45 ]
  br label %66

50:                                               ; preds = %18, %50
  %51 = phi i64 [ %62, %50 ], [ 0, %18 ]
  %52 = load ptr, ptr @results, align 8, !tbaa !6
  %53 = getelementptr inbounds nuw %struct.one_result, ptr %52, i64 %51
  %54 = getelementptr inbounds nuw i8, ptr %53, i64 8
  %55 = load ptr, ptr %54, align 8, !tbaa !17
  %56 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %55) #14
  %57 = trunc i64 %56 to i32
  %58 = sub i32 %19, %57
  %59 = load double, ptr %53, align 8, !tbaa !13
  %60 = trunc nuw nsw i64 %51 to i32
  %61 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %0, ptr noundef nonnull @.str.10, i32 noundef %60, i32 noundef %58, ptr noundef nonnull @.str.5, ptr noundef nonnull %55, double noundef %59) #15
  %62 = add nuw nsw i64 %51, 1
  %63 = load i32, ptr @current_test, align 4, !tbaa !11
  %64 = sext i32 %63 to i64
  %65 = icmp slt i64 %62, %64
  br i1 %65, label %50, label %25, !llvm.loop !28

66:                                               ; preds = %47, %66
  %67 = phi i64 [ %72, %66 ], [ %48, %47 ]
  %68 = phi double [ %71, %66 ], [ %49, %47 ]
  %69 = getelementptr inbounds nuw %struct.one_result, ptr %28, i64 %67
  %70 = load double, ptr %69, align 8, !tbaa !13
  %71 = fadd double %68, %70
  %72 = add nuw nsw i64 %67, 1
  %73 = icmp eq i64 %72, %29
  br i1 %73, label %74, label %66, !llvm.loop !29

74:                                               ; preds = %66, %45, %18, %25
  %75 = phi double [ 0.000000e+00, %25 ], [ 0.000000e+00, %18 ], [ %42, %45 ], [ %71, %66 ]
  %76 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %0, ptr noundef nonnull @.str.6, ptr noundef %1, double noundef %75) #15
  store i32 0, ptr @current_test, align 4, !tbaa !11
  ret void
}

; Function Attrs: nofree nounwind
declare noundef i32 @fprintf(ptr noundef captures(none), ptr noundef readonly captures(none), ...) local_unnamed_addr #2

; Function Attrs: mustprogress nounwind uwtable
define dso_local void @_Z11start_timerv() local_unnamed_addr #0 {
  %1 = tail call i64 @clock() #15
  store i64 %1, ptr @start_time, align 8, !tbaa !30
  ret void
}

; Function Attrs: nounwind
declare i64 @clock() local_unnamed_addr #7

; Function Attrs: mustprogress nounwind uwtable
define dso_local noundef double @_Z5timerv() local_unnamed_addr #0 {
  %1 = tail call i64 @clock() #15
  store i64 %1, ptr @end_time, align 8, !tbaa !30
  %2 = load i64, ptr @start_time, align 8, !tbaa !30
  %3 = sub nsw i64 %1, %2
  %4 = sitofp i64 %3 to double
  %5 = fdiv double %4, 1.000000e+06
  ret double %5
}

; Function Attrs: mustprogress norecurse uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #8 {
  %3 = icmp sgt i32 %0, 1
  br i1 %3, label %4, label %14

4:                                                ; preds = %2
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %6 = load ptr, ptr %5, align 8, !tbaa !32
  %7 = tail call i64 @__isoc23_strtol(ptr noundef nonnull %6, ptr noundef null, i32 noundef 10) #15
  %8 = trunc i64 %7 to i32
  store i32 %8, ptr @iterations, align 4, !tbaa !11
  %9 = icmp eq i32 %0, 2
  br i1 %9, label %14, label %10

10:                                               ; preds = %4
  %11 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %12 = load ptr, ptr %11, align 8, !tbaa !32
  %13 = tail call double @strtod(ptr noundef nonnull captures(none) %12, ptr noundef null) #15
  store double %13, ptr @init_value, align 8, !tbaa !33
  br label %14

14:                                               ; preds = %2, %10, %4
  %15 = load double, ptr @init_value, align 8, !tbaa !33
  %16 = fptosi double %15 to i32
  %17 = add nsw i32 %16, 123
  tail call void @srand(i32 noundef %17) #15
  %18 = load ptr, ptr @dpb, align 8, !tbaa !34
  %19 = load ptr, ptr @dpe, align 8, !tbaa !34
  %20 = load double, ptr @init_value, align 8, !tbaa !33
  %21 = icmp eq ptr %18, %19
  br i1 %21, label %53, label %22

22:                                               ; preds = %14
  %23 = ptrtoint ptr %19 to i64
  %24 = ptrtoint ptr %18 to i64
  %25 = add i64 %23, -8
  %26 = sub i64 %25, %24
  %27 = lshr i64 %26, 3
  %28 = add nuw nsw i64 %27, 1
  %29 = icmp ult i64 %26, 24
  br i1 %29, label %45, label %30

30:                                               ; preds = %22
  %31 = and i64 %28, 4611686018427387900
  %32 = shl i64 %31, 3
  %33 = getelementptr i8, ptr %18, i64 %32
  %34 = insertelement <2 x double> poison, double %20, i64 0
  %35 = shufflevector <2 x double> %34, <2 x double> poison, <2 x i32> zeroinitializer
  br label %36

36:                                               ; preds = %36, %30
  %37 = phi i64 [ 0, %30 ], [ %41, %36 ]
  %38 = shl i64 %37, 3
  %39 = getelementptr i8, ptr %18, i64 %38
  %40 = getelementptr i8, ptr %39, i64 16
  store <2 x double> %35, ptr %39, align 8, !tbaa !33
  store <2 x double> %35, ptr %40, align 8, !tbaa !33
  %41 = add nuw i64 %37, 4
  %42 = icmp eq i64 %41, %31
  br i1 %42, label %43, label %36, !llvm.loop !36

43:                                               ; preds = %36
  %44 = icmp eq i64 %28, %31
  br i1 %44, label %51, label %45

45:                                               ; preds = %22, %43
  %46 = phi ptr [ %18, %22 ], [ %33, %43 ]
  br label %47

47:                                               ; preds = %45, %47
  %48 = phi ptr [ %49, %47 ], [ %46, %45 ]
  %49 = getelementptr inbounds nuw i8, ptr %48, i64 8
  store double %20, ptr %48, align 8, !tbaa !33
  %50 = icmp eq ptr %49, %19
  br i1 %50, label %51, label %47, !llvm.loop !37

51:                                               ; preds = %47, %43
  %52 = load double, ptr @init_value, align 8, !tbaa !33
  br label %53

53:                                               ; preds = %51, %14
  %54 = phi double [ %52, %51 ], [ %20, %14 ]
  %55 = load ptr, ptr @DVpb, align 8, !tbaa !38
  %56 = load ptr, ptr @DVpe, align 8, !tbaa !38
  %57 = icmp eq ptr %55, %56
  br i1 %57, label %89, label %58

58:                                               ; preds = %53
  %59 = ptrtoint ptr %56 to i64
  %60 = ptrtoint ptr %55 to i64
  %61 = add i64 %59, -8
  %62 = sub i64 %61, %60
  %63 = lshr i64 %62, 3
  %64 = add nuw nsw i64 %63, 1
  %65 = icmp ult i64 %62, 24
  br i1 %65, label %81, label %66

66:                                               ; preds = %58
  %67 = and i64 %64, 4611686018427387900
  %68 = shl i64 %67, 3
  %69 = getelementptr i8, ptr %55, i64 %68
  %70 = insertelement <2 x double> poison, double %54, i64 0
  %71 = shufflevector <2 x double> %70, <2 x double> poison, <2 x i32> zeroinitializer
  br label %72

72:                                               ; preds = %72, %66
  %73 = phi i64 [ 0, %66 ], [ %77, %72 ]
  %74 = shl i64 %73, 3
  %75 = getelementptr i8, ptr %55, i64 %74
  %76 = getelementptr i8, ptr %75, i64 16
  store <2 x double> %71, ptr %75, align 8, !tbaa !33
  store <2 x double> %71, ptr %76, align 8, !tbaa !33
  %77 = add nuw i64 %73, 4
  %78 = icmp eq i64 %77, %67
  br i1 %78, label %79, label %72, !llvm.loop !40

79:                                               ; preds = %72
  %80 = icmp eq i64 %64, %67
  br i1 %80, label %87, label %81

81:                                               ; preds = %58, %79
  %82 = phi ptr [ %55, %58 ], [ %69, %79 ]
  br label %83

83:                                               ; preds = %81, %83
  %84 = phi ptr [ %85, %83 ], [ %82, %81 ]
  %85 = getelementptr inbounds nuw i8, ptr %84, i64 8
  store double %54, ptr %84, align 8, !tbaa !33
  %86 = icmp eq ptr %85, %56
  br i1 %86, label %87, label %83, !llvm.loop !41

87:                                               ; preds = %83, %79
  %88 = load double, ptr @init_value, align 8, !tbaa !33
  br label %89

89:                                               ; preds = %87, %53
  %90 = phi double [ %88, %87 ], [ %54, %53 ]
  %91 = load ptr, ptr @DV10pb, align 8, !tbaa !42
  %92 = load ptr, ptr @DV10pe, align 8, !tbaa !42
  %93 = icmp eq ptr %91, %92
  br i1 %93, label %123, label %94

94:                                               ; preds = %89
  %95 = ptrtoint ptr %92 to i64
  %96 = ptrtoint ptr %91 to i64
  %97 = add i64 %95, -8
  %98 = sub i64 %97, %96
  %99 = lshr i64 %98, 3
  %100 = add nuw nsw i64 %99, 1
  %101 = icmp ult i64 %98, 24
  br i1 %101, label %117, label %102

102:                                              ; preds = %94
  %103 = and i64 %100, 4611686018427387900
  %104 = shl i64 %103, 3
  %105 = getelementptr i8, ptr %91, i64 %104
  %106 = insertelement <2 x double> poison, double %90, i64 0
  %107 = shufflevector <2 x double> %106, <2 x double> poison, <2 x i32> zeroinitializer
  br label %108

108:                                              ; preds = %108, %102
  %109 = phi i64 [ 0, %102 ], [ %113, %108 ]
  %110 = shl i64 %109, 3
  %111 = getelementptr i8, ptr %91, i64 %110
  %112 = getelementptr i8, ptr %111, i64 16
  store <2 x double> %107, ptr %111, align 8, !tbaa !33
  store <2 x double> %107, ptr %112, align 8, !tbaa !33
  %113 = add nuw i64 %109, 4
  %114 = icmp eq i64 %113, %103
  br i1 %114, label %115, label %108, !llvm.loop !44

115:                                              ; preds = %108
  %116 = icmp eq i64 %100, %103
  br i1 %116, label %123, label %117

117:                                              ; preds = %94, %115
  %118 = phi ptr [ %91, %94 ], [ %105, %115 ]
  br label %119

119:                                              ; preds = %117, %119
  %120 = phi ptr [ %121, %119 ], [ %118, %117 ]
  %121 = getelementptr inbounds nuw i8, ptr %120, i64 8
  store double %90, ptr %120, align 8, !tbaa !33
  %122 = icmp eq ptr %121, %92
  br i1 %122, label %123, label %119, !llvm.loop !45

123:                                              ; preds = %119, %115, %89
  %124 = load i32, ptr @iterations, align 4, !tbaa !11
  %125 = icmp sgt i32 %124, 0
  br i1 %125, label %126, label %402

126:                                              ; preds = %123
  br i1 %21, label %127, label %145

127:                                              ; preds = %126
  %128 = load double, ptr @init_value, align 8, !tbaa !33
  br label %129

129:                                              ; preds = %140, %127
  %130 = phi i32 [ %141, %140 ], [ %124, %127 ]
  %131 = phi double [ %142, %140 ], [ %128, %127 ]
  %132 = phi i32 [ %143, %140 ], [ 0, %127 ]
  %133 = fmul double %131, 2.000000e+03
  %134 = fcmp une double %133, 0.000000e+00
  br i1 %134, label %135, label %140

135:                                              ; preds = %129
  %136 = load i32, ptr @current_test, align 4, !tbaa !11
  %137 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.50, i32 noundef %136)
  %138 = load double, ptr @init_value, align 8, !tbaa !33
  %139 = load i32, ptr @iterations, align 4, !tbaa !11
  br label %140

140:                                              ; preds = %135, %129
  %141 = phi i32 [ %139, %135 ], [ %130, %129 ]
  %142 = phi double [ %138, %135 ], [ %131, %129 ]
  %143 = add nuw nsw i32 %132, 1
  %144 = icmp slt i32 %143, %141
  br i1 %144, label %129, label %167, !llvm.loop !46

145:                                              ; preds = %126, %163
  %146 = phi i32 [ %164, %163 ], [ %124, %126 ]
  %147 = phi i32 [ %165, %163 ], [ 0, %126 ]
  br label %148

148:                                              ; preds = %148, %145
  %149 = phi double [ %153, %148 ], [ 0.000000e+00, %145 ]
  %150 = phi ptr [ %151, %148 ], [ %18, %145 ]
  %151 = getelementptr inbounds nuw i8, ptr %150, i64 8
  %152 = load double, ptr %150, align 8, !tbaa !33
  %153 = fadd double %149, %152
  %154 = icmp eq ptr %151, %19
  br i1 %154, label %155, label %148, !llvm.loop !47

155:                                              ; preds = %148
  %156 = load double, ptr @init_value, align 8, !tbaa !33
  %157 = fmul double %156, 2.000000e+03
  %158 = fcmp une double %153, %157
  br i1 %158, label %159, label %163

159:                                              ; preds = %155
  %160 = load i32, ptr @current_test, align 4, !tbaa !11
  %161 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.50, i32 noundef %160)
  %162 = load i32, ptr @iterations, align 4, !tbaa !11
  br label %163

163:                                              ; preds = %159, %155
  %164 = phi i32 [ %146, %155 ], [ %162, %159 ]
  %165 = add nuw nsw i32 %147, 1
  %166 = icmp slt i32 %165, %164
  br i1 %166, label %145, label %167, !llvm.loop !46

167:                                              ; preds = %163, %140
  %168 = phi i32 [ %141, %140 ], [ %164, %163 ]
  %169 = load ptr, ptr @dPb, align 8, !tbaa !34
  %170 = load ptr, ptr @dPe, align 8, !tbaa !34
  %171 = icmp sgt i32 %168, 0
  br i1 %171, label %172, label %402

172:                                              ; preds = %167
  %173 = icmp eq ptr %169, %170
  br i1 %173, label %174, label %192

174:                                              ; preds = %172
  %175 = load double, ptr @init_value, align 8, !tbaa !33
  br label %176

176:                                              ; preds = %187, %174
  %177 = phi i32 [ %188, %187 ], [ %168, %174 ]
  %178 = phi double [ %189, %187 ], [ %175, %174 ]
  %179 = phi i32 [ %190, %187 ], [ 0, %174 ]
  %180 = fmul double %178, 2.000000e+03
  %181 = fcmp une double %180, 0.000000e+00
  br i1 %181, label %182, label %187

182:                                              ; preds = %176
  %183 = load i32, ptr @current_test, align 4, !tbaa !11
  %184 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.50, i32 noundef %183)
  %185 = load double, ptr @init_value, align 8, !tbaa !33
  %186 = load i32, ptr @iterations, align 4, !tbaa !11
  br label %187

187:                                              ; preds = %182, %176
  %188 = phi i32 [ %186, %182 ], [ %177, %176 ]
  %189 = phi double [ %185, %182 ], [ %178, %176 ]
  %190 = add nuw nsw i32 %179, 1
  %191 = icmp slt i32 %190, %188
  br i1 %191, label %176, label %214, !llvm.loop !48

192:                                              ; preds = %172, %210
  %193 = phi i32 [ %211, %210 ], [ %168, %172 ]
  %194 = phi i32 [ %212, %210 ], [ 0, %172 ]
  br label %195

195:                                              ; preds = %195, %192
  %196 = phi double [ %200, %195 ], [ 0.000000e+00, %192 ]
  %197 = phi ptr [ %198, %195 ], [ %169, %192 ]
  %198 = getelementptr inbounds nuw i8, ptr %197, i64 8
  %199 = load double, ptr %197, align 8, !tbaa !33
  %200 = fadd double %196, %199
  %201 = icmp eq ptr %198, %170
  br i1 %201, label %202, label %195, !llvm.loop !49

202:                                              ; preds = %195
  %203 = load double, ptr @init_value, align 8, !tbaa !33
  %204 = fmul double %203, 2.000000e+03
  %205 = fcmp une double %200, %204
  br i1 %205, label %206, label %210

206:                                              ; preds = %202
  %207 = load i32, ptr @current_test, align 4, !tbaa !11
  %208 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.50, i32 noundef %207)
  %209 = load i32, ptr @iterations, align 4, !tbaa !11
  br label %210

210:                                              ; preds = %206, %202
  %211 = phi i32 [ %193, %202 ], [ %209, %206 ]
  %212 = add nuw nsw i32 %194, 1
  %213 = icmp slt i32 %212, %211
  br i1 %213, label %192, label %214, !llvm.loop !48

214:                                              ; preds = %210, %187
  %215 = phi i32 [ %188, %187 ], [ %211, %210 ]
  %216 = load ptr, ptr @DVpb, align 8, !tbaa !38
  %217 = load ptr, ptr @DVpe, align 8, !tbaa !38
  %218 = icmp sgt i32 %215, 0
  br i1 %218, label %219, label %402

219:                                              ; preds = %214
  %220 = icmp eq ptr %216, %217
  br i1 %220, label %221, label %239

221:                                              ; preds = %219
  %222 = load double, ptr @init_value, align 8, !tbaa !33
  br label %223

223:                                              ; preds = %234, %221
  %224 = phi i32 [ %235, %234 ], [ %215, %221 ]
  %225 = phi double [ %236, %234 ], [ %222, %221 ]
  %226 = phi i32 [ %237, %234 ], [ 0, %221 ]
  %227 = fmul double %225, 2.000000e+03
  %228 = fcmp une double %227, 0.000000e+00
  br i1 %228, label %229, label %234

229:                                              ; preds = %223
  %230 = load i32, ptr @current_test, align 4, !tbaa !11
  %231 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.50, i32 noundef %230)
  %232 = load double, ptr @init_value, align 8, !tbaa !33
  %233 = load i32, ptr @iterations, align 4, !tbaa !11
  br label %234

234:                                              ; preds = %229, %223
  %235 = phi i32 [ %233, %229 ], [ %224, %223 ]
  %236 = phi double [ %232, %229 ], [ %225, %223 ]
  %237 = add nuw nsw i32 %226, 1
  %238 = icmp slt i32 %237, %235
  br i1 %238, label %223, label %261, !llvm.loop !50

239:                                              ; preds = %219, %257
  %240 = phi i32 [ %258, %257 ], [ %215, %219 ]
  %241 = phi i32 [ %259, %257 ], [ 0, %219 ]
  br label %242

242:                                              ; preds = %242, %239
  %243 = phi ptr [ %245, %242 ], [ %216, %239 ]
  %244 = phi double [ %247, %242 ], [ 0.000000e+00, %239 ]
  %245 = getelementptr inbounds nuw i8, ptr %243, i64 8
  %246 = load double, ptr %243, align 8, !tbaa !51
  %247 = fadd double %244, %246
  %248 = icmp eq ptr %245, %217
  br i1 %248, label %249, label %242, !llvm.loop !53

249:                                              ; preds = %242
  %250 = load double, ptr @init_value, align 8, !tbaa !33
  %251 = fmul double %250, 2.000000e+03
  %252 = fcmp une double %247, %251
  br i1 %252, label %253, label %257

253:                                              ; preds = %249
  %254 = load i32, ptr @current_test, align 4, !tbaa !11
  %255 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.50, i32 noundef %254)
  %256 = load i32, ptr @iterations, align 4, !tbaa !11
  br label %257

257:                                              ; preds = %253, %249
  %258 = phi i32 [ %240, %249 ], [ %256, %253 ]
  %259 = add nuw nsw i32 %241, 1
  %260 = icmp slt i32 %259, %258
  br i1 %260, label %239, label %261, !llvm.loop !50

261:                                              ; preds = %257, %234
  %262 = phi i32 [ %235, %234 ], [ %258, %257 ]
  %263 = load ptr, ptr @DVPb, align 8, !tbaa !38
  %264 = load ptr, ptr @DVPe, align 8, !tbaa !38
  %265 = icmp sgt i32 %262, 0
  br i1 %265, label %266, label %402

266:                                              ; preds = %261
  %267 = icmp eq ptr %263, %264
  br i1 %267, label %268, label %286

268:                                              ; preds = %266
  %269 = load double, ptr @init_value, align 8, !tbaa !33
  br label %270

270:                                              ; preds = %281, %268
  %271 = phi i32 [ %282, %281 ], [ %262, %268 ]
  %272 = phi double [ %283, %281 ], [ %269, %268 ]
  %273 = phi i32 [ %284, %281 ], [ 0, %268 ]
  %274 = fmul double %272, 2.000000e+03
  %275 = fcmp une double %274, 0.000000e+00
  br i1 %275, label %276, label %281

276:                                              ; preds = %270
  %277 = load i32, ptr @current_test, align 4, !tbaa !11
  %278 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.50, i32 noundef %277)
  %279 = load double, ptr @init_value, align 8, !tbaa !33
  %280 = load i32, ptr @iterations, align 4, !tbaa !11
  br label %281

281:                                              ; preds = %276, %270
  %282 = phi i32 [ %280, %276 ], [ %271, %270 ]
  %283 = phi double [ %279, %276 ], [ %272, %270 ]
  %284 = add nuw nsw i32 %273, 1
  %285 = icmp slt i32 %284, %282
  br i1 %285, label %270, label %308, !llvm.loop !54

286:                                              ; preds = %266, %304
  %287 = phi i32 [ %305, %304 ], [ %262, %266 ]
  %288 = phi i32 [ %306, %304 ], [ 0, %266 ]
  br label %289

289:                                              ; preds = %289, %286
  %290 = phi ptr [ %292, %289 ], [ %263, %286 ]
  %291 = phi double [ %294, %289 ], [ 0.000000e+00, %286 ]
  %292 = getelementptr inbounds nuw i8, ptr %290, i64 8
  %293 = load double, ptr %290, align 8, !tbaa !51
  %294 = fadd double %291, %293
  %295 = icmp eq ptr %292, %264
  br i1 %295, label %296, label %289, !llvm.loop !55

296:                                              ; preds = %289
  %297 = load double, ptr @init_value, align 8, !tbaa !33
  %298 = fmul double %297, 2.000000e+03
  %299 = fcmp une double %294, %298
  br i1 %299, label %300, label %304

300:                                              ; preds = %296
  %301 = load i32, ptr @current_test, align 4, !tbaa !11
  %302 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.50, i32 noundef %301)
  %303 = load i32, ptr @iterations, align 4, !tbaa !11
  br label %304

304:                                              ; preds = %300, %296
  %305 = phi i32 [ %287, %296 ], [ %303, %300 ]
  %306 = add nuw nsw i32 %288, 1
  %307 = icmp slt i32 %306, %305
  br i1 %307, label %286, label %308, !llvm.loop !54

308:                                              ; preds = %304, %281
  %309 = phi i32 [ %282, %281 ], [ %305, %304 ]
  %310 = load ptr, ptr @DV10pb, align 8, !tbaa !42
  %311 = load ptr, ptr @DV10pe, align 8, !tbaa !42
  %312 = icmp sgt i32 %309, 0
  br i1 %312, label %313, label %402

313:                                              ; preds = %308
  %314 = icmp eq ptr %310, %311
  br i1 %314, label %315, label %333

315:                                              ; preds = %313
  %316 = load double, ptr @init_value, align 8, !tbaa !33
  br label %317

317:                                              ; preds = %328, %315
  %318 = phi i32 [ %329, %328 ], [ %309, %315 ]
  %319 = phi double [ %330, %328 ], [ %316, %315 ]
  %320 = phi i32 [ %331, %328 ], [ 0, %315 ]
  %321 = fmul double %319, 2.000000e+03
  %322 = fcmp une double %321, 0.000000e+00
  br i1 %322, label %323, label %328

323:                                              ; preds = %317
  %324 = load i32, ptr @current_test, align 4, !tbaa !11
  %325 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.50, i32 noundef %324)
  %326 = load double, ptr @init_value, align 8, !tbaa !33
  %327 = load i32, ptr @iterations, align 4, !tbaa !11
  br label %328

328:                                              ; preds = %323, %317
  %329 = phi i32 [ %327, %323 ], [ %318, %317 ]
  %330 = phi double [ %326, %323 ], [ %319, %317 ]
  %331 = add nuw nsw i32 %320, 1
  %332 = icmp slt i32 %331, %329
  br i1 %332, label %317, label %355, !llvm.loop !56

333:                                              ; preds = %313, %351
  %334 = phi i32 [ %352, %351 ], [ %309, %313 ]
  %335 = phi i32 [ %353, %351 ], [ 0, %313 ]
  br label %336

336:                                              ; preds = %336, %333
  %337 = phi ptr [ %339, %336 ], [ %310, %333 ]
  %338 = phi double [ %341, %336 ], [ 0.000000e+00, %333 ]
  %339 = getelementptr inbounds nuw i8, ptr %337, i64 8
  %340 = load double, ptr %337, align 8, !tbaa !51
  %341 = fadd double %338, %340
  %342 = icmp eq ptr %339, %311
  br i1 %342, label %343, label %336, !llvm.loop !57

343:                                              ; preds = %336
  %344 = load double, ptr @init_value, align 8, !tbaa !33
  %345 = fmul double %344, 2.000000e+03
  %346 = fcmp une double %341, %345
  br i1 %346, label %347, label %351

347:                                              ; preds = %343
  %348 = load i32, ptr @current_test, align 4, !tbaa !11
  %349 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.50, i32 noundef %348)
  %350 = load i32, ptr @iterations, align 4, !tbaa !11
  br label %351

351:                                              ; preds = %347, %343
  %352 = phi i32 [ %334, %343 ], [ %350, %347 ]
  %353 = add nuw nsw i32 %335, 1
  %354 = icmp slt i32 %353, %352
  br i1 %354, label %333, label %355, !llvm.loop !56

355:                                              ; preds = %351, %328
  %356 = phi i32 [ %329, %328 ], [ %352, %351 ]
  %357 = load ptr, ptr @DV10Pb, align 8, !tbaa !42
  %358 = load ptr, ptr @DV10Pe, align 8, !tbaa !42
  %359 = icmp sgt i32 %356, 0
  br i1 %359, label %360, label %402

360:                                              ; preds = %355
  %361 = icmp eq ptr %357, %358
  br i1 %361, label %362, label %380

362:                                              ; preds = %360
  %363 = load double, ptr @init_value, align 8, !tbaa !33
  br label %364

364:                                              ; preds = %375, %362
  %365 = phi i32 [ %376, %375 ], [ %356, %362 ]
  %366 = phi double [ %377, %375 ], [ %363, %362 ]
  %367 = phi i32 [ %378, %375 ], [ 0, %362 ]
  %368 = fmul double %366, 2.000000e+03
  %369 = fcmp une double %368, 0.000000e+00
  br i1 %369, label %370, label %375

370:                                              ; preds = %364
  %371 = load i32, ptr @current_test, align 4, !tbaa !11
  %372 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.50, i32 noundef %371)
  %373 = load double, ptr @init_value, align 8, !tbaa !33
  %374 = load i32, ptr @iterations, align 4, !tbaa !11
  br label %375

375:                                              ; preds = %370, %364
  %376 = phi i32 [ %374, %370 ], [ %365, %364 ]
  %377 = phi double [ %373, %370 ], [ %366, %364 ]
  %378 = add nuw nsw i32 %367, 1
  %379 = icmp slt i32 %378, %376
  br i1 %379, label %364, label %402, !llvm.loop !58

380:                                              ; preds = %360, %398
  %381 = phi i32 [ %399, %398 ], [ %356, %360 ]
  %382 = phi i32 [ %400, %398 ], [ 0, %360 ]
  br label %383

383:                                              ; preds = %383, %380
  %384 = phi ptr [ %386, %383 ], [ %357, %380 ]
  %385 = phi double [ %388, %383 ], [ 0.000000e+00, %380 ]
  %386 = getelementptr inbounds nuw i8, ptr %384, i64 8
  %387 = load double, ptr %384, align 8, !tbaa !51
  %388 = fadd double %385, %387
  %389 = icmp eq ptr %386, %358
  br i1 %389, label %390, label %383, !llvm.loop !59

390:                                              ; preds = %383
  %391 = load double, ptr @init_value, align 8, !tbaa !33
  %392 = fmul double %391, 2.000000e+03
  %393 = fcmp une double %388, %392
  br i1 %393, label %394, label %398

394:                                              ; preds = %390
  %395 = load i32, ptr @current_test, align 4, !tbaa !11
  %396 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.50, i32 noundef %395)
  %397 = load i32, ptr @iterations, align 4, !tbaa !11
  br label %398

398:                                              ; preds = %394, %390
  %399 = phi i32 [ %381, %390 ], [ %397, %394 ]
  %400 = add nuw nsw i32 %382, 1
  %401 = icmp slt i32 %400, %399
  br i1 %401, label %380, label %402, !llvm.loop !58

402:                                              ; preds = %398, %375, %123, %167, %261, %214, %308, %355
  %403 = phi i32 [ %124, %123 ], [ %168, %167 ], [ %262, %261 ], [ %215, %214 ], [ %309, %308 ], [ %356, %355 ], [ %376, %375 ], [ %399, %398 ]
  %404 = sdiv i32 %403, 2000
  store i32 %404, ptr @iterations, align 4, !tbaa !11
  %405 = load ptr, ptr @dMpb, align 8, !tbaa !34
  %406 = load ptr, ptr @dMpe, align 8, !tbaa !34
  %407 = icmp eq ptr %405, %406
  br i1 %407, label %417, label %408

408:                                              ; preds = %402, %408
  %409 = phi ptr [ %412, %408 ], [ %405, %402 ]
  %410 = tail call i32 @rand() #15
  %411 = sitofp i32 %410 to double
  %412 = getelementptr inbounds nuw i8, ptr %409, i64 8
  store double %411, ptr %409, align 8, !tbaa !33
  %413 = icmp eq ptr %412, %406
  br i1 %413, label %414, label %408, !llvm.loop !60

414:                                              ; preds = %408
  %415 = load ptr, ptr @dMpb, align 8, !tbaa !34
  %416 = load ptr, ptr @dMpe, align 8, !tbaa !34
  br label %417

417:                                              ; preds = %414, %402
  %418 = phi ptr [ %416, %414 ], [ %406, %402 ]
  %419 = phi ptr [ %415, %414 ], [ %405, %402 ]
  %420 = ptrtoint ptr %418 to i64
  %421 = ptrtoint ptr %419 to i64
  %422 = icmp eq ptr %419, %418
  br i1 %422, label %507, label %423

423:                                              ; preds = %417
  %424 = ptrtoint ptr %419 to i64
  %425 = ptrtoint ptr %418 to i64
  %426 = load ptr, ptr @DVMpb, align 8, !tbaa !38
  %427 = add i64 %425, -8
  %428 = sub i64 %427, %424
  %429 = lshr i64 %428, 3
  %430 = add nuw nsw i64 %429, 1
  %431 = icmp ult i64 %428, 24
  %432 = ptrtoint ptr %426 to i64
  %433 = sub i64 %432, %424
  %434 = icmp ult i64 %433, 32
  %435 = select i1 %431, i1 true, i1 %434
  br i1 %435, label %456, label %436

436:                                              ; preds = %423
  %437 = and i64 %430, 4611686018427387900
  %438 = shl i64 %437, 3
  %439 = getelementptr i8, ptr %426, i64 %438
  %440 = shl i64 %437, 3
  %441 = getelementptr i8, ptr %419, i64 %440
  br label %442

442:                                              ; preds = %442, %436
  %443 = phi i64 [ 0, %436 ], [ %452, %442 ]
  %444 = shl i64 %443, 3
  %445 = getelementptr i8, ptr %426, i64 %444
  %446 = shl i64 %443, 3
  %447 = getelementptr i8, ptr %419, i64 %446
  %448 = getelementptr i8, ptr %447, i64 16
  %449 = load <2 x i64>, ptr %447, align 8, !tbaa !33
  %450 = load <2 x i64>, ptr %448, align 8, !tbaa !33
  %451 = getelementptr i8, ptr %445, i64 16
  store <2 x i64> %449, ptr %445, align 8, !tbaa !33
  store <2 x i64> %450, ptr %451, align 8, !tbaa !33
  %452 = add nuw i64 %443, 4
  %453 = icmp eq i64 %452, %437
  br i1 %453, label %454, label %442, !llvm.loop !61

454:                                              ; preds = %442
  %455 = icmp eq i64 %430, %437
  br i1 %455, label %466, label %456

456:                                              ; preds = %423, %454
  %457 = phi ptr [ %426, %423 ], [ %439, %454 ]
  %458 = phi ptr [ %419, %423 ], [ %441, %454 ]
  br label %459

459:                                              ; preds = %456, %459
  %460 = phi ptr [ %464, %459 ], [ %457, %456 ]
  %461 = phi ptr [ %462, %459 ], [ %458, %456 ]
  %462 = getelementptr inbounds nuw i8, ptr %461, i64 8
  %463 = load i64, ptr %461, align 8, !tbaa !33
  %464 = getelementptr inbounds nuw i8, ptr %460, i64 8
  store i64 %463, ptr %460, align 8, !tbaa !33
  %465 = icmp eq ptr %462, %418
  br i1 %465, label %466, label %459, !llvm.loop !62

466:                                              ; preds = %459, %454
  %467 = load ptr, ptr @DV10Mpb, align 8, !tbaa !42
  %468 = add i64 %420, -8
  %469 = sub i64 %468, %421
  %470 = lshr i64 %469, 3
  %471 = add nuw nsw i64 %470, 1
  %472 = icmp ult i64 %469, 24
  %473 = ptrtoint ptr %467 to i64
  %474 = sub i64 %473, %421
  %475 = icmp ult i64 %474, 32
  %476 = select i1 %472, i1 true, i1 %475
  br i1 %476, label %497, label %477

477:                                              ; preds = %466
  %478 = and i64 %471, 4611686018427387900
  %479 = shl i64 %478, 3
  %480 = getelementptr i8, ptr %467, i64 %479
  %481 = shl i64 %478, 3
  %482 = getelementptr i8, ptr %419, i64 %481
  br label %483

483:                                              ; preds = %483, %477
  %484 = phi i64 [ 0, %477 ], [ %493, %483 ]
  %485 = shl i64 %484, 3
  %486 = getelementptr i8, ptr %467, i64 %485
  %487 = shl i64 %484, 3
  %488 = getelementptr i8, ptr %419, i64 %487
  %489 = getelementptr i8, ptr %488, i64 16
  %490 = load <2 x i64>, ptr %488, align 8, !tbaa !33
  %491 = load <2 x i64>, ptr %489, align 8, !tbaa !33
  %492 = getelementptr i8, ptr %486, i64 16
  store <2 x i64> %490, ptr %486, align 8, !tbaa !33
  store <2 x i64> %491, ptr %492, align 8, !tbaa !33
  %493 = add nuw i64 %484, 4
  %494 = icmp eq i64 %493, %478
  br i1 %494, label %495, label %483, !llvm.loop !63

495:                                              ; preds = %483
  %496 = icmp eq i64 %471, %478
  br i1 %496, label %507, label %497

497:                                              ; preds = %466, %495
  %498 = phi ptr [ %467, %466 ], [ %480, %495 ]
  %499 = phi ptr [ %419, %466 ], [ %482, %495 ]
  br label %500

500:                                              ; preds = %497, %500
  %501 = phi ptr [ %505, %500 ], [ %498, %497 ]
  %502 = phi ptr [ %503, %500 ], [ %499, %497 ]
  %503 = getelementptr inbounds nuw i8, ptr %502, i64 8
  %504 = load i64, ptr %502, align 8, !tbaa !33
  %505 = getelementptr inbounds nuw i8, ptr %501, i64 8
  store i64 %504, ptr %501, align 8, !tbaa !33
  %506 = icmp eq ptr %503, %418
  br i1 %506, label %507, label %500, !llvm.loop !64

507:                                              ; preds = %500, %495, %417
  %508 = load ptr, ptr @dpb, align 8, !tbaa !34
  %509 = load ptr, ptr @dpe, align 8, !tbaa !34
  tail call void @_Z19test_insertion_sortIPddEvT_S1_S1_S1_T0_PKc(ptr noundef %419, ptr noundef %418, ptr noundef %508, ptr noundef %509, double noundef 0.000000e+00, ptr noundef nonnull @.str.32)
  %510 = load ptr, ptr @dMPb, align 8, !tbaa !34
  %511 = load ptr, ptr @dMPe, align 8, !tbaa !34
  %512 = load ptr, ptr @dPb, align 8, !tbaa !34
  %513 = load ptr, ptr @dPe, align 8, !tbaa !34
  tail call void @_Z19test_insertion_sortI14PointerWrapperIdEdEvT_S2_S2_S2_T0_PKc(ptr %510, ptr %511, ptr %512, ptr %513, double noundef 0.000000e+00, ptr noundef nonnull @.str.33)
  %514 = load ptr, ptr @DVMpb, align 8, !tbaa !38
  %515 = load ptr, ptr @DVMpe, align 8, !tbaa !38
  %516 = load ptr, ptr @DVpb, align 8, !tbaa !38
  %517 = load ptr, ptr @DVpe, align 8, !tbaa !38
  tail call void @_Z19test_insertion_sortIP12ValueWrapperIdES1_EvT_S3_S3_S3_T0_PKc(ptr noundef %514, ptr noundef %515, ptr noundef %516, ptr noundef %517, [1 x double] alignstack(8) zeroinitializer, ptr noundef nonnull @.str.34)
  %518 = load ptr, ptr @DVMPb, align 8, !tbaa !38
  %519 = load ptr, ptr @DVMPe, align 8, !tbaa !38
  %520 = load ptr, ptr @DVPb, align 8, !tbaa !38
  %521 = load ptr, ptr @DVPe, align 8, !tbaa !38
  tail call void @_Z19test_insertion_sortI14PointerWrapperI12ValueWrapperIdEES2_EvT_S4_S4_S4_T0_PKc(ptr %518, ptr %519, ptr %520, ptr %521, [1 x double] alignstack(8) zeroinitializer, ptr noundef nonnull @.str.35)
  %522 = load ptr, ptr @DV10Mpb, align 8, !tbaa !42
  %523 = load ptr, ptr @DV10Mpe, align 8, !tbaa !42
  %524 = load ptr, ptr @DV10pb, align 8, !tbaa !42
  %525 = load ptr, ptr @DV10pe, align 8, !tbaa !42
  tail call void @_Z19test_insertion_sortIP12ValueWrapperIS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IdEEEEEEEEEESA_EvT_SC_SC_SC_T0_PKc(ptr noundef %522, ptr noundef %523, ptr noundef %524, ptr noundef %525, [1 x double] alignstack(8) zeroinitializer, ptr noundef nonnull @.str.36)
  %526 = load ptr, ptr @DV10MPb, align 8, !tbaa !42
  %527 = load ptr, ptr @DV10MPe, align 8, !tbaa !42
  %528 = load ptr, ptr @DV10Pb, align 8, !tbaa !42
  %529 = load ptr, ptr @DV10Pe, align 8, !tbaa !42
  tail call void @_Z19test_insertion_sortI14PointerWrapperI12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEEESB_EvT_SD_SD_SD_T0_PKc(ptr %526, ptr %527, ptr %528, ptr %529, [1 x double] alignstack(8) zeroinitializer, ptr noundef nonnull @.str.37)
  %530 = load i32, ptr @iterations, align 4, !tbaa !11
  %531 = shl nsw i32 %530, 3
  store i32 %531, ptr @iterations, align 4, !tbaa !11
  %532 = load ptr, ptr @dMpb, align 8, !tbaa !34
  %533 = load ptr, ptr @dMpe, align 8, !tbaa !34
  %534 = load ptr, ptr @dpb, align 8, !tbaa !34
  %535 = load ptr, ptr @dpe, align 8, !tbaa !34
  tail call void @_Z14test_quicksortIPddEvT_S1_S1_S1_T0_PKc(ptr noundef %532, ptr noundef %533, ptr noundef %534, ptr noundef %535, double noundef 0.000000e+00, ptr noundef nonnull @.str.38)
  %536 = load ptr, ptr @dMPb, align 8, !tbaa !34
  %537 = load ptr, ptr @dMPe, align 8, !tbaa !34
  %538 = load ptr, ptr @dPb, align 8, !tbaa !34
  %539 = load ptr, ptr @dPe, align 8, !tbaa !34
  tail call void @_Z14test_quicksortI14PointerWrapperIdEdEvT_S2_S2_S2_T0_PKc(ptr %536, ptr %537, ptr %538, ptr %539, double noundef 0.000000e+00, ptr noundef nonnull @.str.39)
  %540 = load ptr, ptr @DVMpb, align 8, !tbaa !38
  %541 = load ptr, ptr @DVMpe, align 8, !tbaa !38
  %542 = load ptr, ptr @DVpb, align 8, !tbaa !38
  %543 = load ptr, ptr @DVpe, align 8, !tbaa !38
  tail call void @_Z14test_quicksortIP12ValueWrapperIdES1_EvT_S3_S3_S3_T0_PKc(ptr noundef %540, ptr noundef %541, ptr noundef %542, ptr noundef %543, [1 x double] alignstack(8) zeroinitializer, ptr noundef nonnull @.str.40)
  %544 = load ptr, ptr @DVMPb, align 8, !tbaa !38
  %545 = load ptr, ptr @DVMPe, align 8, !tbaa !38
  %546 = load ptr, ptr @DVPb, align 8, !tbaa !38
  %547 = load ptr, ptr @DVPe, align 8, !tbaa !38
  tail call void @_Z14test_quicksortI14PointerWrapperI12ValueWrapperIdEES2_EvT_S4_S4_S4_T0_PKc(ptr %544, ptr %545, ptr %546, ptr %547, [1 x double] alignstack(8) zeroinitializer, ptr noundef nonnull @.str.41)
  %548 = load ptr, ptr @DV10Mpb, align 8, !tbaa !42
  %549 = load ptr, ptr @DV10Mpe, align 8, !tbaa !42
  %550 = load ptr, ptr @DV10pb, align 8, !tbaa !42
  %551 = load ptr, ptr @DV10pe, align 8, !tbaa !42
  tail call void @_Z14test_quicksortIP12ValueWrapperIS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IdEEEEEEEEEESA_EvT_SC_SC_SC_T0_PKc(ptr noundef %548, ptr noundef %549, ptr noundef %550, ptr noundef %551, [1 x double] alignstack(8) zeroinitializer, ptr noundef nonnull @.str.42)
  %552 = load ptr, ptr @DV10MPb, align 8, !tbaa !42
  %553 = load ptr, ptr @DV10MPe, align 8, !tbaa !42
  %554 = load ptr, ptr @DV10Pb, align 8, !tbaa !42
  %555 = load ptr, ptr @DV10Pe, align 8, !tbaa !42
  tail call void @_Z14test_quicksortI14PointerWrapperI12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEEESB_EvT_SD_SD_SD_T0_PKc(ptr %552, ptr %553, ptr %554, ptr %555, [1 x double] alignstack(8) zeroinitializer, ptr noundef nonnull @.str.43)
  %556 = load ptr, ptr @dMpb, align 8, !tbaa !34
  %557 = load ptr, ptr @dMpe, align 8, !tbaa !34
  %558 = load ptr, ptr @dpb, align 8, !tbaa !34
  %559 = load ptr, ptr @dpe, align 8, !tbaa !34
  tail call void @_Z14test_heap_sortIPddEvT_S1_S1_S1_T0_PKc(ptr noundef %556, ptr noundef %557, ptr noundef %558, ptr noundef %559, double noundef 0.000000e+00, ptr noundef nonnull @.str.44)
  %560 = load ptr, ptr @dMPb, align 8, !tbaa !34
  %561 = load ptr, ptr @dMPe, align 8, !tbaa !34
  %562 = load ptr, ptr @dPb, align 8, !tbaa !34
  %563 = load ptr, ptr @dPe, align 8, !tbaa !34
  tail call void @_Z14test_heap_sortI14PointerWrapperIdEdEvT_S2_S2_S2_T0_PKc(ptr %560, ptr %561, ptr %562, ptr %563, double noundef 0.000000e+00, ptr noundef nonnull @.str.45)
  %564 = load ptr, ptr @DVMpb, align 8, !tbaa !38
  %565 = load ptr, ptr @DVMpe, align 8, !tbaa !38
  %566 = load ptr, ptr @DVpb, align 8, !tbaa !38
  %567 = load ptr, ptr @DVpe, align 8, !tbaa !38
  tail call void @_Z14test_heap_sortIP12ValueWrapperIdES1_EvT_S3_S3_S3_T0_PKc(ptr noundef %564, ptr noundef %565, ptr noundef %566, ptr noundef %567, [1 x double] alignstack(8) zeroinitializer, ptr noundef nonnull @.str.46)
  %568 = load ptr, ptr @DVMPb, align 8, !tbaa !38
  %569 = load ptr, ptr @DVMPe, align 8, !tbaa !38
  %570 = load ptr, ptr @DVPb, align 8, !tbaa !38
  %571 = load ptr, ptr @DVPe, align 8, !tbaa !38
  tail call void @_Z14test_heap_sortI14PointerWrapperI12ValueWrapperIdEES2_EvT_S4_S4_S4_T0_PKc(ptr %568, ptr %569, ptr %570, ptr %571, [1 x double] alignstack(8) zeroinitializer, ptr noundef nonnull @.str.47)
  %572 = load ptr, ptr @DV10Mpb, align 8, !tbaa !42
  %573 = load ptr, ptr @DV10Mpe, align 8, !tbaa !42
  %574 = load ptr, ptr @DV10pb, align 8, !tbaa !42
  %575 = load ptr, ptr @DV10pe, align 8, !tbaa !42
  tail call void @_Z14test_heap_sortIP12ValueWrapperIS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IdEEEEEEEEEESA_EvT_SC_SC_SC_T0_PKc(ptr noundef %572, ptr noundef %573, ptr noundef %574, ptr noundef %575, [1 x double] alignstack(8) zeroinitializer, ptr noundef nonnull @.str.48)
  %576 = load ptr, ptr @DV10MPb, align 8, !tbaa !42
  %577 = load ptr, ptr @DV10MPe, align 8, !tbaa !42
  %578 = load ptr, ptr @DV10Pb, align 8, !tbaa !42
  %579 = load ptr, ptr @DV10Pe, align 8, !tbaa !42
  tail call void @_Z14test_heap_sortI14PointerWrapperI12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEEESB_EvT_SD_SD_SD_T0_PKc(ptr %576, ptr %577, ptr %578, ptr %579, [1 x double] alignstack(8) zeroinitializer, ptr noundef nonnull @.str.49)
  ret i32 0
}

; Function Attrs: nounwind
declare void @srand(i32 noundef) local_unnamed_addr #7

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_Z19test_insertion_sortIPddEvT_S1_S1_S1_T0_PKc(ptr noundef %0, ptr noundef %1, ptr noundef %2, ptr noundef %3, double noundef %4, ptr noundef %5) local_unnamed_addr #9 comdat {
  %7 = ptrtoint ptr %0 to i64
  %8 = ptrtoint ptr %1 to i64
  %9 = ptrtoint ptr %0 to i64
  %10 = ptrtoint ptr %2 to i64
  %11 = ptrtoint ptr %0 to i64
  %12 = ptrtoint ptr %1 to i64
  %13 = ptrtoint ptr %0 to i64
  %14 = ptrtoint ptr %2 to i64
  %15 = load i32, ptr @iterations, align 4, !tbaa !11
  %16 = icmp sgt i32 %15, 0
  br i1 %16, label %17, label %205

17:                                               ; preds = %6
  %18 = icmp eq ptr %0, %1
  %19 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %20 = icmp eq ptr %19, %3
  br i1 %20, label %21, label %99

21:                                               ; preds = %17
  br i1 %18, label %37, label %22

22:                                               ; preds = %21
  %23 = sub i64 %10, %9
  %24 = add i64 %8, -8
  %25 = sub i64 %24, %7
  %26 = lshr i64 %25, 3
  %27 = add nuw nsw i64 %26, 1
  %28 = icmp ult i64 %25, 24
  %29 = icmp ult i64 %23, 32
  %30 = or i1 %28, %29
  %31 = and i64 %27, 4611686018427387900
  %32 = shl i64 %31, 3
  %33 = getelementptr i8, ptr %2, i64 %32
  %34 = shl i64 %31, 3
  %35 = getelementptr i8, ptr %0, i64 %34
  %36 = icmp eq i64 %27, %31
  br label %56

37:                                               ; preds = %21, %52
  %38 = phi i32 [ %53, %52 ], [ %15, %21 ]
  %39 = phi i32 [ %54, %52 ], [ 0, %21 ]
  br label %40

40:                                               ; preds = %44, %37
  %41 = phi ptr [ %2, %37 ], [ %42, %44 ]
  %42 = getelementptr i8, ptr %41, i64 8
  %43 = icmp eq ptr %42, %3
  br i1 %43, label %52, label %44

44:                                               ; preds = %40
  %45 = load double, ptr %42, align 8, !tbaa !33
  %46 = load double, ptr %41, align 8, !tbaa !33
  %47 = fcmp olt double %45, %46
  br i1 %47, label %48, label %40, !llvm.loop !65

48:                                               ; preds = %44
  %49 = load i32, ptr @current_test, align 4, !tbaa !11
  %50 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.51, i32 noundef %49)
  %51 = load i32, ptr @iterations, align 4, !tbaa !11
  br label %52

52:                                               ; preds = %40, %48
  %53 = phi i32 [ %51, %48 ], [ %38, %40 ]
  %54 = add nuw nsw i32 %39, 1
  %55 = icmp slt i32 %54, %53
  br i1 %55, label %37, label %205, !llvm.loop !66

56:                                               ; preds = %22, %95
  %57 = phi i32 [ %96, %95 ], [ %15, %22 ]
  %58 = phi i32 [ %97, %95 ], [ 0, %22 ]
  br i1 %30, label %72, label %59

59:                                               ; preds = %56, %59
  %60 = phi i64 [ %69, %59 ], [ 0, %56 ]
  %61 = shl i64 %60, 3
  %62 = getelementptr i8, ptr %2, i64 %61
  %63 = shl i64 %60, 3
  %64 = getelementptr i8, ptr %0, i64 %63
  %65 = getelementptr i8, ptr %64, i64 16
  %66 = load <2 x double>, ptr %64, align 8, !tbaa !33
  %67 = load <2 x double>, ptr %65, align 8, !tbaa !33
  %68 = getelementptr i8, ptr %62, i64 16
  store <2 x double> %66, ptr %62, align 8, !tbaa !33
  store <2 x double> %67, ptr %68, align 8, !tbaa !33
  %69 = add nuw i64 %60, 4
  %70 = icmp eq i64 %69, %31
  br i1 %70, label %71, label %59, !llvm.loop !67

71:                                               ; preds = %59
  br i1 %36, label %82, label %72

72:                                               ; preds = %56, %71
  %73 = phi ptr [ %2, %56 ], [ %33, %71 ]
  %74 = phi ptr [ %0, %56 ], [ %35, %71 ]
  br label %75

75:                                               ; preds = %72, %75
  %76 = phi ptr [ %80, %75 ], [ %73, %72 ]
  %77 = phi ptr [ %78, %75 ], [ %74, %72 ]
  %78 = getelementptr inbounds nuw i8, ptr %77, i64 8
  %79 = load double, ptr %77, align 8, !tbaa !33
  %80 = getelementptr inbounds nuw i8, ptr %76, i64 8
  store double %79, ptr %76, align 8, !tbaa !33
  %81 = icmp eq ptr %78, %1
  br i1 %81, label %82, label %75, !llvm.loop !68

82:                                               ; preds = %75, %71
  br label %83

83:                                               ; preds = %82, %87
  %84 = phi ptr [ %85, %87 ], [ %2, %82 ]
  %85 = getelementptr i8, ptr %84, i64 8
  %86 = icmp eq ptr %85, %3
  br i1 %86, label %95, label %87

87:                                               ; preds = %83
  %88 = load double, ptr %85, align 8, !tbaa !33
  %89 = load double, ptr %84, align 8, !tbaa !33
  %90 = fcmp olt double %88, %89
  br i1 %90, label %91, label %83, !llvm.loop !65

91:                                               ; preds = %87
  %92 = load i32, ptr @current_test, align 4, !tbaa !11
  %93 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.51, i32 noundef %92)
  %94 = load i32, ptr @iterations, align 4, !tbaa !11
  br label %95

95:                                               ; preds = %83, %91
  %96 = phi i32 [ %94, %91 ], [ %57, %83 ]
  %97 = add nuw nsw i32 %58, 1
  %98 = icmp slt i32 %97, %96
  br i1 %98, label %56, label %205, !llvm.loop !66

99:                                               ; preds = %17
  br i1 %18, label %115, label %100

100:                                              ; preds = %99
  %101 = sub i64 %14, %13
  %102 = add i64 %12, -8
  %103 = sub i64 %102, %11
  %104 = lshr i64 %103, 3
  %105 = add nuw nsw i64 %104, 1
  %106 = icmp ult i64 %103, 24
  %107 = icmp ult i64 %101, 32
  %108 = or i1 %106, %107
  %109 = and i64 %105, 4611686018427387900
  %110 = shl i64 %109, 3
  %111 = getelementptr i8, ptr %2, i64 %110
  %112 = shl i64 %109, 3
  %113 = getelementptr i8, ptr %0, i64 %112
  %114 = icmp eq i64 %105, %109
  br label %148

115:                                              ; preds = %99, %144
  %116 = phi i32 [ %145, %144 ], [ %15, %99 ]
  %117 = phi i32 [ %146, %144 ], [ 0, %99 ]
  br label %118

118:                                              ; preds = %128, %115
  %119 = phi ptr [ %130, %128 ], [ %19, %115 ]
  %120 = load double, ptr %119, align 8, !tbaa !33
  br label %121

121:                                              ; preds = %126, %118
  %122 = phi ptr [ %119, %118 ], [ %123, %126 ]
  %123 = getelementptr i8, ptr %122, i64 -8
  %124 = load double, ptr %123, align 8, !tbaa !33
  %125 = fcmp olt double %120, %124
  br i1 %125, label %126, label %128

126:                                              ; preds = %121
  store double %124, ptr %122, align 8, !tbaa !33
  %127 = icmp eq ptr %123, %2
  br i1 %127, label %128, label %121, !llvm.loop !69

128:                                              ; preds = %126, %121
  %129 = phi ptr [ %2, %126 ], [ %122, %121 ]
  store double %120, ptr %129, align 8, !tbaa !33
  %130 = getelementptr inbounds nuw i8, ptr %119, i64 8
  %131 = icmp eq ptr %130, %3
  br i1 %131, label %132, label %118, !llvm.loop !70

132:                                              ; preds = %128, %136
  %133 = phi ptr [ %134, %136 ], [ %2, %128 ]
  %134 = getelementptr i8, ptr %133, i64 8
  %135 = icmp eq ptr %134, %3
  br i1 %135, label %144, label %136

136:                                              ; preds = %132
  %137 = load double, ptr %134, align 8, !tbaa !33
  %138 = load double, ptr %133, align 8, !tbaa !33
  %139 = fcmp olt double %137, %138
  br i1 %139, label %140, label %132, !llvm.loop !65

140:                                              ; preds = %136
  %141 = load i32, ptr @current_test, align 4, !tbaa !11
  %142 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.51, i32 noundef %141)
  %143 = load i32, ptr @iterations, align 4, !tbaa !11
  br label %144

144:                                              ; preds = %132, %140
  %145 = phi i32 [ %143, %140 ], [ %116, %132 ]
  %146 = add nuw nsw i32 %117, 1
  %147 = icmp slt i32 %146, %145
  br i1 %147, label %115, label %205, !llvm.loop !66

148:                                              ; preds = %100, %201
  %149 = phi i32 [ %202, %201 ], [ %15, %100 ]
  %150 = phi i32 [ %203, %201 ], [ 0, %100 ]
  br i1 %108, label %164, label %151

151:                                              ; preds = %148, %151
  %152 = phi i64 [ %161, %151 ], [ 0, %148 ]
  %153 = shl i64 %152, 3
  %154 = getelementptr i8, ptr %2, i64 %153
  %155 = shl i64 %152, 3
  %156 = getelementptr i8, ptr %0, i64 %155
  %157 = getelementptr i8, ptr %156, i64 16
  %158 = load <2 x double>, ptr %156, align 8, !tbaa !33
  %159 = load <2 x double>, ptr %157, align 8, !tbaa !33
  %160 = getelementptr i8, ptr %154, i64 16
  store <2 x double> %158, ptr %154, align 8, !tbaa !33
  store <2 x double> %159, ptr %160, align 8, !tbaa !33
  %161 = add nuw i64 %152, 4
  %162 = icmp eq i64 %161, %109
  br i1 %162, label %163, label %151, !llvm.loop !71

163:                                              ; preds = %151
  br i1 %114, label %174, label %164

164:                                              ; preds = %148, %163
  %165 = phi ptr [ %2, %148 ], [ %111, %163 ]
  %166 = phi ptr [ %0, %148 ], [ %113, %163 ]
  br label %167

167:                                              ; preds = %164, %167
  %168 = phi ptr [ %172, %167 ], [ %165, %164 ]
  %169 = phi ptr [ %170, %167 ], [ %166, %164 ]
  %170 = getelementptr inbounds nuw i8, ptr %169, i64 8
  %171 = load double, ptr %169, align 8, !tbaa !33
  %172 = getelementptr inbounds nuw i8, ptr %168, i64 8
  store double %171, ptr %168, align 8, !tbaa !33
  %173 = icmp eq ptr %170, %1
  br i1 %173, label %174, label %167, !llvm.loop !72

174:                                              ; preds = %167, %163
  br label %175

175:                                              ; preds = %174, %185
  %176 = phi ptr [ %187, %185 ], [ %19, %174 ]
  %177 = load double, ptr %176, align 8, !tbaa !33
  br label %178

178:                                              ; preds = %183, %175
  %179 = phi ptr [ %176, %175 ], [ %180, %183 ]
  %180 = getelementptr i8, ptr %179, i64 -8
  %181 = load double, ptr %180, align 8, !tbaa !33
  %182 = fcmp olt double %177, %181
  br i1 %182, label %183, label %185

183:                                              ; preds = %178
  store double %181, ptr %179, align 8, !tbaa !33
  %184 = icmp eq ptr %180, %2
  br i1 %184, label %185, label %178, !llvm.loop !69

185:                                              ; preds = %183, %178
  %186 = phi ptr [ %2, %183 ], [ %179, %178 ]
  store double %177, ptr %186, align 8, !tbaa !33
  %187 = getelementptr inbounds nuw i8, ptr %176, i64 8
  %188 = icmp eq ptr %187, %3
  br i1 %188, label %189, label %175, !llvm.loop !70

189:                                              ; preds = %185, %193
  %190 = phi ptr [ %191, %193 ], [ %2, %185 ]
  %191 = getelementptr i8, ptr %190, i64 8
  %192 = icmp eq ptr %191, %3
  br i1 %192, label %201, label %193

193:                                              ; preds = %189
  %194 = load double, ptr %191, align 8, !tbaa !33
  %195 = load double, ptr %190, align 8, !tbaa !33
  %196 = fcmp olt double %194, %195
  br i1 %196, label %197, label %189, !llvm.loop !65

197:                                              ; preds = %193
  %198 = load i32, ptr @current_test, align 4, !tbaa !11
  %199 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.51, i32 noundef %198)
  %200 = load i32, ptr @iterations, align 4, !tbaa !11
  br label %201

201:                                              ; preds = %189, %197
  %202 = phi i32 [ %200, %197 ], [ %149, %189 ]
  %203 = add nuw nsw i32 %150, 1
  %204 = icmp slt i32 %203, %202
  br i1 %204, label %148, label %205, !llvm.loop !66

205:                                              ; preds = %201, %144, %95, %52, %6
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_Z19test_insertion_sortI14PointerWrapperIdEdEvT_S2_S2_S2_T0_PKc(ptr %0, ptr %1, ptr %2, ptr %3, double noundef %4, ptr noundef %5) local_unnamed_addr #9 comdat {
  %7 = ptrtoint ptr %0 to i64
  %8 = ptrtoint ptr %1 to i64
  %9 = ptrtoint ptr %0 to i64
  %10 = ptrtoint ptr %2 to i64
  %11 = ptrtoint ptr %0 to i64
  %12 = ptrtoint ptr %1 to i64
  %13 = ptrtoint ptr %0 to i64
  %14 = ptrtoint ptr %2 to i64
  %15 = load i32, ptr @iterations, align 4, !tbaa !11
  %16 = icmp sgt i32 %15, 0
  br i1 %16, label %17, label %205

17:                                               ; preds = %6
  %18 = icmp eq ptr %0, %1
  %19 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %20 = icmp eq ptr %19, %3
  br i1 %20, label %21, label %99

21:                                               ; preds = %17
  br i1 %18, label %37, label %22

22:                                               ; preds = %21
  %23 = sub i64 %10, %9
  %24 = add i64 %8, -8
  %25 = sub i64 %24, %7
  %26 = lshr i64 %25, 3
  %27 = add nuw nsw i64 %26, 1
  %28 = icmp ult i64 %25, 24
  %29 = icmp ult i64 %23, 32
  %30 = select i1 %28, i1 true, i1 %29
  %31 = and i64 %27, 4611686018427387900
  %32 = shl i64 %31, 3
  %33 = getelementptr i8, ptr %0, i64 %32
  %34 = shl i64 %31, 3
  %35 = getelementptr i8, ptr %2, i64 %34
  %36 = icmp eq i64 %27, %31
  br label %56

37:                                               ; preds = %21, %52
  %38 = phi i32 [ %53, %52 ], [ %15, %21 ]
  %39 = phi i32 [ %54, %52 ], [ 0, %21 ]
  br label %40

40:                                               ; preds = %44, %37
  %41 = phi ptr [ %2, %37 ], [ %42, %44 ]
  %42 = getelementptr i8, ptr %41, i64 8
  %43 = icmp eq ptr %42, %3
  br i1 %43, label %52, label %44

44:                                               ; preds = %40
  %45 = load double, ptr %42, align 8, !tbaa !33
  %46 = load double, ptr %41, align 8, !tbaa !33
  %47 = fcmp olt double %45, %46
  br i1 %47, label %48, label %40, !llvm.loop !73

48:                                               ; preds = %44
  %49 = load i32, ptr @current_test, align 4, !tbaa !11
  %50 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.51, i32 noundef %49)
  %51 = load i32, ptr @iterations, align 4, !tbaa !11
  br label %52

52:                                               ; preds = %40, %48
  %53 = phi i32 [ %51, %48 ], [ %38, %40 ]
  %54 = add nuw nsw i32 %39, 1
  %55 = icmp slt i32 %54, %53
  br i1 %55, label %37, label %205, !llvm.loop !74

56:                                               ; preds = %22, %95
  %57 = phi i32 [ %96, %95 ], [ %15, %22 ]
  %58 = phi i32 [ %97, %95 ], [ 0, %22 ]
  br i1 %30, label %72, label %59

59:                                               ; preds = %56, %59
  %60 = phi i64 [ %69, %59 ], [ 0, %56 ]
  %61 = shl i64 %60, 3
  %62 = getelementptr i8, ptr %0, i64 %61
  %63 = shl i64 %60, 3
  %64 = getelementptr i8, ptr %2, i64 %63
  %65 = getelementptr i8, ptr %62, i64 16
  %66 = load <2 x double>, ptr %62, align 8, !tbaa !33
  %67 = load <2 x double>, ptr %65, align 8, !tbaa !33
  %68 = getelementptr i8, ptr %64, i64 16
  store <2 x double> %66, ptr %64, align 8, !tbaa !33
  store <2 x double> %67, ptr %68, align 8, !tbaa !33
  %69 = add nuw i64 %60, 4
  %70 = icmp eq i64 %69, %31
  br i1 %70, label %71, label %59, !llvm.loop !75

71:                                               ; preds = %59
  br i1 %36, label %82, label %72

72:                                               ; preds = %56, %71
  %73 = phi ptr [ %0, %56 ], [ %33, %71 ]
  %74 = phi ptr [ %2, %56 ], [ %35, %71 ]
  br label %75

75:                                               ; preds = %72, %75
  %76 = phi ptr [ %78, %75 ], [ %73, %72 ]
  %77 = phi ptr [ %80, %75 ], [ %74, %72 ]
  %78 = getelementptr inbounds nuw i8, ptr %76, i64 8
  %79 = load double, ptr %76, align 8, !tbaa !33
  %80 = getelementptr inbounds nuw i8, ptr %77, i64 8
  store double %79, ptr %77, align 8, !tbaa !33
  %81 = icmp eq ptr %78, %1
  br i1 %81, label %82, label %75, !llvm.loop !76

82:                                               ; preds = %75, %71
  br label %83

83:                                               ; preds = %82, %87
  %84 = phi ptr [ %85, %87 ], [ %2, %82 ]
  %85 = getelementptr i8, ptr %84, i64 8
  %86 = icmp eq ptr %85, %3
  br i1 %86, label %95, label %87

87:                                               ; preds = %83
  %88 = load double, ptr %85, align 8, !tbaa !33
  %89 = load double, ptr %84, align 8, !tbaa !33
  %90 = fcmp olt double %88, %89
  br i1 %90, label %91, label %83, !llvm.loop !73

91:                                               ; preds = %87
  %92 = load i32, ptr @current_test, align 4, !tbaa !11
  %93 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.51, i32 noundef %92)
  %94 = load i32, ptr @iterations, align 4, !tbaa !11
  br label %95

95:                                               ; preds = %83, %91
  %96 = phi i32 [ %94, %91 ], [ %57, %83 ]
  %97 = add nuw nsw i32 %58, 1
  %98 = icmp slt i32 %97, %96
  br i1 %98, label %56, label %205, !llvm.loop !74

99:                                               ; preds = %17
  br i1 %18, label %115, label %100

100:                                              ; preds = %99
  %101 = sub i64 %14, %13
  %102 = add i64 %12, -8
  %103 = sub i64 %102, %11
  %104 = lshr i64 %103, 3
  %105 = add nuw nsw i64 %104, 1
  %106 = icmp ult i64 %103, 24
  %107 = icmp ult i64 %101, 32
  %108 = select i1 %106, i1 true, i1 %107
  %109 = and i64 %105, 4611686018427387900
  %110 = shl i64 %109, 3
  %111 = getelementptr i8, ptr %0, i64 %110
  %112 = shl i64 %109, 3
  %113 = getelementptr i8, ptr %2, i64 %112
  %114 = icmp eq i64 %105, %109
  br label %148

115:                                              ; preds = %99, %144
  %116 = phi i32 [ %145, %144 ], [ %15, %99 ]
  %117 = phi i32 [ %146, %144 ], [ 0, %99 ]
  br label %118

118:                                              ; preds = %128, %115
  %119 = phi ptr [ %130, %128 ], [ %19, %115 ]
  %120 = load double, ptr %119, align 8, !tbaa !33
  br label %121

121:                                              ; preds = %126, %118
  %122 = phi ptr [ %119, %118 ], [ %123, %126 ]
  %123 = getelementptr i8, ptr %122, i64 -8
  %124 = load double, ptr %123, align 8, !tbaa !33
  %125 = fcmp olt double %120, %124
  br i1 %125, label %126, label %128

126:                                              ; preds = %121
  store double %124, ptr %122, align 8, !tbaa !33
  %127 = icmp eq ptr %123, %2
  br i1 %127, label %128, label %121, !llvm.loop !77

128:                                              ; preds = %126, %121
  %129 = phi ptr [ %2, %126 ], [ %122, %121 ]
  store double %120, ptr %129, align 8, !tbaa !33
  %130 = getelementptr inbounds nuw i8, ptr %119, i64 8
  %131 = icmp eq ptr %130, %3
  br i1 %131, label %132, label %118, !llvm.loop !78

132:                                              ; preds = %128, %136
  %133 = phi ptr [ %134, %136 ], [ %2, %128 ]
  %134 = getelementptr i8, ptr %133, i64 8
  %135 = icmp eq ptr %134, %3
  br i1 %135, label %144, label %136

136:                                              ; preds = %132
  %137 = load double, ptr %134, align 8, !tbaa !33
  %138 = load double, ptr %133, align 8, !tbaa !33
  %139 = fcmp olt double %137, %138
  br i1 %139, label %140, label %132, !llvm.loop !73

140:                                              ; preds = %136
  %141 = load i32, ptr @current_test, align 4, !tbaa !11
  %142 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.51, i32 noundef %141)
  %143 = load i32, ptr @iterations, align 4, !tbaa !11
  br label %144

144:                                              ; preds = %132, %140
  %145 = phi i32 [ %143, %140 ], [ %116, %132 ]
  %146 = add nuw nsw i32 %117, 1
  %147 = icmp slt i32 %146, %145
  br i1 %147, label %115, label %205, !llvm.loop !74

148:                                              ; preds = %100, %201
  %149 = phi i32 [ %202, %201 ], [ %15, %100 ]
  %150 = phi i32 [ %203, %201 ], [ 0, %100 ]
  br i1 %108, label %164, label %151

151:                                              ; preds = %148, %151
  %152 = phi i64 [ %161, %151 ], [ 0, %148 ]
  %153 = shl i64 %152, 3
  %154 = getelementptr i8, ptr %0, i64 %153
  %155 = shl i64 %152, 3
  %156 = getelementptr i8, ptr %2, i64 %155
  %157 = getelementptr i8, ptr %154, i64 16
  %158 = load <2 x double>, ptr %154, align 8, !tbaa !33
  %159 = load <2 x double>, ptr %157, align 8, !tbaa !33
  %160 = getelementptr i8, ptr %156, i64 16
  store <2 x double> %158, ptr %156, align 8, !tbaa !33
  store <2 x double> %159, ptr %160, align 8, !tbaa !33
  %161 = add nuw i64 %152, 4
  %162 = icmp eq i64 %161, %109
  br i1 %162, label %163, label %151, !llvm.loop !79

163:                                              ; preds = %151
  br i1 %114, label %174, label %164

164:                                              ; preds = %148, %163
  %165 = phi ptr [ %0, %148 ], [ %111, %163 ]
  %166 = phi ptr [ %2, %148 ], [ %113, %163 ]
  br label %167

167:                                              ; preds = %164, %167
  %168 = phi ptr [ %170, %167 ], [ %165, %164 ]
  %169 = phi ptr [ %172, %167 ], [ %166, %164 ]
  %170 = getelementptr inbounds nuw i8, ptr %168, i64 8
  %171 = load double, ptr %168, align 8, !tbaa !33
  %172 = getelementptr inbounds nuw i8, ptr %169, i64 8
  store double %171, ptr %169, align 8, !tbaa !33
  %173 = icmp eq ptr %170, %1
  br i1 %173, label %174, label %167, !llvm.loop !80

174:                                              ; preds = %167, %163
  br label %175

175:                                              ; preds = %174, %185
  %176 = phi ptr [ %187, %185 ], [ %19, %174 ]
  %177 = load double, ptr %176, align 8, !tbaa !33
  br label %178

178:                                              ; preds = %183, %175
  %179 = phi ptr [ %176, %175 ], [ %180, %183 ]
  %180 = getelementptr i8, ptr %179, i64 -8
  %181 = load double, ptr %180, align 8, !tbaa !33
  %182 = fcmp olt double %177, %181
  br i1 %182, label %183, label %185

183:                                              ; preds = %178
  store double %181, ptr %179, align 8, !tbaa !33
  %184 = icmp eq ptr %180, %2
  br i1 %184, label %185, label %178, !llvm.loop !77

185:                                              ; preds = %183, %178
  %186 = phi ptr [ %2, %183 ], [ %179, %178 ]
  store double %177, ptr %186, align 8, !tbaa !33
  %187 = getelementptr inbounds nuw i8, ptr %176, i64 8
  %188 = icmp eq ptr %187, %3
  br i1 %188, label %189, label %175, !llvm.loop !78

189:                                              ; preds = %185, %193
  %190 = phi ptr [ %191, %193 ], [ %2, %185 ]
  %191 = getelementptr i8, ptr %190, i64 8
  %192 = icmp eq ptr %191, %3
  br i1 %192, label %201, label %193

193:                                              ; preds = %189
  %194 = load double, ptr %191, align 8, !tbaa !33
  %195 = load double, ptr %190, align 8, !tbaa !33
  %196 = fcmp olt double %194, %195
  br i1 %196, label %197, label %189, !llvm.loop !73

197:                                              ; preds = %193
  %198 = load i32, ptr @current_test, align 4, !tbaa !11
  %199 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.51, i32 noundef %198)
  %200 = load i32, ptr @iterations, align 4, !tbaa !11
  br label %201

201:                                              ; preds = %189, %197
  %202 = phi i32 [ %200, %197 ], [ %149, %189 ]
  %203 = add nuw nsw i32 %150, 1
  %204 = icmp slt i32 %203, %202
  br i1 %204, label %148, label %205, !llvm.loop !74

205:                                              ; preds = %201, %144, %95, %52, %6
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_Z19test_insertion_sortIP12ValueWrapperIdES1_EvT_S3_S3_S3_T0_PKc(ptr noundef %0, ptr noundef %1, ptr noundef %2, ptr noundef %3, [1 x double] alignstack(8) %4, ptr noundef %5) local_unnamed_addr #9 comdat {
  %7 = ptrtoint ptr %0 to i64
  %8 = ptrtoint ptr %1 to i64
  %9 = ptrtoint ptr %0 to i64
  %10 = ptrtoint ptr %2 to i64
  %11 = ptrtoint ptr %0 to i64
  %12 = ptrtoint ptr %1 to i64
  %13 = ptrtoint ptr %0 to i64
  %14 = ptrtoint ptr %2 to i64
  %15 = load i32, ptr @iterations, align 4, !tbaa !11
  %16 = icmp sgt i32 %15, 0
  br i1 %16, label %17, label %207

17:                                               ; preds = %6
  %18 = icmp eq ptr %0, %1
  %19 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %20 = icmp eq ptr %19, %3
  br i1 %20, label %21, label %99

21:                                               ; preds = %17
  br i1 %18, label %37, label %22

22:                                               ; preds = %21
  %23 = sub i64 %10, %9
  %24 = add i64 %8, -8
  %25 = sub i64 %24, %7
  %26 = lshr i64 %25, 3
  %27 = add nuw nsw i64 %26, 1
  %28 = icmp ult i64 %25, 24
  %29 = icmp ult i64 %23, 32
  %30 = or i1 %28, %29
  %31 = and i64 %27, 4611686018427387900
  %32 = shl i64 %31, 3
  %33 = getelementptr i8, ptr %2, i64 %32
  %34 = shl i64 %31, 3
  %35 = getelementptr i8, ptr %0, i64 %34
  %36 = icmp eq i64 %27, %31
  br label %56

37:                                               ; preds = %21, %52
  %38 = phi i32 [ %53, %52 ], [ %15, %21 ]
  %39 = phi i32 [ %54, %52 ], [ 0, %21 ]
  br label %40

40:                                               ; preds = %44, %37
  %41 = phi ptr [ %2, %37 ], [ %42, %44 ]
  %42 = getelementptr i8, ptr %41, i64 8
  %43 = icmp eq ptr %42, %3
  br i1 %43, label %52, label %44

44:                                               ; preds = %40
  %45 = load double, ptr %42, align 8, !tbaa !51
  %46 = load double, ptr %41, align 8, !tbaa !51
  %47 = fcmp olt double %45, %46
  br i1 %47, label %48, label %40, !llvm.loop !81

48:                                               ; preds = %44
  %49 = load i32, ptr @current_test, align 4, !tbaa !11
  %50 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.51, i32 noundef %49)
  %51 = load i32, ptr @iterations, align 4, !tbaa !11
  br label %52

52:                                               ; preds = %40, %48
  %53 = phi i32 [ %51, %48 ], [ %38, %40 ]
  %54 = add nuw nsw i32 %39, 1
  %55 = icmp slt i32 %54, %53
  br i1 %55, label %37, label %207, !llvm.loop !82

56:                                               ; preds = %22, %95
  %57 = phi i32 [ %96, %95 ], [ %15, %22 ]
  %58 = phi i32 [ %97, %95 ], [ 0, %22 ]
  br i1 %30, label %72, label %59

59:                                               ; preds = %56, %59
  %60 = phi i64 [ %69, %59 ], [ 0, %56 ]
  %61 = shl i64 %60, 3
  %62 = getelementptr i8, ptr %2, i64 %61
  %63 = shl i64 %60, 3
  %64 = getelementptr i8, ptr %0, i64 %63
  %65 = getelementptr i8, ptr %64, i64 16
  %66 = load <2 x i64>, ptr %64, align 8, !tbaa !33
  %67 = load <2 x i64>, ptr %65, align 8, !tbaa !33
  %68 = getelementptr i8, ptr %62, i64 16
  store <2 x i64> %66, ptr %62, align 8, !tbaa !33
  store <2 x i64> %67, ptr %68, align 8, !tbaa !33
  %69 = add nuw i64 %60, 4
  %70 = icmp eq i64 %69, %31
  br i1 %70, label %71, label %59, !llvm.loop !83

71:                                               ; preds = %59
  br i1 %36, label %82, label %72

72:                                               ; preds = %56, %71
  %73 = phi ptr [ %2, %56 ], [ %33, %71 ]
  %74 = phi ptr [ %0, %56 ], [ %35, %71 ]
  br label %75

75:                                               ; preds = %72, %75
  %76 = phi ptr [ %79, %75 ], [ %73, %72 ]
  %77 = phi ptr [ %78, %75 ], [ %74, %72 ]
  %78 = getelementptr inbounds nuw i8, ptr %77, i64 8
  %79 = getelementptr inbounds nuw i8, ptr %76, i64 8
  %80 = load i64, ptr %77, align 8, !tbaa !33
  store i64 %80, ptr %76, align 8, !tbaa !33
  %81 = icmp eq ptr %78, %1
  br i1 %81, label %82, label %75, !llvm.loop !84

82:                                               ; preds = %75, %71
  br label %83

83:                                               ; preds = %82, %87
  %84 = phi ptr [ %85, %87 ], [ %2, %82 ]
  %85 = getelementptr i8, ptr %84, i64 8
  %86 = icmp eq ptr %85, %3
  br i1 %86, label %95, label %87

87:                                               ; preds = %83
  %88 = load double, ptr %85, align 8, !tbaa !51
  %89 = load double, ptr %84, align 8, !tbaa !51
  %90 = fcmp olt double %88, %89
  br i1 %90, label %91, label %83, !llvm.loop !81

91:                                               ; preds = %87
  %92 = load i32, ptr @current_test, align 4, !tbaa !11
  %93 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.51, i32 noundef %92)
  %94 = load i32, ptr @iterations, align 4, !tbaa !11
  br label %95

95:                                               ; preds = %83, %91
  %96 = phi i32 [ %94, %91 ], [ %57, %83 ]
  %97 = add nuw nsw i32 %58, 1
  %98 = icmp slt i32 %97, %96
  br i1 %98, label %56, label %207, !llvm.loop !82

99:                                               ; preds = %17
  br i1 %18, label %115, label %100

100:                                              ; preds = %99
  %101 = sub i64 %14, %13
  %102 = add i64 %12, -8
  %103 = sub i64 %102, %11
  %104 = lshr i64 %103, 3
  %105 = add nuw nsw i64 %104, 1
  %106 = icmp ult i64 %103, 24
  %107 = icmp ult i64 %101, 32
  %108 = or i1 %106, %107
  %109 = and i64 %105, 4611686018427387900
  %110 = shl i64 %109, 3
  %111 = getelementptr i8, ptr %2, i64 %110
  %112 = shl i64 %109, 3
  %113 = getelementptr i8, ptr %0, i64 %112
  %114 = icmp eq i64 %105, %109
  br label %149

115:                                              ; preds = %99, %145
  %116 = phi i32 [ %146, %145 ], [ %15, %99 ]
  %117 = phi i32 [ %147, %145 ], [ 0, %99 ]
  br label %118

118:                                              ; preds = %129, %115
  %119 = phi ptr [ %131, %129 ], [ %19, %115 ]
  %120 = load i64, ptr %119, align 8, !tbaa !33
  %121 = bitcast i64 %120 to double
  br label %122

122:                                              ; preds = %127, %118
  %123 = phi ptr [ %119, %118 ], [ %124, %127 ]
  %124 = getelementptr i8, ptr %123, i64 -8
  %125 = load double, ptr %124, align 8
  %126 = fcmp ogt double %125, %121
  br i1 %126, label %127, label %129

127:                                              ; preds = %122
  store double %125, ptr %123, align 8, !tbaa !33
  %128 = icmp eq ptr %124, %2
  br i1 %128, label %129, label %122, !llvm.loop !85

129:                                              ; preds = %127, %122
  %130 = phi ptr [ %2, %127 ], [ %123, %122 ]
  store i64 %120, ptr %130, align 8, !tbaa !33
  %131 = getelementptr inbounds nuw i8, ptr %119, i64 8
  %132 = icmp eq ptr %131, %3
  br i1 %132, label %133, label %118, !llvm.loop !86

133:                                              ; preds = %129, %137
  %134 = phi ptr [ %135, %137 ], [ %2, %129 ]
  %135 = getelementptr i8, ptr %134, i64 8
  %136 = icmp eq ptr %135, %3
  br i1 %136, label %145, label %137

137:                                              ; preds = %133
  %138 = load double, ptr %135, align 8, !tbaa !51
  %139 = load double, ptr %134, align 8, !tbaa !51
  %140 = fcmp olt double %138, %139
  br i1 %140, label %141, label %133, !llvm.loop !81

141:                                              ; preds = %137
  %142 = load i32, ptr @current_test, align 4, !tbaa !11
  %143 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.51, i32 noundef %142)
  %144 = load i32, ptr @iterations, align 4, !tbaa !11
  br label %145

145:                                              ; preds = %133, %141
  %146 = phi i32 [ %144, %141 ], [ %116, %133 ]
  %147 = add nuw nsw i32 %117, 1
  %148 = icmp slt i32 %147, %146
  br i1 %148, label %115, label %207, !llvm.loop !82

149:                                              ; preds = %100, %203
  %150 = phi i32 [ %204, %203 ], [ %15, %100 ]
  %151 = phi i32 [ %205, %203 ], [ 0, %100 ]
  br i1 %108, label %165, label %152

152:                                              ; preds = %149, %152
  %153 = phi i64 [ %162, %152 ], [ 0, %149 ]
  %154 = shl i64 %153, 3
  %155 = getelementptr i8, ptr %2, i64 %154
  %156 = shl i64 %153, 3
  %157 = getelementptr i8, ptr %0, i64 %156
  %158 = getelementptr i8, ptr %157, i64 16
  %159 = load <2 x i64>, ptr %157, align 8, !tbaa !33
  %160 = load <2 x i64>, ptr %158, align 8, !tbaa !33
  %161 = getelementptr i8, ptr %155, i64 16
  store <2 x i64> %159, ptr %155, align 8, !tbaa !33
  store <2 x i64> %160, ptr %161, align 8, !tbaa !33
  %162 = add nuw i64 %153, 4
  %163 = icmp eq i64 %162, %109
  br i1 %163, label %164, label %152, !llvm.loop !87

164:                                              ; preds = %152
  br i1 %114, label %175, label %165

165:                                              ; preds = %149, %164
  %166 = phi ptr [ %2, %149 ], [ %111, %164 ]
  %167 = phi ptr [ %0, %149 ], [ %113, %164 ]
  br label %168

168:                                              ; preds = %165, %168
  %169 = phi ptr [ %172, %168 ], [ %166, %165 ]
  %170 = phi ptr [ %171, %168 ], [ %167, %165 ]
  %171 = getelementptr inbounds nuw i8, ptr %170, i64 8
  %172 = getelementptr inbounds nuw i8, ptr %169, i64 8
  %173 = load i64, ptr %170, align 8, !tbaa !33
  store i64 %173, ptr %169, align 8, !tbaa !33
  %174 = icmp eq ptr %171, %1
  br i1 %174, label %175, label %168, !llvm.loop !88

175:                                              ; preds = %168, %164
  br label %176

176:                                              ; preds = %175, %187
  %177 = phi ptr [ %189, %187 ], [ %19, %175 ]
  %178 = load i64, ptr %177, align 8, !tbaa !33
  %179 = bitcast i64 %178 to double
  br label %180

180:                                              ; preds = %185, %176
  %181 = phi ptr [ %177, %176 ], [ %182, %185 ]
  %182 = getelementptr i8, ptr %181, i64 -8
  %183 = load double, ptr %182, align 8
  %184 = fcmp ogt double %183, %179
  br i1 %184, label %185, label %187

185:                                              ; preds = %180
  store double %183, ptr %181, align 8, !tbaa !33
  %186 = icmp eq ptr %182, %2
  br i1 %186, label %187, label %180, !llvm.loop !85

187:                                              ; preds = %185, %180
  %188 = phi ptr [ %2, %185 ], [ %181, %180 ]
  store i64 %178, ptr %188, align 8, !tbaa !33
  %189 = getelementptr inbounds nuw i8, ptr %177, i64 8
  %190 = icmp eq ptr %189, %3
  br i1 %190, label %191, label %176, !llvm.loop !86

191:                                              ; preds = %187, %195
  %192 = phi ptr [ %193, %195 ], [ %2, %187 ]
  %193 = getelementptr i8, ptr %192, i64 8
  %194 = icmp eq ptr %193, %3
  br i1 %194, label %203, label %195

195:                                              ; preds = %191
  %196 = load double, ptr %193, align 8, !tbaa !51
  %197 = load double, ptr %192, align 8, !tbaa !51
  %198 = fcmp olt double %196, %197
  br i1 %198, label %199, label %191, !llvm.loop !81

199:                                              ; preds = %195
  %200 = load i32, ptr @current_test, align 4, !tbaa !11
  %201 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.51, i32 noundef %200)
  %202 = load i32, ptr @iterations, align 4, !tbaa !11
  br label %203

203:                                              ; preds = %191, %199
  %204 = phi i32 [ %202, %199 ], [ %150, %191 ]
  %205 = add nuw nsw i32 %151, 1
  %206 = icmp slt i32 %205, %204
  br i1 %206, label %149, label %207, !llvm.loop !82

207:                                              ; preds = %203, %145, %95, %52, %6
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_Z19test_insertion_sortI14PointerWrapperI12ValueWrapperIdEES2_EvT_S4_S4_S4_T0_PKc(ptr %0, ptr %1, ptr %2, ptr %3, [1 x double] alignstack(8) %4, ptr noundef %5) local_unnamed_addr #9 comdat {
  %7 = ptrtoint ptr %0 to i64
  %8 = ptrtoint ptr %1 to i64
  %9 = ptrtoint ptr %0 to i64
  %10 = ptrtoint ptr %2 to i64
  %11 = ptrtoint ptr %0 to i64
  %12 = ptrtoint ptr %1 to i64
  %13 = ptrtoint ptr %0 to i64
  %14 = ptrtoint ptr %2 to i64
  %15 = load i32, ptr @iterations, align 4, !tbaa !11
  %16 = icmp sgt i32 %15, 0
  br i1 %16, label %17, label %207

17:                                               ; preds = %6
  %18 = icmp eq ptr %0, %1
  %19 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %20 = icmp eq ptr %19, %3
  br i1 %20, label %21, label %99

21:                                               ; preds = %17
  br i1 %18, label %37, label %22

22:                                               ; preds = %21
  %23 = sub i64 %10, %9
  %24 = add i64 %8, -8
  %25 = sub i64 %24, %7
  %26 = lshr i64 %25, 3
  %27 = add nuw nsw i64 %26, 1
  %28 = icmp ult i64 %25, 24
  %29 = icmp ult i64 %23, 32
  %30 = select i1 %28, i1 true, i1 %29
  %31 = and i64 %27, 4611686018427387900
  %32 = shl i64 %31, 3
  %33 = getelementptr i8, ptr %0, i64 %32
  %34 = shl i64 %31, 3
  %35 = getelementptr i8, ptr %2, i64 %34
  %36 = icmp eq i64 %27, %31
  br label %56

37:                                               ; preds = %21, %52
  %38 = phi i32 [ %53, %52 ], [ %15, %21 ]
  %39 = phi i32 [ %54, %52 ], [ 0, %21 ]
  br label %40

40:                                               ; preds = %44, %37
  %41 = phi ptr [ %2, %37 ], [ %42, %44 ]
  %42 = getelementptr i8, ptr %41, i64 8
  %43 = icmp eq ptr %42, %3
  br i1 %43, label %52, label %44

44:                                               ; preds = %40
  %45 = load double, ptr %42, align 8, !tbaa !51
  %46 = load double, ptr %41, align 8, !tbaa !51
  %47 = fcmp olt double %45, %46
  br i1 %47, label %48, label %40, !llvm.loop !89

48:                                               ; preds = %44
  %49 = load i32, ptr @current_test, align 4, !tbaa !11
  %50 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.51, i32 noundef %49)
  %51 = load i32, ptr @iterations, align 4, !tbaa !11
  br label %52

52:                                               ; preds = %40, %48
  %53 = phi i32 [ %51, %48 ], [ %38, %40 ]
  %54 = add nuw nsw i32 %39, 1
  %55 = icmp slt i32 %54, %53
  br i1 %55, label %37, label %207, !llvm.loop !90

56:                                               ; preds = %22, %95
  %57 = phi i32 [ %96, %95 ], [ %15, %22 ]
  %58 = phi i32 [ %97, %95 ], [ 0, %22 ]
  br i1 %30, label %72, label %59

59:                                               ; preds = %56, %59
  %60 = phi i64 [ %69, %59 ], [ 0, %56 ]
  %61 = shl i64 %60, 3
  %62 = getelementptr i8, ptr %0, i64 %61
  %63 = shl i64 %60, 3
  %64 = getelementptr i8, ptr %2, i64 %63
  %65 = getelementptr i8, ptr %62, i64 16
  %66 = load <2 x i64>, ptr %62, align 8, !tbaa !33
  %67 = load <2 x i64>, ptr %65, align 8, !tbaa !33
  %68 = getelementptr i8, ptr %64, i64 16
  store <2 x i64> %66, ptr %64, align 8, !tbaa !33
  store <2 x i64> %67, ptr %68, align 8, !tbaa !33
  %69 = add nuw i64 %60, 4
  %70 = icmp eq i64 %69, %31
  br i1 %70, label %71, label %59, !llvm.loop !91

71:                                               ; preds = %59
  br i1 %36, label %82, label %72

72:                                               ; preds = %56, %71
  %73 = phi ptr [ %0, %56 ], [ %33, %71 ]
  %74 = phi ptr [ %2, %56 ], [ %35, %71 ]
  br label %75

75:                                               ; preds = %72, %75
  %76 = phi ptr [ %78, %75 ], [ %73, %72 ]
  %77 = phi ptr [ %79, %75 ], [ %74, %72 ]
  %78 = getelementptr inbounds nuw i8, ptr %76, i64 8
  %79 = getelementptr inbounds nuw i8, ptr %77, i64 8
  %80 = load i64, ptr %76, align 8, !tbaa !33
  store i64 %80, ptr %77, align 8, !tbaa !33
  %81 = icmp eq ptr %78, %1
  br i1 %81, label %82, label %75, !llvm.loop !92

82:                                               ; preds = %75, %71
  br label %83

83:                                               ; preds = %82, %87
  %84 = phi ptr [ %85, %87 ], [ %2, %82 ]
  %85 = getelementptr i8, ptr %84, i64 8
  %86 = icmp eq ptr %85, %3
  br i1 %86, label %95, label %87

87:                                               ; preds = %83
  %88 = load double, ptr %85, align 8, !tbaa !51
  %89 = load double, ptr %84, align 8, !tbaa !51
  %90 = fcmp olt double %88, %89
  br i1 %90, label %91, label %83, !llvm.loop !89

91:                                               ; preds = %87
  %92 = load i32, ptr @current_test, align 4, !tbaa !11
  %93 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.51, i32 noundef %92)
  %94 = load i32, ptr @iterations, align 4, !tbaa !11
  br label %95

95:                                               ; preds = %83, %91
  %96 = phi i32 [ %94, %91 ], [ %57, %83 ]
  %97 = add nuw nsw i32 %58, 1
  %98 = icmp slt i32 %97, %96
  br i1 %98, label %56, label %207, !llvm.loop !90

99:                                               ; preds = %17
  br i1 %18, label %115, label %100

100:                                              ; preds = %99
  %101 = sub i64 %14, %13
  %102 = add i64 %12, -8
  %103 = sub i64 %102, %11
  %104 = lshr i64 %103, 3
  %105 = add nuw nsw i64 %104, 1
  %106 = icmp ult i64 %103, 24
  %107 = icmp ult i64 %101, 32
  %108 = select i1 %106, i1 true, i1 %107
  %109 = and i64 %105, 4611686018427387900
  %110 = shl i64 %109, 3
  %111 = getelementptr i8, ptr %0, i64 %110
  %112 = shl i64 %109, 3
  %113 = getelementptr i8, ptr %2, i64 %112
  %114 = icmp eq i64 %105, %109
  br label %149

115:                                              ; preds = %99, %145
  %116 = phi i32 [ %146, %145 ], [ %15, %99 ]
  %117 = phi i32 [ %147, %145 ], [ 0, %99 ]
  br label %118

118:                                              ; preds = %129, %115
  %119 = phi ptr [ %131, %129 ], [ %19, %115 ]
  %120 = load i64, ptr %119, align 8, !tbaa !33
  %121 = bitcast i64 %120 to double
  br label %122

122:                                              ; preds = %127, %118
  %123 = phi ptr [ %119, %118 ], [ %124, %127 ]
  %124 = getelementptr i8, ptr %123, i64 -8
  %125 = load double, ptr %124, align 8
  %126 = fcmp ogt double %125, %121
  br i1 %126, label %127, label %129

127:                                              ; preds = %122
  store double %125, ptr %123, align 8, !tbaa !33
  %128 = icmp eq ptr %124, %2
  br i1 %128, label %129, label %122, !llvm.loop !93

129:                                              ; preds = %127, %122
  %130 = phi ptr [ %2, %127 ], [ %123, %122 ]
  store i64 %120, ptr %130, align 8, !tbaa !33
  %131 = getelementptr inbounds nuw i8, ptr %119, i64 8
  %132 = icmp eq ptr %131, %3
  br i1 %132, label %133, label %118, !llvm.loop !94

133:                                              ; preds = %129, %137
  %134 = phi ptr [ %135, %137 ], [ %2, %129 ]
  %135 = getelementptr i8, ptr %134, i64 8
  %136 = icmp eq ptr %135, %3
  br i1 %136, label %145, label %137

137:                                              ; preds = %133
  %138 = load double, ptr %135, align 8, !tbaa !51
  %139 = load double, ptr %134, align 8, !tbaa !51
  %140 = fcmp olt double %138, %139
  br i1 %140, label %141, label %133, !llvm.loop !89

141:                                              ; preds = %137
  %142 = load i32, ptr @current_test, align 4, !tbaa !11
  %143 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.51, i32 noundef %142)
  %144 = load i32, ptr @iterations, align 4, !tbaa !11
  br label %145

145:                                              ; preds = %133, %141
  %146 = phi i32 [ %144, %141 ], [ %116, %133 ]
  %147 = add nuw nsw i32 %117, 1
  %148 = icmp slt i32 %147, %146
  br i1 %148, label %115, label %207, !llvm.loop !90

149:                                              ; preds = %100, %203
  %150 = phi i32 [ %204, %203 ], [ %15, %100 ]
  %151 = phi i32 [ %205, %203 ], [ 0, %100 ]
  br i1 %108, label %165, label %152

152:                                              ; preds = %149, %152
  %153 = phi i64 [ %162, %152 ], [ 0, %149 ]
  %154 = shl i64 %153, 3
  %155 = getelementptr i8, ptr %0, i64 %154
  %156 = shl i64 %153, 3
  %157 = getelementptr i8, ptr %2, i64 %156
  %158 = getelementptr i8, ptr %155, i64 16
  %159 = load <2 x i64>, ptr %155, align 8, !tbaa !33
  %160 = load <2 x i64>, ptr %158, align 8, !tbaa !33
  %161 = getelementptr i8, ptr %157, i64 16
  store <2 x i64> %159, ptr %157, align 8, !tbaa !33
  store <2 x i64> %160, ptr %161, align 8, !tbaa !33
  %162 = add nuw i64 %153, 4
  %163 = icmp eq i64 %162, %109
  br i1 %163, label %164, label %152, !llvm.loop !95

164:                                              ; preds = %152
  br i1 %114, label %175, label %165

165:                                              ; preds = %149, %164
  %166 = phi ptr [ %0, %149 ], [ %111, %164 ]
  %167 = phi ptr [ %2, %149 ], [ %113, %164 ]
  br label %168

168:                                              ; preds = %165, %168
  %169 = phi ptr [ %171, %168 ], [ %166, %165 ]
  %170 = phi ptr [ %172, %168 ], [ %167, %165 ]
  %171 = getelementptr inbounds nuw i8, ptr %169, i64 8
  %172 = getelementptr inbounds nuw i8, ptr %170, i64 8
  %173 = load i64, ptr %169, align 8, !tbaa !33
  store i64 %173, ptr %170, align 8, !tbaa !33
  %174 = icmp eq ptr %171, %1
  br i1 %174, label %175, label %168, !llvm.loop !96

175:                                              ; preds = %168, %164
  br label %176

176:                                              ; preds = %175, %187
  %177 = phi ptr [ %189, %187 ], [ %19, %175 ]
  %178 = load i64, ptr %177, align 8, !tbaa !33
  %179 = bitcast i64 %178 to double
  br label %180

180:                                              ; preds = %185, %176
  %181 = phi ptr [ %177, %176 ], [ %182, %185 ]
  %182 = getelementptr i8, ptr %181, i64 -8
  %183 = load double, ptr %182, align 8
  %184 = fcmp ogt double %183, %179
  br i1 %184, label %185, label %187

185:                                              ; preds = %180
  store double %183, ptr %181, align 8, !tbaa !33
  %186 = icmp eq ptr %182, %2
  br i1 %186, label %187, label %180, !llvm.loop !93

187:                                              ; preds = %185, %180
  %188 = phi ptr [ %2, %185 ], [ %181, %180 ]
  store i64 %178, ptr %188, align 8, !tbaa !33
  %189 = getelementptr inbounds nuw i8, ptr %177, i64 8
  %190 = icmp eq ptr %189, %3
  br i1 %190, label %191, label %176, !llvm.loop !94

191:                                              ; preds = %187, %195
  %192 = phi ptr [ %193, %195 ], [ %2, %187 ]
  %193 = getelementptr i8, ptr %192, i64 8
  %194 = icmp eq ptr %193, %3
  br i1 %194, label %203, label %195

195:                                              ; preds = %191
  %196 = load double, ptr %193, align 8, !tbaa !51
  %197 = load double, ptr %192, align 8, !tbaa !51
  %198 = fcmp olt double %196, %197
  br i1 %198, label %199, label %191, !llvm.loop !89

199:                                              ; preds = %195
  %200 = load i32, ptr @current_test, align 4, !tbaa !11
  %201 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.51, i32 noundef %200)
  %202 = load i32, ptr @iterations, align 4, !tbaa !11
  br label %203

203:                                              ; preds = %191, %199
  %204 = phi i32 [ %202, %199 ], [ %150, %191 ]
  %205 = add nuw nsw i32 %151, 1
  %206 = icmp slt i32 %205, %204
  br i1 %206, label %149, label %207, !llvm.loop !90

207:                                              ; preds = %203, %145, %95, %52, %6
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_Z19test_insertion_sortIP12ValueWrapperIS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IdEEEEEEEEEESA_EvT_SC_SC_SC_T0_PKc(ptr noundef %0, ptr noundef %1, ptr noundef %2, ptr noundef %3, [1 x double] alignstack(8) %4, ptr noundef %5) local_unnamed_addr #9 comdat {
  %7 = ptrtoint ptr %0 to i64
  %8 = ptrtoint ptr %1 to i64
  %9 = ptrtoint ptr %0 to i64
  %10 = ptrtoint ptr %2 to i64
  %11 = ptrtoint ptr %0 to i64
  %12 = ptrtoint ptr %1 to i64
  %13 = ptrtoint ptr %0 to i64
  %14 = ptrtoint ptr %2 to i64
  %15 = load i32, ptr @iterations, align 4, !tbaa !11
  %16 = icmp sgt i32 %15, 0
  br i1 %16, label %17, label %207

17:                                               ; preds = %6
  %18 = icmp eq ptr %0, %1
  %19 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %20 = icmp eq ptr %19, %3
  br i1 %20, label %21, label %99

21:                                               ; preds = %17
  br i1 %18, label %37, label %22

22:                                               ; preds = %21
  %23 = sub i64 %10, %9
  %24 = add i64 %8, -8
  %25 = sub i64 %24, %7
  %26 = lshr i64 %25, 3
  %27 = add nuw nsw i64 %26, 1
  %28 = icmp ult i64 %25, 24
  %29 = icmp ult i64 %23, 32
  %30 = or i1 %28, %29
  %31 = and i64 %27, 4611686018427387900
  %32 = shl i64 %31, 3
  %33 = getelementptr i8, ptr %2, i64 %32
  %34 = shl i64 %31, 3
  %35 = getelementptr i8, ptr %0, i64 %34
  %36 = icmp eq i64 %27, %31
  br label %56

37:                                               ; preds = %21, %52
  %38 = phi i32 [ %53, %52 ], [ %15, %21 ]
  %39 = phi i32 [ %54, %52 ], [ 0, %21 ]
  br label %40

40:                                               ; preds = %44, %37
  %41 = phi ptr [ %2, %37 ], [ %42, %44 ]
  %42 = getelementptr i8, ptr %41, i64 8
  %43 = icmp eq ptr %42, %3
  br i1 %43, label %52, label %44

44:                                               ; preds = %40
  %45 = load double, ptr %42, align 8, !tbaa !51
  %46 = load double, ptr %41, align 8, !tbaa !51
  %47 = fcmp olt double %45, %46
  br i1 %47, label %48, label %40, !llvm.loop !97

48:                                               ; preds = %44
  %49 = load i32, ptr @current_test, align 4, !tbaa !11
  %50 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.51, i32 noundef %49)
  %51 = load i32, ptr @iterations, align 4, !tbaa !11
  br label %52

52:                                               ; preds = %40, %48
  %53 = phi i32 [ %51, %48 ], [ %38, %40 ]
  %54 = add nuw nsw i32 %39, 1
  %55 = icmp slt i32 %54, %53
  br i1 %55, label %37, label %207, !llvm.loop !98

56:                                               ; preds = %22, %95
  %57 = phi i32 [ %96, %95 ], [ %15, %22 ]
  %58 = phi i32 [ %97, %95 ], [ 0, %22 ]
  br i1 %30, label %72, label %59

59:                                               ; preds = %56, %59
  %60 = phi i64 [ %69, %59 ], [ 0, %56 ]
  %61 = shl i64 %60, 3
  %62 = getelementptr i8, ptr %2, i64 %61
  %63 = shl i64 %60, 3
  %64 = getelementptr i8, ptr %0, i64 %63
  %65 = getelementptr i8, ptr %64, i64 16
  %66 = load <2 x i64>, ptr %64, align 8, !tbaa !33
  %67 = load <2 x i64>, ptr %65, align 8, !tbaa !33
  %68 = getelementptr i8, ptr %62, i64 16
  store <2 x i64> %66, ptr %62, align 8, !tbaa !33
  store <2 x i64> %67, ptr %68, align 8, !tbaa !33
  %69 = add nuw i64 %60, 4
  %70 = icmp eq i64 %69, %31
  br i1 %70, label %71, label %59, !llvm.loop !99

71:                                               ; preds = %59
  br i1 %36, label %82, label %72

72:                                               ; preds = %56, %71
  %73 = phi ptr [ %2, %56 ], [ %33, %71 ]
  %74 = phi ptr [ %0, %56 ], [ %35, %71 ]
  br label %75

75:                                               ; preds = %72, %75
  %76 = phi ptr [ %79, %75 ], [ %73, %72 ]
  %77 = phi ptr [ %78, %75 ], [ %74, %72 ]
  %78 = getelementptr inbounds nuw i8, ptr %77, i64 8
  %79 = getelementptr inbounds nuw i8, ptr %76, i64 8
  %80 = load i64, ptr %77, align 8, !tbaa !33
  store i64 %80, ptr %76, align 8, !tbaa !33
  %81 = icmp eq ptr %78, %1
  br i1 %81, label %82, label %75, !llvm.loop !100

82:                                               ; preds = %75, %71
  br label %83

83:                                               ; preds = %82, %87
  %84 = phi ptr [ %85, %87 ], [ %2, %82 ]
  %85 = getelementptr i8, ptr %84, i64 8
  %86 = icmp eq ptr %85, %3
  br i1 %86, label %95, label %87

87:                                               ; preds = %83
  %88 = load double, ptr %85, align 8, !tbaa !51
  %89 = load double, ptr %84, align 8, !tbaa !51
  %90 = fcmp olt double %88, %89
  br i1 %90, label %91, label %83, !llvm.loop !97

91:                                               ; preds = %87
  %92 = load i32, ptr @current_test, align 4, !tbaa !11
  %93 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.51, i32 noundef %92)
  %94 = load i32, ptr @iterations, align 4, !tbaa !11
  br label %95

95:                                               ; preds = %83, %91
  %96 = phi i32 [ %94, %91 ], [ %57, %83 ]
  %97 = add nuw nsw i32 %58, 1
  %98 = icmp slt i32 %97, %96
  br i1 %98, label %56, label %207, !llvm.loop !98

99:                                               ; preds = %17
  br i1 %18, label %115, label %100

100:                                              ; preds = %99
  %101 = sub i64 %14, %13
  %102 = add i64 %12, -8
  %103 = sub i64 %102, %11
  %104 = lshr i64 %103, 3
  %105 = add nuw nsw i64 %104, 1
  %106 = icmp ult i64 %103, 24
  %107 = icmp ult i64 %101, 32
  %108 = or i1 %106, %107
  %109 = and i64 %105, 4611686018427387900
  %110 = shl i64 %109, 3
  %111 = getelementptr i8, ptr %2, i64 %110
  %112 = shl i64 %109, 3
  %113 = getelementptr i8, ptr %0, i64 %112
  %114 = icmp eq i64 %105, %109
  br label %149

115:                                              ; preds = %99, %145
  %116 = phi i32 [ %146, %145 ], [ %15, %99 ]
  %117 = phi i32 [ %147, %145 ], [ 0, %99 ]
  br label %118

118:                                              ; preds = %129, %115
  %119 = phi ptr [ %131, %129 ], [ %19, %115 ]
  %120 = load i64, ptr %119, align 8, !tbaa !33
  %121 = bitcast i64 %120 to double
  br label %122

122:                                              ; preds = %127, %118
  %123 = phi ptr [ %119, %118 ], [ %124, %127 ]
  %124 = getelementptr i8, ptr %123, i64 -8
  %125 = load double, ptr %124, align 8
  %126 = fcmp ogt double %125, %121
  br i1 %126, label %127, label %129

127:                                              ; preds = %122
  store double %125, ptr %123, align 8, !tbaa !33
  %128 = icmp eq ptr %124, %2
  br i1 %128, label %129, label %122, !llvm.loop !101

129:                                              ; preds = %127, %122
  %130 = phi ptr [ %2, %127 ], [ %123, %122 ]
  store i64 %120, ptr %130, align 8, !tbaa !33
  %131 = getelementptr inbounds nuw i8, ptr %119, i64 8
  %132 = icmp eq ptr %131, %3
  br i1 %132, label %133, label %118, !llvm.loop !102

133:                                              ; preds = %129, %137
  %134 = phi ptr [ %135, %137 ], [ %2, %129 ]
  %135 = getelementptr i8, ptr %134, i64 8
  %136 = icmp eq ptr %135, %3
  br i1 %136, label %145, label %137

137:                                              ; preds = %133
  %138 = load double, ptr %135, align 8, !tbaa !51
  %139 = load double, ptr %134, align 8, !tbaa !51
  %140 = fcmp olt double %138, %139
  br i1 %140, label %141, label %133, !llvm.loop !97

141:                                              ; preds = %137
  %142 = load i32, ptr @current_test, align 4, !tbaa !11
  %143 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.51, i32 noundef %142)
  %144 = load i32, ptr @iterations, align 4, !tbaa !11
  br label %145

145:                                              ; preds = %133, %141
  %146 = phi i32 [ %144, %141 ], [ %116, %133 ]
  %147 = add nuw nsw i32 %117, 1
  %148 = icmp slt i32 %147, %146
  br i1 %148, label %115, label %207, !llvm.loop !98

149:                                              ; preds = %100, %203
  %150 = phi i32 [ %204, %203 ], [ %15, %100 ]
  %151 = phi i32 [ %205, %203 ], [ 0, %100 ]
  br i1 %108, label %165, label %152

152:                                              ; preds = %149, %152
  %153 = phi i64 [ %162, %152 ], [ 0, %149 ]
  %154 = shl i64 %153, 3
  %155 = getelementptr i8, ptr %2, i64 %154
  %156 = shl i64 %153, 3
  %157 = getelementptr i8, ptr %0, i64 %156
  %158 = getelementptr i8, ptr %157, i64 16
  %159 = load <2 x i64>, ptr %157, align 8, !tbaa !33
  %160 = load <2 x i64>, ptr %158, align 8, !tbaa !33
  %161 = getelementptr i8, ptr %155, i64 16
  store <2 x i64> %159, ptr %155, align 8, !tbaa !33
  store <2 x i64> %160, ptr %161, align 8, !tbaa !33
  %162 = add nuw i64 %153, 4
  %163 = icmp eq i64 %162, %109
  br i1 %163, label %164, label %152, !llvm.loop !103

164:                                              ; preds = %152
  br i1 %114, label %175, label %165

165:                                              ; preds = %149, %164
  %166 = phi ptr [ %2, %149 ], [ %111, %164 ]
  %167 = phi ptr [ %0, %149 ], [ %113, %164 ]
  br label %168

168:                                              ; preds = %165, %168
  %169 = phi ptr [ %172, %168 ], [ %166, %165 ]
  %170 = phi ptr [ %171, %168 ], [ %167, %165 ]
  %171 = getelementptr inbounds nuw i8, ptr %170, i64 8
  %172 = getelementptr inbounds nuw i8, ptr %169, i64 8
  %173 = load i64, ptr %170, align 8, !tbaa !33
  store i64 %173, ptr %169, align 8, !tbaa !33
  %174 = icmp eq ptr %171, %1
  br i1 %174, label %175, label %168, !llvm.loop !104

175:                                              ; preds = %168, %164
  br label %176

176:                                              ; preds = %175, %187
  %177 = phi ptr [ %189, %187 ], [ %19, %175 ]
  %178 = load i64, ptr %177, align 8, !tbaa !33
  %179 = bitcast i64 %178 to double
  br label %180

180:                                              ; preds = %185, %176
  %181 = phi ptr [ %177, %176 ], [ %182, %185 ]
  %182 = getelementptr i8, ptr %181, i64 -8
  %183 = load double, ptr %182, align 8
  %184 = fcmp ogt double %183, %179
  br i1 %184, label %185, label %187

185:                                              ; preds = %180
  store double %183, ptr %181, align 8, !tbaa !33
  %186 = icmp eq ptr %182, %2
  br i1 %186, label %187, label %180, !llvm.loop !101

187:                                              ; preds = %185, %180
  %188 = phi ptr [ %2, %185 ], [ %181, %180 ]
  store i64 %178, ptr %188, align 8, !tbaa !33
  %189 = getelementptr inbounds nuw i8, ptr %177, i64 8
  %190 = icmp eq ptr %189, %3
  br i1 %190, label %191, label %176, !llvm.loop !102

191:                                              ; preds = %187, %195
  %192 = phi ptr [ %193, %195 ], [ %2, %187 ]
  %193 = getelementptr i8, ptr %192, i64 8
  %194 = icmp eq ptr %193, %3
  br i1 %194, label %203, label %195

195:                                              ; preds = %191
  %196 = load double, ptr %193, align 8, !tbaa !51
  %197 = load double, ptr %192, align 8, !tbaa !51
  %198 = fcmp olt double %196, %197
  br i1 %198, label %199, label %191, !llvm.loop !97

199:                                              ; preds = %195
  %200 = load i32, ptr @current_test, align 4, !tbaa !11
  %201 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.51, i32 noundef %200)
  %202 = load i32, ptr @iterations, align 4, !tbaa !11
  br label %203

203:                                              ; preds = %191, %199
  %204 = phi i32 [ %202, %199 ], [ %150, %191 ]
  %205 = add nuw nsw i32 %151, 1
  %206 = icmp slt i32 %205, %204
  br i1 %206, label %149, label %207, !llvm.loop !98

207:                                              ; preds = %203, %145, %95, %52, %6
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_Z19test_insertion_sortI14PointerWrapperI12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEEESB_EvT_SD_SD_SD_T0_PKc(ptr %0, ptr %1, ptr %2, ptr %3, [1 x double] alignstack(8) %4, ptr noundef %5) local_unnamed_addr #9 comdat {
  %7 = ptrtoint ptr %0 to i64
  %8 = ptrtoint ptr %1 to i64
  %9 = ptrtoint ptr %0 to i64
  %10 = ptrtoint ptr %2 to i64
  %11 = ptrtoint ptr %0 to i64
  %12 = ptrtoint ptr %1 to i64
  %13 = ptrtoint ptr %0 to i64
  %14 = ptrtoint ptr %2 to i64
  %15 = load i32, ptr @iterations, align 4, !tbaa !11
  %16 = icmp sgt i32 %15, 0
  br i1 %16, label %17, label %207

17:                                               ; preds = %6
  %18 = icmp eq ptr %0, %1
  %19 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %20 = icmp eq ptr %19, %3
  br i1 %20, label %21, label %99

21:                                               ; preds = %17
  br i1 %18, label %37, label %22

22:                                               ; preds = %21
  %23 = sub i64 %10, %9
  %24 = add i64 %8, -8
  %25 = sub i64 %24, %7
  %26 = lshr i64 %25, 3
  %27 = add nuw nsw i64 %26, 1
  %28 = icmp ult i64 %25, 24
  %29 = icmp ult i64 %23, 32
  %30 = select i1 %28, i1 true, i1 %29
  %31 = and i64 %27, 4611686018427387900
  %32 = shl i64 %31, 3
  %33 = getelementptr i8, ptr %0, i64 %32
  %34 = shl i64 %31, 3
  %35 = getelementptr i8, ptr %2, i64 %34
  %36 = icmp eq i64 %27, %31
  br label %56

37:                                               ; preds = %21, %52
  %38 = phi i32 [ %53, %52 ], [ %15, %21 ]
  %39 = phi i32 [ %54, %52 ], [ 0, %21 ]
  br label %40

40:                                               ; preds = %44, %37
  %41 = phi ptr [ %2, %37 ], [ %42, %44 ]
  %42 = getelementptr i8, ptr %41, i64 8
  %43 = icmp eq ptr %42, %3
  br i1 %43, label %52, label %44

44:                                               ; preds = %40
  %45 = load double, ptr %42, align 8, !tbaa !51
  %46 = load double, ptr %41, align 8, !tbaa !51
  %47 = fcmp olt double %45, %46
  br i1 %47, label %48, label %40, !llvm.loop !105

48:                                               ; preds = %44
  %49 = load i32, ptr @current_test, align 4, !tbaa !11
  %50 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.51, i32 noundef %49)
  %51 = load i32, ptr @iterations, align 4, !tbaa !11
  br label %52

52:                                               ; preds = %40, %48
  %53 = phi i32 [ %51, %48 ], [ %38, %40 ]
  %54 = add nuw nsw i32 %39, 1
  %55 = icmp slt i32 %54, %53
  br i1 %55, label %37, label %207, !llvm.loop !106

56:                                               ; preds = %22, %95
  %57 = phi i32 [ %96, %95 ], [ %15, %22 ]
  %58 = phi i32 [ %97, %95 ], [ 0, %22 ]
  br i1 %30, label %72, label %59

59:                                               ; preds = %56, %59
  %60 = phi i64 [ %69, %59 ], [ 0, %56 ]
  %61 = shl i64 %60, 3
  %62 = getelementptr i8, ptr %0, i64 %61
  %63 = shl i64 %60, 3
  %64 = getelementptr i8, ptr %2, i64 %63
  %65 = getelementptr i8, ptr %62, i64 16
  %66 = load <2 x i64>, ptr %62, align 8, !tbaa !33
  %67 = load <2 x i64>, ptr %65, align 8, !tbaa !33
  %68 = getelementptr i8, ptr %64, i64 16
  store <2 x i64> %66, ptr %64, align 8, !tbaa !33
  store <2 x i64> %67, ptr %68, align 8, !tbaa !33
  %69 = add nuw i64 %60, 4
  %70 = icmp eq i64 %69, %31
  br i1 %70, label %71, label %59, !llvm.loop !107

71:                                               ; preds = %59
  br i1 %36, label %82, label %72

72:                                               ; preds = %56, %71
  %73 = phi ptr [ %0, %56 ], [ %33, %71 ]
  %74 = phi ptr [ %2, %56 ], [ %35, %71 ]
  br label %75

75:                                               ; preds = %72, %75
  %76 = phi ptr [ %78, %75 ], [ %73, %72 ]
  %77 = phi ptr [ %79, %75 ], [ %74, %72 ]
  %78 = getelementptr inbounds nuw i8, ptr %76, i64 8
  %79 = getelementptr inbounds nuw i8, ptr %77, i64 8
  %80 = load i64, ptr %76, align 8, !tbaa !33
  store i64 %80, ptr %77, align 8, !tbaa !33
  %81 = icmp eq ptr %78, %1
  br i1 %81, label %82, label %75, !llvm.loop !108

82:                                               ; preds = %75, %71
  br label %83

83:                                               ; preds = %82, %87
  %84 = phi ptr [ %85, %87 ], [ %2, %82 ]
  %85 = getelementptr i8, ptr %84, i64 8
  %86 = icmp eq ptr %85, %3
  br i1 %86, label %95, label %87

87:                                               ; preds = %83
  %88 = load double, ptr %85, align 8, !tbaa !51
  %89 = load double, ptr %84, align 8, !tbaa !51
  %90 = fcmp olt double %88, %89
  br i1 %90, label %91, label %83, !llvm.loop !105

91:                                               ; preds = %87
  %92 = load i32, ptr @current_test, align 4, !tbaa !11
  %93 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.51, i32 noundef %92)
  %94 = load i32, ptr @iterations, align 4, !tbaa !11
  br label %95

95:                                               ; preds = %83, %91
  %96 = phi i32 [ %94, %91 ], [ %57, %83 ]
  %97 = add nuw nsw i32 %58, 1
  %98 = icmp slt i32 %97, %96
  br i1 %98, label %56, label %207, !llvm.loop !106

99:                                               ; preds = %17
  br i1 %18, label %115, label %100

100:                                              ; preds = %99
  %101 = sub i64 %14, %13
  %102 = add i64 %12, -8
  %103 = sub i64 %102, %11
  %104 = lshr i64 %103, 3
  %105 = add nuw nsw i64 %104, 1
  %106 = icmp ult i64 %103, 24
  %107 = icmp ult i64 %101, 32
  %108 = select i1 %106, i1 true, i1 %107
  %109 = and i64 %105, 4611686018427387900
  %110 = shl i64 %109, 3
  %111 = getelementptr i8, ptr %0, i64 %110
  %112 = shl i64 %109, 3
  %113 = getelementptr i8, ptr %2, i64 %112
  %114 = icmp eq i64 %105, %109
  br label %149

115:                                              ; preds = %99, %145
  %116 = phi i32 [ %146, %145 ], [ %15, %99 ]
  %117 = phi i32 [ %147, %145 ], [ 0, %99 ]
  br label %118

118:                                              ; preds = %129, %115
  %119 = phi ptr [ %131, %129 ], [ %19, %115 ]
  %120 = load i64, ptr %119, align 8, !tbaa !33
  %121 = bitcast i64 %120 to double
  br label %122

122:                                              ; preds = %127, %118
  %123 = phi ptr [ %119, %118 ], [ %124, %127 ]
  %124 = getelementptr i8, ptr %123, i64 -8
  %125 = load double, ptr %124, align 8
  %126 = fcmp ogt double %125, %121
  br i1 %126, label %127, label %129

127:                                              ; preds = %122
  store double %125, ptr %123, align 8, !tbaa !33
  %128 = icmp eq ptr %124, %2
  br i1 %128, label %129, label %122, !llvm.loop !109

129:                                              ; preds = %127, %122
  %130 = phi ptr [ %2, %127 ], [ %123, %122 ]
  store i64 %120, ptr %130, align 8, !tbaa !33
  %131 = getelementptr inbounds nuw i8, ptr %119, i64 8
  %132 = icmp eq ptr %131, %3
  br i1 %132, label %133, label %118, !llvm.loop !110

133:                                              ; preds = %129, %137
  %134 = phi ptr [ %135, %137 ], [ %2, %129 ]
  %135 = getelementptr i8, ptr %134, i64 8
  %136 = icmp eq ptr %135, %3
  br i1 %136, label %145, label %137

137:                                              ; preds = %133
  %138 = load double, ptr %135, align 8, !tbaa !51
  %139 = load double, ptr %134, align 8, !tbaa !51
  %140 = fcmp olt double %138, %139
  br i1 %140, label %141, label %133, !llvm.loop !105

141:                                              ; preds = %137
  %142 = load i32, ptr @current_test, align 4, !tbaa !11
  %143 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.51, i32 noundef %142)
  %144 = load i32, ptr @iterations, align 4, !tbaa !11
  br label %145

145:                                              ; preds = %133, %141
  %146 = phi i32 [ %144, %141 ], [ %116, %133 ]
  %147 = add nuw nsw i32 %117, 1
  %148 = icmp slt i32 %147, %146
  br i1 %148, label %115, label %207, !llvm.loop !106

149:                                              ; preds = %100, %203
  %150 = phi i32 [ %204, %203 ], [ %15, %100 ]
  %151 = phi i32 [ %205, %203 ], [ 0, %100 ]
  br i1 %108, label %165, label %152

152:                                              ; preds = %149, %152
  %153 = phi i64 [ %162, %152 ], [ 0, %149 ]
  %154 = shl i64 %153, 3
  %155 = getelementptr i8, ptr %0, i64 %154
  %156 = shl i64 %153, 3
  %157 = getelementptr i8, ptr %2, i64 %156
  %158 = getelementptr i8, ptr %155, i64 16
  %159 = load <2 x i64>, ptr %155, align 8, !tbaa !33
  %160 = load <2 x i64>, ptr %158, align 8, !tbaa !33
  %161 = getelementptr i8, ptr %157, i64 16
  store <2 x i64> %159, ptr %157, align 8, !tbaa !33
  store <2 x i64> %160, ptr %161, align 8, !tbaa !33
  %162 = add nuw i64 %153, 4
  %163 = icmp eq i64 %162, %109
  br i1 %163, label %164, label %152, !llvm.loop !111

164:                                              ; preds = %152
  br i1 %114, label %175, label %165

165:                                              ; preds = %149, %164
  %166 = phi ptr [ %0, %149 ], [ %111, %164 ]
  %167 = phi ptr [ %2, %149 ], [ %113, %164 ]
  br label %168

168:                                              ; preds = %165, %168
  %169 = phi ptr [ %171, %168 ], [ %166, %165 ]
  %170 = phi ptr [ %172, %168 ], [ %167, %165 ]
  %171 = getelementptr inbounds nuw i8, ptr %169, i64 8
  %172 = getelementptr inbounds nuw i8, ptr %170, i64 8
  %173 = load i64, ptr %169, align 8, !tbaa !33
  store i64 %173, ptr %170, align 8, !tbaa !33
  %174 = icmp eq ptr %171, %1
  br i1 %174, label %175, label %168, !llvm.loop !112

175:                                              ; preds = %168, %164
  br label %176

176:                                              ; preds = %175, %187
  %177 = phi ptr [ %189, %187 ], [ %19, %175 ]
  %178 = load i64, ptr %177, align 8, !tbaa !33
  %179 = bitcast i64 %178 to double
  br label %180

180:                                              ; preds = %185, %176
  %181 = phi ptr [ %177, %176 ], [ %182, %185 ]
  %182 = getelementptr i8, ptr %181, i64 -8
  %183 = load double, ptr %182, align 8
  %184 = fcmp ogt double %183, %179
  br i1 %184, label %185, label %187

185:                                              ; preds = %180
  store double %183, ptr %181, align 8, !tbaa !33
  %186 = icmp eq ptr %182, %2
  br i1 %186, label %187, label %180, !llvm.loop !109

187:                                              ; preds = %185, %180
  %188 = phi ptr [ %2, %185 ], [ %181, %180 ]
  store i64 %178, ptr %188, align 8, !tbaa !33
  %189 = getelementptr inbounds nuw i8, ptr %177, i64 8
  %190 = icmp eq ptr %189, %3
  br i1 %190, label %191, label %176, !llvm.loop !110

191:                                              ; preds = %187, %195
  %192 = phi ptr [ %193, %195 ], [ %2, %187 ]
  %193 = getelementptr i8, ptr %192, i64 8
  %194 = icmp eq ptr %193, %3
  br i1 %194, label %203, label %195

195:                                              ; preds = %191
  %196 = load double, ptr %193, align 8, !tbaa !51
  %197 = load double, ptr %192, align 8, !tbaa !51
  %198 = fcmp olt double %196, %197
  br i1 %198, label %199, label %191, !llvm.loop !105

199:                                              ; preds = %195
  %200 = load i32, ptr @current_test, align 4, !tbaa !11
  %201 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.51, i32 noundef %200)
  %202 = load i32, ptr @iterations, align 4, !tbaa !11
  br label %203

203:                                              ; preds = %191, %199
  %204 = phi i32 [ %202, %199 ], [ %150, %191 ]
  %205 = add nuw nsw i32 %151, 1
  %206 = icmp slt i32 %205, %204
  br i1 %206, label %149, label %207, !llvm.loop !106

207:                                              ; preds = %203, %145, %95, %52, %6
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_Z14test_quicksortIPddEvT_S1_S1_S1_T0_PKc(ptr noundef %0, ptr noundef %1, ptr noundef %2, ptr noundef %3, double noundef %4, ptr noundef %5) local_unnamed_addr #9 comdat {
  %7 = ptrtoint ptr %0 to i64
  %8 = ptrtoint ptr %1 to i64
  %9 = ptrtoint ptr %0 to i64
  %10 = ptrtoint ptr %2 to i64
  %11 = load i32, ptr @iterations, align 4, !tbaa !11
  %12 = icmp sgt i32 %11, 0
  br i1 %12, label %13, label %88

13:                                               ; preds = %6
  %14 = icmp eq ptr %0, %1
  br i1 %14, label %30, label %15

15:                                               ; preds = %13
  %16 = sub i64 %10, %9
  %17 = add i64 %8, -8
  %18 = sub i64 %17, %7
  %19 = lshr i64 %18, 3
  %20 = add nuw nsw i64 %19, 1
  %21 = icmp ult i64 %18, 24
  %22 = icmp ult i64 %16, 32
  %23 = or i1 %21, %22
  %24 = and i64 %20, 4611686018427387900
  %25 = shl i64 %24, 3
  %26 = getelementptr i8, ptr %2, i64 %25
  %27 = shl i64 %24, 3
  %28 = getelementptr i8, ptr %0, i64 %27
  %29 = icmp eq i64 %20, %24
  br label %47

30:                                               ; preds = %13, %43
  %31 = phi i32 [ %44, %43 ], [ 0, %13 ]
  tail call void @_ZN9benchmark9quicksortIPddEEvT_S2_(ptr noundef %2, ptr noundef %3)
  br label %32

32:                                               ; preds = %36, %30
  %33 = phi ptr [ %2, %30 ], [ %34, %36 ]
  %34 = getelementptr i8, ptr %33, i64 8
  %35 = icmp eq ptr %34, %3
  br i1 %35, label %43, label %36

36:                                               ; preds = %32
  %37 = load double, ptr %34, align 8, !tbaa !33
  %38 = load double, ptr %33, align 8, !tbaa !33
  %39 = fcmp olt double %37, %38
  br i1 %39, label %40, label %32, !llvm.loop !65

40:                                               ; preds = %36
  %41 = load i32, ptr @current_test, align 4, !tbaa !11
  %42 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.51, i32 noundef %41)
  br label %43

43:                                               ; preds = %32, %40
  %44 = add nuw nsw i32 %31, 1
  %45 = load i32, ptr @iterations, align 4, !tbaa !11
  %46 = icmp slt i32 %44, %45
  br i1 %46, label %30, label %88, !llvm.loop !113

47:                                               ; preds = %15, %84
  %48 = phi i32 [ %85, %84 ], [ 0, %15 ]
  br i1 %23, label %62, label %49

49:                                               ; preds = %47, %49
  %50 = phi i64 [ %59, %49 ], [ 0, %47 ]
  %51 = shl i64 %50, 3
  %52 = getelementptr i8, ptr %2, i64 %51
  %53 = shl i64 %50, 3
  %54 = getelementptr i8, ptr %0, i64 %53
  %55 = getelementptr i8, ptr %54, i64 16
  %56 = load <2 x double>, ptr %54, align 8, !tbaa !33
  %57 = load <2 x double>, ptr %55, align 8, !tbaa !33
  %58 = getelementptr i8, ptr %52, i64 16
  store <2 x double> %56, ptr %52, align 8, !tbaa !33
  store <2 x double> %57, ptr %58, align 8, !tbaa !33
  %59 = add nuw i64 %50, 4
  %60 = icmp eq i64 %59, %24
  br i1 %60, label %61, label %49, !llvm.loop !114

61:                                               ; preds = %49
  br i1 %29, label %72, label %62

62:                                               ; preds = %47, %61
  %63 = phi ptr [ %2, %47 ], [ %26, %61 ]
  %64 = phi ptr [ %0, %47 ], [ %28, %61 ]
  br label %65

65:                                               ; preds = %62, %65
  %66 = phi ptr [ %70, %65 ], [ %63, %62 ]
  %67 = phi ptr [ %68, %65 ], [ %64, %62 ]
  %68 = getelementptr inbounds nuw i8, ptr %67, i64 8
  %69 = load double, ptr %67, align 8, !tbaa !33
  %70 = getelementptr inbounds nuw i8, ptr %66, i64 8
  store double %69, ptr %66, align 8, !tbaa !33
  %71 = icmp eq ptr %68, %1
  br i1 %71, label %72, label %65, !llvm.loop !115

72:                                               ; preds = %65, %61
  tail call void @_ZN9benchmark9quicksortIPddEEvT_S2_(ptr noundef %2, ptr noundef %3)
  br label %73

73:                                               ; preds = %77, %72
  %74 = phi ptr [ %2, %72 ], [ %75, %77 ]
  %75 = getelementptr i8, ptr %74, i64 8
  %76 = icmp eq ptr %75, %3
  br i1 %76, label %84, label %77

77:                                               ; preds = %73
  %78 = load double, ptr %75, align 8, !tbaa !33
  %79 = load double, ptr %74, align 8, !tbaa !33
  %80 = fcmp olt double %78, %79
  br i1 %80, label %81, label %73, !llvm.loop !65

81:                                               ; preds = %77
  %82 = load i32, ptr @current_test, align 4, !tbaa !11
  %83 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.51, i32 noundef %82)
  br label %84

84:                                               ; preds = %73, %81
  %85 = add nuw nsw i32 %48, 1
  %86 = load i32, ptr @iterations, align 4, !tbaa !11
  %87 = icmp slt i32 %85, %86
  br i1 %87, label %47, label %88, !llvm.loop !113

88:                                               ; preds = %84, %43, %6
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_Z14test_quicksortI14PointerWrapperIdEdEvT_S2_S2_S2_T0_PKc(ptr %0, ptr %1, ptr %2, ptr %3, double noundef %4, ptr noundef %5) local_unnamed_addr #9 comdat {
  %7 = ptrtoint ptr %0 to i64
  %8 = ptrtoint ptr %1 to i64
  %9 = ptrtoint ptr %0 to i64
  %10 = ptrtoint ptr %2 to i64
  %11 = load i32, ptr @iterations, align 4, !tbaa !11
  %12 = icmp sgt i32 %11, 0
  br i1 %12, label %13, label %88

13:                                               ; preds = %6
  %14 = icmp eq ptr %0, %1
  br i1 %14, label %30, label %15

15:                                               ; preds = %13
  %16 = sub i64 %10, %9
  %17 = add i64 %8, -8
  %18 = sub i64 %17, %7
  %19 = lshr i64 %18, 3
  %20 = add nuw nsw i64 %19, 1
  %21 = icmp ult i64 %18, 24
  %22 = icmp ult i64 %16, 32
  %23 = select i1 %21, i1 true, i1 %22
  %24 = and i64 %20, 4611686018427387900
  %25 = shl i64 %24, 3
  %26 = getelementptr i8, ptr %0, i64 %25
  %27 = shl i64 %24, 3
  %28 = getelementptr i8, ptr %2, i64 %27
  %29 = icmp eq i64 %20, %24
  br label %47

30:                                               ; preds = %13, %43
  %31 = phi i32 [ %44, %43 ], [ 0, %13 ]
  tail call void @_ZN9benchmark9quicksortI14PointerWrapperIdEdEEvT_S3_(ptr %2, ptr %3)
  br label %32

32:                                               ; preds = %36, %30
  %33 = phi ptr [ %2, %30 ], [ %34, %36 ]
  %34 = getelementptr i8, ptr %33, i64 8
  %35 = icmp eq ptr %34, %3
  br i1 %35, label %43, label %36

36:                                               ; preds = %32
  %37 = load double, ptr %34, align 8, !tbaa !33
  %38 = load double, ptr %33, align 8, !tbaa !33
  %39 = fcmp olt double %37, %38
  br i1 %39, label %40, label %32, !llvm.loop !73

40:                                               ; preds = %36
  %41 = load i32, ptr @current_test, align 4, !tbaa !11
  %42 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.51, i32 noundef %41)
  br label %43

43:                                               ; preds = %32, %40
  %44 = add nuw nsw i32 %31, 1
  %45 = load i32, ptr @iterations, align 4, !tbaa !11
  %46 = icmp slt i32 %44, %45
  br i1 %46, label %30, label %88, !llvm.loop !116

47:                                               ; preds = %15, %84
  %48 = phi i32 [ %85, %84 ], [ 0, %15 ]
  br i1 %23, label %62, label %49

49:                                               ; preds = %47, %49
  %50 = phi i64 [ %59, %49 ], [ 0, %47 ]
  %51 = shl i64 %50, 3
  %52 = getelementptr i8, ptr %0, i64 %51
  %53 = shl i64 %50, 3
  %54 = getelementptr i8, ptr %2, i64 %53
  %55 = getelementptr i8, ptr %52, i64 16
  %56 = load <2 x double>, ptr %52, align 8, !tbaa !33
  %57 = load <2 x double>, ptr %55, align 8, !tbaa !33
  %58 = getelementptr i8, ptr %54, i64 16
  store <2 x double> %56, ptr %54, align 8, !tbaa !33
  store <2 x double> %57, ptr %58, align 8, !tbaa !33
  %59 = add nuw i64 %50, 4
  %60 = icmp eq i64 %59, %24
  br i1 %60, label %61, label %49, !llvm.loop !117

61:                                               ; preds = %49
  br i1 %29, label %72, label %62

62:                                               ; preds = %47, %61
  %63 = phi ptr [ %0, %47 ], [ %26, %61 ]
  %64 = phi ptr [ %2, %47 ], [ %28, %61 ]
  br label %65

65:                                               ; preds = %62, %65
  %66 = phi ptr [ %68, %65 ], [ %63, %62 ]
  %67 = phi ptr [ %70, %65 ], [ %64, %62 ]
  %68 = getelementptr inbounds nuw i8, ptr %66, i64 8
  %69 = load double, ptr %66, align 8, !tbaa !33
  %70 = getelementptr inbounds nuw i8, ptr %67, i64 8
  store double %69, ptr %67, align 8, !tbaa !33
  %71 = icmp eq ptr %68, %1
  br i1 %71, label %72, label %65, !llvm.loop !118

72:                                               ; preds = %65, %61
  tail call void @_ZN9benchmark9quicksortI14PointerWrapperIdEdEEvT_S3_(ptr %2, ptr %3)
  br label %73

73:                                               ; preds = %77, %72
  %74 = phi ptr [ %2, %72 ], [ %75, %77 ]
  %75 = getelementptr i8, ptr %74, i64 8
  %76 = icmp eq ptr %75, %3
  br i1 %76, label %84, label %77

77:                                               ; preds = %73
  %78 = load double, ptr %75, align 8, !tbaa !33
  %79 = load double, ptr %74, align 8, !tbaa !33
  %80 = fcmp olt double %78, %79
  br i1 %80, label %81, label %73, !llvm.loop !73

81:                                               ; preds = %77
  %82 = load i32, ptr @current_test, align 4, !tbaa !11
  %83 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.51, i32 noundef %82)
  br label %84

84:                                               ; preds = %73, %81
  %85 = add nuw nsw i32 %48, 1
  %86 = load i32, ptr @iterations, align 4, !tbaa !11
  %87 = icmp slt i32 %85, %86
  br i1 %87, label %47, label %88, !llvm.loop !116

88:                                               ; preds = %84, %43, %6
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_Z14test_quicksortIP12ValueWrapperIdES1_EvT_S3_S3_S3_T0_PKc(ptr noundef %0, ptr noundef %1, ptr noundef %2, ptr noundef %3, [1 x double] alignstack(8) %4, ptr noundef %5) local_unnamed_addr #9 comdat {
  %7 = ptrtoint ptr %0 to i64
  %8 = ptrtoint ptr %1 to i64
  %9 = ptrtoint ptr %0 to i64
  %10 = ptrtoint ptr %2 to i64
  %11 = load i32, ptr @iterations, align 4, !tbaa !11
  %12 = icmp sgt i32 %11, 0
  br i1 %12, label %13, label %88

13:                                               ; preds = %6
  %14 = icmp eq ptr %0, %1
  br i1 %14, label %30, label %15

15:                                               ; preds = %13
  %16 = sub i64 %10, %9
  %17 = add i64 %8, -8
  %18 = sub i64 %17, %7
  %19 = lshr i64 %18, 3
  %20 = add nuw nsw i64 %19, 1
  %21 = icmp ult i64 %18, 24
  %22 = icmp ult i64 %16, 32
  %23 = or i1 %21, %22
  %24 = and i64 %20, 4611686018427387900
  %25 = shl i64 %24, 3
  %26 = getelementptr i8, ptr %2, i64 %25
  %27 = shl i64 %24, 3
  %28 = getelementptr i8, ptr %0, i64 %27
  %29 = icmp eq i64 %20, %24
  br label %47

30:                                               ; preds = %13, %43
  %31 = phi i32 [ %44, %43 ], [ 0, %13 ]
  tail call void @_ZN9benchmark9quicksortIP12ValueWrapperIdES2_EEvT_S4_(ptr noundef %2, ptr noundef %3)
  br label %32

32:                                               ; preds = %36, %30
  %33 = phi ptr [ %2, %30 ], [ %34, %36 ]
  %34 = getelementptr i8, ptr %33, i64 8
  %35 = icmp eq ptr %34, %3
  br i1 %35, label %43, label %36

36:                                               ; preds = %32
  %37 = load double, ptr %34, align 8, !tbaa !51
  %38 = load double, ptr %33, align 8, !tbaa !51
  %39 = fcmp olt double %37, %38
  br i1 %39, label %40, label %32, !llvm.loop !81

40:                                               ; preds = %36
  %41 = load i32, ptr @current_test, align 4, !tbaa !11
  %42 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.51, i32 noundef %41)
  br label %43

43:                                               ; preds = %32, %40
  %44 = add nuw nsw i32 %31, 1
  %45 = load i32, ptr @iterations, align 4, !tbaa !11
  %46 = icmp slt i32 %44, %45
  br i1 %46, label %30, label %88, !llvm.loop !119

47:                                               ; preds = %15, %84
  %48 = phi i32 [ %85, %84 ], [ 0, %15 ]
  br i1 %23, label %62, label %49

49:                                               ; preds = %47, %49
  %50 = phi i64 [ %59, %49 ], [ 0, %47 ]
  %51 = shl i64 %50, 3
  %52 = getelementptr i8, ptr %2, i64 %51
  %53 = shl i64 %50, 3
  %54 = getelementptr i8, ptr %0, i64 %53
  %55 = getelementptr i8, ptr %54, i64 16
  %56 = load <2 x i64>, ptr %54, align 8, !tbaa !33
  %57 = load <2 x i64>, ptr %55, align 8, !tbaa !33
  %58 = getelementptr i8, ptr %52, i64 16
  store <2 x i64> %56, ptr %52, align 8, !tbaa !33
  store <2 x i64> %57, ptr %58, align 8, !tbaa !33
  %59 = add nuw i64 %50, 4
  %60 = icmp eq i64 %59, %24
  br i1 %60, label %61, label %49, !llvm.loop !120

61:                                               ; preds = %49
  br i1 %29, label %72, label %62

62:                                               ; preds = %47, %61
  %63 = phi ptr [ %2, %47 ], [ %26, %61 ]
  %64 = phi ptr [ %0, %47 ], [ %28, %61 ]
  br label %65

65:                                               ; preds = %62, %65
  %66 = phi ptr [ %69, %65 ], [ %63, %62 ]
  %67 = phi ptr [ %68, %65 ], [ %64, %62 ]
  %68 = getelementptr inbounds nuw i8, ptr %67, i64 8
  %69 = getelementptr inbounds nuw i8, ptr %66, i64 8
  %70 = load i64, ptr %67, align 8, !tbaa !33
  store i64 %70, ptr %66, align 8, !tbaa !33
  %71 = icmp eq ptr %68, %1
  br i1 %71, label %72, label %65, !llvm.loop !121

72:                                               ; preds = %65, %61
  tail call void @_ZN9benchmark9quicksortIP12ValueWrapperIdES2_EEvT_S4_(ptr noundef %2, ptr noundef %3)
  br label %73

73:                                               ; preds = %77, %72
  %74 = phi ptr [ %2, %72 ], [ %75, %77 ]
  %75 = getelementptr i8, ptr %74, i64 8
  %76 = icmp eq ptr %75, %3
  br i1 %76, label %84, label %77

77:                                               ; preds = %73
  %78 = load double, ptr %75, align 8, !tbaa !51
  %79 = load double, ptr %74, align 8, !tbaa !51
  %80 = fcmp olt double %78, %79
  br i1 %80, label %81, label %73, !llvm.loop !81

81:                                               ; preds = %77
  %82 = load i32, ptr @current_test, align 4, !tbaa !11
  %83 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.51, i32 noundef %82)
  br label %84

84:                                               ; preds = %73, %81
  %85 = add nuw nsw i32 %48, 1
  %86 = load i32, ptr @iterations, align 4, !tbaa !11
  %87 = icmp slt i32 %85, %86
  br i1 %87, label %47, label %88, !llvm.loop !119

88:                                               ; preds = %84, %43, %6
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_Z14test_quicksortI14PointerWrapperI12ValueWrapperIdEES2_EvT_S4_S4_S4_T0_PKc(ptr %0, ptr %1, ptr %2, ptr %3, [1 x double] alignstack(8) %4, ptr noundef %5) local_unnamed_addr #9 comdat {
  %7 = ptrtoint ptr %0 to i64
  %8 = ptrtoint ptr %1 to i64
  %9 = ptrtoint ptr %0 to i64
  %10 = ptrtoint ptr %2 to i64
  %11 = load i32, ptr @iterations, align 4, !tbaa !11
  %12 = icmp sgt i32 %11, 0
  br i1 %12, label %13, label %88

13:                                               ; preds = %6
  %14 = icmp eq ptr %0, %1
  br i1 %14, label %30, label %15

15:                                               ; preds = %13
  %16 = sub i64 %10, %9
  %17 = add i64 %8, -8
  %18 = sub i64 %17, %7
  %19 = lshr i64 %18, 3
  %20 = add nuw nsw i64 %19, 1
  %21 = icmp ult i64 %18, 24
  %22 = icmp ult i64 %16, 32
  %23 = select i1 %21, i1 true, i1 %22
  %24 = and i64 %20, 4611686018427387900
  %25 = shl i64 %24, 3
  %26 = getelementptr i8, ptr %0, i64 %25
  %27 = shl i64 %24, 3
  %28 = getelementptr i8, ptr %2, i64 %27
  %29 = icmp eq i64 %20, %24
  br label %47

30:                                               ; preds = %13, %43
  %31 = phi i32 [ %44, %43 ], [ 0, %13 ]
  tail call void @_ZN9benchmark9quicksortI14PointerWrapperI12ValueWrapperIdEES3_EEvT_S5_(ptr %2, ptr %3)
  br label %32

32:                                               ; preds = %36, %30
  %33 = phi ptr [ %2, %30 ], [ %34, %36 ]
  %34 = getelementptr i8, ptr %33, i64 8
  %35 = icmp eq ptr %34, %3
  br i1 %35, label %43, label %36

36:                                               ; preds = %32
  %37 = load double, ptr %34, align 8, !tbaa !51
  %38 = load double, ptr %33, align 8, !tbaa !51
  %39 = fcmp olt double %37, %38
  br i1 %39, label %40, label %32, !llvm.loop !89

40:                                               ; preds = %36
  %41 = load i32, ptr @current_test, align 4, !tbaa !11
  %42 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.51, i32 noundef %41)
  br label %43

43:                                               ; preds = %32, %40
  %44 = add nuw nsw i32 %31, 1
  %45 = load i32, ptr @iterations, align 4, !tbaa !11
  %46 = icmp slt i32 %44, %45
  br i1 %46, label %30, label %88, !llvm.loop !122

47:                                               ; preds = %15, %84
  %48 = phi i32 [ %85, %84 ], [ 0, %15 ]
  br i1 %23, label %62, label %49

49:                                               ; preds = %47, %49
  %50 = phi i64 [ %59, %49 ], [ 0, %47 ]
  %51 = shl i64 %50, 3
  %52 = getelementptr i8, ptr %0, i64 %51
  %53 = shl i64 %50, 3
  %54 = getelementptr i8, ptr %2, i64 %53
  %55 = getelementptr i8, ptr %52, i64 16
  %56 = load <2 x i64>, ptr %52, align 8, !tbaa !33
  %57 = load <2 x i64>, ptr %55, align 8, !tbaa !33
  %58 = getelementptr i8, ptr %54, i64 16
  store <2 x i64> %56, ptr %54, align 8, !tbaa !33
  store <2 x i64> %57, ptr %58, align 8, !tbaa !33
  %59 = add nuw i64 %50, 4
  %60 = icmp eq i64 %59, %24
  br i1 %60, label %61, label %49, !llvm.loop !123

61:                                               ; preds = %49
  br i1 %29, label %72, label %62

62:                                               ; preds = %47, %61
  %63 = phi ptr [ %0, %47 ], [ %26, %61 ]
  %64 = phi ptr [ %2, %47 ], [ %28, %61 ]
  br label %65

65:                                               ; preds = %62, %65
  %66 = phi ptr [ %68, %65 ], [ %63, %62 ]
  %67 = phi ptr [ %69, %65 ], [ %64, %62 ]
  %68 = getelementptr inbounds nuw i8, ptr %66, i64 8
  %69 = getelementptr inbounds nuw i8, ptr %67, i64 8
  %70 = load i64, ptr %66, align 8, !tbaa !33
  store i64 %70, ptr %67, align 8, !tbaa !33
  %71 = icmp eq ptr %68, %1
  br i1 %71, label %72, label %65, !llvm.loop !124

72:                                               ; preds = %65, %61
  tail call void @_ZN9benchmark9quicksortI14PointerWrapperI12ValueWrapperIdEES3_EEvT_S5_(ptr %2, ptr %3)
  br label %73

73:                                               ; preds = %77, %72
  %74 = phi ptr [ %2, %72 ], [ %75, %77 ]
  %75 = getelementptr i8, ptr %74, i64 8
  %76 = icmp eq ptr %75, %3
  br i1 %76, label %84, label %77

77:                                               ; preds = %73
  %78 = load double, ptr %75, align 8, !tbaa !51
  %79 = load double, ptr %74, align 8, !tbaa !51
  %80 = fcmp olt double %78, %79
  br i1 %80, label %81, label %73, !llvm.loop !89

81:                                               ; preds = %77
  %82 = load i32, ptr @current_test, align 4, !tbaa !11
  %83 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.51, i32 noundef %82)
  br label %84

84:                                               ; preds = %73, %81
  %85 = add nuw nsw i32 %48, 1
  %86 = load i32, ptr @iterations, align 4, !tbaa !11
  %87 = icmp slt i32 %85, %86
  br i1 %87, label %47, label %88, !llvm.loop !122

88:                                               ; preds = %84, %43, %6
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_Z14test_quicksortIP12ValueWrapperIS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IdEEEEEEEEEESA_EvT_SC_SC_SC_T0_PKc(ptr noundef %0, ptr noundef %1, ptr noundef %2, ptr noundef %3, [1 x double] alignstack(8) %4, ptr noundef %5) local_unnamed_addr #9 comdat {
  %7 = ptrtoint ptr %0 to i64
  %8 = ptrtoint ptr %1 to i64
  %9 = ptrtoint ptr %0 to i64
  %10 = ptrtoint ptr %2 to i64
  %11 = load i32, ptr @iterations, align 4, !tbaa !11
  %12 = icmp sgt i32 %11, 0
  br i1 %12, label %13, label %88

13:                                               ; preds = %6
  %14 = icmp eq ptr %0, %1
  br i1 %14, label %30, label %15

15:                                               ; preds = %13
  %16 = sub i64 %10, %9
  %17 = add i64 %8, -8
  %18 = sub i64 %17, %7
  %19 = lshr i64 %18, 3
  %20 = add nuw nsw i64 %19, 1
  %21 = icmp ult i64 %18, 24
  %22 = icmp ult i64 %16, 32
  %23 = or i1 %21, %22
  %24 = and i64 %20, 4611686018427387900
  %25 = shl i64 %24, 3
  %26 = getelementptr i8, ptr %2, i64 %25
  %27 = shl i64 %24, 3
  %28 = getelementptr i8, ptr %0, i64 %27
  %29 = icmp eq i64 %20, %24
  br label %47

30:                                               ; preds = %13, %43
  %31 = phi i32 [ %44, %43 ], [ 0, %13 ]
  tail call void @_ZN9benchmark9quicksortIP12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEESB_EEvT_SD_(ptr noundef %2, ptr noundef %3)
  br label %32

32:                                               ; preds = %36, %30
  %33 = phi ptr [ %2, %30 ], [ %34, %36 ]
  %34 = getelementptr i8, ptr %33, i64 8
  %35 = icmp eq ptr %34, %3
  br i1 %35, label %43, label %36

36:                                               ; preds = %32
  %37 = load double, ptr %34, align 8, !tbaa !51
  %38 = load double, ptr %33, align 8, !tbaa !51
  %39 = fcmp olt double %37, %38
  br i1 %39, label %40, label %32, !llvm.loop !97

40:                                               ; preds = %36
  %41 = load i32, ptr @current_test, align 4, !tbaa !11
  %42 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.51, i32 noundef %41)
  br label %43

43:                                               ; preds = %32, %40
  %44 = add nuw nsw i32 %31, 1
  %45 = load i32, ptr @iterations, align 4, !tbaa !11
  %46 = icmp slt i32 %44, %45
  br i1 %46, label %30, label %88, !llvm.loop !125

47:                                               ; preds = %15, %84
  %48 = phi i32 [ %85, %84 ], [ 0, %15 ]
  br i1 %23, label %62, label %49

49:                                               ; preds = %47, %49
  %50 = phi i64 [ %59, %49 ], [ 0, %47 ]
  %51 = shl i64 %50, 3
  %52 = getelementptr i8, ptr %2, i64 %51
  %53 = shl i64 %50, 3
  %54 = getelementptr i8, ptr %0, i64 %53
  %55 = getelementptr i8, ptr %54, i64 16
  %56 = load <2 x i64>, ptr %54, align 8, !tbaa !33
  %57 = load <2 x i64>, ptr %55, align 8, !tbaa !33
  %58 = getelementptr i8, ptr %52, i64 16
  store <2 x i64> %56, ptr %52, align 8, !tbaa !33
  store <2 x i64> %57, ptr %58, align 8, !tbaa !33
  %59 = add nuw i64 %50, 4
  %60 = icmp eq i64 %59, %24
  br i1 %60, label %61, label %49, !llvm.loop !126

61:                                               ; preds = %49
  br i1 %29, label %72, label %62

62:                                               ; preds = %47, %61
  %63 = phi ptr [ %2, %47 ], [ %26, %61 ]
  %64 = phi ptr [ %0, %47 ], [ %28, %61 ]
  br label %65

65:                                               ; preds = %62, %65
  %66 = phi ptr [ %69, %65 ], [ %63, %62 ]
  %67 = phi ptr [ %68, %65 ], [ %64, %62 ]
  %68 = getelementptr inbounds nuw i8, ptr %67, i64 8
  %69 = getelementptr inbounds nuw i8, ptr %66, i64 8
  %70 = load i64, ptr %67, align 8, !tbaa !33
  store i64 %70, ptr %66, align 8, !tbaa !33
  %71 = icmp eq ptr %68, %1
  br i1 %71, label %72, label %65, !llvm.loop !127

72:                                               ; preds = %65, %61
  tail call void @_ZN9benchmark9quicksortIP12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEESB_EEvT_SD_(ptr noundef %2, ptr noundef %3)
  br label %73

73:                                               ; preds = %77, %72
  %74 = phi ptr [ %2, %72 ], [ %75, %77 ]
  %75 = getelementptr i8, ptr %74, i64 8
  %76 = icmp eq ptr %75, %3
  br i1 %76, label %84, label %77

77:                                               ; preds = %73
  %78 = load double, ptr %75, align 8, !tbaa !51
  %79 = load double, ptr %74, align 8, !tbaa !51
  %80 = fcmp olt double %78, %79
  br i1 %80, label %81, label %73, !llvm.loop !97

81:                                               ; preds = %77
  %82 = load i32, ptr @current_test, align 4, !tbaa !11
  %83 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.51, i32 noundef %82)
  br label %84

84:                                               ; preds = %73, %81
  %85 = add nuw nsw i32 %48, 1
  %86 = load i32, ptr @iterations, align 4, !tbaa !11
  %87 = icmp slt i32 %85, %86
  br i1 %87, label %47, label %88, !llvm.loop !125

88:                                               ; preds = %84, %43, %6
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_Z14test_quicksortI14PointerWrapperI12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEEESB_EvT_SD_SD_SD_T0_PKc(ptr %0, ptr %1, ptr %2, ptr %3, [1 x double] alignstack(8) %4, ptr noundef %5) local_unnamed_addr #9 comdat {
  %7 = ptrtoint ptr %0 to i64
  %8 = ptrtoint ptr %1 to i64
  %9 = ptrtoint ptr %0 to i64
  %10 = ptrtoint ptr %2 to i64
  %11 = load i32, ptr @iterations, align 4, !tbaa !11
  %12 = icmp sgt i32 %11, 0
  br i1 %12, label %13, label %88

13:                                               ; preds = %6
  %14 = icmp eq ptr %0, %1
  br i1 %14, label %30, label %15

15:                                               ; preds = %13
  %16 = sub i64 %10, %9
  %17 = add i64 %8, -8
  %18 = sub i64 %17, %7
  %19 = lshr i64 %18, 3
  %20 = add nuw nsw i64 %19, 1
  %21 = icmp ult i64 %18, 24
  %22 = icmp ult i64 %16, 32
  %23 = select i1 %21, i1 true, i1 %22
  %24 = and i64 %20, 4611686018427387900
  %25 = shl i64 %24, 3
  %26 = getelementptr i8, ptr %0, i64 %25
  %27 = shl i64 %24, 3
  %28 = getelementptr i8, ptr %2, i64 %27
  %29 = icmp eq i64 %20, %24
  br label %47

30:                                               ; preds = %13, %43
  %31 = phi i32 [ %44, %43 ], [ 0, %13 ]
  tail call void @_ZN9benchmark9quicksortI14PointerWrapperI12ValueWrapperIS2_IS2_IS2_IS2_IS2_IS2_IS2_IS2_IS2_IdEEEEEEEEEEESC_EEvT_SE_(ptr %2, ptr %3)
  br label %32

32:                                               ; preds = %36, %30
  %33 = phi ptr [ %2, %30 ], [ %34, %36 ]
  %34 = getelementptr i8, ptr %33, i64 8
  %35 = icmp eq ptr %34, %3
  br i1 %35, label %43, label %36

36:                                               ; preds = %32
  %37 = load double, ptr %34, align 8, !tbaa !51
  %38 = load double, ptr %33, align 8, !tbaa !51
  %39 = fcmp olt double %37, %38
  br i1 %39, label %40, label %32, !llvm.loop !105

40:                                               ; preds = %36
  %41 = load i32, ptr @current_test, align 4, !tbaa !11
  %42 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.51, i32 noundef %41)
  br label %43

43:                                               ; preds = %32, %40
  %44 = add nuw nsw i32 %31, 1
  %45 = load i32, ptr @iterations, align 4, !tbaa !11
  %46 = icmp slt i32 %44, %45
  br i1 %46, label %30, label %88, !llvm.loop !128

47:                                               ; preds = %15, %84
  %48 = phi i32 [ %85, %84 ], [ 0, %15 ]
  br i1 %23, label %62, label %49

49:                                               ; preds = %47, %49
  %50 = phi i64 [ %59, %49 ], [ 0, %47 ]
  %51 = shl i64 %50, 3
  %52 = getelementptr i8, ptr %0, i64 %51
  %53 = shl i64 %50, 3
  %54 = getelementptr i8, ptr %2, i64 %53
  %55 = getelementptr i8, ptr %52, i64 16
  %56 = load <2 x i64>, ptr %52, align 8, !tbaa !33
  %57 = load <2 x i64>, ptr %55, align 8, !tbaa !33
  %58 = getelementptr i8, ptr %54, i64 16
  store <2 x i64> %56, ptr %54, align 8, !tbaa !33
  store <2 x i64> %57, ptr %58, align 8, !tbaa !33
  %59 = add nuw i64 %50, 4
  %60 = icmp eq i64 %59, %24
  br i1 %60, label %61, label %49, !llvm.loop !129

61:                                               ; preds = %49
  br i1 %29, label %72, label %62

62:                                               ; preds = %47, %61
  %63 = phi ptr [ %0, %47 ], [ %26, %61 ]
  %64 = phi ptr [ %2, %47 ], [ %28, %61 ]
  br label %65

65:                                               ; preds = %62, %65
  %66 = phi ptr [ %68, %65 ], [ %63, %62 ]
  %67 = phi ptr [ %69, %65 ], [ %64, %62 ]
  %68 = getelementptr inbounds nuw i8, ptr %66, i64 8
  %69 = getelementptr inbounds nuw i8, ptr %67, i64 8
  %70 = load i64, ptr %66, align 8, !tbaa !33
  store i64 %70, ptr %67, align 8, !tbaa !33
  %71 = icmp eq ptr %68, %1
  br i1 %71, label %72, label %65, !llvm.loop !130

72:                                               ; preds = %65, %61
  tail call void @_ZN9benchmark9quicksortI14PointerWrapperI12ValueWrapperIS2_IS2_IS2_IS2_IS2_IS2_IS2_IS2_IS2_IdEEEEEEEEEEESC_EEvT_SE_(ptr %2, ptr %3)
  br label %73

73:                                               ; preds = %77, %72
  %74 = phi ptr [ %2, %72 ], [ %75, %77 ]
  %75 = getelementptr i8, ptr %74, i64 8
  %76 = icmp eq ptr %75, %3
  br i1 %76, label %84, label %77

77:                                               ; preds = %73
  %78 = load double, ptr %75, align 8, !tbaa !51
  %79 = load double, ptr %74, align 8, !tbaa !51
  %80 = fcmp olt double %78, %79
  br i1 %80, label %81, label %73, !llvm.loop !105

81:                                               ; preds = %77
  %82 = load i32, ptr @current_test, align 4, !tbaa !11
  %83 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.51, i32 noundef %82)
  br label %84

84:                                               ; preds = %73, %81
  %85 = add nuw nsw i32 %48, 1
  %86 = load i32, ptr @iterations, align 4, !tbaa !11
  %87 = icmp slt i32 %85, %86
  br i1 %87, label %47, label %88, !llvm.loop !128

88:                                               ; preds = %84, %43, %6
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_Z14test_heap_sortIPddEvT_S1_S1_S1_T0_PKc(ptr noundef %0, ptr noundef %1, ptr noundef %2, ptr noundef %3, double noundef %4, ptr noundef %5) local_unnamed_addr #9 comdat {
  %7 = ptrtoint ptr %0 to i64
  %8 = ptrtoint ptr %1 to i64
  %9 = ptrtoint ptr %0 to i64
  %10 = ptrtoint ptr %2 to i64
  %11 = load i32, ptr @iterations, align 4, !tbaa !11
  %12 = icmp sgt i32 %11, 0
  br i1 %12, label %13, label %88

13:                                               ; preds = %6
  %14 = icmp eq ptr %0, %1
  br i1 %14, label %30, label %15

15:                                               ; preds = %13
  %16 = sub i64 %10, %9
  %17 = add i64 %8, -8
  %18 = sub i64 %17, %7
  %19 = lshr i64 %18, 3
  %20 = add nuw nsw i64 %19, 1
  %21 = icmp ult i64 %18, 24
  %22 = icmp ult i64 %16, 32
  %23 = or i1 %21, %22
  %24 = and i64 %20, 4611686018427387900
  %25 = shl i64 %24, 3
  %26 = getelementptr i8, ptr %2, i64 %25
  %27 = shl i64 %24, 3
  %28 = getelementptr i8, ptr %0, i64 %27
  %29 = icmp eq i64 %20, %24
  br label %47

30:                                               ; preds = %13, %43
  %31 = phi i32 [ %44, %43 ], [ 0, %13 ]
  tail call void @_ZN9benchmark8heapsortIPddEEvT_S2_(ptr noundef %2, ptr noundef %3)
  br label %32

32:                                               ; preds = %36, %30
  %33 = phi ptr [ %2, %30 ], [ %34, %36 ]
  %34 = getelementptr i8, ptr %33, i64 8
  %35 = icmp eq ptr %34, %3
  br i1 %35, label %43, label %36

36:                                               ; preds = %32
  %37 = load double, ptr %34, align 8, !tbaa !33
  %38 = load double, ptr %33, align 8, !tbaa !33
  %39 = fcmp olt double %37, %38
  br i1 %39, label %40, label %32, !llvm.loop !65

40:                                               ; preds = %36
  %41 = load i32, ptr @current_test, align 4, !tbaa !11
  %42 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.51, i32 noundef %41)
  br label %43

43:                                               ; preds = %32, %40
  %44 = add nuw nsw i32 %31, 1
  %45 = load i32, ptr @iterations, align 4, !tbaa !11
  %46 = icmp slt i32 %44, %45
  br i1 %46, label %30, label %88, !llvm.loop !131

47:                                               ; preds = %15, %84
  %48 = phi i32 [ %85, %84 ], [ 0, %15 ]
  br i1 %23, label %62, label %49

49:                                               ; preds = %47, %49
  %50 = phi i64 [ %59, %49 ], [ 0, %47 ]
  %51 = shl i64 %50, 3
  %52 = getelementptr i8, ptr %2, i64 %51
  %53 = shl i64 %50, 3
  %54 = getelementptr i8, ptr %0, i64 %53
  %55 = getelementptr i8, ptr %54, i64 16
  %56 = load <2 x double>, ptr %54, align 8, !tbaa !33
  %57 = load <2 x double>, ptr %55, align 8, !tbaa !33
  %58 = getelementptr i8, ptr %52, i64 16
  store <2 x double> %56, ptr %52, align 8, !tbaa !33
  store <2 x double> %57, ptr %58, align 8, !tbaa !33
  %59 = add nuw i64 %50, 4
  %60 = icmp eq i64 %59, %24
  br i1 %60, label %61, label %49, !llvm.loop !132

61:                                               ; preds = %49
  br i1 %29, label %72, label %62

62:                                               ; preds = %47, %61
  %63 = phi ptr [ %2, %47 ], [ %26, %61 ]
  %64 = phi ptr [ %0, %47 ], [ %28, %61 ]
  br label %65

65:                                               ; preds = %62, %65
  %66 = phi ptr [ %70, %65 ], [ %63, %62 ]
  %67 = phi ptr [ %68, %65 ], [ %64, %62 ]
  %68 = getelementptr inbounds nuw i8, ptr %67, i64 8
  %69 = load double, ptr %67, align 8, !tbaa !33
  %70 = getelementptr inbounds nuw i8, ptr %66, i64 8
  store double %69, ptr %66, align 8, !tbaa !33
  %71 = icmp eq ptr %68, %1
  br i1 %71, label %72, label %65, !llvm.loop !133

72:                                               ; preds = %65, %61
  tail call void @_ZN9benchmark8heapsortIPddEEvT_S2_(ptr noundef %2, ptr noundef %3)
  br label %73

73:                                               ; preds = %77, %72
  %74 = phi ptr [ %2, %72 ], [ %75, %77 ]
  %75 = getelementptr i8, ptr %74, i64 8
  %76 = icmp eq ptr %75, %3
  br i1 %76, label %84, label %77

77:                                               ; preds = %73
  %78 = load double, ptr %75, align 8, !tbaa !33
  %79 = load double, ptr %74, align 8, !tbaa !33
  %80 = fcmp olt double %78, %79
  br i1 %80, label %81, label %73, !llvm.loop !65

81:                                               ; preds = %77
  %82 = load i32, ptr @current_test, align 4, !tbaa !11
  %83 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.51, i32 noundef %82)
  br label %84

84:                                               ; preds = %73, %81
  %85 = add nuw nsw i32 %48, 1
  %86 = load i32, ptr @iterations, align 4, !tbaa !11
  %87 = icmp slt i32 %85, %86
  br i1 %87, label %47, label %88, !llvm.loop !131

88:                                               ; preds = %84, %43, %6
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_Z14test_heap_sortI14PointerWrapperIdEdEvT_S2_S2_S2_T0_PKc(ptr %0, ptr %1, ptr %2, ptr %3, double noundef %4, ptr noundef %5) local_unnamed_addr #9 comdat {
  %7 = ptrtoint ptr %0 to i64
  %8 = ptrtoint ptr %1 to i64
  %9 = ptrtoint ptr %0 to i64
  %10 = ptrtoint ptr %2 to i64
  %11 = load i32, ptr @iterations, align 4, !tbaa !11
  %12 = icmp sgt i32 %11, 0
  br i1 %12, label %13, label %88

13:                                               ; preds = %6
  %14 = icmp eq ptr %0, %1
  br i1 %14, label %30, label %15

15:                                               ; preds = %13
  %16 = sub i64 %10, %9
  %17 = add i64 %8, -8
  %18 = sub i64 %17, %7
  %19 = lshr i64 %18, 3
  %20 = add nuw nsw i64 %19, 1
  %21 = icmp ult i64 %18, 24
  %22 = icmp ult i64 %16, 32
  %23 = select i1 %21, i1 true, i1 %22
  %24 = and i64 %20, 4611686018427387900
  %25 = shl i64 %24, 3
  %26 = getelementptr i8, ptr %0, i64 %25
  %27 = shl i64 %24, 3
  %28 = getelementptr i8, ptr %2, i64 %27
  %29 = icmp eq i64 %20, %24
  br label %47

30:                                               ; preds = %13, %43
  %31 = phi i32 [ %44, %43 ], [ 0, %13 ]
  tail call void @_ZN9benchmark8heapsortI14PointerWrapperIdEdEEvT_S3_(ptr %2, ptr %3)
  br label %32

32:                                               ; preds = %36, %30
  %33 = phi ptr [ %2, %30 ], [ %34, %36 ]
  %34 = getelementptr i8, ptr %33, i64 8
  %35 = icmp eq ptr %34, %3
  br i1 %35, label %43, label %36

36:                                               ; preds = %32
  %37 = load double, ptr %34, align 8, !tbaa !33
  %38 = load double, ptr %33, align 8, !tbaa !33
  %39 = fcmp olt double %37, %38
  br i1 %39, label %40, label %32, !llvm.loop !73

40:                                               ; preds = %36
  %41 = load i32, ptr @current_test, align 4, !tbaa !11
  %42 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.51, i32 noundef %41)
  br label %43

43:                                               ; preds = %32, %40
  %44 = add nuw nsw i32 %31, 1
  %45 = load i32, ptr @iterations, align 4, !tbaa !11
  %46 = icmp slt i32 %44, %45
  br i1 %46, label %30, label %88, !llvm.loop !134

47:                                               ; preds = %15, %84
  %48 = phi i32 [ %85, %84 ], [ 0, %15 ]
  br i1 %23, label %62, label %49

49:                                               ; preds = %47, %49
  %50 = phi i64 [ %59, %49 ], [ 0, %47 ]
  %51 = shl i64 %50, 3
  %52 = getelementptr i8, ptr %0, i64 %51
  %53 = shl i64 %50, 3
  %54 = getelementptr i8, ptr %2, i64 %53
  %55 = getelementptr i8, ptr %52, i64 16
  %56 = load <2 x double>, ptr %52, align 8, !tbaa !33
  %57 = load <2 x double>, ptr %55, align 8, !tbaa !33
  %58 = getelementptr i8, ptr %54, i64 16
  store <2 x double> %56, ptr %54, align 8, !tbaa !33
  store <2 x double> %57, ptr %58, align 8, !tbaa !33
  %59 = add nuw i64 %50, 4
  %60 = icmp eq i64 %59, %24
  br i1 %60, label %61, label %49, !llvm.loop !135

61:                                               ; preds = %49
  br i1 %29, label %72, label %62

62:                                               ; preds = %47, %61
  %63 = phi ptr [ %0, %47 ], [ %26, %61 ]
  %64 = phi ptr [ %2, %47 ], [ %28, %61 ]
  br label %65

65:                                               ; preds = %62, %65
  %66 = phi ptr [ %68, %65 ], [ %63, %62 ]
  %67 = phi ptr [ %70, %65 ], [ %64, %62 ]
  %68 = getelementptr inbounds nuw i8, ptr %66, i64 8
  %69 = load double, ptr %66, align 8, !tbaa !33
  %70 = getelementptr inbounds nuw i8, ptr %67, i64 8
  store double %69, ptr %67, align 8, !tbaa !33
  %71 = icmp eq ptr %68, %1
  br i1 %71, label %72, label %65, !llvm.loop !136

72:                                               ; preds = %65, %61
  tail call void @_ZN9benchmark8heapsortI14PointerWrapperIdEdEEvT_S3_(ptr %2, ptr %3)
  br label %73

73:                                               ; preds = %77, %72
  %74 = phi ptr [ %2, %72 ], [ %75, %77 ]
  %75 = getelementptr i8, ptr %74, i64 8
  %76 = icmp eq ptr %75, %3
  br i1 %76, label %84, label %77

77:                                               ; preds = %73
  %78 = load double, ptr %75, align 8, !tbaa !33
  %79 = load double, ptr %74, align 8, !tbaa !33
  %80 = fcmp olt double %78, %79
  br i1 %80, label %81, label %73, !llvm.loop !73

81:                                               ; preds = %77
  %82 = load i32, ptr @current_test, align 4, !tbaa !11
  %83 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.51, i32 noundef %82)
  br label %84

84:                                               ; preds = %73, %81
  %85 = add nuw nsw i32 %48, 1
  %86 = load i32, ptr @iterations, align 4, !tbaa !11
  %87 = icmp slt i32 %85, %86
  br i1 %87, label %47, label %88, !llvm.loop !134

88:                                               ; preds = %84, %43, %6
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_Z14test_heap_sortIP12ValueWrapperIdES1_EvT_S3_S3_S3_T0_PKc(ptr noundef %0, ptr noundef %1, ptr noundef %2, ptr noundef %3, [1 x double] alignstack(8) %4, ptr noundef %5) local_unnamed_addr #9 comdat {
  %7 = ptrtoint ptr %0 to i64
  %8 = ptrtoint ptr %1 to i64
  %9 = ptrtoint ptr %0 to i64
  %10 = ptrtoint ptr %2 to i64
  %11 = load i32, ptr @iterations, align 4, !tbaa !11
  %12 = icmp sgt i32 %11, 0
  br i1 %12, label %13, label %88

13:                                               ; preds = %6
  %14 = icmp eq ptr %0, %1
  br i1 %14, label %30, label %15

15:                                               ; preds = %13
  %16 = sub i64 %10, %9
  %17 = add i64 %8, -8
  %18 = sub i64 %17, %7
  %19 = lshr i64 %18, 3
  %20 = add nuw nsw i64 %19, 1
  %21 = icmp ult i64 %18, 24
  %22 = icmp ult i64 %16, 32
  %23 = or i1 %21, %22
  %24 = and i64 %20, 4611686018427387900
  %25 = shl i64 %24, 3
  %26 = getelementptr i8, ptr %2, i64 %25
  %27 = shl i64 %24, 3
  %28 = getelementptr i8, ptr %0, i64 %27
  %29 = icmp eq i64 %20, %24
  br label %47

30:                                               ; preds = %13, %43
  %31 = phi i32 [ %44, %43 ], [ 0, %13 ]
  tail call void @_ZN9benchmark8heapsortIP12ValueWrapperIdES2_EEvT_S4_(ptr noundef %2, ptr noundef %3)
  br label %32

32:                                               ; preds = %36, %30
  %33 = phi ptr [ %2, %30 ], [ %34, %36 ]
  %34 = getelementptr i8, ptr %33, i64 8
  %35 = icmp eq ptr %34, %3
  br i1 %35, label %43, label %36

36:                                               ; preds = %32
  %37 = load double, ptr %34, align 8, !tbaa !51
  %38 = load double, ptr %33, align 8, !tbaa !51
  %39 = fcmp olt double %37, %38
  br i1 %39, label %40, label %32, !llvm.loop !81

40:                                               ; preds = %36
  %41 = load i32, ptr @current_test, align 4, !tbaa !11
  %42 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.51, i32 noundef %41)
  br label %43

43:                                               ; preds = %32, %40
  %44 = add nuw nsw i32 %31, 1
  %45 = load i32, ptr @iterations, align 4, !tbaa !11
  %46 = icmp slt i32 %44, %45
  br i1 %46, label %30, label %88, !llvm.loop !137

47:                                               ; preds = %15, %84
  %48 = phi i32 [ %85, %84 ], [ 0, %15 ]
  br i1 %23, label %62, label %49

49:                                               ; preds = %47, %49
  %50 = phi i64 [ %59, %49 ], [ 0, %47 ]
  %51 = shl i64 %50, 3
  %52 = getelementptr i8, ptr %2, i64 %51
  %53 = shl i64 %50, 3
  %54 = getelementptr i8, ptr %0, i64 %53
  %55 = getelementptr i8, ptr %54, i64 16
  %56 = load <2 x i64>, ptr %54, align 8, !tbaa !33
  %57 = load <2 x i64>, ptr %55, align 8, !tbaa !33
  %58 = getelementptr i8, ptr %52, i64 16
  store <2 x i64> %56, ptr %52, align 8, !tbaa !33
  store <2 x i64> %57, ptr %58, align 8, !tbaa !33
  %59 = add nuw i64 %50, 4
  %60 = icmp eq i64 %59, %24
  br i1 %60, label %61, label %49, !llvm.loop !138

61:                                               ; preds = %49
  br i1 %29, label %72, label %62

62:                                               ; preds = %47, %61
  %63 = phi ptr [ %2, %47 ], [ %26, %61 ]
  %64 = phi ptr [ %0, %47 ], [ %28, %61 ]
  br label %65

65:                                               ; preds = %62, %65
  %66 = phi ptr [ %69, %65 ], [ %63, %62 ]
  %67 = phi ptr [ %68, %65 ], [ %64, %62 ]
  %68 = getelementptr inbounds nuw i8, ptr %67, i64 8
  %69 = getelementptr inbounds nuw i8, ptr %66, i64 8
  %70 = load i64, ptr %67, align 8, !tbaa !33
  store i64 %70, ptr %66, align 8, !tbaa !33
  %71 = icmp eq ptr %68, %1
  br i1 %71, label %72, label %65, !llvm.loop !139

72:                                               ; preds = %65, %61
  tail call void @_ZN9benchmark8heapsortIP12ValueWrapperIdES2_EEvT_S4_(ptr noundef %2, ptr noundef %3)
  br label %73

73:                                               ; preds = %77, %72
  %74 = phi ptr [ %2, %72 ], [ %75, %77 ]
  %75 = getelementptr i8, ptr %74, i64 8
  %76 = icmp eq ptr %75, %3
  br i1 %76, label %84, label %77

77:                                               ; preds = %73
  %78 = load double, ptr %75, align 8, !tbaa !51
  %79 = load double, ptr %74, align 8, !tbaa !51
  %80 = fcmp olt double %78, %79
  br i1 %80, label %81, label %73, !llvm.loop !81

81:                                               ; preds = %77
  %82 = load i32, ptr @current_test, align 4, !tbaa !11
  %83 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.51, i32 noundef %82)
  br label %84

84:                                               ; preds = %73, %81
  %85 = add nuw nsw i32 %48, 1
  %86 = load i32, ptr @iterations, align 4, !tbaa !11
  %87 = icmp slt i32 %85, %86
  br i1 %87, label %47, label %88, !llvm.loop !137

88:                                               ; preds = %84, %43, %6
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_Z14test_heap_sortI14PointerWrapperI12ValueWrapperIdEES2_EvT_S4_S4_S4_T0_PKc(ptr %0, ptr %1, ptr %2, ptr %3, [1 x double] alignstack(8) %4, ptr noundef %5) local_unnamed_addr #9 comdat {
  %7 = ptrtoint ptr %0 to i64
  %8 = ptrtoint ptr %1 to i64
  %9 = ptrtoint ptr %0 to i64
  %10 = ptrtoint ptr %2 to i64
  %11 = load i32, ptr @iterations, align 4, !tbaa !11
  %12 = icmp sgt i32 %11, 0
  br i1 %12, label %13, label %88

13:                                               ; preds = %6
  %14 = icmp eq ptr %0, %1
  br i1 %14, label %30, label %15

15:                                               ; preds = %13
  %16 = sub i64 %10, %9
  %17 = add i64 %8, -8
  %18 = sub i64 %17, %7
  %19 = lshr i64 %18, 3
  %20 = add nuw nsw i64 %19, 1
  %21 = icmp ult i64 %18, 24
  %22 = icmp ult i64 %16, 32
  %23 = select i1 %21, i1 true, i1 %22
  %24 = and i64 %20, 4611686018427387900
  %25 = shl i64 %24, 3
  %26 = getelementptr i8, ptr %0, i64 %25
  %27 = shl i64 %24, 3
  %28 = getelementptr i8, ptr %2, i64 %27
  %29 = icmp eq i64 %20, %24
  br label %47

30:                                               ; preds = %13, %43
  %31 = phi i32 [ %44, %43 ], [ 0, %13 ]
  tail call void @_ZN9benchmark8heapsortI14PointerWrapperI12ValueWrapperIdEES3_EEvT_S5_(ptr %2, ptr %3)
  br label %32

32:                                               ; preds = %36, %30
  %33 = phi ptr [ %2, %30 ], [ %34, %36 ]
  %34 = getelementptr i8, ptr %33, i64 8
  %35 = icmp eq ptr %34, %3
  br i1 %35, label %43, label %36

36:                                               ; preds = %32
  %37 = load double, ptr %34, align 8, !tbaa !51
  %38 = load double, ptr %33, align 8, !tbaa !51
  %39 = fcmp olt double %37, %38
  br i1 %39, label %40, label %32, !llvm.loop !89

40:                                               ; preds = %36
  %41 = load i32, ptr @current_test, align 4, !tbaa !11
  %42 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.51, i32 noundef %41)
  br label %43

43:                                               ; preds = %32, %40
  %44 = add nuw nsw i32 %31, 1
  %45 = load i32, ptr @iterations, align 4, !tbaa !11
  %46 = icmp slt i32 %44, %45
  br i1 %46, label %30, label %88, !llvm.loop !140

47:                                               ; preds = %15, %84
  %48 = phi i32 [ %85, %84 ], [ 0, %15 ]
  br i1 %23, label %62, label %49

49:                                               ; preds = %47, %49
  %50 = phi i64 [ %59, %49 ], [ 0, %47 ]
  %51 = shl i64 %50, 3
  %52 = getelementptr i8, ptr %0, i64 %51
  %53 = shl i64 %50, 3
  %54 = getelementptr i8, ptr %2, i64 %53
  %55 = getelementptr i8, ptr %52, i64 16
  %56 = load <2 x i64>, ptr %52, align 8, !tbaa !33
  %57 = load <2 x i64>, ptr %55, align 8, !tbaa !33
  %58 = getelementptr i8, ptr %54, i64 16
  store <2 x i64> %56, ptr %54, align 8, !tbaa !33
  store <2 x i64> %57, ptr %58, align 8, !tbaa !33
  %59 = add nuw i64 %50, 4
  %60 = icmp eq i64 %59, %24
  br i1 %60, label %61, label %49, !llvm.loop !141

61:                                               ; preds = %49
  br i1 %29, label %72, label %62

62:                                               ; preds = %47, %61
  %63 = phi ptr [ %0, %47 ], [ %26, %61 ]
  %64 = phi ptr [ %2, %47 ], [ %28, %61 ]
  br label %65

65:                                               ; preds = %62, %65
  %66 = phi ptr [ %68, %65 ], [ %63, %62 ]
  %67 = phi ptr [ %69, %65 ], [ %64, %62 ]
  %68 = getelementptr inbounds nuw i8, ptr %66, i64 8
  %69 = getelementptr inbounds nuw i8, ptr %67, i64 8
  %70 = load i64, ptr %66, align 8, !tbaa !33
  store i64 %70, ptr %67, align 8, !tbaa !33
  %71 = icmp eq ptr %68, %1
  br i1 %71, label %72, label %65, !llvm.loop !142

72:                                               ; preds = %65, %61
  tail call void @_ZN9benchmark8heapsortI14PointerWrapperI12ValueWrapperIdEES3_EEvT_S5_(ptr %2, ptr %3)
  br label %73

73:                                               ; preds = %77, %72
  %74 = phi ptr [ %2, %72 ], [ %75, %77 ]
  %75 = getelementptr i8, ptr %74, i64 8
  %76 = icmp eq ptr %75, %3
  br i1 %76, label %84, label %77

77:                                               ; preds = %73
  %78 = load double, ptr %75, align 8, !tbaa !51
  %79 = load double, ptr %74, align 8, !tbaa !51
  %80 = fcmp olt double %78, %79
  br i1 %80, label %81, label %73, !llvm.loop !89

81:                                               ; preds = %77
  %82 = load i32, ptr @current_test, align 4, !tbaa !11
  %83 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.51, i32 noundef %82)
  br label %84

84:                                               ; preds = %73, %81
  %85 = add nuw nsw i32 %48, 1
  %86 = load i32, ptr @iterations, align 4, !tbaa !11
  %87 = icmp slt i32 %85, %86
  br i1 %87, label %47, label %88, !llvm.loop !140

88:                                               ; preds = %84, %43, %6
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_Z14test_heap_sortIP12ValueWrapperIS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IdEEEEEEEEEESA_EvT_SC_SC_SC_T0_PKc(ptr noundef %0, ptr noundef %1, ptr noundef %2, ptr noundef %3, [1 x double] alignstack(8) %4, ptr noundef %5) local_unnamed_addr #9 comdat {
  %7 = ptrtoint ptr %0 to i64
  %8 = ptrtoint ptr %1 to i64
  %9 = ptrtoint ptr %0 to i64
  %10 = ptrtoint ptr %2 to i64
  %11 = load i32, ptr @iterations, align 4, !tbaa !11
  %12 = icmp sgt i32 %11, 0
  br i1 %12, label %13, label %88

13:                                               ; preds = %6
  %14 = icmp eq ptr %0, %1
  br i1 %14, label %30, label %15

15:                                               ; preds = %13
  %16 = sub i64 %10, %9
  %17 = add i64 %8, -8
  %18 = sub i64 %17, %7
  %19 = lshr i64 %18, 3
  %20 = add nuw nsw i64 %19, 1
  %21 = icmp ult i64 %18, 24
  %22 = icmp ult i64 %16, 32
  %23 = or i1 %21, %22
  %24 = and i64 %20, 4611686018427387900
  %25 = shl i64 %24, 3
  %26 = getelementptr i8, ptr %2, i64 %25
  %27 = shl i64 %24, 3
  %28 = getelementptr i8, ptr %0, i64 %27
  %29 = icmp eq i64 %20, %24
  br label %47

30:                                               ; preds = %13, %43
  %31 = phi i32 [ %44, %43 ], [ 0, %13 ]
  tail call void @_ZN9benchmark8heapsortIP12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEESB_EEvT_SD_(ptr noundef %2, ptr noundef %3)
  br label %32

32:                                               ; preds = %36, %30
  %33 = phi ptr [ %2, %30 ], [ %34, %36 ]
  %34 = getelementptr i8, ptr %33, i64 8
  %35 = icmp eq ptr %34, %3
  br i1 %35, label %43, label %36

36:                                               ; preds = %32
  %37 = load double, ptr %34, align 8, !tbaa !51
  %38 = load double, ptr %33, align 8, !tbaa !51
  %39 = fcmp olt double %37, %38
  br i1 %39, label %40, label %32, !llvm.loop !97

40:                                               ; preds = %36
  %41 = load i32, ptr @current_test, align 4, !tbaa !11
  %42 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.51, i32 noundef %41)
  br label %43

43:                                               ; preds = %32, %40
  %44 = add nuw nsw i32 %31, 1
  %45 = load i32, ptr @iterations, align 4, !tbaa !11
  %46 = icmp slt i32 %44, %45
  br i1 %46, label %30, label %88, !llvm.loop !143

47:                                               ; preds = %15, %84
  %48 = phi i32 [ %85, %84 ], [ 0, %15 ]
  br i1 %23, label %62, label %49

49:                                               ; preds = %47, %49
  %50 = phi i64 [ %59, %49 ], [ 0, %47 ]
  %51 = shl i64 %50, 3
  %52 = getelementptr i8, ptr %2, i64 %51
  %53 = shl i64 %50, 3
  %54 = getelementptr i8, ptr %0, i64 %53
  %55 = getelementptr i8, ptr %54, i64 16
  %56 = load <2 x i64>, ptr %54, align 8, !tbaa !33
  %57 = load <2 x i64>, ptr %55, align 8, !tbaa !33
  %58 = getelementptr i8, ptr %52, i64 16
  store <2 x i64> %56, ptr %52, align 8, !tbaa !33
  store <2 x i64> %57, ptr %58, align 8, !tbaa !33
  %59 = add nuw i64 %50, 4
  %60 = icmp eq i64 %59, %24
  br i1 %60, label %61, label %49, !llvm.loop !144

61:                                               ; preds = %49
  br i1 %29, label %72, label %62

62:                                               ; preds = %47, %61
  %63 = phi ptr [ %2, %47 ], [ %26, %61 ]
  %64 = phi ptr [ %0, %47 ], [ %28, %61 ]
  br label %65

65:                                               ; preds = %62, %65
  %66 = phi ptr [ %69, %65 ], [ %63, %62 ]
  %67 = phi ptr [ %68, %65 ], [ %64, %62 ]
  %68 = getelementptr inbounds nuw i8, ptr %67, i64 8
  %69 = getelementptr inbounds nuw i8, ptr %66, i64 8
  %70 = load i64, ptr %67, align 8, !tbaa !33
  store i64 %70, ptr %66, align 8, !tbaa !33
  %71 = icmp eq ptr %68, %1
  br i1 %71, label %72, label %65, !llvm.loop !145

72:                                               ; preds = %65, %61
  tail call void @_ZN9benchmark8heapsortIP12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEESB_EEvT_SD_(ptr noundef %2, ptr noundef %3)
  br label %73

73:                                               ; preds = %77, %72
  %74 = phi ptr [ %2, %72 ], [ %75, %77 ]
  %75 = getelementptr i8, ptr %74, i64 8
  %76 = icmp eq ptr %75, %3
  br i1 %76, label %84, label %77

77:                                               ; preds = %73
  %78 = load double, ptr %75, align 8, !tbaa !51
  %79 = load double, ptr %74, align 8, !tbaa !51
  %80 = fcmp olt double %78, %79
  br i1 %80, label %81, label %73, !llvm.loop !97

81:                                               ; preds = %77
  %82 = load i32, ptr @current_test, align 4, !tbaa !11
  %83 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.51, i32 noundef %82)
  br label %84

84:                                               ; preds = %73, %81
  %85 = add nuw nsw i32 %48, 1
  %86 = load i32, ptr @iterations, align 4, !tbaa !11
  %87 = icmp slt i32 %85, %86
  br i1 %87, label %47, label %88, !llvm.loop !143

88:                                               ; preds = %84, %43, %6
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_Z14test_heap_sortI14PointerWrapperI12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEEESB_EvT_SD_SD_SD_T0_PKc(ptr %0, ptr %1, ptr %2, ptr %3, [1 x double] alignstack(8) %4, ptr noundef %5) local_unnamed_addr #9 comdat {
  %7 = ptrtoint ptr %0 to i64
  %8 = ptrtoint ptr %1 to i64
  %9 = ptrtoint ptr %0 to i64
  %10 = ptrtoint ptr %2 to i64
  %11 = load i32, ptr @iterations, align 4, !tbaa !11
  %12 = icmp sgt i32 %11, 0
  br i1 %12, label %13, label %88

13:                                               ; preds = %6
  %14 = icmp eq ptr %0, %1
  br i1 %14, label %30, label %15

15:                                               ; preds = %13
  %16 = sub i64 %10, %9
  %17 = add i64 %8, -8
  %18 = sub i64 %17, %7
  %19 = lshr i64 %18, 3
  %20 = add nuw nsw i64 %19, 1
  %21 = icmp ult i64 %18, 24
  %22 = icmp ult i64 %16, 32
  %23 = select i1 %21, i1 true, i1 %22
  %24 = and i64 %20, 4611686018427387900
  %25 = shl i64 %24, 3
  %26 = getelementptr i8, ptr %0, i64 %25
  %27 = shl i64 %24, 3
  %28 = getelementptr i8, ptr %2, i64 %27
  %29 = icmp eq i64 %20, %24
  br label %47

30:                                               ; preds = %13, %43
  %31 = phi i32 [ %44, %43 ], [ 0, %13 ]
  tail call void @_ZN9benchmark8heapsortI14PointerWrapperI12ValueWrapperIS2_IS2_IS2_IS2_IS2_IS2_IS2_IS2_IS2_IdEEEEEEEEEEESC_EEvT_SE_(ptr %2, ptr %3)
  br label %32

32:                                               ; preds = %36, %30
  %33 = phi ptr [ %2, %30 ], [ %34, %36 ]
  %34 = getelementptr i8, ptr %33, i64 8
  %35 = icmp eq ptr %34, %3
  br i1 %35, label %43, label %36

36:                                               ; preds = %32
  %37 = load double, ptr %34, align 8, !tbaa !51
  %38 = load double, ptr %33, align 8, !tbaa !51
  %39 = fcmp olt double %37, %38
  br i1 %39, label %40, label %32, !llvm.loop !105

40:                                               ; preds = %36
  %41 = load i32, ptr @current_test, align 4, !tbaa !11
  %42 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.51, i32 noundef %41)
  br label %43

43:                                               ; preds = %32, %40
  %44 = add nuw nsw i32 %31, 1
  %45 = load i32, ptr @iterations, align 4, !tbaa !11
  %46 = icmp slt i32 %44, %45
  br i1 %46, label %30, label %88, !llvm.loop !146

47:                                               ; preds = %15, %84
  %48 = phi i32 [ %85, %84 ], [ 0, %15 ]
  br i1 %23, label %62, label %49

49:                                               ; preds = %47, %49
  %50 = phi i64 [ %59, %49 ], [ 0, %47 ]
  %51 = shl i64 %50, 3
  %52 = getelementptr i8, ptr %0, i64 %51
  %53 = shl i64 %50, 3
  %54 = getelementptr i8, ptr %2, i64 %53
  %55 = getelementptr i8, ptr %52, i64 16
  %56 = load <2 x i64>, ptr %52, align 8, !tbaa !33
  %57 = load <2 x i64>, ptr %55, align 8, !tbaa !33
  %58 = getelementptr i8, ptr %54, i64 16
  store <2 x i64> %56, ptr %54, align 8, !tbaa !33
  store <2 x i64> %57, ptr %58, align 8, !tbaa !33
  %59 = add nuw i64 %50, 4
  %60 = icmp eq i64 %59, %24
  br i1 %60, label %61, label %49, !llvm.loop !147

61:                                               ; preds = %49
  br i1 %29, label %72, label %62

62:                                               ; preds = %47, %61
  %63 = phi ptr [ %0, %47 ], [ %26, %61 ]
  %64 = phi ptr [ %2, %47 ], [ %28, %61 ]
  br label %65

65:                                               ; preds = %62, %65
  %66 = phi ptr [ %68, %65 ], [ %63, %62 ]
  %67 = phi ptr [ %69, %65 ], [ %64, %62 ]
  %68 = getelementptr inbounds nuw i8, ptr %66, i64 8
  %69 = getelementptr inbounds nuw i8, ptr %67, i64 8
  %70 = load i64, ptr %66, align 8, !tbaa !33
  store i64 %70, ptr %67, align 8, !tbaa !33
  %71 = icmp eq ptr %68, %1
  br i1 %71, label %72, label %65, !llvm.loop !148

72:                                               ; preds = %65, %61
  tail call void @_ZN9benchmark8heapsortI14PointerWrapperI12ValueWrapperIS2_IS2_IS2_IS2_IS2_IS2_IS2_IS2_IS2_IdEEEEEEEEEEESC_EEvT_SE_(ptr %2, ptr %3)
  br label %73

73:                                               ; preds = %77, %72
  %74 = phi ptr [ %2, %72 ], [ %75, %77 ]
  %75 = getelementptr i8, ptr %74, i64 8
  %76 = icmp eq ptr %75, %3
  br i1 %76, label %84, label %77

77:                                               ; preds = %73
  %78 = load double, ptr %75, align 8, !tbaa !51
  %79 = load double, ptr %74, align 8, !tbaa !51
  %80 = fcmp olt double %78, %79
  br i1 %80, label %81, label %73, !llvm.loop !105

81:                                               ; preds = %77
  %82 = load i32, ptr @current_test, align 4, !tbaa !11
  %83 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.51, i32 noundef %82)
  br label %84

84:                                               ; preds = %73, %81
  %85 = add nuw nsw i32 %48, 1
  %86 = load i32, ptr @iterations, align 4, !tbaa !11
  %87 = icmp slt i32 %85, %86
  br i1 %87, label %47, label %88, !llvm.loop !146

88:                                               ; preds = %84, %43, %6
  ret void
}

; Function Attrs: nounwind
declare i64 @__isoc23_strtol(ptr noundef, ptr noundef, i32 noundef) local_unnamed_addr #7

; Function Attrs: mustprogress nocallback nofree nounwind willreturn
declare double @strtod(ptr noundef readonly, ptr noundef captures(none)) local_unnamed_addr #10

; Function Attrs: nounwind
declare i32 @rand() local_unnamed_addr #7

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZN9benchmark9quicksortIPddEEvT_S2_(ptr noundef %0, ptr noundef %1) local_unnamed_addr #9 comdat {
  %3 = ptrtoint ptr %1 to i64
  %4 = ptrtoint ptr %0 to i64
  %5 = sub i64 %3, %4
  %6 = icmp sgt i64 %5, 8
  br i1 %6, label %7, label %32

7:                                                ; preds = %2, %28
  %8 = phi ptr [ %14, %28 ], [ %0, %2 ]
  %9 = load double, ptr %8, align 8, !tbaa !33
  br label %10

10:                                               ; preds = %27, %7
  %11 = phi ptr [ %1, %7 ], [ %15, %27 ]
  %12 = phi ptr [ %8, %7 ], [ %21, %27 ]
  br label %13

13:                                               ; preds = %13, %10
  %14 = phi ptr [ %11, %10 ], [ %15, %13 ]
  %15 = getelementptr inbounds i8, ptr %14, i64 -8
  %16 = load double, ptr %15, align 8, !tbaa !33
  %17 = fcmp olt double %9, %16
  br i1 %17, label %13, label %18, !llvm.loop !149

18:                                               ; preds = %13
  %19 = icmp ult ptr %12, %15
  br i1 %19, label %20, label %28

20:                                               ; preds = %18, %20
  %21 = phi ptr [ %24, %20 ], [ %12, %18 ]
  %22 = load double, ptr %21, align 8, !tbaa !33
  %23 = fcmp olt double %22, %9
  %24 = getelementptr inbounds nuw i8, ptr %21, i64 8
  br i1 %23, label %20, label %25, !llvm.loop !150

25:                                               ; preds = %20
  %26 = icmp ult ptr %21, %15
  br i1 %26, label %27, label %28

27:                                               ; preds = %25
  store double %22, ptr %15, align 8, !tbaa !33
  store double %16, ptr %21, align 8, !tbaa !33
  br label %10, !llvm.loop !151

28:                                               ; preds = %25, %18
  tail call void @_ZN9benchmark9quicksortIPddEEvT_S2_(ptr noundef nonnull %8, ptr noundef nonnull %14)
  %29 = ptrtoint ptr %14 to i64
  %30 = sub i64 %3, %29
  %31 = icmp sgt i64 %30, 8
  br i1 %31, label %7, label %32

32:                                               ; preds = %28, %2
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZN9benchmark9quicksortI14PointerWrapperIdEdEEvT_S3_(ptr %0, ptr %1) local_unnamed_addr #9 comdat {
  %3 = ptrtoint ptr %1 to i64
  %4 = ptrtoint ptr %0 to i64
  %5 = sub i64 %3, %4
  %6 = icmp sgt i64 %5, 8
  br i1 %6, label %7, label %32

7:                                                ; preds = %2, %28
  %8 = phi ptr [ %14, %28 ], [ %0, %2 ]
  %9 = load double, ptr %8, align 8, !tbaa !33
  br label %10

10:                                               ; preds = %27, %7
  %11 = phi ptr [ %8, %7 ], [ %21, %27 ]
  %12 = phi ptr [ %1, %7 ], [ %15, %27 ]
  br label %13

13:                                               ; preds = %13, %10
  %14 = phi ptr [ %12, %10 ], [ %15, %13 ]
  %15 = getelementptr inbounds i8, ptr %14, i64 -8
  %16 = load double, ptr %15, align 8, !tbaa !33
  %17 = fcmp olt double %9, %16
  br i1 %17, label %13, label %18, !llvm.loop !152

18:                                               ; preds = %13
  %19 = icmp ult ptr %11, %15
  br i1 %19, label %20, label %28

20:                                               ; preds = %18, %20
  %21 = phi ptr [ %24, %20 ], [ %11, %18 ]
  %22 = load double, ptr %21, align 8, !tbaa !33
  %23 = fcmp olt double %22, %9
  %24 = getelementptr inbounds nuw i8, ptr %21, i64 8
  br i1 %23, label %20, label %25, !llvm.loop !153

25:                                               ; preds = %20
  %26 = icmp ult ptr %21, %15
  br i1 %26, label %27, label %28

27:                                               ; preds = %25
  store double %22, ptr %15, align 8, !tbaa !33
  store double %16, ptr %21, align 8, !tbaa !33
  br label %10, !llvm.loop !154

28:                                               ; preds = %25, %18
  tail call void @_ZN9benchmark9quicksortI14PointerWrapperIdEdEEvT_S3_(ptr nonnull %8, ptr nonnull %14)
  %29 = ptrtoint ptr %14 to i64
  %30 = sub i64 %3, %29
  %31 = icmp sgt i64 %30, 8
  br i1 %31, label %7, label %32

32:                                               ; preds = %28, %2
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZN9benchmark9quicksortIP12ValueWrapperIdES2_EEvT_S4_(ptr noundef %0, ptr noundef %1) local_unnamed_addr #9 comdat {
  %3 = ptrtoint ptr %1 to i64
  %4 = ptrtoint ptr %0 to i64
  %5 = sub i64 %3, %4
  %6 = icmp sgt i64 %5, 8
  br i1 %6, label %7, label %32

7:                                                ; preds = %2, %28
  %8 = phi ptr [ %14, %28 ], [ %0, %2 ]
  %9 = load double, ptr %8, align 8, !tbaa !33
  br label %10

10:                                               ; preds = %27, %7
  %11 = phi ptr [ %1, %7 ], [ %15, %27 ]
  %12 = phi ptr [ %8, %7 ], [ %21, %27 ]
  br label %13

13:                                               ; preds = %13, %10
  %14 = phi ptr [ %11, %10 ], [ %15, %13 ]
  %15 = getelementptr inbounds i8, ptr %14, i64 -8
  %16 = load double, ptr %15, align 8, !tbaa !51
  %17 = fcmp olt double %9, %16
  br i1 %17, label %13, label %18, !llvm.loop !155

18:                                               ; preds = %13
  %19 = icmp ult ptr %12, %15
  br i1 %19, label %20, label %28

20:                                               ; preds = %18, %20
  %21 = phi ptr [ %24, %20 ], [ %12, %18 ]
  %22 = load double, ptr %21, align 8
  %23 = fcmp olt double %22, %9
  %24 = getelementptr inbounds nuw i8, ptr %21, i64 8
  br i1 %23, label %20, label %25, !llvm.loop !156

25:                                               ; preds = %20
  %26 = icmp ult ptr %21, %15
  br i1 %26, label %27, label %28

27:                                               ; preds = %25
  store double %22, ptr %15, align 8, !tbaa !33
  store double %16, ptr %21, align 8, !tbaa !33
  br label %10, !llvm.loop !157

28:                                               ; preds = %25, %18
  tail call void @_ZN9benchmark9quicksortIP12ValueWrapperIdES2_EEvT_S4_(ptr noundef nonnull %8, ptr noundef nonnull %14)
  %29 = ptrtoint ptr %14 to i64
  %30 = sub i64 %3, %29
  %31 = icmp sgt i64 %30, 8
  br i1 %31, label %7, label %32

32:                                               ; preds = %28, %2
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZN9benchmark9quicksortI14PointerWrapperI12ValueWrapperIdEES3_EEvT_S5_(ptr %0, ptr %1) local_unnamed_addr #9 comdat {
  %3 = ptrtoint ptr %1 to i64
  %4 = ptrtoint ptr %0 to i64
  %5 = sub i64 %3, %4
  %6 = icmp sgt i64 %5, 8
  br i1 %6, label %7, label %32

7:                                                ; preds = %2, %28
  %8 = phi ptr [ %14, %28 ], [ %0, %2 ]
  %9 = load double, ptr %8, align 8, !tbaa !33
  br label %10

10:                                               ; preds = %27, %7
  %11 = phi ptr [ %8, %7 ], [ %21, %27 ]
  %12 = phi ptr [ %1, %7 ], [ %15, %27 ]
  br label %13

13:                                               ; preds = %13, %10
  %14 = phi ptr [ %12, %10 ], [ %15, %13 ]
  %15 = getelementptr inbounds i8, ptr %14, i64 -8
  %16 = load double, ptr %15, align 8, !tbaa !51
  %17 = fcmp olt double %9, %16
  br i1 %17, label %13, label %18, !llvm.loop !158

18:                                               ; preds = %13
  %19 = icmp ult ptr %11, %15
  br i1 %19, label %20, label %28

20:                                               ; preds = %18, %20
  %21 = phi ptr [ %24, %20 ], [ %11, %18 ]
  %22 = load double, ptr %21, align 8
  %23 = fcmp olt double %22, %9
  %24 = getelementptr inbounds nuw i8, ptr %21, i64 8
  br i1 %23, label %20, label %25, !llvm.loop !159

25:                                               ; preds = %20
  %26 = icmp ult ptr %21, %15
  br i1 %26, label %27, label %28

27:                                               ; preds = %25
  store double %22, ptr %15, align 8, !tbaa !33
  store double %16, ptr %21, align 8, !tbaa !33
  br label %10, !llvm.loop !160

28:                                               ; preds = %25, %18
  tail call void @_ZN9benchmark9quicksortI14PointerWrapperI12ValueWrapperIdEES3_EEvT_S5_(ptr nonnull %8, ptr nonnull %14)
  %29 = ptrtoint ptr %14 to i64
  %30 = sub i64 %3, %29
  %31 = icmp sgt i64 %30, 8
  br i1 %31, label %7, label %32

32:                                               ; preds = %28, %2
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZN9benchmark9quicksortIP12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEESB_EEvT_SD_(ptr noundef %0, ptr noundef %1) local_unnamed_addr #9 comdat {
  %3 = ptrtoint ptr %1 to i64
  %4 = ptrtoint ptr %0 to i64
  %5 = sub i64 %3, %4
  %6 = icmp sgt i64 %5, 8
  br i1 %6, label %7, label %32

7:                                                ; preds = %2, %28
  %8 = phi ptr [ %14, %28 ], [ %0, %2 ]
  %9 = load double, ptr %8, align 8, !tbaa !33
  br label %10

10:                                               ; preds = %27, %7
  %11 = phi ptr [ %1, %7 ], [ %15, %27 ]
  %12 = phi ptr [ %8, %7 ], [ %21, %27 ]
  br label %13

13:                                               ; preds = %13, %10
  %14 = phi ptr [ %11, %10 ], [ %15, %13 ]
  %15 = getelementptr inbounds i8, ptr %14, i64 -8
  %16 = load double, ptr %15, align 8, !tbaa !51
  %17 = fcmp olt double %9, %16
  br i1 %17, label %13, label %18, !llvm.loop !161

18:                                               ; preds = %13
  %19 = icmp ult ptr %12, %15
  br i1 %19, label %20, label %28

20:                                               ; preds = %18, %20
  %21 = phi ptr [ %24, %20 ], [ %12, %18 ]
  %22 = load double, ptr %21, align 8
  %23 = fcmp olt double %22, %9
  %24 = getelementptr inbounds nuw i8, ptr %21, i64 8
  br i1 %23, label %20, label %25, !llvm.loop !162

25:                                               ; preds = %20
  %26 = icmp ult ptr %21, %15
  br i1 %26, label %27, label %28

27:                                               ; preds = %25
  store double %22, ptr %15, align 8, !tbaa !33
  store double %16, ptr %21, align 8, !tbaa !33
  br label %10, !llvm.loop !163

28:                                               ; preds = %25, %18
  tail call void @_ZN9benchmark9quicksortIP12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEESB_EEvT_SD_(ptr noundef nonnull %8, ptr noundef nonnull %14)
  %29 = ptrtoint ptr %14 to i64
  %30 = sub i64 %3, %29
  %31 = icmp sgt i64 %30, 8
  br i1 %31, label %7, label %32

32:                                               ; preds = %28, %2
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZN9benchmark9quicksortI14PointerWrapperI12ValueWrapperIS2_IS2_IS2_IS2_IS2_IS2_IS2_IS2_IS2_IdEEEEEEEEEEESC_EEvT_SE_(ptr %0, ptr %1) local_unnamed_addr #9 comdat {
  %3 = ptrtoint ptr %1 to i64
  %4 = ptrtoint ptr %0 to i64
  %5 = sub i64 %3, %4
  %6 = icmp sgt i64 %5, 8
  br i1 %6, label %7, label %32

7:                                                ; preds = %2, %28
  %8 = phi ptr [ %14, %28 ], [ %0, %2 ]
  %9 = load double, ptr %8, align 8, !tbaa !33
  br label %10

10:                                               ; preds = %27, %7
  %11 = phi ptr [ %8, %7 ], [ %21, %27 ]
  %12 = phi ptr [ %1, %7 ], [ %15, %27 ]
  br label %13

13:                                               ; preds = %13, %10
  %14 = phi ptr [ %12, %10 ], [ %15, %13 ]
  %15 = getelementptr inbounds i8, ptr %14, i64 -8
  %16 = load double, ptr %15, align 8, !tbaa !51
  %17 = fcmp olt double %9, %16
  br i1 %17, label %13, label %18, !llvm.loop !164

18:                                               ; preds = %13
  %19 = icmp ult ptr %11, %15
  br i1 %19, label %20, label %28

20:                                               ; preds = %18, %20
  %21 = phi ptr [ %24, %20 ], [ %11, %18 ]
  %22 = load double, ptr %21, align 8
  %23 = fcmp olt double %22, %9
  %24 = getelementptr inbounds nuw i8, ptr %21, i64 8
  br i1 %23, label %20, label %25, !llvm.loop !165

25:                                               ; preds = %20
  %26 = icmp ult ptr %21, %15
  br i1 %26, label %27, label %28

27:                                               ; preds = %25
  store double %22, ptr %15, align 8, !tbaa !33
  store double %16, ptr %21, align 8, !tbaa !33
  br label %10, !llvm.loop !166

28:                                               ; preds = %25, %18
  tail call void @_ZN9benchmark9quicksortI14PointerWrapperI12ValueWrapperIS2_IS2_IS2_IS2_IS2_IS2_IS2_IS2_IS2_IdEEEEEEEEEEESC_EEvT_SE_(ptr nonnull %8, ptr nonnull %14)
  %29 = ptrtoint ptr %14 to i64
  %30 = sub i64 %3, %29
  %31 = icmp sgt i64 %30, 8
  br i1 %31, label %7, label %32

32:                                               ; preds = %28, %2
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZN9benchmark8heapsortIPddEEvT_S2_(ptr noundef %0, ptr noundef %1) local_unnamed_addr #9 comdat {
  %3 = ptrtoint ptr %1 to i64
  %4 = ptrtoint ptr %0 to i64
  %5 = sub i64 %3, %4
  %6 = ashr exact i64 %5, 3
  %7 = icmp sgt i64 %6, 1
  br i1 %7, label %8, label %110

8:                                                ; preds = %2
  %9 = lshr i64 %6, 1
  %10 = add nsw i64 %6, -1
  %11 = getelementptr inbounds nuw double, ptr %0, i64 %10
  br label %12

12:                                               ; preds = %8, %56
  %13 = phi i64 [ %9, %8 ], [ %14, %56 ]
  %14 = add nsw i64 %13, -1
  %15 = getelementptr inbounds nuw double, ptr %0, i64 %14
  %16 = load double, ptr %15, align 8, !tbaa !33
  %17 = shl nuw i64 %14, 1
  %18 = add nuw nsw i64 %17, 2
  %19 = icmp slt i64 %18, %6
  br i1 %19, label %20, label %36

20:                                               ; preds = %12, %20
  %21 = phi i64 [ %30, %20 ], [ %14, %12 ]
  %22 = phi i64 [ %34, %20 ], [ %18, %12 ]
  %23 = getelementptr double, ptr %0, i64 %22
  %24 = getelementptr i8, ptr %23, i64 -8
  %25 = load double, ptr %24, align 8, !tbaa !33
  %26 = load double, ptr %23, align 8, !tbaa !33
  %27 = fcmp olt double %25, %26
  %28 = zext i1 %27 to i64
  %29 = add nsw i64 %22, %28
  %30 = add nsw i64 %29, -1
  %31 = getelementptr inbounds double, ptr %0, i64 %30
  %32 = load double, ptr %31, align 8, !tbaa !33
  %33 = getelementptr inbounds double, ptr %0, i64 %21
  store double %32, ptr %33, align 8, !tbaa !33
  %34 = shl nsw i64 %29, 1
  %35 = icmp slt i64 %34, %6
  br i1 %35, label %20, label %36, !llvm.loop !167

36:                                               ; preds = %20, %12
  %37 = phi i64 [ %18, %12 ], [ %34, %20 ]
  %38 = phi i64 [ %14, %12 ], [ %30, %20 ]
  %39 = icmp eq i64 %37, %6
  br i1 %39, label %40, label %43

40:                                               ; preds = %36
  %41 = load double, ptr %11, align 8, !tbaa !33
  %42 = getelementptr inbounds double, ptr %0, i64 %38
  store double %41, ptr %42, align 8, !tbaa !33
  br label %43

43:                                               ; preds = %40, %36
  %44 = phi i64 [ %10, %40 ], [ %38, %36 ]
  %45 = icmp slt i64 %44, %13
  br i1 %45, label %56, label %46

46:                                               ; preds = %43, %53
  %47 = phi i64 [ %49, %53 ], [ %44, %43 ]
  %48 = add nsw i64 %47, -1
  %49 = sdiv i64 %48, 2
  %50 = getelementptr inbounds double, ptr %0, i64 %49
  %51 = load double, ptr %50, align 8, !tbaa !33
  %52 = fcmp olt double %51, %16
  br i1 %52, label %53, label %56

53:                                               ; preds = %46
  %54 = getelementptr inbounds double, ptr %0, i64 %47
  store double %51, ptr %54, align 8, !tbaa !33
  %55 = icmp slt i64 %49, %13
  br i1 %55, label %56, label %46, !llvm.loop !168

56:                                               ; preds = %46, %53, %43
  %57 = phi i64 [ %44, %43 ], [ %47, %46 ], [ %49, %53 ]
  %58 = getelementptr inbounds double, ptr %0, i64 %57
  store double %16, ptr %58, align 8, !tbaa !33
  %59 = icmp sgt i64 %13, 1
  br i1 %59, label %12, label %60, !llvm.loop !169

60:                                               ; preds = %56, %106
  %61 = phi i64 [ %62, %106 ], [ %6, %56 ]
  %62 = add nsw i64 %61, -1
  %63 = getelementptr inbounds nuw double, ptr %0, i64 %62
  %64 = load double, ptr %63, align 8, !tbaa !33
  %65 = load double, ptr %0, align 8, !tbaa !33
  store double %65, ptr %63, align 8, !tbaa !33
  %66 = icmp samesign ugt i64 %62, 2
  br i1 %66, label %67, label %83

67:                                               ; preds = %60, %67
  %68 = phi i64 [ %77, %67 ], [ 0, %60 ]
  %69 = phi i64 [ %81, %67 ], [ 2, %60 ]
  %70 = getelementptr double, ptr %0, i64 %69
  %71 = getelementptr i8, ptr %70, i64 -8
  %72 = load double, ptr %71, align 8, !tbaa !33
  %73 = load double, ptr %70, align 8, !tbaa !33
  %74 = fcmp olt double %72, %73
  %75 = zext i1 %74 to i64
  %76 = or disjoint i64 %69, %75
  %77 = add nsw i64 %76, -1
  %78 = getelementptr inbounds double, ptr %0, i64 %77
  %79 = load double, ptr %78, align 8, !tbaa !33
  %80 = getelementptr inbounds double, ptr %0, i64 %68
  store double %79, ptr %80, align 8, !tbaa !33
  %81 = shl nsw i64 %76, 1
  %82 = icmp slt i64 %81, %62
  br i1 %82, label %67, label %83, !llvm.loop !167

83:                                               ; preds = %67, %60
  %84 = phi i64 [ 2, %60 ], [ %81, %67 ]
  %85 = phi i64 [ 0, %60 ], [ %77, %67 ]
  %86 = icmp eq i64 %84, %62
  br i1 %86, label %87, label %92

87:                                               ; preds = %83
  %88 = add nsw i64 %61, -2
  %89 = getelementptr inbounds double, ptr %0, i64 %88
  %90 = load double, ptr %89, align 8, !tbaa !33
  %91 = getelementptr inbounds double, ptr %0, i64 %85
  store double %90, ptr %91, align 8, !tbaa !33
  br label %94

92:                                               ; preds = %83
  %93 = icmp sgt i64 %85, 0
  br i1 %93, label %94, label %106

94:                                               ; preds = %87, %92
  %95 = phi i64 [ %85, %92 ], [ %88, %87 ]
  br label %96

96:                                               ; preds = %94, %103
  %97 = phi i64 [ %99, %103 ], [ %95, %94 ]
  %98 = add nsw i64 %97, -1
  %99 = lshr i64 %98, 1
  %100 = getelementptr inbounds nuw double, ptr %0, i64 %99
  %101 = load double, ptr %100, align 8, !tbaa !33
  %102 = fcmp olt double %101, %64
  br i1 %102, label %103, label %106

103:                                              ; preds = %96
  %104 = getelementptr inbounds nuw double, ptr %0, i64 %97
  store double %101, ptr %104, align 8, !tbaa !33
  %105 = icmp ult i64 %98, 2
  br i1 %105, label %106, label %96, !llvm.loop !168

106:                                              ; preds = %96, %103, %92
  %107 = phi i64 [ %85, %92 ], [ %97, %96 ], [ 0, %103 ]
  %108 = getelementptr inbounds double, ptr %0, i64 %107
  store double %64, ptr %108, align 8, !tbaa !33
  %109 = icmp sgt i64 %61, 2
  br i1 %109, label %60, label %110, !llvm.loop !170

110:                                              ; preds = %106, %2
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZN9benchmark8heapsortI14PointerWrapperIdEdEEvT_S3_(ptr %0, ptr %1) local_unnamed_addr #9 comdat {
  %3 = ptrtoint ptr %1 to i64
  %4 = ptrtoint ptr %0 to i64
  %5 = sub i64 %3, %4
  %6 = ashr exact i64 %5, 3
  %7 = icmp sgt i64 %6, 1
  br i1 %7, label %8, label %110

8:                                                ; preds = %2
  %9 = lshr i64 %6, 1
  %10 = add nsw i64 %6, -1
  %11 = getelementptr inbounds nuw double, ptr %0, i64 %10
  br label %12

12:                                               ; preds = %8, %56
  %13 = phi i64 [ %9, %8 ], [ %14, %56 ]
  %14 = add nsw i64 %13, -1
  %15 = getelementptr inbounds double, ptr %0, i64 %14
  %16 = load double, ptr %15, align 8, !tbaa !33
  %17 = shl nuw i64 %14, 1
  %18 = add nuw nsw i64 %17, 2
  %19 = icmp slt i64 %18, %6
  br i1 %19, label %20, label %36

20:                                               ; preds = %12, %20
  %21 = phi i64 [ %30, %20 ], [ %14, %12 ]
  %22 = phi i64 [ %34, %20 ], [ %18, %12 ]
  %23 = getelementptr double, ptr %0, i64 %22
  %24 = getelementptr i8, ptr %23, i64 -8
  %25 = load double, ptr %24, align 8, !tbaa !33
  %26 = load double, ptr %23, align 8, !tbaa !33
  %27 = fcmp olt double %25, %26
  %28 = zext i1 %27 to i64
  %29 = add nsw i64 %22, %28
  %30 = add nsw i64 %29, -1
  %31 = getelementptr inbounds double, ptr %0, i64 %30
  %32 = load double, ptr %31, align 8, !tbaa !33
  %33 = getelementptr inbounds double, ptr %0, i64 %21
  store double %32, ptr %33, align 8, !tbaa !33
  %34 = shl nsw i64 %29, 1
  %35 = icmp slt i64 %34, %6
  br i1 %35, label %20, label %36, !llvm.loop !171

36:                                               ; preds = %20, %12
  %37 = phi i64 [ %18, %12 ], [ %34, %20 ]
  %38 = phi i64 [ %14, %12 ], [ %30, %20 ]
  %39 = icmp eq i64 %37, %6
  br i1 %39, label %40, label %43

40:                                               ; preds = %36
  %41 = load double, ptr %11, align 8, !tbaa !33
  %42 = getelementptr inbounds double, ptr %0, i64 %38
  store double %41, ptr %42, align 8, !tbaa !33
  br label %43

43:                                               ; preds = %40, %36
  %44 = phi i64 [ %10, %40 ], [ %38, %36 ]
  %45 = icmp slt i64 %44, %13
  br i1 %45, label %56, label %46

46:                                               ; preds = %43, %53
  %47 = phi i64 [ %49, %53 ], [ %44, %43 ]
  %48 = add nsw i64 %47, -1
  %49 = sdiv i64 %48, 2
  %50 = getelementptr inbounds double, ptr %0, i64 %49
  %51 = load double, ptr %50, align 8, !tbaa !33
  %52 = fcmp olt double %51, %16
  br i1 %52, label %53, label %56

53:                                               ; preds = %46
  %54 = getelementptr inbounds double, ptr %0, i64 %47
  store double %51, ptr %54, align 8, !tbaa !33
  %55 = icmp slt i64 %49, %13
  br i1 %55, label %56, label %46, !llvm.loop !172

56:                                               ; preds = %46, %53, %43
  %57 = phi i64 [ %44, %43 ], [ %47, %46 ], [ %49, %53 ]
  %58 = getelementptr inbounds double, ptr %0, i64 %57
  store double %16, ptr %58, align 8, !tbaa !33
  %59 = icmp sgt i64 %13, 1
  br i1 %59, label %12, label %60, !llvm.loop !173

60:                                               ; preds = %56, %106
  %61 = phi i64 [ %62, %106 ], [ %6, %56 ]
  %62 = add nsw i64 %61, -1
  %63 = getelementptr inbounds double, ptr %0, i64 %62
  %64 = load double, ptr %63, align 8, !tbaa !33
  %65 = load double, ptr %0, align 8, !tbaa !33
  store double %65, ptr %63, align 8, !tbaa !33
  %66 = icmp samesign ugt i64 %62, 2
  br i1 %66, label %67, label %83

67:                                               ; preds = %60, %67
  %68 = phi i64 [ %77, %67 ], [ 0, %60 ]
  %69 = phi i64 [ %81, %67 ], [ 2, %60 ]
  %70 = getelementptr double, ptr %0, i64 %69
  %71 = getelementptr i8, ptr %70, i64 -8
  %72 = load double, ptr %71, align 8, !tbaa !33
  %73 = load double, ptr %70, align 8, !tbaa !33
  %74 = fcmp olt double %72, %73
  %75 = zext i1 %74 to i64
  %76 = or disjoint i64 %69, %75
  %77 = add nsw i64 %76, -1
  %78 = getelementptr inbounds double, ptr %0, i64 %77
  %79 = load double, ptr %78, align 8, !tbaa !33
  %80 = getelementptr inbounds double, ptr %0, i64 %68
  store double %79, ptr %80, align 8, !tbaa !33
  %81 = shl nsw i64 %76, 1
  %82 = icmp slt i64 %81, %62
  br i1 %82, label %67, label %83, !llvm.loop !171

83:                                               ; preds = %67, %60
  %84 = phi i64 [ 2, %60 ], [ %81, %67 ]
  %85 = phi i64 [ 0, %60 ], [ %77, %67 ]
  %86 = icmp eq i64 %84, %62
  br i1 %86, label %87, label %92

87:                                               ; preds = %83
  %88 = add nsw i64 %61, -2
  %89 = getelementptr inbounds double, ptr %0, i64 %88
  %90 = load double, ptr %89, align 8, !tbaa !33
  %91 = getelementptr inbounds double, ptr %0, i64 %85
  store double %90, ptr %91, align 8, !tbaa !33
  br label %94

92:                                               ; preds = %83
  %93 = icmp sgt i64 %85, 0
  br i1 %93, label %94, label %106

94:                                               ; preds = %87, %92
  %95 = phi i64 [ %85, %92 ], [ %88, %87 ]
  br label %96

96:                                               ; preds = %94, %103
  %97 = phi i64 [ %99, %103 ], [ %95, %94 ]
  %98 = add nsw i64 %97, -1
  %99 = lshr i64 %98, 1
  %100 = getelementptr inbounds nuw double, ptr %0, i64 %99
  %101 = load double, ptr %100, align 8, !tbaa !33
  %102 = fcmp olt double %101, %64
  br i1 %102, label %103, label %106

103:                                              ; preds = %96
  %104 = getelementptr inbounds nuw double, ptr %0, i64 %97
  store double %101, ptr %104, align 8, !tbaa !33
  %105 = icmp ult i64 %98, 2
  br i1 %105, label %106, label %96, !llvm.loop !172

106:                                              ; preds = %96, %103, %92
  %107 = phi i64 [ %85, %92 ], [ %97, %96 ], [ 0, %103 ]
  %108 = getelementptr inbounds double, ptr %0, i64 %107
  store double %64, ptr %108, align 8, !tbaa !33
  %109 = icmp sgt i64 %61, 2
  br i1 %109, label %60, label %110, !llvm.loop !174

110:                                              ; preds = %106, %2
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZN9benchmark8heapsortIP12ValueWrapperIdES2_EEvT_S4_(ptr noundef %0, ptr noundef %1) local_unnamed_addr #9 comdat {
  %3 = ptrtoint ptr %1 to i64
  %4 = ptrtoint ptr %0 to i64
  %5 = sub i64 %3, %4
  %6 = ashr exact i64 %5, 3
  %7 = icmp sgt i64 %6, 1
  br i1 %7, label %8, label %110

8:                                                ; preds = %2
  %9 = lshr i64 %6, 1
  %10 = add nsw i64 %6, -1
  %11 = getelementptr inbounds nuw %struct.ValueWrapper, ptr %0, i64 %10
  br label %12

12:                                               ; preds = %8, %56
  %13 = phi i64 [ %9, %8 ], [ %14, %56 ]
  %14 = add nsw i64 %13, -1
  %15 = getelementptr inbounds nuw %struct.ValueWrapper, ptr %0, i64 %14
  %16 = load double, ptr %15, align 8, !tbaa !33
  %17 = shl nuw i64 %14, 1
  %18 = add nuw nsw i64 %17, 2
  %19 = icmp slt i64 %18, %6
  br i1 %19, label %20, label %36

20:                                               ; preds = %12, %20
  %21 = phi i64 [ %30, %20 ], [ %14, %12 ]
  %22 = phi i64 [ %34, %20 ], [ %18, %12 ]
  %23 = getelementptr %struct.ValueWrapper, ptr %0, i64 %22
  %24 = getelementptr i8, ptr %23, i64 -8
  %25 = load double, ptr %24, align 8, !tbaa !51
  %26 = load double, ptr %23, align 8, !tbaa !51
  %27 = fcmp olt double %25, %26
  %28 = zext i1 %27 to i64
  %29 = add nsw i64 %22, %28
  %30 = add nsw i64 %29, -1
  %31 = getelementptr inbounds %struct.ValueWrapper, ptr %0, i64 %30
  %32 = getelementptr inbounds %struct.ValueWrapper, ptr %0, i64 %21
  %33 = load i64, ptr %31, align 8, !tbaa !33
  store i64 %33, ptr %32, align 8, !tbaa !33
  %34 = shl nsw i64 %29, 1
  %35 = icmp slt i64 %34, %6
  br i1 %35, label %20, label %36, !llvm.loop !175

36:                                               ; preds = %20, %12
  %37 = phi i64 [ %18, %12 ], [ %34, %20 ]
  %38 = phi i64 [ %14, %12 ], [ %30, %20 ]
  %39 = icmp eq i64 %37, %6
  br i1 %39, label %40, label %43

40:                                               ; preds = %36
  %41 = getelementptr inbounds %struct.ValueWrapper, ptr %0, i64 %38
  %42 = load i64, ptr %11, align 8, !tbaa !33
  store i64 %42, ptr %41, align 8, !tbaa !33
  br label %43

43:                                               ; preds = %40, %36
  %44 = phi i64 [ %10, %40 ], [ %38, %36 ]
  %45 = icmp slt i64 %44, %13
  br i1 %45, label %56, label %46

46:                                               ; preds = %43, %53
  %47 = phi i64 [ %49, %53 ], [ %44, %43 ]
  %48 = add nsw i64 %47, -1
  %49 = sdiv i64 %48, 2
  %50 = getelementptr inbounds %struct.ValueWrapper, ptr %0, i64 %49
  %51 = load double, ptr %50, align 8
  %52 = fcmp olt double %51, %16
  br i1 %52, label %53, label %56

53:                                               ; preds = %46
  %54 = getelementptr inbounds %struct.ValueWrapper, ptr %0, i64 %47
  store double %51, ptr %54, align 8, !tbaa !33
  %55 = icmp slt i64 %49, %13
  br i1 %55, label %56, label %46, !llvm.loop !176

56:                                               ; preds = %46, %53, %43
  %57 = phi i64 [ %44, %43 ], [ %47, %46 ], [ %49, %53 ]
  %58 = getelementptr inbounds %struct.ValueWrapper, ptr %0, i64 %57
  store double %16, ptr %58, align 8, !tbaa !33
  %59 = icmp sgt i64 %13, 1
  br i1 %59, label %12, label %60, !llvm.loop !177

60:                                               ; preds = %56, %106
  %61 = phi i64 [ %62, %106 ], [ %6, %56 ]
  %62 = add nsw i64 %61, -1
  %63 = getelementptr inbounds nuw %struct.ValueWrapper, ptr %0, i64 %62
  %64 = load double, ptr %63, align 8, !tbaa !33
  %65 = load i64, ptr %0, align 8, !tbaa !33
  store i64 %65, ptr %63, align 8, !tbaa !33
  %66 = icmp samesign ugt i64 %62, 2
  br i1 %66, label %67, label %83

67:                                               ; preds = %60, %67
  %68 = phi i64 [ %77, %67 ], [ 0, %60 ]
  %69 = phi i64 [ %81, %67 ], [ 2, %60 ]
  %70 = getelementptr %struct.ValueWrapper, ptr %0, i64 %69
  %71 = getelementptr i8, ptr %70, i64 -8
  %72 = load double, ptr %71, align 8, !tbaa !51
  %73 = load double, ptr %70, align 8, !tbaa !51
  %74 = fcmp olt double %72, %73
  %75 = zext i1 %74 to i64
  %76 = or disjoint i64 %69, %75
  %77 = add nsw i64 %76, -1
  %78 = getelementptr inbounds %struct.ValueWrapper, ptr %0, i64 %77
  %79 = getelementptr inbounds %struct.ValueWrapper, ptr %0, i64 %68
  %80 = load i64, ptr %78, align 8, !tbaa !33
  store i64 %80, ptr %79, align 8, !tbaa !33
  %81 = shl nsw i64 %76, 1
  %82 = icmp slt i64 %81, %62
  br i1 %82, label %67, label %83, !llvm.loop !175

83:                                               ; preds = %67, %60
  %84 = phi i64 [ 2, %60 ], [ %81, %67 ]
  %85 = phi i64 [ 0, %60 ], [ %77, %67 ]
  %86 = icmp eq i64 %84, %62
  br i1 %86, label %87, label %92

87:                                               ; preds = %83
  %88 = add nsw i64 %61, -2
  %89 = getelementptr inbounds %struct.ValueWrapper, ptr %0, i64 %88
  %90 = getelementptr inbounds %struct.ValueWrapper, ptr %0, i64 %85
  %91 = load i64, ptr %89, align 8, !tbaa !33
  store i64 %91, ptr %90, align 8, !tbaa !33
  br label %94

92:                                               ; preds = %83
  %93 = icmp sgt i64 %85, 0
  br i1 %93, label %94, label %106

94:                                               ; preds = %87, %92
  %95 = phi i64 [ %85, %92 ], [ %88, %87 ]
  br label %96

96:                                               ; preds = %94, %103
  %97 = phi i64 [ %99, %103 ], [ %95, %94 ]
  %98 = add nsw i64 %97, -1
  %99 = lshr i64 %98, 1
  %100 = getelementptr inbounds nuw %struct.ValueWrapper, ptr %0, i64 %99
  %101 = load double, ptr %100, align 8
  %102 = fcmp olt double %101, %64
  br i1 %102, label %103, label %106

103:                                              ; preds = %96
  %104 = getelementptr inbounds nuw %struct.ValueWrapper, ptr %0, i64 %97
  store double %101, ptr %104, align 8, !tbaa !33
  %105 = icmp ult i64 %98, 2
  br i1 %105, label %106, label %96, !llvm.loop !176

106:                                              ; preds = %96, %103, %92
  %107 = phi i64 [ %85, %92 ], [ %97, %96 ], [ 0, %103 ]
  %108 = getelementptr inbounds %struct.ValueWrapper, ptr %0, i64 %107
  store double %64, ptr %108, align 8, !tbaa !33
  %109 = icmp sgt i64 %61, 2
  br i1 %109, label %60, label %110, !llvm.loop !178

110:                                              ; preds = %106, %2
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZN9benchmark8heapsortI14PointerWrapperI12ValueWrapperIdEES3_EEvT_S5_(ptr %0, ptr %1) local_unnamed_addr #9 comdat {
  %3 = ptrtoint ptr %1 to i64
  %4 = ptrtoint ptr %0 to i64
  %5 = sub i64 %3, %4
  %6 = ashr exact i64 %5, 3
  %7 = icmp sgt i64 %6, 1
  br i1 %7, label %8, label %110

8:                                                ; preds = %2
  %9 = lshr i64 %6, 1
  %10 = add nsw i64 %6, -1
  %11 = getelementptr inbounds nuw %struct.ValueWrapper, ptr %0, i64 %10
  br label %12

12:                                               ; preds = %8, %56
  %13 = phi i64 [ %9, %8 ], [ %14, %56 ]
  %14 = add nsw i64 %13, -1
  %15 = getelementptr inbounds %struct.ValueWrapper, ptr %0, i64 %14
  %16 = load double, ptr %15, align 8, !tbaa !33
  %17 = shl nuw i64 %14, 1
  %18 = add nuw nsw i64 %17, 2
  %19 = icmp slt i64 %18, %6
  br i1 %19, label %20, label %36

20:                                               ; preds = %12, %20
  %21 = phi i64 [ %30, %20 ], [ %14, %12 ]
  %22 = phi i64 [ %34, %20 ], [ %18, %12 ]
  %23 = getelementptr %struct.ValueWrapper, ptr %0, i64 %22
  %24 = getelementptr i8, ptr %23, i64 -8
  %25 = load double, ptr %24, align 8, !tbaa !51
  %26 = load double, ptr %23, align 8, !tbaa !51
  %27 = fcmp olt double %25, %26
  %28 = zext i1 %27 to i64
  %29 = add nsw i64 %22, %28
  %30 = add nsw i64 %29, -1
  %31 = getelementptr inbounds %struct.ValueWrapper, ptr %0, i64 %30
  %32 = getelementptr inbounds %struct.ValueWrapper, ptr %0, i64 %21
  %33 = load i64, ptr %31, align 8, !tbaa !33
  store i64 %33, ptr %32, align 8, !tbaa !33
  %34 = shl nsw i64 %29, 1
  %35 = icmp slt i64 %34, %6
  br i1 %35, label %20, label %36, !llvm.loop !179

36:                                               ; preds = %20, %12
  %37 = phi i64 [ %18, %12 ], [ %34, %20 ]
  %38 = phi i64 [ %14, %12 ], [ %30, %20 ]
  %39 = icmp eq i64 %37, %6
  br i1 %39, label %40, label %43

40:                                               ; preds = %36
  %41 = getelementptr inbounds %struct.ValueWrapper, ptr %0, i64 %38
  %42 = load i64, ptr %11, align 8, !tbaa !33
  store i64 %42, ptr %41, align 8, !tbaa !33
  br label %43

43:                                               ; preds = %40, %36
  %44 = phi i64 [ %10, %40 ], [ %38, %36 ]
  %45 = icmp slt i64 %44, %13
  br i1 %45, label %56, label %46

46:                                               ; preds = %43, %53
  %47 = phi i64 [ %49, %53 ], [ %44, %43 ]
  %48 = add nsw i64 %47, -1
  %49 = sdiv i64 %48, 2
  %50 = getelementptr inbounds %struct.ValueWrapper, ptr %0, i64 %49
  %51 = load double, ptr %50, align 8
  %52 = fcmp olt double %51, %16
  br i1 %52, label %53, label %56

53:                                               ; preds = %46
  %54 = getelementptr inbounds %struct.ValueWrapper, ptr %0, i64 %47
  store double %51, ptr %54, align 8, !tbaa !33
  %55 = icmp slt i64 %49, %13
  br i1 %55, label %56, label %46, !llvm.loop !180

56:                                               ; preds = %46, %53, %43
  %57 = phi i64 [ %44, %43 ], [ %47, %46 ], [ %49, %53 ]
  %58 = getelementptr inbounds %struct.ValueWrapper, ptr %0, i64 %57
  store double %16, ptr %58, align 8, !tbaa !33
  %59 = icmp sgt i64 %13, 1
  br i1 %59, label %12, label %60, !llvm.loop !181

60:                                               ; preds = %56, %106
  %61 = phi i64 [ %62, %106 ], [ %6, %56 ]
  %62 = add nsw i64 %61, -1
  %63 = getelementptr inbounds %struct.ValueWrapper, ptr %0, i64 %62
  %64 = load double, ptr %63, align 8, !tbaa !33
  %65 = load i64, ptr %0, align 8, !tbaa !33
  store i64 %65, ptr %63, align 8, !tbaa !33
  %66 = icmp samesign ugt i64 %62, 2
  br i1 %66, label %67, label %83

67:                                               ; preds = %60, %67
  %68 = phi i64 [ %77, %67 ], [ 0, %60 ]
  %69 = phi i64 [ %81, %67 ], [ 2, %60 ]
  %70 = getelementptr %struct.ValueWrapper, ptr %0, i64 %69
  %71 = getelementptr i8, ptr %70, i64 -8
  %72 = load double, ptr %71, align 8, !tbaa !51
  %73 = load double, ptr %70, align 8, !tbaa !51
  %74 = fcmp olt double %72, %73
  %75 = zext i1 %74 to i64
  %76 = or disjoint i64 %69, %75
  %77 = add nsw i64 %76, -1
  %78 = getelementptr inbounds %struct.ValueWrapper, ptr %0, i64 %77
  %79 = getelementptr inbounds %struct.ValueWrapper, ptr %0, i64 %68
  %80 = load i64, ptr %78, align 8, !tbaa !33
  store i64 %80, ptr %79, align 8, !tbaa !33
  %81 = shl nsw i64 %76, 1
  %82 = icmp slt i64 %81, %62
  br i1 %82, label %67, label %83, !llvm.loop !179

83:                                               ; preds = %67, %60
  %84 = phi i64 [ 2, %60 ], [ %81, %67 ]
  %85 = phi i64 [ 0, %60 ], [ %77, %67 ]
  %86 = icmp eq i64 %84, %62
  br i1 %86, label %87, label %92

87:                                               ; preds = %83
  %88 = add nsw i64 %61, -2
  %89 = getelementptr inbounds %struct.ValueWrapper, ptr %0, i64 %88
  %90 = getelementptr inbounds %struct.ValueWrapper, ptr %0, i64 %85
  %91 = load i64, ptr %89, align 8, !tbaa !33
  store i64 %91, ptr %90, align 8, !tbaa !33
  br label %94

92:                                               ; preds = %83
  %93 = icmp sgt i64 %85, 0
  br i1 %93, label %94, label %106

94:                                               ; preds = %87, %92
  %95 = phi i64 [ %85, %92 ], [ %88, %87 ]
  br label %96

96:                                               ; preds = %94, %103
  %97 = phi i64 [ %99, %103 ], [ %95, %94 ]
  %98 = add nsw i64 %97, -1
  %99 = lshr i64 %98, 1
  %100 = getelementptr inbounds nuw %struct.ValueWrapper, ptr %0, i64 %99
  %101 = load double, ptr %100, align 8
  %102 = fcmp olt double %101, %64
  br i1 %102, label %103, label %106

103:                                              ; preds = %96
  %104 = getelementptr inbounds nuw %struct.ValueWrapper, ptr %0, i64 %97
  store double %101, ptr %104, align 8, !tbaa !33
  %105 = icmp ult i64 %98, 2
  br i1 %105, label %106, label %96, !llvm.loop !180

106:                                              ; preds = %96, %103, %92
  %107 = phi i64 [ %85, %92 ], [ %97, %96 ], [ 0, %103 ]
  %108 = getelementptr inbounds %struct.ValueWrapper, ptr %0, i64 %107
  store double %64, ptr %108, align 8, !tbaa !33
  %109 = icmp sgt i64 %61, 2
  br i1 %109, label %60, label %110, !llvm.loop !182

110:                                              ; preds = %106, %2
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZN9benchmark8heapsortIP12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEESB_EEvT_SD_(ptr noundef %0, ptr noundef %1) local_unnamed_addr #9 comdat {
  %3 = ptrtoint ptr %1 to i64
  %4 = ptrtoint ptr %0 to i64
  %5 = sub i64 %3, %4
  %6 = ashr exact i64 %5, 3
  %7 = icmp sgt i64 %6, 1
  br i1 %7, label %8, label %110

8:                                                ; preds = %2
  %9 = lshr i64 %6, 1
  %10 = add nsw i64 %6, -1
  %11 = getelementptr inbounds nuw %struct.ValueWrapper.0, ptr %0, i64 %10
  br label %12

12:                                               ; preds = %8, %56
  %13 = phi i64 [ %9, %8 ], [ %14, %56 ]
  %14 = add nsw i64 %13, -1
  %15 = getelementptr inbounds nuw %struct.ValueWrapper.0, ptr %0, i64 %14
  %16 = load double, ptr %15, align 8, !tbaa !33
  %17 = shl nuw i64 %14, 1
  %18 = add nuw nsw i64 %17, 2
  %19 = icmp slt i64 %18, %6
  br i1 %19, label %20, label %36

20:                                               ; preds = %12, %20
  %21 = phi i64 [ %30, %20 ], [ %14, %12 ]
  %22 = phi i64 [ %34, %20 ], [ %18, %12 ]
  %23 = getelementptr %struct.ValueWrapper.0, ptr %0, i64 %22
  %24 = getelementptr i8, ptr %23, i64 -8
  %25 = load double, ptr %24, align 8, !tbaa !51
  %26 = load double, ptr %23, align 8, !tbaa !51
  %27 = fcmp olt double %25, %26
  %28 = zext i1 %27 to i64
  %29 = add nsw i64 %22, %28
  %30 = add nsw i64 %29, -1
  %31 = getelementptr inbounds %struct.ValueWrapper.0, ptr %0, i64 %30
  %32 = getelementptr inbounds %struct.ValueWrapper.0, ptr %0, i64 %21
  %33 = load i64, ptr %31, align 8, !tbaa !33
  store i64 %33, ptr %32, align 8, !tbaa !33
  %34 = shl nsw i64 %29, 1
  %35 = icmp slt i64 %34, %6
  br i1 %35, label %20, label %36, !llvm.loop !183

36:                                               ; preds = %20, %12
  %37 = phi i64 [ %18, %12 ], [ %34, %20 ]
  %38 = phi i64 [ %14, %12 ], [ %30, %20 ]
  %39 = icmp eq i64 %37, %6
  br i1 %39, label %40, label %43

40:                                               ; preds = %36
  %41 = getelementptr inbounds %struct.ValueWrapper.0, ptr %0, i64 %38
  %42 = load i64, ptr %11, align 8, !tbaa !33
  store i64 %42, ptr %41, align 8, !tbaa !33
  br label %43

43:                                               ; preds = %40, %36
  %44 = phi i64 [ %10, %40 ], [ %38, %36 ]
  %45 = icmp slt i64 %44, %13
  br i1 %45, label %56, label %46

46:                                               ; preds = %43, %53
  %47 = phi i64 [ %49, %53 ], [ %44, %43 ]
  %48 = add nsw i64 %47, -1
  %49 = sdiv i64 %48, 2
  %50 = getelementptr inbounds %struct.ValueWrapper.0, ptr %0, i64 %49
  %51 = load double, ptr %50, align 8
  %52 = fcmp olt double %51, %16
  br i1 %52, label %53, label %56

53:                                               ; preds = %46
  %54 = getelementptr inbounds %struct.ValueWrapper.0, ptr %0, i64 %47
  store double %51, ptr %54, align 8, !tbaa !33
  %55 = icmp slt i64 %49, %13
  br i1 %55, label %56, label %46, !llvm.loop !184

56:                                               ; preds = %46, %53, %43
  %57 = phi i64 [ %44, %43 ], [ %47, %46 ], [ %49, %53 ]
  %58 = getelementptr inbounds %struct.ValueWrapper.0, ptr %0, i64 %57
  store double %16, ptr %58, align 8, !tbaa !33
  %59 = icmp sgt i64 %13, 1
  br i1 %59, label %12, label %60, !llvm.loop !185

60:                                               ; preds = %56, %106
  %61 = phi i64 [ %62, %106 ], [ %6, %56 ]
  %62 = add nsw i64 %61, -1
  %63 = getelementptr inbounds nuw %struct.ValueWrapper.0, ptr %0, i64 %62
  %64 = load double, ptr %63, align 8, !tbaa !33
  %65 = load i64, ptr %0, align 8, !tbaa !33
  store i64 %65, ptr %63, align 8, !tbaa !33
  %66 = icmp samesign ugt i64 %62, 2
  br i1 %66, label %67, label %83

67:                                               ; preds = %60, %67
  %68 = phi i64 [ %77, %67 ], [ 0, %60 ]
  %69 = phi i64 [ %81, %67 ], [ 2, %60 ]
  %70 = getelementptr %struct.ValueWrapper.0, ptr %0, i64 %69
  %71 = getelementptr i8, ptr %70, i64 -8
  %72 = load double, ptr %71, align 8, !tbaa !51
  %73 = load double, ptr %70, align 8, !tbaa !51
  %74 = fcmp olt double %72, %73
  %75 = zext i1 %74 to i64
  %76 = or disjoint i64 %69, %75
  %77 = add nsw i64 %76, -1
  %78 = getelementptr inbounds %struct.ValueWrapper.0, ptr %0, i64 %77
  %79 = getelementptr inbounds %struct.ValueWrapper.0, ptr %0, i64 %68
  %80 = load i64, ptr %78, align 8, !tbaa !33
  store i64 %80, ptr %79, align 8, !tbaa !33
  %81 = shl nsw i64 %76, 1
  %82 = icmp slt i64 %81, %62
  br i1 %82, label %67, label %83, !llvm.loop !183

83:                                               ; preds = %67, %60
  %84 = phi i64 [ 2, %60 ], [ %81, %67 ]
  %85 = phi i64 [ 0, %60 ], [ %77, %67 ]
  %86 = icmp eq i64 %84, %62
  br i1 %86, label %87, label %92

87:                                               ; preds = %83
  %88 = add nsw i64 %61, -2
  %89 = getelementptr inbounds %struct.ValueWrapper.0, ptr %0, i64 %88
  %90 = getelementptr inbounds %struct.ValueWrapper.0, ptr %0, i64 %85
  %91 = load i64, ptr %89, align 8, !tbaa !33
  store i64 %91, ptr %90, align 8, !tbaa !33
  br label %94

92:                                               ; preds = %83
  %93 = icmp sgt i64 %85, 0
  br i1 %93, label %94, label %106

94:                                               ; preds = %87, %92
  %95 = phi i64 [ %85, %92 ], [ %88, %87 ]
  br label %96

96:                                               ; preds = %94, %103
  %97 = phi i64 [ %99, %103 ], [ %95, %94 ]
  %98 = add nsw i64 %97, -1
  %99 = lshr i64 %98, 1
  %100 = getelementptr inbounds nuw %struct.ValueWrapper.0, ptr %0, i64 %99
  %101 = load double, ptr %100, align 8
  %102 = fcmp olt double %101, %64
  br i1 %102, label %103, label %106

103:                                              ; preds = %96
  %104 = getelementptr inbounds nuw %struct.ValueWrapper.0, ptr %0, i64 %97
  store double %101, ptr %104, align 8, !tbaa !33
  %105 = icmp ult i64 %98, 2
  br i1 %105, label %106, label %96, !llvm.loop !184

106:                                              ; preds = %96, %103, %92
  %107 = phi i64 [ %85, %92 ], [ %97, %96 ], [ 0, %103 ]
  %108 = getelementptr inbounds %struct.ValueWrapper.0, ptr %0, i64 %107
  store double %64, ptr %108, align 8, !tbaa !33
  %109 = icmp sgt i64 %61, 2
  br i1 %109, label %60, label %110, !llvm.loop !186

110:                                              ; preds = %106, %2
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZN9benchmark8heapsortI14PointerWrapperI12ValueWrapperIS2_IS2_IS2_IS2_IS2_IS2_IS2_IS2_IS2_IdEEEEEEEEEEESC_EEvT_SE_(ptr %0, ptr %1) local_unnamed_addr #9 comdat {
  %3 = ptrtoint ptr %1 to i64
  %4 = ptrtoint ptr %0 to i64
  %5 = sub i64 %3, %4
  %6 = ashr exact i64 %5, 3
  %7 = icmp sgt i64 %6, 1
  br i1 %7, label %8, label %110

8:                                                ; preds = %2
  %9 = lshr i64 %6, 1
  %10 = add nsw i64 %6, -1
  %11 = getelementptr inbounds nuw %struct.ValueWrapper.0, ptr %0, i64 %10
  br label %12

12:                                               ; preds = %8, %56
  %13 = phi i64 [ %9, %8 ], [ %14, %56 ]
  %14 = add nsw i64 %13, -1
  %15 = getelementptr inbounds %struct.ValueWrapper.0, ptr %0, i64 %14
  %16 = load double, ptr %15, align 8, !tbaa !33
  %17 = shl nuw i64 %14, 1
  %18 = add nuw nsw i64 %17, 2
  %19 = icmp slt i64 %18, %6
  br i1 %19, label %20, label %36

20:                                               ; preds = %12, %20
  %21 = phi i64 [ %30, %20 ], [ %14, %12 ]
  %22 = phi i64 [ %34, %20 ], [ %18, %12 ]
  %23 = getelementptr %struct.ValueWrapper.0, ptr %0, i64 %22
  %24 = getelementptr i8, ptr %23, i64 -8
  %25 = load double, ptr %24, align 8, !tbaa !51
  %26 = load double, ptr %23, align 8, !tbaa !51
  %27 = fcmp olt double %25, %26
  %28 = zext i1 %27 to i64
  %29 = add nsw i64 %22, %28
  %30 = add nsw i64 %29, -1
  %31 = getelementptr inbounds %struct.ValueWrapper.0, ptr %0, i64 %30
  %32 = getelementptr inbounds %struct.ValueWrapper.0, ptr %0, i64 %21
  %33 = load i64, ptr %31, align 8, !tbaa !33
  store i64 %33, ptr %32, align 8, !tbaa !33
  %34 = shl nsw i64 %29, 1
  %35 = icmp slt i64 %34, %6
  br i1 %35, label %20, label %36, !llvm.loop !187

36:                                               ; preds = %20, %12
  %37 = phi i64 [ %18, %12 ], [ %34, %20 ]
  %38 = phi i64 [ %14, %12 ], [ %30, %20 ]
  %39 = icmp eq i64 %37, %6
  br i1 %39, label %40, label %43

40:                                               ; preds = %36
  %41 = getelementptr inbounds %struct.ValueWrapper.0, ptr %0, i64 %38
  %42 = load i64, ptr %11, align 8, !tbaa !33
  store i64 %42, ptr %41, align 8, !tbaa !33
  br label %43

43:                                               ; preds = %40, %36
  %44 = phi i64 [ %10, %40 ], [ %38, %36 ]
  %45 = icmp slt i64 %44, %13
  br i1 %45, label %56, label %46

46:                                               ; preds = %43, %53
  %47 = phi i64 [ %49, %53 ], [ %44, %43 ]
  %48 = add nsw i64 %47, -1
  %49 = sdiv i64 %48, 2
  %50 = getelementptr inbounds %struct.ValueWrapper.0, ptr %0, i64 %49
  %51 = load double, ptr %50, align 8
  %52 = fcmp olt double %51, %16
  br i1 %52, label %53, label %56

53:                                               ; preds = %46
  %54 = getelementptr inbounds %struct.ValueWrapper.0, ptr %0, i64 %47
  store double %51, ptr %54, align 8, !tbaa !33
  %55 = icmp slt i64 %49, %13
  br i1 %55, label %56, label %46, !llvm.loop !188

56:                                               ; preds = %46, %53, %43
  %57 = phi i64 [ %44, %43 ], [ %47, %46 ], [ %49, %53 ]
  %58 = getelementptr inbounds %struct.ValueWrapper.0, ptr %0, i64 %57
  store double %16, ptr %58, align 8, !tbaa !33
  %59 = icmp sgt i64 %13, 1
  br i1 %59, label %12, label %60, !llvm.loop !189

60:                                               ; preds = %56, %106
  %61 = phi i64 [ %62, %106 ], [ %6, %56 ]
  %62 = add nsw i64 %61, -1
  %63 = getelementptr inbounds %struct.ValueWrapper.0, ptr %0, i64 %62
  %64 = load double, ptr %63, align 8, !tbaa !33
  %65 = load i64, ptr %0, align 8, !tbaa !33
  store i64 %65, ptr %63, align 8, !tbaa !33
  %66 = icmp samesign ugt i64 %62, 2
  br i1 %66, label %67, label %83

67:                                               ; preds = %60, %67
  %68 = phi i64 [ %77, %67 ], [ 0, %60 ]
  %69 = phi i64 [ %81, %67 ], [ 2, %60 ]
  %70 = getelementptr %struct.ValueWrapper.0, ptr %0, i64 %69
  %71 = getelementptr i8, ptr %70, i64 -8
  %72 = load double, ptr %71, align 8, !tbaa !51
  %73 = load double, ptr %70, align 8, !tbaa !51
  %74 = fcmp olt double %72, %73
  %75 = zext i1 %74 to i64
  %76 = or disjoint i64 %69, %75
  %77 = add nsw i64 %76, -1
  %78 = getelementptr inbounds %struct.ValueWrapper.0, ptr %0, i64 %77
  %79 = getelementptr inbounds %struct.ValueWrapper.0, ptr %0, i64 %68
  %80 = load i64, ptr %78, align 8, !tbaa !33
  store i64 %80, ptr %79, align 8, !tbaa !33
  %81 = shl nsw i64 %76, 1
  %82 = icmp slt i64 %81, %62
  br i1 %82, label %67, label %83, !llvm.loop !187

83:                                               ; preds = %67, %60
  %84 = phi i64 [ 2, %60 ], [ %81, %67 ]
  %85 = phi i64 [ 0, %60 ], [ %77, %67 ]
  %86 = icmp eq i64 %84, %62
  br i1 %86, label %87, label %92

87:                                               ; preds = %83
  %88 = add nsw i64 %61, -2
  %89 = getelementptr inbounds %struct.ValueWrapper.0, ptr %0, i64 %88
  %90 = getelementptr inbounds %struct.ValueWrapper.0, ptr %0, i64 %85
  %91 = load i64, ptr %89, align 8, !tbaa !33
  store i64 %91, ptr %90, align 8, !tbaa !33
  br label %94

92:                                               ; preds = %83
  %93 = icmp sgt i64 %85, 0
  br i1 %93, label %94, label %106

94:                                               ; preds = %87, %92
  %95 = phi i64 [ %85, %92 ], [ %88, %87 ]
  br label %96

96:                                               ; preds = %94, %103
  %97 = phi i64 [ %99, %103 ], [ %95, %94 ]
  %98 = add nsw i64 %97, -1
  %99 = lshr i64 %98, 1
  %100 = getelementptr inbounds nuw %struct.ValueWrapper.0, ptr %0, i64 %99
  %101 = load double, ptr %100, align 8
  %102 = fcmp olt double %101, %64
  br i1 %102, label %103, label %106

103:                                              ; preds = %96
  %104 = getelementptr inbounds nuw %struct.ValueWrapper.0, ptr %0, i64 %97
  store double %101, ptr %104, align 8, !tbaa !33
  %105 = icmp ult i64 %98, 2
  br i1 %105, label %106, label %96, !llvm.loop !188

106:                                              ; preds = %96, %103, %92
  %107 = phi i64 [ %85, %92 ], [ %97, %96 ], [ 0, %103 ]
  %108 = getelementptr inbounds %struct.ValueWrapper.0, ptr %0, i64 %107
  store double %64, ptr %108, align 8, !tbaa !33
  %109 = icmp sgt i64 %61, 2
  br i1 %109, label %60, label %110, !llvm.loop !190

110:                                              ; preds = %106, %2
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smax.i32(i32, i32) #11

attributes #0 = { mustprogress nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nounwind willreturn allockind("realloc") allocsize(1) memory(argmem: readwrite, inaccessiblemem: readwrite) "alloc-family"="malloc" "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { mustprogress nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: read) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { mustprogress nocallback nofree nounwind willreturn memory(errnomem: write) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #8 = { mustprogress norecurse uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #9 = { mustprogress uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #10 = { mustprogress nocallback nofree nounwind willreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #11 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #12 = { nounwind allocsize(1) }
attributes #13 = { cold noreturn nounwind }
attributes #14 = { nounwind willreturn memory(read) }
attributes #15 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"p1 _ZTS10one_result", !8, i64 0}
!8 = !{!"any pointer", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C++ TBAA"}
!11 = !{!12, !12, i64 0}
!12 = !{!"int", !9, i64 0}
!13 = !{!14, !15, i64 0}
!14 = !{!"_ZTS10one_result", !15, i64 0, !16, i64 8}
!15 = !{!"double", !9, i64 0}
!16 = !{!"p1 omnipotent char", !8, i64 0}
!17 = !{!14, !16, i64 8}
!18 = distinct !{!18, !19}
!19 = !{!"llvm.loop.mustprogress"}
!20 = distinct !{!20, !19, !21, !22}
!21 = !{!"llvm.loop.isvectorized", i32 1}
!22 = !{!"llvm.loop.unroll.runtime.disable"}
!23 = distinct !{!23, !19}
!24 = distinct !{!24, !19, !21}
!25 = distinct !{!25, !19}
!26 = distinct !{!26, !19}
!27 = distinct !{!27, !19, !21, !22}
!28 = distinct !{!28, !19}
!29 = distinct !{!29, !19, !21}
!30 = !{!31, !31, i64 0}
!31 = !{!"long", !9, i64 0}
!32 = !{!16, !16, i64 0}
!33 = !{!15, !15, i64 0}
!34 = !{!35, !35, i64 0}
!35 = !{!"p1 double", !8, i64 0}
!36 = distinct !{!36, !19, !21, !22}
!37 = distinct !{!37, !19, !22, !21}
!38 = !{!39, !39, i64 0}
!39 = !{!"p1 _ZTS12ValueWrapperIdE", !8, i64 0}
!40 = distinct !{!40, !19, !21, !22}
!41 = distinct !{!41, !19, !22, !21}
!42 = !{!43, !43, i64 0}
!43 = !{!"p1 _ZTS12ValueWrapperIS_IS_IS_IS_IS_IS_IS_IS_IS_IdEEEEEEEEEE", !8, i64 0}
!44 = distinct !{!44, !19, !21, !22}
!45 = distinct !{!45, !19, !22, !21}
!46 = distinct !{!46, !19}
!47 = distinct !{!47, !19}
!48 = distinct !{!48, !19}
!49 = distinct !{!49, !19}
!50 = distinct !{!50, !19}
!51 = !{!52, !15, i64 0}
!52 = !{!"_ZTS12ValueWrapperIdE", !15, i64 0}
!53 = distinct !{!53, !19}
!54 = distinct !{!54, !19}
!55 = distinct !{!55, !19}
!56 = distinct !{!56, !19}
!57 = distinct !{!57, !19}
!58 = distinct !{!58, !19}
!59 = distinct !{!59, !19}
!60 = distinct !{!60, !19}
!61 = distinct !{!61, !19, !21, !22}
!62 = distinct !{!62, !19, !21}
!63 = distinct !{!63, !19, !21, !22}
!64 = distinct !{!64, !19, !21}
!65 = distinct !{!65, !19}
!66 = distinct !{!66, !19}
!67 = distinct !{!67, !19, !21, !22}
!68 = distinct !{!68, !19, !21}
!69 = distinct !{!69, !19}
!70 = distinct !{!70, !19}
!71 = distinct !{!71, !19, !21, !22}
!72 = distinct !{!72, !19, !21}
!73 = distinct !{!73, !19}
!74 = distinct !{!74, !19}
!75 = distinct !{!75, !19, !21, !22}
!76 = distinct !{!76, !19, !21}
!77 = distinct !{!77, !19}
!78 = distinct !{!78, !19}
!79 = distinct !{!79, !19, !21, !22}
!80 = distinct !{!80, !19, !21}
!81 = distinct !{!81, !19}
!82 = distinct !{!82, !19}
!83 = distinct !{!83, !19, !21, !22}
!84 = distinct !{!84, !19, !21}
!85 = distinct !{!85, !19}
!86 = distinct !{!86, !19}
!87 = distinct !{!87, !19, !21, !22}
!88 = distinct !{!88, !19, !21}
!89 = distinct !{!89, !19}
!90 = distinct !{!90, !19}
!91 = distinct !{!91, !19, !21, !22}
!92 = distinct !{!92, !19, !21}
!93 = distinct !{!93, !19}
!94 = distinct !{!94, !19}
!95 = distinct !{!95, !19, !21, !22}
!96 = distinct !{!96, !19, !21}
!97 = distinct !{!97, !19}
!98 = distinct !{!98, !19}
!99 = distinct !{!99, !19, !21, !22}
!100 = distinct !{!100, !19, !21}
!101 = distinct !{!101, !19}
!102 = distinct !{!102, !19}
!103 = distinct !{!103, !19, !21, !22}
!104 = distinct !{!104, !19, !21}
!105 = distinct !{!105, !19}
!106 = distinct !{!106, !19}
!107 = distinct !{!107, !19, !21, !22}
!108 = distinct !{!108, !19, !21}
!109 = distinct !{!109, !19}
!110 = distinct !{!110, !19}
!111 = distinct !{!111, !19, !21, !22}
!112 = distinct !{!112, !19, !21}
!113 = distinct !{!113, !19}
!114 = distinct !{!114, !19, !21, !22}
!115 = distinct !{!115, !19, !21}
!116 = distinct !{!116, !19}
!117 = distinct !{!117, !19, !21, !22}
!118 = distinct !{!118, !19, !21}
!119 = distinct !{!119, !19}
!120 = distinct !{!120, !19, !21, !22}
!121 = distinct !{!121, !19, !21}
!122 = distinct !{!122, !19}
!123 = distinct !{!123, !19, !21, !22}
!124 = distinct !{!124, !19, !21}
!125 = distinct !{!125, !19}
!126 = distinct !{!126, !19, !21, !22}
!127 = distinct !{!127, !19, !21}
!128 = distinct !{!128, !19}
!129 = distinct !{!129, !19, !21, !22}
!130 = distinct !{!130, !19, !21}
!131 = distinct !{!131, !19}
!132 = distinct !{!132, !19, !21, !22}
!133 = distinct !{!133, !19, !21}
!134 = distinct !{!134, !19}
!135 = distinct !{!135, !19, !21, !22}
!136 = distinct !{!136, !19, !21}
!137 = distinct !{!137, !19}
!138 = distinct !{!138, !19, !21, !22}
!139 = distinct !{!139, !19, !21}
!140 = distinct !{!140, !19}
!141 = distinct !{!141, !19, !21, !22}
!142 = distinct !{!142, !19, !21}
!143 = distinct !{!143, !19}
!144 = distinct !{!144, !19, !21, !22}
!145 = distinct !{!145, !19, !21}
!146 = distinct !{!146, !19}
!147 = distinct !{!147, !19, !21, !22}
!148 = distinct !{!148, !19, !21}
!149 = distinct !{!149, !19}
!150 = distinct !{!150, !19}
!151 = distinct !{!151, !19}
!152 = distinct !{!152, !19}
!153 = distinct !{!153, !19}
!154 = distinct !{!154, !19}
!155 = distinct !{!155, !19}
!156 = distinct !{!156, !19}
!157 = distinct !{!157, !19}
!158 = distinct !{!158, !19}
!159 = distinct !{!159, !19}
!160 = distinct !{!160, !19}
!161 = distinct !{!161, !19}
!162 = distinct !{!162, !19}
!163 = distinct !{!163, !19}
!164 = distinct !{!164, !19}
!165 = distinct !{!165, !19}
!166 = distinct !{!166, !19}
!167 = distinct !{!167, !19}
!168 = distinct !{!168, !19}
!169 = distinct !{!169, !19}
!170 = distinct !{!170, !19}
!171 = distinct !{!171, !19}
!172 = distinct !{!172, !19}
!173 = distinct !{!173, !19}
!174 = distinct !{!174, !19}
!175 = distinct !{!175, !19}
!176 = distinct !{!176, !19}
!177 = distinct !{!177, !19}
!178 = distinct !{!178, !19}
!179 = distinct !{!179, !19}
!180 = distinct !{!180, !19}
!181 = distinct !{!181, !19}
!182 = distinct !{!182, !19}
!183 = distinct !{!183, !19}
!184 = distinct !{!184, !19}
!185 = distinct !{!185, !19}
!186 = distinct !{!186, !19}
!187 = distinct !{!187, !19}
!188 = distinct !{!188, !19}
!189 = distinct !{!189, !19}
!190 = distinct !{!190, !19}
