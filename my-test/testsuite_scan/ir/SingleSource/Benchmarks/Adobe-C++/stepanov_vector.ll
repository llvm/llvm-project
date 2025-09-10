; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Adobe-C++/stepanov_vector.cpp'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Adobe-C++/stepanov_vector.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%"class.std::reverse_iterator" = type { ptr }
%"class.std::reverse_iterator.0" = type { [8 x i8], %"class.std::reverse_iterator" }
%struct.one_result = type { double, ptr }
%"class.std::reverse_iterator.2" = type { [8 x i8], %"class.std::reverse_iterator.1" }
%"class.std::reverse_iterator.1" = type { %"class.__gnu_cxx::__normal_iterator" }
%"class.__gnu_cxx::__normal_iterator" = type { ptr }

$_Z19test_insertion_sortIPddEvT_S1_S1_S1_T0_PKc = comdat any

$_Z19test_insertion_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEvT_S7_S7_S7_T0_PKc = comdat any

$_Z14test_quicksortIPddEvT_S1_S1_S1_T0_PKc = comdat any

$_Z14test_quicksortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEvT_S7_S7_S7_T0_PKc = comdat any

$_Z14test_heap_sortIPddEvT_S1_S1_S1_T0_PKc = comdat any

$_Z14test_heap_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEvT_S7_S7_S7_T0_PKc = comdat any

$_ZN9benchmark9quicksortIPddEEvT_S2_ = comdat any

$_ZN9benchmark9quicksortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEEvT_S8_ = comdat any

$_ZN9benchmark9quicksortISt16reverse_iteratorIPdEdEEvT_S4_ = comdat any

$_ZN9benchmark9quicksortISt16reverse_iteratorIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEdEEvT_SA_ = comdat any

$_ZN9benchmark9quicksortISt16reverse_iteratorIS1_IPdEEdEEvT_S5_ = comdat any

$_ZN9benchmark9quicksortISt16reverse_iteratorIS1_IN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEEdEEvT_SB_ = comdat any

$_ZN9benchmark8heapsortIPddEEvT_S2_ = comdat any

$_ZN9benchmark8heapsortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEEvT_S8_ = comdat any

$_ZN9benchmark8heapsortISt16reverse_iteratorIPdEdEEvT_S4_ = comdat any

$_ZN9benchmark8heapsortISt16reverse_iteratorIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEdEEvT_SA_ = comdat any

$_ZN9benchmark8heapsortISt16reverse_iteratorIS1_IPdEEdEEvT_S5_ = comdat any

$_ZN9benchmark8heapsortISt16reverse_iteratorIS1_IN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEEdEEvT_SB_ = comdat any

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
@iterations = dso_local local_unnamed_addr global i32 60000, align 4
@init_value = dso_local local_unnamed_addr global double 3.000000e+00, align 8
@data = dso_local global [2000 x double] zeroinitializer, align 8
@dataMaster = dso_local global [2000 x double] zeroinitializer, align 8
@dpb = dso_local local_unnamed_addr global ptr @data, align 8
@dpe = dso_local local_unnamed_addr global ptr getelementptr inbounds nuw (i8, ptr @data, i64 16000), align 8
@dMpb = dso_local local_unnamed_addr global ptr @dataMaster, align 8
@dMpe = dso_local local_unnamed_addr global ptr getelementptr inbounds nuw (i8, ptr @dataMaster, i64 16000), align 8
@rdpb = dso_local local_unnamed_addr global %"class.std::reverse_iterator" { ptr getelementptr inbounds nuw (i8, ptr @data, i64 16000) }, align 8
@rdpe = dso_local local_unnamed_addr global %"class.std::reverse_iterator" { ptr @data }, align 8
@rdMpb = dso_local local_unnamed_addr global %"class.std::reverse_iterator" { ptr getelementptr inbounds nuw (i8, ptr @dataMaster, i64 16000) }, align 8
@rdMpe = dso_local local_unnamed_addr global %"class.std::reverse_iterator" { ptr @dataMaster }, align 8
@rrdpb = dso_local local_unnamed_addr global %"class.std::reverse_iterator.0" { [8 x i8] zeroinitializer, %"class.std::reverse_iterator" { ptr @data } }, align 8
@rrdpe = dso_local local_unnamed_addr global %"class.std::reverse_iterator.0" { [8 x i8] zeroinitializer, %"class.std::reverse_iterator" { ptr getelementptr inbounds nuw (i8, ptr @data, i64 16000) } }, align 8
@rrdMpb = dso_local local_unnamed_addr global %"class.std::reverse_iterator.0" { [8 x i8] zeroinitializer, %"class.std::reverse_iterator" { ptr @dataMaster } }, align 8
@rrdMpe = dso_local local_unnamed_addr global %"class.std::reverse_iterator.0" { [8 x i8] zeroinitializer, %"class.std::reverse_iterator" { ptr getelementptr inbounds nuw (i8, ptr @dataMaster, i64 16000) } }, align 8
@.str.26 = private unnamed_addr constant [38 x i8] c"insertion_sort double pointer verify2\00", align 1
@.str.27 = private unnamed_addr constant [38 x i8] c"insertion_sort double vector iterator\00", align 1
@.str.34 = private unnamed_addr constant [33 x i8] c"quicksort double pointer verify2\00", align 1
@.str.35 = private unnamed_addr constant [33 x i8] c"quicksort double vector iterator\00", align 1
@.str.42 = private unnamed_addr constant [33 x i8] c"heap_sort double pointer verify2\00", align 1
@.str.43 = private unnamed_addr constant [33 x i8] c"heap_sort double vector iterator\00", align 1
@.str.51 = private unnamed_addr constant [16 x i8] c"test %i failed\0A\00", align 1
@.str.52 = private unnamed_addr constant [21 x i8] c"sort test %i failed\0A\00", align 1
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
  %13 = tail call ptr @realloc(ptr noundef %3, i64 noundef %12) #17
  store ptr %13, ptr @results, align 8, !tbaa !6
  %14 = icmp eq ptr %13, null
  br i1 %14, label %17, label %15

15:                                               ; preds = %9
  %16 = load i32, ptr @current_test, align 4, !tbaa !11
  br label %20

17:                                               ; preds = %9
  %18 = load i32, ptr @allocated_results, align 4, !tbaa !11
  %19 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %18)
  tail call void @exit(i32 noundef -1) #18
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
  %20 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %19) #19
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
  %63 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %62) #19
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
  %100 = tail call double @log(double noundef %99) #20, !tbaa !11
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
  %110 = tail call double @exp(double noundef %109) #20, !tbaa !11
  %111 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.7, ptr noundef %0, double noundef %110)
  br label %112

112:                                              ; preds = %106, %84
  store i32 0, ptr @current_test, align 4, !tbaa !11
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #5

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: read)
declare i64 @strlen(ptr noundef captures(none)) local_unnamed_addr #6

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #5

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(errnomem: write)
declare double @log(double noundef) local_unnamed_addr #7

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(errnomem: write)
declare double @exp(double noundef) local_unnamed_addr #7

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
  %13 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %12) #19
  %14 = trunc i64 %13 to i32
  %15 = tail call i32 @llvm.smax.i32(i32 %10, i32 %14)
  %16 = add nuw nsw i64 %9, 1
  %17 = icmp eq i64 %16, %7
  br i1 %17, label %18, label %8, !llvm.loop !26

18:                                               ; preds = %8, %2
  %19 = phi i32 [ 12, %2 ], [ %15, %8 ]
  %20 = add nsw i32 %19, -12
  %21 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %0, ptr noundef nonnull @.str.8, i32 noundef %20, ptr noundef nonnull @.str.2) #20
  %22 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %0, ptr noundef nonnull @.str.9, i32 noundef %19, ptr noundef nonnull @.str.2) #20
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
  %56 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %55) #19
  %57 = trunc i64 %56 to i32
  %58 = sub i32 %19, %57
  %59 = load double, ptr %53, align 8, !tbaa !13
  %60 = trunc nuw nsw i64 %51 to i32
  %61 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %0, ptr noundef nonnull @.str.10, i32 noundef %60, i32 noundef %58, ptr noundef nonnull @.str.5, ptr noundef nonnull %55, double noundef %59) #20
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
  %76 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %0, ptr noundef nonnull @.str.6, ptr noundef %1, double noundef %75) #20
  store i32 0, ptr @current_test, align 4, !tbaa !11
  ret void
}

; Function Attrs: nofree nounwind
declare noundef i32 @fprintf(ptr noundef captures(none), ptr noundef readonly captures(none), ...) local_unnamed_addr #2

; Function Attrs: mustprogress nounwind uwtable
define dso_local void @_Z11start_timerv() local_unnamed_addr #0 {
  %1 = tail call i64 @clock() #20
  store i64 %1, ptr @start_time, align 8, !tbaa !30
  ret void
}

; Function Attrs: nounwind
declare i64 @clock() local_unnamed_addr #8

; Function Attrs: mustprogress nounwind uwtable
define dso_local noundef double @_Z5timerv() local_unnamed_addr #0 {
  %1 = tail call i64 @clock() #20
  store i64 %1, ptr @end_time, align 8, !tbaa !30
  %2 = load i64, ptr @start_time, align 8, !tbaa !30
  %3 = sub nsw i64 %1, %2
  %4 = sitofp i64 %3 to double
  %5 = fdiv double %4, 1.000000e+06
  ret double %5
}

; Function Attrs: mustprogress norecurse uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #9 personality ptr @__gxx_personality_v0 {
  %3 = alloca %"class.std::reverse_iterator.2", align 8
  %4 = alloca %"class.std::reverse_iterator.2", align 8
  %5 = alloca %"class.std::reverse_iterator.2", align 8
  %6 = alloca %"class.std::reverse_iterator.2", align 8
  %7 = alloca %"class.std::reverse_iterator.0", align 8
  %8 = alloca %"class.std::reverse_iterator.0", align 8
  %9 = alloca %"class.std::reverse_iterator.1", align 8
  %10 = alloca %"class.std::reverse_iterator.1", align 8
  %11 = alloca %"class.std::reverse_iterator.1", align 8
  %12 = alloca %"class.std::reverse_iterator.1", align 8
  %13 = alloca %"class.std::reverse_iterator", align 8
  %14 = alloca %"class.std::reverse_iterator", align 8
  %15 = alloca %"class.std::reverse_iterator.2", align 8
  %16 = alloca %"class.std::reverse_iterator.2", align 8
  %17 = alloca %"class.std::reverse_iterator.2", align 8
  %18 = alloca %"class.std::reverse_iterator.2", align 8
  %19 = alloca %"class.std::reverse_iterator.0", align 8
  %20 = alloca %"class.std::reverse_iterator.0", align 8
  %21 = alloca %"class.std::reverse_iterator.1", align 8
  %22 = alloca %"class.std::reverse_iterator.1", align 8
  %23 = alloca %"class.std::reverse_iterator.1", align 8
  %24 = alloca %"class.std::reverse_iterator.1", align 8
  %25 = alloca %"class.std::reverse_iterator", align 8
  %26 = alloca %"class.std::reverse_iterator", align 8
  %27 = icmp sgt i32 %0, 1
  br i1 %27, label %28, label %38

28:                                               ; preds = %2
  %29 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %30 = load ptr, ptr %29, align 8, !tbaa !32
  %31 = tail call i64 @__isoc23_strtol(ptr noundef nonnull %30, ptr noundef null, i32 noundef 10) #20
  %32 = trunc i64 %31 to i32
  store i32 %32, ptr @iterations, align 4, !tbaa !11
  %33 = icmp eq i32 %0, 2
  br i1 %33, label %38, label %34

34:                                               ; preds = %28
  %35 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %36 = load ptr, ptr %35, align 8, !tbaa !32
  %37 = tail call double @strtod(ptr noundef nonnull captures(none) %36, ptr noundef null) #20
  store double %37, ptr @init_value, align 8, !tbaa !33
  br label %38

38:                                               ; preds = %2, %34, %28
  %39 = load double, ptr @init_value, align 8, !tbaa !33
  %40 = fptosi double %39 to i32
  %41 = add nsw i32 %40, 123
  tail call void @srand(i32 noundef %41) #20
  %42 = load ptr, ptr @dpb, align 8, !tbaa !34
  %43 = load ptr, ptr @dpe, align 8, !tbaa !34
  %44 = load double, ptr @init_value, align 8, !tbaa !33
  %45 = icmp eq ptr %42, %43
  br i1 %45, label %77, label %46

46:                                               ; preds = %38
  %47 = ptrtoint ptr %43 to i64
  %48 = ptrtoint ptr %42 to i64
  %49 = add i64 %47, -8
  %50 = sub i64 %49, %48
  %51 = lshr i64 %50, 3
  %52 = add nuw nsw i64 %51, 1
  %53 = icmp ult i64 %50, 24
  br i1 %53, label %69, label %54

54:                                               ; preds = %46
  %55 = and i64 %52, 4611686018427387900
  %56 = shl i64 %55, 3
  %57 = getelementptr i8, ptr %42, i64 %56
  %58 = insertelement <2 x double> poison, double %44, i64 0
  %59 = shufflevector <2 x double> %58, <2 x double> poison, <2 x i32> zeroinitializer
  br label %60

60:                                               ; preds = %60, %54
  %61 = phi i64 [ 0, %54 ], [ %65, %60 ]
  %62 = shl i64 %61, 3
  %63 = getelementptr i8, ptr %42, i64 %62
  %64 = getelementptr i8, ptr %63, i64 16
  store <2 x double> %59, ptr %63, align 8, !tbaa !33
  store <2 x double> %59, ptr %64, align 8, !tbaa !33
  %65 = add nuw i64 %61, 4
  %66 = icmp eq i64 %65, %55
  br i1 %66, label %67, label %60, !llvm.loop !36

67:                                               ; preds = %60
  %68 = icmp eq i64 %52, %55
  br i1 %68, label %75, label %69

69:                                               ; preds = %46, %67
  %70 = phi ptr [ %42, %46 ], [ %57, %67 ]
  br label %71

71:                                               ; preds = %69, %71
  %72 = phi ptr [ %73, %71 ], [ %70, %69 ]
  %73 = getelementptr inbounds nuw i8, ptr %72, i64 8
  store double %44, ptr %72, align 8, !tbaa !33
  %74 = icmp eq ptr %73, %43
  br i1 %74, label %75, label %71, !llvm.loop !37

75:                                               ; preds = %71, %67
  %76 = load double, ptr @init_value, align 8, !tbaa !33
  br label %77

77:                                               ; preds = %75, %38
  %78 = phi double [ %76, %75 ], [ %44, %38 ]
  %79 = tail call noalias noundef nonnull dereferenceable(16000) ptr @_Znwm(i64 noundef 16000) #21
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16000) %79, i8 0, i64 16000, i1 false)
  %80 = insertelement <2 x double> poison, double %78, i64 0
  %81 = shufflevector <2 x double> %80, <2 x double> poison, <2 x i32> zeroinitializer
  br label %82

82:                                               ; preds = %82, %77
  %83 = phi i64 [ 0, %77 ], [ %87, %82 ]
  %84 = shl i64 %83, 3
  %85 = getelementptr inbounds nuw i8, ptr %79, i64 %84
  %86 = getelementptr inbounds nuw i8, ptr %85, i64 16
  store <2 x double> %81, ptr %85, align 8, !tbaa !33
  store <2 x double> %81, ptr %86, align 8, !tbaa !33
  %87 = add nuw i64 %83, 4
  %88 = icmp eq i64 %87, 2000
  br i1 %88, label %89, label %82, !llvm.loop !38

89:                                               ; preds = %82
  %90 = getelementptr inbounds nuw i8, ptr %79, i64 16000
  %91 = ptrtoint ptr %79 to i64
  %92 = ptrtoint ptr %90 to i64
  %93 = load i32, ptr @iterations, align 4, !tbaa !11
  %94 = icmp sgt i32 %93, 0
  br i1 %94, label %95, label %328

95:                                               ; preds = %89
  br i1 %45, label %96, label %112

96:                                               ; preds = %95, %107
  %97 = phi i32 [ %108, %107 ], [ %93, %95 ]
  %98 = phi double [ %109, %107 ], [ %78, %95 ]
  %99 = phi i32 [ %110, %107 ], [ 0, %95 ]
  %100 = fmul double %98, 2.000000e+03
  %101 = fcmp une double %100, 0.000000e+00
  br i1 %101, label %102, label %107

102:                                              ; preds = %96
  %103 = load i32, ptr @current_test, align 4, !tbaa !11
  %104 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.51, i32 noundef %103)
  %105 = load double, ptr @init_value, align 8, !tbaa !33
  %106 = load i32, ptr @iterations, align 4, !tbaa !11
  br label %107

107:                                              ; preds = %102, %96
  %108 = phi i32 [ %106, %102 ], [ %97, %96 ]
  %109 = phi double [ %105, %102 ], [ %98, %96 ]
  %110 = add nuw nsw i32 %99, 1
  %111 = icmp slt i32 %110, %108
  br i1 %111, label %96, label %134, !llvm.loop !39

112:                                              ; preds = %95, %130
  %113 = phi i32 [ %131, %130 ], [ %93, %95 ]
  %114 = phi i32 [ %132, %130 ], [ 0, %95 ]
  br label %115

115:                                              ; preds = %115, %112
  %116 = phi double [ %120, %115 ], [ 0.000000e+00, %112 ]
  %117 = phi ptr [ %118, %115 ], [ %42, %112 ]
  %118 = getelementptr inbounds nuw i8, ptr %117, i64 8
  %119 = load double, ptr %117, align 8, !tbaa !33
  %120 = fadd double %116, %119
  %121 = icmp eq ptr %118, %43
  br i1 %121, label %122, label %115, !llvm.loop !40

122:                                              ; preds = %115
  %123 = load double, ptr @init_value, align 8, !tbaa !33
  %124 = fmul double %123, 2.000000e+03
  %125 = fcmp une double %120, %124
  br i1 %125, label %126, label %130

126:                                              ; preds = %122
  %127 = load i32, ptr @current_test, align 4, !tbaa !11
  %128 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.51, i32 noundef %127)
  %129 = load i32, ptr @iterations, align 4, !tbaa !11
  br label %130

130:                                              ; preds = %126, %122
  %131 = phi i32 [ %113, %122 ], [ %129, %126 ]
  %132 = add nuw nsw i32 %114, 1
  %133 = icmp slt i32 %132, %131
  br i1 %133, label %112, label %134, !llvm.loop !39

134:                                              ; preds = %130, %107
  %135 = phi i32 [ %108, %107 ], [ %131, %130 ]
  %136 = icmp sgt i32 %135, 0
  br i1 %136, label %137, label %328

137:                                              ; preds = %134, %160
  %138 = phi i32 [ %161, %160 ], [ %135, %134 ]
  %139 = phi i32 [ %162, %160 ], [ 0, %134 ]
  br label %140

140:                                              ; preds = %140, %137
  %141 = phi i64 [ 0, %137 ], [ %150, %140 ]
  %142 = phi double [ 0.000000e+00, %137 ], [ %149, %140 ]
  %143 = shl i64 %141, 3
  %144 = getelementptr inbounds nuw i8, ptr %79, i64 %143
  %145 = getelementptr inbounds nuw i8, ptr %144, i64 16
  %146 = load <2 x double>, ptr %144, align 8, !tbaa !33
  %147 = load <2 x double>, ptr %145, align 8, !tbaa !33
  %148 = tail call double @llvm.vector.reduce.fadd.v2f64(double %142, <2 x double> %146)
  %149 = tail call double @llvm.vector.reduce.fadd.v2f64(double %148, <2 x double> %147)
  %150 = add nuw i64 %141, 4
  %151 = icmp eq i64 %150, 2000
  br i1 %151, label %152, label %140, !llvm.loop !41

152:                                              ; preds = %140
  %153 = load double, ptr @init_value, align 8, !tbaa !33
  %154 = fmul double %153, 2.000000e+03
  %155 = fcmp une double %149, %154
  br i1 %155, label %156, label %160

156:                                              ; preds = %152
  %157 = load i32, ptr @current_test, align 4, !tbaa !11
  %158 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.51, i32 noundef %157)
  %159 = load i32, ptr @iterations, align 4, !tbaa !11
  br label %160

160:                                              ; preds = %156, %152
  %161 = phi i32 [ %138, %152 ], [ %159, %156 ]
  %162 = add nuw nsw i32 %139, 1
  %163 = icmp slt i32 %162, %161
  br i1 %163, label %137, label %164, !llvm.loop !42

164:                                              ; preds = %160
  %165 = load ptr, ptr @rdpb, align 8, !tbaa !43
  %166 = load ptr, ptr @rdpe, align 8, !tbaa !43
  %167 = icmp sgt i32 %161, 0
  br i1 %167, label %168, label %328

168:                                              ; preds = %164
  %169 = icmp eq ptr %165, %166
  br label %170

170:                                              ; preds = %168, %189
  %171 = phi i32 [ %190, %189 ], [ %161, %168 ]
  %172 = phi i32 [ %191, %189 ], [ 0, %168 ]
  br i1 %169, label %180, label %173

173:                                              ; preds = %170, %173
  %174 = phi ptr [ %176, %173 ], [ %165, %170 ]
  %175 = phi double [ %178, %173 ], [ 0.000000e+00, %170 ]
  %176 = getelementptr inbounds i8, ptr %174, i64 -8
  %177 = load double, ptr %176, align 8, !tbaa !33
  %178 = fadd double %175, %177
  %179 = icmp eq ptr %176, %166
  br i1 %179, label %180, label %173, !llvm.loop !45

180:                                              ; preds = %173, %170
  %181 = phi double [ 0.000000e+00, %170 ], [ %178, %173 ]
  %182 = load double, ptr @init_value, align 8, !tbaa !33
  %183 = fmul double %182, 2.000000e+03
  %184 = fcmp une double %181, %183
  br i1 %184, label %185, label %189

185:                                              ; preds = %180
  %186 = load i32, ptr @current_test, align 4, !tbaa !11
  %187 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.51, i32 noundef %186)
  %188 = load i32, ptr @iterations, align 4, !tbaa !11
  br label %189

189:                                              ; preds = %185, %180
  %190 = phi i32 [ %171, %180 ], [ %188, %185 ]
  %191 = add nuw nsw i32 %172, 1
  %192 = icmp slt i32 %191, %190
  br i1 %192, label %170, label %193, !llvm.loop !46

193:                                              ; preds = %189
  %194 = icmp sgt i32 %190, 0
  br i1 %194, label %195, label %328

195:                                              ; preds = %193, %213
  %196 = phi i32 [ %214, %213 ], [ %190, %193 ]
  %197 = phi i32 [ %215, %213 ], [ 0, %193 ]
  br label %198

198:                                              ; preds = %195, %198
  %199 = phi ptr [ %201, %198 ], [ %90, %195 ]
  %200 = phi double [ %203, %198 ], [ 0.000000e+00, %195 ]
  %201 = getelementptr inbounds i8, ptr %199, i64 -8
  %202 = load double, ptr %201, align 8, !tbaa !33
  %203 = fadd double %200, %202
  %204 = icmp eq ptr %201, %79
  br i1 %204, label %205, label %198, !llvm.loop !47

205:                                              ; preds = %198
  %206 = load double, ptr @init_value, align 8, !tbaa !33
  %207 = fmul double %206, 2.000000e+03
  %208 = fcmp une double %203, %207
  br i1 %208, label %209, label %213

209:                                              ; preds = %205
  %210 = load i32, ptr @current_test, align 4, !tbaa !11
  %211 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.51, i32 noundef %210)
  %212 = load i32, ptr @iterations, align 4, !tbaa !11
  br label %213

213:                                              ; preds = %209, %205
  %214 = phi i32 [ %196, %205 ], [ %212, %209 ]
  %215 = add nuw nsw i32 %197, 1
  %216 = icmp slt i32 %215, %214
  br i1 %216, label %195, label %217, !llvm.loop !48

217:                                              ; preds = %213
  %218 = icmp sgt i32 %214, 0
  br i1 %218, label %219, label %328

219:                                              ; preds = %217, %237
  %220 = phi i32 [ %238, %237 ], [ %214, %217 ]
  %221 = phi i32 [ %239, %237 ], [ 0, %217 ]
  br label %222

222:                                              ; preds = %219, %222
  %223 = phi ptr [ %225, %222 ], [ %90, %219 ]
  %224 = phi double [ %227, %222 ], [ 0.000000e+00, %219 ]
  %225 = getelementptr inbounds i8, ptr %223, i64 -8
  %226 = load double, ptr %225, align 8, !tbaa !33
  %227 = fadd double %224, %226
  %228 = icmp eq ptr %225, %79
  br i1 %228, label %229, label %222, !llvm.loop !47

229:                                              ; preds = %222
  %230 = load double, ptr @init_value, align 8, !tbaa !33
  %231 = fmul double %230, 2.000000e+03
  %232 = fcmp une double %227, %231
  br i1 %232, label %233, label %237

233:                                              ; preds = %229
  %234 = load i32, ptr @current_test, align 4, !tbaa !11
  %235 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.51, i32 noundef %234)
  %236 = load i32, ptr @iterations, align 4, !tbaa !11
  br label %237

237:                                              ; preds = %233, %229
  %238 = phi i32 [ %220, %229 ], [ %236, %233 ]
  %239 = add nuw nsw i32 %221, 1
  %240 = icmp slt i32 %239, %238
  br i1 %240, label %219, label %241, !llvm.loop !48

241:                                              ; preds = %237
  %242 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @rrdpb, i64 8), align 8, !tbaa !43
  %243 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @rrdpe, i64 8), align 8, !tbaa !43
  %244 = icmp sgt i32 %238, 0
  br i1 %244, label %245, label %328

245:                                              ; preds = %241
  %246 = icmp eq ptr %242, %243
  br label %247

247:                                              ; preds = %245, %266
  %248 = phi i32 [ %267, %266 ], [ %238, %245 ]
  %249 = phi i32 [ %268, %266 ], [ 0, %245 ]
  br i1 %246, label %257, label %250

250:                                              ; preds = %247, %250
  %251 = phi ptr [ %253, %250 ], [ %242, %247 ]
  %252 = phi double [ %255, %250 ], [ 0.000000e+00, %247 ]
  %253 = getelementptr inbounds nuw i8, ptr %251, i64 8
  %254 = load double, ptr %251, align 8, !tbaa !33
  %255 = fadd double %252, %254
  %256 = icmp eq ptr %253, %243
  br i1 %256, label %257, label %250, !llvm.loop !49

257:                                              ; preds = %250, %247
  %258 = phi double [ 0.000000e+00, %247 ], [ %255, %250 ]
  %259 = load double, ptr @init_value, align 8, !tbaa !33
  %260 = fmul double %259, 2.000000e+03
  %261 = fcmp une double %258, %260
  br i1 %261, label %262, label %266

262:                                              ; preds = %257
  %263 = load i32, ptr @current_test, align 4, !tbaa !11
  %264 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.51, i32 noundef %263)
  %265 = load i32, ptr @iterations, align 4, !tbaa !11
  br label %266

266:                                              ; preds = %262, %257
  %267 = phi i32 [ %248, %257 ], [ %265, %262 ]
  %268 = add nuw nsw i32 %249, 1
  %269 = icmp slt i32 %268, %267
  br i1 %269, label %247, label %270, !llvm.loop !50

270:                                              ; preds = %266
  %271 = icmp sgt i32 %267, 0
  br i1 %271, label %272, label %328

272:                                              ; preds = %270, %295
  %273 = phi i32 [ %296, %295 ], [ %267, %270 ]
  %274 = phi i32 [ %297, %295 ], [ 0, %270 ]
  br label %275

275:                                              ; preds = %275, %272
  %276 = phi i64 [ 0, %272 ], [ %285, %275 ]
  %277 = phi double [ 0.000000e+00, %272 ], [ %284, %275 ]
  %278 = shl i64 %276, 3
  %279 = getelementptr inbounds nuw i8, ptr %79, i64 %278
  %280 = getelementptr inbounds nuw i8, ptr %279, i64 16
  %281 = load <2 x double>, ptr %279, align 8, !tbaa !33
  %282 = load <2 x double>, ptr %280, align 8, !tbaa !33
  %283 = tail call double @llvm.vector.reduce.fadd.v2f64(double %277, <2 x double> %281)
  %284 = tail call double @llvm.vector.reduce.fadd.v2f64(double %283, <2 x double> %282)
  %285 = add nuw i64 %276, 4
  %286 = icmp eq i64 %285, 2000
  br i1 %286, label %287, label %275, !llvm.loop !51

287:                                              ; preds = %275
  %288 = load double, ptr @init_value, align 8, !tbaa !33
  %289 = fmul double %288, 2.000000e+03
  %290 = fcmp une double %284, %289
  br i1 %290, label %291, label %295

291:                                              ; preds = %287
  %292 = load i32, ptr @current_test, align 4, !tbaa !11
  %293 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.51, i32 noundef %292)
  %294 = load i32, ptr @iterations, align 4, !tbaa !11
  br label %295

295:                                              ; preds = %291, %287
  %296 = phi i32 [ %273, %287 ], [ %294, %291 ]
  %297 = add nuw nsw i32 %274, 1
  %298 = icmp slt i32 %297, %296
  br i1 %298, label %272, label %299, !llvm.loop !52

299:                                              ; preds = %295
  %300 = icmp sgt i32 %296, 0
  br i1 %300, label %301, label %328

301:                                              ; preds = %299, %324
  %302 = phi i32 [ %325, %324 ], [ %296, %299 ]
  %303 = phi i32 [ %326, %324 ], [ 0, %299 ]
  br label %304

304:                                              ; preds = %304, %301
  %305 = phi i64 [ 0, %301 ], [ %314, %304 ]
  %306 = phi double [ 0.000000e+00, %301 ], [ %313, %304 ]
  %307 = shl i64 %305, 3
  %308 = getelementptr inbounds nuw i8, ptr %79, i64 %307
  %309 = getelementptr inbounds nuw i8, ptr %308, i64 16
  %310 = load <2 x double>, ptr %308, align 8, !tbaa !33
  %311 = load <2 x double>, ptr %309, align 8, !tbaa !33
  %312 = tail call double @llvm.vector.reduce.fadd.v2f64(double %306, <2 x double> %310)
  %313 = tail call double @llvm.vector.reduce.fadd.v2f64(double %312, <2 x double> %311)
  %314 = add nuw i64 %305, 4
  %315 = icmp eq i64 %314, 2000
  br i1 %315, label %316, label %304, !llvm.loop !53

316:                                              ; preds = %304
  %317 = load double, ptr @init_value, align 8, !tbaa !33
  %318 = fmul double %317, 2.000000e+03
  %319 = fcmp une double %313, %318
  br i1 %319, label %320, label %324

320:                                              ; preds = %316
  %321 = load i32, ptr @current_test, align 4, !tbaa !11
  %322 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.51, i32 noundef %321)
  %323 = load i32, ptr @iterations, align 4, !tbaa !11
  br label %324

324:                                              ; preds = %320, %316
  %325 = phi i32 [ %302, %316 ], [ %323, %320 ]
  %326 = add nuw nsw i32 %303, 1
  %327 = icmp slt i32 %326, %325
  br i1 %327, label %301, label %328, !llvm.loop !52

328:                                              ; preds = %324, %89, %134, %164, %217, %193, %241, %270, %299
  %329 = phi i32 [ %214, %217 ], [ %190, %193 ], [ %238, %241 ], [ %267, %270 ], [ %296, %299 ], [ %161, %164 ], [ %93, %89 ], [ %135, %134 ], [ %325, %324 ]
  %330 = sdiv i32 %329, 1000
  store i32 %330, ptr @iterations, align 4, !tbaa !11
  %331 = invoke noalias noundef nonnull dereferenceable(16000) ptr @_Znwm(i64 noundef 16000) #21
          to label %332 unwind label %1214

332:                                              ; preds = %328
  %333 = ptrtoint ptr %331 to i64
  %334 = getelementptr inbounds nuw i8, ptr %331, i64 16000
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16000) %331, i8 0, i64 16000, i1 false)
  %335 = load ptr, ptr @dMpb, align 8, !tbaa !34
  %336 = load ptr, ptr @dMpe, align 8, !tbaa !34
  %337 = icmp eq ptr %335, %336
  br i1 %337, label %347, label %338

338:                                              ; preds = %332, %338
  %339 = phi ptr [ %342, %338 ], [ %335, %332 ]
  %340 = tail call i32 @rand() #20
  %341 = sitofp i32 %340 to double
  %342 = getelementptr inbounds nuw i8, ptr %339, i64 8
  store double %341, ptr %339, align 8, !tbaa !33
  %343 = icmp eq ptr %342, %336
  br i1 %343, label %344, label %338, !llvm.loop !54

344:                                              ; preds = %338
  %345 = load ptr, ptr @dMpb, align 8, !tbaa !34
  %346 = load ptr, ptr @dMpe, align 8, !tbaa !34
  br label %347

347:                                              ; preds = %344, %332
  %348 = phi ptr [ %346, %344 ], [ %336, %332 ]
  %349 = phi ptr [ %345, %344 ], [ %335, %332 ]
  %350 = icmp eq ptr %349, %348
  br i1 %350, label %392, label %351

351:                                              ; preds = %347
  %352 = ptrtoint ptr %349 to i64
  %353 = ptrtoint ptr %348 to i64
  %354 = add i64 %353, -8
  %355 = sub i64 %354, %352
  %356 = lshr i64 %355, 3
  %357 = add nuw nsw i64 %356, 1
  %358 = icmp ult i64 %355, 24
  %359 = sub i64 %333, %352
  %360 = icmp ult i64 %359, 32
  %361 = or i1 %358, %360
  br i1 %361, label %382, label %362

362:                                              ; preds = %351
  %363 = and i64 %357, 4611686018427387900
  %364 = shl i64 %363, 3
  %365 = getelementptr i8, ptr %349, i64 %364
  %366 = shl i64 %363, 3
  %367 = getelementptr i8, ptr %331, i64 %366
  br label %368

368:                                              ; preds = %368, %362
  %369 = phi i64 [ 0, %362 ], [ %378, %368 ]
  %370 = shl i64 %369, 3
  %371 = getelementptr i8, ptr %349, i64 %370
  %372 = shl i64 %369, 3
  %373 = getelementptr i8, ptr %331, i64 %372
  %374 = getelementptr i8, ptr %371, i64 16
  %375 = load <2 x double>, ptr %371, align 8, !tbaa !33
  %376 = load <2 x double>, ptr %374, align 8, !tbaa !33
  %377 = getelementptr i8, ptr %373, i64 16
  store <2 x double> %375, ptr %373, align 8, !tbaa !33
  store <2 x double> %376, ptr %377, align 8, !tbaa !33
  %378 = add nuw i64 %369, 4
  %379 = icmp eq i64 %378, %363
  br i1 %379, label %380, label %368, !llvm.loop !55

380:                                              ; preds = %368
  %381 = icmp eq i64 %357, %363
  br i1 %381, label %392, label %382

382:                                              ; preds = %351, %380
  %383 = phi ptr [ %349, %351 ], [ %365, %380 ]
  %384 = phi ptr [ %331, %351 ], [ %367, %380 ]
  br label %385

385:                                              ; preds = %382, %385
  %386 = phi ptr [ %388, %385 ], [ %383, %382 ]
  %387 = phi ptr [ %390, %385 ], [ %384, %382 ]
  %388 = getelementptr inbounds nuw i8, ptr %386, i64 8
  %389 = load double, ptr %386, align 8, !tbaa !33
  %390 = getelementptr inbounds nuw i8, ptr %387, i64 8
  store double %389, ptr %387, align 8, !tbaa !33
  %391 = icmp eq ptr %388, %348
  br i1 %391, label %392, label %385, !llvm.loop !56

392:                                              ; preds = %385, %380, %347
  %393 = load ptr, ptr @dpb, align 8, !tbaa !34
  %394 = load ptr, ptr @dpe, align 8, !tbaa !34
  invoke void @_Z19test_insertion_sortIPddEvT_S1_S1_S1_T0_PKc(ptr noundef %349, ptr noundef %348, ptr noundef %393, ptr noundef %394, double noundef 0.000000e+00, ptr noundef nonnull @.str.26)
          to label %395 unwind label %1240

395:                                              ; preds = %392
  invoke void @_Z19test_insertion_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEvT_S7_S7_S7_T0_PKc(ptr nonnull %331, ptr nonnull %334, ptr nonnull %79, ptr nonnull %90, double noundef 0.000000e+00, ptr noundef nonnull @.str.27)
          to label %396 unwind label %1240

396:                                              ; preds = %395
  %397 = load ptr, ptr @rdMpb, align 8, !tbaa !43
  %398 = load ptr, ptr @rdMpe, align 8, !tbaa !43
  %399 = load ptr, ptr @rdpb, align 8, !tbaa !43
  %400 = load ptr, ptr @rdpe, align 8, !tbaa !43
  %401 = load i32, ptr @iterations, align 4, !tbaa !11
  %402 = icmp sgt i32 %401, 0
  br i1 %402, label %403, label %729

403:                                              ; preds = %396
  %404 = ptrtoint ptr %399 to i64
  %405 = ptrtoint ptr %398 to i64
  %406 = ptrtoint ptr %397 to i64
  %407 = icmp eq ptr %397, %398
  %408 = getelementptr inbounds i8, ptr %399, i64 -8
  %409 = icmp eq ptr %408, %400
  %410 = sub i64 %406, %404
  %411 = add i64 %406, -8
  %412 = sub i64 %411, %405
  %413 = lshr i64 %412, 3
  %414 = add nuw nsw i64 %413, 1
  %415 = icmp ult i64 %412, 24
  %416 = icmp ult i64 %410, 32
  %417 = select i1 %415, i1 true, i1 %416
  %418 = and i64 %414, 4611686018427387900
  %419 = mul i64 %418, -8
  %420 = getelementptr i8, ptr %399, i64 %419
  %421 = mul i64 %418, -8
  %422 = getelementptr i8, ptr %397, i64 %421
  %423 = icmp eq i64 %414, %418
  br label %424

424:                                              ; preds = %403, %485
  %425 = phi i32 [ %486, %485 ], [ %401, %403 ]
  %426 = phi i32 [ %487, %485 ], [ 0, %403 ]
  br i1 %407, label %453, label %427

427:                                              ; preds = %424
  br i1 %417, label %443, label %428

428:                                              ; preds = %427, %428
  %429 = phi i64 [ %440, %428 ], [ 0, %427 ]
  %430 = mul i64 %429, -8
  %431 = getelementptr i8, ptr %399, i64 %430
  %432 = mul i64 %429, -8
  %433 = getelementptr i8, ptr %397, i64 %432
  %434 = getelementptr inbounds i8, ptr %433, i64 -16
  %435 = getelementptr inbounds i8, ptr %433, i64 -32
  %436 = load <2 x double>, ptr %434, align 8, !tbaa !33
  %437 = load <2 x double>, ptr %435, align 8, !tbaa !33
  %438 = getelementptr inbounds i8, ptr %431, i64 -16
  %439 = getelementptr inbounds i8, ptr %431, i64 -32
  store <2 x double> %436, ptr %438, align 8, !tbaa !33
  store <2 x double> %437, ptr %439, align 8, !tbaa !33
  %440 = add nuw i64 %429, 4
  %441 = icmp eq i64 %440, %418
  br i1 %441, label %442, label %428, !llvm.loop !57

442:                                              ; preds = %428
  br i1 %423, label %453, label %443

443:                                              ; preds = %427, %442
  %444 = phi ptr [ %399, %427 ], [ %420, %442 ]
  %445 = phi ptr [ %397, %427 ], [ %422, %442 ]
  br label %446

446:                                              ; preds = %443, %446
  %447 = phi ptr [ %451, %446 ], [ %444, %443 ]
  %448 = phi ptr [ %449, %446 ], [ %445, %443 ]
  %449 = getelementptr inbounds i8, ptr %448, i64 -8
  %450 = load double, ptr %449, align 8, !tbaa !33
  %451 = getelementptr inbounds i8, ptr %447, i64 -8
  store double %450, ptr %451, align 8, !tbaa !33
  %452 = icmp eq ptr %449, %398
  br i1 %452, label %453, label %446, !llvm.loop !58

453:                                              ; preds = %446, %442, %424
  br i1 %409, label %470, label %454

454:                                              ; preds = %453, %466
  %455 = phi ptr [ %456, %466 ], [ %408, %453 ]
  %456 = getelementptr inbounds i8, ptr %455, i64 -8
  %457 = load double, ptr %456, align 8, !tbaa !33
  br label %458

458:                                              ; preds = %462, %454
  %459 = phi ptr [ %455, %454 ], [ %463, %462 ]
  %460 = load double, ptr %459, align 8, !tbaa !33
  %461 = fcmp olt double %457, %460
  br i1 %461, label %462, label %466

462:                                              ; preds = %458
  %463 = getelementptr i8, ptr %459, i64 8
  %464 = getelementptr inbounds i8, ptr %459, i64 -8
  store double %460, ptr %464, align 8, !tbaa !33
  %465 = icmp eq ptr %463, %399
  br i1 %465, label %466, label %458, !llvm.loop !59

466:                                              ; preds = %462, %458
  %467 = phi ptr [ %399, %462 ], [ %459, %458 ]
  %468 = getelementptr inbounds i8, ptr %467, i64 -8
  store double %457, ptr %468, align 8, !tbaa !33
  %469 = icmp eq ptr %456, %400
  br i1 %469, label %470, label %454, !llvm.loop !60

470:                                              ; preds = %466, %453
  br label %471

471:                                              ; preds = %470, %475
  %472 = phi ptr [ %476, %475 ], [ %408, %470 ]
  %473 = phi ptr [ %478, %475 ], [ %399, %470 ]
  %474 = icmp eq ptr %472, %400
  br i1 %474, label %485, label %475

475:                                              ; preds = %471
  %476 = getelementptr inbounds i8, ptr %472, i64 -8
  %477 = load double, ptr %476, align 8, !tbaa !33
  %478 = getelementptr inbounds i8, ptr %473, i64 -8
  %479 = load double, ptr %478, align 8, !tbaa !33
  %480 = fcmp olt double %477, %479
  br i1 %480, label %481, label %471, !llvm.loop !61

481:                                              ; preds = %475
  %482 = load i32, ptr @current_test, align 4, !tbaa !11
  %483 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.52, i32 noundef %482)
  %484 = load i32, ptr @iterations, align 4, !tbaa !11
  br label %485

485:                                              ; preds = %471, %481
  %486 = phi i32 [ %484, %481 ], [ %425, %471 ]
  %487 = add nuw nsw i32 %426, 1
  %488 = icmp slt i32 %487, %486
  br i1 %488, label %424, label %489, !llvm.loop !62

489:                                              ; preds = %485
  %490 = icmp sgt i32 %486, 0
  br i1 %490, label %491, label %729

491:                                              ; preds = %489
  %492 = getelementptr inbounds nuw i8, ptr %79, i64 15992
  br label %493

493:                                              ; preds = %491, %526
  %494 = phi i32 [ %527, %526 ], [ %486, %491 ]
  %495 = phi i32 [ %528, %526 ], [ 0, %491 ]
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(16000) %79, ptr noundef nonnull align 8 dereferenceable(16000) %331, i64 16000, i1 false), !tbaa !33
  br label %496

496:                                              ; preds = %493, %508
  %497 = phi ptr [ %498, %508 ], [ %492, %493 ]
  %498 = getelementptr inbounds i8, ptr %497, i64 -8
  %499 = load double, ptr %498, align 8, !tbaa !33
  br label %500

500:                                              ; preds = %504, %496
  %501 = phi ptr [ %505, %504 ], [ %497, %496 ]
  %502 = load double, ptr %501, align 8, !tbaa !33
  %503 = fcmp olt double %499, %502
  br i1 %503, label %504, label %508

504:                                              ; preds = %500
  %505 = getelementptr i8, ptr %501, i64 8
  %506 = getelementptr inbounds i8, ptr %501, i64 -8
  store double %502, ptr %506, align 8, !tbaa !33
  %507 = icmp eq ptr %505, %90
  br i1 %507, label %508, label %500, !llvm.loop !63

508:                                              ; preds = %504, %500
  %509 = phi ptr [ %501, %500 ], [ %90, %504 ]
  %510 = getelementptr inbounds i8, ptr %509, i64 -8
  store double %499, ptr %510, align 8, !tbaa !33
  %511 = icmp eq ptr %498, %79
  br i1 %511, label %512, label %496, !llvm.loop !64

512:                                              ; preds = %508, %516
  %513 = phi ptr [ %517, %516 ], [ %492, %508 ]
  %514 = phi ptr [ %519, %516 ], [ %90, %508 ]
  %515 = icmp eq ptr %513, %79
  br i1 %515, label %526, label %516

516:                                              ; preds = %512
  %517 = getelementptr inbounds i8, ptr %513, i64 -8
  %518 = load double, ptr %517, align 8, !tbaa !33
  %519 = getelementptr inbounds i8, ptr %514, i64 -8
  %520 = load double, ptr %519, align 8, !tbaa !33
  %521 = fcmp olt double %518, %520
  br i1 %521, label %522, label %512, !llvm.loop !65

522:                                              ; preds = %516
  %523 = load i32, ptr @current_test, align 4, !tbaa !11
  %524 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.52, i32 noundef %523)
  %525 = load i32, ptr @iterations, align 4, !tbaa !11
  br label %526

526:                                              ; preds = %512, %522
  %527 = phi i32 [ %525, %522 ], [ %494, %512 ]
  %528 = add nuw nsw i32 %495, 1
  %529 = icmp slt i32 %528, %527
  br i1 %529, label %493, label %530, !llvm.loop !66

530:                                              ; preds = %526
  %531 = icmp sgt i32 %527, 0
  br i1 %531, label %532, label %729

532:                                              ; preds = %530
  %533 = getelementptr inbounds nuw i8, ptr %79, i64 15992
  br label %534

534:                                              ; preds = %532, %567
  %535 = phi i32 [ %568, %567 ], [ %527, %532 ]
  %536 = phi i32 [ %569, %567 ], [ 0, %532 ]
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(16000) %79, ptr noundef nonnull align 8 dereferenceable(16000) %331, i64 16000, i1 false), !tbaa !33
  br label %537

537:                                              ; preds = %534, %549
  %538 = phi ptr [ %539, %549 ], [ %533, %534 ]
  %539 = getelementptr inbounds i8, ptr %538, i64 -8
  %540 = load double, ptr %539, align 8, !tbaa !33
  br label %541

541:                                              ; preds = %545, %537
  %542 = phi ptr [ %546, %545 ], [ %538, %537 ]
  %543 = load double, ptr %542, align 8, !tbaa !33
  %544 = fcmp olt double %540, %543
  br i1 %544, label %545, label %549

545:                                              ; preds = %541
  %546 = getelementptr i8, ptr %542, i64 8
  %547 = getelementptr inbounds i8, ptr %542, i64 -8
  store double %543, ptr %547, align 8, !tbaa !33
  %548 = icmp eq ptr %546, %90
  br i1 %548, label %549, label %541, !llvm.loop !63

549:                                              ; preds = %545, %541
  %550 = phi ptr [ %542, %541 ], [ %90, %545 ]
  %551 = getelementptr inbounds i8, ptr %550, i64 -8
  store double %540, ptr %551, align 8, !tbaa !33
  %552 = icmp eq ptr %539, %79
  br i1 %552, label %553, label %537, !llvm.loop !64

553:                                              ; preds = %549, %557
  %554 = phi ptr [ %558, %557 ], [ %533, %549 ]
  %555 = phi ptr [ %560, %557 ], [ %90, %549 ]
  %556 = icmp eq ptr %554, %79
  br i1 %556, label %567, label %557

557:                                              ; preds = %553
  %558 = getelementptr inbounds i8, ptr %554, i64 -8
  %559 = load double, ptr %558, align 8, !tbaa !33
  %560 = getelementptr inbounds i8, ptr %555, i64 -8
  %561 = load double, ptr %560, align 8, !tbaa !33
  %562 = fcmp olt double %559, %561
  br i1 %562, label %563, label %553, !llvm.loop !65

563:                                              ; preds = %557
  %564 = load i32, ptr @current_test, align 4, !tbaa !11
  %565 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.52, i32 noundef %564)
  %566 = load i32, ptr @iterations, align 4, !tbaa !11
  br label %567

567:                                              ; preds = %553, %563
  %568 = phi i32 [ %566, %563 ], [ %535, %553 ]
  %569 = add nuw nsw i32 %536, 1
  %570 = icmp slt i32 %569, %568
  br i1 %570, label %534, label %571, !llvm.loop !66

571:                                              ; preds = %567
  %572 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @rrdMpb, i64 8), align 8, !tbaa !43
  %573 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @rrdMpe, i64 8), align 8, !tbaa !43
  %574 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @rrdpb, i64 8), align 8, !tbaa !43
  %575 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @rrdpe, i64 8), align 8, !tbaa !43
  %576 = icmp sgt i32 %568, 0
  br i1 %576, label %577, label %729

577:                                              ; preds = %571
  %578 = ptrtoint ptr %574 to i64
  %579 = ptrtoint ptr %573 to i64
  %580 = ptrtoint ptr %572 to i64
  %581 = icmp eq ptr %572, %573
  %582 = getelementptr inbounds nuw i8, ptr %574, i64 8
  %583 = icmp eq ptr %582, %575
  %584 = sub i64 %578, %580
  %585 = add i64 %579, -8
  %586 = sub i64 %585, %580
  %587 = lshr i64 %586, 3
  %588 = add nuw nsw i64 %587, 1
  %589 = icmp ult i64 %586, 24
  %590 = icmp ult i64 %584, 32
  %591 = select i1 %589, i1 true, i1 %590
  %592 = and i64 %588, 4611686018427387900
  %593 = shl i64 %592, 3
  %594 = getelementptr i8, ptr %574, i64 %593
  %595 = shl i64 %592, 3
  %596 = getelementptr i8, ptr %572, i64 %595
  %597 = icmp eq i64 %588, %592
  br label %598

598:                                              ; preds = %577, %653
  %599 = phi i32 [ %654, %653 ], [ %568, %577 ]
  %600 = phi i32 [ %655, %653 ], [ 0, %577 ]
  br i1 %581, label %625, label %601

601:                                              ; preds = %598
  br i1 %591, label %615, label %602

602:                                              ; preds = %601, %602
  %603 = phi i64 [ %612, %602 ], [ 0, %601 ]
  %604 = shl i64 %603, 3
  %605 = getelementptr i8, ptr %574, i64 %604
  %606 = shl i64 %603, 3
  %607 = getelementptr i8, ptr %572, i64 %606
  %608 = getelementptr i8, ptr %607, i64 16
  %609 = load <2 x double>, ptr %607, align 8, !tbaa !33
  %610 = load <2 x double>, ptr %608, align 8, !tbaa !33
  %611 = getelementptr i8, ptr %605, i64 16
  store <2 x double> %609, ptr %605, align 8, !tbaa !33
  store <2 x double> %610, ptr %611, align 8, !tbaa !33
  %612 = add nuw i64 %603, 4
  %613 = icmp eq i64 %612, %592
  br i1 %613, label %614, label %602, !llvm.loop !67

614:                                              ; preds = %602
  br i1 %597, label %625, label %615

615:                                              ; preds = %601, %614
  %616 = phi ptr [ %574, %601 ], [ %594, %614 ]
  %617 = phi ptr [ %572, %601 ], [ %596, %614 ]
  br label %618

618:                                              ; preds = %615, %618
  %619 = phi ptr [ %623, %618 ], [ %616, %615 ]
  %620 = phi ptr [ %621, %618 ], [ %617, %615 ]
  %621 = getelementptr inbounds nuw i8, ptr %620, i64 8
  %622 = load double, ptr %620, align 8, !tbaa !33
  %623 = getelementptr inbounds nuw i8, ptr %619, i64 8
  store double %622, ptr %619, align 8, !tbaa !33
  %624 = icmp eq ptr %621, %573
  br i1 %624, label %625, label %618, !llvm.loop !68

625:                                              ; preds = %618, %614, %598
  br i1 %583, label %640, label %626

626:                                              ; preds = %625, %636
  %627 = phi ptr [ %638, %636 ], [ %582, %625 ]
  %628 = load double, ptr %627, align 8, !tbaa !33
  br label %629

629:                                              ; preds = %634, %626
  %630 = phi ptr [ %627, %626 ], [ %631, %634 ]
  %631 = getelementptr i8, ptr %630, i64 -8
  %632 = load double, ptr %631, align 8, !tbaa !33
  %633 = fcmp olt double %628, %632
  br i1 %633, label %634, label %636

634:                                              ; preds = %629
  store double %632, ptr %630, align 8, !tbaa !33
  %635 = icmp eq ptr %631, %574
  br i1 %635, label %636, label %629, !llvm.loop !69

636:                                              ; preds = %634, %629
  %637 = phi ptr [ %574, %634 ], [ %630, %629 ]
  store double %628, ptr %637, align 8, !tbaa !33
  %638 = getelementptr inbounds nuw i8, ptr %627, i64 8
  %639 = icmp eq ptr %638, %575
  br i1 %639, label %640, label %626, !llvm.loop !70

640:                                              ; preds = %636, %625
  br label %641

641:                                              ; preds = %640, %645
  %642 = phi ptr [ %643, %645 ], [ %574, %640 ]
  %643 = getelementptr i8, ptr %642, i64 8
  %644 = icmp eq ptr %643, %575
  br i1 %644, label %653, label %645

645:                                              ; preds = %641
  %646 = load double, ptr %643, align 8, !tbaa !33
  %647 = load double, ptr %642, align 8, !tbaa !33
  %648 = fcmp olt double %646, %647
  br i1 %648, label %649, label %641, !llvm.loop !71

649:                                              ; preds = %645
  %650 = load i32, ptr @current_test, align 4, !tbaa !11
  %651 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.52, i32 noundef %650)
  %652 = load i32, ptr @iterations, align 4, !tbaa !11
  br label %653

653:                                              ; preds = %641, %649
  %654 = phi i32 [ %652, %649 ], [ %599, %641 ]
  %655 = add nuw nsw i32 %600, 1
  %656 = icmp slt i32 %655, %654
  br i1 %656, label %598, label %657, !llvm.loop !72

657:                                              ; preds = %653
  %658 = icmp sgt i32 %654, 0
  br i1 %658, label %659, label %729

659:                                              ; preds = %657, %689
  %660 = phi i32 [ %690, %689 ], [ %654, %657 ]
  %661 = phi i32 [ %691, %689 ], [ 0, %657 ]
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(16000) %79, ptr noundef nonnull align 8 dereferenceable(16000) %331, i64 16000, i1 false), !tbaa !33
  br label %662

662:                                              ; preds = %659, %673
  %663 = phi i64 [ %675, %673 ], [ 8, %659 ]
  %664 = getelementptr inbounds nuw i8, ptr %79, i64 %663
  %665 = load double, ptr %664, align 8, !tbaa !33
  br label %666

666:                                              ; preds = %662, %671
  %667 = phi ptr [ %668, %671 ], [ %664, %662 ]
  %668 = getelementptr i8, ptr %667, i64 -8
  %669 = load double, ptr %668, align 8, !tbaa !33
  %670 = fcmp olt double %665, %669
  br i1 %670, label %671, label %673

671:                                              ; preds = %666
  store double %669, ptr %667, align 8, !tbaa !33
  %672 = icmp eq ptr %79, %668
  br i1 %672, label %673, label %666, !llvm.loop !73

673:                                              ; preds = %671, %666
  %674 = phi ptr [ %667, %666 ], [ %79, %671 ]
  store double %665, ptr %674, align 8, !tbaa !33
  %675 = add nuw nsw i64 %663, 8
  %676 = icmp eq i64 %675, 16000
  br i1 %676, label %677, label %662, !llvm.loop !74

677:                                              ; preds = %673, %681
  %678 = phi ptr [ %679, %681 ], [ %79, %673 ]
  %679 = getelementptr i8, ptr %678, i64 8
  %680 = icmp eq ptr %90, %679
  br i1 %680, label %689, label %681

681:                                              ; preds = %677
  %682 = load double, ptr %679, align 8, !tbaa !33
  %683 = load double, ptr %678, align 8, !tbaa !33
  %684 = fcmp olt double %682, %683
  br i1 %684, label %685, label %677, !llvm.loop !75

685:                                              ; preds = %681
  %686 = load i32, ptr @current_test, align 4, !tbaa !11
  %687 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.52, i32 noundef %686)
  %688 = load i32, ptr @iterations, align 4, !tbaa !11
  br label %689

689:                                              ; preds = %677, %685
  %690 = phi i32 [ %688, %685 ], [ %660, %677 ]
  %691 = add nuw nsw i32 %661, 1
  %692 = icmp slt i32 %691, %690
  br i1 %692, label %659, label %693, !llvm.loop !76

693:                                              ; preds = %689
  %694 = icmp sgt i32 %690, 0
  br i1 %694, label %695, label %729

695:                                              ; preds = %693, %725
  %696 = phi i32 [ %726, %725 ], [ %690, %693 ]
  %697 = phi i32 [ %727, %725 ], [ 0, %693 ]
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(16000) %79, ptr noundef nonnull align 8 dereferenceable(16000) %331, i64 16000, i1 false), !tbaa !33
  br label %698

698:                                              ; preds = %695, %709
  %699 = phi i64 [ %711, %709 ], [ 8, %695 ]
  %700 = getelementptr inbounds nuw i8, ptr %79, i64 %699
  %701 = load double, ptr %700, align 8, !tbaa !33
  br label %702

702:                                              ; preds = %698, %707
  %703 = phi ptr [ %704, %707 ], [ %700, %698 ]
  %704 = getelementptr i8, ptr %703, i64 -8
  %705 = load double, ptr %704, align 8, !tbaa !33
  %706 = fcmp olt double %701, %705
  br i1 %706, label %707, label %709

707:                                              ; preds = %702
  store double %705, ptr %703, align 8, !tbaa !33
  %708 = icmp eq ptr %79, %704
  br i1 %708, label %709, label %702, !llvm.loop !73

709:                                              ; preds = %707, %702
  %710 = phi ptr [ %703, %702 ], [ %79, %707 ]
  store double %701, ptr %710, align 8, !tbaa !33
  %711 = add nuw nsw i64 %699, 8
  %712 = icmp eq i64 %711, 16000
  br i1 %712, label %713, label %698, !llvm.loop !74

713:                                              ; preds = %709, %717
  %714 = phi ptr [ %715, %717 ], [ %79, %709 ]
  %715 = getelementptr i8, ptr %714, i64 8
  %716 = icmp eq ptr %90, %715
  br i1 %716, label %725, label %717

717:                                              ; preds = %713
  %718 = load double, ptr %715, align 8, !tbaa !33
  %719 = load double, ptr %714, align 8, !tbaa !33
  %720 = fcmp olt double %718, %719
  br i1 %720, label %721, label %713, !llvm.loop !75

721:                                              ; preds = %717
  %722 = load i32, ptr @current_test, align 4, !tbaa !11
  %723 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.52, i32 noundef %722)
  %724 = load i32, ptr @iterations, align 4, !tbaa !11
  br label %725

725:                                              ; preds = %713, %721
  %726 = phi i32 [ %724, %721 ], [ %696, %713 ]
  %727 = add nuw nsw i32 %697, 1
  %728 = icmp slt i32 %727, %726
  br i1 %728, label %695, label %729, !llvm.loop !76

729:                                              ; preds = %725, %571, %489, %396, %530, %657, %693
  %730 = phi i32 [ %527, %530 ], [ %654, %657 ], [ %690, %693 ], [ %486, %489 ], [ %401, %396 ], [ %568, %571 ], [ %726, %725 ]
  %731 = shl nsw i32 %730, 3
  store i32 %731, ptr @iterations, align 4, !tbaa !11
  %732 = load ptr, ptr @dMpb, align 8, !tbaa !34
  %733 = load ptr, ptr @dMpe, align 8, !tbaa !34
  %734 = load ptr, ptr @dpb, align 8, !tbaa !34
  %735 = load ptr, ptr @dpe, align 8, !tbaa !34
  invoke void @_Z14test_quicksortIPddEvT_S1_S1_S1_T0_PKc(ptr noundef %732, ptr noundef %733, ptr noundef %734, ptr noundef %735, double noundef 0.000000e+00, ptr noundef nonnull @.str.34)
          to label %736 unwind label %1240

736:                                              ; preds = %729
  invoke void @_Z14test_quicksortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEvT_S7_S7_S7_T0_PKc(ptr nonnull %331, ptr nonnull %334, ptr nonnull %79, ptr nonnull %90, double noundef 0.000000e+00, ptr noundef nonnull @.str.35)
          to label %737 unwind label %1240

737:                                              ; preds = %736
  %738 = load ptr, ptr @rdMpb, align 8, !tbaa !43
  %739 = load ptr, ptr @rdMpe, align 8, !tbaa !43
  %740 = load ptr, ptr @rdpb, align 8, !tbaa !43
  %741 = load ptr, ptr @rdpe, align 8, !tbaa !43
  call void @llvm.lifetime.start.p0(ptr nonnull %25)
  call void @llvm.lifetime.start.p0(ptr nonnull %26)
  %742 = load i32, ptr @iterations, align 4, !tbaa !11
  %743 = icmp sgt i32 %742, 0
  br i1 %743, label %745, label %744

744:                                              ; preds = %737
  call void @llvm.lifetime.end.p0(ptr nonnull %25)
  call void @llvm.lifetime.end.p0(ptr nonnull %26)
  call void @llvm.lifetime.start.p0(ptr nonnull %23)
  call void @llvm.lifetime.start.p0(ptr nonnull %24)
  br label %815

745:                                              ; preds = %737
  %746 = ptrtoint ptr %740 to i64
  %747 = ptrtoint ptr %739 to i64
  %748 = ptrtoint ptr %738 to i64
  %749 = icmp eq ptr %738, %739
  %750 = getelementptr inbounds i8, ptr %740, i64 -8
  %751 = sub i64 %748, %746
  %752 = add i64 %748, -8
  %753 = sub i64 %752, %747
  %754 = lshr i64 %753, 3
  %755 = add nuw nsw i64 %754, 1
  %756 = icmp ult i64 %753, 24
  %757 = icmp ult i64 %751, 32
  %758 = select i1 %756, i1 true, i1 %757
  %759 = and i64 %755, 4611686018427387900
  %760 = mul i64 %759, -8
  %761 = getelementptr i8, ptr %740, i64 %760
  %762 = mul i64 %759, -8
  %763 = getelementptr i8, ptr %738, i64 %762
  %764 = icmp eq i64 %755, %759
  br label %765

765:                                              ; preds = %745, %807
  %766 = phi i32 [ %808, %807 ], [ 0, %745 ]
  br i1 %749, label %793, label %767

767:                                              ; preds = %765
  br i1 %758, label %783, label %768

768:                                              ; preds = %767, %768
  %769 = phi i64 [ %780, %768 ], [ 0, %767 ]
  %770 = mul i64 %769, -8
  %771 = getelementptr i8, ptr %740, i64 %770
  %772 = mul i64 %769, -8
  %773 = getelementptr i8, ptr %738, i64 %772
  %774 = getelementptr inbounds i8, ptr %773, i64 -16
  %775 = getelementptr inbounds i8, ptr %773, i64 -32
  %776 = load <2 x double>, ptr %774, align 8, !tbaa !33
  %777 = load <2 x double>, ptr %775, align 8, !tbaa !33
  %778 = getelementptr inbounds i8, ptr %771, i64 -16
  %779 = getelementptr inbounds i8, ptr %771, i64 -32
  store <2 x double> %776, ptr %778, align 8, !tbaa !33
  store <2 x double> %777, ptr %779, align 8, !tbaa !33
  %780 = add nuw i64 %769, 4
  %781 = icmp eq i64 %780, %759
  br i1 %781, label %782, label %768, !llvm.loop !77

782:                                              ; preds = %768
  br i1 %764, label %793, label %783

783:                                              ; preds = %767, %782
  %784 = phi ptr [ %740, %767 ], [ %761, %782 ]
  %785 = phi ptr [ %738, %767 ], [ %763, %782 ]
  br label %786

786:                                              ; preds = %783, %786
  %787 = phi ptr [ %791, %786 ], [ %784, %783 ]
  %788 = phi ptr [ %789, %786 ], [ %785, %783 ]
  %789 = getelementptr inbounds i8, ptr %788, i64 -8
  %790 = load double, ptr %789, align 8, !tbaa !33
  %791 = getelementptr inbounds i8, ptr %787, i64 -8
  store double %790, ptr %791, align 8, !tbaa !33
  %792 = icmp eq ptr %789, %739
  br i1 %792, label %793, label %786, !llvm.loop !78

793:                                              ; preds = %786, %782, %765
  store ptr %740, ptr %25, align 8, !tbaa !43
  store ptr %741, ptr %26, align 8, !tbaa !43
  invoke void @_ZN9benchmark9quicksortISt16reverse_iteratorIPdEdEEvT_S4_(ptr dead_on_return noundef nonnull %25, ptr dead_on_return noundef nonnull %26)
          to label %794 unwind label %1238

794:                                              ; preds = %793, %798
  %795 = phi ptr [ %799, %798 ], [ %750, %793 ]
  %796 = phi ptr [ %801, %798 ], [ %740, %793 ]
  %797 = icmp eq ptr %795, %741
  br i1 %797, label %807, label %798

798:                                              ; preds = %794
  %799 = getelementptr inbounds i8, ptr %795, i64 -8
  %800 = load double, ptr %799, align 8, !tbaa !33
  %801 = getelementptr inbounds i8, ptr %796, i64 -8
  %802 = load double, ptr %801, align 8, !tbaa !33
  %803 = fcmp olt double %800, %802
  br i1 %803, label %804, label %794, !llvm.loop !61

804:                                              ; preds = %798
  %805 = load i32, ptr @current_test, align 4, !tbaa !11
  %806 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.52, i32 noundef %805)
  br label %807

807:                                              ; preds = %794, %804
  %808 = add nuw nsw i32 %766, 1
  %809 = load i32, ptr @iterations, align 4, !tbaa !11
  %810 = icmp slt i32 %808, %809
  br i1 %810, label %765, label %811, !llvm.loop !79

811:                                              ; preds = %807
  call void @llvm.lifetime.end.p0(ptr nonnull %25)
  call void @llvm.lifetime.end.p0(ptr nonnull %26)
  call void @llvm.lifetime.start.p0(ptr nonnull %23)
  call void @llvm.lifetime.start.p0(ptr nonnull %24)
  %812 = icmp sgt i32 %809, 0
  br i1 %812, label %813, label %815

813:                                              ; preds = %811
  %814 = getelementptr inbounds nuw i8, ptr %79, i64 15992
  br label %816

815:                                              ; preds = %744, %811
  call void @llvm.lifetime.end.p0(ptr nonnull %23)
  call void @llvm.lifetime.end.p0(ptr nonnull %24)
  call void @llvm.lifetime.start.p0(ptr nonnull %21)
  call void @llvm.lifetime.start.p0(ptr nonnull %22)
  br label %856

816:                                              ; preds = %813, %831
  %817 = phi i32 [ %832, %831 ], [ 0, %813 ]
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(16000) %79, ptr noundef nonnull align 8 dereferenceable(16000) %331, i64 16000, i1 false), !tbaa !33
  store i64 %92, ptr %23, align 8, !tbaa !34
  store i64 %91, ptr %24, align 8, !tbaa !34
  invoke void @_ZN9benchmark9quicksortISt16reverse_iteratorIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEdEEvT_SA_(ptr dead_on_return noundef nonnull %23, ptr dead_on_return noundef nonnull %24)
          to label %818 unwind label %1236

818:                                              ; preds = %816, %822
  %819 = phi ptr [ %823, %822 ], [ %814, %816 ]
  %820 = phi ptr [ %825, %822 ], [ %90, %816 ]
  %821 = icmp eq ptr %819, %79
  br i1 %821, label %831, label %822

822:                                              ; preds = %818
  %823 = getelementptr inbounds i8, ptr %819, i64 -8
  %824 = load double, ptr %823, align 8, !tbaa !33
  %825 = getelementptr inbounds i8, ptr %820, i64 -8
  %826 = load double, ptr %825, align 8, !tbaa !33
  %827 = fcmp olt double %824, %826
  br i1 %827, label %828, label %818, !llvm.loop !65

828:                                              ; preds = %822
  %829 = load i32, ptr @current_test, align 4, !tbaa !11
  %830 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.52, i32 noundef %829)
  br label %831

831:                                              ; preds = %818, %828
  %832 = add nuw nsw i32 %817, 1
  %833 = load i32, ptr @iterations, align 4, !tbaa !11
  %834 = icmp slt i32 %832, %833
  br i1 %834, label %816, label %835, !llvm.loop !80

835:                                              ; preds = %831
  call void @llvm.lifetime.end.p0(ptr nonnull %23)
  call void @llvm.lifetime.end.p0(ptr nonnull %24)
  call void @llvm.lifetime.start.p0(ptr nonnull %21)
  call void @llvm.lifetime.start.p0(ptr nonnull %22)
  %836 = icmp sgt i32 %833, 0
  br i1 %836, label %837, label %856

837:                                              ; preds = %835, %852
  %838 = phi i32 [ %853, %852 ], [ 0, %835 ]
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(16000) %79, ptr noundef nonnull align 8 dereferenceable(16000) %331, i64 16000, i1 false), !tbaa !33
  store i64 %92, ptr %21, align 8, !tbaa !34
  store i64 %91, ptr %22, align 8, !tbaa !34
  invoke void @_ZN9benchmark9quicksortISt16reverse_iteratorIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEdEEvT_SA_(ptr dead_on_return noundef nonnull %21, ptr dead_on_return noundef nonnull %22)
          to label %839 unwind label %1234

839:                                              ; preds = %837, %843
  %840 = phi ptr [ %844, %843 ], [ %814, %837 ]
  %841 = phi ptr [ %846, %843 ], [ %90, %837 ]
  %842 = icmp eq ptr %840, %79
  br i1 %842, label %852, label %843

843:                                              ; preds = %839
  %844 = getelementptr inbounds i8, ptr %840, i64 -8
  %845 = load double, ptr %844, align 8, !tbaa !33
  %846 = getelementptr inbounds i8, ptr %841, i64 -8
  %847 = load double, ptr %846, align 8, !tbaa !33
  %848 = fcmp olt double %845, %847
  br i1 %848, label %849, label %839, !llvm.loop !65

849:                                              ; preds = %843
  %850 = load i32, ptr @current_test, align 4, !tbaa !11
  %851 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.52, i32 noundef %850)
  br label %852

852:                                              ; preds = %839, %849
  %853 = add nuw nsw i32 %838, 1
  %854 = load i32, ptr @iterations, align 4, !tbaa !11
  %855 = icmp slt i32 %853, %854
  br i1 %855, label %837, label %857, !llvm.loop !80

856:                                              ; preds = %835, %815
  call void @llvm.lifetime.end.p0(ptr nonnull %21)
  call void @llvm.lifetime.end.p0(ptr nonnull %22)
  call void @llvm.lifetime.start.p0(ptr nonnull %19)
  call void @llvm.lifetime.start.p0(ptr nonnull %20)
  br label %863

857:                                              ; preds = %852
  call void @llvm.lifetime.end.p0(ptr nonnull %21)
  call void @llvm.lifetime.end.p0(ptr nonnull %22)
  %858 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @rrdMpb, i64 8), align 8, !tbaa !43
  %859 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @rrdMpe, i64 8), align 8, !tbaa !43
  %860 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @rrdpb, i64 8), align 8, !tbaa !43
  %861 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @rrdpe, i64 8), align 8, !tbaa !43
  call void @llvm.lifetime.start.p0(ptr nonnull %19)
  call void @llvm.lifetime.start.p0(ptr nonnull %20)
  %862 = icmp sgt i32 %854, 0
  br i1 %862, label %864, label %863

863:                                              ; preds = %856, %857
  call void @llvm.lifetime.end.p0(ptr nonnull %19)
  call void @llvm.lifetime.end.p0(ptr nonnull %20)
  call void @llvm.lifetime.start.p0(ptr nonnull %17)
  call void @llvm.lifetime.start.p0(ptr nonnull %18)
  br label %949

864:                                              ; preds = %857
  %865 = ptrtoint ptr %860 to i64
  %866 = ptrtoint ptr %859 to i64
  %867 = ptrtoint ptr %858 to i64
  %868 = getelementptr inbounds nuw i8, ptr %19, i64 8
  %869 = getelementptr inbounds nuw i8, ptr %20, i64 8
  %870 = icmp eq ptr %858, %859
  %871 = sub i64 %865, %867
  %872 = add i64 %866, -8
  %873 = sub i64 %872, %867
  %874 = lshr i64 %873, 3
  %875 = add nuw nsw i64 %874, 1
  %876 = icmp ult i64 %873, 24
  %877 = icmp ult i64 %871, 32
  %878 = select i1 %876, i1 true, i1 %877
  %879 = and i64 %875, 4611686018427387900
  %880 = shl i64 %879, 3
  %881 = getelementptr i8, ptr %860, i64 %880
  %882 = shl i64 %879, 3
  %883 = getelementptr i8, ptr %858, i64 %882
  %884 = icmp eq i64 %875, %879
  br label %885

885:                                              ; preds = %923, %864
  %886 = phi i32 [ 0, %864 ], [ %924, %923 ]
  br i1 %870, label %911, label %887

887:                                              ; preds = %885
  br i1 %878, label %901, label %888

888:                                              ; preds = %887, %888
  %889 = phi i64 [ %898, %888 ], [ 0, %887 ]
  %890 = shl i64 %889, 3
  %891 = getelementptr i8, ptr %860, i64 %890
  %892 = shl i64 %889, 3
  %893 = getelementptr i8, ptr %858, i64 %892
  %894 = getelementptr i8, ptr %893, i64 16
  %895 = load <2 x double>, ptr %893, align 8, !tbaa !33
  %896 = load <2 x double>, ptr %894, align 8, !tbaa !33
  %897 = getelementptr i8, ptr %891, i64 16
  store <2 x double> %895, ptr %891, align 8, !tbaa !33
  store <2 x double> %896, ptr %897, align 8, !tbaa !33
  %898 = add nuw i64 %889, 4
  %899 = icmp eq i64 %898, %879
  br i1 %899, label %900, label %888, !llvm.loop !81

900:                                              ; preds = %888
  br i1 %884, label %911, label %901

901:                                              ; preds = %887, %900
  %902 = phi ptr [ %860, %887 ], [ %881, %900 ]
  %903 = phi ptr [ %858, %887 ], [ %883, %900 ]
  br label %904

904:                                              ; preds = %901, %904
  %905 = phi ptr [ %909, %904 ], [ %902, %901 ]
  %906 = phi ptr [ %907, %904 ], [ %903, %901 ]
  %907 = getelementptr inbounds nuw i8, ptr %906, i64 8
  %908 = load double, ptr %906, align 8, !tbaa !33
  %909 = getelementptr inbounds nuw i8, ptr %905, i64 8
  store double %908, ptr %905, align 8, !tbaa !33
  %910 = icmp eq ptr %907, %859
  br i1 %910, label %911, label %904, !llvm.loop !82

911:                                              ; preds = %904, %900, %885
  store ptr %860, ptr %868, align 8, !tbaa !43
  store ptr %861, ptr %869, align 8, !tbaa !43
  invoke void @_ZN9benchmark9quicksortISt16reverse_iteratorIS1_IPdEEdEEvT_S5_(ptr dead_on_return noundef nonnull %19, ptr dead_on_return noundef nonnull %20)
          to label %912 unwind label %1232

912:                                              ; preds = %911, %916
  %913 = phi ptr [ %914, %916 ], [ %860, %911 ]
  %914 = getelementptr i8, ptr %913, i64 8
  %915 = icmp eq ptr %914, %861
  br i1 %915, label %923, label %916

916:                                              ; preds = %912
  %917 = load double, ptr %914, align 8, !tbaa !33
  %918 = load double, ptr %913, align 8, !tbaa !33
  %919 = fcmp olt double %917, %918
  br i1 %919, label %920, label %912, !llvm.loop !71

920:                                              ; preds = %916
  %921 = load i32, ptr @current_test, align 4, !tbaa !11
  %922 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.52, i32 noundef %921)
  br label %923

923:                                              ; preds = %912, %920
  %924 = add nuw nsw i32 %886, 1
  %925 = load i32, ptr @iterations, align 4, !tbaa !11
  %926 = icmp slt i32 %924, %925
  br i1 %926, label %885, label %927, !llvm.loop !83

927:                                              ; preds = %923
  call void @llvm.lifetime.end.p0(ptr nonnull %19)
  call void @llvm.lifetime.end.p0(ptr nonnull %20)
  call void @llvm.lifetime.start.p0(ptr nonnull %17)
  call void @llvm.lifetime.start.p0(ptr nonnull %18)
  %928 = icmp sgt i32 %925, 0
  br i1 %928, label %929, label %949

929:                                              ; preds = %927
  %930 = getelementptr inbounds nuw i8, ptr %17, i64 8
  %931 = getelementptr inbounds nuw i8, ptr %18, i64 8
  br label %932

932:                                              ; preds = %945, %929
  %933 = phi i32 [ 0, %929 ], [ %946, %945 ]
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(16000) %79, ptr noundef nonnull align 8 dereferenceable(16000) %331, i64 16000, i1 false), !tbaa !33
  store i64 %91, ptr %930, align 8, !tbaa !34
  store i64 %92, ptr %931, align 8, !tbaa !34
  invoke void @_ZN9benchmark9quicksortISt16reverse_iteratorIS1_IN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEEdEEvT_SB_(ptr dead_on_return noundef nonnull %17, ptr dead_on_return noundef nonnull %18)
          to label %934 unwind label %1230

934:                                              ; preds = %932, %938
  %935 = phi ptr [ %936, %938 ], [ %79, %932 ]
  %936 = getelementptr i8, ptr %935, i64 8
  %937 = icmp eq ptr %90, %936
  br i1 %937, label %945, label %938

938:                                              ; preds = %934
  %939 = load double, ptr %936, align 8, !tbaa !33
  %940 = load double, ptr %935, align 8, !tbaa !33
  %941 = fcmp olt double %939, %940
  br i1 %941, label %942, label %934, !llvm.loop !75

942:                                              ; preds = %938
  %943 = load i32, ptr @current_test, align 4, !tbaa !11
  %944 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.52, i32 noundef %943)
  br label %945

945:                                              ; preds = %934, %942
  %946 = add nuw nsw i32 %933, 1
  %947 = load i32, ptr @iterations, align 4, !tbaa !11
  %948 = icmp slt i32 %946, %947
  br i1 %948, label %932, label %950, !llvm.loop !84

949:                                              ; preds = %863, %927
  call void @llvm.lifetime.end.p0(ptr nonnull %17)
  call void @llvm.lifetime.end.p0(ptr nonnull %18)
  call void @llvm.lifetime.start.p0(ptr nonnull %15)
  call void @llvm.lifetime.start.p0(ptr nonnull %16)
  br label %972

950:                                              ; preds = %945
  call void @llvm.lifetime.end.p0(ptr nonnull %17)
  call void @llvm.lifetime.end.p0(ptr nonnull %18)
  call void @llvm.lifetime.start.p0(ptr nonnull %15)
  call void @llvm.lifetime.start.p0(ptr nonnull %16)
  %951 = icmp sgt i32 %947, 0
  br i1 %951, label %952, label %972

952:                                              ; preds = %950
  %953 = getelementptr inbounds nuw i8, ptr %15, i64 8
  %954 = getelementptr inbounds nuw i8, ptr %16, i64 8
  br label %955

955:                                              ; preds = %968, %952
  %956 = phi i32 [ 0, %952 ], [ %969, %968 ]
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(16000) %79, ptr noundef nonnull align 8 dereferenceable(16000) %331, i64 16000, i1 false), !tbaa !33
  store i64 %91, ptr %953, align 8, !tbaa !34
  store i64 %92, ptr %954, align 8, !tbaa !34
  invoke void @_ZN9benchmark9quicksortISt16reverse_iteratorIS1_IN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEEdEEvT_SB_(ptr dead_on_return noundef nonnull %15, ptr dead_on_return noundef nonnull %16)
          to label %957 unwind label %1228

957:                                              ; preds = %955, %961
  %958 = phi ptr [ %959, %961 ], [ %79, %955 ]
  %959 = getelementptr i8, ptr %958, i64 8
  %960 = icmp eq ptr %90, %959
  br i1 %960, label %968, label %961

961:                                              ; preds = %957
  %962 = load double, ptr %959, align 8, !tbaa !33
  %963 = load double, ptr %958, align 8, !tbaa !33
  %964 = fcmp olt double %962, %963
  br i1 %964, label %965, label %957, !llvm.loop !75

965:                                              ; preds = %961
  %966 = load i32, ptr @current_test, align 4, !tbaa !11
  %967 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.52, i32 noundef %966)
  br label %968

968:                                              ; preds = %957, %965
  %969 = add nuw nsw i32 %956, 1
  %970 = load i32, ptr @iterations, align 4, !tbaa !11
  %971 = icmp slt i32 %969, %970
  br i1 %971, label %955, label %972, !llvm.loop !84

972:                                              ; preds = %968, %949, %950
  call void @llvm.lifetime.end.p0(ptr nonnull %15)
  call void @llvm.lifetime.end.p0(ptr nonnull %16)
  %973 = load ptr, ptr @dMpb, align 8, !tbaa !34
  %974 = load ptr, ptr @dMpe, align 8, !tbaa !34
  %975 = load ptr, ptr @dpb, align 8, !tbaa !34
  %976 = load ptr, ptr @dpe, align 8, !tbaa !34
  invoke void @_Z14test_heap_sortIPddEvT_S1_S1_S1_T0_PKc(ptr noundef %973, ptr noundef %974, ptr noundef %975, ptr noundef %976, double noundef 0.000000e+00, ptr noundef nonnull @.str.42)
          to label %977 unwind label %1240

977:                                              ; preds = %972
  invoke void @_Z14test_heap_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEvT_S7_S7_S7_T0_PKc(ptr nonnull %331, ptr nonnull %334, ptr nonnull %79, ptr nonnull %90, double noundef 0.000000e+00, ptr noundef nonnull @.str.43)
          to label %978 unwind label %1240

978:                                              ; preds = %977
  %979 = load ptr, ptr @rdMpb, align 8, !tbaa !43
  %980 = load ptr, ptr @rdMpe, align 8, !tbaa !43
  %981 = load ptr, ptr @rdpb, align 8, !tbaa !43
  %982 = load ptr, ptr @rdpe, align 8, !tbaa !43
  call void @llvm.lifetime.start.p0(ptr nonnull %13)
  call void @llvm.lifetime.start.p0(ptr nonnull %14)
  %983 = load i32, ptr @iterations, align 4, !tbaa !11
  %984 = icmp sgt i32 %983, 0
  br i1 %984, label %986, label %985

985:                                              ; preds = %978
  call void @llvm.lifetime.end.p0(ptr nonnull %13)
  call void @llvm.lifetime.end.p0(ptr nonnull %14)
  call void @llvm.lifetime.start.p0(ptr nonnull %11)
  call void @llvm.lifetime.start.p0(ptr nonnull %12)
  br label %1056

986:                                              ; preds = %978
  %987 = ptrtoint ptr %981 to i64
  %988 = ptrtoint ptr %980 to i64
  %989 = ptrtoint ptr %979 to i64
  %990 = icmp eq ptr %979, %980
  %991 = getelementptr inbounds i8, ptr %981, i64 -8
  %992 = sub i64 %989, %987
  %993 = add i64 %989, -8
  %994 = sub i64 %993, %988
  %995 = lshr i64 %994, 3
  %996 = add nuw nsw i64 %995, 1
  %997 = icmp ult i64 %994, 24
  %998 = icmp ult i64 %992, 32
  %999 = select i1 %997, i1 true, i1 %998
  %1000 = and i64 %996, 4611686018427387900
  %1001 = mul i64 %1000, -8
  %1002 = getelementptr i8, ptr %981, i64 %1001
  %1003 = mul i64 %1000, -8
  %1004 = getelementptr i8, ptr %979, i64 %1003
  %1005 = icmp eq i64 %996, %1000
  br label %1006

1006:                                             ; preds = %986, %1048
  %1007 = phi i32 [ %1049, %1048 ], [ 0, %986 ]
  br i1 %990, label %1034, label %1008

1008:                                             ; preds = %1006
  br i1 %999, label %1024, label %1009

1009:                                             ; preds = %1008, %1009
  %1010 = phi i64 [ %1021, %1009 ], [ 0, %1008 ]
  %1011 = mul i64 %1010, -8
  %1012 = getelementptr i8, ptr %981, i64 %1011
  %1013 = mul i64 %1010, -8
  %1014 = getelementptr i8, ptr %979, i64 %1013
  %1015 = getelementptr inbounds i8, ptr %1014, i64 -16
  %1016 = getelementptr inbounds i8, ptr %1014, i64 -32
  %1017 = load <2 x double>, ptr %1015, align 8, !tbaa !33
  %1018 = load <2 x double>, ptr %1016, align 8, !tbaa !33
  %1019 = getelementptr inbounds i8, ptr %1012, i64 -16
  %1020 = getelementptr inbounds i8, ptr %1012, i64 -32
  store <2 x double> %1017, ptr %1019, align 8, !tbaa !33
  store <2 x double> %1018, ptr %1020, align 8, !tbaa !33
  %1021 = add nuw i64 %1010, 4
  %1022 = icmp eq i64 %1021, %1000
  br i1 %1022, label %1023, label %1009, !llvm.loop !85

1023:                                             ; preds = %1009
  br i1 %1005, label %1034, label %1024

1024:                                             ; preds = %1008, %1023
  %1025 = phi ptr [ %981, %1008 ], [ %1002, %1023 ]
  %1026 = phi ptr [ %979, %1008 ], [ %1004, %1023 ]
  br label %1027

1027:                                             ; preds = %1024, %1027
  %1028 = phi ptr [ %1032, %1027 ], [ %1025, %1024 ]
  %1029 = phi ptr [ %1030, %1027 ], [ %1026, %1024 ]
  %1030 = getelementptr inbounds i8, ptr %1029, i64 -8
  %1031 = load double, ptr %1030, align 8, !tbaa !33
  %1032 = getelementptr inbounds i8, ptr %1028, i64 -8
  store double %1031, ptr %1032, align 8, !tbaa !33
  %1033 = icmp eq ptr %1030, %980
  br i1 %1033, label %1034, label %1027, !llvm.loop !86

1034:                                             ; preds = %1027, %1023, %1006
  store ptr %981, ptr %13, align 8, !tbaa !43
  store ptr %982, ptr %14, align 8, !tbaa !43
  invoke void @_ZN9benchmark8heapsortISt16reverse_iteratorIPdEdEEvT_S4_(ptr dead_on_return noundef nonnull %13, ptr dead_on_return noundef nonnull %14)
          to label %1035 unwind label %1226

1035:                                             ; preds = %1034, %1039
  %1036 = phi ptr [ %1040, %1039 ], [ %991, %1034 ]
  %1037 = phi ptr [ %1042, %1039 ], [ %981, %1034 ]
  %1038 = icmp eq ptr %1036, %982
  br i1 %1038, label %1048, label %1039

1039:                                             ; preds = %1035
  %1040 = getelementptr inbounds i8, ptr %1036, i64 -8
  %1041 = load double, ptr %1040, align 8, !tbaa !33
  %1042 = getelementptr inbounds i8, ptr %1037, i64 -8
  %1043 = load double, ptr %1042, align 8, !tbaa !33
  %1044 = fcmp olt double %1041, %1043
  br i1 %1044, label %1045, label %1035, !llvm.loop !61

1045:                                             ; preds = %1039
  %1046 = load i32, ptr @current_test, align 4, !tbaa !11
  %1047 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.52, i32 noundef %1046)
  br label %1048

1048:                                             ; preds = %1035, %1045
  %1049 = add nuw nsw i32 %1007, 1
  %1050 = load i32, ptr @iterations, align 4, !tbaa !11
  %1051 = icmp slt i32 %1049, %1050
  br i1 %1051, label %1006, label %1052, !llvm.loop !87

1052:                                             ; preds = %1048
  call void @llvm.lifetime.end.p0(ptr nonnull %13)
  call void @llvm.lifetime.end.p0(ptr nonnull %14)
  call void @llvm.lifetime.start.p0(ptr nonnull %11)
  call void @llvm.lifetime.start.p0(ptr nonnull %12)
  %1053 = icmp sgt i32 %1050, 0
  br i1 %1053, label %1054, label %1056

1054:                                             ; preds = %1052
  %1055 = getelementptr inbounds nuw i8, ptr %79, i64 15992
  br label %1057

1056:                                             ; preds = %985, %1052
  call void @llvm.lifetime.end.p0(ptr nonnull %11)
  call void @llvm.lifetime.end.p0(ptr nonnull %12)
  call void @llvm.lifetime.start.p0(ptr nonnull %9)
  call void @llvm.lifetime.start.p0(ptr nonnull %10)
  br label %1097

1057:                                             ; preds = %1054, %1072
  %1058 = phi i32 [ %1073, %1072 ], [ 0, %1054 ]
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(16000) %79, ptr noundef nonnull align 8 dereferenceable(16000) %331, i64 16000, i1 false), !tbaa !33
  store i64 %92, ptr %11, align 8, !tbaa !34
  store i64 %91, ptr %12, align 8, !tbaa !34
  invoke void @_ZN9benchmark8heapsortISt16reverse_iteratorIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEdEEvT_SA_(ptr dead_on_return noundef nonnull %11, ptr dead_on_return noundef nonnull %12)
          to label %1059 unwind label %1224

1059:                                             ; preds = %1057, %1063
  %1060 = phi ptr [ %1064, %1063 ], [ %1055, %1057 ]
  %1061 = phi ptr [ %1066, %1063 ], [ %90, %1057 ]
  %1062 = icmp eq ptr %1060, %79
  br i1 %1062, label %1072, label %1063

1063:                                             ; preds = %1059
  %1064 = getelementptr inbounds i8, ptr %1060, i64 -8
  %1065 = load double, ptr %1064, align 8, !tbaa !33
  %1066 = getelementptr inbounds i8, ptr %1061, i64 -8
  %1067 = load double, ptr %1066, align 8, !tbaa !33
  %1068 = fcmp olt double %1065, %1067
  br i1 %1068, label %1069, label %1059, !llvm.loop !65

1069:                                             ; preds = %1063
  %1070 = load i32, ptr @current_test, align 4, !tbaa !11
  %1071 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.52, i32 noundef %1070)
  br label %1072

1072:                                             ; preds = %1059, %1069
  %1073 = add nuw nsw i32 %1058, 1
  %1074 = load i32, ptr @iterations, align 4, !tbaa !11
  %1075 = icmp slt i32 %1073, %1074
  br i1 %1075, label %1057, label %1076, !llvm.loop !88

1076:                                             ; preds = %1072
  call void @llvm.lifetime.end.p0(ptr nonnull %11)
  call void @llvm.lifetime.end.p0(ptr nonnull %12)
  call void @llvm.lifetime.start.p0(ptr nonnull %9)
  call void @llvm.lifetime.start.p0(ptr nonnull %10)
  %1077 = icmp sgt i32 %1074, 0
  br i1 %1077, label %1078, label %1097

1078:                                             ; preds = %1076, %1093
  %1079 = phi i32 [ %1094, %1093 ], [ 0, %1076 ]
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(16000) %79, ptr noundef nonnull align 8 dereferenceable(16000) %331, i64 16000, i1 false), !tbaa !33
  store i64 %92, ptr %9, align 8, !tbaa !34
  store i64 %91, ptr %10, align 8, !tbaa !34
  invoke void @_ZN9benchmark8heapsortISt16reverse_iteratorIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEdEEvT_SA_(ptr dead_on_return noundef nonnull %9, ptr dead_on_return noundef nonnull %10)
          to label %1080 unwind label %1222

1080:                                             ; preds = %1078, %1084
  %1081 = phi ptr [ %1085, %1084 ], [ %1055, %1078 ]
  %1082 = phi ptr [ %1087, %1084 ], [ %90, %1078 ]
  %1083 = icmp eq ptr %1081, %79
  br i1 %1083, label %1093, label %1084

1084:                                             ; preds = %1080
  %1085 = getelementptr inbounds i8, ptr %1081, i64 -8
  %1086 = load double, ptr %1085, align 8, !tbaa !33
  %1087 = getelementptr inbounds i8, ptr %1082, i64 -8
  %1088 = load double, ptr %1087, align 8, !tbaa !33
  %1089 = fcmp olt double %1086, %1088
  br i1 %1089, label %1090, label %1080, !llvm.loop !65

1090:                                             ; preds = %1084
  %1091 = load i32, ptr @current_test, align 4, !tbaa !11
  %1092 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.52, i32 noundef %1091)
  br label %1093

1093:                                             ; preds = %1080, %1090
  %1094 = add nuw nsw i32 %1079, 1
  %1095 = load i32, ptr @iterations, align 4, !tbaa !11
  %1096 = icmp slt i32 %1094, %1095
  br i1 %1096, label %1078, label %1098, !llvm.loop !88

1097:                                             ; preds = %1076, %1056
  call void @llvm.lifetime.end.p0(ptr nonnull %9)
  call void @llvm.lifetime.end.p0(ptr nonnull %10)
  call void @llvm.lifetime.start.p0(ptr nonnull %7)
  call void @llvm.lifetime.start.p0(ptr nonnull %8)
  br label %1104

1098:                                             ; preds = %1093
  call void @llvm.lifetime.end.p0(ptr nonnull %9)
  call void @llvm.lifetime.end.p0(ptr nonnull %10)
  %1099 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @rrdMpb, i64 8), align 8, !tbaa !43
  %1100 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @rrdMpe, i64 8), align 8, !tbaa !43
  %1101 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @rrdpb, i64 8), align 8, !tbaa !43
  %1102 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @rrdpe, i64 8), align 8, !tbaa !43
  call void @llvm.lifetime.start.p0(ptr nonnull %7)
  call void @llvm.lifetime.start.p0(ptr nonnull %8)
  %1103 = icmp sgt i32 %1095, 0
  br i1 %1103, label %1105, label %1104

1104:                                             ; preds = %1097, %1098
  call void @llvm.lifetime.end.p0(ptr nonnull %7)
  call void @llvm.lifetime.end.p0(ptr nonnull %8)
  call void @llvm.lifetime.start.p0(ptr nonnull %5)
  call void @llvm.lifetime.start.p0(ptr nonnull %6)
  br label %1190

1105:                                             ; preds = %1098
  %1106 = ptrtoint ptr %1101 to i64
  %1107 = ptrtoint ptr %1100 to i64
  %1108 = ptrtoint ptr %1099 to i64
  %1109 = getelementptr inbounds nuw i8, ptr %7, i64 8
  %1110 = getelementptr inbounds nuw i8, ptr %8, i64 8
  %1111 = icmp eq ptr %1099, %1100
  %1112 = sub i64 %1106, %1108
  %1113 = add i64 %1107, -8
  %1114 = sub i64 %1113, %1108
  %1115 = lshr i64 %1114, 3
  %1116 = add nuw nsw i64 %1115, 1
  %1117 = icmp ult i64 %1114, 24
  %1118 = icmp ult i64 %1112, 32
  %1119 = select i1 %1117, i1 true, i1 %1118
  %1120 = and i64 %1116, 4611686018427387900
  %1121 = shl i64 %1120, 3
  %1122 = getelementptr i8, ptr %1101, i64 %1121
  %1123 = shl i64 %1120, 3
  %1124 = getelementptr i8, ptr %1099, i64 %1123
  %1125 = icmp eq i64 %1116, %1120
  br label %1126

1126:                                             ; preds = %1164, %1105
  %1127 = phi i32 [ 0, %1105 ], [ %1165, %1164 ]
  br i1 %1111, label %1152, label %1128

1128:                                             ; preds = %1126
  br i1 %1119, label %1142, label %1129

1129:                                             ; preds = %1128, %1129
  %1130 = phi i64 [ %1139, %1129 ], [ 0, %1128 ]
  %1131 = shl i64 %1130, 3
  %1132 = getelementptr i8, ptr %1101, i64 %1131
  %1133 = shl i64 %1130, 3
  %1134 = getelementptr i8, ptr %1099, i64 %1133
  %1135 = getelementptr i8, ptr %1134, i64 16
  %1136 = load <2 x double>, ptr %1134, align 8, !tbaa !33
  %1137 = load <2 x double>, ptr %1135, align 8, !tbaa !33
  %1138 = getelementptr i8, ptr %1132, i64 16
  store <2 x double> %1136, ptr %1132, align 8, !tbaa !33
  store <2 x double> %1137, ptr %1138, align 8, !tbaa !33
  %1139 = add nuw i64 %1130, 4
  %1140 = icmp eq i64 %1139, %1120
  br i1 %1140, label %1141, label %1129, !llvm.loop !89

1141:                                             ; preds = %1129
  br i1 %1125, label %1152, label %1142

1142:                                             ; preds = %1128, %1141
  %1143 = phi ptr [ %1101, %1128 ], [ %1122, %1141 ]
  %1144 = phi ptr [ %1099, %1128 ], [ %1124, %1141 ]
  br label %1145

1145:                                             ; preds = %1142, %1145
  %1146 = phi ptr [ %1150, %1145 ], [ %1143, %1142 ]
  %1147 = phi ptr [ %1148, %1145 ], [ %1144, %1142 ]
  %1148 = getelementptr inbounds nuw i8, ptr %1147, i64 8
  %1149 = load double, ptr %1147, align 8, !tbaa !33
  %1150 = getelementptr inbounds nuw i8, ptr %1146, i64 8
  store double %1149, ptr %1146, align 8, !tbaa !33
  %1151 = icmp eq ptr %1148, %1100
  br i1 %1151, label %1152, label %1145, !llvm.loop !90

1152:                                             ; preds = %1145, %1141, %1126
  store ptr %1101, ptr %1109, align 8, !tbaa !43
  store ptr %1102, ptr %1110, align 8, !tbaa !43
  invoke void @_ZN9benchmark8heapsortISt16reverse_iteratorIS1_IPdEEdEEvT_S5_(ptr dead_on_return noundef nonnull %7, ptr dead_on_return noundef nonnull %8)
          to label %1153 unwind label %1220

1153:                                             ; preds = %1152, %1157
  %1154 = phi ptr [ %1155, %1157 ], [ %1101, %1152 ]
  %1155 = getelementptr i8, ptr %1154, i64 8
  %1156 = icmp eq ptr %1155, %1102
  br i1 %1156, label %1164, label %1157

1157:                                             ; preds = %1153
  %1158 = load double, ptr %1155, align 8, !tbaa !33
  %1159 = load double, ptr %1154, align 8, !tbaa !33
  %1160 = fcmp olt double %1158, %1159
  br i1 %1160, label %1161, label %1153, !llvm.loop !71

1161:                                             ; preds = %1157
  %1162 = load i32, ptr @current_test, align 4, !tbaa !11
  %1163 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.52, i32 noundef %1162)
  br label %1164

1164:                                             ; preds = %1153, %1161
  %1165 = add nuw nsw i32 %1127, 1
  %1166 = load i32, ptr @iterations, align 4, !tbaa !11
  %1167 = icmp slt i32 %1165, %1166
  br i1 %1167, label %1126, label %1168, !llvm.loop !91

1168:                                             ; preds = %1164
  call void @llvm.lifetime.end.p0(ptr nonnull %7)
  call void @llvm.lifetime.end.p0(ptr nonnull %8)
  call void @llvm.lifetime.start.p0(ptr nonnull %5)
  call void @llvm.lifetime.start.p0(ptr nonnull %6)
  %1169 = icmp sgt i32 %1166, 0
  br i1 %1169, label %1170, label %1190

1170:                                             ; preds = %1168
  %1171 = getelementptr inbounds nuw i8, ptr %5, i64 8
  %1172 = getelementptr inbounds nuw i8, ptr %6, i64 8
  br label %1173

1173:                                             ; preds = %1186, %1170
  %1174 = phi i32 [ 0, %1170 ], [ %1187, %1186 ]
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(16000) %79, ptr noundef nonnull align 8 dereferenceable(16000) %331, i64 16000, i1 false), !tbaa !33
  store i64 %91, ptr %1171, align 8, !tbaa !34
  store i64 %92, ptr %1172, align 8, !tbaa !34
  invoke void @_ZN9benchmark8heapsortISt16reverse_iteratorIS1_IN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEEdEEvT_SB_(ptr dead_on_return noundef nonnull %5, ptr dead_on_return noundef nonnull %6)
          to label %1175 unwind label %1218

1175:                                             ; preds = %1173, %1179
  %1176 = phi ptr [ %1177, %1179 ], [ %79, %1173 ]
  %1177 = getelementptr i8, ptr %1176, i64 8
  %1178 = icmp eq ptr %90, %1177
  br i1 %1178, label %1186, label %1179

1179:                                             ; preds = %1175
  %1180 = load double, ptr %1177, align 8, !tbaa !33
  %1181 = load double, ptr %1176, align 8, !tbaa !33
  %1182 = fcmp olt double %1180, %1181
  br i1 %1182, label %1183, label %1175, !llvm.loop !75

1183:                                             ; preds = %1179
  %1184 = load i32, ptr @current_test, align 4, !tbaa !11
  %1185 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.52, i32 noundef %1184)
  br label %1186

1186:                                             ; preds = %1175, %1183
  %1187 = add nuw nsw i32 %1174, 1
  %1188 = load i32, ptr @iterations, align 4, !tbaa !11
  %1189 = icmp slt i32 %1187, %1188
  br i1 %1189, label %1173, label %1191, !llvm.loop !92

1190:                                             ; preds = %1104, %1168
  call void @llvm.lifetime.end.p0(ptr nonnull %5)
  call void @llvm.lifetime.end.p0(ptr nonnull %6)
  call void @llvm.lifetime.start.p0(ptr nonnull %3)
  call void @llvm.lifetime.start.p0(ptr nonnull %4)
  br label %1213

1191:                                             ; preds = %1186
  call void @llvm.lifetime.end.p0(ptr nonnull %5)
  call void @llvm.lifetime.end.p0(ptr nonnull %6)
  call void @llvm.lifetime.start.p0(ptr nonnull %3)
  call void @llvm.lifetime.start.p0(ptr nonnull %4)
  %1192 = icmp sgt i32 %1188, 0
  br i1 %1192, label %1193, label %1213

1193:                                             ; preds = %1191
  %1194 = getelementptr inbounds nuw i8, ptr %3, i64 8
  %1195 = getelementptr inbounds nuw i8, ptr %4, i64 8
  br label %1196

1196:                                             ; preds = %1209, %1193
  %1197 = phi i32 [ 0, %1193 ], [ %1210, %1209 ]
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(16000) %79, ptr noundef nonnull align 8 dereferenceable(16000) %331, i64 16000, i1 false), !tbaa !33
  store i64 %91, ptr %1194, align 8, !tbaa !34
  store i64 %92, ptr %1195, align 8, !tbaa !34
  invoke void @_ZN9benchmark8heapsortISt16reverse_iteratorIS1_IN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEEdEEvT_SB_(ptr dead_on_return noundef nonnull %3, ptr dead_on_return noundef nonnull %4)
          to label %1198 unwind label %1216

1198:                                             ; preds = %1196, %1202
  %1199 = phi ptr [ %1200, %1202 ], [ %79, %1196 ]
  %1200 = getelementptr i8, ptr %1199, i64 8
  %1201 = icmp eq ptr %90, %1200
  br i1 %1201, label %1209, label %1202

1202:                                             ; preds = %1198
  %1203 = load double, ptr %1200, align 8, !tbaa !33
  %1204 = load double, ptr %1199, align 8, !tbaa !33
  %1205 = fcmp olt double %1203, %1204
  br i1 %1205, label %1206, label %1198, !llvm.loop !75

1206:                                             ; preds = %1202
  %1207 = load i32, ptr @current_test, align 4, !tbaa !11
  %1208 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.52, i32 noundef %1207)
  br label %1209

1209:                                             ; preds = %1198, %1206
  %1210 = add nuw nsw i32 %1197, 1
  %1211 = load i32, ptr @iterations, align 4, !tbaa !11
  %1212 = icmp slt i32 %1210, %1211
  br i1 %1212, label %1196, label %1213, !llvm.loop !92

1213:                                             ; preds = %1209, %1190, %1191
  call void @llvm.lifetime.end.p0(ptr nonnull %3)
  call void @llvm.lifetime.end.p0(ptr nonnull %4)
  call void @_ZdlPvm(ptr noundef nonnull %331, i64 noundef 16000) #22
  call void @_ZdlPvm(ptr noundef nonnull %79, i64 noundef 16000) #22
  ret i32 0

1214:                                             ; preds = %328
  %1215 = landingpad { ptr, i32 }
          cleanup
  br label %1244

1216:                                             ; preds = %1196
  %1217 = landingpad { ptr, i32 }
          cleanup
  br label %1242

1218:                                             ; preds = %1173
  %1219 = landingpad { ptr, i32 }
          cleanup
  br label %1242

1220:                                             ; preds = %1152
  %1221 = landingpad { ptr, i32 }
          cleanup
  br label %1242

1222:                                             ; preds = %1078
  %1223 = landingpad { ptr, i32 }
          cleanup
  br label %1242

1224:                                             ; preds = %1057
  %1225 = landingpad { ptr, i32 }
          cleanup
  br label %1242

1226:                                             ; preds = %1034
  %1227 = landingpad { ptr, i32 }
          cleanup
  br label %1242

1228:                                             ; preds = %955
  %1229 = landingpad { ptr, i32 }
          cleanup
  br label %1242

1230:                                             ; preds = %932
  %1231 = landingpad { ptr, i32 }
          cleanup
  br label %1242

1232:                                             ; preds = %911
  %1233 = landingpad { ptr, i32 }
          cleanup
  br label %1242

1234:                                             ; preds = %837
  %1235 = landingpad { ptr, i32 }
          cleanup
  br label %1242

1236:                                             ; preds = %816
  %1237 = landingpad { ptr, i32 }
          cleanup
  br label %1242

1238:                                             ; preds = %793
  %1239 = landingpad { ptr, i32 }
          cleanup
  br label %1242

1240:                                             ; preds = %977, %972, %736, %729, %395, %392
  %1241 = landingpad { ptr, i32 }
          cleanup
  br label %1242

1242:                                             ; preds = %1218, %1222, %1226, %1230, %1234, %1238, %1240, %1236, %1232, %1228, %1224, %1220, %1216
  %1243 = phi { ptr, i32 } [ %1217, %1216 ], [ %1219, %1218 ], [ %1221, %1220 ], [ %1223, %1222 ], [ %1225, %1224 ], [ %1227, %1226 ], [ %1229, %1228 ], [ %1231, %1230 ], [ %1233, %1232 ], [ %1235, %1234 ], [ %1237, %1236 ], [ %1239, %1238 ], [ %1241, %1240 ]
  call void @_ZdlPvm(ptr noundef nonnull %331, i64 noundef 16000) #22
  br label %1244

1244:                                             ; preds = %1242, %1214
  %1245 = phi { ptr, i32 } [ %1243, %1242 ], [ %1215, %1214 ]
  call void @_ZdlPvm(ptr noundef nonnull %79, i64 noundef 16000) #22
  resume { ptr, i32 } %1245
}

; Function Attrs: nounwind
declare void @srand(i32 noundef) local_unnamed_addr #8

declare i32 @__gxx_personality_v0(...)

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_Z19test_insertion_sortIPddEvT_S1_S1_S1_T0_PKc(ptr noundef %0, ptr noundef %1, ptr noundef %2, ptr noundef %3, double noundef %4, ptr noundef %5) local_unnamed_addr #10 comdat {
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
  br i1 %47, label %48, label %40, !llvm.loop !93

48:                                               ; preds = %44
  %49 = load i32, ptr @current_test, align 4, !tbaa !11
  %50 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.52, i32 noundef %49)
  %51 = load i32, ptr @iterations, align 4, !tbaa !11
  br label %52

52:                                               ; preds = %40, %48
  %53 = phi i32 [ %51, %48 ], [ %38, %40 ]
  %54 = add nuw nsw i32 %39, 1
  %55 = icmp slt i32 %54, %53
  br i1 %55, label %37, label %205, !llvm.loop !94

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
  br i1 %70, label %71, label %59, !llvm.loop !95

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
  br i1 %81, label %82, label %75, !llvm.loop !96

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
  br i1 %90, label %91, label %83, !llvm.loop !93

91:                                               ; preds = %87
  %92 = load i32, ptr @current_test, align 4, !tbaa !11
  %93 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.52, i32 noundef %92)
  %94 = load i32, ptr @iterations, align 4, !tbaa !11
  br label %95

95:                                               ; preds = %83, %91
  %96 = phi i32 [ %94, %91 ], [ %57, %83 ]
  %97 = add nuw nsw i32 %58, 1
  %98 = icmp slt i32 %97, %96
  br i1 %98, label %56, label %205, !llvm.loop !94

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
  br i1 %127, label %128, label %121, !llvm.loop !97

128:                                              ; preds = %126, %121
  %129 = phi ptr [ %2, %126 ], [ %122, %121 ]
  store double %120, ptr %129, align 8, !tbaa !33
  %130 = getelementptr inbounds nuw i8, ptr %119, i64 8
  %131 = icmp eq ptr %130, %3
  br i1 %131, label %132, label %118, !llvm.loop !98

132:                                              ; preds = %128, %136
  %133 = phi ptr [ %134, %136 ], [ %2, %128 ]
  %134 = getelementptr i8, ptr %133, i64 8
  %135 = icmp eq ptr %134, %3
  br i1 %135, label %144, label %136

136:                                              ; preds = %132
  %137 = load double, ptr %134, align 8, !tbaa !33
  %138 = load double, ptr %133, align 8, !tbaa !33
  %139 = fcmp olt double %137, %138
  br i1 %139, label %140, label %132, !llvm.loop !93

140:                                              ; preds = %136
  %141 = load i32, ptr @current_test, align 4, !tbaa !11
  %142 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.52, i32 noundef %141)
  %143 = load i32, ptr @iterations, align 4, !tbaa !11
  br label %144

144:                                              ; preds = %132, %140
  %145 = phi i32 [ %143, %140 ], [ %116, %132 ]
  %146 = add nuw nsw i32 %117, 1
  %147 = icmp slt i32 %146, %145
  br i1 %147, label %115, label %205, !llvm.loop !94

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
  br i1 %162, label %163, label %151, !llvm.loop !99

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
  br i1 %173, label %174, label %167, !llvm.loop !100

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
  br i1 %184, label %185, label %178, !llvm.loop !97

185:                                              ; preds = %183, %178
  %186 = phi ptr [ %2, %183 ], [ %179, %178 ]
  store double %177, ptr %186, align 8, !tbaa !33
  %187 = getelementptr inbounds nuw i8, ptr %176, i64 8
  %188 = icmp eq ptr %187, %3
  br i1 %188, label %189, label %175, !llvm.loop !98

189:                                              ; preds = %185, %193
  %190 = phi ptr [ %191, %193 ], [ %2, %185 ]
  %191 = getelementptr i8, ptr %190, i64 8
  %192 = icmp eq ptr %191, %3
  br i1 %192, label %201, label %193

193:                                              ; preds = %189
  %194 = load double, ptr %191, align 8, !tbaa !33
  %195 = load double, ptr %190, align 8, !tbaa !33
  %196 = fcmp olt double %194, %195
  br i1 %196, label %197, label %189, !llvm.loop !93

197:                                              ; preds = %193
  %198 = load i32, ptr @current_test, align 4, !tbaa !11
  %199 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.52, i32 noundef %198)
  %200 = load i32, ptr @iterations, align 4, !tbaa !11
  br label %201

201:                                              ; preds = %189, %197
  %202 = phi i32 [ %200, %197 ], [ %149, %189 ]
  %203 = add nuw nsw i32 %150, 1
  %204 = icmp slt i32 %203, %202
  br i1 %204, label %148, label %205, !llvm.loop !94

205:                                              ; preds = %201, %144, %95, %52, %6
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_Z19test_insertion_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEvT_S7_S7_S7_T0_PKc(ptr %0, ptr %1, ptr %2, ptr %3, double noundef %4, ptr noundef %5) local_unnamed_addr #10 comdat {
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
  br i1 %47, label %48, label %40, !llvm.loop !101

48:                                               ; preds = %44
  %49 = load i32, ptr @current_test, align 4, !tbaa !11
  %50 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.52, i32 noundef %49)
  %51 = load i32, ptr @iterations, align 4, !tbaa !11
  br label %52

52:                                               ; preds = %40, %48
  %53 = phi i32 [ %51, %48 ], [ %38, %40 ]
  %54 = add nuw nsw i32 %39, 1
  %55 = icmp slt i32 %54, %53
  br i1 %55, label %37, label %205, !llvm.loop !102

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
  br i1 %70, label %71, label %59, !llvm.loop !103

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
  br i1 %81, label %82, label %75, !llvm.loop !104

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
  br i1 %90, label %91, label %83, !llvm.loop !101

91:                                               ; preds = %87
  %92 = load i32, ptr @current_test, align 4, !tbaa !11
  %93 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.52, i32 noundef %92)
  %94 = load i32, ptr @iterations, align 4, !tbaa !11
  br label %95

95:                                               ; preds = %83, %91
  %96 = phi i32 [ %94, %91 ], [ %57, %83 ]
  %97 = add nuw nsw i32 %58, 1
  %98 = icmp slt i32 %97, %96
  br i1 %98, label %56, label %205, !llvm.loop !102

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
  br i1 %127, label %128, label %121, !llvm.loop !105

128:                                              ; preds = %126, %121
  %129 = phi ptr [ %2, %126 ], [ %122, %121 ]
  store double %120, ptr %129, align 8, !tbaa !33
  %130 = getelementptr inbounds nuw i8, ptr %119, i64 8
  %131 = icmp eq ptr %130, %3
  br i1 %131, label %132, label %118, !llvm.loop !106

132:                                              ; preds = %128, %136
  %133 = phi ptr [ %134, %136 ], [ %2, %128 ]
  %134 = getelementptr i8, ptr %133, i64 8
  %135 = icmp eq ptr %134, %3
  br i1 %135, label %144, label %136

136:                                              ; preds = %132
  %137 = load double, ptr %134, align 8, !tbaa !33
  %138 = load double, ptr %133, align 8, !tbaa !33
  %139 = fcmp olt double %137, %138
  br i1 %139, label %140, label %132, !llvm.loop !101

140:                                              ; preds = %136
  %141 = load i32, ptr @current_test, align 4, !tbaa !11
  %142 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.52, i32 noundef %141)
  %143 = load i32, ptr @iterations, align 4, !tbaa !11
  br label %144

144:                                              ; preds = %132, %140
  %145 = phi i32 [ %143, %140 ], [ %116, %132 ]
  %146 = add nuw nsw i32 %117, 1
  %147 = icmp slt i32 %146, %145
  br i1 %147, label %115, label %205, !llvm.loop !102

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
  br i1 %162, label %163, label %151, !llvm.loop !107

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
  br i1 %173, label %174, label %167, !llvm.loop !108

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
  br i1 %184, label %185, label %178, !llvm.loop !105

185:                                              ; preds = %183, %178
  %186 = phi ptr [ %2, %183 ], [ %179, %178 ]
  store double %177, ptr %186, align 8, !tbaa !33
  %187 = getelementptr inbounds nuw i8, ptr %176, i64 8
  %188 = icmp eq ptr %187, %3
  br i1 %188, label %189, label %175, !llvm.loop !106

189:                                              ; preds = %185, %193
  %190 = phi ptr [ %191, %193 ], [ %2, %185 ]
  %191 = getelementptr i8, ptr %190, i64 8
  %192 = icmp eq ptr %191, %3
  br i1 %192, label %201, label %193

193:                                              ; preds = %189
  %194 = load double, ptr %191, align 8, !tbaa !33
  %195 = load double, ptr %190, align 8, !tbaa !33
  %196 = fcmp olt double %194, %195
  br i1 %196, label %197, label %189, !llvm.loop !101

197:                                              ; preds = %193
  %198 = load i32, ptr @current_test, align 4, !tbaa !11
  %199 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.52, i32 noundef %198)
  %200 = load i32, ptr @iterations, align 4, !tbaa !11
  br label %201

201:                                              ; preds = %189, %197
  %202 = phi i32 [ %200, %197 ], [ %149, %189 ]
  %203 = add nuw nsw i32 %150, 1
  %204 = icmp slt i32 %203, %202
  br i1 %204, label %148, label %205, !llvm.loop !102

205:                                              ; preds = %201, %144, %95, %52, %6
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_Z14test_quicksortIPddEvT_S1_S1_S1_T0_PKc(ptr noundef %0, ptr noundef %1, ptr noundef %2, ptr noundef %3, double noundef %4, ptr noundef %5) local_unnamed_addr #10 comdat {
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
  br i1 %39, label %40, label %32, !llvm.loop !93

40:                                               ; preds = %36
  %41 = load i32, ptr @current_test, align 4, !tbaa !11
  %42 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.52, i32 noundef %41)
  br label %43

43:                                               ; preds = %32, %40
  %44 = add nuw nsw i32 %31, 1
  %45 = load i32, ptr @iterations, align 4, !tbaa !11
  %46 = icmp slt i32 %44, %45
  br i1 %46, label %30, label %88, !llvm.loop !109

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
  br i1 %60, label %61, label %49, !llvm.loop !110

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
  br i1 %71, label %72, label %65, !llvm.loop !111

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
  br i1 %80, label %81, label %73, !llvm.loop !93

81:                                               ; preds = %77
  %82 = load i32, ptr @current_test, align 4, !tbaa !11
  %83 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.52, i32 noundef %82)
  br label %84

84:                                               ; preds = %73, %81
  %85 = add nuw nsw i32 %48, 1
  %86 = load i32, ptr @iterations, align 4, !tbaa !11
  %87 = icmp slt i32 %85, %86
  br i1 %87, label %47, label %88, !llvm.loop !109

88:                                               ; preds = %84, %43, %6
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_Z14test_quicksortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEvT_S7_S7_S7_T0_PKc(ptr %0, ptr %1, ptr %2, ptr %3, double noundef %4, ptr noundef %5) local_unnamed_addr #10 comdat {
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
  tail call void @_ZN9benchmark9quicksortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEEvT_S8_(ptr %2, ptr %3)
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
  br i1 %39, label %40, label %32, !llvm.loop !101

40:                                               ; preds = %36
  %41 = load i32, ptr @current_test, align 4, !tbaa !11
  %42 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.52, i32 noundef %41)
  br label %43

43:                                               ; preds = %32, %40
  %44 = add nuw nsw i32 %31, 1
  %45 = load i32, ptr @iterations, align 4, !tbaa !11
  %46 = icmp slt i32 %44, %45
  br i1 %46, label %30, label %88, !llvm.loop !112

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
  br i1 %60, label %61, label %49, !llvm.loop !113

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
  br i1 %71, label %72, label %65, !llvm.loop !114

72:                                               ; preds = %65, %61
  tail call void @_ZN9benchmark9quicksortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEEvT_S8_(ptr %2, ptr %3)
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
  br i1 %80, label %81, label %73, !llvm.loop !101

81:                                               ; preds = %77
  %82 = load i32, ptr @current_test, align 4, !tbaa !11
  %83 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.52, i32 noundef %82)
  br label %84

84:                                               ; preds = %73, %81
  %85 = add nuw nsw i32 %48, 1
  %86 = load i32, ptr @iterations, align 4, !tbaa !11
  %87 = icmp slt i32 %85, %86
  br i1 %87, label %47, label %88, !llvm.loop !112

88:                                               ; preds = %84, %43, %6
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_Z14test_heap_sortIPddEvT_S1_S1_S1_T0_PKc(ptr noundef %0, ptr noundef %1, ptr noundef %2, ptr noundef %3, double noundef %4, ptr noundef %5) local_unnamed_addr #10 comdat {
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
  br i1 %39, label %40, label %32, !llvm.loop !93

40:                                               ; preds = %36
  %41 = load i32, ptr @current_test, align 4, !tbaa !11
  %42 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.52, i32 noundef %41)
  br label %43

43:                                               ; preds = %32, %40
  %44 = add nuw nsw i32 %31, 1
  %45 = load i32, ptr @iterations, align 4, !tbaa !11
  %46 = icmp slt i32 %44, %45
  br i1 %46, label %30, label %88, !llvm.loop !115

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
  br i1 %60, label %61, label %49, !llvm.loop !116

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
  br i1 %71, label %72, label %65, !llvm.loop !117

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
  br i1 %80, label %81, label %73, !llvm.loop !93

81:                                               ; preds = %77
  %82 = load i32, ptr @current_test, align 4, !tbaa !11
  %83 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.52, i32 noundef %82)
  br label %84

84:                                               ; preds = %73, %81
  %85 = add nuw nsw i32 %48, 1
  %86 = load i32, ptr @iterations, align 4, !tbaa !11
  %87 = icmp slt i32 %85, %86
  br i1 %87, label %47, label %88, !llvm.loop !115

88:                                               ; preds = %84, %43, %6
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_Z14test_heap_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEvT_S7_S7_S7_T0_PKc(ptr %0, ptr %1, ptr %2, ptr %3, double noundef %4, ptr noundef %5) local_unnamed_addr #10 comdat {
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
  tail call void @_ZN9benchmark8heapsortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEEvT_S8_(ptr %2, ptr %3)
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
  br i1 %39, label %40, label %32, !llvm.loop !101

40:                                               ; preds = %36
  %41 = load i32, ptr @current_test, align 4, !tbaa !11
  %42 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.52, i32 noundef %41)
  br label %43

43:                                               ; preds = %32, %40
  %44 = add nuw nsw i32 %31, 1
  %45 = load i32, ptr @iterations, align 4, !tbaa !11
  %46 = icmp slt i32 %44, %45
  br i1 %46, label %30, label %88, !llvm.loop !118

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
  br i1 %60, label %61, label %49, !llvm.loop !119

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
  br i1 %71, label %72, label %65, !llvm.loop !120

72:                                               ; preds = %65, %61
  tail call void @_ZN9benchmark8heapsortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEEvT_S8_(ptr %2, ptr %3)
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
  br i1 %80, label %81, label %73, !llvm.loop !101

81:                                               ; preds = %77
  %82 = load i32, ptr @current_test, align 4, !tbaa !11
  %83 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.52, i32 noundef %82)
  br label %84

84:                                               ; preds = %73, %81
  %85 = add nuw nsw i32 %48, 1
  %86 = load i32, ptr @iterations, align 4, !tbaa !11
  %87 = icmp slt i32 %85, %86
  br i1 %87, label %47, label %88, !llvm.loop !118

88:                                               ; preds = %84, %43, %6
  ret void
}

; Function Attrs: nounwind
declare i64 @__isoc23_strtol(ptr noundef, ptr noundef, i32 noundef) local_unnamed_addr #8

; Function Attrs: mustprogress nocallback nofree nounwind willreturn
declare double @strtod(ptr noundef readonly, ptr noundef captures(none)) local_unnamed_addr #11

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #12

; Function Attrs: nobuiltin nounwind
declare void @_ZdlPvm(ptr noundef, i64 noundef) local_unnamed_addr #13

; Function Attrs: nobuiltin allocsize(0)
declare noundef nonnull ptr @_Znwm(i64 noundef) local_unnamed_addr #14

; Function Attrs: nounwind
declare i32 @rand() local_unnamed_addr #8

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZN9benchmark9quicksortIPddEEvT_S2_(ptr noundef %0, ptr noundef %1) local_unnamed_addr #10 comdat {
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
  br i1 %17, label %13, label %18, !llvm.loop !121

18:                                               ; preds = %13
  %19 = icmp ult ptr %12, %15
  br i1 %19, label %20, label %28

20:                                               ; preds = %18, %20
  %21 = phi ptr [ %24, %20 ], [ %12, %18 ]
  %22 = load double, ptr %21, align 8, !tbaa !33
  %23 = fcmp olt double %22, %9
  %24 = getelementptr inbounds nuw i8, ptr %21, i64 8
  br i1 %23, label %20, label %25, !llvm.loop !122

25:                                               ; preds = %20
  %26 = icmp ult ptr %21, %15
  br i1 %26, label %27, label %28

27:                                               ; preds = %25
  store double %22, ptr %15, align 8, !tbaa !33
  store double %16, ptr %21, align 8, !tbaa !33
  br label %10, !llvm.loop !123

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
define linkonce_odr dso_local void @_ZN9benchmark9quicksortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEEvT_S8_(ptr %0, ptr %1) local_unnamed_addr #10 comdat {
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
  br i1 %17, label %13, label %18, !llvm.loop !124

18:                                               ; preds = %13
  %19 = icmp ult ptr %11, %15
  br i1 %19, label %20, label %28

20:                                               ; preds = %18, %20
  %21 = phi ptr [ %24, %20 ], [ %11, %18 ]
  %22 = load double, ptr %21, align 8, !tbaa !33
  %23 = fcmp olt double %22, %9
  %24 = getelementptr inbounds nuw i8, ptr %21, i64 8
  br i1 %23, label %20, label %25, !llvm.loop !125

25:                                               ; preds = %20
  %26 = icmp ult ptr %21, %15
  br i1 %26, label %27, label %28

27:                                               ; preds = %25
  store double %22, ptr %15, align 8, !tbaa !33
  store double %16, ptr %21, align 8, !tbaa !33
  br label %10, !llvm.loop !126

28:                                               ; preds = %25, %18
  tail call void @_ZN9benchmark9quicksortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEEvT_S8_(ptr nonnull %8, ptr nonnull %14)
  %29 = ptrtoint ptr %14 to i64
  %30 = sub i64 %3, %29
  %31 = icmp sgt i64 %30, 8
  br i1 %31, label %7, label %32

32:                                               ; preds = %28, %2
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZN9benchmark9quicksortISt16reverse_iteratorIPdEdEEvT_S4_(ptr dead_on_return noundef %0, ptr dead_on_return noundef %1) local_unnamed_addr #10 comdat {
  %3 = alloca %"class.std::reverse_iterator", align 8
  %4 = alloca %"class.std::reverse_iterator", align 8
  %5 = alloca %"class.std::reverse_iterator", align 8
  %6 = alloca %"class.std::reverse_iterator", align 8
  %7 = load ptr, ptr %0, align 8, !tbaa !43
  %8 = load ptr, ptr %1, align 8, !tbaa !43
  %9 = ptrtoint ptr %7 to i64
  %10 = ptrtoint ptr %8 to i64
  %11 = sub i64 %9, %10
  %12 = icmp sgt i64 %11, 8
  br i1 %12, label %13, label %36

13:                                               ; preds = %2
  %14 = getelementptr inbounds i8, ptr %7, i64 -8
  %15 = load double, ptr %14, align 8, !tbaa !33
  br label %16

16:                                               ; preds = %33, %13
  %17 = phi ptr [ %7, %13 ], [ %27, %33 ]
  %18 = phi ptr [ %8, %13 ], [ %21, %33 ]
  br label %19

19:                                               ; preds = %19, %16
  %20 = phi ptr [ %18, %16 ], [ %21, %19 ]
  %21 = getelementptr inbounds nuw i8, ptr %20, i64 8
  %22 = load double, ptr %20, align 8, !tbaa !33
  %23 = fcmp olt double %15, %22
  br i1 %23, label %19, label %24, !llvm.loop !127

24:                                               ; preds = %19
  %25 = icmp ult ptr %21, %17
  br i1 %25, label %26, label %34

26:                                               ; preds = %24, %26
  %27 = phi ptr [ %28, %26 ], [ %17, %24 ]
  %28 = getelementptr inbounds i8, ptr %27, i64 -8
  %29 = load double, ptr %28, align 8, !tbaa !33
  %30 = fcmp olt double %29, %15
  br i1 %30, label %26, label %31, !llvm.loop !128

31:                                               ; preds = %26
  %32 = icmp ult ptr %21, %27
  br i1 %32, label %33, label %34

33:                                               ; preds = %31
  store double %29, ptr %20, align 8, !tbaa !33
  store double %22, ptr %28, align 8, !tbaa !33
  br label %16, !llvm.loop !129

34:                                               ; preds = %31, %24
  store ptr %7, ptr %3, align 8, !tbaa !43
  store ptr %20, ptr %4, align 8, !tbaa !43, !alias.scope !130
  call void @_ZN9benchmark9quicksortISt16reverse_iteratorIPdEdEEvT_S4_(ptr dead_on_return noundef nonnull %3, ptr dead_on_return noundef nonnull %4)
  store ptr %20, ptr %5, align 8, !tbaa !43, !alias.scope !133
  %35 = load ptr, ptr %1, align 8, !tbaa !43
  store ptr %35, ptr %6, align 8, !tbaa !43
  call void @_ZN9benchmark9quicksortISt16reverse_iteratorIPdEdEEvT_S4_(ptr dead_on_return noundef nonnull %5, ptr dead_on_return noundef nonnull %6)
  br label %36

36:                                               ; preds = %34, %2
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZN9benchmark9quicksortISt16reverse_iteratorIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEdEEvT_SA_(ptr dead_on_return noundef %0, ptr dead_on_return noundef %1) local_unnamed_addr #10 comdat {
  %3 = alloca %"class.std::reverse_iterator.1", align 8
  %4 = alloca %"class.std::reverse_iterator.1", align 8
  %5 = alloca %"class.std::reverse_iterator.1", align 8
  %6 = alloca %"class.std::reverse_iterator.1", align 8
  %7 = load ptr, ptr %0, align 8
  %8 = ptrtoint ptr %7 to i64
  %9 = load ptr, ptr %1, align 8
  %10 = ptrtoint ptr %9 to i64
  %11 = sub i64 %8, %10
  %12 = icmp sgt i64 %11, 8
  br i1 %12, label %13, label %36

13:                                               ; preds = %2
  %14 = getelementptr inbounds i8, ptr %7, i64 -8
  %15 = load double, ptr %14, align 8, !tbaa !33
  br label %16

16:                                               ; preds = %33, %13
  %17 = phi ptr [ %7, %13 ], [ %27, %33 ]
  %18 = phi ptr [ %9, %13 ], [ %21, %33 ]
  br label %19

19:                                               ; preds = %19, %16
  %20 = phi ptr [ %18, %16 ], [ %21, %19 ]
  %21 = getelementptr inbounds nuw i8, ptr %20, i64 8
  %22 = load double, ptr %20, align 8, !tbaa !33
  %23 = fcmp olt double %15, %22
  br i1 %23, label %19, label %24, !llvm.loop !136

24:                                               ; preds = %19
  %25 = icmp ult ptr %21, %17
  br i1 %25, label %26, label %34

26:                                               ; preds = %24, %26
  %27 = phi ptr [ %28, %26 ], [ %17, %24 ]
  %28 = getelementptr inbounds i8, ptr %27, i64 -8
  %29 = load double, ptr %28, align 8, !tbaa !33
  %30 = fcmp olt double %29, %15
  br i1 %30, label %26, label %31, !llvm.loop !137

31:                                               ; preds = %26
  %32 = icmp ult ptr %21, %27
  br i1 %32, label %33, label %34

33:                                               ; preds = %31
  store double %29, ptr %20, align 8, !tbaa !33
  store double %22, ptr %28, align 8, !tbaa !33
  br label %16, !llvm.loop !138

34:                                               ; preds = %31, %24
  store i64 %8, ptr %3, align 8, !tbaa !34
  store ptr %20, ptr %4, align 8, !tbaa !34, !alias.scope !139
  call void @_ZN9benchmark9quicksortISt16reverse_iteratorIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEdEEvT_SA_(ptr dead_on_return noundef nonnull %3, ptr dead_on_return noundef nonnull %4)
  store ptr %20, ptr %5, align 8, !tbaa !34, !alias.scope !142
  %35 = load i64, ptr %1, align 8, !tbaa !34
  store i64 %35, ptr %6, align 8, !tbaa !34
  call void @_ZN9benchmark9quicksortISt16reverse_iteratorIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEdEEvT_SA_(ptr dead_on_return noundef nonnull %5, ptr dead_on_return noundef nonnull %6)
  br label %36

36:                                               ; preds = %34, %2
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZN9benchmark9quicksortISt16reverse_iteratorIS1_IPdEEdEEvT_S5_(ptr dead_on_return noundef %0, ptr dead_on_return noundef %1) local_unnamed_addr #10 comdat {
  %3 = alloca %"class.std::reverse_iterator.0", align 8
  %4 = alloca %"class.std::reverse_iterator.0", align 8
  %5 = alloca %"class.std::reverse_iterator.0", align 8
  %6 = alloca %"class.std::reverse_iterator.0", align 8
  %7 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %8 = load ptr, ptr %7, align 8, !tbaa !43, !noalias !145
  %9 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %10 = load ptr, ptr %9, align 8, !tbaa !43, !noalias !148
  %11 = ptrtoint ptr %10 to i64
  %12 = ptrtoint ptr %8 to i64
  %13 = sub i64 %11, %12
  %14 = icmp sgt i64 %13, 8
  br i1 %14, label %15, label %41

15:                                               ; preds = %2
  %16 = load double, ptr %8, align 8, !tbaa !33
  br label %17

17:                                               ; preds = %34, %15
  %18 = phi ptr [ %10, %15 ], [ %22, %34 ]
  %19 = phi ptr [ %8, %15 ], [ %28, %34 ]
  br label %20

20:                                               ; preds = %20, %17
  %21 = phi ptr [ %18, %17 ], [ %22, %20 ]
  %22 = getelementptr inbounds i8, ptr %21, i64 -8
  %23 = load double, ptr %22, align 8, !tbaa !33
  %24 = fcmp olt double %16, %23
  br i1 %24, label %20, label %25, !llvm.loop !151

25:                                               ; preds = %20
  %26 = icmp ult ptr %19, %22
  br i1 %26, label %27, label %35

27:                                               ; preds = %25, %27
  %28 = phi ptr [ %31, %27 ], [ %19, %25 ]
  %29 = load double, ptr %28, align 8, !tbaa !33
  %30 = fcmp olt double %29, %16
  %31 = getelementptr inbounds nuw i8, ptr %28, i64 8
  br i1 %30, label %27, label %32, !llvm.loop !152

32:                                               ; preds = %27
  %33 = icmp ult ptr %28, %22
  br i1 %33, label %34, label %35

34:                                               ; preds = %32
  store double %29, ptr %22, align 8, !tbaa !33
  store double %23, ptr %28, align 8, !tbaa !33
  br label %17, !llvm.loop !153

35:                                               ; preds = %32, %25
  %36 = getelementptr inbounds nuw i8, ptr %3, i64 8
  store ptr %8, ptr %36, align 8, !tbaa !43
  %37 = getelementptr inbounds nuw i8, ptr %4, i64 8
  store ptr %21, ptr %37, align 8, !tbaa !43, !alias.scope !154
  call void @_ZN9benchmark9quicksortISt16reverse_iteratorIS1_IPdEEdEEvT_S5_(ptr dead_on_return noundef nonnull %3, ptr dead_on_return noundef nonnull %4)
  %38 = getelementptr inbounds nuw i8, ptr %5, i64 8
  store ptr %21, ptr %38, align 8, !tbaa !43, !alias.scope !157
  %39 = getelementptr inbounds nuw i8, ptr %6, i64 8
  %40 = load ptr, ptr %9, align 8, !tbaa !43
  store ptr %40, ptr %39, align 8, !tbaa !43
  call void @_ZN9benchmark9quicksortISt16reverse_iteratorIS1_IPdEEdEEvT_S5_(ptr dead_on_return noundef nonnull %5, ptr dead_on_return noundef nonnull %6)
  br label %41

41:                                               ; preds = %35, %2
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZN9benchmark9quicksortISt16reverse_iteratorIS1_IN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEEdEEvT_SB_(ptr dead_on_return noundef %0, ptr dead_on_return noundef %1) local_unnamed_addr #10 comdat {
  %3 = alloca %"class.std::reverse_iterator.2", align 8
  %4 = alloca %"class.std::reverse_iterator.2", align 8
  %5 = alloca %"class.std::reverse_iterator.2", align 8
  %6 = alloca %"class.std::reverse_iterator.2", align 8
  %7 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %8 = load i64, ptr %7, align 8, !tbaa !34, !noalias !160
  %9 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %10 = load i64, ptr %9, align 8, !tbaa !34, !noalias !163
  %11 = sub i64 %10, %8
  %12 = icmp sgt i64 %11, 8
  br i1 %12, label %13, label %42

13:                                               ; preds = %2
  %14 = inttoptr i64 %8 to ptr
  %15 = load double, ptr %14, align 8, !tbaa !33
  %16 = inttoptr i64 %10 to ptr
  br label %17

17:                                               ; preds = %34, %13
  %18 = phi ptr [ %16, %13 ], [ %22, %34 ]
  %19 = phi ptr [ %14, %13 ], [ %28, %34 ]
  br label %20

20:                                               ; preds = %20, %17
  %21 = phi ptr [ %18, %17 ], [ %22, %20 ]
  %22 = getelementptr inbounds i8, ptr %21, i64 -8
  %23 = load double, ptr %22, align 8, !tbaa !33
  %24 = fcmp olt double %15, %23
  br i1 %24, label %20, label %25, !llvm.loop !166

25:                                               ; preds = %20
  %26 = icmp ult ptr %19, %22
  br i1 %26, label %27, label %35

27:                                               ; preds = %25, %27
  %28 = phi ptr [ %31, %27 ], [ %19, %25 ]
  %29 = load double, ptr %28, align 8, !tbaa !33
  %30 = fcmp olt double %29, %15
  %31 = getelementptr inbounds nuw i8, ptr %28, i64 8
  br i1 %30, label %27, label %32, !llvm.loop !167

32:                                               ; preds = %27
  %33 = icmp ult ptr %28, %22
  br i1 %33, label %34, label %35

34:                                               ; preds = %32
  store double %29, ptr %22, align 8, !tbaa !33
  store double %23, ptr %28, align 8, !tbaa !33
  br label %17, !llvm.loop !168

35:                                               ; preds = %32, %25
  %36 = getelementptr inbounds nuw i8, ptr %3, i64 8
  store i64 %8, ptr %36, align 8, !tbaa !34
  %37 = getelementptr inbounds nuw i8, ptr %4, i64 8
  %38 = ptrtoint ptr %21 to i64
  store i64 %38, ptr %37, align 8, !tbaa !34, !alias.scope !169
  call void @_ZN9benchmark9quicksortISt16reverse_iteratorIS1_IN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEEdEEvT_SB_(ptr dead_on_return noundef nonnull %3, ptr dead_on_return noundef nonnull %4)
  %39 = getelementptr inbounds nuw i8, ptr %5, i64 8
  store i64 %38, ptr %39, align 8, !tbaa !34, !alias.scope !172
  %40 = getelementptr inbounds nuw i8, ptr %6, i64 8
  %41 = load i64, ptr %9, align 8, !tbaa !34
  store i64 %41, ptr %40, align 8, !tbaa !34
  call void @_ZN9benchmark9quicksortISt16reverse_iteratorIS1_IN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEEdEEvT_SB_(ptr dead_on_return noundef nonnull %5, ptr dead_on_return noundef nonnull %6)
  br label %42

42:                                               ; preds = %35, %2
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZN9benchmark8heapsortIPddEEvT_S2_(ptr noundef %0, ptr noundef %1) local_unnamed_addr #10 comdat {
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
  br i1 %35, label %20, label %36, !llvm.loop !175

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
  br i1 %55, label %56, label %46, !llvm.loop !176

56:                                               ; preds = %46, %53, %43
  %57 = phi i64 [ %44, %43 ], [ %47, %46 ], [ %49, %53 ]
  %58 = getelementptr inbounds double, ptr %0, i64 %57
  store double %16, ptr %58, align 8, !tbaa !33
  %59 = icmp sgt i64 %13, 1
  br i1 %59, label %12, label %60, !llvm.loop !177

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
  br i1 %82, label %67, label %83, !llvm.loop !175

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
  br i1 %105, label %106, label %96, !llvm.loop !176

106:                                              ; preds = %96, %103, %92
  %107 = phi i64 [ %85, %92 ], [ %97, %96 ], [ 0, %103 ]
  %108 = getelementptr inbounds double, ptr %0, i64 %107
  store double %64, ptr %108, align 8, !tbaa !33
  %109 = icmp sgt i64 %61, 2
  br i1 %109, label %60, label %110, !llvm.loop !178

110:                                              ; preds = %106, %2
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZN9benchmark8heapsortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEEvT_S8_(ptr %0, ptr %1) local_unnamed_addr #10 comdat {
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
  br i1 %35, label %20, label %36, !llvm.loop !179

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
  br i1 %55, label %56, label %46, !llvm.loop !180

56:                                               ; preds = %46, %53, %43
  %57 = phi i64 [ %44, %43 ], [ %47, %46 ], [ %49, %53 ]
  %58 = getelementptr inbounds double, ptr %0, i64 %57
  store double %16, ptr %58, align 8, !tbaa !33
  %59 = icmp sgt i64 %13, 1
  br i1 %59, label %12, label %60, !llvm.loop !181

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
  br i1 %82, label %67, label %83, !llvm.loop !179

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
  br i1 %105, label %106, label %96, !llvm.loop !180

106:                                              ; preds = %96, %103, %92
  %107 = phi i64 [ %85, %92 ], [ %97, %96 ], [ 0, %103 ]
  %108 = getelementptr inbounds double, ptr %0, i64 %107
  store double %64, ptr %108, align 8, !tbaa !33
  %109 = icmp sgt i64 %61, 2
  br i1 %109, label %60, label %110, !llvm.loop !182

110:                                              ; preds = %106, %2
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZN9benchmark8heapsortISt16reverse_iteratorIPdEdEEvT_S4_(ptr dead_on_return noundef %0, ptr dead_on_return noundef %1) local_unnamed_addr #10 comdat {
  %3 = load ptr, ptr %0, align 8, !tbaa !43
  %4 = load ptr, ptr %1, align 8, !tbaa !43
  %5 = ptrtoint ptr %3 to i64
  %6 = ptrtoint ptr %4 to i64
  %7 = sub i64 %5, %6
  %8 = ashr exact i64 %7, 3
  %9 = icmp sgt i64 %8, 1
  br i1 %9, label %10, label %154

10:                                               ; preds = %2
  %11 = lshr i64 %8, 1
  %12 = add nsw i64 %8, -1
  %13 = sub nsw i64 1, %8
  %14 = getelementptr inbounds double, ptr %3, i64 %13
  %15 = getelementptr inbounds i8, ptr %14, i64 -8
  br label %18

16:                                               ; preds = %78
  %17 = getelementptr inbounds i8, ptr %3, i64 -8
  br label %84

18:                                               ; preds = %10, %78
  %19 = phi i64 [ %11, %10 ], [ %20, %78 ]
  %20 = add nsw i64 %19, -1
  %21 = sub nsw i64 1, %19
  %22 = getelementptr inbounds double, ptr %3, i64 %21
  %23 = getelementptr inbounds i8, ptr %22, i64 -8
  %24 = load double, ptr %23, align 8, !tbaa !33
  %25 = shl nuw i64 %20, 1
  %26 = add nuw nsw i64 %25, 2
  %27 = icmp slt i64 %26, %8
  br i1 %27, label %28, label %52

28:                                               ; preds = %18, %28
  %29 = phi i64 [ %42, %28 ], [ %20, %18 ]
  %30 = phi i64 [ %50, %28 ], [ %26, %18 ]
  %31 = sub i64 1, %30
  %32 = getelementptr inbounds double, ptr %3, i64 %31
  %33 = getelementptr inbounds i8, ptr %32, i64 -8
  %34 = load double, ptr %33, align 8, !tbaa !33
  %35 = sub i64 0, %30
  %36 = getelementptr inbounds double, ptr %3, i64 %35
  %37 = getelementptr inbounds i8, ptr %36, i64 -8
  %38 = load double, ptr %37, align 8, !tbaa !33
  %39 = fcmp olt double %34, %38
  %40 = zext i1 %39 to i64
  %41 = add nsw i64 %30, %40
  %42 = add nsw i64 %41, -1
  %43 = sub i64 1, %41
  %44 = getelementptr inbounds double, ptr %3, i64 %43
  %45 = getelementptr inbounds i8, ptr %44, i64 -8
  %46 = load double, ptr %45, align 8, !tbaa !33
  %47 = sub i64 0, %29
  %48 = getelementptr inbounds double, ptr %3, i64 %47
  %49 = getelementptr inbounds i8, ptr %48, i64 -8
  store double %46, ptr %49, align 8, !tbaa !33
  %50 = shl nsw i64 %41, 1
  %51 = icmp slt i64 %50, %8
  br i1 %51, label %28, label %52, !llvm.loop !183

52:                                               ; preds = %28, %18
  %53 = phi i64 [ %26, %18 ], [ %50, %28 ]
  %54 = phi i64 [ %20, %18 ], [ %42, %28 ]
  %55 = icmp eq i64 %53, %8
  br i1 %55, label %56, label %61

56:                                               ; preds = %52
  %57 = load double, ptr %15, align 8, !tbaa !33
  %58 = sub i64 0, %54
  %59 = getelementptr inbounds double, ptr %3, i64 %58
  %60 = getelementptr inbounds i8, ptr %59, i64 -8
  store double %57, ptr %60, align 8, !tbaa !33
  br label %61

61:                                               ; preds = %56, %52
  %62 = phi i64 [ %12, %56 ], [ %54, %52 ]
  %63 = icmp slt i64 %62, %19
  br i1 %63, label %78, label %64

64:                                               ; preds = %61, %73
  %65 = phi i64 [ %67, %73 ], [ %62, %61 ]
  %66 = add nsw i64 %65, -1
  %67 = sdiv i64 %66, 2
  %68 = sub nsw i64 0, %67
  %69 = getelementptr inbounds double, ptr %3, i64 %68
  %70 = getelementptr inbounds i8, ptr %69, i64 -8
  %71 = load double, ptr %70, align 8, !tbaa !33
  %72 = fcmp olt double %71, %24
  br i1 %72, label %73, label %78

73:                                               ; preds = %64
  %74 = sub nsw i64 0, %65
  %75 = getelementptr inbounds double, ptr %3, i64 %74
  %76 = getelementptr inbounds i8, ptr %75, i64 -8
  store double %71, ptr %76, align 8, !tbaa !33
  %77 = icmp slt i64 %67, %19
  br i1 %77, label %78, label %64, !llvm.loop !184

78:                                               ; preds = %64, %73, %61
  %79 = phi i64 [ %62, %61 ], [ %67, %73 ], [ %65, %64 ]
  %80 = sub i64 0, %79
  %81 = getelementptr inbounds double, ptr %3, i64 %80
  %82 = getelementptr inbounds i8, ptr %81, i64 -8
  store double %24, ptr %82, align 8, !tbaa !33
  %83 = icmp sgt i64 %19, 1
  br i1 %83, label %18, label %16, !llvm.loop !185

84:                                               ; preds = %16, %148
  %85 = phi i64 [ %8, %16 ], [ %86, %148 ]
  %86 = add nsw i64 %85, -1
  %87 = sub nsw i64 1, %85
  %88 = getelementptr inbounds double, ptr %3, i64 %87
  %89 = getelementptr inbounds i8, ptr %88, i64 -8
  %90 = load double, ptr %89, align 8, !tbaa !33
  %91 = load double, ptr %17, align 8, !tbaa !33
  store double %91, ptr %89, align 8, !tbaa !33
  %92 = icmp samesign ugt i64 %86, 2
  br i1 %92, label %93, label %117

93:                                               ; preds = %84, %93
  %94 = phi i64 [ %107, %93 ], [ 0, %84 ]
  %95 = phi i64 [ %115, %93 ], [ 2, %84 ]
  %96 = sub i64 1, %95
  %97 = getelementptr inbounds double, ptr %3, i64 %96
  %98 = getelementptr inbounds i8, ptr %97, i64 -8
  %99 = load double, ptr %98, align 8, !tbaa !33
  %100 = sub i64 0, %95
  %101 = getelementptr inbounds double, ptr %3, i64 %100
  %102 = getelementptr inbounds i8, ptr %101, i64 -8
  %103 = load double, ptr %102, align 8, !tbaa !33
  %104 = fcmp olt double %99, %103
  %105 = zext i1 %104 to i64
  %106 = or disjoint i64 %95, %105
  %107 = add nsw i64 %106, -1
  %108 = sub i64 1, %106
  %109 = getelementptr inbounds double, ptr %3, i64 %108
  %110 = getelementptr inbounds i8, ptr %109, i64 -8
  %111 = load double, ptr %110, align 8, !tbaa !33
  %112 = sub i64 0, %94
  %113 = getelementptr inbounds double, ptr %3, i64 %112
  %114 = getelementptr inbounds i8, ptr %113, i64 -8
  store double %111, ptr %114, align 8, !tbaa !33
  %115 = shl nsw i64 %106, 1
  %116 = icmp slt i64 %115, %86
  br i1 %116, label %93, label %117, !llvm.loop !183

117:                                              ; preds = %93, %84
  %118 = phi i64 [ 2, %84 ], [ %115, %93 ]
  %119 = phi i64 [ 0, %84 ], [ %107, %93 ]
  %120 = icmp eq i64 %118, %86
  br i1 %120, label %121, label %130

121:                                              ; preds = %117
  %122 = add nsw i64 %85, -2
  %123 = sub nsw i64 2, %85
  %124 = getelementptr inbounds double, ptr %3, i64 %123
  %125 = getelementptr inbounds i8, ptr %124, i64 -8
  %126 = load double, ptr %125, align 8, !tbaa !33
  %127 = sub i64 0, %119
  %128 = getelementptr inbounds double, ptr %3, i64 %127
  %129 = getelementptr inbounds i8, ptr %128, i64 -8
  store double %126, ptr %129, align 8, !tbaa !33
  br label %132

130:                                              ; preds = %117
  %131 = icmp sgt i64 %119, 0
  br i1 %131, label %132, label %148

132:                                              ; preds = %121, %130
  %133 = phi i64 [ %119, %130 ], [ %122, %121 ]
  br label %134

134:                                              ; preds = %132, %143
  %135 = phi i64 [ %137, %143 ], [ %133, %132 ]
  %136 = add nsw i64 %135, -1
  %137 = lshr i64 %136, 1
  %138 = sub nsw i64 0, %137
  %139 = getelementptr inbounds double, ptr %3, i64 %138
  %140 = getelementptr inbounds i8, ptr %139, i64 -8
  %141 = load double, ptr %140, align 8, !tbaa !33
  %142 = fcmp olt double %141, %90
  br i1 %142, label %143, label %148

143:                                              ; preds = %134
  %144 = sub nsw i64 0, %135
  %145 = getelementptr inbounds double, ptr %3, i64 %144
  %146 = getelementptr inbounds i8, ptr %145, i64 -8
  store double %141, ptr %146, align 8, !tbaa !33
  %147 = icmp ult i64 %136, 2
  br i1 %147, label %148, label %134, !llvm.loop !184

148:                                              ; preds = %134, %143, %130
  %149 = phi i64 [ %119, %130 ], [ 0, %143 ], [ %135, %134 ]
  %150 = sub i64 0, %149
  %151 = getelementptr inbounds double, ptr %3, i64 %150
  %152 = getelementptr inbounds i8, ptr %151, i64 -8
  store double %90, ptr %152, align 8, !tbaa !33
  %153 = icmp sgt i64 %85, 2
  br i1 %153, label %84, label %154, !llvm.loop !186

154:                                              ; preds = %148, %2
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZN9benchmark8heapsortISt16reverse_iteratorIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEdEEvT_SA_(ptr dead_on_return noundef %0, ptr dead_on_return noundef %1) local_unnamed_addr #10 comdat {
  %3 = load ptr, ptr %0, align 8
  %4 = ptrtoint ptr %3 to i64
  %5 = load ptr, ptr %1, align 8, !tbaa !34
  %6 = ptrtoint ptr %5 to i64
  %7 = sub i64 %4, %6
  %8 = ashr exact i64 %7, 3
  %9 = icmp sgt i64 %8, 1
  br i1 %9, label %10, label %154

10:                                               ; preds = %2
  %11 = lshr i64 %8, 1
  %12 = add nsw i64 %8, -1
  %13 = sub nsw i64 1, %8
  %14 = getelementptr inbounds double, ptr %3, i64 %13
  %15 = getelementptr inbounds i8, ptr %14, i64 -8
  br label %18

16:                                               ; preds = %78
  %17 = getelementptr inbounds i8, ptr %3, i64 -8
  br label %84

18:                                               ; preds = %10, %78
  %19 = phi i64 [ %11, %10 ], [ %20, %78 ]
  %20 = add nsw i64 %19, -1
  %21 = sub nsw i64 1, %19
  %22 = getelementptr inbounds double, ptr %3, i64 %21
  %23 = getelementptr inbounds i8, ptr %22, i64 -8
  %24 = load double, ptr %23, align 8, !tbaa !33
  %25 = shl nuw i64 %20, 1
  %26 = add nuw nsw i64 %25, 2
  %27 = icmp slt i64 %26, %8
  br i1 %27, label %28, label %52

28:                                               ; preds = %18, %28
  %29 = phi i64 [ %42, %28 ], [ %20, %18 ]
  %30 = phi i64 [ %50, %28 ], [ %26, %18 ]
  %31 = sub i64 1, %30
  %32 = getelementptr inbounds double, ptr %3, i64 %31
  %33 = getelementptr inbounds i8, ptr %32, i64 -8
  %34 = load double, ptr %33, align 8, !tbaa !33
  %35 = sub i64 0, %30
  %36 = getelementptr inbounds double, ptr %3, i64 %35
  %37 = getelementptr inbounds i8, ptr %36, i64 -8
  %38 = load double, ptr %37, align 8, !tbaa !33
  %39 = fcmp olt double %34, %38
  %40 = zext i1 %39 to i64
  %41 = add nsw i64 %30, %40
  %42 = add nsw i64 %41, -1
  %43 = sub i64 1, %41
  %44 = getelementptr inbounds double, ptr %3, i64 %43
  %45 = getelementptr inbounds i8, ptr %44, i64 -8
  %46 = load double, ptr %45, align 8, !tbaa !33
  %47 = sub i64 0, %29
  %48 = getelementptr inbounds double, ptr %3, i64 %47
  %49 = getelementptr inbounds i8, ptr %48, i64 -8
  store double %46, ptr %49, align 8, !tbaa !33
  %50 = shl nsw i64 %41, 1
  %51 = icmp slt i64 %50, %8
  br i1 %51, label %28, label %52, !llvm.loop !187

52:                                               ; preds = %28, %18
  %53 = phi i64 [ %26, %18 ], [ %50, %28 ]
  %54 = phi i64 [ %20, %18 ], [ %42, %28 ]
  %55 = icmp eq i64 %53, %8
  br i1 %55, label %56, label %61

56:                                               ; preds = %52
  %57 = load double, ptr %15, align 8, !tbaa !33
  %58 = sub i64 0, %54
  %59 = getelementptr inbounds double, ptr %3, i64 %58
  %60 = getelementptr inbounds i8, ptr %59, i64 -8
  store double %57, ptr %60, align 8, !tbaa !33
  br label %61

61:                                               ; preds = %56, %52
  %62 = phi i64 [ %12, %56 ], [ %54, %52 ]
  %63 = icmp slt i64 %62, %19
  br i1 %63, label %78, label %64

64:                                               ; preds = %61, %73
  %65 = phi i64 [ %67, %73 ], [ %62, %61 ]
  %66 = add nsw i64 %65, -1
  %67 = sdiv i64 %66, 2
  %68 = sub nsw i64 0, %67
  %69 = getelementptr inbounds double, ptr %3, i64 %68
  %70 = getelementptr inbounds i8, ptr %69, i64 -8
  %71 = load double, ptr %70, align 8, !tbaa !33
  %72 = fcmp olt double %71, %24
  br i1 %72, label %73, label %78

73:                                               ; preds = %64
  %74 = sub nsw i64 0, %65
  %75 = getelementptr inbounds double, ptr %3, i64 %74
  %76 = getelementptr inbounds i8, ptr %75, i64 -8
  store double %71, ptr %76, align 8, !tbaa !33
  %77 = icmp slt i64 %67, %19
  br i1 %77, label %78, label %64, !llvm.loop !188

78:                                               ; preds = %64, %73, %61
  %79 = phi i64 [ %62, %61 ], [ %67, %73 ], [ %65, %64 ]
  %80 = sub i64 0, %79
  %81 = getelementptr inbounds double, ptr %3, i64 %80
  %82 = getelementptr inbounds i8, ptr %81, i64 -8
  store double %24, ptr %82, align 8, !tbaa !33
  %83 = icmp sgt i64 %19, 1
  br i1 %83, label %18, label %16, !llvm.loop !189

84:                                               ; preds = %16, %148
  %85 = phi i64 [ %8, %16 ], [ %86, %148 ]
  %86 = add nsw i64 %85, -1
  %87 = sub nsw i64 1, %85
  %88 = getelementptr inbounds double, ptr %3, i64 %87
  %89 = getelementptr inbounds i8, ptr %88, i64 -8
  %90 = load double, ptr %89, align 8, !tbaa !33
  %91 = load double, ptr %17, align 8, !tbaa !33
  store double %91, ptr %89, align 8, !tbaa !33
  %92 = icmp samesign ugt i64 %86, 2
  br i1 %92, label %93, label %117

93:                                               ; preds = %84, %93
  %94 = phi i64 [ %107, %93 ], [ 0, %84 ]
  %95 = phi i64 [ %115, %93 ], [ 2, %84 ]
  %96 = sub i64 1, %95
  %97 = getelementptr inbounds double, ptr %3, i64 %96
  %98 = getelementptr inbounds i8, ptr %97, i64 -8
  %99 = load double, ptr %98, align 8, !tbaa !33
  %100 = sub i64 0, %95
  %101 = getelementptr inbounds double, ptr %3, i64 %100
  %102 = getelementptr inbounds i8, ptr %101, i64 -8
  %103 = load double, ptr %102, align 8, !tbaa !33
  %104 = fcmp olt double %99, %103
  %105 = zext i1 %104 to i64
  %106 = or disjoint i64 %95, %105
  %107 = add nsw i64 %106, -1
  %108 = sub i64 1, %106
  %109 = getelementptr inbounds double, ptr %3, i64 %108
  %110 = getelementptr inbounds i8, ptr %109, i64 -8
  %111 = load double, ptr %110, align 8, !tbaa !33
  %112 = sub i64 0, %94
  %113 = getelementptr inbounds double, ptr %3, i64 %112
  %114 = getelementptr inbounds i8, ptr %113, i64 -8
  store double %111, ptr %114, align 8, !tbaa !33
  %115 = shl nsw i64 %106, 1
  %116 = icmp slt i64 %115, %86
  br i1 %116, label %93, label %117, !llvm.loop !187

117:                                              ; preds = %93, %84
  %118 = phi i64 [ 2, %84 ], [ %115, %93 ]
  %119 = phi i64 [ 0, %84 ], [ %107, %93 ]
  %120 = icmp eq i64 %118, %86
  br i1 %120, label %121, label %130

121:                                              ; preds = %117
  %122 = add nsw i64 %85, -2
  %123 = sub nsw i64 2, %85
  %124 = getelementptr inbounds double, ptr %3, i64 %123
  %125 = getelementptr inbounds i8, ptr %124, i64 -8
  %126 = load double, ptr %125, align 8, !tbaa !33
  %127 = sub i64 0, %119
  %128 = getelementptr inbounds double, ptr %3, i64 %127
  %129 = getelementptr inbounds i8, ptr %128, i64 -8
  store double %126, ptr %129, align 8, !tbaa !33
  br label %132

130:                                              ; preds = %117
  %131 = icmp sgt i64 %119, 0
  br i1 %131, label %132, label %148

132:                                              ; preds = %121, %130
  %133 = phi i64 [ %119, %130 ], [ %122, %121 ]
  br label %134

134:                                              ; preds = %132, %143
  %135 = phi i64 [ %137, %143 ], [ %133, %132 ]
  %136 = add nsw i64 %135, -1
  %137 = lshr i64 %136, 1
  %138 = sub nsw i64 0, %137
  %139 = getelementptr inbounds double, ptr %3, i64 %138
  %140 = getelementptr inbounds i8, ptr %139, i64 -8
  %141 = load double, ptr %140, align 8, !tbaa !33
  %142 = fcmp olt double %141, %90
  br i1 %142, label %143, label %148

143:                                              ; preds = %134
  %144 = sub nsw i64 0, %135
  %145 = getelementptr inbounds double, ptr %3, i64 %144
  %146 = getelementptr inbounds i8, ptr %145, i64 -8
  store double %141, ptr %146, align 8, !tbaa !33
  %147 = icmp ult i64 %136, 2
  br i1 %147, label %148, label %134, !llvm.loop !188

148:                                              ; preds = %134, %143, %130
  %149 = phi i64 [ %119, %130 ], [ 0, %143 ], [ %135, %134 ]
  %150 = sub i64 0, %149
  %151 = getelementptr inbounds double, ptr %3, i64 %150
  %152 = getelementptr inbounds i8, ptr %151, i64 -8
  store double %90, ptr %152, align 8, !tbaa !33
  %153 = icmp sgt i64 %85, 2
  br i1 %153, label %84, label %154, !llvm.loop !190

154:                                              ; preds = %148, %2
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZN9benchmark8heapsortISt16reverse_iteratorIS1_IPdEEdEEvT_S5_(ptr dead_on_return noundef %0, ptr dead_on_return noundef %1) local_unnamed_addr #10 comdat {
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %4 = load ptr, ptr %3, align 8, !tbaa !43, !noalias !191
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %6 = load ptr, ptr %5, align 8, !tbaa !43, !noalias !194
  %7 = ptrtoint ptr %6 to i64
  %8 = ptrtoint ptr %4 to i64
  %9 = sub i64 %7, %8
  %10 = ashr exact i64 %9, 3
  %11 = icmp sgt i64 %10, 1
  br i1 %11, label %12, label %114

12:                                               ; preds = %2
  %13 = lshr i64 %10, 1
  %14 = add nsw i64 %10, -1
  %15 = getelementptr inbounds nuw double, ptr %4, i64 %14
  br label %16

16:                                               ; preds = %12, %60
  %17 = phi i64 [ %13, %12 ], [ %18, %60 ]
  %18 = add nsw i64 %17, -1
  %19 = getelementptr inbounds double, ptr %4, i64 %18
  %20 = load double, ptr %19, align 8, !tbaa !33
  %21 = shl nuw i64 %18, 1
  %22 = add nuw nsw i64 %21, 2
  %23 = icmp slt i64 %22, %10
  br i1 %23, label %24, label %40

24:                                               ; preds = %16, %24
  %25 = phi i64 [ %34, %24 ], [ %18, %16 ]
  %26 = phi i64 [ %38, %24 ], [ %22, %16 ]
  %27 = getelementptr double, ptr %4, i64 %26
  %28 = getelementptr i8, ptr %27, i64 -8
  %29 = load double, ptr %28, align 8, !tbaa !33
  %30 = load double, ptr %27, align 8, !tbaa !33
  %31 = fcmp olt double %29, %30
  %32 = zext i1 %31 to i64
  %33 = add nsw i64 %26, %32
  %34 = add nsw i64 %33, -1
  %35 = getelementptr inbounds double, ptr %4, i64 %34
  %36 = load double, ptr %35, align 8, !tbaa !33
  %37 = getelementptr inbounds double, ptr %4, i64 %25
  store double %36, ptr %37, align 8, !tbaa !33
  %38 = shl nsw i64 %33, 1
  %39 = icmp slt i64 %38, %10
  br i1 %39, label %24, label %40, !llvm.loop !197

40:                                               ; preds = %24, %16
  %41 = phi i64 [ %22, %16 ], [ %38, %24 ]
  %42 = phi i64 [ %18, %16 ], [ %34, %24 ]
  %43 = icmp eq i64 %41, %10
  br i1 %43, label %44, label %47

44:                                               ; preds = %40
  %45 = load double, ptr %15, align 8, !tbaa !33
  %46 = getelementptr inbounds double, ptr %4, i64 %42
  store double %45, ptr %46, align 8, !tbaa !33
  br label %47

47:                                               ; preds = %44, %40
  %48 = phi i64 [ %14, %44 ], [ %42, %40 ]
  %49 = icmp slt i64 %48, %17
  br i1 %49, label %60, label %50

50:                                               ; preds = %47, %57
  %51 = phi i64 [ %53, %57 ], [ %48, %47 ]
  %52 = add nsw i64 %51, -1
  %53 = sdiv i64 %52, 2
  %54 = getelementptr inbounds double, ptr %4, i64 %53
  %55 = load double, ptr %54, align 8, !tbaa !33
  %56 = fcmp olt double %55, %20
  br i1 %56, label %57, label %60

57:                                               ; preds = %50
  %58 = getelementptr inbounds double, ptr %4, i64 %51
  store double %55, ptr %58, align 8, !tbaa !33
  %59 = icmp slt i64 %53, %17
  br i1 %59, label %60, label %50, !llvm.loop !198

60:                                               ; preds = %50, %57, %47
  %61 = phi i64 [ %48, %47 ], [ %51, %50 ], [ %53, %57 ]
  %62 = getelementptr inbounds double, ptr %4, i64 %61
  store double %20, ptr %62, align 8, !tbaa !33
  %63 = icmp sgt i64 %17, 1
  br i1 %63, label %16, label %64, !llvm.loop !199

64:                                               ; preds = %60, %110
  %65 = phi i64 [ %66, %110 ], [ %10, %60 ]
  %66 = add nsw i64 %65, -1
  %67 = getelementptr inbounds double, ptr %4, i64 %66
  %68 = load double, ptr %67, align 8, !tbaa !33
  %69 = load double, ptr %4, align 8, !tbaa !33
  store double %69, ptr %67, align 8, !tbaa !33
  %70 = icmp samesign ugt i64 %66, 2
  br i1 %70, label %71, label %87

71:                                               ; preds = %64, %71
  %72 = phi i64 [ %81, %71 ], [ 0, %64 ]
  %73 = phi i64 [ %85, %71 ], [ 2, %64 ]
  %74 = getelementptr double, ptr %4, i64 %73
  %75 = getelementptr i8, ptr %74, i64 -8
  %76 = load double, ptr %75, align 8, !tbaa !33
  %77 = load double, ptr %74, align 8, !tbaa !33
  %78 = fcmp olt double %76, %77
  %79 = zext i1 %78 to i64
  %80 = or disjoint i64 %73, %79
  %81 = add nsw i64 %80, -1
  %82 = getelementptr inbounds double, ptr %4, i64 %81
  %83 = load double, ptr %82, align 8, !tbaa !33
  %84 = getelementptr inbounds double, ptr %4, i64 %72
  store double %83, ptr %84, align 8, !tbaa !33
  %85 = shl nsw i64 %80, 1
  %86 = icmp slt i64 %85, %66
  br i1 %86, label %71, label %87, !llvm.loop !197

87:                                               ; preds = %71, %64
  %88 = phi i64 [ 2, %64 ], [ %85, %71 ]
  %89 = phi i64 [ 0, %64 ], [ %81, %71 ]
  %90 = icmp eq i64 %88, %66
  br i1 %90, label %91, label %96

91:                                               ; preds = %87
  %92 = add nsw i64 %65, -2
  %93 = getelementptr inbounds double, ptr %4, i64 %92
  %94 = load double, ptr %93, align 8, !tbaa !33
  %95 = getelementptr inbounds double, ptr %4, i64 %89
  store double %94, ptr %95, align 8, !tbaa !33
  br label %98

96:                                               ; preds = %87
  %97 = icmp sgt i64 %89, 0
  br i1 %97, label %98, label %110

98:                                               ; preds = %91, %96
  %99 = phi i64 [ %89, %96 ], [ %92, %91 ]
  br label %100

100:                                              ; preds = %98, %107
  %101 = phi i64 [ %103, %107 ], [ %99, %98 ]
  %102 = add nsw i64 %101, -1
  %103 = lshr i64 %102, 1
  %104 = getelementptr inbounds nuw double, ptr %4, i64 %103
  %105 = load double, ptr %104, align 8, !tbaa !33
  %106 = fcmp olt double %105, %68
  br i1 %106, label %107, label %110

107:                                              ; preds = %100
  %108 = getelementptr inbounds nuw double, ptr %4, i64 %101
  store double %105, ptr %108, align 8, !tbaa !33
  %109 = icmp ult i64 %102, 2
  br i1 %109, label %110, label %100, !llvm.loop !198

110:                                              ; preds = %100, %107, %96
  %111 = phi i64 [ %89, %96 ], [ %101, %100 ], [ 0, %107 ]
  %112 = getelementptr inbounds double, ptr %4, i64 %111
  store double %68, ptr %112, align 8, !tbaa !33
  %113 = icmp sgt i64 %65, 2
  br i1 %113, label %64, label %114, !llvm.loop !200

114:                                              ; preds = %110, %2
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZN9benchmark8heapsortISt16reverse_iteratorIS1_IN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEEdEEvT_SB_(ptr dead_on_return noundef %0, ptr dead_on_return noundef %1) local_unnamed_addr #10 comdat {
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %4 = load i64, ptr %3, align 8
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %6 = load i64, ptr %5, align 8, !tbaa !34, !noalias !201
  %7 = sub i64 %6, %4
  %8 = ashr exact i64 %7, 3
  %9 = icmp sgt i64 %8, 1
  %10 = inttoptr i64 %4 to ptr
  br i1 %9, label %11, label %115

11:                                               ; preds = %2
  %12 = lshr i64 %8, 1
  %13 = add nsw i64 %8, -1
  %14 = getelementptr inbounds nuw double, ptr %10, i64 %13
  br label %17

15:                                               ; preds = %61
  %16 = load ptr, ptr %3, align 8, !tbaa !204, !noalias !206
  br label %65

17:                                               ; preds = %11, %61
  %18 = phi i64 [ %12, %11 ], [ %19, %61 ]
  %19 = add nsw i64 %18, -1
  %20 = getelementptr inbounds double, ptr %10, i64 %19
  %21 = load double, ptr %20, align 8, !tbaa !33
  %22 = shl nuw i64 %19, 1
  %23 = add nuw nsw i64 %22, 2
  %24 = icmp slt i64 %23, %8
  br i1 %24, label %25, label %41

25:                                               ; preds = %17, %25
  %26 = phi i64 [ %35, %25 ], [ %19, %17 ]
  %27 = phi i64 [ %39, %25 ], [ %23, %17 ]
  %28 = getelementptr double, ptr %10, i64 %27
  %29 = getelementptr i8, ptr %28, i64 -8
  %30 = load double, ptr %29, align 8, !tbaa !33
  %31 = load double, ptr %28, align 8, !tbaa !33
  %32 = fcmp olt double %30, %31
  %33 = zext i1 %32 to i64
  %34 = add nsw i64 %27, %33
  %35 = add nsw i64 %34, -1
  %36 = getelementptr inbounds double, ptr %10, i64 %35
  %37 = load double, ptr %36, align 8, !tbaa !33
  %38 = getelementptr inbounds double, ptr %10, i64 %26
  store double %37, ptr %38, align 8, !tbaa !33
  %39 = shl nsw i64 %34, 1
  %40 = icmp slt i64 %39, %8
  br i1 %40, label %25, label %41, !llvm.loop !211

41:                                               ; preds = %25, %17
  %42 = phi i64 [ %23, %17 ], [ %39, %25 ]
  %43 = phi i64 [ %19, %17 ], [ %35, %25 ]
  %44 = icmp eq i64 %42, %8
  br i1 %44, label %45, label %48

45:                                               ; preds = %41
  %46 = load double, ptr %14, align 8, !tbaa !33
  %47 = getelementptr inbounds double, ptr %10, i64 %43
  store double %46, ptr %47, align 8, !tbaa !33
  br label %48

48:                                               ; preds = %45, %41
  %49 = phi i64 [ %13, %45 ], [ %43, %41 ]
  %50 = icmp slt i64 %49, %18
  br i1 %50, label %61, label %51

51:                                               ; preds = %48, %58
  %52 = phi i64 [ %54, %58 ], [ %49, %48 ]
  %53 = add nsw i64 %52, -1
  %54 = sdiv i64 %53, 2
  %55 = getelementptr inbounds double, ptr %10, i64 %54
  %56 = load double, ptr %55, align 8, !tbaa !33
  %57 = fcmp olt double %56, %21
  br i1 %57, label %58, label %61

58:                                               ; preds = %51
  %59 = getelementptr inbounds double, ptr %10, i64 %52
  store double %56, ptr %59, align 8, !tbaa !33
  %60 = icmp slt i64 %54, %18
  br i1 %60, label %61, label %51, !llvm.loop !212

61:                                               ; preds = %51, %58, %48
  %62 = phi i64 [ %49, %48 ], [ %52, %51 ], [ %54, %58 ]
  %63 = getelementptr inbounds double, ptr %10, i64 %62
  store double %21, ptr %63, align 8, !tbaa !33
  %64 = icmp sgt i64 %18, 1
  br i1 %64, label %17, label %15, !llvm.loop !213

65:                                               ; preds = %15, %111
  %66 = phi i64 [ %8, %15 ], [ %67, %111 ]
  %67 = add nsw i64 %66, -1
  %68 = getelementptr inbounds double, ptr %16, i64 %67
  %69 = load double, ptr %68, align 8, !tbaa !33
  %70 = load double, ptr %16, align 8, !tbaa !33
  store double %70, ptr %68, align 8, !tbaa !33
  %71 = icmp samesign ugt i64 %67, 2
  br i1 %71, label %72, label %88

72:                                               ; preds = %65, %72
  %73 = phi i64 [ %82, %72 ], [ 0, %65 ]
  %74 = phi i64 [ %86, %72 ], [ 2, %65 ]
  %75 = getelementptr double, ptr %16, i64 %74
  %76 = getelementptr i8, ptr %75, i64 -8
  %77 = load double, ptr %76, align 8, !tbaa !33
  %78 = load double, ptr %75, align 8, !tbaa !33
  %79 = fcmp olt double %77, %78
  %80 = zext i1 %79 to i64
  %81 = or disjoint i64 %74, %80
  %82 = add nsw i64 %81, -1
  %83 = getelementptr inbounds double, ptr %16, i64 %82
  %84 = load double, ptr %83, align 8, !tbaa !33
  %85 = getelementptr inbounds double, ptr %16, i64 %73
  store double %84, ptr %85, align 8, !tbaa !33
  %86 = shl nsw i64 %81, 1
  %87 = icmp slt i64 %86, %67
  br i1 %87, label %72, label %88, !llvm.loop !211

88:                                               ; preds = %72, %65
  %89 = phi i64 [ 2, %65 ], [ %86, %72 ]
  %90 = phi i64 [ 0, %65 ], [ %82, %72 ]
  %91 = icmp eq i64 %89, %67
  br i1 %91, label %92, label %97

92:                                               ; preds = %88
  %93 = add nsw i64 %66, -2
  %94 = getelementptr inbounds double, ptr %16, i64 %93
  %95 = load double, ptr %94, align 8, !tbaa !33
  %96 = getelementptr inbounds double, ptr %16, i64 %90
  store double %95, ptr %96, align 8, !tbaa !33
  br label %99

97:                                               ; preds = %88
  %98 = icmp sgt i64 %90, 0
  br i1 %98, label %99, label %111

99:                                               ; preds = %92, %97
  %100 = phi i64 [ %90, %97 ], [ %93, %92 ]
  br label %101

101:                                              ; preds = %99, %108
  %102 = phi i64 [ %104, %108 ], [ %100, %99 ]
  %103 = add nsw i64 %102, -1
  %104 = lshr i64 %103, 1
  %105 = getelementptr inbounds nuw double, ptr %16, i64 %104
  %106 = load double, ptr %105, align 8, !tbaa !33
  %107 = fcmp olt double %106, %69
  br i1 %107, label %108, label %111

108:                                              ; preds = %101
  %109 = getelementptr inbounds nuw double, ptr %16, i64 %102
  store double %106, ptr %109, align 8, !tbaa !33
  %110 = icmp ult i64 %103, 2
  br i1 %110, label %111, label %101, !llvm.loop !212

111:                                              ; preds = %101, %108, %97
  %112 = phi i64 [ %90, %97 ], [ %102, %101 ], [ 0, %108 ]
  %113 = getelementptr inbounds double, ptr %16, i64 %112
  store double %69, ptr %113, align 8, !tbaa !33
  %114 = icmp sgt i64 %66, 2
  br i1 %114, label %65, label %115, !llvm.loop !214

115:                                              ; preds = %111, %2
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smax.i32(i32, i32) #15

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #16

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.vector.reduce.fadd.v2f64(double, <2 x double>) #15

attributes #0 = { mustprogress nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nounwind willreturn allockind("realloc") allocsize(1) memory(argmem: readwrite, inaccessiblemem: readwrite) "alloc-family"="malloc" "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { mustprogress nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #6 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: read) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { mustprogress nocallback nofree nounwind willreturn memory(errnomem: write) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #8 = { nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #9 = { mustprogress norecurse uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #10 = { mustprogress uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #11 = { mustprogress nocallback nofree nounwind willreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #12 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #13 = { nobuiltin nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #14 = { nobuiltin allocsize(0) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #15 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #16 = { nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #17 = { nounwind allocsize(1) }
attributes #18 = { cold noreturn nounwind }
attributes #19 = { nounwind willreturn memory(read) }
attributes #20 = { nounwind }
attributes #21 = { builtin allocsize(0) }
attributes #22 = { builtin nounwind }

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
!38 = distinct !{!38, !19, !21, !22}
!39 = distinct !{!39, !19}
!40 = distinct !{!40, !19}
!41 = distinct !{!41, !19, !21, !22}
!42 = distinct !{!42, !19}
!43 = !{!44, !35, i64 0}
!44 = !{!"_ZTSSt16reverse_iteratorIPdE", !35, i64 0}
!45 = distinct !{!45, !19}
!46 = distinct !{!46, !19}
!47 = distinct !{!47, !19}
!48 = distinct !{!48, !19}
!49 = distinct !{!49, !19}
!50 = distinct !{!50, !19}
!51 = distinct !{!51, !19, !21, !22}
!52 = distinct !{!52, !19}
!53 = distinct !{!53, !19, !21, !22}
!54 = distinct !{!54, !19}
!55 = distinct !{!55, !19, !21, !22}
!56 = distinct !{!56, !19, !21}
!57 = distinct !{!57, !19, !21, !22}
!58 = distinct !{!58, !19, !21}
!59 = distinct !{!59, !19}
!60 = distinct !{!60, !19}
!61 = distinct !{!61, !19}
!62 = distinct !{!62, !19}
!63 = distinct !{!63, !19}
!64 = distinct !{!64, !19}
!65 = distinct !{!65, !19}
!66 = distinct !{!66, !19}
!67 = distinct !{!67, !19, !21, !22}
!68 = distinct !{!68, !19, !21}
!69 = distinct !{!69, !19}
!70 = distinct !{!70, !19}
!71 = distinct !{!71, !19}
!72 = distinct !{!72, !19}
!73 = distinct !{!73, !19}
!74 = distinct !{!74, !19}
!75 = distinct !{!75, !19}
!76 = distinct !{!76, !19}
!77 = distinct !{!77, !19, !21, !22}
!78 = distinct !{!78, !19, !21}
!79 = distinct !{!79, !19}
!80 = distinct !{!80, !19}
!81 = distinct !{!81, !19, !21, !22}
!82 = distinct !{!82, !19, !21}
!83 = distinct !{!83, !19}
!84 = distinct !{!84, !19}
!85 = distinct !{!85, !19, !21, !22}
!86 = distinct !{!86, !19, !21}
!87 = distinct !{!87, !19}
!88 = distinct !{!88, !19}
!89 = distinct !{!89, !19, !21, !22}
!90 = distinct !{!90, !19, !21}
!91 = distinct !{!91, !19}
!92 = distinct !{!92, !19}
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
!110 = distinct !{!110, !19, !21, !22}
!111 = distinct !{!111, !19, !21}
!112 = distinct !{!112, !19}
!113 = distinct !{!113, !19, !21, !22}
!114 = distinct !{!114, !19, !21}
!115 = distinct !{!115, !19}
!116 = distinct !{!116, !19, !21, !22}
!117 = distinct !{!117, !19, !21}
!118 = distinct !{!118, !19}
!119 = distinct !{!119, !19, !21, !22}
!120 = distinct !{!120, !19, !21}
!121 = distinct !{!121, !19}
!122 = distinct !{!122, !19}
!123 = distinct !{!123, !19}
!124 = distinct !{!124, !19}
!125 = distinct !{!125, !19}
!126 = distinct !{!126, !19}
!127 = distinct !{!127, !19}
!128 = distinct !{!128, !19}
!129 = distinct !{!129, !19}
!130 = !{!131}
!131 = distinct !{!131, !132, !"_ZNKSt16reverse_iteratorIPdEplEl: argument 0"}
!132 = distinct !{!132, !"_ZNKSt16reverse_iteratorIPdEplEl"}
!133 = !{!134}
!134 = distinct !{!134, !135, !"_ZNKSt16reverse_iteratorIPdEplEl: argument 0"}
!135 = distinct !{!135, !"_ZNKSt16reverse_iteratorIPdEplEl"}
!136 = distinct !{!136, !19}
!137 = distinct !{!137, !19}
!138 = distinct !{!138, !19}
!139 = !{!140}
!140 = distinct !{!140, !141, !"_ZNKSt16reverse_iteratorIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEplEl: argument 0"}
!141 = distinct !{!141, !"_ZNKSt16reverse_iteratorIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEplEl"}
!142 = !{!143}
!143 = distinct !{!143, !144, !"_ZNKSt16reverse_iteratorIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEplEl: argument 0"}
!144 = distinct !{!144, !"_ZNKSt16reverse_iteratorIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEplEl"}
!145 = !{!146}
!146 = distinct !{!146, !147, !"_ZNKSt16reverse_iteratorIS_IPdEE4baseEv: argument 0"}
!147 = distinct !{!147, !"_ZNKSt16reverse_iteratorIS_IPdEE4baseEv"}
!148 = !{!149}
!149 = distinct !{!149, !150, !"_ZNKSt16reverse_iteratorIS_IPdEE4baseEv: argument 0"}
!150 = distinct !{!150, !"_ZNKSt16reverse_iteratorIS_IPdEE4baseEv"}
!151 = distinct !{!151, !19}
!152 = distinct !{!152, !19}
!153 = distinct !{!153, !19}
!154 = !{!155}
!155 = distinct !{!155, !156, !"_ZNKSt16reverse_iteratorIS_IPdEEplEl: argument 0"}
!156 = distinct !{!156, !"_ZNKSt16reverse_iteratorIS_IPdEEplEl"}
!157 = !{!158}
!158 = distinct !{!158, !159, !"_ZNKSt16reverse_iteratorIS_IPdEEplEl: argument 0"}
!159 = distinct !{!159, !"_ZNKSt16reverse_iteratorIS_IPdEEplEl"}
!160 = !{!161}
!161 = distinct !{!161, !162, !"_ZNKSt16reverse_iteratorIS_IN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEE4baseEv: argument 0"}
!162 = distinct !{!162, !"_ZNKSt16reverse_iteratorIS_IN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEE4baseEv"}
!163 = !{!164}
!164 = distinct !{!164, !165, !"_ZNKSt16reverse_iteratorIS_IN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEE4baseEv: argument 0"}
!165 = distinct !{!165, !"_ZNKSt16reverse_iteratorIS_IN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEE4baseEv"}
!166 = distinct !{!166, !19}
!167 = distinct !{!167, !19}
!168 = distinct !{!168, !19}
!169 = !{!170}
!170 = distinct !{!170, !171, !"_ZNKSt16reverse_iteratorIS_IN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEEplEl: argument 0"}
!171 = distinct !{!171, !"_ZNKSt16reverse_iteratorIS_IN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEEplEl"}
!172 = !{!173}
!173 = distinct !{!173, !174, !"_ZNKSt16reverse_iteratorIS_IN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEEplEl: argument 0"}
!174 = distinct !{!174, !"_ZNKSt16reverse_iteratorIS_IN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEEplEl"}
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
!191 = !{!192}
!192 = distinct !{!192, !193, !"_ZNKSt16reverse_iteratorIS_IPdEE4baseEv: argument 0"}
!193 = distinct !{!193, !"_ZNKSt16reverse_iteratorIS_IPdEE4baseEv"}
!194 = !{!195}
!195 = distinct !{!195, !196, !"_ZNKSt16reverse_iteratorIS_IPdEE4baseEv: argument 0"}
!196 = distinct !{!196, !"_ZNKSt16reverse_iteratorIS_IPdEE4baseEv"}
!197 = distinct !{!197, !19}
!198 = distinct !{!198, !19}
!199 = distinct !{!199, !19}
!200 = distinct !{!200, !19}
!201 = !{!202}
!202 = distinct !{!202, !203, !"_ZNKSt16reverse_iteratorIS_IN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEE4baseEv: argument 0"}
!203 = distinct !{!203, !"_ZNKSt16reverse_iteratorIS_IN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEE4baseEv"}
!204 = !{!205, !35, i64 0}
!205 = !{!"_ZTSN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEE", !35, i64 0}
!206 = !{!207, !209}
!207 = distinct !{!207, !208, !"_ZNKSt16reverse_iteratorIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEmiEl: argument 0"}
!208 = distinct !{!208, !"_ZNKSt16reverse_iteratorIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEmiEl"}
!209 = distinct !{!209, !210, !"_ZNKSt16reverse_iteratorIS_IN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEEplEl: argument 0"}
!210 = distinct !{!210, !"_ZNKSt16reverse_iteratorIS_IN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEEplEl"}
!211 = distinct !{!211, !19}
!212 = distinct !{!212, !19}
!213 = distinct !{!213, !19}
!214 = distinct !{!214, !19}
