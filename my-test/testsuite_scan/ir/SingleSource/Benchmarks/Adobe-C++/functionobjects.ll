; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Adobe-C++/functionobjects.cpp'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Adobe-C++/functionobjects.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.one_result = type { double, ptr }
%"struct.__gnu_cxx::__ops::_Iter_comp_iter" = type { ptr }
%"struct.__gnu_cxx::__ops::_Iter_comp_iter.0" = type { i8 }
%"struct.__gnu_cxx::__ops::_Iter_comp_iter.3" = type { i8 }
%"struct.__gnu_cxx::__ops::_Iter_comp_iter.6" = type { i8 }
%"struct.__gnu_cxx::__ops::_Iter_less_iter" = type { i8 }

$_Z9quicksortIPdPFbddEEvT_S3_T0_ = comdat any

$_Z9quicksortIPd17less_than_functorEvT_S2_T0_ = comdat any

$_Z9quicksortIPd24inline_less_than_functorEvT_S2_T0_ = comdat any

$_Z9quicksortIPdSt4lessIdEEvT_S3_T0_ = comdat any

$_Z9quicksortIPdEvT_S1_ = comdat any

$_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterIPFbddEEEEvT_S7_T0_T1_ = comdat any

$_ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_comp_iterIPFbddEEEEvT_S7_RT0_ = comdat any

$_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterI17less_than_functorEEEvT_S6_T0_T1_ = comdat any

$_ZSt22__final_insertion_sortIPdN9__gnu_cxx5__ops15_Iter_comp_iterI17less_than_functorEEEvT_S6_T0_ = comdat any

$_ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_comp_iterI17less_than_functorEEEvT_S6_RT0_ = comdat any

$_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterI24inline_less_than_functorEEEvT_S6_T0_T1_ = comdat any

$_ZSt22__final_insertion_sortIPdN9__gnu_cxx5__ops15_Iter_comp_iterI24inline_less_than_functorEEEvT_S6_T0_ = comdat any

$_ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_comp_iterI24inline_less_than_functorEEEvT_S6_RT0_ = comdat any

$_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterISt4lessIdEEEEvT_S7_T0_T1_ = comdat any

$_ZSt22__final_insertion_sortIPdN9__gnu_cxx5__ops15_Iter_comp_iterISt4lessIdEEEEvT_S7_T0_ = comdat any

$_ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_comp_iterISt4lessIdEEEEvT_S7_RT0_ = comdat any

$_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_ = comdat any

$_ZSt22__final_insertion_sortIPdN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_ = comdat any

$_ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_RT0_ = comdat any

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
@.str.11 = private unnamed_addr constant [16 x i8] c"test %i failed\0A\00", align 1

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
  %13 = tail call ptr @realloc(ptr noundef %3, i64 noundef %12) #20
  store ptr %13, ptr @results, align 8, !tbaa !6
  %14 = icmp eq ptr %13, null
  br i1 %14, label %17, label %15

15:                                               ; preds = %9
  %16 = load i32, ptr @current_test, align 4, !tbaa !11
  br label %20

17:                                               ; preds = %9
  %18 = load i32, ptr @allocated_results, align 4, !tbaa !11
  %19 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %18)
  tail call void @exit(i32 noundef -1) #21
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
  %20 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %19) #22
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
  %63 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %62) #22
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
  %100 = tail call double @log(double noundef %99) #23, !tbaa !11
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
  %110 = tail call double @exp(double noundef %109) #23, !tbaa !11
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
  %13 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %12) #22
  %14 = trunc i64 %13 to i32
  %15 = tail call i32 @llvm.smax.i32(i32 %10, i32 %14)
  %16 = add nuw nsw i64 %9, 1
  %17 = icmp eq i64 %16, %7
  br i1 %17, label %18, label %8, !llvm.loop !26

18:                                               ; preds = %8, %2
  %19 = phi i32 [ 12, %2 ], [ %15, %8 ]
  %20 = add nsw i32 %19, -12
  %21 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %0, ptr noundef nonnull @.str.8, i32 noundef %20, ptr noundef nonnull @.str.2) #23
  %22 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %0, ptr noundef nonnull @.str.9, i32 noundef %19, ptr noundef nonnull @.str.2) #23
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
  %56 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %55) #22
  %57 = trunc i64 %56 to i32
  %58 = sub i32 %19, %57
  %59 = load double, ptr %53, align 8, !tbaa !13
  %60 = trunc nuw nsw i64 %51 to i32
  %61 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %0, ptr noundef nonnull @.str.10, i32 noundef %60, i32 noundef %58, ptr noundef nonnull @.str.5, ptr noundef nonnull %55, double noundef %59) #23
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
  %76 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %0, ptr noundef nonnull @.str.6, ptr noundef %1, double noundef %75) #23
  store i32 0, ptr @current_test, align 4, !tbaa !11
  ret void
}

; Function Attrs: nofree nounwind
declare noundef i32 @fprintf(ptr noundef captures(none), ptr noundef readonly captures(none), ...) local_unnamed_addr #2

; Function Attrs: mustprogress nounwind uwtable
define dso_local void @_Z11start_timerv() local_unnamed_addr #0 {
  %1 = tail call i64 @clock() #23
  store i64 %1, ptr @start_time, align 8, !tbaa !30
  ret void
}

; Function Attrs: nounwind
declare i64 @clock() local_unnamed_addr #8

; Function Attrs: mustprogress nounwind uwtable
define dso_local noundef double @_Z5timerv() local_unnamed_addr #0 {
  %1 = tail call i64 @clock() #23
  store i64 %1, ptr @end_time, align 8, !tbaa !30
  %2 = load i64, ptr @start_time, align 8, !tbaa !30
  %3 = sub nsw i64 %1, %2
  %4 = sitofp i64 %3 to double
  %5 = fdiv double %4, 1.000000e+06
  ret double %5
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read) uwtable
define dso_local noundef range(i32 -1, 2) i32 @_Z19less_than_function1PKvS0_(ptr noundef readonly captures(none) %0, ptr noundef readonly captures(none) %1) #9 {
  %3 = load double, ptr %0, align 8, !tbaa !32
  %4 = load double, ptr %1, align 8, !tbaa !32
  %5 = fcmp olt double %3, %4
  %6 = fcmp ogt double %3, %4
  %7 = zext i1 %6 to i32
  %8 = select i1 %5, i32 -1, i32 %7
  ret i32 %8
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i1 @_Z19less_than_function2dd(double noundef %0, double noundef %1) #10 {
  %3 = fcmp olt double %0, %1
  ret i1 %3
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read) uwtable
define dso_local noundef i1 @_ZNK17less_than_functorclERKdS1_(ptr noundef nonnull readnone align 1 captures(none) dereferenceable(1) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2) local_unnamed_addr #9 {
  %4 = load double, ptr %1, align 8, !tbaa !32
  %5 = load double, ptr %2, align 8, !tbaa !32
  %6 = fcmp olt double %4, %5
  ret i1 %6
}

; Function Attrs: mustprogress uwtable
define dso_local void @_Z18quicksort_functionPdS_PFbddE(ptr noundef %0, ptr noundef %1, ptr noundef %2) local_unnamed_addr #11 {
  %4 = ptrtoint ptr %1 to i64
  %5 = ptrtoint ptr %0 to i64
  %6 = sub i64 %4, %5
  %7 = icmp sgt i64 %6, 8
  br i1 %7, label %8, label %31

8:                                                ; preds = %3
  %9 = load double, ptr %0, align 8, !tbaa !32
  br label %10

10:                                               ; preds = %27, %8
  %11 = phi ptr [ %1, %8 ], [ %15, %27 ]
  %12 = phi ptr [ %0, %8 ], [ %21, %27 ]
  br label %13

13:                                               ; preds = %13, %10
  %14 = phi ptr [ %11, %10 ], [ %15, %13 ]
  %15 = getelementptr inbounds i8, ptr %14, i64 -8
  %16 = load double, ptr %15, align 8, !tbaa !32
  %17 = tail call noundef i1 %2(double noundef %9, double noundef %16)
  br i1 %17, label %13, label %18, !llvm.loop !33

18:                                               ; preds = %13
  %19 = icmp ult ptr %12, %15
  br i1 %19, label %20, label %30

20:                                               ; preds = %18, %20
  %21 = phi ptr [ %24, %20 ], [ %12, %18 ]
  %22 = load double, ptr %21, align 8, !tbaa !32
  %23 = tail call noundef i1 %2(double noundef %22, double noundef %9)
  %24 = getelementptr inbounds nuw i8, ptr %21, i64 8
  br i1 %23, label %20, label %25, !llvm.loop !34

25:                                               ; preds = %20
  %26 = icmp ult ptr %21, %15
  br i1 %26, label %27, label %30

27:                                               ; preds = %25
  %28 = load double, ptr %15, align 8, !tbaa !32
  %29 = load double, ptr %21, align 8, !tbaa !32
  store double %29, ptr %15, align 8, !tbaa !32
  store double %28, ptr %21, align 8, !tbaa !32
  br label %10, !llvm.loop !35

30:                                               ; preds = %25, %18
  tail call void @_Z9quicksortIPdPFbddEEvT_S3_T0_(ptr noundef nonnull %0, ptr noundef nonnull %14, ptr noundef %2)
  tail call void @_Z9quicksortIPdPFbddEEvT_S3_T0_(ptr noundef nonnull %14, ptr noundef %1, ptr noundef %2)
  br label %31

31:                                               ; preds = %30, %3
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_Z9quicksortIPdPFbddEEvT_S3_T0_(ptr noundef %0, ptr noundef %1, ptr noundef %2) local_unnamed_addr #11 comdat {
  %4 = ptrtoint ptr %1 to i64
  %5 = ptrtoint ptr %0 to i64
  %6 = sub i64 %4, %5
  %7 = icmp sgt i64 %6, 8
  br i1 %7, label %8, label %35

8:                                                ; preds = %3, %31
  %9 = phi ptr [ %15, %31 ], [ %0, %3 ]
  %10 = load double, ptr %9, align 8, !tbaa !32
  br label %11

11:                                               ; preds = %28, %8
  %12 = phi ptr [ %1, %8 ], [ %16, %28 ]
  %13 = phi ptr [ %9, %8 ], [ %22, %28 ]
  br label %14

14:                                               ; preds = %14, %11
  %15 = phi ptr [ %12, %11 ], [ %16, %14 ]
  %16 = getelementptr inbounds i8, ptr %15, i64 -8
  %17 = load double, ptr %16, align 8, !tbaa !32
  %18 = tail call noundef i1 %2(double noundef %10, double noundef %17)
  br i1 %18, label %14, label %19, !llvm.loop !36

19:                                               ; preds = %14
  %20 = icmp ult ptr %13, %16
  br i1 %20, label %21, label %31

21:                                               ; preds = %19, %21
  %22 = phi ptr [ %25, %21 ], [ %13, %19 ]
  %23 = load double, ptr %22, align 8, !tbaa !32
  %24 = tail call noundef i1 %2(double noundef %23, double noundef %10)
  %25 = getelementptr inbounds nuw i8, ptr %22, i64 8
  br i1 %24, label %21, label %26, !llvm.loop !37

26:                                               ; preds = %21
  %27 = icmp ult ptr %22, %16
  br i1 %27, label %28, label %31

28:                                               ; preds = %26
  %29 = load double, ptr %16, align 8, !tbaa !32
  %30 = load double, ptr %22, align 8, !tbaa !32
  store double %30, ptr %16, align 8, !tbaa !32
  store double %29, ptr %22, align 8, !tbaa !32
  br label %11, !llvm.loop !38

31:                                               ; preds = %26, %19
  tail call void @_Z9quicksortIPdPFbddEEvT_S3_T0_(ptr noundef nonnull %9, ptr noundef nonnull %15, ptr noundef %2)
  %32 = ptrtoint ptr %15 to i64
  %33 = sub i64 %4, %32
  %34 = icmp sgt i64 %33, 8
  br i1 %34, label %8, label %35

35:                                               ; preds = %31, %3
  ret void
}

; Function Attrs: mustprogress norecurse uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #12 {
  %3 = icmp sgt i32 %0, 1
  br i1 %3, label %4, label %16

4:                                                ; preds = %2
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %6 = load ptr, ptr %5, align 8, !tbaa !39
  %7 = tail call i64 @__isoc23_strtol(ptr noundef nonnull %6, ptr noundef null, i32 noundef 10) #23
  %8 = trunc i64 %7 to i32
  %9 = icmp eq i32 %0, 2
  br i1 %9, label %16, label %10

10:                                               ; preds = %4
  %11 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %12 = load ptr, ptr %11, align 8, !tbaa !39
  %13 = tail call i64 @__isoc23_strtol(ptr noundef nonnull %12, ptr noundef null, i32 noundef 10) #23
  %14 = freeze i64 %13
  %15 = trunc i64 %14 to i32
  br label %16

16:                                               ; preds = %2, %4, %10
  %17 = phi i32 [ %8, %10 ], [ %8, %4 ], [ 300, %2 ]
  %18 = phi i32 [ %15, %10 ], [ 10000, %4 ], [ 10000, %2 ]
  %19 = add nsw i32 %18, 123
  tail call void @srand(i32 noundef %19) #23
  %20 = sext i32 %18 to i64
  %21 = icmp slt i32 %18, 0
  %22 = shl nsw i64 %20, 3
  br i1 %21, label %23, label %25

23:                                               ; preds = %16
  %24 = tail call noalias noundef nonnull dereferenceable(18446744073709551615) ptr @_Znam(i64 noundef -1) #24
  br label %30

25:                                               ; preds = %16
  %26 = tail call noalias noundef nonnull ptr @_Znam(i64 noundef %22) #24
  %27 = icmp eq i32 %18, 0
  br i1 %27, label %30, label %28

28:                                               ; preds = %25
  %29 = zext nneg i32 %18 to i64
  br label %39

30:                                               ; preds = %39, %23, %25
  %31 = phi ptr [ %24, %23 ], [ %26, %25 ], [ %26, %39 ]
  %32 = phi i64 [ -1, %23 ], [ %22, %25 ], [ %22, %39 ]
  %33 = tail call noalias noundef nonnull ptr @_Znam(i64 noundef %32) #24
  %34 = icmp sgt i32 %17, 0
  br i1 %34, label %35, label %506

35:                                               ; preds = %30
  %36 = icmp sgt i32 %18, 1
  %37 = icmp eq i32 %18, 1
  %38 = getelementptr inbounds double, ptr %33, i64 %20
  br label %50

39:                                               ; preds = %28, %39
  %40 = phi i64 [ 0, %28 ], [ %44, %39 ]
  %41 = tail call i32 @rand() #23
  %42 = sitofp i32 %41 to double
  %43 = getelementptr inbounds nuw double, ptr %26, i64 %40
  store double %42, ptr %43, align 8, !tbaa !32
  %44 = add nuw nsw i64 %40, 1
  %45 = icmp eq i64 %44, %29
  br i1 %45, label %30, label %39, !llvm.loop !40

46:                                               ; preds = %68
  %47 = icmp sgt i32 %18, 1
  %48 = icmp eq i32 %18, 1
  %49 = getelementptr inbounds double, ptr %33, i64 %20
  br label %75

50:                                               ; preds = %35, %68
  %51 = phi i32 [ 0, %35 ], [ %69, %68 ]
  br i1 %36, label %52, label %53, !prof !41

52:                                               ; preds = %50
  tail call void @llvm.memcpy.p0.p0.i64(ptr nonnull align 8 %33, ptr nonnull align 8 %31, i64 %22, i1 false)
  br label %56

53:                                               ; preds = %50
  br i1 %37, label %54, label %56

54:                                               ; preds = %53
  %55 = load double, ptr %31, align 8, !tbaa !32
  store double %55, ptr %33, align 8, !tbaa !32
  br label %56

56:                                               ; preds = %52, %53, %54
  tail call void @qsort(ptr noundef nonnull %33, i64 noundef %20, i64 noundef 8, ptr noundef nonnull @_Z19less_than_function1PKvS0_)
  br label %57

57:                                               ; preds = %61, %56
  %58 = phi ptr [ %33, %56 ], [ %59, %61 ]
  %59 = getelementptr i8, ptr %58, i64 8
  %60 = icmp eq ptr %59, %38
  br i1 %60, label %68, label %61

61:                                               ; preds = %57
  %62 = load double, ptr %59, align 8, !tbaa !32
  %63 = load double, ptr %58, align 8, !tbaa !32
  %64 = fcmp olt double %62, %63
  br i1 %64, label %65, label %57, !llvm.loop !42

65:                                               ; preds = %61
  %66 = load i32, ptr @current_test, align 4, !tbaa !11
  %67 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.11, i32 noundef %66)
  br label %68

68:                                               ; preds = %57, %65
  %69 = add nuw nsw i32 %51, 1
  %70 = icmp eq i32 %69, %17
  br i1 %70, label %46, label %50, !llvm.loop !43

71:                                               ; preds = %113
  %72 = icmp sgt i32 %18, 1
  %73 = icmp eq i32 %18, 1
  %74 = getelementptr inbounds double, ptr %33, i64 %20
  br label %120

75:                                               ; preds = %46, %113
  %76 = phi i32 [ 0, %46 ], [ %114, %113 ]
  br i1 %47, label %80, label %77, !prof !41

77:                                               ; preds = %75
  br i1 %48, label %78, label %101

78:                                               ; preds = %77
  %79 = load double, ptr %31, align 8, !tbaa !32
  store double %79, ptr %33, align 8, !tbaa !32
  br label %101

80:                                               ; preds = %75
  tail call void @llvm.memcpy.p0.p0.i64(ptr nonnull align 8 %33, ptr nonnull align 8 %31, i64 %22, i1 false)
  %81 = load double, ptr %33, align 8, !tbaa !32
  br label %82

82:                                               ; preds = %99, %80
  %83 = phi ptr [ %49, %80 ], [ %87, %99 ]
  %84 = phi ptr [ %33, %80 ], [ %93, %99 ]
  br label %85

85:                                               ; preds = %85, %82
  %86 = phi ptr [ %83, %82 ], [ %87, %85 ]
  %87 = getelementptr inbounds i8, ptr %86, i64 -8
  %88 = load double, ptr %87, align 8, !tbaa !32
  %89 = fcmp olt double %81, %88
  br i1 %89, label %85, label %90, !llvm.loop !33

90:                                               ; preds = %85
  %91 = icmp ult ptr %84, %87
  br i1 %91, label %92, label %100

92:                                               ; preds = %90, %92
  %93 = phi ptr [ %96, %92 ], [ %84, %90 ]
  %94 = load double, ptr %93, align 8, !tbaa !32
  %95 = fcmp olt double %94, %81
  %96 = getelementptr inbounds nuw i8, ptr %93, i64 8
  br i1 %95, label %92, label %97, !llvm.loop !34

97:                                               ; preds = %92
  %98 = icmp ult ptr %93, %87
  br i1 %98, label %99, label %100

99:                                               ; preds = %97
  store double %94, ptr %87, align 8, !tbaa !32
  store double %88, ptr %93, align 8, !tbaa !32
  br label %82, !llvm.loop !35

100:                                              ; preds = %97, %90
  tail call void @_Z9quicksortIPdPFbddEEvT_S3_T0_(ptr noundef nonnull %33, ptr noundef nonnull %86, ptr noundef nonnull @_Z19less_than_function2dd)
  tail call void @_Z9quicksortIPdPFbddEEvT_S3_T0_(ptr noundef nonnull %86, ptr noundef nonnull %49, ptr noundef nonnull @_Z19less_than_function2dd)
  br label %101

101:                                              ; preds = %78, %77, %100
  br label %102

102:                                              ; preds = %101, %106
  %103 = phi ptr [ %104, %106 ], [ %33, %101 ]
  %104 = getelementptr i8, ptr %103, i64 8
  %105 = icmp eq ptr %104, %49
  br i1 %105, label %113, label %106

106:                                              ; preds = %102
  %107 = load double, ptr %104, align 8, !tbaa !32
  %108 = load double, ptr %103, align 8, !tbaa !32
  %109 = fcmp olt double %107, %108
  br i1 %109, label %110, label %102, !llvm.loop !42

110:                                              ; preds = %106
  %111 = load i32, ptr @current_test, align 4, !tbaa !11
  %112 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.11, i32 noundef %111)
  br label %113

113:                                              ; preds = %102, %110
  %114 = add nuw nsw i32 %76, 1
  %115 = icmp eq i32 %114, %17
  br i1 %115, label %71, label %75, !llvm.loop !44

116:                                              ; preds = %138
  %117 = icmp sgt i32 %18, 1
  %118 = icmp eq i32 %18, 1
  %119 = getelementptr inbounds double, ptr %33, i64 %20
  br label %151

120:                                              ; preds = %71, %138
  %121 = phi i32 [ 0, %71 ], [ %139, %138 ]
  br i1 %72, label %122, label %123, !prof !41

122:                                              ; preds = %120
  tail call void @llvm.memcpy.p0.p0.i64(ptr nonnull align 8 %33, ptr nonnull align 8 %31, i64 %22, i1 false)
  br label %126

123:                                              ; preds = %120
  br i1 %73, label %124, label %126

124:                                              ; preds = %123
  %125 = load double, ptr %31, align 8, !tbaa !32
  store double %125, ptr %33, align 8, !tbaa !32
  br label %126

126:                                              ; preds = %122, %123, %124
  tail call void @_Z9quicksortIPdPFbddEEvT_S3_T0_(ptr noundef nonnull %33, ptr noundef nonnull %74, ptr noundef nonnull @_Z19less_than_function2dd)
  br label %127

127:                                              ; preds = %131, %126
  %128 = phi ptr [ %33, %126 ], [ %129, %131 ]
  %129 = getelementptr i8, ptr %128, i64 8
  %130 = icmp eq ptr %129, %74
  br i1 %130, label %138, label %131

131:                                              ; preds = %127
  %132 = load double, ptr %129, align 8, !tbaa !32
  %133 = load double, ptr %128, align 8, !tbaa !32
  %134 = fcmp olt double %132, %133
  br i1 %134, label %135, label %127, !llvm.loop !42

135:                                              ; preds = %131
  %136 = load i32, ptr @current_test, align 4, !tbaa !11
  %137 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.11, i32 noundef %136)
  br label %138

138:                                              ; preds = %127, %135
  %139 = add nuw nsw i32 %121, 1
  %140 = icmp eq i32 %139, %17
  br i1 %140, label %116, label %120, !llvm.loop !45

141:                                              ; preds = %189
  %142 = icmp sgt i32 %18, 1
  %143 = getelementptr inbounds i8, ptr %33, i64 %22
  %144 = ptrtoint ptr %33 to i64
  %145 = tail call range(i64 0, 65) i64 @llvm.ctlz.i64(i64 %20, i1 true)
  %146 = shl nuw nsw i64 %145, 1
  %147 = xor i64 %146, 126
  %148 = icmp sgt i32 %18, 16
  %149 = getelementptr i8, ptr %33, i64 8
  %150 = getelementptr inbounds nuw i8, ptr %33, i64 128
  br label %196

151:                                              ; preds = %116, %189
  %152 = phi i32 [ 0, %116 ], [ %190, %189 ]
  br i1 %117, label %156, label %153, !prof !41

153:                                              ; preds = %151
  br i1 %118, label %154, label %177

154:                                              ; preds = %153
  %155 = load double, ptr %31, align 8, !tbaa !32
  store double %155, ptr %33, align 8, !tbaa !32
  br label %177

156:                                              ; preds = %151
  tail call void @llvm.memcpy.p0.p0.i64(ptr nonnull align 8 %33, ptr nonnull align 8 %31, i64 %22, i1 false)
  %157 = load double, ptr %33, align 8, !tbaa !32
  br label %158

158:                                              ; preds = %175, %156
  %159 = phi ptr [ %119, %156 ], [ %163, %175 ]
  %160 = phi ptr [ %33, %156 ], [ %169, %175 ]
  br label %161

161:                                              ; preds = %161, %158
  %162 = phi ptr [ %159, %158 ], [ %163, %161 ]
  %163 = getelementptr inbounds i8, ptr %162, i64 -8
  %164 = load double, ptr %163, align 8, !tbaa !32
  %165 = fcmp olt double %157, %164
  br i1 %165, label %161, label %166, !llvm.loop !46

166:                                              ; preds = %161
  %167 = icmp ult ptr %160, %163
  br i1 %167, label %168, label %176

168:                                              ; preds = %166, %168
  %169 = phi ptr [ %172, %168 ], [ %160, %166 ]
  %170 = load double, ptr %169, align 8, !tbaa !32
  %171 = fcmp olt double %170, %157
  %172 = getelementptr inbounds nuw i8, ptr %169, i64 8
  br i1 %171, label %168, label %173, !llvm.loop !47

173:                                              ; preds = %168
  %174 = icmp ult ptr %169, %163
  br i1 %174, label %175, label %176

175:                                              ; preds = %173
  store double %170, ptr %163, align 8, !tbaa !32
  store double %164, ptr %169, align 8, !tbaa !32
  br label %158, !llvm.loop !48

176:                                              ; preds = %173, %166
  tail call void @_Z9quicksortIPdPFbddEEvT_S3_T0_(ptr noundef nonnull %33, ptr noundef nonnull %162, ptr noundef nonnull @_Z19less_than_function2dd)
  tail call void @_Z9quicksortIPdPFbddEEvT_S3_T0_(ptr noundef nonnull %162, ptr noundef nonnull %119, ptr noundef nonnull @_Z19less_than_function2dd)
  br label %177

177:                                              ; preds = %154, %153, %176
  br label %178

178:                                              ; preds = %177, %182
  %179 = phi ptr [ %180, %182 ], [ %33, %177 ]
  %180 = getelementptr i8, ptr %179, i64 8
  %181 = icmp eq ptr %180, %119
  br i1 %181, label %189, label %182

182:                                              ; preds = %178
  %183 = load double, ptr %180, align 8, !tbaa !32
  %184 = load double, ptr %179, align 8, !tbaa !32
  %185 = fcmp olt double %183, %184
  br i1 %185, label %186, label %178, !llvm.loop !42

186:                                              ; preds = %182
  %187 = load i32, ptr @current_test, align 4, !tbaa !11
  %188 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.11, i32 noundef %187)
  br label %189

189:                                              ; preds = %178, %186
  %190 = add nuw nsw i32 %152, 1
  %191 = icmp eq i32 %190, %17
  br i1 %191, label %141, label %151, !llvm.loop !49

192:                                              ; preds = %292
  %193 = icmp sgt i32 %18, 1
  %194 = icmp eq i32 %18, 1
  %195 = getelementptr inbounds double, ptr %33, i64 %20
  br label %303

196:                                              ; preds = %141, %292
  %197 = phi i32 [ 0, %141 ], [ %293, %292 ]
  br i1 %142, label %201, label %198, !prof !41

198:                                              ; preds = %196
  switch i32 %18, label %245 [
    i32 1, label %199
    i32 0, label %280
  ]

199:                                              ; preds = %198
  %200 = load double, ptr %31, align 8, !tbaa !32
  store double %200, ptr %33, align 8, !tbaa !32
  tail call void @_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterIPFbddEEEEvT_S7_T0_T1_(ptr noundef nonnull %33, ptr noundef nonnull %143, i64 noundef %147, ptr nonnull @_Z19less_than_function2dd)
  br label %280

201:                                              ; preds = %196
  tail call void @llvm.memcpy.p0.p0.i64(ptr nonnull align 8 %33, ptr nonnull align 8 %31, i64 %22, i1 false)
  tail call void @_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterIPFbddEEEEvT_S7_T0_T1_(ptr noundef nonnull %33, ptr noundef nonnull %143, i64 noundef %147, ptr nonnull @_Z19less_than_function2dd)
  br i1 %148, label %202, label %246

202:                                              ; preds = %201, %224
  %203 = phi i64 [ %226, %224 ], [ 8, %201 ]
  %204 = phi ptr [ %205, %224 ], [ %33, %201 ]
  %205 = getelementptr inbounds nuw i8, ptr %33, i64 %203
  %206 = load double, ptr %205, align 8, !tbaa !32
  %207 = load double, ptr %33, align 8, !tbaa !32
  %208 = fcmp olt double %206, %207
  br i1 %208, label %209, label %214

209:                                              ; preds = %202
  %210 = icmp samesign ugt i64 %203, 8
  br i1 %210, label %211, label %212, !prof !41

211:                                              ; preds = %209
  tail call void @llvm.memmove.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(1) %149, ptr noundef nonnull align 8 dereferenceable(1) %33, i64 %203, i1 false)
  br label %224

212:                                              ; preds = %209
  %213 = getelementptr inbounds nuw i8, ptr %204, i64 8
  store double %207, ptr %213, align 8, !tbaa !32
  br label %224

214:                                              ; preds = %202
  %215 = load double, ptr %204, align 8, !tbaa !32
  %216 = fcmp olt double %206, %215
  br i1 %216, label %217, label %224

217:                                              ; preds = %214, %217
  %218 = phi double [ %222, %217 ], [ %215, %214 ]
  %219 = phi ptr [ %221, %217 ], [ %204, %214 ]
  %220 = phi ptr [ %219, %217 ], [ %205, %214 ]
  store double %218, ptr %220, align 8, !tbaa !32
  %221 = getelementptr inbounds i8, ptr %219, i64 -8
  %222 = load double, ptr %221, align 8, !tbaa !32
  %223 = fcmp olt double %206, %222
  br i1 %223, label %217, label %224, !llvm.loop !50

224:                                              ; preds = %217, %214, %212, %211
  %225 = phi ptr [ %33, %211 ], [ %33, %212 ], [ %205, %214 ], [ %219, %217 ]
  store double %206, ptr %225, align 8, !tbaa !32
  %226 = add nuw nsw i64 %203, 8
  %227 = icmp eq i64 %226, 128
  br i1 %227, label %228, label %202, !llvm.loop !51

228:                                              ; preds = %224, %241
  %229 = phi ptr [ %243, %241 ], [ %150, %224 ]
  %230 = load double, ptr %229, align 8, !tbaa !32
  %231 = getelementptr inbounds i8, ptr %229, i64 -8
  %232 = load double, ptr %231, align 8, !tbaa !32
  %233 = fcmp olt double %230, %232
  br i1 %233, label %234, label %241

234:                                              ; preds = %228, %234
  %235 = phi double [ %239, %234 ], [ %232, %228 ]
  %236 = phi ptr [ %238, %234 ], [ %231, %228 ]
  %237 = phi ptr [ %236, %234 ], [ %229, %228 ]
  store double %235, ptr %237, align 8, !tbaa !32
  %238 = getelementptr inbounds i8, ptr %236, i64 -8
  %239 = load double, ptr %238, align 8, !tbaa !32
  %240 = fcmp olt double %230, %239
  br i1 %240, label %234, label %241, !llvm.loop !50

241:                                              ; preds = %234, %228
  %242 = phi ptr [ %229, %228 ], [ %236, %234 ]
  store double %230, ptr %242, align 8, !tbaa !32
  %243 = getelementptr inbounds nuw i8, ptr %229, i64 8
  %244 = icmp eq ptr %243, %143
  br i1 %244, label %280, label %228, !llvm.loop !52

245:                                              ; preds = %198
  tail call void @_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterIPFbddEEEEvT_S7_T0_T1_(ptr noundef nonnull %33, ptr noundef nonnull %143, i64 noundef %147, ptr nonnull @_Z19less_than_function2dd)
  br label %246

246:                                              ; preds = %201, %245
  br label %247

247:                                              ; preds = %246, %276
  %248 = phi ptr [ %278, %276 ], [ %149, %246 ]
  %249 = phi ptr [ %248, %276 ], [ %33, %246 ]
  %250 = load double, ptr %248, align 8, !tbaa !32
  %251 = load double, ptr %33, align 8, !tbaa !32
  %252 = fcmp olt double %250, %251
  br i1 %252, label %253, label %266

253:                                              ; preds = %247
  %254 = ptrtoint ptr %248 to i64
  %255 = sub i64 %254, %144
  %256 = ashr exact i64 %255, 3
  %257 = icmp sgt i64 %256, 1
  br i1 %257, label %258, label %262, !prof !41

258:                                              ; preds = %253
  %259 = getelementptr inbounds nuw i8, ptr %249, i64 16
  %260 = sub nsw i64 0, %256
  %261 = getelementptr inbounds double, ptr %259, i64 %260
  tail call void @llvm.memmove.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(1) %261, ptr noundef nonnull align 8 dereferenceable(1) %33, i64 %255, i1 false)
  br label %276

262:                                              ; preds = %253
  %263 = icmp eq i64 %255, 8
  br i1 %263, label %264, label %276

264:                                              ; preds = %262
  %265 = getelementptr inbounds nuw i8, ptr %249, i64 8
  store double %251, ptr %265, align 8, !tbaa !32
  br label %276

266:                                              ; preds = %247
  %267 = load double, ptr %249, align 8, !tbaa !32
  %268 = fcmp olt double %250, %267
  br i1 %268, label %269, label %276

269:                                              ; preds = %266, %269
  %270 = phi double [ %274, %269 ], [ %267, %266 ]
  %271 = phi ptr [ %273, %269 ], [ %249, %266 ]
  %272 = phi ptr [ %271, %269 ], [ %248, %266 ]
  store double %270, ptr %272, align 8, !tbaa !32
  %273 = getelementptr inbounds i8, ptr %271, i64 -8
  %274 = load double, ptr %273, align 8, !tbaa !32
  %275 = fcmp olt double %250, %274
  br i1 %275, label %269, label %276, !llvm.loop !50

276:                                              ; preds = %269, %266, %264, %262, %258
  %277 = phi ptr [ %33, %258 ], [ %33, %262 ], [ %33, %264 ], [ %248, %266 ], [ %271, %269 ]
  store double %250, ptr %277, align 8, !tbaa !32
  %278 = getelementptr inbounds nuw i8, ptr %248, i64 8
  %279 = icmp eq ptr %278, %143
  br i1 %279, label %280, label %247, !llvm.loop !51

280:                                              ; preds = %276, %241, %199, %198
  br label %281

281:                                              ; preds = %280, %285
  %282 = phi ptr [ %283, %285 ], [ %33, %280 ]
  %283 = getelementptr i8, ptr %282, i64 8
  %284 = icmp eq ptr %283, %143
  br i1 %284, label %292, label %285

285:                                              ; preds = %281
  %286 = load double, ptr %283, align 8, !tbaa !32
  %287 = load double, ptr %282, align 8, !tbaa !32
  %288 = fcmp olt double %286, %287
  br i1 %288, label %289, label %281, !llvm.loop !42

289:                                              ; preds = %285
  %290 = load i32, ptr @current_test, align 4, !tbaa !11
  %291 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.11, i32 noundef %290)
  br label %292

292:                                              ; preds = %281, %289
  %293 = add nuw nsw i32 %197, 1
  %294 = icmp eq i32 %293, %17
  br i1 %294, label %192, label %196, !llvm.loop !53

295:                                              ; preds = %321
  %296 = getelementptr inbounds i8, ptr %33, i64 %22
  %297 = icmp eq i32 %18, 0
  %298 = tail call range(i64 0, 65) i64 @llvm.ctlz.i64(i64 %20, i1 true)
  %299 = shl nuw nsw i64 %298, 1
  %300 = xor i64 %299, 126
  %301 = icmp sgt i32 %18, 1
  %302 = icmp eq i32 %18, 1
  br label %328

303:                                              ; preds = %192, %321
  %304 = phi i32 [ 0, %192 ], [ %322, %321 ]
  br i1 %193, label %305, label %306, !prof !41

305:                                              ; preds = %303
  tail call void @llvm.memcpy.p0.p0.i64(ptr nonnull align 8 %33, ptr nonnull align 8 %31, i64 %22, i1 false)
  br label %309

306:                                              ; preds = %303
  br i1 %194, label %307, label %309

307:                                              ; preds = %306
  %308 = load double, ptr %31, align 8, !tbaa !32
  store double %308, ptr %33, align 8, !tbaa !32
  br label %309

309:                                              ; preds = %305, %306, %307
  tail call void @_Z9quicksortIPd17less_than_functorEvT_S2_T0_(ptr noundef nonnull %33, ptr noundef nonnull %195, i8 undef)
  br label %310

310:                                              ; preds = %314, %309
  %311 = phi ptr [ %33, %309 ], [ %312, %314 ]
  %312 = getelementptr i8, ptr %311, i64 8
  %313 = icmp eq ptr %312, %195
  br i1 %313, label %321, label %314

314:                                              ; preds = %310
  %315 = load double, ptr %312, align 8, !tbaa !32
  %316 = load double, ptr %311, align 8, !tbaa !32
  %317 = fcmp olt double %315, %316
  br i1 %317, label %318, label %310, !llvm.loop !42

318:                                              ; preds = %314
  %319 = load i32, ptr @current_test, align 4, !tbaa !11
  %320 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.11, i32 noundef %319)
  br label %321

321:                                              ; preds = %310, %318
  %322 = add nuw nsw i32 %304, 1
  %323 = icmp eq i32 %322, %17
  br i1 %323, label %295, label %303, !llvm.loop !54

324:                                              ; preds = %348
  %325 = icmp sgt i32 %18, 1
  %326 = icmp eq i32 %18, 1
  %327 = getelementptr inbounds double, ptr %33, i64 %20
  br label %357

328:                                              ; preds = %295, %348
  %329 = phi i32 [ 0, %295 ], [ %349, %348 ]
  br i1 %297, label %331, label %330

330:                                              ; preds = %328
  tail call void @_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterI17less_than_functorEEEvT_S6_T0_T1_(ptr noundef nonnull %33, ptr noundef nonnull %296, i64 noundef %300, i64 0)
  tail call void @_ZSt22__final_insertion_sortIPdN9__gnu_cxx5__ops15_Iter_comp_iterI17less_than_functorEEEvT_S6_T0_(ptr noundef nonnull %33, ptr noundef nonnull %296, i64 0)
  br label %331

331:                                              ; preds = %328, %330
  br label %332

332:                                              ; preds = %331, %336
  %333 = phi ptr [ %334, %336 ], [ %33, %331 ]
  %334 = getelementptr i8, ptr %333, i64 8
  %335 = icmp eq ptr %334, %296
  br i1 %335, label %343, label %336

336:                                              ; preds = %332
  %337 = load double, ptr %334, align 8, !tbaa !32
  %338 = load double, ptr %333, align 8, !tbaa !32
  %339 = fcmp olt double %337, %338
  br i1 %339, label %340, label %332, !llvm.loop !42

340:                                              ; preds = %336
  %341 = load i32, ptr @current_test, align 4, !tbaa !11
  %342 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.11, i32 noundef %341)
  br label %343

343:                                              ; preds = %332, %340
  br i1 %301, label %344, label %345, !prof !41

344:                                              ; preds = %343
  tail call void @llvm.memcpy.p0.p0.i64(ptr nonnull align 8 %33, ptr nonnull align 8 %31, i64 %22, i1 false)
  br label %348

345:                                              ; preds = %343
  br i1 %302, label %346, label %348

346:                                              ; preds = %345
  %347 = load double, ptr %31, align 8, !tbaa !32
  store double %347, ptr %33, align 8, !tbaa !32
  br label %348

348:                                              ; preds = %344, %345, %346
  %349 = add nuw nsw i32 %329, 1
  %350 = icmp eq i32 %349, %17
  br i1 %350, label %324, label %328, !llvm.loop !55

351:                                              ; preds = %375
  %352 = icmp sgt i32 %18, 1
  %353 = getelementptr inbounds i8, ptr %33, i64 %22
  %354 = tail call range(i64 0, 65) i64 @llvm.ctlz.i64(i64 %20, i1 true)
  %355 = shl nuw nsw i64 %354, 1
  %356 = xor i64 %355, 126
  br label %382

357:                                              ; preds = %324, %375
  %358 = phi i32 [ 0, %324 ], [ %376, %375 ]
  br i1 %325, label %359, label %360, !prof !41

359:                                              ; preds = %357
  tail call void @llvm.memcpy.p0.p0.i64(ptr nonnull align 8 %33, ptr nonnull align 8 %31, i64 %22, i1 false)
  br label %363

360:                                              ; preds = %357
  br i1 %326, label %361, label %363

361:                                              ; preds = %360
  %362 = load double, ptr %31, align 8, !tbaa !32
  store double %362, ptr %33, align 8, !tbaa !32
  br label %363

363:                                              ; preds = %359, %360, %361
  tail call void @_Z9quicksortIPd24inline_less_than_functorEvT_S2_T0_(ptr noundef nonnull %33, ptr noundef nonnull %327, i8 undef)
  br label %364

364:                                              ; preds = %368, %363
  %365 = phi ptr [ %33, %363 ], [ %366, %368 ]
  %366 = getelementptr i8, ptr %365, i64 8
  %367 = icmp eq ptr %366, %327
  br i1 %367, label %375, label %368

368:                                              ; preds = %364
  %369 = load double, ptr %366, align 8, !tbaa !32
  %370 = load double, ptr %365, align 8, !tbaa !32
  %371 = fcmp olt double %369, %370
  br i1 %371, label %372, label %364, !llvm.loop !42

372:                                              ; preds = %368
  %373 = load i32, ptr @current_test, align 4, !tbaa !11
  %374 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.11, i32 noundef %373)
  br label %375

375:                                              ; preds = %364, %372
  %376 = add nuw nsw i32 %358, 1
  %377 = icmp eq i32 %376, %17
  br i1 %377, label %351, label %357, !llvm.loop !56

378:                                              ; preds = %401
  %379 = icmp sgt i32 %18, 1
  %380 = icmp eq i32 %18, 1
  %381 = getelementptr inbounds double, ptr %33, i64 %20
  br label %410

382:                                              ; preds = %351, %401
  %383 = phi i32 [ 0, %351 ], [ %402, %401 ]
  br i1 %352, label %384, label %385, !prof !41

384:                                              ; preds = %382
  tail call void @llvm.memcpy.p0.p0.i64(ptr nonnull align 8 %33, ptr nonnull align 8 %31, i64 %22, i1 false)
  br label %388

385:                                              ; preds = %382
  switch i32 %18, label %388 [
    i32 1, label %386
    i32 0, label %389
  ]

386:                                              ; preds = %385
  %387 = load double, ptr %31, align 8, !tbaa !32
  store double %387, ptr %33, align 8, !tbaa !32
  br label %388

388:                                              ; preds = %386, %384, %385
  tail call void @_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterI24inline_less_than_functorEEEvT_S6_T0_T1_(ptr noundef nonnull %33, ptr noundef nonnull %353, i64 noundef %356, i64 0)
  tail call void @_ZSt22__final_insertion_sortIPdN9__gnu_cxx5__ops15_Iter_comp_iterI24inline_less_than_functorEEEvT_S6_T0_(ptr noundef nonnull %33, ptr noundef nonnull %353, i64 0)
  br label %389

389:                                              ; preds = %385, %388
  br label %390

390:                                              ; preds = %389, %394
  %391 = phi ptr [ %392, %394 ], [ %33, %389 ]
  %392 = getelementptr i8, ptr %391, i64 8
  %393 = icmp eq ptr %392, %353
  br i1 %393, label %401, label %394

394:                                              ; preds = %390
  %395 = load double, ptr %392, align 8, !tbaa !32
  %396 = load double, ptr %391, align 8, !tbaa !32
  %397 = fcmp olt double %395, %396
  br i1 %397, label %398, label %390, !llvm.loop !42

398:                                              ; preds = %394
  %399 = load i32, ptr @current_test, align 4, !tbaa !11
  %400 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.11, i32 noundef %399)
  br label %401

401:                                              ; preds = %390, %398
  %402 = add nuw nsw i32 %383, 1
  %403 = icmp eq i32 %402, %17
  br i1 %403, label %378, label %382, !llvm.loop !57

404:                                              ; preds = %428
  %405 = icmp sgt i32 %18, 1
  %406 = getelementptr inbounds i8, ptr %33, i64 %22
  %407 = tail call range(i64 0, 65) i64 @llvm.ctlz.i64(i64 %20, i1 true)
  %408 = shl nuw nsw i64 %407, 1
  %409 = xor i64 %408, 126
  br label %435

410:                                              ; preds = %378, %428
  %411 = phi i32 [ 0, %378 ], [ %429, %428 ]
  br i1 %379, label %412, label %413, !prof !41

412:                                              ; preds = %410
  tail call void @llvm.memcpy.p0.p0.i64(ptr nonnull align 8 %33, ptr nonnull align 8 %31, i64 %22, i1 false)
  br label %416

413:                                              ; preds = %410
  br i1 %380, label %414, label %416

414:                                              ; preds = %413
  %415 = load double, ptr %31, align 8, !tbaa !32
  store double %415, ptr %33, align 8, !tbaa !32
  br label %416

416:                                              ; preds = %412, %413, %414
  tail call void @_Z9quicksortIPdSt4lessIdEEvT_S3_T0_(ptr noundef nonnull %33, ptr noundef nonnull %381, i8 undef)
  br label %417

417:                                              ; preds = %421, %416
  %418 = phi ptr [ %33, %416 ], [ %419, %421 ]
  %419 = getelementptr i8, ptr %418, i64 8
  %420 = icmp eq ptr %419, %381
  br i1 %420, label %428, label %421

421:                                              ; preds = %417
  %422 = load double, ptr %419, align 8, !tbaa !32
  %423 = load double, ptr %418, align 8, !tbaa !32
  %424 = fcmp olt double %422, %423
  br i1 %424, label %425, label %417, !llvm.loop !42

425:                                              ; preds = %421
  %426 = load i32, ptr @current_test, align 4, !tbaa !11
  %427 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.11, i32 noundef %426)
  br label %428

428:                                              ; preds = %417, %425
  %429 = add nuw nsw i32 %411, 1
  %430 = icmp eq i32 %429, %17
  br i1 %430, label %404, label %410, !llvm.loop !58

431:                                              ; preds = %454
  %432 = icmp sgt i32 %18, 1
  %433 = icmp eq i32 %18, 1
  %434 = getelementptr inbounds double, ptr %33, i64 %20
  br label %463

435:                                              ; preds = %404, %454
  %436 = phi i32 [ 0, %404 ], [ %455, %454 ]
  br i1 %405, label %437, label %438, !prof !41

437:                                              ; preds = %435
  tail call void @llvm.memcpy.p0.p0.i64(ptr nonnull align 8 %33, ptr nonnull align 8 %31, i64 %22, i1 false)
  br label %441

438:                                              ; preds = %435
  switch i32 %18, label %441 [
    i32 1, label %439
    i32 0, label %442
  ]

439:                                              ; preds = %438
  %440 = load double, ptr %31, align 8, !tbaa !32
  store double %440, ptr %33, align 8, !tbaa !32
  br label %441

441:                                              ; preds = %439, %437, %438
  tail call void @_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterISt4lessIdEEEEvT_S7_T0_T1_(ptr noundef nonnull %33, ptr noundef nonnull %406, i64 noundef %409, i64 0)
  tail call void @_ZSt22__final_insertion_sortIPdN9__gnu_cxx5__ops15_Iter_comp_iterISt4lessIdEEEEvT_S7_T0_(ptr noundef nonnull %33, ptr noundef nonnull %406, i64 0)
  br label %442

442:                                              ; preds = %438, %441
  br label %443

443:                                              ; preds = %442, %447
  %444 = phi ptr [ %445, %447 ], [ %33, %442 ]
  %445 = getelementptr i8, ptr %444, i64 8
  %446 = icmp eq ptr %445, %406
  br i1 %446, label %454, label %447

447:                                              ; preds = %443
  %448 = load double, ptr %445, align 8, !tbaa !32
  %449 = load double, ptr %444, align 8, !tbaa !32
  %450 = fcmp olt double %448, %449
  br i1 %450, label %451, label %443, !llvm.loop !42

451:                                              ; preds = %447
  %452 = load i32, ptr @current_test, align 4, !tbaa !11
  %453 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.11, i32 noundef %452)
  br label %454

454:                                              ; preds = %443, %451
  %455 = add nuw nsw i32 %436, 1
  %456 = icmp eq i32 %455, %17
  br i1 %456, label %431, label %435, !llvm.loop !59

457:                                              ; preds = %481
  %458 = icmp sgt i32 %18, 1
  %459 = getelementptr inbounds i8, ptr %33, i64 %22
  %460 = tail call range(i64 0, 65) i64 @llvm.ctlz.i64(i64 %20, i1 true)
  %461 = shl nuw nsw i64 %460, 1
  %462 = xor i64 %461, 126
  br label %484

463:                                              ; preds = %431, %481
  %464 = phi i32 [ 0, %431 ], [ %482, %481 ]
  br i1 %432, label %465, label %466, !prof !41

465:                                              ; preds = %463
  tail call void @llvm.memcpy.p0.p0.i64(ptr nonnull align 8 %33, ptr nonnull align 8 %31, i64 %22, i1 false)
  br label %469

466:                                              ; preds = %463
  br i1 %433, label %467, label %469

467:                                              ; preds = %466
  %468 = load double, ptr %31, align 8, !tbaa !32
  store double %468, ptr %33, align 8, !tbaa !32
  br label %469

469:                                              ; preds = %465, %466, %467
  tail call void @_Z9quicksortIPdEvT_S1_(ptr noundef nonnull %33, ptr noundef nonnull %434)
  br label %470

470:                                              ; preds = %474, %469
  %471 = phi ptr [ %33, %469 ], [ %472, %474 ]
  %472 = getelementptr i8, ptr %471, i64 8
  %473 = icmp eq ptr %472, %434
  br i1 %473, label %481, label %474

474:                                              ; preds = %470
  %475 = load double, ptr %472, align 8, !tbaa !32
  %476 = load double, ptr %471, align 8, !tbaa !32
  %477 = fcmp olt double %475, %476
  br i1 %477, label %478, label %470, !llvm.loop !42

478:                                              ; preds = %474
  %479 = load i32, ptr @current_test, align 4, !tbaa !11
  %480 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.11, i32 noundef %479)
  br label %481

481:                                              ; preds = %470, %478
  %482 = add nuw nsw i32 %464, 1
  %483 = icmp eq i32 %482, %17
  br i1 %483, label %457, label %463, !llvm.loop !60

484:                                              ; preds = %457, %503
  %485 = phi i32 [ 0, %457 ], [ %504, %503 ]
  br i1 %458, label %486, label %487, !prof !41

486:                                              ; preds = %484
  tail call void @llvm.memcpy.p0.p0.i64(ptr nonnull align 8 %33, ptr nonnull align 8 %31, i64 %22, i1 false)
  br label %490

487:                                              ; preds = %484
  switch i32 %18, label %490 [
    i32 1, label %488
    i32 0, label %491
  ]

488:                                              ; preds = %487
  %489 = load double, ptr %31, align 8, !tbaa !32
  store double %489, ptr %33, align 8, !tbaa !32
  br label %490

490:                                              ; preds = %488, %486, %487
  tail call void @_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_(ptr noundef nonnull %33, ptr noundef nonnull %459, i64 noundef %462, i8 undef)
  tail call void @_ZSt22__final_insertion_sortIPdN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_(ptr noundef nonnull %33, ptr noundef nonnull %459, i8 undef)
  br label %491

491:                                              ; preds = %487, %490
  br label %492

492:                                              ; preds = %491, %496
  %493 = phi ptr [ %494, %496 ], [ %33, %491 ]
  %494 = getelementptr i8, ptr %493, i64 8
  %495 = icmp eq ptr %494, %459
  br i1 %495, label %503, label %496

496:                                              ; preds = %492
  %497 = load double, ptr %494, align 8, !tbaa !32
  %498 = load double, ptr %493, align 8, !tbaa !32
  %499 = fcmp olt double %497, %498
  br i1 %499, label %500, label %492, !llvm.loop !42

500:                                              ; preds = %496
  %501 = load i32, ptr @current_test, align 4, !tbaa !11
  %502 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.11, i32 noundef %501)
  br label %503

503:                                              ; preds = %492, %500
  %504 = add nuw nsw i32 %485, 1
  %505 = icmp eq i32 %504, %17
  br i1 %505, label %506, label %484, !llvm.loop !61

506:                                              ; preds = %503, %30
  tail call void @_ZdaPv(ptr noundef nonnull %33) #25
  tail call void @_ZdaPv(ptr noundef nonnull %31) #25
  ret i32 0
}

; Function Attrs: nounwind
declare void @srand(i32 noundef) local_unnamed_addr #8

; Function Attrs: nobuiltin allocsize(0)
declare noundef nonnull ptr @_Znam(i64 noundef) local_unnamed_addr #13

; Function Attrs: nounwind
declare i32 @rand() local_unnamed_addr #8

; Function Attrs: nofree
declare void @qsort(ptr noundef, i64 noundef, i64 noundef, ptr noundef captures(none)) local_unnamed_addr #14

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_Z9quicksortIPd17less_than_functorEvT_S2_T0_(ptr noundef %0, ptr noundef %1, i8 %2) local_unnamed_addr #11 comdat {
  %4 = ptrtoint ptr %1 to i64
  %5 = ptrtoint ptr %0 to i64
  %6 = sub i64 %4, %5
  %7 = icmp sgt i64 %6, 8
  br i1 %7, label %8, label %33

8:                                                ; preds = %3, %29
  %9 = phi ptr [ %15, %29 ], [ %0, %3 ]
  %10 = load double, ptr %9, align 8, !tbaa !32
  br label %11

11:                                               ; preds = %28, %8
  %12 = phi ptr [ %1, %8 ], [ %16, %28 ]
  %13 = phi ptr [ %9, %8 ], [ %22, %28 ]
  br label %14

14:                                               ; preds = %14, %11
  %15 = phi ptr [ %12, %11 ], [ %16, %14 ]
  %16 = getelementptr inbounds i8, ptr %15, i64 -8
  %17 = load double, ptr %16, align 8, !tbaa !32
  %18 = fcmp olt double %10, %17
  br i1 %18, label %14, label %19, !llvm.loop !62

19:                                               ; preds = %14
  %20 = icmp ult ptr %13, %16
  br i1 %20, label %21, label %29

21:                                               ; preds = %19, %21
  %22 = phi ptr [ %25, %21 ], [ %13, %19 ]
  %23 = load double, ptr %22, align 8, !tbaa !32
  %24 = fcmp olt double %23, %10
  %25 = getelementptr inbounds nuw i8, ptr %22, i64 8
  br i1 %24, label %21, label %26, !llvm.loop !63

26:                                               ; preds = %21
  %27 = icmp ult ptr %22, %16
  br i1 %27, label %28, label %29

28:                                               ; preds = %26
  store double %23, ptr %16, align 8, !tbaa !32
  store double %17, ptr %22, align 8, !tbaa !32
  br label %11, !llvm.loop !64

29:                                               ; preds = %26, %19
  tail call void @_Z9quicksortIPd17less_than_functorEvT_S2_T0_(ptr noundef nonnull %9, ptr noundef nonnull %15, i8 undef)
  %30 = ptrtoint ptr %15 to i64
  %31 = sub i64 %4, %30
  %32 = icmp sgt i64 %31, 8
  br i1 %32, label %8, label %33

33:                                               ; preds = %29, %3
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_Z9quicksortIPd24inline_less_than_functorEvT_S2_T0_(ptr noundef %0, ptr noundef %1, i8 %2) local_unnamed_addr #11 comdat {
  %4 = ptrtoint ptr %1 to i64
  %5 = ptrtoint ptr %0 to i64
  %6 = sub i64 %4, %5
  %7 = icmp sgt i64 %6, 8
  br i1 %7, label %8, label %33

8:                                                ; preds = %3, %29
  %9 = phi ptr [ %15, %29 ], [ %0, %3 ]
  %10 = load double, ptr %9, align 8, !tbaa !32
  br label %11

11:                                               ; preds = %28, %8
  %12 = phi ptr [ %1, %8 ], [ %16, %28 ]
  %13 = phi ptr [ %9, %8 ], [ %22, %28 ]
  br label %14

14:                                               ; preds = %14, %11
  %15 = phi ptr [ %12, %11 ], [ %16, %14 ]
  %16 = getelementptr inbounds i8, ptr %15, i64 -8
  %17 = load double, ptr %16, align 8, !tbaa !32
  %18 = fcmp olt double %10, %17
  br i1 %18, label %14, label %19, !llvm.loop !65

19:                                               ; preds = %14
  %20 = icmp ult ptr %13, %16
  br i1 %20, label %21, label %29

21:                                               ; preds = %19, %21
  %22 = phi ptr [ %25, %21 ], [ %13, %19 ]
  %23 = load double, ptr %22, align 8, !tbaa !32
  %24 = fcmp olt double %23, %10
  %25 = getelementptr inbounds nuw i8, ptr %22, i64 8
  br i1 %24, label %21, label %26, !llvm.loop !66

26:                                               ; preds = %21
  %27 = icmp ult ptr %22, %16
  br i1 %27, label %28, label %29

28:                                               ; preds = %26
  store double %23, ptr %16, align 8, !tbaa !32
  store double %17, ptr %22, align 8, !tbaa !32
  br label %11, !llvm.loop !67

29:                                               ; preds = %26, %19
  tail call void @_Z9quicksortIPd24inline_less_than_functorEvT_S2_T0_(ptr noundef nonnull %9, ptr noundef nonnull %15, i8 undef)
  %30 = ptrtoint ptr %15 to i64
  %31 = sub i64 %4, %30
  %32 = icmp sgt i64 %31, 8
  br i1 %32, label %8, label %33

33:                                               ; preds = %29, %3
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_Z9quicksortIPdSt4lessIdEEvT_S3_T0_(ptr noundef %0, ptr noundef %1, i8 %2) local_unnamed_addr #11 comdat {
  %4 = ptrtoint ptr %1 to i64
  %5 = ptrtoint ptr %0 to i64
  %6 = sub i64 %4, %5
  %7 = icmp sgt i64 %6, 8
  br i1 %7, label %8, label %33

8:                                                ; preds = %3, %29
  %9 = phi ptr [ %15, %29 ], [ %0, %3 ]
  %10 = load double, ptr %9, align 8, !tbaa !32
  br label %11

11:                                               ; preds = %28, %8
  %12 = phi ptr [ %1, %8 ], [ %16, %28 ]
  %13 = phi ptr [ %9, %8 ], [ %22, %28 ]
  br label %14

14:                                               ; preds = %14, %11
  %15 = phi ptr [ %12, %11 ], [ %16, %14 ]
  %16 = getelementptr inbounds i8, ptr %15, i64 -8
  %17 = load double, ptr %16, align 8, !tbaa !32
  %18 = fcmp olt double %10, %17
  br i1 %18, label %14, label %19, !llvm.loop !68

19:                                               ; preds = %14
  %20 = icmp ult ptr %13, %16
  br i1 %20, label %21, label %29

21:                                               ; preds = %19, %21
  %22 = phi ptr [ %25, %21 ], [ %13, %19 ]
  %23 = load double, ptr %22, align 8, !tbaa !32
  %24 = fcmp olt double %23, %10
  %25 = getelementptr inbounds nuw i8, ptr %22, i64 8
  br i1 %24, label %21, label %26, !llvm.loop !69

26:                                               ; preds = %21
  %27 = icmp ult ptr %22, %16
  br i1 %27, label %28, label %29

28:                                               ; preds = %26
  store double %23, ptr %16, align 8, !tbaa !32
  store double %17, ptr %22, align 8, !tbaa !32
  br label %11, !llvm.loop !70

29:                                               ; preds = %26, %19
  tail call void @_Z9quicksortIPdSt4lessIdEEvT_S3_T0_(ptr noundef nonnull %9, ptr noundef nonnull %15, i8 undef)
  %30 = ptrtoint ptr %15 to i64
  %31 = sub i64 %4, %30
  %32 = icmp sgt i64 %31, 8
  br i1 %32, label %8, label %33

33:                                               ; preds = %29, %3
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_Z9quicksortIPdEvT_S1_(ptr noundef %0, ptr noundef %1) local_unnamed_addr #11 comdat {
  %3 = ptrtoint ptr %1 to i64
  %4 = ptrtoint ptr %0 to i64
  %5 = sub i64 %3, %4
  %6 = icmp sgt i64 %5, 8
  br i1 %6, label %7, label %32

7:                                                ; preds = %2, %28
  %8 = phi ptr [ %14, %28 ], [ %0, %2 ]
  %9 = load double, ptr %8, align 8, !tbaa !32
  br label %10

10:                                               ; preds = %27, %7
  %11 = phi ptr [ %1, %7 ], [ %15, %27 ]
  %12 = phi ptr [ %8, %7 ], [ %21, %27 ]
  br label %13

13:                                               ; preds = %13, %10
  %14 = phi ptr [ %11, %10 ], [ %15, %13 ]
  %15 = getelementptr inbounds i8, ptr %14, i64 -8
  %16 = load double, ptr %15, align 8, !tbaa !32
  %17 = fcmp olt double %9, %16
  br i1 %17, label %13, label %18, !llvm.loop !71

18:                                               ; preds = %13
  %19 = icmp ult ptr %12, %15
  br i1 %19, label %20, label %28

20:                                               ; preds = %18, %20
  %21 = phi ptr [ %24, %20 ], [ %12, %18 ]
  %22 = load double, ptr %21, align 8, !tbaa !32
  %23 = fcmp olt double %22, %9
  %24 = getelementptr inbounds nuw i8, ptr %21, i64 8
  br i1 %23, label %20, label %25, !llvm.loop !72

25:                                               ; preds = %20
  %26 = icmp ult ptr %21, %15
  br i1 %26, label %27, label %28

27:                                               ; preds = %25
  store double %22, ptr %15, align 8, !tbaa !32
  store double %16, ptr %21, align 8, !tbaa !32
  br label %10, !llvm.loop !73

28:                                               ; preds = %25, %18
  tail call void @_Z9quicksortIPdEvT_S1_(ptr noundef nonnull %8, ptr noundef nonnull %14)
  %29 = ptrtoint ptr %14 to i64
  %30 = sub i64 %3, %29
  %31 = icmp sgt i64 %30, 8
  br i1 %31, label %7, label %32

32:                                               ; preds = %28, %2
  ret void
}

; Function Attrs: nobuiltin nounwind
declare void @_ZdaPv(ptr noundef) local_unnamed_addr #15

; Function Attrs: nounwind
declare i64 @__isoc23_strtol(ptr noundef, ptr noundef, i32 noundef) local_unnamed_addr #8

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memmove.p0.p0.i64(ptr writeonly captures(none), ptr readonly captures(none), i64, i1 immarg) #16

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterIPFbddEEEEvT_S7_T0_T1_(ptr noundef %0, ptr noundef %1, i64 noundef %2, ptr %3) local_unnamed_addr #11 comdat {
  %5 = alloca %"struct.__gnu_cxx::__ops::_Iter_comp_iter", align 8
  %6 = ptrtoint ptr %0 to i64
  %7 = ptrtoint ptr %1 to i64
  %8 = sub i64 %7, %6
  %9 = icmp sgt i64 %8, 128
  br i1 %9, label %10, label %142

10:                                               ; preds = %4
  %11 = getelementptr inbounds nuw i8, ptr %0, i64 8
  br label %12

12:                                               ; preds = %10, %138
  %13 = phi i64 [ %8, %10 ], [ %140, %138 ]
  %14 = phi ptr [ %1, %10 ], [ %122, %138 ]
  %15 = phi i64 [ %2, %10 ], [ %79, %138 ]
  %16 = icmp eq i64 %15, 0
  br i1 %16, label %17, label %78

17:                                               ; preds = %12
  call void @llvm.lifetime.start.p0(ptr nonnull %5)
  store ptr %3, ptr %5, align 8
  call void @_ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_comp_iterIPFbddEEEEvT_S7_RT0_(ptr noundef %0, ptr noundef %14, ptr noundef nonnull align 8 dereferenceable(8) %5)
  call void @llvm.lifetime.end.p0(ptr nonnull %5)
  br label %18

18:                                               ; preds = %17, %74
  %19 = phi ptr [ %20, %74 ], [ %14, %17 ]
  %20 = getelementptr inbounds i8, ptr %19, i64 -8
  %21 = load double, ptr %20, align 8, !tbaa !32
  %22 = load double, ptr %0, align 8, !tbaa !32
  store double %22, ptr %20, align 8, !tbaa !32
  %23 = ptrtoint ptr %20 to i64
  %24 = sub i64 %23, %6
  %25 = ashr exact i64 %24, 3
  %26 = add nsw i64 %25, -1
  %27 = sdiv i64 %26, 2
  %28 = icmp sgt i64 %25, 2
  br i1 %28, label %29, label %45

29:                                               ; preds = %18, %29
  %30 = phi i64 [ %40, %29 ], [ 0, %18 ]
  %31 = shl i64 %30, 1
  %32 = add i64 %31, 2
  %33 = getelementptr inbounds double, ptr %0, i64 %32
  %34 = getelementptr double, ptr %0, i64 %31
  %35 = getelementptr i8, ptr %34, i64 8
  %36 = load double, ptr %33, align 8, !tbaa !32
  %37 = load double, ptr %35, align 8, !tbaa !32
  %38 = call noundef i1 %3(double noundef %36, double noundef %37)
  %39 = or disjoint i64 %31, 1
  %40 = select i1 %38, i64 %39, i64 %32
  %41 = getelementptr inbounds double, ptr %0, i64 %40
  %42 = load double, ptr %41, align 8, !tbaa !32
  %43 = getelementptr inbounds double, ptr %0, i64 %30
  store double %42, ptr %43, align 8, !tbaa !32
  %44 = icmp slt i64 %40, %27
  br i1 %44, label %29, label %45, !llvm.loop !74

45:                                               ; preds = %29, %18
  %46 = phi i64 [ 0, %18 ], [ %40, %29 ]
  %47 = and i64 %24, 8
  %48 = icmp eq i64 %47, 0
  br i1 %48, label %49, label %59

49:                                               ; preds = %45
  %50 = add nsw i64 %25, -2
  %51 = ashr exact i64 %50, 1
  %52 = icmp eq i64 %46, %51
  br i1 %52, label %53, label %59

53:                                               ; preds = %49
  %54 = shl nuw nsw i64 %46, 1
  %55 = or disjoint i64 %54, 1
  %56 = getelementptr inbounds nuw double, ptr %0, i64 %55
  %57 = load double, ptr %56, align 8, !tbaa !32
  %58 = getelementptr inbounds double, ptr %0, i64 %46
  store double %57, ptr %58, align 8, !tbaa !32
  br label %61

59:                                               ; preds = %49, %45
  %60 = icmp eq i64 %46, 0
  br i1 %60, label %74, label %61

61:                                               ; preds = %59, %53
  %62 = phi i64 [ %46, %59 ], [ %55, %53 ]
  br label %63

63:                                               ; preds = %61, %70
  %64 = phi i64 [ %66, %70 ], [ %62, %61 ]
  %65 = add nsw i64 %64, -1
  %66 = lshr i64 %65, 1
  %67 = getelementptr inbounds nuw double, ptr %0, i64 %66
  %68 = load double, ptr %67, align 8, !tbaa !32
  %69 = call noundef i1 %3(double noundef %68, double noundef %21)
  br i1 %69, label %70, label %74

70:                                               ; preds = %63
  %71 = load double, ptr %67, align 8, !tbaa !32
  %72 = getelementptr inbounds double, ptr %0, i64 %64
  store double %71, ptr %72, align 8, !tbaa !32
  %73 = icmp ult i64 %65, 2
  br i1 %73, label %74, label %63, !llvm.loop !75

74:                                               ; preds = %70, %63, %59
  %75 = phi i64 [ 0, %59 ], [ %64, %63 ], [ 0, %70 ]
  %76 = getelementptr inbounds double, ptr %0, i64 %75
  store double %21, ptr %76, align 8, !tbaa !32
  %77 = icmp sgt i64 %24, 8
  br i1 %77, label %18, label %142, !llvm.loop !76

78:                                               ; preds = %12
  %79 = add nsw i64 %15, -1
  %80 = lshr i64 %13, 4
  %81 = getelementptr inbounds nuw double, ptr %0, i64 %80
  %82 = getelementptr inbounds i8, ptr %14, i64 -8
  %83 = load double, ptr %11, align 8, !tbaa !32
  %84 = load double, ptr %81, align 8, !tbaa !32
  %85 = tail call noundef i1 %3(double noundef %83, double noundef %84)
  %86 = load double, ptr %82, align 8, !tbaa !32
  br i1 %85, label %87, label %102

87:                                               ; preds = %78
  %88 = load double, ptr %81, align 8, !tbaa !32
  %89 = tail call noundef i1 %3(double noundef %88, double noundef %86)
  br i1 %89, label %90, label %93

90:                                               ; preds = %87
  %91 = load double, ptr %0, align 8, !tbaa !32
  %92 = load double, ptr %81, align 8, !tbaa !32
  store double %92, ptr %0, align 8, !tbaa !32
  store double %91, ptr %81, align 8, !tbaa !32
  br label %117

93:                                               ; preds = %87
  %94 = load double, ptr %11, align 8, !tbaa !32
  %95 = load double, ptr %82, align 8, !tbaa !32
  %96 = tail call noundef i1 %3(double noundef %94, double noundef %95)
  %97 = load double, ptr %0, align 8, !tbaa !32
  br i1 %96, label %98, label %100

98:                                               ; preds = %93
  %99 = load double, ptr %82, align 8, !tbaa !32
  store double %99, ptr %0, align 8, !tbaa !32
  store double %97, ptr %82, align 8, !tbaa !32
  br label %117

100:                                              ; preds = %93
  %101 = load double, ptr %11, align 8, !tbaa !32
  store double %101, ptr %0, align 8, !tbaa !32
  store double %97, ptr %11, align 8, !tbaa !32
  br label %117

102:                                              ; preds = %78
  %103 = load double, ptr %11, align 8, !tbaa !32
  %104 = tail call noundef i1 %3(double noundef %103, double noundef %86)
  br i1 %104, label %105, label %108

105:                                              ; preds = %102
  %106 = load <2 x double>, ptr %0, align 8, !tbaa !32
  %107 = shufflevector <2 x double> %106, <2 x double> poison, <2 x i32> <i32 1, i32 0>
  store <2 x double> %107, ptr %0, align 8, !tbaa !32
  br label %117

108:                                              ; preds = %102
  %109 = load double, ptr %81, align 8, !tbaa !32
  %110 = load double, ptr %82, align 8, !tbaa !32
  %111 = tail call noundef i1 %3(double noundef %109, double noundef %110)
  %112 = load double, ptr %0, align 8, !tbaa !32
  br i1 %111, label %113, label %115

113:                                              ; preds = %108
  %114 = load double, ptr %82, align 8, !tbaa !32
  store double %114, ptr %0, align 8, !tbaa !32
  store double %112, ptr %82, align 8, !tbaa !32
  br label %117

115:                                              ; preds = %108
  %116 = load double, ptr %81, align 8, !tbaa !32
  store double %116, ptr %0, align 8, !tbaa !32
  store double %112, ptr %81, align 8, !tbaa !32
  br label %117

117:                                              ; preds = %115, %113, %105, %100, %98, %90
  br label %118

118:                                              ; preds = %117, %135
  %119 = phi ptr [ %129, %135 ], [ %14, %117 ]
  %120 = phi ptr [ %126, %135 ], [ %11, %117 ]
  br label %121

121:                                              ; preds = %121, %118
  %122 = phi ptr [ %120, %118 ], [ %126, %121 ]
  %123 = load double, ptr %122, align 8, !tbaa !32
  %124 = load double, ptr %0, align 8, !tbaa !32
  %125 = tail call noundef i1 %3(double noundef %123, double noundef %124)
  %126 = getelementptr inbounds nuw i8, ptr %122, i64 8
  br i1 %125, label %121, label %127, !llvm.loop !77

127:                                              ; preds = %121, %127
  %128 = phi ptr [ %129, %127 ], [ %119, %121 ]
  %129 = getelementptr inbounds i8, ptr %128, i64 -8
  %130 = load double, ptr %0, align 8, !tbaa !32
  %131 = load double, ptr %129, align 8, !tbaa !32
  %132 = tail call noundef i1 %3(double noundef %130, double noundef %131)
  br i1 %132, label %127, label %133, !llvm.loop !78

133:                                              ; preds = %127
  %134 = icmp ult ptr %122, %129
  br i1 %134, label %135, label %138

135:                                              ; preds = %133
  %136 = load double, ptr %122, align 8, !tbaa !32
  %137 = load double, ptr %129, align 8, !tbaa !32
  store double %137, ptr %122, align 8, !tbaa !32
  store double %136, ptr %129, align 8, !tbaa !32
  br label %118, !llvm.loop !79

138:                                              ; preds = %133
  tail call void @_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterIPFbddEEEEvT_S7_T0_T1_(ptr noundef nonnull %122, ptr noundef %14, i64 noundef %79, ptr %3)
  %139 = ptrtoint ptr %122 to i64
  %140 = sub i64 %139, %6
  %141 = icmp sgt i64 %140, 128
  br i1 %141, label %12, label %142, !llvm.loop !80

142:                                              ; preds = %138, %74, %4
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_comp_iterIPFbddEEEEvT_S7_RT0_(ptr noundef %0, ptr noundef %1, ptr noundef nonnull align 8 dereferenceable(8) %2) local_unnamed_addr #11 comdat {
  %4 = ptrtoint ptr %1 to i64
  %5 = ptrtoint ptr %0 to i64
  %6 = sub i64 %4, %5
  %7 = ashr exact i64 %6, 3
  %8 = icmp slt i64 %7, 2
  br i1 %8, label %107, label %9

9:                                                ; preds = %3
  %10 = add nsw i64 %7, -2
  %11 = lshr i64 %10, 1
  %12 = add nsw i64 %7, -1
  %13 = lshr i64 %12, 1
  %14 = and i64 %6, 8
  %15 = icmp eq i64 %14, 0
  %16 = lshr exact i64 %10, 1
  br i1 %15, label %17, label %21

17:                                               ; preds = %9
  %18 = or disjoint i64 %10, 1
  %19 = getelementptr inbounds nuw double, ptr %0, i64 %18
  %20 = getelementptr inbounds nuw double, ptr %0, i64 %16
  br label %61

21:                                               ; preds = %9, %56
  %22 = phi i64 [ %60, %56 ], [ %11, %9 ]
  %23 = getelementptr inbounds nuw double, ptr %0, i64 %22
  %24 = load double, ptr %23, align 8, !tbaa !32
  %25 = load ptr, ptr %2, align 8, !tbaa !81
  %26 = icmp slt i64 %22, %13
  br i1 %26, label %27, label %56

27:                                               ; preds = %21, %27
  %28 = phi i64 [ %38, %27 ], [ %22, %21 ]
  %29 = shl i64 %28, 1
  %30 = add i64 %29, 2
  %31 = getelementptr inbounds double, ptr %0, i64 %30
  %32 = getelementptr double, ptr %0, i64 %29
  %33 = getelementptr i8, ptr %32, i64 8
  %34 = load double, ptr %31, align 8, !tbaa !32
  %35 = load double, ptr %33, align 8, !tbaa !32
  %36 = tail call noundef i1 %25(double noundef %34, double noundef %35)
  %37 = or disjoint i64 %29, 1
  %38 = select i1 %36, i64 %37, i64 %30
  %39 = getelementptr inbounds double, ptr %0, i64 %38
  %40 = load double, ptr %39, align 8, !tbaa !32
  %41 = getelementptr inbounds double, ptr %0, i64 %28
  store double %40, ptr %41, align 8, !tbaa !32
  %42 = icmp slt i64 %38, %13
  br i1 %42, label %27, label %43, !llvm.loop !74

43:                                               ; preds = %27
  %44 = icmp sgt i64 %38, %22
  br i1 %44, label %45, label %56

45:                                               ; preds = %43, %52
  %46 = phi i64 [ %48, %52 ], [ %38, %43 ]
  %47 = add nsw i64 %46, -1
  %48 = sdiv i64 %47, 2
  %49 = getelementptr inbounds nuw double, ptr %0, i64 %48
  %50 = load double, ptr %49, align 8, !tbaa !32
  %51 = tail call noundef i1 %25(double noundef %50, double noundef %24)
  br i1 %51, label %52, label %56

52:                                               ; preds = %45
  %53 = load double, ptr %49, align 8, !tbaa !32
  %54 = getelementptr inbounds nuw double, ptr %0, i64 %46
  store double %53, ptr %54, align 8, !tbaa !32
  %55 = icmp sgt i64 %48, %22
  br i1 %55, label %45, label %56, !llvm.loop !75

56:                                               ; preds = %45, %52, %21, %43
  %57 = phi i64 [ %38, %43 ], [ %22, %21 ], [ %48, %52 ], [ %46, %45 ]
  %58 = getelementptr inbounds nuw double, ptr %0, i64 %57
  store double %24, ptr %58, align 8, !tbaa !32
  %59 = icmp eq i64 %22, 0
  %60 = add nsw i64 %22, -1
  br i1 %59, label %107, label %21, !llvm.loop !82

61:                                               ; preds = %17, %102
  %62 = phi i64 [ %106, %102 ], [ %11, %17 ]
  %63 = getelementptr inbounds nuw double, ptr %0, i64 %62
  %64 = load double, ptr %63, align 8, !tbaa !32
  %65 = load ptr, ptr %2, align 8, !tbaa !81
  %66 = icmp slt i64 %62, %13
  br i1 %66, label %67, label %83

67:                                               ; preds = %61, %67
  %68 = phi i64 [ %78, %67 ], [ %62, %61 ]
  %69 = shl i64 %68, 1
  %70 = add i64 %69, 2
  %71 = getelementptr inbounds double, ptr %0, i64 %70
  %72 = getelementptr double, ptr %0, i64 %69
  %73 = getelementptr i8, ptr %72, i64 8
  %74 = load double, ptr %71, align 8, !tbaa !32
  %75 = load double, ptr %73, align 8, !tbaa !32
  %76 = tail call noundef i1 %65(double noundef %74, double noundef %75)
  %77 = or disjoint i64 %69, 1
  %78 = select i1 %76, i64 %77, i64 %70
  %79 = getelementptr inbounds double, ptr %0, i64 %78
  %80 = load double, ptr %79, align 8, !tbaa !32
  %81 = getelementptr inbounds double, ptr %0, i64 %68
  store double %80, ptr %81, align 8, !tbaa !32
  %82 = icmp slt i64 %78, %13
  br i1 %82, label %67, label %83, !llvm.loop !74

83:                                               ; preds = %67, %61
  %84 = phi i64 [ %62, %61 ], [ %78, %67 ]
  %85 = icmp eq i64 %84, %16
  br i1 %85, label %86, label %88

86:                                               ; preds = %83
  %87 = load double, ptr %19, align 8, !tbaa !32
  store double %87, ptr %20, align 8, !tbaa !32
  br label %88

88:                                               ; preds = %86, %83
  %89 = phi i64 [ %18, %86 ], [ %84, %83 ]
  %90 = icmp sgt i64 %89, %62
  br i1 %90, label %91, label %102

91:                                               ; preds = %88, %98
  %92 = phi i64 [ %94, %98 ], [ %89, %88 ]
  %93 = add nsw i64 %92, -1
  %94 = sdiv i64 %93, 2
  %95 = getelementptr inbounds nuw double, ptr %0, i64 %94
  %96 = load double, ptr %95, align 8, !tbaa !32
  %97 = tail call noundef i1 %65(double noundef %96, double noundef %64)
  br i1 %97, label %98, label %102

98:                                               ; preds = %91
  %99 = load double, ptr %95, align 8, !tbaa !32
  %100 = getelementptr inbounds nuw double, ptr %0, i64 %92
  store double %99, ptr %100, align 8, !tbaa !32
  %101 = icmp sgt i64 %94, %62
  br i1 %101, label %91, label %102, !llvm.loop !75

102:                                              ; preds = %91, %98, %88
  %103 = phi i64 [ %89, %88 ], [ %94, %98 ], [ %92, %91 ]
  %104 = getelementptr inbounds nuw double, ptr %0, i64 %103
  store double %64, ptr %104, align 8, !tbaa !32
  %105 = icmp eq i64 %62, 0
  %106 = add nsw i64 %62, -1
  br i1 %105, label %107, label %61, !llvm.loop !82

107:                                              ; preds = %56, %102, %3
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.ctlz.i64(i64, i1 immarg) #17

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterI17less_than_functorEEEvT_S6_T0_T1_(ptr noundef %0, ptr noundef %1, i64 noundef %2, i64 %3) local_unnamed_addr #11 comdat {
  %5 = alloca %"struct.__gnu_cxx::__ops::_Iter_comp_iter.0", align 4
  %6 = ptrtoint ptr %0 to i64
  %7 = ptrtoint ptr %1 to i64
  %8 = sub i64 %7, %6
  %9 = icmp sgt i64 %8, 128
  br i1 %9, label %10, label %128

10:                                               ; preds = %4
  %11 = and i64 %3, 255
  %12 = getelementptr inbounds nuw i8, ptr %0, i64 8
  br label %13

13:                                               ; preds = %10, %124
  %14 = phi i64 [ %8, %10 ], [ %126, %124 ]
  %15 = phi ptr [ %1, %10 ], [ %112, %124 ]
  %16 = phi i64 [ %2, %10 ], [ %80, %124 ]
  %17 = icmp eq i64 %16, 0
  br i1 %17, label %18, label %79

18:                                               ; preds = %13
  call void @llvm.lifetime.start.p0(ptr nonnull %5)
  %19 = trunc i64 %3 to i8
  store i8 %19, ptr %5, align 4
  call void @_ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_comp_iterI17less_than_functorEEEvT_S6_RT0_(ptr noundef %0, ptr noundef %15, ptr noundef nonnull align 1 dereferenceable(1) %5)
  call void @llvm.lifetime.end.p0(ptr nonnull %5)
  br label %20

20:                                               ; preds = %18, %75
  %21 = phi ptr [ %22, %75 ], [ %15, %18 ]
  %22 = getelementptr inbounds i8, ptr %21, i64 -8
  %23 = load double, ptr %22, align 8, !tbaa !32
  %24 = load double, ptr %0, align 8, !tbaa !32
  store double %24, ptr %22, align 8, !tbaa !32
  %25 = ptrtoint ptr %22 to i64
  %26 = sub i64 %25, %6
  %27 = ashr exact i64 %26, 3
  %28 = add nsw i64 %27, -1
  %29 = sdiv i64 %28, 2
  %30 = icmp sgt i64 %27, 2
  br i1 %30, label %31, label %47

31:                                               ; preds = %20, %31
  %32 = phi i64 [ %42, %31 ], [ 0, %20 ]
  %33 = shl i64 %32, 1
  %34 = add i64 %33, 2
  %35 = getelementptr inbounds double, ptr %0, i64 %34
  %36 = getelementptr double, ptr %0, i64 %33
  %37 = getelementptr i8, ptr %36, i64 8
  %38 = load double, ptr %35, align 8, !tbaa !32
  %39 = load double, ptr %37, align 8, !tbaa !32
  %40 = fcmp olt double %38, %39
  %41 = or disjoint i64 %33, 1
  %42 = select i1 %40, i64 %41, i64 %34
  %43 = getelementptr inbounds double, ptr %0, i64 %42
  %44 = load double, ptr %43, align 8, !tbaa !32
  %45 = getelementptr inbounds double, ptr %0, i64 %32
  store double %44, ptr %45, align 8, !tbaa !32
  %46 = icmp slt i64 %42, %29
  br i1 %46, label %31, label %47, !llvm.loop !83

47:                                               ; preds = %31, %20
  %48 = phi i64 [ 0, %20 ], [ %42, %31 ]
  %49 = and i64 %26, 8
  %50 = icmp eq i64 %49, 0
  br i1 %50, label %51, label %61

51:                                               ; preds = %47
  %52 = add nsw i64 %27, -2
  %53 = ashr exact i64 %52, 1
  %54 = icmp eq i64 %48, %53
  br i1 %54, label %55, label %61

55:                                               ; preds = %51
  %56 = shl nuw nsw i64 %48, 1
  %57 = or disjoint i64 %56, 1
  %58 = getelementptr inbounds nuw double, ptr %0, i64 %57
  %59 = load double, ptr %58, align 8, !tbaa !32
  %60 = getelementptr inbounds double, ptr %0, i64 %48
  store double %59, ptr %60, align 8, !tbaa !32
  br label %63

61:                                               ; preds = %51, %47
  %62 = icmp eq i64 %48, 0
  br i1 %62, label %75, label %63

63:                                               ; preds = %61, %55
  %64 = phi i64 [ %48, %61 ], [ %57, %55 ]
  br label %65

65:                                               ; preds = %63, %72
  %66 = phi i64 [ %68, %72 ], [ %64, %63 ]
  %67 = add nsw i64 %66, -1
  %68 = lshr i64 %67, 1
  %69 = getelementptr inbounds nuw double, ptr %0, i64 %68
  %70 = load double, ptr %69, align 8, !tbaa !32
  %71 = fcmp olt double %70, %23
  br i1 %71, label %72, label %75

72:                                               ; preds = %65
  %73 = getelementptr inbounds double, ptr %0, i64 %66
  store double %70, ptr %73, align 8, !tbaa !32
  %74 = icmp ult i64 %67, 2
  br i1 %74, label %75, label %65, !llvm.loop !84

75:                                               ; preds = %72, %65, %61
  %76 = phi i64 [ 0, %61 ], [ %66, %65 ], [ 0, %72 ]
  %77 = getelementptr inbounds double, ptr %0, i64 %76
  store double %23, ptr %77, align 8, !tbaa !32
  %78 = icmp sgt i64 %26, 8
  br i1 %78, label %20, label %128, !llvm.loop !85

79:                                               ; preds = %13
  %80 = add nsw i64 %16, -1
  %81 = lshr i64 %14, 4
  %82 = getelementptr inbounds nuw double, ptr %0, i64 %81
  %83 = getelementptr inbounds i8, ptr %15, i64 -8
  %84 = load double, ptr %12, align 8, !tbaa !32
  %85 = load double, ptr %82, align 8, !tbaa !32
  %86 = fcmp olt double %84, %85
  %87 = load double, ptr %83, align 8, !tbaa !32
  br i1 %86, label %88, label %97

88:                                               ; preds = %79
  %89 = fcmp olt double %85, %87
  br i1 %89, label %90, label %92

90:                                               ; preds = %88
  %91 = load double, ptr %0, align 8, !tbaa !32
  store double %85, ptr %0, align 8, !tbaa !32
  store double %91, ptr %82, align 8, !tbaa !32
  br label %106

92:                                               ; preds = %88
  %93 = fcmp olt double %84, %87
  %94 = load double, ptr %0, align 8, !tbaa !32
  br i1 %93, label %95, label %96

95:                                               ; preds = %92
  store double %87, ptr %0, align 8, !tbaa !32
  store double %94, ptr %83, align 8, !tbaa !32
  br label %106

96:                                               ; preds = %92
  store double %84, ptr %0, align 8, !tbaa !32
  store double %94, ptr %12, align 8, !tbaa !32
  br label %106

97:                                               ; preds = %79
  %98 = fcmp olt double %84, %87
  br i1 %98, label %99, label %101

99:                                               ; preds = %97
  %100 = load double, ptr %0, align 8, !tbaa !32
  store double %84, ptr %0, align 8, !tbaa !32
  store double %100, ptr %12, align 8, !tbaa !32
  br label %106

101:                                              ; preds = %97
  %102 = fcmp olt double %85, %87
  %103 = load double, ptr %0, align 8, !tbaa !32
  br i1 %102, label %104, label %105

104:                                              ; preds = %101
  store double %87, ptr %0, align 8, !tbaa !32
  store double %103, ptr %83, align 8, !tbaa !32
  br label %106

105:                                              ; preds = %101
  store double %85, ptr %0, align 8, !tbaa !32
  store double %103, ptr %82, align 8, !tbaa !32
  br label %106

106:                                              ; preds = %105, %104, %99, %96, %95, %90
  br label %107

107:                                              ; preds = %106, %123
  %108 = phi ptr [ %118, %123 ], [ %15, %106 ]
  %109 = phi ptr [ %115, %123 ], [ %12, %106 ]
  %110 = load double, ptr %0, align 8, !tbaa !32
  br label %111

111:                                              ; preds = %111, %107
  %112 = phi ptr [ %109, %107 ], [ %115, %111 ]
  %113 = load double, ptr %112, align 8, !tbaa !32
  %114 = fcmp olt double %113, %110
  %115 = getelementptr inbounds nuw i8, ptr %112, i64 8
  br i1 %114, label %111, label %116, !llvm.loop !86

116:                                              ; preds = %111, %116
  %117 = phi ptr [ %118, %116 ], [ %108, %111 ]
  %118 = getelementptr inbounds i8, ptr %117, i64 -8
  %119 = load double, ptr %118, align 8, !tbaa !32
  %120 = fcmp olt double %110, %119
  br i1 %120, label %116, label %121, !llvm.loop !87

121:                                              ; preds = %116
  %122 = icmp ult ptr %112, %118
  br i1 %122, label %123, label %124

123:                                              ; preds = %121
  store double %119, ptr %112, align 8, !tbaa !32
  store double %113, ptr %118, align 8, !tbaa !32
  br label %107, !llvm.loop !88

124:                                              ; preds = %121
  tail call void @_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterI17less_than_functorEEEvT_S6_T0_T1_(ptr noundef nonnull %112, ptr noundef %15, i64 noundef %80, i64 %11)
  %125 = ptrtoint ptr %112 to i64
  %126 = sub i64 %125, %6
  %127 = icmp sgt i64 %126, 128
  br i1 %127, label %13, label %128, !llvm.loop !89

128:                                              ; preds = %124, %75, %4
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZSt22__final_insertion_sortIPdN9__gnu_cxx5__ops15_Iter_comp_iterI17less_than_functorEEEvT_S6_T0_(ptr noundef %0, ptr noundef %1, i64 %2) local_unnamed_addr #11 comdat {
  %4 = ptrtoint ptr %1 to i64
  %5 = ptrtoint ptr %0 to i64
  %6 = sub i64 %4, %5
  %7 = icmp sgt i64 %6, 128
  br i1 %7, label %8, label %56

8:                                                ; preds = %3
  %9 = getelementptr i8, ptr %0, i64 8
  br label %10

10:                                               ; preds = %32, %8
  %11 = phi i64 [ 8, %8 ], [ %34, %32 ]
  %12 = phi ptr [ %0, %8 ], [ %13, %32 ]
  %13 = getelementptr inbounds nuw i8, ptr %0, i64 %11
  %14 = load double, ptr %13, align 8, !tbaa !32
  %15 = load double, ptr %0, align 8, !tbaa !32
  %16 = fcmp olt double %14, %15
  br i1 %16, label %17, label %22

17:                                               ; preds = %10
  %18 = icmp samesign ugt i64 %11, 8
  br i1 %18, label %19, label %20, !prof !41

19:                                               ; preds = %17
  tail call void @llvm.memmove.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(1) %9, ptr noundef nonnull align 8 dereferenceable(1) %0, i64 %11, i1 false)
  br label %32

20:                                               ; preds = %17
  %21 = getelementptr inbounds nuw i8, ptr %12, i64 8
  store double %15, ptr %21, align 8, !tbaa !32
  br label %32

22:                                               ; preds = %10
  %23 = load double, ptr %12, align 8, !tbaa !32
  %24 = fcmp olt double %14, %23
  br i1 %24, label %25, label %32

25:                                               ; preds = %22, %25
  %26 = phi double [ %30, %25 ], [ %23, %22 ]
  %27 = phi ptr [ %29, %25 ], [ %12, %22 ]
  %28 = phi ptr [ %27, %25 ], [ %13, %22 ]
  store double %26, ptr %28, align 8, !tbaa !32
  %29 = getelementptr inbounds i8, ptr %27, i64 -8
  %30 = load double, ptr %29, align 8, !tbaa !32
  %31 = fcmp olt double %14, %30
  br i1 %31, label %25, label %32, !llvm.loop !90

32:                                               ; preds = %25, %22, %20, %19
  %33 = phi ptr [ %0, %19 ], [ %0, %20 ], [ %13, %22 ], [ %27, %25 ]
  store double %14, ptr %33, align 8, !tbaa !32
  %34 = add nuw nsw i64 %11, 8
  %35 = icmp eq i64 %34, 128
  br i1 %35, label %36, label %10, !llvm.loop !91

36:                                               ; preds = %32
  %37 = getelementptr inbounds nuw i8, ptr %0, i64 128
  %38 = icmp eq ptr %37, %1
  br i1 %38, label %94, label %39

39:                                               ; preds = %36, %52
  %40 = phi ptr [ %54, %52 ], [ %37, %36 ]
  %41 = load double, ptr %40, align 8, !tbaa !32
  %42 = getelementptr inbounds i8, ptr %40, i64 -8
  %43 = load double, ptr %42, align 8, !tbaa !32
  %44 = fcmp olt double %41, %43
  br i1 %44, label %45, label %52

45:                                               ; preds = %39, %45
  %46 = phi double [ %50, %45 ], [ %43, %39 ]
  %47 = phi ptr [ %49, %45 ], [ %42, %39 ]
  %48 = phi ptr [ %47, %45 ], [ %40, %39 ]
  store double %46, ptr %48, align 8, !tbaa !32
  %49 = getelementptr inbounds i8, ptr %47, i64 -8
  %50 = load double, ptr %49, align 8, !tbaa !32
  %51 = fcmp olt double %41, %50
  br i1 %51, label %45, label %52, !llvm.loop !90

52:                                               ; preds = %45, %39
  %53 = phi ptr [ %40, %39 ], [ %47, %45 ]
  store double %41, ptr %53, align 8, !tbaa !32
  %54 = getelementptr inbounds nuw i8, ptr %40, i64 8
  %55 = icmp eq ptr %54, %1
  br i1 %55, label %94, label %39, !llvm.loop !92

56:                                               ; preds = %3
  %57 = icmp eq ptr %0, %1
  %58 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %59 = icmp eq ptr %58, %1
  %60 = select i1 %57, i1 true, i1 %59
  br i1 %60, label %94, label %61

61:                                               ; preds = %56, %90
  %62 = phi ptr [ %92, %90 ], [ %58, %56 ]
  %63 = phi ptr [ %62, %90 ], [ %0, %56 ]
  %64 = load double, ptr %62, align 8, !tbaa !32
  %65 = load double, ptr %0, align 8, !tbaa !32
  %66 = fcmp olt double %64, %65
  br i1 %66, label %67, label %80

67:                                               ; preds = %61
  %68 = ptrtoint ptr %62 to i64
  %69 = sub i64 %68, %5
  %70 = ashr exact i64 %69, 3
  %71 = icmp sgt i64 %70, 1
  br i1 %71, label %72, label %76, !prof !41

72:                                               ; preds = %67
  %73 = getelementptr inbounds nuw i8, ptr %63, i64 16
  %74 = sub nsw i64 0, %70
  %75 = getelementptr inbounds double, ptr %73, i64 %74
  tail call void @llvm.memmove.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(1) %75, ptr noundef nonnull align 8 dereferenceable(1) %0, i64 %69, i1 false)
  br label %90

76:                                               ; preds = %67
  %77 = icmp eq i64 %69, 8
  br i1 %77, label %78, label %90

78:                                               ; preds = %76
  %79 = getelementptr inbounds nuw i8, ptr %63, i64 8
  store double %65, ptr %79, align 8, !tbaa !32
  br label %90

80:                                               ; preds = %61
  %81 = load double, ptr %63, align 8, !tbaa !32
  %82 = fcmp olt double %64, %81
  br i1 %82, label %83, label %90

83:                                               ; preds = %80, %83
  %84 = phi double [ %88, %83 ], [ %81, %80 ]
  %85 = phi ptr [ %87, %83 ], [ %63, %80 ]
  %86 = phi ptr [ %85, %83 ], [ %62, %80 ]
  store double %84, ptr %86, align 8, !tbaa !32
  %87 = getelementptr inbounds i8, ptr %85, i64 -8
  %88 = load double, ptr %87, align 8, !tbaa !32
  %89 = fcmp olt double %64, %88
  br i1 %89, label %83, label %90, !llvm.loop !90

90:                                               ; preds = %83, %80, %78, %76, %72
  %91 = phi ptr [ %0, %72 ], [ %0, %76 ], [ %0, %78 ], [ %62, %80 ], [ %85, %83 ]
  store double %64, ptr %91, align 8, !tbaa !32
  %92 = getelementptr inbounds nuw i8, ptr %62, i64 8
  %93 = icmp eq ptr %92, %1
  br i1 %93, label %94, label %61, !llvm.loop !91

94:                                               ; preds = %90, %52, %56, %36
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_comp_iterI17less_than_functorEEEvT_S6_RT0_(ptr noundef %0, ptr noundef %1, ptr noundef nonnull align 1 dereferenceable(1) %2) local_unnamed_addr #11 comdat {
  %4 = ptrtoint ptr %1 to i64
  %5 = ptrtoint ptr %0 to i64
  %6 = sub i64 %4, %5
  %7 = ashr exact i64 %6, 3
  %8 = icmp slt i64 %7, 2
  br i1 %8, label %103, label %9

9:                                                ; preds = %3
  %10 = add nsw i64 %7, -2
  %11 = lshr i64 %10, 1
  %12 = add nsw i64 %7, -1
  %13 = lshr i64 %12, 1
  %14 = and i64 %6, 8
  %15 = icmp eq i64 %14, 0
  %16 = lshr exact i64 %10, 1
  br i1 %15, label %17, label %21

17:                                               ; preds = %9
  %18 = or disjoint i64 %10, 1
  %19 = getelementptr inbounds nuw double, ptr %0, i64 %18
  %20 = getelementptr inbounds nuw double, ptr %0, i64 %16
  br label %59

21:                                               ; preds = %9, %54
  %22 = phi i64 [ %58, %54 ], [ %11, %9 ]
  %23 = getelementptr inbounds nuw double, ptr %0, i64 %22
  %24 = load double, ptr %23, align 8, !tbaa !32
  %25 = icmp slt i64 %22, %13
  br i1 %25, label %26, label %54

26:                                               ; preds = %21, %26
  %27 = phi i64 [ %37, %26 ], [ %22, %21 ]
  %28 = shl i64 %27, 1
  %29 = add i64 %28, 2
  %30 = getelementptr inbounds double, ptr %0, i64 %29
  %31 = getelementptr double, ptr %0, i64 %28
  %32 = getelementptr i8, ptr %31, i64 8
  %33 = load double, ptr %30, align 8, !tbaa !32
  %34 = load double, ptr %32, align 8, !tbaa !32
  %35 = fcmp olt double %33, %34
  %36 = or disjoint i64 %28, 1
  %37 = select i1 %35, i64 %36, i64 %29
  %38 = getelementptr inbounds double, ptr %0, i64 %37
  %39 = load double, ptr %38, align 8, !tbaa !32
  %40 = getelementptr inbounds double, ptr %0, i64 %27
  store double %39, ptr %40, align 8, !tbaa !32
  %41 = icmp slt i64 %37, %13
  br i1 %41, label %26, label %42, !llvm.loop !83

42:                                               ; preds = %26
  %43 = icmp sgt i64 %37, %22
  br i1 %43, label %44, label %54

44:                                               ; preds = %42, %51
  %45 = phi i64 [ %47, %51 ], [ %37, %42 ]
  %46 = add nsw i64 %45, -1
  %47 = sdiv i64 %46, 2
  %48 = getelementptr inbounds nuw double, ptr %0, i64 %47
  %49 = load double, ptr %48, align 8, !tbaa !32
  %50 = fcmp olt double %49, %24
  br i1 %50, label %51, label %54

51:                                               ; preds = %44
  %52 = getelementptr inbounds nuw double, ptr %0, i64 %45
  store double %49, ptr %52, align 8, !tbaa !32
  %53 = icmp sgt i64 %47, %22
  br i1 %53, label %44, label %54, !llvm.loop !84

54:                                               ; preds = %44, %51, %21, %42
  %55 = phi i64 [ %37, %42 ], [ %22, %21 ], [ %47, %51 ], [ %45, %44 ]
  %56 = getelementptr inbounds nuw double, ptr %0, i64 %55
  store double %24, ptr %56, align 8, !tbaa !32
  %57 = icmp eq i64 %22, 0
  %58 = add nsw i64 %22, -1
  br i1 %57, label %103, label %21, !llvm.loop !93

59:                                               ; preds = %17, %98
  %60 = phi i64 [ %102, %98 ], [ %11, %17 ]
  %61 = getelementptr inbounds nuw double, ptr %0, i64 %60
  %62 = load double, ptr %61, align 8, !tbaa !32
  %63 = icmp slt i64 %60, %13
  br i1 %63, label %64, label %80

64:                                               ; preds = %59, %64
  %65 = phi i64 [ %75, %64 ], [ %60, %59 ]
  %66 = shl i64 %65, 1
  %67 = add i64 %66, 2
  %68 = getelementptr inbounds double, ptr %0, i64 %67
  %69 = getelementptr double, ptr %0, i64 %66
  %70 = getelementptr i8, ptr %69, i64 8
  %71 = load double, ptr %68, align 8, !tbaa !32
  %72 = load double, ptr %70, align 8, !tbaa !32
  %73 = fcmp olt double %71, %72
  %74 = or disjoint i64 %66, 1
  %75 = select i1 %73, i64 %74, i64 %67
  %76 = getelementptr inbounds double, ptr %0, i64 %75
  %77 = load double, ptr %76, align 8, !tbaa !32
  %78 = getelementptr inbounds double, ptr %0, i64 %65
  store double %77, ptr %78, align 8, !tbaa !32
  %79 = icmp slt i64 %75, %13
  br i1 %79, label %64, label %80, !llvm.loop !83

80:                                               ; preds = %64, %59
  %81 = phi i64 [ %60, %59 ], [ %75, %64 ]
  %82 = icmp eq i64 %81, %16
  br i1 %82, label %83, label %85

83:                                               ; preds = %80
  %84 = load double, ptr %19, align 8, !tbaa !32
  store double %84, ptr %20, align 8, !tbaa !32
  br label %85

85:                                               ; preds = %83, %80
  %86 = phi i64 [ %18, %83 ], [ %81, %80 ]
  %87 = icmp sgt i64 %86, %60
  br i1 %87, label %88, label %98

88:                                               ; preds = %85, %95
  %89 = phi i64 [ %91, %95 ], [ %86, %85 ]
  %90 = add nsw i64 %89, -1
  %91 = sdiv i64 %90, 2
  %92 = getelementptr inbounds nuw double, ptr %0, i64 %91
  %93 = load double, ptr %92, align 8, !tbaa !32
  %94 = fcmp olt double %93, %62
  br i1 %94, label %95, label %98

95:                                               ; preds = %88
  %96 = getelementptr inbounds nuw double, ptr %0, i64 %89
  store double %93, ptr %96, align 8, !tbaa !32
  %97 = icmp sgt i64 %91, %60
  br i1 %97, label %88, label %98, !llvm.loop !84

98:                                               ; preds = %88, %95, %85
  %99 = phi i64 [ %86, %85 ], [ %91, %95 ], [ %89, %88 ]
  %100 = getelementptr inbounds nuw double, ptr %0, i64 %99
  store double %62, ptr %100, align 8, !tbaa !32
  %101 = icmp eq i64 %60, 0
  %102 = add nsw i64 %60, -1
  br i1 %101, label %103, label %59, !llvm.loop !93

103:                                              ; preds = %54, %98, %3
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterI24inline_less_than_functorEEEvT_S6_T0_T1_(ptr noundef %0, ptr noundef %1, i64 noundef %2, i64 %3) local_unnamed_addr #11 comdat {
  %5 = alloca %"struct.__gnu_cxx::__ops::_Iter_comp_iter.3", align 4
  %6 = ptrtoint ptr %0 to i64
  %7 = ptrtoint ptr %1 to i64
  %8 = sub i64 %7, %6
  %9 = icmp sgt i64 %8, 128
  br i1 %9, label %10, label %128

10:                                               ; preds = %4
  %11 = and i64 %3, 255
  %12 = getelementptr inbounds nuw i8, ptr %0, i64 8
  br label %13

13:                                               ; preds = %10, %124
  %14 = phi i64 [ %8, %10 ], [ %126, %124 ]
  %15 = phi ptr [ %1, %10 ], [ %112, %124 ]
  %16 = phi i64 [ %2, %10 ], [ %80, %124 ]
  %17 = icmp eq i64 %16, 0
  br i1 %17, label %18, label %79

18:                                               ; preds = %13
  call void @llvm.lifetime.start.p0(ptr nonnull %5)
  %19 = trunc i64 %3 to i8
  store i8 %19, ptr %5, align 4
  call void @_ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_comp_iterI24inline_less_than_functorEEEvT_S6_RT0_(ptr noundef %0, ptr noundef %15, ptr noundef nonnull align 1 dereferenceable(1) %5)
  call void @llvm.lifetime.end.p0(ptr nonnull %5)
  br label %20

20:                                               ; preds = %18, %75
  %21 = phi ptr [ %22, %75 ], [ %15, %18 ]
  %22 = getelementptr inbounds i8, ptr %21, i64 -8
  %23 = load double, ptr %22, align 8, !tbaa !32
  %24 = load double, ptr %0, align 8, !tbaa !32
  store double %24, ptr %22, align 8, !tbaa !32
  %25 = ptrtoint ptr %22 to i64
  %26 = sub i64 %25, %6
  %27 = ashr exact i64 %26, 3
  %28 = add nsw i64 %27, -1
  %29 = sdiv i64 %28, 2
  %30 = icmp sgt i64 %27, 2
  br i1 %30, label %31, label %47

31:                                               ; preds = %20, %31
  %32 = phi i64 [ %42, %31 ], [ 0, %20 ]
  %33 = shl i64 %32, 1
  %34 = add i64 %33, 2
  %35 = getelementptr inbounds double, ptr %0, i64 %34
  %36 = getelementptr double, ptr %0, i64 %33
  %37 = getelementptr i8, ptr %36, i64 8
  %38 = load double, ptr %35, align 8, !tbaa !32
  %39 = load double, ptr %37, align 8, !tbaa !32
  %40 = fcmp olt double %38, %39
  %41 = or disjoint i64 %33, 1
  %42 = select i1 %40, i64 %41, i64 %34
  %43 = getelementptr inbounds double, ptr %0, i64 %42
  %44 = load double, ptr %43, align 8, !tbaa !32
  %45 = getelementptr inbounds double, ptr %0, i64 %32
  store double %44, ptr %45, align 8, !tbaa !32
  %46 = icmp slt i64 %42, %29
  br i1 %46, label %31, label %47, !llvm.loop !94

47:                                               ; preds = %31, %20
  %48 = phi i64 [ 0, %20 ], [ %42, %31 ]
  %49 = and i64 %26, 8
  %50 = icmp eq i64 %49, 0
  br i1 %50, label %51, label %61

51:                                               ; preds = %47
  %52 = add nsw i64 %27, -2
  %53 = ashr exact i64 %52, 1
  %54 = icmp eq i64 %48, %53
  br i1 %54, label %55, label %61

55:                                               ; preds = %51
  %56 = shl nuw nsw i64 %48, 1
  %57 = or disjoint i64 %56, 1
  %58 = getelementptr inbounds nuw double, ptr %0, i64 %57
  %59 = load double, ptr %58, align 8, !tbaa !32
  %60 = getelementptr inbounds double, ptr %0, i64 %48
  store double %59, ptr %60, align 8, !tbaa !32
  br label %63

61:                                               ; preds = %51, %47
  %62 = icmp eq i64 %48, 0
  br i1 %62, label %75, label %63

63:                                               ; preds = %61, %55
  %64 = phi i64 [ %48, %61 ], [ %57, %55 ]
  br label %65

65:                                               ; preds = %63, %72
  %66 = phi i64 [ %68, %72 ], [ %64, %63 ]
  %67 = add nsw i64 %66, -1
  %68 = lshr i64 %67, 1
  %69 = getelementptr inbounds nuw double, ptr %0, i64 %68
  %70 = load double, ptr %69, align 8, !tbaa !32
  %71 = fcmp olt double %70, %23
  br i1 %71, label %72, label %75

72:                                               ; preds = %65
  %73 = getelementptr inbounds double, ptr %0, i64 %66
  store double %70, ptr %73, align 8, !tbaa !32
  %74 = icmp ult i64 %67, 2
  br i1 %74, label %75, label %65, !llvm.loop !95

75:                                               ; preds = %72, %65, %61
  %76 = phi i64 [ 0, %61 ], [ %66, %65 ], [ 0, %72 ]
  %77 = getelementptr inbounds double, ptr %0, i64 %76
  store double %23, ptr %77, align 8, !tbaa !32
  %78 = icmp sgt i64 %26, 8
  br i1 %78, label %20, label %128, !llvm.loop !96

79:                                               ; preds = %13
  %80 = add nsw i64 %16, -1
  %81 = lshr i64 %14, 4
  %82 = getelementptr inbounds nuw double, ptr %0, i64 %81
  %83 = getelementptr inbounds i8, ptr %15, i64 -8
  %84 = load double, ptr %12, align 8, !tbaa !32
  %85 = load double, ptr %82, align 8, !tbaa !32
  %86 = fcmp olt double %84, %85
  %87 = load double, ptr %83, align 8, !tbaa !32
  br i1 %86, label %88, label %97

88:                                               ; preds = %79
  %89 = fcmp olt double %85, %87
  br i1 %89, label %90, label %92

90:                                               ; preds = %88
  %91 = load double, ptr %0, align 8, !tbaa !32
  store double %85, ptr %0, align 8, !tbaa !32
  store double %91, ptr %82, align 8, !tbaa !32
  br label %106

92:                                               ; preds = %88
  %93 = fcmp olt double %84, %87
  %94 = load double, ptr %0, align 8, !tbaa !32
  br i1 %93, label %95, label %96

95:                                               ; preds = %92
  store double %87, ptr %0, align 8, !tbaa !32
  store double %94, ptr %83, align 8, !tbaa !32
  br label %106

96:                                               ; preds = %92
  store double %84, ptr %0, align 8, !tbaa !32
  store double %94, ptr %12, align 8, !tbaa !32
  br label %106

97:                                               ; preds = %79
  %98 = fcmp olt double %84, %87
  br i1 %98, label %99, label %101

99:                                               ; preds = %97
  %100 = load double, ptr %0, align 8, !tbaa !32
  store double %84, ptr %0, align 8, !tbaa !32
  store double %100, ptr %12, align 8, !tbaa !32
  br label %106

101:                                              ; preds = %97
  %102 = fcmp olt double %85, %87
  %103 = load double, ptr %0, align 8, !tbaa !32
  br i1 %102, label %104, label %105

104:                                              ; preds = %101
  store double %87, ptr %0, align 8, !tbaa !32
  store double %103, ptr %83, align 8, !tbaa !32
  br label %106

105:                                              ; preds = %101
  store double %85, ptr %0, align 8, !tbaa !32
  store double %103, ptr %82, align 8, !tbaa !32
  br label %106

106:                                              ; preds = %105, %104, %99, %96, %95, %90
  br label %107

107:                                              ; preds = %106, %123
  %108 = phi ptr [ %118, %123 ], [ %15, %106 ]
  %109 = phi ptr [ %115, %123 ], [ %12, %106 ]
  %110 = load double, ptr %0, align 8, !tbaa !32
  br label %111

111:                                              ; preds = %111, %107
  %112 = phi ptr [ %109, %107 ], [ %115, %111 ]
  %113 = load double, ptr %112, align 8, !tbaa !32
  %114 = fcmp olt double %113, %110
  %115 = getelementptr inbounds nuw i8, ptr %112, i64 8
  br i1 %114, label %111, label %116, !llvm.loop !97

116:                                              ; preds = %111, %116
  %117 = phi ptr [ %118, %116 ], [ %108, %111 ]
  %118 = getelementptr inbounds i8, ptr %117, i64 -8
  %119 = load double, ptr %118, align 8, !tbaa !32
  %120 = fcmp olt double %110, %119
  br i1 %120, label %116, label %121, !llvm.loop !98

121:                                              ; preds = %116
  %122 = icmp ult ptr %112, %118
  br i1 %122, label %123, label %124

123:                                              ; preds = %121
  store double %119, ptr %112, align 8, !tbaa !32
  store double %113, ptr %118, align 8, !tbaa !32
  br label %107, !llvm.loop !99

124:                                              ; preds = %121
  tail call void @_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterI24inline_less_than_functorEEEvT_S6_T0_T1_(ptr noundef nonnull %112, ptr noundef %15, i64 noundef %80, i64 %11)
  %125 = ptrtoint ptr %112 to i64
  %126 = sub i64 %125, %6
  %127 = icmp sgt i64 %126, 128
  br i1 %127, label %13, label %128, !llvm.loop !100

128:                                              ; preds = %124, %75, %4
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZSt22__final_insertion_sortIPdN9__gnu_cxx5__ops15_Iter_comp_iterI24inline_less_than_functorEEEvT_S6_T0_(ptr noundef %0, ptr noundef %1, i64 %2) local_unnamed_addr #11 comdat {
  %4 = ptrtoint ptr %1 to i64
  %5 = ptrtoint ptr %0 to i64
  %6 = sub i64 %4, %5
  %7 = icmp sgt i64 %6, 128
  br i1 %7, label %8, label %56

8:                                                ; preds = %3
  %9 = getelementptr i8, ptr %0, i64 8
  br label %10

10:                                               ; preds = %32, %8
  %11 = phi i64 [ 8, %8 ], [ %34, %32 ]
  %12 = phi ptr [ %0, %8 ], [ %13, %32 ]
  %13 = getelementptr inbounds nuw i8, ptr %0, i64 %11
  %14 = load double, ptr %13, align 8, !tbaa !32
  %15 = load double, ptr %0, align 8, !tbaa !32
  %16 = fcmp olt double %14, %15
  br i1 %16, label %17, label %22

17:                                               ; preds = %10
  %18 = icmp samesign ugt i64 %11, 8
  br i1 %18, label %19, label %20, !prof !41

19:                                               ; preds = %17
  tail call void @llvm.memmove.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(1) %9, ptr noundef nonnull align 8 dereferenceable(1) %0, i64 %11, i1 false)
  br label %32

20:                                               ; preds = %17
  %21 = getelementptr inbounds nuw i8, ptr %12, i64 8
  store double %15, ptr %21, align 8, !tbaa !32
  br label %32

22:                                               ; preds = %10
  %23 = load double, ptr %12, align 8, !tbaa !32
  %24 = fcmp olt double %14, %23
  br i1 %24, label %25, label %32

25:                                               ; preds = %22, %25
  %26 = phi double [ %30, %25 ], [ %23, %22 ]
  %27 = phi ptr [ %29, %25 ], [ %12, %22 ]
  %28 = phi ptr [ %27, %25 ], [ %13, %22 ]
  store double %26, ptr %28, align 8, !tbaa !32
  %29 = getelementptr inbounds i8, ptr %27, i64 -8
  %30 = load double, ptr %29, align 8, !tbaa !32
  %31 = fcmp olt double %14, %30
  br i1 %31, label %25, label %32, !llvm.loop !101

32:                                               ; preds = %25, %22, %20, %19
  %33 = phi ptr [ %0, %19 ], [ %0, %20 ], [ %13, %22 ], [ %27, %25 ]
  store double %14, ptr %33, align 8, !tbaa !32
  %34 = add nuw nsw i64 %11, 8
  %35 = icmp eq i64 %34, 128
  br i1 %35, label %36, label %10, !llvm.loop !102

36:                                               ; preds = %32
  %37 = getelementptr inbounds nuw i8, ptr %0, i64 128
  %38 = icmp eq ptr %37, %1
  br i1 %38, label %94, label %39

39:                                               ; preds = %36, %52
  %40 = phi ptr [ %54, %52 ], [ %37, %36 ]
  %41 = load double, ptr %40, align 8, !tbaa !32
  %42 = getelementptr inbounds i8, ptr %40, i64 -8
  %43 = load double, ptr %42, align 8, !tbaa !32
  %44 = fcmp olt double %41, %43
  br i1 %44, label %45, label %52

45:                                               ; preds = %39, %45
  %46 = phi double [ %50, %45 ], [ %43, %39 ]
  %47 = phi ptr [ %49, %45 ], [ %42, %39 ]
  %48 = phi ptr [ %47, %45 ], [ %40, %39 ]
  store double %46, ptr %48, align 8, !tbaa !32
  %49 = getelementptr inbounds i8, ptr %47, i64 -8
  %50 = load double, ptr %49, align 8, !tbaa !32
  %51 = fcmp olt double %41, %50
  br i1 %51, label %45, label %52, !llvm.loop !101

52:                                               ; preds = %45, %39
  %53 = phi ptr [ %40, %39 ], [ %47, %45 ]
  store double %41, ptr %53, align 8, !tbaa !32
  %54 = getelementptr inbounds nuw i8, ptr %40, i64 8
  %55 = icmp eq ptr %54, %1
  br i1 %55, label %94, label %39, !llvm.loop !103

56:                                               ; preds = %3
  %57 = icmp eq ptr %0, %1
  %58 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %59 = icmp eq ptr %58, %1
  %60 = select i1 %57, i1 true, i1 %59
  br i1 %60, label %94, label %61

61:                                               ; preds = %56, %90
  %62 = phi ptr [ %92, %90 ], [ %58, %56 ]
  %63 = phi ptr [ %62, %90 ], [ %0, %56 ]
  %64 = load double, ptr %62, align 8, !tbaa !32
  %65 = load double, ptr %0, align 8, !tbaa !32
  %66 = fcmp olt double %64, %65
  br i1 %66, label %67, label %80

67:                                               ; preds = %61
  %68 = ptrtoint ptr %62 to i64
  %69 = sub i64 %68, %5
  %70 = ashr exact i64 %69, 3
  %71 = icmp sgt i64 %70, 1
  br i1 %71, label %72, label %76, !prof !41

72:                                               ; preds = %67
  %73 = getelementptr inbounds nuw i8, ptr %63, i64 16
  %74 = sub nsw i64 0, %70
  %75 = getelementptr inbounds double, ptr %73, i64 %74
  tail call void @llvm.memmove.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(1) %75, ptr noundef nonnull align 8 dereferenceable(1) %0, i64 %69, i1 false)
  br label %90

76:                                               ; preds = %67
  %77 = icmp eq i64 %69, 8
  br i1 %77, label %78, label %90

78:                                               ; preds = %76
  %79 = getelementptr inbounds nuw i8, ptr %63, i64 8
  store double %65, ptr %79, align 8, !tbaa !32
  br label %90

80:                                               ; preds = %61
  %81 = load double, ptr %63, align 8, !tbaa !32
  %82 = fcmp olt double %64, %81
  br i1 %82, label %83, label %90

83:                                               ; preds = %80, %83
  %84 = phi double [ %88, %83 ], [ %81, %80 ]
  %85 = phi ptr [ %87, %83 ], [ %63, %80 ]
  %86 = phi ptr [ %85, %83 ], [ %62, %80 ]
  store double %84, ptr %86, align 8, !tbaa !32
  %87 = getelementptr inbounds i8, ptr %85, i64 -8
  %88 = load double, ptr %87, align 8, !tbaa !32
  %89 = fcmp olt double %64, %88
  br i1 %89, label %83, label %90, !llvm.loop !101

90:                                               ; preds = %83, %80, %78, %76, %72
  %91 = phi ptr [ %0, %72 ], [ %0, %76 ], [ %0, %78 ], [ %62, %80 ], [ %85, %83 ]
  store double %64, ptr %91, align 8, !tbaa !32
  %92 = getelementptr inbounds nuw i8, ptr %62, i64 8
  %93 = icmp eq ptr %92, %1
  br i1 %93, label %94, label %61, !llvm.loop !102

94:                                               ; preds = %90, %52, %56, %36
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_comp_iterI24inline_less_than_functorEEEvT_S6_RT0_(ptr noundef %0, ptr noundef %1, ptr noundef nonnull align 1 dereferenceable(1) %2) local_unnamed_addr #11 comdat {
  %4 = ptrtoint ptr %1 to i64
  %5 = ptrtoint ptr %0 to i64
  %6 = sub i64 %4, %5
  %7 = ashr exact i64 %6, 3
  %8 = icmp slt i64 %7, 2
  br i1 %8, label %103, label %9

9:                                                ; preds = %3
  %10 = add nsw i64 %7, -2
  %11 = lshr i64 %10, 1
  %12 = add nsw i64 %7, -1
  %13 = lshr i64 %12, 1
  %14 = and i64 %6, 8
  %15 = icmp eq i64 %14, 0
  %16 = lshr exact i64 %10, 1
  br i1 %15, label %17, label %21

17:                                               ; preds = %9
  %18 = or disjoint i64 %10, 1
  %19 = getelementptr inbounds nuw double, ptr %0, i64 %18
  %20 = getelementptr inbounds nuw double, ptr %0, i64 %16
  br label %59

21:                                               ; preds = %9, %54
  %22 = phi i64 [ %58, %54 ], [ %11, %9 ]
  %23 = getelementptr inbounds nuw double, ptr %0, i64 %22
  %24 = load double, ptr %23, align 8, !tbaa !32
  %25 = icmp slt i64 %22, %13
  br i1 %25, label %26, label %54

26:                                               ; preds = %21, %26
  %27 = phi i64 [ %37, %26 ], [ %22, %21 ]
  %28 = shl i64 %27, 1
  %29 = add i64 %28, 2
  %30 = getelementptr inbounds double, ptr %0, i64 %29
  %31 = getelementptr double, ptr %0, i64 %28
  %32 = getelementptr i8, ptr %31, i64 8
  %33 = load double, ptr %30, align 8, !tbaa !32
  %34 = load double, ptr %32, align 8, !tbaa !32
  %35 = fcmp olt double %33, %34
  %36 = or disjoint i64 %28, 1
  %37 = select i1 %35, i64 %36, i64 %29
  %38 = getelementptr inbounds double, ptr %0, i64 %37
  %39 = load double, ptr %38, align 8, !tbaa !32
  %40 = getelementptr inbounds double, ptr %0, i64 %27
  store double %39, ptr %40, align 8, !tbaa !32
  %41 = icmp slt i64 %37, %13
  br i1 %41, label %26, label %42, !llvm.loop !94

42:                                               ; preds = %26
  %43 = icmp sgt i64 %37, %22
  br i1 %43, label %44, label %54

44:                                               ; preds = %42, %51
  %45 = phi i64 [ %47, %51 ], [ %37, %42 ]
  %46 = add nsw i64 %45, -1
  %47 = sdiv i64 %46, 2
  %48 = getelementptr inbounds nuw double, ptr %0, i64 %47
  %49 = load double, ptr %48, align 8, !tbaa !32
  %50 = fcmp olt double %49, %24
  br i1 %50, label %51, label %54

51:                                               ; preds = %44
  %52 = getelementptr inbounds nuw double, ptr %0, i64 %45
  store double %49, ptr %52, align 8, !tbaa !32
  %53 = icmp sgt i64 %47, %22
  br i1 %53, label %44, label %54, !llvm.loop !95

54:                                               ; preds = %44, %51, %21, %42
  %55 = phi i64 [ %37, %42 ], [ %22, %21 ], [ %47, %51 ], [ %45, %44 ]
  %56 = getelementptr inbounds nuw double, ptr %0, i64 %55
  store double %24, ptr %56, align 8, !tbaa !32
  %57 = icmp eq i64 %22, 0
  %58 = add nsw i64 %22, -1
  br i1 %57, label %103, label %21, !llvm.loop !104

59:                                               ; preds = %17, %98
  %60 = phi i64 [ %102, %98 ], [ %11, %17 ]
  %61 = getelementptr inbounds nuw double, ptr %0, i64 %60
  %62 = load double, ptr %61, align 8, !tbaa !32
  %63 = icmp slt i64 %60, %13
  br i1 %63, label %64, label %80

64:                                               ; preds = %59, %64
  %65 = phi i64 [ %75, %64 ], [ %60, %59 ]
  %66 = shl i64 %65, 1
  %67 = add i64 %66, 2
  %68 = getelementptr inbounds double, ptr %0, i64 %67
  %69 = getelementptr double, ptr %0, i64 %66
  %70 = getelementptr i8, ptr %69, i64 8
  %71 = load double, ptr %68, align 8, !tbaa !32
  %72 = load double, ptr %70, align 8, !tbaa !32
  %73 = fcmp olt double %71, %72
  %74 = or disjoint i64 %66, 1
  %75 = select i1 %73, i64 %74, i64 %67
  %76 = getelementptr inbounds double, ptr %0, i64 %75
  %77 = load double, ptr %76, align 8, !tbaa !32
  %78 = getelementptr inbounds double, ptr %0, i64 %65
  store double %77, ptr %78, align 8, !tbaa !32
  %79 = icmp slt i64 %75, %13
  br i1 %79, label %64, label %80, !llvm.loop !94

80:                                               ; preds = %64, %59
  %81 = phi i64 [ %60, %59 ], [ %75, %64 ]
  %82 = icmp eq i64 %81, %16
  br i1 %82, label %83, label %85

83:                                               ; preds = %80
  %84 = load double, ptr %19, align 8, !tbaa !32
  store double %84, ptr %20, align 8, !tbaa !32
  br label %85

85:                                               ; preds = %83, %80
  %86 = phi i64 [ %18, %83 ], [ %81, %80 ]
  %87 = icmp sgt i64 %86, %60
  br i1 %87, label %88, label %98

88:                                               ; preds = %85, %95
  %89 = phi i64 [ %91, %95 ], [ %86, %85 ]
  %90 = add nsw i64 %89, -1
  %91 = sdiv i64 %90, 2
  %92 = getelementptr inbounds nuw double, ptr %0, i64 %91
  %93 = load double, ptr %92, align 8, !tbaa !32
  %94 = fcmp olt double %93, %62
  br i1 %94, label %95, label %98

95:                                               ; preds = %88
  %96 = getelementptr inbounds nuw double, ptr %0, i64 %89
  store double %93, ptr %96, align 8, !tbaa !32
  %97 = icmp sgt i64 %91, %60
  br i1 %97, label %88, label %98, !llvm.loop !95

98:                                               ; preds = %88, %95, %85
  %99 = phi i64 [ %86, %85 ], [ %91, %95 ], [ %89, %88 ]
  %100 = getelementptr inbounds nuw double, ptr %0, i64 %99
  store double %62, ptr %100, align 8, !tbaa !32
  %101 = icmp eq i64 %60, 0
  %102 = add nsw i64 %60, -1
  br i1 %101, label %103, label %59, !llvm.loop !104

103:                                              ; preds = %54, %98, %3
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterISt4lessIdEEEEvT_S7_T0_T1_(ptr noundef %0, ptr noundef %1, i64 noundef %2, i64 %3) local_unnamed_addr #11 comdat {
  %5 = alloca %"struct.__gnu_cxx::__ops::_Iter_comp_iter.6", align 4
  %6 = ptrtoint ptr %0 to i64
  %7 = ptrtoint ptr %1 to i64
  %8 = sub i64 %7, %6
  %9 = icmp sgt i64 %8, 128
  br i1 %9, label %10, label %128

10:                                               ; preds = %4
  %11 = and i64 %3, 255
  %12 = getelementptr inbounds nuw i8, ptr %0, i64 8
  br label %13

13:                                               ; preds = %10, %124
  %14 = phi i64 [ %8, %10 ], [ %126, %124 ]
  %15 = phi ptr [ %1, %10 ], [ %112, %124 ]
  %16 = phi i64 [ %2, %10 ], [ %80, %124 ]
  %17 = icmp eq i64 %16, 0
  br i1 %17, label %18, label %79

18:                                               ; preds = %13
  call void @llvm.lifetime.start.p0(ptr nonnull %5)
  %19 = trunc i64 %3 to i8
  store i8 %19, ptr %5, align 4
  call void @_ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_comp_iterISt4lessIdEEEEvT_S7_RT0_(ptr noundef %0, ptr noundef %15, ptr noundef nonnull align 1 dereferenceable(1) %5)
  call void @llvm.lifetime.end.p0(ptr nonnull %5)
  br label %20

20:                                               ; preds = %18, %75
  %21 = phi ptr [ %22, %75 ], [ %15, %18 ]
  %22 = getelementptr inbounds i8, ptr %21, i64 -8
  %23 = load double, ptr %22, align 8, !tbaa !32
  %24 = load double, ptr %0, align 8, !tbaa !32
  store double %24, ptr %22, align 8, !tbaa !32
  %25 = ptrtoint ptr %22 to i64
  %26 = sub i64 %25, %6
  %27 = ashr exact i64 %26, 3
  %28 = add nsw i64 %27, -1
  %29 = sdiv i64 %28, 2
  %30 = icmp sgt i64 %27, 2
  br i1 %30, label %31, label %47

31:                                               ; preds = %20, %31
  %32 = phi i64 [ %42, %31 ], [ 0, %20 ]
  %33 = shl i64 %32, 1
  %34 = add i64 %33, 2
  %35 = getelementptr inbounds double, ptr %0, i64 %34
  %36 = getelementptr double, ptr %0, i64 %33
  %37 = getelementptr i8, ptr %36, i64 8
  %38 = load double, ptr %35, align 8, !tbaa !32
  %39 = load double, ptr %37, align 8, !tbaa !32
  %40 = fcmp olt double %38, %39
  %41 = or disjoint i64 %33, 1
  %42 = select i1 %40, i64 %41, i64 %34
  %43 = getelementptr inbounds double, ptr %0, i64 %42
  %44 = load double, ptr %43, align 8, !tbaa !32
  %45 = getelementptr inbounds double, ptr %0, i64 %32
  store double %44, ptr %45, align 8, !tbaa !32
  %46 = icmp slt i64 %42, %29
  br i1 %46, label %31, label %47, !llvm.loop !105

47:                                               ; preds = %31, %20
  %48 = phi i64 [ 0, %20 ], [ %42, %31 ]
  %49 = and i64 %26, 8
  %50 = icmp eq i64 %49, 0
  br i1 %50, label %51, label %61

51:                                               ; preds = %47
  %52 = add nsw i64 %27, -2
  %53 = ashr exact i64 %52, 1
  %54 = icmp eq i64 %48, %53
  br i1 %54, label %55, label %61

55:                                               ; preds = %51
  %56 = shl nuw nsw i64 %48, 1
  %57 = or disjoint i64 %56, 1
  %58 = getelementptr inbounds nuw double, ptr %0, i64 %57
  %59 = load double, ptr %58, align 8, !tbaa !32
  %60 = getelementptr inbounds double, ptr %0, i64 %48
  store double %59, ptr %60, align 8, !tbaa !32
  br label %63

61:                                               ; preds = %51, %47
  %62 = icmp eq i64 %48, 0
  br i1 %62, label %75, label %63

63:                                               ; preds = %61, %55
  %64 = phi i64 [ %48, %61 ], [ %57, %55 ]
  br label %65

65:                                               ; preds = %63, %72
  %66 = phi i64 [ %68, %72 ], [ %64, %63 ]
  %67 = add nsw i64 %66, -1
  %68 = lshr i64 %67, 1
  %69 = getelementptr inbounds nuw double, ptr %0, i64 %68
  %70 = load double, ptr %69, align 8, !tbaa !32
  %71 = fcmp olt double %70, %23
  br i1 %71, label %72, label %75

72:                                               ; preds = %65
  %73 = getelementptr inbounds double, ptr %0, i64 %66
  store double %70, ptr %73, align 8, !tbaa !32
  %74 = icmp ult i64 %67, 2
  br i1 %74, label %75, label %65, !llvm.loop !106

75:                                               ; preds = %72, %65, %61
  %76 = phi i64 [ 0, %61 ], [ %66, %65 ], [ 0, %72 ]
  %77 = getelementptr inbounds double, ptr %0, i64 %76
  store double %23, ptr %77, align 8, !tbaa !32
  %78 = icmp sgt i64 %26, 8
  br i1 %78, label %20, label %128, !llvm.loop !107

79:                                               ; preds = %13
  %80 = add nsw i64 %16, -1
  %81 = lshr i64 %14, 4
  %82 = getelementptr inbounds nuw double, ptr %0, i64 %81
  %83 = getelementptr inbounds i8, ptr %15, i64 -8
  %84 = load double, ptr %12, align 8, !tbaa !32
  %85 = load double, ptr %82, align 8, !tbaa !32
  %86 = fcmp olt double %84, %85
  %87 = load double, ptr %83, align 8, !tbaa !32
  br i1 %86, label %88, label %97

88:                                               ; preds = %79
  %89 = fcmp olt double %85, %87
  br i1 %89, label %90, label %92

90:                                               ; preds = %88
  %91 = load double, ptr %0, align 8, !tbaa !32
  store double %85, ptr %0, align 8, !tbaa !32
  store double %91, ptr %82, align 8, !tbaa !32
  br label %106

92:                                               ; preds = %88
  %93 = fcmp olt double %84, %87
  %94 = load double, ptr %0, align 8, !tbaa !32
  br i1 %93, label %95, label %96

95:                                               ; preds = %92
  store double %87, ptr %0, align 8, !tbaa !32
  store double %94, ptr %83, align 8, !tbaa !32
  br label %106

96:                                               ; preds = %92
  store double %84, ptr %0, align 8, !tbaa !32
  store double %94, ptr %12, align 8, !tbaa !32
  br label %106

97:                                               ; preds = %79
  %98 = fcmp olt double %84, %87
  br i1 %98, label %99, label %101

99:                                               ; preds = %97
  %100 = load double, ptr %0, align 8, !tbaa !32
  store double %84, ptr %0, align 8, !tbaa !32
  store double %100, ptr %12, align 8, !tbaa !32
  br label %106

101:                                              ; preds = %97
  %102 = fcmp olt double %85, %87
  %103 = load double, ptr %0, align 8, !tbaa !32
  br i1 %102, label %104, label %105

104:                                              ; preds = %101
  store double %87, ptr %0, align 8, !tbaa !32
  store double %103, ptr %83, align 8, !tbaa !32
  br label %106

105:                                              ; preds = %101
  store double %85, ptr %0, align 8, !tbaa !32
  store double %103, ptr %82, align 8, !tbaa !32
  br label %106

106:                                              ; preds = %105, %104, %99, %96, %95, %90
  br label %107

107:                                              ; preds = %106, %123
  %108 = phi ptr [ %118, %123 ], [ %15, %106 ]
  %109 = phi ptr [ %115, %123 ], [ %12, %106 ]
  %110 = load double, ptr %0, align 8, !tbaa !32
  br label %111

111:                                              ; preds = %111, %107
  %112 = phi ptr [ %109, %107 ], [ %115, %111 ]
  %113 = load double, ptr %112, align 8, !tbaa !32
  %114 = fcmp olt double %113, %110
  %115 = getelementptr inbounds nuw i8, ptr %112, i64 8
  br i1 %114, label %111, label %116, !llvm.loop !108

116:                                              ; preds = %111, %116
  %117 = phi ptr [ %118, %116 ], [ %108, %111 ]
  %118 = getelementptr inbounds i8, ptr %117, i64 -8
  %119 = load double, ptr %118, align 8, !tbaa !32
  %120 = fcmp olt double %110, %119
  br i1 %120, label %116, label %121, !llvm.loop !109

121:                                              ; preds = %116
  %122 = icmp ult ptr %112, %118
  br i1 %122, label %123, label %124

123:                                              ; preds = %121
  store double %119, ptr %112, align 8, !tbaa !32
  store double %113, ptr %118, align 8, !tbaa !32
  br label %107, !llvm.loop !110

124:                                              ; preds = %121
  tail call void @_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterISt4lessIdEEEEvT_S7_T0_T1_(ptr noundef nonnull %112, ptr noundef %15, i64 noundef %80, i64 %11)
  %125 = ptrtoint ptr %112 to i64
  %126 = sub i64 %125, %6
  %127 = icmp sgt i64 %126, 128
  br i1 %127, label %13, label %128, !llvm.loop !111

128:                                              ; preds = %124, %75, %4
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZSt22__final_insertion_sortIPdN9__gnu_cxx5__ops15_Iter_comp_iterISt4lessIdEEEEvT_S7_T0_(ptr noundef %0, ptr noundef %1, i64 %2) local_unnamed_addr #11 comdat {
  %4 = ptrtoint ptr %1 to i64
  %5 = ptrtoint ptr %0 to i64
  %6 = sub i64 %4, %5
  %7 = icmp sgt i64 %6, 128
  br i1 %7, label %8, label %56

8:                                                ; preds = %3
  %9 = getelementptr i8, ptr %0, i64 8
  br label %10

10:                                               ; preds = %32, %8
  %11 = phi i64 [ 8, %8 ], [ %34, %32 ]
  %12 = phi ptr [ %0, %8 ], [ %13, %32 ]
  %13 = getelementptr inbounds nuw i8, ptr %0, i64 %11
  %14 = load double, ptr %13, align 8, !tbaa !32
  %15 = load double, ptr %0, align 8, !tbaa !32
  %16 = fcmp olt double %14, %15
  br i1 %16, label %17, label %22

17:                                               ; preds = %10
  %18 = icmp samesign ugt i64 %11, 8
  br i1 %18, label %19, label %20, !prof !41

19:                                               ; preds = %17
  tail call void @llvm.memmove.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(1) %9, ptr noundef nonnull align 8 dereferenceable(1) %0, i64 %11, i1 false)
  br label %32

20:                                               ; preds = %17
  %21 = getelementptr inbounds nuw i8, ptr %12, i64 8
  store double %15, ptr %21, align 8, !tbaa !32
  br label %32

22:                                               ; preds = %10
  %23 = load double, ptr %12, align 8, !tbaa !32
  %24 = fcmp olt double %14, %23
  br i1 %24, label %25, label %32

25:                                               ; preds = %22, %25
  %26 = phi double [ %30, %25 ], [ %23, %22 ]
  %27 = phi ptr [ %29, %25 ], [ %12, %22 ]
  %28 = phi ptr [ %27, %25 ], [ %13, %22 ]
  store double %26, ptr %28, align 8, !tbaa !32
  %29 = getelementptr inbounds i8, ptr %27, i64 -8
  %30 = load double, ptr %29, align 8, !tbaa !32
  %31 = fcmp olt double %14, %30
  br i1 %31, label %25, label %32, !llvm.loop !112

32:                                               ; preds = %25, %22, %20, %19
  %33 = phi ptr [ %0, %19 ], [ %0, %20 ], [ %13, %22 ], [ %27, %25 ]
  store double %14, ptr %33, align 8, !tbaa !32
  %34 = add nuw nsw i64 %11, 8
  %35 = icmp eq i64 %34, 128
  br i1 %35, label %36, label %10, !llvm.loop !113

36:                                               ; preds = %32
  %37 = getelementptr inbounds nuw i8, ptr %0, i64 128
  %38 = icmp eq ptr %37, %1
  br i1 %38, label %94, label %39

39:                                               ; preds = %36, %52
  %40 = phi ptr [ %54, %52 ], [ %37, %36 ]
  %41 = load double, ptr %40, align 8, !tbaa !32
  %42 = getelementptr inbounds i8, ptr %40, i64 -8
  %43 = load double, ptr %42, align 8, !tbaa !32
  %44 = fcmp olt double %41, %43
  br i1 %44, label %45, label %52

45:                                               ; preds = %39, %45
  %46 = phi double [ %50, %45 ], [ %43, %39 ]
  %47 = phi ptr [ %49, %45 ], [ %42, %39 ]
  %48 = phi ptr [ %47, %45 ], [ %40, %39 ]
  store double %46, ptr %48, align 8, !tbaa !32
  %49 = getelementptr inbounds i8, ptr %47, i64 -8
  %50 = load double, ptr %49, align 8, !tbaa !32
  %51 = fcmp olt double %41, %50
  br i1 %51, label %45, label %52, !llvm.loop !112

52:                                               ; preds = %45, %39
  %53 = phi ptr [ %40, %39 ], [ %47, %45 ]
  store double %41, ptr %53, align 8, !tbaa !32
  %54 = getelementptr inbounds nuw i8, ptr %40, i64 8
  %55 = icmp eq ptr %54, %1
  br i1 %55, label %94, label %39, !llvm.loop !114

56:                                               ; preds = %3
  %57 = icmp eq ptr %0, %1
  %58 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %59 = icmp eq ptr %58, %1
  %60 = select i1 %57, i1 true, i1 %59
  br i1 %60, label %94, label %61

61:                                               ; preds = %56, %90
  %62 = phi ptr [ %92, %90 ], [ %58, %56 ]
  %63 = phi ptr [ %62, %90 ], [ %0, %56 ]
  %64 = load double, ptr %62, align 8, !tbaa !32
  %65 = load double, ptr %0, align 8, !tbaa !32
  %66 = fcmp olt double %64, %65
  br i1 %66, label %67, label %80

67:                                               ; preds = %61
  %68 = ptrtoint ptr %62 to i64
  %69 = sub i64 %68, %5
  %70 = ashr exact i64 %69, 3
  %71 = icmp sgt i64 %70, 1
  br i1 %71, label %72, label %76, !prof !41

72:                                               ; preds = %67
  %73 = getelementptr inbounds nuw i8, ptr %63, i64 16
  %74 = sub nsw i64 0, %70
  %75 = getelementptr inbounds double, ptr %73, i64 %74
  tail call void @llvm.memmove.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(1) %75, ptr noundef nonnull align 8 dereferenceable(1) %0, i64 %69, i1 false)
  br label %90

76:                                               ; preds = %67
  %77 = icmp eq i64 %69, 8
  br i1 %77, label %78, label %90

78:                                               ; preds = %76
  %79 = getelementptr inbounds nuw i8, ptr %63, i64 8
  store double %65, ptr %79, align 8, !tbaa !32
  br label %90

80:                                               ; preds = %61
  %81 = load double, ptr %63, align 8, !tbaa !32
  %82 = fcmp olt double %64, %81
  br i1 %82, label %83, label %90

83:                                               ; preds = %80, %83
  %84 = phi double [ %88, %83 ], [ %81, %80 ]
  %85 = phi ptr [ %87, %83 ], [ %63, %80 ]
  %86 = phi ptr [ %85, %83 ], [ %62, %80 ]
  store double %84, ptr %86, align 8, !tbaa !32
  %87 = getelementptr inbounds i8, ptr %85, i64 -8
  %88 = load double, ptr %87, align 8, !tbaa !32
  %89 = fcmp olt double %64, %88
  br i1 %89, label %83, label %90, !llvm.loop !112

90:                                               ; preds = %83, %80, %78, %76, %72
  %91 = phi ptr [ %0, %72 ], [ %0, %76 ], [ %0, %78 ], [ %62, %80 ], [ %85, %83 ]
  store double %64, ptr %91, align 8, !tbaa !32
  %92 = getelementptr inbounds nuw i8, ptr %62, i64 8
  %93 = icmp eq ptr %92, %1
  br i1 %93, label %94, label %61, !llvm.loop !113

94:                                               ; preds = %90, %52, %56, %36
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_comp_iterISt4lessIdEEEEvT_S7_RT0_(ptr noundef %0, ptr noundef %1, ptr noundef nonnull align 1 dereferenceable(1) %2) local_unnamed_addr #11 comdat {
  %4 = ptrtoint ptr %1 to i64
  %5 = ptrtoint ptr %0 to i64
  %6 = sub i64 %4, %5
  %7 = ashr exact i64 %6, 3
  %8 = icmp slt i64 %7, 2
  br i1 %8, label %103, label %9

9:                                                ; preds = %3
  %10 = add nsw i64 %7, -2
  %11 = lshr i64 %10, 1
  %12 = add nsw i64 %7, -1
  %13 = lshr i64 %12, 1
  %14 = and i64 %6, 8
  %15 = icmp eq i64 %14, 0
  %16 = lshr exact i64 %10, 1
  br i1 %15, label %17, label %21

17:                                               ; preds = %9
  %18 = or disjoint i64 %10, 1
  %19 = getelementptr inbounds nuw double, ptr %0, i64 %18
  %20 = getelementptr inbounds nuw double, ptr %0, i64 %16
  br label %59

21:                                               ; preds = %9, %54
  %22 = phi i64 [ %58, %54 ], [ %11, %9 ]
  %23 = getelementptr inbounds nuw double, ptr %0, i64 %22
  %24 = load double, ptr %23, align 8, !tbaa !32
  %25 = icmp slt i64 %22, %13
  br i1 %25, label %26, label %54

26:                                               ; preds = %21, %26
  %27 = phi i64 [ %37, %26 ], [ %22, %21 ]
  %28 = shl i64 %27, 1
  %29 = add i64 %28, 2
  %30 = getelementptr inbounds double, ptr %0, i64 %29
  %31 = getelementptr double, ptr %0, i64 %28
  %32 = getelementptr i8, ptr %31, i64 8
  %33 = load double, ptr %30, align 8, !tbaa !32
  %34 = load double, ptr %32, align 8, !tbaa !32
  %35 = fcmp olt double %33, %34
  %36 = or disjoint i64 %28, 1
  %37 = select i1 %35, i64 %36, i64 %29
  %38 = getelementptr inbounds double, ptr %0, i64 %37
  %39 = load double, ptr %38, align 8, !tbaa !32
  %40 = getelementptr inbounds double, ptr %0, i64 %27
  store double %39, ptr %40, align 8, !tbaa !32
  %41 = icmp slt i64 %37, %13
  br i1 %41, label %26, label %42, !llvm.loop !105

42:                                               ; preds = %26
  %43 = icmp sgt i64 %37, %22
  br i1 %43, label %44, label %54

44:                                               ; preds = %42, %51
  %45 = phi i64 [ %47, %51 ], [ %37, %42 ]
  %46 = add nsw i64 %45, -1
  %47 = sdiv i64 %46, 2
  %48 = getelementptr inbounds nuw double, ptr %0, i64 %47
  %49 = load double, ptr %48, align 8, !tbaa !32
  %50 = fcmp olt double %49, %24
  br i1 %50, label %51, label %54

51:                                               ; preds = %44
  %52 = getelementptr inbounds nuw double, ptr %0, i64 %45
  store double %49, ptr %52, align 8, !tbaa !32
  %53 = icmp sgt i64 %47, %22
  br i1 %53, label %44, label %54, !llvm.loop !106

54:                                               ; preds = %44, %51, %21, %42
  %55 = phi i64 [ %37, %42 ], [ %22, %21 ], [ %47, %51 ], [ %45, %44 ]
  %56 = getelementptr inbounds nuw double, ptr %0, i64 %55
  store double %24, ptr %56, align 8, !tbaa !32
  %57 = icmp eq i64 %22, 0
  %58 = add nsw i64 %22, -1
  br i1 %57, label %103, label %21, !llvm.loop !115

59:                                               ; preds = %17, %98
  %60 = phi i64 [ %102, %98 ], [ %11, %17 ]
  %61 = getelementptr inbounds nuw double, ptr %0, i64 %60
  %62 = load double, ptr %61, align 8, !tbaa !32
  %63 = icmp slt i64 %60, %13
  br i1 %63, label %64, label %80

64:                                               ; preds = %59, %64
  %65 = phi i64 [ %75, %64 ], [ %60, %59 ]
  %66 = shl i64 %65, 1
  %67 = add i64 %66, 2
  %68 = getelementptr inbounds double, ptr %0, i64 %67
  %69 = getelementptr double, ptr %0, i64 %66
  %70 = getelementptr i8, ptr %69, i64 8
  %71 = load double, ptr %68, align 8, !tbaa !32
  %72 = load double, ptr %70, align 8, !tbaa !32
  %73 = fcmp olt double %71, %72
  %74 = or disjoint i64 %66, 1
  %75 = select i1 %73, i64 %74, i64 %67
  %76 = getelementptr inbounds double, ptr %0, i64 %75
  %77 = load double, ptr %76, align 8, !tbaa !32
  %78 = getelementptr inbounds double, ptr %0, i64 %65
  store double %77, ptr %78, align 8, !tbaa !32
  %79 = icmp slt i64 %75, %13
  br i1 %79, label %64, label %80, !llvm.loop !105

80:                                               ; preds = %64, %59
  %81 = phi i64 [ %60, %59 ], [ %75, %64 ]
  %82 = icmp eq i64 %81, %16
  br i1 %82, label %83, label %85

83:                                               ; preds = %80
  %84 = load double, ptr %19, align 8, !tbaa !32
  store double %84, ptr %20, align 8, !tbaa !32
  br label %85

85:                                               ; preds = %83, %80
  %86 = phi i64 [ %18, %83 ], [ %81, %80 ]
  %87 = icmp sgt i64 %86, %60
  br i1 %87, label %88, label %98

88:                                               ; preds = %85, %95
  %89 = phi i64 [ %91, %95 ], [ %86, %85 ]
  %90 = add nsw i64 %89, -1
  %91 = sdiv i64 %90, 2
  %92 = getelementptr inbounds nuw double, ptr %0, i64 %91
  %93 = load double, ptr %92, align 8, !tbaa !32
  %94 = fcmp olt double %93, %62
  br i1 %94, label %95, label %98

95:                                               ; preds = %88
  %96 = getelementptr inbounds nuw double, ptr %0, i64 %89
  store double %93, ptr %96, align 8, !tbaa !32
  %97 = icmp sgt i64 %91, %60
  br i1 %97, label %88, label %98, !llvm.loop !106

98:                                               ; preds = %88, %95, %85
  %99 = phi i64 [ %86, %85 ], [ %91, %95 ], [ %89, %88 ]
  %100 = getelementptr inbounds nuw double, ptr %0, i64 %99
  store double %62, ptr %100, align 8, !tbaa !32
  %101 = icmp eq i64 %60, 0
  %102 = add nsw i64 %60, -1
  br i1 %101, label %103, label %59, !llvm.loop !115

103:                                              ; preds = %54, %98, %3
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_(ptr noundef %0, ptr noundef %1, i64 noundef %2, i8 %3) local_unnamed_addr #11 comdat {
  %5 = alloca %"struct.__gnu_cxx::__ops::_Iter_less_iter", align 1
  %6 = ptrtoint ptr %0 to i64
  %7 = ptrtoint ptr %1 to i64
  %8 = sub i64 %7, %6
  %9 = icmp sgt i64 %8, 128
  br i1 %9, label %10, label %126

10:                                               ; preds = %4
  %11 = getelementptr inbounds nuw i8, ptr %0, i64 8
  br label %12

12:                                               ; preds = %10, %122
  %13 = phi i64 [ %8, %10 ], [ %124, %122 ]
  %14 = phi ptr [ %1, %10 ], [ %110, %122 ]
  %15 = phi i64 [ %2, %10 ], [ %78, %122 ]
  %16 = icmp eq i64 %15, 0
  br i1 %16, label %17, label %77

17:                                               ; preds = %12
  call void @llvm.lifetime.start.p0(ptr nonnull %5)
  call void @_ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_RT0_(ptr noundef %0, ptr noundef %14, ptr noundef nonnull align 1 dereferenceable(1) %5)
  call void @llvm.lifetime.end.p0(ptr nonnull %5)
  br label %18

18:                                               ; preds = %17, %73
  %19 = phi ptr [ %20, %73 ], [ %14, %17 ]
  %20 = getelementptr inbounds i8, ptr %19, i64 -8
  %21 = load double, ptr %20, align 8, !tbaa !32
  %22 = load double, ptr %0, align 8, !tbaa !32
  store double %22, ptr %20, align 8, !tbaa !32
  %23 = ptrtoint ptr %20 to i64
  %24 = sub i64 %23, %6
  %25 = ashr exact i64 %24, 3
  %26 = add nsw i64 %25, -1
  %27 = sdiv i64 %26, 2
  %28 = icmp sgt i64 %25, 2
  br i1 %28, label %29, label %45

29:                                               ; preds = %18, %29
  %30 = phi i64 [ %40, %29 ], [ 0, %18 ]
  %31 = shl i64 %30, 1
  %32 = add i64 %31, 2
  %33 = getelementptr inbounds double, ptr %0, i64 %32
  %34 = getelementptr double, ptr %0, i64 %31
  %35 = getelementptr i8, ptr %34, i64 8
  %36 = load double, ptr %33, align 8, !tbaa !32
  %37 = load double, ptr %35, align 8, !tbaa !32
  %38 = fcmp olt double %36, %37
  %39 = or disjoint i64 %31, 1
  %40 = select i1 %38, i64 %39, i64 %32
  %41 = getelementptr inbounds double, ptr %0, i64 %40
  %42 = load double, ptr %41, align 8, !tbaa !32
  %43 = getelementptr inbounds double, ptr %0, i64 %30
  store double %42, ptr %43, align 8, !tbaa !32
  %44 = icmp slt i64 %40, %27
  br i1 %44, label %29, label %45, !llvm.loop !116

45:                                               ; preds = %29, %18
  %46 = phi i64 [ 0, %18 ], [ %40, %29 ]
  %47 = and i64 %24, 8
  %48 = icmp eq i64 %47, 0
  br i1 %48, label %49, label %59

49:                                               ; preds = %45
  %50 = add nsw i64 %25, -2
  %51 = ashr exact i64 %50, 1
  %52 = icmp eq i64 %46, %51
  br i1 %52, label %53, label %59

53:                                               ; preds = %49
  %54 = shl nuw nsw i64 %46, 1
  %55 = or disjoint i64 %54, 1
  %56 = getelementptr inbounds nuw double, ptr %0, i64 %55
  %57 = load double, ptr %56, align 8, !tbaa !32
  %58 = getelementptr inbounds double, ptr %0, i64 %46
  store double %57, ptr %58, align 8, !tbaa !32
  br label %61

59:                                               ; preds = %49, %45
  %60 = icmp eq i64 %46, 0
  br i1 %60, label %73, label %61

61:                                               ; preds = %59, %53
  %62 = phi i64 [ %46, %59 ], [ %55, %53 ]
  br label %63

63:                                               ; preds = %61, %70
  %64 = phi i64 [ %66, %70 ], [ %62, %61 ]
  %65 = add nsw i64 %64, -1
  %66 = lshr i64 %65, 1
  %67 = getelementptr inbounds nuw double, ptr %0, i64 %66
  %68 = load double, ptr %67, align 8, !tbaa !32
  %69 = fcmp olt double %68, %21
  br i1 %69, label %70, label %73

70:                                               ; preds = %63
  %71 = getelementptr inbounds double, ptr %0, i64 %64
  store double %68, ptr %71, align 8, !tbaa !32
  %72 = icmp ult i64 %65, 2
  br i1 %72, label %73, label %63, !llvm.loop !117

73:                                               ; preds = %70, %63, %59
  %74 = phi i64 [ 0, %59 ], [ %64, %63 ], [ 0, %70 ]
  %75 = getelementptr inbounds double, ptr %0, i64 %74
  store double %21, ptr %75, align 8, !tbaa !32
  %76 = icmp sgt i64 %24, 8
  br i1 %76, label %18, label %126, !llvm.loop !118

77:                                               ; preds = %12
  %78 = add nsw i64 %15, -1
  %79 = lshr i64 %13, 4
  %80 = getelementptr inbounds nuw double, ptr %0, i64 %79
  %81 = getelementptr inbounds i8, ptr %14, i64 -8
  %82 = load double, ptr %11, align 8, !tbaa !32
  %83 = load double, ptr %80, align 8, !tbaa !32
  %84 = fcmp olt double %82, %83
  %85 = load double, ptr %81, align 8, !tbaa !32
  br i1 %84, label %86, label %95

86:                                               ; preds = %77
  %87 = fcmp olt double %83, %85
  br i1 %87, label %88, label %90

88:                                               ; preds = %86
  %89 = load double, ptr %0, align 8, !tbaa !32
  store double %83, ptr %0, align 8, !tbaa !32
  store double %89, ptr %80, align 8, !tbaa !32
  br label %104

90:                                               ; preds = %86
  %91 = fcmp olt double %82, %85
  %92 = load double, ptr %0, align 8, !tbaa !32
  br i1 %91, label %93, label %94

93:                                               ; preds = %90
  store double %85, ptr %0, align 8, !tbaa !32
  store double %92, ptr %81, align 8, !tbaa !32
  br label %104

94:                                               ; preds = %90
  store double %82, ptr %0, align 8, !tbaa !32
  store double %92, ptr %11, align 8, !tbaa !32
  br label %104

95:                                               ; preds = %77
  %96 = fcmp olt double %82, %85
  br i1 %96, label %97, label %99

97:                                               ; preds = %95
  %98 = load double, ptr %0, align 8, !tbaa !32
  store double %82, ptr %0, align 8, !tbaa !32
  store double %98, ptr %11, align 8, !tbaa !32
  br label %104

99:                                               ; preds = %95
  %100 = fcmp olt double %83, %85
  %101 = load double, ptr %0, align 8, !tbaa !32
  br i1 %100, label %102, label %103

102:                                              ; preds = %99
  store double %85, ptr %0, align 8, !tbaa !32
  store double %101, ptr %81, align 8, !tbaa !32
  br label %104

103:                                              ; preds = %99
  store double %83, ptr %0, align 8, !tbaa !32
  store double %101, ptr %80, align 8, !tbaa !32
  br label %104

104:                                              ; preds = %103, %102, %97, %94, %93, %88
  br label %105

105:                                              ; preds = %104, %121
  %106 = phi ptr [ %116, %121 ], [ %14, %104 ]
  %107 = phi ptr [ %113, %121 ], [ %11, %104 ]
  %108 = load double, ptr %0, align 8, !tbaa !32
  br label %109

109:                                              ; preds = %109, %105
  %110 = phi ptr [ %107, %105 ], [ %113, %109 ]
  %111 = load double, ptr %110, align 8, !tbaa !32
  %112 = fcmp olt double %111, %108
  %113 = getelementptr inbounds nuw i8, ptr %110, i64 8
  br i1 %112, label %109, label %114, !llvm.loop !119

114:                                              ; preds = %109, %114
  %115 = phi ptr [ %116, %114 ], [ %106, %109 ]
  %116 = getelementptr inbounds i8, ptr %115, i64 -8
  %117 = load double, ptr %116, align 8, !tbaa !32
  %118 = fcmp olt double %108, %117
  br i1 %118, label %114, label %119, !llvm.loop !120

119:                                              ; preds = %114
  %120 = icmp ult ptr %110, %116
  br i1 %120, label %121, label %122

121:                                              ; preds = %119
  store double %117, ptr %110, align 8, !tbaa !32
  store double %111, ptr %116, align 8, !tbaa !32
  br label %105, !llvm.loop !121

122:                                              ; preds = %119
  tail call void @_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_(ptr noundef nonnull %110, ptr noundef %14, i64 noundef %78, i8 undef)
  %123 = ptrtoint ptr %110 to i64
  %124 = sub i64 %123, %6
  %125 = icmp sgt i64 %124, 128
  br i1 %125, label %12, label %126, !llvm.loop !122

126:                                              ; preds = %122, %73, %4
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZSt22__final_insertion_sortIPdN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_(ptr noundef %0, ptr noundef %1, i8 %2) local_unnamed_addr #11 comdat {
  %4 = ptrtoint ptr %1 to i64
  %5 = ptrtoint ptr %0 to i64
  %6 = sub i64 %4, %5
  %7 = icmp sgt i64 %6, 128
  br i1 %7, label %8, label %56

8:                                                ; preds = %3
  %9 = getelementptr i8, ptr %0, i64 8
  br label %10

10:                                               ; preds = %32, %8
  %11 = phi i64 [ 8, %8 ], [ %34, %32 ]
  %12 = phi ptr [ %0, %8 ], [ %13, %32 ]
  %13 = getelementptr inbounds nuw i8, ptr %0, i64 %11
  %14 = load double, ptr %13, align 8, !tbaa !32
  %15 = load double, ptr %0, align 8, !tbaa !32
  %16 = fcmp olt double %14, %15
  br i1 %16, label %17, label %22

17:                                               ; preds = %10
  %18 = icmp samesign ugt i64 %11, 8
  br i1 %18, label %19, label %20, !prof !41

19:                                               ; preds = %17
  tail call void @llvm.memmove.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(1) %9, ptr noundef nonnull align 8 dereferenceable(1) %0, i64 %11, i1 false)
  br label %32

20:                                               ; preds = %17
  %21 = getelementptr inbounds nuw i8, ptr %12, i64 8
  store double %15, ptr %21, align 8, !tbaa !32
  br label %32

22:                                               ; preds = %10
  %23 = load double, ptr %12, align 8, !tbaa !32
  %24 = fcmp olt double %14, %23
  br i1 %24, label %25, label %32

25:                                               ; preds = %22, %25
  %26 = phi double [ %30, %25 ], [ %23, %22 ]
  %27 = phi ptr [ %29, %25 ], [ %12, %22 ]
  %28 = phi ptr [ %27, %25 ], [ %13, %22 ]
  store double %26, ptr %28, align 8, !tbaa !32
  %29 = getelementptr inbounds i8, ptr %27, i64 -8
  %30 = load double, ptr %29, align 8, !tbaa !32
  %31 = fcmp olt double %14, %30
  br i1 %31, label %25, label %32, !llvm.loop !123

32:                                               ; preds = %25, %22, %20, %19
  %33 = phi ptr [ %0, %19 ], [ %0, %20 ], [ %13, %22 ], [ %27, %25 ]
  store double %14, ptr %33, align 8, !tbaa !32
  %34 = add nuw nsw i64 %11, 8
  %35 = icmp eq i64 %34, 128
  br i1 %35, label %36, label %10, !llvm.loop !124

36:                                               ; preds = %32
  %37 = getelementptr inbounds nuw i8, ptr %0, i64 128
  %38 = icmp eq ptr %37, %1
  br i1 %38, label %94, label %39

39:                                               ; preds = %36, %52
  %40 = phi ptr [ %54, %52 ], [ %37, %36 ]
  %41 = load double, ptr %40, align 8, !tbaa !32
  %42 = getelementptr inbounds i8, ptr %40, i64 -8
  %43 = load double, ptr %42, align 8, !tbaa !32
  %44 = fcmp olt double %41, %43
  br i1 %44, label %45, label %52

45:                                               ; preds = %39, %45
  %46 = phi double [ %50, %45 ], [ %43, %39 ]
  %47 = phi ptr [ %49, %45 ], [ %42, %39 ]
  %48 = phi ptr [ %47, %45 ], [ %40, %39 ]
  store double %46, ptr %48, align 8, !tbaa !32
  %49 = getelementptr inbounds i8, ptr %47, i64 -8
  %50 = load double, ptr %49, align 8, !tbaa !32
  %51 = fcmp olt double %41, %50
  br i1 %51, label %45, label %52, !llvm.loop !123

52:                                               ; preds = %45, %39
  %53 = phi ptr [ %40, %39 ], [ %47, %45 ]
  store double %41, ptr %53, align 8, !tbaa !32
  %54 = getelementptr inbounds nuw i8, ptr %40, i64 8
  %55 = icmp eq ptr %54, %1
  br i1 %55, label %94, label %39, !llvm.loop !125

56:                                               ; preds = %3
  %57 = icmp eq ptr %0, %1
  %58 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %59 = icmp eq ptr %58, %1
  %60 = select i1 %57, i1 true, i1 %59
  br i1 %60, label %94, label %61

61:                                               ; preds = %56, %90
  %62 = phi ptr [ %92, %90 ], [ %58, %56 ]
  %63 = phi ptr [ %62, %90 ], [ %0, %56 ]
  %64 = load double, ptr %62, align 8, !tbaa !32
  %65 = load double, ptr %0, align 8, !tbaa !32
  %66 = fcmp olt double %64, %65
  br i1 %66, label %67, label %80

67:                                               ; preds = %61
  %68 = ptrtoint ptr %62 to i64
  %69 = sub i64 %68, %5
  %70 = ashr exact i64 %69, 3
  %71 = icmp sgt i64 %70, 1
  br i1 %71, label %72, label %76, !prof !41

72:                                               ; preds = %67
  %73 = getelementptr inbounds nuw i8, ptr %63, i64 16
  %74 = sub nsw i64 0, %70
  %75 = getelementptr inbounds double, ptr %73, i64 %74
  tail call void @llvm.memmove.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(1) %75, ptr noundef nonnull align 8 dereferenceable(1) %0, i64 %69, i1 false)
  br label %90

76:                                               ; preds = %67
  %77 = icmp eq i64 %69, 8
  br i1 %77, label %78, label %90

78:                                               ; preds = %76
  %79 = getelementptr inbounds nuw i8, ptr %63, i64 8
  store double %65, ptr %79, align 8, !tbaa !32
  br label %90

80:                                               ; preds = %61
  %81 = load double, ptr %63, align 8, !tbaa !32
  %82 = fcmp olt double %64, %81
  br i1 %82, label %83, label %90

83:                                               ; preds = %80, %83
  %84 = phi double [ %88, %83 ], [ %81, %80 ]
  %85 = phi ptr [ %87, %83 ], [ %63, %80 ]
  %86 = phi ptr [ %85, %83 ], [ %62, %80 ]
  store double %84, ptr %86, align 8, !tbaa !32
  %87 = getelementptr inbounds i8, ptr %85, i64 -8
  %88 = load double, ptr %87, align 8, !tbaa !32
  %89 = fcmp olt double %64, %88
  br i1 %89, label %83, label %90, !llvm.loop !123

90:                                               ; preds = %83, %80, %78, %76, %72
  %91 = phi ptr [ %0, %72 ], [ %0, %76 ], [ %0, %78 ], [ %62, %80 ], [ %85, %83 ]
  store double %64, ptr %91, align 8, !tbaa !32
  %92 = getelementptr inbounds nuw i8, ptr %62, i64 8
  %93 = icmp eq ptr %92, %1
  br i1 %93, label %94, label %61, !llvm.loop !124

94:                                               ; preds = %90, %52, %56, %36
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_RT0_(ptr noundef %0, ptr noundef %1, ptr noundef nonnull align 1 dereferenceable(1) %2) local_unnamed_addr #11 comdat {
  %4 = ptrtoint ptr %1 to i64
  %5 = ptrtoint ptr %0 to i64
  %6 = sub i64 %4, %5
  %7 = ashr exact i64 %6, 3
  %8 = icmp slt i64 %7, 2
  br i1 %8, label %103, label %9

9:                                                ; preds = %3
  %10 = add nsw i64 %7, -2
  %11 = lshr i64 %10, 1
  %12 = add nsw i64 %7, -1
  %13 = lshr i64 %12, 1
  %14 = and i64 %6, 8
  %15 = icmp eq i64 %14, 0
  %16 = lshr exact i64 %10, 1
  br i1 %15, label %17, label %21

17:                                               ; preds = %9
  %18 = or disjoint i64 %10, 1
  %19 = getelementptr inbounds nuw double, ptr %0, i64 %18
  %20 = getelementptr inbounds nuw double, ptr %0, i64 %16
  br label %59

21:                                               ; preds = %9, %54
  %22 = phi i64 [ %58, %54 ], [ %11, %9 ]
  %23 = getelementptr inbounds nuw double, ptr %0, i64 %22
  %24 = load double, ptr %23, align 8, !tbaa !32
  %25 = icmp slt i64 %22, %13
  br i1 %25, label %26, label %54

26:                                               ; preds = %21, %26
  %27 = phi i64 [ %37, %26 ], [ %22, %21 ]
  %28 = shl i64 %27, 1
  %29 = add i64 %28, 2
  %30 = getelementptr inbounds double, ptr %0, i64 %29
  %31 = getelementptr double, ptr %0, i64 %28
  %32 = getelementptr i8, ptr %31, i64 8
  %33 = load double, ptr %30, align 8, !tbaa !32
  %34 = load double, ptr %32, align 8, !tbaa !32
  %35 = fcmp olt double %33, %34
  %36 = or disjoint i64 %28, 1
  %37 = select i1 %35, i64 %36, i64 %29
  %38 = getelementptr inbounds double, ptr %0, i64 %37
  %39 = load double, ptr %38, align 8, !tbaa !32
  %40 = getelementptr inbounds double, ptr %0, i64 %27
  store double %39, ptr %40, align 8, !tbaa !32
  %41 = icmp slt i64 %37, %13
  br i1 %41, label %26, label %42, !llvm.loop !116

42:                                               ; preds = %26
  %43 = icmp sgt i64 %37, %22
  br i1 %43, label %44, label %54

44:                                               ; preds = %42, %51
  %45 = phi i64 [ %47, %51 ], [ %37, %42 ]
  %46 = add nsw i64 %45, -1
  %47 = sdiv i64 %46, 2
  %48 = getelementptr inbounds nuw double, ptr %0, i64 %47
  %49 = load double, ptr %48, align 8, !tbaa !32
  %50 = fcmp olt double %49, %24
  br i1 %50, label %51, label %54

51:                                               ; preds = %44
  %52 = getelementptr inbounds nuw double, ptr %0, i64 %45
  store double %49, ptr %52, align 8, !tbaa !32
  %53 = icmp sgt i64 %47, %22
  br i1 %53, label %44, label %54, !llvm.loop !117

54:                                               ; preds = %44, %51, %21, %42
  %55 = phi i64 [ %37, %42 ], [ %22, %21 ], [ %47, %51 ], [ %45, %44 ]
  %56 = getelementptr inbounds nuw double, ptr %0, i64 %55
  store double %24, ptr %56, align 8, !tbaa !32
  %57 = icmp eq i64 %22, 0
  %58 = add nsw i64 %22, -1
  br i1 %57, label %103, label %21, !llvm.loop !126

59:                                               ; preds = %17, %98
  %60 = phi i64 [ %102, %98 ], [ %11, %17 ]
  %61 = getelementptr inbounds nuw double, ptr %0, i64 %60
  %62 = load double, ptr %61, align 8, !tbaa !32
  %63 = icmp slt i64 %60, %13
  br i1 %63, label %64, label %80

64:                                               ; preds = %59, %64
  %65 = phi i64 [ %75, %64 ], [ %60, %59 ]
  %66 = shl i64 %65, 1
  %67 = add i64 %66, 2
  %68 = getelementptr inbounds double, ptr %0, i64 %67
  %69 = getelementptr double, ptr %0, i64 %66
  %70 = getelementptr i8, ptr %69, i64 8
  %71 = load double, ptr %68, align 8, !tbaa !32
  %72 = load double, ptr %70, align 8, !tbaa !32
  %73 = fcmp olt double %71, %72
  %74 = or disjoint i64 %66, 1
  %75 = select i1 %73, i64 %74, i64 %67
  %76 = getelementptr inbounds double, ptr %0, i64 %75
  %77 = load double, ptr %76, align 8, !tbaa !32
  %78 = getelementptr inbounds double, ptr %0, i64 %65
  store double %77, ptr %78, align 8, !tbaa !32
  %79 = icmp slt i64 %75, %13
  br i1 %79, label %64, label %80, !llvm.loop !116

80:                                               ; preds = %64, %59
  %81 = phi i64 [ %60, %59 ], [ %75, %64 ]
  %82 = icmp eq i64 %81, %16
  br i1 %82, label %83, label %85

83:                                               ; preds = %80
  %84 = load double, ptr %19, align 8, !tbaa !32
  store double %84, ptr %20, align 8, !tbaa !32
  br label %85

85:                                               ; preds = %83, %80
  %86 = phi i64 [ %18, %83 ], [ %81, %80 ]
  %87 = icmp sgt i64 %86, %60
  br i1 %87, label %88, label %98

88:                                               ; preds = %85, %95
  %89 = phi i64 [ %91, %95 ], [ %86, %85 ]
  %90 = add nsw i64 %89, -1
  %91 = sdiv i64 %90, 2
  %92 = getelementptr inbounds nuw double, ptr %0, i64 %91
  %93 = load double, ptr %92, align 8, !tbaa !32
  %94 = fcmp olt double %93, %62
  br i1 %94, label %95, label %98

95:                                               ; preds = %88
  %96 = getelementptr inbounds nuw double, ptr %0, i64 %89
  store double %93, ptr %96, align 8, !tbaa !32
  %97 = icmp sgt i64 %91, %60
  br i1 %97, label %88, label %98, !llvm.loop !117

98:                                               ; preds = %88, %95, %85
  %99 = phi i64 [ %86, %85 ], [ %91, %95 ], [ %89, %88 ]
  %100 = getelementptr inbounds nuw double, ptr %0, i64 %99
  store double %62, ptr %100, align 8, !tbaa !32
  %101 = icmp eq i64 %60, 0
  %102 = add nsw i64 %60, -1
  br i1 %101, label %103, label %59, !llvm.loop !126

103:                                              ; preds = %54, %98, %3
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smax.i32(i32, i32) #18

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #19

attributes #0 = { mustprogress nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nounwind willreturn allockind("realloc") allocsize(1) memory(argmem: readwrite, inaccessiblemem: readwrite) "alloc-family"="malloc" "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { mustprogress nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #6 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: read) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { mustprogress nocallback nofree nounwind willreturn memory(errnomem: write) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #8 = { nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #9 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #10 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #11 = { mustprogress uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #12 = { mustprogress norecurse uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #13 = { nobuiltin allocsize(0) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #14 = { nofree "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #15 = { nobuiltin nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #16 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #17 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #18 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #19 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #20 = { nounwind allocsize(1) }
attributes #21 = { cold noreturn nounwind }
attributes #22 = { nounwind willreturn memory(read) }
attributes #23 = { nounwind }
attributes #24 = { builtin allocsize(0) }
attributes #25 = { builtin nounwind }

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
!32 = !{!15, !15, i64 0}
!33 = distinct !{!33, !19}
!34 = distinct !{!34, !19}
!35 = distinct !{!35, !19}
!36 = distinct !{!36, !19}
!37 = distinct !{!37, !19}
!38 = distinct !{!38, !19}
!39 = !{!16, !16, i64 0}
!40 = distinct !{!40, !19}
!41 = !{!"branch_weights", !"expected", i32 2000, i32 1}
!42 = distinct !{!42, !19}
!43 = distinct !{!43, !19}
!44 = distinct !{!44, !19}
!45 = distinct !{!45, !19}
!46 = distinct !{!46, !19}
!47 = distinct !{!47, !19}
!48 = distinct !{!48, !19}
!49 = distinct !{!49, !19}
!50 = distinct !{!50, !19}
!51 = distinct !{!51, !19}
!52 = distinct !{!52, !19}
!53 = distinct !{!53, !19}
!54 = distinct !{!54, !19}
!55 = distinct !{!55, !19}
!56 = distinct !{!56, !19}
!57 = distinct !{!57, !19}
!58 = distinct !{!58, !19}
!59 = distinct !{!59, !19}
!60 = distinct !{!60, !19}
!61 = distinct !{!61, !19}
!62 = distinct !{!62, !19}
!63 = distinct !{!63, !19}
!64 = distinct !{!64, !19}
!65 = distinct !{!65, !19}
!66 = distinct !{!66, !19}
!67 = distinct !{!67, !19}
!68 = distinct !{!68, !19}
!69 = distinct !{!69, !19}
!70 = distinct !{!70, !19}
!71 = distinct !{!71, !19}
!72 = distinct !{!72, !19}
!73 = distinct !{!73, !19}
!74 = distinct !{!74, !19}
!75 = distinct !{!75, !19}
!76 = distinct !{!76, !19}
!77 = distinct !{!77, !19}
!78 = distinct !{!78, !19}
!79 = distinct !{!79, !19}
!80 = distinct !{!80, !19}
!81 = !{!8, !8, i64 0}
!82 = distinct !{!82, !19}
!83 = distinct !{!83, !19}
!84 = distinct !{!84, !19}
!85 = distinct !{!85, !19}
!86 = distinct !{!86, !19}
!87 = distinct !{!87, !19}
!88 = distinct !{!88, !19}
!89 = distinct !{!89, !19}
!90 = distinct !{!90, !19}
!91 = distinct !{!91, !19}
!92 = distinct !{!92, !19}
!93 = distinct !{!93, !19}
!94 = distinct !{!94, !19}
!95 = distinct !{!95, !19}
!96 = distinct !{!96, !19}
!97 = distinct !{!97, !19}
!98 = distinct !{!98, !19}
!99 = distinct !{!99, !19}
!100 = distinct !{!100, !19}
!101 = distinct !{!101, !19}
!102 = distinct !{!102, !19}
!103 = distinct !{!103, !19}
!104 = distinct !{!104, !19}
!105 = distinct !{!105, !19}
!106 = distinct !{!106, !19}
!107 = distinct !{!107, !19}
!108 = distinct !{!108, !19}
!109 = distinct !{!109, !19}
!110 = distinct !{!110, !19}
!111 = distinct !{!111, !19}
!112 = distinct !{!112, !19}
!113 = distinct !{!113, !19}
!114 = distinct !{!114, !19}
!115 = distinct !{!115, !19}
!116 = distinct !{!116, !19}
!117 = distinct !{!117, !19}
!118 = distinct !{!118, !19}
!119 = distinct !{!119, !19}
!120 = distinct !{!120, !19}
!121 = distinct !{!121, !19}
!122 = distinct !{!122, !19}
!123 = distinct !{!123, !19}
!124 = distinct !{!124, !19}
!125 = distinct !{!125, !19}
!126 = distinct !{!126, !19}
