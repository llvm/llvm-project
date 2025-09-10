; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Shootout-C++/moments.cpp'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Shootout-C++/moments.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.moments = type { double, double, double, double, double, double, double }
%"struct.__gnu_cxx::__ops::_Iter_less_iter" = type { i8 }

$_ZN7momentsIdEC2IN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEET_S9_ = comdat any

$_ZSt13__introselectIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_S9_T0_T1_ = comdat any

$_ZSt13__heap_selectIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_S9_T0_ = comdat any

$_ZSt11__make_heapIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_RT0_ = comdat any

@.str = private unnamed_addr constant [24 x i8] c"n:                  %d\0A\00", align 1
@.str.1 = private unnamed_addr constant [24 x i8] c"median:             %f\0A\00", align 1
@.str.2 = private unnamed_addr constant [24 x i8] c"mean:               %f\0A\00", align 1
@.str.3 = private unnamed_addr constant [24 x i8] c"average_deviation:  %f\0A\00", align 1
@.str.4 = private unnamed_addr constant [24 x i8] c"standard_deviation: %f\0A\00", align 1
@.str.5 = private unnamed_addr constant [24 x i8] c"variance:           %f\0A\00", align 1
@.str.6 = private unnamed_addr constant [24 x i8] c"skew:               %f\0A\00", align 1
@.str.7 = private unnamed_addr constant [24 x i8] c"kurtosis:           %f\0A\00", align 1
@.str.8 = private unnamed_addr constant [26 x i8] c"vector::_M_realloc_append\00", align 1

; Function Attrs: mustprogress norecurse uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 personality ptr @__gxx_personality_v0 {
  %3 = alloca %struct.moments, align 8
  %4 = icmp eq i32 %0, 2
  br i1 %4, label %5, label %11

5:                                                ; preds = %2
  %6 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %7 = load ptr, ptr %6, align 8, !tbaa !6
  %8 = tail call i64 @__isoc23_strtol(ptr noundef nonnull %7, ptr noundef null, i32 noundef 10) #14
  %9 = trunc i64 %8 to i32
  %10 = icmp eq i32 %9, 0
  br i1 %10, label %13, label %11

11:                                               ; preds = %2, %5
  %12 = phi i32 [ %9, %5 ], [ 5000000, %2 ]
  br label %18

13:                                               ; preds = %52, %5
  %14 = phi ptr [ null, %5 ], [ %53, %52 ]
  %15 = phi ptr [ null, %5 ], [ %56, %52 ]
  %16 = phi ptr [ null, %5 ], [ %55, %52 ]
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #14
  %17 = ptrtoint ptr %16 to i64
  invoke void @_ZN7momentsIdEC2IN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEET_S9_(ptr noundef nonnull align 8 dereferenceable(56) %3, ptr %16, ptr %15)
          to label %63 unwind label %93

18:                                               ; preds = %11, %52
  %19 = phi i32 [ %57, %52 ], [ 0, %11 ]
  %20 = phi ptr [ %55, %52 ], [ null, %11 ]
  %21 = phi ptr [ %56, %52 ], [ null, %11 ]
  %22 = phi ptr [ %53, %52 ], [ null, %11 ]
  %23 = uitofp i32 %19 to double
  %24 = icmp eq ptr %21, %22
  br i1 %24, label %26, label %25

25:                                               ; preds = %18
  store double %23, ptr %21, align 8, !tbaa !11
  br label %52

26:                                               ; preds = %18
  %27 = ptrtoint ptr %21 to i64
  %28 = ptrtoint ptr %20 to i64
  %29 = sub i64 %27, %28
  %30 = icmp eq i64 %29, 9223372036854775800
  br i1 %30, label %31, label %33

31:                                               ; preds = %26
  invoke void @_ZSt20__throw_length_errorPKc(ptr noundef nonnull @.str.8) #15
          to label %32 unwind label %61

32:                                               ; preds = %31
  unreachable

33:                                               ; preds = %26
  %34 = ashr exact i64 %29, 3
  %35 = tail call i64 @llvm.umax.i64(i64 %34, i64 1)
  %36 = add nsw i64 %35, %34
  %37 = icmp ult i64 %36, %34
  %38 = tail call i64 @llvm.umin.i64(i64 %36, i64 1152921504606846975)
  %39 = select i1 %37, i64 1152921504606846975, i64 %38
  %40 = icmp ne i64 %39, 0
  tail call void @llvm.assume(i1 %40)
  %41 = shl nuw nsw i64 %39, 3
  %42 = invoke noalias noundef nonnull ptr @_Znwm(i64 noundef %41) #16
          to label %43 unwind label %59

43:                                               ; preds = %33
  %44 = getelementptr inbounds i8, ptr %42, i64 %29
  store double %23, ptr %44, align 8, !tbaa !11
  %45 = icmp sgt i64 %29, 0
  br i1 %45, label %46, label %47

46:                                               ; preds = %43
  tail call void @llvm.memcpy.p0.p0.i64(ptr nonnull align 8 %42, ptr align 8 %20, i64 %29, i1 false)
  br label %47

47:                                               ; preds = %46, %43
  %48 = icmp eq ptr %20, null
  br i1 %48, label %50, label %49

49:                                               ; preds = %47
  tail call void @_ZdlPvm(ptr noundef nonnull %20, i64 noundef %29) #17
  br label %50

50:                                               ; preds = %49, %47
  %51 = getelementptr inbounds nuw double, ptr %42, i64 %39
  br label %52

52:                                               ; preds = %50, %25
  %53 = phi ptr [ %51, %50 ], [ %22, %25 ]
  %54 = phi ptr [ %44, %50 ], [ %21, %25 ]
  %55 = phi ptr [ %42, %50 ], [ %20, %25 ]
  %56 = getelementptr inbounds nuw i8, ptr %54, i64 8
  %57 = add nuw i32 %19, 1
  %58 = icmp eq i32 %57, %12
  br i1 %58, label %13, label %18, !llvm.loop !13

59:                                               ; preds = %33
  %60 = landingpad { ptr, i32 }
          cleanup
  br label %95

61:                                               ; preds = %31
  %62 = landingpad { ptr, i32 }
          cleanup
  br label %95

63:                                               ; preds = %13
  %64 = ptrtoint ptr %15 to i64
  %65 = sub i64 %64, %17
  %66 = ashr exact i64 %65, 3
  %67 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i64 noundef %66)
  %68 = load double, ptr %3, align 8, !tbaa !15
  %69 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, double noundef %68)
  %70 = getelementptr inbounds nuw i8, ptr %3, i64 8
  %71 = load double, ptr %70, align 8, !tbaa !17
  %72 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, double noundef %71)
  %73 = getelementptr inbounds nuw i8, ptr %3, i64 16
  %74 = load double, ptr %73, align 8, !tbaa !18
  %75 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, double noundef %74)
  %76 = getelementptr inbounds nuw i8, ptr %3, i64 24
  %77 = load double, ptr %76, align 8, !tbaa !19
  %78 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.4, double noundef %77)
  %79 = getelementptr inbounds nuw i8, ptr %3, i64 32
  %80 = load double, ptr %79, align 8, !tbaa !20
  %81 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.5, double noundef %80)
  %82 = getelementptr inbounds nuw i8, ptr %3, i64 40
  %83 = load double, ptr %82, align 8, !tbaa !21
  %84 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, double noundef %83)
  %85 = getelementptr inbounds nuw i8, ptr %3, i64 48
  %86 = load double, ptr %85, align 8, !tbaa !22
  %87 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.7, double noundef %86)
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #14
  %88 = icmp eq ptr %16, null
  br i1 %88, label %92, label %89

89:                                               ; preds = %63
  %90 = ptrtoint ptr %14 to i64
  %91 = sub i64 %90, %17
  call void @_ZdlPvm(ptr noundef nonnull %16, i64 noundef %91) #17
  br label %92

92:                                               ; preds = %63, %89
  ret i32 0

93:                                               ; preds = %13
  %94 = landingpad { ptr, i32 }
          cleanup
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #14
  br label %95

95:                                               ; preds = %59, %61, %93
  %96 = phi ptr [ %14, %93 ], [ %21, %59 ], [ %21, %61 ]
  %97 = phi ptr [ %16, %93 ], [ %20, %59 ], [ %20, %61 ]
  %98 = phi { ptr, i32 } [ %94, %93 ], [ %60, %59 ], [ %62, %61 ]
  %99 = icmp eq ptr %97, null
  br i1 %99, label %104, label %100

100:                                              ; preds = %95
  %101 = ptrtoint ptr %96 to i64
  %102 = ptrtoint ptr %97 to i64
  %103 = sub i64 %101, %102
  call void @_ZdlPvm(ptr noundef nonnull %97, i64 noundef %103) #17
  br label %104

104:                                              ; preds = %95, %100
  resume { ptr, i32 } %98
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

declare i32 @__gxx_personality_v0(...)

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZN7momentsIdEC2IN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEET_S9_(ptr noundef nonnull align 8 dereferenceable(56) %0, ptr %1, ptr %2) unnamed_addr #2 comdat {
  %4 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %5 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %6 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %7 = getelementptr inbounds nuw i8, ptr %0, i64 32
  %8 = getelementptr inbounds nuw i8, ptr %0, i64 40
  %9 = getelementptr inbounds nuw i8, ptr %0, i64 48
  %10 = icmp eq ptr %1, %2
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(56) %0, i8 0, i64 56, i1 false)
  %11 = ptrtoint ptr %2 to i64
  %12 = ptrtoint ptr %1 to i64
  br i1 %10, label %41, label %13

13:                                               ; preds = %3
  %14 = add i64 %11, -8
  %15 = sub i64 %14, %12
  %16 = lshr i64 %15, 3
  %17 = add nuw nsw i64 %16, 1
  %18 = icmp ult i64 %15, 8
  br i1 %18, label %38, label %19

19:                                               ; preds = %13
  %20 = and i64 %17, 4611686018427387902
  %21 = shl i64 %20, 3
  %22 = getelementptr i8, ptr %1, i64 %21
  br label %23

23:                                               ; preds = %23, %19
  %24 = phi i64 [ 0, %19 ], [ %34, %23 ]
  %25 = phi double [ 0.000000e+00, %19 ], [ %33, %23 ]
  %26 = shl i64 %24, 3
  %27 = getelementptr i8, ptr %1, i64 %26
  %28 = getelementptr i8, ptr %1, i64 %26
  %29 = getelementptr i8, ptr %28, i64 8
  %30 = load double, ptr %27, align 8, !tbaa !11
  %31 = load double, ptr %29, align 8, !tbaa !11
  %32 = fadd double %25, %30
  %33 = fadd double %32, %31
  %34 = add nuw i64 %24, 2
  %35 = icmp eq i64 %34, %20
  br i1 %35, label %36, label %23, !llvm.loop !23

36:                                               ; preds = %23
  %37 = icmp eq i64 %17, %20
  br i1 %37, label %53, label %38

38:                                               ; preds = %13, %36
  %39 = phi double [ 0.000000e+00, %13 ], [ %33, %36 ]
  %40 = phi ptr [ %1, %13 ], [ %22, %36 ]
  br label %46

41:                                               ; preds = %3
  %42 = sub i64 %11, %12
  %43 = ashr exact i64 %42, 3
  %44 = uitofp i64 %43 to double
  %45 = fdiv double 0.000000e+00, %44
  store double %45, ptr %4, align 8, !tbaa !17
  br label %61

46:                                               ; preds = %38, %46
  %47 = phi double [ %50, %46 ], [ %39, %38 ]
  %48 = phi ptr [ %51, %46 ], [ %40, %38 ]
  %49 = load double, ptr %48, align 8, !tbaa !11
  %50 = fadd double %47, %49
  %51 = getelementptr inbounds nuw i8, ptr %48, i64 8
  %52 = icmp eq ptr %51, %2
  br i1 %52, label %53, label %46, !llvm.loop !26

53:                                               ; preds = %46, %36
  %54 = phi double [ %33, %36 ], [ %50, %46 ]
  %55 = ptrtoint ptr %2 to i64
  %56 = ptrtoint ptr %1 to i64
  %57 = sub i64 %55, %56
  %58 = ashr exact i64 %57, 3
  %59 = uitofp i64 %58 to double
  %60 = fdiv double %54, %59
  store double %60, ptr %4, align 8, !tbaa !17
  br label %73

61:                                               ; preds = %73, %41
  %62 = phi double [ %44, %41 ], [ %59, %73 ]
  %63 = phi i64 [ %43, %41 ], [ %58, %73 ]
  %64 = phi i64 [ %42, %41 ], [ %57, %73 ]
  %65 = phi double [ 0.000000e+00, %41 ], [ %84, %73 ]
  %66 = phi double [ 0.000000e+00, %41 ], [ %82, %73 ]
  %67 = fdiv double %66, %62
  store double %67, ptr %5, align 8, !tbaa !18
  %68 = add nsw i64 %63, -1
  %69 = uitofp i64 %68 to double
  %70 = fdiv double %65, %69
  store double %70, ptr %7, align 8, !tbaa !20
  %71 = tail call double @sqrt(double noundef %70) #14, !tbaa !27
  store double %71, ptr %6, align 8, !tbaa !19
  %72 = fcmp une double %70, 0.000000e+00
  br i1 %72, label %90, label %99

73:                                               ; preds = %53, %73
  %74 = phi ptr [ %88, %73 ], [ %1, %53 ]
  %75 = phi double [ %82, %73 ], [ 0.000000e+00, %53 ]
  %76 = phi double [ %84, %73 ], [ 0.000000e+00, %53 ]
  %77 = phi double [ %86, %73 ], [ 0.000000e+00, %53 ]
  %78 = phi double [ %87, %73 ], [ 0.000000e+00, %53 ]
  %79 = load double, ptr %74, align 8, !tbaa !11
  %80 = fsub double %79, %60
  %81 = tail call double @llvm.fabs.f64(double %80)
  %82 = fadd double %75, %81
  store double %82, ptr %5, align 8, !tbaa !18
  %83 = fmul double %80, %80
  %84 = fadd double %83, %76
  store double %84, ptr %7, align 8, !tbaa !20
  %85 = fmul double %80, %83
  %86 = fadd double %85, %77
  store double %86, ptr %8, align 8, !tbaa !21
  %87 = tail call double @llvm.fmuladd.f64(double %85, double %80, double %78)
  store double %87, ptr %9, align 8, !tbaa !22
  %88 = getelementptr inbounds nuw i8, ptr %74, i64 8
  %89 = icmp eq ptr %88, %2
  br i1 %89, label %61, label %73, !llvm.loop !29

90:                                               ; preds = %61
  %91 = fmul double %70, %62
  %92 = fmul double %91, %71
  %93 = load double, ptr %8, align 8, !tbaa !21
  %94 = fdiv double %93, %92
  store double %94, ptr %8, align 8, !tbaa !21
  %95 = load double, ptr %9, align 8, !tbaa !22
  %96 = fmul double %70, %91
  %97 = fdiv double %95, %96
  %98 = fadd double %97, -3.000000e+00
  store double %98, ptr %9, align 8, !tbaa !22
  br label %99

99:                                               ; preds = %90, %61
  %100 = ashr exact i64 %64, 1
  %101 = and i64 %100, -8
  %102 = getelementptr inbounds nuw i8, ptr %1, i64 %101
  %103 = icmp eq ptr %102, %2
  %104 = select i1 %10, i1 true, i1 %103
  br i1 %104, label %109, label %105

105:                                              ; preds = %99
  %106 = tail call range(i64 0, 65) i64 @llvm.ctlz.i64(i64 %63, i1 true)
  %107 = shl nuw nsw i64 %106, 1
  %108 = xor i64 %107, 126
  tail call void @_ZSt13__introselectIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_S9_T0_T1_(ptr %1, ptr %102, ptr %2, i64 noundef %108, i8 undef)
  br label %109

109:                                              ; preds = %99, %105
  %110 = and i64 %64, 8
  %111 = icmp eq i64 %110, 0
  br i1 %111, label %112, label %133

112:                                              ; preds = %109
  %113 = icmp ult i64 %100, 16
  br i1 %113, label %127, label %114

114:                                              ; preds = %112
  %115 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %116 = load double, ptr %1, align 8, !tbaa !11
  br label %117

117:                                              ; preds = %117, %114
  %118 = phi double [ %123, %117 ], [ %116, %114 ]
  %119 = phi ptr [ %125, %117 ], [ %115, %114 ]
  %120 = phi ptr [ %124, %117 ], [ %1, %114 ]
  %121 = load double, ptr %119, align 8, !tbaa !11
  %122 = fcmp olt double %118, %121
  %123 = select i1 %122, double %121, double %118
  %124 = select i1 %122, ptr %119, ptr %120
  %125 = getelementptr inbounds nuw i8, ptr %119, i64 8
  %126 = icmp eq ptr %125, %102
  br i1 %126, label %127, label %117, !llvm.loop !30

127:                                              ; preds = %117, %112
  %128 = phi ptr [ %1, %112 ], [ %124, %117 ]
  %129 = load double, ptr %102, align 8, !tbaa !11
  %130 = load double, ptr %128, align 8, !tbaa !11
  %131 = fadd double %129, %130
  %132 = fmul double %131, 5.000000e-01
  br label %135

133:                                              ; preds = %109
  %134 = load double, ptr %102, align 8, !tbaa !11
  br label %135

135:                                              ; preds = %133, %127
  %136 = phi double [ %134, %133 ], [ %132, %127 ]
  store double %136, ptr %0, align 8, !tbaa !15
  ret void
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #3

; Function Attrs: nounwind
declare i64 @__isoc23_strtol(ptr noundef, ptr noundef, i32 noundef) local_unnamed_addr #4

; Function Attrs: nobuiltin nounwind
declare void @_ZdlPvm(ptr noundef, i64 noundef) local_unnamed_addr #5

; Function Attrs: cold noreturn
declare void @_ZSt20__throw_length_errorPKc(ptr noundef) local_unnamed_addr #6

; Function Attrs: nobuiltin allocsize(0)
declare noundef nonnull ptr @_Znwm(i64 noundef) local_unnamed_addr #7

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #8

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fabs.f64(double) #9

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fmuladd.f64(double, double, double) #9

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(errnomem: write)
declare double @sqrt(double noundef) local_unnamed_addr #10

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZSt13__introselectIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_S9_T0_T1_(ptr %0, ptr %1, ptr %2, i64 noundef %3, i8 %4) local_unnamed_addr #2 comdat {
  %6 = ptrtoint ptr %2 to i64
  %7 = ptrtoint ptr %0 to i64
  %8 = sub i64 %6, %7
  %9 = ashr exact i64 %8, 3
  %10 = icmp sgt i64 %9, 3
  br i1 %10, label %11, label %76

11:                                               ; preds = %5, %67
  %12 = phi i64 [ %74, %67 ], [ %9, %5 ]
  %13 = phi i64 [ %22, %67 ], [ %3, %5 ]
  %14 = phi ptr [ %70, %67 ], [ %0, %5 ]
  %15 = phi ptr [ %69, %67 ], [ %2, %5 ]
  %16 = icmp eq i64 %13, 0
  br i1 %16, label %17, label %21

17:                                               ; preds = %11
  %18 = getelementptr inbounds nuw i8, ptr %1, i64 8
  tail call void @_ZSt13__heap_selectIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_S9_T0_(ptr %14, ptr nonnull %18, ptr %15, i8 undef)
  %19 = load double, ptr %14, align 8, !tbaa !11
  %20 = load double, ptr %1, align 8, !tbaa !11
  store double %20, ptr %14, align 8, !tbaa !11
  store double %19, ptr %1, align 8, !tbaa !11
  br label %117

21:                                               ; preds = %11
  %22 = add nsw i64 %13, -1
  %23 = lshr i64 %12, 1
  %24 = getelementptr inbounds nuw double, ptr %14, i64 %23
  %25 = getelementptr inbounds nuw i8, ptr %14, i64 8
  %26 = getelementptr inbounds i8, ptr %15, i64 -8
  %27 = load double, ptr %25, align 8, !tbaa !11
  %28 = load double, ptr %24, align 8, !tbaa !11
  %29 = fcmp olt double %27, %28
  %30 = load double, ptr %26, align 8, !tbaa !11
  br i1 %29, label %31, label %40

31:                                               ; preds = %21
  %32 = fcmp olt double %28, %30
  br i1 %32, label %33, label %35

33:                                               ; preds = %31
  %34 = load double, ptr %14, align 8, !tbaa !11
  store double %28, ptr %14, align 8, !tbaa !11
  store double %34, ptr %24, align 8, !tbaa !11
  br label %49

35:                                               ; preds = %31
  %36 = fcmp olt double %27, %30
  %37 = load double, ptr %14, align 8, !tbaa !11
  br i1 %36, label %38, label %39

38:                                               ; preds = %35
  store double %30, ptr %14, align 8, !tbaa !11
  store double %37, ptr %26, align 8, !tbaa !11
  br label %49

39:                                               ; preds = %35
  store double %27, ptr %14, align 8, !tbaa !11
  store double %37, ptr %25, align 8, !tbaa !11
  br label %49

40:                                               ; preds = %21
  %41 = fcmp olt double %27, %30
  br i1 %41, label %42, label %44

42:                                               ; preds = %40
  %43 = load double, ptr %14, align 8, !tbaa !11
  store double %27, ptr %14, align 8, !tbaa !11
  store double %43, ptr %25, align 8, !tbaa !11
  br label %49

44:                                               ; preds = %40
  %45 = fcmp olt double %28, %30
  %46 = load double, ptr %14, align 8, !tbaa !11
  br i1 %45, label %47, label %48

47:                                               ; preds = %44
  store double %30, ptr %14, align 8, !tbaa !11
  store double %46, ptr %26, align 8, !tbaa !11
  br label %49

48:                                               ; preds = %44
  store double %28, ptr %14, align 8, !tbaa !11
  store double %46, ptr %24, align 8, !tbaa !11
  br label %49

49:                                               ; preds = %48, %47, %42, %39, %38, %33
  br label %50

50:                                               ; preds = %49, %66
  %51 = phi ptr [ %61, %66 ], [ %15, %49 ]
  %52 = phi ptr [ %58, %66 ], [ %25, %49 ]
  %53 = load double, ptr %14, align 8, !tbaa !11
  br label %54

54:                                               ; preds = %54, %50
  %55 = phi ptr [ %52, %50 ], [ %58, %54 ]
  %56 = load double, ptr %55, align 8, !tbaa !11
  %57 = fcmp olt double %56, %53
  %58 = getelementptr inbounds nuw i8, ptr %55, i64 8
  br i1 %57, label %54, label %59, !llvm.loop !31

59:                                               ; preds = %54, %59
  %60 = phi ptr [ %61, %59 ], [ %51, %54 ]
  %61 = getelementptr inbounds i8, ptr %60, i64 -8
  %62 = load double, ptr %61, align 8, !tbaa !11
  %63 = fcmp olt double %53, %62
  br i1 %63, label %59, label %64, !llvm.loop !32

64:                                               ; preds = %59
  %65 = icmp ult ptr %55, %61
  br i1 %65, label %66, label %67

66:                                               ; preds = %64
  store double %62, ptr %55, align 8, !tbaa !11
  store double %56, ptr %61, align 8, !tbaa !11
  br label %50, !llvm.loop !33

67:                                               ; preds = %64
  %68 = icmp ugt ptr %55, %1
  %69 = select i1 %68, ptr %55, ptr %15
  %70 = select i1 %68, ptr %14, ptr %55
  %71 = ptrtoint ptr %69 to i64
  %72 = ptrtoint ptr %70 to i64
  %73 = sub i64 %71, %72
  %74 = ashr exact i64 %73, 3
  %75 = icmp sgt i64 %74, 3
  br i1 %75, label %11, label %76, !llvm.loop !34

76:                                               ; preds = %67, %5
  %77 = phi ptr [ %2, %5 ], [ %69, %67 ]
  %78 = phi ptr [ %0, %5 ], [ %70, %67 ]
  %79 = phi i64 [ %7, %5 ], [ %72, %67 ]
  %80 = icmp eq ptr %78, %77
  %81 = getelementptr inbounds nuw i8, ptr %78, i64 8
  %82 = icmp eq ptr %81, %77
  %83 = select i1 %80, i1 true, i1 %82
  br i1 %83, label %117, label %84

84:                                               ; preds = %76, %113
  %85 = phi ptr [ %115, %113 ], [ %81, %76 ]
  %86 = phi ptr [ %85, %113 ], [ %78, %76 ]
  %87 = load double, ptr %85, align 8, !tbaa !11
  %88 = load double, ptr %78, align 8, !tbaa !11
  %89 = fcmp olt double %87, %88
  br i1 %89, label %90, label %103

90:                                               ; preds = %84
  %91 = ptrtoint ptr %85 to i64
  %92 = sub i64 %91, %79
  %93 = ashr exact i64 %92, 3
  %94 = icmp sgt i64 %93, 1
  br i1 %94, label %95, label %99, !prof !35

95:                                               ; preds = %90
  %96 = getelementptr inbounds nuw i8, ptr %86, i64 16
  %97 = sub nsw i64 0, %93
  %98 = getelementptr inbounds double, ptr %96, i64 %97
  tail call void @llvm.memmove.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(1) %98, ptr noundef nonnull align 8 dereferenceable(1) %78, i64 %92, i1 false)
  br label %113

99:                                               ; preds = %90
  %100 = icmp eq i64 %92, 8
  br i1 %100, label %101, label %113

101:                                              ; preds = %99
  %102 = getelementptr inbounds nuw i8, ptr %86, i64 8
  store double %88, ptr %102, align 8, !tbaa !11
  br label %113

103:                                              ; preds = %84
  %104 = load double, ptr %86, align 8, !tbaa !11
  %105 = fcmp olt double %87, %104
  br i1 %105, label %106, label %113

106:                                              ; preds = %103, %106
  %107 = phi double [ %111, %106 ], [ %104, %103 ]
  %108 = phi ptr [ %110, %106 ], [ %86, %103 ]
  %109 = phi ptr [ %108, %106 ], [ %85, %103 ]
  store double %107, ptr %109, align 8, !tbaa !11
  %110 = getelementptr inbounds i8, ptr %108, i64 -8
  %111 = load double, ptr %110, align 8, !tbaa !11
  %112 = fcmp olt double %87, %111
  br i1 %112, label %106, label %113, !llvm.loop !36

113:                                              ; preds = %106, %103, %101, %99, %95
  %114 = phi ptr [ %78, %95 ], [ %78, %99 ], [ %78, %101 ], [ %85, %103 ], [ %108, %106 ]
  store double %87, ptr %114, align 8, !tbaa !11
  %115 = getelementptr inbounds nuw i8, ptr %85, i64 8
  %116 = icmp eq ptr %115, %77
  br i1 %116, label %117, label %84, !llvm.loop !37

117:                                              ; preds = %113, %76, %17
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZSt13__heap_selectIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_S9_T0_(ptr %0, ptr %1, ptr %2, i8 %3) local_unnamed_addr #2 comdat {
  %5 = alloca %"struct.__gnu_cxx::__ops::_Iter_less_iter", align 4
  %6 = freeze ptr %0
  %7 = freeze ptr %1
  store i8 %3, ptr %5, align 4
  call void @_ZSt11__make_heapIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_RT0_(ptr %6, ptr %7, ptr noundef nonnull align 1 dereferenceable(1) %5)
  %8 = icmp ult ptr %7, %2
  br i1 %8, label %9, label %102

9:                                                ; preds = %4
  %10 = ptrtoint ptr %7 to i64
  %11 = ptrtoint ptr %6 to i64
  %12 = sub i64 %10, %11
  %13 = ashr i64 %12, 3
  %14 = add nsw i64 %13, -1
  %15 = sdiv i64 %14, 2
  %16 = icmp sgt i64 %13, 2
  %17 = and i64 %12, 8
  %18 = icmp eq i64 %17, 0
  %19 = add nsw i64 %13, -2
  %20 = ashr exact i64 %19, 1
  br i1 %16, label %21, label %71

21:                                               ; preds = %9
  %22 = or disjoint i64 %19, 1
  %23 = getelementptr inbounds nuw double, ptr %6, i64 %22
  %24 = getelementptr inbounds double, ptr %6, i64 %20
  br label %25

25:                                               ; preds = %21, %65
  %26 = phi ptr [ %66, %65 ], [ %7, %21 ]
  %27 = load double, ptr %26, align 8, !tbaa !11
  %28 = load double, ptr %6, align 8, !tbaa !11
  %29 = fcmp olt double %27, %28
  br i1 %29, label %30, label %65

30:                                               ; preds = %25
  store double %28, ptr %26, align 8, !tbaa !11
  br label %31

31:                                               ; preds = %30, %31
  %32 = phi i64 [ %41, %31 ], [ 0, %30 ]
  %33 = shl i64 %32, 1
  %34 = add i64 %33, 2
  %35 = getelementptr inbounds double, ptr %6, i64 %34
  %36 = or disjoint i64 %33, 1
  %37 = getelementptr inbounds double, ptr %6, i64 %36
  %38 = load double, ptr %35, align 8, !tbaa !11
  %39 = load double, ptr %37, align 8, !tbaa !11
  %40 = fcmp olt double %38, %39
  %41 = select i1 %40, i64 %36, i64 %34
  %42 = getelementptr inbounds double, ptr %6, i64 %41
  %43 = load double, ptr %42, align 8, !tbaa !11
  %44 = getelementptr inbounds double, ptr %6, i64 %32
  store double %43, ptr %44, align 8, !tbaa !11
  %45 = icmp slt i64 %41, %15
  br i1 %45, label %31, label %68, !llvm.loop !38

46:                                               ; preds = %68
  %47 = icmp eq i64 %41, 0
  br i1 %47, label %62, label %50

48:                                               ; preds = %68
  %49 = load double, ptr %23, align 8, !tbaa !11
  store double %49, ptr %24, align 8, !tbaa !11
  br label %50

50:                                               ; preds = %48, %46
  %51 = phi i64 [ %41, %46 ], [ %22, %48 ]
  br label %52

52:                                               ; preds = %50, %59
  %53 = phi i64 [ %55, %59 ], [ %51, %50 ]
  %54 = add nsw i64 %53, -1
  %55 = lshr i64 %54, 1
  %56 = getelementptr inbounds nuw double, ptr %6, i64 %55
  %57 = load double, ptr %56, align 8, !tbaa !11
  %58 = fcmp olt double %57, %27
  br i1 %58, label %59, label %62

59:                                               ; preds = %52
  %60 = getelementptr inbounds double, ptr %6, i64 %53
  store double %57, ptr %60, align 8, !tbaa !11
  %61 = icmp ult i64 %54, 2
  br i1 %61, label %62, label %52, !llvm.loop !39

62:                                               ; preds = %52, %59, %46
  %63 = phi i64 [ 0, %46 ], [ %53, %52 ], [ 0, %59 ]
  %64 = getelementptr inbounds double, ptr %6, i64 %63
  store double %27, ptr %64, align 8, !tbaa !11
  br label %65

65:                                               ; preds = %62, %25
  %66 = getelementptr inbounds nuw i8, ptr %26, i64 8
  %67 = icmp ult ptr %66, %2
  br i1 %67, label %25, label %102, !llvm.loop !40

68:                                               ; preds = %31
  %69 = icmp eq i64 %41, %20
  %70 = select i1 %18, i1 %69, i1 false
  br i1 %70, label %48, label %46

71:                                               ; preds = %9
  %72 = getelementptr inbounds nuw i8, ptr %6, i64 8
  br i1 %18, label %75, label %73

73:                                               ; preds = %71
  %74 = load double, ptr %6, align 8, !tbaa !11
  br label %103

75:                                               ; preds = %71
  %76 = icmp eq i64 %19, 0
  br i1 %76, label %79, label %77

77:                                               ; preds = %75
  %78 = load double, ptr %6, align 8, !tbaa !11
  br label %92

79:                                               ; preds = %75, %89
  %80 = phi ptr [ %90, %89 ], [ %7, %75 ]
  %81 = load double, ptr %80, align 8, !tbaa !11
  %82 = load double, ptr %6, align 8, !tbaa !11
  %83 = fcmp olt double %81, %82
  br i1 %83, label %84, label %89

84:                                               ; preds = %79
  store double %82, ptr %80, align 8, !tbaa !11
  %85 = load double, ptr %72, align 8, !tbaa !11
  store double %85, ptr %6, align 8, !tbaa !11
  %86 = fcmp uge double %85, %81
  %87 = zext i1 %86 to i64
  %88 = getelementptr inbounds nuw double, ptr %6, i64 %87
  store double %81, ptr %88, align 8, !tbaa !11
  br label %89

89:                                               ; preds = %84, %79
  %90 = getelementptr inbounds nuw i8, ptr %80, i64 8
  %91 = icmp ult ptr %90, %2
  br i1 %91, label %79, label %102, !llvm.loop !40

92:                                               ; preds = %77, %98
  %93 = phi double [ %99, %98 ], [ %78, %77 ]
  %94 = phi ptr [ %100, %98 ], [ %7, %77 ]
  %95 = load double, ptr %94, align 8, !tbaa !11
  %96 = fcmp olt double %95, %93
  br i1 %96, label %97, label %98

97:                                               ; preds = %92
  store double %93, ptr %94, align 8, !tbaa !11
  store double %95, ptr %6, align 8, !tbaa !11
  br label %98

98:                                               ; preds = %97, %92
  %99 = phi double [ %95, %97 ], [ %93, %92 ]
  %100 = getelementptr inbounds nuw i8, ptr %94, i64 8
  %101 = icmp ult ptr %100, %2
  br i1 %101, label %92, label %102, !llvm.loop !40

102:                                              ; preds = %109, %98, %89, %65, %4
  ret void

103:                                              ; preds = %73, %109
  %104 = phi double [ %110, %109 ], [ %74, %73 ]
  %105 = phi ptr [ %111, %109 ], [ %7, %73 ]
  %106 = load double, ptr %105, align 8, !tbaa !11
  %107 = fcmp olt double %106, %104
  br i1 %107, label %108, label %109

108:                                              ; preds = %103
  store double %104, ptr %105, align 8, !tbaa !11
  store double %106, ptr %6, align 8, !tbaa !11
  br label %109

109:                                              ; preds = %103, %108
  %110 = phi double [ %104, %103 ], [ %106, %108 ]
  %111 = getelementptr inbounds nuw i8, ptr %105, i64 8
  %112 = icmp ult ptr %111, %2
  br i1 %112, label %103, label %102, !llvm.loop !40
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZSt11__make_heapIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEENS0_5__ops15_Iter_less_iterEEvT_S9_RT0_(ptr %0, ptr %1, ptr noundef nonnull align 1 dereferenceable(1) %2) local_unnamed_addr #2 comdat {
  %4 = freeze ptr %0
  %5 = freeze ptr %1
  %6 = ptrtoint ptr %5 to i64
  %7 = ptrtoint ptr %4 to i64
  %8 = sub i64 %6, %7
  %9 = ashr exact i64 %8, 3
  %10 = icmp slt i64 %9, 2
  br i1 %10, label %103, label %11

11:                                               ; preds = %3
  %12 = add nsw i64 %9, -2
  %13 = lshr i64 %12, 1
  %14 = add nsw i64 %9, -1
  %15 = lshr i64 %14, 1
  %16 = and i64 %8, 8
  %17 = icmp eq i64 %16, 0
  %18 = lshr exact i64 %12, 1
  br i1 %17, label %19, label %23

19:                                               ; preds = %11
  %20 = or disjoint i64 %12, 1
  %21 = getelementptr inbounds nuw double, ptr %4, i64 %20
  %22 = getelementptr inbounds nuw double, ptr %4, i64 %18
  br label %60

23:                                               ; preds = %11, %55
  %24 = phi i64 [ %59, %55 ], [ %13, %11 ]
  %25 = getelementptr inbounds double, ptr %4, i64 %24
  %26 = load double, ptr %25, align 8, !tbaa !11
  %27 = icmp slt i64 %24, %15
  br i1 %27, label %28, label %55

28:                                               ; preds = %23, %28
  %29 = phi i64 [ %38, %28 ], [ %24, %23 ]
  %30 = shl i64 %29, 1
  %31 = add i64 %30, 2
  %32 = getelementptr inbounds double, ptr %4, i64 %31
  %33 = or disjoint i64 %30, 1
  %34 = getelementptr inbounds double, ptr %4, i64 %33
  %35 = load double, ptr %32, align 8, !tbaa !11
  %36 = load double, ptr %34, align 8, !tbaa !11
  %37 = fcmp olt double %35, %36
  %38 = select i1 %37, i64 %33, i64 %31
  %39 = getelementptr inbounds double, ptr %4, i64 %38
  %40 = load double, ptr %39, align 8, !tbaa !11
  %41 = getelementptr inbounds double, ptr %4, i64 %29
  store double %40, ptr %41, align 8, !tbaa !11
  %42 = icmp slt i64 %38, %15
  br i1 %42, label %28, label %43, !llvm.loop !38

43:                                               ; preds = %28
  %44 = icmp sgt i64 %38, %24
  br i1 %44, label %45, label %55

45:                                               ; preds = %43, %52
  %46 = phi i64 [ %48, %52 ], [ %38, %43 ]
  %47 = add nsw i64 %46, -1
  %48 = sdiv i64 %47, 2
  %49 = getelementptr inbounds nuw double, ptr %4, i64 %48
  %50 = load double, ptr %49, align 8, !tbaa !11
  %51 = fcmp olt double %50, %26
  br i1 %51, label %52, label %55

52:                                               ; preds = %45
  %53 = getelementptr inbounds nuw double, ptr %4, i64 %46
  store double %50, ptr %53, align 8, !tbaa !11
  %54 = icmp sgt i64 %48, %24
  br i1 %54, label %45, label %55, !llvm.loop !39

55:                                               ; preds = %45, %52, %23, %43
  %56 = phi i64 [ %38, %43 ], [ %24, %23 ], [ %48, %52 ], [ %46, %45 ]
  %57 = getelementptr inbounds nuw double, ptr %4, i64 %56
  store double %26, ptr %57, align 8, !tbaa !11
  %58 = icmp eq i64 %24, 0
  %59 = add nsw i64 %24, -1
  br i1 %58, label %103, label %23, !llvm.loop !41

60:                                               ; preds = %19, %98
  %61 = phi i64 [ %102, %98 ], [ %13, %19 ]
  %62 = getelementptr inbounds double, ptr %4, i64 %61
  %63 = load double, ptr %62, align 8, !tbaa !11
  %64 = icmp slt i64 %61, %15
  br i1 %64, label %65, label %80

65:                                               ; preds = %60, %65
  %66 = phi i64 [ %75, %65 ], [ %61, %60 ]
  %67 = shl i64 %66, 1
  %68 = add i64 %67, 2
  %69 = getelementptr inbounds double, ptr %4, i64 %68
  %70 = or disjoint i64 %67, 1
  %71 = getelementptr inbounds double, ptr %4, i64 %70
  %72 = load double, ptr %69, align 8, !tbaa !11
  %73 = load double, ptr %71, align 8, !tbaa !11
  %74 = fcmp olt double %72, %73
  %75 = select i1 %74, i64 %70, i64 %68
  %76 = getelementptr inbounds double, ptr %4, i64 %75
  %77 = load double, ptr %76, align 8, !tbaa !11
  %78 = getelementptr inbounds double, ptr %4, i64 %66
  store double %77, ptr %78, align 8, !tbaa !11
  %79 = icmp slt i64 %75, %15
  br i1 %79, label %65, label %80, !llvm.loop !38

80:                                               ; preds = %65, %60
  %81 = phi i64 [ %61, %60 ], [ %75, %65 ]
  %82 = icmp eq i64 %81, %18
  br i1 %82, label %83, label %85

83:                                               ; preds = %80
  %84 = load double, ptr %21, align 8, !tbaa !11
  store double %84, ptr %22, align 8, !tbaa !11
  br label %85

85:                                               ; preds = %83, %80
  %86 = phi i64 [ %20, %83 ], [ %81, %80 ]
  %87 = icmp sgt i64 %86, %61
  br i1 %87, label %88, label %98

88:                                               ; preds = %85, %95
  %89 = phi i64 [ %91, %95 ], [ %86, %85 ]
  %90 = add nsw i64 %89, -1
  %91 = sdiv i64 %90, 2
  %92 = getelementptr inbounds nuw double, ptr %4, i64 %91
  %93 = load double, ptr %92, align 8, !tbaa !11
  %94 = fcmp olt double %93, %63
  br i1 %94, label %95, label %98

95:                                               ; preds = %88
  %96 = getelementptr inbounds nuw double, ptr %4, i64 %89
  store double %93, ptr %96, align 8, !tbaa !11
  %97 = icmp sgt i64 %91, %61
  br i1 %97, label %88, label %98, !llvm.loop !39

98:                                               ; preds = %88, %95, %85
  %99 = phi i64 [ %86, %85 ], [ %91, %95 ], [ %89, %88 ]
  %100 = getelementptr inbounds nuw double, ptr %4, i64 %99
  store double %63, ptr %100, align 8, !tbaa !11
  %101 = icmp eq i64 %61, 0
  %102 = add nsw i64 %61, -1
  br i1 %101, label %103, label %60, !llvm.loop !41

103:                                              ; preds = %55, %98, %3
  ret void
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memmove.p0.p0.i64(ptr writeonly captures(none), ptr readonly captures(none), i64, i1 immarg) #8

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.ctlz.i64(i64, i1 immarg) #9

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #11

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #12

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.umax.i64(i64, i64) #13

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.umin.i64(i64, i64) #13

attributes #0 = { mustprogress norecurse uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { nobuiltin nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { cold noreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { nobuiltin allocsize(0) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #8 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #9 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #10 = { mustprogress nocallback nofree nounwind willreturn memory(errnomem: write) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #11 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
attributes #12 = { nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #13 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #14 = { nounwind }
attributes #15 = { cold noreturn }
attributes #16 = { builtin allocsize(0) }
attributes #17 = { builtin nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"p1 omnipotent char", !8, i64 0}
!8 = !{!"any pointer", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C++ TBAA"}
!11 = !{!12, !12, i64 0}
!12 = !{!"double", !9, i64 0}
!13 = distinct !{!13, !14}
!14 = !{!"llvm.loop.mustprogress"}
!15 = !{!16, !12, i64 0}
!16 = !{!"_ZTS7momentsIdE", !12, i64 0, !12, i64 8, !12, i64 16, !12, i64 24, !12, i64 32, !12, i64 40, !12, i64 48}
!17 = !{!16, !12, i64 8}
!18 = !{!16, !12, i64 16}
!19 = !{!16, !12, i64 24}
!20 = !{!16, !12, i64 32}
!21 = !{!16, !12, i64 40}
!22 = !{!16, !12, i64 48}
!23 = distinct !{!23, !14, !24, !25}
!24 = !{!"llvm.loop.isvectorized", i32 1}
!25 = !{!"llvm.loop.unroll.runtime.disable"}
!26 = distinct !{!26, !14, !24}
!27 = !{!28, !28, i64 0}
!28 = !{!"int", !9, i64 0}
!29 = distinct !{!29, !14}
!30 = distinct !{!30, !14}
!31 = distinct !{!31, !14}
!32 = distinct !{!32, !14}
!33 = distinct !{!33, !14}
!34 = distinct !{!34, !14}
!35 = !{!"branch_weights", !"expected", i32 2000, i32 1}
!36 = distinct !{!36, !14}
!37 = distinct !{!37, !14}
!38 = distinct !{!38, !14}
!39 = distinct !{!39, !14}
!40 = distinct !{!40, !14}
!41 = distinct !{!41, !14}
