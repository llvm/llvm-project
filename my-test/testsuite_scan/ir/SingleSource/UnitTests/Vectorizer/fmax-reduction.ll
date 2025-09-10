; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/Vectorizer/fmax-reduction.cpp'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/Vectorizer/fmax-reduction.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

module asm ".globl _ZSt21ios_base_library_initv"

%"class.std::mersenne_twister_engine" = type { [624 x i64], i64 }
%"class.std::basic_ostream" = type { ptr, %"class.std::basic_ios" }
%"class.std::basic_ios" = type { %"class.std::ios_base", ptr, i8, i8, ptr, ptr, ptr, ptr }
%"class.std::ios_base" = type { ptr, i64, i64, i32, i32, i32, ptr, %"struct.std::ios_base::_Words", [8 x %"struct.std::ios_base::_Words"], i32, ptr, %"class.std::locale" }
%"struct.std::ios_base::_Words" = type { ptr, i64 }
%"class.std::locale" = type { ptr }
%"class.std::function" = type { %"class.std::_Function_base", ptr }
%"class.std::_Function_base" = type { %"union.std::_Any_data", ptr }
%"union.std::_Any_data" = type { %"union.std::_Nocopy_types" }
%"union.std::_Nocopy_types" = type { { i64, i64 } }
%"struct.__gnu_cxx::__ops::_Iter_less_iter" = type { i8 }

$__clang_call_terminate = comdat any

$_ZSt16__introsort_loopIPflN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_ = comdat any

$_ZSt22__final_insertion_sortIPfN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_ = comdat any

$_ZSt11__make_heapIPfN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_RT0_ = comdat any

@_ZL3rng = internal unnamed_addr global %"class.std::mersenne_twister_engine" zeroinitializer, align 8
@.str = private unnamed_addr constant [20 x i8] c"fmaxnum_start_neg_2\00", align 1
@.str.1 = private unnamed_addr constant [18 x i8] c"fmaxnum_start_min\00", align 1
@.str.2 = private unnamed_addr constant [25 x i8] c"fmaxnum_start_denorm_min\00", align 1
@.str.3 = private unnamed_addr constant [21 x i8] c"fmaxnum_start_is_nan\00", align 1
@.str.4 = private unnamed_addr constant [24 x i8] c"fmax_strict_start_neg_2\00", align 1
@.str.5 = private unnamed_addr constant [22 x i8] c"fmax_strict_start_min\00", align 1
@.str.6 = private unnamed_addr constant [29 x i8] c"fmax_strict_start_denorm_min\00", align 1
@.str.7 = private unnamed_addr constant [22 x i8] c"fmax_strict_start_nan\00", align 1
@.str.8 = private unnamed_addr constant [28 x i8] c"fmax_non_strict_start_neg_2\00", align 1
@.str.9 = private unnamed_addr constant [28 x i8] c"fmax_cmp_max_gt_start_neg_2\00", align 1
@.str.10 = private unnamed_addr constant [28 x i8] c"fmax_cmp_max_lt_start_neg_2\00", align 1
@.str.11 = private unnamed_addr constant [33 x i8] c"fmax_cmp_max_lt_start_denorm_min\00", align 1
@.str.12 = private unnamed_addr constant [30 x i8] c"fmax_cmp_max_lt_start_neg_nan\00", align 1
@_ZSt4cout = external global %"class.std::basic_ostream", align 8
@.str.13 = private unnamed_addr constant [10 x i8] c"Checking \00", align 1
@.str.14 = private unnamed_addr constant [2 x i8] c"\0A\00", align 1
@.str.15 = private unnamed_addr constant [7 x i8] c"sorted\00", align 1
@.str.16 = private unnamed_addr constant [15 x i8] c"reverse-sorted\00", align 1
@.str.17 = private unnamed_addr constant [8 x i8] c"all-max\00", align 1
@.str.18 = private unnamed_addr constant [8 x i8] c"all-min\00", align 1
@.str.19 = private unnamed_addr constant [10 x i8] c"denormals\00", align 1
@.str.20 = private unnamed_addr constant [10 x i8] c"all-zeros\00", align 1
@.str.21 = private unnamed_addr constant [4 x i8] c"NaN\00", align 1
@.str.22 = private unnamed_addr constant [13 x i8] c"signed-zeros\00", align 1
@.str.23 = private unnamed_addr constant [5 x i8] c"full\00", align 1
@.str.24 = private unnamed_addr constant [14 x i8] c"full-with-nan\00", align 1
@.str.25 = private unnamed_addr constant [23 x i8] c"full-with-multiple-nan\00", align 1
@.str.26 = private unnamed_addr constant [9 x i8] c"infinity\00", align 1
@_ZSt4cerr = external global %"class.std::basic_ostream", align 8
@.str.27 = private unnamed_addr constant [12 x i8] c"Miscompare \00", align 1
@.str.28 = private unnamed_addr constant [3 x i8] c": \00", align 1
@.str.29 = private unnamed_addr constant [5 x i8] c" != \00", align 1
@"_ZTIZ4mainE3$_0" = internal constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @"_ZTSZ4mainE3$_0" }, align 8
@_ZTVN10__cxxabiv117__class_type_infoE = external global [0 x ptr]
@"_ZTSZ4mainE3$_0" = internal constant [12 x i8] c"Z4mainE3$_0\00", align 1
@"_ZTIZ4mainE3$_1" = internal constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @"_ZTSZ4mainE3$_1" }, align 8
@"_ZTSZ4mainE3$_1" = internal constant [12 x i8] c"Z4mainE3$_1\00", align 1
@"_ZTIZ4mainE3$_2" = internal constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @"_ZTSZ4mainE3$_2" }, align 8
@"_ZTSZ4mainE3$_2" = internal constant [12 x i8] c"Z4mainE3$_2\00", align 1
@"_ZTIZ4mainE3$_3" = internal constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @"_ZTSZ4mainE3$_3" }, align 8
@"_ZTSZ4mainE3$_3" = internal constant [12 x i8] c"Z4mainE3$_3\00", align 1
@"_ZTIZ4mainE3$_4" = internal constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @"_ZTSZ4mainE3$_4" }, align 8
@"_ZTSZ4mainE3$_4" = internal constant [12 x i8] c"Z4mainE3$_4\00", align 1
@"_ZTIZ4mainE3$_5" = internal constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @"_ZTSZ4mainE3$_5" }, align 8
@"_ZTSZ4mainE3$_5" = internal constant [12 x i8] c"Z4mainE3$_5\00", align 1
@"_ZTIZ4mainE3$_6" = internal constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @"_ZTSZ4mainE3$_6" }, align 8
@"_ZTSZ4mainE3$_6" = internal constant [12 x i8] c"Z4mainE3$_6\00", align 1
@"_ZTIZ4mainE3$_7" = internal constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @"_ZTSZ4mainE3$_7" }, align 8
@"_ZTSZ4mainE3$_7" = internal constant [12 x i8] c"Z4mainE3$_7\00", align 1
@"_ZTIZ4mainE3$_8" = internal constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @"_ZTSZ4mainE3$_8" }, align 8
@"_ZTSZ4mainE3$_8" = internal constant [12 x i8] c"Z4mainE3$_8\00", align 1
@"_ZTIZ4mainE3$_9" = internal constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @"_ZTSZ4mainE3$_9" }, align 8
@"_ZTSZ4mainE3$_9" = internal constant [12 x i8] c"Z4mainE3$_9\00", align 1
@"_ZTIZ4mainE4$_10" = internal constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @"_ZTSZ4mainE4$_10" }, align 8
@"_ZTSZ4mainE4$_10" = internal constant [13 x i8] c"Z4mainE4$_10\00", align 1
@"_ZTIZ4mainE4$_11" = internal constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @"_ZTSZ4mainE4$_11" }, align 8
@"_ZTSZ4mainE4$_11" = internal constant [13 x i8] c"Z4mainE4$_11\00", align 1
@"_ZTIZ4mainE4$_12" = internal constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @"_ZTSZ4mainE4$_12" }, align 8
@"_ZTSZ4mainE4$_12" = internal constant [13 x i8] c"Z4mainE4$_12\00", align 1
@"_ZTIZ4mainE4$_13" = internal constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @"_ZTSZ4mainE4$_13" }, align 8
@"_ZTSZ4mainE4$_13" = internal constant [13 x i8] c"Z4mainE4$_13\00", align 1
@"_ZTIZ4mainE4$_14" = internal constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @"_ZTSZ4mainE4$_14" }, align 8
@"_ZTSZ4mainE4$_14" = internal constant [13 x i8] c"Z4mainE4$_14\00", align 1
@"_ZTIZ4mainE4$_15" = internal constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @"_ZTSZ4mainE4$_15" }, align 8
@"_ZTSZ4mainE4$_15" = internal constant [13 x i8] c"Z4mainE4$_15\00", align 1
@"_ZTIZ4mainE4$_16" = internal constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @"_ZTSZ4mainE4$_16" }, align 8
@"_ZTSZ4mainE4$_16" = internal constant [13 x i8] c"Z4mainE4$_16\00", align 1
@"_ZTIZ4mainE4$_17" = internal constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @"_ZTSZ4mainE4$_17" }, align 8
@"_ZTSZ4mainE4$_17" = internal constant [13 x i8] c"Z4mainE4$_17\00", align 1
@"_ZTIZ4mainE4$_18" = internal constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @"_ZTSZ4mainE4$_18" }, align 8
@"_ZTSZ4mainE4$_18" = internal constant [13 x i8] c"Z4mainE4$_18\00", align 1
@"_ZTIZ4mainE4$_19" = internal constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @"_ZTSZ4mainE4$_19" }, align 8
@"_ZTSZ4mainE4$_19" = internal constant [13 x i8] c"Z4mainE4$_19\00", align 1
@"_ZTIZ4mainE4$_20" = internal constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @"_ZTSZ4mainE4$_20" }, align 8
@"_ZTSZ4mainE4$_20" = internal constant [13 x i8] c"Z4mainE4$_20\00", align 1
@"_ZTIZ4mainE4$_21" = internal constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @"_ZTSZ4mainE4$_21" }, align 8
@"_ZTSZ4mainE4$_21" = internal constant [13 x i8] c"Z4mainE4$_21\00", align 1
@"_ZTIZ4mainE4$_22" = internal constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @"_ZTSZ4mainE4$_22" }, align 8
@"_ZTSZ4mainE4$_22" = internal constant [13 x i8] c"Z4mainE4$_22\00", align 1
@"_ZTIZ4mainE4$_23" = internal constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @"_ZTSZ4mainE4$_23" }, align 8
@"_ZTSZ4mainE4$_23" = internal constant [13 x i8] c"Z4mainE4$_23\00", align 1
@"_ZTIZ4mainE4$_24" = internal constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @"_ZTSZ4mainE4$_24" }, align 8
@"_ZTSZ4mainE4$_24" = internal constant [13 x i8] c"Z4mainE4$_24\00", align 1
@"_ZTIZ4mainE4$_25" = internal constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @"_ZTSZ4mainE4$_25" }, align 8
@"_ZTSZ4mainE4$_25" = internal constant [13 x i8] c"Z4mainE4$_25\00", align 1
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_fmax_reduction.cpp, ptr null }]

; Function Attrs: mustprogress norecurse uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 personality ptr @__gxx_personality_v0 {
  %1 = alloca %"class.std::mersenne_twister_engine", align 8
  %2 = alloca %"class.std::function", align 8
  %3 = alloca %"class.std::function", align 8
  %4 = alloca %"class.std::function", align 8
  %5 = alloca %"class.std::function", align 8
  %6 = alloca %"class.std::function", align 8
  %7 = alloca %"class.std::function", align 8
  %8 = alloca %"class.std::function", align 8
  %9 = alloca %"class.std::function", align 8
  %10 = alloca %"class.std::function", align 8
  %11 = alloca %"class.std::function", align 8
  %12 = alloca %"class.std::function", align 8
  %13 = alloca %"class.std::function", align 8
  %14 = alloca %"class.std::function", align 8
  %15 = alloca %"class.std::function", align 8
  %16 = alloca %"class.std::function", align 8
  %17 = alloca %"class.std::function", align 8
  %18 = alloca %"class.std::function", align 8
  %19 = alloca %"class.std::function", align 8
  %20 = alloca %"class.std::function", align 8
  %21 = alloca %"class.std::function", align 8
  %22 = alloca %"class.std::function", align 8
  %23 = alloca %"class.std::function", align 8
  %24 = alloca %"class.std::function", align 8
  %25 = alloca %"class.std::function", align 8
  %26 = alloca %"class.std::function", align 8
  %27 = alloca %"class.std::function", align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #20
  store i64 15, ptr %1, align 8, !tbaa !6
  br label %28

28:                                               ; preds = %28, %0
  %29 = phi i64 [ 15, %0 ], [ %36, %28 ]
  %30 = phi i64 [ 1, %0 ], [ %37, %28 ]
  %31 = getelementptr i64, ptr %1, i64 %30
  %32 = lshr i64 %29, 30
  %33 = xor i64 %32, %29
  %34 = mul nuw nsw i64 %33, 1812433253
  %35 = add nuw i64 %34, %30
  %36 = and i64 %35, 4294967295
  store i64 %36, ptr %31, align 8, !tbaa !6
  %37 = add nuw nsw i64 %30, 1
  %38 = icmp eq i64 %37, 624
  br i1 %38, label %39, label %28, !llvm.loop !10

39:                                               ; preds = %28
  %40 = getelementptr inbounds nuw i8, ptr %1, i64 4992
  store i64 624, ptr %40, align 8, !tbaa !12
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(5000) @_ZL3rng, ptr noundef nonnull align 8 dereferenceable(5000) %1, i64 5000, i1 false), !tbaa.struct !14
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #20
  %41 = getelementptr inbounds nuw i8, ptr %2, i64 16
  %42 = getelementptr inbounds nuw i8, ptr %2, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %2, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFfPfjEZ4mainE3$_0E9_M_invokeERKSt9_Any_dataOS0_Oj", ptr %42, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFfPfjEZ4mainE3$_0E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %41, align 8, !tbaa !20
  %43 = getelementptr inbounds nuw i8, ptr %3, i64 16
  %44 = getelementptr inbounds nuw i8, ptr %3, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %3, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFfPfjEZ4mainE3$_1E9_M_invokeERKSt9_Any_dataOS0_Oj", ptr %44, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFfPfjEZ4mainE3$_1E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %43, align 8, !tbaa !20
  invoke fastcc void @_ZL19checkVectorFunctionIfEvSt8functionIFT_PS1_jEES4_PKc(ptr noundef %2, ptr noundef %3, ptr noundef nonnull @.str)
          to label %45 unwind label %314

45:                                               ; preds = %39
  %46 = load ptr, ptr %43, align 8, !tbaa !20
  %47 = icmp eq ptr %46, null
  br i1 %47, label %53, label %48

48:                                               ; preds = %45
  %49 = invoke noundef i1 %46(ptr noundef nonnull align 8 dereferenceable(32) %3, ptr noundef nonnull align 8 dereferenceable(32) %3, i32 noundef 3)
          to label %53 unwind label %50

50:                                               ; preds = %48
  %51 = landingpad { ptr, i32 }
          catch ptr null
  %52 = extractvalue { ptr, i32 } %51, 0
  call void @__clang_call_terminate(ptr %52) #21
  unreachable

53:                                               ; preds = %45, %48
  %54 = load ptr, ptr %41, align 8, !tbaa !20
  %55 = icmp eq ptr %54, null
  br i1 %55, label %61, label %56

56:                                               ; preds = %53
  %57 = invoke noundef i1 %54(ptr noundef nonnull align 8 dereferenceable(32) %2, ptr noundef nonnull align 8 dereferenceable(32) %2, i32 noundef 3)
          to label %61 unwind label %58

58:                                               ; preds = %56
  %59 = landingpad { ptr, i32 }
          catch ptr null
  %60 = extractvalue { ptr, i32 } %59, 0
  call void @__clang_call_terminate(ptr %60) #21
  unreachable

61:                                               ; preds = %53, %56
  %62 = getelementptr inbounds nuw i8, ptr %4, i64 16
  %63 = getelementptr inbounds nuw i8, ptr %4, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %4, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFfPfjEZ4mainE3$_2E9_M_invokeERKSt9_Any_dataOS0_Oj", ptr %63, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFfPfjEZ4mainE3$_2E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %62, align 8, !tbaa !20
  %64 = getelementptr inbounds nuw i8, ptr %5, i64 16
  %65 = getelementptr inbounds nuw i8, ptr %5, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %5, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFfPfjEZ4mainE3$_3E9_M_invokeERKSt9_Any_dataOS0_Oj", ptr %65, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFfPfjEZ4mainE3$_3E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %64, align 8, !tbaa !20
  invoke fastcc void @_ZL19checkVectorFunctionIfEvSt8functionIFT_PS1_jEES4_PKc(ptr noundef %4, ptr noundef %5, ptr noundef nonnull @.str.1)
          to label %66 unwind label %331

66:                                               ; preds = %61
  %67 = load ptr, ptr %64, align 8, !tbaa !20
  %68 = icmp eq ptr %67, null
  br i1 %68, label %74, label %69

69:                                               ; preds = %66
  %70 = invoke noundef i1 %67(ptr noundef nonnull align 8 dereferenceable(32) %5, ptr noundef nonnull align 8 dereferenceable(32) %5, i32 noundef 3)
          to label %74 unwind label %71

71:                                               ; preds = %69
  %72 = landingpad { ptr, i32 }
          catch ptr null
  %73 = extractvalue { ptr, i32 } %72, 0
  call void @__clang_call_terminate(ptr %73) #21
  unreachable

74:                                               ; preds = %66, %69
  %75 = load ptr, ptr %62, align 8, !tbaa !20
  %76 = icmp eq ptr %75, null
  br i1 %76, label %82, label %77

77:                                               ; preds = %74
  %78 = invoke noundef i1 %75(ptr noundef nonnull align 8 dereferenceable(32) %4, ptr noundef nonnull align 8 dereferenceable(32) %4, i32 noundef 3)
          to label %82 unwind label %79

79:                                               ; preds = %77
  %80 = landingpad { ptr, i32 }
          catch ptr null
  %81 = extractvalue { ptr, i32 } %80, 0
  call void @__clang_call_terminate(ptr %81) #21
  unreachable

82:                                               ; preds = %74, %77
  %83 = getelementptr inbounds nuw i8, ptr %6, i64 16
  %84 = getelementptr inbounds nuw i8, ptr %6, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %6, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFfPfjEZ4mainE3$_4E9_M_invokeERKSt9_Any_dataOS0_Oj", ptr %84, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFfPfjEZ4mainE3$_4E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %83, align 8, !tbaa !20
  %85 = getelementptr inbounds nuw i8, ptr %7, i64 16
  %86 = getelementptr inbounds nuw i8, ptr %7, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %7, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFfPfjEZ4mainE3$_5E9_M_invokeERKSt9_Any_dataOS0_Oj", ptr %86, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFfPfjEZ4mainE3$_5E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %85, align 8, !tbaa !20
  invoke fastcc void @_ZL19checkVectorFunctionIfEvSt8functionIFT_PS1_jEES4_PKc(ptr noundef %6, ptr noundef %7, ptr noundef nonnull @.str.2)
          to label %87 unwind label %348

87:                                               ; preds = %82
  %88 = load ptr, ptr %85, align 8, !tbaa !20
  %89 = icmp eq ptr %88, null
  br i1 %89, label %95, label %90

90:                                               ; preds = %87
  %91 = invoke noundef i1 %88(ptr noundef nonnull align 8 dereferenceable(32) %7, ptr noundef nonnull align 8 dereferenceable(32) %7, i32 noundef 3)
          to label %95 unwind label %92

92:                                               ; preds = %90
  %93 = landingpad { ptr, i32 }
          catch ptr null
  %94 = extractvalue { ptr, i32 } %93, 0
  call void @__clang_call_terminate(ptr %94) #21
  unreachable

95:                                               ; preds = %87, %90
  %96 = load ptr, ptr %83, align 8, !tbaa !20
  %97 = icmp eq ptr %96, null
  br i1 %97, label %103, label %98

98:                                               ; preds = %95
  %99 = invoke noundef i1 %96(ptr noundef nonnull align 8 dereferenceable(32) %6, ptr noundef nonnull align 8 dereferenceable(32) %6, i32 noundef 3)
          to label %103 unwind label %100

100:                                              ; preds = %98
  %101 = landingpad { ptr, i32 }
          catch ptr null
  %102 = extractvalue { ptr, i32 } %101, 0
  call void @__clang_call_terminate(ptr %102) #21
  unreachable

103:                                              ; preds = %95, %98
  %104 = getelementptr inbounds nuw i8, ptr %8, i64 16
  %105 = getelementptr inbounds nuw i8, ptr %8, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %8, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFfPfjEZ4mainE3$_6E9_M_invokeERKSt9_Any_dataOS0_Oj", ptr %105, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFfPfjEZ4mainE3$_6E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %104, align 8, !tbaa !20
  %106 = getelementptr inbounds nuw i8, ptr %9, i64 16
  %107 = getelementptr inbounds nuw i8, ptr %9, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %9, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFfPfjEZ4mainE3$_7E9_M_invokeERKSt9_Any_dataOS0_Oj", ptr %107, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFfPfjEZ4mainE3$_7E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %106, align 8, !tbaa !20
  invoke fastcc void @_ZL19checkVectorFunctionIfEvSt8functionIFT_PS1_jEES4_PKc(ptr noundef %8, ptr noundef %9, ptr noundef nonnull @.str.3)
          to label %108 unwind label %365

108:                                              ; preds = %103
  %109 = load ptr, ptr %106, align 8, !tbaa !20
  %110 = icmp eq ptr %109, null
  br i1 %110, label %116, label %111

111:                                              ; preds = %108
  %112 = invoke noundef i1 %109(ptr noundef nonnull align 8 dereferenceable(32) %9, ptr noundef nonnull align 8 dereferenceable(32) %9, i32 noundef 3)
          to label %116 unwind label %113

113:                                              ; preds = %111
  %114 = landingpad { ptr, i32 }
          catch ptr null
  %115 = extractvalue { ptr, i32 } %114, 0
  call void @__clang_call_terminate(ptr %115) #21
  unreachable

116:                                              ; preds = %108, %111
  %117 = load ptr, ptr %104, align 8, !tbaa !20
  %118 = icmp eq ptr %117, null
  br i1 %118, label %124, label %119

119:                                              ; preds = %116
  %120 = invoke noundef i1 %117(ptr noundef nonnull align 8 dereferenceable(32) %8, ptr noundef nonnull align 8 dereferenceable(32) %8, i32 noundef 3)
          to label %124 unwind label %121

121:                                              ; preds = %119
  %122 = landingpad { ptr, i32 }
          catch ptr null
  %123 = extractvalue { ptr, i32 } %122, 0
  call void @__clang_call_terminate(ptr %123) #21
  unreachable

124:                                              ; preds = %116, %119
  %125 = getelementptr inbounds nuw i8, ptr %10, i64 16
  %126 = getelementptr inbounds nuw i8, ptr %10, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %10, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFfPfjEZ4mainE3$_8E9_M_invokeERKSt9_Any_dataOS0_Oj", ptr %126, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFfPfjEZ4mainE3$_8E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %125, align 8, !tbaa !20
  %127 = getelementptr inbounds nuw i8, ptr %11, i64 16
  %128 = getelementptr inbounds nuw i8, ptr %11, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %11, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFfPfjEZ4mainE3$_9E9_M_invokeERKSt9_Any_dataOS0_Oj", ptr %128, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFfPfjEZ4mainE3$_9E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %127, align 8, !tbaa !20
  invoke fastcc void @_ZL19checkVectorFunctionIfEvSt8functionIFT_PS1_jEES4_PKc(ptr noundef %10, ptr noundef %11, ptr noundef nonnull @.str.4)
          to label %129 unwind label %382

129:                                              ; preds = %124
  %130 = load ptr, ptr %127, align 8, !tbaa !20
  %131 = icmp eq ptr %130, null
  br i1 %131, label %137, label %132

132:                                              ; preds = %129
  %133 = invoke noundef i1 %130(ptr noundef nonnull align 8 dereferenceable(32) %11, ptr noundef nonnull align 8 dereferenceable(32) %11, i32 noundef 3)
          to label %137 unwind label %134

134:                                              ; preds = %132
  %135 = landingpad { ptr, i32 }
          catch ptr null
  %136 = extractvalue { ptr, i32 } %135, 0
  call void @__clang_call_terminate(ptr %136) #21
  unreachable

137:                                              ; preds = %129, %132
  %138 = load ptr, ptr %125, align 8, !tbaa !20
  %139 = icmp eq ptr %138, null
  br i1 %139, label %145, label %140

140:                                              ; preds = %137
  %141 = invoke noundef i1 %138(ptr noundef nonnull align 8 dereferenceable(32) %10, ptr noundef nonnull align 8 dereferenceable(32) %10, i32 noundef 3)
          to label %145 unwind label %142

142:                                              ; preds = %140
  %143 = landingpad { ptr, i32 }
          catch ptr null
  %144 = extractvalue { ptr, i32 } %143, 0
  call void @__clang_call_terminate(ptr %144) #21
  unreachable

145:                                              ; preds = %137, %140
  %146 = getelementptr inbounds nuw i8, ptr %12, i64 16
  %147 = getelementptr inbounds nuw i8, ptr %12, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %12, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFfPfjEZ4mainE4$_10E9_M_invokeERKSt9_Any_dataOS0_Oj", ptr %147, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFfPfjEZ4mainE4$_10E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %146, align 8, !tbaa !20
  %148 = getelementptr inbounds nuw i8, ptr %13, i64 16
  %149 = getelementptr inbounds nuw i8, ptr %13, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %13, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFfPfjEZ4mainE4$_11E9_M_invokeERKSt9_Any_dataOS0_Oj", ptr %149, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFfPfjEZ4mainE4$_11E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %148, align 8, !tbaa !20
  invoke fastcc void @_ZL19checkVectorFunctionIfEvSt8functionIFT_PS1_jEES4_PKc(ptr noundef %12, ptr noundef %13, ptr noundef nonnull @.str.5)
          to label %150 unwind label %399

150:                                              ; preds = %145
  %151 = load ptr, ptr %148, align 8, !tbaa !20
  %152 = icmp eq ptr %151, null
  br i1 %152, label %158, label %153

153:                                              ; preds = %150
  %154 = invoke noundef i1 %151(ptr noundef nonnull align 8 dereferenceable(32) %13, ptr noundef nonnull align 8 dereferenceable(32) %13, i32 noundef 3)
          to label %158 unwind label %155

155:                                              ; preds = %153
  %156 = landingpad { ptr, i32 }
          catch ptr null
  %157 = extractvalue { ptr, i32 } %156, 0
  call void @__clang_call_terminate(ptr %157) #21
  unreachable

158:                                              ; preds = %150, %153
  %159 = load ptr, ptr %146, align 8, !tbaa !20
  %160 = icmp eq ptr %159, null
  br i1 %160, label %166, label %161

161:                                              ; preds = %158
  %162 = invoke noundef i1 %159(ptr noundef nonnull align 8 dereferenceable(32) %12, ptr noundef nonnull align 8 dereferenceable(32) %12, i32 noundef 3)
          to label %166 unwind label %163

163:                                              ; preds = %161
  %164 = landingpad { ptr, i32 }
          catch ptr null
  %165 = extractvalue { ptr, i32 } %164, 0
  call void @__clang_call_terminate(ptr %165) #21
  unreachable

166:                                              ; preds = %158, %161
  %167 = getelementptr inbounds nuw i8, ptr %14, i64 16
  %168 = getelementptr inbounds nuw i8, ptr %14, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %14, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFfPfjEZ4mainE4$_12E9_M_invokeERKSt9_Any_dataOS0_Oj", ptr %168, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFfPfjEZ4mainE4$_12E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %167, align 8, !tbaa !20
  %169 = getelementptr inbounds nuw i8, ptr %15, i64 16
  %170 = getelementptr inbounds nuw i8, ptr %15, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %15, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFfPfjEZ4mainE4$_13E9_M_invokeERKSt9_Any_dataOS0_Oj", ptr %170, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFfPfjEZ4mainE4$_13E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %169, align 8, !tbaa !20
  invoke fastcc void @_ZL19checkVectorFunctionIfEvSt8functionIFT_PS1_jEES4_PKc(ptr noundef %14, ptr noundef %15, ptr noundef nonnull @.str.6)
          to label %171 unwind label %416

171:                                              ; preds = %166
  %172 = load ptr, ptr %169, align 8, !tbaa !20
  %173 = icmp eq ptr %172, null
  br i1 %173, label %179, label %174

174:                                              ; preds = %171
  %175 = invoke noundef i1 %172(ptr noundef nonnull align 8 dereferenceable(32) %15, ptr noundef nonnull align 8 dereferenceable(32) %15, i32 noundef 3)
          to label %179 unwind label %176

176:                                              ; preds = %174
  %177 = landingpad { ptr, i32 }
          catch ptr null
  %178 = extractvalue { ptr, i32 } %177, 0
  call void @__clang_call_terminate(ptr %178) #21
  unreachable

179:                                              ; preds = %171, %174
  %180 = load ptr, ptr %167, align 8, !tbaa !20
  %181 = icmp eq ptr %180, null
  br i1 %181, label %187, label %182

182:                                              ; preds = %179
  %183 = invoke noundef i1 %180(ptr noundef nonnull align 8 dereferenceable(32) %14, ptr noundef nonnull align 8 dereferenceable(32) %14, i32 noundef 3)
          to label %187 unwind label %184

184:                                              ; preds = %182
  %185 = landingpad { ptr, i32 }
          catch ptr null
  %186 = extractvalue { ptr, i32 } %185, 0
  call void @__clang_call_terminate(ptr %186) #21
  unreachable

187:                                              ; preds = %179, %182
  %188 = getelementptr inbounds nuw i8, ptr %16, i64 16
  %189 = getelementptr inbounds nuw i8, ptr %16, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %16, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFfPfjEZ4mainE4$_14E9_M_invokeERKSt9_Any_dataOS0_Oj", ptr %189, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFfPfjEZ4mainE4$_14E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %188, align 8, !tbaa !20
  %190 = getelementptr inbounds nuw i8, ptr %17, i64 16
  %191 = getelementptr inbounds nuw i8, ptr %17, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %17, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFfPfjEZ4mainE4$_15E9_M_invokeERKSt9_Any_dataOS0_Oj", ptr %191, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFfPfjEZ4mainE4$_15E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %190, align 8, !tbaa !20
  invoke fastcc void @_ZL19checkVectorFunctionIfEvSt8functionIFT_PS1_jEES4_PKc(ptr noundef %16, ptr noundef %17, ptr noundef nonnull @.str.7)
          to label %192 unwind label %433

192:                                              ; preds = %187
  %193 = load ptr, ptr %190, align 8, !tbaa !20
  %194 = icmp eq ptr %193, null
  br i1 %194, label %200, label %195

195:                                              ; preds = %192
  %196 = invoke noundef i1 %193(ptr noundef nonnull align 8 dereferenceable(32) %17, ptr noundef nonnull align 8 dereferenceable(32) %17, i32 noundef 3)
          to label %200 unwind label %197

197:                                              ; preds = %195
  %198 = landingpad { ptr, i32 }
          catch ptr null
  %199 = extractvalue { ptr, i32 } %198, 0
  call void @__clang_call_terminate(ptr %199) #21
  unreachable

200:                                              ; preds = %192, %195
  %201 = load ptr, ptr %188, align 8, !tbaa !20
  %202 = icmp eq ptr %201, null
  br i1 %202, label %208, label %203

203:                                              ; preds = %200
  %204 = invoke noundef i1 %201(ptr noundef nonnull align 8 dereferenceable(32) %16, ptr noundef nonnull align 8 dereferenceable(32) %16, i32 noundef 3)
          to label %208 unwind label %205

205:                                              ; preds = %203
  %206 = landingpad { ptr, i32 }
          catch ptr null
  %207 = extractvalue { ptr, i32 } %206, 0
  call void @__clang_call_terminate(ptr %207) #21
  unreachable

208:                                              ; preds = %200, %203
  %209 = getelementptr inbounds nuw i8, ptr %18, i64 16
  %210 = getelementptr inbounds nuw i8, ptr %18, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %18, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFfPfjEZ4mainE4$_16E9_M_invokeERKSt9_Any_dataOS0_Oj", ptr %210, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFfPfjEZ4mainE4$_16E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %209, align 8, !tbaa !20
  %211 = getelementptr inbounds nuw i8, ptr %19, i64 16
  %212 = getelementptr inbounds nuw i8, ptr %19, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %19, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFfPfjEZ4mainE4$_17E9_M_invokeERKSt9_Any_dataOS0_Oj", ptr %212, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFfPfjEZ4mainE4$_17E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %211, align 8, !tbaa !20
  invoke fastcc void @_ZL19checkVectorFunctionIfEvSt8functionIFT_PS1_jEES4_PKc(ptr noundef %18, ptr noundef %19, ptr noundef nonnull @.str.8)
          to label %213 unwind label %450

213:                                              ; preds = %208
  %214 = load ptr, ptr %211, align 8, !tbaa !20
  %215 = icmp eq ptr %214, null
  br i1 %215, label %221, label %216

216:                                              ; preds = %213
  %217 = invoke noundef i1 %214(ptr noundef nonnull align 8 dereferenceable(32) %19, ptr noundef nonnull align 8 dereferenceable(32) %19, i32 noundef 3)
          to label %221 unwind label %218

218:                                              ; preds = %216
  %219 = landingpad { ptr, i32 }
          catch ptr null
  %220 = extractvalue { ptr, i32 } %219, 0
  call void @__clang_call_terminate(ptr %220) #21
  unreachable

221:                                              ; preds = %213, %216
  %222 = load ptr, ptr %209, align 8, !tbaa !20
  %223 = icmp eq ptr %222, null
  br i1 %223, label %229, label %224

224:                                              ; preds = %221
  %225 = invoke noundef i1 %222(ptr noundef nonnull align 8 dereferenceable(32) %18, ptr noundef nonnull align 8 dereferenceable(32) %18, i32 noundef 3)
          to label %229 unwind label %226

226:                                              ; preds = %224
  %227 = landingpad { ptr, i32 }
          catch ptr null
  %228 = extractvalue { ptr, i32 } %227, 0
  call void @__clang_call_terminate(ptr %228) #21
  unreachable

229:                                              ; preds = %221, %224
  %230 = getelementptr inbounds nuw i8, ptr %20, i64 16
  %231 = getelementptr inbounds nuw i8, ptr %20, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %20, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFfPfjEZ4mainE4$_18E9_M_invokeERKSt9_Any_dataOS0_Oj", ptr %231, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFfPfjEZ4mainE4$_18E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %230, align 8, !tbaa !20
  %232 = getelementptr inbounds nuw i8, ptr %21, i64 16
  %233 = getelementptr inbounds nuw i8, ptr %21, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %21, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFfPfjEZ4mainE4$_19E9_M_invokeERKSt9_Any_dataOS0_Oj", ptr %233, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFfPfjEZ4mainE4$_19E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %232, align 8, !tbaa !20
  invoke fastcc void @_ZL19checkVectorFunctionIfEvSt8functionIFT_PS1_jEES4_PKc(ptr noundef %20, ptr noundef %21, ptr noundef nonnull @.str.9)
          to label %234 unwind label %467

234:                                              ; preds = %229
  %235 = load ptr, ptr %232, align 8, !tbaa !20
  %236 = icmp eq ptr %235, null
  br i1 %236, label %242, label %237

237:                                              ; preds = %234
  %238 = invoke noundef i1 %235(ptr noundef nonnull align 8 dereferenceable(32) %21, ptr noundef nonnull align 8 dereferenceable(32) %21, i32 noundef 3)
          to label %242 unwind label %239

239:                                              ; preds = %237
  %240 = landingpad { ptr, i32 }
          catch ptr null
  %241 = extractvalue { ptr, i32 } %240, 0
  call void @__clang_call_terminate(ptr %241) #21
  unreachable

242:                                              ; preds = %234, %237
  %243 = load ptr, ptr %230, align 8, !tbaa !20
  %244 = icmp eq ptr %243, null
  br i1 %244, label %250, label %245

245:                                              ; preds = %242
  %246 = invoke noundef i1 %243(ptr noundef nonnull align 8 dereferenceable(32) %20, ptr noundef nonnull align 8 dereferenceable(32) %20, i32 noundef 3)
          to label %250 unwind label %247

247:                                              ; preds = %245
  %248 = landingpad { ptr, i32 }
          catch ptr null
  %249 = extractvalue { ptr, i32 } %248, 0
  call void @__clang_call_terminate(ptr %249) #21
  unreachable

250:                                              ; preds = %242, %245
  %251 = getelementptr inbounds nuw i8, ptr %22, i64 16
  %252 = getelementptr inbounds nuw i8, ptr %22, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %22, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFfPfjEZ4mainE4$_20E9_M_invokeERKSt9_Any_dataOS0_Oj", ptr %252, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFfPfjEZ4mainE4$_20E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %251, align 8, !tbaa !20
  %253 = getelementptr inbounds nuw i8, ptr %23, i64 16
  %254 = getelementptr inbounds nuw i8, ptr %23, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %23, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFfPfjEZ4mainE4$_21E9_M_invokeERKSt9_Any_dataOS0_Oj", ptr %254, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFfPfjEZ4mainE4$_21E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %253, align 8, !tbaa !20
  invoke fastcc void @_ZL19checkVectorFunctionIfEvSt8functionIFT_PS1_jEES4_PKc(ptr noundef %22, ptr noundef %23, ptr noundef nonnull @.str.10)
          to label %255 unwind label %484

255:                                              ; preds = %250
  %256 = load ptr, ptr %253, align 8, !tbaa !20
  %257 = icmp eq ptr %256, null
  br i1 %257, label %263, label %258

258:                                              ; preds = %255
  %259 = invoke noundef i1 %256(ptr noundef nonnull align 8 dereferenceable(32) %23, ptr noundef nonnull align 8 dereferenceable(32) %23, i32 noundef 3)
          to label %263 unwind label %260

260:                                              ; preds = %258
  %261 = landingpad { ptr, i32 }
          catch ptr null
  %262 = extractvalue { ptr, i32 } %261, 0
  call void @__clang_call_terminate(ptr %262) #21
  unreachable

263:                                              ; preds = %255, %258
  %264 = load ptr, ptr %251, align 8, !tbaa !20
  %265 = icmp eq ptr %264, null
  br i1 %265, label %271, label %266

266:                                              ; preds = %263
  %267 = invoke noundef i1 %264(ptr noundef nonnull align 8 dereferenceable(32) %22, ptr noundef nonnull align 8 dereferenceable(32) %22, i32 noundef 3)
          to label %271 unwind label %268

268:                                              ; preds = %266
  %269 = landingpad { ptr, i32 }
          catch ptr null
  %270 = extractvalue { ptr, i32 } %269, 0
  call void @__clang_call_terminate(ptr %270) #21
  unreachable

271:                                              ; preds = %263, %266
  %272 = getelementptr inbounds nuw i8, ptr %24, i64 16
  %273 = getelementptr inbounds nuw i8, ptr %24, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %24, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFfPfjEZ4mainE4$_22E9_M_invokeERKSt9_Any_dataOS0_Oj", ptr %273, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFfPfjEZ4mainE4$_22E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %272, align 8, !tbaa !20
  %274 = getelementptr inbounds nuw i8, ptr %25, i64 16
  %275 = getelementptr inbounds nuw i8, ptr %25, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %25, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFfPfjEZ4mainE4$_23E9_M_invokeERKSt9_Any_dataOS0_Oj", ptr %275, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFfPfjEZ4mainE4$_23E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %274, align 8, !tbaa !20
  invoke fastcc void @_ZL19checkVectorFunctionIfEvSt8functionIFT_PS1_jEES4_PKc(ptr noundef %24, ptr noundef %25, ptr noundef nonnull @.str.11)
          to label %276 unwind label %501

276:                                              ; preds = %271
  %277 = load ptr, ptr %274, align 8, !tbaa !20
  %278 = icmp eq ptr %277, null
  br i1 %278, label %284, label %279

279:                                              ; preds = %276
  %280 = invoke noundef i1 %277(ptr noundef nonnull align 8 dereferenceable(32) %25, ptr noundef nonnull align 8 dereferenceable(32) %25, i32 noundef 3)
          to label %284 unwind label %281

281:                                              ; preds = %279
  %282 = landingpad { ptr, i32 }
          catch ptr null
  %283 = extractvalue { ptr, i32 } %282, 0
  call void @__clang_call_terminate(ptr %283) #21
  unreachable

284:                                              ; preds = %276, %279
  %285 = load ptr, ptr %272, align 8, !tbaa !20
  %286 = icmp eq ptr %285, null
  br i1 %286, label %292, label %287

287:                                              ; preds = %284
  %288 = invoke noundef i1 %285(ptr noundef nonnull align 8 dereferenceable(32) %24, ptr noundef nonnull align 8 dereferenceable(32) %24, i32 noundef 3)
          to label %292 unwind label %289

289:                                              ; preds = %287
  %290 = landingpad { ptr, i32 }
          catch ptr null
  %291 = extractvalue { ptr, i32 } %290, 0
  call void @__clang_call_terminate(ptr %291) #21
  unreachable

292:                                              ; preds = %284, %287
  %293 = getelementptr inbounds nuw i8, ptr %26, i64 16
  %294 = getelementptr inbounds nuw i8, ptr %26, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %26, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFfPfjEZ4mainE4$_24E9_M_invokeERKSt9_Any_dataOS0_Oj", ptr %294, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFfPfjEZ4mainE4$_24E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %293, align 8, !tbaa !20
  %295 = getelementptr inbounds nuw i8, ptr %27, i64 16
  %296 = getelementptr inbounds nuw i8, ptr %27, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %27, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFfPfjEZ4mainE4$_25E9_M_invokeERKSt9_Any_dataOS0_Oj", ptr %296, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFfPfjEZ4mainE4$_25E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %295, align 8, !tbaa !20
  invoke fastcc void @_ZL19checkVectorFunctionIfEvSt8functionIFT_PS1_jEES4_PKc(ptr noundef %26, ptr noundef %27, ptr noundef nonnull @.str.12)
          to label %297 unwind label %518

297:                                              ; preds = %292
  %298 = load ptr, ptr %295, align 8, !tbaa !20
  %299 = icmp eq ptr %298, null
  br i1 %299, label %305, label %300

300:                                              ; preds = %297
  %301 = invoke noundef i1 %298(ptr noundef nonnull align 8 dereferenceable(32) %27, ptr noundef nonnull align 8 dereferenceable(32) %27, i32 noundef 3)
          to label %305 unwind label %302

302:                                              ; preds = %300
  %303 = landingpad { ptr, i32 }
          catch ptr null
  %304 = extractvalue { ptr, i32 } %303, 0
  call void @__clang_call_terminate(ptr %304) #21
  unreachable

305:                                              ; preds = %297, %300
  %306 = load ptr, ptr %293, align 8, !tbaa !20
  %307 = icmp eq ptr %306, null
  br i1 %307, label %313, label %308

308:                                              ; preds = %305
  %309 = invoke noundef i1 %306(ptr noundef nonnull align 8 dereferenceable(32) %26, ptr noundef nonnull align 8 dereferenceable(32) %26, i32 noundef 3)
          to label %313 unwind label %310

310:                                              ; preds = %308
  %311 = landingpad { ptr, i32 }
          catch ptr null
  %312 = extractvalue { ptr, i32 } %311, 0
  call void @__clang_call_terminate(ptr %312) #21
  unreachable

313:                                              ; preds = %305, %308
  ret i32 0

314:                                              ; preds = %39
  %315 = landingpad { ptr, i32 }
          cleanup
  %316 = load ptr, ptr %43, align 8, !tbaa !20
  %317 = icmp eq ptr %316, null
  br i1 %317, label %323, label %318

318:                                              ; preds = %314
  %319 = invoke noundef i1 %316(ptr noundef nonnull align 8 dereferenceable(32) %3, ptr noundef nonnull align 8 dereferenceable(32) %3, i32 noundef 3)
          to label %323 unwind label %320

320:                                              ; preds = %318
  %321 = landingpad { ptr, i32 }
          catch ptr null
  %322 = extractvalue { ptr, i32 } %321, 0
  call void @__clang_call_terminate(ptr %322) #21
  unreachable

323:                                              ; preds = %314, %318
  %324 = load ptr, ptr %41, align 8, !tbaa !20
  %325 = icmp eq ptr %324, null
  br i1 %325, label %535, label %326

326:                                              ; preds = %323
  %327 = invoke noundef i1 %324(ptr noundef nonnull align 8 dereferenceable(32) %2, ptr noundef nonnull align 8 dereferenceable(32) %2, i32 noundef 3)
          to label %535 unwind label %328

328:                                              ; preds = %326
  %329 = landingpad { ptr, i32 }
          catch ptr null
  %330 = extractvalue { ptr, i32 } %329, 0
  call void @__clang_call_terminate(ptr %330) #21
  unreachable

331:                                              ; preds = %61
  %332 = landingpad { ptr, i32 }
          cleanup
  %333 = load ptr, ptr %64, align 8, !tbaa !20
  %334 = icmp eq ptr %333, null
  br i1 %334, label %340, label %335

335:                                              ; preds = %331
  %336 = invoke noundef i1 %333(ptr noundef nonnull align 8 dereferenceable(32) %5, ptr noundef nonnull align 8 dereferenceable(32) %5, i32 noundef 3)
          to label %340 unwind label %337

337:                                              ; preds = %335
  %338 = landingpad { ptr, i32 }
          catch ptr null
  %339 = extractvalue { ptr, i32 } %338, 0
  call void @__clang_call_terminate(ptr %339) #21
  unreachable

340:                                              ; preds = %331, %335
  %341 = load ptr, ptr %62, align 8, !tbaa !20
  %342 = icmp eq ptr %341, null
  br i1 %342, label %535, label %343

343:                                              ; preds = %340
  %344 = invoke noundef i1 %341(ptr noundef nonnull align 8 dereferenceable(32) %4, ptr noundef nonnull align 8 dereferenceable(32) %4, i32 noundef 3)
          to label %535 unwind label %345

345:                                              ; preds = %343
  %346 = landingpad { ptr, i32 }
          catch ptr null
  %347 = extractvalue { ptr, i32 } %346, 0
  call void @__clang_call_terminate(ptr %347) #21
  unreachable

348:                                              ; preds = %82
  %349 = landingpad { ptr, i32 }
          cleanup
  %350 = load ptr, ptr %85, align 8, !tbaa !20
  %351 = icmp eq ptr %350, null
  br i1 %351, label %357, label %352

352:                                              ; preds = %348
  %353 = invoke noundef i1 %350(ptr noundef nonnull align 8 dereferenceable(32) %7, ptr noundef nonnull align 8 dereferenceable(32) %7, i32 noundef 3)
          to label %357 unwind label %354

354:                                              ; preds = %352
  %355 = landingpad { ptr, i32 }
          catch ptr null
  %356 = extractvalue { ptr, i32 } %355, 0
  call void @__clang_call_terminate(ptr %356) #21
  unreachable

357:                                              ; preds = %348, %352
  %358 = load ptr, ptr %83, align 8, !tbaa !20
  %359 = icmp eq ptr %358, null
  br i1 %359, label %535, label %360

360:                                              ; preds = %357
  %361 = invoke noundef i1 %358(ptr noundef nonnull align 8 dereferenceable(32) %6, ptr noundef nonnull align 8 dereferenceable(32) %6, i32 noundef 3)
          to label %535 unwind label %362

362:                                              ; preds = %360
  %363 = landingpad { ptr, i32 }
          catch ptr null
  %364 = extractvalue { ptr, i32 } %363, 0
  call void @__clang_call_terminate(ptr %364) #21
  unreachable

365:                                              ; preds = %103
  %366 = landingpad { ptr, i32 }
          cleanup
  %367 = load ptr, ptr %106, align 8, !tbaa !20
  %368 = icmp eq ptr %367, null
  br i1 %368, label %374, label %369

369:                                              ; preds = %365
  %370 = invoke noundef i1 %367(ptr noundef nonnull align 8 dereferenceable(32) %9, ptr noundef nonnull align 8 dereferenceable(32) %9, i32 noundef 3)
          to label %374 unwind label %371

371:                                              ; preds = %369
  %372 = landingpad { ptr, i32 }
          catch ptr null
  %373 = extractvalue { ptr, i32 } %372, 0
  call void @__clang_call_terminate(ptr %373) #21
  unreachable

374:                                              ; preds = %365, %369
  %375 = load ptr, ptr %104, align 8, !tbaa !20
  %376 = icmp eq ptr %375, null
  br i1 %376, label %535, label %377

377:                                              ; preds = %374
  %378 = invoke noundef i1 %375(ptr noundef nonnull align 8 dereferenceable(32) %8, ptr noundef nonnull align 8 dereferenceable(32) %8, i32 noundef 3)
          to label %535 unwind label %379

379:                                              ; preds = %377
  %380 = landingpad { ptr, i32 }
          catch ptr null
  %381 = extractvalue { ptr, i32 } %380, 0
  call void @__clang_call_terminate(ptr %381) #21
  unreachable

382:                                              ; preds = %124
  %383 = landingpad { ptr, i32 }
          cleanup
  %384 = load ptr, ptr %127, align 8, !tbaa !20
  %385 = icmp eq ptr %384, null
  br i1 %385, label %391, label %386

386:                                              ; preds = %382
  %387 = invoke noundef i1 %384(ptr noundef nonnull align 8 dereferenceable(32) %11, ptr noundef nonnull align 8 dereferenceable(32) %11, i32 noundef 3)
          to label %391 unwind label %388

388:                                              ; preds = %386
  %389 = landingpad { ptr, i32 }
          catch ptr null
  %390 = extractvalue { ptr, i32 } %389, 0
  call void @__clang_call_terminate(ptr %390) #21
  unreachable

391:                                              ; preds = %382, %386
  %392 = load ptr, ptr %125, align 8, !tbaa !20
  %393 = icmp eq ptr %392, null
  br i1 %393, label %535, label %394

394:                                              ; preds = %391
  %395 = invoke noundef i1 %392(ptr noundef nonnull align 8 dereferenceable(32) %10, ptr noundef nonnull align 8 dereferenceable(32) %10, i32 noundef 3)
          to label %535 unwind label %396

396:                                              ; preds = %394
  %397 = landingpad { ptr, i32 }
          catch ptr null
  %398 = extractvalue { ptr, i32 } %397, 0
  call void @__clang_call_terminate(ptr %398) #21
  unreachable

399:                                              ; preds = %145
  %400 = landingpad { ptr, i32 }
          cleanup
  %401 = load ptr, ptr %148, align 8, !tbaa !20
  %402 = icmp eq ptr %401, null
  br i1 %402, label %408, label %403

403:                                              ; preds = %399
  %404 = invoke noundef i1 %401(ptr noundef nonnull align 8 dereferenceable(32) %13, ptr noundef nonnull align 8 dereferenceable(32) %13, i32 noundef 3)
          to label %408 unwind label %405

405:                                              ; preds = %403
  %406 = landingpad { ptr, i32 }
          catch ptr null
  %407 = extractvalue { ptr, i32 } %406, 0
  call void @__clang_call_terminate(ptr %407) #21
  unreachable

408:                                              ; preds = %399, %403
  %409 = load ptr, ptr %146, align 8, !tbaa !20
  %410 = icmp eq ptr %409, null
  br i1 %410, label %535, label %411

411:                                              ; preds = %408
  %412 = invoke noundef i1 %409(ptr noundef nonnull align 8 dereferenceable(32) %12, ptr noundef nonnull align 8 dereferenceable(32) %12, i32 noundef 3)
          to label %535 unwind label %413

413:                                              ; preds = %411
  %414 = landingpad { ptr, i32 }
          catch ptr null
  %415 = extractvalue { ptr, i32 } %414, 0
  call void @__clang_call_terminate(ptr %415) #21
  unreachable

416:                                              ; preds = %166
  %417 = landingpad { ptr, i32 }
          cleanup
  %418 = load ptr, ptr %169, align 8, !tbaa !20
  %419 = icmp eq ptr %418, null
  br i1 %419, label %425, label %420

420:                                              ; preds = %416
  %421 = invoke noundef i1 %418(ptr noundef nonnull align 8 dereferenceable(32) %15, ptr noundef nonnull align 8 dereferenceable(32) %15, i32 noundef 3)
          to label %425 unwind label %422

422:                                              ; preds = %420
  %423 = landingpad { ptr, i32 }
          catch ptr null
  %424 = extractvalue { ptr, i32 } %423, 0
  call void @__clang_call_terminate(ptr %424) #21
  unreachable

425:                                              ; preds = %416, %420
  %426 = load ptr, ptr %167, align 8, !tbaa !20
  %427 = icmp eq ptr %426, null
  br i1 %427, label %535, label %428

428:                                              ; preds = %425
  %429 = invoke noundef i1 %426(ptr noundef nonnull align 8 dereferenceable(32) %14, ptr noundef nonnull align 8 dereferenceable(32) %14, i32 noundef 3)
          to label %535 unwind label %430

430:                                              ; preds = %428
  %431 = landingpad { ptr, i32 }
          catch ptr null
  %432 = extractvalue { ptr, i32 } %431, 0
  call void @__clang_call_terminate(ptr %432) #21
  unreachable

433:                                              ; preds = %187
  %434 = landingpad { ptr, i32 }
          cleanup
  %435 = load ptr, ptr %190, align 8, !tbaa !20
  %436 = icmp eq ptr %435, null
  br i1 %436, label %442, label %437

437:                                              ; preds = %433
  %438 = invoke noundef i1 %435(ptr noundef nonnull align 8 dereferenceable(32) %17, ptr noundef nonnull align 8 dereferenceable(32) %17, i32 noundef 3)
          to label %442 unwind label %439

439:                                              ; preds = %437
  %440 = landingpad { ptr, i32 }
          catch ptr null
  %441 = extractvalue { ptr, i32 } %440, 0
  call void @__clang_call_terminate(ptr %441) #21
  unreachable

442:                                              ; preds = %433, %437
  %443 = load ptr, ptr %188, align 8, !tbaa !20
  %444 = icmp eq ptr %443, null
  br i1 %444, label %535, label %445

445:                                              ; preds = %442
  %446 = invoke noundef i1 %443(ptr noundef nonnull align 8 dereferenceable(32) %16, ptr noundef nonnull align 8 dereferenceable(32) %16, i32 noundef 3)
          to label %535 unwind label %447

447:                                              ; preds = %445
  %448 = landingpad { ptr, i32 }
          catch ptr null
  %449 = extractvalue { ptr, i32 } %448, 0
  call void @__clang_call_terminate(ptr %449) #21
  unreachable

450:                                              ; preds = %208
  %451 = landingpad { ptr, i32 }
          cleanup
  %452 = load ptr, ptr %211, align 8, !tbaa !20
  %453 = icmp eq ptr %452, null
  br i1 %453, label %459, label %454

454:                                              ; preds = %450
  %455 = invoke noundef i1 %452(ptr noundef nonnull align 8 dereferenceable(32) %19, ptr noundef nonnull align 8 dereferenceable(32) %19, i32 noundef 3)
          to label %459 unwind label %456

456:                                              ; preds = %454
  %457 = landingpad { ptr, i32 }
          catch ptr null
  %458 = extractvalue { ptr, i32 } %457, 0
  call void @__clang_call_terminate(ptr %458) #21
  unreachable

459:                                              ; preds = %450, %454
  %460 = load ptr, ptr %209, align 8, !tbaa !20
  %461 = icmp eq ptr %460, null
  br i1 %461, label %535, label %462

462:                                              ; preds = %459
  %463 = invoke noundef i1 %460(ptr noundef nonnull align 8 dereferenceable(32) %18, ptr noundef nonnull align 8 dereferenceable(32) %18, i32 noundef 3)
          to label %535 unwind label %464

464:                                              ; preds = %462
  %465 = landingpad { ptr, i32 }
          catch ptr null
  %466 = extractvalue { ptr, i32 } %465, 0
  call void @__clang_call_terminate(ptr %466) #21
  unreachable

467:                                              ; preds = %229
  %468 = landingpad { ptr, i32 }
          cleanup
  %469 = load ptr, ptr %232, align 8, !tbaa !20
  %470 = icmp eq ptr %469, null
  br i1 %470, label %476, label %471

471:                                              ; preds = %467
  %472 = invoke noundef i1 %469(ptr noundef nonnull align 8 dereferenceable(32) %21, ptr noundef nonnull align 8 dereferenceable(32) %21, i32 noundef 3)
          to label %476 unwind label %473

473:                                              ; preds = %471
  %474 = landingpad { ptr, i32 }
          catch ptr null
  %475 = extractvalue { ptr, i32 } %474, 0
  call void @__clang_call_terminate(ptr %475) #21
  unreachable

476:                                              ; preds = %467, %471
  %477 = load ptr, ptr %230, align 8, !tbaa !20
  %478 = icmp eq ptr %477, null
  br i1 %478, label %535, label %479

479:                                              ; preds = %476
  %480 = invoke noundef i1 %477(ptr noundef nonnull align 8 dereferenceable(32) %20, ptr noundef nonnull align 8 dereferenceable(32) %20, i32 noundef 3)
          to label %535 unwind label %481

481:                                              ; preds = %479
  %482 = landingpad { ptr, i32 }
          catch ptr null
  %483 = extractvalue { ptr, i32 } %482, 0
  call void @__clang_call_terminate(ptr %483) #21
  unreachable

484:                                              ; preds = %250
  %485 = landingpad { ptr, i32 }
          cleanup
  %486 = load ptr, ptr %253, align 8, !tbaa !20
  %487 = icmp eq ptr %486, null
  br i1 %487, label %493, label %488

488:                                              ; preds = %484
  %489 = invoke noundef i1 %486(ptr noundef nonnull align 8 dereferenceable(32) %23, ptr noundef nonnull align 8 dereferenceable(32) %23, i32 noundef 3)
          to label %493 unwind label %490

490:                                              ; preds = %488
  %491 = landingpad { ptr, i32 }
          catch ptr null
  %492 = extractvalue { ptr, i32 } %491, 0
  call void @__clang_call_terminate(ptr %492) #21
  unreachable

493:                                              ; preds = %484, %488
  %494 = load ptr, ptr %251, align 8, !tbaa !20
  %495 = icmp eq ptr %494, null
  br i1 %495, label %535, label %496

496:                                              ; preds = %493
  %497 = invoke noundef i1 %494(ptr noundef nonnull align 8 dereferenceable(32) %22, ptr noundef nonnull align 8 dereferenceable(32) %22, i32 noundef 3)
          to label %535 unwind label %498

498:                                              ; preds = %496
  %499 = landingpad { ptr, i32 }
          catch ptr null
  %500 = extractvalue { ptr, i32 } %499, 0
  call void @__clang_call_terminate(ptr %500) #21
  unreachable

501:                                              ; preds = %271
  %502 = landingpad { ptr, i32 }
          cleanup
  %503 = load ptr, ptr %274, align 8, !tbaa !20
  %504 = icmp eq ptr %503, null
  br i1 %504, label %510, label %505

505:                                              ; preds = %501
  %506 = invoke noundef i1 %503(ptr noundef nonnull align 8 dereferenceable(32) %25, ptr noundef nonnull align 8 dereferenceable(32) %25, i32 noundef 3)
          to label %510 unwind label %507

507:                                              ; preds = %505
  %508 = landingpad { ptr, i32 }
          catch ptr null
  %509 = extractvalue { ptr, i32 } %508, 0
  call void @__clang_call_terminate(ptr %509) #21
  unreachable

510:                                              ; preds = %501, %505
  %511 = load ptr, ptr %272, align 8, !tbaa !20
  %512 = icmp eq ptr %511, null
  br i1 %512, label %535, label %513

513:                                              ; preds = %510
  %514 = invoke noundef i1 %511(ptr noundef nonnull align 8 dereferenceable(32) %24, ptr noundef nonnull align 8 dereferenceable(32) %24, i32 noundef 3)
          to label %535 unwind label %515

515:                                              ; preds = %513
  %516 = landingpad { ptr, i32 }
          catch ptr null
  %517 = extractvalue { ptr, i32 } %516, 0
  call void @__clang_call_terminate(ptr %517) #21
  unreachable

518:                                              ; preds = %292
  %519 = landingpad { ptr, i32 }
          cleanup
  %520 = load ptr, ptr %295, align 8, !tbaa !20
  %521 = icmp eq ptr %520, null
  br i1 %521, label %527, label %522

522:                                              ; preds = %518
  %523 = invoke noundef i1 %520(ptr noundef nonnull align 8 dereferenceable(32) %27, ptr noundef nonnull align 8 dereferenceable(32) %27, i32 noundef 3)
          to label %527 unwind label %524

524:                                              ; preds = %522
  %525 = landingpad { ptr, i32 }
          catch ptr null
  %526 = extractvalue { ptr, i32 } %525, 0
  call void @__clang_call_terminate(ptr %526) #21
  unreachable

527:                                              ; preds = %518, %522
  %528 = load ptr, ptr %293, align 8, !tbaa !20
  %529 = icmp eq ptr %528, null
  br i1 %529, label %535, label %530

530:                                              ; preds = %527
  %531 = invoke noundef i1 %528(ptr noundef nonnull align 8 dereferenceable(32) %26, ptr noundef nonnull align 8 dereferenceable(32) %26, i32 noundef 3)
          to label %535 unwind label %532

532:                                              ; preds = %530
  %533 = landingpad { ptr, i32 }
          catch ptr null
  %534 = extractvalue { ptr, i32 } %533, 0
  call void @__clang_call_terminate(ptr %534) #21
  unreachable

535:                                              ; preds = %530, %527, %513, %510, %496, %493, %479, %476, %462, %459, %445, %442, %428, %425, %411, %408, %394, %391, %377, %374, %360, %357, %343, %340, %326, %323
  %536 = phi { ptr, i32 } [ %315, %323 ], [ %315, %326 ], [ %332, %340 ], [ %332, %343 ], [ %349, %357 ], [ %349, %360 ], [ %366, %374 ], [ %366, %377 ], [ %383, %391 ], [ %383, %394 ], [ %400, %408 ], [ %400, %411 ], [ %417, %425 ], [ %417, %428 ], [ %434, %442 ], [ %434, %445 ], [ %451, %459 ], [ %451, %462 ], [ %468, %476 ], [ %468, %479 ], [ %485, %493 ], [ %485, %496 ], [ %502, %510 ], [ %502, %513 ], [ %519, %527 ], [ %519, %530 ]
  resume { ptr, i32 } %536
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: mustprogress norecurse uwtable
define internal fastcc void @_ZL19checkVectorFunctionIfEvSt8functionIFT_PS1_jEES4_PKc(ptr noundef nonnull %0, ptr noundef nonnull %1, ptr noundef %2) unnamed_addr #0 personality ptr @__gxx_personality_v0 {
  %4 = alloca ptr, align 8
  %5 = alloca i32, align 4
  %6 = alloca ptr, align 8
  %7 = alloca i32, align 4
  %8 = alloca ptr, align 8
  %9 = alloca i32, align 4
  %10 = alloca ptr, align 8
  %11 = alloca i32, align 4
  %12 = alloca ptr, align 8
  %13 = alloca i32, align 4
  %14 = alloca ptr, align 8
  %15 = alloca i32, align 4
  %16 = alloca ptr, align 8
  %17 = alloca i32, align 4
  %18 = alloca ptr, align 8
  %19 = alloca i32, align 4
  %20 = alloca ptr, align 8
  %21 = alloca i32, align 4
  %22 = alloca ptr, align 8
  %23 = alloca i32, align 4
  %24 = alloca %"class.std::function", align 8
  %25 = alloca %"class.std::function", align 8
  %26 = alloca %"class.std::function", align 8
  %27 = alloca %"class.std::function", align 8
  %28 = alloca %"class.std::function", align 8
  %29 = alloca %"class.std::function", align 8
  %30 = alloca %"class.std::function", align 8
  %31 = alloca %"class.std::function", align 8
  %32 = alloca %"class.std::function", align 8
  %33 = alloca %"class.std::function", align 8
  %34 = alloca %"class.std::function", align 8
  %35 = alloca %"class.std::function", align 8
  %36 = alloca %"class.std::function", align 8
  %37 = alloca %"class.std::function", align 8
  %38 = alloca %"class.std::function", align 8
  %39 = alloca %"class.std::function", align 8
  %40 = alloca %"class.std::function", align 8
  %41 = alloca %"class.std::function", align 8
  %42 = alloca %"class.std::function", align 8
  %43 = alloca %"class.std::function", align 8
  %44 = alloca %"class.std::function", align 8
  %45 = alloca %"class.std::function", align 8
  %46 = alloca %"class.std::function", align 8
  %47 = alloca %"class.std::function", align 8
  %48 = alloca %"class.std::function", align 8
  %49 = alloca %"class.std::function", align 8
  %50 = alloca %"class.std::function", align 8
  %51 = alloca %"class.std::function", align 8
  %52 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.13, i64 noundef 9)
  %53 = icmp eq ptr %2, null
  br i1 %53, label %54, label %62

54:                                               ; preds = %3
  %55 = load ptr, ptr @_ZSt4cout, align 8, !tbaa !21
  %56 = getelementptr i8, ptr %55, i64 -24
  %57 = load i64, ptr %56, align 8
  %58 = getelementptr inbounds i8, ptr @_ZSt4cout, i64 %57
  %59 = getelementptr inbounds nuw i8, ptr %58, i64 32
  %60 = load i32, ptr %59, align 8, !tbaa !23
  %61 = or i32 %60, 1
  tail call void @_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate(ptr noundef nonnull align 8 dereferenceable(264) %58, i32 noundef %61)
  br label %65

62:                                               ; preds = %3
  %63 = tail call noundef i64 @strlen(ptr noundef nonnull dereferenceable(1) %2) #20
  %64 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull %2, i64 noundef %63)
  br label %65

65:                                               ; preds = %54, %62
  %66 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.14, i64 noundef 1)
  %67 = tail call noalias noundef nonnull dereferenceable(4096) ptr @_Znam(i64 noundef 4096) #22
  %68 = tail call fp128 @llvm.log.f128(fp128 0xL0000000000000000401F000000000000), !tbaa !33
  %69 = tail call fp128 @llvm.log.f128(fp128 0xL00000000000000004000000000000000), !tbaa !33
  %70 = fdiv fp128 %68, %69
  %71 = fptoui fp128 %70 to i64
  %72 = add i64 %71, 23
  %73 = udiv i64 %72, %71
  %74 = tail call i64 @llvm.umax.i64(i64 %73, i64 1)
  %75 = load i64, ptr getelementptr inbounds nuw (i8, ptr @_ZL3rng, i64 4992), align 8, !tbaa !12
  br label %76

76:                                               ; preds = %221, %65
  %77 = phi i64 [ %75, %65 ], [ %200, %221 ]
  %78 = phi i64 [ 0, %65 ], [ %225, %221 ]
  br label %82

79:                                               ; preds = %198
  %80 = fdiv float %215, %216
  %81 = fcmp ult float %80, 1.000000e+00
  br i1 %81, label %221, label %219, !prof !34

82:                                               ; preds = %198, %76
  %83 = phi i64 [ %77, %76 ], [ %200, %198 ]
  %84 = phi i64 [ %74, %76 ], [ %217, %198 ]
  %85 = phi float [ 1.000000e+00, %76 ], [ %216, %198 ]
  %86 = phi float [ 0.000000e+00, %76 ], [ %215, %198 ]
  %87 = icmp ugt i64 %83, 623
  br i1 %87, label %88, label %198

88:                                               ; preds = %82
  %89 = load i64, ptr @_ZL3rng, align 8, !tbaa !6
  %90 = insertelement <2 x i64> poison, i64 %89, i64 1
  br label %91

91:                                               ; preds = %91, %88
  %92 = phi i64 [ 0, %88 ], [ %125, %91 ]
  %93 = phi <2 x i64> [ %90, %88 ], [ %99, %91 ]
  %94 = getelementptr inbounds nuw i64, ptr @_ZL3rng, i64 %92
  %95 = getelementptr inbounds nuw i64, ptr @_ZL3rng, i64 %92
  %96 = getelementptr inbounds nuw i8, ptr %95, i64 8
  %97 = getelementptr inbounds nuw i8, ptr %95, i64 24
  %98 = load <2 x i64>, ptr %96, align 8, !tbaa !6
  %99 = load <2 x i64>, ptr %97, align 8, !tbaa !6
  %100 = shufflevector <2 x i64> %93, <2 x i64> %98, <2 x i32> <i32 1, i32 2>
  %101 = shufflevector <2 x i64> %98, <2 x i64> %99, <2 x i32> <i32 1, i32 2>
  %102 = and <2 x i64> %100, splat (i64 -2147483648)
  %103 = and <2 x i64> %101, splat (i64 -2147483648)
  %104 = and <2 x i64> %98, splat (i64 2147483646)
  %105 = and <2 x i64> %99, splat (i64 2147483646)
  %106 = or disjoint <2 x i64> %104, %102
  %107 = or disjoint <2 x i64> %105, %103
  %108 = getelementptr inbounds nuw i8, ptr %94, i64 3176
  %109 = getelementptr inbounds nuw i8, ptr %94, i64 3192
  %110 = load <2 x i64>, ptr %108, align 8, !tbaa !6
  %111 = load <2 x i64>, ptr %109, align 8, !tbaa !6
  %112 = lshr exact <2 x i64> %106, splat (i64 1)
  %113 = lshr exact <2 x i64> %107, splat (i64 1)
  %114 = xor <2 x i64> %112, %110
  %115 = xor <2 x i64> %113, %111
  %116 = and <2 x i64> %98, splat (i64 1)
  %117 = and <2 x i64> %99, splat (i64 1)
  %118 = icmp eq <2 x i64> %116, zeroinitializer
  %119 = icmp eq <2 x i64> %117, zeroinitializer
  %120 = select <2 x i1> %118, <2 x i64> zeroinitializer, <2 x i64> splat (i64 2567483615)
  %121 = select <2 x i1> %119, <2 x i64> zeroinitializer, <2 x i64> splat (i64 2567483615)
  %122 = xor <2 x i64> %114, %120
  %123 = xor <2 x i64> %115, %121
  %124 = getelementptr inbounds nuw i8, ptr %94, i64 16
  store <2 x i64> %122, ptr %94, align 8, !tbaa !6
  store <2 x i64> %123, ptr %124, align 8, !tbaa !6
  %125 = add nuw i64 %92, 4
  %126 = icmp eq i64 %125, 224
  br i1 %126, label %127, label %91, !llvm.loop !35

127:                                              ; preds = %91
  %128 = extractelement <2 x i64> %99, i64 1
  %129 = and i64 %128, -2147483648
  %130 = load i64, ptr getelementptr inbounds nuw (i8, ptr @_ZL3rng, i64 1800), align 8, !tbaa !6
  %131 = and i64 %130, 2147483646
  %132 = or disjoint i64 %131, %129
  %133 = load i64, ptr getelementptr inbounds nuw (i8, ptr @_ZL3rng, i64 4968), align 8, !tbaa !6
  %134 = lshr exact i64 %132, 1
  %135 = xor i64 %134, %133
  %136 = and i64 %130, 1
  %137 = icmp eq i64 %136, 0
  %138 = select i1 %137, i64 0, i64 2567483615
  %139 = xor i64 %135, %138
  store i64 %139, ptr getelementptr inbounds nuw (i8, ptr @_ZL3rng, i64 1792), align 8, !tbaa !6
  %140 = and i64 %130, -2147483648
  %141 = load i64, ptr getelementptr inbounds nuw (i8, ptr @_ZL3rng, i64 1808), align 8, !tbaa !6
  %142 = and i64 %141, 2147483646
  %143 = or disjoint i64 %142, %140
  %144 = load i64, ptr getelementptr inbounds nuw (i8, ptr @_ZL3rng, i64 4976), align 8, !tbaa !6
  %145 = lshr exact i64 %143, 1
  %146 = xor i64 %145, %144
  %147 = and i64 %141, 1
  %148 = icmp eq i64 %147, 0
  %149 = select i1 %148, i64 0, i64 2567483615
  %150 = xor i64 %146, %149
  store i64 %150, ptr getelementptr inbounds nuw (i8, ptr @_ZL3rng, i64 1800), align 8, !tbaa !6
  %151 = and i64 %141, -2147483648
  %152 = load i64, ptr getelementptr inbounds nuw (i8, ptr @_ZL3rng, i64 1816), align 8, !tbaa !6
  %153 = and i64 %152, 2147483646
  %154 = or disjoint i64 %153, %151
  %155 = load i64, ptr getelementptr inbounds nuw (i8, ptr @_ZL3rng, i64 4984), align 8, !tbaa !6
  %156 = lshr exact i64 %154, 1
  %157 = xor i64 %156, %155
  %158 = and i64 %152, 1
  %159 = icmp eq i64 %158, 0
  %160 = select i1 %159, i64 0, i64 2567483615
  %161 = xor i64 %157, %160
  store i64 %161, ptr getelementptr inbounds nuw (i8, ptr @_ZL3rng, i64 1808), align 8, !tbaa !6
  %162 = load i64, ptr getelementptr inbounds nuw (i8, ptr @_ZL3rng, i64 1816), align 8, !tbaa !6
  %163 = insertelement <2 x i64> poison, i64 %162, i64 1
  br label %164

164:                                              ; preds = %164, %127
  %165 = phi i64 [ 0, %127 ], [ %183, %164 ]
  %166 = phi <2 x i64> [ %163, %127 ], [ %171, %164 ]
  %167 = getelementptr i64, ptr @_ZL3rng, i64 %165
  %168 = getelementptr i8, ptr %167, i64 1816
  %169 = getelementptr i64, ptr @_ZL3rng, i64 %165
  %170 = getelementptr i8, ptr %169, i64 1824
  %171 = load <2 x i64>, ptr %170, align 8, !tbaa !6
  %172 = shufflevector <2 x i64> %166, <2 x i64> %171, <2 x i32> <i32 1, i32 2>
  %173 = and <2 x i64> %172, splat (i64 -2147483648)
  %174 = and <2 x i64> %171, splat (i64 2147483646)
  %175 = or disjoint <2 x i64> %174, %173
  %176 = load <2 x i64>, ptr %167, align 8, !tbaa !6
  %177 = lshr exact <2 x i64> %175, splat (i64 1)
  %178 = xor <2 x i64> %177, %176
  %179 = and <2 x i64> %171, splat (i64 1)
  %180 = icmp eq <2 x i64> %179, zeroinitializer
  %181 = select <2 x i1> %180, <2 x i64> zeroinitializer, <2 x i64> splat (i64 2567483615)
  %182 = xor <2 x i64> %178, %181
  store <2 x i64> %182, ptr %168, align 8, !tbaa !6
  %183 = add nuw i64 %165, 2
  %184 = icmp eq i64 %183, 396
  br i1 %184, label %185, label %164, !llvm.loop !38

185:                                              ; preds = %164
  %186 = load i64, ptr getelementptr inbounds nuw (i8, ptr @_ZL3rng, i64 4984), align 8, !tbaa !6
  %187 = and i64 %186, -2147483648
  %188 = load i64, ptr @_ZL3rng, align 8, !tbaa !6
  %189 = and i64 %188, 2147483646
  %190 = or disjoint i64 %189, %187
  %191 = load i64, ptr getelementptr inbounds nuw (i8, ptr @_ZL3rng, i64 3168), align 8, !tbaa !6
  %192 = lshr exact i64 %190, 1
  %193 = xor i64 %192, %191
  %194 = and i64 %188, 1
  %195 = icmp eq i64 %194, 0
  %196 = select i1 %195, i64 0, i64 2567483615
  %197 = xor i64 %193, %196
  store i64 %197, ptr getelementptr inbounds nuw (i8, ptr @_ZL3rng, i64 4984), align 8, !tbaa !6
  br label %198

198:                                              ; preds = %185, %82
  %199 = phi i64 [ 0, %185 ], [ %83, %82 ]
  %200 = add nuw nsw i64 %199, 1
  store i64 %200, ptr getelementptr inbounds nuw (i8, ptr @_ZL3rng, i64 4992), align 8, !tbaa !12
  %201 = getelementptr inbounds nuw i64, ptr @_ZL3rng, i64 %199
  %202 = load i64, ptr %201, align 8, !tbaa !6
  %203 = lshr i64 %202, 11
  %204 = and i64 %203, 4294967295
  %205 = xor i64 %204, %202
  %206 = shl i64 %205, 7
  %207 = and i64 %206, 2636928640
  %208 = xor i64 %207, %205
  %209 = shl i64 %208, 15
  %210 = and i64 %209, 4022730752
  %211 = xor i64 %210, %208
  %212 = lshr i64 %211, 18
  %213 = xor i64 %212, %211
  %214 = uitofp i64 %213 to float
  %215 = tail call float @llvm.fmuladd.f32(float %214, float %85, float %86)
  %216 = fmul float %85, 0x41F0000000000000
  %217 = add i64 %84, -1
  %218 = icmp eq i64 %217, 0
  br i1 %218, label %79, label %82, !llvm.loop !39

219:                                              ; preds = %79
  %220 = tail call noundef float @nextafterf(float noundef 1.000000e+00, float noundef 0.000000e+00) #20, !tbaa !33
  br label %221

221:                                              ; preds = %219, %79
  %222 = phi float [ %220, %219 ], [ %80, %79 ]
  %223 = tail call noundef float @llvm.fmuladd.f32(float %222, float 0x47EFFFFFE0000000, float 0x3810000000000000)
  %224 = getelementptr inbounds nuw float, ptr %67, i64 %78
  store float %223, ptr %224, align 4, !tbaa !40
  %225 = add nuw nsw i64 %78, 1
  %226 = icmp eq i64 %225, 1024
  br i1 %226, label %227, label %76, !llvm.loop !42

227:                                              ; preds = %221
  %228 = getelementptr inbounds nuw i8, ptr %67, i64 4096
  invoke void @_ZSt16__introsort_loopIPflN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_(ptr noundef nonnull %67, ptr noundef nonnull %228, i64 noundef 20, i8 undef)
          to label %229 unwind label %379

229:                                              ; preds = %227
  invoke void @_ZSt22__final_insertion_sortIPfN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_(ptr noundef nonnull %67, ptr noundef nonnull %228, i8 undef)
          to label %230 unwind label %379

230:                                              ; preds = %229
  %231 = getelementptr inbounds nuw i8, ptr %24, i64 16
  %232 = getelementptr inbounds nuw i8, ptr %0, i64 16
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %24, i8 0, i64 32, i1 false)
  %233 = load ptr, ptr %232, align 8, !tbaa !20
  %234 = icmp eq ptr %233, null
  br i1 %234, label %248, label %235

235:                                              ; preds = %230
  %236 = invoke noundef i1 %233(ptr noundef nonnull align 8 dereferenceable(32) %24, ptr noundef nonnull align 8 dereferenceable(32) %0, i32 noundef 2)
          to label %237 unwind label %239

237:                                              ; preds = %235
  %238 = load <2 x ptr>, ptr %232, align 8, !tbaa !43
  store <2 x ptr> %238, ptr %231, align 8, !tbaa !43
  br label %248

239:                                              ; preds = %235
  %240 = landingpad { ptr, i32 }
          cleanup
  %241 = load ptr, ptr %231, align 8, !tbaa !20
  %242 = icmp eq ptr %241, null
  br i1 %242, label %1667, label %243

243:                                              ; preds = %239
  %244 = invoke noundef i1 %241(ptr noundef nonnull align 8 dereferenceable(32) %24, ptr noundef nonnull align 8 dereferenceable(32) %24, i32 noundef 3)
          to label %1667 unwind label %245

245:                                              ; preds = %243
  %246 = landingpad { ptr, i32 }
          catch ptr null
  %247 = extractvalue { ptr, i32 } %246, 0
  call void @__clang_call_terminate(ptr %247) #21
  unreachable

248:                                              ; preds = %237, %230
  %249 = getelementptr inbounds nuw i8, ptr %25, i64 16
  %250 = getelementptr inbounds nuw i8, ptr %1, i64 16
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %25, i8 0, i64 32, i1 false)
  %251 = load ptr, ptr %250, align 8, !tbaa !20
  %252 = icmp eq ptr %251, null
  br i1 %252, label %266, label %253

253:                                              ; preds = %248
  %254 = invoke noundef i1 %251(ptr noundef nonnull align 8 dereferenceable(32) %25, ptr noundef nonnull align 8 dereferenceable(32) %1, i32 noundef 2)
          to label %255 unwind label %257

255:                                              ; preds = %253
  %256 = load <2 x ptr>, ptr %250, align 8, !tbaa !43
  store <2 x ptr> %256, ptr %249, align 8, !tbaa !43
  br label %266

257:                                              ; preds = %253
  %258 = landingpad { ptr, i32 }
          cleanup
  %259 = load ptr, ptr %249, align 8, !tbaa !20
  %260 = icmp eq ptr %259, null
  br i1 %260, label %390, label %261

261:                                              ; preds = %257
  %262 = invoke noundef i1 %259(ptr noundef nonnull align 8 dereferenceable(32) %25, ptr noundef nonnull align 8 dereferenceable(32) %25, i32 noundef 3)
          to label %390 unwind label %263

263:                                              ; preds = %261
  %264 = landingpad { ptr, i32 }
          catch ptr null
  %265 = extractvalue { ptr, i32 } %264, 0
  call void @__clang_call_terminate(ptr %265) #21
  unreachable

266:                                              ; preds = %255, %248
  invoke fastcc void @_ZL5checkIfEvSt8functionIFT_PS1_jEES4_PfjPKc(ptr noundef %24, ptr noundef %25, ptr noundef %67, ptr noundef nonnull @.str.15)
          to label %267 unwind label %381

267:                                              ; preds = %266
  %268 = load ptr, ptr %249, align 8, !tbaa !20
  %269 = icmp eq ptr %268, null
  br i1 %269, label %275, label %270

270:                                              ; preds = %267
  %271 = invoke noundef i1 %268(ptr noundef nonnull align 8 dereferenceable(32) %25, ptr noundef nonnull align 8 dereferenceable(32) %25, i32 noundef 3)
          to label %275 unwind label %272

272:                                              ; preds = %270
  %273 = landingpad { ptr, i32 }
          catch ptr null
  %274 = extractvalue { ptr, i32 } %273, 0
  call void @__clang_call_terminate(ptr %274) #21
  unreachable

275:                                              ; preds = %267, %270
  %276 = load ptr, ptr %231, align 8, !tbaa !20
  %277 = icmp eq ptr %276, null
  br i1 %277, label %283, label %278

278:                                              ; preds = %275
  %279 = invoke noundef i1 %276(ptr noundef nonnull align 8 dereferenceable(32) %24, ptr noundef nonnull align 8 dereferenceable(32) %24, i32 noundef 3)
          to label %283 unwind label %280

280:                                              ; preds = %278
  %281 = landingpad { ptr, i32 }
          catch ptr null
  %282 = extractvalue { ptr, i32 } %281, 0
  call void @__clang_call_terminate(ptr %282) #21
  unreachable

283:                                              ; preds = %275, %278
  %284 = getelementptr inbounds nuw i8, ptr %67, i64 4092
  br label %285

285:                                              ; preds = %285, %283
  %286 = phi i64 [ 0, %283 ], [ %302, %285 ]
  %287 = mul i64 %286, -4
  %288 = getelementptr i8, ptr %284, i64 %287
  %289 = shl i64 %286, 2
  %290 = getelementptr i8, ptr %67, i64 %289
  %291 = getelementptr i8, ptr %290, i64 16
  %292 = load <4 x float>, ptr %290, align 4, !tbaa !40
  %293 = load <4 x float>, ptr %291, align 4, !tbaa !40
  %294 = getelementptr i8, ptr %288, i64 -12
  %295 = getelementptr i8, ptr %288, i64 -28
  %296 = load <4 x float>, ptr %294, align 4, !tbaa !40
  %297 = shufflevector <4 x float> %296, <4 x float> poison, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  %298 = load <4 x float>, ptr %295, align 4, !tbaa !40
  %299 = shufflevector <4 x float> %298, <4 x float> poison, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  store <4 x float> %297, ptr %290, align 4, !tbaa !40
  store <4 x float> %299, ptr %291, align 4, !tbaa !40
  %300 = shufflevector <4 x float> %292, <4 x float> poison, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  store <4 x float> %300, ptr %294, align 4, !tbaa !40
  %301 = shufflevector <4 x float> %293, <4 x float> poison, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  store <4 x float> %301, ptr %295, align 4, !tbaa !40
  %302 = add nuw i64 %286, 8
  %303 = icmp eq i64 %302, 512
  br i1 %303, label %304, label %285, !llvm.loop !44

304:                                              ; preds = %285
  %305 = getelementptr inbounds nuw i8, ptr %26, i64 16
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %26, i8 0, i64 32, i1 false)
  %306 = load ptr, ptr %232, align 8, !tbaa !20
  %307 = icmp eq ptr %306, null
  br i1 %307, label %321, label %308

308:                                              ; preds = %304
  %309 = invoke noundef i1 %306(ptr noundef nonnull align 8 dereferenceable(32) %26, ptr noundef nonnull align 8 dereferenceable(32) %0, i32 noundef 2)
          to label %310 unwind label %312

310:                                              ; preds = %308
  %311 = load <2 x ptr>, ptr %232, align 8, !tbaa !43
  store <2 x ptr> %311, ptr %305, align 8, !tbaa !43
  br label %321

312:                                              ; preds = %308
  %313 = landingpad { ptr, i32 }
          cleanup
  %314 = load ptr, ptr %305, align 8, !tbaa !20
  %315 = icmp eq ptr %314, null
  br i1 %315, label %1667, label %316

316:                                              ; preds = %312
  %317 = invoke noundef i1 %314(ptr noundef nonnull align 8 dereferenceable(32) %26, ptr noundef nonnull align 8 dereferenceable(32) %26, i32 noundef 3)
          to label %1667 unwind label %318

318:                                              ; preds = %316
  %319 = landingpad { ptr, i32 }
          catch ptr null
  %320 = extractvalue { ptr, i32 } %319, 0
  call void @__clang_call_terminate(ptr %320) #21
  unreachable

321:                                              ; preds = %310, %304
  %322 = getelementptr inbounds nuw i8, ptr %27, i64 16
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %27, i8 0, i64 32, i1 false)
  %323 = load ptr, ptr %250, align 8, !tbaa !20
  %324 = icmp eq ptr %323, null
  br i1 %324, label %338, label %325

325:                                              ; preds = %321
  %326 = invoke noundef i1 %323(ptr noundef nonnull align 8 dereferenceable(32) %27, ptr noundef nonnull align 8 dereferenceable(32) %1, i32 noundef 2)
          to label %327 unwind label %329

327:                                              ; preds = %325
  %328 = load <2 x ptr>, ptr %250, align 8, !tbaa !43
  store <2 x ptr> %328, ptr %322, align 8, !tbaa !43
  br label %338

329:                                              ; preds = %325
  %330 = landingpad { ptr, i32 }
          cleanup
  %331 = load ptr, ptr %322, align 8, !tbaa !20
  %332 = icmp eq ptr %331, null
  br i1 %332, label %408, label %333

333:                                              ; preds = %329
  %334 = invoke noundef i1 %331(ptr noundef nonnull align 8 dereferenceable(32) %27, ptr noundef nonnull align 8 dereferenceable(32) %27, i32 noundef 3)
          to label %408 unwind label %335

335:                                              ; preds = %333
  %336 = landingpad { ptr, i32 }
          catch ptr null
  %337 = extractvalue { ptr, i32 } %336, 0
  call void @__clang_call_terminate(ptr %337) #21
  unreachable

338:                                              ; preds = %327, %321
  invoke fastcc void @_ZL5checkIfEvSt8functionIFT_PS1_jEES4_PfjPKc(ptr noundef %26, ptr noundef %27, ptr noundef %67, ptr noundef nonnull @.str.16)
          to label %339 unwind label %399

339:                                              ; preds = %338
  %340 = load ptr, ptr %322, align 8, !tbaa !20
  %341 = icmp eq ptr %340, null
  br i1 %341, label %347, label %342

342:                                              ; preds = %339
  %343 = invoke noundef i1 %340(ptr noundef nonnull align 8 dereferenceable(32) %27, ptr noundef nonnull align 8 dereferenceable(32) %27, i32 noundef 3)
          to label %347 unwind label %344

344:                                              ; preds = %342
  %345 = landingpad { ptr, i32 }
          catch ptr null
  %346 = extractvalue { ptr, i32 } %345, 0
  call void @__clang_call_terminate(ptr %346) #21
  unreachable

347:                                              ; preds = %339, %342
  %348 = load ptr, ptr %305, align 8, !tbaa !20
  %349 = icmp eq ptr %348, null
  br i1 %349, label %350, label %351

350:                                              ; preds = %351, %347
  br label %356

351:                                              ; preds = %347
  %352 = invoke noundef i1 %348(ptr noundef nonnull align 8 dereferenceable(32) %26, ptr noundef nonnull align 8 dereferenceable(32) %26, i32 noundef 3)
          to label %350 unwind label %353

353:                                              ; preds = %351
  %354 = landingpad { ptr, i32 }
          catch ptr null
  %355 = extractvalue { ptr, i32 } %354, 0
  call void @__clang_call_terminate(ptr %355) #21
  unreachable

356:                                              ; preds = %350, %356
  %357 = phi i64 [ %360, %356 ], [ 0, %350 ]
  %358 = getelementptr inbounds nuw float, ptr %67, i64 %357
  %359 = getelementptr inbounds nuw i8, ptr %358, i64 16
  store <4 x float> splat (float 0x47EFFFFFE0000000), ptr %358, align 4, !tbaa !40
  store <4 x float> splat (float 0x47EFFFFFE0000000), ptr %359, align 4, !tbaa !40
  %360 = add nuw i64 %357, 8
  %361 = icmp eq i64 %360, 1024
  br i1 %361, label %362, label %356, !llvm.loop !45

362:                                              ; preds = %356
  %363 = getelementptr inbounds nuw i8, ptr %28, i64 16
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %28, i8 0, i64 32, i1 false)
  %364 = load ptr, ptr %232, align 8, !tbaa !20
  %365 = icmp eq ptr %364, null
  br i1 %365, label %417, label %366

366:                                              ; preds = %362
  %367 = invoke noundef i1 %364(ptr noundef nonnull align 8 dereferenceable(32) %28, ptr noundef nonnull align 8 dereferenceable(32) %0, i32 noundef 2)
          to label %368 unwind label %370

368:                                              ; preds = %366
  %369 = load <2 x ptr>, ptr %232, align 8, !tbaa !43
  store <2 x ptr> %369, ptr %363, align 8, !tbaa !43
  br label %417

370:                                              ; preds = %366
  %371 = landingpad { ptr, i32 }
          cleanup
  %372 = load ptr, ptr %363, align 8, !tbaa !20
  %373 = icmp eq ptr %372, null
  br i1 %373, label %1667, label %374

374:                                              ; preds = %370
  %375 = invoke noundef i1 %372(ptr noundef nonnull align 8 dereferenceable(32) %28, ptr noundef nonnull align 8 dereferenceable(32) %28, i32 noundef 3)
          to label %1667 unwind label %376

376:                                              ; preds = %374
  %377 = landingpad { ptr, i32 }
          catch ptr null
  %378 = extractvalue { ptr, i32 } %377, 0
  call void @__clang_call_terminate(ptr %378) #21
  unreachable

379:                                              ; preds = %229, %227
  %380 = landingpad { ptr, i32 }
          cleanup
  br label %1667

381:                                              ; preds = %266
  %382 = landingpad { ptr, i32 }
          cleanup
  %383 = load ptr, ptr %249, align 8, !tbaa !20
  %384 = icmp eq ptr %383, null
  br i1 %384, label %390, label %385

385:                                              ; preds = %381
  %386 = invoke noundef i1 %383(ptr noundef nonnull align 8 dereferenceable(32) %25, ptr noundef nonnull align 8 dereferenceable(32) %25, i32 noundef 3)
          to label %390 unwind label %387

387:                                              ; preds = %385
  %388 = landingpad { ptr, i32 }
          catch ptr null
  %389 = extractvalue { ptr, i32 } %388, 0
  call void @__clang_call_terminate(ptr %389) #21
  unreachable

390:                                              ; preds = %385, %381, %261, %257
  %391 = phi { ptr, i32 } [ %258, %261 ], [ %258, %257 ], [ %382, %381 ], [ %382, %385 ]
  %392 = load ptr, ptr %231, align 8, !tbaa !20
  %393 = icmp eq ptr %392, null
  br i1 %393, label %1667, label %394

394:                                              ; preds = %390
  %395 = invoke noundef i1 %392(ptr noundef nonnull align 8 dereferenceable(32) %24, ptr noundef nonnull align 8 dereferenceable(32) %24, i32 noundef 3)
          to label %1667 unwind label %396

396:                                              ; preds = %394
  %397 = landingpad { ptr, i32 }
          catch ptr null
  %398 = extractvalue { ptr, i32 } %397, 0
  call void @__clang_call_terminate(ptr %398) #21
  unreachable

399:                                              ; preds = %338
  %400 = landingpad { ptr, i32 }
          cleanup
  %401 = load ptr, ptr %322, align 8, !tbaa !20
  %402 = icmp eq ptr %401, null
  br i1 %402, label %408, label %403

403:                                              ; preds = %399
  %404 = invoke noundef i1 %401(ptr noundef nonnull align 8 dereferenceable(32) %27, ptr noundef nonnull align 8 dereferenceable(32) %27, i32 noundef 3)
          to label %408 unwind label %405

405:                                              ; preds = %403
  %406 = landingpad { ptr, i32 }
          catch ptr null
  %407 = extractvalue { ptr, i32 } %406, 0
  call void @__clang_call_terminate(ptr %407) #21
  unreachable

408:                                              ; preds = %403, %399, %333, %329
  %409 = phi { ptr, i32 } [ %330, %333 ], [ %330, %329 ], [ %400, %399 ], [ %400, %403 ]
  %410 = load ptr, ptr %305, align 8, !tbaa !20
  %411 = icmp eq ptr %410, null
  br i1 %411, label %1667, label %412

412:                                              ; preds = %408
  %413 = invoke noundef i1 %410(ptr noundef nonnull align 8 dereferenceable(32) %26, ptr noundef nonnull align 8 dereferenceable(32) %26, i32 noundef 3)
          to label %1667 unwind label %414

414:                                              ; preds = %412
  %415 = landingpad { ptr, i32 }
          catch ptr null
  %416 = extractvalue { ptr, i32 } %415, 0
  call void @__clang_call_terminate(ptr %416) #21
  unreachable

417:                                              ; preds = %368, %362
  %418 = getelementptr inbounds nuw i8, ptr %29, i64 16
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %29, i8 0, i64 32, i1 false)
  %419 = load ptr, ptr %250, align 8, !tbaa !20
  %420 = icmp eq ptr %419, null
  br i1 %420, label %434, label %421

421:                                              ; preds = %417
  %422 = invoke noundef i1 %419(ptr noundef nonnull align 8 dereferenceable(32) %29, ptr noundef nonnull align 8 dereferenceable(32) %1, i32 noundef 2)
          to label %423 unwind label %425

423:                                              ; preds = %421
  %424 = load <2 x ptr>, ptr %250, align 8, !tbaa !43
  store <2 x ptr> %424, ptr %418, align 8, !tbaa !43
  br label %434

425:                                              ; preds = %421
  %426 = landingpad { ptr, i32 }
          cleanup
  %427 = load ptr, ptr %418, align 8, !tbaa !20
  %428 = icmp eq ptr %427, null
  br i1 %428, label %484, label %429

429:                                              ; preds = %425
  %430 = invoke noundef i1 %427(ptr noundef nonnull align 8 dereferenceable(32) %29, ptr noundef nonnull align 8 dereferenceable(32) %29, i32 noundef 3)
          to label %484 unwind label %431

431:                                              ; preds = %429
  %432 = landingpad { ptr, i32 }
          catch ptr null
  %433 = extractvalue { ptr, i32 } %432, 0
  call void @__clang_call_terminate(ptr %433) #21
  unreachable

434:                                              ; preds = %423, %417
  invoke fastcc void @_ZL5checkIfEvSt8functionIFT_PS1_jEES4_PfjPKc(ptr noundef %28, ptr noundef %29, ptr noundef %67, ptr noundef nonnull @.str.17)
          to label %435 unwind label %475

435:                                              ; preds = %434
  %436 = load ptr, ptr %418, align 8, !tbaa !20
  %437 = icmp eq ptr %436, null
  br i1 %437, label %443, label %438

438:                                              ; preds = %435
  %439 = invoke noundef i1 %436(ptr noundef nonnull align 8 dereferenceable(32) %29, ptr noundef nonnull align 8 dereferenceable(32) %29, i32 noundef 3)
          to label %443 unwind label %440

440:                                              ; preds = %438
  %441 = landingpad { ptr, i32 }
          catch ptr null
  %442 = extractvalue { ptr, i32 } %441, 0
  call void @__clang_call_terminate(ptr %442) #21
  unreachable

443:                                              ; preds = %435, %438
  %444 = load ptr, ptr %363, align 8, !tbaa !20
  %445 = icmp eq ptr %444, null
  br i1 %445, label %446, label %447

446:                                              ; preds = %447, %443
  br label %452

447:                                              ; preds = %443
  %448 = invoke noundef i1 %444(ptr noundef nonnull align 8 dereferenceable(32) %28, ptr noundef nonnull align 8 dereferenceable(32) %28, i32 noundef 3)
          to label %446 unwind label %449

449:                                              ; preds = %447
  %450 = landingpad { ptr, i32 }
          catch ptr null
  %451 = extractvalue { ptr, i32 } %450, 0
  call void @__clang_call_terminate(ptr %451) #21
  unreachable

452:                                              ; preds = %446, %452
  %453 = phi i64 [ %456, %452 ], [ 0, %446 ]
  %454 = getelementptr inbounds nuw float, ptr %67, i64 %453
  %455 = getelementptr inbounds nuw i8, ptr %454, i64 16
  store <4 x float> splat (float 0x3810000000000000), ptr %454, align 4, !tbaa !40
  store <4 x float> splat (float 0x3810000000000000), ptr %455, align 4, !tbaa !40
  %456 = add nuw i64 %453, 8
  %457 = icmp eq i64 %456, 1024
  br i1 %457, label %458, label %452, !llvm.loop !46

458:                                              ; preds = %452
  %459 = getelementptr inbounds nuw i8, ptr %30, i64 16
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %30, i8 0, i64 32, i1 false)
  %460 = load ptr, ptr %232, align 8, !tbaa !20
  %461 = icmp eq ptr %460, null
  br i1 %461, label %493, label %462

462:                                              ; preds = %458
  %463 = invoke noundef i1 %460(ptr noundef nonnull align 8 dereferenceable(32) %30, ptr noundef nonnull align 8 dereferenceable(32) %0, i32 noundef 2)
          to label %464 unwind label %466

464:                                              ; preds = %462
  %465 = load <2 x ptr>, ptr %232, align 8, !tbaa !43
  store <2 x ptr> %465, ptr %459, align 8, !tbaa !43
  br label %493

466:                                              ; preds = %462
  %467 = landingpad { ptr, i32 }
          cleanup
  %468 = load ptr, ptr %459, align 8, !tbaa !20
  %469 = icmp eq ptr %468, null
  br i1 %469, label %1667, label %470

470:                                              ; preds = %466
  %471 = invoke noundef i1 %468(ptr noundef nonnull align 8 dereferenceable(32) %30, ptr noundef nonnull align 8 dereferenceable(32) %30, i32 noundef 3)
          to label %1667 unwind label %472

472:                                              ; preds = %470
  %473 = landingpad { ptr, i32 }
          catch ptr null
  %474 = extractvalue { ptr, i32 } %473, 0
  call void @__clang_call_terminate(ptr %474) #21
  unreachable

475:                                              ; preds = %434
  %476 = landingpad { ptr, i32 }
          cleanup
  %477 = load ptr, ptr %418, align 8, !tbaa !20
  %478 = icmp eq ptr %477, null
  br i1 %478, label %484, label %479

479:                                              ; preds = %475
  %480 = invoke noundef i1 %477(ptr noundef nonnull align 8 dereferenceable(32) %29, ptr noundef nonnull align 8 dereferenceable(32) %29, i32 noundef 3)
          to label %484 unwind label %481

481:                                              ; preds = %479
  %482 = landingpad { ptr, i32 }
          catch ptr null
  %483 = extractvalue { ptr, i32 } %482, 0
  call void @__clang_call_terminate(ptr %483) #21
  unreachable

484:                                              ; preds = %479, %475, %429, %425
  %485 = phi { ptr, i32 } [ %426, %429 ], [ %426, %425 ], [ %476, %475 ], [ %476, %479 ]
  %486 = load ptr, ptr %363, align 8, !tbaa !20
  %487 = icmp eq ptr %486, null
  br i1 %487, label %1667, label %488

488:                                              ; preds = %484
  %489 = invoke noundef i1 %486(ptr noundef nonnull align 8 dereferenceable(32) %28, ptr noundef nonnull align 8 dereferenceable(32) %28, i32 noundef 3)
          to label %1667 unwind label %490

490:                                              ; preds = %488
  %491 = landingpad { ptr, i32 }
          catch ptr null
  %492 = extractvalue { ptr, i32 } %491, 0
  call void @__clang_call_terminate(ptr %492) #21
  unreachable

493:                                              ; preds = %464, %458
  %494 = getelementptr inbounds nuw i8, ptr %31, i64 16
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %31, i8 0, i64 32, i1 false)
  %495 = load ptr, ptr %250, align 8, !tbaa !20
  %496 = icmp eq ptr %495, null
  br i1 %496, label %510, label %497

497:                                              ; preds = %493
  %498 = invoke noundef i1 %495(ptr noundef nonnull align 8 dereferenceable(32) %31, ptr noundef nonnull align 8 dereferenceable(32) %1, i32 noundef 2)
          to label %499 unwind label %501

499:                                              ; preds = %497
  %500 = load <2 x ptr>, ptr %250, align 8, !tbaa !43
  store <2 x ptr> %500, ptr %494, align 8, !tbaa !43
  br label %510

501:                                              ; preds = %497
  %502 = landingpad { ptr, i32 }
          cleanup
  %503 = load ptr, ptr %494, align 8, !tbaa !20
  %504 = icmp eq ptr %503, null
  br i1 %504, label %564, label %505

505:                                              ; preds = %501
  %506 = invoke noundef i1 %503(ptr noundef nonnull align 8 dereferenceable(32) %31, ptr noundef nonnull align 8 dereferenceable(32) %31, i32 noundef 3)
          to label %564 unwind label %507

507:                                              ; preds = %505
  %508 = landingpad { ptr, i32 }
          catch ptr null
  %509 = extractvalue { ptr, i32 } %508, 0
  call void @__clang_call_terminate(ptr %509) #21
  unreachable

510:                                              ; preds = %499, %493
  invoke fastcc void @_ZL5checkIfEvSt8functionIFT_PS1_jEES4_PfjPKc(ptr noundef %30, ptr noundef %31, ptr noundef %67, ptr noundef nonnull @.str.18)
          to label %511 unwind label %555

511:                                              ; preds = %510
  %512 = load ptr, ptr %494, align 8, !tbaa !20
  %513 = icmp eq ptr %512, null
  br i1 %513, label %519, label %514

514:                                              ; preds = %511
  %515 = invoke noundef i1 %512(ptr noundef nonnull align 8 dereferenceable(32) %31, ptr noundef nonnull align 8 dereferenceable(32) %31, i32 noundef 3)
          to label %519 unwind label %516

516:                                              ; preds = %514
  %517 = landingpad { ptr, i32 }
          catch ptr null
  %518 = extractvalue { ptr, i32 } %517, 0
  call void @__clang_call_terminate(ptr %518) #21
  unreachable

519:                                              ; preds = %511, %514
  %520 = load ptr, ptr %459, align 8, !tbaa !20
  %521 = icmp eq ptr %520, null
  br i1 %521, label %527, label %522

522:                                              ; preds = %519
  %523 = invoke noundef i1 %520(ptr noundef nonnull align 8 dereferenceable(32) %30, ptr noundef nonnull align 8 dereferenceable(32) %30, i32 noundef 3)
          to label %527 unwind label %524

524:                                              ; preds = %522
  %525 = landingpad { ptr, i32 }
          catch ptr null
  %526 = extractvalue { ptr, i32 } %525, 0
  call void @__clang_call_terminate(ptr %526) #21
  unreachable

527:                                              ; preds = %519, %522
  store float 0x36A0000000000000, ptr %67, align 4, !tbaa !40
  br label %528

528:                                              ; preds = %528, %527
  %529 = phi i64 [ 0, %527 ], [ %539, %528 ]
  %530 = phi <4 x i32> [ <i32 1, i32 2, i32 3, i32 4>, %527 ], [ %540, %528 ]
  %531 = add <4 x i32> %530, splat (i32 4)
  %532 = uitofp <4 x i32> %530 to <4 x float>
  %533 = uitofp <4 x i32> %531 to <4 x float>
  %534 = fdiv <4 x float> splat (float 0x36A0000000000000), %532
  %535 = fdiv <4 x float> splat (float 0x36A0000000000000), %533
  %536 = getelementptr inbounds nuw float, ptr %67, i64 %529
  %537 = getelementptr inbounds nuw i8, ptr %536, i64 4
  %538 = getelementptr inbounds nuw i8, ptr %536, i64 20
  store <4 x float> %534, ptr %537, align 4, !tbaa !40
  store <4 x float> %535, ptr %538, align 4, !tbaa !40
  %539 = add nuw i64 %529, 8
  %540 = add <4 x i32> %530, splat (i32 8)
  %541 = icmp eq i64 %539, 1016
  br i1 %541, label %573, label %528, !llvm.loop !47

542:                                              ; preds = %573
  %543 = invoke noundef i1 %582(ptr noundef nonnull align 8 dereferenceable(32) %32, ptr noundef nonnull align 8 dereferenceable(32) %0, i32 noundef 2)
          to label %544 unwind label %546

544:                                              ; preds = %542
  %545 = load <2 x ptr>, ptr %232, align 8, !tbaa !43
  store <2 x ptr> %545, ptr %581, align 8, !tbaa !43
  br label %584

546:                                              ; preds = %542
  %547 = landingpad { ptr, i32 }
          cleanup
  %548 = load ptr, ptr %581, align 8, !tbaa !20
  %549 = icmp eq ptr %548, null
  br i1 %549, label %1667, label %550

550:                                              ; preds = %546
  %551 = invoke noundef i1 %548(ptr noundef nonnull align 8 dereferenceable(32) %32, ptr noundef nonnull align 8 dereferenceable(32) %32, i32 noundef 3)
          to label %1667 unwind label %552

552:                                              ; preds = %550
  %553 = landingpad { ptr, i32 }
          catch ptr null
  %554 = extractvalue { ptr, i32 } %553, 0
  call void @__clang_call_terminate(ptr %554) #21
  unreachable

555:                                              ; preds = %510
  %556 = landingpad { ptr, i32 }
          cleanup
  %557 = load ptr, ptr %494, align 8, !tbaa !20
  %558 = icmp eq ptr %557, null
  br i1 %558, label %564, label %559

559:                                              ; preds = %555
  %560 = invoke noundef i1 %557(ptr noundef nonnull align 8 dereferenceable(32) %31, ptr noundef nonnull align 8 dereferenceable(32) %31, i32 noundef 3)
          to label %564 unwind label %561

561:                                              ; preds = %559
  %562 = landingpad { ptr, i32 }
          catch ptr null
  %563 = extractvalue { ptr, i32 } %562, 0
  call void @__clang_call_terminate(ptr %563) #21
  unreachable

564:                                              ; preds = %559, %555, %505, %501
  %565 = phi { ptr, i32 } [ %502, %505 ], [ %502, %501 ], [ %556, %555 ], [ %556, %559 ]
  %566 = load ptr, ptr %459, align 8, !tbaa !20
  %567 = icmp eq ptr %566, null
  br i1 %567, label %1667, label %568

568:                                              ; preds = %564
  %569 = invoke noundef i1 %566(ptr noundef nonnull align 8 dereferenceable(32) %30, ptr noundef nonnull align 8 dereferenceable(32) %30, i32 noundef 3)
          to label %1667 unwind label %570

570:                                              ; preds = %568
  %571 = landingpad { ptr, i32 }
          catch ptr null
  %572 = extractvalue { ptr, i32 } %571, 0
  call void @__clang_call_terminate(ptr %572) #21
  unreachable

573:                                              ; preds = %528
  %574 = getelementptr inbounds nuw i8, ptr %67, i64 4068
  store float 0.000000e+00, ptr %574, align 4, !tbaa !40
  %575 = getelementptr inbounds nuw i8, ptr %67, i64 4072
  store float 0.000000e+00, ptr %575, align 4, !tbaa !40
  %576 = getelementptr inbounds nuw i8, ptr %67, i64 4076
  store float 0.000000e+00, ptr %576, align 4, !tbaa !40
  %577 = getelementptr inbounds nuw i8, ptr %67, i64 4080
  store float 0.000000e+00, ptr %577, align 4, !tbaa !40
  %578 = getelementptr inbounds nuw i8, ptr %67, i64 4084
  store float 0.000000e+00, ptr %578, align 4, !tbaa !40
  %579 = getelementptr inbounds nuw i8, ptr %67, i64 4088
  store float 0.000000e+00, ptr %579, align 4, !tbaa !40
  %580 = getelementptr inbounds nuw i8, ptr %67, i64 4092
  store float 0.000000e+00, ptr %580, align 4, !tbaa !40
  %581 = getelementptr inbounds nuw i8, ptr %32, i64 16
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %32, i8 0, i64 32, i1 false)
  %582 = load ptr, ptr %232, align 8, !tbaa !20
  %583 = icmp eq ptr %582, null
  br i1 %583, label %584, label %542

584:                                              ; preds = %544, %573
  %585 = getelementptr inbounds nuw i8, ptr %33, i64 16
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %33, i8 0, i64 32, i1 false)
  %586 = load ptr, ptr %250, align 8, !tbaa !20
  %587 = icmp eq ptr %586, null
  br i1 %587, label %601, label %588

588:                                              ; preds = %584
  %589 = invoke noundef i1 %586(ptr noundef nonnull align 8 dereferenceable(32) %33, ptr noundef nonnull align 8 dereferenceable(32) %1, i32 noundef 2)
          to label %590 unwind label %592

590:                                              ; preds = %588
  %591 = load <2 x ptr>, ptr %250, align 8, !tbaa !43
  store <2 x ptr> %591, ptr %585, align 8, !tbaa !43
  br label %601

592:                                              ; preds = %588
  %593 = landingpad { ptr, i32 }
          cleanup
  %594 = load ptr, ptr %585, align 8, !tbaa !20
  %595 = icmp eq ptr %594, null
  br i1 %595, label %644, label %596

596:                                              ; preds = %592
  %597 = invoke noundef i1 %594(ptr noundef nonnull align 8 dereferenceable(32) %33, ptr noundef nonnull align 8 dereferenceable(32) %33, i32 noundef 3)
          to label %644 unwind label %598

598:                                              ; preds = %596
  %599 = landingpad { ptr, i32 }
          catch ptr null
  %600 = extractvalue { ptr, i32 } %599, 0
  call void @__clang_call_terminate(ptr %600) #21
  unreachable

601:                                              ; preds = %590, %584
  invoke fastcc void @_ZL5checkIfEvSt8functionIFT_PS1_jEES4_PfjPKc(ptr noundef %32, ptr noundef %33, ptr noundef %67, ptr noundef nonnull @.str.19)
          to label %602 unwind label %635

602:                                              ; preds = %601
  %603 = load ptr, ptr %585, align 8, !tbaa !20
  %604 = icmp eq ptr %603, null
  br i1 %604, label %610, label %605

605:                                              ; preds = %602
  %606 = invoke noundef i1 %603(ptr noundef nonnull align 8 dereferenceable(32) %33, ptr noundef nonnull align 8 dereferenceable(32) %33, i32 noundef 3)
          to label %610 unwind label %607

607:                                              ; preds = %605
  %608 = landingpad { ptr, i32 }
          catch ptr null
  %609 = extractvalue { ptr, i32 } %608, 0
  call void @__clang_call_terminate(ptr %609) #21
  unreachable

610:                                              ; preds = %602, %605
  %611 = load ptr, ptr %581, align 8, !tbaa !20
  %612 = icmp eq ptr %611, null
  br i1 %612, label %618, label %613

613:                                              ; preds = %610
  %614 = invoke noundef i1 %611(ptr noundef nonnull align 8 dereferenceable(32) %32, ptr noundef nonnull align 8 dereferenceable(32) %32, i32 noundef 3)
          to label %618 unwind label %615

615:                                              ; preds = %613
  %616 = landingpad { ptr, i32 }
          catch ptr null
  %617 = extractvalue { ptr, i32 } %616, 0
  call void @__clang_call_terminate(ptr %617) #21
  unreachable

618:                                              ; preds = %610, %613
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(4096) %67, i8 0, i64 4096, i1 false), !tbaa !40
  %619 = getelementptr inbounds nuw i8, ptr %34, i64 16
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %34, i8 0, i64 32, i1 false)
  %620 = load ptr, ptr %232, align 8, !tbaa !20
  %621 = icmp eq ptr %620, null
  br i1 %621, label %653, label %622

622:                                              ; preds = %618
  %623 = invoke noundef i1 %620(ptr noundef nonnull align 8 dereferenceable(32) %34, ptr noundef nonnull align 8 dereferenceable(32) %0, i32 noundef 2)
          to label %624 unwind label %626

624:                                              ; preds = %622
  %625 = load <2 x ptr>, ptr %232, align 8, !tbaa !43
  store <2 x ptr> %625, ptr %619, align 8, !tbaa !43
  br label %653

626:                                              ; preds = %622
  %627 = landingpad { ptr, i32 }
          cleanup
  %628 = load ptr, ptr %619, align 8, !tbaa !20
  %629 = icmp eq ptr %628, null
  br i1 %629, label %1667, label %630

630:                                              ; preds = %626
  %631 = invoke noundef i1 %628(ptr noundef nonnull align 8 dereferenceable(32) %34, ptr noundef nonnull align 8 dereferenceable(32) %34, i32 noundef 3)
          to label %1667 unwind label %632

632:                                              ; preds = %630
  %633 = landingpad { ptr, i32 }
          catch ptr null
  %634 = extractvalue { ptr, i32 } %633, 0
  call void @__clang_call_terminate(ptr %634) #21
  unreachable

635:                                              ; preds = %601
  %636 = landingpad { ptr, i32 }
          cleanup
  %637 = load ptr, ptr %585, align 8, !tbaa !20
  %638 = icmp eq ptr %637, null
  br i1 %638, label %644, label %639

639:                                              ; preds = %635
  %640 = invoke noundef i1 %637(ptr noundef nonnull align 8 dereferenceable(32) %33, ptr noundef nonnull align 8 dereferenceable(32) %33, i32 noundef 3)
          to label %644 unwind label %641

641:                                              ; preds = %639
  %642 = landingpad { ptr, i32 }
          catch ptr null
  %643 = extractvalue { ptr, i32 } %642, 0
  call void @__clang_call_terminate(ptr %643) #21
  unreachable

644:                                              ; preds = %639, %635, %596, %592
  %645 = phi { ptr, i32 } [ %593, %596 ], [ %593, %592 ], [ %636, %635 ], [ %636, %639 ]
  %646 = load ptr, ptr %581, align 8, !tbaa !20
  %647 = icmp eq ptr %646, null
  br i1 %647, label %1667, label %648

648:                                              ; preds = %644
  %649 = invoke noundef i1 %646(ptr noundef nonnull align 8 dereferenceable(32) %32, ptr noundef nonnull align 8 dereferenceable(32) %32, i32 noundef 3)
          to label %1667 unwind label %650

650:                                              ; preds = %648
  %651 = landingpad { ptr, i32 }
          catch ptr null
  %652 = extractvalue { ptr, i32 } %651, 0
  call void @__clang_call_terminate(ptr %652) #21
  unreachable

653:                                              ; preds = %624, %618
  %654 = getelementptr inbounds nuw i8, ptr %35, i64 16
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %35, i8 0, i64 32, i1 false)
  %655 = load ptr, ptr %250, align 8, !tbaa !20
  %656 = icmp eq ptr %655, null
  br i1 %656, label %670, label %657

657:                                              ; preds = %653
  %658 = invoke noundef i1 %655(ptr noundef nonnull align 8 dereferenceable(32) %35, ptr noundef nonnull align 8 dereferenceable(32) %1, i32 noundef 2)
          to label %659 unwind label %661

659:                                              ; preds = %657
  %660 = load <2 x ptr>, ptr %250, align 8, !tbaa !43
  store <2 x ptr> %660, ptr %654, align 8, !tbaa !43
  br label %670

661:                                              ; preds = %657
  %662 = landingpad { ptr, i32 }
          cleanup
  %663 = load ptr, ptr %654, align 8, !tbaa !20
  %664 = icmp eq ptr %663, null
  br i1 %664, label %712, label %665

665:                                              ; preds = %661
  %666 = invoke noundef i1 %663(ptr noundef nonnull align 8 dereferenceable(32) %35, ptr noundef nonnull align 8 dereferenceable(32) %35, i32 noundef 3)
          to label %712 unwind label %667

667:                                              ; preds = %665
  %668 = landingpad { ptr, i32 }
          catch ptr null
  %669 = extractvalue { ptr, i32 } %668, 0
  call void @__clang_call_terminate(ptr %669) #21
  unreachable

670:                                              ; preds = %659, %653
  invoke fastcc void @_ZL5checkIfEvSt8functionIFT_PS1_jEES4_PfjPKc(ptr noundef %34, ptr noundef %35, ptr noundef %67, ptr noundef nonnull @.str.20)
          to label %671 unwind label %703

671:                                              ; preds = %670
  %672 = load ptr, ptr %654, align 8, !tbaa !20
  %673 = icmp eq ptr %672, null
  br i1 %673, label %679, label %674

674:                                              ; preds = %671
  %675 = invoke noundef i1 %672(ptr noundef nonnull align 8 dereferenceable(32) %35, ptr noundef nonnull align 8 dereferenceable(32) %35, i32 noundef 3)
          to label %679 unwind label %676

676:                                              ; preds = %674
  %677 = landingpad { ptr, i32 }
          catch ptr null
  %678 = extractvalue { ptr, i32 } %677, 0
  call void @__clang_call_terminate(ptr %678) #21
  unreachable

679:                                              ; preds = %671, %674
  %680 = load ptr, ptr %619, align 8, !tbaa !20
  %681 = icmp eq ptr %680, null
  br i1 %681, label %687, label %682

682:                                              ; preds = %679
  %683 = invoke noundef i1 %680(ptr noundef nonnull align 8 dereferenceable(32) %34, ptr noundef nonnull align 8 dereferenceable(32) %34, i32 noundef 3)
          to label %687 unwind label %684

684:                                              ; preds = %682
  %685 = landingpad { ptr, i32 }
          catch ptr null
  %686 = extractvalue { ptr, i32 } %685, 0
  call void @__clang_call_terminate(ptr %686) #21
  unreachable

687:                                              ; preds = %679, %682
  %688 = getelementptr inbounds nuw i8, ptr %36, i64 16
  %689 = getelementptr inbounds nuw i8, ptr %37, i64 16
  br label %690

690:                                              ; preds = %771, %687
  %691 = phi i64 [ 3, %687 ], [ %772, %771 ]
  br label %692

692:                                              ; preds = %692, %690
  %693 = phi i64 [ 0, %690 ], [ %696, %692 ]
  %694 = getelementptr inbounds nuw float, ptr %67, i64 %693
  %695 = getelementptr inbounds nuw i8, ptr %694, i64 16
  store <4 x float> splat (float 1.000000e+02), ptr %694, align 4, !tbaa !40
  store <4 x float> splat (float 1.000000e+02), ptr %695, align 4, !tbaa !40
  %696 = add nuw i64 %693, 8
  %697 = icmp eq i64 %696, 1024
  br i1 %697, label %721, label %692, !llvm.loop !48

698:                                              ; preds = %771
  %699 = getelementptr inbounds nuw i8, ptr %38, i64 16
  %700 = getelementptr inbounds nuw i8, ptr %38, i64 24
  %701 = getelementptr inbounds nuw i8, ptr %39, i64 16
  %702 = getelementptr inbounds nuw i8, ptr %39, i64 24
  br label %792

703:                                              ; preds = %670
  %704 = landingpad { ptr, i32 }
          cleanup
  %705 = load ptr, ptr %654, align 8, !tbaa !20
  %706 = icmp eq ptr %705, null
  br i1 %706, label %712, label %707

707:                                              ; preds = %703
  %708 = invoke noundef i1 %705(ptr noundef nonnull align 8 dereferenceable(32) %35, ptr noundef nonnull align 8 dereferenceable(32) %35, i32 noundef 3)
          to label %712 unwind label %709

709:                                              ; preds = %707
  %710 = landingpad { ptr, i32 }
          catch ptr null
  %711 = extractvalue { ptr, i32 } %710, 0
  call void @__clang_call_terminate(ptr %711) #21
  unreachable

712:                                              ; preds = %707, %703, %665, %661
  %713 = phi { ptr, i32 } [ %662, %665 ], [ %662, %661 ], [ %704, %703 ], [ %704, %707 ]
  %714 = load ptr, ptr %619, align 8, !tbaa !20
  %715 = icmp eq ptr %714, null
  br i1 %715, label %1667, label %716

716:                                              ; preds = %712
  %717 = invoke noundef i1 %714(ptr noundef nonnull align 8 dereferenceable(32) %34, ptr noundef nonnull align 8 dereferenceable(32) %34, i32 noundef 3)
          to label %1667 unwind label %718

718:                                              ; preds = %716
  %719 = landingpad { ptr, i32 }
          catch ptr null
  %720 = extractvalue { ptr, i32 } %719, 0
  call void @__clang_call_terminate(ptr %720) #21
  unreachable

721:                                              ; preds = %692
  %722 = getelementptr inbounds nuw float, ptr %67, i64 %691
  store float 0x7FF8000000000000, ptr %722, align 4, !tbaa !40
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %36, i8 0, i64 32, i1 false)
  %723 = load ptr, ptr %232, align 8, !tbaa !20
  %724 = icmp eq ptr %723, null
  br i1 %724, label %738, label %725

725:                                              ; preds = %721
  %726 = invoke noundef i1 %723(ptr noundef nonnull align 8 dereferenceable(32) %36, ptr noundef nonnull align 8 dereferenceable(32) %0, i32 noundef 2)
          to label %727 unwind label %729

727:                                              ; preds = %725
  %728 = load <2 x ptr>, ptr %232, align 8, !tbaa !43
  store <2 x ptr> %728, ptr %688, align 8, !tbaa !43
  br label %738

729:                                              ; preds = %725
  %730 = landingpad { ptr, i32 }
          cleanup
  %731 = load ptr, ptr %688, align 8, !tbaa !20
  %732 = icmp eq ptr %731, null
  br i1 %732, label %1667, label %733

733:                                              ; preds = %729
  %734 = invoke noundef i1 %731(ptr noundef nonnull align 8 dereferenceable(32) %36, ptr noundef nonnull align 8 dereferenceable(32) %36, i32 noundef 3)
          to label %1667 unwind label %735

735:                                              ; preds = %733
  %736 = landingpad { ptr, i32 }
          catch ptr null
  %737 = extractvalue { ptr, i32 } %736, 0
  call void @__clang_call_terminate(ptr %737) #21
  unreachable

738:                                              ; preds = %727, %721
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %37, i8 0, i64 32, i1 false)
  %739 = load ptr, ptr %250, align 8, !tbaa !20
  %740 = icmp eq ptr %739, null
  br i1 %740, label %754, label %741

741:                                              ; preds = %738
  %742 = invoke noundef i1 %739(ptr noundef nonnull align 8 dereferenceable(32) %37, ptr noundef nonnull align 8 dereferenceable(32) %1, i32 noundef 2)
          to label %743 unwind label %745

743:                                              ; preds = %741
  %744 = load <2 x ptr>, ptr %250, align 8, !tbaa !43
  store <2 x ptr> %744, ptr %689, align 8, !tbaa !43
  br label %754

745:                                              ; preds = %741
  %746 = landingpad { ptr, i32 }
          cleanup
  %747 = load ptr, ptr %689, align 8, !tbaa !20
  %748 = icmp eq ptr %747, null
  br i1 %748, label %783, label %749

749:                                              ; preds = %745
  %750 = invoke noundef i1 %747(ptr noundef nonnull align 8 dereferenceable(32) %37, ptr noundef nonnull align 8 dereferenceable(32) %37, i32 noundef 3)
          to label %783 unwind label %751

751:                                              ; preds = %749
  %752 = landingpad { ptr, i32 }
          catch ptr null
  %753 = extractvalue { ptr, i32 } %752, 0
  call void @__clang_call_terminate(ptr %753) #21
  unreachable

754:                                              ; preds = %743, %738
  invoke fastcc void @_ZL5checkIfEvSt8functionIFT_PS1_jEES4_PfjPKc(ptr noundef %36, ptr noundef %37, ptr noundef %67, ptr noundef nonnull @.str.21)
          to label %755 unwind label %774

755:                                              ; preds = %754
  %756 = load ptr, ptr %689, align 8, !tbaa !20
  %757 = icmp eq ptr %756, null
  br i1 %757, label %763, label %758

758:                                              ; preds = %755
  %759 = invoke noundef i1 %756(ptr noundef nonnull align 8 dereferenceable(32) %37, ptr noundef nonnull align 8 dereferenceable(32) %37, i32 noundef 3)
          to label %763 unwind label %760

760:                                              ; preds = %758
  %761 = landingpad { ptr, i32 }
          catch ptr null
  %762 = extractvalue { ptr, i32 } %761, 0
  call void @__clang_call_terminate(ptr %762) #21
  unreachable

763:                                              ; preds = %755, %758
  %764 = load ptr, ptr %688, align 8, !tbaa !20
  %765 = icmp eq ptr %764, null
  br i1 %765, label %771, label %766

766:                                              ; preds = %763
  %767 = invoke noundef i1 %764(ptr noundef nonnull align 8 dereferenceable(32) %36, ptr noundef nonnull align 8 dereferenceable(32) %36, i32 noundef 3)
          to label %771 unwind label %768

768:                                              ; preds = %766
  %769 = landingpad { ptr, i32 }
          catch ptr null
  %770 = extractvalue { ptr, i32 } %769, 0
  call void @__clang_call_terminate(ptr %770) #21
  unreachable

771:                                              ; preds = %763, %766
  %772 = add nuw nsw i64 %691, 1
  %773 = icmp eq i64 %772, 32
  br i1 %773, label %698, label %690, !llvm.loop !49

774:                                              ; preds = %754
  %775 = landingpad { ptr, i32 }
          cleanup
  %776 = load ptr, ptr %689, align 8, !tbaa !20
  %777 = icmp eq ptr %776, null
  br i1 %777, label %783, label %778

778:                                              ; preds = %774
  %779 = invoke noundef i1 %776(ptr noundef nonnull align 8 dereferenceable(32) %37, ptr noundef nonnull align 8 dereferenceable(32) %37, i32 noundef 3)
          to label %783 unwind label %780

780:                                              ; preds = %778
  %781 = landingpad { ptr, i32 }
          catch ptr null
  %782 = extractvalue { ptr, i32 } %781, 0
  call void @__clang_call_terminate(ptr %782) #21
  unreachable

783:                                              ; preds = %778, %774, %749, %745
  %784 = phi { ptr, i32 } [ %746, %749 ], [ %746, %745 ], [ %775, %774 ], [ %775, %778 ]
  %785 = load ptr, ptr %688, align 8, !tbaa !20
  %786 = icmp eq ptr %785, null
  br i1 %786, label %1667, label %787

787:                                              ; preds = %783
  %788 = invoke noundef i1 %785(ptr noundef nonnull align 8 dereferenceable(32) %36, ptr noundef nonnull align 8 dereferenceable(32) %36, i32 noundef 3)
          to label %1667 unwind label %789

789:                                              ; preds = %787
  %790 = landingpad { ptr, i32 }
          catch ptr null
  %791 = extractvalue { ptr, i32 } %790, 0
  call void @__clang_call_terminate(ptr %791) #21
  unreachable

792:                                              ; preds = %808, %698
  %793 = phi i64 [ 0, %698 ], [ %809, %808 ]
  br label %794

794:                                              ; preds = %794, %792
  %795 = phi i64 [ 0, %792 ], [ %798, %794 ]
  %796 = getelementptr inbounds nuw float, ptr %67, i64 %795
  %797 = getelementptr inbounds nuw i8, ptr %796, i64 16
  store <4 x float> splat (float -1.000000e+00), ptr %796, align 4, !tbaa !40
  store <4 x float> splat (float -1.000000e+00), ptr %797, align 4, !tbaa !40
  %798 = add nuw i64 %795, 8
  %799 = icmp eq i64 %798, 1024
  br i1 %799, label %805, label %794, !llvm.loop !50

800:                                              ; preds = %808
  %801 = getelementptr inbounds nuw i8, ptr %40, i64 16
  %802 = getelementptr inbounds nuw i8, ptr %40, i64 24
  %803 = getelementptr inbounds nuw i8, ptr %41, i64 16
  %804 = getelementptr inbounds nuw i8, ptr %41, i64 24
  br label %937

805:                                              ; preds = %794
  %806 = getelementptr inbounds nuw float, ptr %67, i64 %793
  %807 = getelementptr inbounds nuw float, ptr %67, i64 %793
  br label %811

808:                                              ; preds = %912
  %809 = add nuw nsw i64 %793, 1
  %810 = icmp eq i64 %809, 64
  br i1 %810, label %800, label %792, !llvm.loop !51

811:                                              ; preds = %805, %912
  %812 = phi i64 [ 1, %805 ], [ %913, %912 ]
  store float -0.000000e+00, ptr %806, align 4, !tbaa !40
  %813 = getelementptr inbounds nuw float, ptr %807, i64 %812
  store float 0.000000e+00, ptr %813, align 4, !tbaa !40
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %38, i8 0, i64 32, i1 false)
  %814 = load ptr, ptr %232, align 8, !tbaa !20
  %815 = icmp eq ptr %814, null
  br i1 %815, label %830, label %816

816:                                              ; preds = %811
  %817 = invoke noundef i1 %814(ptr noundef nonnull align 8 dereferenceable(32) %38, ptr noundef nonnull align 8 dereferenceable(32) %0, i32 noundef 2)
          to label %818 unwind label %821

818:                                              ; preds = %816
  %819 = load <2 x ptr>, ptr %232, align 8, !tbaa !43
  store <2 x ptr> %819, ptr %699, align 8, !tbaa !43
  %820 = extractelement <2 x ptr> %819, i64 0
  br label %830

821:                                              ; preds = %816
  %822 = landingpad { ptr, i32 }
          cleanup
  %823 = load ptr, ptr %699, align 8, !tbaa !20
  %824 = icmp eq ptr %823, null
  br i1 %824, label %1667, label %825

825:                                              ; preds = %821
  %826 = invoke noundef i1 %823(ptr noundef nonnull align 8 dereferenceable(32) %38, ptr noundef nonnull align 8 dereferenceable(32) %38, i32 noundef 3)
          to label %1667 unwind label %827

827:                                              ; preds = %825
  %828 = landingpad { ptr, i32 }
          catch ptr null
  %829 = extractvalue { ptr, i32 } %828, 0
  call void @__clang_call_terminate(ptr %829) #21
  unreachable

830:                                              ; preds = %818, %811
  %831 = phi ptr [ %820, %818 ], [ null, %811 ]
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %39, i8 0, i64 32, i1 false)
  %832 = load ptr, ptr %250, align 8, !tbaa !20
  %833 = icmp eq ptr %832, null
  br i1 %833, label %848, label %834

834:                                              ; preds = %830
  %835 = invoke noundef i1 %832(ptr noundef nonnull align 8 dereferenceable(32) %39, ptr noundef nonnull align 8 dereferenceable(32) %1, i32 noundef 2)
          to label %836 unwind label %839

836:                                              ; preds = %834
  %837 = load <2 x ptr>, ptr %250, align 8, !tbaa !43
  store <2 x ptr> %837, ptr %701, align 8, !tbaa !43
  %838 = load ptr, ptr %699, align 8, !tbaa !20
  br label %848

839:                                              ; preds = %834
  %840 = landingpad { ptr, i32 }
          cleanup
  %841 = load ptr, ptr %701, align 8, !tbaa !20
  %842 = icmp eq ptr %841, null
  br i1 %842, label %928, label %843

843:                                              ; preds = %839
  %844 = invoke noundef i1 %841(ptr noundef nonnull align 8 dereferenceable(32) %39, ptr noundef nonnull align 8 dereferenceable(32) %39, i32 noundef 3)
          to label %928 unwind label %845

845:                                              ; preds = %843
  %846 = landingpad { ptr, i32 }
          catch ptr null
  %847 = extractvalue { ptr, i32 } %846, 0
  call void @__clang_call_terminate(ptr %847) #21
  unreachable

848:                                              ; preds = %836, %830
  %849 = phi ptr [ %838, %836 ], [ %831, %830 ]
  call void @llvm.lifetime.start.p0(ptr nonnull %22)
  call void @llvm.lifetime.start.p0(ptr nonnull %23)
  store ptr %67, ptr %22, align 8, !tbaa !52
  store i32 1024, ptr %23, align 4, !tbaa !33
  %850 = icmp eq ptr %849, null
  br i1 %850, label %851, label %853

851:                                              ; preds = %856, %848
  invoke void @_ZSt25__throw_bad_function_callv() #23
          to label %852 unwind label %917

852:                                              ; preds = %851
  unreachable

853:                                              ; preds = %848
  %854 = load ptr, ptr %700, align 8, !tbaa !16
  %855 = invoke noundef float %854(ptr noundef nonnull align 8 dereferenceable(32) %38, ptr noundef nonnull align 8 dereferenceable(8) %22, ptr noundef nonnull align 4 dereferenceable(4) %23)
          to label %856 unwind label %915

856:                                              ; preds = %853
  call void @llvm.lifetime.end.p0(ptr nonnull %22)
  call void @llvm.lifetime.end.p0(ptr nonnull %23)
  call void @llvm.lifetime.start.p0(ptr nonnull %20)
  call void @llvm.lifetime.start.p0(ptr nonnull %21)
  store ptr %67, ptr %20, align 8, !tbaa !52
  store i32 1024, ptr %21, align 4, !tbaa !33
  %857 = load ptr, ptr %701, align 8, !tbaa !20
  %858 = icmp eq ptr %857, null
  br i1 %858, label %851, label %859

859:                                              ; preds = %856
  %860 = load ptr, ptr %702, align 8, !tbaa !16
  %861 = invoke noundef float %860(ptr noundef nonnull align 8 dereferenceable(32) %39, ptr noundef nonnull align 8 dereferenceable(8) %20, ptr noundef nonnull align 4 dereferenceable(4) %21)
          to label %862 unwind label %915

862:                                              ; preds = %859
  call void @llvm.lifetime.end.p0(ptr nonnull %20)
  call void @llvm.lifetime.end.p0(ptr nonnull %21)
  %863 = fcmp uno float %855, %861
  br i1 %863, label %875, label %864

864:                                              ; preds = %862
  %865 = fcmp oeq float %855, 0.000000e+00
  br i1 %865, label %866, label %873

866:                                              ; preds = %864
  %867 = fcmp oeq float %861, 0.000000e+00
  br i1 %867, label %868, label %879

868:                                              ; preds = %866
  %869 = bitcast float %855 to i32
  %870 = bitcast float %861 to i32
  %871 = xor i32 %870, %869
  %872 = icmp sgt i32 %871, -1
  br i1 %872, label %896, label %879

873:                                              ; preds = %864
  %874 = fcmp oeq float %855, %861
  br i1 %874, label %896, label %879

875:                                              ; preds = %862
  %876 = fcmp uno float %855, 0.000000e+00
  %877 = fcmp uno float %861, 0.000000e+00
  %878 = and i1 %876, %877
  br i1 %878, label %896, label %879

879:                                              ; preds = %875, %873, %868, %866
  %880 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.27, i64 noundef 11)
          to label %881 unwind label %917

881:                                              ; preds = %879
  %882 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.22, i64 noundef 12)
          to label %883 unwind label %917

883:                                              ; preds = %881
  %884 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.28, i64 noundef 2)
          to label %885 unwind label %917

885:                                              ; preds = %883
  %886 = fpext float %855 to double
  %887 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo9_M_insertIdEERSoT_(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, double noundef %886)
          to label %888 unwind label %917

888:                                              ; preds = %885
  %889 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %887, ptr noundef nonnull @.str.29, i64 noundef 4)
          to label %890 unwind label %917

890:                                              ; preds = %888
  %891 = fpext float %861 to double
  %892 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo9_M_insertIdEERSoT_(ptr noundef nonnull align 8 dereferenceable(8) %887, double noundef %891)
          to label %893 unwind label %917

893:                                              ; preds = %890
  %894 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) %892, ptr noundef nonnull @.str.14)
          to label %895 unwind label %917

895:                                              ; preds = %893
  call void @exit(i32 noundef 1) #24
  unreachable

896:                                              ; preds = %875, %873, %868
  %897 = load ptr, ptr %701, align 8, !tbaa !20
  %898 = icmp eq ptr %897, null
  br i1 %898, label %904, label %899

899:                                              ; preds = %896
  %900 = invoke noundef i1 %897(ptr noundef nonnull align 8 dereferenceable(32) %39, ptr noundef nonnull align 8 dereferenceable(32) %39, i32 noundef 3)
          to label %904 unwind label %901

901:                                              ; preds = %899
  %902 = landingpad { ptr, i32 }
          catch ptr null
  %903 = extractvalue { ptr, i32 } %902, 0
  call void @__clang_call_terminate(ptr %903) #21
  unreachable

904:                                              ; preds = %896, %899
  %905 = load ptr, ptr %699, align 8, !tbaa !20
  %906 = icmp eq ptr %905, null
  br i1 %906, label %912, label %907

907:                                              ; preds = %904
  %908 = invoke noundef i1 %905(ptr noundef nonnull align 8 dereferenceable(32) %38, ptr noundef nonnull align 8 dereferenceable(32) %38, i32 noundef 3)
          to label %912 unwind label %909

909:                                              ; preds = %907
  %910 = landingpad { ptr, i32 }
          catch ptr null
  %911 = extractvalue { ptr, i32 } %910, 0
  call void @__clang_call_terminate(ptr %911) #21
  unreachable

912:                                              ; preds = %904, %907
  %913 = add nuw nsw i64 %812, 1
  %914 = icmp eq i64 %913, 32
  br i1 %914, label %808, label %811, !llvm.loop !54

915:                                              ; preds = %853, %859
  %916 = landingpad { ptr, i32 }
          cleanup
  br label %919

917:                                              ; preds = %851, %893, %890, %888, %885, %883, %881, %879
  %918 = landingpad { ptr, i32 }
          cleanup
  br label %919

919:                                              ; preds = %917, %915
  %920 = phi { ptr, i32 } [ %916, %915 ], [ %918, %917 ]
  %921 = load ptr, ptr %701, align 8, !tbaa !20
  %922 = icmp eq ptr %921, null
  br i1 %922, label %928, label %923

923:                                              ; preds = %919
  %924 = invoke noundef i1 %921(ptr noundef nonnull align 8 dereferenceable(32) %39, ptr noundef nonnull align 8 dereferenceable(32) %39, i32 noundef 3)
          to label %928 unwind label %925

925:                                              ; preds = %923
  %926 = landingpad { ptr, i32 }
          catch ptr null
  %927 = extractvalue { ptr, i32 } %926, 0
  call void @__clang_call_terminate(ptr %927) #21
  unreachable

928:                                              ; preds = %923, %919, %843, %839
  %929 = phi { ptr, i32 } [ %840, %843 ], [ %840, %839 ], [ %920, %919 ], [ %920, %923 ]
  %930 = load ptr, ptr %699, align 8, !tbaa !20
  %931 = icmp eq ptr %930, null
  br i1 %931, label %1667, label %932

932:                                              ; preds = %928
  %933 = invoke noundef i1 %930(ptr noundef nonnull align 8 dereferenceable(32) %38, ptr noundef nonnull align 8 dereferenceable(32) %38, i32 noundef 3)
          to label %1667 unwind label %934

934:                                              ; preds = %932
  %935 = landingpad { ptr, i32 }
          catch ptr null
  %936 = extractvalue { ptr, i32 } %935, 0
  call void @__clang_call_terminate(ptr %936) #21
  unreachable

937:                                              ; preds = %955, %800
  %938 = phi i64 [ 0, %800 ], [ %956, %955 ]
  br label %939

939:                                              ; preds = %939, %937
  %940 = phi i64 [ 0, %937 ], [ %943, %939 ]
  %941 = getelementptr inbounds nuw float, ptr %67, i64 %940
  %942 = getelementptr inbounds nuw i8, ptr %941, i64 16
  store <4 x float> splat (float -1.000000e+00), ptr %941, align 4, !tbaa !40
  store <4 x float> splat (float -1.000000e+00), ptr %942, align 4, !tbaa !40
  %943 = add nuw i64 %940, 8
  %944 = icmp eq i64 %943, 1024
  br i1 %944, label %952, label %939, !llvm.loop !55

945:                                              ; preds = %955
  %946 = getelementptr inbounds nuw i8, ptr %42, i64 16
  %947 = getelementptr inbounds nuw i8, ptr %43, i64 16
  %948 = getelementptr inbounds nuw i8, ptr %44, i64 16
  %949 = getelementptr inbounds nuw i8, ptr %44, i64 24
  %950 = getelementptr inbounds nuw i8, ptr %45, i64 16
  %951 = getelementptr inbounds nuw i8, ptr %45, i64 24
  br label %1084

952:                                              ; preds = %939
  %953 = getelementptr inbounds nuw float, ptr %67, i64 %938
  %954 = getelementptr inbounds nuw float, ptr %67, i64 %938
  br label %958

955:                                              ; preds = %1059
  %956 = add nuw nsw i64 %938, 1
  %957 = icmp eq i64 %956, 64
  br i1 %957, label %945, label %937, !llvm.loop !56

958:                                              ; preds = %952, %1059
  %959 = phi i64 [ 1, %952 ], [ %1060, %1059 ]
  store float 0.000000e+00, ptr %953, align 4, !tbaa !40
  %960 = getelementptr inbounds nuw float, ptr %954, i64 %959
  store float -0.000000e+00, ptr %960, align 4, !tbaa !40
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %40, i8 0, i64 32, i1 false)
  %961 = load ptr, ptr %232, align 8, !tbaa !20
  %962 = icmp eq ptr %961, null
  br i1 %962, label %977, label %963

963:                                              ; preds = %958
  %964 = invoke noundef i1 %961(ptr noundef nonnull align 8 dereferenceable(32) %40, ptr noundef nonnull align 8 dereferenceable(32) %0, i32 noundef 2)
          to label %965 unwind label %968

965:                                              ; preds = %963
  %966 = load <2 x ptr>, ptr %232, align 8, !tbaa !43
  store <2 x ptr> %966, ptr %801, align 8, !tbaa !43
  %967 = extractelement <2 x ptr> %966, i64 0
  br label %977

968:                                              ; preds = %963
  %969 = landingpad { ptr, i32 }
          cleanup
  %970 = load ptr, ptr %801, align 8, !tbaa !20
  %971 = icmp eq ptr %970, null
  br i1 %971, label %1667, label %972

972:                                              ; preds = %968
  %973 = invoke noundef i1 %970(ptr noundef nonnull align 8 dereferenceable(32) %40, ptr noundef nonnull align 8 dereferenceable(32) %40, i32 noundef 3)
          to label %1667 unwind label %974

974:                                              ; preds = %972
  %975 = landingpad { ptr, i32 }
          catch ptr null
  %976 = extractvalue { ptr, i32 } %975, 0
  call void @__clang_call_terminate(ptr %976) #21
  unreachable

977:                                              ; preds = %965, %958
  %978 = phi ptr [ %967, %965 ], [ null, %958 ]
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %41, i8 0, i64 32, i1 false)
  %979 = load ptr, ptr %250, align 8, !tbaa !20
  %980 = icmp eq ptr %979, null
  br i1 %980, label %995, label %981

981:                                              ; preds = %977
  %982 = invoke noundef i1 %979(ptr noundef nonnull align 8 dereferenceable(32) %41, ptr noundef nonnull align 8 dereferenceable(32) %1, i32 noundef 2)
          to label %983 unwind label %986

983:                                              ; preds = %981
  %984 = load <2 x ptr>, ptr %250, align 8, !tbaa !43
  store <2 x ptr> %984, ptr %803, align 8, !tbaa !43
  %985 = load ptr, ptr %801, align 8, !tbaa !20
  br label %995

986:                                              ; preds = %981
  %987 = landingpad { ptr, i32 }
          cleanup
  %988 = load ptr, ptr %803, align 8, !tbaa !20
  %989 = icmp eq ptr %988, null
  br i1 %989, label %1075, label %990

990:                                              ; preds = %986
  %991 = invoke noundef i1 %988(ptr noundef nonnull align 8 dereferenceable(32) %41, ptr noundef nonnull align 8 dereferenceable(32) %41, i32 noundef 3)
          to label %1075 unwind label %992

992:                                              ; preds = %990
  %993 = landingpad { ptr, i32 }
          catch ptr null
  %994 = extractvalue { ptr, i32 } %993, 0
  call void @__clang_call_terminate(ptr %994) #21
  unreachable

995:                                              ; preds = %983, %977
  %996 = phi ptr [ %985, %983 ], [ %978, %977 ]
  call void @llvm.lifetime.start.p0(ptr nonnull %18)
  call void @llvm.lifetime.start.p0(ptr nonnull %19)
  store ptr %67, ptr %18, align 8, !tbaa !52
  store i32 1024, ptr %19, align 4, !tbaa !33
  %997 = icmp eq ptr %996, null
  br i1 %997, label %998, label %1000

998:                                              ; preds = %1003, %995
  invoke void @_ZSt25__throw_bad_function_callv() #23
          to label %999 unwind label %1064

999:                                              ; preds = %998
  unreachable

1000:                                             ; preds = %995
  %1001 = load ptr, ptr %802, align 8, !tbaa !16
  %1002 = invoke noundef float %1001(ptr noundef nonnull align 8 dereferenceable(32) %40, ptr noundef nonnull align 8 dereferenceable(8) %18, ptr noundef nonnull align 4 dereferenceable(4) %19)
          to label %1003 unwind label %1062

1003:                                             ; preds = %1000
  call void @llvm.lifetime.end.p0(ptr nonnull %18)
  call void @llvm.lifetime.end.p0(ptr nonnull %19)
  call void @llvm.lifetime.start.p0(ptr nonnull %16)
  call void @llvm.lifetime.start.p0(ptr nonnull %17)
  store ptr %67, ptr %16, align 8, !tbaa !52
  store i32 1024, ptr %17, align 4, !tbaa !33
  %1004 = load ptr, ptr %803, align 8, !tbaa !20
  %1005 = icmp eq ptr %1004, null
  br i1 %1005, label %998, label %1006

1006:                                             ; preds = %1003
  %1007 = load ptr, ptr %804, align 8, !tbaa !16
  %1008 = invoke noundef float %1007(ptr noundef nonnull align 8 dereferenceable(32) %41, ptr noundef nonnull align 8 dereferenceable(8) %16, ptr noundef nonnull align 4 dereferenceable(4) %17)
          to label %1009 unwind label %1062

1009:                                             ; preds = %1006
  call void @llvm.lifetime.end.p0(ptr nonnull %16)
  call void @llvm.lifetime.end.p0(ptr nonnull %17)
  %1010 = fcmp uno float %1002, %1008
  br i1 %1010, label %1022, label %1011

1011:                                             ; preds = %1009
  %1012 = fcmp oeq float %1002, 0.000000e+00
  br i1 %1012, label %1013, label %1020

1013:                                             ; preds = %1011
  %1014 = fcmp oeq float %1008, 0.000000e+00
  br i1 %1014, label %1015, label %1026

1015:                                             ; preds = %1013
  %1016 = bitcast float %1002 to i32
  %1017 = bitcast float %1008 to i32
  %1018 = xor i32 %1017, %1016
  %1019 = icmp sgt i32 %1018, -1
  br i1 %1019, label %1043, label %1026

1020:                                             ; preds = %1011
  %1021 = fcmp oeq float %1002, %1008
  br i1 %1021, label %1043, label %1026

1022:                                             ; preds = %1009
  %1023 = fcmp uno float %1002, 0.000000e+00
  %1024 = fcmp uno float %1008, 0.000000e+00
  %1025 = and i1 %1023, %1024
  br i1 %1025, label %1043, label %1026

1026:                                             ; preds = %1022, %1020, %1015, %1013
  %1027 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.27, i64 noundef 11)
          to label %1028 unwind label %1064

1028:                                             ; preds = %1026
  %1029 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.22, i64 noundef 12)
          to label %1030 unwind label %1064

1030:                                             ; preds = %1028
  %1031 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.28, i64 noundef 2)
          to label %1032 unwind label %1064

1032:                                             ; preds = %1030
  %1033 = fpext float %1002 to double
  %1034 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo9_M_insertIdEERSoT_(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, double noundef %1033)
          to label %1035 unwind label %1064

1035:                                             ; preds = %1032
  %1036 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %1034, ptr noundef nonnull @.str.29, i64 noundef 4)
          to label %1037 unwind label %1064

1037:                                             ; preds = %1035
  %1038 = fpext float %1008 to double
  %1039 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo9_M_insertIdEERSoT_(ptr noundef nonnull align 8 dereferenceable(8) %1034, double noundef %1038)
          to label %1040 unwind label %1064

1040:                                             ; preds = %1037
  %1041 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) %1039, ptr noundef nonnull @.str.14)
          to label %1042 unwind label %1064

1042:                                             ; preds = %1040
  call void @exit(i32 noundef 1) #24
  unreachable

1043:                                             ; preds = %1022, %1020, %1015
  %1044 = load ptr, ptr %803, align 8, !tbaa !20
  %1045 = icmp eq ptr %1044, null
  br i1 %1045, label %1051, label %1046

1046:                                             ; preds = %1043
  %1047 = invoke noundef i1 %1044(ptr noundef nonnull align 8 dereferenceable(32) %41, ptr noundef nonnull align 8 dereferenceable(32) %41, i32 noundef 3)
          to label %1051 unwind label %1048

1048:                                             ; preds = %1046
  %1049 = landingpad { ptr, i32 }
          catch ptr null
  %1050 = extractvalue { ptr, i32 } %1049, 0
  call void @__clang_call_terminate(ptr %1050) #21
  unreachable

1051:                                             ; preds = %1043, %1046
  %1052 = load ptr, ptr %801, align 8, !tbaa !20
  %1053 = icmp eq ptr %1052, null
  br i1 %1053, label %1059, label %1054

1054:                                             ; preds = %1051
  %1055 = invoke noundef i1 %1052(ptr noundef nonnull align 8 dereferenceable(32) %40, ptr noundef nonnull align 8 dereferenceable(32) %40, i32 noundef 3)
          to label %1059 unwind label %1056

1056:                                             ; preds = %1054
  %1057 = landingpad { ptr, i32 }
          catch ptr null
  %1058 = extractvalue { ptr, i32 } %1057, 0
  call void @__clang_call_terminate(ptr %1058) #21
  unreachable

1059:                                             ; preds = %1051, %1054
  %1060 = add nuw nsw i64 %959, 1
  %1061 = icmp eq i64 %1060, 32
  br i1 %1061, label %955, label %958, !llvm.loop !57

1062:                                             ; preds = %1000, %1006
  %1063 = landingpad { ptr, i32 }
          cleanup
  br label %1066

1064:                                             ; preds = %998, %1040, %1037, %1035, %1032, %1030, %1028, %1026
  %1065 = landingpad { ptr, i32 }
          cleanup
  br label %1066

1066:                                             ; preds = %1064, %1062
  %1067 = phi { ptr, i32 } [ %1063, %1062 ], [ %1065, %1064 ]
  %1068 = load ptr, ptr %803, align 8, !tbaa !20
  %1069 = icmp eq ptr %1068, null
  br i1 %1069, label %1075, label %1070

1070:                                             ; preds = %1066
  %1071 = invoke noundef i1 %1068(ptr noundef nonnull align 8 dereferenceable(32) %41, ptr noundef nonnull align 8 dereferenceable(32) %41, i32 noundef 3)
          to label %1075 unwind label %1072

1072:                                             ; preds = %1070
  %1073 = landingpad { ptr, i32 }
          catch ptr null
  %1074 = extractvalue { ptr, i32 } %1073, 0
  call void @__clang_call_terminate(ptr %1074) #21
  unreachable

1075:                                             ; preds = %1070, %1066, %990, %986
  %1076 = phi { ptr, i32 } [ %987, %990 ], [ %987, %986 ], [ %1067, %1066 ], [ %1067, %1070 ]
  %1077 = load ptr, ptr %801, align 8, !tbaa !20
  %1078 = icmp eq ptr %1077, null
  br i1 %1078, label %1667, label %1079

1079:                                             ; preds = %1075
  %1080 = invoke noundef i1 %1077(ptr noundef nonnull align 8 dereferenceable(32) %40, ptr noundef nonnull align 8 dereferenceable(32) %40, i32 noundef 3)
          to label %1667 unwind label %1081

1081:                                             ; preds = %1079
  %1082 = landingpad { ptr, i32 }
          catch ptr null
  %1083 = extractvalue { ptr, i32 } %1082, 0
  call void @__clang_call_terminate(ptr %1083) #21
  unreachable

1084:                                             ; preds = %1155, %945
  %1085 = phi i64 [ 0, %945 ], [ %1156, %1155 ]
  br label %1086

1086:                                             ; preds = %1086, %1084
  %1087 = phi i64 [ 0, %1084 ], [ %1094, %1086 ]
  %1088 = phi <4 x i32> [ <i32 0, i32 1, i32 2, i32 3>, %1084 ], [ %1095, %1086 ]
  %1089 = add <4 x i32> %1088, splat (i32 4)
  %1090 = uitofp <4 x i32> %1088 to <4 x float>
  %1091 = uitofp <4 x i32> %1089 to <4 x float>
  %1092 = getelementptr inbounds nuw float, ptr %67, i64 %1087
  %1093 = getelementptr inbounds nuw i8, ptr %1092, i64 16
  store <4 x float> %1090, ptr %1092, align 4, !tbaa !40
  store <4 x float> %1091, ptr %1093, align 4, !tbaa !40
  %1094 = add nuw i64 %1087, 8
  %1095 = add <4 x i32> %1088, splat (i32 8)
  %1096 = icmp eq i64 %1094, 1024
  br i1 %1096, label %1104, label %1086, !llvm.loop !58

1097:                                             ; preds = %1155
  %1098 = getelementptr inbounds nuw i8, ptr %46, i64 16
  %1099 = getelementptr inbounds nuw i8, ptr %47, i64 16
  %1100 = getelementptr inbounds nuw i8, ptr %48, i64 16
  %1101 = getelementptr inbounds nuw i8, ptr %48, i64 24
  %1102 = getelementptr inbounds nuw i8, ptr %49, i64 16
  %1103 = getelementptr inbounds nuw i8, ptr %49, i64 24
  br label %1306

1104:                                             ; preds = %1086
  %1105 = getelementptr inbounds nuw float, ptr %67, i64 %1085
  store float 1.025000e+03, ptr %1105, align 4, !tbaa !40
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %42, i8 0, i64 32, i1 false)
  %1106 = load ptr, ptr %232, align 8, !tbaa !20
  %1107 = icmp eq ptr %1106, null
  br i1 %1107, label %1121, label %1108

1108:                                             ; preds = %1104
  %1109 = invoke noundef i1 %1106(ptr noundef nonnull align 8 dereferenceable(32) %42, ptr noundef nonnull align 8 dereferenceable(32) %0, i32 noundef 2)
          to label %1110 unwind label %1112

1110:                                             ; preds = %1108
  %1111 = load <2 x ptr>, ptr %232, align 8, !tbaa !43
  store <2 x ptr> %1111, ptr %946, align 8, !tbaa !43
  br label %1121

1112:                                             ; preds = %1108
  %1113 = landingpad { ptr, i32 }
          cleanup
  %1114 = load ptr, ptr %946, align 8, !tbaa !20
  %1115 = icmp eq ptr %1114, null
  br i1 %1115, label %1667, label %1116

1116:                                             ; preds = %1112
  %1117 = invoke noundef i1 %1114(ptr noundef nonnull align 8 dereferenceable(32) %42, ptr noundef nonnull align 8 dereferenceable(32) %42, i32 noundef 3)
          to label %1667 unwind label %1118

1118:                                             ; preds = %1116
  %1119 = landingpad { ptr, i32 }
          catch ptr null
  %1120 = extractvalue { ptr, i32 } %1119, 0
  call void @__clang_call_terminate(ptr %1120) #21
  unreachable

1121:                                             ; preds = %1110, %1104
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %43, i8 0, i64 32, i1 false)
  %1122 = load ptr, ptr %250, align 8, !tbaa !20
  %1123 = icmp eq ptr %1122, null
  br i1 %1123, label %1137, label %1124

1124:                                             ; preds = %1121
  %1125 = invoke noundef i1 %1122(ptr noundef nonnull align 8 dereferenceable(32) %43, ptr noundef nonnull align 8 dereferenceable(32) %1, i32 noundef 2)
          to label %1126 unwind label %1128

1126:                                             ; preds = %1124
  %1127 = load <2 x ptr>, ptr %250, align 8, !tbaa !43
  store <2 x ptr> %1127, ptr %947, align 8, !tbaa !43
  br label %1137

1128:                                             ; preds = %1124
  %1129 = landingpad { ptr, i32 }
          cleanup
  %1130 = load ptr, ptr %947, align 8, !tbaa !20
  %1131 = icmp eq ptr %1130, null
  br i1 %1131, label %1167, label %1132

1132:                                             ; preds = %1128
  %1133 = invoke noundef i1 %1130(ptr noundef nonnull align 8 dereferenceable(32) %43, ptr noundef nonnull align 8 dereferenceable(32) %43, i32 noundef 3)
          to label %1167 unwind label %1134

1134:                                             ; preds = %1132
  %1135 = landingpad { ptr, i32 }
          catch ptr null
  %1136 = extractvalue { ptr, i32 } %1135, 0
  call void @__clang_call_terminate(ptr %1136) #21
  unreachable

1137:                                             ; preds = %1126, %1121
  invoke fastcc void @_ZL5checkIfEvSt8functionIFT_PS1_jEES4_PfjPKc(ptr noundef %42, ptr noundef %43, ptr noundef %67, ptr noundef nonnull @.str.23)
          to label %1138 unwind label %1158

1138:                                             ; preds = %1137
  %1139 = load ptr, ptr %947, align 8, !tbaa !20
  %1140 = icmp eq ptr %1139, null
  br i1 %1140, label %1146, label %1141

1141:                                             ; preds = %1138
  %1142 = invoke noundef i1 %1139(ptr noundef nonnull align 8 dereferenceable(32) %43, ptr noundef nonnull align 8 dereferenceable(32) %43, i32 noundef 3)
          to label %1146 unwind label %1143

1143:                                             ; preds = %1141
  %1144 = landingpad { ptr, i32 }
          catch ptr null
  %1145 = extractvalue { ptr, i32 } %1144, 0
  call void @__clang_call_terminate(ptr %1145) #21
  unreachable

1146:                                             ; preds = %1138, %1141
  %1147 = load ptr, ptr %946, align 8, !tbaa !20
  %1148 = icmp eq ptr %1147, null
  br i1 %1148, label %1151, label %1149

1149:                                             ; preds = %1146
  %1150 = invoke noundef i1 %1147(ptr noundef nonnull align 8 dereferenceable(32) %42, ptr noundef nonnull align 8 dereferenceable(32) %42, i32 noundef 3)
          to label %1151 unwind label %1152

1151:                                             ; preds = %1146, %1149
  br label %1176

1152:                                             ; preds = %1149
  %1153 = landingpad { ptr, i32 }
          catch ptr null
  %1154 = extractvalue { ptr, i32 } %1153, 0
  call void @__clang_call_terminate(ptr %1154) #21
  unreachable

1155:                                             ; preds = %1281
  %1156 = add nuw nsw i64 %1085, 1
  %1157 = icmp eq i64 %1156, 1024
  br i1 %1157, label %1097, label %1084, !llvm.loop !59

1158:                                             ; preds = %1137
  %1159 = landingpad { ptr, i32 }
          cleanup
  %1160 = load ptr, ptr %947, align 8, !tbaa !20
  %1161 = icmp eq ptr %1160, null
  br i1 %1161, label %1167, label %1162

1162:                                             ; preds = %1158
  %1163 = invoke noundef i1 %1160(ptr noundef nonnull align 8 dereferenceable(32) %43, ptr noundef nonnull align 8 dereferenceable(32) %43, i32 noundef 3)
          to label %1167 unwind label %1164

1164:                                             ; preds = %1162
  %1165 = landingpad { ptr, i32 }
          catch ptr null
  %1166 = extractvalue { ptr, i32 } %1165, 0
  call void @__clang_call_terminate(ptr %1166) #21
  unreachable

1167:                                             ; preds = %1162, %1158, %1132, %1128
  %1168 = phi { ptr, i32 } [ %1129, %1132 ], [ %1129, %1128 ], [ %1159, %1158 ], [ %1159, %1162 ]
  %1169 = load ptr, ptr %946, align 8, !tbaa !20
  %1170 = icmp eq ptr %1169, null
  br i1 %1170, label %1667, label %1171

1171:                                             ; preds = %1167
  %1172 = invoke noundef i1 %1169(ptr noundef nonnull align 8 dereferenceable(32) %42, ptr noundef nonnull align 8 dereferenceable(32) %42, i32 noundef 3)
          to label %1667 unwind label %1173

1173:                                             ; preds = %1171
  %1174 = landingpad { ptr, i32 }
          catch ptr null
  %1175 = extractvalue { ptr, i32 } %1174, 0
  call void @__clang_call_terminate(ptr %1175) #21
  unreachable

1176:                                             ; preds = %1151, %1281
  %1177 = phi i64 [ %1282, %1281 ], [ 1, %1151 ]
  %1178 = add nuw nsw i64 %1177, %1085
  %1179 = icmp samesign ult i64 %1178, 1024
  br i1 %1179, label %1180, label %1182

1180:                                             ; preds = %1176
  %1181 = getelementptr inbounds nuw float, ptr %67, i64 %1178
  store float 1.025000e+03, ptr %1181, align 4, !tbaa !40
  br label %1182

1182:                                             ; preds = %1180, %1176
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %44, i8 0, i64 32, i1 false)
  %1183 = load ptr, ptr %232, align 8, !tbaa !20
  %1184 = icmp eq ptr %1183, null
  br i1 %1184, label %1199, label %1185

1185:                                             ; preds = %1182
  %1186 = invoke noundef i1 %1183(ptr noundef nonnull align 8 dereferenceable(32) %44, ptr noundef nonnull align 8 dereferenceable(32) %0, i32 noundef 2)
          to label %1187 unwind label %1190

1187:                                             ; preds = %1185
  %1188 = load <2 x ptr>, ptr %232, align 8, !tbaa !43
  store <2 x ptr> %1188, ptr %948, align 8, !tbaa !43
  %1189 = extractelement <2 x ptr> %1188, i64 0
  br label %1199

1190:                                             ; preds = %1185
  %1191 = landingpad { ptr, i32 }
          cleanup
  %1192 = load ptr, ptr %948, align 8, !tbaa !20
  %1193 = icmp eq ptr %1192, null
  br i1 %1193, label %1667, label %1194

1194:                                             ; preds = %1190
  %1195 = invoke noundef i1 %1192(ptr noundef nonnull align 8 dereferenceable(32) %44, ptr noundef nonnull align 8 dereferenceable(32) %44, i32 noundef 3)
          to label %1667 unwind label %1196

1196:                                             ; preds = %1194
  %1197 = landingpad { ptr, i32 }
          catch ptr null
  %1198 = extractvalue { ptr, i32 } %1197, 0
  call void @__clang_call_terminate(ptr %1198) #21
  unreachable

1199:                                             ; preds = %1187, %1182
  %1200 = phi ptr [ %1189, %1187 ], [ null, %1182 ]
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %45, i8 0, i64 32, i1 false)
  %1201 = load ptr, ptr %250, align 8, !tbaa !20
  %1202 = icmp eq ptr %1201, null
  br i1 %1202, label %1217, label %1203

1203:                                             ; preds = %1199
  %1204 = invoke noundef i1 %1201(ptr noundef nonnull align 8 dereferenceable(32) %45, ptr noundef nonnull align 8 dereferenceable(32) %1, i32 noundef 2)
          to label %1205 unwind label %1208

1205:                                             ; preds = %1203
  %1206 = load <2 x ptr>, ptr %250, align 8, !tbaa !43
  store <2 x ptr> %1206, ptr %950, align 8, !tbaa !43
  %1207 = load ptr, ptr %948, align 8, !tbaa !20
  br label %1217

1208:                                             ; preds = %1203
  %1209 = landingpad { ptr, i32 }
          cleanup
  %1210 = load ptr, ptr %950, align 8, !tbaa !20
  %1211 = icmp eq ptr %1210, null
  br i1 %1211, label %1297, label %1212

1212:                                             ; preds = %1208
  %1213 = invoke noundef i1 %1210(ptr noundef nonnull align 8 dereferenceable(32) %45, ptr noundef nonnull align 8 dereferenceable(32) %45, i32 noundef 3)
          to label %1297 unwind label %1214

1214:                                             ; preds = %1212
  %1215 = landingpad { ptr, i32 }
          catch ptr null
  %1216 = extractvalue { ptr, i32 } %1215, 0
  call void @__clang_call_terminate(ptr %1216) #21
  unreachable

1217:                                             ; preds = %1205, %1199
  %1218 = phi ptr [ %1207, %1205 ], [ %1200, %1199 ]
  call void @llvm.lifetime.start.p0(ptr nonnull %14)
  call void @llvm.lifetime.start.p0(ptr nonnull %15)
  store ptr %67, ptr %14, align 8, !tbaa !52
  store i32 1024, ptr %15, align 4, !tbaa !33
  %1219 = icmp eq ptr %1218, null
  br i1 %1219, label %1220, label %1222

1220:                                             ; preds = %1225, %1217
  invoke void @_ZSt25__throw_bad_function_callv() #23
          to label %1221 unwind label %1286

1221:                                             ; preds = %1220
  unreachable

1222:                                             ; preds = %1217
  %1223 = load ptr, ptr %949, align 8, !tbaa !16
  %1224 = invoke noundef float %1223(ptr noundef nonnull align 8 dereferenceable(32) %44, ptr noundef nonnull align 8 dereferenceable(8) %14, ptr noundef nonnull align 4 dereferenceable(4) %15)
          to label %1225 unwind label %1284

1225:                                             ; preds = %1222
  call void @llvm.lifetime.end.p0(ptr nonnull %14)
  call void @llvm.lifetime.end.p0(ptr nonnull %15)
  call void @llvm.lifetime.start.p0(ptr nonnull %12)
  call void @llvm.lifetime.start.p0(ptr nonnull %13)
  store ptr %67, ptr %12, align 8, !tbaa !52
  store i32 1024, ptr %13, align 4, !tbaa !33
  %1226 = load ptr, ptr %950, align 8, !tbaa !20
  %1227 = icmp eq ptr %1226, null
  br i1 %1227, label %1220, label %1228

1228:                                             ; preds = %1225
  %1229 = load ptr, ptr %951, align 8, !tbaa !16
  %1230 = invoke noundef float %1229(ptr noundef nonnull align 8 dereferenceable(32) %45, ptr noundef nonnull align 8 dereferenceable(8) %12, ptr noundef nonnull align 4 dereferenceable(4) %13)
          to label %1231 unwind label %1284

1231:                                             ; preds = %1228
  call void @llvm.lifetime.end.p0(ptr nonnull %12)
  call void @llvm.lifetime.end.p0(ptr nonnull %13)
  %1232 = fcmp uno float %1224, %1230
  br i1 %1232, label %1244, label %1233

1233:                                             ; preds = %1231
  %1234 = fcmp oeq float %1224, 0.000000e+00
  br i1 %1234, label %1235, label %1242

1235:                                             ; preds = %1233
  %1236 = fcmp oeq float %1230, 0.000000e+00
  br i1 %1236, label %1237, label %1248

1237:                                             ; preds = %1235
  %1238 = bitcast float %1224 to i32
  %1239 = bitcast float %1230 to i32
  %1240 = xor i32 %1239, %1238
  %1241 = icmp sgt i32 %1240, -1
  br i1 %1241, label %1265, label %1248

1242:                                             ; preds = %1233
  %1243 = fcmp oeq float %1224, %1230
  br i1 %1243, label %1265, label %1248

1244:                                             ; preds = %1231
  %1245 = fcmp uno float %1224, 0.000000e+00
  %1246 = fcmp uno float %1230, 0.000000e+00
  %1247 = and i1 %1245, %1246
  br i1 %1247, label %1265, label %1248

1248:                                             ; preds = %1244, %1242, %1237, %1235
  %1249 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.27, i64 noundef 11)
          to label %1250 unwind label %1286

1250:                                             ; preds = %1248
  %1251 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.23, i64 noundef 4)
          to label %1252 unwind label %1286

1252:                                             ; preds = %1250
  %1253 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.28, i64 noundef 2)
          to label %1254 unwind label %1286

1254:                                             ; preds = %1252
  %1255 = fpext float %1224 to double
  %1256 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo9_M_insertIdEERSoT_(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, double noundef %1255)
          to label %1257 unwind label %1286

1257:                                             ; preds = %1254
  %1258 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %1256, ptr noundef nonnull @.str.29, i64 noundef 4)
          to label %1259 unwind label %1286

1259:                                             ; preds = %1257
  %1260 = fpext float %1230 to double
  %1261 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo9_M_insertIdEERSoT_(ptr noundef nonnull align 8 dereferenceable(8) %1256, double noundef %1260)
          to label %1262 unwind label %1286

1262:                                             ; preds = %1259
  %1263 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) %1261, ptr noundef nonnull @.str.14)
          to label %1264 unwind label %1286

1264:                                             ; preds = %1262
  call void @exit(i32 noundef 1) #24
  unreachable

1265:                                             ; preds = %1244, %1242, %1237
  %1266 = load ptr, ptr %950, align 8, !tbaa !20
  %1267 = icmp eq ptr %1266, null
  br i1 %1267, label %1273, label %1268

1268:                                             ; preds = %1265
  %1269 = invoke noundef i1 %1266(ptr noundef nonnull align 8 dereferenceable(32) %45, ptr noundef nonnull align 8 dereferenceable(32) %45, i32 noundef 3)
          to label %1273 unwind label %1270

1270:                                             ; preds = %1268
  %1271 = landingpad { ptr, i32 }
          catch ptr null
  %1272 = extractvalue { ptr, i32 } %1271, 0
  call void @__clang_call_terminate(ptr %1272) #21
  unreachable

1273:                                             ; preds = %1265, %1268
  %1274 = load ptr, ptr %948, align 8, !tbaa !20
  %1275 = icmp eq ptr %1274, null
  br i1 %1275, label %1281, label %1276

1276:                                             ; preds = %1273
  %1277 = invoke noundef i1 %1274(ptr noundef nonnull align 8 dereferenceable(32) %44, ptr noundef nonnull align 8 dereferenceable(32) %44, i32 noundef 3)
          to label %1281 unwind label %1278

1278:                                             ; preds = %1276
  %1279 = landingpad { ptr, i32 }
          catch ptr null
  %1280 = extractvalue { ptr, i32 } %1279, 0
  call void @__clang_call_terminate(ptr %1280) #21
  unreachable

1281:                                             ; preds = %1273, %1276
  %1282 = add nuw nsw i64 %1177, 1
  %1283 = icmp eq i64 %1282, 16
  br i1 %1283, label %1155, label %1176, !llvm.loop !60

1284:                                             ; preds = %1222, %1228
  %1285 = landingpad { ptr, i32 }
          cleanup
  br label %1288

1286:                                             ; preds = %1220, %1262, %1259, %1257, %1254, %1252, %1250, %1248
  %1287 = landingpad { ptr, i32 }
          cleanup
  br label %1288

1288:                                             ; preds = %1286, %1284
  %1289 = phi { ptr, i32 } [ %1285, %1284 ], [ %1287, %1286 ]
  %1290 = load ptr, ptr %950, align 8, !tbaa !20
  %1291 = icmp eq ptr %1290, null
  br i1 %1291, label %1297, label %1292

1292:                                             ; preds = %1288
  %1293 = invoke noundef i1 %1290(ptr noundef nonnull align 8 dereferenceable(32) %45, ptr noundef nonnull align 8 dereferenceable(32) %45, i32 noundef 3)
          to label %1297 unwind label %1294

1294:                                             ; preds = %1292
  %1295 = landingpad { ptr, i32 }
          catch ptr null
  %1296 = extractvalue { ptr, i32 } %1295, 0
  call void @__clang_call_terminate(ptr %1296) #21
  unreachable

1297:                                             ; preds = %1292, %1288, %1212, %1208
  %1298 = phi { ptr, i32 } [ %1209, %1212 ], [ %1209, %1208 ], [ %1289, %1288 ], [ %1289, %1292 ]
  %1299 = load ptr, ptr %948, align 8, !tbaa !20
  %1300 = icmp eq ptr %1299, null
  br i1 %1300, label %1667, label %1301

1301:                                             ; preds = %1297
  %1302 = invoke noundef i1 %1299(ptr noundef nonnull align 8 dereferenceable(32) %44, ptr noundef nonnull align 8 dereferenceable(32) %44, i32 noundef 3)
          to label %1667 unwind label %1303

1303:                                             ; preds = %1301
  %1304 = landingpad { ptr, i32 }
          catch ptr null
  %1305 = extractvalue { ptr, i32 } %1304, 0
  call void @__clang_call_terminate(ptr %1305) #21
  unreachable

1306:                                             ; preds = %1375, %1097
  %1307 = phi i64 [ 0, %1097 ], [ %1376, %1375 ]
  br label %1308

1308:                                             ; preds = %1308, %1306
  %1309 = phi i64 [ 0, %1306 ], [ %1316, %1308 ]
  %1310 = phi <4 x i32> [ <i32 0, i32 1, i32 2, i32 3>, %1306 ], [ %1317, %1308 ]
  %1311 = add <4 x i32> %1310, splat (i32 4)
  %1312 = uitofp <4 x i32> %1310 to <4 x float>
  %1313 = uitofp <4 x i32> %1311 to <4 x float>
  %1314 = getelementptr inbounds nuw float, ptr %67, i64 %1309
  %1315 = getelementptr inbounds nuw i8, ptr %1314, i64 16
  store <4 x float> %1312, ptr %1314, align 4, !tbaa !40
  store <4 x float> %1313, ptr %1315, align 4, !tbaa !40
  %1316 = add nuw i64 %1309, 8
  %1317 = add <4 x i32> %1310, splat (i32 8)
  %1318 = icmp eq i64 %1316, 1024
  br i1 %1318, label %1324, label %1308, !llvm.loop !61

1319:                                             ; preds = %1375
  %1320 = getelementptr inbounds nuw i8, ptr %50, i64 16
  %1321 = getelementptr inbounds nuw i8, ptr %50, i64 24
  %1322 = getelementptr inbounds nuw i8, ptr %51, i64 16
  %1323 = getelementptr inbounds nuw i8, ptr %51, i64 24
  br label %1526

1324:                                             ; preds = %1308
  %1325 = getelementptr inbounds nuw float, ptr %67, i64 %1307
  store float 0x7FF8000000000000, ptr %1325, align 4, !tbaa !40
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %46, i8 0, i64 32, i1 false)
  %1326 = load ptr, ptr %232, align 8, !tbaa !20
  %1327 = icmp eq ptr %1326, null
  br i1 %1327, label %1341, label %1328

1328:                                             ; preds = %1324
  %1329 = invoke noundef i1 %1326(ptr noundef nonnull align 8 dereferenceable(32) %46, ptr noundef nonnull align 8 dereferenceable(32) %0, i32 noundef 2)
          to label %1330 unwind label %1332

1330:                                             ; preds = %1328
  %1331 = load <2 x ptr>, ptr %232, align 8, !tbaa !43
  store <2 x ptr> %1331, ptr %1098, align 8, !tbaa !43
  br label %1341

1332:                                             ; preds = %1328
  %1333 = landingpad { ptr, i32 }
          cleanup
  %1334 = load ptr, ptr %1098, align 8, !tbaa !20
  %1335 = icmp eq ptr %1334, null
  br i1 %1335, label %1667, label %1336

1336:                                             ; preds = %1332
  %1337 = invoke noundef i1 %1334(ptr noundef nonnull align 8 dereferenceable(32) %46, ptr noundef nonnull align 8 dereferenceable(32) %46, i32 noundef 3)
          to label %1667 unwind label %1338

1338:                                             ; preds = %1336
  %1339 = landingpad { ptr, i32 }
          catch ptr null
  %1340 = extractvalue { ptr, i32 } %1339, 0
  call void @__clang_call_terminate(ptr %1340) #21
  unreachable

1341:                                             ; preds = %1330, %1324
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %47, i8 0, i64 32, i1 false)
  %1342 = load ptr, ptr %250, align 8, !tbaa !20
  %1343 = icmp eq ptr %1342, null
  br i1 %1343, label %1357, label %1344

1344:                                             ; preds = %1341
  %1345 = invoke noundef i1 %1342(ptr noundef nonnull align 8 dereferenceable(32) %47, ptr noundef nonnull align 8 dereferenceable(32) %1, i32 noundef 2)
          to label %1346 unwind label %1348

1346:                                             ; preds = %1344
  %1347 = load <2 x ptr>, ptr %250, align 8, !tbaa !43
  store <2 x ptr> %1347, ptr %1099, align 8, !tbaa !43
  br label %1357

1348:                                             ; preds = %1344
  %1349 = landingpad { ptr, i32 }
          cleanup
  %1350 = load ptr, ptr %1099, align 8, !tbaa !20
  %1351 = icmp eq ptr %1350, null
  br i1 %1351, label %1387, label %1352

1352:                                             ; preds = %1348
  %1353 = invoke noundef i1 %1350(ptr noundef nonnull align 8 dereferenceable(32) %47, ptr noundef nonnull align 8 dereferenceable(32) %47, i32 noundef 3)
          to label %1387 unwind label %1354

1354:                                             ; preds = %1352
  %1355 = landingpad { ptr, i32 }
          catch ptr null
  %1356 = extractvalue { ptr, i32 } %1355, 0
  call void @__clang_call_terminate(ptr %1356) #21
  unreachable

1357:                                             ; preds = %1346, %1341
  invoke fastcc void @_ZL5checkIfEvSt8functionIFT_PS1_jEES4_PfjPKc(ptr noundef %46, ptr noundef %47, ptr noundef %67, ptr noundef nonnull @.str.24)
          to label %1358 unwind label %1378

1358:                                             ; preds = %1357
  %1359 = load ptr, ptr %1099, align 8, !tbaa !20
  %1360 = icmp eq ptr %1359, null
  br i1 %1360, label %1366, label %1361

1361:                                             ; preds = %1358
  %1362 = invoke noundef i1 %1359(ptr noundef nonnull align 8 dereferenceable(32) %47, ptr noundef nonnull align 8 dereferenceable(32) %47, i32 noundef 3)
          to label %1366 unwind label %1363

1363:                                             ; preds = %1361
  %1364 = landingpad { ptr, i32 }
          catch ptr null
  %1365 = extractvalue { ptr, i32 } %1364, 0
  call void @__clang_call_terminate(ptr %1365) #21
  unreachable

1366:                                             ; preds = %1358, %1361
  %1367 = load ptr, ptr %1098, align 8, !tbaa !20
  %1368 = icmp eq ptr %1367, null
  br i1 %1368, label %1371, label %1369

1369:                                             ; preds = %1366
  %1370 = invoke noundef i1 %1367(ptr noundef nonnull align 8 dereferenceable(32) %46, ptr noundef nonnull align 8 dereferenceable(32) %46, i32 noundef 3)
          to label %1371 unwind label %1372

1371:                                             ; preds = %1366, %1369
  br label %1396

1372:                                             ; preds = %1369
  %1373 = landingpad { ptr, i32 }
          catch ptr null
  %1374 = extractvalue { ptr, i32 } %1373, 0
  call void @__clang_call_terminate(ptr %1374) #21
  unreachable

1375:                                             ; preds = %1501
  %1376 = add nuw nsw i64 %1307, 1
  %1377 = icmp eq i64 %1376, 1024
  br i1 %1377, label %1319, label %1306, !llvm.loop !62

1378:                                             ; preds = %1357
  %1379 = landingpad { ptr, i32 }
          cleanup
  %1380 = load ptr, ptr %1099, align 8, !tbaa !20
  %1381 = icmp eq ptr %1380, null
  br i1 %1381, label %1387, label %1382

1382:                                             ; preds = %1378
  %1383 = invoke noundef i1 %1380(ptr noundef nonnull align 8 dereferenceable(32) %47, ptr noundef nonnull align 8 dereferenceable(32) %47, i32 noundef 3)
          to label %1387 unwind label %1384

1384:                                             ; preds = %1382
  %1385 = landingpad { ptr, i32 }
          catch ptr null
  %1386 = extractvalue { ptr, i32 } %1385, 0
  call void @__clang_call_terminate(ptr %1386) #21
  unreachable

1387:                                             ; preds = %1382, %1378, %1352, %1348
  %1388 = phi { ptr, i32 } [ %1349, %1352 ], [ %1349, %1348 ], [ %1379, %1378 ], [ %1379, %1382 ]
  %1389 = load ptr, ptr %1098, align 8, !tbaa !20
  %1390 = icmp eq ptr %1389, null
  br i1 %1390, label %1667, label %1391

1391:                                             ; preds = %1387
  %1392 = invoke noundef i1 %1389(ptr noundef nonnull align 8 dereferenceable(32) %46, ptr noundef nonnull align 8 dereferenceable(32) %46, i32 noundef 3)
          to label %1667 unwind label %1393

1393:                                             ; preds = %1391
  %1394 = landingpad { ptr, i32 }
          catch ptr null
  %1395 = extractvalue { ptr, i32 } %1394, 0
  call void @__clang_call_terminate(ptr %1395) #21
  unreachable

1396:                                             ; preds = %1371, %1501
  %1397 = phi i64 [ %1502, %1501 ], [ 1, %1371 ]
  %1398 = add nuw nsw i64 %1397, %1307
  %1399 = icmp samesign ult i64 %1398, 1024
  br i1 %1399, label %1400, label %1402

1400:                                             ; preds = %1396
  %1401 = getelementptr inbounds nuw float, ptr %67, i64 %1398
  store float 0x7FF8000000000000, ptr %1401, align 4, !tbaa !40
  br label %1402

1402:                                             ; preds = %1400, %1396
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %48, i8 0, i64 32, i1 false)
  %1403 = load ptr, ptr %232, align 8, !tbaa !20
  %1404 = icmp eq ptr %1403, null
  br i1 %1404, label %1419, label %1405

1405:                                             ; preds = %1402
  %1406 = invoke noundef i1 %1403(ptr noundef nonnull align 8 dereferenceable(32) %48, ptr noundef nonnull align 8 dereferenceable(32) %0, i32 noundef 2)
          to label %1407 unwind label %1410

1407:                                             ; preds = %1405
  %1408 = load <2 x ptr>, ptr %232, align 8, !tbaa !43
  store <2 x ptr> %1408, ptr %1100, align 8, !tbaa !43
  %1409 = extractelement <2 x ptr> %1408, i64 0
  br label %1419

1410:                                             ; preds = %1405
  %1411 = landingpad { ptr, i32 }
          cleanup
  %1412 = load ptr, ptr %1100, align 8, !tbaa !20
  %1413 = icmp eq ptr %1412, null
  br i1 %1413, label %1667, label %1414

1414:                                             ; preds = %1410
  %1415 = invoke noundef i1 %1412(ptr noundef nonnull align 8 dereferenceable(32) %48, ptr noundef nonnull align 8 dereferenceable(32) %48, i32 noundef 3)
          to label %1667 unwind label %1416

1416:                                             ; preds = %1414
  %1417 = landingpad { ptr, i32 }
          catch ptr null
  %1418 = extractvalue { ptr, i32 } %1417, 0
  call void @__clang_call_terminate(ptr %1418) #21
  unreachable

1419:                                             ; preds = %1407, %1402
  %1420 = phi ptr [ %1409, %1407 ], [ null, %1402 ]
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %49, i8 0, i64 32, i1 false)
  %1421 = load ptr, ptr %250, align 8, !tbaa !20
  %1422 = icmp eq ptr %1421, null
  br i1 %1422, label %1437, label %1423

1423:                                             ; preds = %1419
  %1424 = invoke noundef i1 %1421(ptr noundef nonnull align 8 dereferenceable(32) %49, ptr noundef nonnull align 8 dereferenceable(32) %1, i32 noundef 2)
          to label %1425 unwind label %1428

1425:                                             ; preds = %1423
  %1426 = load <2 x ptr>, ptr %250, align 8, !tbaa !43
  store <2 x ptr> %1426, ptr %1102, align 8, !tbaa !43
  %1427 = load ptr, ptr %1100, align 8, !tbaa !20
  br label %1437

1428:                                             ; preds = %1423
  %1429 = landingpad { ptr, i32 }
          cleanup
  %1430 = load ptr, ptr %1102, align 8, !tbaa !20
  %1431 = icmp eq ptr %1430, null
  br i1 %1431, label %1517, label %1432

1432:                                             ; preds = %1428
  %1433 = invoke noundef i1 %1430(ptr noundef nonnull align 8 dereferenceable(32) %49, ptr noundef nonnull align 8 dereferenceable(32) %49, i32 noundef 3)
          to label %1517 unwind label %1434

1434:                                             ; preds = %1432
  %1435 = landingpad { ptr, i32 }
          catch ptr null
  %1436 = extractvalue { ptr, i32 } %1435, 0
  call void @__clang_call_terminate(ptr %1436) #21
  unreachable

1437:                                             ; preds = %1425, %1419
  %1438 = phi ptr [ %1427, %1425 ], [ %1420, %1419 ]
  call void @llvm.lifetime.start.p0(ptr nonnull %10)
  call void @llvm.lifetime.start.p0(ptr nonnull %11)
  store ptr %67, ptr %10, align 8, !tbaa !52
  store i32 1024, ptr %11, align 4, !tbaa !33
  %1439 = icmp eq ptr %1438, null
  br i1 %1439, label %1440, label %1442

1440:                                             ; preds = %1445, %1437
  invoke void @_ZSt25__throw_bad_function_callv() #23
          to label %1441 unwind label %1506

1441:                                             ; preds = %1440
  unreachable

1442:                                             ; preds = %1437
  %1443 = load ptr, ptr %1101, align 8, !tbaa !16
  %1444 = invoke noundef float %1443(ptr noundef nonnull align 8 dereferenceable(32) %48, ptr noundef nonnull align 8 dereferenceable(8) %10, ptr noundef nonnull align 4 dereferenceable(4) %11)
          to label %1445 unwind label %1504

1445:                                             ; preds = %1442
  call void @llvm.lifetime.end.p0(ptr nonnull %10)
  call void @llvm.lifetime.end.p0(ptr nonnull %11)
  call void @llvm.lifetime.start.p0(ptr nonnull %8)
  call void @llvm.lifetime.start.p0(ptr nonnull %9)
  store ptr %67, ptr %8, align 8, !tbaa !52
  store i32 1024, ptr %9, align 4, !tbaa !33
  %1446 = load ptr, ptr %1102, align 8, !tbaa !20
  %1447 = icmp eq ptr %1446, null
  br i1 %1447, label %1440, label %1448

1448:                                             ; preds = %1445
  %1449 = load ptr, ptr %1103, align 8, !tbaa !16
  %1450 = invoke noundef float %1449(ptr noundef nonnull align 8 dereferenceable(32) %49, ptr noundef nonnull align 8 dereferenceable(8) %8, ptr noundef nonnull align 4 dereferenceable(4) %9)
          to label %1451 unwind label %1504

1451:                                             ; preds = %1448
  call void @llvm.lifetime.end.p0(ptr nonnull %8)
  call void @llvm.lifetime.end.p0(ptr nonnull %9)
  %1452 = fcmp uno float %1444, %1450
  br i1 %1452, label %1464, label %1453

1453:                                             ; preds = %1451
  %1454 = fcmp oeq float %1444, 0.000000e+00
  br i1 %1454, label %1455, label %1462

1455:                                             ; preds = %1453
  %1456 = fcmp oeq float %1450, 0.000000e+00
  br i1 %1456, label %1457, label %1468

1457:                                             ; preds = %1455
  %1458 = bitcast float %1444 to i32
  %1459 = bitcast float %1450 to i32
  %1460 = xor i32 %1459, %1458
  %1461 = icmp sgt i32 %1460, -1
  br i1 %1461, label %1485, label %1468

1462:                                             ; preds = %1453
  %1463 = fcmp oeq float %1444, %1450
  br i1 %1463, label %1485, label %1468

1464:                                             ; preds = %1451
  %1465 = fcmp uno float %1444, 0.000000e+00
  %1466 = fcmp uno float %1450, 0.000000e+00
  %1467 = and i1 %1465, %1466
  br i1 %1467, label %1485, label %1468

1468:                                             ; preds = %1464, %1462, %1457, %1455
  %1469 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.27, i64 noundef 11)
          to label %1470 unwind label %1506

1470:                                             ; preds = %1468
  %1471 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.25, i64 noundef 22)
          to label %1472 unwind label %1506

1472:                                             ; preds = %1470
  %1473 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.28, i64 noundef 2)
          to label %1474 unwind label %1506

1474:                                             ; preds = %1472
  %1475 = fpext float %1444 to double
  %1476 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo9_M_insertIdEERSoT_(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, double noundef %1475)
          to label %1477 unwind label %1506

1477:                                             ; preds = %1474
  %1478 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %1476, ptr noundef nonnull @.str.29, i64 noundef 4)
          to label %1479 unwind label %1506

1479:                                             ; preds = %1477
  %1480 = fpext float %1450 to double
  %1481 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo9_M_insertIdEERSoT_(ptr noundef nonnull align 8 dereferenceable(8) %1476, double noundef %1480)
          to label %1482 unwind label %1506

1482:                                             ; preds = %1479
  %1483 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) %1481, ptr noundef nonnull @.str.14)
          to label %1484 unwind label %1506

1484:                                             ; preds = %1482
  call void @exit(i32 noundef 1) #24
  unreachable

1485:                                             ; preds = %1464, %1462, %1457
  %1486 = load ptr, ptr %1102, align 8, !tbaa !20
  %1487 = icmp eq ptr %1486, null
  br i1 %1487, label %1493, label %1488

1488:                                             ; preds = %1485
  %1489 = invoke noundef i1 %1486(ptr noundef nonnull align 8 dereferenceable(32) %49, ptr noundef nonnull align 8 dereferenceable(32) %49, i32 noundef 3)
          to label %1493 unwind label %1490

1490:                                             ; preds = %1488
  %1491 = landingpad { ptr, i32 }
          catch ptr null
  %1492 = extractvalue { ptr, i32 } %1491, 0
  call void @__clang_call_terminate(ptr %1492) #21
  unreachable

1493:                                             ; preds = %1485, %1488
  %1494 = load ptr, ptr %1100, align 8, !tbaa !20
  %1495 = icmp eq ptr %1494, null
  br i1 %1495, label %1501, label %1496

1496:                                             ; preds = %1493
  %1497 = invoke noundef i1 %1494(ptr noundef nonnull align 8 dereferenceable(32) %48, ptr noundef nonnull align 8 dereferenceable(32) %48, i32 noundef 3)
          to label %1501 unwind label %1498

1498:                                             ; preds = %1496
  %1499 = landingpad { ptr, i32 }
          catch ptr null
  %1500 = extractvalue { ptr, i32 } %1499, 0
  call void @__clang_call_terminate(ptr %1500) #21
  unreachable

1501:                                             ; preds = %1493, %1496
  %1502 = add nuw nsw i64 %1397, 1
  %1503 = icmp eq i64 %1502, 16
  br i1 %1503, label %1375, label %1396, !llvm.loop !63

1504:                                             ; preds = %1442, %1448
  %1505 = landingpad { ptr, i32 }
          cleanup
  br label %1508

1506:                                             ; preds = %1440, %1482, %1479, %1477, %1474, %1472, %1470, %1468
  %1507 = landingpad { ptr, i32 }
          cleanup
  br label %1508

1508:                                             ; preds = %1506, %1504
  %1509 = phi { ptr, i32 } [ %1505, %1504 ], [ %1507, %1506 ]
  %1510 = load ptr, ptr %1102, align 8, !tbaa !20
  %1511 = icmp eq ptr %1510, null
  br i1 %1511, label %1517, label %1512

1512:                                             ; preds = %1508
  %1513 = invoke noundef i1 %1510(ptr noundef nonnull align 8 dereferenceable(32) %49, ptr noundef nonnull align 8 dereferenceable(32) %49, i32 noundef 3)
          to label %1517 unwind label %1514

1514:                                             ; preds = %1512
  %1515 = landingpad { ptr, i32 }
          catch ptr null
  %1516 = extractvalue { ptr, i32 } %1515, 0
  call void @__clang_call_terminate(ptr %1516) #21
  unreachable

1517:                                             ; preds = %1512, %1508, %1432, %1428
  %1518 = phi { ptr, i32 } [ %1429, %1432 ], [ %1429, %1428 ], [ %1509, %1508 ], [ %1509, %1512 ]
  %1519 = load ptr, ptr %1100, align 8, !tbaa !20
  %1520 = icmp eq ptr %1519, null
  br i1 %1520, label %1667, label %1521

1521:                                             ; preds = %1517
  %1522 = invoke noundef i1 %1519(ptr noundef nonnull align 8 dereferenceable(32) %48, ptr noundef nonnull align 8 dereferenceable(32) %48, i32 noundef 3)
          to label %1667 unwind label %1523

1523:                                             ; preds = %1521
  %1524 = landingpad { ptr, i32 }
          catch ptr null
  %1525 = extractvalue { ptr, i32 } %1524, 0
  call void @__clang_call_terminate(ptr %1525) #21
  unreachable

1526:                                             ; preds = %1538, %1319
  %1527 = phi i64 [ 0, %1319 ], [ %1539, %1538 ]
  br label %1528

1528:                                             ; preds = %1528, %1526
  %1529 = phi i64 [ 0, %1526 ], [ %1532, %1528 ]
  %1530 = getelementptr inbounds nuw float, ptr %67, i64 %1529
  %1531 = getelementptr inbounds nuw i8, ptr %1530, i64 16
  store <4 x float> splat (float -1.000000e+00), ptr %1530, align 4, !tbaa !40
  store <4 x float> splat (float -1.000000e+00), ptr %1531, align 4, !tbaa !40
  %1532 = add nuw i64 %1529, 8
  %1533 = icmp eq i64 %1532, 1024
  br i1 %1533, label %1535, label %1528, !llvm.loop !64

1534:                                             ; preds = %1538
  call void @_ZdaPv(ptr noundef nonnull %67) #25
  ret void

1535:                                             ; preds = %1528
  %1536 = getelementptr inbounds nuw float, ptr %67, i64 %1527
  %1537 = getelementptr inbounds nuw float, ptr %67, i64 %1527
  br label %1541

1538:                                             ; preds = %1642
  %1539 = add nuw nsw i64 %1527, 1
  %1540 = icmp eq i64 %1539, 64
  br i1 %1540, label %1534, label %1526, !llvm.loop !65

1541:                                             ; preds = %1535, %1642
  %1542 = phi i64 [ 1, %1535 ], [ %1643, %1642 ]
  store float 0xFFF0000000000000, ptr %1536, align 4, !tbaa !40
  %1543 = getelementptr inbounds nuw float, ptr %1537, i64 %1542
  store float 0x7FF0000000000000, ptr %1543, align 4, !tbaa !40
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %50, i8 0, i64 32, i1 false)
  %1544 = load ptr, ptr %232, align 8, !tbaa !20
  %1545 = icmp eq ptr %1544, null
  br i1 %1545, label %1560, label %1546

1546:                                             ; preds = %1541
  %1547 = invoke noundef i1 %1544(ptr noundef nonnull align 8 dereferenceable(32) %50, ptr noundef nonnull align 8 dereferenceable(32) %0, i32 noundef 2)
          to label %1548 unwind label %1551

1548:                                             ; preds = %1546
  %1549 = load <2 x ptr>, ptr %232, align 8, !tbaa !43
  store <2 x ptr> %1549, ptr %1320, align 8, !tbaa !43
  %1550 = extractelement <2 x ptr> %1549, i64 0
  br label %1560

1551:                                             ; preds = %1546
  %1552 = landingpad { ptr, i32 }
          cleanup
  %1553 = load ptr, ptr %1320, align 8, !tbaa !20
  %1554 = icmp eq ptr %1553, null
  br i1 %1554, label %1667, label %1555

1555:                                             ; preds = %1551
  %1556 = invoke noundef i1 %1553(ptr noundef nonnull align 8 dereferenceable(32) %50, ptr noundef nonnull align 8 dereferenceable(32) %50, i32 noundef 3)
          to label %1667 unwind label %1557

1557:                                             ; preds = %1555
  %1558 = landingpad { ptr, i32 }
          catch ptr null
  %1559 = extractvalue { ptr, i32 } %1558, 0
  call void @__clang_call_terminate(ptr %1559) #21
  unreachable

1560:                                             ; preds = %1548, %1541
  %1561 = phi ptr [ %1550, %1548 ], [ null, %1541 ]
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %51, i8 0, i64 32, i1 false)
  %1562 = load ptr, ptr %250, align 8, !tbaa !20
  %1563 = icmp eq ptr %1562, null
  br i1 %1563, label %1578, label %1564

1564:                                             ; preds = %1560
  %1565 = invoke noundef i1 %1562(ptr noundef nonnull align 8 dereferenceable(32) %51, ptr noundef nonnull align 8 dereferenceable(32) %1, i32 noundef 2)
          to label %1566 unwind label %1569

1566:                                             ; preds = %1564
  %1567 = load <2 x ptr>, ptr %250, align 8, !tbaa !43
  store <2 x ptr> %1567, ptr %1322, align 8, !tbaa !43
  %1568 = load ptr, ptr %1320, align 8, !tbaa !20
  br label %1578

1569:                                             ; preds = %1564
  %1570 = landingpad { ptr, i32 }
          cleanup
  %1571 = load ptr, ptr %1322, align 8, !tbaa !20
  %1572 = icmp eq ptr %1571, null
  br i1 %1572, label %1658, label %1573

1573:                                             ; preds = %1569
  %1574 = invoke noundef i1 %1571(ptr noundef nonnull align 8 dereferenceable(32) %51, ptr noundef nonnull align 8 dereferenceable(32) %51, i32 noundef 3)
          to label %1658 unwind label %1575

1575:                                             ; preds = %1573
  %1576 = landingpad { ptr, i32 }
          catch ptr null
  %1577 = extractvalue { ptr, i32 } %1576, 0
  call void @__clang_call_terminate(ptr %1577) #21
  unreachable

1578:                                             ; preds = %1566, %1560
  %1579 = phi ptr [ %1568, %1566 ], [ %1561, %1560 ]
  call void @llvm.lifetime.start.p0(ptr nonnull %6)
  call void @llvm.lifetime.start.p0(ptr nonnull %7)
  store ptr %67, ptr %6, align 8, !tbaa !52
  store i32 1024, ptr %7, align 4, !tbaa !33
  %1580 = icmp eq ptr %1579, null
  br i1 %1580, label %1581, label %1583

1581:                                             ; preds = %1586, %1578
  invoke void @_ZSt25__throw_bad_function_callv() #23
          to label %1582 unwind label %1647

1582:                                             ; preds = %1581
  unreachable

1583:                                             ; preds = %1578
  %1584 = load ptr, ptr %1321, align 8, !tbaa !16
  %1585 = invoke noundef float %1584(ptr noundef nonnull align 8 dereferenceable(32) %50, ptr noundef nonnull align 8 dereferenceable(8) %6, ptr noundef nonnull align 4 dereferenceable(4) %7)
          to label %1586 unwind label %1645

1586:                                             ; preds = %1583
  call void @llvm.lifetime.end.p0(ptr nonnull %6)
  call void @llvm.lifetime.end.p0(ptr nonnull %7)
  call void @llvm.lifetime.start.p0(ptr nonnull %4)
  call void @llvm.lifetime.start.p0(ptr nonnull %5)
  store ptr %67, ptr %4, align 8, !tbaa !52
  store i32 1024, ptr %5, align 4, !tbaa !33
  %1587 = load ptr, ptr %1322, align 8, !tbaa !20
  %1588 = icmp eq ptr %1587, null
  br i1 %1588, label %1581, label %1589

1589:                                             ; preds = %1586
  %1590 = load ptr, ptr %1323, align 8, !tbaa !16
  %1591 = invoke noundef float %1590(ptr noundef nonnull align 8 dereferenceable(32) %51, ptr noundef nonnull align 8 dereferenceable(8) %4, ptr noundef nonnull align 4 dereferenceable(4) %5)
          to label %1592 unwind label %1645

1592:                                             ; preds = %1589
  call void @llvm.lifetime.end.p0(ptr nonnull %4)
  call void @llvm.lifetime.end.p0(ptr nonnull %5)
  %1593 = fcmp uno float %1585, %1591
  br i1 %1593, label %1605, label %1594

1594:                                             ; preds = %1592
  %1595 = fcmp oeq float %1585, 0.000000e+00
  br i1 %1595, label %1596, label %1603

1596:                                             ; preds = %1594
  %1597 = fcmp oeq float %1591, 0.000000e+00
  br i1 %1597, label %1598, label %1609

1598:                                             ; preds = %1596
  %1599 = bitcast float %1585 to i32
  %1600 = bitcast float %1591 to i32
  %1601 = xor i32 %1600, %1599
  %1602 = icmp sgt i32 %1601, -1
  br i1 %1602, label %1626, label %1609

1603:                                             ; preds = %1594
  %1604 = fcmp oeq float %1585, %1591
  br i1 %1604, label %1626, label %1609

1605:                                             ; preds = %1592
  %1606 = fcmp uno float %1585, 0.000000e+00
  %1607 = fcmp uno float %1591, 0.000000e+00
  %1608 = and i1 %1606, %1607
  br i1 %1608, label %1626, label %1609

1609:                                             ; preds = %1605, %1603, %1598, %1596
  %1610 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.27, i64 noundef 11)
          to label %1611 unwind label %1647

1611:                                             ; preds = %1609
  %1612 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.26, i64 noundef 8)
          to label %1613 unwind label %1647

1613:                                             ; preds = %1611
  %1614 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.28, i64 noundef 2)
          to label %1615 unwind label %1647

1615:                                             ; preds = %1613
  %1616 = fpext float %1585 to double
  %1617 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo9_M_insertIdEERSoT_(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, double noundef %1616)
          to label %1618 unwind label %1647

1618:                                             ; preds = %1615
  %1619 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %1617, ptr noundef nonnull @.str.29, i64 noundef 4)
          to label %1620 unwind label %1647

1620:                                             ; preds = %1618
  %1621 = fpext float %1591 to double
  %1622 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo9_M_insertIdEERSoT_(ptr noundef nonnull align 8 dereferenceable(8) %1617, double noundef %1621)
          to label %1623 unwind label %1647

1623:                                             ; preds = %1620
  %1624 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) %1622, ptr noundef nonnull @.str.14)
          to label %1625 unwind label %1647

1625:                                             ; preds = %1623
  call void @exit(i32 noundef 1) #24
  unreachable

1626:                                             ; preds = %1605, %1603, %1598
  %1627 = load ptr, ptr %1322, align 8, !tbaa !20
  %1628 = icmp eq ptr %1627, null
  br i1 %1628, label %1634, label %1629

1629:                                             ; preds = %1626
  %1630 = invoke noundef i1 %1627(ptr noundef nonnull align 8 dereferenceable(32) %51, ptr noundef nonnull align 8 dereferenceable(32) %51, i32 noundef 3)
          to label %1634 unwind label %1631

1631:                                             ; preds = %1629
  %1632 = landingpad { ptr, i32 }
          catch ptr null
  %1633 = extractvalue { ptr, i32 } %1632, 0
  call void @__clang_call_terminate(ptr %1633) #21
  unreachable

1634:                                             ; preds = %1626, %1629
  %1635 = load ptr, ptr %1320, align 8, !tbaa !20
  %1636 = icmp eq ptr %1635, null
  br i1 %1636, label %1642, label %1637

1637:                                             ; preds = %1634
  %1638 = invoke noundef i1 %1635(ptr noundef nonnull align 8 dereferenceable(32) %50, ptr noundef nonnull align 8 dereferenceable(32) %50, i32 noundef 3)
          to label %1642 unwind label %1639

1639:                                             ; preds = %1637
  %1640 = landingpad { ptr, i32 }
          catch ptr null
  %1641 = extractvalue { ptr, i32 } %1640, 0
  call void @__clang_call_terminate(ptr %1641) #21
  unreachable

1642:                                             ; preds = %1634, %1637
  %1643 = add nuw nsw i64 %1542, 1
  %1644 = icmp eq i64 %1643, 16
  br i1 %1644, label %1538, label %1541, !llvm.loop !66

1645:                                             ; preds = %1583, %1589
  %1646 = landingpad { ptr, i32 }
          cleanup
  br label %1649

1647:                                             ; preds = %1581, %1623, %1620, %1618, %1615, %1613, %1611, %1609
  %1648 = landingpad { ptr, i32 }
          cleanup
  br label %1649

1649:                                             ; preds = %1647, %1645
  %1650 = phi { ptr, i32 } [ %1646, %1645 ], [ %1648, %1647 ]
  %1651 = load ptr, ptr %1322, align 8, !tbaa !20
  %1652 = icmp eq ptr %1651, null
  br i1 %1652, label %1658, label %1653

1653:                                             ; preds = %1649
  %1654 = invoke noundef i1 %1651(ptr noundef nonnull align 8 dereferenceable(32) %51, ptr noundef nonnull align 8 dereferenceable(32) %51, i32 noundef 3)
          to label %1658 unwind label %1655

1655:                                             ; preds = %1653
  %1656 = landingpad { ptr, i32 }
          catch ptr null
  %1657 = extractvalue { ptr, i32 } %1656, 0
  call void @__clang_call_terminate(ptr %1657) #21
  unreachable

1658:                                             ; preds = %1653, %1649, %1573, %1569
  %1659 = phi { ptr, i32 } [ %1570, %1573 ], [ %1570, %1569 ], [ %1650, %1649 ], [ %1650, %1653 ]
  %1660 = load ptr, ptr %1320, align 8, !tbaa !20
  %1661 = icmp eq ptr %1660, null
  br i1 %1661, label %1667, label %1662

1662:                                             ; preds = %1658
  %1663 = invoke noundef i1 %1660(ptr noundef nonnull align 8 dereferenceable(32) %50, ptr noundef nonnull align 8 dereferenceable(32) %50, i32 noundef 3)
          to label %1667 unwind label %1664

1664:                                             ; preds = %1662
  %1665 = landingpad { ptr, i32 }
          catch ptr null
  %1666 = extractvalue { ptr, i32 } %1665, 0
  call void @__clang_call_terminate(ptr %1666) #21
  unreachable

1667:                                             ; preds = %1662, %1658, %1555, %1551, %1521, %1517, %1410, %1414, %1391, %1387, %1336, %1332, %1301, %1297, %1190, %1194, %1171, %1167, %1116, %1112, %1079, %1075, %972, %968, %932, %928, %825, %821, %787, %783, %733, %729, %716, %712, %648, %644, %626, %630, %568, %564, %546, %550, %488, %484, %466, %470, %412, %408, %394, %390, %243, %239, %374, %370, %379, %312, %316
  %1668 = phi { ptr, i32 } [ %240, %243 ], [ %240, %239 ], [ %313, %316 ], [ %313, %312 ], [ %371, %374 ], [ %371, %370 ], [ %380, %379 ], [ %391, %390 ], [ %391, %394 ], [ %409, %408 ], [ %409, %412 ], [ %467, %470 ], [ %467, %466 ], [ %485, %484 ], [ %485, %488 ], [ %547, %550 ], [ %547, %546 ], [ %565, %564 ], [ %565, %568 ], [ %627, %630 ], [ %627, %626 ], [ %645, %644 ], [ %645, %648 ], [ %713, %712 ], [ %713, %716 ], [ %730, %733 ], [ %730, %729 ], [ %784, %783 ], [ %784, %787 ], [ %822, %825 ], [ %822, %821 ], [ %929, %928 ], [ %929, %932 ], [ %969, %972 ], [ %969, %968 ], [ %1076, %1075 ], [ %1076, %1079 ], [ %1113, %1116 ], [ %1113, %1112 ], [ %1168, %1167 ], [ %1168, %1171 ], [ %1191, %1194 ], [ %1191, %1190 ], [ %1298, %1297 ], [ %1298, %1301 ], [ %1333, %1336 ], [ %1333, %1332 ], [ %1388, %1387 ], [ %1388, %1391 ], [ %1411, %1414 ], [ %1411, %1410 ], [ %1518, %1517 ], [ %1518, %1521 ], [ %1552, %1555 ], [ %1552, %1551 ], [ %1659, %1658 ], [ %1659, %1662 ]
  call void @_ZdaPv(ptr noundef nonnull %67) #25
  resume { ptr, i32 } %1668
}

declare i32 @__gxx_personality_v0(...)

; Function Attrs: noinline noreturn nounwind uwtable
define linkonce_odr hidden void @__clang_call_terminate(ptr noundef %0) local_unnamed_addr #3 comdat {
  %2 = tail call ptr @__cxa_begin_catch(ptr %0) #20
  tail call void @_ZSt9terminatev() #21
  unreachable
}

declare ptr @__cxa_begin_catch(ptr) local_unnamed_addr

; Function Attrs: cold nofree noreturn
declare void @_ZSt9terminatev() local_unnamed_addr #4

; Function Attrs: inlinehint mustprogress uwtable
declare noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8), ptr noundef) local_unnamed_addr #5

; Function Attrs: nobuiltin allocsize(0)
declare noundef nonnull ptr @_Znam(i64 noundef) local_unnamed_addr #6

; Function Attrs: mustprogress norecurse uwtable
define internal fastcc void @_ZL5checkIfEvSt8functionIFT_PS1_jEES4_PfjPKc(ptr noundef nonnull %0, ptr noundef nonnull %1, ptr noundef nonnull %2, ptr noundef %3) unnamed_addr #0 {
  %5 = alloca ptr, align 8
  %6 = alloca i32, align 4
  %7 = alloca ptr, align 8
  %8 = alloca i32, align 4
  call void @llvm.lifetime.start.p0(ptr nonnull %7)
  call void @llvm.lifetime.start.p0(ptr nonnull %8)
  store ptr %2, ptr %7, align 8, !tbaa !52
  store i32 1024, ptr %8, align 4, !tbaa !33
  %9 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %10 = load ptr, ptr %9, align 8, !tbaa !20
  %11 = icmp eq ptr %10, null
  br i1 %11, label %12, label %13

12:                                               ; preds = %4
  tail call void @_ZSt25__throw_bad_function_callv() #23
  unreachable

13:                                               ; preds = %4
  %14 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %15 = load ptr, ptr %14, align 8, !tbaa !16
  %16 = call noundef float %15(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %7, ptr noundef nonnull align 4 dereferenceable(4) %8)
  call void @llvm.lifetime.end.p0(ptr nonnull %7)
  call void @llvm.lifetime.end.p0(ptr nonnull %8)
  call void @llvm.lifetime.start.p0(ptr nonnull %5)
  call void @llvm.lifetime.start.p0(ptr nonnull %6)
  store ptr %2, ptr %5, align 8, !tbaa !52
  store i32 1024, ptr %6, align 4, !tbaa !33
  %17 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %18 = load ptr, ptr %17, align 8, !tbaa !20
  %19 = icmp eq ptr %18, null
  br i1 %19, label %20, label %21

20:                                               ; preds = %13
  call void @_ZSt25__throw_bad_function_callv() #23
  unreachable

21:                                               ; preds = %13
  %22 = getelementptr inbounds nuw i8, ptr %1, i64 24
  %23 = load ptr, ptr %22, align 8, !tbaa !16
  %24 = call noundef float %23(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %5, ptr noundef nonnull align 4 dereferenceable(4) %6)
  call void @llvm.lifetime.end.p0(ptr nonnull %5)
  call void @llvm.lifetime.end.p0(ptr nonnull %6)
  %25 = fcmp uno float %16, %24
  br i1 %25, label %37, label %26

26:                                               ; preds = %21
  %27 = fcmp oeq float %16, 0.000000e+00
  br i1 %27, label %28, label %35

28:                                               ; preds = %26
  %29 = fcmp oeq float %24, 0.000000e+00
  br i1 %29, label %30, label %41

30:                                               ; preds = %28
  %31 = bitcast float %16 to i32
  %32 = bitcast float %24 to i32
  %33 = xor i32 %32, %31
  %34 = icmp sgt i32 %33, -1
  br i1 %34, label %49, label %41

35:                                               ; preds = %26
  %36 = fcmp oeq float %16, %24
  br i1 %36, label %49, label %41

37:                                               ; preds = %21
  %38 = fcmp uno float %16, 0.000000e+00
  %39 = fcmp uno float %24, 0.000000e+00
  %40 = and i1 %38, %39
  br i1 %40, label %49, label %41

41:                                               ; preds = %28, %30, %35, %37
  %42 = call noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.27)
  %43 = call noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) %42, ptr noundef %3)
  %44 = call noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) %43, ptr noundef nonnull @.str.28)
  %45 = call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEf(ptr noundef nonnull align 8 dereferenceable(8) %44, float noundef %16)
  %46 = call noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) %45, ptr noundef nonnull @.str.29)
  %47 = call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEf(ptr noundef nonnull align 8 dereferenceable(8) %46, float noundef %24)
  %48 = call noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) %47, ptr noundef nonnull @.str.14)
  call void @exit(i32 noundef 1) #24
  unreachable

49:                                               ; preds = %30, %35, %37
  ret void
}

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8), ptr noundef, i64 noundef) local_unnamed_addr #7

declare void @_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate(ptr noundef nonnull align 8 dereferenceable(264), i32 noundef) local_unnamed_addr #7

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: read)
declare i64 @strlen(ptr noundef captures(none)) local_unnamed_addr #8

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fmuladd.f32(float, float, float) #9

; Function Attrs: nounwind
declare float @nextafterf(float noundef, float noundef) local_unnamed_addr #10

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZSt16__introsort_loopIPflN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_(ptr noundef %0, ptr noundef %1, i64 noundef %2, i8 %3) local_unnamed_addr #11 comdat {
  %5 = alloca %"struct.__gnu_cxx::__ops::_Iter_less_iter", align 1
  %6 = ptrtoint ptr %0 to i64
  %7 = ptrtoint ptr %1 to i64
  %8 = sub i64 %7, %6
  %9 = icmp sgt i64 %8, 64
  br i1 %9, label %10, label %126

10:                                               ; preds = %4
  %11 = getelementptr inbounds nuw i8, ptr %0, i64 4
  br label %12

12:                                               ; preds = %10, %122
  %13 = phi i64 [ %8, %10 ], [ %124, %122 ]
  %14 = phi ptr [ %1, %10 ], [ %110, %122 ]
  %15 = phi i64 [ %2, %10 ], [ %78, %122 ]
  %16 = icmp eq i64 %15, 0
  br i1 %16, label %17, label %77

17:                                               ; preds = %12
  call void @llvm.lifetime.start.p0(ptr nonnull %5)
  call void @_ZSt11__make_heapIPfN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_RT0_(ptr noundef %0, ptr noundef %14, ptr noundef nonnull align 1 dereferenceable(1) %5)
  call void @llvm.lifetime.end.p0(ptr nonnull %5)
  br label %18

18:                                               ; preds = %17, %73
  %19 = phi ptr [ %20, %73 ], [ %14, %17 ]
  %20 = getelementptr inbounds i8, ptr %19, i64 -4
  %21 = load float, ptr %20, align 4, !tbaa !40
  %22 = load float, ptr %0, align 4, !tbaa !40
  store float %22, ptr %20, align 4, !tbaa !40
  %23 = ptrtoint ptr %20 to i64
  %24 = sub i64 %23, %6
  %25 = ashr exact i64 %24, 2
  %26 = add nsw i64 %25, -1
  %27 = sdiv i64 %26, 2
  %28 = icmp sgt i64 %25, 2
  br i1 %28, label %29, label %45

29:                                               ; preds = %18, %29
  %30 = phi i64 [ %40, %29 ], [ 0, %18 ]
  %31 = shl i64 %30, 1
  %32 = add i64 %31, 2
  %33 = getelementptr inbounds float, ptr %0, i64 %32
  %34 = getelementptr float, ptr %0, i64 %31
  %35 = getelementptr i8, ptr %34, i64 4
  %36 = load float, ptr %33, align 4, !tbaa !40
  %37 = load float, ptr %35, align 4, !tbaa !40
  %38 = fcmp olt float %36, %37
  %39 = or disjoint i64 %31, 1
  %40 = select i1 %38, i64 %39, i64 %32
  %41 = getelementptr inbounds float, ptr %0, i64 %40
  %42 = load float, ptr %41, align 4, !tbaa !40
  %43 = getelementptr inbounds float, ptr %0, i64 %30
  store float %42, ptr %43, align 4, !tbaa !40
  %44 = icmp slt i64 %40, %27
  br i1 %44, label %29, label %45, !llvm.loop !67

45:                                               ; preds = %29, %18
  %46 = phi i64 [ 0, %18 ], [ %40, %29 ]
  %47 = and i64 %24, 4
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
  %56 = getelementptr inbounds nuw float, ptr %0, i64 %55
  %57 = load float, ptr %56, align 4, !tbaa !40
  %58 = getelementptr inbounds float, ptr %0, i64 %46
  store float %57, ptr %58, align 4, !tbaa !40
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
  %67 = getelementptr inbounds nuw float, ptr %0, i64 %66
  %68 = load float, ptr %67, align 4, !tbaa !40
  %69 = fcmp olt float %68, %21
  br i1 %69, label %70, label %73

70:                                               ; preds = %63
  %71 = getelementptr inbounds float, ptr %0, i64 %64
  store float %68, ptr %71, align 4, !tbaa !40
  %72 = icmp ult i64 %65, 2
  br i1 %72, label %73, label %63, !llvm.loop !68

73:                                               ; preds = %70, %63, %59
  %74 = phi i64 [ 0, %59 ], [ %64, %63 ], [ 0, %70 ]
  %75 = getelementptr inbounds float, ptr %0, i64 %74
  store float %21, ptr %75, align 4, !tbaa !40
  %76 = icmp sgt i64 %24, 4
  br i1 %76, label %18, label %126, !llvm.loop !69

77:                                               ; preds = %12
  %78 = add nsw i64 %15, -1
  %79 = lshr i64 %13, 3
  %80 = getelementptr inbounds nuw float, ptr %0, i64 %79
  %81 = getelementptr inbounds i8, ptr %14, i64 -4
  %82 = load float, ptr %11, align 4, !tbaa !40
  %83 = load float, ptr %80, align 4, !tbaa !40
  %84 = fcmp olt float %82, %83
  %85 = load float, ptr %81, align 4, !tbaa !40
  br i1 %84, label %86, label %95

86:                                               ; preds = %77
  %87 = fcmp olt float %83, %85
  br i1 %87, label %88, label %90

88:                                               ; preds = %86
  %89 = load float, ptr %0, align 4, !tbaa !40
  store float %83, ptr %0, align 4, !tbaa !40
  store float %89, ptr %80, align 4, !tbaa !40
  br label %104

90:                                               ; preds = %86
  %91 = fcmp olt float %82, %85
  %92 = load float, ptr %0, align 4, !tbaa !40
  br i1 %91, label %93, label %94

93:                                               ; preds = %90
  store float %85, ptr %0, align 4, !tbaa !40
  store float %92, ptr %81, align 4, !tbaa !40
  br label %104

94:                                               ; preds = %90
  store float %82, ptr %0, align 4, !tbaa !40
  store float %92, ptr %11, align 4, !tbaa !40
  br label %104

95:                                               ; preds = %77
  %96 = fcmp olt float %82, %85
  br i1 %96, label %97, label %99

97:                                               ; preds = %95
  %98 = load float, ptr %0, align 4, !tbaa !40
  store float %82, ptr %0, align 4, !tbaa !40
  store float %98, ptr %11, align 4, !tbaa !40
  br label %104

99:                                               ; preds = %95
  %100 = fcmp olt float %83, %85
  %101 = load float, ptr %0, align 4, !tbaa !40
  br i1 %100, label %102, label %103

102:                                              ; preds = %99
  store float %85, ptr %0, align 4, !tbaa !40
  store float %101, ptr %81, align 4, !tbaa !40
  br label %104

103:                                              ; preds = %99
  store float %83, ptr %0, align 4, !tbaa !40
  store float %101, ptr %80, align 4, !tbaa !40
  br label %104

104:                                              ; preds = %103, %102, %97, %94, %93, %88
  br label %105

105:                                              ; preds = %104, %121
  %106 = phi ptr [ %116, %121 ], [ %14, %104 ]
  %107 = phi ptr [ %113, %121 ], [ %11, %104 ]
  %108 = load float, ptr %0, align 4, !tbaa !40
  br label %109

109:                                              ; preds = %109, %105
  %110 = phi ptr [ %107, %105 ], [ %113, %109 ]
  %111 = load float, ptr %110, align 4, !tbaa !40
  %112 = fcmp olt float %111, %108
  %113 = getelementptr inbounds nuw i8, ptr %110, i64 4
  br i1 %112, label %109, label %114, !llvm.loop !70

114:                                              ; preds = %109, %114
  %115 = phi ptr [ %116, %114 ], [ %106, %109 ]
  %116 = getelementptr inbounds i8, ptr %115, i64 -4
  %117 = load float, ptr %116, align 4, !tbaa !40
  %118 = fcmp olt float %108, %117
  br i1 %118, label %114, label %119, !llvm.loop !71

119:                                              ; preds = %114
  %120 = icmp ult ptr %110, %116
  br i1 %120, label %121, label %122

121:                                              ; preds = %119
  store float %117, ptr %110, align 4, !tbaa !40
  store float %111, ptr %116, align 4, !tbaa !40
  br label %105, !llvm.loop !72

122:                                              ; preds = %119
  tail call void @_ZSt16__introsort_loopIPflN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_(ptr noundef nonnull %110, ptr noundef %14, i64 noundef %78, i8 undef)
  %123 = ptrtoint ptr %110 to i64
  %124 = sub i64 %123, %6
  %125 = icmp sgt i64 %124, 64
  br i1 %125, label %12, label %126, !llvm.loop !73

126:                                              ; preds = %122, %73, %4
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZSt22__final_insertion_sortIPfN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_(ptr noundef %0, ptr noundef %1, i8 %2) local_unnamed_addr #11 comdat {
  %4 = ptrtoint ptr %1 to i64
  %5 = ptrtoint ptr %0 to i64
  %6 = sub i64 %4, %5
  %7 = icmp sgt i64 %6, 64
  br i1 %7, label %8, label %56

8:                                                ; preds = %3
  %9 = getelementptr i8, ptr %0, i64 4
  br label %10

10:                                               ; preds = %32, %8
  %11 = phi i64 [ 4, %8 ], [ %34, %32 ]
  %12 = phi ptr [ %0, %8 ], [ %13, %32 ]
  %13 = getelementptr inbounds nuw i8, ptr %0, i64 %11
  %14 = load float, ptr %13, align 4, !tbaa !40
  %15 = load float, ptr %0, align 4, !tbaa !40
  %16 = fcmp olt float %14, %15
  br i1 %16, label %17, label %22

17:                                               ; preds = %10
  %18 = icmp samesign ugt i64 %11, 4
  br i1 %18, label %19, label %20, !prof !34

19:                                               ; preds = %17
  tail call void @llvm.memmove.p0.p0.i64(ptr noundef nonnull align 4 dereferenceable(1) %9, ptr noundef nonnull align 4 dereferenceable(1) %0, i64 %11, i1 false)
  br label %32

20:                                               ; preds = %17
  %21 = getelementptr inbounds nuw i8, ptr %12, i64 4
  store float %15, ptr %21, align 4, !tbaa !40
  br label %32

22:                                               ; preds = %10
  %23 = load float, ptr %12, align 4, !tbaa !40
  %24 = fcmp olt float %14, %23
  br i1 %24, label %25, label %32

25:                                               ; preds = %22, %25
  %26 = phi float [ %30, %25 ], [ %23, %22 ]
  %27 = phi ptr [ %29, %25 ], [ %12, %22 ]
  %28 = phi ptr [ %27, %25 ], [ %13, %22 ]
  store float %26, ptr %28, align 4, !tbaa !40
  %29 = getelementptr inbounds i8, ptr %27, i64 -4
  %30 = load float, ptr %29, align 4, !tbaa !40
  %31 = fcmp olt float %14, %30
  br i1 %31, label %25, label %32, !llvm.loop !74

32:                                               ; preds = %25, %22, %20, %19
  %33 = phi ptr [ %0, %19 ], [ %0, %20 ], [ %13, %22 ], [ %27, %25 ]
  store float %14, ptr %33, align 4, !tbaa !40
  %34 = add nuw nsw i64 %11, 4
  %35 = icmp eq i64 %34, 64
  br i1 %35, label %36, label %10, !llvm.loop !75

36:                                               ; preds = %32
  %37 = getelementptr inbounds nuw i8, ptr %0, i64 64
  %38 = icmp eq ptr %37, %1
  br i1 %38, label %94, label %39

39:                                               ; preds = %36, %52
  %40 = phi ptr [ %54, %52 ], [ %37, %36 ]
  %41 = load float, ptr %40, align 4, !tbaa !40
  %42 = getelementptr inbounds i8, ptr %40, i64 -4
  %43 = load float, ptr %42, align 4, !tbaa !40
  %44 = fcmp olt float %41, %43
  br i1 %44, label %45, label %52

45:                                               ; preds = %39, %45
  %46 = phi float [ %50, %45 ], [ %43, %39 ]
  %47 = phi ptr [ %49, %45 ], [ %42, %39 ]
  %48 = phi ptr [ %47, %45 ], [ %40, %39 ]
  store float %46, ptr %48, align 4, !tbaa !40
  %49 = getelementptr inbounds i8, ptr %47, i64 -4
  %50 = load float, ptr %49, align 4, !tbaa !40
  %51 = fcmp olt float %41, %50
  br i1 %51, label %45, label %52, !llvm.loop !74

52:                                               ; preds = %45, %39
  %53 = phi ptr [ %40, %39 ], [ %47, %45 ]
  store float %41, ptr %53, align 4, !tbaa !40
  %54 = getelementptr inbounds nuw i8, ptr %40, i64 4
  %55 = icmp eq ptr %54, %1
  br i1 %55, label %94, label %39, !llvm.loop !76

56:                                               ; preds = %3
  %57 = icmp eq ptr %0, %1
  %58 = getelementptr inbounds nuw i8, ptr %0, i64 4
  %59 = icmp eq ptr %58, %1
  %60 = select i1 %57, i1 true, i1 %59
  br i1 %60, label %94, label %61

61:                                               ; preds = %56, %90
  %62 = phi ptr [ %92, %90 ], [ %58, %56 ]
  %63 = phi ptr [ %62, %90 ], [ %0, %56 ]
  %64 = load float, ptr %62, align 4, !tbaa !40
  %65 = load float, ptr %0, align 4, !tbaa !40
  %66 = fcmp olt float %64, %65
  br i1 %66, label %67, label %80

67:                                               ; preds = %61
  %68 = ptrtoint ptr %62 to i64
  %69 = sub i64 %68, %5
  %70 = ashr exact i64 %69, 2
  %71 = icmp sgt i64 %70, 1
  br i1 %71, label %72, label %76, !prof !34

72:                                               ; preds = %67
  %73 = getelementptr inbounds nuw i8, ptr %63, i64 8
  %74 = sub nsw i64 0, %70
  %75 = getelementptr inbounds float, ptr %73, i64 %74
  tail call void @llvm.memmove.p0.p0.i64(ptr noundef nonnull align 4 dereferenceable(1) %75, ptr noundef nonnull align 4 dereferenceable(1) %0, i64 %69, i1 false)
  br label %90

76:                                               ; preds = %67
  %77 = icmp eq i64 %69, 4
  br i1 %77, label %78, label %90

78:                                               ; preds = %76
  %79 = getelementptr inbounds nuw i8, ptr %63, i64 4
  store float %65, ptr %79, align 4, !tbaa !40
  br label %90

80:                                               ; preds = %61
  %81 = load float, ptr %63, align 4, !tbaa !40
  %82 = fcmp olt float %64, %81
  br i1 %82, label %83, label %90

83:                                               ; preds = %80, %83
  %84 = phi float [ %88, %83 ], [ %81, %80 ]
  %85 = phi ptr [ %87, %83 ], [ %63, %80 ]
  %86 = phi ptr [ %85, %83 ], [ %62, %80 ]
  store float %84, ptr %86, align 4, !tbaa !40
  %87 = getelementptr inbounds i8, ptr %85, i64 -4
  %88 = load float, ptr %87, align 4, !tbaa !40
  %89 = fcmp olt float %64, %88
  br i1 %89, label %83, label %90, !llvm.loop !74

90:                                               ; preds = %83, %80, %78, %76, %72
  %91 = phi ptr [ %0, %72 ], [ %0, %76 ], [ %0, %78 ], [ %62, %80 ], [ %85, %83 ]
  store float %64, ptr %91, align 4, !tbaa !40
  %92 = getelementptr inbounds nuw i8, ptr %62, i64 4
  %93 = icmp eq ptr %92, %1
  br i1 %93, label %94, label %61, !llvm.loop !75

94:                                               ; preds = %90, %52, %56, %36
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZSt11__make_heapIPfN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_RT0_(ptr noundef %0, ptr noundef %1, ptr noundef nonnull align 1 dereferenceable(1) %2) local_unnamed_addr #11 comdat {
  %4 = ptrtoint ptr %1 to i64
  %5 = ptrtoint ptr %0 to i64
  %6 = sub i64 %4, %5
  %7 = ashr exact i64 %6, 2
  %8 = icmp slt i64 %7, 2
  br i1 %8, label %103, label %9

9:                                                ; preds = %3
  %10 = add nsw i64 %7, -2
  %11 = lshr i64 %10, 1
  %12 = add nsw i64 %7, -1
  %13 = lshr i64 %12, 1
  %14 = and i64 %6, 4
  %15 = icmp eq i64 %14, 0
  %16 = lshr exact i64 %10, 1
  br i1 %15, label %17, label %21

17:                                               ; preds = %9
  %18 = or disjoint i64 %10, 1
  %19 = getelementptr inbounds nuw float, ptr %0, i64 %18
  %20 = getelementptr inbounds nuw float, ptr %0, i64 %16
  br label %59

21:                                               ; preds = %9, %54
  %22 = phi i64 [ %58, %54 ], [ %11, %9 ]
  %23 = getelementptr inbounds nuw float, ptr %0, i64 %22
  %24 = load float, ptr %23, align 4, !tbaa !40
  %25 = icmp slt i64 %22, %13
  br i1 %25, label %26, label %54

26:                                               ; preds = %21, %26
  %27 = phi i64 [ %37, %26 ], [ %22, %21 ]
  %28 = shl i64 %27, 1
  %29 = add i64 %28, 2
  %30 = getelementptr inbounds float, ptr %0, i64 %29
  %31 = getelementptr float, ptr %0, i64 %28
  %32 = getelementptr i8, ptr %31, i64 4
  %33 = load float, ptr %30, align 4, !tbaa !40
  %34 = load float, ptr %32, align 4, !tbaa !40
  %35 = fcmp olt float %33, %34
  %36 = or disjoint i64 %28, 1
  %37 = select i1 %35, i64 %36, i64 %29
  %38 = getelementptr inbounds float, ptr %0, i64 %37
  %39 = load float, ptr %38, align 4, !tbaa !40
  %40 = getelementptr inbounds float, ptr %0, i64 %27
  store float %39, ptr %40, align 4, !tbaa !40
  %41 = icmp slt i64 %37, %13
  br i1 %41, label %26, label %42, !llvm.loop !67

42:                                               ; preds = %26
  %43 = icmp sgt i64 %37, %22
  br i1 %43, label %44, label %54

44:                                               ; preds = %42, %51
  %45 = phi i64 [ %47, %51 ], [ %37, %42 ]
  %46 = add nsw i64 %45, -1
  %47 = sdiv i64 %46, 2
  %48 = getelementptr inbounds nuw float, ptr %0, i64 %47
  %49 = load float, ptr %48, align 4, !tbaa !40
  %50 = fcmp olt float %49, %24
  br i1 %50, label %51, label %54

51:                                               ; preds = %44
  %52 = getelementptr inbounds nuw float, ptr %0, i64 %45
  store float %49, ptr %52, align 4, !tbaa !40
  %53 = icmp sgt i64 %47, %22
  br i1 %53, label %44, label %54, !llvm.loop !68

54:                                               ; preds = %44, %51, %21, %42
  %55 = phi i64 [ %37, %42 ], [ %22, %21 ], [ %47, %51 ], [ %45, %44 ]
  %56 = getelementptr inbounds nuw float, ptr %0, i64 %55
  store float %24, ptr %56, align 4, !tbaa !40
  %57 = icmp eq i64 %22, 0
  %58 = add nsw i64 %22, -1
  br i1 %57, label %103, label %21, !llvm.loop !77

59:                                               ; preds = %17, %98
  %60 = phi i64 [ %102, %98 ], [ %11, %17 ]
  %61 = getelementptr inbounds nuw float, ptr %0, i64 %60
  %62 = load float, ptr %61, align 4, !tbaa !40
  %63 = icmp slt i64 %60, %13
  br i1 %63, label %64, label %80

64:                                               ; preds = %59, %64
  %65 = phi i64 [ %75, %64 ], [ %60, %59 ]
  %66 = shl i64 %65, 1
  %67 = add i64 %66, 2
  %68 = getelementptr inbounds float, ptr %0, i64 %67
  %69 = getelementptr float, ptr %0, i64 %66
  %70 = getelementptr i8, ptr %69, i64 4
  %71 = load float, ptr %68, align 4, !tbaa !40
  %72 = load float, ptr %70, align 4, !tbaa !40
  %73 = fcmp olt float %71, %72
  %74 = or disjoint i64 %66, 1
  %75 = select i1 %73, i64 %74, i64 %67
  %76 = getelementptr inbounds float, ptr %0, i64 %75
  %77 = load float, ptr %76, align 4, !tbaa !40
  %78 = getelementptr inbounds float, ptr %0, i64 %65
  store float %77, ptr %78, align 4, !tbaa !40
  %79 = icmp slt i64 %75, %13
  br i1 %79, label %64, label %80, !llvm.loop !67

80:                                               ; preds = %64, %59
  %81 = phi i64 [ %60, %59 ], [ %75, %64 ]
  %82 = icmp eq i64 %81, %16
  br i1 %82, label %83, label %85

83:                                               ; preds = %80
  %84 = load float, ptr %19, align 4, !tbaa !40
  store float %84, ptr %20, align 4, !tbaa !40
  br label %85

85:                                               ; preds = %83, %80
  %86 = phi i64 [ %18, %83 ], [ %81, %80 ]
  %87 = icmp sgt i64 %86, %60
  br i1 %87, label %88, label %98

88:                                               ; preds = %85, %95
  %89 = phi i64 [ %91, %95 ], [ %86, %85 ]
  %90 = add nsw i64 %89, -1
  %91 = sdiv i64 %90, 2
  %92 = getelementptr inbounds nuw float, ptr %0, i64 %91
  %93 = load float, ptr %92, align 4, !tbaa !40
  %94 = fcmp olt float %93, %62
  br i1 %94, label %95, label %98

95:                                               ; preds = %88
  %96 = getelementptr inbounds nuw float, ptr %0, i64 %89
  store float %93, ptr %96, align 4, !tbaa !40
  %97 = icmp sgt i64 %91, %60
  br i1 %97, label %88, label %98, !llvm.loop !68

98:                                               ; preds = %88, %95, %85
  %99 = phi i64 [ %86, %85 ], [ %91, %95 ], [ %89, %88 ]
  %100 = getelementptr inbounds nuw float, ptr %0, i64 %99
  store float %62, ptr %100, align 4, !tbaa !40
  %101 = icmp eq i64 %60, 0
  %102 = add nsw i64 %60, -1
  br i1 %101, label %103, label %59, !llvm.loop !77

103:                                              ; preds = %54, %98, %3
  ret void
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memmove.p0.p0.i64(ptr writeonly captures(none), ptr readonly captures(none), i64, i1 immarg) #2

; Function Attrs: mustprogress uwtable
declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEf(ptr noundef nonnull align 8 dereferenceable(8), float noundef) local_unnamed_addr #11

; Function Attrs: nofree noreturn nounwind
declare void @exit(i32 noundef) local_unnamed_addr #12

; Function Attrs: cold noreturn
declare void @_ZSt25__throw_bad_function_callv() local_unnamed_addr #13

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo9_M_insertIdEERSoT_(ptr noundef nonnull align 8 dereferenceable(8), double noundef) local_unnamed_addr #7

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #14

; Function Attrs: nobuiltin nounwind
declare void @_ZdaPv(ptr noundef) local_unnamed_addr #15

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef float @"_ZNSt17_Function_handlerIFfPfjEZ4mainE3$_0E9_M_invokeERKSt9_Any_dataOS0_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr nonnull readonly align 4 captures(none) %2) #16 {
  %4 = load ptr, ptr %1, align 8, !tbaa !52
  br label %5

5:                                                ; preds = %5, %3
  %6 = phi i64 [ 0, %3 ], [ %11, %5 ]
  %7 = phi float [ -2.000000e+00, %3 ], [ %10, %5 ]
  %8 = getelementptr inbounds nuw float, ptr %4, i64 %6
  %9 = load float, ptr %8, align 4, !tbaa !40
  %10 = tail call noundef float @llvm.maxnum.f32(float %7, float %9)
  %11 = add nuw nsw i64 %6, 1
  %12 = icmp eq i64 %11, 1024
  br i1 %12, label %13, label %5, !llvm.loop !78

13:                                               ; preds = %5
  ret float %10
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFfPfjEZ4mainE3$_0E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #17 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_0", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !43
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maxnum.f32(float, float) #9

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef float @"_ZNSt17_Function_handlerIFfPfjEZ4mainE3$_1E9_M_invokeERKSt9_Any_dataOS0_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr nonnull readonly align 4 captures(none) %2) #16 {
  %4 = load ptr, ptr %1, align 8, !tbaa !52
  br label %5

5:                                                ; preds = %5, %3
  %6 = phi i64 [ 0, %3 ], [ %17, %5 ]
  %7 = phi <4 x float> [ splat (float -2.000000e+00), %3 ], [ %15, %5 ]
  %8 = phi <4 x float> [ splat (float -2.000000e+00), %3 ], [ %16, %5 ]
  %9 = getelementptr inbounds nuw float, ptr %4, i64 %6
  %10 = getelementptr inbounds nuw i8, ptr %9, i64 16
  %11 = load <4 x float>, ptr %9, align 4, !tbaa !40
  %12 = freeze <4 x float> %11
  %13 = load <4 x float>, ptr %10, align 4, !tbaa !40
  %14 = freeze <4 x float> %13
  %15 = tail call <4 x float> @llvm.maxnum.v4f32(<4 x float> %7, <4 x float> %12)
  %16 = tail call <4 x float> @llvm.maxnum.v4f32(<4 x float> %8, <4 x float> %14)
  %17 = add nuw i64 %6, 8
  %18 = icmp eq i64 %17, 1024
  %19 = fcmp uno <4 x float> %12, %14
  %20 = bitcast <4 x i1> %19 to i4
  %21 = icmp ne i4 %20, 0
  %22 = or i1 %21, %18
  br i1 %22, label %23, label %5, !llvm.loop !81

23:                                               ; preds = %5
  %24 = insertelement <4 x i1> poison, i1 %21, i64 0
  %25 = shufflevector <4 x i1> %24, <4 x i1> poison, <4 x i32> zeroinitializer
  %26 = select <4 x i1> %25, <4 x float> %7, <4 x float> %15
  %27 = select <4 x i1> %25, <4 x float> %8, <4 x float> %16
  %28 = tail call <4 x float> @llvm.maxnum.v4f32(<4 x float> %26, <4 x float> %27)
  %29 = tail call float @llvm.vector.reduce.fmax.v4f32(<4 x float> %28)
  br i1 %21, label %30, label %38

30:                                               ; preds = %23, %30
  %31 = phi i64 [ %36, %30 ], [ %6, %23 ]
  %32 = phi float [ %35, %30 ], [ %29, %23 ]
  %33 = getelementptr inbounds nuw float, ptr %4, i64 %31
  %34 = load float, ptr %33, align 4, !tbaa !40
  %35 = tail call noundef float @llvm.maxnum.f32(float %32, float %34)
  %36 = add nuw nsw i64 %31, 1
  %37 = icmp eq i64 %36, 1024
  br i1 %37, label %38, label %30, !llvm.loop !82

38:                                               ; preds = %30, %23
  %39 = phi float [ %29, %23 ], [ %35, %30 ]
  ret float %39
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFfPfjEZ4mainE3$_1E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #17 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_1", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !43
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef float @"_ZNSt17_Function_handlerIFfPfjEZ4mainE3$_2E9_M_invokeERKSt9_Any_dataOS0_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr nonnull readonly align 4 captures(none) %2) #16 {
  %4 = load ptr, ptr %1, align 8, !tbaa !52
  br label %5

5:                                                ; preds = %5, %3
  %6 = phi i64 [ 0, %3 ], [ %11, %5 ]
  %7 = phi float [ 0x3810000000000000, %3 ], [ %10, %5 ]
  %8 = getelementptr inbounds nuw float, ptr %4, i64 %6
  %9 = load float, ptr %8, align 4, !tbaa !40
  %10 = tail call noundef float @llvm.maxnum.f32(float %7, float %9)
  %11 = add nuw nsw i64 %6, 1
  %12 = icmp eq i64 %11, 1024
  br i1 %12, label %13, label %5, !llvm.loop !83

13:                                               ; preds = %5
  ret float %10
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFfPfjEZ4mainE3$_2E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #17 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_2", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !43
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef float @"_ZNSt17_Function_handlerIFfPfjEZ4mainE3$_3E9_M_invokeERKSt9_Any_dataOS0_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr nonnull readonly align 4 captures(none) %2) #16 {
  %4 = load ptr, ptr %1, align 8, !tbaa !52
  br label %5

5:                                                ; preds = %5, %3
  %6 = phi i64 [ 0, %3 ], [ %17, %5 ]
  %7 = phi <4 x float> [ splat (float 0x3810000000000000), %3 ], [ %15, %5 ]
  %8 = phi <4 x float> [ splat (float 0x3810000000000000), %3 ], [ %16, %5 ]
  %9 = getelementptr inbounds nuw float, ptr %4, i64 %6
  %10 = getelementptr inbounds nuw i8, ptr %9, i64 16
  %11 = load <4 x float>, ptr %9, align 4, !tbaa !40
  %12 = freeze <4 x float> %11
  %13 = load <4 x float>, ptr %10, align 4, !tbaa !40
  %14 = freeze <4 x float> %13
  %15 = tail call <4 x float> @llvm.maxnum.v4f32(<4 x float> %7, <4 x float> %12)
  %16 = tail call <4 x float> @llvm.maxnum.v4f32(<4 x float> %8, <4 x float> %14)
  %17 = add nuw i64 %6, 8
  %18 = icmp eq i64 %17, 1024
  %19 = fcmp uno <4 x float> %12, %14
  %20 = bitcast <4 x i1> %19 to i4
  %21 = icmp ne i4 %20, 0
  %22 = or i1 %21, %18
  br i1 %22, label %23, label %5, !llvm.loop !84

23:                                               ; preds = %5
  %24 = insertelement <4 x i1> poison, i1 %21, i64 0
  %25 = shufflevector <4 x i1> %24, <4 x i1> poison, <4 x i32> zeroinitializer
  %26 = select <4 x i1> %25, <4 x float> %7, <4 x float> %15
  %27 = select <4 x i1> %25, <4 x float> %8, <4 x float> %16
  %28 = tail call <4 x float> @llvm.maxnum.v4f32(<4 x float> %26, <4 x float> %27)
  %29 = tail call float @llvm.vector.reduce.fmax.v4f32(<4 x float> %28)
  br i1 %21, label %30, label %38

30:                                               ; preds = %23, %30
  %31 = phi i64 [ %36, %30 ], [ %6, %23 ]
  %32 = phi float [ %35, %30 ], [ %29, %23 ]
  %33 = getelementptr inbounds nuw float, ptr %4, i64 %31
  %34 = load float, ptr %33, align 4, !tbaa !40
  %35 = tail call noundef float @llvm.maxnum.f32(float %32, float %34)
  %36 = add nuw nsw i64 %31, 1
  %37 = icmp eq i64 %36, 1024
  br i1 %37, label %38, label %30, !llvm.loop !85

38:                                               ; preds = %30, %23
  %39 = phi float [ %29, %23 ], [ %35, %30 ]
  ret float %39
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFfPfjEZ4mainE3$_3E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #17 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_3", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !43
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef float @"_ZNSt17_Function_handlerIFfPfjEZ4mainE3$_4E9_M_invokeERKSt9_Any_dataOS0_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr nonnull readonly align 4 captures(none) %2) #16 {
  %4 = load ptr, ptr %1, align 8, !tbaa !52
  br label %5

5:                                                ; preds = %5, %3
  %6 = phi i64 [ 0, %3 ], [ %11, %5 ]
  %7 = phi float [ 0x36A0000000000000, %3 ], [ %10, %5 ]
  %8 = getelementptr inbounds nuw float, ptr %4, i64 %6
  %9 = load float, ptr %8, align 4, !tbaa !40
  %10 = tail call noundef float @llvm.maxnum.f32(float %7, float %9)
  %11 = add nuw nsw i64 %6, 1
  %12 = icmp eq i64 %11, 1024
  br i1 %12, label %13, label %5, !llvm.loop !86

13:                                               ; preds = %5
  ret float %10
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFfPfjEZ4mainE3$_4E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #17 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_4", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !43
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef float @"_ZNSt17_Function_handlerIFfPfjEZ4mainE3$_5E9_M_invokeERKSt9_Any_dataOS0_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr nonnull readonly align 4 captures(none) %2) #16 {
  %4 = load ptr, ptr %1, align 8, !tbaa !52
  br label %5

5:                                                ; preds = %5, %3
  %6 = phi i64 [ 0, %3 ], [ %17, %5 ]
  %7 = phi <4 x float> [ splat (float 0x36A0000000000000), %3 ], [ %15, %5 ]
  %8 = phi <4 x float> [ splat (float 0x36A0000000000000), %3 ], [ %16, %5 ]
  %9 = getelementptr inbounds nuw float, ptr %4, i64 %6
  %10 = getelementptr inbounds nuw i8, ptr %9, i64 16
  %11 = load <4 x float>, ptr %9, align 4, !tbaa !40
  %12 = freeze <4 x float> %11
  %13 = load <4 x float>, ptr %10, align 4, !tbaa !40
  %14 = freeze <4 x float> %13
  %15 = tail call <4 x float> @llvm.maxnum.v4f32(<4 x float> %7, <4 x float> %12)
  %16 = tail call <4 x float> @llvm.maxnum.v4f32(<4 x float> %8, <4 x float> %14)
  %17 = add nuw i64 %6, 8
  %18 = icmp eq i64 %17, 1024
  %19 = fcmp uno <4 x float> %12, %14
  %20 = bitcast <4 x i1> %19 to i4
  %21 = icmp ne i4 %20, 0
  %22 = or i1 %21, %18
  br i1 %22, label %23, label %5, !llvm.loop !87

23:                                               ; preds = %5
  %24 = insertelement <4 x i1> poison, i1 %21, i64 0
  %25 = shufflevector <4 x i1> %24, <4 x i1> poison, <4 x i32> zeroinitializer
  %26 = select <4 x i1> %25, <4 x float> %7, <4 x float> %15
  %27 = select <4 x i1> %25, <4 x float> %8, <4 x float> %16
  %28 = tail call <4 x float> @llvm.maxnum.v4f32(<4 x float> %26, <4 x float> %27)
  %29 = tail call float @llvm.vector.reduce.fmax.v4f32(<4 x float> %28)
  br i1 %21, label %30, label %38

30:                                               ; preds = %23, %30
  %31 = phi i64 [ %36, %30 ], [ %6, %23 ]
  %32 = phi float [ %35, %30 ], [ %29, %23 ]
  %33 = getelementptr inbounds nuw float, ptr %4, i64 %31
  %34 = load float, ptr %33, align 4, !tbaa !40
  %35 = tail call noundef float @llvm.maxnum.f32(float %32, float %34)
  %36 = add nuw nsw i64 %31, 1
  %37 = icmp eq i64 %36, 1024
  br i1 %37, label %38, label %30, !llvm.loop !88

38:                                               ; preds = %30, %23
  %39 = phi float [ %29, %23 ], [ %35, %30 ]
  ret float %39
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFfPfjEZ4mainE3$_5E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #17 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_5", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !43
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef float @"_ZNSt17_Function_handlerIFfPfjEZ4mainE3$_6E9_M_invokeERKSt9_Any_dataOS0_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr nonnull readonly align 4 captures(none) %2) #16 {
  %4 = load ptr, ptr %1, align 8, !tbaa !52
  br label %5

5:                                                ; preds = %5, %3
  %6 = phi i64 [ 0, %3 ], [ %11, %5 ]
  %7 = phi float [ 0x7FF8000000000000, %3 ], [ %10, %5 ]
  %8 = getelementptr inbounds nuw float, ptr %4, i64 %6
  %9 = load float, ptr %8, align 4, !tbaa !40
  %10 = tail call noundef float @llvm.maxnum.f32(float %7, float %9)
  %11 = add nuw nsw i64 %6, 1
  %12 = icmp eq i64 %11, 1024
  br i1 %12, label %13, label %5, !llvm.loop !89

13:                                               ; preds = %5
  ret float %10
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFfPfjEZ4mainE3$_6E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #17 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_6", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !43
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef float @"_ZNSt17_Function_handlerIFfPfjEZ4mainE3$_7E9_M_invokeERKSt9_Any_dataOS0_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr nonnull readonly align 4 captures(none) %2) #16 {
  %4 = load ptr, ptr %1, align 8, !tbaa !52
  br label %5

5:                                                ; preds = %5, %3
  %6 = phi i64 [ 0, %3 ], [ %17, %5 ]
  %7 = phi <4 x float> [ splat (float 0x7FF8000000000000), %3 ], [ %15, %5 ]
  %8 = phi <4 x float> [ splat (float 0x7FF8000000000000), %3 ], [ %16, %5 ]
  %9 = getelementptr inbounds nuw float, ptr %4, i64 %6
  %10 = getelementptr inbounds nuw i8, ptr %9, i64 16
  %11 = load <4 x float>, ptr %9, align 4, !tbaa !40
  %12 = freeze <4 x float> %11
  %13 = load <4 x float>, ptr %10, align 4, !tbaa !40
  %14 = freeze <4 x float> %13
  %15 = tail call <4 x float> @llvm.maxnum.v4f32(<4 x float> %7, <4 x float> %12)
  %16 = tail call <4 x float> @llvm.maxnum.v4f32(<4 x float> %8, <4 x float> %14)
  %17 = add nuw i64 %6, 8
  %18 = icmp eq i64 %17, 1024
  %19 = fcmp uno <4 x float> %12, %14
  %20 = bitcast <4 x i1> %19 to i4
  %21 = icmp ne i4 %20, 0
  %22 = or i1 %21, %18
  br i1 %22, label %23, label %5, !llvm.loop !90

23:                                               ; preds = %5
  %24 = insertelement <4 x i1> poison, i1 %21, i64 0
  %25 = shufflevector <4 x i1> %24, <4 x i1> poison, <4 x i32> zeroinitializer
  %26 = select <4 x i1> %25, <4 x float> %7, <4 x float> %15
  %27 = select <4 x i1> %25, <4 x float> %8, <4 x float> %16
  %28 = tail call <4 x float> @llvm.maxnum.v4f32(<4 x float> %26, <4 x float> %27)
  %29 = tail call float @llvm.vector.reduce.fmax.v4f32(<4 x float> %28)
  br i1 %21, label %30, label %38

30:                                               ; preds = %23, %30
  %31 = phi i64 [ %36, %30 ], [ %6, %23 ]
  %32 = phi float [ %35, %30 ], [ %29, %23 ]
  %33 = getelementptr inbounds nuw float, ptr %4, i64 %31
  %34 = load float, ptr %33, align 4, !tbaa !40
  %35 = tail call noundef float @llvm.maxnum.f32(float %32, float %34)
  %36 = add nuw nsw i64 %31, 1
  %37 = icmp eq i64 %36, 1024
  br i1 %37, label %38, label %30, !llvm.loop !91

38:                                               ; preds = %30, %23
  %39 = phi float [ %29, %23 ], [ %35, %30 ]
  ret float %39
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFfPfjEZ4mainE3$_7E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #17 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_7", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !43
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef float @"_ZNSt17_Function_handlerIFfPfjEZ4mainE3$_8E9_M_invokeERKSt9_Any_dataOS0_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr nonnull readonly align 4 captures(none) %2) #16 {
  %4 = load ptr, ptr %1, align 8, !tbaa !52
  br label %5

5:                                                ; preds = %5, %3
  %6 = phi i64 [ 0, %3 ], [ %12, %5 ]
  %7 = phi float [ -2.000000e+00, %3 ], [ %11, %5 ]
  %8 = getelementptr inbounds nuw float, ptr %4, i64 %6
  %9 = load float, ptr %8, align 4, !tbaa !40
  %10 = fcmp ogt float %9, %7
  %11 = select i1 %10, float %9, float %7
  %12 = add nuw nsw i64 %6, 1
  %13 = icmp eq i64 %12, 1024
  br i1 %13, label %14, label %5, !llvm.loop !92

14:                                               ; preds = %5
  ret float %11
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFfPfjEZ4mainE3$_8E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #17 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_8", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !43
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef float @"_ZNSt17_Function_handlerIFfPfjEZ4mainE3$_9E9_M_invokeERKSt9_Any_dataOS0_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr nonnull readonly align 4 captures(none) %2) #16 {
  %4 = load ptr, ptr %1, align 8, !tbaa !52
  br label %5

5:                                                ; preds = %5, %3
  %6 = phi i64 [ 0, %3 ], [ %12, %5 ]
  %7 = phi float [ -2.000000e+00, %3 ], [ %11, %5 ]
  %8 = getelementptr inbounds nuw float, ptr %4, i64 %6
  %9 = load float, ptr %8, align 4, !tbaa !40
  %10 = fcmp ogt float %9, %7
  %11 = select i1 %10, float %9, float %7
  %12 = add nuw nsw i64 %6, 1
  %13 = icmp eq i64 %12, 1024
  br i1 %13, label %14, label %5, !llvm.loop !93

14:                                               ; preds = %5
  ret float %11
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFfPfjEZ4mainE3$_9E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #17 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_9", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !43
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef float @"_ZNSt17_Function_handlerIFfPfjEZ4mainE4$_10E9_M_invokeERKSt9_Any_dataOS0_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr nonnull readonly align 4 captures(none) %2) #16 {
  %4 = load ptr, ptr %1, align 8, !tbaa !52
  br label %5

5:                                                ; preds = %5, %3
  %6 = phi i64 [ 0, %3 ], [ %12, %5 ]
  %7 = phi float [ 0x3810000000000000, %3 ], [ %11, %5 ]
  %8 = getelementptr inbounds nuw float, ptr %4, i64 %6
  %9 = load float, ptr %8, align 4, !tbaa !40
  %10 = fcmp ogt float %9, %7
  %11 = select i1 %10, float %9, float %7
  %12 = add nuw nsw i64 %6, 1
  %13 = icmp eq i64 %12, 1024
  br i1 %13, label %14, label %5, !llvm.loop !95

14:                                               ; preds = %5
  ret float %11
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFfPfjEZ4mainE4$_10E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #17 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_10", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !43
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef float @"_ZNSt17_Function_handlerIFfPfjEZ4mainE4$_11E9_M_invokeERKSt9_Any_dataOS0_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr nonnull readonly align 4 captures(none) %2) #16 {
  %4 = load ptr, ptr %1, align 8, !tbaa !52
  br label %5

5:                                                ; preds = %5, %3
  %6 = phi i64 [ 0, %3 ], [ %12, %5 ]
  %7 = phi float [ 0x3810000000000000, %3 ], [ %11, %5 ]
  %8 = getelementptr inbounds nuw float, ptr %4, i64 %6
  %9 = load float, ptr %8, align 4, !tbaa !40
  %10 = fcmp ogt float %9, %7
  %11 = select i1 %10, float %9, float %7
  %12 = add nuw nsw i64 %6, 1
  %13 = icmp eq i64 %12, 1024
  br i1 %13, label %14, label %5, !llvm.loop !96

14:                                               ; preds = %5
  ret float %11
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFfPfjEZ4mainE4$_11E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #17 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_11", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !43
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef float @"_ZNSt17_Function_handlerIFfPfjEZ4mainE4$_12E9_M_invokeERKSt9_Any_dataOS0_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr nonnull readonly align 4 captures(none) %2) #16 {
  %4 = load ptr, ptr %1, align 8, !tbaa !52
  br label %5

5:                                                ; preds = %5, %3
  %6 = phi i64 [ 0, %3 ], [ %12, %5 ]
  %7 = phi float [ 0x36A0000000000000, %3 ], [ %11, %5 ]
  %8 = getelementptr inbounds nuw float, ptr %4, i64 %6
  %9 = load float, ptr %8, align 4, !tbaa !40
  %10 = fcmp ogt float %9, %7
  %11 = select i1 %10, float %9, float %7
  %12 = add nuw nsw i64 %6, 1
  %13 = icmp eq i64 %12, 1025
  br i1 %13, label %14, label %5, !llvm.loop !97

14:                                               ; preds = %5
  ret float %11
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFfPfjEZ4mainE4$_12E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #17 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_12", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !43
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef float @"_ZNSt17_Function_handlerIFfPfjEZ4mainE4$_13E9_M_invokeERKSt9_Any_dataOS0_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr nonnull readonly align 4 captures(none) %2) #16 {
  %4 = load ptr, ptr %1, align 8, !tbaa !52
  br label %5

5:                                                ; preds = %5, %3
  %6 = phi i64 [ 0, %3 ], [ %12, %5 ]
  %7 = phi float [ 0x36A0000000000000, %3 ], [ %11, %5 ]
  %8 = getelementptr inbounds nuw float, ptr %4, i64 %6
  %9 = load float, ptr %8, align 4, !tbaa !40
  %10 = fcmp ogt float %9, %7
  %11 = select i1 %10, float %9, float %7
  %12 = add nuw nsw i64 %6, 1
  %13 = icmp eq i64 %12, 1025
  br i1 %13, label %14, label %5, !llvm.loop !98

14:                                               ; preds = %5
  ret float %11
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFfPfjEZ4mainE4$_13E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #17 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_13", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !43
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef float @"_ZNSt17_Function_handlerIFfPfjEZ4mainE4$_14E9_M_invokeERKSt9_Any_dataOS0_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr nonnull readonly align 4 captures(none) %2) #16 {
  %4 = load ptr, ptr %1, align 8, !tbaa !52
  br label %5

5:                                                ; preds = %5, %3
  %6 = phi i64 [ 0, %3 ], [ %12, %5 ]
  %7 = phi float [ 0x7FF8000000000000, %3 ], [ %11, %5 ]
  %8 = getelementptr inbounds nuw float, ptr %4, i64 %6
  %9 = load float, ptr %8, align 4, !tbaa !40
  %10 = fcmp ogt float %9, %7
  %11 = select i1 %10, float %9, float %7
  %12 = add nuw nsw i64 %6, 1
  %13 = icmp eq i64 %12, 1025
  br i1 %13, label %14, label %5, !llvm.loop !99

14:                                               ; preds = %5
  ret float %11
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFfPfjEZ4mainE4$_14E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #17 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_14", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !43
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef float @"_ZNSt17_Function_handlerIFfPfjEZ4mainE4$_15E9_M_invokeERKSt9_Any_dataOS0_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr nonnull readonly align 4 captures(none) %2) #16 {
  %4 = load ptr, ptr %1, align 8, !tbaa !52
  br label %5

5:                                                ; preds = %5, %3
  %6 = phi i64 [ 0, %3 ], [ %12, %5 ]
  %7 = phi float [ 0x7FF8000000000000, %3 ], [ %11, %5 ]
  %8 = getelementptr inbounds nuw float, ptr %4, i64 %6
  %9 = load float, ptr %8, align 4, !tbaa !40
  %10 = fcmp ogt float %9, %7
  %11 = select i1 %10, float %9, float %7
  %12 = add nuw nsw i64 %6, 1
  %13 = icmp eq i64 %12, 1025
  br i1 %13, label %14, label %5, !llvm.loop !100

14:                                               ; preds = %5
  ret float %11
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFfPfjEZ4mainE4$_15E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #17 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_15", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !43
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef float @"_ZNSt17_Function_handlerIFfPfjEZ4mainE4$_16E9_M_invokeERKSt9_Any_dataOS0_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr nonnull readonly align 4 captures(none) %2) #16 {
  %4 = load ptr, ptr %1, align 8, !tbaa !52
  br label %5

5:                                                ; preds = %5, %3
  %6 = phi i64 [ 0, %3 ], [ %12, %5 ]
  %7 = phi float [ -2.000000e+00, %3 ], [ %11, %5 ]
  %8 = getelementptr inbounds nuw float, ptr %4, i64 %6
  %9 = load float, ptr %8, align 4, !tbaa !40
  %10 = fcmp oge float %7, %9
  %11 = select i1 %10, float %7, float %9
  %12 = add nuw nsw i64 %6, 1
  %13 = icmp eq i64 %12, 1024
  br i1 %13, label %14, label %5, !llvm.loop !101

14:                                               ; preds = %5
  ret float %11
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFfPfjEZ4mainE4$_16E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #17 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_16", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !43
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef float @"_ZNSt17_Function_handlerIFfPfjEZ4mainE4$_17E9_M_invokeERKSt9_Any_dataOS0_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr nonnull readonly align 4 captures(none) %2) #16 {
  %4 = load ptr, ptr %1, align 8, !tbaa !52
  br label %5

5:                                                ; preds = %5, %3
  %6 = phi i64 [ 0, %3 ], [ %12, %5 ]
  %7 = phi float [ -2.000000e+00, %3 ], [ %11, %5 ]
  %8 = getelementptr inbounds nuw float, ptr %4, i64 %6
  %9 = load float, ptr %8, align 4, !tbaa !40
  %10 = fcmp oge float %7, %9
  %11 = select i1 %10, float %7, float %9
  %12 = add nuw nsw i64 %6, 1
  %13 = icmp eq i64 %12, 1024
  br i1 %13, label %14, label %5, !llvm.loop !102

14:                                               ; preds = %5
  ret float %11
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFfPfjEZ4mainE4$_17E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #17 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_17", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !43
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef float @"_ZNSt17_Function_handlerIFfPfjEZ4mainE4$_18E9_M_invokeERKSt9_Any_dataOS0_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr nonnull readonly align 4 captures(none) %2) #16 {
  %4 = load ptr, ptr %1, align 8, !tbaa !52
  br label %5

5:                                                ; preds = %5, %3
  %6 = phi i64 [ 0, %3 ], [ %12, %5 ]
  %7 = phi float [ -2.000000e+00, %3 ], [ %11, %5 ]
  %8 = getelementptr inbounds nuw float, ptr %4, i64 %6
  %9 = load float, ptr %8, align 4, !tbaa !40
  %10 = fcmp ogt float %7, %9
  %11 = select i1 %10, float %7, float %9
  %12 = add nuw nsw i64 %6, 1
  %13 = icmp eq i64 %12, 1024
  br i1 %13, label %14, label %5, !llvm.loop !103

14:                                               ; preds = %5
  ret float %11
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFfPfjEZ4mainE4$_18E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #17 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_18", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !43
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef float @"_ZNSt17_Function_handlerIFfPfjEZ4mainE4$_19E9_M_invokeERKSt9_Any_dataOS0_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr nonnull readonly align 4 captures(none) %2) #16 {
  %4 = load ptr, ptr %1, align 8, !tbaa !52
  br label %5

5:                                                ; preds = %5, %3
  %6 = phi i64 [ 0, %3 ], [ %12, %5 ]
  %7 = phi float [ -2.000000e+00, %3 ], [ %11, %5 ]
  %8 = getelementptr inbounds nuw float, ptr %4, i64 %6
  %9 = load float, ptr %8, align 4, !tbaa !40
  %10 = fcmp ogt float %7, %9
  %11 = select i1 %10, float %7, float %9
  %12 = add nuw nsw i64 %6, 1
  %13 = icmp eq i64 %12, 1024
  br i1 %13, label %14, label %5, !llvm.loop !104

14:                                               ; preds = %5
  ret float %11
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFfPfjEZ4mainE4$_19E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #17 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_19", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !43
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef float @"_ZNSt17_Function_handlerIFfPfjEZ4mainE4$_20E9_M_invokeERKSt9_Any_dataOS0_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr nonnull readonly align 4 captures(none) %2) #16 {
  %4 = load ptr, ptr %1, align 8, !tbaa !52
  br label %5

5:                                                ; preds = %5, %3
  %6 = phi i64 [ 0, %3 ], [ %12, %5 ]
  %7 = phi float [ -2.000000e+00, %3 ], [ %11, %5 ]
  %8 = getelementptr inbounds nuw float, ptr %4, i64 %6
  %9 = load float, ptr %8, align 4, !tbaa !40
  %10 = fcmp olt float %7, %9
  %11 = select i1 %10, float %9, float %7
  %12 = add nuw nsw i64 %6, 1
  %13 = icmp eq i64 %12, 1024
  br i1 %13, label %14, label %5, !llvm.loop !105

14:                                               ; preds = %5
  ret float %11
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFfPfjEZ4mainE4$_20E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #17 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_20", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !43
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef float @"_ZNSt17_Function_handlerIFfPfjEZ4mainE4$_21E9_M_invokeERKSt9_Any_dataOS0_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr nonnull readonly align 4 captures(none) %2) #16 {
  %4 = load ptr, ptr %1, align 8, !tbaa !52
  br label %5

5:                                                ; preds = %5, %3
  %6 = phi i64 [ 0, %3 ], [ %12, %5 ]
  %7 = phi float [ -2.000000e+00, %3 ], [ %11, %5 ]
  %8 = getelementptr inbounds nuw float, ptr %4, i64 %6
  %9 = load float, ptr %8, align 4, !tbaa !40
  %10 = fcmp olt float %7, %9
  %11 = select i1 %10, float %9, float %7
  %12 = add nuw nsw i64 %6, 1
  %13 = icmp eq i64 %12, 1024
  br i1 %13, label %14, label %5, !llvm.loop !106

14:                                               ; preds = %5
  ret float %11
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFfPfjEZ4mainE4$_21E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #17 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_21", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !43
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef float @"_ZNSt17_Function_handlerIFfPfjEZ4mainE4$_22E9_M_invokeERKSt9_Any_dataOS0_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr nonnull readonly align 4 captures(none) %2) #16 {
  %4 = load ptr, ptr %1, align 8, !tbaa !52
  br label %5

5:                                                ; preds = %5, %3
  %6 = phi i64 [ 0, %3 ], [ %12, %5 ]
  %7 = phi float [ 0x36A0000000000000, %3 ], [ %11, %5 ]
  %8 = getelementptr inbounds nuw float, ptr %4, i64 %6
  %9 = load float, ptr %8, align 4, !tbaa !40
  %10 = fcmp olt float %7, %9
  %11 = select i1 %10, float %9, float %7
  %12 = add nuw nsw i64 %6, 1
  %13 = icmp eq i64 %12, 1024
  br i1 %13, label %14, label %5, !llvm.loop !107

14:                                               ; preds = %5
  ret float %11
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFfPfjEZ4mainE4$_22E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #17 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_22", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !43
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef float @"_ZNSt17_Function_handlerIFfPfjEZ4mainE4$_23E9_M_invokeERKSt9_Any_dataOS0_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr nonnull readonly align 4 captures(none) %2) #16 {
  %4 = load ptr, ptr %1, align 8, !tbaa !52
  br label %5

5:                                                ; preds = %5, %3
  %6 = phi i64 [ 0, %3 ], [ %12, %5 ]
  %7 = phi float [ 0x36A0000000000000, %3 ], [ %11, %5 ]
  %8 = getelementptr inbounds nuw float, ptr %4, i64 %6
  %9 = load float, ptr %8, align 4, !tbaa !40
  %10 = fcmp olt float %7, %9
  %11 = select i1 %10, float %9, float %7
  %12 = add nuw nsw i64 %6, 1
  %13 = icmp eq i64 %12, 1024
  br i1 %13, label %14, label %5, !llvm.loop !108

14:                                               ; preds = %5
  ret float %11
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFfPfjEZ4mainE4$_23E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #17 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_23", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !43
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef float @"_ZNSt17_Function_handlerIFfPfjEZ4mainE4$_24E9_M_invokeERKSt9_Any_dataOS0_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr nonnull readonly align 4 captures(none) %2) #16 {
  %4 = load ptr, ptr %1, align 8, !tbaa !52
  br label %5

5:                                                ; preds = %5, %3
  %6 = phi i64 [ 0, %3 ], [ %12, %5 ]
  %7 = phi float [ 0x7FF8000000000000, %3 ], [ %11, %5 ]
  %8 = getelementptr inbounds nuw float, ptr %4, i64 %6
  %9 = load float, ptr %8, align 4, !tbaa !40
  %10 = fcmp olt float %7, %9
  %11 = select i1 %10, float %9, float %7
  %12 = add nuw nsw i64 %6, 1
  %13 = icmp eq i64 %12, 1024
  br i1 %13, label %14, label %5, !llvm.loop !109

14:                                               ; preds = %5
  ret float %11
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFfPfjEZ4mainE4$_24E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #17 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_24", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !43
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef float @"_ZNSt17_Function_handlerIFfPfjEZ4mainE4$_25E9_M_invokeERKSt9_Any_dataOS0_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr nonnull readonly align 4 captures(none) %2) #16 {
  %4 = load ptr, ptr %1, align 8, !tbaa !52
  br label %5

5:                                                ; preds = %5, %3
  %6 = phi i64 [ 0, %3 ], [ %12, %5 ]
  %7 = phi float [ 0x7FF8000000000000, %3 ], [ %11, %5 ]
  %8 = getelementptr inbounds nuw float, ptr %4, i64 %6
  %9 = load float, ptr %8, align 4, !tbaa !40
  %10 = fcmp olt float %7, %9
  %11 = select i1 %10, float %9, float %7
  %12 = add nuw nsw i64 %6, 1
  %13 = icmp eq i64 %12, 1024
  br i1 %13, label %14, label %5, !llvm.loop !110

14:                                               ; preds = %5
  ret float %11
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFfPfjEZ4mainE4$_25E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #17 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_25", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !43
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define internal void @_GLOBAL__sub_I_fmax_reduction.cpp() #18 section ".text.startup" {
  store i64 5489, ptr @_ZL3rng, align 8, !tbaa !6
  br label %1

1:                                                ; preds = %1, %0
  %2 = phi i64 [ 5489, %0 ], [ %9, %1 ]
  %3 = phi i64 [ 1, %0 ], [ %10, %1 ]
  %4 = getelementptr i64, ptr @_ZL3rng, i64 %3
  %5 = lshr i64 %2, 30
  %6 = xor i64 %5, %2
  %7 = mul nuw nsw i64 %6, 1812433253
  %8 = add nuw i64 %7, %3
  %9 = and i64 %8, 4294967295
  store i64 %9, ptr %4, align 8, !tbaa !6
  %10 = add nuw nsw i64 %3, 1
  %11 = icmp eq i64 %10, 624
  br i1 %11, label %12, label %1, !llvm.loop !10

12:                                               ; preds = %1
  store i64 624, ptr getelementptr inbounds nuw (i8, ptr @_ZL3rng, i64 4992), align 8, !tbaa !12
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare fp128 @llvm.log.f128(fp128) #19

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.umax.i64(i64, i64) #19

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <4 x float> @llvm.maxnum.v4f32(<4 x float>, <4 x float>) #19

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.vector.reduce.fmax.v4f32(<4 x float>) #19

attributes #0 = { mustprogress norecurse uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { noinline noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { cold nofree noreturn }
attributes #5 = { inlinehint mustprogress uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { nobuiltin allocsize(0) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #8 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: read) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #9 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #10 = { nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #11 = { mustprogress uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #12 = { nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #13 = { cold noreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #14 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #15 = { nobuiltin nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #16 = { mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #17 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #18 = { nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #19 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #20 = { nounwind }
attributes #21 = { noreturn nounwind }
attributes #22 = { builtin allocsize(0) }
attributes #23 = { cold noreturn }
attributes #24 = { cold noreturn nounwind }
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
!7 = !{!"long", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C++ TBAA"}
!10 = distinct !{!10, !11}
!11 = !{!"llvm.loop.mustprogress"}
!12 = !{!13, !7, i64 4992}
!13 = !{!"_ZTSSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EE", !8, i64 0, !7, i64 4992}
!14 = !{i64 0, i64 4992, !15, i64 4992, i64 8, !6}
!15 = !{!8, !8, i64 0}
!16 = !{!17, !19, i64 24}
!17 = !{!"_ZTSSt8functionIFfPfjEE", !18, i64 0, !19, i64 24}
!18 = !{!"_ZTSSt14_Function_base", !8, i64 0, !19, i64 16}
!19 = !{!"any pointer", !8, i64 0}
!20 = !{!18, !19, i64 16}
!21 = !{!22, !22, i64 0}
!22 = !{!"vtable pointer", !9, i64 0}
!23 = !{!24, !26, i64 32}
!24 = !{!"_ZTSSt8ios_base", !7, i64 8, !7, i64 16, !25, i64 24, !26, i64 28, !26, i64 32, !27, i64 40, !28, i64 48, !8, i64 64, !29, i64 192, !30, i64 200, !31, i64 208}
!25 = !{!"_ZTSSt13_Ios_Fmtflags", !8, i64 0}
!26 = !{!"_ZTSSt12_Ios_Iostate", !8, i64 0}
!27 = !{!"p1 _ZTSNSt8ios_base14_Callback_listE", !19, i64 0}
!28 = !{!"_ZTSNSt8ios_base6_WordsE", !19, i64 0, !7, i64 8}
!29 = !{!"int", !8, i64 0}
!30 = !{!"p1 _ZTSNSt8ios_base6_WordsE", !19, i64 0}
!31 = !{!"_ZTSSt6locale", !32, i64 0}
!32 = !{!"p1 _ZTSNSt6locale5_ImplE", !19, i64 0}
!33 = !{!29, !29, i64 0}
!34 = !{!"branch_weights", !"expected", i32 2000, i32 1}
!35 = distinct !{!35, !11, !36, !37}
!36 = !{!"llvm.loop.isvectorized", i32 1}
!37 = !{!"llvm.loop.unroll.runtime.disable"}
!38 = distinct !{!38, !11, !36, !37}
!39 = distinct !{!39, !11}
!40 = !{!41, !41, i64 0}
!41 = !{!"float", !8, i64 0}
!42 = distinct !{!42, !11}
!43 = !{!19, !19, i64 0}
!44 = distinct !{!44, !11, !36, !37}
!45 = distinct !{!45, !11, !36, !37}
!46 = distinct !{!46, !11, !36, !37}
!47 = distinct !{!47, !11, !36, !37}
!48 = distinct !{!48, !11, !36, !37}
!49 = distinct !{!49, !11}
!50 = distinct !{!50, !11, !36, !37}
!51 = distinct !{!51, !11}
!52 = !{!53, !53, i64 0}
!53 = !{!"p1 float", !19, i64 0}
!54 = distinct !{!54, !11}
!55 = distinct !{!55, !11, !36, !37}
!56 = distinct !{!56, !11}
!57 = distinct !{!57, !11}
!58 = distinct !{!58, !11, !36, !37}
!59 = distinct !{!59, !11}
!60 = distinct !{!60, !11}
!61 = distinct !{!61, !11, !36, !37}
!62 = distinct !{!62, !11}
!63 = distinct !{!63, !11}
!64 = distinct !{!64, !11, !36, !37}
!65 = distinct !{!65, !11}
!66 = distinct !{!66, !11}
!67 = distinct !{!67, !11}
!68 = distinct !{!68, !11}
!69 = distinct !{!69, !11}
!70 = distinct !{!70, !11}
!71 = distinct !{!71, !11}
!72 = distinct !{!72, !11}
!73 = distinct !{!73, !11}
!74 = distinct !{!74, !11}
!75 = distinct !{!75, !11}
!76 = distinct !{!76, !11}
!77 = distinct !{!77, !11}
!78 = distinct !{!78, !11, !79, !80}
!79 = !{!"llvm.loop.vectorize.width", i32 1}
!80 = !{!"llvm.loop.interleave.count", i32 1}
!81 = distinct !{!81, !11, !36, !37}
!82 = distinct !{!82, !11, !37, !36}
!83 = distinct !{!83, !11, !79, !80}
!84 = distinct !{!84, !11, !36, !37}
!85 = distinct !{!85, !11, !37, !36}
!86 = distinct !{!86, !11, !79, !80}
!87 = distinct !{!87, !11, !36, !37}
!88 = distinct !{!88, !11, !37, !36}
!89 = distinct !{!89, !11, !79, !80}
!90 = distinct !{!90, !11, !36, !37}
!91 = distinct !{!91, !11, !37, !36}
!92 = distinct !{!92, !11, !79, !80}
!93 = distinct !{!93, !11, !94}
!94 = !{!"llvm.loop.vectorize.enable", i1 true}
!95 = distinct !{!95, !11, !79, !80}
!96 = distinct !{!96, !11, !94}
!97 = distinct !{!97, !11, !79, !80}
!98 = distinct !{!98, !11, !94}
!99 = distinct !{!99, !11, !79, !80}
!100 = distinct !{!100, !11, !94}
!101 = distinct !{!101, !11, !79, !80}
!102 = distinct !{!102, !11, !94}
!103 = distinct !{!103, !11, !79, !80}
!104 = distinct !{!104, !11, !94}
!105 = distinct !{!105, !11, !79, !80}
!106 = distinct !{!106, !11, !94}
!107 = distinct !{!107, !11, !79, !80}
!108 = distinct !{!108, !11, !94}
!109 = distinct !{!109, !11, !79, !80}
!110 = distinct !{!110, !11, !94}
