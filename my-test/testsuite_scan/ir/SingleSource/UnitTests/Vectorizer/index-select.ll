; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/Vectorizer/index-select.cpp'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/Vectorizer/index-select.cpp"
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
%"class.std::uniform_int_distribution" = type { %"struct.std::uniform_int_distribution<unsigned int>::param_type" }
%"struct.std::uniform_int_distribution<unsigned int>::param_type" = type { i32, i32 }
%"struct.__gnu_cxx::__ops::_Iter_less_iter" = type { i8 }

$__clang_call_terminate = comdat any

$_ZNSt24uniform_int_distributionIjEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEjRT_RKNS0_10param_typeE = comdat any

$_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv = comdat any

$_ZSt16__introsort_loopIPjlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_ = comdat any

$_ZSt22__final_insertion_sortIPjN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_ = comdat any

$_ZSt11__make_heapIPjN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_RT0_ = comdat any

@_ZL3rng = internal global %"class.std::mersenne_twister_engine" zeroinitializer, align 8
@.str = private unnamed_addr constant [33 x i8] c"min_index_select_u32_u32_start_0\00", align 1
@.str.1 = private unnamed_addr constant [39 x i8] c"min_index_select_u32_u32_start_0_inc_2\00", align 1
@.str.2 = private unnamed_addr constant [33 x i8] c"min_index_select_u32_u32_start_2\00", align 1
@.str.3 = private unnamed_addr constant [36 x i8] c"min_index_select_u32_u32_with_trunc\00", align 1
@.str.4 = private unnamed_addr constant [45 x i8] c"min_index_select_u32_u32_induction_decrement\00", align 1
@.str.5 = private unnamed_addr constant [46 x i8] c"min_index_select_u32_u32_start_0_min_idx_neg1\00", align 1
@.str.6 = private unnamed_addr constant [43 x i8] c"min_index_select_u32_u32_start_3_min_idx_3\00", align 1
@.str.7 = private unnamed_addr constant [43 x i8] c"min_index_select_u32_u32_start_3_min_idx_2\00", align 1
@.str.8 = private unnamed_addr constant [43 x i8] c"min_index_select_u32_u32_start_3_min_idx_4\00", align 1
@_ZSt4cout = external global %"class.std::basic_ostream", align 8
@.str.9 = private unnamed_addr constant [10 x i8] c"Checking \00", align 1
@.str.10 = private unnamed_addr constant [2 x i8] c"\0A\00", align 1
@_ZSt4cerr = external global %"class.std::basic_ostream", align 8
@.str.11 = private unnamed_addr constant [12 x i8] c"Miscompare\0A\00", align 1
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
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_index_select.cpp, ptr null }]

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
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #18
  store i64 15, ptr %1, align 8, !tbaa !6
  br label %20

20:                                               ; preds = %20, %0
  %21 = phi i64 [ 15, %0 ], [ %28, %20 ]
  %22 = phi i64 [ 1, %0 ], [ %29, %20 ]
  %23 = getelementptr i64, ptr %1, i64 %22
  %24 = lshr i64 %21, 30
  %25 = xor i64 %24, %21
  %26 = mul nuw nsw i64 %25, 1812433253
  %27 = add nuw i64 %26, %22
  %28 = and i64 %27, 4294967295
  store i64 %28, ptr %23, align 8, !tbaa !6
  %29 = add nuw nsw i64 %22, 1
  %30 = icmp eq i64 %29, 624
  br i1 %30, label %31, label %20, !llvm.loop !10

31:                                               ; preds = %20
  %32 = getelementptr inbounds nuw i8, ptr %1, i64 4992
  store i64 624, ptr %32, align 8, !tbaa !12
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(5000) @_ZL3rng, ptr noundef nonnull align 8 dereferenceable(5000) %1, i64 5000, i1 false), !tbaa.struct !14
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #18
  %33 = getelementptr inbounds nuw i8, ptr %2, i64 16
  %34 = getelementptr inbounds nuw i8, ptr %2, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %2, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE3$_0E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %34, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE3$_0E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %33, align 8, !tbaa !20
  %35 = getelementptr inbounds nuw i8, ptr %3, i64 16
  %36 = getelementptr inbounds nuw i8, ptr %3, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %3, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE3$_1E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %36, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE3$_1E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %35, align 8, !tbaa !20
  invoke fastcc void @_ZL19checkVectorFunctionIjjEvSt8functionIFT_PT0_S3_jEES5_PKc(ptr noundef %2, ptr noundef %3, ptr noundef nonnull @.str)
          to label %37 unwind label %222

37:                                               ; preds = %31
  %38 = load ptr, ptr %35, align 8, !tbaa !20
  %39 = icmp eq ptr %38, null
  br i1 %39, label %45, label %40

40:                                               ; preds = %37
  %41 = invoke noundef i1 %38(ptr noundef nonnull align 8 dereferenceable(32) %3, ptr noundef nonnull align 8 dereferenceable(32) %3, i32 noundef 3)
          to label %45 unwind label %42

42:                                               ; preds = %40
  %43 = landingpad { ptr, i32 }
          catch ptr null
  %44 = extractvalue { ptr, i32 } %43, 0
  call void @__clang_call_terminate(ptr %44) #19
  unreachable

45:                                               ; preds = %37, %40
  %46 = load ptr, ptr %33, align 8, !tbaa !20
  %47 = icmp eq ptr %46, null
  br i1 %47, label %53, label %48

48:                                               ; preds = %45
  %49 = invoke noundef i1 %46(ptr noundef nonnull align 8 dereferenceable(32) %2, ptr noundef nonnull align 8 dereferenceable(32) %2, i32 noundef 3)
          to label %53 unwind label %50

50:                                               ; preds = %48
  %51 = landingpad { ptr, i32 }
          catch ptr null
  %52 = extractvalue { ptr, i32 } %51, 0
  call void @__clang_call_terminate(ptr %52) #19
  unreachable

53:                                               ; preds = %45, %48
  %54 = getelementptr inbounds nuw i8, ptr %4, i64 16
  %55 = getelementptr inbounds nuw i8, ptr %4, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %4, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE3$_2E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %55, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE3$_2E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %54, align 8, !tbaa !20
  %56 = getelementptr inbounds nuw i8, ptr %5, i64 16
  %57 = getelementptr inbounds nuw i8, ptr %5, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %5, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE3$_3E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %57, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE3$_3E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %56, align 8, !tbaa !20
  invoke fastcc void @_ZL19checkVectorFunctionIjjEvSt8functionIFT_PT0_S3_jEES5_PKc(ptr noundef %4, ptr noundef %5, ptr noundef nonnull @.str.1)
          to label %58 unwind label %239

58:                                               ; preds = %53
  %59 = load ptr, ptr %56, align 8, !tbaa !20
  %60 = icmp eq ptr %59, null
  br i1 %60, label %66, label %61

61:                                               ; preds = %58
  %62 = invoke noundef i1 %59(ptr noundef nonnull align 8 dereferenceable(32) %5, ptr noundef nonnull align 8 dereferenceable(32) %5, i32 noundef 3)
          to label %66 unwind label %63

63:                                               ; preds = %61
  %64 = landingpad { ptr, i32 }
          catch ptr null
  %65 = extractvalue { ptr, i32 } %64, 0
  call void @__clang_call_terminate(ptr %65) #19
  unreachable

66:                                               ; preds = %58, %61
  %67 = load ptr, ptr %54, align 8, !tbaa !20
  %68 = icmp eq ptr %67, null
  br i1 %68, label %74, label %69

69:                                               ; preds = %66
  %70 = invoke noundef i1 %67(ptr noundef nonnull align 8 dereferenceable(32) %4, ptr noundef nonnull align 8 dereferenceable(32) %4, i32 noundef 3)
          to label %74 unwind label %71

71:                                               ; preds = %69
  %72 = landingpad { ptr, i32 }
          catch ptr null
  %73 = extractvalue { ptr, i32 } %72, 0
  call void @__clang_call_terminate(ptr %73) #19
  unreachable

74:                                               ; preds = %66, %69
  %75 = getelementptr inbounds nuw i8, ptr %6, i64 16
  %76 = getelementptr inbounds nuw i8, ptr %6, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %6, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE3$_4E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %76, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE3$_4E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %75, align 8, !tbaa !20
  %77 = getelementptr inbounds nuw i8, ptr %7, i64 16
  %78 = getelementptr inbounds nuw i8, ptr %7, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %7, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE3$_5E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %78, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE3$_5E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %77, align 8, !tbaa !20
  invoke fastcc void @_ZL19checkVectorFunctionIjjEvSt8functionIFT_PT0_S3_jEES5_PKc(ptr noundef %6, ptr noundef %7, ptr noundef nonnull @.str.2)
          to label %79 unwind label %256

79:                                               ; preds = %74
  %80 = load ptr, ptr %77, align 8, !tbaa !20
  %81 = icmp eq ptr %80, null
  br i1 %81, label %87, label %82

82:                                               ; preds = %79
  %83 = invoke noundef i1 %80(ptr noundef nonnull align 8 dereferenceable(32) %7, ptr noundef nonnull align 8 dereferenceable(32) %7, i32 noundef 3)
          to label %87 unwind label %84

84:                                               ; preds = %82
  %85 = landingpad { ptr, i32 }
          catch ptr null
  %86 = extractvalue { ptr, i32 } %85, 0
  call void @__clang_call_terminate(ptr %86) #19
  unreachable

87:                                               ; preds = %79, %82
  %88 = load ptr, ptr %75, align 8, !tbaa !20
  %89 = icmp eq ptr %88, null
  br i1 %89, label %95, label %90

90:                                               ; preds = %87
  %91 = invoke noundef i1 %88(ptr noundef nonnull align 8 dereferenceable(32) %6, ptr noundef nonnull align 8 dereferenceable(32) %6, i32 noundef 3)
          to label %95 unwind label %92

92:                                               ; preds = %90
  %93 = landingpad { ptr, i32 }
          catch ptr null
  %94 = extractvalue { ptr, i32 } %93, 0
  call void @__clang_call_terminate(ptr %94) #19
  unreachable

95:                                               ; preds = %87, %90
  %96 = getelementptr inbounds nuw i8, ptr %8, i64 16
  %97 = getelementptr inbounds nuw i8, ptr %8, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %8, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE3$_6E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %97, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE3$_6E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %96, align 8, !tbaa !20
  %98 = getelementptr inbounds nuw i8, ptr %9, i64 16
  %99 = getelementptr inbounds nuw i8, ptr %9, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %9, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE3$_7E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %99, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE3$_7E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %98, align 8, !tbaa !20
  invoke fastcc void @_ZL19checkVectorFunctionIjjEvSt8functionIFT_PT0_S3_jEES5_PKc(ptr noundef %8, ptr noundef %9, ptr noundef nonnull @.str.3)
          to label %100 unwind label %273

100:                                              ; preds = %95
  %101 = load ptr, ptr %98, align 8, !tbaa !20
  %102 = icmp eq ptr %101, null
  br i1 %102, label %108, label %103

103:                                              ; preds = %100
  %104 = invoke noundef i1 %101(ptr noundef nonnull align 8 dereferenceable(32) %9, ptr noundef nonnull align 8 dereferenceable(32) %9, i32 noundef 3)
          to label %108 unwind label %105

105:                                              ; preds = %103
  %106 = landingpad { ptr, i32 }
          catch ptr null
  %107 = extractvalue { ptr, i32 } %106, 0
  call void @__clang_call_terminate(ptr %107) #19
  unreachable

108:                                              ; preds = %100, %103
  %109 = load ptr, ptr %96, align 8, !tbaa !20
  %110 = icmp eq ptr %109, null
  br i1 %110, label %116, label %111

111:                                              ; preds = %108
  %112 = invoke noundef i1 %109(ptr noundef nonnull align 8 dereferenceable(32) %8, ptr noundef nonnull align 8 dereferenceable(32) %8, i32 noundef 3)
          to label %116 unwind label %113

113:                                              ; preds = %111
  %114 = landingpad { ptr, i32 }
          catch ptr null
  %115 = extractvalue { ptr, i32 } %114, 0
  call void @__clang_call_terminate(ptr %115) #19
  unreachable

116:                                              ; preds = %108, %111
  %117 = getelementptr inbounds nuw i8, ptr %10, i64 16
  %118 = getelementptr inbounds nuw i8, ptr %10, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %10, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE3$_8E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %118, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE3$_8E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %117, align 8, !tbaa !20
  %119 = getelementptr inbounds nuw i8, ptr %11, i64 16
  %120 = getelementptr inbounds nuw i8, ptr %11, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %11, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE3$_9E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %120, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE3$_9E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %119, align 8, !tbaa !20
  invoke fastcc void @_ZL19checkVectorFunctionIjjEvSt8functionIFT_PT0_S3_jEES5_PKc(ptr noundef %10, ptr noundef %11, ptr noundef nonnull @.str.4)
          to label %121 unwind label %290

121:                                              ; preds = %116
  %122 = load ptr, ptr %119, align 8, !tbaa !20
  %123 = icmp eq ptr %122, null
  br i1 %123, label %129, label %124

124:                                              ; preds = %121
  %125 = invoke noundef i1 %122(ptr noundef nonnull align 8 dereferenceable(32) %11, ptr noundef nonnull align 8 dereferenceable(32) %11, i32 noundef 3)
          to label %129 unwind label %126

126:                                              ; preds = %124
  %127 = landingpad { ptr, i32 }
          catch ptr null
  %128 = extractvalue { ptr, i32 } %127, 0
  call void @__clang_call_terminate(ptr %128) #19
  unreachable

129:                                              ; preds = %121, %124
  %130 = load ptr, ptr %117, align 8, !tbaa !20
  %131 = icmp eq ptr %130, null
  br i1 %131, label %137, label %132

132:                                              ; preds = %129
  %133 = invoke noundef i1 %130(ptr noundef nonnull align 8 dereferenceable(32) %10, ptr noundef nonnull align 8 dereferenceable(32) %10, i32 noundef 3)
          to label %137 unwind label %134

134:                                              ; preds = %132
  %135 = landingpad { ptr, i32 }
          catch ptr null
  %136 = extractvalue { ptr, i32 } %135, 0
  call void @__clang_call_terminate(ptr %136) #19
  unreachable

137:                                              ; preds = %129, %132
  %138 = getelementptr inbounds nuw i8, ptr %12, i64 16
  %139 = getelementptr inbounds nuw i8, ptr %12, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %12, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE4$_10E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %139, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE4$_10E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %138, align 8, !tbaa !20
  %140 = getelementptr inbounds nuw i8, ptr %13, i64 16
  %141 = getelementptr inbounds nuw i8, ptr %13, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %13, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE4$_11E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %141, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE4$_11E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %140, align 8, !tbaa !20
  invoke fastcc void @_ZL19checkVectorFunctionIjjEvSt8functionIFT_PT0_S3_jEES5_PKc(ptr noundef %12, ptr noundef %13, ptr noundef nonnull @.str.5)
          to label %142 unwind label %307

142:                                              ; preds = %137
  %143 = load ptr, ptr %140, align 8, !tbaa !20
  %144 = icmp eq ptr %143, null
  br i1 %144, label %150, label %145

145:                                              ; preds = %142
  %146 = invoke noundef i1 %143(ptr noundef nonnull align 8 dereferenceable(32) %13, ptr noundef nonnull align 8 dereferenceable(32) %13, i32 noundef 3)
          to label %150 unwind label %147

147:                                              ; preds = %145
  %148 = landingpad { ptr, i32 }
          catch ptr null
  %149 = extractvalue { ptr, i32 } %148, 0
  call void @__clang_call_terminate(ptr %149) #19
  unreachable

150:                                              ; preds = %142, %145
  %151 = load ptr, ptr %138, align 8, !tbaa !20
  %152 = icmp eq ptr %151, null
  br i1 %152, label %158, label %153

153:                                              ; preds = %150
  %154 = invoke noundef i1 %151(ptr noundef nonnull align 8 dereferenceable(32) %12, ptr noundef nonnull align 8 dereferenceable(32) %12, i32 noundef 3)
          to label %158 unwind label %155

155:                                              ; preds = %153
  %156 = landingpad { ptr, i32 }
          catch ptr null
  %157 = extractvalue { ptr, i32 } %156, 0
  call void @__clang_call_terminate(ptr %157) #19
  unreachable

158:                                              ; preds = %150, %153
  %159 = getelementptr inbounds nuw i8, ptr %14, i64 16
  %160 = getelementptr inbounds nuw i8, ptr %14, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %14, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE4$_12E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %160, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE4$_12E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %159, align 8, !tbaa !20
  %161 = getelementptr inbounds nuw i8, ptr %15, i64 16
  %162 = getelementptr inbounds nuw i8, ptr %15, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %15, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE4$_13E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %162, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE4$_13E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %161, align 8, !tbaa !20
  invoke fastcc void @_ZL19checkVectorFunctionIjjEvSt8functionIFT_PT0_S3_jEES5_PKc(ptr noundef %14, ptr noundef %15, ptr noundef nonnull @.str.6)
          to label %163 unwind label %324

163:                                              ; preds = %158
  %164 = load ptr, ptr %161, align 8, !tbaa !20
  %165 = icmp eq ptr %164, null
  br i1 %165, label %171, label %166

166:                                              ; preds = %163
  %167 = invoke noundef i1 %164(ptr noundef nonnull align 8 dereferenceable(32) %15, ptr noundef nonnull align 8 dereferenceable(32) %15, i32 noundef 3)
          to label %171 unwind label %168

168:                                              ; preds = %166
  %169 = landingpad { ptr, i32 }
          catch ptr null
  %170 = extractvalue { ptr, i32 } %169, 0
  call void @__clang_call_terminate(ptr %170) #19
  unreachable

171:                                              ; preds = %163, %166
  %172 = load ptr, ptr %159, align 8, !tbaa !20
  %173 = icmp eq ptr %172, null
  br i1 %173, label %179, label %174

174:                                              ; preds = %171
  %175 = invoke noundef i1 %172(ptr noundef nonnull align 8 dereferenceable(32) %14, ptr noundef nonnull align 8 dereferenceable(32) %14, i32 noundef 3)
          to label %179 unwind label %176

176:                                              ; preds = %174
  %177 = landingpad { ptr, i32 }
          catch ptr null
  %178 = extractvalue { ptr, i32 } %177, 0
  call void @__clang_call_terminate(ptr %178) #19
  unreachable

179:                                              ; preds = %171, %174
  %180 = getelementptr inbounds nuw i8, ptr %16, i64 16
  %181 = getelementptr inbounds nuw i8, ptr %16, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %16, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE4$_14E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %181, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE4$_14E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %180, align 8, !tbaa !20
  %182 = getelementptr inbounds nuw i8, ptr %17, i64 16
  %183 = getelementptr inbounds nuw i8, ptr %17, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %17, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE4$_15E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %183, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE4$_15E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %182, align 8, !tbaa !20
  invoke fastcc void @_ZL19checkVectorFunctionIjjEvSt8functionIFT_PT0_S3_jEES5_PKc(ptr noundef %16, ptr noundef %17, ptr noundef nonnull @.str.7)
          to label %184 unwind label %341

184:                                              ; preds = %179
  %185 = load ptr, ptr %182, align 8, !tbaa !20
  %186 = icmp eq ptr %185, null
  br i1 %186, label %192, label %187

187:                                              ; preds = %184
  %188 = invoke noundef i1 %185(ptr noundef nonnull align 8 dereferenceable(32) %17, ptr noundef nonnull align 8 dereferenceable(32) %17, i32 noundef 3)
          to label %192 unwind label %189

189:                                              ; preds = %187
  %190 = landingpad { ptr, i32 }
          catch ptr null
  %191 = extractvalue { ptr, i32 } %190, 0
  call void @__clang_call_terminate(ptr %191) #19
  unreachable

192:                                              ; preds = %184, %187
  %193 = load ptr, ptr %180, align 8, !tbaa !20
  %194 = icmp eq ptr %193, null
  br i1 %194, label %200, label %195

195:                                              ; preds = %192
  %196 = invoke noundef i1 %193(ptr noundef nonnull align 8 dereferenceable(32) %16, ptr noundef nonnull align 8 dereferenceable(32) %16, i32 noundef 3)
          to label %200 unwind label %197

197:                                              ; preds = %195
  %198 = landingpad { ptr, i32 }
          catch ptr null
  %199 = extractvalue { ptr, i32 } %198, 0
  call void @__clang_call_terminate(ptr %199) #19
  unreachable

200:                                              ; preds = %192, %195
  %201 = getelementptr inbounds nuw i8, ptr %18, i64 16
  %202 = getelementptr inbounds nuw i8, ptr %18, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %18, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE4$_16E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %202, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE4$_16E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %201, align 8, !tbaa !20
  %203 = getelementptr inbounds nuw i8, ptr %19, i64 16
  %204 = getelementptr inbounds nuw i8, ptr %19, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %19, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE4$_17E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %204, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE4$_17E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %203, align 8, !tbaa !20
  invoke fastcc void @_ZL19checkVectorFunctionIjjEvSt8functionIFT_PT0_S3_jEES5_PKc(ptr noundef %18, ptr noundef %19, ptr noundef nonnull @.str.8)
          to label %205 unwind label %358

205:                                              ; preds = %200
  %206 = load ptr, ptr %203, align 8, !tbaa !20
  %207 = icmp eq ptr %206, null
  br i1 %207, label %213, label %208

208:                                              ; preds = %205
  %209 = invoke noundef i1 %206(ptr noundef nonnull align 8 dereferenceable(32) %19, ptr noundef nonnull align 8 dereferenceable(32) %19, i32 noundef 3)
          to label %213 unwind label %210

210:                                              ; preds = %208
  %211 = landingpad { ptr, i32 }
          catch ptr null
  %212 = extractvalue { ptr, i32 } %211, 0
  call void @__clang_call_terminate(ptr %212) #19
  unreachable

213:                                              ; preds = %205, %208
  %214 = load ptr, ptr %201, align 8, !tbaa !20
  %215 = icmp eq ptr %214, null
  br i1 %215, label %221, label %216

216:                                              ; preds = %213
  %217 = invoke noundef i1 %214(ptr noundef nonnull align 8 dereferenceable(32) %18, ptr noundef nonnull align 8 dereferenceable(32) %18, i32 noundef 3)
          to label %221 unwind label %218

218:                                              ; preds = %216
  %219 = landingpad { ptr, i32 }
          catch ptr null
  %220 = extractvalue { ptr, i32 } %219, 0
  call void @__clang_call_terminate(ptr %220) #19
  unreachable

221:                                              ; preds = %213, %216
  ret i32 0

222:                                              ; preds = %31
  %223 = landingpad { ptr, i32 }
          cleanup
  %224 = load ptr, ptr %35, align 8, !tbaa !20
  %225 = icmp eq ptr %224, null
  br i1 %225, label %231, label %226

226:                                              ; preds = %222
  %227 = invoke noundef i1 %224(ptr noundef nonnull align 8 dereferenceable(32) %3, ptr noundef nonnull align 8 dereferenceable(32) %3, i32 noundef 3)
          to label %231 unwind label %228

228:                                              ; preds = %226
  %229 = landingpad { ptr, i32 }
          catch ptr null
  %230 = extractvalue { ptr, i32 } %229, 0
  call void @__clang_call_terminate(ptr %230) #19
  unreachable

231:                                              ; preds = %222, %226
  %232 = load ptr, ptr %33, align 8, !tbaa !20
  %233 = icmp eq ptr %232, null
  br i1 %233, label %375, label %234

234:                                              ; preds = %231
  %235 = invoke noundef i1 %232(ptr noundef nonnull align 8 dereferenceable(32) %2, ptr noundef nonnull align 8 dereferenceable(32) %2, i32 noundef 3)
          to label %375 unwind label %236

236:                                              ; preds = %234
  %237 = landingpad { ptr, i32 }
          catch ptr null
  %238 = extractvalue { ptr, i32 } %237, 0
  call void @__clang_call_terminate(ptr %238) #19
  unreachable

239:                                              ; preds = %53
  %240 = landingpad { ptr, i32 }
          cleanup
  %241 = load ptr, ptr %56, align 8, !tbaa !20
  %242 = icmp eq ptr %241, null
  br i1 %242, label %248, label %243

243:                                              ; preds = %239
  %244 = invoke noundef i1 %241(ptr noundef nonnull align 8 dereferenceable(32) %5, ptr noundef nonnull align 8 dereferenceable(32) %5, i32 noundef 3)
          to label %248 unwind label %245

245:                                              ; preds = %243
  %246 = landingpad { ptr, i32 }
          catch ptr null
  %247 = extractvalue { ptr, i32 } %246, 0
  call void @__clang_call_terminate(ptr %247) #19
  unreachable

248:                                              ; preds = %239, %243
  %249 = load ptr, ptr %54, align 8, !tbaa !20
  %250 = icmp eq ptr %249, null
  br i1 %250, label %375, label %251

251:                                              ; preds = %248
  %252 = invoke noundef i1 %249(ptr noundef nonnull align 8 dereferenceable(32) %4, ptr noundef nonnull align 8 dereferenceable(32) %4, i32 noundef 3)
          to label %375 unwind label %253

253:                                              ; preds = %251
  %254 = landingpad { ptr, i32 }
          catch ptr null
  %255 = extractvalue { ptr, i32 } %254, 0
  call void @__clang_call_terminate(ptr %255) #19
  unreachable

256:                                              ; preds = %74
  %257 = landingpad { ptr, i32 }
          cleanup
  %258 = load ptr, ptr %77, align 8, !tbaa !20
  %259 = icmp eq ptr %258, null
  br i1 %259, label %265, label %260

260:                                              ; preds = %256
  %261 = invoke noundef i1 %258(ptr noundef nonnull align 8 dereferenceable(32) %7, ptr noundef nonnull align 8 dereferenceable(32) %7, i32 noundef 3)
          to label %265 unwind label %262

262:                                              ; preds = %260
  %263 = landingpad { ptr, i32 }
          catch ptr null
  %264 = extractvalue { ptr, i32 } %263, 0
  call void @__clang_call_terminate(ptr %264) #19
  unreachable

265:                                              ; preds = %256, %260
  %266 = load ptr, ptr %75, align 8, !tbaa !20
  %267 = icmp eq ptr %266, null
  br i1 %267, label %375, label %268

268:                                              ; preds = %265
  %269 = invoke noundef i1 %266(ptr noundef nonnull align 8 dereferenceable(32) %6, ptr noundef nonnull align 8 dereferenceable(32) %6, i32 noundef 3)
          to label %375 unwind label %270

270:                                              ; preds = %268
  %271 = landingpad { ptr, i32 }
          catch ptr null
  %272 = extractvalue { ptr, i32 } %271, 0
  call void @__clang_call_terminate(ptr %272) #19
  unreachable

273:                                              ; preds = %95
  %274 = landingpad { ptr, i32 }
          cleanup
  %275 = load ptr, ptr %98, align 8, !tbaa !20
  %276 = icmp eq ptr %275, null
  br i1 %276, label %282, label %277

277:                                              ; preds = %273
  %278 = invoke noundef i1 %275(ptr noundef nonnull align 8 dereferenceable(32) %9, ptr noundef nonnull align 8 dereferenceable(32) %9, i32 noundef 3)
          to label %282 unwind label %279

279:                                              ; preds = %277
  %280 = landingpad { ptr, i32 }
          catch ptr null
  %281 = extractvalue { ptr, i32 } %280, 0
  call void @__clang_call_terminate(ptr %281) #19
  unreachable

282:                                              ; preds = %273, %277
  %283 = load ptr, ptr %96, align 8, !tbaa !20
  %284 = icmp eq ptr %283, null
  br i1 %284, label %375, label %285

285:                                              ; preds = %282
  %286 = invoke noundef i1 %283(ptr noundef nonnull align 8 dereferenceable(32) %8, ptr noundef nonnull align 8 dereferenceable(32) %8, i32 noundef 3)
          to label %375 unwind label %287

287:                                              ; preds = %285
  %288 = landingpad { ptr, i32 }
          catch ptr null
  %289 = extractvalue { ptr, i32 } %288, 0
  call void @__clang_call_terminate(ptr %289) #19
  unreachable

290:                                              ; preds = %116
  %291 = landingpad { ptr, i32 }
          cleanup
  %292 = load ptr, ptr %119, align 8, !tbaa !20
  %293 = icmp eq ptr %292, null
  br i1 %293, label %299, label %294

294:                                              ; preds = %290
  %295 = invoke noundef i1 %292(ptr noundef nonnull align 8 dereferenceable(32) %11, ptr noundef nonnull align 8 dereferenceable(32) %11, i32 noundef 3)
          to label %299 unwind label %296

296:                                              ; preds = %294
  %297 = landingpad { ptr, i32 }
          catch ptr null
  %298 = extractvalue { ptr, i32 } %297, 0
  call void @__clang_call_terminate(ptr %298) #19
  unreachable

299:                                              ; preds = %290, %294
  %300 = load ptr, ptr %117, align 8, !tbaa !20
  %301 = icmp eq ptr %300, null
  br i1 %301, label %375, label %302

302:                                              ; preds = %299
  %303 = invoke noundef i1 %300(ptr noundef nonnull align 8 dereferenceable(32) %10, ptr noundef nonnull align 8 dereferenceable(32) %10, i32 noundef 3)
          to label %375 unwind label %304

304:                                              ; preds = %302
  %305 = landingpad { ptr, i32 }
          catch ptr null
  %306 = extractvalue { ptr, i32 } %305, 0
  call void @__clang_call_terminate(ptr %306) #19
  unreachable

307:                                              ; preds = %137
  %308 = landingpad { ptr, i32 }
          cleanup
  %309 = load ptr, ptr %140, align 8, !tbaa !20
  %310 = icmp eq ptr %309, null
  br i1 %310, label %316, label %311

311:                                              ; preds = %307
  %312 = invoke noundef i1 %309(ptr noundef nonnull align 8 dereferenceable(32) %13, ptr noundef nonnull align 8 dereferenceable(32) %13, i32 noundef 3)
          to label %316 unwind label %313

313:                                              ; preds = %311
  %314 = landingpad { ptr, i32 }
          catch ptr null
  %315 = extractvalue { ptr, i32 } %314, 0
  call void @__clang_call_terminate(ptr %315) #19
  unreachable

316:                                              ; preds = %307, %311
  %317 = load ptr, ptr %138, align 8, !tbaa !20
  %318 = icmp eq ptr %317, null
  br i1 %318, label %375, label %319

319:                                              ; preds = %316
  %320 = invoke noundef i1 %317(ptr noundef nonnull align 8 dereferenceable(32) %12, ptr noundef nonnull align 8 dereferenceable(32) %12, i32 noundef 3)
          to label %375 unwind label %321

321:                                              ; preds = %319
  %322 = landingpad { ptr, i32 }
          catch ptr null
  %323 = extractvalue { ptr, i32 } %322, 0
  call void @__clang_call_terminate(ptr %323) #19
  unreachable

324:                                              ; preds = %158
  %325 = landingpad { ptr, i32 }
          cleanup
  %326 = load ptr, ptr %161, align 8, !tbaa !20
  %327 = icmp eq ptr %326, null
  br i1 %327, label %333, label %328

328:                                              ; preds = %324
  %329 = invoke noundef i1 %326(ptr noundef nonnull align 8 dereferenceable(32) %15, ptr noundef nonnull align 8 dereferenceable(32) %15, i32 noundef 3)
          to label %333 unwind label %330

330:                                              ; preds = %328
  %331 = landingpad { ptr, i32 }
          catch ptr null
  %332 = extractvalue { ptr, i32 } %331, 0
  call void @__clang_call_terminate(ptr %332) #19
  unreachable

333:                                              ; preds = %324, %328
  %334 = load ptr, ptr %159, align 8, !tbaa !20
  %335 = icmp eq ptr %334, null
  br i1 %335, label %375, label %336

336:                                              ; preds = %333
  %337 = invoke noundef i1 %334(ptr noundef nonnull align 8 dereferenceable(32) %14, ptr noundef nonnull align 8 dereferenceable(32) %14, i32 noundef 3)
          to label %375 unwind label %338

338:                                              ; preds = %336
  %339 = landingpad { ptr, i32 }
          catch ptr null
  %340 = extractvalue { ptr, i32 } %339, 0
  call void @__clang_call_terminate(ptr %340) #19
  unreachable

341:                                              ; preds = %179
  %342 = landingpad { ptr, i32 }
          cleanup
  %343 = load ptr, ptr %182, align 8, !tbaa !20
  %344 = icmp eq ptr %343, null
  br i1 %344, label %350, label %345

345:                                              ; preds = %341
  %346 = invoke noundef i1 %343(ptr noundef nonnull align 8 dereferenceable(32) %17, ptr noundef nonnull align 8 dereferenceable(32) %17, i32 noundef 3)
          to label %350 unwind label %347

347:                                              ; preds = %345
  %348 = landingpad { ptr, i32 }
          catch ptr null
  %349 = extractvalue { ptr, i32 } %348, 0
  call void @__clang_call_terminate(ptr %349) #19
  unreachable

350:                                              ; preds = %341, %345
  %351 = load ptr, ptr %180, align 8, !tbaa !20
  %352 = icmp eq ptr %351, null
  br i1 %352, label %375, label %353

353:                                              ; preds = %350
  %354 = invoke noundef i1 %351(ptr noundef nonnull align 8 dereferenceable(32) %16, ptr noundef nonnull align 8 dereferenceable(32) %16, i32 noundef 3)
          to label %375 unwind label %355

355:                                              ; preds = %353
  %356 = landingpad { ptr, i32 }
          catch ptr null
  %357 = extractvalue { ptr, i32 } %356, 0
  call void @__clang_call_terminate(ptr %357) #19
  unreachable

358:                                              ; preds = %200
  %359 = landingpad { ptr, i32 }
          cleanup
  %360 = load ptr, ptr %203, align 8, !tbaa !20
  %361 = icmp eq ptr %360, null
  br i1 %361, label %367, label %362

362:                                              ; preds = %358
  %363 = invoke noundef i1 %360(ptr noundef nonnull align 8 dereferenceable(32) %19, ptr noundef nonnull align 8 dereferenceable(32) %19, i32 noundef 3)
          to label %367 unwind label %364

364:                                              ; preds = %362
  %365 = landingpad { ptr, i32 }
          catch ptr null
  %366 = extractvalue { ptr, i32 } %365, 0
  call void @__clang_call_terminate(ptr %366) #19
  unreachable

367:                                              ; preds = %358, %362
  %368 = load ptr, ptr %201, align 8, !tbaa !20
  %369 = icmp eq ptr %368, null
  br i1 %369, label %375, label %370

370:                                              ; preds = %367
  %371 = invoke noundef i1 %368(ptr noundef nonnull align 8 dereferenceable(32) %18, ptr noundef nonnull align 8 dereferenceable(32) %18, i32 noundef 3)
          to label %375 unwind label %372

372:                                              ; preds = %370
  %373 = landingpad { ptr, i32 }
          catch ptr null
  %374 = extractvalue { ptr, i32 } %373, 0
  call void @__clang_call_terminate(ptr %374) #19
  unreachable

375:                                              ; preds = %370, %367, %353, %350, %336, %333, %319, %316, %302, %299, %285, %282, %268, %265, %251, %248, %234, %231
  %376 = phi { ptr, i32 } [ %223, %231 ], [ %223, %234 ], [ %240, %248 ], [ %240, %251 ], [ %257, %265 ], [ %257, %268 ], [ %274, %282 ], [ %274, %285 ], [ %291, %299 ], [ %291, %302 ], [ %308, %316 ], [ %308, %319 ], [ %325, %333 ], [ %325, %336 ], [ %342, %350 ], [ %342, %353 ], [ %359, %367 ], [ %359, %370 ]
  resume { ptr, i32 } %376
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: mustprogress norecurse uwtable
define internal fastcc void @_ZL19checkVectorFunctionIjjEvSt8functionIFT_PT0_S3_jEES5_PKc(ptr noundef nonnull %0, ptr noundef nonnull %1, ptr noundef %2) unnamed_addr #0 personality ptr @__gxx_personality_v0 {
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca i32, align 4
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  %9 = alloca i32, align 4
  %10 = alloca ptr, align 8
  %11 = alloca ptr, align 8
  %12 = alloca i32, align 4
  %13 = alloca ptr, align 8
  %14 = alloca ptr, align 8
  %15 = alloca i32, align 4
  %16 = alloca ptr, align 8
  %17 = alloca ptr, align 8
  %18 = alloca i32, align 4
  %19 = alloca ptr, align 8
  %20 = alloca ptr, align 8
  %21 = alloca i32, align 4
  %22 = alloca ptr, align 8
  %23 = alloca ptr, align 8
  %24 = alloca i32, align 4
  %25 = alloca ptr, align 8
  %26 = alloca ptr, align 8
  %27 = alloca i32, align 4
  %28 = alloca ptr, align 8
  %29 = alloca ptr, align 8
  %30 = alloca i32, align 4
  %31 = alloca ptr, align 8
  %32 = alloca ptr, align 8
  %33 = alloca i32, align 4
  %34 = alloca ptr, align 8
  %35 = alloca ptr, align 8
  %36 = alloca i32, align 4
  %37 = alloca ptr, align 8
  %38 = alloca ptr, align 8
  %39 = alloca i32, align 4
  %40 = alloca %"class.std::uniform_int_distribution", align 8
  %41 = alloca %"class.std::uniform_int_distribution", align 8
  %42 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.9, i64 noundef 9)
  %43 = icmp eq ptr %2, null
  br i1 %43, label %44, label %52

44:                                               ; preds = %3
  %45 = load ptr, ptr @_ZSt4cout, align 8, !tbaa !21
  %46 = getelementptr i8, ptr %45, i64 -24
  %47 = load i64, ptr %46, align 8
  %48 = getelementptr inbounds i8, ptr @_ZSt4cout, i64 %47
  %49 = getelementptr inbounds nuw i8, ptr %48, i64 32
  %50 = load i32, ptr %49, align 8, !tbaa !23
  %51 = or i32 %50, 1
  tail call void @_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate(ptr noundef nonnull align 8 dereferenceable(264) %48, i32 noundef %51)
  br label %55

52:                                               ; preds = %3
  %53 = tail call noundef i64 @strlen(ptr noundef nonnull dereferenceable(1) %2) #18
  %54 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull %2, i64 noundef %53)
  br label %55

55:                                               ; preds = %44, %52
  %56 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.10, i64 noundef 1)
  %57 = tail call noalias noundef nonnull dereferenceable(4000) ptr @_Znam(i64 noundef 4000) #20
  %58 = invoke noalias noundef nonnull dereferenceable(4000) ptr @_Znam(i64 noundef 4000) #20
          to label %59 unwind label %100

59:                                               ; preds = %55
  call void @llvm.lifetime.start.p0(ptr nonnull %41) #18
  store <2 x i32> <i32 0, i32 -1>, ptr %41, align 8, !tbaa !33
  br label %60

60:                                               ; preds = %63, %59
  %61 = phi i64 [ 0, %59 ], [ %65, %63 ]
  %62 = invoke noundef i32 @_ZNSt24uniform_int_distributionIjEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEjRT_RKNS0_10param_typeE(ptr noundef nonnull align 4 dereferenceable(8) %41, ptr noundef nonnull align 8 dereferenceable(5000) @_ZL3rng, ptr noundef nonnull align 4 dereferenceable(8) %41)
          to label %63 unwind label %104

63:                                               ; preds = %60
  %64 = getelementptr inbounds nuw i32, ptr %57, i64 %61
  store i32 %62, ptr %64, align 4, !tbaa !33
  %65 = add nuw nsw i64 %61, 1
  %66 = icmp eq i64 %65, 1000
  br i1 %66, label %67, label %60, !llvm.loop !34

67:                                               ; preds = %63
  call void @llvm.lifetime.end.p0(ptr nonnull %41) #18
  call void @llvm.lifetime.start.p0(ptr nonnull %40) #18
  store <2 x i32> <i32 0, i32 -1>, ptr %40, align 8, !tbaa !33
  br label %68

68:                                               ; preds = %71, %67
  %69 = phi i64 [ 0, %67 ], [ %73, %71 ]
  %70 = invoke noundef i32 @_ZNSt24uniform_int_distributionIjEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEjRT_RKNS0_10param_typeE(ptr noundef nonnull align 4 dereferenceable(8) %40, ptr noundef nonnull align 8 dereferenceable(5000) @_ZL3rng, ptr noundef nonnull align 4 dereferenceable(8) %40)
          to label %71 unwind label %102

71:                                               ; preds = %68
  %72 = getelementptr inbounds nuw i32, ptr %58, i64 %69
  store i32 %70, ptr %72, align 4, !tbaa !33
  %73 = add nuw nsw i64 %69, 1
  %74 = icmp eq i64 %73, 1000
  br i1 %74, label %75, label %68, !llvm.loop !34

75:                                               ; preds = %71
  call void @llvm.lifetime.end.p0(ptr nonnull %40) #18
  call void @llvm.lifetime.start.p0(ptr nonnull %37)
  call void @llvm.lifetime.start.p0(ptr nonnull %38)
  call void @llvm.lifetime.start.p0(ptr nonnull %39)
  store ptr %57, ptr %37, align 8, !tbaa !35
  store ptr %58, ptr %38, align 8, !tbaa !35
  store i32 1000, ptr %39, align 4, !tbaa !33
  %76 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %77 = load ptr, ptr %76, align 8, !tbaa !20
  %78 = icmp eq ptr %77, null
  br i1 %78, label %79, label %81

79:                                               ; preds = %75
  invoke void @_ZSt25__throw_bad_function_callv() #21
          to label %80 unwind label %108

80:                                               ; preds = %79
  unreachable

81:                                               ; preds = %75
  %82 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %83 = load ptr, ptr %82, align 8, !tbaa !16
  %84 = invoke noundef i32 %83(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %37, ptr noundef nonnull align 8 dereferenceable(8) %38, ptr noundef nonnull align 4 dereferenceable(4) %39)
          to label %85 unwind label %108

85:                                               ; preds = %81
  call void @llvm.lifetime.end.p0(ptr nonnull %37)
  call void @llvm.lifetime.end.p0(ptr nonnull %38)
  call void @llvm.lifetime.end.p0(ptr nonnull %39)
  call void @llvm.lifetime.start.p0(ptr nonnull %34)
  call void @llvm.lifetime.start.p0(ptr nonnull %35)
  call void @llvm.lifetime.start.p0(ptr nonnull %36)
  store ptr %57, ptr %34, align 8, !tbaa !35
  store ptr %58, ptr %35, align 8, !tbaa !35
  store i32 1000, ptr %36, align 4, !tbaa !33
  %86 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %87 = load ptr, ptr %86, align 8, !tbaa !20
  %88 = icmp eq ptr %87, null
  br i1 %88, label %89, label %91

89:                                               ; preds = %85
  invoke void @_ZSt25__throw_bad_function_callv() #21
          to label %90 unwind label %110

90:                                               ; preds = %89
  unreachable

91:                                               ; preds = %85
  %92 = getelementptr inbounds nuw i8, ptr %1, i64 24
  %93 = load ptr, ptr %92, align 8, !tbaa !16
  %94 = invoke noundef i32 %93(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %34, ptr noundef nonnull align 8 dereferenceable(8) %35, ptr noundef nonnull align 4 dereferenceable(4) %36)
          to label %95 unwind label %110

95:                                               ; preds = %91
  call void @llvm.lifetime.end.p0(ptr nonnull %34)
  call void @llvm.lifetime.end.p0(ptr nonnull %35)
  call void @llvm.lifetime.end.p0(ptr nonnull %36)
  %96 = icmp eq i32 %84, %94
  br i1 %96, label %112, label %97

97:                                               ; preds = %95
  %98 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.11)
          to label %99 unwind label %110

99:                                               ; preds = %97
  call void @exit(i32 noundef 1) #22
  unreachable

100:                                              ; preds = %55
  %101 = landingpad { ptr, i32 }
          cleanup
  br label %246

102:                                              ; preds = %68
  %103 = landingpad { ptr, i32 }
          cleanup
  br label %244

104:                                              ; preds = %60
  %105 = landingpad { ptr, i32 }
          cleanup
  br label %244

106:                                              ; preds = %117, %115, %114, %112
  %107 = landingpad { ptr, i32 }
          cleanup
  br label %244

108:                                              ; preds = %81, %79
  %109 = landingpad { ptr, i32 }
          cleanup
  br label %244

110:                                              ; preds = %91, %89, %97
  %111 = landingpad { ptr, i32 }
          cleanup
  br label %244

112:                                              ; preds = %95
  %113 = getelementptr inbounds nuw i8, ptr %57, i64 4000
  invoke void @_ZSt16__introsort_loopIPjlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_(ptr noundef nonnull %57, ptr noundef nonnull %113, i64 noundef 18, i8 undef)
          to label %114 unwind label %106

114:                                              ; preds = %112
  invoke void @_ZSt22__final_insertion_sortIPjN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_(ptr noundef nonnull %57, ptr noundef nonnull %113, i8 undef)
          to label %115 unwind label %106

115:                                              ; preds = %114
  %116 = getelementptr inbounds nuw i8, ptr %58, i64 4000
  invoke void @_ZSt16__introsort_loopIPjlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_(ptr noundef nonnull %58, ptr noundef nonnull %116, i64 noundef 18, i8 undef)
          to label %117 unwind label %106

117:                                              ; preds = %115
  invoke void @_ZSt22__final_insertion_sortIPjN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_(ptr noundef nonnull %58, ptr noundef nonnull %116, i8 undef)
          to label %118 unwind label %106

118:                                              ; preds = %117
  call void @llvm.lifetime.start.p0(ptr nonnull %31)
  call void @llvm.lifetime.start.p0(ptr nonnull %32)
  call void @llvm.lifetime.start.p0(ptr nonnull %33)
  store ptr %57, ptr %31, align 8, !tbaa !35
  store ptr %58, ptr %32, align 8, !tbaa !35
  store i32 1000, ptr %33, align 4, !tbaa !33
  %119 = load ptr, ptr %76, align 8, !tbaa !20
  %120 = icmp eq ptr %119, null
  br i1 %120, label %121, label %123

121:                                              ; preds = %118
  invoke void @_ZSt25__throw_bad_function_callv() #21
          to label %122 unwind label %142

122:                                              ; preds = %121
  unreachable

123:                                              ; preds = %118
  %124 = load ptr, ptr %82, align 8, !tbaa !16
  %125 = invoke noundef i32 %124(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %31, ptr noundef nonnull align 8 dereferenceable(8) %32, ptr noundef nonnull align 4 dereferenceable(4) %33)
          to label %126 unwind label %142

126:                                              ; preds = %123
  call void @llvm.lifetime.end.p0(ptr nonnull %31)
  call void @llvm.lifetime.end.p0(ptr nonnull %32)
  call void @llvm.lifetime.end.p0(ptr nonnull %33)
  call void @llvm.lifetime.start.p0(ptr nonnull %28)
  call void @llvm.lifetime.start.p0(ptr nonnull %29)
  call void @llvm.lifetime.start.p0(ptr nonnull %30)
  store ptr %57, ptr %28, align 8, !tbaa !35
  store ptr %58, ptr %29, align 8, !tbaa !35
  store i32 1000, ptr %30, align 4, !tbaa !33
  %127 = load ptr, ptr %86, align 8, !tbaa !20
  %128 = icmp eq ptr %127, null
  br i1 %128, label %129, label %131

129:                                              ; preds = %126
  invoke void @_ZSt25__throw_bad_function_callv() #21
          to label %130 unwind label %144

130:                                              ; preds = %129
  unreachable

131:                                              ; preds = %126
  %132 = load ptr, ptr %92, align 8, !tbaa !16
  %133 = invoke noundef i32 %132(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %28, ptr noundef nonnull align 8 dereferenceable(8) %29, ptr noundef nonnull align 4 dereferenceable(4) %30)
          to label %134 unwind label %144

134:                                              ; preds = %131
  call void @llvm.lifetime.end.p0(ptr nonnull %28)
  call void @llvm.lifetime.end.p0(ptr nonnull %29)
  call void @llvm.lifetime.end.p0(ptr nonnull %30)
  %135 = icmp eq i32 %125, %133
  br i1 %135, label %136, label %139

136:                                              ; preds = %134
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(4000) %57, i8 -1, i64 4000, i1 false), !tbaa !33
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(4000) %58, i8 -1, i64 4000, i1 false), !tbaa !33
  call void @llvm.lifetime.start.p0(ptr nonnull %25)
  call void @llvm.lifetime.start.p0(ptr nonnull %26)
  call void @llvm.lifetime.start.p0(ptr nonnull %27)
  store ptr %57, ptr %25, align 8, !tbaa !35
  store ptr %58, ptr %26, align 8, !tbaa !35
  store i32 1000, ptr %27, align 4, !tbaa !33
  %137 = load ptr, ptr %76, align 8, !tbaa !20
  %138 = icmp eq ptr %137, null
  br i1 %138, label %146, label %148

139:                                              ; preds = %134
  %140 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.11)
          to label %141 unwind label %144

141:                                              ; preds = %139
  call void @exit(i32 noundef 1) #22
  unreachable

142:                                              ; preds = %123, %121
  %143 = landingpad { ptr, i32 }
          cleanup
  br label %244

144:                                              ; preds = %131, %129, %139
  %145 = landingpad { ptr, i32 }
          cleanup
  br label %244

146:                                              ; preds = %136
  invoke void @_ZSt25__throw_bad_function_callv() #21
          to label %147 unwind label %167

147:                                              ; preds = %146
  unreachable

148:                                              ; preds = %136
  %149 = load ptr, ptr %82, align 8, !tbaa !16
  %150 = invoke noundef i32 %149(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %25, ptr noundef nonnull align 8 dereferenceable(8) %26, ptr noundef nonnull align 4 dereferenceable(4) %27)
          to label %151 unwind label %167

151:                                              ; preds = %148
  call void @llvm.lifetime.end.p0(ptr nonnull %25)
  call void @llvm.lifetime.end.p0(ptr nonnull %26)
  call void @llvm.lifetime.end.p0(ptr nonnull %27)
  call void @llvm.lifetime.start.p0(ptr nonnull %22)
  call void @llvm.lifetime.start.p0(ptr nonnull %23)
  call void @llvm.lifetime.start.p0(ptr nonnull %24)
  store ptr %57, ptr %22, align 8, !tbaa !35
  store ptr %58, ptr %23, align 8, !tbaa !35
  store i32 1000, ptr %24, align 4, !tbaa !33
  %152 = load ptr, ptr %86, align 8, !tbaa !20
  %153 = icmp eq ptr %152, null
  br i1 %153, label %154, label %156

154:                                              ; preds = %151
  invoke void @_ZSt25__throw_bad_function_callv() #21
          to label %155 unwind label %169

155:                                              ; preds = %154
  unreachable

156:                                              ; preds = %151
  %157 = load ptr, ptr %92, align 8, !tbaa !16
  %158 = invoke noundef i32 %157(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %22, ptr noundef nonnull align 8 dereferenceable(8) %23, ptr noundef nonnull align 4 dereferenceable(4) %24)
          to label %159 unwind label %169

159:                                              ; preds = %156
  call void @llvm.lifetime.end.p0(ptr nonnull %22)
  call void @llvm.lifetime.end.p0(ptr nonnull %23)
  call void @llvm.lifetime.end.p0(ptr nonnull %24)
  %160 = icmp eq i32 %150, %158
  br i1 %160, label %161, label %164

161:                                              ; preds = %159
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(4000) %57, i8 0, i64 4000, i1 false), !tbaa !33
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(4000) %58, i8 -1, i64 4000, i1 false), !tbaa !33
  call void @llvm.lifetime.start.p0(ptr nonnull %19)
  call void @llvm.lifetime.start.p0(ptr nonnull %20)
  call void @llvm.lifetime.start.p0(ptr nonnull %21)
  store ptr %57, ptr %19, align 8, !tbaa !35
  store ptr %58, ptr %20, align 8, !tbaa !35
  store i32 1000, ptr %21, align 4, !tbaa !33
  %162 = load ptr, ptr %76, align 8, !tbaa !20
  %163 = icmp eq ptr %162, null
  br i1 %163, label %171, label %173

164:                                              ; preds = %159
  %165 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.11)
          to label %166 unwind label %169

166:                                              ; preds = %164
  call void @exit(i32 noundef 1) #22
  unreachable

167:                                              ; preds = %148, %146
  %168 = landingpad { ptr, i32 }
          cleanup
  br label %244

169:                                              ; preds = %156, %154, %164
  %170 = landingpad { ptr, i32 }
          cleanup
  br label %244

171:                                              ; preds = %161
  invoke void @_ZSt25__throw_bad_function_callv() #21
          to label %172 unwind label %192

172:                                              ; preds = %171
  unreachable

173:                                              ; preds = %161
  %174 = load ptr, ptr %82, align 8, !tbaa !16
  %175 = invoke noundef i32 %174(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %19, ptr noundef nonnull align 8 dereferenceable(8) %20, ptr noundef nonnull align 4 dereferenceable(4) %21)
          to label %176 unwind label %192

176:                                              ; preds = %173
  call void @llvm.lifetime.end.p0(ptr nonnull %19)
  call void @llvm.lifetime.end.p0(ptr nonnull %20)
  call void @llvm.lifetime.end.p0(ptr nonnull %21)
  call void @llvm.lifetime.start.p0(ptr nonnull %16)
  call void @llvm.lifetime.start.p0(ptr nonnull %17)
  call void @llvm.lifetime.start.p0(ptr nonnull %18)
  store ptr %57, ptr %16, align 8, !tbaa !35
  store ptr %58, ptr %17, align 8, !tbaa !35
  store i32 1000, ptr %18, align 4, !tbaa !33
  %177 = load ptr, ptr %86, align 8, !tbaa !20
  %178 = icmp eq ptr %177, null
  br i1 %178, label %179, label %181

179:                                              ; preds = %176
  invoke void @_ZSt25__throw_bad_function_callv() #21
          to label %180 unwind label %194

180:                                              ; preds = %179
  unreachable

181:                                              ; preds = %176
  %182 = load ptr, ptr %92, align 8, !tbaa !16
  %183 = invoke noundef i32 %182(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %16, ptr noundef nonnull align 8 dereferenceable(8) %17, ptr noundef nonnull align 4 dereferenceable(4) %18)
          to label %184 unwind label %194

184:                                              ; preds = %181
  call void @llvm.lifetime.end.p0(ptr nonnull %16)
  call void @llvm.lifetime.end.p0(ptr nonnull %17)
  call void @llvm.lifetime.end.p0(ptr nonnull %18)
  %185 = icmp eq i32 %175, %183
  br i1 %185, label %186, label %189

186:                                              ; preds = %184
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(4000) %57, i8 -1, i64 4000, i1 false), !tbaa !33
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(4000) %58, i8 0, i64 4000, i1 false), !tbaa !33
  call void @llvm.lifetime.start.p0(ptr nonnull %13)
  call void @llvm.lifetime.start.p0(ptr nonnull %14)
  call void @llvm.lifetime.start.p0(ptr nonnull %15)
  store ptr %57, ptr %13, align 8, !tbaa !35
  store ptr %58, ptr %14, align 8, !tbaa !35
  store i32 1000, ptr %15, align 4, !tbaa !33
  %187 = load ptr, ptr %76, align 8, !tbaa !20
  %188 = icmp eq ptr %187, null
  br i1 %188, label %196, label %198

189:                                              ; preds = %184
  %190 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.11)
          to label %191 unwind label %194

191:                                              ; preds = %189
  call void @exit(i32 noundef 1) #22
  unreachable

192:                                              ; preds = %173, %171
  %193 = landingpad { ptr, i32 }
          cleanup
  br label %244

194:                                              ; preds = %181, %179, %189
  %195 = landingpad { ptr, i32 }
          cleanup
  br label %244

196:                                              ; preds = %186
  invoke void @_ZSt25__throw_bad_function_callv() #21
          to label %197 unwind label %217

197:                                              ; preds = %196
  unreachable

198:                                              ; preds = %186
  %199 = load ptr, ptr %82, align 8, !tbaa !16
  %200 = invoke noundef i32 %199(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %13, ptr noundef nonnull align 8 dereferenceable(8) %14, ptr noundef nonnull align 4 dereferenceable(4) %15)
          to label %201 unwind label %217

201:                                              ; preds = %198
  call void @llvm.lifetime.end.p0(ptr nonnull %13)
  call void @llvm.lifetime.end.p0(ptr nonnull %14)
  call void @llvm.lifetime.end.p0(ptr nonnull %15)
  call void @llvm.lifetime.start.p0(ptr nonnull %10)
  call void @llvm.lifetime.start.p0(ptr nonnull %11)
  call void @llvm.lifetime.start.p0(ptr nonnull %12)
  store ptr %57, ptr %10, align 8, !tbaa !35
  store ptr %58, ptr %11, align 8, !tbaa !35
  store i32 1000, ptr %12, align 4, !tbaa !33
  %202 = load ptr, ptr %86, align 8, !tbaa !20
  %203 = icmp eq ptr %202, null
  br i1 %203, label %204, label %206

204:                                              ; preds = %201
  invoke void @_ZSt25__throw_bad_function_callv() #21
          to label %205 unwind label %219

205:                                              ; preds = %204
  unreachable

206:                                              ; preds = %201
  %207 = load ptr, ptr %92, align 8, !tbaa !16
  %208 = invoke noundef i32 %207(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %10, ptr noundef nonnull align 8 dereferenceable(8) %11, ptr noundef nonnull align 4 dereferenceable(4) %12)
          to label %209 unwind label %219

209:                                              ; preds = %206
  call void @llvm.lifetime.end.p0(ptr nonnull %10)
  call void @llvm.lifetime.end.p0(ptr nonnull %11)
  call void @llvm.lifetime.end.p0(ptr nonnull %12)
  %210 = icmp eq i32 %200, %208
  br i1 %210, label %211, label %214

211:                                              ; preds = %209
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(4000) %57, i8 0, i64 4000, i1 false), !tbaa !33
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(4000) %58, i8 0, i64 4000, i1 false), !tbaa !33
  call void @llvm.lifetime.start.p0(ptr nonnull %7)
  call void @llvm.lifetime.start.p0(ptr nonnull %8)
  call void @llvm.lifetime.start.p0(ptr nonnull %9)
  store ptr %57, ptr %7, align 8, !tbaa !35
  store ptr %58, ptr %8, align 8, !tbaa !35
  store i32 1000, ptr %9, align 4, !tbaa !33
  %212 = load ptr, ptr %76, align 8, !tbaa !20
  %213 = icmp eq ptr %212, null
  br i1 %213, label %221, label %223

214:                                              ; preds = %209
  %215 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.11)
          to label %216 unwind label %219

216:                                              ; preds = %214
  call void @exit(i32 noundef 1) #22
  unreachable

217:                                              ; preds = %198, %196
  %218 = landingpad { ptr, i32 }
          cleanup
  br label %244

219:                                              ; preds = %206, %204, %214
  %220 = landingpad { ptr, i32 }
          cleanup
  br label %244

221:                                              ; preds = %211
  invoke void @_ZSt25__throw_bad_function_callv() #21
          to label %222 unwind label %239

222:                                              ; preds = %221
  unreachable

223:                                              ; preds = %211
  %224 = load ptr, ptr %82, align 8, !tbaa !16
  %225 = invoke noundef i32 %224(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %7, ptr noundef nonnull align 8 dereferenceable(8) %8, ptr noundef nonnull align 4 dereferenceable(4) %9)
          to label %226 unwind label %239

226:                                              ; preds = %223
  call void @llvm.lifetime.end.p0(ptr nonnull %7)
  call void @llvm.lifetime.end.p0(ptr nonnull %8)
  call void @llvm.lifetime.end.p0(ptr nonnull %9)
  call void @llvm.lifetime.start.p0(ptr nonnull %4)
  call void @llvm.lifetime.start.p0(ptr nonnull %5)
  call void @llvm.lifetime.start.p0(ptr nonnull %6)
  store ptr %57, ptr %4, align 8, !tbaa !35
  store ptr %58, ptr %5, align 8, !tbaa !35
  store i32 1000, ptr %6, align 4, !tbaa !33
  %227 = load ptr, ptr %86, align 8, !tbaa !20
  %228 = icmp eq ptr %227, null
  br i1 %228, label %229, label %231

229:                                              ; preds = %226
  invoke void @_ZSt25__throw_bad_function_callv() #21
          to label %230 unwind label %241

230:                                              ; preds = %229
  unreachable

231:                                              ; preds = %226
  %232 = load ptr, ptr %92, align 8, !tbaa !16
  %233 = invoke noundef i32 %232(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %4, ptr noundef nonnull align 8 dereferenceable(8) %5, ptr noundef nonnull align 4 dereferenceable(4) %6)
          to label %234 unwind label %241

234:                                              ; preds = %231
  call void @llvm.lifetime.end.p0(ptr nonnull %4)
  call void @llvm.lifetime.end.p0(ptr nonnull %5)
  call void @llvm.lifetime.end.p0(ptr nonnull %6)
  %235 = icmp eq i32 %225, %233
  br i1 %235, label %243, label %236

236:                                              ; preds = %234
  %237 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.11)
          to label %238 unwind label %241

238:                                              ; preds = %236
  call void @exit(i32 noundef 1) #22
  unreachable

239:                                              ; preds = %223, %221
  %240 = landingpad { ptr, i32 }
          cleanup
  br label %244

241:                                              ; preds = %231, %229, %236
  %242 = landingpad { ptr, i32 }
          cleanup
  br label %244

243:                                              ; preds = %234
  call void @_ZdaPv(ptr noundef nonnull %58) #23
  call void @_ZdaPv(ptr noundef nonnull %57) #23
  ret void

244:                                              ; preds = %102, %106, %104, %239, %241, %217, %219, %192, %194, %167, %169, %142, %144, %108, %110
  %245 = phi { ptr, i32 } [ %111, %110 ], [ %109, %108 ], [ %145, %144 ], [ %143, %142 ], [ %170, %169 ], [ %168, %167 ], [ %195, %194 ], [ %193, %192 ], [ %220, %219 ], [ %218, %217 ], [ %242, %241 ], [ %240, %239 ], [ %103, %102 ], [ %105, %104 ], [ %107, %106 ]
  call void @_ZdaPv(ptr noundef nonnull %58) #23
  br label %246

246:                                              ; preds = %244, %100
  %247 = phi { ptr, i32 } [ %245, %244 ], [ %101, %100 ]
  call void @_ZdaPv(ptr noundef nonnull %57) #23
  resume { ptr, i32 } %247
}

declare i32 @__gxx_personality_v0(...)

; Function Attrs: noinline noreturn nounwind uwtable
define linkonce_odr hidden void @__clang_call_terminate(ptr noundef %0) local_unnamed_addr #3 comdat {
  %2 = tail call ptr @__cxa_begin_catch(ptr %0) #18
  tail call void @_ZSt9terminatev() #19
  unreachable
}

declare ptr @__cxa_begin_catch(ptr) local_unnamed_addr

; Function Attrs: cold nofree noreturn
declare void @_ZSt9terminatev() local_unnamed_addr #4

; Function Attrs: inlinehint mustprogress uwtable
declare noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8), ptr noundef) local_unnamed_addr #5

; Function Attrs: nobuiltin allocsize(0)
declare noundef nonnull ptr @_Znam(i64 noundef) local_unnamed_addr #6

; Function Attrs: nofree noreturn nounwind
declare void @exit(i32 noundef) local_unnamed_addr #7

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8), ptr noundef, i64 noundef) local_unnamed_addr #8

declare void @_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate(ptr noundef nonnull align 8 dereferenceable(264), i32 noundef) local_unnamed_addr #8

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: read)
declare i64 @strlen(ptr noundef captures(none)) local_unnamed_addr #9

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local noundef i32 @_ZNSt24uniform_int_distributionIjEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEjRT_RKNS0_10param_typeE(ptr noundef nonnull align 4 dereferenceable(8) %0, ptr noundef nonnull align 8 dereferenceable(5000) %1, ptr noundef nonnull align 4 dereferenceable(8) %2) local_unnamed_addr #10 comdat {
  %4 = alloca %"struct.std::uniform_int_distribution<unsigned int>::param_type", align 8
  %5 = getelementptr inbounds nuw i8, ptr %2, i64 4
  %6 = load i32, ptr %5, align 4, !tbaa !37
  %7 = zext i32 %6 to i64
  %8 = load i32, ptr %2, align 4, !tbaa !39
  %9 = zext i32 %8 to i64
  %10 = sub nsw i64 %7, %9
  %11 = icmp ult i64 %10, 4294967295
  br i1 %11, label %12, label %32

12:                                               ; preds = %3
  %13 = trunc nuw i64 %10 to i32
  %14 = add nuw i32 %13, 1
  %15 = tail call noundef i64 @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv(ptr noundef nonnull align 8 dereferenceable(5000) %1)
  %16 = zext i32 %14 to i64
  %17 = mul i64 %15, %16
  %18 = trunc i64 %17 to i32
  %19 = icmp ult i32 %13, %18
  br i1 %19, label %29, label %20

20:                                               ; preds = %12
  %21 = xor i32 %13, -1
  %22 = urem i32 %21, %14
  %23 = icmp ugt i32 %22, %18
  br i1 %23, label %24, label %29

24:                                               ; preds = %20, %24
  %25 = tail call noundef i64 @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv(ptr noundef nonnull align 8 dereferenceable(5000) %1)
  %26 = mul i64 %25, %16
  %27 = trunc i64 %26 to i32
  %28 = icmp ugt i32 %22, %27
  br i1 %28, label %24, label %29, !llvm.loop !40

29:                                               ; preds = %24, %12, %20
  %30 = phi i64 [ %17, %12 ], [ %17, %20 ], [ %26, %24 ]
  %31 = lshr i64 %30, 32
  br label %45

32:                                               ; preds = %3
  %33 = icmp eq i64 %10, 4294967295
  br i1 %33, label %43, label %34

34:                                               ; preds = %32, %34
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #18
  store <2 x i32> <i32 0, i32 -1>, ptr %4, align 8, !tbaa !33
  %35 = call noundef i32 @_ZNSt24uniform_int_distributionIjEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEjRT_RKNS0_10param_typeE(ptr noundef nonnull align 4 dereferenceable(8) %0, ptr noundef nonnull align 8 dereferenceable(5000) %1, ptr noundef nonnull align 4 dereferenceable(8) %4)
  %36 = zext i32 %35 to i64
  %37 = shl nuw i64 %36, 32
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #18
  %38 = call noundef i64 @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv(ptr noundef nonnull align 8 dereferenceable(5000) %1)
  %39 = add i64 %37, %38
  %40 = icmp ugt i64 %39, %10
  %41 = icmp ult i64 %39, %37
  %42 = or i1 %40, %41
  br i1 %42, label %34, label %45, !llvm.loop !41

43:                                               ; preds = %32
  %44 = tail call noundef i64 @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv(ptr noundef nonnull align 8 dereferenceable(5000) %1)
  br label %45

45:                                               ; preds = %34, %43, %29
  %46 = phi i64 [ %31, %29 ], [ %44, %43 ], [ %39, %34 ]
  %47 = load i32, ptr %2, align 4, !tbaa !39
  %48 = trunc i64 %46 to i32
  %49 = add i32 %47, %48
  ret i32 %49
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local noundef i64 @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv(ptr noundef nonnull align 8 dereferenceable(5000) %0) local_unnamed_addr #10 comdat {
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 4992
  %3 = load i64, ptr %2, align 8, !tbaa !12
  %4 = icmp ugt i64 %3, 623
  br i1 %4, label %5, label %127

5:                                                ; preds = %1
  %6 = load i64, ptr %0, align 8, !tbaa !6
  %7 = insertelement <2 x i64> poison, i64 %6, i64 1
  br label %8

8:                                                ; preds = %8, %5
  %9 = phi i64 [ 0, %5 ], [ %42, %8 ]
  %10 = phi <2 x i64> [ %7, %5 ], [ %16, %8 ]
  %11 = getelementptr inbounds nuw i64, ptr %0, i64 %9
  %12 = getelementptr inbounds nuw i64, ptr %0, i64 %9
  %13 = getelementptr inbounds nuw i8, ptr %12, i64 8
  %14 = getelementptr inbounds nuw i8, ptr %12, i64 24
  %15 = load <2 x i64>, ptr %13, align 8, !tbaa !6
  %16 = load <2 x i64>, ptr %14, align 8, !tbaa !6
  %17 = shufflevector <2 x i64> %10, <2 x i64> %15, <2 x i32> <i32 1, i32 2>
  %18 = shufflevector <2 x i64> %15, <2 x i64> %16, <2 x i32> <i32 1, i32 2>
  %19 = and <2 x i64> %17, splat (i64 -2147483648)
  %20 = and <2 x i64> %18, splat (i64 -2147483648)
  %21 = and <2 x i64> %15, splat (i64 2147483646)
  %22 = and <2 x i64> %16, splat (i64 2147483646)
  %23 = or disjoint <2 x i64> %21, %19
  %24 = or disjoint <2 x i64> %22, %20
  %25 = getelementptr inbounds nuw i8, ptr %11, i64 3176
  %26 = getelementptr inbounds nuw i8, ptr %11, i64 3192
  %27 = load <2 x i64>, ptr %25, align 8, !tbaa !6
  %28 = load <2 x i64>, ptr %26, align 8, !tbaa !6
  %29 = lshr exact <2 x i64> %23, splat (i64 1)
  %30 = lshr exact <2 x i64> %24, splat (i64 1)
  %31 = xor <2 x i64> %29, %27
  %32 = xor <2 x i64> %30, %28
  %33 = and <2 x i64> %15, splat (i64 1)
  %34 = and <2 x i64> %16, splat (i64 1)
  %35 = icmp eq <2 x i64> %33, zeroinitializer
  %36 = icmp eq <2 x i64> %34, zeroinitializer
  %37 = select <2 x i1> %35, <2 x i64> zeroinitializer, <2 x i64> splat (i64 2567483615)
  %38 = select <2 x i1> %36, <2 x i64> zeroinitializer, <2 x i64> splat (i64 2567483615)
  %39 = xor <2 x i64> %31, %37
  %40 = xor <2 x i64> %32, %38
  %41 = getelementptr inbounds nuw i8, ptr %11, i64 16
  store <2 x i64> %39, ptr %11, align 8, !tbaa !6
  store <2 x i64> %40, ptr %41, align 8, !tbaa !6
  %42 = add nuw i64 %9, 4
  %43 = icmp eq i64 %42, 224
  br i1 %43, label %44, label %8, !llvm.loop !42

44:                                               ; preds = %8
  %45 = extractelement <2 x i64> %16, i64 1
  %46 = getelementptr inbounds nuw i8, ptr %0, i64 1792
  %47 = and i64 %45, -2147483648
  %48 = getelementptr inbounds nuw i8, ptr %0, i64 1800
  %49 = load i64, ptr %48, align 8, !tbaa !6
  %50 = and i64 %49, 2147483646
  %51 = or disjoint i64 %50, %47
  %52 = getelementptr inbounds nuw i8, ptr %0, i64 4968
  %53 = load i64, ptr %52, align 8, !tbaa !6
  %54 = lshr exact i64 %51, 1
  %55 = xor i64 %54, %53
  %56 = and i64 %49, 1
  %57 = icmp eq i64 %56, 0
  %58 = select i1 %57, i64 0, i64 2567483615
  %59 = xor i64 %55, %58
  store i64 %59, ptr %46, align 8, !tbaa !6
  %60 = getelementptr inbounds nuw i8, ptr %0, i64 1800
  %61 = and i64 %49, -2147483648
  %62 = getelementptr inbounds nuw i8, ptr %0, i64 1808
  %63 = load i64, ptr %62, align 8, !tbaa !6
  %64 = and i64 %63, 2147483646
  %65 = or disjoint i64 %64, %61
  %66 = getelementptr inbounds nuw i8, ptr %0, i64 4976
  %67 = load i64, ptr %66, align 8, !tbaa !6
  %68 = lshr exact i64 %65, 1
  %69 = xor i64 %68, %67
  %70 = and i64 %63, 1
  %71 = icmp eq i64 %70, 0
  %72 = select i1 %71, i64 0, i64 2567483615
  %73 = xor i64 %69, %72
  store i64 %73, ptr %60, align 8, !tbaa !6
  %74 = getelementptr inbounds nuw i8, ptr %0, i64 1808
  %75 = and i64 %63, -2147483648
  %76 = getelementptr inbounds nuw i8, ptr %0, i64 1816
  %77 = load i64, ptr %76, align 8, !tbaa !6
  %78 = and i64 %77, 2147483646
  %79 = or disjoint i64 %78, %75
  %80 = getelementptr inbounds nuw i8, ptr %0, i64 4984
  %81 = load i64, ptr %80, align 8, !tbaa !6
  %82 = lshr exact i64 %79, 1
  %83 = xor i64 %82, %81
  %84 = and i64 %77, 1
  %85 = icmp eq i64 %84, 0
  %86 = select i1 %85, i64 0, i64 2567483615
  %87 = xor i64 %83, %86
  store i64 %87, ptr %74, align 8, !tbaa !6
  %88 = getelementptr inbounds nuw i8, ptr %0, i64 1816
  %89 = load i64, ptr %88, align 8, !tbaa !6
  %90 = insertelement <2 x i64> poison, i64 %89, i64 1
  br label %91

91:                                               ; preds = %91, %44
  %92 = phi i64 [ 0, %44 ], [ %110, %91 ]
  %93 = phi <2 x i64> [ %90, %44 ], [ %98, %91 ]
  %94 = getelementptr i64, ptr %0, i64 %92
  %95 = getelementptr i8, ptr %94, i64 1816
  %96 = getelementptr i64, ptr %0, i64 %92
  %97 = getelementptr i8, ptr %96, i64 1824
  %98 = load <2 x i64>, ptr %97, align 8, !tbaa !6
  %99 = shufflevector <2 x i64> %93, <2 x i64> %98, <2 x i32> <i32 1, i32 2>
  %100 = and <2 x i64> %99, splat (i64 -2147483648)
  %101 = and <2 x i64> %98, splat (i64 2147483646)
  %102 = or disjoint <2 x i64> %101, %100
  %103 = load <2 x i64>, ptr %94, align 8, !tbaa !6
  %104 = lshr exact <2 x i64> %102, splat (i64 1)
  %105 = xor <2 x i64> %104, %103
  %106 = and <2 x i64> %98, splat (i64 1)
  %107 = icmp eq <2 x i64> %106, zeroinitializer
  %108 = select <2 x i1> %107, <2 x i64> zeroinitializer, <2 x i64> splat (i64 2567483615)
  %109 = xor <2 x i64> %105, %108
  store <2 x i64> %109, ptr %95, align 8, !tbaa !6
  %110 = add nuw i64 %92, 2
  %111 = icmp eq i64 %110, 396
  br i1 %111, label %112, label %91, !llvm.loop !45

112:                                              ; preds = %91
  %113 = getelementptr inbounds nuw i8, ptr %0, i64 4984
  %114 = load i64, ptr %113, align 8, !tbaa !6
  %115 = and i64 %114, -2147483648
  %116 = load i64, ptr %0, align 8, !tbaa !6
  %117 = and i64 %116, 2147483646
  %118 = or disjoint i64 %117, %115
  %119 = getelementptr inbounds nuw i8, ptr %0, i64 3168
  %120 = load i64, ptr %119, align 8, !tbaa !6
  %121 = lshr exact i64 %118, 1
  %122 = xor i64 %121, %120
  %123 = and i64 %116, 1
  %124 = icmp eq i64 %123, 0
  %125 = select i1 %124, i64 0, i64 2567483615
  %126 = xor i64 %122, %125
  store i64 %126, ptr %113, align 8, !tbaa !6
  br label %127

127:                                              ; preds = %112, %1
  %128 = phi i64 [ 0, %112 ], [ %3, %1 ]
  %129 = add nuw nsw i64 %128, 1
  store i64 %129, ptr %2, align 8, !tbaa !12
  %130 = getelementptr inbounds nuw i64, ptr %0, i64 %128
  %131 = load i64, ptr %130, align 8, !tbaa !6
  %132 = lshr i64 %131, 11
  %133 = and i64 %132, 4294967295
  %134 = xor i64 %133, %131
  %135 = shl i64 %134, 7
  %136 = and i64 %135, 2636928640
  %137 = xor i64 %136, %134
  %138 = shl i64 %137, 15
  %139 = and i64 %138, 4022730752
  %140 = xor i64 %139, %137
  %141 = lshr i64 %140, 18
  %142 = xor i64 %141, %140
  ret i64 %142
}

; Function Attrs: cold noreturn
declare void @_ZSt25__throw_bad_function_callv() local_unnamed_addr #11

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZSt16__introsort_loopIPjlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_(ptr noundef %0, ptr noundef %1, i64 noundef %2, i8 %3) local_unnamed_addr #10 comdat {
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
  call void @_ZSt11__make_heapIPjN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_RT0_(ptr noundef %0, ptr noundef %14, ptr noundef nonnull align 1 dereferenceable(1) %5)
  call void @llvm.lifetime.end.p0(ptr nonnull %5)
  br label %18

18:                                               ; preds = %17, %73
  %19 = phi ptr [ %20, %73 ], [ %14, %17 ]
  %20 = getelementptr inbounds i8, ptr %19, i64 -4
  %21 = load i32, ptr %20, align 4, !tbaa !33
  %22 = load i32, ptr %0, align 4, !tbaa !33
  store i32 %22, ptr %20, align 4, !tbaa !33
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
  %33 = getelementptr inbounds i32, ptr %0, i64 %32
  %34 = getelementptr i32, ptr %0, i64 %31
  %35 = getelementptr i8, ptr %34, i64 4
  %36 = load i32, ptr %33, align 4, !tbaa !33
  %37 = load i32, ptr %35, align 4, !tbaa !33
  %38 = icmp ult i32 %36, %37
  %39 = or disjoint i64 %31, 1
  %40 = select i1 %38, i64 %39, i64 %32
  %41 = getelementptr inbounds i32, ptr %0, i64 %40
  %42 = load i32, ptr %41, align 4, !tbaa !33
  %43 = getelementptr inbounds i32, ptr %0, i64 %30
  store i32 %42, ptr %43, align 4, !tbaa !33
  %44 = icmp slt i64 %40, %27
  br i1 %44, label %29, label %45, !llvm.loop !46

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
  %56 = getelementptr inbounds nuw i32, ptr %0, i64 %55
  %57 = load i32, ptr %56, align 4, !tbaa !33
  %58 = getelementptr inbounds i32, ptr %0, i64 %46
  store i32 %57, ptr %58, align 4, !tbaa !33
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
  %67 = getelementptr inbounds nuw i32, ptr %0, i64 %66
  %68 = load i32, ptr %67, align 4, !tbaa !33
  %69 = icmp ult i32 %68, %21
  br i1 %69, label %70, label %73

70:                                               ; preds = %63
  %71 = getelementptr inbounds i32, ptr %0, i64 %64
  store i32 %68, ptr %71, align 4, !tbaa !33
  %72 = icmp ult i64 %65, 2
  br i1 %72, label %73, label %63, !llvm.loop !47

73:                                               ; preds = %70, %63, %59
  %74 = phi i64 [ 0, %59 ], [ %64, %63 ], [ 0, %70 ]
  %75 = getelementptr inbounds i32, ptr %0, i64 %74
  store i32 %21, ptr %75, align 4, !tbaa !33
  %76 = icmp sgt i64 %24, 4
  br i1 %76, label %18, label %126, !llvm.loop !48

77:                                               ; preds = %12
  %78 = add nsw i64 %15, -1
  %79 = lshr i64 %13, 3
  %80 = getelementptr inbounds nuw i32, ptr %0, i64 %79
  %81 = getelementptr inbounds i8, ptr %14, i64 -4
  %82 = load i32, ptr %11, align 4, !tbaa !33
  %83 = load i32, ptr %80, align 4, !tbaa !33
  %84 = icmp ult i32 %82, %83
  %85 = load i32, ptr %81, align 4, !tbaa !33
  br i1 %84, label %86, label %95

86:                                               ; preds = %77
  %87 = icmp ult i32 %83, %85
  br i1 %87, label %88, label %90

88:                                               ; preds = %86
  %89 = load i32, ptr %0, align 4, !tbaa !33
  store i32 %83, ptr %0, align 4, !tbaa !33
  store i32 %89, ptr %80, align 4, !tbaa !33
  br label %104

90:                                               ; preds = %86
  %91 = icmp ult i32 %82, %85
  %92 = load i32, ptr %0, align 4, !tbaa !33
  br i1 %91, label %93, label %94

93:                                               ; preds = %90
  store i32 %85, ptr %0, align 4, !tbaa !33
  store i32 %92, ptr %81, align 4, !tbaa !33
  br label %104

94:                                               ; preds = %90
  store i32 %82, ptr %0, align 4, !tbaa !33
  store i32 %92, ptr %11, align 4, !tbaa !33
  br label %104

95:                                               ; preds = %77
  %96 = icmp ult i32 %82, %85
  br i1 %96, label %97, label %99

97:                                               ; preds = %95
  %98 = load i32, ptr %0, align 4, !tbaa !33
  store i32 %82, ptr %0, align 4, !tbaa !33
  store i32 %98, ptr %11, align 4, !tbaa !33
  br label %104

99:                                               ; preds = %95
  %100 = icmp ult i32 %83, %85
  %101 = load i32, ptr %0, align 4, !tbaa !33
  br i1 %100, label %102, label %103

102:                                              ; preds = %99
  store i32 %85, ptr %0, align 4, !tbaa !33
  store i32 %101, ptr %81, align 4, !tbaa !33
  br label %104

103:                                              ; preds = %99
  store i32 %83, ptr %0, align 4, !tbaa !33
  store i32 %101, ptr %80, align 4, !tbaa !33
  br label %104

104:                                              ; preds = %103, %102, %97, %94, %93, %88
  br label %105

105:                                              ; preds = %104, %121
  %106 = phi ptr [ %116, %121 ], [ %14, %104 ]
  %107 = phi ptr [ %113, %121 ], [ %11, %104 ]
  %108 = load i32, ptr %0, align 4, !tbaa !33
  br label %109

109:                                              ; preds = %109, %105
  %110 = phi ptr [ %107, %105 ], [ %113, %109 ]
  %111 = load i32, ptr %110, align 4, !tbaa !33
  %112 = icmp ult i32 %111, %108
  %113 = getelementptr inbounds nuw i8, ptr %110, i64 4
  br i1 %112, label %109, label %114, !llvm.loop !49

114:                                              ; preds = %109, %114
  %115 = phi ptr [ %116, %114 ], [ %106, %109 ]
  %116 = getelementptr inbounds i8, ptr %115, i64 -4
  %117 = load i32, ptr %116, align 4, !tbaa !33
  %118 = icmp ult i32 %108, %117
  br i1 %118, label %114, label %119, !llvm.loop !50

119:                                              ; preds = %114
  %120 = icmp ult ptr %110, %116
  br i1 %120, label %121, label %122

121:                                              ; preds = %119
  store i32 %117, ptr %110, align 4, !tbaa !33
  store i32 %111, ptr %116, align 4, !tbaa !33
  br label %105, !llvm.loop !51

122:                                              ; preds = %119
  tail call void @_ZSt16__introsort_loopIPjlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_(ptr noundef nonnull %110, ptr noundef %14, i64 noundef %78, i8 undef)
  %123 = ptrtoint ptr %110 to i64
  %124 = sub i64 %123, %6
  %125 = icmp sgt i64 %124, 64
  br i1 %125, label %12, label %126, !llvm.loop !52

126:                                              ; preds = %122, %73, %4
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZSt22__final_insertion_sortIPjN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_(ptr noundef %0, ptr noundef %1, i8 %2) local_unnamed_addr #10 comdat {
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
  %14 = load i32, ptr %13, align 4, !tbaa !33
  %15 = load i32, ptr %0, align 4, !tbaa !33
  %16 = icmp ult i32 %14, %15
  br i1 %16, label %17, label %22

17:                                               ; preds = %10
  %18 = icmp samesign ugt i64 %11, 4
  br i1 %18, label %19, label %20, !prof !53

19:                                               ; preds = %17
  tail call void @llvm.memmove.p0.p0.i64(ptr noundef nonnull align 4 dereferenceable(1) %9, ptr noundef nonnull align 4 dereferenceable(1) %0, i64 %11, i1 false)
  br label %32

20:                                               ; preds = %17
  %21 = getelementptr inbounds nuw i8, ptr %12, i64 4
  store i32 %15, ptr %21, align 4, !tbaa !33
  br label %32

22:                                               ; preds = %10
  %23 = load i32, ptr %12, align 4, !tbaa !33
  %24 = icmp ult i32 %14, %23
  br i1 %24, label %25, label %32

25:                                               ; preds = %22, %25
  %26 = phi i32 [ %30, %25 ], [ %23, %22 ]
  %27 = phi ptr [ %29, %25 ], [ %12, %22 ]
  %28 = phi ptr [ %27, %25 ], [ %13, %22 ]
  store i32 %26, ptr %28, align 4, !tbaa !33
  %29 = getelementptr inbounds i8, ptr %27, i64 -4
  %30 = load i32, ptr %29, align 4, !tbaa !33
  %31 = icmp ult i32 %14, %30
  br i1 %31, label %25, label %32, !llvm.loop !54

32:                                               ; preds = %25, %22, %20, %19
  %33 = phi ptr [ %0, %19 ], [ %0, %20 ], [ %13, %22 ], [ %27, %25 ]
  store i32 %14, ptr %33, align 4, !tbaa !33
  %34 = add nuw nsw i64 %11, 4
  %35 = icmp eq i64 %34, 64
  br i1 %35, label %36, label %10, !llvm.loop !55

36:                                               ; preds = %32
  %37 = getelementptr inbounds nuw i8, ptr %0, i64 64
  %38 = icmp eq ptr %37, %1
  br i1 %38, label %94, label %39

39:                                               ; preds = %36, %52
  %40 = phi ptr [ %54, %52 ], [ %37, %36 ]
  %41 = load i32, ptr %40, align 4, !tbaa !33
  %42 = getelementptr inbounds i8, ptr %40, i64 -4
  %43 = load i32, ptr %42, align 4, !tbaa !33
  %44 = icmp ult i32 %41, %43
  br i1 %44, label %45, label %52

45:                                               ; preds = %39, %45
  %46 = phi i32 [ %50, %45 ], [ %43, %39 ]
  %47 = phi ptr [ %49, %45 ], [ %42, %39 ]
  %48 = phi ptr [ %47, %45 ], [ %40, %39 ]
  store i32 %46, ptr %48, align 4, !tbaa !33
  %49 = getelementptr inbounds i8, ptr %47, i64 -4
  %50 = load i32, ptr %49, align 4, !tbaa !33
  %51 = icmp ult i32 %41, %50
  br i1 %51, label %45, label %52, !llvm.loop !54

52:                                               ; preds = %45, %39
  %53 = phi ptr [ %40, %39 ], [ %47, %45 ]
  store i32 %41, ptr %53, align 4, !tbaa !33
  %54 = getelementptr inbounds nuw i8, ptr %40, i64 4
  %55 = icmp eq ptr %54, %1
  br i1 %55, label %94, label %39, !llvm.loop !56

56:                                               ; preds = %3
  %57 = icmp eq ptr %0, %1
  %58 = getelementptr inbounds nuw i8, ptr %0, i64 4
  %59 = icmp eq ptr %58, %1
  %60 = select i1 %57, i1 true, i1 %59
  br i1 %60, label %94, label %61

61:                                               ; preds = %56, %90
  %62 = phi ptr [ %92, %90 ], [ %58, %56 ]
  %63 = phi ptr [ %62, %90 ], [ %0, %56 ]
  %64 = load i32, ptr %62, align 4, !tbaa !33
  %65 = load i32, ptr %0, align 4, !tbaa !33
  %66 = icmp ult i32 %64, %65
  br i1 %66, label %67, label %80

67:                                               ; preds = %61
  %68 = ptrtoint ptr %62 to i64
  %69 = sub i64 %68, %5
  %70 = ashr exact i64 %69, 2
  %71 = icmp sgt i64 %70, 1
  br i1 %71, label %72, label %76, !prof !53

72:                                               ; preds = %67
  %73 = getelementptr inbounds nuw i8, ptr %63, i64 8
  %74 = sub nsw i64 0, %70
  %75 = getelementptr inbounds i32, ptr %73, i64 %74
  tail call void @llvm.memmove.p0.p0.i64(ptr noundef nonnull align 4 dereferenceable(1) %75, ptr noundef nonnull align 4 dereferenceable(1) %0, i64 %69, i1 false)
  br label %90

76:                                               ; preds = %67
  %77 = icmp eq i64 %69, 4
  br i1 %77, label %78, label %90

78:                                               ; preds = %76
  %79 = getelementptr inbounds nuw i8, ptr %63, i64 4
  store i32 %65, ptr %79, align 4, !tbaa !33
  br label %90

80:                                               ; preds = %61
  %81 = load i32, ptr %63, align 4, !tbaa !33
  %82 = icmp ult i32 %64, %81
  br i1 %82, label %83, label %90

83:                                               ; preds = %80, %83
  %84 = phi i32 [ %88, %83 ], [ %81, %80 ]
  %85 = phi ptr [ %87, %83 ], [ %63, %80 ]
  %86 = phi ptr [ %85, %83 ], [ %62, %80 ]
  store i32 %84, ptr %86, align 4, !tbaa !33
  %87 = getelementptr inbounds i8, ptr %85, i64 -4
  %88 = load i32, ptr %87, align 4, !tbaa !33
  %89 = icmp ult i32 %64, %88
  br i1 %89, label %83, label %90, !llvm.loop !54

90:                                               ; preds = %83, %80, %78, %76, %72
  %91 = phi ptr [ %0, %72 ], [ %0, %76 ], [ %0, %78 ], [ %62, %80 ], [ %85, %83 ]
  store i32 %64, ptr %91, align 4, !tbaa !33
  %92 = getelementptr inbounds nuw i8, ptr %62, i64 4
  %93 = icmp eq ptr %92, %1
  br i1 %93, label %94, label %61, !llvm.loop !55

94:                                               ; preds = %90, %52, %56, %36
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZSt11__make_heapIPjN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_RT0_(ptr noundef %0, ptr noundef %1, ptr noundef nonnull align 1 dereferenceable(1) %2) local_unnamed_addr #10 comdat {
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
  %19 = getelementptr inbounds nuw i32, ptr %0, i64 %18
  %20 = getelementptr inbounds nuw i32, ptr %0, i64 %16
  br label %59

21:                                               ; preds = %9, %54
  %22 = phi i64 [ %58, %54 ], [ %11, %9 ]
  %23 = getelementptr inbounds nuw i32, ptr %0, i64 %22
  %24 = load i32, ptr %23, align 4, !tbaa !33
  %25 = icmp slt i64 %22, %13
  br i1 %25, label %26, label %54

26:                                               ; preds = %21, %26
  %27 = phi i64 [ %37, %26 ], [ %22, %21 ]
  %28 = shl i64 %27, 1
  %29 = add i64 %28, 2
  %30 = getelementptr inbounds i32, ptr %0, i64 %29
  %31 = getelementptr i32, ptr %0, i64 %28
  %32 = getelementptr i8, ptr %31, i64 4
  %33 = load i32, ptr %30, align 4, !tbaa !33
  %34 = load i32, ptr %32, align 4, !tbaa !33
  %35 = icmp ult i32 %33, %34
  %36 = or disjoint i64 %28, 1
  %37 = select i1 %35, i64 %36, i64 %29
  %38 = getelementptr inbounds i32, ptr %0, i64 %37
  %39 = load i32, ptr %38, align 4, !tbaa !33
  %40 = getelementptr inbounds i32, ptr %0, i64 %27
  store i32 %39, ptr %40, align 4, !tbaa !33
  %41 = icmp slt i64 %37, %13
  br i1 %41, label %26, label %42, !llvm.loop !46

42:                                               ; preds = %26
  %43 = icmp sgt i64 %37, %22
  br i1 %43, label %44, label %54

44:                                               ; preds = %42, %51
  %45 = phi i64 [ %47, %51 ], [ %37, %42 ]
  %46 = add nsw i64 %45, -1
  %47 = sdiv i64 %46, 2
  %48 = getelementptr inbounds nuw i32, ptr %0, i64 %47
  %49 = load i32, ptr %48, align 4, !tbaa !33
  %50 = icmp ult i32 %49, %24
  br i1 %50, label %51, label %54

51:                                               ; preds = %44
  %52 = getelementptr inbounds nuw i32, ptr %0, i64 %45
  store i32 %49, ptr %52, align 4, !tbaa !33
  %53 = icmp sgt i64 %47, %22
  br i1 %53, label %44, label %54, !llvm.loop !47

54:                                               ; preds = %44, %51, %21, %42
  %55 = phi i64 [ %37, %42 ], [ %22, %21 ], [ %47, %51 ], [ %45, %44 ]
  %56 = getelementptr inbounds nuw i32, ptr %0, i64 %55
  store i32 %24, ptr %56, align 4, !tbaa !33
  %57 = icmp eq i64 %22, 0
  %58 = add nsw i64 %22, -1
  br i1 %57, label %103, label %21, !llvm.loop !57

59:                                               ; preds = %17, %98
  %60 = phi i64 [ %102, %98 ], [ %11, %17 ]
  %61 = getelementptr inbounds nuw i32, ptr %0, i64 %60
  %62 = load i32, ptr %61, align 4, !tbaa !33
  %63 = icmp slt i64 %60, %13
  br i1 %63, label %64, label %80

64:                                               ; preds = %59, %64
  %65 = phi i64 [ %75, %64 ], [ %60, %59 ]
  %66 = shl i64 %65, 1
  %67 = add i64 %66, 2
  %68 = getelementptr inbounds i32, ptr %0, i64 %67
  %69 = getelementptr i32, ptr %0, i64 %66
  %70 = getelementptr i8, ptr %69, i64 4
  %71 = load i32, ptr %68, align 4, !tbaa !33
  %72 = load i32, ptr %70, align 4, !tbaa !33
  %73 = icmp ult i32 %71, %72
  %74 = or disjoint i64 %66, 1
  %75 = select i1 %73, i64 %74, i64 %67
  %76 = getelementptr inbounds i32, ptr %0, i64 %75
  %77 = load i32, ptr %76, align 4, !tbaa !33
  %78 = getelementptr inbounds i32, ptr %0, i64 %65
  store i32 %77, ptr %78, align 4, !tbaa !33
  %79 = icmp slt i64 %75, %13
  br i1 %79, label %64, label %80, !llvm.loop !46

80:                                               ; preds = %64, %59
  %81 = phi i64 [ %60, %59 ], [ %75, %64 ]
  %82 = icmp eq i64 %81, %16
  br i1 %82, label %83, label %85

83:                                               ; preds = %80
  %84 = load i32, ptr %19, align 4, !tbaa !33
  store i32 %84, ptr %20, align 4, !tbaa !33
  br label %85

85:                                               ; preds = %83, %80
  %86 = phi i64 [ %18, %83 ], [ %81, %80 ]
  %87 = icmp sgt i64 %86, %60
  br i1 %87, label %88, label %98

88:                                               ; preds = %85, %95
  %89 = phi i64 [ %91, %95 ], [ %86, %85 ]
  %90 = add nsw i64 %89, -1
  %91 = sdiv i64 %90, 2
  %92 = getelementptr inbounds nuw i32, ptr %0, i64 %91
  %93 = load i32, ptr %92, align 4, !tbaa !33
  %94 = icmp ult i32 %93, %62
  br i1 %94, label %95, label %98

95:                                               ; preds = %88
  %96 = getelementptr inbounds nuw i32, ptr %0, i64 %89
  store i32 %93, ptr %96, align 4, !tbaa !33
  %97 = icmp sgt i64 %91, %60
  br i1 %97, label %88, label %98, !llvm.loop !47

98:                                               ; preds = %88, %95, %85
  %99 = phi i64 [ %86, %85 ], [ %91, %95 ], [ %89, %88 ]
  %100 = getelementptr inbounds nuw i32, ptr %0, i64 %99
  store i32 %62, ptr %100, align 4, !tbaa !33
  %101 = icmp eq i64 %60, 0
  %102 = add nsw i64 %60, -1
  br i1 %101, label %103, label %59, !llvm.loop !57

103:                                              ; preds = %54, %98, %3
  ret void
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memmove.p0.p0.i64(ptr writeonly captures(none), ptr readonly captures(none), i64, i1 immarg) #2

; Function Attrs: nobuiltin nounwind
declare void @_ZdaPv(ptr noundef) local_unnamed_addr #12

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #13

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i32 @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE3$_0E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !35
  %6 = load ptr, ptr %2, align 8, !tbaa !35
  %7 = load i32, ptr %3, align 4, !tbaa !33
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %26, label %9

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 0, %9 ], [ %24, %11 ]
  %13 = phi i32 [ -1, %9 ], [ %23, %11 ]
  %14 = phi i32 [ 0, %9 ], [ %22, %11 ]
  %15 = getelementptr inbounds nuw i32, ptr %5, i64 %12
  %16 = load i32, ptr %15, align 4, !tbaa !33
  %17 = getelementptr inbounds nuw i32, ptr %6, i64 %12
  %18 = load i32, ptr %17, align 4, !tbaa !33
  %19 = add i32 %18, %16
  %20 = icmp ult i32 %19, %13
  %21 = trunc nuw i64 %12 to i32
  %22 = select i1 %20, i32 %21, i32 %14
  %23 = tail call i32 @llvm.umin.i32(i32 %19, i32 %13)
  %24 = add nuw nsw i64 %12, 1
  %25 = icmp eq i64 %24, %10
  br i1 %25, label %26, label %11, !llvm.loop !58

26:                                               ; preds = %11, %4
  %27 = phi i32 [ 0, %4 ], [ %22, %11 ]
  ret i32 %27
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE3$_0E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_0", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !61
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i32 @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE3$_1E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !35
  %6 = load ptr, ptr %2, align 8, !tbaa !35
  %7 = load i32, ptr %3, align 4, !tbaa !33
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %26, label %9

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 0, %9 ], [ %24, %11 ]
  %13 = phi i32 [ -1, %9 ], [ %23, %11 ]
  %14 = phi i32 [ 0, %9 ], [ %22, %11 ]
  %15 = getelementptr inbounds nuw i32, ptr %5, i64 %12
  %16 = load i32, ptr %15, align 4, !tbaa !33
  %17 = getelementptr inbounds nuw i32, ptr %6, i64 %12
  %18 = load i32, ptr %17, align 4, !tbaa !33
  %19 = add i32 %18, %16
  %20 = icmp ult i32 %19, %13
  %21 = trunc nuw i64 %12 to i32
  %22 = select i1 %20, i32 %21, i32 %14
  %23 = tail call i32 @llvm.umin.i32(i32 %19, i32 %13)
  %24 = add nuw nsw i64 %12, 1
  %25 = icmp eq i64 %24, %10
  br i1 %25, label %26, label %11, !llvm.loop !62

26:                                               ; preds = %11, %4
  %27 = phi i32 [ 0, %4 ], [ %22, %11 ]
  ret i32 %27
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE3$_1E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_1", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !61
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i32 @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE3$_2E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !35
  %6 = load ptr, ptr %2, align 8, !tbaa !35
  %7 = load i32, ptr %3, align 4, !tbaa !33
  %8 = icmp ult i32 %7, 2
  br i1 %8, label %27, label %9

9:                                                ; preds = %4
  %10 = lshr i32 %7, 1
  %11 = zext nneg i32 %10 to i64
  br label %12

12:                                               ; preds = %12, %9
  %13 = phi i64 [ 0, %9 ], [ %25, %12 ]
  %14 = phi i32 [ -1, %9 ], [ %24, %12 ]
  %15 = phi i32 [ 0, %9 ], [ %23, %12 ]
  %16 = getelementptr inbounds nuw i32, ptr %5, i64 %13
  %17 = load i32, ptr %16, align 4, !tbaa !33
  %18 = getelementptr inbounds nuw i32, ptr %6, i64 %13
  %19 = load i32, ptr %18, align 4, !tbaa !33
  %20 = add i32 %19, %17
  %21 = icmp ult i32 %20, %14
  %22 = trunc nuw nsw i64 %13 to i32
  %23 = select i1 %21, i32 %22, i32 %15
  %24 = tail call i32 @llvm.umin.i32(i32 %20, i32 %14)
  %25 = add nuw nsw i64 %13, 2
  %26 = icmp samesign ult i64 %25, %11
  br i1 %26, label %12, label %27, !llvm.loop !64

27:                                               ; preds = %12, %4
  %28 = phi i32 [ 0, %4 ], [ %23, %12 ]
  ret i32 %28
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE3$_2E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_2", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !61
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i32 @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE3$_3E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !35
  %6 = load ptr, ptr %2, align 8, !tbaa !35
  %7 = load i32, ptr %3, align 4, !tbaa !33
  %8 = icmp ult i32 %7, 2
  br i1 %8, label %27, label %9

9:                                                ; preds = %4
  %10 = lshr i32 %7, 1
  %11 = zext nneg i32 %10 to i64
  br label %12

12:                                               ; preds = %12, %9
  %13 = phi i64 [ 0, %9 ], [ %25, %12 ]
  %14 = phi i32 [ -1, %9 ], [ %24, %12 ]
  %15 = phi i32 [ 0, %9 ], [ %23, %12 ]
  %16 = getelementptr inbounds nuw i32, ptr %5, i64 %13
  %17 = load i32, ptr %16, align 4, !tbaa !33
  %18 = getelementptr inbounds nuw i32, ptr %6, i64 %13
  %19 = load i32, ptr %18, align 4, !tbaa !33
  %20 = add i32 %19, %17
  %21 = icmp ult i32 %20, %14
  %22 = trunc nuw nsw i64 %13 to i32
  %23 = select i1 %21, i32 %22, i32 %15
  %24 = tail call i32 @llvm.umin.i32(i32 %20, i32 %14)
  %25 = add nuw nsw i64 %13, 2
  %26 = icmp samesign ult i64 %25, %11
  br i1 %26, label %12, label %27, !llvm.loop !65

27:                                               ; preds = %12, %4
  %28 = phi i32 [ 0, %4 ], [ %23, %12 ]
  ret i32 %28
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE3$_3E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_3", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !61
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i32 @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE3$_4E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !35
  %6 = load ptr, ptr %2, align 8, !tbaa !35
  %7 = load i32, ptr %3, align 4, !tbaa !33
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %26, label %9

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 0, %9 ], [ %24, %11 ]
  %13 = phi i32 [ -1, %9 ], [ %23, %11 ]
  %14 = phi i32 [ 2, %9 ], [ %22, %11 ]
  %15 = getelementptr inbounds nuw i32, ptr %5, i64 %12
  %16 = load i32, ptr %15, align 4, !tbaa !33
  %17 = getelementptr inbounds nuw i32, ptr %6, i64 %12
  %18 = load i32, ptr %17, align 4, !tbaa !33
  %19 = add i32 %18, %16
  %20 = icmp ult i32 %19, %13
  %21 = trunc nuw i64 %12 to i32
  %22 = select i1 %20, i32 %21, i32 %14
  %23 = tail call i32 @llvm.umin.i32(i32 %19, i32 %13)
  %24 = add nuw nsw i64 %12, 1
  %25 = icmp eq i64 %24, %10
  br i1 %25, label %26, label %11, !llvm.loop !66

26:                                               ; preds = %11, %4
  %27 = phi i32 [ 2, %4 ], [ %22, %11 ]
  ret i32 %27
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE3$_4E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_4", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !61
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i32 @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE3$_5E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !35
  %6 = load ptr, ptr %2, align 8, !tbaa !35
  %7 = load i32, ptr %3, align 4, !tbaa !33
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %26, label %9

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 0, %9 ], [ %24, %11 ]
  %13 = phi i32 [ -1, %9 ], [ %23, %11 ]
  %14 = phi i32 [ 2, %9 ], [ %22, %11 ]
  %15 = getelementptr inbounds nuw i32, ptr %5, i64 %12
  %16 = load i32, ptr %15, align 4, !tbaa !33
  %17 = getelementptr inbounds nuw i32, ptr %6, i64 %12
  %18 = load i32, ptr %17, align 4, !tbaa !33
  %19 = add i32 %18, %16
  %20 = icmp ult i32 %19, %13
  %21 = trunc nuw i64 %12 to i32
  %22 = select i1 %20, i32 %21, i32 %14
  %23 = tail call i32 @llvm.umin.i32(i32 %19, i32 %13)
  %24 = add nuw nsw i64 %12, 1
  %25 = icmp eq i64 %24, %10
  br i1 %25, label %26, label %11, !llvm.loop !67

26:                                               ; preds = %11, %4
  %27 = phi i32 [ 2, %4 ], [ %22, %11 ]
  ret i32 %27
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE3$_5E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_5", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !61
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i32 @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE3$_6E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !35
  %6 = load ptr, ptr %2, align 8, !tbaa !35
  %7 = load i32, ptr %3, align 4, !tbaa !33
  %8 = zext i32 %7 to i64
  %9 = icmp eq i32 %7, 0
  br i1 %9, label %25, label %10

10:                                               ; preds = %4, %10
  %11 = phi i32 [ %22, %10 ], [ -1, %4 ]
  %12 = phi i64 [ %23, %10 ], [ 0, %4 ]
  %13 = phi i32 [ %21, %10 ], [ 0, %4 ]
  %14 = getelementptr inbounds nuw i32, ptr %5, i64 %12
  %15 = load i32, ptr %14, align 4, !tbaa !33
  %16 = getelementptr inbounds nuw i32, ptr %6, i64 %12
  %17 = load i32, ptr %16, align 4, !tbaa !33
  %18 = add i32 %17, %15
  %19 = icmp ult i32 %18, %11
  %20 = trunc nuw i64 %12 to i32
  %21 = select i1 %19, i32 %20, i32 %13
  %22 = tail call i32 @llvm.umin.i32(i32 %18, i32 %11)
  %23 = add nuw nsw i64 %12, 1
  %24 = icmp eq i64 %23, %8
  br i1 %24, label %25, label %10, !llvm.loop !68

25:                                               ; preds = %10, %4
  %26 = phi i32 [ 0, %4 ], [ %21, %10 ]
  ret i32 %26
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE3$_6E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_6", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !61
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i32 @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE3$_7E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !35
  %6 = load ptr, ptr %2, align 8, !tbaa !35
  %7 = load i32, ptr %3, align 4, !tbaa !33
  %8 = zext i32 %7 to i64
  %9 = icmp eq i32 %7, 0
  br i1 %9, label %25, label %10

10:                                               ; preds = %4, %10
  %11 = phi i32 [ %22, %10 ], [ -1, %4 ]
  %12 = phi i64 [ %23, %10 ], [ 0, %4 ]
  %13 = phi i32 [ %21, %10 ], [ 0, %4 ]
  %14 = getelementptr inbounds nuw i32, ptr %5, i64 %12
  %15 = load i32, ptr %14, align 4, !tbaa !33
  %16 = getelementptr inbounds nuw i32, ptr %6, i64 %12
  %17 = load i32, ptr %16, align 4, !tbaa !33
  %18 = add i32 %17, %15
  %19 = icmp ult i32 %18, %11
  %20 = trunc nuw i64 %12 to i32
  %21 = select i1 %19, i32 %20, i32 %13
  %22 = tail call i32 @llvm.umin.i32(i32 %18, i32 %11)
  %23 = add nuw nsw i64 %12, 1
  %24 = icmp eq i64 %23, %8
  br i1 %24, label %25, label %10, !llvm.loop !69

25:                                               ; preds = %10, %4
  %26 = phi i32 [ 0, %4 ], [ %21, %10 ]
  ret i32 %26
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE3$_7E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_7", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !61
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i32 @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE3$_8E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !35
  %6 = load ptr, ptr %2, align 8, !tbaa !35
  %7 = load i32, ptr %3, align 4, !tbaa !33
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %27, label %9

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ %10, %9 ], [ %24, %11 ]
  %13 = phi i32 [ -1, %9 ], [ %23, %11 ]
  %14 = phi i32 [ 0, %9 ], [ %22, %11 ]
  %15 = getelementptr inbounds nuw i32, ptr %5, i64 %12
  %16 = load i32, ptr %15, align 4, !tbaa !33
  %17 = getelementptr inbounds nuw i32, ptr %6, i64 %12
  %18 = load i32, ptr %17, align 4, !tbaa !33
  %19 = add i32 %18, %16
  %20 = icmp ult i32 %19, %13
  %21 = trunc nuw i64 %12 to i32
  %22 = select i1 %20, i32 %21, i32 %14
  %23 = tail call i32 @llvm.umin.i32(i32 %19, i32 %13)
  %24 = add nsw i64 %12, -1
  %25 = and i64 %24, 4294967295
  %26 = icmp eq i64 %25, 0
  br i1 %26, label %27, label %11, !llvm.loop !70

27:                                               ; preds = %11, %4
  %28 = phi i32 [ 0, %4 ], [ %22, %11 ]
  ret i32 %28
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE3$_8E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_8", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !61
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i32 @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE3$_9E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !35
  %6 = load ptr, ptr %2, align 8, !tbaa !35
  %7 = load i32, ptr %3, align 4, !tbaa !33
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %27, label %9

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ %10, %9 ], [ %24, %11 ]
  %13 = phi i32 [ -1, %9 ], [ %23, %11 ]
  %14 = phi i32 [ 0, %9 ], [ %22, %11 ]
  %15 = getelementptr inbounds nuw i32, ptr %5, i64 %12
  %16 = load i32, ptr %15, align 4, !tbaa !33
  %17 = getelementptr inbounds nuw i32, ptr %6, i64 %12
  %18 = load i32, ptr %17, align 4, !tbaa !33
  %19 = add i32 %18, %16
  %20 = icmp ult i32 %19, %13
  %21 = trunc nuw i64 %12 to i32
  %22 = select i1 %20, i32 %21, i32 %14
  %23 = tail call i32 @llvm.umin.i32(i32 %19, i32 %13)
  %24 = add nsw i64 %12, -1
  %25 = and i64 %24, 4294967295
  %26 = icmp eq i64 %25, 0
  br i1 %26, label %27, label %11, !llvm.loop !71

27:                                               ; preds = %11, %4
  %28 = phi i32 [ 0, %4 ], [ %22, %11 ]
  ret i32 %28
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE3$_9E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_9", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !61
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i32 @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE4$_10E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !35
  %6 = load ptr, ptr %2, align 8, !tbaa !35
  %7 = load i32, ptr %3, align 4, !tbaa !33
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %26, label %9

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 0, %9 ], [ %24, %11 ]
  %13 = phi i32 [ -1, %9 ], [ %23, %11 ]
  %14 = phi i32 [ -1, %9 ], [ %22, %11 ]
  %15 = getelementptr inbounds nuw i32, ptr %5, i64 %12
  %16 = load i32, ptr %15, align 4, !tbaa !33
  %17 = getelementptr inbounds nuw i32, ptr %6, i64 %12
  %18 = load i32, ptr %17, align 4, !tbaa !33
  %19 = add i32 %18, %16
  %20 = icmp ult i32 %19, %13
  %21 = trunc nuw i64 %12 to i32
  %22 = select i1 %20, i32 %21, i32 %14
  %23 = tail call i32 @llvm.umin.i32(i32 %19, i32 %13)
  %24 = add nuw nsw i64 %12, 1
  %25 = icmp eq i64 %24, %10
  br i1 %25, label %26, label %11, !llvm.loop !72

26:                                               ; preds = %11, %4
  %27 = phi i32 [ -1, %4 ], [ %22, %11 ]
  ret i32 %27
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE4$_10E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_10", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !61
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i32 @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE4$_11E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !35
  %6 = load ptr, ptr %2, align 8, !tbaa !35
  %7 = load i32, ptr %3, align 4, !tbaa !33
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %26, label %9

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 0, %9 ], [ %24, %11 ]
  %13 = phi i32 [ -1, %9 ], [ %23, %11 ]
  %14 = phi i32 [ -1, %9 ], [ %22, %11 ]
  %15 = getelementptr inbounds nuw i32, ptr %5, i64 %12
  %16 = load i32, ptr %15, align 4, !tbaa !33
  %17 = getelementptr inbounds nuw i32, ptr %6, i64 %12
  %18 = load i32, ptr %17, align 4, !tbaa !33
  %19 = add i32 %18, %16
  %20 = icmp ult i32 %19, %13
  %21 = trunc nuw i64 %12 to i32
  %22 = select i1 %20, i32 %21, i32 %14
  %23 = tail call i32 @llvm.umin.i32(i32 %19, i32 %13)
  %24 = add nuw nsw i64 %12, 1
  %25 = icmp eq i64 %24, %10
  br i1 %25, label %26, label %11, !llvm.loop !73

26:                                               ; preds = %11, %4
  %27 = phi i32 [ -1, %4 ], [ %22, %11 ]
  ret i32 %27
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE4$_11E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_11", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !61
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i32 @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE4$_12E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !35
  %6 = load ptr, ptr %2, align 8, !tbaa !35
  %7 = load i32, ptr %3, align 4, !tbaa !33
  %8 = icmp ugt i32 %7, 3
  br i1 %8, label %9, label %26

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 3, %9 ], [ %24, %11 ]
  %13 = phi i32 [ -1, %9 ], [ %23, %11 ]
  %14 = phi i32 [ 3, %9 ], [ %22, %11 ]
  %15 = getelementptr inbounds nuw i32, ptr %5, i64 %12
  %16 = load i32, ptr %15, align 4, !tbaa !33
  %17 = getelementptr inbounds nuw i32, ptr %6, i64 %12
  %18 = load i32, ptr %17, align 4, !tbaa !33
  %19 = add i32 %18, %16
  %20 = icmp ult i32 %19, %13
  %21 = trunc nuw i64 %12 to i32
  %22 = select i1 %20, i32 %21, i32 %14
  %23 = tail call i32 @llvm.umin.i32(i32 %19, i32 %13)
  %24 = add nuw nsw i64 %12, 1
  %25 = icmp eq i64 %24, %10
  br i1 %25, label %26, label %11, !llvm.loop !74

26:                                               ; preds = %11, %4
  %27 = phi i32 [ 3, %4 ], [ %22, %11 ]
  ret i32 %27
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE4$_12E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_12", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !61
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i32 @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE4$_13E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !35
  %6 = load ptr, ptr %2, align 8, !tbaa !35
  %7 = load i32, ptr %3, align 4, !tbaa !33
  %8 = icmp ugt i32 %7, 3
  br i1 %8, label %9, label %26

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 3, %9 ], [ %24, %11 ]
  %13 = phi i32 [ -1, %9 ], [ %23, %11 ]
  %14 = phi i32 [ 3, %9 ], [ %22, %11 ]
  %15 = getelementptr inbounds nuw i32, ptr %5, i64 %12
  %16 = load i32, ptr %15, align 4, !tbaa !33
  %17 = getelementptr inbounds nuw i32, ptr %6, i64 %12
  %18 = load i32, ptr %17, align 4, !tbaa !33
  %19 = add i32 %18, %16
  %20 = icmp ult i32 %19, %13
  %21 = trunc nuw i64 %12 to i32
  %22 = select i1 %20, i32 %21, i32 %14
  %23 = tail call i32 @llvm.umin.i32(i32 %19, i32 %13)
  %24 = add nuw nsw i64 %12, 1
  %25 = icmp eq i64 %24, %10
  br i1 %25, label %26, label %11, !llvm.loop !75

26:                                               ; preds = %11, %4
  %27 = phi i32 [ 3, %4 ], [ %22, %11 ]
  ret i32 %27
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE4$_13E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_13", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !61
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i32 @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE4$_14E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !35
  %6 = load ptr, ptr %2, align 8, !tbaa !35
  %7 = load i32, ptr %3, align 4, !tbaa !33
  %8 = icmp ugt i32 %7, 3
  br i1 %8, label %9, label %26

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 3, %9 ], [ %24, %11 ]
  %13 = phi i32 [ -1, %9 ], [ %23, %11 ]
  %14 = phi i32 [ 2, %9 ], [ %22, %11 ]
  %15 = getelementptr inbounds nuw i32, ptr %5, i64 %12
  %16 = load i32, ptr %15, align 4, !tbaa !33
  %17 = getelementptr inbounds nuw i32, ptr %6, i64 %12
  %18 = load i32, ptr %17, align 4, !tbaa !33
  %19 = add i32 %18, %16
  %20 = icmp ult i32 %19, %13
  %21 = trunc nuw i64 %12 to i32
  %22 = select i1 %20, i32 %21, i32 %14
  %23 = tail call i32 @llvm.umin.i32(i32 %19, i32 %13)
  %24 = add nuw nsw i64 %12, 1
  %25 = icmp eq i64 %24, %10
  br i1 %25, label %26, label %11, !llvm.loop !76

26:                                               ; preds = %11, %4
  %27 = phi i32 [ 2, %4 ], [ %22, %11 ]
  ret i32 %27
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE4$_14E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_14", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !61
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i32 @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE4$_15E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !35
  %6 = load ptr, ptr %2, align 8, !tbaa !35
  %7 = load i32, ptr %3, align 4, !tbaa !33
  %8 = icmp ugt i32 %7, 3
  br i1 %8, label %9, label %26

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 3, %9 ], [ %24, %11 ]
  %13 = phi i32 [ -1, %9 ], [ %23, %11 ]
  %14 = phi i32 [ 2, %9 ], [ %22, %11 ]
  %15 = getelementptr inbounds nuw i32, ptr %5, i64 %12
  %16 = load i32, ptr %15, align 4, !tbaa !33
  %17 = getelementptr inbounds nuw i32, ptr %6, i64 %12
  %18 = load i32, ptr %17, align 4, !tbaa !33
  %19 = add i32 %18, %16
  %20 = icmp ult i32 %19, %13
  %21 = trunc nuw i64 %12 to i32
  %22 = select i1 %20, i32 %21, i32 %14
  %23 = tail call i32 @llvm.umin.i32(i32 %19, i32 %13)
  %24 = add nuw nsw i64 %12, 1
  %25 = icmp eq i64 %24, %10
  br i1 %25, label %26, label %11, !llvm.loop !77

26:                                               ; preds = %11, %4
  %27 = phi i32 [ 2, %4 ], [ %22, %11 ]
  ret i32 %27
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE4$_15E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_15", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !61
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i32 @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE4$_16E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !35
  %6 = load ptr, ptr %2, align 8, !tbaa !35
  %7 = load i32, ptr %3, align 4, !tbaa !33
  %8 = icmp ugt i32 %7, 3
  br i1 %8, label %9, label %26

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 3, %9 ], [ %24, %11 ]
  %13 = phi i32 [ -1, %9 ], [ %23, %11 ]
  %14 = phi i32 [ 4, %9 ], [ %22, %11 ]
  %15 = getelementptr inbounds nuw i32, ptr %5, i64 %12
  %16 = load i32, ptr %15, align 4, !tbaa !33
  %17 = getelementptr inbounds nuw i32, ptr %6, i64 %12
  %18 = load i32, ptr %17, align 4, !tbaa !33
  %19 = add i32 %18, %16
  %20 = icmp ult i32 %19, %13
  %21 = trunc nuw i64 %12 to i32
  %22 = select i1 %20, i32 %21, i32 %14
  %23 = tail call i32 @llvm.umin.i32(i32 %19, i32 %13)
  %24 = add nuw nsw i64 %12, 1
  %25 = icmp eq i64 %24, %10
  br i1 %25, label %26, label %11, !llvm.loop !78

26:                                               ; preds = %11, %4
  %27 = phi i32 [ 4, %4 ], [ %22, %11 ]
  ret i32 %27
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE4$_16E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_16", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !61
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i32 @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE4$_17E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !35
  %6 = load ptr, ptr %2, align 8, !tbaa !35
  %7 = load i32, ptr %3, align 4, !tbaa !33
  %8 = icmp ugt i32 %7, 3
  br i1 %8, label %9, label %26

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 3, %9 ], [ %24, %11 ]
  %13 = phi i32 [ -1, %9 ], [ %23, %11 ]
  %14 = phi i32 [ 4, %9 ], [ %22, %11 ]
  %15 = getelementptr inbounds nuw i32, ptr %5, i64 %12
  %16 = load i32, ptr %15, align 4, !tbaa !33
  %17 = getelementptr inbounds nuw i32, ptr %6, i64 %12
  %18 = load i32, ptr %17, align 4, !tbaa !33
  %19 = add i32 %18, %16
  %20 = icmp ult i32 %19, %13
  %21 = trunc nuw i64 %12 to i32
  %22 = select i1 %20, i32 %21, i32 %14
  %23 = tail call i32 @llvm.umin.i32(i32 %19, i32 %13)
  %24 = add nuw nsw i64 %12, 1
  %25 = icmp eq i64 %24, %10
  br i1 %25, label %26, label %11, !llvm.loop !79

26:                                               ; preds = %11, %4
  %27 = phi i32 [ 4, %4 ], [ %22, %11 ]
  ret i32 %27
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE4$_17E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_17", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !61
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define internal void @_GLOBAL__sub_I_index_select.cpp() #16 section ".text.startup" {
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
declare i32 @llvm.umin.i32(i32, i32) #17

attributes #0 = { mustprogress norecurse uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { noinline noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { cold nofree noreturn }
attributes #5 = { inlinehint mustprogress uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { nobuiltin allocsize(0) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #8 = { "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #9 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: read) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #10 = { mustprogress uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #11 = { cold noreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #12 = { nobuiltin nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #13 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #14 = { mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #15 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #16 = { nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #17 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #18 = { nounwind }
attributes #19 = { noreturn nounwind }
attributes #20 = { builtin allocsize(0) }
attributes #21 = { cold noreturn }
attributes #22 = { cold noreturn nounwind }
attributes #23 = { builtin nounwind }

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
!17 = !{!"_ZTSSt8functionIFjPjS0_jEE", !18, i64 0, !19, i64 24}
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
!34 = distinct !{!34, !11}
!35 = !{!36, !36, i64 0}
!36 = !{!"p1 int", !19, i64 0}
!37 = !{!38, !29, i64 4}
!38 = !{!"_ZTSNSt24uniform_int_distributionIjE10param_typeE", !29, i64 0, !29, i64 4}
!39 = !{!38, !29, i64 0}
!40 = distinct !{!40, !11}
!41 = distinct !{!41, !11}
!42 = distinct !{!42, !11, !43, !44}
!43 = !{!"llvm.loop.isvectorized", i32 1}
!44 = !{!"llvm.loop.unroll.runtime.disable"}
!45 = distinct !{!45, !11, !43, !44}
!46 = distinct !{!46, !11}
!47 = distinct !{!47, !11}
!48 = distinct !{!48, !11}
!49 = distinct !{!49, !11}
!50 = distinct !{!50, !11}
!51 = distinct !{!51, !11}
!52 = distinct !{!52, !11}
!53 = !{!"branch_weights", !"expected", i32 2000, i32 1}
!54 = distinct !{!54, !11}
!55 = distinct !{!55, !11}
!56 = distinct !{!56, !11}
!57 = distinct !{!57, !11}
!58 = distinct !{!58, !11, !59, !60}
!59 = !{!"llvm.loop.vectorize.width", i32 1}
!60 = !{!"llvm.loop.interleave.count", i32 1}
!61 = !{!19, !19, i64 0}
!62 = distinct !{!62, !11, !63}
!63 = !{!"llvm.loop.vectorize.enable", i1 true}
!64 = distinct !{!64, !11, !59, !60}
!65 = distinct !{!65, !11, !63}
!66 = distinct !{!66, !11, !59, !60}
!67 = distinct !{!67, !11, !63}
!68 = distinct !{!68, !11, !59, !60}
!69 = distinct !{!69, !11, !63}
!70 = distinct !{!70, !11, !59, !60}
!71 = distinct !{!71, !11, !63}
!72 = distinct !{!72, !11, !59, !60}
!73 = distinct !{!73, !11, !63}
!74 = distinct !{!74, !11, !59, !60}
!75 = distinct !{!75, !11, !63}
!76 = distinct !{!76, !11, !59, !60}
!77 = distinct !{!77, !11, !63}
!78 = distinct !{!78, !11, !59, !60}
!79 = distinct !{!79, !11, !63}
