; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/Vectorizer/any-of.cpp'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/Vectorizer/any-of.cpp"
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
%"class.std::function.2" = type { %"class.std::_Function_base", ptr }
%"class.std::function.8" = type { %"class.std::_Function_base", ptr }
%"class.std::function.22" = type { %"class.std::_Function_base", ptr }
%"class.std::function.24" = type { %"class.std::_Function_base", ptr }
%"class.std::function.30" = type { %"class.std::_Function_base", ptr }
%"class.std::uniform_int_distribution" = type { %"struct.std::uniform_int_distribution<>::param_type" }
%"struct.std::uniform_int_distribution<>::param_type" = type { i32, i32 }
%"class.std::uniform_int_distribution.62" = type { %"struct.std::uniform_int_distribution<short>::param_type" }
%"struct.std::uniform_int_distribution<short>::param_type" = type { i16, i16 }
%"class.std::uniform_int_distribution.73" = type { %"struct.std::uniform_int_distribution<unsigned int>::param_type" }
%"struct.std::uniform_int_distribution<unsigned int>::param_type" = type { i32, i32 }
%"class.std::uniform_int_distribution.84" = type { %"struct.std::uniform_int_distribution<unsigned short>::param_type" }
%"struct.std::uniform_int_distribution<unsigned short>::param_type" = type { i16, i16 }

$__clang_call_terminate = comdat any

$_ZNSt24uniform_int_distributionIiEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEiRT_RKNS0_10param_typeE = comdat any

$_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv = comdat any

$_ZNSt24uniform_int_distributionIsEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEsRT_RKNS0_10param_typeE = comdat any

$_ZNSt24uniform_int_distributionIjEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEjRT_RKNS0_10param_typeE = comdat any

$_ZNSt24uniform_int_distributionItEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEtRT_RKNS0_10param_typeE = comdat any

@_ZL3rng = internal global %"class.std::mersenne_twister_engine" zeroinitializer, align 8
@.str = private unnamed_addr constant [27 x i8] c"anyof_icmp_s32_true_update\00", align 1
@.str.1 = private unnamed_addr constant [27 x i8] c"anyof_fcmp_s32_true_update\00", align 1
@.str.2 = private unnamed_addr constant [27 x i8] c"anyof_icmp_s16_true_update\00", align 1
@.str.3 = private unnamed_addr constant [28 x i8] c"anyof_icmp_s32_false_update\00", align 1
@.str.4 = private unnamed_addr constant [28 x i8] c"anyof_fcmp_s32_false_update\00", align 1
@.str.5 = private unnamed_addr constant [28 x i8] c"anyof_icmp_s16_false_update\00", align 1
@.str.6 = private unnamed_addr constant [24 x i8] c"anyof_icmp_u32_start_TC\00", align 1
@.str.7 = private unnamed_addr constant [24 x i8] c"anyof_fcmp_u32_start_TC\00", align 1
@.str.8 = private unnamed_addr constant [24 x i8] c"anyof_icmp_u16_start_TC\00", align 1
@.str.9 = private unnamed_addr constant [28 x i8] c"anyof_icmp_u32_update_by_TC\00", align 1
@.str.10 = private unnamed_addr constant [28 x i8] c"anyof_fcmp_u32_update_by_TC\00", align 1
@.str.11 = private unnamed_addr constant [28 x i8] c"anyof_icmp_u16_update_by_TC\00", align 1
@_ZSt4cout = external global %"class.std::basic_ostream", align 8
@.str.12 = private unnamed_addr constant [10 x i8] c"Checking \00", align 1
@.str.13 = private unnamed_addr constant [2 x i8] c"\0A\00", align 1
@_ZSt4cerr = external global %"class.std::basic_ostream", align 8
@.str.14 = private unnamed_addr constant [12 x i8] c"Miscompare\0A\00", align 1
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
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_any_of.cpp, ptr null }]

; Function Attrs: mustprogress norecurse uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 personality ptr @__gxx_personality_v0 {
  %1 = alloca %"class.std::mersenne_twister_engine", align 8
  %2 = alloca %"class.std::function", align 8
  %3 = alloca %"class.std::function", align 8
  %4 = alloca %"class.std::function.2", align 8
  %5 = alloca %"class.std::function.2", align 8
  %6 = alloca %"class.std::function.8", align 8
  %7 = alloca %"class.std::function.8", align 8
  %8 = alloca %"class.std::function", align 8
  %9 = alloca %"class.std::function", align 8
  %10 = alloca %"class.std::function.2", align 8
  %11 = alloca %"class.std::function.2", align 8
  %12 = alloca %"class.std::function.8", align 8
  %13 = alloca %"class.std::function.8", align 8
  %14 = alloca %"class.std::function.22", align 8
  %15 = alloca %"class.std::function.22", align 8
  %16 = alloca %"class.std::function.24", align 8
  %17 = alloca %"class.std::function.24", align 8
  %18 = alloca %"class.std::function.30", align 8
  %19 = alloca %"class.std::function.30", align 8
  %20 = alloca %"class.std::function.22", align 8
  %21 = alloca %"class.std::function.22", align 8
  %22 = alloca %"class.std::function.24", align 8
  %23 = alloca %"class.std::function.24", align 8
  %24 = alloca %"class.std::function.30", align 8
  %25 = alloca %"class.std::function.30", align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #21
  store i64 15, ptr %1, align 8, !tbaa !6
  br label %26

26:                                               ; preds = %26, %0
  %27 = phi i64 [ 15, %0 ], [ %34, %26 ]
  %28 = phi i64 [ 1, %0 ], [ %35, %26 ]
  %29 = getelementptr i64, ptr %1, i64 %28
  %30 = lshr i64 %27, 30
  %31 = xor i64 %30, %27
  %32 = mul nuw nsw i64 %31, 1812433253
  %33 = add nuw i64 %32, %28
  %34 = and i64 %33, 4294967295
  store i64 %34, ptr %29, align 8, !tbaa !6
  %35 = add nuw nsw i64 %28, 1
  %36 = icmp eq i64 %35, 624
  br i1 %36, label %37, label %26, !llvm.loop !10

37:                                               ; preds = %26
  %38 = getelementptr inbounds nuw i8, ptr %1, i64 4992
  store i64 624, ptr %38, align 8, !tbaa !12
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(5000) @_ZL3rng, ptr noundef nonnull align 8 dereferenceable(5000) %1, i64 5000, i1 false), !tbaa.struct !14
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #21
  %39 = getelementptr inbounds nuw i8, ptr %2, i64 16
  %40 = getelementptr inbounds nuw i8, ptr %2, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %2, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFiPiS0_jEZ4mainE3$_0E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %40, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFiPiS0_jEZ4mainE3$_0E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %39, align 8, !tbaa !20
  %41 = getelementptr inbounds nuw i8, ptr %3, i64 16
  %42 = getelementptr inbounds nuw i8, ptr %3, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %3, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFiPiS0_jEZ4mainE3$_1E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %42, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFiPiS0_jEZ4mainE3$_1E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %41, align 8, !tbaa !20
  invoke fastcc void @_ZL19checkVectorFunctionIiiEvSt8functionIFT_PT0_S3_jEES5_PKc(ptr noundef %2, ptr noundef %3, ptr noundef nonnull @.str)
          to label %43 unwind label %291

43:                                               ; preds = %37
  %44 = load ptr, ptr %41, align 8, !tbaa !20
  %45 = icmp eq ptr %44, null
  br i1 %45, label %51, label %46

46:                                               ; preds = %43
  %47 = invoke noundef i1 %44(ptr noundef nonnull align 8 dereferenceable(32) %3, ptr noundef nonnull align 8 dereferenceable(32) %3, i32 noundef 3)
          to label %51 unwind label %48

48:                                               ; preds = %46
  %49 = landingpad { ptr, i32 }
          catch ptr null
  %50 = extractvalue { ptr, i32 } %49, 0
  call void @__clang_call_terminate(ptr %50) #22
  unreachable

51:                                               ; preds = %43, %46
  %52 = load ptr, ptr %39, align 8, !tbaa !20
  %53 = icmp eq ptr %52, null
  br i1 %53, label %59, label %54

54:                                               ; preds = %51
  %55 = invoke noundef i1 %52(ptr noundef nonnull align 8 dereferenceable(32) %2, ptr noundef nonnull align 8 dereferenceable(32) %2, i32 noundef 3)
          to label %59 unwind label %56

56:                                               ; preds = %54
  %57 = landingpad { ptr, i32 }
          catch ptr null
  %58 = extractvalue { ptr, i32 } %57, 0
  call void @__clang_call_terminate(ptr %58) #22
  unreachable

59:                                               ; preds = %51, %54
  %60 = getelementptr inbounds nuw i8, ptr %4, i64 16
  %61 = getelementptr inbounds nuw i8, ptr %4, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %4, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFiPfS0_jEZ4mainE3$_0E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %61, align 8, !tbaa !21
  store ptr @"_ZNSt17_Function_handlerIFiPfS0_jEZ4mainE3$_0E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %60, align 8, !tbaa !20
  %62 = getelementptr inbounds nuw i8, ptr %5, i64 16
  %63 = getelementptr inbounds nuw i8, ptr %5, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %5, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFiPfS0_jEZ4mainE3$_1E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %63, align 8, !tbaa !21
  store ptr @"_ZNSt17_Function_handlerIFiPfS0_jEZ4mainE3$_1E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %62, align 8, !tbaa !20
  invoke fastcc void @_ZL19checkVectorFunctionIifEvSt8functionIFT_PT0_S3_jEES5_PKc(ptr noundef %4, ptr noundef %5, ptr noundef nonnull @.str.1)
          to label %64 unwind label %308

64:                                               ; preds = %59
  %65 = load ptr, ptr %62, align 8, !tbaa !20
  %66 = icmp eq ptr %65, null
  br i1 %66, label %72, label %67

67:                                               ; preds = %64
  %68 = invoke noundef i1 %65(ptr noundef nonnull align 8 dereferenceable(32) %5, ptr noundef nonnull align 8 dereferenceable(32) %5, i32 noundef 3)
          to label %72 unwind label %69

69:                                               ; preds = %67
  %70 = landingpad { ptr, i32 }
          catch ptr null
  %71 = extractvalue { ptr, i32 } %70, 0
  call void @__clang_call_terminate(ptr %71) #22
  unreachable

72:                                               ; preds = %64, %67
  %73 = load ptr, ptr %60, align 8, !tbaa !20
  %74 = icmp eq ptr %73, null
  br i1 %74, label %80, label %75

75:                                               ; preds = %72
  %76 = invoke noundef i1 %73(ptr noundef nonnull align 8 dereferenceable(32) %4, ptr noundef nonnull align 8 dereferenceable(32) %4, i32 noundef 3)
          to label %80 unwind label %77

77:                                               ; preds = %75
  %78 = landingpad { ptr, i32 }
          catch ptr null
  %79 = extractvalue { ptr, i32 } %78, 0
  call void @__clang_call_terminate(ptr %79) #22
  unreachable

80:                                               ; preds = %72, %75
  %81 = getelementptr inbounds nuw i8, ptr %6, i64 16
  %82 = getelementptr inbounds nuw i8, ptr %6, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %6, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFsPsS0_jEZ4mainE3$_2E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %82, align 8, !tbaa !23
  store ptr @"_ZNSt17_Function_handlerIFsPsS0_jEZ4mainE3$_2E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %81, align 8, !tbaa !20
  %83 = getelementptr inbounds nuw i8, ptr %7, i64 16
  %84 = getelementptr inbounds nuw i8, ptr %7, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %7, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFsPsS0_jEZ4mainE3$_3E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %84, align 8, !tbaa !23
  store ptr @"_ZNSt17_Function_handlerIFsPsS0_jEZ4mainE3$_3E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %83, align 8, !tbaa !20
  invoke fastcc void @_ZL19checkVectorFunctionIssEvSt8functionIFT_PT0_S3_jEES5_PKc(ptr noundef %6, ptr noundef %7, ptr noundef nonnull @.str.2)
          to label %85 unwind label %325

85:                                               ; preds = %80
  %86 = load ptr, ptr %83, align 8, !tbaa !20
  %87 = icmp eq ptr %86, null
  br i1 %87, label %93, label %88

88:                                               ; preds = %85
  %89 = invoke noundef i1 %86(ptr noundef nonnull align 8 dereferenceable(32) %7, ptr noundef nonnull align 8 dereferenceable(32) %7, i32 noundef 3)
          to label %93 unwind label %90

90:                                               ; preds = %88
  %91 = landingpad { ptr, i32 }
          catch ptr null
  %92 = extractvalue { ptr, i32 } %91, 0
  call void @__clang_call_terminate(ptr %92) #22
  unreachable

93:                                               ; preds = %85, %88
  %94 = load ptr, ptr %81, align 8, !tbaa !20
  %95 = icmp eq ptr %94, null
  br i1 %95, label %101, label %96

96:                                               ; preds = %93
  %97 = invoke noundef i1 %94(ptr noundef nonnull align 8 dereferenceable(32) %6, ptr noundef nonnull align 8 dereferenceable(32) %6, i32 noundef 3)
          to label %101 unwind label %98

98:                                               ; preds = %96
  %99 = landingpad { ptr, i32 }
          catch ptr null
  %100 = extractvalue { ptr, i32 } %99, 0
  call void @__clang_call_terminate(ptr %100) #22
  unreachable

101:                                              ; preds = %93, %96
  %102 = getelementptr inbounds nuw i8, ptr %8, i64 16
  %103 = getelementptr inbounds nuw i8, ptr %8, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %8, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFiPiS0_jEZ4mainE3$_4E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %103, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFiPiS0_jEZ4mainE3$_4E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %102, align 8, !tbaa !20
  %104 = getelementptr inbounds nuw i8, ptr %9, i64 16
  %105 = getelementptr inbounds nuw i8, ptr %9, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %9, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFiPiS0_jEZ4mainE3$_5E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %105, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFiPiS0_jEZ4mainE3$_5E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %104, align 8, !tbaa !20
  invoke fastcc void @_ZL19checkVectorFunctionIiiEvSt8functionIFT_PT0_S3_jEES5_PKc(ptr noundef %8, ptr noundef %9, ptr noundef nonnull @.str.3)
          to label %106 unwind label %342

106:                                              ; preds = %101
  %107 = load ptr, ptr %104, align 8, !tbaa !20
  %108 = icmp eq ptr %107, null
  br i1 %108, label %114, label %109

109:                                              ; preds = %106
  %110 = invoke noundef i1 %107(ptr noundef nonnull align 8 dereferenceable(32) %9, ptr noundef nonnull align 8 dereferenceable(32) %9, i32 noundef 3)
          to label %114 unwind label %111

111:                                              ; preds = %109
  %112 = landingpad { ptr, i32 }
          catch ptr null
  %113 = extractvalue { ptr, i32 } %112, 0
  call void @__clang_call_terminate(ptr %113) #22
  unreachable

114:                                              ; preds = %106, %109
  %115 = load ptr, ptr %102, align 8, !tbaa !20
  %116 = icmp eq ptr %115, null
  br i1 %116, label %122, label %117

117:                                              ; preds = %114
  %118 = invoke noundef i1 %115(ptr noundef nonnull align 8 dereferenceable(32) %8, ptr noundef nonnull align 8 dereferenceable(32) %8, i32 noundef 3)
          to label %122 unwind label %119

119:                                              ; preds = %117
  %120 = landingpad { ptr, i32 }
          catch ptr null
  %121 = extractvalue { ptr, i32 } %120, 0
  call void @__clang_call_terminate(ptr %121) #22
  unreachable

122:                                              ; preds = %114, %117
  %123 = getelementptr inbounds nuw i8, ptr %10, i64 16
  %124 = getelementptr inbounds nuw i8, ptr %10, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %10, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFiPfS0_jEZ4mainE3$_4E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %124, align 8, !tbaa !21
  store ptr @"_ZNSt17_Function_handlerIFiPfS0_jEZ4mainE3$_4E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %123, align 8, !tbaa !20
  %125 = getelementptr inbounds nuw i8, ptr %11, i64 16
  %126 = getelementptr inbounds nuw i8, ptr %11, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %11, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFiPfS0_jEZ4mainE3$_5E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %126, align 8, !tbaa !21
  store ptr @"_ZNSt17_Function_handlerIFiPfS0_jEZ4mainE3$_5E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %125, align 8, !tbaa !20
  invoke fastcc void @_ZL19checkVectorFunctionIifEvSt8functionIFT_PT0_S3_jEES5_PKc(ptr noundef %10, ptr noundef %11, ptr noundef nonnull @.str.4)
          to label %127 unwind label %359

127:                                              ; preds = %122
  %128 = load ptr, ptr %125, align 8, !tbaa !20
  %129 = icmp eq ptr %128, null
  br i1 %129, label %135, label %130

130:                                              ; preds = %127
  %131 = invoke noundef i1 %128(ptr noundef nonnull align 8 dereferenceable(32) %11, ptr noundef nonnull align 8 dereferenceable(32) %11, i32 noundef 3)
          to label %135 unwind label %132

132:                                              ; preds = %130
  %133 = landingpad { ptr, i32 }
          catch ptr null
  %134 = extractvalue { ptr, i32 } %133, 0
  call void @__clang_call_terminate(ptr %134) #22
  unreachable

135:                                              ; preds = %127, %130
  %136 = load ptr, ptr %123, align 8, !tbaa !20
  %137 = icmp eq ptr %136, null
  br i1 %137, label %143, label %138

138:                                              ; preds = %135
  %139 = invoke noundef i1 %136(ptr noundef nonnull align 8 dereferenceable(32) %10, ptr noundef nonnull align 8 dereferenceable(32) %10, i32 noundef 3)
          to label %143 unwind label %140

140:                                              ; preds = %138
  %141 = landingpad { ptr, i32 }
          catch ptr null
  %142 = extractvalue { ptr, i32 } %141, 0
  call void @__clang_call_terminate(ptr %142) #22
  unreachable

143:                                              ; preds = %135, %138
  %144 = getelementptr inbounds nuw i8, ptr %12, i64 16
  %145 = getelementptr inbounds nuw i8, ptr %12, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %12, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFsPsS0_jEZ4mainE3$_6E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %145, align 8, !tbaa !23
  store ptr @"_ZNSt17_Function_handlerIFsPsS0_jEZ4mainE3$_6E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %144, align 8, !tbaa !20
  %146 = getelementptr inbounds nuw i8, ptr %13, i64 16
  %147 = getelementptr inbounds nuw i8, ptr %13, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %13, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFsPsS0_jEZ4mainE3$_7E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %147, align 8, !tbaa !23
  store ptr @"_ZNSt17_Function_handlerIFsPsS0_jEZ4mainE3$_7E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %146, align 8, !tbaa !20
  invoke fastcc void @_ZL19checkVectorFunctionIssEvSt8functionIFT_PT0_S3_jEES5_PKc(ptr noundef %12, ptr noundef %13, ptr noundef nonnull @.str.5)
          to label %148 unwind label %376

148:                                              ; preds = %143
  %149 = load ptr, ptr %146, align 8, !tbaa !20
  %150 = icmp eq ptr %149, null
  br i1 %150, label %156, label %151

151:                                              ; preds = %148
  %152 = invoke noundef i1 %149(ptr noundef nonnull align 8 dereferenceable(32) %13, ptr noundef nonnull align 8 dereferenceable(32) %13, i32 noundef 3)
          to label %156 unwind label %153

153:                                              ; preds = %151
  %154 = landingpad { ptr, i32 }
          catch ptr null
  %155 = extractvalue { ptr, i32 } %154, 0
  call void @__clang_call_terminate(ptr %155) #22
  unreachable

156:                                              ; preds = %148, %151
  %157 = load ptr, ptr %144, align 8, !tbaa !20
  %158 = icmp eq ptr %157, null
  br i1 %158, label %164, label %159

159:                                              ; preds = %156
  %160 = invoke noundef i1 %157(ptr noundef nonnull align 8 dereferenceable(32) %12, ptr noundef nonnull align 8 dereferenceable(32) %12, i32 noundef 3)
          to label %164 unwind label %161

161:                                              ; preds = %159
  %162 = landingpad { ptr, i32 }
          catch ptr null
  %163 = extractvalue { ptr, i32 } %162, 0
  call void @__clang_call_terminate(ptr %163) #22
  unreachable

164:                                              ; preds = %156, %159
  %165 = getelementptr inbounds nuw i8, ptr %14, i64 16
  %166 = getelementptr inbounds nuw i8, ptr %14, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %14, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE3$_8E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %166, align 8, !tbaa !25
  store ptr @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE3$_8E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %165, align 8, !tbaa !20
  %167 = getelementptr inbounds nuw i8, ptr %15, i64 16
  %168 = getelementptr inbounds nuw i8, ptr %15, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %15, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE3$_9E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %168, align 8, !tbaa !25
  store ptr @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE3$_9E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %167, align 8, !tbaa !20
  invoke fastcc void @_ZL19checkVectorFunctionIjjEvSt8functionIFT_PT0_S3_jEES5_PKc(ptr noundef %14, ptr noundef %15, ptr noundef nonnull @.str.6)
          to label %169 unwind label %393

169:                                              ; preds = %164
  %170 = load ptr, ptr %167, align 8, !tbaa !20
  %171 = icmp eq ptr %170, null
  br i1 %171, label %177, label %172

172:                                              ; preds = %169
  %173 = invoke noundef i1 %170(ptr noundef nonnull align 8 dereferenceable(32) %15, ptr noundef nonnull align 8 dereferenceable(32) %15, i32 noundef 3)
          to label %177 unwind label %174

174:                                              ; preds = %172
  %175 = landingpad { ptr, i32 }
          catch ptr null
  %176 = extractvalue { ptr, i32 } %175, 0
  call void @__clang_call_terminate(ptr %176) #22
  unreachable

177:                                              ; preds = %169, %172
  %178 = load ptr, ptr %165, align 8, !tbaa !20
  %179 = icmp eq ptr %178, null
  br i1 %179, label %185, label %180

180:                                              ; preds = %177
  %181 = invoke noundef i1 %178(ptr noundef nonnull align 8 dereferenceable(32) %14, ptr noundef nonnull align 8 dereferenceable(32) %14, i32 noundef 3)
          to label %185 unwind label %182

182:                                              ; preds = %180
  %183 = landingpad { ptr, i32 }
          catch ptr null
  %184 = extractvalue { ptr, i32 } %183, 0
  call void @__clang_call_terminate(ptr %184) #22
  unreachable

185:                                              ; preds = %177, %180
  %186 = getelementptr inbounds nuw i8, ptr %16, i64 16
  %187 = getelementptr inbounds nuw i8, ptr %16, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %16, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFjPfS0_jEZ4mainE3$_8E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %187, align 8, !tbaa !27
  store ptr @"_ZNSt17_Function_handlerIFjPfS0_jEZ4mainE3$_8E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %186, align 8, !tbaa !20
  %188 = getelementptr inbounds nuw i8, ptr %17, i64 16
  %189 = getelementptr inbounds nuw i8, ptr %17, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %17, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFjPfS0_jEZ4mainE3$_9E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %189, align 8, !tbaa !27
  store ptr @"_ZNSt17_Function_handlerIFjPfS0_jEZ4mainE3$_9E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %188, align 8, !tbaa !20
  invoke fastcc void @_ZL19checkVectorFunctionIjfEvSt8functionIFT_PT0_S3_jEES5_PKc(ptr noundef %16, ptr noundef %17, ptr noundef nonnull @.str.7)
          to label %190 unwind label %410

190:                                              ; preds = %185
  %191 = load ptr, ptr %188, align 8, !tbaa !20
  %192 = icmp eq ptr %191, null
  br i1 %192, label %198, label %193

193:                                              ; preds = %190
  %194 = invoke noundef i1 %191(ptr noundef nonnull align 8 dereferenceable(32) %17, ptr noundef nonnull align 8 dereferenceable(32) %17, i32 noundef 3)
          to label %198 unwind label %195

195:                                              ; preds = %193
  %196 = landingpad { ptr, i32 }
          catch ptr null
  %197 = extractvalue { ptr, i32 } %196, 0
  call void @__clang_call_terminate(ptr %197) #22
  unreachable

198:                                              ; preds = %190, %193
  %199 = load ptr, ptr %186, align 8, !tbaa !20
  %200 = icmp eq ptr %199, null
  br i1 %200, label %206, label %201

201:                                              ; preds = %198
  %202 = invoke noundef i1 %199(ptr noundef nonnull align 8 dereferenceable(32) %16, ptr noundef nonnull align 8 dereferenceable(32) %16, i32 noundef 3)
          to label %206 unwind label %203

203:                                              ; preds = %201
  %204 = landingpad { ptr, i32 }
          catch ptr null
  %205 = extractvalue { ptr, i32 } %204, 0
  call void @__clang_call_terminate(ptr %205) #22
  unreachable

206:                                              ; preds = %198, %201
  %207 = getelementptr inbounds nuw i8, ptr %18, i64 16
  %208 = getelementptr inbounds nuw i8, ptr %18, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %18, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFtPtS0_jEZ4mainE4$_10E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %208, align 8, !tbaa !29
  store ptr @"_ZNSt17_Function_handlerIFtPtS0_jEZ4mainE4$_10E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %207, align 8, !tbaa !20
  %209 = getelementptr inbounds nuw i8, ptr %19, i64 16
  %210 = getelementptr inbounds nuw i8, ptr %19, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %19, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFtPtS0_jEZ4mainE4$_11E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %210, align 8, !tbaa !29
  store ptr @"_ZNSt17_Function_handlerIFtPtS0_jEZ4mainE4$_11E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %209, align 8, !tbaa !20
  invoke fastcc void @_ZL19checkVectorFunctionIttEvSt8functionIFT_PT0_S3_jEES5_PKc(ptr noundef %18, ptr noundef %19, ptr noundef nonnull @.str.8)
          to label %211 unwind label %427

211:                                              ; preds = %206
  %212 = load ptr, ptr %209, align 8, !tbaa !20
  %213 = icmp eq ptr %212, null
  br i1 %213, label %219, label %214

214:                                              ; preds = %211
  %215 = invoke noundef i1 %212(ptr noundef nonnull align 8 dereferenceable(32) %19, ptr noundef nonnull align 8 dereferenceable(32) %19, i32 noundef 3)
          to label %219 unwind label %216

216:                                              ; preds = %214
  %217 = landingpad { ptr, i32 }
          catch ptr null
  %218 = extractvalue { ptr, i32 } %217, 0
  call void @__clang_call_terminate(ptr %218) #22
  unreachable

219:                                              ; preds = %211, %214
  %220 = load ptr, ptr %207, align 8, !tbaa !20
  %221 = icmp eq ptr %220, null
  br i1 %221, label %227, label %222

222:                                              ; preds = %219
  %223 = invoke noundef i1 %220(ptr noundef nonnull align 8 dereferenceable(32) %18, ptr noundef nonnull align 8 dereferenceable(32) %18, i32 noundef 3)
          to label %227 unwind label %224

224:                                              ; preds = %222
  %225 = landingpad { ptr, i32 }
          catch ptr null
  %226 = extractvalue { ptr, i32 } %225, 0
  call void @__clang_call_terminate(ptr %226) #22
  unreachable

227:                                              ; preds = %219, %222
  %228 = getelementptr inbounds nuw i8, ptr %20, i64 16
  %229 = getelementptr inbounds nuw i8, ptr %20, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %20, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE4$_12E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %229, align 8, !tbaa !25
  store ptr @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE4$_12E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %228, align 8, !tbaa !20
  %230 = getelementptr inbounds nuw i8, ptr %21, i64 16
  %231 = getelementptr inbounds nuw i8, ptr %21, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %21, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE4$_13E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %231, align 8, !tbaa !25
  store ptr @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE4$_13E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %230, align 8, !tbaa !20
  invoke fastcc void @_ZL19checkVectorFunctionIjjEvSt8functionIFT_PT0_S3_jEES5_PKc(ptr noundef %20, ptr noundef %21, ptr noundef nonnull @.str.9)
          to label %232 unwind label %444

232:                                              ; preds = %227
  %233 = load ptr, ptr %230, align 8, !tbaa !20
  %234 = icmp eq ptr %233, null
  br i1 %234, label %240, label %235

235:                                              ; preds = %232
  %236 = invoke noundef i1 %233(ptr noundef nonnull align 8 dereferenceable(32) %21, ptr noundef nonnull align 8 dereferenceable(32) %21, i32 noundef 3)
          to label %240 unwind label %237

237:                                              ; preds = %235
  %238 = landingpad { ptr, i32 }
          catch ptr null
  %239 = extractvalue { ptr, i32 } %238, 0
  call void @__clang_call_terminate(ptr %239) #22
  unreachable

240:                                              ; preds = %232, %235
  %241 = load ptr, ptr %228, align 8, !tbaa !20
  %242 = icmp eq ptr %241, null
  br i1 %242, label %248, label %243

243:                                              ; preds = %240
  %244 = invoke noundef i1 %241(ptr noundef nonnull align 8 dereferenceable(32) %20, ptr noundef nonnull align 8 dereferenceable(32) %20, i32 noundef 3)
          to label %248 unwind label %245

245:                                              ; preds = %243
  %246 = landingpad { ptr, i32 }
          catch ptr null
  %247 = extractvalue { ptr, i32 } %246, 0
  call void @__clang_call_terminate(ptr %247) #22
  unreachable

248:                                              ; preds = %240, %243
  %249 = getelementptr inbounds nuw i8, ptr %22, i64 16
  %250 = getelementptr inbounds nuw i8, ptr %22, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %22, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFjPfS0_jEZ4mainE4$_12E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %250, align 8, !tbaa !27
  store ptr @"_ZNSt17_Function_handlerIFjPfS0_jEZ4mainE4$_12E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %249, align 8, !tbaa !20
  %251 = getelementptr inbounds nuw i8, ptr %23, i64 16
  %252 = getelementptr inbounds nuw i8, ptr %23, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %23, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFjPfS0_jEZ4mainE4$_13E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %252, align 8, !tbaa !27
  store ptr @"_ZNSt17_Function_handlerIFjPfS0_jEZ4mainE4$_13E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %251, align 8, !tbaa !20
  invoke fastcc void @_ZL19checkVectorFunctionIjfEvSt8functionIFT_PT0_S3_jEES5_PKc(ptr noundef %22, ptr noundef %23, ptr noundef nonnull @.str.10)
          to label %253 unwind label %461

253:                                              ; preds = %248
  %254 = load ptr, ptr %251, align 8, !tbaa !20
  %255 = icmp eq ptr %254, null
  br i1 %255, label %261, label %256

256:                                              ; preds = %253
  %257 = invoke noundef i1 %254(ptr noundef nonnull align 8 dereferenceable(32) %23, ptr noundef nonnull align 8 dereferenceable(32) %23, i32 noundef 3)
          to label %261 unwind label %258

258:                                              ; preds = %256
  %259 = landingpad { ptr, i32 }
          catch ptr null
  %260 = extractvalue { ptr, i32 } %259, 0
  call void @__clang_call_terminate(ptr %260) #22
  unreachable

261:                                              ; preds = %253, %256
  %262 = load ptr, ptr %249, align 8, !tbaa !20
  %263 = icmp eq ptr %262, null
  br i1 %263, label %269, label %264

264:                                              ; preds = %261
  %265 = invoke noundef i1 %262(ptr noundef nonnull align 8 dereferenceable(32) %22, ptr noundef nonnull align 8 dereferenceable(32) %22, i32 noundef 3)
          to label %269 unwind label %266

266:                                              ; preds = %264
  %267 = landingpad { ptr, i32 }
          catch ptr null
  %268 = extractvalue { ptr, i32 } %267, 0
  call void @__clang_call_terminate(ptr %268) #22
  unreachable

269:                                              ; preds = %261, %264
  %270 = getelementptr inbounds nuw i8, ptr %24, i64 16
  %271 = getelementptr inbounds nuw i8, ptr %24, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %24, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFtPtS0_jEZ4mainE4$_14E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %271, align 8, !tbaa !29
  store ptr @"_ZNSt17_Function_handlerIFtPtS0_jEZ4mainE4$_14E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %270, align 8, !tbaa !20
  %272 = getelementptr inbounds nuw i8, ptr %25, i64 16
  %273 = getelementptr inbounds nuw i8, ptr %25, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %25, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFtPtS0_jEZ4mainE4$_15E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %273, align 8, !tbaa !29
  store ptr @"_ZNSt17_Function_handlerIFtPtS0_jEZ4mainE4$_15E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %272, align 8, !tbaa !20
  invoke fastcc void @_ZL19checkVectorFunctionIttEvSt8functionIFT_PT0_S3_jEES5_PKc(ptr noundef %24, ptr noundef %25, ptr noundef nonnull @.str.11)
          to label %274 unwind label %478

274:                                              ; preds = %269
  %275 = load ptr, ptr %272, align 8, !tbaa !20
  %276 = icmp eq ptr %275, null
  br i1 %276, label %282, label %277

277:                                              ; preds = %274
  %278 = invoke noundef i1 %275(ptr noundef nonnull align 8 dereferenceable(32) %25, ptr noundef nonnull align 8 dereferenceable(32) %25, i32 noundef 3)
          to label %282 unwind label %279

279:                                              ; preds = %277
  %280 = landingpad { ptr, i32 }
          catch ptr null
  %281 = extractvalue { ptr, i32 } %280, 0
  call void @__clang_call_terminate(ptr %281) #22
  unreachable

282:                                              ; preds = %274, %277
  %283 = load ptr, ptr %270, align 8, !tbaa !20
  %284 = icmp eq ptr %283, null
  br i1 %284, label %290, label %285

285:                                              ; preds = %282
  %286 = invoke noundef i1 %283(ptr noundef nonnull align 8 dereferenceable(32) %24, ptr noundef nonnull align 8 dereferenceable(32) %24, i32 noundef 3)
          to label %290 unwind label %287

287:                                              ; preds = %285
  %288 = landingpad { ptr, i32 }
          catch ptr null
  %289 = extractvalue { ptr, i32 } %288, 0
  call void @__clang_call_terminate(ptr %289) #22
  unreachable

290:                                              ; preds = %282, %285
  ret i32 0

291:                                              ; preds = %37
  %292 = landingpad { ptr, i32 }
          cleanup
  %293 = load ptr, ptr %41, align 8, !tbaa !20
  %294 = icmp eq ptr %293, null
  br i1 %294, label %300, label %295

295:                                              ; preds = %291
  %296 = invoke noundef i1 %293(ptr noundef nonnull align 8 dereferenceable(32) %3, ptr noundef nonnull align 8 dereferenceable(32) %3, i32 noundef 3)
          to label %300 unwind label %297

297:                                              ; preds = %295
  %298 = landingpad { ptr, i32 }
          catch ptr null
  %299 = extractvalue { ptr, i32 } %298, 0
  call void @__clang_call_terminate(ptr %299) #22
  unreachable

300:                                              ; preds = %291, %295
  %301 = load ptr, ptr %39, align 8, !tbaa !20
  %302 = icmp eq ptr %301, null
  br i1 %302, label %495, label %303

303:                                              ; preds = %300
  %304 = invoke noundef i1 %301(ptr noundef nonnull align 8 dereferenceable(32) %2, ptr noundef nonnull align 8 dereferenceable(32) %2, i32 noundef 3)
          to label %495 unwind label %305

305:                                              ; preds = %303
  %306 = landingpad { ptr, i32 }
          catch ptr null
  %307 = extractvalue { ptr, i32 } %306, 0
  call void @__clang_call_terminate(ptr %307) #22
  unreachable

308:                                              ; preds = %59
  %309 = landingpad { ptr, i32 }
          cleanup
  %310 = load ptr, ptr %62, align 8, !tbaa !20
  %311 = icmp eq ptr %310, null
  br i1 %311, label %317, label %312

312:                                              ; preds = %308
  %313 = invoke noundef i1 %310(ptr noundef nonnull align 8 dereferenceable(32) %5, ptr noundef nonnull align 8 dereferenceable(32) %5, i32 noundef 3)
          to label %317 unwind label %314

314:                                              ; preds = %312
  %315 = landingpad { ptr, i32 }
          catch ptr null
  %316 = extractvalue { ptr, i32 } %315, 0
  call void @__clang_call_terminate(ptr %316) #22
  unreachable

317:                                              ; preds = %308, %312
  %318 = load ptr, ptr %60, align 8, !tbaa !20
  %319 = icmp eq ptr %318, null
  br i1 %319, label %495, label %320

320:                                              ; preds = %317
  %321 = invoke noundef i1 %318(ptr noundef nonnull align 8 dereferenceable(32) %4, ptr noundef nonnull align 8 dereferenceable(32) %4, i32 noundef 3)
          to label %495 unwind label %322

322:                                              ; preds = %320
  %323 = landingpad { ptr, i32 }
          catch ptr null
  %324 = extractvalue { ptr, i32 } %323, 0
  call void @__clang_call_terminate(ptr %324) #22
  unreachable

325:                                              ; preds = %80
  %326 = landingpad { ptr, i32 }
          cleanup
  %327 = load ptr, ptr %83, align 8, !tbaa !20
  %328 = icmp eq ptr %327, null
  br i1 %328, label %334, label %329

329:                                              ; preds = %325
  %330 = invoke noundef i1 %327(ptr noundef nonnull align 8 dereferenceable(32) %7, ptr noundef nonnull align 8 dereferenceable(32) %7, i32 noundef 3)
          to label %334 unwind label %331

331:                                              ; preds = %329
  %332 = landingpad { ptr, i32 }
          catch ptr null
  %333 = extractvalue { ptr, i32 } %332, 0
  call void @__clang_call_terminate(ptr %333) #22
  unreachable

334:                                              ; preds = %325, %329
  %335 = load ptr, ptr %81, align 8, !tbaa !20
  %336 = icmp eq ptr %335, null
  br i1 %336, label %495, label %337

337:                                              ; preds = %334
  %338 = invoke noundef i1 %335(ptr noundef nonnull align 8 dereferenceable(32) %6, ptr noundef nonnull align 8 dereferenceable(32) %6, i32 noundef 3)
          to label %495 unwind label %339

339:                                              ; preds = %337
  %340 = landingpad { ptr, i32 }
          catch ptr null
  %341 = extractvalue { ptr, i32 } %340, 0
  call void @__clang_call_terminate(ptr %341) #22
  unreachable

342:                                              ; preds = %101
  %343 = landingpad { ptr, i32 }
          cleanup
  %344 = load ptr, ptr %104, align 8, !tbaa !20
  %345 = icmp eq ptr %344, null
  br i1 %345, label %351, label %346

346:                                              ; preds = %342
  %347 = invoke noundef i1 %344(ptr noundef nonnull align 8 dereferenceable(32) %9, ptr noundef nonnull align 8 dereferenceable(32) %9, i32 noundef 3)
          to label %351 unwind label %348

348:                                              ; preds = %346
  %349 = landingpad { ptr, i32 }
          catch ptr null
  %350 = extractvalue { ptr, i32 } %349, 0
  call void @__clang_call_terminate(ptr %350) #22
  unreachable

351:                                              ; preds = %342, %346
  %352 = load ptr, ptr %102, align 8, !tbaa !20
  %353 = icmp eq ptr %352, null
  br i1 %353, label %495, label %354

354:                                              ; preds = %351
  %355 = invoke noundef i1 %352(ptr noundef nonnull align 8 dereferenceable(32) %8, ptr noundef nonnull align 8 dereferenceable(32) %8, i32 noundef 3)
          to label %495 unwind label %356

356:                                              ; preds = %354
  %357 = landingpad { ptr, i32 }
          catch ptr null
  %358 = extractvalue { ptr, i32 } %357, 0
  call void @__clang_call_terminate(ptr %358) #22
  unreachable

359:                                              ; preds = %122
  %360 = landingpad { ptr, i32 }
          cleanup
  %361 = load ptr, ptr %125, align 8, !tbaa !20
  %362 = icmp eq ptr %361, null
  br i1 %362, label %368, label %363

363:                                              ; preds = %359
  %364 = invoke noundef i1 %361(ptr noundef nonnull align 8 dereferenceable(32) %11, ptr noundef nonnull align 8 dereferenceable(32) %11, i32 noundef 3)
          to label %368 unwind label %365

365:                                              ; preds = %363
  %366 = landingpad { ptr, i32 }
          catch ptr null
  %367 = extractvalue { ptr, i32 } %366, 0
  call void @__clang_call_terminate(ptr %367) #22
  unreachable

368:                                              ; preds = %359, %363
  %369 = load ptr, ptr %123, align 8, !tbaa !20
  %370 = icmp eq ptr %369, null
  br i1 %370, label %495, label %371

371:                                              ; preds = %368
  %372 = invoke noundef i1 %369(ptr noundef nonnull align 8 dereferenceable(32) %10, ptr noundef nonnull align 8 dereferenceable(32) %10, i32 noundef 3)
          to label %495 unwind label %373

373:                                              ; preds = %371
  %374 = landingpad { ptr, i32 }
          catch ptr null
  %375 = extractvalue { ptr, i32 } %374, 0
  call void @__clang_call_terminate(ptr %375) #22
  unreachable

376:                                              ; preds = %143
  %377 = landingpad { ptr, i32 }
          cleanup
  %378 = load ptr, ptr %146, align 8, !tbaa !20
  %379 = icmp eq ptr %378, null
  br i1 %379, label %385, label %380

380:                                              ; preds = %376
  %381 = invoke noundef i1 %378(ptr noundef nonnull align 8 dereferenceable(32) %13, ptr noundef nonnull align 8 dereferenceable(32) %13, i32 noundef 3)
          to label %385 unwind label %382

382:                                              ; preds = %380
  %383 = landingpad { ptr, i32 }
          catch ptr null
  %384 = extractvalue { ptr, i32 } %383, 0
  call void @__clang_call_terminate(ptr %384) #22
  unreachable

385:                                              ; preds = %376, %380
  %386 = load ptr, ptr %144, align 8, !tbaa !20
  %387 = icmp eq ptr %386, null
  br i1 %387, label %495, label %388

388:                                              ; preds = %385
  %389 = invoke noundef i1 %386(ptr noundef nonnull align 8 dereferenceable(32) %12, ptr noundef nonnull align 8 dereferenceable(32) %12, i32 noundef 3)
          to label %495 unwind label %390

390:                                              ; preds = %388
  %391 = landingpad { ptr, i32 }
          catch ptr null
  %392 = extractvalue { ptr, i32 } %391, 0
  call void @__clang_call_terminate(ptr %392) #22
  unreachable

393:                                              ; preds = %164
  %394 = landingpad { ptr, i32 }
          cleanup
  %395 = load ptr, ptr %167, align 8, !tbaa !20
  %396 = icmp eq ptr %395, null
  br i1 %396, label %402, label %397

397:                                              ; preds = %393
  %398 = invoke noundef i1 %395(ptr noundef nonnull align 8 dereferenceable(32) %15, ptr noundef nonnull align 8 dereferenceable(32) %15, i32 noundef 3)
          to label %402 unwind label %399

399:                                              ; preds = %397
  %400 = landingpad { ptr, i32 }
          catch ptr null
  %401 = extractvalue { ptr, i32 } %400, 0
  call void @__clang_call_terminate(ptr %401) #22
  unreachable

402:                                              ; preds = %393, %397
  %403 = load ptr, ptr %165, align 8, !tbaa !20
  %404 = icmp eq ptr %403, null
  br i1 %404, label %495, label %405

405:                                              ; preds = %402
  %406 = invoke noundef i1 %403(ptr noundef nonnull align 8 dereferenceable(32) %14, ptr noundef nonnull align 8 dereferenceable(32) %14, i32 noundef 3)
          to label %495 unwind label %407

407:                                              ; preds = %405
  %408 = landingpad { ptr, i32 }
          catch ptr null
  %409 = extractvalue { ptr, i32 } %408, 0
  call void @__clang_call_terminate(ptr %409) #22
  unreachable

410:                                              ; preds = %185
  %411 = landingpad { ptr, i32 }
          cleanup
  %412 = load ptr, ptr %188, align 8, !tbaa !20
  %413 = icmp eq ptr %412, null
  br i1 %413, label %419, label %414

414:                                              ; preds = %410
  %415 = invoke noundef i1 %412(ptr noundef nonnull align 8 dereferenceable(32) %17, ptr noundef nonnull align 8 dereferenceable(32) %17, i32 noundef 3)
          to label %419 unwind label %416

416:                                              ; preds = %414
  %417 = landingpad { ptr, i32 }
          catch ptr null
  %418 = extractvalue { ptr, i32 } %417, 0
  call void @__clang_call_terminate(ptr %418) #22
  unreachable

419:                                              ; preds = %410, %414
  %420 = load ptr, ptr %186, align 8, !tbaa !20
  %421 = icmp eq ptr %420, null
  br i1 %421, label %495, label %422

422:                                              ; preds = %419
  %423 = invoke noundef i1 %420(ptr noundef nonnull align 8 dereferenceable(32) %16, ptr noundef nonnull align 8 dereferenceable(32) %16, i32 noundef 3)
          to label %495 unwind label %424

424:                                              ; preds = %422
  %425 = landingpad { ptr, i32 }
          catch ptr null
  %426 = extractvalue { ptr, i32 } %425, 0
  call void @__clang_call_terminate(ptr %426) #22
  unreachable

427:                                              ; preds = %206
  %428 = landingpad { ptr, i32 }
          cleanup
  %429 = load ptr, ptr %209, align 8, !tbaa !20
  %430 = icmp eq ptr %429, null
  br i1 %430, label %436, label %431

431:                                              ; preds = %427
  %432 = invoke noundef i1 %429(ptr noundef nonnull align 8 dereferenceable(32) %19, ptr noundef nonnull align 8 dereferenceable(32) %19, i32 noundef 3)
          to label %436 unwind label %433

433:                                              ; preds = %431
  %434 = landingpad { ptr, i32 }
          catch ptr null
  %435 = extractvalue { ptr, i32 } %434, 0
  call void @__clang_call_terminate(ptr %435) #22
  unreachable

436:                                              ; preds = %427, %431
  %437 = load ptr, ptr %207, align 8, !tbaa !20
  %438 = icmp eq ptr %437, null
  br i1 %438, label %495, label %439

439:                                              ; preds = %436
  %440 = invoke noundef i1 %437(ptr noundef nonnull align 8 dereferenceable(32) %18, ptr noundef nonnull align 8 dereferenceable(32) %18, i32 noundef 3)
          to label %495 unwind label %441

441:                                              ; preds = %439
  %442 = landingpad { ptr, i32 }
          catch ptr null
  %443 = extractvalue { ptr, i32 } %442, 0
  call void @__clang_call_terminate(ptr %443) #22
  unreachable

444:                                              ; preds = %227
  %445 = landingpad { ptr, i32 }
          cleanup
  %446 = load ptr, ptr %230, align 8, !tbaa !20
  %447 = icmp eq ptr %446, null
  br i1 %447, label %453, label %448

448:                                              ; preds = %444
  %449 = invoke noundef i1 %446(ptr noundef nonnull align 8 dereferenceable(32) %21, ptr noundef nonnull align 8 dereferenceable(32) %21, i32 noundef 3)
          to label %453 unwind label %450

450:                                              ; preds = %448
  %451 = landingpad { ptr, i32 }
          catch ptr null
  %452 = extractvalue { ptr, i32 } %451, 0
  call void @__clang_call_terminate(ptr %452) #22
  unreachable

453:                                              ; preds = %444, %448
  %454 = load ptr, ptr %228, align 8, !tbaa !20
  %455 = icmp eq ptr %454, null
  br i1 %455, label %495, label %456

456:                                              ; preds = %453
  %457 = invoke noundef i1 %454(ptr noundef nonnull align 8 dereferenceable(32) %20, ptr noundef nonnull align 8 dereferenceable(32) %20, i32 noundef 3)
          to label %495 unwind label %458

458:                                              ; preds = %456
  %459 = landingpad { ptr, i32 }
          catch ptr null
  %460 = extractvalue { ptr, i32 } %459, 0
  call void @__clang_call_terminate(ptr %460) #22
  unreachable

461:                                              ; preds = %248
  %462 = landingpad { ptr, i32 }
          cleanup
  %463 = load ptr, ptr %251, align 8, !tbaa !20
  %464 = icmp eq ptr %463, null
  br i1 %464, label %470, label %465

465:                                              ; preds = %461
  %466 = invoke noundef i1 %463(ptr noundef nonnull align 8 dereferenceable(32) %23, ptr noundef nonnull align 8 dereferenceable(32) %23, i32 noundef 3)
          to label %470 unwind label %467

467:                                              ; preds = %465
  %468 = landingpad { ptr, i32 }
          catch ptr null
  %469 = extractvalue { ptr, i32 } %468, 0
  call void @__clang_call_terminate(ptr %469) #22
  unreachable

470:                                              ; preds = %461, %465
  %471 = load ptr, ptr %249, align 8, !tbaa !20
  %472 = icmp eq ptr %471, null
  br i1 %472, label %495, label %473

473:                                              ; preds = %470
  %474 = invoke noundef i1 %471(ptr noundef nonnull align 8 dereferenceable(32) %22, ptr noundef nonnull align 8 dereferenceable(32) %22, i32 noundef 3)
          to label %495 unwind label %475

475:                                              ; preds = %473
  %476 = landingpad { ptr, i32 }
          catch ptr null
  %477 = extractvalue { ptr, i32 } %476, 0
  call void @__clang_call_terminate(ptr %477) #22
  unreachable

478:                                              ; preds = %269
  %479 = landingpad { ptr, i32 }
          cleanup
  %480 = load ptr, ptr %272, align 8, !tbaa !20
  %481 = icmp eq ptr %480, null
  br i1 %481, label %487, label %482

482:                                              ; preds = %478
  %483 = invoke noundef i1 %480(ptr noundef nonnull align 8 dereferenceable(32) %25, ptr noundef nonnull align 8 dereferenceable(32) %25, i32 noundef 3)
          to label %487 unwind label %484

484:                                              ; preds = %482
  %485 = landingpad { ptr, i32 }
          catch ptr null
  %486 = extractvalue { ptr, i32 } %485, 0
  call void @__clang_call_terminate(ptr %486) #22
  unreachable

487:                                              ; preds = %478, %482
  %488 = load ptr, ptr %270, align 8, !tbaa !20
  %489 = icmp eq ptr %488, null
  br i1 %489, label %495, label %490

490:                                              ; preds = %487
  %491 = invoke noundef i1 %488(ptr noundef nonnull align 8 dereferenceable(32) %24, ptr noundef nonnull align 8 dereferenceable(32) %24, i32 noundef 3)
          to label %495 unwind label %492

492:                                              ; preds = %490
  %493 = landingpad { ptr, i32 }
          catch ptr null
  %494 = extractvalue { ptr, i32 } %493, 0
  call void @__clang_call_terminate(ptr %494) #22
  unreachable

495:                                              ; preds = %490, %487, %453, %456, %470, %473, %439, %436, %402, %405, %419, %422, %388, %385, %351, %354, %368, %371, %337, %334, %300, %303, %317, %320
  %496 = phi { ptr, i32 } [ %292, %300 ], [ %292, %303 ], [ %309, %317 ], [ %309, %320 ], [ %326, %334 ], [ %326, %337 ], [ %343, %351 ], [ %343, %354 ], [ %360, %368 ], [ %360, %371 ], [ %377, %385 ], [ %377, %388 ], [ %394, %402 ], [ %394, %405 ], [ %411, %419 ], [ %411, %422 ], [ %428, %436 ], [ %428, %439 ], [ %445, %453 ], [ %445, %456 ], [ %462, %470 ], [ %462, %473 ], [ %479, %487 ], [ %479, %490 ]
  resume { ptr, i32 } %496
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: mustprogress norecurse uwtable
define internal fastcc void @_ZL19checkVectorFunctionIiiEvSt8functionIFT_PT0_S3_jEES5_PKc(ptr noundef nonnull %0, ptr noundef nonnull %1, ptr noundef %2) unnamed_addr #0 personality ptr @__gxx_personality_v0 {
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
  %40 = alloca ptr, align 8
  %41 = alloca ptr, align 8
  %42 = alloca i32, align 4
  %43 = alloca ptr, align 8
  %44 = alloca ptr, align 8
  %45 = alloca i32, align 4
  %46 = alloca %"class.std::uniform_int_distribution", align 8
  %47 = alloca %"class.std::uniform_int_distribution", align 8
  %48 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.12, i64 noundef 9)
  %49 = icmp eq ptr %2, null
  br i1 %49, label %50, label %58

50:                                               ; preds = %3
  %51 = load ptr, ptr @_ZSt4cout, align 8, !tbaa !31
  %52 = getelementptr i8, ptr %51, i64 -24
  %53 = load i64, ptr %52, align 8
  %54 = getelementptr inbounds i8, ptr @_ZSt4cout, i64 %53
  %55 = getelementptr inbounds nuw i8, ptr %54, i64 32
  %56 = load i32, ptr %55, align 8, !tbaa !33
  %57 = or i32 %56, 1
  tail call void @_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate(ptr noundef nonnull align 8 dereferenceable(264) %54, i32 noundef %57)
  br label %61

58:                                               ; preds = %3
  %59 = tail call noundef i64 @strlen(ptr noundef nonnull dereferenceable(1) %2) #21
  %60 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull %2, i64 noundef %59)
  br label %61

61:                                               ; preds = %50, %58
  %62 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.13, i64 noundef 1)
  %63 = tail call noalias noundef nonnull dereferenceable(4000) ptr @_Znam(i64 noundef 4000) #23
  %64 = invoke noalias noundef nonnull dereferenceable(4000) ptr @_Znam(i64 noundef 4000) #23
          to label %65 unwind label %114

65:                                               ; preds = %61
  call void @llvm.lifetime.start.p0(ptr nonnull %47) #21
  store <2 x i32> <i32 -2147483648, i32 2147483647>, ptr %47, align 8, !tbaa !43
  br label %66

66:                                               ; preds = %69, %65
  %67 = phi i64 [ 0, %65 ], [ %71, %69 ]
  %68 = invoke noundef i32 @_ZNSt24uniform_int_distributionIiEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEiRT_RKNS0_10param_typeE(ptr noundef nonnull align 4 dereferenceable(8) %47, ptr noundef nonnull align 8 dereferenceable(5000) @_ZL3rng, ptr noundef nonnull align 4 dereferenceable(8) %47)
          to label %69 unwind label %118

69:                                               ; preds = %66
  %70 = getelementptr inbounds nuw i32, ptr %63, i64 %67
  store i32 %68, ptr %70, align 4, !tbaa !43
  %71 = add nuw nsw i64 %67, 1
  %72 = icmp eq i64 %71, 1000
  br i1 %72, label %73, label %66, !llvm.loop !44

73:                                               ; preds = %69
  call void @llvm.lifetime.end.p0(ptr nonnull %47) #21
  call void @llvm.lifetime.start.p0(ptr nonnull %46) #21
  store <2 x i32> <i32 -2147483648, i32 2147483647>, ptr %46, align 8, !tbaa !43
  br label %74

74:                                               ; preds = %77, %73
  %75 = phi i64 [ 0, %73 ], [ %79, %77 ]
  %76 = invoke noundef i32 @_ZNSt24uniform_int_distributionIiEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEiRT_RKNS0_10param_typeE(ptr noundef nonnull align 4 dereferenceable(8) %46, ptr noundef nonnull align 8 dereferenceable(5000) @_ZL3rng, ptr noundef nonnull align 4 dereferenceable(8) %46)
          to label %77 unwind label %116

77:                                               ; preds = %74
  %78 = getelementptr inbounds nuw i32, ptr %64, i64 %75
  store i32 %76, ptr %78, align 4, !tbaa !43
  %79 = add nuw nsw i64 %75, 1
  %80 = icmp eq i64 %79, 1000
  br i1 %80, label %81, label %74, !llvm.loop !44

81:                                               ; preds = %77
  call void @llvm.lifetime.end.p0(ptr nonnull %46) #21
  call void @llvm.lifetime.start.p0(ptr nonnull %43)
  call void @llvm.lifetime.start.p0(ptr nonnull %44)
  call void @llvm.lifetime.start.p0(ptr nonnull %45)
  store ptr %63, ptr %43, align 8, !tbaa !45
  store ptr %64, ptr %44, align 8, !tbaa !45
  store i32 1000, ptr %45, align 4, !tbaa !43
  %82 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %83 = load ptr, ptr %82, align 8, !tbaa !20
  %84 = icmp eq ptr %83, null
  br i1 %84, label %85, label %87

85:                                               ; preds = %81
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %86 unwind label %120

86:                                               ; preds = %85
  unreachable

87:                                               ; preds = %81
  %88 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %89 = load ptr, ptr %88, align 8, !tbaa !16
  %90 = invoke noundef i32 %89(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %43, ptr noundef nonnull align 8 dereferenceable(8) %44, ptr noundef nonnull align 4 dereferenceable(4) %45)
          to label %91 unwind label %120

91:                                               ; preds = %87
  call void @llvm.lifetime.end.p0(ptr nonnull %43)
  call void @llvm.lifetime.end.p0(ptr nonnull %44)
  call void @llvm.lifetime.end.p0(ptr nonnull %45)
  call void @llvm.lifetime.start.p0(ptr nonnull %40)
  call void @llvm.lifetime.start.p0(ptr nonnull %41)
  call void @llvm.lifetime.start.p0(ptr nonnull %42)
  store ptr %63, ptr %40, align 8, !tbaa !45
  store ptr %64, ptr %41, align 8, !tbaa !45
  store i32 1000, ptr %42, align 4, !tbaa !43
  %92 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %93 = load ptr, ptr %92, align 8, !tbaa !20
  %94 = icmp eq ptr %93, null
  br i1 %94, label %95, label %97

95:                                               ; preds = %91
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %96 unwind label %122

96:                                               ; preds = %95
  unreachable

97:                                               ; preds = %91
  %98 = getelementptr inbounds nuw i8, ptr %1, i64 24
  %99 = load ptr, ptr %98, align 8, !tbaa !16
  %100 = invoke noundef i32 %99(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %40, ptr noundef nonnull align 8 dereferenceable(8) %41, ptr noundef nonnull align 4 dereferenceable(4) %42)
          to label %101 unwind label %122

101:                                              ; preds = %97
  call void @llvm.lifetime.end.p0(ptr nonnull %40)
  call void @llvm.lifetime.end.p0(ptr nonnull %41)
  call void @llvm.lifetime.end.p0(ptr nonnull %42)
  %102 = icmp eq i32 %90, %100
  br i1 %102, label %103, label %111

103:                                              ; preds = %101, %103
  %104 = phi i64 [ %109, %103 ], [ 0, %101 ]
  %105 = getelementptr inbounds nuw i32, ptr %63, i64 %104
  %106 = getelementptr inbounds nuw i8, ptr %105, i64 16
  store <4 x i32> splat (i32 2147483647), ptr %105, align 4, !tbaa !43
  store <4 x i32> splat (i32 2147483647), ptr %106, align 4, !tbaa !43
  %107 = getelementptr inbounds nuw i32, ptr %64, i64 %104
  %108 = getelementptr inbounds nuw i8, ptr %107, i64 16
  store <4 x i32> splat (i32 -2147483648), ptr %107, align 4, !tbaa !43
  store <4 x i32> splat (i32 -2147483648), ptr %108, align 4, !tbaa !43
  %109 = add nuw i64 %104, 8
  %110 = icmp eq i64 %109, 1000
  br i1 %110, label %124, label %103, !llvm.loop !47

111:                                              ; preds = %101
  %112 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.14)
          to label %113 unwind label %122

113:                                              ; preds = %111
  call void @exit(i32 noundef 1) #25
  unreachable

114:                                              ; preds = %61
  %115 = landingpad { ptr, i32 }
          cleanup
  br label %319

116:                                              ; preds = %74
  %117 = landingpad { ptr, i32 }
          cleanup
  br label %317

118:                                              ; preds = %66
  %119 = landingpad { ptr, i32 }
          cleanup
  br label %317

120:                                              ; preds = %87, %85
  %121 = landingpad { ptr, i32 }
          cleanup
  br label %317

122:                                              ; preds = %97, %95, %111
  %123 = landingpad { ptr, i32 }
          cleanup
  br label %317

124:                                              ; preds = %103
  call void @llvm.lifetime.start.p0(ptr nonnull %37)
  call void @llvm.lifetime.start.p0(ptr nonnull %38)
  call void @llvm.lifetime.start.p0(ptr nonnull %39)
  store ptr %63, ptr %37, align 8, !tbaa !45
  store ptr %64, ptr %38, align 8, !tbaa !45
  store i32 1000, ptr %39, align 4, !tbaa !43
  %125 = load ptr, ptr %82, align 8, !tbaa !20
  %126 = icmp eq ptr %125, null
  br i1 %126, label %127, label %129

127:                                              ; preds = %124
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %128 unwind label %153

128:                                              ; preds = %127
  unreachable

129:                                              ; preds = %124
  %130 = load ptr, ptr %88, align 8, !tbaa !16
  %131 = invoke noundef i32 %130(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %37, ptr noundef nonnull align 8 dereferenceable(8) %38, ptr noundef nonnull align 4 dereferenceable(4) %39)
          to label %132 unwind label %153

132:                                              ; preds = %129
  call void @llvm.lifetime.end.p0(ptr nonnull %37)
  call void @llvm.lifetime.end.p0(ptr nonnull %38)
  call void @llvm.lifetime.end.p0(ptr nonnull %39)
  call void @llvm.lifetime.start.p0(ptr nonnull %34)
  call void @llvm.lifetime.start.p0(ptr nonnull %35)
  call void @llvm.lifetime.start.p0(ptr nonnull %36)
  store ptr %63, ptr %34, align 8, !tbaa !45
  store ptr %64, ptr %35, align 8, !tbaa !45
  store i32 1000, ptr %36, align 4, !tbaa !43
  %133 = load ptr, ptr %92, align 8, !tbaa !20
  %134 = icmp eq ptr %133, null
  br i1 %134, label %135, label %137

135:                                              ; preds = %132
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %136 unwind label %155

136:                                              ; preds = %135
  unreachable

137:                                              ; preds = %132
  %138 = load ptr, ptr %98, align 8, !tbaa !16
  %139 = invoke noundef i32 %138(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %34, ptr noundef nonnull align 8 dereferenceable(8) %35, ptr noundef nonnull align 4 dereferenceable(4) %36)
          to label %140 unwind label %155

140:                                              ; preds = %137
  call void @llvm.lifetime.end.p0(ptr nonnull %34)
  call void @llvm.lifetime.end.p0(ptr nonnull %35)
  call void @llvm.lifetime.end.p0(ptr nonnull %36)
  %141 = icmp eq i32 %131, %139
  br i1 %141, label %142, label %150

142:                                              ; preds = %140, %142
  %143 = phi i64 [ %148, %142 ], [ 0, %140 ]
  %144 = getelementptr inbounds nuw i32, ptr %63, i64 %143
  %145 = getelementptr inbounds nuw i8, ptr %144, i64 16
  store <4 x i32> splat (i32 -2147483648), ptr %144, align 4, !tbaa !43
  store <4 x i32> splat (i32 -2147483648), ptr %145, align 4, !tbaa !43
  %146 = getelementptr inbounds nuw i32, ptr %64, i64 %143
  %147 = getelementptr inbounds nuw i8, ptr %146, i64 16
  store <4 x i32> splat (i32 2147483647), ptr %146, align 4, !tbaa !43
  store <4 x i32> splat (i32 2147483647), ptr %147, align 4, !tbaa !43
  %148 = add nuw i64 %143, 8
  %149 = icmp eq i64 %148, 1000
  br i1 %149, label %157, label %142, !llvm.loop !50

150:                                              ; preds = %140
  %151 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.14)
          to label %152 unwind label %155

152:                                              ; preds = %150
  call void @exit(i32 noundef 1) #25
  unreachable

153:                                              ; preds = %129, %127
  %154 = landingpad { ptr, i32 }
          cleanup
  br label %317

155:                                              ; preds = %137, %135, %150
  %156 = landingpad { ptr, i32 }
          cleanup
  br label %317

157:                                              ; preds = %142
  call void @llvm.lifetime.start.p0(ptr nonnull %31)
  call void @llvm.lifetime.start.p0(ptr nonnull %32)
  call void @llvm.lifetime.start.p0(ptr nonnull %33)
  store ptr %63, ptr %31, align 8, !tbaa !45
  store ptr %64, ptr %32, align 8, !tbaa !45
  store i32 1000, ptr %33, align 4, !tbaa !43
  %158 = load ptr, ptr %82, align 8, !tbaa !20
  %159 = icmp eq ptr %158, null
  br i1 %159, label %160, label %162

160:                                              ; preds = %157
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %161 unwind label %186

161:                                              ; preds = %160
  unreachable

162:                                              ; preds = %157
  %163 = load ptr, ptr %88, align 8, !tbaa !16
  %164 = invoke noundef i32 %163(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %31, ptr noundef nonnull align 8 dereferenceable(8) %32, ptr noundef nonnull align 4 dereferenceable(4) %33)
          to label %165 unwind label %186

165:                                              ; preds = %162
  call void @llvm.lifetime.end.p0(ptr nonnull %31)
  call void @llvm.lifetime.end.p0(ptr nonnull %32)
  call void @llvm.lifetime.end.p0(ptr nonnull %33)
  call void @llvm.lifetime.start.p0(ptr nonnull %28)
  call void @llvm.lifetime.start.p0(ptr nonnull %29)
  call void @llvm.lifetime.start.p0(ptr nonnull %30)
  store ptr %63, ptr %28, align 8, !tbaa !45
  store ptr %64, ptr %29, align 8, !tbaa !45
  store i32 1000, ptr %30, align 4, !tbaa !43
  %166 = load ptr, ptr %92, align 8, !tbaa !20
  %167 = icmp eq ptr %166, null
  br i1 %167, label %168, label %170

168:                                              ; preds = %165
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %169 unwind label %188

169:                                              ; preds = %168
  unreachable

170:                                              ; preds = %165
  %171 = load ptr, ptr %98, align 8, !tbaa !16
  %172 = invoke noundef i32 %171(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %28, ptr noundef nonnull align 8 dereferenceable(8) %29, ptr noundef nonnull align 4 dereferenceable(4) %30)
          to label %173 unwind label %188

173:                                              ; preds = %170
  call void @llvm.lifetime.end.p0(ptr nonnull %28)
  call void @llvm.lifetime.end.p0(ptr nonnull %29)
  call void @llvm.lifetime.end.p0(ptr nonnull %30)
  %174 = icmp eq i32 %164, %172
  br i1 %174, label %175, label %183

175:                                              ; preds = %173, %175
  %176 = phi i64 [ %181, %175 ], [ 0, %173 ]
  %177 = getelementptr inbounds nuw i32, ptr %64, i64 %176
  %178 = getelementptr inbounds nuw i8, ptr %177, i64 16
  store <4 x i32> splat (i32 -2147483648), ptr %177, align 4, !tbaa !43
  store <4 x i32> splat (i32 -2147483648), ptr %178, align 4, !tbaa !43
  %179 = getelementptr inbounds nuw i32, ptr %63, i64 %176
  %180 = getelementptr inbounds nuw i8, ptr %179, i64 16
  store <4 x i32> splat (i32 -2147483648), ptr %179, align 4, !tbaa !43
  store <4 x i32> splat (i32 -2147483648), ptr %180, align 4, !tbaa !43
  %181 = add nuw i64 %176, 8
  %182 = icmp eq i64 %181, 1000
  br i1 %182, label %190, label %175, !llvm.loop !51

183:                                              ; preds = %173
  %184 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.14)
          to label %185 unwind label %188

185:                                              ; preds = %183
  call void @exit(i32 noundef 1) #25
  unreachable

186:                                              ; preds = %162, %160
  %187 = landingpad { ptr, i32 }
          cleanup
  br label %317

188:                                              ; preds = %170, %168, %183
  %189 = landingpad { ptr, i32 }
          cleanup
  br label %317

190:                                              ; preds = %175
  %191 = getelementptr inbounds nuw i8, ptr %63, i64 3992
  store i32 2147483647, ptr %191, align 4, !tbaa !43
  call void @llvm.lifetime.start.p0(ptr nonnull %25)
  call void @llvm.lifetime.start.p0(ptr nonnull %26)
  call void @llvm.lifetime.start.p0(ptr nonnull %27)
  store ptr %63, ptr %25, align 8, !tbaa !45
  store ptr %64, ptr %26, align 8, !tbaa !45
  store i32 1000, ptr %27, align 4, !tbaa !43
  %192 = load ptr, ptr %82, align 8, !tbaa !20
  %193 = icmp eq ptr %192, null
  br i1 %193, label %194, label %196

194:                                              ; preds = %190
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %195 unwind label %220

195:                                              ; preds = %194
  unreachable

196:                                              ; preds = %190
  %197 = load ptr, ptr %88, align 8, !tbaa !16
  %198 = invoke noundef i32 %197(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %25, ptr noundef nonnull align 8 dereferenceable(8) %26, ptr noundef nonnull align 4 dereferenceable(4) %27)
          to label %199 unwind label %220

199:                                              ; preds = %196
  call void @llvm.lifetime.end.p0(ptr nonnull %25)
  call void @llvm.lifetime.end.p0(ptr nonnull %26)
  call void @llvm.lifetime.end.p0(ptr nonnull %27)
  call void @llvm.lifetime.start.p0(ptr nonnull %22)
  call void @llvm.lifetime.start.p0(ptr nonnull %23)
  call void @llvm.lifetime.start.p0(ptr nonnull %24)
  store ptr %63, ptr %22, align 8, !tbaa !45
  store ptr %64, ptr %23, align 8, !tbaa !45
  store i32 1000, ptr %24, align 4, !tbaa !43
  %200 = load ptr, ptr %92, align 8, !tbaa !20
  %201 = icmp eq ptr %200, null
  br i1 %201, label %202, label %204

202:                                              ; preds = %199
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %203 unwind label %222

203:                                              ; preds = %202
  unreachable

204:                                              ; preds = %199
  %205 = load ptr, ptr %98, align 8, !tbaa !16
  %206 = invoke noundef i32 %205(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %22, ptr noundef nonnull align 8 dereferenceable(8) %23, ptr noundef nonnull align 4 dereferenceable(4) %24)
          to label %207 unwind label %222

207:                                              ; preds = %204
  call void @llvm.lifetime.end.p0(ptr nonnull %22)
  call void @llvm.lifetime.end.p0(ptr nonnull %23)
  call void @llvm.lifetime.end.p0(ptr nonnull %24)
  %208 = icmp eq i32 %198, %206
  br i1 %208, label %209, label %217

209:                                              ; preds = %207, %209
  %210 = phi i64 [ %215, %209 ], [ 0, %207 ]
  %211 = getelementptr inbounds nuw i32, ptr %64, i64 %210
  %212 = getelementptr inbounds nuw i8, ptr %211, i64 16
  store <4 x i32> splat (i32 -2147483648), ptr %211, align 4, !tbaa !43
  store <4 x i32> splat (i32 -2147483648), ptr %212, align 4, !tbaa !43
  %213 = getelementptr inbounds nuw i32, ptr %63, i64 %210
  %214 = getelementptr inbounds nuw i8, ptr %213, i64 16
  store <4 x i32> splat (i32 -2147483648), ptr %213, align 4, !tbaa !43
  store <4 x i32> splat (i32 -2147483648), ptr %214, align 4, !tbaa !43
  %215 = add nuw i64 %210, 8
  %216 = icmp eq i64 %215, 1000
  br i1 %216, label %224, label %209, !llvm.loop !52

217:                                              ; preds = %207
  %218 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.14)
          to label %219 unwind label %222

219:                                              ; preds = %217
  call void @exit(i32 noundef 1) #25
  unreachable

220:                                              ; preds = %196, %194
  %221 = landingpad { ptr, i32 }
          cleanup
  br label %317

222:                                              ; preds = %204, %202, %217
  %223 = landingpad { ptr, i32 }
          cleanup
  br label %317

224:                                              ; preds = %209
  store i32 2147483647, ptr %63, align 4, !tbaa !43
  call void @llvm.lifetime.start.p0(ptr nonnull %19)
  call void @llvm.lifetime.start.p0(ptr nonnull %20)
  call void @llvm.lifetime.start.p0(ptr nonnull %21)
  store ptr %63, ptr %19, align 8, !tbaa !45
  store ptr %64, ptr %20, align 8, !tbaa !45
  store i32 1000, ptr %21, align 4, !tbaa !43
  %225 = load ptr, ptr %82, align 8, !tbaa !20
  %226 = icmp eq ptr %225, null
  br i1 %226, label %227, label %229

227:                                              ; preds = %224
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %228 unwind label %253

228:                                              ; preds = %227
  unreachable

229:                                              ; preds = %224
  %230 = load ptr, ptr %88, align 8, !tbaa !16
  %231 = invoke noundef i32 %230(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %19, ptr noundef nonnull align 8 dereferenceable(8) %20, ptr noundef nonnull align 4 dereferenceable(4) %21)
          to label %232 unwind label %253

232:                                              ; preds = %229
  call void @llvm.lifetime.end.p0(ptr nonnull %19)
  call void @llvm.lifetime.end.p0(ptr nonnull %20)
  call void @llvm.lifetime.end.p0(ptr nonnull %21)
  call void @llvm.lifetime.start.p0(ptr nonnull %16)
  call void @llvm.lifetime.start.p0(ptr nonnull %17)
  call void @llvm.lifetime.start.p0(ptr nonnull %18)
  store ptr %63, ptr %16, align 8, !tbaa !45
  store ptr %64, ptr %17, align 8, !tbaa !45
  store i32 1000, ptr %18, align 4, !tbaa !43
  %233 = load ptr, ptr %92, align 8, !tbaa !20
  %234 = icmp eq ptr %233, null
  br i1 %234, label %235, label %237

235:                                              ; preds = %232
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %236 unwind label %255

236:                                              ; preds = %235
  unreachable

237:                                              ; preds = %232
  %238 = load ptr, ptr %98, align 8, !tbaa !16
  %239 = invoke noundef i32 %238(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %16, ptr noundef nonnull align 8 dereferenceable(8) %17, ptr noundef nonnull align 4 dereferenceable(4) %18)
          to label %240 unwind label %255

240:                                              ; preds = %237
  call void @llvm.lifetime.end.p0(ptr nonnull %16)
  call void @llvm.lifetime.end.p0(ptr nonnull %17)
  call void @llvm.lifetime.end.p0(ptr nonnull %18)
  %241 = icmp eq i32 %231, %239
  br i1 %241, label %242, label %250

242:                                              ; preds = %240, %242
  %243 = phi i64 [ %248, %242 ], [ 0, %240 ]
  %244 = getelementptr inbounds nuw i32, ptr %64, i64 %243
  %245 = getelementptr inbounds nuw i8, ptr %244, i64 16
  store <4 x i32> splat (i32 -2147483648), ptr %244, align 4, !tbaa !43
  store <4 x i32> splat (i32 -2147483648), ptr %245, align 4, !tbaa !43
  %246 = getelementptr inbounds nuw i32, ptr %63, i64 %243
  %247 = getelementptr inbounds nuw i8, ptr %246, i64 16
  store <4 x i32> splat (i32 -2147483648), ptr %246, align 4, !tbaa !43
  store <4 x i32> splat (i32 -2147483648), ptr %247, align 4, !tbaa !43
  %248 = add nuw i64 %243, 8
  %249 = icmp eq i64 %248, 1000
  br i1 %249, label %257, label %242, !llvm.loop !53

250:                                              ; preds = %240
  %251 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.14)
          to label %252 unwind label %255

252:                                              ; preds = %250
  call void @exit(i32 noundef 1) #25
  unreachable

253:                                              ; preds = %229, %227
  %254 = landingpad { ptr, i32 }
          cleanup
  br label %317

255:                                              ; preds = %237, %235, %250
  %256 = landingpad { ptr, i32 }
          cleanup
  br label %317

257:                                              ; preds = %242
  %258 = getelementptr inbounds nuw i8, ptr %63, i64 3996
  store i32 2147483647, ptr %258, align 4, !tbaa !43
  call void @llvm.lifetime.start.p0(ptr nonnull %13)
  call void @llvm.lifetime.start.p0(ptr nonnull %14)
  call void @llvm.lifetime.start.p0(ptr nonnull %15)
  store ptr %63, ptr %13, align 8, !tbaa !45
  store ptr %64, ptr %14, align 8, !tbaa !45
  store i32 1000, ptr %15, align 4, !tbaa !43
  %259 = load ptr, ptr %82, align 8, !tbaa !20
  %260 = icmp eq ptr %259, null
  br i1 %260, label %261, label %263

261:                                              ; preds = %257
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %262 unwind label %287

262:                                              ; preds = %261
  unreachable

263:                                              ; preds = %257
  %264 = load ptr, ptr %88, align 8, !tbaa !16
  %265 = invoke noundef i32 %264(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %13, ptr noundef nonnull align 8 dereferenceable(8) %14, ptr noundef nonnull align 4 dereferenceable(4) %15)
          to label %266 unwind label %287

266:                                              ; preds = %263
  call void @llvm.lifetime.end.p0(ptr nonnull %13)
  call void @llvm.lifetime.end.p0(ptr nonnull %14)
  call void @llvm.lifetime.end.p0(ptr nonnull %15)
  call void @llvm.lifetime.start.p0(ptr nonnull %10)
  call void @llvm.lifetime.start.p0(ptr nonnull %11)
  call void @llvm.lifetime.start.p0(ptr nonnull %12)
  store ptr %63, ptr %10, align 8, !tbaa !45
  store ptr %64, ptr %11, align 8, !tbaa !45
  store i32 1000, ptr %12, align 4, !tbaa !43
  %267 = load ptr, ptr %92, align 8, !tbaa !20
  %268 = icmp eq ptr %267, null
  br i1 %268, label %269, label %271

269:                                              ; preds = %266
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %270 unwind label %289

270:                                              ; preds = %269
  unreachable

271:                                              ; preds = %266
  %272 = load ptr, ptr %98, align 8, !tbaa !16
  %273 = invoke noundef i32 %272(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %10, ptr noundef nonnull align 8 dereferenceable(8) %11, ptr noundef nonnull align 4 dereferenceable(4) %12)
          to label %274 unwind label %289

274:                                              ; preds = %271
  call void @llvm.lifetime.end.p0(ptr nonnull %10)
  call void @llvm.lifetime.end.p0(ptr nonnull %11)
  call void @llvm.lifetime.end.p0(ptr nonnull %12)
  %275 = icmp eq i32 %265, %273
  br i1 %275, label %276, label %284

276:                                              ; preds = %274, %276
  %277 = phi i64 [ %282, %276 ], [ 0, %274 ]
  %278 = getelementptr inbounds nuw i32, ptr %64, i64 %277
  %279 = getelementptr inbounds nuw i8, ptr %278, i64 16
  store <4 x i32> splat (i32 -2147483648), ptr %278, align 4, !tbaa !43
  store <4 x i32> splat (i32 -2147483648), ptr %279, align 4, !tbaa !43
  %280 = getelementptr inbounds nuw i32, ptr %63, i64 %277
  %281 = getelementptr inbounds nuw i8, ptr %280, i64 16
  store <4 x i32> splat (i32 -2147483648), ptr %280, align 4, !tbaa !43
  store <4 x i32> splat (i32 -2147483648), ptr %281, align 4, !tbaa !43
  %282 = add nuw i64 %277, 8
  %283 = icmp eq i64 %282, 1000
  br i1 %283, label %291, label %276, !llvm.loop !54

284:                                              ; preds = %274
  %285 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.14)
          to label %286 unwind label %289

286:                                              ; preds = %284
  call void @exit(i32 noundef 1) #25
  unreachable

287:                                              ; preds = %263, %261
  %288 = landingpad { ptr, i32 }
          cleanup
  br label %317

289:                                              ; preds = %271, %269, %284
  %290 = landingpad { ptr, i32 }
          cleanup
  br label %317

291:                                              ; preds = %276
  store i32 2147483647, ptr %258, align 4, !tbaa !43
  store i32 2147483647, ptr %63, align 4, !tbaa !43
  call void @llvm.lifetime.start.p0(ptr nonnull %7)
  call void @llvm.lifetime.start.p0(ptr nonnull %8)
  call void @llvm.lifetime.start.p0(ptr nonnull %9)
  store ptr %63, ptr %7, align 8, !tbaa !45
  store ptr %64, ptr %8, align 8, !tbaa !45
  store i32 1000, ptr %9, align 4, !tbaa !43
  %292 = load ptr, ptr %82, align 8, !tbaa !20
  %293 = icmp eq ptr %292, null
  br i1 %293, label %294, label %296

294:                                              ; preds = %291
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %295 unwind label %312

295:                                              ; preds = %294
  unreachable

296:                                              ; preds = %291
  %297 = load ptr, ptr %88, align 8, !tbaa !16
  %298 = invoke noundef i32 %297(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %7, ptr noundef nonnull align 8 dereferenceable(8) %8, ptr noundef nonnull align 4 dereferenceable(4) %9)
          to label %299 unwind label %312

299:                                              ; preds = %296
  call void @llvm.lifetime.end.p0(ptr nonnull %7)
  call void @llvm.lifetime.end.p0(ptr nonnull %8)
  call void @llvm.lifetime.end.p0(ptr nonnull %9)
  call void @llvm.lifetime.start.p0(ptr nonnull %4)
  call void @llvm.lifetime.start.p0(ptr nonnull %5)
  call void @llvm.lifetime.start.p0(ptr nonnull %6)
  store ptr %63, ptr %4, align 8, !tbaa !45
  store ptr %64, ptr %5, align 8, !tbaa !45
  store i32 1000, ptr %6, align 4, !tbaa !43
  %300 = load ptr, ptr %92, align 8, !tbaa !20
  %301 = icmp eq ptr %300, null
  br i1 %301, label %302, label %304

302:                                              ; preds = %299
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %303 unwind label %314

303:                                              ; preds = %302
  unreachable

304:                                              ; preds = %299
  %305 = load ptr, ptr %98, align 8, !tbaa !16
  %306 = invoke noundef i32 %305(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %4, ptr noundef nonnull align 8 dereferenceable(8) %5, ptr noundef nonnull align 4 dereferenceable(4) %6)
          to label %307 unwind label %314

307:                                              ; preds = %304
  call void @llvm.lifetime.end.p0(ptr nonnull %4)
  call void @llvm.lifetime.end.p0(ptr nonnull %5)
  call void @llvm.lifetime.end.p0(ptr nonnull %6)
  %308 = icmp eq i32 %298, %306
  br i1 %308, label %316, label %309

309:                                              ; preds = %307
  %310 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.14)
          to label %311 unwind label %314

311:                                              ; preds = %309
  call void @exit(i32 noundef 1) #25
  unreachable

312:                                              ; preds = %296, %294
  %313 = landingpad { ptr, i32 }
          cleanup
  br label %317

314:                                              ; preds = %304, %302, %309
  %315 = landingpad { ptr, i32 }
          cleanup
  br label %317

316:                                              ; preds = %307
  call void @_ZdaPv(ptr noundef nonnull %64) #26
  call void @_ZdaPv(ptr noundef nonnull %63) #26
  ret void

317:                                              ; preds = %116, %118, %312, %314, %287, %289, %253, %255, %220, %222, %186, %188, %153, %155, %120, %122
  %318 = phi { ptr, i32 } [ %123, %122 ], [ %121, %120 ], [ %156, %155 ], [ %154, %153 ], [ %189, %188 ], [ %187, %186 ], [ %223, %222 ], [ %221, %220 ], [ %256, %255 ], [ %254, %253 ], [ %290, %289 ], [ %288, %287 ], [ %315, %314 ], [ %313, %312 ], [ %117, %116 ], [ %119, %118 ]
  call void @_ZdaPv(ptr noundef nonnull %64) #26
  br label %319

319:                                              ; preds = %317, %114
  %320 = phi { ptr, i32 } [ %318, %317 ], [ %115, %114 ]
  call void @_ZdaPv(ptr noundef nonnull %63) #26
  resume { ptr, i32 } %320
}

declare i32 @__gxx_personality_v0(...)

; Function Attrs: mustprogress norecurse uwtable
define internal fastcc void @_ZL19checkVectorFunctionIifEvSt8functionIFT_PT0_S3_jEES5_PKc(ptr noundef nonnull %0, ptr noundef nonnull %1, ptr noundef %2) unnamed_addr #0 personality ptr @__gxx_personality_v0 {
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
  %40 = alloca ptr, align 8
  %41 = alloca ptr, align 8
  %42 = alloca i32, align 4
  %43 = alloca ptr, align 8
  %44 = alloca ptr, align 8
  %45 = alloca i32, align 4
  %46 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.12, i64 noundef 9)
  %47 = icmp eq ptr %2, null
  br i1 %47, label %48, label %56

48:                                               ; preds = %3
  %49 = load ptr, ptr @_ZSt4cout, align 8, !tbaa !31
  %50 = getelementptr i8, ptr %49, i64 -24
  %51 = load i64, ptr %50, align 8
  %52 = getelementptr inbounds i8, ptr @_ZSt4cout, i64 %51
  %53 = getelementptr inbounds nuw i8, ptr %52, i64 32
  %54 = load i32, ptr %53, align 8, !tbaa !33
  %55 = or i32 %54, 1
  tail call void @_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate(ptr noundef nonnull align 8 dereferenceable(264) %52, i32 noundef %55)
  br label %59

56:                                               ; preds = %3
  %57 = tail call noundef i64 @strlen(ptr noundef nonnull dereferenceable(1) %2) #21
  %58 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull %2, i64 noundef %57)
  br label %59

59:                                               ; preds = %48, %56
  %60 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.13, i64 noundef 1)
  %61 = tail call noalias noundef nonnull dereferenceable(4000) ptr @_Znam(i64 noundef 4000) #23
  %62 = invoke noalias noundef nonnull dereferenceable(4000) ptr @_Znam(i64 noundef 4000) #23
          to label %63 unwind label %96

63:                                               ; preds = %59
  tail call fastcc void @_ZL9init_dataIfEvRKSt10unique_ptrIA_T_St14default_deleteIS2_EEj(ptr nonnull %61)
  tail call fastcc void @_ZL9init_dataIfEvRKSt10unique_ptrIA_T_St14default_deleteIS2_EEj(ptr nonnull %62)
  call void @llvm.lifetime.start.p0(ptr nonnull %43)
  call void @llvm.lifetime.start.p0(ptr nonnull %44)
  call void @llvm.lifetime.start.p0(ptr nonnull %45)
  store ptr %61, ptr %43, align 8, !tbaa !55
  store ptr %62, ptr %44, align 8, !tbaa !55
  store i32 1000, ptr %45, align 4, !tbaa !43
  %64 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %65 = load ptr, ptr %64, align 8, !tbaa !20
  %66 = icmp eq ptr %65, null
  br i1 %66, label %67, label %69

67:                                               ; preds = %63
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %68 unwind label %98

68:                                               ; preds = %67
  unreachable

69:                                               ; preds = %63
  %70 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %71 = load ptr, ptr %70, align 8, !tbaa !21
  %72 = invoke noundef i32 %71(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %43, ptr noundef nonnull align 8 dereferenceable(8) %44, ptr noundef nonnull align 4 dereferenceable(4) %45)
          to label %73 unwind label %98

73:                                               ; preds = %69
  call void @llvm.lifetime.end.p0(ptr nonnull %43)
  call void @llvm.lifetime.end.p0(ptr nonnull %44)
  call void @llvm.lifetime.end.p0(ptr nonnull %45)
  call void @llvm.lifetime.start.p0(ptr nonnull %40)
  call void @llvm.lifetime.start.p0(ptr nonnull %41)
  call void @llvm.lifetime.start.p0(ptr nonnull %42)
  store ptr %61, ptr %40, align 8, !tbaa !55
  store ptr %62, ptr %41, align 8, !tbaa !55
  store i32 1000, ptr %42, align 4, !tbaa !43
  %74 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %75 = load ptr, ptr %74, align 8, !tbaa !20
  %76 = icmp eq ptr %75, null
  br i1 %76, label %77, label %79

77:                                               ; preds = %73
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %78 unwind label %100

78:                                               ; preds = %77
  unreachable

79:                                               ; preds = %73
  %80 = getelementptr inbounds nuw i8, ptr %1, i64 24
  %81 = load ptr, ptr %80, align 8, !tbaa !21
  %82 = invoke noundef i32 %81(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %40, ptr noundef nonnull align 8 dereferenceable(8) %41, ptr noundef nonnull align 4 dereferenceable(4) %42)
          to label %83 unwind label %100

83:                                               ; preds = %79
  call void @llvm.lifetime.end.p0(ptr nonnull %40)
  call void @llvm.lifetime.end.p0(ptr nonnull %41)
  call void @llvm.lifetime.end.p0(ptr nonnull %42)
  %84 = icmp eq i32 %72, %82
  br i1 %84, label %85, label %93

85:                                               ; preds = %83, %85
  %86 = phi i64 [ %91, %85 ], [ 0, %83 ]
  %87 = getelementptr inbounds nuw float, ptr %61, i64 %86
  %88 = getelementptr inbounds nuw i8, ptr %87, i64 16
  store <4 x float> splat (float 0x47EFFFFFE0000000), ptr %87, align 4, !tbaa !57
  store <4 x float> splat (float 0x47EFFFFFE0000000), ptr %88, align 4, !tbaa !57
  %89 = getelementptr inbounds nuw float, ptr %62, i64 %86
  %90 = getelementptr inbounds nuw i8, ptr %89, i64 16
  store <4 x float> splat (float 0x3810000000000000), ptr %89, align 4, !tbaa !57
  store <4 x float> splat (float 0x3810000000000000), ptr %90, align 4, !tbaa !57
  %91 = add nuw i64 %86, 8
  %92 = icmp eq i64 %91, 1000
  br i1 %92, label %102, label %85, !llvm.loop !59

93:                                               ; preds = %83
  %94 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.14)
          to label %95 unwind label %100

95:                                               ; preds = %93
  call void @exit(i32 noundef 1) #25
  unreachable

96:                                               ; preds = %59
  %97 = landingpad { ptr, i32 }
          cleanup
  br label %297

98:                                               ; preds = %69, %67
  %99 = landingpad { ptr, i32 }
          cleanup
  br label %295

100:                                              ; preds = %79, %77, %93
  %101 = landingpad { ptr, i32 }
          cleanup
  br label %295

102:                                              ; preds = %85
  call void @llvm.lifetime.start.p0(ptr nonnull %37)
  call void @llvm.lifetime.start.p0(ptr nonnull %38)
  call void @llvm.lifetime.start.p0(ptr nonnull %39)
  store ptr %61, ptr %37, align 8, !tbaa !55
  store ptr %62, ptr %38, align 8, !tbaa !55
  store i32 1000, ptr %39, align 4, !tbaa !43
  %103 = load ptr, ptr %64, align 8, !tbaa !20
  %104 = icmp eq ptr %103, null
  br i1 %104, label %105, label %107

105:                                              ; preds = %102
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %106 unwind label %131

106:                                              ; preds = %105
  unreachable

107:                                              ; preds = %102
  %108 = load ptr, ptr %70, align 8, !tbaa !21
  %109 = invoke noundef i32 %108(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %37, ptr noundef nonnull align 8 dereferenceable(8) %38, ptr noundef nonnull align 4 dereferenceable(4) %39)
          to label %110 unwind label %131

110:                                              ; preds = %107
  call void @llvm.lifetime.end.p0(ptr nonnull %37)
  call void @llvm.lifetime.end.p0(ptr nonnull %38)
  call void @llvm.lifetime.end.p0(ptr nonnull %39)
  call void @llvm.lifetime.start.p0(ptr nonnull %34)
  call void @llvm.lifetime.start.p0(ptr nonnull %35)
  call void @llvm.lifetime.start.p0(ptr nonnull %36)
  store ptr %61, ptr %34, align 8, !tbaa !55
  store ptr %62, ptr %35, align 8, !tbaa !55
  store i32 1000, ptr %36, align 4, !tbaa !43
  %111 = load ptr, ptr %74, align 8, !tbaa !20
  %112 = icmp eq ptr %111, null
  br i1 %112, label %113, label %115

113:                                              ; preds = %110
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %114 unwind label %133

114:                                              ; preds = %113
  unreachable

115:                                              ; preds = %110
  %116 = load ptr, ptr %80, align 8, !tbaa !21
  %117 = invoke noundef i32 %116(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %34, ptr noundef nonnull align 8 dereferenceable(8) %35, ptr noundef nonnull align 4 dereferenceable(4) %36)
          to label %118 unwind label %133

118:                                              ; preds = %115
  call void @llvm.lifetime.end.p0(ptr nonnull %34)
  call void @llvm.lifetime.end.p0(ptr nonnull %35)
  call void @llvm.lifetime.end.p0(ptr nonnull %36)
  %119 = icmp eq i32 %109, %117
  br i1 %119, label %120, label %128

120:                                              ; preds = %118, %120
  %121 = phi i64 [ %126, %120 ], [ 0, %118 ]
  %122 = getelementptr inbounds nuw float, ptr %61, i64 %121
  %123 = getelementptr inbounds nuw i8, ptr %122, i64 16
  store <4 x float> splat (float 0x3810000000000000), ptr %122, align 4, !tbaa !57
  store <4 x float> splat (float 0x3810000000000000), ptr %123, align 4, !tbaa !57
  %124 = getelementptr inbounds nuw float, ptr %62, i64 %121
  %125 = getelementptr inbounds nuw i8, ptr %124, i64 16
  store <4 x float> splat (float 0x47EFFFFFE0000000), ptr %124, align 4, !tbaa !57
  store <4 x float> splat (float 0x47EFFFFFE0000000), ptr %125, align 4, !tbaa !57
  %126 = add nuw i64 %121, 8
  %127 = icmp eq i64 %126, 1000
  br i1 %127, label %135, label %120, !llvm.loop !60

128:                                              ; preds = %118
  %129 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.14)
          to label %130 unwind label %133

130:                                              ; preds = %128
  call void @exit(i32 noundef 1) #25
  unreachable

131:                                              ; preds = %107, %105
  %132 = landingpad { ptr, i32 }
          cleanup
  br label %295

133:                                              ; preds = %115, %113, %128
  %134 = landingpad { ptr, i32 }
          cleanup
  br label %295

135:                                              ; preds = %120
  call void @llvm.lifetime.start.p0(ptr nonnull %31)
  call void @llvm.lifetime.start.p0(ptr nonnull %32)
  call void @llvm.lifetime.start.p0(ptr nonnull %33)
  store ptr %61, ptr %31, align 8, !tbaa !55
  store ptr %62, ptr %32, align 8, !tbaa !55
  store i32 1000, ptr %33, align 4, !tbaa !43
  %136 = load ptr, ptr %64, align 8, !tbaa !20
  %137 = icmp eq ptr %136, null
  br i1 %137, label %138, label %140

138:                                              ; preds = %135
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %139 unwind label %164

139:                                              ; preds = %138
  unreachable

140:                                              ; preds = %135
  %141 = load ptr, ptr %70, align 8, !tbaa !21
  %142 = invoke noundef i32 %141(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %31, ptr noundef nonnull align 8 dereferenceable(8) %32, ptr noundef nonnull align 4 dereferenceable(4) %33)
          to label %143 unwind label %164

143:                                              ; preds = %140
  call void @llvm.lifetime.end.p0(ptr nonnull %31)
  call void @llvm.lifetime.end.p0(ptr nonnull %32)
  call void @llvm.lifetime.end.p0(ptr nonnull %33)
  call void @llvm.lifetime.start.p0(ptr nonnull %28)
  call void @llvm.lifetime.start.p0(ptr nonnull %29)
  call void @llvm.lifetime.start.p0(ptr nonnull %30)
  store ptr %61, ptr %28, align 8, !tbaa !55
  store ptr %62, ptr %29, align 8, !tbaa !55
  store i32 1000, ptr %30, align 4, !tbaa !43
  %144 = load ptr, ptr %74, align 8, !tbaa !20
  %145 = icmp eq ptr %144, null
  br i1 %145, label %146, label %148

146:                                              ; preds = %143
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %147 unwind label %166

147:                                              ; preds = %146
  unreachable

148:                                              ; preds = %143
  %149 = load ptr, ptr %80, align 8, !tbaa !21
  %150 = invoke noundef i32 %149(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %28, ptr noundef nonnull align 8 dereferenceable(8) %29, ptr noundef nonnull align 4 dereferenceable(4) %30)
          to label %151 unwind label %166

151:                                              ; preds = %148
  call void @llvm.lifetime.end.p0(ptr nonnull %28)
  call void @llvm.lifetime.end.p0(ptr nonnull %29)
  call void @llvm.lifetime.end.p0(ptr nonnull %30)
  %152 = icmp eq i32 %142, %150
  br i1 %152, label %153, label %161

153:                                              ; preds = %151, %153
  %154 = phi i64 [ %159, %153 ], [ 0, %151 ]
  %155 = getelementptr inbounds nuw float, ptr %62, i64 %154
  %156 = getelementptr inbounds nuw i8, ptr %155, i64 16
  store <4 x float> splat (float 0x3810000000000000), ptr %155, align 4, !tbaa !57
  store <4 x float> splat (float 0x3810000000000000), ptr %156, align 4, !tbaa !57
  %157 = getelementptr inbounds nuw float, ptr %61, i64 %154
  %158 = getelementptr inbounds nuw i8, ptr %157, i64 16
  store <4 x float> splat (float 0x3810000000000000), ptr %157, align 4, !tbaa !57
  store <4 x float> splat (float 0x3810000000000000), ptr %158, align 4, !tbaa !57
  %159 = add nuw i64 %154, 8
  %160 = icmp eq i64 %159, 1000
  br i1 %160, label %168, label %153, !llvm.loop !61

161:                                              ; preds = %151
  %162 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.14)
          to label %163 unwind label %166

163:                                              ; preds = %161
  call void @exit(i32 noundef 1) #25
  unreachable

164:                                              ; preds = %140, %138
  %165 = landingpad { ptr, i32 }
          cleanup
  br label %295

166:                                              ; preds = %148, %146, %161
  %167 = landingpad { ptr, i32 }
          cleanup
  br label %295

168:                                              ; preds = %153
  %169 = getelementptr inbounds nuw i8, ptr %61, i64 3992
  store float 0x47EFFFFFE0000000, ptr %169, align 4, !tbaa !57
  call void @llvm.lifetime.start.p0(ptr nonnull %25)
  call void @llvm.lifetime.start.p0(ptr nonnull %26)
  call void @llvm.lifetime.start.p0(ptr nonnull %27)
  store ptr %61, ptr %25, align 8, !tbaa !55
  store ptr %62, ptr %26, align 8, !tbaa !55
  store i32 1000, ptr %27, align 4, !tbaa !43
  %170 = load ptr, ptr %64, align 8, !tbaa !20
  %171 = icmp eq ptr %170, null
  br i1 %171, label %172, label %174

172:                                              ; preds = %168
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %173 unwind label %198

173:                                              ; preds = %172
  unreachable

174:                                              ; preds = %168
  %175 = load ptr, ptr %70, align 8, !tbaa !21
  %176 = invoke noundef i32 %175(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %25, ptr noundef nonnull align 8 dereferenceable(8) %26, ptr noundef nonnull align 4 dereferenceable(4) %27)
          to label %177 unwind label %198

177:                                              ; preds = %174
  call void @llvm.lifetime.end.p0(ptr nonnull %25)
  call void @llvm.lifetime.end.p0(ptr nonnull %26)
  call void @llvm.lifetime.end.p0(ptr nonnull %27)
  call void @llvm.lifetime.start.p0(ptr nonnull %22)
  call void @llvm.lifetime.start.p0(ptr nonnull %23)
  call void @llvm.lifetime.start.p0(ptr nonnull %24)
  store ptr %61, ptr %22, align 8, !tbaa !55
  store ptr %62, ptr %23, align 8, !tbaa !55
  store i32 1000, ptr %24, align 4, !tbaa !43
  %178 = load ptr, ptr %74, align 8, !tbaa !20
  %179 = icmp eq ptr %178, null
  br i1 %179, label %180, label %182

180:                                              ; preds = %177
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %181 unwind label %200

181:                                              ; preds = %180
  unreachable

182:                                              ; preds = %177
  %183 = load ptr, ptr %80, align 8, !tbaa !21
  %184 = invoke noundef i32 %183(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %22, ptr noundef nonnull align 8 dereferenceable(8) %23, ptr noundef nonnull align 4 dereferenceable(4) %24)
          to label %185 unwind label %200

185:                                              ; preds = %182
  call void @llvm.lifetime.end.p0(ptr nonnull %22)
  call void @llvm.lifetime.end.p0(ptr nonnull %23)
  call void @llvm.lifetime.end.p0(ptr nonnull %24)
  %186 = icmp eq i32 %176, %184
  br i1 %186, label %187, label %195

187:                                              ; preds = %185, %187
  %188 = phi i64 [ %193, %187 ], [ 0, %185 ]
  %189 = getelementptr inbounds nuw float, ptr %62, i64 %188
  %190 = getelementptr inbounds nuw i8, ptr %189, i64 16
  store <4 x float> splat (float 0x3810000000000000), ptr %189, align 4, !tbaa !57
  store <4 x float> splat (float 0x3810000000000000), ptr %190, align 4, !tbaa !57
  %191 = getelementptr inbounds nuw float, ptr %61, i64 %188
  %192 = getelementptr inbounds nuw i8, ptr %191, i64 16
  store <4 x float> splat (float 0x3810000000000000), ptr %191, align 4, !tbaa !57
  store <4 x float> splat (float 0x3810000000000000), ptr %192, align 4, !tbaa !57
  %193 = add nuw i64 %188, 8
  %194 = icmp eq i64 %193, 1000
  br i1 %194, label %202, label %187, !llvm.loop !62

195:                                              ; preds = %185
  %196 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.14)
          to label %197 unwind label %200

197:                                              ; preds = %195
  call void @exit(i32 noundef 1) #25
  unreachable

198:                                              ; preds = %174, %172
  %199 = landingpad { ptr, i32 }
          cleanup
  br label %295

200:                                              ; preds = %182, %180, %195
  %201 = landingpad { ptr, i32 }
          cleanup
  br label %295

202:                                              ; preds = %187
  store float 0x47EFFFFFE0000000, ptr %61, align 4, !tbaa !57
  call void @llvm.lifetime.start.p0(ptr nonnull %19)
  call void @llvm.lifetime.start.p0(ptr nonnull %20)
  call void @llvm.lifetime.start.p0(ptr nonnull %21)
  store ptr %61, ptr %19, align 8, !tbaa !55
  store ptr %62, ptr %20, align 8, !tbaa !55
  store i32 1000, ptr %21, align 4, !tbaa !43
  %203 = load ptr, ptr %64, align 8, !tbaa !20
  %204 = icmp eq ptr %203, null
  br i1 %204, label %205, label %207

205:                                              ; preds = %202
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %206 unwind label %231

206:                                              ; preds = %205
  unreachable

207:                                              ; preds = %202
  %208 = load ptr, ptr %70, align 8, !tbaa !21
  %209 = invoke noundef i32 %208(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %19, ptr noundef nonnull align 8 dereferenceable(8) %20, ptr noundef nonnull align 4 dereferenceable(4) %21)
          to label %210 unwind label %231

210:                                              ; preds = %207
  call void @llvm.lifetime.end.p0(ptr nonnull %19)
  call void @llvm.lifetime.end.p0(ptr nonnull %20)
  call void @llvm.lifetime.end.p0(ptr nonnull %21)
  call void @llvm.lifetime.start.p0(ptr nonnull %16)
  call void @llvm.lifetime.start.p0(ptr nonnull %17)
  call void @llvm.lifetime.start.p0(ptr nonnull %18)
  store ptr %61, ptr %16, align 8, !tbaa !55
  store ptr %62, ptr %17, align 8, !tbaa !55
  store i32 1000, ptr %18, align 4, !tbaa !43
  %211 = load ptr, ptr %74, align 8, !tbaa !20
  %212 = icmp eq ptr %211, null
  br i1 %212, label %213, label %215

213:                                              ; preds = %210
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %214 unwind label %233

214:                                              ; preds = %213
  unreachable

215:                                              ; preds = %210
  %216 = load ptr, ptr %80, align 8, !tbaa !21
  %217 = invoke noundef i32 %216(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %16, ptr noundef nonnull align 8 dereferenceable(8) %17, ptr noundef nonnull align 4 dereferenceable(4) %18)
          to label %218 unwind label %233

218:                                              ; preds = %215
  call void @llvm.lifetime.end.p0(ptr nonnull %16)
  call void @llvm.lifetime.end.p0(ptr nonnull %17)
  call void @llvm.lifetime.end.p0(ptr nonnull %18)
  %219 = icmp eq i32 %209, %217
  br i1 %219, label %220, label %228

220:                                              ; preds = %218, %220
  %221 = phi i64 [ %226, %220 ], [ 0, %218 ]
  %222 = getelementptr inbounds nuw float, ptr %62, i64 %221
  %223 = getelementptr inbounds nuw i8, ptr %222, i64 16
  store <4 x float> splat (float 0x3810000000000000), ptr %222, align 4, !tbaa !57
  store <4 x float> splat (float 0x3810000000000000), ptr %223, align 4, !tbaa !57
  %224 = getelementptr inbounds nuw float, ptr %61, i64 %221
  %225 = getelementptr inbounds nuw i8, ptr %224, i64 16
  store <4 x float> splat (float 0x3810000000000000), ptr %224, align 4, !tbaa !57
  store <4 x float> splat (float 0x3810000000000000), ptr %225, align 4, !tbaa !57
  %226 = add nuw i64 %221, 8
  %227 = icmp eq i64 %226, 1000
  br i1 %227, label %235, label %220, !llvm.loop !63

228:                                              ; preds = %218
  %229 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.14)
          to label %230 unwind label %233

230:                                              ; preds = %228
  call void @exit(i32 noundef 1) #25
  unreachable

231:                                              ; preds = %207, %205
  %232 = landingpad { ptr, i32 }
          cleanup
  br label %295

233:                                              ; preds = %215, %213, %228
  %234 = landingpad { ptr, i32 }
          cleanup
  br label %295

235:                                              ; preds = %220
  %236 = getelementptr inbounds nuw i8, ptr %61, i64 3996
  store float 0x47EFFFFFE0000000, ptr %236, align 4, !tbaa !57
  call void @llvm.lifetime.start.p0(ptr nonnull %13)
  call void @llvm.lifetime.start.p0(ptr nonnull %14)
  call void @llvm.lifetime.start.p0(ptr nonnull %15)
  store ptr %61, ptr %13, align 8, !tbaa !55
  store ptr %62, ptr %14, align 8, !tbaa !55
  store i32 1000, ptr %15, align 4, !tbaa !43
  %237 = load ptr, ptr %64, align 8, !tbaa !20
  %238 = icmp eq ptr %237, null
  br i1 %238, label %239, label %241

239:                                              ; preds = %235
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %240 unwind label %265

240:                                              ; preds = %239
  unreachable

241:                                              ; preds = %235
  %242 = load ptr, ptr %70, align 8, !tbaa !21
  %243 = invoke noundef i32 %242(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %13, ptr noundef nonnull align 8 dereferenceable(8) %14, ptr noundef nonnull align 4 dereferenceable(4) %15)
          to label %244 unwind label %265

244:                                              ; preds = %241
  call void @llvm.lifetime.end.p0(ptr nonnull %13)
  call void @llvm.lifetime.end.p0(ptr nonnull %14)
  call void @llvm.lifetime.end.p0(ptr nonnull %15)
  call void @llvm.lifetime.start.p0(ptr nonnull %10)
  call void @llvm.lifetime.start.p0(ptr nonnull %11)
  call void @llvm.lifetime.start.p0(ptr nonnull %12)
  store ptr %61, ptr %10, align 8, !tbaa !55
  store ptr %62, ptr %11, align 8, !tbaa !55
  store i32 1000, ptr %12, align 4, !tbaa !43
  %245 = load ptr, ptr %74, align 8, !tbaa !20
  %246 = icmp eq ptr %245, null
  br i1 %246, label %247, label %249

247:                                              ; preds = %244
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %248 unwind label %267

248:                                              ; preds = %247
  unreachable

249:                                              ; preds = %244
  %250 = load ptr, ptr %80, align 8, !tbaa !21
  %251 = invoke noundef i32 %250(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %10, ptr noundef nonnull align 8 dereferenceable(8) %11, ptr noundef nonnull align 4 dereferenceable(4) %12)
          to label %252 unwind label %267

252:                                              ; preds = %249
  call void @llvm.lifetime.end.p0(ptr nonnull %10)
  call void @llvm.lifetime.end.p0(ptr nonnull %11)
  call void @llvm.lifetime.end.p0(ptr nonnull %12)
  %253 = icmp eq i32 %243, %251
  br i1 %253, label %254, label %262

254:                                              ; preds = %252, %254
  %255 = phi i64 [ %260, %254 ], [ 0, %252 ]
  %256 = getelementptr inbounds nuw float, ptr %62, i64 %255
  %257 = getelementptr inbounds nuw i8, ptr %256, i64 16
  store <4 x float> splat (float 0x3810000000000000), ptr %256, align 4, !tbaa !57
  store <4 x float> splat (float 0x3810000000000000), ptr %257, align 4, !tbaa !57
  %258 = getelementptr inbounds nuw float, ptr %61, i64 %255
  %259 = getelementptr inbounds nuw i8, ptr %258, i64 16
  store <4 x float> splat (float 0x3810000000000000), ptr %258, align 4, !tbaa !57
  store <4 x float> splat (float 0x3810000000000000), ptr %259, align 4, !tbaa !57
  %260 = add nuw i64 %255, 8
  %261 = icmp eq i64 %260, 1000
  br i1 %261, label %269, label %254, !llvm.loop !64

262:                                              ; preds = %252
  %263 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.14)
          to label %264 unwind label %267

264:                                              ; preds = %262
  call void @exit(i32 noundef 1) #25
  unreachable

265:                                              ; preds = %241, %239
  %266 = landingpad { ptr, i32 }
          cleanup
  br label %295

267:                                              ; preds = %249, %247, %262
  %268 = landingpad { ptr, i32 }
          cleanup
  br label %295

269:                                              ; preds = %254
  store float 0x47EFFFFFE0000000, ptr %236, align 4, !tbaa !57
  store float 0x47EFFFFFE0000000, ptr %61, align 4, !tbaa !57
  call void @llvm.lifetime.start.p0(ptr nonnull %7)
  call void @llvm.lifetime.start.p0(ptr nonnull %8)
  call void @llvm.lifetime.start.p0(ptr nonnull %9)
  store ptr %61, ptr %7, align 8, !tbaa !55
  store ptr %62, ptr %8, align 8, !tbaa !55
  store i32 1000, ptr %9, align 4, !tbaa !43
  %270 = load ptr, ptr %64, align 8, !tbaa !20
  %271 = icmp eq ptr %270, null
  br i1 %271, label %272, label %274

272:                                              ; preds = %269
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %273 unwind label %290

273:                                              ; preds = %272
  unreachable

274:                                              ; preds = %269
  %275 = load ptr, ptr %70, align 8, !tbaa !21
  %276 = invoke noundef i32 %275(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %7, ptr noundef nonnull align 8 dereferenceable(8) %8, ptr noundef nonnull align 4 dereferenceable(4) %9)
          to label %277 unwind label %290

277:                                              ; preds = %274
  call void @llvm.lifetime.end.p0(ptr nonnull %7)
  call void @llvm.lifetime.end.p0(ptr nonnull %8)
  call void @llvm.lifetime.end.p0(ptr nonnull %9)
  call void @llvm.lifetime.start.p0(ptr nonnull %4)
  call void @llvm.lifetime.start.p0(ptr nonnull %5)
  call void @llvm.lifetime.start.p0(ptr nonnull %6)
  store ptr %61, ptr %4, align 8, !tbaa !55
  store ptr %62, ptr %5, align 8, !tbaa !55
  store i32 1000, ptr %6, align 4, !tbaa !43
  %278 = load ptr, ptr %74, align 8, !tbaa !20
  %279 = icmp eq ptr %278, null
  br i1 %279, label %280, label %282

280:                                              ; preds = %277
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %281 unwind label %292

281:                                              ; preds = %280
  unreachable

282:                                              ; preds = %277
  %283 = load ptr, ptr %80, align 8, !tbaa !21
  %284 = invoke noundef i32 %283(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %4, ptr noundef nonnull align 8 dereferenceable(8) %5, ptr noundef nonnull align 4 dereferenceable(4) %6)
          to label %285 unwind label %292

285:                                              ; preds = %282
  call void @llvm.lifetime.end.p0(ptr nonnull %4)
  call void @llvm.lifetime.end.p0(ptr nonnull %5)
  call void @llvm.lifetime.end.p0(ptr nonnull %6)
  %286 = icmp eq i32 %276, %284
  br i1 %286, label %294, label %287

287:                                              ; preds = %285
  %288 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.14)
          to label %289 unwind label %292

289:                                              ; preds = %287
  call void @exit(i32 noundef 1) #25
  unreachable

290:                                              ; preds = %274, %272
  %291 = landingpad { ptr, i32 }
          cleanup
  br label %295

292:                                              ; preds = %282, %280, %287
  %293 = landingpad { ptr, i32 }
          cleanup
  br label %295

294:                                              ; preds = %285
  call void @_ZdaPv(ptr noundef nonnull %62) #26
  call void @_ZdaPv(ptr noundef nonnull %61) #26
  ret void

295:                                              ; preds = %290, %292, %265, %267, %231, %233, %198, %200, %164, %166, %131, %133, %98, %100
  %296 = phi { ptr, i32 } [ %101, %100 ], [ %99, %98 ], [ %134, %133 ], [ %132, %131 ], [ %167, %166 ], [ %165, %164 ], [ %201, %200 ], [ %199, %198 ], [ %234, %233 ], [ %232, %231 ], [ %268, %267 ], [ %266, %265 ], [ %293, %292 ], [ %291, %290 ]
  call void @_ZdaPv(ptr noundef nonnull %62) #26
  br label %297

297:                                              ; preds = %295, %96
  %298 = phi { ptr, i32 } [ %296, %295 ], [ %97, %96 ]
  call void @_ZdaPv(ptr noundef nonnull %61) #26
  resume { ptr, i32 } %298
}

; Function Attrs: mustprogress norecurse uwtable
define internal fastcc void @_ZL19checkVectorFunctionIssEvSt8functionIFT_PT0_S3_jEES5_PKc(ptr noundef nonnull %0, ptr noundef nonnull %1, ptr noundef %2) unnamed_addr #0 personality ptr @__gxx_personality_v0 {
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
  %40 = alloca ptr, align 8
  %41 = alloca ptr, align 8
  %42 = alloca i32, align 4
  %43 = alloca ptr, align 8
  %44 = alloca ptr, align 8
  %45 = alloca i32, align 4
  %46 = alloca %"class.std::uniform_int_distribution.62", align 4
  %47 = alloca %"class.std::uniform_int_distribution.62", align 4
  %48 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.12, i64 noundef 9)
  %49 = icmp eq ptr %2, null
  br i1 %49, label %50, label %58

50:                                               ; preds = %3
  %51 = load ptr, ptr @_ZSt4cout, align 8, !tbaa !31
  %52 = getelementptr i8, ptr %51, i64 -24
  %53 = load i64, ptr %52, align 8
  %54 = getelementptr inbounds i8, ptr @_ZSt4cout, i64 %53
  %55 = getelementptr inbounds nuw i8, ptr %54, i64 32
  %56 = load i32, ptr %55, align 8, !tbaa !33
  %57 = or i32 %56, 1
  tail call void @_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate(ptr noundef nonnull align 8 dereferenceable(264) %54, i32 noundef %57)
  br label %61

58:                                               ; preds = %3
  %59 = tail call noundef i64 @strlen(ptr noundef nonnull dereferenceable(1) %2) #21
  %60 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull %2, i64 noundef %59)
  br label %61

61:                                               ; preds = %50, %58
  %62 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.13, i64 noundef 1)
  %63 = tail call noalias noundef nonnull dereferenceable(2000) ptr @_Znam(i64 noundef 2000) #23
  %64 = invoke noalias noundef nonnull dereferenceable(2000) ptr @_Znam(i64 noundef 2000) #23
          to label %65 unwind label %121

65:                                               ; preds = %61
  call void @llvm.lifetime.start.p0(ptr nonnull %47) #21
  store i16 -32768, ptr %47, align 4, !tbaa !65
  %66 = getelementptr inbounds nuw i8, ptr %47, i64 2
  store i16 32767, ptr %66, align 2, !tbaa !68
  br label %67

67:                                               ; preds = %70, %65
  %68 = phi i64 [ 0, %65 ], [ %72, %70 ]
  %69 = invoke noundef i16 @_ZNSt24uniform_int_distributionIsEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEsRT_RKNS0_10param_typeE(ptr noundef nonnull align 2 dereferenceable(4) %47, ptr noundef nonnull align 8 dereferenceable(5000) @_ZL3rng, ptr noundef nonnull align 2 dereferenceable(4) %47)
          to label %70 unwind label %125

70:                                               ; preds = %67
  %71 = getelementptr inbounds nuw i16, ptr %63, i64 %68
  store i16 %69, ptr %71, align 2, !tbaa !69
  %72 = add nuw nsw i64 %68, 1
  %73 = icmp eq i64 %72, 1000
  br i1 %73, label %74, label %67, !llvm.loop !70

74:                                               ; preds = %70
  call void @llvm.lifetime.end.p0(ptr nonnull %47) #21
  call void @llvm.lifetime.start.p0(ptr nonnull %46) #21
  store i16 -32768, ptr %46, align 4, !tbaa !65
  %75 = getelementptr inbounds nuw i8, ptr %46, i64 2
  store i16 32767, ptr %75, align 2, !tbaa !68
  br label %76

76:                                               ; preds = %79, %74
  %77 = phi i64 [ 0, %74 ], [ %81, %79 ]
  %78 = invoke noundef i16 @_ZNSt24uniform_int_distributionIsEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEsRT_RKNS0_10param_typeE(ptr noundef nonnull align 2 dereferenceable(4) %46, ptr noundef nonnull align 8 dereferenceable(5000) @_ZL3rng, ptr noundef nonnull align 2 dereferenceable(4) %46)
          to label %79 unwind label %123

79:                                               ; preds = %76
  %80 = getelementptr inbounds nuw i16, ptr %64, i64 %77
  store i16 %78, ptr %80, align 2, !tbaa !69
  %81 = add nuw nsw i64 %77, 1
  %82 = icmp eq i64 %81, 1000
  br i1 %82, label %83, label %76, !llvm.loop !70

83:                                               ; preds = %79
  call void @llvm.lifetime.end.p0(ptr nonnull %46) #21
  call void @llvm.lifetime.start.p0(ptr nonnull %43)
  call void @llvm.lifetime.start.p0(ptr nonnull %44)
  call void @llvm.lifetime.start.p0(ptr nonnull %45)
  store ptr %63, ptr %43, align 8, !tbaa !71
  store ptr %64, ptr %44, align 8, !tbaa !71
  store i32 1000, ptr %45, align 4, !tbaa !43
  %84 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %85 = load ptr, ptr %84, align 8, !tbaa !20
  %86 = icmp eq ptr %85, null
  br i1 %86, label %87, label %89

87:                                               ; preds = %83
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %88 unwind label %127

88:                                               ; preds = %87
  unreachable

89:                                               ; preds = %83
  %90 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %91 = load ptr, ptr %90, align 8, !tbaa !23
  %92 = invoke noundef i16 %91(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %43, ptr noundef nonnull align 8 dereferenceable(8) %44, ptr noundef nonnull align 4 dereferenceable(4) %45)
          to label %93 unwind label %127

93:                                               ; preds = %89
  call void @llvm.lifetime.end.p0(ptr nonnull %43)
  call void @llvm.lifetime.end.p0(ptr nonnull %44)
  call void @llvm.lifetime.end.p0(ptr nonnull %45)
  call void @llvm.lifetime.start.p0(ptr nonnull %40)
  call void @llvm.lifetime.start.p0(ptr nonnull %41)
  call void @llvm.lifetime.start.p0(ptr nonnull %42)
  store ptr %63, ptr %40, align 8, !tbaa !71
  store ptr %64, ptr %41, align 8, !tbaa !71
  store i32 1000, ptr %42, align 4, !tbaa !43
  %94 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %95 = load ptr, ptr %94, align 8, !tbaa !20
  %96 = icmp eq ptr %95, null
  br i1 %96, label %97, label %99

97:                                               ; preds = %93
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %98 unwind label %129

98:                                               ; preds = %97
  unreachable

99:                                               ; preds = %93
  %100 = getelementptr inbounds nuw i8, ptr %1, i64 24
  %101 = load ptr, ptr %100, align 8, !tbaa !23
  %102 = invoke noundef i16 %101(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %40, ptr noundef nonnull align 8 dereferenceable(8) %41, ptr noundef nonnull align 4 dereferenceable(4) %42)
          to label %103 unwind label %129

103:                                              ; preds = %99
  call void @llvm.lifetime.end.p0(ptr nonnull %40)
  call void @llvm.lifetime.end.p0(ptr nonnull %41)
  call void @llvm.lifetime.end.p0(ptr nonnull %42)
  %104 = icmp eq i16 %92, %102
  br i1 %104, label %105, label %118

105:                                              ; preds = %103, %105
  %106 = phi i64 [ %111, %105 ], [ 0, %103 ]
  %107 = getelementptr inbounds nuw i16, ptr %63, i64 %106
  %108 = getelementptr inbounds nuw i8, ptr %107, i64 16
  store <8 x i16> splat (i16 32767), ptr %107, align 2, !tbaa !69
  store <8 x i16> splat (i16 32767), ptr %108, align 2, !tbaa !69
  %109 = getelementptr inbounds nuw i16, ptr %64, i64 %106
  %110 = getelementptr inbounds nuw i8, ptr %109, i64 16
  store <8 x i16> splat (i16 -32768), ptr %109, align 2, !tbaa !69
  store <8 x i16> splat (i16 -32768), ptr %110, align 2, !tbaa !69
  %111 = add nuw i64 %106, 16
  %112 = icmp eq i64 %111, 992
  br i1 %112, label %113, label %105, !llvm.loop !73

113:                                              ; preds = %105
  %114 = getelementptr inbounds nuw i8, ptr %63, i64 1984
  store <8 x i16> splat (i16 32767), ptr %114, align 2, !tbaa !69
  %115 = getelementptr inbounds nuw i8, ptr %64, i64 1984
  store <8 x i16> splat (i16 -32768), ptr %115, align 2, !tbaa !69
  call void @llvm.lifetime.start.p0(ptr nonnull %37)
  call void @llvm.lifetime.start.p0(ptr nonnull %38)
  call void @llvm.lifetime.start.p0(ptr nonnull %39)
  store ptr %63, ptr %37, align 8, !tbaa !71
  store ptr %64, ptr %38, align 8, !tbaa !71
  store i32 1000, ptr %39, align 4, !tbaa !43
  %116 = load ptr, ptr %84, align 8, !tbaa !20
  %117 = icmp eq ptr %116, null
  br i1 %117, label %131, label %133

118:                                              ; preds = %103
  %119 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.14)
          to label %120 unwind label %129

120:                                              ; preds = %118
  call void @exit(i32 noundef 1) #25
  unreachable

121:                                              ; preds = %61
  %122 = landingpad { ptr, i32 }
          cleanup
  br label %333

123:                                              ; preds = %76
  %124 = landingpad { ptr, i32 }
          cleanup
  br label %331

125:                                              ; preds = %67
  %126 = landingpad { ptr, i32 }
          cleanup
  br label %331

127:                                              ; preds = %89, %87
  %128 = landingpad { ptr, i32 }
          cleanup
  br label %331

129:                                              ; preds = %99, %97, %118
  %130 = landingpad { ptr, i32 }
          cleanup
  br label %331

131:                                              ; preds = %113
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %132 unwind label %162

132:                                              ; preds = %131
  unreachable

133:                                              ; preds = %113
  %134 = load ptr, ptr %90, align 8, !tbaa !23
  %135 = invoke noundef i16 %134(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %37, ptr noundef nonnull align 8 dereferenceable(8) %38, ptr noundef nonnull align 4 dereferenceable(4) %39)
          to label %136 unwind label %162

136:                                              ; preds = %133
  call void @llvm.lifetime.end.p0(ptr nonnull %37)
  call void @llvm.lifetime.end.p0(ptr nonnull %38)
  call void @llvm.lifetime.end.p0(ptr nonnull %39)
  call void @llvm.lifetime.start.p0(ptr nonnull %34)
  call void @llvm.lifetime.start.p0(ptr nonnull %35)
  call void @llvm.lifetime.start.p0(ptr nonnull %36)
  store ptr %63, ptr %34, align 8, !tbaa !71
  store ptr %64, ptr %35, align 8, !tbaa !71
  store i32 1000, ptr %36, align 4, !tbaa !43
  %137 = load ptr, ptr %94, align 8, !tbaa !20
  %138 = icmp eq ptr %137, null
  br i1 %138, label %139, label %141

139:                                              ; preds = %136
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %140 unwind label %164

140:                                              ; preds = %139
  unreachable

141:                                              ; preds = %136
  %142 = load ptr, ptr %100, align 8, !tbaa !23
  %143 = invoke noundef i16 %142(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %34, ptr noundef nonnull align 8 dereferenceable(8) %35, ptr noundef nonnull align 4 dereferenceable(4) %36)
          to label %144 unwind label %164

144:                                              ; preds = %141
  call void @llvm.lifetime.end.p0(ptr nonnull %34)
  call void @llvm.lifetime.end.p0(ptr nonnull %35)
  call void @llvm.lifetime.end.p0(ptr nonnull %36)
  %145 = icmp eq i16 %135, %143
  br i1 %145, label %146, label %159

146:                                              ; preds = %144, %146
  %147 = phi i64 [ %152, %146 ], [ 0, %144 ]
  %148 = getelementptr inbounds nuw i16, ptr %63, i64 %147
  %149 = getelementptr inbounds nuw i8, ptr %148, i64 16
  store <8 x i16> splat (i16 -32768), ptr %148, align 2, !tbaa !69
  store <8 x i16> splat (i16 -32768), ptr %149, align 2, !tbaa !69
  %150 = getelementptr inbounds nuw i16, ptr %64, i64 %147
  %151 = getelementptr inbounds nuw i8, ptr %150, i64 16
  store <8 x i16> splat (i16 32767), ptr %150, align 2, !tbaa !69
  store <8 x i16> splat (i16 32767), ptr %151, align 2, !tbaa !69
  %152 = add nuw i64 %147, 16
  %153 = icmp eq i64 %152, 992
  br i1 %153, label %154, label %146, !llvm.loop !74

154:                                              ; preds = %146
  %155 = getelementptr inbounds nuw i8, ptr %63, i64 1984
  store <8 x i16> splat (i16 -32768), ptr %155, align 2, !tbaa !69
  %156 = getelementptr inbounds nuw i8, ptr %64, i64 1984
  store <8 x i16> splat (i16 32767), ptr %156, align 2, !tbaa !69
  call void @llvm.lifetime.start.p0(ptr nonnull %31)
  call void @llvm.lifetime.start.p0(ptr nonnull %32)
  call void @llvm.lifetime.start.p0(ptr nonnull %33)
  store ptr %63, ptr %31, align 8, !tbaa !71
  store ptr %64, ptr %32, align 8, !tbaa !71
  store i32 1000, ptr %33, align 4, !tbaa !43
  %157 = load ptr, ptr %84, align 8, !tbaa !20
  %158 = icmp eq ptr %157, null
  br i1 %158, label %166, label %168

159:                                              ; preds = %144
  %160 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.14)
          to label %161 unwind label %164

161:                                              ; preds = %159
  call void @exit(i32 noundef 1) #25
  unreachable

162:                                              ; preds = %133, %131
  %163 = landingpad { ptr, i32 }
          cleanup
  br label %331

164:                                              ; preds = %141, %139, %159
  %165 = landingpad { ptr, i32 }
          cleanup
  br label %331

166:                                              ; preds = %154
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %167 unwind label %198

167:                                              ; preds = %166
  unreachable

168:                                              ; preds = %154
  %169 = load ptr, ptr %90, align 8, !tbaa !23
  %170 = invoke noundef i16 %169(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %31, ptr noundef nonnull align 8 dereferenceable(8) %32, ptr noundef nonnull align 4 dereferenceable(4) %33)
          to label %171 unwind label %198

171:                                              ; preds = %168
  call void @llvm.lifetime.end.p0(ptr nonnull %31)
  call void @llvm.lifetime.end.p0(ptr nonnull %32)
  call void @llvm.lifetime.end.p0(ptr nonnull %33)
  call void @llvm.lifetime.start.p0(ptr nonnull %28)
  call void @llvm.lifetime.start.p0(ptr nonnull %29)
  call void @llvm.lifetime.start.p0(ptr nonnull %30)
  store ptr %63, ptr %28, align 8, !tbaa !71
  store ptr %64, ptr %29, align 8, !tbaa !71
  store i32 1000, ptr %30, align 4, !tbaa !43
  %172 = load ptr, ptr %94, align 8, !tbaa !20
  %173 = icmp eq ptr %172, null
  br i1 %173, label %174, label %176

174:                                              ; preds = %171
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %175 unwind label %200

175:                                              ; preds = %174
  unreachable

176:                                              ; preds = %171
  %177 = load ptr, ptr %100, align 8, !tbaa !23
  %178 = invoke noundef i16 %177(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %28, ptr noundef nonnull align 8 dereferenceable(8) %29, ptr noundef nonnull align 4 dereferenceable(4) %30)
          to label %179 unwind label %200

179:                                              ; preds = %176
  call void @llvm.lifetime.end.p0(ptr nonnull %28)
  call void @llvm.lifetime.end.p0(ptr nonnull %29)
  call void @llvm.lifetime.end.p0(ptr nonnull %30)
  %180 = icmp eq i16 %170, %178
  br i1 %180, label %181, label %195

181:                                              ; preds = %179, %181
  %182 = phi i64 [ %187, %181 ], [ 0, %179 ]
  %183 = getelementptr inbounds nuw i16, ptr %64, i64 %182
  %184 = getelementptr inbounds nuw i8, ptr %183, i64 16
  store <8 x i16> splat (i16 -32768), ptr %183, align 2, !tbaa !69
  store <8 x i16> splat (i16 -32768), ptr %184, align 2, !tbaa !69
  %185 = getelementptr inbounds nuw i16, ptr %63, i64 %182
  %186 = getelementptr inbounds nuw i8, ptr %185, i64 16
  store <8 x i16> splat (i16 -32768), ptr %185, align 2, !tbaa !69
  store <8 x i16> splat (i16 -32768), ptr %186, align 2, !tbaa !69
  %187 = add nuw i64 %182, 16
  %188 = icmp eq i64 %187, 992
  br i1 %188, label %189, label %181, !llvm.loop !75

189:                                              ; preds = %181
  %190 = getelementptr inbounds nuw i8, ptr %64, i64 1984
  store <8 x i16> splat (i16 -32768), ptr %190, align 2, !tbaa !69
  %191 = getelementptr inbounds nuw i8, ptr %63, i64 1984
  store <8 x i16> splat (i16 -32768), ptr %191, align 2, !tbaa !69
  %192 = getelementptr inbounds nuw i8, ptr %63, i64 1996
  store i16 32767, ptr %192, align 2, !tbaa !69
  call void @llvm.lifetime.start.p0(ptr nonnull %25)
  call void @llvm.lifetime.start.p0(ptr nonnull %26)
  call void @llvm.lifetime.start.p0(ptr nonnull %27)
  store ptr %63, ptr %25, align 8, !tbaa !71
  store ptr %64, ptr %26, align 8, !tbaa !71
  store i32 1000, ptr %27, align 4, !tbaa !43
  %193 = load ptr, ptr %84, align 8, !tbaa !20
  %194 = icmp eq ptr %193, null
  br i1 %194, label %202, label %204

195:                                              ; preds = %179
  %196 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.14)
          to label %197 unwind label %200

197:                                              ; preds = %195
  call void @exit(i32 noundef 1) #25
  unreachable

198:                                              ; preds = %168, %166
  %199 = landingpad { ptr, i32 }
          cleanup
  br label %331

200:                                              ; preds = %176, %174, %195
  %201 = landingpad { ptr, i32 }
          cleanup
  br label %331

202:                                              ; preds = %189
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %203 unwind label %233

203:                                              ; preds = %202
  unreachable

204:                                              ; preds = %189
  %205 = load ptr, ptr %90, align 8, !tbaa !23
  %206 = invoke noundef i16 %205(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %25, ptr noundef nonnull align 8 dereferenceable(8) %26, ptr noundef nonnull align 4 dereferenceable(4) %27)
          to label %207 unwind label %233

207:                                              ; preds = %204
  call void @llvm.lifetime.end.p0(ptr nonnull %25)
  call void @llvm.lifetime.end.p0(ptr nonnull %26)
  call void @llvm.lifetime.end.p0(ptr nonnull %27)
  call void @llvm.lifetime.start.p0(ptr nonnull %22)
  call void @llvm.lifetime.start.p0(ptr nonnull %23)
  call void @llvm.lifetime.start.p0(ptr nonnull %24)
  store ptr %63, ptr %22, align 8, !tbaa !71
  store ptr %64, ptr %23, align 8, !tbaa !71
  store i32 1000, ptr %24, align 4, !tbaa !43
  %208 = load ptr, ptr %94, align 8, !tbaa !20
  %209 = icmp eq ptr %208, null
  br i1 %209, label %210, label %212

210:                                              ; preds = %207
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %211 unwind label %235

211:                                              ; preds = %210
  unreachable

212:                                              ; preds = %207
  %213 = load ptr, ptr %100, align 8, !tbaa !23
  %214 = invoke noundef i16 %213(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %22, ptr noundef nonnull align 8 dereferenceable(8) %23, ptr noundef nonnull align 4 dereferenceable(4) %24)
          to label %215 unwind label %235

215:                                              ; preds = %212
  call void @llvm.lifetime.end.p0(ptr nonnull %22)
  call void @llvm.lifetime.end.p0(ptr nonnull %23)
  call void @llvm.lifetime.end.p0(ptr nonnull %24)
  %216 = icmp eq i16 %206, %214
  br i1 %216, label %217, label %230

217:                                              ; preds = %215, %217
  %218 = phi i64 [ %223, %217 ], [ 0, %215 ]
  %219 = getelementptr inbounds nuw i16, ptr %64, i64 %218
  %220 = getelementptr inbounds nuw i8, ptr %219, i64 16
  store <8 x i16> splat (i16 -32768), ptr %219, align 2, !tbaa !69
  store <8 x i16> splat (i16 -32768), ptr %220, align 2, !tbaa !69
  %221 = getelementptr inbounds nuw i16, ptr %63, i64 %218
  %222 = getelementptr inbounds nuw i8, ptr %221, i64 16
  store <8 x i16> splat (i16 -32768), ptr %221, align 2, !tbaa !69
  store <8 x i16> splat (i16 -32768), ptr %222, align 2, !tbaa !69
  %223 = add nuw i64 %218, 16
  %224 = icmp eq i64 %223, 992
  br i1 %224, label %225, label %217, !llvm.loop !76

225:                                              ; preds = %217
  %226 = getelementptr inbounds nuw i8, ptr %64, i64 1984
  store <8 x i16> splat (i16 -32768), ptr %226, align 2, !tbaa !69
  %227 = getelementptr inbounds nuw i8, ptr %63, i64 1984
  store <8 x i16> splat (i16 -32768), ptr %227, align 2, !tbaa !69
  store i16 32767, ptr %63, align 2, !tbaa !69
  call void @llvm.lifetime.start.p0(ptr nonnull %19)
  call void @llvm.lifetime.start.p0(ptr nonnull %20)
  call void @llvm.lifetime.start.p0(ptr nonnull %21)
  store ptr %63, ptr %19, align 8, !tbaa !71
  store ptr %64, ptr %20, align 8, !tbaa !71
  store i32 1000, ptr %21, align 4, !tbaa !43
  %228 = load ptr, ptr %84, align 8, !tbaa !20
  %229 = icmp eq ptr %228, null
  br i1 %229, label %237, label %239

230:                                              ; preds = %215
  %231 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.14)
          to label %232 unwind label %235

232:                                              ; preds = %230
  call void @exit(i32 noundef 1) #25
  unreachable

233:                                              ; preds = %204, %202
  %234 = landingpad { ptr, i32 }
          cleanup
  br label %331

235:                                              ; preds = %212, %210, %230
  %236 = landingpad { ptr, i32 }
          cleanup
  br label %331

237:                                              ; preds = %225
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %238 unwind label %269

238:                                              ; preds = %237
  unreachable

239:                                              ; preds = %225
  %240 = load ptr, ptr %90, align 8, !tbaa !23
  %241 = invoke noundef i16 %240(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %19, ptr noundef nonnull align 8 dereferenceable(8) %20, ptr noundef nonnull align 4 dereferenceable(4) %21)
          to label %242 unwind label %269

242:                                              ; preds = %239
  call void @llvm.lifetime.end.p0(ptr nonnull %19)
  call void @llvm.lifetime.end.p0(ptr nonnull %20)
  call void @llvm.lifetime.end.p0(ptr nonnull %21)
  call void @llvm.lifetime.start.p0(ptr nonnull %16)
  call void @llvm.lifetime.start.p0(ptr nonnull %17)
  call void @llvm.lifetime.start.p0(ptr nonnull %18)
  store ptr %63, ptr %16, align 8, !tbaa !71
  store ptr %64, ptr %17, align 8, !tbaa !71
  store i32 1000, ptr %18, align 4, !tbaa !43
  %243 = load ptr, ptr %94, align 8, !tbaa !20
  %244 = icmp eq ptr %243, null
  br i1 %244, label %245, label %247

245:                                              ; preds = %242
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %246 unwind label %271

246:                                              ; preds = %245
  unreachable

247:                                              ; preds = %242
  %248 = load ptr, ptr %100, align 8, !tbaa !23
  %249 = invoke noundef i16 %248(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %16, ptr noundef nonnull align 8 dereferenceable(8) %17, ptr noundef nonnull align 4 dereferenceable(4) %18)
          to label %250 unwind label %271

250:                                              ; preds = %247
  call void @llvm.lifetime.end.p0(ptr nonnull %16)
  call void @llvm.lifetime.end.p0(ptr nonnull %17)
  call void @llvm.lifetime.end.p0(ptr nonnull %18)
  %251 = icmp eq i16 %241, %249
  br i1 %251, label %252, label %266

252:                                              ; preds = %250, %252
  %253 = phi i64 [ %258, %252 ], [ 0, %250 ]
  %254 = getelementptr inbounds nuw i16, ptr %64, i64 %253
  %255 = getelementptr inbounds nuw i8, ptr %254, i64 16
  store <8 x i16> splat (i16 -32768), ptr %254, align 2, !tbaa !69
  store <8 x i16> splat (i16 -32768), ptr %255, align 2, !tbaa !69
  %256 = getelementptr inbounds nuw i16, ptr %63, i64 %253
  %257 = getelementptr inbounds nuw i8, ptr %256, i64 16
  store <8 x i16> splat (i16 -32768), ptr %256, align 2, !tbaa !69
  store <8 x i16> splat (i16 -32768), ptr %257, align 2, !tbaa !69
  %258 = add nuw i64 %253, 16
  %259 = icmp eq i64 %258, 992
  br i1 %259, label %260, label %252, !llvm.loop !77

260:                                              ; preds = %252
  %261 = getelementptr inbounds nuw i8, ptr %64, i64 1984
  store <8 x i16> splat (i16 -32768), ptr %261, align 2, !tbaa !69
  %262 = getelementptr inbounds nuw i8, ptr %63, i64 1984
  store <8 x i16> splat (i16 -32768), ptr %262, align 2, !tbaa !69
  %263 = getelementptr inbounds nuw i8, ptr %63, i64 1998
  store i16 32767, ptr %263, align 2, !tbaa !69
  call void @llvm.lifetime.start.p0(ptr nonnull %13)
  call void @llvm.lifetime.start.p0(ptr nonnull %14)
  call void @llvm.lifetime.start.p0(ptr nonnull %15)
  store ptr %63, ptr %13, align 8, !tbaa !71
  store ptr %64, ptr %14, align 8, !tbaa !71
  store i32 1000, ptr %15, align 4, !tbaa !43
  %264 = load ptr, ptr %84, align 8, !tbaa !20
  %265 = icmp eq ptr %264, null
  br i1 %265, label %273, label %275

266:                                              ; preds = %250
  %267 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.14)
          to label %268 unwind label %271

268:                                              ; preds = %266
  call void @exit(i32 noundef 1) #25
  unreachable

269:                                              ; preds = %239, %237
  %270 = landingpad { ptr, i32 }
          cleanup
  br label %331

271:                                              ; preds = %247, %245, %266
  %272 = landingpad { ptr, i32 }
          cleanup
  br label %331

273:                                              ; preds = %260
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %274 unwind label %304

274:                                              ; preds = %273
  unreachable

275:                                              ; preds = %260
  %276 = load ptr, ptr %90, align 8, !tbaa !23
  %277 = invoke noundef i16 %276(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %13, ptr noundef nonnull align 8 dereferenceable(8) %14, ptr noundef nonnull align 4 dereferenceable(4) %15)
          to label %278 unwind label %304

278:                                              ; preds = %275
  call void @llvm.lifetime.end.p0(ptr nonnull %13)
  call void @llvm.lifetime.end.p0(ptr nonnull %14)
  call void @llvm.lifetime.end.p0(ptr nonnull %15)
  call void @llvm.lifetime.start.p0(ptr nonnull %10)
  call void @llvm.lifetime.start.p0(ptr nonnull %11)
  call void @llvm.lifetime.start.p0(ptr nonnull %12)
  store ptr %63, ptr %10, align 8, !tbaa !71
  store ptr %64, ptr %11, align 8, !tbaa !71
  store i32 1000, ptr %12, align 4, !tbaa !43
  %279 = load ptr, ptr %94, align 8, !tbaa !20
  %280 = icmp eq ptr %279, null
  br i1 %280, label %281, label %283

281:                                              ; preds = %278
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %282 unwind label %306

282:                                              ; preds = %281
  unreachable

283:                                              ; preds = %278
  %284 = load ptr, ptr %100, align 8, !tbaa !23
  %285 = invoke noundef i16 %284(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %10, ptr noundef nonnull align 8 dereferenceable(8) %11, ptr noundef nonnull align 4 dereferenceable(4) %12)
          to label %286 unwind label %306

286:                                              ; preds = %283
  call void @llvm.lifetime.end.p0(ptr nonnull %10)
  call void @llvm.lifetime.end.p0(ptr nonnull %11)
  call void @llvm.lifetime.end.p0(ptr nonnull %12)
  %287 = icmp eq i16 %277, %285
  br i1 %287, label %288, label %301

288:                                              ; preds = %286, %288
  %289 = phi i64 [ %294, %288 ], [ 0, %286 ]
  %290 = getelementptr inbounds nuw i16, ptr %64, i64 %289
  %291 = getelementptr inbounds nuw i8, ptr %290, i64 16
  store <8 x i16> splat (i16 -32768), ptr %290, align 2, !tbaa !69
  store <8 x i16> splat (i16 -32768), ptr %291, align 2, !tbaa !69
  %292 = getelementptr inbounds nuw i16, ptr %63, i64 %289
  %293 = getelementptr inbounds nuw i8, ptr %292, i64 16
  store <8 x i16> splat (i16 -32768), ptr %292, align 2, !tbaa !69
  store <8 x i16> splat (i16 -32768), ptr %293, align 2, !tbaa !69
  %294 = add nuw i64 %289, 16
  %295 = icmp eq i64 %294, 992
  br i1 %295, label %296, label %288, !llvm.loop !78

296:                                              ; preds = %288
  %297 = getelementptr inbounds nuw i8, ptr %64, i64 1984
  store <8 x i16> splat (i16 -32768), ptr %297, align 2, !tbaa !69
  %298 = getelementptr inbounds nuw i8, ptr %63, i64 1984
  store <8 x i16> splat (i16 -32768), ptr %298, align 2, !tbaa !69
  store i16 32767, ptr %263, align 2, !tbaa !69
  store i16 32767, ptr %63, align 2, !tbaa !69
  call void @llvm.lifetime.start.p0(ptr nonnull %7)
  call void @llvm.lifetime.start.p0(ptr nonnull %8)
  call void @llvm.lifetime.start.p0(ptr nonnull %9)
  store ptr %63, ptr %7, align 8, !tbaa !71
  store ptr %64, ptr %8, align 8, !tbaa !71
  store i32 1000, ptr %9, align 4, !tbaa !43
  %299 = load ptr, ptr %84, align 8, !tbaa !20
  %300 = icmp eq ptr %299, null
  br i1 %300, label %308, label %310

301:                                              ; preds = %286
  %302 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.14)
          to label %303 unwind label %306

303:                                              ; preds = %301
  call void @exit(i32 noundef 1) #25
  unreachable

304:                                              ; preds = %275, %273
  %305 = landingpad { ptr, i32 }
          cleanup
  br label %331

306:                                              ; preds = %283, %281, %301
  %307 = landingpad { ptr, i32 }
          cleanup
  br label %331

308:                                              ; preds = %296
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %309 unwind label %326

309:                                              ; preds = %308
  unreachable

310:                                              ; preds = %296
  %311 = load ptr, ptr %90, align 8, !tbaa !23
  %312 = invoke noundef i16 %311(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %7, ptr noundef nonnull align 8 dereferenceable(8) %8, ptr noundef nonnull align 4 dereferenceable(4) %9)
          to label %313 unwind label %326

313:                                              ; preds = %310
  call void @llvm.lifetime.end.p0(ptr nonnull %7)
  call void @llvm.lifetime.end.p0(ptr nonnull %8)
  call void @llvm.lifetime.end.p0(ptr nonnull %9)
  call void @llvm.lifetime.start.p0(ptr nonnull %4)
  call void @llvm.lifetime.start.p0(ptr nonnull %5)
  call void @llvm.lifetime.start.p0(ptr nonnull %6)
  store ptr %63, ptr %4, align 8, !tbaa !71
  store ptr %64, ptr %5, align 8, !tbaa !71
  store i32 1000, ptr %6, align 4, !tbaa !43
  %314 = load ptr, ptr %94, align 8, !tbaa !20
  %315 = icmp eq ptr %314, null
  br i1 %315, label %316, label %318

316:                                              ; preds = %313
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %317 unwind label %328

317:                                              ; preds = %316
  unreachable

318:                                              ; preds = %313
  %319 = load ptr, ptr %100, align 8, !tbaa !23
  %320 = invoke noundef i16 %319(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %4, ptr noundef nonnull align 8 dereferenceable(8) %5, ptr noundef nonnull align 4 dereferenceable(4) %6)
          to label %321 unwind label %328

321:                                              ; preds = %318
  call void @llvm.lifetime.end.p0(ptr nonnull %4)
  call void @llvm.lifetime.end.p0(ptr nonnull %5)
  call void @llvm.lifetime.end.p0(ptr nonnull %6)
  %322 = icmp eq i16 %312, %320
  br i1 %322, label %330, label %323

323:                                              ; preds = %321
  %324 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.14)
          to label %325 unwind label %328

325:                                              ; preds = %323
  call void @exit(i32 noundef 1) #25
  unreachable

326:                                              ; preds = %310, %308
  %327 = landingpad { ptr, i32 }
          cleanup
  br label %331

328:                                              ; preds = %318, %316, %323
  %329 = landingpad { ptr, i32 }
          cleanup
  br label %331

330:                                              ; preds = %321
  call void @_ZdaPv(ptr noundef nonnull %64) #26
  call void @_ZdaPv(ptr noundef nonnull %63) #26
  ret void

331:                                              ; preds = %123, %125, %326, %328, %304, %306, %269, %271, %233, %235, %198, %200, %162, %164, %127, %129
  %332 = phi { ptr, i32 } [ %130, %129 ], [ %128, %127 ], [ %165, %164 ], [ %163, %162 ], [ %201, %200 ], [ %199, %198 ], [ %236, %235 ], [ %234, %233 ], [ %272, %271 ], [ %270, %269 ], [ %307, %306 ], [ %305, %304 ], [ %329, %328 ], [ %327, %326 ], [ %124, %123 ], [ %126, %125 ]
  call void @_ZdaPv(ptr noundef nonnull %64) #26
  br label %333

333:                                              ; preds = %331, %121
  %334 = phi { ptr, i32 } [ %332, %331 ], [ %122, %121 ]
  call void @_ZdaPv(ptr noundef nonnull %63) #26
  resume { ptr, i32 } %334
}

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
  %40 = alloca ptr, align 8
  %41 = alloca ptr, align 8
  %42 = alloca i32, align 4
  %43 = alloca ptr, align 8
  %44 = alloca ptr, align 8
  %45 = alloca i32, align 4
  %46 = alloca %"class.std::uniform_int_distribution.73", align 8
  %47 = alloca %"class.std::uniform_int_distribution.73", align 8
  %48 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.12, i64 noundef 9)
  %49 = icmp eq ptr %2, null
  br i1 %49, label %50, label %58

50:                                               ; preds = %3
  %51 = load ptr, ptr @_ZSt4cout, align 8, !tbaa !31
  %52 = getelementptr i8, ptr %51, i64 -24
  %53 = load i64, ptr %52, align 8
  %54 = getelementptr inbounds i8, ptr @_ZSt4cout, i64 %53
  %55 = getelementptr inbounds nuw i8, ptr %54, i64 32
  %56 = load i32, ptr %55, align 8, !tbaa !33
  %57 = or i32 %56, 1
  tail call void @_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate(ptr noundef nonnull align 8 dereferenceable(264) %54, i32 noundef %57)
  br label %61

58:                                               ; preds = %3
  %59 = tail call noundef i64 @strlen(ptr noundef nonnull dereferenceable(1) %2) #21
  %60 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull %2, i64 noundef %59)
  br label %61

61:                                               ; preds = %50, %58
  %62 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.13, i64 noundef 1)
  %63 = tail call noalias noundef nonnull dereferenceable(4000) ptr @_Znam(i64 noundef 4000) #23
  %64 = invoke noalias noundef nonnull dereferenceable(4000) ptr @_Znam(i64 noundef 4000) #23
          to label %65 unwind label %109

65:                                               ; preds = %61
  call void @llvm.lifetime.start.p0(ptr nonnull %47) #21
  store <2 x i32> <i32 0, i32 -1>, ptr %47, align 8, !tbaa !43
  br label %66

66:                                               ; preds = %69, %65
  %67 = phi i64 [ 0, %65 ], [ %71, %69 ]
  %68 = invoke noundef i32 @_ZNSt24uniform_int_distributionIjEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEjRT_RKNS0_10param_typeE(ptr noundef nonnull align 4 dereferenceable(8) %47, ptr noundef nonnull align 8 dereferenceable(5000) @_ZL3rng, ptr noundef nonnull align 4 dereferenceable(8) %47)
          to label %69 unwind label %113

69:                                               ; preds = %66
  %70 = getelementptr inbounds nuw i32, ptr %63, i64 %67
  store i32 %68, ptr %70, align 4, !tbaa !43
  %71 = add nuw nsw i64 %67, 1
  %72 = icmp eq i64 %71, 1000
  br i1 %72, label %73, label %66, !llvm.loop !79

73:                                               ; preds = %69
  call void @llvm.lifetime.end.p0(ptr nonnull %47) #21
  call void @llvm.lifetime.start.p0(ptr nonnull %46) #21
  store <2 x i32> <i32 0, i32 -1>, ptr %46, align 8, !tbaa !43
  br label %74

74:                                               ; preds = %77, %73
  %75 = phi i64 [ 0, %73 ], [ %79, %77 ]
  %76 = invoke noundef i32 @_ZNSt24uniform_int_distributionIjEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEjRT_RKNS0_10param_typeE(ptr noundef nonnull align 4 dereferenceable(8) %46, ptr noundef nonnull align 8 dereferenceable(5000) @_ZL3rng, ptr noundef nonnull align 4 dereferenceable(8) %46)
          to label %77 unwind label %111

77:                                               ; preds = %74
  %78 = getelementptr inbounds nuw i32, ptr %64, i64 %75
  store i32 %76, ptr %78, align 4, !tbaa !43
  %79 = add nuw nsw i64 %75, 1
  %80 = icmp eq i64 %79, 1000
  br i1 %80, label %81, label %74, !llvm.loop !79

81:                                               ; preds = %77
  call void @llvm.lifetime.end.p0(ptr nonnull %46) #21
  call void @llvm.lifetime.start.p0(ptr nonnull %43)
  call void @llvm.lifetime.start.p0(ptr nonnull %44)
  call void @llvm.lifetime.start.p0(ptr nonnull %45)
  store ptr %63, ptr %43, align 8, !tbaa !45
  store ptr %64, ptr %44, align 8, !tbaa !45
  store i32 1000, ptr %45, align 4, !tbaa !43
  %82 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %83 = load ptr, ptr %82, align 8, !tbaa !20
  %84 = icmp eq ptr %83, null
  br i1 %84, label %85, label %87

85:                                               ; preds = %81
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %86 unwind label %115

86:                                               ; preds = %85
  unreachable

87:                                               ; preds = %81
  %88 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %89 = load ptr, ptr %88, align 8, !tbaa !25
  %90 = invoke noundef i32 %89(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %43, ptr noundef nonnull align 8 dereferenceable(8) %44, ptr noundef nonnull align 4 dereferenceable(4) %45)
          to label %91 unwind label %115

91:                                               ; preds = %87
  call void @llvm.lifetime.end.p0(ptr nonnull %43)
  call void @llvm.lifetime.end.p0(ptr nonnull %44)
  call void @llvm.lifetime.end.p0(ptr nonnull %45)
  call void @llvm.lifetime.start.p0(ptr nonnull %40)
  call void @llvm.lifetime.start.p0(ptr nonnull %41)
  call void @llvm.lifetime.start.p0(ptr nonnull %42)
  store ptr %63, ptr %40, align 8, !tbaa !45
  store ptr %64, ptr %41, align 8, !tbaa !45
  store i32 1000, ptr %42, align 4, !tbaa !43
  %92 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %93 = load ptr, ptr %92, align 8, !tbaa !20
  %94 = icmp eq ptr %93, null
  br i1 %94, label %95, label %97

95:                                               ; preds = %91
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %96 unwind label %117

96:                                               ; preds = %95
  unreachable

97:                                               ; preds = %91
  %98 = getelementptr inbounds nuw i8, ptr %1, i64 24
  %99 = load ptr, ptr %98, align 8, !tbaa !25
  %100 = invoke noundef i32 %99(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %40, ptr noundef nonnull align 8 dereferenceable(8) %41, ptr noundef nonnull align 4 dereferenceable(4) %42)
          to label %101 unwind label %117

101:                                              ; preds = %97
  call void @llvm.lifetime.end.p0(ptr nonnull %40)
  call void @llvm.lifetime.end.p0(ptr nonnull %41)
  call void @llvm.lifetime.end.p0(ptr nonnull %42)
  %102 = icmp eq i32 %90, %100
  br i1 %102, label %103, label %106

103:                                              ; preds = %101
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(4000) %63, i8 -1, i64 4000, i1 false), !tbaa !43
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(4000) %64, i8 0, i64 4000, i1 false), !tbaa !43
  call void @llvm.lifetime.start.p0(ptr nonnull %37)
  call void @llvm.lifetime.start.p0(ptr nonnull %38)
  call void @llvm.lifetime.start.p0(ptr nonnull %39)
  store ptr %63, ptr %37, align 8, !tbaa !45
  store ptr %64, ptr %38, align 8, !tbaa !45
  store i32 1000, ptr %39, align 4, !tbaa !43
  %104 = load ptr, ptr %82, align 8, !tbaa !20
  %105 = icmp eq ptr %104, null
  br i1 %105, label %119, label %121

106:                                              ; preds = %101
  %107 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.14)
          to label %108 unwind label %117

108:                                              ; preds = %106
  call void @exit(i32 noundef 1) #25
  unreachable

109:                                              ; preds = %61
  %110 = landingpad { ptr, i32 }
          cleanup
  br label %273

111:                                              ; preds = %74
  %112 = landingpad { ptr, i32 }
          cleanup
  br label %271

113:                                              ; preds = %66
  %114 = landingpad { ptr, i32 }
          cleanup
  br label %271

115:                                              ; preds = %87, %85
  %116 = landingpad { ptr, i32 }
          cleanup
  br label %271

117:                                              ; preds = %97, %95, %106
  %118 = landingpad { ptr, i32 }
          cleanup
  br label %271

119:                                              ; preds = %103
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %120 unwind label %140

120:                                              ; preds = %119
  unreachable

121:                                              ; preds = %103
  %122 = load ptr, ptr %88, align 8, !tbaa !25
  %123 = invoke noundef i32 %122(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %37, ptr noundef nonnull align 8 dereferenceable(8) %38, ptr noundef nonnull align 4 dereferenceable(4) %39)
          to label %124 unwind label %140

124:                                              ; preds = %121
  call void @llvm.lifetime.end.p0(ptr nonnull %37)
  call void @llvm.lifetime.end.p0(ptr nonnull %38)
  call void @llvm.lifetime.end.p0(ptr nonnull %39)
  call void @llvm.lifetime.start.p0(ptr nonnull %34)
  call void @llvm.lifetime.start.p0(ptr nonnull %35)
  call void @llvm.lifetime.start.p0(ptr nonnull %36)
  store ptr %63, ptr %34, align 8, !tbaa !45
  store ptr %64, ptr %35, align 8, !tbaa !45
  store i32 1000, ptr %36, align 4, !tbaa !43
  %125 = load ptr, ptr %92, align 8, !tbaa !20
  %126 = icmp eq ptr %125, null
  br i1 %126, label %127, label %129

127:                                              ; preds = %124
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %128 unwind label %142

128:                                              ; preds = %127
  unreachable

129:                                              ; preds = %124
  %130 = load ptr, ptr %98, align 8, !tbaa !25
  %131 = invoke noundef i32 %130(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %34, ptr noundef nonnull align 8 dereferenceable(8) %35, ptr noundef nonnull align 4 dereferenceable(4) %36)
          to label %132 unwind label %142

132:                                              ; preds = %129
  call void @llvm.lifetime.end.p0(ptr nonnull %34)
  call void @llvm.lifetime.end.p0(ptr nonnull %35)
  call void @llvm.lifetime.end.p0(ptr nonnull %36)
  %133 = icmp eq i32 %123, %131
  br i1 %133, label %134, label %137

134:                                              ; preds = %132
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(4000) %63, i8 0, i64 4000, i1 false), !tbaa !43
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(4000) %64, i8 -1, i64 4000, i1 false), !tbaa !43
  call void @llvm.lifetime.start.p0(ptr nonnull %31)
  call void @llvm.lifetime.start.p0(ptr nonnull %32)
  call void @llvm.lifetime.start.p0(ptr nonnull %33)
  store ptr %63, ptr %31, align 8, !tbaa !45
  store ptr %64, ptr %32, align 8, !tbaa !45
  store i32 1000, ptr %33, align 4, !tbaa !43
  %135 = load ptr, ptr %82, align 8, !tbaa !20
  %136 = icmp eq ptr %135, null
  br i1 %136, label %144, label %146

137:                                              ; preds = %132
  %138 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.14)
          to label %139 unwind label %142

139:                                              ; preds = %137
  call void @exit(i32 noundef 1) #25
  unreachable

140:                                              ; preds = %121, %119
  %141 = landingpad { ptr, i32 }
          cleanup
  br label %271

142:                                              ; preds = %129, %127, %137
  %143 = landingpad { ptr, i32 }
          cleanup
  br label %271

144:                                              ; preds = %134
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %145 unwind label %166

145:                                              ; preds = %144
  unreachable

146:                                              ; preds = %134
  %147 = load ptr, ptr %88, align 8, !tbaa !25
  %148 = invoke noundef i32 %147(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %31, ptr noundef nonnull align 8 dereferenceable(8) %32, ptr noundef nonnull align 4 dereferenceable(4) %33)
          to label %149 unwind label %166

149:                                              ; preds = %146
  call void @llvm.lifetime.end.p0(ptr nonnull %31)
  call void @llvm.lifetime.end.p0(ptr nonnull %32)
  call void @llvm.lifetime.end.p0(ptr nonnull %33)
  call void @llvm.lifetime.start.p0(ptr nonnull %28)
  call void @llvm.lifetime.start.p0(ptr nonnull %29)
  call void @llvm.lifetime.start.p0(ptr nonnull %30)
  store ptr %63, ptr %28, align 8, !tbaa !45
  store ptr %64, ptr %29, align 8, !tbaa !45
  store i32 1000, ptr %30, align 4, !tbaa !43
  %150 = load ptr, ptr %92, align 8, !tbaa !20
  %151 = icmp eq ptr %150, null
  br i1 %151, label %152, label %154

152:                                              ; preds = %149
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %153 unwind label %168

153:                                              ; preds = %152
  unreachable

154:                                              ; preds = %149
  %155 = load ptr, ptr %98, align 8, !tbaa !25
  %156 = invoke noundef i32 %155(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %28, ptr noundef nonnull align 8 dereferenceable(8) %29, ptr noundef nonnull align 4 dereferenceable(4) %30)
          to label %157 unwind label %168

157:                                              ; preds = %154
  call void @llvm.lifetime.end.p0(ptr nonnull %28)
  call void @llvm.lifetime.end.p0(ptr nonnull %29)
  call void @llvm.lifetime.end.p0(ptr nonnull %30)
  %158 = icmp eq i32 %148, %156
  br i1 %158, label %159, label %163

159:                                              ; preds = %157
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(4000) %64, i8 0, i64 4000, i1 false), !tbaa !43
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(4000) %63, i8 0, i64 4000, i1 false), !tbaa !43
  %160 = getelementptr inbounds nuw i8, ptr %63, i64 3992
  store i32 -1, ptr %160, align 4, !tbaa !43
  call void @llvm.lifetime.start.p0(ptr nonnull %25)
  call void @llvm.lifetime.start.p0(ptr nonnull %26)
  call void @llvm.lifetime.start.p0(ptr nonnull %27)
  store ptr %63, ptr %25, align 8, !tbaa !45
  store ptr %64, ptr %26, align 8, !tbaa !45
  store i32 1000, ptr %27, align 4, !tbaa !43
  %161 = load ptr, ptr %82, align 8, !tbaa !20
  %162 = icmp eq ptr %161, null
  br i1 %162, label %170, label %172

163:                                              ; preds = %157
  %164 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.14)
          to label %165 unwind label %168

165:                                              ; preds = %163
  call void @exit(i32 noundef 1) #25
  unreachable

166:                                              ; preds = %146, %144
  %167 = landingpad { ptr, i32 }
          cleanup
  br label %271

168:                                              ; preds = %154, %152, %163
  %169 = landingpad { ptr, i32 }
          cleanup
  br label %271

170:                                              ; preds = %159
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %171 unwind label %192

171:                                              ; preds = %170
  unreachable

172:                                              ; preds = %159
  %173 = load ptr, ptr %88, align 8, !tbaa !25
  %174 = invoke noundef i32 %173(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %25, ptr noundef nonnull align 8 dereferenceable(8) %26, ptr noundef nonnull align 4 dereferenceable(4) %27)
          to label %175 unwind label %192

175:                                              ; preds = %172
  call void @llvm.lifetime.end.p0(ptr nonnull %25)
  call void @llvm.lifetime.end.p0(ptr nonnull %26)
  call void @llvm.lifetime.end.p0(ptr nonnull %27)
  call void @llvm.lifetime.start.p0(ptr nonnull %22)
  call void @llvm.lifetime.start.p0(ptr nonnull %23)
  call void @llvm.lifetime.start.p0(ptr nonnull %24)
  store ptr %63, ptr %22, align 8, !tbaa !45
  store ptr %64, ptr %23, align 8, !tbaa !45
  store i32 1000, ptr %24, align 4, !tbaa !43
  %176 = load ptr, ptr %92, align 8, !tbaa !20
  %177 = icmp eq ptr %176, null
  br i1 %177, label %178, label %180

178:                                              ; preds = %175
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %179 unwind label %194

179:                                              ; preds = %178
  unreachable

180:                                              ; preds = %175
  %181 = load ptr, ptr %98, align 8, !tbaa !25
  %182 = invoke noundef i32 %181(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %22, ptr noundef nonnull align 8 dereferenceable(8) %23, ptr noundef nonnull align 4 dereferenceable(4) %24)
          to label %183 unwind label %194

183:                                              ; preds = %180
  call void @llvm.lifetime.end.p0(ptr nonnull %22)
  call void @llvm.lifetime.end.p0(ptr nonnull %23)
  call void @llvm.lifetime.end.p0(ptr nonnull %24)
  %184 = icmp eq i32 %174, %182
  br i1 %184, label %185, label %189

185:                                              ; preds = %183
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(4000) %64, i8 0, i64 4000, i1 false), !tbaa !43
  %186 = getelementptr inbounds nuw i8, ptr %63, i64 4
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(3996) %186, i8 0, i64 3996, i1 false), !tbaa !43
  store i32 -1, ptr %63, align 4, !tbaa !43
  call void @llvm.lifetime.start.p0(ptr nonnull %19)
  call void @llvm.lifetime.start.p0(ptr nonnull %20)
  call void @llvm.lifetime.start.p0(ptr nonnull %21)
  store ptr %63, ptr %19, align 8, !tbaa !45
  store ptr %64, ptr %20, align 8, !tbaa !45
  store i32 1000, ptr %21, align 4, !tbaa !43
  %187 = load ptr, ptr %82, align 8, !tbaa !20
  %188 = icmp eq ptr %187, null
  br i1 %188, label %196, label %198

189:                                              ; preds = %183
  %190 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.14)
          to label %191 unwind label %194

191:                                              ; preds = %189
  call void @exit(i32 noundef 1) #25
  unreachable

192:                                              ; preds = %172, %170
  %193 = landingpad { ptr, i32 }
          cleanup
  br label %271

194:                                              ; preds = %180, %178, %189
  %195 = landingpad { ptr, i32 }
          cleanup
  br label %271

196:                                              ; preds = %185
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %197 unwind label %218

197:                                              ; preds = %196
  unreachable

198:                                              ; preds = %185
  %199 = load ptr, ptr %88, align 8, !tbaa !25
  %200 = invoke noundef i32 %199(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %19, ptr noundef nonnull align 8 dereferenceable(8) %20, ptr noundef nonnull align 4 dereferenceable(4) %21)
          to label %201 unwind label %218

201:                                              ; preds = %198
  call void @llvm.lifetime.end.p0(ptr nonnull %19)
  call void @llvm.lifetime.end.p0(ptr nonnull %20)
  call void @llvm.lifetime.end.p0(ptr nonnull %21)
  call void @llvm.lifetime.start.p0(ptr nonnull %16)
  call void @llvm.lifetime.start.p0(ptr nonnull %17)
  call void @llvm.lifetime.start.p0(ptr nonnull %18)
  store ptr %63, ptr %16, align 8, !tbaa !45
  store ptr %64, ptr %17, align 8, !tbaa !45
  store i32 1000, ptr %18, align 4, !tbaa !43
  %202 = load ptr, ptr %92, align 8, !tbaa !20
  %203 = icmp eq ptr %202, null
  br i1 %203, label %204, label %206

204:                                              ; preds = %201
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %205 unwind label %220

205:                                              ; preds = %204
  unreachable

206:                                              ; preds = %201
  %207 = load ptr, ptr %98, align 8, !tbaa !25
  %208 = invoke noundef i32 %207(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %16, ptr noundef nonnull align 8 dereferenceable(8) %17, ptr noundef nonnull align 4 dereferenceable(4) %18)
          to label %209 unwind label %220

209:                                              ; preds = %206
  call void @llvm.lifetime.end.p0(ptr nonnull %16)
  call void @llvm.lifetime.end.p0(ptr nonnull %17)
  call void @llvm.lifetime.end.p0(ptr nonnull %18)
  %210 = icmp eq i32 %200, %208
  br i1 %210, label %211, label %215

211:                                              ; preds = %209
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(4000) %64, i8 0, i64 4000, i1 false), !tbaa !43
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(4000) %63, i8 0, i64 3996, i1 false), !tbaa !43
  %212 = getelementptr inbounds nuw i8, ptr %63, i64 3996
  store i32 -1, ptr %212, align 4, !tbaa !43
  call void @llvm.lifetime.start.p0(ptr nonnull %13)
  call void @llvm.lifetime.start.p0(ptr nonnull %14)
  call void @llvm.lifetime.start.p0(ptr nonnull %15)
  store ptr %63, ptr %13, align 8, !tbaa !45
  store ptr %64, ptr %14, align 8, !tbaa !45
  store i32 1000, ptr %15, align 4, !tbaa !43
  %213 = load ptr, ptr %82, align 8, !tbaa !20
  %214 = icmp eq ptr %213, null
  br i1 %214, label %222, label %224

215:                                              ; preds = %209
  %216 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.14)
          to label %217 unwind label %220

217:                                              ; preds = %215
  call void @exit(i32 noundef 1) #25
  unreachable

218:                                              ; preds = %198, %196
  %219 = landingpad { ptr, i32 }
          cleanup
  br label %271

220:                                              ; preds = %206, %204, %215
  %221 = landingpad { ptr, i32 }
          cleanup
  br label %271

222:                                              ; preds = %211
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %223 unwind label %244

223:                                              ; preds = %222
  unreachable

224:                                              ; preds = %211
  %225 = load ptr, ptr %88, align 8, !tbaa !25
  %226 = invoke noundef i32 %225(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %13, ptr noundef nonnull align 8 dereferenceable(8) %14, ptr noundef nonnull align 4 dereferenceable(4) %15)
          to label %227 unwind label %244

227:                                              ; preds = %224
  call void @llvm.lifetime.end.p0(ptr nonnull %13)
  call void @llvm.lifetime.end.p0(ptr nonnull %14)
  call void @llvm.lifetime.end.p0(ptr nonnull %15)
  call void @llvm.lifetime.start.p0(ptr nonnull %10)
  call void @llvm.lifetime.start.p0(ptr nonnull %11)
  call void @llvm.lifetime.start.p0(ptr nonnull %12)
  store ptr %63, ptr %10, align 8, !tbaa !45
  store ptr %64, ptr %11, align 8, !tbaa !45
  store i32 1000, ptr %12, align 4, !tbaa !43
  %228 = load ptr, ptr %92, align 8, !tbaa !20
  %229 = icmp eq ptr %228, null
  br i1 %229, label %230, label %232

230:                                              ; preds = %227
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %231 unwind label %246

231:                                              ; preds = %230
  unreachable

232:                                              ; preds = %227
  %233 = load ptr, ptr %98, align 8, !tbaa !25
  %234 = invoke noundef i32 %233(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %10, ptr noundef nonnull align 8 dereferenceable(8) %11, ptr noundef nonnull align 4 dereferenceable(4) %12)
          to label %235 unwind label %246

235:                                              ; preds = %232
  call void @llvm.lifetime.end.p0(ptr nonnull %10)
  call void @llvm.lifetime.end.p0(ptr nonnull %11)
  call void @llvm.lifetime.end.p0(ptr nonnull %12)
  %236 = icmp eq i32 %226, %234
  br i1 %236, label %237, label %241

237:                                              ; preds = %235
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(4000) %64, i8 0, i64 4000, i1 false), !tbaa !43
  %238 = getelementptr inbounds nuw i8, ptr %63, i64 4
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(3992) %238, i8 0, i64 3992, i1 false), !tbaa !43
  store i32 -1, ptr %212, align 4, !tbaa !43
  store i32 -1, ptr %63, align 4, !tbaa !43
  call void @llvm.lifetime.start.p0(ptr nonnull %7)
  call void @llvm.lifetime.start.p0(ptr nonnull %8)
  call void @llvm.lifetime.start.p0(ptr nonnull %9)
  store ptr %63, ptr %7, align 8, !tbaa !45
  store ptr %64, ptr %8, align 8, !tbaa !45
  store i32 1000, ptr %9, align 4, !tbaa !43
  %239 = load ptr, ptr %82, align 8, !tbaa !20
  %240 = icmp eq ptr %239, null
  br i1 %240, label %248, label %250

241:                                              ; preds = %235
  %242 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.14)
          to label %243 unwind label %246

243:                                              ; preds = %241
  call void @exit(i32 noundef 1) #25
  unreachable

244:                                              ; preds = %224, %222
  %245 = landingpad { ptr, i32 }
          cleanup
  br label %271

246:                                              ; preds = %232, %230, %241
  %247 = landingpad { ptr, i32 }
          cleanup
  br label %271

248:                                              ; preds = %237
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %249 unwind label %266

249:                                              ; preds = %248
  unreachable

250:                                              ; preds = %237
  %251 = load ptr, ptr %88, align 8, !tbaa !25
  %252 = invoke noundef i32 %251(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %7, ptr noundef nonnull align 8 dereferenceable(8) %8, ptr noundef nonnull align 4 dereferenceable(4) %9)
          to label %253 unwind label %266

253:                                              ; preds = %250
  call void @llvm.lifetime.end.p0(ptr nonnull %7)
  call void @llvm.lifetime.end.p0(ptr nonnull %8)
  call void @llvm.lifetime.end.p0(ptr nonnull %9)
  call void @llvm.lifetime.start.p0(ptr nonnull %4)
  call void @llvm.lifetime.start.p0(ptr nonnull %5)
  call void @llvm.lifetime.start.p0(ptr nonnull %6)
  store ptr %63, ptr %4, align 8, !tbaa !45
  store ptr %64, ptr %5, align 8, !tbaa !45
  store i32 1000, ptr %6, align 4, !tbaa !43
  %254 = load ptr, ptr %92, align 8, !tbaa !20
  %255 = icmp eq ptr %254, null
  br i1 %255, label %256, label %258

256:                                              ; preds = %253
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %257 unwind label %268

257:                                              ; preds = %256
  unreachable

258:                                              ; preds = %253
  %259 = load ptr, ptr %98, align 8, !tbaa !25
  %260 = invoke noundef i32 %259(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %4, ptr noundef nonnull align 8 dereferenceable(8) %5, ptr noundef nonnull align 4 dereferenceable(4) %6)
          to label %261 unwind label %268

261:                                              ; preds = %258
  call void @llvm.lifetime.end.p0(ptr nonnull %4)
  call void @llvm.lifetime.end.p0(ptr nonnull %5)
  call void @llvm.lifetime.end.p0(ptr nonnull %6)
  %262 = icmp eq i32 %252, %260
  br i1 %262, label %270, label %263

263:                                              ; preds = %261
  %264 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.14)
          to label %265 unwind label %268

265:                                              ; preds = %263
  call void @exit(i32 noundef 1) #25
  unreachable

266:                                              ; preds = %250, %248
  %267 = landingpad { ptr, i32 }
          cleanup
  br label %271

268:                                              ; preds = %258, %256, %263
  %269 = landingpad { ptr, i32 }
          cleanup
  br label %271

270:                                              ; preds = %261
  call void @_ZdaPv(ptr noundef nonnull %64) #26
  call void @_ZdaPv(ptr noundef nonnull %63) #26
  ret void

271:                                              ; preds = %111, %113, %266, %268, %244, %246, %218, %220, %192, %194, %166, %168, %140, %142, %115, %117
  %272 = phi { ptr, i32 } [ %118, %117 ], [ %116, %115 ], [ %143, %142 ], [ %141, %140 ], [ %169, %168 ], [ %167, %166 ], [ %195, %194 ], [ %193, %192 ], [ %221, %220 ], [ %219, %218 ], [ %247, %246 ], [ %245, %244 ], [ %269, %268 ], [ %267, %266 ], [ %112, %111 ], [ %114, %113 ]
  call void @_ZdaPv(ptr noundef nonnull %64) #26
  br label %273

273:                                              ; preds = %271, %109
  %274 = phi { ptr, i32 } [ %272, %271 ], [ %110, %109 ]
  call void @_ZdaPv(ptr noundef nonnull %63) #26
  resume { ptr, i32 } %274
}

; Function Attrs: mustprogress norecurse uwtable
define internal fastcc void @_ZL19checkVectorFunctionIjfEvSt8functionIFT_PT0_S3_jEES5_PKc(ptr noundef nonnull %0, ptr noundef nonnull %1, ptr noundef %2) unnamed_addr #0 personality ptr @__gxx_personality_v0 {
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
  %40 = alloca ptr, align 8
  %41 = alloca ptr, align 8
  %42 = alloca i32, align 4
  %43 = alloca ptr, align 8
  %44 = alloca ptr, align 8
  %45 = alloca i32, align 4
  %46 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.12, i64 noundef 9)
  %47 = icmp eq ptr %2, null
  br i1 %47, label %48, label %56

48:                                               ; preds = %3
  %49 = load ptr, ptr @_ZSt4cout, align 8, !tbaa !31
  %50 = getelementptr i8, ptr %49, i64 -24
  %51 = load i64, ptr %50, align 8
  %52 = getelementptr inbounds i8, ptr @_ZSt4cout, i64 %51
  %53 = getelementptr inbounds nuw i8, ptr %52, i64 32
  %54 = load i32, ptr %53, align 8, !tbaa !33
  %55 = or i32 %54, 1
  tail call void @_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate(ptr noundef nonnull align 8 dereferenceable(264) %52, i32 noundef %55)
  br label %59

56:                                               ; preds = %3
  %57 = tail call noundef i64 @strlen(ptr noundef nonnull dereferenceable(1) %2) #21
  %58 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull %2, i64 noundef %57)
  br label %59

59:                                               ; preds = %48, %56
  %60 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.13, i64 noundef 1)
  %61 = tail call noalias noundef nonnull dereferenceable(4000) ptr @_Znam(i64 noundef 4000) #23
  %62 = invoke noalias noundef nonnull dereferenceable(4000) ptr @_Znam(i64 noundef 4000) #23
          to label %63 unwind label %96

63:                                               ; preds = %59
  tail call fastcc void @_ZL9init_dataIfEvRKSt10unique_ptrIA_T_St14default_deleteIS2_EEj(ptr nonnull %61)
  tail call fastcc void @_ZL9init_dataIfEvRKSt10unique_ptrIA_T_St14default_deleteIS2_EEj(ptr nonnull %62)
  call void @llvm.lifetime.start.p0(ptr nonnull %43)
  call void @llvm.lifetime.start.p0(ptr nonnull %44)
  call void @llvm.lifetime.start.p0(ptr nonnull %45)
  store ptr %61, ptr %43, align 8, !tbaa !55
  store ptr %62, ptr %44, align 8, !tbaa !55
  store i32 1000, ptr %45, align 4, !tbaa !43
  %64 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %65 = load ptr, ptr %64, align 8, !tbaa !20
  %66 = icmp eq ptr %65, null
  br i1 %66, label %67, label %69

67:                                               ; preds = %63
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %68 unwind label %98

68:                                               ; preds = %67
  unreachable

69:                                               ; preds = %63
  %70 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %71 = load ptr, ptr %70, align 8, !tbaa !27
  %72 = invoke noundef i32 %71(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %43, ptr noundef nonnull align 8 dereferenceable(8) %44, ptr noundef nonnull align 4 dereferenceable(4) %45)
          to label %73 unwind label %98

73:                                               ; preds = %69
  call void @llvm.lifetime.end.p0(ptr nonnull %43)
  call void @llvm.lifetime.end.p0(ptr nonnull %44)
  call void @llvm.lifetime.end.p0(ptr nonnull %45)
  call void @llvm.lifetime.start.p0(ptr nonnull %40)
  call void @llvm.lifetime.start.p0(ptr nonnull %41)
  call void @llvm.lifetime.start.p0(ptr nonnull %42)
  store ptr %61, ptr %40, align 8, !tbaa !55
  store ptr %62, ptr %41, align 8, !tbaa !55
  store i32 1000, ptr %42, align 4, !tbaa !43
  %74 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %75 = load ptr, ptr %74, align 8, !tbaa !20
  %76 = icmp eq ptr %75, null
  br i1 %76, label %77, label %79

77:                                               ; preds = %73
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %78 unwind label %100

78:                                               ; preds = %77
  unreachable

79:                                               ; preds = %73
  %80 = getelementptr inbounds nuw i8, ptr %1, i64 24
  %81 = load ptr, ptr %80, align 8, !tbaa !27
  %82 = invoke noundef i32 %81(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %40, ptr noundef nonnull align 8 dereferenceable(8) %41, ptr noundef nonnull align 4 dereferenceable(4) %42)
          to label %83 unwind label %100

83:                                               ; preds = %79
  call void @llvm.lifetime.end.p0(ptr nonnull %40)
  call void @llvm.lifetime.end.p0(ptr nonnull %41)
  call void @llvm.lifetime.end.p0(ptr nonnull %42)
  %84 = icmp eq i32 %72, %82
  br i1 %84, label %85, label %93

85:                                               ; preds = %83, %85
  %86 = phi i64 [ %91, %85 ], [ 0, %83 ]
  %87 = getelementptr inbounds nuw float, ptr %61, i64 %86
  %88 = getelementptr inbounds nuw i8, ptr %87, i64 16
  store <4 x float> splat (float 0x47EFFFFFE0000000), ptr %87, align 4, !tbaa !57
  store <4 x float> splat (float 0x47EFFFFFE0000000), ptr %88, align 4, !tbaa !57
  %89 = getelementptr inbounds nuw float, ptr %62, i64 %86
  %90 = getelementptr inbounds nuw i8, ptr %89, i64 16
  store <4 x float> splat (float 0x3810000000000000), ptr %89, align 4, !tbaa !57
  store <4 x float> splat (float 0x3810000000000000), ptr %90, align 4, !tbaa !57
  %91 = add nuw i64 %86, 8
  %92 = icmp eq i64 %91, 1000
  br i1 %92, label %102, label %85, !llvm.loop !80

93:                                               ; preds = %83
  %94 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.14)
          to label %95 unwind label %100

95:                                               ; preds = %93
  call void @exit(i32 noundef 1) #25
  unreachable

96:                                               ; preds = %59
  %97 = landingpad { ptr, i32 }
          cleanup
  br label %297

98:                                               ; preds = %69, %67
  %99 = landingpad { ptr, i32 }
          cleanup
  br label %295

100:                                              ; preds = %79, %77, %93
  %101 = landingpad { ptr, i32 }
          cleanup
  br label %295

102:                                              ; preds = %85
  call void @llvm.lifetime.start.p0(ptr nonnull %37)
  call void @llvm.lifetime.start.p0(ptr nonnull %38)
  call void @llvm.lifetime.start.p0(ptr nonnull %39)
  store ptr %61, ptr %37, align 8, !tbaa !55
  store ptr %62, ptr %38, align 8, !tbaa !55
  store i32 1000, ptr %39, align 4, !tbaa !43
  %103 = load ptr, ptr %64, align 8, !tbaa !20
  %104 = icmp eq ptr %103, null
  br i1 %104, label %105, label %107

105:                                              ; preds = %102
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %106 unwind label %131

106:                                              ; preds = %105
  unreachable

107:                                              ; preds = %102
  %108 = load ptr, ptr %70, align 8, !tbaa !27
  %109 = invoke noundef i32 %108(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %37, ptr noundef nonnull align 8 dereferenceable(8) %38, ptr noundef nonnull align 4 dereferenceable(4) %39)
          to label %110 unwind label %131

110:                                              ; preds = %107
  call void @llvm.lifetime.end.p0(ptr nonnull %37)
  call void @llvm.lifetime.end.p0(ptr nonnull %38)
  call void @llvm.lifetime.end.p0(ptr nonnull %39)
  call void @llvm.lifetime.start.p0(ptr nonnull %34)
  call void @llvm.lifetime.start.p0(ptr nonnull %35)
  call void @llvm.lifetime.start.p0(ptr nonnull %36)
  store ptr %61, ptr %34, align 8, !tbaa !55
  store ptr %62, ptr %35, align 8, !tbaa !55
  store i32 1000, ptr %36, align 4, !tbaa !43
  %111 = load ptr, ptr %74, align 8, !tbaa !20
  %112 = icmp eq ptr %111, null
  br i1 %112, label %113, label %115

113:                                              ; preds = %110
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %114 unwind label %133

114:                                              ; preds = %113
  unreachable

115:                                              ; preds = %110
  %116 = load ptr, ptr %80, align 8, !tbaa !27
  %117 = invoke noundef i32 %116(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %34, ptr noundef nonnull align 8 dereferenceable(8) %35, ptr noundef nonnull align 4 dereferenceable(4) %36)
          to label %118 unwind label %133

118:                                              ; preds = %115
  call void @llvm.lifetime.end.p0(ptr nonnull %34)
  call void @llvm.lifetime.end.p0(ptr nonnull %35)
  call void @llvm.lifetime.end.p0(ptr nonnull %36)
  %119 = icmp eq i32 %109, %117
  br i1 %119, label %120, label %128

120:                                              ; preds = %118, %120
  %121 = phi i64 [ %126, %120 ], [ 0, %118 ]
  %122 = getelementptr inbounds nuw float, ptr %61, i64 %121
  %123 = getelementptr inbounds nuw i8, ptr %122, i64 16
  store <4 x float> splat (float 0x3810000000000000), ptr %122, align 4, !tbaa !57
  store <4 x float> splat (float 0x3810000000000000), ptr %123, align 4, !tbaa !57
  %124 = getelementptr inbounds nuw float, ptr %62, i64 %121
  %125 = getelementptr inbounds nuw i8, ptr %124, i64 16
  store <4 x float> splat (float 0x47EFFFFFE0000000), ptr %124, align 4, !tbaa !57
  store <4 x float> splat (float 0x47EFFFFFE0000000), ptr %125, align 4, !tbaa !57
  %126 = add nuw i64 %121, 8
  %127 = icmp eq i64 %126, 1000
  br i1 %127, label %135, label %120, !llvm.loop !81

128:                                              ; preds = %118
  %129 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.14)
          to label %130 unwind label %133

130:                                              ; preds = %128
  call void @exit(i32 noundef 1) #25
  unreachable

131:                                              ; preds = %107, %105
  %132 = landingpad { ptr, i32 }
          cleanup
  br label %295

133:                                              ; preds = %115, %113, %128
  %134 = landingpad { ptr, i32 }
          cleanup
  br label %295

135:                                              ; preds = %120
  call void @llvm.lifetime.start.p0(ptr nonnull %31)
  call void @llvm.lifetime.start.p0(ptr nonnull %32)
  call void @llvm.lifetime.start.p0(ptr nonnull %33)
  store ptr %61, ptr %31, align 8, !tbaa !55
  store ptr %62, ptr %32, align 8, !tbaa !55
  store i32 1000, ptr %33, align 4, !tbaa !43
  %136 = load ptr, ptr %64, align 8, !tbaa !20
  %137 = icmp eq ptr %136, null
  br i1 %137, label %138, label %140

138:                                              ; preds = %135
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %139 unwind label %164

139:                                              ; preds = %138
  unreachable

140:                                              ; preds = %135
  %141 = load ptr, ptr %70, align 8, !tbaa !27
  %142 = invoke noundef i32 %141(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %31, ptr noundef nonnull align 8 dereferenceable(8) %32, ptr noundef nonnull align 4 dereferenceable(4) %33)
          to label %143 unwind label %164

143:                                              ; preds = %140
  call void @llvm.lifetime.end.p0(ptr nonnull %31)
  call void @llvm.lifetime.end.p0(ptr nonnull %32)
  call void @llvm.lifetime.end.p0(ptr nonnull %33)
  call void @llvm.lifetime.start.p0(ptr nonnull %28)
  call void @llvm.lifetime.start.p0(ptr nonnull %29)
  call void @llvm.lifetime.start.p0(ptr nonnull %30)
  store ptr %61, ptr %28, align 8, !tbaa !55
  store ptr %62, ptr %29, align 8, !tbaa !55
  store i32 1000, ptr %30, align 4, !tbaa !43
  %144 = load ptr, ptr %74, align 8, !tbaa !20
  %145 = icmp eq ptr %144, null
  br i1 %145, label %146, label %148

146:                                              ; preds = %143
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %147 unwind label %166

147:                                              ; preds = %146
  unreachable

148:                                              ; preds = %143
  %149 = load ptr, ptr %80, align 8, !tbaa !27
  %150 = invoke noundef i32 %149(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %28, ptr noundef nonnull align 8 dereferenceable(8) %29, ptr noundef nonnull align 4 dereferenceable(4) %30)
          to label %151 unwind label %166

151:                                              ; preds = %148
  call void @llvm.lifetime.end.p0(ptr nonnull %28)
  call void @llvm.lifetime.end.p0(ptr nonnull %29)
  call void @llvm.lifetime.end.p0(ptr nonnull %30)
  %152 = icmp eq i32 %142, %150
  br i1 %152, label %153, label %161

153:                                              ; preds = %151, %153
  %154 = phi i64 [ %159, %153 ], [ 0, %151 ]
  %155 = getelementptr inbounds nuw float, ptr %62, i64 %154
  %156 = getelementptr inbounds nuw i8, ptr %155, i64 16
  store <4 x float> splat (float 0x3810000000000000), ptr %155, align 4, !tbaa !57
  store <4 x float> splat (float 0x3810000000000000), ptr %156, align 4, !tbaa !57
  %157 = getelementptr inbounds nuw float, ptr %61, i64 %154
  %158 = getelementptr inbounds nuw i8, ptr %157, i64 16
  store <4 x float> splat (float 0x3810000000000000), ptr %157, align 4, !tbaa !57
  store <4 x float> splat (float 0x3810000000000000), ptr %158, align 4, !tbaa !57
  %159 = add nuw i64 %154, 8
  %160 = icmp eq i64 %159, 1000
  br i1 %160, label %168, label %153, !llvm.loop !82

161:                                              ; preds = %151
  %162 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.14)
          to label %163 unwind label %166

163:                                              ; preds = %161
  call void @exit(i32 noundef 1) #25
  unreachable

164:                                              ; preds = %140, %138
  %165 = landingpad { ptr, i32 }
          cleanup
  br label %295

166:                                              ; preds = %148, %146, %161
  %167 = landingpad { ptr, i32 }
          cleanup
  br label %295

168:                                              ; preds = %153
  %169 = getelementptr inbounds nuw i8, ptr %61, i64 3992
  store float 0x47EFFFFFE0000000, ptr %169, align 4, !tbaa !57
  call void @llvm.lifetime.start.p0(ptr nonnull %25)
  call void @llvm.lifetime.start.p0(ptr nonnull %26)
  call void @llvm.lifetime.start.p0(ptr nonnull %27)
  store ptr %61, ptr %25, align 8, !tbaa !55
  store ptr %62, ptr %26, align 8, !tbaa !55
  store i32 1000, ptr %27, align 4, !tbaa !43
  %170 = load ptr, ptr %64, align 8, !tbaa !20
  %171 = icmp eq ptr %170, null
  br i1 %171, label %172, label %174

172:                                              ; preds = %168
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %173 unwind label %198

173:                                              ; preds = %172
  unreachable

174:                                              ; preds = %168
  %175 = load ptr, ptr %70, align 8, !tbaa !27
  %176 = invoke noundef i32 %175(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %25, ptr noundef nonnull align 8 dereferenceable(8) %26, ptr noundef nonnull align 4 dereferenceable(4) %27)
          to label %177 unwind label %198

177:                                              ; preds = %174
  call void @llvm.lifetime.end.p0(ptr nonnull %25)
  call void @llvm.lifetime.end.p0(ptr nonnull %26)
  call void @llvm.lifetime.end.p0(ptr nonnull %27)
  call void @llvm.lifetime.start.p0(ptr nonnull %22)
  call void @llvm.lifetime.start.p0(ptr nonnull %23)
  call void @llvm.lifetime.start.p0(ptr nonnull %24)
  store ptr %61, ptr %22, align 8, !tbaa !55
  store ptr %62, ptr %23, align 8, !tbaa !55
  store i32 1000, ptr %24, align 4, !tbaa !43
  %178 = load ptr, ptr %74, align 8, !tbaa !20
  %179 = icmp eq ptr %178, null
  br i1 %179, label %180, label %182

180:                                              ; preds = %177
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %181 unwind label %200

181:                                              ; preds = %180
  unreachable

182:                                              ; preds = %177
  %183 = load ptr, ptr %80, align 8, !tbaa !27
  %184 = invoke noundef i32 %183(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %22, ptr noundef nonnull align 8 dereferenceable(8) %23, ptr noundef nonnull align 4 dereferenceable(4) %24)
          to label %185 unwind label %200

185:                                              ; preds = %182
  call void @llvm.lifetime.end.p0(ptr nonnull %22)
  call void @llvm.lifetime.end.p0(ptr nonnull %23)
  call void @llvm.lifetime.end.p0(ptr nonnull %24)
  %186 = icmp eq i32 %176, %184
  br i1 %186, label %187, label %195

187:                                              ; preds = %185, %187
  %188 = phi i64 [ %193, %187 ], [ 0, %185 ]
  %189 = getelementptr inbounds nuw float, ptr %62, i64 %188
  %190 = getelementptr inbounds nuw i8, ptr %189, i64 16
  store <4 x float> splat (float 0x3810000000000000), ptr %189, align 4, !tbaa !57
  store <4 x float> splat (float 0x3810000000000000), ptr %190, align 4, !tbaa !57
  %191 = getelementptr inbounds nuw float, ptr %61, i64 %188
  %192 = getelementptr inbounds nuw i8, ptr %191, i64 16
  store <4 x float> splat (float 0x3810000000000000), ptr %191, align 4, !tbaa !57
  store <4 x float> splat (float 0x3810000000000000), ptr %192, align 4, !tbaa !57
  %193 = add nuw i64 %188, 8
  %194 = icmp eq i64 %193, 1000
  br i1 %194, label %202, label %187, !llvm.loop !83

195:                                              ; preds = %185
  %196 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.14)
          to label %197 unwind label %200

197:                                              ; preds = %195
  call void @exit(i32 noundef 1) #25
  unreachable

198:                                              ; preds = %174, %172
  %199 = landingpad { ptr, i32 }
          cleanup
  br label %295

200:                                              ; preds = %182, %180, %195
  %201 = landingpad { ptr, i32 }
          cleanup
  br label %295

202:                                              ; preds = %187
  store float 0x47EFFFFFE0000000, ptr %61, align 4, !tbaa !57
  call void @llvm.lifetime.start.p0(ptr nonnull %19)
  call void @llvm.lifetime.start.p0(ptr nonnull %20)
  call void @llvm.lifetime.start.p0(ptr nonnull %21)
  store ptr %61, ptr %19, align 8, !tbaa !55
  store ptr %62, ptr %20, align 8, !tbaa !55
  store i32 1000, ptr %21, align 4, !tbaa !43
  %203 = load ptr, ptr %64, align 8, !tbaa !20
  %204 = icmp eq ptr %203, null
  br i1 %204, label %205, label %207

205:                                              ; preds = %202
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %206 unwind label %231

206:                                              ; preds = %205
  unreachable

207:                                              ; preds = %202
  %208 = load ptr, ptr %70, align 8, !tbaa !27
  %209 = invoke noundef i32 %208(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %19, ptr noundef nonnull align 8 dereferenceable(8) %20, ptr noundef nonnull align 4 dereferenceable(4) %21)
          to label %210 unwind label %231

210:                                              ; preds = %207
  call void @llvm.lifetime.end.p0(ptr nonnull %19)
  call void @llvm.lifetime.end.p0(ptr nonnull %20)
  call void @llvm.lifetime.end.p0(ptr nonnull %21)
  call void @llvm.lifetime.start.p0(ptr nonnull %16)
  call void @llvm.lifetime.start.p0(ptr nonnull %17)
  call void @llvm.lifetime.start.p0(ptr nonnull %18)
  store ptr %61, ptr %16, align 8, !tbaa !55
  store ptr %62, ptr %17, align 8, !tbaa !55
  store i32 1000, ptr %18, align 4, !tbaa !43
  %211 = load ptr, ptr %74, align 8, !tbaa !20
  %212 = icmp eq ptr %211, null
  br i1 %212, label %213, label %215

213:                                              ; preds = %210
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %214 unwind label %233

214:                                              ; preds = %213
  unreachable

215:                                              ; preds = %210
  %216 = load ptr, ptr %80, align 8, !tbaa !27
  %217 = invoke noundef i32 %216(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %16, ptr noundef nonnull align 8 dereferenceable(8) %17, ptr noundef nonnull align 4 dereferenceable(4) %18)
          to label %218 unwind label %233

218:                                              ; preds = %215
  call void @llvm.lifetime.end.p0(ptr nonnull %16)
  call void @llvm.lifetime.end.p0(ptr nonnull %17)
  call void @llvm.lifetime.end.p0(ptr nonnull %18)
  %219 = icmp eq i32 %209, %217
  br i1 %219, label %220, label %228

220:                                              ; preds = %218, %220
  %221 = phi i64 [ %226, %220 ], [ 0, %218 ]
  %222 = getelementptr inbounds nuw float, ptr %62, i64 %221
  %223 = getelementptr inbounds nuw i8, ptr %222, i64 16
  store <4 x float> splat (float 0x3810000000000000), ptr %222, align 4, !tbaa !57
  store <4 x float> splat (float 0x3810000000000000), ptr %223, align 4, !tbaa !57
  %224 = getelementptr inbounds nuw float, ptr %61, i64 %221
  %225 = getelementptr inbounds nuw i8, ptr %224, i64 16
  store <4 x float> splat (float 0x3810000000000000), ptr %224, align 4, !tbaa !57
  store <4 x float> splat (float 0x3810000000000000), ptr %225, align 4, !tbaa !57
  %226 = add nuw i64 %221, 8
  %227 = icmp eq i64 %226, 1000
  br i1 %227, label %235, label %220, !llvm.loop !84

228:                                              ; preds = %218
  %229 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.14)
          to label %230 unwind label %233

230:                                              ; preds = %228
  call void @exit(i32 noundef 1) #25
  unreachable

231:                                              ; preds = %207, %205
  %232 = landingpad { ptr, i32 }
          cleanup
  br label %295

233:                                              ; preds = %215, %213, %228
  %234 = landingpad { ptr, i32 }
          cleanup
  br label %295

235:                                              ; preds = %220
  %236 = getelementptr inbounds nuw i8, ptr %61, i64 3996
  store float 0x47EFFFFFE0000000, ptr %236, align 4, !tbaa !57
  call void @llvm.lifetime.start.p0(ptr nonnull %13)
  call void @llvm.lifetime.start.p0(ptr nonnull %14)
  call void @llvm.lifetime.start.p0(ptr nonnull %15)
  store ptr %61, ptr %13, align 8, !tbaa !55
  store ptr %62, ptr %14, align 8, !tbaa !55
  store i32 1000, ptr %15, align 4, !tbaa !43
  %237 = load ptr, ptr %64, align 8, !tbaa !20
  %238 = icmp eq ptr %237, null
  br i1 %238, label %239, label %241

239:                                              ; preds = %235
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %240 unwind label %265

240:                                              ; preds = %239
  unreachable

241:                                              ; preds = %235
  %242 = load ptr, ptr %70, align 8, !tbaa !27
  %243 = invoke noundef i32 %242(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %13, ptr noundef nonnull align 8 dereferenceable(8) %14, ptr noundef nonnull align 4 dereferenceable(4) %15)
          to label %244 unwind label %265

244:                                              ; preds = %241
  call void @llvm.lifetime.end.p0(ptr nonnull %13)
  call void @llvm.lifetime.end.p0(ptr nonnull %14)
  call void @llvm.lifetime.end.p0(ptr nonnull %15)
  call void @llvm.lifetime.start.p0(ptr nonnull %10)
  call void @llvm.lifetime.start.p0(ptr nonnull %11)
  call void @llvm.lifetime.start.p0(ptr nonnull %12)
  store ptr %61, ptr %10, align 8, !tbaa !55
  store ptr %62, ptr %11, align 8, !tbaa !55
  store i32 1000, ptr %12, align 4, !tbaa !43
  %245 = load ptr, ptr %74, align 8, !tbaa !20
  %246 = icmp eq ptr %245, null
  br i1 %246, label %247, label %249

247:                                              ; preds = %244
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %248 unwind label %267

248:                                              ; preds = %247
  unreachable

249:                                              ; preds = %244
  %250 = load ptr, ptr %80, align 8, !tbaa !27
  %251 = invoke noundef i32 %250(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %10, ptr noundef nonnull align 8 dereferenceable(8) %11, ptr noundef nonnull align 4 dereferenceable(4) %12)
          to label %252 unwind label %267

252:                                              ; preds = %249
  call void @llvm.lifetime.end.p0(ptr nonnull %10)
  call void @llvm.lifetime.end.p0(ptr nonnull %11)
  call void @llvm.lifetime.end.p0(ptr nonnull %12)
  %253 = icmp eq i32 %243, %251
  br i1 %253, label %254, label %262

254:                                              ; preds = %252, %254
  %255 = phi i64 [ %260, %254 ], [ 0, %252 ]
  %256 = getelementptr inbounds nuw float, ptr %62, i64 %255
  %257 = getelementptr inbounds nuw i8, ptr %256, i64 16
  store <4 x float> splat (float 0x3810000000000000), ptr %256, align 4, !tbaa !57
  store <4 x float> splat (float 0x3810000000000000), ptr %257, align 4, !tbaa !57
  %258 = getelementptr inbounds nuw float, ptr %61, i64 %255
  %259 = getelementptr inbounds nuw i8, ptr %258, i64 16
  store <4 x float> splat (float 0x3810000000000000), ptr %258, align 4, !tbaa !57
  store <4 x float> splat (float 0x3810000000000000), ptr %259, align 4, !tbaa !57
  %260 = add nuw i64 %255, 8
  %261 = icmp eq i64 %260, 1000
  br i1 %261, label %269, label %254, !llvm.loop !85

262:                                              ; preds = %252
  %263 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.14)
          to label %264 unwind label %267

264:                                              ; preds = %262
  call void @exit(i32 noundef 1) #25
  unreachable

265:                                              ; preds = %241, %239
  %266 = landingpad { ptr, i32 }
          cleanup
  br label %295

267:                                              ; preds = %249, %247, %262
  %268 = landingpad { ptr, i32 }
          cleanup
  br label %295

269:                                              ; preds = %254
  store float 0x47EFFFFFE0000000, ptr %236, align 4, !tbaa !57
  store float 0x47EFFFFFE0000000, ptr %61, align 4, !tbaa !57
  call void @llvm.lifetime.start.p0(ptr nonnull %7)
  call void @llvm.lifetime.start.p0(ptr nonnull %8)
  call void @llvm.lifetime.start.p0(ptr nonnull %9)
  store ptr %61, ptr %7, align 8, !tbaa !55
  store ptr %62, ptr %8, align 8, !tbaa !55
  store i32 1000, ptr %9, align 4, !tbaa !43
  %270 = load ptr, ptr %64, align 8, !tbaa !20
  %271 = icmp eq ptr %270, null
  br i1 %271, label %272, label %274

272:                                              ; preds = %269
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %273 unwind label %290

273:                                              ; preds = %272
  unreachable

274:                                              ; preds = %269
  %275 = load ptr, ptr %70, align 8, !tbaa !27
  %276 = invoke noundef i32 %275(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %7, ptr noundef nonnull align 8 dereferenceable(8) %8, ptr noundef nonnull align 4 dereferenceable(4) %9)
          to label %277 unwind label %290

277:                                              ; preds = %274
  call void @llvm.lifetime.end.p0(ptr nonnull %7)
  call void @llvm.lifetime.end.p0(ptr nonnull %8)
  call void @llvm.lifetime.end.p0(ptr nonnull %9)
  call void @llvm.lifetime.start.p0(ptr nonnull %4)
  call void @llvm.lifetime.start.p0(ptr nonnull %5)
  call void @llvm.lifetime.start.p0(ptr nonnull %6)
  store ptr %61, ptr %4, align 8, !tbaa !55
  store ptr %62, ptr %5, align 8, !tbaa !55
  store i32 1000, ptr %6, align 4, !tbaa !43
  %278 = load ptr, ptr %74, align 8, !tbaa !20
  %279 = icmp eq ptr %278, null
  br i1 %279, label %280, label %282

280:                                              ; preds = %277
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %281 unwind label %292

281:                                              ; preds = %280
  unreachable

282:                                              ; preds = %277
  %283 = load ptr, ptr %80, align 8, !tbaa !27
  %284 = invoke noundef i32 %283(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %4, ptr noundef nonnull align 8 dereferenceable(8) %5, ptr noundef nonnull align 4 dereferenceable(4) %6)
          to label %285 unwind label %292

285:                                              ; preds = %282
  call void @llvm.lifetime.end.p0(ptr nonnull %4)
  call void @llvm.lifetime.end.p0(ptr nonnull %5)
  call void @llvm.lifetime.end.p0(ptr nonnull %6)
  %286 = icmp eq i32 %276, %284
  br i1 %286, label %294, label %287

287:                                              ; preds = %285
  %288 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.14)
          to label %289 unwind label %292

289:                                              ; preds = %287
  call void @exit(i32 noundef 1) #25
  unreachable

290:                                              ; preds = %274, %272
  %291 = landingpad { ptr, i32 }
          cleanup
  br label %295

292:                                              ; preds = %282, %280, %287
  %293 = landingpad { ptr, i32 }
          cleanup
  br label %295

294:                                              ; preds = %285
  call void @_ZdaPv(ptr noundef nonnull %62) #26
  call void @_ZdaPv(ptr noundef nonnull %61) #26
  ret void

295:                                              ; preds = %290, %292, %265, %267, %231, %233, %198, %200, %164, %166, %131, %133, %98, %100
  %296 = phi { ptr, i32 } [ %101, %100 ], [ %99, %98 ], [ %134, %133 ], [ %132, %131 ], [ %167, %166 ], [ %165, %164 ], [ %201, %200 ], [ %199, %198 ], [ %234, %233 ], [ %232, %231 ], [ %268, %267 ], [ %266, %265 ], [ %293, %292 ], [ %291, %290 ]
  call void @_ZdaPv(ptr noundef nonnull %62) #26
  br label %297

297:                                              ; preds = %295, %96
  %298 = phi { ptr, i32 } [ %296, %295 ], [ %97, %96 ]
  call void @_ZdaPv(ptr noundef nonnull %61) #26
  resume { ptr, i32 } %298
}

; Function Attrs: mustprogress norecurse uwtable
define internal fastcc void @_ZL19checkVectorFunctionIttEvSt8functionIFT_PT0_S3_jEES5_PKc(ptr noundef nonnull %0, ptr noundef nonnull %1, ptr noundef %2) unnamed_addr #0 personality ptr @__gxx_personality_v0 {
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
  %40 = alloca ptr, align 8
  %41 = alloca ptr, align 8
  %42 = alloca i32, align 4
  %43 = alloca ptr, align 8
  %44 = alloca ptr, align 8
  %45 = alloca i32, align 4
  %46 = alloca %"class.std::uniform_int_distribution.84", align 4
  %47 = alloca %"class.std::uniform_int_distribution.84", align 4
  %48 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.12, i64 noundef 9)
  %49 = icmp eq ptr %2, null
  br i1 %49, label %50, label %58

50:                                               ; preds = %3
  %51 = load ptr, ptr @_ZSt4cout, align 8, !tbaa !31
  %52 = getelementptr i8, ptr %51, i64 -24
  %53 = load i64, ptr %52, align 8
  %54 = getelementptr inbounds i8, ptr @_ZSt4cout, i64 %53
  %55 = getelementptr inbounds nuw i8, ptr %54, i64 32
  %56 = load i32, ptr %55, align 8, !tbaa !33
  %57 = or i32 %56, 1
  tail call void @_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate(ptr noundef nonnull align 8 dereferenceable(264) %54, i32 noundef %57)
  br label %61

58:                                               ; preds = %3
  %59 = tail call noundef i64 @strlen(ptr noundef nonnull dereferenceable(1) %2) #21
  %60 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull %2, i64 noundef %59)
  br label %61

61:                                               ; preds = %50, %58
  %62 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.13, i64 noundef 1)
  %63 = tail call noalias noundef nonnull dereferenceable(2000) ptr @_Znam(i64 noundef 2000) #23
  %64 = invoke noalias noundef nonnull dereferenceable(2000) ptr @_Znam(i64 noundef 2000) #23
          to label %65 unwind label %111

65:                                               ; preds = %61
  call void @llvm.lifetime.start.p0(ptr nonnull %47) #21
  store i16 0, ptr %47, align 4, !tbaa !86
  %66 = getelementptr inbounds nuw i8, ptr %47, i64 2
  store i16 -1, ptr %66, align 2, !tbaa !88
  br label %67

67:                                               ; preds = %70, %65
  %68 = phi i64 [ 0, %65 ], [ %72, %70 ]
  %69 = invoke noundef i16 @_ZNSt24uniform_int_distributionItEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEtRT_RKNS0_10param_typeE(ptr noundef nonnull align 2 dereferenceable(4) %47, ptr noundef nonnull align 8 dereferenceable(5000) @_ZL3rng, ptr noundef nonnull align 2 dereferenceable(4) %47)
          to label %70 unwind label %115

70:                                               ; preds = %67
  %71 = getelementptr inbounds nuw i16, ptr %63, i64 %68
  store i16 %69, ptr %71, align 2, !tbaa !69
  %72 = add nuw nsw i64 %68, 1
  %73 = icmp eq i64 %72, 1000
  br i1 %73, label %74, label %67, !llvm.loop !89

74:                                               ; preds = %70
  call void @llvm.lifetime.end.p0(ptr nonnull %47) #21
  call void @llvm.lifetime.start.p0(ptr nonnull %46) #21
  store i16 0, ptr %46, align 4, !tbaa !86
  %75 = getelementptr inbounds nuw i8, ptr %46, i64 2
  store i16 -1, ptr %75, align 2, !tbaa !88
  br label %76

76:                                               ; preds = %79, %74
  %77 = phi i64 [ 0, %74 ], [ %81, %79 ]
  %78 = invoke noundef i16 @_ZNSt24uniform_int_distributionItEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEtRT_RKNS0_10param_typeE(ptr noundef nonnull align 2 dereferenceable(4) %46, ptr noundef nonnull align 8 dereferenceable(5000) @_ZL3rng, ptr noundef nonnull align 2 dereferenceable(4) %46)
          to label %79 unwind label %113

79:                                               ; preds = %76
  %80 = getelementptr inbounds nuw i16, ptr %64, i64 %77
  store i16 %78, ptr %80, align 2, !tbaa !69
  %81 = add nuw nsw i64 %77, 1
  %82 = icmp eq i64 %81, 1000
  br i1 %82, label %83, label %76, !llvm.loop !89

83:                                               ; preds = %79
  call void @llvm.lifetime.end.p0(ptr nonnull %46) #21
  call void @llvm.lifetime.start.p0(ptr nonnull %43)
  call void @llvm.lifetime.start.p0(ptr nonnull %44)
  call void @llvm.lifetime.start.p0(ptr nonnull %45)
  store ptr %63, ptr %43, align 8, !tbaa !71
  store ptr %64, ptr %44, align 8, !tbaa !71
  store i32 1000, ptr %45, align 4, !tbaa !43
  %84 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %85 = load ptr, ptr %84, align 8, !tbaa !20
  %86 = icmp eq ptr %85, null
  br i1 %86, label %87, label %89

87:                                               ; preds = %83
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %88 unwind label %117

88:                                               ; preds = %87
  unreachable

89:                                               ; preds = %83
  %90 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %91 = load ptr, ptr %90, align 8, !tbaa !29
  %92 = invoke noundef i16 %91(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %43, ptr noundef nonnull align 8 dereferenceable(8) %44, ptr noundef nonnull align 4 dereferenceable(4) %45)
          to label %93 unwind label %117

93:                                               ; preds = %89
  call void @llvm.lifetime.end.p0(ptr nonnull %43)
  call void @llvm.lifetime.end.p0(ptr nonnull %44)
  call void @llvm.lifetime.end.p0(ptr nonnull %45)
  call void @llvm.lifetime.start.p0(ptr nonnull %40)
  call void @llvm.lifetime.start.p0(ptr nonnull %41)
  call void @llvm.lifetime.start.p0(ptr nonnull %42)
  store ptr %63, ptr %40, align 8, !tbaa !71
  store ptr %64, ptr %41, align 8, !tbaa !71
  store i32 1000, ptr %42, align 4, !tbaa !43
  %94 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %95 = load ptr, ptr %94, align 8, !tbaa !20
  %96 = icmp eq ptr %95, null
  br i1 %96, label %97, label %99

97:                                               ; preds = %93
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %98 unwind label %119

98:                                               ; preds = %97
  unreachable

99:                                               ; preds = %93
  %100 = getelementptr inbounds nuw i8, ptr %1, i64 24
  %101 = load ptr, ptr %100, align 8, !tbaa !29
  %102 = invoke noundef i16 %101(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %40, ptr noundef nonnull align 8 dereferenceable(8) %41, ptr noundef nonnull align 4 dereferenceable(4) %42)
          to label %103 unwind label %119

103:                                              ; preds = %99
  call void @llvm.lifetime.end.p0(ptr nonnull %40)
  call void @llvm.lifetime.end.p0(ptr nonnull %41)
  call void @llvm.lifetime.end.p0(ptr nonnull %42)
  %104 = icmp eq i16 %92, %102
  br i1 %104, label %105, label %108

105:                                              ; preds = %103
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 2 dereferenceable(2000) %63, i8 -1, i64 2000, i1 false), !tbaa !69
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 2 dereferenceable(2000) %64, i8 0, i64 2000, i1 false), !tbaa !69
  call void @llvm.lifetime.start.p0(ptr nonnull %37)
  call void @llvm.lifetime.start.p0(ptr nonnull %38)
  call void @llvm.lifetime.start.p0(ptr nonnull %39)
  store ptr %63, ptr %37, align 8, !tbaa !71
  store ptr %64, ptr %38, align 8, !tbaa !71
  store i32 1000, ptr %39, align 4, !tbaa !43
  %106 = load ptr, ptr %84, align 8, !tbaa !20
  %107 = icmp eq ptr %106, null
  br i1 %107, label %121, label %123

108:                                              ; preds = %103
  %109 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.14)
          to label %110 unwind label %119

110:                                              ; preds = %108
  call void @exit(i32 noundef 1) #25
  unreachable

111:                                              ; preds = %61
  %112 = landingpad { ptr, i32 }
          cleanup
  br label %275

113:                                              ; preds = %76
  %114 = landingpad { ptr, i32 }
          cleanup
  br label %273

115:                                              ; preds = %67
  %116 = landingpad { ptr, i32 }
          cleanup
  br label %273

117:                                              ; preds = %89, %87
  %118 = landingpad { ptr, i32 }
          cleanup
  br label %273

119:                                              ; preds = %99, %97, %108
  %120 = landingpad { ptr, i32 }
          cleanup
  br label %273

121:                                              ; preds = %105
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %122 unwind label %142

122:                                              ; preds = %121
  unreachable

123:                                              ; preds = %105
  %124 = load ptr, ptr %90, align 8, !tbaa !29
  %125 = invoke noundef i16 %124(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %37, ptr noundef nonnull align 8 dereferenceable(8) %38, ptr noundef nonnull align 4 dereferenceable(4) %39)
          to label %126 unwind label %142

126:                                              ; preds = %123
  call void @llvm.lifetime.end.p0(ptr nonnull %37)
  call void @llvm.lifetime.end.p0(ptr nonnull %38)
  call void @llvm.lifetime.end.p0(ptr nonnull %39)
  call void @llvm.lifetime.start.p0(ptr nonnull %34)
  call void @llvm.lifetime.start.p0(ptr nonnull %35)
  call void @llvm.lifetime.start.p0(ptr nonnull %36)
  store ptr %63, ptr %34, align 8, !tbaa !71
  store ptr %64, ptr %35, align 8, !tbaa !71
  store i32 1000, ptr %36, align 4, !tbaa !43
  %127 = load ptr, ptr %94, align 8, !tbaa !20
  %128 = icmp eq ptr %127, null
  br i1 %128, label %129, label %131

129:                                              ; preds = %126
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %130 unwind label %144

130:                                              ; preds = %129
  unreachable

131:                                              ; preds = %126
  %132 = load ptr, ptr %100, align 8, !tbaa !29
  %133 = invoke noundef i16 %132(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %34, ptr noundef nonnull align 8 dereferenceable(8) %35, ptr noundef nonnull align 4 dereferenceable(4) %36)
          to label %134 unwind label %144

134:                                              ; preds = %131
  call void @llvm.lifetime.end.p0(ptr nonnull %34)
  call void @llvm.lifetime.end.p0(ptr nonnull %35)
  call void @llvm.lifetime.end.p0(ptr nonnull %36)
  %135 = icmp eq i16 %125, %133
  br i1 %135, label %136, label %139

136:                                              ; preds = %134
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 2 dereferenceable(2000) %63, i8 0, i64 2000, i1 false), !tbaa !69
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 2 dereferenceable(2000) %64, i8 -1, i64 2000, i1 false), !tbaa !69
  call void @llvm.lifetime.start.p0(ptr nonnull %31)
  call void @llvm.lifetime.start.p0(ptr nonnull %32)
  call void @llvm.lifetime.start.p0(ptr nonnull %33)
  store ptr %63, ptr %31, align 8, !tbaa !71
  store ptr %64, ptr %32, align 8, !tbaa !71
  store i32 1000, ptr %33, align 4, !tbaa !43
  %137 = load ptr, ptr %84, align 8, !tbaa !20
  %138 = icmp eq ptr %137, null
  br i1 %138, label %146, label %148

139:                                              ; preds = %134
  %140 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.14)
          to label %141 unwind label %144

141:                                              ; preds = %139
  call void @exit(i32 noundef 1) #25
  unreachable

142:                                              ; preds = %123, %121
  %143 = landingpad { ptr, i32 }
          cleanup
  br label %273

144:                                              ; preds = %131, %129, %139
  %145 = landingpad { ptr, i32 }
          cleanup
  br label %273

146:                                              ; preds = %136
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %147 unwind label %168

147:                                              ; preds = %146
  unreachable

148:                                              ; preds = %136
  %149 = load ptr, ptr %90, align 8, !tbaa !29
  %150 = invoke noundef i16 %149(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %31, ptr noundef nonnull align 8 dereferenceable(8) %32, ptr noundef nonnull align 4 dereferenceable(4) %33)
          to label %151 unwind label %168

151:                                              ; preds = %148
  call void @llvm.lifetime.end.p0(ptr nonnull %31)
  call void @llvm.lifetime.end.p0(ptr nonnull %32)
  call void @llvm.lifetime.end.p0(ptr nonnull %33)
  call void @llvm.lifetime.start.p0(ptr nonnull %28)
  call void @llvm.lifetime.start.p0(ptr nonnull %29)
  call void @llvm.lifetime.start.p0(ptr nonnull %30)
  store ptr %63, ptr %28, align 8, !tbaa !71
  store ptr %64, ptr %29, align 8, !tbaa !71
  store i32 1000, ptr %30, align 4, !tbaa !43
  %152 = load ptr, ptr %94, align 8, !tbaa !20
  %153 = icmp eq ptr %152, null
  br i1 %153, label %154, label %156

154:                                              ; preds = %151
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %155 unwind label %170

155:                                              ; preds = %154
  unreachable

156:                                              ; preds = %151
  %157 = load ptr, ptr %100, align 8, !tbaa !29
  %158 = invoke noundef i16 %157(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %28, ptr noundef nonnull align 8 dereferenceable(8) %29, ptr noundef nonnull align 4 dereferenceable(4) %30)
          to label %159 unwind label %170

159:                                              ; preds = %156
  call void @llvm.lifetime.end.p0(ptr nonnull %28)
  call void @llvm.lifetime.end.p0(ptr nonnull %29)
  call void @llvm.lifetime.end.p0(ptr nonnull %30)
  %160 = icmp eq i16 %150, %158
  br i1 %160, label %161, label %165

161:                                              ; preds = %159
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 2 dereferenceable(2000) %64, i8 0, i64 2000, i1 false), !tbaa !69
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 2 dereferenceable(2000) %63, i8 0, i64 2000, i1 false), !tbaa !69
  %162 = getelementptr inbounds nuw i8, ptr %63, i64 1996
  store i16 -1, ptr %162, align 2, !tbaa !69
  call void @llvm.lifetime.start.p0(ptr nonnull %25)
  call void @llvm.lifetime.start.p0(ptr nonnull %26)
  call void @llvm.lifetime.start.p0(ptr nonnull %27)
  store ptr %63, ptr %25, align 8, !tbaa !71
  store ptr %64, ptr %26, align 8, !tbaa !71
  store i32 1000, ptr %27, align 4, !tbaa !43
  %163 = load ptr, ptr %84, align 8, !tbaa !20
  %164 = icmp eq ptr %163, null
  br i1 %164, label %172, label %174

165:                                              ; preds = %159
  %166 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.14)
          to label %167 unwind label %170

167:                                              ; preds = %165
  call void @exit(i32 noundef 1) #25
  unreachable

168:                                              ; preds = %148, %146
  %169 = landingpad { ptr, i32 }
          cleanup
  br label %273

170:                                              ; preds = %156, %154, %165
  %171 = landingpad { ptr, i32 }
          cleanup
  br label %273

172:                                              ; preds = %161
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %173 unwind label %194

173:                                              ; preds = %172
  unreachable

174:                                              ; preds = %161
  %175 = load ptr, ptr %90, align 8, !tbaa !29
  %176 = invoke noundef i16 %175(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %25, ptr noundef nonnull align 8 dereferenceable(8) %26, ptr noundef nonnull align 4 dereferenceable(4) %27)
          to label %177 unwind label %194

177:                                              ; preds = %174
  call void @llvm.lifetime.end.p0(ptr nonnull %25)
  call void @llvm.lifetime.end.p0(ptr nonnull %26)
  call void @llvm.lifetime.end.p0(ptr nonnull %27)
  call void @llvm.lifetime.start.p0(ptr nonnull %22)
  call void @llvm.lifetime.start.p0(ptr nonnull %23)
  call void @llvm.lifetime.start.p0(ptr nonnull %24)
  store ptr %63, ptr %22, align 8, !tbaa !71
  store ptr %64, ptr %23, align 8, !tbaa !71
  store i32 1000, ptr %24, align 4, !tbaa !43
  %178 = load ptr, ptr %94, align 8, !tbaa !20
  %179 = icmp eq ptr %178, null
  br i1 %179, label %180, label %182

180:                                              ; preds = %177
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %181 unwind label %196

181:                                              ; preds = %180
  unreachable

182:                                              ; preds = %177
  %183 = load ptr, ptr %100, align 8, !tbaa !29
  %184 = invoke noundef i16 %183(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %22, ptr noundef nonnull align 8 dereferenceable(8) %23, ptr noundef nonnull align 4 dereferenceable(4) %24)
          to label %185 unwind label %196

185:                                              ; preds = %182
  call void @llvm.lifetime.end.p0(ptr nonnull %22)
  call void @llvm.lifetime.end.p0(ptr nonnull %23)
  call void @llvm.lifetime.end.p0(ptr nonnull %24)
  %186 = icmp eq i16 %176, %184
  br i1 %186, label %187, label %191

187:                                              ; preds = %185
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 2 dereferenceable(2000) %64, i8 0, i64 2000, i1 false), !tbaa !69
  %188 = getelementptr inbounds nuw i8, ptr %63, i64 2
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 2 dereferenceable(1998) %188, i8 0, i64 1998, i1 false), !tbaa !69
  store i16 -1, ptr %63, align 2, !tbaa !69
  call void @llvm.lifetime.start.p0(ptr nonnull %19)
  call void @llvm.lifetime.start.p0(ptr nonnull %20)
  call void @llvm.lifetime.start.p0(ptr nonnull %21)
  store ptr %63, ptr %19, align 8, !tbaa !71
  store ptr %64, ptr %20, align 8, !tbaa !71
  store i32 1000, ptr %21, align 4, !tbaa !43
  %189 = load ptr, ptr %84, align 8, !tbaa !20
  %190 = icmp eq ptr %189, null
  br i1 %190, label %198, label %200

191:                                              ; preds = %185
  %192 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.14)
          to label %193 unwind label %196

193:                                              ; preds = %191
  call void @exit(i32 noundef 1) #25
  unreachable

194:                                              ; preds = %174, %172
  %195 = landingpad { ptr, i32 }
          cleanup
  br label %273

196:                                              ; preds = %182, %180, %191
  %197 = landingpad { ptr, i32 }
          cleanup
  br label %273

198:                                              ; preds = %187
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %199 unwind label %220

199:                                              ; preds = %198
  unreachable

200:                                              ; preds = %187
  %201 = load ptr, ptr %90, align 8, !tbaa !29
  %202 = invoke noundef i16 %201(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %19, ptr noundef nonnull align 8 dereferenceable(8) %20, ptr noundef nonnull align 4 dereferenceable(4) %21)
          to label %203 unwind label %220

203:                                              ; preds = %200
  call void @llvm.lifetime.end.p0(ptr nonnull %19)
  call void @llvm.lifetime.end.p0(ptr nonnull %20)
  call void @llvm.lifetime.end.p0(ptr nonnull %21)
  call void @llvm.lifetime.start.p0(ptr nonnull %16)
  call void @llvm.lifetime.start.p0(ptr nonnull %17)
  call void @llvm.lifetime.start.p0(ptr nonnull %18)
  store ptr %63, ptr %16, align 8, !tbaa !71
  store ptr %64, ptr %17, align 8, !tbaa !71
  store i32 1000, ptr %18, align 4, !tbaa !43
  %204 = load ptr, ptr %94, align 8, !tbaa !20
  %205 = icmp eq ptr %204, null
  br i1 %205, label %206, label %208

206:                                              ; preds = %203
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %207 unwind label %222

207:                                              ; preds = %206
  unreachable

208:                                              ; preds = %203
  %209 = load ptr, ptr %100, align 8, !tbaa !29
  %210 = invoke noundef i16 %209(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %16, ptr noundef nonnull align 8 dereferenceable(8) %17, ptr noundef nonnull align 4 dereferenceable(4) %18)
          to label %211 unwind label %222

211:                                              ; preds = %208
  call void @llvm.lifetime.end.p0(ptr nonnull %16)
  call void @llvm.lifetime.end.p0(ptr nonnull %17)
  call void @llvm.lifetime.end.p0(ptr nonnull %18)
  %212 = icmp eq i16 %202, %210
  br i1 %212, label %213, label %217

213:                                              ; preds = %211
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 2 dereferenceable(2000) %64, i8 0, i64 2000, i1 false), !tbaa !69
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 2 dereferenceable(2000) %63, i8 0, i64 1998, i1 false), !tbaa !69
  %214 = getelementptr inbounds nuw i8, ptr %63, i64 1998
  store i16 -1, ptr %214, align 2, !tbaa !69
  call void @llvm.lifetime.start.p0(ptr nonnull %13)
  call void @llvm.lifetime.start.p0(ptr nonnull %14)
  call void @llvm.lifetime.start.p0(ptr nonnull %15)
  store ptr %63, ptr %13, align 8, !tbaa !71
  store ptr %64, ptr %14, align 8, !tbaa !71
  store i32 1000, ptr %15, align 4, !tbaa !43
  %215 = load ptr, ptr %84, align 8, !tbaa !20
  %216 = icmp eq ptr %215, null
  br i1 %216, label %224, label %226

217:                                              ; preds = %211
  %218 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.14)
          to label %219 unwind label %222

219:                                              ; preds = %217
  call void @exit(i32 noundef 1) #25
  unreachable

220:                                              ; preds = %200, %198
  %221 = landingpad { ptr, i32 }
          cleanup
  br label %273

222:                                              ; preds = %208, %206, %217
  %223 = landingpad { ptr, i32 }
          cleanup
  br label %273

224:                                              ; preds = %213
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %225 unwind label %246

225:                                              ; preds = %224
  unreachable

226:                                              ; preds = %213
  %227 = load ptr, ptr %90, align 8, !tbaa !29
  %228 = invoke noundef i16 %227(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %13, ptr noundef nonnull align 8 dereferenceable(8) %14, ptr noundef nonnull align 4 dereferenceable(4) %15)
          to label %229 unwind label %246

229:                                              ; preds = %226
  call void @llvm.lifetime.end.p0(ptr nonnull %13)
  call void @llvm.lifetime.end.p0(ptr nonnull %14)
  call void @llvm.lifetime.end.p0(ptr nonnull %15)
  call void @llvm.lifetime.start.p0(ptr nonnull %10)
  call void @llvm.lifetime.start.p0(ptr nonnull %11)
  call void @llvm.lifetime.start.p0(ptr nonnull %12)
  store ptr %63, ptr %10, align 8, !tbaa !71
  store ptr %64, ptr %11, align 8, !tbaa !71
  store i32 1000, ptr %12, align 4, !tbaa !43
  %230 = load ptr, ptr %94, align 8, !tbaa !20
  %231 = icmp eq ptr %230, null
  br i1 %231, label %232, label %234

232:                                              ; preds = %229
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %233 unwind label %248

233:                                              ; preds = %232
  unreachable

234:                                              ; preds = %229
  %235 = load ptr, ptr %100, align 8, !tbaa !29
  %236 = invoke noundef i16 %235(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %10, ptr noundef nonnull align 8 dereferenceable(8) %11, ptr noundef nonnull align 4 dereferenceable(4) %12)
          to label %237 unwind label %248

237:                                              ; preds = %234
  call void @llvm.lifetime.end.p0(ptr nonnull %10)
  call void @llvm.lifetime.end.p0(ptr nonnull %11)
  call void @llvm.lifetime.end.p0(ptr nonnull %12)
  %238 = icmp eq i16 %228, %236
  br i1 %238, label %239, label %243

239:                                              ; preds = %237
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 2 dereferenceable(2000) %64, i8 0, i64 2000, i1 false), !tbaa !69
  %240 = getelementptr inbounds nuw i8, ptr %63, i64 2
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 2 dereferenceable(1996) %240, i8 0, i64 1996, i1 false), !tbaa !69
  store i16 -1, ptr %214, align 2, !tbaa !69
  store i16 -1, ptr %63, align 2, !tbaa !69
  call void @llvm.lifetime.start.p0(ptr nonnull %7)
  call void @llvm.lifetime.start.p0(ptr nonnull %8)
  call void @llvm.lifetime.start.p0(ptr nonnull %9)
  store ptr %63, ptr %7, align 8, !tbaa !71
  store ptr %64, ptr %8, align 8, !tbaa !71
  store i32 1000, ptr %9, align 4, !tbaa !43
  %241 = load ptr, ptr %84, align 8, !tbaa !20
  %242 = icmp eq ptr %241, null
  br i1 %242, label %250, label %252

243:                                              ; preds = %237
  %244 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.14)
          to label %245 unwind label %248

245:                                              ; preds = %243
  call void @exit(i32 noundef 1) #25
  unreachable

246:                                              ; preds = %226, %224
  %247 = landingpad { ptr, i32 }
          cleanup
  br label %273

248:                                              ; preds = %234, %232, %243
  %249 = landingpad { ptr, i32 }
          cleanup
  br label %273

250:                                              ; preds = %239
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %251 unwind label %268

251:                                              ; preds = %250
  unreachable

252:                                              ; preds = %239
  %253 = load ptr, ptr %90, align 8, !tbaa !29
  %254 = invoke noundef i16 %253(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %7, ptr noundef nonnull align 8 dereferenceable(8) %8, ptr noundef nonnull align 4 dereferenceable(4) %9)
          to label %255 unwind label %268

255:                                              ; preds = %252
  call void @llvm.lifetime.end.p0(ptr nonnull %7)
  call void @llvm.lifetime.end.p0(ptr nonnull %8)
  call void @llvm.lifetime.end.p0(ptr nonnull %9)
  call void @llvm.lifetime.start.p0(ptr nonnull %4)
  call void @llvm.lifetime.start.p0(ptr nonnull %5)
  call void @llvm.lifetime.start.p0(ptr nonnull %6)
  store ptr %63, ptr %4, align 8, !tbaa !71
  store ptr %64, ptr %5, align 8, !tbaa !71
  store i32 1000, ptr %6, align 4, !tbaa !43
  %256 = load ptr, ptr %94, align 8, !tbaa !20
  %257 = icmp eq ptr %256, null
  br i1 %257, label %258, label %260

258:                                              ; preds = %255
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %259 unwind label %270

259:                                              ; preds = %258
  unreachable

260:                                              ; preds = %255
  %261 = load ptr, ptr %100, align 8, !tbaa !29
  %262 = invoke noundef i16 %261(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %4, ptr noundef nonnull align 8 dereferenceable(8) %5, ptr noundef nonnull align 4 dereferenceable(4) %6)
          to label %263 unwind label %270

263:                                              ; preds = %260
  call void @llvm.lifetime.end.p0(ptr nonnull %4)
  call void @llvm.lifetime.end.p0(ptr nonnull %5)
  call void @llvm.lifetime.end.p0(ptr nonnull %6)
  %264 = icmp eq i16 %254, %262
  br i1 %264, label %272, label %265

265:                                              ; preds = %263
  %266 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.14)
          to label %267 unwind label %270

267:                                              ; preds = %265
  call void @exit(i32 noundef 1) #25
  unreachable

268:                                              ; preds = %252, %250
  %269 = landingpad { ptr, i32 }
          cleanup
  br label %273

270:                                              ; preds = %260, %258, %265
  %271 = landingpad { ptr, i32 }
          cleanup
  br label %273

272:                                              ; preds = %263
  call void @_ZdaPv(ptr noundef nonnull %64) #26
  call void @_ZdaPv(ptr noundef nonnull %63) #26
  ret void

273:                                              ; preds = %113, %115, %268, %270, %246, %248, %220, %222, %194, %196, %168, %170, %142, %144, %117, %119
  %274 = phi { ptr, i32 } [ %120, %119 ], [ %118, %117 ], [ %145, %144 ], [ %143, %142 ], [ %171, %170 ], [ %169, %168 ], [ %197, %196 ], [ %195, %194 ], [ %223, %222 ], [ %221, %220 ], [ %249, %248 ], [ %247, %246 ], [ %271, %270 ], [ %269, %268 ], [ %114, %113 ], [ %116, %115 ]
  call void @_ZdaPv(ptr noundef nonnull %64) #26
  br label %275

275:                                              ; preds = %273, %111
  %276 = phi { ptr, i32 } [ %274, %273 ], [ %112, %111 ]
  call void @_ZdaPv(ptr noundef nonnull %63) #26
  resume { ptr, i32 } %276
}

; Function Attrs: noinline noreturn nounwind uwtable
define linkonce_odr hidden void @__clang_call_terminate(ptr noundef %0) local_unnamed_addr #3 comdat {
  %2 = tail call ptr @__cxa_begin_catch(ptr %0) #21
  tail call void @_ZSt9terminatev() #22
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
define linkonce_odr dso_local noundef i32 @_ZNSt24uniform_int_distributionIiEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEiRT_RKNS0_10param_typeE(ptr noundef nonnull align 4 dereferenceable(8) %0, ptr noundef nonnull align 8 dereferenceable(5000) %1, ptr noundef nonnull align 4 dereferenceable(8) %2) local_unnamed_addr #10 comdat {
  %4 = alloca %"struct.std::uniform_int_distribution<>::param_type", align 8
  %5 = getelementptr inbounds nuw i8, ptr %2, i64 4
  %6 = load i32, ptr %5, align 4, !tbaa !90
  %7 = sext i32 %6 to i64
  %8 = load i32, ptr %2, align 4, !tbaa !92
  %9 = sext i32 %8 to i64
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
  br i1 %28, label %24, label %29, !llvm.loop !93

29:                                               ; preds = %24, %12, %20
  %30 = phi i64 [ %17, %12 ], [ %17, %20 ], [ %26, %24 ]
  %31 = lshr i64 %30, 32
  br label %45

32:                                               ; preds = %3
  %33 = icmp eq i64 %10, 4294967295
  br i1 %33, label %43, label %34

34:                                               ; preds = %32, %34
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #21
  store <2 x i32> <i32 0, i32 -1>, ptr %4, align 8, !tbaa !43
  %35 = call noundef i32 @_ZNSt24uniform_int_distributionIiEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEiRT_RKNS0_10param_typeE(ptr noundef nonnull align 4 dereferenceable(8) %0, ptr noundef nonnull align 8 dereferenceable(5000) %1, ptr noundef nonnull align 4 dereferenceable(8) %4)
  %36 = sext i32 %35 to i64
  %37 = shl nsw i64 %36, 32
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #21
  %38 = call noundef i64 @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv(ptr noundef nonnull align 8 dereferenceable(5000) %1)
  %39 = add i64 %37, %38
  %40 = icmp ugt i64 %39, %10
  %41 = icmp ult i64 %39, %37
  %42 = or i1 %40, %41
  br i1 %42, label %34, label %45, !llvm.loop !94

43:                                               ; preds = %32
  %44 = tail call noundef i64 @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv(ptr noundef nonnull align 8 dereferenceable(5000) %1)
  br label %45

45:                                               ; preds = %34, %43, %29
  %46 = phi i64 [ %31, %29 ], [ %44, %43 ], [ %39, %34 ]
  %47 = load i32, ptr %2, align 4, !tbaa !92
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
  br i1 %43, label %44, label %8, !llvm.loop !95

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
  br i1 %111, label %112, label %91, !llvm.loop !96

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

; Function Attrs: nobuiltin nounwind
declare void @_ZdaPv(ptr noundef) local_unnamed_addr #12

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #13

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef range(i32 -1, 4) i32 @"_ZNSt17_Function_handlerIFiPiS0_jEZ4mainE3$_0E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !45
  %6 = load ptr, ptr %2, align 8, !tbaa !45
  %7 = load i32, ptr %3, align 4, !tbaa !43
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %22, label %9

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 0, %9 ], [ %20, %11 ]
  %13 = phi i32 [ -1, %9 ], [ %19, %11 ]
  %14 = getelementptr inbounds nuw i32, ptr %5, i64 %12
  %15 = load i32, ptr %14, align 4, !tbaa !43
  %16 = getelementptr inbounds nuw i32, ptr %6, i64 %12
  %17 = load i32, ptr %16, align 4, !tbaa !43
  %18 = icmp sgt i32 %15, %17
  %19 = select i1 %18, i32 3, i32 %13
  %20 = add nuw nsw i64 %12, 1
  %21 = icmp eq i64 %20, %10
  br i1 %21, label %22, label %11, !llvm.loop !97

22:                                               ; preds = %11, %4
  %23 = phi i32 [ -1, %4 ], [ %19, %11 ]
  ret i32 %23
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFiPiS0_jEZ4mainE3$_0E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_0", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !100
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef range(i32 -1, 4) i32 @"_ZNSt17_Function_handlerIFiPiS0_jEZ4mainE3$_1E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !45
  %6 = load ptr, ptr %2, align 8, !tbaa !45
  %7 = load i32, ptr %3, align 4, !tbaa !43
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %56, label %9

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  %11 = icmp ult i32 %7, 8
  br i1 %11, label %42, label %12

12:                                               ; preds = %9
  %13 = and i64 %10, 4294967288
  br label %14

14:                                               ; preds = %14, %12
  %15 = phi i64 [ 0, %12 ], [ %34, %14 ]
  %16 = phi <4 x i1> [ zeroinitializer, %12 ], [ %32, %14 ]
  %17 = phi <4 x i1> [ zeroinitializer, %12 ], [ %33, %14 ]
  %18 = getelementptr inbounds nuw i32, ptr %5, i64 %15
  %19 = getelementptr inbounds nuw i8, ptr %18, i64 16
  %20 = load <4 x i32>, ptr %18, align 4, !tbaa !43
  %21 = load <4 x i32>, ptr %19, align 4, !tbaa !43
  %22 = getelementptr inbounds nuw i32, ptr %6, i64 %15
  %23 = getelementptr inbounds nuw i8, ptr %22, i64 16
  %24 = load <4 x i32>, ptr %22, align 4, !tbaa !43
  %25 = load <4 x i32>, ptr %23, align 4, !tbaa !43
  %26 = freeze <4 x i32> %20
  %27 = freeze <4 x i32> %24
  %28 = icmp sgt <4 x i32> %26, %27
  %29 = freeze <4 x i32> %21
  %30 = freeze <4 x i32> %25
  %31 = icmp sgt <4 x i32> %29, %30
  %32 = or <4 x i1> %16, %28
  %33 = or <4 x i1> %17, %31
  %34 = add nuw i64 %15, 8
  %35 = icmp eq i64 %34, %13
  br i1 %35, label %36, label %14, !llvm.loop !101

36:                                               ; preds = %14
  %37 = or <4 x i1> %33, %32
  %38 = bitcast <4 x i1> %37 to i4
  %39 = icmp eq i4 %38, 0
  %40 = select i1 %39, i32 -1, i32 3
  %41 = icmp eq i64 %13, %10
  br i1 %41, label %56, label %42

42:                                               ; preds = %9, %36
  %43 = phi i64 [ 0, %9 ], [ %13, %36 ]
  %44 = phi i32 [ -1, %9 ], [ %40, %36 ]
  br label %45

45:                                               ; preds = %42, %45
  %46 = phi i64 [ %54, %45 ], [ %43, %42 ]
  %47 = phi i32 [ %53, %45 ], [ %44, %42 ]
  %48 = getelementptr inbounds nuw i32, ptr %5, i64 %46
  %49 = load i32, ptr %48, align 4, !tbaa !43
  %50 = getelementptr inbounds nuw i32, ptr %6, i64 %46
  %51 = load i32, ptr %50, align 4, !tbaa !43
  %52 = icmp sgt i32 %49, %51
  %53 = select i1 %52, i32 3, i32 %47
  %54 = add nuw nsw i64 %46, 1
  %55 = icmp eq i64 %54, %10
  br i1 %55, label %56, label %45, !llvm.loop !102

56:                                               ; preds = %45, %36, %4
  %57 = phi i32 [ -1, %4 ], [ %40, %36 ], [ %53, %45 ]
  ret i32 %57
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFiPiS0_jEZ4mainE3$_1E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_1", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !100
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress norecurse nounwind uwtable
define internal fastcc void @_ZL9init_dataIfEvRKSt10unique_ptrIA_T_St14default_deleteIS2_EEj(ptr writeonly captures(none) %0) unnamed_addr #16 {
  %2 = tail call fp128 @llvm.log.f128(fp128 0xL0000000000000000401F000000000000), !tbaa !43
  %3 = tail call fp128 @llvm.log.f128(fp128 0xL00000000000000004000000000000000), !tbaa !43
  %4 = fdiv fp128 %2, %3
  %5 = fptoui fp128 %4 to i64
  %6 = add i64 %5, 23
  %7 = udiv i64 %6, %5
  %8 = tail call i64 @llvm.umax.i64(i64 %7, i64 1)
  %9 = load i64, ptr getelementptr inbounds nuw (i8, ptr @_ZL3rng, i64 4992), align 8, !tbaa !12
  br label %11

10:                                               ; preds = %156
  ret void

11:                                               ; preds = %1, %156
  %12 = phi i64 [ %9, %1 ], [ %135, %156 ]
  %13 = phi i64 [ 0, %1 ], [ %160, %156 ]
  br label %17

14:                                               ; preds = %133
  %15 = fdiv float %150, %151
  %16 = fcmp ult float %15, 1.000000e+00
  br i1 %16, label %156, label %154, !prof !103

17:                                               ; preds = %133, %11
  %18 = phi i64 [ %12, %11 ], [ %135, %133 ]
  %19 = phi i64 [ %8, %11 ], [ %152, %133 ]
  %20 = phi float [ 1.000000e+00, %11 ], [ %151, %133 ]
  %21 = phi float [ 0.000000e+00, %11 ], [ %150, %133 ]
  %22 = icmp ugt i64 %18, 623
  br i1 %22, label %23, label %133

23:                                               ; preds = %17
  %24 = load i64, ptr @_ZL3rng, align 8, !tbaa !6
  %25 = insertelement <2 x i64> poison, i64 %24, i64 1
  br label %26

26:                                               ; preds = %26, %23
  %27 = phi i64 [ 0, %23 ], [ %60, %26 ]
  %28 = phi <2 x i64> [ %25, %23 ], [ %34, %26 ]
  %29 = getelementptr inbounds nuw i64, ptr @_ZL3rng, i64 %27
  %30 = getelementptr inbounds nuw i64, ptr @_ZL3rng, i64 %27
  %31 = getelementptr inbounds nuw i8, ptr %30, i64 8
  %32 = getelementptr inbounds nuw i8, ptr %30, i64 24
  %33 = load <2 x i64>, ptr %31, align 8, !tbaa !6
  %34 = load <2 x i64>, ptr %32, align 8, !tbaa !6
  %35 = shufflevector <2 x i64> %28, <2 x i64> %33, <2 x i32> <i32 1, i32 2>
  %36 = shufflevector <2 x i64> %33, <2 x i64> %34, <2 x i32> <i32 1, i32 2>
  %37 = and <2 x i64> %35, splat (i64 -2147483648)
  %38 = and <2 x i64> %36, splat (i64 -2147483648)
  %39 = and <2 x i64> %33, splat (i64 2147483646)
  %40 = and <2 x i64> %34, splat (i64 2147483646)
  %41 = or disjoint <2 x i64> %39, %37
  %42 = or disjoint <2 x i64> %40, %38
  %43 = getelementptr inbounds nuw i8, ptr %29, i64 3176
  %44 = getelementptr inbounds nuw i8, ptr %29, i64 3192
  %45 = load <2 x i64>, ptr %43, align 8, !tbaa !6
  %46 = load <2 x i64>, ptr %44, align 8, !tbaa !6
  %47 = lshr exact <2 x i64> %41, splat (i64 1)
  %48 = lshr exact <2 x i64> %42, splat (i64 1)
  %49 = xor <2 x i64> %47, %45
  %50 = xor <2 x i64> %48, %46
  %51 = and <2 x i64> %33, splat (i64 1)
  %52 = and <2 x i64> %34, splat (i64 1)
  %53 = icmp eq <2 x i64> %51, zeroinitializer
  %54 = icmp eq <2 x i64> %52, zeroinitializer
  %55 = select <2 x i1> %53, <2 x i64> zeroinitializer, <2 x i64> splat (i64 2567483615)
  %56 = select <2 x i1> %54, <2 x i64> zeroinitializer, <2 x i64> splat (i64 2567483615)
  %57 = xor <2 x i64> %49, %55
  %58 = xor <2 x i64> %50, %56
  %59 = getelementptr inbounds nuw i8, ptr %29, i64 16
  store <2 x i64> %57, ptr %29, align 8, !tbaa !6
  store <2 x i64> %58, ptr %59, align 8, !tbaa !6
  %60 = add nuw i64 %27, 4
  %61 = icmp eq i64 %60, 224
  br i1 %61, label %62, label %26, !llvm.loop !104

62:                                               ; preds = %26
  %63 = extractelement <2 x i64> %34, i64 1
  %64 = and i64 %63, -2147483648
  %65 = load i64, ptr getelementptr inbounds nuw (i8, ptr @_ZL3rng, i64 1800), align 8, !tbaa !6
  %66 = and i64 %65, 2147483646
  %67 = or disjoint i64 %66, %64
  %68 = load i64, ptr getelementptr inbounds nuw (i8, ptr @_ZL3rng, i64 4968), align 8, !tbaa !6
  %69 = lshr exact i64 %67, 1
  %70 = xor i64 %69, %68
  %71 = and i64 %65, 1
  %72 = icmp eq i64 %71, 0
  %73 = select i1 %72, i64 0, i64 2567483615
  %74 = xor i64 %70, %73
  store i64 %74, ptr getelementptr inbounds nuw (i8, ptr @_ZL3rng, i64 1792), align 8, !tbaa !6
  %75 = and i64 %65, -2147483648
  %76 = load i64, ptr getelementptr inbounds nuw (i8, ptr @_ZL3rng, i64 1808), align 8, !tbaa !6
  %77 = and i64 %76, 2147483646
  %78 = or disjoint i64 %77, %75
  %79 = load i64, ptr getelementptr inbounds nuw (i8, ptr @_ZL3rng, i64 4976), align 8, !tbaa !6
  %80 = lshr exact i64 %78, 1
  %81 = xor i64 %80, %79
  %82 = and i64 %76, 1
  %83 = icmp eq i64 %82, 0
  %84 = select i1 %83, i64 0, i64 2567483615
  %85 = xor i64 %81, %84
  store i64 %85, ptr getelementptr inbounds nuw (i8, ptr @_ZL3rng, i64 1800), align 8, !tbaa !6
  %86 = and i64 %76, -2147483648
  %87 = load i64, ptr getelementptr inbounds nuw (i8, ptr @_ZL3rng, i64 1816), align 8, !tbaa !6
  %88 = and i64 %87, 2147483646
  %89 = or disjoint i64 %88, %86
  %90 = load i64, ptr getelementptr inbounds nuw (i8, ptr @_ZL3rng, i64 4984), align 8, !tbaa !6
  %91 = lshr exact i64 %89, 1
  %92 = xor i64 %91, %90
  %93 = and i64 %87, 1
  %94 = icmp eq i64 %93, 0
  %95 = select i1 %94, i64 0, i64 2567483615
  %96 = xor i64 %92, %95
  store i64 %96, ptr getelementptr inbounds nuw (i8, ptr @_ZL3rng, i64 1808), align 8, !tbaa !6
  %97 = load i64, ptr getelementptr inbounds nuw (i8, ptr @_ZL3rng, i64 1816), align 8, !tbaa !6
  %98 = insertelement <2 x i64> poison, i64 %97, i64 1
  br label %99

99:                                               ; preds = %99, %62
  %100 = phi i64 [ 0, %62 ], [ %118, %99 ]
  %101 = phi <2 x i64> [ %98, %62 ], [ %106, %99 ]
  %102 = getelementptr i64, ptr @_ZL3rng, i64 %100
  %103 = getelementptr i8, ptr %102, i64 1816
  %104 = getelementptr i64, ptr @_ZL3rng, i64 %100
  %105 = getelementptr i8, ptr %104, i64 1824
  %106 = load <2 x i64>, ptr %105, align 8, !tbaa !6
  %107 = shufflevector <2 x i64> %101, <2 x i64> %106, <2 x i32> <i32 1, i32 2>
  %108 = and <2 x i64> %107, splat (i64 -2147483648)
  %109 = and <2 x i64> %106, splat (i64 2147483646)
  %110 = or disjoint <2 x i64> %109, %108
  %111 = load <2 x i64>, ptr %102, align 8, !tbaa !6
  %112 = lshr exact <2 x i64> %110, splat (i64 1)
  %113 = xor <2 x i64> %112, %111
  %114 = and <2 x i64> %106, splat (i64 1)
  %115 = icmp eq <2 x i64> %114, zeroinitializer
  %116 = select <2 x i1> %115, <2 x i64> zeroinitializer, <2 x i64> splat (i64 2567483615)
  %117 = xor <2 x i64> %113, %116
  store <2 x i64> %117, ptr %103, align 8, !tbaa !6
  %118 = add nuw i64 %100, 2
  %119 = icmp eq i64 %118, 396
  br i1 %119, label %120, label %99, !llvm.loop !105

120:                                              ; preds = %99
  %121 = load i64, ptr getelementptr inbounds nuw (i8, ptr @_ZL3rng, i64 4984), align 8, !tbaa !6
  %122 = and i64 %121, -2147483648
  %123 = load i64, ptr @_ZL3rng, align 8, !tbaa !6
  %124 = and i64 %123, 2147483646
  %125 = or disjoint i64 %124, %122
  %126 = load i64, ptr getelementptr inbounds nuw (i8, ptr @_ZL3rng, i64 3168), align 8, !tbaa !6
  %127 = lshr exact i64 %125, 1
  %128 = xor i64 %127, %126
  %129 = and i64 %123, 1
  %130 = icmp eq i64 %129, 0
  %131 = select i1 %130, i64 0, i64 2567483615
  %132 = xor i64 %128, %131
  store i64 %132, ptr getelementptr inbounds nuw (i8, ptr @_ZL3rng, i64 4984), align 8, !tbaa !6
  br label %133

133:                                              ; preds = %17, %120
  %134 = phi i64 [ 0, %120 ], [ %18, %17 ]
  %135 = add nuw nsw i64 %134, 1
  store i64 %135, ptr getelementptr inbounds nuw (i8, ptr @_ZL3rng, i64 4992), align 8, !tbaa !12
  %136 = getelementptr inbounds nuw i64, ptr @_ZL3rng, i64 %134
  %137 = load i64, ptr %136, align 8, !tbaa !6
  %138 = lshr i64 %137, 11
  %139 = and i64 %138, 4294967295
  %140 = xor i64 %139, %137
  %141 = shl i64 %140, 7
  %142 = and i64 %141, 2636928640
  %143 = xor i64 %142, %140
  %144 = shl i64 %143, 15
  %145 = and i64 %144, 4022730752
  %146 = xor i64 %145, %143
  %147 = lshr i64 %146, 18
  %148 = xor i64 %147, %146
  %149 = uitofp i64 %148 to float
  %150 = tail call float @llvm.fmuladd.f32(float %149, float %20, float %21)
  %151 = fmul float %20, 0x41F0000000000000
  %152 = add i64 %19, -1
  %153 = icmp eq i64 %152, 0
  br i1 %153, label %14, label %17, !llvm.loop !106

154:                                              ; preds = %14
  %155 = tail call noundef float @nextafterf(float noundef 1.000000e+00, float noundef 0.000000e+00) #21, !tbaa !43
  br label %156

156:                                              ; preds = %14, %154
  %157 = phi float [ %155, %154 ], [ %15, %14 ]
  %158 = tail call noundef float @llvm.fmuladd.f32(float %157, float 0x47EFFFFFE0000000, float 0x3810000000000000)
  %159 = getelementptr inbounds nuw float, ptr %0, i64 %13
  store float %158, ptr %159, align 4, !tbaa !57
  %160 = add nuw nsw i64 %13, 1
  %161 = icmp eq i64 %160, 1000
  br i1 %161, label %10, label %11, !llvm.loop !107
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fmuladd.f32(float, float, float) #17

; Function Attrs: nounwind
declare float @nextafterf(float noundef, float noundef) local_unnamed_addr #18

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef range(i32 -1, 4) i32 @"_ZNSt17_Function_handlerIFiPfS0_jEZ4mainE3$_0E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !55
  %6 = load ptr, ptr %2, align 8, !tbaa !55
  %7 = load i32, ptr %3, align 4, !tbaa !43
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %22, label %9

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 0, %9 ], [ %20, %11 ]
  %13 = phi i32 [ -1, %9 ], [ %19, %11 ]
  %14 = getelementptr inbounds nuw float, ptr %5, i64 %12
  %15 = load float, ptr %14, align 4, !tbaa !57
  %16 = getelementptr inbounds nuw float, ptr %6, i64 %12
  %17 = load float, ptr %16, align 4, !tbaa !57
  %18 = fcmp ogt float %15, %17
  %19 = select i1 %18, i32 3, i32 %13
  %20 = add nuw nsw i64 %12, 1
  %21 = icmp eq i64 %20, %10
  br i1 %21, label %22, label %11, !llvm.loop !108

22:                                               ; preds = %11, %4
  %23 = phi i32 [ -1, %4 ], [ %19, %11 ]
  ret i32 %23
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFiPfS0_jEZ4mainE3$_0E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_0", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !100
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef range(i32 -1, 4) i32 @"_ZNSt17_Function_handlerIFiPfS0_jEZ4mainE3$_1E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !55
  %6 = load ptr, ptr %2, align 8, !tbaa !55
  %7 = load i32, ptr %3, align 4, !tbaa !43
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %56, label %9

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  %11 = icmp ult i32 %7, 8
  br i1 %11, label %42, label %12

12:                                               ; preds = %9
  %13 = and i64 %10, 4294967288
  br label %14

14:                                               ; preds = %14, %12
  %15 = phi i64 [ 0, %12 ], [ %34, %14 ]
  %16 = phi <4 x i1> [ zeroinitializer, %12 ], [ %32, %14 ]
  %17 = phi <4 x i1> [ zeroinitializer, %12 ], [ %33, %14 ]
  %18 = getelementptr inbounds nuw float, ptr %5, i64 %15
  %19 = getelementptr inbounds nuw i8, ptr %18, i64 16
  %20 = load <4 x float>, ptr %18, align 4, !tbaa !57
  %21 = load <4 x float>, ptr %19, align 4, !tbaa !57
  %22 = getelementptr inbounds nuw float, ptr %6, i64 %15
  %23 = getelementptr inbounds nuw i8, ptr %22, i64 16
  %24 = load <4 x float>, ptr %22, align 4, !tbaa !57
  %25 = load <4 x float>, ptr %23, align 4, !tbaa !57
  %26 = freeze <4 x float> %20
  %27 = freeze <4 x float> %24
  %28 = fcmp ogt <4 x float> %26, %27
  %29 = freeze <4 x float> %21
  %30 = freeze <4 x float> %25
  %31 = fcmp ogt <4 x float> %29, %30
  %32 = or <4 x i1> %16, %28
  %33 = or <4 x i1> %17, %31
  %34 = add nuw i64 %15, 8
  %35 = icmp eq i64 %34, %13
  br i1 %35, label %36, label %14, !llvm.loop !109

36:                                               ; preds = %14
  %37 = or <4 x i1> %33, %32
  %38 = bitcast <4 x i1> %37 to i4
  %39 = icmp eq i4 %38, 0
  %40 = select i1 %39, i32 -1, i32 3
  %41 = icmp eq i64 %13, %10
  br i1 %41, label %56, label %42

42:                                               ; preds = %9, %36
  %43 = phi i64 [ 0, %9 ], [ %13, %36 ]
  %44 = phi i32 [ -1, %9 ], [ %40, %36 ]
  br label %45

45:                                               ; preds = %42, %45
  %46 = phi i64 [ %54, %45 ], [ %43, %42 ]
  %47 = phi i32 [ %53, %45 ], [ %44, %42 ]
  %48 = getelementptr inbounds nuw float, ptr %5, i64 %46
  %49 = load float, ptr %48, align 4, !tbaa !57
  %50 = getelementptr inbounds nuw float, ptr %6, i64 %46
  %51 = load float, ptr %50, align 4, !tbaa !57
  %52 = fcmp ogt float %49, %51
  %53 = select i1 %52, i32 3, i32 %47
  %54 = add nuw nsw i64 %46, 1
  %55 = icmp eq i64 %54, %10
  br i1 %55, label %56, label %45, !llvm.loop !110

56:                                               ; preds = %45, %36, %4
  %57 = phi i32 [ -1, %4 ], [ %40, %36 ], [ %53, %45 ]
  ret i32 %57
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFiPfS0_jEZ4mainE3$_1E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_1", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !100
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local noundef i16 @_ZNSt24uniform_int_distributionIsEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEsRT_RKNS0_10param_typeE(ptr noundef nonnull align 2 dereferenceable(4) %0, ptr noundef nonnull align 8 dereferenceable(5000) %1, ptr noundef nonnull align 2 dereferenceable(4) %2) local_unnamed_addr #10 comdat {
  %4 = alloca %"struct.std::uniform_int_distribution<short>::param_type", align 4
  %5 = getelementptr inbounds nuw i8, ptr %2, i64 2
  %6 = load i16, ptr %5, align 2, !tbaa !68
  %7 = sext i16 %6 to i64
  %8 = load i16, ptr %2, align 2, !tbaa !65
  %9 = sext i16 %8 to i64
  %10 = sub nsw i64 %7, %9
  %11 = icmp ult i64 %10, 4294967295
  br i1 %11, label %14, label %12

12:                                               ; preds = %3
  %13 = getelementptr inbounds nuw i8, ptr %4, i64 2
  br label %34

14:                                               ; preds = %3
  %15 = trunc nuw nsw i64 %10 to i32
  %16 = add nuw nsw i32 %15, 1
  %17 = tail call noundef i64 @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv(ptr noundef nonnull align 8 dereferenceable(5000) %1)
  %18 = zext nneg i32 %16 to i64
  %19 = mul i64 %17, %18
  %20 = trunc i64 %19 to i32
  %21 = icmp ult i32 %15, %20
  br i1 %21, label %31, label %22

22:                                               ; preds = %14
  %23 = xor i32 %15, -1
  %24 = urem i32 %23, %16
  %25 = icmp samesign ugt i32 %24, %20
  br i1 %25, label %26, label %31

26:                                               ; preds = %22, %26
  %27 = tail call noundef i64 @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv(ptr noundef nonnull align 8 dereferenceable(5000) %1)
  %28 = mul i64 %27, %18
  %29 = trunc i64 %28 to i32
  %30 = icmp ugt i32 %24, %29
  br i1 %30, label %26, label %31, !llvm.loop !111

31:                                               ; preds = %26, %14, %22
  %32 = phi i64 [ %19, %14 ], [ %19, %22 ], [ %28, %26 ]
  %33 = lshr i64 %32, 32
  br label %43

34:                                               ; preds = %12, %34
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #21
  store i16 0, ptr %4, align 4, !tbaa !65
  store i16 -1, ptr %13, align 2, !tbaa !68
  %35 = call noundef i16 @_ZNSt24uniform_int_distributionIsEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEsRT_RKNS0_10param_typeE(ptr noundef nonnull align 2 dereferenceable(4) %0, ptr noundef nonnull align 8 dereferenceable(5000) %1, ptr noundef nonnull align 2 dereferenceable(4) %4)
  %36 = sext i16 %35 to i64
  %37 = shl nsw i64 %36, 32
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #21
  %38 = call noundef i64 @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv(ptr noundef nonnull align 8 dereferenceable(5000) %1)
  %39 = add i64 %37, %38
  %40 = icmp ugt i64 %39, %10
  %41 = icmp ult i64 %39, %37
  %42 = or i1 %40, %41
  br i1 %42, label %34, label %43, !llvm.loop !112

43:                                               ; preds = %34, %31
  %44 = phi i64 [ %33, %31 ], [ %39, %34 ]
  %45 = load i16, ptr %2, align 2, !tbaa !65
  %46 = trunc i64 %44 to i16
  %47 = add i16 %45, %46
  ret i16 %47
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef range(i16 -1, 4) i16 @"_ZNSt17_Function_handlerIFsPsS0_jEZ4mainE3$_2E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !71
  %6 = load ptr, ptr %2, align 8, !tbaa !71
  %7 = load i32, ptr %3, align 4, !tbaa !43
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %22, label %9

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 0, %9 ], [ %20, %11 ]
  %13 = phi i16 [ -1, %9 ], [ %19, %11 ]
  %14 = getelementptr inbounds nuw i16, ptr %5, i64 %12
  %15 = load i16, ptr %14, align 2, !tbaa !69
  %16 = getelementptr inbounds nuw i16, ptr %6, i64 %12
  %17 = load i16, ptr %16, align 2, !tbaa !69
  %18 = icmp sgt i16 %15, %17
  %19 = select i1 %18, i16 3, i16 %13
  %20 = add nuw nsw i64 %12, 1
  %21 = icmp eq i64 %20, %10
  br i1 %21, label %22, label %11, !llvm.loop !113

22:                                               ; preds = %11, %4
  %23 = phi i16 [ -1, %4 ], [ %19, %11 ]
  ret i16 %23
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFsPsS0_jEZ4mainE3$_2E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_2", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !100
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef range(i16 -1, 4) i16 @"_ZNSt17_Function_handlerIFsPsS0_jEZ4mainE3$_3E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !71
  %6 = load ptr, ptr %2, align 8, !tbaa !71
  %7 = load i32, ptr %3, align 4, !tbaa !43
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %87, label %9

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  %11 = icmp ult i32 %7, 4
  br i1 %11, label %73, label %12

12:                                               ; preds = %9
  %13 = icmp ult i32 %7, 16
  br i1 %13, label %47, label %14

14:                                               ; preds = %12
  %15 = and i64 %10, 4294967280
  br label %16

16:                                               ; preds = %16, %14
  %17 = phi i64 [ 0, %14 ], [ %36, %16 ]
  %18 = phi <8 x i1> [ zeroinitializer, %14 ], [ %34, %16 ]
  %19 = phi <8 x i1> [ zeroinitializer, %14 ], [ %35, %16 ]
  %20 = getelementptr inbounds nuw i16, ptr %5, i64 %17
  %21 = getelementptr inbounds nuw i8, ptr %20, i64 16
  %22 = load <8 x i16>, ptr %20, align 2, !tbaa !69
  %23 = load <8 x i16>, ptr %21, align 2, !tbaa !69
  %24 = getelementptr inbounds nuw i16, ptr %6, i64 %17
  %25 = getelementptr inbounds nuw i8, ptr %24, i64 16
  %26 = load <8 x i16>, ptr %24, align 2, !tbaa !69
  %27 = load <8 x i16>, ptr %25, align 2, !tbaa !69
  %28 = freeze <8 x i16> %22
  %29 = freeze <8 x i16> %26
  %30 = icmp sgt <8 x i16> %28, %29
  %31 = freeze <8 x i16> %23
  %32 = freeze <8 x i16> %27
  %33 = icmp sgt <8 x i16> %31, %32
  %34 = or <8 x i1> %18, %30
  %35 = or <8 x i1> %19, %33
  %36 = add nuw i64 %17, 16
  %37 = icmp eq i64 %36, %15
  br i1 %37, label %38, label %16, !llvm.loop !114

38:                                               ; preds = %16
  %39 = or <8 x i1> %35, %34
  %40 = bitcast <8 x i1> %39 to i8
  %41 = icmp eq i8 %40, 0
  %42 = select i1 %41, i16 -1, i16 3
  %43 = icmp eq i64 %15, %10
  br i1 %43, label %87, label %44

44:                                               ; preds = %38
  %45 = and i64 %10, 12
  %46 = icmp eq i64 %45, 0
  br i1 %46, label %73, label %47

47:                                               ; preds = %44, %12
  %48 = phi i64 [ %15, %44 ], [ 0, %12 ]
  %49 = phi i16 [ %42, %44 ], [ -1, %12 ]
  %50 = freeze i16 %49
  %51 = icmp ne i16 %50, -1
  %52 = and i64 %10, 4294967292
  %53 = insertelement <4 x i1> poison, i1 %51, i64 0
  %54 = shufflevector <4 x i1> %53, <4 x i1> poison, <4 x i32> zeroinitializer
  br label %55

55:                                               ; preds = %55, %47
  %56 = phi i64 [ %48, %47 ], [ %66, %55 ]
  %57 = phi <4 x i1> [ %54, %47 ], [ %65, %55 ]
  %58 = getelementptr inbounds nuw i16, ptr %5, i64 %56
  %59 = load <4 x i16>, ptr %58, align 2, !tbaa !69
  %60 = getelementptr inbounds nuw i16, ptr %6, i64 %56
  %61 = load <4 x i16>, ptr %60, align 2, !tbaa !69
  %62 = freeze <4 x i16> %59
  %63 = freeze <4 x i16> %61
  %64 = icmp sgt <4 x i16> %62, %63
  %65 = or <4 x i1> %57, %64
  %66 = add nuw i64 %56, 4
  %67 = icmp eq i64 %66, %52
  br i1 %67, label %68, label %55, !llvm.loop !115

68:                                               ; preds = %55
  %69 = bitcast <4 x i1> %65 to i4
  %70 = icmp eq i4 %69, 0
  %71 = select i1 %70, i16 -1, i16 3
  %72 = icmp eq i64 %52, %10
  br i1 %72, label %87, label %73

73:                                               ; preds = %44, %68, %9
  %74 = phi i64 [ 0, %9 ], [ %15, %44 ], [ %52, %68 ]
  %75 = phi i16 [ -1, %9 ], [ %42, %44 ], [ %71, %68 ]
  br label %76

76:                                               ; preds = %73, %76
  %77 = phi i64 [ %85, %76 ], [ %74, %73 ]
  %78 = phi i16 [ %84, %76 ], [ %75, %73 ]
  %79 = getelementptr inbounds nuw i16, ptr %5, i64 %77
  %80 = load i16, ptr %79, align 2, !tbaa !69
  %81 = getelementptr inbounds nuw i16, ptr %6, i64 %77
  %82 = load i16, ptr %81, align 2, !tbaa !69
  %83 = icmp sgt i16 %80, %82
  %84 = select i1 %83, i16 3, i16 %78
  %85 = add nuw nsw i64 %77, 1
  %86 = icmp eq i64 %85, %10
  br i1 %86, label %87, label %76, !llvm.loop !116

87:                                               ; preds = %76, %38, %68, %4
  %88 = phi i16 [ -1, %4 ], [ %42, %38 ], [ %71, %68 ], [ %84, %76 ]
  ret i16 %88
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFsPsS0_jEZ4mainE3$_3E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_3", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !100
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef range(i32 -1, 4) i32 @"_ZNSt17_Function_handlerIFiPiS0_jEZ4mainE3$_4E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !45
  %6 = load ptr, ptr %2, align 8, !tbaa !45
  %7 = load i32, ptr %3, align 4, !tbaa !43
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %22, label %9

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 0, %9 ], [ %20, %11 ]
  %13 = phi i32 [ -1, %9 ], [ %19, %11 ]
  %14 = getelementptr inbounds nuw i32, ptr %5, i64 %12
  %15 = load i32, ptr %14, align 4, !tbaa !43
  %16 = getelementptr inbounds nuw i32, ptr %6, i64 %12
  %17 = load i32, ptr %16, align 4, !tbaa !43
  %18 = icmp sgt i32 %15, %17
  %19 = select i1 %18, i32 %13, i32 3
  %20 = add nuw nsw i64 %12, 1
  %21 = icmp eq i64 %20, %10
  br i1 %21, label %22, label %11, !llvm.loop !117

22:                                               ; preds = %11, %4
  %23 = phi i32 [ -1, %4 ], [ %19, %11 ]
  ret i32 %23
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFiPiS0_jEZ4mainE3$_4E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_4", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !100
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef range(i32 -1, 4) i32 @"_ZNSt17_Function_handlerIFiPiS0_jEZ4mainE3$_5E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !45
  %6 = load ptr, ptr %2, align 8, !tbaa !45
  %7 = load i32, ptr %3, align 4, !tbaa !43
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %56, label %9

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  %11 = icmp ult i32 %7, 8
  br i1 %11, label %42, label %12

12:                                               ; preds = %9
  %13 = and i64 %10, 4294967288
  br label %14

14:                                               ; preds = %14, %12
  %15 = phi i64 [ 0, %12 ], [ %34, %14 ]
  %16 = phi <4 x i1> [ zeroinitializer, %12 ], [ %32, %14 ]
  %17 = phi <4 x i1> [ zeroinitializer, %12 ], [ %33, %14 ]
  %18 = getelementptr inbounds nuw i32, ptr %5, i64 %15
  %19 = getelementptr inbounds nuw i8, ptr %18, i64 16
  %20 = load <4 x i32>, ptr %18, align 4, !tbaa !43
  %21 = load <4 x i32>, ptr %19, align 4, !tbaa !43
  %22 = getelementptr inbounds nuw i32, ptr %6, i64 %15
  %23 = getelementptr inbounds nuw i8, ptr %22, i64 16
  %24 = load <4 x i32>, ptr %22, align 4, !tbaa !43
  %25 = load <4 x i32>, ptr %23, align 4, !tbaa !43
  %26 = freeze <4 x i32> %20
  %27 = freeze <4 x i32> %24
  %28 = icmp sle <4 x i32> %26, %27
  %29 = freeze <4 x i32> %21
  %30 = freeze <4 x i32> %25
  %31 = icmp sle <4 x i32> %29, %30
  %32 = or <4 x i1> %16, %28
  %33 = or <4 x i1> %17, %31
  %34 = add nuw i64 %15, 8
  %35 = icmp eq i64 %34, %13
  br i1 %35, label %36, label %14, !llvm.loop !118

36:                                               ; preds = %14
  %37 = or <4 x i1> %33, %32
  %38 = bitcast <4 x i1> %37 to i4
  %39 = icmp eq i4 %38, 0
  %40 = select i1 %39, i32 -1, i32 3
  %41 = icmp eq i64 %13, %10
  br i1 %41, label %56, label %42

42:                                               ; preds = %9, %36
  %43 = phi i64 [ 0, %9 ], [ %13, %36 ]
  %44 = phi i32 [ -1, %9 ], [ %40, %36 ]
  br label %45

45:                                               ; preds = %42, %45
  %46 = phi i64 [ %54, %45 ], [ %43, %42 ]
  %47 = phi i32 [ %53, %45 ], [ %44, %42 ]
  %48 = getelementptr inbounds nuw i32, ptr %5, i64 %46
  %49 = load i32, ptr %48, align 4, !tbaa !43
  %50 = getelementptr inbounds nuw i32, ptr %6, i64 %46
  %51 = load i32, ptr %50, align 4, !tbaa !43
  %52 = icmp sgt i32 %49, %51
  %53 = select i1 %52, i32 %47, i32 3
  %54 = add nuw nsw i64 %46, 1
  %55 = icmp eq i64 %54, %10
  br i1 %55, label %56, label %45, !llvm.loop !119

56:                                               ; preds = %45, %36, %4
  %57 = phi i32 [ -1, %4 ], [ %40, %36 ], [ %53, %45 ]
  ret i32 %57
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFiPiS0_jEZ4mainE3$_5E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_5", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !100
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef range(i32 -1, 4) i32 @"_ZNSt17_Function_handlerIFiPfS0_jEZ4mainE3$_4E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !55
  %6 = load ptr, ptr %2, align 8, !tbaa !55
  %7 = load i32, ptr %3, align 4, !tbaa !43
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %22, label %9

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 0, %9 ], [ %20, %11 ]
  %13 = phi i32 [ -1, %9 ], [ %19, %11 ]
  %14 = getelementptr inbounds nuw float, ptr %5, i64 %12
  %15 = load float, ptr %14, align 4, !tbaa !57
  %16 = getelementptr inbounds nuw float, ptr %6, i64 %12
  %17 = load float, ptr %16, align 4, !tbaa !57
  %18 = fcmp ogt float %15, %17
  %19 = select i1 %18, i32 %13, i32 3
  %20 = add nuw nsw i64 %12, 1
  %21 = icmp eq i64 %20, %10
  br i1 %21, label %22, label %11, !llvm.loop !120

22:                                               ; preds = %11, %4
  %23 = phi i32 [ -1, %4 ], [ %19, %11 ]
  ret i32 %23
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFiPfS0_jEZ4mainE3$_4E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_4", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !100
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef range(i32 -1, 4) i32 @"_ZNSt17_Function_handlerIFiPfS0_jEZ4mainE3$_5E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !55
  %6 = load ptr, ptr %2, align 8, !tbaa !55
  %7 = load i32, ptr %3, align 4, !tbaa !43
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %56, label %9

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  %11 = icmp ult i32 %7, 8
  br i1 %11, label %42, label %12

12:                                               ; preds = %9
  %13 = and i64 %10, 4294967288
  br label %14

14:                                               ; preds = %14, %12
  %15 = phi i64 [ 0, %12 ], [ %34, %14 ]
  %16 = phi <4 x i1> [ zeroinitializer, %12 ], [ %32, %14 ]
  %17 = phi <4 x i1> [ zeroinitializer, %12 ], [ %33, %14 ]
  %18 = getelementptr inbounds nuw float, ptr %5, i64 %15
  %19 = getelementptr inbounds nuw i8, ptr %18, i64 16
  %20 = load <4 x float>, ptr %18, align 4, !tbaa !57
  %21 = load <4 x float>, ptr %19, align 4, !tbaa !57
  %22 = getelementptr inbounds nuw float, ptr %6, i64 %15
  %23 = getelementptr inbounds nuw i8, ptr %22, i64 16
  %24 = load <4 x float>, ptr %22, align 4, !tbaa !57
  %25 = load <4 x float>, ptr %23, align 4, !tbaa !57
  %26 = freeze <4 x float> %20
  %27 = freeze <4 x float> %24
  %28 = fcmp ule <4 x float> %26, %27
  %29 = freeze <4 x float> %21
  %30 = freeze <4 x float> %25
  %31 = fcmp ule <4 x float> %29, %30
  %32 = or <4 x i1> %16, %28
  %33 = or <4 x i1> %17, %31
  %34 = add nuw i64 %15, 8
  %35 = icmp eq i64 %34, %13
  br i1 %35, label %36, label %14, !llvm.loop !121

36:                                               ; preds = %14
  %37 = or <4 x i1> %33, %32
  %38 = bitcast <4 x i1> %37 to i4
  %39 = icmp eq i4 %38, 0
  %40 = select i1 %39, i32 -1, i32 3
  %41 = icmp eq i64 %13, %10
  br i1 %41, label %56, label %42

42:                                               ; preds = %9, %36
  %43 = phi i64 [ 0, %9 ], [ %13, %36 ]
  %44 = phi i32 [ -1, %9 ], [ %40, %36 ]
  br label %45

45:                                               ; preds = %42, %45
  %46 = phi i64 [ %54, %45 ], [ %43, %42 ]
  %47 = phi i32 [ %53, %45 ], [ %44, %42 ]
  %48 = getelementptr inbounds nuw float, ptr %5, i64 %46
  %49 = load float, ptr %48, align 4, !tbaa !57
  %50 = getelementptr inbounds nuw float, ptr %6, i64 %46
  %51 = load float, ptr %50, align 4, !tbaa !57
  %52 = fcmp ogt float %49, %51
  %53 = select i1 %52, i32 %47, i32 3
  %54 = add nuw nsw i64 %46, 1
  %55 = icmp eq i64 %54, %10
  br i1 %55, label %56, label %45, !llvm.loop !122

56:                                               ; preds = %45, %36, %4
  %57 = phi i32 [ -1, %4 ], [ %40, %36 ], [ %53, %45 ]
  ret i32 %57
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFiPfS0_jEZ4mainE3$_5E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_5", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !100
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef range(i16 -1, 4) i16 @"_ZNSt17_Function_handlerIFsPsS0_jEZ4mainE3$_6E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !71
  %6 = load ptr, ptr %2, align 8, !tbaa !71
  %7 = load i32, ptr %3, align 4, !tbaa !43
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %22, label %9

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 0, %9 ], [ %20, %11 ]
  %13 = phi i16 [ -1, %9 ], [ %19, %11 ]
  %14 = getelementptr inbounds nuw i16, ptr %5, i64 %12
  %15 = load i16, ptr %14, align 2, !tbaa !69
  %16 = getelementptr inbounds nuw i16, ptr %6, i64 %12
  %17 = load i16, ptr %16, align 2, !tbaa !69
  %18 = icmp sgt i16 %15, %17
  %19 = select i1 %18, i16 %13, i16 3
  %20 = add nuw nsw i64 %12, 1
  %21 = icmp eq i64 %20, %10
  br i1 %21, label %22, label %11, !llvm.loop !123

22:                                               ; preds = %11, %4
  %23 = phi i16 [ -1, %4 ], [ %19, %11 ]
  ret i16 %23
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFsPsS0_jEZ4mainE3$_6E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_6", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !100
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef range(i16 -1, 4) i16 @"_ZNSt17_Function_handlerIFsPsS0_jEZ4mainE3$_7E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !71
  %6 = load ptr, ptr %2, align 8, !tbaa !71
  %7 = load i32, ptr %3, align 4, !tbaa !43
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %87, label %9

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  %11 = icmp ult i32 %7, 4
  br i1 %11, label %73, label %12

12:                                               ; preds = %9
  %13 = icmp ult i32 %7, 16
  br i1 %13, label %47, label %14

14:                                               ; preds = %12
  %15 = and i64 %10, 4294967280
  br label %16

16:                                               ; preds = %16, %14
  %17 = phi i64 [ 0, %14 ], [ %36, %16 ]
  %18 = phi <8 x i1> [ zeroinitializer, %14 ], [ %34, %16 ]
  %19 = phi <8 x i1> [ zeroinitializer, %14 ], [ %35, %16 ]
  %20 = getelementptr inbounds nuw i16, ptr %5, i64 %17
  %21 = getelementptr inbounds nuw i8, ptr %20, i64 16
  %22 = load <8 x i16>, ptr %20, align 2, !tbaa !69
  %23 = load <8 x i16>, ptr %21, align 2, !tbaa !69
  %24 = getelementptr inbounds nuw i16, ptr %6, i64 %17
  %25 = getelementptr inbounds nuw i8, ptr %24, i64 16
  %26 = load <8 x i16>, ptr %24, align 2, !tbaa !69
  %27 = load <8 x i16>, ptr %25, align 2, !tbaa !69
  %28 = freeze <8 x i16> %22
  %29 = freeze <8 x i16> %26
  %30 = icmp sle <8 x i16> %28, %29
  %31 = freeze <8 x i16> %23
  %32 = freeze <8 x i16> %27
  %33 = icmp sle <8 x i16> %31, %32
  %34 = or <8 x i1> %18, %30
  %35 = or <8 x i1> %19, %33
  %36 = add nuw i64 %17, 16
  %37 = icmp eq i64 %36, %15
  br i1 %37, label %38, label %16, !llvm.loop !124

38:                                               ; preds = %16
  %39 = or <8 x i1> %35, %34
  %40 = bitcast <8 x i1> %39 to i8
  %41 = icmp eq i8 %40, 0
  %42 = select i1 %41, i16 -1, i16 3
  %43 = icmp eq i64 %15, %10
  br i1 %43, label %87, label %44

44:                                               ; preds = %38
  %45 = and i64 %10, 12
  %46 = icmp eq i64 %45, 0
  br i1 %46, label %73, label %47

47:                                               ; preds = %44, %12
  %48 = phi i64 [ %15, %44 ], [ 0, %12 ]
  %49 = phi i16 [ %42, %44 ], [ -1, %12 ]
  %50 = freeze i16 %49
  %51 = icmp ne i16 %50, -1
  %52 = and i64 %10, 4294967292
  %53 = insertelement <4 x i1> poison, i1 %51, i64 0
  %54 = shufflevector <4 x i1> %53, <4 x i1> poison, <4 x i32> zeroinitializer
  br label %55

55:                                               ; preds = %55, %47
  %56 = phi i64 [ %48, %47 ], [ %66, %55 ]
  %57 = phi <4 x i1> [ %54, %47 ], [ %65, %55 ]
  %58 = getelementptr inbounds nuw i16, ptr %5, i64 %56
  %59 = load <4 x i16>, ptr %58, align 2, !tbaa !69
  %60 = getelementptr inbounds nuw i16, ptr %6, i64 %56
  %61 = load <4 x i16>, ptr %60, align 2, !tbaa !69
  %62 = freeze <4 x i16> %59
  %63 = freeze <4 x i16> %61
  %64 = icmp sle <4 x i16> %62, %63
  %65 = or <4 x i1> %57, %64
  %66 = add nuw i64 %56, 4
  %67 = icmp eq i64 %66, %52
  br i1 %67, label %68, label %55, !llvm.loop !125

68:                                               ; preds = %55
  %69 = bitcast <4 x i1> %65 to i4
  %70 = icmp eq i4 %69, 0
  %71 = select i1 %70, i16 -1, i16 3
  %72 = icmp eq i64 %52, %10
  br i1 %72, label %87, label %73

73:                                               ; preds = %44, %68, %9
  %74 = phi i64 [ 0, %9 ], [ %15, %44 ], [ %52, %68 ]
  %75 = phi i16 [ -1, %9 ], [ %42, %44 ], [ %71, %68 ]
  br label %76

76:                                               ; preds = %73, %76
  %77 = phi i64 [ %85, %76 ], [ %74, %73 ]
  %78 = phi i16 [ %84, %76 ], [ %75, %73 ]
  %79 = getelementptr inbounds nuw i16, ptr %5, i64 %77
  %80 = load i16, ptr %79, align 2, !tbaa !69
  %81 = getelementptr inbounds nuw i16, ptr %6, i64 %77
  %82 = load i16, ptr %81, align 2, !tbaa !69
  %83 = icmp sgt i16 %80, %82
  %84 = select i1 %83, i16 %78, i16 3
  %85 = add nuw nsw i64 %77, 1
  %86 = icmp eq i64 %85, %10
  br i1 %86, label %87, label %76, !llvm.loop !126

87:                                               ; preds = %76, %38, %68, %4
  %88 = phi i16 [ -1, %4 ], [ %42, %38 ], [ %71, %68 ], [ %84, %76 ]
  ret i16 %88
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFsPsS0_jEZ4mainE3$_7E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_7", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !100
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local noundef i32 @_ZNSt24uniform_int_distributionIjEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEjRT_RKNS0_10param_typeE(ptr noundef nonnull align 4 dereferenceable(8) %0, ptr noundef nonnull align 8 dereferenceable(5000) %1, ptr noundef nonnull align 4 dereferenceable(8) %2) local_unnamed_addr #10 comdat {
  %4 = alloca %"struct.std::uniform_int_distribution<unsigned int>::param_type", align 8
  %5 = getelementptr inbounds nuw i8, ptr %2, i64 4
  %6 = load i32, ptr %5, align 4, !tbaa !127
  %7 = zext i32 %6 to i64
  %8 = load i32, ptr %2, align 4, !tbaa !129
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
  br i1 %28, label %24, label %29, !llvm.loop !130

29:                                               ; preds = %24, %12, %20
  %30 = phi i64 [ %17, %12 ], [ %17, %20 ], [ %26, %24 ]
  %31 = lshr i64 %30, 32
  br label %45

32:                                               ; preds = %3
  %33 = icmp eq i64 %10, 4294967295
  br i1 %33, label %43, label %34

34:                                               ; preds = %32, %34
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #21
  store <2 x i32> <i32 0, i32 -1>, ptr %4, align 8, !tbaa !43
  %35 = call noundef i32 @_ZNSt24uniform_int_distributionIjEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEjRT_RKNS0_10param_typeE(ptr noundef nonnull align 4 dereferenceable(8) %0, ptr noundef nonnull align 8 dereferenceable(5000) %1, ptr noundef nonnull align 4 dereferenceable(8) %4)
  %36 = zext i32 %35 to i64
  %37 = shl nuw i64 %36, 32
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #21
  %38 = call noundef i64 @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv(ptr noundef nonnull align 8 dereferenceable(5000) %1)
  %39 = add i64 %37, %38
  %40 = icmp ugt i64 %39, %10
  %41 = icmp ult i64 %39, %37
  %42 = or i1 %40, %41
  br i1 %42, label %34, label %45, !llvm.loop !131

43:                                               ; preds = %32
  %44 = tail call noundef i64 @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv(ptr noundef nonnull align 8 dereferenceable(5000) %1)
  br label %45

45:                                               ; preds = %34, %43, %29
  %46 = phi i64 [ %31, %29 ], [ %44, %43 ], [ %39, %34 ]
  %47 = load i32, ptr %2, align 4, !tbaa !129
  %48 = trunc i64 %46 to i32
  %49 = add i32 %47, %48
  ret i32 %49
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i32 @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE3$_8E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !45
  %6 = load ptr, ptr %2, align 8, !tbaa !45
  %7 = load i32, ptr %3, align 4, !tbaa !43
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %22, label %9

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 0, %9 ], [ %20, %11 ]
  %13 = phi i32 [ %7, %9 ], [ %19, %11 ]
  %14 = getelementptr inbounds nuw i32, ptr %5, i64 %12
  %15 = load i32, ptr %14, align 4, !tbaa !43
  %16 = getelementptr inbounds nuw i32, ptr %6, i64 %12
  %17 = load i32, ptr %16, align 4, !tbaa !43
  %18 = icmp ugt i32 %15, %17
  %19 = select i1 %18, i32 3, i32 %13
  %20 = add nuw nsw i64 %12, 1
  %21 = icmp eq i64 %20, %10
  br i1 %21, label %22, label %11, !llvm.loop !132

22:                                               ; preds = %11, %4
  %23 = phi i32 [ 0, %4 ], [ %19, %11 ]
  ret i32 %23
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
  store ptr %6, ptr %0, align 8, !tbaa !100
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i32 @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE3$_9E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !45
  %6 = load ptr, ptr %2, align 8, !tbaa !45
  %7 = load i32, ptr %3, align 4, !tbaa !43
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %56, label %9

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  %11 = icmp ult i32 %7, 8
  br i1 %11, label %42, label %12

12:                                               ; preds = %9
  %13 = and i64 %10, 4294967288
  br label %14

14:                                               ; preds = %14, %12
  %15 = phi i64 [ 0, %12 ], [ %34, %14 ]
  %16 = phi <4 x i1> [ zeroinitializer, %12 ], [ %32, %14 ]
  %17 = phi <4 x i1> [ zeroinitializer, %12 ], [ %33, %14 ]
  %18 = getelementptr inbounds nuw i32, ptr %5, i64 %15
  %19 = getelementptr inbounds nuw i8, ptr %18, i64 16
  %20 = load <4 x i32>, ptr %18, align 4, !tbaa !43
  %21 = load <4 x i32>, ptr %19, align 4, !tbaa !43
  %22 = getelementptr inbounds nuw i32, ptr %6, i64 %15
  %23 = getelementptr inbounds nuw i8, ptr %22, i64 16
  %24 = load <4 x i32>, ptr %22, align 4, !tbaa !43
  %25 = load <4 x i32>, ptr %23, align 4, !tbaa !43
  %26 = freeze <4 x i32> %20
  %27 = freeze <4 x i32> %24
  %28 = icmp ugt <4 x i32> %26, %27
  %29 = freeze <4 x i32> %21
  %30 = freeze <4 x i32> %25
  %31 = icmp ugt <4 x i32> %29, %30
  %32 = or <4 x i1> %16, %28
  %33 = or <4 x i1> %17, %31
  %34 = add nuw i64 %15, 8
  %35 = icmp eq i64 %34, %13
  br i1 %35, label %36, label %14, !llvm.loop !133

36:                                               ; preds = %14
  %37 = or <4 x i1> %33, %32
  %38 = bitcast <4 x i1> %37 to i4
  %39 = icmp eq i4 %38, 0
  %40 = select i1 %39, i32 %7, i32 3
  %41 = icmp eq i64 %13, %10
  br i1 %41, label %56, label %42

42:                                               ; preds = %9, %36
  %43 = phi i64 [ 0, %9 ], [ %13, %36 ]
  %44 = phi i32 [ %7, %9 ], [ %40, %36 ]
  br label %45

45:                                               ; preds = %42, %45
  %46 = phi i64 [ %54, %45 ], [ %43, %42 ]
  %47 = phi i32 [ %53, %45 ], [ %44, %42 ]
  %48 = getelementptr inbounds nuw i32, ptr %5, i64 %46
  %49 = load i32, ptr %48, align 4, !tbaa !43
  %50 = getelementptr inbounds nuw i32, ptr %6, i64 %46
  %51 = load i32, ptr %50, align 4, !tbaa !43
  %52 = icmp ugt i32 %49, %51
  %53 = select i1 %52, i32 3, i32 %47
  %54 = add nuw nsw i64 %46, 1
  %55 = icmp eq i64 %54, %10
  br i1 %55, label %56, label %45, !llvm.loop !134

56:                                               ; preds = %45, %36, %4
  %57 = phi i32 [ 0, %4 ], [ %40, %36 ], [ %53, %45 ]
  ret i32 %57
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
  store ptr %6, ptr %0, align 8, !tbaa !100
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i32 @"_ZNSt17_Function_handlerIFjPfS0_jEZ4mainE3$_8E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !55
  %6 = load ptr, ptr %2, align 8, !tbaa !55
  %7 = load i32, ptr %3, align 4, !tbaa !43
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %22, label %9

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 0, %9 ], [ %20, %11 ]
  %13 = phi i32 [ %7, %9 ], [ %19, %11 ]
  %14 = getelementptr inbounds nuw float, ptr %5, i64 %12
  %15 = load float, ptr %14, align 4, !tbaa !57
  %16 = getelementptr inbounds nuw float, ptr %6, i64 %12
  %17 = load float, ptr %16, align 4, !tbaa !57
  %18 = fcmp ogt float %15, %17
  %19 = select i1 %18, i32 3, i32 %13
  %20 = add nuw nsw i64 %12, 1
  %21 = icmp eq i64 %20, %10
  br i1 %21, label %22, label %11, !llvm.loop !135

22:                                               ; preds = %11, %4
  %23 = phi i32 [ 0, %4 ], [ %19, %11 ]
  ret i32 %23
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFjPfS0_jEZ4mainE3$_8E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_8", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !100
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i32 @"_ZNSt17_Function_handlerIFjPfS0_jEZ4mainE3$_9E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !55
  %6 = load ptr, ptr %2, align 8, !tbaa !55
  %7 = load i32, ptr %3, align 4, !tbaa !43
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %56, label %9

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  %11 = icmp ult i32 %7, 8
  br i1 %11, label %42, label %12

12:                                               ; preds = %9
  %13 = and i64 %10, 4294967288
  br label %14

14:                                               ; preds = %14, %12
  %15 = phi i64 [ 0, %12 ], [ %34, %14 ]
  %16 = phi <4 x i1> [ zeroinitializer, %12 ], [ %32, %14 ]
  %17 = phi <4 x i1> [ zeroinitializer, %12 ], [ %33, %14 ]
  %18 = getelementptr inbounds nuw float, ptr %5, i64 %15
  %19 = getelementptr inbounds nuw i8, ptr %18, i64 16
  %20 = load <4 x float>, ptr %18, align 4, !tbaa !57
  %21 = load <4 x float>, ptr %19, align 4, !tbaa !57
  %22 = getelementptr inbounds nuw float, ptr %6, i64 %15
  %23 = getelementptr inbounds nuw i8, ptr %22, i64 16
  %24 = load <4 x float>, ptr %22, align 4, !tbaa !57
  %25 = load <4 x float>, ptr %23, align 4, !tbaa !57
  %26 = freeze <4 x float> %20
  %27 = freeze <4 x float> %24
  %28 = fcmp ogt <4 x float> %26, %27
  %29 = freeze <4 x float> %21
  %30 = freeze <4 x float> %25
  %31 = fcmp ogt <4 x float> %29, %30
  %32 = or <4 x i1> %16, %28
  %33 = or <4 x i1> %17, %31
  %34 = add nuw i64 %15, 8
  %35 = icmp eq i64 %34, %13
  br i1 %35, label %36, label %14, !llvm.loop !136

36:                                               ; preds = %14
  %37 = or <4 x i1> %33, %32
  %38 = bitcast <4 x i1> %37 to i4
  %39 = icmp eq i4 %38, 0
  %40 = select i1 %39, i32 %7, i32 3
  %41 = icmp eq i64 %13, %10
  br i1 %41, label %56, label %42

42:                                               ; preds = %9, %36
  %43 = phi i64 [ 0, %9 ], [ %13, %36 ]
  %44 = phi i32 [ %7, %9 ], [ %40, %36 ]
  br label %45

45:                                               ; preds = %42, %45
  %46 = phi i64 [ %54, %45 ], [ %43, %42 ]
  %47 = phi i32 [ %53, %45 ], [ %44, %42 ]
  %48 = getelementptr inbounds nuw float, ptr %5, i64 %46
  %49 = load float, ptr %48, align 4, !tbaa !57
  %50 = getelementptr inbounds nuw float, ptr %6, i64 %46
  %51 = load float, ptr %50, align 4, !tbaa !57
  %52 = fcmp ogt float %49, %51
  %53 = select i1 %52, i32 3, i32 %47
  %54 = add nuw nsw i64 %46, 1
  %55 = icmp eq i64 %54, %10
  br i1 %55, label %56, label %45, !llvm.loop !137

56:                                               ; preds = %45, %36, %4
  %57 = phi i32 [ 0, %4 ], [ %40, %36 ], [ %53, %45 ]
  ret i32 %57
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFjPfS0_jEZ4mainE3$_9E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_9", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !100
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local noundef i16 @_ZNSt24uniform_int_distributionItEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEtRT_RKNS0_10param_typeE(ptr noundef nonnull align 2 dereferenceable(4) %0, ptr noundef nonnull align 8 dereferenceable(5000) %1, ptr noundef nonnull align 2 dereferenceable(4) %2) local_unnamed_addr #10 comdat {
  %4 = alloca %"struct.std::uniform_int_distribution<unsigned short>::param_type", align 4
  %5 = getelementptr inbounds nuw i8, ptr %2, i64 2
  %6 = load i16, ptr %5, align 2, !tbaa !88
  %7 = zext i16 %6 to i64
  %8 = load i16, ptr %2, align 2, !tbaa !86
  %9 = zext i16 %8 to i64
  %10 = sub nsw i64 %7, %9
  %11 = icmp ult i64 %10, 4294967295
  br i1 %11, label %14, label %12

12:                                               ; preds = %3
  %13 = getelementptr inbounds nuw i8, ptr %4, i64 2
  br label %34

14:                                               ; preds = %3
  %15 = trunc nuw nsw i64 %10 to i32
  %16 = add nuw nsw i32 %15, 1
  %17 = tail call noundef i64 @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv(ptr noundef nonnull align 8 dereferenceable(5000) %1)
  %18 = zext nneg i32 %16 to i64
  %19 = mul i64 %17, %18
  %20 = trunc i64 %19 to i32
  %21 = icmp ult i32 %15, %20
  br i1 %21, label %31, label %22

22:                                               ; preds = %14
  %23 = xor i32 %15, -1
  %24 = urem i32 %23, %16
  %25 = icmp samesign ugt i32 %24, %20
  br i1 %25, label %26, label %31

26:                                               ; preds = %22, %26
  %27 = tail call noundef i64 @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv(ptr noundef nonnull align 8 dereferenceable(5000) %1)
  %28 = mul i64 %27, %18
  %29 = trunc i64 %28 to i32
  %30 = icmp ugt i32 %24, %29
  br i1 %30, label %26, label %31, !llvm.loop !138

31:                                               ; preds = %26, %14, %22
  %32 = phi i64 [ %19, %14 ], [ %19, %22 ], [ %28, %26 ]
  %33 = lshr i64 %32, 32
  br label %43

34:                                               ; preds = %12, %34
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #21
  store i16 0, ptr %4, align 4, !tbaa !86
  store i16 -1, ptr %13, align 2, !tbaa !88
  %35 = call noundef i16 @_ZNSt24uniform_int_distributionItEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEtRT_RKNS0_10param_typeE(ptr noundef nonnull align 2 dereferenceable(4) %0, ptr noundef nonnull align 8 dereferenceable(5000) %1, ptr noundef nonnull align 2 dereferenceable(4) %4)
  %36 = zext i16 %35 to i64
  %37 = shl nuw nsw i64 %36, 32
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #21
  %38 = call noundef i64 @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv(ptr noundef nonnull align 8 dereferenceable(5000) %1)
  %39 = add i64 %37, %38
  %40 = icmp ugt i64 %39, %10
  %41 = icmp ult i64 %39, %37
  %42 = or i1 %40, %41
  br i1 %42, label %34, label %43, !llvm.loop !139

43:                                               ; preds = %34, %31
  %44 = phi i64 [ %33, %31 ], [ %39, %34 ]
  %45 = load i16, ptr %2, align 2, !tbaa !86
  %46 = trunc i64 %44 to i16
  %47 = add i16 %45, %46
  ret i16 %47
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i16 @"_ZNSt17_Function_handlerIFtPtS0_jEZ4mainE4$_10E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !71
  %6 = load ptr, ptr %2, align 8, !tbaa !71
  %7 = load i32, ptr %3, align 4, !tbaa !43
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %24, label %9

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  br label %13

11:                                               ; preds = %13
  %12 = trunc i32 %21 to i16
  br label %24

13:                                               ; preds = %13, %9
  %14 = phi i64 [ 0, %9 ], [ %22, %13 ]
  %15 = phi i32 [ %7, %9 ], [ %21, %13 ]
  %16 = getelementptr inbounds nuw i16, ptr %5, i64 %14
  %17 = load i16, ptr %16, align 2, !tbaa !69
  %18 = getelementptr inbounds nuw i16, ptr %6, i64 %14
  %19 = load i16, ptr %18, align 2, !tbaa !69
  %20 = icmp ugt i16 %17, %19
  %21 = select i1 %20, i32 3, i32 %15
  %22 = add nuw nsw i64 %14, 1
  %23 = icmp eq i64 %22, %10
  br i1 %23, label %11, label %13, !llvm.loop !140

24:                                               ; preds = %4, %11
  %25 = phi i16 [ 0, %4 ], [ %12, %11 ]
  ret i16 %25
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFtPtS0_jEZ4mainE4$_10E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_10", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !100
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i16 @"_ZNSt17_Function_handlerIFtPtS0_jEZ4mainE4$_11E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !71
  %6 = load ptr, ptr %2, align 8, !tbaa !71
  %7 = load i32, ptr %3, align 4, !tbaa !43
  %8 = freeze i32 %7
  %9 = icmp eq i32 %8, 0
  br i1 %9, label %91, label %10

10:                                               ; preds = %4
  %11 = zext i32 %8 to i64
  %12 = icmp ult i32 %8, 4
  br i1 %12, label %13, label %16

13:                                               ; preds = %48, %72, %10
  %14 = phi i64 [ 0, %10 ], [ %19, %48 ], [ %56, %72 ]
  %15 = phi i32 [ %8, %10 ], [ %46, %48 ], [ %75, %72 ]
  br label %80

16:                                               ; preds = %10
  %17 = icmp ult i32 %8, 16
  br i1 %17, label %51, label %18

18:                                               ; preds = %16
  %19 = and i64 %11, 4294967280
  br label %20

20:                                               ; preds = %20, %18
  %21 = phi i64 [ 0, %18 ], [ %40, %20 ]
  %22 = phi <8 x i1> [ zeroinitializer, %18 ], [ %38, %20 ]
  %23 = phi <8 x i1> [ zeroinitializer, %18 ], [ %39, %20 ]
  %24 = getelementptr inbounds nuw i16, ptr %5, i64 %21
  %25 = getelementptr inbounds nuw i8, ptr %24, i64 16
  %26 = load <8 x i16>, ptr %24, align 2, !tbaa !69
  %27 = load <8 x i16>, ptr %25, align 2, !tbaa !69
  %28 = getelementptr inbounds nuw i16, ptr %6, i64 %21
  %29 = getelementptr inbounds nuw i8, ptr %28, i64 16
  %30 = load <8 x i16>, ptr %28, align 2, !tbaa !69
  %31 = load <8 x i16>, ptr %29, align 2, !tbaa !69
  %32 = freeze <8 x i16> %26
  %33 = freeze <8 x i16> %30
  %34 = icmp ugt <8 x i16> %32, %33
  %35 = freeze <8 x i16> %27
  %36 = freeze <8 x i16> %31
  %37 = icmp ugt <8 x i16> %35, %36
  %38 = or <8 x i1> %22, %34
  %39 = or <8 x i1> %23, %37
  %40 = add nuw i64 %21, 16
  %41 = icmp eq i64 %40, %19
  br i1 %41, label %42, label %20, !llvm.loop !141

42:                                               ; preds = %20
  %43 = or <8 x i1> %39, %38
  %44 = bitcast <8 x i1> %43 to i8
  %45 = icmp eq i8 %44, 0
  %46 = select i1 %45, i32 %8, i32 3
  %47 = icmp eq i64 %19, %11
  br i1 %47, label %77, label %48

48:                                               ; preds = %42
  %49 = and i64 %11, 12
  %50 = icmp eq i64 %49, 0
  br i1 %50, label %13, label %51

51:                                               ; preds = %48, %16
  %52 = phi i64 [ %19, %48 ], [ 0, %16 ]
  %53 = phi i32 [ %46, %48 ], [ %8, %16 ]
  %54 = freeze i32 %53
  %55 = icmp ne i32 %54, %8
  %56 = and i64 %11, 4294967292
  %57 = insertelement <4 x i1> poison, i1 %55, i64 0
  %58 = shufflevector <4 x i1> %57, <4 x i1> poison, <4 x i32> zeroinitializer
  br label %59

59:                                               ; preds = %59, %51
  %60 = phi i64 [ %52, %51 ], [ %70, %59 ]
  %61 = phi <4 x i1> [ %58, %51 ], [ %69, %59 ]
  %62 = getelementptr inbounds nuw i16, ptr %5, i64 %60
  %63 = load <4 x i16>, ptr %62, align 2, !tbaa !69
  %64 = getelementptr inbounds nuw i16, ptr %6, i64 %60
  %65 = load <4 x i16>, ptr %64, align 2, !tbaa !69
  %66 = freeze <4 x i16> %63
  %67 = freeze <4 x i16> %65
  %68 = icmp ugt <4 x i16> %66, %67
  %69 = or <4 x i1> %61, %68
  %70 = add nuw i64 %60, 4
  %71 = icmp eq i64 %70, %56
  br i1 %71, label %72, label %59, !llvm.loop !142

72:                                               ; preds = %59
  %73 = bitcast <4 x i1> %69 to i4
  %74 = icmp eq i4 %73, 0
  %75 = select i1 %74, i32 %8, i32 3
  %76 = icmp eq i64 %56, %11
  br i1 %76, label %77, label %13

77:                                               ; preds = %80, %72, %42
  %78 = phi i32 [ %46, %42 ], [ %75, %72 ], [ %88, %80 ]
  %79 = trunc i32 %78 to i16
  br label %91

80:                                               ; preds = %13, %80
  %81 = phi i64 [ %89, %80 ], [ %14, %13 ]
  %82 = phi i32 [ %88, %80 ], [ %15, %13 ]
  %83 = getelementptr inbounds nuw i16, ptr %5, i64 %81
  %84 = load i16, ptr %83, align 2, !tbaa !69
  %85 = getelementptr inbounds nuw i16, ptr %6, i64 %81
  %86 = load i16, ptr %85, align 2, !tbaa !69
  %87 = icmp ugt i16 %84, %86
  %88 = select i1 %87, i32 3, i32 %82
  %89 = add nuw nsw i64 %81, 1
  %90 = icmp eq i64 %89, %11
  br i1 %90, label %77, label %80, !llvm.loop !143

91:                                               ; preds = %4, %77
  %92 = phi i16 [ 0, %4 ], [ %79, %77 ]
  ret i16 %92
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFtPtS0_jEZ4mainE4$_11E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_11", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !100
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef range(i32 1, 0) i32 @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE4$_12E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !45
  %6 = load ptr, ptr %2, align 8, !tbaa !45
  %7 = load i32, ptr %3, align 4, !tbaa !43
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %22, label %9

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 0, %9 ], [ %20, %11 ]
  %13 = phi i32 [ 3, %9 ], [ %19, %11 ]
  %14 = getelementptr inbounds nuw i32, ptr %5, i64 %12
  %15 = load i32, ptr %14, align 4, !tbaa !43
  %16 = getelementptr inbounds nuw i32, ptr %6, i64 %12
  %17 = load i32, ptr %16, align 4, !tbaa !43
  %18 = icmp ugt i32 %15, %17
  %19 = select i1 %18, i32 %7, i32 %13
  %20 = add nuw nsw i64 %12, 1
  %21 = icmp eq i64 %20, %10
  br i1 %21, label %22, label %11, !llvm.loop !144

22:                                               ; preds = %11, %4
  %23 = phi i32 [ 3, %4 ], [ %19, %11 ]
  ret i32 %23
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
  store ptr %6, ptr %0, align 8, !tbaa !100
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef range(i32 1, 0) i32 @"_ZNSt17_Function_handlerIFjPjS0_jEZ4mainE4$_13E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !45
  %6 = load ptr, ptr %2, align 8, !tbaa !45
  %7 = load i32, ptr %3, align 4, !tbaa !43
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %56, label %9

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  %11 = icmp ult i32 %7, 8
  br i1 %11, label %42, label %12

12:                                               ; preds = %9
  %13 = and i64 %10, 4294967288
  br label %14

14:                                               ; preds = %14, %12
  %15 = phi i64 [ 0, %12 ], [ %34, %14 ]
  %16 = phi <4 x i1> [ zeroinitializer, %12 ], [ %32, %14 ]
  %17 = phi <4 x i1> [ zeroinitializer, %12 ], [ %33, %14 ]
  %18 = getelementptr inbounds nuw i32, ptr %5, i64 %15
  %19 = getelementptr inbounds nuw i8, ptr %18, i64 16
  %20 = load <4 x i32>, ptr %18, align 4, !tbaa !43
  %21 = load <4 x i32>, ptr %19, align 4, !tbaa !43
  %22 = getelementptr inbounds nuw i32, ptr %6, i64 %15
  %23 = getelementptr inbounds nuw i8, ptr %22, i64 16
  %24 = load <4 x i32>, ptr %22, align 4, !tbaa !43
  %25 = load <4 x i32>, ptr %23, align 4, !tbaa !43
  %26 = freeze <4 x i32> %20
  %27 = freeze <4 x i32> %24
  %28 = icmp ugt <4 x i32> %26, %27
  %29 = freeze <4 x i32> %21
  %30 = freeze <4 x i32> %25
  %31 = icmp ugt <4 x i32> %29, %30
  %32 = or <4 x i1> %16, %28
  %33 = or <4 x i1> %17, %31
  %34 = add nuw i64 %15, 8
  %35 = icmp eq i64 %34, %13
  br i1 %35, label %36, label %14, !llvm.loop !145

36:                                               ; preds = %14
  %37 = or <4 x i1> %33, %32
  %38 = bitcast <4 x i1> %37 to i4
  %39 = icmp eq i4 %38, 0
  %40 = select i1 %39, i32 3, i32 %7
  %41 = icmp eq i64 %13, %10
  br i1 %41, label %56, label %42

42:                                               ; preds = %9, %36
  %43 = phi i64 [ 0, %9 ], [ %13, %36 ]
  %44 = phi i32 [ 3, %9 ], [ %40, %36 ]
  br label %45

45:                                               ; preds = %42, %45
  %46 = phi i64 [ %54, %45 ], [ %43, %42 ]
  %47 = phi i32 [ %53, %45 ], [ %44, %42 ]
  %48 = getelementptr inbounds nuw i32, ptr %5, i64 %46
  %49 = load i32, ptr %48, align 4, !tbaa !43
  %50 = getelementptr inbounds nuw i32, ptr %6, i64 %46
  %51 = load i32, ptr %50, align 4, !tbaa !43
  %52 = icmp ugt i32 %49, %51
  %53 = select i1 %52, i32 %7, i32 %47
  %54 = add nuw nsw i64 %46, 1
  %55 = icmp eq i64 %54, %10
  br i1 %55, label %56, label %45, !llvm.loop !146

56:                                               ; preds = %45, %36, %4
  %57 = phi i32 [ 3, %4 ], [ %40, %36 ], [ %53, %45 ]
  ret i32 %57
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
  store ptr %6, ptr %0, align 8, !tbaa !100
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef range(i32 1, 0) i32 @"_ZNSt17_Function_handlerIFjPfS0_jEZ4mainE4$_12E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !55
  %6 = load ptr, ptr %2, align 8, !tbaa !55
  %7 = load i32, ptr %3, align 4, !tbaa !43
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %22, label %9

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 0, %9 ], [ %20, %11 ]
  %13 = phi i32 [ 3, %9 ], [ %19, %11 ]
  %14 = getelementptr inbounds nuw float, ptr %5, i64 %12
  %15 = load float, ptr %14, align 4, !tbaa !57
  %16 = getelementptr inbounds nuw float, ptr %6, i64 %12
  %17 = load float, ptr %16, align 4, !tbaa !57
  %18 = fcmp ogt float %15, %17
  %19 = select i1 %18, i32 %7, i32 %13
  %20 = add nuw nsw i64 %12, 1
  %21 = icmp eq i64 %20, %10
  br i1 %21, label %22, label %11, !llvm.loop !147

22:                                               ; preds = %11, %4
  %23 = phi i32 [ 3, %4 ], [ %19, %11 ]
  ret i32 %23
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFjPfS0_jEZ4mainE4$_12E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_12", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !100
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef range(i32 1, 0) i32 @"_ZNSt17_Function_handlerIFjPfS0_jEZ4mainE4$_13E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !55
  %6 = load ptr, ptr %2, align 8, !tbaa !55
  %7 = load i32, ptr %3, align 4, !tbaa !43
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %56, label %9

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  %11 = icmp ult i32 %7, 8
  br i1 %11, label %42, label %12

12:                                               ; preds = %9
  %13 = and i64 %10, 4294967288
  br label %14

14:                                               ; preds = %14, %12
  %15 = phi i64 [ 0, %12 ], [ %34, %14 ]
  %16 = phi <4 x i1> [ zeroinitializer, %12 ], [ %32, %14 ]
  %17 = phi <4 x i1> [ zeroinitializer, %12 ], [ %33, %14 ]
  %18 = getelementptr inbounds nuw float, ptr %5, i64 %15
  %19 = getelementptr inbounds nuw i8, ptr %18, i64 16
  %20 = load <4 x float>, ptr %18, align 4, !tbaa !57
  %21 = load <4 x float>, ptr %19, align 4, !tbaa !57
  %22 = getelementptr inbounds nuw float, ptr %6, i64 %15
  %23 = getelementptr inbounds nuw i8, ptr %22, i64 16
  %24 = load <4 x float>, ptr %22, align 4, !tbaa !57
  %25 = load <4 x float>, ptr %23, align 4, !tbaa !57
  %26 = freeze <4 x float> %20
  %27 = freeze <4 x float> %24
  %28 = fcmp ogt <4 x float> %26, %27
  %29 = freeze <4 x float> %21
  %30 = freeze <4 x float> %25
  %31 = fcmp ogt <4 x float> %29, %30
  %32 = or <4 x i1> %16, %28
  %33 = or <4 x i1> %17, %31
  %34 = add nuw i64 %15, 8
  %35 = icmp eq i64 %34, %13
  br i1 %35, label %36, label %14, !llvm.loop !148

36:                                               ; preds = %14
  %37 = or <4 x i1> %33, %32
  %38 = bitcast <4 x i1> %37 to i4
  %39 = icmp eq i4 %38, 0
  %40 = select i1 %39, i32 3, i32 %7
  %41 = icmp eq i64 %13, %10
  br i1 %41, label %56, label %42

42:                                               ; preds = %9, %36
  %43 = phi i64 [ 0, %9 ], [ %13, %36 ]
  %44 = phi i32 [ 3, %9 ], [ %40, %36 ]
  br label %45

45:                                               ; preds = %42, %45
  %46 = phi i64 [ %54, %45 ], [ %43, %42 ]
  %47 = phi i32 [ %53, %45 ], [ %44, %42 ]
  %48 = getelementptr inbounds nuw float, ptr %5, i64 %46
  %49 = load float, ptr %48, align 4, !tbaa !57
  %50 = getelementptr inbounds nuw float, ptr %6, i64 %46
  %51 = load float, ptr %50, align 4, !tbaa !57
  %52 = fcmp ogt float %49, %51
  %53 = select i1 %52, i32 %7, i32 %47
  %54 = add nuw nsw i64 %46, 1
  %55 = icmp eq i64 %54, %10
  br i1 %55, label %56, label %45, !llvm.loop !149

56:                                               ; preds = %45, %36, %4
  %57 = phi i32 [ 3, %4 ], [ %40, %36 ], [ %53, %45 ]
  ret i32 %57
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFjPfS0_jEZ4mainE4$_13E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_13", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !100
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i16 @"_ZNSt17_Function_handlerIFtPtS0_jEZ4mainE4$_14E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !71
  %6 = load ptr, ptr %2, align 8, !tbaa !71
  %7 = load i32, ptr %3, align 4, !tbaa !43
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %23, label %9

9:                                                ; preds = %4
  %10 = trunc i32 %7 to i16
  %11 = zext i32 %7 to i64
  br label %12

12:                                               ; preds = %12, %9
  %13 = phi i64 [ 0, %9 ], [ %21, %12 ]
  %14 = phi i16 [ 3, %9 ], [ %20, %12 ]
  %15 = getelementptr inbounds nuw i16, ptr %5, i64 %13
  %16 = load i16, ptr %15, align 2, !tbaa !69
  %17 = getelementptr inbounds nuw i16, ptr %6, i64 %13
  %18 = load i16, ptr %17, align 2, !tbaa !69
  %19 = icmp ugt i16 %16, %18
  %20 = select i1 %19, i16 %10, i16 %14
  %21 = add nuw nsw i64 %13, 1
  %22 = icmp eq i64 %21, %11
  br i1 %22, label %23, label %12, !llvm.loop !150

23:                                               ; preds = %12, %4
  %24 = phi i16 [ 3, %4 ], [ %20, %12 ]
  ret i16 %24
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFtPtS0_jEZ4mainE4$_14E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_14", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !100
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i16 @"_ZNSt17_Function_handlerIFtPtS0_jEZ4mainE4$_15E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !71
  %6 = load ptr, ptr %2, align 8, !tbaa !71
  %7 = load i32, ptr %3, align 4, !tbaa !43
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %88, label %9

9:                                                ; preds = %4
  %10 = trunc i32 %7 to i16
  %11 = zext i32 %7 to i64
  %12 = icmp ult i32 %7, 4
  br i1 %12, label %74, label %13

13:                                               ; preds = %9
  %14 = icmp ult i32 %7, 16
  br i1 %14, label %48, label %15

15:                                               ; preds = %13
  %16 = and i64 %11, 4294967280
  br label %17

17:                                               ; preds = %17, %15
  %18 = phi i64 [ 0, %15 ], [ %37, %17 ]
  %19 = phi <8 x i1> [ zeroinitializer, %15 ], [ %35, %17 ]
  %20 = phi <8 x i1> [ zeroinitializer, %15 ], [ %36, %17 ]
  %21 = getelementptr inbounds nuw i16, ptr %5, i64 %18
  %22 = getelementptr inbounds nuw i8, ptr %21, i64 16
  %23 = load <8 x i16>, ptr %21, align 2, !tbaa !69
  %24 = load <8 x i16>, ptr %22, align 2, !tbaa !69
  %25 = getelementptr inbounds nuw i16, ptr %6, i64 %18
  %26 = getelementptr inbounds nuw i8, ptr %25, i64 16
  %27 = load <8 x i16>, ptr %25, align 2, !tbaa !69
  %28 = load <8 x i16>, ptr %26, align 2, !tbaa !69
  %29 = freeze <8 x i16> %23
  %30 = freeze <8 x i16> %27
  %31 = icmp ugt <8 x i16> %29, %30
  %32 = freeze <8 x i16> %24
  %33 = freeze <8 x i16> %28
  %34 = icmp ugt <8 x i16> %32, %33
  %35 = or <8 x i1> %19, %31
  %36 = or <8 x i1> %20, %34
  %37 = add nuw i64 %18, 16
  %38 = icmp eq i64 %37, %16
  br i1 %38, label %39, label %17, !llvm.loop !151

39:                                               ; preds = %17
  %40 = or <8 x i1> %36, %35
  %41 = bitcast <8 x i1> %40 to i8
  %42 = icmp eq i8 %41, 0
  %43 = select i1 %42, i16 3, i16 %10
  %44 = icmp eq i64 %16, %11
  br i1 %44, label %88, label %45

45:                                               ; preds = %39
  %46 = and i64 %11, 12
  %47 = icmp eq i64 %46, 0
  br i1 %47, label %74, label %48

48:                                               ; preds = %45, %13
  %49 = phi i64 [ %16, %45 ], [ 0, %13 ]
  %50 = phi i16 [ %43, %45 ], [ 3, %13 ]
  %51 = freeze i16 %50
  %52 = icmp ne i16 %51, 3
  %53 = and i64 %11, 4294967292
  %54 = insertelement <4 x i1> poison, i1 %52, i64 0
  %55 = shufflevector <4 x i1> %54, <4 x i1> poison, <4 x i32> zeroinitializer
  br label %56

56:                                               ; preds = %56, %48
  %57 = phi i64 [ %49, %48 ], [ %67, %56 ]
  %58 = phi <4 x i1> [ %55, %48 ], [ %66, %56 ]
  %59 = getelementptr inbounds nuw i16, ptr %5, i64 %57
  %60 = load <4 x i16>, ptr %59, align 2, !tbaa !69
  %61 = getelementptr inbounds nuw i16, ptr %6, i64 %57
  %62 = load <4 x i16>, ptr %61, align 2, !tbaa !69
  %63 = freeze <4 x i16> %60
  %64 = freeze <4 x i16> %62
  %65 = icmp ugt <4 x i16> %63, %64
  %66 = or <4 x i1> %58, %65
  %67 = add nuw i64 %57, 4
  %68 = icmp eq i64 %67, %53
  br i1 %68, label %69, label %56, !llvm.loop !152

69:                                               ; preds = %56
  %70 = bitcast <4 x i1> %66 to i4
  %71 = icmp eq i4 %70, 0
  %72 = select i1 %71, i16 3, i16 %10
  %73 = icmp eq i64 %53, %11
  br i1 %73, label %88, label %74

74:                                               ; preds = %45, %69, %9
  %75 = phi i64 [ 0, %9 ], [ %16, %45 ], [ %53, %69 ]
  %76 = phi i16 [ 3, %9 ], [ %43, %45 ], [ %72, %69 ]
  br label %77

77:                                               ; preds = %74, %77
  %78 = phi i64 [ %86, %77 ], [ %75, %74 ]
  %79 = phi i16 [ %85, %77 ], [ %76, %74 ]
  %80 = getelementptr inbounds nuw i16, ptr %5, i64 %78
  %81 = load i16, ptr %80, align 2, !tbaa !69
  %82 = getelementptr inbounds nuw i16, ptr %6, i64 %78
  %83 = load i16, ptr %82, align 2, !tbaa !69
  %84 = icmp ugt i16 %81, %83
  %85 = select i1 %84, i16 %10, i16 %79
  %86 = add nuw nsw i64 %78, 1
  %87 = icmp eq i64 %86, %11
  br i1 %87, label %88, label %77, !llvm.loop !153

88:                                               ; preds = %77, %39, %69, %4
  %89 = phi i16 [ 3, %4 ], [ %43, %39 ], [ %72, %69 ], [ %85, %77 ]
  ret i16 %89
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFtPtS0_jEZ4mainE4$_15E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_15", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !100
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define internal void @_GLOBAL__sub_I_any_of.cpp() #19 section ".text.startup" {
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
declare fp128 @llvm.log.f128(fp128) #20

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.umax.i64(i64, i64) #20

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
attributes #16 = { mustprogress norecurse nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #17 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #18 = { nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #19 = { nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #20 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #21 = { nounwind }
attributes #22 = { noreturn nounwind }
attributes #23 = { builtin allocsize(0) }
attributes #24 = { cold noreturn }
attributes #25 = { cold noreturn nounwind }
attributes #26 = { builtin nounwind }

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
!17 = !{!"_ZTSSt8functionIFiPiS0_jEE", !18, i64 0, !19, i64 24}
!18 = !{!"_ZTSSt14_Function_base", !8, i64 0, !19, i64 16}
!19 = !{!"any pointer", !8, i64 0}
!20 = !{!18, !19, i64 16}
!21 = !{!22, !19, i64 24}
!22 = !{!"_ZTSSt8functionIFiPfS0_jEE", !18, i64 0, !19, i64 24}
!23 = !{!24, !19, i64 24}
!24 = !{!"_ZTSSt8functionIFsPsS0_jEE", !18, i64 0, !19, i64 24}
!25 = !{!26, !19, i64 24}
!26 = !{!"_ZTSSt8functionIFjPjS0_jEE", !18, i64 0, !19, i64 24}
!27 = !{!28, !19, i64 24}
!28 = !{!"_ZTSSt8functionIFjPfS0_jEE", !18, i64 0, !19, i64 24}
!29 = !{!30, !19, i64 24}
!30 = !{!"_ZTSSt8functionIFtPtS0_jEE", !18, i64 0, !19, i64 24}
!31 = !{!32, !32, i64 0}
!32 = !{!"vtable pointer", !9, i64 0}
!33 = !{!34, !36, i64 32}
!34 = !{!"_ZTSSt8ios_base", !7, i64 8, !7, i64 16, !35, i64 24, !36, i64 28, !36, i64 32, !37, i64 40, !38, i64 48, !8, i64 64, !39, i64 192, !40, i64 200, !41, i64 208}
!35 = !{!"_ZTSSt13_Ios_Fmtflags", !8, i64 0}
!36 = !{!"_ZTSSt12_Ios_Iostate", !8, i64 0}
!37 = !{!"p1 _ZTSNSt8ios_base14_Callback_listE", !19, i64 0}
!38 = !{!"_ZTSNSt8ios_base6_WordsE", !19, i64 0, !7, i64 8}
!39 = !{!"int", !8, i64 0}
!40 = !{!"p1 _ZTSNSt8ios_base6_WordsE", !19, i64 0}
!41 = !{!"_ZTSSt6locale", !42, i64 0}
!42 = !{!"p1 _ZTSNSt6locale5_ImplE", !19, i64 0}
!43 = !{!39, !39, i64 0}
!44 = distinct !{!44, !11}
!45 = !{!46, !46, i64 0}
!46 = !{!"p1 int", !19, i64 0}
!47 = distinct !{!47, !11, !48, !49}
!48 = !{!"llvm.loop.isvectorized", i32 1}
!49 = !{!"llvm.loop.unroll.runtime.disable"}
!50 = distinct !{!50, !11, !48, !49}
!51 = distinct !{!51, !11, !48, !49}
!52 = distinct !{!52, !11, !48, !49}
!53 = distinct !{!53, !11, !48, !49}
!54 = distinct !{!54, !11, !48, !49}
!55 = !{!56, !56, i64 0}
!56 = !{!"p1 float", !19, i64 0}
!57 = !{!58, !58, i64 0}
!58 = !{!"float", !8, i64 0}
!59 = distinct !{!59, !11, !48, !49}
!60 = distinct !{!60, !11, !48, !49}
!61 = distinct !{!61, !11, !48, !49}
!62 = distinct !{!62, !11, !48, !49}
!63 = distinct !{!63, !11, !48, !49}
!64 = distinct !{!64, !11, !48, !49}
!65 = !{!66, !67, i64 0}
!66 = !{!"_ZTSNSt24uniform_int_distributionIsE10param_typeE", !67, i64 0, !67, i64 2}
!67 = !{!"short", !8, i64 0}
!68 = !{!66, !67, i64 2}
!69 = !{!67, !67, i64 0}
!70 = distinct !{!70, !11}
!71 = !{!72, !72, i64 0}
!72 = !{!"p1 short", !19, i64 0}
!73 = distinct !{!73, !11, !48, !49}
!74 = distinct !{!74, !11, !48, !49}
!75 = distinct !{!75, !11, !48, !49}
!76 = distinct !{!76, !11, !48, !49}
!77 = distinct !{!77, !11, !48, !49}
!78 = distinct !{!78, !11, !48, !49}
!79 = distinct !{!79, !11}
!80 = distinct !{!80, !11, !48, !49}
!81 = distinct !{!81, !11, !48, !49}
!82 = distinct !{!82, !11, !48, !49}
!83 = distinct !{!83, !11, !48, !49}
!84 = distinct !{!84, !11, !48, !49}
!85 = distinct !{!85, !11, !48, !49}
!86 = !{!87, !67, i64 0}
!87 = !{!"_ZTSNSt24uniform_int_distributionItE10param_typeE", !67, i64 0, !67, i64 2}
!88 = !{!87, !67, i64 2}
!89 = distinct !{!89, !11}
!90 = !{!91, !39, i64 4}
!91 = !{!"_ZTSNSt24uniform_int_distributionIiE10param_typeE", !39, i64 0, !39, i64 4}
!92 = !{!91, !39, i64 0}
!93 = distinct !{!93, !11}
!94 = distinct !{!94, !11}
!95 = distinct !{!95, !11, !48, !49}
!96 = distinct !{!96, !11, !48, !49}
!97 = distinct !{!97, !11, !98, !99}
!98 = !{!"llvm.loop.vectorize.width", i32 1}
!99 = !{!"llvm.loop.interleave.count", i32 1}
!100 = !{!19, !19, i64 0}
!101 = distinct !{!101, !11, !48, !49}
!102 = distinct !{!102, !11, !49, !48}
!103 = !{!"branch_weights", !"expected", i32 2000, i32 1}
!104 = distinct !{!104, !11, !48, !49}
!105 = distinct !{!105, !11, !48, !49}
!106 = distinct !{!106, !11}
!107 = distinct !{!107, !11}
!108 = distinct !{!108, !11, !98, !99}
!109 = distinct !{!109, !11, !48, !49}
!110 = distinct !{!110, !11, !49, !48}
!111 = distinct !{!111, !11}
!112 = distinct !{!112, !11}
!113 = distinct !{!113, !11, !98, !99}
!114 = distinct !{!114, !11, !48, !49}
!115 = distinct !{!115, !11, !48, !49}
!116 = distinct !{!116, !11, !49, !48}
!117 = distinct !{!117, !11, !98, !99}
!118 = distinct !{!118, !11, !48, !49}
!119 = distinct !{!119, !11, !49, !48}
!120 = distinct !{!120, !11, !98, !99}
!121 = distinct !{!121, !11, !48, !49}
!122 = distinct !{!122, !11, !49, !48}
!123 = distinct !{!123, !11, !98, !99}
!124 = distinct !{!124, !11, !48, !49}
!125 = distinct !{!125, !11, !48, !49}
!126 = distinct !{!126, !11, !49, !48}
!127 = !{!128, !39, i64 4}
!128 = !{!"_ZTSNSt24uniform_int_distributionIjE10param_typeE", !39, i64 0, !39, i64 4}
!129 = !{!128, !39, i64 0}
!130 = distinct !{!130, !11}
!131 = distinct !{!131, !11}
!132 = distinct !{!132, !11, !98, !99}
!133 = distinct !{!133, !11, !48, !49}
!134 = distinct !{!134, !11, !49, !48}
!135 = distinct !{!135, !11, !98, !99}
!136 = distinct !{!136, !11, !48, !49}
!137 = distinct !{!137, !11, !49, !48}
!138 = distinct !{!138, !11}
!139 = distinct !{!139, !11}
!140 = distinct !{!140, !11, !98, !99}
!141 = distinct !{!141, !11, !48, !49}
!142 = distinct !{!142, !11, !48, !49}
!143 = distinct !{!143, !11, !49, !48}
!144 = distinct !{!144, !11, !98, !99}
!145 = distinct !{!145, !11, !48, !49}
!146 = distinct !{!146, !11, !49, !48}
!147 = distinct !{!147, !11, !98, !99}
!148 = distinct !{!148, !11, !48, !49}
!149 = distinct !{!149, !11, !49, !48}
!150 = distinct !{!150, !11, !98, !99}
!151 = distinct !{!151, !11, !48, !49}
!152 = distinct !{!152, !11, !48, !49}
!153 = distinct !{!153, !11, !49, !48}
