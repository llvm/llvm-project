; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/Vectorizer/find-last.cpp'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/Vectorizer/find-last.cpp"
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
%"class.std::uniform_int_distribution" = type { %"struct.std::uniform_int_distribution<>::param_type" }
%"struct.std::uniform_int_distribution<>::param_type" = type { i32, i32 }
%"class.std::uniform_int_distribution.96" = type { %"struct.std::uniform_int_distribution<short>::param_type" }
%"struct.std::uniform_int_distribution<short>::param_type" = type { i16, i16 }

$__clang_call_terminate = comdat any

$_ZNSt24uniform_int_distributionIiEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEiRT_RKNS0_10param_typeE = comdat any

$_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv = comdat any

$_ZNSt24uniform_int_distributionIsEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEsRT_RKNS0_10param_typeE = comdat any

@_ZL3rng = internal global %"class.std::mersenne_twister_engine" zeroinitializer, align 8
@.str = private unnamed_addr constant [30 x i8] c"findlast_icmp_s32_true_update\00", align 1
@.str.1 = private unnamed_addr constant [30 x i8] c"findlast_fcmp_s32_true_update\00", align 1
@.str.2 = private unnamed_addr constant [30 x i8] c"findlast_icmp_s16_true_update\00", align 1
@.str.3 = private unnamed_addr constant [31 x i8] c"findlast_icmp_s32_false_update\00", align 1
@.str.4 = private unnamed_addr constant [31 x i8] c"findlast_fcmp_s32_false_update\00", align 1
@.str.5 = private unnamed_addr constant [31 x i8] c"findlast_icmp_s16_false_update\00", align 1
@.str.6 = private unnamed_addr constant [27 x i8] c"findlast_icmp_s32_start_TC\00", align 1
@.str.7 = private unnamed_addr constant [27 x i8] c"findlast_fcmp_s32_start_TC\00", align 1
@.str.8 = private unnamed_addr constant [27 x i8] c"findlast_icmp_s16_start_TC\00", align 1
@.str.9 = private unnamed_addr constant [24 x i8] c"findlast_icmp_s32_inc_2\00", align 1
@.str.10 = private unnamed_addr constant [24 x i8] c"findlast_fcmp_s32_inc_2\00", align 1
@.str.11 = private unnamed_addr constant [24 x i8] c"findlast_icmp_s16_inc_2\00", align 1
@.str.12 = private unnamed_addr constant [45 x i8] c"findlast_icmp_s32_start_decreasing_induction\00", align 1
@.str.13 = private unnamed_addr constant [45 x i8] c"findlast_fcmp_s32_start_decreasing_induction\00", align 1
@.str.14 = private unnamed_addr constant [45 x i8] c"findlast_icmp_s16_start_decreasing_induction\00", align 1
@.str.15 = private unnamed_addr constant [29 x i8] c"findlast_icmp_s32_iv_start_3\00", align 1
@.str.16 = private unnamed_addr constant [29 x i8] c"findlast_fcmp_s32_iv_start_3\00", align 1
@.str.17 = private unnamed_addr constant [29 x i8] c"findlast_icmp_s16_iv_start_3\00", align 1
@.str.18 = private unnamed_addr constant [37 x i8] c"findlast_icmp_s32_start_3_iv_start_3\00", align 1
@.str.19 = private unnamed_addr constant [37 x i8] c"findlast_fcmp_s32_start_3_iv_start_3\00", align 1
@.str.20 = private unnamed_addr constant [37 x i8] c"findlast_icmp_s16_start_3_iv_start_3\00", align 1
@.str.21 = private unnamed_addr constant [37 x i8] c"findlast_icmp_s32_start_2_iv_start_3\00", align 1
@.str.22 = private unnamed_addr constant [37 x i8] c"findlast_fcmp_s32_start_2_iv_start_3\00", align 1
@.str.23 = private unnamed_addr constant [37 x i8] c"findlast_icmp_s16_start_2_iv_start_3\00", align 1
@.str.24 = private unnamed_addr constant [37 x i8] c"findlast_icmp_s32_start_4_iv_start_3\00", align 1
@.str.25 = private unnamed_addr constant [37 x i8] c"findlast_fcmp_s32_start_4_iv_start_3\00", align 1
@.str.26 = private unnamed_addr constant [37 x i8] c"findlast_icmp_s16_start_4_iv_start_3\00", align 1
@_ZSt4cout = external global %"class.std::basic_ostream", align 8
@.str.27 = private unnamed_addr constant [10 x i8] c"Checking \00", align 1
@.str.28 = private unnamed_addr constant [2 x i8] c"\0A\00", align 1
@_ZSt4cerr = external global %"class.std::basic_ostream", align 8
@.str.29 = private unnamed_addr constant [12 x i8] c"Miscompare\0A\00", align 1
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
@"_ZTIZ4mainE4$_26" = internal constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @"_ZTSZ4mainE4$_26" }, align 8
@"_ZTSZ4mainE4$_26" = internal constant [13 x i8] c"Z4mainE4$_26\00", align 1
@"_ZTIZ4mainE4$_27" = internal constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @"_ZTSZ4mainE4$_27" }, align 8
@"_ZTSZ4mainE4$_27" = internal constant [13 x i8] c"Z4mainE4$_27\00", align 1
@"_ZTIZ4mainE4$_28" = internal constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @"_ZTSZ4mainE4$_28" }, align 8
@"_ZTSZ4mainE4$_28" = internal constant [13 x i8] c"Z4mainE4$_28\00", align 1
@"_ZTIZ4mainE4$_29" = internal constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @"_ZTSZ4mainE4$_29" }, align 8
@"_ZTSZ4mainE4$_29" = internal constant [13 x i8] c"Z4mainE4$_29\00", align 1
@"_ZTIZ4mainE4$_30" = internal constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @"_ZTSZ4mainE4$_30" }, align 8
@"_ZTSZ4mainE4$_30" = internal constant [13 x i8] c"Z4mainE4$_30\00", align 1
@"_ZTIZ4mainE4$_31" = internal constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @"_ZTSZ4mainE4$_31" }, align 8
@"_ZTSZ4mainE4$_31" = internal constant [13 x i8] c"Z4mainE4$_31\00", align 1
@"_ZTIZ4mainE4$_32" = internal constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @"_ZTSZ4mainE4$_32" }, align 8
@"_ZTSZ4mainE4$_32" = internal constant [13 x i8] c"Z4mainE4$_32\00", align 1
@"_ZTIZ4mainE4$_33" = internal constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @"_ZTSZ4mainE4$_33" }, align 8
@"_ZTSZ4mainE4$_33" = internal constant [13 x i8] c"Z4mainE4$_33\00", align 1
@"_ZTIZ4mainE4$_34" = internal constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @"_ZTSZ4mainE4$_34" }, align 8
@"_ZTSZ4mainE4$_34" = internal constant [13 x i8] c"Z4mainE4$_34\00", align 1
@"_ZTIZ4mainE4$_35" = internal constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @"_ZTSZ4mainE4$_35" }, align 8
@"_ZTSZ4mainE4$_35" = internal constant [13 x i8] c"Z4mainE4$_35\00", align 1
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_find_last.cpp, ptr null }]

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
  %14 = alloca %"class.std::function", align 8
  %15 = alloca %"class.std::function", align 8
  %16 = alloca %"class.std::function.2", align 8
  %17 = alloca %"class.std::function.2", align 8
  %18 = alloca %"class.std::function.8", align 8
  %19 = alloca %"class.std::function.8", align 8
  %20 = alloca %"class.std::function", align 8
  %21 = alloca %"class.std::function", align 8
  %22 = alloca %"class.std::function.2", align 8
  %23 = alloca %"class.std::function.2", align 8
  %24 = alloca %"class.std::function.8", align 8
  %25 = alloca %"class.std::function.8", align 8
  %26 = alloca %"class.std::function", align 8
  %27 = alloca %"class.std::function", align 8
  %28 = alloca %"class.std::function.2", align 8
  %29 = alloca %"class.std::function.2", align 8
  %30 = alloca %"class.std::function.8", align 8
  %31 = alloca %"class.std::function.8", align 8
  %32 = alloca %"class.std::function", align 8
  %33 = alloca %"class.std::function", align 8
  %34 = alloca %"class.std::function.2", align 8
  %35 = alloca %"class.std::function.2", align 8
  %36 = alloca %"class.std::function.8", align 8
  %37 = alloca %"class.std::function.8", align 8
  %38 = alloca %"class.std::function", align 8
  %39 = alloca %"class.std::function", align 8
  %40 = alloca %"class.std::function.2", align 8
  %41 = alloca %"class.std::function.2", align 8
  %42 = alloca %"class.std::function.8", align 8
  %43 = alloca %"class.std::function.8", align 8
  %44 = alloca %"class.std::function", align 8
  %45 = alloca %"class.std::function", align 8
  %46 = alloca %"class.std::function.2", align 8
  %47 = alloca %"class.std::function.2", align 8
  %48 = alloca %"class.std::function.8", align 8
  %49 = alloca %"class.std::function.8", align 8
  %50 = alloca %"class.std::function", align 8
  %51 = alloca %"class.std::function", align 8
  %52 = alloca %"class.std::function.2", align 8
  %53 = alloca %"class.std::function.2", align 8
  %54 = alloca %"class.std::function.8", align 8
  %55 = alloca %"class.std::function.8", align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #21
  store i64 15, ptr %1, align 8, !tbaa !6
  br label %56

56:                                               ; preds = %56, %0
  %57 = phi i64 [ 15, %0 ], [ %64, %56 ]
  %58 = phi i64 [ 1, %0 ], [ %65, %56 ]
  %59 = getelementptr i64, ptr %1, i64 %58
  %60 = lshr i64 %57, 30
  %61 = xor i64 %60, %57
  %62 = mul nuw nsw i64 %61, 1812433253
  %63 = add nuw i64 %62, %58
  %64 = and i64 %63, 4294967295
  store i64 %64, ptr %59, align 8, !tbaa !6
  %65 = add nuw nsw i64 %58, 1
  %66 = icmp eq i64 %65, 624
  br i1 %66, label %67, label %56, !llvm.loop !10

67:                                               ; preds = %56
  %68 = getelementptr inbounds nuw i8, ptr %1, i64 4992
  store i64 624, ptr %68, align 8, !tbaa !12
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(5000) @_ZL3rng, ptr noundef nonnull align 8 dereferenceable(5000) %1, i64 5000, i1 false), !tbaa.struct !14
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #21
  %69 = getelementptr inbounds nuw i8, ptr %2, i64 16
  %70 = getelementptr inbounds nuw i8, ptr %2, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %2, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE3$_0E9_M_invokeERKSt9_Any_dataOS0_S7_Oi", ptr %70, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE3$_0E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %69, align 8, !tbaa !20
  %71 = getelementptr inbounds nuw i8, ptr %3, i64 16
  %72 = getelementptr inbounds nuw i8, ptr %3, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %3, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE3$_1E9_M_invokeERKSt9_Any_dataOS0_S7_Oi", ptr %72, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE3$_1E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %71, align 8, !tbaa !20
  invoke fastcc void @_ZL19checkVectorFunctionIiiEvSt8functionIFT_PT0_S3_S1_EES5_PKc(ptr noundef %2, ptr noundef %3, ptr noundef nonnull @.str)
          to label %73 unwind label %636

73:                                               ; preds = %67
  %74 = load ptr, ptr %71, align 8, !tbaa !20
  %75 = icmp eq ptr %74, null
  br i1 %75, label %81, label %76

76:                                               ; preds = %73
  %77 = invoke noundef i1 %74(ptr noundef nonnull align 8 dereferenceable(32) %3, ptr noundef nonnull align 8 dereferenceable(32) %3, i32 noundef 3)
          to label %81 unwind label %78

78:                                               ; preds = %76
  %79 = landingpad { ptr, i32 }
          catch ptr null
  %80 = extractvalue { ptr, i32 } %79, 0
  call void @__clang_call_terminate(ptr %80) #22
  unreachable

81:                                               ; preds = %73, %76
  %82 = load ptr, ptr %69, align 8, !tbaa !20
  %83 = icmp eq ptr %82, null
  br i1 %83, label %89, label %84

84:                                               ; preds = %81
  %85 = invoke noundef i1 %82(ptr noundef nonnull align 8 dereferenceable(32) %2, ptr noundef nonnull align 8 dereferenceable(32) %2, i32 noundef 3)
          to label %89 unwind label %86

86:                                               ; preds = %84
  %87 = landingpad { ptr, i32 }
          catch ptr null
  %88 = extractvalue { ptr, i32 } %87, 0
  call void @__clang_call_terminate(ptr %88) #22
  unreachable

89:                                               ; preds = %81, %84
  %90 = getelementptr inbounds nuw i8, ptr %4, i64 16
  %91 = getelementptr inbounds nuw i8, ptr %4, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %4, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE3$_0E9_M_invokeERKSt9_Any_dataOS0_S7_Oi", ptr %91, align 8, !tbaa !21
  store ptr @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE3$_0E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %90, align 8, !tbaa !20
  %92 = getelementptr inbounds nuw i8, ptr %5, i64 16
  %93 = getelementptr inbounds nuw i8, ptr %5, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %5, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE3$_1E9_M_invokeERKSt9_Any_dataOS0_S7_Oi", ptr %93, align 8, !tbaa !21
  store ptr @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE3$_1E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %92, align 8, !tbaa !20
  invoke fastcc void @_ZL19checkVectorFunctionIifEvSt8functionIFT_PT0_S3_S1_EES5_PKc(ptr noundef %4, ptr noundef %5, ptr noundef nonnull @.str.1)
          to label %94 unwind label %653

94:                                               ; preds = %89
  %95 = load ptr, ptr %92, align 8, !tbaa !20
  %96 = icmp eq ptr %95, null
  br i1 %96, label %102, label %97

97:                                               ; preds = %94
  %98 = invoke noundef i1 %95(ptr noundef nonnull align 8 dereferenceable(32) %5, ptr noundef nonnull align 8 dereferenceable(32) %5, i32 noundef 3)
          to label %102 unwind label %99

99:                                               ; preds = %97
  %100 = landingpad { ptr, i32 }
          catch ptr null
  %101 = extractvalue { ptr, i32 } %100, 0
  call void @__clang_call_terminate(ptr %101) #22
  unreachable

102:                                              ; preds = %94, %97
  %103 = load ptr, ptr %90, align 8, !tbaa !20
  %104 = icmp eq ptr %103, null
  br i1 %104, label %110, label %105

105:                                              ; preds = %102
  %106 = invoke noundef i1 %103(ptr noundef nonnull align 8 dereferenceable(32) %4, ptr noundef nonnull align 8 dereferenceable(32) %4, i32 noundef 3)
          to label %110 unwind label %107

107:                                              ; preds = %105
  %108 = landingpad { ptr, i32 }
          catch ptr null
  %109 = extractvalue { ptr, i32 } %108, 0
  call void @__clang_call_terminate(ptr %109) #22
  unreachable

110:                                              ; preds = %102, %105
  %111 = getelementptr inbounds nuw i8, ptr %6, i64 16
  %112 = getelementptr inbounds nuw i8, ptr %6, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %6, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE3$_2E9_M_invokeERKSt9_Any_dataOS0_S7_Os", ptr %112, align 8, !tbaa !23
  store ptr @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE3$_2E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %111, align 8, !tbaa !20
  %113 = getelementptr inbounds nuw i8, ptr %7, i64 16
  %114 = getelementptr inbounds nuw i8, ptr %7, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %7, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE3$_3E9_M_invokeERKSt9_Any_dataOS0_S7_Os", ptr %114, align 8, !tbaa !23
  store ptr @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE3$_3E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %113, align 8, !tbaa !20
  invoke fastcc void @_ZL19checkVectorFunctionIssEvSt8functionIFT_PT0_S3_S1_EES5_PKc(ptr noundef %6, ptr noundef %7, ptr noundef nonnull @.str.2)
          to label %115 unwind label %670

115:                                              ; preds = %110
  %116 = load ptr, ptr %113, align 8, !tbaa !20
  %117 = icmp eq ptr %116, null
  br i1 %117, label %123, label %118

118:                                              ; preds = %115
  %119 = invoke noundef i1 %116(ptr noundef nonnull align 8 dereferenceable(32) %7, ptr noundef nonnull align 8 dereferenceable(32) %7, i32 noundef 3)
          to label %123 unwind label %120

120:                                              ; preds = %118
  %121 = landingpad { ptr, i32 }
          catch ptr null
  %122 = extractvalue { ptr, i32 } %121, 0
  call void @__clang_call_terminate(ptr %122) #22
  unreachable

123:                                              ; preds = %115, %118
  %124 = load ptr, ptr %111, align 8, !tbaa !20
  %125 = icmp eq ptr %124, null
  br i1 %125, label %131, label %126

126:                                              ; preds = %123
  %127 = invoke noundef i1 %124(ptr noundef nonnull align 8 dereferenceable(32) %6, ptr noundef nonnull align 8 dereferenceable(32) %6, i32 noundef 3)
          to label %131 unwind label %128

128:                                              ; preds = %126
  %129 = landingpad { ptr, i32 }
          catch ptr null
  %130 = extractvalue { ptr, i32 } %129, 0
  call void @__clang_call_terminate(ptr %130) #22
  unreachable

131:                                              ; preds = %123, %126
  %132 = getelementptr inbounds nuw i8, ptr %8, i64 16
  %133 = getelementptr inbounds nuw i8, ptr %8, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %8, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE3$_4E9_M_invokeERKSt9_Any_dataOS0_S7_Oi", ptr %133, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE3$_4E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %132, align 8, !tbaa !20
  %134 = getelementptr inbounds nuw i8, ptr %9, i64 16
  %135 = getelementptr inbounds nuw i8, ptr %9, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %9, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE3$_5E9_M_invokeERKSt9_Any_dataOS0_S7_Oi", ptr %135, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE3$_5E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %134, align 8, !tbaa !20
  invoke fastcc void @_ZL19checkVectorFunctionIiiEvSt8functionIFT_PT0_S3_S1_EES5_PKc(ptr noundef %8, ptr noundef %9, ptr noundef nonnull @.str.3)
          to label %136 unwind label %687

136:                                              ; preds = %131
  %137 = load ptr, ptr %134, align 8, !tbaa !20
  %138 = icmp eq ptr %137, null
  br i1 %138, label %144, label %139

139:                                              ; preds = %136
  %140 = invoke noundef i1 %137(ptr noundef nonnull align 8 dereferenceable(32) %9, ptr noundef nonnull align 8 dereferenceable(32) %9, i32 noundef 3)
          to label %144 unwind label %141

141:                                              ; preds = %139
  %142 = landingpad { ptr, i32 }
          catch ptr null
  %143 = extractvalue { ptr, i32 } %142, 0
  call void @__clang_call_terminate(ptr %143) #22
  unreachable

144:                                              ; preds = %136, %139
  %145 = load ptr, ptr %132, align 8, !tbaa !20
  %146 = icmp eq ptr %145, null
  br i1 %146, label %152, label %147

147:                                              ; preds = %144
  %148 = invoke noundef i1 %145(ptr noundef nonnull align 8 dereferenceable(32) %8, ptr noundef nonnull align 8 dereferenceable(32) %8, i32 noundef 3)
          to label %152 unwind label %149

149:                                              ; preds = %147
  %150 = landingpad { ptr, i32 }
          catch ptr null
  %151 = extractvalue { ptr, i32 } %150, 0
  call void @__clang_call_terminate(ptr %151) #22
  unreachable

152:                                              ; preds = %144, %147
  %153 = getelementptr inbounds nuw i8, ptr %10, i64 16
  %154 = getelementptr inbounds nuw i8, ptr %10, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %10, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE3$_4E9_M_invokeERKSt9_Any_dataOS0_S7_Oi", ptr %154, align 8, !tbaa !21
  store ptr @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE3$_4E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %153, align 8, !tbaa !20
  %155 = getelementptr inbounds nuw i8, ptr %11, i64 16
  %156 = getelementptr inbounds nuw i8, ptr %11, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %11, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE3$_5E9_M_invokeERKSt9_Any_dataOS0_S7_Oi", ptr %156, align 8, !tbaa !21
  store ptr @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE3$_5E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %155, align 8, !tbaa !20
  invoke fastcc void @_ZL19checkVectorFunctionIifEvSt8functionIFT_PT0_S3_S1_EES5_PKc(ptr noundef %10, ptr noundef %11, ptr noundef nonnull @.str.4)
          to label %157 unwind label %704

157:                                              ; preds = %152
  %158 = load ptr, ptr %155, align 8, !tbaa !20
  %159 = icmp eq ptr %158, null
  br i1 %159, label %165, label %160

160:                                              ; preds = %157
  %161 = invoke noundef i1 %158(ptr noundef nonnull align 8 dereferenceable(32) %11, ptr noundef nonnull align 8 dereferenceable(32) %11, i32 noundef 3)
          to label %165 unwind label %162

162:                                              ; preds = %160
  %163 = landingpad { ptr, i32 }
          catch ptr null
  %164 = extractvalue { ptr, i32 } %163, 0
  call void @__clang_call_terminate(ptr %164) #22
  unreachable

165:                                              ; preds = %157, %160
  %166 = load ptr, ptr %153, align 8, !tbaa !20
  %167 = icmp eq ptr %166, null
  br i1 %167, label %173, label %168

168:                                              ; preds = %165
  %169 = invoke noundef i1 %166(ptr noundef nonnull align 8 dereferenceable(32) %10, ptr noundef nonnull align 8 dereferenceable(32) %10, i32 noundef 3)
          to label %173 unwind label %170

170:                                              ; preds = %168
  %171 = landingpad { ptr, i32 }
          catch ptr null
  %172 = extractvalue { ptr, i32 } %171, 0
  call void @__clang_call_terminate(ptr %172) #22
  unreachable

173:                                              ; preds = %165, %168
  %174 = getelementptr inbounds nuw i8, ptr %12, i64 16
  %175 = getelementptr inbounds nuw i8, ptr %12, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %12, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE3$_6E9_M_invokeERKSt9_Any_dataOS0_S7_Os", ptr %175, align 8, !tbaa !23
  store ptr @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE3$_6E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %174, align 8, !tbaa !20
  %176 = getelementptr inbounds nuw i8, ptr %13, i64 16
  %177 = getelementptr inbounds nuw i8, ptr %13, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %13, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE3$_7E9_M_invokeERKSt9_Any_dataOS0_S7_Os", ptr %177, align 8, !tbaa !23
  store ptr @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE3$_7E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %176, align 8, !tbaa !20
  invoke fastcc void @_ZL19checkVectorFunctionIssEvSt8functionIFT_PT0_S3_S1_EES5_PKc(ptr noundef %12, ptr noundef %13, ptr noundef nonnull @.str.5)
          to label %178 unwind label %721

178:                                              ; preds = %173
  %179 = load ptr, ptr %176, align 8, !tbaa !20
  %180 = icmp eq ptr %179, null
  br i1 %180, label %186, label %181

181:                                              ; preds = %178
  %182 = invoke noundef i1 %179(ptr noundef nonnull align 8 dereferenceable(32) %13, ptr noundef nonnull align 8 dereferenceable(32) %13, i32 noundef 3)
          to label %186 unwind label %183

183:                                              ; preds = %181
  %184 = landingpad { ptr, i32 }
          catch ptr null
  %185 = extractvalue { ptr, i32 } %184, 0
  call void @__clang_call_terminate(ptr %185) #22
  unreachable

186:                                              ; preds = %178, %181
  %187 = load ptr, ptr %174, align 8, !tbaa !20
  %188 = icmp eq ptr %187, null
  br i1 %188, label %194, label %189

189:                                              ; preds = %186
  %190 = invoke noundef i1 %187(ptr noundef nonnull align 8 dereferenceable(32) %12, ptr noundef nonnull align 8 dereferenceable(32) %12, i32 noundef 3)
          to label %194 unwind label %191

191:                                              ; preds = %189
  %192 = landingpad { ptr, i32 }
          catch ptr null
  %193 = extractvalue { ptr, i32 } %192, 0
  call void @__clang_call_terminate(ptr %193) #22
  unreachable

194:                                              ; preds = %186, %189
  %195 = getelementptr inbounds nuw i8, ptr %14, i64 16
  %196 = getelementptr inbounds nuw i8, ptr %14, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %14, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE3$_8E9_M_invokeERKSt9_Any_dataOS0_S7_Oi", ptr %196, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE3$_8E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %195, align 8, !tbaa !20
  %197 = getelementptr inbounds nuw i8, ptr %15, i64 16
  %198 = getelementptr inbounds nuw i8, ptr %15, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %15, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE3$_9E9_M_invokeERKSt9_Any_dataOS0_S7_Oi", ptr %198, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE3$_9E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %197, align 8, !tbaa !20
  invoke fastcc void @_ZL19checkVectorFunctionIiiEvSt8functionIFT_PT0_S3_S1_EES5_PKc(ptr noundef %14, ptr noundef %15, ptr noundef nonnull @.str.6)
          to label %199 unwind label %738

199:                                              ; preds = %194
  %200 = load ptr, ptr %197, align 8, !tbaa !20
  %201 = icmp eq ptr %200, null
  br i1 %201, label %207, label %202

202:                                              ; preds = %199
  %203 = invoke noundef i1 %200(ptr noundef nonnull align 8 dereferenceable(32) %15, ptr noundef nonnull align 8 dereferenceable(32) %15, i32 noundef 3)
          to label %207 unwind label %204

204:                                              ; preds = %202
  %205 = landingpad { ptr, i32 }
          catch ptr null
  %206 = extractvalue { ptr, i32 } %205, 0
  call void @__clang_call_terminate(ptr %206) #22
  unreachable

207:                                              ; preds = %199, %202
  %208 = load ptr, ptr %195, align 8, !tbaa !20
  %209 = icmp eq ptr %208, null
  br i1 %209, label %215, label %210

210:                                              ; preds = %207
  %211 = invoke noundef i1 %208(ptr noundef nonnull align 8 dereferenceable(32) %14, ptr noundef nonnull align 8 dereferenceable(32) %14, i32 noundef 3)
          to label %215 unwind label %212

212:                                              ; preds = %210
  %213 = landingpad { ptr, i32 }
          catch ptr null
  %214 = extractvalue { ptr, i32 } %213, 0
  call void @__clang_call_terminate(ptr %214) #22
  unreachable

215:                                              ; preds = %207, %210
  %216 = getelementptr inbounds nuw i8, ptr %16, i64 16
  %217 = getelementptr inbounds nuw i8, ptr %16, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %16, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE3$_8E9_M_invokeERKSt9_Any_dataOS0_S7_Oi", ptr %217, align 8, !tbaa !21
  store ptr @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE3$_8E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %216, align 8, !tbaa !20
  %218 = getelementptr inbounds nuw i8, ptr %17, i64 16
  %219 = getelementptr inbounds nuw i8, ptr %17, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %17, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE3$_9E9_M_invokeERKSt9_Any_dataOS0_S7_Oi", ptr %219, align 8, !tbaa !21
  store ptr @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE3$_9E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %218, align 8, !tbaa !20
  invoke fastcc void @_ZL19checkVectorFunctionIifEvSt8functionIFT_PT0_S3_S1_EES5_PKc(ptr noundef %16, ptr noundef %17, ptr noundef nonnull @.str.7)
          to label %220 unwind label %755

220:                                              ; preds = %215
  %221 = load ptr, ptr %218, align 8, !tbaa !20
  %222 = icmp eq ptr %221, null
  br i1 %222, label %228, label %223

223:                                              ; preds = %220
  %224 = invoke noundef i1 %221(ptr noundef nonnull align 8 dereferenceable(32) %17, ptr noundef nonnull align 8 dereferenceable(32) %17, i32 noundef 3)
          to label %228 unwind label %225

225:                                              ; preds = %223
  %226 = landingpad { ptr, i32 }
          catch ptr null
  %227 = extractvalue { ptr, i32 } %226, 0
  call void @__clang_call_terminate(ptr %227) #22
  unreachable

228:                                              ; preds = %220, %223
  %229 = load ptr, ptr %216, align 8, !tbaa !20
  %230 = icmp eq ptr %229, null
  br i1 %230, label %236, label %231

231:                                              ; preds = %228
  %232 = invoke noundef i1 %229(ptr noundef nonnull align 8 dereferenceable(32) %16, ptr noundef nonnull align 8 dereferenceable(32) %16, i32 noundef 3)
          to label %236 unwind label %233

233:                                              ; preds = %231
  %234 = landingpad { ptr, i32 }
          catch ptr null
  %235 = extractvalue { ptr, i32 } %234, 0
  call void @__clang_call_terminate(ptr %235) #22
  unreachable

236:                                              ; preds = %228, %231
  %237 = getelementptr inbounds nuw i8, ptr %18, i64 16
  %238 = getelementptr inbounds nuw i8, ptr %18, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %18, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE4$_10E9_M_invokeERKSt9_Any_dataOS0_S7_Os", ptr %238, align 8, !tbaa !23
  store ptr @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE4$_10E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %237, align 8, !tbaa !20
  %239 = getelementptr inbounds nuw i8, ptr %19, i64 16
  %240 = getelementptr inbounds nuw i8, ptr %19, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %19, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE4$_11E9_M_invokeERKSt9_Any_dataOS0_S7_Os", ptr %240, align 8, !tbaa !23
  store ptr @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE4$_11E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %239, align 8, !tbaa !20
  invoke fastcc void @_ZL19checkVectorFunctionIssEvSt8functionIFT_PT0_S3_S1_EES5_PKc(ptr noundef %18, ptr noundef %19, ptr noundef nonnull @.str.8)
          to label %241 unwind label %772

241:                                              ; preds = %236
  %242 = load ptr, ptr %239, align 8, !tbaa !20
  %243 = icmp eq ptr %242, null
  br i1 %243, label %249, label %244

244:                                              ; preds = %241
  %245 = invoke noundef i1 %242(ptr noundef nonnull align 8 dereferenceable(32) %19, ptr noundef nonnull align 8 dereferenceable(32) %19, i32 noundef 3)
          to label %249 unwind label %246

246:                                              ; preds = %244
  %247 = landingpad { ptr, i32 }
          catch ptr null
  %248 = extractvalue { ptr, i32 } %247, 0
  call void @__clang_call_terminate(ptr %248) #22
  unreachable

249:                                              ; preds = %241, %244
  %250 = load ptr, ptr %237, align 8, !tbaa !20
  %251 = icmp eq ptr %250, null
  br i1 %251, label %257, label %252

252:                                              ; preds = %249
  %253 = invoke noundef i1 %250(ptr noundef nonnull align 8 dereferenceable(32) %18, ptr noundef nonnull align 8 dereferenceable(32) %18, i32 noundef 3)
          to label %257 unwind label %254

254:                                              ; preds = %252
  %255 = landingpad { ptr, i32 }
          catch ptr null
  %256 = extractvalue { ptr, i32 } %255, 0
  call void @__clang_call_terminate(ptr %256) #22
  unreachable

257:                                              ; preds = %249, %252
  %258 = getelementptr inbounds nuw i8, ptr %20, i64 16
  %259 = getelementptr inbounds nuw i8, ptr %20, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %20, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE4$_12E9_M_invokeERKSt9_Any_dataOS0_S7_Oi", ptr %259, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE4$_12E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %258, align 8, !tbaa !20
  %260 = getelementptr inbounds nuw i8, ptr %21, i64 16
  %261 = getelementptr inbounds nuw i8, ptr %21, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %21, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE4$_13E9_M_invokeERKSt9_Any_dataOS0_S7_Oi", ptr %261, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE4$_13E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %260, align 8, !tbaa !20
  invoke fastcc void @_ZL19checkVectorFunctionIiiEvSt8functionIFT_PT0_S3_S1_EES5_PKc(ptr noundef %20, ptr noundef %21, ptr noundef nonnull @.str.9)
          to label %262 unwind label %789

262:                                              ; preds = %257
  %263 = load ptr, ptr %260, align 8, !tbaa !20
  %264 = icmp eq ptr %263, null
  br i1 %264, label %270, label %265

265:                                              ; preds = %262
  %266 = invoke noundef i1 %263(ptr noundef nonnull align 8 dereferenceable(32) %21, ptr noundef nonnull align 8 dereferenceable(32) %21, i32 noundef 3)
          to label %270 unwind label %267

267:                                              ; preds = %265
  %268 = landingpad { ptr, i32 }
          catch ptr null
  %269 = extractvalue { ptr, i32 } %268, 0
  call void @__clang_call_terminate(ptr %269) #22
  unreachable

270:                                              ; preds = %262, %265
  %271 = load ptr, ptr %258, align 8, !tbaa !20
  %272 = icmp eq ptr %271, null
  br i1 %272, label %278, label %273

273:                                              ; preds = %270
  %274 = invoke noundef i1 %271(ptr noundef nonnull align 8 dereferenceable(32) %20, ptr noundef nonnull align 8 dereferenceable(32) %20, i32 noundef 3)
          to label %278 unwind label %275

275:                                              ; preds = %273
  %276 = landingpad { ptr, i32 }
          catch ptr null
  %277 = extractvalue { ptr, i32 } %276, 0
  call void @__clang_call_terminate(ptr %277) #22
  unreachable

278:                                              ; preds = %270, %273
  %279 = getelementptr inbounds nuw i8, ptr %22, i64 16
  %280 = getelementptr inbounds nuw i8, ptr %22, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %22, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE4$_12E9_M_invokeERKSt9_Any_dataOS0_S7_Oi", ptr %280, align 8, !tbaa !21
  store ptr @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE4$_12E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %279, align 8, !tbaa !20
  %281 = getelementptr inbounds nuw i8, ptr %23, i64 16
  %282 = getelementptr inbounds nuw i8, ptr %23, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %23, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE4$_13E9_M_invokeERKSt9_Any_dataOS0_S7_Oi", ptr %282, align 8, !tbaa !21
  store ptr @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE4$_13E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %281, align 8, !tbaa !20
  invoke fastcc void @_ZL19checkVectorFunctionIifEvSt8functionIFT_PT0_S3_S1_EES5_PKc(ptr noundef %22, ptr noundef %23, ptr noundef nonnull @.str.10)
          to label %283 unwind label %806

283:                                              ; preds = %278
  %284 = load ptr, ptr %281, align 8, !tbaa !20
  %285 = icmp eq ptr %284, null
  br i1 %285, label %291, label %286

286:                                              ; preds = %283
  %287 = invoke noundef i1 %284(ptr noundef nonnull align 8 dereferenceable(32) %23, ptr noundef nonnull align 8 dereferenceable(32) %23, i32 noundef 3)
          to label %291 unwind label %288

288:                                              ; preds = %286
  %289 = landingpad { ptr, i32 }
          catch ptr null
  %290 = extractvalue { ptr, i32 } %289, 0
  call void @__clang_call_terminate(ptr %290) #22
  unreachable

291:                                              ; preds = %283, %286
  %292 = load ptr, ptr %279, align 8, !tbaa !20
  %293 = icmp eq ptr %292, null
  br i1 %293, label %299, label %294

294:                                              ; preds = %291
  %295 = invoke noundef i1 %292(ptr noundef nonnull align 8 dereferenceable(32) %22, ptr noundef nonnull align 8 dereferenceable(32) %22, i32 noundef 3)
          to label %299 unwind label %296

296:                                              ; preds = %294
  %297 = landingpad { ptr, i32 }
          catch ptr null
  %298 = extractvalue { ptr, i32 } %297, 0
  call void @__clang_call_terminate(ptr %298) #22
  unreachable

299:                                              ; preds = %291, %294
  %300 = getelementptr inbounds nuw i8, ptr %24, i64 16
  %301 = getelementptr inbounds nuw i8, ptr %24, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %24, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE4$_14E9_M_invokeERKSt9_Any_dataOS0_S7_Os", ptr %301, align 8, !tbaa !23
  store ptr @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE4$_14E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %300, align 8, !tbaa !20
  %302 = getelementptr inbounds nuw i8, ptr %25, i64 16
  %303 = getelementptr inbounds nuw i8, ptr %25, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %25, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE4$_15E9_M_invokeERKSt9_Any_dataOS0_S7_Os", ptr %303, align 8, !tbaa !23
  store ptr @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE4$_15E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %302, align 8, !tbaa !20
  invoke fastcc void @_ZL19checkVectorFunctionIssEvSt8functionIFT_PT0_S3_S1_EES5_PKc(ptr noundef %24, ptr noundef %25, ptr noundef nonnull @.str.11)
          to label %304 unwind label %823

304:                                              ; preds = %299
  %305 = load ptr, ptr %302, align 8, !tbaa !20
  %306 = icmp eq ptr %305, null
  br i1 %306, label %312, label %307

307:                                              ; preds = %304
  %308 = invoke noundef i1 %305(ptr noundef nonnull align 8 dereferenceable(32) %25, ptr noundef nonnull align 8 dereferenceable(32) %25, i32 noundef 3)
          to label %312 unwind label %309

309:                                              ; preds = %307
  %310 = landingpad { ptr, i32 }
          catch ptr null
  %311 = extractvalue { ptr, i32 } %310, 0
  call void @__clang_call_terminate(ptr %311) #22
  unreachable

312:                                              ; preds = %304, %307
  %313 = load ptr, ptr %300, align 8, !tbaa !20
  %314 = icmp eq ptr %313, null
  br i1 %314, label %320, label %315

315:                                              ; preds = %312
  %316 = invoke noundef i1 %313(ptr noundef nonnull align 8 dereferenceable(32) %24, ptr noundef nonnull align 8 dereferenceable(32) %24, i32 noundef 3)
          to label %320 unwind label %317

317:                                              ; preds = %315
  %318 = landingpad { ptr, i32 }
          catch ptr null
  %319 = extractvalue { ptr, i32 } %318, 0
  call void @__clang_call_terminate(ptr %319) #22
  unreachable

320:                                              ; preds = %312, %315
  %321 = getelementptr inbounds nuw i8, ptr %26, i64 16
  %322 = getelementptr inbounds nuw i8, ptr %26, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %26, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE4$_16E9_M_invokeERKSt9_Any_dataOS0_S7_Oi", ptr %322, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE4$_16E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %321, align 8, !tbaa !20
  %323 = getelementptr inbounds nuw i8, ptr %27, i64 16
  %324 = getelementptr inbounds nuw i8, ptr %27, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %27, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE4$_17E9_M_invokeERKSt9_Any_dataOS0_S7_Oi", ptr %324, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE4$_17E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %323, align 8, !tbaa !20
  invoke fastcc void @_ZL19checkVectorFunctionIiiEvSt8functionIFT_PT0_S3_S1_EES5_PKc(ptr noundef %26, ptr noundef %27, ptr noundef nonnull @.str.12)
          to label %325 unwind label %840

325:                                              ; preds = %320
  %326 = load ptr, ptr %323, align 8, !tbaa !20
  %327 = icmp eq ptr %326, null
  br i1 %327, label %333, label %328

328:                                              ; preds = %325
  %329 = invoke noundef i1 %326(ptr noundef nonnull align 8 dereferenceable(32) %27, ptr noundef nonnull align 8 dereferenceable(32) %27, i32 noundef 3)
          to label %333 unwind label %330

330:                                              ; preds = %328
  %331 = landingpad { ptr, i32 }
          catch ptr null
  %332 = extractvalue { ptr, i32 } %331, 0
  call void @__clang_call_terminate(ptr %332) #22
  unreachable

333:                                              ; preds = %325, %328
  %334 = load ptr, ptr %321, align 8, !tbaa !20
  %335 = icmp eq ptr %334, null
  br i1 %335, label %341, label %336

336:                                              ; preds = %333
  %337 = invoke noundef i1 %334(ptr noundef nonnull align 8 dereferenceable(32) %26, ptr noundef nonnull align 8 dereferenceable(32) %26, i32 noundef 3)
          to label %341 unwind label %338

338:                                              ; preds = %336
  %339 = landingpad { ptr, i32 }
          catch ptr null
  %340 = extractvalue { ptr, i32 } %339, 0
  call void @__clang_call_terminate(ptr %340) #22
  unreachable

341:                                              ; preds = %333, %336
  %342 = getelementptr inbounds nuw i8, ptr %28, i64 16
  %343 = getelementptr inbounds nuw i8, ptr %28, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %28, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE4$_16E9_M_invokeERKSt9_Any_dataOS0_S7_Oi", ptr %343, align 8, !tbaa !21
  store ptr @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE4$_16E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %342, align 8, !tbaa !20
  %344 = getelementptr inbounds nuw i8, ptr %29, i64 16
  %345 = getelementptr inbounds nuw i8, ptr %29, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %29, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE4$_17E9_M_invokeERKSt9_Any_dataOS0_S7_Oi", ptr %345, align 8, !tbaa !21
  store ptr @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE4$_17E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %344, align 8, !tbaa !20
  invoke fastcc void @_ZL19checkVectorFunctionIifEvSt8functionIFT_PT0_S3_S1_EES5_PKc(ptr noundef %28, ptr noundef %29, ptr noundef nonnull @.str.13)
          to label %346 unwind label %857

346:                                              ; preds = %341
  %347 = load ptr, ptr %344, align 8, !tbaa !20
  %348 = icmp eq ptr %347, null
  br i1 %348, label %354, label %349

349:                                              ; preds = %346
  %350 = invoke noundef i1 %347(ptr noundef nonnull align 8 dereferenceable(32) %29, ptr noundef nonnull align 8 dereferenceable(32) %29, i32 noundef 3)
          to label %354 unwind label %351

351:                                              ; preds = %349
  %352 = landingpad { ptr, i32 }
          catch ptr null
  %353 = extractvalue { ptr, i32 } %352, 0
  call void @__clang_call_terminate(ptr %353) #22
  unreachable

354:                                              ; preds = %346, %349
  %355 = load ptr, ptr %342, align 8, !tbaa !20
  %356 = icmp eq ptr %355, null
  br i1 %356, label %362, label %357

357:                                              ; preds = %354
  %358 = invoke noundef i1 %355(ptr noundef nonnull align 8 dereferenceable(32) %28, ptr noundef nonnull align 8 dereferenceable(32) %28, i32 noundef 3)
          to label %362 unwind label %359

359:                                              ; preds = %357
  %360 = landingpad { ptr, i32 }
          catch ptr null
  %361 = extractvalue { ptr, i32 } %360, 0
  call void @__clang_call_terminate(ptr %361) #22
  unreachable

362:                                              ; preds = %354, %357
  %363 = getelementptr inbounds nuw i8, ptr %30, i64 16
  %364 = getelementptr inbounds nuw i8, ptr %30, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %30, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE4$_18E9_M_invokeERKSt9_Any_dataOS0_S7_Os", ptr %364, align 8, !tbaa !23
  store ptr @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE4$_18E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %363, align 8, !tbaa !20
  %365 = getelementptr inbounds nuw i8, ptr %31, i64 16
  %366 = getelementptr inbounds nuw i8, ptr %31, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %31, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE4$_19E9_M_invokeERKSt9_Any_dataOS0_S7_Os", ptr %366, align 8, !tbaa !23
  store ptr @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE4$_19E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %365, align 8, !tbaa !20
  invoke fastcc void @_ZL19checkVectorFunctionIssEvSt8functionIFT_PT0_S3_S1_EES5_PKc(ptr noundef %30, ptr noundef %31, ptr noundef nonnull @.str.14)
          to label %367 unwind label %874

367:                                              ; preds = %362
  %368 = load ptr, ptr %365, align 8, !tbaa !20
  %369 = icmp eq ptr %368, null
  br i1 %369, label %375, label %370

370:                                              ; preds = %367
  %371 = invoke noundef i1 %368(ptr noundef nonnull align 8 dereferenceable(32) %31, ptr noundef nonnull align 8 dereferenceable(32) %31, i32 noundef 3)
          to label %375 unwind label %372

372:                                              ; preds = %370
  %373 = landingpad { ptr, i32 }
          catch ptr null
  %374 = extractvalue { ptr, i32 } %373, 0
  call void @__clang_call_terminate(ptr %374) #22
  unreachable

375:                                              ; preds = %367, %370
  %376 = load ptr, ptr %363, align 8, !tbaa !20
  %377 = icmp eq ptr %376, null
  br i1 %377, label %383, label %378

378:                                              ; preds = %375
  %379 = invoke noundef i1 %376(ptr noundef nonnull align 8 dereferenceable(32) %30, ptr noundef nonnull align 8 dereferenceable(32) %30, i32 noundef 3)
          to label %383 unwind label %380

380:                                              ; preds = %378
  %381 = landingpad { ptr, i32 }
          catch ptr null
  %382 = extractvalue { ptr, i32 } %381, 0
  call void @__clang_call_terminate(ptr %382) #22
  unreachable

383:                                              ; preds = %375, %378
  %384 = getelementptr inbounds nuw i8, ptr %32, i64 16
  %385 = getelementptr inbounds nuw i8, ptr %32, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %32, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE4$_20E9_M_invokeERKSt9_Any_dataOS0_S7_Oi", ptr %385, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE4$_20E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %384, align 8, !tbaa !20
  %386 = getelementptr inbounds nuw i8, ptr %33, i64 16
  %387 = getelementptr inbounds nuw i8, ptr %33, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %33, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE4$_21E9_M_invokeERKSt9_Any_dataOS0_S7_Oi", ptr %387, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE4$_21E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %386, align 8, !tbaa !20
  invoke fastcc void @_ZL19checkVectorFunctionIiiEvSt8functionIFT_PT0_S3_S1_EES5_PKc(ptr noundef %32, ptr noundef %33, ptr noundef nonnull @.str.15)
          to label %388 unwind label %891

388:                                              ; preds = %383
  %389 = load ptr, ptr %386, align 8, !tbaa !20
  %390 = icmp eq ptr %389, null
  br i1 %390, label %396, label %391

391:                                              ; preds = %388
  %392 = invoke noundef i1 %389(ptr noundef nonnull align 8 dereferenceable(32) %33, ptr noundef nonnull align 8 dereferenceable(32) %33, i32 noundef 3)
          to label %396 unwind label %393

393:                                              ; preds = %391
  %394 = landingpad { ptr, i32 }
          catch ptr null
  %395 = extractvalue { ptr, i32 } %394, 0
  call void @__clang_call_terminate(ptr %395) #22
  unreachable

396:                                              ; preds = %388, %391
  %397 = load ptr, ptr %384, align 8, !tbaa !20
  %398 = icmp eq ptr %397, null
  br i1 %398, label %404, label %399

399:                                              ; preds = %396
  %400 = invoke noundef i1 %397(ptr noundef nonnull align 8 dereferenceable(32) %32, ptr noundef nonnull align 8 dereferenceable(32) %32, i32 noundef 3)
          to label %404 unwind label %401

401:                                              ; preds = %399
  %402 = landingpad { ptr, i32 }
          catch ptr null
  %403 = extractvalue { ptr, i32 } %402, 0
  call void @__clang_call_terminate(ptr %403) #22
  unreachable

404:                                              ; preds = %396, %399
  %405 = getelementptr inbounds nuw i8, ptr %34, i64 16
  %406 = getelementptr inbounds nuw i8, ptr %34, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %34, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE4$_20E9_M_invokeERKSt9_Any_dataOS0_S7_Oi", ptr %406, align 8, !tbaa !21
  store ptr @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE4$_20E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %405, align 8, !tbaa !20
  %407 = getelementptr inbounds nuw i8, ptr %35, i64 16
  %408 = getelementptr inbounds nuw i8, ptr %35, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %35, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE4$_21E9_M_invokeERKSt9_Any_dataOS0_S7_Oi", ptr %408, align 8, !tbaa !21
  store ptr @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE4$_21E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %407, align 8, !tbaa !20
  invoke fastcc void @_ZL19checkVectorFunctionIifEvSt8functionIFT_PT0_S3_S1_EES5_PKc(ptr noundef %34, ptr noundef %35, ptr noundef nonnull @.str.16)
          to label %409 unwind label %908

409:                                              ; preds = %404
  %410 = load ptr, ptr %407, align 8, !tbaa !20
  %411 = icmp eq ptr %410, null
  br i1 %411, label %417, label %412

412:                                              ; preds = %409
  %413 = invoke noundef i1 %410(ptr noundef nonnull align 8 dereferenceable(32) %35, ptr noundef nonnull align 8 dereferenceable(32) %35, i32 noundef 3)
          to label %417 unwind label %414

414:                                              ; preds = %412
  %415 = landingpad { ptr, i32 }
          catch ptr null
  %416 = extractvalue { ptr, i32 } %415, 0
  call void @__clang_call_terminate(ptr %416) #22
  unreachable

417:                                              ; preds = %409, %412
  %418 = load ptr, ptr %405, align 8, !tbaa !20
  %419 = icmp eq ptr %418, null
  br i1 %419, label %425, label %420

420:                                              ; preds = %417
  %421 = invoke noundef i1 %418(ptr noundef nonnull align 8 dereferenceable(32) %34, ptr noundef nonnull align 8 dereferenceable(32) %34, i32 noundef 3)
          to label %425 unwind label %422

422:                                              ; preds = %420
  %423 = landingpad { ptr, i32 }
          catch ptr null
  %424 = extractvalue { ptr, i32 } %423, 0
  call void @__clang_call_terminate(ptr %424) #22
  unreachable

425:                                              ; preds = %417, %420
  %426 = getelementptr inbounds nuw i8, ptr %36, i64 16
  %427 = getelementptr inbounds nuw i8, ptr %36, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %36, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE4$_22E9_M_invokeERKSt9_Any_dataOS0_S7_Os", ptr %427, align 8, !tbaa !23
  store ptr @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE4$_22E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %426, align 8, !tbaa !20
  %428 = getelementptr inbounds nuw i8, ptr %37, i64 16
  %429 = getelementptr inbounds nuw i8, ptr %37, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %37, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE4$_23E9_M_invokeERKSt9_Any_dataOS0_S7_Os", ptr %429, align 8, !tbaa !23
  store ptr @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE4$_23E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %428, align 8, !tbaa !20
  invoke fastcc void @_ZL19checkVectorFunctionIssEvSt8functionIFT_PT0_S3_S1_EES5_PKc(ptr noundef %36, ptr noundef %37, ptr noundef nonnull @.str.17)
          to label %430 unwind label %925

430:                                              ; preds = %425
  %431 = load ptr, ptr %428, align 8, !tbaa !20
  %432 = icmp eq ptr %431, null
  br i1 %432, label %438, label %433

433:                                              ; preds = %430
  %434 = invoke noundef i1 %431(ptr noundef nonnull align 8 dereferenceable(32) %37, ptr noundef nonnull align 8 dereferenceable(32) %37, i32 noundef 3)
          to label %438 unwind label %435

435:                                              ; preds = %433
  %436 = landingpad { ptr, i32 }
          catch ptr null
  %437 = extractvalue { ptr, i32 } %436, 0
  call void @__clang_call_terminate(ptr %437) #22
  unreachable

438:                                              ; preds = %430, %433
  %439 = load ptr, ptr %426, align 8, !tbaa !20
  %440 = icmp eq ptr %439, null
  br i1 %440, label %446, label %441

441:                                              ; preds = %438
  %442 = invoke noundef i1 %439(ptr noundef nonnull align 8 dereferenceable(32) %36, ptr noundef nonnull align 8 dereferenceable(32) %36, i32 noundef 3)
          to label %446 unwind label %443

443:                                              ; preds = %441
  %444 = landingpad { ptr, i32 }
          catch ptr null
  %445 = extractvalue { ptr, i32 } %444, 0
  call void @__clang_call_terminate(ptr %445) #22
  unreachable

446:                                              ; preds = %438, %441
  %447 = getelementptr inbounds nuw i8, ptr %38, i64 16
  %448 = getelementptr inbounds nuw i8, ptr %38, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %38, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE4$_24E9_M_invokeERKSt9_Any_dataOS0_S7_Oi", ptr %448, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE4$_24E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %447, align 8, !tbaa !20
  %449 = getelementptr inbounds nuw i8, ptr %39, i64 16
  %450 = getelementptr inbounds nuw i8, ptr %39, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %39, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE4$_25E9_M_invokeERKSt9_Any_dataOS0_S7_Oi", ptr %450, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE4$_25E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %449, align 8, !tbaa !20
  invoke fastcc void @_ZL19checkVectorFunctionIiiEvSt8functionIFT_PT0_S3_S1_EES5_PKc(ptr noundef %38, ptr noundef %39, ptr noundef nonnull @.str.18)
          to label %451 unwind label %942

451:                                              ; preds = %446
  %452 = load ptr, ptr %449, align 8, !tbaa !20
  %453 = icmp eq ptr %452, null
  br i1 %453, label %459, label %454

454:                                              ; preds = %451
  %455 = invoke noundef i1 %452(ptr noundef nonnull align 8 dereferenceable(32) %39, ptr noundef nonnull align 8 dereferenceable(32) %39, i32 noundef 3)
          to label %459 unwind label %456

456:                                              ; preds = %454
  %457 = landingpad { ptr, i32 }
          catch ptr null
  %458 = extractvalue { ptr, i32 } %457, 0
  call void @__clang_call_terminate(ptr %458) #22
  unreachable

459:                                              ; preds = %451, %454
  %460 = load ptr, ptr %447, align 8, !tbaa !20
  %461 = icmp eq ptr %460, null
  br i1 %461, label %467, label %462

462:                                              ; preds = %459
  %463 = invoke noundef i1 %460(ptr noundef nonnull align 8 dereferenceable(32) %38, ptr noundef nonnull align 8 dereferenceable(32) %38, i32 noundef 3)
          to label %467 unwind label %464

464:                                              ; preds = %462
  %465 = landingpad { ptr, i32 }
          catch ptr null
  %466 = extractvalue { ptr, i32 } %465, 0
  call void @__clang_call_terminate(ptr %466) #22
  unreachable

467:                                              ; preds = %459, %462
  %468 = getelementptr inbounds nuw i8, ptr %40, i64 16
  %469 = getelementptr inbounds nuw i8, ptr %40, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %40, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE4$_24E9_M_invokeERKSt9_Any_dataOS0_S7_Oi", ptr %469, align 8, !tbaa !21
  store ptr @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE4$_24E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %468, align 8, !tbaa !20
  %470 = getelementptr inbounds nuw i8, ptr %41, i64 16
  %471 = getelementptr inbounds nuw i8, ptr %41, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %41, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE4$_25E9_M_invokeERKSt9_Any_dataOS0_S7_Oi", ptr %471, align 8, !tbaa !21
  store ptr @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE4$_25E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %470, align 8, !tbaa !20
  invoke fastcc void @_ZL19checkVectorFunctionIifEvSt8functionIFT_PT0_S3_S1_EES5_PKc(ptr noundef %40, ptr noundef %41, ptr noundef nonnull @.str.19)
          to label %472 unwind label %959

472:                                              ; preds = %467
  %473 = load ptr, ptr %470, align 8, !tbaa !20
  %474 = icmp eq ptr %473, null
  br i1 %474, label %480, label %475

475:                                              ; preds = %472
  %476 = invoke noundef i1 %473(ptr noundef nonnull align 8 dereferenceable(32) %41, ptr noundef nonnull align 8 dereferenceable(32) %41, i32 noundef 3)
          to label %480 unwind label %477

477:                                              ; preds = %475
  %478 = landingpad { ptr, i32 }
          catch ptr null
  %479 = extractvalue { ptr, i32 } %478, 0
  call void @__clang_call_terminate(ptr %479) #22
  unreachable

480:                                              ; preds = %472, %475
  %481 = load ptr, ptr %468, align 8, !tbaa !20
  %482 = icmp eq ptr %481, null
  br i1 %482, label %488, label %483

483:                                              ; preds = %480
  %484 = invoke noundef i1 %481(ptr noundef nonnull align 8 dereferenceable(32) %40, ptr noundef nonnull align 8 dereferenceable(32) %40, i32 noundef 3)
          to label %488 unwind label %485

485:                                              ; preds = %483
  %486 = landingpad { ptr, i32 }
          catch ptr null
  %487 = extractvalue { ptr, i32 } %486, 0
  call void @__clang_call_terminate(ptr %487) #22
  unreachable

488:                                              ; preds = %480, %483
  %489 = getelementptr inbounds nuw i8, ptr %42, i64 16
  %490 = getelementptr inbounds nuw i8, ptr %42, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %42, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE4$_26E9_M_invokeERKSt9_Any_dataOS0_S7_Os", ptr %490, align 8, !tbaa !23
  store ptr @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE4$_26E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %489, align 8, !tbaa !20
  %491 = getelementptr inbounds nuw i8, ptr %43, i64 16
  %492 = getelementptr inbounds nuw i8, ptr %43, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %43, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE4$_27E9_M_invokeERKSt9_Any_dataOS0_S7_Os", ptr %492, align 8, !tbaa !23
  store ptr @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE4$_27E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %491, align 8, !tbaa !20
  invoke fastcc void @_ZL19checkVectorFunctionIssEvSt8functionIFT_PT0_S3_S1_EES5_PKc(ptr noundef %42, ptr noundef %43, ptr noundef nonnull @.str.20)
          to label %493 unwind label %976

493:                                              ; preds = %488
  %494 = load ptr, ptr %491, align 8, !tbaa !20
  %495 = icmp eq ptr %494, null
  br i1 %495, label %501, label %496

496:                                              ; preds = %493
  %497 = invoke noundef i1 %494(ptr noundef nonnull align 8 dereferenceable(32) %43, ptr noundef nonnull align 8 dereferenceable(32) %43, i32 noundef 3)
          to label %501 unwind label %498

498:                                              ; preds = %496
  %499 = landingpad { ptr, i32 }
          catch ptr null
  %500 = extractvalue { ptr, i32 } %499, 0
  call void @__clang_call_terminate(ptr %500) #22
  unreachable

501:                                              ; preds = %493, %496
  %502 = load ptr, ptr %489, align 8, !tbaa !20
  %503 = icmp eq ptr %502, null
  br i1 %503, label %509, label %504

504:                                              ; preds = %501
  %505 = invoke noundef i1 %502(ptr noundef nonnull align 8 dereferenceable(32) %42, ptr noundef nonnull align 8 dereferenceable(32) %42, i32 noundef 3)
          to label %509 unwind label %506

506:                                              ; preds = %504
  %507 = landingpad { ptr, i32 }
          catch ptr null
  %508 = extractvalue { ptr, i32 } %507, 0
  call void @__clang_call_terminate(ptr %508) #22
  unreachable

509:                                              ; preds = %501, %504
  %510 = getelementptr inbounds nuw i8, ptr %44, i64 16
  %511 = getelementptr inbounds nuw i8, ptr %44, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %44, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE4$_28E9_M_invokeERKSt9_Any_dataOS0_S7_Oi", ptr %511, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE4$_28E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %510, align 8, !tbaa !20
  %512 = getelementptr inbounds nuw i8, ptr %45, i64 16
  %513 = getelementptr inbounds nuw i8, ptr %45, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %45, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE4$_29E9_M_invokeERKSt9_Any_dataOS0_S7_Oi", ptr %513, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE4$_29E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %512, align 8, !tbaa !20
  invoke fastcc void @_ZL19checkVectorFunctionIiiEvSt8functionIFT_PT0_S3_S1_EES5_PKc(ptr noundef %44, ptr noundef %45, ptr noundef nonnull @.str.21)
          to label %514 unwind label %993

514:                                              ; preds = %509
  %515 = load ptr, ptr %512, align 8, !tbaa !20
  %516 = icmp eq ptr %515, null
  br i1 %516, label %522, label %517

517:                                              ; preds = %514
  %518 = invoke noundef i1 %515(ptr noundef nonnull align 8 dereferenceable(32) %45, ptr noundef nonnull align 8 dereferenceable(32) %45, i32 noundef 3)
          to label %522 unwind label %519

519:                                              ; preds = %517
  %520 = landingpad { ptr, i32 }
          catch ptr null
  %521 = extractvalue { ptr, i32 } %520, 0
  call void @__clang_call_terminate(ptr %521) #22
  unreachable

522:                                              ; preds = %514, %517
  %523 = load ptr, ptr %510, align 8, !tbaa !20
  %524 = icmp eq ptr %523, null
  br i1 %524, label %530, label %525

525:                                              ; preds = %522
  %526 = invoke noundef i1 %523(ptr noundef nonnull align 8 dereferenceable(32) %44, ptr noundef nonnull align 8 dereferenceable(32) %44, i32 noundef 3)
          to label %530 unwind label %527

527:                                              ; preds = %525
  %528 = landingpad { ptr, i32 }
          catch ptr null
  %529 = extractvalue { ptr, i32 } %528, 0
  call void @__clang_call_terminate(ptr %529) #22
  unreachable

530:                                              ; preds = %522, %525
  %531 = getelementptr inbounds nuw i8, ptr %46, i64 16
  %532 = getelementptr inbounds nuw i8, ptr %46, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %46, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE4$_28E9_M_invokeERKSt9_Any_dataOS0_S7_Oi", ptr %532, align 8, !tbaa !21
  store ptr @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE4$_28E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %531, align 8, !tbaa !20
  %533 = getelementptr inbounds nuw i8, ptr %47, i64 16
  %534 = getelementptr inbounds nuw i8, ptr %47, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %47, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE4$_29E9_M_invokeERKSt9_Any_dataOS0_S7_Oi", ptr %534, align 8, !tbaa !21
  store ptr @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE4$_29E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %533, align 8, !tbaa !20
  invoke fastcc void @_ZL19checkVectorFunctionIifEvSt8functionIFT_PT0_S3_S1_EES5_PKc(ptr noundef %46, ptr noundef %47, ptr noundef nonnull @.str.22)
          to label %535 unwind label %1010

535:                                              ; preds = %530
  %536 = load ptr, ptr %533, align 8, !tbaa !20
  %537 = icmp eq ptr %536, null
  br i1 %537, label %543, label %538

538:                                              ; preds = %535
  %539 = invoke noundef i1 %536(ptr noundef nonnull align 8 dereferenceable(32) %47, ptr noundef nonnull align 8 dereferenceable(32) %47, i32 noundef 3)
          to label %543 unwind label %540

540:                                              ; preds = %538
  %541 = landingpad { ptr, i32 }
          catch ptr null
  %542 = extractvalue { ptr, i32 } %541, 0
  call void @__clang_call_terminate(ptr %542) #22
  unreachable

543:                                              ; preds = %535, %538
  %544 = load ptr, ptr %531, align 8, !tbaa !20
  %545 = icmp eq ptr %544, null
  br i1 %545, label %551, label %546

546:                                              ; preds = %543
  %547 = invoke noundef i1 %544(ptr noundef nonnull align 8 dereferenceable(32) %46, ptr noundef nonnull align 8 dereferenceable(32) %46, i32 noundef 3)
          to label %551 unwind label %548

548:                                              ; preds = %546
  %549 = landingpad { ptr, i32 }
          catch ptr null
  %550 = extractvalue { ptr, i32 } %549, 0
  call void @__clang_call_terminate(ptr %550) #22
  unreachable

551:                                              ; preds = %543, %546
  %552 = getelementptr inbounds nuw i8, ptr %48, i64 16
  %553 = getelementptr inbounds nuw i8, ptr %48, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %48, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE4$_30E9_M_invokeERKSt9_Any_dataOS0_S7_Os", ptr %553, align 8, !tbaa !23
  store ptr @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE4$_30E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %552, align 8, !tbaa !20
  %554 = getelementptr inbounds nuw i8, ptr %49, i64 16
  %555 = getelementptr inbounds nuw i8, ptr %49, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %49, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE4$_31E9_M_invokeERKSt9_Any_dataOS0_S7_Os", ptr %555, align 8, !tbaa !23
  store ptr @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE4$_31E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %554, align 8, !tbaa !20
  invoke fastcc void @_ZL19checkVectorFunctionIssEvSt8functionIFT_PT0_S3_S1_EES5_PKc(ptr noundef %48, ptr noundef %49, ptr noundef nonnull @.str.23)
          to label %556 unwind label %1027

556:                                              ; preds = %551
  %557 = load ptr, ptr %554, align 8, !tbaa !20
  %558 = icmp eq ptr %557, null
  br i1 %558, label %564, label %559

559:                                              ; preds = %556
  %560 = invoke noundef i1 %557(ptr noundef nonnull align 8 dereferenceable(32) %49, ptr noundef nonnull align 8 dereferenceable(32) %49, i32 noundef 3)
          to label %564 unwind label %561

561:                                              ; preds = %559
  %562 = landingpad { ptr, i32 }
          catch ptr null
  %563 = extractvalue { ptr, i32 } %562, 0
  call void @__clang_call_terminate(ptr %563) #22
  unreachable

564:                                              ; preds = %556, %559
  %565 = load ptr, ptr %552, align 8, !tbaa !20
  %566 = icmp eq ptr %565, null
  br i1 %566, label %572, label %567

567:                                              ; preds = %564
  %568 = invoke noundef i1 %565(ptr noundef nonnull align 8 dereferenceable(32) %48, ptr noundef nonnull align 8 dereferenceable(32) %48, i32 noundef 3)
          to label %572 unwind label %569

569:                                              ; preds = %567
  %570 = landingpad { ptr, i32 }
          catch ptr null
  %571 = extractvalue { ptr, i32 } %570, 0
  call void @__clang_call_terminate(ptr %571) #22
  unreachable

572:                                              ; preds = %564, %567
  %573 = getelementptr inbounds nuw i8, ptr %50, i64 16
  %574 = getelementptr inbounds nuw i8, ptr %50, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %50, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE4$_32E9_M_invokeERKSt9_Any_dataOS0_S7_Oi", ptr %574, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE4$_32E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %573, align 8, !tbaa !20
  %575 = getelementptr inbounds nuw i8, ptr %51, i64 16
  %576 = getelementptr inbounds nuw i8, ptr %51, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %51, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE4$_33E9_M_invokeERKSt9_Any_dataOS0_S7_Oi", ptr %576, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE4$_33E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %575, align 8, !tbaa !20
  invoke fastcc void @_ZL19checkVectorFunctionIiiEvSt8functionIFT_PT0_S3_S1_EES5_PKc(ptr noundef %50, ptr noundef %51, ptr noundef nonnull @.str.24)
          to label %577 unwind label %1044

577:                                              ; preds = %572
  %578 = load ptr, ptr %575, align 8, !tbaa !20
  %579 = icmp eq ptr %578, null
  br i1 %579, label %585, label %580

580:                                              ; preds = %577
  %581 = invoke noundef i1 %578(ptr noundef nonnull align 8 dereferenceable(32) %51, ptr noundef nonnull align 8 dereferenceable(32) %51, i32 noundef 3)
          to label %585 unwind label %582

582:                                              ; preds = %580
  %583 = landingpad { ptr, i32 }
          catch ptr null
  %584 = extractvalue { ptr, i32 } %583, 0
  call void @__clang_call_terminate(ptr %584) #22
  unreachable

585:                                              ; preds = %577, %580
  %586 = load ptr, ptr %573, align 8, !tbaa !20
  %587 = icmp eq ptr %586, null
  br i1 %587, label %593, label %588

588:                                              ; preds = %585
  %589 = invoke noundef i1 %586(ptr noundef nonnull align 8 dereferenceable(32) %50, ptr noundef nonnull align 8 dereferenceable(32) %50, i32 noundef 3)
          to label %593 unwind label %590

590:                                              ; preds = %588
  %591 = landingpad { ptr, i32 }
          catch ptr null
  %592 = extractvalue { ptr, i32 } %591, 0
  call void @__clang_call_terminate(ptr %592) #22
  unreachable

593:                                              ; preds = %585, %588
  %594 = getelementptr inbounds nuw i8, ptr %52, i64 16
  %595 = getelementptr inbounds nuw i8, ptr %52, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %52, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE4$_32E9_M_invokeERKSt9_Any_dataOS0_S7_Oi", ptr %595, align 8, !tbaa !21
  store ptr @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE4$_32E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %594, align 8, !tbaa !20
  %596 = getelementptr inbounds nuw i8, ptr %53, i64 16
  %597 = getelementptr inbounds nuw i8, ptr %53, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %53, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE4$_33E9_M_invokeERKSt9_Any_dataOS0_S7_Oi", ptr %597, align 8, !tbaa !21
  store ptr @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE4$_33E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %596, align 8, !tbaa !20
  invoke fastcc void @_ZL19checkVectorFunctionIifEvSt8functionIFT_PT0_S3_S1_EES5_PKc(ptr noundef %52, ptr noundef %53, ptr noundef nonnull @.str.25)
          to label %598 unwind label %1061

598:                                              ; preds = %593
  %599 = load ptr, ptr %596, align 8, !tbaa !20
  %600 = icmp eq ptr %599, null
  br i1 %600, label %606, label %601

601:                                              ; preds = %598
  %602 = invoke noundef i1 %599(ptr noundef nonnull align 8 dereferenceable(32) %53, ptr noundef nonnull align 8 dereferenceable(32) %53, i32 noundef 3)
          to label %606 unwind label %603

603:                                              ; preds = %601
  %604 = landingpad { ptr, i32 }
          catch ptr null
  %605 = extractvalue { ptr, i32 } %604, 0
  call void @__clang_call_terminate(ptr %605) #22
  unreachable

606:                                              ; preds = %598, %601
  %607 = load ptr, ptr %594, align 8, !tbaa !20
  %608 = icmp eq ptr %607, null
  br i1 %608, label %614, label %609

609:                                              ; preds = %606
  %610 = invoke noundef i1 %607(ptr noundef nonnull align 8 dereferenceable(32) %52, ptr noundef nonnull align 8 dereferenceable(32) %52, i32 noundef 3)
          to label %614 unwind label %611

611:                                              ; preds = %609
  %612 = landingpad { ptr, i32 }
          catch ptr null
  %613 = extractvalue { ptr, i32 } %612, 0
  call void @__clang_call_terminate(ptr %613) #22
  unreachable

614:                                              ; preds = %606, %609
  %615 = getelementptr inbounds nuw i8, ptr %54, i64 16
  %616 = getelementptr inbounds nuw i8, ptr %54, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %54, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE4$_34E9_M_invokeERKSt9_Any_dataOS0_S7_Os", ptr %616, align 8, !tbaa !23
  store ptr @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE4$_34E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %615, align 8, !tbaa !20
  %617 = getelementptr inbounds nuw i8, ptr %55, i64 16
  %618 = getelementptr inbounds nuw i8, ptr %55, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %55, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE4$_35E9_M_invokeERKSt9_Any_dataOS0_S7_Os", ptr %618, align 8, !tbaa !23
  store ptr @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE4$_35E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %617, align 8, !tbaa !20
  invoke fastcc void @_ZL19checkVectorFunctionIssEvSt8functionIFT_PT0_S3_S1_EES5_PKc(ptr noundef %54, ptr noundef %55, ptr noundef nonnull @.str.26)
          to label %619 unwind label %1078

619:                                              ; preds = %614
  %620 = load ptr, ptr %617, align 8, !tbaa !20
  %621 = icmp eq ptr %620, null
  br i1 %621, label %627, label %622

622:                                              ; preds = %619
  %623 = invoke noundef i1 %620(ptr noundef nonnull align 8 dereferenceable(32) %55, ptr noundef nonnull align 8 dereferenceable(32) %55, i32 noundef 3)
          to label %627 unwind label %624

624:                                              ; preds = %622
  %625 = landingpad { ptr, i32 }
          catch ptr null
  %626 = extractvalue { ptr, i32 } %625, 0
  call void @__clang_call_terminate(ptr %626) #22
  unreachable

627:                                              ; preds = %619, %622
  %628 = load ptr, ptr %615, align 8, !tbaa !20
  %629 = icmp eq ptr %628, null
  br i1 %629, label %635, label %630

630:                                              ; preds = %627
  %631 = invoke noundef i1 %628(ptr noundef nonnull align 8 dereferenceable(32) %54, ptr noundef nonnull align 8 dereferenceable(32) %54, i32 noundef 3)
          to label %635 unwind label %632

632:                                              ; preds = %630
  %633 = landingpad { ptr, i32 }
          catch ptr null
  %634 = extractvalue { ptr, i32 } %633, 0
  call void @__clang_call_terminate(ptr %634) #22
  unreachable

635:                                              ; preds = %627, %630
  ret i32 0

636:                                              ; preds = %67
  %637 = landingpad { ptr, i32 }
          cleanup
  %638 = load ptr, ptr %71, align 8, !tbaa !20
  %639 = icmp eq ptr %638, null
  br i1 %639, label %645, label %640

640:                                              ; preds = %636
  %641 = invoke noundef i1 %638(ptr noundef nonnull align 8 dereferenceable(32) %3, ptr noundef nonnull align 8 dereferenceable(32) %3, i32 noundef 3)
          to label %645 unwind label %642

642:                                              ; preds = %640
  %643 = landingpad { ptr, i32 }
          catch ptr null
  %644 = extractvalue { ptr, i32 } %643, 0
  call void @__clang_call_terminate(ptr %644) #22
  unreachable

645:                                              ; preds = %636, %640
  %646 = load ptr, ptr %69, align 8, !tbaa !20
  %647 = icmp eq ptr %646, null
  br i1 %647, label %1095, label %648

648:                                              ; preds = %645
  %649 = invoke noundef i1 %646(ptr noundef nonnull align 8 dereferenceable(32) %2, ptr noundef nonnull align 8 dereferenceable(32) %2, i32 noundef 3)
          to label %1095 unwind label %650

650:                                              ; preds = %648
  %651 = landingpad { ptr, i32 }
          catch ptr null
  %652 = extractvalue { ptr, i32 } %651, 0
  call void @__clang_call_terminate(ptr %652) #22
  unreachable

653:                                              ; preds = %89
  %654 = landingpad { ptr, i32 }
          cleanup
  %655 = load ptr, ptr %92, align 8, !tbaa !20
  %656 = icmp eq ptr %655, null
  br i1 %656, label %662, label %657

657:                                              ; preds = %653
  %658 = invoke noundef i1 %655(ptr noundef nonnull align 8 dereferenceable(32) %5, ptr noundef nonnull align 8 dereferenceable(32) %5, i32 noundef 3)
          to label %662 unwind label %659

659:                                              ; preds = %657
  %660 = landingpad { ptr, i32 }
          catch ptr null
  %661 = extractvalue { ptr, i32 } %660, 0
  call void @__clang_call_terminate(ptr %661) #22
  unreachable

662:                                              ; preds = %653, %657
  %663 = load ptr, ptr %90, align 8, !tbaa !20
  %664 = icmp eq ptr %663, null
  br i1 %664, label %1095, label %665

665:                                              ; preds = %662
  %666 = invoke noundef i1 %663(ptr noundef nonnull align 8 dereferenceable(32) %4, ptr noundef nonnull align 8 dereferenceable(32) %4, i32 noundef 3)
          to label %1095 unwind label %667

667:                                              ; preds = %665
  %668 = landingpad { ptr, i32 }
          catch ptr null
  %669 = extractvalue { ptr, i32 } %668, 0
  call void @__clang_call_terminate(ptr %669) #22
  unreachable

670:                                              ; preds = %110
  %671 = landingpad { ptr, i32 }
          cleanup
  %672 = load ptr, ptr %113, align 8, !tbaa !20
  %673 = icmp eq ptr %672, null
  br i1 %673, label %679, label %674

674:                                              ; preds = %670
  %675 = invoke noundef i1 %672(ptr noundef nonnull align 8 dereferenceable(32) %7, ptr noundef nonnull align 8 dereferenceable(32) %7, i32 noundef 3)
          to label %679 unwind label %676

676:                                              ; preds = %674
  %677 = landingpad { ptr, i32 }
          catch ptr null
  %678 = extractvalue { ptr, i32 } %677, 0
  call void @__clang_call_terminate(ptr %678) #22
  unreachable

679:                                              ; preds = %670, %674
  %680 = load ptr, ptr %111, align 8, !tbaa !20
  %681 = icmp eq ptr %680, null
  br i1 %681, label %1095, label %682

682:                                              ; preds = %679
  %683 = invoke noundef i1 %680(ptr noundef nonnull align 8 dereferenceable(32) %6, ptr noundef nonnull align 8 dereferenceable(32) %6, i32 noundef 3)
          to label %1095 unwind label %684

684:                                              ; preds = %682
  %685 = landingpad { ptr, i32 }
          catch ptr null
  %686 = extractvalue { ptr, i32 } %685, 0
  call void @__clang_call_terminate(ptr %686) #22
  unreachable

687:                                              ; preds = %131
  %688 = landingpad { ptr, i32 }
          cleanup
  %689 = load ptr, ptr %134, align 8, !tbaa !20
  %690 = icmp eq ptr %689, null
  br i1 %690, label %696, label %691

691:                                              ; preds = %687
  %692 = invoke noundef i1 %689(ptr noundef nonnull align 8 dereferenceable(32) %9, ptr noundef nonnull align 8 dereferenceable(32) %9, i32 noundef 3)
          to label %696 unwind label %693

693:                                              ; preds = %691
  %694 = landingpad { ptr, i32 }
          catch ptr null
  %695 = extractvalue { ptr, i32 } %694, 0
  call void @__clang_call_terminate(ptr %695) #22
  unreachable

696:                                              ; preds = %687, %691
  %697 = load ptr, ptr %132, align 8, !tbaa !20
  %698 = icmp eq ptr %697, null
  br i1 %698, label %1095, label %699

699:                                              ; preds = %696
  %700 = invoke noundef i1 %697(ptr noundef nonnull align 8 dereferenceable(32) %8, ptr noundef nonnull align 8 dereferenceable(32) %8, i32 noundef 3)
          to label %1095 unwind label %701

701:                                              ; preds = %699
  %702 = landingpad { ptr, i32 }
          catch ptr null
  %703 = extractvalue { ptr, i32 } %702, 0
  call void @__clang_call_terminate(ptr %703) #22
  unreachable

704:                                              ; preds = %152
  %705 = landingpad { ptr, i32 }
          cleanup
  %706 = load ptr, ptr %155, align 8, !tbaa !20
  %707 = icmp eq ptr %706, null
  br i1 %707, label %713, label %708

708:                                              ; preds = %704
  %709 = invoke noundef i1 %706(ptr noundef nonnull align 8 dereferenceable(32) %11, ptr noundef nonnull align 8 dereferenceable(32) %11, i32 noundef 3)
          to label %713 unwind label %710

710:                                              ; preds = %708
  %711 = landingpad { ptr, i32 }
          catch ptr null
  %712 = extractvalue { ptr, i32 } %711, 0
  call void @__clang_call_terminate(ptr %712) #22
  unreachable

713:                                              ; preds = %704, %708
  %714 = load ptr, ptr %153, align 8, !tbaa !20
  %715 = icmp eq ptr %714, null
  br i1 %715, label %1095, label %716

716:                                              ; preds = %713
  %717 = invoke noundef i1 %714(ptr noundef nonnull align 8 dereferenceable(32) %10, ptr noundef nonnull align 8 dereferenceable(32) %10, i32 noundef 3)
          to label %1095 unwind label %718

718:                                              ; preds = %716
  %719 = landingpad { ptr, i32 }
          catch ptr null
  %720 = extractvalue { ptr, i32 } %719, 0
  call void @__clang_call_terminate(ptr %720) #22
  unreachable

721:                                              ; preds = %173
  %722 = landingpad { ptr, i32 }
          cleanup
  %723 = load ptr, ptr %176, align 8, !tbaa !20
  %724 = icmp eq ptr %723, null
  br i1 %724, label %730, label %725

725:                                              ; preds = %721
  %726 = invoke noundef i1 %723(ptr noundef nonnull align 8 dereferenceable(32) %13, ptr noundef nonnull align 8 dereferenceable(32) %13, i32 noundef 3)
          to label %730 unwind label %727

727:                                              ; preds = %725
  %728 = landingpad { ptr, i32 }
          catch ptr null
  %729 = extractvalue { ptr, i32 } %728, 0
  call void @__clang_call_terminate(ptr %729) #22
  unreachable

730:                                              ; preds = %721, %725
  %731 = load ptr, ptr %174, align 8, !tbaa !20
  %732 = icmp eq ptr %731, null
  br i1 %732, label %1095, label %733

733:                                              ; preds = %730
  %734 = invoke noundef i1 %731(ptr noundef nonnull align 8 dereferenceable(32) %12, ptr noundef nonnull align 8 dereferenceable(32) %12, i32 noundef 3)
          to label %1095 unwind label %735

735:                                              ; preds = %733
  %736 = landingpad { ptr, i32 }
          catch ptr null
  %737 = extractvalue { ptr, i32 } %736, 0
  call void @__clang_call_terminate(ptr %737) #22
  unreachable

738:                                              ; preds = %194
  %739 = landingpad { ptr, i32 }
          cleanup
  %740 = load ptr, ptr %197, align 8, !tbaa !20
  %741 = icmp eq ptr %740, null
  br i1 %741, label %747, label %742

742:                                              ; preds = %738
  %743 = invoke noundef i1 %740(ptr noundef nonnull align 8 dereferenceable(32) %15, ptr noundef nonnull align 8 dereferenceable(32) %15, i32 noundef 3)
          to label %747 unwind label %744

744:                                              ; preds = %742
  %745 = landingpad { ptr, i32 }
          catch ptr null
  %746 = extractvalue { ptr, i32 } %745, 0
  call void @__clang_call_terminate(ptr %746) #22
  unreachable

747:                                              ; preds = %738, %742
  %748 = load ptr, ptr %195, align 8, !tbaa !20
  %749 = icmp eq ptr %748, null
  br i1 %749, label %1095, label %750

750:                                              ; preds = %747
  %751 = invoke noundef i1 %748(ptr noundef nonnull align 8 dereferenceable(32) %14, ptr noundef nonnull align 8 dereferenceable(32) %14, i32 noundef 3)
          to label %1095 unwind label %752

752:                                              ; preds = %750
  %753 = landingpad { ptr, i32 }
          catch ptr null
  %754 = extractvalue { ptr, i32 } %753, 0
  call void @__clang_call_terminate(ptr %754) #22
  unreachable

755:                                              ; preds = %215
  %756 = landingpad { ptr, i32 }
          cleanup
  %757 = load ptr, ptr %218, align 8, !tbaa !20
  %758 = icmp eq ptr %757, null
  br i1 %758, label %764, label %759

759:                                              ; preds = %755
  %760 = invoke noundef i1 %757(ptr noundef nonnull align 8 dereferenceable(32) %17, ptr noundef nonnull align 8 dereferenceable(32) %17, i32 noundef 3)
          to label %764 unwind label %761

761:                                              ; preds = %759
  %762 = landingpad { ptr, i32 }
          catch ptr null
  %763 = extractvalue { ptr, i32 } %762, 0
  call void @__clang_call_terminate(ptr %763) #22
  unreachable

764:                                              ; preds = %755, %759
  %765 = load ptr, ptr %216, align 8, !tbaa !20
  %766 = icmp eq ptr %765, null
  br i1 %766, label %1095, label %767

767:                                              ; preds = %764
  %768 = invoke noundef i1 %765(ptr noundef nonnull align 8 dereferenceable(32) %16, ptr noundef nonnull align 8 dereferenceable(32) %16, i32 noundef 3)
          to label %1095 unwind label %769

769:                                              ; preds = %767
  %770 = landingpad { ptr, i32 }
          catch ptr null
  %771 = extractvalue { ptr, i32 } %770, 0
  call void @__clang_call_terminate(ptr %771) #22
  unreachable

772:                                              ; preds = %236
  %773 = landingpad { ptr, i32 }
          cleanup
  %774 = load ptr, ptr %239, align 8, !tbaa !20
  %775 = icmp eq ptr %774, null
  br i1 %775, label %781, label %776

776:                                              ; preds = %772
  %777 = invoke noundef i1 %774(ptr noundef nonnull align 8 dereferenceable(32) %19, ptr noundef nonnull align 8 dereferenceable(32) %19, i32 noundef 3)
          to label %781 unwind label %778

778:                                              ; preds = %776
  %779 = landingpad { ptr, i32 }
          catch ptr null
  %780 = extractvalue { ptr, i32 } %779, 0
  call void @__clang_call_terminate(ptr %780) #22
  unreachable

781:                                              ; preds = %772, %776
  %782 = load ptr, ptr %237, align 8, !tbaa !20
  %783 = icmp eq ptr %782, null
  br i1 %783, label %1095, label %784

784:                                              ; preds = %781
  %785 = invoke noundef i1 %782(ptr noundef nonnull align 8 dereferenceable(32) %18, ptr noundef nonnull align 8 dereferenceable(32) %18, i32 noundef 3)
          to label %1095 unwind label %786

786:                                              ; preds = %784
  %787 = landingpad { ptr, i32 }
          catch ptr null
  %788 = extractvalue { ptr, i32 } %787, 0
  call void @__clang_call_terminate(ptr %788) #22
  unreachable

789:                                              ; preds = %257
  %790 = landingpad { ptr, i32 }
          cleanup
  %791 = load ptr, ptr %260, align 8, !tbaa !20
  %792 = icmp eq ptr %791, null
  br i1 %792, label %798, label %793

793:                                              ; preds = %789
  %794 = invoke noundef i1 %791(ptr noundef nonnull align 8 dereferenceable(32) %21, ptr noundef nonnull align 8 dereferenceable(32) %21, i32 noundef 3)
          to label %798 unwind label %795

795:                                              ; preds = %793
  %796 = landingpad { ptr, i32 }
          catch ptr null
  %797 = extractvalue { ptr, i32 } %796, 0
  call void @__clang_call_terminate(ptr %797) #22
  unreachable

798:                                              ; preds = %789, %793
  %799 = load ptr, ptr %258, align 8, !tbaa !20
  %800 = icmp eq ptr %799, null
  br i1 %800, label %1095, label %801

801:                                              ; preds = %798
  %802 = invoke noundef i1 %799(ptr noundef nonnull align 8 dereferenceable(32) %20, ptr noundef nonnull align 8 dereferenceable(32) %20, i32 noundef 3)
          to label %1095 unwind label %803

803:                                              ; preds = %801
  %804 = landingpad { ptr, i32 }
          catch ptr null
  %805 = extractvalue { ptr, i32 } %804, 0
  call void @__clang_call_terminate(ptr %805) #22
  unreachable

806:                                              ; preds = %278
  %807 = landingpad { ptr, i32 }
          cleanup
  %808 = load ptr, ptr %281, align 8, !tbaa !20
  %809 = icmp eq ptr %808, null
  br i1 %809, label %815, label %810

810:                                              ; preds = %806
  %811 = invoke noundef i1 %808(ptr noundef nonnull align 8 dereferenceable(32) %23, ptr noundef nonnull align 8 dereferenceable(32) %23, i32 noundef 3)
          to label %815 unwind label %812

812:                                              ; preds = %810
  %813 = landingpad { ptr, i32 }
          catch ptr null
  %814 = extractvalue { ptr, i32 } %813, 0
  call void @__clang_call_terminate(ptr %814) #22
  unreachable

815:                                              ; preds = %806, %810
  %816 = load ptr, ptr %279, align 8, !tbaa !20
  %817 = icmp eq ptr %816, null
  br i1 %817, label %1095, label %818

818:                                              ; preds = %815
  %819 = invoke noundef i1 %816(ptr noundef nonnull align 8 dereferenceable(32) %22, ptr noundef nonnull align 8 dereferenceable(32) %22, i32 noundef 3)
          to label %1095 unwind label %820

820:                                              ; preds = %818
  %821 = landingpad { ptr, i32 }
          catch ptr null
  %822 = extractvalue { ptr, i32 } %821, 0
  call void @__clang_call_terminate(ptr %822) #22
  unreachable

823:                                              ; preds = %299
  %824 = landingpad { ptr, i32 }
          cleanup
  %825 = load ptr, ptr %302, align 8, !tbaa !20
  %826 = icmp eq ptr %825, null
  br i1 %826, label %832, label %827

827:                                              ; preds = %823
  %828 = invoke noundef i1 %825(ptr noundef nonnull align 8 dereferenceable(32) %25, ptr noundef nonnull align 8 dereferenceable(32) %25, i32 noundef 3)
          to label %832 unwind label %829

829:                                              ; preds = %827
  %830 = landingpad { ptr, i32 }
          catch ptr null
  %831 = extractvalue { ptr, i32 } %830, 0
  call void @__clang_call_terminate(ptr %831) #22
  unreachable

832:                                              ; preds = %823, %827
  %833 = load ptr, ptr %300, align 8, !tbaa !20
  %834 = icmp eq ptr %833, null
  br i1 %834, label %1095, label %835

835:                                              ; preds = %832
  %836 = invoke noundef i1 %833(ptr noundef nonnull align 8 dereferenceable(32) %24, ptr noundef nonnull align 8 dereferenceable(32) %24, i32 noundef 3)
          to label %1095 unwind label %837

837:                                              ; preds = %835
  %838 = landingpad { ptr, i32 }
          catch ptr null
  %839 = extractvalue { ptr, i32 } %838, 0
  call void @__clang_call_terminate(ptr %839) #22
  unreachable

840:                                              ; preds = %320
  %841 = landingpad { ptr, i32 }
          cleanup
  %842 = load ptr, ptr %323, align 8, !tbaa !20
  %843 = icmp eq ptr %842, null
  br i1 %843, label %849, label %844

844:                                              ; preds = %840
  %845 = invoke noundef i1 %842(ptr noundef nonnull align 8 dereferenceable(32) %27, ptr noundef nonnull align 8 dereferenceable(32) %27, i32 noundef 3)
          to label %849 unwind label %846

846:                                              ; preds = %844
  %847 = landingpad { ptr, i32 }
          catch ptr null
  %848 = extractvalue { ptr, i32 } %847, 0
  call void @__clang_call_terminate(ptr %848) #22
  unreachable

849:                                              ; preds = %840, %844
  %850 = load ptr, ptr %321, align 8, !tbaa !20
  %851 = icmp eq ptr %850, null
  br i1 %851, label %1095, label %852

852:                                              ; preds = %849
  %853 = invoke noundef i1 %850(ptr noundef nonnull align 8 dereferenceable(32) %26, ptr noundef nonnull align 8 dereferenceable(32) %26, i32 noundef 3)
          to label %1095 unwind label %854

854:                                              ; preds = %852
  %855 = landingpad { ptr, i32 }
          catch ptr null
  %856 = extractvalue { ptr, i32 } %855, 0
  call void @__clang_call_terminate(ptr %856) #22
  unreachable

857:                                              ; preds = %341
  %858 = landingpad { ptr, i32 }
          cleanup
  %859 = load ptr, ptr %344, align 8, !tbaa !20
  %860 = icmp eq ptr %859, null
  br i1 %860, label %866, label %861

861:                                              ; preds = %857
  %862 = invoke noundef i1 %859(ptr noundef nonnull align 8 dereferenceable(32) %29, ptr noundef nonnull align 8 dereferenceable(32) %29, i32 noundef 3)
          to label %866 unwind label %863

863:                                              ; preds = %861
  %864 = landingpad { ptr, i32 }
          catch ptr null
  %865 = extractvalue { ptr, i32 } %864, 0
  call void @__clang_call_terminate(ptr %865) #22
  unreachable

866:                                              ; preds = %857, %861
  %867 = load ptr, ptr %342, align 8, !tbaa !20
  %868 = icmp eq ptr %867, null
  br i1 %868, label %1095, label %869

869:                                              ; preds = %866
  %870 = invoke noundef i1 %867(ptr noundef nonnull align 8 dereferenceable(32) %28, ptr noundef nonnull align 8 dereferenceable(32) %28, i32 noundef 3)
          to label %1095 unwind label %871

871:                                              ; preds = %869
  %872 = landingpad { ptr, i32 }
          catch ptr null
  %873 = extractvalue { ptr, i32 } %872, 0
  call void @__clang_call_terminate(ptr %873) #22
  unreachable

874:                                              ; preds = %362
  %875 = landingpad { ptr, i32 }
          cleanup
  %876 = load ptr, ptr %365, align 8, !tbaa !20
  %877 = icmp eq ptr %876, null
  br i1 %877, label %883, label %878

878:                                              ; preds = %874
  %879 = invoke noundef i1 %876(ptr noundef nonnull align 8 dereferenceable(32) %31, ptr noundef nonnull align 8 dereferenceable(32) %31, i32 noundef 3)
          to label %883 unwind label %880

880:                                              ; preds = %878
  %881 = landingpad { ptr, i32 }
          catch ptr null
  %882 = extractvalue { ptr, i32 } %881, 0
  call void @__clang_call_terminate(ptr %882) #22
  unreachable

883:                                              ; preds = %874, %878
  %884 = load ptr, ptr %363, align 8, !tbaa !20
  %885 = icmp eq ptr %884, null
  br i1 %885, label %1095, label %886

886:                                              ; preds = %883
  %887 = invoke noundef i1 %884(ptr noundef nonnull align 8 dereferenceable(32) %30, ptr noundef nonnull align 8 dereferenceable(32) %30, i32 noundef 3)
          to label %1095 unwind label %888

888:                                              ; preds = %886
  %889 = landingpad { ptr, i32 }
          catch ptr null
  %890 = extractvalue { ptr, i32 } %889, 0
  call void @__clang_call_terminate(ptr %890) #22
  unreachable

891:                                              ; preds = %383
  %892 = landingpad { ptr, i32 }
          cleanup
  %893 = load ptr, ptr %386, align 8, !tbaa !20
  %894 = icmp eq ptr %893, null
  br i1 %894, label %900, label %895

895:                                              ; preds = %891
  %896 = invoke noundef i1 %893(ptr noundef nonnull align 8 dereferenceable(32) %33, ptr noundef nonnull align 8 dereferenceable(32) %33, i32 noundef 3)
          to label %900 unwind label %897

897:                                              ; preds = %895
  %898 = landingpad { ptr, i32 }
          catch ptr null
  %899 = extractvalue { ptr, i32 } %898, 0
  call void @__clang_call_terminate(ptr %899) #22
  unreachable

900:                                              ; preds = %891, %895
  %901 = load ptr, ptr %384, align 8, !tbaa !20
  %902 = icmp eq ptr %901, null
  br i1 %902, label %1095, label %903

903:                                              ; preds = %900
  %904 = invoke noundef i1 %901(ptr noundef nonnull align 8 dereferenceable(32) %32, ptr noundef nonnull align 8 dereferenceable(32) %32, i32 noundef 3)
          to label %1095 unwind label %905

905:                                              ; preds = %903
  %906 = landingpad { ptr, i32 }
          catch ptr null
  %907 = extractvalue { ptr, i32 } %906, 0
  call void @__clang_call_terminate(ptr %907) #22
  unreachable

908:                                              ; preds = %404
  %909 = landingpad { ptr, i32 }
          cleanup
  %910 = load ptr, ptr %407, align 8, !tbaa !20
  %911 = icmp eq ptr %910, null
  br i1 %911, label %917, label %912

912:                                              ; preds = %908
  %913 = invoke noundef i1 %910(ptr noundef nonnull align 8 dereferenceable(32) %35, ptr noundef nonnull align 8 dereferenceable(32) %35, i32 noundef 3)
          to label %917 unwind label %914

914:                                              ; preds = %912
  %915 = landingpad { ptr, i32 }
          catch ptr null
  %916 = extractvalue { ptr, i32 } %915, 0
  call void @__clang_call_terminate(ptr %916) #22
  unreachable

917:                                              ; preds = %908, %912
  %918 = load ptr, ptr %405, align 8, !tbaa !20
  %919 = icmp eq ptr %918, null
  br i1 %919, label %1095, label %920

920:                                              ; preds = %917
  %921 = invoke noundef i1 %918(ptr noundef nonnull align 8 dereferenceable(32) %34, ptr noundef nonnull align 8 dereferenceable(32) %34, i32 noundef 3)
          to label %1095 unwind label %922

922:                                              ; preds = %920
  %923 = landingpad { ptr, i32 }
          catch ptr null
  %924 = extractvalue { ptr, i32 } %923, 0
  call void @__clang_call_terminate(ptr %924) #22
  unreachable

925:                                              ; preds = %425
  %926 = landingpad { ptr, i32 }
          cleanup
  %927 = load ptr, ptr %428, align 8, !tbaa !20
  %928 = icmp eq ptr %927, null
  br i1 %928, label %934, label %929

929:                                              ; preds = %925
  %930 = invoke noundef i1 %927(ptr noundef nonnull align 8 dereferenceable(32) %37, ptr noundef nonnull align 8 dereferenceable(32) %37, i32 noundef 3)
          to label %934 unwind label %931

931:                                              ; preds = %929
  %932 = landingpad { ptr, i32 }
          catch ptr null
  %933 = extractvalue { ptr, i32 } %932, 0
  call void @__clang_call_terminate(ptr %933) #22
  unreachable

934:                                              ; preds = %925, %929
  %935 = load ptr, ptr %426, align 8, !tbaa !20
  %936 = icmp eq ptr %935, null
  br i1 %936, label %1095, label %937

937:                                              ; preds = %934
  %938 = invoke noundef i1 %935(ptr noundef nonnull align 8 dereferenceable(32) %36, ptr noundef nonnull align 8 dereferenceable(32) %36, i32 noundef 3)
          to label %1095 unwind label %939

939:                                              ; preds = %937
  %940 = landingpad { ptr, i32 }
          catch ptr null
  %941 = extractvalue { ptr, i32 } %940, 0
  call void @__clang_call_terminate(ptr %941) #22
  unreachable

942:                                              ; preds = %446
  %943 = landingpad { ptr, i32 }
          cleanup
  %944 = load ptr, ptr %449, align 8, !tbaa !20
  %945 = icmp eq ptr %944, null
  br i1 %945, label %951, label %946

946:                                              ; preds = %942
  %947 = invoke noundef i1 %944(ptr noundef nonnull align 8 dereferenceable(32) %39, ptr noundef nonnull align 8 dereferenceable(32) %39, i32 noundef 3)
          to label %951 unwind label %948

948:                                              ; preds = %946
  %949 = landingpad { ptr, i32 }
          catch ptr null
  %950 = extractvalue { ptr, i32 } %949, 0
  call void @__clang_call_terminate(ptr %950) #22
  unreachable

951:                                              ; preds = %942, %946
  %952 = load ptr, ptr %447, align 8, !tbaa !20
  %953 = icmp eq ptr %952, null
  br i1 %953, label %1095, label %954

954:                                              ; preds = %951
  %955 = invoke noundef i1 %952(ptr noundef nonnull align 8 dereferenceable(32) %38, ptr noundef nonnull align 8 dereferenceable(32) %38, i32 noundef 3)
          to label %1095 unwind label %956

956:                                              ; preds = %954
  %957 = landingpad { ptr, i32 }
          catch ptr null
  %958 = extractvalue { ptr, i32 } %957, 0
  call void @__clang_call_terminate(ptr %958) #22
  unreachable

959:                                              ; preds = %467
  %960 = landingpad { ptr, i32 }
          cleanup
  %961 = load ptr, ptr %470, align 8, !tbaa !20
  %962 = icmp eq ptr %961, null
  br i1 %962, label %968, label %963

963:                                              ; preds = %959
  %964 = invoke noundef i1 %961(ptr noundef nonnull align 8 dereferenceable(32) %41, ptr noundef nonnull align 8 dereferenceable(32) %41, i32 noundef 3)
          to label %968 unwind label %965

965:                                              ; preds = %963
  %966 = landingpad { ptr, i32 }
          catch ptr null
  %967 = extractvalue { ptr, i32 } %966, 0
  call void @__clang_call_terminate(ptr %967) #22
  unreachable

968:                                              ; preds = %959, %963
  %969 = load ptr, ptr %468, align 8, !tbaa !20
  %970 = icmp eq ptr %969, null
  br i1 %970, label %1095, label %971

971:                                              ; preds = %968
  %972 = invoke noundef i1 %969(ptr noundef nonnull align 8 dereferenceable(32) %40, ptr noundef nonnull align 8 dereferenceable(32) %40, i32 noundef 3)
          to label %1095 unwind label %973

973:                                              ; preds = %971
  %974 = landingpad { ptr, i32 }
          catch ptr null
  %975 = extractvalue { ptr, i32 } %974, 0
  call void @__clang_call_terminate(ptr %975) #22
  unreachable

976:                                              ; preds = %488
  %977 = landingpad { ptr, i32 }
          cleanup
  %978 = load ptr, ptr %491, align 8, !tbaa !20
  %979 = icmp eq ptr %978, null
  br i1 %979, label %985, label %980

980:                                              ; preds = %976
  %981 = invoke noundef i1 %978(ptr noundef nonnull align 8 dereferenceable(32) %43, ptr noundef nonnull align 8 dereferenceable(32) %43, i32 noundef 3)
          to label %985 unwind label %982

982:                                              ; preds = %980
  %983 = landingpad { ptr, i32 }
          catch ptr null
  %984 = extractvalue { ptr, i32 } %983, 0
  call void @__clang_call_terminate(ptr %984) #22
  unreachable

985:                                              ; preds = %976, %980
  %986 = load ptr, ptr %489, align 8, !tbaa !20
  %987 = icmp eq ptr %986, null
  br i1 %987, label %1095, label %988

988:                                              ; preds = %985
  %989 = invoke noundef i1 %986(ptr noundef nonnull align 8 dereferenceable(32) %42, ptr noundef nonnull align 8 dereferenceable(32) %42, i32 noundef 3)
          to label %1095 unwind label %990

990:                                              ; preds = %988
  %991 = landingpad { ptr, i32 }
          catch ptr null
  %992 = extractvalue { ptr, i32 } %991, 0
  call void @__clang_call_terminate(ptr %992) #22
  unreachable

993:                                              ; preds = %509
  %994 = landingpad { ptr, i32 }
          cleanup
  %995 = load ptr, ptr %512, align 8, !tbaa !20
  %996 = icmp eq ptr %995, null
  br i1 %996, label %1002, label %997

997:                                              ; preds = %993
  %998 = invoke noundef i1 %995(ptr noundef nonnull align 8 dereferenceable(32) %45, ptr noundef nonnull align 8 dereferenceable(32) %45, i32 noundef 3)
          to label %1002 unwind label %999

999:                                              ; preds = %997
  %1000 = landingpad { ptr, i32 }
          catch ptr null
  %1001 = extractvalue { ptr, i32 } %1000, 0
  call void @__clang_call_terminate(ptr %1001) #22
  unreachable

1002:                                             ; preds = %993, %997
  %1003 = load ptr, ptr %510, align 8, !tbaa !20
  %1004 = icmp eq ptr %1003, null
  br i1 %1004, label %1095, label %1005

1005:                                             ; preds = %1002
  %1006 = invoke noundef i1 %1003(ptr noundef nonnull align 8 dereferenceable(32) %44, ptr noundef nonnull align 8 dereferenceable(32) %44, i32 noundef 3)
          to label %1095 unwind label %1007

1007:                                             ; preds = %1005
  %1008 = landingpad { ptr, i32 }
          catch ptr null
  %1009 = extractvalue { ptr, i32 } %1008, 0
  call void @__clang_call_terminate(ptr %1009) #22
  unreachable

1010:                                             ; preds = %530
  %1011 = landingpad { ptr, i32 }
          cleanup
  %1012 = load ptr, ptr %533, align 8, !tbaa !20
  %1013 = icmp eq ptr %1012, null
  br i1 %1013, label %1019, label %1014

1014:                                             ; preds = %1010
  %1015 = invoke noundef i1 %1012(ptr noundef nonnull align 8 dereferenceable(32) %47, ptr noundef nonnull align 8 dereferenceable(32) %47, i32 noundef 3)
          to label %1019 unwind label %1016

1016:                                             ; preds = %1014
  %1017 = landingpad { ptr, i32 }
          catch ptr null
  %1018 = extractvalue { ptr, i32 } %1017, 0
  call void @__clang_call_terminate(ptr %1018) #22
  unreachable

1019:                                             ; preds = %1010, %1014
  %1020 = load ptr, ptr %531, align 8, !tbaa !20
  %1021 = icmp eq ptr %1020, null
  br i1 %1021, label %1095, label %1022

1022:                                             ; preds = %1019
  %1023 = invoke noundef i1 %1020(ptr noundef nonnull align 8 dereferenceable(32) %46, ptr noundef nonnull align 8 dereferenceable(32) %46, i32 noundef 3)
          to label %1095 unwind label %1024

1024:                                             ; preds = %1022
  %1025 = landingpad { ptr, i32 }
          catch ptr null
  %1026 = extractvalue { ptr, i32 } %1025, 0
  call void @__clang_call_terminate(ptr %1026) #22
  unreachable

1027:                                             ; preds = %551
  %1028 = landingpad { ptr, i32 }
          cleanup
  %1029 = load ptr, ptr %554, align 8, !tbaa !20
  %1030 = icmp eq ptr %1029, null
  br i1 %1030, label %1036, label %1031

1031:                                             ; preds = %1027
  %1032 = invoke noundef i1 %1029(ptr noundef nonnull align 8 dereferenceable(32) %49, ptr noundef nonnull align 8 dereferenceable(32) %49, i32 noundef 3)
          to label %1036 unwind label %1033

1033:                                             ; preds = %1031
  %1034 = landingpad { ptr, i32 }
          catch ptr null
  %1035 = extractvalue { ptr, i32 } %1034, 0
  call void @__clang_call_terminate(ptr %1035) #22
  unreachable

1036:                                             ; preds = %1027, %1031
  %1037 = load ptr, ptr %552, align 8, !tbaa !20
  %1038 = icmp eq ptr %1037, null
  br i1 %1038, label %1095, label %1039

1039:                                             ; preds = %1036
  %1040 = invoke noundef i1 %1037(ptr noundef nonnull align 8 dereferenceable(32) %48, ptr noundef nonnull align 8 dereferenceable(32) %48, i32 noundef 3)
          to label %1095 unwind label %1041

1041:                                             ; preds = %1039
  %1042 = landingpad { ptr, i32 }
          catch ptr null
  %1043 = extractvalue { ptr, i32 } %1042, 0
  call void @__clang_call_terminate(ptr %1043) #22
  unreachable

1044:                                             ; preds = %572
  %1045 = landingpad { ptr, i32 }
          cleanup
  %1046 = load ptr, ptr %575, align 8, !tbaa !20
  %1047 = icmp eq ptr %1046, null
  br i1 %1047, label %1053, label %1048

1048:                                             ; preds = %1044
  %1049 = invoke noundef i1 %1046(ptr noundef nonnull align 8 dereferenceable(32) %51, ptr noundef nonnull align 8 dereferenceable(32) %51, i32 noundef 3)
          to label %1053 unwind label %1050

1050:                                             ; preds = %1048
  %1051 = landingpad { ptr, i32 }
          catch ptr null
  %1052 = extractvalue { ptr, i32 } %1051, 0
  call void @__clang_call_terminate(ptr %1052) #22
  unreachable

1053:                                             ; preds = %1044, %1048
  %1054 = load ptr, ptr %573, align 8, !tbaa !20
  %1055 = icmp eq ptr %1054, null
  br i1 %1055, label %1095, label %1056

1056:                                             ; preds = %1053
  %1057 = invoke noundef i1 %1054(ptr noundef nonnull align 8 dereferenceable(32) %50, ptr noundef nonnull align 8 dereferenceable(32) %50, i32 noundef 3)
          to label %1095 unwind label %1058

1058:                                             ; preds = %1056
  %1059 = landingpad { ptr, i32 }
          catch ptr null
  %1060 = extractvalue { ptr, i32 } %1059, 0
  call void @__clang_call_terminate(ptr %1060) #22
  unreachable

1061:                                             ; preds = %593
  %1062 = landingpad { ptr, i32 }
          cleanup
  %1063 = load ptr, ptr %596, align 8, !tbaa !20
  %1064 = icmp eq ptr %1063, null
  br i1 %1064, label %1070, label %1065

1065:                                             ; preds = %1061
  %1066 = invoke noundef i1 %1063(ptr noundef nonnull align 8 dereferenceable(32) %53, ptr noundef nonnull align 8 dereferenceable(32) %53, i32 noundef 3)
          to label %1070 unwind label %1067

1067:                                             ; preds = %1065
  %1068 = landingpad { ptr, i32 }
          catch ptr null
  %1069 = extractvalue { ptr, i32 } %1068, 0
  call void @__clang_call_terminate(ptr %1069) #22
  unreachable

1070:                                             ; preds = %1061, %1065
  %1071 = load ptr, ptr %594, align 8, !tbaa !20
  %1072 = icmp eq ptr %1071, null
  br i1 %1072, label %1095, label %1073

1073:                                             ; preds = %1070
  %1074 = invoke noundef i1 %1071(ptr noundef nonnull align 8 dereferenceable(32) %52, ptr noundef nonnull align 8 dereferenceable(32) %52, i32 noundef 3)
          to label %1095 unwind label %1075

1075:                                             ; preds = %1073
  %1076 = landingpad { ptr, i32 }
          catch ptr null
  %1077 = extractvalue { ptr, i32 } %1076, 0
  call void @__clang_call_terminate(ptr %1077) #22
  unreachable

1078:                                             ; preds = %614
  %1079 = landingpad { ptr, i32 }
          cleanup
  %1080 = load ptr, ptr %617, align 8, !tbaa !20
  %1081 = icmp eq ptr %1080, null
  br i1 %1081, label %1087, label %1082

1082:                                             ; preds = %1078
  %1083 = invoke noundef i1 %1080(ptr noundef nonnull align 8 dereferenceable(32) %55, ptr noundef nonnull align 8 dereferenceable(32) %55, i32 noundef 3)
          to label %1087 unwind label %1084

1084:                                             ; preds = %1082
  %1085 = landingpad { ptr, i32 }
          catch ptr null
  %1086 = extractvalue { ptr, i32 } %1085, 0
  call void @__clang_call_terminate(ptr %1086) #22
  unreachable

1087:                                             ; preds = %1078, %1082
  %1088 = load ptr, ptr %615, align 8, !tbaa !20
  %1089 = icmp eq ptr %1088, null
  br i1 %1089, label %1095, label %1090

1090:                                             ; preds = %1087
  %1091 = invoke noundef i1 %1088(ptr noundef nonnull align 8 dereferenceable(32) %54, ptr noundef nonnull align 8 dereferenceable(32) %54, i32 noundef 3)
          to label %1095 unwind label %1092

1092:                                             ; preds = %1090
  %1093 = landingpad { ptr, i32 }
          catch ptr null
  %1094 = extractvalue { ptr, i32 } %1093, 0
  call void @__clang_call_terminate(ptr %1094) #22
  unreachable

1095:                                             ; preds = %1090, %1087, %1053, %1056, %1070, %1073, %1039, %1036, %1002, %1005, %1019, %1022, %988, %985, %951, %954, %968, %971, %937, %934, %900, %903, %917, %920, %886, %883, %849, %852, %866, %869, %835, %832, %798, %801, %815, %818, %784, %781, %747, %750, %764, %767, %733, %730, %696, %699, %713, %716, %682, %679, %645, %648, %662, %665
  %1096 = phi { ptr, i32 } [ %637, %645 ], [ %637, %648 ], [ %654, %662 ], [ %654, %665 ], [ %671, %679 ], [ %671, %682 ], [ %688, %696 ], [ %688, %699 ], [ %705, %713 ], [ %705, %716 ], [ %722, %730 ], [ %722, %733 ], [ %739, %747 ], [ %739, %750 ], [ %756, %764 ], [ %756, %767 ], [ %773, %781 ], [ %773, %784 ], [ %790, %798 ], [ %790, %801 ], [ %807, %815 ], [ %807, %818 ], [ %824, %832 ], [ %824, %835 ], [ %841, %849 ], [ %841, %852 ], [ %858, %866 ], [ %858, %869 ], [ %875, %883 ], [ %875, %886 ], [ %892, %900 ], [ %892, %903 ], [ %909, %917 ], [ %909, %920 ], [ %926, %934 ], [ %926, %937 ], [ %943, %951 ], [ %943, %954 ], [ %960, %968 ], [ %960, %971 ], [ %977, %985 ], [ %977, %988 ], [ %994, %1002 ], [ %994, %1005 ], [ %1011, %1019 ], [ %1011, %1022 ], [ %1028, %1036 ], [ %1028, %1039 ], [ %1045, %1053 ], [ %1045, %1056 ], [ %1062, %1070 ], [ %1062, %1073 ], [ %1079, %1087 ], [ %1079, %1090 ]
  resume { ptr, i32 } %1096
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: mustprogress norecurse uwtable
define internal fastcc void @_ZL19checkVectorFunctionIiiEvSt8functionIFT_PT0_S3_S1_EES5_PKc(ptr noundef nonnull %0, ptr noundef nonnull %1, ptr noundef %2) unnamed_addr #0 personality ptr @__gxx_personality_v0 {
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
  %48 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.27, i64 noundef 9)
  %49 = icmp eq ptr %2, null
  br i1 %49, label %50, label %58

50:                                               ; preds = %3
  %51 = load ptr, ptr @_ZSt4cout, align 8, !tbaa !25
  %52 = getelementptr i8, ptr %51, i64 -24
  %53 = load i64, ptr %52, align 8
  %54 = getelementptr inbounds i8, ptr @_ZSt4cout, i64 %53
  %55 = getelementptr inbounds nuw i8, ptr %54, i64 32
  %56 = load i32, ptr %55, align 8, !tbaa !27
  %57 = or i32 %56, 1
  tail call void @_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate(ptr noundef nonnull align 8 dereferenceable(264) %54, i32 noundef %57)
  br label %61

58:                                               ; preds = %3
  %59 = tail call noundef i64 @strlen(ptr noundef nonnull dereferenceable(1) %2) #21
  %60 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull %2, i64 noundef %59)
  br label %61

61:                                               ; preds = %50, %58
  %62 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.28, i64 noundef 1)
  %63 = tail call noalias noundef nonnull dereferenceable(4000) ptr @_Znam(i64 noundef 4000) #23
  %64 = invoke noalias noundef nonnull dereferenceable(4000) ptr @_Znam(i64 noundef 4000) #23
          to label %65 unwind label %114

65:                                               ; preds = %61
  call void @llvm.lifetime.start.p0(ptr nonnull %47) #21
  store <2 x i32> <i32 -2147483648, i32 2147483647>, ptr %47, align 8, !tbaa !37
  br label %66

66:                                               ; preds = %69, %65
  %67 = phi i64 [ 0, %65 ], [ %71, %69 ]
  %68 = invoke noundef i32 @_ZNSt24uniform_int_distributionIiEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEiRT_RKNS0_10param_typeE(ptr noundef nonnull align 4 dereferenceable(8) %47, ptr noundef nonnull align 8 dereferenceable(5000) @_ZL3rng, ptr noundef nonnull align 4 dereferenceable(8) %47)
          to label %69 unwind label %118

69:                                               ; preds = %66
  %70 = getelementptr inbounds nuw i32, ptr %63, i64 %67
  store i32 %68, ptr %70, align 4, !tbaa !37
  %71 = add nuw nsw i64 %67, 1
  %72 = icmp eq i64 %71, 1000
  br i1 %72, label %73, label %66, !llvm.loop !38

73:                                               ; preds = %69
  call void @llvm.lifetime.end.p0(ptr nonnull %47) #21
  call void @llvm.lifetime.start.p0(ptr nonnull %46) #21
  store <2 x i32> <i32 -2147483648, i32 2147483647>, ptr %46, align 8, !tbaa !37
  br label %74

74:                                               ; preds = %77, %73
  %75 = phi i64 [ 0, %73 ], [ %79, %77 ]
  %76 = invoke noundef i32 @_ZNSt24uniform_int_distributionIiEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEiRT_RKNS0_10param_typeE(ptr noundef nonnull align 4 dereferenceable(8) %46, ptr noundef nonnull align 8 dereferenceable(5000) @_ZL3rng, ptr noundef nonnull align 4 dereferenceable(8) %46)
          to label %77 unwind label %116

77:                                               ; preds = %74
  %78 = getelementptr inbounds nuw i32, ptr %64, i64 %75
  store i32 %76, ptr %78, align 4, !tbaa !37
  %79 = add nuw nsw i64 %75, 1
  %80 = icmp eq i64 %79, 1000
  br i1 %80, label %81, label %74, !llvm.loop !38

81:                                               ; preds = %77
  call void @llvm.lifetime.end.p0(ptr nonnull %46) #21
  call void @llvm.lifetime.start.p0(ptr nonnull %43)
  call void @llvm.lifetime.start.p0(ptr nonnull %44)
  call void @llvm.lifetime.start.p0(ptr nonnull %45)
  store ptr %63, ptr %43, align 8, !tbaa !39
  store ptr %64, ptr %44, align 8, !tbaa !39
  store i32 1000, ptr %45, align 4, !tbaa !37
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
  store ptr %63, ptr %40, align 8, !tbaa !39
  store ptr %64, ptr %41, align 8, !tbaa !39
  store i32 1000, ptr %42, align 4, !tbaa !37
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
  store <4 x i32> splat (i32 2147483647), ptr %105, align 4, !tbaa !37
  store <4 x i32> splat (i32 2147483647), ptr %106, align 4, !tbaa !37
  %107 = getelementptr inbounds nuw i32, ptr %64, i64 %104
  %108 = getelementptr inbounds nuw i8, ptr %107, i64 16
  store <4 x i32> splat (i32 -2147483648), ptr %107, align 4, !tbaa !37
  store <4 x i32> splat (i32 -2147483648), ptr %108, align 4, !tbaa !37
  %109 = add nuw i64 %104, 8
  %110 = icmp eq i64 %109, 1000
  br i1 %110, label %124, label %103, !llvm.loop !41

111:                                              ; preds = %101
  %112 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.29)
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
  store ptr %63, ptr %37, align 8, !tbaa !39
  store ptr %64, ptr %38, align 8, !tbaa !39
  store i32 1000, ptr %39, align 4, !tbaa !37
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
  store ptr %63, ptr %34, align 8, !tbaa !39
  store ptr %64, ptr %35, align 8, !tbaa !39
  store i32 1000, ptr %36, align 4, !tbaa !37
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
  store <4 x i32> splat (i32 -2147483648), ptr %144, align 4, !tbaa !37
  store <4 x i32> splat (i32 -2147483648), ptr %145, align 4, !tbaa !37
  %146 = getelementptr inbounds nuw i32, ptr %64, i64 %143
  %147 = getelementptr inbounds nuw i8, ptr %146, i64 16
  store <4 x i32> splat (i32 2147483647), ptr %146, align 4, !tbaa !37
  store <4 x i32> splat (i32 2147483647), ptr %147, align 4, !tbaa !37
  %148 = add nuw i64 %143, 8
  %149 = icmp eq i64 %148, 1000
  br i1 %149, label %157, label %142, !llvm.loop !44

150:                                              ; preds = %140
  %151 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.29)
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
  store ptr %63, ptr %31, align 8, !tbaa !39
  store ptr %64, ptr %32, align 8, !tbaa !39
  store i32 1000, ptr %33, align 4, !tbaa !37
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
  store ptr %63, ptr %28, align 8, !tbaa !39
  store ptr %64, ptr %29, align 8, !tbaa !39
  store i32 1000, ptr %30, align 4, !tbaa !37
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
  store <4 x i32> splat (i32 -2147483648), ptr %177, align 4, !tbaa !37
  store <4 x i32> splat (i32 -2147483648), ptr %178, align 4, !tbaa !37
  %179 = getelementptr inbounds nuw i32, ptr %63, i64 %176
  %180 = getelementptr inbounds nuw i8, ptr %179, i64 16
  store <4 x i32> splat (i32 -2147483648), ptr %179, align 4, !tbaa !37
  store <4 x i32> splat (i32 -2147483648), ptr %180, align 4, !tbaa !37
  %181 = add nuw i64 %176, 8
  %182 = icmp eq i64 %181, 1000
  br i1 %182, label %190, label %175, !llvm.loop !45

183:                                              ; preds = %173
  %184 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.29)
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
  store i32 2147483647, ptr %191, align 4, !tbaa !37
  call void @llvm.lifetime.start.p0(ptr nonnull %25)
  call void @llvm.lifetime.start.p0(ptr nonnull %26)
  call void @llvm.lifetime.start.p0(ptr nonnull %27)
  store ptr %63, ptr %25, align 8, !tbaa !39
  store ptr %64, ptr %26, align 8, !tbaa !39
  store i32 1000, ptr %27, align 4, !tbaa !37
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
  store ptr %63, ptr %22, align 8, !tbaa !39
  store ptr %64, ptr %23, align 8, !tbaa !39
  store i32 1000, ptr %24, align 4, !tbaa !37
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
  store <4 x i32> splat (i32 -2147483648), ptr %211, align 4, !tbaa !37
  store <4 x i32> splat (i32 -2147483648), ptr %212, align 4, !tbaa !37
  %213 = getelementptr inbounds nuw i32, ptr %63, i64 %210
  %214 = getelementptr inbounds nuw i8, ptr %213, i64 16
  store <4 x i32> splat (i32 -2147483648), ptr %213, align 4, !tbaa !37
  store <4 x i32> splat (i32 -2147483648), ptr %214, align 4, !tbaa !37
  %215 = add nuw i64 %210, 8
  %216 = icmp eq i64 %215, 1000
  br i1 %216, label %224, label %209, !llvm.loop !46

217:                                              ; preds = %207
  %218 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.29)
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
  store i32 2147483647, ptr %63, align 4, !tbaa !37
  call void @llvm.lifetime.start.p0(ptr nonnull %19)
  call void @llvm.lifetime.start.p0(ptr nonnull %20)
  call void @llvm.lifetime.start.p0(ptr nonnull %21)
  store ptr %63, ptr %19, align 8, !tbaa !39
  store ptr %64, ptr %20, align 8, !tbaa !39
  store i32 1000, ptr %21, align 4, !tbaa !37
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
  store ptr %63, ptr %16, align 8, !tbaa !39
  store ptr %64, ptr %17, align 8, !tbaa !39
  store i32 1000, ptr %18, align 4, !tbaa !37
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
  store <4 x i32> splat (i32 -2147483648), ptr %244, align 4, !tbaa !37
  store <4 x i32> splat (i32 -2147483648), ptr %245, align 4, !tbaa !37
  %246 = getelementptr inbounds nuw i32, ptr %63, i64 %243
  %247 = getelementptr inbounds nuw i8, ptr %246, i64 16
  store <4 x i32> splat (i32 -2147483648), ptr %246, align 4, !tbaa !37
  store <4 x i32> splat (i32 -2147483648), ptr %247, align 4, !tbaa !37
  %248 = add nuw i64 %243, 8
  %249 = icmp eq i64 %248, 1000
  br i1 %249, label %257, label %242, !llvm.loop !47

250:                                              ; preds = %240
  %251 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.29)
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
  store i32 2147483647, ptr %258, align 4, !tbaa !37
  call void @llvm.lifetime.start.p0(ptr nonnull %13)
  call void @llvm.lifetime.start.p0(ptr nonnull %14)
  call void @llvm.lifetime.start.p0(ptr nonnull %15)
  store ptr %63, ptr %13, align 8, !tbaa !39
  store ptr %64, ptr %14, align 8, !tbaa !39
  store i32 1000, ptr %15, align 4, !tbaa !37
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
  store ptr %63, ptr %10, align 8, !tbaa !39
  store ptr %64, ptr %11, align 8, !tbaa !39
  store i32 1000, ptr %12, align 4, !tbaa !37
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
  store <4 x i32> splat (i32 -2147483648), ptr %278, align 4, !tbaa !37
  store <4 x i32> splat (i32 -2147483648), ptr %279, align 4, !tbaa !37
  %280 = getelementptr inbounds nuw i32, ptr %63, i64 %277
  %281 = getelementptr inbounds nuw i8, ptr %280, i64 16
  store <4 x i32> splat (i32 -2147483648), ptr %280, align 4, !tbaa !37
  store <4 x i32> splat (i32 -2147483648), ptr %281, align 4, !tbaa !37
  %282 = add nuw i64 %277, 8
  %283 = icmp eq i64 %282, 1000
  br i1 %283, label %291, label %276, !llvm.loop !48

284:                                              ; preds = %274
  %285 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.29)
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
  store i32 2147483647, ptr %258, align 4, !tbaa !37
  store i32 2147483647, ptr %63, align 4, !tbaa !37
  call void @llvm.lifetime.start.p0(ptr nonnull %7)
  call void @llvm.lifetime.start.p0(ptr nonnull %8)
  call void @llvm.lifetime.start.p0(ptr nonnull %9)
  store ptr %63, ptr %7, align 8, !tbaa !39
  store ptr %64, ptr %8, align 8, !tbaa !39
  store i32 1000, ptr %9, align 4, !tbaa !37
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
  store ptr %63, ptr %4, align 8, !tbaa !39
  store ptr %64, ptr %5, align 8, !tbaa !39
  store i32 1000, ptr %6, align 4, !tbaa !37
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
  %310 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.29)
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
define internal fastcc void @_ZL19checkVectorFunctionIifEvSt8functionIFT_PT0_S3_S1_EES5_PKc(ptr noundef nonnull %0, ptr noundef nonnull %1, ptr noundef %2) unnamed_addr #0 personality ptr @__gxx_personality_v0 {
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
  %46 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.27, i64 noundef 9)
  %47 = icmp eq ptr %2, null
  br i1 %47, label %48, label %56

48:                                               ; preds = %3
  %49 = load ptr, ptr @_ZSt4cout, align 8, !tbaa !25
  %50 = getelementptr i8, ptr %49, i64 -24
  %51 = load i64, ptr %50, align 8
  %52 = getelementptr inbounds i8, ptr @_ZSt4cout, i64 %51
  %53 = getelementptr inbounds nuw i8, ptr %52, i64 32
  %54 = load i32, ptr %53, align 8, !tbaa !27
  %55 = or i32 %54, 1
  tail call void @_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate(ptr noundef nonnull align 8 dereferenceable(264) %52, i32 noundef %55)
  br label %59

56:                                               ; preds = %3
  %57 = tail call noundef i64 @strlen(ptr noundef nonnull dereferenceable(1) %2) #21
  %58 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull %2, i64 noundef %57)
  br label %59

59:                                               ; preds = %48, %56
  %60 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.28, i64 noundef 1)
  %61 = tail call noalias noundef nonnull dereferenceable(4000) ptr @_Znam(i64 noundef 4000) #23
  %62 = invoke noalias noundef nonnull dereferenceable(4000) ptr @_Znam(i64 noundef 4000) #23
          to label %63 unwind label %96

63:                                               ; preds = %59
  tail call fastcc void @_ZL9init_dataIfEvRKSt10unique_ptrIA_T_St14default_deleteIS2_EEj(ptr nonnull %61)
  tail call fastcc void @_ZL9init_dataIfEvRKSt10unique_ptrIA_T_St14default_deleteIS2_EEj(ptr nonnull %62)
  call void @llvm.lifetime.start.p0(ptr nonnull %43)
  call void @llvm.lifetime.start.p0(ptr nonnull %44)
  call void @llvm.lifetime.start.p0(ptr nonnull %45)
  store ptr %61, ptr %43, align 8, !tbaa !49
  store ptr %62, ptr %44, align 8, !tbaa !49
  store i32 1000, ptr %45, align 4, !tbaa !37
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
  store ptr %61, ptr %40, align 8, !tbaa !49
  store ptr %62, ptr %41, align 8, !tbaa !49
  store i32 1000, ptr %42, align 4, !tbaa !37
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
  store <4 x float> splat (float 0x47EFFFFFE0000000), ptr %87, align 4, !tbaa !51
  store <4 x float> splat (float 0x47EFFFFFE0000000), ptr %88, align 4, !tbaa !51
  %89 = getelementptr inbounds nuw float, ptr %62, i64 %86
  %90 = getelementptr inbounds nuw i8, ptr %89, i64 16
  store <4 x float> splat (float 0x3810000000000000), ptr %89, align 4, !tbaa !51
  store <4 x float> splat (float 0x3810000000000000), ptr %90, align 4, !tbaa !51
  %91 = add nuw i64 %86, 8
  %92 = icmp eq i64 %91, 1000
  br i1 %92, label %102, label %85, !llvm.loop !53

93:                                               ; preds = %83
  %94 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.29)
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
  store ptr %61, ptr %37, align 8, !tbaa !49
  store ptr %62, ptr %38, align 8, !tbaa !49
  store i32 1000, ptr %39, align 4, !tbaa !37
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
  store ptr %61, ptr %34, align 8, !tbaa !49
  store ptr %62, ptr %35, align 8, !tbaa !49
  store i32 1000, ptr %36, align 4, !tbaa !37
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
  store <4 x float> splat (float 0x3810000000000000), ptr %122, align 4, !tbaa !51
  store <4 x float> splat (float 0x3810000000000000), ptr %123, align 4, !tbaa !51
  %124 = getelementptr inbounds nuw float, ptr %62, i64 %121
  %125 = getelementptr inbounds nuw i8, ptr %124, i64 16
  store <4 x float> splat (float 0x47EFFFFFE0000000), ptr %124, align 4, !tbaa !51
  store <4 x float> splat (float 0x47EFFFFFE0000000), ptr %125, align 4, !tbaa !51
  %126 = add nuw i64 %121, 8
  %127 = icmp eq i64 %126, 1000
  br i1 %127, label %135, label %120, !llvm.loop !54

128:                                              ; preds = %118
  %129 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.29)
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
  store ptr %61, ptr %31, align 8, !tbaa !49
  store ptr %62, ptr %32, align 8, !tbaa !49
  store i32 1000, ptr %33, align 4, !tbaa !37
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
  store ptr %61, ptr %28, align 8, !tbaa !49
  store ptr %62, ptr %29, align 8, !tbaa !49
  store i32 1000, ptr %30, align 4, !tbaa !37
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
  store <4 x float> splat (float 0x3810000000000000), ptr %155, align 4, !tbaa !51
  store <4 x float> splat (float 0x3810000000000000), ptr %156, align 4, !tbaa !51
  %157 = getelementptr inbounds nuw float, ptr %61, i64 %154
  %158 = getelementptr inbounds nuw i8, ptr %157, i64 16
  store <4 x float> splat (float 0x3810000000000000), ptr %157, align 4, !tbaa !51
  store <4 x float> splat (float 0x3810000000000000), ptr %158, align 4, !tbaa !51
  %159 = add nuw i64 %154, 8
  %160 = icmp eq i64 %159, 1000
  br i1 %160, label %168, label %153, !llvm.loop !55

161:                                              ; preds = %151
  %162 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.29)
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
  store float 0x47EFFFFFE0000000, ptr %169, align 4, !tbaa !51
  call void @llvm.lifetime.start.p0(ptr nonnull %25)
  call void @llvm.lifetime.start.p0(ptr nonnull %26)
  call void @llvm.lifetime.start.p0(ptr nonnull %27)
  store ptr %61, ptr %25, align 8, !tbaa !49
  store ptr %62, ptr %26, align 8, !tbaa !49
  store i32 1000, ptr %27, align 4, !tbaa !37
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
  store ptr %61, ptr %22, align 8, !tbaa !49
  store ptr %62, ptr %23, align 8, !tbaa !49
  store i32 1000, ptr %24, align 4, !tbaa !37
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
  store <4 x float> splat (float 0x3810000000000000), ptr %189, align 4, !tbaa !51
  store <4 x float> splat (float 0x3810000000000000), ptr %190, align 4, !tbaa !51
  %191 = getelementptr inbounds nuw float, ptr %61, i64 %188
  %192 = getelementptr inbounds nuw i8, ptr %191, i64 16
  store <4 x float> splat (float 0x3810000000000000), ptr %191, align 4, !tbaa !51
  store <4 x float> splat (float 0x3810000000000000), ptr %192, align 4, !tbaa !51
  %193 = add nuw i64 %188, 8
  %194 = icmp eq i64 %193, 1000
  br i1 %194, label %202, label %187, !llvm.loop !56

195:                                              ; preds = %185
  %196 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.29)
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
  store float 0x47EFFFFFE0000000, ptr %61, align 4, !tbaa !51
  call void @llvm.lifetime.start.p0(ptr nonnull %19)
  call void @llvm.lifetime.start.p0(ptr nonnull %20)
  call void @llvm.lifetime.start.p0(ptr nonnull %21)
  store ptr %61, ptr %19, align 8, !tbaa !49
  store ptr %62, ptr %20, align 8, !tbaa !49
  store i32 1000, ptr %21, align 4, !tbaa !37
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
  store ptr %61, ptr %16, align 8, !tbaa !49
  store ptr %62, ptr %17, align 8, !tbaa !49
  store i32 1000, ptr %18, align 4, !tbaa !37
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
  store <4 x float> splat (float 0x3810000000000000), ptr %222, align 4, !tbaa !51
  store <4 x float> splat (float 0x3810000000000000), ptr %223, align 4, !tbaa !51
  %224 = getelementptr inbounds nuw float, ptr %61, i64 %221
  %225 = getelementptr inbounds nuw i8, ptr %224, i64 16
  store <4 x float> splat (float 0x3810000000000000), ptr %224, align 4, !tbaa !51
  store <4 x float> splat (float 0x3810000000000000), ptr %225, align 4, !tbaa !51
  %226 = add nuw i64 %221, 8
  %227 = icmp eq i64 %226, 1000
  br i1 %227, label %235, label %220, !llvm.loop !57

228:                                              ; preds = %218
  %229 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.29)
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
  store float 0x47EFFFFFE0000000, ptr %236, align 4, !tbaa !51
  call void @llvm.lifetime.start.p0(ptr nonnull %13)
  call void @llvm.lifetime.start.p0(ptr nonnull %14)
  call void @llvm.lifetime.start.p0(ptr nonnull %15)
  store ptr %61, ptr %13, align 8, !tbaa !49
  store ptr %62, ptr %14, align 8, !tbaa !49
  store i32 1000, ptr %15, align 4, !tbaa !37
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
  store ptr %61, ptr %10, align 8, !tbaa !49
  store ptr %62, ptr %11, align 8, !tbaa !49
  store i32 1000, ptr %12, align 4, !tbaa !37
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
  store <4 x float> splat (float 0x3810000000000000), ptr %256, align 4, !tbaa !51
  store <4 x float> splat (float 0x3810000000000000), ptr %257, align 4, !tbaa !51
  %258 = getelementptr inbounds nuw float, ptr %61, i64 %255
  %259 = getelementptr inbounds nuw i8, ptr %258, i64 16
  store <4 x float> splat (float 0x3810000000000000), ptr %258, align 4, !tbaa !51
  store <4 x float> splat (float 0x3810000000000000), ptr %259, align 4, !tbaa !51
  %260 = add nuw i64 %255, 8
  %261 = icmp eq i64 %260, 1000
  br i1 %261, label %269, label %254, !llvm.loop !58

262:                                              ; preds = %252
  %263 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.29)
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
  store float 0x47EFFFFFE0000000, ptr %236, align 4, !tbaa !51
  store float 0x47EFFFFFE0000000, ptr %61, align 4, !tbaa !51
  call void @llvm.lifetime.start.p0(ptr nonnull %7)
  call void @llvm.lifetime.start.p0(ptr nonnull %8)
  call void @llvm.lifetime.start.p0(ptr nonnull %9)
  store ptr %61, ptr %7, align 8, !tbaa !49
  store ptr %62, ptr %8, align 8, !tbaa !49
  store i32 1000, ptr %9, align 4, !tbaa !37
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
  store ptr %61, ptr %4, align 8, !tbaa !49
  store ptr %62, ptr %5, align 8, !tbaa !49
  store i32 1000, ptr %6, align 4, !tbaa !37
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
  %288 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.29)
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
define internal fastcc void @_ZL19checkVectorFunctionIssEvSt8functionIFT_PT0_S3_S1_EES5_PKc(ptr noundef nonnull %0, ptr noundef nonnull %1, ptr noundef %2) unnamed_addr #0 personality ptr @__gxx_personality_v0 {
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca i16, align 4
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  %9 = alloca i16, align 4
  %10 = alloca ptr, align 8
  %11 = alloca ptr, align 8
  %12 = alloca i16, align 4
  %13 = alloca ptr, align 8
  %14 = alloca ptr, align 8
  %15 = alloca i16, align 4
  %16 = alloca ptr, align 8
  %17 = alloca ptr, align 8
  %18 = alloca i16, align 4
  %19 = alloca ptr, align 8
  %20 = alloca ptr, align 8
  %21 = alloca i16, align 4
  %22 = alloca ptr, align 8
  %23 = alloca ptr, align 8
  %24 = alloca i16, align 4
  %25 = alloca ptr, align 8
  %26 = alloca ptr, align 8
  %27 = alloca i16, align 4
  %28 = alloca ptr, align 8
  %29 = alloca ptr, align 8
  %30 = alloca i16, align 4
  %31 = alloca ptr, align 8
  %32 = alloca ptr, align 8
  %33 = alloca i16, align 4
  %34 = alloca ptr, align 8
  %35 = alloca ptr, align 8
  %36 = alloca i16, align 4
  %37 = alloca ptr, align 8
  %38 = alloca ptr, align 8
  %39 = alloca i16, align 4
  %40 = alloca ptr, align 8
  %41 = alloca ptr, align 8
  %42 = alloca i16, align 4
  %43 = alloca ptr, align 8
  %44 = alloca ptr, align 8
  %45 = alloca i16, align 4
  %46 = alloca %"class.std::uniform_int_distribution.96", align 4
  %47 = alloca %"class.std::uniform_int_distribution.96", align 4
  %48 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.27, i64 noundef 9)
  %49 = icmp eq ptr %2, null
  br i1 %49, label %50, label %58

50:                                               ; preds = %3
  %51 = load ptr, ptr @_ZSt4cout, align 8, !tbaa !25
  %52 = getelementptr i8, ptr %51, i64 -24
  %53 = load i64, ptr %52, align 8
  %54 = getelementptr inbounds i8, ptr @_ZSt4cout, i64 %53
  %55 = getelementptr inbounds nuw i8, ptr %54, i64 32
  %56 = load i32, ptr %55, align 8, !tbaa !27
  %57 = or i32 %56, 1
  tail call void @_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate(ptr noundef nonnull align 8 dereferenceable(264) %54, i32 noundef %57)
  br label %61

58:                                               ; preds = %3
  %59 = tail call noundef i64 @strlen(ptr noundef nonnull dereferenceable(1) %2) #21
  %60 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull %2, i64 noundef %59)
  br label %61

61:                                               ; preds = %50, %58
  %62 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.28, i64 noundef 1)
  %63 = tail call noalias noundef nonnull dereferenceable(2000) ptr @_Znam(i64 noundef 2000) #23
  %64 = invoke noalias noundef nonnull dereferenceable(2000) ptr @_Znam(i64 noundef 2000) #23
          to label %65 unwind label %121

65:                                               ; preds = %61
  call void @llvm.lifetime.start.p0(ptr nonnull %47) #21
  store i16 -32768, ptr %47, align 4, !tbaa !59
  %66 = getelementptr inbounds nuw i8, ptr %47, i64 2
  store i16 32767, ptr %66, align 2, !tbaa !62
  br label %67

67:                                               ; preds = %70, %65
  %68 = phi i64 [ 0, %65 ], [ %72, %70 ]
  %69 = invoke noundef i16 @_ZNSt24uniform_int_distributionIsEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEsRT_RKNS0_10param_typeE(ptr noundef nonnull align 2 dereferenceable(4) %47, ptr noundef nonnull align 8 dereferenceable(5000) @_ZL3rng, ptr noundef nonnull align 2 dereferenceable(4) %47)
          to label %70 unwind label %125

70:                                               ; preds = %67
  %71 = getelementptr inbounds nuw i16, ptr %63, i64 %68
  store i16 %69, ptr %71, align 2, !tbaa !63
  %72 = add nuw nsw i64 %68, 1
  %73 = icmp eq i64 %72, 1000
  br i1 %73, label %74, label %67, !llvm.loop !64

74:                                               ; preds = %70
  call void @llvm.lifetime.end.p0(ptr nonnull %47) #21
  call void @llvm.lifetime.start.p0(ptr nonnull %46) #21
  store i16 -32768, ptr %46, align 4, !tbaa !59
  %75 = getelementptr inbounds nuw i8, ptr %46, i64 2
  store i16 32767, ptr %75, align 2, !tbaa !62
  br label %76

76:                                               ; preds = %79, %74
  %77 = phi i64 [ 0, %74 ], [ %81, %79 ]
  %78 = invoke noundef i16 @_ZNSt24uniform_int_distributionIsEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEsRT_RKNS0_10param_typeE(ptr noundef nonnull align 2 dereferenceable(4) %46, ptr noundef nonnull align 8 dereferenceable(5000) @_ZL3rng, ptr noundef nonnull align 2 dereferenceable(4) %46)
          to label %79 unwind label %123

79:                                               ; preds = %76
  %80 = getelementptr inbounds nuw i16, ptr %64, i64 %77
  store i16 %78, ptr %80, align 2, !tbaa !63
  %81 = add nuw nsw i64 %77, 1
  %82 = icmp eq i64 %81, 1000
  br i1 %82, label %83, label %76, !llvm.loop !64

83:                                               ; preds = %79
  call void @llvm.lifetime.end.p0(ptr nonnull %46) #21
  call void @llvm.lifetime.start.p0(ptr nonnull %43)
  call void @llvm.lifetime.start.p0(ptr nonnull %44)
  call void @llvm.lifetime.start.p0(ptr nonnull %45)
  store ptr %63, ptr %43, align 8, !tbaa !65
  store ptr %64, ptr %44, align 8, !tbaa !65
  store i16 1000, ptr %45, align 4, !tbaa !63
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
  %92 = invoke noundef i16 %91(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %43, ptr noundef nonnull align 8 dereferenceable(8) %44, ptr noundef nonnull align 2 dereferenceable(2) %45)
          to label %93 unwind label %127

93:                                               ; preds = %89
  call void @llvm.lifetime.end.p0(ptr nonnull %43)
  call void @llvm.lifetime.end.p0(ptr nonnull %44)
  call void @llvm.lifetime.end.p0(ptr nonnull %45)
  call void @llvm.lifetime.start.p0(ptr nonnull %40)
  call void @llvm.lifetime.start.p0(ptr nonnull %41)
  call void @llvm.lifetime.start.p0(ptr nonnull %42)
  store ptr %63, ptr %40, align 8, !tbaa !65
  store ptr %64, ptr %41, align 8, !tbaa !65
  store i16 1000, ptr %42, align 4, !tbaa !63
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
  %102 = invoke noundef i16 %101(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %40, ptr noundef nonnull align 8 dereferenceable(8) %41, ptr noundef nonnull align 2 dereferenceable(2) %42)
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
  store <8 x i16> splat (i16 32767), ptr %107, align 2, !tbaa !63
  store <8 x i16> splat (i16 32767), ptr %108, align 2, !tbaa !63
  %109 = getelementptr inbounds nuw i16, ptr %64, i64 %106
  %110 = getelementptr inbounds nuw i8, ptr %109, i64 16
  store <8 x i16> splat (i16 -32768), ptr %109, align 2, !tbaa !63
  store <8 x i16> splat (i16 -32768), ptr %110, align 2, !tbaa !63
  %111 = add nuw i64 %106, 16
  %112 = icmp eq i64 %111, 992
  br i1 %112, label %113, label %105, !llvm.loop !67

113:                                              ; preds = %105
  %114 = getelementptr inbounds nuw i8, ptr %63, i64 1984
  store <8 x i16> splat (i16 32767), ptr %114, align 2, !tbaa !63
  %115 = getelementptr inbounds nuw i8, ptr %64, i64 1984
  store <8 x i16> splat (i16 -32768), ptr %115, align 2, !tbaa !63
  call void @llvm.lifetime.start.p0(ptr nonnull %37)
  call void @llvm.lifetime.start.p0(ptr nonnull %38)
  call void @llvm.lifetime.start.p0(ptr nonnull %39)
  store ptr %63, ptr %37, align 8, !tbaa !65
  store ptr %64, ptr %38, align 8, !tbaa !65
  store i16 1000, ptr %39, align 4, !tbaa !63
  %116 = load ptr, ptr %84, align 8, !tbaa !20
  %117 = icmp eq ptr %116, null
  br i1 %117, label %131, label %133

118:                                              ; preds = %103
  %119 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.29)
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
  %135 = invoke noundef i16 %134(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %37, ptr noundef nonnull align 8 dereferenceable(8) %38, ptr noundef nonnull align 2 dereferenceable(2) %39)
          to label %136 unwind label %162

136:                                              ; preds = %133
  call void @llvm.lifetime.end.p0(ptr nonnull %37)
  call void @llvm.lifetime.end.p0(ptr nonnull %38)
  call void @llvm.lifetime.end.p0(ptr nonnull %39)
  call void @llvm.lifetime.start.p0(ptr nonnull %34)
  call void @llvm.lifetime.start.p0(ptr nonnull %35)
  call void @llvm.lifetime.start.p0(ptr nonnull %36)
  store ptr %63, ptr %34, align 8, !tbaa !65
  store ptr %64, ptr %35, align 8, !tbaa !65
  store i16 1000, ptr %36, align 4, !tbaa !63
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
  %143 = invoke noundef i16 %142(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %34, ptr noundef nonnull align 8 dereferenceable(8) %35, ptr noundef nonnull align 2 dereferenceable(2) %36)
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
  store <8 x i16> splat (i16 -32768), ptr %148, align 2, !tbaa !63
  store <8 x i16> splat (i16 -32768), ptr %149, align 2, !tbaa !63
  %150 = getelementptr inbounds nuw i16, ptr %64, i64 %147
  %151 = getelementptr inbounds nuw i8, ptr %150, i64 16
  store <8 x i16> splat (i16 32767), ptr %150, align 2, !tbaa !63
  store <8 x i16> splat (i16 32767), ptr %151, align 2, !tbaa !63
  %152 = add nuw i64 %147, 16
  %153 = icmp eq i64 %152, 992
  br i1 %153, label %154, label %146, !llvm.loop !68

154:                                              ; preds = %146
  %155 = getelementptr inbounds nuw i8, ptr %63, i64 1984
  store <8 x i16> splat (i16 -32768), ptr %155, align 2, !tbaa !63
  %156 = getelementptr inbounds nuw i8, ptr %64, i64 1984
  store <8 x i16> splat (i16 32767), ptr %156, align 2, !tbaa !63
  call void @llvm.lifetime.start.p0(ptr nonnull %31)
  call void @llvm.lifetime.start.p0(ptr nonnull %32)
  call void @llvm.lifetime.start.p0(ptr nonnull %33)
  store ptr %63, ptr %31, align 8, !tbaa !65
  store ptr %64, ptr %32, align 8, !tbaa !65
  store i16 1000, ptr %33, align 4, !tbaa !63
  %157 = load ptr, ptr %84, align 8, !tbaa !20
  %158 = icmp eq ptr %157, null
  br i1 %158, label %166, label %168

159:                                              ; preds = %144
  %160 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.29)
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
  %170 = invoke noundef i16 %169(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %31, ptr noundef nonnull align 8 dereferenceable(8) %32, ptr noundef nonnull align 2 dereferenceable(2) %33)
          to label %171 unwind label %198

171:                                              ; preds = %168
  call void @llvm.lifetime.end.p0(ptr nonnull %31)
  call void @llvm.lifetime.end.p0(ptr nonnull %32)
  call void @llvm.lifetime.end.p0(ptr nonnull %33)
  call void @llvm.lifetime.start.p0(ptr nonnull %28)
  call void @llvm.lifetime.start.p0(ptr nonnull %29)
  call void @llvm.lifetime.start.p0(ptr nonnull %30)
  store ptr %63, ptr %28, align 8, !tbaa !65
  store ptr %64, ptr %29, align 8, !tbaa !65
  store i16 1000, ptr %30, align 4, !tbaa !63
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
  %178 = invoke noundef i16 %177(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %28, ptr noundef nonnull align 8 dereferenceable(8) %29, ptr noundef nonnull align 2 dereferenceable(2) %30)
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
  store <8 x i16> splat (i16 -32768), ptr %183, align 2, !tbaa !63
  store <8 x i16> splat (i16 -32768), ptr %184, align 2, !tbaa !63
  %185 = getelementptr inbounds nuw i16, ptr %63, i64 %182
  %186 = getelementptr inbounds nuw i8, ptr %185, i64 16
  store <8 x i16> splat (i16 -32768), ptr %185, align 2, !tbaa !63
  store <8 x i16> splat (i16 -32768), ptr %186, align 2, !tbaa !63
  %187 = add nuw i64 %182, 16
  %188 = icmp eq i64 %187, 992
  br i1 %188, label %189, label %181, !llvm.loop !69

189:                                              ; preds = %181
  %190 = getelementptr inbounds nuw i8, ptr %64, i64 1984
  store <8 x i16> splat (i16 -32768), ptr %190, align 2, !tbaa !63
  %191 = getelementptr inbounds nuw i8, ptr %63, i64 1984
  store <8 x i16> splat (i16 -32768), ptr %191, align 2, !tbaa !63
  %192 = getelementptr inbounds nuw i8, ptr %63, i64 1996
  store i16 32767, ptr %192, align 2, !tbaa !63
  call void @llvm.lifetime.start.p0(ptr nonnull %25)
  call void @llvm.lifetime.start.p0(ptr nonnull %26)
  call void @llvm.lifetime.start.p0(ptr nonnull %27)
  store ptr %63, ptr %25, align 8, !tbaa !65
  store ptr %64, ptr %26, align 8, !tbaa !65
  store i16 1000, ptr %27, align 4, !tbaa !63
  %193 = load ptr, ptr %84, align 8, !tbaa !20
  %194 = icmp eq ptr %193, null
  br i1 %194, label %202, label %204

195:                                              ; preds = %179
  %196 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.29)
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
  %206 = invoke noundef i16 %205(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %25, ptr noundef nonnull align 8 dereferenceable(8) %26, ptr noundef nonnull align 2 dereferenceable(2) %27)
          to label %207 unwind label %233

207:                                              ; preds = %204
  call void @llvm.lifetime.end.p0(ptr nonnull %25)
  call void @llvm.lifetime.end.p0(ptr nonnull %26)
  call void @llvm.lifetime.end.p0(ptr nonnull %27)
  call void @llvm.lifetime.start.p0(ptr nonnull %22)
  call void @llvm.lifetime.start.p0(ptr nonnull %23)
  call void @llvm.lifetime.start.p0(ptr nonnull %24)
  store ptr %63, ptr %22, align 8, !tbaa !65
  store ptr %64, ptr %23, align 8, !tbaa !65
  store i16 1000, ptr %24, align 4, !tbaa !63
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
  %214 = invoke noundef i16 %213(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %22, ptr noundef nonnull align 8 dereferenceable(8) %23, ptr noundef nonnull align 2 dereferenceable(2) %24)
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
  store <8 x i16> splat (i16 -32768), ptr %219, align 2, !tbaa !63
  store <8 x i16> splat (i16 -32768), ptr %220, align 2, !tbaa !63
  %221 = getelementptr inbounds nuw i16, ptr %63, i64 %218
  %222 = getelementptr inbounds nuw i8, ptr %221, i64 16
  store <8 x i16> splat (i16 -32768), ptr %221, align 2, !tbaa !63
  store <8 x i16> splat (i16 -32768), ptr %222, align 2, !tbaa !63
  %223 = add nuw i64 %218, 16
  %224 = icmp eq i64 %223, 992
  br i1 %224, label %225, label %217, !llvm.loop !70

225:                                              ; preds = %217
  %226 = getelementptr inbounds nuw i8, ptr %64, i64 1984
  store <8 x i16> splat (i16 -32768), ptr %226, align 2, !tbaa !63
  %227 = getelementptr inbounds nuw i8, ptr %63, i64 1984
  store <8 x i16> splat (i16 -32768), ptr %227, align 2, !tbaa !63
  store i16 32767, ptr %63, align 2, !tbaa !63
  call void @llvm.lifetime.start.p0(ptr nonnull %19)
  call void @llvm.lifetime.start.p0(ptr nonnull %20)
  call void @llvm.lifetime.start.p0(ptr nonnull %21)
  store ptr %63, ptr %19, align 8, !tbaa !65
  store ptr %64, ptr %20, align 8, !tbaa !65
  store i16 1000, ptr %21, align 4, !tbaa !63
  %228 = load ptr, ptr %84, align 8, !tbaa !20
  %229 = icmp eq ptr %228, null
  br i1 %229, label %237, label %239

230:                                              ; preds = %215
  %231 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.29)
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
  %241 = invoke noundef i16 %240(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %19, ptr noundef nonnull align 8 dereferenceable(8) %20, ptr noundef nonnull align 2 dereferenceable(2) %21)
          to label %242 unwind label %269

242:                                              ; preds = %239
  call void @llvm.lifetime.end.p0(ptr nonnull %19)
  call void @llvm.lifetime.end.p0(ptr nonnull %20)
  call void @llvm.lifetime.end.p0(ptr nonnull %21)
  call void @llvm.lifetime.start.p0(ptr nonnull %16)
  call void @llvm.lifetime.start.p0(ptr nonnull %17)
  call void @llvm.lifetime.start.p0(ptr nonnull %18)
  store ptr %63, ptr %16, align 8, !tbaa !65
  store ptr %64, ptr %17, align 8, !tbaa !65
  store i16 1000, ptr %18, align 4, !tbaa !63
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
  %249 = invoke noundef i16 %248(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %16, ptr noundef nonnull align 8 dereferenceable(8) %17, ptr noundef nonnull align 2 dereferenceable(2) %18)
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
  store <8 x i16> splat (i16 -32768), ptr %254, align 2, !tbaa !63
  store <8 x i16> splat (i16 -32768), ptr %255, align 2, !tbaa !63
  %256 = getelementptr inbounds nuw i16, ptr %63, i64 %253
  %257 = getelementptr inbounds nuw i8, ptr %256, i64 16
  store <8 x i16> splat (i16 -32768), ptr %256, align 2, !tbaa !63
  store <8 x i16> splat (i16 -32768), ptr %257, align 2, !tbaa !63
  %258 = add nuw i64 %253, 16
  %259 = icmp eq i64 %258, 992
  br i1 %259, label %260, label %252, !llvm.loop !71

260:                                              ; preds = %252
  %261 = getelementptr inbounds nuw i8, ptr %64, i64 1984
  store <8 x i16> splat (i16 -32768), ptr %261, align 2, !tbaa !63
  %262 = getelementptr inbounds nuw i8, ptr %63, i64 1984
  store <8 x i16> splat (i16 -32768), ptr %262, align 2, !tbaa !63
  %263 = getelementptr inbounds nuw i8, ptr %63, i64 1998
  store i16 32767, ptr %263, align 2, !tbaa !63
  call void @llvm.lifetime.start.p0(ptr nonnull %13)
  call void @llvm.lifetime.start.p0(ptr nonnull %14)
  call void @llvm.lifetime.start.p0(ptr nonnull %15)
  store ptr %63, ptr %13, align 8, !tbaa !65
  store ptr %64, ptr %14, align 8, !tbaa !65
  store i16 1000, ptr %15, align 4, !tbaa !63
  %264 = load ptr, ptr %84, align 8, !tbaa !20
  %265 = icmp eq ptr %264, null
  br i1 %265, label %273, label %275

266:                                              ; preds = %250
  %267 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.29)
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
  %277 = invoke noundef i16 %276(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %13, ptr noundef nonnull align 8 dereferenceable(8) %14, ptr noundef nonnull align 2 dereferenceable(2) %15)
          to label %278 unwind label %304

278:                                              ; preds = %275
  call void @llvm.lifetime.end.p0(ptr nonnull %13)
  call void @llvm.lifetime.end.p0(ptr nonnull %14)
  call void @llvm.lifetime.end.p0(ptr nonnull %15)
  call void @llvm.lifetime.start.p0(ptr nonnull %10)
  call void @llvm.lifetime.start.p0(ptr nonnull %11)
  call void @llvm.lifetime.start.p0(ptr nonnull %12)
  store ptr %63, ptr %10, align 8, !tbaa !65
  store ptr %64, ptr %11, align 8, !tbaa !65
  store i16 1000, ptr %12, align 4, !tbaa !63
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
  %285 = invoke noundef i16 %284(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %10, ptr noundef nonnull align 8 dereferenceable(8) %11, ptr noundef nonnull align 2 dereferenceable(2) %12)
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
  store <8 x i16> splat (i16 -32768), ptr %290, align 2, !tbaa !63
  store <8 x i16> splat (i16 -32768), ptr %291, align 2, !tbaa !63
  %292 = getelementptr inbounds nuw i16, ptr %63, i64 %289
  %293 = getelementptr inbounds nuw i8, ptr %292, i64 16
  store <8 x i16> splat (i16 -32768), ptr %292, align 2, !tbaa !63
  store <8 x i16> splat (i16 -32768), ptr %293, align 2, !tbaa !63
  %294 = add nuw i64 %289, 16
  %295 = icmp eq i64 %294, 992
  br i1 %295, label %296, label %288, !llvm.loop !72

296:                                              ; preds = %288
  %297 = getelementptr inbounds nuw i8, ptr %64, i64 1984
  store <8 x i16> splat (i16 -32768), ptr %297, align 2, !tbaa !63
  %298 = getelementptr inbounds nuw i8, ptr %63, i64 1984
  store <8 x i16> splat (i16 -32768), ptr %298, align 2, !tbaa !63
  store i16 32767, ptr %263, align 2, !tbaa !63
  store i16 32767, ptr %63, align 2, !tbaa !63
  call void @llvm.lifetime.start.p0(ptr nonnull %7)
  call void @llvm.lifetime.start.p0(ptr nonnull %8)
  call void @llvm.lifetime.start.p0(ptr nonnull %9)
  store ptr %63, ptr %7, align 8, !tbaa !65
  store ptr %64, ptr %8, align 8, !tbaa !65
  store i16 1000, ptr %9, align 4, !tbaa !63
  %299 = load ptr, ptr %84, align 8, !tbaa !20
  %300 = icmp eq ptr %299, null
  br i1 %300, label %308, label %310

301:                                              ; preds = %286
  %302 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.29)
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
  %312 = invoke noundef i16 %311(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %7, ptr noundef nonnull align 8 dereferenceable(8) %8, ptr noundef nonnull align 2 dereferenceable(2) %9)
          to label %313 unwind label %326

313:                                              ; preds = %310
  call void @llvm.lifetime.end.p0(ptr nonnull %7)
  call void @llvm.lifetime.end.p0(ptr nonnull %8)
  call void @llvm.lifetime.end.p0(ptr nonnull %9)
  call void @llvm.lifetime.start.p0(ptr nonnull %4)
  call void @llvm.lifetime.start.p0(ptr nonnull %5)
  call void @llvm.lifetime.start.p0(ptr nonnull %6)
  store ptr %63, ptr %4, align 8, !tbaa !65
  store ptr %64, ptr %5, align 8, !tbaa !65
  store i16 1000, ptr %6, align 4, !tbaa !63
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
  %320 = invoke noundef i16 %319(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %4, ptr noundef nonnull align 8 dereferenceable(8) %5, ptr noundef nonnull align 2 dereferenceable(2) %6)
          to label %321 unwind label %328

321:                                              ; preds = %318
  call void @llvm.lifetime.end.p0(ptr nonnull %4)
  call void @llvm.lifetime.end.p0(ptr nonnull %5)
  call void @llvm.lifetime.end.p0(ptr nonnull %6)
  %322 = icmp eq i16 %312, %320
  br i1 %322, label %330, label %323

323:                                              ; preds = %321
  %324 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.29)
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
  %6 = load i32, ptr %5, align 4, !tbaa !73
  %7 = sext i32 %6 to i64
  %8 = load i32, ptr %2, align 4, !tbaa !75
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
  br i1 %28, label %24, label %29, !llvm.loop !76

29:                                               ; preds = %24, %12, %20
  %30 = phi i64 [ %17, %12 ], [ %17, %20 ], [ %26, %24 ]
  %31 = lshr i64 %30, 32
  br label %45

32:                                               ; preds = %3
  %33 = icmp eq i64 %10, 4294967295
  br i1 %33, label %43, label %34

34:                                               ; preds = %32, %34
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #21
  store <2 x i32> <i32 0, i32 -1>, ptr %4, align 8, !tbaa !37
  %35 = call noundef i32 @_ZNSt24uniform_int_distributionIiEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEiRT_RKNS0_10param_typeE(ptr noundef nonnull align 4 dereferenceable(8) %0, ptr noundef nonnull align 8 dereferenceable(5000) %1, ptr noundef nonnull align 4 dereferenceable(8) %4)
  %36 = sext i32 %35 to i64
  %37 = shl nsw i64 %36, 32
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #21
  %38 = call noundef i64 @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv(ptr noundef nonnull align 8 dereferenceable(5000) %1)
  %39 = add i64 %37, %38
  %40 = icmp ugt i64 %39, %10
  %41 = icmp ult i64 %39, %37
  %42 = or i1 %40, %41
  br i1 %42, label %34, label %45, !llvm.loop !77

43:                                               ; preds = %32
  %44 = tail call noundef i64 @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv(ptr noundef nonnull align 8 dereferenceable(5000) %1)
  br label %45

45:                                               ; preds = %34, %43, %29
  %46 = phi i64 [ %31, %29 ], [ %44, %43 ], [ %39, %34 ]
  %47 = load i32, ptr %2, align 4, !tbaa !75
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
  br i1 %43, label %44, label %8, !llvm.loop !78

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
  br i1 %111, label %112, label %91, !llvm.loop !79

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
define internal noundef i32 @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE3$_0E9_M_invokeERKSt9_Any_dataOS0_S7_Oi"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !39
  %6 = load ptr, ptr %2, align 8, !tbaa !39
  %7 = load i32, ptr %3, align 4, !tbaa !37
  %8 = icmp sgt i32 %7, 0
  br i1 %8, label %9, label %23

9:                                                ; preds = %4
  %10 = zext nneg i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 0, %9 ], [ %21, %11 ]
  %13 = phi i32 [ -1, %9 ], [ %20, %11 ]
  %14 = getelementptr inbounds nuw i32, ptr %5, i64 %12
  %15 = load i32, ptr %14, align 4, !tbaa !37
  %16 = getelementptr inbounds nuw i32, ptr %6, i64 %12
  %17 = load i32, ptr %16, align 4, !tbaa !37
  %18 = icmp sgt i32 %15, %17
  %19 = trunc nuw nsw i64 %12 to i32
  %20 = select i1 %18, i32 %19, i32 %13
  %21 = add nuw nsw i64 %12, 1
  %22 = icmp eq i64 %21, %10
  br i1 %22, label %23, label %11, !llvm.loop !80

23:                                               ; preds = %11, %4
  %24 = phi i32 [ -1, %4 ], [ %20, %11 ]
  ret i32 %24
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE3$_0E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_0", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !83
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i32 @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE3$_1E9_M_invokeERKSt9_Any_dataOS0_S7_Oi"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !39
  %6 = load ptr, ptr %2, align 8, !tbaa !39
  %7 = load i32, ptr %3, align 4, !tbaa !37
  %8 = icmp sgt i32 %7, 0
  br i1 %8, label %9, label %56

9:                                                ; preds = %4
  %10 = zext nneg i32 %7 to i64
  %11 = icmp ult i32 %7, 8
  br i1 %11, label %41, label %12

12:                                               ; preds = %9
  %13 = and i64 %10, 2147483640
  br label %14

14:                                               ; preds = %14, %12
  %15 = phi i64 [ 0, %12 ], [ %32, %14 ]
  %16 = phi <4 x i32> [ splat (i32 -2147483648), %12 ], [ %30, %14 ]
  %17 = phi <4 x i32> [ splat (i32 -2147483648), %12 ], [ %31, %14 ]
  %18 = phi <4 x i32> [ <i32 0, i32 1, i32 2, i32 3>, %12 ], [ %33, %14 ]
  %19 = add <4 x i32> %18, splat (i32 4)
  %20 = getelementptr inbounds nuw i32, ptr %5, i64 %15
  %21 = getelementptr inbounds nuw i8, ptr %20, i64 16
  %22 = load <4 x i32>, ptr %20, align 4, !tbaa !37
  %23 = load <4 x i32>, ptr %21, align 4, !tbaa !37
  %24 = getelementptr inbounds nuw i32, ptr %6, i64 %15
  %25 = getelementptr inbounds nuw i8, ptr %24, i64 16
  %26 = load <4 x i32>, ptr %24, align 4, !tbaa !37
  %27 = load <4 x i32>, ptr %25, align 4, !tbaa !37
  %28 = icmp sgt <4 x i32> %22, %26
  %29 = icmp sgt <4 x i32> %23, %27
  %30 = select <4 x i1> %28, <4 x i32> %18, <4 x i32> %16
  %31 = select <4 x i1> %29, <4 x i32> %19, <4 x i32> %17
  %32 = add nuw i64 %15, 8
  %33 = add <4 x i32> %18, splat (i32 8)
  %34 = icmp eq i64 %32, %13
  br i1 %34, label %35, label %14, !llvm.loop !84

35:                                               ; preds = %14
  %36 = tail call <4 x i32> @llvm.smax.v4i32(<4 x i32> %30, <4 x i32> %31)
  %37 = tail call i32 @llvm.vector.reduce.smax.v4i32(<4 x i32> %36)
  %38 = icmp eq i32 %37, -2147483648
  %39 = select i1 %38, i32 -1, i32 %37
  %40 = icmp eq i64 %13, %10
  br i1 %40, label %56, label %41

41:                                               ; preds = %9, %35
  %42 = phi i64 [ 0, %9 ], [ %13, %35 ]
  %43 = phi i32 [ -1, %9 ], [ %39, %35 ]
  br label %44

44:                                               ; preds = %41, %44
  %45 = phi i64 [ %54, %44 ], [ %42, %41 ]
  %46 = phi i32 [ %53, %44 ], [ %43, %41 ]
  %47 = getelementptr inbounds nuw i32, ptr %5, i64 %45
  %48 = load i32, ptr %47, align 4, !tbaa !37
  %49 = getelementptr inbounds nuw i32, ptr %6, i64 %45
  %50 = load i32, ptr %49, align 4, !tbaa !37
  %51 = icmp sgt i32 %48, %50
  %52 = trunc nuw nsw i64 %45 to i32
  %53 = select i1 %51, i32 %52, i32 %46
  %54 = add nuw nsw i64 %45, 1
  %55 = icmp eq i64 %54, %10
  br i1 %55, label %56, label %44, !llvm.loop !85

56:                                               ; preds = %44, %35, %4
  %57 = phi i32 [ -1, %4 ], [ %39, %35 ], [ %53, %44 ]
  ret i32 %57
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE3$_1E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_1", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !83
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress norecurse nounwind uwtable
define internal fastcc void @_ZL9init_dataIfEvRKSt10unique_ptrIA_T_St14default_deleteIS2_EEj(ptr writeonly captures(none) %0) unnamed_addr #16 {
  %2 = tail call fp128 @llvm.log.f128(fp128 0xL0000000000000000401F000000000000), !tbaa !37
  %3 = tail call fp128 @llvm.log.f128(fp128 0xL00000000000000004000000000000000), !tbaa !37
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
  br i1 %16, label %156, label %154, !prof !86

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
  br i1 %61, label %62, label %26, !llvm.loop !87

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
  br i1 %119, label %120, label %99, !llvm.loop !88

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
  br i1 %153, label %14, label %17, !llvm.loop !89

154:                                              ; preds = %14
  %155 = tail call noundef float @nextafterf(float noundef 1.000000e+00, float noundef 0.000000e+00) #21, !tbaa !37
  br label %156

156:                                              ; preds = %14, %154
  %157 = phi float [ %155, %154 ], [ %15, %14 ]
  %158 = tail call noundef float @llvm.fmuladd.f32(float %157, float 0x47EFFFFFE0000000, float 0x3810000000000000)
  %159 = getelementptr inbounds nuw float, ptr %0, i64 %13
  store float %158, ptr %159, align 4, !tbaa !51
  %160 = add nuw nsw i64 %13, 1
  %161 = icmp eq i64 %160, 1000
  br i1 %161, label %10, label %11, !llvm.loop !90
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fmuladd.f32(float, float, float) #17

; Function Attrs: nounwind
declare float @nextafterf(float noundef, float noundef) local_unnamed_addr #18

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i32 @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE3$_0E9_M_invokeERKSt9_Any_dataOS0_S7_Oi"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !49
  %6 = load ptr, ptr %2, align 8, !tbaa !49
  %7 = load i32, ptr %3, align 4, !tbaa !37
  %8 = icmp sgt i32 %7, 0
  br i1 %8, label %9, label %23

9:                                                ; preds = %4
  %10 = zext nneg i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 0, %9 ], [ %21, %11 ]
  %13 = phi i32 [ -1, %9 ], [ %20, %11 ]
  %14 = getelementptr inbounds nuw float, ptr %5, i64 %12
  %15 = load float, ptr %14, align 4, !tbaa !51
  %16 = getelementptr inbounds nuw float, ptr %6, i64 %12
  %17 = load float, ptr %16, align 4, !tbaa !51
  %18 = fcmp ogt float %15, %17
  %19 = trunc nuw nsw i64 %12 to i32
  %20 = select i1 %18, i32 %19, i32 %13
  %21 = add nuw nsw i64 %12, 1
  %22 = icmp eq i64 %21, %10
  br i1 %22, label %23, label %11, !llvm.loop !91

23:                                               ; preds = %11, %4
  %24 = phi i32 [ -1, %4 ], [ %20, %11 ]
  ret i32 %24
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE3$_0E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_0", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !83
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i32 @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE3$_1E9_M_invokeERKSt9_Any_dataOS0_S7_Oi"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !49
  %6 = load ptr, ptr %2, align 8, !tbaa !49
  %7 = load i32, ptr %3, align 4, !tbaa !37
  %8 = icmp sgt i32 %7, 0
  br i1 %8, label %9, label %56

9:                                                ; preds = %4
  %10 = zext nneg i32 %7 to i64
  %11 = icmp ult i32 %7, 8
  br i1 %11, label %41, label %12

12:                                               ; preds = %9
  %13 = and i64 %10, 2147483640
  br label %14

14:                                               ; preds = %14, %12
  %15 = phi i64 [ 0, %12 ], [ %32, %14 ]
  %16 = phi <4 x i32> [ splat (i32 -2147483648), %12 ], [ %30, %14 ]
  %17 = phi <4 x i32> [ splat (i32 -2147483648), %12 ], [ %31, %14 ]
  %18 = phi <4 x i32> [ <i32 0, i32 1, i32 2, i32 3>, %12 ], [ %33, %14 ]
  %19 = add <4 x i32> %18, splat (i32 4)
  %20 = getelementptr inbounds nuw float, ptr %5, i64 %15
  %21 = getelementptr inbounds nuw i8, ptr %20, i64 16
  %22 = load <4 x float>, ptr %20, align 4, !tbaa !51
  %23 = load <4 x float>, ptr %21, align 4, !tbaa !51
  %24 = getelementptr inbounds nuw float, ptr %6, i64 %15
  %25 = getelementptr inbounds nuw i8, ptr %24, i64 16
  %26 = load <4 x float>, ptr %24, align 4, !tbaa !51
  %27 = load <4 x float>, ptr %25, align 4, !tbaa !51
  %28 = fcmp ogt <4 x float> %22, %26
  %29 = fcmp ogt <4 x float> %23, %27
  %30 = select <4 x i1> %28, <4 x i32> %18, <4 x i32> %16
  %31 = select <4 x i1> %29, <4 x i32> %19, <4 x i32> %17
  %32 = add nuw i64 %15, 8
  %33 = add <4 x i32> %18, splat (i32 8)
  %34 = icmp eq i64 %32, %13
  br i1 %34, label %35, label %14, !llvm.loop !92

35:                                               ; preds = %14
  %36 = tail call <4 x i32> @llvm.smax.v4i32(<4 x i32> %30, <4 x i32> %31)
  %37 = tail call i32 @llvm.vector.reduce.smax.v4i32(<4 x i32> %36)
  %38 = icmp eq i32 %37, -2147483648
  %39 = select i1 %38, i32 -1, i32 %37
  %40 = icmp eq i64 %13, %10
  br i1 %40, label %56, label %41

41:                                               ; preds = %9, %35
  %42 = phi i64 [ 0, %9 ], [ %13, %35 ]
  %43 = phi i32 [ -1, %9 ], [ %39, %35 ]
  br label %44

44:                                               ; preds = %41, %44
  %45 = phi i64 [ %54, %44 ], [ %42, %41 ]
  %46 = phi i32 [ %53, %44 ], [ %43, %41 ]
  %47 = getelementptr inbounds nuw float, ptr %5, i64 %45
  %48 = load float, ptr %47, align 4, !tbaa !51
  %49 = getelementptr inbounds nuw float, ptr %6, i64 %45
  %50 = load float, ptr %49, align 4, !tbaa !51
  %51 = fcmp ogt float %48, %50
  %52 = trunc nuw nsw i64 %45 to i32
  %53 = select i1 %51, i32 %52, i32 %46
  %54 = add nuw nsw i64 %45, 1
  %55 = icmp eq i64 %54, %10
  br i1 %55, label %56, label %44, !llvm.loop !93

56:                                               ; preds = %44, %35, %4
  %57 = phi i32 [ -1, %4 ], [ %39, %35 ], [ %53, %44 ]
  ret i32 %57
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE3$_1E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_1", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !83
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local noundef i16 @_ZNSt24uniform_int_distributionIsEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEsRT_RKNS0_10param_typeE(ptr noundef nonnull align 2 dereferenceable(4) %0, ptr noundef nonnull align 8 dereferenceable(5000) %1, ptr noundef nonnull align 2 dereferenceable(4) %2) local_unnamed_addr #10 comdat {
  %4 = alloca %"struct.std::uniform_int_distribution<short>::param_type", align 4
  %5 = getelementptr inbounds nuw i8, ptr %2, i64 2
  %6 = load i16, ptr %5, align 2, !tbaa !62
  %7 = sext i16 %6 to i64
  %8 = load i16, ptr %2, align 2, !tbaa !59
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
  br i1 %30, label %26, label %31, !llvm.loop !94

31:                                               ; preds = %26, %14, %22
  %32 = phi i64 [ %19, %14 ], [ %19, %22 ], [ %28, %26 ]
  %33 = lshr i64 %32, 32
  br label %43

34:                                               ; preds = %12, %34
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #21
  store i16 0, ptr %4, align 4, !tbaa !59
  store i16 -1, ptr %13, align 2, !tbaa !62
  %35 = call noundef i16 @_ZNSt24uniform_int_distributionIsEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEsRT_RKNS0_10param_typeE(ptr noundef nonnull align 2 dereferenceable(4) %0, ptr noundef nonnull align 8 dereferenceable(5000) %1, ptr noundef nonnull align 2 dereferenceable(4) %4)
  %36 = sext i16 %35 to i64
  %37 = shl nsw i64 %36, 32
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #21
  %38 = call noundef i64 @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv(ptr noundef nonnull align 8 dereferenceable(5000) %1)
  %39 = add i64 %37, %38
  %40 = icmp ugt i64 %39, %10
  %41 = icmp ult i64 %39, %37
  %42 = or i1 %40, %41
  br i1 %42, label %34, label %43, !llvm.loop !95

43:                                               ; preds = %34, %31
  %44 = phi i64 [ %33, %31 ], [ %39, %34 ]
  %45 = load i16, ptr %2, align 2, !tbaa !59
  %46 = trunc i64 %44 to i16
  %47 = add i16 %45, %46
  ret i16 %47
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i16 @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE3$_2E9_M_invokeERKSt9_Any_dataOS0_S7_Os"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 2 captures(none) dereferenceable(2) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !65
  %6 = load ptr, ptr %2, align 8, !tbaa !65
  %7 = load i16, ptr %3, align 2, !tbaa !63
  %8 = icmp sgt i16 %7, 0
  br i1 %8, label %9, label %23

9:                                                ; preds = %4
  %10 = zext nneg i16 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 0, %9 ], [ %21, %11 ]
  %13 = phi i16 [ -1, %9 ], [ %20, %11 ]
  %14 = getelementptr inbounds nuw i16, ptr %5, i64 %12
  %15 = load i16, ptr %14, align 2, !tbaa !63
  %16 = getelementptr inbounds nuw i16, ptr %6, i64 %12
  %17 = load i16, ptr %16, align 2, !tbaa !63
  %18 = icmp sgt i16 %15, %17
  %19 = trunc nuw nsw i64 %12 to i16
  %20 = select i1 %18, i16 %19, i16 %13
  %21 = add nuw nsw i64 %12, 1
  %22 = icmp eq i64 %21, %10
  br i1 %22, label %23, label %11, !llvm.loop !96

23:                                               ; preds = %11, %4
  %24 = phi i16 [ -1, %4 ], [ %20, %11 ]
  ret i16 %24
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE3$_2E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_2", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !83
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i16 @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE3$_3E9_M_invokeERKSt9_Any_dataOS0_S7_Os"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 2 captures(none) dereferenceable(2) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !65
  %6 = load ptr, ptr %2, align 8, !tbaa !65
  %7 = load i16, ptr %3, align 2, !tbaa !63
  %8 = icmp sgt i16 %7, 0
  br i1 %8, label %9, label %91

9:                                                ; preds = %4
  %10 = zext nneg i16 %7 to i64
  %11 = icmp ult i16 %7, 4
  br i1 %11, label %76, label %12

12:                                               ; preds = %9
  %13 = icmp ult i16 %7, 16
  br i1 %13, label %46, label %14

14:                                               ; preds = %12
  %15 = and i64 %10, 32752
  br label %16

16:                                               ; preds = %16, %14
  %17 = phi i64 [ 0, %14 ], [ %34, %16 ]
  %18 = phi <8 x i16> [ splat (i16 -32768), %14 ], [ %32, %16 ]
  %19 = phi <8 x i16> [ splat (i16 -32768), %14 ], [ %33, %16 ]
  %20 = phi <8 x i16> [ <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, %14 ], [ %35, %16 ]
  %21 = add <8 x i16> %20, splat (i16 8)
  %22 = getelementptr inbounds nuw i16, ptr %5, i64 %17
  %23 = getelementptr inbounds nuw i8, ptr %22, i64 16
  %24 = load <8 x i16>, ptr %22, align 2, !tbaa !63
  %25 = load <8 x i16>, ptr %23, align 2, !tbaa !63
  %26 = getelementptr inbounds nuw i16, ptr %6, i64 %17
  %27 = getelementptr inbounds nuw i8, ptr %26, i64 16
  %28 = load <8 x i16>, ptr %26, align 2, !tbaa !63
  %29 = load <8 x i16>, ptr %27, align 2, !tbaa !63
  %30 = icmp sgt <8 x i16> %24, %28
  %31 = icmp sgt <8 x i16> %25, %29
  %32 = select <8 x i1> %30, <8 x i16> %20, <8 x i16> %18
  %33 = select <8 x i1> %31, <8 x i16> %21, <8 x i16> %19
  %34 = add nuw i64 %17, 16
  %35 = add <8 x i16> %20, splat (i16 16)
  %36 = icmp eq i64 %34, %15
  br i1 %36, label %37, label %16, !llvm.loop !97

37:                                               ; preds = %16
  %38 = tail call <8 x i16> @llvm.smax.v8i16(<8 x i16> %32, <8 x i16> %33)
  %39 = tail call i16 @llvm.vector.reduce.smax.v8i16(<8 x i16> %38)
  %40 = icmp eq i16 %39, -32768
  %41 = select i1 %40, i16 -1, i16 %39
  %42 = icmp eq i64 %15, %10
  br i1 %42, label %91, label %43

43:                                               ; preds = %37
  %44 = and i64 %10, 12
  %45 = icmp eq i64 %44, 0
  br i1 %45, label %76, label %46

46:                                               ; preds = %43, %12
  %47 = phi i64 [ %15, %43 ], [ 0, %12 ]
  %48 = phi i16 [ %41, %43 ], [ -1, %12 ]
  %49 = icmp eq i16 %48, -1
  %50 = select i1 %49, i16 -32768, i16 %48
  %51 = and i64 %10, 32764
  %52 = insertelement <4 x i16> poison, i16 %50, i64 0
  %53 = shufflevector <4 x i16> %52, <4 x i16> poison, <4 x i32> zeroinitializer
  %54 = trunc nuw nsw i64 %47 to i16
  %55 = insertelement <4 x i16> poison, i16 %54, i64 0
  %56 = shufflevector <4 x i16> %55, <4 x i16> poison, <4 x i32> zeroinitializer
  %57 = or disjoint <4 x i16> %56, <i16 0, i16 1, i16 2, i16 3>
  br label %58

58:                                               ; preds = %58, %46
  %59 = phi i64 [ %47, %46 ], [ %68, %58 ]
  %60 = phi <4 x i16> [ %53, %46 ], [ %67, %58 ]
  %61 = phi <4 x i16> [ %57, %46 ], [ %69, %58 ]
  %62 = getelementptr inbounds nuw i16, ptr %5, i64 %59
  %63 = load <4 x i16>, ptr %62, align 2, !tbaa !63
  %64 = getelementptr inbounds nuw i16, ptr %6, i64 %59
  %65 = load <4 x i16>, ptr %64, align 2, !tbaa !63
  %66 = icmp sgt <4 x i16> %63, %65
  %67 = select <4 x i1> %66, <4 x i16> %61, <4 x i16> %60
  %68 = add nuw i64 %59, 4
  %69 = add <4 x i16> %61, splat (i16 4)
  %70 = icmp eq i64 %68, %51
  br i1 %70, label %71, label %58, !llvm.loop !98

71:                                               ; preds = %58
  %72 = tail call i16 @llvm.vector.reduce.smax.v4i16(<4 x i16> %67)
  %73 = icmp eq i16 %72, -32768
  %74 = select i1 %73, i16 -1, i16 %72
  %75 = icmp eq i64 %51, %10
  br i1 %75, label %91, label %76

76:                                               ; preds = %43, %71, %9
  %77 = phi i64 [ 0, %9 ], [ %15, %43 ], [ %51, %71 ]
  %78 = phi i16 [ -1, %9 ], [ %41, %43 ], [ %74, %71 ]
  br label %79

79:                                               ; preds = %76, %79
  %80 = phi i64 [ %89, %79 ], [ %77, %76 ]
  %81 = phi i16 [ %88, %79 ], [ %78, %76 ]
  %82 = getelementptr inbounds nuw i16, ptr %5, i64 %80
  %83 = load i16, ptr %82, align 2, !tbaa !63
  %84 = getelementptr inbounds nuw i16, ptr %6, i64 %80
  %85 = load i16, ptr %84, align 2, !tbaa !63
  %86 = icmp sgt i16 %83, %85
  %87 = trunc nuw nsw i64 %80 to i16
  %88 = select i1 %86, i16 %87, i16 %81
  %89 = add nuw nsw i64 %80, 1
  %90 = icmp eq i64 %89, %10
  br i1 %90, label %91, label %79, !llvm.loop !99

91:                                               ; preds = %79, %37, %71, %4
  %92 = phi i16 [ -1, %4 ], [ %41, %37 ], [ %74, %71 ], [ %88, %79 ]
  ret i16 %92
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE3$_3E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_3", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !83
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i32 @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE3$_4E9_M_invokeERKSt9_Any_dataOS0_S7_Oi"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !39
  %6 = load ptr, ptr %2, align 8, !tbaa !39
  %7 = load i32, ptr %3, align 4, !tbaa !37
  %8 = icmp sgt i32 %7, 0
  br i1 %8, label %9, label %23

9:                                                ; preds = %4
  %10 = zext nneg i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 0, %9 ], [ %21, %11 ]
  %13 = phi i32 [ -1, %9 ], [ %20, %11 ]
  %14 = getelementptr inbounds nuw i32, ptr %5, i64 %12
  %15 = load i32, ptr %14, align 4, !tbaa !37
  %16 = getelementptr inbounds nuw i32, ptr %6, i64 %12
  %17 = load i32, ptr %16, align 4, !tbaa !37
  %18 = icmp sgt i32 %15, %17
  %19 = trunc nuw nsw i64 %12 to i32
  %20 = select i1 %18, i32 %13, i32 %19
  %21 = add nuw nsw i64 %12, 1
  %22 = icmp eq i64 %21, %10
  br i1 %22, label %23, label %11, !llvm.loop !100

23:                                               ; preds = %11, %4
  %24 = phi i32 [ -1, %4 ], [ %20, %11 ]
  ret i32 %24
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE3$_4E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_4", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !83
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i32 @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE3$_5E9_M_invokeERKSt9_Any_dataOS0_S7_Oi"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !39
  %6 = load ptr, ptr %2, align 8, !tbaa !39
  %7 = load i32, ptr %3, align 4, !tbaa !37
  %8 = icmp sgt i32 %7, 0
  br i1 %8, label %9, label %56

9:                                                ; preds = %4
  %10 = zext nneg i32 %7 to i64
  %11 = icmp ult i32 %7, 8
  br i1 %11, label %41, label %12

12:                                               ; preds = %9
  %13 = and i64 %10, 2147483640
  br label %14

14:                                               ; preds = %14, %12
  %15 = phi i64 [ 0, %12 ], [ %32, %14 ]
  %16 = phi <4 x i32> [ splat (i32 -2147483648), %12 ], [ %30, %14 ]
  %17 = phi <4 x i32> [ splat (i32 -2147483648), %12 ], [ %31, %14 ]
  %18 = phi <4 x i32> [ <i32 0, i32 1, i32 2, i32 3>, %12 ], [ %33, %14 ]
  %19 = add <4 x i32> %18, splat (i32 4)
  %20 = getelementptr inbounds nuw i32, ptr %5, i64 %15
  %21 = getelementptr inbounds nuw i8, ptr %20, i64 16
  %22 = load <4 x i32>, ptr %20, align 4, !tbaa !37
  %23 = load <4 x i32>, ptr %21, align 4, !tbaa !37
  %24 = getelementptr inbounds nuw i32, ptr %6, i64 %15
  %25 = getelementptr inbounds nuw i8, ptr %24, i64 16
  %26 = load <4 x i32>, ptr %24, align 4, !tbaa !37
  %27 = load <4 x i32>, ptr %25, align 4, !tbaa !37
  %28 = icmp sgt <4 x i32> %22, %26
  %29 = icmp sgt <4 x i32> %23, %27
  %30 = select <4 x i1> %28, <4 x i32> %16, <4 x i32> %18
  %31 = select <4 x i1> %29, <4 x i32> %17, <4 x i32> %19
  %32 = add nuw i64 %15, 8
  %33 = add <4 x i32> %18, splat (i32 8)
  %34 = icmp eq i64 %32, %13
  br i1 %34, label %35, label %14, !llvm.loop !101

35:                                               ; preds = %14
  %36 = tail call <4 x i32> @llvm.smax.v4i32(<4 x i32> %30, <4 x i32> %31)
  %37 = tail call i32 @llvm.vector.reduce.smax.v4i32(<4 x i32> %36)
  %38 = icmp eq i32 %37, -2147483648
  %39 = select i1 %38, i32 -1, i32 %37
  %40 = icmp eq i64 %13, %10
  br i1 %40, label %56, label %41

41:                                               ; preds = %9, %35
  %42 = phi i64 [ 0, %9 ], [ %13, %35 ]
  %43 = phi i32 [ -1, %9 ], [ %39, %35 ]
  br label %44

44:                                               ; preds = %41, %44
  %45 = phi i64 [ %54, %44 ], [ %42, %41 ]
  %46 = phi i32 [ %53, %44 ], [ %43, %41 ]
  %47 = getelementptr inbounds nuw i32, ptr %5, i64 %45
  %48 = load i32, ptr %47, align 4, !tbaa !37
  %49 = getelementptr inbounds nuw i32, ptr %6, i64 %45
  %50 = load i32, ptr %49, align 4, !tbaa !37
  %51 = icmp sgt i32 %48, %50
  %52 = trunc nuw nsw i64 %45 to i32
  %53 = select i1 %51, i32 %46, i32 %52
  %54 = add nuw nsw i64 %45, 1
  %55 = icmp eq i64 %54, %10
  br i1 %55, label %56, label %44, !llvm.loop !102

56:                                               ; preds = %44, %35, %4
  %57 = phi i32 [ -1, %4 ], [ %39, %35 ], [ %53, %44 ]
  ret i32 %57
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE3$_5E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_5", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !83
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i32 @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE3$_4E9_M_invokeERKSt9_Any_dataOS0_S7_Oi"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !49
  %6 = load ptr, ptr %2, align 8, !tbaa !49
  %7 = load i32, ptr %3, align 4, !tbaa !37
  %8 = icmp sgt i32 %7, 0
  br i1 %8, label %9, label %23

9:                                                ; preds = %4
  %10 = zext nneg i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 0, %9 ], [ %21, %11 ]
  %13 = phi i32 [ -1, %9 ], [ %20, %11 ]
  %14 = getelementptr inbounds nuw float, ptr %5, i64 %12
  %15 = load float, ptr %14, align 4, !tbaa !51
  %16 = getelementptr inbounds nuw float, ptr %6, i64 %12
  %17 = load float, ptr %16, align 4, !tbaa !51
  %18 = fcmp ogt float %15, %17
  %19 = trunc nuw nsw i64 %12 to i32
  %20 = select i1 %18, i32 %13, i32 %19
  %21 = add nuw nsw i64 %12, 1
  %22 = icmp eq i64 %21, %10
  br i1 %22, label %23, label %11, !llvm.loop !103

23:                                               ; preds = %11, %4
  %24 = phi i32 [ -1, %4 ], [ %20, %11 ]
  ret i32 %24
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE3$_4E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_4", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !83
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i32 @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE3$_5E9_M_invokeERKSt9_Any_dataOS0_S7_Oi"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !49
  %6 = load ptr, ptr %2, align 8, !tbaa !49
  %7 = load i32, ptr %3, align 4, !tbaa !37
  %8 = icmp sgt i32 %7, 0
  br i1 %8, label %9, label %56

9:                                                ; preds = %4
  %10 = zext nneg i32 %7 to i64
  %11 = icmp ult i32 %7, 8
  br i1 %11, label %41, label %12

12:                                               ; preds = %9
  %13 = and i64 %10, 2147483640
  br label %14

14:                                               ; preds = %14, %12
  %15 = phi i64 [ 0, %12 ], [ %32, %14 ]
  %16 = phi <4 x i32> [ splat (i32 -2147483648), %12 ], [ %30, %14 ]
  %17 = phi <4 x i32> [ splat (i32 -2147483648), %12 ], [ %31, %14 ]
  %18 = phi <4 x i32> [ <i32 0, i32 1, i32 2, i32 3>, %12 ], [ %33, %14 ]
  %19 = add <4 x i32> %18, splat (i32 4)
  %20 = getelementptr inbounds nuw float, ptr %5, i64 %15
  %21 = getelementptr inbounds nuw i8, ptr %20, i64 16
  %22 = load <4 x float>, ptr %20, align 4, !tbaa !51
  %23 = load <4 x float>, ptr %21, align 4, !tbaa !51
  %24 = getelementptr inbounds nuw float, ptr %6, i64 %15
  %25 = getelementptr inbounds nuw i8, ptr %24, i64 16
  %26 = load <4 x float>, ptr %24, align 4, !tbaa !51
  %27 = load <4 x float>, ptr %25, align 4, !tbaa !51
  %28 = fcmp ogt <4 x float> %22, %26
  %29 = fcmp ogt <4 x float> %23, %27
  %30 = select <4 x i1> %28, <4 x i32> %16, <4 x i32> %18
  %31 = select <4 x i1> %29, <4 x i32> %17, <4 x i32> %19
  %32 = add nuw i64 %15, 8
  %33 = add <4 x i32> %18, splat (i32 8)
  %34 = icmp eq i64 %32, %13
  br i1 %34, label %35, label %14, !llvm.loop !104

35:                                               ; preds = %14
  %36 = tail call <4 x i32> @llvm.smax.v4i32(<4 x i32> %30, <4 x i32> %31)
  %37 = tail call i32 @llvm.vector.reduce.smax.v4i32(<4 x i32> %36)
  %38 = icmp eq i32 %37, -2147483648
  %39 = select i1 %38, i32 -1, i32 %37
  %40 = icmp eq i64 %13, %10
  br i1 %40, label %56, label %41

41:                                               ; preds = %9, %35
  %42 = phi i64 [ 0, %9 ], [ %13, %35 ]
  %43 = phi i32 [ -1, %9 ], [ %39, %35 ]
  br label %44

44:                                               ; preds = %41, %44
  %45 = phi i64 [ %54, %44 ], [ %42, %41 ]
  %46 = phi i32 [ %53, %44 ], [ %43, %41 ]
  %47 = getelementptr inbounds nuw float, ptr %5, i64 %45
  %48 = load float, ptr %47, align 4, !tbaa !51
  %49 = getelementptr inbounds nuw float, ptr %6, i64 %45
  %50 = load float, ptr %49, align 4, !tbaa !51
  %51 = fcmp ogt float %48, %50
  %52 = trunc nuw nsw i64 %45 to i32
  %53 = select i1 %51, i32 %46, i32 %52
  %54 = add nuw nsw i64 %45, 1
  %55 = icmp eq i64 %54, %10
  br i1 %55, label %56, label %44, !llvm.loop !105

56:                                               ; preds = %44, %35, %4
  %57 = phi i32 [ -1, %4 ], [ %39, %35 ], [ %53, %44 ]
  ret i32 %57
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE3$_5E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_5", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !83
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i16 @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE3$_6E9_M_invokeERKSt9_Any_dataOS0_S7_Os"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 2 captures(none) dereferenceable(2) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !65
  %6 = load ptr, ptr %2, align 8, !tbaa !65
  %7 = load i16, ptr %3, align 2, !tbaa !63
  %8 = icmp sgt i16 %7, 0
  br i1 %8, label %9, label %23

9:                                                ; preds = %4
  %10 = zext nneg i16 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 0, %9 ], [ %21, %11 ]
  %13 = phi i16 [ -1, %9 ], [ %20, %11 ]
  %14 = getelementptr inbounds nuw i16, ptr %5, i64 %12
  %15 = load i16, ptr %14, align 2, !tbaa !63
  %16 = getelementptr inbounds nuw i16, ptr %6, i64 %12
  %17 = load i16, ptr %16, align 2, !tbaa !63
  %18 = icmp sgt i16 %15, %17
  %19 = trunc nuw nsw i64 %12 to i16
  %20 = select i1 %18, i16 %13, i16 %19
  %21 = add nuw nsw i64 %12, 1
  %22 = icmp eq i64 %21, %10
  br i1 %22, label %23, label %11, !llvm.loop !106

23:                                               ; preds = %11, %4
  %24 = phi i16 [ -1, %4 ], [ %20, %11 ]
  ret i16 %24
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE3$_6E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_6", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !83
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i16 @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE3$_7E9_M_invokeERKSt9_Any_dataOS0_S7_Os"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 2 captures(none) dereferenceable(2) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !65
  %6 = load ptr, ptr %2, align 8, !tbaa !65
  %7 = load i16, ptr %3, align 2, !tbaa !63
  %8 = icmp sgt i16 %7, 0
  br i1 %8, label %9, label %91

9:                                                ; preds = %4
  %10 = zext nneg i16 %7 to i64
  %11 = icmp ult i16 %7, 4
  br i1 %11, label %76, label %12

12:                                               ; preds = %9
  %13 = icmp ult i16 %7, 16
  br i1 %13, label %46, label %14

14:                                               ; preds = %12
  %15 = and i64 %10, 32752
  br label %16

16:                                               ; preds = %16, %14
  %17 = phi i64 [ 0, %14 ], [ %34, %16 ]
  %18 = phi <8 x i16> [ splat (i16 -32768), %14 ], [ %32, %16 ]
  %19 = phi <8 x i16> [ splat (i16 -32768), %14 ], [ %33, %16 ]
  %20 = phi <8 x i16> [ <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, %14 ], [ %35, %16 ]
  %21 = add <8 x i16> %20, splat (i16 8)
  %22 = getelementptr inbounds nuw i16, ptr %5, i64 %17
  %23 = getelementptr inbounds nuw i8, ptr %22, i64 16
  %24 = load <8 x i16>, ptr %22, align 2, !tbaa !63
  %25 = load <8 x i16>, ptr %23, align 2, !tbaa !63
  %26 = getelementptr inbounds nuw i16, ptr %6, i64 %17
  %27 = getelementptr inbounds nuw i8, ptr %26, i64 16
  %28 = load <8 x i16>, ptr %26, align 2, !tbaa !63
  %29 = load <8 x i16>, ptr %27, align 2, !tbaa !63
  %30 = icmp sgt <8 x i16> %24, %28
  %31 = icmp sgt <8 x i16> %25, %29
  %32 = select <8 x i1> %30, <8 x i16> %18, <8 x i16> %20
  %33 = select <8 x i1> %31, <8 x i16> %19, <8 x i16> %21
  %34 = add nuw i64 %17, 16
  %35 = add <8 x i16> %20, splat (i16 16)
  %36 = icmp eq i64 %34, %15
  br i1 %36, label %37, label %16, !llvm.loop !107

37:                                               ; preds = %16
  %38 = tail call <8 x i16> @llvm.smax.v8i16(<8 x i16> %32, <8 x i16> %33)
  %39 = tail call i16 @llvm.vector.reduce.smax.v8i16(<8 x i16> %38)
  %40 = icmp eq i16 %39, -32768
  %41 = select i1 %40, i16 -1, i16 %39
  %42 = icmp eq i64 %15, %10
  br i1 %42, label %91, label %43

43:                                               ; preds = %37
  %44 = and i64 %10, 12
  %45 = icmp eq i64 %44, 0
  br i1 %45, label %76, label %46

46:                                               ; preds = %43, %12
  %47 = phi i64 [ %15, %43 ], [ 0, %12 ]
  %48 = phi i16 [ %41, %43 ], [ -1, %12 ]
  %49 = icmp eq i16 %48, -1
  %50 = select i1 %49, i16 -32768, i16 %48
  %51 = and i64 %10, 32764
  %52 = insertelement <4 x i16> poison, i16 %50, i64 0
  %53 = shufflevector <4 x i16> %52, <4 x i16> poison, <4 x i32> zeroinitializer
  %54 = trunc nuw nsw i64 %47 to i16
  %55 = insertelement <4 x i16> poison, i16 %54, i64 0
  %56 = shufflevector <4 x i16> %55, <4 x i16> poison, <4 x i32> zeroinitializer
  %57 = or disjoint <4 x i16> %56, <i16 0, i16 1, i16 2, i16 3>
  br label %58

58:                                               ; preds = %58, %46
  %59 = phi i64 [ %47, %46 ], [ %68, %58 ]
  %60 = phi <4 x i16> [ %53, %46 ], [ %67, %58 ]
  %61 = phi <4 x i16> [ %57, %46 ], [ %69, %58 ]
  %62 = getelementptr inbounds nuw i16, ptr %5, i64 %59
  %63 = load <4 x i16>, ptr %62, align 2, !tbaa !63
  %64 = getelementptr inbounds nuw i16, ptr %6, i64 %59
  %65 = load <4 x i16>, ptr %64, align 2, !tbaa !63
  %66 = icmp sgt <4 x i16> %63, %65
  %67 = select <4 x i1> %66, <4 x i16> %60, <4 x i16> %61
  %68 = add nuw i64 %59, 4
  %69 = add <4 x i16> %61, splat (i16 4)
  %70 = icmp eq i64 %68, %51
  br i1 %70, label %71, label %58, !llvm.loop !108

71:                                               ; preds = %58
  %72 = tail call i16 @llvm.vector.reduce.smax.v4i16(<4 x i16> %67)
  %73 = icmp eq i16 %72, -32768
  %74 = select i1 %73, i16 -1, i16 %72
  %75 = icmp eq i64 %51, %10
  br i1 %75, label %91, label %76

76:                                               ; preds = %43, %71, %9
  %77 = phi i64 [ 0, %9 ], [ %15, %43 ], [ %51, %71 ]
  %78 = phi i16 [ -1, %9 ], [ %41, %43 ], [ %74, %71 ]
  br label %79

79:                                               ; preds = %76, %79
  %80 = phi i64 [ %89, %79 ], [ %77, %76 ]
  %81 = phi i16 [ %88, %79 ], [ %78, %76 ]
  %82 = getelementptr inbounds nuw i16, ptr %5, i64 %80
  %83 = load i16, ptr %82, align 2, !tbaa !63
  %84 = getelementptr inbounds nuw i16, ptr %6, i64 %80
  %85 = load i16, ptr %84, align 2, !tbaa !63
  %86 = icmp sgt i16 %83, %85
  %87 = trunc nuw nsw i64 %80 to i16
  %88 = select i1 %86, i16 %81, i16 %87
  %89 = add nuw nsw i64 %80, 1
  %90 = icmp eq i64 %89, %10
  br i1 %90, label %91, label %79, !llvm.loop !109

91:                                               ; preds = %79, %37, %71, %4
  %92 = phi i16 [ -1, %4 ], [ %41, %37 ], [ %74, %71 ], [ %88, %79 ]
  ret i16 %92
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE3$_7E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_7", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !83
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i32 @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE3$_8E9_M_invokeERKSt9_Any_dataOS0_S7_Oi"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !39
  %6 = load ptr, ptr %2, align 8, !tbaa !39
  %7 = load i32, ptr %3, align 4, !tbaa !37
  %8 = icmp sgt i32 %7, 0
  br i1 %8, label %9, label %23

9:                                                ; preds = %4
  %10 = zext nneg i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 0, %9 ], [ %21, %11 ]
  %13 = phi i32 [ %7, %9 ], [ %20, %11 ]
  %14 = getelementptr inbounds nuw i32, ptr %5, i64 %12
  %15 = load i32, ptr %14, align 4, !tbaa !37
  %16 = getelementptr inbounds nuw i32, ptr %6, i64 %12
  %17 = load i32, ptr %16, align 4, !tbaa !37
  %18 = icmp sgt i32 %15, %17
  %19 = trunc nuw nsw i64 %12 to i32
  %20 = select i1 %18, i32 %19, i32 %13
  %21 = add nuw nsw i64 %12, 1
  %22 = icmp eq i64 %21, %10
  br i1 %22, label %23, label %11, !llvm.loop !110

23:                                               ; preds = %11, %4
  %24 = phi i32 [ %7, %4 ], [ %20, %11 ]
  ret i32 %24
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE3$_8E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_8", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !83
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i32 @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE3$_9E9_M_invokeERKSt9_Any_dataOS0_S7_Oi"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !39
  %6 = load ptr, ptr %2, align 8, !tbaa !39
  %7 = load i32, ptr %3, align 4, !tbaa !37
  %8 = icmp sgt i32 %7, 0
  br i1 %8, label %9, label %56

9:                                                ; preds = %4
  %10 = zext nneg i32 %7 to i64
  %11 = icmp ult i32 %7, 8
  br i1 %11, label %41, label %12

12:                                               ; preds = %9
  %13 = and i64 %10, 2147483640
  br label %14

14:                                               ; preds = %14, %12
  %15 = phi i64 [ 0, %12 ], [ %32, %14 ]
  %16 = phi <4 x i32> [ splat (i32 -2147483648), %12 ], [ %30, %14 ]
  %17 = phi <4 x i32> [ splat (i32 -2147483648), %12 ], [ %31, %14 ]
  %18 = phi <4 x i32> [ <i32 0, i32 1, i32 2, i32 3>, %12 ], [ %33, %14 ]
  %19 = add <4 x i32> %18, splat (i32 4)
  %20 = getelementptr inbounds nuw i32, ptr %5, i64 %15
  %21 = getelementptr inbounds nuw i8, ptr %20, i64 16
  %22 = load <4 x i32>, ptr %20, align 4, !tbaa !37
  %23 = load <4 x i32>, ptr %21, align 4, !tbaa !37
  %24 = getelementptr inbounds nuw i32, ptr %6, i64 %15
  %25 = getelementptr inbounds nuw i8, ptr %24, i64 16
  %26 = load <4 x i32>, ptr %24, align 4, !tbaa !37
  %27 = load <4 x i32>, ptr %25, align 4, !tbaa !37
  %28 = icmp sgt <4 x i32> %22, %26
  %29 = icmp sgt <4 x i32> %23, %27
  %30 = select <4 x i1> %28, <4 x i32> %18, <4 x i32> %16
  %31 = select <4 x i1> %29, <4 x i32> %19, <4 x i32> %17
  %32 = add nuw i64 %15, 8
  %33 = add <4 x i32> %18, splat (i32 8)
  %34 = icmp eq i64 %32, %13
  br i1 %34, label %35, label %14, !llvm.loop !111

35:                                               ; preds = %14
  %36 = tail call <4 x i32> @llvm.smax.v4i32(<4 x i32> %30, <4 x i32> %31)
  %37 = tail call i32 @llvm.vector.reduce.smax.v4i32(<4 x i32> %36)
  %38 = icmp eq i32 %37, -2147483648
  %39 = select i1 %38, i32 %7, i32 %37
  %40 = icmp eq i64 %13, %10
  br i1 %40, label %56, label %41

41:                                               ; preds = %9, %35
  %42 = phi i64 [ 0, %9 ], [ %13, %35 ]
  %43 = phi i32 [ %7, %9 ], [ %39, %35 ]
  br label %44

44:                                               ; preds = %41, %44
  %45 = phi i64 [ %54, %44 ], [ %42, %41 ]
  %46 = phi i32 [ %53, %44 ], [ %43, %41 ]
  %47 = getelementptr inbounds nuw i32, ptr %5, i64 %45
  %48 = load i32, ptr %47, align 4, !tbaa !37
  %49 = getelementptr inbounds nuw i32, ptr %6, i64 %45
  %50 = load i32, ptr %49, align 4, !tbaa !37
  %51 = icmp sgt i32 %48, %50
  %52 = trunc nuw nsw i64 %45 to i32
  %53 = select i1 %51, i32 %52, i32 %46
  %54 = add nuw nsw i64 %45, 1
  %55 = icmp eq i64 %54, %10
  br i1 %55, label %56, label %44, !llvm.loop !112

56:                                               ; preds = %44, %35, %4
  %57 = phi i32 [ %7, %4 ], [ %39, %35 ], [ %53, %44 ]
  ret i32 %57
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE3$_9E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_9", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !83
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i32 @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE3$_8E9_M_invokeERKSt9_Any_dataOS0_S7_Oi"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !49
  %6 = load ptr, ptr %2, align 8, !tbaa !49
  %7 = load i32, ptr %3, align 4, !tbaa !37
  %8 = icmp sgt i32 %7, 0
  br i1 %8, label %9, label %23

9:                                                ; preds = %4
  %10 = zext nneg i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 0, %9 ], [ %21, %11 ]
  %13 = phi i32 [ %7, %9 ], [ %20, %11 ]
  %14 = getelementptr inbounds nuw float, ptr %5, i64 %12
  %15 = load float, ptr %14, align 4, !tbaa !51
  %16 = getelementptr inbounds nuw float, ptr %6, i64 %12
  %17 = load float, ptr %16, align 4, !tbaa !51
  %18 = fcmp ogt float %15, %17
  %19 = trunc nuw nsw i64 %12 to i32
  %20 = select i1 %18, i32 %19, i32 %13
  %21 = add nuw nsw i64 %12, 1
  %22 = icmp eq i64 %21, %10
  br i1 %22, label %23, label %11, !llvm.loop !113

23:                                               ; preds = %11, %4
  %24 = phi i32 [ %7, %4 ], [ %20, %11 ]
  ret i32 %24
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE3$_8E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_8", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !83
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i32 @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE3$_9E9_M_invokeERKSt9_Any_dataOS0_S7_Oi"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !49
  %6 = load ptr, ptr %2, align 8, !tbaa !49
  %7 = load i32, ptr %3, align 4, !tbaa !37
  %8 = icmp sgt i32 %7, 0
  br i1 %8, label %9, label %56

9:                                                ; preds = %4
  %10 = zext nneg i32 %7 to i64
  %11 = icmp ult i32 %7, 8
  br i1 %11, label %41, label %12

12:                                               ; preds = %9
  %13 = and i64 %10, 2147483640
  br label %14

14:                                               ; preds = %14, %12
  %15 = phi i64 [ 0, %12 ], [ %32, %14 ]
  %16 = phi <4 x i32> [ splat (i32 -2147483648), %12 ], [ %30, %14 ]
  %17 = phi <4 x i32> [ splat (i32 -2147483648), %12 ], [ %31, %14 ]
  %18 = phi <4 x i32> [ <i32 0, i32 1, i32 2, i32 3>, %12 ], [ %33, %14 ]
  %19 = add <4 x i32> %18, splat (i32 4)
  %20 = getelementptr inbounds nuw float, ptr %5, i64 %15
  %21 = getelementptr inbounds nuw i8, ptr %20, i64 16
  %22 = load <4 x float>, ptr %20, align 4, !tbaa !51
  %23 = load <4 x float>, ptr %21, align 4, !tbaa !51
  %24 = getelementptr inbounds nuw float, ptr %6, i64 %15
  %25 = getelementptr inbounds nuw i8, ptr %24, i64 16
  %26 = load <4 x float>, ptr %24, align 4, !tbaa !51
  %27 = load <4 x float>, ptr %25, align 4, !tbaa !51
  %28 = fcmp ogt <4 x float> %22, %26
  %29 = fcmp ogt <4 x float> %23, %27
  %30 = select <4 x i1> %28, <4 x i32> %18, <4 x i32> %16
  %31 = select <4 x i1> %29, <4 x i32> %19, <4 x i32> %17
  %32 = add nuw i64 %15, 8
  %33 = add <4 x i32> %18, splat (i32 8)
  %34 = icmp eq i64 %32, %13
  br i1 %34, label %35, label %14, !llvm.loop !114

35:                                               ; preds = %14
  %36 = tail call <4 x i32> @llvm.smax.v4i32(<4 x i32> %30, <4 x i32> %31)
  %37 = tail call i32 @llvm.vector.reduce.smax.v4i32(<4 x i32> %36)
  %38 = icmp eq i32 %37, -2147483648
  %39 = select i1 %38, i32 %7, i32 %37
  %40 = icmp eq i64 %13, %10
  br i1 %40, label %56, label %41

41:                                               ; preds = %9, %35
  %42 = phi i64 [ 0, %9 ], [ %13, %35 ]
  %43 = phi i32 [ %7, %9 ], [ %39, %35 ]
  br label %44

44:                                               ; preds = %41, %44
  %45 = phi i64 [ %54, %44 ], [ %42, %41 ]
  %46 = phi i32 [ %53, %44 ], [ %43, %41 ]
  %47 = getelementptr inbounds nuw float, ptr %5, i64 %45
  %48 = load float, ptr %47, align 4, !tbaa !51
  %49 = getelementptr inbounds nuw float, ptr %6, i64 %45
  %50 = load float, ptr %49, align 4, !tbaa !51
  %51 = fcmp ogt float %48, %50
  %52 = trunc nuw nsw i64 %45 to i32
  %53 = select i1 %51, i32 %52, i32 %46
  %54 = add nuw nsw i64 %45, 1
  %55 = icmp eq i64 %54, %10
  br i1 %55, label %56, label %44, !llvm.loop !115

56:                                               ; preds = %44, %35, %4
  %57 = phi i32 [ %7, %4 ], [ %39, %35 ], [ %53, %44 ]
  ret i32 %57
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE3$_9E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_9", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !83
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i16 @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE4$_10E9_M_invokeERKSt9_Any_dataOS0_S7_Os"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 2 captures(none) dereferenceable(2) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !65
  %6 = load ptr, ptr %2, align 8, !tbaa !65
  %7 = load i16, ptr %3, align 2, !tbaa !63
  %8 = icmp sgt i16 %7, 0
  br i1 %8, label %9, label %23

9:                                                ; preds = %4
  %10 = zext nneg i16 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 0, %9 ], [ %21, %11 ]
  %13 = phi i16 [ %7, %9 ], [ %20, %11 ]
  %14 = getelementptr inbounds nuw i16, ptr %5, i64 %12
  %15 = load i16, ptr %14, align 2, !tbaa !63
  %16 = getelementptr inbounds nuw i16, ptr %6, i64 %12
  %17 = load i16, ptr %16, align 2, !tbaa !63
  %18 = icmp sgt i16 %15, %17
  %19 = trunc nuw nsw i64 %12 to i16
  %20 = select i1 %18, i16 %19, i16 %13
  %21 = add nuw nsw i64 %12, 1
  %22 = icmp eq i64 %21, %10
  br i1 %22, label %23, label %11, !llvm.loop !116

23:                                               ; preds = %11, %4
  %24 = phi i16 [ %7, %4 ], [ %20, %11 ]
  ret i16 %24
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE4$_10E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_10", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !83
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i16 @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE4$_11E9_M_invokeERKSt9_Any_dataOS0_S7_Os"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 2 captures(none) dereferenceable(2) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !65
  %6 = load ptr, ptr %2, align 8, !tbaa !65
  %7 = load i16, ptr %3, align 2, !tbaa !63
  %8 = freeze i16 %7
  %9 = icmp sgt i16 %8, 0
  br i1 %9, label %10, label %92

10:                                               ; preds = %4
  %11 = zext nneg i16 %8 to i64
  %12 = icmp ult i16 %8, 4
  br i1 %12, label %77, label %13

13:                                               ; preds = %10
  %14 = icmp ult i16 %8, 16
  br i1 %14, label %47, label %15

15:                                               ; preds = %13
  %16 = and i64 %11, 32752
  br label %17

17:                                               ; preds = %17, %15
  %18 = phi i64 [ 0, %15 ], [ %35, %17 ]
  %19 = phi <8 x i16> [ splat (i16 -32768), %15 ], [ %33, %17 ]
  %20 = phi <8 x i16> [ splat (i16 -32768), %15 ], [ %34, %17 ]
  %21 = phi <8 x i16> [ <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, %15 ], [ %36, %17 ]
  %22 = add <8 x i16> %21, splat (i16 8)
  %23 = getelementptr inbounds nuw i16, ptr %5, i64 %18
  %24 = getelementptr inbounds nuw i8, ptr %23, i64 16
  %25 = load <8 x i16>, ptr %23, align 2, !tbaa !63
  %26 = load <8 x i16>, ptr %24, align 2, !tbaa !63
  %27 = getelementptr inbounds nuw i16, ptr %6, i64 %18
  %28 = getelementptr inbounds nuw i8, ptr %27, i64 16
  %29 = load <8 x i16>, ptr %27, align 2, !tbaa !63
  %30 = load <8 x i16>, ptr %28, align 2, !tbaa !63
  %31 = icmp sgt <8 x i16> %25, %29
  %32 = icmp sgt <8 x i16> %26, %30
  %33 = select <8 x i1> %31, <8 x i16> %21, <8 x i16> %19
  %34 = select <8 x i1> %32, <8 x i16> %22, <8 x i16> %20
  %35 = add nuw i64 %18, 16
  %36 = add <8 x i16> %21, splat (i16 16)
  %37 = icmp eq i64 %35, %16
  br i1 %37, label %38, label %17, !llvm.loop !117

38:                                               ; preds = %17
  %39 = tail call <8 x i16> @llvm.smax.v8i16(<8 x i16> %33, <8 x i16> %34)
  %40 = tail call i16 @llvm.vector.reduce.smax.v8i16(<8 x i16> %39)
  %41 = icmp eq i16 %40, -32768
  %42 = select i1 %41, i16 %8, i16 %40
  %43 = icmp eq i64 %16, %11
  br i1 %43, label %92, label %44

44:                                               ; preds = %38
  %45 = and i64 %11, 12
  %46 = icmp eq i64 %45, 0
  br i1 %46, label %77, label %47

47:                                               ; preds = %44, %13
  %48 = phi i64 [ %16, %44 ], [ 0, %13 ]
  %49 = phi i16 [ %42, %44 ], [ %8, %13 ]
  %50 = icmp eq i16 %49, %8
  %51 = select i1 %50, i16 -32768, i16 %49
  %52 = and i64 %11, 32764
  %53 = insertelement <4 x i16> poison, i16 %51, i64 0
  %54 = shufflevector <4 x i16> %53, <4 x i16> poison, <4 x i32> zeroinitializer
  %55 = trunc nuw nsw i64 %48 to i16
  %56 = insertelement <4 x i16> poison, i16 %55, i64 0
  %57 = shufflevector <4 x i16> %56, <4 x i16> poison, <4 x i32> zeroinitializer
  %58 = or disjoint <4 x i16> %57, <i16 0, i16 1, i16 2, i16 3>
  br label %59

59:                                               ; preds = %59, %47
  %60 = phi i64 [ %48, %47 ], [ %69, %59 ]
  %61 = phi <4 x i16> [ %54, %47 ], [ %68, %59 ]
  %62 = phi <4 x i16> [ %58, %47 ], [ %70, %59 ]
  %63 = getelementptr inbounds nuw i16, ptr %5, i64 %60
  %64 = load <4 x i16>, ptr %63, align 2, !tbaa !63
  %65 = getelementptr inbounds nuw i16, ptr %6, i64 %60
  %66 = load <4 x i16>, ptr %65, align 2, !tbaa !63
  %67 = icmp sgt <4 x i16> %64, %66
  %68 = select <4 x i1> %67, <4 x i16> %62, <4 x i16> %61
  %69 = add nuw i64 %60, 4
  %70 = add <4 x i16> %62, splat (i16 4)
  %71 = icmp eq i64 %69, %52
  br i1 %71, label %72, label %59, !llvm.loop !118

72:                                               ; preds = %59
  %73 = tail call i16 @llvm.vector.reduce.smax.v4i16(<4 x i16> %68)
  %74 = icmp eq i16 %73, -32768
  %75 = select i1 %74, i16 %8, i16 %73
  %76 = icmp eq i64 %52, %11
  br i1 %76, label %92, label %77

77:                                               ; preds = %44, %72, %10
  %78 = phi i64 [ 0, %10 ], [ %16, %44 ], [ %52, %72 ]
  %79 = phi i16 [ %8, %10 ], [ %42, %44 ], [ %75, %72 ]
  br label %80

80:                                               ; preds = %77, %80
  %81 = phi i64 [ %90, %80 ], [ %78, %77 ]
  %82 = phi i16 [ %89, %80 ], [ %79, %77 ]
  %83 = getelementptr inbounds nuw i16, ptr %5, i64 %81
  %84 = load i16, ptr %83, align 2, !tbaa !63
  %85 = getelementptr inbounds nuw i16, ptr %6, i64 %81
  %86 = load i16, ptr %85, align 2, !tbaa !63
  %87 = icmp sgt i16 %84, %86
  %88 = trunc nuw nsw i64 %81 to i16
  %89 = select i1 %87, i16 %88, i16 %82
  %90 = add nuw nsw i64 %81, 1
  %91 = icmp eq i64 %90, %11
  br i1 %91, label %92, label %80, !llvm.loop !119

92:                                               ; preds = %80, %38, %72, %4
  %93 = phi i16 [ %8, %4 ], [ %42, %38 ], [ %75, %72 ], [ %89, %80 ]
  ret i16 %93
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE4$_11E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_11", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !83
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i32 @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE4$_12E9_M_invokeERKSt9_Any_dataOS0_S7_Oi"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !39
  %6 = load ptr, ptr %2, align 8, !tbaa !39
  %7 = load i32, ptr %3, align 4, !tbaa !37
  %8 = icmp sgt i32 %7, 0
  br i1 %8, label %9, label %23

9:                                                ; preds = %4
  %10 = zext nneg i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 0, %9 ], [ %21, %11 ]
  %13 = phi i32 [ -1, %9 ], [ %20, %11 ]
  %14 = getelementptr inbounds nuw i32, ptr %5, i64 %12
  %15 = load i32, ptr %14, align 4, !tbaa !37
  %16 = getelementptr inbounds nuw i32, ptr %6, i64 %12
  %17 = load i32, ptr %16, align 4, !tbaa !37
  %18 = icmp sgt i32 %15, %17
  %19 = trunc nuw nsw i64 %12 to i32
  %20 = select i1 %18, i32 %19, i32 %13
  %21 = add nuw nsw i64 %12, 2
  %22 = icmp samesign ult i64 %21, %10
  br i1 %22, label %11, label %23, !llvm.loop !120

23:                                               ; preds = %11, %4
  %24 = phi i32 [ -1, %4 ], [ %20, %11 ]
  ret i32 %24
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE4$_12E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_12", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !83
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i32 @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE4$_13E9_M_invokeERKSt9_Any_dataOS0_S7_Oi"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !39
  %6 = load ptr, ptr %2, align 8, !tbaa !39
  %7 = load i32, ptr %3, align 4, !tbaa !37
  %8 = icmp sgt i32 %7, 0
  br i1 %8, label %9, label %23

9:                                                ; preds = %4
  %10 = zext nneg i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 0, %9 ], [ %21, %11 ]
  %13 = phi i32 [ -1, %9 ], [ %20, %11 ]
  %14 = getelementptr inbounds nuw i32, ptr %5, i64 %12
  %15 = load i32, ptr %14, align 4, !tbaa !37
  %16 = getelementptr inbounds nuw i32, ptr %6, i64 %12
  %17 = load i32, ptr %16, align 4, !tbaa !37
  %18 = icmp sgt i32 %15, %17
  %19 = trunc nuw nsw i64 %12 to i32
  %20 = select i1 %18, i32 %19, i32 %13
  %21 = add nuw nsw i64 %12, 2
  %22 = icmp samesign ult i64 %21, %10
  br i1 %22, label %11, label %23, !llvm.loop !121

23:                                               ; preds = %11, %4
  %24 = phi i32 [ -1, %4 ], [ %20, %11 ]
  ret i32 %24
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE4$_13E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_13", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !83
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i32 @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE4$_12E9_M_invokeERKSt9_Any_dataOS0_S7_Oi"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !49
  %6 = load ptr, ptr %2, align 8, !tbaa !49
  %7 = load i32, ptr %3, align 4, !tbaa !37
  %8 = icmp sgt i32 %7, 0
  br i1 %8, label %9, label %23

9:                                                ; preds = %4
  %10 = zext nneg i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 0, %9 ], [ %21, %11 ]
  %13 = phi i32 [ -1, %9 ], [ %20, %11 ]
  %14 = getelementptr inbounds nuw float, ptr %5, i64 %12
  %15 = load float, ptr %14, align 4, !tbaa !51
  %16 = getelementptr inbounds nuw float, ptr %6, i64 %12
  %17 = load float, ptr %16, align 4, !tbaa !51
  %18 = fcmp ogt float %15, %17
  %19 = trunc nuw nsw i64 %12 to i32
  %20 = select i1 %18, i32 %19, i32 %13
  %21 = add nuw nsw i64 %12, 2
  %22 = icmp samesign ult i64 %21, %10
  br i1 %22, label %11, label %23, !llvm.loop !123

23:                                               ; preds = %11, %4
  %24 = phi i32 [ -1, %4 ], [ %20, %11 ]
  ret i32 %24
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE4$_12E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_12", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !83
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i32 @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE4$_13E9_M_invokeERKSt9_Any_dataOS0_S7_Oi"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !49
  %6 = load ptr, ptr %2, align 8, !tbaa !49
  %7 = load i32, ptr %3, align 4, !tbaa !37
  %8 = icmp sgt i32 %7, 0
  br i1 %8, label %9, label %23

9:                                                ; preds = %4
  %10 = zext nneg i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 0, %9 ], [ %21, %11 ]
  %13 = phi i32 [ -1, %9 ], [ %20, %11 ]
  %14 = getelementptr inbounds nuw float, ptr %5, i64 %12
  %15 = load float, ptr %14, align 4, !tbaa !51
  %16 = getelementptr inbounds nuw float, ptr %6, i64 %12
  %17 = load float, ptr %16, align 4, !tbaa !51
  %18 = fcmp ogt float %15, %17
  %19 = trunc nuw nsw i64 %12 to i32
  %20 = select i1 %18, i32 %19, i32 %13
  %21 = add nuw nsw i64 %12, 2
  %22 = icmp samesign ult i64 %21, %10
  br i1 %22, label %11, label %23, !llvm.loop !124

23:                                               ; preds = %11, %4
  %24 = phi i32 [ -1, %4 ], [ %20, %11 ]
  ret i32 %24
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE4$_13E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_13", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !83
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i16 @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE4$_14E9_M_invokeERKSt9_Any_dataOS0_S7_Os"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 2 captures(none) dereferenceable(2) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !65
  %6 = load ptr, ptr %2, align 8, !tbaa !65
  %7 = load i16, ptr %3, align 2, !tbaa !63
  %8 = icmp sgt i16 %7, 0
  br i1 %8, label %9, label %23

9:                                                ; preds = %4
  %10 = zext nneg i16 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 0, %9 ], [ %21, %11 ]
  %13 = phi i16 [ -1, %9 ], [ %20, %11 ]
  %14 = getelementptr inbounds nuw i16, ptr %5, i64 %12
  %15 = load i16, ptr %14, align 2, !tbaa !63
  %16 = getelementptr inbounds nuw i16, ptr %6, i64 %12
  %17 = load i16, ptr %16, align 2, !tbaa !63
  %18 = icmp sgt i16 %15, %17
  %19 = trunc nuw nsw i64 %12 to i16
  %20 = select i1 %18, i16 %19, i16 %13
  %21 = add nuw nsw i64 %12, 2
  %22 = icmp samesign ult i64 %21, %10
  br i1 %22, label %11, label %23, !llvm.loop !125

23:                                               ; preds = %11, %4
  %24 = phi i16 [ -1, %4 ], [ %20, %11 ]
  ret i16 %24
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE4$_14E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_14", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !83
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i16 @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE4$_15E9_M_invokeERKSt9_Any_dataOS0_S7_Os"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 2 captures(none) dereferenceable(2) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !65
  %6 = load ptr, ptr %2, align 8, !tbaa !65
  %7 = load i16, ptr %3, align 2, !tbaa !63
  %8 = icmp sgt i16 %7, 0
  br i1 %8, label %9, label %23

9:                                                ; preds = %4
  %10 = zext nneg i16 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 0, %9 ], [ %21, %11 ]
  %13 = phi i16 [ -1, %9 ], [ %20, %11 ]
  %14 = getelementptr inbounds nuw i16, ptr %5, i64 %12
  %15 = load i16, ptr %14, align 2, !tbaa !63
  %16 = getelementptr inbounds nuw i16, ptr %6, i64 %12
  %17 = load i16, ptr %16, align 2, !tbaa !63
  %18 = icmp sgt i16 %15, %17
  %19 = trunc nuw nsw i64 %12 to i16
  %20 = select i1 %18, i16 %19, i16 %13
  %21 = add nuw nsw i64 %12, 2
  %22 = icmp samesign ult i64 %21, %10
  br i1 %22, label %11, label %23, !llvm.loop !126

23:                                               ; preds = %11, %4
  %24 = phi i16 [ -1, %4 ], [ %20, %11 ]
  ret i16 %24
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE4$_15E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_15", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !83
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef range(i32 -1, -2147483648) i32 @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE4$_16E9_M_invokeERKSt9_Any_dataOS0_S7_Oi"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !39
  %6 = load ptr, ptr %2, align 8, !tbaa !39
  %7 = load i32, ptr %3, align 4, !tbaa !37
  %8 = icmp sgt i32 %7, 0
  br i1 %8, label %9, label %23

9:                                                ; preds = %4
  %10 = zext nneg i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ %10, %9 ], [ %21, %11 ]
  %13 = phi i32 [ -1, %9 ], [ %20, %11 ]
  %14 = getelementptr inbounds nuw i32, ptr %5, i64 %12
  %15 = load i32, ptr %14, align 4, !tbaa !37
  %16 = getelementptr inbounds nuw i32, ptr %6, i64 %12
  %17 = load i32, ptr %16, align 4, !tbaa !37
  %18 = icmp sgt i32 %15, %17
  %19 = trunc nuw nsw i64 %12 to i32
  %20 = select i1 %18, i32 %19, i32 %13
  %21 = add nsw i64 %12, -1
  %22 = icmp samesign ugt i64 %12, 1
  br i1 %22, label %11, label %23, !llvm.loop !127

23:                                               ; preds = %11, %4
  %24 = phi i32 [ -1, %4 ], [ %20, %11 ]
  ret i32 %24
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE4$_16E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_16", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !83
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef range(i32 -1, -2147483648) i32 @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE4$_17E9_M_invokeERKSt9_Any_dataOS0_S7_Oi"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !39
  %6 = load ptr, ptr %2, align 8, !tbaa !39
  %7 = load i32, ptr %3, align 4, !tbaa !37
  %8 = icmp sgt i32 %7, 0
  br i1 %8, label %9, label %23

9:                                                ; preds = %4
  %10 = zext nneg i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ %10, %9 ], [ %21, %11 ]
  %13 = phi i32 [ -1, %9 ], [ %20, %11 ]
  %14 = getelementptr inbounds nuw i32, ptr %5, i64 %12
  %15 = load i32, ptr %14, align 4, !tbaa !37
  %16 = getelementptr inbounds nuw i32, ptr %6, i64 %12
  %17 = load i32, ptr %16, align 4, !tbaa !37
  %18 = icmp sgt i32 %15, %17
  %19 = trunc nuw nsw i64 %12 to i32
  %20 = select i1 %18, i32 %19, i32 %13
  %21 = add nsw i64 %12, -1
  %22 = icmp samesign ugt i64 %12, 1
  br i1 %22, label %11, label %23, !llvm.loop !128

23:                                               ; preds = %11, %4
  %24 = phi i32 [ -1, %4 ], [ %20, %11 ]
  ret i32 %24
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE4$_17E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_17", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !83
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef range(i32 -1, -2147483648) i32 @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE4$_16E9_M_invokeERKSt9_Any_dataOS0_S7_Oi"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !49
  %6 = load ptr, ptr %2, align 8, !tbaa !49
  %7 = load i32, ptr %3, align 4, !tbaa !37
  %8 = icmp sgt i32 %7, 0
  br i1 %8, label %9, label %23

9:                                                ; preds = %4
  %10 = zext nneg i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ %10, %9 ], [ %21, %11 ]
  %13 = phi i32 [ -1, %9 ], [ %20, %11 ]
  %14 = getelementptr inbounds nuw float, ptr %5, i64 %12
  %15 = load float, ptr %14, align 4, !tbaa !51
  %16 = getelementptr inbounds nuw float, ptr %6, i64 %12
  %17 = load float, ptr %16, align 4, !tbaa !51
  %18 = fcmp ogt float %15, %17
  %19 = trunc nuw nsw i64 %12 to i32
  %20 = select i1 %18, i32 %19, i32 %13
  %21 = add nsw i64 %12, -1
  %22 = icmp samesign ugt i64 %12, 1
  br i1 %22, label %11, label %23, !llvm.loop !129

23:                                               ; preds = %11, %4
  %24 = phi i32 [ -1, %4 ], [ %20, %11 ]
  ret i32 %24
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE4$_16E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_16", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !83
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef range(i32 -1, -2147483648) i32 @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE4$_17E9_M_invokeERKSt9_Any_dataOS0_S7_Oi"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !49
  %6 = load ptr, ptr %2, align 8, !tbaa !49
  %7 = load i32, ptr %3, align 4, !tbaa !37
  %8 = icmp sgt i32 %7, 0
  br i1 %8, label %9, label %23

9:                                                ; preds = %4
  %10 = zext nneg i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ %10, %9 ], [ %21, %11 ]
  %13 = phi i32 [ -1, %9 ], [ %20, %11 ]
  %14 = getelementptr inbounds nuw float, ptr %5, i64 %12
  %15 = load float, ptr %14, align 4, !tbaa !51
  %16 = getelementptr inbounds nuw float, ptr %6, i64 %12
  %17 = load float, ptr %16, align 4, !tbaa !51
  %18 = fcmp ogt float %15, %17
  %19 = trunc nuw nsw i64 %12 to i32
  %20 = select i1 %18, i32 %19, i32 %13
  %21 = add nsw i64 %12, -1
  %22 = icmp samesign ugt i64 %12, 1
  br i1 %22, label %11, label %23, !llvm.loop !130

23:                                               ; preds = %11, %4
  %24 = phi i32 [ -1, %4 ], [ %20, %11 ]
  ret i32 %24
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE4$_17E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_17", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !83
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i16 @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE4$_18E9_M_invokeERKSt9_Any_dataOS0_S7_Os"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 2 captures(none) dereferenceable(2) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !65
  %6 = load ptr, ptr %2, align 8, !tbaa !65
  %7 = load i16, ptr %3, align 2, !tbaa !63
  %8 = icmp sgt i16 %7, 0
  br i1 %8, label %9, label %23

9:                                                ; preds = %4
  %10 = zext nneg i16 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ %10, %9 ], [ %21, %11 ]
  %13 = phi i16 [ -1, %9 ], [ %20, %11 ]
  %14 = getelementptr inbounds nuw i16, ptr %5, i64 %12
  %15 = load i16, ptr %14, align 2, !tbaa !63
  %16 = getelementptr inbounds nuw i16, ptr %6, i64 %12
  %17 = load i16, ptr %16, align 2, !tbaa !63
  %18 = icmp sgt i16 %15, %17
  %19 = trunc nuw nsw i64 %12 to i16
  %20 = select i1 %18, i16 %19, i16 %13
  %21 = add nsw i64 %12, -1
  %22 = icmp samesign ugt i64 %12, 1
  br i1 %22, label %11, label %23, !llvm.loop !131

23:                                               ; preds = %11, %4
  %24 = phi i16 [ -1, %4 ], [ %20, %11 ]
  ret i16 %24
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE4$_18E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_18", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !83
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i16 @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE4$_19E9_M_invokeERKSt9_Any_dataOS0_S7_Os"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 2 captures(none) dereferenceable(2) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !65
  %6 = load ptr, ptr %2, align 8, !tbaa !65
  %7 = load i16, ptr %3, align 2, !tbaa !63
  %8 = icmp sgt i16 %7, 0
  br i1 %8, label %9, label %23

9:                                                ; preds = %4
  %10 = zext nneg i16 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ %10, %9 ], [ %21, %11 ]
  %13 = phi i16 [ -1, %9 ], [ %20, %11 ]
  %14 = getelementptr inbounds nuw i16, ptr %5, i64 %12
  %15 = load i16, ptr %14, align 2, !tbaa !63
  %16 = getelementptr inbounds nuw i16, ptr %6, i64 %12
  %17 = load i16, ptr %16, align 2, !tbaa !63
  %18 = icmp sgt i16 %15, %17
  %19 = trunc nuw nsw i64 %12 to i16
  %20 = select i1 %18, i16 %19, i16 %13
  %21 = add nsw i64 %12, -1
  %22 = icmp samesign ugt i64 %12, 1
  br i1 %22, label %11, label %23, !llvm.loop !132

23:                                               ; preds = %11, %4
  %24 = phi i16 [ -1, %4 ], [ %20, %11 ]
  ret i16 %24
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE4$_19E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_19", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !83
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i32 @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE4$_20E9_M_invokeERKSt9_Any_dataOS0_S7_Oi"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !39
  %6 = load ptr, ptr %2, align 8, !tbaa !39
  %7 = load i32, ptr %3, align 4, !tbaa !37
  %8 = icmp sgt i32 %7, 3
  br i1 %8, label %9, label %23

9:                                                ; preds = %4
  %10 = zext nneg i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 3, %9 ], [ %21, %11 ]
  %13 = phi i32 [ -1, %9 ], [ %20, %11 ]
  %14 = getelementptr inbounds nuw i32, ptr %5, i64 %12
  %15 = load i32, ptr %14, align 4, !tbaa !37
  %16 = getelementptr inbounds nuw i32, ptr %6, i64 %12
  %17 = load i32, ptr %16, align 4, !tbaa !37
  %18 = icmp sgt i32 %15, %17
  %19 = trunc nuw nsw i64 %12 to i32
  %20 = select i1 %18, i32 %19, i32 %13
  %21 = add nuw nsw i64 %12, 1
  %22 = icmp eq i64 %21, %10
  br i1 %22, label %23, label %11, !llvm.loop !133

23:                                               ; preds = %11, %4
  %24 = phi i32 [ -1, %4 ], [ %20, %11 ]
  ret i32 %24
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE4$_20E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_20", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !83
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i32 @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE4$_21E9_M_invokeERKSt9_Any_dataOS0_S7_Oi"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !39
  %6 = load ptr, ptr %2, align 8, !tbaa !39
  %7 = load i32, ptr %3, align 4, !tbaa !37
  %8 = icmp sgt i32 %7, 3
  br i1 %8, label %9, label %59

9:                                                ; preds = %4
  %10 = zext nneg i32 %7 to i64
  %11 = add nsw i64 %10, -3
  %12 = icmp ult i64 %11, 8
  br i1 %12, label %44, label %13

13:                                               ; preds = %9
  %14 = and i64 %11, -8
  %15 = or disjoint i64 %14, 3
  br label %16

16:                                               ; preds = %16, %13
  %17 = phi i64 [ 0, %13 ], [ %35, %16 ]
  %18 = phi <4 x i32> [ splat (i32 -2147483648), %13 ], [ %33, %16 ]
  %19 = phi <4 x i32> [ splat (i32 -2147483648), %13 ], [ %34, %16 ]
  %20 = phi <4 x i32> [ <i32 3, i32 4, i32 5, i32 6>, %13 ], [ %36, %16 ]
  %21 = add <4 x i32> %20, splat (i32 4)
  %22 = or disjoint i64 %17, 3
  %23 = getelementptr inbounds nuw i32, ptr %5, i64 %22
  %24 = getelementptr inbounds nuw i8, ptr %23, i64 16
  %25 = load <4 x i32>, ptr %23, align 4, !tbaa !37
  %26 = load <4 x i32>, ptr %24, align 4, !tbaa !37
  %27 = getelementptr inbounds nuw i32, ptr %6, i64 %22
  %28 = getelementptr inbounds nuw i8, ptr %27, i64 16
  %29 = load <4 x i32>, ptr %27, align 4, !tbaa !37
  %30 = load <4 x i32>, ptr %28, align 4, !tbaa !37
  %31 = icmp sgt <4 x i32> %25, %29
  %32 = icmp sgt <4 x i32> %26, %30
  %33 = select <4 x i1> %31, <4 x i32> %20, <4 x i32> %18
  %34 = select <4 x i1> %32, <4 x i32> %21, <4 x i32> %19
  %35 = add nuw i64 %17, 8
  %36 = add <4 x i32> %20, splat (i32 8)
  %37 = icmp eq i64 %35, %14
  br i1 %37, label %38, label %16, !llvm.loop !134

38:                                               ; preds = %16
  %39 = tail call <4 x i32> @llvm.smax.v4i32(<4 x i32> %33, <4 x i32> %34)
  %40 = tail call i32 @llvm.vector.reduce.smax.v4i32(<4 x i32> %39)
  %41 = icmp eq i32 %40, -2147483648
  %42 = select i1 %41, i32 -1, i32 %40
  %43 = icmp eq i64 %11, %14
  br i1 %43, label %59, label %44

44:                                               ; preds = %9, %38
  %45 = phi i64 [ 3, %9 ], [ %15, %38 ]
  %46 = phi i32 [ -1, %9 ], [ %42, %38 ]
  br label %47

47:                                               ; preds = %44, %47
  %48 = phi i64 [ %57, %47 ], [ %45, %44 ]
  %49 = phi i32 [ %56, %47 ], [ %46, %44 ]
  %50 = getelementptr inbounds nuw i32, ptr %5, i64 %48
  %51 = load i32, ptr %50, align 4, !tbaa !37
  %52 = getelementptr inbounds nuw i32, ptr %6, i64 %48
  %53 = load i32, ptr %52, align 4, !tbaa !37
  %54 = icmp sgt i32 %51, %53
  %55 = trunc nuw nsw i64 %48 to i32
  %56 = select i1 %54, i32 %55, i32 %49
  %57 = add nuw nsw i64 %48, 1
  %58 = icmp eq i64 %57, %10
  br i1 %58, label %59, label %47, !llvm.loop !135

59:                                               ; preds = %47, %38, %4
  %60 = phi i32 [ -1, %4 ], [ %42, %38 ], [ %56, %47 ]
  ret i32 %60
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE4$_21E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_21", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !83
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i32 @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE4$_20E9_M_invokeERKSt9_Any_dataOS0_S7_Oi"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !49
  %6 = load ptr, ptr %2, align 8, !tbaa !49
  %7 = load i32, ptr %3, align 4, !tbaa !37
  %8 = icmp sgt i32 %7, 3
  br i1 %8, label %9, label %23

9:                                                ; preds = %4
  %10 = zext nneg i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 3, %9 ], [ %21, %11 ]
  %13 = phi i32 [ -1, %9 ], [ %20, %11 ]
  %14 = getelementptr inbounds nuw float, ptr %5, i64 %12
  %15 = load float, ptr %14, align 4, !tbaa !51
  %16 = getelementptr inbounds nuw float, ptr %6, i64 %12
  %17 = load float, ptr %16, align 4, !tbaa !51
  %18 = fcmp ogt float %15, %17
  %19 = trunc nuw nsw i64 %12 to i32
  %20 = select i1 %18, i32 %19, i32 %13
  %21 = add nuw nsw i64 %12, 1
  %22 = icmp eq i64 %21, %10
  br i1 %22, label %23, label %11, !llvm.loop !136

23:                                               ; preds = %11, %4
  %24 = phi i32 [ -1, %4 ], [ %20, %11 ]
  ret i32 %24
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE4$_20E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_20", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !83
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i32 @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE4$_21E9_M_invokeERKSt9_Any_dataOS0_S7_Oi"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !49
  %6 = load ptr, ptr %2, align 8, !tbaa !49
  %7 = load i32, ptr %3, align 4, !tbaa !37
  %8 = icmp sgt i32 %7, 3
  br i1 %8, label %9, label %59

9:                                                ; preds = %4
  %10 = zext nneg i32 %7 to i64
  %11 = add nsw i64 %10, -3
  %12 = icmp ult i64 %11, 8
  br i1 %12, label %44, label %13

13:                                               ; preds = %9
  %14 = and i64 %11, -8
  %15 = or disjoint i64 %14, 3
  br label %16

16:                                               ; preds = %16, %13
  %17 = phi i64 [ 0, %13 ], [ %35, %16 ]
  %18 = phi <4 x i32> [ splat (i32 -2147483648), %13 ], [ %33, %16 ]
  %19 = phi <4 x i32> [ splat (i32 -2147483648), %13 ], [ %34, %16 ]
  %20 = phi <4 x i32> [ <i32 3, i32 4, i32 5, i32 6>, %13 ], [ %36, %16 ]
  %21 = add <4 x i32> %20, splat (i32 4)
  %22 = or disjoint i64 %17, 3
  %23 = getelementptr inbounds nuw float, ptr %5, i64 %22
  %24 = getelementptr inbounds nuw i8, ptr %23, i64 16
  %25 = load <4 x float>, ptr %23, align 4, !tbaa !51
  %26 = load <4 x float>, ptr %24, align 4, !tbaa !51
  %27 = getelementptr inbounds nuw float, ptr %6, i64 %22
  %28 = getelementptr inbounds nuw i8, ptr %27, i64 16
  %29 = load <4 x float>, ptr %27, align 4, !tbaa !51
  %30 = load <4 x float>, ptr %28, align 4, !tbaa !51
  %31 = fcmp ogt <4 x float> %25, %29
  %32 = fcmp ogt <4 x float> %26, %30
  %33 = select <4 x i1> %31, <4 x i32> %20, <4 x i32> %18
  %34 = select <4 x i1> %32, <4 x i32> %21, <4 x i32> %19
  %35 = add nuw i64 %17, 8
  %36 = add <4 x i32> %20, splat (i32 8)
  %37 = icmp eq i64 %35, %14
  br i1 %37, label %38, label %16, !llvm.loop !137

38:                                               ; preds = %16
  %39 = tail call <4 x i32> @llvm.smax.v4i32(<4 x i32> %33, <4 x i32> %34)
  %40 = tail call i32 @llvm.vector.reduce.smax.v4i32(<4 x i32> %39)
  %41 = icmp eq i32 %40, -2147483648
  %42 = select i1 %41, i32 -1, i32 %40
  %43 = icmp eq i64 %11, %14
  br i1 %43, label %59, label %44

44:                                               ; preds = %9, %38
  %45 = phi i64 [ 3, %9 ], [ %15, %38 ]
  %46 = phi i32 [ -1, %9 ], [ %42, %38 ]
  br label %47

47:                                               ; preds = %44, %47
  %48 = phi i64 [ %57, %47 ], [ %45, %44 ]
  %49 = phi i32 [ %56, %47 ], [ %46, %44 ]
  %50 = getelementptr inbounds nuw float, ptr %5, i64 %48
  %51 = load float, ptr %50, align 4, !tbaa !51
  %52 = getelementptr inbounds nuw float, ptr %6, i64 %48
  %53 = load float, ptr %52, align 4, !tbaa !51
  %54 = fcmp ogt float %51, %53
  %55 = trunc nuw nsw i64 %48 to i32
  %56 = select i1 %54, i32 %55, i32 %49
  %57 = add nuw nsw i64 %48, 1
  %58 = icmp eq i64 %57, %10
  br i1 %58, label %59, label %47, !llvm.loop !138

59:                                               ; preds = %47, %38, %4
  %60 = phi i32 [ -1, %4 ], [ %42, %38 ], [ %56, %47 ]
  ret i32 %60
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE4$_21E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_21", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !83
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i16 @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE4$_22E9_M_invokeERKSt9_Any_dataOS0_S7_Os"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 2 captures(none) dereferenceable(2) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !65
  %6 = load ptr, ptr %2, align 8, !tbaa !65
  %7 = load i16, ptr %3, align 2, !tbaa !63
  %8 = icmp sgt i16 %7, 3
  br i1 %8, label %9, label %23

9:                                                ; preds = %4
  %10 = zext nneg i16 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 3, %9 ], [ %21, %11 ]
  %13 = phi i16 [ -1, %9 ], [ %20, %11 ]
  %14 = getelementptr inbounds nuw i16, ptr %5, i64 %12
  %15 = load i16, ptr %14, align 2, !tbaa !63
  %16 = getelementptr inbounds nuw i16, ptr %6, i64 %12
  %17 = load i16, ptr %16, align 2, !tbaa !63
  %18 = icmp sgt i16 %15, %17
  %19 = trunc nuw nsw i64 %12 to i16
  %20 = select i1 %18, i16 %19, i16 %13
  %21 = add nuw nsw i64 %12, 1
  %22 = icmp eq i64 %21, %10
  br i1 %22, label %23, label %11, !llvm.loop !139

23:                                               ; preds = %11, %4
  %24 = phi i16 [ -1, %4 ], [ %20, %11 ]
  ret i16 %24
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE4$_22E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_22", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !83
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i16 @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE4$_23E9_M_invokeERKSt9_Any_dataOS0_S7_Os"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 2 captures(none) dereferenceable(2) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !65
  %6 = load ptr, ptr %2, align 8, !tbaa !65
  %7 = load i16, ptr %3, align 2, !tbaa !63
  %8 = icmp sgt i16 %7, 3
  br i1 %8, label %9, label %98

9:                                                ; preds = %4
  %10 = zext nneg i16 %7 to i64
  %11 = add nsw i64 %10, -3
  %12 = icmp ult i64 %11, 4
  br i1 %12, label %83, label %13

13:                                               ; preds = %9
  %14 = icmp ult i64 %11, 16
  br i1 %14, label %50, label %15

15:                                               ; preds = %13
  %16 = and i64 %11, -16
  %17 = or disjoint i64 %16, 3
  br label %18

18:                                               ; preds = %18, %15
  %19 = phi i64 [ 0, %15 ], [ %37, %18 ]
  %20 = phi <8 x i16> [ splat (i16 -32768), %15 ], [ %35, %18 ]
  %21 = phi <8 x i16> [ splat (i16 -32768), %15 ], [ %36, %18 ]
  %22 = phi <8 x i16> [ <i16 3, i16 4, i16 5, i16 6, i16 7, i16 8, i16 9, i16 10>, %15 ], [ %38, %18 ]
  %23 = add <8 x i16> %22, splat (i16 8)
  %24 = or disjoint i64 %19, 3
  %25 = getelementptr inbounds nuw i16, ptr %5, i64 %24
  %26 = getelementptr inbounds nuw i8, ptr %25, i64 16
  %27 = load <8 x i16>, ptr %25, align 2, !tbaa !63
  %28 = load <8 x i16>, ptr %26, align 2, !tbaa !63
  %29 = getelementptr inbounds nuw i16, ptr %6, i64 %24
  %30 = getelementptr inbounds nuw i8, ptr %29, i64 16
  %31 = load <8 x i16>, ptr %29, align 2, !tbaa !63
  %32 = load <8 x i16>, ptr %30, align 2, !tbaa !63
  %33 = icmp sgt <8 x i16> %27, %31
  %34 = icmp sgt <8 x i16> %28, %32
  %35 = select <8 x i1> %33, <8 x i16> %22, <8 x i16> %20
  %36 = select <8 x i1> %34, <8 x i16> %23, <8 x i16> %21
  %37 = add nuw i64 %19, 16
  %38 = add <8 x i16> %22, splat (i16 16)
  %39 = icmp eq i64 %37, %16
  br i1 %39, label %40, label %18, !llvm.loop !140

40:                                               ; preds = %18
  %41 = tail call <8 x i16> @llvm.smax.v8i16(<8 x i16> %35, <8 x i16> %36)
  %42 = tail call i16 @llvm.vector.reduce.smax.v8i16(<8 x i16> %41)
  %43 = icmp eq i16 %42, -32768
  %44 = select i1 %43, i16 -1, i16 %42
  %45 = icmp eq i64 %11, %16
  br i1 %45, label %98, label %46

46:                                               ; preds = %40
  %47 = or disjoint i64 %16, 3
  %48 = and i64 %11, 12
  %49 = icmp eq i64 %48, 0
  br i1 %49, label %83, label %50

50:                                               ; preds = %46, %13
  %51 = phi i64 [ %16, %46 ], [ 0, %13 ]
  %52 = phi i64 [ %17, %46 ], [ 3, %13 ]
  %53 = phi i16 [ %44, %46 ], [ -1, %13 ]
  %54 = icmp eq i16 %53, -1
  %55 = select i1 %54, i16 -32768, i16 %53
  %56 = and i64 %11, -4
  %57 = or i64 %11, 3
  %58 = insertelement <4 x i16> poison, i16 %55, i64 0
  %59 = shufflevector <4 x i16> %58, <4 x i16> poison, <4 x i32> zeroinitializer
  %60 = trunc i64 %52 to i16
  %61 = insertelement <4 x i16> poison, i16 %60, i64 0
  %62 = shufflevector <4 x i16> %61, <4 x i16> poison, <4 x i32> zeroinitializer
  %63 = add <4 x i16> %62, <i16 0, i16 1, i16 2, i16 3>
  br label %64

64:                                               ; preds = %64, %50
  %65 = phi i64 [ %51, %50 ], [ %75, %64 ]
  %66 = phi <4 x i16> [ %59, %50 ], [ %74, %64 ]
  %67 = phi <4 x i16> [ %63, %50 ], [ %76, %64 ]
  %68 = or disjoint i64 %65, 3
  %69 = getelementptr inbounds nuw i16, ptr %5, i64 %68
  %70 = load <4 x i16>, ptr %69, align 2, !tbaa !63
  %71 = getelementptr inbounds nuw i16, ptr %6, i64 %68
  %72 = load <4 x i16>, ptr %71, align 2, !tbaa !63
  %73 = icmp sgt <4 x i16> %70, %72
  %74 = select <4 x i1> %73, <4 x i16> %67, <4 x i16> %66
  %75 = add nuw i64 %65, 4
  %76 = add <4 x i16> %67, splat (i16 4)
  %77 = icmp eq i64 %75, %56
  br i1 %77, label %78, label %64, !llvm.loop !141

78:                                               ; preds = %64
  %79 = tail call i16 @llvm.vector.reduce.smax.v4i16(<4 x i16> %74)
  %80 = icmp eq i16 %79, -32768
  %81 = select i1 %80, i16 -1, i16 %79
  %82 = icmp eq i64 %11, %56
  br i1 %82, label %98, label %83

83:                                               ; preds = %46, %78, %9
  %84 = phi i64 [ 3, %9 ], [ %47, %46 ], [ %57, %78 ]
  %85 = phi i16 [ -1, %9 ], [ %44, %46 ], [ %81, %78 ]
  br label %86

86:                                               ; preds = %83, %86
  %87 = phi i64 [ %96, %86 ], [ %84, %83 ]
  %88 = phi i16 [ %95, %86 ], [ %85, %83 ]
  %89 = getelementptr inbounds nuw i16, ptr %5, i64 %87
  %90 = load i16, ptr %89, align 2, !tbaa !63
  %91 = getelementptr inbounds nuw i16, ptr %6, i64 %87
  %92 = load i16, ptr %91, align 2, !tbaa !63
  %93 = icmp sgt i16 %90, %92
  %94 = trunc nuw nsw i64 %87 to i16
  %95 = select i1 %93, i16 %94, i16 %88
  %96 = add nuw nsw i64 %87, 1
  %97 = icmp eq i64 %96, %10
  br i1 %97, label %98, label %86, !llvm.loop !142

98:                                               ; preds = %86, %40, %78, %4
  %99 = phi i16 [ -1, %4 ], [ %44, %40 ], [ %81, %78 ], [ %95, %86 ]
  ret i16 %99
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE4$_23E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_23", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !83
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i32 @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE4$_24E9_M_invokeERKSt9_Any_dataOS0_S7_Oi"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !39
  %6 = load ptr, ptr %2, align 8, !tbaa !39
  %7 = load i32, ptr %3, align 4, !tbaa !37
  %8 = icmp sgt i32 %7, 3
  br i1 %8, label %9, label %23

9:                                                ; preds = %4
  %10 = zext nneg i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 3, %9 ], [ %21, %11 ]
  %13 = phi i32 [ 3, %9 ], [ %20, %11 ]
  %14 = getelementptr inbounds nuw i32, ptr %5, i64 %12
  %15 = load i32, ptr %14, align 4, !tbaa !37
  %16 = getelementptr inbounds nuw i32, ptr %6, i64 %12
  %17 = load i32, ptr %16, align 4, !tbaa !37
  %18 = icmp sgt i32 %15, %17
  %19 = trunc nuw nsw i64 %12 to i32
  %20 = select i1 %18, i32 %19, i32 %13
  %21 = add nuw nsw i64 %12, 1
  %22 = icmp eq i64 %21, %10
  br i1 %22, label %23, label %11, !llvm.loop !143

23:                                               ; preds = %11, %4
  %24 = phi i32 [ 3, %4 ], [ %20, %11 ]
  ret i32 %24
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE4$_24E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_24", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !83
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i32 @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE4$_25E9_M_invokeERKSt9_Any_dataOS0_S7_Oi"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !39
  %6 = load ptr, ptr %2, align 8, !tbaa !39
  %7 = load i32, ptr %3, align 4, !tbaa !37
  %8 = icmp sgt i32 %7, 3
  br i1 %8, label %9, label %59

9:                                                ; preds = %4
  %10 = zext nneg i32 %7 to i64
  %11 = add nsw i64 %10, -3
  %12 = icmp ult i64 %11, 8
  br i1 %12, label %44, label %13

13:                                               ; preds = %9
  %14 = and i64 %11, -8
  %15 = or disjoint i64 %14, 3
  br label %16

16:                                               ; preds = %16, %13
  %17 = phi i64 [ 0, %13 ], [ %35, %16 ]
  %18 = phi <4 x i32> [ splat (i32 -2147483648), %13 ], [ %33, %16 ]
  %19 = phi <4 x i32> [ splat (i32 -2147483648), %13 ], [ %34, %16 ]
  %20 = phi <4 x i32> [ <i32 3, i32 4, i32 5, i32 6>, %13 ], [ %36, %16 ]
  %21 = add <4 x i32> %20, splat (i32 4)
  %22 = or disjoint i64 %17, 3
  %23 = getelementptr inbounds nuw i32, ptr %5, i64 %22
  %24 = getelementptr inbounds nuw i8, ptr %23, i64 16
  %25 = load <4 x i32>, ptr %23, align 4, !tbaa !37
  %26 = load <4 x i32>, ptr %24, align 4, !tbaa !37
  %27 = getelementptr inbounds nuw i32, ptr %6, i64 %22
  %28 = getelementptr inbounds nuw i8, ptr %27, i64 16
  %29 = load <4 x i32>, ptr %27, align 4, !tbaa !37
  %30 = load <4 x i32>, ptr %28, align 4, !tbaa !37
  %31 = icmp sgt <4 x i32> %25, %29
  %32 = icmp sgt <4 x i32> %26, %30
  %33 = select <4 x i1> %31, <4 x i32> %20, <4 x i32> %18
  %34 = select <4 x i1> %32, <4 x i32> %21, <4 x i32> %19
  %35 = add nuw i64 %17, 8
  %36 = add <4 x i32> %20, splat (i32 8)
  %37 = icmp eq i64 %35, %14
  br i1 %37, label %38, label %16, !llvm.loop !144

38:                                               ; preds = %16
  %39 = tail call <4 x i32> @llvm.smax.v4i32(<4 x i32> %33, <4 x i32> %34)
  %40 = tail call i32 @llvm.vector.reduce.smax.v4i32(<4 x i32> %39)
  %41 = icmp eq i32 %40, -2147483648
  %42 = select i1 %41, i32 3, i32 %40
  %43 = icmp eq i64 %11, %14
  br i1 %43, label %59, label %44

44:                                               ; preds = %9, %38
  %45 = phi i64 [ 3, %9 ], [ %15, %38 ]
  %46 = phi i32 [ 3, %9 ], [ %42, %38 ]
  br label %47

47:                                               ; preds = %44, %47
  %48 = phi i64 [ %57, %47 ], [ %45, %44 ]
  %49 = phi i32 [ %56, %47 ], [ %46, %44 ]
  %50 = getelementptr inbounds nuw i32, ptr %5, i64 %48
  %51 = load i32, ptr %50, align 4, !tbaa !37
  %52 = getelementptr inbounds nuw i32, ptr %6, i64 %48
  %53 = load i32, ptr %52, align 4, !tbaa !37
  %54 = icmp sgt i32 %51, %53
  %55 = trunc nuw nsw i64 %48 to i32
  %56 = select i1 %54, i32 %55, i32 %49
  %57 = add nuw nsw i64 %48, 1
  %58 = icmp eq i64 %57, %10
  br i1 %58, label %59, label %47, !llvm.loop !145

59:                                               ; preds = %47, %38, %4
  %60 = phi i32 [ 3, %4 ], [ %42, %38 ], [ %56, %47 ]
  ret i32 %60
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE4$_25E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_25", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !83
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i32 @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE4$_24E9_M_invokeERKSt9_Any_dataOS0_S7_Oi"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !49
  %6 = load ptr, ptr %2, align 8, !tbaa !49
  %7 = load i32, ptr %3, align 4, !tbaa !37
  %8 = icmp sgt i32 %7, 3
  br i1 %8, label %9, label %23

9:                                                ; preds = %4
  %10 = zext nneg i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 3, %9 ], [ %21, %11 ]
  %13 = phi i32 [ 3, %9 ], [ %20, %11 ]
  %14 = getelementptr inbounds nuw float, ptr %5, i64 %12
  %15 = load float, ptr %14, align 4, !tbaa !51
  %16 = getelementptr inbounds nuw float, ptr %6, i64 %12
  %17 = load float, ptr %16, align 4, !tbaa !51
  %18 = fcmp ogt float %15, %17
  %19 = trunc nuw nsw i64 %12 to i32
  %20 = select i1 %18, i32 %19, i32 %13
  %21 = add nuw nsw i64 %12, 1
  %22 = icmp eq i64 %21, %10
  br i1 %22, label %23, label %11, !llvm.loop !146

23:                                               ; preds = %11, %4
  %24 = phi i32 [ 3, %4 ], [ %20, %11 ]
  ret i32 %24
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE4$_24E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_24", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !83
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i32 @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE4$_25E9_M_invokeERKSt9_Any_dataOS0_S7_Oi"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !49
  %6 = load ptr, ptr %2, align 8, !tbaa !49
  %7 = load i32, ptr %3, align 4, !tbaa !37
  %8 = icmp sgt i32 %7, 3
  br i1 %8, label %9, label %59

9:                                                ; preds = %4
  %10 = zext nneg i32 %7 to i64
  %11 = add nsw i64 %10, -3
  %12 = icmp ult i64 %11, 8
  br i1 %12, label %44, label %13

13:                                               ; preds = %9
  %14 = and i64 %11, -8
  %15 = or disjoint i64 %14, 3
  br label %16

16:                                               ; preds = %16, %13
  %17 = phi i64 [ 0, %13 ], [ %35, %16 ]
  %18 = phi <4 x i32> [ splat (i32 -2147483648), %13 ], [ %33, %16 ]
  %19 = phi <4 x i32> [ splat (i32 -2147483648), %13 ], [ %34, %16 ]
  %20 = phi <4 x i32> [ <i32 3, i32 4, i32 5, i32 6>, %13 ], [ %36, %16 ]
  %21 = add <4 x i32> %20, splat (i32 4)
  %22 = or disjoint i64 %17, 3
  %23 = getelementptr inbounds nuw float, ptr %5, i64 %22
  %24 = getelementptr inbounds nuw i8, ptr %23, i64 16
  %25 = load <4 x float>, ptr %23, align 4, !tbaa !51
  %26 = load <4 x float>, ptr %24, align 4, !tbaa !51
  %27 = getelementptr inbounds nuw float, ptr %6, i64 %22
  %28 = getelementptr inbounds nuw i8, ptr %27, i64 16
  %29 = load <4 x float>, ptr %27, align 4, !tbaa !51
  %30 = load <4 x float>, ptr %28, align 4, !tbaa !51
  %31 = fcmp ogt <4 x float> %25, %29
  %32 = fcmp ogt <4 x float> %26, %30
  %33 = select <4 x i1> %31, <4 x i32> %20, <4 x i32> %18
  %34 = select <4 x i1> %32, <4 x i32> %21, <4 x i32> %19
  %35 = add nuw i64 %17, 8
  %36 = add <4 x i32> %20, splat (i32 8)
  %37 = icmp eq i64 %35, %14
  br i1 %37, label %38, label %16, !llvm.loop !147

38:                                               ; preds = %16
  %39 = tail call <4 x i32> @llvm.smax.v4i32(<4 x i32> %33, <4 x i32> %34)
  %40 = tail call i32 @llvm.vector.reduce.smax.v4i32(<4 x i32> %39)
  %41 = icmp eq i32 %40, -2147483648
  %42 = select i1 %41, i32 3, i32 %40
  %43 = icmp eq i64 %11, %14
  br i1 %43, label %59, label %44

44:                                               ; preds = %9, %38
  %45 = phi i64 [ 3, %9 ], [ %15, %38 ]
  %46 = phi i32 [ 3, %9 ], [ %42, %38 ]
  br label %47

47:                                               ; preds = %44, %47
  %48 = phi i64 [ %57, %47 ], [ %45, %44 ]
  %49 = phi i32 [ %56, %47 ], [ %46, %44 ]
  %50 = getelementptr inbounds nuw float, ptr %5, i64 %48
  %51 = load float, ptr %50, align 4, !tbaa !51
  %52 = getelementptr inbounds nuw float, ptr %6, i64 %48
  %53 = load float, ptr %52, align 4, !tbaa !51
  %54 = fcmp ogt float %51, %53
  %55 = trunc nuw nsw i64 %48 to i32
  %56 = select i1 %54, i32 %55, i32 %49
  %57 = add nuw nsw i64 %48, 1
  %58 = icmp eq i64 %57, %10
  br i1 %58, label %59, label %47, !llvm.loop !148

59:                                               ; preds = %47, %38, %4
  %60 = phi i32 [ 3, %4 ], [ %42, %38 ], [ %56, %47 ]
  ret i32 %60
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE4$_25E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_25", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !83
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i16 @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE4$_26E9_M_invokeERKSt9_Any_dataOS0_S7_Os"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 2 captures(none) dereferenceable(2) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !65
  %6 = load ptr, ptr %2, align 8, !tbaa !65
  %7 = load i16, ptr %3, align 2, !tbaa !63
  %8 = icmp sgt i16 %7, 3
  br i1 %8, label %9, label %23

9:                                                ; preds = %4
  %10 = zext nneg i16 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 3, %9 ], [ %21, %11 ]
  %13 = phi i16 [ 3, %9 ], [ %20, %11 ]
  %14 = getelementptr inbounds nuw i16, ptr %5, i64 %12
  %15 = load i16, ptr %14, align 2, !tbaa !63
  %16 = getelementptr inbounds nuw i16, ptr %6, i64 %12
  %17 = load i16, ptr %16, align 2, !tbaa !63
  %18 = icmp sgt i16 %15, %17
  %19 = trunc nuw nsw i64 %12 to i16
  %20 = select i1 %18, i16 %19, i16 %13
  %21 = add nuw nsw i64 %12, 1
  %22 = icmp eq i64 %21, %10
  br i1 %22, label %23, label %11, !llvm.loop !149

23:                                               ; preds = %11, %4
  %24 = phi i16 [ 3, %4 ], [ %20, %11 ]
  ret i16 %24
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE4$_26E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_26", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !83
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i16 @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE4$_27E9_M_invokeERKSt9_Any_dataOS0_S7_Os"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 2 captures(none) dereferenceable(2) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !65
  %6 = load ptr, ptr %2, align 8, !tbaa !65
  %7 = load i16, ptr %3, align 2, !tbaa !63
  %8 = icmp sgt i16 %7, 3
  br i1 %8, label %9, label %98

9:                                                ; preds = %4
  %10 = zext nneg i16 %7 to i64
  %11 = add nsw i64 %10, -3
  %12 = icmp ult i64 %11, 4
  br i1 %12, label %83, label %13

13:                                               ; preds = %9
  %14 = icmp ult i64 %11, 16
  br i1 %14, label %50, label %15

15:                                               ; preds = %13
  %16 = and i64 %11, -16
  %17 = or disjoint i64 %16, 3
  br label %18

18:                                               ; preds = %18, %15
  %19 = phi i64 [ 0, %15 ], [ %37, %18 ]
  %20 = phi <8 x i16> [ splat (i16 -32768), %15 ], [ %35, %18 ]
  %21 = phi <8 x i16> [ splat (i16 -32768), %15 ], [ %36, %18 ]
  %22 = phi <8 x i16> [ <i16 3, i16 4, i16 5, i16 6, i16 7, i16 8, i16 9, i16 10>, %15 ], [ %38, %18 ]
  %23 = add <8 x i16> %22, splat (i16 8)
  %24 = or disjoint i64 %19, 3
  %25 = getelementptr inbounds nuw i16, ptr %5, i64 %24
  %26 = getelementptr inbounds nuw i8, ptr %25, i64 16
  %27 = load <8 x i16>, ptr %25, align 2, !tbaa !63
  %28 = load <8 x i16>, ptr %26, align 2, !tbaa !63
  %29 = getelementptr inbounds nuw i16, ptr %6, i64 %24
  %30 = getelementptr inbounds nuw i8, ptr %29, i64 16
  %31 = load <8 x i16>, ptr %29, align 2, !tbaa !63
  %32 = load <8 x i16>, ptr %30, align 2, !tbaa !63
  %33 = icmp sgt <8 x i16> %27, %31
  %34 = icmp sgt <8 x i16> %28, %32
  %35 = select <8 x i1> %33, <8 x i16> %22, <8 x i16> %20
  %36 = select <8 x i1> %34, <8 x i16> %23, <8 x i16> %21
  %37 = add nuw i64 %19, 16
  %38 = add <8 x i16> %22, splat (i16 16)
  %39 = icmp eq i64 %37, %16
  br i1 %39, label %40, label %18, !llvm.loop !150

40:                                               ; preds = %18
  %41 = tail call <8 x i16> @llvm.smax.v8i16(<8 x i16> %35, <8 x i16> %36)
  %42 = tail call i16 @llvm.vector.reduce.smax.v8i16(<8 x i16> %41)
  %43 = icmp eq i16 %42, -32768
  %44 = select i1 %43, i16 3, i16 %42
  %45 = icmp eq i64 %11, %16
  br i1 %45, label %98, label %46

46:                                               ; preds = %40
  %47 = or disjoint i64 %16, 3
  %48 = and i64 %11, 12
  %49 = icmp eq i64 %48, 0
  br i1 %49, label %83, label %50

50:                                               ; preds = %46, %13
  %51 = phi i64 [ %16, %46 ], [ 0, %13 ]
  %52 = phi i64 [ %17, %46 ], [ 3, %13 ]
  %53 = phi i16 [ %44, %46 ], [ 3, %13 ]
  %54 = icmp eq i16 %53, 3
  %55 = select i1 %54, i16 -32768, i16 %53
  %56 = and i64 %11, -4
  %57 = or i64 %11, 3
  %58 = insertelement <4 x i16> poison, i16 %55, i64 0
  %59 = shufflevector <4 x i16> %58, <4 x i16> poison, <4 x i32> zeroinitializer
  %60 = trunc i64 %52 to i16
  %61 = insertelement <4 x i16> poison, i16 %60, i64 0
  %62 = shufflevector <4 x i16> %61, <4 x i16> poison, <4 x i32> zeroinitializer
  %63 = add <4 x i16> %62, <i16 0, i16 1, i16 2, i16 3>
  br label %64

64:                                               ; preds = %64, %50
  %65 = phi i64 [ %51, %50 ], [ %75, %64 ]
  %66 = phi <4 x i16> [ %59, %50 ], [ %74, %64 ]
  %67 = phi <4 x i16> [ %63, %50 ], [ %76, %64 ]
  %68 = or disjoint i64 %65, 3
  %69 = getelementptr inbounds nuw i16, ptr %5, i64 %68
  %70 = load <4 x i16>, ptr %69, align 2, !tbaa !63
  %71 = getelementptr inbounds nuw i16, ptr %6, i64 %68
  %72 = load <4 x i16>, ptr %71, align 2, !tbaa !63
  %73 = icmp sgt <4 x i16> %70, %72
  %74 = select <4 x i1> %73, <4 x i16> %67, <4 x i16> %66
  %75 = add nuw i64 %65, 4
  %76 = add <4 x i16> %67, splat (i16 4)
  %77 = icmp eq i64 %75, %56
  br i1 %77, label %78, label %64, !llvm.loop !151

78:                                               ; preds = %64
  %79 = tail call i16 @llvm.vector.reduce.smax.v4i16(<4 x i16> %74)
  %80 = icmp eq i16 %79, -32768
  %81 = select i1 %80, i16 3, i16 %79
  %82 = icmp eq i64 %11, %56
  br i1 %82, label %98, label %83

83:                                               ; preds = %46, %78, %9
  %84 = phi i64 [ 3, %9 ], [ %47, %46 ], [ %57, %78 ]
  %85 = phi i16 [ 3, %9 ], [ %44, %46 ], [ %81, %78 ]
  br label %86

86:                                               ; preds = %83, %86
  %87 = phi i64 [ %96, %86 ], [ %84, %83 ]
  %88 = phi i16 [ %95, %86 ], [ %85, %83 ]
  %89 = getelementptr inbounds nuw i16, ptr %5, i64 %87
  %90 = load i16, ptr %89, align 2, !tbaa !63
  %91 = getelementptr inbounds nuw i16, ptr %6, i64 %87
  %92 = load i16, ptr %91, align 2, !tbaa !63
  %93 = icmp sgt i16 %90, %92
  %94 = trunc nuw nsw i64 %87 to i16
  %95 = select i1 %93, i16 %94, i16 %88
  %96 = add nuw nsw i64 %87, 1
  %97 = icmp eq i64 %96, %10
  br i1 %97, label %98, label %86, !llvm.loop !152

98:                                               ; preds = %86, %40, %78, %4
  %99 = phi i16 [ 3, %4 ], [ %44, %40 ], [ %81, %78 ], [ %95, %86 ]
  ret i16 %99
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE4$_27E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_27", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !83
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i32 @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE4$_28E9_M_invokeERKSt9_Any_dataOS0_S7_Oi"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !39
  %6 = load ptr, ptr %2, align 8, !tbaa !39
  %7 = load i32, ptr %3, align 4, !tbaa !37
  %8 = icmp sgt i32 %7, 3
  br i1 %8, label %9, label %23

9:                                                ; preds = %4
  %10 = zext nneg i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 3, %9 ], [ %21, %11 ]
  %13 = phi i32 [ 2, %9 ], [ %20, %11 ]
  %14 = getelementptr inbounds nuw i32, ptr %5, i64 %12
  %15 = load i32, ptr %14, align 4, !tbaa !37
  %16 = getelementptr inbounds nuw i32, ptr %6, i64 %12
  %17 = load i32, ptr %16, align 4, !tbaa !37
  %18 = icmp sgt i32 %15, %17
  %19 = trunc nuw nsw i64 %12 to i32
  %20 = select i1 %18, i32 %19, i32 %13
  %21 = add nuw nsw i64 %12, 1
  %22 = icmp eq i64 %21, %10
  br i1 %22, label %23, label %11, !llvm.loop !153

23:                                               ; preds = %11, %4
  %24 = phi i32 [ 2, %4 ], [ %20, %11 ]
  ret i32 %24
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE4$_28E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_28", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !83
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i32 @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE4$_29E9_M_invokeERKSt9_Any_dataOS0_S7_Oi"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !39
  %6 = load ptr, ptr %2, align 8, !tbaa !39
  %7 = load i32, ptr %3, align 4, !tbaa !37
  %8 = icmp sgt i32 %7, 3
  br i1 %8, label %9, label %59

9:                                                ; preds = %4
  %10 = zext nneg i32 %7 to i64
  %11 = add nsw i64 %10, -3
  %12 = icmp ult i64 %11, 8
  br i1 %12, label %44, label %13

13:                                               ; preds = %9
  %14 = and i64 %11, -8
  %15 = or disjoint i64 %14, 3
  br label %16

16:                                               ; preds = %16, %13
  %17 = phi i64 [ 0, %13 ], [ %35, %16 ]
  %18 = phi <4 x i32> [ splat (i32 -2147483648), %13 ], [ %33, %16 ]
  %19 = phi <4 x i32> [ splat (i32 -2147483648), %13 ], [ %34, %16 ]
  %20 = phi <4 x i32> [ <i32 3, i32 4, i32 5, i32 6>, %13 ], [ %36, %16 ]
  %21 = add <4 x i32> %20, splat (i32 4)
  %22 = or disjoint i64 %17, 3
  %23 = getelementptr inbounds nuw i32, ptr %5, i64 %22
  %24 = getelementptr inbounds nuw i8, ptr %23, i64 16
  %25 = load <4 x i32>, ptr %23, align 4, !tbaa !37
  %26 = load <4 x i32>, ptr %24, align 4, !tbaa !37
  %27 = getelementptr inbounds nuw i32, ptr %6, i64 %22
  %28 = getelementptr inbounds nuw i8, ptr %27, i64 16
  %29 = load <4 x i32>, ptr %27, align 4, !tbaa !37
  %30 = load <4 x i32>, ptr %28, align 4, !tbaa !37
  %31 = icmp sgt <4 x i32> %25, %29
  %32 = icmp sgt <4 x i32> %26, %30
  %33 = select <4 x i1> %31, <4 x i32> %20, <4 x i32> %18
  %34 = select <4 x i1> %32, <4 x i32> %21, <4 x i32> %19
  %35 = add nuw i64 %17, 8
  %36 = add <4 x i32> %20, splat (i32 8)
  %37 = icmp eq i64 %35, %14
  br i1 %37, label %38, label %16, !llvm.loop !154

38:                                               ; preds = %16
  %39 = tail call <4 x i32> @llvm.smax.v4i32(<4 x i32> %33, <4 x i32> %34)
  %40 = tail call i32 @llvm.vector.reduce.smax.v4i32(<4 x i32> %39)
  %41 = icmp eq i32 %40, -2147483648
  %42 = select i1 %41, i32 2, i32 %40
  %43 = icmp eq i64 %11, %14
  br i1 %43, label %59, label %44

44:                                               ; preds = %9, %38
  %45 = phi i64 [ 3, %9 ], [ %15, %38 ]
  %46 = phi i32 [ 2, %9 ], [ %42, %38 ]
  br label %47

47:                                               ; preds = %44, %47
  %48 = phi i64 [ %57, %47 ], [ %45, %44 ]
  %49 = phi i32 [ %56, %47 ], [ %46, %44 ]
  %50 = getelementptr inbounds nuw i32, ptr %5, i64 %48
  %51 = load i32, ptr %50, align 4, !tbaa !37
  %52 = getelementptr inbounds nuw i32, ptr %6, i64 %48
  %53 = load i32, ptr %52, align 4, !tbaa !37
  %54 = icmp sgt i32 %51, %53
  %55 = trunc nuw nsw i64 %48 to i32
  %56 = select i1 %54, i32 %55, i32 %49
  %57 = add nuw nsw i64 %48, 1
  %58 = icmp eq i64 %57, %10
  br i1 %58, label %59, label %47, !llvm.loop !155

59:                                               ; preds = %47, %38, %4
  %60 = phi i32 [ 2, %4 ], [ %42, %38 ], [ %56, %47 ]
  ret i32 %60
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE4$_29E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_29", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !83
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i32 @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE4$_28E9_M_invokeERKSt9_Any_dataOS0_S7_Oi"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !49
  %6 = load ptr, ptr %2, align 8, !tbaa !49
  %7 = load i32, ptr %3, align 4, !tbaa !37
  %8 = icmp sgt i32 %7, 3
  br i1 %8, label %9, label %23

9:                                                ; preds = %4
  %10 = zext nneg i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 3, %9 ], [ %21, %11 ]
  %13 = phi i32 [ 2, %9 ], [ %20, %11 ]
  %14 = getelementptr inbounds nuw float, ptr %5, i64 %12
  %15 = load float, ptr %14, align 4, !tbaa !51
  %16 = getelementptr inbounds nuw float, ptr %6, i64 %12
  %17 = load float, ptr %16, align 4, !tbaa !51
  %18 = fcmp ogt float %15, %17
  %19 = trunc nuw nsw i64 %12 to i32
  %20 = select i1 %18, i32 %19, i32 %13
  %21 = add nuw nsw i64 %12, 1
  %22 = icmp eq i64 %21, %10
  br i1 %22, label %23, label %11, !llvm.loop !156

23:                                               ; preds = %11, %4
  %24 = phi i32 [ 2, %4 ], [ %20, %11 ]
  ret i32 %24
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE4$_28E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_28", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !83
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i32 @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE4$_29E9_M_invokeERKSt9_Any_dataOS0_S7_Oi"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !49
  %6 = load ptr, ptr %2, align 8, !tbaa !49
  %7 = load i32, ptr %3, align 4, !tbaa !37
  %8 = icmp sgt i32 %7, 3
  br i1 %8, label %9, label %59

9:                                                ; preds = %4
  %10 = zext nneg i32 %7 to i64
  %11 = add nsw i64 %10, -3
  %12 = icmp ult i64 %11, 8
  br i1 %12, label %44, label %13

13:                                               ; preds = %9
  %14 = and i64 %11, -8
  %15 = or disjoint i64 %14, 3
  br label %16

16:                                               ; preds = %16, %13
  %17 = phi i64 [ 0, %13 ], [ %35, %16 ]
  %18 = phi <4 x i32> [ splat (i32 -2147483648), %13 ], [ %33, %16 ]
  %19 = phi <4 x i32> [ splat (i32 -2147483648), %13 ], [ %34, %16 ]
  %20 = phi <4 x i32> [ <i32 3, i32 4, i32 5, i32 6>, %13 ], [ %36, %16 ]
  %21 = add <4 x i32> %20, splat (i32 4)
  %22 = or disjoint i64 %17, 3
  %23 = getelementptr inbounds nuw float, ptr %5, i64 %22
  %24 = getelementptr inbounds nuw i8, ptr %23, i64 16
  %25 = load <4 x float>, ptr %23, align 4, !tbaa !51
  %26 = load <4 x float>, ptr %24, align 4, !tbaa !51
  %27 = getelementptr inbounds nuw float, ptr %6, i64 %22
  %28 = getelementptr inbounds nuw i8, ptr %27, i64 16
  %29 = load <4 x float>, ptr %27, align 4, !tbaa !51
  %30 = load <4 x float>, ptr %28, align 4, !tbaa !51
  %31 = fcmp ogt <4 x float> %25, %29
  %32 = fcmp ogt <4 x float> %26, %30
  %33 = select <4 x i1> %31, <4 x i32> %20, <4 x i32> %18
  %34 = select <4 x i1> %32, <4 x i32> %21, <4 x i32> %19
  %35 = add nuw i64 %17, 8
  %36 = add <4 x i32> %20, splat (i32 8)
  %37 = icmp eq i64 %35, %14
  br i1 %37, label %38, label %16, !llvm.loop !157

38:                                               ; preds = %16
  %39 = tail call <4 x i32> @llvm.smax.v4i32(<4 x i32> %33, <4 x i32> %34)
  %40 = tail call i32 @llvm.vector.reduce.smax.v4i32(<4 x i32> %39)
  %41 = icmp eq i32 %40, -2147483648
  %42 = select i1 %41, i32 2, i32 %40
  %43 = icmp eq i64 %11, %14
  br i1 %43, label %59, label %44

44:                                               ; preds = %9, %38
  %45 = phi i64 [ 3, %9 ], [ %15, %38 ]
  %46 = phi i32 [ 2, %9 ], [ %42, %38 ]
  br label %47

47:                                               ; preds = %44, %47
  %48 = phi i64 [ %57, %47 ], [ %45, %44 ]
  %49 = phi i32 [ %56, %47 ], [ %46, %44 ]
  %50 = getelementptr inbounds nuw float, ptr %5, i64 %48
  %51 = load float, ptr %50, align 4, !tbaa !51
  %52 = getelementptr inbounds nuw float, ptr %6, i64 %48
  %53 = load float, ptr %52, align 4, !tbaa !51
  %54 = fcmp ogt float %51, %53
  %55 = trunc nuw nsw i64 %48 to i32
  %56 = select i1 %54, i32 %55, i32 %49
  %57 = add nuw nsw i64 %48, 1
  %58 = icmp eq i64 %57, %10
  br i1 %58, label %59, label %47, !llvm.loop !158

59:                                               ; preds = %47, %38, %4
  %60 = phi i32 [ 2, %4 ], [ %42, %38 ], [ %56, %47 ]
  ret i32 %60
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE4$_29E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_29", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !83
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i16 @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE4$_30E9_M_invokeERKSt9_Any_dataOS0_S7_Os"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 2 captures(none) dereferenceable(2) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !65
  %6 = load ptr, ptr %2, align 8, !tbaa !65
  %7 = load i16, ptr %3, align 2, !tbaa !63
  %8 = icmp sgt i16 %7, 3
  br i1 %8, label %9, label %23

9:                                                ; preds = %4
  %10 = zext nneg i16 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 3, %9 ], [ %21, %11 ]
  %13 = phi i16 [ 2, %9 ], [ %20, %11 ]
  %14 = getelementptr inbounds nuw i16, ptr %5, i64 %12
  %15 = load i16, ptr %14, align 2, !tbaa !63
  %16 = getelementptr inbounds nuw i16, ptr %6, i64 %12
  %17 = load i16, ptr %16, align 2, !tbaa !63
  %18 = icmp sgt i16 %15, %17
  %19 = trunc nuw nsw i64 %12 to i16
  %20 = select i1 %18, i16 %19, i16 %13
  %21 = add nuw nsw i64 %12, 1
  %22 = icmp eq i64 %21, %10
  br i1 %22, label %23, label %11, !llvm.loop !159

23:                                               ; preds = %11, %4
  %24 = phi i16 [ 2, %4 ], [ %20, %11 ]
  ret i16 %24
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE4$_30E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_30", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !83
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i16 @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE4$_31E9_M_invokeERKSt9_Any_dataOS0_S7_Os"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 2 captures(none) dereferenceable(2) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !65
  %6 = load ptr, ptr %2, align 8, !tbaa !65
  %7 = load i16, ptr %3, align 2, !tbaa !63
  %8 = icmp sgt i16 %7, 3
  br i1 %8, label %9, label %98

9:                                                ; preds = %4
  %10 = zext nneg i16 %7 to i64
  %11 = add nsw i64 %10, -3
  %12 = icmp ult i64 %11, 4
  br i1 %12, label %83, label %13

13:                                               ; preds = %9
  %14 = icmp ult i64 %11, 16
  br i1 %14, label %50, label %15

15:                                               ; preds = %13
  %16 = and i64 %11, -16
  %17 = or disjoint i64 %16, 3
  br label %18

18:                                               ; preds = %18, %15
  %19 = phi i64 [ 0, %15 ], [ %37, %18 ]
  %20 = phi <8 x i16> [ splat (i16 -32768), %15 ], [ %35, %18 ]
  %21 = phi <8 x i16> [ splat (i16 -32768), %15 ], [ %36, %18 ]
  %22 = phi <8 x i16> [ <i16 3, i16 4, i16 5, i16 6, i16 7, i16 8, i16 9, i16 10>, %15 ], [ %38, %18 ]
  %23 = add <8 x i16> %22, splat (i16 8)
  %24 = or disjoint i64 %19, 3
  %25 = getelementptr inbounds nuw i16, ptr %5, i64 %24
  %26 = getelementptr inbounds nuw i8, ptr %25, i64 16
  %27 = load <8 x i16>, ptr %25, align 2, !tbaa !63
  %28 = load <8 x i16>, ptr %26, align 2, !tbaa !63
  %29 = getelementptr inbounds nuw i16, ptr %6, i64 %24
  %30 = getelementptr inbounds nuw i8, ptr %29, i64 16
  %31 = load <8 x i16>, ptr %29, align 2, !tbaa !63
  %32 = load <8 x i16>, ptr %30, align 2, !tbaa !63
  %33 = icmp sgt <8 x i16> %27, %31
  %34 = icmp sgt <8 x i16> %28, %32
  %35 = select <8 x i1> %33, <8 x i16> %22, <8 x i16> %20
  %36 = select <8 x i1> %34, <8 x i16> %23, <8 x i16> %21
  %37 = add nuw i64 %19, 16
  %38 = add <8 x i16> %22, splat (i16 16)
  %39 = icmp eq i64 %37, %16
  br i1 %39, label %40, label %18, !llvm.loop !160

40:                                               ; preds = %18
  %41 = tail call <8 x i16> @llvm.smax.v8i16(<8 x i16> %35, <8 x i16> %36)
  %42 = tail call i16 @llvm.vector.reduce.smax.v8i16(<8 x i16> %41)
  %43 = icmp eq i16 %42, -32768
  %44 = select i1 %43, i16 2, i16 %42
  %45 = icmp eq i64 %11, %16
  br i1 %45, label %98, label %46

46:                                               ; preds = %40
  %47 = or disjoint i64 %16, 3
  %48 = and i64 %11, 12
  %49 = icmp eq i64 %48, 0
  br i1 %49, label %83, label %50

50:                                               ; preds = %46, %13
  %51 = phi i64 [ %16, %46 ], [ 0, %13 ]
  %52 = phi i64 [ %17, %46 ], [ 3, %13 ]
  %53 = phi i16 [ %44, %46 ], [ 2, %13 ]
  %54 = icmp eq i16 %53, 2
  %55 = select i1 %54, i16 -32768, i16 %53
  %56 = and i64 %11, -4
  %57 = or i64 %11, 3
  %58 = insertelement <4 x i16> poison, i16 %55, i64 0
  %59 = shufflevector <4 x i16> %58, <4 x i16> poison, <4 x i32> zeroinitializer
  %60 = trunc i64 %52 to i16
  %61 = insertelement <4 x i16> poison, i16 %60, i64 0
  %62 = shufflevector <4 x i16> %61, <4 x i16> poison, <4 x i32> zeroinitializer
  %63 = add <4 x i16> %62, <i16 0, i16 1, i16 2, i16 3>
  br label %64

64:                                               ; preds = %64, %50
  %65 = phi i64 [ %51, %50 ], [ %75, %64 ]
  %66 = phi <4 x i16> [ %59, %50 ], [ %74, %64 ]
  %67 = phi <4 x i16> [ %63, %50 ], [ %76, %64 ]
  %68 = or disjoint i64 %65, 3
  %69 = getelementptr inbounds nuw i16, ptr %5, i64 %68
  %70 = load <4 x i16>, ptr %69, align 2, !tbaa !63
  %71 = getelementptr inbounds nuw i16, ptr %6, i64 %68
  %72 = load <4 x i16>, ptr %71, align 2, !tbaa !63
  %73 = icmp sgt <4 x i16> %70, %72
  %74 = select <4 x i1> %73, <4 x i16> %67, <4 x i16> %66
  %75 = add nuw i64 %65, 4
  %76 = add <4 x i16> %67, splat (i16 4)
  %77 = icmp eq i64 %75, %56
  br i1 %77, label %78, label %64, !llvm.loop !161

78:                                               ; preds = %64
  %79 = tail call i16 @llvm.vector.reduce.smax.v4i16(<4 x i16> %74)
  %80 = icmp eq i16 %79, -32768
  %81 = select i1 %80, i16 2, i16 %79
  %82 = icmp eq i64 %11, %56
  br i1 %82, label %98, label %83

83:                                               ; preds = %46, %78, %9
  %84 = phi i64 [ 3, %9 ], [ %47, %46 ], [ %57, %78 ]
  %85 = phi i16 [ 2, %9 ], [ %44, %46 ], [ %81, %78 ]
  br label %86

86:                                               ; preds = %83, %86
  %87 = phi i64 [ %96, %86 ], [ %84, %83 ]
  %88 = phi i16 [ %95, %86 ], [ %85, %83 ]
  %89 = getelementptr inbounds nuw i16, ptr %5, i64 %87
  %90 = load i16, ptr %89, align 2, !tbaa !63
  %91 = getelementptr inbounds nuw i16, ptr %6, i64 %87
  %92 = load i16, ptr %91, align 2, !tbaa !63
  %93 = icmp sgt i16 %90, %92
  %94 = trunc nuw nsw i64 %87 to i16
  %95 = select i1 %93, i16 %94, i16 %88
  %96 = add nuw nsw i64 %87, 1
  %97 = icmp eq i64 %96, %10
  br i1 %97, label %98, label %86, !llvm.loop !162

98:                                               ; preds = %86, %40, %78, %4
  %99 = phi i16 [ 2, %4 ], [ %44, %40 ], [ %81, %78 ], [ %95, %86 ]
  ret i16 %99
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE4$_31E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_31", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !83
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i32 @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE4$_32E9_M_invokeERKSt9_Any_dataOS0_S7_Oi"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !39
  %6 = load ptr, ptr %2, align 8, !tbaa !39
  %7 = load i32, ptr %3, align 4, !tbaa !37
  %8 = icmp sgt i32 %7, 3
  br i1 %8, label %9, label %23

9:                                                ; preds = %4
  %10 = zext nneg i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 3, %9 ], [ %21, %11 ]
  %13 = phi i32 [ 4, %9 ], [ %20, %11 ]
  %14 = getelementptr inbounds nuw i32, ptr %5, i64 %12
  %15 = load i32, ptr %14, align 4, !tbaa !37
  %16 = getelementptr inbounds nuw i32, ptr %6, i64 %12
  %17 = load i32, ptr %16, align 4, !tbaa !37
  %18 = icmp sgt i32 %15, %17
  %19 = trunc nuw nsw i64 %12 to i32
  %20 = select i1 %18, i32 %19, i32 %13
  %21 = add nuw nsw i64 %12, 1
  %22 = icmp eq i64 %21, %10
  br i1 %22, label %23, label %11, !llvm.loop !163

23:                                               ; preds = %11, %4
  %24 = phi i32 [ 4, %4 ], [ %20, %11 ]
  ret i32 %24
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE4$_32E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_32", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !83
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i32 @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE4$_33E9_M_invokeERKSt9_Any_dataOS0_S7_Oi"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !39
  %6 = load ptr, ptr %2, align 8, !tbaa !39
  %7 = load i32, ptr %3, align 4, !tbaa !37
  %8 = icmp sgt i32 %7, 3
  br i1 %8, label %9, label %59

9:                                                ; preds = %4
  %10 = zext nneg i32 %7 to i64
  %11 = add nsw i64 %10, -3
  %12 = icmp ult i64 %11, 8
  br i1 %12, label %44, label %13

13:                                               ; preds = %9
  %14 = and i64 %11, -8
  %15 = or disjoint i64 %14, 3
  br label %16

16:                                               ; preds = %16, %13
  %17 = phi i64 [ 0, %13 ], [ %35, %16 ]
  %18 = phi <4 x i32> [ splat (i32 -2147483648), %13 ], [ %33, %16 ]
  %19 = phi <4 x i32> [ splat (i32 -2147483648), %13 ], [ %34, %16 ]
  %20 = phi <4 x i32> [ <i32 3, i32 4, i32 5, i32 6>, %13 ], [ %36, %16 ]
  %21 = add <4 x i32> %20, splat (i32 4)
  %22 = or disjoint i64 %17, 3
  %23 = getelementptr inbounds nuw i32, ptr %5, i64 %22
  %24 = getelementptr inbounds nuw i8, ptr %23, i64 16
  %25 = load <4 x i32>, ptr %23, align 4, !tbaa !37
  %26 = load <4 x i32>, ptr %24, align 4, !tbaa !37
  %27 = getelementptr inbounds nuw i32, ptr %6, i64 %22
  %28 = getelementptr inbounds nuw i8, ptr %27, i64 16
  %29 = load <4 x i32>, ptr %27, align 4, !tbaa !37
  %30 = load <4 x i32>, ptr %28, align 4, !tbaa !37
  %31 = icmp sgt <4 x i32> %25, %29
  %32 = icmp sgt <4 x i32> %26, %30
  %33 = select <4 x i1> %31, <4 x i32> %20, <4 x i32> %18
  %34 = select <4 x i1> %32, <4 x i32> %21, <4 x i32> %19
  %35 = add nuw i64 %17, 8
  %36 = add <4 x i32> %20, splat (i32 8)
  %37 = icmp eq i64 %35, %14
  br i1 %37, label %38, label %16, !llvm.loop !164

38:                                               ; preds = %16
  %39 = tail call <4 x i32> @llvm.smax.v4i32(<4 x i32> %33, <4 x i32> %34)
  %40 = tail call i32 @llvm.vector.reduce.smax.v4i32(<4 x i32> %39)
  %41 = icmp eq i32 %40, -2147483648
  %42 = select i1 %41, i32 4, i32 %40
  %43 = icmp eq i64 %11, %14
  br i1 %43, label %59, label %44

44:                                               ; preds = %9, %38
  %45 = phi i64 [ 3, %9 ], [ %15, %38 ]
  %46 = phi i32 [ 4, %9 ], [ %42, %38 ]
  br label %47

47:                                               ; preds = %44, %47
  %48 = phi i64 [ %57, %47 ], [ %45, %44 ]
  %49 = phi i32 [ %56, %47 ], [ %46, %44 ]
  %50 = getelementptr inbounds nuw i32, ptr %5, i64 %48
  %51 = load i32, ptr %50, align 4, !tbaa !37
  %52 = getelementptr inbounds nuw i32, ptr %6, i64 %48
  %53 = load i32, ptr %52, align 4, !tbaa !37
  %54 = icmp sgt i32 %51, %53
  %55 = trunc nuw nsw i64 %48 to i32
  %56 = select i1 %54, i32 %55, i32 %49
  %57 = add nuw nsw i64 %48, 1
  %58 = icmp eq i64 %57, %10
  br i1 %58, label %59, label %47, !llvm.loop !165

59:                                               ; preds = %47, %38, %4
  %60 = phi i32 [ 4, %4 ], [ %42, %38 ], [ %56, %47 ]
  ret i32 %60
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFiPiS0_iEZ4mainE4$_33E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_33", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !83
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i32 @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE4$_32E9_M_invokeERKSt9_Any_dataOS0_S7_Oi"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !49
  %6 = load ptr, ptr %2, align 8, !tbaa !49
  %7 = load i32, ptr %3, align 4, !tbaa !37
  %8 = icmp sgt i32 %7, 3
  br i1 %8, label %9, label %23

9:                                                ; preds = %4
  %10 = zext nneg i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 3, %9 ], [ %21, %11 ]
  %13 = phi i32 [ 4, %9 ], [ %20, %11 ]
  %14 = getelementptr inbounds nuw float, ptr %5, i64 %12
  %15 = load float, ptr %14, align 4, !tbaa !51
  %16 = getelementptr inbounds nuw float, ptr %6, i64 %12
  %17 = load float, ptr %16, align 4, !tbaa !51
  %18 = fcmp ogt float %15, %17
  %19 = trunc nuw nsw i64 %12 to i32
  %20 = select i1 %18, i32 %19, i32 %13
  %21 = add nuw nsw i64 %12, 1
  %22 = icmp eq i64 %21, %10
  br i1 %22, label %23, label %11, !llvm.loop !166

23:                                               ; preds = %11, %4
  %24 = phi i32 [ 4, %4 ], [ %20, %11 ]
  ret i32 %24
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE4$_32E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_32", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !83
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i32 @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE4$_33E9_M_invokeERKSt9_Any_dataOS0_S7_Oi"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !49
  %6 = load ptr, ptr %2, align 8, !tbaa !49
  %7 = load i32, ptr %3, align 4, !tbaa !37
  %8 = icmp sgt i32 %7, 3
  br i1 %8, label %9, label %59

9:                                                ; preds = %4
  %10 = zext nneg i32 %7 to i64
  %11 = add nsw i64 %10, -3
  %12 = icmp ult i64 %11, 8
  br i1 %12, label %44, label %13

13:                                               ; preds = %9
  %14 = and i64 %11, -8
  %15 = or disjoint i64 %14, 3
  br label %16

16:                                               ; preds = %16, %13
  %17 = phi i64 [ 0, %13 ], [ %35, %16 ]
  %18 = phi <4 x i32> [ splat (i32 -2147483648), %13 ], [ %33, %16 ]
  %19 = phi <4 x i32> [ splat (i32 -2147483648), %13 ], [ %34, %16 ]
  %20 = phi <4 x i32> [ <i32 3, i32 4, i32 5, i32 6>, %13 ], [ %36, %16 ]
  %21 = add <4 x i32> %20, splat (i32 4)
  %22 = or disjoint i64 %17, 3
  %23 = getelementptr inbounds nuw float, ptr %5, i64 %22
  %24 = getelementptr inbounds nuw i8, ptr %23, i64 16
  %25 = load <4 x float>, ptr %23, align 4, !tbaa !51
  %26 = load <4 x float>, ptr %24, align 4, !tbaa !51
  %27 = getelementptr inbounds nuw float, ptr %6, i64 %22
  %28 = getelementptr inbounds nuw i8, ptr %27, i64 16
  %29 = load <4 x float>, ptr %27, align 4, !tbaa !51
  %30 = load <4 x float>, ptr %28, align 4, !tbaa !51
  %31 = fcmp ogt <4 x float> %25, %29
  %32 = fcmp ogt <4 x float> %26, %30
  %33 = select <4 x i1> %31, <4 x i32> %20, <4 x i32> %18
  %34 = select <4 x i1> %32, <4 x i32> %21, <4 x i32> %19
  %35 = add nuw i64 %17, 8
  %36 = add <4 x i32> %20, splat (i32 8)
  %37 = icmp eq i64 %35, %14
  br i1 %37, label %38, label %16, !llvm.loop !167

38:                                               ; preds = %16
  %39 = tail call <4 x i32> @llvm.smax.v4i32(<4 x i32> %33, <4 x i32> %34)
  %40 = tail call i32 @llvm.vector.reduce.smax.v4i32(<4 x i32> %39)
  %41 = icmp eq i32 %40, -2147483648
  %42 = select i1 %41, i32 4, i32 %40
  %43 = icmp eq i64 %11, %14
  br i1 %43, label %59, label %44

44:                                               ; preds = %9, %38
  %45 = phi i64 [ 3, %9 ], [ %15, %38 ]
  %46 = phi i32 [ 4, %9 ], [ %42, %38 ]
  br label %47

47:                                               ; preds = %44, %47
  %48 = phi i64 [ %57, %47 ], [ %45, %44 ]
  %49 = phi i32 [ %56, %47 ], [ %46, %44 ]
  %50 = getelementptr inbounds nuw float, ptr %5, i64 %48
  %51 = load float, ptr %50, align 4, !tbaa !51
  %52 = getelementptr inbounds nuw float, ptr %6, i64 %48
  %53 = load float, ptr %52, align 4, !tbaa !51
  %54 = fcmp ogt float %51, %53
  %55 = trunc nuw nsw i64 %48 to i32
  %56 = select i1 %54, i32 %55, i32 %49
  %57 = add nuw nsw i64 %48, 1
  %58 = icmp eq i64 %57, %10
  br i1 %58, label %59, label %47, !llvm.loop !168

59:                                               ; preds = %47, %38, %4
  %60 = phi i32 [ 4, %4 ], [ %42, %38 ], [ %56, %47 ]
  ret i32 %60
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFiPfS0_iEZ4mainE4$_33E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_33", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !83
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i16 @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE4$_34E9_M_invokeERKSt9_Any_dataOS0_S7_Os"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 2 captures(none) dereferenceable(2) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !65
  %6 = load ptr, ptr %2, align 8, !tbaa !65
  %7 = load i16, ptr %3, align 2, !tbaa !63
  %8 = icmp sgt i16 %7, 3
  br i1 %8, label %9, label %23

9:                                                ; preds = %4
  %10 = zext nneg i16 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 3, %9 ], [ %21, %11 ]
  %13 = phi i16 [ 4, %9 ], [ %20, %11 ]
  %14 = getelementptr inbounds nuw i16, ptr %5, i64 %12
  %15 = load i16, ptr %14, align 2, !tbaa !63
  %16 = getelementptr inbounds nuw i16, ptr %6, i64 %12
  %17 = load i16, ptr %16, align 2, !tbaa !63
  %18 = icmp sgt i16 %15, %17
  %19 = trunc nuw nsw i64 %12 to i16
  %20 = select i1 %18, i16 %19, i16 %13
  %21 = add nuw nsw i64 %12, 1
  %22 = icmp eq i64 %21, %10
  br i1 %22, label %23, label %11, !llvm.loop !169

23:                                               ; preds = %11, %4
  %24 = phi i16 [ 4, %4 ], [ %20, %11 ]
  ret i16 %24
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE4$_34E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_34", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !83
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, inaccessiblemem: none) uwtable
define internal noundef i16 @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE4$_35E9_M_invokeERKSt9_Any_dataOS0_S7_Os"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 2 captures(none) dereferenceable(2) %3) #14 {
  %5 = load ptr, ptr %1, align 8, !tbaa !65
  %6 = load ptr, ptr %2, align 8, !tbaa !65
  %7 = load i16, ptr %3, align 2, !tbaa !63
  %8 = icmp sgt i16 %7, 3
  br i1 %8, label %9, label %98

9:                                                ; preds = %4
  %10 = zext nneg i16 %7 to i64
  %11 = add nsw i64 %10, -3
  %12 = icmp ult i64 %11, 4
  br i1 %12, label %83, label %13

13:                                               ; preds = %9
  %14 = icmp ult i64 %11, 16
  br i1 %14, label %50, label %15

15:                                               ; preds = %13
  %16 = and i64 %11, -16
  %17 = or disjoint i64 %16, 3
  br label %18

18:                                               ; preds = %18, %15
  %19 = phi i64 [ 0, %15 ], [ %37, %18 ]
  %20 = phi <8 x i16> [ splat (i16 -32768), %15 ], [ %35, %18 ]
  %21 = phi <8 x i16> [ splat (i16 -32768), %15 ], [ %36, %18 ]
  %22 = phi <8 x i16> [ <i16 3, i16 4, i16 5, i16 6, i16 7, i16 8, i16 9, i16 10>, %15 ], [ %38, %18 ]
  %23 = add <8 x i16> %22, splat (i16 8)
  %24 = or disjoint i64 %19, 3
  %25 = getelementptr inbounds nuw i16, ptr %5, i64 %24
  %26 = getelementptr inbounds nuw i8, ptr %25, i64 16
  %27 = load <8 x i16>, ptr %25, align 2, !tbaa !63
  %28 = load <8 x i16>, ptr %26, align 2, !tbaa !63
  %29 = getelementptr inbounds nuw i16, ptr %6, i64 %24
  %30 = getelementptr inbounds nuw i8, ptr %29, i64 16
  %31 = load <8 x i16>, ptr %29, align 2, !tbaa !63
  %32 = load <8 x i16>, ptr %30, align 2, !tbaa !63
  %33 = icmp sgt <8 x i16> %27, %31
  %34 = icmp sgt <8 x i16> %28, %32
  %35 = select <8 x i1> %33, <8 x i16> %22, <8 x i16> %20
  %36 = select <8 x i1> %34, <8 x i16> %23, <8 x i16> %21
  %37 = add nuw i64 %19, 16
  %38 = add <8 x i16> %22, splat (i16 16)
  %39 = icmp eq i64 %37, %16
  br i1 %39, label %40, label %18, !llvm.loop !170

40:                                               ; preds = %18
  %41 = tail call <8 x i16> @llvm.smax.v8i16(<8 x i16> %35, <8 x i16> %36)
  %42 = tail call i16 @llvm.vector.reduce.smax.v8i16(<8 x i16> %41)
  %43 = icmp eq i16 %42, -32768
  %44 = select i1 %43, i16 4, i16 %42
  %45 = icmp eq i64 %11, %16
  br i1 %45, label %98, label %46

46:                                               ; preds = %40
  %47 = or disjoint i64 %16, 3
  %48 = and i64 %11, 12
  %49 = icmp eq i64 %48, 0
  br i1 %49, label %83, label %50

50:                                               ; preds = %46, %13
  %51 = phi i64 [ %16, %46 ], [ 0, %13 ]
  %52 = phi i64 [ %17, %46 ], [ 3, %13 ]
  %53 = phi i16 [ %44, %46 ], [ 4, %13 ]
  %54 = icmp eq i16 %53, 4
  %55 = select i1 %54, i16 -32768, i16 %53
  %56 = and i64 %11, -4
  %57 = or i64 %11, 3
  %58 = insertelement <4 x i16> poison, i16 %55, i64 0
  %59 = shufflevector <4 x i16> %58, <4 x i16> poison, <4 x i32> zeroinitializer
  %60 = trunc i64 %52 to i16
  %61 = insertelement <4 x i16> poison, i16 %60, i64 0
  %62 = shufflevector <4 x i16> %61, <4 x i16> poison, <4 x i32> zeroinitializer
  %63 = add <4 x i16> %62, <i16 0, i16 1, i16 2, i16 3>
  br label %64

64:                                               ; preds = %64, %50
  %65 = phi i64 [ %51, %50 ], [ %75, %64 ]
  %66 = phi <4 x i16> [ %59, %50 ], [ %74, %64 ]
  %67 = phi <4 x i16> [ %63, %50 ], [ %76, %64 ]
  %68 = or disjoint i64 %65, 3
  %69 = getelementptr inbounds nuw i16, ptr %5, i64 %68
  %70 = load <4 x i16>, ptr %69, align 2, !tbaa !63
  %71 = getelementptr inbounds nuw i16, ptr %6, i64 %68
  %72 = load <4 x i16>, ptr %71, align 2, !tbaa !63
  %73 = icmp sgt <4 x i16> %70, %72
  %74 = select <4 x i1> %73, <4 x i16> %67, <4 x i16> %66
  %75 = add nuw i64 %65, 4
  %76 = add <4 x i16> %67, splat (i16 4)
  %77 = icmp eq i64 %75, %56
  br i1 %77, label %78, label %64, !llvm.loop !171

78:                                               ; preds = %64
  %79 = tail call i16 @llvm.vector.reduce.smax.v4i16(<4 x i16> %74)
  %80 = icmp eq i16 %79, -32768
  %81 = select i1 %80, i16 4, i16 %79
  %82 = icmp eq i64 %11, %56
  br i1 %82, label %98, label %83

83:                                               ; preds = %46, %78, %9
  %84 = phi i64 [ 3, %9 ], [ %47, %46 ], [ %57, %78 ]
  %85 = phi i16 [ 4, %9 ], [ %44, %46 ], [ %81, %78 ]
  br label %86

86:                                               ; preds = %83, %86
  %87 = phi i64 [ %96, %86 ], [ %84, %83 ]
  %88 = phi i16 [ %95, %86 ], [ %85, %83 ]
  %89 = getelementptr inbounds nuw i16, ptr %5, i64 %87
  %90 = load i16, ptr %89, align 2, !tbaa !63
  %91 = getelementptr inbounds nuw i16, ptr %6, i64 %87
  %92 = load i16, ptr %91, align 2, !tbaa !63
  %93 = icmp sgt i16 %90, %92
  %94 = trunc nuw nsw i64 %87 to i16
  %95 = select i1 %93, i16 %94, i16 %88
  %96 = add nuw nsw i64 %87, 1
  %97 = icmp eq i64 %96, %10
  br i1 %97, label %98, label %86, !llvm.loop !172

98:                                               ; preds = %86, %40, %78, %4
  %99 = phi i16 [ 4, %4 ], [ %44, %40 ], [ %81, %78 ], [ %95, %86 ]
  ret i16 %99
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFsPsS0_sEZ4mainE4$_35E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_35", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !83
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define internal void @_GLOBAL__sub_I_find_last.cpp() #19 section ".text.startup" {
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

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <4 x i32> @llvm.smax.v4i32(<4 x i32>, <4 x i32>) #20

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.vector.reduce.smax.v4i32(<4 x i32>) #20

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <8 x i16> @llvm.smax.v8i16(<8 x i16>, <8 x i16>) #20

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i16 @llvm.vector.reduce.smax.v8i16(<8 x i16>) #20

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i16 @llvm.vector.reduce.smax.v4i16(<4 x i16>) #20

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
!17 = !{!"_ZTSSt8functionIFiPiS0_iEE", !18, i64 0, !19, i64 24}
!18 = !{!"_ZTSSt14_Function_base", !8, i64 0, !19, i64 16}
!19 = !{!"any pointer", !8, i64 0}
!20 = !{!18, !19, i64 16}
!21 = !{!22, !19, i64 24}
!22 = !{!"_ZTSSt8functionIFiPfS0_iEE", !18, i64 0, !19, i64 24}
!23 = !{!24, !19, i64 24}
!24 = !{!"_ZTSSt8functionIFsPsS0_sEE", !18, i64 0, !19, i64 24}
!25 = !{!26, !26, i64 0}
!26 = !{!"vtable pointer", !9, i64 0}
!27 = !{!28, !30, i64 32}
!28 = !{!"_ZTSSt8ios_base", !7, i64 8, !7, i64 16, !29, i64 24, !30, i64 28, !30, i64 32, !31, i64 40, !32, i64 48, !8, i64 64, !33, i64 192, !34, i64 200, !35, i64 208}
!29 = !{!"_ZTSSt13_Ios_Fmtflags", !8, i64 0}
!30 = !{!"_ZTSSt12_Ios_Iostate", !8, i64 0}
!31 = !{!"p1 _ZTSNSt8ios_base14_Callback_listE", !19, i64 0}
!32 = !{!"_ZTSNSt8ios_base6_WordsE", !19, i64 0, !7, i64 8}
!33 = !{!"int", !8, i64 0}
!34 = !{!"p1 _ZTSNSt8ios_base6_WordsE", !19, i64 0}
!35 = !{!"_ZTSSt6locale", !36, i64 0}
!36 = !{!"p1 _ZTSNSt6locale5_ImplE", !19, i64 0}
!37 = !{!33, !33, i64 0}
!38 = distinct !{!38, !11}
!39 = !{!40, !40, i64 0}
!40 = !{!"p1 int", !19, i64 0}
!41 = distinct !{!41, !11, !42, !43}
!42 = !{!"llvm.loop.isvectorized", i32 1}
!43 = !{!"llvm.loop.unroll.runtime.disable"}
!44 = distinct !{!44, !11, !42, !43}
!45 = distinct !{!45, !11, !42, !43}
!46 = distinct !{!46, !11, !42, !43}
!47 = distinct !{!47, !11, !42, !43}
!48 = distinct !{!48, !11, !42, !43}
!49 = !{!50, !50, i64 0}
!50 = !{!"p1 float", !19, i64 0}
!51 = !{!52, !52, i64 0}
!52 = !{!"float", !8, i64 0}
!53 = distinct !{!53, !11, !42, !43}
!54 = distinct !{!54, !11, !42, !43}
!55 = distinct !{!55, !11, !42, !43}
!56 = distinct !{!56, !11, !42, !43}
!57 = distinct !{!57, !11, !42, !43}
!58 = distinct !{!58, !11, !42, !43}
!59 = !{!60, !61, i64 0}
!60 = !{!"_ZTSNSt24uniform_int_distributionIsE10param_typeE", !61, i64 0, !61, i64 2}
!61 = !{!"short", !8, i64 0}
!62 = !{!60, !61, i64 2}
!63 = !{!61, !61, i64 0}
!64 = distinct !{!64, !11}
!65 = !{!66, !66, i64 0}
!66 = !{!"p1 short", !19, i64 0}
!67 = distinct !{!67, !11, !42, !43}
!68 = distinct !{!68, !11, !42, !43}
!69 = distinct !{!69, !11, !42, !43}
!70 = distinct !{!70, !11, !42, !43}
!71 = distinct !{!71, !11, !42, !43}
!72 = distinct !{!72, !11, !42, !43}
!73 = !{!74, !33, i64 4}
!74 = !{!"_ZTSNSt24uniform_int_distributionIiE10param_typeE", !33, i64 0, !33, i64 4}
!75 = !{!74, !33, i64 0}
!76 = distinct !{!76, !11}
!77 = distinct !{!77, !11}
!78 = distinct !{!78, !11, !42, !43}
!79 = distinct !{!79, !11, !42, !43}
!80 = distinct !{!80, !11, !81, !82}
!81 = !{!"llvm.loop.vectorize.width", i32 1}
!82 = !{!"llvm.loop.interleave.count", i32 1}
!83 = !{!19, !19, i64 0}
!84 = distinct !{!84, !11, !42, !43}
!85 = distinct !{!85, !11, !43, !42}
!86 = !{!"branch_weights", !"expected", i32 2000, i32 1}
!87 = distinct !{!87, !11, !42, !43}
!88 = distinct !{!88, !11, !42, !43}
!89 = distinct !{!89, !11}
!90 = distinct !{!90, !11}
!91 = distinct !{!91, !11, !81, !82}
!92 = distinct !{!92, !11, !42, !43}
!93 = distinct !{!93, !11, !43, !42}
!94 = distinct !{!94, !11}
!95 = distinct !{!95, !11}
!96 = distinct !{!96, !11, !81, !82}
!97 = distinct !{!97, !11, !42, !43}
!98 = distinct !{!98, !11, !42, !43}
!99 = distinct !{!99, !11, !43, !42}
!100 = distinct !{!100, !11, !81, !82}
!101 = distinct !{!101, !11, !42, !43}
!102 = distinct !{!102, !11, !43, !42}
!103 = distinct !{!103, !11, !81, !82}
!104 = distinct !{!104, !11, !42, !43}
!105 = distinct !{!105, !11, !43, !42}
!106 = distinct !{!106, !11, !81, !82}
!107 = distinct !{!107, !11, !42, !43}
!108 = distinct !{!108, !11, !42, !43}
!109 = distinct !{!109, !11, !43, !42}
!110 = distinct !{!110, !11, !81, !82}
!111 = distinct !{!111, !11, !42, !43}
!112 = distinct !{!112, !11, !43, !42}
!113 = distinct !{!113, !11, !81, !82}
!114 = distinct !{!114, !11, !42, !43}
!115 = distinct !{!115, !11, !43, !42}
!116 = distinct !{!116, !11, !81, !82}
!117 = distinct !{!117, !11, !42, !43}
!118 = distinct !{!118, !11, !42, !43}
!119 = distinct !{!119, !11, !43, !42}
!120 = distinct !{!120, !11, !81, !82}
!121 = distinct !{!121, !11, !122}
!122 = !{!"llvm.loop.vectorize.enable", i1 true}
!123 = distinct !{!123, !11, !81, !82}
!124 = distinct !{!124, !11, !122}
!125 = distinct !{!125, !11, !81, !82}
!126 = distinct !{!126, !11, !122}
!127 = distinct !{!127, !11, !81, !82}
!128 = distinct !{!128, !11, !122}
!129 = distinct !{!129, !11, !81, !82}
!130 = distinct !{!130, !11, !122}
!131 = distinct !{!131, !11, !81, !82}
!132 = distinct !{!132, !11, !122}
!133 = distinct !{!133, !11, !81, !82}
!134 = distinct !{!134, !11, !42, !43}
!135 = distinct !{!135, !11, !43, !42}
!136 = distinct !{!136, !11, !81, !82}
!137 = distinct !{!137, !11, !42, !43}
!138 = distinct !{!138, !11, !43, !42}
!139 = distinct !{!139, !11, !81, !82}
!140 = distinct !{!140, !11, !42, !43}
!141 = distinct !{!141, !11, !42, !43}
!142 = distinct !{!142, !11, !43, !42}
!143 = distinct !{!143, !11, !81, !82}
!144 = distinct !{!144, !11, !42, !43}
!145 = distinct !{!145, !11, !43, !42}
!146 = distinct !{!146, !11, !81, !82}
!147 = distinct !{!147, !11, !42, !43}
!148 = distinct !{!148, !11, !43, !42}
!149 = distinct !{!149, !11, !81, !82}
!150 = distinct !{!150, !11, !42, !43}
!151 = distinct !{!151, !11, !42, !43}
!152 = distinct !{!152, !11, !43, !42}
!153 = distinct !{!153, !11, !81, !82}
!154 = distinct !{!154, !11, !42, !43}
!155 = distinct !{!155, !11, !43, !42}
!156 = distinct !{!156, !11, !81, !82}
!157 = distinct !{!157, !11, !42, !43}
!158 = distinct !{!158, !11, !43, !42}
!159 = distinct !{!159, !11, !81, !82}
!160 = distinct !{!160, !11, !42, !43}
!161 = distinct !{!161, !11, !42, !43}
!162 = distinct !{!162, !11, !43, !42}
!163 = distinct !{!163, !11, !81, !82}
!164 = distinct !{!164, !11, !42, !43}
!165 = distinct !{!165, !11, !43, !42}
!166 = distinct !{!166, !11, !81, !82}
!167 = distinct !{!167, !11, !42, !43}
!168 = distinct !{!168, !11, !43, !42}
!169 = distinct !{!169, !11, !81, !82}
!170 = distinct !{!170, !11, !42, !43}
!171 = distinct !{!171, !11, !42, !43}
!172 = distinct !{!172, !11, !43, !42}
