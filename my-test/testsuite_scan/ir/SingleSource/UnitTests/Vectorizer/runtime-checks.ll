; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/Vectorizer/runtime-checks.cpp'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/Vectorizer/runtime-checks.cpp"
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
%"class.std::function.4" = type { %"class.std::_Function_base", ptr }
%"class.std::function.38" = type { %"class.std::_Function_base", ptr }
%"class.std::function.40" = type { %"class.std::_Function_base", ptr }
%"class.std::function.42" = type { %"class.std::_Function_base", ptr }
%"class.std::function.52" = type { %"class.std::_Function_base", ptr }
%"class.std::function.54" = type { %"class.std::_Function_base", ptr }
%"class.std::function.56" = type { %"class.std::_Function_base", ptr }
%"class.std::uniform_int_distribution" = type { %"struct.std::uniform_int_distribution<unsigned char>::param_type" }
%"struct.std::uniform_int_distribution<unsigned char>::param_type" = type { i8, i8 }
%"class.std::uniform_int_distribution.84" = type { %"struct.std::uniform_int_distribution<unsigned int>::param_type" }
%"struct.std::uniform_int_distribution<unsigned int>::param_type" = type { i32, i32 }
%"class.std::uniform_int_distribution.96" = type { %"struct.std::uniform_int_distribution<unsigned long>::param_type" }
%"struct.std::uniform_int_distribution<unsigned long>::param_type" = type { i64, i64 }

$__clang_call_terminate = comdat any

$_ZNKSt8functionIFvPhS0_jEEclES0_S0_j = comdat any

$_ZNSt24uniform_int_distributionIhEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEhRT_RKNS0_10param_typeE = comdat any

$_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv = comdat any

$_ZNKSt8functionIFvPjS0_jEEclES0_S0_j = comdat any

$_ZNSt24uniform_int_distributionIjEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEjRT_RKNS0_10param_typeE = comdat any

$_ZNKSt8functionIFvPmS0_jEEclES0_S0_j = comdat any

$_ZNSt24uniform_int_distributionImEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEmRT_RKNS0_10param_typeE = comdat any

$_ZNKSt8functionIFvPjS0_S0_jEEclES0_S0_S0_j = comdat any

$_ZNKSt8functionIFvPhS0_S0_jEEclES0_S0_S0_j = comdat any

$_ZNKSt8functionIFvPmS0_S0_jEEclES0_S0_S0_j = comdat any

$_ZNKSt8functionIFvPhS0_jjEEclES0_S0_jj = comdat any

$_ZNKSt8functionIFvPjS0_jjEEclES0_S0_jj = comdat any

$_ZNKSt8functionIFvPmS0_jjEEclES0_S0_jj = comdat any

@_ZL3rng = internal global %"class.std::mersenne_twister_engine" zeroinitializer, align 8
@.str = private unnamed_addr constant [33 x i8] c"1 read, 1 write, step 1, uint8_t\00", align 1
@.str.1 = private unnamed_addr constant [34 x i8] c"1 read, 1 write, step 1, uint32_t\00", align 1
@.str.2 = private unnamed_addr constant [34 x i8] c"1 read, 1 write, step 1, uint64_t\00", align 1
@.str.3 = private unnamed_addr constant [35 x i8] c"1 read, 1 write, offset 3, uint8_t\00", align 1
@.str.4 = private unnamed_addr constant [36 x i8] c"1 read, 1 write, offset 3, uint32_t\00", align 1
@.str.5 = private unnamed_addr constant [36 x i8] c"1 read, 1 write, offset 3, uint64_t\00", align 1
@.str.6 = private unnamed_addr constant [36 x i8] c"1 read, 1 write, offset -3, uint8_t\00", align 1
@.str.7 = private unnamed_addr constant [37 x i8] c"1 read, 1 write, offset -3, uint32_t\00", align 1
@.str.8 = private unnamed_addr constant [37 x i8] c"1 read, 1 write, offset -3, uint64_t\00", align 1
@.str.9 = private unnamed_addr constant [43 x i8] c"1 read, 1 write, index count down, uint8_t\00", align 1
@.str.10 = private unnamed_addr constant [44 x i8] c"1 read, 1 write, index count down, uint32_t\00", align 1
@.str.11 = private unnamed_addr constant [44 x i8] c"1 read, 1 write, index count down, uint64_t\00", align 1
@.str.12 = private unnamed_addr constant [45 x i8] c"1 read, 1 write, index count down 2, uint8_t\00", align 1
@.str.13 = private unnamed_addr constant [46 x i8] c"1 read, 1 write, index count down 2, uint32_t\00", align 1
@.str.14 = private unnamed_addr constant [46 x i8] c"1 read, 1 write, index count down 2, uint64_t\00", align 1
@.str.15 = private unnamed_addr constant [56 x i8] c"1 read, 1 write, 2 inductions, different steps, uint8_t\00", align 1
@.str.16 = private unnamed_addr constant [57 x i8] c"1 read, 1 write, 2 inductions, different steps, uint32_t\00", align 1
@.str.17 = private unnamed_addr constant [57 x i8] c"1 read, 1 write, 2 inductions, different steps, uint64_t\00", align 1
@.str.18 = private unnamed_addr constant [48 x i8] c"1 read, 1 write, induction increment 2, uint8_t\00", align 1
@.str.19 = private unnamed_addr constant [49 x i8] c"1 read, 1 write, induction increment 2, uint32_t\00", align 1
@.str.20 = private unnamed_addr constant [49 x i8] c"1 read, 1 write, induction increment 2, uint64_t\00", align 1
@.str.21 = private unnamed_addr constant [54 x i8] c"1 read, 1 write to invariant address, step 1, uint8_t\00", align 1
@.str.22 = private unnamed_addr constant [55 x i8] c"1 read, 1 write to invariant address, step 1, uint32_t\00", align 1
@.str.23 = private unnamed_addr constant [55 x i8] c"1 read, 1 write to invariant address, step 1, uint64_t\00", align 1
@.str.24 = private unnamed_addr constant [43 x i8] c"2 reads, 1 write, simple indices, uint32_t\00", align 1
@.str.25 = private unnamed_addr constant [42 x i8] c"2 reads, 1 write, simple indices, uint8_t\00", align 1
@.str.26 = private unnamed_addr constant [43 x i8] c"2 reads, 1 write, simple indices, uint64_t\00", align 1
@.str.27 = private unnamed_addr constant [42 x i8] c"1 read, 2 writes, simple indices, uint8_t\00", align 1
@.str.28 = private unnamed_addr constant [43 x i8] c"1 read, 2 writes, simple indices, uint32_t\00", align 1
@.str.29 = private unnamed_addr constant [43 x i8] c"1 read, 2 writes, simple indices, uint64_t\00", align 1
@.str.30 = private unnamed_addr constant [61 x i8] c"1 read, 1 write, nested loop (matching trip counts), uint8_t\00", align 1
@.str.31 = private unnamed_addr constant [62 x i8] c"1 read, 1 write, nested loop (matching trip counts), uint32_t\00", align 1
@.str.32 = private unnamed_addr constant [62 x i8] c"1 read, 1 write, nested loop (matching trip counts), uint64_t\00", align 1
@.str.33 = private unnamed_addr constant [62 x i8] c"1 read, 1 write, nested loop (different trip counts), uint8_t\00", align 1
@.str.34 = private unnamed_addr constant [63 x i8] c"1 read, 1 write, nested loop (different trip counts), uint32_t\00", align 1
@.str.35 = private unnamed_addr constant [63 x i8] c"1 read, 1 write, nested loop (different trip counts), uint64_t\00", align 1
@.str.36 = private unnamed_addr constant [62 x i8] c"2 reads, 1 write, nested loop (matching trip counts), uint8_t\00", align 1
@.str.37 = private unnamed_addr constant [63 x i8] c"2 reads, 1 write, nested loop (matching trip counts), uint32_t\00", align 1
@.str.38 = private unnamed_addr constant [63 x i8] c"2 reads, 1 write, nested loop (matching trip counts), uint64_t\00", align 1
@.str.39 = private unnamed_addr constant [63 x i8] c"2 reads, 1 write, nested loop (different trip counts), uint8_t\00", align 1
@.str.40 = private unnamed_addr constant [64 x i8] c"2 reads, 1 write, nested loop (different trip counts), uint32_t\00", align 1
@.str.41 = private unnamed_addr constant [64 x i8] c"2 reads, 1 write, nested loop (different trip counts), uint64_t\00", align 1
@.str.42 = private unnamed_addr constant [82 x i8] c"1 read, 1 write, nested loop (decreasing outer iv, matching trip counts), uint8_t\00", align 1
@.str.43 = private unnamed_addr constant [83 x i8] c"1 read, 1 write, nested loop (decreasing outer iv, matching trip counts), uint32_t\00", align 1
@.str.44 = private unnamed_addr constant [83 x i8] c"1 read, 1 write, nested loop (decreasing outer iv, matching trip counts), uint64_t\00", align 1
@.str.45 = private unnamed_addr constant [83 x i8] c"2 reads, 1 write, nested loop (decreasing outer iv, matching trip counts), uint8_t\00", align 1
@.str.46 = private unnamed_addr constant [84 x i8] c"2 reads, 1 write, nested loop (decreasing outer iv, matching trip counts), uint32_t\00", align 1
@.str.47 = private unnamed_addr constant [84 x i8] c"2 reads, 1 write, nested loop (decreasing outer iv, matching trip counts), uint64_t\00", align 1
@_ZSt4cout = external global %"class.std::basic_ostream", align 8
@.str.48 = private unnamed_addr constant [10 x i8] c"Checking \00", align 1
@.str.49 = private unnamed_addr constant [2 x i8] c"\0A\00", align 1
@_ZSt4cerr = external global %"class.std::basic_ostream", align 8
@.str.50 = private unnamed_addr constant [24 x i8] c"Miscompare with offset \00", align 1
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
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_runtime_checks.cpp, ptr null }]

; Function Attrs: mustprogress norecurse uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 personality ptr @__gxx_personality_v0 {
  %1 = alloca %"class.std::mersenne_twister_engine", align 8
  %2 = alloca %"class.std::function", align 8
  %3 = alloca %"class.std::function", align 8
  %4 = alloca %"class.std::function.2", align 8
  %5 = alloca %"class.std::function.2", align 8
  %6 = alloca %"class.std::function.4", align 8
  %7 = alloca %"class.std::function.4", align 8
  %8 = alloca %"class.std::function", align 8
  %9 = alloca %"class.std::function", align 8
  %10 = alloca %"class.std::function.2", align 8
  %11 = alloca %"class.std::function.2", align 8
  %12 = alloca %"class.std::function.4", align 8
  %13 = alloca %"class.std::function.4", align 8
  %14 = alloca %"class.std::function", align 8
  %15 = alloca %"class.std::function", align 8
  %16 = alloca %"class.std::function.2", align 8
  %17 = alloca %"class.std::function.2", align 8
  %18 = alloca %"class.std::function.4", align 8
  %19 = alloca %"class.std::function.4", align 8
  %20 = alloca %"class.std::function", align 8
  %21 = alloca %"class.std::function", align 8
  %22 = alloca %"class.std::function.2", align 8
  %23 = alloca %"class.std::function.2", align 8
  %24 = alloca %"class.std::function.4", align 8
  %25 = alloca %"class.std::function.4", align 8
  %26 = alloca %"class.std::function", align 8
  %27 = alloca %"class.std::function", align 8
  %28 = alloca %"class.std::function.2", align 8
  %29 = alloca %"class.std::function.2", align 8
  %30 = alloca %"class.std::function.4", align 8
  %31 = alloca %"class.std::function.4", align 8
  %32 = alloca %"class.std::function", align 8
  %33 = alloca %"class.std::function", align 8
  %34 = alloca %"class.std::function.2", align 8
  %35 = alloca %"class.std::function.2", align 8
  %36 = alloca %"class.std::function.4", align 8
  %37 = alloca %"class.std::function.4", align 8
  %38 = alloca %"class.std::function", align 8
  %39 = alloca %"class.std::function", align 8
  %40 = alloca %"class.std::function.2", align 8
  %41 = alloca %"class.std::function.2", align 8
  %42 = alloca %"class.std::function.4", align 8
  %43 = alloca %"class.std::function.4", align 8
  %44 = alloca %"class.std::function", align 8
  %45 = alloca %"class.std::function", align 8
  %46 = alloca %"class.std::function.2", align 8
  %47 = alloca %"class.std::function.2", align 8
  %48 = alloca %"class.std::function.4", align 8
  %49 = alloca %"class.std::function.4", align 8
  %50 = alloca %"class.std::function.38", align 8
  %51 = alloca %"class.std::function.38", align 8
  %52 = alloca %"class.std::function.40", align 8
  %53 = alloca %"class.std::function.40", align 8
  %54 = alloca %"class.std::function.42", align 8
  %55 = alloca %"class.std::function.42", align 8
  %56 = alloca %"class.std::function.40", align 8
  %57 = alloca %"class.std::function.40", align 8
  %58 = alloca %"class.std::function.38", align 8
  %59 = alloca %"class.std::function.38", align 8
  %60 = alloca %"class.std::function.42", align 8
  %61 = alloca %"class.std::function.42", align 8
  %62 = alloca %"class.std::function.52", align 8
  %63 = alloca %"class.std::function.52", align 8
  %64 = alloca %"class.std::function.54", align 8
  %65 = alloca %"class.std::function.54", align 8
  %66 = alloca %"class.std::function.56", align 8
  %67 = alloca %"class.std::function.56", align 8
  %68 = alloca %"class.std::function.52", align 8
  %69 = alloca %"class.std::function.52", align 8
  %70 = alloca %"class.std::function.54", align 8
  %71 = alloca %"class.std::function.54", align 8
  %72 = alloca %"class.std::function.56", align 8
  %73 = alloca %"class.std::function.56", align 8
  %74 = alloca %"class.std::function.52", align 8
  %75 = alloca %"class.std::function.52", align 8
  %76 = alloca %"class.std::function.54", align 8
  %77 = alloca %"class.std::function.54", align 8
  %78 = alloca %"class.std::function.56", align 8
  %79 = alloca %"class.std::function.56", align 8
  %80 = alloca %"class.std::function.52", align 8
  %81 = alloca %"class.std::function.52", align 8
  %82 = alloca %"class.std::function.54", align 8
  %83 = alloca %"class.std::function.54", align 8
  %84 = alloca %"class.std::function.56", align 8
  %85 = alloca %"class.std::function.56", align 8
  %86 = alloca %"class.std::function.52", align 8
  %87 = alloca %"class.std::function.52", align 8
  %88 = alloca %"class.std::function.54", align 8
  %89 = alloca %"class.std::function.54", align 8
  %90 = alloca %"class.std::function.56", align 8
  %91 = alloca %"class.std::function.56", align 8
  %92 = alloca %"class.std::function.52", align 8
  %93 = alloca %"class.std::function.52", align 8
  %94 = alloca %"class.std::function.54", align 8
  %95 = alloca %"class.std::function.54", align 8
  %96 = alloca %"class.std::function.56", align 8
  %97 = alloca %"class.std::function.56", align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #20
  store i64 15, ptr %1, align 8, !tbaa !6
  br label %98

98:                                               ; preds = %98, %0
  %99 = phi i64 [ 15, %0 ], [ %106, %98 ]
  %100 = phi i64 [ 1, %0 ], [ %107, %98 ]
  %101 = getelementptr i64, ptr %1, i64 %100
  %102 = lshr i64 %99, 30
  %103 = xor i64 %102, %99
  %104 = mul nuw nsw i64 %103, 1812433253
  %105 = add nuw i64 %104, %100
  %106 = and i64 %105, 4294967295
  store i64 %106, ptr %101, align 8, !tbaa !6
  %107 = add nuw nsw i64 %100, 1
  %108 = icmp eq i64 %107, 624
  br i1 %108, label %109, label %98, !llvm.loop !10

109:                                              ; preds = %98
  %110 = getelementptr inbounds nuw i8, ptr %1, i64 4992
  store i64 624, ptr %110, align 8, !tbaa !12
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(5000) @_ZL3rng, ptr noundef nonnull align 8 dereferenceable(5000) %1, i64 5000, i1 false), !tbaa.struct !14
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #20
  %111 = getelementptr inbounds nuw i8, ptr %2, i64 16
  %112 = getelementptr inbounds nuw i8, ptr %2, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %2, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPhS0_jEZ4mainE3$_0E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %112, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFvPhS0_jEZ4mainE3$_0E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %111, align 8, !tbaa !20
  %113 = getelementptr inbounds nuw i8, ptr %3, i64 16
  %114 = getelementptr inbounds nuw i8, ptr %3, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %3, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPhS0_jEZ4mainE3$_1E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %114, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFvPhS0_jEZ4mainE3$_1E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %113, align 8, !tbaa !20
  invoke fastcc void @_ZL37checkOverlappingMemoryOneRuntimeCheckIhEvSt8functionIFvPT_S2_jEES4_PKc(ptr noundef %2, ptr noundef %3, ptr noundef nonnull @.str)
          to label %115 unwind label %1119

115:                                              ; preds = %109
  %116 = load ptr, ptr %113, align 8, !tbaa !20
  %117 = icmp eq ptr %116, null
  br i1 %117, label %123, label %118

118:                                              ; preds = %115
  %119 = invoke noundef i1 %116(ptr noundef nonnull align 8 dereferenceable(32) %3, ptr noundef nonnull align 8 dereferenceable(32) %3, i32 noundef 3)
          to label %123 unwind label %120

120:                                              ; preds = %118
  %121 = landingpad { ptr, i32 }
          catch ptr null
  %122 = extractvalue { ptr, i32 } %121, 0
  call void @__clang_call_terminate(ptr %122) #21
  unreachable

123:                                              ; preds = %115, %118
  %124 = load ptr, ptr %111, align 8, !tbaa !20
  %125 = icmp eq ptr %124, null
  br i1 %125, label %131, label %126

126:                                              ; preds = %123
  %127 = invoke noundef i1 %124(ptr noundef nonnull align 8 dereferenceable(32) %2, ptr noundef nonnull align 8 dereferenceable(32) %2, i32 noundef 3)
          to label %131 unwind label %128

128:                                              ; preds = %126
  %129 = landingpad { ptr, i32 }
          catch ptr null
  %130 = extractvalue { ptr, i32 } %129, 0
  call void @__clang_call_terminate(ptr %130) #21
  unreachable

131:                                              ; preds = %123, %126
  %132 = getelementptr inbounds nuw i8, ptr %4, i64 16
  %133 = getelementptr inbounds nuw i8, ptr %4, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %4, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE3$_0E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %133, align 8, !tbaa !21
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE3$_0E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %132, align 8, !tbaa !20
  %134 = getelementptr inbounds nuw i8, ptr %5, i64 16
  %135 = getelementptr inbounds nuw i8, ptr %5, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %5, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE3$_1E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %135, align 8, !tbaa !21
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE3$_1E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %134, align 8, !tbaa !20
  invoke fastcc void @_ZL37checkOverlappingMemoryOneRuntimeCheckIjEvSt8functionIFvPT_S2_jEES4_PKc(ptr noundef %4, ptr noundef %5, ptr noundef nonnull @.str.1)
          to label %136 unwind label %1136

136:                                              ; preds = %131
  %137 = load ptr, ptr %134, align 8, !tbaa !20
  %138 = icmp eq ptr %137, null
  br i1 %138, label %144, label %139

139:                                              ; preds = %136
  %140 = invoke noundef i1 %137(ptr noundef nonnull align 8 dereferenceable(32) %5, ptr noundef nonnull align 8 dereferenceable(32) %5, i32 noundef 3)
          to label %144 unwind label %141

141:                                              ; preds = %139
  %142 = landingpad { ptr, i32 }
          catch ptr null
  %143 = extractvalue { ptr, i32 } %142, 0
  call void @__clang_call_terminate(ptr %143) #21
  unreachable

144:                                              ; preds = %136, %139
  %145 = load ptr, ptr %132, align 8, !tbaa !20
  %146 = icmp eq ptr %145, null
  br i1 %146, label %152, label %147

147:                                              ; preds = %144
  %148 = invoke noundef i1 %145(ptr noundef nonnull align 8 dereferenceable(32) %4, ptr noundef nonnull align 8 dereferenceable(32) %4, i32 noundef 3)
          to label %152 unwind label %149

149:                                              ; preds = %147
  %150 = landingpad { ptr, i32 }
          catch ptr null
  %151 = extractvalue { ptr, i32 } %150, 0
  call void @__clang_call_terminate(ptr %151) #21
  unreachable

152:                                              ; preds = %144, %147
  %153 = getelementptr inbounds nuw i8, ptr %6, i64 16
  %154 = getelementptr inbounds nuw i8, ptr %6, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %6, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPmS0_jEZ4mainE3$_0E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %154, align 8, !tbaa !23
  store ptr @"_ZNSt17_Function_handlerIFvPmS0_jEZ4mainE3$_0E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %153, align 8, !tbaa !20
  %155 = getelementptr inbounds nuw i8, ptr %7, i64 16
  %156 = getelementptr inbounds nuw i8, ptr %7, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %7, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPmS0_jEZ4mainE3$_1E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %156, align 8, !tbaa !23
  store ptr @"_ZNSt17_Function_handlerIFvPmS0_jEZ4mainE3$_1E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %155, align 8, !tbaa !20
  invoke fastcc void @_ZL37checkOverlappingMemoryOneRuntimeCheckImEvSt8functionIFvPT_S2_jEES4_PKc(ptr noundef %6, ptr noundef %7, ptr noundef nonnull @.str.2)
          to label %157 unwind label %1153

157:                                              ; preds = %152
  %158 = load ptr, ptr %155, align 8, !tbaa !20
  %159 = icmp eq ptr %158, null
  br i1 %159, label %165, label %160

160:                                              ; preds = %157
  %161 = invoke noundef i1 %158(ptr noundef nonnull align 8 dereferenceable(32) %7, ptr noundef nonnull align 8 dereferenceable(32) %7, i32 noundef 3)
          to label %165 unwind label %162

162:                                              ; preds = %160
  %163 = landingpad { ptr, i32 }
          catch ptr null
  %164 = extractvalue { ptr, i32 } %163, 0
  call void @__clang_call_terminate(ptr %164) #21
  unreachable

165:                                              ; preds = %157, %160
  %166 = load ptr, ptr %153, align 8, !tbaa !20
  %167 = icmp eq ptr %166, null
  br i1 %167, label %173, label %168

168:                                              ; preds = %165
  %169 = invoke noundef i1 %166(ptr noundef nonnull align 8 dereferenceable(32) %6, ptr noundef nonnull align 8 dereferenceable(32) %6, i32 noundef 3)
          to label %173 unwind label %170

170:                                              ; preds = %168
  %171 = landingpad { ptr, i32 }
          catch ptr null
  %172 = extractvalue { ptr, i32 } %171, 0
  call void @__clang_call_terminate(ptr %172) #21
  unreachable

173:                                              ; preds = %165, %168
  %174 = getelementptr inbounds nuw i8, ptr %8, i64 16
  %175 = getelementptr inbounds nuw i8, ptr %8, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %8, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPhS0_jEZ4mainE3$_2E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %175, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFvPhS0_jEZ4mainE3$_2E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %174, align 8, !tbaa !20
  %176 = getelementptr inbounds nuw i8, ptr %9, i64 16
  %177 = getelementptr inbounds nuw i8, ptr %9, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %9, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPhS0_jEZ4mainE3$_3E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %177, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFvPhS0_jEZ4mainE3$_3E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %176, align 8, !tbaa !20
  invoke fastcc void @_ZL37checkOverlappingMemoryOneRuntimeCheckIhEvSt8functionIFvPT_S2_jEES4_PKc(ptr noundef %8, ptr noundef %9, ptr noundef nonnull @.str.3)
          to label %178 unwind label %1170

178:                                              ; preds = %173
  %179 = load ptr, ptr %176, align 8, !tbaa !20
  %180 = icmp eq ptr %179, null
  br i1 %180, label %186, label %181

181:                                              ; preds = %178
  %182 = invoke noundef i1 %179(ptr noundef nonnull align 8 dereferenceable(32) %9, ptr noundef nonnull align 8 dereferenceable(32) %9, i32 noundef 3)
          to label %186 unwind label %183

183:                                              ; preds = %181
  %184 = landingpad { ptr, i32 }
          catch ptr null
  %185 = extractvalue { ptr, i32 } %184, 0
  call void @__clang_call_terminate(ptr %185) #21
  unreachable

186:                                              ; preds = %178, %181
  %187 = load ptr, ptr %174, align 8, !tbaa !20
  %188 = icmp eq ptr %187, null
  br i1 %188, label %194, label %189

189:                                              ; preds = %186
  %190 = invoke noundef i1 %187(ptr noundef nonnull align 8 dereferenceable(32) %8, ptr noundef nonnull align 8 dereferenceable(32) %8, i32 noundef 3)
          to label %194 unwind label %191

191:                                              ; preds = %189
  %192 = landingpad { ptr, i32 }
          catch ptr null
  %193 = extractvalue { ptr, i32 } %192, 0
  call void @__clang_call_terminate(ptr %193) #21
  unreachable

194:                                              ; preds = %186, %189
  %195 = getelementptr inbounds nuw i8, ptr %10, i64 16
  %196 = getelementptr inbounds nuw i8, ptr %10, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %10, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE3$_2E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %196, align 8, !tbaa !21
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE3$_2E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %195, align 8, !tbaa !20
  %197 = getelementptr inbounds nuw i8, ptr %11, i64 16
  %198 = getelementptr inbounds nuw i8, ptr %11, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %11, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE3$_3E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %198, align 8, !tbaa !21
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE3$_3E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %197, align 8, !tbaa !20
  invoke fastcc void @_ZL37checkOverlappingMemoryOneRuntimeCheckIjEvSt8functionIFvPT_S2_jEES4_PKc(ptr noundef %10, ptr noundef %11, ptr noundef nonnull @.str.4)
          to label %199 unwind label %1187

199:                                              ; preds = %194
  %200 = load ptr, ptr %197, align 8, !tbaa !20
  %201 = icmp eq ptr %200, null
  br i1 %201, label %207, label %202

202:                                              ; preds = %199
  %203 = invoke noundef i1 %200(ptr noundef nonnull align 8 dereferenceable(32) %11, ptr noundef nonnull align 8 dereferenceable(32) %11, i32 noundef 3)
          to label %207 unwind label %204

204:                                              ; preds = %202
  %205 = landingpad { ptr, i32 }
          catch ptr null
  %206 = extractvalue { ptr, i32 } %205, 0
  call void @__clang_call_terminate(ptr %206) #21
  unreachable

207:                                              ; preds = %199, %202
  %208 = load ptr, ptr %195, align 8, !tbaa !20
  %209 = icmp eq ptr %208, null
  br i1 %209, label %215, label %210

210:                                              ; preds = %207
  %211 = invoke noundef i1 %208(ptr noundef nonnull align 8 dereferenceable(32) %10, ptr noundef nonnull align 8 dereferenceable(32) %10, i32 noundef 3)
          to label %215 unwind label %212

212:                                              ; preds = %210
  %213 = landingpad { ptr, i32 }
          catch ptr null
  %214 = extractvalue { ptr, i32 } %213, 0
  call void @__clang_call_terminate(ptr %214) #21
  unreachable

215:                                              ; preds = %207, %210
  %216 = getelementptr inbounds nuw i8, ptr %12, i64 16
  %217 = getelementptr inbounds nuw i8, ptr %12, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %12, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPmS0_jEZ4mainE3$_2E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %217, align 8, !tbaa !23
  store ptr @"_ZNSt17_Function_handlerIFvPmS0_jEZ4mainE3$_2E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %216, align 8, !tbaa !20
  %218 = getelementptr inbounds nuw i8, ptr %13, i64 16
  %219 = getelementptr inbounds nuw i8, ptr %13, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %13, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPmS0_jEZ4mainE3$_3E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %219, align 8, !tbaa !23
  store ptr @"_ZNSt17_Function_handlerIFvPmS0_jEZ4mainE3$_3E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %218, align 8, !tbaa !20
  invoke fastcc void @_ZL37checkOverlappingMemoryOneRuntimeCheckImEvSt8functionIFvPT_S2_jEES4_PKc(ptr noundef %12, ptr noundef %13, ptr noundef nonnull @.str.5)
          to label %220 unwind label %1204

220:                                              ; preds = %215
  %221 = load ptr, ptr %218, align 8, !tbaa !20
  %222 = icmp eq ptr %221, null
  br i1 %222, label %228, label %223

223:                                              ; preds = %220
  %224 = invoke noundef i1 %221(ptr noundef nonnull align 8 dereferenceable(32) %13, ptr noundef nonnull align 8 dereferenceable(32) %13, i32 noundef 3)
          to label %228 unwind label %225

225:                                              ; preds = %223
  %226 = landingpad { ptr, i32 }
          catch ptr null
  %227 = extractvalue { ptr, i32 } %226, 0
  call void @__clang_call_terminate(ptr %227) #21
  unreachable

228:                                              ; preds = %220, %223
  %229 = load ptr, ptr %216, align 8, !tbaa !20
  %230 = icmp eq ptr %229, null
  br i1 %230, label %236, label %231

231:                                              ; preds = %228
  %232 = invoke noundef i1 %229(ptr noundef nonnull align 8 dereferenceable(32) %12, ptr noundef nonnull align 8 dereferenceable(32) %12, i32 noundef 3)
          to label %236 unwind label %233

233:                                              ; preds = %231
  %234 = landingpad { ptr, i32 }
          catch ptr null
  %235 = extractvalue { ptr, i32 } %234, 0
  call void @__clang_call_terminate(ptr %235) #21
  unreachable

236:                                              ; preds = %228, %231
  %237 = getelementptr inbounds nuw i8, ptr %14, i64 16
  %238 = getelementptr inbounds nuw i8, ptr %14, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %14, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPhS0_jEZ4mainE3$_4E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %238, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFvPhS0_jEZ4mainE3$_4E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %237, align 8, !tbaa !20
  %239 = getelementptr inbounds nuw i8, ptr %15, i64 16
  %240 = getelementptr inbounds nuw i8, ptr %15, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %15, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPhS0_jEZ4mainE3$_5E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %240, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFvPhS0_jEZ4mainE3$_5E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %239, align 8, !tbaa !20
  invoke fastcc void @_ZL37checkOverlappingMemoryOneRuntimeCheckIhEvSt8functionIFvPT_S2_jEES4_PKc(ptr noundef %14, ptr noundef %15, ptr noundef nonnull @.str.6)
          to label %241 unwind label %1221

241:                                              ; preds = %236
  %242 = load ptr, ptr %239, align 8, !tbaa !20
  %243 = icmp eq ptr %242, null
  br i1 %243, label %249, label %244

244:                                              ; preds = %241
  %245 = invoke noundef i1 %242(ptr noundef nonnull align 8 dereferenceable(32) %15, ptr noundef nonnull align 8 dereferenceable(32) %15, i32 noundef 3)
          to label %249 unwind label %246

246:                                              ; preds = %244
  %247 = landingpad { ptr, i32 }
          catch ptr null
  %248 = extractvalue { ptr, i32 } %247, 0
  call void @__clang_call_terminate(ptr %248) #21
  unreachable

249:                                              ; preds = %241, %244
  %250 = load ptr, ptr %237, align 8, !tbaa !20
  %251 = icmp eq ptr %250, null
  br i1 %251, label %257, label %252

252:                                              ; preds = %249
  %253 = invoke noundef i1 %250(ptr noundef nonnull align 8 dereferenceable(32) %14, ptr noundef nonnull align 8 dereferenceable(32) %14, i32 noundef 3)
          to label %257 unwind label %254

254:                                              ; preds = %252
  %255 = landingpad { ptr, i32 }
          catch ptr null
  %256 = extractvalue { ptr, i32 } %255, 0
  call void @__clang_call_terminate(ptr %256) #21
  unreachable

257:                                              ; preds = %249, %252
  %258 = getelementptr inbounds nuw i8, ptr %16, i64 16
  %259 = getelementptr inbounds nuw i8, ptr %16, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %16, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE3$_4E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %259, align 8, !tbaa !21
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE3$_4E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %258, align 8, !tbaa !20
  %260 = getelementptr inbounds nuw i8, ptr %17, i64 16
  %261 = getelementptr inbounds nuw i8, ptr %17, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %17, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE3$_5E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %261, align 8, !tbaa !21
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE3$_5E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %260, align 8, !tbaa !20
  invoke fastcc void @_ZL37checkOverlappingMemoryOneRuntimeCheckIjEvSt8functionIFvPT_S2_jEES4_PKc(ptr noundef %16, ptr noundef %17, ptr noundef nonnull @.str.7)
          to label %262 unwind label %1238

262:                                              ; preds = %257
  %263 = load ptr, ptr %260, align 8, !tbaa !20
  %264 = icmp eq ptr %263, null
  br i1 %264, label %270, label %265

265:                                              ; preds = %262
  %266 = invoke noundef i1 %263(ptr noundef nonnull align 8 dereferenceable(32) %17, ptr noundef nonnull align 8 dereferenceable(32) %17, i32 noundef 3)
          to label %270 unwind label %267

267:                                              ; preds = %265
  %268 = landingpad { ptr, i32 }
          catch ptr null
  %269 = extractvalue { ptr, i32 } %268, 0
  call void @__clang_call_terminate(ptr %269) #21
  unreachable

270:                                              ; preds = %262, %265
  %271 = load ptr, ptr %258, align 8, !tbaa !20
  %272 = icmp eq ptr %271, null
  br i1 %272, label %278, label %273

273:                                              ; preds = %270
  %274 = invoke noundef i1 %271(ptr noundef nonnull align 8 dereferenceable(32) %16, ptr noundef nonnull align 8 dereferenceable(32) %16, i32 noundef 3)
          to label %278 unwind label %275

275:                                              ; preds = %273
  %276 = landingpad { ptr, i32 }
          catch ptr null
  %277 = extractvalue { ptr, i32 } %276, 0
  call void @__clang_call_terminate(ptr %277) #21
  unreachable

278:                                              ; preds = %270, %273
  %279 = getelementptr inbounds nuw i8, ptr %18, i64 16
  %280 = getelementptr inbounds nuw i8, ptr %18, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %18, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPmS0_jEZ4mainE3$_4E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %280, align 8, !tbaa !23
  store ptr @"_ZNSt17_Function_handlerIFvPmS0_jEZ4mainE3$_4E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %279, align 8, !tbaa !20
  %281 = getelementptr inbounds nuw i8, ptr %19, i64 16
  %282 = getelementptr inbounds nuw i8, ptr %19, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %19, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPmS0_jEZ4mainE3$_5E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %282, align 8, !tbaa !23
  store ptr @"_ZNSt17_Function_handlerIFvPmS0_jEZ4mainE3$_5E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %281, align 8, !tbaa !20
  invoke fastcc void @_ZL37checkOverlappingMemoryOneRuntimeCheckImEvSt8functionIFvPT_S2_jEES4_PKc(ptr noundef %18, ptr noundef %19, ptr noundef nonnull @.str.8)
          to label %283 unwind label %1255

283:                                              ; preds = %278
  %284 = load ptr, ptr %281, align 8, !tbaa !20
  %285 = icmp eq ptr %284, null
  br i1 %285, label %291, label %286

286:                                              ; preds = %283
  %287 = invoke noundef i1 %284(ptr noundef nonnull align 8 dereferenceable(32) %19, ptr noundef nonnull align 8 dereferenceable(32) %19, i32 noundef 3)
          to label %291 unwind label %288

288:                                              ; preds = %286
  %289 = landingpad { ptr, i32 }
          catch ptr null
  %290 = extractvalue { ptr, i32 } %289, 0
  call void @__clang_call_terminate(ptr %290) #21
  unreachable

291:                                              ; preds = %283, %286
  %292 = load ptr, ptr %279, align 8, !tbaa !20
  %293 = icmp eq ptr %292, null
  br i1 %293, label %299, label %294

294:                                              ; preds = %291
  %295 = invoke noundef i1 %292(ptr noundef nonnull align 8 dereferenceable(32) %18, ptr noundef nonnull align 8 dereferenceable(32) %18, i32 noundef 3)
          to label %299 unwind label %296

296:                                              ; preds = %294
  %297 = landingpad { ptr, i32 }
          catch ptr null
  %298 = extractvalue { ptr, i32 } %297, 0
  call void @__clang_call_terminate(ptr %298) #21
  unreachable

299:                                              ; preds = %291, %294
  %300 = getelementptr inbounds nuw i8, ptr %20, i64 16
  %301 = getelementptr inbounds nuw i8, ptr %20, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %20, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPhS0_jEZ4mainE3$_6E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %301, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFvPhS0_jEZ4mainE3$_6E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %300, align 8, !tbaa !20
  %302 = getelementptr inbounds nuw i8, ptr %21, i64 16
  %303 = getelementptr inbounds nuw i8, ptr %21, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %21, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPhS0_jEZ4mainE3$_7E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %303, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFvPhS0_jEZ4mainE3$_7E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %302, align 8, !tbaa !20
  invoke fastcc void @_ZL37checkOverlappingMemoryOneRuntimeCheckIhEvSt8functionIFvPT_S2_jEES4_PKc(ptr noundef %20, ptr noundef %21, ptr noundef nonnull @.str.9)
          to label %304 unwind label %1272

304:                                              ; preds = %299
  %305 = load ptr, ptr %302, align 8, !tbaa !20
  %306 = icmp eq ptr %305, null
  br i1 %306, label %312, label %307

307:                                              ; preds = %304
  %308 = invoke noundef i1 %305(ptr noundef nonnull align 8 dereferenceable(32) %21, ptr noundef nonnull align 8 dereferenceable(32) %21, i32 noundef 3)
          to label %312 unwind label %309

309:                                              ; preds = %307
  %310 = landingpad { ptr, i32 }
          catch ptr null
  %311 = extractvalue { ptr, i32 } %310, 0
  call void @__clang_call_terminate(ptr %311) #21
  unreachable

312:                                              ; preds = %304, %307
  %313 = load ptr, ptr %300, align 8, !tbaa !20
  %314 = icmp eq ptr %313, null
  br i1 %314, label %320, label %315

315:                                              ; preds = %312
  %316 = invoke noundef i1 %313(ptr noundef nonnull align 8 dereferenceable(32) %20, ptr noundef nonnull align 8 dereferenceable(32) %20, i32 noundef 3)
          to label %320 unwind label %317

317:                                              ; preds = %315
  %318 = landingpad { ptr, i32 }
          catch ptr null
  %319 = extractvalue { ptr, i32 } %318, 0
  call void @__clang_call_terminate(ptr %319) #21
  unreachable

320:                                              ; preds = %312, %315
  %321 = getelementptr inbounds nuw i8, ptr %22, i64 16
  %322 = getelementptr inbounds nuw i8, ptr %22, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %22, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE3$_6E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %322, align 8, !tbaa !21
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE3$_6E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %321, align 8, !tbaa !20
  %323 = getelementptr inbounds nuw i8, ptr %23, i64 16
  %324 = getelementptr inbounds nuw i8, ptr %23, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %23, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE3$_7E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %324, align 8, !tbaa !21
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE3$_7E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %323, align 8, !tbaa !20
  invoke fastcc void @_ZL37checkOverlappingMemoryOneRuntimeCheckIjEvSt8functionIFvPT_S2_jEES4_PKc(ptr noundef %22, ptr noundef %23, ptr noundef nonnull @.str.10)
          to label %325 unwind label %1289

325:                                              ; preds = %320
  %326 = load ptr, ptr %323, align 8, !tbaa !20
  %327 = icmp eq ptr %326, null
  br i1 %327, label %333, label %328

328:                                              ; preds = %325
  %329 = invoke noundef i1 %326(ptr noundef nonnull align 8 dereferenceable(32) %23, ptr noundef nonnull align 8 dereferenceable(32) %23, i32 noundef 3)
          to label %333 unwind label %330

330:                                              ; preds = %328
  %331 = landingpad { ptr, i32 }
          catch ptr null
  %332 = extractvalue { ptr, i32 } %331, 0
  call void @__clang_call_terminate(ptr %332) #21
  unreachable

333:                                              ; preds = %325, %328
  %334 = load ptr, ptr %321, align 8, !tbaa !20
  %335 = icmp eq ptr %334, null
  br i1 %335, label %341, label %336

336:                                              ; preds = %333
  %337 = invoke noundef i1 %334(ptr noundef nonnull align 8 dereferenceable(32) %22, ptr noundef nonnull align 8 dereferenceable(32) %22, i32 noundef 3)
          to label %341 unwind label %338

338:                                              ; preds = %336
  %339 = landingpad { ptr, i32 }
          catch ptr null
  %340 = extractvalue { ptr, i32 } %339, 0
  call void @__clang_call_terminate(ptr %340) #21
  unreachable

341:                                              ; preds = %333, %336
  %342 = getelementptr inbounds nuw i8, ptr %24, i64 16
  %343 = getelementptr inbounds nuw i8, ptr %24, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %24, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPmS0_jEZ4mainE3$_6E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %343, align 8, !tbaa !23
  store ptr @"_ZNSt17_Function_handlerIFvPmS0_jEZ4mainE3$_6E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %342, align 8, !tbaa !20
  %344 = getelementptr inbounds nuw i8, ptr %25, i64 16
  %345 = getelementptr inbounds nuw i8, ptr %25, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %25, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPmS0_jEZ4mainE3$_7E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %345, align 8, !tbaa !23
  store ptr @"_ZNSt17_Function_handlerIFvPmS0_jEZ4mainE3$_7E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %344, align 8, !tbaa !20
  invoke fastcc void @_ZL37checkOverlappingMemoryOneRuntimeCheckImEvSt8functionIFvPT_S2_jEES4_PKc(ptr noundef %24, ptr noundef %25, ptr noundef nonnull @.str.11)
          to label %346 unwind label %1306

346:                                              ; preds = %341
  %347 = load ptr, ptr %344, align 8, !tbaa !20
  %348 = icmp eq ptr %347, null
  br i1 %348, label %354, label %349

349:                                              ; preds = %346
  %350 = invoke noundef i1 %347(ptr noundef nonnull align 8 dereferenceable(32) %25, ptr noundef nonnull align 8 dereferenceable(32) %25, i32 noundef 3)
          to label %354 unwind label %351

351:                                              ; preds = %349
  %352 = landingpad { ptr, i32 }
          catch ptr null
  %353 = extractvalue { ptr, i32 } %352, 0
  call void @__clang_call_terminate(ptr %353) #21
  unreachable

354:                                              ; preds = %346, %349
  %355 = load ptr, ptr %342, align 8, !tbaa !20
  %356 = icmp eq ptr %355, null
  br i1 %356, label %362, label %357

357:                                              ; preds = %354
  %358 = invoke noundef i1 %355(ptr noundef nonnull align 8 dereferenceable(32) %24, ptr noundef nonnull align 8 dereferenceable(32) %24, i32 noundef 3)
          to label %362 unwind label %359

359:                                              ; preds = %357
  %360 = landingpad { ptr, i32 }
          catch ptr null
  %361 = extractvalue { ptr, i32 } %360, 0
  call void @__clang_call_terminate(ptr %361) #21
  unreachable

362:                                              ; preds = %354, %357
  %363 = getelementptr inbounds nuw i8, ptr %26, i64 16
  %364 = getelementptr inbounds nuw i8, ptr %26, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %26, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPhS0_jEZ4mainE3$_8E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %364, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFvPhS0_jEZ4mainE3$_8E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %363, align 8, !tbaa !20
  %365 = getelementptr inbounds nuw i8, ptr %27, i64 16
  %366 = getelementptr inbounds nuw i8, ptr %27, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %27, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPhS0_jEZ4mainE3$_9E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %366, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFvPhS0_jEZ4mainE3$_9E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %365, align 8, !tbaa !20
  invoke fastcc void @_ZL37checkOverlappingMemoryOneRuntimeCheckIhEvSt8functionIFvPT_S2_jEES4_PKc(ptr noundef %26, ptr noundef %27, ptr noundef nonnull @.str.12)
          to label %367 unwind label %1323

367:                                              ; preds = %362
  %368 = load ptr, ptr %365, align 8, !tbaa !20
  %369 = icmp eq ptr %368, null
  br i1 %369, label %375, label %370

370:                                              ; preds = %367
  %371 = invoke noundef i1 %368(ptr noundef nonnull align 8 dereferenceable(32) %27, ptr noundef nonnull align 8 dereferenceable(32) %27, i32 noundef 3)
          to label %375 unwind label %372

372:                                              ; preds = %370
  %373 = landingpad { ptr, i32 }
          catch ptr null
  %374 = extractvalue { ptr, i32 } %373, 0
  call void @__clang_call_terminate(ptr %374) #21
  unreachable

375:                                              ; preds = %367, %370
  %376 = load ptr, ptr %363, align 8, !tbaa !20
  %377 = icmp eq ptr %376, null
  br i1 %377, label %383, label %378

378:                                              ; preds = %375
  %379 = invoke noundef i1 %376(ptr noundef nonnull align 8 dereferenceable(32) %26, ptr noundef nonnull align 8 dereferenceable(32) %26, i32 noundef 3)
          to label %383 unwind label %380

380:                                              ; preds = %378
  %381 = landingpad { ptr, i32 }
          catch ptr null
  %382 = extractvalue { ptr, i32 } %381, 0
  call void @__clang_call_terminate(ptr %382) #21
  unreachable

383:                                              ; preds = %375, %378
  %384 = getelementptr inbounds nuw i8, ptr %28, i64 16
  %385 = getelementptr inbounds nuw i8, ptr %28, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %28, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE3$_8E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %385, align 8, !tbaa !21
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE3$_8E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %384, align 8, !tbaa !20
  %386 = getelementptr inbounds nuw i8, ptr %29, i64 16
  %387 = getelementptr inbounds nuw i8, ptr %29, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %29, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE3$_9E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %387, align 8, !tbaa !21
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE3$_9E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %386, align 8, !tbaa !20
  invoke fastcc void @_ZL37checkOverlappingMemoryOneRuntimeCheckIjEvSt8functionIFvPT_S2_jEES4_PKc(ptr noundef %28, ptr noundef %29, ptr noundef nonnull @.str.13)
          to label %388 unwind label %1340

388:                                              ; preds = %383
  %389 = load ptr, ptr %386, align 8, !tbaa !20
  %390 = icmp eq ptr %389, null
  br i1 %390, label %396, label %391

391:                                              ; preds = %388
  %392 = invoke noundef i1 %389(ptr noundef nonnull align 8 dereferenceable(32) %29, ptr noundef nonnull align 8 dereferenceable(32) %29, i32 noundef 3)
          to label %396 unwind label %393

393:                                              ; preds = %391
  %394 = landingpad { ptr, i32 }
          catch ptr null
  %395 = extractvalue { ptr, i32 } %394, 0
  call void @__clang_call_terminate(ptr %395) #21
  unreachable

396:                                              ; preds = %388, %391
  %397 = load ptr, ptr %384, align 8, !tbaa !20
  %398 = icmp eq ptr %397, null
  br i1 %398, label %404, label %399

399:                                              ; preds = %396
  %400 = invoke noundef i1 %397(ptr noundef nonnull align 8 dereferenceable(32) %28, ptr noundef nonnull align 8 dereferenceable(32) %28, i32 noundef 3)
          to label %404 unwind label %401

401:                                              ; preds = %399
  %402 = landingpad { ptr, i32 }
          catch ptr null
  %403 = extractvalue { ptr, i32 } %402, 0
  call void @__clang_call_terminate(ptr %403) #21
  unreachable

404:                                              ; preds = %396, %399
  %405 = getelementptr inbounds nuw i8, ptr %30, i64 16
  %406 = getelementptr inbounds nuw i8, ptr %30, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %30, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPmS0_jEZ4mainE3$_8E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %406, align 8, !tbaa !23
  store ptr @"_ZNSt17_Function_handlerIFvPmS0_jEZ4mainE3$_8E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %405, align 8, !tbaa !20
  %407 = getelementptr inbounds nuw i8, ptr %31, i64 16
  %408 = getelementptr inbounds nuw i8, ptr %31, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %31, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPmS0_jEZ4mainE3$_9E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %408, align 8, !tbaa !23
  store ptr @"_ZNSt17_Function_handlerIFvPmS0_jEZ4mainE3$_9E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %407, align 8, !tbaa !20
  invoke fastcc void @_ZL37checkOverlappingMemoryOneRuntimeCheckImEvSt8functionIFvPT_S2_jEES4_PKc(ptr noundef %30, ptr noundef %31, ptr noundef nonnull @.str.14)
          to label %409 unwind label %1357

409:                                              ; preds = %404
  %410 = load ptr, ptr %407, align 8, !tbaa !20
  %411 = icmp eq ptr %410, null
  br i1 %411, label %417, label %412

412:                                              ; preds = %409
  %413 = invoke noundef i1 %410(ptr noundef nonnull align 8 dereferenceable(32) %31, ptr noundef nonnull align 8 dereferenceable(32) %31, i32 noundef 3)
          to label %417 unwind label %414

414:                                              ; preds = %412
  %415 = landingpad { ptr, i32 }
          catch ptr null
  %416 = extractvalue { ptr, i32 } %415, 0
  call void @__clang_call_terminate(ptr %416) #21
  unreachable

417:                                              ; preds = %409, %412
  %418 = load ptr, ptr %405, align 8, !tbaa !20
  %419 = icmp eq ptr %418, null
  br i1 %419, label %425, label %420

420:                                              ; preds = %417
  %421 = invoke noundef i1 %418(ptr noundef nonnull align 8 dereferenceable(32) %30, ptr noundef nonnull align 8 dereferenceable(32) %30, i32 noundef 3)
          to label %425 unwind label %422

422:                                              ; preds = %420
  %423 = landingpad { ptr, i32 }
          catch ptr null
  %424 = extractvalue { ptr, i32 } %423, 0
  call void @__clang_call_terminate(ptr %424) #21
  unreachable

425:                                              ; preds = %417, %420
  %426 = getelementptr inbounds nuw i8, ptr %32, i64 16
  %427 = getelementptr inbounds nuw i8, ptr %32, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %32, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPhS0_jEZ4mainE4$_10E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %427, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFvPhS0_jEZ4mainE4$_10E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %426, align 8, !tbaa !20
  %428 = getelementptr inbounds nuw i8, ptr %33, i64 16
  %429 = getelementptr inbounds nuw i8, ptr %33, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %33, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPhS0_jEZ4mainE4$_11E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %429, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFvPhS0_jEZ4mainE4$_11E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %428, align 8, !tbaa !20
  invoke fastcc void @_ZL37checkOverlappingMemoryOneRuntimeCheckIhEvSt8functionIFvPT_S2_jEES4_PKc(ptr noundef %32, ptr noundef %33, ptr noundef nonnull @.str.15)
          to label %430 unwind label %1374

430:                                              ; preds = %425
  %431 = load ptr, ptr %428, align 8, !tbaa !20
  %432 = icmp eq ptr %431, null
  br i1 %432, label %438, label %433

433:                                              ; preds = %430
  %434 = invoke noundef i1 %431(ptr noundef nonnull align 8 dereferenceable(32) %33, ptr noundef nonnull align 8 dereferenceable(32) %33, i32 noundef 3)
          to label %438 unwind label %435

435:                                              ; preds = %433
  %436 = landingpad { ptr, i32 }
          catch ptr null
  %437 = extractvalue { ptr, i32 } %436, 0
  call void @__clang_call_terminate(ptr %437) #21
  unreachable

438:                                              ; preds = %430, %433
  %439 = load ptr, ptr %426, align 8, !tbaa !20
  %440 = icmp eq ptr %439, null
  br i1 %440, label %446, label %441

441:                                              ; preds = %438
  %442 = invoke noundef i1 %439(ptr noundef nonnull align 8 dereferenceable(32) %32, ptr noundef nonnull align 8 dereferenceable(32) %32, i32 noundef 3)
          to label %446 unwind label %443

443:                                              ; preds = %441
  %444 = landingpad { ptr, i32 }
          catch ptr null
  %445 = extractvalue { ptr, i32 } %444, 0
  call void @__clang_call_terminate(ptr %445) #21
  unreachable

446:                                              ; preds = %438, %441
  %447 = getelementptr inbounds nuw i8, ptr %34, i64 16
  %448 = getelementptr inbounds nuw i8, ptr %34, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %34, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE4$_10E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %448, align 8, !tbaa !21
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE4$_10E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %447, align 8, !tbaa !20
  %449 = getelementptr inbounds nuw i8, ptr %35, i64 16
  %450 = getelementptr inbounds nuw i8, ptr %35, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %35, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE4$_11E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %450, align 8, !tbaa !21
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE4$_11E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %449, align 8, !tbaa !20
  invoke fastcc void @_ZL37checkOverlappingMemoryOneRuntimeCheckIjEvSt8functionIFvPT_S2_jEES4_PKc(ptr noundef %34, ptr noundef %35, ptr noundef nonnull @.str.16)
          to label %451 unwind label %1391

451:                                              ; preds = %446
  %452 = load ptr, ptr %449, align 8, !tbaa !20
  %453 = icmp eq ptr %452, null
  br i1 %453, label %459, label %454

454:                                              ; preds = %451
  %455 = invoke noundef i1 %452(ptr noundef nonnull align 8 dereferenceable(32) %35, ptr noundef nonnull align 8 dereferenceable(32) %35, i32 noundef 3)
          to label %459 unwind label %456

456:                                              ; preds = %454
  %457 = landingpad { ptr, i32 }
          catch ptr null
  %458 = extractvalue { ptr, i32 } %457, 0
  call void @__clang_call_terminate(ptr %458) #21
  unreachable

459:                                              ; preds = %451, %454
  %460 = load ptr, ptr %447, align 8, !tbaa !20
  %461 = icmp eq ptr %460, null
  br i1 %461, label %467, label %462

462:                                              ; preds = %459
  %463 = invoke noundef i1 %460(ptr noundef nonnull align 8 dereferenceable(32) %34, ptr noundef nonnull align 8 dereferenceable(32) %34, i32 noundef 3)
          to label %467 unwind label %464

464:                                              ; preds = %462
  %465 = landingpad { ptr, i32 }
          catch ptr null
  %466 = extractvalue { ptr, i32 } %465, 0
  call void @__clang_call_terminate(ptr %466) #21
  unreachable

467:                                              ; preds = %459, %462
  %468 = getelementptr inbounds nuw i8, ptr %36, i64 16
  %469 = getelementptr inbounds nuw i8, ptr %36, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %36, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPmS0_jEZ4mainE4$_10E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %469, align 8, !tbaa !23
  store ptr @"_ZNSt17_Function_handlerIFvPmS0_jEZ4mainE4$_10E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %468, align 8, !tbaa !20
  %470 = getelementptr inbounds nuw i8, ptr %37, i64 16
  %471 = getelementptr inbounds nuw i8, ptr %37, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %37, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPmS0_jEZ4mainE4$_11E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %471, align 8, !tbaa !23
  store ptr @"_ZNSt17_Function_handlerIFvPmS0_jEZ4mainE4$_11E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %470, align 8, !tbaa !20
  invoke fastcc void @_ZL37checkOverlappingMemoryOneRuntimeCheckImEvSt8functionIFvPT_S2_jEES4_PKc(ptr noundef %36, ptr noundef %37, ptr noundef nonnull @.str.17)
          to label %472 unwind label %1408

472:                                              ; preds = %467
  %473 = load ptr, ptr %470, align 8, !tbaa !20
  %474 = icmp eq ptr %473, null
  br i1 %474, label %480, label %475

475:                                              ; preds = %472
  %476 = invoke noundef i1 %473(ptr noundef nonnull align 8 dereferenceable(32) %37, ptr noundef nonnull align 8 dereferenceable(32) %37, i32 noundef 3)
          to label %480 unwind label %477

477:                                              ; preds = %475
  %478 = landingpad { ptr, i32 }
          catch ptr null
  %479 = extractvalue { ptr, i32 } %478, 0
  call void @__clang_call_terminate(ptr %479) #21
  unreachable

480:                                              ; preds = %472, %475
  %481 = load ptr, ptr %468, align 8, !tbaa !20
  %482 = icmp eq ptr %481, null
  br i1 %482, label %488, label %483

483:                                              ; preds = %480
  %484 = invoke noundef i1 %481(ptr noundef nonnull align 8 dereferenceable(32) %36, ptr noundef nonnull align 8 dereferenceable(32) %36, i32 noundef 3)
          to label %488 unwind label %485

485:                                              ; preds = %483
  %486 = landingpad { ptr, i32 }
          catch ptr null
  %487 = extractvalue { ptr, i32 } %486, 0
  call void @__clang_call_terminate(ptr %487) #21
  unreachable

488:                                              ; preds = %480, %483
  %489 = getelementptr inbounds nuw i8, ptr %38, i64 16
  %490 = getelementptr inbounds nuw i8, ptr %38, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %38, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPhS0_jEZ4mainE4$_12E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %490, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFvPhS0_jEZ4mainE4$_12E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %489, align 8, !tbaa !20
  %491 = getelementptr inbounds nuw i8, ptr %39, i64 16
  %492 = getelementptr inbounds nuw i8, ptr %39, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %39, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPhS0_jEZ4mainE4$_13E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %492, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFvPhS0_jEZ4mainE4$_13E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %491, align 8, !tbaa !20
  invoke fastcc void @_ZL37checkOverlappingMemoryOneRuntimeCheckIhEvSt8functionIFvPT_S2_jEES4_PKc(ptr noundef %38, ptr noundef %39, ptr noundef nonnull @.str.18)
          to label %493 unwind label %1425

493:                                              ; preds = %488
  %494 = load ptr, ptr %491, align 8, !tbaa !20
  %495 = icmp eq ptr %494, null
  br i1 %495, label %501, label %496

496:                                              ; preds = %493
  %497 = invoke noundef i1 %494(ptr noundef nonnull align 8 dereferenceable(32) %39, ptr noundef nonnull align 8 dereferenceable(32) %39, i32 noundef 3)
          to label %501 unwind label %498

498:                                              ; preds = %496
  %499 = landingpad { ptr, i32 }
          catch ptr null
  %500 = extractvalue { ptr, i32 } %499, 0
  call void @__clang_call_terminate(ptr %500) #21
  unreachable

501:                                              ; preds = %493, %496
  %502 = load ptr, ptr %489, align 8, !tbaa !20
  %503 = icmp eq ptr %502, null
  br i1 %503, label %509, label %504

504:                                              ; preds = %501
  %505 = invoke noundef i1 %502(ptr noundef nonnull align 8 dereferenceable(32) %38, ptr noundef nonnull align 8 dereferenceable(32) %38, i32 noundef 3)
          to label %509 unwind label %506

506:                                              ; preds = %504
  %507 = landingpad { ptr, i32 }
          catch ptr null
  %508 = extractvalue { ptr, i32 } %507, 0
  call void @__clang_call_terminate(ptr %508) #21
  unreachable

509:                                              ; preds = %501, %504
  %510 = getelementptr inbounds nuw i8, ptr %40, i64 16
  %511 = getelementptr inbounds nuw i8, ptr %40, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %40, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE4$_12E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %511, align 8, !tbaa !21
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE4$_12E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %510, align 8, !tbaa !20
  %512 = getelementptr inbounds nuw i8, ptr %41, i64 16
  %513 = getelementptr inbounds nuw i8, ptr %41, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %41, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE4$_13E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %513, align 8, !tbaa !21
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE4$_13E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %512, align 8, !tbaa !20
  invoke fastcc void @_ZL37checkOverlappingMemoryOneRuntimeCheckIjEvSt8functionIFvPT_S2_jEES4_PKc(ptr noundef %40, ptr noundef %41, ptr noundef nonnull @.str.19)
          to label %514 unwind label %1442

514:                                              ; preds = %509
  %515 = load ptr, ptr %512, align 8, !tbaa !20
  %516 = icmp eq ptr %515, null
  br i1 %516, label %522, label %517

517:                                              ; preds = %514
  %518 = invoke noundef i1 %515(ptr noundef nonnull align 8 dereferenceable(32) %41, ptr noundef nonnull align 8 dereferenceable(32) %41, i32 noundef 3)
          to label %522 unwind label %519

519:                                              ; preds = %517
  %520 = landingpad { ptr, i32 }
          catch ptr null
  %521 = extractvalue { ptr, i32 } %520, 0
  call void @__clang_call_terminate(ptr %521) #21
  unreachable

522:                                              ; preds = %514, %517
  %523 = load ptr, ptr %510, align 8, !tbaa !20
  %524 = icmp eq ptr %523, null
  br i1 %524, label %530, label %525

525:                                              ; preds = %522
  %526 = invoke noundef i1 %523(ptr noundef nonnull align 8 dereferenceable(32) %40, ptr noundef nonnull align 8 dereferenceable(32) %40, i32 noundef 3)
          to label %530 unwind label %527

527:                                              ; preds = %525
  %528 = landingpad { ptr, i32 }
          catch ptr null
  %529 = extractvalue { ptr, i32 } %528, 0
  call void @__clang_call_terminate(ptr %529) #21
  unreachable

530:                                              ; preds = %522, %525
  %531 = getelementptr inbounds nuw i8, ptr %42, i64 16
  %532 = getelementptr inbounds nuw i8, ptr %42, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %42, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPmS0_jEZ4mainE4$_12E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %532, align 8, !tbaa !23
  store ptr @"_ZNSt17_Function_handlerIFvPmS0_jEZ4mainE4$_12E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %531, align 8, !tbaa !20
  %533 = getelementptr inbounds nuw i8, ptr %43, i64 16
  %534 = getelementptr inbounds nuw i8, ptr %43, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %43, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPmS0_jEZ4mainE4$_13E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %534, align 8, !tbaa !23
  store ptr @"_ZNSt17_Function_handlerIFvPmS0_jEZ4mainE4$_13E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %533, align 8, !tbaa !20
  invoke fastcc void @_ZL37checkOverlappingMemoryOneRuntimeCheckImEvSt8functionIFvPT_S2_jEES4_PKc(ptr noundef %42, ptr noundef %43, ptr noundef nonnull @.str.20)
          to label %535 unwind label %1459

535:                                              ; preds = %530
  %536 = load ptr, ptr %533, align 8, !tbaa !20
  %537 = icmp eq ptr %536, null
  br i1 %537, label %543, label %538

538:                                              ; preds = %535
  %539 = invoke noundef i1 %536(ptr noundef nonnull align 8 dereferenceable(32) %43, ptr noundef nonnull align 8 dereferenceable(32) %43, i32 noundef 3)
          to label %543 unwind label %540

540:                                              ; preds = %538
  %541 = landingpad { ptr, i32 }
          catch ptr null
  %542 = extractvalue { ptr, i32 } %541, 0
  call void @__clang_call_terminate(ptr %542) #21
  unreachable

543:                                              ; preds = %535, %538
  %544 = load ptr, ptr %531, align 8, !tbaa !20
  %545 = icmp eq ptr %544, null
  br i1 %545, label %551, label %546

546:                                              ; preds = %543
  %547 = invoke noundef i1 %544(ptr noundef nonnull align 8 dereferenceable(32) %42, ptr noundef nonnull align 8 dereferenceable(32) %42, i32 noundef 3)
          to label %551 unwind label %548

548:                                              ; preds = %546
  %549 = landingpad { ptr, i32 }
          catch ptr null
  %550 = extractvalue { ptr, i32 } %549, 0
  call void @__clang_call_terminate(ptr %550) #21
  unreachable

551:                                              ; preds = %543, %546
  %552 = getelementptr inbounds nuw i8, ptr %44, i64 16
  %553 = getelementptr inbounds nuw i8, ptr %44, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %44, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPhS0_jEZ4mainE4$_14E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %553, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFvPhS0_jEZ4mainE4$_14E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %552, align 8, !tbaa !20
  %554 = getelementptr inbounds nuw i8, ptr %45, i64 16
  %555 = getelementptr inbounds nuw i8, ptr %45, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %45, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPhS0_jEZ4mainE4$_15E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %555, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFvPhS0_jEZ4mainE4$_15E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %554, align 8, !tbaa !20
  invoke fastcc void @_ZL37checkOverlappingMemoryOneRuntimeCheckIhEvSt8functionIFvPT_S2_jEES4_PKc(ptr noundef %44, ptr noundef %45, ptr noundef nonnull @.str.21)
          to label %556 unwind label %1476

556:                                              ; preds = %551
  %557 = load ptr, ptr %554, align 8, !tbaa !20
  %558 = icmp eq ptr %557, null
  br i1 %558, label %564, label %559

559:                                              ; preds = %556
  %560 = invoke noundef i1 %557(ptr noundef nonnull align 8 dereferenceable(32) %45, ptr noundef nonnull align 8 dereferenceable(32) %45, i32 noundef 3)
          to label %564 unwind label %561

561:                                              ; preds = %559
  %562 = landingpad { ptr, i32 }
          catch ptr null
  %563 = extractvalue { ptr, i32 } %562, 0
  call void @__clang_call_terminate(ptr %563) #21
  unreachable

564:                                              ; preds = %556, %559
  %565 = load ptr, ptr %552, align 8, !tbaa !20
  %566 = icmp eq ptr %565, null
  br i1 %566, label %572, label %567

567:                                              ; preds = %564
  %568 = invoke noundef i1 %565(ptr noundef nonnull align 8 dereferenceable(32) %44, ptr noundef nonnull align 8 dereferenceable(32) %44, i32 noundef 3)
          to label %572 unwind label %569

569:                                              ; preds = %567
  %570 = landingpad { ptr, i32 }
          catch ptr null
  %571 = extractvalue { ptr, i32 } %570, 0
  call void @__clang_call_terminate(ptr %571) #21
  unreachable

572:                                              ; preds = %564, %567
  %573 = getelementptr inbounds nuw i8, ptr %46, i64 16
  %574 = getelementptr inbounds nuw i8, ptr %46, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %46, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE4$_14E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %574, align 8, !tbaa !21
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE4$_14E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %573, align 8, !tbaa !20
  %575 = getelementptr inbounds nuw i8, ptr %47, i64 16
  %576 = getelementptr inbounds nuw i8, ptr %47, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %47, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE4$_15E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %576, align 8, !tbaa !21
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE4$_15E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %575, align 8, !tbaa !20
  invoke fastcc void @_ZL37checkOverlappingMemoryOneRuntimeCheckIjEvSt8functionIFvPT_S2_jEES4_PKc(ptr noundef %46, ptr noundef %47, ptr noundef nonnull @.str.22)
          to label %577 unwind label %1493

577:                                              ; preds = %572
  %578 = load ptr, ptr %575, align 8, !tbaa !20
  %579 = icmp eq ptr %578, null
  br i1 %579, label %585, label %580

580:                                              ; preds = %577
  %581 = invoke noundef i1 %578(ptr noundef nonnull align 8 dereferenceable(32) %47, ptr noundef nonnull align 8 dereferenceable(32) %47, i32 noundef 3)
          to label %585 unwind label %582

582:                                              ; preds = %580
  %583 = landingpad { ptr, i32 }
          catch ptr null
  %584 = extractvalue { ptr, i32 } %583, 0
  call void @__clang_call_terminate(ptr %584) #21
  unreachable

585:                                              ; preds = %577, %580
  %586 = load ptr, ptr %573, align 8, !tbaa !20
  %587 = icmp eq ptr %586, null
  br i1 %587, label %593, label %588

588:                                              ; preds = %585
  %589 = invoke noundef i1 %586(ptr noundef nonnull align 8 dereferenceable(32) %46, ptr noundef nonnull align 8 dereferenceable(32) %46, i32 noundef 3)
          to label %593 unwind label %590

590:                                              ; preds = %588
  %591 = landingpad { ptr, i32 }
          catch ptr null
  %592 = extractvalue { ptr, i32 } %591, 0
  call void @__clang_call_terminate(ptr %592) #21
  unreachable

593:                                              ; preds = %585, %588
  %594 = getelementptr inbounds nuw i8, ptr %48, i64 16
  %595 = getelementptr inbounds nuw i8, ptr %48, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %48, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPmS0_jEZ4mainE4$_14E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %595, align 8, !tbaa !23
  store ptr @"_ZNSt17_Function_handlerIFvPmS0_jEZ4mainE4$_14E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %594, align 8, !tbaa !20
  %596 = getelementptr inbounds nuw i8, ptr %49, i64 16
  %597 = getelementptr inbounds nuw i8, ptr %49, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %49, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPmS0_jEZ4mainE4$_15E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %597, align 8, !tbaa !23
  store ptr @"_ZNSt17_Function_handlerIFvPmS0_jEZ4mainE4$_15E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %596, align 8, !tbaa !20
  invoke fastcc void @_ZL37checkOverlappingMemoryOneRuntimeCheckImEvSt8functionIFvPT_S2_jEES4_PKc(ptr noundef %48, ptr noundef %49, ptr noundef nonnull @.str.23)
          to label %598 unwind label %1510

598:                                              ; preds = %593
  %599 = load ptr, ptr %596, align 8, !tbaa !20
  %600 = icmp eq ptr %599, null
  br i1 %600, label %606, label %601

601:                                              ; preds = %598
  %602 = invoke noundef i1 %599(ptr noundef nonnull align 8 dereferenceable(32) %49, ptr noundef nonnull align 8 dereferenceable(32) %49, i32 noundef 3)
          to label %606 unwind label %603

603:                                              ; preds = %601
  %604 = landingpad { ptr, i32 }
          catch ptr null
  %605 = extractvalue { ptr, i32 } %604, 0
  call void @__clang_call_terminate(ptr %605) #21
  unreachable

606:                                              ; preds = %598, %601
  %607 = load ptr, ptr %594, align 8, !tbaa !20
  %608 = icmp eq ptr %607, null
  br i1 %608, label %614, label %609

609:                                              ; preds = %606
  %610 = invoke noundef i1 %607(ptr noundef nonnull align 8 dereferenceable(32) %48, ptr noundef nonnull align 8 dereferenceable(32) %48, i32 noundef 3)
          to label %614 unwind label %611

611:                                              ; preds = %609
  %612 = landingpad { ptr, i32 }
          catch ptr null
  %613 = extractvalue { ptr, i32 } %612, 0
  call void @__clang_call_terminate(ptr %613) #21
  unreachable

614:                                              ; preds = %606, %609
  %615 = getelementptr inbounds nuw i8, ptr %50, i64 16
  %616 = getelementptr inbounds nuw i8, ptr %50, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %50, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_S0_jEZ4mainE4$_16E9_M_invokeERKSt9_Any_dataOS0_S7_S7_Oj", ptr %616, align 8, !tbaa !25
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_S0_jEZ4mainE4$_16E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %615, align 8, !tbaa !20
  %617 = getelementptr inbounds nuw i8, ptr %51, i64 16
  %618 = getelementptr inbounds nuw i8, ptr %51, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %51, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_S0_jEZ4mainE4$_17E9_M_invokeERKSt9_Any_dataOS0_S7_S7_Oj", ptr %618, align 8, !tbaa !25
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_S0_jEZ4mainE4$_17E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %617, align 8, !tbaa !20
  invoke fastcc void @_ZL38checkOverlappingMemoryTwoRuntimeChecksIjEvSt8functionIFvPT_S2_S2_jEES4_PKc(ptr noundef %50, ptr noundef %51, ptr noundef nonnull @.str.24)
          to label %619 unwind label %1527

619:                                              ; preds = %614
  %620 = load ptr, ptr %617, align 8, !tbaa !20
  %621 = icmp eq ptr %620, null
  br i1 %621, label %627, label %622

622:                                              ; preds = %619
  %623 = invoke noundef i1 %620(ptr noundef nonnull align 8 dereferenceable(32) %51, ptr noundef nonnull align 8 dereferenceable(32) %51, i32 noundef 3)
          to label %627 unwind label %624

624:                                              ; preds = %622
  %625 = landingpad { ptr, i32 }
          catch ptr null
  %626 = extractvalue { ptr, i32 } %625, 0
  call void @__clang_call_terminate(ptr %626) #21
  unreachable

627:                                              ; preds = %619, %622
  %628 = load ptr, ptr %615, align 8, !tbaa !20
  %629 = icmp eq ptr %628, null
  br i1 %629, label %635, label %630

630:                                              ; preds = %627
  %631 = invoke noundef i1 %628(ptr noundef nonnull align 8 dereferenceable(32) %50, ptr noundef nonnull align 8 dereferenceable(32) %50, i32 noundef 3)
          to label %635 unwind label %632

632:                                              ; preds = %630
  %633 = landingpad { ptr, i32 }
          catch ptr null
  %634 = extractvalue { ptr, i32 } %633, 0
  call void @__clang_call_terminate(ptr %634) #21
  unreachable

635:                                              ; preds = %627, %630
  %636 = getelementptr inbounds nuw i8, ptr %52, i64 16
  %637 = getelementptr inbounds nuw i8, ptr %52, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %52, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPhS0_S0_jEZ4mainE4$_16E9_M_invokeERKSt9_Any_dataOS0_S7_S7_Oj", ptr %637, align 8, !tbaa !27
  store ptr @"_ZNSt17_Function_handlerIFvPhS0_S0_jEZ4mainE4$_16E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %636, align 8, !tbaa !20
  %638 = getelementptr inbounds nuw i8, ptr %53, i64 16
  %639 = getelementptr inbounds nuw i8, ptr %53, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %53, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPhS0_S0_jEZ4mainE4$_17E9_M_invokeERKSt9_Any_dataOS0_S7_S7_Oj", ptr %639, align 8, !tbaa !27
  store ptr @"_ZNSt17_Function_handlerIFvPhS0_S0_jEZ4mainE4$_17E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %638, align 8, !tbaa !20
  invoke fastcc void @_ZL38checkOverlappingMemoryTwoRuntimeChecksIhEvSt8functionIFvPT_S2_S2_jEES4_PKc(ptr noundef %52, ptr noundef %53, ptr noundef nonnull @.str.25)
          to label %640 unwind label %1544

640:                                              ; preds = %635
  %641 = load ptr, ptr %638, align 8, !tbaa !20
  %642 = icmp eq ptr %641, null
  br i1 %642, label %648, label %643

643:                                              ; preds = %640
  %644 = invoke noundef i1 %641(ptr noundef nonnull align 8 dereferenceable(32) %53, ptr noundef nonnull align 8 dereferenceable(32) %53, i32 noundef 3)
          to label %648 unwind label %645

645:                                              ; preds = %643
  %646 = landingpad { ptr, i32 }
          catch ptr null
  %647 = extractvalue { ptr, i32 } %646, 0
  call void @__clang_call_terminate(ptr %647) #21
  unreachable

648:                                              ; preds = %640, %643
  %649 = load ptr, ptr %636, align 8, !tbaa !20
  %650 = icmp eq ptr %649, null
  br i1 %650, label %656, label %651

651:                                              ; preds = %648
  %652 = invoke noundef i1 %649(ptr noundef nonnull align 8 dereferenceable(32) %52, ptr noundef nonnull align 8 dereferenceable(32) %52, i32 noundef 3)
          to label %656 unwind label %653

653:                                              ; preds = %651
  %654 = landingpad { ptr, i32 }
          catch ptr null
  %655 = extractvalue { ptr, i32 } %654, 0
  call void @__clang_call_terminate(ptr %655) #21
  unreachable

656:                                              ; preds = %648, %651
  %657 = getelementptr inbounds nuw i8, ptr %54, i64 16
  %658 = getelementptr inbounds nuw i8, ptr %54, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %54, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPmS0_S0_jEZ4mainE4$_16E9_M_invokeERKSt9_Any_dataOS0_S7_S7_Oj", ptr %658, align 8, !tbaa !29
  store ptr @"_ZNSt17_Function_handlerIFvPmS0_S0_jEZ4mainE4$_16E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %657, align 8, !tbaa !20
  %659 = getelementptr inbounds nuw i8, ptr %55, i64 16
  %660 = getelementptr inbounds nuw i8, ptr %55, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %55, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPmS0_S0_jEZ4mainE4$_17E9_M_invokeERKSt9_Any_dataOS0_S7_S7_Oj", ptr %660, align 8, !tbaa !29
  store ptr @"_ZNSt17_Function_handlerIFvPmS0_S0_jEZ4mainE4$_17E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %659, align 8, !tbaa !20
  invoke fastcc void @_ZL38checkOverlappingMemoryTwoRuntimeChecksImEvSt8functionIFvPT_S2_S2_jEES4_PKc(ptr noundef %54, ptr noundef %55, ptr noundef nonnull @.str.26)
          to label %661 unwind label %1561

661:                                              ; preds = %656
  %662 = load ptr, ptr %659, align 8, !tbaa !20
  %663 = icmp eq ptr %662, null
  br i1 %663, label %669, label %664

664:                                              ; preds = %661
  %665 = invoke noundef i1 %662(ptr noundef nonnull align 8 dereferenceable(32) %55, ptr noundef nonnull align 8 dereferenceable(32) %55, i32 noundef 3)
          to label %669 unwind label %666

666:                                              ; preds = %664
  %667 = landingpad { ptr, i32 }
          catch ptr null
  %668 = extractvalue { ptr, i32 } %667, 0
  call void @__clang_call_terminate(ptr %668) #21
  unreachable

669:                                              ; preds = %661, %664
  %670 = load ptr, ptr %657, align 8, !tbaa !20
  %671 = icmp eq ptr %670, null
  br i1 %671, label %677, label %672

672:                                              ; preds = %669
  %673 = invoke noundef i1 %670(ptr noundef nonnull align 8 dereferenceable(32) %54, ptr noundef nonnull align 8 dereferenceable(32) %54, i32 noundef 3)
          to label %677 unwind label %674

674:                                              ; preds = %672
  %675 = landingpad { ptr, i32 }
          catch ptr null
  %676 = extractvalue { ptr, i32 } %675, 0
  call void @__clang_call_terminate(ptr %676) #21
  unreachable

677:                                              ; preds = %669, %672
  %678 = getelementptr inbounds nuw i8, ptr %56, i64 16
  %679 = getelementptr inbounds nuw i8, ptr %56, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %56, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPhS0_S0_jEZ4mainE4$_18E9_M_invokeERKSt9_Any_dataOS0_S7_S7_Oj", ptr %679, align 8, !tbaa !27
  store ptr @"_ZNSt17_Function_handlerIFvPhS0_S0_jEZ4mainE4$_18E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %678, align 8, !tbaa !20
  %680 = getelementptr inbounds nuw i8, ptr %57, i64 16
  %681 = getelementptr inbounds nuw i8, ptr %57, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %57, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPhS0_S0_jEZ4mainE4$_19E9_M_invokeERKSt9_Any_dataOS0_S7_S7_Oj", ptr %681, align 8, !tbaa !27
  store ptr @"_ZNSt17_Function_handlerIFvPhS0_S0_jEZ4mainE4$_19E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %680, align 8, !tbaa !20
  invoke fastcc void @_ZL38checkOverlappingMemoryTwoRuntimeChecksIhEvSt8functionIFvPT_S2_S2_jEES4_PKc(ptr noundef %56, ptr noundef %57, ptr noundef nonnull @.str.27)
          to label %682 unwind label %1578

682:                                              ; preds = %677
  %683 = load ptr, ptr %680, align 8, !tbaa !20
  %684 = icmp eq ptr %683, null
  br i1 %684, label %690, label %685

685:                                              ; preds = %682
  %686 = invoke noundef i1 %683(ptr noundef nonnull align 8 dereferenceable(32) %57, ptr noundef nonnull align 8 dereferenceable(32) %57, i32 noundef 3)
          to label %690 unwind label %687

687:                                              ; preds = %685
  %688 = landingpad { ptr, i32 }
          catch ptr null
  %689 = extractvalue { ptr, i32 } %688, 0
  call void @__clang_call_terminate(ptr %689) #21
  unreachable

690:                                              ; preds = %682, %685
  %691 = load ptr, ptr %678, align 8, !tbaa !20
  %692 = icmp eq ptr %691, null
  br i1 %692, label %698, label %693

693:                                              ; preds = %690
  %694 = invoke noundef i1 %691(ptr noundef nonnull align 8 dereferenceable(32) %56, ptr noundef nonnull align 8 dereferenceable(32) %56, i32 noundef 3)
          to label %698 unwind label %695

695:                                              ; preds = %693
  %696 = landingpad { ptr, i32 }
          catch ptr null
  %697 = extractvalue { ptr, i32 } %696, 0
  call void @__clang_call_terminate(ptr %697) #21
  unreachable

698:                                              ; preds = %690, %693
  %699 = getelementptr inbounds nuw i8, ptr %58, i64 16
  %700 = getelementptr inbounds nuw i8, ptr %58, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %58, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_S0_jEZ4mainE4$_18E9_M_invokeERKSt9_Any_dataOS0_S7_S7_Oj", ptr %700, align 8, !tbaa !25
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_S0_jEZ4mainE4$_18E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %699, align 8, !tbaa !20
  %701 = getelementptr inbounds nuw i8, ptr %59, i64 16
  %702 = getelementptr inbounds nuw i8, ptr %59, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %59, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_S0_jEZ4mainE4$_19E9_M_invokeERKSt9_Any_dataOS0_S7_S7_Oj", ptr %702, align 8, !tbaa !25
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_S0_jEZ4mainE4$_19E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %701, align 8, !tbaa !20
  invoke fastcc void @_ZL38checkOverlappingMemoryTwoRuntimeChecksIjEvSt8functionIFvPT_S2_S2_jEES4_PKc(ptr noundef %58, ptr noundef %59, ptr noundef nonnull @.str.28)
          to label %703 unwind label %1595

703:                                              ; preds = %698
  %704 = load ptr, ptr %701, align 8, !tbaa !20
  %705 = icmp eq ptr %704, null
  br i1 %705, label %711, label %706

706:                                              ; preds = %703
  %707 = invoke noundef i1 %704(ptr noundef nonnull align 8 dereferenceable(32) %59, ptr noundef nonnull align 8 dereferenceable(32) %59, i32 noundef 3)
          to label %711 unwind label %708

708:                                              ; preds = %706
  %709 = landingpad { ptr, i32 }
          catch ptr null
  %710 = extractvalue { ptr, i32 } %709, 0
  call void @__clang_call_terminate(ptr %710) #21
  unreachable

711:                                              ; preds = %703, %706
  %712 = load ptr, ptr %699, align 8, !tbaa !20
  %713 = icmp eq ptr %712, null
  br i1 %713, label %719, label %714

714:                                              ; preds = %711
  %715 = invoke noundef i1 %712(ptr noundef nonnull align 8 dereferenceable(32) %58, ptr noundef nonnull align 8 dereferenceable(32) %58, i32 noundef 3)
          to label %719 unwind label %716

716:                                              ; preds = %714
  %717 = landingpad { ptr, i32 }
          catch ptr null
  %718 = extractvalue { ptr, i32 } %717, 0
  call void @__clang_call_terminate(ptr %718) #21
  unreachable

719:                                              ; preds = %711, %714
  %720 = getelementptr inbounds nuw i8, ptr %60, i64 16
  %721 = getelementptr inbounds nuw i8, ptr %60, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %60, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPmS0_S0_jEZ4mainE4$_18E9_M_invokeERKSt9_Any_dataOS0_S7_S7_Oj", ptr %721, align 8, !tbaa !29
  store ptr @"_ZNSt17_Function_handlerIFvPmS0_S0_jEZ4mainE4$_18E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %720, align 8, !tbaa !20
  %722 = getelementptr inbounds nuw i8, ptr %61, i64 16
  %723 = getelementptr inbounds nuw i8, ptr %61, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %61, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPmS0_S0_jEZ4mainE4$_19E9_M_invokeERKSt9_Any_dataOS0_S7_S7_Oj", ptr %723, align 8, !tbaa !29
  store ptr @"_ZNSt17_Function_handlerIFvPmS0_S0_jEZ4mainE4$_19E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %722, align 8, !tbaa !20
  invoke fastcc void @_ZL38checkOverlappingMemoryTwoRuntimeChecksImEvSt8functionIFvPT_S2_S2_jEES4_PKc(ptr noundef %60, ptr noundef %61, ptr noundef nonnull @.str.29)
          to label %724 unwind label %1612

724:                                              ; preds = %719
  %725 = load ptr, ptr %722, align 8, !tbaa !20
  %726 = icmp eq ptr %725, null
  br i1 %726, label %732, label %727

727:                                              ; preds = %724
  %728 = invoke noundef i1 %725(ptr noundef nonnull align 8 dereferenceable(32) %61, ptr noundef nonnull align 8 dereferenceable(32) %61, i32 noundef 3)
          to label %732 unwind label %729

729:                                              ; preds = %727
  %730 = landingpad { ptr, i32 }
          catch ptr null
  %731 = extractvalue { ptr, i32 } %730, 0
  call void @__clang_call_terminate(ptr %731) #21
  unreachable

732:                                              ; preds = %724, %727
  %733 = load ptr, ptr %720, align 8, !tbaa !20
  %734 = icmp eq ptr %733, null
  br i1 %734, label %740, label %735

735:                                              ; preds = %732
  %736 = invoke noundef i1 %733(ptr noundef nonnull align 8 dereferenceable(32) %60, ptr noundef nonnull align 8 dereferenceable(32) %60, i32 noundef 3)
          to label %740 unwind label %737

737:                                              ; preds = %735
  %738 = landingpad { ptr, i32 }
          catch ptr null
  %739 = extractvalue { ptr, i32 } %738, 0
  call void @__clang_call_terminate(ptr %739) #21
  unreachable

740:                                              ; preds = %732, %735
  %741 = getelementptr inbounds nuw i8, ptr %62, i64 16
  %742 = getelementptr inbounds nuw i8, ptr %62, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %62, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPhS0_jjEZ4mainE4$_20E9_M_invokeERKSt9_Any_dataOS0_S7_OjS8_", ptr %742, align 8, !tbaa !31
  store ptr @"_ZNSt17_Function_handlerIFvPhS0_jjEZ4mainE4$_20E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %741, align 8, !tbaa !20
  %743 = getelementptr inbounds nuw i8, ptr %63, i64 16
  %744 = getelementptr inbounds nuw i8, ptr %63, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %63, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPhS0_jjEZ4mainE4$_21E9_M_invokeERKSt9_Any_dataOS0_S7_OjS8_", ptr %744, align 8, !tbaa !31
  store ptr @"_ZNSt17_Function_handlerIFvPhS0_jjEZ4mainE4$_21E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %743, align 8, !tbaa !20
  invoke fastcc void @_ZL44checkOverlappingMemoryTwoRuntimeChecksNestedIhEvSt8functionIFvPT_S2_jjEES4_iiPKc(ptr noundef %62, ptr noundef %63, i32 noundef 100, ptr noundef nonnull @.str.30)
          to label %745 unwind label %1629

745:                                              ; preds = %740
  %746 = load ptr, ptr %743, align 8, !tbaa !20
  %747 = icmp eq ptr %746, null
  br i1 %747, label %753, label %748

748:                                              ; preds = %745
  %749 = invoke noundef i1 %746(ptr noundef nonnull align 8 dereferenceable(32) %63, ptr noundef nonnull align 8 dereferenceable(32) %63, i32 noundef 3)
          to label %753 unwind label %750

750:                                              ; preds = %748
  %751 = landingpad { ptr, i32 }
          catch ptr null
  %752 = extractvalue { ptr, i32 } %751, 0
  call void @__clang_call_terminate(ptr %752) #21
  unreachable

753:                                              ; preds = %745, %748
  %754 = load ptr, ptr %741, align 8, !tbaa !20
  %755 = icmp eq ptr %754, null
  br i1 %755, label %761, label %756

756:                                              ; preds = %753
  %757 = invoke noundef i1 %754(ptr noundef nonnull align 8 dereferenceable(32) %62, ptr noundef nonnull align 8 dereferenceable(32) %62, i32 noundef 3)
          to label %761 unwind label %758

758:                                              ; preds = %756
  %759 = landingpad { ptr, i32 }
          catch ptr null
  %760 = extractvalue { ptr, i32 } %759, 0
  call void @__clang_call_terminate(ptr %760) #21
  unreachable

761:                                              ; preds = %753, %756
  %762 = getelementptr inbounds nuw i8, ptr %64, i64 16
  %763 = getelementptr inbounds nuw i8, ptr %64, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %64, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jjEZ4mainE4$_20E9_M_invokeERKSt9_Any_dataOS0_S7_OjS8_", ptr %763, align 8, !tbaa !33
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jjEZ4mainE4$_20E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %762, align 8, !tbaa !20
  %764 = getelementptr inbounds nuw i8, ptr %65, i64 16
  %765 = getelementptr inbounds nuw i8, ptr %65, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %65, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jjEZ4mainE4$_21E9_M_invokeERKSt9_Any_dataOS0_S7_OjS8_", ptr %765, align 8, !tbaa !33
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jjEZ4mainE4$_21E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %764, align 8, !tbaa !20
  invoke fastcc void @_ZL44checkOverlappingMemoryTwoRuntimeChecksNestedIjEvSt8functionIFvPT_S2_jjEES4_iiPKc(ptr noundef %64, ptr noundef %65, i32 noundef 100, ptr noundef nonnull @.str.31)
          to label %766 unwind label %1646

766:                                              ; preds = %761
  %767 = load ptr, ptr %764, align 8, !tbaa !20
  %768 = icmp eq ptr %767, null
  br i1 %768, label %774, label %769

769:                                              ; preds = %766
  %770 = invoke noundef i1 %767(ptr noundef nonnull align 8 dereferenceable(32) %65, ptr noundef nonnull align 8 dereferenceable(32) %65, i32 noundef 3)
          to label %774 unwind label %771

771:                                              ; preds = %769
  %772 = landingpad { ptr, i32 }
          catch ptr null
  %773 = extractvalue { ptr, i32 } %772, 0
  call void @__clang_call_terminate(ptr %773) #21
  unreachable

774:                                              ; preds = %766, %769
  %775 = load ptr, ptr %762, align 8, !tbaa !20
  %776 = icmp eq ptr %775, null
  br i1 %776, label %782, label %777

777:                                              ; preds = %774
  %778 = invoke noundef i1 %775(ptr noundef nonnull align 8 dereferenceable(32) %64, ptr noundef nonnull align 8 dereferenceable(32) %64, i32 noundef 3)
          to label %782 unwind label %779

779:                                              ; preds = %777
  %780 = landingpad { ptr, i32 }
          catch ptr null
  %781 = extractvalue { ptr, i32 } %780, 0
  call void @__clang_call_terminate(ptr %781) #21
  unreachable

782:                                              ; preds = %774, %777
  %783 = getelementptr inbounds nuw i8, ptr %66, i64 16
  %784 = getelementptr inbounds nuw i8, ptr %66, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %66, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPmS0_jjEZ4mainE4$_20E9_M_invokeERKSt9_Any_dataOS0_S7_OjS8_", ptr %784, align 8, !tbaa !35
  store ptr @"_ZNSt17_Function_handlerIFvPmS0_jjEZ4mainE4$_20E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %783, align 8, !tbaa !20
  %785 = getelementptr inbounds nuw i8, ptr %67, i64 16
  %786 = getelementptr inbounds nuw i8, ptr %67, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %67, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPmS0_jjEZ4mainE4$_21E9_M_invokeERKSt9_Any_dataOS0_S7_OjS8_", ptr %786, align 8, !tbaa !35
  store ptr @"_ZNSt17_Function_handlerIFvPmS0_jjEZ4mainE4$_21E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %785, align 8, !tbaa !20
  invoke fastcc void @_ZL44checkOverlappingMemoryTwoRuntimeChecksNestedImEvSt8functionIFvPT_S2_jjEES4_iiPKc(ptr noundef %66, ptr noundef %67, i32 noundef 100, ptr noundef nonnull @.str.32)
          to label %787 unwind label %1663

787:                                              ; preds = %782
  %788 = load ptr, ptr %785, align 8, !tbaa !20
  %789 = icmp eq ptr %788, null
  br i1 %789, label %795, label %790

790:                                              ; preds = %787
  %791 = invoke noundef i1 %788(ptr noundef nonnull align 8 dereferenceable(32) %67, ptr noundef nonnull align 8 dereferenceable(32) %67, i32 noundef 3)
          to label %795 unwind label %792

792:                                              ; preds = %790
  %793 = landingpad { ptr, i32 }
          catch ptr null
  %794 = extractvalue { ptr, i32 } %793, 0
  call void @__clang_call_terminate(ptr %794) #21
  unreachable

795:                                              ; preds = %787, %790
  %796 = load ptr, ptr %783, align 8, !tbaa !20
  %797 = icmp eq ptr %796, null
  br i1 %797, label %803, label %798

798:                                              ; preds = %795
  %799 = invoke noundef i1 %796(ptr noundef nonnull align 8 dereferenceable(32) %66, ptr noundef nonnull align 8 dereferenceable(32) %66, i32 noundef 3)
          to label %803 unwind label %800

800:                                              ; preds = %798
  %801 = landingpad { ptr, i32 }
          catch ptr null
  %802 = extractvalue { ptr, i32 } %801, 0
  call void @__clang_call_terminate(ptr %802) #21
  unreachable

803:                                              ; preds = %795, %798
  %804 = getelementptr inbounds nuw i8, ptr %68, i64 16
  %805 = getelementptr inbounds nuw i8, ptr %68, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %68, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPhS0_jjEZ4mainE4$_20E9_M_invokeERKSt9_Any_dataOS0_S7_OjS8_", ptr %805, align 8, !tbaa !31
  store ptr @"_ZNSt17_Function_handlerIFvPhS0_jjEZ4mainE4$_20E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %804, align 8, !tbaa !20
  %806 = getelementptr inbounds nuw i8, ptr %69, i64 16
  %807 = getelementptr inbounds nuw i8, ptr %69, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %69, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPhS0_jjEZ4mainE4$_21E9_M_invokeERKSt9_Any_dataOS0_S7_OjS8_", ptr %807, align 8, !tbaa !31
  store ptr @"_ZNSt17_Function_handlerIFvPhS0_jjEZ4mainE4$_21E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %806, align 8, !tbaa !20
  invoke fastcc void @_ZL44checkOverlappingMemoryTwoRuntimeChecksNestedIhEvSt8functionIFvPT_S2_jjEES4_iiPKc(ptr noundef %68, ptr noundef %69, i32 noundef 50, ptr noundef nonnull @.str.33)
          to label %808 unwind label %1680

808:                                              ; preds = %803
  %809 = load ptr, ptr %806, align 8, !tbaa !20
  %810 = icmp eq ptr %809, null
  br i1 %810, label %816, label %811

811:                                              ; preds = %808
  %812 = invoke noundef i1 %809(ptr noundef nonnull align 8 dereferenceable(32) %69, ptr noundef nonnull align 8 dereferenceable(32) %69, i32 noundef 3)
          to label %816 unwind label %813

813:                                              ; preds = %811
  %814 = landingpad { ptr, i32 }
          catch ptr null
  %815 = extractvalue { ptr, i32 } %814, 0
  call void @__clang_call_terminate(ptr %815) #21
  unreachable

816:                                              ; preds = %808, %811
  %817 = load ptr, ptr %804, align 8, !tbaa !20
  %818 = icmp eq ptr %817, null
  br i1 %818, label %824, label %819

819:                                              ; preds = %816
  %820 = invoke noundef i1 %817(ptr noundef nonnull align 8 dereferenceable(32) %68, ptr noundef nonnull align 8 dereferenceable(32) %68, i32 noundef 3)
          to label %824 unwind label %821

821:                                              ; preds = %819
  %822 = landingpad { ptr, i32 }
          catch ptr null
  %823 = extractvalue { ptr, i32 } %822, 0
  call void @__clang_call_terminate(ptr %823) #21
  unreachable

824:                                              ; preds = %816, %819
  %825 = getelementptr inbounds nuw i8, ptr %70, i64 16
  %826 = getelementptr inbounds nuw i8, ptr %70, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %70, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jjEZ4mainE4$_20E9_M_invokeERKSt9_Any_dataOS0_S7_OjS8_", ptr %826, align 8, !tbaa !33
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jjEZ4mainE4$_20E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %825, align 8, !tbaa !20
  %827 = getelementptr inbounds nuw i8, ptr %71, i64 16
  %828 = getelementptr inbounds nuw i8, ptr %71, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %71, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jjEZ4mainE4$_21E9_M_invokeERKSt9_Any_dataOS0_S7_OjS8_", ptr %828, align 8, !tbaa !33
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jjEZ4mainE4$_21E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %827, align 8, !tbaa !20
  invoke fastcc void @_ZL44checkOverlappingMemoryTwoRuntimeChecksNestedIjEvSt8functionIFvPT_S2_jjEES4_iiPKc(ptr noundef %70, ptr noundef %71, i32 noundef 50, ptr noundef nonnull @.str.34)
          to label %829 unwind label %1697

829:                                              ; preds = %824
  %830 = load ptr, ptr %827, align 8, !tbaa !20
  %831 = icmp eq ptr %830, null
  br i1 %831, label %837, label %832

832:                                              ; preds = %829
  %833 = invoke noundef i1 %830(ptr noundef nonnull align 8 dereferenceable(32) %71, ptr noundef nonnull align 8 dereferenceable(32) %71, i32 noundef 3)
          to label %837 unwind label %834

834:                                              ; preds = %832
  %835 = landingpad { ptr, i32 }
          catch ptr null
  %836 = extractvalue { ptr, i32 } %835, 0
  call void @__clang_call_terminate(ptr %836) #21
  unreachable

837:                                              ; preds = %829, %832
  %838 = load ptr, ptr %825, align 8, !tbaa !20
  %839 = icmp eq ptr %838, null
  br i1 %839, label %845, label %840

840:                                              ; preds = %837
  %841 = invoke noundef i1 %838(ptr noundef nonnull align 8 dereferenceable(32) %70, ptr noundef nonnull align 8 dereferenceable(32) %70, i32 noundef 3)
          to label %845 unwind label %842

842:                                              ; preds = %840
  %843 = landingpad { ptr, i32 }
          catch ptr null
  %844 = extractvalue { ptr, i32 } %843, 0
  call void @__clang_call_terminate(ptr %844) #21
  unreachable

845:                                              ; preds = %837, %840
  %846 = getelementptr inbounds nuw i8, ptr %72, i64 16
  %847 = getelementptr inbounds nuw i8, ptr %72, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %72, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPmS0_jjEZ4mainE4$_20E9_M_invokeERKSt9_Any_dataOS0_S7_OjS8_", ptr %847, align 8, !tbaa !35
  store ptr @"_ZNSt17_Function_handlerIFvPmS0_jjEZ4mainE4$_20E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %846, align 8, !tbaa !20
  %848 = getelementptr inbounds nuw i8, ptr %73, i64 16
  %849 = getelementptr inbounds nuw i8, ptr %73, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %73, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPmS0_jjEZ4mainE4$_21E9_M_invokeERKSt9_Any_dataOS0_S7_OjS8_", ptr %849, align 8, !tbaa !35
  store ptr @"_ZNSt17_Function_handlerIFvPmS0_jjEZ4mainE4$_21E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %848, align 8, !tbaa !20
  invoke fastcc void @_ZL44checkOverlappingMemoryTwoRuntimeChecksNestedImEvSt8functionIFvPT_S2_jjEES4_iiPKc(ptr noundef %72, ptr noundef %73, i32 noundef 50, ptr noundef nonnull @.str.35)
          to label %850 unwind label %1714

850:                                              ; preds = %845
  %851 = load ptr, ptr %848, align 8, !tbaa !20
  %852 = icmp eq ptr %851, null
  br i1 %852, label %858, label %853

853:                                              ; preds = %850
  %854 = invoke noundef i1 %851(ptr noundef nonnull align 8 dereferenceable(32) %73, ptr noundef nonnull align 8 dereferenceable(32) %73, i32 noundef 3)
          to label %858 unwind label %855

855:                                              ; preds = %853
  %856 = landingpad { ptr, i32 }
          catch ptr null
  %857 = extractvalue { ptr, i32 } %856, 0
  call void @__clang_call_terminate(ptr %857) #21
  unreachable

858:                                              ; preds = %850, %853
  %859 = load ptr, ptr %846, align 8, !tbaa !20
  %860 = icmp eq ptr %859, null
  br i1 %860, label %866, label %861

861:                                              ; preds = %858
  %862 = invoke noundef i1 %859(ptr noundef nonnull align 8 dereferenceable(32) %72, ptr noundef nonnull align 8 dereferenceable(32) %72, i32 noundef 3)
          to label %866 unwind label %863

863:                                              ; preds = %861
  %864 = landingpad { ptr, i32 }
          catch ptr null
  %865 = extractvalue { ptr, i32 } %864, 0
  call void @__clang_call_terminate(ptr %865) #21
  unreachable

866:                                              ; preds = %858, %861
  %867 = getelementptr inbounds nuw i8, ptr %74, i64 16
  %868 = getelementptr inbounds nuw i8, ptr %74, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %74, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPhS0_jjEZ4mainE4$_22E9_M_invokeERKSt9_Any_dataOS0_S7_OjS8_", ptr %868, align 8, !tbaa !31
  store ptr @"_ZNSt17_Function_handlerIFvPhS0_jjEZ4mainE4$_22E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %867, align 8, !tbaa !20
  %869 = getelementptr inbounds nuw i8, ptr %75, i64 16
  %870 = getelementptr inbounds nuw i8, ptr %75, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %75, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPhS0_jjEZ4mainE4$_23E9_M_invokeERKSt9_Any_dataOS0_S7_OjS8_", ptr %870, align 8, !tbaa !31
  store ptr @"_ZNSt17_Function_handlerIFvPhS0_jjEZ4mainE4$_23E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %869, align 8, !tbaa !20
  invoke fastcc void @_ZL44checkOverlappingMemoryTwoRuntimeChecksNestedIhEvSt8functionIFvPT_S2_jjEES4_iiPKc(ptr noundef %74, ptr noundef %75, i32 noundef 100, ptr noundef nonnull @.str.36)
          to label %871 unwind label %1731

871:                                              ; preds = %866
  %872 = load ptr, ptr %869, align 8, !tbaa !20
  %873 = icmp eq ptr %872, null
  br i1 %873, label %879, label %874

874:                                              ; preds = %871
  %875 = invoke noundef i1 %872(ptr noundef nonnull align 8 dereferenceable(32) %75, ptr noundef nonnull align 8 dereferenceable(32) %75, i32 noundef 3)
          to label %879 unwind label %876

876:                                              ; preds = %874
  %877 = landingpad { ptr, i32 }
          catch ptr null
  %878 = extractvalue { ptr, i32 } %877, 0
  call void @__clang_call_terminate(ptr %878) #21
  unreachable

879:                                              ; preds = %871, %874
  %880 = load ptr, ptr %867, align 8, !tbaa !20
  %881 = icmp eq ptr %880, null
  br i1 %881, label %887, label %882

882:                                              ; preds = %879
  %883 = invoke noundef i1 %880(ptr noundef nonnull align 8 dereferenceable(32) %74, ptr noundef nonnull align 8 dereferenceable(32) %74, i32 noundef 3)
          to label %887 unwind label %884

884:                                              ; preds = %882
  %885 = landingpad { ptr, i32 }
          catch ptr null
  %886 = extractvalue { ptr, i32 } %885, 0
  call void @__clang_call_terminate(ptr %886) #21
  unreachable

887:                                              ; preds = %879, %882
  %888 = getelementptr inbounds nuw i8, ptr %76, i64 16
  %889 = getelementptr inbounds nuw i8, ptr %76, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %76, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jjEZ4mainE4$_22E9_M_invokeERKSt9_Any_dataOS0_S7_OjS8_", ptr %889, align 8, !tbaa !33
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jjEZ4mainE4$_22E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %888, align 8, !tbaa !20
  %890 = getelementptr inbounds nuw i8, ptr %77, i64 16
  %891 = getelementptr inbounds nuw i8, ptr %77, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %77, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jjEZ4mainE4$_23E9_M_invokeERKSt9_Any_dataOS0_S7_OjS8_", ptr %891, align 8, !tbaa !33
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jjEZ4mainE4$_23E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %890, align 8, !tbaa !20
  invoke fastcc void @_ZL44checkOverlappingMemoryTwoRuntimeChecksNestedIjEvSt8functionIFvPT_S2_jjEES4_iiPKc(ptr noundef %76, ptr noundef %77, i32 noundef 100, ptr noundef nonnull @.str.37)
          to label %892 unwind label %1748

892:                                              ; preds = %887
  %893 = load ptr, ptr %890, align 8, !tbaa !20
  %894 = icmp eq ptr %893, null
  br i1 %894, label %900, label %895

895:                                              ; preds = %892
  %896 = invoke noundef i1 %893(ptr noundef nonnull align 8 dereferenceable(32) %77, ptr noundef nonnull align 8 dereferenceable(32) %77, i32 noundef 3)
          to label %900 unwind label %897

897:                                              ; preds = %895
  %898 = landingpad { ptr, i32 }
          catch ptr null
  %899 = extractvalue { ptr, i32 } %898, 0
  call void @__clang_call_terminate(ptr %899) #21
  unreachable

900:                                              ; preds = %892, %895
  %901 = load ptr, ptr %888, align 8, !tbaa !20
  %902 = icmp eq ptr %901, null
  br i1 %902, label %908, label %903

903:                                              ; preds = %900
  %904 = invoke noundef i1 %901(ptr noundef nonnull align 8 dereferenceable(32) %76, ptr noundef nonnull align 8 dereferenceable(32) %76, i32 noundef 3)
          to label %908 unwind label %905

905:                                              ; preds = %903
  %906 = landingpad { ptr, i32 }
          catch ptr null
  %907 = extractvalue { ptr, i32 } %906, 0
  call void @__clang_call_terminate(ptr %907) #21
  unreachable

908:                                              ; preds = %900, %903
  %909 = getelementptr inbounds nuw i8, ptr %78, i64 16
  %910 = getelementptr inbounds nuw i8, ptr %78, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %78, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPmS0_jjEZ4mainE4$_22E9_M_invokeERKSt9_Any_dataOS0_S7_OjS8_", ptr %910, align 8, !tbaa !35
  store ptr @"_ZNSt17_Function_handlerIFvPmS0_jjEZ4mainE4$_22E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %909, align 8, !tbaa !20
  %911 = getelementptr inbounds nuw i8, ptr %79, i64 16
  %912 = getelementptr inbounds nuw i8, ptr %79, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %79, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPmS0_jjEZ4mainE4$_23E9_M_invokeERKSt9_Any_dataOS0_S7_OjS8_", ptr %912, align 8, !tbaa !35
  store ptr @"_ZNSt17_Function_handlerIFvPmS0_jjEZ4mainE4$_23E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %911, align 8, !tbaa !20
  invoke fastcc void @_ZL44checkOverlappingMemoryTwoRuntimeChecksNestedImEvSt8functionIFvPT_S2_jjEES4_iiPKc(ptr noundef %78, ptr noundef %79, i32 noundef 100, ptr noundef nonnull @.str.38)
          to label %913 unwind label %1765

913:                                              ; preds = %908
  %914 = load ptr, ptr %911, align 8, !tbaa !20
  %915 = icmp eq ptr %914, null
  br i1 %915, label %921, label %916

916:                                              ; preds = %913
  %917 = invoke noundef i1 %914(ptr noundef nonnull align 8 dereferenceable(32) %79, ptr noundef nonnull align 8 dereferenceable(32) %79, i32 noundef 3)
          to label %921 unwind label %918

918:                                              ; preds = %916
  %919 = landingpad { ptr, i32 }
          catch ptr null
  %920 = extractvalue { ptr, i32 } %919, 0
  call void @__clang_call_terminate(ptr %920) #21
  unreachable

921:                                              ; preds = %913, %916
  %922 = load ptr, ptr %909, align 8, !tbaa !20
  %923 = icmp eq ptr %922, null
  br i1 %923, label %929, label %924

924:                                              ; preds = %921
  %925 = invoke noundef i1 %922(ptr noundef nonnull align 8 dereferenceable(32) %78, ptr noundef nonnull align 8 dereferenceable(32) %78, i32 noundef 3)
          to label %929 unwind label %926

926:                                              ; preds = %924
  %927 = landingpad { ptr, i32 }
          catch ptr null
  %928 = extractvalue { ptr, i32 } %927, 0
  call void @__clang_call_terminate(ptr %928) #21
  unreachable

929:                                              ; preds = %921, %924
  %930 = getelementptr inbounds nuw i8, ptr %80, i64 16
  %931 = getelementptr inbounds nuw i8, ptr %80, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %80, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPhS0_jjEZ4mainE4$_22E9_M_invokeERKSt9_Any_dataOS0_S7_OjS8_", ptr %931, align 8, !tbaa !31
  store ptr @"_ZNSt17_Function_handlerIFvPhS0_jjEZ4mainE4$_22E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %930, align 8, !tbaa !20
  %932 = getelementptr inbounds nuw i8, ptr %81, i64 16
  %933 = getelementptr inbounds nuw i8, ptr %81, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %81, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPhS0_jjEZ4mainE4$_23E9_M_invokeERKSt9_Any_dataOS0_S7_OjS8_", ptr %933, align 8, !tbaa !31
  store ptr @"_ZNSt17_Function_handlerIFvPhS0_jjEZ4mainE4$_23E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %932, align 8, !tbaa !20
  invoke fastcc void @_ZL44checkOverlappingMemoryTwoRuntimeChecksNestedIhEvSt8functionIFvPT_S2_jjEES4_iiPKc(ptr noundef %80, ptr noundef %81, i32 noundef 50, ptr noundef nonnull @.str.39)
          to label %934 unwind label %1782

934:                                              ; preds = %929
  %935 = load ptr, ptr %932, align 8, !tbaa !20
  %936 = icmp eq ptr %935, null
  br i1 %936, label %942, label %937

937:                                              ; preds = %934
  %938 = invoke noundef i1 %935(ptr noundef nonnull align 8 dereferenceable(32) %81, ptr noundef nonnull align 8 dereferenceable(32) %81, i32 noundef 3)
          to label %942 unwind label %939

939:                                              ; preds = %937
  %940 = landingpad { ptr, i32 }
          catch ptr null
  %941 = extractvalue { ptr, i32 } %940, 0
  call void @__clang_call_terminate(ptr %941) #21
  unreachable

942:                                              ; preds = %934, %937
  %943 = load ptr, ptr %930, align 8, !tbaa !20
  %944 = icmp eq ptr %943, null
  br i1 %944, label %950, label %945

945:                                              ; preds = %942
  %946 = invoke noundef i1 %943(ptr noundef nonnull align 8 dereferenceable(32) %80, ptr noundef nonnull align 8 dereferenceable(32) %80, i32 noundef 3)
          to label %950 unwind label %947

947:                                              ; preds = %945
  %948 = landingpad { ptr, i32 }
          catch ptr null
  %949 = extractvalue { ptr, i32 } %948, 0
  call void @__clang_call_terminate(ptr %949) #21
  unreachable

950:                                              ; preds = %942, %945
  %951 = getelementptr inbounds nuw i8, ptr %82, i64 16
  %952 = getelementptr inbounds nuw i8, ptr %82, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %82, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jjEZ4mainE4$_22E9_M_invokeERKSt9_Any_dataOS0_S7_OjS8_", ptr %952, align 8, !tbaa !33
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jjEZ4mainE4$_22E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %951, align 8, !tbaa !20
  %953 = getelementptr inbounds nuw i8, ptr %83, i64 16
  %954 = getelementptr inbounds nuw i8, ptr %83, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %83, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jjEZ4mainE4$_23E9_M_invokeERKSt9_Any_dataOS0_S7_OjS8_", ptr %954, align 8, !tbaa !33
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jjEZ4mainE4$_23E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %953, align 8, !tbaa !20
  invoke fastcc void @_ZL44checkOverlappingMemoryTwoRuntimeChecksNestedIjEvSt8functionIFvPT_S2_jjEES4_iiPKc(ptr noundef %82, ptr noundef %83, i32 noundef 50, ptr noundef nonnull @.str.40)
          to label %955 unwind label %1799

955:                                              ; preds = %950
  %956 = load ptr, ptr %953, align 8, !tbaa !20
  %957 = icmp eq ptr %956, null
  br i1 %957, label %963, label %958

958:                                              ; preds = %955
  %959 = invoke noundef i1 %956(ptr noundef nonnull align 8 dereferenceable(32) %83, ptr noundef nonnull align 8 dereferenceable(32) %83, i32 noundef 3)
          to label %963 unwind label %960

960:                                              ; preds = %958
  %961 = landingpad { ptr, i32 }
          catch ptr null
  %962 = extractvalue { ptr, i32 } %961, 0
  call void @__clang_call_terminate(ptr %962) #21
  unreachable

963:                                              ; preds = %955, %958
  %964 = load ptr, ptr %951, align 8, !tbaa !20
  %965 = icmp eq ptr %964, null
  br i1 %965, label %971, label %966

966:                                              ; preds = %963
  %967 = invoke noundef i1 %964(ptr noundef nonnull align 8 dereferenceable(32) %82, ptr noundef nonnull align 8 dereferenceable(32) %82, i32 noundef 3)
          to label %971 unwind label %968

968:                                              ; preds = %966
  %969 = landingpad { ptr, i32 }
          catch ptr null
  %970 = extractvalue { ptr, i32 } %969, 0
  call void @__clang_call_terminate(ptr %970) #21
  unreachable

971:                                              ; preds = %963, %966
  %972 = getelementptr inbounds nuw i8, ptr %84, i64 16
  %973 = getelementptr inbounds nuw i8, ptr %84, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %84, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPmS0_jjEZ4mainE4$_22E9_M_invokeERKSt9_Any_dataOS0_S7_OjS8_", ptr %973, align 8, !tbaa !35
  store ptr @"_ZNSt17_Function_handlerIFvPmS0_jjEZ4mainE4$_22E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %972, align 8, !tbaa !20
  %974 = getelementptr inbounds nuw i8, ptr %85, i64 16
  %975 = getelementptr inbounds nuw i8, ptr %85, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %85, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPmS0_jjEZ4mainE4$_23E9_M_invokeERKSt9_Any_dataOS0_S7_OjS8_", ptr %975, align 8, !tbaa !35
  store ptr @"_ZNSt17_Function_handlerIFvPmS0_jjEZ4mainE4$_23E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %974, align 8, !tbaa !20
  invoke fastcc void @_ZL44checkOverlappingMemoryTwoRuntimeChecksNestedImEvSt8functionIFvPT_S2_jjEES4_iiPKc(ptr noundef %84, ptr noundef %85, i32 noundef 50, ptr noundef nonnull @.str.41)
          to label %976 unwind label %1816

976:                                              ; preds = %971
  %977 = load ptr, ptr %974, align 8, !tbaa !20
  %978 = icmp eq ptr %977, null
  br i1 %978, label %984, label %979

979:                                              ; preds = %976
  %980 = invoke noundef i1 %977(ptr noundef nonnull align 8 dereferenceable(32) %85, ptr noundef nonnull align 8 dereferenceable(32) %85, i32 noundef 3)
          to label %984 unwind label %981

981:                                              ; preds = %979
  %982 = landingpad { ptr, i32 }
          catch ptr null
  %983 = extractvalue { ptr, i32 } %982, 0
  call void @__clang_call_terminate(ptr %983) #21
  unreachable

984:                                              ; preds = %976, %979
  %985 = load ptr, ptr %972, align 8, !tbaa !20
  %986 = icmp eq ptr %985, null
  br i1 %986, label %992, label %987

987:                                              ; preds = %984
  %988 = invoke noundef i1 %985(ptr noundef nonnull align 8 dereferenceable(32) %84, ptr noundef nonnull align 8 dereferenceable(32) %84, i32 noundef 3)
          to label %992 unwind label %989

989:                                              ; preds = %987
  %990 = landingpad { ptr, i32 }
          catch ptr null
  %991 = extractvalue { ptr, i32 } %990, 0
  call void @__clang_call_terminate(ptr %991) #21
  unreachable

992:                                              ; preds = %984, %987
  %993 = getelementptr inbounds nuw i8, ptr %86, i64 16
  %994 = getelementptr inbounds nuw i8, ptr %86, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %86, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPhS0_jjEZ4mainE4$_24E9_M_invokeERKSt9_Any_dataOS0_S7_OjS8_", ptr %994, align 8, !tbaa !31
  store ptr @"_ZNSt17_Function_handlerIFvPhS0_jjEZ4mainE4$_24E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %993, align 8, !tbaa !20
  %995 = getelementptr inbounds nuw i8, ptr %87, i64 16
  %996 = getelementptr inbounds nuw i8, ptr %87, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %87, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPhS0_jjEZ4mainE4$_25E9_M_invokeERKSt9_Any_dataOS0_S7_OjS8_", ptr %996, align 8, !tbaa !31
  store ptr @"_ZNSt17_Function_handlerIFvPhS0_jjEZ4mainE4$_25E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %995, align 8, !tbaa !20
  invoke fastcc void @_ZL44checkOverlappingMemoryTwoRuntimeChecksNestedIhEvSt8functionIFvPT_S2_jjEES4_iiPKc(ptr noundef %86, ptr noundef %87, i32 noundef 100, ptr noundef nonnull @.str.42)
          to label %997 unwind label %1833

997:                                              ; preds = %992
  %998 = load ptr, ptr %995, align 8, !tbaa !20
  %999 = icmp eq ptr %998, null
  br i1 %999, label %1005, label %1000

1000:                                             ; preds = %997
  %1001 = invoke noundef i1 %998(ptr noundef nonnull align 8 dereferenceable(32) %87, ptr noundef nonnull align 8 dereferenceable(32) %87, i32 noundef 3)
          to label %1005 unwind label %1002

1002:                                             ; preds = %1000
  %1003 = landingpad { ptr, i32 }
          catch ptr null
  %1004 = extractvalue { ptr, i32 } %1003, 0
  call void @__clang_call_terminate(ptr %1004) #21
  unreachable

1005:                                             ; preds = %997, %1000
  %1006 = load ptr, ptr %993, align 8, !tbaa !20
  %1007 = icmp eq ptr %1006, null
  br i1 %1007, label %1013, label %1008

1008:                                             ; preds = %1005
  %1009 = invoke noundef i1 %1006(ptr noundef nonnull align 8 dereferenceable(32) %86, ptr noundef nonnull align 8 dereferenceable(32) %86, i32 noundef 3)
          to label %1013 unwind label %1010

1010:                                             ; preds = %1008
  %1011 = landingpad { ptr, i32 }
          catch ptr null
  %1012 = extractvalue { ptr, i32 } %1011, 0
  call void @__clang_call_terminate(ptr %1012) #21
  unreachable

1013:                                             ; preds = %1005, %1008
  %1014 = getelementptr inbounds nuw i8, ptr %88, i64 16
  %1015 = getelementptr inbounds nuw i8, ptr %88, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %88, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jjEZ4mainE4$_24E9_M_invokeERKSt9_Any_dataOS0_S7_OjS8_", ptr %1015, align 8, !tbaa !33
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jjEZ4mainE4$_24E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %1014, align 8, !tbaa !20
  %1016 = getelementptr inbounds nuw i8, ptr %89, i64 16
  %1017 = getelementptr inbounds nuw i8, ptr %89, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %89, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jjEZ4mainE4$_25E9_M_invokeERKSt9_Any_dataOS0_S7_OjS8_", ptr %1017, align 8, !tbaa !33
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jjEZ4mainE4$_25E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %1016, align 8, !tbaa !20
  invoke fastcc void @_ZL44checkOverlappingMemoryTwoRuntimeChecksNestedIjEvSt8functionIFvPT_S2_jjEES4_iiPKc(ptr noundef %88, ptr noundef %89, i32 noundef 100, ptr noundef nonnull @.str.43)
          to label %1018 unwind label %1850

1018:                                             ; preds = %1013
  %1019 = load ptr, ptr %1016, align 8, !tbaa !20
  %1020 = icmp eq ptr %1019, null
  br i1 %1020, label %1026, label %1021

1021:                                             ; preds = %1018
  %1022 = invoke noundef i1 %1019(ptr noundef nonnull align 8 dereferenceable(32) %89, ptr noundef nonnull align 8 dereferenceable(32) %89, i32 noundef 3)
          to label %1026 unwind label %1023

1023:                                             ; preds = %1021
  %1024 = landingpad { ptr, i32 }
          catch ptr null
  %1025 = extractvalue { ptr, i32 } %1024, 0
  call void @__clang_call_terminate(ptr %1025) #21
  unreachable

1026:                                             ; preds = %1018, %1021
  %1027 = load ptr, ptr %1014, align 8, !tbaa !20
  %1028 = icmp eq ptr %1027, null
  br i1 %1028, label %1034, label %1029

1029:                                             ; preds = %1026
  %1030 = invoke noundef i1 %1027(ptr noundef nonnull align 8 dereferenceable(32) %88, ptr noundef nonnull align 8 dereferenceable(32) %88, i32 noundef 3)
          to label %1034 unwind label %1031

1031:                                             ; preds = %1029
  %1032 = landingpad { ptr, i32 }
          catch ptr null
  %1033 = extractvalue { ptr, i32 } %1032, 0
  call void @__clang_call_terminate(ptr %1033) #21
  unreachable

1034:                                             ; preds = %1026, %1029
  %1035 = getelementptr inbounds nuw i8, ptr %90, i64 16
  %1036 = getelementptr inbounds nuw i8, ptr %90, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %90, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPmS0_jjEZ4mainE4$_24E9_M_invokeERKSt9_Any_dataOS0_S7_OjS8_", ptr %1036, align 8, !tbaa !35
  store ptr @"_ZNSt17_Function_handlerIFvPmS0_jjEZ4mainE4$_24E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %1035, align 8, !tbaa !20
  %1037 = getelementptr inbounds nuw i8, ptr %91, i64 16
  %1038 = getelementptr inbounds nuw i8, ptr %91, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %91, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPmS0_jjEZ4mainE4$_25E9_M_invokeERKSt9_Any_dataOS0_S7_OjS8_", ptr %1038, align 8, !tbaa !35
  store ptr @"_ZNSt17_Function_handlerIFvPmS0_jjEZ4mainE4$_25E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %1037, align 8, !tbaa !20
  invoke fastcc void @_ZL44checkOverlappingMemoryTwoRuntimeChecksNestedImEvSt8functionIFvPT_S2_jjEES4_iiPKc(ptr noundef %90, ptr noundef %91, i32 noundef 100, ptr noundef nonnull @.str.44)
          to label %1039 unwind label %1867

1039:                                             ; preds = %1034
  %1040 = load ptr, ptr %1037, align 8, !tbaa !20
  %1041 = icmp eq ptr %1040, null
  br i1 %1041, label %1047, label %1042

1042:                                             ; preds = %1039
  %1043 = invoke noundef i1 %1040(ptr noundef nonnull align 8 dereferenceable(32) %91, ptr noundef nonnull align 8 dereferenceable(32) %91, i32 noundef 3)
          to label %1047 unwind label %1044

1044:                                             ; preds = %1042
  %1045 = landingpad { ptr, i32 }
          catch ptr null
  %1046 = extractvalue { ptr, i32 } %1045, 0
  call void @__clang_call_terminate(ptr %1046) #21
  unreachable

1047:                                             ; preds = %1039, %1042
  %1048 = load ptr, ptr %1035, align 8, !tbaa !20
  %1049 = icmp eq ptr %1048, null
  br i1 %1049, label %1055, label %1050

1050:                                             ; preds = %1047
  %1051 = invoke noundef i1 %1048(ptr noundef nonnull align 8 dereferenceable(32) %90, ptr noundef nonnull align 8 dereferenceable(32) %90, i32 noundef 3)
          to label %1055 unwind label %1052

1052:                                             ; preds = %1050
  %1053 = landingpad { ptr, i32 }
          catch ptr null
  %1054 = extractvalue { ptr, i32 } %1053, 0
  call void @__clang_call_terminate(ptr %1054) #21
  unreachable

1055:                                             ; preds = %1047, %1050
  %1056 = getelementptr inbounds nuw i8, ptr %92, i64 16
  %1057 = getelementptr inbounds nuw i8, ptr %92, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %92, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPhS0_jjEZ4mainE4$_26E9_M_invokeERKSt9_Any_dataOS0_S7_OjS8_", ptr %1057, align 8, !tbaa !31
  store ptr @"_ZNSt17_Function_handlerIFvPhS0_jjEZ4mainE4$_26E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %1056, align 8, !tbaa !20
  %1058 = getelementptr inbounds nuw i8, ptr %93, i64 16
  %1059 = getelementptr inbounds nuw i8, ptr %93, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %93, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPhS0_jjEZ4mainE4$_27E9_M_invokeERKSt9_Any_dataOS0_S7_OjS8_", ptr %1059, align 8, !tbaa !31
  store ptr @"_ZNSt17_Function_handlerIFvPhS0_jjEZ4mainE4$_27E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %1058, align 8, !tbaa !20
  invoke fastcc void @_ZL44checkOverlappingMemoryTwoRuntimeChecksNestedIhEvSt8functionIFvPT_S2_jjEES4_iiPKc(ptr noundef %92, ptr noundef %93, i32 noundef 100, ptr noundef nonnull @.str.45)
          to label %1060 unwind label %1884

1060:                                             ; preds = %1055
  %1061 = load ptr, ptr %1058, align 8, !tbaa !20
  %1062 = icmp eq ptr %1061, null
  br i1 %1062, label %1068, label %1063

1063:                                             ; preds = %1060
  %1064 = invoke noundef i1 %1061(ptr noundef nonnull align 8 dereferenceable(32) %93, ptr noundef nonnull align 8 dereferenceable(32) %93, i32 noundef 3)
          to label %1068 unwind label %1065

1065:                                             ; preds = %1063
  %1066 = landingpad { ptr, i32 }
          catch ptr null
  %1067 = extractvalue { ptr, i32 } %1066, 0
  call void @__clang_call_terminate(ptr %1067) #21
  unreachable

1068:                                             ; preds = %1060, %1063
  %1069 = load ptr, ptr %1056, align 8, !tbaa !20
  %1070 = icmp eq ptr %1069, null
  br i1 %1070, label %1076, label %1071

1071:                                             ; preds = %1068
  %1072 = invoke noundef i1 %1069(ptr noundef nonnull align 8 dereferenceable(32) %92, ptr noundef nonnull align 8 dereferenceable(32) %92, i32 noundef 3)
          to label %1076 unwind label %1073

1073:                                             ; preds = %1071
  %1074 = landingpad { ptr, i32 }
          catch ptr null
  %1075 = extractvalue { ptr, i32 } %1074, 0
  call void @__clang_call_terminate(ptr %1075) #21
  unreachable

1076:                                             ; preds = %1068, %1071
  %1077 = getelementptr inbounds nuw i8, ptr %94, i64 16
  %1078 = getelementptr inbounds nuw i8, ptr %94, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %94, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jjEZ4mainE4$_26E9_M_invokeERKSt9_Any_dataOS0_S7_OjS8_", ptr %1078, align 8, !tbaa !33
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jjEZ4mainE4$_26E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %1077, align 8, !tbaa !20
  %1079 = getelementptr inbounds nuw i8, ptr %95, i64 16
  %1080 = getelementptr inbounds nuw i8, ptr %95, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %95, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jjEZ4mainE4$_27E9_M_invokeERKSt9_Any_dataOS0_S7_OjS8_", ptr %1080, align 8, !tbaa !33
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jjEZ4mainE4$_27E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %1079, align 8, !tbaa !20
  invoke fastcc void @_ZL44checkOverlappingMemoryTwoRuntimeChecksNestedIjEvSt8functionIFvPT_S2_jjEES4_iiPKc(ptr noundef %94, ptr noundef %95, i32 noundef 100, ptr noundef nonnull @.str.46)
          to label %1081 unwind label %1901

1081:                                             ; preds = %1076
  %1082 = load ptr, ptr %1079, align 8, !tbaa !20
  %1083 = icmp eq ptr %1082, null
  br i1 %1083, label %1089, label %1084

1084:                                             ; preds = %1081
  %1085 = invoke noundef i1 %1082(ptr noundef nonnull align 8 dereferenceable(32) %95, ptr noundef nonnull align 8 dereferenceable(32) %95, i32 noundef 3)
          to label %1089 unwind label %1086

1086:                                             ; preds = %1084
  %1087 = landingpad { ptr, i32 }
          catch ptr null
  %1088 = extractvalue { ptr, i32 } %1087, 0
  call void @__clang_call_terminate(ptr %1088) #21
  unreachable

1089:                                             ; preds = %1081, %1084
  %1090 = load ptr, ptr %1077, align 8, !tbaa !20
  %1091 = icmp eq ptr %1090, null
  br i1 %1091, label %1097, label %1092

1092:                                             ; preds = %1089
  %1093 = invoke noundef i1 %1090(ptr noundef nonnull align 8 dereferenceable(32) %94, ptr noundef nonnull align 8 dereferenceable(32) %94, i32 noundef 3)
          to label %1097 unwind label %1094

1094:                                             ; preds = %1092
  %1095 = landingpad { ptr, i32 }
          catch ptr null
  %1096 = extractvalue { ptr, i32 } %1095, 0
  call void @__clang_call_terminate(ptr %1096) #21
  unreachable

1097:                                             ; preds = %1089, %1092
  %1098 = getelementptr inbounds nuw i8, ptr %96, i64 16
  %1099 = getelementptr inbounds nuw i8, ptr %96, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %96, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPmS0_jjEZ4mainE4$_26E9_M_invokeERKSt9_Any_dataOS0_S7_OjS8_", ptr %1099, align 8, !tbaa !35
  store ptr @"_ZNSt17_Function_handlerIFvPmS0_jjEZ4mainE4$_26E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %1098, align 8, !tbaa !20
  %1100 = getelementptr inbounds nuw i8, ptr %97, i64 16
  %1101 = getelementptr inbounds nuw i8, ptr %97, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %97, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPmS0_jjEZ4mainE4$_27E9_M_invokeERKSt9_Any_dataOS0_S7_OjS8_", ptr %1101, align 8, !tbaa !35
  store ptr @"_ZNSt17_Function_handlerIFvPmS0_jjEZ4mainE4$_27E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %1100, align 8, !tbaa !20
  invoke fastcc void @_ZL44checkOverlappingMemoryTwoRuntimeChecksNestedImEvSt8functionIFvPT_S2_jjEES4_iiPKc(ptr noundef %96, ptr noundef %97, i32 noundef 100, ptr noundef nonnull @.str.47)
          to label %1102 unwind label %1918

1102:                                             ; preds = %1097
  %1103 = load ptr, ptr %1100, align 8, !tbaa !20
  %1104 = icmp eq ptr %1103, null
  br i1 %1104, label %1110, label %1105

1105:                                             ; preds = %1102
  %1106 = invoke noundef i1 %1103(ptr noundef nonnull align 8 dereferenceable(32) %97, ptr noundef nonnull align 8 dereferenceable(32) %97, i32 noundef 3)
          to label %1110 unwind label %1107

1107:                                             ; preds = %1105
  %1108 = landingpad { ptr, i32 }
          catch ptr null
  %1109 = extractvalue { ptr, i32 } %1108, 0
  call void @__clang_call_terminate(ptr %1109) #21
  unreachable

1110:                                             ; preds = %1102, %1105
  %1111 = load ptr, ptr %1098, align 8, !tbaa !20
  %1112 = icmp eq ptr %1111, null
  br i1 %1112, label %1118, label %1113

1113:                                             ; preds = %1110
  %1114 = invoke noundef i1 %1111(ptr noundef nonnull align 8 dereferenceable(32) %96, ptr noundef nonnull align 8 dereferenceable(32) %96, i32 noundef 3)
          to label %1118 unwind label %1115

1115:                                             ; preds = %1113
  %1116 = landingpad { ptr, i32 }
          catch ptr null
  %1117 = extractvalue { ptr, i32 } %1116, 0
  call void @__clang_call_terminate(ptr %1117) #21
  unreachable

1118:                                             ; preds = %1110, %1113
  ret i32 0

1119:                                             ; preds = %109
  %1120 = landingpad { ptr, i32 }
          cleanup
  %1121 = load ptr, ptr %113, align 8, !tbaa !20
  %1122 = icmp eq ptr %1121, null
  br i1 %1122, label %1128, label %1123

1123:                                             ; preds = %1119
  %1124 = invoke noundef i1 %1121(ptr noundef nonnull align 8 dereferenceable(32) %3, ptr noundef nonnull align 8 dereferenceable(32) %3, i32 noundef 3)
          to label %1128 unwind label %1125

1125:                                             ; preds = %1123
  %1126 = landingpad { ptr, i32 }
          catch ptr null
  %1127 = extractvalue { ptr, i32 } %1126, 0
  call void @__clang_call_terminate(ptr %1127) #21
  unreachable

1128:                                             ; preds = %1119, %1123
  %1129 = load ptr, ptr %111, align 8, !tbaa !20
  %1130 = icmp eq ptr %1129, null
  br i1 %1130, label %1935, label %1131

1131:                                             ; preds = %1128
  %1132 = invoke noundef i1 %1129(ptr noundef nonnull align 8 dereferenceable(32) %2, ptr noundef nonnull align 8 dereferenceable(32) %2, i32 noundef 3)
          to label %1935 unwind label %1133

1133:                                             ; preds = %1131
  %1134 = landingpad { ptr, i32 }
          catch ptr null
  %1135 = extractvalue { ptr, i32 } %1134, 0
  call void @__clang_call_terminate(ptr %1135) #21
  unreachable

1136:                                             ; preds = %131
  %1137 = landingpad { ptr, i32 }
          cleanup
  %1138 = load ptr, ptr %134, align 8, !tbaa !20
  %1139 = icmp eq ptr %1138, null
  br i1 %1139, label %1145, label %1140

1140:                                             ; preds = %1136
  %1141 = invoke noundef i1 %1138(ptr noundef nonnull align 8 dereferenceable(32) %5, ptr noundef nonnull align 8 dereferenceable(32) %5, i32 noundef 3)
          to label %1145 unwind label %1142

1142:                                             ; preds = %1140
  %1143 = landingpad { ptr, i32 }
          catch ptr null
  %1144 = extractvalue { ptr, i32 } %1143, 0
  call void @__clang_call_terminate(ptr %1144) #21
  unreachable

1145:                                             ; preds = %1136, %1140
  %1146 = load ptr, ptr %132, align 8, !tbaa !20
  %1147 = icmp eq ptr %1146, null
  br i1 %1147, label %1935, label %1148

1148:                                             ; preds = %1145
  %1149 = invoke noundef i1 %1146(ptr noundef nonnull align 8 dereferenceable(32) %4, ptr noundef nonnull align 8 dereferenceable(32) %4, i32 noundef 3)
          to label %1935 unwind label %1150

1150:                                             ; preds = %1148
  %1151 = landingpad { ptr, i32 }
          catch ptr null
  %1152 = extractvalue { ptr, i32 } %1151, 0
  call void @__clang_call_terminate(ptr %1152) #21
  unreachable

1153:                                             ; preds = %152
  %1154 = landingpad { ptr, i32 }
          cleanup
  %1155 = load ptr, ptr %155, align 8, !tbaa !20
  %1156 = icmp eq ptr %1155, null
  br i1 %1156, label %1162, label %1157

1157:                                             ; preds = %1153
  %1158 = invoke noundef i1 %1155(ptr noundef nonnull align 8 dereferenceable(32) %7, ptr noundef nonnull align 8 dereferenceable(32) %7, i32 noundef 3)
          to label %1162 unwind label %1159

1159:                                             ; preds = %1157
  %1160 = landingpad { ptr, i32 }
          catch ptr null
  %1161 = extractvalue { ptr, i32 } %1160, 0
  call void @__clang_call_terminate(ptr %1161) #21
  unreachable

1162:                                             ; preds = %1153, %1157
  %1163 = load ptr, ptr %153, align 8, !tbaa !20
  %1164 = icmp eq ptr %1163, null
  br i1 %1164, label %1935, label %1165

1165:                                             ; preds = %1162
  %1166 = invoke noundef i1 %1163(ptr noundef nonnull align 8 dereferenceable(32) %6, ptr noundef nonnull align 8 dereferenceable(32) %6, i32 noundef 3)
          to label %1935 unwind label %1167

1167:                                             ; preds = %1165
  %1168 = landingpad { ptr, i32 }
          catch ptr null
  %1169 = extractvalue { ptr, i32 } %1168, 0
  call void @__clang_call_terminate(ptr %1169) #21
  unreachable

1170:                                             ; preds = %173
  %1171 = landingpad { ptr, i32 }
          cleanup
  %1172 = load ptr, ptr %176, align 8, !tbaa !20
  %1173 = icmp eq ptr %1172, null
  br i1 %1173, label %1179, label %1174

1174:                                             ; preds = %1170
  %1175 = invoke noundef i1 %1172(ptr noundef nonnull align 8 dereferenceable(32) %9, ptr noundef nonnull align 8 dereferenceable(32) %9, i32 noundef 3)
          to label %1179 unwind label %1176

1176:                                             ; preds = %1174
  %1177 = landingpad { ptr, i32 }
          catch ptr null
  %1178 = extractvalue { ptr, i32 } %1177, 0
  call void @__clang_call_terminate(ptr %1178) #21
  unreachable

1179:                                             ; preds = %1170, %1174
  %1180 = load ptr, ptr %174, align 8, !tbaa !20
  %1181 = icmp eq ptr %1180, null
  br i1 %1181, label %1935, label %1182

1182:                                             ; preds = %1179
  %1183 = invoke noundef i1 %1180(ptr noundef nonnull align 8 dereferenceable(32) %8, ptr noundef nonnull align 8 dereferenceable(32) %8, i32 noundef 3)
          to label %1935 unwind label %1184

1184:                                             ; preds = %1182
  %1185 = landingpad { ptr, i32 }
          catch ptr null
  %1186 = extractvalue { ptr, i32 } %1185, 0
  call void @__clang_call_terminate(ptr %1186) #21
  unreachable

1187:                                             ; preds = %194
  %1188 = landingpad { ptr, i32 }
          cleanup
  %1189 = load ptr, ptr %197, align 8, !tbaa !20
  %1190 = icmp eq ptr %1189, null
  br i1 %1190, label %1196, label %1191

1191:                                             ; preds = %1187
  %1192 = invoke noundef i1 %1189(ptr noundef nonnull align 8 dereferenceable(32) %11, ptr noundef nonnull align 8 dereferenceable(32) %11, i32 noundef 3)
          to label %1196 unwind label %1193

1193:                                             ; preds = %1191
  %1194 = landingpad { ptr, i32 }
          catch ptr null
  %1195 = extractvalue { ptr, i32 } %1194, 0
  call void @__clang_call_terminate(ptr %1195) #21
  unreachable

1196:                                             ; preds = %1187, %1191
  %1197 = load ptr, ptr %195, align 8, !tbaa !20
  %1198 = icmp eq ptr %1197, null
  br i1 %1198, label %1935, label %1199

1199:                                             ; preds = %1196
  %1200 = invoke noundef i1 %1197(ptr noundef nonnull align 8 dereferenceable(32) %10, ptr noundef nonnull align 8 dereferenceable(32) %10, i32 noundef 3)
          to label %1935 unwind label %1201

1201:                                             ; preds = %1199
  %1202 = landingpad { ptr, i32 }
          catch ptr null
  %1203 = extractvalue { ptr, i32 } %1202, 0
  call void @__clang_call_terminate(ptr %1203) #21
  unreachable

1204:                                             ; preds = %215
  %1205 = landingpad { ptr, i32 }
          cleanup
  %1206 = load ptr, ptr %218, align 8, !tbaa !20
  %1207 = icmp eq ptr %1206, null
  br i1 %1207, label %1213, label %1208

1208:                                             ; preds = %1204
  %1209 = invoke noundef i1 %1206(ptr noundef nonnull align 8 dereferenceable(32) %13, ptr noundef nonnull align 8 dereferenceable(32) %13, i32 noundef 3)
          to label %1213 unwind label %1210

1210:                                             ; preds = %1208
  %1211 = landingpad { ptr, i32 }
          catch ptr null
  %1212 = extractvalue { ptr, i32 } %1211, 0
  call void @__clang_call_terminate(ptr %1212) #21
  unreachable

1213:                                             ; preds = %1204, %1208
  %1214 = load ptr, ptr %216, align 8, !tbaa !20
  %1215 = icmp eq ptr %1214, null
  br i1 %1215, label %1935, label %1216

1216:                                             ; preds = %1213
  %1217 = invoke noundef i1 %1214(ptr noundef nonnull align 8 dereferenceable(32) %12, ptr noundef nonnull align 8 dereferenceable(32) %12, i32 noundef 3)
          to label %1935 unwind label %1218

1218:                                             ; preds = %1216
  %1219 = landingpad { ptr, i32 }
          catch ptr null
  %1220 = extractvalue { ptr, i32 } %1219, 0
  call void @__clang_call_terminate(ptr %1220) #21
  unreachable

1221:                                             ; preds = %236
  %1222 = landingpad { ptr, i32 }
          cleanup
  %1223 = load ptr, ptr %239, align 8, !tbaa !20
  %1224 = icmp eq ptr %1223, null
  br i1 %1224, label %1230, label %1225

1225:                                             ; preds = %1221
  %1226 = invoke noundef i1 %1223(ptr noundef nonnull align 8 dereferenceable(32) %15, ptr noundef nonnull align 8 dereferenceable(32) %15, i32 noundef 3)
          to label %1230 unwind label %1227

1227:                                             ; preds = %1225
  %1228 = landingpad { ptr, i32 }
          catch ptr null
  %1229 = extractvalue { ptr, i32 } %1228, 0
  call void @__clang_call_terminate(ptr %1229) #21
  unreachable

1230:                                             ; preds = %1221, %1225
  %1231 = load ptr, ptr %237, align 8, !tbaa !20
  %1232 = icmp eq ptr %1231, null
  br i1 %1232, label %1935, label %1233

1233:                                             ; preds = %1230
  %1234 = invoke noundef i1 %1231(ptr noundef nonnull align 8 dereferenceable(32) %14, ptr noundef nonnull align 8 dereferenceable(32) %14, i32 noundef 3)
          to label %1935 unwind label %1235

1235:                                             ; preds = %1233
  %1236 = landingpad { ptr, i32 }
          catch ptr null
  %1237 = extractvalue { ptr, i32 } %1236, 0
  call void @__clang_call_terminate(ptr %1237) #21
  unreachable

1238:                                             ; preds = %257
  %1239 = landingpad { ptr, i32 }
          cleanup
  %1240 = load ptr, ptr %260, align 8, !tbaa !20
  %1241 = icmp eq ptr %1240, null
  br i1 %1241, label %1247, label %1242

1242:                                             ; preds = %1238
  %1243 = invoke noundef i1 %1240(ptr noundef nonnull align 8 dereferenceable(32) %17, ptr noundef nonnull align 8 dereferenceable(32) %17, i32 noundef 3)
          to label %1247 unwind label %1244

1244:                                             ; preds = %1242
  %1245 = landingpad { ptr, i32 }
          catch ptr null
  %1246 = extractvalue { ptr, i32 } %1245, 0
  call void @__clang_call_terminate(ptr %1246) #21
  unreachable

1247:                                             ; preds = %1238, %1242
  %1248 = load ptr, ptr %258, align 8, !tbaa !20
  %1249 = icmp eq ptr %1248, null
  br i1 %1249, label %1935, label %1250

1250:                                             ; preds = %1247
  %1251 = invoke noundef i1 %1248(ptr noundef nonnull align 8 dereferenceable(32) %16, ptr noundef nonnull align 8 dereferenceable(32) %16, i32 noundef 3)
          to label %1935 unwind label %1252

1252:                                             ; preds = %1250
  %1253 = landingpad { ptr, i32 }
          catch ptr null
  %1254 = extractvalue { ptr, i32 } %1253, 0
  call void @__clang_call_terminate(ptr %1254) #21
  unreachable

1255:                                             ; preds = %278
  %1256 = landingpad { ptr, i32 }
          cleanup
  %1257 = load ptr, ptr %281, align 8, !tbaa !20
  %1258 = icmp eq ptr %1257, null
  br i1 %1258, label %1264, label %1259

1259:                                             ; preds = %1255
  %1260 = invoke noundef i1 %1257(ptr noundef nonnull align 8 dereferenceable(32) %19, ptr noundef nonnull align 8 dereferenceable(32) %19, i32 noundef 3)
          to label %1264 unwind label %1261

1261:                                             ; preds = %1259
  %1262 = landingpad { ptr, i32 }
          catch ptr null
  %1263 = extractvalue { ptr, i32 } %1262, 0
  call void @__clang_call_terminate(ptr %1263) #21
  unreachable

1264:                                             ; preds = %1255, %1259
  %1265 = load ptr, ptr %279, align 8, !tbaa !20
  %1266 = icmp eq ptr %1265, null
  br i1 %1266, label %1935, label %1267

1267:                                             ; preds = %1264
  %1268 = invoke noundef i1 %1265(ptr noundef nonnull align 8 dereferenceable(32) %18, ptr noundef nonnull align 8 dereferenceable(32) %18, i32 noundef 3)
          to label %1935 unwind label %1269

1269:                                             ; preds = %1267
  %1270 = landingpad { ptr, i32 }
          catch ptr null
  %1271 = extractvalue { ptr, i32 } %1270, 0
  call void @__clang_call_terminate(ptr %1271) #21
  unreachable

1272:                                             ; preds = %299
  %1273 = landingpad { ptr, i32 }
          cleanup
  %1274 = load ptr, ptr %302, align 8, !tbaa !20
  %1275 = icmp eq ptr %1274, null
  br i1 %1275, label %1281, label %1276

1276:                                             ; preds = %1272
  %1277 = invoke noundef i1 %1274(ptr noundef nonnull align 8 dereferenceable(32) %21, ptr noundef nonnull align 8 dereferenceable(32) %21, i32 noundef 3)
          to label %1281 unwind label %1278

1278:                                             ; preds = %1276
  %1279 = landingpad { ptr, i32 }
          catch ptr null
  %1280 = extractvalue { ptr, i32 } %1279, 0
  call void @__clang_call_terminate(ptr %1280) #21
  unreachable

1281:                                             ; preds = %1272, %1276
  %1282 = load ptr, ptr %300, align 8, !tbaa !20
  %1283 = icmp eq ptr %1282, null
  br i1 %1283, label %1935, label %1284

1284:                                             ; preds = %1281
  %1285 = invoke noundef i1 %1282(ptr noundef nonnull align 8 dereferenceable(32) %20, ptr noundef nonnull align 8 dereferenceable(32) %20, i32 noundef 3)
          to label %1935 unwind label %1286

1286:                                             ; preds = %1284
  %1287 = landingpad { ptr, i32 }
          catch ptr null
  %1288 = extractvalue { ptr, i32 } %1287, 0
  call void @__clang_call_terminate(ptr %1288) #21
  unreachable

1289:                                             ; preds = %320
  %1290 = landingpad { ptr, i32 }
          cleanup
  %1291 = load ptr, ptr %323, align 8, !tbaa !20
  %1292 = icmp eq ptr %1291, null
  br i1 %1292, label %1298, label %1293

1293:                                             ; preds = %1289
  %1294 = invoke noundef i1 %1291(ptr noundef nonnull align 8 dereferenceable(32) %23, ptr noundef nonnull align 8 dereferenceable(32) %23, i32 noundef 3)
          to label %1298 unwind label %1295

1295:                                             ; preds = %1293
  %1296 = landingpad { ptr, i32 }
          catch ptr null
  %1297 = extractvalue { ptr, i32 } %1296, 0
  call void @__clang_call_terminate(ptr %1297) #21
  unreachable

1298:                                             ; preds = %1289, %1293
  %1299 = load ptr, ptr %321, align 8, !tbaa !20
  %1300 = icmp eq ptr %1299, null
  br i1 %1300, label %1935, label %1301

1301:                                             ; preds = %1298
  %1302 = invoke noundef i1 %1299(ptr noundef nonnull align 8 dereferenceable(32) %22, ptr noundef nonnull align 8 dereferenceable(32) %22, i32 noundef 3)
          to label %1935 unwind label %1303

1303:                                             ; preds = %1301
  %1304 = landingpad { ptr, i32 }
          catch ptr null
  %1305 = extractvalue { ptr, i32 } %1304, 0
  call void @__clang_call_terminate(ptr %1305) #21
  unreachable

1306:                                             ; preds = %341
  %1307 = landingpad { ptr, i32 }
          cleanup
  %1308 = load ptr, ptr %344, align 8, !tbaa !20
  %1309 = icmp eq ptr %1308, null
  br i1 %1309, label %1315, label %1310

1310:                                             ; preds = %1306
  %1311 = invoke noundef i1 %1308(ptr noundef nonnull align 8 dereferenceable(32) %25, ptr noundef nonnull align 8 dereferenceable(32) %25, i32 noundef 3)
          to label %1315 unwind label %1312

1312:                                             ; preds = %1310
  %1313 = landingpad { ptr, i32 }
          catch ptr null
  %1314 = extractvalue { ptr, i32 } %1313, 0
  call void @__clang_call_terminate(ptr %1314) #21
  unreachable

1315:                                             ; preds = %1306, %1310
  %1316 = load ptr, ptr %342, align 8, !tbaa !20
  %1317 = icmp eq ptr %1316, null
  br i1 %1317, label %1935, label %1318

1318:                                             ; preds = %1315
  %1319 = invoke noundef i1 %1316(ptr noundef nonnull align 8 dereferenceable(32) %24, ptr noundef nonnull align 8 dereferenceable(32) %24, i32 noundef 3)
          to label %1935 unwind label %1320

1320:                                             ; preds = %1318
  %1321 = landingpad { ptr, i32 }
          catch ptr null
  %1322 = extractvalue { ptr, i32 } %1321, 0
  call void @__clang_call_terminate(ptr %1322) #21
  unreachable

1323:                                             ; preds = %362
  %1324 = landingpad { ptr, i32 }
          cleanup
  %1325 = load ptr, ptr %365, align 8, !tbaa !20
  %1326 = icmp eq ptr %1325, null
  br i1 %1326, label %1332, label %1327

1327:                                             ; preds = %1323
  %1328 = invoke noundef i1 %1325(ptr noundef nonnull align 8 dereferenceable(32) %27, ptr noundef nonnull align 8 dereferenceable(32) %27, i32 noundef 3)
          to label %1332 unwind label %1329

1329:                                             ; preds = %1327
  %1330 = landingpad { ptr, i32 }
          catch ptr null
  %1331 = extractvalue { ptr, i32 } %1330, 0
  call void @__clang_call_terminate(ptr %1331) #21
  unreachable

1332:                                             ; preds = %1323, %1327
  %1333 = load ptr, ptr %363, align 8, !tbaa !20
  %1334 = icmp eq ptr %1333, null
  br i1 %1334, label %1935, label %1335

1335:                                             ; preds = %1332
  %1336 = invoke noundef i1 %1333(ptr noundef nonnull align 8 dereferenceable(32) %26, ptr noundef nonnull align 8 dereferenceable(32) %26, i32 noundef 3)
          to label %1935 unwind label %1337

1337:                                             ; preds = %1335
  %1338 = landingpad { ptr, i32 }
          catch ptr null
  %1339 = extractvalue { ptr, i32 } %1338, 0
  call void @__clang_call_terminate(ptr %1339) #21
  unreachable

1340:                                             ; preds = %383
  %1341 = landingpad { ptr, i32 }
          cleanup
  %1342 = load ptr, ptr %386, align 8, !tbaa !20
  %1343 = icmp eq ptr %1342, null
  br i1 %1343, label %1349, label %1344

1344:                                             ; preds = %1340
  %1345 = invoke noundef i1 %1342(ptr noundef nonnull align 8 dereferenceable(32) %29, ptr noundef nonnull align 8 dereferenceable(32) %29, i32 noundef 3)
          to label %1349 unwind label %1346

1346:                                             ; preds = %1344
  %1347 = landingpad { ptr, i32 }
          catch ptr null
  %1348 = extractvalue { ptr, i32 } %1347, 0
  call void @__clang_call_terminate(ptr %1348) #21
  unreachable

1349:                                             ; preds = %1340, %1344
  %1350 = load ptr, ptr %384, align 8, !tbaa !20
  %1351 = icmp eq ptr %1350, null
  br i1 %1351, label %1935, label %1352

1352:                                             ; preds = %1349
  %1353 = invoke noundef i1 %1350(ptr noundef nonnull align 8 dereferenceable(32) %28, ptr noundef nonnull align 8 dereferenceable(32) %28, i32 noundef 3)
          to label %1935 unwind label %1354

1354:                                             ; preds = %1352
  %1355 = landingpad { ptr, i32 }
          catch ptr null
  %1356 = extractvalue { ptr, i32 } %1355, 0
  call void @__clang_call_terminate(ptr %1356) #21
  unreachable

1357:                                             ; preds = %404
  %1358 = landingpad { ptr, i32 }
          cleanup
  %1359 = load ptr, ptr %407, align 8, !tbaa !20
  %1360 = icmp eq ptr %1359, null
  br i1 %1360, label %1366, label %1361

1361:                                             ; preds = %1357
  %1362 = invoke noundef i1 %1359(ptr noundef nonnull align 8 dereferenceable(32) %31, ptr noundef nonnull align 8 dereferenceable(32) %31, i32 noundef 3)
          to label %1366 unwind label %1363

1363:                                             ; preds = %1361
  %1364 = landingpad { ptr, i32 }
          catch ptr null
  %1365 = extractvalue { ptr, i32 } %1364, 0
  call void @__clang_call_terminate(ptr %1365) #21
  unreachable

1366:                                             ; preds = %1357, %1361
  %1367 = load ptr, ptr %405, align 8, !tbaa !20
  %1368 = icmp eq ptr %1367, null
  br i1 %1368, label %1935, label %1369

1369:                                             ; preds = %1366
  %1370 = invoke noundef i1 %1367(ptr noundef nonnull align 8 dereferenceable(32) %30, ptr noundef nonnull align 8 dereferenceable(32) %30, i32 noundef 3)
          to label %1935 unwind label %1371

1371:                                             ; preds = %1369
  %1372 = landingpad { ptr, i32 }
          catch ptr null
  %1373 = extractvalue { ptr, i32 } %1372, 0
  call void @__clang_call_terminate(ptr %1373) #21
  unreachable

1374:                                             ; preds = %425
  %1375 = landingpad { ptr, i32 }
          cleanup
  %1376 = load ptr, ptr %428, align 8, !tbaa !20
  %1377 = icmp eq ptr %1376, null
  br i1 %1377, label %1383, label %1378

1378:                                             ; preds = %1374
  %1379 = invoke noundef i1 %1376(ptr noundef nonnull align 8 dereferenceable(32) %33, ptr noundef nonnull align 8 dereferenceable(32) %33, i32 noundef 3)
          to label %1383 unwind label %1380

1380:                                             ; preds = %1378
  %1381 = landingpad { ptr, i32 }
          catch ptr null
  %1382 = extractvalue { ptr, i32 } %1381, 0
  call void @__clang_call_terminate(ptr %1382) #21
  unreachable

1383:                                             ; preds = %1374, %1378
  %1384 = load ptr, ptr %426, align 8, !tbaa !20
  %1385 = icmp eq ptr %1384, null
  br i1 %1385, label %1935, label %1386

1386:                                             ; preds = %1383
  %1387 = invoke noundef i1 %1384(ptr noundef nonnull align 8 dereferenceable(32) %32, ptr noundef nonnull align 8 dereferenceable(32) %32, i32 noundef 3)
          to label %1935 unwind label %1388

1388:                                             ; preds = %1386
  %1389 = landingpad { ptr, i32 }
          catch ptr null
  %1390 = extractvalue { ptr, i32 } %1389, 0
  call void @__clang_call_terminate(ptr %1390) #21
  unreachable

1391:                                             ; preds = %446
  %1392 = landingpad { ptr, i32 }
          cleanup
  %1393 = load ptr, ptr %449, align 8, !tbaa !20
  %1394 = icmp eq ptr %1393, null
  br i1 %1394, label %1400, label %1395

1395:                                             ; preds = %1391
  %1396 = invoke noundef i1 %1393(ptr noundef nonnull align 8 dereferenceable(32) %35, ptr noundef nonnull align 8 dereferenceable(32) %35, i32 noundef 3)
          to label %1400 unwind label %1397

1397:                                             ; preds = %1395
  %1398 = landingpad { ptr, i32 }
          catch ptr null
  %1399 = extractvalue { ptr, i32 } %1398, 0
  call void @__clang_call_terminate(ptr %1399) #21
  unreachable

1400:                                             ; preds = %1391, %1395
  %1401 = load ptr, ptr %447, align 8, !tbaa !20
  %1402 = icmp eq ptr %1401, null
  br i1 %1402, label %1935, label %1403

1403:                                             ; preds = %1400
  %1404 = invoke noundef i1 %1401(ptr noundef nonnull align 8 dereferenceable(32) %34, ptr noundef nonnull align 8 dereferenceable(32) %34, i32 noundef 3)
          to label %1935 unwind label %1405

1405:                                             ; preds = %1403
  %1406 = landingpad { ptr, i32 }
          catch ptr null
  %1407 = extractvalue { ptr, i32 } %1406, 0
  call void @__clang_call_terminate(ptr %1407) #21
  unreachable

1408:                                             ; preds = %467
  %1409 = landingpad { ptr, i32 }
          cleanup
  %1410 = load ptr, ptr %470, align 8, !tbaa !20
  %1411 = icmp eq ptr %1410, null
  br i1 %1411, label %1417, label %1412

1412:                                             ; preds = %1408
  %1413 = invoke noundef i1 %1410(ptr noundef nonnull align 8 dereferenceable(32) %37, ptr noundef nonnull align 8 dereferenceable(32) %37, i32 noundef 3)
          to label %1417 unwind label %1414

1414:                                             ; preds = %1412
  %1415 = landingpad { ptr, i32 }
          catch ptr null
  %1416 = extractvalue { ptr, i32 } %1415, 0
  call void @__clang_call_terminate(ptr %1416) #21
  unreachable

1417:                                             ; preds = %1408, %1412
  %1418 = load ptr, ptr %468, align 8, !tbaa !20
  %1419 = icmp eq ptr %1418, null
  br i1 %1419, label %1935, label %1420

1420:                                             ; preds = %1417
  %1421 = invoke noundef i1 %1418(ptr noundef nonnull align 8 dereferenceable(32) %36, ptr noundef nonnull align 8 dereferenceable(32) %36, i32 noundef 3)
          to label %1935 unwind label %1422

1422:                                             ; preds = %1420
  %1423 = landingpad { ptr, i32 }
          catch ptr null
  %1424 = extractvalue { ptr, i32 } %1423, 0
  call void @__clang_call_terminate(ptr %1424) #21
  unreachable

1425:                                             ; preds = %488
  %1426 = landingpad { ptr, i32 }
          cleanup
  %1427 = load ptr, ptr %491, align 8, !tbaa !20
  %1428 = icmp eq ptr %1427, null
  br i1 %1428, label %1434, label %1429

1429:                                             ; preds = %1425
  %1430 = invoke noundef i1 %1427(ptr noundef nonnull align 8 dereferenceable(32) %39, ptr noundef nonnull align 8 dereferenceable(32) %39, i32 noundef 3)
          to label %1434 unwind label %1431

1431:                                             ; preds = %1429
  %1432 = landingpad { ptr, i32 }
          catch ptr null
  %1433 = extractvalue { ptr, i32 } %1432, 0
  call void @__clang_call_terminate(ptr %1433) #21
  unreachable

1434:                                             ; preds = %1425, %1429
  %1435 = load ptr, ptr %489, align 8, !tbaa !20
  %1436 = icmp eq ptr %1435, null
  br i1 %1436, label %1935, label %1437

1437:                                             ; preds = %1434
  %1438 = invoke noundef i1 %1435(ptr noundef nonnull align 8 dereferenceable(32) %38, ptr noundef nonnull align 8 dereferenceable(32) %38, i32 noundef 3)
          to label %1935 unwind label %1439

1439:                                             ; preds = %1437
  %1440 = landingpad { ptr, i32 }
          catch ptr null
  %1441 = extractvalue { ptr, i32 } %1440, 0
  call void @__clang_call_terminate(ptr %1441) #21
  unreachable

1442:                                             ; preds = %509
  %1443 = landingpad { ptr, i32 }
          cleanup
  %1444 = load ptr, ptr %512, align 8, !tbaa !20
  %1445 = icmp eq ptr %1444, null
  br i1 %1445, label %1451, label %1446

1446:                                             ; preds = %1442
  %1447 = invoke noundef i1 %1444(ptr noundef nonnull align 8 dereferenceable(32) %41, ptr noundef nonnull align 8 dereferenceable(32) %41, i32 noundef 3)
          to label %1451 unwind label %1448

1448:                                             ; preds = %1446
  %1449 = landingpad { ptr, i32 }
          catch ptr null
  %1450 = extractvalue { ptr, i32 } %1449, 0
  call void @__clang_call_terminate(ptr %1450) #21
  unreachable

1451:                                             ; preds = %1442, %1446
  %1452 = load ptr, ptr %510, align 8, !tbaa !20
  %1453 = icmp eq ptr %1452, null
  br i1 %1453, label %1935, label %1454

1454:                                             ; preds = %1451
  %1455 = invoke noundef i1 %1452(ptr noundef nonnull align 8 dereferenceable(32) %40, ptr noundef nonnull align 8 dereferenceable(32) %40, i32 noundef 3)
          to label %1935 unwind label %1456

1456:                                             ; preds = %1454
  %1457 = landingpad { ptr, i32 }
          catch ptr null
  %1458 = extractvalue { ptr, i32 } %1457, 0
  call void @__clang_call_terminate(ptr %1458) #21
  unreachable

1459:                                             ; preds = %530
  %1460 = landingpad { ptr, i32 }
          cleanup
  %1461 = load ptr, ptr %533, align 8, !tbaa !20
  %1462 = icmp eq ptr %1461, null
  br i1 %1462, label %1468, label %1463

1463:                                             ; preds = %1459
  %1464 = invoke noundef i1 %1461(ptr noundef nonnull align 8 dereferenceable(32) %43, ptr noundef nonnull align 8 dereferenceable(32) %43, i32 noundef 3)
          to label %1468 unwind label %1465

1465:                                             ; preds = %1463
  %1466 = landingpad { ptr, i32 }
          catch ptr null
  %1467 = extractvalue { ptr, i32 } %1466, 0
  call void @__clang_call_terminate(ptr %1467) #21
  unreachable

1468:                                             ; preds = %1459, %1463
  %1469 = load ptr, ptr %531, align 8, !tbaa !20
  %1470 = icmp eq ptr %1469, null
  br i1 %1470, label %1935, label %1471

1471:                                             ; preds = %1468
  %1472 = invoke noundef i1 %1469(ptr noundef nonnull align 8 dereferenceable(32) %42, ptr noundef nonnull align 8 dereferenceable(32) %42, i32 noundef 3)
          to label %1935 unwind label %1473

1473:                                             ; preds = %1471
  %1474 = landingpad { ptr, i32 }
          catch ptr null
  %1475 = extractvalue { ptr, i32 } %1474, 0
  call void @__clang_call_terminate(ptr %1475) #21
  unreachable

1476:                                             ; preds = %551
  %1477 = landingpad { ptr, i32 }
          cleanup
  %1478 = load ptr, ptr %554, align 8, !tbaa !20
  %1479 = icmp eq ptr %1478, null
  br i1 %1479, label %1485, label %1480

1480:                                             ; preds = %1476
  %1481 = invoke noundef i1 %1478(ptr noundef nonnull align 8 dereferenceable(32) %45, ptr noundef nonnull align 8 dereferenceable(32) %45, i32 noundef 3)
          to label %1485 unwind label %1482

1482:                                             ; preds = %1480
  %1483 = landingpad { ptr, i32 }
          catch ptr null
  %1484 = extractvalue { ptr, i32 } %1483, 0
  call void @__clang_call_terminate(ptr %1484) #21
  unreachable

1485:                                             ; preds = %1476, %1480
  %1486 = load ptr, ptr %552, align 8, !tbaa !20
  %1487 = icmp eq ptr %1486, null
  br i1 %1487, label %1935, label %1488

1488:                                             ; preds = %1485
  %1489 = invoke noundef i1 %1486(ptr noundef nonnull align 8 dereferenceable(32) %44, ptr noundef nonnull align 8 dereferenceable(32) %44, i32 noundef 3)
          to label %1935 unwind label %1490

1490:                                             ; preds = %1488
  %1491 = landingpad { ptr, i32 }
          catch ptr null
  %1492 = extractvalue { ptr, i32 } %1491, 0
  call void @__clang_call_terminate(ptr %1492) #21
  unreachable

1493:                                             ; preds = %572
  %1494 = landingpad { ptr, i32 }
          cleanup
  %1495 = load ptr, ptr %575, align 8, !tbaa !20
  %1496 = icmp eq ptr %1495, null
  br i1 %1496, label %1502, label %1497

1497:                                             ; preds = %1493
  %1498 = invoke noundef i1 %1495(ptr noundef nonnull align 8 dereferenceable(32) %47, ptr noundef nonnull align 8 dereferenceable(32) %47, i32 noundef 3)
          to label %1502 unwind label %1499

1499:                                             ; preds = %1497
  %1500 = landingpad { ptr, i32 }
          catch ptr null
  %1501 = extractvalue { ptr, i32 } %1500, 0
  call void @__clang_call_terminate(ptr %1501) #21
  unreachable

1502:                                             ; preds = %1493, %1497
  %1503 = load ptr, ptr %573, align 8, !tbaa !20
  %1504 = icmp eq ptr %1503, null
  br i1 %1504, label %1935, label %1505

1505:                                             ; preds = %1502
  %1506 = invoke noundef i1 %1503(ptr noundef nonnull align 8 dereferenceable(32) %46, ptr noundef nonnull align 8 dereferenceable(32) %46, i32 noundef 3)
          to label %1935 unwind label %1507

1507:                                             ; preds = %1505
  %1508 = landingpad { ptr, i32 }
          catch ptr null
  %1509 = extractvalue { ptr, i32 } %1508, 0
  call void @__clang_call_terminate(ptr %1509) #21
  unreachable

1510:                                             ; preds = %593
  %1511 = landingpad { ptr, i32 }
          cleanup
  %1512 = load ptr, ptr %596, align 8, !tbaa !20
  %1513 = icmp eq ptr %1512, null
  br i1 %1513, label %1519, label %1514

1514:                                             ; preds = %1510
  %1515 = invoke noundef i1 %1512(ptr noundef nonnull align 8 dereferenceable(32) %49, ptr noundef nonnull align 8 dereferenceable(32) %49, i32 noundef 3)
          to label %1519 unwind label %1516

1516:                                             ; preds = %1514
  %1517 = landingpad { ptr, i32 }
          catch ptr null
  %1518 = extractvalue { ptr, i32 } %1517, 0
  call void @__clang_call_terminate(ptr %1518) #21
  unreachable

1519:                                             ; preds = %1510, %1514
  %1520 = load ptr, ptr %594, align 8, !tbaa !20
  %1521 = icmp eq ptr %1520, null
  br i1 %1521, label %1935, label %1522

1522:                                             ; preds = %1519
  %1523 = invoke noundef i1 %1520(ptr noundef nonnull align 8 dereferenceable(32) %48, ptr noundef nonnull align 8 dereferenceable(32) %48, i32 noundef 3)
          to label %1935 unwind label %1524

1524:                                             ; preds = %1522
  %1525 = landingpad { ptr, i32 }
          catch ptr null
  %1526 = extractvalue { ptr, i32 } %1525, 0
  call void @__clang_call_terminate(ptr %1526) #21
  unreachable

1527:                                             ; preds = %614
  %1528 = landingpad { ptr, i32 }
          cleanup
  %1529 = load ptr, ptr %617, align 8, !tbaa !20
  %1530 = icmp eq ptr %1529, null
  br i1 %1530, label %1536, label %1531

1531:                                             ; preds = %1527
  %1532 = invoke noundef i1 %1529(ptr noundef nonnull align 8 dereferenceable(32) %51, ptr noundef nonnull align 8 dereferenceable(32) %51, i32 noundef 3)
          to label %1536 unwind label %1533

1533:                                             ; preds = %1531
  %1534 = landingpad { ptr, i32 }
          catch ptr null
  %1535 = extractvalue { ptr, i32 } %1534, 0
  call void @__clang_call_terminate(ptr %1535) #21
  unreachable

1536:                                             ; preds = %1527, %1531
  %1537 = load ptr, ptr %615, align 8, !tbaa !20
  %1538 = icmp eq ptr %1537, null
  br i1 %1538, label %1935, label %1539

1539:                                             ; preds = %1536
  %1540 = invoke noundef i1 %1537(ptr noundef nonnull align 8 dereferenceable(32) %50, ptr noundef nonnull align 8 dereferenceable(32) %50, i32 noundef 3)
          to label %1935 unwind label %1541

1541:                                             ; preds = %1539
  %1542 = landingpad { ptr, i32 }
          catch ptr null
  %1543 = extractvalue { ptr, i32 } %1542, 0
  call void @__clang_call_terminate(ptr %1543) #21
  unreachable

1544:                                             ; preds = %635
  %1545 = landingpad { ptr, i32 }
          cleanup
  %1546 = load ptr, ptr %638, align 8, !tbaa !20
  %1547 = icmp eq ptr %1546, null
  br i1 %1547, label %1553, label %1548

1548:                                             ; preds = %1544
  %1549 = invoke noundef i1 %1546(ptr noundef nonnull align 8 dereferenceable(32) %53, ptr noundef nonnull align 8 dereferenceable(32) %53, i32 noundef 3)
          to label %1553 unwind label %1550

1550:                                             ; preds = %1548
  %1551 = landingpad { ptr, i32 }
          catch ptr null
  %1552 = extractvalue { ptr, i32 } %1551, 0
  call void @__clang_call_terminate(ptr %1552) #21
  unreachable

1553:                                             ; preds = %1544, %1548
  %1554 = load ptr, ptr %636, align 8, !tbaa !20
  %1555 = icmp eq ptr %1554, null
  br i1 %1555, label %1935, label %1556

1556:                                             ; preds = %1553
  %1557 = invoke noundef i1 %1554(ptr noundef nonnull align 8 dereferenceable(32) %52, ptr noundef nonnull align 8 dereferenceable(32) %52, i32 noundef 3)
          to label %1935 unwind label %1558

1558:                                             ; preds = %1556
  %1559 = landingpad { ptr, i32 }
          catch ptr null
  %1560 = extractvalue { ptr, i32 } %1559, 0
  call void @__clang_call_terminate(ptr %1560) #21
  unreachable

1561:                                             ; preds = %656
  %1562 = landingpad { ptr, i32 }
          cleanup
  %1563 = load ptr, ptr %659, align 8, !tbaa !20
  %1564 = icmp eq ptr %1563, null
  br i1 %1564, label %1570, label %1565

1565:                                             ; preds = %1561
  %1566 = invoke noundef i1 %1563(ptr noundef nonnull align 8 dereferenceable(32) %55, ptr noundef nonnull align 8 dereferenceable(32) %55, i32 noundef 3)
          to label %1570 unwind label %1567

1567:                                             ; preds = %1565
  %1568 = landingpad { ptr, i32 }
          catch ptr null
  %1569 = extractvalue { ptr, i32 } %1568, 0
  call void @__clang_call_terminate(ptr %1569) #21
  unreachable

1570:                                             ; preds = %1561, %1565
  %1571 = load ptr, ptr %657, align 8, !tbaa !20
  %1572 = icmp eq ptr %1571, null
  br i1 %1572, label %1935, label %1573

1573:                                             ; preds = %1570
  %1574 = invoke noundef i1 %1571(ptr noundef nonnull align 8 dereferenceable(32) %54, ptr noundef nonnull align 8 dereferenceable(32) %54, i32 noundef 3)
          to label %1935 unwind label %1575

1575:                                             ; preds = %1573
  %1576 = landingpad { ptr, i32 }
          catch ptr null
  %1577 = extractvalue { ptr, i32 } %1576, 0
  call void @__clang_call_terminate(ptr %1577) #21
  unreachable

1578:                                             ; preds = %677
  %1579 = landingpad { ptr, i32 }
          cleanup
  %1580 = load ptr, ptr %680, align 8, !tbaa !20
  %1581 = icmp eq ptr %1580, null
  br i1 %1581, label %1587, label %1582

1582:                                             ; preds = %1578
  %1583 = invoke noundef i1 %1580(ptr noundef nonnull align 8 dereferenceable(32) %57, ptr noundef nonnull align 8 dereferenceable(32) %57, i32 noundef 3)
          to label %1587 unwind label %1584

1584:                                             ; preds = %1582
  %1585 = landingpad { ptr, i32 }
          catch ptr null
  %1586 = extractvalue { ptr, i32 } %1585, 0
  call void @__clang_call_terminate(ptr %1586) #21
  unreachable

1587:                                             ; preds = %1578, %1582
  %1588 = load ptr, ptr %678, align 8, !tbaa !20
  %1589 = icmp eq ptr %1588, null
  br i1 %1589, label %1935, label %1590

1590:                                             ; preds = %1587
  %1591 = invoke noundef i1 %1588(ptr noundef nonnull align 8 dereferenceable(32) %56, ptr noundef nonnull align 8 dereferenceable(32) %56, i32 noundef 3)
          to label %1935 unwind label %1592

1592:                                             ; preds = %1590
  %1593 = landingpad { ptr, i32 }
          catch ptr null
  %1594 = extractvalue { ptr, i32 } %1593, 0
  call void @__clang_call_terminate(ptr %1594) #21
  unreachable

1595:                                             ; preds = %698
  %1596 = landingpad { ptr, i32 }
          cleanup
  %1597 = load ptr, ptr %701, align 8, !tbaa !20
  %1598 = icmp eq ptr %1597, null
  br i1 %1598, label %1604, label %1599

1599:                                             ; preds = %1595
  %1600 = invoke noundef i1 %1597(ptr noundef nonnull align 8 dereferenceable(32) %59, ptr noundef nonnull align 8 dereferenceable(32) %59, i32 noundef 3)
          to label %1604 unwind label %1601

1601:                                             ; preds = %1599
  %1602 = landingpad { ptr, i32 }
          catch ptr null
  %1603 = extractvalue { ptr, i32 } %1602, 0
  call void @__clang_call_terminate(ptr %1603) #21
  unreachable

1604:                                             ; preds = %1595, %1599
  %1605 = load ptr, ptr %699, align 8, !tbaa !20
  %1606 = icmp eq ptr %1605, null
  br i1 %1606, label %1935, label %1607

1607:                                             ; preds = %1604
  %1608 = invoke noundef i1 %1605(ptr noundef nonnull align 8 dereferenceable(32) %58, ptr noundef nonnull align 8 dereferenceable(32) %58, i32 noundef 3)
          to label %1935 unwind label %1609

1609:                                             ; preds = %1607
  %1610 = landingpad { ptr, i32 }
          catch ptr null
  %1611 = extractvalue { ptr, i32 } %1610, 0
  call void @__clang_call_terminate(ptr %1611) #21
  unreachable

1612:                                             ; preds = %719
  %1613 = landingpad { ptr, i32 }
          cleanup
  %1614 = load ptr, ptr %722, align 8, !tbaa !20
  %1615 = icmp eq ptr %1614, null
  br i1 %1615, label %1621, label %1616

1616:                                             ; preds = %1612
  %1617 = invoke noundef i1 %1614(ptr noundef nonnull align 8 dereferenceable(32) %61, ptr noundef nonnull align 8 dereferenceable(32) %61, i32 noundef 3)
          to label %1621 unwind label %1618

1618:                                             ; preds = %1616
  %1619 = landingpad { ptr, i32 }
          catch ptr null
  %1620 = extractvalue { ptr, i32 } %1619, 0
  call void @__clang_call_terminate(ptr %1620) #21
  unreachable

1621:                                             ; preds = %1612, %1616
  %1622 = load ptr, ptr %720, align 8, !tbaa !20
  %1623 = icmp eq ptr %1622, null
  br i1 %1623, label %1935, label %1624

1624:                                             ; preds = %1621
  %1625 = invoke noundef i1 %1622(ptr noundef nonnull align 8 dereferenceable(32) %60, ptr noundef nonnull align 8 dereferenceable(32) %60, i32 noundef 3)
          to label %1935 unwind label %1626

1626:                                             ; preds = %1624
  %1627 = landingpad { ptr, i32 }
          catch ptr null
  %1628 = extractvalue { ptr, i32 } %1627, 0
  call void @__clang_call_terminate(ptr %1628) #21
  unreachable

1629:                                             ; preds = %740
  %1630 = landingpad { ptr, i32 }
          cleanup
  %1631 = load ptr, ptr %743, align 8, !tbaa !20
  %1632 = icmp eq ptr %1631, null
  br i1 %1632, label %1638, label %1633

1633:                                             ; preds = %1629
  %1634 = invoke noundef i1 %1631(ptr noundef nonnull align 8 dereferenceable(32) %63, ptr noundef nonnull align 8 dereferenceable(32) %63, i32 noundef 3)
          to label %1638 unwind label %1635

1635:                                             ; preds = %1633
  %1636 = landingpad { ptr, i32 }
          catch ptr null
  %1637 = extractvalue { ptr, i32 } %1636, 0
  call void @__clang_call_terminate(ptr %1637) #21
  unreachable

1638:                                             ; preds = %1629, %1633
  %1639 = load ptr, ptr %741, align 8, !tbaa !20
  %1640 = icmp eq ptr %1639, null
  br i1 %1640, label %1935, label %1641

1641:                                             ; preds = %1638
  %1642 = invoke noundef i1 %1639(ptr noundef nonnull align 8 dereferenceable(32) %62, ptr noundef nonnull align 8 dereferenceable(32) %62, i32 noundef 3)
          to label %1935 unwind label %1643

1643:                                             ; preds = %1641
  %1644 = landingpad { ptr, i32 }
          catch ptr null
  %1645 = extractvalue { ptr, i32 } %1644, 0
  call void @__clang_call_terminate(ptr %1645) #21
  unreachable

1646:                                             ; preds = %761
  %1647 = landingpad { ptr, i32 }
          cleanup
  %1648 = load ptr, ptr %764, align 8, !tbaa !20
  %1649 = icmp eq ptr %1648, null
  br i1 %1649, label %1655, label %1650

1650:                                             ; preds = %1646
  %1651 = invoke noundef i1 %1648(ptr noundef nonnull align 8 dereferenceable(32) %65, ptr noundef nonnull align 8 dereferenceable(32) %65, i32 noundef 3)
          to label %1655 unwind label %1652

1652:                                             ; preds = %1650
  %1653 = landingpad { ptr, i32 }
          catch ptr null
  %1654 = extractvalue { ptr, i32 } %1653, 0
  call void @__clang_call_terminate(ptr %1654) #21
  unreachable

1655:                                             ; preds = %1646, %1650
  %1656 = load ptr, ptr %762, align 8, !tbaa !20
  %1657 = icmp eq ptr %1656, null
  br i1 %1657, label %1935, label %1658

1658:                                             ; preds = %1655
  %1659 = invoke noundef i1 %1656(ptr noundef nonnull align 8 dereferenceable(32) %64, ptr noundef nonnull align 8 dereferenceable(32) %64, i32 noundef 3)
          to label %1935 unwind label %1660

1660:                                             ; preds = %1658
  %1661 = landingpad { ptr, i32 }
          catch ptr null
  %1662 = extractvalue { ptr, i32 } %1661, 0
  call void @__clang_call_terminate(ptr %1662) #21
  unreachable

1663:                                             ; preds = %782
  %1664 = landingpad { ptr, i32 }
          cleanup
  %1665 = load ptr, ptr %785, align 8, !tbaa !20
  %1666 = icmp eq ptr %1665, null
  br i1 %1666, label %1672, label %1667

1667:                                             ; preds = %1663
  %1668 = invoke noundef i1 %1665(ptr noundef nonnull align 8 dereferenceable(32) %67, ptr noundef nonnull align 8 dereferenceable(32) %67, i32 noundef 3)
          to label %1672 unwind label %1669

1669:                                             ; preds = %1667
  %1670 = landingpad { ptr, i32 }
          catch ptr null
  %1671 = extractvalue { ptr, i32 } %1670, 0
  call void @__clang_call_terminate(ptr %1671) #21
  unreachable

1672:                                             ; preds = %1663, %1667
  %1673 = load ptr, ptr %783, align 8, !tbaa !20
  %1674 = icmp eq ptr %1673, null
  br i1 %1674, label %1935, label %1675

1675:                                             ; preds = %1672
  %1676 = invoke noundef i1 %1673(ptr noundef nonnull align 8 dereferenceable(32) %66, ptr noundef nonnull align 8 dereferenceable(32) %66, i32 noundef 3)
          to label %1935 unwind label %1677

1677:                                             ; preds = %1675
  %1678 = landingpad { ptr, i32 }
          catch ptr null
  %1679 = extractvalue { ptr, i32 } %1678, 0
  call void @__clang_call_terminate(ptr %1679) #21
  unreachable

1680:                                             ; preds = %803
  %1681 = landingpad { ptr, i32 }
          cleanup
  %1682 = load ptr, ptr %806, align 8, !tbaa !20
  %1683 = icmp eq ptr %1682, null
  br i1 %1683, label %1689, label %1684

1684:                                             ; preds = %1680
  %1685 = invoke noundef i1 %1682(ptr noundef nonnull align 8 dereferenceable(32) %69, ptr noundef nonnull align 8 dereferenceable(32) %69, i32 noundef 3)
          to label %1689 unwind label %1686

1686:                                             ; preds = %1684
  %1687 = landingpad { ptr, i32 }
          catch ptr null
  %1688 = extractvalue { ptr, i32 } %1687, 0
  call void @__clang_call_terminate(ptr %1688) #21
  unreachable

1689:                                             ; preds = %1680, %1684
  %1690 = load ptr, ptr %804, align 8, !tbaa !20
  %1691 = icmp eq ptr %1690, null
  br i1 %1691, label %1935, label %1692

1692:                                             ; preds = %1689
  %1693 = invoke noundef i1 %1690(ptr noundef nonnull align 8 dereferenceable(32) %68, ptr noundef nonnull align 8 dereferenceable(32) %68, i32 noundef 3)
          to label %1935 unwind label %1694

1694:                                             ; preds = %1692
  %1695 = landingpad { ptr, i32 }
          catch ptr null
  %1696 = extractvalue { ptr, i32 } %1695, 0
  call void @__clang_call_terminate(ptr %1696) #21
  unreachable

1697:                                             ; preds = %824
  %1698 = landingpad { ptr, i32 }
          cleanup
  %1699 = load ptr, ptr %827, align 8, !tbaa !20
  %1700 = icmp eq ptr %1699, null
  br i1 %1700, label %1706, label %1701

1701:                                             ; preds = %1697
  %1702 = invoke noundef i1 %1699(ptr noundef nonnull align 8 dereferenceable(32) %71, ptr noundef nonnull align 8 dereferenceable(32) %71, i32 noundef 3)
          to label %1706 unwind label %1703

1703:                                             ; preds = %1701
  %1704 = landingpad { ptr, i32 }
          catch ptr null
  %1705 = extractvalue { ptr, i32 } %1704, 0
  call void @__clang_call_terminate(ptr %1705) #21
  unreachable

1706:                                             ; preds = %1697, %1701
  %1707 = load ptr, ptr %825, align 8, !tbaa !20
  %1708 = icmp eq ptr %1707, null
  br i1 %1708, label %1935, label %1709

1709:                                             ; preds = %1706
  %1710 = invoke noundef i1 %1707(ptr noundef nonnull align 8 dereferenceable(32) %70, ptr noundef nonnull align 8 dereferenceable(32) %70, i32 noundef 3)
          to label %1935 unwind label %1711

1711:                                             ; preds = %1709
  %1712 = landingpad { ptr, i32 }
          catch ptr null
  %1713 = extractvalue { ptr, i32 } %1712, 0
  call void @__clang_call_terminate(ptr %1713) #21
  unreachable

1714:                                             ; preds = %845
  %1715 = landingpad { ptr, i32 }
          cleanup
  %1716 = load ptr, ptr %848, align 8, !tbaa !20
  %1717 = icmp eq ptr %1716, null
  br i1 %1717, label %1723, label %1718

1718:                                             ; preds = %1714
  %1719 = invoke noundef i1 %1716(ptr noundef nonnull align 8 dereferenceable(32) %73, ptr noundef nonnull align 8 dereferenceable(32) %73, i32 noundef 3)
          to label %1723 unwind label %1720

1720:                                             ; preds = %1718
  %1721 = landingpad { ptr, i32 }
          catch ptr null
  %1722 = extractvalue { ptr, i32 } %1721, 0
  call void @__clang_call_terminate(ptr %1722) #21
  unreachable

1723:                                             ; preds = %1714, %1718
  %1724 = load ptr, ptr %846, align 8, !tbaa !20
  %1725 = icmp eq ptr %1724, null
  br i1 %1725, label %1935, label %1726

1726:                                             ; preds = %1723
  %1727 = invoke noundef i1 %1724(ptr noundef nonnull align 8 dereferenceable(32) %72, ptr noundef nonnull align 8 dereferenceable(32) %72, i32 noundef 3)
          to label %1935 unwind label %1728

1728:                                             ; preds = %1726
  %1729 = landingpad { ptr, i32 }
          catch ptr null
  %1730 = extractvalue { ptr, i32 } %1729, 0
  call void @__clang_call_terminate(ptr %1730) #21
  unreachable

1731:                                             ; preds = %866
  %1732 = landingpad { ptr, i32 }
          cleanup
  %1733 = load ptr, ptr %869, align 8, !tbaa !20
  %1734 = icmp eq ptr %1733, null
  br i1 %1734, label %1740, label %1735

1735:                                             ; preds = %1731
  %1736 = invoke noundef i1 %1733(ptr noundef nonnull align 8 dereferenceable(32) %75, ptr noundef nonnull align 8 dereferenceable(32) %75, i32 noundef 3)
          to label %1740 unwind label %1737

1737:                                             ; preds = %1735
  %1738 = landingpad { ptr, i32 }
          catch ptr null
  %1739 = extractvalue { ptr, i32 } %1738, 0
  call void @__clang_call_terminate(ptr %1739) #21
  unreachable

1740:                                             ; preds = %1731, %1735
  %1741 = load ptr, ptr %867, align 8, !tbaa !20
  %1742 = icmp eq ptr %1741, null
  br i1 %1742, label %1935, label %1743

1743:                                             ; preds = %1740
  %1744 = invoke noundef i1 %1741(ptr noundef nonnull align 8 dereferenceable(32) %74, ptr noundef nonnull align 8 dereferenceable(32) %74, i32 noundef 3)
          to label %1935 unwind label %1745

1745:                                             ; preds = %1743
  %1746 = landingpad { ptr, i32 }
          catch ptr null
  %1747 = extractvalue { ptr, i32 } %1746, 0
  call void @__clang_call_terminate(ptr %1747) #21
  unreachable

1748:                                             ; preds = %887
  %1749 = landingpad { ptr, i32 }
          cleanup
  %1750 = load ptr, ptr %890, align 8, !tbaa !20
  %1751 = icmp eq ptr %1750, null
  br i1 %1751, label %1757, label %1752

1752:                                             ; preds = %1748
  %1753 = invoke noundef i1 %1750(ptr noundef nonnull align 8 dereferenceable(32) %77, ptr noundef nonnull align 8 dereferenceable(32) %77, i32 noundef 3)
          to label %1757 unwind label %1754

1754:                                             ; preds = %1752
  %1755 = landingpad { ptr, i32 }
          catch ptr null
  %1756 = extractvalue { ptr, i32 } %1755, 0
  call void @__clang_call_terminate(ptr %1756) #21
  unreachable

1757:                                             ; preds = %1748, %1752
  %1758 = load ptr, ptr %888, align 8, !tbaa !20
  %1759 = icmp eq ptr %1758, null
  br i1 %1759, label %1935, label %1760

1760:                                             ; preds = %1757
  %1761 = invoke noundef i1 %1758(ptr noundef nonnull align 8 dereferenceable(32) %76, ptr noundef nonnull align 8 dereferenceable(32) %76, i32 noundef 3)
          to label %1935 unwind label %1762

1762:                                             ; preds = %1760
  %1763 = landingpad { ptr, i32 }
          catch ptr null
  %1764 = extractvalue { ptr, i32 } %1763, 0
  call void @__clang_call_terminate(ptr %1764) #21
  unreachable

1765:                                             ; preds = %908
  %1766 = landingpad { ptr, i32 }
          cleanup
  %1767 = load ptr, ptr %911, align 8, !tbaa !20
  %1768 = icmp eq ptr %1767, null
  br i1 %1768, label %1774, label %1769

1769:                                             ; preds = %1765
  %1770 = invoke noundef i1 %1767(ptr noundef nonnull align 8 dereferenceable(32) %79, ptr noundef nonnull align 8 dereferenceable(32) %79, i32 noundef 3)
          to label %1774 unwind label %1771

1771:                                             ; preds = %1769
  %1772 = landingpad { ptr, i32 }
          catch ptr null
  %1773 = extractvalue { ptr, i32 } %1772, 0
  call void @__clang_call_terminate(ptr %1773) #21
  unreachable

1774:                                             ; preds = %1765, %1769
  %1775 = load ptr, ptr %909, align 8, !tbaa !20
  %1776 = icmp eq ptr %1775, null
  br i1 %1776, label %1935, label %1777

1777:                                             ; preds = %1774
  %1778 = invoke noundef i1 %1775(ptr noundef nonnull align 8 dereferenceable(32) %78, ptr noundef nonnull align 8 dereferenceable(32) %78, i32 noundef 3)
          to label %1935 unwind label %1779

1779:                                             ; preds = %1777
  %1780 = landingpad { ptr, i32 }
          catch ptr null
  %1781 = extractvalue { ptr, i32 } %1780, 0
  call void @__clang_call_terminate(ptr %1781) #21
  unreachable

1782:                                             ; preds = %929
  %1783 = landingpad { ptr, i32 }
          cleanup
  %1784 = load ptr, ptr %932, align 8, !tbaa !20
  %1785 = icmp eq ptr %1784, null
  br i1 %1785, label %1791, label %1786

1786:                                             ; preds = %1782
  %1787 = invoke noundef i1 %1784(ptr noundef nonnull align 8 dereferenceable(32) %81, ptr noundef nonnull align 8 dereferenceable(32) %81, i32 noundef 3)
          to label %1791 unwind label %1788

1788:                                             ; preds = %1786
  %1789 = landingpad { ptr, i32 }
          catch ptr null
  %1790 = extractvalue { ptr, i32 } %1789, 0
  call void @__clang_call_terminate(ptr %1790) #21
  unreachable

1791:                                             ; preds = %1782, %1786
  %1792 = load ptr, ptr %930, align 8, !tbaa !20
  %1793 = icmp eq ptr %1792, null
  br i1 %1793, label %1935, label %1794

1794:                                             ; preds = %1791
  %1795 = invoke noundef i1 %1792(ptr noundef nonnull align 8 dereferenceable(32) %80, ptr noundef nonnull align 8 dereferenceable(32) %80, i32 noundef 3)
          to label %1935 unwind label %1796

1796:                                             ; preds = %1794
  %1797 = landingpad { ptr, i32 }
          catch ptr null
  %1798 = extractvalue { ptr, i32 } %1797, 0
  call void @__clang_call_terminate(ptr %1798) #21
  unreachable

1799:                                             ; preds = %950
  %1800 = landingpad { ptr, i32 }
          cleanup
  %1801 = load ptr, ptr %953, align 8, !tbaa !20
  %1802 = icmp eq ptr %1801, null
  br i1 %1802, label %1808, label %1803

1803:                                             ; preds = %1799
  %1804 = invoke noundef i1 %1801(ptr noundef nonnull align 8 dereferenceable(32) %83, ptr noundef nonnull align 8 dereferenceable(32) %83, i32 noundef 3)
          to label %1808 unwind label %1805

1805:                                             ; preds = %1803
  %1806 = landingpad { ptr, i32 }
          catch ptr null
  %1807 = extractvalue { ptr, i32 } %1806, 0
  call void @__clang_call_terminate(ptr %1807) #21
  unreachable

1808:                                             ; preds = %1799, %1803
  %1809 = load ptr, ptr %951, align 8, !tbaa !20
  %1810 = icmp eq ptr %1809, null
  br i1 %1810, label %1935, label %1811

1811:                                             ; preds = %1808
  %1812 = invoke noundef i1 %1809(ptr noundef nonnull align 8 dereferenceable(32) %82, ptr noundef nonnull align 8 dereferenceable(32) %82, i32 noundef 3)
          to label %1935 unwind label %1813

1813:                                             ; preds = %1811
  %1814 = landingpad { ptr, i32 }
          catch ptr null
  %1815 = extractvalue { ptr, i32 } %1814, 0
  call void @__clang_call_terminate(ptr %1815) #21
  unreachable

1816:                                             ; preds = %971
  %1817 = landingpad { ptr, i32 }
          cleanup
  %1818 = load ptr, ptr %974, align 8, !tbaa !20
  %1819 = icmp eq ptr %1818, null
  br i1 %1819, label %1825, label %1820

1820:                                             ; preds = %1816
  %1821 = invoke noundef i1 %1818(ptr noundef nonnull align 8 dereferenceable(32) %85, ptr noundef nonnull align 8 dereferenceable(32) %85, i32 noundef 3)
          to label %1825 unwind label %1822

1822:                                             ; preds = %1820
  %1823 = landingpad { ptr, i32 }
          catch ptr null
  %1824 = extractvalue { ptr, i32 } %1823, 0
  call void @__clang_call_terminate(ptr %1824) #21
  unreachable

1825:                                             ; preds = %1816, %1820
  %1826 = load ptr, ptr %972, align 8, !tbaa !20
  %1827 = icmp eq ptr %1826, null
  br i1 %1827, label %1935, label %1828

1828:                                             ; preds = %1825
  %1829 = invoke noundef i1 %1826(ptr noundef nonnull align 8 dereferenceable(32) %84, ptr noundef nonnull align 8 dereferenceable(32) %84, i32 noundef 3)
          to label %1935 unwind label %1830

1830:                                             ; preds = %1828
  %1831 = landingpad { ptr, i32 }
          catch ptr null
  %1832 = extractvalue { ptr, i32 } %1831, 0
  call void @__clang_call_terminate(ptr %1832) #21
  unreachable

1833:                                             ; preds = %992
  %1834 = landingpad { ptr, i32 }
          cleanup
  %1835 = load ptr, ptr %995, align 8, !tbaa !20
  %1836 = icmp eq ptr %1835, null
  br i1 %1836, label %1842, label %1837

1837:                                             ; preds = %1833
  %1838 = invoke noundef i1 %1835(ptr noundef nonnull align 8 dereferenceable(32) %87, ptr noundef nonnull align 8 dereferenceable(32) %87, i32 noundef 3)
          to label %1842 unwind label %1839

1839:                                             ; preds = %1837
  %1840 = landingpad { ptr, i32 }
          catch ptr null
  %1841 = extractvalue { ptr, i32 } %1840, 0
  call void @__clang_call_terminate(ptr %1841) #21
  unreachable

1842:                                             ; preds = %1833, %1837
  %1843 = load ptr, ptr %993, align 8, !tbaa !20
  %1844 = icmp eq ptr %1843, null
  br i1 %1844, label %1935, label %1845

1845:                                             ; preds = %1842
  %1846 = invoke noundef i1 %1843(ptr noundef nonnull align 8 dereferenceable(32) %86, ptr noundef nonnull align 8 dereferenceable(32) %86, i32 noundef 3)
          to label %1935 unwind label %1847

1847:                                             ; preds = %1845
  %1848 = landingpad { ptr, i32 }
          catch ptr null
  %1849 = extractvalue { ptr, i32 } %1848, 0
  call void @__clang_call_terminate(ptr %1849) #21
  unreachable

1850:                                             ; preds = %1013
  %1851 = landingpad { ptr, i32 }
          cleanup
  %1852 = load ptr, ptr %1016, align 8, !tbaa !20
  %1853 = icmp eq ptr %1852, null
  br i1 %1853, label %1859, label %1854

1854:                                             ; preds = %1850
  %1855 = invoke noundef i1 %1852(ptr noundef nonnull align 8 dereferenceable(32) %89, ptr noundef nonnull align 8 dereferenceable(32) %89, i32 noundef 3)
          to label %1859 unwind label %1856

1856:                                             ; preds = %1854
  %1857 = landingpad { ptr, i32 }
          catch ptr null
  %1858 = extractvalue { ptr, i32 } %1857, 0
  call void @__clang_call_terminate(ptr %1858) #21
  unreachable

1859:                                             ; preds = %1850, %1854
  %1860 = load ptr, ptr %1014, align 8, !tbaa !20
  %1861 = icmp eq ptr %1860, null
  br i1 %1861, label %1935, label %1862

1862:                                             ; preds = %1859
  %1863 = invoke noundef i1 %1860(ptr noundef nonnull align 8 dereferenceable(32) %88, ptr noundef nonnull align 8 dereferenceable(32) %88, i32 noundef 3)
          to label %1935 unwind label %1864

1864:                                             ; preds = %1862
  %1865 = landingpad { ptr, i32 }
          catch ptr null
  %1866 = extractvalue { ptr, i32 } %1865, 0
  call void @__clang_call_terminate(ptr %1866) #21
  unreachable

1867:                                             ; preds = %1034
  %1868 = landingpad { ptr, i32 }
          cleanup
  %1869 = load ptr, ptr %1037, align 8, !tbaa !20
  %1870 = icmp eq ptr %1869, null
  br i1 %1870, label %1876, label %1871

1871:                                             ; preds = %1867
  %1872 = invoke noundef i1 %1869(ptr noundef nonnull align 8 dereferenceable(32) %91, ptr noundef nonnull align 8 dereferenceable(32) %91, i32 noundef 3)
          to label %1876 unwind label %1873

1873:                                             ; preds = %1871
  %1874 = landingpad { ptr, i32 }
          catch ptr null
  %1875 = extractvalue { ptr, i32 } %1874, 0
  call void @__clang_call_terminate(ptr %1875) #21
  unreachable

1876:                                             ; preds = %1867, %1871
  %1877 = load ptr, ptr %1035, align 8, !tbaa !20
  %1878 = icmp eq ptr %1877, null
  br i1 %1878, label %1935, label %1879

1879:                                             ; preds = %1876
  %1880 = invoke noundef i1 %1877(ptr noundef nonnull align 8 dereferenceable(32) %90, ptr noundef nonnull align 8 dereferenceable(32) %90, i32 noundef 3)
          to label %1935 unwind label %1881

1881:                                             ; preds = %1879
  %1882 = landingpad { ptr, i32 }
          catch ptr null
  %1883 = extractvalue { ptr, i32 } %1882, 0
  call void @__clang_call_terminate(ptr %1883) #21
  unreachable

1884:                                             ; preds = %1055
  %1885 = landingpad { ptr, i32 }
          cleanup
  %1886 = load ptr, ptr %1058, align 8, !tbaa !20
  %1887 = icmp eq ptr %1886, null
  br i1 %1887, label %1893, label %1888

1888:                                             ; preds = %1884
  %1889 = invoke noundef i1 %1886(ptr noundef nonnull align 8 dereferenceable(32) %93, ptr noundef nonnull align 8 dereferenceable(32) %93, i32 noundef 3)
          to label %1893 unwind label %1890

1890:                                             ; preds = %1888
  %1891 = landingpad { ptr, i32 }
          catch ptr null
  %1892 = extractvalue { ptr, i32 } %1891, 0
  call void @__clang_call_terminate(ptr %1892) #21
  unreachable

1893:                                             ; preds = %1884, %1888
  %1894 = load ptr, ptr %1056, align 8, !tbaa !20
  %1895 = icmp eq ptr %1894, null
  br i1 %1895, label %1935, label %1896

1896:                                             ; preds = %1893
  %1897 = invoke noundef i1 %1894(ptr noundef nonnull align 8 dereferenceable(32) %92, ptr noundef nonnull align 8 dereferenceable(32) %92, i32 noundef 3)
          to label %1935 unwind label %1898

1898:                                             ; preds = %1896
  %1899 = landingpad { ptr, i32 }
          catch ptr null
  %1900 = extractvalue { ptr, i32 } %1899, 0
  call void @__clang_call_terminate(ptr %1900) #21
  unreachable

1901:                                             ; preds = %1076
  %1902 = landingpad { ptr, i32 }
          cleanup
  %1903 = load ptr, ptr %1079, align 8, !tbaa !20
  %1904 = icmp eq ptr %1903, null
  br i1 %1904, label %1910, label %1905

1905:                                             ; preds = %1901
  %1906 = invoke noundef i1 %1903(ptr noundef nonnull align 8 dereferenceable(32) %95, ptr noundef nonnull align 8 dereferenceable(32) %95, i32 noundef 3)
          to label %1910 unwind label %1907

1907:                                             ; preds = %1905
  %1908 = landingpad { ptr, i32 }
          catch ptr null
  %1909 = extractvalue { ptr, i32 } %1908, 0
  call void @__clang_call_terminate(ptr %1909) #21
  unreachable

1910:                                             ; preds = %1901, %1905
  %1911 = load ptr, ptr %1077, align 8, !tbaa !20
  %1912 = icmp eq ptr %1911, null
  br i1 %1912, label %1935, label %1913

1913:                                             ; preds = %1910
  %1914 = invoke noundef i1 %1911(ptr noundef nonnull align 8 dereferenceable(32) %94, ptr noundef nonnull align 8 dereferenceable(32) %94, i32 noundef 3)
          to label %1935 unwind label %1915

1915:                                             ; preds = %1913
  %1916 = landingpad { ptr, i32 }
          catch ptr null
  %1917 = extractvalue { ptr, i32 } %1916, 0
  call void @__clang_call_terminate(ptr %1917) #21
  unreachable

1918:                                             ; preds = %1097
  %1919 = landingpad { ptr, i32 }
          cleanup
  %1920 = load ptr, ptr %1100, align 8, !tbaa !20
  %1921 = icmp eq ptr %1920, null
  br i1 %1921, label %1927, label %1922

1922:                                             ; preds = %1918
  %1923 = invoke noundef i1 %1920(ptr noundef nonnull align 8 dereferenceable(32) %97, ptr noundef nonnull align 8 dereferenceable(32) %97, i32 noundef 3)
          to label %1927 unwind label %1924

1924:                                             ; preds = %1922
  %1925 = landingpad { ptr, i32 }
          catch ptr null
  %1926 = extractvalue { ptr, i32 } %1925, 0
  call void @__clang_call_terminate(ptr %1926) #21
  unreachable

1927:                                             ; preds = %1918, %1922
  %1928 = load ptr, ptr %1098, align 8, !tbaa !20
  %1929 = icmp eq ptr %1928, null
  br i1 %1929, label %1935, label %1930

1930:                                             ; preds = %1927
  %1931 = invoke noundef i1 %1928(ptr noundef nonnull align 8 dereferenceable(32) %96, ptr noundef nonnull align 8 dereferenceable(32) %96, i32 noundef 3)
          to label %1935 unwind label %1932

1932:                                             ; preds = %1930
  %1933 = landingpad { ptr, i32 }
          catch ptr null
  %1934 = extractvalue { ptr, i32 } %1933, 0
  call void @__clang_call_terminate(ptr %1934) #21
  unreachable

1935:                                             ; preds = %1893, %1896, %1910, %1913, %1927, %1930, %1842, %1845, %1859, %1862, %1876, %1879, %1740, %1743, %1757, %1760, %1774, %1777, %1791, %1794, %1808, %1811, %1825, %1828, %1638, %1641, %1655, %1658, %1672, %1675, %1689, %1692, %1706, %1709, %1723, %1726, %1587, %1590, %1604, %1607, %1621, %1624, %1536, %1539, %1553, %1556, %1570, %1573, %1485, %1488, %1502, %1505, %1519, %1522, %1434, %1437, %1451, %1454, %1468, %1471, %1383, %1386, %1400, %1403, %1417, %1420, %1332, %1335, %1349, %1352, %1366, %1369, %1281, %1284, %1298, %1301, %1315, %1318, %1230, %1233, %1247, %1250, %1264, %1267, %1179, %1182, %1196, %1199, %1213, %1216, %1128, %1131, %1145, %1148, %1162, %1165
  %1936 = phi { ptr, i32 } [ %1120, %1128 ], [ %1120, %1131 ], [ %1137, %1145 ], [ %1137, %1148 ], [ %1154, %1162 ], [ %1154, %1165 ], [ %1171, %1179 ], [ %1171, %1182 ], [ %1188, %1196 ], [ %1188, %1199 ], [ %1205, %1213 ], [ %1205, %1216 ], [ %1222, %1230 ], [ %1222, %1233 ], [ %1239, %1247 ], [ %1239, %1250 ], [ %1256, %1264 ], [ %1256, %1267 ], [ %1273, %1281 ], [ %1273, %1284 ], [ %1290, %1298 ], [ %1290, %1301 ], [ %1307, %1315 ], [ %1307, %1318 ], [ %1324, %1332 ], [ %1324, %1335 ], [ %1341, %1349 ], [ %1341, %1352 ], [ %1358, %1366 ], [ %1358, %1369 ], [ %1375, %1383 ], [ %1375, %1386 ], [ %1392, %1400 ], [ %1392, %1403 ], [ %1409, %1417 ], [ %1409, %1420 ], [ %1426, %1434 ], [ %1426, %1437 ], [ %1443, %1451 ], [ %1443, %1454 ], [ %1460, %1468 ], [ %1460, %1471 ], [ %1477, %1485 ], [ %1477, %1488 ], [ %1494, %1502 ], [ %1494, %1505 ], [ %1511, %1519 ], [ %1511, %1522 ], [ %1528, %1536 ], [ %1528, %1539 ], [ %1545, %1553 ], [ %1545, %1556 ], [ %1562, %1570 ], [ %1562, %1573 ], [ %1579, %1587 ], [ %1579, %1590 ], [ %1596, %1604 ], [ %1596, %1607 ], [ %1613, %1621 ], [ %1613, %1624 ], [ %1630, %1638 ], [ %1630, %1641 ], [ %1647, %1655 ], [ %1647, %1658 ], [ %1664, %1672 ], [ %1664, %1675 ], [ %1681, %1689 ], [ %1681, %1692 ], [ %1698, %1706 ], [ %1698, %1709 ], [ %1715, %1723 ], [ %1715, %1726 ], [ %1732, %1740 ], [ %1732, %1743 ], [ %1749, %1757 ], [ %1749, %1760 ], [ %1766, %1774 ], [ %1766, %1777 ], [ %1783, %1791 ], [ %1783, %1794 ], [ %1800, %1808 ], [ %1800, %1811 ], [ %1817, %1825 ], [ %1817, %1828 ], [ %1834, %1842 ], [ %1834, %1845 ], [ %1851, %1859 ], [ %1851, %1862 ], [ %1868, %1876 ], [ %1868, %1879 ], [ %1885, %1893 ], [ %1885, %1896 ], [ %1902, %1910 ], [ %1902, %1913 ], [ %1919, %1927 ], [ %1919, %1930 ]
  resume { ptr, i32 } %1936
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: mustprogress norecurse uwtable
define internal fastcc void @_ZL37checkOverlappingMemoryOneRuntimeCheckIhEvSt8functionIFvPT_S2_jEES4_PKc(ptr noundef nonnull %0, ptr noundef nonnull %1, ptr noundef %2) unnamed_addr #0 personality ptr @__gxx_personality_v0 {
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca i32, align 4
  %7 = alloca %"class.std::uniform_int_distribution", align 4
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  %10 = alloca i32, align 4
  %11 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.48, i64 noundef 9)
  %12 = icmp eq ptr %2, null
  br i1 %12, label %13, label %21

13:                                               ; preds = %3
  %14 = load ptr, ptr @_ZSt4cout, align 8, !tbaa !37
  %15 = getelementptr i8, ptr %14, i64 -24
  %16 = load i64, ptr %15, align 8
  %17 = getelementptr inbounds i8, ptr @_ZSt4cout, i64 %16
  %18 = getelementptr inbounds nuw i8, ptr %17, i64 32
  %19 = load i32, ptr %18, align 8, !tbaa !39
  %20 = or i32 %19, 1
  tail call void @_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate(ptr noundef nonnull align 8 dereferenceable(264) %17, i32 noundef %20)
  br label %24

21:                                               ; preds = %3
  %22 = tail call noundef i64 @strlen(ptr noundef nonnull dereferenceable(1) %2) #20
  %23 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull %2, i64 noundef %22)
  br label %24

24:                                               ; preds = %13, %21
  %25 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.49, i64 noundef 1)
  call void @llvm.lifetime.start.p0(ptr nonnull %10) #20
  store i32 100, ptr %10, align 4, !tbaa !49
  %26 = tail call noalias noundef nonnull dereferenceable(800) ptr @_Znam(i64 noundef 800) #22
  %27 = invoke noalias noundef nonnull dereferenceable(800) ptr @_Znam(i64 noundef 800) #22
          to label %28 unwind label %37

28:                                               ; preds = %24
  %29 = invoke noalias noundef nonnull dereferenceable(800) ptr @_Znam(i64 noundef 800) #22
          to label %30 unwind label %39

30:                                               ; preds = %28
  %31 = getelementptr inbounds nuw i8, ptr %7, i64 1
  %32 = getelementptr inbounds nuw i8, ptr %27, i64 400
  %33 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %34 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %35 = getelementptr inbounds nuw i8, ptr %29, i64 400
  br label %41

36:                                               ; preds = %72
  call void @_ZdaPv(ptr noundef nonnull %29) #23
  call void @_ZdaPv(ptr noundef nonnull %27) #23
  call void @_ZdaPv(ptr noundef nonnull %26) #23
  call void @llvm.lifetime.end.p0(ptr nonnull %10) #20
  ret void

37:                                               ; preds = %24
  %38 = landingpad { ptr, i32 }
          cleanup
  br label %85

39:                                               ; preds = %28
  %40 = landingpad { ptr, i32 }
          cleanup
  br label %83

41:                                               ; preds = %30, %72
  %42 = phi i64 [ -100, %30 ], [ %73, %72 ]
  call void @llvm.lifetime.start.p0(ptr nonnull %7) #20
  store i8 0, ptr %7, align 4, !tbaa !50
  store i8 -1, ptr %31, align 1, !tbaa !52
  br label %43

43:                                               ; preds = %46, %41
  %44 = phi i64 [ 0, %41 ], [ %48, %46 ]
  %45 = invoke noundef i8 @_ZNSt24uniform_int_distributionIhEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEhRT_RKNS0_10param_typeE(ptr noundef nonnull align 1 dereferenceable(2) %7, ptr noundef nonnull align 8 dereferenceable(5000) @_ZL3rng, ptr noundef nonnull align 1 dereferenceable(2) %7)
          to label %46 unwind label %75

46:                                               ; preds = %43
  %47 = getelementptr inbounds nuw i8, ptr %26, i64 %44
  store i8 %45, ptr %47, align 1, !tbaa !15
  %48 = add nuw nsw i64 %44, 1
  %49 = icmp eq i64 %48, 800
  br i1 %49, label %50, label %43, !llvm.loop !53

50:                                               ; preds = %46
  call void @llvm.lifetime.end.p0(ptr nonnull %7) #20
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(800) %27, ptr noundef nonnull align 1 dereferenceable(800) %26, i64 800, i1 false), !tbaa !15
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(800) %29, ptr noundef nonnull align 1 dereferenceable(800) %26, i64 800, i1 false), !tbaa !15
  %51 = getelementptr inbounds i8, ptr %32, i64 %42
  %52 = load i32, ptr %10, align 4, !tbaa !49
  call void @llvm.lifetime.start.p0(ptr nonnull %4)
  call void @llvm.lifetime.start.p0(ptr nonnull %5)
  call void @llvm.lifetime.start.p0(ptr nonnull %6)
  store ptr %51, ptr %4, align 8, !tbaa !54
  store ptr %32, ptr %5, align 8, !tbaa !54
  store i32 %52, ptr %6, align 4, !tbaa !49
  %53 = load ptr, ptr %33, align 8, !tbaa !20
  %54 = icmp eq ptr %53, null
  br i1 %54, label %55, label %57

55:                                               ; preds = %50
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %56 unwind label %79

56:                                               ; preds = %55
  unreachable

57:                                               ; preds = %50
  %58 = load ptr, ptr %34, align 8, !tbaa !16
  invoke void %58(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %4, ptr noundef nonnull align 8 dereferenceable(8) %5, ptr noundef nonnull align 4 dereferenceable(4) %6)
          to label %59 unwind label %77

59:                                               ; preds = %57
  call void @llvm.lifetime.end.p0(ptr nonnull %4)
  call void @llvm.lifetime.end.p0(ptr nonnull %5)
  call void @llvm.lifetime.end.p0(ptr nonnull %6)
  call void @llvm.lifetime.start.p0(ptr nonnull %8) #20
  store ptr %35, ptr %8, align 8, !tbaa !54
  call void @llvm.lifetime.start.p0(ptr nonnull %9) #20
  %60 = getelementptr inbounds i8, ptr %35, i64 %42
  store ptr %60, ptr %9, align 8, !tbaa !54
  invoke fastcc void @_ZL18callThroughOptnoneIRSt8functionIFvPhS1_jEEJS1_RS1_RjEEvOT_DpOT0_(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %9, ptr noundef nonnull align 8 dereferenceable(8) %8, ptr noundef nonnull align 4 dereferenceable(4) %10)
          to label %61 unwind label %77

61:                                               ; preds = %59
  call void @llvm.lifetime.end.p0(ptr nonnull %9) #20
  %62 = call i32 @bcmp(ptr noundef nonnull readonly dereferenceable(800) %27, ptr noundef nonnull readonly dereferenceable(800) %29, i64 800)
  %63 = icmp eq i32 %62, 0
  br i1 %63, label %72, label %64

64:                                               ; preds = %61
  %65 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.50, i64 noundef 23)
          to label %66 unwind label %79

66:                                               ; preds = %64
  %67 = trunc nsw i64 %42 to i32
  %68 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEi(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, i32 noundef range(i32 -2147483648, 101) %67)
          to label %69 unwind label %79

69:                                               ; preds = %66
  %70 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) %68, ptr noundef nonnull @.str.49)
          to label %71 unwind label %79

71:                                               ; preds = %69
  call void @exit(i32 noundef 1) #25
  unreachable

72:                                               ; preds = %61
  call void @llvm.lifetime.end.p0(ptr nonnull %8) #20
  %73 = add nsw i64 %42, 1
  %74 = icmp eq i64 %73, 101
  br i1 %74, label %36, label %41, !llvm.loop !56

75:                                               ; preds = %43
  %76 = landingpad { ptr, i32 }
          cleanup
  br label %81

77:                                               ; preds = %59, %57
  %78 = landingpad { ptr, i32 }
          cleanup
  br label %81

79:                                               ; preds = %64, %69, %66, %55
  %80 = landingpad { ptr, i32 }
          cleanup
  br label %81

81:                                               ; preds = %77, %79, %75
  %82 = phi { ptr, i32 } [ %76, %75 ], [ %78, %77 ], [ %80, %79 ]
  call void @_ZdaPv(ptr noundef nonnull %29) #23
  br label %83

83:                                               ; preds = %81, %39
  %84 = phi { ptr, i32 } [ %82, %81 ], [ %40, %39 ]
  call void @_ZdaPv(ptr noundef nonnull %27) #23
  br label %85

85:                                               ; preds = %83, %37
  %86 = phi { ptr, i32 } [ %84, %83 ], [ %38, %37 ]
  call void @_ZdaPv(ptr noundef nonnull %26) #23
  call void @llvm.lifetime.end.p0(ptr nonnull %10) #20
  resume { ptr, i32 } %86
}

declare i32 @__gxx_personality_v0(...)

; Function Attrs: mustprogress norecurse uwtable
define internal fastcc void @_ZL37checkOverlappingMemoryOneRuntimeCheckIjEvSt8functionIFvPT_S2_jEES4_PKc(ptr noundef nonnull %0, ptr noundef nonnull %1, ptr noundef %2) unnamed_addr #0 personality ptr @__gxx_personality_v0 {
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca i32, align 4
  %7 = alloca %"class.std::uniform_int_distribution.84", align 8
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  %10 = alloca i32, align 4
  %11 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.48, i64 noundef 9)
  %12 = icmp eq ptr %2, null
  br i1 %12, label %13, label %21

13:                                               ; preds = %3
  %14 = load ptr, ptr @_ZSt4cout, align 8, !tbaa !37
  %15 = getelementptr i8, ptr %14, i64 -24
  %16 = load i64, ptr %15, align 8
  %17 = getelementptr inbounds i8, ptr @_ZSt4cout, i64 %16
  %18 = getelementptr inbounds nuw i8, ptr %17, i64 32
  %19 = load i32, ptr %18, align 8, !tbaa !39
  %20 = or i32 %19, 1
  tail call void @_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate(ptr noundef nonnull align 8 dereferenceable(264) %17, i32 noundef %20)
  br label %24

21:                                               ; preds = %3
  %22 = tail call noundef i64 @strlen(ptr noundef nonnull dereferenceable(1) %2) #20
  %23 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull %2, i64 noundef %22)
  br label %24

24:                                               ; preds = %13, %21
  %25 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.49, i64 noundef 1)
  call void @llvm.lifetime.start.p0(ptr nonnull %10) #20
  store i32 100, ptr %10, align 4, !tbaa !49
  %26 = tail call noalias noundef nonnull dereferenceable(3200) ptr @_Znam(i64 noundef 3200) #22
  %27 = invoke noalias noundef nonnull dereferenceable(3200) ptr @_Znam(i64 noundef 3200) #22
          to label %28 unwind label %36

28:                                               ; preds = %24
  %29 = invoke noalias noundef nonnull dereferenceable(3200) ptr @_Znam(i64 noundef 3200) #22
          to label %30 unwind label %38

30:                                               ; preds = %28
  %31 = getelementptr inbounds nuw i8, ptr %27, i64 1600
  %32 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %33 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %34 = getelementptr inbounds nuw i8, ptr %29, i64 1600
  br label %40

35:                                               ; preds = %71
  call void @_ZdaPv(ptr noundef nonnull %29) #23
  call void @_ZdaPv(ptr noundef nonnull %27) #23
  call void @_ZdaPv(ptr noundef nonnull %26) #23
  call void @llvm.lifetime.end.p0(ptr nonnull %10) #20
  ret void

36:                                               ; preds = %24
  %37 = landingpad { ptr, i32 }
          cleanup
  br label %84

38:                                               ; preds = %28
  %39 = landingpad { ptr, i32 }
          cleanup
  br label %82

40:                                               ; preds = %30, %71
  %41 = phi i64 [ -100, %30 ], [ %72, %71 ]
  call void @llvm.lifetime.start.p0(ptr nonnull %7) #20
  store <2 x i32> <i32 0, i32 -1>, ptr %7, align 8, !tbaa !49
  br label %42

42:                                               ; preds = %45, %40
  %43 = phi i64 [ 0, %40 ], [ %47, %45 ]
  %44 = invoke noundef i32 @_ZNSt24uniform_int_distributionIjEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEjRT_RKNS0_10param_typeE(ptr noundef nonnull align 4 dereferenceable(8) %7, ptr noundef nonnull align 8 dereferenceable(5000) @_ZL3rng, ptr noundef nonnull align 4 dereferenceable(8) %7)
          to label %45 unwind label %74

45:                                               ; preds = %42
  %46 = getelementptr inbounds nuw i32, ptr %26, i64 %43
  store i32 %44, ptr %46, align 4, !tbaa !49
  %47 = add nuw nsw i64 %43, 1
  %48 = icmp eq i64 %47, 800
  br i1 %48, label %49, label %42, !llvm.loop !57

49:                                               ; preds = %45
  call void @llvm.lifetime.end.p0(ptr nonnull %7) #20
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 4 dereferenceable(3200) %27, ptr noundef nonnull align 4 dereferenceable(3200) %26, i64 3200, i1 false), !tbaa !49
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 4 dereferenceable(3200) %29, ptr noundef nonnull align 4 dereferenceable(3200) %26, i64 3200, i1 false), !tbaa !49
  %50 = getelementptr inbounds i32, ptr %31, i64 %41
  %51 = load i32, ptr %10, align 4, !tbaa !49
  call void @llvm.lifetime.start.p0(ptr nonnull %4)
  call void @llvm.lifetime.start.p0(ptr nonnull %5)
  call void @llvm.lifetime.start.p0(ptr nonnull %6)
  store ptr %50, ptr %4, align 8, !tbaa !58
  store ptr %31, ptr %5, align 8, !tbaa !58
  store i32 %51, ptr %6, align 4, !tbaa !49
  %52 = load ptr, ptr %32, align 8, !tbaa !20
  %53 = icmp eq ptr %52, null
  br i1 %53, label %54, label %56

54:                                               ; preds = %49
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %55 unwind label %78

55:                                               ; preds = %54
  unreachable

56:                                               ; preds = %49
  %57 = load ptr, ptr %33, align 8, !tbaa !21
  invoke void %57(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %4, ptr noundef nonnull align 8 dereferenceable(8) %5, ptr noundef nonnull align 4 dereferenceable(4) %6)
          to label %58 unwind label %76

58:                                               ; preds = %56
  call void @llvm.lifetime.end.p0(ptr nonnull %4)
  call void @llvm.lifetime.end.p0(ptr nonnull %5)
  call void @llvm.lifetime.end.p0(ptr nonnull %6)
  call void @llvm.lifetime.start.p0(ptr nonnull %8) #20
  store ptr %34, ptr %8, align 8, !tbaa !58
  call void @llvm.lifetime.start.p0(ptr nonnull %9) #20
  %59 = getelementptr inbounds i32, ptr %34, i64 %41
  store ptr %59, ptr %9, align 8, !tbaa !58
  invoke fastcc void @_ZL18callThroughOptnoneIRSt8functionIFvPjS1_jEEJS1_RS1_RjEEvOT_DpOT0_(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %9, ptr noundef nonnull align 8 dereferenceable(8) %8, ptr noundef nonnull align 4 dereferenceable(4) %10)
          to label %60 unwind label %76

60:                                               ; preds = %58
  call void @llvm.lifetime.end.p0(ptr nonnull %9) #20
  %61 = call i32 @bcmp(ptr noundef nonnull readonly dereferenceable(3200) %27, ptr noundef nonnull readonly dereferenceable(3200) %29, i64 3200)
  %62 = icmp eq i32 %61, 0
  br i1 %62, label %71, label %63

63:                                               ; preds = %60
  %64 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.50, i64 noundef 23)
          to label %65 unwind label %78

65:                                               ; preds = %63
  %66 = trunc nsw i64 %41 to i32
  %67 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEi(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, i32 noundef range(i32 -2147483648, 101) %66)
          to label %68 unwind label %78

68:                                               ; preds = %65
  %69 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) %67, ptr noundef nonnull @.str.49)
          to label %70 unwind label %78

70:                                               ; preds = %68
  call void @exit(i32 noundef 1) #25
  unreachable

71:                                               ; preds = %60
  call void @llvm.lifetime.end.p0(ptr nonnull %8) #20
  %72 = add nsw i64 %41, 1
  %73 = icmp eq i64 %72, 101
  br i1 %73, label %35, label %40, !llvm.loop !60

74:                                               ; preds = %42
  %75 = landingpad { ptr, i32 }
          cleanup
  br label %80

76:                                               ; preds = %58, %56
  %77 = landingpad { ptr, i32 }
          cleanup
  br label %80

78:                                               ; preds = %63, %68, %65, %54
  %79 = landingpad { ptr, i32 }
          cleanup
  br label %80

80:                                               ; preds = %76, %78, %74
  %81 = phi { ptr, i32 } [ %75, %74 ], [ %77, %76 ], [ %79, %78 ]
  call void @_ZdaPv(ptr noundef nonnull %29) #23
  br label %82

82:                                               ; preds = %80, %38
  %83 = phi { ptr, i32 } [ %81, %80 ], [ %39, %38 ]
  call void @_ZdaPv(ptr noundef nonnull %27) #23
  br label %84

84:                                               ; preds = %82, %36
  %85 = phi { ptr, i32 } [ %83, %82 ], [ %37, %36 ]
  call void @_ZdaPv(ptr noundef nonnull %26) #23
  call void @llvm.lifetime.end.p0(ptr nonnull %10) #20
  resume { ptr, i32 } %85
}

; Function Attrs: mustprogress norecurse uwtable
define internal fastcc void @_ZL37checkOverlappingMemoryOneRuntimeCheckImEvSt8functionIFvPT_S2_jEES4_PKc(ptr noundef nonnull %0, ptr noundef nonnull %1, ptr noundef %2) unnamed_addr #0 personality ptr @__gxx_personality_v0 {
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca i32, align 4
  %7 = alloca %"class.std::uniform_int_distribution.96", align 16
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  %10 = alloca i32, align 4
  %11 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.48, i64 noundef 9)
  %12 = icmp eq ptr %2, null
  br i1 %12, label %13, label %21

13:                                               ; preds = %3
  %14 = load ptr, ptr @_ZSt4cout, align 8, !tbaa !37
  %15 = getelementptr i8, ptr %14, i64 -24
  %16 = load i64, ptr %15, align 8
  %17 = getelementptr inbounds i8, ptr @_ZSt4cout, i64 %16
  %18 = getelementptr inbounds nuw i8, ptr %17, i64 32
  %19 = load i32, ptr %18, align 8, !tbaa !39
  %20 = or i32 %19, 1
  tail call void @_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate(ptr noundef nonnull align 8 dereferenceable(264) %17, i32 noundef %20)
  br label %24

21:                                               ; preds = %3
  %22 = tail call noundef i64 @strlen(ptr noundef nonnull dereferenceable(1) %2) #20
  %23 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull %2, i64 noundef %22)
  br label %24

24:                                               ; preds = %13, %21
  %25 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.49, i64 noundef 1)
  call void @llvm.lifetime.start.p0(ptr nonnull %10) #20
  store i32 100, ptr %10, align 4, !tbaa !49
  %26 = tail call noalias noundef nonnull dereferenceable(6400) ptr @_Znam(i64 noundef 6400) #22
  %27 = invoke noalias noundef nonnull dereferenceable(6400) ptr @_Znam(i64 noundef 6400) #22
          to label %28 unwind label %36

28:                                               ; preds = %24
  %29 = invoke noalias noundef nonnull dereferenceable(6400) ptr @_Znam(i64 noundef 6400) #22
          to label %30 unwind label %38

30:                                               ; preds = %28
  %31 = getelementptr inbounds nuw i8, ptr %27, i64 3200
  %32 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %33 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %34 = getelementptr inbounds nuw i8, ptr %29, i64 3200
  br label %40

35:                                               ; preds = %71
  call void @_ZdaPv(ptr noundef nonnull %29) #23
  call void @_ZdaPv(ptr noundef nonnull %27) #23
  call void @_ZdaPv(ptr noundef nonnull %26) #23
  call void @llvm.lifetime.end.p0(ptr nonnull %10) #20
  ret void

36:                                               ; preds = %24
  %37 = landingpad { ptr, i32 }
          cleanup
  br label %84

38:                                               ; preds = %28
  %39 = landingpad { ptr, i32 }
          cleanup
  br label %82

40:                                               ; preds = %30, %71
  %41 = phi i64 [ -100, %30 ], [ %72, %71 ]
  call void @llvm.lifetime.start.p0(ptr nonnull %7) #20
  store <2 x i64> <i64 0, i64 -1>, ptr %7, align 16, !tbaa !6
  br label %42

42:                                               ; preds = %45, %40
  %43 = phi i64 [ 0, %40 ], [ %47, %45 ]
  %44 = invoke noundef i64 @_ZNSt24uniform_int_distributionImEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEmRT_RKNS0_10param_typeE(ptr noundef nonnull align 8 dereferenceable(16) %7, ptr noundef nonnull align 8 dereferenceable(5000) @_ZL3rng, ptr noundef nonnull align 8 dereferenceable(16) %7)
          to label %45 unwind label %74

45:                                               ; preds = %42
  %46 = getelementptr inbounds nuw i64, ptr %26, i64 %43
  store i64 %44, ptr %46, align 8, !tbaa !6
  %47 = add nuw nsw i64 %43, 1
  %48 = icmp eq i64 %47, 800
  br i1 %48, label %49, label %42, !llvm.loop !61

49:                                               ; preds = %45
  call void @llvm.lifetime.end.p0(ptr nonnull %7) #20
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(6400) %27, ptr noundef nonnull align 8 dereferenceable(6400) %26, i64 6400, i1 false), !tbaa !6
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(6400) %29, ptr noundef nonnull align 8 dereferenceable(6400) %26, i64 6400, i1 false), !tbaa !6
  %50 = getelementptr inbounds i64, ptr %31, i64 %41
  %51 = load i32, ptr %10, align 4, !tbaa !49
  call void @llvm.lifetime.start.p0(ptr nonnull %4)
  call void @llvm.lifetime.start.p0(ptr nonnull %5)
  call void @llvm.lifetime.start.p0(ptr nonnull %6)
  store ptr %50, ptr %4, align 8, !tbaa !62
  store ptr %31, ptr %5, align 8, !tbaa !62
  store i32 %51, ptr %6, align 4, !tbaa !49
  %52 = load ptr, ptr %32, align 8, !tbaa !20
  %53 = icmp eq ptr %52, null
  br i1 %53, label %54, label %56

54:                                               ; preds = %49
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %55 unwind label %78

55:                                               ; preds = %54
  unreachable

56:                                               ; preds = %49
  %57 = load ptr, ptr %33, align 8, !tbaa !23
  invoke void %57(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %4, ptr noundef nonnull align 8 dereferenceable(8) %5, ptr noundef nonnull align 4 dereferenceable(4) %6)
          to label %58 unwind label %76

58:                                               ; preds = %56
  call void @llvm.lifetime.end.p0(ptr nonnull %4)
  call void @llvm.lifetime.end.p0(ptr nonnull %5)
  call void @llvm.lifetime.end.p0(ptr nonnull %6)
  call void @llvm.lifetime.start.p0(ptr nonnull %8) #20
  store ptr %34, ptr %8, align 8, !tbaa !62
  call void @llvm.lifetime.start.p0(ptr nonnull %9) #20
  %59 = getelementptr inbounds i64, ptr %34, i64 %41
  store ptr %59, ptr %9, align 8, !tbaa !62
  invoke fastcc void @_ZL18callThroughOptnoneIRSt8functionIFvPmS1_jEEJS1_RS1_RjEEvOT_DpOT0_(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %9, ptr noundef nonnull align 8 dereferenceable(8) %8, ptr noundef nonnull align 4 dereferenceable(4) %10)
          to label %60 unwind label %76

60:                                               ; preds = %58
  call void @llvm.lifetime.end.p0(ptr nonnull %9) #20
  %61 = call i32 @bcmp(ptr noundef nonnull readonly dereferenceable(6400) %27, ptr noundef nonnull readonly dereferenceable(6400) %29, i64 6400)
  %62 = icmp eq i32 %61, 0
  br i1 %62, label %71, label %63

63:                                               ; preds = %60
  %64 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.50, i64 noundef 23)
          to label %65 unwind label %78

65:                                               ; preds = %63
  %66 = trunc nsw i64 %41 to i32
  %67 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEi(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, i32 noundef range(i32 -2147483648, 101) %66)
          to label %68 unwind label %78

68:                                               ; preds = %65
  %69 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) %67, ptr noundef nonnull @.str.49)
          to label %70 unwind label %78

70:                                               ; preds = %68
  call void @exit(i32 noundef 1) #25
  unreachable

71:                                               ; preds = %60
  call void @llvm.lifetime.end.p0(ptr nonnull %8) #20
  %72 = add nsw i64 %41, 1
  %73 = icmp eq i64 %72, 101
  br i1 %73, label %35, label %40, !llvm.loop !64

74:                                               ; preds = %42
  %75 = landingpad { ptr, i32 }
          cleanup
  br label %80

76:                                               ; preds = %58, %56
  %77 = landingpad { ptr, i32 }
          cleanup
  br label %80

78:                                               ; preds = %63, %68, %65, %54
  %79 = landingpad { ptr, i32 }
          cleanup
  br label %80

80:                                               ; preds = %76, %78, %74
  %81 = phi { ptr, i32 } [ %75, %74 ], [ %77, %76 ], [ %79, %78 ]
  call void @_ZdaPv(ptr noundef nonnull %29) #23
  br label %82

82:                                               ; preds = %80, %38
  %83 = phi { ptr, i32 } [ %81, %80 ], [ %39, %38 ]
  call void @_ZdaPv(ptr noundef nonnull %27) #23
  br label %84

84:                                               ; preds = %82, %36
  %85 = phi { ptr, i32 } [ %83, %82 ], [ %37, %36 ]
  call void @_ZdaPv(ptr noundef nonnull %26) #23
  call void @llvm.lifetime.end.p0(ptr nonnull %10) #20
  resume { ptr, i32 } %85
}

; Function Attrs: mustprogress norecurse uwtable
define internal fastcc void @_ZL38checkOverlappingMemoryTwoRuntimeChecksIjEvSt8functionIFvPT_S2_S2_jEES4_PKc(ptr noundef nonnull %0, ptr noundef nonnull %1, ptr noundef %2) unnamed_addr #0 personality ptr @__gxx_personality_v0 {
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca i32, align 4
  %8 = alloca %"class.std::uniform_int_distribution.84", align 8
  %9 = alloca %"class.std::uniform_int_distribution.84", align 8
  %10 = alloca ptr, align 8
  %11 = alloca ptr, align 8
  %12 = alloca ptr, align 8
  %13 = alloca i32, align 4
  %14 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.48, i64 noundef 9)
  %15 = icmp eq ptr %2, null
  br i1 %15, label %16, label %24

16:                                               ; preds = %3
  %17 = load ptr, ptr @_ZSt4cout, align 8, !tbaa !37
  %18 = getelementptr i8, ptr %17, i64 -24
  %19 = load i64, ptr %18, align 8
  %20 = getelementptr inbounds i8, ptr @_ZSt4cout, i64 %19
  %21 = getelementptr inbounds nuw i8, ptr %20, i64 32
  %22 = load i32, ptr %21, align 8, !tbaa !39
  %23 = or i32 %22, 1
  tail call void @_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate(ptr noundef nonnull align 8 dereferenceable(264) %20, i32 noundef %23)
  br label %27

24:                                               ; preds = %3
  %25 = tail call noundef i64 @strlen(ptr noundef nonnull dereferenceable(1) %2) #20
  %26 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull %2, i64 noundef %25)
  br label %27

27:                                               ; preds = %16, %24
  %28 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.49, i64 noundef 1)
  call void @llvm.lifetime.start.p0(ptr nonnull %13) #20
  store i32 100, ptr %13, align 4, !tbaa !49
  %29 = tail call noalias noundef nonnull dereferenceable(3200) ptr @_Znam(i64 noundef 3200) #22
  %30 = invoke noalias noundef nonnull dereferenceable(3200) ptr @_Znam(i64 noundef 3200) #22
          to label %31 unwind label %41

31:                                               ; preds = %27
  %32 = invoke noalias noundef nonnull dereferenceable(3200) ptr @_Znam(i64 noundef 3200) #22
          to label %33 unwind label %43

33:                                               ; preds = %31
  %34 = invoke noalias noundef nonnull dereferenceable(3200) ptr @_Znam(i64 noundef 3200) #22
          to label %35 unwind label %45

35:                                               ; preds = %33
  %36 = getelementptr inbounds nuw i8, ptr %32, i64 1600
  %37 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %38 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %39 = getelementptr inbounds nuw i8, ptr %34, i64 1600
  br label %47

40:                                               ; preds = %86
  call void @_ZdaPv(ptr noundef nonnull %34) #23
  call void @_ZdaPv(ptr noundef nonnull %32) #23
  call void @_ZdaPv(ptr noundef nonnull %30) #23
  call void @_ZdaPv(ptr noundef nonnull %29) #23
  call void @llvm.lifetime.end.p0(ptr nonnull %13) #20
  ret void

41:                                               ; preds = %27
  %42 = landingpad { ptr, i32 }
          cleanup
  br label %103

43:                                               ; preds = %31
  %44 = landingpad { ptr, i32 }
          cleanup
  br label %101

45:                                               ; preds = %33
  %46 = landingpad { ptr, i32 }
          cleanup
  br label %99

47:                                               ; preds = %35, %86
  %48 = phi i64 [ -100, %35 ], [ %87, %86 ]
  call void @llvm.lifetime.start.p0(ptr nonnull %9) #20
  store <2 x i32> <i32 0, i32 -1>, ptr %9, align 8, !tbaa !49
  br label %49

49:                                               ; preds = %52, %47
  %50 = phi i64 [ 0, %47 ], [ %54, %52 ]
  %51 = invoke noundef i32 @_ZNSt24uniform_int_distributionIjEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEjRT_RKNS0_10param_typeE(ptr noundef nonnull align 4 dereferenceable(8) %9, ptr noundef nonnull align 8 dereferenceable(5000) @_ZL3rng, ptr noundef nonnull align 4 dereferenceable(8) %9)
          to label %52 unwind label %91

52:                                               ; preds = %49
  %53 = getelementptr inbounds nuw i32, ptr %29, i64 %50
  store i32 %51, ptr %53, align 4, !tbaa !49
  %54 = add nuw nsw i64 %50, 1
  %55 = icmp eq i64 %54, 800
  br i1 %55, label %56, label %49, !llvm.loop !57

56:                                               ; preds = %52
  call void @llvm.lifetime.end.p0(ptr nonnull %9) #20
  call void @llvm.lifetime.start.p0(ptr nonnull %8) #20
  store <2 x i32> <i32 0, i32 -1>, ptr %8, align 8, !tbaa !49
  br label %57

57:                                               ; preds = %60, %56
  %58 = phi i64 [ 0, %56 ], [ %62, %60 ]
  %59 = invoke noundef i32 @_ZNSt24uniform_int_distributionIjEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEjRT_RKNS0_10param_typeE(ptr noundef nonnull align 4 dereferenceable(8) %8, ptr noundef nonnull align 8 dereferenceable(5000) @_ZL3rng, ptr noundef nonnull align 4 dereferenceable(8) %8)
          to label %60 unwind label %89

60:                                               ; preds = %57
  %61 = getelementptr inbounds nuw i32, ptr %30, i64 %58
  store i32 %59, ptr %61, align 4, !tbaa !49
  %62 = add nuw nsw i64 %58, 1
  %63 = icmp eq i64 %62, 800
  br i1 %63, label %64, label %57, !llvm.loop !57

64:                                               ; preds = %60
  call void @llvm.lifetime.end.p0(ptr nonnull %8) #20
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 4 dereferenceable(3200) %32, ptr noundef nonnull align 4 dereferenceable(3200) %29, i64 3200, i1 false), !tbaa !49
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 4 dereferenceable(3200) %34, ptr noundef nonnull align 4 dereferenceable(3200) %29, i64 3200, i1 false), !tbaa !49
  %65 = getelementptr inbounds i32, ptr %36, i64 %48
  %66 = load i32, ptr %13, align 4, !tbaa !49
  call void @llvm.lifetime.start.p0(ptr nonnull %4)
  call void @llvm.lifetime.start.p0(ptr nonnull %5)
  call void @llvm.lifetime.start.p0(ptr nonnull %6)
  call void @llvm.lifetime.start.p0(ptr nonnull %7)
  store ptr %65, ptr %4, align 8, !tbaa !58
  store ptr %30, ptr %5, align 8, !tbaa !58
  store ptr %36, ptr %6, align 8, !tbaa !58
  store i32 %66, ptr %7, align 4, !tbaa !49
  %67 = load ptr, ptr %37, align 8, !tbaa !20
  %68 = icmp eq ptr %67, null
  br i1 %68, label %69, label %71

69:                                               ; preds = %64
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %70 unwind label %95

70:                                               ; preds = %69
  unreachable

71:                                               ; preds = %64
  %72 = load ptr, ptr %38, align 8, !tbaa !25
  invoke void %72(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %4, ptr noundef nonnull align 8 dereferenceable(8) %5, ptr noundef nonnull align 8 dereferenceable(8) %6, ptr noundef nonnull align 4 dereferenceable(4) %7)
          to label %73 unwind label %93

73:                                               ; preds = %71
  call void @llvm.lifetime.end.p0(ptr nonnull %4)
  call void @llvm.lifetime.end.p0(ptr nonnull %5)
  call void @llvm.lifetime.end.p0(ptr nonnull %6)
  call void @llvm.lifetime.end.p0(ptr nonnull %7)
  call void @llvm.lifetime.start.p0(ptr nonnull %10) #20
  store ptr %39, ptr %10, align 8, !tbaa !58
  call void @llvm.lifetime.start.p0(ptr nonnull %11) #20
  %74 = getelementptr inbounds i32, ptr %39, i64 %48
  store ptr %74, ptr %11, align 8, !tbaa !58
  call void @llvm.lifetime.start.p0(ptr nonnull %12) #20
  store ptr %30, ptr %12, align 8, !tbaa !58
  invoke fastcc void @_ZL18callThroughOptnoneIRSt8functionIFvPjS1_S1_jEEJS1_S1_RS1_RjEEvOT_DpOT0_(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %11, ptr noundef nonnull align 8 dereferenceable(8) %12, ptr noundef nonnull align 8 dereferenceable(8) %10, ptr noundef nonnull align 4 dereferenceable(4) %13)
          to label %75 unwind label %93

75:                                               ; preds = %73
  call void @llvm.lifetime.end.p0(ptr nonnull %12) #20
  call void @llvm.lifetime.end.p0(ptr nonnull %11) #20
  %76 = call i32 @bcmp(ptr noundef nonnull readonly dereferenceable(3200) %32, ptr noundef nonnull readonly dereferenceable(3200) %34, i64 3200)
  %77 = icmp eq i32 %76, 0
  br i1 %77, label %86, label %78

78:                                               ; preds = %75
  %79 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.50, i64 noundef 23)
          to label %80 unwind label %95

80:                                               ; preds = %78
  %81 = trunc nsw i64 %48 to i32
  %82 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEi(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, i32 noundef range(i32 -2147483648, 101) %81)
          to label %83 unwind label %95

83:                                               ; preds = %80
  %84 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) %82, ptr noundef nonnull @.str.49)
          to label %85 unwind label %95

85:                                               ; preds = %83
  call void @exit(i32 noundef 1) #25
  unreachable

86:                                               ; preds = %75
  call void @llvm.lifetime.end.p0(ptr nonnull %10) #20
  %87 = add nsw i64 %48, 1
  %88 = icmp eq i64 %87, 101
  br i1 %88, label %40, label %47, !llvm.loop !65

89:                                               ; preds = %57
  %90 = landingpad { ptr, i32 }
          cleanup
  br label %97

91:                                               ; preds = %49
  %92 = landingpad { ptr, i32 }
          cleanup
  br label %97

93:                                               ; preds = %71, %73
  %94 = landingpad { ptr, i32 }
          cleanup
  br label %97

95:                                               ; preds = %69, %80, %83, %78
  %96 = landingpad { ptr, i32 }
          cleanup
  br label %97

97:                                               ; preds = %91, %95, %93, %89
  %98 = phi { ptr, i32 } [ %90, %89 ], [ %92, %91 ], [ %94, %93 ], [ %96, %95 ]
  call void @_ZdaPv(ptr noundef nonnull %34) #23
  br label %99

99:                                               ; preds = %97, %45
  %100 = phi { ptr, i32 } [ %98, %97 ], [ %46, %45 ]
  call void @_ZdaPv(ptr noundef nonnull %32) #23
  br label %101

101:                                              ; preds = %99, %43
  %102 = phi { ptr, i32 } [ %100, %99 ], [ %44, %43 ]
  call void @_ZdaPv(ptr noundef nonnull %30) #23
  br label %103

103:                                              ; preds = %101, %41
  %104 = phi { ptr, i32 } [ %102, %101 ], [ %42, %41 ]
  call void @_ZdaPv(ptr noundef nonnull %29) #23
  call void @llvm.lifetime.end.p0(ptr nonnull %13) #20
  resume { ptr, i32 } %104
}

; Function Attrs: mustprogress norecurse uwtable
define internal fastcc void @_ZL38checkOverlappingMemoryTwoRuntimeChecksIhEvSt8functionIFvPT_S2_S2_jEES4_PKc(ptr noundef nonnull %0, ptr noundef nonnull %1, ptr noundef %2) unnamed_addr #0 personality ptr @__gxx_personality_v0 {
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca i32, align 4
  %8 = alloca %"class.std::uniform_int_distribution", align 4
  %9 = alloca %"class.std::uniform_int_distribution", align 4
  %10 = alloca ptr, align 8
  %11 = alloca ptr, align 8
  %12 = alloca ptr, align 8
  %13 = alloca i32, align 4
  %14 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.48, i64 noundef 9)
  %15 = icmp eq ptr %2, null
  br i1 %15, label %16, label %24

16:                                               ; preds = %3
  %17 = load ptr, ptr @_ZSt4cout, align 8, !tbaa !37
  %18 = getelementptr i8, ptr %17, i64 -24
  %19 = load i64, ptr %18, align 8
  %20 = getelementptr inbounds i8, ptr @_ZSt4cout, i64 %19
  %21 = getelementptr inbounds nuw i8, ptr %20, i64 32
  %22 = load i32, ptr %21, align 8, !tbaa !39
  %23 = or i32 %22, 1
  tail call void @_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate(ptr noundef nonnull align 8 dereferenceable(264) %20, i32 noundef %23)
  br label %27

24:                                               ; preds = %3
  %25 = tail call noundef i64 @strlen(ptr noundef nonnull dereferenceable(1) %2) #20
  %26 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull %2, i64 noundef %25)
  br label %27

27:                                               ; preds = %16, %24
  %28 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.49, i64 noundef 1)
  call void @llvm.lifetime.start.p0(ptr nonnull %13) #20
  store i32 100, ptr %13, align 4, !tbaa !49
  %29 = tail call noalias noundef nonnull dereferenceable(800) ptr @_Znam(i64 noundef 800) #22
  %30 = invoke noalias noundef nonnull dereferenceable(800) ptr @_Znam(i64 noundef 800) #22
          to label %31 unwind label %43

31:                                               ; preds = %27
  %32 = invoke noalias noundef nonnull dereferenceable(800) ptr @_Znam(i64 noundef 800) #22
          to label %33 unwind label %45

33:                                               ; preds = %31
  %34 = invoke noalias noundef nonnull dereferenceable(800) ptr @_Znam(i64 noundef 800) #22
          to label %35 unwind label %47

35:                                               ; preds = %33
  %36 = getelementptr inbounds nuw i8, ptr %9, i64 1
  %37 = getelementptr inbounds nuw i8, ptr %8, i64 1
  %38 = getelementptr inbounds nuw i8, ptr %32, i64 400
  %39 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %40 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %41 = getelementptr inbounds nuw i8, ptr %34, i64 400
  br label %49

42:                                               ; preds = %88
  call void @_ZdaPv(ptr noundef nonnull %34) #23
  call void @_ZdaPv(ptr noundef nonnull %32) #23
  call void @_ZdaPv(ptr noundef nonnull %30) #23
  call void @_ZdaPv(ptr noundef nonnull %29) #23
  call void @llvm.lifetime.end.p0(ptr nonnull %13) #20
  ret void

43:                                               ; preds = %27
  %44 = landingpad { ptr, i32 }
          cleanup
  br label %105

45:                                               ; preds = %31
  %46 = landingpad { ptr, i32 }
          cleanup
  br label %103

47:                                               ; preds = %33
  %48 = landingpad { ptr, i32 }
          cleanup
  br label %101

49:                                               ; preds = %35, %88
  %50 = phi i64 [ -100, %35 ], [ %89, %88 ]
  call void @llvm.lifetime.start.p0(ptr nonnull %9) #20
  store i8 0, ptr %9, align 4, !tbaa !50
  store i8 -1, ptr %36, align 1, !tbaa !52
  br label %51

51:                                               ; preds = %54, %49
  %52 = phi i64 [ 0, %49 ], [ %56, %54 ]
  %53 = invoke noundef i8 @_ZNSt24uniform_int_distributionIhEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEhRT_RKNS0_10param_typeE(ptr noundef nonnull align 1 dereferenceable(2) %9, ptr noundef nonnull align 8 dereferenceable(5000) @_ZL3rng, ptr noundef nonnull align 1 dereferenceable(2) %9)
          to label %54 unwind label %93

54:                                               ; preds = %51
  %55 = getelementptr inbounds nuw i8, ptr %29, i64 %52
  store i8 %53, ptr %55, align 1, !tbaa !15
  %56 = add nuw nsw i64 %52, 1
  %57 = icmp eq i64 %56, 800
  br i1 %57, label %58, label %51, !llvm.loop !53

58:                                               ; preds = %54
  call void @llvm.lifetime.end.p0(ptr nonnull %9) #20
  call void @llvm.lifetime.start.p0(ptr nonnull %8) #20
  store i8 0, ptr %8, align 4, !tbaa !50
  store i8 -1, ptr %37, align 1, !tbaa !52
  br label %59

59:                                               ; preds = %62, %58
  %60 = phi i64 [ 0, %58 ], [ %64, %62 ]
  %61 = invoke noundef i8 @_ZNSt24uniform_int_distributionIhEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEhRT_RKNS0_10param_typeE(ptr noundef nonnull align 1 dereferenceable(2) %8, ptr noundef nonnull align 8 dereferenceable(5000) @_ZL3rng, ptr noundef nonnull align 1 dereferenceable(2) %8)
          to label %62 unwind label %91

62:                                               ; preds = %59
  %63 = getelementptr inbounds nuw i8, ptr %30, i64 %60
  store i8 %61, ptr %63, align 1, !tbaa !15
  %64 = add nuw nsw i64 %60, 1
  %65 = icmp eq i64 %64, 800
  br i1 %65, label %66, label %59, !llvm.loop !53

66:                                               ; preds = %62
  call void @llvm.lifetime.end.p0(ptr nonnull %8) #20
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(800) %32, ptr noundef nonnull align 1 dereferenceable(800) %29, i64 800, i1 false), !tbaa !15
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(800) %34, ptr noundef nonnull align 1 dereferenceable(800) %29, i64 800, i1 false), !tbaa !15
  %67 = getelementptr inbounds i8, ptr %38, i64 %50
  %68 = load i32, ptr %13, align 4, !tbaa !49
  call void @llvm.lifetime.start.p0(ptr nonnull %4)
  call void @llvm.lifetime.start.p0(ptr nonnull %5)
  call void @llvm.lifetime.start.p0(ptr nonnull %6)
  call void @llvm.lifetime.start.p0(ptr nonnull %7)
  store ptr %67, ptr %4, align 8, !tbaa !54
  store ptr %30, ptr %5, align 8, !tbaa !54
  store ptr %38, ptr %6, align 8, !tbaa !54
  store i32 %68, ptr %7, align 4, !tbaa !49
  %69 = load ptr, ptr %39, align 8, !tbaa !20
  %70 = icmp eq ptr %69, null
  br i1 %70, label %71, label %73

71:                                               ; preds = %66
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %72 unwind label %97

72:                                               ; preds = %71
  unreachable

73:                                               ; preds = %66
  %74 = load ptr, ptr %40, align 8, !tbaa !27
  invoke void %74(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %4, ptr noundef nonnull align 8 dereferenceable(8) %5, ptr noundef nonnull align 8 dereferenceable(8) %6, ptr noundef nonnull align 4 dereferenceable(4) %7)
          to label %75 unwind label %95

75:                                               ; preds = %73
  call void @llvm.lifetime.end.p0(ptr nonnull %4)
  call void @llvm.lifetime.end.p0(ptr nonnull %5)
  call void @llvm.lifetime.end.p0(ptr nonnull %6)
  call void @llvm.lifetime.end.p0(ptr nonnull %7)
  call void @llvm.lifetime.start.p0(ptr nonnull %10) #20
  store ptr %41, ptr %10, align 8, !tbaa !54
  call void @llvm.lifetime.start.p0(ptr nonnull %11) #20
  %76 = getelementptr inbounds i8, ptr %41, i64 %50
  store ptr %76, ptr %11, align 8, !tbaa !54
  call void @llvm.lifetime.start.p0(ptr nonnull %12) #20
  store ptr %30, ptr %12, align 8, !tbaa !54
  invoke fastcc void @_ZL18callThroughOptnoneIRSt8functionIFvPhS1_S1_jEEJS1_S1_RS1_RjEEvOT_DpOT0_(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %11, ptr noundef nonnull align 8 dereferenceable(8) %12, ptr noundef nonnull align 8 dereferenceable(8) %10, ptr noundef nonnull align 4 dereferenceable(4) %13)
          to label %77 unwind label %95

77:                                               ; preds = %75
  call void @llvm.lifetime.end.p0(ptr nonnull %12) #20
  call void @llvm.lifetime.end.p0(ptr nonnull %11) #20
  %78 = call i32 @bcmp(ptr noundef nonnull readonly dereferenceable(800) %32, ptr noundef nonnull readonly dereferenceable(800) %34, i64 800)
  %79 = icmp eq i32 %78, 0
  br i1 %79, label %88, label %80

80:                                               ; preds = %77
  %81 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.50, i64 noundef 23)
          to label %82 unwind label %97

82:                                               ; preds = %80
  %83 = trunc nsw i64 %50 to i32
  %84 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEi(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, i32 noundef range(i32 -2147483648, 101) %83)
          to label %85 unwind label %97

85:                                               ; preds = %82
  %86 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) %84, ptr noundef nonnull @.str.49)
          to label %87 unwind label %97

87:                                               ; preds = %85
  call void @exit(i32 noundef 1) #25
  unreachable

88:                                               ; preds = %77
  call void @llvm.lifetime.end.p0(ptr nonnull %10) #20
  %89 = add nsw i64 %50, 1
  %90 = icmp eq i64 %89, 101
  br i1 %90, label %42, label %49, !llvm.loop !66

91:                                               ; preds = %59
  %92 = landingpad { ptr, i32 }
          cleanup
  br label %99

93:                                               ; preds = %51
  %94 = landingpad { ptr, i32 }
          cleanup
  br label %99

95:                                               ; preds = %73, %75
  %96 = landingpad { ptr, i32 }
          cleanup
  br label %99

97:                                               ; preds = %71, %82, %85, %80
  %98 = landingpad { ptr, i32 }
          cleanup
  br label %99

99:                                               ; preds = %93, %97, %95, %91
  %100 = phi { ptr, i32 } [ %92, %91 ], [ %94, %93 ], [ %96, %95 ], [ %98, %97 ]
  call void @_ZdaPv(ptr noundef nonnull %34) #23
  br label %101

101:                                              ; preds = %47, %99
  %102 = phi { ptr, i32 } [ %100, %99 ], [ %48, %47 ]
  call void @_ZdaPv(ptr noundef nonnull %32) #23
  br label %103

103:                                              ; preds = %45, %101
  %104 = phi { ptr, i32 } [ %46, %45 ], [ %102, %101 ]
  call void @_ZdaPv(ptr noundef nonnull %30) #23
  br label %105

105:                                              ; preds = %103, %43
  %106 = phi { ptr, i32 } [ %104, %103 ], [ %44, %43 ]
  call void @_ZdaPv(ptr noundef nonnull %29) #23
  call void @llvm.lifetime.end.p0(ptr nonnull %13) #20
  resume { ptr, i32 } %106
}

; Function Attrs: mustprogress norecurse uwtable
define internal fastcc void @_ZL38checkOverlappingMemoryTwoRuntimeChecksImEvSt8functionIFvPT_S2_S2_jEES4_PKc(ptr noundef nonnull %0, ptr noundef nonnull %1, ptr noundef %2) unnamed_addr #0 personality ptr @__gxx_personality_v0 {
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca i32, align 4
  %8 = alloca %"class.std::uniform_int_distribution.96", align 16
  %9 = alloca %"class.std::uniform_int_distribution.96", align 16
  %10 = alloca ptr, align 8
  %11 = alloca ptr, align 8
  %12 = alloca ptr, align 8
  %13 = alloca i32, align 4
  %14 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.48, i64 noundef 9)
  %15 = icmp eq ptr %2, null
  br i1 %15, label %16, label %24

16:                                               ; preds = %3
  %17 = load ptr, ptr @_ZSt4cout, align 8, !tbaa !37
  %18 = getelementptr i8, ptr %17, i64 -24
  %19 = load i64, ptr %18, align 8
  %20 = getelementptr inbounds i8, ptr @_ZSt4cout, i64 %19
  %21 = getelementptr inbounds nuw i8, ptr %20, i64 32
  %22 = load i32, ptr %21, align 8, !tbaa !39
  %23 = or i32 %22, 1
  tail call void @_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate(ptr noundef nonnull align 8 dereferenceable(264) %20, i32 noundef %23)
  br label %27

24:                                               ; preds = %3
  %25 = tail call noundef i64 @strlen(ptr noundef nonnull dereferenceable(1) %2) #20
  %26 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull %2, i64 noundef %25)
  br label %27

27:                                               ; preds = %16, %24
  %28 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.49, i64 noundef 1)
  call void @llvm.lifetime.start.p0(ptr nonnull %13) #20
  store i32 100, ptr %13, align 4, !tbaa !49
  %29 = tail call noalias noundef nonnull dereferenceable(6400) ptr @_Znam(i64 noundef 6400) #22
  %30 = invoke noalias noundef nonnull dereferenceable(6400) ptr @_Znam(i64 noundef 6400) #22
          to label %31 unwind label %41

31:                                               ; preds = %27
  %32 = invoke noalias noundef nonnull dereferenceable(6400) ptr @_Znam(i64 noundef 6400) #22
          to label %33 unwind label %43

33:                                               ; preds = %31
  %34 = invoke noalias noundef nonnull dereferenceable(6400) ptr @_Znam(i64 noundef 6400) #22
          to label %35 unwind label %45

35:                                               ; preds = %33
  %36 = getelementptr inbounds nuw i8, ptr %32, i64 3200
  %37 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %38 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %39 = getelementptr inbounds nuw i8, ptr %34, i64 3200
  br label %47

40:                                               ; preds = %86
  call void @_ZdaPv(ptr noundef nonnull %34) #23
  call void @_ZdaPv(ptr noundef nonnull %32) #23
  call void @_ZdaPv(ptr noundef nonnull %30) #23
  call void @_ZdaPv(ptr noundef nonnull %29) #23
  call void @llvm.lifetime.end.p0(ptr nonnull %13) #20
  ret void

41:                                               ; preds = %27
  %42 = landingpad { ptr, i32 }
          cleanup
  br label %103

43:                                               ; preds = %31
  %44 = landingpad { ptr, i32 }
          cleanup
  br label %101

45:                                               ; preds = %33
  %46 = landingpad { ptr, i32 }
          cleanup
  br label %99

47:                                               ; preds = %35, %86
  %48 = phi i64 [ -100, %35 ], [ %87, %86 ]
  call void @llvm.lifetime.start.p0(ptr nonnull %9) #20
  store <2 x i64> <i64 0, i64 -1>, ptr %9, align 16, !tbaa !6
  br label %49

49:                                               ; preds = %52, %47
  %50 = phi i64 [ 0, %47 ], [ %54, %52 ]
  %51 = invoke noundef i64 @_ZNSt24uniform_int_distributionImEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEmRT_RKNS0_10param_typeE(ptr noundef nonnull align 8 dereferenceable(16) %9, ptr noundef nonnull align 8 dereferenceable(5000) @_ZL3rng, ptr noundef nonnull align 8 dereferenceable(16) %9)
          to label %52 unwind label %91

52:                                               ; preds = %49
  %53 = getelementptr inbounds nuw i64, ptr %29, i64 %50
  store i64 %51, ptr %53, align 8, !tbaa !6
  %54 = add nuw nsw i64 %50, 1
  %55 = icmp eq i64 %54, 800
  br i1 %55, label %56, label %49, !llvm.loop !61

56:                                               ; preds = %52
  call void @llvm.lifetime.end.p0(ptr nonnull %9) #20
  call void @llvm.lifetime.start.p0(ptr nonnull %8) #20
  store <2 x i64> <i64 0, i64 -1>, ptr %8, align 16, !tbaa !6
  br label %57

57:                                               ; preds = %60, %56
  %58 = phi i64 [ 0, %56 ], [ %62, %60 ]
  %59 = invoke noundef i64 @_ZNSt24uniform_int_distributionImEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEmRT_RKNS0_10param_typeE(ptr noundef nonnull align 8 dereferenceable(16) %8, ptr noundef nonnull align 8 dereferenceable(5000) @_ZL3rng, ptr noundef nonnull align 8 dereferenceable(16) %8)
          to label %60 unwind label %89

60:                                               ; preds = %57
  %61 = getelementptr inbounds nuw i64, ptr %30, i64 %58
  store i64 %59, ptr %61, align 8, !tbaa !6
  %62 = add nuw nsw i64 %58, 1
  %63 = icmp eq i64 %62, 800
  br i1 %63, label %64, label %57, !llvm.loop !61

64:                                               ; preds = %60
  call void @llvm.lifetime.end.p0(ptr nonnull %8) #20
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(6400) %32, ptr noundef nonnull align 8 dereferenceable(6400) %29, i64 6400, i1 false), !tbaa !6
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(6400) %34, ptr noundef nonnull align 8 dereferenceable(6400) %29, i64 6400, i1 false), !tbaa !6
  %65 = getelementptr inbounds i64, ptr %36, i64 %48
  %66 = load i32, ptr %13, align 4, !tbaa !49
  call void @llvm.lifetime.start.p0(ptr nonnull %4)
  call void @llvm.lifetime.start.p0(ptr nonnull %5)
  call void @llvm.lifetime.start.p0(ptr nonnull %6)
  call void @llvm.lifetime.start.p0(ptr nonnull %7)
  store ptr %65, ptr %4, align 8, !tbaa !62
  store ptr %30, ptr %5, align 8, !tbaa !62
  store ptr %36, ptr %6, align 8, !tbaa !62
  store i32 %66, ptr %7, align 4, !tbaa !49
  %67 = load ptr, ptr %37, align 8, !tbaa !20
  %68 = icmp eq ptr %67, null
  br i1 %68, label %69, label %71

69:                                               ; preds = %64
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %70 unwind label %95

70:                                               ; preds = %69
  unreachable

71:                                               ; preds = %64
  %72 = load ptr, ptr %38, align 8, !tbaa !29
  invoke void %72(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %4, ptr noundef nonnull align 8 dereferenceable(8) %5, ptr noundef nonnull align 8 dereferenceable(8) %6, ptr noundef nonnull align 4 dereferenceable(4) %7)
          to label %73 unwind label %93

73:                                               ; preds = %71
  call void @llvm.lifetime.end.p0(ptr nonnull %4)
  call void @llvm.lifetime.end.p0(ptr nonnull %5)
  call void @llvm.lifetime.end.p0(ptr nonnull %6)
  call void @llvm.lifetime.end.p0(ptr nonnull %7)
  call void @llvm.lifetime.start.p0(ptr nonnull %10) #20
  store ptr %39, ptr %10, align 8, !tbaa !62
  call void @llvm.lifetime.start.p0(ptr nonnull %11) #20
  %74 = getelementptr inbounds i64, ptr %39, i64 %48
  store ptr %74, ptr %11, align 8, !tbaa !62
  call void @llvm.lifetime.start.p0(ptr nonnull %12) #20
  store ptr %30, ptr %12, align 8, !tbaa !62
  invoke fastcc void @_ZL18callThroughOptnoneIRSt8functionIFvPmS1_S1_jEEJS1_S1_RS1_RjEEvOT_DpOT0_(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %11, ptr noundef nonnull align 8 dereferenceable(8) %12, ptr noundef nonnull align 8 dereferenceable(8) %10, ptr noundef nonnull align 4 dereferenceable(4) %13)
          to label %75 unwind label %93

75:                                               ; preds = %73
  call void @llvm.lifetime.end.p0(ptr nonnull %12) #20
  call void @llvm.lifetime.end.p0(ptr nonnull %11) #20
  %76 = call i32 @bcmp(ptr noundef nonnull readonly dereferenceable(6400) %32, ptr noundef nonnull readonly dereferenceable(6400) %34, i64 6400)
  %77 = icmp eq i32 %76, 0
  br i1 %77, label %86, label %78

78:                                               ; preds = %75
  %79 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.50, i64 noundef 23)
          to label %80 unwind label %95

80:                                               ; preds = %78
  %81 = trunc nsw i64 %48 to i32
  %82 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEi(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, i32 noundef range(i32 -2147483648, 101) %81)
          to label %83 unwind label %95

83:                                               ; preds = %80
  %84 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) %82, ptr noundef nonnull @.str.49)
          to label %85 unwind label %95

85:                                               ; preds = %83
  call void @exit(i32 noundef 1) #25
  unreachable

86:                                               ; preds = %75
  call void @llvm.lifetime.end.p0(ptr nonnull %10) #20
  %87 = add nsw i64 %48, 1
  %88 = icmp eq i64 %87, 101
  br i1 %88, label %40, label %47, !llvm.loop !67

89:                                               ; preds = %57
  %90 = landingpad { ptr, i32 }
          cleanup
  br label %97

91:                                               ; preds = %49
  %92 = landingpad { ptr, i32 }
          cleanup
  br label %97

93:                                               ; preds = %71, %73
  %94 = landingpad { ptr, i32 }
          cleanup
  br label %97

95:                                               ; preds = %69, %80, %83, %78
  %96 = landingpad { ptr, i32 }
          cleanup
  br label %97

97:                                               ; preds = %91, %95, %93, %89
  %98 = phi { ptr, i32 } [ %90, %89 ], [ %92, %91 ], [ %94, %93 ], [ %96, %95 ]
  call void @_ZdaPv(ptr noundef nonnull %34) #23
  br label %99

99:                                               ; preds = %97, %45
  %100 = phi { ptr, i32 } [ %98, %97 ], [ %46, %45 ]
  call void @_ZdaPv(ptr noundef nonnull %32) #23
  br label %101

101:                                              ; preds = %99, %43
  %102 = phi { ptr, i32 } [ %100, %99 ], [ %44, %43 ]
  call void @_ZdaPv(ptr noundef nonnull %30) #23
  br label %103

103:                                              ; preds = %101, %41
  %104 = phi { ptr, i32 } [ %102, %101 ], [ %42, %41 ]
  call void @_ZdaPv(ptr noundef nonnull %29) #23
  call void @llvm.lifetime.end.p0(ptr nonnull %13) #20
  resume { ptr, i32 } %104
}

; Function Attrs: mustprogress norecurse uwtable
define internal fastcc void @_ZL44checkOverlappingMemoryTwoRuntimeChecksNestedIhEvSt8functionIFvPT_S2_jjEES4_iiPKc(ptr noundef nonnull %0, ptr noundef nonnull %1, i32 noundef range(i32 50, 101) %2, ptr noundef %3) unnamed_addr #0 personality ptr @__gxx_personality_v0 {
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca i32, align 4
  %8 = alloca i32, align 4
  %9 = alloca %"class.std::uniform_int_distribution", align 4
  %10 = alloca ptr, align 8
  %11 = alloca ptr, align 8
  %12 = alloca i32, align 4
  %13 = alloca i32, align 4
  store i32 100, ptr %12, align 4, !tbaa !49
  store i32 %2, ptr %13, align 4, !tbaa !49
  %14 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.48, i64 noundef 9)
  %15 = icmp eq ptr %3, null
  br i1 %15, label %16, label %24

16:                                               ; preds = %4
  %17 = load ptr, ptr @_ZSt4cout, align 8, !tbaa !37
  %18 = getelementptr i8, ptr %17, i64 -24
  %19 = load i64, ptr %18, align 8
  %20 = getelementptr inbounds i8, ptr @_ZSt4cout, i64 %19
  %21 = getelementptr inbounds nuw i8, ptr %20, i64 32
  %22 = load i32, ptr %21, align 8, !tbaa !39
  %23 = or i32 %22, 1
  tail call void @_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate(ptr noundef nonnull align 8 dereferenceable(264) %20, i32 noundef %23)
  br label %27

24:                                               ; preds = %4
  %25 = tail call noundef i64 @strlen(ptr noundef nonnull dereferenceable(1) %3) #20
  %26 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull %3, i64 noundef %25)
  br label %27

27:                                               ; preds = %16, %24
  %28 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.49, i64 noundef 1)
  %29 = mul nuw nsw i32 %2, 808
  %30 = zext nneg i32 %29 to i64
  %31 = tail call noalias noundef nonnull ptr @_Znam(i64 noundef %30) #22
  %32 = invoke noalias noundef nonnull ptr @_Znam(i64 noundef %30) #22
          to label %33 unwind label %47

33:                                               ; preds = %27
  %34 = invoke noalias noundef nonnull ptr @_Znam(i64 noundef %30) #22
          to label %35 unwind label %49

35:                                               ; preds = %33
  %36 = shl nuw nsw i32 %2, 1
  %37 = sub nuw nsw i32 -2, %36
  %38 = getelementptr inbounds nuw i8, ptr %9, i64 1
  %39 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %40 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %41 = lshr exact i32 %29, 1
  %42 = zext nneg i32 %41 to i64
  %43 = getelementptr inbounds nuw i8, ptr %34, i64 %42
  %44 = getelementptr inbounds nuw i8, ptr %32, i64 %42
  %45 = sext i32 %37 to i64
  br label %51

46:                                               ; preds = %83
  call void @_ZdaPv(ptr noundef nonnull %34) #23
  call void @_ZdaPv(ptr noundef nonnull %32) #23
  call void @_ZdaPv(ptr noundef nonnull %31) #23
  ret void

47:                                               ; preds = %27
  %48 = landingpad { ptr, i32 }
          cleanup
  br label %100

49:                                               ; preds = %33
  %50 = landingpad { ptr, i32 }
          cleanup
  br label %98

51:                                               ; preds = %35, %83
  %52 = phi i64 [ %45, %35 ], [ %84, %83 ]
  call void @llvm.lifetime.start.p0(ptr nonnull %9) #20
  store i8 0, ptr %9, align 4, !tbaa !50
  store i8 -1, ptr %38, align 1, !tbaa !52
  br label %53

53:                                               ; preds = %51, %56
  %54 = phi i64 [ %58, %56 ], [ 0, %51 ]
  %55 = invoke noundef i8 @_ZNSt24uniform_int_distributionIhEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEhRT_RKNS0_10param_typeE(ptr noundef nonnull align 1 dereferenceable(2) %9, ptr noundef nonnull align 8 dereferenceable(5000) @_ZL3rng, ptr noundef nonnull align 1 dereferenceable(2) %9)
          to label %56 unwind label %90

56:                                               ; preds = %53
  %57 = getelementptr inbounds nuw i8, ptr %31, i64 %54
  store i8 %55, ptr %57, align 1, !tbaa !15
  %58 = add nuw nsw i64 %54, 1
  %59 = icmp eq i64 %58, %30
  br i1 %59, label %60, label %53, !llvm.loop !53

60:                                               ; preds = %56
  call void @llvm.lifetime.end.p0(ptr nonnull %9) #20
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(1) %32, ptr noundef nonnull align 1 dereferenceable(1) %31, i64 %30, i1 false), !tbaa !15
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(1) %34, ptr noundef nonnull align 1 dereferenceable(1) %31, i64 %30, i1 false), !tbaa !15
  %61 = getelementptr inbounds i8, ptr %44, i64 %52
  %62 = load i32, ptr %12, align 4, !tbaa !49
  %63 = load i32, ptr %13, align 4, !tbaa !49
  call void @llvm.lifetime.start.p0(ptr nonnull %5)
  call void @llvm.lifetime.start.p0(ptr nonnull %6)
  call void @llvm.lifetime.start.p0(ptr nonnull %7)
  call void @llvm.lifetime.start.p0(ptr nonnull %8)
  store ptr %61, ptr %5, align 8, !tbaa !54
  store ptr %44, ptr %6, align 8, !tbaa !54
  store i32 %62, ptr %7, align 4, !tbaa !49
  store i32 %63, ptr %8, align 4, !tbaa !49
  %64 = load ptr, ptr %39, align 8, !tbaa !20
  %65 = icmp eq ptr %64, null
  br i1 %65, label %66, label %68

66:                                               ; preds = %60
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %67 unwind label %94

67:                                               ; preds = %66
  unreachable

68:                                               ; preds = %60
  %69 = load ptr, ptr %40, align 8, !tbaa !31
  invoke void %69(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %5, ptr noundef nonnull align 8 dereferenceable(8) %6, ptr noundef nonnull align 4 dereferenceable(4) %7, ptr noundef nonnull align 4 dereferenceable(4) %8)
          to label %70 unwind label %92

70:                                               ; preds = %68
  call void @llvm.lifetime.end.p0(ptr nonnull %5)
  call void @llvm.lifetime.end.p0(ptr nonnull %6)
  call void @llvm.lifetime.end.p0(ptr nonnull %7)
  call void @llvm.lifetime.end.p0(ptr nonnull %8)
  call void @llvm.lifetime.start.p0(ptr nonnull %10) #20
  store ptr %43, ptr %10, align 8, !tbaa !54
  call void @llvm.lifetime.start.p0(ptr nonnull %11) #20
  %71 = getelementptr inbounds i8, ptr %43, i64 %52
  store ptr %71, ptr %11, align 8, !tbaa !54
  invoke fastcc void @_ZL18callThroughOptnoneIRSt8functionIFvPhS1_jjEEJS1_RS1_RKiS7_EEvOT_DpOT0_(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %11, ptr noundef nonnull align 8 dereferenceable(8) %10, ptr noundef nonnull align 4 dereferenceable(4) %12, ptr noundef nonnull align 4 dereferenceable(4) %13)
          to label %72 unwind label %92

72:                                               ; preds = %70
  call void @llvm.lifetime.end.p0(ptr nonnull %11) #20
  %73 = call i32 @bcmp(ptr noundef nonnull readonly dereferenceable(1) %32, ptr noundef nonnull readonly dereferenceable(1) %34, i64 %30)
  %74 = icmp eq i32 %73, 0
  br i1 %74, label %83, label %75

75:                                               ; preds = %72
  %76 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.50, i64 noundef 23)
          to label %77 unwind label %94

77:                                               ; preds = %75
  %78 = trunc nsw i64 %52 to i32
  %79 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEi(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, i32 noundef range(i32 -2147483647, -2147483648) %78)
          to label %80 unwind label %94

80:                                               ; preds = %77
  %81 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) %79, ptr noundef nonnull @.str.49)
          to label %82 unwind label %94

82:                                               ; preds = %80
  call void @exit(i32 noundef 1) #25
  unreachable

83:                                               ; preds = %72
  call void @llvm.lifetime.end.p0(ptr nonnull %10) #20
  %84 = add nsw i64 %52, 1
  %85 = load i32, ptr %13, align 4, !tbaa !49
  %86 = shl i32 %85, 1
  %87 = add i32 %86, 2
  %88 = sext i32 %87 to i64
  %89 = icmp slt i64 %52, %88
  br i1 %89, label %51, label %46, !llvm.loop !68

90:                                               ; preds = %53
  %91 = landingpad { ptr, i32 }
          cleanup
  br label %96

92:                                               ; preds = %70, %68
  %93 = landingpad { ptr, i32 }
          cleanup
  br label %96

94:                                               ; preds = %75, %80, %77, %66
  %95 = landingpad { ptr, i32 }
          cleanup
  br label %96

96:                                               ; preds = %94, %92, %90
  %97 = phi { ptr, i32 } [ %91, %90 ], [ %95, %94 ], [ %93, %92 ]
  call void @_ZdaPv(ptr noundef nonnull %34) #23
  br label %98

98:                                               ; preds = %96, %49
  %99 = phi { ptr, i32 } [ %97, %96 ], [ %50, %49 ]
  call void @_ZdaPv(ptr noundef nonnull %32) #23
  br label %100

100:                                              ; preds = %98, %47
  %101 = phi { ptr, i32 } [ %99, %98 ], [ %48, %47 ]
  call void @_ZdaPv(ptr noundef nonnull %31) #23
  resume { ptr, i32 } %101
}

; Function Attrs: mustprogress norecurse uwtable
define internal fastcc void @_ZL44checkOverlappingMemoryTwoRuntimeChecksNestedIjEvSt8functionIFvPT_S2_jjEES4_iiPKc(ptr noundef nonnull %0, ptr noundef nonnull %1, i32 noundef range(i32 50, 101) %2, ptr noundef %3) unnamed_addr #0 personality ptr @__gxx_personality_v0 {
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca i32, align 4
  %8 = alloca i32, align 4
  %9 = alloca %"class.std::uniform_int_distribution.84", align 8
  %10 = alloca ptr, align 8
  %11 = alloca ptr, align 8
  %12 = alloca i32, align 4
  %13 = alloca i32, align 4
  store i32 100, ptr %12, align 4, !tbaa !49
  store i32 %2, ptr %13, align 4, !tbaa !49
  %14 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.48, i64 noundef 9)
  %15 = icmp eq ptr %3, null
  br i1 %15, label %16, label %24

16:                                               ; preds = %4
  %17 = load ptr, ptr @_ZSt4cout, align 8, !tbaa !37
  %18 = getelementptr i8, ptr %17, i64 -24
  %19 = load i64, ptr %18, align 8
  %20 = getelementptr inbounds i8, ptr @_ZSt4cout, i64 %19
  %21 = getelementptr inbounds nuw i8, ptr %20, i64 32
  %22 = load i32, ptr %21, align 8, !tbaa !39
  %23 = or i32 %22, 1
  tail call void @_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate(ptr noundef nonnull align 8 dereferenceable(264) %20, i32 noundef %23)
  br label %27

24:                                               ; preds = %4
  %25 = tail call noundef i64 @strlen(ptr noundef nonnull dereferenceable(1) %3) #20
  %26 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull %3, i64 noundef %25)
  br label %27

27:                                               ; preds = %16, %24
  %28 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.49, i64 noundef 1)
  %29 = mul nuw nsw i32 %2, 808
  %30 = zext nneg i32 %29 to i64
  %31 = shl nuw nsw i64 %30, 2
  %32 = tail call noalias noundef nonnull ptr @_Znam(i64 noundef %31) #22
  %33 = invoke noalias noundef nonnull ptr @_Znam(i64 noundef %31) #22
          to label %34 unwind label %47

34:                                               ; preds = %27
  %35 = invoke noalias noundef nonnull ptr @_Znam(i64 noundef %31) #22
          to label %36 unwind label %49

36:                                               ; preds = %34
  %37 = shl nuw nsw i32 %2, 1
  %38 = sub nuw nsw i32 -2, %37
  %39 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %40 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %41 = lshr exact i32 %29, 1
  %42 = zext nneg i32 %41 to i64
  %43 = getelementptr inbounds nuw i32, ptr %35, i64 %42
  %44 = getelementptr inbounds nuw i32, ptr %33, i64 %42
  %45 = sext i32 %38 to i64
  br label %51

46:                                               ; preds = %83
  call void @_ZdaPv(ptr noundef nonnull %35) #23
  call void @_ZdaPv(ptr noundef nonnull %33) #23
  call void @_ZdaPv(ptr noundef nonnull %32) #23
  ret void

47:                                               ; preds = %27
  %48 = landingpad { ptr, i32 }
          cleanup
  br label %100

49:                                               ; preds = %34
  %50 = landingpad { ptr, i32 }
          cleanup
  br label %98

51:                                               ; preds = %36, %83
  %52 = phi i64 [ %45, %36 ], [ %84, %83 ]
  call void @llvm.lifetime.start.p0(ptr nonnull %9) #20
  store <2 x i32> <i32 0, i32 -1>, ptr %9, align 8, !tbaa !49
  br label %53

53:                                               ; preds = %51, %56
  %54 = phi i64 [ %58, %56 ], [ 0, %51 ]
  %55 = invoke noundef i32 @_ZNSt24uniform_int_distributionIjEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEjRT_RKNS0_10param_typeE(ptr noundef nonnull align 4 dereferenceable(8) %9, ptr noundef nonnull align 8 dereferenceable(5000) @_ZL3rng, ptr noundef nonnull align 4 dereferenceable(8) %9)
          to label %56 unwind label %90

56:                                               ; preds = %53
  %57 = getelementptr inbounds nuw i32, ptr %32, i64 %54
  store i32 %55, ptr %57, align 4, !tbaa !49
  %58 = add nuw nsw i64 %54, 1
  %59 = icmp eq i64 %58, %30
  br i1 %59, label %60, label %53, !llvm.loop !57

60:                                               ; preds = %56
  call void @llvm.lifetime.end.p0(ptr nonnull %9) #20
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 4 dereferenceable(1) %33, ptr noundef nonnull align 4 dereferenceable(1) %32, i64 %31, i1 false), !tbaa !49
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 4 dereferenceable(1) %35, ptr noundef nonnull align 4 dereferenceable(1) %32, i64 %31, i1 false), !tbaa !49
  %61 = getelementptr inbounds i32, ptr %44, i64 %52
  %62 = load i32, ptr %12, align 4, !tbaa !49
  %63 = load i32, ptr %13, align 4, !tbaa !49
  call void @llvm.lifetime.start.p0(ptr nonnull %5)
  call void @llvm.lifetime.start.p0(ptr nonnull %6)
  call void @llvm.lifetime.start.p0(ptr nonnull %7)
  call void @llvm.lifetime.start.p0(ptr nonnull %8)
  store ptr %61, ptr %5, align 8, !tbaa !58
  store ptr %44, ptr %6, align 8, !tbaa !58
  store i32 %62, ptr %7, align 4, !tbaa !49
  store i32 %63, ptr %8, align 4, !tbaa !49
  %64 = load ptr, ptr %39, align 8, !tbaa !20
  %65 = icmp eq ptr %64, null
  br i1 %65, label %66, label %68

66:                                               ; preds = %60
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %67 unwind label %94

67:                                               ; preds = %66
  unreachable

68:                                               ; preds = %60
  %69 = load ptr, ptr %40, align 8, !tbaa !33
  invoke void %69(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %5, ptr noundef nonnull align 8 dereferenceable(8) %6, ptr noundef nonnull align 4 dereferenceable(4) %7, ptr noundef nonnull align 4 dereferenceable(4) %8)
          to label %70 unwind label %92

70:                                               ; preds = %68
  call void @llvm.lifetime.end.p0(ptr nonnull %5)
  call void @llvm.lifetime.end.p0(ptr nonnull %6)
  call void @llvm.lifetime.end.p0(ptr nonnull %7)
  call void @llvm.lifetime.end.p0(ptr nonnull %8)
  call void @llvm.lifetime.start.p0(ptr nonnull %10) #20
  store ptr %43, ptr %10, align 8, !tbaa !58
  call void @llvm.lifetime.start.p0(ptr nonnull %11) #20
  %71 = getelementptr inbounds i32, ptr %43, i64 %52
  store ptr %71, ptr %11, align 8, !tbaa !58
  invoke fastcc void @_ZL18callThroughOptnoneIRSt8functionIFvPjS1_jjEEJS1_RS1_RKiS7_EEvOT_DpOT0_(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %11, ptr noundef nonnull align 8 dereferenceable(8) %10, ptr noundef nonnull align 4 dereferenceable(4) %12, ptr noundef nonnull align 4 dereferenceable(4) %13)
          to label %72 unwind label %92

72:                                               ; preds = %70
  call void @llvm.lifetime.end.p0(ptr nonnull %11) #20
  %73 = call i32 @bcmp(ptr noundef nonnull readonly dereferenceable(1) %33, ptr noundef nonnull readonly dereferenceable(1) %35, i64 %31)
  %74 = icmp eq i32 %73, 0
  br i1 %74, label %83, label %75

75:                                               ; preds = %72
  %76 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.50, i64 noundef 23)
          to label %77 unwind label %94

77:                                               ; preds = %75
  %78 = trunc nsw i64 %52 to i32
  %79 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEi(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, i32 noundef range(i32 -2147483647, -2147483648) %78)
          to label %80 unwind label %94

80:                                               ; preds = %77
  %81 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) %79, ptr noundef nonnull @.str.49)
          to label %82 unwind label %94

82:                                               ; preds = %80
  call void @exit(i32 noundef 1) #25
  unreachable

83:                                               ; preds = %72
  call void @llvm.lifetime.end.p0(ptr nonnull %10) #20
  %84 = add nsw i64 %52, 1
  %85 = load i32, ptr %13, align 4, !tbaa !49
  %86 = shl i32 %85, 1
  %87 = add i32 %86, 2
  %88 = sext i32 %87 to i64
  %89 = icmp slt i64 %52, %88
  br i1 %89, label %51, label %46, !llvm.loop !69

90:                                               ; preds = %53
  %91 = landingpad { ptr, i32 }
          cleanup
  br label %96

92:                                               ; preds = %70, %68
  %93 = landingpad { ptr, i32 }
          cleanup
  br label %96

94:                                               ; preds = %75, %80, %77, %66
  %95 = landingpad { ptr, i32 }
          cleanup
  br label %96

96:                                               ; preds = %94, %92, %90
  %97 = phi { ptr, i32 } [ %91, %90 ], [ %95, %94 ], [ %93, %92 ]
  call void @_ZdaPv(ptr noundef nonnull %35) #23
  br label %98

98:                                               ; preds = %96, %49
  %99 = phi { ptr, i32 } [ %97, %96 ], [ %50, %49 ]
  call void @_ZdaPv(ptr noundef nonnull %33) #23
  br label %100

100:                                              ; preds = %98, %47
  %101 = phi { ptr, i32 } [ %99, %98 ], [ %48, %47 ]
  call void @_ZdaPv(ptr noundef nonnull %32) #23
  resume { ptr, i32 } %101
}

; Function Attrs: mustprogress norecurse uwtable
define internal fastcc void @_ZL44checkOverlappingMemoryTwoRuntimeChecksNestedImEvSt8functionIFvPT_S2_jjEES4_iiPKc(ptr noundef nonnull %0, ptr noundef nonnull %1, i32 noundef range(i32 50, 101) %2, ptr noundef %3) unnamed_addr #0 personality ptr @__gxx_personality_v0 {
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca i32, align 4
  %8 = alloca i32, align 4
  %9 = alloca %"class.std::uniform_int_distribution.96", align 16
  %10 = alloca ptr, align 8
  %11 = alloca ptr, align 8
  %12 = alloca i32, align 4
  %13 = alloca i32, align 4
  store i32 100, ptr %12, align 4, !tbaa !49
  store i32 %2, ptr %13, align 4, !tbaa !49
  %14 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.48, i64 noundef 9)
  %15 = icmp eq ptr %3, null
  br i1 %15, label %16, label %24

16:                                               ; preds = %4
  %17 = load ptr, ptr @_ZSt4cout, align 8, !tbaa !37
  %18 = getelementptr i8, ptr %17, i64 -24
  %19 = load i64, ptr %18, align 8
  %20 = getelementptr inbounds i8, ptr @_ZSt4cout, i64 %19
  %21 = getelementptr inbounds nuw i8, ptr %20, i64 32
  %22 = load i32, ptr %21, align 8, !tbaa !39
  %23 = or i32 %22, 1
  tail call void @_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate(ptr noundef nonnull align 8 dereferenceable(264) %20, i32 noundef %23)
  br label %27

24:                                               ; preds = %4
  %25 = tail call noundef i64 @strlen(ptr noundef nonnull dereferenceable(1) %3) #20
  %26 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull %3, i64 noundef %25)
  br label %27

27:                                               ; preds = %16, %24
  %28 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.49, i64 noundef 1)
  %29 = mul nuw nsw i32 %2, 808
  %30 = zext nneg i32 %29 to i64
  %31 = shl nuw nsw i64 %30, 3
  %32 = tail call noalias noundef nonnull ptr @_Znam(i64 noundef %31) #22
  %33 = invoke noalias noundef nonnull ptr @_Znam(i64 noundef %31) #22
          to label %34 unwind label %47

34:                                               ; preds = %27
  %35 = invoke noalias noundef nonnull ptr @_Znam(i64 noundef %31) #22
          to label %36 unwind label %49

36:                                               ; preds = %34
  %37 = shl nuw nsw i32 %2, 1
  %38 = sub nuw nsw i32 -2, %37
  %39 = lshr exact i32 %29, 1
  %40 = zext nneg i32 %39 to i64
  %41 = getelementptr inbounds nuw i64, ptr %33, i64 %40
  %42 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %43 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %44 = getelementptr inbounds nuw i64, ptr %35, i64 %40
  %45 = sext i32 %38 to i64
  br label %51

46:                                               ; preds = %83
  call void @_ZdaPv(ptr noundef nonnull %35) #23
  call void @_ZdaPv(ptr noundef nonnull %33) #23
  call void @_ZdaPv(ptr noundef nonnull %32) #23
  ret void

47:                                               ; preds = %27
  %48 = landingpad { ptr, i32 }
          cleanup
  br label %100

49:                                               ; preds = %34
  %50 = landingpad { ptr, i32 }
          cleanup
  br label %98

51:                                               ; preds = %36, %83
  %52 = phi i64 [ %45, %36 ], [ %84, %83 ]
  call void @llvm.lifetime.start.p0(ptr nonnull %9) #20
  store <2 x i64> <i64 0, i64 -1>, ptr %9, align 16, !tbaa !6
  br label %53

53:                                               ; preds = %51, %56
  %54 = phi i64 [ %58, %56 ], [ 0, %51 ]
  %55 = invoke noundef i64 @_ZNSt24uniform_int_distributionImEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEmRT_RKNS0_10param_typeE(ptr noundef nonnull align 8 dereferenceable(16) %9, ptr noundef nonnull align 8 dereferenceable(5000) @_ZL3rng, ptr noundef nonnull align 8 dereferenceable(16) %9)
          to label %56 unwind label %90

56:                                               ; preds = %53
  %57 = getelementptr inbounds nuw i64, ptr %32, i64 %54
  store i64 %55, ptr %57, align 8, !tbaa !6
  %58 = add nuw nsw i64 %54, 1
  %59 = icmp eq i64 %58, %30
  br i1 %59, label %60, label %53, !llvm.loop !61

60:                                               ; preds = %56
  call void @llvm.lifetime.end.p0(ptr nonnull %9) #20
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(1) %33, ptr noundef nonnull align 8 dereferenceable(1) %32, i64 %31, i1 false), !tbaa !6
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(1) %35, ptr noundef nonnull align 8 dereferenceable(1) %32, i64 %31, i1 false), !tbaa !6
  %61 = getelementptr inbounds i64, ptr %41, i64 %52
  %62 = load i32, ptr %12, align 4, !tbaa !49
  %63 = load i32, ptr %13, align 4, !tbaa !49
  call void @llvm.lifetime.start.p0(ptr nonnull %5)
  call void @llvm.lifetime.start.p0(ptr nonnull %6)
  call void @llvm.lifetime.start.p0(ptr nonnull %7)
  call void @llvm.lifetime.start.p0(ptr nonnull %8)
  store ptr %61, ptr %5, align 8, !tbaa !62
  store ptr %41, ptr %6, align 8, !tbaa !62
  store i32 %62, ptr %7, align 4, !tbaa !49
  store i32 %63, ptr %8, align 4, !tbaa !49
  %64 = load ptr, ptr %42, align 8, !tbaa !20
  %65 = icmp eq ptr %64, null
  br i1 %65, label %66, label %68

66:                                               ; preds = %60
  invoke void @_ZSt25__throw_bad_function_callv() #24
          to label %67 unwind label %94

67:                                               ; preds = %66
  unreachable

68:                                               ; preds = %60
  %69 = load ptr, ptr %43, align 8, !tbaa !35
  invoke void %69(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %5, ptr noundef nonnull align 8 dereferenceable(8) %6, ptr noundef nonnull align 4 dereferenceable(4) %7, ptr noundef nonnull align 4 dereferenceable(4) %8)
          to label %70 unwind label %92

70:                                               ; preds = %68
  call void @llvm.lifetime.end.p0(ptr nonnull %5)
  call void @llvm.lifetime.end.p0(ptr nonnull %6)
  call void @llvm.lifetime.end.p0(ptr nonnull %7)
  call void @llvm.lifetime.end.p0(ptr nonnull %8)
  call void @llvm.lifetime.start.p0(ptr nonnull %10) #20
  store ptr %44, ptr %10, align 8, !tbaa !62
  call void @llvm.lifetime.start.p0(ptr nonnull %11) #20
  %71 = getelementptr inbounds i64, ptr %44, i64 %52
  store ptr %71, ptr %11, align 8, !tbaa !62
  invoke fastcc void @_ZL18callThroughOptnoneIRSt8functionIFvPmS1_jjEEJS1_RS1_RKiS7_EEvOT_DpOT0_(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %11, ptr noundef nonnull align 8 dereferenceable(8) %10, ptr noundef nonnull align 4 dereferenceable(4) %12, ptr noundef nonnull align 4 dereferenceable(4) %13)
          to label %72 unwind label %92

72:                                               ; preds = %70
  call void @llvm.lifetime.end.p0(ptr nonnull %11) #20
  %73 = call i32 @bcmp(ptr noundef nonnull readonly dereferenceable(1) %33, ptr noundef nonnull readonly dereferenceable(1) %35, i64 %31)
  %74 = icmp eq i32 %73, 0
  br i1 %74, label %83, label %75

75:                                               ; preds = %72
  %76 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.50, i64 noundef 23)
          to label %77 unwind label %94

77:                                               ; preds = %75
  %78 = trunc nsw i64 %52 to i32
  %79 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEi(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, i32 noundef range(i32 -2147483647, -2147483648) %78)
          to label %80 unwind label %94

80:                                               ; preds = %77
  %81 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) %79, ptr noundef nonnull @.str.49)
          to label %82 unwind label %94

82:                                               ; preds = %80
  call void @exit(i32 noundef 1) #25
  unreachable

83:                                               ; preds = %72
  call void @llvm.lifetime.end.p0(ptr nonnull %10) #20
  %84 = add nsw i64 %52, 1
  %85 = load i32, ptr %13, align 4, !tbaa !49
  %86 = shl i32 %85, 1
  %87 = add i32 %86, 2
  %88 = sext i32 %87 to i64
  %89 = icmp slt i64 %52, %88
  br i1 %89, label %51, label %46, !llvm.loop !70

90:                                               ; preds = %53
  %91 = landingpad { ptr, i32 }
          cleanup
  br label %96

92:                                               ; preds = %70, %68
  %93 = landingpad { ptr, i32 }
          cleanup
  br label %96

94:                                               ; preds = %75, %80, %77, %66
  %95 = landingpad { ptr, i32 }
          cleanup
  br label %96

96:                                               ; preds = %94, %92, %90
  %97 = phi { ptr, i32 } [ %91, %90 ], [ %95, %94 ], [ %93, %92 ]
  call void @_ZdaPv(ptr noundef nonnull %35) #23
  br label %98

98:                                               ; preds = %96, %49
  %99 = phi { ptr, i32 } [ %97, %96 ], [ %50, %49 ]
  call void @_ZdaPv(ptr noundef nonnull %33) #23
  br label %100

100:                                              ; preds = %98, %47
  %101 = phi { ptr, i32 } [ %99, %98 ], [ %48, %47 ]
  call void @_ZdaPv(ptr noundef nonnull %32) #23
  resume { ptr, i32 } %101
}

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

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8), ptr noundef, i64 noundef) local_unnamed_addr #7

declare void @_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate(ptr noundef nonnull align 8 dereferenceable(264), i32 noundef) local_unnamed_addr #7

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: read)
declare i64 @strlen(ptr noundef captures(none)) local_unnamed_addr #8

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZNKSt8functionIFvPhS0_jEEclES0_S0_j(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef %1, ptr noundef %2, i32 noundef %3) local_unnamed_addr #9 comdat {
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca i32, align 4
  store ptr %1, ptr %5, align 8, !tbaa !54
  store ptr %2, ptr %6, align 8, !tbaa !54
  store i32 %3, ptr %7, align 4, !tbaa !49
  %8 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %9 = load ptr, ptr %8, align 8, !tbaa !20
  %10 = icmp eq ptr %9, null
  br i1 %10, label %11, label %12

11:                                               ; preds = %4
  tail call void @_ZSt25__throw_bad_function_callv() #24
  unreachable

12:                                               ; preds = %4
  %13 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %14 = load ptr, ptr %13, align 8, !tbaa !16
  call void %14(ptr noundef nonnull align 8 dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(8) %5, ptr noundef nonnull align 8 dereferenceable(8) %6, ptr noundef nonnull align 4 dereferenceable(4) %7)
  ret void
}

; Function Attrs: mustprogress noinline norecurse optnone uwtable
define internal fastcc void @_ZL18callThroughOptnoneIRSt8functionIFvPhS1_jEEJS1_RS1_RjEEvOT_DpOT0_(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %1, ptr noundef nonnull align 8 dereferenceable(8) %2, ptr noundef nonnull align 4 dereferenceable(4) %3) unnamed_addr #10 {
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  store ptr %0, ptr %5, align 8, !tbaa !71
  store ptr %1, ptr %6, align 8, !tbaa !73
  store ptr %2, ptr %7, align 8, !tbaa !73
  store ptr %3, ptr %8, align 8, !tbaa !58
  %9 = load ptr, ptr %5, align 8, !tbaa !71, !nonnull !76, !align !77
  %10 = load ptr, ptr %6, align 8, !tbaa !73, !nonnull !76, !align !77
  %11 = load ptr, ptr %10, align 8, !tbaa !54
  %12 = load ptr, ptr %7, align 8, !tbaa !73, !nonnull !76, !align !77
  %13 = load ptr, ptr %12, align 8, !tbaa !54
  %14 = load ptr, ptr %8, align 8, !tbaa !58, !nonnull !76, !align !78
  %15 = load i32, ptr %14, align 4, !tbaa !49
  call void @_ZNKSt8functionIFvPhS0_jEEclES0_S0_j(ptr noundef nonnull align 8 dereferenceable(32) %9, ptr noundef %11, ptr noundef %13, i32 noundef %15)
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local noundef i8 @_ZNSt24uniform_int_distributionIhEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEhRT_RKNS0_10param_typeE(ptr noundef nonnull align 1 dereferenceable(2) %0, ptr noundef nonnull align 8 dereferenceable(5000) %1, ptr noundef nonnull align 1 dereferenceable(2) %2) local_unnamed_addr #9 comdat {
  %4 = alloca %"struct.std::uniform_int_distribution<unsigned char>::param_type", align 4
  %5 = getelementptr inbounds nuw i8, ptr %2, i64 1
  %6 = load i8, ptr %5, align 1, !tbaa !52
  %7 = zext i8 %6 to i64
  %8 = load i8, ptr %2, align 1, !tbaa !50
  %9 = zext i8 %8 to i64
  %10 = sub nsw i64 %7, %9
  %11 = icmp ult i64 %10, 4294967295
  br i1 %11, label %14, label %12

12:                                               ; preds = %3
  %13 = getelementptr inbounds nuw i8, ptr %4, i64 1
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
  br i1 %30, label %26, label %31, !llvm.loop !79

31:                                               ; preds = %26, %14, %22
  %32 = phi i64 [ %19, %14 ], [ %19, %22 ], [ %28, %26 ]
  %33 = lshr i64 %32, 32
  br label %43

34:                                               ; preds = %12, %34
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #20
  store i8 0, ptr %4, align 4, !tbaa !50
  store i8 -1, ptr %13, align 1, !tbaa !52
  %35 = call noundef i8 @_ZNSt24uniform_int_distributionIhEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEhRT_RKNS0_10param_typeE(ptr noundef nonnull align 1 dereferenceable(2) %0, ptr noundef nonnull align 8 dereferenceable(5000) %1, ptr noundef nonnull align 1 dereferenceable(2) %4)
  %36 = zext i8 %35 to i64
  %37 = shl nuw nsw i64 %36, 32
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #20
  %38 = call noundef i64 @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv(ptr noundef nonnull align 8 dereferenceable(5000) %1)
  %39 = add i64 %37, %38
  %40 = icmp ugt i64 %39, %10
  %41 = icmp ult i64 %39, %37
  %42 = or i1 %40, %41
  br i1 %42, label %34, label %43, !llvm.loop !80

43:                                               ; preds = %34, %31
  %44 = phi i64 [ %33, %31 ], [ %39, %34 ]
  %45 = load i8, ptr %2, align 1, !tbaa !50
  %46 = trunc i64 %44 to i8
  %47 = add i8 %45, %46
  ret i8 %47
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local noundef i64 @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv(ptr noundef nonnull align 8 dereferenceable(5000) %0) local_unnamed_addr #9 comdat {
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
  br i1 %43, label %44, label %8, !llvm.loop !81

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
  br i1 %111, label %112, label %91, !llvm.loop !84

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

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEi(ptr noundef nonnull align 8 dereferenceable(8), i32 noundef) local_unnamed_addr #7

; Function Attrs: nofree noreturn nounwind
declare void @exit(i32 noundef) local_unnamed_addr #12

; Function Attrs: nobuiltin nounwind
declare void @_ZdaPv(ptr noundef) local_unnamed_addr #13

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #14

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPhS0_jEZ4mainE3$_0E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #15 {
  %5 = load ptr, ptr %1, align 8, !tbaa !54
  %6 = load ptr, ptr %2, align 8, !tbaa !54
  %7 = load i32, ptr %3, align 4, !tbaa !49
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %19, label %9

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 0, %9 ], [ %17, %11 ]
  %13 = getelementptr inbounds nuw i8, ptr %6, i64 %12
  %14 = load i8, ptr %13, align 1, !tbaa !15
  %15 = add i8 %14, 10
  %16 = getelementptr inbounds nuw i8, ptr %5, i64 %12
  store i8 %15, ptr %16, align 1, !tbaa !15
  %17 = add nuw nsw i64 %12, 1
  %18 = icmp eq i64 %17, %10
  br i1 %18, label %19, label %11, !llvm.loop !85

19:                                               ; preds = %11, %4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPhS0_jEZ4mainE3$_0E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_0", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPhS0_jEZ4mainE3$_1E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #15 {
  %5 = load ptr, ptr %1, align 8, !tbaa !54
  %6 = load ptr, ptr %2, align 8, !tbaa !54
  %7 = load i32, ptr %3, align 4, !tbaa !49
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %61, label %9

9:                                                ; preds = %4
  %10 = ptrtoint ptr %6 to i64
  %11 = ptrtoint ptr %5 to i64
  %12 = zext i32 %7 to i64
  %13 = icmp ult i32 %7, 8
  %14 = sub i64 %11, %10
  %15 = icmp ult i64 %14, 32
  %16 = select i1 %13, i1 true, i1 %15
  br i1 %16, label %51, label %17

17:                                               ; preds = %9
  %18 = icmp ult i32 %7, 32
  br i1 %18, label %38, label %19

19:                                               ; preds = %17
  %20 = and i64 %12, 4294967264
  br label %21

21:                                               ; preds = %21, %19
  %22 = phi i64 [ 0, %19 ], [ %31, %21 ]
  %23 = getelementptr inbounds nuw i8, ptr %6, i64 %22
  %24 = getelementptr inbounds nuw i8, ptr %23, i64 16
  %25 = load <16 x i8>, ptr %23, align 1, !tbaa !15
  %26 = load <16 x i8>, ptr %24, align 1, !tbaa !15
  %27 = add <16 x i8> %25, splat (i8 10)
  %28 = add <16 x i8> %26, splat (i8 10)
  %29 = getelementptr inbounds nuw i8, ptr %5, i64 %22
  %30 = getelementptr inbounds nuw i8, ptr %29, i64 16
  store <16 x i8> %27, ptr %29, align 1, !tbaa !15
  store <16 x i8> %28, ptr %30, align 1, !tbaa !15
  %31 = add nuw i64 %22, 32
  %32 = icmp eq i64 %31, %20
  br i1 %32, label %33, label %21, !llvm.loop !89

33:                                               ; preds = %21
  %34 = icmp eq i64 %20, %12
  br i1 %34, label %61, label %35

35:                                               ; preds = %33
  %36 = and i64 %12, 24
  %37 = icmp eq i64 %36, 0
  br i1 %37, label %51, label %38

38:                                               ; preds = %35, %17
  %39 = phi i64 [ %20, %35 ], [ 0, %17 ]
  %40 = and i64 %12, 4294967288
  br label %41

41:                                               ; preds = %41, %38
  %42 = phi i64 [ %39, %38 ], [ %47, %41 ]
  %43 = getelementptr inbounds nuw i8, ptr %6, i64 %42
  %44 = load <8 x i8>, ptr %43, align 1, !tbaa !15
  %45 = add <8 x i8> %44, splat (i8 10)
  %46 = getelementptr inbounds nuw i8, ptr %5, i64 %42
  store <8 x i8> %45, ptr %46, align 1, !tbaa !15
  %47 = add nuw i64 %42, 8
  %48 = icmp eq i64 %47, %40
  br i1 %48, label %49, label %41, !llvm.loop !90

49:                                               ; preds = %41
  %50 = icmp eq i64 %40, %12
  br i1 %50, label %61, label %51

51:                                               ; preds = %35, %49, %9
  %52 = phi i64 [ 0, %9 ], [ %20, %35 ], [ %40, %49 ]
  br label %53

53:                                               ; preds = %51, %53
  %54 = phi i64 [ %59, %53 ], [ %52, %51 ]
  %55 = getelementptr inbounds nuw i8, ptr %6, i64 %54
  %56 = load i8, ptr %55, align 1, !tbaa !15
  %57 = add i8 %56, 10
  %58 = getelementptr inbounds nuw i8, ptr %5, i64 %54
  store i8 %57, ptr %58, align 1, !tbaa !15
  %59 = add nuw nsw i64 %54, 1
  %60 = icmp eq i64 %59, %12
  br i1 %60, label %61, label %53, !llvm.loop !91

61:                                               ; preds = %53, %33, %49, %4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPhS0_jEZ4mainE3$_1E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_1", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZNKSt8functionIFvPjS0_jEEclES0_S0_j(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef %1, ptr noundef %2, i32 noundef %3) local_unnamed_addr #9 comdat {
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca i32, align 4
  store ptr %1, ptr %5, align 8, !tbaa !58
  store ptr %2, ptr %6, align 8, !tbaa !58
  store i32 %3, ptr %7, align 4, !tbaa !49
  %8 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %9 = load ptr, ptr %8, align 8, !tbaa !20
  %10 = icmp eq ptr %9, null
  br i1 %10, label %11, label %12

11:                                               ; preds = %4
  tail call void @_ZSt25__throw_bad_function_callv() #24
  unreachable

12:                                               ; preds = %4
  %13 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %14 = load ptr, ptr %13, align 8, !tbaa !21
  call void %14(ptr noundef nonnull align 8 dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(8) %5, ptr noundef nonnull align 8 dereferenceable(8) %6, ptr noundef nonnull align 4 dereferenceable(4) %7)
  ret void
}

; Function Attrs: mustprogress noinline norecurse optnone uwtable
define internal fastcc void @_ZL18callThroughOptnoneIRSt8functionIFvPjS1_jEEJS1_RS1_RjEEvOT_DpOT0_(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %1, ptr noundef nonnull align 8 dereferenceable(8) %2, ptr noundef nonnull align 4 dereferenceable(4) %3) unnamed_addr #10 {
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  store ptr %0, ptr %5, align 8, !tbaa !92
  store ptr %1, ptr %6, align 8, !tbaa !94
  store ptr %2, ptr %7, align 8, !tbaa !94
  store ptr %3, ptr %8, align 8, !tbaa !58
  %9 = load ptr, ptr %5, align 8, !tbaa !92, !nonnull !76, !align !77
  %10 = load ptr, ptr %6, align 8, !tbaa !94, !nonnull !76, !align !77
  %11 = load ptr, ptr %10, align 8, !tbaa !58
  %12 = load ptr, ptr %7, align 8, !tbaa !94, !nonnull !76, !align !77
  %13 = load ptr, ptr %12, align 8, !tbaa !58
  %14 = load ptr, ptr %8, align 8, !tbaa !58, !nonnull !76, !align !78
  %15 = load i32, ptr %14, align 4, !tbaa !49
  call void @_ZNKSt8functionIFvPjS0_jEEclES0_S0_j(ptr noundef nonnull align 8 dereferenceable(32) %9, ptr noundef %11, ptr noundef %13, i32 noundef %15)
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local noundef i32 @_ZNSt24uniform_int_distributionIjEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEjRT_RKNS0_10param_typeE(ptr noundef nonnull align 4 dereferenceable(8) %0, ptr noundef nonnull align 8 dereferenceable(5000) %1, ptr noundef nonnull align 4 dereferenceable(8) %2) local_unnamed_addr #9 comdat {
  %4 = alloca %"struct.std::uniform_int_distribution<unsigned int>::param_type", align 8
  %5 = getelementptr inbounds nuw i8, ptr %2, i64 4
  %6 = load i32, ptr %5, align 4, !tbaa !96
  %7 = zext i32 %6 to i64
  %8 = load i32, ptr %2, align 4, !tbaa !98
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
  br i1 %28, label %24, label %29, !llvm.loop !99

29:                                               ; preds = %24, %12, %20
  %30 = phi i64 [ %17, %12 ], [ %17, %20 ], [ %26, %24 ]
  %31 = lshr i64 %30, 32
  br label %45

32:                                               ; preds = %3
  %33 = icmp eq i64 %10, 4294967295
  br i1 %33, label %43, label %34

34:                                               ; preds = %32, %34
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #20
  store <2 x i32> <i32 0, i32 -1>, ptr %4, align 8, !tbaa !49
  %35 = call noundef i32 @_ZNSt24uniform_int_distributionIjEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEjRT_RKNS0_10param_typeE(ptr noundef nonnull align 4 dereferenceable(8) %0, ptr noundef nonnull align 8 dereferenceable(5000) %1, ptr noundef nonnull align 4 dereferenceable(8) %4)
  %36 = zext i32 %35 to i64
  %37 = shl nuw i64 %36, 32
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #20
  %38 = call noundef i64 @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv(ptr noundef nonnull align 8 dereferenceable(5000) %1)
  %39 = add i64 %37, %38
  %40 = icmp ugt i64 %39, %10
  %41 = icmp ult i64 %39, %37
  %42 = or i1 %40, %41
  br i1 %42, label %34, label %45, !llvm.loop !100

43:                                               ; preds = %32
  %44 = tail call noundef i64 @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv(ptr noundef nonnull align 8 dereferenceable(5000) %1)
  br label %45

45:                                               ; preds = %34, %43, %29
  %46 = phi i64 [ %31, %29 ], [ %44, %43 ], [ %39, %34 ]
  %47 = load i32, ptr %2, align 4, !tbaa !98
  %48 = trunc i64 %46 to i32
  %49 = add i32 %47, %48
  ret i32 %49
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE3$_0E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #15 {
  %5 = load ptr, ptr %1, align 8, !tbaa !58
  %6 = load ptr, ptr %2, align 8, !tbaa !58
  %7 = load i32, ptr %3, align 4, !tbaa !49
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %19, label %9

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 0, %9 ], [ %17, %11 ]
  %13 = getelementptr inbounds nuw i32, ptr %6, i64 %12
  %14 = load i32, ptr %13, align 4, !tbaa !49
  %15 = add i32 %14, 10
  %16 = getelementptr inbounds nuw i32, ptr %5, i64 %12
  store i32 %15, ptr %16, align 4, !tbaa !49
  %17 = add nuw nsw i64 %12, 1
  %18 = icmp eq i64 %17, %10
  br i1 %18, label %19, label %11, !llvm.loop !101

19:                                               ; preds = %11, %4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE3$_0E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_0", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE3$_1E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #15 {
  %5 = load ptr, ptr %1, align 8, !tbaa !58
  %6 = load ptr, ptr %2, align 8, !tbaa !58
  %7 = load i32, ptr %3, align 4, !tbaa !49
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %43, label %9

9:                                                ; preds = %4
  %10 = ptrtoint ptr %6 to i64
  %11 = ptrtoint ptr %5 to i64
  %12 = zext i32 %7 to i64
  %13 = icmp ult i32 %7, 8
  %14 = sub i64 %11, %10
  %15 = icmp ult i64 %14, 32
  %16 = select i1 %13, i1 true, i1 %15
  br i1 %16, label %33, label %17

17:                                               ; preds = %9
  %18 = and i64 %12, 4294967288
  br label %19

19:                                               ; preds = %19, %17
  %20 = phi i64 [ 0, %17 ], [ %29, %19 ]
  %21 = getelementptr inbounds nuw i32, ptr %6, i64 %20
  %22 = getelementptr inbounds nuw i8, ptr %21, i64 16
  %23 = load <4 x i32>, ptr %21, align 4, !tbaa !49
  %24 = load <4 x i32>, ptr %22, align 4, !tbaa !49
  %25 = add <4 x i32> %23, splat (i32 10)
  %26 = add <4 x i32> %24, splat (i32 10)
  %27 = getelementptr inbounds nuw i32, ptr %5, i64 %20
  %28 = getelementptr inbounds nuw i8, ptr %27, i64 16
  store <4 x i32> %25, ptr %27, align 4, !tbaa !49
  store <4 x i32> %26, ptr %28, align 4, !tbaa !49
  %29 = add nuw i64 %20, 8
  %30 = icmp eq i64 %29, %18
  br i1 %30, label %31, label %19, !llvm.loop !102

31:                                               ; preds = %19
  %32 = icmp eq i64 %18, %12
  br i1 %32, label %43, label %33

33:                                               ; preds = %9, %31
  %34 = phi i64 [ 0, %9 ], [ %18, %31 ]
  br label %35

35:                                               ; preds = %33, %35
  %36 = phi i64 [ %41, %35 ], [ %34, %33 ]
  %37 = getelementptr inbounds nuw i32, ptr %6, i64 %36
  %38 = load i32, ptr %37, align 4, !tbaa !49
  %39 = add i32 %38, 10
  %40 = getelementptr inbounds nuw i32, ptr %5, i64 %36
  store i32 %39, ptr %40, align 4, !tbaa !49
  %41 = add nuw nsw i64 %36, 1
  %42 = icmp eq i64 %41, %12
  br i1 %42, label %43, label %35, !llvm.loop !103

43:                                               ; preds = %35, %31, %4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE3$_1E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_1", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZNKSt8functionIFvPmS0_jEEclES0_S0_j(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef %1, ptr noundef %2, i32 noundef %3) local_unnamed_addr #9 comdat {
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca i32, align 4
  store ptr %1, ptr %5, align 8, !tbaa !62
  store ptr %2, ptr %6, align 8, !tbaa !62
  store i32 %3, ptr %7, align 4, !tbaa !49
  %8 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %9 = load ptr, ptr %8, align 8, !tbaa !20
  %10 = icmp eq ptr %9, null
  br i1 %10, label %11, label %12

11:                                               ; preds = %4
  tail call void @_ZSt25__throw_bad_function_callv() #24
  unreachable

12:                                               ; preds = %4
  %13 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %14 = load ptr, ptr %13, align 8, !tbaa !23
  call void %14(ptr noundef nonnull align 8 dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(8) %5, ptr noundef nonnull align 8 dereferenceable(8) %6, ptr noundef nonnull align 4 dereferenceable(4) %7)
  ret void
}

; Function Attrs: mustprogress noinline norecurse optnone uwtable
define internal fastcc void @_ZL18callThroughOptnoneIRSt8functionIFvPmS1_jEEJS1_RS1_RjEEvOT_DpOT0_(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %1, ptr noundef nonnull align 8 dereferenceable(8) %2, ptr noundef nonnull align 4 dereferenceable(4) %3) unnamed_addr #10 {
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  store ptr %0, ptr %5, align 8, !tbaa !104
  store ptr %1, ptr %6, align 8, !tbaa !106
  store ptr %2, ptr %7, align 8, !tbaa !106
  store ptr %3, ptr %8, align 8, !tbaa !58
  %9 = load ptr, ptr %5, align 8, !tbaa !104, !nonnull !76, !align !77
  %10 = load ptr, ptr %6, align 8, !tbaa !106, !nonnull !76, !align !77
  %11 = load ptr, ptr %10, align 8, !tbaa !62
  %12 = load ptr, ptr %7, align 8, !tbaa !106, !nonnull !76, !align !77
  %13 = load ptr, ptr %12, align 8, !tbaa !62
  %14 = load ptr, ptr %8, align 8, !tbaa !58, !nonnull !76, !align !78
  %15 = load i32, ptr %14, align 4, !tbaa !49
  call void @_ZNKSt8functionIFvPmS0_jEEclES0_S0_j(ptr noundef nonnull align 8 dereferenceable(32) %9, ptr noundef %11, ptr noundef %13, i32 noundef %15)
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local noundef i64 @_ZNSt24uniform_int_distributionImEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEmRT_RKNS0_10param_typeE(ptr noundef nonnull align 8 dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(5000) %1, ptr noundef nonnull align 8 dereferenceable(16) %2) local_unnamed_addr #9 comdat {
  %4 = alloca %"struct.std::uniform_int_distribution<unsigned long>::param_type", align 8
  %5 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %6 = load i64, ptr %5, align 8, !tbaa !108
  %7 = load i64, ptr %2, align 8, !tbaa !110
  %8 = sub i64 %6, %7
  %9 = icmp ult i64 %8, 4294967295
  br i1 %9, label %10, label %30

10:                                               ; preds = %3
  %11 = trunc nuw i64 %8 to i32
  %12 = add nuw i32 %11, 1
  %13 = tail call noundef i64 @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv(ptr noundef nonnull align 8 dereferenceable(5000) %1)
  %14 = zext i32 %12 to i64
  %15 = mul i64 %13, %14
  %16 = trunc i64 %15 to i32
  %17 = icmp ult i32 %11, %16
  br i1 %17, label %27, label %18

18:                                               ; preds = %10
  %19 = xor i32 %11, -1
  %20 = urem i32 %19, %12
  %21 = icmp ugt i32 %20, %16
  br i1 %21, label %22, label %27

22:                                               ; preds = %18, %22
  %23 = tail call noundef i64 @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv(ptr noundef nonnull align 8 dereferenceable(5000) %1)
  %24 = mul i64 %23, %14
  %25 = trunc i64 %24 to i32
  %26 = icmp ugt i32 %20, %25
  br i1 %26, label %22, label %27, !llvm.loop !111

27:                                               ; preds = %22, %10, %18
  %28 = phi i64 [ %15, %10 ], [ %15, %18 ], [ %24, %22 ]
  %29 = lshr i64 %28, 32
  br label %45

30:                                               ; preds = %3
  %31 = icmp eq i64 %8, 4294967295
  br i1 %31, label %43, label %32

32:                                               ; preds = %30
  %33 = lshr i64 %8, 32
  %34 = getelementptr inbounds nuw i8, ptr %4, i64 8
  br label %35

35:                                               ; preds = %32, %35
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #20
  store i64 0, ptr %4, align 8, !tbaa !110
  store i64 %33, ptr %34, align 8, !tbaa !108
  %36 = call noundef i64 @_ZNSt24uniform_int_distributionImEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEmRT_RKNS0_10param_typeE(ptr noundef nonnull align 8 dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(5000) %1, ptr noundef nonnull align 8 dereferenceable(16) %4)
  %37 = shl i64 %36, 32
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #20
  %38 = call noundef i64 @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv(ptr noundef nonnull align 8 dereferenceable(5000) %1)
  %39 = add i64 %38, %37
  %40 = icmp ugt i64 %39, %8
  %41 = icmp ult i64 %39, %37
  %42 = or i1 %40, %41
  br i1 %42, label %35, label %45, !llvm.loop !112

43:                                               ; preds = %30
  %44 = tail call noundef i64 @_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv(ptr noundef nonnull align 8 dereferenceable(5000) %1)
  br label %45

45:                                               ; preds = %35, %43, %27
  %46 = phi i64 [ %29, %27 ], [ %44, %43 ], [ %39, %35 ]
  %47 = load i64, ptr %2, align 8, !tbaa !110
  %48 = add i64 %47, %46
  ret i64 %48
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPmS0_jEZ4mainE3$_0E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #15 {
  %5 = load ptr, ptr %1, align 8, !tbaa !62
  %6 = load ptr, ptr %2, align 8, !tbaa !62
  %7 = load i32, ptr %3, align 4, !tbaa !49
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %19, label %9

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 0, %9 ], [ %17, %11 ]
  %13 = getelementptr inbounds nuw i64, ptr %6, i64 %12
  %14 = load i64, ptr %13, align 8, !tbaa !6
  %15 = add i64 %14, 10
  %16 = getelementptr inbounds nuw i64, ptr %5, i64 %12
  store i64 %15, ptr %16, align 8, !tbaa !6
  %17 = add nuw nsw i64 %12, 1
  %18 = icmp eq i64 %17, %10
  br i1 %18, label %19, label %11, !llvm.loop !113

19:                                               ; preds = %11, %4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPmS0_jEZ4mainE3$_0E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_0", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPmS0_jEZ4mainE3$_1E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #15 {
  %5 = load ptr, ptr %1, align 8, !tbaa !62
  %6 = load ptr, ptr %2, align 8, !tbaa !62
  %7 = load i32, ptr %3, align 4, !tbaa !49
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %43, label %9

9:                                                ; preds = %4
  %10 = ptrtoint ptr %6 to i64
  %11 = ptrtoint ptr %5 to i64
  %12 = zext i32 %7 to i64
  %13 = icmp ult i32 %7, 4
  %14 = sub i64 %11, %10
  %15 = icmp ult i64 %14, 32
  %16 = select i1 %13, i1 true, i1 %15
  br i1 %16, label %33, label %17

17:                                               ; preds = %9
  %18 = and i64 %12, 4294967292
  br label %19

19:                                               ; preds = %19, %17
  %20 = phi i64 [ 0, %17 ], [ %29, %19 ]
  %21 = getelementptr inbounds nuw i64, ptr %6, i64 %20
  %22 = getelementptr inbounds nuw i8, ptr %21, i64 16
  %23 = load <2 x i64>, ptr %21, align 8, !tbaa !6
  %24 = load <2 x i64>, ptr %22, align 8, !tbaa !6
  %25 = add <2 x i64> %23, splat (i64 10)
  %26 = add <2 x i64> %24, splat (i64 10)
  %27 = getelementptr inbounds nuw i64, ptr %5, i64 %20
  %28 = getelementptr inbounds nuw i8, ptr %27, i64 16
  store <2 x i64> %25, ptr %27, align 8, !tbaa !6
  store <2 x i64> %26, ptr %28, align 8, !tbaa !6
  %29 = add nuw i64 %20, 4
  %30 = icmp eq i64 %29, %18
  br i1 %30, label %31, label %19, !llvm.loop !114

31:                                               ; preds = %19
  %32 = icmp eq i64 %18, %12
  br i1 %32, label %43, label %33

33:                                               ; preds = %9, %31
  %34 = phi i64 [ 0, %9 ], [ %18, %31 ]
  br label %35

35:                                               ; preds = %33, %35
  %36 = phi i64 [ %41, %35 ], [ %34, %33 ]
  %37 = getelementptr inbounds nuw i64, ptr %6, i64 %36
  %38 = load i64, ptr %37, align 8, !tbaa !6
  %39 = add i64 %38, 10
  %40 = getelementptr inbounds nuw i64, ptr %5, i64 %36
  store i64 %39, ptr %40, align 8, !tbaa !6
  %41 = add nuw nsw i64 %36, 1
  %42 = icmp eq i64 %41, %12
  br i1 %42, label %43, label %35, !llvm.loop !115

43:                                               ; preds = %35, %31, %4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPmS0_jEZ4mainE3$_1E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_1", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPhS0_jEZ4mainE3$_2E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #15 {
  %5 = load ptr, ptr %1, align 8, !tbaa !54
  %6 = load ptr, ptr %2, align 8, !tbaa !54
  %7 = load i32, ptr %3, align 4, !tbaa !49
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %21, label %9

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 0, %9 ], [ %19, %11 ]
  %13 = add nuw nsw i64 %12, 3
  %14 = and i64 %13, 4294967295
  %15 = getelementptr inbounds nuw i8, ptr %6, i64 %14
  %16 = load i8, ptr %15, align 1, !tbaa !15
  %17 = add i8 %16, 10
  %18 = getelementptr inbounds nuw i8, ptr %5, i64 %12
  store i8 %17, ptr %18, align 1, !tbaa !15
  %19 = add nuw nsw i64 %12, 1
  %20 = icmp eq i64 %19, %10
  br i1 %20, label %21, label %11, !llvm.loop !116

21:                                               ; preds = %11, %4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPhS0_jEZ4mainE3$_2E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_2", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPhS0_jEZ4mainE3$_3E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #15 {
  %5 = load ptr, ptr %1, align 8, !tbaa !54
  %6 = ptrtoint ptr %5 to i64
  %7 = load ptr, ptr %2, align 8, !tbaa !54
  %8 = ptrtoint ptr %7 to i64
  %9 = load i32, ptr %3, align 4, !tbaa !49
  %10 = icmp eq i32 %9, 0
  br i1 %10, label %74, label %11

11:                                               ; preds = %4
  %12 = zext i32 %9 to i64
  %13 = icmp ult i32 %9, 8
  br i1 %13, label %62, label %14

14:                                               ; preds = %11
  %15 = add nsw i64 %12, -1
  %16 = trunc i64 %15 to i32
  %17 = icmp ugt i32 %16, -4
  %18 = icmp ugt i64 %15, 4294967295
  %19 = or i1 %17, %18
  br i1 %19, label %62, label %20

20:                                               ; preds = %14
  %21 = add i64 %6, -3
  %22 = sub i64 %21, %8
  %23 = icmp ult i64 %22, 32
  br i1 %23, label %62, label %24

24:                                               ; preds = %20
  %25 = icmp ult i32 %9, 32
  br i1 %25, label %47, label %26

26:                                               ; preds = %24
  %27 = and i64 %12, 4294967264
  br label %28

28:                                               ; preds = %28, %26
  %29 = phi i64 [ 0, %26 ], [ %40, %28 ]
  %30 = and i64 %29, 4294967264
  %31 = getelementptr inbounds nuw i8, ptr %7, i64 %30
  %32 = getelementptr inbounds nuw i8, ptr %31, i64 3
  %33 = getelementptr inbounds nuw i8, ptr %31, i64 19
  %34 = load <16 x i8>, ptr %32, align 1, !tbaa !15
  %35 = load <16 x i8>, ptr %33, align 1, !tbaa !15
  %36 = add <16 x i8> %34, splat (i8 10)
  %37 = add <16 x i8> %35, splat (i8 10)
  %38 = getelementptr inbounds nuw i8, ptr %5, i64 %29
  %39 = getelementptr inbounds nuw i8, ptr %38, i64 16
  store <16 x i8> %36, ptr %38, align 1, !tbaa !15
  store <16 x i8> %37, ptr %39, align 1, !tbaa !15
  %40 = add nuw i64 %29, 32
  %41 = icmp eq i64 %40, %27
  br i1 %41, label %42, label %28, !llvm.loop !117

42:                                               ; preds = %28
  %43 = icmp eq i64 %27, %12
  br i1 %43, label %74, label %44

44:                                               ; preds = %42
  %45 = and i64 %12, 24
  %46 = icmp eq i64 %45, 0
  br i1 %46, label %62, label %47

47:                                               ; preds = %44, %24
  %48 = phi i64 [ %27, %44 ], [ 0, %24 ]
  %49 = and i64 %12, 4294967288
  br label %50

50:                                               ; preds = %50, %47
  %51 = phi i64 [ %48, %47 ], [ %58, %50 ]
  %52 = and i64 %51, 4294967288
  %53 = getelementptr inbounds nuw i8, ptr %7, i64 %52
  %54 = getelementptr inbounds nuw i8, ptr %53, i64 3
  %55 = load <8 x i8>, ptr %54, align 1, !tbaa !15
  %56 = add <8 x i8> %55, splat (i8 10)
  %57 = getelementptr inbounds nuw i8, ptr %5, i64 %51
  store <8 x i8> %56, ptr %57, align 1, !tbaa !15
  %58 = add nuw i64 %51, 8
  %59 = icmp eq i64 %58, %49
  br i1 %59, label %60, label %50, !llvm.loop !118

60:                                               ; preds = %50
  %61 = icmp eq i64 %49, %12
  br i1 %61, label %74, label %62

62:                                               ; preds = %44, %60, %20, %14, %11
  %63 = phi i64 [ 0, %11 ], [ 0, %14 ], [ 0, %20 ], [ %27, %44 ], [ %49, %60 ]
  br label %64

64:                                               ; preds = %62, %64
  %65 = phi i64 [ %72, %64 ], [ %63, %62 ]
  %66 = add nuw nsw i64 %65, 3
  %67 = and i64 %66, 4294967295
  %68 = getelementptr inbounds nuw i8, ptr %7, i64 %67
  %69 = load i8, ptr %68, align 1, !tbaa !15
  %70 = add i8 %69, 10
  %71 = getelementptr inbounds nuw i8, ptr %5, i64 %65
  store i8 %70, ptr %71, align 1, !tbaa !15
  %72 = add nuw nsw i64 %65, 1
  %73 = icmp eq i64 %72, %12
  br i1 %73, label %74, label %64, !llvm.loop !119

74:                                               ; preds = %64, %42, %60, %4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPhS0_jEZ4mainE3$_3E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_3", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE3$_2E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #15 {
  %5 = load ptr, ptr %1, align 8, !tbaa !58
  %6 = load ptr, ptr %2, align 8, !tbaa !58
  %7 = load i32, ptr %3, align 4, !tbaa !49
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %21, label %9

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 0, %9 ], [ %19, %11 ]
  %13 = add nuw nsw i64 %12, 3
  %14 = and i64 %13, 4294967295
  %15 = getelementptr inbounds nuw i32, ptr %6, i64 %14
  %16 = load i32, ptr %15, align 4, !tbaa !49
  %17 = add i32 %16, 10
  %18 = getelementptr inbounds nuw i32, ptr %5, i64 %12
  store i32 %17, ptr %18, align 4, !tbaa !49
  %19 = add nuw nsw i64 %12, 1
  %20 = icmp eq i64 %19, %10
  br i1 %20, label %21, label %11, !llvm.loop !120

21:                                               ; preds = %11, %4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE3$_2E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_2", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE3$_3E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #15 {
  %5 = load ptr, ptr %1, align 8, !tbaa !58
  %6 = ptrtoint ptr %5 to i64
  %7 = load ptr, ptr %2, align 8, !tbaa !58
  %8 = ptrtoint ptr %7 to i64
  %9 = load i32, ptr %3, align 4, !tbaa !49
  %10 = icmp eq i32 %9, 0
  br i1 %10, label %54, label %11

11:                                               ; preds = %4
  %12 = zext i32 %9 to i64
  %13 = icmp ult i32 %9, 8
  br i1 %13, label %42, label %14

14:                                               ; preds = %11
  %15 = add nsw i64 %12, -1
  %16 = trunc i64 %15 to i32
  %17 = icmp ugt i32 %16, -4
  %18 = icmp ugt i64 %15, 4294967295
  %19 = or i1 %17, %18
  br i1 %19, label %42, label %20

20:                                               ; preds = %14
  %21 = add i64 %6, -12
  %22 = sub i64 %21, %8
  %23 = icmp ult i64 %22, 32
  br i1 %23, label %42, label %24

24:                                               ; preds = %20
  %25 = and i64 %12, 4294967288
  br label %26

26:                                               ; preds = %26, %24
  %27 = phi i64 [ 0, %24 ], [ %38, %26 ]
  %28 = and i64 %27, 4294967288
  %29 = getelementptr inbounds nuw i32, ptr %7, i64 %28
  %30 = getelementptr inbounds nuw i8, ptr %29, i64 12
  %31 = getelementptr inbounds nuw i8, ptr %29, i64 28
  %32 = load <4 x i32>, ptr %30, align 4, !tbaa !49
  %33 = load <4 x i32>, ptr %31, align 4, !tbaa !49
  %34 = add <4 x i32> %32, splat (i32 10)
  %35 = add <4 x i32> %33, splat (i32 10)
  %36 = getelementptr inbounds nuw i32, ptr %5, i64 %27
  %37 = getelementptr inbounds nuw i8, ptr %36, i64 16
  store <4 x i32> %34, ptr %36, align 4, !tbaa !49
  store <4 x i32> %35, ptr %37, align 4, !tbaa !49
  %38 = add nuw i64 %27, 8
  %39 = icmp eq i64 %38, %25
  br i1 %39, label %40, label %26, !llvm.loop !121

40:                                               ; preds = %26
  %41 = icmp eq i64 %25, %12
  br i1 %41, label %54, label %42

42:                                               ; preds = %20, %14, %11, %40
  %43 = phi i64 [ 0, %20 ], [ 0, %14 ], [ 0, %11 ], [ %25, %40 ]
  br label %44

44:                                               ; preds = %42, %44
  %45 = phi i64 [ %52, %44 ], [ %43, %42 ]
  %46 = add nuw nsw i64 %45, 3
  %47 = and i64 %46, 4294967295
  %48 = getelementptr inbounds nuw i32, ptr %7, i64 %47
  %49 = load i32, ptr %48, align 4, !tbaa !49
  %50 = add i32 %49, 10
  %51 = getelementptr inbounds nuw i32, ptr %5, i64 %45
  store i32 %50, ptr %51, align 4, !tbaa !49
  %52 = add nuw nsw i64 %45, 1
  %53 = icmp eq i64 %52, %12
  br i1 %53, label %54, label %44, !llvm.loop !122

54:                                               ; preds = %44, %40, %4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE3$_3E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_3", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPmS0_jEZ4mainE3$_2E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #15 {
  %5 = load ptr, ptr %1, align 8, !tbaa !62
  %6 = load ptr, ptr %2, align 8, !tbaa !62
  %7 = load i32, ptr %3, align 4, !tbaa !49
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %21, label %9

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 0, %9 ], [ %19, %11 ]
  %13 = add nuw nsw i64 %12, 3
  %14 = and i64 %13, 4294967295
  %15 = getelementptr inbounds nuw i64, ptr %6, i64 %14
  %16 = load i64, ptr %15, align 8, !tbaa !6
  %17 = add i64 %16, 10
  %18 = getelementptr inbounds nuw i64, ptr %5, i64 %12
  store i64 %17, ptr %18, align 8, !tbaa !6
  %19 = add nuw nsw i64 %12, 1
  %20 = icmp eq i64 %19, %10
  br i1 %20, label %21, label %11, !llvm.loop !123

21:                                               ; preds = %11, %4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPmS0_jEZ4mainE3$_2E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_2", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPmS0_jEZ4mainE3$_3E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #15 {
  %5 = load ptr, ptr %1, align 8, !tbaa !62
  %6 = ptrtoint ptr %5 to i64
  %7 = load ptr, ptr %2, align 8, !tbaa !62
  %8 = ptrtoint ptr %7 to i64
  %9 = load i32, ptr %3, align 4, !tbaa !49
  %10 = icmp eq i32 %9, 0
  br i1 %10, label %54, label %11

11:                                               ; preds = %4
  %12 = zext i32 %9 to i64
  %13 = icmp ult i32 %9, 4
  br i1 %13, label %42, label %14

14:                                               ; preds = %11
  %15 = add nsw i64 %12, -1
  %16 = trunc i64 %15 to i32
  %17 = icmp ugt i32 %16, -4
  %18 = icmp ugt i64 %15, 4294967295
  %19 = or i1 %17, %18
  br i1 %19, label %42, label %20

20:                                               ; preds = %14
  %21 = add i64 %6, -24
  %22 = sub i64 %21, %8
  %23 = icmp ult i64 %22, 32
  br i1 %23, label %42, label %24

24:                                               ; preds = %20
  %25 = and i64 %12, 4294967292
  br label %26

26:                                               ; preds = %26, %24
  %27 = phi i64 [ 0, %24 ], [ %38, %26 ]
  %28 = and i64 %27, 4294967292
  %29 = getelementptr inbounds nuw i64, ptr %7, i64 %28
  %30 = getelementptr inbounds nuw i8, ptr %29, i64 24
  %31 = getelementptr inbounds nuw i8, ptr %29, i64 40
  %32 = load <2 x i64>, ptr %30, align 8, !tbaa !6
  %33 = load <2 x i64>, ptr %31, align 8, !tbaa !6
  %34 = add <2 x i64> %32, splat (i64 10)
  %35 = add <2 x i64> %33, splat (i64 10)
  %36 = getelementptr inbounds nuw i64, ptr %5, i64 %27
  %37 = getelementptr inbounds nuw i8, ptr %36, i64 16
  store <2 x i64> %34, ptr %36, align 8, !tbaa !6
  store <2 x i64> %35, ptr %37, align 8, !tbaa !6
  %38 = add nuw i64 %27, 4
  %39 = icmp eq i64 %38, %25
  br i1 %39, label %40, label %26, !llvm.loop !124

40:                                               ; preds = %26
  %41 = icmp eq i64 %25, %12
  br i1 %41, label %54, label %42

42:                                               ; preds = %20, %14, %11, %40
  %43 = phi i64 [ 0, %20 ], [ 0, %14 ], [ 0, %11 ], [ %25, %40 ]
  br label %44

44:                                               ; preds = %42, %44
  %45 = phi i64 [ %52, %44 ], [ %43, %42 ]
  %46 = add nuw nsw i64 %45, 3
  %47 = and i64 %46, 4294967295
  %48 = getelementptr inbounds nuw i64, ptr %7, i64 %47
  %49 = load i64, ptr %48, align 8, !tbaa !6
  %50 = add i64 %49, 10
  %51 = getelementptr inbounds nuw i64, ptr %5, i64 %45
  store i64 %50, ptr %51, align 8, !tbaa !6
  %52 = add nuw nsw i64 %45, 1
  %53 = icmp eq i64 %52, %12
  br i1 %53, label %54, label %44, !llvm.loop !125

54:                                               ; preds = %44, %40, %4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPmS0_jEZ4mainE3$_3E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_3", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPhS0_jEZ4mainE3$_4E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #15 {
  %5 = load ptr, ptr %1, align 8, !tbaa !54
  %6 = load ptr, ptr %2, align 8, !tbaa !54
  %7 = load i32, ptr %3, align 4, !tbaa !49
  %8 = icmp ugt i32 %7, 3
  br i1 %8, label %9, label %20

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 3, %9 ], [ %18, %11 ]
  %13 = getelementptr i8, ptr %6, i64 %12
  %14 = getelementptr i8, ptr %13, i64 -3
  %15 = load i8, ptr %14, align 1, !tbaa !15
  %16 = add i8 %15, 10
  %17 = getelementptr inbounds nuw i8, ptr %5, i64 %12
  store i8 %16, ptr %17, align 1, !tbaa !15
  %18 = add nuw nsw i64 %12, 1
  %19 = icmp eq i64 %18, %10
  br i1 %19, label %20, label %11, !llvm.loop !126

20:                                               ; preds = %11, %4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPhS0_jEZ4mainE3$_4E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_4", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPhS0_jEZ4mainE3$_5E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #15 {
  %5 = load ptr, ptr %1, align 8, !tbaa !54
  %6 = ptrtoint ptr %5 to i64
  %7 = load ptr, ptr %2, align 8, !tbaa !54
  %8 = ptrtoint ptr %7 to i64
  %9 = load i32, ptr %3, align 4, !tbaa !49
  %10 = icmp ugt i32 %9, 3
  br i1 %10, label %11, label %70

11:                                               ; preds = %4
  %12 = zext i32 %9 to i64
  %13 = add nsw i64 %12, -3
  %14 = icmp ult i64 %13, 8
  br i1 %14, label %59, label %15

15:                                               ; preds = %11
  %16 = add i64 %6, 3
  %17 = sub i64 %16, %8
  %18 = icmp ult i64 %17, 32
  br i1 %18, label %59, label %19

19:                                               ; preds = %15
  %20 = icmp ult i64 %13, 32
  br i1 %20, label %43, label %21

21:                                               ; preds = %19
  %22 = and i64 %13, -32
  br label %23

23:                                               ; preds = %23, %21
  %24 = phi i64 [ 0, %21 ], [ %35, %23 ]
  %25 = or disjoint i64 %24, 3
  %26 = getelementptr i8, ptr %7, i64 %25
  %27 = getelementptr i8, ptr %26, i64 -3
  %28 = getelementptr i8, ptr %26, i64 13
  %29 = load <16 x i8>, ptr %27, align 1, !tbaa !15
  %30 = load <16 x i8>, ptr %28, align 1, !tbaa !15
  %31 = add <16 x i8> %29, splat (i8 10)
  %32 = add <16 x i8> %30, splat (i8 10)
  %33 = getelementptr inbounds nuw i8, ptr %5, i64 %25
  %34 = getelementptr inbounds nuw i8, ptr %33, i64 16
  store <16 x i8> %31, ptr %33, align 1, !tbaa !15
  store <16 x i8> %32, ptr %34, align 1, !tbaa !15
  %35 = add nuw i64 %24, 32
  %36 = icmp eq i64 %35, %22
  br i1 %36, label %37, label %23, !llvm.loop !127

37:                                               ; preds = %23
  %38 = icmp eq i64 %13, %22
  br i1 %38, label %70, label %39

39:                                               ; preds = %37
  %40 = or disjoint i64 %22, 3
  %41 = and i64 %13, 24
  %42 = icmp eq i64 %41, 0
  br i1 %42, label %59, label %43

43:                                               ; preds = %39, %19
  %44 = phi i64 [ %22, %39 ], [ 0, %19 ]
  %45 = and i64 %13, -8
  %46 = or disjoint i64 %45, 3
  br label %47

47:                                               ; preds = %47, %43
  %48 = phi i64 [ %44, %43 ], [ %55, %47 ]
  %49 = or disjoint i64 %48, 3
  %50 = getelementptr i8, ptr %7, i64 %49
  %51 = getelementptr i8, ptr %50, i64 -3
  %52 = load <8 x i8>, ptr %51, align 1, !tbaa !15
  %53 = add <8 x i8> %52, splat (i8 10)
  %54 = getelementptr inbounds nuw i8, ptr %5, i64 %49
  store <8 x i8> %53, ptr %54, align 1, !tbaa !15
  %55 = add nuw i64 %48, 8
  %56 = icmp eq i64 %55, %45
  br i1 %56, label %57, label %47, !llvm.loop !128

57:                                               ; preds = %47
  %58 = icmp eq i64 %13, %45
  br i1 %58, label %70, label %59

59:                                               ; preds = %39, %57, %15, %11
  %60 = phi i64 [ 3, %11 ], [ 3, %15 ], [ %40, %39 ], [ %46, %57 ]
  br label %61

61:                                               ; preds = %59, %61
  %62 = phi i64 [ %68, %61 ], [ %60, %59 ]
  %63 = getelementptr i8, ptr %7, i64 %62
  %64 = getelementptr i8, ptr %63, i64 -3
  %65 = load i8, ptr %64, align 1, !tbaa !15
  %66 = add i8 %65, 10
  %67 = getelementptr inbounds nuw i8, ptr %5, i64 %62
  store i8 %66, ptr %67, align 1, !tbaa !15
  %68 = add nuw nsw i64 %62, 1
  %69 = icmp eq i64 %68, %12
  br i1 %69, label %70, label %61, !llvm.loop !129

70:                                               ; preds = %61, %37, %57, %4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPhS0_jEZ4mainE3$_5E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_5", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE3$_4E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #15 {
  %5 = load ptr, ptr %1, align 8, !tbaa !58
  %6 = load ptr, ptr %2, align 8, !tbaa !58
  %7 = load i32, ptr %3, align 4, !tbaa !49
  %8 = icmp ugt i32 %7, 3
  br i1 %8, label %9, label %20

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 3, %9 ], [ %18, %11 ]
  %13 = getelementptr i32, ptr %6, i64 %12
  %14 = getelementptr i8, ptr %13, i64 -12
  %15 = load i32, ptr %14, align 4, !tbaa !49
  %16 = add i32 %15, 10
  %17 = getelementptr inbounds nuw i32, ptr %5, i64 %12
  store i32 %16, ptr %17, align 4, !tbaa !49
  %18 = add nuw nsw i64 %12, 1
  %19 = icmp eq i64 %18, %10
  br i1 %19, label %20, label %11, !llvm.loop !130

20:                                               ; preds = %11, %4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE3$_4E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_4", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE3$_5E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #15 {
  %5 = load ptr, ptr %1, align 8, !tbaa !58
  %6 = ptrtoint ptr %5 to i64
  %7 = load ptr, ptr %2, align 8, !tbaa !58
  %8 = ptrtoint ptr %7 to i64
  %9 = load i32, ptr %3, align 4, !tbaa !49
  %10 = icmp ugt i32 %9, 3
  br i1 %10, label %11, label %49

11:                                               ; preds = %4
  %12 = zext i32 %9 to i64
  %13 = add nsw i64 %12, -3
  %14 = icmp ult i64 %13, 8
  br i1 %14, label %38, label %15

15:                                               ; preds = %11
  %16 = add i64 %6, 12
  %17 = sub i64 %16, %8
  %18 = icmp ult i64 %17, 32
  br i1 %18, label %38, label %19

19:                                               ; preds = %15
  %20 = and i64 %13, -8
  %21 = or disjoint i64 %20, 3
  br label %22

22:                                               ; preds = %22, %19
  %23 = phi i64 [ 0, %19 ], [ %34, %22 ]
  %24 = or disjoint i64 %23, 3
  %25 = getelementptr i32, ptr %7, i64 %24
  %26 = getelementptr i8, ptr %25, i64 -12
  %27 = getelementptr i8, ptr %25, i64 4
  %28 = load <4 x i32>, ptr %26, align 4, !tbaa !49
  %29 = load <4 x i32>, ptr %27, align 4, !tbaa !49
  %30 = add <4 x i32> %28, splat (i32 10)
  %31 = add <4 x i32> %29, splat (i32 10)
  %32 = getelementptr inbounds nuw i32, ptr %5, i64 %24
  %33 = getelementptr inbounds nuw i8, ptr %32, i64 16
  store <4 x i32> %30, ptr %32, align 4, !tbaa !49
  store <4 x i32> %31, ptr %33, align 4, !tbaa !49
  %34 = add nuw i64 %23, 8
  %35 = icmp eq i64 %34, %20
  br i1 %35, label %36, label %22, !llvm.loop !131

36:                                               ; preds = %22
  %37 = icmp eq i64 %13, %20
  br i1 %37, label %49, label %38

38:                                               ; preds = %15, %11, %36
  %39 = phi i64 [ 3, %15 ], [ 3, %11 ], [ %21, %36 ]
  br label %40

40:                                               ; preds = %38, %40
  %41 = phi i64 [ %47, %40 ], [ %39, %38 ]
  %42 = getelementptr i32, ptr %7, i64 %41
  %43 = getelementptr i8, ptr %42, i64 -12
  %44 = load i32, ptr %43, align 4, !tbaa !49
  %45 = add i32 %44, 10
  %46 = getelementptr inbounds nuw i32, ptr %5, i64 %41
  store i32 %45, ptr %46, align 4, !tbaa !49
  %47 = add nuw nsw i64 %41, 1
  %48 = icmp eq i64 %47, %12
  br i1 %48, label %49, label %40, !llvm.loop !132

49:                                               ; preds = %40, %36, %4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE3$_5E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_5", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPmS0_jEZ4mainE3$_4E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #15 {
  %5 = load ptr, ptr %1, align 8, !tbaa !62
  %6 = load ptr, ptr %2, align 8, !tbaa !62
  %7 = load i32, ptr %3, align 4, !tbaa !49
  %8 = icmp ugt i32 %7, 3
  br i1 %8, label %9, label %20

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 3, %9 ], [ %18, %11 ]
  %13 = getelementptr i64, ptr %6, i64 %12
  %14 = getelementptr i8, ptr %13, i64 -24
  %15 = load i64, ptr %14, align 8, !tbaa !6
  %16 = add i64 %15, 10
  %17 = getelementptr inbounds nuw i64, ptr %5, i64 %12
  store i64 %16, ptr %17, align 8, !tbaa !6
  %18 = add nuw nsw i64 %12, 1
  %19 = icmp eq i64 %18, %10
  br i1 %19, label %20, label %11, !llvm.loop !133

20:                                               ; preds = %11, %4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPmS0_jEZ4mainE3$_4E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_4", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPmS0_jEZ4mainE3$_5E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #15 {
  %5 = load ptr, ptr %1, align 8, !tbaa !62
  %6 = ptrtoint ptr %5 to i64
  %7 = load ptr, ptr %2, align 8, !tbaa !62
  %8 = ptrtoint ptr %7 to i64
  %9 = load i32, ptr %3, align 4, !tbaa !49
  %10 = icmp ugt i32 %9, 3
  br i1 %10, label %11, label %49

11:                                               ; preds = %4
  %12 = zext i32 %9 to i64
  %13 = add nsw i64 %12, -3
  %14 = icmp ult i64 %13, 4
  br i1 %14, label %38, label %15

15:                                               ; preds = %11
  %16 = add i64 %6, 24
  %17 = sub i64 %16, %8
  %18 = icmp ult i64 %17, 32
  br i1 %18, label %38, label %19

19:                                               ; preds = %15
  %20 = and i64 %13, -4
  %21 = or i64 %13, 3
  br label %22

22:                                               ; preds = %22, %19
  %23 = phi i64 [ 0, %19 ], [ %34, %22 ]
  %24 = or disjoint i64 %23, 3
  %25 = getelementptr i64, ptr %7, i64 %24
  %26 = getelementptr i8, ptr %25, i64 -24
  %27 = getelementptr i8, ptr %25, i64 -8
  %28 = load <2 x i64>, ptr %26, align 8, !tbaa !6
  %29 = load <2 x i64>, ptr %27, align 8, !tbaa !6
  %30 = add <2 x i64> %28, splat (i64 10)
  %31 = add <2 x i64> %29, splat (i64 10)
  %32 = getelementptr inbounds nuw i64, ptr %5, i64 %24
  %33 = getelementptr inbounds nuw i8, ptr %32, i64 16
  store <2 x i64> %30, ptr %32, align 8, !tbaa !6
  store <2 x i64> %31, ptr %33, align 8, !tbaa !6
  %34 = add nuw i64 %23, 4
  %35 = icmp eq i64 %34, %20
  br i1 %35, label %36, label %22, !llvm.loop !134

36:                                               ; preds = %22
  %37 = icmp eq i64 %13, %20
  br i1 %37, label %49, label %38

38:                                               ; preds = %15, %11, %36
  %39 = phi i64 [ 3, %15 ], [ 3, %11 ], [ %21, %36 ]
  br label %40

40:                                               ; preds = %38, %40
  %41 = phi i64 [ %47, %40 ], [ %39, %38 ]
  %42 = getelementptr i64, ptr %7, i64 %41
  %43 = getelementptr i8, ptr %42, i64 -24
  %44 = load i64, ptr %43, align 8, !tbaa !6
  %45 = add i64 %44, 10
  %46 = getelementptr inbounds nuw i64, ptr %5, i64 %41
  store i64 %45, ptr %46, align 8, !tbaa !6
  %47 = add nuw nsw i64 %41, 1
  %48 = icmp eq i64 %47, %12
  br i1 %48, label %49, label %40, !llvm.loop !135

49:                                               ; preds = %40, %36, %4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPmS0_jEZ4mainE3$_5E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_5", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPhS0_jEZ4mainE3$_6E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #15 {
  %5 = load ptr, ptr %1, align 8, !tbaa !54
  %6 = load ptr, ptr %2, align 8, !tbaa !54
  %7 = load i32, ptr %3, align 4, !tbaa !49
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %20, label %9

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ %10, %9 ], [ %17, %11 ]
  %13 = getelementptr inbounds nuw i8, ptr %6, i64 %12
  %14 = load i8, ptr %13, align 1, !tbaa !15
  %15 = add i8 %14, 10
  %16 = getelementptr inbounds nuw i8, ptr %5, i64 %12
  store i8 %15, ptr %16, align 1, !tbaa !15
  %17 = add nsw i64 %12, -1
  %18 = and i64 %17, 4294967295
  %19 = icmp eq i64 %18, 0
  br i1 %19, label %20, label %11, !llvm.loop !136

20:                                               ; preds = %11, %4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPhS0_jEZ4mainE3$_6E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_6", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPhS0_jEZ4mainE3$_7E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #15 {
  %5 = load ptr, ptr %1, align 8, !tbaa !54
  %6 = load ptr, ptr %2, align 8, !tbaa !54
  %7 = load i32, ptr %3, align 4, !tbaa !49
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %70, label %9

9:                                                ; preds = %4
  %10 = ptrtoint ptr %6 to i64
  %11 = ptrtoint ptr %5 to i64
  %12 = zext i32 %7 to i64
  %13 = icmp ult i32 %7, 8
  %14 = sub i64 %10, %11
  %15 = icmp ult i64 %14, 32
  %16 = select i1 %13, i1 true, i1 %15
  br i1 %16, label %59, label %17

17:                                               ; preds = %9
  %18 = icmp ult i32 %7, 32
  br i1 %18, label %42, label %19

19:                                               ; preds = %17
  %20 = and i64 %12, 4294967264
  br label %21

21:                                               ; preds = %21, %19
  %22 = phi i64 [ 0, %19 ], [ %34, %21 ]
  %23 = sub i64 %12, %22
  %24 = getelementptr inbounds nuw i8, ptr %6, i64 %23
  %25 = getelementptr inbounds i8, ptr %24, i64 -15
  %26 = getelementptr inbounds i8, ptr %24, i64 -31
  %27 = load <16 x i8>, ptr %25, align 1, !tbaa !15
  %28 = load <16 x i8>, ptr %26, align 1, !tbaa !15
  %29 = add <16 x i8> %27, splat (i8 10)
  %30 = add <16 x i8> %28, splat (i8 10)
  %31 = getelementptr inbounds nuw i8, ptr %5, i64 %23
  %32 = getelementptr inbounds i8, ptr %31, i64 -15
  %33 = getelementptr inbounds i8, ptr %31, i64 -31
  store <16 x i8> %29, ptr %32, align 1, !tbaa !15
  store <16 x i8> %30, ptr %33, align 1, !tbaa !15
  %34 = add nuw i64 %22, 32
  %35 = icmp eq i64 %34, %20
  br i1 %35, label %36, label %21, !llvm.loop !137

36:                                               ; preds = %21
  %37 = icmp eq i64 %20, %12
  br i1 %37, label %70, label %38

38:                                               ; preds = %36
  %39 = and i64 %12, 31
  %40 = and i64 %12, 24
  %41 = icmp eq i64 %40, 0
  br i1 %41, label %59, label %42

42:                                               ; preds = %38, %17
  %43 = phi i64 [ %20, %38 ], [ 0, %17 ]
  %44 = and i64 %12, 4294967288
  %45 = and i64 %12, 7
  br label %46

46:                                               ; preds = %46, %42
  %47 = phi i64 [ %43, %42 ], [ %55, %46 ]
  %48 = sub i64 %12, %47
  %49 = getelementptr inbounds nuw i8, ptr %6, i64 %48
  %50 = getelementptr inbounds i8, ptr %49, i64 -7
  %51 = load <8 x i8>, ptr %50, align 1, !tbaa !15
  %52 = add <8 x i8> %51, splat (i8 10)
  %53 = getelementptr inbounds nuw i8, ptr %5, i64 %48
  %54 = getelementptr inbounds i8, ptr %53, i64 -7
  store <8 x i8> %52, ptr %54, align 1, !tbaa !15
  %55 = add nuw i64 %47, 8
  %56 = icmp eq i64 %55, %44
  br i1 %56, label %57, label %46, !llvm.loop !138

57:                                               ; preds = %46
  %58 = icmp eq i64 %44, %12
  br i1 %58, label %70, label %59

59:                                               ; preds = %38, %57, %9
  %60 = phi i64 [ %12, %9 ], [ %39, %38 ], [ %45, %57 ]
  br label %61

61:                                               ; preds = %59, %61
  %62 = phi i64 [ %67, %61 ], [ %60, %59 ]
  %63 = getelementptr inbounds nuw i8, ptr %6, i64 %62
  %64 = load i8, ptr %63, align 1, !tbaa !15
  %65 = add i8 %64, 10
  %66 = getelementptr inbounds nuw i8, ptr %5, i64 %62
  store i8 %65, ptr %66, align 1, !tbaa !15
  %67 = add nsw i64 %62, -1
  %68 = and i64 %67, 4294967295
  %69 = icmp eq i64 %68, 0
  br i1 %69, label %70, label %61, !llvm.loop !139

70:                                               ; preds = %61, %36, %57, %4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPhS0_jEZ4mainE3$_7E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_7", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE3$_6E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #15 {
  %5 = load ptr, ptr %1, align 8, !tbaa !58
  %6 = load ptr, ptr %2, align 8, !tbaa !58
  %7 = load i32, ptr %3, align 4, !tbaa !49
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %20, label %9

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ %10, %9 ], [ %17, %11 ]
  %13 = getelementptr inbounds nuw i32, ptr %6, i64 %12
  %14 = load i32, ptr %13, align 4, !tbaa !49
  %15 = add i32 %14, 10
  %16 = getelementptr inbounds nuw i32, ptr %5, i64 %12
  store i32 %15, ptr %16, align 4, !tbaa !49
  %17 = add nsw i64 %12, -1
  %18 = and i64 %17, 4294967295
  %19 = icmp eq i64 %18, 0
  br i1 %19, label %20, label %11, !llvm.loop !140

20:                                               ; preds = %11, %4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE3$_6E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_6", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE3$_7E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #15 {
  %5 = load ptr, ptr %1, align 8, !tbaa !58
  %6 = load ptr, ptr %2, align 8, !tbaa !58
  %7 = load i32, ptr %3, align 4, !tbaa !49
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %48, label %9

9:                                                ; preds = %4
  %10 = ptrtoint ptr %6 to i64
  %11 = ptrtoint ptr %5 to i64
  %12 = zext i32 %7 to i64
  %13 = icmp ult i32 %7, 8
  %14 = sub i64 %10, %11
  %15 = icmp ult i64 %14, 32
  %16 = select i1 %13, i1 true, i1 %15
  br i1 %16, label %37, label %17

17:                                               ; preds = %9
  %18 = and i64 %12, 4294967288
  %19 = and i64 %12, 7
  br label %20

20:                                               ; preds = %20, %17
  %21 = phi i64 [ 0, %17 ], [ %33, %20 ]
  %22 = sub i64 %12, %21
  %23 = getelementptr inbounds nuw i32, ptr %6, i64 %22
  %24 = getelementptr inbounds i8, ptr %23, i64 -12
  %25 = getelementptr inbounds i8, ptr %23, i64 -28
  %26 = load <4 x i32>, ptr %24, align 4, !tbaa !49
  %27 = load <4 x i32>, ptr %25, align 4, !tbaa !49
  %28 = add <4 x i32> %26, splat (i32 10)
  %29 = add <4 x i32> %27, splat (i32 10)
  %30 = getelementptr inbounds nuw i32, ptr %5, i64 %22
  %31 = getelementptr inbounds i8, ptr %30, i64 -12
  %32 = getelementptr inbounds i8, ptr %30, i64 -28
  store <4 x i32> %28, ptr %31, align 4, !tbaa !49
  store <4 x i32> %29, ptr %32, align 4, !tbaa !49
  %33 = add nuw i64 %21, 8
  %34 = icmp eq i64 %33, %18
  br i1 %34, label %35, label %20, !llvm.loop !141

35:                                               ; preds = %20
  %36 = icmp eq i64 %18, %12
  br i1 %36, label %48, label %37

37:                                               ; preds = %9, %35
  %38 = phi i64 [ %12, %9 ], [ %19, %35 ]
  br label %39

39:                                               ; preds = %37, %39
  %40 = phi i64 [ %45, %39 ], [ %38, %37 ]
  %41 = getelementptr inbounds nuw i32, ptr %6, i64 %40
  %42 = load i32, ptr %41, align 4, !tbaa !49
  %43 = add i32 %42, 10
  %44 = getelementptr inbounds nuw i32, ptr %5, i64 %40
  store i32 %43, ptr %44, align 4, !tbaa !49
  %45 = add nsw i64 %40, -1
  %46 = and i64 %45, 4294967295
  %47 = icmp eq i64 %46, 0
  br i1 %47, label %48, label %39, !llvm.loop !142

48:                                               ; preds = %39, %35, %4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE3$_7E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_7", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPmS0_jEZ4mainE3$_6E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #15 {
  %5 = load ptr, ptr %1, align 8, !tbaa !62
  %6 = load ptr, ptr %2, align 8, !tbaa !62
  %7 = load i32, ptr %3, align 4, !tbaa !49
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %20, label %9

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ %10, %9 ], [ %17, %11 ]
  %13 = getelementptr inbounds nuw i64, ptr %6, i64 %12
  %14 = load i64, ptr %13, align 8, !tbaa !6
  %15 = add i64 %14, 10
  %16 = getelementptr inbounds nuw i64, ptr %5, i64 %12
  store i64 %15, ptr %16, align 8, !tbaa !6
  %17 = add nsw i64 %12, -1
  %18 = and i64 %17, 4294967295
  %19 = icmp eq i64 %18, 0
  br i1 %19, label %20, label %11, !llvm.loop !143

20:                                               ; preds = %11, %4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPmS0_jEZ4mainE3$_6E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_6", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPmS0_jEZ4mainE3$_7E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #15 {
  %5 = load ptr, ptr %1, align 8, !tbaa !62
  %6 = load ptr, ptr %2, align 8, !tbaa !62
  %7 = load i32, ptr %3, align 4, !tbaa !49
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %48, label %9

9:                                                ; preds = %4
  %10 = ptrtoint ptr %6 to i64
  %11 = ptrtoint ptr %5 to i64
  %12 = zext i32 %7 to i64
  %13 = icmp ult i32 %7, 4
  %14 = sub i64 %10, %11
  %15 = icmp ult i64 %14, 32
  %16 = select i1 %13, i1 true, i1 %15
  br i1 %16, label %37, label %17

17:                                               ; preds = %9
  %18 = and i64 %12, 4294967292
  %19 = and i64 %12, 3
  br label %20

20:                                               ; preds = %20, %17
  %21 = phi i64 [ 0, %17 ], [ %33, %20 ]
  %22 = sub i64 %12, %21
  %23 = getelementptr inbounds nuw i64, ptr %6, i64 %22
  %24 = getelementptr inbounds i8, ptr %23, i64 -8
  %25 = getelementptr inbounds i8, ptr %23, i64 -24
  %26 = load <2 x i64>, ptr %24, align 8, !tbaa !6
  %27 = load <2 x i64>, ptr %25, align 8, !tbaa !6
  %28 = add <2 x i64> %26, splat (i64 10)
  %29 = add <2 x i64> %27, splat (i64 10)
  %30 = getelementptr inbounds nuw i64, ptr %5, i64 %22
  %31 = getelementptr inbounds i8, ptr %30, i64 -8
  %32 = getelementptr inbounds i8, ptr %30, i64 -24
  store <2 x i64> %28, ptr %31, align 8, !tbaa !6
  store <2 x i64> %29, ptr %32, align 8, !tbaa !6
  %33 = add nuw i64 %21, 4
  %34 = icmp eq i64 %33, %18
  br i1 %34, label %35, label %20, !llvm.loop !144

35:                                               ; preds = %20
  %36 = icmp eq i64 %18, %12
  br i1 %36, label %48, label %37

37:                                               ; preds = %9, %35
  %38 = phi i64 [ %12, %9 ], [ %19, %35 ]
  br label %39

39:                                               ; preds = %37, %39
  %40 = phi i64 [ %45, %39 ], [ %38, %37 ]
  %41 = getelementptr inbounds nuw i64, ptr %6, i64 %40
  %42 = load i64, ptr %41, align 8, !tbaa !6
  %43 = add i64 %42, 10
  %44 = getelementptr inbounds nuw i64, ptr %5, i64 %40
  store i64 %43, ptr %44, align 8, !tbaa !6
  %45 = add nsw i64 %40, -1
  %46 = and i64 %45, 4294967295
  %47 = icmp eq i64 %46, 0
  br i1 %47, label %48, label %39, !llvm.loop !145

48:                                               ; preds = %39, %35, %4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPmS0_jEZ4mainE3$_7E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_7", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPhS0_jEZ4mainE3$_8E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #15 {
  %5 = load ptr, ptr %1, align 8, !tbaa !54
  %6 = load ptr, ptr %2, align 8, !tbaa !54
  %7 = load i32, ptr %3, align 4, !tbaa !49
  %8 = icmp ugt i32 %7, 2
  br i1 %8, label %9, label %20

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ %10, %9 ], [ %17, %11 ]
  %13 = getelementptr inbounds nuw i8, ptr %6, i64 %12
  %14 = load i8, ptr %13, align 1, !tbaa !15
  %15 = add i8 %14, 10
  %16 = getelementptr inbounds nuw i8, ptr %5, i64 %12
  store i8 %15, ptr %16, align 1, !tbaa !15
  %17 = add nsw i64 %12, -2
  %18 = trunc i64 %17 to i32
  %19 = icmp ugt i32 %18, 2
  br i1 %19, label %11, label %20, !llvm.loop !146

20:                                               ; preds = %11, %4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPhS0_jEZ4mainE3$_8E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_8", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPhS0_jEZ4mainE3$_9E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #15 {
  %5 = load ptr, ptr %1, align 8, !tbaa !54
  %6 = load ptr, ptr %2, align 8, !tbaa !54
  %7 = load i32, ptr %3, align 4, !tbaa !49
  %8 = icmp ugt i32 %7, 2
  br i1 %8, label %9, label %148

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  %11 = add i32 %7, -3
  %12 = lshr i32 %11, 1
  %13 = add nuw i32 %12, 1
  %14 = zext i32 %13 to i64
  %15 = icmp ult i32 %11, 30
  br i1 %15, label %137, label %16

16:                                               ; preds = %9
  %17 = add i32 %7, -3
  %18 = and i32 %17, -2
  %19 = zext i32 %18 to i64
  %20 = sub nsw i64 %10, %19
  %21 = getelementptr i8, ptr %5, i64 %20
  %22 = add nuw nsw i64 %10, 1
  %23 = getelementptr i8, ptr %5, i64 %22
  %24 = getelementptr i8, ptr %6, i64 %20
  %25 = getelementptr i8, ptr %6, i64 %22
  %26 = icmp ult ptr %21, %25
  %27 = icmp ult ptr %24, %23
  %28 = and i1 %26, %27
  br i1 %28, label %137, label %29

29:                                               ; preds = %16
  %30 = and i64 %14, 4294967280
  %31 = shl nuw nsw i64 %30, 1
  %32 = sub nsw i64 %10, %31
  br label %33

33:                                               ; preds = %33, %29
  %34 = phi i64 [ 0, %29 ], [ %133, %33 ]
  %35 = shl i64 %34, 1
  %36 = sub i64 %10, %35
  %37 = add i64 %36, -2
  %38 = add i64 %36, -4
  %39 = add i64 %36, -6
  %40 = add i64 %36, -8
  %41 = add i64 %36, -10
  %42 = add i64 %36, -12
  %43 = add i64 %36, -14
  %44 = add i64 %36, -16
  %45 = add i64 %36, -18
  %46 = add i64 %36, -20
  %47 = add i64 %36, -22
  %48 = add i64 %36, -24
  %49 = add i64 %36, -26
  %50 = add i64 %36, -28
  %51 = add i64 %36, -30
  %52 = getelementptr inbounds nuw i8, ptr %6, i64 %36
  %53 = getelementptr inbounds nuw i8, ptr %6, i64 %37
  %54 = getelementptr inbounds nuw i8, ptr %6, i64 %38
  %55 = getelementptr inbounds nuw i8, ptr %6, i64 %39
  %56 = getelementptr inbounds nuw i8, ptr %6, i64 %40
  %57 = getelementptr inbounds nuw i8, ptr %6, i64 %41
  %58 = getelementptr inbounds nuw i8, ptr %6, i64 %42
  %59 = getelementptr inbounds nuw i8, ptr %6, i64 %43
  %60 = getelementptr inbounds nuw i8, ptr %6, i64 %44
  %61 = getelementptr inbounds nuw i8, ptr %6, i64 %45
  %62 = getelementptr inbounds nuw i8, ptr %6, i64 %46
  %63 = getelementptr inbounds nuw i8, ptr %6, i64 %47
  %64 = getelementptr inbounds nuw i8, ptr %6, i64 %48
  %65 = getelementptr inbounds nuw i8, ptr %6, i64 %49
  %66 = getelementptr inbounds nuw i8, ptr %6, i64 %50
  %67 = getelementptr inbounds nuw i8, ptr %6, i64 %51
  %68 = load i8, ptr %52, align 1, !tbaa !15, !alias.scope !147
  %69 = load i8, ptr %53, align 1, !tbaa !15, !alias.scope !147
  %70 = load i8, ptr %54, align 1, !tbaa !15, !alias.scope !147
  %71 = load i8, ptr %55, align 1, !tbaa !15, !alias.scope !147
  %72 = load i8, ptr %56, align 1, !tbaa !15, !alias.scope !147
  %73 = load i8, ptr %57, align 1, !tbaa !15, !alias.scope !147
  %74 = load i8, ptr %58, align 1, !tbaa !15, !alias.scope !147
  %75 = load i8, ptr %59, align 1, !tbaa !15, !alias.scope !147
  %76 = load i8, ptr %60, align 1, !tbaa !15, !alias.scope !147
  %77 = load i8, ptr %61, align 1, !tbaa !15, !alias.scope !147
  %78 = load i8, ptr %62, align 1, !tbaa !15, !alias.scope !147
  %79 = load i8, ptr %63, align 1, !tbaa !15, !alias.scope !147
  %80 = load i8, ptr %64, align 1, !tbaa !15, !alias.scope !147
  %81 = load i8, ptr %65, align 1, !tbaa !15, !alias.scope !147
  %82 = load i8, ptr %66, align 1, !tbaa !15, !alias.scope !147
  %83 = load i8, ptr %67, align 1, !tbaa !15, !alias.scope !147
  %84 = insertelement <16 x i8> poison, i8 %68, i64 0
  %85 = insertelement <16 x i8> %84, i8 %69, i64 1
  %86 = insertelement <16 x i8> %85, i8 %70, i64 2
  %87 = insertelement <16 x i8> %86, i8 %71, i64 3
  %88 = insertelement <16 x i8> %87, i8 %72, i64 4
  %89 = insertelement <16 x i8> %88, i8 %73, i64 5
  %90 = insertelement <16 x i8> %89, i8 %74, i64 6
  %91 = insertelement <16 x i8> %90, i8 %75, i64 7
  %92 = insertelement <16 x i8> %91, i8 %76, i64 8
  %93 = insertelement <16 x i8> %92, i8 %77, i64 9
  %94 = insertelement <16 x i8> %93, i8 %78, i64 10
  %95 = insertelement <16 x i8> %94, i8 %79, i64 11
  %96 = insertelement <16 x i8> %95, i8 %80, i64 12
  %97 = insertelement <16 x i8> %96, i8 %81, i64 13
  %98 = insertelement <16 x i8> %97, i8 %82, i64 14
  %99 = insertelement <16 x i8> %98, i8 %83, i64 15
  %100 = add <16 x i8> %99, splat (i8 10)
  %101 = getelementptr inbounds nuw i8, ptr %5, i64 %36
  %102 = getelementptr inbounds nuw i8, ptr %5, i64 %37
  %103 = getelementptr inbounds nuw i8, ptr %5, i64 %38
  %104 = getelementptr inbounds nuw i8, ptr %5, i64 %39
  %105 = getelementptr inbounds nuw i8, ptr %5, i64 %40
  %106 = getelementptr inbounds nuw i8, ptr %5, i64 %41
  %107 = getelementptr inbounds nuw i8, ptr %5, i64 %42
  %108 = getelementptr inbounds nuw i8, ptr %5, i64 %43
  %109 = getelementptr inbounds nuw i8, ptr %5, i64 %44
  %110 = getelementptr inbounds nuw i8, ptr %5, i64 %45
  %111 = getelementptr inbounds nuw i8, ptr %5, i64 %46
  %112 = getelementptr inbounds nuw i8, ptr %5, i64 %47
  %113 = getelementptr inbounds nuw i8, ptr %5, i64 %48
  %114 = getelementptr inbounds nuw i8, ptr %5, i64 %49
  %115 = getelementptr inbounds nuw i8, ptr %5, i64 %50
  %116 = getelementptr inbounds nuw i8, ptr %5, i64 %51
  %117 = extractelement <16 x i8> %100, i64 0
  store i8 %117, ptr %101, align 1, !tbaa !15, !alias.scope !150, !noalias !147
  %118 = extractelement <16 x i8> %100, i64 1
  store i8 %118, ptr %102, align 1, !tbaa !15, !alias.scope !150, !noalias !147
  %119 = extractelement <16 x i8> %100, i64 2
  store i8 %119, ptr %103, align 1, !tbaa !15, !alias.scope !150, !noalias !147
  %120 = extractelement <16 x i8> %100, i64 3
  store i8 %120, ptr %104, align 1, !tbaa !15, !alias.scope !150, !noalias !147
  %121 = extractelement <16 x i8> %100, i64 4
  store i8 %121, ptr %105, align 1, !tbaa !15, !alias.scope !150, !noalias !147
  %122 = extractelement <16 x i8> %100, i64 5
  store i8 %122, ptr %106, align 1, !tbaa !15, !alias.scope !150, !noalias !147
  %123 = extractelement <16 x i8> %100, i64 6
  store i8 %123, ptr %107, align 1, !tbaa !15, !alias.scope !150, !noalias !147
  %124 = extractelement <16 x i8> %100, i64 7
  store i8 %124, ptr %108, align 1, !tbaa !15, !alias.scope !150, !noalias !147
  %125 = extractelement <16 x i8> %100, i64 8
  store i8 %125, ptr %109, align 1, !tbaa !15, !alias.scope !150, !noalias !147
  %126 = extractelement <16 x i8> %100, i64 9
  store i8 %126, ptr %110, align 1, !tbaa !15, !alias.scope !150, !noalias !147
  %127 = extractelement <16 x i8> %100, i64 10
  store i8 %127, ptr %111, align 1, !tbaa !15, !alias.scope !150, !noalias !147
  %128 = extractelement <16 x i8> %100, i64 11
  store i8 %128, ptr %112, align 1, !tbaa !15, !alias.scope !150, !noalias !147
  %129 = extractelement <16 x i8> %100, i64 12
  store i8 %129, ptr %113, align 1, !tbaa !15, !alias.scope !150, !noalias !147
  %130 = extractelement <16 x i8> %100, i64 13
  store i8 %130, ptr %114, align 1, !tbaa !15, !alias.scope !150, !noalias !147
  %131 = extractelement <16 x i8> %100, i64 14
  store i8 %131, ptr %115, align 1, !tbaa !15, !alias.scope !150, !noalias !147
  %132 = extractelement <16 x i8> %100, i64 15
  store i8 %132, ptr %116, align 1, !tbaa !15, !alias.scope !150, !noalias !147
  %133 = add nuw i64 %34, 16
  %134 = icmp eq i64 %133, %30
  br i1 %134, label %135, label %33, !llvm.loop !152

135:                                              ; preds = %33
  %136 = icmp eq i64 %30, %14
  br i1 %136, label %148, label %137

137:                                              ; preds = %16, %9, %135
  %138 = phi i64 [ %10, %16 ], [ %10, %9 ], [ %32, %135 ]
  br label %139

139:                                              ; preds = %137, %139
  %140 = phi i64 [ %145, %139 ], [ %138, %137 ]
  %141 = getelementptr inbounds nuw i8, ptr %6, i64 %140
  %142 = load i8, ptr %141, align 1, !tbaa !15
  %143 = add i8 %142, 10
  %144 = getelementptr inbounds nuw i8, ptr %5, i64 %140
  store i8 %143, ptr %144, align 1, !tbaa !15
  %145 = add nsw i64 %140, -2
  %146 = trunc i64 %145 to i32
  %147 = icmp ugt i32 %146, 2
  br i1 %147, label %139, label %148, !llvm.loop !153

148:                                              ; preds = %139, %135, %4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPhS0_jEZ4mainE3$_9E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_9", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE3$_8E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #15 {
  %5 = load ptr, ptr %1, align 8, !tbaa !58
  %6 = load ptr, ptr %2, align 8, !tbaa !58
  %7 = load i32, ptr %3, align 4, !tbaa !49
  %8 = icmp ugt i32 %7, 2
  br i1 %8, label %9, label %20

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ %10, %9 ], [ %17, %11 ]
  %13 = getelementptr inbounds nuw i32, ptr %6, i64 %12
  %14 = load i32, ptr %13, align 4, !tbaa !49
  %15 = add i32 %14, 10
  %16 = getelementptr inbounds nuw i32, ptr %5, i64 %12
  store i32 %15, ptr %16, align 4, !tbaa !49
  %17 = add nsw i64 %12, -2
  %18 = trunc i64 %17 to i32
  %19 = icmp ugt i32 %18, 2
  br i1 %19, label %11, label %20, !llvm.loop !154

20:                                               ; preds = %11, %4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE3$_8E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_8", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE3$_9E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #15 {
  %5 = load ptr, ptr %1, align 8, !tbaa !58
  %6 = load ptr, ptr %2, align 8, !tbaa !58
  %7 = load i32, ptr %3, align 4, !tbaa !49
  %8 = icmp ugt i32 %7, 2
  br i1 %8, label %9, label %78

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  %11 = add i32 %7, -3
  %12 = lshr i32 %11, 1
  %13 = add nuw i32 %12, 1
  %14 = zext i32 %13 to i64
  %15 = icmp ult i32 %11, 6
  br i1 %15, label %67, label %16

16:                                               ; preds = %9
  %17 = shl nuw nsw i64 %10, 2
  %18 = add i32 %7, -3
  %19 = lshr i32 %18, 1
  %20 = zext nneg i32 %19 to i64
  %21 = shl nuw nsw i64 %20, 3
  %22 = sub nsw i64 %17, %21
  %23 = getelementptr i8, ptr %5, i64 %22
  %24 = add nuw nsw i64 %17, 4
  %25 = getelementptr i8, ptr %5, i64 %24
  %26 = getelementptr i8, ptr %6, i64 %22
  %27 = getelementptr i8, ptr %6, i64 %24
  %28 = icmp ult ptr %23, %27
  %29 = icmp ult ptr %26, %25
  %30 = and i1 %28, %29
  br i1 %30, label %67, label %31

31:                                               ; preds = %16
  %32 = and i64 %14, 4294967292
  %33 = shl nuw nsw i64 %32, 1
  %34 = sub nsw i64 %10, %33
  br label %35

35:                                               ; preds = %35, %31
  %36 = phi i64 [ 0, %31 ], [ %63, %35 ]
  %37 = shl i64 %36, 1
  %38 = sub i64 %10, %37
  %39 = add i64 %38, -2
  %40 = add i64 %38, -4
  %41 = add i64 %38, -6
  %42 = getelementptr inbounds nuw i32, ptr %6, i64 %38
  %43 = getelementptr inbounds nuw i32, ptr %6, i64 %39
  %44 = getelementptr inbounds nuw i32, ptr %6, i64 %40
  %45 = getelementptr inbounds nuw i32, ptr %6, i64 %41
  %46 = load i32, ptr %42, align 4, !tbaa !49, !alias.scope !155
  %47 = load i32, ptr %43, align 4, !tbaa !49, !alias.scope !155
  %48 = load i32, ptr %44, align 4, !tbaa !49, !alias.scope !155
  %49 = load i32, ptr %45, align 4, !tbaa !49, !alias.scope !155
  %50 = insertelement <4 x i32> poison, i32 %46, i64 0
  %51 = insertelement <4 x i32> %50, i32 %47, i64 1
  %52 = insertelement <4 x i32> %51, i32 %48, i64 2
  %53 = insertelement <4 x i32> %52, i32 %49, i64 3
  %54 = add <4 x i32> %53, splat (i32 10)
  %55 = getelementptr inbounds nuw i32, ptr %5, i64 %38
  %56 = getelementptr inbounds nuw i32, ptr %5, i64 %39
  %57 = getelementptr inbounds nuw i32, ptr %5, i64 %40
  %58 = getelementptr inbounds nuw i32, ptr %5, i64 %41
  %59 = extractelement <4 x i32> %54, i64 0
  store i32 %59, ptr %55, align 4, !tbaa !49, !alias.scope !158, !noalias !155
  %60 = extractelement <4 x i32> %54, i64 1
  store i32 %60, ptr %56, align 4, !tbaa !49, !alias.scope !158, !noalias !155
  %61 = extractelement <4 x i32> %54, i64 2
  store i32 %61, ptr %57, align 4, !tbaa !49, !alias.scope !158, !noalias !155
  %62 = extractelement <4 x i32> %54, i64 3
  store i32 %62, ptr %58, align 4, !tbaa !49, !alias.scope !158, !noalias !155
  %63 = add nuw i64 %36, 4
  %64 = icmp eq i64 %63, %32
  br i1 %64, label %65, label %35, !llvm.loop !160

65:                                               ; preds = %35
  %66 = icmp eq i64 %32, %14
  br i1 %66, label %78, label %67

67:                                               ; preds = %16, %9, %65
  %68 = phi i64 [ %10, %16 ], [ %10, %9 ], [ %34, %65 ]
  br label %69

69:                                               ; preds = %67, %69
  %70 = phi i64 [ %75, %69 ], [ %68, %67 ]
  %71 = getelementptr inbounds nuw i32, ptr %6, i64 %70
  %72 = load i32, ptr %71, align 4, !tbaa !49
  %73 = add i32 %72, 10
  %74 = getelementptr inbounds nuw i32, ptr %5, i64 %70
  store i32 %73, ptr %74, align 4, !tbaa !49
  %75 = add nsw i64 %70, -2
  %76 = trunc i64 %75 to i32
  %77 = icmp ugt i32 %76, 2
  br i1 %77, label %69, label %78, !llvm.loop !161

78:                                               ; preds = %69, %65, %4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE3$_9E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_9", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPmS0_jEZ4mainE3$_8E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #15 {
  %5 = load ptr, ptr %1, align 8, !tbaa !62
  %6 = load ptr, ptr %2, align 8, !tbaa !62
  %7 = load i32, ptr %3, align 4, !tbaa !49
  %8 = icmp ugt i32 %7, 2
  br i1 %8, label %9, label %20

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ %10, %9 ], [ %17, %11 ]
  %13 = getelementptr inbounds nuw i64, ptr %6, i64 %12
  %14 = load i64, ptr %13, align 8, !tbaa !6
  %15 = add i64 %14, 10
  %16 = getelementptr inbounds nuw i64, ptr %5, i64 %12
  store i64 %15, ptr %16, align 8, !tbaa !6
  %17 = add nsw i64 %12, -2
  %18 = trunc i64 %17 to i32
  %19 = icmp ugt i32 %18, 2
  br i1 %19, label %11, label %20, !llvm.loop !162

20:                                               ; preds = %11, %4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPmS0_jEZ4mainE3$_8E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_8", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPmS0_jEZ4mainE3$_9E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #15 {
  %5 = load ptr, ptr %1, align 8, !tbaa !62
  %6 = load ptr, ptr %2, align 8, !tbaa !62
  %7 = load i32, ptr %3, align 4, !tbaa !49
  %8 = icmp ugt i32 %7, 2
  br i1 %8, label %9, label %66

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  %11 = add i32 %7, -3
  %12 = lshr i32 %11, 1
  %13 = add nuw i32 %12, 1
  %14 = zext i32 %13 to i64
  %15 = icmp ult i32 %11, 2
  br i1 %15, label %55, label %16

16:                                               ; preds = %9
  %17 = shl nuw nsw i64 %10, 3
  %18 = add i32 %7, -3
  %19 = lshr i32 %18, 1
  %20 = zext nneg i32 %19 to i64
  %21 = shl nuw nsw i64 %20, 4
  %22 = sub nsw i64 %17, %21
  %23 = getelementptr i8, ptr %5, i64 %22
  %24 = add nuw nsw i64 %17, 8
  %25 = getelementptr i8, ptr %5, i64 %24
  %26 = getelementptr i8, ptr %6, i64 %22
  %27 = getelementptr i8, ptr %6, i64 %24
  %28 = icmp ult ptr %23, %27
  %29 = icmp ult ptr %26, %25
  %30 = and i1 %28, %29
  br i1 %30, label %55, label %31

31:                                               ; preds = %16
  %32 = and i64 %14, 4294967294
  %33 = shl nuw nsw i64 %32, 1
  %34 = sub nsw i64 %10, %33
  br label %35

35:                                               ; preds = %35, %31
  %36 = phi i64 [ 0, %31 ], [ %51, %35 ]
  %37 = shl i64 %36, 1
  %38 = sub i64 %10, %37
  %39 = add i64 %38, -2
  %40 = getelementptr inbounds nuw i64, ptr %6, i64 %38
  %41 = getelementptr inbounds nuw i64, ptr %6, i64 %39
  %42 = load i64, ptr %40, align 8, !tbaa !6, !alias.scope !163
  %43 = load i64, ptr %41, align 8, !tbaa !6, !alias.scope !163
  %44 = insertelement <2 x i64> poison, i64 %42, i64 0
  %45 = insertelement <2 x i64> %44, i64 %43, i64 1
  %46 = add <2 x i64> %45, splat (i64 10)
  %47 = getelementptr inbounds nuw i64, ptr %5, i64 %38
  %48 = getelementptr inbounds nuw i64, ptr %5, i64 %39
  %49 = extractelement <2 x i64> %46, i64 0
  store i64 %49, ptr %47, align 8, !tbaa !6, !alias.scope !166, !noalias !163
  %50 = extractelement <2 x i64> %46, i64 1
  store i64 %50, ptr %48, align 8, !tbaa !6, !alias.scope !166, !noalias !163
  %51 = add nuw i64 %36, 2
  %52 = icmp eq i64 %51, %32
  br i1 %52, label %53, label %35, !llvm.loop !168

53:                                               ; preds = %35
  %54 = icmp eq i64 %32, %14
  br i1 %54, label %66, label %55

55:                                               ; preds = %16, %9, %53
  %56 = phi i64 [ %10, %16 ], [ %10, %9 ], [ %34, %53 ]
  br label %57

57:                                               ; preds = %55, %57
  %58 = phi i64 [ %63, %57 ], [ %56, %55 ]
  %59 = getelementptr inbounds nuw i64, ptr %6, i64 %58
  %60 = load i64, ptr %59, align 8, !tbaa !6
  %61 = add i64 %60, 10
  %62 = getelementptr inbounds nuw i64, ptr %5, i64 %58
  store i64 %61, ptr %62, align 8, !tbaa !6
  %63 = add nsw i64 %58, -2
  %64 = trunc i64 %63 to i32
  %65 = icmp ugt i32 %64, 2
  br i1 %65, label %57, label %66, !llvm.loop !169

66:                                               ; preds = %57, %53, %4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPmS0_jEZ4mainE3$_9E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_9", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPhS0_jEZ4mainE4$_10E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #15 {
  %5 = load ptr, ptr %1, align 8, !tbaa !54
  %6 = load ptr, ptr %2, align 8, !tbaa !54
  %7 = load i32, ptr %3, align 4, !tbaa !49
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %22, label %9

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 0, %9 ], [ %20, %11 ]
  %13 = phi i32 [ 0, %9 ], [ %19, %11 ]
  %14 = zext i32 %13 to i64
  %15 = getelementptr inbounds nuw i8, ptr %6, i64 %14
  %16 = load i8, ptr %15, align 1, !tbaa !15
  %17 = add i8 %16, 10
  %18 = getelementptr inbounds nuw i8, ptr %5, i64 %12
  store i8 %17, ptr %18, align 1, !tbaa !15
  %19 = add i32 %13, 2
  %20 = add nuw nsw i64 %12, 1
  %21 = icmp eq i64 %20, %10
  br i1 %21, label %22, label %11, !llvm.loop !170

22:                                               ; preds = %11, %4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPhS0_jEZ4mainE4$_10E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_10", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPhS0_jEZ4mainE4$_11E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #15 {
  %5 = load ptr, ptr %1, align 8, !tbaa !54
  %6 = load ptr, ptr %2, align 8, !tbaa !54
  %7 = load i32, ptr %3, align 4, !tbaa !49
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %83, label %9

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  %11 = icmp ult i32 %7, 9
  br i1 %11, label %69, label %12

12:                                               ; preds = %9
  %13 = getelementptr i8, ptr %5, i64 %10
  %14 = shl nuw nsw i64 %10, 1
  %15 = getelementptr i8, ptr %6, i64 %14
  %16 = getelementptr i8, ptr %15, i64 -1
  %17 = icmp ult ptr %5, %16
  %18 = icmp ult ptr %6, %13
  %19 = and i1 %17, %18
  br i1 %19, label %69, label %20

20:                                               ; preds = %12
  %21 = icmp ult i32 %7, 33
  br i1 %21, label %50, label %22

22:                                               ; preds = %20
  %23 = and i64 %10, 31
  %24 = icmp eq i64 %23, 0
  %25 = select i1 %24, i64 32, i64 %23
  %26 = sub nsw i64 %10, %25
  br label %27

27:                                               ; preds = %27, %22
  %28 = phi i64 [ 0, %22 ], [ %44, %27 ]
  %29 = trunc i64 %28 to i32
  %30 = shl i32 %29, 1
  %31 = or disjoint i32 %30, 32
  %32 = zext i32 %30 to i64
  %33 = zext i32 %31 to i64
  %34 = getelementptr inbounds nuw i8, ptr %6, i64 %32
  %35 = getelementptr inbounds nuw i8, ptr %6, i64 %33
  %36 = load <31 x i8>, ptr %34, align 1, !tbaa !15, !alias.scope !171
  %37 = shufflevector <31 x i8> %36, <31 x i8> poison, <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30>
  %38 = load <31 x i8>, ptr %35, align 1, !tbaa !15, !alias.scope !171
  %39 = shufflevector <31 x i8> %38, <31 x i8> poison, <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30>
  %40 = add <16 x i8> %37, splat (i8 10)
  %41 = add <16 x i8> %39, splat (i8 10)
  %42 = getelementptr inbounds nuw i8, ptr %5, i64 %28
  %43 = getelementptr inbounds nuw i8, ptr %42, i64 16
  store <16 x i8> %40, ptr %42, align 1, !tbaa !15, !alias.scope !174, !noalias !171
  store <16 x i8> %41, ptr %43, align 1, !tbaa !15, !alias.scope !174, !noalias !171
  %44 = add nuw i64 %28, 32
  %45 = icmp eq i64 %44, %26
  br i1 %45, label %46, label %27, !llvm.loop !176

46:                                               ; preds = %27
  %47 = trunc i64 %26 to i32
  %48 = shl i32 %47, 1
  %49 = icmp samesign ult i64 %25, 9
  br i1 %49, label %69, label %50

50:                                               ; preds = %46, %20
  %51 = phi i64 [ %26, %46 ], [ 0, %20 ]
  %52 = and i64 %10, 7
  %53 = icmp eq i64 %52, 0
  %54 = select i1 %53, i64 8, i64 %52
  %55 = sub nsw i64 %10, %54
  %56 = trunc i64 %55 to i32
  %57 = shl i32 %56, 1
  br label %58

58:                                               ; preds = %58, %50
  %59 = phi i64 [ %51, %50 ], [ %67, %58 ]
  %60 = shl i64 %59, 1
  %61 = and i64 %60, 4294967294
  %62 = getelementptr inbounds nuw i8, ptr %6, i64 %61
  %63 = load <16 x i8>, ptr %62, align 1, !tbaa !15, !alias.scope !171
  %64 = shufflevector <16 x i8> %63, <16 x i8> poison, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
  %65 = add <8 x i8> %64, splat (i8 10)
  %66 = getelementptr inbounds nuw i8, ptr %5, i64 %59
  store <8 x i8> %65, ptr %66, align 1, !tbaa !15, !alias.scope !174, !noalias !171
  %67 = add nuw i64 %59, 8
  %68 = icmp eq i64 %67, %55
  br i1 %68, label %69, label %58, !llvm.loop !177

69:                                               ; preds = %58, %46, %12, %9
  %70 = phi i64 [ 0, %9 ], [ 0, %12 ], [ %26, %46 ], [ %55, %58 ]
  %71 = phi i32 [ 0, %9 ], [ 0, %12 ], [ %48, %46 ], [ %57, %58 ]
  br label %72

72:                                               ; preds = %69, %72
  %73 = phi i64 [ %81, %72 ], [ %70, %69 ]
  %74 = phi i32 [ %80, %72 ], [ %71, %69 ]
  %75 = zext i32 %74 to i64
  %76 = getelementptr inbounds nuw i8, ptr %6, i64 %75
  %77 = load i8, ptr %76, align 1, !tbaa !15
  %78 = add i8 %77, 10
  %79 = getelementptr inbounds nuw i8, ptr %5, i64 %73
  store i8 %78, ptr %79, align 1, !tbaa !15
  %80 = add i32 %74, 2
  %81 = add nuw nsw i64 %73, 1
  %82 = icmp eq i64 %81, %10
  br i1 %82, label %83, label %72, !llvm.loop !178

83:                                               ; preds = %72, %4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPhS0_jEZ4mainE4$_11E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_11", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE4$_10E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #15 {
  %5 = load ptr, ptr %1, align 8, !tbaa !58
  %6 = load ptr, ptr %2, align 8, !tbaa !58
  %7 = load i32, ptr %3, align 4, !tbaa !49
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %22, label %9

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 0, %9 ], [ %20, %11 ]
  %13 = phi i32 [ 0, %9 ], [ %19, %11 ]
  %14 = zext i32 %13 to i64
  %15 = getelementptr inbounds nuw i32, ptr %6, i64 %14
  %16 = load i32, ptr %15, align 4, !tbaa !49
  %17 = add i32 %16, 10
  %18 = getelementptr inbounds nuw i32, ptr %5, i64 %12
  store i32 %17, ptr %18, align 4, !tbaa !49
  %19 = add i32 %13, 2
  %20 = add nuw nsw i64 %12, 1
  %21 = icmp eq i64 %20, %10
  br i1 %21, label %22, label %11, !llvm.loop !179

22:                                               ; preds = %11, %4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE4$_10E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_10", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE4$_11E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #15 {
  %5 = load ptr, ptr %1, align 8, !tbaa !58
  %6 = load ptr, ptr %2, align 8, !tbaa !58
  %7 = load i32, ptr %3, align 4, !tbaa !49
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %61, label %9

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  %11 = icmp ult i32 %7, 9
  br i1 %11, label %47, label %12

12:                                               ; preds = %9
  %13 = shl nuw nsw i64 %10, 2
  %14 = getelementptr i8, ptr %5, i64 %13
  %15 = shl nuw nsw i64 %10, 3
  %16 = getelementptr i8, ptr %6, i64 %15
  %17 = getelementptr i8, ptr %16, i64 -4
  %18 = icmp ult ptr %5, %17
  %19 = icmp ult ptr %6, %14
  %20 = and i1 %18, %19
  br i1 %20, label %47, label %21

21:                                               ; preds = %12
  %22 = and i64 %10, 7
  %23 = icmp eq i64 %22, 0
  %24 = select i1 %23, i64 8, i64 %22
  %25 = sub nsw i64 %10, %24
  %26 = trunc i64 %25 to i32
  %27 = shl i32 %26, 1
  br label %28

28:                                               ; preds = %28, %21
  %29 = phi i64 [ 0, %21 ], [ %45, %28 ]
  %30 = trunc i64 %29 to i32
  %31 = shl i32 %30, 1
  %32 = or disjoint i32 %31, 8
  %33 = zext i32 %31 to i64
  %34 = zext i32 %32 to i64
  %35 = getelementptr inbounds nuw i32, ptr %6, i64 %33
  %36 = getelementptr inbounds nuw i32, ptr %6, i64 %34
  %37 = load <7 x i32>, ptr %35, align 4, !tbaa !49, !alias.scope !180
  %38 = shufflevector <7 x i32> %37, <7 x i32> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %39 = load <7 x i32>, ptr %36, align 4, !tbaa !49, !alias.scope !180
  %40 = shufflevector <7 x i32> %39, <7 x i32> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %41 = add <4 x i32> %38, splat (i32 10)
  %42 = add <4 x i32> %40, splat (i32 10)
  %43 = getelementptr inbounds nuw i32, ptr %5, i64 %29
  %44 = getelementptr inbounds nuw i8, ptr %43, i64 16
  store <4 x i32> %41, ptr %43, align 4, !tbaa !49, !alias.scope !183, !noalias !180
  store <4 x i32> %42, ptr %44, align 4, !tbaa !49, !alias.scope !183, !noalias !180
  %45 = add nuw i64 %29, 8
  %46 = icmp eq i64 %45, %25
  br i1 %46, label %47, label %28, !llvm.loop !185

47:                                               ; preds = %28, %12, %9
  %48 = phi i64 [ 0, %12 ], [ 0, %9 ], [ %25, %28 ]
  %49 = phi i32 [ 0, %12 ], [ 0, %9 ], [ %27, %28 ]
  br label %50

50:                                               ; preds = %47, %50
  %51 = phi i64 [ %59, %50 ], [ %48, %47 ]
  %52 = phi i32 [ %58, %50 ], [ %49, %47 ]
  %53 = zext i32 %52 to i64
  %54 = getelementptr inbounds nuw i32, ptr %6, i64 %53
  %55 = load i32, ptr %54, align 4, !tbaa !49
  %56 = add i32 %55, 10
  %57 = getelementptr inbounds nuw i32, ptr %5, i64 %51
  store i32 %56, ptr %57, align 4, !tbaa !49
  %58 = add i32 %52, 2
  %59 = add nuw nsw i64 %51, 1
  %60 = icmp eq i64 %59, %10
  br i1 %60, label %61, label %50, !llvm.loop !186

61:                                               ; preds = %50, %4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE4$_11E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_11", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPmS0_jEZ4mainE4$_10E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #15 {
  %5 = load ptr, ptr %1, align 8, !tbaa !62
  %6 = load ptr, ptr %2, align 8, !tbaa !62
  %7 = load i32, ptr %3, align 4, !tbaa !49
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %22, label %9

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 0, %9 ], [ %20, %11 ]
  %13 = phi i32 [ 0, %9 ], [ %19, %11 ]
  %14 = zext i32 %13 to i64
  %15 = getelementptr inbounds nuw i64, ptr %6, i64 %14
  %16 = load i64, ptr %15, align 8, !tbaa !6
  %17 = add i64 %16, 10
  %18 = getelementptr inbounds nuw i64, ptr %5, i64 %12
  store i64 %17, ptr %18, align 8, !tbaa !6
  %19 = add i32 %13, 2
  %20 = add nuw nsw i64 %12, 1
  %21 = icmp eq i64 %20, %10
  br i1 %21, label %22, label %11, !llvm.loop !187

22:                                               ; preds = %11, %4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPmS0_jEZ4mainE4$_10E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_10", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPmS0_jEZ4mainE4$_11E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #15 {
  %5 = load ptr, ptr %1, align 8, !tbaa !62
  %6 = load ptr, ptr %2, align 8, !tbaa !62
  %7 = load i32, ptr %3, align 4, !tbaa !49
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %61, label %9

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  %11 = icmp ult i32 %7, 5
  br i1 %11, label %47, label %12

12:                                               ; preds = %9
  %13 = shl nuw nsw i64 %10, 3
  %14 = getelementptr i8, ptr %5, i64 %13
  %15 = shl nuw nsw i64 %10, 4
  %16 = getelementptr i8, ptr %6, i64 %15
  %17 = getelementptr i8, ptr %16, i64 -8
  %18 = icmp ult ptr %5, %17
  %19 = icmp ult ptr %6, %14
  %20 = and i1 %18, %19
  br i1 %20, label %47, label %21

21:                                               ; preds = %12
  %22 = and i64 %10, 3
  %23 = icmp eq i64 %22, 0
  %24 = select i1 %23, i64 4, i64 %22
  %25 = sub nsw i64 %10, %24
  %26 = trunc i64 %25 to i32
  %27 = shl i32 %26, 1
  br label %28

28:                                               ; preds = %28, %21
  %29 = phi i64 [ 0, %21 ], [ %45, %28 ]
  %30 = trunc i64 %29 to i32
  %31 = shl i32 %30, 1
  %32 = or disjoint i32 %31, 4
  %33 = zext i32 %31 to i64
  %34 = zext i32 %32 to i64
  %35 = getelementptr inbounds nuw i64, ptr %6, i64 %33
  %36 = getelementptr inbounds nuw i64, ptr %6, i64 %34
  %37 = load <3 x i64>, ptr %35, align 8, !tbaa !6, !alias.scope !188
  %38 = shufflevector <3 x i64> %37, <3 x i64> poison, <2 x i32> <i32 0, i32 2>
  %39 = load <3 x i64>, ptr %36, align 8, !tbaa !6, !alias.scope !188
  %40 = shufflevector <3 x i64> %39, <3 x i64> poison, <2 x i32> <i32 0, i32 2>
  %41 = add <2 x i64> %38, splat (i64 10)
  %42 = add <2 x i64> %40, splat (i64 10)
  %43 = getelementptr inbounds nuw i64, ptr %5, i64 %29
  %44 = getelementptr inbounds nuw i8, ptr %43, i64 16
  store <2 x i64> %41, ptr %43, align 8, !tbaa !6, !alias.scope !191, !noalias !188
  store <2 x i64> %42, ptr %44, align 8, !tbaa !6, !alias.scope !191, !noalias !188
  %45 = add nuw i64 %29, 4
  %46 = icmp eq i64 %45, %25
  br i1 %46, label %47, label %28, !llvm.loop !193

47:                                               ; preds = %28, %12, %9
  %48 = phi i64 [ 0, %12 ], [ 0, %9 ], [ %25, %28 ]
  %49 = phi i32 [ 0, %12 ], [ 0, %9 ], [ %27, %28 ]
  br label %50

50:                                               ; preds = %47, %50
  %51 = phi i64 [ %59, %50 ], [ %48, %47 ]
  %52 = phi i32 [ %58, %50 ], [ %49, %47 ]
  %53 = zext i32 %52 to i64
  %54 = getelementptr inbounds nuw i64, ptr %6, i64 %53
  %55 = load i64, ptr %54, align 8, !tbaa !6
  %56 = add i64 %55, 10
  %57 = getelementptr inbounds nuw i64, ptr %5, i64 %51
  store i64 %56, ptr %57, align 8, !tbaa !6
  %58 = add i32 %52, 2
  %59 = add nuw nsw i64 %51, 1
  %60 = icmp eq i64 %59, %10
  br i1 %60, label %61, label %50, !llvm.loop !194

61:                                               ; preds = %50, %4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPmS0_jEZ4mainE4$_11E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_11", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPhS0_jEZ4mainE4$_12E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #15 {
  %5 = load ptr, ptr %1, align 8, !tbaa !54
  %6 = load ptr, ptr %2, align 8, !tbaa !54
  %7 = load i32, ptr %3, align 4, !tbaa !49
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %19, label %9

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 0, %9 ], [ %17, %11 ]
  %13 = getelementptr inbounds nuw i8, ptr %6, i64 %12
  %14 = load i8, ptr %13, align 1, !tbaa !15
  %15 = add i8 %14, 10
  %16 = getelementptr inbounds nuw i8, ptr %5, i64 %12
  store i8 %15, ptr %16, align 1, !tbaa !15
  %17 = add nuw nsw i64 %12, 2
  %18 = icmp samesign ult i64 %17, %10
  br i1 %18, label %11, label %19, !llvm.loop !195

19:                                               ; preds = %11, %4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPhS0_jEZ4mainE4$_12E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_12", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPhS0_jEZ4mainE4$_13E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #15 {
  %5 = load ptr, ptr %1, align 8, !tbaa !54
  %6 = load ptr, ptr %2, align 8, !tbaa !54
  %7 = load i32, ptr %3, align 4, !tbaa !49
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %138, label %9

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  %11 = add nsw i64 %10, -1
  %12 = lshr i64 %11, 1
  %13 = add nuw i64 %12, 1
  %14 = icmp ult i32 %7, 17
  br i1 %14, label %128, label %15

15:                                               ; preds = %9
  %16 = add nsw i64 %10, -1
  %17 = or i64 %16, 1
  %18 = getelementptr i8, ptr %5, i64 %17
  %19 = getelementptr i8, ptr %6, i64 %17
  %20 = icmp ult ptr %5, %19
  %21 = icmp ult ptr %6, %18
  %22 = and i1 %20, %21
  br i1 %22, label %128, label %23

23:                                               ; preds = %15
  %24 = icmp ult i32 %7, 33
  br i1 %24, label %89, label %25

25:                                               ; preds = %23
  %26 = and i64 %13, 15
  %27 = icmp eq i64 %26, 0
  %28 = select i1 %27, i64 16, i64 %26
  %29 = sub i64 %13, %28
  br label %30

30:                                               ; preds = %30, %25
  %31 = phi i64 [ 0, %25 ], [ %84, %30 ]
  %32 = shl i64 %31, 1
  %33 = getelementptr inbounds nuw i8, ptr %6, i64 %32
  %34 = load <31 x i8>, ptr %33, align 1, !tbaa !15, !alias.scope !197
  %35 = shufflevector <31 x i8> %34, <31 x i8> poison, <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30>
  %36 = add <16 x i8> %35, splat (i8 10)
  %37 = getelementptr inbounds nuw i8, ptr %5, i64 %32
  %38 = getelementptr inbounds nuw i8, ptr %5, i64 %32
  %39 = getelementptr inbounds nuw i8, ptr %38, i64 2
  %40 = getelementptr inbounds nuw i8, ptr %5, i64 %32
  %41 = getelementptr inbounds nuw i8, ptr %40, i64 4
  %42 = getelementptr inbounds nuw i8, ptr %5, i64 %32
  %43 = getelementptr inbounds nuw i8, ptr %42, i64 6
  %44 = getelementptr inbounds nuw i8, ptr %5, i64 %32
  %45 = getelementptr inbounds nuw i8, ptr %44, i64 8
  %46 = getelementptr inbounds nuw i8, ptr %5, i64 %32
  %47 = getelementptr inbounds nuw i8, ptr %46, i64 10
  %48 = getelementptr inbounds nuw i8, ptr %5, i64 %32
  %49 = getelementptr inbounds nuw i8, ptr %48, i64 12
  %50 = getelementptr inbounds nuw i8, ptr %5, i64 %32
  %51 = getelementptr inbounds nuw i8, ptr %50, i64 14
  %52 = getelementptr inbounds nuw i8, ptr %5, i64 %32
  %53 = getelementptr inbounds nuw i8, ptr %52, i64 16
  %54 = getelementptr inbounds nuw i8, ptr %5, i64 %32
  %55 = getelementptr inbounds nuw i8, ptr %54, i64 18
  %56 = getelementptr inbounds nuw i8, ptr %5, i64 %32
  %57 = getelementptr inbounds nuw i8, ptr %56, i64 20
  %58 = getelementptr inbounds nuw i8, ptr %5, i64 %32
  %59 = getelementptr inbounds nuw i8, ptr %58, i64 22
  %60 = getelementptr inbounds nuw i8, ptr %5, i64 %32
  %61 = getelementptr inbounds nuw i8, ptr %60, i64 24
  %62 = getelementptr inbounds nuw i8, ptr %5, i64 %32
  %63 = getelementptr inbounds nuw i8, ptr %62, i64 26
  %64 = getelementptr inbounds nuw i8, ptr %5, i64 %32
  %65 = getelementptr inbounds nuw i8, ptr %64, i64 28
  %66 = getelementptr inbounds nuw i8, ptr %5, i64 %32
  %67 = getelementptr inbounds nuw i8, ptr %66, i64 30
  %68 = extractelement <16 x i8> %36, i64 0
  store i8 %68, ptr %37, align 1, !tbaa !15, !alias.scope !200, !noalias !197
  %69 = extractelement <16 x i8> %36, i64 1
  store i8 %69, ptr %39, align 1, !tbaa !15, !alias.scope !200, !noalias !197
  %70 = extractelement <16 x i8> %36, i64 2
  store i8 %70, ptr %41, align 1, !tbaa !15, !alias.scope !200, !noalias !197
  %71 = extractelement <16 x i8> %36, i64 3
  store i8 %71, ptr %43, align 1, !tbaa !15, !alias.scope !200, !noalias !197
  %72 = extractelement <16 x i8> %36, i64 4
  store i8 %72, ptr %45, align 1, !tbaa !15, !alias.scope !200, !noalias !197
  %73 = extractelement <16 x i8> %36, i64 5
  store i8 %73, ptr %47, align 1, !tbaa !15, !alias.scope !200, !noalias !197
  %74 = extractelement <16 x i8> %36, i64 6
  store i8 %74, ptr %49, align 1, !tbaa !15, !alias.scope !200, !noalias !197
  %75 = extractelement <16 x i8> %36, i64 7
  store i8 %75, ptr %51, align 1, !tbaa !15, !alias.scope !200, !noalias !197
  %76 = extractelement <16 x i8> %36, i64 8
  store i8 %76, ptr %53, align 1, !tbaa !15, !alias.scope !200, !noalias !197
  %77 = extractelement <16 x i8> %36, i64 9
  store i8 %77, ptr %55, align 1, !tbaa !15, !alias.scope !200, !noalias !197
  %78 = extractelement <16 x i8> %36, i64 10
  store i8 %78, ptr %57, align 1, !tbaa !15, !alias.scope !200, !noalias !197
  %79 = extractelement <16 x i8> %36, i64 11
  store i8 %79, ptr %59, align 1, !tbaa !15, !alias.scope !200, !noalias !197
  %80 = extractelement <16 x i8> %36, i64 12
  store i8 %80, ptr %61, align 1, !tbaa !15, !alias.scope !200, !noalias !197
  %81 = extractelement <16 x i8> %36, i64 13
  store i8 %81, ptr %63, align 1, !tbaa !15, !alias.scope !200, !noalias !197
  %82 = extractelement <16 x i8> %36, i64 14
  store i8 %82, ptr %65, align 1, !tbaa !15, !alias.scope !200, !noalias !197
  %83 = extractelement <16 x i8> %36, i64 15
  store i8 %83, ptr %67, align 1, !tbaa !15, !alias.scope !200, !noalias !197
  %84 = add nuw i64 %31, 16
  %85 = icmp eq i64 %84, %29
  br i1 %85, label %86, label %30, !llvm.loop !202

86:                                               ; preds = %30
  %87 = shl i64 %29, 1
  %88 = icmp samesign ult i64 %28, 9
  br i1 %88, label %128, label %89

89:                                               ; preds = %86, %23
  %90 = phi i64 [ %29, %86 ], [ 0, %23 ]
  %91 = and i64 %13, 7
  %92 = icmp eq i64 %91, 0
  %93 = select i1 %92, i64 8, i64 %91
  %94 = sub i64 %13, %93
  %95 = shl i64 %94, 1
  br label %96

96:                                               ; preds = %96, %89
  %97 = phi i64 [ %90, %89 ], [ %126, %96 ]
  %98 = shl i64 %97, 1
  %99 = getelementptr inbounds nuw i8, ptr %6, i64 %98
  %100 = load <16 x i8>, ptr %99, align 1, !tbaa !15, !alias.scope !197
  %101 = shufflevector <16 x i8> %100, <16 x i8> poison, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
  %102 = add <8 x i8> %101, splat (i8 10)
  %103 = getelementptr inbounds nuw i8, ptr %5, i64 %98
  %104 = getelementptr i8, ptr %5, i64 %98
  %105 = getelementptr i8, ptr %104, i64 2
  %106 = getelementptr i8, ptr %5, i64 %98
  %107 = getelementptr i8, ptr %106, i64 4
  %108 = getelementptr i8, ptr %5, i64 %98
  %109 = getelementptr i8, ptr %108, i64 6
  %110 = getelementptr i8, ptr %5, i64 %98
  %111 = getelementptr i8, ptr %110, i64 8
  %112 = getelementptr i8, ptr %5, i64 %98
  %113 = getelementptr i8, ptr %112, i64 10
  %114 = getelementptr i8, ptr %5, i64 %98
  %115 = getelementptr i8, ptr %114, i64 12
  %116 = getelementptr i8, ptr %5, i64 %98
  %117 = getelementptr i8, ptr %116, i64 14
  %118 = extractelement <8 x i8> %102, i64 0
  store i8 %118, ptr %103, align 1, !tbaa !15, !alias.scope !200, !noalias !197
  %119 = extractelement <8 x i8> %102, i64 1
  store i8 %119, ptr %105, align 1, !tbaa !15, !alias.scope !200, !noalias !197
  %120 = extractelement <8 x i8> %102, i64 2
  store i8 %120, ptr %107, align 1, !tbaa !15, !alias.scope !200, !noalias !197
  %121 = extractelement <8 x i8> %102, i64 3
  store i8 %121, ptr %109, align 1, !tbaa !15, !alias.scope !200, !noalias !197
  %122 = extractelement <8 x i8> %102, i64 4
  store i8 %122, ptr %111, align 1, !tbaa !15, !alias.scope !200, !noalias !197
  %123 = extractelement <8 x i8> %102, i64 5
  store i8 %123, ptr %113, align 1, !tbaa !15, !alias.scope !200, !noalias !197
  %124 = extractelement <8 x i8> %102, i64 6
  store i8 %124, ptr %115, align 1, !tbaa !15, !alias.scope !200, !noalias !197
  %125 = extractelement <8 x i8> %102, i64 7
  store i8 %125, ptr %117, align 1, !tbaa !15, !alias.scope !200, !noalias !197
  %126 = add nuw i64 %97, 8
  %127 = icmp eq i64 %126, %94
  br i1 %127, label %128, label %96, !llvm.loop !203

128:                                              ; preds = %96, %86, %15, %9
  %129 = phi i64 [ 0, %9 ], [ 0, %15 ], [ %87, %86 ], [ %95, %96 ]
  br label %130

130:                                              ; preds = %128, %130
  %131 = phi i64 [ %136, %130 ], [ %129, %128 ]
  %132 = getelementptr inbounds nuw i8, ptr %6, i64 %131
  %133 = load i8, ptr %132, align 1, !tbaa !15
  %134 = add i8 %133, 10
  %135 = getelementptr inbounds nuw i8, ptr %5, i64 %131
  store i8 %134, ptr %135, align 1, !tbaa !15
  %136 = add nuw nsw i64 %131, 2
  %137 = icmp samesign ult i64 %136, %10
  br i1 %137, label %130, label %138, !llvm.loop !204

138:                                              ; preds = %130, %4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPhS0_jEZ4mainE4$_13E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_13", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE4$_12E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #15 {
  %5 = load ptr, ptr %1, align 8, !tbaa !58
  %6 = load ptr, ptr %2, align 8, !tbaa !58
  %7 = load i32, ptr %3, align 4, !tbaa !49
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %19, label %9

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 0, %9 ], [ %17, %11 ]
  %13 = getelementptr inbounds nuw i32, ptr %6, i64 %12
  %14 = load i32, ptr %13, align 4, !tbaa !49
  %15 = add i32 %14, 10
  %16 = getelementptr inbounds nuw i32, ptr %5, i64 %12
  store i32 %15, ptr %16, align 4, !tbaa !49
  %17 = add nuw nsw i64 %12, 2
  %18 = icmp samesign ult i64 %17, %10
  br i1 %18, label %11, label %19, !llvm.loop !205

19:                                               ; preds = %11, %4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE4$_12E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_12", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE4$_13E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #15 {
  %5 = load ptr, ptr %1, align 8, !tbaa !58
  %6 = load ptr, ptr %2, align 8, !tbaa !58
  %7 = load i32, ptr %3, align 4, !tbaa !49
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %60, label %9

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  %11 = add nsw i64 %10, -1
  %12 = lshr i64 %11, 1
  %13 = add nuw i64 %12, 1
  %14 = icmp ult i32 %7, 9
  br i1 %14, label %50, label %15

15:                                               ; preds = %9
  %16 = shl nuw nsw i64 %10, 2
  %17 = add nsw i64 %16, -4
  %18 = or i64 %17, 4
  %19 = getelementptr i8, ptr %5, i64 %18
  %20 = getelementptr i8, ptr %6, i64 %18
  %21 = icmp ult ptr %5, %20
  %22 = icmp ult ptr %6, %19
  %23 = and i1 %21, %22
  br i1 %23, label %50, label %24

24:                                               ; preds = %15
  %25 = and i64 %13, 3
  %26 = icmp eq i64 %25, 0
  %27 = select i1 %26, i64 4, i64 %25
  %28 = sub i64 %13, %27
  %29 = shl i64 %28, 1
  br label %30

30:                                               ; preds = %30, %24
  %31 = phi i64 [ 0, %24 ], [ %48, %30 ]
  %32 = shl i64 %31, 1
  %33 = getelementptr inbounds nuw i32, ptr %6, i64 %32
  %34 = load <7 x i32>, ptr %33, align 4, !tbaa !49, !alias.scope !206
  %35 = shufflevector <7 x i32> %34, <7 x i32> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %36 = add <4 x i32> %35, splat (i32 10)
  %37 = getelementptr inbounds nuw i32, ptr %5, i64 %32
  %38 = getelementptr inbounds nuw i32, ptr %5, i64 %32
  %39 = getelementptr inbounds nuw i8, ptr %38, i64 8
  %40 = getelementptr inbounds nuw i32, ptr %5, i64 %32
  %41 = getelementptr inbounds nuw i8, ptr %40, i64 16
  %42 = getelementptr inbounds nuw i32, ptr %5, i64 %32
  %43 = getelementptr inbounds nuw i8, ptr %42, i64 24
  %44 = extractelement <4 x i32> %36, i64 0
  store i32 %44, ptr %37, align 4, !tbaa !49, !alias.scope !209, !noalias !206
  %45 = extractelement <4 x i32> %36, i64 1
  store i32 %45, ptr %39, align 4, !tbaa !49, !alias.scope !209, !noalias !206
  %46 = extractelement <4 x i32> %36, i64 2
  store i32 %46, ptr %41, align 4, !tbaa !49, !alias.scope !209, !noalias !206
  %47 = extractelement <4 x i32> %36, i64 3
  store i32 %47, ptr %43, align 4, !tbaa !49, !alias.scope !209, !noalias !206
  %48 = add nuw i64 %31, 4
  %49 = icmp eq i64 %48, %28
  br i1 %49, label %50, label %30, !llvm.loop !211

50:                                               ; preds = %30, %15, %9
  %51 = phi i64 [ 0, %15 ], [ 0, %9 ], [ %29, %30 ]
  br label %52

52:                                               ; preds = %50, %52
  %53 = phi i64 [ %58, %52 ], [ %51, %50 ]
  %54 = getelementptr inbounds nuw i32, ptr %6, i64 %53
  %55 = load i32, ptr %54, align 4, !tbaa !49
  %56 = add i32 %55, 10
  %57 = getelementptr inbounds nuw i32, ptr %5, i64 %53
  store i32 %56, ptr %57, align 4, !tbaa !49
  %58 = add nuw nsw i64 %53, 2
  %59 = icmp samesign ult i64 %58, %10
  br i1 %59, label %52, label %60, !llvm.loop !212

60:                                               ; preds = %52, %4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE4$_13E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_13", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPmS0_jEZ4mainE4$_12E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #15 {
  %5 = load ptr, ptr %1, align 8, !tbaa !62
  %6 = load ptr, ptr %2, align 8, !tbaa !62
  %7 = load i32, ptr %3, align 4, !tbaa !49
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %19, label %9

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 0, %9 ], [ %17, %11 ]
  %13 = getelementptr inbounds nuw i64, ptr %6, i64 %12
  %14 = load i64, ptr %13, align 8, !tbaa !6
  %15 = add i64 %14, 10
  %16 = getelementptr inbounds nuw i64, ptr %5, i64 %12
  store i64 %15, ptr %16, align 8, !tbaa !6
  %17 = add nuw nsw i64 %12, 2
  %18 = icmp samesign ult i64 %17, %10
  br i1 %18, label %11, label %19, !llvm.loop !213

19:                                               ; preds = %11, %4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPmS0_jEZ4mainE4$_12E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_12", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPmS0_jEZ4mainE4$_13E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #15 {
  %5 = load ptr, ptr %1, align 8, !tbaa !62
  %6 = load ptr, ptr %2, align 8, !tbaa !62
  %7 = load i32, ptr %3, align 4, !tbaa !49
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %54, label %9

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  %11 = add nsw i64 %10, -1
  %12 = lshr i64 %11, 1
  %13 = add nuw i64 %12, 1
  %14 = icmp ult i32 %7, 5
  br i1 %14, label %44, label %15

15:                                               ; preds = %9
  %16 = shl nuw nsw i64 %10, 3
  %17 = add nsw i64 %16, -8
  %18 = or i64 %17, 8
  %19 = getelementptr i8, ptr %5, i64 %18
  %20 = getelementptr i8, ptr %6, i64 %18
  %21 = icmp ult ptr %5, %20
  %22 = icmp ult ptr %6, %19
  %23 = and i1 %21, %22
  br i1 %23, label %44, label %24

24:                                               ; preds = %15
  %25 = and i64 %11, 2
  %26 = icmp eq i64 %25, 0
  %27 = select i1 %26, i64 -1, i64 -2
  %28 = add i64 %27, %13
  %29 = shl i64 %28, 1
  br label %30

30:                                               ; preds = %30, %24
  %31 = phi i64 [ 0, %24 ], [ %42, %30 ]
  %32 = shl i64 %31, 1
  %33 = getelementptr inbounds nuw i64, ptr %6, i64 %32
  %34 = load <3 x i64>, ptr %33, align 8, !tbaa !6, !alias.scope !214
  %35 = shufflevector <3 x i64> %34, <3 x i64> poison, <2 x i32> <i32 0, i32 2>
  %36 = add <2 x i64> %35, splat (i64 10)
  %37 = getelementptr inbounds nuw i64, ptr %5, i64 %32
  %38 = getelementptr inbounds nuw i64, ptr %5, i64 %32
  %39 = getelementptr inbounds nuw i8, ptr %38, i64 16
  %40 = extractelement <2 x i64> %36, i64 0
  store i64 %40, ptr %37, align 8, !tbaa !6, !alias.scope !217, !noalias !214
  %41 = extractelement <2 x i64> %36, i64 1
  store i64 %41, ptr %39, align 8, !tbaa !6, !alias.scope !217, !noalias !214
  %42 = add nuw i64 %31, 2
  %43 = icmp eq i64 %42, %28
  br i1 %43, label %44, label %30, !llvm.loop !219

44:                                               ; preds = %30, %15, %9
  %45 = phi i64 [ 0, %15 ], [ 0, %9 ], [ %29, %30 ]
  br label %46

46:                                               ; preds = %44, %46
  %47 = phi i64 [ %52, %46 ], [ %45, %44 ]
  %48 = getelementptr inbounds nuw i64, ptr %6, i64 %47
  %49 = load i64, ptr %48, align 8, !tbaa !6
  %50 = add i64 %49, 10
  %51 = getelementptr inbounds nuw i64, ptr %5, i64 %47
  store i64 %50, ptr %51, align 8, !tbaa !6
  %52 = add nuw nsw i64 %47, 2
  %53 = icmp samesign ult i64 %52, %10
  br i1 %53, label %46, label %54, !llvm.loop !220

54:                                               ; preds = %46, %4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPmS0_jEZ4mainE4$_13E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_13", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPhS0_jEZ4mainE4$_14E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #15 {
  %5 = load ptr, ptr %1, align 8, !tbaa !54
  %6 = load i32, ptr %3, align 4, !tbaa !49
  %7 = icmp eq i32 %6, 0
  br i1 %7, label %25, label %8

8:                                                ; preds = %4
  %9 = load ptr, ptr %2, align 8, !tbaa !54
  %10 = lshr i32 %6, 1
  %11 = zext nneg i32 %10 to i64
  %12 = getelementptr inbounds nuw i8, ptr %9, i64 %11
  %13 = zext i32 %6 to i64
  br label %14

14:                                               ; preds = %14, %8
  %15 = phi i64 [ 0, %8 ], [ %23, %14 ]
  %16 = phi i32 [ 0, %8 ], [ %21, %14 ]
  %17 = getelementptr inbounds nuw i8, ptr %5, i64 %15
  %18 = load i8, ptr %17, align 1, !tbaa !15
  %19 = zext i8 %18 to i32
  %20 = add i32 %16, 10
  %21 = add i32 %20, %19
  %22 = trunc i32 %21 to i8
  store i8 %22, ptr %12, align 1, !tbaa !15
  %23 = add nuw nsw i64 %15, 1
  %24 = icmp eq i64 %23, %13
  br i1 %24, label %25, label %14, !llvm.loop !221

25:                                               ; preds = %14, %4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPhS0_jEZ4mainE4$_14E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_14", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPhS0_jEZ4mainE4$_15E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #15 {
  %5 = load ptr, ptr %1, align 8, !tbaa !54
  %6 = load i32, ptr %3, align 4, !tbaa !49
  %7 = icmp eq i32 %6, 0
  br i1 %7, label %25, label %8

8:                                                ; preds = %4
  %9 = load ptr, ptr %2, align 8, !tbaa !54
  %10 = lshr i32 %6, 1
  %11 = zext nneg i32 %10 to i64
  %12 = getelementptr inbounds nuw i8, ptr %9, i64 %11
  %13 = zext i32 %6 to i64
  br label %14

14:                                               ; preds = %14, %8
  %15 = phi i64 [ 0, %8 ], [ %23, %14 ]
  %16 = phi i32 [ 0, %8 ], [ %21, %14 ]
  %17 = getelementptr inbounds nuw i8, ptr %5, i64 %15
  %18 = load i8, ptr %17, align 1, !tbaa !15
  %19 = zext i8 %18 to i32
  %20 = add i32 %16, 10
  %21 = add i32 %20, %19
  %22 = trunc i32 %21 to i8
  store i8 %22, ptr %12, align 1, !tbaa !15
  %23 = add nuw nsw i64 %15, 1
  %24 = icmp eq i64 %23, %13
  br i1 %24, label %25, label %14, !llvm.loop !222

25:                                               ; preds = %14, %4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPhS0_jEZ4mainE4$_15E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_15", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE4$_14E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #15 {
  %5 = load ptr, ptr %1, align 8, !tbaa !58
  %6 = load i32, ptr %3, align 4, !tbaa !49
  %7 = icmp eq i32 %6, 0
  br i1 %7, label %23, label %8

8:                                                ; preds = %4
  %9 = load ptr, ptr %2, align 8, !tbaa !58
  %10 = lshr i32 %6, 1
  %11 = zext nneg i32 %10 to i64
  %12 = getelementptr inbounds nuw i32, ptr %9, i64 %11
  %13 = zext i32 %6 to i64
  br label %14

14:                                               ; preds = %14, %8
  %15 = phi i64 [ 0, %8 ], [ %21, %14 ]
  %16 = phi i32 [ 0, %8 ], [ %20, %14 ]
  %17 = getelementptr inbounds nuw i32, ptr %5, i64 %15
  %18 = load i32, ptr %17, align 4, !tbaa !49
  %19 = add i32 %16, 10
  %20 = add i32 %19, %18
  store i32 %20, ptr %12, align 4, !tbaa !49
  %21 = add nuw nsw i64 %15, 1
  %22 = icmp eq i64 %21, %13
  br i1 %22, label %23, label %14, !llvm.loop !224

23:                                               ; preds = %14, %4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE4$_14E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_14", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE4$_15E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #15 {
  %5 = load ptr, ptr %1, align 8, !tbaa !58
  %6 = load i32, ptr %3, align 4, !tbaa !49
  %7 = icmp eq i32 %6, 0
  br i1 %7, label %56, label %8

8:                                                ; preds = %4
  %9 = load ptr, ptr %2, align 8, !tbaa !58
  %10 = lshr i32 %6, 1
  %11 = zext nneg i32 %10 to i64
  %12 = getelementptr inbounds nuw i32, ptr %9, i64 %11
  %13 = zext i32 %6 to i64
  %14 = icmp ult i32 %6, 8
  br i1 %14, label %44, label %15

15:                                               ; preds = %8
  %16 = shl nuw nsw i64 %11, 2
  %17 = getelementptr i8, ptr %9, i64 %16
  %18 = getelementptr i8, ptr %17, i64 4
  %19 = shl nuw nsw i64 %13, 2
  %20 = getelementptr i8, ptr %5, i64 %19
  %21 = icmp ult ptr %12, %20
  %22 = icmp ult ptr %5, %18
  %23 = and i1 %21, %22
  br i1 %23, label %44, label %24

24:                                               ; preds = %15
  %25 = and i64 %13, 4294967288
  br label %26

26:                                               ; preds = %26, %24
  %27 = phi i64 [ 0, %24 ], [ %38, %26 ]
  %28 = phi <4 x i32> [ zeroinitializer, %24 ], [ %36, %26 ]
  %29 = phi <4 x i32> [ zeroinitializer, %24 ], [ %37, %26 ]
  %30 = getelementptr inbounds nuw i32, ptr %5, i64 %27
  %31 = getelementptr inbounds nuw i8, ptr %30, i64 16
  %32 = load <4 x i32>, ptr %30, align 4, !tbaa !49, !alias.scope !225
  %33 = load <4 x i32>, ptr %31, align 4, !tbaa !49, !alias.scope !225
  %34 = add <4 x i32> %28, splat (i32 10)
  %35 = add <4 x i32> %29, splat (i32 10)
  %36 = add <4 x i32> %34, %32
  %37 = add <4 x i32> %35, %33
  %38 = add nuw i64 %27, 8
  %39 = icmp eq i64 %38, %25
  br i1 %39, label %40, label %26, !llvm.loop !228

40:                                               ; preds = %26
  %41 = add <4 x i32> %37, %36
  %42 = tail call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> %41)
  store i32 %42, ptr %12, align 4, !tbaa !49, !alias.scope !229, !noalias !225
  %43 = icmp eq i64 %25, %13
  br i1 %43, label %56, label %44

44:                                               ; preds = %15, %8, %40
  %45 = phi i64 [ 0, %15 ], [ 0, %8 ], [ %25, %40 ]
  %46 = phi i32 [ 0, %15 ], [ 0, %8 ], [ %42, %40 ]
  br label %47

47:                                               ; preds = %44, %47
  %48 = phi i64 [ %54, %47 ], [ %45, %44 ]
  %49 = phi i32 [ %53, %47 ], [ %46, %44 ]
  %50 = getelementptr inbounds nuw i32, ptr %5, i64 %48
  %51 = load i32, ptr %50, align 4, !tbaa !49
  %52 = add i32 %49, 10
  %53 = add i32 %52, %51
  store i32 %53, ptr %12, align 4, !tbaa !49
  %54 = add nuw nsw i64 %48, 1
  %55 = icmp eq i64 %54, %13
  br i1 %55, label %56, label %47, !llvm.loop !231

56:                                               ; preds = %47, %40, %4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE4$_15E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_15", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPmS0_jEZ4mainE4$_14E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #15 {
  %5 = load ptr, ptr %1, align 8, !tbaa !62
  %6 = load i32, ptr %3, align 4, !tbaa !49
  %7 = icmp eq i32 %6, 0
  br i1 %7, label %24, label %8

8:                                                ; preds = %4
  %9 = load ptr, ptr %2, align 8, !tbaa !62
  %10 = lshr i32 %6, 1
  %11 = zext nneg i32 %10 to i64
  %12 = getelementptr inbounds nuw i64, ptr %9, i64 %11
  %13 = zext i32 %6 to i64
  br label %14

14:                                               ; preds = %14, %8
  %15 = phi i64 [ 0, %8 ], [ %22, %14 ]
  %16 = phi i64 [ 0, %8 ], [ %20, %14 ]
  %17 = getelementptr inbounds nuw i64, ptr %5, i64 %15
  %18 = load i64, ptr %17, align 8, !tbaa !6
  %19 = add i64 %16, 10
  %20 = add i64 %19, %18
  %21 = and i64 %20, 4294967295
  store i64 %21, ptr %12, align 8, !tbaa !6
  %22 = add nuw nsw i64 %15, 1
  %23 = icmp eq i64 %22, %13
  br i1 %23, label %24, label %14, !llvm.loop !232

24:                                               ; preds = %14, %4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPmS0_jEZ4mainE4$_14E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_14", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPmS0_jEZ4mainE4$_15E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #15 {
  %5 = load ptr, ptr %1, align 8, !tbaa !62
  %6 = load i32, ptr %3, align 4, !tbaa !49
  %7 = icmp eq i32 %6, 0
  br i1 %7, label %24, label %8

8:                                                ; preds = %4
  %9 = load ptr, ptr %2, align 8, !tbaa !62
  %10 = lshr i32 %6, 1
  %11 = zext nneg i32 %10 to i64
  %12 = getelementptr inbounds nuw i64, ptr %9, i64 %11
  %13 = zext i32 %6 to i64
  br label %14

14:                                               ; preds = %14, %8
  %15 = phi i64 [ 0, %8 ], [ %22, %14 ]
  %16 = phi i64 [ 0, %8 ], [ %20, %14 ]
  %17 = getelementptr inbounds nuw i64, ptr %5, i64 %15
  %18 = load i64, ptr %17, align 8, !tbaa !6
  %19 = add i64 %16, 10
  %20 = add i64 %19, %18
  %21 = and i64 %20, 4294967295
  store i64 %21, ptr %12, align 8, !tbaa !6
  %22 = add nuw nsw i64 %15, 1
  %23 = icmp eq i64 %22, %13
  br i1 %23, label %24, label %14, !llvm.loop !233

24:                                               ; preds = %14, %4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPmS0_jEZ4mainE4$_15E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_15", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZNKSt8functionIFvPjS0_S0_jEEclES0_S0_S0_j(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef %1, ptr noundef %2, ptr noundef %3, i32 noundef %4) local_unnamed_addr #9 comdat {
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  %9 = alloca i32, align 4
  store ptr %1, ptr %6, align 8, !tbaa !58
  store ptr %2, ptr %7, align 8, !tbaa !58
  store ptr %3, ptr %8, align 8, !tbaa !58
  store i32 %4, ptr %9, align 4, !tbaa !49
  %10 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %11 = load ptr, ptr %10, align 8, !tbaa !20
  %12 = icmp eq ptr %11, null
  br i1 %12, label %13, label %14

13:                                               ; preds = %5
  tail call void @_ZSt25__throw_bad_function_callv() #24
  unreachable

14:                                               ; preds = %5
  %15 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %16 = load ptr, ptr %15, align 8, !tbaa !25
  call void %16(ptr noundef nonnull align 8 dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(8) %6, ptr noundef nonnull align 8 dereferenceable(8) %7, ptr noundef nonnull align 8 dereferenceable(8) %8, ptr noundef nonnull align 4 dereferenceable(4) %9)
  ret void
}

; Function Attrs: mustprogress noinline norecurse optnone uwtable
define internal fastcc void @_ZL18callThroughOptnoneIRSt8functionIFvPjS1_S1_jEEJS1_S1_RS1_RjEEvOT_DpOT0_(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %1, ptr noundef nonnull align 8 dereferenceable(8) %2, ptr noundef nonnull align 8 dereferenceable(8) %3, ptr noundef nonnull align 4 dereferenceable(4) %4) unnamed_addr #10 {
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  %10 = alloca ptr, align 8
  store ptr %0, ptr %6, align 8, !tbaa !234
  store ptr %1, ptr %7, align 8, !tbaa !94
  store ptr %2, ptr %8, align 8, !tbaa !94
  store ptr %3, ptr %9, align 8, !tbaa !94
  store ptr %4, ptr %10, align 8, !tbaa !58
  %11 = load ptr, ptr %6, align 8, !tbaa !234, !nonnull !76, !align !77
  %12 = load ptr, ptr %7, align 8, !tbaa !94, !nonnull !76, !align !77
  %13 = load ptr, ptr %12, align 8, !tbaa !58
  %14 = load ptr, ptr %8, align 8, !tbaa !94, !nonnull !76, !align !77
  %15 = load ptr, ptr %14, align 8, !tbaa !58
  %16 = load ptr, ptr %9, align 8, !tbaa !94, !nonnull !76, !align !77
  %17 = load ptr, ptr %16, align 8, !tbaa !58
  %18 = load ptr, ptr %10, align 8, !tbaa !58, !nonnull !76, !align !78
  %19 = load i32, ptr %18, align 4, !tbaa !49
  call void @_ZNKSt8functionIFvPjS0_S0_jEEclES0_S0_S0_j(ptr noundef nonnull align 8 dereferenceable(32) %11, ptr noundef %13, ptr noundef %15, ptr noundef %17, i32 noundef %19)
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPjS0_S0_jEZ4mainE4$_16E9_M_invokeERKSt9_Any_dataOS0_S7_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %3, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %4) #15 {
  %6 = load ptr, ptr %1, align 8, !tbaa !58
  %7 = load ptr, ptr %2, align 8, !tbaa !58
  %8 = load ptr, ptr %3, align 8, !tbaa !58
  %9 = load i32, ptr %4, align 4, !tbaa !49
  %10 = icmp eq i32 %9, 0
  br i1 %10, label %24, label %11

11:                                               ; preds = %5
  %12 = zext i32 %9 to i64
  br label %13

13:                                               ; preds = %13, %11
  %14 = phi i64 [ 0, %11 ], [ %22, %13 ]
  %15 = getelementptr inbounds nuw i32, ptr %7, i64 %14
  %16 = load i32, ptr %15, align 4, !tbaa !49
  %17 = getelementptr inbounds nuw i32, ptr %8, i64 %14
  %18 = load i32, ptr %17, align 4, !tbaa !49
  %19 = add i32 %16, 10
  %20 = add i32 %19, %18
  %21 = getelementptr inbounds nuw i32, ptr %6, i64 %14
  store i32 %20, ptr %21, align 4, !tbaa !49
  %22 = add nuw nsw i64 %14, 1
  %23 = icmp eq i64 %22, %12
  br i1 %23, label %24, label %13, !llvm.loop !236

24:                                               ; preds = %13, %5
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPjS0_S0_jEZ4mainE4$_16E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_16", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPjS0_S0_jEZ4mainE4$_17E9_M_invokeERKSt9_Any_dataOS0_S7_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %3, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %4) #15 {
  %6 = load ptr, ptr %1, align 8, !tbaa !58
  %7 = ptrtoint ptr %6 to i64
  %8 = load ptr, ptr %2, align 8, !tbaa !58
  %9 = ptrtoint ptr %8 to i64
  %10 = load ptr, ptr %3, align 8, !tbaa !58
  %11 = ptrtoint ptr %10 to i64
  %12 = load i32, ptr %4, align 4, !tbaa !49
  %13 = icmp eq i32 %12, 0
  br i1 %13, label %58, label %14

14:                                               ; preds = %5
  %15 = zext i32 %12 to i64
  %16 = icmp ult i32 %12, 8
  br i1 %16, label %45, label %17

17:                                               ; preds = %14
  %18 = sub i64 %7, %9
  %19 = icmp ult i64 %18, 32
  %20 = sub i64 %7, %11
  %21 = icmp ult i64 %20, 32
  %22 = or i1 %19, %21
  br i1 %22, label %45, label %23

23:                                               ; preds = %17
  %24 = and i64 %15, 4294967288
  br label %25

25:                                               ; preds = %25, %23
  %26 = phi i64 [ 0, %23 ], [ %41, %25 ]
  %27 = getelementptr inbounds nuw i32, ptr %8, i64 %26
  %28 = getelementptr inbounds nuw i8, ptr %27, i64 16
  %29 = load <4 x i32>, ptr %27, align 4, !tbaa !49
  %30 = load <4 x i32>, ptr %28, align 4, !tbaa !49
  %31 = getelementptr inbounds nuw i32, ptr %10, i64 %26
  %32 = getelementptr inbounds nuw i8, ptr %31, i64 16
  %33 = load <4 x i32>, ptr %31, align 4, !tbaa !49
  %34 = load <4 x i32>, ptr %32, align 4, !tbaa !49
  %35 = add <4 x i32> %29, splat (i32 10)
  %36 = add <4 x i32> %30, splat (i32 10)
  %37 = add <4 x i32> %35, %33
  %38 = add <4 x i32> %36, %34
  %39 = getelementptr inbounds nuw i32, ptr %6, i64 %26
  %40 = getelementptr inbounds nuw i8, ptr %39, i64 16
  store <4 x i32> %37, ptr %39, align 4, !tbaa !49
  store <4 x i32> %38, ptr %40, align 4, !tbaa !49
  %41 = add nuw i64 %26, 8
  %42 = icmp eq i64 %41, %24
  br i1 %42, label %43, label %25, !llvm.loop !237

43:                                               ; preds = %25
  %44 = icmp eq i64 %24, %15
  br i1 %44, label %58, label %45

45:                                               ; preds = %17, %14, %43
  %46 = phi i64 [ 0, %17 ], [ 0, %14 ], [ %24, %43 ]
  br label %47

47:                                               ; preds = %45, %47
  %48 = phi i64 [ %56, %47 ], [ %46, %45 ]
  %49 = getelementptr inbounds nuw i32, ptr %8, i64 %48
  %50 = load i32, ptr %49, align 4, !tbaa !49
  %51 = getelementptr inbounds nuw i32, ptr %10, i64 %48
  %52 = load i32, ptr %51, align 4, !tbaa !49
  %53 = add i32 %50, 10
  %54 = add i32 %53, %52
  %55 = getelementptr inbounds nuw i32, ptr %6, i64 %48
  store i32 %54, ptr %55, align 4, !tbaa !49
  %56 = add nuw nsw i64 %48, 1
  %57 = icmp eq i64 %56, %15
  br i1 %57, label %58, label %47, !llvm.loop !238

58:                                               ; preds = %47, %43, %5
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPjS0_S0_jEZ4mainE4$_17E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_17", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZNKSt8functionIFvPhS0_S0_jEEclES0_S0_S0_j(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef %1, ptr noundef %2, ptr noundef %3, i32 noundef %4) local_unnamed_addr #9 comdat {
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  %9 = alloca i32, align 4
  store ptr %1, ptr %6, align 8, !tbaa !54
  store ptr %2, ptr %7, align 8, !tbaa !54
  store ptr %3, ptr %8, align 8, !tbaa !54
  store i32 %4, ptr %9, align 4, !tbaa !49
  %10 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %11 = load ptr, ptr %10, align 8, !tbaa !20
  %12 = icmp eq ptr %11, null
  br i1 %12, label %13, label %14

13:                                               ; preds = %5
  tail call void @_ZSt25__throw_bad_function_callv() #24
  unreachable

14:                                               ; preds = %5
  %15 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %16 = load ptr, ptr %15, align 8, !tbaa !27
  call void %16(ptr noundef nonnull align 8 dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(8) %6, ptr noundef nonnull align 8 dereferenceable(8) %7, ptr noundef nonnull align 8 dereferenceable(8) %8, ptr noundef nonnull align 4 dereferenceable(4) %9)
  ret void
}

; Function Attrs: mustprogress noinline norecurse optnone uwtable
define internal fastcc void @_ZL18callThroughOptnoneIRSt8functionIFvPhS1_S1_jEEJS1_S1_RS1_RjEEvOT_DpOT0_(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %1, ptr noundef nonnull align 8 dereferenceable(8) %2, ptr noundef nonnull align 8 dereferenceable(8) %3, ptr noundef nonnull align 4 dereferenceable(4) %4) unnamed_addr #10 {
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  %10 = alloca ptr, align 8
  store ptr %0, ptr %6, align 8, !tbaa !239
  store ptr %1, ptr %7, align 8, !tbaa !73
  store ptr %2, ptr %8, align 8, !tbaa !73
  store ptr %3, ptr %9, align 8, !tbaa !73
  store ptr %4, ptr %10, align 8, !tbaa !58
  %11 = load ptr, ptr %6, align 8, !tbaa !239, !nonnull !76, !align !77
  %12 = load ptr, ptr %7, align 8, !tbaa !73, !nonnull !76, !align !77
  %13 = load ptr, ptr %12, align 8, !tbaa !54
  %14 = load ptr, ptr %8, align 8, !tbaa !73, !nonnull !76, !align !77
  %15 = load ptr, ptr %14, align 8, !tbaa !54
  %16 = load ptr, ptr %9, align 8, !tbaa !73, !nonnull !76, !align !77
  %17 = load ptr, ptr %16, align 8, !tbaa !54
  %18 = load ptr, ptr %10, align 8, !tbaa !58, !nonnull !76, !align !78
  %19 = load i32, ptr %18, align 4, !tbaa !49
  call void @_ZNKSt8functionIFvPhS0_S0_jEEclES0_S0_S0_j(ptr noundef nonnull align 8 dereferenceable(32) %11, ptr noundef %13, ptr noundef %15, ptr noundef %17, i32 noundef %19)
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPhS0_S0_jEZ4mainE4$_16E9_M_invokeERKSt9_Any_dataOS0_S7_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %3, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %4) #15 {
  %6 = load ptr, ptr %1, align 8, !tbaa !54
  %7 = load ptr, ptr %2, align 8, !tbaa !54
  %8 = load ptr, ptr %3, align 8, !tbaa !54
  %9 = load i32, ptr %4, align 4, !tbaa !49
  %10 = icmp eq i32 %9, 0
  br i1 %10, label %24, label %11

11:                                               ; preds = %5
  %12 = zext i32 %9 to i64
  br label %13

13:                                               ; preds = %13, %11
  %14 = phi i64 [ 0, %11 ], [ %22, %13 ]
  %15 = getelementptr inbounds nuw i8, ptr %7, i64 %14
  %16 = load i8, ptr %15, align 1, !tbaa !15
  %17 = getelementptr inbounds nuw i8, ptr %8, i64 %14
  %18 = load i8, ptr %17, align 1, !tbaa !15
  %19 = add i8 %16, 10
  %20 = add i8 %19, %18
  %21 = getelementptr inbounds nuw i8, ptr %6, i64 %14
  store i8 %20, ptr %21, align 1, !tbaa !15
  %22 = add nuw nsw i64 %14, 1
  %23 = icmp eq i64 %22, %12
  br i1 %23, label %24, label %13, !llvm.loop !241

24:                                               ; preds = %13, %5
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPhS0_S0_jEZ4mainE4$_16E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_16", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPhS0_S0_jEZ4mainE4$_17E9_M_invokeERKSt9_Any_dataOS0_S7_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %3, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %4) #15 {
  %6 = load ptr, ptr %1, align 8, !tbaa !54
  %7 = ptrtoint ptr %6 to i64
  %8 = load ptr, ptr %2, align 8, !tbaa !54
  %9 = ptrtoint ptr %8 to i64
  %10 = load ptr, ptr %3, align 8, !tbaa !54
  %11 = ptrtoint ptr %10 to i64
  %12 = load i32, ptr %4, align 4, !tbaa !49
  %13 = icmp eq i32 %12, 0
  br i1 %13, label %79, label %14

14:                                               ; preds = %5
  %15 = zext i32 %12 to i64
  %16 = icmp ult i32 %12, 8
  br i1 %16, label %66, label %17

17:                                               ; preds = %14
  %18 = sub i64 %7, %9
  %19 = icmp ult i64 %18, 32
  %20 = sub i64 %7, %11
  %21 = icmp ult i64 %20, 32
  %22 = or i1 %19, %21
  br i1 %22, label %66, label %23

23:                                               ; preds = %17
  %24 = icmp ult i32 %12, 32
  br i1 %24, label %50, label %25

25:                                               ; preds = %23
  %26 = and i64 %15, 4294967264
  br label %27

27:                                               ; preds = %27, %25
  %28 = phi i64 [ 0, %25 ], [ %43, %27 ]
  %29 = getelementptr inbounds nuw i8, ptr %8, i64 %28
  %30 = getelementptr inbounds nuw i8, ptr %29, i64 16
  %31 = load <16 x i8>, ptr %29, align 1, !tbaa !15
  %32 = load <16 x i8>, ptr %30, align 1, !tbaa !15
  %33 = getelementptr inbounds nuw i8, ptr %10, i64 %28
  %34 = getelementptr inbounds nuw i8, ptr %33, i64 16
  %35 = load <16 x i8>, ptr %33, align 1, !tbaa !15
  %36 = load <16 x i8>, ptr %34, align 1, !tbaa !15
  %37 = add <16 x i8> %31, splat (i8 10)
  %38 = add <16 x i8> %32, splat (i8 10)
  %39 = add <16 x i8> %37, %35
  %40 = add <16 x i8> %38, %36
  %41 = getelementptr inbounds nuw i8, ptr %6, i64 %28
  %42 = getelementptr inbounds nuw i8, ptr %41, i64 16
  store <16 x i8> %39, ptr %41, align 1, !tbaa !15
  store <16 x i8> %40, ptr %42, align 1, !tbaa !15
  %43 = add nuw i64 %28, 32
  %44 = icmp eq i64 %43, %26
  br i1 %44, label %45, label %27, !llvm.loop !242

45:                                               ; preds = %27
  %46 = icmp eq i64 %26, %15
  br i1 %46, label %79, label %47

47:                                               ; preds = %45
  %48 = and i64 %15, 24
  %49 = icmp eq i64 %48, 0
  br i1 %49, label %66, label %50

50:                                               ; preds = %47, %23
  %51 = phi i64 [ %26, %47 ], [ 0, %23 ]
  %52 = and i64 %15, 4294967288
  br label %53

53:                                               ; preds = %53, %50
  %54 = phi i64 [ %51, %50 ], [ %62, %53 ]
  %55 = getelementptr inbounds nuw i8, ptr %8, i64 %54
  %56 = load <8 x i8>, ptr %55, align 1, !tbaa !15
  %57 = getelementptr inbounds nuw i8, ptr %10, i64 %54
  %58 = load <8 x i8>, ptr %57, align 1, !tbaa !15
  %59 = add <8 x i8> %56, splat (i8 10)
  %60 = add <8 x i8> %59, %58
  %61 = getelementptr inbounds nuw i8, ptr %6, i64 %54
  store <8 x i8> %60, ptr %61, align 1, !tbaa !15
  %62 = add nuw i64 %54, 8
  %63 = icmp eq i64 %62, %52
  br i1 %63, label %64, label %53, !llvm.loop !243

64:                                               ; preds = %53
  %65 = icmp eq i64 %52, %15
  br i1 %65, label %79, label %66

66:                                               ; preds = %47, %64, %17, %14
  %67 = phi i64 [ 0, %14 ], [ 0, %17 ], [ %26, %47 ], [ %52, %64 ]
  br label %68

68:                                               ; preds = %66, %68
  %69 = phi i64 [ %77, %68 ], [ %67, %66 ]
  %70 = getelementptr inbounds nuw i8, ptr %8, i64 %69
  %71 = load i8, ptr %70, align 1, !tbaa !15
  %72 = getelementptr inbounds nuw i8, ptr %10, i64 %69
  %73 = load i8, ptr %72, align 1, !tbaa !15
  %74 = add i8 %71, 10
  %75 = add i8 %74, %73
  %76 = getelementptr inbounds nuw i8, ptr %6, i64 %69
  store i8 %75, ptr %76, align 1, !tbaa !15
  %77 = add nuw nsw i64 %69, 1
  %78 = icmp eq i64 %77, %15
  br i1 %78, label %79, label %68, !llvm.loop !244

79:                                               ; preds = %68, %45, %64, %5
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPhS0_S0_jEZ4mainE4$_17E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_17", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZNKSt8functionIFvPmS0_S0_jEEclES0_S0_S0_j(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef %1, ptr noundef %2, ptr noundef %3, i32 noundef %4) local_unnamed_addr #9 comdat {
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  %9 = alloca i32, align 4
  store ptr %1, ptr %6, align 8, !tbaa !62
  store ptr %2, ptr %7, align 8, !tbaa !62
  store ptr %3, ptr %8, align 8, !tbaa !62
  store i32 %4, ptr %9, align 4, !tbaa !49
  %10 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %11 = load ptr, ptr %10, align 8, !tbaa !20
  %12 = icmp eq ptr %11, null
  br i1 %12, label %13, label %14

13:                                               ; preds = %5
  tail call void @_ZSt25__throw_bad_function_callv() #24
  unreachable

14:                                               ; preds = %5
  %15 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %16 = load ptr, ptr %15, align 8, !tbaa !29
  call void %16(ptr noundef nonnull align 8 dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(8) %6, ptr noundef nonnull align 8 dereferenceable(8) %7, ptr noundef nonnull align 8 dereferenceable(8) %8, ptr noundef nonnull align 4 dereferenceable(4) %9)
  ret void
}

; Function Attrs: mustprogress noinline norecurse optnone uwtable
define internal fastcc void @_ZL18callThroughOptnoneIRSt8functionIFvPmS1_S1_jEEJS1_S1_RS1_RjEEvOT_DpOT0_(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %1, ptr noundef nonnull align 8 dereferenceable(8) %2, ptr noundef nonnull align 8 dereferenceable(8) %3, ptr noundef nonnull align 4 dereferenceable(4) %4) unnamed_addr #10 {
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  %10 = alloca ptr, align 8
  store ptr %0, ptr %6, align 8, !tbaa !245
  store ptr %1, ptr %7, align 8, !tbaa !106
  store ptr %2, ptr %8, align 8, !tbaa !106
  store ptr %3, ptr %9, align 8, !tbaa !106
  store ptr %4, ptr %10, align 8, !tbaa !58
  %11 = load ptr, ptr %6, align 8, !tbaa !245, !nonnull !76, !align !77
  %12 = load ptr, ptr %7, align 8, !tbaa !106, !nonnull !76, !align !77
  %13 = load ptr, ptr %12, align 8, !tbaa !62
  %14 = load ptr, ptr %8, align 8, !tbaa !106, !nonnull !76, !align !77
  %15 = load ptr, ptr %14, align 8, !tbaa !62
  %16 = load ptr, ptr %9, align 8, !tbaa !106, !nonnull !76, !align !77
  %17 = load ptr, ptr %16, align 8, !tbaa !62
  %18 = load ptr, ptr %10, align 8, !tbaa !58, !nonnull !76, !align !78
  %19 = load i32, ptr %18, align 4, !tbaa !49
  call void @_ZNKSt8functionIFvPmS0_S0_jEEclES0_S0_S0_j(ptr noundef nonnull align 8 dereferenceable(32) %11, ptr noundef %13, ptr noundef %15, ptr noundef %17, i32 noundef %19)
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPmS0_S0_jEZ4mainE4$_16E9_M_invokeERKSt9_Any_dataOS0_S7_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %3, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %4) #15 {
  %6 = load ptr, ptr %1, align 8, !tbaa !62
  %7 = load ptr, ptr %2, align 8, !tbaa !62
  %8 = load ptr, ptr %3, align 8, !tbaa !62
  %9 = load i32, ptr %4, align 4, !tbaa !49
  %10 = icmp eq i32 %9, 0
  br i1 %10, label %24, label %11

11:                                               ; preds = %5
  %12 = zext i32 %9 to i64
  br label %13

13:                                               ; preds = %13, %11
  %14 = phi i64 [ 0, %11 ], [ %22, %13 ]
  %15 = getelementptr inbounds nuw i64, ptr %7, i64 %14
  %16 = load i64, ptr %15, align 8, !tbaa !6
  %17 = getelementptr inbounds nuw i64, ptr %8, i64 %14
  %18 = load i64, ptr %17, align 8, !tbaa !6
  %19 = add i64 %16, 10
  %20 = add i64 %19, %18
  %21 = getelementptr inbounds nuw i64, ptr %6, i64 %14
  store i64 %20, ptr %21, align 8, !tbaa !6
  %22 = add nuw nsw i64 %14, 1
  %23 = icmp eq i64 %22, %12
  br i1 %23, label %24, label %13, !llvm.loop !247

24:                                               ; preds = %13, %5
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPmS0_S0_jEZ4mainE4$_16E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_16", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPmS0_S0_jEZ4mainE4$_17E9_M_invokeERKSt9_Any_dataOS0_S7_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %3, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %4) #15 {
  %6 = load ptr, ptr %1, align 8, !tbaa !62
  %7 = ptrtoint ptr %6 to i64
  %8 = load ptr, ptr %2, align 8, !tbaa !62
  %9 = ptrtoint ptr %8 to i64
  %10 = load ptr, ptr %3, align 8, !tbaa !62
  %11 = ptrtoint ptr %10 to i64
  %12 = load i32, ptr %4, align 4, !tbaa !49
  %13 = icmp eq i32 %12, 0
  br i1 %13, label %58, label %14

14:                                               ; preds = %5
  %15 = zext i32 %12 to i64
  %16 = icmp ult i32 %12, 4
  br i1 %16, label %45, label %17

17:                                               ; preds = %14
  %18 = sub i64 %7, %9
  %19 = icmp ult i64 %18, 32
  %20 = sub i64 %7, %11
  %21 = icmp ult i64 %20, 32
  %22 = or i1 %19, %21
  br i1 %22, label %45, label %23

23:                                               ; preds = %17
  %24 = and i64 %15, 4294967292
  br label %25

25:                                               ; preds = %25, %23
  %26 = phi i64 [ 0, %23 ], [ %41, %25 ]
  %27 = getelementptr inbounds nuw i64, ptr %8, i64 %26
  %28 = getelementptr inbounds nuw i8, ptr %27, i64 16
  %29 = load <2 x i64>, ptr %27, align 8, !tbaa !6
  %30 = load <2 x i64>, ptr %28, align 8, !tbaa !6
  %31 = getelementptr inbounds nuw i64, ptr %10, i64 %26
  %32 = getelementptr inbounds nuw i8, ptr %31, i64 16
  %33 = load <2 x i64>, ptr %31, align 8, !tbaa !6
  %34 = load <2 x i64>, ptr %32, align 8, !tbaa !6
  %35 = add <2 x i64> %29, splat (i64 10)
  %36 = add <2 x i64> %30, splat (i64 10)
  %37 = add <2 x i64> %35, %33
  %38 = add <2 x i64> %36, %34
  %39 = getelementptr inbounds nuw i64, ptr %6, i64 %26
  %40 = getelementptr inbounds nuw i8, ptr %39, i64 16
  store <2 x i64> %37, ptr %39, align 8, !tbaa !6
  store <2 x i64> %38, ptr %40, align 8, !tbaa !6
  %41 = add nuw i64 %26, 4
  %42 = icmp eq i64 %41, %24
  br i1 %42, label %43, label %25, !llvm.loop !248

43:                                               ; preds = %25
  %44 = icmp eq i64 %24, %15
  br i1 %44, label %58, label %45

45:                                               ; preds = %17, %14, %43
  %46 = phi i64 [ 0, %17 ], [ 0, %14 ], [ %24, %43 ]
  br label %47

47:                                               ; preds = %45, %47
  %48 = phi i64 [ %56, %47 ], [ %46, %45 ]
  %49 = getelementptr inbounds nuw i64, ptr %8, i64 %48
  %50 = load i64, ptr %49, align 8, !tbaa !6
  %51 = getelementptr inbounds nuw i64, ptr %10, i64 %48
  %52 = load i64, ptr %51, align 8, !tbaa !6
  %53 = add i64 %50, 10
  %54 = add i64 %53, %52
  %55 = getelementptr inbounds nuw i64, ptr %6, i64 %48
  store i64 %54, ptr %55, align 8, !tbaa !6
  %56 = add nuw nsw i64 %48, 1
  %57 = icmp eq i64 %56, %15
  br i1 %57, label %58, label %47, !llvm.loop !249

58:                                               ; preds = %47, %43, %5
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPmS0_S0_jEZ4mainE4$_17E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_17", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPhS0_S0_jEZ4mainE4$_18E9_M_invokeERKSt9_Any_dataOS0_S7_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %3, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %4) #15 {
  %6 = load ptr, ptr %1, align 8, !tbaa !54
  %7 = load ptr, ptr %2, align 8, !tbaa !54
  %8 = load ptr, ptr %3, align 8, !tbaa !54
  %9 = load i32, ptr %4, align 4, !tbaa !49
  %10 = icmp eq i32 %9, 0
  br i1 %10, label %23, label %11

11:                                               ; preds = %5
  %12 = zext i32 %9 to i64
  br label %13

13:                                               ; preds = %13, %11
  %14 = phi i64 [ 0, %11 ], [ %21, %13 ]
  %15 = getelementptr inbounds nuw i8, ptr %8, i64 %14
  %16 = load i8, ptr %15, align 1, !tbaa !15
  %17 = add i8 %16, 10
  %18 = getelementptr inbounds nuw i8, ptr %6, i64 %14
  store i8 %17, ptr %18, align 1, !tbaa !15
  %19 = add i8 %16, 19
  %20 = getelementptr inbounds nuw i8, ptr %7, i64 %14
  store i8 %19, ptr %20, align 1, !tbaa !15
  %21 = add nuw nsw i64 %14, 1
  %22 = icmp eq i64 %21, %12
  br i1 %22, label %23, label %13, !llvm.loop !250

23:                                               ; preds = %13, %5
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPhS0_S0_jEZ4mainE4$_18E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_18", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPhS0_S0_jEZ4mainE4$_19E9_M_invokeERKSt9_Any_dataOS0_S7_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %3, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %4) #15 {
  %6 = load ptr, ptr %1, align 8, !tbaa !54
  %7 = ptrtoint ptr %6 to i64
  %8 = load ptr, ptr %2, align 8, !tbaa !54
  %9 = ptrtoint ptr %8 to i64
  %10 = load ptr, ptr %3, align 8, !tbaa !54
  %11 = ptrtoint ptr %10 to i64
  %12 = load i32, ptr %4, align 4, !tbaa !49
  %13 = icmp eq i32 %12, 0
  br i1 %13, label %78, label %14

14:                                               ; preds = %5
  %15 = zext i32 %12 to i64
  %16 = icmp ult i32 %12, 8
  br i1 %16, label %66, label %17

17:                                               ; preds = %14
  %18 = sub i64 %9, %7
  %19 = icmp ult i64 %18, 32
  %20 = sub i64 %7, %11
  %21 = icmp ult i64 %20, 32
  %22 = or i1 %19, %21
  %23 = sub i64 %9, %11
  %24 = icmp ult i64 %23, 32
  %25 = or i1 %22, %24
  br i1 %25, label %66, label %26

26:                                               ; preds = %17
  %27 = icmp ult i32 %12, 32
  br i1 %27, label %51, label %28

28:                                               ; preds = %26
  %29 = and i64 %15, 4294967264
  br label %30

30:                                               ; preds = %30, %28
  %31 = phi i64 [ 0, %28 ], [ %44, %30 ]
  %32 = getelementptr inbounds nuw i8, ptr %10, i64 %31
  %33 = getelementptr inbounds nuw i8, ptr %32, i64 16
  %34 = load <16 x i8>, ptr %32, align 1, !tbaa !15
  %35 = load <16 x i8>, ptr %33, align 1, !tbaa !15
  %36 = add <16 x i8> %34, splat (i8 10)
  %37 = add <16 x i8> %35, splat (i8 10)
  %38 = getelementptr inbounds nuw i8, ptr %6, i64 %31
  %39 = getelementptr inbounds nuw i8, ptr %38, i64 16
  store <16 x i8> %36, ptr %38, align 1, !tbaa !15
  store <16 x i8> %37, ptr %39, align 1, !tbaa !15
  %40 = add <16 x i8> %34, splat (i8 19)
  %41 = add <16 x i8> %35, splat (i8 19)
  %42 = getelementptr inbounds nuw i8, ptr %8, i64 %31
  %43 = getelementptr inbounds nuw i8, ptr %42, i64 16
  store <16 x i8> %40, ptr %42, align 1, !tbaa !15
  store <16 x i8> %41, ptr %43, align 1, !tbaa !15
  %44 = add nuw i64 %31, 32
  %45 = icmp eq i64 %44, %29
  br i1 %45, label %46, label %30, !llvm.loop !251

46:                                               ; preds = %30
  %47 = icmp eq i64 %29, %15
  br i1 %47, label %78, label %48

48:                                               ; preds = %46
  %49 = and i64 %15, 24
  %50 = icmp eq i64 %49, 0
  br i1 %50, label %66, label %51

51:                                               ; preds = %48, %26
  %52 = phi i64 [ %29, %48 ], [ 0, %26 ]
  %53 = and i64 %15, 4294967288
  br label %54

54:                                               ; preds = %54, %51
  %55 = phi i64 [ %52, %51 ], [ %62, %54 ]
  %56 = getelementptr inbounds nuw i8, ptr %10, i64 %55
  %57 = load <8 x i8>, ptr %56, align 1, !tbaa !15
  %58 = add <8 x i8> %57, splat (i8 10)
  %59 = getelementptr inbounds nuw i8, ptr %6, i64 %55
  store <8 x i8> %58, ptr %59, align 1, !tbaa !15
  %60 = add <8 x i8> %57, splat (i8 19)
  %61 = getelementptr inbounds nuw i8, ptr %8, i64 %55
  store <8 x i8> %60, ptr %61, align 1, !tbaa !15
  %62 = add nuw i64 %55, 8
  %63 = icmp eq i64 %62, %53
  br i1 %63, label %64, label %54, !llvm.loop !252

64:                                               ; preds = %54
  %65 = icmp eq i64 %53, %15
  br i1 %65, label %78, label %66

66:                                               ; preds = %48, %64, %17, %14
  %67 = phi i64 [ 0, %14 ], [ 0, %17 ], [ %29, %48 ], [ %53, %64 ]
  br label %68

68:                                               ; preds = %66, %68
  %69 = phi i64 [ %76, %68 ], [ %67, %66 ]
  %70 = getelementptr inbounds nuw i8, ptr %10, i64 %69
  %71 = load i8, ptr %70, align 1, !tbaa !15
  %72 = add i8 %71, 10
  %73 = getelementptr inbounds nuw i8, ptr %6, i64 %69
  store i8 %72, ptr %73, align 1, !tbaa !15
  %74 = add i8 %71, 19
  %75 = getelementptr inbounds nuw i8, ptr %8, i64 %69
  store i8 %74, ptr %75, align 1, !tbaa !15
  %76 = add nuw nsw i64 %69, 1
  %77 = icmp eq i64 %76, %15
  br i1 %77, label %78, label %68, !llvm.loop !253

78:                                               ; preds = %68, %46, %64, %5
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPhS0_S0_jEZ4mainE4$_19E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_19", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPjS0_S0_jEZ4mainE4$_18E9_M_invokeERKSt9_Any_dataOS0_S7_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %3, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %4) #15 {
  %6 = load ptr, ptr %1, align 8, !tbaa !58
  %7 = load ptr, ptr %2, align 8, !tbaa !58
  %8 = load ptr, ptr %3, align 8, !tbaa !58
  %9 = load i32, ptr %4, align 4, !tbaa !49
  %10 = icmp eq i32 %9, 0
  br i1 %10, label %23, label %11

11:                                               ; preds = %5
  %12 = zext i32 %9 to i64
  br label %13

13:                                               ; preds = %13, %11
  %14 = phi i64 [ 0, %11 ], [ %21, %13 ]
  %15 = getelementptr inbounds nuw i32, ptr %8, i64 %14
  %16 = load i32, ptr %15, align 4, !tbaa !49
  %17 = add i32 %16, 10
  %18 = getelementptr inbounds nuw i32, ptr %6, i64 %14
  store i32 %17, ptr %18, align 4, !tbaa !49
  %19 = add i32 %16, 19
  %20 = getelementptr inbounds nuw i32, ptr %7, i64 %14
  store i32 %19, ptr %20, align 4, !tbaa !49
  %21 = add nuw nsw i64 %14, 1
  %22 = icmp eq i64 %21, %12
  br i1 %22, label %23, label %13, !llvm.loop !254

23:                                               ; preds = %13, %5
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPjS0_S0_jEZ4mainE4$_18E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_18", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPjS0_S0_jEZ4mainE4$_19E9_M_invokeERKSt9_Any_dataOS0_S7_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %3, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %4) #15 {
  %6 = load ptr, ptr %1, align 8, !tbaa !58
  %7 = ptrtoint ptr %6 to i64
  %8 = load ptr, ptr %2, align 8, !tbaa !58
  %9 = ptrtoint ptr %8 to i64
  %10 = load ptr, ptr %3, align 8, !tbaa !58
  %11 = ptrtoint ptr %10 to i64
  %12 = load i32, ptr %4, align 4, !tbaa !49
  %13 = icmp eq i32 %12, 0
  br i1 %13, label %58, label %14

14:                                               ; preds = %5
  %15 = zext i32 %12 to i64
  %16 = icmp ult i32 %12, 8
  br i1 %16, label %46, label %17

17:                                               ; preds = %14
  %18 = sub i64 %9, %7
  %19 = icmp ult i64 %18, 32
  %20 = sub i64 %7, %11
  %21 = icmp ult i64 %20, 32
  %22 = or i1 %19, %21
  %23 = sub i64 %9, %11
  %24 = icmp ult i64 %23, 32
  %25 = or i1 %22, %24
  br i1 %25, label %46, label %26

26:                                               ; preds = %17
  %27 = and i64 %15, 4294967288
  br label %28

28:                                               ; preds = %28, %26
  %29 = phi i64 [ 0, %26 ], [ %42, %28 ]
  %30 = getelementptr inbounds nuw i32, ptr %10, i64 %29
  %31 = getelementptr inbounds nuw i8, ptr %30, i64 16
  %32 = load <4 x i32>, ptr %30, align 4, !tbaa !49
  %33 = load <4 x i32>, ptr %31, align 4, !tbaa !49
  %34 = add <4 x i32> %32, splat (i32 10)
  %35 = add <4 x i32> %33, splat (i32 10)
  %36 = getelementptr inbounds nuw i32, ptr %6, i64 %29
  %37 = getelementptr inbounds nuw i8, ptr %36, i64 16
  store <4 x i32> %34, ptr %36, align 4, !tbaa !49
  store <4 x i32> %35, ptr %37, align 4, !tbaa !49
  %38 = add <4 x i32> %32, splat (i32 19)
  %39 = add <4 x i32> %33, splat (i32 19)
  %40 = getelementptr inbounds nuw i32, ptr %8, i64 %29
  %41 = getelementptr inbounds nuw i8, ptr %40, i64 16
  store <4 x i32> %38, ptr %40, align 4, !tbaa !49
  store <4 x i32> %39, ptr %41, align 4, !tbaa !49
  %42 = add nuw i64 %29, 8
  %43 = icmp eq i64 %42, %27
  br i1 %43, label %44, label %28, !llvm.loop !255

44:                                               ; preds = %28
  %45 = icmp eq i64 %27, %15
  br i1 %45, label %58, label %46

46:                                               ; preds = %17, %14, %44
  %47 = phi i64 [ 0, %17 ], [ 0, %14 ], [ %27, %44 ]
  br label %48

48:                                               ; preds = %46, %48
  %49 = phi i64 [ %56, %48 ], [ %47, %46 ]
  %50 = getelementptr inbounds nuw i32, ptr %10, i64 %49
  %51 = load i32, ptr %50, align 4, !tbaa !49
  %52 = add i32 %51, 10
  %53 = getelementptr inbounds nuw i32, ptr %6, i64 %49
  store i32 %52, ptr %53, align 4, !tbaa !49
  %54 = add i32 %51, 19
  %55 = getelementptr inbounds nuw i32, ptr %8, i64 %49
  store i32 %54, ptr %55, align 4, !tbaa !49
  %56 = add nuw nsw i64 %49, 1
  %57 = icmp eq i64 %56, %15
  br i1 %57, label %58, label %48, !llvm.loop !256

58:                                               ; preds = %48, %44, %5
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPjS0_S0_jEZ4mainE4$_19E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_19", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPmS0_S0_jEZ4mainE4$_18E9_M_invokeERKSt9_Any_dataOS0_S7_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %3, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %4) #15 {
  %6 = load ptr, ptr %1, align 8, !tbaa !62
  %7 = load ptr, ptr %2, align 8, !tbaa !62
  %8 = load ptr, ptr %3, align 8, !tbaa !62
  %9 = load i32, ptr %4, align 4, !tbaa !49
  %10 = icmp eq i32 %9, 0
  br i1 %10, label %23, label %11

11:                                               ; preds = %5
  %12 = zext i32 %9 to i64
  br label %13

13:                                               ; preds = %13, %11
  %14 = phi i64 [ 0, %11 ], [ %21, %13 ]
  %15 = getelementptr inbounds nuw i64, ptr %8, i64 %14
  %16 = load i64, ptr %15, align 8, !tbaa !6
  %17 = add i64 %16, 10
  %18 = getelementptr inbounds nuw i64, ptr %6, i64 %14
  store i64 %17, ptr %18, align 8, !tbaa !6
  %19 = add i64 %16, 19
  %20 = getelementptr inbounds nuw i64, ptr %7, i64 %14
  store i64 %19, ptr %20, align 8, !tbaa !6
  %21 = add nuw nsw i64 %14, 1
  %22 = icmp eq i64 %21, %12
  br i1 %22, label %23, label %13, !llvm.loop !257

23:                                               ; preds = %13, %5
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPmS0_S0_jEZ4mainE4$_18E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_18", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPmS0_S0_jEZ4mainE4$_19E9_M_invokeERKSt9_Any_dataOS0_S7_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %3, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %4) #15 {
  %6 = load ptr, ptr %1, align 8, !tbaa !62
  %7 = ptrtoint ptr %6 to i64
  %8 = load ptr, ptr %2, align 8, !tbaa !62
  %9 = ptrtoint ptr %8 to i64
  %10 = load ptr, ptr %3, align 8, !tbaa !62
  %11 = ptrtoint ptr %10 to i64
  %12 = load i32, ptr %4, align 4, !tbaa !49
  %13 = icmp eq i32 %12, 0
  br i1 %13, label %58, label %14

14:                                               ; preds = %5
  %15 = zext i32 %12 to i64
  %16 = icmp ult i32 %12, 4
  br i1 %16, label %46, label %17

17:                                               ; preds = %14
  %18 = sub i64 %9, %7
  %19 = icmp ult i64 %18, 32
  %20 = sub i64 %7, %11
  %21 = icmp ult i64 %20, 32
  %22 = or i1 %19, %21
  %23 = sub i64 %9, %11
  %24 = icmp ult i64 %23, 32
  %25 = or i1 %22, %24
  br i1 %25, label %46, label %26

26:                                               ; preds = %17
  %27 = and i64 %15, 4294967292
  br label %28

28:                                               ; preds = %28, %26
  %29 = phi i64 [ 0, %26 ], [ %42, %28 ]
  %30 = getelementptr inbounds nuw i64, ptr %10, i64 %29
  %31 = getelementptr inbounds nuw i8, ptr %30, i64 16
  %32 = load <2 x i64>, ptr %30, align 8, !tbaa !6
  %33 = load <2 x i64>, ptr %31, align 8, !tbaa !6
  %34 = add <2 x i64> %32, splat (i64 10)
  %35 = add <2 x i64> %33, splat (i64 10)
  %36 = getelementptr inbounds nuw i64, ptr %6, i64 %29
  %37 = getelementptr inbounds nuw i8, ptr %36, i64 16
  store <2 x i64> %34, ptr %36, align 8, !tbaa !6
  store <2 x i64> %35, ptr %37, align 8, !tbaa !6
  %38 = add <2 x i64> %32, splat (i64 19)
  %39 = add <2 x i64> %33, splat (i64 19)
  %40 = getelementptr inbounds nuw i64, ptr %8, i64 %29
  %41 = getelementptr inbounds nuw i8, ptr %40, i64 16
  store <2 x i64> %38, ptr %40, align 8, !tbaa !6
  store <2 x i64> %39, ptr %41, align 8, !tbaa !6
  %42 = add nuw i64 %29, 4
  %43 = icmp eq i64 %42, %27
  br i1 %43, label %44, label %28, !llvm.loop !258

44:                                               ; preds = %28
  %45 = icmp eq i64 %27, %15
  br i1 %45, label %58, label %46

46:                                               ; preds = %17, %14, %44
  %47 = phi i64 [ 0, %17 ], [ 0, %14 ], [ %27, %44 ]
  br label %48

48:                                               ; preds = %46, %48
  %49 = phi i64 [ %56, %48 ], [ %47, %46 ]
  %50 = getelementptr inbounds nuw i64, ptr %10, i64 %49
  %51 = load i64, ptr %50, align 8, !tbaa !6
  %52 = add i64 %51, 10
  %53 = getelementptr inbounds nuw i64, ptr %6, i64 %49
  store i64 %52, ptr %53, align 8, !tbaa !6
  %54 = add i64 %51, 19
  %55 = getelementptr inbounds nuw i64, ptr %8, i64 %49
  store i64 %54, ptr %55, align 8, !tbaa !6
  %56 = add nuw nsw i64 %49, 1
  %57 = icmp eq i64 %56, %15
  br i1 %57, label %58, label %48, !llvm.loop !259

58:                                               ; preds = %48, %44, %5
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPmS0_S0_jEZ4mainE4$_19E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_19", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZNKSt8functionIFvPhS0_jjEEclES0_S0_jj(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef %1, ptr noundef %2, i32 noundef %3, i32 noundef %4) local_unnamed_addr #9 comdat {
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca i32, align 4
  %9 = alloca i32, align 4
  store ptr %1, ptr %6, align 8, !tbaa !54
  store ptr %2, ptr %7, align 8, !tbaa !54
  store i32 %3, ptr %8, align 4, !tbaa !49
  store i32 %4, ptr %9, align 4, !tbaa !49
  %10 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %11 = load ptr, ptr %10, align 8, !tbaa !20
  %12 = icmp eq ptr %11, null
  br i1 %12, label %13, label %14

13:                                               ; preds = %5
  tail call void @_ZSt25__throw_bad_function_callv() #24
  unreachable

14:                                               ; preds = %5
  %15 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %16 = load ptr, ptr %15, align 8, !tbaa !31
  call void %16(ptr noundef nonnull align 8 dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(8) %6, ptr noundef nonnull align 8 dereferenceable(8) %7, ptr noundef nonnull align 4 dereferenceable(4) %8, ptr noundef nonnull align 4 dereferenceable(4) %9)
  ret void
}

; Function Attrs: mustprogress noinline norecurse optnone uwtable
define internal fastcc void @_ZL18callThroughOptnoneIRSt8functionIFvPhS1_jjEEJS1_RS1_RKiS7_EEvOT_DpOT0_(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %1, ptr noundef nonnull align 8 dereferenceable(8) %2, ptr noundef nonnull align 4 dereferenceable(4) %3, ptr noundef nonnull align 4 dereferenceable(4) %4) unnamed_addr #10 {
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  %10 = alloca ptr, align 8
  store ptr %0, ptr %6, align 8, !tbaa !260
  store ptr %1, ptr %7, align 8, !tbaa !73
  store ptr %2, ptr %8, align 8, !tbaa !73
  store ptr %3, ptr %9, align 8, !tbaa !58
  store ptr %4, ptr %10, align 8, !tbaa !58
  %11 = load ptr, ptr %6, align 8, !tbaa !260, !nonnull !76, !align !77
  %12 = load ptr, ptr %7, align 8, !tbaa !73, !nonnull !76, !align !77
  %13 = load ptr, ptr %12, align 8, !tbaa !54
  %14 = load ptr, ptr %8, align 8, !tbaa !73, !nonnull !76, !align !77
  %15 = load ptr, ptr %14, align 8, !tbaa !54
  %16 = load ptr, ptr %9, align 8, !tbaa !58, !nonnull !76, !align !78
  %17 = load i32, ptr %16, align 4, !tbaa !49
  %18 = load ptr, ptr %10, align 8, !tbaa !58, !nonnull !76, !align !78
  %19 = load i32, ptr %18, align 4, !tbaa !49
  call void @_ZNKSt8functionIFvPhS0_jjEEclES0_S0_jj(ptr noundef nonnull align 8 dereferenceable(32) %11, ptr noundef %13, ptr noundef %15, i32 noundef %17, i32 noundef %19)
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPhS0_jjEZ4mainE4$_20E9_M_invokeERKSt9_Any_dataOS0_S7_OjS8_"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %4) #15 {
  %6 = load ptr, ptr %1, align 8, !tbaa !54
  %7 = load ptr, ptr %2, align 8, !tbaa !54
  %8 = load i32, ptr %3, align 4, !tbaa !49
  %9 = zext i32 %8 to i64
  %10 = icmp eq i32 %8, 0
  br i1 %10, label %33, label %11

11:                                               ; preds = %5
  %12 = load i32, ptr %4, align 4, !tbaa !49
  %13 = zext i32 %12 to i64
  %14 = icmp eq i32 %12, 0
  %15 = add i32 %8, 1
  %16 = zext i32 %15 to i64
  br i1 %14, label %33, label %17

17:                                               ; preds = %11, %30
  %18 = phi i64 [ %31, %30 ], [ 0, %11 ]
  %19 = mul nuw i64 %18, %9
  %20 = getelementptr inbounds nuw i8, ptr %7, i64 %19
  %21 = mul nuw i64 %18, %16
  %22 = getelementptr inbounds nuw i8, ptr %6, i64 %21
  br label %23

23:                                               ; preds = %23, %17
  %24 = phi i64 [ 0, %17 ], [ %28, %23 ]
  %25 = getelementptr inbounds nuw i8, ptr %20, i64 %24
  %26 = load i8, ptr %25, align 1, !tbaa !15
  %27 = getelementptr inbounds nuw i8, ptr %22, i64 %24
  store i8 %26, ptr %27, align 1, !tbaa !15
  %28 = add nuw nsw i64 %24, 1
  %29 = icmp eq i64 %28, %13
  br i1 %29, label %30, label %23, !llvm.loop !262

30:                                               ; preds = %23
  %31 = add nuw nsw i64 %18, 1
  %32 = icmp eq i64 %31, %9
  br i1 %32, label %33, label %17, !llvm.loop !263

33:                                               ; preds = %30, %5, %11
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPhS0_jjEZ4mainE4$_20E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_20", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPhS0_jjEZ4mainE4$_21E9_M_invokeERKSt9_Any_dataOS0_S7_OjS8_"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %4) #15 {
  %6 = load ptr, ptr %1, align 8, !tbaa !54
  %7 = load ptr, ptr %2, align 8, !tbaa !54
  %8 = load i32, ptr %3, align 4, !tbaa !49
  %9 = zext i32 %8 to i64
  %10 = icmp eq i32 %8, 0
  br i1 %10, label %78, label %11

11:                                               ; preds = %5
  %12 = load i32, ptr %4, align 4, !tbaa !49
  %13 = zext i32 %12 to i64
  %14 = icmp eq i32 %12, 0
  %15 = add i32 %8, 1
  %16 = zext i32 %15 to i64
  br i1 %14, label %78, label %17

17:                                               ; preds = %11
  %18 = add nsw i64 %9, -1
  %19 = mul i64 %18, %16
  %20 = getelementptr i8, ptr %6, i64 %19
  %21 = getelementptr i8, ptr %20, i64 %13
  %22 = mul i64 %18, %9
  %23 = getelementptr i8, ptr %7, i64 %22
  %24 = getelementptr i8, ptr %23, i64 %13
  %25 = icmp ult i32 %12, 8
  %26 = icmp ult ptr %6, %24
  %27 = icmp ult ptr %7, %21
  %28 = and i1 %26, %27
  %29 = icmp ult i32 %12, 32
  %30 = and i64 %13, 4294967264
  %31 = icmp eq i64 %30, %13
  %32 = and i64 %13, 24
  %33 = icmp eq i64 %32, 0
  %34 = and i64 %13, 4294967288
  %35 = icmp eq i64 %34, %13
  br label %36

36:                                               ; preds = %17, %75
  %37 = phi i64 [ %76, %75 ], [ 0, %17 ]
  %38 = mul nuw i64 %37, %9
  %39 = getelementptr inbounds nuw i8, ptr %7, i64 %38
  %40 = mul nuw i64 %37, %16
  %41 = getelementptr inbounds nuw i8, ptr %6, i64 %40
  %42 = select i1 %25, i1 true, i1 %28
  br i1 %42, label %66, label %43

43:                                               ; preds = %36
  br i1 %29, label %56, label %44

44:                                               ; preds = %43, %44
  %45 = phi i64 [ %52, %44 ], [ 0, %43 ]
  %46 = getelementptr inbounds nuw i8, ptr %39, i64 %45
  %47 = getelementptr inbounds nuw i8, ptr %46, i64 16
  %48 = load <16 x i8>, ptr %46, align 1, !tbaa !15, !alias.scope !264
  %49 = load <16 x i8>, ptr %47, align 1, !tbaa !15, !alias.scope !264
  %50 = getelementptr inbounds nuw i8, ptr %41, i64 %45
  %51 = getelementptr inbounds nuw i8, ptr %50, i64 16
  store <16 x i8> %48, ptr %50, align 1, !tbaa !15, !alias.scope !267, !noalias !264
  store <16 x i8> %49, ptr %51, align 1, !tbaa !15, !alias.scope !267, !noalias !264
  %52 = add nuw i64 %45, 32
  %53 = icmp eq i64 %52, %30
  br i1 %53, label %54, label %44, !llvm.loop !269

54:                                               ; preds = %44
  br i1 %31, label %75, label %55

55:                                               ; preds = %54
  br i1 %33, label %66, label %56

56:                                               ; preds = %55, %43
  %57 = phi i64 [ %30, %55 ], [ 0, %43 ]
  br label %58

58:                                               ; preds = %58, %56
  %59 = phi i64 [ %57, %56 ], [ %63, %58 ]
  %60 = getelementptr inbounds nuw i8, ptr %39, i64 %59
  %61 = load <8 x i8>, ptr %60, align 1, !tbaa !15, !alias.scope !264
  %62 = getelementptr inbounds nuw i8, ptr %41, i64 %59
  store <8 x i8> %61, ptr %62, align 1, !tbaa !15, !alias.scope !267, !noalias !264
  %63 = add nuw i64 %59, 8
  %64 = icmp eq i64 %63, %34
  br i1 %64, label %65, label %58, !llvm.loop !270

65:                                               ; preds = %58
  br i1 %35, label %75, label %66

66:                                               ; preds = %36, %55, %65
  %67 = phi i64 [ 0, %36 ], [ %30, %55 ], [ %34, %65 ]
  br label %68

68:                                               ; preds = %66, %68
  %69 = phi i64 [ %73, %68 ], [ %67, %66 ]
  %70 = getelementptr inbounds nuw i8, ptr %39, i64 %69
  %71 = load i8, ptr %70, align 1, !tbaa !15
  %72 = getelementptr inbounds nuw i8, ptr %41, i64 %69
  store i8 %71, ptr %72, align 1, !tbaa !15
  %73 = add nuw nsw i64 %69, 1
  %74 = icmp eq i64 %73, %13
  br i1 %74, label %75, label %68, !llvm.loop !271

75:                                               ; preds = %68, %65, %54
  %76 = add nuw nsw i64 %37, 1
  %77 = icmp eq i64 %76, %9
  br i1 %77, label %78, label %36, !llvm.loop !272

78:                                               ; preds = %75, %5, %11
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPhS0_jjEZ4mainE4$_21E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_21", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZNKSt8functionIFvPjS0_jjEEclES0_S0_jj(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef %1, ptr noundef %2, i32 noundef %3, i32 noundef %4) local_unnamed_addr #9 comdat {
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca i32, align 4
  %9 = alloca i32, align 4
  store ptr %1, ptr %6, align 8, !tbaa !58
  store ptr %2, ptr %7, align 8, !tbaa !58
  store i32 %3, ptr %8, align 4, !tbaa !49
  store i32 %4, ptr %9, align 4, !tbaa !49
  %10 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %11 = load ptr, ptr %10, align 8, !tbaa !20
  %12 = icmp eq ptr %11, null
  br i1 %12, label %13, label %14

13:                                               ; preds = %5
  tail call void @_ZSt25__throw_bad_function_callv() #24
  unreachable

14:                                               ; preds = %5
  %15 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %16 = load ptr, ptr %15, align 8, !tbaa !33
  call void %16(ptr noundef nonnull align 8 dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(8) %6, ptr noundef nonnull align 8 dereferenceable(8) %7, ptr noundef nonnull align 4 dereferenceable(4) %8, ptr noundef nonnull align 4 dereferenceable(4) %9)
  ret void
}

; Function Attrs: mustprogress noinline norecurse optnone uwtable
define internal fastcc void @_ZL18callThroughOptnoneIRSt8functionIFvPjS1_jjEEJS1_RS1_RKiS7_EEvOT_DpOT0_(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %1, ptr noundef nonnull align 8 dereferenceable(8) %2, ptr noundef nonnull align 4 dereferenceable(4) %3, ptr noundef nonnull align 4 dereferenceable(4) %4) unnamed_addr #10 {
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  %10 = alloca ptr, align 8
  store ptr %0, ptr %6, align 8, !tbaa !273
  store ptr %1, ptr %7, align 8, !tbaa !94
  store ptr %2, ptr %8, align 8, !tbaa !94
  store ptr %3, ptr %9, align 8, !tbaa !58
  store ptr %4, ptr %10, align 8, !tbaa !58
  %11 = load ptr, ptr %6, align 8, !tbaa !273, !nonnull !76, !align !77
  %12 = load ptr, ptr %7, align 8, !tbaa !94, !nonnull !76, !align !77
  %13 = load ptr, ptr %12, align 8, !tbaa !58
  %14 = load ptr, ptr %8, align 8, !tbaa !94, !nonnull !76, !align !77
  %15 = load ptr, ptr %14, align 8, !tbaa !58
  %16 = load ptr, ptr %9, align 8, !tbaa !58, !nonnull !76, !align !78
  %17 = load i32, ptr %16, align 4, !tbaa !49
  %18 = load ptr, ptr %10, align 8, !tbaa !58, !nonnull !76, !align !78
  %19 = load i32, ptr %18, align 4, !tbaa !49
  call void @_ZNKSt8functionIFvPjS0_jjEEclES0_S0_jj(ptr noundef nonnull align 8 dereferenceable(32) %11, ptr noundef %13, ptr noundef %15, i32 noundef %17, i32 noundef %19)
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPjS0_jjEZ4mainE4$_20E9_M_invokeERKSt9_Any_dataOS0_S7_OjS8_"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %4) #15 {
  %6 = load ptr, ptr %1, align 8, !tbaa !58
  %7 = load ptr, ptr %2, align 8, !tbaa !58
  %8 = load i32, ptr %3, align 4, !tbaa !49
  %9 = zext i32 %8 to i64
  %10 = icmp eq i32 %8, 0
  br i1 %10, label %33, label %11

11:                                               ; preds = %5
  %12 = load i32, ptr %4, align 4, !tbaa !49
  %13 = zext i32 %12 to i64
  %14 = icmp eq i32 %12, 0
  %15 = add i32 %8, 1
  %16 = zext i32 %15 to i64
  br i1 %14, label %33, label %17

17:                                               ; preds = %11, %30
  %18 = phi i64 [ %31, %30 ], [ 0, %11 ]
  %19 = mul nuw i64 %18, %9
  %20 = getelementptr inbounds nuw i32, ptr %7, i64 %19
  %21 = mul nuw i64 %18, %16
  %22 = getelementptr inbounds nuw i32, ptr %6, i64 %21
  br label %23

23:                                               ; preds = %23, %17
  %24 = phi i64 [ 0, %17 ], [ %28, %23 ]
  %25 = getelementptr inbounds nuw i32, ptr %20, i64 %24
  %26 = load i32, ptr %25, align 4, !tbaa !49
  %27 = getelementptr inbounds nuw i32, ptr %22, i64 %24
  store i32 %26, ptr %27, align 4, !tbaa !49
  %28 = add nuw nsw i64 %24, 1
  %29 = icmp eq i64 %28, %13
  br i1 %29, label %30, label %23, !llvm.loop !275

30:                                               ; preds = %23
  %31 = add nuw nsw i64 %18, 1
  %32 = icmp eq i64 %31, %9
  br i1 %32, label %33, label %17, !llvm.loop !276

33:                                               ; preds = %30, %5, %11
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPjS0_jjEZ4mainE4$_20E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_20", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPjS0_jjEZ4mainE4$_21E9_M_invokeERKSt9_Any_dataOS0_S7_OjS8_"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %4) #15 {
  %6 = load ptr, ptr %1, align 8, !tbaa !58
  %7 = load ptr, ptr %2, align 8, !tbaa !58
  %8 = load i32, ptr %3, align 4, !tbaa !49
  %9 = zext i32 %8 to i64
  %10 = icmp eq i32 %8, 0
  br i1 %10, label %63, label %11

11:                                               ; preds = %5
  %12 = load i32, ptr %4, align 4, !tbaa !49
  %13 = zext i32 %12 to i64
  %14 = icmp eq i32 %12, 0
  %15 = add i32 %8, 1
  %16 = zext i32 %15 to i64
  br i1 %14, label %63, label %17

17:                                               ; preds = %11
  %18 = add nsw i64 %9, -1
  %19 = mul i64 %18, %16
  %20 = add i64 %19, %13
  %21 = shl i64 %20, 2
  %22 = getelementptr i8, ptr %6, i64 %21
  %23 = mul i64 %18, %9
  %24 = add i64 %23, %13
  %25 = shl i64 %24, 2
  %26 = getelementptr i8, ptr %7, i64 %25
  %27 = icmp ult i32 %12, 8
  %28 = icmp ult ptr %6, %26
  %29 = icmp ult ptr %7, %22
  %30 = and i1 %28, %29
  %31 = and i64 %13, 4294967288
  %32 = icmp eq i64 %31, %13
  br label %33

33:                                               ; preds = %17, %60
  %34 = phi i64 [ %61, %60 ], [ 0, %17 ]
  %35 = mul nuw i64 %34, %9
  %36 = getelementptr inbounds nuw i32, ptr %7, i64 %35
  %37 = mul nuw i64 %34, %16
  %38 = getelementptr inbounds nuw i32, ptr %6, i64 %37
  %39 = select i1 %27, i1 true, i1 %30
  br i1 %39, label %51, label %40

40:                                               ; preds = %33, %40
  %41 = phi i64 [ %48, %40 ], [ 0, %33 ]
  %42 = getelementptr inbounds nuw i32, ptr %36, i64 %41
  %43 = getelementptr inbounds nuw i8, ptr %42, i64 16
  %44 = load <4 x i32>, ptr %42, align 4, !tbaa !49, !alias.scope !277
  %45 = load <4 x i32>, ptr %43, align 4, !tbaa !49, !alias.scope !277
  %46 = getelementptr inbounds nuw i32, ptr %38, i64 %41
  %47 = getelementptr inbounds nuw i8, ptr %46, i64 16
  store <4 x i32> %44, ptr %46, align 4, !tbaa !49, !alias.scope !280, !noalias !277
  store <4 x i32> %45, ptr %47, align 4, !tbaa !49, !alias.scope !280, !noalias !277
  %48 = add nuw i64 %41, 8
  %49 = icmp eq i64 %48, %31
  br i1 %49, label %50, label %40, !llvm.loop !282

50:                                               ; preds = %40
  br i1 %32, label %60, label %51

51:                                               ; preds = %33, %50
  %52 = phi i64 [ 0, %33 ], [ %31, %50 ]
  br label %53

53:                                               ; preds = %51, %53
  %54 = phi i64 [ %58, %53 ], [ %52, %51 ]
  %55 = getelementptr inbounds nuw i32, ptr %36, i64 %54
  %56 = load i32, ptr %55, align 4, !tbaa !49
  %57 = getelementptr inbounds nuw i32, ptr %38, i64 %54
  store i32 %56, ptr %57, align 4, !tbaa !49
  %58 = add nuw nsw i64 %54, 1
  %59 = icmp eq i64 %58, %13
  br i1 %59, label %60, label %53, !llvm.loop !283

60:                                               ; preds = %53, %50
  %61 = add nuw nsw i64 %34, 1
  %62 = icmp eq i64 %61, %9
  br i1 %62, label %63, label %33, !llvm.loop !284

63:                                               ; preds = %60, %5, %11
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPjS0_jjEZ4mainE4$_21E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_21", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZNKSt8functionIFvPmS0_jjEEclES0_S0_jj(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef %1, ptr noundef %2, i32 noundef %3, i32 noundef %4) local_unnamed_addr #9 comdat {
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca i32, align 4
  %9 = alloca i32, align 4
  store ptr %1, ptr %6, align 8, !tbaa !62
  store ptr %2, ptr %7, align 8, !tbaa !62
  store i32 %3, ptr %8, align 4, !tbaa !49
  store i32 %4, ptr %9, align 4, !tbaa !49
  %10 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %11 = load ptr, ptr %10, align 8, !tbaa !20
  %12 = icmp eq ptr %11, null
  br i1 %12, label %13, label %14

13:                                               ; preds = %5
  tail call void @_ZSt25__throw_bad_function_callv() #24
  unreachable

14:                                               ; preds = %5
  %15 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %16 = load ptr, ptr %15, align 8, !tbaa !35
  call void %16(ptr noundef nonnull align 8 dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(8) %6, ptr noundef nonnull align 8 dereferenceable(8) %7, ptr noundef nonnull align 4 dereferenceable(4) %8, ptr noundef nonnull align 4 dereferenceable(4) %9)
  ret void
}

; Function Attrs: mustprogress noinline norecurse optnone uwtable
define internal fastcc void @_ZL18callThroughOptnoneIRSt8functionIFvPmS1_jjEEJS1_RS1_RKiS7_EEvOT_DpOT0_(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %1, ptr noundef nonnull align 8 dereferenceable(8) %2, ptr noundef nonnull align 4 dereferenceable(4) %3, ptr noundef nonnull align 4 dereferenceable(4) %4) unnamed_addr #10 {
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  %10 = alloca ptr, align 8
  store ptr %0, ptr %6, align 8, !tbaa !285
  store ptr %1, ptr %7, align 8, !tbaa !106
  store ptr %2, ptr %8, align 8, !tbaa !106
  store ptr %3, ptr %9, align 8, !tbaa !58
  store ptr %4, ptr %10, align 8, !tbaa !58
  %11 = load ptr, ptr %6, align 8, !tbaa !285, !nonnull !76, !align !77
  %12 = load ptr, ptr %7, align 8, !tbaa !106, !nonnull !76, !align !77
  %13 = load ptr, ptr %12, align 8, !tbaa !62
  %14 = load ptr, ptr %8, align 8, !tbaa !106, !nonnull !76, !align !77
  %15 = load ptr, ptr %14, align 8, !tbaa !62
  %16 = load ptr, ptr %9, align 8, !tbaa !58, !nonnull !76, !align !78
  %17 = load i32, ptr %16, align 4, !tbaa !49
  %18 = load ptr, ptr %10, align 8, !tbaa !58, !nonnull !76, !align !78
  %19 = load i32, ptr %18, align 4, !tbaa !49
  call void @_ZNKSt8functionIFvPmS0_jjEEclES0_S0_jj(ptr noundef nonnull align 8 dereferenceable(32) %11, ptr noundef %13, ptr noundef %15, i32 noundef %17, i32 noundef %19)
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPmS0_jjEZ4mainE4$_20E9_M_invokeERKSt9_Any_dataOS0_S7_OjS8_"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %4) #15 {
  %6 = load ptr, ptr %1, align 8, !tbaa !62
  %7 = load ptr, ptr %2, align 8, !tbaa !62
  %8 = load i32, ptr %3, align 4, !tbaa !49
  %9 = zext i32 %8 to i64
  %10 = icmp eq i32 %8, 0
  br i1 %10, label %33, label %11

11:                                               ; preds = %5
  %12 = load i32, ptr %4, align 4, !tbaa !49
  %13 = zext i32 %12 to i64
  %14 = icmp eq i32 %12, 0
  %15 = add i32 %8, 1
  %16 = zext i32 %15 to i64
  br i1 %14, label %33, label %17

17:                                               ; preds = %11, %30
  %18 = phi i64 [ %31, %30 ], [ 0, %11 ]
  %19 = mul nuw i64 %18, %9
  %20 = getelementptr inbounds nuw i64, ptr %7, i64 %19
  %21 = mul nuw i64 %18, %16
  %22 = getelementptr inbounds nuw i64, ptr %6, i64 %21
  br label %23

23:                                               ; preds = %23, %17
  %24 = phi i64 [ 0, %17 ], [ %28, %23 ]
  %25 = getelementptr inbounds nuw i64, ptr %20, i64 %24
  %26 = load i64, ptr %25, align 8, !tbaa !6
  %27 = getelementptr inbounds nuw i64, ptr %22, i64 %24
  store i64 %26, ptr %27, align 8, !tbaa !6
  %28 = add nuw nsw i64 %24, 1
  %29 = icmp eq i64 %28, %13
  br i1 %29, label %30, label %23, !llvm.loop !287

30:                                               ; preds = %23
  %31 = add nuw nsw i64 %18, 1
  %32 = icmp eq i64 %31, %9
  br i1 %32, label %33, label %17, !llvm.loop !288

33:                                               ; preds = %30, %5, %11
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPmS0_jjEZ4mainE4$_20E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_20", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPmS0_jjEZ4mainE4$_21E9_M_invokeERKSt9_Any_dataOS0_S7_OjS8_"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %4) #15 {
  %6 = load ptr, ptr %1, align 8, !tbaa !62
  %7 = load ptr, ptr %2, align 8, !tbaa !62
  %8 = load i32, ptr %3, align 4, !tbaa !49
  %9 = zext i32 %8 to i64
  %10 = icmp eq i32 %8, 0
  br i1 %10, label %63, label %11

11:                                               ; preds = %5
  %12 = load i32, ptr %4, align 4, !tbaa !49
  %13 = zext i32 %12 to i64
  %14 = icmp eq i32 %12, 0
  %15 = add i32 %8, 1
  %16 = zext i32 %15 to i64
  br i1 %14, label %63, label %17

17:                                               ; preds = %11
  %18 = add nsw i64 %9, -1
  %19 = mul i64 %18, %16
  %20 = add i64 %19, %13
  %21 = shl i64 %20, 3
  %22 = getelementptr i8, ptr %6, i64 %21
  %23 = mul i64 %18, %9
  %24 = add i64 %23, %13
  %25 = shl i64 %24, 3
  %26 = getelementptr i8, ptr %7, i64 %25
  %27 = icmp ult i32 %12, 4
  %28 = icmp ult ptr %6, %26
  %29 = icmp ult ptr %7, %22
  %30 = and i1 %28, %29
  %31 = and i64 %13, 4294967292
  %32 = icmp eq i64 %31, %13
  br label %33

33:                                               ; preds = %17, %60
  %34 = phi i64 [ %61, %60 ], [ 0, %17 ]
  %35 = mul nuw i64 %34, %9
  %36 = getelementptr inbounds nuw i64, ptr %7, i64 %35
  %37 = mul nuw i64 %34, %16
  %38 = getelementptr inbounds nuw i64, ptr %6, i64 %37
  %39 = select i1 %27, i1 true, i1 %30
  br i1 %39, label %51, label %40

40:                                               ; preds = %33, %40
  %41 = phi i64 [ %48, %40 ], [ 0, %33 ]
  %42 = getelementptr inbounds nuw i64, ptr %36, i64 %41
  %43 = getelementptr inbounds nuw i8, ptr %42, i64 16
  %44 = load <2 x i64>, ptr %42, align 8, !tbaa !6, !alias.scope !289
  %45 = load <2 x i64>, ptr %43, align 8, !tbaa !6, !alias.scope !289
  %46 = getelementptr inbounds nuw i64, ptr %38, i64 %41
  %47 = getelementptr inbounds nuw i8, ptr %46, i64 16
  store <2 x i64> %44, ptr %46, align 8, !tbaa !6, !alias.scope !292, !noalias !289
  store <2 x i64> %45, ptr %47, align 8, !tbaa !6, !alias.scope !292, !noalias !289
  %48 = add nuw i64 %41, 4
  %49 = icmp eq i64 %48, %31
  br i1 %49, label %50, label %40, !llvm.loop !294

50:                                               ; preds = %40
  br i1 %32, label %60, label %51

51:                                               ; preds = %33, %50
  %52 = phi i64 [ 0, %33 ], [ %31, %50 ]
  br label %53

53:                                               ; preds = %51, %53
  %54 = phi i64 [ %58, %53 ], [ %52, %51 ]
  %55 = getelementptr inbounds nuw i64, ptr %36, i64 %54
  %56 = load i64, ptr %55, align 8, !tbaa !6
  %57 = getelementptr inbounds nuw i64, ptr %38, i64 %54
  store i64 %56, ptr %57, align 8, !tbaa !6
  %58 = add nuw nsw i64 %54, 1
  %59 = icmp eq i64 %58, %13
  br i1 %59, label %60, label %53, !llvm.loop !295

60:                                               ; preds = %53, %50
  %61 = add nuw nsw i64 %34, 1
  %62 = icmp eq i64 %61, %9
  br i1 %62, label %63, label %33, !llvm.loop !296

63:                                               ; preds = %60, %5, %11
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPmS0_jjEZ4mainE4$_21E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_21", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPhS0_jjEZ4mainE4$_22E9_M_invokeERKSt9_Any_dataOS0_S7_OjS8_"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %4) #15 {
  %6 = load ptr, ptr %1, align 8, !tbaa !54
  %7 = load ptr, ptr %2, align 8, !tbaa !54
  %8 = load i32, ptr %3, align 4, !tbaa !49
  %9 = zext i32 %8 to i64
  %10 = icmp eq i32 %8, 0
  br i1 %10, label %35, label %11

11:                                               ; preds = %5
  %12 = load i32, ptr %4, align 4, !tbaa !49
  %13 = zext i32 %12 to i64
  %14 = icmp eq i32 %12, 0
  %15 = add i32 %8, 1
  %16 = zext i32 %15 to i64
  br i1 %14, label %35, label %17

17:                                               ; preds = %11, %32
  %18 = phi i64 [ %33, %32 ], [ 0, %11 ]
  %19 = mul nuw i64 %18, %9
  %20 = getelementptr inbounds nuw i8, ptr %7, i64 %19
  %21 = mul nuw i64 %18, %16
  %22 = getelementptr inbounds nuw i8, ptr %6, i64 %21
  br label %23

23:                                               ; preds = %23, %17
  %24 = phi i64 [ 0, %17 ], [ %30, %23 ]
  %25 = getelementptr inbounds nuw i8, ptr %20, i64 %24
  %26 = load i8, ptr %25, align 1, !tbaa !15
  %27 = getelementptr inbounds nuw i8, ptr %22, i64 %24
  %28 = load i8, ptr %27, align 1, !tbaa !15
  %29 = add i8 %28, %26
  store i8 %29, ptr %27, align 1, !tbaa !15
  %30 = add nuw nsw i64 %24, 1
  %31 = icmp eq i64 %30, %13
  br i1 %31, label %32, label %23, !llvm.loop !297

32:                                               ; preds = %23
  %33 = add nuw nsw i64 %18, 1
  %34 = icmp eq i64 %33, %9
  br i1 %34, label %35, label %17, !llvm.loop !298

35:                                               ; preds = %32, %5, %11
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPhS0_jjEZ4mainE4$_22E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_22", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPhS0_jjEZ4mainE4$_23E9_M_invokeERKSt9_Any_dataOS0_S7_OjS8_"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %4) #15 {
  %6 = load ptr, ptr %1, align 8, !tbaa !54
  %7 = load ptr, ptr %2, align 8, !tbaa !54
  %8 = load i32, ptr %3, align 4, !tbaa !49
  %9 = zext i32 %8 to i64
  %10 = icmp eq i32 %8, 0
  br i1 %10, label %86, label %11

11:                                               ; preds = %5
  %12 = load i32, ptr %4, align 4, !tbaa !49
  %13 = zext i32 %12 to i64
  %14 = icmp eq i32 %12, 0
  %15 = add i32 %8, 1
  %16 = zext i32 %15 to i64
  br i1 %14, label %86, label %17

17:                                               ; preds = %11
  %18 = add nsw i64 %9, -1
  %19 = mul i64 %18, %16
  %20 = getelementptr i8, ptr %6, i64 %19
  %21 = getelementptr i8, ptr %20, i64 %13
  %22 = mul i64 %18, %9
  %23 = getelementptr i8, ptr %7, i64 %22
  %24 = getelementptr i8, ptr %23, i64 %13
  %25 = icmp ult i32 %12, 8
  %26 = icmp ult ptr %6, %24
  %27 = icmp ult ptr %7, %21
  %28 = and i1 %26, %27
  %29 = icmp ult i32 %12, 32
  %30 = and i64 %13, 4294967264
  %31 = icmp eq i64 %30, %13
  %32 = and i64 %13, 24
  %33 = icmp eq i64 %32, 0
  %34 = and i64 %13, 4294967288
  %35 = icmp eq i64 %34, %13
  br label %36

36:                                               ; preds = %17, %83
  %37 = phi i64 [ %84, %83 ], [ 0, %17 ]
  %38 = mul nuw i64 %37, %9
  %39 = getelementptr inbounds nuw i8, ptr %7, i64 %38
  %40 = mul nuw i64 %37, %16
  %41 = getelementptr inbounds nuw i8, ptr %6, i64 %40
  %42 = select i1 %25, i1 true, i1 %28
  br i1 %42, label %72, label %43

43:                                               ; preds = %36
  br i1 %29, label %60, label %44

44:                                               ; preds = %43, %44
  %45 = phi i64 [ %56, %44 ], [ 0, %43 ]
  %46 = getelementptr inbounds nuw i8, ptr %39, i64 %45
  %47 = getelementptr inbounds nuw i8, ptr %46, i64 16
  %48 = load <16 x i8>, ptr %46, align 1, !tbaa !15, !alias.scope !299
  %49 = load <16 x i8>, ptr %47, align 1, !tbaa !15, !alias.scope !299
  %50 = getelementptr inbounds nuw i8, ptr %41, i64 %45
  %51 = getelementptr inbounds nuw i8, ptr %50, i64 16
  %52 = load <16 x i8>, ptr %50, align 1, !tbaa !15, !alias.scope !302, !noalias !299
  %53 = load <16 x i8>, ptr %51, align 1, !tbaa !15, !alias.scope !302, !noalias !299
  %54 = add <16 x i8> %52, %48
  %55 = add <16 x i8> %53, %49
  store <16 x i8> %54, ptr %50, align 1, !tbaa !15, !alias.scope !302, !noalias !299
  store <16 x i8> %55, ptr %51, align 1, !tbaa !15, !alias.scope !302, !noalias !299
  %56 = add nuw i64 %45, 32
  %57 = icmp eq i64 %56, %30
  br i1 %57, label %58, label %44, !llvm.loop !304

58:                                               ; preds = %44
  br i1 %31, label %83, label %59

59:                                               ; preds = %58
  br i1 %33, label %72, label %60

60:                                               ; preds = %59, %43
  %61 = phi i64 [ %30, %59 ], [ 0, %43 ]
  br label %62

62:                                               ; preds = %62, %60
  %63 = phi i64 [ %61, %60 ], [ %69, %62 ]
  %64 = getelementptr inbounds nuw i8, ptr %39, i64 %63
  %65 = load <8 x i8>, ptr %64, align 1, !tbaa !15, !alias.scope !299
  %66 = getelementptr inbounds nuw i8, ptr %41, i64 %63
  %67 = load <8 x i8>, ptr %66, align 1, !tbaa !15, !alias.scope !302, !noalias !299
  %68 = add <8 x i8> %67, %65
  store <8 x i8> %68, ptr %66, align 1, !tbaa !15, !alias.scope !302, !noalias !299
  %69 = add nuw i64 %63, 8
  %70 = icmp eq i64 %69, %34
  br i1 %70, label %71, label %62, !llvm.loop !305

71:                                               ; preds = %62
  br i1 %35, label %83, label %72

72:                                               ; preds = %36, %59, %71
  %73 = phi i64 [ 0, %36 ], [ %30, %59 ], [ %34, %71 ]
  br label %74

74:                                               ; preds = %72, %74
  %75 = phi i64 [ %81, %74 ], [ %73, %72 ]
  %76 = getelementptr inbounds nuw i8, ptr %39, i64 %75
  %77 = load i8, ptr %76, align 1, !tbaa !15
  %78 = getelementptr inbounds nuw i8, ptr %41, i64 %75
  %79 = load i8, ptr %78, align 1, !tbaa !15
  %80 = add i8 %79, %77
  store i8 %80, ptr %78, align 1, !tbaa !15
  %81 = add nuw nsw i64 %75, 1
  %82 = icmp eq i64 %81, %13
  br i1 %82, label %83, label %74, !llvm.loop !306

83:                                               ; preds = %74, %71, %58
  %84 = add nuw nsw i64 %37, 1
  %85 = icmp eq i64 %84, %9
  br i1 %85, label %86, label %36, !llvm.loop !307

86:                                               ; preds = %83, %5, %11
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPhS0_jjEZ4mainE4$_23E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_23", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPjS0_jjEZ4mainE4$_22E9_M_invokeERKSt9_Any_dataOS0_S7_OjS8_"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %4) #15 {
  %6 = load ptr, ptr %1, align 8, !tbaa !58
  %7 = load ptr, ptr %2, align 8, !tbaa !58
  %8 = load i32, ptr %3, align 4, !tbaa !49
  %9 = zext i32 %8 to i64
  %10 = icmp eq i32 %8, 0
  br i1 %10, label %35, label %11

11:                                               ; preds = %5
  %12 = load i32, ptr %4, align 4, !tbaa !49
  %13 = zext i32 %12 to i64
  %14 = icmp eq i32 %12, 0
  %15 = add i32 %8, 1
  %16 = zext i32 %15 to i64
  br i1 %14, label %35, label %17

17:                                               ; preds = %11, %32
  %18 = phi i64 [ %33, %32 ], [ 0, %11 ]
  %19 = mul nuw i64 %18, %9
  %20 = getelementptr inbounds nuw i32, ptr %7, i64 %19
  %21 = mul nuw i64 %18, %16
  %22 = getelementptr inbounds nuw i32, ptr %6, i64 %21
  br label %23

23:                                               ; preds = %23, %17
  %24 = phi i64 [ 0, %17 ], [ %30, %23 ]
  %25 = getelementptr inbounds nuw i32, ptr %20, i64 %24
  %26 = load i32, ptr %25, align 4, !tbaa !49
  %27 = getelementptr inbounds nuw i32, ptr %22, i64 %24
  %28 = load i32, ptr %27, align 4, !tbaa !49
  %29 = add i32 %28, %26
  store i32 %29, ptr %27, align 4, !tbaa !49
  %30 = add nuw nsw i64 %24, 1
  %31 = icmp eq i64 %30, %13
  br i1 %31, label %32, label %23, !llvm.loop !308

32:                                               ; preds = %23
  %33 = add nuw nsw i64 %18, 1
  %34 = icmp eq i64 %33, %9
  br i1 %34, label %35, label %17, !llvm.loop !309

35:                                               ; preds = %32, %5, %11
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPjS0_jjEZ4mainE4$_22E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_22", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPjS0_jjEZ4mainE4$_23E9_M_invokeERKSt9_Any_dataOS0_S7_OjS8_"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %4) #15 {
  %6 = load ptr, ptr %1, align 8, !tbaa !58
  %7 = load ptr, ptr %2, align 8, !tbaa !58
  %8 = load i32, ptr %3, align 4, !tbaa !49
  %9 = zext i32 %8 to i64
  %10 = icmp eq i32 %8, 0
  br i1 %10, label %69, label %11

11:                                               ; preds = %5
  %12 = load i32, ptr %4, align 4, !tbaa !49
  %13 = zext i32 %12 to i64
  %14 = icmp eq i32 %12, 0
  %15 = add i32 %8, 1
  %16 = zext i32 %15 to i64
  br i1 %14, label %69, label %17

17:                                               ; preds = %11
  %18 = add nsw i64 %9, -1
  %19 = mul i64 %18, %16
  %20 = add i64 %19, %13
  %21 = shl i64 %20, 2
  %22 = getelementptr i8, ptr %6, i64 %21
  %23 = mul i64 %18, %9
  %24 = add i64 %23, %13
  %25 = shl i64 %24, 2
  %26 = getelementptr i8, ptr %7, i64 %25
  %27 = icmp ult i32 %12, 8
  %28 = icmp ult ptr %6, %26
  %29 = icmp ult ptr %7, %22
  %30 = and i1 %28, %29
  %31 = and i64 %13, 4294967288
  %32 = icmp eq i64 %31, %13
  br label %33

33:                                               ; preds = %17, %66
  %34 = phi i64 [ %67, %66 ], [ 0, %17 ]
  %35 = mul nuw i64 %34, %9
  %36 = getelementptr inbounds nuw i32, ptr %7, i64 %35
  %37 = mul nuw i64 %34, %16
  %38 = getelementptr inbounds nuw i32, ptr %6, i64 %37
  %39 = select i1 %27, i1 true, i1 %30
  br i1 %39, label %55, label %40

40:                                               ; preds = %33, %40
  %41 = phi i64 [ %52, %40 ], [ 0, %33 ]
  %42 = getelementptr inbounds nuw i32, ptr %36, i64 %41
  %43 = getelementptr inbounds nuw i8, ptr %42, i64 16
  %44 = load <4 x i32>, ptr %42, align 4, !tbaa !49, !alias.scope !310
  %45 = load <4 x i32>, ptr %43, align 4, !tbaa !49, !alias.scope !310
  %46 = getelementptr inbounds nuw i32, ptr %38, i64 %41
  %47 = getelementptr inbounds nuw i8, ptr %46, i64 16
  %48 = load <4 x i32>, ptr %46, align 4, !tbaa !49, !alias.scope !313, !noalias !310
  %49 = load <4 x i32>, ptr %47, align 4, !tbaa !49, !alias.scope !313, !noalias !310
  %50 = add <4 x i32> %48, %44
  %51 = add <4 x i32> %49, %45
  store <4 x i32> %50, ptr %46, align 4, !tbaa !49, !alias.scope !313, !noalias !310
  store <4 x i32> %51, ptr %47, align 4, !tbaa !49, !alias.scope !313, !noalias !310
  %52 = add nuw i64 %41, 8
  %53 = icmp eq i64 %52, %31
  br i1 %53, label %54, label %40, !llvm.loop !315

54:                                               ; preds = %40
  br i1 %32, label %66, label %55

55:                                               ; preds = %33, %54
  %56 = phi i64 [ 0, %33 ], [ %31, %54 ]
  br label %57

57:                                               ; preds = %55, %57
  %58 = phi i64 [ %64, %57 ], [ %56, %55 ]
  %59 = getelementptr inbounds nuw i32, ptr %36, i64 %58
  %60 = load i32, ptr %59, align 4, !tbaa !49
  %61 = getelementptr inbounds nuw i32, ptr %38, i64 %58
  %62 = load i32, ptr %61, align 4, !tbaa !49
  %63 = add i32 %62, %60
  store i32 %63, ptr %61, align 4, !tbaa !49
  %64 = add nuw nsw i64 %58, 1
  %65 = icmp eq i64 %64, %13
  br i1 %65, label %66, label %57, !llvm.loop !316

66:                                               ; preds = %57, %54
  %67 = add nuw nsw i64 %34, 1
  %68 = icmp eq i64 %67, %9
  br i1 %68, label %69, label %33, !llvm.loop !317

69:                                               ; preds = %66, %5, %11
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPjS0_jjEZ4mainE4$_23E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_23", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPmS0_jjEZ4mainE4$_22E9_M_invokeERKSt9_Any_dataOS0_S7_OjS8_"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %4) #15 {
  %6 = load ptr, ptr %1, align 8, !tbaa !62
  %7 = load ptr, ptr %2, align 8, !tbaa !62
  %8 = load i32, ptr %3, align 4, !tbaa !49
  %9 = zext i32 %8 to i64
  %10 = icmp eq i32 %8, 0
  br i1 %10, label %35, label %11

11:                                               ; preds = %5
  %12 = load i32, ptr %4, align 4, !tbaa !49
  %13 = zext i32 %12 to i64
  %14 = icmp eq i32 %12, 0
  %15 = add i32 %8, 1
  %16 = zext i32 %15 to i64
  br i1 %14, label %35, label %17

17:                                               ; preds = %11, %32
  %18 = phi i64 [ %33, %32 ], [ 0, %11 ]
  %19 = mul nuw i64 %18, %9
  %20 = getelementptr inbounds nuw i64, ptr %7, i64 %19
  %21 = mul nuw i64 %18, %16
  %22 = getelementptr inbounds nuw i64, ptr %6, i64 %21
  br label %23

23:                                               ; preds = %23, %17
  %24 = phi i64 [ 0, %17 ], [ %30, %23 ]
  %25 = getelementptr inbounds nuw i64, ptr %20, i64 %24
  %26 = load i64, ptr %25, align 8, !tbaa !6
  %27 = getelementptr inbounds nuw i64, ptr %22, i64 %24
  %28 = load i64, ptr %27, align 8, !tbaa !6
  %29 = add i64 %28, %26
  store i64 %29, ptr %27, align 8, !tbaa !6
  %30 = add nuw nsw i64 %24, 1
  %31 = icmp eq i64 %30, %13
  br i1 %31, label %32, label %23, !llvm.loop !318

32:                                               ; preds = %23
  %33 = add nuw nsw i64 %18, 1
  %34 = icmp eq i64 %33, %9
  br i1 %34, label %35, label %17, !llvm.loop !319

35:                                               ; preds = %32, %5, %11
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPmS0_jjEZ4mainE4$_22E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_22", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPmS0_jjEZ4mainE4$_23E9_M_invokeERKSt9_Any_dataOS0_S7_OjS8_"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %4) #15 {
  %6 = load ptr, ptr %1, align 8, !tbaa !62
  %7 = load ptr, ptr %2, align 8, !tbaa !62
  %8 = load i32, ptr %3, align 4, !tbaa !49
  %9 = zext i32 %8 to i64
  %10 = icmp eq i32 %8, 0
  br i1 %10, label %69, label %11

11:                                               ; preds = %5
  %12 = load i32, ptr %4, align 4, !tbaa !49
  %13 = zext i32 %12 to i64
  %14 = icmp eq i32 %12, 0
  %15 = add i32 %8, 1
  %16 = zext i32 %15 to i64
  br i1 %14, label %69, label %17

17:                                               ; preds = %11
  %18 = add nsw i64 %9, -1
  %19 = mul i64 %18, %16
  %20 = add i64 %19, %13
  %21 = shl i64 %20, 3
  %22 = getelementptr i8, ptr %6, i64 %21
  %23 = mul i64 %18, %9
  %24 = add i64 %23, %13
  %25 = shl i64 %24, 3
  %26 = getelementptr i8, ptr %7, i64 %25
  %27 = icmp ult i32 %12, 4
  %28 = icmp ult ptr %6, %26
  %29 = icmp ult ptr %7, %22
  %30 = and i1 %28, %29
  %31 = and i64 %13, 4294967292
  %32 = icmp eq i64 %31, %13
  br label %33

33:                                               ; preds = %17, %66
  %34 = phi i64 [ %67, %66 ], [ 0, %17 ]
  %35 = mul nuw i64 %34, %9
  %36 = getelementptr inbounds nuw i64, ptr %7, i64 %35
  %37 = mul nuw i64 %34, %16
  %38 = getelementptr inbounds nuw i64, ptr %6, i64 %37
  %39 = select i1 %27, i1 true, i1 %30
  br i1 %39, label %55, label %40

40:                                               ; preds = %33, %40
  %41 = phi i64 [ %52, %40 ], [ 0, %33 ]
  %42 = getelementptr inbounds nuw i64, ptr %36, i64 %41
  %43 = getelementptr inbounds nuw i8, ptr %42, i64 16
  %44 = load <2 x i64>, ptr %42, align 8, !tbaa !6, !alias.scope !320
  %45 = load <2 x i64>, ptr %43, align 8, !tbaa !6, !alias.scope !320
  %46 = getelementptr inbounds nuw i64, ptr %38, i64 %41
  %47 = getelementptr inbounds nuw i8, ptr %46, i64 16
  %48 = load <2 x i64>, ptr %46, align 8, !tbaa !6, !alias.scope !323, !noalias !320
  %49 = load <2 x i64>, ptr %47, align 8, !tbaa !6, !alias.scope !323, !noalias !320
  %50 = add <2 x i64> %48, %44
  %51 = add <2 x i64> %49, %45
  store <2 x i64> %50, ptr %46, align 8, !tbaa !6, !alias.scope !323, !noalias !320
  store <2 x i64> %51, ptr %47, align 8, !tbaa !6, !alias.scope !323, !noalias !320
  %52 = add nuw i64 %41, 4
  %53 = icmp eq i64 %52, %31
  br i1 %53, label %54, label %40, !llvm.loop !325

54:                                               ; preds = %40
  br i1 %32, label %66, label %55

55:                                               ; preds = %33, %54
  %56 = phi i64 [ 0, %33 ], [ %31, %54 ]
  br label %57

57:                                               ; preds = %55, %57
  %58 = phi i64 [ %64, %57 ], [ %56, %55 ]
  %59 = getelementptr inbounds nuw i64, ptr %36, i64 %58
  %60 = load i64, ptr %59, align 8, !tbaa !6
  %61 = getelementptr inbounds nuw i64, ptr %38, i64 %58
  %62 = load i64, ptr %61, align 8, !tbaa !6
  %63 = add i64 %62, %60
  store i64 %63, ptr %61, align 8, !tbaa !6
  %64 = add nuw nsw i64 %58, 1
  %65 = icmp eq i64 %64, %13
  br i1 %65, label %66, label %57, !llvm.loop !326

66:                                               ; preds = %57, %54
  %67 = add nuw nsw i64 %34, 1
  %68 = icmp eq i64 %67, %9
  br i1 %68, label %69, label %33, !llvm.loop !327

69:                                               ; preds = %66, %5, %11
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPmS0_jjEZ4mainE4$_23E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_23", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPhS0_jjEZ4mainE4$_24E9_M_invokeERKSt9_Any_dataOS0_S7_OjS8_"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %4) #15 {
  %6 = load ptr, ptr %1, align 8, !tbaa !54
  %7 = load ptr, ptr %2, align 8, !tbaa !54
  %8 = load i32, ptr %3, align 4, !tbaa !49
  %9 = load i32, ptr %4, align 4, !tbaa !49
  %10 = zext i32 %9 to i64
  %11 = icmp eq i32 %9, 0
  %12 = zext i32 %8 to i64
  %13 = add i32 %8, 1
  %14 = zext i32 %13 to i64
  br i1 %11, label %34, label %15

15:                                               ; preds = %5
  %16 = add i32 %8, -1
  %17 = zext i32 %16 to i64
  br label %18

18:                                               ; preds = %31, %15
  %19 = phi i64 [ %32, %31 ], [ %17, %15 ]
  %20 = mul nuw nsw i64 %19, %12
  %21 = getelementptr inbounds nuw i8, ptr %7, i64 %20
  %22 = mul nuw nsw i64 %19, %14
  %23 = getelementptr inbounds nuw i8, ptr %6, i64 %22
  br label %24

24:                                               ; preds = %24, %18
  %25 = phi i64 [ 0, %18 ], [ %29, %24 ]
  %26 = getelementptr inbounds nuw i8, ptr %21, i64 %25
  %27 = load i8, ptr %26, align 1, !tbaa !15
  %28 = getelementptr inbounds nuw i8, ptr %23, i64 %25
  store i8 %27, ptr %28, align 1, !tbaa !15
  %29 = add nuw nsw i64 %25, 1
  %30 = icmp eq i64 %29, %10
  br i1 %30, label %31, label %24, !llvm.loop !328

31:                                               ; preds = %24
  %32 = add nsw i64 %19, -1
  %33 = icmp sgt i64 %19, 0
  br i1 %33, label %18, label %34, !llvm.loop !329

34:                                               ; preds = %31, %5
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPhS0_jjEZ4mainE4$_24E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_24", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPhS0_jjEZ4mainE4$_25E9_M_invokeERKSt9_Any_dataOS0_S7_OjS8_"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %4) #15 {
  %6 = load ptr, ptr %1, align 8, !tbaa !54
  %7 = load ptr, ptr %2, align 8, !tbaa !54
  %8 = load i32, ptr %3, align 4, !tbaa !49
  %9 = load i32, ptr %4, align 4, !tbaa !49
  %10 = zext i32 %9 to i64
  %11 = icmp eq i32 %9, 0
  %12 = zext i32 %8 to i64
  %13 = add i32 %8, 1
  %14 = zext i32 %13 to i64
  br i1 %11, label %34, label %15

15:                                               ; preds = %5
  %16 = add i32 %8, -1
  %17 = zext i32 %16 to i64
  br label %18

18:                                               ; preds = %31, %15
  %19 = phi i64 [ %32, %31 ], [ %17, %15 ]
  %20 = mul nuw nsw i64 %19, %12
  %21 = getelementptr inbounds nuw i8, ptr %7, i64 %20
  %22 = mul nuw nsw i64 %19, %14
  %23 = getelementptr inbounds nuw i8, ptr %6, i64 %22
  br label %24

24:                                               ; preds = %18, %24
  %25 = phi i64 [ 0, %18 ], [ %29, %24 ]
  %26 = getelementptr inbounds nuw i8, ptr %21, i64 %25
  %27 = load i8, ptr %26, align 1, !tbaa !15
  %28 = getelementptr inbounds nuw i8, ptr %23, i64 %25
  store i8 %27, ptr %28, align 1, !tbaa !15
  %29 = add nuw nsw i64 %25, 1
  %30 = icmp eq i64 %29, %10
  br i1 %30, label %31, label %24, !llvm.loop !330

31:                                               ; preds = %24
  %32 = add nsw i64 %19, -1
  %33 = icmp sgt i64 %19, 0
  br i1 %33, label %18, label %34, !llvm.loop !331

34:                                               ; preds = %31, %5
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPhS0_jjEZ4mainE4$_25E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_25", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPjS0_jjEZ4mainE4$_24E9_M_invokeERKSt9_Any_dataOS0_S7_OjS8_"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %4) #15 {
  %6 = load ptr, ptr %1, align 8, !tbaa !58
  %7 = load ptr, ptr %2, align 8, !tbaa !58
  %8 = load i32, ptr %3, align 4, !tbaa !49
  %9 = load i32, ptr %4, align 4, !tbaa !49
  %10 = zext i32 %9 to i64
  %11 = icmp eq i32 %9, 0
  %12 = zext i32 %8 to i64
  %13 = add i32 %8, 1
  %14 = zext i32 %13 to i64
  br i1 %11, label %34, label %15

15:                                               ; preds = %5
  %16 = add i32 %8, -1
  %17 = zext i32 %16 to i64
  br label %18

18:                                               ; preds = %31, %15
  %19 = phi i64 [ %32, %31 ], [ %17, %15 ]
  %20 = mul nuw nsw i64 %19, %12
  %21 = getelementptr inbounds nuw i32, ptr %7, i64 %20
  %22 = mul nuw nsw i64 %19, %14
  %23 = getelementptr inbounds nuw i32, ptr %6, i64 %22
  br label %24

24:                                               ; preds = %24, %18
  %25 = phi i64 [ 0, %18 ], [ %29, %24 ]
  %26 = getelementptr inbounds nuw i32, ptr %21, i64 %25
  %27 = load i32, ptr %26, align 4, !tbaa !49
  %28 = getelementptr inbounds nuw i32, ptr %23, i64 %25
  store i32 %27, ptr %28, align 4, !tbaa !49
  %29 = add nuw nsw i64 %25, 1
  %30 = icmp eq i64 %29, %10
  br i1 %30, label %31, label %24, !llvm.loop !332

31:                                               ; preds = %24
  %32 = add nsw i64 %19, -1
  %33 = icmp sgt i64 %19, 0
  br i1 %33, label %18, label %34, !llvm.loop !333

34:                                               ; preds = %31, %5
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPjS0_jjEZ4mainE4$_24E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_24", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPjS0_jjEZ4mainE4$_25E9_M_invokeERKSt9_Any_dataOS0_S7_OjS8_"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %4) #15 {
  %6 = load ptr, ptr %1, align 8, !tbaa !58
  %7 = load ptr, ptr %2, align 8, !tbaa !58
  %8 = load i32, ptr %3, align 4, !tbaa !49
  %9 = load i32, ptr %4, align 4, !tbaa !49
  %10 = zext i32 %9 to i64
  %11 = icmp eq i32 %9, 0
  %12 = zext i32 %8 to i64
  %13 = add i32 %8, 1
  %14 = zext i32 %13 to i64
  br i1 %11, label %34, label %15

15:                                               ; preds = %5
  %16 = add i32 %8, -1
  %17 = zext i32 %16 to i64
  br label %18

18:                                               ; preds = %31, %15
  %19 = phi i64 [ %32, %31 ], [ %17, %15 ]
  %20 = mul nuw nsw i64 %19, %12
  %21 = getelementptr inbounds nuw i32, ptr %7, i64 %20
  %22 = mul nuw nsw i64 %19, %14
  %23 = getelementptr inbounds nuw i32, ptr %6, i64 %22
  br label %24

24:                                               ; preds = %18, %24
  %25 = phi i64 [ 0, %18 ], [ %29, %24 ]
  %26 = getelementptr inbounds nuw i32, ptr %21, i64 %25
  %27 = load i32, ptr %26, align 4, !tbaa !49
  %28 = getelementptr inbounds nuw i32, ptr %23, i64 %25
  store i32 %27, ptr %28, align 4, !tbaa !49
  %29 = add nuw nsw i64 %25, 1
  %30 = icmp eq i64 %29, %10
  br i1 %30, label %31, label %24, !llvm.loop !334

31:                                               ; preds = %24
  %32 = add nsw i64 %19, -1
  %33 = icmp sgt i64 %19, 0
  br i1 %33, label %18, label %34, !llvm.loop !335

34:                                               ; preds = %31, %5
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPjS0_jjEZ4mainE4$_25E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_25", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPmS0_jjEZ4mainE4$_24E9_M_invokeERKSt9_Any_dataOS0_S7_OjS8_"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %4) #15 {
  %6 = load ptr, ptr %1, align 8, !tbaa !62
  %7 = load ptr, ptr %2, align 8, !tbaa !62
  %8 = load i32, ptr %3, align 4, !tbaa !49
  %9 = load i32, ptr %4, align 4, !tbaa !49
  %10 = zext i32 %9 to i64
  %11 = icmp eq i32 %9, 0
  %12 = zext i32 %8 to i64
  %13 = add i32 %8, 1
  %14 = zext i32 %13 to i64
  br i1 %11, label %34, label %15

15:                                               ; preds = %5
  %16 = add i32 %8, -1
  %17 = zext i32 %16 to i64
  br label %18

18:                                               ; preds = %31, %15
  %19 = phi i64 [ %32, %31 ], [ %17, %15 ]
  %20 = mul nuw nsw i64 %19, %12
  %21 = getelementptr inbounds nuw i64, ptr %7, i64 %20
  %22 = mul nuw nsw i64 %19, %14
  %23 = getelementptr inbounds nuw i64, ptr %6, i64 %22
  br label %24

24:                                               ; preds = %24, %18
  %25 = phi i64 [ 0, %18 ], [ %29, %24 ]
  %26 = getelementptr inbounds nuw i64, ptr %21, i64 %25
  %27 = load i64, ptr %26, align 8, !tbaa !6
  %28 = getelementptr inbounds nuw i64, ptr %23, i64 %25
  store i64 %27, ptr %28, align 8, !tbaa !6
  %29 = add nuw nsw i64 %25, 1
  %30 = icmp eq i64 %29, %10
  br i1 %30, label %31, label %24, !llvm.loop !336

31:                                               ; preds = %24
  %32 = add nsw i64 %19, -1
  %33 = icmp sgt i64 %19, 0
  br i1 %33, label %18, label %34, !llvm.loop !337

34:                                               ; preds = %31, %5
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPmS0_jjEZ4mainE4$_24E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_24", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPmS0_jjEZ4mainE4$_25E9_M_invokeERKSt9_Any_dataOS0_S7_OjS8_"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %4) #15 {
  %6 = load ptr, ptr %1, align 8, !tbaa !62
  %7 = load ptr, ptr %2, align 8, !tbaa !62
  %8 = load i32, ptr %3, align 4, !tbaa !49
  %9 = load i32, ptr %4, align 4, !tbaa !49
  %10 = zext i32 %9 to i64
  %11 = icmp eq i32 %9, 0
  %12 = zext i32 %8 to i64
  %13 = add i32 %8, 1
  %14 = zext i32 %13 to i64
  br i1 %11, label %34, label %15

15:                                               ; preds = %5
  %16 = add i32 %8, -1
  %17 = zext i32 %16 to i64
  br label %18

18:                                               ; preds = %31, %15
  %19 = phi i64 [ %32, %31 ], [ %17, %15 ]
  %20 = mul nuw nsw i64 %19, %12
  %21 = getelementptr inbounds nuw i64, ptr %7, i64 %20
  %22 = mul nuw nsw i64 %19, %14
  %23 = getelementptr inbounds nuw i64, ptr %6, i64 %22
  br label %24

24:                                               ; preds = %18, %24
  %25 = phi i64 [ 0, %18 ], [ %29, %24 ]
  %26 = getelementptr inbounds nuw i64, ptr %21, i64 %25
  %27 = load i64, ptr %26, align 8, !tbaa !6
  %28 = getelementptr inbounds nuw i64, ptr %23, i64 %25
  store i64 %27, ptr %28, align 8, !tbaa !6
  %29 = add nuw nsw i64 %25, 1
  %30 = icmp eq i64 %29, %10
  br i1 %30, label %31, label %24, !llvm.loop !338

31:                                               ; preds = %24
  %32 = add nsw i64 %19, -1
  %33 = icmp sgt i64 %19, 0
  br i1 %33, label %18, label %34, !llvm.loop !339

34:                                               ; preds = %31, %5
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPmS0_jjEZ4mainE4$_25E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_25", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPhS0_jjEZ4mainE4$_26E9_M_invokeERKSt9_Any_dataOS0_S7_OjS8_"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %4) #15 {
  %6 = load ptr, ptr %1, align 8, !tbaa !54
  %7 = load ptr, ptr %2, align 8, !tbaa !54
  %8 = load i32, ptr %3, align 4, !tbaa !49
  %9 = load i32, ptr %4, align 4, !tbaa !49
  %10 = zext i32 %9 to i64
  %11 = icmp eq i32 %9, 0
  %12 = zext i32 %8 to i64
  %13 = add i32 %8, 1
  %14 = zext i32 %13 to i64
  br i1 %11, label %36, label %15

15:                                               ; preds = %5
  %16 = add i32 %8, -1
  %17 = zext i32 %16 to i64
  br label %18

18:                                               ; preds = %33, %15
  %19 = phi i64 [ %34, %33 ], [ %17, %15 ]
  %20 = mul nuw nsw i64 %19, %12
  %21 = getelementptr inbounds nuw i8, ptr %7, i64 %20
  %22 = mul nuw nsw i64 %19, %14
  %23 = getelementptr inbounds nuw i8, ptr %6, i64 %22
  br label %24

24:                                               ; preds = %24, %18
  %25 = phi i64 [ 0, %18 ], [ %31, %24 ]
  %26 = getelementptr inbounds nuw i8, ptr %21, i64 %25
  %27 = load i8, ptr %26, align 1, !tbaa !15
  %28 = getelementptr inbounds nuw i8, ptr %23, i64 %25
  %29 = load i8, ptr %28, align 1, !tbaa !15
  %30 = add i8 %29, %27
  store i8 %30, ptr %28, align 1, !tbaa !15
  %31 = add nuw nsw i64 %25, 1
  %32 = icmp eq i64 %31, %10
  br i1 %32, label %33, label %24, !llvm.loop !340

33:                                               ; preds = %24
  %34 = add nsw i64 %19, -1
  %35 = icmp sgt i64 %19, 0
  br i1 %35, label %18, label %36, !llvm.loop !341

36:                                               ; preds = %33, %5
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPhS0_jjEZ4mainE4$_26E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_26", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPhS0_jjEZ4mainE4$_27E9_M_invokeERKSt9_Any_dataOS0_S7_OjS8_"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %4) #15 {
  %6 = load ptr, ptr %1, align 8, !tbaa !54
  %7 = load ptr, ptr %2, align 8, !tbaa !54
  %8 = load i32, ptr %3, align 4, !tbaa !49
  %9 = load i32, ptr %4, align 4, !tbaa !49
  %10 = zext i32 %9 to i64
  %11 = icmp eq i32 %9, 0
  %12 = zext i32 %8 to i64
  %13 = add i32 %8, 1
  %14 = zext i32 %13 to i64
  br i1 %11, label %36, label %15

15:                                               ; preds = %5
  %16 = add i32 %8, -1
  %17 = zext i32 %16 to i64
  br label %18

18:                                               ; preds = %33, %15
  %19 = phi i64 [ %34, %33 ], [ %17, %15 ]
  %20 = mul nuw nsw i64 %19, %12
  %21 = getelementptr inbounds nuw i8, ptr %7, i64 %20
  %22 = mul nuw nsw i64 %19, %14
  %23 = getelementptr inbounds nuw i8, ptr %6, i64 %22
  br label %24

24:                                               ; preds = %18, %24
  %25 = phi i64 [ 0, %18 ], [ %31, %24 ]
  %26 = getelementptr inbounds nuw i8, ptr %21, i64 %25
  %27 = load i8, ptr %26, align 1, !tbaa !15
  %28 = getelementptr inbounds nuw i8, ptr %23, i64 %25
  %29 = load i8, ptr %28, align 1, !tbaa !15
  %30 = add i8 %29, %27
  store i8 %30, ptr %28, align 1, !tbaa !15
  %31 = add nuw nsw i64 %25, 1
  %32 = icmp eq i64 %31, %10
  br i1 %32, label %33, label %24, !llvm.loop !342

33:                                               ; preds = %24
  %34 = add nsw i64 %19, -1
  %35 = icmp sgt i64 %19, 0
  br i1 %35, label %18, label %36, !llvm.loop !343

36:                                               ; preds = %33, %5
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPhS0_jjEZ4mainE4$_27E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_27", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPjS0_jjEZ4mainE4$_26E9_M_invokeERKSt9_Any_dataOS0_S7_OjS8_"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %4) #15 {
  %6 = load ptr, ptr %1, align 8, !tbaa !58
  %7 = load ptr, ptr %2, align 8, !tbaa !58
  %8 = load i32, ptr %3, align 4, !tbaa !49
  %9 = load i32, ptr %4, align 4, !tbaa !49
  %10 = zext i32 %9 to i64
  %11 = icmp eq i32 %9, 0
  %12 = zext i32 %8 to i64
  %13 = add i32 %8, 1
  %14 = zext i32 %13 to i64
  br i1 %11, label %36, label %15

15:                                               ; preds = %5
  %16 = add i32 %8, -1
  %17 = zext i32 %16 to i64
  br label %18

18:                                               ; preds = %33, %15
  %19 = phi i64 [ %34, %33 ], [ %17, %15 ]
  %20 = mul nuw nsw i64 %19, %12
  %21 = getelementptr inbounds nuw i32, ptr %7, i64 %20
  %22 = mul nuw nsw i64 %19, %14
  %23 = getelementptr inbounds nuw i32, ptr %6, i64 %22
  br label %24

24:                                               ; preds = %24, %18
  %25 = phi i64 [ 0, %18 ], [ %31, %24 ]
  %26 = getelementptr inbounds nuw i32, ptr %21, i64 %25
  %27 = load i32, ptr %26, align 4, !tbaa !49
  %28 = getelementptr inbounds nuw i32, ptr %23, i64 %25
  %29 = load i32, ptr %28, align 4, !tbaa !49
  %30 = add i32 %29, %27
  store i32 %30, ptr %28, align 4, !tbaa !49
  %31 = add nuw nsw i64 %25, 1
  %32 = icmp eq i64 %31, %10
  br i1 %32, label %33, label %24, !llvm.loop !344

33:                                               ; preds = %24
  %34 = add nsw i64 %19, -1
  %35 = icmp sgt i64 %19, 0
  br i1 %35, label %18, label %36, !llvm.loop !345

36:                                               ; preds = %33, %5
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPjS0_jjEZ4mainE4$_26E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_26", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPjS0_jjEZ4mainE4$_27E9_M_invokeERKSt9_Any_dataOS0_S7_OjS8_"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %4) #15 {
  %6 = load ptr, ptr %1, align 8, !tbaa !58
  %7 = load ptr, ptr %2, align 8, !tbaa !58
  %8 = load i32, ptr %3, align 4, !tbaa !49
  %9 = load i32, ptr %4, align 4, !tbaa !49
  %10 = zext i32 %9 to i64
  %11 = icmp eq i32 %9, 0
  %12 = zext i32 %8 to i64
  %13 = add i32 %8, 1
  %14 = zext i32 %13 to i64
  br i1 %11, label %36, label %15

15:                                               ; preds = %5
  %16 = add i32 %8, -1
  %17 = zext i32 %16 to i64
  br label %18

18:                                               ; preds = %33, %15
  %19 = phi i64 [ %34, %33 ], [ %17, %15 ]
  %20 = mul nuw nsw i64 %19, %12
  %21 = getelementptr inbounds nuw i32, ptr %7, i64 %20
  %22 = mul nuw nsw i64 %19, %14
  %23 = getelementptr inbounds nuw i32, ptr %6, i64 %22
  br label %24

24:                                               ; preds = %18, %24
  %25 = phi i64 [ 0, %18 ], [ %31, %24 ]
  %26 = getelementptr inbounds nuw i32, ptr %21, i64 %25
  %27 = load i32, ptr %26, align 4, !tbaa !49
  %28 = getelementptr inbounds nuw i32, ptr %23, i64 %25
  %29 = load i32, ptr %28, align 4, !tbaa !49
  %30 = add i32 %29, %27
  store i32 %30, ptr %28, align 4, !tbaa !49
  %31 = add nuw nsw i64 %25, 1
  %32 = icmp eq i64 %31, %10
  br i1 %32, label %33, label %24, !llvm.loop !346

33:                                               ; preds = %24
  %34 = add nsw i64 %19, -1
  %35 = icmp sgt i64 %19, 0
  br i1 %35, label %18, label %36, !llvm.loop !347

36:                                               ; preds = %33, %5
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPjS0_jjEZ4mainE4$_27E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_27", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPmS0_jjEZ4mainE4$_26E9_M_invokeERKSt9_Any_dataOS0_S7_OjS8_"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %4) #15 {
  %6 = load ptr, ptr %1, align 8, !tbaa !62
  %7 = load ptr, ptr %2, align 8, !tbaa !62
  %8 = load i32, ptr %3, align 4, !tbaa !49
  %9 = load i32, ptr %4, align 4, !tbaa !49
  %10 = zext i32 %9 to i64
  %11 = icmp eq i32 %9, 0
  %12 = zext i32 %8 to i64
  %13 = add i32 %8, 1
  %14 = zext i32 %13 to i64
  br i1 %11, label %36, label %15

15:                                               ; preds = %5
  %16 = add i32 %8, -1
  %17 = zext i32 %16 to i64
  br label %18

18:                                               ; preds = %33, %15
  %19 = phi i64 [ %34, %33 ], [ %17, %15 ]
  %20 = mul nuw nsw i64 %19, %12
  %21 = getelementptr inbounds nuw i64, ptr %7, i64 %20
  %22 = mul nuw nsw i64 %19, %14
  %23 = getelementptr inbounds nuw i64, ptr %6, i64 %22
  br label %24

24:                                               ; preds = %24, %18
  %25 = phi i64 [ 0, %18 ], [ %31, %24 ]
  %26 = getelementptr inbounds nuw i64, ptr %21, i64 %25
  %27 = load i64, ptr %26, align 8, !tbaa !6
  %28 = getelementptr inbounds nuw i64, ptr %23, i64 %25
  %29 = load i64, ptr %28, align 8, !tbaa !6
  %30 = add i64 %29, %27
  store i64 %30, ptr %28, align 8, !tbaa !6
  %31 = add nuw nsw i64 %25, 1
  %32 = icmp eq i64 %31, %10
  br i1 %32, label %33, label %24, !llvm.loop !348

33:                                               ; preds = %24
  %34 = add nsw i64 %19, -1
  %35 = icmp sgt i64 %19, 0
  br i1 %35, label %18, label %36, !llvm.loop !349

36:                                               ; preds = %33, %5
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPmS0_jjEZ4mainE4$_26E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_26", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPmS0_jjEZ4mainE4$_27E9_M_invokeERKSt9_Any_dataOS0_S7_OjS8_"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %4) #15 {
  %6 = load ptr, ptr %1, align 8, !tbaa !62
  %7 = load ptr, ptr %2, align 8, !tbaa !62
  %8 = load i32, ptr %3, align 4, !tbaa !49
  %9 = load i32, ptr %4, align 4, !tbaa !49
  %10 = zext i32 %9 to i64
  %11 = icmp eq i32 %9, 0
  %12 = zext i32 %8 to i64
  %13 = add i32 %8, 1
  %14 = zext i32 %13 to i64
  br i1 %11, label %36, label %15

15:                                               ; preds = %5
  %16 = add i32 %8, -1
  %17 = zext i32 %16 to i64
  br label %18

18:                                               ; preds = %33, %15
  %19 = phi i64 [ %34, %33 ], [ %17, %15 ]
  %20 = mul nuw nsw i64 %19, %12
  %21 = getelementptr inbounds nuw i64, ptr %7, i64 %20
  %22 = mul nuw nsw i64 %19, %14
  %23 = getelementptr inbounds nuw i64, ptr %6, i64 %22
  br label %24

24:                                               ; preds = %18, %24
  %25 = phi i64 [ 0, %18 ], [ %31, %24 ]
  %26 = getelementptr inbounds nuw i64, ptr %21, i64 %25
  %27 = load i64, ptr %26, align 8, !tbaa !6
  %28 = getelementptr inbounds nuw i64, ptr %23, i64 %25
  %29 = load i64, ptr %28, align 8, !tbaa !6
  %30 = add i64 %29, %27
  store i64 %30, ptr %28, align 8, !tbaa !6
  %31 = add nuw nsw i64 %25, 1
  %32 = icmp eq i64 %31, %10
  br i1 %32, label %33, label %24, !llvm.loop !350

33:                                               ; preds = %24
  %34 = add nsw i64 %19, -1
  %35 = icmp sgt i64 %19, 0
  br i1 %35, label %18, label %36, !llvm.loop !351

36:                                               ; preds = %33, %5
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPmS0_jjEZ4mainE4$_27E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #16 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE4$_27", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !88
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define internal void @_GLOBAL__sub_I_runtime_checks.cpp() #17 section ".text.startup" {
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

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: read)
declare i32 @bcmp(ptr captures(none), ptr captures(none), i64) local_unnamed_addr #18

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.vector.reduce.add.v4i32(<4 x i32>) #19

attributes #0 = { mustprogress norecurse uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { noinline noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { cold nofree noreturn }
attributes #5 = { inlinehint mustprogress uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { nobuiltin allocsize(0) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #8 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: read) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #9 = { mustprogress uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #10 = { mustprogress noinline norecurse optnone uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #11 = { cold noreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #12 = { nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #13 = { nobuiltin nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #14 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #15 = { mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #16 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #17 = { nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #18 = { nocallback nofree nounwind willreturn memory(argmem: read) }
attributes #19 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #20 = { nounwind }
attributes #21 = { noreturn nounwind }
attributes #22 = { builtin allocsize(0) }
attributes #23 = { builtin nounwind }
attributes #24 = { cold noreturn }
attributes #25 = { cold noreturn nounwind }

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
!17 = !{!"_ZTSSt8functionIFvPhS0_jEE", !18, i64 0, !19, i64 24}
!18 = !{!"_ZTSSt14_Function_base", !8, i64 0, !19, i64 16}
!19 = !{!"any pointer", !8, i64 0}
!20 = !{!18, !19, i64 16}
!21 = !{!22, !19, i64 24}
!22 = !{!"_ZTSSt8functionIFvPjS0_jEE", !18, i64 0, !19, i64 24}
!23 = !{!24, !19, i64 24}
!24 = !{!"_ZTSSt8functionIFvPmS0_jEE", !18, i64 0, !19, i64 24}
!25 = !{!26, !19, i64 24}
!26 = !{!"_ZTSSt8functionIFvPjS0_S0_jEE", !18, i64 0, !19, i64 24}
!27 = !{!28, !19, i64 24}
!28 = !{!"_ZTSSt8functionIFvPhS0_S0_jEE", !18, i64 0, !19, i64 24}
!29 = !{!30, !19, i64 24}
!30 = !{!"_ZTSSt8functionIFvPmS0_S0_jEE", !18, i64 0, !19, i64 24}
!31 = !{!32, !19, i64 24}
!32 = !{!"_ZTSSt8functionIFvPhS0_jjEE", !18, i64 0, !19, i64 24}
!33 = !{!34, !19, i64 24}
!34 = !{!"_ZTSSt8functionIFvPjS0_jjEE", !18, i64 0, !19, i64 24}
!35 = !{!36, !19, i64 24}
!36 = !{!"_ZTSSt8functionIFvPmS0_jjEE", !18, i64 0, !19, i64 24}
!37 = !{!38, !38, i64 0}
!38 = !{!"vtable pointer", !9, i64 0}
!39 = !{!40, !42, i64 32}
!40 = !{!"_ZTSSt8ios_base", !7, i64 8, !7, i64 16, !41, i64 24, !42, i64 28, !42, i64 32, !43, i64 40, !44, i64 48, !8, i64 64, !45, i64 192, !46, i64 200, !47, i64 208}
!41 = !{!"_ZTSSt13_Ios_Fmtflags", !8, i64 0}
!42 = !{!"_ZTSSt12_Ios_Iostate", !8, i64 0}
!43 = !{!"p1 _ZTSNSt8ios_base14_Callback_listE", !19, i64 0}
!44 = !{!"_ZTSNSt8ios_base6_WordsE", !19, i64 0, !7, i64 8}
!45 = !{!"int", !8, i64 0}
!46 = !{!"p1 _ZTSNSt8ios_base6_WordsE", !19, i64 0}
!47 = !{!"_ZTSSt6locale", !48, i64 0}
!48 = !{!"p1 _ZTSNSt6locale5_ImplE", !19, i64 0}
!49 = !{!45, !45, i64 0}
!50 = !{!51, !8, i64 0}
!51 = !{!"_ZTSNSt24uniform_int_distributionIhE10param_typeE", !8, i64 0, !8, i64 1}
!52 = !{!51, !8, i64 1}
!53 = distinct !{!53, !11}
!54 = !{!55, !55, i64 0}
!55 = !{!"p1 omnipotent char", !19, i64 0}
!56 = distinct !{!56, !11}
!57 = distinct !{!57, !11}
!58 = !{!59, !59, i64 0}
!59 = !{!"p1 int", !19, i64 0}
!60 = distinct !{!60, !11}
!61 = distinct !{!61, !11}
!62 = !{!63, !63, i64 0}
!63 = !{!"p1 long", !19, i64 0}
!64 = distinct !{!64, !11}
!65 = distinct !{!65, !11}
!66 = distinct !{!66, !11}
!67 = distinct !{!67, !11}
!68 = distinct !{!68, !11}
!69 = distinct !{!69, !11}
!70 = distinct !{!70, !11}
!71 = !{!72, !72, i64 0}
!72 = !{!"p1 _ZTSSt8functionIFvPhS0_jEE", !19, i64 0}
!73 = !{!74, !74, i64 0}
!74 = !{!"p2 omnipotent char", !75, i64 0}
!75 = !{!"any p2 pointer", !19, i64 0}
!76 = !{}
!77 = !{i64 8}
!78 = !{i64 4}
!79 = distinct !{!79, !11}
!80 = distinct !{!80, !11}
!81 = distinct !{!81, !11, !82, !83}
!82 = !{!"llvm.loop.isvectorized", i32 1}
!83 = !{!"llvm.loop.unroll.runtime.disable"}
!84 = distinct !{!84, !11, !82, !83}
!85 = distinct !{!85, !11, !86, !87}
!86 = !{!"llvm.loop.vectorize.width", i32 1}
!87 = !{!"llvm.loop.interleave.count", i32 1}
!88 = !{!19, !19, i64 0}
!89 = distinct !{!89, !11, !82, !83}
!90 = distinct !{!90, !11, !82, !83}
!91 = distinct !{!91, !11, !82}
!92 = !{!93, !93, i64 0}
!93 = !{!"p1 _ZTSSt8functionIFvPjS0_jEE", !19, i64 0}
!94 = !{!95, !95, i64 0}
!95 = !{!"p2 int", !75, i64 0}
!96 = !{!97, !45, i64 4}
!97 = !{!"_ZTSNSt24uniform_int_distributionIjE10param_typeE", !45, i64 0, !45, i64 4}
!98 = !{!97, !45, i64 0}
!99 = distinct !{!99, !11}
!100 = distinct !{!100, !11}
!101 = distinct !{!101, !11, !86, !87}
!102 = distinct !{!102, !11, !82, !83}
!103 = distinct !{!103, !11, !82}
!104 = !{!105, !105, i64 0}
!105 = !{!"p1 _ZTSSt8functionIFvPmS0_jEE", !19, i64 0}
!106 = !{!107, !107, i64 0}
!107 = !{!"p2 long", !75, i64 0}
!108 = !{!109, !7, i64 8}
!109 = !{!"_ZTSNSt24uniform_int_distributionImE10param_typeE", !7, i64 0, !7, i64 8}
!110 = !{!109, !7, i64 0}
!111 = distinct !{!111, !11}
!112 = distinct !{!112, !11}
!113 = distinct !{!113, !11, !86, !87}
!114 = distinct !{!114, !11, !82, !83}
!115 = distinct !{!115, !11, !82}
!116 = distinct !{!116, !11, !86, !87}
!117 = distinct !{!117, !11, !82, !83}
!118 = distinct !{!118, !11, !82, !83}
!119 = distinct !{!119, !11, !82}
!120 = distinct !{!120, !11, !86, !87}
!121 = distinct !{!121, !11, !82, !83}
!122 = distinct !{!122, !11, !82}
!123 = distinct !{!123, !11, !86, !87}
!124 = distinct !{!124, !11, !82, !83}
!125 = distinct !{!125, !11, !82}
!126 = distinct !{!126, !11, !86, !87}
!127 = distinct !{!127, !11, !82, !83}
!128 = distinct !{!128, !11, !82, !83}
!129 = distinct !{!129, !11, !82}
!130 = distinct !{!130, !11, !86, !87}
!131 = distinct !{!131, !11, !82, !83}
!132 = distinct !{!132, !11, !82}
!133 = distinct !{!133, !11, !86, !87}
!134 = distinct !{!134, !11, !82, !83}
!135 = distinct !{!135, !11, !82}
!136 = distinct !{!136, !11, !86, !87}
!137 = distinct !{!137, !11, !82, !83}
!138 = distinct !{!138, !11, !82, !83}
!139 = distinct !{!139, !11, !82}
!140 = distinct !{!140, !11, !86, !87}
!141 = distinct !{!141, !11, !82, !83}
!142 = distinct !{!142, !11, !82}
!143 = distinct !{!143, !11, !86, !87}
!144 = distinct !{!144, !11, !82, !83}
!145 = distinct !{!145, !11, !82}
!146 = distinct !{!146, !11, !86, !87}
!147 = !{!148}
!148 = distinct !{!148, !149}
!149 = distinct !{!149, !"LVerDomain"}
!150 = !{!151}
!151 = distinct !{!151, !149}
!152 = distinct !{!152, !11, !82, !83}
!153 = distinct !{!153, !11, !82}
!154 = distinct !{!154, !11, !86, !87}
!155 = !{!156}
!156 = distinct !{!156, !157}
!157 = distinct !{!157, !"LVerDomain"}
!158 = !{!159}
!159 = distinct !{!159, !157}
!160 = distinct !{!160, !11, !82, !83}
!161 = distinct !{!161, !11, !82}
!162 = distinct !{!162, !11, !86, !87}
!163 = !{!164}
!164 = distinct !{!164, !165}
!165 = distinct !{!165, !"LVerDomain"}
!166 = !{!167}
!167 = distinct !{!167, !165}
!168 = distinct !{!168, !11, !82, !83}
!169 = distinct !{!169, !11, !82}
!170 = distinct !{!170, !11, !86, !87}
!171 = !{!172}
!172 = distinct !{!172, !173}
!173 = distinct !{!173, !"LVerDomain"}
!174 = !{!175}
!175 = distinct !{!175, !173}
!176 = distinct !{!176, !11, !82, !83}
!177 = distinct !{!177, !11, !82, !83}
!178 = distinct !{!178, !11, !82}
!179 = distinct !{!179, !11, !86, !87}
!180 = !{!181}
!181 = distinct !{!181, !182}
!182 = distinct !{!182, !"LVerDomain"}
!183 = !{!184}
!184 = distinct !{!184, !182}
!185 = distinct !{!185, !11, !82, !83}
!186 = distinct !{!186, !11, !82}
!187 = distinct !{!187, !11, !86, !87}
!188 = !{!189}
!189 = distinct !{!189, !190}
!190 = distinct !{!190, !"LVerDomain"}
!191 = !{!192}
!192 = distinct !{!192, !190}
!193 = distinct !{!193, !11, !82, !83}
!194 = distinct !{!194, !11, !82}
!195 = distinct !{!195, !11, !196, !86, !87}
!196 = !{!"llvm.loop.unroll.disable"}
!197 = !{!198}
!198 = distinct !{!198, !199}
!199 = distinct !{!199, !"LVerDomain"}
!200 = !{!201}
!201 = distinct !{!201, !199}
!202 = distinct !{!202, !11, !196, !82, !83}
!203 = distinct !{!203, !11, !196, !82, !83}
!204 = distinct !{!204, !11, !196, !82}
!205 = distinct !{!205, !11, !196, !86, !87}
!206 = !{!207}
!207 = distinct !{!207, !208}
!208 = distinct !{!208, !"LVerDomain"}
!209 = !{!210}
!210 = distinct !{!210, !208}
!211 = distinct !{!211, !11, !196, !82, !83}
!212 = distinct !{!212, !11, !196, !82}
!213 = distinct !{!213, !11, !196, !86, !87}
!214 = !{!215}
!215 = distinct !{!215, !216}
!216 = distinct !{!216, !"LVerDomain"}
!217 = !{!218}
!218 = distinct !{!218, !216}
!219 = distinct !{!219, !11, !196, !82, !83}
!220 = distinct !{!220, !11, !196, !82}
!221 = distinct !{!221, !11, !86, !87}
!222 = distinct !{!222, !11, !223}
!223 = !{!"llvm.loop.vectorize.enable", i1 true}
!224 = distinct !{!224, !11, !86, !87}
!225 = !{!226}
!226 = distinct !{!226, !227}
!227 = distinct !{!227, !"LVerDomain"}
!228 = distinct !{!228, !11, !82, !83}
!229 = !{!230}
!230 = distinct !{!230, !227}
!231 = distinct !{!231, !11, !82}
!232 = distinct !{!232, !11, !86, !87}
!233 = distinct !{!233, !11, !223}
!234 = !{!235, !235, i64 0}
!235 = !{!"p1 _ZTSSt8functionIFvPjS0_S0_jEE", !19, i64 0}
!236 = distinct !{!236, !11, !86, !87}
!237 = distinct !{!237, !11, !82, !83}
!238 = distinct !{!238, !11, !82}
!239 = !{!240, !240, i64 0}
!240 = !{!"p1 _ZTSSt8functionIFvPhS0_S0_jEE", !19, i64 0}
!241 = distinct !{!241, !11, !86, !87}
!242 = distinct !{!242, !11, !82, !83}
!243 = distinct !{!243, !11, !82, !83}
!244 = distinct !{!244, !11, !82}
!245 = !{!246, !246, i64 0}
!246 = !{!"p1 _ZTSSt8functionIFvPmS0_S0_jEE", !19, i64 0}
!247 = distinct !{!247, !11, !86, !87}
!248 = distinct !{!248, !11, !82, !83}
!249 = distinct !{!249, !11, !82}
!250 = distinct !{!250, !11, !86, !87}
!251 = distinct !{!251, !11, !82, !83}
!252 = distinct !{!252, !11, !82, !83}
!253 = distinct !{!253, !11, !82}
!254 = distinct !{!254, !11, !86, !87}
!255 = distinct !{!255, !11, !82, !83}
!256 = distinct !{!256, !11, !82}
!257 = distinct !{!257, !11, !86, !87}
!258 = distinct !{!258, !11, !82, !83}
!259 = distinct !{!259, !11, !82}
!260 = !{!261, !261, i64 0}
!261 = !{!"p1 _ZTSSt8functionIFvPhS0_jjEE", !19, i64 0}
!262 = distinct !{!262, !11, !86, !87}
!263 = distinct !{!263, !11}
!264 = !{!265}
!265 = distinct !{!265, !266}
!266 = distinct !{!266, !"LVerDomain"}
!267 = !{!268}
!268 = distinct !{!268, !266}
!269 = distinct !{!269, !11, !82, !83}
!270 = distinct !{!270, !11, !82, !83}
!271 = distinct !{!271, !11, !82}
!272 = distinct !{!272, !11}
!273 = !{!274, !274, i64 0}
!274 = !{!"p1 _ZTSSt8functionIFvPjS0_jjEE", !19, i64 0}
!275 = distinct !{!275, !11, !86, !87}
!276 = distinct !{!276, !11}
!277 = !{!278}
!278 = distinct !{!278, !279}
!279 = distinct !{!279, !"LVerDomain"}
!280 = !{!281}
!281 = distinct !{!281, !279}
!282 = distinct !{!282, !11, !82, !83}
!283 = distinct !{!283, !11, !82}
!284 = distinct !{!284, !11}
!285 = !{!286, !286, i64 0}
!286 = !{!"p1 _ZTSSt8functionIFvPmS0_jjEE", !19, i64 0}
!287 = distinct !{!287, !11, !86, !87}
!288 = distinct !{!288, !11}
!289 = !{!290}
!290 = distinct !{!290, !291}
!291 = distinct !{!291, !"LVerDomain"}
!292 = !{!293}
!293 = distinct !{!293, !291}
!294 = distinct !{!294, !11, !82, !83}
!295 = distinct !{!295, !11, !82}
!296 = distinct !{!296, !11}
!297 = distinct !{!297, !11, !86, !87}
!298 = distinct !{!298, !11}
!299 = !{!300}
!300 = distinct !{!300, !301}
!301 = distinct !{!301, !"LVerDomain"}
!302 = !{!303}
!303 = distinct !{!303, !301}
!304 = distinct !{!304, !11, !82, !83}
!305 = distinct !{!305, !11, !82, !83}
!306 = distinct !{!306, !11, !82}
!307 = distinct !{!307, !11}
!308 = distinct !{!308, !11, !86, !87}
!309 = distinct !{!309, !11}
!310 = !{!311}
!311 = distinct !{!311, !312}
!312 = distinct !{!312, !"LVerDomain"}
!313 = !{!314}
!314 = distinct !{!314, !312}
!315 = distinct !{!315, !11, !82, !83}
!316 = distinct !{!316, !11, !82}
!317 = distinct !{!317, !11}
!318 = distinct !{!318, !11, !86, !87}
!319 = distinct !{!319, !11}
!320 = !{!321}
!321 = distinct !{!321, !322}
!322 = distinct !{!322, !"LVerDomain"}
!323 = !{!324}
!324 = distinct !{!324, !322}
!325 = distinct !{!325, !11, !82, !83}
!326 = distinct !{!326, !11, !82}
!327 = distinct !{!327, !11}
!328 = distinct !{!328, !11, !86, !87}
!329 = distinct !{!329, !11}
!330 = distinct !{!330, !11, !82}
!331 = distinct !{!331, !11}
!332 = distinct !{!332, !11, !86, !87}
!333 = distinct !{!333, !11}
!334 = distinct !{!334, !11, !82}
!335 = distinct !{!335, !11}
!336 = distinct !{!336, !11, !86, !87}
!337 = distinct !{!337, !11}
!338 = distinct !{!338, !11, !82}
!339 = distinct !{!339, !11}
!340 = distinct !{!340, !11, !86, !87}
!341 = distinct !{!341, !11}
!342 = distinct !{!342, !11, !82}
!343 = distinct !{!343, !11}
!344 = distinct !{!344, !11, !86, !87}
!345 = distinct !{!345, !11}
!346 = distinct !{!346, !11, !82}
!347 = distinct !{!347, !11}
!348 = distinct !{!348, !11, !86, !87}
!349 = distinct !{!349, !11}
!350 = distinct !{!350, !11, !82}
!351 = distinct !{!351, !11}
