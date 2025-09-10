; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/Vectorizer/recurrences.cpp'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/Vectorizer/recurrences.cpp"
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

$__clang_call_terminate = comdat any

$_ZNSt24uniform_int_distributionIjEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEjRT_RKNS0_10param_typeE = comdat any

$_ZNSt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEclEv = comdat any

@_ZL3rng = internal global %"class.std::mersenne_twister_engine" zeroinitializer, align 8
@.str = private unnamed_addr constant [23 x i8] c"first_order_recurrence\00", align 1
@.str.1 = private unnamed_addr constant [24 x i8] c"second_order_recurrence\00", align 1
@.str.2 = private unnamed_addr constant [23 x i8] c"third_order_recurrence\00", align 1
@_ZSt4cout = external global %"class.std::basic_ostream", align 8
@.str.3 = private unnamed_addr constant [10 x i8] c"Checking \00", align 1
@.str.4 = private unnamed_addr constant [2 x i8] c"\0A\00", align 1
@_ZSt4cerr = external global %"class.std::basic_ostream", align 8
@.str.5 = private unnamed_addr constant [12 x i8] c"Miscompare\0A\00", align 1
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
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_recurrences.cpp, ptr null }]

; Function Attrs: mustprogress norecurse uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 personality ptr @__gxx_personality_v0 {
  %1 = alloca %"class.std::mersenne_twister_engine", align 8
  %2 = alloca %"class.std::function", align 8
  %3 = alloca %"class.std::function", align 8
  %4 = alloca %"class.std::function", align 8
  %5 = alloca %"class.std::function", align 8
  %6 = alloca %"class.std::function", align 8
  %7 = alloca %"class.std::function", align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #19
  store i64 15, ptr %1, align 8, !tbaa !6
  br label %8

8:                                                ; preds = %8, %0
  %9 = phi i64 [ 15, %0 ], [ %16, %8 ]
  %10 = phi i64 [ 1, %0 ], [ %17, %8 ]
  %11 = getelementptr i64, ptr %1, i64 %10
  %12 = lshr i64 %9, 30
  %13 = xor i64 %12, %9
  %14 = mul nuw nsw i64 %13, 1812433253
  %15 = add nuw i64 %14, %10
  %16 = and i64 %15, 4294967295
  store i64 %16, ptr %11, align 8, !tbaa !6
  %17 = add nuw nsw i64 %10, 1
  %18 = icmp eq i64 %17, 624
  br i1 %18, label %19, label %8, !llvm.loop !10

19:                                               ; preds = %8
  %20 = getelementptr inbounds nuw i8, ptr %1, i64 4992
  store i64 624, ptr %20, align 8, !tbaa !12
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(5000) @_ZL3rng, ptr noundef nonnull align 8 dereferenceable(5000) %1, i64 5000, i1 false), !tbaa.struct !14
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #19
  %21 = getelementptr inbounds nuw i8, ptr %2, i64 16
  %22 = getelementptr inbounds nuw i8, ptr %2, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %2, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE3$_0E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %22, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE3$_0E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %21, align 8, !tbaa !20
  %23 = getelementptr inbounds nuw i8, ptr %3, i64 16
  %24 = getelementptr inbounds nuw i8, ptr %3, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %3, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE3$_1E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %24, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE3$_1E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %23, align 8, !tbaa !20
  invoke fastcc void @_ZL19checkVectorFunctionIjEvSt8functionIFvPT_S2_jEES4_PKc(ptr noundef %2, ptr noundef %3, ptr noundef nonnull @.str)
          to label %25 unwind label %84

25:                                               ; preds = %19
  %26 = load ptr, ptr %23, align 8, !tbaa !20
  %27 = icmp eq ptr %26, null
  br i1 %27, label %33, label %28

28:                                               ; preds = %25
  %29 = invoke noundef i1 %26(ptr noundef nonnull align 8 dereferenceable(32) %3, ptr noundef nonnull align 8 dereferenceable(32) %3, i32 noundef 3)
          to label %33 unwind label %30

30:                                               ; preds = %28
  %31 = landingpad { ptr, i32 }
          catch ptr null
  %32 = extractvalue { ptr, i32 } %31, 0
  call void @__clang_call_terminate(ptr %32) #20
  unreachable

33:                                               ; preds = %25, %28
  %34 = load ptr, ptr %21, align 8, !tbaa !20
  %35 = icmp eq ptr %34, null
  br i1 %35, label %41, label %36

36:                                               ; preds = %33
  %37 = invoke noundef i1 %34(ptr noundef nonnull align 8 dereferenceable(32) %2, ptr noundef nonnull align 8 dereferenceable(32) %2, i32 noundef 3)
          to label %41 unwind label %38

38:                                               ; preds = %36
  %39 = landingpad { ptr, i32 }
          catch ptr null
  %40 = extractvalue { ptr, i32 } %39, 0
  call void @__clang_call_terminate(ptr %40) #20
  unreachable

41:                                               ; preds = %33, %36
  %42 = getelementptr inbounds nuw i8, ptr %4, i64 16
  %43 = getelementptr inbounds nuw i8, ptr %4, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %4, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE3$_2E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %43, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE3$_2E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %42, align 8, !tbaa !20
  %44 = getelementptr inbounds nuw i8, ptr %5, i64 16
  %45 = getelementptr inbounds nuw i8, ptr %5, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %5, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE3$_3E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %45, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE3$_3E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %44, align 8, !tbaa !20
  invoke fastcc void @_ZL19checkVectorFunctionIjEvSt8functionIFvPT_S2_jEES4_PKc(ptr noundef %4, ptr noundef %5, ptr noundef nonnull @.str.1)
          to label %46 unwind label %101

46:                                               ; preds = %41
  %47 = load ptr, ptr %44, align 8, !tbaa !20
  %48 = icmp eq ptr %47, null
  br i1 %48, label %54, label %49

49:                                               ; preds = %46
  %50 = invoke noundef i1 %47(ptr noundef nonnull align 8 dereferenceable(32) %5, ptr noundef nonnull align 8 dereferenceable(32) %5, i32 noundef 3)
          to label %54 unwind label %51

51:                                               ; preds = %49
  %52 = landingpad { ptr, i32 }
          catch ptr null
  %53 = extractvalue { ptr, i32 } %52, 0
  call void @__clang_call_terminate(ptr %53) #20
  unreachable

54:                                               ; preds = %46, %49
  %55 = load ptr, ptr %42, align 8, !tbaa !20
  %56 = icmp eq ptr %55, null
  br i1 %56, label %62, label %57

57:                                               ; preds = %54
  %58 = invoke noundef i1 %55(ptr noundef nonnull align 8 dereferenceable(32) %4, ptr noundef nonnull align 8 dereferenceable(32) %4, i32 noundef 3)
          to label %62 unwind label %59

59:                                               ; preds = %57
  %60 = landingpad { ptr, i32 }
          catch ptr null
  %61 = extractvalue { ptr, i32 } %60, 0
  call void @__clang_call_terminate(ptr %61) #20
  unreachable

62:                                               ; preds = %54, %57
  %63 = getelementptr inbounds nuw i8, ptr %6, i64 16
  %64 = getelementptr inbounds nuw i8, ptr %6, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %6, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE3$_4E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %64, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE3$_4E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %63, align 8, !tbaa !20
  %65 = getelementptr inbounds nuw i8, ptr %7, i64 16
  %66 = getelementptr inbounds nuw i8, ptr %7, i64 24
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %7, i8 0, i64 16, i1 false)
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE3$_5E9_M_invokeERKSt9_Any_dataOS0_S7_Oj", ptr %66, align 8, !tbaa !16
  store ptr @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE3$_5E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation", ptr %65, align 8, !tbaa !20
  invoke fastcc void @_ZL19checkVectorFunctionIjEvSt8functionIFvPT_S2_jEES4_PKc(ptr noundef %6, ptr noundef %7, ptr noundef nonnull @.str.2)
          to label %67 unwind label %118

67:                                               ; preds = %62
  %68 = load ptr, ptr %65, align 8, !tbaa !20
  %69 = icmp eq ptr %68, null
  br i1 %69, label %75, label %70

70:                                               ; preds = %67
  %71 = invoke noundef i1 %68(ptr noundef nonnull align 8 dereferenceable(32) %7, ptr noundef nonnull align 8 dereferenceable(32) %7, i32 noundef 3)
          to label %75 unwind label %72

72:                                               ; preds = %70
  %73 = landingpad { ptr, i32 }
          catch ptr null
  %74 = extractvalue { ptr, i32 } %73, 0
  call void @__clang_call_terminate(ptr %74) #20
  unreachable

75:                                               ; preds = %67, %70
  %76 = load ptr, ptr %63, align 8, !tbaa !20
  %77 = icmp eq ptr %76, null
  br i1 %77, label %83, label %78

78:                                               ; preds = %75
  %79 = invoke noundef i1 %76(ptr noundef nonnull align 8 dereferenceable(32) %6, ptr noundef nonnull align 8 dereferenceable(32) %6, i32 noundef 3)
          to label %83 unwind label %80

80:                                               ; preds = %78
  %81 = landingpad { ptr, i32 }
          catch ptr null
  %82 = extractvalue { ptr, i32 } %81, 0
  call void @__clang_call_terminate(ptr %82) #20
  unreachable

83:                                               ; preds = %75, %78
  ret i32 0

84:                                               ; preds = %19
  %85 = landingpad { ptr, i32 }
          cleanup
  %86 = load ptr, ptr %23, align 8, !tbaa !20
  %87 = icmp eq ptr %86, null
  br i1 %87, label %93, label %88

88:                                               ; preds = %84
  %89 = invoke noundef i1 %86(ptr noundef nonnull align 8 dereferenceable(32) %3, ptr noundef nonnull align 8 dereferenceable(32) %3, i32 noundef 3)
          to label %93 unwind label %90

90:                                               ; preds = %88
  %91 = landingpad { ptr, i32 }
          catch ptr null
  %92 = extractvalue { ptr, i32 } %91, 0
  call void @__clang_call_terminate(ptr %92) #20
  unreachable

93:                                               ; preds = %84, %88
  %94 = load ptr, ptr %21, align 8, !tbaa !20
  %95 = icmp eq ptr %94, null
  br i1 %95, label %135, label %96

96:                                               ; preds = %93
  %97 = invoke noundef i1 %94(ptr noundef nonnull align 8 dereferenceable(32) %2, ptr noundef nonnull align 8 dereferenceable(32) %2, i32 noundef 3)
          to label %135 unwind label %98

98:                                               ; preds = %96
  %99 = landingpad { ptr, i32 }
          catch ptr null
  %100 = extractvalue { ptr, i32 } %99, 0
  call void @__clang_call_terminate(ptr %100) #20
  unreachable

101:                                              ; preds = %41
  %102 = landingpad { ptr, i32 }
          cleanup
  %103 = load ptr, ptr %44, align 8, !tbaa !20
  %104 = icmp eq ptr %103, null
  br i1 %104, label %110, label %105

105:                                              ; preds = %101
  %106 = invoke noundef i1 %103(ptr noundef nonnull align 8 dereferenceable(32) %5, ptr noundef nonnull align 8 dereferenceable(32) %5, i32 noundef 3)
          to label %110 unwind label %107

107:                                              ; preds = %105
  %108 = landingpad { ptr, i32 }
          catch ptr null
  %109 = extractvalue { ptr, i32 } %108, 0
  call void @__clang_call_terminate(ptr %109) #20
  unreachable

110:                                              ; preds = %101, %105
  %111 = load ptr, ptr %42, align 8, !tbaa !20
  %112 = icmp eq ptr %111, null
  br i1 %112, label %135, label %113

113:                                              ; preds = %110
  %114 = invoke noundef i1 %111(ptr noundef nonnull align 8 dereferenceable(32) %4, ptr noundef nonnull align 8 dereferenceable(32) %4, i32 noundef 3)
          to label %135 unwind label %115

115:                                              ; preds = %113
  %116 = landingpad { ptr, i32 }
          catch ptr null
  %117 = extractvalue { ptr, i32 } %116, 0
  call void @__clang_call_terminate(ptr %117) #20
  unreachable

118:                                              ; preds = %62
  %119 = landingpad { ptr, i32 }
          cleanup
  %120 = load ptr, ptr %65, align 8, !tbaa !20
  %121 = icmp eq ptr %120, null
  br i1 %121, label %127, label %122

122:                                              ; preds = %118
  %123 = invoke noundef i1 %120(ptr noundef nonnull align 8 dereferenceable(32) %7, ptr noundef nonnull align 8 dereferenceable(32) %7, i32 noundef 3)
          to label %127 unwind label %124

124:                                              ; preds = %122
  %125 = landingpad { ptr, i32 }
          catch ptr null
  %126 = extractvalue { ptr, i32 } %125, 0
  call void @__clang_call_terminate(ptr %126) #20
  unreachable

127:                                              ; preds = %118, %122
  %128 = load ptr, ptr %63, align 8, !tbaa !20
  %129 = icmp eq ptr %128, null
  br i1 %129, label %135, label %130

130:                                              ; preds = %127
  %131 = invoke noundef i1 %128(ptr noundef nonnull align 8 dereferenceable(32) %6, ptr noundef nonnull align 8 dereferenceable(32) %6, i32 noundef 3)
          to label %135 unwind label %132

132:                                              ; preds = %130
  %133 = landingpad { ptr, i32 }
          catch ptr null
  %134 = extractvalue { ptr, i32 } %133, 0
  call void @__clang_call_terminate(ptr %134) #20
  unreachable

135:                                              ; preds = %130, %127, %113, %110, %96, %93
  %136 = phi { ptr, i32 } [ %85, %93 ], [ %85, %96 ], [ %102, %110 ], [ %102, %113 ], [ %119, %127 ], [ %119, %130 ]
  resume { ptr, i32 } %136
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: mustprogress norecurse uwtable
define internal fastcc void @_ZL19checkVectorFunctionIjEvSt8functionIFvPT_S2_jEES4_PKc(ptr noundef nonnull %0, ptr noundef nonnull %1, ptr noundef %2) unnamed_addr #0 personality ptr @__gxx_personality_v0 {
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca i32, align 4
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  %9 = alloca i32, align 4
  %10 = alloca %"class.std::uniform_int_distribution", align 8
  %11 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.3, i64 noundef 9)
  %12 = icmp eq ptr %2, null
  br i1 %12, label %13, label %21

13:                                               ; preds = %3
  %14 = load ptr, ptr @_ZSt4cout, align 8, !tbaa !21
  %15 = getelementptr i8, ptr %14, i64 -24
  %16 = load i64, ptr %15, align 8
  %17 = getelementptr inbounds i8, ptr @_ZSt4cout, i64 %16
  %18 = getelementptr inbounds nuw i8, ptr %17, i64 32
  %19 = load i32, ptr %18, align 8, !tbaa !23
  %20 = or i32 %19, 1
  tail call void @_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate(ptr noundef nonnull align 8 dereferenceable(264) %17, i32 noundef %20)
  br label %24

21:                                               ; preds = %3
  %22 = tail call noundef i64 @strlen(ptr noundef nonnull dereferenceable(1) %2) #19
  %23 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull %2, i64 noundef %22)
  br label %24

24:                                               ; preds = %13, %21
  %25 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.4, i64 noundef 1)
  %26 = tail call noalias noundef nonnull dereferenceable(4000) ptr @_Znam(i64 noundef 4000) #21
  call void @llvm.lifetime.start.p0(ptr nonnull %10) #19
  store <2 x i32> <i32 0, i32 -1>, ptr %10, align 8, !tbaa !33
  br label %27

27:                                               ; preds = %30, %24
  %28 = phi i64 [ 0, %24 ], [ %32, %30 ]
  %29 = invoke noundef i32 @_ZNSt24uniform_int_distributionIjEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEjRT_RKNS0_10param_typeE(ptr noundef nonnull align 4 dereferenceable(8) %10, ptr noundef nonnull align 8 dereferenceable(5000) @_ZL3rng, ptr noundef nonnull align 4 dereferenceable(8) %10)
          to label %30 unwind label %38

30:                                               ; preds = %27
  %31 = getelementptr inbounds nuw i32, ptr %26, i64 %28
  store i32 %29, ptr %31, align 4, !tbaa !33
  %32 = add nuw nsw i64 %28, 1
  %33 = icmp eq i64 %32, 1000
  br i1 %33, label %34, label %27, !llvm.loop !34

34:                                               ; preds = %30
  call void @llvm.lifetime.end.p0(ptr nonnull %10) #19
  %35 = invoke noalias noundef nonnull dereferenceable(4000) ptr @_Znam(i64 noundef 4000) #21
          to label %36 unwind label %40

36:                                               ; preds = %34
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 4 dereferenceable(4000) %35, ptr noundef nonnull align 4 dereferenceable(4000) %26, i64 4000, i1 false), !tbaa !33
  %37 = invoke noalias noundef nonnull dereferenceable(4000) ptr @_Znam(i64 noundef 4000) #21
          to label %42 unwind label %67

38:                                               ; preds = %27
  %39 = landingpad { ptr, i32 }
          cleanup
  br label %77

40:                                               ; preds = %34
  %41 = landingpad { ptr, i32 }
          cleanup
  br label %77

42:                                               ; preds = %36
  %43 = invoke noalias noundef nonnull dereferenceable(4000) ptr @_Znam(i64 noundef 4000) #21
          to label %44 unwind label %69

44:                                               ; preds = %42
  call void @llvm.lifetime.start.p0(ptr nonnull %7)
  call void @llvm.lifetime.start.p0(ptr nonnull %8)
  call void @llvm.lifetime.start.p0(ptr nonnull %9)
  store ptr %26, ptr %7, align 8, !tbaa !35
  store ptr %37, ptr %8, align 8, !tbaa !35
  store i32 1000, ptr %9, align 4, !tbaa !33
  %45 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %46 = load ptr, ptr %45, align 8, !tbaa !20
  %47 = icmp eq ptr %46, null
  br i1 %47, label %55, label %48

48:                                               ; preds = %44
  %49 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %50 = load ptr, ptr %49, align 8, !tbaa !16
  invoke void %50(ptr noundef nonnull align 8 dereferenceable(32) %0, ptr noundef nonnull align 8 dereferenceable(8) %7, ptr noundef nonnull align 8 dereferenceable(8) %8, ptr noundef nonnull align 4 dereferenceable(4) %9)
          to label %51 unwind label %71

51:                                               ; preds = %48
  call void @llvm.lifetime.end.p0(ptr nonnull %7)
  call void @llvm.lifetime.end.p0(ptr nonnull %8)
  call void @llvm.lifetime.end.p0(ptr nonnull %9)
  call void @llvm.lifetime.start.p0(ptr nonnull %4)
  call void @llvm.lifetime.start.p0(ptr nonnull %5)
  call void @llvm.lifetime.start.p0(ptr nonnull %6)
  store ptr %35, ptr %4, align 8, !tbaa !35
  store ptr %43, ptr %5, align 8, !tbaa !35
  store i32 1000, ptr %6, align 4, !tbaa !33
  %52 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %53 = load ptr, ptr %52, align 8, !tbaa !20
  %54 = icmp eq ptr %53, null
  br i1 %54, label %55, label %57

55:                                               ; preds = %51, %44
  invoke void @_ZSt25__throw_bad_function_callv() #22
          to label %56 unwind label %71

56:                                               ; preds = %55
  unreachable

57:                                               ; preds = %51
  %58 = getelementptr inbounds nuw i8, ptr %1, i64 24
  %59 = load ptr, ptr %58, align 8, !tbaa !16
  invoke void %59(ptr noundef nonnull align 8 dereferenceable(32) %1, ptr noundef nonnull align 8 dereferenceable(8) %4, ptr noundef nonnull align 8 dereferenceable(8) %5, ptr noundef nonnull align 4 dereferenceable(4) %6)
          to label %60 unwind label %71

60:                                               ; preds = %57
  call void @llvm.lifetime.end.p0(ptr nonnull %4)
  call void @llvm.lifetime.end.p0(ptr nonnull %5)
  call void @llvm.lifetime.end.p0(ptr nonnull %6)
  %61 = call i32 @bcmp(ptr noundef nonnull readonly dereferenceable(4000) %37, ptr noundef nonnull readonly dereferenceable(4000) %43, i64 4000)
  %62 = icmp eq i32 %61, 0
  br i1 %62, label %66, label %63

63:                                               ; preds = %60
  %64 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.5)
          to label %65 unwind label %71

65:                                               ; preds = %63
  call void @exit(i32 noundef 1) #23
  unreachable

66:                                               ; preds = %60
  call void @_ZdaPv(ptr noundef nonnull %43) #24
  call void @_ZdaPv(ptr noundef nonnull %37) #24
  call void @_ZdaPv(ptr noundef nonnull %35) #24
  call void @_ZdaPv(ptr noundef nonnull %26) #24
  ret void

67:                                               ; preds = %36
  %68 = landingpad { ptr, i32 }
          cleanup
  br label %75

69:                                               ; preds = %42
  %70 = landingpad { ptr, i32 }
          cleanup
  br label %73

71:                                               ; preds = %55, %63, %57, %48
  %72 = landingpad { ptr, i32 }
          cleanup
  call void @_ZdaPv(ptr noundef nonnull %43) #24
  br label %73

73:                                               ; preds = %71, %69
  %74 = phi { ptr, i32 } [ %72, %71 ], [ %70, %69 ]
  call void @_ZdaPv(ptr noundef nonnull %37) #24
  br label %75

75:                                               ; preds = %73, %67
  %76 = phi { ptr, i32 } [ %74, %73 ], [ %68, %67 ]
  call void @_ZdaPv(ptr noundef nonnull %35) #24
  br label %77

77:                                               ; preds = %40, %75, %38
  %78 = phi { ptr, i32 } [ %39, %38 ], [ %76, %75 ], [ %41, %40 ]
  call void @_ZdaPv(ptr noundef nonnull %26) #24
  resume { ptr, i32 } %78
}

declare i32 @__gxx_personality_v0(...)

; Function Attrs: noinline noreturn nounwind uwtable
define linkonce_odr hidden void @__clang_call_terminate(ptr noundef %0) local_unnamed_addr #3 comdat {
  %2 = tail call ptr @__cxa_begin_catch(ptr %0) #19
  tail call void @_ZSt9terminatev() #20
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
define linkonce_odr dso_local noundef i32 @_ZNSt24uniform_int_distributionIjEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEjRT_RKNS0_10param_typeE(ptr noundef nonnull align 4 dereferenceable(8) %0, ptr noundef nonnull align 8 dereferenceable(5000) %1, ptr noundef nonnull align 4 dereferenceable(8) %2) local_unnamed_addr #9 comdat {
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
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #19
  store <2 x i32> <i32 0, i32 -1>, ptr %4, align 8, !tbaa !33
  %35 = call noundef i32 @_ZNSt24uniform_int_distributionIjEclISt23mersenne_twister_engineImLm32ELm624ELm397ELm31ELm2567483615ELm11ELm4294967295ELm7ELm2636928640ELm15ELm4022730752ELm18ELm1812433253EEEEjRT_RKNS0_10param_typeE(ptr noundef nonnull align 4 dereferenceable(8) %0, ptr noundef nonnull align 8 dereferenceable(5000) %1, ptr noundef nonnull align 4 dereferenceable(8) %4)
  %36 = zext i32 %35 to i64
  %37 = shl nuw i64 %36, 32
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #19
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
declare void @_ZSt25__throw_bad_function_callv() local_unnamed_addr #10

; Function Attrs: nofree noreturn nounwind
declare void @exit(i32 noundef) local_unnamed_addr #11

; Function Attrs: nobuiltin nounwind
declare void @_ZdaPv(ptr noundef) local_unnamed_addr #12

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #13

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(write, argmem: readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE3$_0E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr nonnull readnone align 8 captures(none) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %2, align 8, !tbaa !35
  %6 = load i32, ptr %3, align 4, !tbaa !33
  %7 = icmp eq i32 %6, 0
  br i1 %7, label %18, label %8

8:                                                ; preds = %4
  %9 = zext i32 %6 to i64
  br label %10

10:                                               ; preds = %10, %8
  %11 = phi i64 [ 0, %8 ], [ %16, %10 ]
  %12 = phi i32 [ 33, %8 ], [ %13, %10 ]
  %13 = trunc nuw i64 %11 to i32
  %14 = add i32 %12, %13
  %15 = getelementptr inbounds nuw i32, ptr %5, i64 %11
  store i32 %14, ptr %15, align 4, !tbaa !33
  %16 = add nuw nsw i64 %11, 1
  %17 = icmp eq i64 %16, %9
  br i1 %17, label %18, label %10, !llvm.loop !46

18:                                               ; preds = %10, %4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE3$_0E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_0", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !49
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(write, argmem: readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE3$_1E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr nonnull readnone align 8 captures(none) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #14 {
  %5 = load ptr, ptr %2, align 8, !tbaa !35
  %6 = load i32, ptr %3, align 4, !tbaa !33
  %7 = icmp eq i32 %6, 0
  br i1 %7, label %41, label %8

8:                                                ; preds = %4
  %9 = zext i32 %6 to i64
  %10 = icmp ult i32 %6, 8
  br i1 %10, label %30, label %11

11:                                               ; preds = %8
  %12 = and i64 %9, 4294967288
  br label %13

13:                                               ; preds = %13, %11
  %14 = phi i64 [ 0, %11 ], [ %24, %13 ]
  %15 = phi <4 x i32> [ <i32 poison, i32 poison, i32 poison, i32 33>, %11 ], [ %17, %13 ]
  %16 = phi <4 x i32> [ <i32 0, i32 1, i32 2, i32 3>, %11 ], [ %25, %13 ]
  %17 = add <4 x i32> %16, splat (i32 4)
  %18 = shufflevector <4 x i32> %15, <4 x i32> %16, <4 x i32> <i32 3, i32 4, i32 5, i32 6>
  %19 = shufflevector <4 x i32> %16, <4 x i32> %17, <4 x i32> <i32 3, i32 4, i32 5, i32 6>
  %20 = add <4 x i32> %18, %16
  %21 = add <4 x i32> %19, %17
  %22 = getelementptr inbounds nuw i32, ptr %5, i64 %14
  %23 = getelementptr inbounds nuw i8, ptr %22, i64 16
  store <4 x i32> %20, ptr %22, align 4, !tbaa !33
  store <4 x i32> %21, ptr %23, align 4, !tbaa !33
  %24 = add nuw i64 %14, 8
  %25 = add <4 x i32> %16, splat (i32 8)
  %26 = icmp eq i64 %24, %12
  br i1 %26, label %27, label %13, !llvm.loop !50

27:                                               ; preds = %13
  %28 = extractelement <4 x i32> %17, i64 3
  %29 = icmp eq i64 %12, %9
  br i1 %29, label %41, label %30

30:                                               ; preds = %8, %27
  %31 = phi i64 [ 0, %8 ], [ %12, %27 ]
  %32 = phi i32 [ 33, %8 ], [ %28, %27 ]
  br label %33

33:                                               ; preds = %30, %33
  %34 = phi i64 [ %39, %33 ], [ %31, %30 ]
  %35 = phi i32 [ %36, %33 ], [ %32, %30 ]
  %36 = trunc nuw i64 %34 to i32
  %37 = add i32 %35, %36
  %38 = getelementptr inbounds nuw i32, ptr %5, i64 %34
  store i32 %37, ptr %38, align 4, !tbaa !33
  %39 = add nuw nsw i64 %34, 1
  %40 = icmp eq i64 %39, %9
  br i1 %40, label %41, label %33, !llvm.loop !51

41:                                               ; preds = %33, %27, %4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE3$_1E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_1", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !49
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE3$_2E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #16 {
  %5 = load ptr, ptr %1, align 8, !tbaa !35
  %6 = load ptr, ptr %2, align 8, !tbaa !35
  %7 = load i32, ptr %3, align 4, !tbaa !33
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %21, label %9

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 0, %9 ], [ %19, %11 ]
  %13 = phi i32 [ 22, %9 ], [ %14, %11 ]
  %14 = phi i32 [ 33, %9 ], [ %16, %11 ]
  %15 = getelementptr inbounds nuw i32, ptr %5, i64 %12
  %16 = load i32, ptr %15, align 4, !tbaa !33
  %17 = add nsw i32 %14, %13
  %18 = getelementptr inbounds nuw i32, ptr %6, i64 %12
  store i32 %17, ptr %18, align 4, !tbaa !33
  %19 = add nuw nsw i64 %12, 1
  %20 = icmp eq i64 %19, %10
  br i1 %20, label %21, label %11, !llvm.loop !52

21:                                               ; preds = %11, %4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE3$_2E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_2", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !49
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE3$_3E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #16 {
  %5 = load ptr, ptr %1, align 8, !tbaa !35
  %6 = load ptr, ptr %2, align 8, !tbaa !35
  %7 = load i32, ptr %3, align 4, !tbaa !33
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %55, label %9

9:                                                ; preds = %4
  %10 = ptrtoint ptr %6 to i64
  %11 = ptrtoint ptr %5 to i64
  %12 = zext i32 %7 to i64
  %13 = icmp ult i32 %7, 8
  %14 = sub i64 %10, %11
  %15 = icmp ult i64 %14, 32
  %16 = select i1 %13, i1 true, i1 %15
  br i1 %16, label %41, label %17

17:                                               ; preds = %9
  %18 = and i64 %12, 4294967288
  br label %19

19:                                               ; preds = %19, %17
  %20 = phi i64 [ 0, %17 ], [ %35, %19 ]
  %21 = phi <4 x i32> [ <i32 poison, i32 poison, i32 poison, i32 22>, %17 ], [ %28, %19 ]
  %22 = phi <4 x i32> [ <i32 poison, i32 poison, i32 poison, i32 33>, %17 ], [ %26, %19 ]
  %23 = getelementptr inbounds nuw i32, ptr %5, i64 %20
  %24 = getelementptr inbounds nuw i8, ptr %23, i64 16
  %25 = load <4 x i32>, ptr %23, align 4, !tbaa !33
  %26 = load <4 x i32>, ptr %24, align 4, !tbaa !33
  %27 = shufflevector <4 x i32> %22, <4 x i32> %25, <4 x i32> <i32 3, i32 4, i32 5, i32 6>
  %28 = shufflevector <4 x i32> %25, <4 x i32> %26, <4 x i32> <i32 3, i32 4, i32 5, i32 6>
  %29 = shufflevector <4 x i32> %21, <4 x i32> %27, <4 x i32> <i32 3, i32 4, i32 5, i32 6>
  %30 = shufflevector <4 x i32> %25, <4 x i32> %26, <4 x i32> <i32 2, i32 3, i32 4, i32 5>
  %31 = add nsw <4 x i32> %27, %29
  %32 = add nsw <4 x i32> %28, %30
  %33 = getelementptr inbounds nuw i32, ptr %6, i64 %20
  %34 = getelementptr inbounds nuw i8, ptr %33, i64 16
  store <4 x i32> %31, ptr %33, align 4, !tbaa !33
  store <4 x i32> %32, ptr %34, align 4, !tbaa !33
  %35 = add nuw i64 %20, 8
  %36 = icmp eq i64 %35, %18
  br i1 %36, label %37, label %19, !llvm.loop !53

37:                                               ; preds = %19
  %38 = extractelement <4 x i32> %26, i64 2
  %39 = extractelement <4 x i32> %26, i64 3
  %40 = icmp eq i64 %18, %12
  br i1 %40, label %55, label %41

41:                                               ; preds = %9, %37
  %42 = phi i64 [ 0, %9 ], [ %18, %37 ]
  %43 = phi i32 [ 22, %9 ], [ %38, %37 ]
  %44 = phi i32 [ 33, %9 ], [ %39, %37 ]
  br label %45

45:                                               ; preds = %41, %45
  %46 = phi i64 [ %53, %45 ], [ %42, %41 ]
  %47 = phi i32 [ %48, %45 ], [ %43, %41 ]
  %48 = phi i32 [ %50, %45 ], [ %44, %41 ]
  %49 = getelementptr inbounds nuw i32, ptr %5, i64 %46
  %50 = load i32, ptr %49, align 4, !tbaa !33
  %51 = add nsw i32 %48, %47
  %52 = getelementptr inbounds nuw i32, ptr %6, i64 %46
  store i32 %51, ptr %52, align 4, !tbaa !33
  %53 = add nuw nsw i64 %46, 1
  %54 = icmp eq i64 %53, %12
  br i1 %54, label %55, label %45, !llvm.loop !54

55:                                               ; preds = %45, %37, %4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE3$_3E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_3", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !49
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE3$_4E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #16 {
  %5 = load ptr, ptr %1, align 8, !tbaa !35
  %6 = load ptr, ptr %2, align 8, !tbaa !35
  %7 = load i32, ptr %3, align 4, !tbaa !33
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %23, label %9

9:                                                ; preds = %4
  %10 = zext i32 %7 to i64
  br label %11

11:                                               ; preds = %11, %9
  %12 = phi i64 [ 0, %9 ], [ %21, %11 ]
  %13 = phi i32 [ 33, %9 ], [ %17, %11 ]
  %14 = phi i32 [ 22, %9 ], [ %13, %11 ]
  %15 = phi i32 [ 11, %9 ], [ %14, %11 ]
  %16 = getelementptr inbounds nuw i32, ptr %5, i64 %12
  %17 = load i32, ptr %16, align 4, !tbaa !33
  %18 = add i32 %14, %13
  %19 = add i32 %18, %15
  %20 = getelementptr inbounds nuw i32, ptr %6, i64 %12
  store i32 %19, ptr %20, align 4, !tbaa !33
  %21 = add nuw nsw i64 %12, 1
  %22 = icmp eq i64 %21, %10
  br i1 %22, label %23, label %11, !llvm.loop !55

23:                                               ; preds = %11, %4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE3$_4E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_4", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !49
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define internal void @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE3$_5E9_M_invokeERKSt9_Any_dataOS0_S7_Oj"(ptr nonnull readnone align 8 captures(none) %0, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %1, ptr noundef nonnull readonly align 8 captures(none) dereferenceable(8) %2, ptr noundef nonnull readonly align 4 captures(none) dereferenceable(4) %3) #16 {
  %5 = load ptr, ptr %1, align 8, !tbaa !35
  %6 = load ptr, ptr %2, align 8, !tbaa !35
  %7 = load i32, ptr %3, align 4, !tbaa !33
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %64, label %9

9:                                                ; preds = %4
  %10 = ptrtoint ptr %6 to i64
  %11 = ptrtoint ptr %5 to i64
  %12 = zext i32 %7 to i64
  %13 = icmp ult i32 %7, 8
  %14 = sub i64 %10, %11
  %15 = icmp ult i64 %14, 32
  %16 = select i1 %13, i1 true, i1 %15
  br i1 %16, label %47, label %17

17:                                               ; preds = %9
  %18 = and i64 %12, 4294967288
  br label %19

19:                                               ; preds = %19, %17
  %20 = phi i64 [ 0, %17 ], [ %40, %19 ]
  %21 = phi <4 x i32> [ <i32 poison, i32 poison, i32 poison, i32 33>, %17 ], [ %27, %19 ]
  %22 = phi <4 x i32> [ <i32 poison, i32 poison, i32 poison, i32 22>, %17 ], [ %29, %19 ]
  %23 = phi <4 x i32> [ <i32 poison, i32 poison, i32 poison, i32 11>, %17 ], [ %31, %19 ]
  %24 = getelementptr inbounds nuw i32, ptr %5, i64 %20
  %25 = getelementptr inbounds nuw i8, ptr %24, i64 16
  %26 = load <4 x i32>, ptr %24, align 4, !tbaa !33
  %27 = load <4 x i32>, ptr %25, align 4, !tbaa !33
  %28 = shufflevector <4 x i32> %21, <4 x i32> %26, <4 x i32> <i32 3, i32 4, i32 5, i32 6>
  %29 = shufflevector <4 x i32> %26, <4 x i32> %27, <4 x i32> <i32 3, i32 4, i32 5, i32 6>
  %30 = shufflevector <4 x i32> %22, <4 x i32> %28, <4 x i32> <i32 3, i32 4, i32 5, i32 6>
  %31 = shufflevector <4 x i32> %26, <4 x i32> %27, <4 x i32> <i32 2, i32 3, i32 4, i32 5>
  %32 = shufflevector <4 x i32> %23, <4 x i32> %30, <4 x i32> <i32 3, i32 4, i32 5, i32 6>
  %33 = shufflevector <4 x i32> %30, <4 x i32> %31, <4 x i32> <i32 3, i32 4, i32 5, i32 6>
  %34 = add <4 x i32> %30, %28
  %35 = add <4 x i32> %31, %29
  %36 = add <4 x i32> %34, %32
  %37 = add <4 x i32> %35, %33
  %38 = getelementptr inbounds nuw i32, ptr %6, i64 %20
  %39 = getelementptr inbounds nuw i8, ptr %38, i64 16
  store <4 x i32> %36, ptr %38, align 4, !tbaa !33
  store <4 x i32> %37, ptr %39, align 4, !tbaa !33
  %40 = add nuw i64 %20, 8
  %41 = icmp eq i64 %40, %18
  br i1 %41, label %42, label %19, !llvm.loop !56

42:                                               ; preds = %19
  %43 = extractelement <4 x i32> %27, i64 3
  %44 = extractelement <4 x i32> %27, i64 2
  %45 = extractelement <4 x i32> %27, i64 1
  %46 = icmp eq i64 %18, %12
  br i1 %46, label %64, label %47

47:                                               ; preds = %9, %42
  %48 = phi i64 [ 0, %9 ], [ %18, %42 ]
  %49 = phi i32 [ 33, %9 ], [ %43, %42 ]
  %50 = phi i32 [ 22, %9 ], [ %44, %42 ]
  %51 = phi i32 [ 11, %9 ], [ %45, %42 ]
  br label %52

52:                                               ; preds = %47, %52
  %53 = phi i64 [ %62, %52 ], [ %48, %47 ]
  %54 = phi i32 [ %58, %52 ], [ %49, %47 ]
  %55 = phi i32 [ %54, %52 ], [ %50, %47 ]
  %56 = phi i32 [ %55, %52 ], [ %51, %47 ]
  %57 = getelementptr inbounds nuw i32, ptr %5, i64 %53
  %58 = load i32, ptr %57, align 4, !tbaa !33
  %59 = add i32 %55, %54
  %60 = add i32 %59, %56
  %61 = getelementptr inbounds nuw i32, ptr %6, i64 %53
  store i32 %60, ptr %61, align 4, !tbaa !33
  %62 = add nuw nsw i64 %53, 1
  %63 = icmp eq i64 %62, %12
  br i1 %63, label %64, label %52, !llvm.loop !57

64:                                               ; preds = %52, %42, %4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal noundef i1 @"_ZNSt17_Function_handlerIFvPjS0_jEZ4mainE3$_5E10_M_managerERSt9_Any_dataRKS4_St18_Manager_operation"(ptr noundef nonnull writeonly align 8 captures(none) dereferenceable(16) %0, ptr noundef nonnull align 8 dereferenceable(16) %1, i32 noundef %2) #15 personality ptr @__gxx_personality_v0 {
  switch i32 %2, label %7 [
    i32 0, label %5
    i32 1, label %4
  ]

4:                                                ; preds = %3
  br label %5

5:                                                ; preds = %3, %4
  %6 = phi ptr [ %1, %4 ], [ @"_ZTIZ4mainE3$_5", %3 ]
  store ptr %6, ptr %0, align 8, !tbaa !49
  br label %7

7:                                                ; preds = %5, %3
  ret i1 false
}

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define internal void @_GLOBAL__sub_I_recurrences.cpp() #17 section ".text.startup" {
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
attributes #10 = { cold noreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #11 = { nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #12 = { nobuiltin nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #13 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #14 = { mustprogress nofree norecurse nosync nounwind memory(write, argmem: readwrite, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #15 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #16 = { mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #17 = { nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #18 = { nocallback nofree nounwind willreturn memory(argmem: read) }
attributes #19 = { nounwind }
attributes #20 = { noreturn nounwind }
attributes #21 = { builtin allocsize(0) }
attributes #22 = { cold noreturn }
attributes #23 = { cold noreturn nounwind }
attributes #24 = { builtin nounwind }

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
!17 = !{!"_ZTSSt8functionIFvPjS0_jEE", !18, i64 0, !19, i64 24}
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
!46 = distinct !{!46, !11, !47, !48}
!47 = !{!"llvm.loop.vectorize.width", i32 1}
!48 = !{!"llvm.loop.interleave.count", i32 1}
!49 = !{!19, !19, i64 0}
!50 = distinct !{!50, !11, !43, !44}
!51 = distinct !{!51, !11, !44, !43}
!52 = distinct !{!52, !11, !47, !48}
!53 = distinct !{!53, !11, !43, !44}
!54 = distinct !{!54, !11, !43}
!55 = distinct !{!55, !11, !47, !48}
!56 = distinct !{!56, !11, !43, !44}
!57 = distinct !{!57, !11, !43}
