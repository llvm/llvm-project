; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Shootout-C++/wordfreq.cpp'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Shootout-C++/wordfreq.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%"class.std::map" = type { %"class.std::_Rb_tree" }
%"class.std::_Rb_tree" = type { %"struct.std::_Rb_tree<const char *, std::pair<const char *const, int>, std::_Select1st<std::pair<const char *const, int>>, std::less<const char *>>::_Rb_tree_impl" }
%"struct.std::_Rb_tree<const char *, std::pair<const char *const, int>, std::_Select1st<std::pair<const char *const, int>>, std::less<const char *>>::_Rb_tree_impl" = type { [8 x i8], %"struct.std::_Rb_tree_header" }
%"struct.std::_Rb_tree_header" = type { %"struct.std::_Rb_tree_node_base", i64 }
%"struct.std::_Rb_tree_node_base" = type { i32, ptr, ptr, ptr }
%class.word_reader = type { i32, [4097 x i8], ptr, ptr, ptr }
%"struct.std::pair.3" = type <{ ptr, i32, [4 x i8] }>

$_ZNSt3mapIPKciSt4lessIS1_ESaISt4pairIKS1_iEEEixEOS1_ = comdat any

$_ZNSt3mapIPKciSt4lessIS1_ESaISt4pairIKS1_iEEED2Ev = comdat any

$__clang_call_terminate = comdat any

$_ZNSt8_Rb_treeIPKcSt4pairIKS1_iESt10_Select1stIS4_ESt4lessIS1_ESaIS4_EE8_M_eraseEPSt13_Rb_tree_nodeIS4_E = comdat any

$_ZNSt8_Rb_treeIPKcSt4pairIKS1_iESt10_Select1stIS4_ESt4lessIS1_ESaIS4_EE29_M_get_insert_hint_unique_posESt23_Rb_tree_const_iteratorIS4_ERS3_ = comdat any

$_ZSt6__sortIN9__gnu_cxx17__normal_iteratorIPSt4pairIPKciESt6vectorIS5_SaIS5_EEEENS0_5__ops15_Iter_less_iterEEvT_SD_T0_ = comdat any

$_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPSt4pairIPKciESt6vectorIS5_SaIS5_EEEElNS0_5__ops15_Iter_less_iterEEvT_SD_T0_T1_ = comdat any

$_ZSt13__adjust_heapIN9__gnu_cxx17__normal_iteratorIPSt4pairIPKciESt6vectorIS5_SaIS5_EEEElS5_NS0_5__ops15_Iter_less_iterEEvT_T0_SE_T1_T2_ = comdat any

$_ZSt22__move_median_to_firstIN9__gnu_cxx17__normal_iteratorIPSt4pairIPKciESt6vectorIS5_SaIS5_EEEENS0_5__ops15_Iter_less_iterEEvT_SD_SD_SD_T0_ = comdat any

$_ZSt16__insertion_sortIN9__gnu_cxx17__normal_iteratorIPSt4pairIPKciESt6vectorIS5_SaIS5_EEEENS0_5__ops15_Iter_less_iterEEvT_SD_T0_ = comdat any

@stdin = external local_unnamed_addr global ptr, align 8
@.str = private unnamed_addr constant [8 x i8] c"%7d\09%s\0A\00", align 1
@.str.1 = private unnamed_addr constant [49 x i8] c"cannot create std::vector larger than max_size()\00", align 1

; Function Attrs: mustprogress uwtable
define dso_local noundef i32 @_ZN11word_readerclEPPKc(ptr noundef nonnull align 8 dereferenceable(4128) %0, ptr noundef writeonly captures(none) %1) local_unnamed_addr #0 {
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 4
  %4 = getelementptr inbounds nuw i8, ptr %0, i64 4120
  %5 = getelementptr inbounds nuw i8, ptr %0, i64 4104
  %6 = getelementptr inbounds nuw i8, ptr %0, i64 4112
  br label %7

7:                                                ; preds = %45, %2
  %8 = phi i32 [ 0, %2 ], [ %35, %45 ]
  %9 = icmp eq i32 %8, 0
  br label %10

10:                                               ; preds = %7, %46
  %11 = load ptr, ptr %5, align 8, !tbaa !6
  %12 = load i8, ptr %11, align 1, !tbaa !14
  %13 = icmp eq i8 %12, 0
  br i1 %13, label %14, label %22

14:                                               ; preds = %10
  %15 = load ptr, ptr %4, align 8, !tbaa !15
  %16 = tail call i64 @fread(ptr noundef nonnull %3, i64 noundef 1, i64 noundef 4096, ptr noundef %15)
  %17 = trunc i64 %16 to i32
  %18 = shl i64 %16, 32
  %19 = ashr exact i64 %18, 32
  %20 = getelementptr inbounds i8, ptr %3, i64 %19
  store i8 0, ptr %20, align 1, !tbaa !14
  store ptr %3, ptr %5, align 8, !tbaa !6
  %21 = icmp sgt i32 %17, 0
  br i1 %21, label %22, label %47

22:                                               ; preds = %10, %14
  %23 = phi ptr [ %11, %10 ], [ %3, %14 ]
  %24 = getelementptr inbounds nuw i8, ptr %23, i64 1
  store ptr %24, ptr %5, align 8, !tbaa !6
  %25 = load i8, ptr %23, align 1, !tbaa !14
  %26 = zext i8 %25 to i32
  %27 = tail call i32 @isalpha(i32 noundef %26) #17
  %28 = icmp eq i32 %27, 0
  br i1 %28, label %46, label %29

29:                                               ; preds = %22
  %30 = tail call i32 @tolower(i32 noundef %26) #17
  %31 = trunc i32 %30 to i8
  %32 = load ptr, ptr %6, align 8, !tbaa !16
  %33 = zext nneg i32 %8 to i64
  %34 = getelementptr inbounds nuw i8, ptr %32, i64 %33
  store i8 %31, ptr %34, align 1, !tbaa !14
  %35 = add nuw nsw i32 %8, 1
  %36 = load i32, ptr %0, align 8, !tbaa !17
  %37 = icmp eq i32 %35, %36
  br i1 %37, label %38, label %45

38:                                               ; preds = %29
  %39 = shl nuw nsw i32 %35, 1
  store i32 %39, ptr %0, align 8, !tbaa !17
  %40 = or disjoint i32 %39, 1
  %41 = zext nneg i32 %40 to i64
  %42 = tail call noalias noundef nonnull ptr @_Znam(i64 noundef %41) #18
  %43 = load ptr, ptr %6, align 8, !tbaa !16
  %44 = zext nneg i32 %35 to i64
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(1) %42, ptr noundef nonnull align 1 dereferenceable(1) %43, i64 %44, i1 false)
  tail call void @_ZdaPv(ptr noundef nonnull %43) #19
  store ptr %42, ptr %6, align 8, !tbaa !16
  br label %45

45:                                               ; preds = %38, %29
  br label %7, !llvm.loop !18

46:                                               ; preds = %22
  br i1 %9, label %10, label %47, !llvm.loop !18

47:                                               ; preds = %46, %14
  %48 = load ptr, ptr %6, align 8, !tbaa !16
  store ptr %48, ptr %1, align 8, !tbaa !20
  %49 = zext nneg i32 %8 to i64
  %50 = getelementptr inbounds nuw i8, ptr %48, i64 %49
  store i8 0, ptr %50, align 1, !tbaa !14
  ret i32 %8
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nofree nounwind willreturn memory(read)
declare i32 @isalpha(i32 noundef) local_unnamed_addr #2

; Function Attrs: mustprogress nofree nounwind willreturn memory(read)
declare i32 @tolower(i32 noundef) local_unnamed_addr #2

; Function Attrs: nobuiltin allocsize(0)
declare noundef nonnull ptr @_Znam(i64 noundef) local_unnamed_addr #3

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #4

; Function Attrs: nobuiltin nounwind
declare void @_ZdaPv(ptr noundef) local_unnamed_addr #5

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: mustprogress norecurse uwtable
define dso_local noundef i32 @main() local_unnamed_addr #6 personality ptr @__gxx_personality_v0 {
  %1 = alloca ptr, align 8
  %2 = alloca %"class.std::map", align 8
  %3 = alloca %class.word_reader, align 8
  %4 = alloca ptr, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #20
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #20
  %5 = getelementptr inbounds nuw i8, ptr %2, i64 8
  store i32 0, ptr %5, align 8, !tbaa !21
  %6 = getelementptr inbounds nuw i8, ptr %2, i64 16
  store ptr null, ptr %6, align 8, !tbaa !27
  %7 = getelementptr inbounds nuw i8, ptr %2, i64 24
  store ptr %5, ptr %7, align 8, !tbaa !28
  %8 = getelementptr inbounds nuw i8, ptr %2, i64 32
  store ptr %5, ptr %8, align 8, !tbaa !29
  %9 = getelementptr inbounds nuw i8, ptr %2, i64 40
  store i64 0, ptr %9, align 8, !tbaa !30
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #20
  %10 = load ptr, ptr @stdin, align 8, !tbaa !31
  store i32 64, ptr %3, align 8, !tbaa !17
  %11 = getelementptr inbounds nuw i8, ptr %3, i64 4104
  %12 = getelementptr inbounds nuw i8, ptr %3, i64 4
  store ptr %12, ptr %11, align 8, !tbaa !6
  %13 = invoke noalias noundef nonnull dereferenceable(65) ptr @_Znam(i64 noundef 65) #18
          to label %14 unwind label %55

14:                                               ; preds = %0
  %15 = getelementptr inbounds nuw i8, ptr %3, i64 4112
  store ptr %13, ptr %15, align 8, !tbaa !16
  %16 = getelementptr inbounds nuw i8, ptr %3, i64 4120
  store ptr %10, ptr %16, align 8, !tbaa !15
  store i8 0, ptr %13, align 1, !tbaa !14
  store i8 0, ptr %12, align 4, !tbaa !14
  br label %17

17:                                               ; preds = %63, %14
  %18 = invoke noundef i32 @_ZN11word_readerclEPPKc(ptr noundef nonnull align 8 dereferenceable(4128) %3, ptr noundef nonnull %1)
          to label %19 unwind label %53

19:                                               ; preds = %17
  %20 = icmp sgt i32 %18, 0
  br i1 %20, label %21, label %64

21:                                               ; preds = %19
  %22 = load ptr, ptr %6, align 8, !tbaa !27
  %23 = icmp eq ptr %22, null
  br i1 %23, label %44, label %24

24:                                               ; preds = %21
  %25 = load ptr, ptr %1, align 8, !tbaa !20
  br label %26

26:                                               ; preds = %26, %24
  %27 = phi ptr [ %22, %24 ], [ %35, %26 ]
  %28 = phi ptr [ %5, %24 ], [ %32, %26 ]
  %29 = getelementptr inbounds nuw i8, ptr %27, i64 32
  %30 = load ptr, ptr %29, align 8, !tbaa !20
  %31 = icmp ult ptr %30, %25
  %32 = select i1 %31, ptr %28, ptr %27
  %33 = select i1 %31, i64 24, i64 16
  %34 = getelementptr inbounds nuw i8, ptr %27, i64 %33
  %35 = load ptr, ptr %34, align 8, !tbaa !32
  %36 = icmp eq ptr %35, null
  br i1 %36, label %37, label %26, !llvm.loop !33

37:                                               ; preds = %26
  %38 = icmp eq ptr %32, %5
  br i1 %38, label %44, label %39

39:                                               ; preds = %37
  %40 = select i1 %31, ptr %28, ptr %27
  %41 = getelementptr inbounds nuw i8, ptr %40, i64 32
  %42 = load ptr, ptr %41, align 8, !tbaa !20
  %43 = icmp ult ptr %25, %42
  br i1 %43, label %44, label %59

44:                                               ; preds = %21, %37, %39
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #20
  %45 = add nuw nsw i32 %18, 1
  %46 = zext nneg i32 %45 to i64
  %47 = invoke noalias noundef nonnull ptr @_Znam(i64 noundef %46) #18
          to label %48 unwind label %57

48:                                               ; preds = %44
  %49 = load ptr, ptr %1, align 8, !tbaa !20
  %50 = call ptr @strcpy(ptr noundef nonnull dereferenceable(1) %47, ptr noundef nonnull dereferenceable(1) %49) #20
  store ptr %47, ptr %4, align 8, !tbaa !20
  %51 = invoke noundef nonnull align 4 dereferenceable(4) ptr @_ZNSt3mapIPKciSt4lessIS1_ESaISt4pairIKS1_iEEEixEOS1_(ptr noundef nonnull align 8 dereferenceable(48) %2, ptr noundef nonnull align 8 dereferenceable(8) %4)
          to label %52 unwind label %57

52:                                               ; preds = %48
  store i32 1, ptr %51, align 4, !tbaa !34
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #20
  br label %63

53:                                               ; preds = %17
  %54 = landingpad { ptr, i32 }
          cleanup
  br label %131

55:                                               ; preds = %0
  %56 = landingpad { ptr, i32 }
          cleanup
  br label %131

57:                                               ; preds = %48, %44
  %58 = landingpad { ptr, i32 }
          cleanup
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #20
  br label %131

59:                                               ; preds = %39
  %60 = getelementptr inbounds nuw i8, ptr %32, i64 40
  %61 = load i32, ptr %60, align 8, !tbaa !35
  %62 = add nsw i32 %61, 1
  store i32 %62, ptr %60, align 8, !tbaa !35
  br label %63

63:                                               ; preds = %59, %52
  br label %17, !llvm.loop !37

64:                                               ; preds = %19
  %65 = load ptr, ptr %7, align 8, !tbaa !28
  %66 = icmp eq ptr %65, %5
  br i1 %66, label %96, label %67

67:                                               ; preds = %64, %67
  %68 = phi i64 [ %71, %67 ], [ 0, %64 ]
  %69 = phi ptr [ %70, %67 ], [ %65, %64 ]
  %70 = call noundef ptr @_ZSt18_Rb_tree_incrementPSt18_Rb_tree_node_base(ptr noundef %69) #17
  %71 = add nuw nsw i64 %68, 1
  %72 = icmp eq ptr %70, %5
  br i1 %72, label %73, label %67, !llvm.loop !38

73:                                               ; preds = %67
  %74 = icmp samesign ugt i64 %68, 576460752303423486
  br i1 %74, label %75, label %77

75:                                               ; preds = %73
  invoke void @_ZSt20__throw_length_errorPKc(ptr noundef nonnull @.str.1) #21
          to label %76 unwind label %91

76:                                               ; preds = %75
  unreachable

77:                                               ; preds = %73
  %78 = shl nuw nsw i64 %71, 4
  %79 = invoke noalias noundef nonnull ptr @_Znwm(i64 noundef %78) #18
          to label %80 unwind label %91

80:                                               ; preds = %77, %80
  %81 = phi ptr [ %89, %80 ], [ %79, %77 ]
  %82 = phi ptr [ %88, %80 ], [ %65, %77 ]
  %83 = getelementptr inbounds nuw i8, ptr %82, i64 32
  %84 = load ptr, ptr %83, align 8, !tbaa !39
  store ptr %84, ptr %81, align 8, !tbaa !40
  %85 = getelementptr inbounds nuw i8, ptr %81, i64 8
  %86 = getelementptr inbounds nuw i8, ptr %82, i64 40
  %87 = load i32, ptr %86, align 8, !tbaa !35
  store i32 %87, ptr %85, align 8, !tbaa !42
  %88 = call noundef ptr @_ZSt18_Rb_tree_incrementPSt18_Rb_tree_node_base(ptr noundef %82) #17
  %89 = getelementptr inbounds nuw i8, ptr %81, i64 16
  %90 = icmp eq ptr %88, %5
  br i1 %90, label %93, label %80, !llvm.loop !43

91:                                               ; preds = %75, %77
  %92 = landingpad { ptr, i32 }
          cleanup
  br label %131

93:                                               ; preds = %80
  %94 = getelementptr inbounds nuw %"struct.std::pair.3", ptr %79, i64 %71
  %95 = ptrtoint ptr %94 to i64
  br label %96

96:                                               ; preds = %93, %64
  %97 = phi i64 [ 0, %64 ], [ %95, %93 ]
  %98 = phi ptr [ null, %64 ], [ %79, %93 ]
  %99 = phi ptr [ null, %64 ], [ %89, %93 ]
  %100 = ptrtoint ptr %98 to i64
  invoke void @_ZSt6__sortIN9__gnu_cxx17__normal_iteratorIPSt4pairIPKciESt6vectorIS5_SaIS5_EEEENS0_5__ops15_Iter_less_iterEEvT_SD_T0_(ptr %98, ptr %99, i8 undef)
          to label %101 unwind label %117

101:                                              ; preds = %96
  %102 = icmp eq ptr %99, %98
  br i1 %102, label %107, label %103

103:                                              ; preds = %101
  %104 = ptrtoint ptr %99 to i64
  %105 = sub i64 %104, %100
  %106 = ashr exact i64 %105, 4
  br label %122

107:                                              ; preds = %101
  %108 = icmp eq ptr %98, null
  br i1 %108, label %111, label %109

109:                                              ; preds = %122, %107
  %110 = sub i64 %97, %100
  call void @_ZdlPvm(ptr noundef nonnull %98, i64 noundef %110) #19
  br label %111

111:                                              ; preds = %107, %109
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #20
  %112 = load ptr, ptr %6, align 8, !tbaa !27
  invoke void @_ZNSt8_Rb_treeIPKcSt4pairIKS1_iESt10_Select1stIS4_ESt4lessIS1_ESaIS4_EE8_M_eraseEPSt13_Rb_tree_nodeIS4_E(ptr noundef nonnull align 8 dereferenceable(48) %2, ptr noundef %112)
          to label %116 unwind label %113

113:                                              ; preds = %111
  %114 = landingpad { ptr, i32 }
          catch ptr null
  %115 = extractvalue { ptr, i32 } %114, 0
  call void @__clang_call_terminate(ptr %115) #22
  unreachable

116:                                              ; preds = %111
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #20
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #20
  ret i32 0

117:                                              ; preds = %96
  %118 = landingpad { ptr, i32 }
          cleanup
  %119 = icmp eq ptr %98, null
  br i1 %119, label %131, label %120

120:                                              ; preds = %117
  %121 = sub i64 %97, %100
  call void @_ZdlPvm(ptr noundef nonnull %98, i64 noundef %121) #19
  br label %131

122:                                              ; preds = %103, %122
  %123 = phi i64 [ %129, %122 ], [ 0, %103 ]
  %124 = getelementptr inbounds nuw %"struct.std::pair.3", ptr %98, i64 %123
  %125 = getelementptr inbounds nuw i8, ptr %124, i64 8
  %126 = load i32, ptr %125, align 8, !tbaa !42
  %127 = load ptr, ptr %124, align 8, !tbaa !40
  %128 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %126, ptr noundef %127)
  %129 = add nuw i64 %123, 1
  %130 = icmp eq i64 %129, %106
  br i1 %130, label %109, label %122, !llvm.loop !44

131:                                              ; preds = %53, %55, %91, %117, %120, %57
  %132 = phi { ptr, i32 } [ %58, %57 ], [ %92, %91 ], [ %118, %117 ], [ %118, %120 ], [ %54, %53 ], [ %56, %55 ]
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #20
  call void @_ZNSt3mapIPKciSt4lessIS1_ESaISt4pairIKS1_iEEED2Ev(ptr noundef nonnull align 8 dereferenceable(48) %2) #20
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #20
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #20
  resume { ptr, i32 } %132
}

declare i32 @__gxx_personality_v0(...)

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local noundef nonnull align 4 dereferenceable(4) ptr @_ZNSt3mapIPKciSt4lessIS1_ESaISt4pairIKS1_iEEEixEOS1_(ptr noundef nonnull align 8 dereferenceable(48) %0, ptr noundef nonnull align 8 dereferenceable(8) %1) local_unnamed_addr #0 comdat personality ptr @__gxx_personality_v0 {
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %4 = load ptr, ptr %3, align 8, !tbaa !27
  %5 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %6 = icmp eq ptr %4, null
  %7 = load ptr, ptr %1, align 8, !tbaa !20
  br i1 %6, label %25, label %8

8:                                                ; preds = %2, %8
  %9 = phi ptr [ %17, %8 ], [ %4, %2 ]
  %10 = phi ptr [ %14, %8 ], [ %5, %2 ]
  %11 = getelementptr inbounds nuw i8, ptr %9, i64 32
  %12 = load ptr, ptr %11, align 8, !tbaa !20
  %13 = icmp ult ptr %12, %7
  %14 = select i1 %13, ptr %10, ptr %9
  %15 = select i1 %13, i64 24, i64 16
  %16 = getelementptr inbounds nuw i8, ptr %9, i64 %15
  %17 = load ptr, ptr %16, align 8, !tbaa !32
  %18 = icmp eq ptr %17, null
  br i1 %18, label %19, label %8, !llvm.loop !33

19:                                               ; preds = %8
  %20 = icmp eq ptr %14, %5
  br i1 %20, label %25, label %21

21:                                               ; preds = %19
  %22 = getelementptr inbounds nuw i8, ptr %14, i64 32
  %23 = load ptr, ptr %22, align 8, !tbaa !39
  %24 = icmp ult ptr %7, %23
  br i1 %24, label %25, label %54

25:                                               ; preds = %2, %19, %21
  %26 = phi ptr [ %14, %21 ], [ %14, %19 ], [ %5, %2 ]
  %27 = tail call noalias noundef nonnull dereferenceable(48) ptr @_Znwm(i64 noundef 48) #18
  %28 = getelementptr inbounds nuw i8, ptr %27, i64 32
  store ptr %7, ptr %28, align 8, !tbaa !39
  %29 = getelementptr inbounds nuw i8, ptr %27, i64 40
  store i32 0, ptr %29, align 8, !tbaa !35
  %30 = invoke [2 x i64] @_ZNSt8_Rb_treeIPKcSt4pairIKS1_iESt10_Select1stIS4_ESt4lessIS1_ESaIS4_EE29_M_get_insert_hint_unique_posESt23_Rb_tree_const_iteratorIS4_ERS3_(ptr noundef nonnull align 8 dereferenceable(48) %0, ptr %26, ptr noundef nonnull align 8 dereferenceable(8) %28)
          to label %31 unwind label %50

31:                                               ; preds = %25
  %32 = extractvalue [2 x i64] %30, 1
  %33 = icmp eq i64 %32, 0
  %34 = extractvalue [2 x i64] %30, 0
  br i1 %33, label %52, label %35

35:                                               ; preds = %31
  %36 = inttoptr i64 %32 to ptr
  %37 = icmp ne i64 %34, 0
  %38 = icmp eq ptr %5, %36
  %39 = select i1 %37, i1 true, i1 %38
  br i1 %39, label %45, label %40

40:                                               ; preds = %35
  %41 = load ptr, ptr %28, align 8, !tbaa !20
  %42 = getelementptr inbounds nuw i8, ptr %36, i64 32
  %43 = load ptr, ptr %42, align 8, !tbaa !20
  %44 = icmp ult ptr %41, %43
  br label %45

45:                                               ; preds = %40, %35
  %46 = phi i1 [ true, %35 ], [ %44, %40 ]
  tail call void @_ZSt29_Rb_tree_insert_and_rebalancebPSt18_Rb_tree_node_baseS0_RS_(i1 noundef %46, ptr noundef nonnull %27, ptr noundef nonnull %36, ptr noundef nonnull align 8 dereferenceable(32) %5) #20
  %47 = getelementptr inbounds nuw i8, ptr %0, i64 40
  %48 = load i64, ptr %47, align 8, !tbaa !30
  %49 = add i64 %48, 1
  store i64 %49, ptr %47, align 8, !tbaa !30
  br label %54

50:                                               ; preds = %25
  %51 = landingpad { ptr, i32 }
          cleanup
  tail call void @_ZdlPvm(ptr noundef nonnull %27, i64 noundef 48) #19
  resume { ptr, i32 } %51

52:                                               ; preds = %31
  tail call void @_ZdlPvm(ptr noundef nonnull %27, i64 noundef 48) #19
  %53 = inttoptr i64 %34 to ptr
  br label %54

54:                                               ; preds = %52, %45, %21
  %55 = phi ptr [ %14, %21 ], [ %27, %45 ], [ %53, %52 ]
  %56 = getelementptr inbounds nuw i8, ptr %55, i64 40
  ret ptr %56
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare ptr @strcpy(ptr noalias noundef returned writeonly, ptr noalias noundef readonly captures(none)) local_unnamed_addr #7

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #8

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local void @_ZNSt3mapIPKciSt4lessIS1_ESaISt4pairIKS1_iEEED2Ev(ptr noundef nonnull align 8 dereferenceable(48) %0) unnamed_addr #9 comdat personality ptr @__gxx_personality_v0 {
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %3 = load ptr, ptr %2, align 8, !tbaa !27
  invoke void @_ZNSt8_Rb_treeIPKcSt4pairIKS1_iESt10_Select1stIS4_ESt4lessIS1_ESaIS4_EE8_M_eraseEPSt13_Rb_tree_nodeIS4_E(ptr noundef nonnull align 8 dereferenceable(48) %0, ptr noundef %3)
          to label %7 unwind label %4

4:                                                ; preds = %1
  %5 = landingpad { ptr, i32 }
          catch ptr null
  %6 = extractvalue { ptr, i32 } %5, 0
  tail call void @__clang_call_terminate(ptr %6) #22
  unreachable

7:                                                ; preds = %1
  ret void
}

; Function Attrs: nofree nounwind
declare noundef i64 @fread(ptr noundef writeonly captures(none), i64 noundef, i64 noundef, ptr noundef captures(none)) local_unnamed_addr #8

; Function Attrs: noinline noreturn nounwind uwtable
define linkonce_odr hidden void @__clang_call_terminate(ptr noundef %0) local_unnamed_addr #10 comdat {
  %2 = tail call ptr @__cxa_begin_catch(ptr %0) #20
  tail call void @_ZSt9terminatev() #22
  unreachable
}

declare ptr @__cxa_begin_catch(ptr) local_unnamed_addr

; Function Attrs: cold nofree noreturn
declare void @_ZSt9terminatev() local_unnamed_addr #11

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZNSt8_Rb_treeIPKcSt4pairIKS1_iESt10_Select1stIS4_ESt4lessIS1_ESaIS4_EE8_M_eraseEPSt13_Rb_tree_nodeIS4_E(ptr noundef nonnull align 8 dereferenceable(48) %0, ptr noundef %1) local_unnamed_addr #0 comdat personality ptr @__gxx_personality_v0 {
  %3 = icmp eq ptr %1, null
  br i1 %3, label %11, label %4

4:                                                ; preds = %2, %4
  %5 = phi ptr [ %9, %4 ], [ %1, %2 ]
  %6 = getelementptr inbounds nuw i8, ptr %5, i64 24
  %7 = load ptr, ptr %6, align 8, !tbaa !45
  tail call void @_ZNSt8_Rb_treeIPKcSt4pairIKS1_iESt10_Select1stIS4_ESt4lessIS1_ESaIS4_EE8_M_eraseEPSt13_Rb_tree_nodeIS4_E(ptr noundef nonnull align 8 dereferenceable(48) %0, ptr noundef %7)
  %8 = getelementptr inbounds nuw i8, ptr %5, i64 16
  %9 = load ptr, ptr %8, align 8, !tbaa !46
  tail call void @_ZdlPvm(ptr noundef nonnull %5, i64 noundef 48) #19
  %10 = icmp eq ptr %9, null
  br i1 %10, label %11, label %4, !llvm.loop !47

11:                                               ; preds = %4, %2
  ret void
}

; Function Attrs: nobuiltin nounwind
declare void @_ZdlPvm(ptr noundef, i64 noundef) local_unnamed_addr #5

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local [2 x i64] @_ZNSt8_Rb_treeIPKcSt4pairIKS1_iESt10_Select1stIS4_ESt4lessIS1_ESaIS4_EE29_M_get_insert_hint_unique_posESt23_Rb_tree_const_iteratorIS4_ERS3_(ptr noundef nonnull align 8 dereferenceable(48) %0, ptr %1, ptr noundef nonnull align 8 dereferenceable(8) %2) local_unnamed_addr #0 comdat {
  %4 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %5 = icmp eq ptr %4, %1
  br i1 %5, label %6, label %51

6:                                                ; preds = %3
  %7 = getelementptr inbounds nuw i8, ptr %0, i64 40
  %8 = load i64, ptr %7, align 8, !tbaa !30
  %9 = icmp eq i64 %8, 0
  br i1 %9, label %17, label %10

10:                                               ; preds = %6
  %11 = getelementptr inbounds nuw i8, ptr %0, i64 32
  %12 = load ptr, ptr %11, align 8, !tbaa !32
  %13 = getelementptr inbounds nuw i8, ptr %12, i64 32
  %14 = load ptr, ptr %13, align 8, !tbaa !20
  %15 = load ptr, ptr %2, align 8, !tbaa !20
  %16 = icmp ult ptr %14, %15
  br i1 %16, label %146, label %17

17:                                               ; preds = %10, %6
  %18 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %19 = load ptr, ptr %18, align 8, !tbaa !32
  %20 = icmp eq ptr %19, null
  br i1 %20, label %33, label %21

21:                                               ; preds = %17
  %22 = load ptr, ptr %2, align 8, !tbaa !20
  br label %23

23:                                               ; preds = %23, %21
  %24 = phi ptr [ %19, %21 ], [ %30, %23 ]
  %25 = getelementptr inbounds nuw i8, ptr %24, i64 32
  %26 = load ptr, ptr %25, align 8, !tbaa !20
  %27 = icmp ult ptr %22, %26
  %28 = select i1 %27, i64 16, i64 24
  %29 = getelementptr inbounds nuw i8, ptr %24, i64 %28
  %30 = load ptr, ptr %29, align 8, !tbaa !32
  %31 = icmp eq ptr %30, null
  br i1 %31, label %32, label %23, !llvm.loop !48

32:                                               ; preds = %23
  br i1 %27, label %33, label %43

33:                                               ; preds = %32, %17
  %34 = phi ptr [ %24, %32 ], [ %4, %17 ]
  %35 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %36 = load ptr, ptr %35, align 8, !tbaa !28
  %37 = icmp eq ptr %34, %36
  br i1 %37, label %146, label %38

38:                                               ; preds = %33
  %39 = tail call noundef ptr @_ZSt18_Rb_tree_decrementPSt18_Rb_tree_node_base(ptr noundef nonnull %34) #17
  %40 = getelementptr inbounds nuw i8, ptr %39, i64 32
  %41 = load ptr, ptr %40, align 8, !tbaa !20
  %42 = load ptr, ptr %2, align 8, !tbaa !20
  br label %43

43:                                               ; preds = %38, %32
  %44 = phi ptr [ %42, %38 ], [ %22, %32 ]
  %45 = phi ptr [ %41, %38 ], [ %26, %32 ]
  %46 = phi ptr [ %34, %38 ], [ %24, %32 ]
  %47 = phi ptr [ %39, %38 ], [ %24, %32 ]
  %48 = icmp ult ptr %45, %44
  %49 = select i1 %48, ptr null, ptr %47
  %50 = select i1 %48, ptr %46, ptr null
  br label %146

51:                                               ; preds = %3
  %52 = load ptr, ptr %2, align 8, !tbaa !20
  %53 = getelementptr inbounds nuw i8, ptr %1, i64 32
  %54 = load ptr, ptr %53, align 8, !tbaa !20
  %55 = icmp ult ptr %52, %54
  br i1 %55, label %56, label %99

56:                                               ; preds = %51
  %57 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %58 = load ptr, ptr %57, align 8, !tbaa !32
  %59 = icmp eq ptr %58, %1
  br i1 %59, label %146, label %60

60:                                               ; preds = %56
  %61 = tail call noundef ptr @_ZSt18_Rb_tree_decrementPSt18_Rb_tree_node_base(ptr noundef nonnull %1) #17
  %62 = getelementptr inbounds nuw i8, ptr %61, i64 32
  %63 = load ptr, ptr %62, align 8, !tbaa !20
  %64 = icmp ult ptr %63, %52
  br i1 %64, label %65, label %71

65:                                               ; preds = %60
  %66 = getelementptr inbounds nuw i8, ptr %61, i64 24
  %67 = load ptr, ptr %66, align 8, !tbaa !45
  %68 = icmp eq ptr %67, null
  %69 = select i1 %68, ptr null, ptr %1
  %70 = select i1 %68, ptr %61, ptr %1
  br label %146

71:                                               ; preds = %60
  %72 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %73 = load ptr, ptr %72, align 8, !tbaa !32
  %74 = icmp eq ptr %73, null
  br i1 %74, label %85, label %75

75:                                               ; preds = %71, %75
  %76 = phi ptr [ %82, %75 ], [ %73, %71 ]
  %77 = getelementptr inbounds nuw i8, ptr %76, i64 32
  %78 = load ptr, ptr %77, align 8, !tbaa !20
  %79 = icmp ult ptr %52, %78
  %80 = select i1 %79, i64 16, i64 24
  %81 = getelementptr inbounds nuw i8, ptr %76, i64 %80
  %82 = load ptr, ptr %81, align 8, !tbaa !32
  %83 = icmp eq ptr %82, null
  br i1 %83, label %84, label %75, !llvm.loop !48

84:                                               ; preds = %75
  br i1 %79, label %85, label %92

85:                                               ; preds = %84, %71
  %86 = phi ptr [ %76, %84 ], [ %4, %71 ]
  %87 = icmp eq ptr %86, %58
  br i1 %87, label %146, label %88

88:                                               ; preds = %85
  %89 = tail call noundef ptr @_ZSt18_Rb_tree_decrementPSt18_Rb_tree_node_base(ptr noundef nonnull %86) #17
  %90 = getelementptr inbounds nuw i8, ptr %89, i64 32
  %91 = load ptr, ptr %90, align 8, !tbaa !20
  br label %92

92:                                               ; preds = %88, %84
  %93 = phi ptr [ %91, %88 ], [ %78, %84 ]
  %94 = phi ptr [ %86, %88 ], [ %76, %84 ]
  %95 = phi ptr [ %89, %88 ], [ %76, %84 ]
  %96 = icmp ult ptr %93, %52
  %97 = select i1 %96, ptr null, ptr %95
  %98 = select i1 %96, ptr %94, ptr null
  br label %146

99:                                               ; preds = %51
  %100 = icmp ult ptr %54, %52
  br i1 %100, label %101, label %146

101:                                              ; preds = %99
  %102 = getelementptr inbounds nuw i8, ptr %0, i64 32
  %103 = load ptr, ptr %102, align 8, !tbaa !32
  %104 = icmp eq ptr %103, %1
  br i1 %104, label %146, label %105

105:                                              ; preds = %101
  %106 = tail call noundef ptr @_ZSt18_Rb_tree_incrementPSt18_Rb_tree_node_base(ptr noundef nonnull %1) #17
  %107 = getelementptr inbounds nuw i8, ptr %106, i64 32
  %108 = load ptr, ptr %107, align 8, !tbaa !20
  %109 = icmp ult ptr %52, %108
  br i1 %109, label %110, label %116

110:                                              ; preds = %105
  %111 = getelementptr inbounds nuw i8, ptr %1, i64 24
  %112 = load ptr, ptr %111, align 8, !tbaa !45
  %113 = icmp eq ptr %112, null
  %114 = select i1 %113, ptr null, ptr %106
  %115 = select i1 %113, ptr %1, ptr %106
  br label %146

116:                                              ; preds = %105
  %117 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %118 = load ptr, ptr %117, align 8, !tbaa !32
  %119 = icmp eq ptr %118, null
  br i1 %119, label %130, label %120

120:                                              ; preds = %116, %120
  %121 = phi ptr [ %127, %120 ], [ %118, %116 ]
  %122 = getelementptr inbounds nuw i8, ptr %121, i64 32
  %123 = load ptr, ptr %122, align 8, !tbaa !20
  %124 = icmp ult ptr %52, %123
  %125 = select i1 %124, i64 16, i64 24
  %126 = getelementptr inbounds nuw i8, ptr %121, i64 %125
  %127 = load ptr, ptr %126, align 8, !tbaa !32
  %128 = icmp eq ptr %127, null
  br i1 %128, label %129, label %120, !llvm.loop !48

129:                                              ; preds = %120
  br i1 %124, label %130, label %139

130:                                              ; preds = %129, %116
  %131 = phi ptr [ %121, %129 ], [ %4, %116 ]
  %132 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %133 = load ptr, ptr %132, align 8, !tbaa !28
  %134 = icmp eq ptr %131, %133
  br i1 %134, label %146, label %135

135:                                              ; preds = %130
  %136 = tail call noundef ptr @_ZSt18_Rb_tree_decrementPSt18_Rb_tree_node_base(ptr noundef nonnull %131) #17
  %137 = getelementptr inbounds nuw i8, ptr %136, i64 32
  %138 = load ptr, ptr %137, align 8, !tbaa !20
  br label %139

139:                                              ; preds = %135, %129
  %140 = phi ptr [ %138, %135 ], [ %123, %129 ]
  %141 = phi ptr [ %131, %135 ], [ %121, %129 ]
  %142 = phi ptr [ %136, %135 ], [ %121, %129 ]
  %143 = icmp ult ptr %140, %52
  %144 = select i1 %143, ptr null, ptr %142
  %145 = select i1 %143, ptr %141, ptr null
  br label %146

146:                                              ; preds = %139, %130, %92, %85, %43, %33, %110, %65, %99, %101, %56, %10
  %147 = phi ptr [ null, %10 ], [ %1, %56 ], [ null, %101 ], [ %1, %99 ], [ %69, %65 ], [ %114, %110 ], [ null, %33 ], [ %49, %43 ], [ null, %85 ], [ %97, %92 ], [ null, %130 ], [ %144, %139 ]
  %148 = phi ptr [ %12, %10 ], [ %1, %56 ], [ %1, %101 ], [ null, %99 ], [ %70, %65 ], [ %115, %110 ], [ %34, %33 ], [ %50, %43 ], [ %58, %85 ], [ %98, %92 ], [ %131, %130 ], [ %145, %139 ]
  %149 = ptrtoint ptr %147 to i64
  %150 = insertvalue [2 x i64] poison, i64 %149, 0
  %151 = ptrtoint ptr %148 to i64
  %152 = insertvalue [2 x i64] %150, i64 %151, 1
  ret [2 x i64] %152
}

; Function Attrs: nobuiltin allocsize(0)
declare noundef nonnull ptr @_Znwm(i64 noundef) local_unnamed_addr #3

; Function Attrs: mustprogress nofree nounwind willreturn memory(read)
declare noundef ptr @_ZSt18_Rb_tree_decrementPSt18_Rb_tree_node_base(ptr noundef) local_unnamed_addr #2

; Function Attrs: mustprogress nofree nounwind willreturn memory(read)
declare noundef ptr @_ZSt18_Rb_tree_incrementPSt18_Rb_tree_node_base(ptr noundef) local_unnamed_addr #2

; Function Attrs: nounwind
declare void @_ZSt29_Rb_tree_insert_and_rebalancebPSt18_Rb_tree_node_baseS0_RS_(i1 noundef, ptr noundef, ptr noundef, ptr noundef nonnull align 8 dereferenceable(32)) local_unnamed_addr #12

; Function Attrs: cold noreturn
declare void @_ZSt20__throw_length_errorPKc(ptr noundef) local_unnamed_addr #13

; Function Attrs: inlinehint mustprogress uwtable
define linkonce_odr dso_local void @_ZSt6__sortIN9__gnu_cxx17__normal_iteratorIPSt4pairIPKciESt6vectorIS5_SaIS5_EEEENS0_5__ops15_Iter_less_iterEEvT_SD_T0_(ptr %0, ptr %1, i8 %2) local_unnamed_addr #14 comdat {
  %4 = icmp eq ptr %0, %1
  br i1 %4, label %103, label %5

5:                                                ; preds = %3
  %6 = ptrtoint ptr %1 to i64
  %7 = ptrtoint ptr %0 to i64
  %8 = sub i64 %6, %7
  %9 = ashr exact i64 %8, 4
  %10 = tail call range(i64 0, 65) i64 @llvm.ctlz.i64(i64 %9, i1 true)
  %11 = shl nuw nsw i64 %10, 1
  %12 = xor i64 %11, 126
  tail call void @_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPSt4pairIPKciESt6vectorIS5_SaIS5_EEEElNS0_5__ops15_Iter_less_iterEEvT_SD_T0_T1_(ptr %0, ptr %1, i64 noundef %12, i8 undef)
  %13 = icmp sgt i64 %8, 256
  br i1 %13, label %14, label %102

14:                                               ; preds = %5
  %15 = getelementptr inbounds nuw i8, ptr %0, i64 8
  br label %16

16:                                               ; preds = %70, %14
  %17 = phi i64 [ 16, %14 ], [ %71, %70 ]
  %18 = phi ptr [ %0, %14 ], [ %19, %70 ]
  %19 = getelementptr inbounds nuw i8, ptr %0, i64 %17
  %20 = getelementptr inbounds nuw i8, ptr %18, i64 24
  %21 = load i32, ptr %20, align 8, !tbaa !42
  %22 = load i32, ptr %15, align 8, !tbaa !42
  %23 = icmp eq i32 %21, %22
  br i1 %23, label %27, label %24

24:                                               ; preds = %16
  %25 = icmp sgt i32 %21, %22
  %26 = load ptr, ptr %19, align 8
  br i1 %25, label %32, label %49

27:                                               ; preds = %16
  %28 = load ptr, ptr %19, align 8, !tbaa !40
  %29 = load ptr, ptr %0, align 8, !tbaa !40
  %30 = tail call i32 @strcmp(ptr noundef nonnull dereferenceable(1) %28, ptr noundef nonnull dereferenceable(1) %29) #17
  %31 = icmp sgt i32 %30, 0
  br i1 %31, label %32, label %49

32:                                               ; preds = %24, %27
  %33 = phi ptr [ %26, %24 ], [ %28, %27 ]
  %34 = lshr exact i64 %17, 4
  %35 = getelementptr inbounds nuw i8, ptr %18, i64 32
  br label %36

36:                                               ; preds = %36, %32
  %37 = phi i64 [ %46, %36 ], [ %34, %32 ]
  %38 = phi ptr [ %41, %36 ], [ %35, %32 ]
  %39 = phi ptr [ %40, %36 ], [ %19, %32 ]
  %40 = getelementptr inbounds i8, ptr %39, i64 -16
  %41 = getelementptr inbounds i8, ptr %38, i64 -16
  %42 = load ptr, ptr %40, align 8, !tbaa !20
  store ptr %42, ptr %41, align 8, !tbaa !40
  %43 = getelementptr inbounds i8, ptr %39, i64 -8
  %44 = load i32, ptr %43, align 8, !tbaa !34
  %45 = getelementptr inbounds i8, ptr %38, i64 -8
  store i32 %44, ptr %45, align 8, !tbaa !42
  %46 = add nsw i64 %37, -1
  %47 = icmp samesign ugt i64 %37, 1
  br i1 %47, label %36, label %48, !llvm.loop !49

48:                                               ; preds = %36
  store ptr %33, ptr %0, align 8, !tbaa !40
  store i32 %21, ptr %15, align 8, !tbaa !42
  br label %70

49:                                               ; preds = %27, %24
  %50 = phi ptr [ %28, %27 ], [ %26, %24 ]
  br label %51

51:                                               ; preds = %65, %49
  %52 = phi ptr [ %19, %49 ], [ %53, %65 ]
  %53 = getelementptr inbounds i8, ptr %52, i64 -16
  %54 = getelementptr inbounds i8, ptr %52, i64 -8
  %55 = load i32, ptr %54, align 8, !tbaa !42
  %56 = icmp eq i32 %21, %55
  br i1 %56, label %61, label %57

57:                                               ; preds = %51
  %58 = icmp sgt i32 %21, %55
  br i1 %58, label %59, label %68

59:                                               ; preds = %57
  %60 = load ptr, ptr %53, align 8, !tbaa !20
  br label %65

61:                                               ; preds = %51
  %62 = load ptr, ptr %53, align 8, !tbaa !40
  %63 = tail call i32 @strcmp(ptr noundef nonnull dereferenceable(1) %50, ptr noundef nonnull dereferenceable(1) %62) #17
  %64 = icmp sgt i32 %63, 0
  br i1 %64, label %65, label %68

65:                                               ; preds = %61, %59
  %66 = phi ptr [ %60, %59 ], [ %62, %61 ]
  store ptr %66, ptr %52, align 8, !tbaa !40
  %67 = getelementptr inbounds nuw i8, ptr %52, i64 8
  store i32 %55, ptr %67, align 8, !tbaa !42
  br label %51, !llvm.loop !50

68:                                               ; preds = %61, %57
  store ptr %50, ptr %52, align 8, !tbaa !40
  %69 = getelementptr inbounds nuw i8, ptr %52, i64 8
  store i32 %21, ptr %69, align 8, !tbaa !42
  br label %70

70:                                               ; preds = %68, %48
  %71 = add nuw nsw i64 %17, 16
  %72 = icmp eq i64 %71, 256
  br i1 %72, label %73, label %16, !llvm.loop !51

73:                                               ; preds = %70
  %74 = getelementptr inbounds nuw i8, ptr %0, i64 256
  %75 = icmp eq ptr %74, %1
  br i1 %75, label %103, label %76

76:                                               ; preds = %73, %98
  %77 = phi ptr [ %100, %98 ], [ %74, %73 ]
  %78 = load ptr, ptr %77, align 8
  %79 = getelementptr inbounds nuw i8, ptr %77, i64 8
  %80 = load i32, ptr %79, align 8
  br label %81

81:                                               ; preds = %95, %76
  %82 = phi ptr [ %77, %76 ], [ %83, %95 ]
  %83 = getelementptr inbounds i8, ptr %82, i64 -16
  %84 = getelementptr inbounds i8, ptr %82, i64 -8
  %85 = load i32, ptr %84, align 8, !tbaa !42
  %86 = icmp eq i32 %80, %85
  br i1 %86, label %91, label %87

87:                                               ; preds = %81
  %88 = icmp sgt i32 %80, %85
  br i1 %88, label %89, label %98

89:                                               ; preds = %87
  %90 = load ptr, ptr %83, align 8, !tbaa !20
  br label %95

91:                                               ; preds = %81
  %92 = load ptr, ptr %83, align 8, !tbaa !40
  %93 = tail call i32 @strcmp(ptr noundef nonnull dereferenceable(1) %78, ptr noundef nonnull dereferenceable(1) %92) #17
  %94 = icmp sgt i32 %93, 0
  br i1 %94, label %95, label %98

95:                                               ; preds = %91, %89
  %96 = phi ptr [ %90, %89 ], [ %92, %91 ]
  store ptr %96, ptr %82, align 8, !tbaa !40
  %97 = getelementptr inbounds nuw i8, ptr %82, i64 8
  store i32 %85, ptr %97, align 8, !tbaa !42
  br label %81, !llvm.loop !50

98:                                               ; preds = %91, %87
  store ptr %78, ptr %82, align 8, !tbaa !40
  %99 = getelementptr inbounds nuw i8, ptr %82, i64 8
  store i32 %80, ptr %99, align 8, !tbaa !42
  %100 = getelementptr inbounds nuw i8, ptr %77, i64 16
  %101 = icmp eq ptr %100, %1
  br i1 %101, label %103, label %76, !llvm.loop !52

102:                                              ; preds = %5
  tail call void @_ZSt16__insertion_sortIN9__gnu_cxx17__normal_iteratorIPSt4pairIPKciESt6vectorIS5_SaIS5_EEEENS0_5__ops15_Iter_less_iterEEvT_SD_T0_(ptr %0, ptr %1, i8 undef)
  br label %103

103:                                              ; preds = %98, %102, %73, %3
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPSt4pairIPKciESt6vectorIS5_SaIS5_EEEElNS0_5__ops15_Iter_less_iterEEvT_SD_T0_T1_(ptr %0, ptr %1, i64 noundef %2, i8 %3) local_unnamed_addr #0 comdat {
  %5 = ptrtoint ptr %0 to i64
  %6 = ptrtoint ptr %1 to i64
  %7 = sub i64 %6, %5
  %8 = ashr exact i64 %7, 4
  %9 = icmp sgt i64 %8, 16
  br i1 %9, label %10, label %98

10:                                               ; preds = %4
  %11 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %12 = getelementptr inbounds nuw i8, ptr %0, i64 8
  br label %13

13:                                               ; preds = %10, %92
  %14 = phi i64 [ %8, %10 ], [ %96, %92 ]
  %15 = phi i64 [ %2, %10 ], [ %93, %92 ]
  %16 = phi ptr [ %1, %10 ], [ %54, %92 ]
  %17 = icmp eq i64 %15, 0
  br i1 %17, label %18, label %45

18:                                               ; preds = %13
  %19 = add nsw i64 %14, -2
  %20 = lshr i64 %19, 1
  br label %21

21:                                               ; preds = %21, %18
  %22 = phi i64 [ %20, %18 ], [ %30, %21 ]
  %23 = getelementptr inbounds %"struct.std::pair.3", ptr %0, i64 %22
  %24 = load i64, ptr %23, align 8
  %25 = getelementptr inbounds nuw i8, ptr %23, i64 8
  %26 = load i64, ptr %25, align 8
  %27 = insertvalue [2 x i64] poison, i64 %24, 0
  %28 = insertvalue [2 x i64] %27, i64 %26, 1
  tail call void @_ZSt13__adjust_heapIN9__gnu_cxx17__normal_iteratorIPSt4pairIPKciESt6vectorIS5_SaIS5_EEEElS5_NS0_5__ops15_Iter_less_iterEEvT_T0_SE_T1_T2_(ptr %0, i64 noundef %22, i64 noundef %14, [2 x i64] %28, i8 undef)
  %29 = icmp eq i64 %22, 0
  %30 = add nsw i64 %22, -1
  br i1 %29, label %31, label %21, !llvm.loop !53

31:                                               ; preds = %21, %31
  %32 = phi ptr [ %33, %31 ], [ %16, %21 ]
  %33 = getelementptr inbounds i8, ptr %32, i64 -16
  %34 = load i64, ptr %33, align 8
  %35 = getelementptr inbounds i8, ptr %32, i64 -8
  %36 = load i64, ptr %35, align 8
  %37 = load ptr, ptr %0, align 8, !tbaa !20
  store ptr %37, ptr %33, align 8, !tbaa !40
  %38 = load i32, ptr %12, align 8, !tbaa !34
  store i32 %38, ptr %35, align 8, !tbaa !42
  %39 = ptrtoint ptr %33 to i64
  %40 = sub i64 %39, %5
  %41 = ashr exact i64 %40, 4
  %42 = insertvalue [2 x i64] poison, i64 %34, 0
  %43 = insertvalue [2 x i64] %42, i64 %36, 1
  tail call void @_ZSt13__adjust_heapIN9__gnu_cxx17__normal_iteratorIPSt4pairIPKciESt6vectorIS5_SaIS5_EEEElS5_NS0_5__ops15_Iter_less_iterEEvT_T0_SE_T1_T2_(ptr nonnull %0, i64 noundef 0, i64 noundef %41, [2 x i64] %43, i8 undef)
  %44 = icmp sgt i64 %40, 16
  br i1 %44, label %31, label %98, !llvm.loop !54

45:                                               ; preds = %13
  %46 = lshr i64 %14, 1
  %47 = getelementptr inbounds nuw %"struct.std::pair.3", ptr %0, i64 %46
  %48 = getelementptr inbounds i8, ptr %16, i64 -16
  tail call void @_ZSt22__move_median_to_firstIN9__gnu_cxx17__normal_iteratorIPSt4pairIPKciESt6vectorIS5_SaIS5_EEEENS0_5__ops15_Iter_less_iterEEvT_SD_SD_SD_T0_(ptr %0, ptr nonnull %11, ptr %47, ptr nonnull %48, i8 undef)
  br label %49

49:                                               ; preds = %87, %45
  %50 = phi ptr [ %16, %45 ], [ %72, %87 ]
  %51 = phi ptr [ %11, %45 ], [ %91, %87 ]
  %52 = load i32, ptr %12, align 8, !tbaa !42
  br label %53

53:                                               ; preds = %65, %49
  %54 = phi ptr [ %51, %49 ], [ %66, %65 ]
  %55 = getelementptr inbounds nuw i8, ptr %54, i64 8
  %56 = load i32, ptr %55, align 8, !tbaa !42
  %57 = icmp eq i32 %56, %52
  br i1 %57, label %60, label %58

58:                                               ; preds = %53
  %59 = icmp sgt i32 %56, %52
  br i1 %59, label %65, label %67

60:                                               ; preds = %53
  %61 = load ptr, ptr %54, align 8, !tbaa !40
  %62 = load ptr, ptr %0, align 8, !tbaa !40
  %63 = tail call i32 @strcmp(ptr noundef nonnull dereferenceable(1) %61, ptr noundef nonnull dereferenceable(1) %62) #17
  %64 = icmp sgt i32 %63, 0
  br i1 %64, label %65, label %67

65:                                               ; preds = %60, %58
  %66 = getelementptr inbounds nuw i8, ptr %54, i64 16
  br label %53, !llvm.loop !55

67:                                               ; preds = %60, %58
  %68 = phi i32 [ %56, %58 ], [ %52, %60 ]
  %69 = getelementptr inbounds nuw i8, ptr %54, i64 8
  br label %70

70:                                               ; preds = %83, %67
  %71 = phi ptr [ %50, %67 ], [ %72, %83 ]
  %72 = getelementptr inbounds i8, ptr %71, i64 -16
  %73 = getelementptr inbounds i8, ptr %71, i64 -8
  %74 = load i32, ptr %73, align 8, !tbaa !42
  %75 = icmp eq i32 %52, %74
  br i1 %75, label %78, label %76

76:                                               ; preds = %70
  %77 = icmp sgt i32 %52, %74
  br i1 %77, label %83, label %84

78:                                               ; preds = %70
  %79 = load ptr, ptr %0, align 8, !tbaa !40
  %80 = load ptr, ptr %72, align 8, !tbaa !40
  %81 = tail call i32 @strcmp(ptr noundef nonnull dereferenceable(1) %79, ptr noundef nonnull dereferenceable(1) %80) #17
  %82 = icmp sgt i32 %81, 0
  br i1 %82, label %83, label %84

83:                                               ; preds = %78, %76
  br label %70, !llvm.loop !56

84:                                               ; preds = %78, %76
  %85 = phi i32 [ %74, %76 ], [ %52, %78 ]
  %86 = icmp ult ptr %54, %72
  br i1 %86, label %87, label %92

87:                                               ; preds = %84
  %88 = getelementptr inbounds i8, ptr %71, i64 -8
  %89 = load ptr, ptr %54, align 8, !tbaa !20
  %90 = load ptr, ptr %72, align 8, !tbaa !20
  store ptr %90, ptr %54, align 8, !tbaa !20
  store ptr %89, ptr %72, align 8, !tbaa !20
  store i32 %85, ptr %69, align 8, !tbaa !34
  store i32 %68, ptr %88, align 4, !tbaa !34
  %91 = getelementptr inbounds nuw i8, ptr %54, i64 16
  br label %49, !llvm.loop !57

92:                                               ; preds = %84
  %93 = add nsw i64 %15, -1
  tail call void @_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPSt4pairIPKciESt6vectorIS5_SaIS5_EEEElNS0_5__ops15_Iter_less_iterEEvT_SD_T0_T1_(ptr %54, ptr %16, i64 noundef %93, i8 undef)
  %94 = ptrtoint ptr %54 to i64
  %95 = sub i64 %94, %5
  %96 = ashr exact i64 %95, 4
  %97 = icmp sgt i64 %96, 16
  br i1 %97, label %13, label %98, !llvm.loop !58

98:                                               ; preds = %92, %31, %4
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZSt13__adjust_heapIN9__gnu_cxx17__normal_iteratorIPSt4pairIPKciESt6vectorIS5_SaIS5_EEEElS5_NS0_5__ops15_Iter_less_iterEEvT_T0_SE_T1_T2_(ptr %0, i64 noundef %1, i64 noundef %2, [2 x i64] %3, i8 %4) local_unnamed_addr #0 comdat {
  %6 = add nsw i64 %2, -1
  %7 = sdiv i64 %6, 2
  %8 = icmp slt i64 %1, %7
  br i1 %8, label %9, label %38

9:                                                ; preds = %5, %28
  %10 = phi i64 [ %30, %28 ], [ %1, %5 ]
  %11 = shl i64 %10, 1
  %12 = add i64 %11, 2
  %13 = getelementptr inbounds %"struct.std::pair.3", ptr %0, i64 %12
  %14 = or disjoint i64 %11, 1
  %15 = getelementptr inbounds %"struct.std::pair.3", ptr %0, i64 %14
  %16 = getelementptr inbounds nuw i8, ptr %13, i64 8
  %17 = load i32, ptr %16, align 8, !tbaa !42
  %18 = getelementptr inbounds nuw i8, ptr %15, i64 8
  %19 = load i32, ptr %18, align 8, !tbaa !42
  %20 = icmp eq i32 %17, %19
  br i1 %20, label %23, label %21

21:                                               ; preds = %9
  %22 = icmp sgt i32 %17, %19
  br label %28

23:                                               ; preds = %9
  %24 = load ptr, ptr %13, align 8, !tbaa !40
  %25 = load ptr, ptr %15, align 8, !tbaa !40
  %26 = tail call i32 @strcmp(ptr noundef nonnull dereferenceable(1) %24, ptr noundef nonnull dereferenceable(1) %25) #17
  %27 = icmp sgt i32 %26, 0
  br label %28

28:                                               ; preds = %21, %23
  %29 = phi i1 [ %22, %21 ], [ %27, %23 ]
  %30 = select i1 %29, i64 %14, i64 %12
  %31 = getelementptr inbounds %"struct.std::pair.3", ptr %0, i64 %30
  %32 = getelementptr inbounds %"struct.std::pair.3", ptr %0, i64 %10
  %33 = load ptr, ptr %31, align 8, !tbaa !20
  store ptr %33, ptr %32, align 8, !tbaa !40
  %34 = getelementptr inbounds nuw i8, ptr %31, i64 8
  %35 = load i32, ptr %34, align 8, !tbaa !34
  %36 = getelementptr inbounds nuw i8, ptr %32, i64 8
  store i32 %35, ptr %36, align 8, !tbaa !42
  %37 = icmp slt i64 %30, %7
  br i1 %37, label %9, label %38, !llvm.loop !59

38:                                               ; preds = %28, %5
  %39 = phi i64 [ %1, %5 ], [ %30, %28 ]
  %40 = and i64 %2, 1
  %41 = icmp eq i64 %40, 0
  br i1 %41, label %42, label %55

42:                                               ; preds = %38
  %43 = add nsw i64 %2, -2
  %44 = ashr exact i64 %43, 1
  %45 = icmp eq i64 %39, %44
  br i1 %45, label %46, label %55

46:                                               ; preds = %42
  %47 = shl nsw i64 %39, 1
  %48 = or disjoint i64 %47, 1
  %49 = getelementptr inbounds %"struct.std::pair.3", ptr %0, i64 %48
  %50 = getelementptr inbounds %"struct.std::pair.3", ptr %0, i64 %39
  %51 = load ptr, ptr %49, align 8, !tbaa !20
  store ptr %51, ptr %50, align 8, !tbaa !40
  %52 = getelementptr inbounds nuw i8, ptr %49, i64 8
  %53 = load i32, ptr %52, align 8, !tbaa !34
  %54 = getelementptr inbounds nuw i8, ptr %50, i64 8
  store i32 %53, ptr %54, align 8, !tbaa !42
  br label %55

55:                                               ; preds = %46, %42, %38
  %56 = phi i64 [ %48, %46 ], [ %39, %42 ], [ %39, %38 ]
  %57 = extractvalue [2 x i64] %3, 0
  %58 = inttoptr i64 %57 to ptr
  %59 = extractvalue [2 x i64] %3, 1
  %60 = trunc i64 %59 to i32
  %61 = icmp sgt i64 %56, %1
  br i1 %61, label %62, label %83

62:                                               ; preds = %55, %78
  %63 = phi i64 [ %65, %78 ], [ %56, %55 ]
  %64 = add nsw i64 %63, -1
  %65 = sdiv i64 %64, 2
  %66 = getelementptr inbounds %"struct.std::pair.3", ptr %0, i64 %65
  %67 = getelementptr inbounds nuw i8, ptr %66, i64 8
  %68 = load i32, ptr %67, align 8, !tbaa !42
  %69 = icmp eq i32 %68, %60
  br i1 %69, label %74, label %70

70:                                               ; preds = %62
  %71 = icmp sgt i32 %68, %60
  br i1 %71, label %72, label %83

72:                                               ; preds = %70
  %73 = load ptr, ptr %66, align 8, !tbaa !20
  br label %78

74:                                               ; preds = %62
  %75 = load ptr, ptr %66, align 8, !tbaa !40
  %76 = tail call i32 @strcmp(ptr noundef nonnull dereferenceable(1) %75, ptr noundef nonnull dereferenceable(1) %58) #17
  %77 = icmp sgt i32 %76, 0
  br i1 %77, label %78, label %83

78:                                               ; preds = %74, %72
  %79 = phi ptr [ %73, %72 ], [ %75, %74 ]
  %80 = getelementptr inbounds %"struct.std::pair.3", ptr %0, i64 %63
  store ptr %79, ptr %80, align 8, !tbaa !40
  %81 = getelementptr inbounds nuw i8, ptr %80, i64 8
  store i32 %68, ptr %81, align 8, !tbaa !42
  %82 = icmp sgt i64 %65, %1
  br i1 %82, label %62, label %83, !llvm.loop !60

83:                                               ; preds = %70, %74, %78, %55
  %84 = phi i64 [ %56, %55 ], [ %63, %74 ], [ %65, %78 ], [ %63, %70 ]
  %85 = getelementptr inbounds %"struct.std::pair.3", ptr %0, i64 %84
  store ptr %58, ptr %85, align 8, !tbaa !40
  %86 = getelementptr inbounds nuw i8, ptr %85, i64 8
  store i32 %60, ptr %86, align 8, !tbaa !42
  ret void
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: read)
declare i32 @strcmp(ptr noundef captures(none), ptr noundef captures(none)) local_unnamed_addr #15

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZSt22__move_median_to_firstIN9__gnu_cxx17__normal_iteratorIPSt4pairIPKciESt6vectorIS5_SaIS5_EEEENS0_5__ops15_Iter_less_iterEEvT_SD_SD_SD_T0_(ptr %0, ptr %1, ptr %2, ptr %3, i8 %4) local_unnamed_addr #0 comdat {
  %6 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %7 = load i32, ptr %6, align 8, !tbaa !42
  %8 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %9 = load i32, ptr %8, align 8, !tbaa !42
  %10 = icmp eq i32 %7, %9
  br i1 %10, label %13, label %11

11:                                               ; preds = %5
  %12 = icmp sgt i32 %7, %9
  br i1 %12, label %18, label %59

13:                                               ; preds = %5
  %14 = load ptr, ptr %1, align 8, !tbaa !40
  %15 = load ptr, ptr %2, align 8, !tbaa !40
  %16 = tail call i32 @strcmp(ptr noundef nonnull dereferenceable(1) %14, ptr noundef nonnull dereferenceable(1) %15) #17
  %17 = icmp sgt i32 %16, 0
  br i1 %17, label %18, label %59

18:                                               ; preds = %11, %13
  %19 = getelementptr inbounds nuw i8, ptr %3, i64 8
  %20 = load i32, ptr %19, align 8, !tbaa !42
  %21 = icmp eq i32 %9, %20
  br i1 %21, label %26, label %22

22:                                               ; preds = %18
  %23 = icmp sgt i32 %9, %20
  br i1 %23, label %24, label %36

24:                                               ; preds = %22
  %25 = load ptr, ptr %2, align 8, !tbaa !20
  br label %31

26:                                               ; preds = %18
  %27 = load ptr, ptr %2, align 8, !tbaa !40
  %28 = load ptr, ptr %3, align 8, !tbaa !40
  %29 = tail call i32 @strcmp(ptr noundef nonnull dereferenceable(1) %27, ptr noundef nonnull dereferenceable(1) %28) #17
  %30 = icmp sgt i32 %29, 0
  br i1 %30, label %31, label %36

31:                                               ; preds = %24, %26
  %32 = phi ptr [ %25, %24 ], [ %27, %26 ]
  %33 = load ptr, ptr %0, align 8, !tbaa !20
  store ptr %32, ptr %0, align 8, !tbaa !20
  store ptr %33, ptr %2, align 8, !tbaa !20
  %34 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %35 = load i32, ptr %34, align 8, !tbaa !34
  store i32 %9, ptr %34, align 8, !tbaa !34
  store i32 %35, ptr %8, align 8, !tbaa !34
  br label %100

36:                                               ; preds = %22, %26
  %37 = icmp eq i32 %7, %20
  br i1 %37, label %44, label %38

38:                                               ; preds = %36
  %39 = icmp sgt i32 %7, %20
  br i1 %39, label %40, label %42

40:                                               ; preds = %38
  %41 = load ptr, ptr %3, align 8, !tbaa !20
  br label %49

42:                                               ; preds = %38
  %43 = load ptr, ptr %1, align 8, !tbaa !20
  br label %54

44:                                               ; preds = %36
  %45 = load ptr, ptr %1, align 8, !tbaa !40
  %46 = load ptr, ptr %3, align 8, !tbaa !40
  %47 = tail call i32 @strcmp(ptr noundef nonnull dereferenceable(1) %45, ptr noundef nonnull dereferenceable(1) %46) #17
  %48 = icmp sgt i32 %47, 0
  br i1 %48, label %49, label %54

49:                                               ; preds = %40, %44
  %50 = phi ptr [ %41, %40 ], [ %46, %44 ]
  %51 = load ptr, ptr %0, align 8, !tbaa !20
  store ptr %50, ptr %0, align 8, !tbaa !20
  store ptr %51, ptr %3, align 8, !tbaa !20
  %52 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %53 = load i32, ptr %52, align 8, !tbaa !34
  store i32 %20, ptr %52, align 8, !tbaa !34
  store i32 %53, ptr %19, align 8, !tbaa !34
  br label %100

54:                                               ; preds = %42, %44
  %55 = phi ptr [ %43, %42 ], [ %45, %44 ]
  %56 = load ptr, ptr %0, align 8, !tbaa !20
  store ptr %55, ptr %0, align 8, !tbaa !20
  store ptr %56, ptr %1, align 8, !tbaa !20
  %57 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %58 = load i32, ptr %57, align 8, !tbaa !34
  store i32 %7, ptr %57, align 8, !tbaa !34
  store i32 %58, ptr %6, align 8, !tbaa !34
  br label %100

59:                                               ; preds = %11, %13
  %60 = getelementptr inbounds nuw i8, ptr %3, i64 8
  %61 = load i32, ptr %60, align 8, !tbaa !42
  %62 = icmp eq i32 %7, %61
  br i1 %62, label %67, label %63

63:                                               ; preds = %59
  %64 = icmp sgt i32 %7, %61
  br i1 %64, label %65, label %77

65:                                               ; preds = %63
  %66 = load ptr, ptr %1, align 8, !tbaa !20
  br label %72

67:                                               ; preds = %59
  %68 = load ptr, ptr %1, align 8, !tbaa !40
  %69 = load ptr, ptr %3, align 8, !tbaa !40
  %70 = tail call i32 @strcmp(ptr noundef nonnull dereferenceable(1) %68, ptr noundef nonnull dereferenceable(1) %69) #17
  %71 = icmp sgt i32 %70, 0
  br i1 %71, label %72, label %77

72:                                               ; preds = %65, %67
  %73 = phi ptr [ %66, %65 ], [ %68, %67 ]
  %74 = load ptr, ptr %0, align 8, !tbaa !20
  store ptr %73, ptr %0, align 8, !tbaa !20
  store ptr %74, ptr %1, align 8, !tbaa !20
  %75 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %76 = load i32, ptr %75, align 8, !tbaa !34
  store i32 %7, ptr %75, align 8, !tbaa !34
  store i32 %76, ptr %6, align 8, !tbaa !34
  br label %100

77:                                               ; preds = %63, %67
  %78 = icmp eq i32 %9, %61
  br i1 %78, label %85, label %79

79:                                               ; preds = %77
  %80 = icmp sgt i32 %9, %61
  br i1 %80, label %81, label %83

81:                                               ; preds = %79
  %82 = load ptr, ptr %3, align 8, !tbaa !20
  br label %90

83:                                               ; preds = %79
  %84 = load ptr, ptr %2, align 8, !tbaa !20
  br label %95

85:                                               ; preds = %77
  %86 = load ptr, ptr %2, align 8, !tbaa !40
  %87 = load ptr, ptr %3, align 8, !tbaa !40
  %88 = tail call i32 @strcmp(ptr noundef nonnull dereferenceable(1) %86, ptr noundef nonnull dereferenceable(1) %87) #17
  %89 = icmp sgt i32 %88, 0
  br i1 %89, label %90, label %95

90:                                               ; preds = %81, %85
  %91 = phi ptr [ %82, %81 ], [ %87, %85 ]
  %92 = load ptr, ptr %0, align 8, !tbaa !20
  store ptr %91, ptr %0, align 8, !tbaa !20
  store ptr %92, ptr %3, align 8, !tbaa !20
  %93 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %94 = load i32, ptr %93, align 8, !tbaa !34
  store i32 %61, ptr %93, align 8, !tbaa !34
  store i32 %94, ptr %60, align 8, !tbaa !34
  br label %100

95:                                               ; preds = %83, %85
  %96 = phi ptr [ %84, %83 ], [ %86, %85 ]
  %97 = load ptr, ptr %0, align 8, !tbaa !20
  store ptr %96, ptr %0, align 8, !tbaa !20
  store ptr %97, ptr %2, align 8, !tbaa !20
  %98 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %99 = load i32, ptr %98, align 8, !tbaa !34
  store i32 %9, ptr %98, align 8, !tbaa !34
  store i32 %99, ptr %8, align 8, !tbaa !34
  br label %100

100:                                              ; preds = %72, %95, %90, %31, %54, %49
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.ctlz.i64(i64, i1 immarg) #16

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZSt16__insertion_sortIN9__gnu_cxx17__normal_iteratorIPSt4pairIPKciESt6vectorIS5_SaIS5_EEEENS0_5__ops15_Iter_less_iterEEvT_SD_T0_(ptr %0, ptr %1, i8 %2) local_unnamed_addr #0 comdat {
  %4 = icmp eq ptr %0, %1
  br i1 %4, label %107, label %5

5:                                                ; preds = %3
  %6 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %7 = icmp eq ptr %6, %1
  br i1 %7, label %107, label %8

8:                                                ; preds = %5
  %9 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %10 = ptrtoint ptr %0 to i64
  br label %11

11:                                               ; preds = %8, %104
  %12 = phi ptr [ %6, %8 ], [ %105, %104 ]
  %13 = phi ptr [ %0, %8 ], [ %12, %104 ]
  %14 = getelementptr inbounds nuw i8, ptr %13, i64 24
  %15 = load i32, ptr %14, align 8, !tbaa !42
  %16 = load i32, ptr %9, align 8, !tbaa !42
  %17 = icmp eq i32 %15, %16
  br i1 %17, label %21, label %18

18:                                               ; preds = %11
  %19 = icmp sgt i32 %15, %16
  %20 = load ptr, ptr %12, align 8
  br i1 %19, label %26, label %83

21:                                               ; preds = %11
  %22 = load ptr, ptr %12, align 8, !tbaa !40
  %23 = load ptr, ptr %0, align 8, !tbaa !40
  %24 = tail call i32 @strcmp(ptr noundef nonnull dereferenceable(1) %22, ptr noundef nonnull dereferenceable(1) %23) #17
  %25 = icmp sgt i32 %24, 0
  br i1 %25, label %26, label %83

26:                                               ; preds = %18, %21
  %27 = phi ptr [ %20, %18 ], [ %22, %21 ]
  %28 = ptrtoint ptr %12 to i64
  %29 = sub i64 %28, %10
  %30 = ashr exact i64 %29, 4
  %31 = icmp sgt i64 %30, 0
  br i1 %31, label %32, label %82

32:                                               ; preds = %26
  %33 = getelementptr inbounds nuw i8, ptr %13, i64 32
  %34 = icmp eq i64 %29, 16
  br i1 %34, label %66, label %35

35:                                               ; preds = %32
  %36 = and i64 %30, 9223372036854775806
  %37 = and i64 %30, 1
  %38 = mul i64 %36, -16
  %39 = getelementptr i8, ptr %33, i64 %38
  %40 = mul i64 %36, -16
  %41 = getelementptr i8, ptr %12, i64 %40
  br label %42

42:                                               ; preds = %42, %35
  %43 = phi i64 [ 0, %35 ], [ %62, %42 ]
  %44 = mul i64 %43, -16
  %45 = getelementptr i8, ptr %33, i64 %44
  %46 = getelementptr i8, ptr %33, i64 %44
  %47 = mul i64 %43, -16
  %48 = getelementptr i8, ptr %12, i64 %47
  %49 = getelementptr i8, ptr %12, i64 %47
  %50 = getelementptr inbounds i8, ptr %48, i64 -16
  %51 = getelementptr i8, ptr %49, i64 -32
  %52 = getelementptr inbounds i8, ptr %45, i64 -16
  %53 = getelementptr i8, ptr %46, i64 -32
  %54 = load ptr, ptr %50, align 8, !tbaa !20
  %55 = load ptr, ptr %51, align 8, !tbaa !20
  store ptr %54, ptr %52, align 8, !tbaa !40
  store ptr %55, ptr %53, align 8, !tbaa !40
  %56 = getelementptr inbounds i8, ptr %48, i64 -8
  %57 = getelementptr i8, ptr %49, i64 -24
  %58 = load i32, ptr %56, align 8, !tbaa !34
  %59 = load i32, ptr %57, align 8, !tbaa !34
  %60 = getelementptr inbounds i8, ptr %45, i64 -8
  %61 = getelementptr i8, ptr %46, i64 -24
  store i32 %58, ptr %60, align 8, !tbaa !42
  store i32 %59, ptr %61, align 8, !tbaa !42
  %62 = add nuw i64 %43, 2
  %63 = icmp eq i64 %62, %36
  br i1 %63, label %64, label %42, !llvm.loop !61

64:                                               ; preds = %42
  %65 = icmp eq i64 %30, %36
  br i1 %65, label %82, label %66

66:                                               ; preds = %32, %64
  %67 = phi i64 [ %30, %32 ], [ %37, %64 ]
  %68 = phi ptr [ %33, %32 ], [ %39, %64 ]
  %69 = phi ptr [ %12, %32 ], [ %41, %64 ]
  br label %70

70:                                               ; preds = %66, %70
  %71 = phi i64 [ %80, %70 ], [ %67, %66 ]
  %72 = phi ptr [ %75, %70 ], [ %68, %66 ]
  %73 = phi ptr [ %74, %70 ], [ %69, %66 ]
  %74 = getelementptr inbounds i8, ptr %73, i64 -16
  %75 = getelementptr inbounds i8, ptr %72, i64 -16
  %76 = load ptr, ptr %74, align 8, !tbaa !20
  store ptr %76, ptr %75, align 8, !tbaa !40
  %77 = getelementptr inbounds i8, ptr %73, i64 -8
  %78 = load i32, ptr %77, align 8, !tbaa !34
  %79 = getelementptr inbounds i8, ptr %72, i64 -8
  store i32 %78, ptr %79, align 8, !tbaa !42
  %80 = add nsw i64 %71, -1
  %81 = icmp samesign ugt i64 %71, 1
  br i1 %81, label %70, label %82, !llvm.loop !64

82:                                               ; preds = %70, %64, %26
  store ptr %27, ptr %0, align 8, !tbaa !40
  store i32 %15, ptr %9, align 8, !tbaa !42
  br label %104

83:                                               ; preds = %18, %21
  %84 = phi ptr [ %22, %21 ], [ %20, %18 ]
  br label %85

85:                                               ; preds = %99, %83
  %86 = phi ptr [ %12, %83 ], [ %87, %99 ]
  %87 = getelementptr inbounds i8, ptr %86, i64 -16
  %88 = getelementptr inbounds i8, ptr %86, i64 -8
  %89 = load i32, ptr %88, align 8, !tbaa !42
  %90 = icmp eq i32 %15, %89
  br i1 %90, label %95, label %91

91:                                               ; preds = %85
  %92 = icmp sgt i32 %15, %89
  br i1 %92, label %93, label %102

93:                                               ; preds = %91
  %94 = load ptr, ptr %87, align 8, !tbaa !20
  br label %99

95:                                               ; preds = %85
  %96 = load ptr, ptr %87, align 8, !tbaa !40
  %97 = tail call i32 @strcmp(ptr noundef nonnull dereferenceable(1) %84, ptr noundef nonnull dereferenceable(1) %96) #17
  %98 = icmp sgt i32 %97, 0
  br i1 %98, label %99, label %102

99:                                               ; preds = %95, %93
  %100 = phi ptr [ %94, %93 ], [ %96, %95 ]
  store ptr %100, ptr %86, align 8, !tbaa !40
  %101 = getelementptr inbounds nuw i8, ptr %86, i64 8
  store i32 %89, ptr %101, align 8, !tbaa !42
  br label %85, !llvm.loop !50

102:                                              ; preds = %91, %95
  store ptr %84, ptr %86, align 8, !tbaa !40
  %103 = getelementptr inbounds nuw i8, ptr %86, i64 8
  store i32 %15, ptr %103, align 8, !tbaa !42
  br label %104

104:                                              ; preds = %82, %102
  %105 = getelementptr inbounds nuw i8, ptr %12, i64 16
  %106 = icmp eq ptr %105, %1
  br i1 %106, label %107, label %11, !llvm.loop !51

107:                                              ; preds = %104, %5, %3
  ret void
}

attributes #0 = { mustprogress uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress nofree nounwind willreturn memory(read) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nobuiltin allocsize(0) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #5 = { nobuiltin nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { mustprogress norecurse uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #8 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #9 = { mustprogress nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #10 = { noinline noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #11 = { cold nofree noreturn }
attributes #12 = { nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #13 = { cold noreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #14 = { inlinehint mustprogress uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #15 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: read) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #16 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #17 = { nounwind willreturn memory(read) }
attributes #18 = { builtin allocsize(0) }
attributes #19 = { builtin nounwind }
attributes #20 = { nounwind }
attributes #21 = { cold noreturn }
attributes #22 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !11, i64 4104}
!7 = !{!"_ZTS11word_reader", !8, i64 0, !9, i64 4, !11, i64 4104, !11, i64 4112, !13, i64 4120}
!8 = !{!"int", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C++ TBAA"}
!11 = !{!"p1 omnipotent char", !12, i64 0}
!12 = !{!"any pointer", !9, i64 0}
!13 = !{!"p1 _ZTS8_IO_FILE", !12, i64 0}
!14 = !{!9, !9, i64 0}
!15 = !{!7, !13, i64 4120}
!16 = !{!7, !11, i64 4112}
!17 = !{!7, !8, i64 0}
!18 = distinct !{!18, !19}
!19 = !{!"llvm.loop.mustprogress"}
!20 = !{!11, !11, i64 0}
!21 = !{!22, !24, i64 0}
!22 = !{!"_ZTSSt15_Rb_tree_header", !23, i64 0, !26, i64 32}
!23 = !{!"_ZTSSt18_Rb_tree_node_base", !24, i64 0, !25, i64 8, !25, i64 16, !25, i64 24}
!24 = !{!"_ZTSSt14_Rb_tree_color", !9, i64 0}
!25 = !{!"p1 _ZTSSt18_Rb_tree_node_base", !12, i64 0}
!26 = !{!"long", !9, i64 0}
!27 = !{!22, !25, i64 8}
!28 = !{!22, !25, i64 16}
!29 = !{!22, !25, i64 24}
!30 = !{!22, !26, i64 32}
!31 = !{!13, !13, i64 0}
!32 = !{!25, !25, i64 0}
!33 = distinct !{!33, !19}
!34 = !{!8, !8, i64 0}
!35 = !{!36, !8, i64 8}
!36 = !{!"_ZTSSt4pairIKPKciE", !11, i64 0, !8, i64 8}
!37 = distinct !{!37, !19}
!38 = distinct !{!38, !19}
!39 = !{!36, !11, i64 0}
!40 = !{!41, !11, i64 0}
!41 = !{!"_ZTSSt4pairIPKciE", !11, i64 0, !8, i64 8}
!42 = !{!41, !8, i64 8}
!43 = distinct !{!43, !19}
!44 = distinct !{!44, !19}
!45 = !{!23, !25, i64 24}
!46 = !{!23, !25, i64 16}
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
!61 = distinct !{!61, !19, !62, !63}
!62 = !{!"llvm.loop.isvectorized", i32 1}
!63 = !{!"llvm.loop.unroll.runtime.disable"}
!64 = distinct !{!64, !19, !62}
