; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Shootout-C++/spellcheck.cpp'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Shootout-C++/spellcheck.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

module asm ".globl _ZSt21ios_base_library_initv"

%"class.std::basic_istream" = type { ptr, i64, %"class.std::basic_ios" }
%"class.std::basic_ios" = type { %"class.std::ios_base", ptr, i8, i8, ptr, ptr, ptr, ptr }
%"class.std::ios_base" = type { ptr, i64, i64, i32, i32, i32, ptr, %"struct.std::ios_base::_Words", [8 x %"struct.std::ios_base::_Words"], i32, ptr, %"class.std::locale" }
%"struct.std::ios_base::_Words" = type { ptr, i64 }
%"class.std::locale" = type { ptr }
%"struct.std::piecewise_construct_t" = type { i8 }
%"class.std::basic_ostream" = type { ptr, %"class.std::basic_ios" }
%class.spell_checker = type { %"class.std::map" }
%"class.std::map" = type { %"class.std::_Rb_tree" }
%"class.std::_Rb_tree" = type { %"struct.std::_Rb_tree<std::pair<const char *, const char *>, std::pair<const std::pair<const char *, const char *>, int>, std::_Select1st<std::pair<const std::pair<const char *, const char *>, int>>, std::less<std::pair<const char *, const char *>>>::_Rb_tree_impl" }
%"struct.std::_Rb_tree<std::pair<const char *, const char *>, std::pair<const std::pair<const char *, const char *>, int>, std::_Select1st<std::pair<const std::pair<const char *, const char *>, int>>, std::less<std::pair<const char *, const char *>>>::_Rb_tree_impl" = type { [8 x i8], %"struct.std::_Rb_tree_header" }
%"struct.std::_Rb_tree_header" = type { %"struct.std::_Rb_tree_node_base", i64 }
%"struct.std::_Rb_tree_node_base" = type { i32, ptr, ptr, ptr }
%"class.std::tuple" = type { %"struct.std::_Tuple_impl" }
%"struct.std::_Tuple_impl" = type { %"struct.std::_Head_base" }
%"struct.std::_Head_base" = type { ptr }
%"class.std::tuple.2" = type { i8 }
%"class.std::basic_ifstream" = type { %"class.std::basic_istream.base", %"class.std::basic_filebuf", %"class.std::basic_ios" }
%"class.std::basic_istream.base" = type { ptr, i64 }
%"class.std::basic_filebuf" = type { %"class.std::basic_streambuf", %union.pthread_mutex_t, %"class.std::__basic_file", i32, %struct.__mbstate_t, %struct.__mbstate_t, %struct.__mbstate_t, ptr, i64, i8, i8, i8, i8, ptr, ptr, i8, ptr, ptr, i64, ptr, ptr }
%"class.std::basic_streambuf" = type { ptr, ptr, ptr, ptr, ptr, ptr, ptr, %"class.std::locale" }
%union.pthread_mutex_t = type { %struct.__pthread_mutex_s, [8 x i8] }
%struct.__pthread_mutex_s = type { i32, i32, i32, i32, i32, i32, %struct.__pthread_internal_list }
%struct.__pthread_internal_list = type { ptr, ptr }
%"class.std::__basic_file" = type <{ ptr, i8, [7 x i8] }>
%struct.__mbstate_t = type { i32, %union.anon }
%union.anon = type { i32 }
%"struct.std::pair" = type { ptr, ptr }

$_ZN13spell_checkerC2Ev = comdat any

$_ZN13spell_checker7processERSi = comdat any

$_ZN13spell_checkerD2Ev = comdat any

$_ZNSt3mapISt4pairIPKcS2_EiSt4lessIS3_ESaIS0_IKS3_iEEED2Ev = comdat any

$__clang_call_terminate = comdat any

$_ZNSt8_Rb_treeISt4pairIPKcS2_ES0_IKS3_iESt10_Select1stIS5_ESt4lessIS3_ESaIS5_EE22_M_emplace_hint_uniqueIJRKSt21piecewise_construct_tSt5tupleIJOS3_EESG_IJEEEEESt17_Rb_tree_iteratorIS5_ESt23_Rb_tree_const_iteratorIS5_EDpOT_ = comdat any

$_ZNSt8_Rb_treeISt4pairIPKcS2_ES0_IKS3_iESt10_Select1stIS5_ESt4lessIS3_ESaIS5_EE29_M_get_insert_hint_unique_posESt23_Rb_tree_const_iteratorIS5_ERS4_ = comdat any

$_ZNSt8_Rb_treeISt4pairIPKcS2_ES0_IKS3_iESt10_Select1stIS5_ESt4lessIS3_ESaIS5_EE8_M_eraseEPSt13_Rb_tree_nodeIS5_E = comdat any

$_ZSt19piecewise_construct = comdat any

@_ZSt3cin = external global %"class.std::basic_istream", align 8
@.str = private unnamed_addr constant [15 x i8] c"Usr.Dict.Words\00", align 1
@_ZSt19piecewise_construct = linkonce_odr dso_local constant %"struct.std::piecewise_construct_t" zeroinitializer, comdat, align 1
@_ZSt4cout = external global %"class.std::basic_ostream", align 8

; Function Attrs: mustprogress norecurse uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 personality ptr @__gxx_personality_v0 {
  %1 = alloca %class.spell_checker, align 8
  %2 = alloca [4096 x i8], align 1
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #15
  call void @_ZN13spell_checkerC2Ev(ptr noundef nonnull align 8 dereferenceable(48) %1)
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #15
  %3 = load ptr, ptr @_ZSt3cin, align 8, !tbaa !6
  %4 = getelementptr i8, ptr %3, i64 -24
  %5 = load i64, ptr %4, align 8
  %6 = getelementptr inbounds i8, ptr @_ZSt3cin, i64 %5
  %7 = getelementptr inbounds nuw i8, ptr %6, i64 232
  %8 = load ptr, ptr %7, align 8, !tbaa !9
  %9 = load ptr, ptr %8, align 8, !tbaa !6
  %10 = getelementptr inbounds nuw i8, ptr %9, i64 24
  %11 = load ptr, ptr %10, align 8
  %12 = invoke noundef ptr %11(ptr noundef nonnull align 8 dereferenceable(64) %8, ptr noundef nonnull %2, i64 noundef 4096)
          to label %13 unwind label %21

13:                                               ; preds = %0
  invoke void @_ZN13spell_checker7processERSi(ptr noundef nonnull align 8 dereferenceable(48) %1, ptr noundef nonnull align 8 dereferenceable(16) @_ZSt3cin)
          to label %14 unwind label %21

14:                                               ; preds = %13
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #15
  %15 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %16 = load ptr, ptr %15, align 8, !tbaa !29
  invoke void @_ZNSt8_Rb_treeISt4pairIPKcS2_ES0_IKS3_iESt10_Select1stIS5_ESt4lessIS3_ESaIS5_EE8_M_eraseEPSt13_Rb_tree_nodeIS5_E(ptr noundef nonnull align 8 dereferenceable(48) %1, ptr noundef %16)
          to label %20 unwind label %17

17:                                               ; preds = %14
  %18 = landingpad { ptr, i32 }
          catch ptr null
  %19 = extractvalue { ptr, i32 } %18, 0
  call void @__clang_call_terminate(ptr %19) #16
  unreachable

20:                                               ; preds = %14
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #15
  ret i32 0

21:                                               ; preds = %0, %13
  %22 = landingpad { ptr, i32 }
          cleanup
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #15
  call void @_ZN13spell_checkerD2Ev(ptr noundef nonnull align 8 dereferenceable(48) %1) #15
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #15
  resume { ptr, i32 } %22
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZN13spell_checkerC2Ev(ptr noundef nonnull align 8 dereferenceable(48) %0) unnamed_addr #2 comdat personality ptr @__gxx_personality_v0 {
  %2 = alloca %"class.std::tuple", align 8
  %3 = alloca %"class.std::tuple.2", align 1
  %4 = alloca %"class.std::basic_ifstream", align 8
  %5 = alloca [32 x i8], align 4
  %6 = alloca %"struct.std::pair", align 8
  %7 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i32 0, ptr %7, align 8, !tbaa !34
  %8 = getelementptr inbounds nuw i8, ptr %0, i64 16
  store ptr null, ptr %8, align 8, !tbaa !29
  %9 = getelementptr inbounds nuw i8, ptr %0, i64 24
  store ptr %7, ptr %9, align 8, !tbaa !35
  %10 = getelementptr inbounds nuw i8, ptr %0, i64 32
  store ptr %7, ptr %10, align 8, !tbaa !36
  %11 = getelementptr inbounds nuw i8, ptr %0, i64 40
  store i64 0, ptr %11, align 8, !tbaa !37
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #15
  invoke void @_ZNSt14basic_ifstreamIcSt11char_traitsIcEEC1EPKcSt13_Ios_Openmode(ptr noundef nonnull align 8 dereferenceable(264) %4, ptr noundef nonnull @.str, i32 noundef 8)
          to label %12 unwind label %141

12:                                               ; preds = %1
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #15
  %13 = load ptr, ptr %4, align 8, !tbaa !6
  %14 = getelementptr i8, ptr %13, i64 -24
  %15 = load i64, ptr %14, align 8
  %16 = getelementptr inbounds i8, ptr %4, i64 %15
  %17 = getelementptr inbounds nuw i8, ptr %16, i64 240
  %18 = load ptr, ptr %17, align 8, !tbaa !38
  %19 = icmp eq ptr %18, null
  br i1 %19, label %23, label %20

20:                                               ; preds = %12
  %21 = getelementptr inbounds nuw i8, ptr %4, i64 8
  %22 = getelementptr inbounds nuw i8, ptr %6, i64 8
  br label %25

23:                                               ; preds = %151, %12
  invoke void @_ZSt16__throw_bad_castv() #17
          to label %24 unwind label %145

24:                                               ; preds = %23
  unreachable

25:                                               ; preds = %20, %151
  %26 = phi ptr [ %18, %20 ], [ %157, %151 ]
  %27 = getelementptr inbounds nuw i8, ptr %26, i64 56
  %28 = load i8, ptr %27, align 8, !tbaa !39
  %29 = icmp eq i8 %28, 0
  br i1 %29, label %33, label %30

30:                                               ; preds = %25
  %31 = getelementptr inbounds nuw i8, ptr %26, i64 67
  %32 = load i8, ptr %31, align 1, !tbaa !45
  br label %39

33:                                               ; preds = %25
  invoke void @_ZNKSt5ctypeIcE13_M_widen_initEv(ptr noundef nonnull align 8 dereferenceable(570) %26)
          to label %34 unwind label %143

34:                                               ; preds = %33
  %35 = load ptr, ptr %26, align 8, !tbaa !6
  %36 = getelementptr inbounds nuw i8, ptr %35, i64 48
  %37 = load ptr, ptr %36, align 8
  %38 = invoke noundef i8 %37(ptr noundef nonnull align 8 dereferenceable(570) %26, i8 noundef 10)
          to label %39 unwind label %143

39:                                               ; preds = %34, %30
  %40 = phi i8 [ %32, %30 ], [ %38, %34 ]
  %41 = invoke noundef nonnull align 8 dereferenceable(16) ptr @_ZNSi7getlineEPclc(ptr noundef nonnull align 8 dereferenceable(16) %4, ptr noundef nonnull %5, i64 noundef 32, i8 noundef %40)
          to label %42 unwind label %143

42:                                               ; preds = %39
  %43 = load ptr, ptr %41, align 8, !tbaa !6
  %44 = getelementptr i8, ptr %43, i64 -24
  %45 = load i64, ptr %44, align 8
  %46 = getelementptr inbounds i8, ptr %41, i64 %45
  %47 = getelementptr inbounds nuw i8, ptr %46, i64 32
  %48 = load i32, ptr %47, align 8, !tbaa !46
  %49 = and i32 %48, 5
  %50 = icmp eq i32 %49, 0
  br i1 %50, label %51, label %159

51:                                               ; preds = %42
  %52 = load i64, ptr %21, align 8, !tbaa !47
  %53 = getelementptr inbounds i8, ptr %5, i64 %52
  %54 = getelementptr inbounds i8, ptr %53, i64 -1
  %55 = load ptr, ptr %8, align 8, !tbaa !29
  %56 = icmp eq ptr %55, null
  br i1 %56, label %88, label %57

57:                                               ; preds = %51, %70
  %58 = phi ptr [ %74, %70 ], [ %55, %51 ]
  %59 = phi ptr [ %72, %70 ], [ %7, %51 ]
  %60 = getelementptr inbounds nuw i8, ptr %58, i64 32
  %61 = load ptr, ptr %60, align 8, !tbaa !49
  %62 = icmp ult ptr %61, %5
  br i1 %62, label %69, label %63

63:                                               ; preds = %57
  %64 = icmp ult ptr %5, %61
  br i1 %64, label %70, label %65

65:                                               ; preds = %63
  %66 = getelementptr inbounds nuw i8, ptr %58, i64 40
  %67 = load ptr, ptr %66, align 8, !tbaa !52
  %68 = icmp ult ptr %67, %54
  br i1 %68, label %69, label %70

69:                                               ; preds = %65, %57
  br label %70

70:                                               ; preds = %69, %65, %63
  %71 = phi i64 [ 24, %69 ], [ 16, %63 ], [ 16, %65 ]
  %72 = phi ptr [ %59, %69 ], [ %58, %63 ], [ %58, %65 ]
  %73 = getelementptr inbounds nuw i8, ptr %58, i64 %71
  %74 = load ptr, ptr %73, align 8, !tbaa !53
  %75 = icmp eq ptr %74, null
  br i1 %75, label %76, label %57, !llvm.loop !54

76:                                               ; preds = %70
  %77 = icmp eq ptr %72, %7
  br i1 %77, label %88, label %78

78:                                               ; preds = %76
  %79 = getelementptr inbounds nuw i8, ptr %72, i64 32
  %80 = load ptr, ptr %79, align 8, !tbaa !49
  %81 = icmp ult ptr %5, %80
  br i1 %81, label %88, label %82

82:                                               ; preds = %78
  %83 = icmp ult ptr %80, %5
  br i1 %83, label %151, label %84

84:                                               ; preds = %82
  %85 = getelementptr inbounds nuw i8, ptr %72, i64 40
  %86 = load ptr, ptr %85, align 8, !tbaa !52
  %87 = icmp ult ptr %54, %86
  br i1 %87, label %88, label %151

88:                                               ; preds = %51, %78, %76, %84
  %89 = add nsw i64 %52, -1
  %90 = invoke noalias noundef nonnull ptr @_Znam(i64 noundef %89) #18
          to label %91 unwind label %147

91:                                               ; preds = %88
  %92 = icmp sgt i64 %52, 2
  br i1 %92, label %93, label %94, !prof !56

93:                                               ; preds = %91
  call void @llvm.memcpy.p0.p0.i64(ptr nonnull align 1 %90, ptr nonnull align 4 %5, i64 %89, i1 false)
  br label %98

94:                                               ; preds = %91
  %95 = icmp eq i64 %89, 1
  br i1 %95, label %96, label %98

96:                                               ; preds = %94
  %97 = load i8, ptr %5, align 4, !tbaa !45
  store i8 %97, ptr %90, align 1, !tbaa !45
  br label %98

98:                                               ; preds = %96, %94, %93
  call void @llvm.lifetime.start.p0(ptr nonnull %6) #15
  %99 = getelementptr inbounds nuw i8, ptr %90, i64 %89
  store ptr %90, ptr %6, align 8, !tbaa !49
  store ptr %99, ptr %22, align 8, !tbaa !52
  br i1 %56, label %131, label %100

100:                                              ; preds = %98, %113
  %101 = phi ptr [ %117, %113 ], [ %55, %98 ]
  %102 = phi ptr [ %115, %113 ], [ %7, %98 ]
  %103 = getelementptr inbounds nuw i8, ptr %101, i64 32
  %104 = load ptr, ptr %103, align 8, !tbaa !49
  %105 = icmp ult ptr %104, %90
  br i1 %105, label %112, label %106

106:                                              ; preds = %100
  %107 = icmp ult ptr %90, %104
  br i1 %107, label %113, label %108

108:                                              ; preds = %106
  %109 = getelementptr inbounds nuw i8, ptr %101, i64 40
  %110 = load ptr, ptr %109, align 8, !tbaa !52
  %111 = icmp ult ptr %110, %99
  br i1 %111, label %112, label %113

112:                                              ; preds = %108, %100
  br label %113

113:                                              ; preds = %112, %108, %106
  %114 = phi i64 [ 24, %112 ], [ 16, %106 ], [ 16, %108 ]
  %115 = phi ptr [ %102, %112 ], [ %101, %106 ], [ %101, %108 ]
  %116 = getelementptr inbounds nuw i8, ptr %101, i64 %114
  %117 = load ptr, ptr %116, align 8, !tbaa !53
  %118 = icmp eq ptr %117, null
  br i1 %118, label %119, label %100, !llvm.loop !54

119:                                              ; preds = %113
  %120 = icmp eq ptr %115, %7
  br i1 %120, label %131, label %121

121:                                              ; preds = %119
  %122 = getelementptr inbounds nuw i8, ptr %115, i64 32
  %123 = load ptr, ptr %122, align 8, !tbaa !49
  %124 = icmp ult ptr %90, %123
  br i1 %124, label %131, label %125

125:                                              ; preds = %121
  %126 = icmp ult ptr %123, %90
  br i1 %126, label %136, label %127

127:                                              ; preds = %125
  %128 = getelementptr inbounds nuw i8, ptr %115, i64 40
  %129 = load ptr, ptr %128, align 8, !tbaa !52
  %130 = icmp ult ptr %99, %129
  br i1 %130, label %131, label %136

131:                                              ; preds = %127, %121, %119, %98
  %132 = phi ptr [ %115, %127 ], [ %115, %119 ], [ %7, %98 ], [ %115, %121 ]
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #15
  store ptr %6, ptr %2, align 8, !tbaa !57, !alias.scope !59
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #15
  %133 = invoke i64 @_ZNSt8_Rb_treeISt4pairIPKcS2_ES0_IKS3_iESt10_Select1stIS5_ESt4lessIS3_ESaIS5_EE22_M_emplace_hint_uniqueIJRKSt21piecewise_construct_tSt5tupleIJOS3_EESG_IJEEEEESt17_Rb_tree_iteratorIS5_ESt23_Rb_tree_const_iteratorIS5_EDpOT_(ptr noundef nonnull align 8 dereferenceable(48) %0, ptr %132, ptr noundef nonnull align 1 dereferenceable(1) @_ZSt19piecewise_construct, ptr noundef nonnull align 8 dereferenceable(8) %2, ptr noundef nonnull align 1 dereferenceable(1) %3)
          to label %134 unwind label %149

134:                                              ; preds = %131
  %135 = inttoptr i64 %133 to ptr
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #15
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #15
  br label %136

136:                                              ; preds = %134, %127, %125
  %137 = phi ptr [ %135, %134 ], [ %115, %127 ], [ %115, %125 ]
  %138 = getelementptr inbounds nuw i8, ptr %137, i64 48
  %139 = load i32, ptr %138, align 4, !tbaa !62
  %140 = add nsw i32 %139, 1
  store i32 %140, ptr %138, align 4, !tbaa !62
  call void @llvm.lifetime.end.p0(ptr nonnull %6) #15
  br label %151

141:                                              ; preds = %1
  %142 = landingpad { ptr, i32 }
          cleanup
  br label %162

143:                                              ; preds = %33, %34, %39
  %144 = landingpad { ptr, i32 }
          cleanup
  br label %160

145:                                              ; preds = %23
  %146 = landingpad { ptr, i32 }
          cleanup
  br label %160

147:                                              ; preds = %88
  %148 = landingpad { ptr, i32 }
          cleanup
  br label %160

149:                                              ; preds = %131
  %150 = landingpad { ptr, i32 }
          cleanup
  call void @llvm.lifetime.end.p0(ptr nonnull %6) #15
  br label %160

151:                                              ; preds = %82, %84, %136
  %152 = load ptr, ptr %4, align 8, !tbaa !6
  %153 = getelementptr i8, ptr %152, i64 -24
  %154 = load i64, ptr %153, align 8
  %155 = getelementptr inbounds i8, ptr %4, i64 %154
  %156 = getelementptr inbounds nuw i8, ptr %155, i64 240
  %157 = load ptr, ptr %156, align 8, !tbaa !38
  %158 = icmp eq ptr %157, null
  br i1 %158, label %23, label %25, !llvm.loop !63

159:                                              ; preds = %42
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #15
  call void @_ZNSt14basic_ifstreamIcSt11char_traitsIcEED1Ev(ptr noundef nonnull align 8 dereferenceable(264) %4) #15
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #15
  ret void

160:                                              ; preds = %143, %145, %149, %147
  %161 = phi { ptr, i32 } [ %150, %149 ], [ %148, %147 ], [ %144, %143 ], [ %146, %145 ]
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #15
  call void @_ZNSt14basic_ifstreamIcSt11char_traitsIcEED1Ev(ptr noundef nonnull align 8 dereferenceable(264) %4) #15
  br label %162

162:                                              ; preds = %160, %141
  %163 = phi { ptr, i32 } [ %161, %160 ], [ %142, %141 ]
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #15
  call void @_ZNSt3mapISt4pairIPKcS2_EiSt4lessIS3_ESaIS0_IKS3_iEEED2Ev(ptr noundef nonnull align 8 dereferenceable(48) %0) #15
  resume { ptr, i32 } %163
}

declare i32 @__gxx_personality_v0(...)

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZN13spell_checker7processERSi(ptr noundef nonnull align 8 dereferenceable(48) %0, ptr noundef nonnull align 8 dereferenceable(16) %1) local_unnamed_addr #2 comdat {
  %3 = alloca i8, align 4
  %4 = alloca [32 x i8], align 1
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #15
  %5 = load ptr, ptr %1, align 8, !tbaa !6
  %6 = getelementptr i8, ptr %5, i64 -24
  %7 = load i64, ptr %6, align 8
  %8 = getelementptr inbounds i8, ptr %1, i64 %7
  %9 = getelementptr inbounds nuw i8, ptr %8, i64 240
  %10 = load ptr, ptr %9, align 8, !tbaa !38
  %11 = icmp eq ptr %10, null
  br i1 %11, label %16, label %12

12:                                               ; preds = %2
  %13 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %14 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %15 = getelementptr inbounds nuw i8, ptr %0, i64 16
  br label %17

16:                                               ; preds = %93, %2
  call void @_ZSt16__throw_bad_castv() #17
  unreachable

17:                                               ; preds = %12, %93
  %18 = phi ptr [ %10, %12 ], [ %99, %93 ]
  %19 = getelementptr inbounds nuw i8, ptr %18, i64 56
  %20 = load i8, ptr %19, align 8, !tbaa !39
  %21 = icmp eq i8 %20, 0
  br i1 %21, label %25, label %22

22:                                               ; preds = %17
  %23 = getelementptr inbounds nuw i8, ptr %18, i64 67
  %24 = load i8, ptr %23, align 1, !tbaa !45
  br label %30

25:                                               ; preds = %17
  call void @_ZNKSt5ctypeIcE13_M_widen_initEv(ptr noundef nonnull align 8 dereferenceable(570) %18)
  %26 = load ptr, ptr %18, align 8, !tbaa !6
  %27 = getelementptr inbounds nuw i8, ptr %26, i64 48
  %28 = load ptr, ptr %27, align 8
  %29 = call noundef i8 %28(ptr noundef nonnull align 8 dereferenceable(570) %18, i8 noundef 10)
  br label %30

30:                                               ; preds = %22, %25
  %31 = phi i8 [ %24, %22 ], [ %29, %25 ]
  %32 = call noundef nonnull align 8 dereferenceable(16) ptr @_ZNSi7getlineEPclc(ptr noundef nonnull align 8 dereferenceable(16) %1, ptr noundef nonnull %4, i64 noundef 32, i8 noundef %31)
  %33 = load ptr, ptr %32, align 8, !tbaa !6
  %34 = getelementptr i8, ptr %33, i64 -24
  %35 = load i64, ptr %34, align 8
  %36 = getelementptr inbounds i8, ptr %32, i64 %35
  %37 = getelementptr inbounds nuw i8, ptr %36, i64 32
  %38 = load i32, ptr %37, align 8, !tbaa !46
  %39 = and i32 %38, 5
  %40 = icmp eq i32 %39, 0
  br i1 %40, label %41, label %101

41:                                               ; preds = %30
  %42 = load i64, ptr %14, align 8, !tbaa !47
  %43 = getelementptr inbounds i8, ptr %4, i64 %42
  %44 = getelementptr inbounds i8, ptr %43, i64 -1
  %45 = load ptr, ptr %15, align 8, !tbaa !29
  %46 = icmp eq ptr %45, null
  br i1 %46, label %78, label %47

47:                                               ; preds = %41, %60
  %48 = phi ptr [ %64, %60 ], [ %45, %41 ]
  %49 = phi ptr [ %62, %60 ], [ %13, %41 ]
  %50 = getelementptr inbounds nuw i8, ptr %48, i64 32
  %51 = load ptr, ptr %50, align 8, !tbaa !49
  %52 = icmp ult ptr %51, %4
  br i1 %52, label %59, label %53

53:                                               ; preds = %47
  %54 = icmp ult ptr %4, %51
  br i1 %54, label %60, label %55

55:                                               ; preds = %53
  %56 = getelementptr inbounds nuw i8, ptr %48, i64 40
  %57 = load ptr, ptr %56, align 8, !tbaa !52
  %58 = icmp ult ptr %57, %44
  br i1 %58, label %59, label %60

59:                                               ; preds = %55, %47
  br label %60

60:                                               ; preds = %59, %55, %53
  %61 = phi i64 [ 24, %59 ], [ 16, %53 ], [ 16, %55 ]
  %62 = phi ptr [ %49, %59 ], [ %48, %53 ], [ %48, %55 ]
  %63 = getelementptr inbounds nuw i8, ptr %48, i64 %61
  %64 = load ptr, ptr %63, align 8, !tbaa !53
  %65 = icmp eq ptr %64, null
  br i1 %65, label %66, label %47, !llvm.loop !54

66:                                               ; preds = %60
  %67 = icmp eq ptr %62, %13
  br i1 %67, label %78, label %68

68:                                               ; preds = %66
  %69 = getelementptr inbounds nuw i8, ptr %62, i64 32
  %70 = load ptr, ptr %69, align 8, !tbaa !49
  %71 = icmp ult ptr %4, %70
  br i1 %71, label %78, label %72

72:                                               ; preds = %68
  %73 = icmp ult ptr %70, %4
  br i1 %73, label %93, label %74

74:                                               ; preds = %72
  %75 = getelementptr inbounds nuw i8, ptr %62, i64 40
  %76 = load ptr, ptr %75, align 8, !tbaa !52
  %77 = icmp ult ptr %44, %76
  br i1 %77, label %78, label %93

78:                                               ; preds = %41, %68, %66, %74
  %79 = call noundef i64 @strlen(ptr noundef nonnull dereferenceable(1) %4) #15
  %80 = call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull %4, i64 noundef %79)
  call void @llvm.lifetime.start.p0(ptr nonnull %3)
  store i8 10, ptr %3, align 4, !tbaa !45
  %81 = load ptr, ptr @_ZSt4cout, align 8, !tbaa !6
  %82 = getelementptr i8, ptr %81, i64 -24
  %83 = load i64, ptr %82, align 8
  %84 = getelementptr inbounds i8, ptr @_ZSt4cout, i64 %83
  %85 = getelementptr inbounds nuw i8, ptr %84, i64 16
  %86 = load i64, ptr %85, align 8, !tbaa !64
  %87 = icmp eq i64 %86, 0
  br i1 %87, label %90, label %88

88:                                               ; preds = %78
  %89 = call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull %3, i64 noundef 1)
  br label %92

90:                                               ; preds = %78
  %91 = call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, i8 noundef 10)
  br label %92

92:                                               ; preds = %88, %90
  call void @llvm.lifetime.end.p0(ptr nonnull %3)
  br label %93

93:                                               ; preds = %72, %74, %92
  %94 = load ptr, ptr %1, align 8, !tbaa !6
  %95 = getelementptr i8, ptr %94, i64 -24
  %96 = load i64, ptr %95, align 8
  %97 = getelementptr inbounds i8, ptr %1, i64 %96
  %98 = getelementptr inbounds nuw i8, ptr %97, i64 240
  %99 = load ptr, ptr %98, align 8, !tbaa !38
  %100 = icmp eq ptr %99, null
  br i1 %100, label %16, label %17, !llvm.loop !65

101:                                              ; preds = %30
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #15
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: inlinehint mustprogress nounwind uwtable
define linkonce_odr dso_local void @_ZN13spell_checkerD2Ev(ptr noundef nonnull align 8 dereferenceable(48) %0) unnamed_addr #3 comdat personality ptr @__gxx_personality_v0 {
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %3 = load ptr, ptr %2, align 8, !tbaa !29
  invoke void @_ZNSt8_Rb_treeISt4pairIPKcS2_ES0_IKS3_iESt10_Select1stIS5_ESt4lessIS3_ESaIS5_EE8_M_eraseEPSt13_Rb_tree_nodeIS5_E(ptr noundef nonnull align 8 dereferenceable(48) %0, ptr noundef %3)
          to label %7 unwind label %4

4:                                                ; preds = %1
  %5 = landingpad { ptr, i32 }
          catch ptr null
  %6 = extractvalue { ptr, i32 } %5, 0
  tail call void @__clang_call_terminate(ptr %6) #16
  unreachable

7:                                                ; preds = %1
  ret void
}

; Function Attrs: mustprogress uwtable
declare void @_ZNSt14basic_ifstreamIcSt11char_traitsIcEEC1EPKcSt13_Ios_Openmode(ptr noundef nonnull align 8 dereferenceable(264), ptr noundef, i32 noundef) unnamed_addr #2

; Function Attrs: nobuiltin allocsize(0)
declare noundef nonnull ptr @_Znam(i64 noundef) local_unnamed_addr #4

; Function Attrs: mustprogress nounwind uwtable
declare void @_ZNSt14basic_ifstreamIcSt11char_traitsIcEED1Ev(ptr noundef nonnull align 8 dereferenceable(264)) unnamed_addr #5

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local void @_ZNSt3mapISt4pairIPKcS2_EiSt4lessIS3_ESaIS0_IKS3_iEEED2Ev(ptr noundef nonnull align 8 dereferenceable(48) %0) unnamed_addr #5 comdat personality ptr @__gxx_personality_v0 {
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %3 = load ptr, ptr %2, align 8, !tbaa !29
  invoke void @_ZNSt8_Rb_treeISt4pairIPKcS2_ES0_IKS3_iESt10_Select1stIS5_ESt4lessIS3_ESaIS5_EE8_M_eraseEPSt13_Rb_tree_nodeIS5_E(ptr noundef nonnull align 8 dereferenceable(48) %0, ptr noundef %3)
          to label %7 unwind label %4

4:                                                ; preds = %1
  %5 = landingpad { ptr, i32 }
          catch ptr null
  %6 = extractvalue { ptr, i32 } %5, 0
  tail call void @__clang_call_terminate(ptr %6) #16
  unreachable

7:                                                ; preds = %1
  ret void
}

; Function Attrs: noinline noreturn nounwind uwtable
define linkonce_odr hidden void @__clang_call_terminate(ptr noundef %0) local_unnamed_addr #6 comdat {
  %2 = tail call ptr @__cxa_begin_catch(ptr %0) #15
  tail call void @_ZSt9terminatev() #16
  unreachable
}

declare ptr @__cxa_begin_catch(ptr) local_unnamed_addr

; Function Attrs: cold nofree noreturn
declare void @_ZSt9terminatev() local_unnamed_addr #7

declare noundef nonnull align 8 dereferenceable(16) ptr @_ZNSi7getlineEPclc(ptr noundef nonnull align 8 dereferenceable(16), ptr noundef, i64 noundef, i8 noundef) local_unnamed_addr #8

; Function Attrs: cold noreturn
declare void @_ZSt16__throw_bad_castv() local_unnamed_addr #9

declare void @_ZNKSt5ctypeIcE13_M_widen_initEv(ptr noundef nonnull align 8 dereferenceable(570)) local_unnamed_addr #8

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #10

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local i64 @_ZNSt8_Rb_treeISt4pairIPKcS2_ES0_IKS3_iESt10_Select1stIS5_ESt4lessIS3_ESaIS5_EE22_M_emplace_hint_uniqueIJRKSt21piecewise_construct_tSt5tupleIJOS3_EESG_IJEEEEESt17_Rb_tree_iteratorIS5_ESt23_Rb_tree_const_iteratorIS5_EDpOT_(ptr noundef nonnull align 8 dereferenceable(48) %0, ptr %1, ptr noundef nonnull align 1 dereferenceable(1) %2, ptr noundef nonnull align 8 dereferenceable(8) %3, ptr noundef nonnull align 1 dereferenceable(1) %4) local_unnamed_addr #2 comdat personality ptr @__gxx_personality_v0 {
  %6 = tail call noalias noundef nonnull dereferenceable(56) ptr @_Znwm(i64 noundef 56) #18
  %7 = getelementptr inbounds nuw i8, ptr %6, i64 32
  %8 = load i64, ptr %3, align 8, !tbaa !57
  %9 = inttoptr i64 %8 to ptr
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(20) %7, ptr noundef nonnull align 8 dereferenceable(16) %9, i64 16, i1 false)
  %10 = getelementptr inbounds nuw i8, ptr %6, i64 48
  store i32 0, ptr %10, align 8, !tbaa !66
  %11 = invoke [2 x i64] @_ZNSt8_Rb_treeISt4pairIPKcS2_ES0_IKS3_iESt10_Select1stIS5_ESt4lessIS3_ESaIS5_EE29_M_get_insert_hint_unique_posESt23_Rb_tree_const_iteratorIS5_ERS4_(ptr noundef nonnull align 8 dereferenceable(48) %0, ptr %1, ptr noundef nonnull align 8 dereferenceable(16) %7)
          to label %12 unwind label %41

12:                                               ; preds = %5
  %13 = extractvalue [2 x i64] %11, 1
  %14 = icmp eq i64 %13, 0
  %15 = extractvalue [2 x i64] %11, 0
  br i1 %14, label %43, label %16

16:                                               ; preds = %12
  %17 = inttoptr i64 %13 to ptr
  %18 = icmp ne i64 %15, 0
  %19 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %20 = icmp eq ptr %19, %17
  %21 = select i1 %18, i1 true, i1 %20
  br i1 %21, label %35, label %22

22:                                               ; preds = %16
  %23 = getelementptr inbounds nuw i8, ptr %17, i64 32
  %24 = load ptr, ptr %7, align 8, !tbaa !49
  %25 = load ptr, ptr %23, align 8, !tbaa !49
  %26 = icmp ult ptr %24, %25
  br i1 %26, label %35, label %27

27:                                               ; preds = %22
  %28 = icmp ult ptr %25, %24
  br i1 %28, label %35, label %29

29:                                               ; preds = %27
  %30 = getelementptr inbounds nuw i8, ptr %6, i64 40
  %31 = load ptr, ptr %30, align 8, !tbaa !52
  %32 = getelementptr inbounds nuw i8, ptr %17, i64 40
  %33 = load ptr, ptr %32, align 8, !tbaa !52
  %34 = icmp ult ptr %31, %33
  br label %35

35:                                               ; preds = %16, %22, %27, %29
  %36 = phi i1 [ true, %16 ], [ true, %22 ], [ false, %27 ], [ %34, %29 ]
  tail call void @_ZSt29_Rb_tree_insert_and_rebalancebPSt18_Rb_tree_node_baseS0_RS_(i1 noundef %36, ptr noundef nonnull %6, ptr noundef nonnull %17, ptr noundef nonnull align 8 dereferenceable(32) %19) #15
  %37 = getelementptr inbounds nuw i8, ptr %0, i64 40
  %38 = load i64, ptr %37, align 8, !tbaa !37
  %39 = add i64 %38, 1
  store i64 %39, ptr %37, align 8, !tbaa !37
  %40 = ptrtoint ptr %6 to i64
  br label %44

41:                                               ; preds = %5
  %42 = landingpad { ptr, i32 }
          cleanup
  tail call void @_ZdlPvm(ptr noundef nonnull %6, i64 noundef 56) #19
  resume { ptr, i32 } %42

43:                                               ; preds = %12
  tail call void @_ZdlPvm(ptr noundef nonnull %6, i64 noundef 56) #19
  br label %44

44:                                               ; preds = %35, %43
  %45 = phi i64 [ %40, %35 ], [ %15, %43 ]
  ret i64 %45
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local [2 x i64] @_ZNSt8_Rb_treeISt4pairIPKcS2_ES0_IKS3_iESt10_Select1stIS5_ESt4lessIS3_ESaIS5_EE29_M_get_insert_hint_unique_posESt23_Rb_tree_const_iteratorIS5_ERS4_(ptr noundef nonnull align 8 dereferenceable(48) %0, ptr %1, ptr noundef nonnull align 8 dereferenceable(16) %2) local_unnamed_addr #2 comdat {
  %4 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %5 = icmp eq ptr %4, %1
  br i1 %5, label %6, label %79

6:                                                ; preds = %3
  %7 = getelementptr inbounds nuw i8, ptr %0, i64 40
  %8 = load i64, ptr %7, align 8, !tbaa !37
  %9 = icmp eq i64 %8, 0
  br i1 %9, label %25, label %10

10:                                               ; preds = %6
  %11 = getelementptr inbounds nuw i8, ptr %0, i64 32
  %12 = load ptr, ptr %11, align 8, !tbaa !53
  %13 = getelementptr inbounds nuw i8, ptr %12, i64 32
  %14 = load ptr, ptr %13, align 8, !tbaa !49
  %15 = load ptr, ptr %2, align 8, !tbaa !49
  %16 = icmp ult ptr %14, %15
  br i1 %16, label %244, label %17

17:                                               ; preds = %10
  %18 = icmp ult ptr %15, %14
  br i1 %18, label %25, label %19

19:                                               ; preds = %17
  %20 = getelementptr inbounds nuw i8, ptr %12, i64 40
  %21 = load ptr, ptr %20, align 8, !tbaa !52
  %22 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %23 = load ptr, ptr %22, align 8, !tbaa !52
  %24 = icmp ult ptr %21, %23
  br i1 %24, label %244, label %25

25:                                               ; preds = %17, %19, %6
  %26 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %27 = load ptr, ptr %26, align 8, !tbaa !53
  %28 = icmp eq ptr %27, null
  br i1 %28, label %54, label %29

29:                                               ; preds = %25
  %30 = load ptr, ptr %2, align 8, !tbaa !49
  %31 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %32 = load ptr, ptr %31, align 8
  br label %33

33:                                               ; preds = %48, %29
  %34 = phi ptr [ %27, %29 ], [ %49, %48 ]
  %35 = getelementptr inbounds nuw i8, ptr %34, i64 32
  %36 = load ptr, ptr %35, align 8, !tbaa !49
  %37 = icmp ult ptr %30, %36
  br i1 %37, label %44, label %38

38:                                               ; preds = %33
  %39 = icmp ult ptr %36, %30
  br i1 %39, label %50, label %40

40:                                               ; preds = %38
  %41 = getelementptr inbounds nuw i8, ptr %34, i64 40
  %42 = load ptr, ptr %41, align 8, !tbaa !52
  %43 = icmp ult ptr %32, %42
  br i1 %43, label %44, label %50

44:                                               ; preds = %40, %33
  %45 = getelementptr inbounds nuw i8, ptr %34, i64 16
  %46 = load ptr, ptr %45, align 8, !tbaa !53
  %47 = icmp eq ptr %46, null
  br i1 %47, label %54, label %48

48:                                               ; preds = %44, %50
  %49 = phi ptr [ %46, %44 ], [ %52, %50 ]
  br label %33, !llvm.loop !68

50:                                               ; preds = %38, %40
  %51 = getelementptr inbounds nuw i8, ptr %34, i64 24
  %52 = load ptr, ptr %51, align 8, !tbaa !53
  %53 = icmp eq ptr %52, null
  br i1 %53, label %64, label %48

54:                                               ; preds = %44, %25
  %55 = phi ptr [ %4, %25 ], [ %34, %44 ]
  %56 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %57 = load ptr, ptr %56, align 8, !tbaa !35
  %58 = icmp eq ptr %55, %57
  br i1 %58, label %244, label %59

59:                                               ; preds = %54
  %60 = tail call noundef ptr @_ZSt18_Rb_tree_decrementPSt18_Rb_tree_node_base(ptr noundef nonnull %55) #20
  %61 = getelementptr inbounds nuw i8, ptr %60, i64 32
  %62 = load ptr, ptr %61, align 8, !tbaa !49
  %63 = load ptr, ptr %2, align 8, !tbaa !49
  br label %64

64:                                               ; preds = %50, %59
  %65 = phi ptr [ %63, %59 ], [ %30, %50 ]
  %66 = phi ptr [ %62, %59 ], [ %36, %50 ]
  %67 = phi ptr [ %55, %59 ], [ %34, %50 ]
  %68 = phi ptr [ %60, %59 ], [ %34, %50 ]
  %69 = icmp ult ptr %66, %65
  br i1 %69, label %244, label %70

70:                                               ; preds = %64
  %71 = icmp ult ptr %65, %66
  br i1 %71, label %78, label %72

72:                                               ; preds = %70
  %73 = getelementptr inbounds nuw i8, ptr %68, i64 40
  %74 = load ptr, ptr %73, align 8, !tbaa !52
  %75 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %76 = load ptr, ptr %75, align 8, !tbaa !52
  %77 = icmp ult ptr %74, %76
  br i1 %77, label %244, label %78

78:                                               ; preds = %72, %70
  br label %244

79:                                               ; preds = %3
  %80 = getelementptr inbounds nuw i8, ptr %1, i64 32
  %81 = load ptr, ptr %2, align 8, !tbaa !49
  %82 = load ptr, ptr %80, align 8, !tbaa !49
  %83 = icmp ult ptr %81, %82
  br i1 %83, label %92, label %84

84:                                               ; preds = %79
  %85 = icmp ult ptr %82, %81
  br i1 %85, label %170, label %86

86:                                               ; preds = %84
  %87 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %88 = load ptr, ptr %87, align 8, !tbaa !52
  %89 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %90 = load ptr, ptr %89, align 8, !tbaa !52
  %91 = icmp ult ptr %88, %90
  br i1 %91, label %92, label %164

92:                                               ; preds = %79, %86
  %93 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %94 = load ptr, ptr %93, align 8, !tbaa !53
  %95 = icmp eq ptr %94, %1
  br i1 %95, label %244, label %96

96:                                               ; preds = %92
  %97 = tail call noundef ptr @_ZSt18_Rb_tree_decrementPSt18_Rb_tree_node_base(ptr noundef nonnull %1) #20
  %98 = getelementptr inbounds nuw i8, ptr %97, i64 32
  %99 = load ptr, ptr %98, align 8, !tbaa !49
  %100 = icmp ult ptr %99, %81
  br i1 %100, label %109, label %101

101:                                              ; preds = %96
  %102 = icmp ult ptr %81, %99
  br i1 %102, label %115, label %103

103:                                              ; preds = %101
  %104 = getelementptr inbounds nuw i8, ptr %97, i64 40
  %105 = load ptr, ptr %104, align 8, !tbaa !52
  %106 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %107 = load ptr, ptr %106, align 8, !tbaa !52
  %108 = icmp ult ptr %105, %107
  br i1 %108, label %109, label %115

109:                                              ; preds = %96, %103
  %110 = getelementptr inbounds nuw i8, ptr %97, i64 24
  %111 = load ptr, ptr %110, align 8, !tbaa !69
  %112 = icmp eq ptr %111, null
  %113 = select i1 %112, ptr null, ptr %1
  %114 = select i1 %112, ptr %97, ptr %1
  br label %244

115:                                              ; preds = %101, %103
  %116 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %117 = load ptr, ptr %116, align 8, !tbaa !53
  %118 = icmp eq ptr %117, null
  br i1 %118, label %143, label %119

119:                                              ; preds = %115
  %120 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %121 = load ptr, ptr %120, align 8
  br label %122

122:                                              ; preds = %137, %119
  %123 = phi ptr [ %117, %119 ], [ %138, %137 ]
  %124 = getelementptr inbounds nuw i8, ptr %123, i64 32
  %125 = load ptr, ptr %124, align 8, !tbaa !49
  %126 = icmp ult ptr %81, %125
  br i1 %126, label %133, label %127

127:                                              ; preds = %122
  %128 = icmp ult ptr %125, %81
  br i1 %128, label %139, label %129

129:                                              ; preds = %127
  %130 = getelementptr inbounds nuw i8, ptr %123, i64 40
  %131 = load ptr, ptr %130, align 8, !tbaa !52
  %132 = icmp ult ptr %121, %131
  br i1 %132, label %133, label %139

133:                                              ; preds = %129, %122
  %134 = getelementptr inbounds nuw i8, ptr %123, i64 16
  %135 = load ptr, ptr %134, align 8, !tbaa !53
  %136 = icmp eq ptr %135, null
  br i1 %136, label %143, label %137

137:                                              ; preds = %133, %139
  %138 = phi ptr [ %135, %133 ], [ %141, %139 ]
  br label %122, !llvm.loop !68

139:                                              ; preds = %127, %129
  %140 = getelementptr inbounds nuw i8, ptr %123, i64 24
  %141 = load ptr, ptr %140, align 8, !tbaa !53
  %142 = icmp eq ptr %141, null
  br i1 %142, label %150, label %137

143:                                              ; preds = %133, %115
  %144 = phi ptr [ %4, %115 ], [ %123, %133 ]
  %145 = icmp eq ptr %144, %94
  br i1 %145, label %244, label %146

146:                                              ; preds = %143
  %147 = tail call noundef ptr @_ZSt18_Rb_tree_decrementPSt18_Rb_tree_node_base(ptr noundef nonnull %144) #20
  %148 = getelementptr inbounds nuw i8, ptr %147, i64 32
  %149 = load ptr, ptr %148, align 8, !tbaa !49
  br label %150

150:                                              ; preds = %139, %146
  %151 = phi ptr [ %149, %146 ], [ %125, %139 ]
  %152 = phi ptr [ %144, %146 ], [ %123, %139 ]
  %153 = phi ptr [ %147, %146 ], [ %123, %139 ]
  %154 = icmp ult ptr %151, %81
  br i1 %154, label %244, label %155

155:                                              ; preds = %150
  %156 = icmp ult ptr %81, %151
  br i1 %156, label %163, label %157

157:                                              ; preds = %155
  %158 = getelementptr inbounds nuw i8, ptr %153, i64 40
  %159 = load ptr, ptr %158, align 8, !tbaa !52
  %160 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %161 = load ptr, ptr %160, align 8, !tbaa !52
  %162 = icmp ult ptr %159, %161
  br i1 %162, label %244, label %163

163:                                              ; preds = %157, %155
  br label %244

164:                                              ; preds = %86
  %165 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %166 = load ptr, ptr %165, align 8, !tbaa !52
  %167 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %168 = load ptr, ptr %167, align 8, !tbaa !52
  %169 = icmp ult ptr %166, %168
  br i1 %169, label %170, label %244

170:                                              ; preds = %84, %164
  %171 = getelementptr inbounds nuw i8, ptr %0, i64 32
  %172 = load ptr, ptr %171, align 8, !tbaa !53
  %173 = icmp eq ptr %172, %1
  br i1 %173, label %244, label %174

174:                                              ; preds = %170
  %175 = tail call noundef ptr @_ZSt18_Rb_tree_incrementPSt18_Rb_tree_node_base(ptr noundef nonnull %1) #20
  %176 = getelementptr inbounds nuw i8, ptr %175, i64 32
  %177 = load ptr, ptr %176, align 8, !tbaa !49
  %178 = icmp ult ptr %81, %177
  br i1 %178, label %187, label %179

179:                                              ; preds = %174
  %180 = icmp ult ptr %177, %81
  br i1 %180, label %193, label %181

181:                                              ; preds = %179
  %182 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %183 = load ptr, ptr %182, align 8, !tbaa !52
  %184 = getelementptr inbounds nuw i8, ptr %175, i64 40
  %185 = load ptr, ptr %184, align 8, !tbaa !52
  %186 = icmp ult ptr %183, %185
  br i1 %186, label %187, label %193

187:                                              ; preds = %174, %181
  %188 = getelementptr inbounds nuw i8, ptr %1, i64 24
  %189 = load ptr, ptr %188, align 8, !tbaa !69
  %190 = icmp eq ptr %189, null
  %191 = select i1 %190, ptr null, ptr %175
  %192 = select i1 %190, ptr %1, ptr %175
  br label %244

193:                                              ; preds = %179, %181
  %194 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %195 = load ptr, ptr %194, align 8, !tbaa !53
  %196 = icmp eq ptr %195, null
  br i1 %196, label %221, label %197

197:                                              ; preds = %193
  %198 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %199 = load ptr, ptr %198, align 8
  br label %200

200:                                              ; preds = %215, %197
  %201 = phi ptr [ %195, %197 ], [ %216, %215 ]
  %202 = getelementptr inbounds nuw i8, ptr %201, i64 32
  %203 = load ptr, ptr %202, align 8, !tbaa !49
  %204 = icmp ult ptr %81, %203
  br i1 %204, label %211, label %205

205:                                              ; preds = %200
  %206 = icmp ult ptr %203, %81
  br i1 %206, label %217, label %207

207:                                              ; preds = %205
  %208 = getelementptr inbounds nuw i8, ptr %201, i64 40
  %209 = load ptr, ptr %208, align 8, !tbaa !52
  %210 = icmp ult ptr %199, %209
  br i1 %210, label %211, label %217

211:                                              ; preds = %207, %200
  %212 = getelementptr inbounds nuw i8, ptr %201, i64 16
  %213 = load ptr, ptr %212, align 8, !tbaa !53
  %214 = icmp eq ptr %213, null
  br i1 %214, label %221, label %215

215:                                              ; preds = %211, %217
  %216 = phi ptr [ %213, %211 ], [ %219, %217 ]
  br label %200, !llvm.loop !68

217:                                              ; preds = %205, %207
  %218 = getelementptr inbounds nuw i8, ptr %201, i64 24
  %219 = load ptr, ptr %218, align 8, !tbaa !53
  %220 = icmp eq ptr %219, null
  br i1 %220, label %230, label %215

221:                                              ; preds = %211, %193
  %222 = phi ptr [ %4, %193 ], [ %201, %211 ]
  %223 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %224 = load ptr, ptr %223, align 8, !tbaa !35
  %225 = icmp eq ptr %222, %224
  br i1 %225, label %244, label %226

226:                                              ; preds = %221
  %227 = tail call noundef ptr @_ZSt18_Rb_tree_decrementPSt18_Rb_tree_node_base(ptr noundef nonnull %222) #20
  %228 = getelementptr inbounds nuw i8, ptr %227, i64 32
  %229 = load ptr, ptr %228, align 8, !tbaa !49
  br label %230

230:                                              ; preds = %217, %226
  %231 = phi ptr [ %229, %226 ], [ %203, %217 ]
  %232 = phi ptr [ %222, %226 ], [ %201, %217 ]
  %233 = phi ptr [ %227, %226 ], [ %201, %217 ]
  %234 = icmp ult ptr %231, %81
  br i1 %234, label %244, label %235

235:                                              ; preds = %230
  %236 = icmp ult ptr %81, %231
  br i1 %236, label %243, label %237

237:                                              ; preds = %235
  %238 = getelementptr inbounds nuw i8, ptr %233, i64 40
  %239 = load ptr, ptr %238, align 8, !tbaa !52
  %240 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %241 = load ptr, ptr %240, align 8, !tbaa !52
  %242 = icmp ult ptr %239, %241
  br i1 %242, label %244, label %243

243:                                              ; preds = %237, %235
  br label %244

244:                                              ; preds = %243, %237, %230, %221, %163, %157, %150, %143, %78, %72, %64, %54, %187, %109, %10, %164, %170, %92, %19
  %245 = phi ptr [ null, %19 ], [ %1, %92 ], [ null, %170 ], [ %1, %164 ], [ null, %10 ], [ %113, %109 ], [ %191, %187 ], [ %68, %78 ], [ null, %54 ], [ null, %72 ], [ null, %64 ], [ %153, %163 ], [ null, %143 ], [ null, %157 ], [ null, %150 ], [ %233, %243 ], [ null, %221 ], [ null, %237 ], [ null, %230 ]
  %246 = phi ptr [ %12, %19 ], [ %1, %92 ], [ %1, %170 ], [ null, %164 ], [ %12, %10 ], [ %114, %109 ], [ %192, %187 ], [ null, %78 ], [ %55, %54 ], [ %67, %72 ], [ %67, %64 ], [ null, %163 ], [ %94, %143 ], [ %152, %157 ], [ %152, %150 ], [ null, %243 ], [ %222, %221 ], [ %232, %237 ], [ %232, %230 ]
  %247 = ptrtoint ptr %245 to i64
  %248 = insertvalue [2 x i64] poison, i64 %247, 0
  %249 = ptrtoint ptr %246 to i64
  %250 = insertvalue [2 x i64] %248, i64 %249, 1
  ret [2 x i64] %250
}

; Function Attrs: nobuiltin allocsize(0)
declare noundef nonnull ptr @_Znwm(i64 noundef) local_unnamed_addr #4

; Function Attrs: nobuiltin nounwind
declare void @_ZdlPvm(ptr noundef, i64 noundef) local_unnamed_addr #11

; Function Attrs: mustprogress nofree nounwind willreturn memory(read)
declare noundef ptr @_ZSt18_Rb_tree_decrementPSt18_Rb_tree_node_base(ptr noundef) local_unnamed_addr #12

; Function Attrs: mustprogress nofree nounwind willreturn memory(read)
declare noundef ptr @_ZSt18_Rb_tree_incrementPSt18_Rb_tree_node_base(ptr noundef) local_unnamed_addr #12

; Function Attrs: nounwind
declare void @_ZSt29_Rb_tree_insert_and_rebalancebPSt18_Rb_tree_node_baseS0_RS_(i1 noundef, ptr noundef, ptr noundef, ptr noundef nonnull align 8 dereferenceable(32)) local_unnamed_addr #13

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZNSt8_Rb_treeISt4pairIPKcS2_ES0_IKS3_iESt10_Select1stIS5_ESt4lessIS3_ESaIS5_EE8_M_eraseEPSt13_Rb_tree_nodeIS5_E(ptr noundef nonnull align 8 dereferenceable(48) %0, ptr noundef %1) local_unnamed_addr #2 comdat personality ptr @__gxx_personality_v0 {
  %3 = icmp eq ptr %1, null
  br i1 %3, label %11, label %4

4:                                                ; preds = %2, %4
  %5 = phi ptr [ %9, %4 ], [ %1, %2 ]
  %6 = getelementptr inbounds nuw i8, ptr %5, i64 24
  %7 = load ptr, ptr %6, align 8, !tbaa !69
  tail call void @_ZNSt8_Rb_treeISt4pairIPKcS2_ES0_IKS3_iESt10_Select1stIS5_ESt4lessIS3_ESaIS5_EE8_M_eraseEPSt13_Rb_tree_nodeIS5_E(ptr noundef nonnull align 8 dereferenceable(48) %0, ptr noundef %7)
  %8 = getelementptr inbounds nuw i8, ptr %5, i64 16
  %9 = load ptr, ptr %8, align 8, !tbaa !70
  tail call void @_ZdlPvm(ptr noundef nonnull %5, i64 noundef 56) #19
  %10 = icmp eq ptr %9, null
  br i1 %10, label %11, label %4, !llvm.loop !71

11:                                               ; preds = %4, %2
  ret void
}

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8), ptr noundef, i64 noundef) local_unnamed_addr #8

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8), i8 noundef) local_unnamed_addr #8

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: read)
declare i64 @strlen(ptr noundef captures(none)) local_unnamed_addr #14

attributes #0 = { mustprogress norecurse uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { inlinehint mustprogress nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nobuiltin allocsize(0) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { mustprogress nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { noinline noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { cold nofree noreturn }
attributes #8 = { "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #9 = { cold noreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #10 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #11 = { nobuiltin nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #12 = { mustprogress nofree nounwind willreturn memory(read) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #13 = { nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #14 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: read) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #15 = { nounwind }
attributes #16 = { noreturn nounwind }
attributes #17 = { cold noreturn }
attributes #18 = { builtin allocsize(0) }
attributes #19 = { builtin nounwind }
attributes #20 = { nounwind willreturn memory(read) }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"vtable pointer", !8, i64 0}
!8 = !{!"Simple C++ TBAA"}
!9 = !{!10, !25, i64 232}
!10 = !{!"_ZTSSt9basic_iosIcSt11char_traitsIcEE", !11, i64 0, !23, i64 216, !13, i64 224, !24, i64 225, !25, i64 232, !26, i64 240, !27, i64 248, !28, i64 256}
!11 = !{!"_ZTSSt8ios_base", !12, i64 8, !12, i64 16, !14, i64 24, !15, i64 28, !15, i64 32, !16, i64 40, !18, i64 48, !13, i64 64, !19, i64 192, !20, i64 200, !21, i64 208}
!12 = !{!"long", !13, i64 0}
!13 = !{!"omnipotent char", !8, i64 0}
!14 = !{!"_ZTSSt13_Ios_Fmtflags", !13, i64 0}
!15 = !{!"_ZTSSt12_Ios_Iostate", !13, i64 0}
!16 = !{!"p1 _ZTSNSt8ios_base14_Callback_listE", !17, i64 0}
!17 = !{!"any pointer", !13, i64 0}
!18 = !{!"_ZTSNSt8ios_base6_WordsE", !17, i64 0, !12, i64 8}
!19 = !{!"int", !13, i64 0}
!20 = !{!"p1 _ZTSNSt8ios_base6_WordsE", !17, i64 0}
!21 = !{!"_ZTSSt6locale", !22, i64 0}
!22 = !{!"p1 _ZTSNSt6locale5_ImplE", !17, i64 0}
!23 = !{!"p1 _ZTSSo", !17, i64 0}
!24 = !{!"bool", !13, i64 0}
!25 = !{!"p1 _ZTSSt15basic_streambufIcSt11char_traitsIcEE", !17, i64 0}
!26 = !{!"p1 _ZTSSt5ctypeIcE", !17, i64 0}
!27 = !{!"p1 _ZTSSt7num_putIcSt19ostreambuf_iteratorIcSt11char_traitsIcEEE", !17, i64 0}
!28 = !{!"p1 _ZTSSt7num_getIcSt19istreambuf_iteratorIcSt11char_traitsIcEEE", !17, i64 0}
!29 = !{!30, !33, i64 8}
!30 = !{!"_ZTSSt15_Rb_tree_header", !31, i64 0, !12, i64 32}
!31 = !{!"_ZTSSt18_Rb_tree_node_base", !32, i64 0, !33, i64 8, !33, i64 16, !33, i64 24}
!32 = !{!"_ZTSSt14_Rb_tree_color", !13, i64 0}
!33 = !{!"p1 _ZTSSt18_Rb_tree_node_base", !17, i64 0}
!34 = !{!30, !32, i64 0}
!35 = !{!30, !33, i64 16}
!36 = !{!30, !33, i64 24}
!37 = !{!30, !12, i64 32}
!38 = !{!10, !26, i64 240}
!39 = !{!40, !13, i64 56}
!40 = !{!"_ZTSSt5ctypeIcE", !41, i64 0, !42, i64 16, !24, i64 24, !43, i64 32, !43, i64 40, !44, i64 48, !13, i64 56, !13, i64 57, !13, i64 313, !13, i64 569}
!41 = !{!"_ZTSNSt6locale5facetE", !19, i64 8}
!42 = !{!"p1 _ZTS15__locale_struct", !17, i64 0}
!43 = !{!"p1 int", !17, i64 0}
!44 = !{!"p1 short", !17, i64 0}
!45 = !{!13, !13, i64 0}
!46 = !{!11, !15, i64 32}
!47 = !{!48, !12, i64 8}
!48 = !{!"_ZTSSi", !12, i64 8}
!49 = !{!50, !51, i64 0}
!50 = !{!"_ZTSSt4pairIPKcS1_E", !51, i64 0, !51, i64 8}
!51 = !{!"p1 omnipotent char", !17, i64 0}
!52 = !{!50, !51, i64 8}
!53 = !{!33, !33, i64 0}
!54 = distinct !{!54, !55}
!55 = !{!"llvm.loop.mustprogress"}
!56 = !{!"branch_weights", !"expected", i32 2000, i32 1}
!57 = !{!58, !58, i64 0}
!58 = !{!"p1 _ZTSSt4pairIPKcS1_E", !17, i64 0}
!59 = !{!60}
!60 = distinct !{!60, !61, !"_ZSt16forward_as_tupleIJSt4pairIPKcS2_EEESt5tupleIJDpOT_EES7_: argument 0"}
!61 = distinct !{!61, !"_ZSt16forward_as_tupleIJSt4pairIPKcS2_EEESt5tupleIJDpOT_EES7_"}
!62 = !{!19, !19, i64 0}
!63 = distinct !{!63, !55}
!64 = !{!11, !12, i64 16}
!65 = distinct !{!65, !55}
!66 = !{!67, !19, i64 16}
!67 = !{!"_ZTSSt4pairIKS_IPKcS1_EiE", !50, i64 0, !19, i64 16}
!68 = distinct !{!68, !55}
!69 = !{!31, !33, i64 24}
!70 = !{!31, !33, i64 16}
!71 = distinct !{!71, !55}
