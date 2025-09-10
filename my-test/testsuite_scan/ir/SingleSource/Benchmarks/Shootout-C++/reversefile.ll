; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Shootout-C++/reversefile.cpp'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Shootout-C++/reversefile.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

module asm ".globl _ZSt21ios_base_library_initv"

%"class.std::basic_istream" = type { ptr, i64, %"class.std::basic_ios" }
%"class.std::basic_ios" = type { %"class.std::ios_base", ptr, i8, i8, ptr, ptr, ptr, ptr }
%"class.std::ios_base" = type { ptr, i64, i64, i32, i32, i32, ptr, %"struct.std::ios_base::_Words", [8 x %"struct.std::ios_base::_Words"], i32, ptr, %"class.std::locale" }
%"struct.std::ios_base::_Words" = type { ptr, i64 }
%"class.std::locale" = type { ptr }
%"class.std::basic_ostream" = type { ptr, %"class.std::basic_ios" }
%"struct.std::_Deque_iterator" = type { ptr, ptr, ptr, ptr }
%"class.std::ostream_iterator" = type { ptr, ptr }
%"class.std::deque" = type { %"class.std::_Deque_base" }
%"class.std::_Deque_base" = type { %"struct.std::_Deque_base<std::__cxx11::basic_string<char>, std::allocator<std::__cxx11::basic_string<char>>>::_Deque_impl" }
%"struct.std::_Deque_base<std::__cxx11::basic_string<char>, std::allocator<std::__cxx11::basic_string<char>>>::_Deque_impl" = type { %"struct.std::_Deque_base<std::__cxx11::basic_string<char>, std::allocator<std::__cxx11::basic_string<char>>>::_Deque_impl_data" }
%"struct.std::_Deque_base<std::__cxx11::basic_string<char>, std::allocator<std::__cxx11::basic_string<char>>>::_Deque_impl_data" = type { ptr, i64, %"struct.std::_Deque_iterator", %"struct.std::_Deque_iterator" }
%"class.std::__cxx11::basic_string" = type { %"struct.std::__cxx11::basic_string<char>::_Alloc_hider", i64, %union.anon }
%"struct.std::__cxx11::basic_string<char>::_Alloc_hider" = type { ptr }
%union.anon = type { i64, [8 x i8] }

$_ZNSt5dequeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESaIS5_EED2Ev = comdat any

$_ZNSt11_Deque_baseINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESaIS5_EE17_M_initialize_mapEm = comdat any

$__clang_call_terminate = comdat any

$_ZNSt5dequeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESaIS5_EE19_M_destroy_data_auxESt15_Deque_iteratorIS5_RS5_PS5_ESB_ = comdat any

$_ZNSt5dequeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESaIS5_EE17_M_push_front_auxIJS5_EEEvDpOT_ = comdat any

$_ZNSt5dequeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESaIS5_EE17_M_reallocate_mapEmb = comdat any

$_ZSt15__copy_move_ditILb0ENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERS5_PS5_St16ostream_iteratorIS5_cS3_EET3_St15_Deque_iteratorIT0_T1_T2_ESF_SA_ = comdat any

@_ZSt3cin = external global %"class.std::basic_istream", align 8
@_ZSt4cout = external global %"class.std::basic_ostream", align 8
@.str = private unnamed_addr constant [2 x i8] c"\0A\00", align 1
@.str.1 = private unnamed_addr constant [48 x i8] c"cannot create std::deque larger than max_size()\00", align 1

; Function Attrs: mustprogress norecurse uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 personality ptr @__gxx_personality_v0 {
  %1 = alloca %"struct.std::_Deque_iterator", align 16
  %2 = alloca %"struct.std::_Deque_iterator", align 16
  %3 = alloca %"class.std::ostream_iterator", align 8
  %4 = alloca %"class.std::ostream_iterator", align 8
  %5 = alloca i64, align 8
  %6 = alloca %"class.std::deque", align 8
  %7 = alloca [256 x i8], align 4
  %8 = alloca [4096 x i8], align 1
  %9 = alloca %"class.std::__cxx11::basic_string", align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %6) #16
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(80) %6, i8 0, i64 80, i1 false)
  call void @_ZNSt11_Deque_baseINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESaIS5_EE17_M_initialize_mapEm(ptr noundef nonnull align 8 dereferenceable(80) %6, i64 noundef 0)
  call void @llvm.lifetime.start.p0(ptr nonnull %7) #16
  call void @llvm.lifetime.start.p0(ptr nonnull %8) #16
  %10 = load ptr, ptr @_ZSt3cin, align 8, !tbaa !6
  %11 = getelementptr i8, ptr %10, i64 -24
  %12 = load i64, ptr %11, align 8
  %13 = getelementptr inbounds i8, ptr @_ZSt3cin, i64 %12
  %14 = getelementptr inbounds nuw i8, ptr %13, i64 232
  %15 = load ptr, ptr %14, align 8, !tbaa !9
  %16 = load ptr, ptr %15, align 8, !tbaa !6
  %17 = getelementptr inbounds nuw i8, ptr %16, i64 24
  %18 = load ptr, ptr %17, align 8
  %19 = invoke noundef ptr %18(ptr noundef nonnull align 8 dereferenceable(64) %15, ptr noundef nonnull %8, i64 noundef 4096)
          to label %20 unwind label %127

20:                                               ; preds = %0
  %21 = load ptr, ptr @_ZSt4cout, align 8, !tbaa !6
  %22 = getelementptr i8, ptr %21, i64 -24
  %23 = load i64, ptr %22, align 8
  %24 = getelementptr inbounds i8, ptr @_ZSt4cout, i64 %23
  %25 = getelementptr inbounds nuw i8, ptr %24, i64 232
  %26 = load ptr, ptr %25, align 8, !tbaa !9
  %27 = load ptr, ptr %26, align 8, !tbaa !6
  %28 = getelementptr inbounds nuw i8, ptr %27, i64 24
  %29 = load ptr, ptr %28, align 8
  %30 = invoke noundef ptr %29(ptr noundef nonnull align 8 dereferenceable(64) %26, ptr noundef nonnull %8, i64 noundef 4096)
          to label %31 unwind label %127

31:                                               ; preds = %20
  %32 = load ptr, ptr @_ZSt3cin, align 8, !tbaa !6
  %33 = getelementptr i8, ptr %32, i64 -24
  %34 = load i64, ptr %33, align 8
  %35 = getelementptr inbounds i8, ptr @_ZSt3cin, i64 %34
  %36 = getelementptr inbounds nuw i8, ptr %35, i64 240
  %37 = load ptr, ptr %36, align 8, !tbaa !29
  %38 = icmp eq ptr %37, null
  br i1 %38, label %44, label %39

39:                                               ; preds = %31
  %40 = getelementptr inbounds nuw i8, ptr %9, i64 16
  %41 = getelementptr inbounds nuw i8, ptr %9, i64 8
  %42 = getelementptr inbounds nuw i8, ptr %6, i64 16
  %43 = getelementptr inbounds nuw i8, ptr %6, i64 24
  br label %46

44:                                               ; preds = %117, %31
  invoke void @_ZSt16__throw_bad_castv() #17
          to label %45 unwind label %127

45:                                               ; preds = %44
  unreachable

46:                                               ; preds = %39, %117
  %47 = phi ptr [ %37, %39 ], [ %123, %117 ]
  %48 = getelementptr inbounds nuw i8, ptr %47, i64 56
  %49 = load i8, ptr %48, align 8, !tbaa !30
  %50 = icmp eq i8 %49, 0
  br i1 %50, label %54, label %51

51:                                               ; preds = %46
  %52 = getelementptr inbounds nuw i8, ptr %47, i64 67
  %53 = load i8, ptr %52, align 1, !tbaa !36
  br label %60

54:                                               ; preds = %46
  invoke void @_ZNKSt5ctypeIcE13_M_widen_initEv(ptr noundef nonnull align 8 dereferenceable(570) %47)
          to label %55 unwind label %125

55:                                               ; preds = %54
  %56 = load ptr, ptr %47, align 8, !tbaa !6
  %57 = getelementptr inbounds nuw i8, ptr %56, i64 48
  %58 = load ptr, ptr %57, align 8
  %59 = invoke noundef i8 %58(ptr noundef nonnull align 8 dereferenceable(570) %47, i8 noundef 10)
          to label %60 unwind label %125

60:                                               ; preds = %55, %51
  %61 = phi i8 [ %53, %51 ], [ %59, %55 ]
  %62 = invoke noundef nonnull align 8 dereferenceable(16) ptr @_ZNSi7getlineEPclc(ptr noundef nonnull align 8 dereferenceable(16) @_ZSt3cin, ptr noundef nonnull %7, i64 noundef 256, i8 noundef %61)
          to label %63 unwind label %125

63:                                               ; preds = %60
  %64 = load ptr, ptr %62, align 8, !tbaa !6
  %65 = getelementptr i8, ptr %64, i64 -24
  %66 = load i64, ptr %65, align 8
  %67 = getelementptr inbounds i8, ptr %62, i64 %66
  %68 = getelementptr inbounds nuw i8, ptr %67, i64 32
  %69 = load i32, ptr %68, align 8, !tbaa !37
  %70 = and i32 %69, 5
  %71 = icmp eq i32 %70, 0
  br i1 %71, label %72, label %143

72:                                               ; preds = %63
  call void @llvm.lifetime.start.p0(ptr nonnull %9) #16
  store ptr %40, ptr %9, align 8, !tbaa !38
  %73 = call noundef i64 @strlen(ptr noundef nonnull dereferenceable(1) %7) #16
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #16
  store i64 %73, ptr %5, align 8, !tbaa !41
  %74 = icmp ugt i64 %73, 15
  br i1 %74, label %75, label %79

75:                                               ; preds = %72
  %76 = invoke noundef ptr @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm(ptr noundef nonnull align 8 dereferenceable(32) %9, ptr noundef nonnull align 8 dereferenceable(8) %5, i64 noundef 0)
          to label %77 unwind label %129

77:                                               ; preds = %75
  store ptr %76, ptr %9, align 8, !tbaa !42
  %78 = load i64, ptr %5, align 8, !tbaa !41
  store i64 %78, ptr %40, align 8, !tbaa !36
  br label %79

79:                                               ; preds = %77, %72
  %80 = phi ptr [ %76, %77 ], [ %40, %72 ]
  switch i64 %73, label %83 [
    i64 1, label %81
    i64 0, label %84
  ]

81:                                               ; preds = %79
  %82 = load i8, ptr %7, align 4, !tbaa !36
  store i8 %82, ptr %80, align 1, !tbaa !36
  br label %84

83:                                               ; preds = %79
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %80, ptr nonnull align 4 %7, i64 %73, i1 false)
  br label %84

84:                                               ; preds = %83, %81, %79
  %85 = load i64, ptr %5, align 8, !tbaa !41
  store i64 %85, ptr %41, align 8, !tbaa !44
  %86 = load ptr, ptr %9, align 8, !tbaa !42
  %87 = getelementptr inbounds nuw i8, ptr %86, i64 %85
  store i8 0, ptr %87, align 1, !tbaa !36
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #16
  %88 = load ptr, ptr %42, align 8, !tbaa !45
  %89 = load ptr, ptr %43, align 8, !tbaa !51
  %90 = icmp eq ptr %88, %89
  br i1 %90, label %107, label %91

91:                                               ; preds = %84
  %92 = getelementptr inbounds i8, ptr %88, i64 -32
  %93 = getelementptr inbounds i8, ptr %88, i64 -16
  store ptr %93, ptr %92, align 8, !tbaa !38
  %94 = load ptr, ptr %9, align 8, !tbaa !42
  %95 = icmp eq ptr %94, %40
  br i1 %95, label %96, label %100

96:                                               ; preds = %91
  %97 = load i64, ptr %41, align 8, !tbaa !44
  %98 = icmp ult i64 %97, 16
  call void @llvm.assume(i1 %98)
  %99 = add nuw nsw i64 %97, 1
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(1) %93, ptr noundef nonnull align 8 dereferenceable(1) %40, i64 %99, i1 false)
  br label %102

100:                                              ; preds = %91
  store ptr %94, ptr %92, align 8, !tbaa !42
  %101 = load i64, ptr %40, align 8, !tbaa !36
  store i64 %101, ptr %93, align 8, !tbaa !36
  br label %102

102:                                              ; preds = %96, %100
  %103 = load i64, ptr %41, align 8, !tbaa !44
  %104 = getelementptr inbounds i8, ptr %88, i64 -24
  store i64 %103, ptr %104, align 8, !tbaa !44
  store ptr %40, ptr %9, align 8, !tbaa !42
  store i64 0, ptr %41, align 8, !tbaa !44
  %105 = load ptr, ptr %42, align 8, !tbaa !45
  %106 = getelementptr inbounds i8, ptr %105, i64 -32
  store ptr %106, ptr %42, align 8, !tbaa !45
  br label %111

107:                                              ; preds = %84
  invoke void @_ZNSt5dequeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESaIS5_EE17_M_push_front_auxIJS5_EEEvDpOT_(ptr noundef nonnull align 8 dereferenceable(80) %6, ptr noundef nonnull align 8 dereferenceable(32) %9)
          to label %108 unwind label %131

108:                                              ; preds = %107
  %109 = load ptr, ptr %9, align 8, !tbaa !42
  %110 = icmp eq ptr %109, %40
  br i1 %110, label %111, label %114

111:                                              ; preds = %102, %108
  %112 = load i64, ptr %41, align 8, !tbaa !44
  %113 = icmp ult i64 %112, 16
  call void @llvm.assume(i1 %113)
  br label %117

114:                                              ; preds = %108
  %115 = load i64, ptr %40, align 8, !tbaa !36
  %116 = add i64 %115, 1
  call void @_ZdlPvm(ptr noundef %109, i64 noundef %116) #18
  br label %117

117:                                              ; preds = %111, %114
  call void @llvm.lifetime.end.p0(ptr nonnull %9) #16
  %118 = load ptr, ptr @_ZSt3cin, align 8, !tbaa !6
  %119 = getelementptr i8, ptr %118, i64 -24
  %120 = load i64, ptr %119, align 8
  %121 = getelementptr inbounds i8, ptr @_ZSt3cin, i64 %120
  %122 = getelementptr inbounds nuw i8, ptr %121, i64 240
  %123 = load ptr, ptr %122, align 8, !tbaa !29
  %124 = icmp eq ptr %123, null
  br i1 %124, label %44, label %46, !llvm.loop !52

125:                                              ; preds = %54, %55, %60
  %126 = landingpad { ptr, i32 }
          cleanup
  br label %157

127:                                              ; preds = %0, %20, %44
  %128 = landingpad { ptr, i32 }
          cleanup
  br label %157

129:                                              ; preds = %75
  %130 = landingpad { ptr, i32 }
          cleanup
  br label %141

131:                                              ; preds = %107
  %132 = landingpad { ptr, i32 }
          cleanup
  %133 = load ptr, ptr %9, align 8, !tbaa !42
  %134 = icmp eq ptr %133, %40
  br i1 %134, label %135, label %138

135:                                              ; preds = %131
  %136 = load i64, ptr %41, align 8, !tbaa !44
  %137 = icmp ult i64 %136, 16
  call void @llvm.assume(i1 %137)
  br label %141

138:                                              ; preds = %131
  %139 = load i64, ptr %40, align 8, !tbaa !36
  %140 = add i64 %139, 1
  call void @_ZdlPvm(ptr noundef %133, i64 noundef %140) #18
  br label %141

141:                                              ; preds = %138, %135, %129
  %142 = phi { ptr, i32 } [ %130, %129 ], [ %132, %135 ], [ %132, %138 ]
  call void @llvm.lifetime.end.p0(ptr nonnull %9) #16
  br label %157

143:                                              ; preds = %63
  %144 = getelementptr inbounds nuw i8, ptr %6, i64 32
  %145 = getelementptr inbounds nuw i8, ptr %6, i64 48
  %146 = getelementptr inbounds nuw i8, ptr %6, i64 64
  call void @llvm.lifetime.start.p0(ptr nonnull %4), !noalias !54
  call void @llvm.lifetime.start.p0(ptr nonnull %1), !noalias !57
  call void @llvm.lifetime.start.p0(ptr nonnull %2), !noalias !57
  call void @llvm.lifetime.start.p0(ptr nonnull %3), !noalias !57
  %147 = load <2 x ptr>, ptr %42, align 8, !tbaa !60, !noalias !61
  store <2 x ptr> %147, ptr %1, align 16, !tbaa !60, !noalias !64
  %148 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %149 = load <2 x ptr>, ptr %144, align 8, !tbaa !67, !noalias !61
  store <2 x ptr> %149, ptr %148, align 16, !tbaa !67, !noalias !64
  %150 = load <2 x ptr>, ptr %145, align 8, !tbaa !60, !noalias !68
  store <2 x ptr> %150, ptr %2, align 16, !tbaa !60, !noalias !64
  %151 = getelementptr inbounds nuw i8, ptr %2, i64 16
  %152 = load <2 x ptr>, ptr %146, align 8, !tbaa !67, !noalias !68
  store <2 x ptr> %152, ptr %151, align 16, !tbaa !67, !noalias !64
  store ptr @_ZSt4cout, ptr %3, align 8, !tbaa !71, !noalias !64
  %153 = getelementptr inbounds nuw i8, ptr %3, i64 8
  store ptr @.str, ptr %153, align 8, !tbaa !73, !noalias !64
  invoke void @_ZSt15__copy_move_ditILb0ENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERS5_PS5_St16ostream_iteratorIS5_cS3_EET3_St15_Deque_iteratorIT0_T1_T2_ESF_SA_(ptr dead_on_unwind nonnull writable sret(%"class.std::ostream_iterator") align 8 %4, ptr dead_on_return noundef nonnull %1, ptr dead_on_return noundef nonnull %2, ptr dead_on_return noundef nonnull %3)
          to label %154 unwind label %155

154:                                              ; preds = %143
  call void @llvm.lifetime.end.p0(ptr nonnull %1), !noalias !57
  call void @llvm.lifetime.end.p0(ptr nonnull %2), !noalias !57
  call void @llvm.lifetime.end.p0(ptr nonnull %3), !noalias !57
  call void @llvm.lifetime.end.p0(ptr nonnull %4), !noalias !54
  call void @llvm.lifetime.end.p0(ptr nonnull %8) #16
  call void @llvm.lifetime.end.p0(ptr nonnull %7) #16
  call void @_ZNSt5dequeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESaIS5_EED2Ev(ptr noundef nonnull align 8 dereferenceable(80) %6) #16
  call void @llvm.lifetime.end.p0(ptr nonnull %6) #16
  ret i32 0

155:                                              ; preds = %143
  %156 = landingpad { ptr, i32 }
          cleanup
  br label %157

157:                                              ; preds = %125, %127, %155, %141
  %158 = phi { ptr, i32 } [ %142, %141 ], [ %156, %155 ], [ %126, %125 ], [ %128, %127 ]
  call void @llvm.lifetime.end.p0(ptr nonnull %8) #16
  call void @llvm.lifetime.end.p0(ptr nonnull %7) #16
  call void @_ZNSt5dequeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESaIS5_EED2Ev(ptr noundef nonnull align 8 dereferenceable(80) %6) #16
  call void @llvm.lifetime.end.p0(ptr nonnull %6) #16
  resume { ptr, i32 } %158
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

declare i32 @__gxx_personality_v0(...)

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local void @_ZNSt5dequeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESaIS5_EED2Ev(ptr noundef nonnull align 8 dereferenceable(80) %0) unnamed_addr #2 comdat personality ptr @__gxx_personality_v0 {
  %2 = alloca %"struct.std::_Deque_iterator", align 16
  %3 = alloca %"struct.std::_Deque_iterator", align 16
  %4 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %5 = getelementptr inbounds nuw i8, ptr %0, i64 32
  %6 = getelementptr inbounds nuw i8, ptr %0, i64 40
  %7 = getelementptr inbounds nuw i8, ptr %0, i64 48
  %8 = getelementptr inbounds nuw i8, ptr %0, i64 64
  %9 = getelementptr inbounds nuw i8, ptr %0, i64 72
  call void @llvm.lifetime.start.p0(ptr nonnull %2)
  call void @llvm.lifetime.start.p0(ptr nonnull %3)
  %10 = load <2 x ptr>, ptr %4, align 8, !tbaa !60, !noalias !74
  store <2 x ptr> %10, ptr %2, align 16, !tbaa !60
  %11 = getelementptr inbounds nuw i8, ptr %2, i64 16
  %12 = load <2 x ptr>, ptr %5, align 8, !tbaa !67, !noalias !74
  store <2 x ptr> %12, ptr %11, align 16, !tbaa !67
  %13 = load <2 x ptr>, ptr %7, align 8, !tbaa !60, !noalias !77
  store <2 x ptr> %13, ptr %3, align 16, !tbaa !60
  %14 = getelementptr inbounds nuw i8, ptr %3, i64 16
  %15 = load <2 x ptr>, ptr %8, align 8, !tbaa !67, !noalias !77
  store <2 x ptr> %15, ptr %14, align 16, !tbaa !67
  invoke void @_ZNSt5dequeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESaIS5_EE19_M_destroy_data_auxESt15_Deque_iteratorIS5_RS5_PS5_ESB_(ptr noundef nonnull align 8 dereferenceable(80) %0, ptr dead_on_return noundef nonnull %2, ptr dead_on_return noundef nonnull %3)
          to label %16 unwind label %37

16:                                               ; preds = %1
  call void @llvm.lifetime.end.p0(ptr nonnull %2)
  call void @llvm.lifetime.end.p0(ptr nonnull %3)
  %17 = load ptr, ptr %0, align 8, !tbaa !80
  %18 = icmp eq ptr %17, null
  br i1 %18, label %36, label %19

19:                                               ; preds = %16
  %20 = load ptr, ptr %6, align 8, !tbaa !81
  %21 = load ptr, ptr %9, align 8, !tbaa !82
  %22 = getelementptr inbounds nuw i8, ptr %21, i64 8
  %23 = icmp ult ptr %20, %22
  br i1 %23, label %24, label %31

24:                                               ; preds = %19, %24
  %25 = phi ptr [ %27, %24 ], [ %20, %19 ]
  %26 = load ptr, ptr %25, align 8, !tbaa !60
  call void @_ZdlPvm(ptr noundef %26, i64 noundef 512) #18
  %27 = getelementptr inbounds nuw i8, ptr %25, i64 8
  %28 = icmp ult ptr %25, %21
  br i1 %28, label %24, label %29, !llvm.loop !83

29:                                               ; preds = %24
  %30 = load ptr, ptr %0, align 8, !tbaa !80
  br label %31

31:                                               ; preds = %29, %19
  %32 = phi ptr [ %30, %29 ], [ %17, %19 ]
  %33 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %34 = load i64, ptr %33, align 8, !tbaa !84
  %35 = shl i64 %34, 3
  call void @_ZdlPvm(ptr noundef %32, i64 noundef %35) #18
  br label %36

36:                                               ; preds = %16, %31
  ret void

37:                                               ; preds = %1
  %38 = landingpad { ptr, i32 }
          catch ptr null
  %39 = extractvalue { ptr, i32 } %38, 0
  call void @__clang_call_terminate(ptr %39) #19
  unreachable
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZNSt11_Deque_baseINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESaIS5_EE17_M_initialize_mapEm(ptr noundef nonnull align 8 dereferenceable(80) %0, i64 noundef %1) local_unnamed_addr #3 comdat personality ptr @__gxx_personality_v0 {
  %3 = lshr i64 %1, 4
  %4 = tail call i64 @llvm.umax.i64(i64 %3, i64 5)
  %5 = add nuw nsw i64 %4, 3
  %6 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i64 %5, ptr %6, align 8, !tbaa !84
  %7 = icmp ugt i64 %1, -49
  br i1 %7, label %8, label %9, !prof !85

8:                                                ; preds = %2
  tail call void @_ZSt17__throw_bad_allocv() #20
  unreachable

9:                                                ; preds = %2
  %10 = add nuw nsw i64 %3, 1
  %11 = shl nuw nsw i64 %5, 3
  %12 = tail call noalias noundef nonnull ptr @_Znwm(i64 noundef %11) #21
  store ptr %12, ptr %0, align 8, !tbaa !80
  %13 = sub nsw i64 %5, %10
  %14 = lshr i64 %13, 1
  %15 = getelementptr inbounds nuw ptr, ptr %12, i64 %14
  %16 = shl nuw nsw i64 %10, 3
  %17 = getelementptr inbounds nuw i8, ptr %15, i64 %16
  br label %18

18:                                               ; preds = %9, %21
  %19 = phi ptr [ %22, %21 ], [ %15, %9 ]
  %20 = invoke noalias noundef nonnull dereferenceable(512) ptr @_Znwm(i64 noundef 512) #21
          to label %21 unwind label %24

21:                                               ; preds = %18
  store ptr %20, ptr %19, align 8, !tbaa !60
  %22 = getelementptr inbounds nuw i8, ptr %19, i64 8
  %23 = icmp ult ptr %22, %17
  br i1 %23, label %18, label %50, !llvm.loop !86

24:                                               ; preds = %18
  %25 = landingpad { ptr, i32 }
          catch ptr null
  %26 = extractvalue { ptr, i32 } %25, 0
  %27 = tail call ptr @__cxa_begin_catch(ptr %26) #16
  %28 = icmp ult ptr %15, %19
  br i1 %28, label %29, label %34

29:                                               ; preds = %24, %29
  %30 = phi ptr [ %32, %29 ], [ %15, %24 ]
  %31 = load ptr, ptr %30, align 8, !tbaa !60
  tail call void @_ZdlPvm(ptr noundef %31, i64 noundef 512) #18
  %32 = getelementptr inbounds nuw i8, ptr %30, i64 8
  %33 = icmp ult ptr %32, %19
  br i1 %33, label %29, label %34, !llvm.loop !83

34:                                               ; preds = %29, %24
  invoke void @__cxa_rethrow() #20
          to label %40 unwind label %35

35:                                               ; preds = %34
  %36 = landingpad { ptr, i32 }
          catch ptr null
  invoke void @__cxa_end_catch()
          to label %41 unwind label %37

37:                                               ; preds = %35
  %38 = landingpad { ptr, i32 }
          catch ptr null
  %39 = extractvalue { ptr, i32 } %38, 0
  tail call void @__clang_call_terminate(ptr %39) #19
  unreachable

40:                                               ; preds = %34
  unreachable

41:                                               ; preds = %35
  %42 = extractvalue { ptr, i32 } %36, 0
  %43 = tail call ptr @__cxa_begin_catch(ptr %42) #16
  %44 = load ptr, ptr %0, align 8, !tbaa !80
  %45 = load i64, ptr %6, align 8, !tbaa !84
  %46 = shl i64 %45, 3
  tail call void @_ZdlPvm(ptr noundef %44, i64 noundef %46) #18
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %0, i8 0, i64 16, i1 false)
  invoke void @__cxa_rethrow() #20
          to label %69 unwind label %47

47:                                               ; preds = %41
  %48 = landingpad { ptr, i32 }
          cleanup
  invoke void @__cxa_end_catch()
          to label %49 unwind label %66

49:                                               ; preds = %47
  resume { ptr, i32 } %48

50:                                               ; preds = %21
  %51 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %52 = getelementptr inbounds nuw i8, ptr %0, i64 40
  store ptr %15, ptr %52, align 8, !tbaa !87
  %53 = load ptr, ptr %15, align 8, !tbaa !60
  %54 = getelementptr inbounds nuw i8, ptr %0, i64 24
  store ptr %53, ptr %54, align 8, !tbaa !88
  %55 = getelementptr inbounds nuw i8, ptr %53, i64 512
  %56 = getelementptr inbounds nuw i8, ptr %0, i64 32
  store ptr %55, ptr %56, align 8, !tbaa !89
  %57 = getelementptr inbounds nuw i8, ptr %0, i64 48
  %58 = getelementptr inbounds i8, ptr %17, i64 -8
  %59 = getelementptr inbounds nuw i8, ptr %0, i64 72
  store ptr %58, ptr %59, align 8, !tbaa !87
  %60 = load ptr, ptr %58, align 8, !tbaa !60
  %61 = getelementptr inbounds nuw i8, ptr %0, i64 56
  store ptr %60, ptr %61, align 8, !tbaa !88
  %62 = getelementptr inbounds nuw i8, ptr %60, i64 512
  %63 = getelementptr inbounds nuw i8, ptr %0, i64 64
  store ptr %62, ptr %63, align 8, !tbaa !89
  store ptr %53, ptr %51, align 8, !tbaa !45
  %64 = and i64 %1, 15
  %65 = getelementptr inbounds nuw %"class.std::__cxx11::basic_string", ptr %60, i64 %64
  store ptr %65, ptr %57, align 8, !tbaa !90
  ret void

66:                                               ; preds = %47
  %67 = landingpad { ptr, i32 }
          catch ptr null
  %68 = extractvalue { ptr, i32 } %67, 0
  tail call void @__clang_call_terminate(ptr %68) #19
  unreachable

69:                                               ; preds = %41
  unreachable
}

declare ptr @__cxa_begin_catch(ptr) local_unnamed_addr

declare void @__cxa_rethrow() local_unnamed_addr

declare void @__cxa_end_catch() local_unnamed_addr

; Function Attrs: noinline noreturn nounwind uwtable
define linkonce_odr hidden void @__clang_call_terminate(ptr noundef %0) local_unnamed_addr #4 comdat {
  %2 = tail call ptr @__cxa_begin_catch(ptr %0) #16
  tail call void @_ZSt9terminatev() #19
  unreachable
}

; Function Attrs: cold nofree noreturn
declare void @_ZSt9terminatev() local_unnamed_addr #5

; Function Attrs: noreturn
declare void @_ZSt28__throw_bad_array_new_lengthv() local_unnamed_addr #6

; Function Attrs: noreturn
declare void @_ZSt17__throw_bad_allocv() local_unnamed_addr #6

; Function Attrs: nobuiltin allocsize(0)
declare noundef nonnull ptr @_Znwm(i64 noundef) local_unnamed_addr #7

; Function Attrs: nobuiltin nounwind
declare void @_ZdlPvm(ptr noundef, i64 noundef) local_unnamed_addr #8

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZNSt5dequeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESaIS5_EE19_M_destroy_data_auxESt15_Deque_iteratorIS5_RS5_PS5_ESB_(ptr noundef nonnull align 8 dereferenceable(80) %0, ptr dead_on_return noundef %1, ptr dead_on_return noundef %2) local_unnamed_addr #3 comdat personality ptr @__gxx_personality_v0 {
  %4 = getelementptr inbounds nuw i8, ptr %1, i64 24
  %5 = load ptr, ptr %4, align 8, !tbaa !87
  %6 = getelementptr inbounds nuw i8, ptr %2, i64 24
  %7 = getelementptr inbounds nuw i8, ptr %5, i64 8
  %8 = load ptr, ptr %6, align 8, !tbaa !87
  %9 = icmp ult ptr %7, %8
  br i1 %9, label %17, label %12

10:                                               ; preds = %210
  %11 = load ptr, ptr %4, align 8, !tbaa !87
  br label %12

12:                                               ; preds = %10, %3
  %13 = phi ptr [ %5, %3 ], [ %11, %10 ]
  %14 = phi ptr [ %8, %3 ], [ %212, %10 ]
  %15 = icmp eq ptr %13, %14
  %16 = load ptr, ptr %1, align 8, !tbaa !91
  br i1 %15, label %253, label %214

17:                                               ; preds = %3, %210
  %18 = phi ptr [ %211, %210 ], [ %7, %3 ]
  %19 = load ptr, ptr %18, align 8, !tbaa !60
  %20 = load ptr, ptr %19, align 8, !tbaa !42
  %21 = getelementptr inbounds nuw i8, ptr %19, i64 16
  %22 = icmp eq ptr %20, %21
  br i1 %22, label %23, label %27

23:                                               ; preds = %17
  %24 = getelementptr inbounds nuw i8, ptr %19, i64 8
  %25 = load i64, ptr %24, align 8, !tbaa !44
  %26 = icmp ult i64 %25, 16
  tail call void @llvm.assume(i1 %26)
  br label %30

27:                                               ; preds = %17
  %28 = load i64, ptr %21, align 8, !tbaa !36
  %29 = add i64 %28, 1
  tail call void @_ZdlPvm(ptr noundef %20, i64 noundef %29) #18
  br label %30

30:                                               ; preds = %27, %23
  %31 = getelementptr inbounds nuw i8, ptr %19, i64 32
  %32 = load ptr, ptr %31, align 8, !tbaa !42
  %33 = getelementptr inbounds nuw i8, ptr %19, i64 48
  %34 = icmp eq ptr %32, %33
  br i1 %34, label %38, label %35

35:                                               ; preds = %30
  %36 = load i64, ptr %33, align 8, !tbaa !36
  %37 = add i64 %36, 1
  tail call void @_ZdlPvm(ptr noundef %32, i64 noundef %37) #18
  br label %42

38:                                               ; preds = %30
  %39 = getelementptr inbounds nuw i8, ptr %19, i64 40
  %40 = load i64, ptr %39, align 8, !tbaa !44
  %41 = icmp ult i64 %40, 16
  tail call void @llvm.assume(i1 %41)
  br label %42

42:                                               ; preds = %38, %35
  %43 = getelementptr inbounds nuw i8, ptr %19, i64 64
  %44 = load ptr, ptr %43, align 8, !tbaa !42
  %45 = getelementptr inbounds nuw i8, ptr %19, i64 80
  %46 = icmp eq ptr %44, %45
  br i1 %46, label %50, label %47

47:                                               ; preds = %42
  %48 = load i64, ptr %45, align 8, !tbaa !36
  %49 = add i64 %48, 1
  tail call void @_ZdlPvm(ptr noundef %44, i64 noundef %49) #18
  br label %54

50:                                               ; preds = %42
  %51 = getelementptr inbounds nuw i8, ptr %19, i64 72
  %52 = load i64, ptr %51, align 8, !tbaa !44
  %53 = icmp ult i64 %52, 16
  tail call void @llvm.assume(i1 %53)
  br label %54

54:                                               ; preds = %50, %47
  %55 = getelementptr inbounds nuw i8, ptr %19, i64 96
  %56 = load ptr, ptr %55, align 8, !tbaa !42
  %57 = getelementptr inbounds nuw i8, ptr %19, i64 112
  %58 = icmp eq ptr %56, %57
  br i1 %58, label %62, label %59

59:                                               ; preds = %54
  %60 = load i64, ptr %57, align 8, !tbaa !36
  %61 = add i64 %60, 1
  tail call void @_ZdlPvm(ptr noundef %56, i64 noundef %61) #18
  br label %66

62:                                               ; preds = %54
  %63 = getelementptr inbounds nuw i8, ptr %19, i64 104
  %64 = load i64, ptr %63, align 8, !tbaa !44
  %65 = icmp ult i64 %64, 16
  tail call void @llvm.assume(i1 %65)
  br label %66

66:                                               ; preds = %62, %59
  %67 = getelementptr inbounds nuw i8, ptr %19, i64 128
  %68 = load ptr, ptr %67, align 8, !tbaa !42
  %69 = getelementptr inbounds nuw i8, ptr %19, i64 144
  %70 = icmp eq ptr %68, %69
  br i1 %70, label %74, label %71

71:                                               ; preds = %66
  %72 = load i64, ptr %69, align 8, !tbaa !36
  %73 = add i64 %72, 1
  tail call void @_ZdlPvm(ptr noundef %68, i64 noundef %73) #18
  br label %78

74:                                               ; preds = %66
  %75 = getelementptr inbounds nuw i8, ptr %19, i64 136
  %76 = load i64, ptr %75, align 8, !tbaa !44
  %77 = icmp ult i64 %76, 16
  tail call void @llvm.assume(i1 %77)
  br label %78

78:                                               ; preds = %74, %71
  %79 = getelementptr inbounds nuw i8, ptr %19, i64 160
  %80 = load ptr, ptr %79, align 8, !tbaa !42
  %81 = getelementptr inbounds nuw i8, ptr %19, i64 176
  %82 = icmp eq ptr %80, %81
  br i1 %82, label %86, label %83

83:                                               ; preds = %78
  %84 = load i64, ptr %81, align 8, !tbaa !36
  %85 = add i64 %84, 1
  tail call void @_ZdlPvm(ptr noundef %80, i64 noundef %85) #18
  br label %90

86:                                               ; preds = %78
  %87 = getelementptr inbounds nuw i8, ptr %19, i64 168
  %88 = load i64, ptr %87, align 8, !tbaa !44
  %89 = icmp ult i64 %88, 16
  tail call void @llvm.assume(i1 %89)
  br label %90

90:                                               ; preds = %86, %83
  %91 = getelementptr inbounds nuw i8, ptr %19, i64 192
  %92 = load ptr, ptr %91, align 8, !tbaa !42
  %93 = getelementptr inbounds nuw i8, ptr %19, i64 208
  %94 = icmp eq ptr %92, %93
  br i1 %94, label %98, label %95

95:                                               ; preds = %90
  %96 = load i64, ptr %93, align 8, !tbaa !36
  %97 = add i64 %96, 1
  tail call void @_ZdlPvm(ptr noundef %92, i64 noundef %97) #18
  br label %102

98:                                               ; preds = %90
  %99 = getelementptr inbounds nuw i8, ptr %19, i64 200
  %100 = load i64, ptr %99, align 8, !tbaa !44
  %101 = icmp ult i64 %100, 16
  tail call void @llvm.assume(i1 %101)
  br label %102

102:                                              ; preds = %98, %95
  %103 = getelementptr inbounds nuw i8, ptr %19, i64 224
  %104 = load ptr, ptr %103, align 8, !tbaa !42
  %105 = getelementptr inbounds nuw i8, ptr %19, i64 240
  %106 = icmp eq ptr %104, %105
  br i1 %106, label %110, label %107

107:                                              ; preds = %102
  %108 = load i64, ptr %105, align 8, !tbaa !36
  %109 = add i64 %108, 1
  tail call void @_ZdlPvm(ptr noundef %104, i64 noundef %109) #18
  br label %114

110:                                              ; preds = %102
  %111 = getelementptr inbounds nuw i8, ptr %19, i64 232
  %112 = load i64, ptr %111, align 8, !tbaa !44
  %113 = icmp ult i64 %112, 16
  tail call void @llvm.assume(i1 %113)
  br label %114

114:                                              ; preds = %110, %107
  %115 = getelementptr inbounds nuw i8, ptr %19, i64 256
  %116 = load ptr, ptr %115, align 8, !tbaa !42
  %117 = getelementptr inbounds nuw i8, ptr %19, i64 272
  %118 = icmp eq ptr %116, %117
  br i1 %118, label %122, label %119

119:                                              ; preds = %114
  %120 = load i64, ptr %117, align 8, !tbaa !36
  %121 = add i64 %120, 1
  tail call void @_ZdlPvm(ptr noundef %116, i64 noundef %121) #18
  br label %126

122:                                              ; preds = %114
  %123 = getelementptr inbounds nuw i8, ptr %19, i64 264
  %124 = load i64, ptr %123, align 8, !tbaa !44
  %125 = icmp ult i64 %124, 16
  tail call void @llvm.assume(i1 %125)
  br label %126

126:                                              ; preds = %122, %119
  %127 = getelementptr inbounds nuw i8, ptr %19, i64 288
  %128 = load ptr, ptr %127, align 8, !tbaa !42
  %129 = getelementptr inbounds nuw i8, ptr %19, i64 304
  %130 = icmp eq ptr %128, %129
  br i1 %130, label %134, label %131

131:                                              ; preds = %126
  %132 = load i64, ptr %129, align 8, !tbaa !36
  %133 = add i64 %132, 1
  tail call void @_ZdlPvm(ptr noundef %128, i64 noundef %133) #18
  br label %138

134:                                              ; preds = %126
  %135 = getelementptr inbounds nuw i8, ptr %19, i64 296
  %136 = load i64, ptr %135, align 8, !tbaa !44
  %137 = icmp ult i64 %136, 16
  tail call void @llvm.assume(i1 %137)
  br label %138

138:                                              ; preds = %134, %131
  %139 = getelementptr inbounds nuw i8, ptr %19, i64 320
  %140 = load ptr, ptr %139, align 8, !tbaa !42
  %141 = getelementptr inbounds nuw i8, ptr %19, i64 336
  %142 = icmp eq ptr %140, %141
  br i1 %142, label %146, label %143

143:                                              ; preds = %138
  %144 = load i64, ptr %141, align 8, !tbaa !36
  %145 = add i64 %144, 1
  tail call void @_ZdlPvm(ptr noundef %140, i64 noundef %145) #18
  br label %150

146:                                              ; preds = %138
  %147 = getelementptr inbounds nuw i8, ptr %19, i64 328
  %148 = load i64, ptr %147, align 8, !tbaa !44
  %149 = icmp ult i64 %148, 16
  tail call void @llvm.assume(i1 %149)
  br label %150

150:                                              ; preds = %146, %143
  %151 = getelementptr inbounds nuw i8, ptr %19, i64 352
  %152 = load ptr, ptr %151, align 8, !tbaa !42
  %153 = getelementptr inbounds nuw i8, ptr %19, i64 368
  %154 = icmp eq ptr %152, %153
  br i1 %154, label %158, label %155

155:                                              ; preds = %150
  %156 = load i64, ptr %153, align 8, !tbaa !36
  %157 = add i64 %156, 1
  tail call void @_ZdlPvm(ptr noundef %152, i64 noundef %157) #18
  br label %162

158:                                              ; preds = %150
  %159 = getelementptr inbounds nuw i8, ptr %19, i64 360
  %160 = load i64, ptr %159, align 8, !tbaa !44
  %161 = icmp ult i64 %160, 16
  tail call void @llvm.assume(i1 %161)
  br label %162

162:                                              ; preds = %158, %155
  %163 = getelementptr inbounds nuw i8, ptr %19, i64 384
  %164 = load ptr, ptr %163, align 8, !tbaa !42
  %165 = getelementptr inbounds nuw i8, ptr %19, i64 400
  %166 = icmp eq ptr %164, %165
  br i1 %166, label %170, label %167

167:                                              ; preds = %162
  %168 = load i64, ptr %165, align 8, !tbaa !36
  %169 = add i64 %168, 1
  tail call void @_ZdlPvm(ptr noundef %164, i64 noundef %169) #18
  br label %174

170:                                              ; preds = %162
  %171 = getelementptr inbounds nuw i8, ptr %19, i64 392
  %172 = load i64, ptr %171, align 8, !tbaa !44
  %173 = icmp ult i64 %172, 16
  tail call void @llvm.assume(i1 %173)
  br label %174

174:                                              ; preds = %170, %167
  %175 = getelementptr inbounds nuw i8, ptr %19, i64 416
  %176 = load ptr, ptr %175, align 8, !tbaa !42
  %177 = getelementptr inbounds nuw i8, ptr %19, i64 432
  %178 = icmp eq ptr %176, %177
  br i1 %178, label %182, label %179

179:                                              ; preds = %174
  %180 = load i64, ptr %177, align 8, !tbaa !36
  %181 = add i64 %180, 1
  tail call void @_ZdlPvm(ptr noundef %176, i64 noundef %181) #18
  br label %186

182:                                              ; preds = %174
  %183 = getelementptr inbounds nuw i8, ptr %19, i64 424
  %184 = load i64, ptr %183, align 8, !tbaa !44
  %185 = icmp ult i64 %184, 16
  tail call void @llvm.assume(i1 %185)
  br label %186

186:                                              ; preds = %182, %179
  %187 = getelementptr inbounds nuw i8, ptr %19, i64 448
  %188 = load ptr, ptr %187, align 8, !tbaa !42
  %189 = getelementptr inbounds nuw i8, ptr %19, i64 464
  %190 = icmp eq ptr %188, %189
  br i1 %190, label %194, label %191

191:                                              ; preds = %186
  %192 = load i64, ptr %189, align 8, !tbaa !36
  %193 = add i64 %192, 1
  tail call void @_ZdlPvm(ptr noundef %188, i64 noundef %193) #18
  br label %198

194:                                              ; preds = %186
  %195 = getelementptr inbounds nuw i8, ptr %19, i64 456
  %196 = load i64, ptr %195, align 8, !tbaa !44
  %197 = icmp ult i64 %196, 16
  tail call void @llvm.assume(i1 %197)
  br label %198

198:                                              ; preds = %194, %191
  %199 = getelementptr inbounds nuw i8, ptr %19, i64 480
  %200 = load ptr, ptr %199, align 8, !tbaa !42
  %201 = getelementptr inbounds nuw i8, ptr %19, i64 496
  %202 = icmp eq ptr %200, %201
  br i1 %202, label %206, label %203

203:                                              ; preds = %198
  %204 = load i64, ptr %201, align 8, !tbaa !36
  %205 = add i64 %204, 1
  tail call void @_ZdlPvm(ptr noundef %200, i64 noundef %205) #18
  br label %210

206:                                              ; preds = %198
  %207 = getelementptr inbounds nuw i8, ptr %19, i64 488
  %208 = load i64, ptr %207, align 8, !tbaa !44
  %209 = icmp ult i64 %208, 16
  tail call void @llvm.assume(i1 %209)
  br label %210

210:                                              ; preds = %206, %203
  %211 = getelementptr inbounds nuw i8, ptr %18, i64 8
  %212 = load ptr, ptr %6, align 8, !tbaa !87
  %213 = icmp ult ptr %211, %212
  br i1 %213, label %17, label %10, !llvm.loop !92

214:                                              ; preds = %12
  %215 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %216 = load ptr, ptr %215, align 8, !tbaa !89
  %217 = icmp eq ptr %16, %216
  br i1 %217, label %233, label %218

218:                                              ; preds = %214, %230
  %219 = phi ptr [ %231, %230 ], [ %16, %214 ]
  %220 = load ptr, ptr %219, align 8, !tbaa !42
  %221 = getelementptr inbounds nuw i8, ptr %219, i64 16
  %222 = icmp eq ptr %220, %221
  br i1 %222, label %223, label %227

223:                                              ; preds = %218
  %224 = getelementptr inbounds nuw i8, ptr %219, i64 8
  %225 = load i64, ptr %224, align 8, !tbaa !44
  %226 = icmp ult i64 %225, 16
  tail call void @llvm.assume(i1 %226)
  br label %230

227:                                              ; preds = %218
  %228 = load i64, ptr %221, align 8, !tbaa !36
  %229 = add i64 %228, 1
  tail call void @_ZdlPvm(ptr noundef %220, i64 noundef %229) #18
  br label %230

230:                                              ; preds = %227, %223
  %231 = getelementptr inbounds nuw i8, ptr %219, i64 32
  %232 = icmp eq ptr %231, %216
  br i1 %232, label %233, label %218, !llvm.loop !93

233:                                              ; preds = %230, %214
  %234 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %235 = load ptr, ptr %234, align 8, !tbaa !88
  %236 = load ptr, ptr %2, align 8, !tbaa !91
  %237 = icmp eq ptr %235, %236
  br i1 %237, label %271, label %238

238:                                              ; preds = %233, %250
  %239 = phi ptr [ %251, %250 ], [ %235, %233 ]
  %240 = load ptr, ptr %239, align 8, !tbaa !42
  %241 = getelementptr inbounds nuw i8, ptr %239, i64 16
  %242 = icmp eq ptr %240, %241
  br i1 %242, label %243, label %247

243:                                              ; preds = %238
  %244 = getelementptr inbounds nuw i8, ptr %239, i64 8
  %245 = load i64, ptr %244, align 8, !tbaa !44
  %246 = icmp ult i64 %245, 16
  tail call void @llvm.assume(i1 %246)
  br label %250

247:                                              ; preds = %238
  %248 = load i64, ptr %241, align 8, !tbaa !36
  %249 = add i64 %248, 1
  tail call void @_ZdlPvm(ptr noundef %240, i64 noundef %249) #18
  br label %250

250:                                              ; preds = %247, %243
  %251 = getelementptr inbounds nuw i8, ptr %239, i64 32
  %252 = icmp eq ptr %251, %236
  br i1 %252, label %271, label %238, !llvm.loop !93

253:                                              ; preds = %12
  %254 = load ptr, ptr %2, align 8, !tbaa !91
  %255 = icmp eq ptr %16, %254
  br i1 %255, label %271, label %256

256:                                              ; preds = %253, %268
  %257 = phi ptr [ %269, %268 ], [ %16, %253 ]
  %258 = load ptr, ptr %257, align 8, !tbaa !42
  %259 = getelementptr inbounds nuw i8, ptr %257, i64 16
  %260 = icmp eq ptr %258, %259
  br i1 %260, label %261, label %265

261:                                              ; preds = %256
  %262 = getelementptr inbounds nuw i8, ptr %257, i64 8
  %263 = load i64, ptr %262, align 8, !tbaa !44
  %264 = icmp ult i64 %263, 16
  tail call void @llvm.assume(i1 %264)
  br label %268

265:                                              ; preds = %256
  %266 = load i64, ptr %259, align 8, !tbaa !36
  %267 = add i64 %266, 1
  tail call void @_ZdlPvm(ptr noundef %258, i64 noundef %267) #18
  br label %268

268:                                              ; preds = %265, %261
  %269 = getelementptr inbounds nuw i8, ptr %257, i64 32
  %270 = icmp eq ptr %269, %254
  br i1 %270, label %271, label %256, !llvm.loop !93

271:                                              ; preds = %250, %268, %253, %233
  ret void
}

declare noundef nonnull align 8 dereferenceable(16) ptr @_ZNSi7getlineEPclc(ptr noundef nonnull align 8 dereferenceable(16), ptr noundef, i64 noundef, i8 noundef) local_unnamed_addr #9

; Function Attrs: cold noreturn
declare void @_ZSt16__throw_bad_castv() local_unnamed_addr #10

declare void @_ZNKSt5ctypeIcE13_M_widen_initEv(ptr noundef nonnull align 8 dereferenceable(570)) local_unnamed_addr #9

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZNSt5dequeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESaIS5_EE17_M_push_front_auxIJS5_EEEvDpOT_(ptr noundef nonnull align 8 dereferenceable(80) %0, ptr noundef nonnull align 8 dereferenceable(32) %1) local_unnamed_addr #3 comdat personality ptr @__gxx_personality_v0 {
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 48
  %4 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %5 = getelementptr inbounds nuw i8, ptr %0, i64 72
  %6 = load ptr, ptr %5, align 8, !tbaa !87
  %7 = getelementptr inbounds nuw i8, ptr %0, i64 40
  %8 = load ptr, ptr %7, align 8, !tbaa !87
  %9 = ptrtoint ptr %6 to i64
  %10 = ptrtoint ptr %8 to i64
  %11 = sub i64 %9, %10
  %12 = ashr exact i64 %11, 3
  %13 = icmp ne ptr %6, null
  %14 = sext i1 %13 to i64
  %15 = add nsw i64 %12, %14
  %16 = shl nsw i64 %15, 4
  %17 = load ptr, ptr %3, align 8, !tbaa !91
  %18 = getelementptr inbounds nuw i8, ptr %0, i64 56
  %19 = load ptr, ptr %18, align 8, !tbaa !88
  %20 = ptrtoint ptr %17 to i64
  %21 = ptrtoint ptr %19 to i64
  %22 = sub i64 %20, %21
  %23 = ashr exact i64 %22, 5
  %24 = add nsw i64 %16, %23
  %25 = getelementptr inbounds nuw i8, ptr %0, i64 32
  %26 = load ptr, ptr %25, align 8, !tbaa !89
  %27 = load ptr, ptr %4, align 8, !tbaa !91
  %28 = ptrtoint ptr %26 to i64
  %29 = ptrtoint ptr %27 to i64
  %30 = sub i64 %28, %29
  %31 = ashr exact i64 %30, 5
  %32 = add nsw i64 %24, %31
  %33 = icmp eq i64 %32, 288230376151711743
  br i1 %33, label %34, label %35

34:                                               ; preds = %2
  tail call void @_ZSt20__throw_length_errorPKc(ptr noundef nonnull @.str.1) #17
  unreachable

35:                                               ; preds = %2
  %36 = load ptr, ptr %0, align 8, !tbaa !80
  %37 = icmp eq ptr %8, %36
  br i1 %37, label %38, label %40

38:                                               ; preds = %35
  tail call void @_ZNSt5dequeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESaIS5_EE17_M_reallocate_mapEmb(ptr noundef nonnull align 8 dereferenceable(80) %0, i64 noundef 1, i1 noundef true)
  %39 = load ptr, ptr %7, align 8, !tbaa !81
  br label %40

40:                                               ; preds = %35, %38
  %41 = phi ptr [ %8, %35 ], [ %39, %38 ]
  %42 = tail call noalias noundef nonnull dereferenceable(512) ptr @_Znwm(i64 noundef 512) #21
  %43 = getelementptr inbounds i8, ptr %41, i64 -8
  store ptr %42, ptr %43, align 8, !tbaa !60
  store ptr %43, ptr %7, align 8, !tbaa !87
  %44 = getelementptr inbounds nuw i8, ptr %0, i64 24
  store ptr %42, ptr %44, align 8, !tbaa !88
  %45 = getelementptr inbounds nuw i8, ptr %42, i64 512
  store ptr %45, ptr %25, align 8, !tbaa !89
  %46 = getelementptr inbounds nuw i8, ptr %42, i64 480
  store ptr %46, ptr %4, align 8, !tbaa !45
  %47 = getelementptr inbounds nuw i8, ptr %42, i64 496
  store ptr %47, ptr %46, align 8, !tbaa !38
  %48 = load ptr, ptr %1, align 8, !tbaa !42
  %49 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %50 = icmp eq ptr %48, %49
  br i1 %50, label %51, label %56

51:                                               ; preds = %40
  %52 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %53 = load i64, ptr %52, align 8, !tbaa !44
  %54 = icmp ult i64 %53, 16
  tail call void @llvm.assume(i1 %54)
  %55 = add nuw nsw i64 %53, 1
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(1) %47, ptr noundef nonnull align 8 dereferenceable(1) %49, i64 %55, i1 false)
  br label %60

56:                                               ; preds = %40
  store ptr %48, ptr %46, align 8, !tbaa !42
  %57 = load i64, ptr %49, align 8, !tbaa !36
  store i64 %57, ptr %47, align 8, !tbaa !36
  %58 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %59 = load i64, ptr %58, align 8, !tbaa !44
  br label %60

60:                                               ; preds = %51, %56
  %61 = phi i64 [ %53, %51 ], [ %59, %56 ]
  %62 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %63 = getelementptr inbounds nuw i8, ptr %42, i64 488
  store i64 %61, ptr %63, align 8, !tbaa !44
  store ptr %49, ptr %1, align 8, !tbaa !42
  store i64 0, ptr %62, align 8, !tbaa !44
  store i8 0, ptr %49, align 8, !tbaa !36
  ret void
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #11

; Function Attrs: cold noreturn
declare void @_ZSt20__throw_length_errorPKc(ptr noundef) local_unnamed_addr #10

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZNSt5dequeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESaIS5_EE17_M_reallocate_mapEmb(ptr noundef nonnull align 8 dereferenceable(80) %0, i64 noundef %1, i1 noundef %2) local_unnamed_addr #3 comdat personality ptr @__gxx_personality_v0 {
  %4 = getelementptr inbounds nuw i8, ptr %0, i64 72
  %5 = load ptr, ptr %4, align 8, !tbaa !82
  %6 = getelementptr inbounds nuw i8, ptr %0, i64 40
  %7 = load ptr, ptr %6, align 8, !tbaa !81
  %8 = ptrtoint ptr %5 to i64
  %9 = ptrtoint ptr %7 to i64
  %10 = sub i64 %8, %9
  %11 = ashr exact i64 %10, 3
  %12 = add nsw i64 %11, 1
  %13 = add i64 %12, %1
  %14 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %15 = load i64, ptr %14, align 8, !tbaa !84
  %16 = shl i64 %13, 1
  %17 = icmp ugt i64 %15, %16
  br i1 %17, label %18, label %50

18:                                               ; preds = %3
  %19 = load ptr, ptr %0, align 8, !tbaa !80
  %20 = sub i64 %15, %13
  %21 = lshr i64 %20, 1
  %22 = getelementptr inbounds nuw ptr, ptr %19, i64 %21
  %23 = select i1 %2, i64 %1, i64 0
  %24 = getelementptr inbounds nuw ptr, ptr %22, i64 %23
  %25 = icmp ult ptr %24, %7
  %26 = getelementptr inbounds nuw i8, ptr %5, i64 8
  br i1 %25, label %27, label %36

27:                                               ; preds = %18
  %28 = ptrtoint ptr %26 to i64
  %29 = sub i64 %28, %9
  %30 = icmp sgt i64 %29, 8
  br i1 %30, label %31, label %32, !prof !94

31:                                               ; preds = %27
  tail call void @llvm.memmove.p0.p0.i64(ptr align 8 %24, ptr nonnull align 8 %7, i64 %29, i1 false)
  br label %79

32:                                               ; preds = %27
  %33 = icmp eq i64 %29, 8
  br i1 %33, label %34, label %79

34:                                               ; preds = %32
  %35 = load ptr, ptr %7, align 8, !tbaa !60
  store ptr %35, ptr %24, align 8, !tbaa !60
  br label %79

36:                                               ; preds = %18
  %37 = getelementptr inbounds nuw ptr, ptr %24, i64 %12
  %38 = ptrtoint ptr %26 to i64
  %39 = sub i64 %38, %9
  %40 = ashr exact i64 %39, 3
  %41 = icmp sgt i64 %40, 1
  br i1 %41, label %42, label %45, !prof !94

42:                                               ; preds = %36
  %43 = sub nsw i64 0, %40
  %44 = getelementptr inbounds ptr, ptr %37, i64 %43
  tail call void @llvm.memmove.p0.p0.i64(ptr nonnull align 8 %44, ptr align 8 %7, i64 %39, i1 false)
  br label %79

45:                                               ; preds = %36
  %46 = icmp eq i64 %39, 8
  br i1 %46, label %47, label %79

47:                                               ; preds = %45
  %48 = getelementptr inbounds i8, ptr %37, i64 -8
  %49 = load ptr, ptr %7, align 8, !tbaa !60
  store ptr %49, ptr %48, align 8, !tbaa !60
  br label %79

50:                                               ; preds = %3
  %51 = tail call i64 @llvm.umax.i64(i64 %15, i64 %1)
  %52 = add i64 %15, 2
  %53 = add i64 %52, %51
  %54 = icmp ugt i64 %53, 1152921504606846975
  br i1 %54, label %55, label %59, !prof !85

55:                                               ; preds = %50
  %56 = icmp ugt i64 %53, 2305843009213693951
  br i1 %56, label %57, label %58

57:                                               ; preds = %55
  tail call void @_ZSt28__throw_bad_array_new_lengthv() #20
  unreachable

58:                                               ; preds = %55
  tail call void @_ZSt17__throw_bad_allocv() #20
  unreachable

59:                                               ; preds = %50
  %60 = shl nuw nsw i64 %53, 3
  %61 = tail call noalias noundef nonnull ptr @_Znwm(i64 noundef %60) #21
  %62 = sub i64 %53, %13
  %63 = lshr i64 %62, 1
  %64 = getelementptr inbounds nuw ptr, ptr %61, i64 %63
  %65 = select i1 %2, i64 %1, i64 0
  %66 = getelementptr inbounds nuw ptr, ptr %64, i64 %65
  %67 = getelementptr inbounds nuw i8, ptr %5, i64 8
  %68 = ptrtoint ptr %67 to i64
  %69 = sub i64 %68, %9
  %70 = icmp sgt i64 %69, 8
  br i1 %70, label %71, label %72, !prof !94

71:                                               ; preds = %59
  tail call void @llvm.memmove.p0.p0.i64(ptr nonnull align 8 %66, ptr align 8 %7, i64 %69, i1 false)
  br label %76

72:                                               ; preds = %59
  %73 = icmp eq i64 %69, 8
  br i1 %73, label %74, label %76

74:                                               ; preds = %72
  %75 = load ptr, ptr %7, align 8, !tbaa !60
  store ptr %75, ptr %66, align 8, !tbaa !60
  br label %76

76:                                               ; preds = %71, %72, %74
  %77 = load ptr, ptr %0, align 8, !tbaa !80
  %78 = shl i64 %15, 3
  tail call void @_ZdlPvm(ptr noundef %77, i64 noundef %78) #18
  store ptr %61, ptr %0, align 8, !tbaa !80
  store i64 %53, ptr %14, align 8, !tbaa !84
  br label %79

79:                                               ; preds = %47, %45, %42, %34, %32, %31, %76
  %80 = phi ptr [ %66, %76 ], [ %24, %31 ], [ %24, %32 ], [ %24, %34 ], [ %24, %42 ], [ %24, %45 ], [ %24, %47 ]
  store ptr %80, ptr %6, align 8, !tbaa !87
  %81 = load ptr, ptr %80, align 8, !tbaa !60
  %82 = getelementptr inbounds nuw i8, ptr %0, i64 24
  store ptr %81, ptr %82, align 8, !tbaa !88
  %83 = getelementptr inbounds nuw i8, ptr %81, i64 512
  %84 = getelementptr inbounds nuw i8, ptr %0, i64 32
  store ptr %83, ptr %84, align 8, !tbaa !89
  %85 = getelementptr inbounds nuw ptr, ptr %80, i64 %12
  %86 = getelementptr inbounds i8, ptr %85, i64 -8
  store ptr %86, ptr %4, align 8, !tbaa !87
  %87 = load ptr, ptr %86, align 8, !tbaa !60
  %88 = getelementptr inbounds nuw i8, ptr %0, i64 56
  store ptr %87, ptr %88, align 8, !tbaa !88
  %89 = getelementptr inbounds nuw i8, ptr %87, i64 512
  %90 = getelementptr inbounds nuw i8, ptr %0, i64 64
  store ptr %89, ptr %90, align 8, !tbaa !89
  ret void
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memmove.p0.p0.i64(ptr writeonly captures(none), ptr readonly captures(none), i64, i1 immarg) #11

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: read)
declare i64 @strlen(ptr noundef captures(none)) local_unnamed_addr #12

declare noundef ptr @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm(ptr noundef nonnull align 8 dereferenceable(32), ptr noundef nonnull align 8 dereferenceable(8), i64 noundef) local_unnamed_addr #9

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZSt15__copy_move_ditILb0ENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERS5_PS5_St16ostream_iteratorIS5_cS3_EET3_St15_Deque_iteratorIT0_T1_T2_ESF_SA_(ptr dead_on_unwind noalias writable sret(%"class.std::ostream_iterator") align 8 %0, ptr dead_on_return noundef %1, ptr dead_on_return noundef %2, ptr dead_on_return noundef %3) local_unnamed_addr #3 comdat {
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 24
  %6 = load ptr, ptr %5, align 8, !tbaa !87
  %7 = getelementptr inbounds nuw i8, ptr %2, i64 24
  %8 = load ptr, ptr %7, align 8, !tbaa !87
  %9 = icmp eq ptr %6, %8
  %10 = load ptr, ptr %1, align 8, !tbaa !91
  br i1 %9, label %282, label %11

11:                                               ; preds = %4
  %12 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %13 = load ptr, ptr %12, align 8, !tbaa !89
  %14 = load ptr, ptr %3, align 8, !tbaa !71
  %15 = getelementptr inbounds nuw i8, ptr %3, i64 8
  %16 = load ptr, ptr %15, align 8, !tbaa !73
  %17 = freeze ptr %16
  %18 = ptrtoint ptr %13 to i64
  %19 = ptrtoint ptr %10 to i64
  %20 = sub i64 %18, %19
  %21 = ashr exact i64 %20, 5
  %22 = icmp sgt i64 %21, 0
  br i1 %22, label %23, label %47

23:                                               ; preds = %11
  %24 = icmp eq ptr %17, null
  br i1 %24, label %25, label %35

25:                                               ; preds = %23, %25
  %26 = phi i64 [ %33, %25 ], [ %21, %23 ]
  %27 = phi ptr [ %32, %25 ], [ %10, %23 ]
  %28 = load ptr, ptr %27, align 8, !tbaa !42, !noalias !95
  %29 = getelementptr inbounds nuw i8, ptr %27, i64 8
  %30 = load i64, ptr %29, align 8, !tbaa !44, !noalias !95
  %31 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %14, ptr noundef %28, i64 noundef %30), !noalias !95
  %32 = getelementptr inbounds nuw i8, ptr %27, i64 32
  %33 = add nsw i64 %26, -1
  %34 = icmp samesign ugt i64 %26, 1
  br i1 %34, label %25, label %47, !llvm.loop !102

35:                                               ; preds = %23, %35
  %36 = phi i64 [ %45, %35 ], [ %21, %23 ]
  %37 = phi ptr [ %44, %35 ], [ %10, %23 ]
  %38 = load ptr, ptr %37, align 8, !tbaa !42, !noalias !95
  %39 = getelementptr inbounds nuw i8, ptr %37, i64 8
  %40 = load i64, ptr %39, align 8, !tbaa !44, !noalias !95
  %41 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %14, ptr noundef %38, i64 noundef %40), !noalias !95
  %42 = tail call noundef i64 @strlen(ptr noundef nonnull dereferenceable(1) %17) #16, !noalias !103
  %43 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %14, ptr noundef nonnull %17, i64 noundef %42), !noalias !103
  %44 = getelementptr inbounds nuw i8, ptr %37, i64 32
  %45 = add nsw i64 %36, -1
  %46 = icmp samesign ugt i64 %36, 1
  br i1 %46, label %35, label %47, !llvm.loop !102

47:                                               ; preds = %35, %25, %11
  store ptr %14, ptr %3, align 8
  store ptr %17, ptr %15, align 8
  %48 = load ptr, ptr %5, align 8, !tbaa !87
  %49 = getelementptr inbounds nuw i8, ptr %48, i64 8
  %50 = load ptr, ptr %7, align 8, !tbaa !87
  %51 = icmp eq ptr %49, %50
  br i1 %51, label %54, label %52

52:                                               ; preds = %47
  %53 = icmp eq ptr %17, null
  br label %87

54:                                               ; preds = %278, %47
  %55 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %56 = load ptr, ptr %55, align 8, !tbaa !88
  %57 = load ptr, ptr %2, align 8, !tbaa !91
  %58 = ptrtoint ptr %57 to i64
  %59 = ptrtoint ptr %56 to i64
  %60 = sub i64 %58, %59
  %61 = ashr exact i64 %60, 5
  %62 = icmp sgt i64 %61, 0
  br i1 %62, label %63, label %317

63:                                               ; preds = %54
  %64 = icmp eq ptr %17, null
  br i1 %64, label %65, label %75

65:                                               ; preds = %63, %65
  %66 = phi i64 [ %73, %65 ], [ %61, %63 ]
  %67 = phi ptr [ %72, %65 ], [ %56, %63 ]
  %68 = load ptr, ptr %67, align 8, !tbaa !42, !noalias !104
  %69 = getelementptr inbounds nuw i8, ptr %67, i64 8
  %70 = load i64, ptr %69, align 8, !tbaa !44, !noalias !104
  %71 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %14, ptr noundef %68, i64 noundef %70), !noalias !104
  %72 = getelementptr inbounds nuw i8, ptr %67, i64 32
  %73 = add nsw i64 %66, -1
  %74 = icmp samesign ugt i64 %66, 1
  br i1 %74, label %65, label %317, !llvm.loop !102

75:                                               ; preds = %63, %75
  %76 = phi i64 [ %85, %75 ], [ %61, %63 ]
  %77 = phi ptr [ %84, %75 ], [ %56, %63 ]
  %78 = load ptr, ptr %77, align 8, !tbaa !42, !noalias !104
  %79 = getelementptr inbounds nuw i8, ptr %77, i64 8
  %80 = load i64, ptr %79, align 8, !tbaa !44, !noalias !104
  %81 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %14, ptr noundef %78, i64 noundef %80), !noalias !104
  %82 = tail call noundef i64 @strlen(ptr noundef nonnull dereferenceable(1) %17) #16, !noalias !111
  %83 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %14, ptr noundef nonnull %17, i64 noundef %82), !noalias !111
  %84 = getelementptr inbounds nuw i8, ptr %77, i64 32
  %85 = add nsw i64 %76, -1
  %86 = icmp samesign ugt i64 %76, 1
  br i1 %86, label %75, label %317, !llvm.loop !102

87:                                               ; preds = %52, %278
  %88 = phi ptr [ %279, %278 ], [ %49, %52 ]
  %89 = load ptr, ptr %88, align 8, !tbaa !60
  %90 = load ptr, ptr %89, align 8, !tbaa !42, !noalias !112
  %91 = getelementptr inbounds nuw i8, ptr %89, i64 8
  %92 = load i64, ptr %91, align 8, !tbaa !44, !noalias !112
  %93 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %14, ptr noundef %90, i64 noundef %92), !noalias !112
  br i1 %53, label %202, label %94

94:                                               ; preds = %87
  %95 = tail call noundef i64 @strlen(ptr noundef nonnull dereferenceable(1) %17) #16, !noalias !119
  %96 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %14, ptr noundef nonnull %17, i64 noundef %95), !noalias !119
  %97 = getelementptr inbounds nuw i8, ptr %89, i64 32
  %98 = load ptr, ptr %97, align 8, !tbaa !42, !noalias !112
  %99 = getelementptr inbounds nuw i8, ptr %89, i64 40
  %100 = load i64, ptr %99, align 8, !tbaa !44, !noalias !112
  %101 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %14, ptr noundef %98, i64 noundef %100), !noalias !112
  %102 = tail call noundef i64 @strlen(ptr noundef nonnull dereferenceable(1) %17) #16, !noalias !119
  %103 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %14, ptr noundef nonnull %17, i64 noundef %102), !noalias !119
  %104 = getelementptr inbounds nuw i8, ptr %89, i64 64
  %105 = load ptr, ptr %104, align 8, !tbaa !42, !noalias !112
  %106 = getelementptr inbounds nuw i8, ptr %89, i64 72
  %107 = load i64, ptr %106, align 8, !tbaa !44, !noalias !112
  %108 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %14, ptr noundef %105, i64 noundef %107), !noalias !112
  %109 = tail call noundef i64 @strlen(ptr noundef nonnull dereferenceable(1) %17) #16, !noalias !119
  %110 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %14, ptr noundef nonnull %17, i64 noundef %109), !noalias !119
  %111 = getelementptr inbounds nuw i8, ptr %89, i64 96
  %112 = load ptr, ptr %111, align 8, !tbaa !42, !noalias !112
  %113 = getelementptr inbounds nuw i8, ptr %89, i64 104
  %114 = load i64, ptr %113, align 8, !tbaa !44, !noalias !112
  %115 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %14, ptr noundef %112, i64 noundef %114), !noalias !112
  %116 = tail call noundef i64 @strlen(ptr noundef nonnull dereferenceable(1) %17) #16, !noalias !119
  %117 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %14, ptr noundef nonnull %17, i64 noundef %116), !noalias !119
  %118 = getelementptr inbounds nuw i8, ptr %89, i64 128
  %119 = load ptr, ptr %118, align 8, !tbaa !42, !noalias !112
  %120 = getelementptr inbounds nuw i8, ptr %89, i64 136
  %121 = load i64, ptr %120, align 8, !tbaa !44, !noalias !112
  %122 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %14, ptr noundef %119, i64 noundef %121), !noalias !112
  %123 = tail call noundef i64 @strlen(ptr noundef nonnull dereferenceable(1) %17) #16, !noalias !119
  %124 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %14, ptr noundef nonnull %17, i64 noundef %123), !noalias !119
  %125 = getelementptr inbounds nuw i8, ptr %89, i64 160
  %126 = load ptr, ptr %125, align 8, !tbaa !42, !noalias !112
  %127 = getelementptr inbounds nuw i8, ptr %89, i64 168
  %128 = load i64, ptr %127, align 8, !tbaa !44, !noalias !112
  %129 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %14, ptr noundef %126, i64 noundef %128), !noalias !112
  %130 = tail call noundef i64 @strlen(ptr noundef nonnull dereferenceable(1) %17) #16, !noalias !119
  %131 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %14, ptr noundef nonnull %17, i64 noundef %130), !noalias !119
  %132 = getelementptr inbounds nuw i8, ptr %89, i64 192
  %133 = load ptr, ptr %132, align 8, !tbaa !42, !noalias !112
  %134 = getelementptr inbounds nuw i8, ptr %89, i64 200
  %135 = load i64, ptr %134, align 8, !tbaa !44, !noalias !112
  %136 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %14, ptr noundef %133, i64 noundef %135), !noalias !112
  %137 = tail call noundef i64 @strlen(ptr noundef nonnull dereferenceable(1) %17) #16, !noalias !119
  %138 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %14, ptr noundef nonnull %17, i64 noundef %137), !noalias !119
  %139 = getelementptr inbounds nuw i8, ptr %89, i64 224
  %140 = load ptr, ptr %139, align 8, !tbaa !42, !noalias !112
  %141 = getelementptr inbounds nuw i8, ptr %89, i64 232
  %142 = load i64, ptr %141, align 8, !tbaa !44, !noalias !112
  %143 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %14, ptr noundef %140, i64 noundef %142), !noalias !112
  %144 = tail call noundef i64 @strlen(ptr noundef nonnull dereferenceable(1) %17) #16, !noalias !119
  %145 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %14, ptr noundef nonnull %17, i64 noundef %144), !noalias !119
  %146 = getelementptr inbounds nuw i8, ptr %89, i64 256
  %147 = load ptr, ptr %146, align 8, !tbaa !42, !noalias !112
  %148 = getelementptr inbounds nuw i8, ptr %89, i64 264
  %149 = load i64, ptr %148, align 8, !tbaa !44, !noalias !112
  %150 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %14, ptr noundef %147, i64 noundef %149), !noalias !112
  %151 = tail call noundef i64 @strlen(ptr noundef nonnull dereferenceable(1) %17) #16, !noalias !119
  %152 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %14, ptr noundef nonnull %17, i64 noundef %151), !noalias !119
  %153 = getelementptr inbounds nuw i8, ptr %89, i64 288
  %154 = load ptr, ptr %153, align 8, !tbaa !42, !noalias !112
  %155 = getelementptr inbounds nuw i8, ptr %89, i64 296
  %156 = load i64, ptr %155, align 8, !tbaa !44, !noalias !112
  %157 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %14, ptr noundef %154, i64 noundef %156), !noalias !112
  %158 = tail call noundef i64 @strlen(ptr noundef nonnull dereferenceable(1) %17) #16, !noalias !119
  %159 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %14, ptr noundef nonnull %17, i64 noundef %158), !noalias !119
  %160 = getelementptr inbounds nuw i8, ptr %89, i64 320
  %161 = load ptr, ptr %160, align 8, !tbaa !42, !noalias !112
  %162 = getelementptr inbounds nuw i8, ptr %89, i64 328
  %163 = load i64, ptr %162, align 8, !tbaa !44, !noalias !112
  %164 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %14, ptr noundef %161, i64 noundef %163), !noalias !112
  %165 = tail call noundef i64 @strlen(ptr noundef nonnull dereferenceable(1) %17) #16, !noalias !119
  %166 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %14, ptr noundef nonnull %17, i64 noundef %165), !noalias !119
  %167 = getelementptr inbounds nuw i8, ptr %89, i64 352
  %168 = load ptr, ptr %167, align 8, !tbaa !42, !noalias !112
  %169 = getelementptr inbounds nuw i8, ptr %89, i64 360
  %170 = load i64, ptr %169, align 8, !tbaa !44, !noalias !112
  %171 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %14, ptr noundef %168, i64 noundef %170), !noalias !112
  %172 = tail call noundef i64 @strlen(ptr noundef nonnull dereferenceable(1) %17) #16, !noalias !119
  %173 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %14, ptr noundef nonnull %17, i64 noundef %172), !noalias !119
  %174 = getelementptr inbounds nuw i8, ptr %89, i64 384
  %175 = load ptr, ptr %174, align 8, !tbaa !42, !noalias !112
  %176 = getelementptr inbounds nuw i8, ptr %89, i64 392
  %177 = load i64, ptr %176, align 8, !tbaa !44, !noalias !112
  %178 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %14, ptr noundef %175, i64 noundef %177), !noalias !112
  %179 = tail call noundef i64 @strlen(ptr noundef nonnull dereferenceable(1) %17) #16, !noalias !119
  %180 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %14, ptr noundef nonnull %17, i64 noundef %179), !noalias !119
  %181 = getelementptr inbounds nuw i8, ptr %89, i64 416
  %182 = load ptr, ptr %181, align 8, !tbaa !42, !noalias !112
  %183 = getelementptr inbounds nuw i8, ptr %89, i64 424
  %184 = load i64, ptr %183, align 8, !tbaa !44, !noalias !112
  %185 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %14, ptr noundef %182, i64 noundef %184), !noalias !112
  %186 = tail call noundef i64 @strlen(ptr noundef nonnull dereferenceable(1) %17) #16, !noalias !119
  %187 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %14, ptr noundef nonnull %17, i64 noundef %186), !noalias !119
  %188 = getelementptr inbounds nuw i8, ptr %89, i64 448
  %189 = load ptr, ptr %188, align 8, !tbaa !42, !noalias !112
  %190 = getelementptr inbounds nuw i8, ptr %89, i64 456
  %191 = load i64, ptr %190, align 8, !tbaa !44, !noalias !112
  %192 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %14, ptr noundef %189, i64 noundef %191), !noalias !112
  %193 = tail call noundef i64 @strlen(ptr noundef nonnull dereferenceable(1) %17) #16, !noalias !119
  %194 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %14, ptr noundef nonnull %17, i64 noundef %193), !noalias !119
  %195 = getelementptr inbounds nuw i8, ptr %89, i64 480
  %196 = load ptr, ptr %195, align 8, !tbaa !42, !noalias !112
  %197 = getelementptr inbounds nuw i8, ptr %89, i64 488
  %198 = load i64, ptr %197, align 8, !tbaa !44, !noalias !112
  %199 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %14, ptr noundef %196, i64 noundef %198), !noalias !112
  %200 = tail call noundef i64 @strlen(ptr noundef nonnull dereferenceable(1) %17) #16, !noalias !119
  %201 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %14, ptr noundef nonnull %17, i64 noundef %200), !noalias !119
  br label %278

202:                                              ; preds = %87
  %203 = getelementptr inbounds nuw i8, ptr %89, i64 32
  %204 = load ptr, ptr %203, align 8, !tbaa !42, !noalias !112
  %205 = getelementptr inbounds nuw i8, ptr %89, i64 40
  %206 = load i64, ptr %205, align 8, !tbaa !44, !noalias !112
  %207 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %14, ptr noundef %204, i64 noundef %206), !noalias !112
  %208 = getelementptr inbounds nuw i8, ptr %89, i64 64
  %209 = load ptr, ptr %208, align 8, !tbaa !42, !noalias !112
  %210 = getelementptr inbounds nuw i8, ptr %89, i64 72
  %211 = load i64, ptr %210, align 8, !tbaa !44, !noalias !112
  %212 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %14, ptr noundef %209, i64 noundef %211), !noalias !112
  %213 = getelementptr inbounds nuw i8, ptr %89, i64 96
  %214 = load ptr, ptr %213, align 8, !tbaa !42, !noalias !112
  %215 = getelementptr inbounds nuw i8, ptr %89, i64 104
  %216 = load i64, ptr %215, align 8, !tbaa !44, !noalias !112
  %217 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %14, ptr noundef %214, i64 noundef %216), !noalias !112
  %218 = getelementptr inbounds nuw i8, ptr %89, i64 128
  %219 = load ptr, ptr %218, align 8, !tbaa !42, !noalias !112
  %220 = getelementptr inbounds nuw i8, ptr %89, i64 136
  %221 = load i64, ptr %220, align 8, !tbaa !44, !noalias !112
  %222 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %14, ptr noundef %219, i64 noundef %221), !noalias !112
  %223 = getelementptr inbounds nuw i8, ptr %89, i64 160
  %224 = load ptr, ptr %223, align 8, !tbaa !42, !noalias !112
  %225 = getelementptr inbounds nuw i8, ptr %89, i64 168
  %226 = load i64, ptr %225, align 8, !tbaa !44, !noalias !112
  %227 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %14, ptr noundef %224, i64 noundef %226), !noalias !112
  %228 = getelementptr inbounds nuw i8, ptr %89, i64 192
  %229 = load ptr, ptr %228, align 8, !tbaa !42, !noalias !112
  %230 = getelementptr inbounds nuw i8, ptr %89, i64 200
  %231 = load i64, ptr %230, align 8, !tbaa !44, !noalias !112
  %232 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %14, ptr noundef %229, i64 noundef %231), !noalias !112
  %233 = getelementptr inbounds nuw i8, ptr %89, i64 224
  %234 = load ptr, ptr %233, align 8, !tbaa !42, !noalias !112
  %235 = getelementptr inbounds nuw i8, ptr %89, i64 232
  %236 = load i64, ptr %235, align 8, !tbaa !44, !noalias !112
  %237 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %14, ptr noundef %234, i64 noundef %236), !noalias !112
  %238 = getelementptr inbounds nuw i8, ptr %89, i64 256
  %239 = load ptr, ptr %238, align 8, !tbaa !42, !noalias !112
  %240 = getelementptr inbounds nuw i8, ptr %89, i64 264
  %241 = load i64, ptr %240, align 8, !tbaa !44, !noalias !112
  %242 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %14, ptr noundef %239, i64 noundef %241), !noalias !112
  %243 = getelementptr inbounds nuw i8, ptr %89, i64 288
  %244 = load ptr, ptr %243, align 8, !tbaa !42, !noalias !112
  %245 = getelementptr inbounds nuw i8, ptr %89, i64 296
  %246 = load i64, ptr %245, align 8, !tbaa !44, !noalias !112
  %247 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %14, ptr noundef %244, i64 noundef %246), !noalias !112
  %248 = getelementptr inbounds nuw i8, ptr %89, i64 320
  %249 = load ptr, ptr %248, align 8, !tbaa !42, !noalias !112
  %250 = getelementptr inbounds nuw i8, ptr %89, i64 328
  %251 = load i64, ptr %250, align 8, !tbaa !44, !noalias !112
  %252 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %14, ptr noundef %249, i64 noundef %251), !noalias !112
  %253 = getelementptr inbounds nuw i8, ptr %89, i64 352
  %254 = load ptr, ptr %253, align 8, !tbaa !42, !noalias !112
  %255 = getelementptr inbounds nuw i8, ptr %89, i64 360
  %256 = load i64, ptr %255, align 8, !tbaa !44, !noalias !112
  %257 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %14, ptr noundef %254, i64 noundef %256), !noalias !112
  %258 = getelementptr inbounds nuw i8, ptr %89, i64 384
  %259 = load ptr, ptr %258, align 8, !tbaa !42, !noalias !112
  %260 = getelementptr inbounds nuw i8, ptr %89, i64 392
  %261 = load i64, ptr %260, align 8, !tbaa !44, !noalias !112
  %262 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %14, ptr noundef %259, i64 noundef %261), !noalias !112
  %263 = getelementptr inbounds nuw i8, ptr %89, i64 416
  %264 = load ptr, ptr %263, align 8, !tbaa !42, !noalias !112
  %265 = getelementptr inbounds nuw i8, ptr %89, i64 424
  %266 = load i64, ptr %265, align 8, !tbaa !44, !noalias !112
  %267 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %14, ptr noundef %264, i64 noundef %266), !noalias !112
  %268 = getelementptr inbounds nuw i8, ptr %89, i64 448
  %269 = load ptr, ptr %268, align 8, !tbaa !42, !noalias !112
  %270 = getelementptr inbounds nuw i8, ptr %89, i64 456
  %271 = load i64, ptr %270, align 8, !tbaa !44, !noalias !112
  %272 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %14, ptr noundef %269, i64 noundef %271), !noalias !112
  %273 = getelementptr inbounds nuw i8, ptr %89, i64 480
  %274 = load ptr, ptr %273, align 8, !tbaa !42, !noalias !112
  %275 = getelementptr inbounds nuw i8, ptr %89, i64 488
  %276 = load i64, ptr %275, align 8, !tbaa !44, !noalias !112
  %277 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %14, ptr noundef %274, i64 noundef %276), !noalias !112
  br label %278

278:                                              ; preds = %94, %202
  store ptr %14, ptr %3, align 8
  store ptr %17, ptr %15, align 8
  %279 = getelementptr inbounds nuw i8, ptr %88, i64 8
  %280 = load ptr, ptr %7, align 8, !tbaa !87
  %281 = icmp eq ptr %279, %280
  br i1 %281, label %54, label %87, !llvm.loop !120

282:                                              ; preds = %4
  %283 = load ptr, ptr %2, align 8, !tbaa !91
  %284 = load ptr, ptr %3, align 8, !tbaa !71
  %285 = getelementptr inbounds nuw i8, ptr %3, i64 8
  %286 = load ptr, ptr %285, align 8, !tbaa !73
  %287 = freeze ptr %286
  %288 = ptrtoint ptr %283 to i64
  %289 = ptrtoint ptr %10 to i64
  %290 = sub i64 %288, %289
  %291 = ashr exact i64 %290, 5
  %292 = icmp sgt i64 %291, 0
  br i1 %292, label %293, label %317

293:                                              ; preds = %282
  %294 = icmp eq ptr %287, null
  br i1 %294, label %295, label %305

295:                                              ; preds = %293, %295
  %296 = phi i64 [ %303, %295 ], [ %291, %293 ]
  %297 = phi ptr [ %302, %295 ], [ %10, %293 ]
  %298 = load ptr, ptr %297, align 8, !tbaa !42, !noalias !121
  %299 = getelementptr inbounds nuw i8, ptr %297, i64 8
  %300 = load i64, ptr %299, align 8, !tbaa !44, !noalias !121
  %301 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %284, ptr noundef %298, i64 noundef %300), !noalias !121
  %302 = getelementptr inbounds nuw i8, ptr %297, i64 32
  %303 = add nsw i64 %296, -1
  %304 = icmp samesign ugt i64 %296, 1
  br i1 %304, label %295, label %317, !llvm.loop !102

305:                                              ; preds = %293, %305
  %306 = phi i64 [ %315, %305 ], [ %291, %293 ]
  %307 = phi ptr [ %314, %305 ], [ %10, %293 ]
  %308 = load ptr, ptr %307, align 8, !tbaa !42, !noalias !121
  %309 = getelementptr inbounds nuw i8, ptr %307, i64 8
  %310 = load i64, ptr %309, align 8, !tbaa !44, !noalias !121
  %311 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %284, ptr noundef %308, i64 noundef %310), !noalias !121
  %312 = tail call noundef i64 @strlen(ptr noundef nonnull dereferenceable(1) %287) #16, !noalias !128
  %313 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %284, ptr noundef nonnull %287, i64 noundef %312), !noalias !128
  %314 = getelementptr inbounds nuw i8, ptr %307, i64 32
  %315 = add nsw i64 %306, -1
  %316 = icmp samesign ugt i64 %306, 1
  br i1 %316, label %305, label %317, !llvm.loop !102

317:                                              ; preds = %75, %65, %305, %295, %282, %54
  %318 = phi ptr [ %14, %54 ], [ %284, %282 ], [ %284, %295 ], [ %284, %305 ], [ %14, %65 ], [ %14, %75 ]
  %319 = phi ptr [ %17, %54 ], [ %287, %282 ], [ %287, %295 ], [ %287, %305 ], [ %17, %65 ], [ %17, %75 ]
  store ptr %318, ptr %0, align 8, !tbaa !71
  %320 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store ptr %319, ptr %320, align 8, !tbaa !73
  ret void
}

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8), ptr noundef, i64 noundef) local_unnamed_addr #9

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #13

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #14

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.umax.i64(i64, i64) #15

attributes #0 = { mustprogress norecurse uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { noinline noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { cold nofree noreturn }
attributes #6 = { noreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { nobuiltin allocsize(0) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #8 = { nobuiltin nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #9 = { "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #10 = { cold noreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #11 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #12 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: read) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #13 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
attributes #14 = { nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #15 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #16 = { nounwind }
attributes #17 = { cold noreturn }
attributes #18 = { builtin nounwind }
attributes #19 = { noreturn nounwind }
attributes #20 = { noreturn }
attributes #21 = { builtin allocsize(0) }

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
!29 = !{!10, !26, i64 240}
!30 = !{!31, !13, i64 56}
!31 = !{!"_ZTSSt5ctypeIcE", !32, i64 0, !33, i64 16, !24, i64 24, !34, i64 32, !34, i64 40, !35, i64 48, !13, i64 56, !13, i64 57, !13, i64 313, !13, i64 569}
!32 = !{!"_ZTSNSt6locale5facetE", !19, i64 8}
!33 = !{!"p1 _ZTS15__locale_struct", !17, i64 0}
!34 = !{!"p1 int", !17, i64 0}
!35 = !{!"p1 short", !17, i64 0}
!36 = !{!13, !13, i64 0}
!37 = !{!11, !15, i64 32}
!38 = !{!39, !40, i64 0}
!39 = !{!"_ZTSNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_Alloc_hiderE", !40, i64 0}
!40 = !{!"p1 omnipotent char", !17, i64 0}
!41 = !{!12, !12, i64 0}
!42 = !{!43, !40, i64 0}
!43 = !{!"_ZTSNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE", !39, i64 0, !12, i64 8, !13, i64 16}
!44 = !{!43, !12, i64 8}
!45 = !{!46, !50, i64 16}
!46 = !{!"_ZTSNSt11_Deque_baseINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESaIS5_EE16_Deque_impl_dataE", !47, i64 0, !12, i64 8, !49, i64 16, !49, i64 48}
!47 = !{!"p2 _ZTSNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE", !48, i64 0}
!48 = !{!"any p2 pointer", !17, i64 0}
!49 = !{!"_ZTSSt15_Deque_iteratorINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERS5_PS5_E", !50, i64 0, !50, i64 8, !50, i64 16, !47, i64 24}
!50 = !{!"p1 _ZTSNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE", !17, i64 0}
!51 = !{!46, !50, i64 24}
!52 = distinct !{!52, !53}
!53 = !{!"llvm.loop.mustprogress"}
!54 = !{!55}
!55 = distinct !{!55, !56, !"_ZSt4copyISt15_Deque_iteratorINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERS6_PS6_ESt16ostream_iteratorIS6_cS4_EET0_T_SD_SC_: argument 0"}
!56 = distinct !{!56, !"_ZSt4copyISt15_Deque_iteratorINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERS6_PS6_ESt16ostream_iteratorIS6_cS4_EET0_T_SD_SC_"}
!57 = !{!58, !55}
!58 = distinct !{!58, !59, !"_ZSt13__copy_move_aILb0ESt15_Deque_iteratorINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERS6_PS6_ESt16ostream_iteratorIS6_cS4_EET1_T0_SD_SC_: argument 0"}
!59 = distinct !{!59, !"_ZSt13__copy_move_aILb0ESt15_Deque_iteratorINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERS6_PS6_ESt16ostream_iteratorIS6_cS4_EET1_T0_SD_SC_"}
!60 = !{!50, !50, i64 0}
!61 = !{!62}
!62 = distinct !{!62, !63, !"_ZNSt5dequeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESaIS5_EE5beginEv: argument 0"}
!63 = distinct !{!63, !"_ZNSt5dequeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESaIS5_EE5beginEv"}
!64 = !{!65, !58, !55}
!65 = distinct !{!65, !66, !"_ZSt14__copy_move_a1ILb0ENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERS5_PS5_St16ostream_iteratorIS5_cS3_EET3_St15_Deque_iteratorIT0_T1_T2_ESF_SA_: argument 0"}
!66 = distinct !{!66, !"_ZSt14__copy_move_a1ILb0ENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERS5_PS5_St16ostream_iteratorIS5_cS3_EET3_St15_Deque_iteratorIT0_T1_T2_ESF_SA_"}
!67 = !{!17, !17, i64 0}
!68 = !{!69}
!69 = distinct !{!69, !70, !"_ZNSt5dequeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESaIS5_EE3endEv: argument 0"}
!70 = distinct !{!70, !"_ZNSt5dequeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESaIS5_EE3endEv"}
!71 = !{!72, !23, i64 0}
!72 = !{!"_ZTSSt16ostream_iteratorINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEcS3_E", !23, i64 0, !40, i64 8}
!73 = !{!72, !40, i64 8}
!74 = !{!75}
!75 = distinct !{!75, !76, !"_ZNSt5dequeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESaIS5_EE5beginEv: argument 0"}
!76 = distinct !{!76, !"_ZNSt5dequeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESaIS5_EE5beginEv"}
!77 = !{!78}
!78 = distinct !{!78, !79, !"_ZNSt5dequeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESaIS5_EE3endEv: argument 0"}
!79 = distinct !{!79, !"_ZNSt5dequeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESaIS5_EE3endEv"}
!80 = !{!46, !47, i64 0}
!81 = !{!46, !47, i64 40}
!82 = !{!46, !47, i64 72}
!83 = distinct !{!83, !53}
!84 = !{!46, !12, i64 8}
!85 = !{!"branch_weights", !"expected", i32 1, i32 2000}
!86 = distinct !{!86, !53}
!87 = !{!49, !47, i64 24}
!88 = !{!49, !50, i64 8}
!89 = !{!49, !50, i64 16}
!90 = !{!46, !50, i64 48}
!91 = !{!49, !50, i64 0}
!92 = distinct !{!92, !53}
!93 = distinct !{!93, !53}
!94 = !{!"branch_weights", !"expected", i32 2000, i32 1}
!95 = !{!96, !98, !100}
!96 = distinct !{!96, !97, !"_ZNSt11__copy_moveILb0ELb0ESt26random_access_iterator_tagE8__copy_mIPNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt16ostream_iteratorIS8_cS6_EEET0_T_SD_SC_: argument 0"}
!97 = distinct !{!97, !"_ZNSt11__copy_moveILb0ELb0ESt26random_access_iterator_tagE8__copy_mIPNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt16ostream_iteratorIS8_cS6_EEET0_T_SD_SC_"}
!98 = distinct !{!98, !99, !"_ZSt14__copy_move_a2ILb0EPNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt16ostream_iteratorIS5_cS3_EET1_T0_SA_S9_: argument 0"}
!99 = distinct !{!99, !"_ZSt14__copy_move_a2ILb0EPNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt16ostream_iteratorIS5_cS3_EET1_T0_SA_S9_"}
!100 = distinct !{!100, !101, !"_ZSt14__copy_move_a1ILb0EPNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt16ostream_iteratorIS5_cS3_EET1_T0_SA_S9_: argument 0"}
!101 = distinct !{!101, !"_ZSt14__copy_move_a1ILb0EPNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt16ostream_iteratorIS5_cS3_EET1_T0_SA_S9_"}
!102 = distinct !{!102, !53}
!103 = !{!96}
!104 = !{!105, !107, !109}
!105 = distinct !{!105, !106, !"_ZNSt11__copy_moveILb0ELb0ESt26random_access_iterator_tagE8__copy_mIPNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt16ostream_iteratorIS8_cS6_EEET0_T_SD_SC_: argument 0"}
!106 = distinct !{!106, !"_ZNSt11__copy_moveILb0ELb0ESt26random_access_iterator_tagE8__copy_mIPNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt16ostream_iteratorIS8_cS6_EEET0_T_SD_SC_"}
!107 = distinct !{!107, !108, !"_ZSt14__copy_move_a2ILb0EPNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt16ostream_iteratorIS5_cS3_EET1_T0_SA_S9_: argument 0"}
!108 = distinct !{!108, !"_ZSt14__copy_move_a2ILb0EPNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt16ostream_iteratorIS5_cS3_EET1_T0_SA_S9_"}
!109 = distinct !{!109, !110, !"_ZSt14__copy_move_a1ILb0EPNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt16ostream_iteratorIS5_cS3_EET1_T0_SA_S9_: argument 0"}
!110 = distinct !{!110, !"_ZSt14__copy_move_a1ILb0EPNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt16ostream_iteratorIS5_cS3_EET1_T0_SA_S9_"}
!111 = !{!105}
!112 = !{!113, !115, !117}
!113 = distinct !{!113, !114, !"_ZNSt11__copy_moveILb0ELb0ESt26random_access_iterator_tagE8__copy_mIPNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt16ostream_iteratorIS8_cS6_EEET0_T_SD_SC_: argument 0"}
!114 = distinct !{!114, !"_ZNSt11__copy_moveILb0ELb0ESt26random_access_iterator_tagE8__copy_mIPNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt16ostream_iteratorIS8_cS6_EEET0_T_SD_SC_"}
!115 = distinct !{!115, !116, !"_ZSt14__copy_move_a2ILb0EPNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt16ostream_iteratorIS5_cS3_EET1_T0_SA_S9_: argument 0"}
!116 = distinct !{!116, !"_ZSt14__copy_move_a2ILb0EPNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt16ostream_iteratorIS5_cS3_EET1_T0_SA_S9_"}
!117 = distinct !{!117, !118, !"_ZSt14__copy_move_a1ILb0EPNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt16ostream_iteratorIS5_cS3_EET1_T0_SA_S9_: argument 0"}
!118 = distinct !{!118, !"_ZSt14__copy_move_a1ILb0EPNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt16ostream_iteratorIS5_cS3_EET1_T0_SA_S9_"}
!119 = !{!113}
!120 = distinct !{!120, !53}
!121 = !{!122, !124, !126}
!122 = distinct !{!122, !123, !"_ZNSt11__copy_moveILb0ELb0ESt26random_access_iterator_tagE8__copy_mIPNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt16ostream_iteratorIS8_cS6_EEET0_T_SD_SC_: argument 0"}
!123 = distinct !{!123, !"_ZNSt11__copy_moveILb0ELb0ESt26random_access_iterator_tagE8__copy_mIPNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt16ostream_iteratorIS8_cS6_EEET0_T_SD_SC_"}
!124 = distinct !{!124, !125, !"_ZSt14__copy_move_a2ILb0EPNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt16ostream_iteratorIS5_cS3_EET1_T0_SA_S9_: argument 0"}
!125 = distinct !{!125, !"_ZSt14__copy_move_a2ILb0EPNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt16ostream_iteratorIS5_cS3_EET1_T0_SA_S9_"}
!126 = distinct !{!126, !127, !"_ZSt14__copy_move_a1ILb0EPNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt16ostream_iteratorIS5_cS3_EET1_T0_SA_S9_: argument 0"}
!127 = distinct !{!127, !"_ZSt14__copy_move_a1ILb0EPNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt16ostream_iteratorIS5_cS3_EET1_T0_SA_S9_"}
!128 = !{!122}
