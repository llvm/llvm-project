; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Misc-C++/bigfib.cpp'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Misc-C++/bigfib.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

module asm ".globl _ZSt21ios_base_library_initv"

%"class.std::basic_ostream" = type { ptr, %"class.std::basic_ios" }
%"class.std::basic_ios" = type { %"class.std::ios_base", ptr, i8, i8, ptr, ptr, ptr, ptr }
%"class.std::ios_base" = type { ptr, i64, i64, i32, i32, i32, ptr, %"struct.std::ios_base::_Words", [8 x %"struct.std::ios_base::_Words"], i32, ptr, %"class.std::locale" }
%"struct.std::ios_base::_Words" = type { ptr, i64 }
%"class.std::locale" = type { ptr }
%class.BigInt = type { %"class.std::vector" }
%"class.std::vector" = type { %"struct.std::_Vector_base" }
%"struct.std::_Vector_base" = type { %"struct.std::_Vector_base<unsigned long, std::allocator<unsigned long>>::_Vector_impl" }
%"struct.std::_Vector_base<unsigned long, std::allocator<unsigned long>>::_Vector_impl" = type { %"struct.std::_Vector_base<unsigned long, std::allocator<unsigned long>>::_Vector_impl_data" }
%"struct.std::_Vector_base<unsigned long, std::allocator<unsigned long>>::_Vector_impl_data" = type { ptr, ptr, ptr }
%"class.std::__cxx11::basic_ostringstream" = type { %"class.std::basic_ostream.base", %"class.std::__cxx11::basic_stringbuf", %"class.std::basic_ios" }
%"class.std::basic_ostream.base" = type { ptr }
%"class.std::__cxx11::basic_stringbuf" = type { %"class.std::basic_streambuf", i32, %"class.std::__cxx11::basic_string" }
%"class.std::basic_streambuf" = type { ptr, ptr, ptr, ptr, ptr, ptr, ptr, %"class.std::locale" }
%"class.std::__cxx11::basic_string" = type { %"struct.std::__cxx11::basic_string<char>::_Alloc_hider", i64, %union.anon }
%"struct.std::__cxx11::basic_string<char>::_Alloc_hider" = type { ptr }
%union.anon = type { i64, [8 x i8] }
%class.Fibonacci = type { %"class.std::vector.0" }
%"class.std::vector.0" = type { %"struct.std::_Vector_base.1" }
%"struct.std::_Vector_base.1" = type { %"struct.std::_Vector_base<BigInt, std::allocator<BigInt>>::_Vector_impl" }
%"struct.std::_Vector_base<BigInt, std::allocator<BigInt>>::_Vector_impl" = type { %"struct.std::_Vector_base<BigInt, std::allocator<BigInt>>::_Vector_impl_data" }
%"struct.std::_Vector_base<BigInt, std::allocator<BigInt>>::_Vector_impl_data" = type { ptr, ptr, ptr }

$_ZN6BigIntC2ES_S_ = comdat any

$_ZN9FibonacciD2Ev = comdat any

$_ZNSt6vectorImSaImEE17_M_default_appendEm = comdat any

$_ZNSt6vectorI6BigIntSaIS0_EED2Ev = comdat any

@.str = private unnamed_addr constant [6 x i8] c"Fib [\00", align 1
@.str.1 = private unnamed_addr constant [5 x i8] c"] = \00", align 1
@.str.2 = private unnamed_addr constant [2 x i8] c"\0A\00", align 1
@_ZSt4cout = external global %"class.std::basic_ostream", align 8
@_ZN6BigInt6head_sE = dso_local local_unnamed_addr global i64 0, align 8
@.str.3 = private unnamed_addr constant [7 x i8] c"bigfib\00", align 1
@_ZSt4cerr = external global %"class.std::basic_ostream", align 8
@.str.4 = private unnamed_addr constant [9 x i8] c"USAGE : \00", align 1
@.str.5 = private unnamed_addr constant [3 x i8] c"  \00", align 1
@.str.6 = private unnamed_addr constant [2 x i8] c" \00", align 1
@.str.7 = private unnamed_addr constant [4 x i8] c"all\00", align 1
@.str.8 = private unnamed_addr constant [41 x i8] c" <N>              ---> Fibonacci [0 - N]\00", align 1
@.str.9 = private unnamed_addr constant [3 x i8] c"th\00", align 1
@.str.10 = private unnamed_addr constant [37 x i8] c" <N>              ---> Fibonacci [N]\00", align 1
@.str.11 = private unnamed_addr constant [5 x i8] c"some\00", align 1
@.str.12 = private unnamed_addr constant [59 x i8] c" <N1> [<N2> ...]  ---> Fibonacci [N1], Fibonacci [N2], ...\00", align 1
@.str.13 = private unnamed_addr constant [5 x i8] c"rand\00", align 1
@.str.14 = private unnamed_addr constant [68 x i8] c" <K>  [<M>]       ---> K random Fibonacci numbers ( < M; Default = \00", align 1
@.str.15 = private unnamed_addr constant [3 x i8] c" )\00", align 1
@.str.16 = private unnamed_addr constant [26 x i8] c"vector::_M_realloc_append\00", align 1
@.str.17 = private unnamed_addr constant [26 x i8] c"vector::_M_default_append\00", align 1
@_ZTTNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEEE = external unnamed_addr constant [4 x ptr], align 8
@_ZTVNSt7__cxx1115basic_stringbufIcSt11char_traitsIcESaIcEEE = external unnamed_addr constant { [16 x ptr] }, align 8
@_ZTVSt15basic_streambufIcSt11char_traitsIcEE = external unnamed_addr constant { [16 x ptr] }, align 8
@.str.19 = private unnamed_addr constant [50 x i8] c"basic_string: construction from null is not valid\00", align 1

; Function Attrs: mustprogress uwtable
define dso_local void @_ZN9Fibonacci10get_numberEj(ptr dead_on_unwind noalias writable writeonly sret(%class.BigInt) align 8 captures(none) %0, ptr noundef nonnull align 8 dereferenceable(24) %1, i32 noundef %2) local_unnamed_addr #0 personality ptr @__gxx_personality_v0 {
  %4 = alloca %class.BigInt, align 16
  %5 = alloca %class.BigInt, align 8
  %6 = alloca %class.BigInt, align 8
  %7 = add i32 %2, 1
  %8 = zext i32 %7 to i64
  %9 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %10 = load ptr, ptr %9, align 8, !tbaa !6
  %11 = load ptr, ptr %1, align 8, !tbaa !12
  %12 = ptrtoint ptr %10 to i64
  %13 = ptrtoint ptr %11 to i64
  %14 = sub i64 %12, %13
  %15 = sdiv exact i64 %14, 24
  %16 = icmp ult i64 %15, %8
  %17 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %18 = load ptr, ptr %17, align 8, !tbaa !13
  br i1 %16, label %19, label %49

19:                                               ; preds = %3
  %20 = ptrtoint ptr %18 to i64
  %21 = sub i64 %20, %13
  %22 = mul nuw nsw i64 %8, 24
  %23 = tail call noalias noundef nonnull ptr @_Znwm(i64 noundef %22) #17
  %24 = icmp eq ptr %11, %18
  br i1 %24, label %37, label %25

25:                                               ; preds = %19, %25
  %26 = phi ptr [ %33, %25 ], [ %23, %19 ]
  %27 = phi ptr [ %32, %25 ], [ %11, %19 ]
  tail call void @llvm.experimental.noalias.scope.decl(metadata !14)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !17)
  %28 = load <2 x ptr>, ptr %27, align 8, !tbaa !19, !alias.scope !17, !noalias !14
  store <2 x ptr> %28, ptr %26, align 8, !tbaa !19, !alias.scope !14, !noalias !17
  %29 = getelementptr inbounds nuw i8, ptr %26, i64 16
  %30 = getelementptr inbounds nuw i8, ptr %27, i64 16
  %31 = load ptr, ptr %30, align 8, !tbaa !21, !alias.scope !17, !noalias !14
  store ptr %31, ptr %29, align 8, !tbaa !21, !alias.scope !14, !noalias !17
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(24) %27, i8 0, i64 24, i1 false), !alias.scope !17, !noalias !14
  %32 = getelementptr inbounds nuw i8, ptr %27, i64 24
  %33 = getelementptr inbounds nuw i8, ptr %26, i64 24
  %34 = icmp eq ptr %32, %18
  br i1 %34, label %35, label %25, !llvm.loop !23

35:                                               ; preds = %25
  %36 = load ptr, ptr %1, align 8, !tbaa !12
  br label %37

37:                                               ; preds = %35, %19
  %38 = phi ptr [ %36, %35 ], [ %11, %19 ]
  %39 = icmp eq ptr %38, null
  br i1 %39, label %45, label %40

40:                                               ; preds = %37
  %41 = load ptr, ptr %9, align 8, !tbaa !6
  %42 = ptrtoint ptr %41 to i64
  %43 = ptrtoint ptr %38 to i64
  %44 = sub i64 %42, %43
  tail call void @_ZdlPvm(ptr noundef nonnull %38, i64 noundef %44) #18
  br label %45

45:                                               ; preds = %40, %37
  store ptr %23, ptr %1, align 8, !tbaa !12
  %46 = getelementptr inbounds nuw i8, ptr %23, i64 %21
  store ptr %46, ptr %17, align 8, !tbaa !13
  %47 = getelementptr inbounds nuw %class.BigInt, ptr %23, i64 %8
  store ptr %47, ptr %9, align 8, !tbaa !6
  %48 = ptrtoint ptr %23 to i64
  br label %49

49:                                               ; preds = %3, %45
  %50 = phi i64 [ %48, %45 ], [ %13, %3 ]
  %51 = phi ptr [ %23, %45 ], [ %11, %3 ]
  %52 = phi ptr [ %46, %45 ], [ %18, %3 ]
  %53 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %54 = ptrtoint ptr %52 to i64
  %55 = sub i64 %54, %50
  %56 = sdiv exact i64 %55, 24
  %57 = trunc i64 %56 to i32
  %58 = icmp ult i32 %2, %57
  br i1 %58, label %65, label %59

59:                                               ; preds = %49
  %60 = getelementptr inbounds nuw i8, ptr %4, i64 16
  %61 = getelementptr inbounds nuw i8, ptr %6, i64 16
  %62 = getelementptr inbounds nuw i8, ptr %5, i64 16
  br label %99

63:                                               ; preds = %367
  %64 = load ptr, ptr %1, align 8, !tbaa !12
  br label %65

65:                                               ; preds = %63, %49
  %66 = phi ptr [ %64, %63 ], [ %51, %49 ]
  %67 = zext i32 %2 to i64
  %68 = getelementptr inbounds nuw %class.BigInt, ptr %66, i64 %67
  %69 = getelementptr inbounds nuw i8, ptr %68, i64 8
  %70 = load ptr, ptr %69, align 8, !tbaa !25
  %71 = load ptr, ptr %68, align 8, !tbaa !26
  %72 = ptrtoint ptr %70 to i64
  %73 = ptrtoint ptr %71 to i64
  %74 = sub i64 %72, %73
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(24) %0, i8 0, i64 24, i1 false)
  %75 = icmp eq ptr %70, %71
  br i1 %75, label %81, label %76

76:                                               ; preds = %65
  %77 = icmp ugt i64 %74, 9223372036854775800
  br i1 %77, label %78, label %79, !prof !27

78:                                               ; preds = %76
  call void @_ZSt28__throw_bad_array_new_lengthv() #19
  unreachable

79:                                               ; preds = %76
  %80 = call noalias noundef nonnull ptr @_Znwm(i64 noundef %74) #17
  br label %81

81:                                               ; preds = %79, %65
  %82 = phi ptr [ null, %65 ], [ %80, %79 ]
  store ptr %82, ptr %0, align 8, !tbaa !26
  %83 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store ptr %82, ptr %83, align 8, !tbaa !25
  %84 = getelementptr inbounds nuw i8, ptr %82, i64 %74
  %85 = getelementptr inbounds nuw i8, ptr %0, i64 16
  store ptr %84, ptr %85, align 8, !tbaa !21
  %86 = load ptr, ptr %68, align 8, !tbaa !19
  %87 = ptrtoint ptr %86 to i64
  %88 = load ptr, ptr %69, align 8, !tbaa !19
  %89 = ptrtoint ptr %88 to i64
  %90 = sub i64 %89, %87
  %91 = icmp sgt i64 %90, 8
  br i1 %91, label %92, label %93, !prof !28

92:                                               ; preds = %81
  call void @llvm.memmove.p0.p0.i64(ptr align 8 %82, ptr align 8 %86, i64 %90, i1 false)
  br label %97

93:                                               ; preds = %81
  %94 = icmp eq i64 %90, 8
  br i1 %94, label %95, label %97

95:                                               ; preds = %93
  %96 = load i64, ptr %86, align 8, !tbaa !29
  store i64 %96, ptr %82, align 8, !tbaa !29
  br label %97

97:                                               ; preds = %92, %93, %95
  %98 = getelementptr inbounds i8, ptr %82, i64 %90
  store ptr %98, ptr %83, align 8, !tbaa !25
  ret void

99:                                               ; preds = %59, %367
  %100 = phi i32 [ %57, %59 ], [ %368, %367 ]
  switch i32 %100, label %250 [
    i32 0, label %101
    i32 1, label %160
  ]

101:                                              ; preds = %99
  %102 = call noalias noundef nonnull dereferenceable(8) ptr @_Znwm(i64 noundef 8) #17
  store i64 0, ptr %102, align 8, !tbaa !29
  %103 = getelementptr inbounds nuw i8, ptr %102, i64 8
  %104 = load ptr, ptr %53, align 8, !tbaa !13
  %105 = load ptr, ptr %9, align 8, !tbaa !6
  %106 = icmp eq ptr %104, %105
  br i1 %106, label %111, label %107

107:                                              ; preds = %101
  store ptr %102, ptr %104, align 8, !tbaa !26
  %108 = getelementptr inbounds nuw i8, ptr %104, i64 8
  store ptr %103, ptr %108, align 8, !tbaa !25
  %109 = getelementptr inbounds nuw i8, ptr %104, i64 16
  store ptr %103, ptr %109, align 8, !tbaa !21
  %110 = getelementptr inbounds nuw i8, ptr %104, i64 24
  store ptr %110, ptr %53, align 8, !tbaa !13
  br label %367

111:                                              ; preds = %101
  %112 = load ptr, ptr %1, align 8, !tbaa !12
  %113 = ptrtoint ptr %104 to i64
  %114 = ptrtoint ptr %112 to i64
  %115 = sub i64 %113, %114
  %116 = icmp eq i64 %115, 9223372036854775800
  br i1 %116, label %117, label %119

117:                                              ; preds = %111
  invoke void @_ZSt20__throw_length_errorPKc(ptr noundef nonnull @.str.16) #20
          to label %118 unwind label %156

118:                                              ; preds = %117
  unreachable

119:                                              ; preds = %111
  %120 = sdiv exact i64 %115, 24
  %121 = call i64 @llvm.umax.i64(i64 %120, i64 1)
  %122 = add nsw i64 %121, %120
  %123 = icmp ult i64 %122, %120
  %124 = call i64 @llvm.umin.i64(i64 %122, i64 384307168202282325)
  %125 = select i1 %123, i64 384307168202282325, i64 %124
  %126 = icmp ne i64 %125, 0
  call void @llvm.assume(i1 %126)
  %127 = mul nuw nsw i64 %125, 24
  %128 = invoke noalias noundef nonnull ptr @_Znwm(i64 noundef %127) #17
          to label %129 unwind label %154

129:                                              ; preds = %119
  %130 = getelementptr inbounds nuw i8, ptr %128, i64 %115
  store ptr %102, ptr %130, align 8, !tbaa !26
  %131 = getelementptr inbounds nuw i8, ptr %130, i64 8
  store ptr %103, ptr %131, align 8, !tbaa !25
  %132 = getelementptr inbounds nuw i8, ptr %130, i64 16
  store ptr %103, ptr %132, align 8, !tbaa !21
  %133 = icmp eq ptr %112, %104
  br i1 %133, label %144, label %134

134:                                              ; preds = %129, %134
  %135 = phi ptr [ %142, %134 ], [ %128, %129 ]
  %136 = phi ptr [ %141, %134 ], [ %112, %129 ]
  call void @llvm.experimental.noalias.scope.decl(metadata !31)
  call void @llvm.experimental.noalias.scope.decl(metadata !34)
  %137 = load <2 x ptr>, ptr %136, align 8, !tbaa !19, !alias.scope !34, !noalias !31
  store <2 x ptr> %137, ptr %135, align 8, !tbaa !19, !alias.scope !31, !noalias !34
  %138 = getelementptr inbounds nuw i8, ptr %135, i64 16
  %139 = getelementptr inbounds nuw i8, ptr %136, i64 16
  %140 = load ptr, ptr %139, align 8, !tbaa !21, !alias.scope !34, !noalias !31
  store ptr %140, ptr %138, align 8, !tbaa !21, !alias.scope !31, !noalias !34
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(24) %136, i8 0, i64 24, i1 false), !alias.scope !34, !noalias !31
  %141 = getelementptr inbounds nuw i8, ptr %136, i64 24
  %142 = getelementptr inbounds nuw i8, ptr %135, i64 24
  %143 = icmp eq ptr %141, %104
  br i1 %143, label %144, label %134, !llvm.loop !23

144:                                              ; preds = %134, %129
  %145 = phi ptr [ %128, %129 ], [ %142, %134 ]
  %146 = icmp eq ptr %112, null
  br i1 %146, label %151, label %147

147:                                              ; preds = %144
  %148 = load ptr, ptr %9, align 8, !tbaa !6
  %149 = ptrtoint ptr %148 to i64
  %150 = sub i64 %149, %114
  call void @_ZdlPvm(ptr noundef nonnull %112, i64 noundef %150) #18
  br label %151

151:                                              ; preds = %147, %144
  %152 = getelementptr inbounds nuw i8, ptr %145, i64 24
  store ptr %128, ptr %1, align 8, !tbaa !12
  store ptr %152, ptr %53, align 8, !tbaa !13
  %153 = getelementptr inbounds nuw %class.BigInt, ptr %128, i64 %125
  store ptr %153, ptr %9, align 8, !tbaa !6
  br label %367

154:                                              ; preds = %119
  %155 = landingpad { ptr, i32 }
          cleanup
  br label %158

156:                                              ; preds = %117
  %157 = landingpad { ptr, i32 }
          cleanup
  br label %158

158:                                              ; preds = %156, %154
  %159 = phi { ptr, i32 } [ %155, %154 ], [ %157, %156 ]
  call void @_ZdlPvm(ptr noundef nonnull %102, i64 noundef 8) #18
  br label %370

160:                                              ; preds = %99
  %161 = load ptr, ptr %1, align 8, !tbaa !36
  %162 = load ptr, ptr %53, align 8, !tbaa !36
  %163 = icmp eq ptr %161, %162
  br i1 %163, label %166, label %164

164:                                              ; preds = %160
  %165 = load ptr, ptr %9, align 8, !tbaa !6
  br label %188

166:                                              ; preds = %160
  %167 = call noalias noundef nonnull dereferenceable(8) ptr @_Znwm(i64 noundef 8) #17
  store i64 0, ptr %167, align 8, !tbaa !29
  %168 = getelementptr inbounds nuw i8, ptr %167, i64 8
  %169 = load ptr, ptr %9, align 8, !tbaa !6
  %170 = icmp eq ptr %161, %169
  br i1 %170, label %175, label %171

171:                                              ; preds = %166
  store ptr %167, ptr %162, align 8, !tbaa !26
  %172 = getelementptr inbounds nuw i8, ptr %162, i64 8
  store ptr %168, ptr %172, align 8, !tbaa !25
  %173 = getelementptr inbounds nuw i8, ptr %162, i64 16
  store ptr %168, ptr %173, align 8, !tbaa !21
  %174 = getelementptr inbounds nuw i8, ptr %162, i64 24
  store ptr %174, ptr %53, align 8, !tbaa !13
  br label %188

175:                                              ; preds = %166
  %176 = invoke noalias noundef nonnull dereferenceable(24) ptr @_Znwm(i64 noundef 24) #17
          to label %177 unwind label %186

177:                                              ; preds = %175
  store ptr %167, ptr %176, align 8, !tbaa !26
  %178 = getelementptr inbounds nuw i8, ptr %176, i64 8
  store ptr %168, ptr %178, align 8, !tbaa !25
  %179 = getelementptr inbounds nuw i8, ptr %176, i64 16
  store ptr %168, ptr %179, align 8, !tbaa !21
  %180 = icmp eq ptr %161, null
  br i1 %180, label %182, label %181

181:                                              ; preds = %177
  call void @_ZdlPvm(ptr noundef nonnull %161, i64 noundef 0) #18
  br label %182

182:                                              ; preds = %177, %181
  %183 = getelementptr inbounds nuw i8, ptr %176, i64 24
  store ptr %176, ptr %1, align 8, !tbaa !12
  store ptr %183, ptr %53, align 8, !tbaa !13
  store ptr %183, ptr %9, align 8, !tbaa !6
  %184 = call noalias noundef nonnull dereferenceable(8) ptr @_Znwm(i64 noundef 8) #17
  store i64 1, ptr %184, align 8, !tbaa !29
  %185 = getelementptr inbounds nuw i8, ptr %184, i64 8
  br label %198

186:                                              ; preds = %175
  %187 = landingpad { ptr, i32 }
          cleanup
  call void @_ZdlPvm(ptr noundef nonnull %167, i64 noundef 8) #18
  br label %370

188:                                              ; preds = %164, %171
  %189 = phi ptr [ %169, %171 ], [ %165, %164 ]
  %190 = phi ptr [ %174, %171 ], [ %162, %164 ]
  %191 = call noalias noundef nonnull dereferenceable(8) ptr @_Znwm(i64 noundef 8) #17
  store i64 1, ptr %191, align 8, !tbaa !29
  %192 = getelementptr inbounds nuw i8, ptr %191, i64 8
  %193 = icmp eq ptr %190, %189
  br i1 %193, label %198, label %194

194:                                              ; preds = %188
  store ptr %191, ptr %190, align 8, !tbaa !26
  %195 = getelementptr inbounds nuw i8, ptr %190, i64 8
  store ptr %192, ptr %195, align 8, !tbaa !25
  %196 = getelementptr inbounds nuw i8, ptr %190, i64 16
  store ptr %192, ptr %196, align 8, !tbaa !21
  %197 = getelementptr inbounds nuw i8, ptr %190, i64 24
  store ptr %197, ptr %53, align 8, !tbaa !13
  br label %367

198:                                              ; preds = %182, %188
  %199 = phi ptr [ %185, %182 ], [ %192, %188 ]
  %200 = phi ptr [ %184, %182 ], [ %191, %188 ]
  %201 = phi ptr [ %183, %182 ], [ %189, %188 ]
  %202 = load ptr, ptr %1, align 8, !tbaa !12
  %203 = ptrtoint ptr %201 to i64
  %204 = ptrtoint ptr %202 to i64
  %205 = sub i64 %203, %204
  %206 = icmp eq i64 %205, 9223372036854775800
  br i1 %206, label %207, label %209

207:                                              ; preds = %198
  invoke void @_ZSt20__throw_length_errorPKc(ptr noundef nonnull @.str.16) #20
          to label %208 unwind label %246

208:                                              ; preds = %207
  unreachable

209:                                              ; preds = %198
  %210 = sdiv exact i64 %205, 24
  %211 = call i64 @llvm.umax.i64(i64 %210, i64 1)
  %212 = add nsw i64 %211, %210
  %213 = icmp ult i64 %212, %210
  %214 = call i64 @llvm.umin.i64(i64 %212, i64 384307168202282325)
  %215 = select i1 %213, i64 384307168202282325, i64 %214
  %216 = icmp ne i64 %215, 0
  call void @llvm.assume(i1 %216)
  %217 = mul nuw nsw i64 %215, 24
  %218 = invoke noalias noundef nonnull ptr @_Znwm(i64 noundef %217) #17
          to label %219 unwind label %244

219:                                              ; preds = %209
  %220 = getelementptr inbounds nuw i8, ptr %218, i64 %205
  store ptr %200, ptr %220, align 8, !tbaa !26
  %221 = getelementptr inbounds nuw i8, ptr %220, i64 8
  store ptr %199, ptr %221, align 8, !tbaa !25
  %222 = getelementptr inbounds nuw i8, ptr %220, i64 16
  store ptr %199, ptr %222, align 8, !tbaa !21
  %223 = icmp eq ptr %202, %201
  br i1 %223, label %234, label %224

224:                                              ; preds = %219, %224
  %225 = phi ptr [ %232, %224 ], [ %218, %219 ]
  %226 = phi ptr [ %231, %224 ], [ %202, %219 ]
  call void @llvm.experimental.noalias.scope.decl(metadata !37)
  call void @llvm.experimental.noalias.scope.decl(metadata !40)
  %227 = load <2 x ptr>, ptr %226, align 8, !tbaa !19, !alias.scope !40, !noalias !37
  store <2 x ptr> %227, ptr %225, align 8, !tbaa !19, !alias.scope !37, !noalias !40
  %228 = getelementptr inbounds nuw i8, ptr %225, i64 16
  %229 = getelementptr inbounds nuw i8, ptr %226, i64 16
  %230 = load ptr, ptr %229, align 8, !tbaa !21, !alias.scope !40, !noalias !37
  store ptr %230, ptr %228, align 8, !tbaa !21, !alias.scope !37, !noalias !40
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(24) %226, i8 0, i64 24, i1 false), !alias.scope !40, !noalias !37
  %231 = getelementptr inbounds nuw i8, ptr %226, i64 24
  %232 = getelementptr inbounds nuw i8, ptr %225, i64 24
  %233 = icmp eq ptr %231, %201
  br i1 %233, label %234, label %224, !llvm.loop !23

234:                                              ; preds = %224, %219
  %235 = phi ptr [ %218, %219 ], [ %232, %224 ]
  %236 = icmp eq ptr %202, null
  br i1 %236, label %241, label %237

237:                                              ; preds = %234
  %238 = load ptr, ptr %9, align 8, !tbaa !6
  %239 = ptrtoint ptr %238 to i64
  %240 = sub i64 %239, %204
  call void @_ZdlPvm(ptr noundef nonnull %202, i64 noundef %240) #18
  br label %241

241:                                              ; preds = %237, %234
  %242 = getelementptr inbounds nuw i8, ptr %235, i64 24
  store ptr %218, ptr %1, align 8, !tbaa !12
  store ptr %242, ptr %53, align 8, !tbaa !13
  %243 = getelementptr inbounds nuw %class.BigInt, ptr %218, i64 %215
  store ptr %243, ptr %9, align 8, !tbaa !6
  br label %367

244:                                              ; preds = %209
  %245 = landingpad { ptr, i32 }
          cleanup
  br label %248

246:                                              ; preds = %207
  %247 = landingpad { ptr, i32 }
          cleanup
  br label %248

248:                                              ; preds = %246, %244
  %249 = phi { ptr, i32 } [ %245, %244 ], [ %247, %246 ]
  call void @_ZdlPvm(ptr noundef nonnull %200, i64 noundef 8) #18
  br label %370

250:                                              ; preds = %99
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #21
  %251 = add i32 %100, -2
  call void @_ZN9Fibonacci10get_numberEj(ptr dead_on_unwind nonnull writable sret(%class.BigInt) align 8 %5, ptr noundef nonnull align 8 dereferenceable(24) %1, i32 noundef %251)
  %252 = add i32 %100, -1
  invoke void @_ZN9Fibonacci10get_numberEj(ptr dead_on_unwind nonnull writable sret(%class.BigInt) align 8 %6, ptr noundef nonnull align 8 dereferenceable(24) %1, i32 noundef %252)
          to label %253 unwind label %331

253:                                              ; preds = %250
  invoke void @_ZN6BigIntC2ES_S_(ptr noundef nonnull align 8 dereferenceable(24) %4, ptr noundef nonnull %5, ptr noundef nonnull %6)
          to label %254 unwind label %333

254:                                              ; preds = %253
  %255 = load ptr, ptr %53, align 8, !tbaa !13
  %256 = load ptr, ptr %9, align 8, !tbaa !6
  %257 = icmp eq ptr %255, %256
  br i1 %257, label %263, label %258

258:                                              ; preds = %254
  %259 = load <2 x ptr>, ptr %4, align 16, !tbaa !19
  store <2 x ptr> %259, ptr %255, align 8, !tbaa !19
  %260 = getelementptr inbounds nuw i8, ptr %255, i64 16
  %261 = load ptr, ptr %60, align 16, !tbaa !21
  store ptr %261, ptr %260, align 8, !tbaa !21
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 16 dereferenceable(24) %4, i8 0, i64 24, i1 false)
  %262 = getelementptr inbounds nuw i8, ptr %255, i64 24
  store ptr %262, ptr %53, align 8, !tbaa !13
  br label %314

263:                                              ; preds = %254
  %264 = load ptr, ptr %1, align 8, !tbaa !12
  %265 = ptrtoint ptr %255 to i64
  %266 = ptrtoint ptr %264 to i64
  %267 = sub i64 %265, %266
  %268 = icmp eq i64 %267, 9223372036854775800
  br i1 %268, label %269, label %271

269:                                              ; preds = %263
  invoke void @_ZSt20__throw_length_errorPKc(ptr noundef nonnull @.str.16) #20
          to label %270 unwind label %337

270:                                              ; preds = %269
  unreachable

271:                                              ; preds = %263
  %272 = sdiv exact i64 %267, 24
  %273 = call i64 @llvm.umax.i64(i64 %272, i64 1)
  %274 = add nsw i64 %273, %272
  %275 = icmp ult i64 %274, %272
  %276 = call i64 @llvm.umin.i64(i64 %274, i64 384307168202282325)
  %277 = select i1 %275, i64 384307168202282325, i64 %276
  %278 = icmp ne i64 %277, 0
  call void @llvm.assume(i1 %278)
  %279 = mul nuw nsw i64 %277, 24
  %280 = invoke noalias noundef nonnull ptr @_Znwm(i64 noundef %279) #17
          to label %281 unwind label %335

281:                                              ; preds = %271
  %282 = getelementptr inbounds nuw i8, ptr %280, i64 %267
  %283 = load <2 x ptr>, ptr %4, align 16, !tbaa !19
  store <2 x ptr> %283, ptr %282, align 8, !tbaa !19
  %284 = getelementptr inbounds nuw i8, ptr %282, i64 16
  %285 = load ptr, ptr %60, align 16, !tbaa !21
  store ptr %285, ptr %284, align 8, !tbaa !21
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 16 dereferenceable(24) %4, i8 0, i64 24, i1 false)
  %286 = icmp eq ptr %264, %255
  br i1 %286, label %297, label %287

287:                                              ; preds = %281, %287
  %288 = phi ptr [ %295, %287 ], [ %280, %281 ]
  %289 = phi ptr [ %294, %287 ], [ %264, %281 ]
  call void @llvm.experimental.noalias.scope.decl(metadata !42)
  call void @llvm.experimental.noalias.scope.decl(metadata !45)
  %290 = load <2 x ptr>, ptr %289, align 8, !tbaa !19, !alias.scope !45, !noalias !42
  store <2 x ptr> %290, ptr %288, align 8, !tbaa !19, !alias.scope !42, !noalias !45
  %291 = getelementptr inbounds nuw i8, ptr %288, i64 16
  %292 = getelementptr inbounds nuw i8, ptr %289, i64 16
  %293 = load ptr, ptr %292, align 8, !tbaa !21, !alias.scope !45, !noalias !42
  store ptr %293, ptr %291, align 8, !tbaa !21, !alias.scope !42, !noalias !45
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(24) %289, i8 0, i64 24, i1 false), !alias.scope !45, !noalias !42
  %294 = getelementptr inbounds nuw i8, ptr %289, i64 24
  %295 = getelementptr inbounds nuw i8, ptr %288, i64 24
  %296 = icmp eq ptr %294, %255
  br i1 %296, label %297, label %287, !llvm.loop !23

297:                                              ; preds = %287, %281
  %298 = phi ptr [ %280, %281 ], [ %295, %287 ]
  %299 = icmp eq ptr %264, null
  br i1 %299, label %304, label %300

300:                                              ; preds = %297
  %301 = load ptr, ptr %9, align 8, !tbaa !6
  %302 = ptrtoint ptr %301 to i64
  %303 = sub i64 %302, %266
  call void @_ZdlPvm(ptr noundef nonnull %264, i64 noundef %303) #18
  br label %304

304:                                              ; preds = %297, %300
  %305 = getelementptr inbounds nuw i8, ptr %298, i64 24
  store ptr %280, ptr %1, align 8, !tbaa !12
  store ptr %305, ptr %53, align 8, !tbaa !13
  %306 = getelementptr inbounds nuw %class.BigInt, ptr %280, i64 %277
  store ptr %306, ptr %9, align 8, !tbaa !6
  %307 = load ptr, ptr %4, align 16, !tbaa !26
  %308 = icmp eq ptr %307, null
  br i1 %308, label %314, label %309

309:                                              ; preds = %304
  %310 = load ptr, ptr %60, align 16, !tbaa !21
  %311 = ptrtoint ptr %310 to i64
  %312 = ptrtoint ptr %307 to i64
  %313 = sub i64 %311, %312
  call void @_ZdlPvm(ptr noundef nonnull %307, i64 noundef %313) #18
  br label %314

314:                                              ; preds = %258, %304, %309
  %315 = load ptr, ptr %6, align 8, !tbaa !26
  %316 = icmp eq ptr %315, null
  br i1 %316, label %322, label %317

317:                                              ; preds = %314
  %318 = load ptr, ptr %61, align 8, !tbaa !21
  %319 = ptrtoint ptr %318 to i64
  %320 = ptrtoint ptr %315 to i64
  %321 = sub i64 %319, %320
  call void @_ZdlPvm(ptr noundef nonnull %315, i64 noundef %321) #18
  br label %322

322:                                              ; preds = %314, %317
  %323 = load ptr, ptr %5, align 8, !tbaa !26
  %324 = icmp eq ptr %323, null
  br i1 %324, label %330, label %325

325:                                              ; preds = %322
  %326 = load ptr, ptr %62, align 8, !tbaa !21
  %327 = ptrtoint ptr %326 to i64
  %328 = ptrtoint ptr %323 to i64
  %329 = sub i64 %327, %328
  call void @_ZdlPvm(ptr noundef nonnull %323, i64 noundef %329) #18
  br label %330

330:                                              ; preds = %322, %325
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #21
  br label %367

331:                                              ; preds = %250
  %332 = landingpad { ptr, i32 }
          cleanup
  br label %357

333:                                              ; preds = %253
  %334 = landingpad { ptr, i32 }
          cleanup
  br label %348

335:                                              ; preds = %271
  %336 = landingpad { ptr, i32 }
          cleanup
  br label %339

337:                                              ; preds = %269
  %338 = landingpad { ptr, i32 }
          cleanup
  br label %339

339:                                              ; preds = %337, %335
  %340 = phi { ptr, i32 } [ %336, %335 ], [ %338, %337 ]
  %341 = load ptr, ptr %4, align 16, !tbaa !26
  %342 = icmp eq ptr %341, null
  br i1 %342, label %348, label %343

343:                                              ; preds = %339
  %344 = load ptr, ptr %60, align 16, !tbaa !21
  %345 = ptrtoint ptr %344 to i64
  %346 = ptrtoint ptr %341 to i64
  %347 = sub i64 %345, %346
  call void @_ZdlPvm(ptr noundef nonnull %341, i64 noundef %347) #18
  br label %348

348:                                              ; preds = %343, %339, %333
  %349 = phi { ptr, i32 } [ %334, %333 ], [ %340, %339 ], [ %340, %343 ]
  %350 = load ptr, ptr %6, align 8, !tbaa !26
  %351 = icmp eq ptr %350, null
  br i1 %351, label %357, label %352

352:                                              ; preds = %348
  %353 = load ptr, ptr %61, align 8, !tbaa !21
  %354 = ptrtoint ptr %353 to i64
  %355 = ptrtoint ptr %350 to i64
  %356 = sub i64 %354, %355
  call void @_ZdlPvm(ptr noundef nonnull %350, i64 noundef %356) #18
  br label %357

357:                                              ; preds = %352, %348, %331
  %358 = phi { ptr, i32 } [ %332, %331 ], [ %349, %348 ], [ %349, %352 ]
  %359 = load ptr, ptr %5, align 8, !tbaa !26
  %360 = icmp eq ptr %359, null
  br i1 %360, label %366, label %361

361:                                              ; preds = %357
  %362 = load ptr, ptr %62, align 8, !tbaa !21
  %363 = ptrtoint ptr %362 to i64
  %364 = ptrtoint ptr %359 to i64
  %365 = sub i64 %363, %364
  call void @_ZdlPvm(ptr noundef nonnull %359, i64 noundef %365) #18
  br label %366

366:                                              ; preds = %357, %361
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #21
  br label %370

367:                                              ; preds = %241, %194, %151, %107, %330
  %368 = add i32 %100, 1
  %369 = icmp ugt i32 %368, %2
  br i1 %369, label %63, label %99, !llvm.loop !47

370:                                              ; preds = %366, %248, %186, %158
  %371 = phi { ptr, i32 } [ %358, %366 ], [ %159, %158 ], [ %249, %248 ], [ %187, %186 ]
  resume { ptr, i32 } %371
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

declare i32 @__gxx_personality_v0(...)

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZN6BigIntC2ES_S_(ptr noundef nonnull align 8 dereferenceable(24) %0, ptr noundef %1, ptr noundef %2) unnamed_addr #0 comdat personality ptr @__gxx_personality_v0 {
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(24) %0, i8 0, i64 24, i1 false)
  %4 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %5 = load ptr, ptr %4, align 8, !tbaa !25
  %6 = load ptr, ptr %1, align 8, !tbaa !26
  %7 = ptrtoint ptr %5 to i64
  %8 = ptrtoint ptr %6 to i64
  %9 = sub i64 %7, %8
  %10 = ashr exact i64 %9, 3
  %11 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %12 = load ptr, ptr %11, align 8, !tbaa !25
  %13 = load ptr, ptr %2, align 8, !tbaa !26
  %14 = ptrtoint ptr %12 to i64
  %15 = ptrtoint ptr %13 to i64
  %16 = sub i64 %14, %15
  %17 = ashr exact i64 %16, 3
  %18 = icmp ugt i64 %10, %17
  br i1 %18, label %30, label %19

19:                                               ; preds = %3
  %20 = icmp ugt i64 %17, %10
  br i1 %20, label %21, label %44

21:                                               ; preds = %19
  %22 = sub nuw nsw i64 %17, %10
  invoke void @_ZNSt6vectorImSaImEE17_M_default_appendEm(ptr noundef nonnull align 8 dereferenceable(24) %1, i64 noundef %22)
          to label %23 unwind label %157

23:                                               ; preds = %21
  %24 = load ptr, ptr %11, align 8, !tbaa !25
  %25 = load ptr, ptr %2, align 8, !tbaa !26
  %26 = ptrtoint ptr %24 to i64
  %27 = ptrtoint ptr %25 to i64
  %28 = sub i64 %26, %27
  %29 = ashr exact i64 %28, 3
  br label %30

30:                                               ; preds = %23, %3
  %31 = phi i64 [ %29, %23 ], [ %17, %3 ]
  %32 = phi ptr [ %25, %23 ], [ %13, %3 ]
  %33 = phi ptr [ %24, %23 ], [ %12, %3 ]
  %34 = phi i64 [ %17, %23 ], [ %10, %3 ]
  %35 = icmp ugt i64 %34, %31
  br i1 %35, label %36, label %38

36:                                               ; preds = %30
  %37 = sub nuw nsw i64 %34, %31
  invoke void @_ZNSt6vectorImSaImEE17_M_default_appendEm(ptr noundef nonnull align 8 dereferenceable(24) %2, i64 noundef %37)
          to label %44 unwind label %157

38:                                               ; preds = %30
  %39 = icmp ult i64 %34, %31
  br i1 %39, label %40, label %44

40:                                               ; preds = %38
  %41 = getelementptr inbounds nuw i64, ptr %32, i64 %34
  %42 = icmp eq ptr %33, %41
  br i1 %42, label %44, label %43

43:                                               ; preds = %40
  store ptr %41, ptr %11, align 8, !tbaa !25
  br label %44

44:                                               ; preds = %19, %43, %40, %38, %36
  %45 = phi i64 [ %34, %43 ], [ %34, %40 ], [ %34, %38 ], [ %34, %36 ], [ %17, %19 ]
  %46 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %47 = load ptr, ptr %46, align 8, !tbaa !25
  %48 = load ptr, ptr %0, align 8, !tbaa !26
  %49 = ptrtoint ptr %47 to i64
  %50 = ptrtoint ptr %48 to i64
  %51 = sub i64 %49, %50
  %52 = ashr exact i64 %51, 3
  %53 = icmp ugt i64 %45, %52
  br i1 %53, label %54, label %60

54:                                               ; preds = %44
  %55 = sub nuw nsw i64 %45, %52
  invoke void @_ZNSt6vectorImSaImEE17_M_default_appendEm(ptr noundef nonnull align 8 dereferenceable(24) %0, i64 noundef %55)
          to label %56 unwind label %157

56:                                               ; preds = %54
  %57 = load ptr, ptr %0, align 8, !tbaa !19
  %58 = load ptr, ptr %46, align 8, !tbaa !25
  %59 = ptrtoint ptr %57 to i64
  br label %66

60:                                               ; preds = %44
  %61 = icmp ult i64 %45, %52
  br i1 %61, label %62, label %66

62:                                               ; preds = %60
  %63 = getelementptr inbounds nuw i64, ptr %48, i64 %45
  %64 = icmp eq ptr %47, %63
  br i1 %64, label %66, label %65

65:                                               ; preds = %62
  store ptr %63, ptr %46, align 8, !tbaa !25
  br label %66

66:                                               ; preds = %56, %65, %62, %60
  %67 = phi i64 [ %59, %56 ], [ %50, %65 ], [ %50, %62 ], [ %50, %60 ]
  %68 = phi ptr [ %58, %56 ], [ %63, %65 ], [ %47, %62 ], [ %47, %60 ]
  %69 = phi ptr [ %57, %56 ], [ %48, %65 ], [ %48, %62 ], [ %48, %60 ]
  store i64 0, ptr @_ZN6BigInt6head_sE, align 8, !tbaa !29
  %70 = load ptr, ptr %1, align 8, !tbaa !19
  %71 = load ptr, ptr %4, align 8, !tbaa !19
  %72 = load ptr, ptr %2, align 8, !tbaa !19
  %73 = ptrtoint ptr %68 to i64
  %74 = sub i64 %73, %67
  %75 = icmp eq ptr %68, %69
  br i1 %75, label %76, label %78

76:                                               ; preds = %66
  %77 = getelementptr inbounds nuw i8, ptr null, i64 %74
  br label %92

78:                                               ; preds = %66
  %79 = icmp ugt i64 %74, 9223372036854775800
  br i1 %79, label %80, label %82, !prof !27

80:                                               ; preds = %78
  invoke void @_ZSt28__throw_bad_array_new_lengthv() #19
          to label %81 unwind label %157

81:                                               ; preds = %80
  unreachable

82:                                               ; preds = %78
  %83 = invoke noalias noundef nonnull ptr @_Znwm(i64 noundef %74) #17
          to label %84 unwind label %157

84:                                               ; preds = %82
  %85 = getelementptr inbounds nuw i8, ptr %83, i64 %74
  %86 = icmp samesign ugt i64 %74, 8
  br i1 %86, label %87, label %88, !prof !48

87:                                               ; preds = %84
  tail call void @llvm.memmove.p0.p0.i64(ptr nonnull align 8 %83, ptr align 8 %69, i64 %74, i1 false)
  br label %92

88:                                               ; preds = %84
  %89 = icmp eq i64 %74, 8
  br i1 %89, label %90, label %92

90:                                               ; preds = %88
  %91 = load i64, ptr %69, align 8, !tbaa !29
  store i64 %91, ptr %83, align 8, !tbaa !29
  br label %92

92:                                               ; preds = %90, %88, %87, %76
  %93 = phi ptr [ %85, %87 ], [ %85, %88 ], [ %85, %90 ], [ %77, %76 ]
  %94 = phi ptr [ %83, %87 ], [ %83, %88 ], [ %83, %90 ], [ null, %76 ]
  %95 = icmp eq ptr %70, %71
  br i1 %95, label %113, label %96

96:                                               ; preds = %92, %96
  %97 = phi ptr [ %111, %96 ], [ %69, %92 ]
  %98 = phi ptr [ %110, %96 ], [ %72, %92 ]
  %99 = phi ptr [ %109, %96 ], [ %70, %92 ]
  %100 = load i64, ptr %99, align 8, !tbaa !29
  %101 = load i64, ptr %98, align 8, !tbaa !29
  %102 = add i64 %101, %100
  %103 = load i64, ptr @_ZN6BigInt6head_sE, align 8, !tbaa !29
  %104 = add i64 %102, %103
  %105 = freeze i64 %104
  %106 = udiv i64 %105, 1000000000
  store i64 %106, ptr @_ZN6BigInt6head_sE, align 8, !tbaa !29
  %107 = mul i64 %106, 1000000000
  %108 = sub i64 %105, %107
  store i64 %108, ptr %97, align 8, !tbaa !29
  %109 = getelementptr inbounds nuw i8, ptr %99, i64 8
  %110 = getelementptr inbounds nuw i8, ptr %98, i64 8
  %111 = getelementptr inbounds nuw i8, ptr %97, i64 8
  %112 = icmp eq ptr %109, %71
  br i1 %112, label %113, label %96, !llvm.loop !49

113:                                              ; preds = %96, %92
  %114 = icmp eq ptr %94, null
  br i1 %114, label %119, label %115

115:                                              ; preds = %113
  %116 = ptrtoint ptr %93 to i64
  %117 = ptrtoint ptr %94 to i64
  %118 = sub i64 %116, %117
  tail call void @_ZdlPvm(ptr noundef nonnull %94, i64 noundef %118) #18
  br label %119

119:                                              ; preds = %113, %115
  %120 = load i64, ptr @_ZN6BigInt6head_sE, align 8, !tbaa !29
  %121 = icmp eq i64 %120, 0
  br i1 %121, label %161, label %122

122:                                              ; preds = %119
  %123 = load ptr, ptr %46, align 8, !tbaa !25
  %124 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %125 = load ptr, ptr %124, align 8, !tbaa !21
  %126 = icmp eq ptr %123, %125
  br i1 %126, label %129, label %127

127:                                              ; preds = %122
  store i64 %120, ptr %123, align 8, !tbaa !29
  %128 = getelementptr inbounds nuw i8, ptr %123, i64 8
  store ptr %128, ptr %46, align 8, !tbaa !25
  br label %161

129:                                              ; preds = %122
  %130 = load ptr, ptr %0, align 8, !tbaa !26
  %131 = ptrtoint ptr %123 to i64
  %132 = ptrtoint ptr %130 to i64
  %133 = sub i64 %131, %132
  %134 = icmp eq i64 %133, 9223372036854775800
  br i1 %134, label %135, label %137

135:                                              ; preds = %129
  invoke void @_ZSt20__throw_length_errorPKc(ptr noundef nonnull @.str.16) #20
          to label %136 unwind label %157

136:                                              ; preds = %135
  unreachable

137:                                              ; preds = %129
  %138 = ashr exact i64 %133, 3
  %139 = tail call i64 @llvm.umax.i64(i64 %138, i64 1)
  %140 = add nsw i64 %139, %138
  %141 = icmp ult i64 %140, %138
  %142 = tail call i64 @llvm.umin.i64(i64 %140, i64 1152921504606846975)
  %143 = select i1 %141, i64 1152921504606846975, i64 %142
  %144 = icmp ne i64 %143, 0
  tail call void @llvm.assume(i1 %144)
  %145 = shl nuw nsw i64 %143, 3
  %146 = invoke noalias noundef nonnull ptr @_Znwm(i64 noundef %145) #17
          to label %147 unwind label %157

147:                                              ; preds = %137
  %148 = getelementptr inbounds i8, ptr %146, i64 %133
  store i64 %120, ptr %148, align 8, !tbaa !29
  %149 = icmp sgt i64 %133, 0
  br i1 %149, label %150, label %151

150:                                              ; preds = %147
  tail call void @llvm.memcpy.p0.p0.i64(ptr nonnull align 8 %146, ptr align 8 %130, i64 %133, i1 false)
  br label %151

151:                                              ; preds = %150, %147
  %152 = icmp eq ptr %130, null
  br i1 %152, label %154, label %153

153:                                              ; preds = %151
  tail call void @_ZdlPvm(ptr noundef nonnull %130, i64 noundef %133) #18
  br label %154

154:                                              ; preds = %153, %151
  %155 = getelementptr inbounds nuw i8, ptr %148, i64 8
  store ptr %146, ptr %0, align 8, !tbaa !26
  store ptr %155, ptr %46, align 8, !tbaa !25
  %156 = getelementptr inbounds nuw i64, ptr %146, i64 %143
  store ptr %156, ptr %124, align 8, !tbaa !21
  br label %161

157:                                              ; preds = %137, %135, %82, %80, %54, %36, %21
  %158 = landingpad { ptr, i32 }
          cleanup
  %159 = load ptr, ptr %0, align 8, !tbaa !26
  %160 = icmp eq ptr %159, null
  br i1 %160, label %168, label %162

161:                                              ; preds = %154, %127, %119
  ret void

162:                                              ; preds = %157
  %163 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %164 = load ptr, ptr %163, align 8, !tbaa !21
  %165 = ptrtoint ptr %164 to i64
  %166 = ptrtoint ptr %159 to i64
  %167 = sub i64 %165, %166
  tail call void @_ZdlPvm(ptr noundef nonnull %159, i64 noundef %167) #18
  br label %168

168:                                              ; preds = %157, %162
  resume { ptr, i32 } %158
}

; Function Attrs: mustprogress uwtable
define dso_local void @_ZNK9Fibonacci16show_all_numbersEv(ptr noundef nonnull readonly align 8 captures(none) dereferenceable(24) %0) local_unnamed_addr #0 personality ptr @__gxx_personality_v0 {
  %2 = alloca %"class.std::__cxx11::basic_ostringstream", align 8
  %3 = alloca %"class.std::__cxx11::basic_string", align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #21
  call void @_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEEC1Ev(ptr noundef nonnull align 8 dereferenceable(112) %2)
  %4 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %5 = load ptr, ptr %4, align 8, !tbaa !13
  %6 = load ptr, ptr %0, align 8, !tbaa !12
  %7 = icmp eq ptr %5, %6
  br i1 %7, label %8, label %39

8:                                                ; preds = %105, %1
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #21
  call void @llvm.experimental.noalias.scope.decl(metadata !50)
  call void @llvm.experimental.noalias.scope.decl(metadata !53)
  %9 = getelementptr inbounds nuw i8, ptr %3, i64 16
  store ptr %9, ptr %3, align 8, !tbaa !56, !alias.scope !59
  %10 = getelementptr inbounds nuw i8, ptr %3, i64 8
  store i64 0, ptr %10, align 8, !tbaa !60, !alias.scope !59
  store i8 0, ptr %9, align 8, !tbaa !62, !alias.scope !59
  %11 = getelementptr inbounds nuw i8, ptr %2, i64 48
  %12 = load ptr, ptr %11, align 8, !tbaa !63, !noalias !59
  %13 = icmp eq ptr %12, null
  %14 = getelementptr inbounds nuw i8, ptr %2, i64 32
  %15 = load ptr, ptr %14, align 8, !noalias !59
  %16 = icmp ugt ptr %12, %15
  %17 = select i1 %16, ptr %12, ptr %15
  %18 = icmp eq ptr %17, null
  %19 = select i1 %13, i1 true, i1 %18
  br i1 %19, label %37, label %20

20:                                               ; preds = %8
  %21 = getelementptr inbounds nuw i8, ptr %2, i64 40
  %22 = load ptr, ptr %21, align 8, !tbaa !67, !noalias !59
  %23 = ptrtoint ptr %17 to i64
  %24 = ptrtoint ptr %22 to i64
  %25 = sub i64 %23, %24
  %26 = invoke noundef nonnull align 8 dereferenceable(32) ptr @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_replaceEmmPKcm(ptr noundef nonnull align 8 dereferenceable(32) %3, i64 noundef 0, i64 noundef 0, ptr noundef %22, i64 noundef %25)
          to label %121 unwind label %27

27:                                               ; preds = %37, %20
  %28 = landingpad { ptr, i32 }
          cleanup
  %29 = load ptr, ptr %3, align 8, !tbaa !68, !alias.scope !59
  %30 = icmp eq ptr %29, %9
  br i1 %30, label %31, label %34

31:                                               ; preds = %27
  %32 = load i64, ptr %10, align 8, !tbaa !60, !alias.scope !59
  %33 = icmp ult i64 %32, 16
  call void @llvm.assume(i1 %33)
  br label %165

34:                                               ; preds = %27
  %35 = load i64, ptr %9, align 8, !tbaa !62, !alias.scope !59
  %36 = add i64 %35, 1
  call void @_ZdlPvm(ptr noundef %29, i64 noundef %36) #18
  br label %165

37:                                               ; preds = %8
  %38 = getelementptr inbounds nuw i8, ptr %2, i64 80
  invoke void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_assignERKS4_(ptr noundef nonnull align 8 dereferenceable(32) %3, ptr noundef nonnull align 8 dereferenceable(32) %38)
          to label %121 unwind label %27

39:                                               ; preds = %1, %105
  %40 = phi i64 [ %107, %105 ], [ 0, %1 ]
  %41 = phi i32 [ %106, %105 ], [ 0, %1 ]
  %42 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %2, ptr noundef nonnull @.str, i64 noundef 5)
          to label %43 unwind label %117

43:                                               ; preds = %39
  %44 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo9_M_insertImEERSoT_(ptr noundef nonnull align 8 dereferenceable(8) %2, i64 noundef %40)
          to label %45 unwind label %117

45:                                               ; preds = %43
  %46 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %44, ptr noundef nonnull @.str.1, i64 noundef 4)
          to label %47 unwind label %117

47:                                               ; preds = %45
  %48 = load ptr, ptr %0, align 8, !tbaa !12
  %49 = getelementptr inbounds nuw %class.BigInt, ptr %48, i64 %40
  %50 = getelementptr inbounds nuw i8, ptr %49, i64 8
  %51 = load ptr, ptr %50, align 8, !tbaa !25
  %52 = load ptr, ptr %49, align 8, !tbaa !26
  %53 = ptrtoint ptr %51 to i64
  %54 = ptrtoint ptr %52 to i64
  %55 = sub i64 %53, %54
  %56 = ashr exact i64 %55, 3
  %57 = add nsw i64 %56, -1
  %58 = icmp eq i64 %57, 0
  br i1 %58, label %61, label %65

59:                                               ; preds = %99
  %60 = load ptr, ptr %49, align 8, !tbaa !26
  br label %61

61:                                               ; preds = %59, %47
  %62 = phi ptr [ %60, %59 ], [ %52, %47 ]
  %63 = load i64, ptr %62, align 8, !tbaa !29
  %64 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo9_M_insertImEERSoT_(ptr noundef nonnull align 8 dereferenceable(8) %44, i64 noundef %63)
          to label %103 unwind label %117

65:                                               ; preds = %47, %99
  %66 = phi i64 [ %101, %99 ], [ %57, %47 ]
  %67 = load ptr, ptr %49, align 8, !tbaa !26
  %68 = getelementptr inbounds nuw i64, ptr %67, i64 %66
  %69 = load i64, ptr %68, align 8, !tbaa !29
  %70 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo9_M_insertImEERSoT_(ptr noundef nonnull align 8 dereferenceable(8) %44, i64 noundef %69)
          to label %71 unwind label %115

71:                                               ; preds = %65
  %72 = load ptr, ptr %70, align 8, !tbaa !69
  %73 = getelementptr i8, ptr %72, i64 -24
  %74 = load i64, ptr %73, align 8
  %75 = getelementptr inbounds i8, ptr %70, i64 %74
  %76 = getelementptr inbounds nuw i8, ptr %75, i64 16
  store i64 9, ptr %76, align 8, !tbaa !71
  %77 = load i64, ptr %73, align 8
  %78 = getelementptr inbounds i8, ptr %70, i64 %77
  %79 = getelementptr inbounds nuw i8, ptr %78, i64 225
  %80 = load i8, ptr %79, align 1, !tbaa !79, !range !87, !noundef !88
  %81 = trunc nuw i8 %80 to i1
  br i1 %81, label %99, label %82

82:                                               ; preds = %71
  %83 = getelementptr inbounds nuw i8, ptr %78, i64 240
  %84 = load ptr, ptr %83, align 8, !tbaa !89
  %85 = icmp eq ptr %84, null
  br i1 %85, label %86, label %88

86:                                               ; preds = %82
  invoke void @_ZSt16__throw_bad_castv() #20
          to label %87 unwind label %119

87:                                               ; preds = %86
  unreachable

88:                                               ; preds = %82
  %89 = getelementptr inbounds nuw i8, ptr %84, i64 56
  %90 = load i8, ptr %89, align 8, !tbaa !90
  %91 = icmp eq i8 %90, 0
  br i1 %91, label %92, label %98

92:                                               ; preds = %88
  invoke void @_ZNKSt5ctypeIcE13_M_widen_initEv(ptr noundef nonnull align 8 dereferenceable(570) %84)
          to label %93 unwind label %115

93:                                               ; preds = %92
  %94 = load ptr, ptr %84, align 8, !tbaa !69
  %95 = getelementptr inbounds nuw i8, ptr %94, i64 48
  %96 = load ptr, ptr %95, align 8
  %97 = invoke noundef i8 %96(ptr noundef nonnull align 8 dereferenceable(570) %84, i8 noundef 32)
          to label %98 unwind label %115

98:                                               ; preds = %93, %88
  store i8 1, ptr %79, align 1, !tbaa !79
  br label %99

99:                                               ; preds = %98, %71
  %100 = getelementptr inbounds nuw i8, ptr %78, i64 224
  store i8 48, ptr %100, align 8, !tbaa !96
  %101 = add i64 %66, -1
  %102 = icmp eq i64 %101, 0
  br i1 %102, label %59, label %65, !llvm.loop !97

103:                                              ; preds = %61
  %104 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %64, ptr noundef nonnull @.str.2, i64 noundef 1)
          to label %105 unwind label %117

105:                                              ; preds = %103
  %106 = add i32 %41, 1
  %107 = zext i32 %106 to i64
  %108 = load ptr, ptr %4, align 8, !tbaa !13
  %109 = load ptr, ptr %0, align 8, !tbaa !12
  %110 = ptrtoint ptr %108 to i64
  %111 = ptrtoint ptr %109 to i64
  %112 = sub i64 %110, %111
  %113 = sdiv exact i64 %112, 24
  %114 = icmp ugt i64 %113, %107
  br i1 %114, label %39, label %8, !llvm.loop !98

115:                                              ; preds = %65, %92, %93
  %116 = landingpad { ptr, i32 }
          cleanup
  br label %167

117:                                              ; preds = %103, %61, %45, %43, %39
  %118 = landingpad { ptr, i32 }
          cleanup
  br label %167

119:                                              ; preds = %86
  %120 = landingpad { ptr, i32 }
          cleanup
  br label %167

121:                                              ; preds = %37, %20
  %122 = load ptr, ptr %3, align 8, !tbaa !68
  %123 = load i64, ptr %10, align 8, !tbaa !60
  %124 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef %122, i64 noundef %123)
          to label %125 unwind label %155

125:                                              ; preds = %121
  %126 = load ptr, ptr %3, align 8, !tbaa !68
  %127 = icmp eq ptr %126, %9
  br i1 %127, label %128, label %131

128:                                              ; preds = %125
  %129 = load i64, ptr %10, align 8, !tbaa !60
  %130 = icmp ult i64 %129, 16
  call void @llvm.assume(i1 %130)
  br label %134

131:                                              ; preds = %125
  %132 = load i64, ptr %9, align 8, !tbaa !62
  %133 = add i64 %132, 1
  call void @_ZdlPvm(ptr noundef %126, i64 noundef %133) #18
  br label %134

134:                                              ; preds = %128, %131
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #21
  %135 = load ptr, ptr @_ZTTNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEEE, align 8
  store ptr %135, ptr %2, align 8, !tbaa !69
  %136 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @_ZTTNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEEE, i64 24), align 8
  %137 = getelementptr i8, ptr %135, i64 -24
  %138 = load i64, ptr %137, align 8
  %139 = getelementptr inbounds i8, ptr %2, i64 %138
  store ptr %136, ptr %139, align 8, !tbaa !69
  %140 = getelementptr inbounds nuw i8, ptr %2, i64 8
  store ptr getelementptr inbounds nuw inrange(-16, 112) (i8, ptr @_ZTVNSt7__cxx1115basic_stringbufIcSt11char_traitsIcESaIcEEE, i64 16), ptr %140, align 8, !tbaa !69
  %141 = getelementptr inbounds nuw i8, ptr %2, i64 80
  %142 = load ptr, ptr %141, align 8, !tbaa !68
  %143 = getelementptr inbounds nuw i8, ptr %2, i64 96
  %144 = icmp eq ptr %142, %143
  br i1 %144, label %145, label %149

145:                                              ; preds = %134
  %146 = getelementptr inbounds nuw i8, ptr %2, i64 88
  %147 = load i64, ptr %146, align 8, !tbaa !60
  %148 = icmp ult i64 %147, 16
  call void @llvm.assume(i1 %148)
  br label %152

149:                                              ; preds = %134
  %150 = load i64, ptr %143, align 8, !tbaa !62
  %151 = add i64 %150, 1
  call void @_ZdlPvm(ptr noundef %142, i64 noundef %151) #18
  br label %152

152:                                              ; preds = %145, %149
  store ptr getelementptr inbounds nuw inrange(-16, 112) (i8, ptr @_ZTVSt15basic_streambufIcSt11char_traitsIcEE, i64 16), ptr %140, align 8, !tbaa !69
  %153 = getelementptr inbounds nuw i8, ptr %2, i64 64
  call void @_ZNSt6localeD1Ev(ptr noundef nonnull align 8 dereferenceable(8) %153) #21
  %154 = getelementptr inbounds nuw i8, ptr %2, i64 112
  call void @_ZNSt8ios_baseD2Ev(ptr noundef nonnull align 8 dereferenceable(264) %154) #21
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #21
  ret void

155:                                              ; preds = %121
  %156 = landingpad { ptr, i32 }
          cleanup
  %157 = load ptr, ptr %3, align 8, !tbaa !68
  %158 = icmp eq ptr %157, %9
  br i1 %158, label %159, label %162

159:                                              ; preds = %155
  %160 = load i64, ptr %10, align 8, !tbaa !60
  %161 = icmp ult i64 %160, 16
  call void @llvm.assume(i1 %161)
  br label %165

162:                                              ; preds = %155
  %163 = load i64, ptr %9, align 8, !tbaa !62
  %164 = add i64 %163, 1
  call void @_ZdlPvm(ptr noundef %157, i64 noundef %164) #18
  br label %165

165:                                              ; preds = %162, %159, %34, %31
  %166 = phi { ptr, i32 } [ %28, %34 ], [ %28, %31 ], [ %156, %159 ], [ %156, %162 ]
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #21
  br label %167

167:                                              ; preds = %115, %119, %117, %165
  %168 = phi { ptr, i32 } [ %166, %165 ], [ %116, %115 ], [ %118, %117 ], [ %120, %119 ]
  call void @_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev(ptr noundef nonnull align 8 dereferenceable(112) %2) #21
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #21
  resume { ptr, i32 } %168
}

; Function Attrs: mustprogress uwtable
declare void @_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEEC1Ev(ptr noundef nonnull align 8 dereferenceable(112)) unnamed_addr #0

; Function Attrs: mustprogress nounwind uwtable
declare void @_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev(ptr noundef nonnull align 8 dereferenceable(112)) unnamed_addr #2

; Function Attrs: mustprogress uwtable
define dso_local void @_ZNK9Fibonacci16show_last_numberEv(ptr noundef nonnull readonly align 8 captures(none) dereferenceable(24) %0) local_unnamed_addr #0 personality ptr @__gxx_personality_v0 {
  %2 = alloca %"class.std::__cxx11::basic_ostringstream", align 8
  %3 = alloca %"class.std::__cxx11::basic_string", align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #21
  call void @_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEEC1Ev(ptr noundef nonnull align 8 dereferenceable(112) %2)
  %4 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %2, ptr noundef nonnull @.str, i64 noundef 5)
          to label %5 unwind label %142

5:                                                ; preds = %1
  %6 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %7 = load ptr, ptr %6, align 8, !tbaa !13
  %8 = load ptr, ptr %0, align 8, !tbaa !12
  %9 = ptrtoint ptr %7 to i64
  %10 = ptrtoint ptr %8 to i64
  %11 = sub i64 %9, %10
  %12 = sdiv exact i64 %11, 24
  %13 = add nsw i64 %12, -1
  %14 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo9_M_insertImEERSoT_(ptr noundef nonnull align 8 dereferenceable(8) %2, i64 noundef %13)
          to label %15 unwind label %142

15:                                               ; preds = %5
  %16 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %14, ptr noundef nonnull @.str.1, i64 noundef 4)
          to label %17 unwind label %142

17:                                               ; preds = %15
  %18 = load ptr, ptr %6, align 8, !tbaa !36
  %19 = getelementptr inbounds i8, ptr %18, i64 -24
  %20 = getelementptr inbounds i8, ptr %18, i64 -16
  %21 = load ptr, ptr %20, align 8, !tbaa !25
  %22 = load ptr, ptr %19, align 8, !tbaa !26
  %23 = ptrtoint ptr %21 to i64
  %24 = ptrtoint ptr %22 to i64
  %25 = sub i64 %23, %24
  %26 = ashr exact i64 %25, 3
  %27 = add nsw i64 %26, -1
  %28 = icmp eq i64 %27, 0
  br i1 %28, label %31, label %35

29:                                               ; preds = %69
  %30 = load ptr, ptr %19, align 8, !tbaa !26
  br label %31

31:                                               ; preds = %29, %17
  %32 = phi ptr [ %30, %29 ], [ %22, %17 ]
  %33 = load i64, ptr %32, align 8, !tbaa !29
  %34 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo9_M_insertImEERSoT_(ptr noundef nonnull align 8 dereferenceable(8) %14, i64 noundef %33)
          to label %73 unwind label %142

35:                                               ; preds = %17, %69
  %36 = phi i64 [ %71, %69 ], [ %27, %17 ]
  %37 = load ptr, ptr %19, align 8, !tbaa !26
  %38 = getelementptr inbounds nuw i64, ptr %37, i64 %36
  %39 = load i64, ptr %38, align 8, !tbaa !29
  %40 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo9_M_insertImEERSoT_(ptr noundef nonnull align 8 dereferenceable(8) %14, i64 noundef %39)
          to label %41 unwind label %140

41:                                               ; preds = %35
  %42 = load ptr, ptr %40, align 8, !tbaa !69
  %43 = getelementptr i8, ptr %42, i64 -24
  %44 = load i64, ptr %43, align 8
  %45 = getelementptr inbounds i8, ptr %40, i64 %44
  %46 = getelementptr inbounds nuw i8, ptr %45, i64 16
  store i64 9, ptr %46, align 8, !tbaa !71
  %47 = load i64, ptr %43, align 8
  %48 = getelementptr inbounds i8, ptr %40, i64 %47
  %49 = getelementptr inbounds nuw i8, ptr %48, i64 225
  %50 = load i8, ptr %49, align 1, !tbaa !79, !range !87, !noundef !88
  %51 = trunc nuw i8 %50 to i1
  br i1 %51, label %69, label %52

52:                                               ; preds = %41
  %53 = getelementptr inbounds nuw i8, ptr %48, i64 240
  %54 = load ptr, ptr %53, align 8, !tbaa !89
  %55 = icmp eq ptr %54, null
  br i1 %55, label %56, label %58

56:                                               ; preds = %52
  invoke void @_ZSt16__throw_bad_castv() #20
          to label %57 unwind label %142

57:                                               ; preds = %56
  unreachable

58:                                               ; preds = %52
  %59 = getelementptr inbounds nuw i8, ptr %54, i64 56
  %60 = load i8, ptr %59, align 8, !tbaa !90
  %61 = icmp eq i8 %60, 0
  br i1 %61, label %62, label %68

62:                                               ; preds = %58
  invoke void @_ZNKSt5ctypeIcE13_M_widen_initEv(ptr noundef nonnull align 8 dereferenceable(570) %54)
          to label %63 unwind label %140

63:                                               ; preds = %62
  %64 = load ptr, ptr %54, align 8, !tbaa !69
  %65 = getelementptr inbounds nuw i8, ptr %64, i64 48
  %66 = load ptr, ptr %65, align 8
  %67 = invoke noundef i8 %66(ptr noundef nonnull align 8 dereferenceable(570) %54, i8 noundef 32)
          to label %68 unwind label %140

68:                                               ; preds = %63, %58
  store i8 1, ptr %49, align 1, !tbaa !79
  br label %69

69:                                               ; preds = %68, %41
  %70 = getelementptr inbounds nuw i8, ptr %48, i64 224
  store i8 48, ptr %70, align 8, !tbaa !96
  %71 = add i64 %36, -1
  %72 = icmp eq i64 %71, 0
  br i1 %72, label %29, label %35, !llvm.loop !97

73:                                               ; preds = %31
  %74 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %34, ptr noundef nonnull @.str.2, i64 noundef 1)
          to label %75 unwind label %142

75:                                               ; preds = %73
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #21
  call void @llvm.experimental.noalias.scope.decl(metadata !99)
  call void @llvm.experimental.noalias.scope.decl(metadata !102)
  %76 = getelementptr inbounds nuw i8, ptr %3, i64 16
  store ptr %76, ptr %3, align 8, !tbaa !56, !alias.scope !105
  %77 = getelementptr inbounds nuw i8, ptr %3, i64 8
  store i64 0, ptr %77, align 8, !tbaa !60, !alias.scope !105
  store i8 0, ptr %76, align 8, !tbaa !62, !alias.scope !105
  %78 = getelementptr inbounds nuw i8, ptr %2, i64 48
  %79 = load ptr, ptr %78, align 8, !tbaa !63, !noalias !105
  %80 = icmp eq ptr %79, null
  %81 = getelementptr inbounds nuw i8, ptr %2, i64 32
  %82 = load ptr, ptr %81, align 8, !noalias !105
  %83 = icmp ugt ptr %79, %82
  %84 = select i1 %83, ptr %79, ptr %82
  %85 = icmp eq ptr %84, null
  %86 = select i1 %80, i1 true, i1 %85
  br i1 %86, label %104, label %87

87:                                               ; preds = %75
  %88 = getelementptr inbounds nuw i8, ptr %2, i64 40
  %89 = load ptr, ptr %88, align 8, !tbaa !67, !noalias !105
  %90 = ptrtoint ptr %84 to i64
  %91 = ptrtoint ptr %89 to i64
  %92 = sub i64 %90, %91
  %93 = invoke noundef nonnull align 8 dereferenceable(32) ptr @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_replaceEmmPKcm(ptr noundef nonnull align 8 dereferenceable(32) %3, i64 noundef 0, i64 noundef 0, ptr noundef %89, i64 noundef %92)
          to label %106 unwind label %94

94:                                               ; preds = %104, %87
  %95 = landingpad { ptr, i32 }
          cleanup
  %96 = load ptr, ptr %3, align 8, !tbaa !68, !alias.scope !105
  %97 = icmp eq ptr %96, %76
  br i1 %97, label %98, label %101

98:                                               ; preds = %94
  %99 = load i64, ptr %77, align 8, !tbaa !60, !alias.scope !105
  %100 = icmp ult i64 %99, 16
  call void @llvm.assume(i1 %100)
  br label %154

101:                                              ; preds = %94
  %102 = load i64, ptr %76, align 8, !tbaa !62, !alias.scope !105
  %103 = add i64 %102, 1
  call void @_ZdlPvm(ptr noundef %96, i64 noundef %103) #18
  br label %154

104:                                              ; preds = %75
  %105 = getelementptr inbounds nuw i8, ptr %2, i64 80
  invoke void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_assignERKS4_(ptr noundef nonnull align 8 dereferenceable(32) %3, ptr noundef nonnull align 8 dereferenceable(32) %105)
          to label %106 unwind label %94

106:                                              ; preds = %104, %87
  %107 = load ptr, ptr %3, align 8, !tbaa !68
  %108 = load i64, ptr %77, align 8, !tbaa !60
  %109 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef %107, i64 noundef %108)
          to label %110 unwind label %144

110:                                              ; preds = %106
  %111 = load ptr, ptr %3, align 8, !tbaa !68
  %112 = icmp eq ptr %111, %76
  br i1 %112, label %113, label %116

113:                                              ; preds = %110
  %114 = load i64, ptr %77, align 8, !tbaa !60
  %115 = icmp ult i64 %114, 16
  call void @llvm.assume(i1 %115)
  br label %119

116:                                              ; preds = %110
  %117 = load i64, ptr %76, align 8, !tbaa !62
  %118 = add i64 %117, 1
  call void @_ZdlPvm(ptr noundef %111, i64 noundef %118) #18
  br label %119

119:                                              ; preds = %113, %116
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #21
  %120 = load ptr, ptr @_ZTTNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEEE, align 8
  store ptr %120, ptr %2, align 8, !tbaa !69
  %121 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @_ZTTNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEEE, i64 24), align 8
  %122 = getelementptr i8, ptr %120, i64 -24
  %123 = load i64, ptr %122, align 8
  %124 = getelementptr inbounds i8, ptr %2, i64 %123
  store ptr %121, ptr %124, align 8, !tbaa !69
  %125 = getelementptr inbounds nuw i8, ptr %2, i64 8
  store ptr getelementptr inbounds nuw inrange(-16, 112) (i8, ptr @_ZTVNSt7__cxx1115basic_stringbufIcSt11char_traitsIcESaIcEEE, i64 16), ptr %125, align 8, !tbaa !69
  %126 = getelementptr inbounds nuw i8, ptr %2, i64 80
  %127 = load ptr, ptr %126, align 8, !tbaa !68
  %128 = getelementptr inbounds nuw i8, ptr %2, i64 96
  %129 = icmp eq ptr %127, %128
  br i1 %129, label %130, label %134

130:                                              ; preds = %119
  %131 = getelementptr inbounds nuw i8, ptr %2, i64 88
  %132 = load i64, ptr %131, align 8, !tbaa !60
  %133 = icmp ult i64 %132, 16
  call void @llvm.assume(i1 %133)
  br label %137

134:                                              ; preds = %119
  %135 = load i64, ptr %128, align 8, !tbaa !62
  %136 = add i64 %135, 1
  call void @_ZdlPvm(ptr noundef %127, i64 noundef %136) #18
  br label %137

137:                                              ; preds = %130, %134
  store ptr getelementptr inbounds nuw inrange(-16, 112) (i8, ptr @_ZTVSt15basic_streambufIcSt11char_traitsIcEE, i64 16), ptr %125, align 8, !tbaa !69
  %138 = getelementptr inbounds nuw i8, ptr %2, i64 64
  call void @_ZNSt6localeD1Ev(ptr noundef nonnull align 8 dereferenceable(8) %138) #21
  %139 = getelementptr inbounds nuw i8, ptr %2, i64 112
  call void @_ZNSt8ios_baseD2Ev(ptr noundef nonnull align 8 dereferenceable(264) %139) #21
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #21
  ret void

140:                                              ; preds = %35, %62, %63
  %141 = landingpad { ptr, i32 }
          cleanup
  br label %156

142:                                              ; preds = %1, %5, %15, %31, %56, %73
  %143 = landingpad { ptr, i32 }
          cleanup
  br label %156

144:                                              ; preds = %106
  %145 = landingpad { ptr, i32 }
          cleanup
  %146 = load ptr, ptr %3, align 8, !tbaa !68
  %147 = icmp eq ptr %146, %76
  br i1 %147, label %148, label %151

148:                                              ; preds = %144
  %149 = load i64, ptr %77, align 8, !tbaa !60
  %150 = icmp ult i64 %149, 16
  call void @llvm.assume(i1 %150)
  br label %154

151:                                              ; preds = %144
  %152 = load i64, ptr %76, align 8, !tbaa !62
  %153 = add i64 %152, 1
  call void @_ZdlPvm(ptr noundef %146, i64 noundef %153) #18
  br label %154

154:                                              ; preds = %151, %148, %101, %98
  %155 = phi { ptr, i32 } [ %95, %101 ], [ %95, %98 ], [ %145, %148 ], [ %145, %151 ]
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #21
  br label %156

156:                                              ; preds = %140, %142, %154
  %157 = phi { ptr, i32 } [ %155, %154 ], [ %141, %140 ], [ %143, %142 ]
  call void @_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev(ptr noundef nonnull align 8 dereferenceable(112) %2) #21
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #21
  resume { ptr, i32 } %157
}

; Function Attrs: mustprogress uwtable
define dso_local void @_ZN9Fibonacci11show_numberEm(ptr noundef nonnull align 8 dereferenceable(24) %0, i64 noundef %1) local_unnamed_addr #0 personality ptr @__gxx_personality_v0 {
  %3 = alloca %"class.std::__cxx11::basic_ostringstream", align 8
  %4 = alloca %class.BigInt, align 8
  %5 = alloca %"class.std::__cxx11::basic_string", align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #21
  call void @_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEEC1Ev(ptr noundef nonnull align 8 dereferenceable(112) %3)
  %6 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %7 = load ptr, ptr %6, align 8, !tbaa !13
  %8 = load ptr, ptr %0, align 8, !tbaa !12
  %9 = ptrtoint ptr %7 to i64
  %10 = ptrtoint ptr %8 to i64
  %11 = sub i64 %9, %10
  %12 = sdiv exact i64 %11, 24
  %13 = icmp ult i64 %1, %12
  br i1 %13, label %29, label %14

14:                                               ; preds = %2
  %15 = trunc i64 %1 to i32
  invoke void @_ZN9Fibonacci10get_numberEj(ptr dead_on_unwind nonnull writable sret(%class.BigInt) align 8 %4, ptr noundef nonnull align 8 dereferenceable(24) %0, i32 noundef %15)
          to label %16 unwind label %27

16:                                               ; preds = %14
  %17 = load ptr, ptr %4, align 8, !tbaa !26
  %18 = icmp eq ptr %17, null
  br i1 %18, label %29, label %19

19:                                               ; preds = %16
  %20 = getelementptr inbounds nuw i8, ptr %4, i64 16
  %21 = load ptr, ptr %20, align 8, !tbaa !21
  %22 = ptrtoint ptr %21 to i64
  %23 = ptrtoint ptr %17 to i64
  %24 = sub i64 %22, %23
  call void @_ZdlPvm(ptr noundef nonnull %17, i64 noundef %24) #18
  br label %29

25:                                               ; preds = %53, %80, %81
  %26 = landingpad { ptr, i32 }
          cleanup
  br label %170

27:                                               ; preds = %14, %29, %31, %33, %49, %74, %91
  %28 = landingpad { ptr, i32 }
          cleanup
  br label %170

29:                                               ; preds = %19, %16, %2
  %30 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %3, ptr noundef nonnull @.str, i64 noundef 5)
          to label %31 unwind label %27

31:                                               ; preds = %29
  %32 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo9_M_insertImEERSoT_(ptr noundef nonnull align 8 dereferenceable(8) %3, i64 noundef %1)
          to label %33 unwind label %27

33:                                               ; preds = %31
  %34 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %32, ptr noundef nonnull @.str.1, i64 noundef 4)
          to label %35 unwind label %27

35:                                               ; preds = %33
  %36 = load ptr, ptr %0, align 8, !tbaa !12
  %37 = getelementptr inbounds nuw %class.BigInt, ptr %36, i64 %1
  %38 = getelementptr inbounds nuw i8, ptr %37, i64 8
  %39 = load ptr, ptr %38, align 8, !tbaa !25
  %40 = load ptr, ptr %37, align 8, !tbaa !26
  %41 = ptrtoint ptr %39 to i64
  %42 = ptrtoint ptr %40 to i64
  %43 = sub i64 %41, %42
  %44 = ashr exact i64 %43, 3
  %45 = add nsw i64 %44, -1
  %46 = icmp eq i64 %45, 0
  br i1 %46, label %49, label %53

47:                                               ; preds = %87
  %48 = load ptr, ptr %37, align 8, !tbaa !26
  br label %49

49:                                               ; preds = %47, %35
  %50 = phi ptr [ %48, %47 ], [ %40, %35 ]
  %51 = load i64, ptr %50, align 8, !tbaa !29
  %52 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo9_M_insertImEERSoT_(ptr noundef nonnull align 8 dereferenceable(8) %32, i64 noundef %51)
          to label %91 unwind label %27

53:                                               ; preds = %35, %87
  %54 = phi i64 [ %89, %87 ], [ %45, %35 ]
  %55 = load ptr, ptr %37, align 8, !tbaa !26
  %56 = getelementptr inbounds nuw i64, ptr %55, i64 %54
  %57 = load i64, ptr %56, align 8, !tbaa !29
  %58 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo9_M_insertImEERSoT_(ptr noundef nonnull align 8 dereferenceable(8) %32, i64 noundef %57)
          to label %59 unwind label %25

59:                                               ; preds = %53
  %60 = load ptr, ptr %58, align 8, !tbaa !69
  %61 = getelementptr i8, ptr %60, i64 -24
  %62 = load i64, ptr %61, align 8
  %63 = getelementptr inbounds i8, ptr %58, i64 %62
  %64 = getelementptr inbounds nuw i8, ptr %63, i64 16
  store i64 9, ptr %64, align 8, !tbaa !71
  %65 = load i64, ptr %61, align 8
  %66 = getelementptr inbounds i8, ptr %58, i64 %65
  %67 = getelementptr inbounds nuw i8, ptr %66, i64 225
  %68 = load i8, ptr %67, align 1, !tbaa !79, !range !87, !noundef !88
  %69 = trunc nuw i8 %68 to i1
  br i1 %69, label %87, label %70

70:                                               ; preds = %59
  %71 = getelementptr inbounds nuw i8, ptr %66, i64 240
  %72 = load ptr, ptr %71, align 8, !tbaa !89
  %73 = icmp eq ptr %72, null
  br i1 %73, label %74, label %76

74:                                               ; preds = %70
  invoke void @_ZSt16__throw_bad_castv() #20
          to label %75 unwind label %27

75:                                               ; preds = %74
  unreachable

76:                                               ; preds = %70
  %77 = getelementptr inbounds nuw i8, ptr %72, i64 56
  %78 = load i8, ptr %77, align 8, !tbaa !90
  %79 = icmp eq i8 %78, 0
  br i1 %79, label %80, label %86

80:                                               ; preds = %76
  invoke void @_ZNKSt5ctypeIcE13_M_widen_initEv(ptr noundef nonnull align 8 dereferenceable(570) %72)
          to label %81 unwind label %25

81:                                               ; preds = %80
  %82 = load ptr, ptr %72, align 8, !tbaa !69
  %83 = getelementptr inbounds nuw i8, ptr %82, i64 48
  %84 = load ptr, ptr %83, align 8
  %85 = invoke noundef i8 %84(ptr noundef nonnull align 8 dereferenceable(570) %72, i8 noundef 32)
          to label %86 unwind label %25

86:                                               ; preds = %81, %76
  store i8 1, ptr %67, align 1, !tbaa !79
  br label %87

87:                                               ; preds = %86, %59
  %88 = getelementptr inbounds nuw i8, ptr %66, i64 224
  store i8 48, ptr %88, align 8, !tbaa !96
  %89 = add i64 %54, -1
  %90 = icmp eq i64 %89, 0
  br i1 %90, label %47, label %53, !llvm.loop !97

91:                                               ; preds = %49
  %92 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %52, ptr noundef nonnull @.str.2, i64 noundef 1)
          to label %93 unwind label %27

93:                                               ; preds = %91
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #21
  call void @llvm.experimental.noalias.scope.decl(metadata !106)
  call void @llvm.experimental.noalias.scope.decl(metadata !109)
  %94 = getelementptr inbounds nuw i8, ptr %5, i64 16
  store ptr %94, ptr %5, align 8, !tbaa !56, !alias.scope !112
  %95 = getelementptr inbounds nuw i8, ptr %5, i64 8
  store i64 0, ptr %95, align 8, !tbaa !60, !alias.scope !112
  store i8 0, ptr %94, align 8, !tbaa !62, !alias.scope !112
  %96 = getelementptr inbounds nuw i8, ptr %3, i64 48
  %97 = load ptr, ptr %96, align 8, !tbaa !63, !noalias !112
  %98 = icmp eq ptr %97, null
  %99 = getelementptr inbounds nuw i8, ptr %3, i64 32
  %100 = load ptr, ptr %99, align 8, !noalias !112
  %101 = icmp ugt ptr %97, %100
  %102 = select i1 %101, ptr %97, ptr %100
  %103 = icmp eq ptr %102, null
  %104 = select i1 %98, i1 true, i1 %103
  br i1 %104, label %122, label %105

105:                                              ; preds = %93
  %106 = getelementptr inbounds nuw i8, ptr %3, i64 40
  %107 = load ptr, ptr %106, align 8, !tbaa !67, !noalias !112
  %108 = ptrtoint ptr %102 to i64
  %109 = ptrtoint ptr %107 to i64
  %110 = sub i64 %108, %109
  %111 = invoke noundef nonnull align 8 dereferenceable(32) ptr @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_replaceEmmPKcm(ptr noundef nonnull align 8 dereferenceable(32) %5, i64 noundef 0, i64 noundef 0, ptr noundef %107, i64 noundef %110)
          to label %124 unwind label %112

112:                                              ; preds = %122, %105
  %113 = landingpad { ptr, i32 }
          cleanup
  %114 = load ptr, ptr %5, align 8, !tbaa !68, !alias.scope !112
  %115 = icmp eq ptr %114, %94
  br i1 %115, label %116, label %119

116:                                              ; preds = %112
  %117 = load i64, ptr %95, align 8, !tbaa !60, !alias.scope !112
  %118 = icmp ult i64 %117, 16
  call void @llvm.assume(i1 %118)
  br label %168

119:                                              ; preds = %112
  %120 = load i64, ptr %94, align 8, !tbaa !62, !alias.scope !112
  %121 = add i64 %120, 1
  call void @_ZdlPvm(ptr noundef %114, i64 noundef %121) #18
  br label %168

122:                                              ; preds = %93
  %123 = getelementptr inbounds nuw i8, ptr %3, i64 80
  invoke void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_assignERKS4_(ptr noundef nonnull align 8 dereferenceable(32) %5, ptr noundef nonnull align 8 dereferenceable(32) %123)
          to label %124 unwind label %112

124:                                              ; preds = %122, %105
  %125 = load ptr, ptr %5, align 8, !tbaa !68
  %126 = load i64, ptr %95, align 8, !tbaa !60
  %127 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef %125, i64 noundef %126)
          to label %128 unwind label %158

128:                                              ; preds = %124
  %129 = load ptr, ptr %5, align 8, !tbaa !68
  %130 = icmp eq ptr %129, %94
  br i1 %130, label %131, label %134

131:                                              ; preds = %128
  %132 = load i64, ptr %95, align 8, !tbaa !60
  %133 = icmp ult i64 %132, 16
  call void @llvm.assume(i1 %133)
  br label %137

134:                                              ; preds = %128
  %135 = load i64, ptr %94, align 8, !tbaa !62
  %136 = add i64 %135, 1
  call void @_ZdlPvm(ptr noundef %129, i64 noundef %136) #18
  br label %137

137:                                              ; preds = %131, %134
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #21
  %138 = load ptr, ptr @_ZTTNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEEE, align 8
  store ptr %138, ptr %3, align 8, !tbaa !69
  %139 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @_ZTTNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEEE, i64 24), align 8
  %140 = getelementptr i8, ptr %138, i64 -24
  %141 = load i64, ptr %140, align 8
  %142 = getelementptr inbounds i8, ptr %3, i64 %141
  store ptr %139, ptr %142, align 8, !tbaa !69
  %143 = getelementptr inbounds nuw i8, ptr %3, i64 8
  store ptr getelementptr inbounds nuw inrange(-16, 112) (i8, ptr @_ZTVNSt7__cxx1115basic_stringbufIcSt11char_traitsIcESaIcEEE, i64 16), ptr %143, align 8, !tbaa !69
  %144 = getelementptr inbounds nuw i8, ptr %3, i64 80
  %145 = load ptr, ptr %144, align 8, !tbaa !68
  %146 = getelementptr inbounds nuw i8, ptr %3, i64 96
  %147 = icmp eq ptr %145, %146
  br i1 %147, label %148, label %152

148:                                              ; preds = %137
  %149 = getelementptr inbounds nuw i8, ptr %3, i64 88
  %150 = load i64, ptr %149, align 8, !tbaa !60
  %151 = icmp ult i64 %150, 16
  call void @llvm.assume(i1 %151)
  br label %155

152:                                              ; preds = %137
  %153 = load i64, ptr %146, align 8, !tbaa !62
  %154 = add i64 %153, 1
  call void @_ZdlPvm(ptr noundef %145, i64 noundef %154) #18
  br label %155

155:                                              ; preds = %148, %152
  store ptr getelementptr inbounds nuw inrange(-16, 112) (i8, ptr @_ZTVSt15basic_streambufIcSt11char_traitsIcEE, i64 16), ptr %143, align 8, !tbaa !69
  %156 = getelementptr inbounds nuw i8, ptr %3, i64 64
  call void @_ZNSt6localeD1Ev(ptr noundef nonnull align 8 dereferenceable(8) %156) #21
  %157 = getelementptr inbounds nuw i8, ptr %3, i64 112
  call void @_ZNSt8ios_baseD2Ev(ptr noundef nonnull align 8 dereferenceable(264) %157) #21
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #21
  ret void

158:                                              ; preds = %124
  %159 = landingpad { ptr, i32 }
          cleanup
  %160 = load ptr, ptr %5, align 8, !tbaa !68
  %161 = icmp eq ptr %160, %94
  br i1 %161, label %162, label %165

162:                                              ; preds = %158
  %163 = load i64, ptr %95, align 8, !tbaa !60
  %164 = icmp ult i64 %163, 16
  call void @llvm.assume(i1 %164)
  br label %168

165:                                              ; preds = %158
  %166 = load i64, ptr %94, align 8, !tbaa !62
  %167 = add i64 %166, 1
  call void @_ZdlPvm(ptr noundef %160, i64 noundef %167) #18
  br label %168

168:                                              ; preds = %165, %162, %119, %116
  %169 = phi { ptr, i32 } [ %113, %119 ], [ %113, %116 ], [ %159, %162 ], [ %159, %165 ]
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #21
  br label %170

170:                                              ; preds = %25, %27, %168
  %171 = phi { ptr, i32 } [ %169, %168 ], [ %26, %25 ], [ %28, %27 ]
  call void @_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev(ptr noundef nonnull align 8 dereferenceable(112) %3) #21
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #21
  resume { ptr, i32 } %171
}

; Function Attrs: mustprogress uwtable
define dso_local void @_Z5usagePPc(ptr noundef captures(none) initializes((0, 8)) %0) local_unnamed_addr #0 {
  store ptr @.str.3, ptr %0, align 8, !tbaa !113
  %2 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, ptr noundef nonnull @.str.4, i64 noundef 8)
  %3 = load ptr, ptr @_ZSt4cerr, align 8, !tbaa !69
  %4 = getelementptr i8, ptr %3, i64 -24
  %5 = load i64, ptr %4, align 8
  %6 = getelementptr inbounds i8, ptr @_ZSt4cerr, i64 %5
  %7 = getelementptr inbounds nuw i8, ptr %6, i64 240
  %8 = load ptr, ptr %7, align 8, !tbaa !89
  %9 = icmp eq ptr %8, null
  br i1 %9, label %10, label %11

10:                                               ; preds = %1
  tail call void @_ZSt16__throw_bad_castv() #20
  unreachable

11:                                               ; preds = %1
  %12 = getelementptr inbounds nuw i8, ptr %8, i64 56
  %13 = load i8, ptr %12, align 8, !tbaa !90
  %14 = icmp eq i8 %13, 0
  br i1 %14, label %18, label %15

15:                                               ; preds = %11
  %16 = getelementptr inbounds nuw i8, ptr %8, i64 67
  %17 = load i8, ptr %16, align 1, !tbaa !62
  br label %23

18:                                               ; preds = %11
  tail call void @_ZNKSt5ctypeIcE13_M_widen_initEv(ptr noundef nonnull align 8 dereferenceable(570) %8)
  %19 = load ptr, ptr %8, align 8, !tbaa !69
  %20 = getelementptr inbounds nuw i8, ptr %19, i64 48
  %21 = load ptr, ptr %20, align 8
  %22 = tail call noundef i8 %21(ptr noundef nonnull align 8 dereferenceable(570) %8, i8 noundef 10)
  br label %23

23:                                               ; preds = %15, %18
  %24 = phi i8 [ %17, %15 ], [ %22, %18 ]
  %25 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, i8 noundef %24)
  %26 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo5flushEv(ptr noundef nonnull align 8 dereferenceable(8) %25)
  %27 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %26, ptr noundef nonnull @.str.5, i64 noundef 2)
  %28 = load ptr, ptr %0, align 8, !tbaa !113
  %29 = icmp eq ptr %28, null
  br i1 %29, label %30, label %38

30:                                               ; preds = %23
  %31 = load ptr, ptr %26, align 8, !tbaa !69
  %32 = getelementptr i8, ptr %31, i64 -24
  %33 = load i64, ptr %32, align 8
  %34 = getelementptr inbounds i8, ptr %26, i64 %33
  %35 = getelementptr inbounds nuw i8, ptr %34, i64 32
  %36 = load i32, ptr %35, align 8, !tbaa !114
  %37 = or i32 %36, 1
  tail call void @_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate(ptr noundef nonnull align 8 dereferenceable(264) %34, i32 noundef %37)
  br label %41

38:                                               ; preds = %23
  %39 = tail call noundef i64 @strlen(ptr noundef nonnull dereferenceable(1) %28) #21
  %40 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %26, ptr noundef nonnull %28, i64 noundef %39)
  br label %41

41:                                               ; preds = %30, %38
  %42 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %26, ptr noundef nonnull @.str.6, i64 noundef 1)
  %43 = load ptr, ptr %26, align 8, !tbaa !69
  %44 = getelementptr i8, ptr %43, i64 -24
  %45 = load i64, ptr %44, align 8
  %46 = getelementptr inbounds i8, ptr %26, i64 %45
  %47 = getelementptr inbounds nuw i8, ptr %46, i64 16
  store i64 4, ptr %47, align 8, !tbaa !71
  %48 = load i64, ptr %44, align 8
  %49 = getelementptr inbounds i8, ptr %26, i64 %48
  %50 = getelementptr inbounds nuw i8, ptr %49, i64 24
  %51 = load i32, ptr %50, align 8, !tbaa !115
  %52 = and i32 %51, -177
  %53 = or disjoint i32 %52, 32
  store i32 %53, ptr %50, align 8, !tbaa !116
  %54 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %26, ptr noundef nonnull @.str.7, i64 noundef 3)
  %55 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %26, ptr noundef nonnull @.str.8, i64 noundef 40)
  %56 = load ptr, ptr %26, align 8, !tbaa !69
  %57 = getelementptr i8, ptr %56, i64 -24
  %58 = load i64, ptr %57, align 8
  %59 = getelementptr inbounds i8, ptr %26, i64 %58
  %60 = getelementptr inbounds nuw i8, ptr %59, i64 240
  %61 = load ptr, ptr %60, align 8, !tbaa !89
  %62 = icmp eq ptr %61, null
  br i1 %62, label %63, label %64

63:                                               ; preds = %41
  tail call void @_ZSt16__throw_bad_castv() #20
  unreachable

64:                                               ; preds = %41
  %65 = getelementptr inbounds nuw i8, ptr %61, i64 56
  %66 = load i8, ptr %65, align 8, !tbaa !90
  %67 = icmp eq i8 %66, 0
  br i1 %67, label %71, label %68

68:                                               ; preds = %64
  %69 = getelementptr inbounds nuw i8, ptr %61, i64 67
  %70 = load i8, ptr %69, align 1, !tbaa !62
  br label %76

71:                                               ; preds = %64
  tail call void @_ZNKSt5ctypeIcE13_M_widen_initEv(ptr noundef nonnull align 8 dereferenceable(570) %61)
  %72 = load ptr, ptr %61, align 8, !tbaa !69
  %73 = getelementptr inbounds nuw i8, ptr %72, i64 48
  %74 = load ptr, ptr %73, align 8
  %75 = tail call noundef i8 %74(ptr noundef nonnull align 8 dereferenceable(570) %61, i8 noundef 10)
  br label %76

76:                                               ; preds = %68, %71
  %77 = phi i8 [ %70, %68 ], [ %75, %71 ]
  %78 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8) %26, i8 noundef %77)
  %79 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo5flushEv(ptr noundef nonnull align 8 dereferenceable(8) %78)
  %80 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %79, ptr noundef nonnull @.str.5, i64 noundef 2)
  %81 = load ptr, ptr %0, align 8, !tbaa !113
  %82 = icmp eq ptr %81, null
  br i1 %82, label %83, label %91

83:                                               ; preds = %76
  %84 = load ptr, ptr %79, align 8, !tbaa !69
  %85 = getelementptr i8, ptr %84, i64 -24
  %86 = load i64, ptr %85, align 8
  %87 = getelementptr inbounds i8, ptr %79, i64 %86
  %88 = getelementptr inbounds nuw i8, ptr %87, i64 32
  %89 = load i32, ptr %88, align 8, !tbaa !114
  %90 = or i32 %89, 1
  tail call void @_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate(ptr noundef nonnull align 8 dereferenceable(264) %87, i32 noundef %90)
  br label %94

91:                                               ; preds = %76
  %92 = tail call noundef i64 @strlen(ptr noundef nonnull dereferenceable(1) %81) #21
  %93 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %79, ptr noundef nonnull %81, i64 noundef %92)
  br label %94

94:                                               ; preds = %83, %91
  %95 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %79, ptr noundef nonnull @.str.6, i64 noundef 1)
  %96 = load ptr, ptr %79, align 8, !tbaa !69
  %97 = getelementptr i8, ptr %96, i64 -24
  %98 = load i64, ptr %97, align 8
  %99 = getelementptr inbounds i8, ptr %79, i64 %98
  %100 = getelementptr inbounds nuw i8, ptr %99, i64 24
  %101 = load i32, ptr %100, align 8, !tbaa !115
  %102 = and i32 %101, -177
  %103 = or disjoint i32 %102, 32
  store i32 %103, ptr %100, align 8, !tbaa !116
  %104 = load i64, ptr %97, align 8
  %105 = getelementptr inbounds i8, ptr %79, i64 %104
  %106 = getelementptr inbounds nuw i8, ptr %105, i64 16
  store i64 4, ptr %106, align 8, !tbaa !71
  %107 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %79, ptr noundef nonnull @.str.9, i64 noundef 2)
  %108 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %79, ptr noundef nonnull @.str.10, i64 noundef 36)
  %109 = load ptr, ptr %79, align 8, !tbaa !69
  %110 = getelementptr i8, ptr %109, i64 -24
  %111 = load i64, ptr %110, align 8
  %112 = getelementptr inbounds i8, ptr %79, i64 %111
  %113 = getelementptr inbounds nuw i8, ptr %112, i64 240
  %114 = load ptr, ptr %113, align 8, !tbaa !89
  %115 = icmp eq ptr %114, null
  br i1 %115, label %116, label %117

116:                                              ; preds = %94
  tail call void @_ZSt16__throw_bad_castv() #20
  unreachable

117:                                              ; preds = %94
  %118 = getelementptr inbounds nuw i8, ptr %114, i64 56
  %119 = load i8, ptr %118, align 8, !tbaa !90
  %120 = icmp eq i8 %119, 0
  br i1 %120, label %124, label %121

121:                                              ; preds = %117
  %122 = getelementptr inbounds nuw i8, ptr %114, i64 67
  %123 = load i8, ptr %122, align 1, !tbaa !62
  br label %129

124:                                              ; preds = %117
  tail call void @_ZNKSt5ctypeIcE13_M_widen_initEv(ptr noundef nonnull align 8 dereferenceable(570) %114)
  %125 = load ptr, ptr %114, align 8, !tbaa !69
  %126 = getelementptr inbounds nuw i8, ptr %125, i64 48
  %127 = load ptr, ptr %126, align 8
  %128 = tail call noundef i8 %127(ptr noundef nonnull align 8 dereferenceable(570) %114, i8 noundef 10)
  br label %129

129:                                              ; preds = %121, %124
  %130 = phi i8 [ %123, %121 ], [ %128, %124 ]
  %131 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8) %79, i8 noundef %130)
  %132 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo5flushEv(ptr noundef nonnull align 8 dereferenceable(8) %131)
  %133 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %132, ptr noundef nonnull @.str.5, i64 noundef 2)
  %134 = load ptr, ptr %0, align 8, !tbaa !113
  %135 = icmp eq ptr %134, null
  br i1 %135, label %136, label %144

136:                                              ; preds = %129
  %137 = load ptr, ptr %132, align 8, !tbaa !69
  %138 = getelementptr i8, ptr %137, i64 -24
  %139 = load i64, ptr %138, align 8
  %140 = getelementptr inbounds i8, ptr %132, i64 %139
  %141 = getelementptr inbounds nuw i8, ptr %140, i64 32
  %142 = load i32, ptr %141, align 8, !tbaa !114
  %143 = or i32 %142, 1
  tail call void @_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate(ptr noundef nonnull align 8 dereferenceable(264) %140, i32 noundef %143)
  br label %147

144:                                              ; preds = %129
  %145 = tail call noundef i64 @strlen(ptr noundef nonnull dereferenceable(1) %134) #21
  %146 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %132, ptr noundef nonnull %134, i64 noundef %145)
  br label %147

147:                                              ; preds = %136, %144
  %148 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %132, ptr noundef nonnull @.str.6, i64 noundef 1)
  %149 = load ptr, ptr %132, align 8, !tbaa !69
  %150 = getelementptr i8, ptr %149, i64 -24
  %151 = load i64, ptr %150, align 8
  %152 = getelementptr inbounds i8, ptr %132, i64 %151
  %153 = getelementptr inbounds nuw i8, ptr %152, i64 24
  %154 = load i32, ptr %153, align 8, !tbaa !115
  %155 = and i32 %154, -177
  %156 = or disjoint i32 %155, 32
  store i32 %156, ptr %153, align 8, !tbaa !116
  %157 = load i64, ptr %150, align 8
  %158 = getelementptr inbounds i8, ptr %132, i64 %157
  %159 = getelementptr inbounds nuw i8, ptr %158, i64 16
  store i64 4, ptr %159, align 8, !tbaa !71
  %160 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %132, ptr noundef nonnull @.str.11, i64 noundef 4)
  %161 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %132, ptr noundef nonnull @.str.12, i64 noundef 58)
  %162 = load ptr, ptr %132, align 8, !tbaa !69
  %163 = getelementptr i8, ptr %162, i64 -24
  %164 = load i64, ptr %163, align 8
  %165 = getelementptr inbounds i8, ptr %132, i64 %164
  %166 = getelementptr inbounds nuw i8, ptr %165, i64 240
  %167 = load ptr, ptr %166, align 8, !tbaa !89
  %168 = icmp eq ptr %167, null
  br i1 %168, label %169, label %170

169:                                              ; preds = %147
  tail call void @_ZSt16__throw_bad_castv() #20
  unreachable

170:                                              ; preds = %147
  %171 = getelementptr inbounds nuw i8, ptr %167, i64 56
  %172 = load i8, ptr %171, align 8, !tbaa !90
  %173 = icmp eq i8 %172, 0
  br i1 %173, label %177, label %174

174:                                              ; preds = %170
  %175 = getelementptr inbounds nuw i8, ptr %167, i64 67
  %176 = load i8, ptr %175, align 1, !tbaa !62
  br label %182

177:                                              ; preds = %170
  tail call void @_ZNKSt5ctypeIcE13_M_widen_initEv(ptr noundef nonnull align 8 dereferenceable(570) %167)
  %178 = load ptr, ptr %167, align 8, !tbaa !69
  %179 = getelementptr inbounds nuw i8, ptr %178, i64 48
  %180 = load ptr, ptr %179, align 8
  %181 = tail call noundef i8 %180(ptr noundef nonnull align 8 dereferenceable(570) %167, i8 noundef 10)
  br label %182

182:                                              ; preds = %174, %177
  %183 = phi i8 [ %176, %174 ], [ %181, %177 ]
  %184 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8) %132, i8 noundef %183)
  %185 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo5flushEv(ptr noundef nonnull align 8 dereferenceable(8) %184)
  %186 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %185, ptr noundef nonnull @.str.5, i64 noundef 2)
  %187 = load ptr, ptr %0, align 8, !tbaa !113
  %188 = icmp eq ptr %187, null
  br i1 %188, label %189, label %197

189:                                              ; preds = %182
  %190 = load ptr, ptr %185, align 8, !tbaa !69
  %191 = getelementptr i8, ptr %190, i64 -24
  %192 = load i64, ptr %191, align 8
  %193 = getelementptr inbounds i8, ptr %185, i64 %192
  %194 = getelementptr inbounds nuw i8, ptr %193, i64 32
  %195 = load i32, ptr %194, align 8, !tbaa !114
  %196 = or i32 %195, 1
  tail call void @_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate(ptr noundef nonnull align 8 dereferenceable(264) %193, i32 noundef %196)
  br label %200

197:                                              ; preds = %182
  %198 = tail call noundef i64 @strlen(ptr noundef nonnull dereferenceable(1) %187) #21
  %199 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %185, ptr noundef nonnull %187, i64 noundef %198)
  br label %200

200:                                              ; preds = %189, %197
  %201 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %185, ptr noundef nonnull @.str.6, i64 noundef 1)
  %202 = load ptr, ptr %185, align 8, !tbaa !69
  %203 = getelementptr i8, ptr %202, i64 -24
  %204 = load i64, ptr %203, align 8
  %205 = getelementptr inbounds i8, ptr %185, i64 %204
  %206 = getelementptr inbounds nuw i8, ptr %205, i64 24
  %207 = load i32, ptr %206, align 8, !tbaa !115
  %208 = and i32 %207, -177
  %209 = or disjoint i32 %208, 32
  store i32 %209, ptr %206, align 8, !tbaa !116
  %210 = load i64, ptr %203, align 8
  %211 = getelementptr inbounds i8, ptr %185, i64 %210
  %212 = getelementptr inbounds nuw i8, ptr %211, i64 16
  store i64 4, ptr %212, align 8, !tbaa !71
  %213 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %185, ptr noundef nonnull @.str.13, i64 noundef 4)
  %214 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %185, ptr noundef nonnull @.str.14, i64 noundef 67)
  %215 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEi(ptr noundef nonnull align 8 dereferenceable(8) %185, i32 noundef 25000)
  %216 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %215, ptr noundef nonnull @.str.15, i64 noundef 2)
  %217 = load ptr, ptr %215, align 8, !tbaa !69
  %218 = getelementptr i8, ptr %217, i64 -24
  %219 = load i64, ptr %218, align 8
  %220 = getelementptr inbounds i8, ptr %215, i64 %219
  %221 = getelementptr inbounds nuw i8, ptr %220, i64 240
  %222 = load ptr, ptr %221, align 8, !tbaa !89
  %223 = icmp eq ptr %222, null
  br i1 %223, label %224, label %225

224:                                              ; preds = %200
  tail call void @_ZSt16__throw_bad_castv() #20
  unreachable

225:                                              ; preds = %200
  %226 = getelementptr inbounds nuw i8, ptr %222, i64 56
  %227 = load i8, ptr %226, align 8, !tbaa !90
  %228 = icmp eq i8 %227, 0
  br i1 %228, label %232, label %229

229:                                              ; preds = %225
  %230 = getelementptr inbounds nuw i8, ptr %222, i64 67
  %231 = load i8, ptr %230, align 1, !tbaa !62
  br label %237

232:                                              ; preds = %225
  tail call void @_ZNKSt5ctypeIcE13_M_widen_initEv(ptr noundef nonnull align 8 dereferenceable(570) %222)
  %233 = load ptr, ptr %222, align 8, !tbaa !69
  %234 = getelementptr inbounds nuw i8, ptr %233, i64 48
  %235 = load ptr, ptr %234, align 8
  %236 = tail call noundef i8 %235(ptr noundef nonnull align 8 dereferenceable(570) %222, i8 noundef 10)
  br label %237

237:                                              ; preds = %229, %232
  %238 = phi i8 [ %231, %229 ], [ %236, %232 ]
  %239 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8) %215, i8 noundef %238)
  %240 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo5flushEv(ptr noundef nonnull align 8 dereferenceable(8) %239)
  ret void
}

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEi(ptr noundef nonnull align 8 dereferenceable(8), i32 noundef) local_unnamed_addr #3

; Function Attrs: mustprogress uwtable
define dso_local void @_Z5checkB5cxx11iPPc(ptr dead_on_unwind noalias writable sret(%"class.std::__cxx11::basic_string") align 8 %0, i32 noundef %1, ptr noundef readonly captures(none) %2) local_unnamed_addr #0 personality ptr @__gxx_personality_v0 {
  %4 = alloca i64, align 8
  %5 = alloca %"class.std::__cxx11::basic_string", align 8
  %6 = icmp slt i32 %1, 3
  br i1 %6, label %7, label %10

7:                                                ; preds = %3
  %8 = getelementptr inbounds nuw i8, ptr %0, i64 16
  store ptr %8, ptr %0, align 8, !tbaa !56
  %9 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i64 0, ptr %9, align 8, !tbaa !60
  store i8 0, ptr %8, align 8, !tbaa !62
  br label %66

10:                                               ; preds = %3
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #21
  %11 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %12 = load ptr, ptr %11, align 8, !tbaa !113
  %13 = getelementptr inbounds nuw i8, ptr %5, i64 16
  store ptr %13, ptr %5, align 8, !tbaa !56
  %14 = icmp eq ptr %12, null
  br i1 %14, label %15, label %16

15:                                               ; preds = %10
  call void @_ZSt19__throw_logic_errorPKc(ptr noundef nonnull @.str.19) #20
  unreachable

16:                                               ; preds = %10
  %17 = call noundef i64 @strlen(ptr noundef nonnull dereferenceable(1) %12) #21
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #21
  store i64 %17, ptr %4, align 8, !tbaa !29
  %18 = icmp ugt i64 %17, 15
  br i1 %18, label %19, label %22

19:                                               ; preds = %16
  %20 = call noundef ptr @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm(ptr noundef nonnull align 8 dereferenceable(32) %5, ptr noundef nonnull align 8 dereferenceable(8) %4, i64 noundef 0)
  store ptr %20, ptr %5, align 8, !tbaa !68
  %21 = load i64, ptr %4, align 8, !tbaa !29
  store i64 %21, ptr %13, align 8, !tbaa !62
  br label %22

22:                                               ; preds = %19, %16
  %23 = phi ptr [ %20, %19 ], [ %13, %16 ]
  switch i64 %17, label %26 [
    i64 1, label %24
    i64 0, label %27
  ]

24:                                               ; preds = %22
  %25 = load i8, ptr %12, align 1, !tbaa !62
  store i8 %25, ptr %23, align 1, !tbaa !62
  br label %27

26:                                               ; preds = %22
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %23, ptr nonnull align 1 %12, i64 %17, i1 false)
  br label %27

27:                                               ; preds = %26, %24, %22
  %28 = load i64, ptr %4, align 8, !tbaa !29
  %29 = getelementptr inbounds nuw i8, ptr %5, i64 8
  store i64 %28, ptr %29, align 8, !tbaa !60
  %30 = load ptr, ptr %5, align 8, !tbaa !68
  %31 = getelementptr inbounds nuw i8, ptr %30, i64 %28
  store i8 0, ptr %31, align 1, !tbaa !62
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #21
  %32 = load i64, ptr %29, align 8, !tbaa !60
  switch i64 %32, label %53 [
    i64 3, label %33
    i64 2, label %37
    i64 4, label %41
  ]

33:                                               ; preds = %27
  %34 = load ptr, ptr %5, align 8, !tbaa !68
  %35 = call i32 @bcmp(ptr noundef nonnull dereferenceable(3) %34, ptr noundef nonnull dereferenceable(3) @.str.7, i64 3)
  %36 = icmp eq i32 %35, 0
  br i1 %36, label %48, label %53

37:                                               ; preds = %27
  %38 = load ptr, ptr %5, align 8, !tbaa !68
  %39 = call i32 @bcmp(ptr %38, ptr nonnull @.str.9, i64 %32)
  %40 = icmp eq i32 %39, 0
  br i1 %40, label %48, label %53

41:                                               ; preds = %27
  %42 = load ptr, ptr %5, align 8, !tbaa !68
  %43 = call i32 @bcmp(ptr %42, ptr nonnull @.str.11, i64 %32)
  %44 = icmp eq i32 %43, 0
  br i1 %44, label %48, label %45

45:                                               ; preds = %41
  %46 = call i32 @bcmp(ptr noundef nonnull dereferenceable(4) %42, ptr noundef nonnull dereferenceable(4) @.str.13, i64 4)
  %47 = icmp eq i32 %46, 0
  br i1 %47, label %48, label %53

48:                                               ; preds = %33, %37, %41, %45
  %49 = phi ptr [ %42, %45 ], [ %42, %41 ], [ %38, %37 ], [ %34, %33 ]
  %50 = getelementptr inbounds nuw i8, ptr %0, i64 16
  store ptr %50, ptr %0, align 8, !tbaa !56
  call void @llvm.memcpy.p0.p0.i64(ptr nonnull align 8 %50, ptr align 1 %49, i64 %32, i1 false)
  %51 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i64 %32, ptr %51, align 8, !tbaa !60
  %52 = getelementptr inbounds nuw i8, ptr %50, i64 %32
  store i8 0, ptr %52, align 1, !tbaa !62
  br label %56

53:                                               ; preds = %37, %33, %27, %45
  %54 = getelementptr inbounds nuw i8, ptr %0, i64 16
  store ptr %54, ptr %0, align 8, !tbaa !56
  %55 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i64 0, ptr %55, align 8, !tbaa !60
  store i8 0, ptr %54, align 8, !tbaa !62
  br label %56

56:                                               ; preds = %48, %53
  %57 = load ptr, ptr %5, align 8, !tbaa !68
  %58 = icmp eq ptr %57, %13
  br i1 %58, label %59, label %62

59:                                               ; preds = %56
  %60 = load i64, ptr %29, align 8, !tbaa !60
  %61 = icmp ult i64 %60, 16
  call void @llvm.assume(i1 %61)
  br label %65

62:                                               ; preds = %56
  %63 = load i64, ptr %13, align 8, !tbaa !62
  %64 = add i64 %63, 1
  call void @_ZdlPvm(ptr noundef %57, i64 noundef %64) #18
  br label %65

65:                                               ; preds = %59, %62
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #21
  br label %66

66:                                               ; preds = %65, %7
  ret void
}

; Function Attrs: mustprogress norecurse uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #4 personality ptr @__gxx_personality_v0 {
  %3 = alloca %class.BigInt, align 8
  %4 = alloca %class.BigInt, align 8
  %5 = alloca %class.BigInt, align 8
  %6 = alloca %class.BigInt, align 8
  %7 = alloca %"class.std::__cxx11::basic_string", align 8
  %8 = alloca %class.Fibonacci, align 8
  %9 = alloca %class.Fibonacci, align 8
  %10 = alloca %class.Fibonacci, align 8
  %11 = alloca %class.Fibonacci, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %7) #21
  call void @_Z5checkB5cxx11iPPc(ptr dead_on_unwind nonnull writable sret(%"class.std::__cxx11::basic_string") align 8 %7, i32 noundef %0, ptr noundef %1)
  %12 = getelementptr inbounds nuw i8, ptr %7, i64 8
  %13 = load i64, ptr %12, align 8, !tbaa !60
  %14 = icmp eq i64 %13, 0
  br i1 %14, label %15, label %19

15:                                               ; preds = %2
  %16 = invoke noundef nonnull align 8 dereferenceable(32) ptr @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_replaceEmmPKcm(ptr noundef nonnull align 8 dereferenceable(32) %7, i64 noundef 0, i64 noundef 0, ptr noundef nonnull @.str.9, i64 noundef 2)
          to label %24 unwind label %17

17:                                               ; preds = %15
  %18 = landingpad { ptr, i32 }
          cleanup
  br label %287

19:                                               ; preds = %2
  %20 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %21 = load ptr, ptr %20, align 8, !tbaa !113
  %22 = call i64 @__isoc23_strtol(ptr noundef nonnull %21, ptr noundef null, i32 noundef 10) #21
  %23 = trunc i64 %22 to i32
  br label %24

24:                                               ; preds = %15, %19
  %25 = phi i32 [ %23, %19 ], [ 50000, %15 ]
  %26 = load i64, ptr %12, align 8, !tbaa !60
  %27 = icmp eq i64 %26, 3
  %28 = load ptr, ptr %7, align 8, !tbaa !68
  br i1 %27, label %29, label %81

29:                                               ; preds = %24
  %30 = call i32 @bcmp(ptr noundef nonnull dereferenceable(3) %28, ptr noundef nonnull dereferenceable(3) @.str.7, i64 3)
  %31 = icmp eq i32 %30, 0
  br i1 %31, label %32, label %276

32:                                               ; preds = %29
  call void @llvm.lifetime.start.p0(ptr nonnull %8) #21
  call void @llvm.lifetime.start.p0(ptr nonnull %6)
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(24) %8, i8 0, i64 24, i1 false)
  invoke void @_ZN9Fibonacci10get_numberEj(ptr dead_on_unwind nonnull writable sret(%class.BigInt) align 8 %6, ptr noundef nonnull align 8 dereferenceable(24) %8, i32 noundef %25)
          to label %33 unwind label %42

33:                                               ; preds = %32
  %34 = load ptr, ptr %6, align 8, !tbaa !26
  %35 = icmp eq ptr %34, null
  br i1 %35, label %44, label %36

36:                                               ; preds = %33
  %37 = getelementptr inbounds nuw i8, ptr %6, i64 16
  %38 = load ptr, ptr %37, align 8, !tbaa !21
  %39 = ptrtoint ptr %38 to i64
  %40 = ptrtoint ptr %34 to i64
  %41 = sub i64 %39, %40
  call void @_ZdlPvm(ptr noundef nonnull %34, i64 noundef %41) #18
  br label %44

42:                                               ; preds = %32
  %43 = landingpad { ptr, i32 }
          cleanup
  call void @_ZNSt6vectorI6BigIntSaIS0_EED2Ev(ptr noundef nonnull align 8 dereferenceable(24) %8) #21
  br label %79

44:                                               ; preds = %36, %33
  call void @llvm.lifetime.end.p0(ptr nonnull %6)
  invoke void @_ZNK9Fibonacci16show_all_numbersEv(ptr noundef nonnull align 8 dereferenceable(24) %8)
          to label %45 unwind label %77

45:                                               ; preds = %44
  %46 = load ptr, ptr %8, align 8, !tbaa !12
  %47 = getelementptr inbounds nuw i8, ptr %8, i64 8
  %48 = load ptr, ptr %47, align 8, !tbaa !13
  %49 = icmp eq ptr %46, %48
  br i1 %49, label %65, label %50

50:                                               ; preds = %45, %60
  %51 = phi ptr [ %61, %60 ], [ %46, %45 ]
  %52 = load ptr, ptr %51, align 8, !tbaa !26
  %53 = icmp eq ptr %52, null
  br i1 %53, label %60, label %54

54:                                               ; preds = %50
  %55 = getelementptr inbounds nuw i8, ptr %51, i64 16
  %56 = load ptr, ptr %55, align 8, !tbaa !21
  %57 = ptrtoint ptr %56 to i64
  %58 = ptrtoint ptr %52 to i64
  %59 = sub i64 %57, %58
  call void @_ZdlPvm(ptr noundef nonnull %52, i64 noundef %59) #18
  br label %60

60:                                               ; preds = %54, %50
  %61 = getelementptr inbounds nuw i8, ptr %51, i64 24
  %62 = icmp eq ptr %61, %48
  br i1 %62, label %63, label %50, !llvm.loop !117

63:                                               ; preds = %60
  %64 = load ptr, ptr %8, align 8, !tbaa !12
  br label %65

65:                                               ; preds = %63, %45
  %66 = phi ptr [ %64, %63 ], [ %46, %45 ]
  %67 = icmp eq ptr %66, null
  br i1 %67, label %74, label %68

68:                                               ; preds = %65
  %69 = getelementptr inbounds nuw i8, ptr %8, i64 16
  %70 = load ptr, ptr %69, align 8, !tbaa !6
  %71 = ptrtoint ptr %70 to i64
  %72 = ptrtoint ptr %66 to i64
  %73 = sub i64 %71, %72
  call void @_ZdlPvm(ptr noundef nonnull %66, i64 noundef %73) #18
  br label %74

74:                                               ; preds = %65, %68
  call void @llvm.lifetime.end.p0(ptr nonnull %8) #21
  %75 = load i64, ptr %12, align 8, !tbaa !60
  %76 = load ptr, ptr %7, align 8, !tbaa !68
  br label %81

77:                                               ; preds = %44
  %78 = landingpad { ptr, i32 }
          cleanup
  call void @_ZN9FibonacciD2Ev(ptr noundef nonnull align 8 dereferenceable(24) %8) #21
  br label %79

79:                                               ; preds = %42, %77
  %80 = phi { ptr, i32 } [ %78, %77 ], [ %43, %42 ]
  call void @llvm.lifetime.end.p0(ptr nonnull %8) #21
  br label %287

81:                                               ; preds = %74, %24
  %82 = phi ptr [ %28, %24 ], [ %76, %74 ]
  %83 = phi i64 [ %26, %24 ], [ %75, %74 ]
  %84 = icmp eq i64 %83, 2
  br i1 %84, label %85, label %137

85:                                               ; preds = %81
  %86 = call i32 @bcmp(ptr noundef nonnull dereferenceable(2) %82, ptr noundef nonnull dereferenceable(2) @.str.9, i64 2)
  %87 = icmp eq i32 %86, 0
  br i1 %87, label %88, label %276

88:                                               ; preds = %85
  call void @llvm.lifetime.start.p0(ptr nonnull %9) #21
  call void @llvm.lifetime.start.p0(ptr nonnull %5)
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(24) %9, i8 0, i64 24, i1 false)
  invoke void @_ZN9Fibonacci10get_numberEj(ptr dead_on_unwind nonnull writable sret(%class.BigInt) align 8 %5, ptr noundef nonnull align 8 dereferenceable(24) %9, i32 noundef %25)
          to label %89 unwind label %98

89:                                               ; preds = %88
  %90 = load ptr, ptr %5, align 8, !tbaa !26
  %91 = icmp eq ptr %90, null
  br i1 %91, label %100, label %92

92:                                               ; preds = %89
  %93 = getelementptr inbounds nuw i8, ptr %5, i64 16
  %94 = load ptr, ptr %93, align 8, !tbaa !21
  %95 = ptrtoint ptr %94 to i64
  %96 = ptrtoint ptr %90 to i64
  %97 = sub i64 %95, %96
  call void @_ZdlPvm(ptr noundef nonnull %90, i64 noundef %97) #18
  br label %100

98:                                               ; preds = %88
  %99 = landingpad { ptr, i32 }
          cleanup
  call void @_ZNSt6vectorI6BigIntSaIS0_EED2Ev(ptr noundef nonnull align 8 dereferenceable(24) %9) #21
  br label %135

100:                                              ; preds = %92, %89
  call void @llvm.lifetime.end.p0(ptr nonnull %5)
  invoke void @_ZNK9Fibonacci16show_last_numberEv(ptr noundef nonnull align 8 dereferenceable(24) %9)
          to label %101 unwind label %133

101:                                              ; preds = %100
  %102 = load ptr, ptr %9, align 8, !tbaa !12
  %103 = getelementptr inbounds nuw i8, ptr %9, i64 8
  %104 = load ptr, ptr %103, align 8, !tbaa !13
  %105 = icmp eq ptr %102, %104
  br i1 %105, label %121, label %106

106:                                              ; preds = %101, %116
  %107 = phi ptr [ %117, %116 ], [ %102, %101 ]
  %108 = load ptr, ptr %107, align 8, !tbaa !26
  %109 = icmp eq ptr %108, null
  br i1 %109, label %116, label %110

110:                                              ; preds = %106
  %111 = getelementptr inbounds nuw i8, ptr %107, i64 16
  %112 = load ptr, ptr %111, align 8, !tbaa !21
  %113 = ptrtoint ptr %112 to i64
  %114 = ptrtoint ptr %108 to i64
  %115 = sub i64 %113, %114
  call void @_ZdlPvm(ptr noundef nonnull %108, i64 noundef %115) #18
  br label %116

116:                                              ; preds = %110, %106
  %117 = getelementptr inbounds nuw i8, ptr %107, i64 24
  %118 = icmp eq ptr %117, %104
  br i1 %118, label %119, label %106, !llvm.loop !117

119:                                              ; preds = %116
  %120 = load ptr, ptr %9, align 8, !tbaa !12
  br label %121

121:                                              ; preds = %119, %101
  %122 = phi ptr [ %120, %119 ], [ %102, %101 ]
  %123 = icmp eq ptr %122, null
  br i1 %123, label %130, label %124

124:                                              ; preds = %121
  %125 = getelementptr inbounds nuw i8, ptr %9, i64 16
  %126 = load ptr, ptr %125, align 8, !tbaa !6
  %127 = ptrtoint ptr %126 to i64
  %128 = ptrtoint ptr %122 to i64
  %129 = sub i64 %127, %128
  call void @_ZdlPvm(ptr noundef nonnull %122, i64 noundef %129) #18
  br label %130

130:                                              ; preds = %121, %124
  call void @llvm.lifetime.end.p0(ptr nonnull %9) #21
  %131 = load i64, ptr %12, align 8, !tbaa !60
  %132 = load ptr, ptr %7, align 8, !tbaa !68
  br label %137

133:                                              ; preds = %100
  %134 = landingpad { ptr, i32 }
          cleanup
  call void @_ZN9FibonacciD2Ev(ptr noundef nonnull align 8 dereferenceable(24) %9) #21
  br label %135

135:                                              ; preds = %98, %133
  %136 = phi { ptr, i32 } [ %134, %133 ], [ %99, %98 ]
  call void @llvm.lifetime.end.p0(ptr nonnull %9) #21
  br label %287

137:                                              ; preds = %81, %130
  %138 = phi ptr [ %82, %81 ], [ %132, %130 ]
  %139 = phi i64 [ %83, %81 ], [ %131, %130 ]
  %140 = icmp eq i64 %139, 4
  br i1 %140, label %141, label %276

141:                                              ; preds = %137
  %142 = call i32 @bcmp(ptr noundef nonnull dereferenceable(4) %138, ptr noundef nonnull dereferenceable(4) @.str.11, i64 4)
  %143 = icmp eq i32 %142, 0
  br i1 %143, label %144, label %207

144:                                              ; preds = %141
  call void @llvm.lifetime.start.p0(ptr nonnull %10) #21
  call void @llvm.lifetime.start.p0(ptr nonnull %4)
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(24) %10, i8 0, i64 24, i1 false)
  invoke void @_ZN9Fibonacci10get_numberEj(ptr dead_on_unwind nonnull writable sret(%class.BigInt) align 8 %4, ptr noundef nonnull align 8 dereferenceable(24) %10, i32 noundef 0)
          to label %145 unwind label %154

145:                                              ; preds = %144
  %146 = load ptr, ptr %4, align 8, !tbaa !26
  %147 = icmp eq ptr %146, null
  br i1 %147, label %156, label %148

148:                                              ; preds = %145
  %149 = getelementptr inbounds nuw i8, ptr %4, i64 16
  %150 = load ptr, ptr %149, align 8, !tbaa !21
  %151 = ptrtoint ptr %150 to i64
  %152 = ptrtoint ptr %146 to i64
  %153 = sub i64 %151, %152
  call void @_ZdlPvm(ptr noundef nonnull %146, i64 noundef %153) #18
  br label %156

154:                                              ; preds = %144
  %155 = landingpad { ptr, i32 }
          cleanup
  call void @_ZNSt6vectorI6BigIntSaIS0_EED2Ev(ptr noundef nonnull align 8 dereferenceable(24) %10) #21
  br label %201

156:                                              ; preds = %145, %148
  call void @llvm.lifetime.end.p0(ptr nonnull %4)
  %157 = icmp sgt i32 %0, 2
  br i1 %157, label %158, label %160

158:                                              ; preds = %156
  %159 = zext nneg i32 %0 to i64
  br label %189

160:                                              ; preds = %196, %156
  %161 = load ptr, ptr %10, align 8, !tbaa !12
  %162 = getelementptr inbounds nuw i8, ptr %10, i64 8
  %163 = load ptr, ptr %162, align 8, !tbaa !13
  %164 = icmp eq ptr %161, %163
  br i1 %164, label %180, label %165

165:                                              ; preds = %160, %175
  %166 = phi ptr [ %176, %175 ], [ %161, %160 ]
  %167 = load ptr, ptr %166, align 8, !tbaa !26
  %168 = icmp eq ptr %167, null
  br i1 %168, label %175, label %169

169:                                              ; preds = %165
  %170 = getelementptr inbounds nuw i8, ptr %166, i64 16
  %171 = load ptr, ptr %170, align 8, !tbaa !21
  %172 = ptrtoint ptr %171 to i64
  %173 = ptrtoint ptr %167 to i64
  %174 = sub i64 %172, %173
  call void @_ZdlPvm(ptr noundef nonnull %167, i64 noundef %174) #18
  br label %175

175:                                              ; preds = %169, %165
  %176 = getelementptr inbounds nuw i8, ptr %166, i64 24
  %177 = icmp eq ptr %176, %163
  br i1 %177, label %178, label %165, !llvm.loop !117

178:                                              ; preds = %175
  %179 = load ptr, ptr %10, align 8, !tbaa !12
  br label %180

180:                                              ; preds = %178, %160
  %181 = phi ptr [ %179, %178 ], [ %161, %160 ]
  %182 = icmp eq ptr %181, null
  br i1 %182, label %203, label %183

183:                                              ; preds = %180
  %184 = getelementptr inbounds nuw i8, ptr %10, i64 16
  %185 = load ptr, ptr %184, align 8, !tbaa !6
  %186 = ptrtoint ptr %185 to i64
  %187 = ptrtoint ptr %181 to i64
  %188 = sub i64 %186, %187
  call void @_ZdlPvm(ptr noundef nonnull %181, i64 noundef %188) #18
  br label %203

189:                                              ; preds = %158, %196
  %190 = phi i64 [ 2, %158 ], [ %197, %196 ]
  %191 = getelementptr inbounds nuw ptr, ptr %1, i64 %190
  %192 = load ptr, ptr %191, align 8, !tbaa !113
  %193 = call i64 @__isoc23_strtol(ptr noundef nonnull %192, ptr noundef null, i32 noundef 10) #21
  %194 = shl i64 %193, 32
  %195 = ashr exact i64 %194, 32
  invoke void @_ZN9Fibonacci11show_numberEm(ptr noundef nonnull align 8 dereferenceable(24) %10, i64 noundef %195)
          to label %196 unwind label %199

196:                                              ; preds = %189
  %197 = add nuw nsw i64 %190, 1
  %198 = icmp eq i64 %197, %159
  br i1 %198, label %160, label %189, !llvm.loop !118

199:                                              ; preds = %189
  %200 = landingpad { ptr, i32 }
          cleanup
  call void @_ZN9FibonacciD2Ev(ptr noundef nonnull align 8 dereferenceable(24) %10) #21
  br label %201

201:                                              ; preds = %154, %199
  %202 = phi { ptr, i32 } [ %200, %199 ], [ %155, %154 ]
  call void @llvm.lifetime.end.p0(ptr nonnull %10) #21
  br label %287

203:                                              ; preds = %183, %180
  call void @llvm.lifetime.end.p0(ptr nonnull %10) #21
  %204 = load i64, ptr %12, align 8, !tbaa !60
  %205 = load ptr, ptr %7, align 8, !tbaa !68
  %206 = icmp eq i64 %204, 4
  br i1 %206, label %207, label %276

207:                                              ; preds = %141, %203
  %208 = phi ptr [ %205, %203 ], [ %138, %141 ]
  %209 = call i32 @bcmp(ptr noundef nonnull dereferenceable(4) %208, ptr noundef nonnull dereferenceable(4) @.str.13, i64 4)
  %210 = icmp eq i32 %209, 0
  br i1 %210, label %211, label %276

211:                                              ; preds = %207
  %212 = icmp eq i32 %0, 3
  br i1 %212, label %218, label %213

213:                                              ; preds = %211
  %214 = getelementptr inbounds nuw i8, ptr %1, i64 24
  %215 = load ptr, ptr %214, align 8, !tbaa !113
  %216 = call i64 @__isoc23_strtol(ptr noundef nonnull %215, ptr noundef null, i32 noundef 10) #21
  %217 = trunc i64 %216 to i32
  br label %218

218:                                              ; preds = %211, %213
  %219 = phi i32 [ %217, %213 ], [ 25000, %211 ]
  call void @llvm.lifetime.start.p0(ptr nonnull %11) #21
  call void @llvm.lifetime.start.p0(ptr nonnull %3)
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(24) %11, i8 0, i64 24, i1 false)
  invoke void @_ZN9Fibonacci10get_numberEj(ptr dead_on_unwind nonnull writable sret(%class.BigInt) align 8 %3, ptr noundef nonnull align 8 dereferenceable(24) %11, i32 noundef 0)
          to label %220 unwind label %229

220:                                              ; preds = %218
  %221 = load ptr, ptr %3, align 8, !tbaa !26
  %222 = icmp eq ptr %221, null
  br i1 %222, label %231, label %223

223:                                              ; preds = %220
  %224 = getelementptr inbounds nuw i8, ptr %3, i64 16
  %225 = load ptr, ptr %224, align 8, !tbaa !21
  %226 = ptrtoint ptr %225 to i64
  %227 = ptrtoint ptr %221 to i64
  %228 = sub i64 %226, %227
  call void @_ZdlPvm(ptr noundef nonnull %221, i64 noundef %228) #18
  br label %231

229:                                              ; preds = %218
  %230 = landingpad { ptr, i32 }
          cleanup
  call void @_ZNSt6vectorI6BigIntSaIS0_EED2Ev(ptr noundef nonnull align 8 dereferenceable(24) %11) #21
  br label %274

231:                                              ; preds = %220, %223
  call void @llvm.lifetime.end.p0(ptr nonnull %3)
  %232 = icmp eq i32 %25, 0
  br i1 %232, label %233, label %264

233:                                              ; preds = %269, %231
  %234 = load ptr, ptr %11, align 8, !tbaa !12
  %235 = getelementptr inbounds nuw i8, ptr %11, i64 8
  %236 = load ptr, ptr %235, align 8, !tbaa !13
  %237 = icmp eq ptr %234, %236
  br i1 %237, label %253, label %238

238:                                              ; preds = %233, %248
  %239 = phi ptr [ %249, %248 ], [ %234, %233 ]
  %240 = load ptr, ptr %239, align 8, !tbaa !26
  %241 = icmp eq ptr %240, null
  br i1 %241, label %248, label %242

242:                                              ; preds = %238
  %243 = getelementptr inbounds nuw i8, ptr %239, i64 16
  %244 = load ptr, ptr %243, align 8, !tbaa !21
  %245 = ptrtoint ptr %244 to i64
  %246 = ptrtoint ptr %240 to i64
  %247 = sub i64 %245, %246
  call void @_ZdlPvm(ptr noundef nonnull %240, i64 noundef %247) #18
  br label %248

248:                                              ; preds = %242, %238
  %249 = getelementptr inbounds nuw i8, ptr %239, i64 24
  %250 = icmp eq ptr %249, %236
  br i1 %250, label %251, label %238, !llvm.loop !117

251:                                              ; preds = %248
  %252 = load ptr, ptr %11, align 8, !tbaa !12
  br label %253

253:                                              ; preds = %251, %233
  %254 = phi ptr [ %252, %251 ], [ %234, %233 ]
  %255 = icmp eq ptr %254, null
  br i1 %255, label %262, label %256

256:                                              ; preds = %253
  %257 = getelementptr inbounds nuw i8, ptr %11, i64 16
  %258 = load ptr, ptr %257, align 8, !tbaa !6
  %259 = ptrtoint ptr %258 to i64
  %260 = ptrtoint ptr %254 to i64
  %261 = sub i64 %259, %260
  call void @_ZdlPvm(ptr noundef nonnull %254, i64 noundef %261) #18
  br label %262

262:                                              ; preds = %253, %256
  call void @llvm.lifetime.end.p0(ptr nonnull %11) #21
  %263 = load ptr, ptr %7, align 8, !tbaa !68
  br label %276

264:                                              ; preds = %231, %269
  %265 = phi i32 [ %270, %269 ], [ 0, %231 ]
  %266 = call i32 @rand() #21
  %267 = srem i32 %266, %219
  %268 = sext i32 %267 to i64
  invoke void @_ZN9Fibonacci11show_numberEm(ptr noundef nonnull align 8 dereferenceable(24) %11, i64 noundef %268)
          to label %269 unwind label %272

269:                                              ; preds = %264
  %270 = add nuw i32 %265, 1
  %271 = icmp eq i32 %270, %25
  br i1 %271, label %233, label %264, !llvm.loop !119

272:                                              ; preds = %264
  %273 = landingpad { ptr, i32 }
          cleanup
  call void @_ZN9FibonacciD2Ev(ptr noundef nonnull align 8 dereferenceable(24) %11) #21
  br label %274

274:                                              ; preds = %229, %272
  %275 = phi { ptr, i32 } [ %273, %272 ], [ %230, %229 ]
  call void @llvm.lifetime.end.p0(ptr nonnull %11) #21
  br label %287

276:                                              ; preds = %29, %85, %137, %203, %262, %207
  %277 = phi ptr [ %205, %203 ], [ %263, %262 ], [ %208, %207 ], [ %138, %137 ], [ %82, %85 ], [ %28, %29 ]
  %278 = getelementptr inbounds nuw i8, ptr %7, i64 16
  %279 = icmp eq ptr %277, %278
  br i1 %279, label %280, label %283

280:                                              ; preds = %276
  %281 = load i64, ptr %12, align 8, !tbaa !60
  %282 = icmp ult i64 %281, 16
  call void @llvm.assume(i1 %282)
  br label %286

283:                                              ; preds = %276
  %284 = load i64, ptr %278, align 8, !tbaa !62
  %285 = add i64 %284, 1
  call void @_ZdlPvm(ptr noundef %277, i64 noundef %285) #18
  br label %286

286:                                              ; preds = %280, %283
  call void @llvm.lifetime.end.p0(ptr nonnull %7) #21
  ret i32 0

287:                                              ; preds = %274, %201, %135, %79, %17
  %288 = phi { ptr, i32 } [ %202, %201 ], [ %275, %274 ], [ %18, %17 ], [ %136, %135 ], [ %80, %79 ]
  %289 = load ptr, ptr %7, align 8, !tbaa !68
  %290 = getelementptr inbounds nuw i8, ptr %7, i64 16
  %291 = icmp eq ptr %289, %290
  br i1 %291, label %292, label %295

292:                                              ; preds = %287
  %293 = load i64, ptr %12, align 8, !tbaa !60
  %294 = icmp ult i64 %293, 16
  call void @llvm.assume(i1 %294)
  br label %298

295:                                              ; preds = %287
  %296 = load i64, ptr %290, align 8, !tbaa !62
  %297 = add i64 %296, 1
  call void @_ZdlPvm(ptr noundef %289, i64 noundef %297) #18
  br label %298

298:                                              ; preds = %292, %295
  call void @llvm.lifetime.end.p0(ptr nonnull %7) #21
  resume { ptr, i32 } %288
}

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local void @_ZN9FibonacciD2Ev(ptr noundef nonnull align 8 dereferenceable(24) %0) unnamed_addr #2 comdat personality ptr @__gxx_personality_v0 {
  %2 = load ptr, ptr %0, align 8, !tbaa !12
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %4 = load ptr, ptr %3, align 8, !tbaa !13
  %5 = icmp eq ptr %2, %4
  br i1 %5, label %21, label %6

6:                                                ; preds = %1, %16
  %7 = phi ptr [ %17, %16 ], [ %2, %1 ]
  %8 = load ptr, ptr %7, align 8, !tbaa !26
  %9 = icmp eq ptr %8, null
  br i1 %9, label %16, label %10

10:                                               ; preds = %6
  %11 = getelementptr inbounds nuw i8, ptr %7, i64 16
  %12 = load ptr, ptr %11, align 8, !tbaa !21
  %13 = ptrtoint ptr %12 to i64
  %14 = ptrtoint ptr %8 to i64
  %15 = sub i64 %13, %14
  tail call void @_ZdlPvm(ptr noundef nonnull %8, i64 noundef %15) #18
  br label %16

16:                                               ; preds = %10, %6
  %17 = getelementptr inbounds nuw i8, ptr %7, i64 24
  %18 = icmp eq ptr %17, %4
  br i1 %18, label %19, label %6, !llvm.loop !117

19:                                               ; preds = %16
  %20 = load ptr, ptr %0, align 8, !tbaa !12
  br label %21

21:                                               ; preds = %19, %1
  %22 = phi ptr [ %20, %19 ], [ %2, %1 ]
  %23 = icmp eq ptr %22, null
  br i1 %23, label %30, label %24

24:                                               ; preds = %21
  %25 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %26 = load ptr, ptr %25, align 8, !tbaa !6
  %27 = ptrtoint ptr %26 to i64
  %28 = ptrtoint ptr %22 to i64
  %29 = sub i64 %27, %28
  tail call void @_ZdlPvm(ptr noundef nonnull %22, i64 noundef %29) #18
  br label %30

30:                                               ; preds = %21, %24
  ret void
}

; Function Attrs: nounwind
declare i32 @rand() local_unnamed_addr #5

; Function Attrs: cold noreturn
declare void @_ZSt20__throw_length_errorPKc(ptr noundef) local_unnamed_addr #6

; Function Attrs: noreturn
declare void @_ZSt28__throw_bad_array_new_lengthv() local_unnamed_addr #7

; Function Attrs: nobuiltin allocsize(0)
declare noundef nonnull ptr @_Znwm(i64 noundef) local_unnamed_addr #8

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #9

; Function Attrs: nobuiltin nounwind
declare void @_ZdlPvm(ptr noundef, i64 noundef) local_unnamed_addr #10

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZNSt6vectorImSaImEE17_M_default_appendEm(ptr noundef nonnull align 8 dereferenceable(24) %0, i64 noundef %1) local_unnamed_addr #0 comdat personality ptr @__gxx_personality_v0 {
  %3 = icmp eq i64 %1, 0
  br i1 %3, label %57, label %4

4:                                                ; preds = %2
  %5 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %6 = load ptr, ptr %5, align 8, !tbaa !25
  %7 = load ptr, ptr %0, align 8, !tbaa !26
  %8 = ptrtoint ptr %6 to i64
  %9 = ptrtoint ptr %7 to i64
  %10 = sub i64 %8, %9
  %11 = ashr exact i64 %10, 3
  %12 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %13 = load ptr, ptr %12, align 8, !tbaa !21
  %14 = ptrtoint ptr %13 to i64
  %15 = sub i64 %14, %8
  %16 = ashr exact i64 %15, 3
  %17 = icmp ult i64 %11, 1152921504606846976
  tail call void @llvm.assume(i1 %17)
  %18 = xor i64 %11, 1152921504606846975
  %19 = icmp ule i64 %16, %18
  tail call void @llvm.assume(i1 %19)
  %20 = icmp ult i64 %16, %1
  br i1 %20, label %32, label %21

21:                                               ; preds = %4
  store i64 0, ptr %6, align 8, !tbaa !29
  %22 = getelementptr i8, ptr %6, i64 8
  %23 = add nsw i64 %1, -1
  %24 = icmp eq i64 %23, 0
  br i1 %24, label %30, label %25

25:                                               ; preds = %21
  %26 = shl nuw nsw i64 %1, 3
  %27 = add nsw i64 %26, -8
  tail call void @llvm.memset.p0.i64(ptr align 8 %22, i8 0, i64 %27, i1 false), !tbaa !29
  %28 = shl nuw nsw i64 %23, 3
  %29 = getelementptr inbounds nuw i8, ptr %22, i64 %28
  br label %30

30:                                               ; preds = %21, %25
  %31 = phi ptr [ %22, %21 ], [ %29, %25 ]
  store ptr %31, ptr %5, align 8, !tbaa !25
  br label %57

32:                                               ; preds = %4
  %33 = icmp ult i64 %18, %1
  br i1 %33, label %34, label %35

34:                                               ; preds = %32
  tail call void @_ZSt20__throw_length_errorPKc(ptr noundef nonnull @.str.17) #20
  unreachable

35:                                               ; preds = %32
  %36 = tail call i64 @llvm.umax.i64(i64 %11, i64 %1)
  %37 = add nuw nsw i64 %36, %11
  %38 = tail call i64 @llvm.umin.i64(i64 %37, i64 1152921504606846975)
  %39 = shl nuw nsw i64 %38, 3
  %40 = tail call noalias noundef nonnull ptr @_Znwm(i64 noundef %39) #17
  %41 = getelementptr inbounds nuw i8, ptr %40, i64 %10
  store i64 0, ptr %41, align 8, !tbaa !29
  %42 = icmp eq i64 %1, 1
  br i1 %42, label %47, label %43

43:                                               ; preds = %35
  %44 = getelementptr i8, ptr %41, i64 8
  %45 = shl nuw nsw i64 %1, 3
  %46 = add nsw i64 %45, -8
  tail call void @llvm.memset.p0.i64(ptr align 8 %44, i8 0, i64 %46, i1 false), !tbaa !29
  br label %47

47:                                               ; preds = %43, %35
  %48 = icmp sgt i64 %10, 0
  br i1 %48, label %49, label %50

49:                                               ; preds = %47
  tail call void @llvm.memcpy.p0.p0.i64(ptr nonnull align 8 %40, ptr align 8 %7, i64 %10, i1 false)
  br label %50

50:                                               ; preds = %47, %49
  %51 = icmp eq ptr %7, null
  br i1 %51, label %54, label %52

52:                                               ; preds = %50
  %53 = sub i64 %14, %9
  tail call void @_ZdlPvm(ptr noundef nonnull %7, i64 noundef %53) #18
  br label %54

54:                                               ; preds = %50, %52
  store ptr %40, ptr %0, align 8, !tbaa !26
  %55 = getelementptr inbounds nuw i64, ptr %41, i64 %1
  store ptr %55, ptr %5, align 8, !tbaa !25
  %56 = getelementptr inbounds nuw i64, ptr %40, i64 %38
  store ptr %56, ptr %12, align 8, !tbaa !21
  br label %57

57:                                               ; preds = %30, %54, %2
  ret void
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memmove.p0.p0.i64(ptr writeonly captures(none), ptr readonly captures(none), i64, i1 immarg) #9

; Function Attrs: cold noreturn
declare void @_ZSt16__throw_bad_castv() local_unnamed_addr #6

declare void @_ZNKSt5ctypeIcE13_M_widen_initEv(ptr noundef nonnull align 8 dereferenceable(570)) local_unnamed_addr #3

; Function Attrs: nounwind
declare i64 @__isoc23_strtol(ptr noundef, ptr noundef, i32 noundef) local_unnamed_addr #5

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local void @_ZNSt6vectorI6BigIntSaIS0_EED2Ev(ptr noundef nonnull align 8 dereferenceable(24) %0) unnamed_addr #2 comdat personality ptr @__gxx_personality_v0 {
  %2 = load ptr, ptr %0, align 8, !tbaa !12
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %4 = load ptr, ptr %3, align 8, !tbaa !13
  %5 = icmp eq ptr %2, %4
  br i1 %5, label %21, label %6

6:                                                ; preds = %1, %16
  %7 = phi ptr [ %17, %16 ], [ %2, %1 ]
  %8 = load ptr, ptr %7, align 8, !tbaa !26
  %9 = icmp eq ptr %8, null
  br i1 %9, label %16, label %10

10:                                               ; preds = %6
  %11 = getelementptr inbounds nuw i8, ptr %7, i64 16
  %12 = load ptr, ptr %11, align 8, !tbaa !21
  %13 = ptrtoint ptr %12 to i64
  %14 = ptrtoint ptr %8 to i64
  %15 = sub i64 %13, %14
  tail call void @_ZdlPvm(ptr noundef nonnull %8, i64 noundef %15) #18
  br label %16

16:                                               ; preds = %10, %6
  %17 = getelementptr inbounds nuw i8, ptr %7, i64 24
  %18 = icmp eq ptr %17, %4
  br i1 %18, label %19, label %6, !llvm.loop !117

19:                                               ; preds = %16
  %20 = load ptr, ptr %0, align 8, !tbaa !12
  br label %21

21:                                               ; preds = %19, %1
  %22 = phi ptr [ %20, %19 ], [ %2, %1 ]
  %23 = icmp eq ptr %22, null
  br i1 %23, label %30, label %24

24:                                               ; preds = %21
  %25 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %26 = load ptr, ptr %25, align 8, !tbaa !6
  %27 = ptrtoint ptr %26 to i64
  %28 = ptrtoint ptr %22 to i64
  %29 = sub i64 %27, %28
  tail call void @_ZdlPvm(ptr noundef nonnull %22, i64 noundef %29) #18
  br label %30

30:                                               ; preds = %21, %24
  ret void
}

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo9_M_insertImEERSoT_(ptr noundef nonnull align 8 dereferenceable(8), i64 noundef) local_unnamed_addr #3

; Function Attrs: nounwind
declare void @_ZNSt8ios_baseD2Ev(ptr noundef nonnull align 8 dereferenceable(216)) unnamed_addr #5

; Function Attrs: nounwind
declare void @_ZNSt6localeD1Ev(ptr noundef nonnull align 8 dereferenceable(8)) unnamed_addr #5

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8), ptr noundef, i64 noundef) local_unnamed_addr #3

declare void @_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate(ptr noundef nonnull align 8 dereferenceable(264), i32 noundef) local_unnamed_addr #3

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: read)
declare i64 @strlen(ptr noundef captures(none)) local_unnamed_addr #11

declare noundef nonnull align 8 dereferenceable(32) ptr @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_replaceEmmPKcm(ptr noundef nonnull align 8 dereferenceable(32), i64 noundef, i64 noundef, ptr noundef, i64 noundef) local_unnamed_addr #3

declare void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_assignERKS4_(ptr noundef nonnull align 8 dereferenceable(32), ptr noundef nonnull align 8 dereferenceable(32)) local_unnamed_addr #3

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8), i8 noundef) local_unnamed_addr #3

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo5flushEv(ptr noundef nonnull align 8 dereferenceable(8)) local_unnamed_addr #3

; Function Attrs: cold noreturn
declare void @_ZSt19__throw_logic_errorPKc(ptr noundef) local_unnamed_addr #6

declare noundef ptr @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm(ptr noundef nonnull align 8 dereferenceable(32), ptr noundef nonnull align 8 dereferenceable(8), i64 noundef) local_unnamed_addr #3

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #12

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.experimental.noalias.scope.decl(metadata) #13

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #14

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.umax.i64(i64, i64) #15

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.umin.i64(i64, i64) #15

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: read)
declare i32 @bcmp(ptr captures(none), ptr captures(none), i64) local_unnamed_addr #16

attributes #0 = { mustprogress uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { mustprogress norecurse uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { cold noreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { noreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #8 = { nobuiltin allocsize(0) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #9 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #10 = { nobuiltin nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #11 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: read) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #12 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
attributes #13 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #14 = { nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #15 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #16 = { nocallback nofree nounwind willreturn memory(argmem: read) }
attributes #17 = { builtin allocsize(0) }
attributes #18 = { builtin nounwind }
attributes #19 = { noreturn }
attributes #20 = { cold noreturn }
attributes #21 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !8, i64 16}
!7 = !{!"_ZTSNSt12_Vector_baseI6BigIntSaIS0_EE17_Vector_impl_dataE", !8, i64 0, !8, i64 8, !8, i64 16}
!8 = !{!"p1 _ZTS6BigInt", !9, i64 0}
!9 = !{!"any pointer", !10, i64 0}
!10 = !{!"omnipotent char", !11, i64 0}
!11 = !{!"Simple C++ TBAA"}
!12 = !{!7, !8, i64 0}
!13 = !{!7, !8, i64 8}
!14 = !{!15}
!15 = distinct !{!15, !16, !"_ZSt19__relocate_object_aI6BigIntS0_SaIS0_EEvPT_PT0_RT1_: argument 0"}
!16 = distinct !{!16, !"_ZSt19__relocate_object_aI6BigIntS0_SaIS0_EEvPT_PT0_RT1_"}
!17 = !{!18}
!18 = distinct !{!18, !16, !"_ZSt19__relocate_object_aI6BigIntS0_SaIS0_EEvPT_PT0_RT1_: argument 1"}
!19 = !{!20, !20, i64 0}
!20 = !{!"p1 long", !9, i64 0}
!21 = !{!22, !20, i64 16}
!22 = !{!"_ZTSNSt12_Vector_baseImSaImEE17_Vector_impl_dataE", !20, i64 0, !20, i64 8, !20, i64 16}
!23 = distinct !{!23, !24}
!24 = !{!"llvm.loop.mustprogress"}
!25 = !{!22, !20, i64 8}
!26 = !{!22, !20, i64 0}
!27 = !{!"branch_weights", !"expected", i32 1, i32 2000}
!28 = !{!"branch_weights", !"expected", i32 2000, i32 1}
!29 = !{!30, !30, i64 0}
!30 = !{!"long", !10, i64 0}
!31 = !{!32}
!32 = distinct !{!32, !33, !"_ZSt19__relocate_object_aI6BigIntS0_SaIS0_EEvPT_PT0_RT1_: argument 0"}
!33 = distinct !{!33, !"_ZSt19__relocate_object_aI6BigIntS0_SaIS0_EEvPT_PT0_RT1_"}
!34 = !{!35}
!35 = distinct !{!35, !33, !"_ZSt19__relocate_object_aI6BigIntS0_SaIS0_EEvPT_PT0_RT1_: argument 1"}
!36 = !{!8, !8, i64 0}
!37 = !{!38}
!38 = distinct !{!38, !39, !"_ZSt19__relocate_object_aI6BigIntS0_SaIS0_EEvPT_PT0_RT1_: argument 0"}
!39 = distinct !{!39, !"_ZSt19__relocate_object_aI6BigIntS0_SaIS0_EEvPT_PT0_RT1_"}
!40 = !{!41}
!41 = distinct !{!41, !39, !"_ZSt19__relocate_object_aI6BigIntS0_SaIS0_EEvPT_PT0_RT1_: argument 1"}
!42 = !{!43}
!43 = distinct !{!43, !44, !"_ZSt19__relocate_object_aI6BigIntS0_SaIS0_EEvPT_PT0_RT1_: argument 0"}
!44 = distinct !{!44, !"_ZSt19__relocate_object_aI6BigIntS0_SaIS0_EEvPT_PT0_RT1_"}
!45 = !{!46}
!46 = distinct !{!46, !44, !"_ZSt19__relocate_object_aI6BigIntS0_SaIS0_EEvPT_PT0_RT1_: argument 1"}
!47 = distinct !{!47, !24}
!48 = !{!"branch_weights", !"expected", i32 -2147483648, i32 0}
!49 = distinct !{!49, !24}
!50 = !{!51}
!51 = distinct !{!51, !52, !"_ZNKSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEE3strEv: argument 0"}
!52 = distinct !{!52, !"_ZNKSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEE3strEv"}
!53 = !{!54}
!54 = distinct !{!54, !55, !"_ZNKSt7__cxx1115basic_stringbufIcSt11char_traitsIcESaIcEE3strEv: argument 0"}
!55 = distinct !{!55, !"_ZNKSt7__cxx1115basic_stringbufIcSt11char_traitsIcESaIcEE3strEv"}
!56 = !{!57, !58, i64 0}
!57 = !{!"_ZTSNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_Alloc_hiderE", !58, i64 0}
!58 = !{!"p1 omnipotent char", !9, i64 0}
!59 = !{!54, !51}
!60 = !{!61, !30, i64 8}
!61 = !{!"_ZTSNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE", !57, i64 0, !30, i64 8, !10, i64 16}
!62 = !{!10, !10, i64 0}
!63 = !{!64, !58, i64 40}
!64 = !{!"_ZTSSt15basic_streambufIcSt11char_traitsIcEE", !58, i64 8, !58, i64 16, !58, i64 24, !58, i64 32, !58, i64 40, !58, i64 48, !65, i64 56}
!65 = !{!"_ZTSSt6locale", !66, i64 0}
!66 = !{!"p1 _ZTSNSt6locale5_ImplE", !9, i64 0}
!67 = !{!64, !58, i64 32}
!68 = !{!61, !58, i64 0}
!69 = !{!70, !70, i64 0}
!70 = !{!"vtable pointer", !11, i64 0}
!71 = !{!72, !30, i64 16}
!72 = !{!"_ZTSSt8ios_base", !30, i64 8, !30, i64 16, !73, i64 24, !74, i64 28, !74, i64 32, !75, i64 40, !76, i64 48, !10, i64 64, !77, i64 192, !78, i64 200, !65, i64 208}
!73 = !{!"_ZTSSt13_Ios_Fmtflags", !10, i64 0}
!74 = !{!"_ZTSSt12_Ios_Iostate", !10, i64 0}
!75 = !{!"p1 _ZTSNSt8ios_base14_Callback_listE", !9, i64 0}
!76 = !{!"_ZTSNSt8ios_base6_WordsE", !9, i64 0, !30, i64 8}
!77 = !{!"int", !10, i64 0}
!78 = !{!"p1 _ZTSNSt8ios_base6_WordsE", !9, i64 0}
!79 = !{!80, !82, i64 225}
!80 = !{!"_ZTSSt9basic_iosIcSt11char_traitsIcEE", !72, i64 0, !81, i64 216, !10, i64 224, !82, i64 225, !83, i64 232, !84, i64 240, !85, i64 248, !86, i64 256}
!81 = !{!"p1 _ZTSSo", !9, i64 0}
!82 = !{!"bool", !10, i64 0}
!83 = !{!"p1 _ZTSSt15basic_streambufIcSt11char_traitsIcEE", !9, i64 0}
!84 = !{!"p1 _ZTSSt5ctypeIcE", !9, i64 0}
!85 = !{!"p1 _ZTSSt7num_putIcSt19ostreambuf_iteratorIcSt11char_traitsIcEEE", !9, i64 0}
!86 = !{!"p1 _ZTSSt7num_getIcSt19istreambuf_iteratorIcSt11char_traitsIcEEE", !9, i64 0}
!87 = !{i8 0, i8 2}
!88 = !{}
!89 = !{!80, !84, i64 240}
!90 = !{!91, !10, i64 56}
!91 = !{!"_ZTSSt5ctypeIcE", !92, i64 0, !93, i64 16, !82, i64 24, !94, i64 32, !94, i64 40, !95, i64 48, !10, i64 56, !10, i64 57, !10, i64 313, !10, i64 569}
!92 = !{!"_ZTSNSt6locale5facetE", !77, i64 8}
!93 = !{!"p1 _ZTS15__locale_struct", !9, i64 0}
!94 = !{!"p1 int", !9, i64 0}
!95 = !{!"p1 short", !9, i64 0}
!96 = !{!80, !10, i64 224}
!97 = distinct !{!97, !24}
!98 = distinct !{!98, !24}
!99 = !{!100}
!100 = distinct !{!100, !101, !"_ZNKSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEE3strEv: argument 0"}
!101 = distinct !{!101, !"_ZNKSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEE3strEv"}
!102 = !{!103}
!103 = distinct !{!103, !104, !"_ZNKSt7__cxx1115basic_stringbufIcSt11char_traitsIcESaIcEE3strEv: argument 0"}
!104 = distinct !{!104, !"_ZNKSt7__cxx1115basic_stringbufIcSt11char_traitsIcESaIcEE3strEv"}
!105 = !{!103, !100}
!106 = !{!107}
!107 = distinct !{!107, !108, !"_ZNKSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEE3strEv: argument 0"}
!108 = distinct !{!108, !"_ZNKSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEE3strEv"}
!109 = !{!110}
!110 = distinct !{!110, !111, !"_ZNKSt7__cxx1115basic_stringbufIcSt11char_traitsIcESaIcEE3strEv: argument 0"}
!111 = distinct !{!111, !"_ZNKSt7__cxx1115basic_stringbufIcSt11char_traitsIcESaIcEE3strEv"}
!112 = !{!110, !107}
!113 = !{!58, !58, i64 0}
!114 = !{!72, !74, i64 32}
!115 = !{!72, !73, i64 24}
!116 = !{!73, !73, i64 0}
!117 = distinct !{!117, !24}
!118 = distinct !{!118, !24}
!119 = distinct !{!119, !24}
