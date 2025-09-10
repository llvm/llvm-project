; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Shootout-C++/hash.cpp'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Shootout-C++/hash.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

module asm ".globl _ZSt21ios_base_library_initv"

%"class.std::basic_ostream" = type { ptr, %"class.std::basic_ios" }
%"class.std::basic_ios" = type { %"class.std::ios_base", ptr, i8, i8, ptr, ptr, ptr, ptr }
%"class.std::ios_base" = type { ptr, i64, i64, i32, i32, i32, ptr, %"struct.std::ios_base::_Words", [8 x %"struct.std::ios_base::_Words"], i32, ptr, %"class.std::locale" }
%"struct.std::ios_base::_Words" = type { ptr, i64 }
%"class.std::locale" = type { ptr }
%"class.__gnu_cxx::hash_map" = type { %"class.__gnu_cxx::hashtable" }
%"class.__gnu_cxx::hashtable" = type { [8 x i8], %"class.std::vector", i64 }
%"class.std::vector" = type { %"struct.std::_Vector_base" }
%"struct.std::_Vector_base" = type { %"struct.std::_Vector_base<__gnu_cxx::_Hashtable_node<std::pair<const char *const, int>> *, std::allocator<__gnu_cxx::_Hashtable_node<std::pair<const char *const, int>> *>>::_Vector_impl" }
%"struct.std::_Vector_base<__gnu_cxx::_Hashtable_node<std::pair<const char *const, int>> *, std::allocator<__gnu_cxx::_Hashtable_node<std::pair<const char *const, int>> *>>::_Vector_impl" = type { %"struct.std::_Vector_base<__gnu_cxx::_Hashtable_node<std::pair<const char *const, int>> *, std::allocator<__gnu_cxx::_Hashtable_node<std::pair<const char *const, int>> *>>::_Vector_impl_data" }
%"struct.std::_Vector_base<__gnu_cxx::_Hashtable_node<std::pair<const char *const, int>> *, std::allocator<__gnu_cxx::_Hashtable_node<std::pair<const char *const, int>> *>>::_Vector_impl_data" = type { ptr, ptr, ptr }

$_ZN9__gnu_cxx8hash_mapIPKciNS_4hashIS2_EE5eqstrSaIiEED2Ev = comdat any

$_ZN9__gnu_cxx9hashtableISt4pairIKPKciES3_NS_4hashIS3_EESt10_Select1stIS5_E5eqstrSaIiEE21_M_initialize_bucketsEm = comdat any

$_ZNSt6vectorIPN9__gnu_cxx15_Hashtable_nodeISt4pairIKPKciEEESaIS8_EE14_M_fill_insertENS0_17__normal_iteratorIPS8_SA_EEmRKS8_ = comdat any

$_ZN9__gnu_cxx9hashtableISt4pairIKPKciES3_NS_4hashIS3_EESt10_Select1stIS5_E5eqstrSaIiEE6resizeEm = comdat any

$_ZN9__gnu_cxx21_Hashtable_prime_listImE16__stl_prime_listE = comdat any

@.str = private unnamed_addr constant [3 x i8] c"%x\00", align 1
@.str.1 = private unnamed_addr constant [3 x i8] c"%d\00", align 1
@_ZSt4cout = external global %"class.std::basic_ostream", align 8
@_ZN9__gnu_cxx21_Hashtable_prime_listImE16__stl_prime_listE = linkonce_odr dso_local constant [29 x i64] [i64 5, i64 53, i64 97, i64 193, i64 389, i64 769, i64 1543, i64 3079, i64 6151, i64 12289, i64 24593, i64 49157, i64 98317, i64 196613, i64 393241, i64 786433, i64 1572869, i64 3145739, i64 6291469, i64 12582917, i64 25165843, i64 50331653, i64 100663319, i64 201326611, i64 402653189, i64 805306457, i64 1610612741, i64 3221225473, i64 4294967291], comdat, align 8
@.str.2 = private unnamed_addr constant [16 x i8] c"vector::reserve\00", align 1
@.str.3 = private unnamed_addr constant [23 x i8] c"vector::_M_fill_insert\00", align 1
@.str.4 = private unnamed_addr constant [49 x i8] c"cannot create std::vector larger than max_size()\00", align 1

; Function Attrs: mustprogress norecurse uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 personality ptr @__gxx_personality_v0 {
  %3 = alloca [16 x i8], align 1
  %4 = alloca %"class.__gnu_cxx::hash_map", align 8
  %5 = icmp eq i32 %0, 2
  br i1 %5, label %6, label %11

6:                                                ; preds = %2
  %7 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %8 = load ptr, ptr %7, align 8, !tbaa !6
  %9 = tail call i64 @__isoc23_strtol(ptr noundef nonnull %8, ptr noundef null, i32 noundef 10) #15
  %10 = trunc i64 %9 to i32
  br label %11

11:                                               ; preds = %2, %6
  %12 = phi i32 [ %10, %6 ], [ 500000, %2 ]
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #15
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #15
  %13 = getelementptr inbounds nuw i8, ptr %4, i64 8
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %13, i8 0, i64 32, i1 false)
  invoke void @_ZN9__gnu_cxx9hashtableISt4pairIKPKciES3_NS_4hashIS3_EESt10_Select1stIS5_E5eqstrSaIiEE21_M_initialize_bucketsEm(ptr noundef nonnull align 8 dereferenceable(40) %4, i64 noundef 100)
          to label %14 unwind label %19

14:                                               ; preds = %11
  %15 = icmp slt i32 %12, 1
  br i1 %15, label %88, label %16

16:                                               ; preds = %14
  %17 = getelementptr inbounds nuw i8, ptr %4, i64 32
  %18 = getelementptr inbounds nuw i8, ptr %4, i64 16
  br label %34

19:                                               ; preds = %11
  %20 = landingpad { ptr, i32 }
          cleanup
  %21 = load ptr, ptr %13, align 8, !tbaa !11
  %22 = icmp eq ptr %21, null
  br i1 %22, label %29, label %23

23:                                               ; preds = %19
  %24 = getelementptr inbounds nuw i8, ptr %4, i64 24
  %25 = load ptr, ptr %24, align 8, !tbaa !15
  %26 = ptrtoint ptr %25 to i64
  %27 = ptrtoint ptr %21 to i64
  %28 = sub i64 %26, %27
  call void @_ZdlPvm(ptr noundef nonnull %21, i64 noundef %28) #16
  br label %29

29:                                               ; preds = %19, %23, %228
  %30 = phi { ptr, i32 } [ %229, %228 ], [ %20, %23 ], [ %20, %19 ]
  resume { ptr, i32 } %30

31:                                               ; preds = %81
  %32 = getelementptr inbounds nuw i8, ptr %4, i64 32
  %33 = getelementptr inbounds nuw i8, ptr %4, i64 16
  br label %91

34:                                               ; preds = %16, %81
  %35 = phi i32 [ 1, %16 ], [ %84, %81 ]
  %36 = call i32 (ptr, ptr, ...) @sprintf(ptr noundef nonnull dereferenceable(1) %3, ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %35) #15
  %37 = call noalias ptr @strdup(ptr noundef nonnull %3) #15
  %38 = load i64, ptr %17, align 8, !tbaa !16
  %39 = add i64 %38, 1
  invoke void @_ZN9__gnu_cxx9hashtableISt4pairIKPKciES3_NS_4hashIS3_EESt10_Select1stIS5_E5eqstrSaIiEE6resizeEm(ptr noundef nonnull align 8 dereferenceable(40) %4, i64 noundef %39)
          to label %40 unwind label %86

40:                                               ; preds = %34
  %41 = load ptr, ptr %18, align 8, !tbaa !26
  %42 = load ptr, ptr %13, align 8, !tbaa !11
  %43 = load i8, ptr %37, align 1, !tbaa !27
  %44 = icmp eq i8 %43, 0
  br i1 %44, label %55, label %45

45:                                               ; preds = %40, %45
  %46 = phi i8 [ %53, %45 ], [ %43, %40 ]
  %47 = phi i64 [ %51, %45 ], [ 0, %40 ]
  %48 = phi ptr [ %52, %45 ], [ %37, %40 ]
  %49 = mul i64 %47, 5
  %50 = zext i8 %46 to i64
  %51 = add i64 %49, %50
  %52 = getelementptr inbounds nuw i8, ptr %48, i64 1
  %53 = load i8, ptr %52, align 1, !tbaa !27
  %54 = icmp eq i8 %53, 0
  br i1 %54, label %55, label %45, !llvm.loop !28

55:                                               ; preds = %45, %40
  %56 = phi i64 [ 0, %40 ], [ %51, %45 ]
  %57 = ptrtoint ptr %41 to i64
  %58 = ptrtoint ptr %42 to i64
  %59 = sub i64 %57, %58
  %60 = ashr exact i64 %59, 3
  %61 = urem i64 %56, %60
  %62 = getelementptr inbounds nuw ptr, ptr %42, i64 %61
  %63 = load ptr, ptr %62, align 8, !tbaa !30
  %64 = icmp eq ptr %63, null
  br i1 %64, label %74, label %65

65:                                               ; preds = %55, %71
  %66 = phi ptr [ %72, %71 ], [ %63, %55 ]
  %67 = getelementptr inbounds nuw i8, ptr %66, i64 8
  %68 = load ptr, ptr %67, align 8, !tbaa !6
  %69 = call i32 @strcmp(ptr noundef nonnull dereferenceable(1) %68, ptr noundef nonnull dereferenceable(1) %37) #17
  %70 = icmp eq i32 %69, 0
  br i1 %70, label %81, label %71

71:                                               ; preds = %65
  %72 = load ptr, ptr %66, align 8, !tbaa !32
  %73 = icmp eq ptr %72, null
  br i1 %73, label %74, label %65, !llvm.loop !36

74:                                               ; preds = %71, %55
  %75 = invoke noalias noundef nonnull dereferenceable(24) ptr @_Znwm(i64 noundef 24) #18
          to label %76 unwind label %86

76:                                               ; preds = %74
  %77 = getelementptr inbounds nuw i8, ptr %75, i64 8
  store ptr %37, ptr %77, align 8
  %78 = getelementptr inbounds nuw i8, ptr %75, i64 16
  store i32 0, ptr %78, align 8
  store ptr %63, ptr %75, align 8, !tbaa !32
  store ptr %75, ptr %62, align 8, !tbaa !30
  %79 = load i64, ptr %17, align 8, !tbaa !16
  %80 = add i64 %79, 1
  store i64 %80, ptr %17, align 8, !tbaa !16
  br label %81

81:                                               ; preds = %65, %76
  %82 = phi ptr [ %75, %76 ], [ %66, %65 ]
  %83 = getelementptr inbounds nuw i8, ptr %82, i64 16
  store i32 %35, ptr %83, align 4, !tbaa !37
  %84 = add nuw i32 %35, 1
  %85 = icmp eq i32 %35, %12
  br i1 %85, label %31, label %34, !llvm.loop !38

86:                                               ; preds = %74, %34
  %87 = landingpad { ptr, i32 }
          cleanup
  br label %228

88:                                               ; preds = %144, %14
  %89 = phi i32 [ 0, %14 ], [ %146, %144 ]
  %90 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEi(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, i32 noundef %89)
          to label %151 unwind label %226

91:                                               ; preds = %31, %144
  %92 = phi i32 [ %12, %31 ], [ %147, %144 ]
  %93 = phi i32 [ 0, %31 ], [ %146, %144 ]
  %94 = call i32 (ptr, ptr, ...) @sprintf(ptr noundef nonnull dereferenceable(1) %3, ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %92) #15
  %95 = call noalias ptr @strdup(ptr noundef nonnull %3) #15
  %96 = load i64, ptr %32, align 8, !tbaa !16
  %97 = add i64 %96, 1
  invoke void @_ZN9__gnu_cxx9hashtableISt4pairIKPKciES3_NS_4hashIS3_EESt10_Select1stIS5_E5eqstrSaIiEE6resizeEm(ptr noundef nonnull align 8 dereferenceable(40) %4, i64 noundef %97)
          to label %98 unwind label %149

98:                                               ; preds = %91
  %99 = load ptr, ptr %33, align 8, !tbaa !26
  %100 = load ptr, ptr %13, align 8, !tbaa !11
  %101 = load i8, ptr %95, align 1, !tbaa !27
  %102 = icmp eq i8 %101, 0
  br i1 %102, label %113, label %103

103:                                              ; preds = %98, %103
  %104 = phi i8 [ %111, %103 ], [ %101, %98 ]
  %105 = phi i64 [ %109, %103 ], [ 0, %98 ]
  %106 = phi ptr [ %110, %103 ], [ %95, %98 ]
  %107 = mul i64 %105, 5
  %108 = zext i8 %104 to i64
  %109 = add i64 %107, %108
  %110 = getelementptr inbounds nuw i8, ptr %106, i64 1
  %111 = load i8, ptr %110, align 1, !tbaa !27
  %112 = icmp eq i8 %111, 0
  br i1 %112, label %113, label %103, !llvm.loop !28

113:                                              ; preds = %103, %98
  %114 = phi i64 [ 0, %98 ], [ %109, %103 ]
  %115 = ptrtoint ptr %99 to i64
  %116 = ptrtoint ptr %100 to i64
  %117 = sub i64 %115, %116
  %118 = ashr exact i64 %117, 3
  %119 = urem i64 %114, %118
  %120 = getelementptr inbounds nuw ptr, ptr %100, i64 %119
  %121 = load ptr, ptr %120, align 8, !tbaa !30
  %122 = icmp eq ptr %121, null
  br i1 %122, label %132, label %123

123:                                              ; preds = %113, %129
  %124 = phi ptr [ %130, %129 ], [ %121, %113 ]
  %125 = getelementptr inbounds nuw i8, ptr %124, i64 8
  %126 = load ptr, ptr %125, align 8, !tbaa !6
  %127 = call i32 @strcmp(ptr noundef nonnull dereferenceable(1) %126, ptr noundef nonnull dereferenceable(1) %95) #17
  %128 = icmp eq i32 %127, 0
  br i1 %128, label %139, label %129

129:                                              ; preds = %123
  %130 = load ptr, ptr %124, align 8, !tbaa !32
  %131 = icmp eq ptr %130, null
  br i1 %131, label %132, label %123, !llvm.loop !36

132:                                              ; preds = %129, %113
  %133 = invoke noalias noundef nonnull dereferenceable(24) ptr @_Znwm(i64 noundef 24) #18
          to label %134 unwind label %149

134:                                              ; preds = %132
  %135 = getelementptr inbounds nuw i8, ptr %133, i64 8
  store ptr %95, ptr %135, align 8
  %136 = getelementptr inbounds nuw i8, ptr %133, i64 16
  store i32 0, ptr %136, align 8
  store ptr %121, ptr %133, align 8, !tbaa !32
  store ptr %133, ptr %120, align 8, !tbaa !30
  %137 = load i64, ptr %32, align 8, !tbaa !16
  %138 = add i64 %137, 1
  store i64 %138, ptr %32, align 8, !tbaa !16
  br label %144

139:                                              ; preds = %123
  %140 = getelementptr inbounds nuw i8, ptr %124, i64 16
  %141 = load i32, ptr %140, align 4, !tbaa !37
  %142 = icmp ne i32 %141, 0
  %143 = zext i1 %142 to i32
  br label %144

144:                                              ; preds = %139, %134
  %145 = phi i32 [ 0, %134 ], [ %143, %139 ]
  %146 = add nuw nsw i32 %93, %145
  %147 = add nsw i32 %92, -1
  %148 = icmp sgt i32 %92, 1
  br i1 %148, label %91, label %88, !llvm.loop !39

149:                                              ; preds = %132, %91
  %150 = landingpad { ptr, i32 }
          cleanup
  br label %228

151:                                              ; preds = %88
  %152 = load ptr, ptr %90, align 8, !tbaa !40
  %153 = getelementptr i8, ptr %152, i64 -24
  %154 = load i64, ptr %153, align 8
  %155 = getelementptr inbounds i8, ptr %90, i64 %154
  %156 = getelementptr inbounds nuw i8, ptr %155, i64 240
  %157 = load ptr, ptr %156, align 8, !tbaa !42
  %158 = icmp eq ptr %157, null
  br i1 %158, label %159, label %161

159:                                              ; preds = %151
  invoke void @_ZSt16__throw_bad_castv() #19
          to label %160 unwind label %226

160:                                              ; preds = %159
  unreachable

161:                                              ; preds = %151
  %162 = getelementptr inbounds nuw i8, ptr %157, i64 56
  %163 = load i8, ptr %162, align 8, !tbaa !58
  %164 = icmp eq i8 %163, 0
  br i1 %164, label %168, label %165

165:                                              ; preds = %161
  %166 = getelementptr inbounds nuw i8, ptr %157, i64 67
  %167 = load i8, ptr %166, align 1, !tbaa !27
  br label %174

168:                                              ; preds = %161
  invoke void @_ZNKSt5ctypeIcE13_M_widen_initEv(ptr noundef nonnull align 8 dereferenceable(570) %157)
          to label %169 unwind label %226

169:                                              ; preds = %168
  %170 = load ptr, ptr %157, align 8, !tbaa !40
  %171 = getelementptr inbounds nuw i8, ptr %170, i64 48
  %172 = load ptr, ptr %171, align 8
  %173 = invoke noundef i8 %172(ptr noundef nonnull align 8 dereferenceable(570) %157, i8 noundef 10)
          to label %174 unwind label %226

174:                                              ; preds = %169, %165
  %175 = phi i8 [ %167, %165 ], [ %173, %169 ]
  %176 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8) %90, i8 noundef %175)
          to label %177 unwind label %226

177:                                              ; preds = %174
  %178 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo5flushEv(ptr noundef nonnull align 8 dereferenceable(8) %176)
          to label %179 unwind label %226

179:                                              ; preds = %177
  %180 = getelementptr inbounds nuw i8, ptr %4, i64 32
  %181 = load i64, ptr %180, align 8, !tbaa !16
  %182 = icmp eq i64 %181, 0
  br i1 %182, label %183, label %185

183:                                              ; preds = %179
  %184 = load ptr, ptr %13, align 8, !tbaa !11
  br label %216

185:                                              ; preds = %179
  %186 = getelementptr inbounds nuw i8, ptr %4, i64 16
  %187 = load ptr, ptr %186, align 8, !tbaa !26
  %188 = load ptr, ptr %13, align 8, !tbaa !11
  %189 = icmp eq ptr %187, %188
  br i1 %189, label %190, label %192

190:                                              ; preds = %206, %185
  %191 = phi ptr [ %188, %185 ], [ %208, %206 ]
  store i64 0, ptr %180, align 8, !tbaa !16
  br label %216

192:                                              ; preds = %185, %206
  %193 = phi ptr [ %207, %206 ], [ %187, %185 ]
  %194 = phi ptr [ %208, %206 ], [ %188, %185 ]
  %195 = phi i64 [ %210, %206 ], [ 0, %185 ]
  %196 = getelementptr inbounds nuw ptr, ptr %194, i64 %195
  %197 = load ptr, ptr %196, align 8, !tbaa !30
  %198 = icmp eq ptr %197, null
  br i1 %198, label %206, label %199

199:                                              ; preds = %192, %199
  %200 = phi ptr [ %201, %199 ], [ %197, %192 ]
  %201 = load ptr, ptr %200, align 8, !tbaa !32
  call void @_ZdlPvm(ptr noundef nonnull %200, i64 noundef 24) #16
  %202 = icmp eq ptr %201, null
  br i1 %202, label %203, label %199, !llvm.loop !64

203:                                              ; preds = %199
  %204 = load ptr, ptr %13, align 8, !tbaa !11
  %205 = load ptr, ptr %186, align 8, !tbaa !26
  br label %206

206:                                              ; preds = %203, %192
  %207 = phi ptr [ %205, %203 ], [ %193, %192 ]
  %208 = phi ptr [ %204, %203 ], [ %194, %192 ]
  %209 = getelementptr inbounds nuw ptr, ptr %208, i64 %195
  store ptr null, ptr %209, align 8, !tbaa !30
  %210 = add nuw i64 %195, 1
  %211 = ptrtoint ptr %207 to i64
  %212 = ptrtoint ptr %208 to i64
  %213 = sub i64 %211, %212
  %214 = ashr exact i64 %213, 3
  %215 = icmp ult i64 %210, %214
  br i1 %215, label %192, label %190, !llvm.loop !65

216:                                              ; preds = %190, %183
  %217 = phi ptr [ %184, %183 ], [ %191, %190 ]
  %218 = icmp eq ptr %217, null
  br i1 %218, label %225, label %219

219:                                              ; preds = %216
  %220 = getelementptr inbounds nuw i8, ptr %4, i64 24
  %221 = load ptr, ptr %220, align 8, !tbaa !15
  %222 = ptrtoint ptr %221 to i64
  %223 = ptrtoint ptr %217 to i64
  %224 = sub i64 %222, %223
  call void @_ZdlPvm(ptr noundef nonnull %217, i64 noundef %224) #16
  br label %225

225:                                              ; preds = %216, %219
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #15
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #15
  ret i32 0

226:                                              ; preds = %177, %174, %169, %168, %159, %88
  %227 = landingpad { ptr, i32 }
          cleanup
  br label %228

228:                                              ; preds = %149, %226, %86
  %229 = phi { ptr, i32 } [ %87, %86 ], [ %150, %149 ], [ %227, %226 ]
  call void @_ZN9__gnu_cxx8hash_mapIPKciNS_4hashIS2_EE5eqstrSaIiEED2Ev(ptr noundef nonnull align 8 dereferenceable(40) %4) #15
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #15
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #15
  br label %29
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: nofree nounwind
declare noundef i32 @sprintf(ptr noalias noundef writeonly captures(none), ptr noundef readonly captures(none), ...) local_unnamed_addr #2

; Function Attrs: mustprogress nofree nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
declare noalias ptr @strdup(ptr noundef readonly captures(none)) local_unnamed_addr #3

declare i32 @__gxx_personality_v0(...)

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEi(ptr noundef nonnull align 8 dereferenceable(8), i32 noundef) local_unnamed_addr #4

; Function Attrs: inlinehint mustprogress nounwind uwtable
define linkonce_odr dso_local void @_ZN9__gnu_cxx8hash_mapIPKciNS_4hashIS2_EE5eqstrSaIiEED2Ev(ptr noundef nonnull align 8 dereferenceable(40) %0) unnamed_addr #5 comdat personality ptr @__gxx_personality_v0 {
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 32
  %3 = load i64, ptr %2, align 8, !tbaa !16
  %4 = icmp eq i64 %3, 0
  %5 = getelementptr inbounds nuw i8, ptr %0, i64 8
  br i1 %4, label %6, label %8

6:                                                ; preds = %1
  %7 = load ptr, ptr %5, align 8, !tbaa !11
  br label %39

8:                                                ; preds = %1
  %9 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %10 = load ptr, ptr %9, align 8, !tbaa !26
  %11 = load ptr, ptr %5, align 8, !tbaa !11
  %12 = icmp eq ptr %10, %11
  br i1 %12, label %13, label %15

13:                                               ; preds = %29, %8
  %14 = phi ptr [ %11, %8 ], [ %31, %29 ]
  store i64 0, ptr %2, align 8, !tbaa !16
  br label %39

15:                                               ; preds = %8, %29
  %16 = phi ptr [ %30, %29 ], [ %10, %8 ]
  %17 = phi ptr [ %31, %29 ], [ %11, %8 ]
  %18 = phi i64 [ %33, %29 ], [ 0, %8 ]
  %19 = getelementptr inbounds nuw ptr, ptr %17, i64 %18
  %20 = load ptr, ptr %19, align 8, !tbaa !30
  %21 = icmp eq ptr %20, null
  br i1 %21, label %29, label %22

22:                                               ; preds = %15, %22
  %23 = phi ptr [ %24, %22 ], [ %20, %15 ]
  %24 = load ptr, ptr %23, align 8, !tbaa !32
  tail call void @_ZdlPvm(ptr noundef nonnull %23, i64 noundef 24) #16
  %25 = icmp eq ptr %24, null
  br i1 %25, label %26, label %22, !llvm.loop !64

26:                                               ; preds = %22
  %27 = load ptr, ptr %5, align 8, !tbaa !11
  %28 = load ptr, ptr %9, align 8, !tbaa !26
  br label %29

29:                                               ; preds = %26, %15
  %30 = phi ptr [ %28, %26 ], [ %16, %15 ]
  %31 = phi ptr [ %27, %26 ], [ %17, %15 ]
  %32 = getelementptr inbounds nuw ptr, ptr %31, i64 %18
  store ptr null, ptr %32, align 8, !tbaa !30
  %33 = add nuw i64 %18, 1
  %34 = ptrtoint ptr %30 to i64
  %35 = ptrtoint ptr %31 to i64
  %36 = sub i64 %34, %35
  %37 = ashr exact i64 %36, 3
  %38 = icmp ult i64 %33, %37
  br i1 %38, label %15, label %13, !llvm.loop !65

39:                                               ; preds = %13, %6
  %40 = phi ptr [ %7, %6 ], [ %14, %13 ]
  %41 = icmp eq ptr %40, null
  br i1 %41, label %48, label %42

42:                                               ; preds = %39
  %43 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %44 = load ptr, ptr %43, align 8, !tbaa !15
  %45 = ptrtoint ptr %44 to i64
  %46 = ptrtoint ptr %40 to i64
  %47 = sub i64 %45, %46
  tail call void @_ZdlPvm(ptr noundef nonnull %40, i64 noundef %47) #16
  br label %48

48:                                               ; preds = %39, %42
  ret void
}

; Function Attrs: nounwind
declare i64 @__isoc23_strtol(ptr noundef, ptr noundef, i32 noundef) local_unnamed_addr #6

; Function Attrs: nobuiltin nounwind
declare void @_ZdlPvm(ptr noundef, i64 noundef) local_unnamed_addr #7

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZN9__gnu_cxx9hashtableISt4pairIKPKciES3_NS_4hashIS3_EESt10_Select1stIS5_E5eqstrSaIiEE21_M_initialize_bucketsEm(ptr noundef nonnull align 8 dereferenceable(40) %0, i64 noundef %1) local_unnamed_addr #8 comdat personality ptr @__gxx_personality_v0 {
  %3 = alloca ptr, align 8
  br label %4

4:                                                ; preds = %4, %2
  %5 = phi ptr [ @_ZN9__gnu_cxx21_Hashtable_prime_listImE16__stl_prime_listE, %2 ], [ %15, %4 ]
  %6 = phi i64 [ 29, %2 ], [ %14, %4 ]
  %7 = lshr i64 %6, 1
  %8 = getelementptr inbounds nuw i64, ptr %5, i64 %7
  %9 = load i64, ptr %8, align 8, !tbaa !66
  %10 = icmp ult i64 %9, %1
  %11 = getelementptr inbounds nuw i8, ptr %8, i64 8
  %12 = xor i64 %7, -1
  %13 = add nsw i64 %6, %12
  %14 = select i1 %10, i64 %13, i64 %7
  %15 = select i1 %10, ptr %11, ptr %5
  %16 = icmp sgt i64 %14, 0
  br i1 %16, label %4, label %17, !llvm.loop !67

17:                                               ; preds = %4
  %18 = icmp eq ptr %15, getelementptr inbounds nuw (i8, ptr @_ZN9__gnu_cxx21_Hashtable_prime_listImE16__stl_prime_listE, i64 232)
  %19 = select i1 %18, ptr getelementptr inbounds nuw (i8, ptr @_ZN9__gnu_cxx21_Hashtable_prime_listImE16__stl_prime_listE, i64 224), ptr %15
  %20 = load i64, ptr %19, align 8, !tbaa !66
  %21 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %22 = icmp ugt i64 %20, 1152921504606846975
  br i1 %22, label %23, label %24

23:                                               ; preds = %17
  tail call void @_ZSt20__throw_length_errorPKc(ptr noundef nonnull @.str.2) #19
  unreachable

24:                                               ; preds = %17
  %25 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %26 = load ptr, ptr %25, align 8, !tbaa !15
  %27 = load ptr, ptr %21, align 8, !tbaa !11
  %28 = ptrtoint ptr %26 to i64
  %29 = ptrtoint ptr %27 to i64
  %30 = sub i64 %28, %29
  %31 = ashr exact i64 %30, 3
  %32 = icmp ult i64 %31, %20
  %33 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %34 = load ptr, ptr %33, align 8, !tbaa !68
  br i1 %32, label %35, label %49

35:                                               ; preds = %24
  %36 = ptrtoint ptr %34 to i64
  %37 = sub i64 %36, %29
  %38 = shl nuw nsw i64 %20, 3
  %39 = tail call noalias noundef nonnull ptr @_Znwm(i64 noundef %38) #18
  %40 = icmp sgt i64 %37, 0
  br i1 %40, label %41, label %42

41:                                               ; preds = %35
  tail call void @llvm.memcpy.p0.p0.i64(ptr nonnull align 8 %39, ptr align 8 %27, i64 %37, i1 false)
  br label %42

42:                                               ; preds = %41, %35
  %43 = icmp eq ptr %27, null
  br i1 %43, label %45, label %44

44:                                               ; preds = %42
  tail call void @_ZdlPvm(ptr noundef nonnull %27, i64 noundef %30) #16
  br label %45

45:                                               ; preds = %44, %42
  store ptr %39, ptr %21, align 8, !tbaa !11
  %46 = getelementptr inbounds nuw i8, ptr %39, i64 %37
  store ptr %46, ptr %33, align 8, !tbaa !26
  %47 = getelementptr inbounds nuw ptr, ptr %39, i64 %20
  store ptr %47, ptr %25, align 8, !tbaa !15
  %48 = ptrtoint ptr %39 to i64
  br label %49

49:                                               ; preds = %24, %45
  %50 = phi i64 [ %48, %45 ], [ %29, %24 ]
  %51 = phi ptr [ %39, %45 ], [ %27, %24 ]
  %52 = phi ptr [ %46, %45 ], [ %34, %24 ]
  %53 = ptrtoint ptr %52 to i64
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #15
  store ptr null, ptr %3, align 8, !tbaa !30
  %54 = sub i64 %53, %50
  %55 = getelementptr inbounds i8, ptr %51, i64 %54
  call void @_ZNSt6vectorIPN9__gnu_cxx15_Hashtable_nodeISt4pairIKPKciEEESaIS8_EE14_M_fill_insertENS0_17__normal_iteratorIPS8_SA_EEmRKS8_(ptr noundef nonnull align 8 dereferenceable(24) %21, ptr %55, i64 noundef %20, ptr noundef nonnull align 8 dereferenceable(8) %3)
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #15
  %56 = getelementptr inbounds nuw i8, ptr %0, i64 32
  store i64 0, ptr %56, align 8, !tbaa !16
  ret void
}

; Function Attrs: cold noreturn
declare void @_ZSt20__throw_length_errorPKc(ptr noundef) local_unnamed_addr #9

; Function Attrs: nobuiltin allocsize(0)
declare noundef nonnull ptr @_Znwm(i64 noundef) local_unnamed_addr #10

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #11

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZNSt6vectorIPN9__gnu_cxx15_Hashtable_nodeISt4pairIKPKciEEESaIS8_EE14_M_fill_insertENS0_17__normal_iteratorIPS8_SA_EEmRKS8_(ptr noundef nonnull align 8 dereferenceable(24) %0, ptr %1, i64 noundef %2, ptr noundef nonnull align 8 dereferenceable(8) %3) local_unnamed_addr #8 comdat personality ptr @__gxx_personality_v0 {
  %5 = icmp eq i64 %2, 0
  br i1 %5, label %223, label %6

6:                                                ; preds = %4
  %7 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %8 = load ptr, ptr %7, align 8, !tbaa !15
  %9 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %10 = load ptr, ptr %9, align 8, !tbaa !26
  %11 = ptrtoint ptr %8 to i64
  %12 = ptrtoint ptr %10 to i64
  %13 = sub i64 %11, %12
  %14 = ashr exact i64 %13, 3
  %15 = icmp ult i64 %14, %2
  br i1 %15, label %149, label %16

16:                                               ; preds = %6
  %17 = load ptr, ptr %3, align 8, !tbaa !30
  %18 = ptrtoint ptr %1 to i64
  %19 = sub i64 %12, %18
  %20 = ashr exact i64 %19, 3
  %21 = icmp ugt i64 %20, %2
  br i1 %21, label %22, label %76

22:                                               ; preds = %16
  %23 = sub i64 0, %2
  %24 = getelementptr inbounds ptr, ptr %10, i64 %23
  %25 = ptrtoint ptr %24 to i64
  %26 = icmp sgt i64 %2, 1
  br i1 %26, label %27, label %30, !prof !69

27:                                               ; preds = %22
  %28 = shl nsw i64 %2, 3
  tail call void @llvm.memmove.p0.p0.i64(ptr align 8 %10, ptr nonnull align 8 %24, i64 %28, i1 false)
  %29 = load ptr, ptr %9, align 8, !tbaa !26
  br label %34

30:                                               ; preds = %22
  %31 = icmp eq i64 %2, 1
  br i1 %31, label %32, label %34

32:                                               ; preds = %30
  %33 = load ptr, ptr %24, align 8, !tbaa !30
  store ptr %33, ptr %10, align 8, !tbaa !30
  br label %34

34:                                               ; preds = %32, %30, %27
  %35 = phi ptr [ %10, %32 ], [ %10, %30 ], [ %29, %27 ]
  %36 = getelementptr inbounds nuw ptr, ptr %35, i64 %2
  store ptr %36, ptr %9, align 8, !tbaa !26
  %37 = sub i64 %25, %18
  %38 = ashr exact i64 %37, 3
  %39 = icmp sgt i64 %38, 1
  br i1 %39, label %40, label %43, !prof !69

40:                                               ; preds = %34
  %41 = sub nsw i64 0, %38
  %42 = getelementptr inbounds ptr, ptr %10, i64 %41
  tail call void @llvm.memmove.p0.p0.i64(ptr nonnull align 8 %42, ptr align 8 %1, i64 %37, i1 false)
  br label %48

43:                                               ; preds = %34
  %44 = icmp eq i64 %37, 8
  br i1 %44, label %45, label %48

45:                                               ; preds = %43
  %46 = getelementptr inbounds i8, ptr %10, i64 -8
  %47 = load ptr, ptr %1, align 8, !tbaa !30
  store ptr %47, ptr %46, align 8, !tbaa !30
  br label %48

48:                                               ; preds = %45, %43, %40
  %49 = shl nuw nsw i64 %2, 3
  %50 = getelementptr inbounds nuw i8, ptr %1, i64 %49
  %51 = add nsw i64 %49, -8
  %52 = lshr exact i64 %51, 3
  %53 = add nuw nsw i64 %52, 1
  %54 = icmp ult i64 %51, 24
  br i1 %54, label %70, label %55

55:                                               ; preds = %48
  %56 = and i64 %53, 4611686018427387900
  %57 = shl i64 %56, 3
  %58 = getelementptr i8, ptr %1, i64 %57
  %59 = insertelement <2 x ptr> poison, ptr %17, i64 0
  %60 = shufflevector <2 x ptr> %59, <2 x ptr> poison, <2 x i32> zeroinitializer
  br label %61

61:                                               ; preds = %61, %55
  %62 = phi i64 [ 0, %55 ], [ %66, %61 ]
  %63 = shl i64 %62, 3
  %64 = getelementptr i8, ptr %1, i64 %63
  %65 = getelementptr i8, ptr %64, i64 16
  store <2 x ptr> %60, ptr %64, align 8, !tbaa !30
  store <2 x ptr> %60, ptr %65, align 8, !tbaa !30
  %66 = add nuw i64 %62, 4
  %67 = icmp eq i64 %66, %56
  br i1 %67, label %68, label %61, !llvm.loop !70

68:                                               ; preds = %61
  %69 = icmp eq i64 %53, %56
  br i1 %69, label %223, label %70

70:                                               ; preds = %48, %68
  %71 = phi ptr [ %1, %48 ], [ %58, %68 ]
  br label %72

72:                                               ; preds = %70, %72
  %73 = phi ptr [ %74, %72 ], [ %71, %70 ]
  store ptr %17, ptr %73, align 8, !tbaa !30
  %74 = getelementptr inbounds nuw i8, ptr %73, i64 8
  %75 = icmp eq ptr %74, %50
  br i1 %75, label %223, label %72, !llvm.loop !73

76:                                               ; preds = %16
  %77 = icmp eq i64 %2, %20
  br i1 %77, label %109, label %78

78:                                               ; preds = %76
  %79 = sub nuw i64 %2, %20
  %80 = shl nuw nsw i64 %79, 3
  %81 = getelementptr inbounds nuw i8, ptr %10, i64 %80
  %82 = shl i64 %2, 3
  %83 = add i64 %82, -8
  %84 = sub i64 %83, %19
  %85 = lshr i64 %84, 3
  %86 = add nuw nsw i64 %85, 1
  %87 = icmp ult i64 %84, 24
  br i1 %87, label %103, label %88

88:                                               ; preds = %78
  %89 = and i64 %86, 4611686018427387900
  %90 = shl i64 %89, 3
  %91 = getelementptr i8, ptr %10, i64 %90
  %92 = insertelement <2 x ptr> poison, ptr %17, i64 0
  %93 = shufflevector <2 x ptr> %92, <2 x ptr> poison, <2 x i32> zeroinitializer
  br label %94

94:                                               ; preds = %94, %88
  %95 = phi i64 [ 0, %88 ], [ %99, %94 ]
  %96 = shl i64 %95, 3
  %97 = getelementptr i8, ptr %10, i64 %96
  %98 = getelementptr i8, ptr %97, i64 16
  store <2 x ptr> %93, ptr %97, align 8, !tbaa !30
  store <2 x ptr> %93, ptr %98, align 8, !tbaa !30
  %99 = add nuw i64 %95, 4
  %100 = icmp eq i64 %99, %89
  br i1 %100, label %101, label %94, !llvm.loop !74

101:                                              ; preds = %94
  %102 = icmp eq i64 %86, %89
  br i1 %102, label %109, label %103

103:                                              ; preds = %78, %101
  %104 = phi ptr [ %10, %78 ], [ %91, %101 ]
  br label %105

105:                                              ; preds = %103, %105
  %106 = phi ptr [ %107, %105 ], [ %104, %103 ]
  store ptr %17, ptr %106, align 8, !tbaa !30
  %107 = getelementptr inbounds nuw i8, ptr %106, i64 8
  %108 = icmp eq ptr %107, %81
  br i1 %108, label %109, label %105, !llvm.loop !75

109:                                              ; preds = %105, %101, %76
  %110 = phi ptr [ %10, %76 ], [ %81, %101 ], [ %81, %105 ]
  store ptr %110, ptr %9, align 8, !tbaa !26
  %111 = icmp sgt i64 %19, 8
  br i1 %111, label %112, label %114, !prof !69

112:                                              ; preds = %109
  tail call void @llvm.memmove.p0.p0.i64(ptr align 8 %110, ptr align 8 %1, i64 %19, i1 false)
  %113 = load ptr, ptr %9, align 8, !tbaa !26
  br label %118

114:                                              ; preds = %109
  %115 = icmp eq i64 %19, 8
  br i1 %115, label %116, label %118

116:                                              ; preds = %114
  %117 = load ptr, ptr %1, align 8, !tbaa !30
  store ptr %117, ptr %110, align 8, !tbaa !30
  br label %118

118:                                              ; preds = %116, %114, %112
  %119 = phi ptr [ %110, %116 ], [ %110, %114 ], [ %113, %112 ]
  %120 = getelementptr inbounds nuw i8, ptr %119, i64 %19
  store ptr %120, ptr %9, align 8, !tbaa !26
  %121 = icmp eq ptr %1, %10
  br i1 %121, label %223, label %122

122:                                              ; preds = %118
  %123 = add i64 %12, -8
  %124 = sub i64 %123, %18
  %125 = lshr i64 %124, 3
  %126 = add nuw nsw i64 %125, 1
  %127 = icmp ult i64 %124, 24
  br i1 %127, label %143, label %128

128:                                              ; preds = %122
  %129 = and i64 %126, 4611686018427387900
  %130 = shl i64 %129, 3
  %131 = getelementptr i8, ptr %1, i64 %130
  %132 = insertelement <2 x ptr> poison, ptr %17, i64 0
  %133 = shufflevector <2 x ptr> %132, <2 x ptr> poison, <2 x i32> zeroinitializer
  br label %134

134:                                              ; preds = %134, %128
  %135 = phi i64 [ 0, %128 ], [ %139, %134 ]
  %136 = shl i64 %135, 3
  %137 = getelementptr i8, ptr %1, i64 %136
  %138 = getelementptr i8, ptr %137, i64 16
  store <2 x ptr> %133, ptr %137, align 8, !tbaa !30
  store <2 x ptr> %133, ptr %138, align 8, !tbaa !30
  %139 = add nuw i64 %135, 4
  %140 = icmp eq i64 %139, %129
  br i1 %140, label %141, label %134, !llvm.loop !76

141:                                              ; preds = %134
  %142 = icmp eq i64 %126, %129
  br i1 %142, label %223, label %143

143:                                              ; preds = %122, %141
  %144 = phi ptr [ %1, %122 ], [ %131, %141 ]
  br label %145

145:                                              ; preds = %143, %145
  %146 = phi ptr [ %147, %145 ], [ %144, %143 ]
  store ptr %17, ptr %146, align 8, !tbaa !30
  %147 = getelementptr inbounds nuw i8, ptr %146, i64 8
  %148 = icmp eq ptr %147, %10
  br i1 %148, label %223, label %145, !llvm.loop !77

149:                                              ; preds = %6
  %150 = load ptr, ptr %0, align 8, !tbaa !11
  %151 = ptrtoint ptr %150 to i64
  %152 = sub i64 %12, %151
  %153 = ashr exact i64 %152, 3
  %154 = sub nsw i64 1152921504606846975, %153
  %155 = icmp ult i64 %154, %2
  br i1 %155, label %156, label %157

156:                                              ; preds = %149
  tail call void @_ZSt20__throw_length_errorPKc(ptr noundef nonnull @.str.3) #19
  unreachable

157:                                              ; preds = %149
  %158 = tail call i64 @llvm.umax.i64(i64 %153, i64 %2)
  %159 = add nsw i64 %158, %153
  %160 = icmp ult i64 %159, %153
  %161 = tail call i64 @llvm.umin.i64(i64 %159, i64 1152921504606846975)
  %162 = select i1 %160, i64 1152921504606846975, i64 %161
  %163 = ptrtoint ptr %1 to i64
  %164 = sub i64 %163, %151
  %165 = icmp eq i64 %162, 0
  br i1 %165, label %169, label %166

166:                                              ; preds = %157
  %167 = shl nuw nsw i64 %162, 3
  %168 = tail call noalias noundef nonnull ptr @_Znwm(i64 noundef %167) #18
  br label %169

169:                                              ; preds = %166, %157
  %170 = phi ptr [ %168, %166 ], [ null, %157 ]
  %171 = getelementptr inbounds i8, ptr %170, i64 %164
  %172 = shl nuw nsw i64 %2, 3
  %173 = getelementptr inbounds nuw i8, ptr %171, i64 %172
  %174 = load ptr, ptr %3, align 8, !tbaa !30
  %175 = add nsw i64 %172, -8
  %176 = lshr exact i64 %175, 3
  %177 = add nuw nsw i64 %176, 1
  %178 = icmp ult i64 %175, 24
  br i1 %178, label %194, label %179

179:                                              ; preds = %169
  %180 = and i64 %177, 4611686018427387900
  %181 = shl i64 %180, 3
  %182 = getelementptr i8, ptr %171, i64 %181
  %183 = insertelement <2 x ptr> poison, ptr %174, i64 0
  %184 = shufflevector <2 x ptr> %183, <2 x ptr> poison, <2 x i32> zeroinitializer
  br label %185

185:                                              ; preds = %185, %179
  %186 = phi i64 [ 0, %179 ], [ %190, %185 ]
  %187 = shl i64 %186, 3
  %188 = getelementptr i8, ptr %171, i64 %187
  %189 = getelementptr i8, ptr %188, i64 16
  store <2 x ptr> %184, ptr %188, align 8, !tbaa !30
  store <2 x ptr> %184, ptr %189, align 8, !tbaa !30
  %190 = add nuw i64 %186, 4
  %191 = icmp eq i64 %190, %180
  br i1 %191, label %192, label %185, !llvm.loop !78

192:                                              ; preds = %185
  %193 = icmp eq i64 %177, %180
  br i1 %193, label %200, label %194

194:                                              ; preds = %169, %192
  %195 = phi ptr [ %171, %169 ], [ %182, %192 ]
  br label %196

196:                                              ; preds = %194, %196
  %197 = phi ptr [ %198, %196 ], [ %195, %194 ]
  store ptr %174, ptr %197, align 8, !tbaa !30
  %198 = getelementptr inbounds nuw i8, ptr %197, i64 8
  %199 = icmp eq ptr %198, %173
  br i1 %199, label %200, label %196, !llvm.loop !79

200:                                              ; preds = %196, %192
  %201 = icmp sgt i64 %164, 8
  br i1 %201, label %202, label %203, !prof !69

202:                                              ; preds = %200
  tail call void @llvm.memmove.p0.p0.i64(ptr align 8 %170, ptr align 8 %150, i64 %164, i1 false)
  br label %207

203:                                              ; preds = %200
  %204 = icmp eq i64 %164, 8
  br i1 %204, label %205, label %207

205:                                              ; preds = %203
  %206 = load ptr, ptr %150, align 8, !tbaa !30
  store ptr %206, ptr %170, align 8, !tbaa !30
  br label %207

207:                                              ; preds = %205, %203, %202
  %208 = getelementptr inbounds nuw ptr, ptr %171, i64 %2
  %209 = sub i64 %12, %163
  %210 = icmp sgt i64 %209, 8
  br i1 %210, label %211, label %212, !prof !69

211:                                              ; preds = %207
  tail call void @llvm.memmove.p0.p0.i64(ptr nonnull align 8 %208, ptr align 8 %1, i64 %209, i1 false)
  br label %216

212:                                              ; preds = %207
  %213 = icmp eq i64 %209, 8
  br i1 %213, label %214, label %216

214:                                              ; preds = %212
  %215 = load ptr, ptr %1, align 8, !tbaa !30
  store ptr %215, ptr %208, align 8, !tbaa !30
  br label %216

216:                                              ; preds = %214, %212, %211
  %217 = getelementptr inbounds i8, ptr %208, i64 %209
  %218 = icmp eq ptr %150, null
  br i1 %218, label %221, label %219

219:                                              ; preds = %216
  %220 = sub i64 %11, %151
  tail call void @_ZdlPvm(ptr noundef nonnull %150, i64 noundef %220) #16
  br label %221

221:                                              ; preds = %216, %219
  store ptr %170, ptr %0, align 8, !tbaa !11
  store ptr %217, ptr %9, align 8, !tbaa !26
  %222 = getelementptr inbounds nuw ptr, ptr %170, i64 %162
  store ptr %222, ptr %7, align 8, !tbaa !15
  br label %223

223:                                              ; preds = %145, %72, %141, %68, %118, %221, %4
  ret void
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memmove.p0.p0.i64(ptr writeonly captures(none), ptr readonly captures(none), i64, i1 immarg) #11

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZN9__gnu_cxx9hashtableISt4pairIKPKciES3_NS_4hashIS3_EESt10_Select1stIS5_E5eqstrSaIiEE6resizeEm(ptr noundef nonnull align 8 dereferenceable(40) %0, i64 noundef %1) local_unnamed_addr #8 comdat personality ptr @__gxx_personality_v0 {
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %4 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %5 = load ptr, ptr %4, align 8, !tbaa !26
  %6 = load ptr, ptr %3, align 8, !tbaa !11
  %7 = ptrtoint ptr %5 to i64
  %8 = ptrtoint ptr %6 to i64
  %9 = sub i64 %7, %8
  %10 = ashr exact i64 %9, 3
  %11 = icmp ugt i64 %1, %10
  br i1 %11, label %12, label %82

12:                                               ; preds = %2, %12
  %13 = phi ptr [ %23, %12 ], [ @_ZN9__gnu_cxx21_Hashtable_prime_listImE16__stl_prime_listE, %2 ]
  %14 = phi i64 [ %22, %12 ], [ 29, %2 ]
  %15 = lshr i64 %14, 1
  %16 = getelementptr inbounds nuw i64, ptr %13, i64 %15
  %17 = load i64, ptr %16, align 8, !tbaa !66
  %18 = icmp ult i64 %17, %1
  %19 = getelementptr inbounds nuw i8, ptr %16, i64 8
  %20 = xor i64 %15, -1
  %21 = add nsw i64 %14, %20
  %22 = select i1 %18, i64 %21, i64 %15
  %23 = select i1 %18, ptr %19, ptr %13
  %24 = icmp sgt i64 %22, 0
  br i1 %24, label %12, label %25, !llvm.loop !67

25:                                               ; preds = %12
  %26 = icmp eq ptr %23, getelementptr inbounds nuw (i8, ptr @_ZN9__gnu_cxx21_Hashtable_prime_listImE16__stl_prime_listE, i64 232)
  %27 = select i1 %26, ptr getelementptr inbounds nuw (i8, ptr @_ZN9__gnu_cxx21_Hashtable_prime_listImE16__stl_prime_listE, i64 224), ptr %23
  %28 = load i64, ptr %27, align 8, !tbaa !66
  %29 = icmp ugt i64 %28, %10
  br i1 %29, label %30, label %82

30:                                               ; preds = %25
  %31 = icmp ugt i64 %28, 1152921504606846975
  br i1 %31, label %32, label %33

32:                                               ; preds = %30
  tail call void @_ZSt20__throw_length_errorPKc(ptr noundef nonnull @.str.4) #19
  unreachable

33:                                               ; preds = %30
  %34 = shl nuw nsw i64 %28, 3
  %35 = tail call noalias noundef nonnull ptr @_Znwm(i64 noundef %34) #18
  tail call void @llvm.memset.p0.i64(ptr nonnull align 8 %35, i8 0, i64 %34, i1 false), !tbaa !30
  %36 = getelementptr inbounds nuw ptr, ptr %35, i64 %28
  %37 = getelementptr inbounds nuw i8, ptr %35, i64 %34
  %38 = icmp eq ptr %5, %6
  br i1 %38, label %39, label %47

39:                                               ; preds = %33
  %40 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %41 = load ptr, ptr %40, align 8, !tbaa !15
  store ptr %35, ptr %3, align 8, !tbaa !11
  store ptr %37, ptr %4, align 8, !tbaa !26
  store ptr %36, ptr %40, align 8, !tbaa !15
  %42 = icmp eq ptr %6, null
  br i1 %42, label %82, label %43

43:                                               ; preds = %79, %39
  %44 = phi ptr [ %81, %79 ], [ %41, %39 ]
  %45 = ptrtoint ptr %44 to i64
  %46 = sub i64 %45, %8
  tail call void @_ZdlPvm(ptr noundef nonnull %6, i64 noundef %46) #16
  br label %82

47:                                               ; preds = %33, %76
  %48 = phi i64 [ %77, %76 ], [ 0, %33 ]
  %49 = getelementptr inbounds nuw ptr, ptr %6, i64 %48
  %50 = load ptr, ptr %49, align 8, !tbaa !30
  %51 = icmp eq ptr %50, null
  br i1 %51, label %76, label %52

52:                                               ; preds = %47, %68
  %53 = phi ptr [ %74, %68 ], [ %50, %47 ]
  %54 = getelementptr inbounds nuw i8, ptr %53, i64 8
  %55 = load ptr, ptr %54, align 8, !tbaa !6
  %56 = load i8, ptr %55, align 1, !tbaa !27
  %57 = icmp eq i8 %56, 0
  br i1 %57, label %68, label %58

58:                                               ; preds = %52, %58
  %59 = phi i8 [ %66, %58 ], [ %56, %52 ]
  %60 = phi i64 [ %64, %58 ], [ 0, %52 ]
  %61 = phi ptr [ %65, %58 ], [ %55, %52 ]
  %62 = mul i64 %60, 5
  %63 = zext i8 %59 to i64
  %64 = add i64 %62, %63
  %65 = getelementptr inbounds nuw i8, ptr %61, i64 1
  %66 = load i8, ptr %65, align 1, !tbaa !27
  %67 = icmp eq i8 %66, 0
  br i1 %67, label %68, label %58, !llvm.loop !28

68:                                               ; preds = %58, %52
  %69 = phi i64 [ 0, %52 ], [ %64, %58 ]
  %70 = urem i64 %69, %28
  %71 = load ptr, ptr %53, align 8, !tbaa !32
  store ptr %71, ptr %49, align 8, !tbaa !30
  %72 = getelementptr inbounds nuw ptr, ptr %35, i64 %70
  %73 = load ptr, ptr %72, align 8, !tbaa !30
  store ptr %73, ptr %53, align 8, !tbaa !32
  store ptr %53, ptr %72, align 8, !tbaa !30
  %74 = load ptr, ptr %49, align 8, !tbaa !30
  %75 = icmp eq ptr %74, null
  br i1 %75, label %76, label %52, !llvm.loop !80

76:                                               ; preds = %68, %47
  %77 = add nuw i64 %48, 1
  %78 = icmp eq i64 %77, %10
  br i1 %78, label %79, label %47, !llvm.loop !81

79:                                               ; preds = %76
  %80 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %81 = load ptr, ptr %80, align 8, !tbaa !15
  store ptr %35, ptr %3, align 8, !tbaa !11
  store ptr %37, ptr %4, align 8, !tbaa !26
  store ptr %36, ptr %80, align 8, !tbaa !15
  br label %43

82:                                               ; preds = %43, %39, %25, %2
  ret void
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: read)
declare i32 @strcmp(ptr noundef captures(none), ptr noundef captures(none)) local_unnamed_addr #12

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8), i8 noundef) local_unnamed_addr #4

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo5flushEv(ptr noundef nonnull align 8 dereferenceable(8)) local_unnamed_addr #4

; Function Attrs: cold noreturn
declare void @_ZSt16__throw_bad_castv() local_unnamed_addr #9

declare void @_ZNKSt5ctypeIcE13_M_widen_initEv(ptr noundef nonnull align 8 dereferenceable(570)) local_unnamed_addr #4

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #13

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.umax.i64(i64, i64) #14

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.umin.i64(i64, i64) #14

attributes #0 = { mustprogress norecurse uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress nofree nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite) "alloc-family"="malloc" "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { inlinehint mustprogress nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { nobuiltin nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #8 = { mustprogress uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #9 = { cold noreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #10 = { nobuiltin allocsize(0) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #11 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #12 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: read) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #13 = { nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #14 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #15 = { nounwind }
attributes #16 = { builtin nounwind }
attributes #17 = { nounwind willreturn memory(read) }
attributes #18 = { builtin allocsize(0) }
attributes #19 = { cold noreturn }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"p1 omnipotent char", !8, i64 0}
!8 = !{!"any pointer", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C++ TBAA"}
!11 = !{!12, !13, i64 0}
!12 = !{!"_ZTSNSt12_Vector_baseIPN9__gnu_cxx15_Hashtable_nodeISt4pairIKPKciEEESaIS8_EE17_Vector_impl_dataE", !13, i64 0, !13, i64 8, !13, i64 16}
!13 = !{!"p2 _ZTSN9__gnu_cxx15_Hashtable_nodeISt4pairIKPKciEEE", !14, i64 0}
!14 = !{!"any p2 pointer", !8, i64 0}
!15 = !{!12, !13, i64 16}
!16 = !{!17, !25, i64 32}
!17 = !{!"_ZTSN9__gnu_cxx9hashtableISt4pairIKPKciES3_NS_4hashIS3_EESt10_Select1stIS5_E5eqstrSaIiEEE", !18, i64 0, !19, i64 1, !20, i64 2, !21, i64 3, !22, i64 8, !25, i64 32}
!18 = !{!"_ZTSSaIN9__gnu_cxx15_Hashtable_nodeISt4pairIKPKciEEEE"}
!19 = !{!"_ZTSN9__gnu_cxx4hashIPKcEE"}
!20 = !{!"_ZTS5eqstr"}
!21 = !{!"_ZTSSt10_Select1stISt4pairIKPKciEE"}
!22 = !{!"_ZTSSt6vectorIPN9__gnu_cxx15_Hashtable_nodeISt4pairIKPKciEEESaIS8_EE", !23, i64 0}
!23 = !{!"_ZTSSt12_Vector_baseIPN9__gnu_cxx15_Hashtable_nodeISt4pairIKPKciEEESaIS8_EE", !24, i64 0}
!24 = !{!"_ZTSNSt12_Vector_baseIPN9__gnu_cxx15_Hashtable_nodeISt4pairIKPKciEEESaIS8_EE12_Vector_implE", !12, i64 0}
!25 = !{!"long", !9, i64 0}
!26 = !{!12, !13, i64 8}
!27 = !{!9, !9, i64 0}
!28 = distinct !{!28, !29}
!29 = !{!"llvm.loop.mustprogress"}
!30 = !{!31, !31, i64 0}
!31 = !{!"p1 _ZTSN9__gnu_cxx15_Hashtable_nodeISt4pairIKPKciEEE", !8, i64 0}
!32 = !{!33, !31, i64 0}
!33 = !{!"_ZTSN9__gnu_cxx15_Hashtable_nodeISt4pairIKPKciEEE", !31, i64 0, !34, i64 8}
!34 = !{!"_ZTSSt4pairIKPKciE", !7, i64 0, !35, i64 8}
!35 = !{!"int", !9, i64 0}
!36 = distinct !{!36, !29}
!37 = !{!35, !35, i64 0}
!38 = distinct !{!38, !29}
!39 = distinct !{!39, !29}
!40 = !{!41, !41, i64 0}
!41 = !{!"vtable pointer", !10, i64 0}
!42 = !{!43, !55, i64 240}
!43 = !{!"_ZTSSt9basic_iosIcSt11char_traitsIcEE", !44, i64 0, !52, i64 216, !9, i64 224, !53, i64 225, !54, i64 232, !55, i64 240, !56, i64 248, !57, i64 256}
!44 = !{!"_ZTSSt8ios_base", !25, i64 8, !25, i64 16, !45, i64 24, !46, i64 28, !46, i64 32, !47, i64 40, !48, i64 48, !9, i64 64, !35, i64 192, !49, i64 200, !50, i64 208}
!45 = !{!"_ZTSSt13_Ios_Fmtflags", !9, i64 0}
!46 = !{!"_ZTSSt12_Ios_Iostate", !9, i64 0}
!47 = !{!"p1 _ZTSNSt8ios_base14_Callback_listE", !8, i64 0}
!48 = !{!"_ZTSNSt8ios_base6_WordsE", !8, i64 0, !25, i64 8}
!49 = !{!"p1 _ZTSNSt8ios_base6_WordsE", !8, i64 0}
!50 = !{!"_ZTSSt6locale", !51, i64 0}
!51 = !{!"p1 _ZTSNSt6locale5_ImplE", !8, i64 0}
!52 = !{!"p1 _ZTSSo", !8, i64 0}
!53 = !{!"bool", !9, i64 0}
!54 = !{!"p1 _ZTSSt15basic_streambufIcSt11char_traitsIcEE", !8, i64 0}
!55 = !{!"p1 _ZTSSt5ctypeIcE", !8, i64 0}
!56 = !{!"p1 _ZTSSt7num_putIcSt19ostreambuf_iteratorIcSt11char_traitsIcEEE", !8, i64 0}
!57 = !{!"p1 _ZTSSt7num_getIcSt19istreambuf_iteratorIcSt11char_traitsIcEEE", !8, i64 0}
!58 = !{!59, !9, i64 56}
!59 = !{!"_ZTSSt5ctypeIcE", !60, i64 0, !61, i64 16, !53, i64 24, !62, i64 32, !62, i64 40, !63, i64 48, !9, i64 56, !9, i64 57, !9, i64 313, !9, i64 569}
!60 = !{!"_ZTSNSt6locale5facetE", !35, i64 8}
!61 = !{!"p1 _ZTS15__locale_struct", !8, i64 0}
!62 = !{!"p1 int", !8, i64 0}
!63 = !{!"p1 short", !8, i64 0}
!64 = distinct !{!64, !29}
!65 = distinct !{!65, !29}
!66 = !{!25, !25, i64 0}
!67 = distinct !{!67, !29}
!68 = !{!13, !13, i64 0}
!69 = !{!"branch_weights", !"expected", i32 2000, i32 1}
!70 = distinct !{!70, !29, !71, !72}
!71 = !{!"llvm.loop.isvectorized", i32 1}
!72 = !{!"llvm.loop.unroll.runtime.disable"}
!73 = distinct !{!73, !29, !72, !71}
!74 = distinct !{!74, !29, !71, !72}
!75 = distinct !{!75, !29, !72, !71}
!76 = distinct !{!76, !29, !71, !72}
!77 = distinct !{!77, !29, !72, !71}
!78 = distinct !{!78, !29, !71, !72}
!79 = distinct !{!79, !29, !72, !71}
!80 = distinct !{!80, !29}
!81 = distinct !{!81, !29}
