; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Shootout-C++/hash2.cpp'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Shootout-C++/hash2.cpp"
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

@.str = private unnamed_addr constant [7 x i8] c"foo_%d\00", align 1
@_ZSt4cout = external global %"class.std::basic_ostream", align 8
@.str.1 = private unnamed_addr constant [6 x i8] c"foo_1\00", align 1
@.str.2 = private unnamed_addr constant [2 x i8] c" \00", align 1
@.str.3 = private unnamed_addr constant [9 x i8] c"foo_9999\00", align 1
@_ZN9__gnu_cxx21_Hashtable_prime_listImE16__stl_prime_listE = linkonce_odr dso_local constant [29 x i64] [i64 5, i64 53, i64 97, i64 193, i64 389, i64 769, i64 1543, i64 3079, i64 6151, i64 12289, i64 24593, i64 49157, i64 98317, i64 196613, i64 393241, i64 786433, i64 1572869, i64 3145739, i64 6291469, i64 12582917, i64 25165843, i64 50331653, i64 100663319, i64 201326611, i64 402653189, i64 805306457, i64 1610612741, i64 3221225473, i64 4294967291], comdat, align 8
@.str.4 = private unnamed_addr constant [16 x i8] c"vector::reserve\00", align 1
@.str.5 = private unnamed_addr constant [23 x i8] c"vector::_M_fill_insert\00", align 1
@.str.6 = private unnamed_addr constant [49 x i8] c"cannot create std::vector larger than max_size()\00", align 1

; Function Attrs: mustprogress norecurse uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 personality ptr @__gxx_personality_v0 {
  %3 = alloca [16 x i8], align 1
  %4 = alloca %"class.__gnu_cxx::hash_map", align 8
  %5 = alloca %"class.__gnu_cxx::hash_map", align 8
  %6 = icmp eq i32 %0, 2
  br i1 %6, label %7, label %12

7:                                                ; preds = %2
  %8 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %9 = load ptr, ptr %8, align 8, !tbaa !6
  %10 = tail call i64 @__isoc23_strtol(ptr noundef nonnull %9, ptr noundef null, i32 noundef 10) #15
  %11 = trunc i64 %10 to i32
  br label %12

12:                                               ; preds = %2, %7
  %13 = phi i32 [ %11, %7 ], [ 2000, %2 ]
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #15
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #15
  %14 = getelementptr inbounds nuw i8, ptr %4, i64 8
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %14, i8 0, i64 32, i1 false)
  invoke void @_ZN9__gnu_cxx9hashtableISt4pairIKPKciES3_NS_4hashIS3_EESt10_Select1stIS5_E5eqstrSaIiEE21_M_initialize_bucketsEm(ptr noundef nonnull align 8 dereferenceable(40) %4, i64 noundef 100)
          to label %27 unwind label %15

15:                                               ; preds = %12
  %16 = landingpad { ptr, i32 }
          cleanup
  %17 = load ptr, ptr %14, align 8, !tbaa !11
  %18 = icmp eq ptr %17, null
  br i1 %18, label %25, label %19

19:                                               ; preds = %15
  %20 = getelementptr inbounds nuw i8, ptr %4, i64 24
  %21 = load ptr, ptr %20, align 8, !tbaa !15
  %22 = ptrtoint ptr %21 to i64
  %23 = ptrtoint ptr %17 to i64
  %24 = sub i64 %22, %23
  call void @_ZdlPvm(ptr noundef nonnull %17, i64 noundef %24) #16
  br label %25

25:                                               ; preds = %15, %19, %702
  %26 = phi { ptr, i32 } [ %703, %702 ], [ %16, %19 ], [ %16, %15 ]
  resume { ptr, i32 } %26

27:                                               ; preds = %12
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #15
  %28 = getelementptr inbounds nuw i8, ptr %5, i64 8
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %28, i8 0, i64 32, i1 false)
  invoke void @_ZN9__gnu_cxx9hashtableISt4pairIKPKciES3_NS_4hashIS3_EESt10_Select1stIS5_E5eqstrSaIiEE21_M_initialize_bucketsEm(ptr noundef nonnull align 8 dereferenceable(40) %5, i64 noundef 100)
          to label %29 unwind label %32

29:                                               ; preds = %27
  %30 = getelementptr inbounds nuw i8, ptr %4, i64 32
  %31 = getelementptr inbounds nuw i8, ptr %4, i64 16
  br label %49

32:                                               ; preds = %27
  %33 = landingpad { ptr, i32 }
          cleanup
  %34 = load ptr, ptr %28, align 8, !tbaa !11
  %35 = icmp eq ptr %34, null
  br i1 %35, label %702, label %36

36:                                               ; preds = %32
  %37 = getelementptr inbounds nuw i8, ptr %5, i64 24
  %38 = load ptr, ptr %37, align 8, !tbaa !15
  %39 = ptrtoint ptr %38 to i64
  %40 = ptrtoint ptr %34 to i64
  %41 = sub i64 %39, %40
  call void @_ZdlPvm(ptr noundef nonnull %34, i64 noundef %41) #16
  br label %702

42:                                               ; preds = %96
  %43 = icmp sgt i32 %13, 0
  br i1 %43, label %44, label %103

44:                                               ; preds = %42
  %45 = getelementptr inbounds nuw i8, ptr %4, i64 24
  %46 = getelementptr inbounds nuw i8, ptr %5, i64 32
  %47 = getelementptr inbounds nuw i8, ptr %5, i64 16
  %48 = getelementptr inbounds nuw i8, ptr %5, i64 24
  br label %133

49:                                               ; preds = %29, %96
  %50 = phi i32 [ 0, %29 ], [ %99, %96 ]
  %51 = call i32 (ptr, ptr, ...) @sprintf(ptr noundef nonnull dereferenceable(1) %3, ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %50) #15
  %52 = call noalias ptr @strdup(ptr noundef nonnull %3) #15
  %53 = load i64, ptr %30, align 8, !tbaa !16
  %54 = add i64 %53, 1
  invoke void @_ZN9__gnu_cxx9hashtableISt4pairIKPKciES3_NS_4hashIS3_EESt10_Select1stIS5_E5eqstrSaIiEE6resizeEm(ptr noundef nonnull align 8 dereferenceable(40) %4, i64 noundef %54)
          to label %55 unwind label %101

55:                                               ; preds = %49
  %56 = load ptr, ptr %31, align 8, !tbaa !26
  %57 = load ptr, ptr %14, align 8, !tbaa !11
  %58 = load i8, ptr %52, align 1, !tbaa !27
  %59 = icmp eq i8 %58, 0
  br i1 %59, label %70, label %60

60:                                               ; preds = %55, %60
  %61 = phi i8 [ %68, %60 ], [ %58, %55 ]
  %62 = phi i64 [ %66, %60 ], [ 0, %55 ]
  %63 = phi ptr [ %67, %60 ], [ %52, %55 ]
  %64 = mul i64 %62, 5
  %65 = zext i8 %61 to i64
  %66 = add i64 %64, %65
  %67 = getelementptr inbounds nuw i8, ptr %63, i64 1
  %68 = load i8, ptr %67, align 1, !tbaa !27
  %69 = icmp eq i8 %68, 0
  br i1 %69, label %70, label %60, !llvm.loop !28

70:                                               ; preds = %60, %55
  %71 = phi i64 [ 0, %55 ], [ %66, %60 ]
  %72 = ptrtoint ptr %56 to i64
  %73 = ptrtoint ptr %57 to i64
  %74 = sub i64 %72, %73
  %75 = ashr exact i64 %74, 3
  %76 = urem i64 %71, %75
  %77 = getelementptr inbounds nuw ptr, ptr %57, i64 %76
  %78 = load ptr, ptr %77, align 8, !tbaa !30
  %79 = icmp eq ptr %78, null
  br i1 %79, label %89, label %80

80:                                               ; preds = %70, %86
  %81 = phi ptr [ %87, %86 ], [ %78, %70 ]
  %82 = getelementptr inbounds nuw i8, ptr %81, i64 8
  %83 = load ptr, ptr %82, align 8, !tbaa !6
  %84 = call i32 @strcmp(ptr noundef nonnull dereferenceable(1) %83, ptr noundef nonnull dereferenceable(1) %52) #17
  %85 = icmp eq i32 %84, 0
  br i1 %85, label %96, label %86

86:                                               ; preds = %80
  %87 = load ptr, ptr %81, align 8, !tbaa !32
  %88 = icmp eq ptr %87, null
  br i1 %88, label %89, label %80, !llvm.loop !36

89:                                               ; preds = %86, %70
  %90 = invoke noalias noundef nonnull dereferenceable(24) ptr @_Znwm(i64 noundef 24) #18
          to label %91 unwind label %101

91:                                               ; preds = %89
  %92 = getelementptr inbounds nuw i8, ptr %90, i64 8
  store ptr %52, ptr %92, align 8
  %93 = getelementptr inbounds nuw i8, ptr %90, i64 16
  store i32 0, ptr %93, align 8
  store ptr %78, ptr %90, align 8, !tbaa !32
  store ptr %90, ptr %77, align 8, !tbaa !30
  %94 = load i64, ptr %30, align 8, !tbaa !16
  %95 = add i64 %94, 1
  store i64 %95, ptr %30, align 8, !tbaa !16
  br label %96

96:                                               ; preds = %80, %91
  %97 = phi ptr [ %90, %91 ], [ %81, %80 ]
  %98 = getelementptr inbounds nuw i8, ptr %97, i64 16
  store i32 %50, ptr %98, align 4, !tbaa !37
  %99 = add nuw nsw i32 %50, 1
  %100 = icmp eq i32 %99, 10000
  br i1 %100, label %42, label %49, !llvm.loop !38

101:                                              ; preds = %89, %49
  %102 = landingpad { ptr, i32 }
          cleanup
  br label %700

103:                                              ; preds = %151, %42
  %104 = load i64, ptr %30, align 8, !tbaa !16
  %105 = add i64 %104, 1
  invoke void @_ZN9__gnu_cxx9hashtableISt4pairIKPKciES3_NS_4hashIS3_EESt10_Select1stIS5_E5eqstrSaIiEE6resizeEm(ptr noundef nonnull align 8 dereferenceable(40) %4, i64 noundef %105)
          to label %106 unwind label %692

106:                                              ; preds = %103
  %107 = load ptr, ptr %31, align 8, !tbaa !26
  %108 = load ptr, ptr %14, align 8, !tbaa !11
  %109 = ptrtoint ptr %107 to i64
  %110 = ptrtoint ptr %108 to i64
  %111 = sub i64 %109, %110
  %112 = ashr exact i64 %111, 3
  %113 = urem i64 80924, %112
  %114 = getelementptr inbounds nuw ptr, ptr %108, i64 %113
  %115 = load ptr, ptr %114, align 8, !tbaa !30
  %116 = icmp eq ptr %115, null
  br i1 %116, label %126, label %117

117:                                              ; preds = %106, %123
  %118 = phi ptr [ %124, %123 ], [ %115, %106 ]
  %119 = getelementptr inbounds nuw i8, ptr %118, i64 8
  %120 = load ptr, ptr %119, align 8, !tbaa !6
  %121 = call i32 @strcmp(ptr noundef nonnull dereferenceable(1) %120, ptr noundef nonnull dereferenceable(6) @.str.1) #17
  %122 = icmp eq i32 %121, 0
  br i1 %122, label %453, label %123

123:                                              ; preds = %117
  %124 = load ptr, ptr %118, align 8, !tbaa !32
  %125 = icmp eq ptr %124, null
  br i1 %125, label %126, label %117, !llvm.loop !36

126:                                              ; preds = %123, %106
  %127 = invoke noalias noundef nonnull dereferenceable(24) ptr @_Znwm(i64 noundef 24) #18
          to label %128 unwind label %692

128:                                              ; preds = %126
  %129 = getelementptr inbounds nuw i8, ptr %127, i64 8
  store ptr @.str.1, ptr %129, align 8
  %130 = getelementptr inbounds nuw i8, ptr %127, i64 16
  store i32 0, ptr %130, align 8
  store ptr %115, ptr %127, align 8, !tbaa !32
  store ptr %127, ptr %114, align 8, !tbaa !30
  %131 = load i64, ptr %30, align 8, !tbaa !16
  %132 = add i64 %131, 1
  store i64 %132, ptr %30, align 8, !tbaa !16
  br label %456

133:                                              ; preds = %44, %151
  %134 = phi i32 [ 0, %44 ], [ %152, %151 ]
  %135 = load ptr, ptr %31, align 8, !tbaa !26
  %136 = load ptr, ptr %14, align 8, !tbaa !11
  %137 = icmp eq ptr %135, %136
  br i1 %137, label %151, label %138

138:                                              ; preds = %133
  %139 = ptrtoint ptr %135 to i64
  %140 = ptrtoint ptr %136 to i64
  %141 = sub i64 %139, %140
  %142 = ashr exact i64 %141, 3
  br label %146

143:                                              ; preds = %146
  %144 = add nuw i64 %147, 1
  %145 = icmp eq i64 %144, %142
  br i1 %145, label %151, label %146, !llvm.loop !39

146:                                              ; preds = %143, %138
  %147 = phi i64 [ %144, %143 ], [ 0, %138 ]
  %148 = getelementptr inbounds nuw ptr, ptr %136, i64 %147
  %149 = load ptr, ptr %148, align 8, !tbaa !30
  %150 = icmp eq ptr %149, null
  br i1 %150, label %143, label %158

151:                                              ; preds = %143, %445, %133
  %152 = add nuw nsw i32 %134, 1
  %153 = icmp eq i32 %152, %13
  br i1 %153, label %103, label %133, !llvm.loop !40

154:                                              ; preds = %277, %403, %193, %319
  %155 = landingpad { ptr, i32 }
          cleanup
  br label %700

156:                                              ; preds = %191
  %157 = landingpad { ptr, i32 }
          cleanup
  br label %700

158:                                              ; preds = %146, %420
  %159 = phi ptr [ %421, %420 ], [ %149, %146 ]
  %160 = getelementptr inbounds nuw i8, ptr %159, i64 8
  %161 = load ptr, ptr %160, align 8, !tbaa !6
  %162 = load i64, ptr %30, align 8, !tbaa !16
  %163 = add i64 %162, 1
  %164 = load ptr, ptr %31, align 8, !tbaa !26
  %165 = load ptr, ptr %14, align 8, !tbaa !11
  %166 = ptrtoint ptr %164 to i64
  %167 = ptrtoint ptr %165 to i64
  %168 = sub i64 %166, %167
  %169 = ashr exact i64 %168, 3
  %170 = icmp ugt i64 %163, %169
  br i1 %170, label %171, label %243

171:                                              ; preds = %158, %171
  %172 = phi ptr [ %182, %171 ], [ @_ZN9__gnu_cxx21_Hashtable_prime_listImE16__stl_prime_listE, %158 ]
  %173 = phi i64 [ %181, %171 ], [ 29, %158 ]
  %174 = lshr i64 %173, 1
  %175 = getelementptr inbounds nuw i64, ptr %172, i64 %174
  %176 = load i64, ptr %175, align 8, !tbaa !41
  %177 = icmp ult i64 %176, %163
  %178 = getelementptr inbounds nuw i8, ptr %175, i64 8
  %179 = xor i64 %174, -1
  %180 = add nsw i64 %173, %179
  %181 = select i1 %177, i64 %180, i64 %174
  %182 = select i1 %177, ptr %178, ptr %172
  %183 = icmp sgt i64 %181, 0
  br i1 %183, label %171, label %184, !llvm.loop !42

184:                                              ; preds = %171
  %185 = icmp eq ptr %182, getelementptr inbounds nuw (i8, ptr @_ZN9__gnu_cxx21_Hashtable_prime_listImE16__stl_prime_listE, i64 232)
  %186 = select i1 %185, ptr getelementptr inbounds nuw (i8, ptr @_ZN9__gnu_cxx21_Hashtable_prime_listImE16__stl_prime_listE, i64 224), ptr %182
  %187 = load i64, ptr %186, align 8, !tbaa !41
  %188 = icmp ugt i64 %187, %169
  br i1 %188, label %189, label %243

189:                                              ; preds = %184
  %190 = icmp ugt i64 %187, 1152921504606846975
  br i1 %190, label %191, label %193

191:                                              ; preds = %317, %189
  invoke void @_ZSt20__throw_length_errorPKc(ptr noundef nonnull @.str.6) #19
          to label %192 unwind label %156

192:                                              ; preds = %191
  unreachable

193:                                              ; preds = %189
  %194 = shl nuw nsw i64 %187, 3
  %195 = invoke noalias noundef nonnull ptr @_Znwm(i64 noundef %194) #18
          to label %196 unwind label %154

196:                                              ; preds = %193
  call void @llvm.memset.p0.i64(ptr nonnull align 8 %195, i8 0, i64 %194, i1 false), !tbaa !30
  %197 = getelementptr inbounds nuw ptr, ptr %195, i64 %187
  %198 = getelementptr inbounds nuw i8, ptr %195, i64 %194
  %199 = icmp eq ptr %164, %165
  br i1 %199, label %200, label %209

200:                                              ; preds = %196
  %201 = load ptr, ptr %45, align 8, !tbaa !15
  store ptr %195, ptr %14, align 8, !tbaa !11
  store ptr %198, ptr %31, align 8, !tbaa !26
  store ptr %197, ptr %45, align 8, !tbaa !15
  %202 = icmp eq ptr %164, null
  br i1 %202, label %243, label %203

203:                                              ; preds = %241, %200
  %204 = phi ptr [ %242, %241 ], [ %201, %200 ]
  %205 = ptrtoint ptr %204 to i64
  %206 = sub i64 %205, %167
  call void @_ZdlPvm(ptr noundef nonnull %165, i64 noundef %206) #16
  %207 = load ptr, ptr %31, align 8, !tbaa !26
  %208 = load ptr, ptr %14, align 8, !tbaa !11
  br label %243

209:                                              ; preds = %196, %238
  %210 = phi i64 [ %239, %238 ], [ 0, %196 ]
  %211 = getelementptr inbounds nuw ptr, ptr %165, i64 %210
  %212 = load ptr, ptr %211, align 8, !tbaa !30
  %213 = icmp eq ptr %212, null
  br i1 %213, label %238, label %214

214:                                              ; preds = %209, %230
  %215 = phi ptr [ %236, %230 ], [ %212, %209 ]
  %216 = getelementptr inbounds nuw i8, ptr %215, i64 8
  %217 = load ptr, ptr %216, align 8, !tbaa !6
  %218 = load i8, ptr %217, align 1, !tbaa !27
  %219 = icmp eq i8 %218, 0
  br i1 %219, label %230, label %220

220:                                              ; preds = %214, %220
  %221 = phi i8 [ %228, %220 ], [ %218, %214 ]
  %222 = phi i64 [ %226, %220 ], [ 0, %214 ]
  %223 = phi ptr [ %227, %220 ], [ %217, %214 ]
  %224 = mul i64 %222, 5
  %225 = zext i8 %221 to i64
  %226 = add i64 %224, %225
  %227 = getelementptr inbounds nuw i8, ptr %223, i64 1
  %228 = load i8, ptr %227, align 1, !tbaa !27
  %229 = icmp eq i8 %228, 0
  br i1 %229, label %230, label %220, !llvm.loop !28

230:                                              ; preds = %220, %214
  %231 = phi i64 [ 0, %214 ], [ %226, %220 ]
  %232 = urem i64 %231, %187
  %233 = load ptr, ptr %215, align 8, !tbaa !32
  store ptr %233, ptr %211, align 8, !tbaa !30
  %234 = getelementptr inbounds nuw ptr, ptr %195, i64 %232
  %235 = load ptr, ptr %234, align 8, !tbaa !30
  store ptr %235, ptr %215, align 8, !tbaa !32
  store ptr %215, ptr %234, align 8, !tbaa !30
  %236 = load ptr, ptr %211, align 8, !tbaa !30
  %237 = icmp eq ptr %236, null
  br i1 %237, label %238, label %214, !llvm.loop !43

238:                                              ; preds = %230, %209
  %239 = add nuw i64 %210, 1
  %240 = icmp eq i64 %239, %169
  br i1 %240, label %241, label %209, !llvm.loop !44

241:                                              ; preds = %238
  %242 = load ptr, ptr %45, align 8, !tbaa !15
  store ptr %195, ptr %14, align 8, !tbaa !11
  store ptr %198, ptr %31, align 8, !tbaa !26
  store ptr %197, ptr %45, align 8, !tbaa !15
  br label %203

243:                                              ; preds = %203, %200, %184, %158
  %244 = phi ptr [ %208, %203 ], [ %195, %200 ], [ %165, %184 ], [ %165, %158 ]
  %245 = phi ptr [ %207, %203 ], [ %198, %200 ], [ %164, %184 ], [ %164, %158 ]
  %246 = load i8, ptr %161, align 1, !tbaa !27
  %247 = icmp eq i8 %246, 0
  br i1 %247, label %258, label %248

248:                                              ; preds = %243, %248
  %249 = phi i8 [ %256, %248 ], [ %246, %243 ]
  %250 = phi i64 [ %254, %248 ], [ 0, %243 ]
  %251 = phi ptr [ %255, %248 ], [ %161, %243 ]
  %252 = mul i64 %250, 5
  %253 = zext i8 %249 to i64
  %254 = add i64 %252, %253
  %255 = getelementptr inbounds nuw i8, ptr %251, i64 1
  %256 = load i8, ptr %255, align 1, !tbaa !27
  %257 = icmp eq i8 %256, 0
  br i1 %257, label %258, label %248, !llvm.loop !28

258:                                              ; preds = %248, %243
  %259 = phi i64 [ 0, %243 ], [ %254, %248 ]
  %260 = ptrtoint ptr %245 to i64
  %261 = ptrtoint ptr %244 to i64
  %262 = sub i64 %260, %261
  %263 = ashr exact i64 %262, 3
  %264 = urem i64 %259, %263
  %265 = getelementptr inbounds nuw ptr, ptr %244, i64 %264
  %266 = load ptr, ptr %265, align 8, !tbaa !30
  %267 = icmp eq ptr %266, null
  br i1 %267, label %277, label %268

268:                                              ; preds = %258, %274
  %269 = phi ptr [ %275, %274 ], [ %266, %258 ]
  %270 = getelementptr inbounds nuw i8, ptr %269, i64 8
  %271 = load ptr, ptr %270, align 8, !tbaa !6
  %272 = call i32 @strcmp(ptr noundef nonnull dereferenceable(1) %271, ptr noundef nonnull dereferenceable(1) %161) #17
  %273 = icmp eq i32 %272, 0
  br i1 %273, label %284, label %274

274:                                              ; preds = %268
  %275 = load ptr, ptr %269, align 8, !tbaa !32
  %276 = icmp eq ptr %275, null
  br i1 %276, label %277, label %268, !llvm.loop !36

277:                                              ; preds = %274, %258
  %278 = invoke noalias noundef nonnull dereferenceable(24) ptr @_Znwm(i64 noundef 24) #18
          to label %279 unwind label %154

279:                                              ; preds = %277
  %280 = getelementptr inbounds nuw i8, ptr %278, i64 8
  store ptr %161, ptr %280, align 8
  %281 = getelementptr inbounds nuw i8, ptr %278, i64 16
  store i32 0, ptr %281, align 8
  store ptr %266, ptr %278, align 8, !tbaa !32
  store ptr %278, ptr %265, align 8, !tbaa !30
  %282 = load i64, ptr %30, align 8, !tbaa !16
  %283 = add i64 %282, 1
  store i64 %283, ptr %30, align 8, !tbaa !16
  br label %287

284:                                              ; preds = %268
  %285 = getelementptr inbounds nuw i8, ptr %269, i64 16
  %286 = load i32, ptr %285, align 4, !tbaa !37
  br label %287

287:                                              ; preds = %284, %279
  %288 = phi i32 [ 0, %279 ], [ %286, %284 ]
  %289 = load ptr, ptr %160, align 8, !tbaa !6
  %290 = load i64, ptr %46, align 8, !tbaa !16
  %291 = add i64 %290, 1
  %292 = load ptr, ptr %47, align 8, !tbaa !26
  %293 = load ptr, ptr %28, align 8, !tbaa !11
  %294 = ptrtoint ptr %292 to i64
  %295 = ptrtoint ptr %293 to i64
  %296 = sub i64 %294, %295
  %297 = ashr exact i64 %296, 3
  %298 = icmp ugt i64 %291, %297
  br i1 %298, label %299, label %369

299:                                              ; preds = %287, %299
  %300 = phi ptr [ %310, %299 ], [ @_ZN9__gnu_cxx21_Hashtable_prime_listImE16__stl_prime_listE, %287 ]
  %301 = phi i64 [ %309, %299 ], [ 29, %287 ]
  %302 = lshr i64 %301, 1
  %303 = getelementptr inbounds nuw i64, ptr %300, i64 %302
  %304 = load i64, ptr %303, align 8, !tbaa !41
  %305 = icmp ult i64 %304, %291
  %306 = getelementptr inbounds nuw i8, ptr %303, i64 8
  %307 = xor i64 %302, -1
  %308 = add nsw i64 %301, %307
  %309 = select i1 %305, i64 %308, i64 %302
  %310 = select i1 %305, ptr %306, ptr %300
  %311 = icmp sgt i64 %309, 0
  br i1 %311, label %299, label %312, !llvm.loop !42

312:                                              ; preds = %299
  %313 = icmp eq ptr %310, getelementptr inbounds nuw (i8, ptr @_ZN9__gnu_cxx21_Hashtable_prime_listImE16__stl_prime_listE, i64 232)
  %314 = select i1 %313, ptr getelementptr inbounds nuw (i8, ptr @_ZN9__gnu_cxx21_Hashtable_prime_listImE16__stl_prime_listE, i64 224), ptr %310
  %315 = load i64, ptr %314, align 8, !tbaa !41
  %316 = icmp ugt i64 %315, %297
  br i1 %316, label %317, label %369

317:                                              ; preds = %312
  %318 = icmp ugt i64 %315, 1152921504606846975
  br i1 %318, label %191, label %319

319:                                              ; preds = %317
  %320 = shl nuw nsw i64 %315, 3
  %321 = invoke noalias noundef nonnull ptr @_Znwm(i64 noundef %320) #18
          to label %322 unwind label %154

322:                                              ; preds = %319
  call void @llvm.memset.p0.i64(ptr nonnull align 8 %321, i8 0, i64 %320, i1 false), !tbaa !30
  %323 = getelementptr inbounds nuw ptr, ptr %321, i64 %315
  %324 = getelementptr inbounds nuw i8, ptr %321, i64 %320
  %325 = icmp eq ptr %292, %293
  br i1 %325, label %326, label %335

326:                                              ; preds = %322
  %327 = load ptr, ptr %48, align 8, !tbaa !15
  store ptr %321, ptr %28, align 8, !tbaa !11
  store ptr %324, ptr %47, align 8, !tbaa !26
  store ptr %323, ptr %48, align 8, !tbaa !15
  %328 = icmp eq ptr %292, null
  br i1 %328, label %369, label %329

329:                                              ; preds = %367, %326
  %330 = phi ptr [ %368, %367 ], [ %327, %326 ]
  %331 = ptrtoint ptr %330 to i64
  %332 = sub i64 %331, %295
  call void @_ZdlPvm(ptr noundef nonnull %293, i64 noundef %332) #16
  %333 = load ptr, ptr %47, align 8, !tbaa !26
  %334 = load ptr, ptr %28, align 8, !tbaa !11
  br label %369

335:                                              ; preds = %322, %364
  %336 = phi i64 [ %365, %364 ], [ 0, %322 ]
  %337 = getelementptr inbounds nuw ptr, ptr %293, i64 %336
  %338 = load ptr, ptr %337, align 8, !tbaa !30
  %339 = icmp eq ptr %338, null
  br i1 %339, label %364, label %340

340:                                              ; preds = %335, %356
  %341 = phi ptr [ %362, %356 ], [ %338, %335 ]
  %342 = getelementptr inbounds nuw i8, ptr %341, i64 8
  %343 = load ptr, ptr %342, align 8, !tbaa !6
  %344 = load i8, ptr %343, align 1, !tbaa !27
  %345 = icmp eq i8 %344, 0
  br i1 %345, label %356, label %346

346:                                              ; preds = %340, %346
  %347 = phi i8 [ %354, %346 ], [ %344, %340 ]
  %348 = phi i64 [ %352, %346 ], [ 0, %340 ]
  %349 = phi ptr [ %353, %346 ], [ %343, %340 ]
  %350 = mul i64 %348, 5
  %351 = zext i8 %347 to i64
  %352 = add i64 %350, %351
  %353 = getelementptr inbounds nuw i8, ptr %349, i64 1
  %354 = load i8, ptr %353, align 1, !tbaa !27
  %355 = icmp eq i8 %354, 0
  br i1 %355, label %356, label %346, !llvm.loop !28

356:                                              ; preds = %346, %340
  %357 = phi i64 [ 0, %340 ], [ %352, %346 ]
  %358 = urem i64 %357, %315
  %359 = load ptr, ptr %341, align 8, !tbaa !32
  store ptr %359, ptr %337, align 8, !tbaa !30
  %360 = getelementptr inbounds nuw ptr, ptr %321, i64 %358
  %361 = load ptr, ptr %360, align 8, !tbaa !30
  store ptr %361, ptr %341, align 8, !tbaa !32
  store ptr %341, ptr %360, align 8, !tbaa !30
  %362 = load ptr, ptr %337, align 8, !tbaa !30
  %363 = icmp eq ptr %362, null
  br i1 %363, label %364, label %340, !llvm.loop !43

364:                                              ; preds = %356, %335
  %365 = add nuw i64 %336, 1
  %366 = icmp eq i64 %365, %297
  br i1 %366, label %367, label %335, !llvm.loop !44

367:                                              ; preds = %364
  %368 = load ptr, ptr %48, align 8, !tbaa !15
  store ptr %321, ptr %28, align 8, !tbaa !11
  store ptr %324, ptr %47, align 8, !tbaa !26
  store ptr %323, ptr %48, align 8, !tbaa !15
  br label %329

369:                                              ; preds = %329, %326, %312, %287
  %370 = phi ptr [ %334, %329 ], [ %321, %326 ], [ %293, %312 ], [ %293, %287 ]
  %371 = phi ptr [ %333, %329 ], [ %324, %326 ], [ %292, %312 ], [ %292, %287 ]
  %372 = load i8, ptr %289, align 1, !tbaa !27
  %373 = icmp eq i8 %372, 0
  br i1 %373, label %384, label %374

374:                                              ; preds = %369, %374
  %375 = phi i8 [ %382, %374 ], [ %372, %369 ]
  %376 = phi i64 [ %380, %374 ], [ 0, %369 ]
  %377 = phi ptr [ %381, %374 ], [ %289, %369 ]
  %378 = mul i64 %376, 5
  %379 = zext i8 %375 to i64
  %380 = add i64 %378, %379
  %381 = getelementptr inbounds nuw i8, ptr %377, i64 1
  %382 = load i8, ptr %381, align 1, !tbaa !27
  %383 = icmp eq i8 %382, 0
  br i1 %383, label %384, label %374, !llvm.loop !28

384:                                              ; preds = %374, %369
  %385 = phi i64 [ 0, %369 ], [ %380, %374 ]
  %386 = ptrtoint ptr %371 to i64
  %387 = ptrtoint ptr %370 to i64
  %388 = sub i64 %386, %387
  %389 = ashr exact i64 %388, 3
  %390 = urem i64 %385, %389
  %391 = getelementptr inbounds nuw ptr, ptr %370, i64 %390
  %392 = load ptr, ptr %391, align 8, !tbaa !30
  %393 = icmp eq ptr %392, null
  br i1 %393, label %403, label %394

394:                                              ; preds = %384, %400
  %395 = phi ptr [ %401, %400 ], [ %392, %384 ]
  %396 = getelementptr inbounds nuw i8, ptr %395, i64 8
  %397 = load ptr, ptr %396, align 8, !tbaa !6
  %398 = call i32 @strcmp(ptr noundef nonnull dereferenceable(1) %397, ptr noundef nonnull dereferenceable(1) %289) #17
  %399 = icmp eq i32 %398, 0
  br i1 %399, label %410, label %400

400:                                              ; preds = %394
  %401 = load ptr, ptr %395, align 8, !tbaa !32
  %402 = icmp eq ptr %401, null
  br i1 %402, label %403, label %394, !llvm.loop !36

403:                                              ; preds = %400, %384
  %404 = invoke noalias noundef nonnull dereferenceable(24) ptr @_Znwm(i64 noundef 24) #18
          to label %405 unwind label %154

405:                                              ; preds = %403
  %406 = getelementptr inbounds nuw i8, ptr %404, i64 8
  store ptr %289, ptr %406, align 8
  %407 = getelementptr inbounds nuw i8, ptr %404, i64 16
  store i32 0, ptr %407, align 8
  store ptr %392, ptr %404, align 8, !tbaa !32
  store ptr %404, ptr %391, align 8, !tbaa !30
  %408 = load i64, ptr %46, align 8, !tbaa !16
  %409 = add i64 %408, 1
  store i64 %409, ptr %46, align 8, !tbaa !16
  br label %413

410:                                              ; preds = %394
  %411 = getelementptr inbounds nuw i8, ptr %395, i64 16
  %412 = load i32, ptr %411, align 4, !tbaa !37
  br label %413

413:                                              ; preds = %410, %405
  %414 = phi i32 [ 0, %405 ], [ %412, %410 ]
  %415 = phi ptr [ %404, %405 ], [ %395, %410 ]
  %416 = getelementptr inbounds nuw i8, ptr %415, i64 16
  %417 = add nsw i32 %414, %288
  store i32 %417, ptr %416, align 4, !tbaa !37
  %418 = load ptr, ptr %159, align 8, !tbaa !32
  %419 = icmp eq ptr %418, null
  br i1 %419, label %422, label %420

420:                                              ; preds = %449, %413
  %421 = phi ptr [ %418, %413 ], [ %451, %449 ]
  br label %158

422:                                              ; preds = %413
  %423 = load ptr, ptr %31, align 8, !tbaa !26
  %424 = load ptr, ptr %14, align 8, !tbaa !11
  %425 = load ptr, ptr %160, align 8, !tbaa !6
  %426 = load i8, ptr %425, align 1, !tbaa !27
  %427 = icmp eq i8 %426, 0
  br i1 %427, label %438, label %428

428:                                              ; preds = %422, %428
  %429 = phi i8 [ %436, %428 ], [ %426, %422 ]
  %430 = phi i64 [ %434, %428 ], [ 0, %422 ]
  %431 = phi ptr [ %435, %428 ], [ %425, %422 ]
  %432 = mul i64 %430, 5
  %433 = zext i8 %429 to i64
  %434 = add i64 %432, %433
  %435 = getelementptr inbounds nuw i8, ptr %431, i64 1
  %436 = load i8, ptr %435, align 1, !tbaa !27
  %437 = icmp eq i8 %436, 0
  br i1 %437, label %438, label %428, !llvm.loop !28

438:                                              ; preds = %428, %422
  %439 = phi i64 [ 0, %422 ], [ %434, %428 ]
  %440 = ptrtoint ptr %423 to i64
  %441 = ptrtoint ptr %424 to i64
  %442 = sub i64 %440, %441
  %443 = ashr exact i64 %442, 3
  %444 = urem i64 %439, %443
  br label %445

445:                                              ; preds = %449, %438
  %446 = phi i64 [ %444, %438 ], [ %447, %449 ]
  %447 = add i64 %446, 1
  %448 = icmp ult i64 %447, %443
  br i1 %448, label %449, label %151

449:                                              ; preds = %445
  %450 = getelementptr inbounds nuw ptr, ptr %424, i64 %447
  %451 = load ptr, ptr %450, align 8, !tbaa !30
  %452 = icmp eq ptr %451, null
  br i1 %452, label %445, label %420, !llvm.loop !45

453:                                              ; preds = %117
  %454 = getelementptr inbounds nuw i8, ptr %118, i64 16
  %455 = load i32, ptr %454, align 4, !tbaa !37
  br label %456

456:                                              ; preds = %453, %128
  %457 = phi i32 [ 0, %128 ], [ %455, %453 ]
  %458 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEi(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, i32 noundef %457)
          to label %459 unwind label %692

459:                                              ; preds = %456
  %460 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %458, ptr noundef nonnull @.str.2, i64 noundef 1)
          to label %461 unwind label %692

461:                                              ; preds = %459
  %462 = load i64, ptr %30, align 8, !tbaa !16
  %463 = add i64 %462, 1
  invoke void @_ZN9__gnu_cxx9hashtableISt4pairIKPKciES3_NS_4hashIS3_EESt10_Select1stIS5_E5eqstrSaIiEE6resizeEm(ptr noundef nonnull align 8 dereferenceable(40) %4, i64 noundef %463)
          to label %464 unwind label %694

464:                                              ; preds = %461
  %465 = load ptr, ptr %31, align 8, !tbaa !26
  %466 = load ptr, ptr %14, align 8, !tbaa !11
  %467 = ptrtoint ptr %465 to i64
  %468 = ptrtoint ptr %466 to i64
  %469 = sub i64 %467, %468
  %470 = ashr exact i64 %469, 3
  %471 = urem i64 10118267, %470
  %472 = getelementptr inbounds nuw ptr, ptr %466, i64 %471
  %473 = load ptr, ptr %472, align 8, !tbaa !30
  %474 = icmp eq ptr %473, null
  br i1 %474, label %484, label %475

475:                                              ; preds = %464, %481
  %476 = phi ptr [ %482, %481 ], [ %473, %464 ]
  %477 = getelementptr inbounds nuw i8, ptr %476, i64 8
  %478 = load ptr, ptr %477, align 8, !tbaa !6
  %479 = call i32 @strcmp(ptr noundef nonnull dereferenceable(1) %478, ptr noundef nonnull dereferenceable(9) @.str.3) #17
  %480 = icmp eq i32 %479, 0
  br i1 %480, label %491, label %481

481:                                              ; preds = %475
  %482 = load ptr, ptr %476, align 8, !tbaa !32
  %483 = icmp eq ptr %482, null
  br i1 %483, label %484, label %475, !llvm.loop !36

484:                                              ; preds = %481, %464
  %485 = invoke noalias noundef nonnull dereferenceable(24) ptr @_Znwm(i64 noundef 24) #18
          to label %486 unwind label %694

486:                                              ; preds = %484
  %487 = getelementptr inbounds nuw i8, ptr %485, i64 8
  store ptr @.str.3, ptr %487, align 8
  %488 = getelementptr inbounds nuw i8, ptr %485, i64 16
  store i32 0, ptr %488, align 8
  store ptr %473, ptr %485, align 8, !tbaa !32
  store ptr %485, ptr %472, align 8, !tbaa !30
  %489 = load i64, ptr %30, align 8, !tbaa !16
  %490 = add i64 %489, 1
  store i64 %490, ptr %30, align 8, !tbaa !16
  br label %494

491:                                              ; preds = %475
  %492 = getelementptr inbounds nuw i8, ptr %476, i64 16
  %493 = load i32, ptr %492, align 4, !tbaa !37
  br label %494

494:                                              ; preds = %491, %486
  %495 = phi i32 [ 0, %486 ], [ %493, %491 ]
  %496 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEi(ptr noundef nonnull align 8 dereferenceable(8) %458, i32 noundef %495)
          to label %497 unwind label %694

497:                                              ; preds = %494
  %498 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %496, ptr noundef nonnull @.str.2, i64 noundef 1)
          to label %499 unwind label %694

499:                                              ; preds = %497
  %500 = getelementptr inbounds nuw i8, ptr %5, i64 32
  %501 = load i64, ptr %500, align 8, !tbaa !16
  %502 = add i64 %501, 1
  invoke void @_ZN9__gnu_cxx9hashtableISt4pairIKPKciES3_NS_4hashIS3_EESt10_Select1stIS5_E5eqstrSaIiEE6resizeEm(ptr noundef nonnull align 8 dereferenceable(40) %5, i64 noundef %502)
          to label %503 unwind label %696

503:                                              ; preds = %499
  %504 = getelementptr inbounds nuw i8, ptr %5, i64 16
  %505 = load ptr, ptr %504, align 8, !tbaa !26
  %506 = load ptr, ptr %28, align 8, !tbaa !11
  %507 = ptrtoint ptr %505 to i64
  %508 = ptrtoint ptr %506 to i64
  %509 = sub i64 %507, %508
  %510 = ashr exact i64 %509, 3
  %511 = urem i64 80924, %510
  %512 = getelementptr inbounds nuw ptr, ptr %506, i64 %511
  %513 = load ptr, ptr %512, align 8, !tbaa !30
  %514 = icmp eq ptr %513, null
  br i1 %514, label %524, label %515

515:                                              ; preds = %503, %521
  %516 = phi ptr [ %522, %521 ], [ %513, %503 ]
  %517 = getelementptr inbounds nuw i8, ptr %516, i64 8
  %518 = load ptr, ptr %517, align 8, !tbaa !6
  %519 = call i32 @strcmp(ptr noundef nonnull dereferenceable(1) %518, ptr noundef nonnull dereferenceable(6) @.str.1) #17
  %520 = icmp eq i32 %519, 0
  br i1 %520, label %531, label %521

521:                                              ; preds = %515
  %522 = load ptr, ptr %516, align 8, !tbaa !32
  %523 = icmp eq ptr %522, null
  br i1 %523, label %524, label %515, !llvm.loop !36

524:                                              ; preds = %521, %503
  %525 = invoke noalias noundef nonnull dereferenceable(24) ptr @_Znwm(i64 noundef 24) #18
          to label %526 unwind label %696

526:                                              ; preds = %524
  %527 = getelementptr inbounds nuw i8, ptr %525, i64 8
  store ptr @.str.1, ptr %527, align 8
  %528 = getelementptr inbounds nuw i8, ptr %525, i64 16
  store i32 0, ptr %528, align 8
  store ptr %513, ptr %525, align 8, !tbaa !32
  store ptr %525, ptr %512, align 8, !tbaa !30
  %529 = load i64, ptr %500, align 8, !tbaa !16
  %530 = add i64 %529, 1
  store i64 %530, ptr %500, align 8, !tbaa !16
  br label %534

531:                                              ; preds = %515
  %532 = getelementptr inbounds nuw i8, ptr %516, i64 16
  %533 = load i32, ptr %532, align 4, !tbaa !37
  br label %534

534:                                              ; preds = %531, %526
  %535 = phi i32 [ 0, %526 ], [ %533, %531 ]
  %536 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEi(ptr noundef nonnull align 8 dereferenceable(8) %496, i32 noundef %535)
          to label %537 unwind label %696

537:                                              ; preds = %534
  %538 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %536, ptr noundef nonnull @.str.2, i64 noundef 1)
          to label %539 unwind label %696

539:                                              ; preds = %537
  %540 = load i64, ptr %500, align 8, !tbaa !16
  %541 = add i64 %540, 1
  invoke void @_ZN9__gnu_cxx9hashtableISt4pairIKPKciES3_NS_4hashIS3_EESt10_Select1stIS5_E5eqstrSaIiEE6resizeEm(ptr noundef nonnull align 8 dereferenceable(40) %5, i64 noundef %541)
          to label %542 unwind label %698

542:                                              ; preds = %539
  %543 = load ptr, ptr %504, align 8, !tbaa !26
  %544 = load ptr, ptr %28, align 8, !tbaa !11
  %545 = ptrtoint ptr %543 to i64
  %546 = ptrtoint ptr %544 to i64
  %547 = sub i64 %545, %546
  %548 = ashr exact i64 %547, 3
  %549 = urem i64 10118267, %548
  %550 = getelementptr inbounds nuw ptr, ptr %544, i64 %549
  %551 = load ptr, ptr %550, align 8, !tbaa !30
  %552 = icmp eq ptr %551, null
  br i1 %552, label %562, label %553

553:                                              ; preds = %542, %559
  %554 = phi ptr [ %560, %559 ], [ %551, %542 ]
  %555 = getelementptr inbounds nuw i8, ptr %554, i64 8
  %556 = load ptr, ptr %555, align 8, !tbaa !6
  %557 = call i32 @strcmp(ptr noundef nonnull dereferenceable(1) %556, ptr noundef nonnull dereferenceable(9) @.str.3) #17
  %558 = icmp eq i32 %557, 0
  br i1 %558, label %569, label %559

559:                                              ; preds = %553
  %560 = load ptr, ptr %554, align 8, !tbaa !32
  %561 = icmp eq ptr %560, null
  br i1 %561, label %562, label %553, !llvm.loop !36

562:                                              ; preds = %559, %542
  %563 = invoke noalias noundef nonnull dereferenceable(24) ptr @_Znwm(i64 noundef 24) #18
          to label %564 unwind label %698

564:                                              ; preds = %562
  %565 = getelementptr inbounds nuw i8, ptr %563, i64 8
  store ptr @.str.3, ptr %565, align 8
  %566 = getelementptr inbounds nuw i8, ptr %563, i64 16
  store i32 0, ptr %566, align 8
  store ptr %551, ptr %563, align 8, !tbaa !32
  store ptr %563, ptr %550, align 8, !tbaa !30
  %567 = load i64, ptr %500, align 8, !tbaa !16
  %568 = add i64 %567, 1
  store i64 %568, ptr %500, align 8, !tbaa !16
  br label %572

569:                                              ; preds = %553
  %570 = getelementptr inbounds nuw i8, ptr %554, i64 16
  %571 = load i32, ptr %570, align 4, !tbaa !37
  br label %572

572:                                              ; preds = %569, %564
  %573 = phi i32 [ 0, %564 ], [ %571, %569 ]
  %574 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEi(ptr noundef nonnull align 8 dereferenceable(8) %536, i32 noundef %573)
          to label %575 unwind label %698

575:                                              ; preds = %572
  %576 = load ptr, ptr %574, align 8, !tbaa !46
  %577 = getelementptr i8, ptr %576, i64 -24
  %578 = load i64, ptr %577, align 8
  %579 = getelementptr inbounds i8, ptr %574, i64 %578
  %580 = getelementptr inbounds nuw i8, ptr %579, i64 240
  %581 = load ptr, ptr %580, align 8, !tbaa !48
  %582 = icmp eq ptr %581, null
  br i1 %582, label %583, label %585

583:                                              ; preds = %575
  invoke void @_ZSt16__throw_bad_castv() #19
          to label %584 unwind label %698

584:                                              ; preds = %583
  unreachable

585:                                              ; preds = %575
  %586 = getelementptr inbounds nuw i8, ptr %581, i64 56
  %587 = load i8, ptr %586, align 8, !tbaa !64
  %588 = icmp eq i8 %587, 0
  br i1 %588, label %592, label %589

589:                                              ; preds = %585
  %590 = getelementptr inbounds nuw i8, ptr %581, i64 67
  %591 = load i8, ptr %590, align 1, !tbaa !27
  br label %598

592:                                              ; preds = %585
  invoke void @_ZNKSt5ctypeIcE13_M_widen_initEv(ptr noundef nonnull align 8 dereferenceable(570) %581)
          to label %593 unwind label %698

593:                                              ; preds = %592
  %594 = load ptr, ptr %581, align 8, !tbaa !46
  %595 = getelementptr inbounds nuw i8, ptr %594, i64 48
  %596 = load ptr, ptr %595, align 8
  %597 = invoke noundef i8 %596(ptr noundef nonnull align 8 dereferenceable(570) %581, i8 noundef 10)
          to label %598 unwind label %698

598:                                              ; preds = %593, %589
  %599 = phi i8 [ %591, %589 ], [ %597, %593 ]
  %600 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8) %574, i8 noundef %599)
          to label %601 unwind label %698

601:                                              ; preds = %598
  %602 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo5flushEv(ptr noundef nonnull align 8 dereferenceable(8) %600)
          to label %603 unwind label %698

603:                                              ; preds = %601
  %604 = load i64, ptr %500, align 8, !tbaa !16
  %605 = icmp eq i64 %604, 0
  br i1 %605, label %606, label %608

606:                                              ; preds = %603
  %607 = load ptr, ptr %28, align 8, !tbaa !11
  br label %638

608:                                              ; preds = %603
  %609 = load ptr, ptr %504, align 8, !tbaa !26
  %610 = load ptr, ptr %28, align 8, !tbaa !11
  %611 = icmp eq ptr %609, %610
  br i1 %611, label %612, label %614

612:                                              ; preds = %628, %608
  %613 = phi ptr [ %610, %608 ], [ %630, %628 ]
  store i64 0, ptr %500, align 8, !tbaa !16
  br label %638

614:                                              ; preds = %608, %628
  %615 = phi ptr [ %629, %628 ], [ %609, %608 ]
  %616 = phi ptr [ %630, %628 ], [ %610, %608 ]
  %617 = phi i64 [ %632, %628 ], [ 0, %608 ]
  %618 = getelementptr inbounds nuw ptr, ptr %616, i64 %617
  %619 = load ptr, ptr %618, align 8, !tbaa !30
  %620 = icmp eq ptr %619, null
  br i1 %620, label %628, label %621

621:                                              ; preds = %614, %621
  %622 = phi ptr [ %623, %621 ], [ %619, %614 ]
  %623 = load ptr, ptr %622, align 8, !tbaa !32
  call void @_ZdlPvm(ptr noundef nonnull %622, i64 noundef 24) #16
  %624 = icmp eq ptr %623, null
  br i1 %624, label %625, label %621, !llvm.loop !70

625:                                              ; preds = %621
  %626 = load ptr, ptr %28, align 8, !tbaa !11
  %627 = load ptr, ptr %504, align 8, !tbaa !26
  br label %628

628:                                              ; preds = %625, %614
  %629 = phi ptr [ %627, %625 ], [ %615, %614 ]
  %630 = phi ptr [ %626, %625 ], [ %616, %614 ]
  %631 = getelementptr inbounds nuw ptr, ptr %630, i64 %617
  store ptr null, ptr %631, align 8, !tbaa !30
  %632 = add nuw i64 %617, 1
  %633 = ptrtoint ptr %629 to i64
  %634 = ptrtoint ptr %630 to i64
  %635 = sub i64 %633, %634
  %636 = ashr exact i64 %635, 3
  %637 = icmp ult i64 %632, %636
  br i1 %637, label %614, label %612, !llvm.loop !71

638:                                              ; preds = %612, %606
  %639 = phi ptr [ %607, %606 ], [ %613, %612 ]
  %640 = icmp eq ptr %639, null
  br i1 %640, label %647, label %641

641:                                              ; preds = %638
  %642 = getelementptr inbounds nuw i8, ptr %5, i64 24
  %643 = load ptr, ptr %642, align 8, !tbaa !15
  %644 = ptrtoint ptr %643 to i64
  %645 = ptrtoint ptr %639 to i64
  %646 = sub i64 %644, %645
  call void @_ZdlPvm(ptr noundef nonnull %639, i64 noundef %646) #16
  br label %647

647:                                              ; preds = %638, %641
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #15
  %648 = load i64, ptr %30, align 8, !tbaa !16
  %649 = icmp eq i64 %648, 0
  br i1 %649, label %650, label %652

650:                                              ; preds = %647
  %651 = load ptr, ptr %14, align 8, !tbaa !11
  br label %682

652:                                              ; preds = %647
  %653 = load ptr, ptr %31, align 8, !tbaa !26
  %654 = load ptr, ptr %14, align 8, !tbaa !11
  %655 = icmp eq ptr %653, %654
  br i1 %655, label %656, label %658

656:                                              ; preds = %672, %652
  %657 = phi ptr [ %654, %652 ], [ %674, %672 ]
  store i64 0, ptr %30, align 8, !tbaa !16
  br label %682

658:                                              ; preds = %652, %672
  %659 = phi ptr [ %673, %672 ], [ %653, %652 ]
  %660 = phi ptr [ %674, %672 ], [ %654, %652 ]
  %661 = phi i64 [ %676, %672 ], [ 0, %652 ]
  %662 = getelementptr inbounds nuw ptr, ptr %660, i64 %661
  %663 = load ptr, ptr %662, align 8, !tbaa !30
  %664 = icmp eq ptr %663, null
  br i1 %664, label %672, label %665

665:                                              ; preds = %658, %665
  %666 = phi ptr [ %667, %665 ], [ %663, %658 ]
  %667 = load ptr, ptr %666, align 8, !tbaa !32
  call void @_ZdlPvm(ptr noundef nonnull %666, i64 noundef 24) #16
  %668 = icmp eq ptr %667, null
  br i1 %668, label %669, label %665, !llvm.loop !70

669:                                              ; preds = %665
  %670 = load ptr, ptr %14, align 8, !tbaa !11
  %671 = load ptr, ptr %31, align 8, !tbaa !26
  br label %672

672:                                              ; preds = %669, %658
  %673 = phi ptr [ %671, %669 ], [ %659, %658 ]
  %674 = phi ptr [ %670, %669 ], [ %660, %658 ]
  %675 = getelementptr inbounds nuw ptr, ptr %674, i64 %661
  store ptr null, ptr %675, align 8, !tbaa !30
  %676 = add nuw i64 %661, 1
  %677 = ptrtoint ptr %673 to i64
  %678 = ptrtoint ptr %674 to i64
  %679 = sub i64 %677, %678
  %680 = ashr exact i64 %679, 3
  %681 = icmp ult i64 %676, %680
  br i1 %681, label %658, label %656, !llvm.loop !71

682:                                              ; preds = %656, %650
  %683 = phi ptr [ %651, %650 ], [ %657, %656 ]
  %684 = icmp eq ptr %683, null
  br i1 %684, label %691, label %685

685:                                              ; preds = %682
  %686 = getelementptr inbounds nuw i8, ptr %4, i64 24
  %687 = load ptr, ptr %686, align 8, !tbaa !15
  %688 = ptrtoint ptr %687 to i64
  %689 = ptrtoint ptr %683 to i64
  %690 = sub i64 %688, %689
  call void @_ZdlPvm(ptr noundef nonnull %683, i64 noundef %690) #16
  br label %691

691:                                              ; preds = %682, %685
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #15
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #15
  ret i32 0

692:                                              ; preds = %459, %126, %103, %456
  %693 = landingpad { ptr, i32 }
          cleanup
  br label %700

694:                                              ; preds = %497, %484, %461, %494
  %695 = landingpad { ptr, i32 }
          cleanup
  br label %700

696:                                              ; preds = %537, %524, %499, %534
  %697 = landingpad { ptr, i32 }
          cleanup
  br label %700

698:                                              ; preds = %601, %598, %593, %592, %583, %562, %539, %572
  %699 = landingpad { ptr, i32 }
          cleanup
  br label %700

700:                                              ; preds = %154, %156, %692, %696, %698, %694, %101
  %701 = phi { ptr, i32 } [ %102, %101 ], [ %693, %692 ], [ %695, %694 ], [ %699, %698 ], [ %697, %696 ], [ %155, %154 ], [ %157, %156 ]
  call void @_ZN9__gnu_cxx8hash_mapIPKciNS_4hashIS2_EE5eqstrSaIiEED2Ev(ptr noundef nonnull align 8 dereferenceable(40) %5) #15
  br label %702

702:                                              ; preds = %36, %32, %700
  %703 = phi { ptr, i32 } [ %701, %700 ], [ %33, %36 ], [ %33, %32 ]
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #15
  call void @_ZN9__gnu_cxx8hash_mapIPKciNS_4hashIS2_EE5eqstrSaIiEED2Ev(ptr noundef nonnull align 8 dereferenceable(40) %4) #15
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #15
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #15
  br label %25
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

declare i32 @__gxx_personality_v0(...)

; Function Attrs: nofree nounwind
declare noundef i32 @sprintf(ptr noalias noundef writeonly captures(none), ptr noundef readonly captures(none), ...) local_unnamed_addr #2

; Function Attrs: mustprogress nofree nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
declare noalias ptr @strdup(ptr noundef readonly captures(none)) local_unnamed_addr #3

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
  br i1 %25, label %26, label %22, !llvm.loop !70

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
  br i1 %38, label %15, label %13, !llvm.loop !71

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
  %9 = load i64, ptr %8, align 8, !tbaa !41
  %10 = icmp ult i64 %9, %1
  %11 = getelementptr inbounds nuw i8, ptr %8, i64 8
  %12 = xor i64 %7, -1
  %13 = add nsw i64 %6, %12
  %14 = select i1 %10, i64 %13, i64 %7
  %15 = select i1 %10, ptr %11, ptr %5
  %16 = icmp sgt i64 %14, 0
  br i1 %16, label %4, label %17, !llvm.loop !42

17:                                               ; preds = %4
  %18 = icmp eq ptr %15, getelementptr inbounds nuw (i8, ptr @_ZN9__gnu_cxx21_Hashtable_prime_listImE16__stl_prime_listE, i64 232)
  %19 = select i1 %18, ptr getelementptr inbounds nuw (i8, ptr @_ZN9__gnu_cxx21_Hashtable_prime_listImE16__stl_prime_listE, i64 224), ptr %15
  %20 = load i64, ptr %19, align 8, !tbaa !41
  %21 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %22 = icmp ugt i64 %20, 1152921504606846975
  br i1 %22, label %23, label %24

23:                                               ; preds = %17
  tail call void @_ZSt20__throw_length_errorPKc(ptr noundef nonnull @.str.4) #19
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
  %34 = load ptr, ptr %33, align 8, !tbaa !72
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
  br i1 %26, label %27, label %30, !prof !73

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
  br i1 %39, label %40, label %43, !prof !73

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
  br i1 %67, label %68, label %61, !llvm.loop !74

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
  br i1 %75, label %223, label %72, !llvm.loop !77

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
  br i1 %100, label %101, label %94, !llvm.loop !78

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
  br i1 %108, label %109, label %105, !llvm.loop !79

109:                                              ; preds = %105, %101, %76
  %110 = phi ptr [ %10, %76 ], [ %81, %101 ], [ %81, %105 ]
  store ptr %110, ptr %9, align 8, !tbaa !26
  %111 = icmp sgt i64 %19, 8
  br i1 %111, label %112, label %114, !prof !73

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
  br i1 %140, label %141, label %134, !llvm.loop !80

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
  br i1 %148, label %223, label %145, !llvm.loop !81

149:                                              ; preds = %6
  %150 = load ptr, ptr %0, align 8, !tbaa !11
  %151 = ptrtoint ptr %150 to i64
  %152 = sub i64 %12, %151
  %153 = ashr exact i64 %152, 3
  %154 = sub nsw i64 1152921504606846975, %153
  %155 = icmp ult i64 %154, %2
  br i1 %155, label %156, label %157

156:                                              ; preds = %149
  tail call void @_ZSt20__throw_length_errorPKc(ptr noundef nonnull @.str.5) #19
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
  br i1 %191, label %192, label %185, !llvm.loop !82

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
  br i1 %199, label %200, label %196, !llvm.loop !83

200:                                              ; preds = %196, %192
  %201 = icmp sgt i64 %164, 8
  br i1 %201, label %202, label %203, !prof !73

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
  br i1 %210, label %211, label %212, !prof !73

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
  %17 = load i64, ptr %16, align 8, !tbaa !41
  %18 = icmp ult i64 %17, %1
  %19 = getelementptr inbounds nuw i8, ptr %16, i64 8
  %20 = xor i64 %15, -1
  %21 = add nsw i64 %14, %20
  %22 = select i1 %18, i64 %21, i64 %15
  %23 = select i1 %18, ptr %19, ptr %13
  %24 = icmp sgt i64 %22, 0
  br i1 %24, label %12, label %25, !llvm.loop !42

25:                                               ; preds = %12
  %26 = icmp eq ptr %23, getelementptr inbounds nuw (i8, ptr @_ZN9__gnu_cxx21_Hashtable_prime_listImE16__stl_prime_listE, i64 232)
  %27 = select i1 %26, ptr getelementptr inbounds nuw (i8, ptr @_ZN9__gnu_cxx21_Hashtable_prime_listImE16__stl_prime_listE, i64 224), ptr %23
  %28 = load i64, ptr %27, align 8, !tbaa !41
  %29 = icmp ugt i64 %28, %10
  br i1 %29, label %30, label %82

30:                                               ; preds = %25
  %31 = icmp ugt i64 %28, 1152921504606846975
  br i1 %31, label %32, label %33

32:                                               ; preds = %30
  tail call void @_ZSt20__throw_length_errorPKc(ptr noundef nonnull @.str.6) #19
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
  br i1 %75, label %76, label %52, !llvm.loop !43

76:                                               ; preds = %68, %47
  %77 = add nuw i64 %48, 1
  %78 = icmp eq i64 %77, %10
  br i1 %78, label %79, label %47, !llvm.loop !44

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

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8), ptr noundef, i64 noundef) local_unnamed_addr #4

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
!40 = distinct !{!40, !29}
!41 = !{!25, !25, i64 0}
!42 = distinct !{!42, !29}
!43 = distinct !{!43, !29}
!44 = distinct !{!44, !29}
!45 = distinct !{!45, !29}
!46 = !{!47, !47, i64 0}
!47 = !{!"vtable pointer", !10, i64 0}
!48 = !{!49, !61, i64 240}
!49 = !{!"_ZTSSt9basic_iosIcSt11char_traitsIcEE", !50, i64 0, !58, i64 216, !9, i64 224, !59, i64 225, !60, i64 232, !61, i64 240, !62, i64 248, !63, i64 256}
!50 = !{!"_ZTSSt8ios_base", !25, i64 8, !25, i64 16, !51, i64 24, !52, i64 28, !52, i64 32, !53, i64 40, !54, i64 48, !9, i64 64, !35, i64 192, !55, i64 200, !56, i64 208}
!51 = !{!"_ZTSSt13_Ios_Fmtflags", !9, i64 0}
!52 = !{!"_ZTSSt12_Ios_Iostate", !9, i64 0}
!53 = !{!"p1 _ZTSNSt8ios_base14_Callback_listE", !8, i64 0}
!54 = !{!"_ZTSNSt8ios_base6_WordsE", !8, i64 0, !25, i64 8}
!55 = !{!"p1 _ZTSNSt8ios_base6_WordsE", !8, i64 0}
!56 = !{!"_ZTSSt6locale", !57, i64 0}
!57 = !{!"p1 _ZTSNSt6locale5_ImplE", !8, i64 0}
!58 = !{!"p1 _ZTSSo", !8, i64 0}
!59 = !{!"bool", !9, i64 0}
!60 = !{!"p1 _ZTSSt15basic_streambufIcSt11char_traitsIcEE", !8, i64 0}
!61 = !{!"p1 _ZTSSt5ctypeIcE", !8, i64 0}
!62 = !{!"p1 _ZTSSt7num_putIcSt19ostreambuf_iteratorIcSt11char_traitsIcEEE", !8, i64 0}
!63 = !{!"p1 _ZTSSt7num_getIcSt19istreambuf_iteratorIcSt11char_traitsIcEEE", !8, i64 0}
!64 = !{!65, !9, i64 56}
!65 = !{!"_ZTSSt5ctypeIcE", !66, i64 0, !67, i64 16, !59, i64 24, !68, i64 32, !68, i64 40, !69, i64 48, !9, i64 56, !9, i64 57, !9, i64 313, !9, i64 569}
!66 = !{!"_ZTSNSt6locale5facetE", !35, i64 8}
!67 = !{!"p1 _ZTS15__locale_struct", !8, i64 0}
!68 = !{!"p1 int", !8, i64 0}
!69 = !{!"p1 short", !8, i64 0}
!70 = distinct !{!70, !29}
!71 = distinct !{!71, !29}
!72 = !{!13, !13, i64 0}
!73 = !{!"branch_weights", !"expected", i32 2000, i32 1}
!74 = distinct !{!74, !29, !75, !76}
!75 = !{!"llvm.loop.isvectorized", i32 1}
!76 = !{!"llvm.loop.unroll.runtime.disable"}
!77 = distinct !{!77, !29, !76, !75}
!78 = distinct !{!78, !29, !75, !76}
!79 = distinct !{!79, !29, !76, !75}
!80 = distinct !{!80, !29, !75, !76}
!81 = distinct !{!81, !29, !76, !75}
!82 = distinct !{!82, !29, !75, !76}
!83 = distinct !{!83, !29, !76, !75}
