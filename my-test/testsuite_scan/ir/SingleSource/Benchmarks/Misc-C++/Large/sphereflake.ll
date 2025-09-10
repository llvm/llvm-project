; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Misc-C++/Large/sphereflake.cpp'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Misc-C++/Large/sphereflake.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

module asm ".globl _ZSt21ios_base_library_initv"

%struct.v_t = type { double, double, double }
%"class.std::basic_ostream" = type { ptr, %"class.std::basic_ios" }
%"class.std::basic_ios" = type { %"class.std::ios_base", ptr, i8, i8, ptr, ptr, ptr, ptr }
%"class.std::ios_base" = type { ptr, i64, i64, i32, i32, i32, ptr, %"struct.std::ios_base::_Words", [8 x %"struct.std::ios_base::_Words"], i32, ptr, %"class.std::locale" }
%"struct.std::ios_base::_Words" = type { ptr, i64 }
%"class.std::locale" = type { ptr }
%struct.node_t = type { %struct.sphere_t, %struct.sphere_t, i64 }
%struct.sphere_t = type { %struct.v_t, double }

@_ZL5light = internal global %struct.v_t zeroinitializer, align 16
@_ZSt4cout = external global %"class.std::basic_ostream", align 8
@.str = private unnamed_addr constant [4 x i8] c"P2\0A\00", align 1
@.str.1 = private unnamed_addr constant [2 x i8] c" \00", align 1
@.str.2 = private unnamed_addr constant [6 x i8] c"\0A256\0A\00", align 1
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_sphereflake.cpp, ptr null }]

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare ptr @llvm.invariant.start.p0(i64 immarg, ptr captures(none)) #0

; Function Attrs: mustprogress norecurse uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #1 personality ptr @__gxx_personality_v0 {
  %3 = alloca [4 x %struct.v_t], align 16
  %4 = icmp eq i32 %0, 2
  br i1 %4, label %5, label %12

5:                                                ; preds = %2
  %6 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %7 = load ptr, ptr %6, align 8, !tbaa !6
  %8 = tail call i64 @__isoc23_strtol(ptr noundef nonnull %7, ptr noundef null, i32 noundef 10) #10
  %9 = trunc i64 %8 to i32
  %10 = tail call i32 @llvm.smax.i32(i32 %9, i32 2)
  %11 = icmp sgt i32 %9, 2
  br i1 %11, label %12, label %23

12:                                               ; preds = %2, %5
  %13 = phi i32 [ %10, %5 ], [ 6, %2 ]
  br label %14

14:                                               ; preds = %12, %14
  %15 = phi i32 [ %19, %14 ], [ 9, %12 ]
  %16 = phi i32 [ %17, %14 ], [ %13, %12 ]
  %17 = add nsw i32 %16, -1
  %18 = mul nsw i32 %15, 9
  %19 = add nsw i32 %18, 9
  %20 = icmp samesign ugt i32 %16, 3
  br i1 %20, label %14, label %21, !llvm.loop !11

21:                                               ; preds = %14
  %22 = add nsw i32 %18, 10
  br label %23

23:                                               ; preds = %21, %5
  %24 = phi i32 [ 2, %5 ], [ %13, %21 ]
  %25 = phi i32 [ 10, %5 ], [ %22, %21 ]
  %26 = sext i32 %25 to i64
  %27 = tail call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %26, i64 72)
  %28 = extractvalue { i64, i1 } %27, 1
  %29 = extractvalue { i64, i1 } %27, 0
  %30 = select i1 %28, i64 -1, i64 %29
  %31 = tail call noalias noundef nonnull ptr @_Znam(i64 noundef %30) #11
  %32 = mul nsw i64 %26, 72
  %33 = getelementptr inbounds i8, ptr %31, i64 %32
  br label %34

34:                                               ; preds = %23, %34
  %35 = phi i32 [ %43, %34 ], [ 100, %23 ]
  %36 = phi double [ %39, %34 ], [ 1.000000e+00, %23 ]
  %37 = fdiv double 1.312500e+00, %36
  %38 = fadd double %36, %37
  %39 = fmul double %38, 5.000000e-01
  %40 = fsub double %39, %36
  %41 = tail call double @llvm.fabs.f64(double %40)
  %42 = fcmp ule double %41, 0x3D719799812DEA11
  %43 = add nsw i32 %35, -1
  %44 = icmp eq i32 %43, 0
  %45 = select i1 %42, i1 true, i1 %44
  br i1 %45, label %46, label %34, !llvm.loop !13

46:                                               ; preds = %34
  %47 = fdiv double 1.000000e+00, %39
  %48 = fmul double %47, 2.500000e-01
  %49 = fmul double %47, -5.000000e-01
  %50 = insertvalue [3 x double] poison, double %48, 0
  %51 = insertvalue [3 x double] %50, double %47, 1
  %52 = insertvalue [3 x double] %51, double %49, 2
  %53 = tail call fastcc noundef ptr @_ZL6createP6node_tii3v_tS1_d(ptr noundef nonnull %31, i32 noundef %24, i32 noundef %25, [3 x double] alignstack(8) zeroinitializer, [3 x double] alignstack(8) %52, double noundef 1.000000e+00)
  %54 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str, i64 noundef 3)
  %55 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEi(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, i32 noundef 1024)
  %56 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %55, ptr noundef nonnull @.str.1, i64 noundef 1)
  %57 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEi(ptr noundef nonnull align 8 dereferenceable(8) %55, i32 noundef 1024)
  %58 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %57, ptr noundef nonnull @.str.2, i64 noundef 5)
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #10
  store <2 x double> <double -5.125000e+02, double 0xC080015555555555>, ptr %3, align 16, !tbaa !14
  %59 = getelementptr inbounds nuw i8, ptr %3, i64 16
  store <2 x double> <double 0.000000e+00, double 0xC07FFD5555555555>, ptr %59, align 16, !tbaa !14
  %60 = getelementptr inbounds nuw i8, ptr %3, i64 32
  store <2 x double> <double -5.125000e+02, double 0.000000e+00>, ptr %60, align 16, !tbaa !14
  %61 = getelementptr inbounds nuw i8, ptr %3, i64 48
  store <2 x double> <double 0xC080015555555555, double -5.115000e+02>, ptr %61, align 16, !tbaa !14
  %62 = getelementptr inbounds nuw i8, ptr %3, i64 64
  store <2 x double> <double 0.000000e+00, double -5.115000e+02>, ptr %62, align 16, !tbaa !14
  %63 = getelementptr inbounds nuw i8, ptr %3, i64 80
  store <2 x double> <double 0xC07FFD5555555555, double 0.000000e+00>, ptr %63, align 16, !tbaa !14
  %64 = icmp sgt i32 %25, 0
  br label %65

65:                                               ; preds = %387, %46
  %66 = phi i32 [ %389, %387 ], [ 1024, %46 ]
  %67 = phi double [ %388, %387 ], [ 1.023000e+03, %46 ]
  br label %89

68:                                               ; preds = %387
  %69 = load ptr, ptr @_ZSt4cout, align 8, !tbaa !16
  %70 = getelementptr i8, ptr %69, i64 -24
  %71 = load i64, ptr %70, align 8
  %72 = getelementptr inbounds i8, ptr @_ZSt4cout, i64 %71
  %73 = getelementptr inbounds nuw i8, ptr %72, i64 240
  %74 = load ptr, ptr %73, align 8, !tbaa !18
  %75 = icmp eq ptr %74, null
  br i1 %75, label %76, label %77

76:                                               ; preds = %68
  tail call void @_ZSt16__throw_bad_castv() #12
  unreachable

77:                                               ; preds = %68
  %78 = getelementptr inbounds nuw i8, ptr %74, i64 56
  %79 = load i8, ptr %78, align 8, !tbaa !36
  %80 = icmp eq i8 %79, 0
  br i1 %80, label %84, label %81

81:                                               ; preds = %77
  %82 = getelementptr inbounds nuw i8, ptr %74, i64 67
  %83 = load i8, ptr %82, align 1, !tbaa !42
  br label %401

84:                                               ; preds = %77
  tail call void @_ZNKSt5ctypeIcE13_M_widen_initEv(ptr noundef nonnull align 8 dereferenceable(570) %74)
  %85 = load ptr, ptr %74, align 8, !tbaa !16
  %86 = getelementptr inbounds nuw i8, ptr %85, i64 48
  %87 = load ptr, ptr %86, align 8
  %88 = tail call noundef i8 %87(ptr noundef nonnull align 8 dereferenceable(570) %74, i8 noundef 10)
  br label %401

89:                                               ; preds = %394, %65
  %90 = phi i32 [ 1024, %65 ], [ %399, %394 ]
  %91 = phi double [ 0.000000e+00, %65 ], [ %398, %394 ]
  %92 = load double, ptr @_ZL5light, align 8
  %93 = load double, ptr getelementptr inbounds nuw (i8, ptr @_ZL5light, i64 8), align 8
  %94 = load double, ptr getelementptr inbounds nuw (i8, ptr @_ZL5light, i64 16), align 8
  %95 = fneg double %92
  %96 = fneg double %93
  %97 = fneg double %94
  br i1 %64, label %98, label %394

98:                                               ; preds = %89, %382
  %99 = phi i64 [ %385, %382 ], [ 0, %89 ]
  %100 = phi double [ %384, %382 ], [ 0.000000e+00, %89 ]
  %101 = getelementptr inbounds nuw %struct.v_t, ptr %3, i64 %99
  %102 = load double, ptr %101, align 8, !tbaa !43
  %103 = fadd double %91, %102
  %104 = getelementptr inbounds nuw i8, ptr %101, i64 8
  %105 = load double, ptr %104, align 8, !tbaa !45
  %106 = fadd double %67, %105
  %107 = getelementptr inbounds nuw i8, ptr %101, i64 16
  %108 = load double, ptr %107, align 8, !tbaa !46
  %109 = fadd double %108, 1.024000e+03
  %110 = fmul double %106, %106
  %111 = tail call double @llvm.fmuladd.f64(double %103, double %103, double %110)
  %112 = tail call noundef double @llvm.fmuladd.f64(double %109, double %109, double %111)
  %113 = fcmp oeq double %112, 0x7FF0000000000000
  br i1 %113, label %126, label %114

114:                                              ; preds = %98, %114
  %115 = phi i32 [ %123, %114 ], [ 100, %98 ]
  %116 = phi double [ %119, %114 ], [ 1.000000e+00, %98 ]
  %117 = fdiv double %112, %116
  %118 = fadd double %116, %117
  %119 = fmul double %118, 5.000000e-01
  %120 = fsub double %119, %116
  %121 = tail call double @llvm.fabs.f64(double %120)
  %122 = fcmp ule double %121, 0x3D719799812DEA11
  %123 = add nsw i32 %115, -1
  %124 = icmp eq i32 %123, 0
  %125 = select i1 %122, i1 true, i1 %124
  br i1 %125, label %126, label %114, !llvm.loop !13

126:                                              ; preds = %114, %98
  %127 = phi double [ 0x7FF0000000000000, %98 ], [ %119, %114 ]
  %128 = fdiv double 1.000000e+00, %127
  %129 = fmul double %103, %128
  %130 = fmul double %106, %128
  %131 = fmul double %109, %128
  br label %132

132:                                              ; preds = %250, %126
  %133 = phi double [ %251, %250 ], [ 0.000000e+00, %126 ]
  %134 = phi double [ %252, %250 ], [ 0x7FF0000000000000, %126 ]
  %135 = phi double [ %253, %250 ], [ 0.000000e+00, %126 ]
  %136 = phi double [ %254, %250 ], [ 0.000000e+00, %126 ]
  %137 = phi double [ %255, %250 ], [ 0x7FF0000000000000, %126 ]
  %138 = phi ptr [ %256, %250 ], [ %31, %126 ]
  %139 = load double, ptr %138, align 8, !tbaa !43
  %140 = getelementptr inbounds nuw i8, ptr %138, i64 8
  %141 = load double, ptr %140, align 8, !tbaa !45
  %142 = getelementptr inbounds nuw i8, ptr %138, i64 16
  %143 = load double, ptr %142, align 8, !tbaa !46
  %144 = fadd double %143, 4.500000e+00
  %145 = fmul double %130, %141
  %146 = tail call double @llvm.fmuladd.f64(double %129, double %139, double %145)
  %147 = tail call noundef double @llvm.fmuladd.f64(double %131, double %144, double %146)
  %148 = fmul double %141, %141
  %149 = tail call double @llvm.fmuladd.f64(double %139, double %139, double %148)
  %150 = tail call noundef double @llvm.fmuladd.f64(double %144, double %144, double %149)
  %151 = fneg double %150
  %152 = tail call double @llvm.fmuladd.f64(double %147, double %147, double %151)
  %153 = getelementptr inbounds nuw i8, ptr %138, i64 24
  %154 = load double, ptr %153, align 8, !tbaa !47
  %155 = tail call double @llvm.fmuladd.f64(double %154, double %154, double %152)
  %156 = fcmp olt double %155, 0.000000e+00
  br i1 %156, label %179, label %157

157:                                              ; preds = %132
  %158 = fcmp oeq double %155, 0x7FF0000000000000
  br i1 %158, label %171, label %159

159:                                              ; preds = %157, %159
  %160 = phi i32 [ %168, %159 ], [ 100, %157 ]
  %161 = phi double [ %164, %159 ], [ 1.000000e+00, %157 ]
  %162 = fdiv double %155, %161
  %163 = fadd double %161, %162
  %164 = fmul double %163, 5.000000e-01
  %165 = fsub double %164, %161
  %166 = tail call double @llvm.fabs.f64(double %165)
  %167 = fcmp ule double %166, 0x3D719799812DEA11
  %168 = add nsw i32 %160, -1
  %169 = icmp eq i32 %168, 0
  %170 = select i1 %167, i1 true, i1 %169
  br i1 %170, label %171, label %159, !llvm.loop !13

171:                                              ; preds = %159, %157
  %172 = phi double [ 0x7FF0000000000000, %157 ], [ %164, %159 ]
  %173 = fadd double %147, %172
  %174 = fcmp olt double %173, 0.000000e+00
  %175 = fsub double %147, %172
  %176 = fcmp ogt double %175, 0.000000e+00
  %177 = select i1 %176, double %175, double %173
  %178 = select i1 %174, double 0x7FF0000000000000, double %177
  br label %179

179:                                              ; preds = %171, %132
  %180 = phi double [ %178, %171 ], [ 0x7FF0000000000000, %132 ]
  %181 = fcmp ult double %180, %137
  br i1 %181, label %186, label %182

182:                                              ; preds = %179
  %183 = getelementptr inbounds nuw i8, ptr %138, i64 64
  %184 = load i64, ptr %183, align 8, !tbaa !49
  %185 = getelementptr inbounds %struct.node_t, ptr %138, i64 %184
  br label %250

186:                                              ; preds = %179
  %187 = getelementptr inbounds nuw i8, ptr %138, i64 32
  %188 = load double, ptr %187, align 8, !tbaa !43
  %189 = getelementptr inbounds nuw i8, ptr %138, i64 40
  %190 = load double, ptr %189, align 8, !tbaa !45
  %191 = getelementptr inbounds nuw i8, ptr %138, i64 48
  %192 = load double, ptr %191, align 8, !tbaa !46
  %193 = fadd double %192, 4.500000e+00
  %194 = fmul double %130, %190
  %195 = tail call double @llvm.fmuladd.f64(double %129, double %188, double %194)
  %196 = tail call noundef double @llvm.fmuladd.f64(double %131, double %193, double %195)
  %197 = fmul double %190, %190
  %198 = tail call double @llvm.fmuladd.f64(double %188, double %188, double %197)
  %199 = tail call noundef double @llvm.fmuladd.f64(double %193, double %193, double %198)
  %200 = fneg double %199
  %201 = tail call double @llvm.fmuladd.f64(double %196, double %196, double %200)
  %202 = getelementptr inbounds nuw i8, ptr %138, i64 56
  %203 = load double, ptr %202, align 8, !tbaa !47
  %204 = tail call double @llvm.fmuladd.f64(double %203, double %203, double %201)
  %205 = fcmp olt double %204, 0.000000e+00
  br i1 %205, label %243, label %206

206:                                              ; preds = %186
  %207 = fcmp oeq double %204, 0x7FF0000000000000
  br i1 %207, label %220, label %208

208:                                              ; preds = %206, %208
  %209 = phi i32 [ %217, %208 ], [ 100, %206 ]
  %210 = phi double [ %213, %208 ], [ 1.000000e+00, %206 ]
  %211 = fdiv double %204, %210
  %212 = fadd double %210, %211
  %213 = fmul double %212, 5.000000e-01
  %214 = fsub double %213, %210
  %215 = tail call double @llvm.fabs.f64(double %214)
  %216 = fcmp ule double %215, 0x3D719799812DEA11
  %217 = add nsw i32 %209, -1
  %218 = icmp eq i32 %217, 0
  %219 = select i1 %216, i1 true, i1 %218
  br i1 %219, label %220, label %208, !llvm.loop !13

220:                                              ; preds = %208, %206
  %221 = phi double [ 0x7FF0000000000000, %206 ], [ %213, %208 ]
  %222 = fadd double %196, %221
  %223 = fcmp olt double %222, 0.000000e+00
  %224 = fsub double %196, %221
  %225 = fcmp ogt double %224, 0.000000e+00
  %226 = select i1 %225, double %224, double %222
  %227 = select i1 %223, double 0x7FF0000000000000, double %226
  %228 = fcmp olt double %227, %137
  br i1 %228, label %229, label %243

229:                                              ; preds = %220
  %230 = fmul double %129, %227
  %231 = fmul double %130, %227
  %232 = fmul double %131, %227
  %233 = fadd double %230, 0.000000e+00
  %234 = fadd double %231, 0.000000e+00
  %235 = fadd double %232, -4.500000e+00
  %236 = fsub double %233, %188
  %237 = fsub double %234, %190
  %238 = fsub double %235, %192
  %239 = fdiv double 1.000000e+00, %203
  %240 = fmul double %239, %236
  %241 = fmul double %239, %237
  %242 = fmul double %239, %238
  br label %243

243:                                              ; preds = %229, %220, %186
  %244 = phi double [ %133, %186 ], [ %242, %229 ], [ %133, %220 ]
  %245 = phi double [ %134, %186 ], [ %227, %229 ], [ %134, %220 ]
  %246 = phi double [ %135, %186 ], [ %241, %229 ], [ %135, %220 ]
  %247 = phi double [ %136, %186 ], [ %240, %229 ], [ %136, %220 ]
  %248 = phi double [ %137, %186 ], [ %227, %229 ], [ %137, %220 ]
  %249 = getelementptr inbounds nuw i8, ptr %138, i64 72
  br label %250

250:                                              ; preds = %243, %182
  %251 = phi double [ %244, %243 ], [ %133, %182 ]
  %252 = phi double [ %245, %243 ], [ %134, %182 ]
  %253 = phi double [ %246, %243 ], [ %135, %182 ]
  %254 = phi double [ %247, %243 ], [ %136, %182 ]
  %255 = phi double [ %248, %243 ], [ %137, %182 ]
  %256 = phi ptr [ %249, %243 ], [ %185, %182 ]
  %257 = icmp ult ptr %256, %33
  br i1 %257, label %132, label %258, !llvm.loop !51

258:                                              ; preds = %250
  %259 = fcmp oeq double %252, 0x7FF0000000000000
  br i1 %259, label %382, label %260

260:                                              ; preds = %258
  %261 = fmul double %93, %253
  %262 = tail call double @llvm.fmuladd.f64(double %254, double %92, double %261)
  %263 = tail call noundef double @llvm.fmuladd.f64(double %251, double %94, double %262)
  %264 = fcmp ult double %263, 0.000000e+00
  br i1 %264, label %265, label %382

265:                                              ; preds = %260
  %266 = fneg double %263
  %267 = fmul double %129, %252
  %268 = fmul double %130, %252
  %269 = fmul double %131, %252
  %270 = fadd double %267, 0.000000e+00
  %271 = fadd double %268, 0.000000e+00
  %272 = fadd double %269, -4.500000e+00
  %273 = fmul double %254, 0x3D719799812DEA11
  %274 = fmul double %253, 0x3D719799812DEA11
  %275 = fmul double %251, 0x3D719799812DEA11
  %276 = fadd double %270, %273
  %277 = fadd double %271, %274
  %278 = fadd double %275, %272
  br label %279

279:                                              ; preds = %375, %265
  %280 = phi ptr [ %376, %375 ], [ %31, %265 ]
  %281 = load double, ptr %280, align 8, !tbaa !43
  %282 = fsub double %281, %276
  %283 = getelementptr inbounds nuw i8, ptr %280, i64 8
  %284 = load double, ptr %283, align 8, !tbaa !45
  %285 = fsub double %284, %277
  %286 = getelementptr inbounds nuw i8, ptr %280, i64 16
  %287 = load double, ptr %286, align 8, !tbaa !46
  %288 = fsub double %287, %278
  %289 = fmul double %285, %96
  %290 = tail call double @llvm.fmuladd.f64(double %95, double %282, double %289)
  %291 = tail call noundef double @llvm.fmuladd.f64(double %97, double %288, double %290)
  %292 = fmul double %285, %285
  %293 = tail call double @llvm.fmuladd.f64(double %282, double %282, double %292)
  %294 = tail call noundef double @llvm.fmuladd.f64(double %288, double %288, double %293)
  %295 = fneg double %294
  %296 = tail call double @llvm.fmuladd.f64(double %291, double %291, double %295)
  %297 = getelementptr inbounds nuw i8, ptr %280, i64 24
  %298 = load double, ptr %297, align 8, !tbaa !47
  %299 = tail call double @llvm.fmuladd.f64(double %298, double %298, double %296)
  %300 = fcmp olt double %299, 0.000000e+00
  br i1 %300, label %371, label %301

301:                                              ; preds = %279
  %302 = fcmp oeq double %299, 0x7FF0000000000000
  br i1 %302, label %315, label %303

303:                                              ; preds = %301, %303
  %304 = phi i32 [ %312, %303 ], [ 100, %301 ]
  %305 = phi double [ %308, %303 ], [ 1.000000e+00, %301 ]
  %306 = fdiv double %299, %305
  %307 = fadd double %305, %306
  %308 = fmul double %307, 5.000000e-01
  %309 = fsub double %308, %305
  %310 = tail call double @llvm.fabs.f64(double %309)
  %311 = fcmp ule double %310, 0x3D719799812DEA11
  %312 = add nsw i32 %304, -1
  %313 = icmp eq i32 %312, 0
  %314 = select i1 %311, i1 true, i1 %313
  br i1 %314, label %315, label %303, !llvm.loop !13

315:                                              ; preds = %303, %301
  %316 = phi double [ 0x7FF0000000000000, %301 ], [ %308, %303 ]
  %317 = fadd double %291, %316
  %318 = fcmp uge double %317, 0.000000e+00
  %319 = fsub double %291, %316
  %320 = fcmp ogt double %319, 0.000000e+00
  %321 = select i1 %320, double %319, double %317
  %322 = fcmp une double %321, 0x7FF0000000000000
  %323 = and i1 %318, %322
  br i1 %323, label %324, label %371

324:                                              ; preds = %315
  %325 = getelementptr inbounds nuw i8, ptr %280, i64 32
  %326 = load double, ptr %325, align 8, !tbaa !43
  %327 = fsub double %326, %276
  %328 = getelementptr inbounds nuw i8, ptr %280, i64 40
  %329 = load double, ptr %328, align 8, !tbaa !45
  %330 = fsub double %329, %277
  %331 = getelementptr inbounds nuw i8, ptr %280, i64 48
  %332 = load double, ptr %331, align 8, !tbaa !46
  %333 = fsub double %332, %278
  %334 = fmul double %330, %96
  %335 = tail call double @llvm.fmuladd.f64(double %95, double %327, double %334)
  %336 = tail call noundef double @llvm.fmuladd.f64(double %97, double %333, double %335)
  %337 = fmul double %330, %330
  %338 = tail call double @llvm.fmuladd.f64(double %327, double %327, double %337)
  %339 = tail call noundef double @llvm.fmuladd.f64(double %333, double %333, double %338)
  %340 = fneg double %339
  %341 = tail call double @llvm.fmuladd.f64(double %336, double %336, double %340)
  %342 = getelementptr inbounds nuw i8, ptr %280, i64 56
  %343 = load double, ptr %342, align 8, !tbaa !47
  %344 = tail call double @llvm.fmuladd.f64(double %343, double %343, double %341)
  %345 = fcmp olt double %344, 0.000000e+00
  br i1 %345, label %369, label %346

346:                                              ; preds = %324
  %347 = fcmp oeq double %344, 0x7FF0000000000000
  br i1 %347, label %360, label %348

348:                                              ; preds = %346, %348
  %349 = phi i32 [ %357, %348 ], [ 100, %346 ]
  %350 = phi double [ %353, %348 ], [ 1.000000e+00, %346 ]
  %351 = fdiv double %344, %350
  %352 = fadd double %350, %351
  %353 = fmul double %352, 5.000000e-01
  %354 = fsub double %353, %350
  %355 = tail call double @llvm.fabs.f64(double %354)
  %356 = fcmp ule double %355, 0x3D719799812DEA11
  %357 = add nsw i32 %349, -1
  %358 = icmp eq i32 %357, 0
  %359 = select i1 %356, i1 true, i1 %358
  br i1 %359, label %360, label %348, !llvm.loop !13

360:                                              ; preds = %348, %346
  %361 = phi double [ 0x7FF0000000000000, %346 ], [ %353, %348 ]
  %362 = fadd double %336, %361
  %363 = fcmp uge double %362, 0.000000e+00
  %364 = fsub double %336, %361
  %365 = fcmp ogt double %364, 0.000000e+00
  %366 = select i1 %365, double %364, double %362
  %367 = fcmp one double %366, 0x7FF0000000000000
  %368 = and i1 %363, %367
  br i1 %368, label %378, label %369

369:                                              ; preds = %360, %324
  %370 = getelementptr inbounds nuw i8, ptr %280, i64 72
  br label %375

371:                                              ; preds = %315, %279
  %372 = getelementptr inbounds nuw i8, ptr %280, i64 64
  %373 = load i64, ptr %372, align 8, !tbaa !49
  %374 = getelementptr inbounds %struct.node_t, ptr %280, i64 %373
  br label %375

375:                                              ; preds = %371, %369
  %376 = phi ptr [ %374, %371 ], [ %370, %369 ]
  %377 = icmp ult ptr %376, %33
  br i1 %377, label %279, label %378, !llvm.loop !52

378:                                              ; preds = %375, %360
  %379 = phi double [ 0x7FF0000000000000, %375 ], [ %366, %360 ]
  %380 = fcmp oeq double %379, 0x7FF0000000000000
  %381 = select i1 %380, double %266, double 0.000000e+00
  br label %382

382:                                              ; preds = %378, %260, %258
  %383 = phi double [ %381, %378 ], [ 0.000000e+00, %260 ], [ 0.000000e+00, %258 ]
  %384 = fadd double %100, %383
  %385 = add nuw nsw i64 %99, 1
  %386 = icmp eq i64 %385, 4
  br i1 %386, label %391, label %98, !llvm.loop !53

387:                                              ; preds = %394
  %388 = fadd double %67, -1.000000e+00
  %389 = add nsw i32 %66, -1
  %390 = icmp eq i32 %389, 0
  br i1 %390, label %68, label %65, !llvm.loop !54

391:                                              ; preds = %382
  %392 = fmul double %384, 6.400000e+01
  %393 = fptosi double %392 to i32
  br label %394

394:                                              ; preds = %391, %89
  %395 = phi i32 [ %393, %391 ], [ 0, %89 ]
  %396 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEi(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, i32 noundef %395)
  %397 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %396, ptr noundef nonnull @.str.1, i64 noundef 1)
  %398 = fadd double %91, 1.000000e+00
  %399 = add nsw i32 %90, -1
  %400 = icmp eq i32 %399, 0
  br i1 %400, label %387, label %89, !llvm.loop !55

401:                                              ; preds = %81, %84
  %402 = phi i8 [ %83, %81 ], [ %88, %84 ]
  %403 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, i8 noundef %402)
  %404 = tail call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo5flushEv(ptr noundef nonnull align 8 dereferenceable(8) %403)
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #10
  ret i32 0
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64) #2

; Function Attrs: nobuiltin allocsize(0)
declare noundef nonnull ptr @_Znam(i64 noundef) local_unnamed_addr #3

declare i32 @__gxx_personality_v0(...)

; Function Attrs: mustprogress nofree nosync nounwind memory(write, inaccessiblemem: none) uwtable
define internal fastcc noundef nonnull ptr @_ZL6createP6node_tii3v_tS1_d(ptr noundef initializes((0, 72)) %0, i32 noundef %1, i32 noundef %2, [3 x double] alignstack(8) %3, [3 x double] alignstack(8) %4, double noundef %5) unnamed_addr #4 {
  %7 = extractvalue [3 x double] %3, 0
  %8 = extractvalue [3 x double] %3, 1
  %9 = extractvalue [3 x double] %3, 2
  %10 = extractvalue [3 x double] %4, 0
  %11 = extractvalue [3 x double] %4, 1
  %12 = extractvalue [3 x double] %4, 2
  %13 = fmul double %5, 2.000000e+00
  %14 = icmp sgt i32 %1, 1
  %15 = select i1 %14, i32 %2, i32 1
  %16 = sext i32 %15 to i64
  store double %7, ptr %0, align 8, !tbaa !14
  %17 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store double %8, ptr %17, align 8, !tbaa !14
  %18 = getelementptr inbounds nuw i8, ptr %0, i64 16
  store double %9, ptr %18, align 8, !tbaa !14
  %19 = getelementptr inbounds nuw i8, ptr %0, i64 24
  store double %13, ptr %19, align 8, !tbaa !14
  %20 = getelementptr inbounds nuw i8, ptr %0, i64 32
  store double %7, ptr %20, align 8, !tbaa !14
  %21 = getelementptr inbounds nuw i8, ptr %0, i64 40
  store double %8, ptr %21, align 8, !tbaa !14
  %22 = getelementptr inbounds nuw i8, ptr %0, i64 48
  store double %9, ptr %22, align 8, !tbaa !14
  %23 = getelementptr inbounds nuw i8, ptr %0, i64 56
  store double %5, ptr %23, align 8, !tbaa !14
  %24 = getelementptr inbounds nuw i8, ptr %0, i64 64
  store i64 %16, ptr %24, align 8, !tbaa !49
  %25 = getelementptr inbounds nuw i8, ptr %0, i64 72
  %26 = icmp slt i32 %1, 2
  br i1 %26, label %382, label %27

27:                                               ; preds = %6
  %28 = add nsw i32 %2, -9
  %29 = sdiv i32 %28, 9
  %30 = tail call i32 @llvm.smax.i32(i32 %29, i32 1)
  %31 = fmul double %11, %11
  %32 = tail call double @llvm.fmuladd.f64(double %10, double %10, double %31)
  %33 = tail call noundef double @llvm.fmuladd.f64(double %12, double %12, double %32)
  %34 = fcmp oeq double %33, 0x7FF0000000000000
  br i1 %34, label %47, label %35

35:                                               ; preds = %27, %35
  %36 = phi i32 [ %44, %35 ], [ 100, %27 ]
  %37 = phi double [ %40, %35 ], [ 1.000000e+00, %27 ]
  %38 = fdiv double %33, %37
  %39 = fadd double %37, %38
  %40 = fmul double %39, 5.000000e-01
  %41 = fsub double %40, %37
  %42 = tail call double @llvm.fabs.f64(double %41)
  %43 = fcmp ule double %42, 0x3D719799812DEA11
  %44 = add nsw i32 %36, -1
  %45 = icmp eq i32 %44, 0
  %46 = select i1 %43, i1 true, i1 %45
  br i1 %46, label %47, label %35, !llvm.loop !13

47:                                               ; preds = %35, %27
  %48 = phi double [ 0x7FF0000000000000, %27 ], [ %40, %35 ]
  %49 = fdiv double 1.000000e+00, %48
  %50 = fmul double %10, %49
  %51 = fmul double %11, %49
  %52 = fmul double %12, %49
  %53 = fmul double %50, %50
  %54 = fcmp une double %53, 1.000000e+00
  %55 = fmul double %51, %51
  %56 = fcmp une double %55, 1.000000e+00
  %57 = and i1 %54, %56
  %58 = fmul double %52, %52
  %59 = fcmp une double %58, 1.000000e+00
  %60 = and i1 %59, %57
  br i1 %60, label %61, label %75

61:                                               ; preds = %47
  %62 = fcmp ogt double %55, %53
  br i1 %62, label %63, label %69

63:                                               ; preds = %61
  %64 = fcmp ogt double %55, %58
  br i1 %64, label %65, label %67

65:                                               ; preds = %63
  %66 = fneg double %51
  br label %75

67:                                               ; preds = %63
  %68 = fneg double %52
  br label %75

69:                                               ; preds = %61
  %70 = fcmp ogt double %58, %53
  br i1 %70, label %71, label %73

71:                                               ; preds = %69
  %72 = fneg double %52
  br label %75

73:                                               ; preds = %69
  %74 = fneg double %50
  br label %75

75:                                               ; preds = %47, %65, %67, %71, %73
  %76 = phi double [ %50, %67 ], [ %50, %65 ], [ %74, %73 ], [ %50, %71 ], [ %52, %47 ]
  %77 = phi double [ %51, %67 ], [ %66, %65 ], [ %51, %73 ], [ %51, %71 ], [ %50, %47 ]
  %78 = phi double [ %68, %67 ], [ %52, %65 ], [ %52, %73 ], [ %72, %71 ], [ %51, %47 ]
  %79 = fneg double %77
  %80 = fmul double %52, %79
  %81 = tail call double @llvm.fmuladd.f64(double %51, double %78, double %80)
  %82 = fneg double %78
  %83 = fmul double %50, %82
  %84 = tail call double @llvm.fmuladd.f64(double %52, double %76, double %83)
  %85 = fneg double %76
  %86 = fmul double %51, %85
  %87 = tail call double @llvm.fmuladd.f64(double %50, double %77, double %86)
  %88 = fneg double %84
  %89 = fmul double %52, %88
  %90 = tail call double @llvm.fmuladd.f64(double %51, double %87, double %89)
  %91 = fneg double %87
  %92 = fmul double %50, %91
  %93 = tail call double @llvm.fmuladd.f64(double %52, double %81, double %92)
  %94 = fneg double %81
  %95 = fmul double %51, %94
  %96 = tail call double @llvm.fmuladd.f64(double %50, double %84, double %95)
  %97 = fdiv double %5, 3.000000e+00
  %98 = fmul double %10, 2.000000e-01
  %99 = fmul double %11, 2.000000e-01
  %100 = fmul double %12, 2.000000e-01
  %101 = add nsw i32 %1, -1
  %102 = fadd double %5, %97
  br label %108

103:                                              ; preds = %223
  %104 = fadd double %242, 0xBFD657184AE74487
  %105 = fmul double %10, 6.000000e-01
  %106 = fmul double %11, 6.000000e-01
  %107 = fmul double %12, 6.000000e-01
  br label %245

108:                                              ; preds = %75, %223
  %109 = phi ptr [ %25, %75 ], [ %241, %223 ]
  %110 = phi double [ 0.000000e+00, %75 ], [ %242, %223 ]
  %111 = phi i32 [ 0, %75 ], [ %243, %223 ]
  %112 = fcmp olt double %110, 0.000000e+00
  br i1 %112, label %116, label %113

113:                                              ; preds = %116, %108
  %114 = phi double [ %110, %108 ], [ %118, %116 ]
  %115 = fcmp ogt double %114, 0x401921FB54411744
  br i1 %115, label %120, label %124

116:                                              ; preds = %108, %116
  %117 = phi double [ %118, %116 ], [ %110, %108 ]
  %118 = fadd double %117, 0x401921FB54411744
  %119 = fcmp olt double %118, 0.000000e+00
  br i1 %119, label %116, label %113, !llvm.loop !56

120:                                              ; preds = %113, %120
  %121 = phi double [ %122, %120 ], [ %114, %113 ]
  %122 = fadd double %121, 0xC01921FB54411744
  %123 = fcmp ogt double %122, 0x401921FB54411744
  br i1 %123, label %120, label %124, !llvm.loop !57

124:                                              ; preds = %120, %113
  %125 = phi double [ %114, %113 ], [ %122, %120 ]
  %126 = fcmp ogt double %125, 0x4012D97C7F713E20
  br i1 %126, label %127, label %129

127:                                              ; preds = %124
  %128 = fsub double 0x401921FB54411744, %125
  br label %137

129:                                              ; preds = %124
  %130 = fcmp ogt double %125, 0x400921FB5496FD7F
  br i1 %130, label %131, label %133

131:                                              ; preds = %129
  %132 = fadd double %125, 0xC00921FB5496FD7F
  br label %137

133:                                              ; preds = %129
  %134 = fcmp ogt double %125, 0x3FF921FB54524550
  br i1 %134, label %135, label %137

135:                                              ; preds = %133
  %136 = fsub double 0x400921FB5496FD7F, %125
  br label %137

137:                                              ; preds = %127, %131, %133, %135
  %138 = phi double [ -1.000000e+00, %127 ], [ -1.000000e+00, %131 ], [ 1.000000e+00, %135 ], [ 1.000000e+00, %133 ]
  %139 = phi double [ %128, %127 ], [ %132, %131 ], [ %136, %135 ], [ %125, %133 ]
  %140 = fmul double %139, %139
  %141 = fmul double %139, %140
  %142 = fmul double %139, %141
  %143 = fmul double %139, %142
  %144 = fdiv double %141, 6.000000e+00
  %145 = fdiv double %143, 1.200000e+02
  %146 = fsub double %139, %144
  %147 = fadd double %146, %145
  %148 = fmul double %138, %147
  %149 = fcmp ogt double %148, 1.000000e+00
  %150 = select i1 %149, double 1.000000e+00, double %148
  %151 = fcmp olt double %150, -1.000000e+00
  %152 = select i1 %151, double -1.000000e+00, double %150
  %153 = fmul double %90, %152
  %154 = fmul double %93, %152
  %155 = fmul double %96, %152
  %156 = fsub double %153, %98
  %157 = fsub double %154, %99
  %158 = fsub double %155, %100
  %159 = fadd double %110, 0x3FF921FB54524550
  %160 = fcmp olt double %159, 0.000000e+00
  br i1 %160, label %164, label %161

161:                                              ; preds = %164, %137
  %162 = phi double [ %159, %137 ], [ %166, %164 ]
  %163 = fcmp ogt double %162, 0x401921FB54411744
  br i1 %163, label %168, label %172

164:                                              ; preds = %137, %164
  %165 = phi double [ %166, %164 ], [ %159, %137 ]
  %166 = fadd double %165, 0x401921FB54411744
  %167 = fcmp olt double %166, 0.000000e+00
  br i1 %167, label %164, label %161, !llvm.loop !56

168:                                              ; preds = %161, %168
  %169 = phi double [ %170, %168 ], [ %162, %161 ]
  %170 = fadd double %169, 0xC01921FB54411744
  %171 = fcmp ogt double %170, 0x401921FB54411744
  br i1 %171, label %168, label %172, !llvm.loop !57

172:                                              ; preds = %168, %161
  %173 = phi double [ %162, %161 ], [ %170, %168 ]
  %174 = fcmp ogt double %173, 0x4012D97C7F713E20
  br i1 %174, label %175, label %177

175:                                              ; preds = %172
  %176 = fsub double 0x401921FB54411744, %173
  br label %185

177:                                              ; preds = %172
  %178 = fcmp ogt double %173, 0x400921FB5496FD7F
  br i1 %178, label %179, label %181

179:                                              ; preds = %177
  %180 = fadd double %173, 0xC00921FB5496FD7F
  br label %185

181:                                              ; preds = %177
  %182 = fcmp ogt double %173, 0x3FF921FB54524550
  br i1 %182, label %183, label %185

183:                                              ; preds = %181
  %184 = fsub double 0x400921FB5496FD7F, %173
  br label %185

185:                                              ; preds = %175, %179, %181, %183
  %186 = phi double [ -1.000000e+00, %175 ], [ -1.000000e+00, %179 ], [ 1.000000e+00, %183 ], [ 1.000000e+00, %181 ]
  %187 = phi double [ %176, %175 ], [ %180, %179 ], [ %184, %183 ], [ %173, %181 ]
  %188 = fmul double %187, %187
  %189 = fmul double %187, %188
  %190 = fmul double %187, %189
  %191 = fmul double %187, %190
  %192 = fdiv double %189, 6.000000e+00
  %193 = fdiv double %191, 1.200000e+02
  %194 = fsub double %187, %192
  %195 = fadd double %194, %193
  %196 = fmul double %186, %195
  %197 = fcmp ogt double %196, 1.000000e+00
  %198 = select i1 %197, double 1.000000e+00, double %196
  %199 = fcmp olt double %198, -1.000000e+00
  %200 = select i1 %199, double -1.000000e+00, double %198
  %201 = fmul double %81, %200
  %202 = fmul double %84, %200
  %203 = fmul double %87, %200
  %204 = fadd double %156, %201
  %205 = fadd double %157, %202
  %206 = fadd double %158, %203
  %207 = fmul double %205, %205
  %208 = tail call double @llvm.fmuladd.f64(double %204, double %204, double %207)
  %209 = tail call noundef double @llvm.fmuladd.f64(double %206, double %206, double %208)
  %210 = fcmp oeq double %209, 0x7FF0000000000000
  br i1 %210, label %223, label %211

211:                                              ; preds = %185, %211
  %212 = phi i32 [ %220, %211 ], [ 100, %185 ]
  %213 = phi double [ %216, %211 ], [ 1.000000e+00, %185 ]
  %214 = fdiv double %209, %213
  %215 = fadd double %213, %214
  %216 = fmul double %215, 5.000000e-01
  %217 = fsub double %216, %213
  %218 = tail call double @llvm.fabs.f64(double %217)
  %219 = fcmp ule double %218, 0x3D719799812DEA11
  %220 = add nsw i32 %212, -1
  %221 = icmp eq i32 %220, 0
  %222 = select i1 %219, i1 true, i1 %221
  br i1 %222, label %223, label %211, !llvm.loop !13

223:                                              ; preds = %211, %185
  %224 = phi double [ 0x7FF0000000000000, %185 ], [ %216, %211 ]
  %225 = fdiv double 1.000000e+00, %224
  %226 = fmul double %204, %225
  %227 = fmul double %205, %225
  %228 = fmul double %206, %225
  %229 = fmul double %102, %226
  %230 = fmul double %102, %227
  %231 = fmul double %102, %228
  %232 = fadd double %7, %229
  %233 = fadd double %8, %230
  %234 = fadd double %9, %231
  %235 = insertvalue [3 x double] poison, double %232, 0
  %236 = insertvalue [3 x double] %235, double %233, 1
  %237 = insertvalue [3 x double] %236, double %234, 2
  %238 = insertvalue [3 x double] poison, double %226, 0
  %239 = insertvalue [3 x double] %238, double %227, 1
  %240 = insertvalue [3 x double] %239, double %228, 2
  %241 = tail call fastcc noundef ptr @_ZL6createP6node_tii3v_tS1_d(ptr noundef nonnull %109, i32 noundef %101, i32 noundef %30, [3 x double] alignstack(8) %237, [3 x double] alignstack(8) %240, double noundef %97)
  %242 = fadd double %110, 0x3FF0C152382D7365
  %243 = add nuw nsw i32 %111, 1
  %244 = icmp eq i32 %243, 6
  br i1 %244, label %103, label %108, !llvm.loop !58

245:                                              ; preds = %103, %360
  %246 = phi ptr [ %241, %103 ], [ %378, %360 ]
  %247 = phi double [ %104, %103 ], [ %379, %360 ]
  %248 = phi i32 [ 0, %103 ], [ %380, %360 ]
  %249 = fcmp olt double %247, 0.000000e+00
  br i1 %249, label %253, label %250

250:                                              ; preds = %253, %245
  %251 = phi double [ %247, %245 ], [ %255, %253 ]
  %252 = fcmp ogt double %251, 0x401921FB54411744
  br i1 %252, label %257, label %261

253:                                              ; preds = %245, %253
  %254 = phi double [ %255, %253 ], [ %247, %245 ]
  %255 = fadd double %254, 0x401921FB54411744
  %256 = fcmp olt double %255, 0.000000e+00
  br i1 %256, label %253, label %250, !llvm.loop !56

257:                                              ; preds = %250, %257
  %258 = phi double [ %259, %257 ], [ %251, %250 ]
  %259 = fadd double %258, 0xC01921FB54411744
  %260 = fcmp ogt double %259, 0x401921FB54411744
  br i1 %260, label %257, label %261, !llvm.loop !57

261:                                              ; preds = %257, %250
  %262 = phi double [ %251, %250 ], [ %259, %257 ]
  %263 = fcmp ogt double %262, 0x4012D97C7F713E20
  br i1 %263, label %264, label %266

264:                                              ; preds = %261
  %265 = fsub double 0x401921FB54411744, %262
  br label %274

266:                                              ; preds = %261
  %267 = fcmp ogt double %262, 0x400921FB5496FD7F
  br i1 %267, label %268, label %270

268:                                              ; preds = %266
  %269 = fadd double %262, 0xC00921FB5496FD7F
  br label %274

270:                                              ; preds = %266
  %271 = fcmp ogt double %262, 0x3FF921FB54524550
  br i1 %271, label %272, label %274

272:                                              ; preds = %270
  %273 = fsub double 0x400921FB5496FD7F, %262
  br label %274

274:                                              ; preds = %264, %268, %270, %272
  %275 = phi double [ -1.000000e+00, %264 ], [ -1.000000e+00, %268 ], [ 1.000000e+00, %272 ], [ 1.000000e+00, %270 ]
  %276 = phi double [ %265, %264 ], [ %269, %268 ], [ %273, %272 ], [ %262, %270 ]
  %277 = fmul double %276, %276
  %278 = fmul double %276, %277
  %279 = fmul double %276, %278
  %280 = fmul double %276, %279
  %281 = fdiv double %278, 6.000000e+00
  %282 = fdiv double %280, 1.200000e+02
  %283 = fsub double %276, %281
  %284 = fadd double %283, %282
  %285 = fmul double %275, %284
  %286 = fcmp ogt double %285, 1.000000e+00
  %287 = select i1 %286, double 1.000000e+00, double %285
  %288 = fcmp olt double %287, -1.000000e+00
  %289 = select i1 %288, double -1.000000e+00, double %287
  %290 = fmul double %90, %289
  %291 = fmul double %93, %289
  %292 = fmul double %96, %289
  %293 = fadd double %105, %290
  %294 = fadd double %106, %291
  %295 = fadd double %107, %292
  %296 = fadd double %247, 0x3FF921FB54524550
  %297 = fcmp olt double %296, 0.000000e+00
  br i1 %297, label %301, label %298

298:                                              ; preds = %301, %274
  %299 = phi double [ %296, %274 ], [ %303, %301 ]
  %300 = fcmp ogt double %299, 0x401921FB54411744
  br i1 %300, label %305, label %309

301:                                              ; preds = %274, %301
  %302 = phi double [ %303, %301 ], [ %296, %274 ]
  %303 = fadd double %302, 0x401921FB54411744
  %304 = fcmp olt double %303, 0.000000e+00
  br i1 %304, label %301, label %298, !llvm.loop !56

305:                                              ; preds = %298, %305
  %306 = phi double [ %307, %305 ], [ %299, %298 ]
  %307 = fadd double %306, 0xC01921FB54411744
  %308 = fcmp ogt double %307, 0x401921FB54411744
  br i1 %308, label %305, label %309, !llvm.loop !57

309:                                              ; preds = %305, %298
  %310 = phi double [ %299, %298 ], [ %307, %305 ]
  %311 = fcmp ogt double %310, 0x4012D97C7F713E20
  br i1 %311, label %312, label %314

312:                                              ; preds = %309
  %313 = fsub double 0x401921FB54411744, %310
  br label %322

314:                                              ; preds = %309
  %315 = fcmp ogt double %310, 0x400921FB5496FD7F
  br i1 %315, label %316, label %318

316:                                              ; preds = %314
  %317 = fadd double %310, 0xC00921FB5496FD7F
  br label %322

318:                                              ; preds = %314
  %319 = fcmp ogt double %310, 0x3FF921FB54524550
  br i1 %319, label %320, label %322

320:                                              ; preds = %318
  %321 = fsub double 0x400921FB5496FD7F, %310
  br label %322

322:                                              ; preds = %312, %316, %318, %320
  %323 = phi double [ -1.000000e+00, %312 ], [ -1.000000e+00, %316 ], [ 1.000000e+00, %320 ], [ 1.000000e+00, %318 ]
  %324 = phi double [ %313, %312 ], [ %317, %316 ], [ %321, %320 ], [ %310, %318 ]
  %325 = fmul double %324, %324
  %326 = fmul double %324, %325
  %327 = fmul double %324, %326
  %328 = fmul double %324, %327
  %329 = fdiv double %326, 6.000000e+00
  %330 = fdiv double %328, 1.200000e+02
  %331 = fsub double %324, %329
  %332 = fadd double %331, %330
  %333 = fmul double %323, %332
  %334 = fcmp ogt double %333, 1.000000e+00
  %335 = select i1 %334, double 1.000000e+00, double %333
  %336 = fcmp olt double %335, -1.000000e+00
  %337 = select i1 %336, double -1.000000e+00, double %335
  %338 = fmul double %81, %337
  %339 = fmul double %84, %337
  %340 = fmul double %87, %337
  %341 = fadd double %293, %338
  %342 = fadd double %294, %339
  %343 = fadd double %295, %340
  %344 = fmul double %342, %342
  %345 = tail call double @llvm.fmuladd.f64(double %341, double %341, double %344)
  %346 = tail call noundef double @llvm.fmuladd.f64(double %343, double %343, double %345)
  %347 = fcmp oeq double %346, 0x7FF0000000000000
  br i1 %347, label %360, label %348

348:                                              ; preds = %322, %348
  %349 = phi i32 [ %357, %348 ], [ 100, %322 ]
  %350 = phi double [ %353, %348 ], [ 1.000000e+00, %322 ]
  %351 = fdiv double %346, %350
  %352 = fadd double %350, %351
  %353 = fmul double %352, 5.000000e-01
  %354 = fsub double %353, %350
  %355 = tail call double @llvm.fabs.f64(double %354)
  %356 = fcmp ule double %355, 0x3D719799812DEA11
  %357 = add nsw i32 %349, -1
  %358 = icmp eq i32 %357, 0
  %359 = select i1 %356, i1 true, i1 %358
  br i1 %359, label %360, label %348, !llvm.loop !13

360:                                              ; preds = %348, %322
  %361 = phi double [ 0x7FF0000000000000, %322 ], [ %353, %348 ]
  %362 = fdiv double 1.000000e+00, %361
  %363 = fmul double %341, %362
  %364 = fmul double %342, %362
  %365 = fmul double %343, %362
  %366 = fmul double %102, %363
  %367 = fmul double %102, %364
  %368 = fmul double %102, %365
  %369 = fadd double %7, %366
  %370 = fadd double %8, %367
  %371 = fadd double %9, %368
  %372 = insertvalue [3 x double] poison, double %369, 0
  %373 = insertvalue [3 x double] %372, double %370, 1
  %374 = insertvalue [3 x double] %373, double %371, 2
  %375 = insertvalue [3 x double] poison, double %363, 0
  %376 = insertvalue [3 x double] %375, double %364, 1
  %377 = insertvalue [3 x double] %376, double %365, 2
  %378 = tail call fastcc noundef ptr @_ZL6createP6node_tii3v_tS1_d(ptr noundef nonnull %246, i32 noundef %101, i32 noundef %30, [3 x double] alignstack(8) %374, [3 x double] alignstack(8) %377, double noundef %97)
  %379 = fadd double %247, 0x4000C152382D7365
  %380 = add nuw nsw i32 %248, 1
  %381 = icmp eq i32 %380, 3
  br i1 %381, label %382, label %245, !llvm.loop !59

382:                                              ; preds = %360, %6
  %383 = phi ptr [ %25, %6 ], [ %378, %360 ]
  ret ptr %383
}

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEi(ptr noundef nonnull align 8 dereferenceable(8), i32 noundef) local_unnamed_addr #5

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fmuladd.f64(double, double, double) #2

; Function Attrs: nounwind
declare i64 @__isoc23_strtol(ptr noundef, ptr noundef, i32 noundef) local_unnamed_addr #6

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo3putEc(ptr noundef nonnull align 8 dereferenceable(8), i8 noundef) local_unnamed_addr #5

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo5flushEv(ptr noundef nonnull align 8 dereferenceable(8)) local_unnamed_addr #5

; Function Attrs: cold noreturn
declare void @_ZSt16__throw_bad_castv() local_unnamed_addr #7

declare void @_ZNKSt5ctypeIcE13_M_widen_initEv(ptr noundef nonnull align 8 dereferenceable(570)) local_unnamed_addr #5

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8), ptr noundef, i64 noundef) local_unnamed_addr #5

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define internal void @_GLOBAL__sub_I_sphereflake.cpp() #8 section ".text.startup" {
  br label %1

1:                                                ; preds = %1, %0
  %2 = phi i32 [ %10, %1 ], [ 100, %0 ]
  %3 = phi double [ %6, %1 ], [ 1.000000e+00, %0 ]
  %4 = fdiv double 0x3FF7B851EB851EB9, %3
  %5 = fadd double %3, %4
  %6 = fmul double %5, 5.000000e-01
  %7 = fsub double %6, %3
  %8 = tail call double @llvm.fabs.f64(double %7)
  %9 = fcmp ule double %8, 0x3D719799812DEA11
  %10 = add nsw i32 %2, -1
  %11 = icmp eq i32 %10, 0
  %12 = select i1 %9, i1 true, i1 %11
  br i1 %12, label %13, label %1, !llvm.loop !13

13:                                               ; preds = %1
  %14 = fdiv double 1.000000e+00, %6
  %15 = fmul double %14, 9.000000e-01
  %16 = insertelement <2 x double> poison, double %14, i64 0
  %17 = shufflevector <2 x double> %16, <2 x double> poison, <2 x i32> zeroinitializer
  %18 = fmul <2 x double> %17, <double -5.000000e-01, double -6.500000e-01>
  store <2 x double> %18, ptr @_ZL5light, align 16
  store double %15, ptr getelementptr inbounds nuw (i8, ptr @_ZL5light, i64 16), align 16
  %19 = tail call ptr @llvm.invariant.start.p0(i64 24, ptr nonnull @_ZL5light)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fabs.f64(double) #9

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smax.i32(i32, i32) #9

attributes #0 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #1 = { mustprogress norecurse uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nobuiltin allocsize(0) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { mustprogress nofree nosync nounwind memory(write, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { cold noreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #8 = { nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #9 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #10 = { nounwind }
attributes #11 = { builtin allocsize(0) }
attributes #12 = { cold noreturn }

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
!11 = distinct !{!11, !12}
!12 = !{!"llvm.loop.mustprogress"}
!13 = distinct !{!13, !12}
!14 = !{!15, !15, i64 0}
!15 = !{!"double", !9, i64 0}
!16 = !{!17, !17, i64 0}
!17 = !{!"vtable pointer", !10, i64 0}
!18 = !{!19, !33, i64 240}
!19 = !{!"_ZTSSt9basic_iosIcSt11char_traitsIcEE", !20, i64 0, !30, i64 216, !9, i64 224, !31, i64 225, !32, i64 232, !33, i64 240, !34, i64 248, !35, i64 256}
!20 = !{!"_ZTSSt8ios_base", !21, i64 8, !21, i64 16, !22, i64 24, !23, i64 28, !23, i64 32, !24, i64 40, !25, i64 48, !9, i64 64, !26, i64 192, !27, i64 200, !28, i64 208}
!21 = !{!"long", !9, i64 0}
!22 = !{!"_ZTSSt13_Ios_Fmtflags", !9, i64 0}
!23 = !{!"_ZTSSt12_Ios_Iostate", !9, i64 0}
!24 = !{!"p1 _ZTSNSt8ios_base14_Callback_listE", !8, i64 0}
!25 = !{!"_ZTSNSt8ios_base6_WordsE", !8, i64 0, !21, i64 8}
!26 = !{!"int", !9, i64 0}
!27 = !{!"p1 _ZTSNSt8ios_base6_WordsE", !8, i64 0}
!28 = !{!"_ZTSSt6locale", !29, i64 0}
!29 = !{!"p1 _ZTSNSt6locale5_ImplE", !8, i64 0}
!30 = !{!"p1 _ZTSSo", !8, i64 0}
!31 = !{!"bool", !9, i64 0}
!32 = !{!"p1 _ZTSSt15basic_streambufIcSt11char_traitsIcEE", !8, i64 0}
!33 = !{!"p1 _ZTSSt5ctypeIcE", !8, i64 0}
!34 = !{!"p1 _ZTSSt7num_putIcSt19ostreambuf_iteratorIcSt11char_traitsIcEEE", !8, i64 0}
!35 = !{!"p1 _ZTSSt7num_getIcSt19istreambuf_iteratorIcSt11char_traitsIcEEE", !8, i64 0}
!36 = !{!37, !9, i64 56}
!37 = !{!"_ZTSSt5ctypeIcE", !38, i64 0, !39, i64 16, !31, i64 24, !40, i64 32, !40, i64 40, !41, i64 48, !9, i64 56, !9, i64 57, !9, i64 313, !9, i64 569}
!38 = !{!"_ZTSNSt6locale5facetE", !26, i64 8}
!39 = !{!"p1 _ZTS15__locale_struct", !8, i64 0}
!40 = !{!"p1 int", !8, i64 0}
!41 = !{!"p1 short", !8, i64 0}
!42 = !{!9, !9, i64 0}
!43 = !{!44, !15, i64 0}
!44 = !{!"_ZTS3v_t", !15, i64 0, !15, i64 8, !15, i64 16}
!45 = !{!44, !15, i64 8}
!46 = !{!44, !15, i64 16}
!47 = !{!48, !15, i64 24}
!48 = !{!"_ZTS8sphere_t", !44, i64 0, !15, i64 24}
!49 = !{!50, !21, i64 64}
!50 = !{!"_ZTS6node_t", !48, i64 0, !48, i64 32, !21, i64 64}
!51 = distinct !{!51, !12}
!52 = distinct !{!52, !12}
!53 = distinct !{!53, !12}
!54 = distinct !{!54, !12}
!55 = distinct !{!55, !12}
!56 = distinct !{!56, !12}
!57 = distinct !{!57, !12}
!58 = distinct !{!58, !12}
!59 = distinct !{!59, !12}
