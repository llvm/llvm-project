; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Misc/whetstone.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Misc/whetstone.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@stderr = external local_unnamed_addr global ptr, align 8
@.str.1 = private unnamed_addr constant [28 x i8] c"usage: whetdc [-c] [loops]\0A\00", align 1
@T = dso_local local_unnamed_addr global double 0.000000e+00, align 8
@T1 = dso_local local_unnamed_addr global double 0.000000e+00, align 8
@T2 = dso_local local_unnamed_addr global double 0.000000e+00, align 8
@E1 = dso_local local_unnamed_addr global [5 x double] zeroinitializer, align 8
@J = dso_local local_unnamed_addr global i32 0, align 4
@K = dso_local local_unnamed_addr global i32 0, align 4
@L = dso_local local_unnamed_addr global i32 0, align 4
@.str.3 = private unnamed_addr constant [44 x i8] c"%7ld %7ld %7ld %12.4e %12.4e %12.4e %12.4e\0A\00", align 1

; Function Attrs: nounwind uwtable
define dso_local range(i32 0, 2) i32 @main(i32 noundef %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = icmp sgt i32 %0, 1
  br i1 %3, label %4, label %8

4:                                                ; preds = %2
  %5 = zext nneg i32 %0 to i64
  br label %29

6:                                                ; preds = %46
  %7 = icmp eq i32 %48, 0
  br label %8

8:                                                ; preds = %6, %2
  %9 = phi i64 [ 100000, %2 ], [ %47, %6 ]
  %10 = phi i1 [ true, %2 ], [ %7, %6 ]
  %11 = mul i64 %9, 12
  %12 = mul i64 %9, 14
  %13 = mul i64 %9, 345
  %14 = mul i64 %9, 210
  %15 = shl i64 %9, 5
  %16 = mul nsw i64 %9, 899
  %17 = mul i64 %9, 616
  %18 = mul i64 %9, 93
  %19 = tail call i64 @llvm.smax.i64(i64 %11, i64 1)
  %20 = tail call i64 @llvm.smax.i64(i64 %12, i64 1)
  %21 = tail call i64 @llvm.smax.i64(i64 %13, i64 1)
  %22 = tail call i64 @llvm.smax.i64(i64 %15, i64 1)
  %23 = tail call i64 @llvm.smax.i64(i64 %17, i64 1)
  %24 = tail call i64 @llvm.smax.i64(i64 %18, i64 1)
  %25 = icmp slt i64 %13, 8
  %26 = and i64 %21, 9223372036854775800
  %27 = or disjoint i64 %26, 1
  %28 = icmp eq i64 %21, %26
  br label %51

29:                                               ; preds = %4, %46
  %30 = phi i64 [ 1, %4 ], [ %49, %46 ]
  %31 = phi i32 [ 0, %4 ], [ %48, %46 ]
  %32 = phi i64 [ 100000, %4 ], [ %47, %46 ]
  %33 = getelementptr inbounds nuw ptr, ptr %1, i64 %30
  %34 = load ptr, ptr %33, align 8, !tbaa !6
  %35 = load i8, ptr %34, align 1
  switch i8 %35, label %40 [
    i8 45, label %36
    i8 99, label %46
  ]

36:                                               ; preds = %29
  %37 = getelementptr inbounds nuw i8, ptr %34, i64 1
  %38 = load i8, ptr %37, align 1
  %39 = icmp eq i8 %38, 99
  br i1 %39, label %46, label %40

40:                                               ; preds = %29, %36
  %41 = tail call i64 @strtol(ptr noundef nonnull captures(none) %34, ptr noundef null, i32 noundef 10) #11
  %42 = icmp sgt i64 %41, 0
  br i1 %42, label %46, label %43

43:                                               ; preds = %40
  %44 = load ptr, ptr @stderr, align 8, !tbaa !11
  %45 = tail call i64 @fwrite(ptr nonnull @.str.1, i64 27, i64 1, ptr %44) #12
  br label %301

46:                                               ; preds = %29, %40, %36
  %47 = phi i64 [ %32, %36 ], [ %41, %40 ], [ %32, %29 ]
  %48 = phi i32 [ 1, %36 ], [ %31, %40 ], [ 1, %29 ]
  %49 = add nuw nsw i64 %30, 1
  %50 = icmp eq i64 %49, %5
  br i1 %50, label %6, label %29, !llvm.loop !13

51:                                               ; preds = %293, %8
  %52 = tail call i64 @time(ptr noundef null) #11
  store double 4.999750e-01, ptr @T, align 8, !tbaa !15
  store double 5.002500e-01, ptr @T1, align 8, !tbaa !15
  store double 2.000000e+00, ptr @T2, align 8, !tbaa !15
  %53 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i64 noundef 0, i64 noundef 0, i64 noundef 0, double noundef 1.000000e+00, double noundef -1.000000e+00, double noundef -1.000000e+00, double noundef -1.000000e+00)
  %54 = load double, ptr @T, align 8, !tbaa !15
  br label %55

55:                                               ; preds = %51, %55
  %56 = phi i64 [ 1, %51 ], [ %77, %55 ]
  %57 = phi double [ 1.000000e+00, %51 ], [ %64, %55 ]
  %58 = phi double [ -1.000000e+00, %51 ], [ %68, %55 ]
  %59 = phi double [ -1.000000e+00, %51 ], [ %72, %55 ]
  %60 = phi double [ -1.000000e+00, %51 ], [ %76, %55 ]
  %61 = fadd double %57, %58
  %62 = fadd double %61, %59
  %63 = fsub double %62, %60
  %64 = fmul double %63, %54
  %65 = fadd double %58, %64
  %66 = fsub double %65, %59
  %67 = fadd double %60, %66
  %68 = fmul double %54, %67
  %69 = fsub double %64, %68
  %70 = fadd double %59, %69
  %71 = fadd double %60, %70
  %72 = fmul double %54, %71
  %73 = fsub double %68, %64
  %74 = fadd double %73, %72
  %75 = fadd double %60, %74
  %76 = fmul double %54, %75
  %77 = add nuw nsw i64 %56, 1
  %78 = icmp eq i64 %56, %19
  br i1 %78, label %79, label %55, !llvm.loop !17

79:                                               ; preds = %55
  store double %64, ptr getelementptr inbounds nuw (i8, ptr @E1, i64 8), align 8, !tbaa !15
  store double %68, ptr getelementptr inbounds nuw (i8, ptr @E1, i64 16), align 8, !tbaa !15
  store double %72, ptr getelementptr inbounds nuw (i8, ptr @E1, i64 24), align 8, !tbaa !15
  store double %76, ptr getelementptr inbounds nuw (i8, ptr @E1, i64 32), align 8, !tbaa !15
  %80 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i64 noundef %11, i64 noundef %12, i64 noundef %11, double noundef %64, double noundef %68, double noundef %72, double noundef %76)
  %81 = load double, ptr getelementptr inbounds nuw (i8, ptr @E1, i64 32), align 8
  %82 = load double, ptr getelementptr inbounds nuw (i8, ptr @E1, i64 24), align 8
  %83 = load double, ptr getelementptr inbounds nuw (i8, ptr @E1, i64 16), align 8
  %84 = load double, ptr getelementptr inbounds nuw (i8, ptr @E1, i64 8), align 8
  %85 = load double, ptr @T, align 8, !tbaa !15
  %86 = load double, ptr @T2, align 8, !tbaa !15
  store i32 6, ptr @J, align 4, !tbaa !18
  br label %87

87:                                               ; preds = %79, %87
  %88 = phi i64 [ 1, %79 ], [ %189, %87 ]
  %89 = phi double [ %84, %79 ], [ %176, %87 ]
  %90 = phi double [ %83, %79 ], [ %180, %87 ]
  %91 = phi double [ %82, %79 ], [ %184, %87 ]
  %92 = phi double [ %81, %79 ], [ %188, %87 ]
  %93 = fadd double %90, %89
  %94 = fadd double %91, %93
  %95 = fsub double %94, %92
  %96 = fmul double %85, %95
  %97 = fadd double %90, %96
  %98 = fsub double %97, %91
  %99 = fadd double %92, %98
  %100 = fmul double %85, %99
  %101 = fsub double %96, %100
  %102 = fadd double %91, %101
  %103 = fadd double %92, %102
  %104 = fmul double %85, %103
  %105 = fsub double %100, %96
  %106 = fadd double %105, %104
  %107 = fadd double %92, %106
  %108 = fdiv double %107, %86
  %109 = fadd double %100, %96
  %110 = fadd double %104, %109
  %111 = fsub double %110, %108
  %112 = fmul double %85, %111
  %113 = fadd double %100, %112
  %114 = fsub double %113, %104
  %115 = fadd double %108, %114
  %116 = fmul double %85, %115
  %117 = fsub double %112, %116
  %118 = fadd double %104, %117
  %119 = fadd double %108, %118
  %120 = fmul double %85, %119
  %121 = fsub double %116, %112
  %122 = fadd double %121, %120
  %123 = fadd double %108, %122
  %124 = fdiv double %123, %86
  %125 = fadd double %116, %112
  %126 = fadd double %120, %125
  %127 = fsub double %126, %124
  %128 = fmul double %85, %127
  %129 = fadd double %116, %128
  %130 = fsub double %129, %120
  %131 = fadd double %124, %130
  %132 = fmul double %85, %131
  %133 = fsub double %128, %132
  %134 = fadd double %120, %133
  %135 = fadd double %124, %134
  %136 = fmul double %85, %135
  %137 = fsub double %132, %128
  %138 = fadd double %137, %136
  %139 = fadd double %124, %138
  %140 = fdiv double %139, %86
  %141 = fadd double %132, %128
  %142 = fadd double %136, %141
  %143 = fsub double %142, %140
  %144 = fmul double %85, %143
  %145 = fadd double %132, %144
  %146 = fsub double %145, %136
  %147 = fadd double %140, %146
  %148 = fmul double %85, %147
  %149 = fsub double %144, %148
  %150 = fadd double %136, %149
  %151 = fadd double %140, %150
  %152 = fmul double %85, %151
  %153 = fsub double %148, %144
  %154 = fadd double %153, %152
  %155 = fadd double %140, %154
  %156 = fdiv double %155, %86
  %157 = fadd double %148, %144
  %158 = fadd double %152, %157
  %159 = fsub double %158, %156
  %160 = fmul double %85, %159
  %161 = fadd double %148, %160
  %162 = fsub double %161, %152
  %163 = fadd double %156, %162
  %164 = fmul double %85, %163
  %165 = fsub double %160, %164
  %166 = fadd double %152, %165
  %167 = fadd double %156, %166
  %168 = fmul double %85, %167
  %169 = fsub double %164, %160
  %170 = fadd double %169, %168
  %171 = fadd double %156, %170
  %172 = fdiv double %171, %86
  %173 = fadd double %164, %160
  %174 = fadd double %168, %173
  %175 = fsub double %174, %172
  %176 = fmul double %85, %175
  %177 = fadd double %164, %176
  %178 = fsub double %177, %168
  %179 = fadd double %172, %178
  %180 = fmul double %85, %179
  %181 = fsub double %176, %180
  %182 = fadd double %168, %181
  %183 = fadd double %172, %182
  %184 = fmul double %85, %183
  %185 = fsub double %180, %176
  %186 = fadd double %185, %184
  %187 = fadd double %172, %186
  %188 = fdiv double %187, %86
  %189 = add nuw nsw i64 %88, 1
  %190 = icmp eq i64 %88, %20
  br i1 %190, label %191, label %87, !llvm.loop !20

191:                                              ; preds = %87
  store double %176, ptr getelementptr inbounds nuw (i8, ptr @E1, i64 8), align 8, !tbaa !15
  store double %180, ptr getelementptr inbounds nuw (i8, ptr @E1, i64 16), align 8, !tbaa !15
  store double %184, ptr getelementptr inbounds nuw (i8, ptr @E1, i64 24), align 8, !tbaa !15
  store double %188, ptr getelementptr inbounds nuw (i8, ptr @E1, i64 32), align 8, !tbaa !15
  %192 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i64 noundef %12, i64 noundef %11, i64 noundef %11, double noundef %176, double noundef %180, double noundef %184, double noundef %188)
  br i1 %25, label %204, label %193

193:                                              ; preds = %191, %193
  %194 = phi i64 [ %199, %193 ], [ 0, %191 ]
  %195 = phi <4 x i32> [ %197, %193 ], [ <i32 1, i32 0, i32 0, i32 0>, %191 ]
  %196 = phi <4 x i32> [ %198, %193 ], [ zeroinitializer, %191 ]
  %197 = xor <4 x i32> %195, splat (i32 1)
  %198 = xor <4 x i32> %196, splat (i32 1)
  %199 = add nuw i64 %194, 8
  %200 = icmp eq i64 %199, %26
  br i1 %200, label %201, label %193, !llvm.loop !21

201:                                              ; preds = %193
  %202 = xor <4 x i32> %196, %195
  %203 = tail call i32 @llvm.vector.reduce.xor.v4i32(<4 x i32> %202)
  br i1 %28, label %213, label %204

204:                                              ; preds = %191, %201
  %205 = phi i64 [ 1, %191 ], [ %27, %201 ]
  %206 = phi i32 [ 1, %191 ], [ %203, %201 ]
  br label %207

207:                                              ; preds = %204, %207
  %208 = phi i64 [ %211, %207 ], [ %205, %204 ]
  %209 = phi i32 [ %210, %207 ], [ %206, %204 ]
  %210 = xor i32 %209, 1
  %211 = add nuw i64 %208, 1
  %212 = icmp eq i64 %208, %21
  br i1 %212, label %213, label %207, !llvm.loop !24

213:                                              ; preds = %207, %201
  %214 = phi i32 [ %203, %201 ], [ %210, %207 ]
  store i32 %214, ptr @J, align 4, !tbaa !18
  %215 = zext nneg i32 %214 to i64
  %216 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i64 noundef %13, i64 noundef %215, i64 noundef %215, double noundef 1.000000e+00, double noundef -1.000000e+00, double noundef -1.000000e+00, double noundef -1.000000e+00)
  store <2 x double> splat (double 6.000000e+00), ptr getelementptr inbounds nuw (i8, ptr @E1, i64 8), align 8, !tbaa !15
  store i32 1, ptr @J, align 4, !tbaa !18
  store i32 2, ptr @K, align 4, !tbaa !18
  store i32 3, ptr @L, align 4, !tbaa !18
  %217 = load double, ptr getelementptr inbounds nuw (i8, ptr @E1, i64 16), align 8, !tbaa !15
  %218 = load double, ptr getelementptr inbounds nuw (i8, ptr @E1, i64 24), align 8, !tbaa !15
  %219 = load double, ptr getelementptr inbounds nuw (i8, ptr @E1, i64 32), align 8, !tbaa !15
  %220 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i64 noundef %14, i64 noundef 1, i64 noundef 2, double noundef 6.000000e+00, double noundef %217, double noundef %218, double noundef %219)
  %221 = load double, ptr @T, align 8, !tbaa !15
  %222 = load double, ptr @T2, align 8, !tbaa !15
  br label %223

223:                                              ; preds = %213, %223
  %224 = phi i64 [ 1, %213 ], [ %253, %223 ]
  %225 = phi double [ 5.000000e-01, %213 ], [ %252, %223 ]
  %226 = phi double [ 5.000000e-01, %213 ], [ %239, %223 ]
  %227 = tail call double @sin(double noundef %226) #11, !tbaa !18
  %228 = fmul double %222, %227
  %229 = tail call double @cos(double noundef %226) #11, !tbaa !18
  %230 = fmul double %228, %229
  %231 = fadd double %226, %225
  %232 = tail call double @cos(double noundef %231) #11, !tbaa !18
  %233 = fsub double %226, %225
  %234 = tail call double @cos(double noundef %233) #11, !tbaa !18
  %235 = fadd double %232, %234
  %236 = fadd double %235, -1.000000e+00
  %237 = fdiv double %230, %236
  %238 = tail call double @atan(double noundef %237) #11, !tbaa !18
  %239 = fmul double %221, %238
  %240 = tail call double @sin(double noundef %225) #11, !tbaa !18
  %241 = fmul double %222, %240
  %242 = tail call double @cos(double noundef %225) #11, !tbaa !18
  %243 = fmul double %241, %242
  %244 = fadd double %225, %239
  %245 = tail call double @cos(double noundef %244) #11, !tbaa !18
  %246 = fsub double %239, %225
  %247 = tail call double @cos(double noundef %246) #11, !tbaa !18
  %248 = fadd double %245, %247
  %249 = fadd double %248, -1.000000e+00
  %250 = fdiv double %243, %249
  %251 = tail call double @atan(double noundef %250) #11, !tbaa !18
  %252 = fmul double %221, %251
  %253 = add nuw nsw i64 %224, 1
  %254 = icmp eq i64 %224, %22
  br i1 %254, label %255, label %223, !llvm.loop !25

255:                                              ; preds = %223
  %256 = load i32, ptr @J, align 4, !tbaa !18
  %257 = sext i32 %256 to i64
  %258 = load i32, ptr @K, align 4, !tbaa !18
  %259 = sext i32 %258 to i64
  %260 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i64 noundef %15, i64 noundef %257, i64 noundef %259, double noundef %239, double noundef %239, double noundef %252, double noundef %252)
  %261 = load double, ptr @T, align 8, !tbaa !15
  %262 = fmul double %261, 2.000000e+00
  %263 = fadd double %262, 1.000000e+00
  %264 = fmul double %261, %263
  %265 = fadd double %262, %264
  %266 = load double, ptr @T2, align 8, !tbaa !15
  %267 = fdiv double %265, %266
  %268 = load i32, ptr @J, align 4, !tbaa !18
  %269 = sext i32 %268 to i64
  %270 = load i32, ptr @K, align 4, !tbaa !18
  %271 = sext i32 %270 to i64
  %272 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i64 noundef %16, i64 noundef %269, i64 noundef %271, double noundef 1.000000e+00, double noundef 1.000000e+00, double noundef %267, double noundef %267)
  store i32 1, ptr @J, align 4, !tbaa !18
  store i32 2, ptr @K, align 4, !tbaa !18
  store i32 3, ptr @L, align 4, !tbaa !18
  br label %273

273:                                              ; preds = %255, %273
  %274 = phi i64 [ %277, %273 ], [ 1, %255 ]
  %275 = phi double [ %276, %273 ], [ 2.000000e+00, %255 ]
  %276 = phi double [ %275, %273 ], [ 3.000000e+00, %255 ]
  %277 = add nuw nsw i64 %274, 1
  %278 = icmp eq i64 %274, %23
  br i1 %278, label %279, label %273, !llvm.loop !26

279:                                              ; preds = %273
  store double %275, ptr getelementptr inbounds nuw (i8, ptr @E1, i64 8), align 8, !tbaa !15
  store double %276, ptr getelementptr inbounds nuw (i8, ptr @E1, i64 16), align 8, !tbaa !15
  store double %275, ptr getelementptr inbounds nuw (i8, ptr @E1, i64 24), align 8, !tbaa !15
  %280 = load double, ptr getelementptr inbounds nuw (i8, ptr @E1, i64 32), align 8, !tbaa !15
  %281 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i64 noundef %17, i64 noundef 1, i64 noundef 2, double noundef %275, double noundef %276, double noundef %275, double noundef %280)
  store i32 2, ptr @J, align 4, !tbaa !18
  store i32 3, ptr @K, align 4, !tbaa !18
  %282 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i64 noundef 0, i64 noundef 2, i64 noundef 3, double noundef 1.000000e+00, double noundef -1.000000e+00, double noundef -1.000000e+00, double noundef -1.000000e+00)
  %283 = load double, ptr @T1, align 8, !tbaa !15
  br label %284

284:                                              ; preds = %279, %284
  %285 = phi i64 [ 1, %279 ], [ %291, %284 ]
  %286 = phi double [ 7.500000e-01, %279 ], [ %290, %284 ]
  %287 = tail call double @log(double noundef %286) #11, !tbaa !18
  %288 = fdiv double %287, %283
  %289 = tail call double @exp(double noundef %288) #11, !tbaa !18
  %290 = tail call double @sqrt(double noundef %289) #11, !tbaa !18
  %291 = add nuw i64 %285, 1
  %292 = icmp eq i64 %285, %24
  br i1 %292, label %293, label %284, !llvm.loop !27

293:                                              ; preds = %284
  %294 = load i32, ptr @J, align 4, !tbaa !18
  %295 = sext i32 %294 to i64
  %296 = load i32, ptr @K, align 4, !tbaa !18
  %297 = sext i32 %296 to i64
  %298 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i64 noundef %18, i64 noundef %295, i64 noundef %297, double noundef %290, double noundef %290, double noundef %290, double noundef %290)
  %299 = tail call i64 @time(ptr noundef null) #11
  %300 = tail call i32 @putchar(i32 10)
  br i1 %10, label %301, label %51

301:                                              ; preds = %293, %43
  %302 = phi i32 [ 1, %43 ], [ 0, %293 ]
  ret i32 %302
}

; Function Attrs: nounwind
declare i64 @time(ptr noundef) local_unnamed_addr #1

; Function Attrs: nofree nounwind uwtable
define dso_local void @POUT(i64 noundef %0, i64 noundef %1, i64 noundef %2, double noundef %3, double noundef %4, double noundef %5, double noundef %6) local_unnamed_addr #2 {
  %8 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i64 noundef %0, i64 noundef %1, i64 noundef %2, double noundef %3, double noundef %4, double noundef %5, double noundef %6)
  ret void
}

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define dso_local void @PA(ptr noundef captures(none) %0) local_unnamed_addr #3 {
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 32
  %3 = load double, ptr %2, align 8, !tbaa !15
  %4 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %5 = load double, ptr %4, align 8, !tbaa !15
  %6 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %7 = load double, ptr %6, align 8, !tbaa !15
  %8 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %9 = load double, ptr %8, align 8, !tbaa !15
  %10 = load double, ptr @T2, align 8, !tbaa !15
  %11 = load double, ptr @T, align 8, !tbaa !15
  %12 = fadd double %9, %7
  %13 = fadd double %12, %5
  %14 = fsub double %13, %3
  %15 = fmul double %14, %11
  %16 = fadd double %7, %15
  %17 = fsub double %16, %5
  %18 = fadd double %3, %17
  %19 = fmul double %11, %18
  %20 = fsub double %15, %19
  %21 = fadd double %5, %20
  %22 = fadd double %3, %21
  %23 = fmul double %11, %22
  %24 = fsub double %19, %15
  %25 = fadd double %24, %23
  %26 = fadd double %3, %25
  %27 = fdiv double %26, %10
  %28 = fadd double %15, %19
  %29 = fadd double %28, %23
  %30 = fsub double %29, %27
  %31 = fmul double %30, %11
  %32 = fadd double %19, %31
  %33 = fsub double %32, %23
  %34 = fadd double %27, %33
  %35 = fmul double %11, %34
  %36 = fsub double %31, %35
  %37 = fadd double %23, %36
  %38 = fadd double %27, %37
  %39 = fmul double %11, %38
  %40 = fsub double %35, %31
  %41 = fadd double %40, %39
  %42 = fadd double %27, %41
  %43 = fdiv double %42, %10
  %44 = fadd double %31, %35
  %45 = fadd double %44, %39
  %46 = fsub double %45, %43
  %47 = fmul double %46, %11
  %48 = fadd double %35, %47
  %49 = fsub double %48, %39
  %50 = fadd double %43, %49
  %51 = fmul double %11, %50
  %52 = fsub double %47, %51
  %53 = fadd double %39, %52
  %54 = fadd double %43, %53
  %55 = fmul double %11, %54
  %56 = fsub double %51, %47
  %57 = fadd double %56, %55
  %58 = fadd double %43, %57
  %59 = fdiv double %58, %10
  %60 = fadd double %47, %51
  %61 = fadd double %60, %55
  %62 = fsub double %61, %59
  %63 = fmul double %62, %11
  %64 = fadd double %51, %63
  %65 = fsub double %64, %55
  %66 = fadd double %59, %65
  %67 = fmul double %11, %66
  %68 = fsub double %63, %67
  %69 = fadd double %55, %68
  %70 = fadd double %59, %69
  %71 = fmul double %11, %70
  %72 = fsub double %67, %63
  %73 = fadd double %72, %71
  %74 = fadd double %59, %73
  %75 = fdiv double %74, %10
  %76 = fadd double %63, %67
  %77 = fadd double %76, %71
  %78 = fsub double %77, %75
  %79 = fmul double %78, %11
  %80 = fadd double %67, %79
  %81 = fsub double %80, %71
  %82 = fadd double %75, %81
  %83 = fmul double %11, %82
  %84 = fsub double %79, %83
  %85 = fadd double %71, %84
  %86 = fadd double %75, %85
  %87 = fmul double %11, %86
  %88 = fsub double %83, %79
  %89 = fadd double %88, %87
  %90 = fadd double %75, %89
  %91 = fdiv double %90, %10
  %92 = fadd double %79, %83
  %93 = fadd double %92, %87
  %94 = fsub double %93, %91
  %95 = fmul double %94, %11
  %96 = fadd double %83, %95
  %97 = fsub double %96, %87
  %98 = fadd double %91, %97
  %99 = fmul double %11, %98
  %100 = fsub double %95, %99
  %101 = fadd double %87, %100
  %102 = fadd double %91, %101
  %103 = fmul double %11, %102
  %104 = fsub double %99, %95
  %105 = fadd double %104, %103
  %106 = fadd double %91, %105
  %107 = fdiv double %106, %10
  store double %95, ptr %8, align 8, !tbaa !15
  store double %99, ptr %6, align 8, !tbaa !15
  store double %103, ptr %4, align 8, !tbaa !15
  store double %107, ptr %2, align 8, !tbaa !15
  store i32 6, ptr @J, align 4, !tbaa !18
  ret void
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(errnomem: write)
declare double @atan(double noundef) local_unnamed_addr #4

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(errnomem: write)
declare double @sin(double noundef) local_unnamed_addr #4

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(errnomem: write)
declare double @cos(double noundef) local_unnamed_addr #4

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: write, inaccessiblemem: none) uwtable
define dso_local void @P3(double noundef %0, double noundef %1, ptr noundef writeonly captures(none) initializes((0, 8)) %2) local_unnamed_addr #5 {
  %4 = load double, ptr @T, align 8, !tbaa !15
  %5 = fadd double %0, %1
  %6 = fmul double %5, %4
  %7 = fadd double %1, %6
  %8 = fmul double %4, %7
  %9 = fadd double %6, %8
  %10 = load double, ptr @T2, align 8, !tbaa !15
  %11 = fdiv double %9, %10
  store double %11, ptr %2, align 8, !tbaa !15
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @P0() local_unnamed_addr #6 {
  %1 = load i32, ptr @K, align 4, !tbaa !18
  %2 = sext i32 %1 to i64
  %3 = getelementptr inbounds double, ptr @E1, i64 %2
  %4 = load double, ptr %3, align 8, !tbaa !15
  %5 = load i32, ptr @J, align 4, !tbaa !18
  %6 = sext i32 %5 to i64
  %7 = getelementptr inbounds double, ptr @E1, i64 %6
  store double %4, ptr %7, align 8, !tbaa !15
  %8 = load i32, ptr @L, align 4, !tbaa !18
  %9 = sext i32 %8 to i64
  %10 = getelementptr inbounds double, ptr @E1, i64 %9
  %11 = load double, ptr %10, align 8, !tbaa !15
  store double %11, ptr %3, align 8, !tbaa !15
  %12 = load double, ptr %7, align 8, !tbaa !15
  store double %12, ptr %10, align 8, !tbaa !15
  ret void
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(errnomem: write)
declare double @sqrt(double noundef) local_unnamed_addr #4

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(errnomem: write)
declare double @exp(double noundef) local_unnamed_addr #4

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(errnomem: write)
declare double @log(double noundef) local_unnamed_addr #4

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #7

; Function Attrs: mustprogress nocallback nofree nounwind willreturn
declare i64 @strtol(ptr noundef readonly, ptr noundef captures(none), i32 noundef) local_unnamed_addr #8

; Function Attrs: nofree nounwind
declare noundef i32 @putchar(i32 noundef) local_unnamed_addr #9

; Function Attrs: nofree nounwind
declare noundef i64 @fwrite(ptr noundef readonly captures(none), i64 noundef, i64 noundef, ptr noundef captures(none)) local_unnamed_addr #9

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smax.i64(i64, i64) #10

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.vector.reduce.xor.v4i32(<4 x i32>) #10

attributes #0 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { mustprogress nocallback nofree nounwind willreturn memory(errnomem: write) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: write, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #8 = { mustprogress nocallback nofree nounwind willreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #9 = { nofree nounwind }
attributes #10 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #11 = { nounwind }
attributes #12 = { cold }

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
!10 = !{!"Simple C/C++ TBAA"}
!11 = !{!12, !12, i64 0}
!12 = !{!"p1 _ZTS8_IO_FILE", !8, i64 0}
!13 = distinct !{!13, !14}
!14 = !{!"llvm.loop.mustprogress"}
!15 = !{!16, !16, i64 0}
!16 = !{!"double", !9, i64 0}
!17 = distinct !{!17, !14}
!18 = !{!19, !19, i64 0}
!19 = !{!"int", !9, i64 0}
!20 = distinct !{!20, !14}
!21 = distinct !{!21, !14, !22, !23}
!22 = !{!"llvm.loop.isvectorized", i32 1}
!23 = !{!"llvm.loop.unroll.runtime.disable"}
!24 = distinct !{!24, !14, !23, !22}
!25 = distinct !{!25, !14}
!26 = distinct !{!26, !14}
!27 = distinct !{!27, !14}
