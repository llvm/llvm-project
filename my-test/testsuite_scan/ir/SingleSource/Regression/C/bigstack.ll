; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/bigstack.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/bigstack.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.Mixed_struct = type { i32, [10 x double], [10 x [10 x double]], [10 x %struct.Flat_struct] }
%struct.Flat_struct = type { i8, float }

@.str = private unnamed_addr constant [16 x i8] c"Sum(M)  = %.2f\0A\00", align 1
@.str.1 = private unnamed_addr constant [20 x i8] c"Sum(MA[%d]) = %.2f\0A\00", align 1

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read) uwtable
define dso_local double @AddMixed(ptr noundef readonly captures(none) %0) local_unnamed_addr #0 {
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %3 = load double, ptr %2, align 8, !tbaa !6
  %4 = fadd double %3, 0.000000e+00
  %5 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %6 = load double, ptr %5, align 8, !tbaa !6
  %7 = fadd double %4, %6
  %8 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %9 = load double, ptr %8, align 8, !tbaa !6
  %10 = fadd double %7, %9
  %11 = getelementptr inbounds nuw i8, ptr %0, i64 32
  %12 = load double, ptr %11, align 8, !tbaa !6
  %13 = fadd double %10, %12
  %14 = getelementptr inbounds nuw i8, ptr %0, i64 40
  %15 = load double, ptr %14, align 8, !tbaa !6
  %16 = fadd double %13, %15
  %17 = getelementptr inbounds nuw i8, ptr %0, i64 48
  %18 = load double, ptr %17, align 8, !tbaa !6
  %19 = fadd double %16, %18
  %20 = getelementptr inbounds nuw i8, ptr %0, i64 56
  %21 = load double, ptr %20, align 8, !tbaa !6
  %22 = fadd double %19, %21
  %23 = getelementptr inbounds nuw i8, ptr %0, i64 64
  %24 = load double, ptr %23, align 8, !tbaa !6
  %25 = fadd double %22, %24
  %26 = getelementptr inbounds nuw i8, ptr %0, i64 72
  %27 = load double, ptr %26, align 8, !tbaa !6
  %28 = fadd double %25, %27
  %29 = getelementptr inbounds nuw i8, ptr %0, i64 80
  %30 = load double, ptr %29, align 8, !tbaa !6
  %31 = fadd double %28, %30
  %32 = getelementptr inbounds nuw i8, ptr %0, i64 88
  %33 = load double, ptr %32, align 8, !tbaa !6
  %34 = fadd double %31, %33
  %35 = getelementptr inbounds nuw i8, ptr %0, i64 96
  %36 = load double, ptr %35, align 8, !tbaa !6
  %37 = fadd double %34, %36
  %38 = getelementptr inbounds nuw i8, ptr %0, i64 104
  %39 = load double, ptr %38, align 8, !tbaa !6
  %40 = fadd double %37, %39
  %41 = getelementptr inbounds nuw i8, ptr %0, i64 112
  %42 = load double, ptr %41, align 8, !tbaa !6
  %43 = fadd double %40, %42
  %44 = getelementptr inbounds nuw i8, ptr %0, i64 120
  %45 = load double, ptr %44, align 8, !tbaa !6
  %46 = fadd double %43, %45
  %47 = getelementptr inbounds nuw i8, ptr %0, i64 128
  %48 = load double, ptr %47, align 8, !tbaa !6
  %49 = fadd double %46, %48
  %50 = getelementptr inbounds nuw i8, ptr %0, i64 136
  %51 = load double, ptr %50, align 8, !tbaa !6
  %52 = fadd double %49, %51
  %53 = getelementptr inbounds nuw i8, ptr %0, i64 144
  %54 = load double, ptr %53, align 8, !tbaa !6
  %55 = fadd double %52, %54
  %56 = getelementptr inbounds nuw i8, ptr %0, i64 152
  %57 = load double, ptr %56, align 8, !tbaa !6
  %58 = fadd double %55, %57
  %59 = getelementptr inbounds nuw i8, ptr %0, i64 160
  %60 = load double, ptr %59, align 8, !tbaa !6
  %61 = fadd double %58, %60
  %62 = getelementptr inbounds nuw i8, ptr %0, i64 168
  %63 = load double, ptr %62, align 8, !tbaa !6
  %64 = fadd double %61, %63
  %65 = getelementptr inbounds nuw i8, ptr %0, i64 176
  %66 = load double, ptr %65, align 8, !tbaa !6
  %67 = fadd double %64, %66
  %68 = getelementptr inbounds nuw i8, ptr %0, i64 184
  %69 = load double, ptr %68, align 8, !tbaa !6
  %70 = fadd double %67, %69
  %71 = getelementptr inbounds nuw i8, ptr %0, i64 192
  %72 = load double, ptr %71, align 8, !tbaa !6
  %73 = fadd double %70, %72
  %74 = getelementptr inbounds nuw i8, ptr %0, i64 200
  %75 = load double, ptr %74, align 8, !tbaa !6
  %76 = fadd double %73, %75
  %77 = getelementptr inbounds nuw i8, ptr %0, i64 208
  %78 = load double, ptr %77, align 8, !tbaa !6
  %79 = fadd double %76, %78
  %80 = getelementptr inbounds nuw i8, ptr %0, i64 216
  %81 = load double, ptr %80, align 8, !tbaa !6
  %82 = fadd double %79, %81
  %83 = getelementptr inbounds nuw i8, ptr %0, i64 224
  %84 = load double, ptr %83, align 8, !tbaa !6
  %85 = fadd double %82, %84
  %86 = getelementptr inbounds nuw i8, ptr %0, i64 232
  %87 = load double, ptr %86, align 8, !tbaa !6
  %88 = fadd double %85, %87
  %89 = getelementptr inbounds nuw i8, ptr %0, i64 240
  %90 = load double, ptr %89, align 8, !tbaa !6
  %91 = fadd double %88, %90
  %92 = getelementptr inbounds nuw i8, ptr %0, i64 248
  %93 = load double, ptr %92, align 8, !tbaa !6
  %94 = fadd double %91, %93
  %95 = getelementptr inbounds nuw i8, ptr %0, i64 256
  %96 = load double, ptr %95, align 8, !tbaa !6
  %97 = fadd double %94, %96
  %98 = getelementptr inbounds nuw i8, ptr %0, i64 264
  %99 = load double, ptr %98, align 8, !tbaa !6
  %100 = fadd double %97, %99
  %101 = getelementptr inbounds nuw i8, ptr %0, i64 272
  %102 = load double, ptr %101, align 8, !tbaa !6
  %103 = fadd double %100, %102
  %104 = getelementptr inbounds nuw i8, ptr %0, i64 280
  %105 = load double, ptr %104, align 8, !tbaa !6
  %106 = fadd double %103, %105
  %107 = getelementptr inbounds nuw i8, ptr %0, i64 288
  %108 = load double, ptr %107, align 8, !tbaa !6
  %109 = fadd double %106, %108
  %110 = getelementptr inbounds nuw i8, ptr %0, i64 296
  %111 = load double, ptr %110, align 8, !tbaa !6
  %112 = fadd double %109, %111
  %113 = getelementptr inbounds nuw i8, ptr %0, i64 304
  %114 = load double, ptr %113, align 8, !tbaa !6
  %115 = fadd double %112, %114
  %116 = getelementptr inbounds nuw i8, ptr %0, i64 312
  %117 = load double, ptr %116, align 8, !tbaa !6
  %118 = fadd double %115, %117
  %119 = getelementptr inbounds nuw i8, ptr %0, i64 320
  %120 = load double, ptr %119, align 8, !tbaa !6
  %121 = fadd double %118, %120
  %122 = getelementptr inbounds nuw i8, ptr %0, i64 328
  %123 = load double, ptr %122, align 8, !tbaa !6
  %124 = fadd double %121, %123
  %125 = getelementptr inbounds nuw i8, ptr %0, i64 336
  %126 = load double, ptr %125, align 8, !tbaa !6
  %127 = fadd double %124, %126
  %128 = getelementptr inbounds nuw i8, ptr %0, i64 344
  %129 = load double, ptr %128, align 8, !tbaa !6
  %130 = fadd double %127, %129
  %131 = getelementptr inbounds nuw i8, ptr %0, i64 352
  %132 = load double, ptr %131, align 8, !tbaa !6
  %133 = fadd double %130, %132
  %134 = getelementptr inbounds nuw i8, ptr %0, i64 360
  %135 = load double, ptr %134, align 8, !tbaa !6
  %136 = fadd double %133, %135
  %137 = getelementptr inbounds nuw i8, ptr %0, i64 368
  %138 = load double, ptr %137, align 8, !tbaa !6
  %139 = fadd double %136, %138
  %140 = getelementptr inbounds nuw i8, ptr %0, i64 376
  %141 = load double, ptr %140, align 8, !tbaa !6
  %142 = fadd double %139, %141
  %143 = getelementptr inbounds nuw i8, ptr %0, i64 384
  %144 = load double, ptr %143, align 8, !tbaa !6
  %145 = fadd double %142, %144
  %146 = getelementptr inbounds nuw i8, ptr %0, i64 392
  %147 = load double, ptr %146, align 8, !tbaa !6
  %148 = fadd double %145, %147
  %149 = getelementptr inbounds nuw i8, ptr %0, i64 400
  %150 = load double, ptr %149, align 8, !tbaa !6
  %151 = fadd double %148, %150
  %152 = getelementptr inbounds nuw i8, ptr %0, i64 408
  %153 = load double, ptr %152, align 8, !tbaa !6
  %154 = fadd double %151, %153
  %155 = getelementptr inbounds nuw i8, ptr %0, i64 416
  %156 = load double, ptr %155, align 8, !tbaa !6
  %157 = fadd double %154, %156
  %158 = getelementptr inbounds nuw i8, ptr %0, i64 424
  %159 = load double, ptr %158, align 8, !tbaa !6
  %160 = fadd double %157, %159
  %161 = getelementptr inbounds nuw i8, ptr %0, i64 432
  %162 = load double, ptr %161, align 8, !tbaa !6
  %163 = fadd double %160, %162
  %164 = getelementptr inbounds nuw i8, ptr %0, i64 440
  %165 = load double, ptr %164, align 8, !tbaa !6
  %166 = fadd double %163, %165
  %167 = getelementptr inbounds nuw i8, ptr %0, i64 448
  %168 = load double, ptr %167, align 8, !tbaa !6
  %169 = fadd double %166, %168
  %170 = getelementptr inbounds nuw i8, ptr %0, i64 456
  %171 = load double, ptr %170, align 8, !tbaa !6
  %172 = fadd double %169, %171
  %173 = getelementptr inbounds nuw i8, ptr %0, i64 464
  %174 = load double, ptr %173, align 8, !tbaa !6
  %175 = fadd double %172, %174
  %176 = getelementptr inbounds nuw i8, ptr %0, i64 472
  %177 = load double, ptr %176, align 8, !tbaa !6
  %178 = fadd double %175, %177
  %179 = getelementptr inbounds nuw i8, ptr %0, i64 480
  %180 = load double, ptr %179, align 8, !tbaa !6
  %181 = fadd double %178, %180
  %182 = getelementptr inbounds nuw i8, ptr %0, i64 488
  %183 = load double, ptr %182, align 8, !tbaa !6
  %184 = fadd double %181, %183
  %185 = getelementptr inbounds nuw i8, ptr %0, i64 496
  %186 = load double, ptr %185, align 8, !tbaa !6
  %187 = fadd double %184, %186
  %188 = getelementptr inbounds nuw i8, ptr %0, i64 504
  %189 = load double, ptr %188, align 8, !tbaa !6
  %190 = fadd double %187, %189
  %191 = getelementptr inbounds nuw i8, ptr %0, i64 512
  %192 = load double, ptr %191, align 8, !tbaa !6
  %193 = fadd double %190, %192
  %194 = getelementptr inbounds nuw i8, ptr %0, i64 520
  %195 = load double, ptr %194, align 8, !tbaa !6
  %196 = fadd double %193, %195
  %197 = getelementptr inbounds nuw i8, ptr %0, i64 528
  %198 = load double, ptr %197, align 8, !tbaa !6
  %199 = fadd double %196, %198
  %200 = getelementptr inbounds nuw i8, ptr %0, i64 536
  %201 = load double, ptr %200, align 8, !tbaa !6
  %202 = fadd double %199, %201
  %203 = getelementptr inbounds nuw i8, ptr %0, i64 544
  %204 = load double, ptr %203, align 8, !tbaa !6
  %205 = fadd double %202, %204
  %206 = getelementptr inbounds nuw i8, ptr %0, i64 552
  %207 = load double, ptr %206, align 8, !tbaa !6
  %208 = fadd double %205, %207
  %209 = getelementptr inbounds nuw i8, ptr %0, i64 560
  %210 = load double, ptr %209, align 8, !tbaa !6
  %211 = fadd double %208, %210
  %212 = getelementptr inbounds nuw i8, ptr %0, i64 568
  %213 = load double, ptr %212, align 8, !tbaa !6
  %214 = fadd double %211, %213
  %215 = getelementptr inbounds nuw i8, ptr %0, i64 576
  %216 = load double, ptr %215, align 8, !tbaa !6
  %217 = fadd double %214, %216
  %218 = getelementptr inbounds nuw i8, ptr %0, i64 584
  %219 = load double, ptr %218, align 8, !tbaa !6
  %220 = fadd double %217, %219
  %221 = getelementptr inbounds nuw i8, ptr %0, i64 592
  %222 = load double, ptr %221, align 8, !tbaa !6
  %223 = fadd double %220, %222
  %224 = getelementptr inbounds nuw i8, ptr %0, i64 600
  %225 = load double, ptr %224, align 8, !tbaa !6
  %226 = fadd double %223, %225
  %227 = getelementptr inbounds nuw i8, ptr %0, i64 608
  %228 = load double, ptr %227, align 8, !tbaa !6
  %229 = fadd double %226, %228
  %230 = getelementptr inbounds nuw i8, ptr %0, i64 616
  %231 = load double, ptr %230, align 8, !tbaa !6
  %232 = fadd double %229, %231
  %233 = getelementptr inbounds nuw i8, ptr %0, i64 624
  %234 = load double, ptr %233, align 8, !tbaa !6
  %235 = fadd double %232, %234
  %236 = getelementptr inbounds nuw i8, ptr %0, i64 632
  %237 = load double, ptr %236, align 8, !tbaa !6
  %238 = fadd double %235, %237
  %239 = getelementptr inbounds nuw i8, ptr %0, i64 640
  %240 = load double, ptr %239, align 8, !tbaa !6
  %241 = fadd double %238, %240
  %242 = getelementptr inbounds nuw i8, ptr %0, i64 648
  %243 = load double, ptr %242, align 8, !tbaa !6
  %244 = fadd double %241, %243
  %245 = getelementptr inbounds nuw i8, ptr %0, i64 656
  %246 = load double, ptr %245, align 8, !tbaa !6
  %247 = fadd double %244, %246
  %248 = getelementptr inbounds nuw i8, ptr %0, i64 664
  %249 = load double, ptr %248, align 8, !tbaa !6
  %250 = fadd double %247, %249
  %251 = getelementptr inbounds nuw i8, ptr %0, i64 672
  %252 = load double, ptr %251, align 8, !tbaa !6
  %253 = fadd double %250, %252
  %254 = getelementptr inbounds nuw i8, ptr %0, i64 680
  %255 = load double, ptr %254, align 8, !tbaa !6
  %256 = fadd double %253, %255
  %257 = getelementptr inbounds nuw i8, ptr %0, i64 688
  %258 = load double, ptr %257, align 8, !tbaa !6
  %259 = fadd double %256, %258
  %260 = getelementptr inbounds nuw i8, ptr %0, i64 696
  %261 = load double, ptr %260, align 8, !tbaa !6
  %262 = fadd double %259, %261
  %263 = getelementptr inbounds nuw i8, ptr %0, i64 704
  %264 = load double, ptr %263, align 8, !tbaa !6
  %265 = fadd double %262, %264
  %266 = getelementptr inbounds nuw i8, ptr %0, i64 712
  %267 = load double, ptr %266, align 8, !tbaa !6
  %268 = fadd double %265, %267
  %269 = getelementptr inbounds nuw i8, ptr %0, i64 720
  %270 = load double, ptr %269, align 8, !tbaa !6
  %271 = fadd double %268, %270
  %272 = getelementptr inbounds nuw i8, ptr %0, i64 728
  %273 = load double, ptr %272, align 8, !tbaa !6
  %274 = fadd double %271, %273
  %275 = getelementptr inbounds nuw i8, ptr %0, i64 736
  %276 = load double, ptr %275, align 8, !tbaa !6
  %277 = fadd double %274, %276
  %278 = getelementptr inbounds nuw i8, ptr %0, i64 744
  %279 = load double, ptr %278, align 8, !tbaa !6
  %280 = fadd double %277, %279
  %281 = getelementptr inbounds nuw i8, ptr %0, i64 752
  %282 = load double, ptr %281, align 8, !tbaa !6
  %283 = fadd double %280, %282
  %284 = getelementptr inbounds nuw i8, ptr %0, i64 760
  %285 = load double, ptr %284, align 8, !tbaa !6
  %286 = fadd double %283, %285
  %287 = getelementptr inbounds nuw i8, ptr %0, i64 768
  %288 = load double, ptr %287, align 8, !tbaa !6
  %289 = fadd double %286, %288
  %290 = getelementptr inbounds nuw i8, ptr %0, i64 776
  %291 = load double, ptr %290, align 8, !tbaa !6
  %292 = fadd double %289, %291
  %293 = getelementptr inbounds nuw i8, ptr %0, i64 784
  %294 = load double, ptr %293, align 8, !tbaa !6
  %295 = fadd double %292, %294
  %296 = getelementptr inbounds nuw i8, ptr %0, i64 792
  %297 = load double, ptr %296, align 8, !tbaa !6
  %298 = fadd double %295, %297
  %299 = getelementptr inbounds nuw i8, ptr %0, i64 800
  %300 = load double, ptr %299, align 8, !tbaa !6
  %301 = fadd double %298, %300
  %302 = getelementptr inbounds nuw i8, ptr %0, i64 808
  %303 = load double, ptr %302, align 8, !tbaa !6
  %304 = fadd double %301, %303
  %305 = getelementptr inbounds nuw i8, ptr %0, i64 816
  %306 = load double, ptr %305, align 8, !tbaa !6
  %307 = fadd double %304, %306
  %308 = getelementptr inbounds nuw i8, ptr %0, i64 824
  %309 = load double, ptr %308, align 8, !tbaa !6
  %310 = fadd double %307, %309
  %311 = getelementptr inbounds nuw i8, ptr %0, i64 832
  %312 = load double, ptr %311, align 8, !tbaa !6
  %313 = fadd double %310, %312
  %314 = getelementptr inbounds nuw i8, ptr %0, i64 840
  %315 = load double, ptr %314, align 8, !tbaa !6
  %316 = fadd double %313, %315
  %317 = getelementptr inbounds nuw i8, ptr %0, i64 848
  %318 = load double, ptr %317, align 8, !tbaa !6
  %319 = fadd double %316, %318
  %320 = getelementptr inbounds nuw i8, ptr %0, i64 856
  %321 = load double, ptr %320, align 8, !tbaa !6
  %322 = fadd double %319, %321
  %323 = getelementptr inbounds nuw i8, ptr %0, i64 864
  %324 = load double, ptr %323, align 8, !tbaa !6
  %325 = fadd double %322, %324
  %326 = getelementptr inbounds nuw i8, ptr %0, i64 872
  %327 = load double, ptr %326, align 8, !tbaa !6
  %328 = fadd double %325, %327
  %329 = getelementptr inbounds nuw i8, ptr %0, i64 880
  %330 = load double, ptr %329, align 8, !tbaa !6
  %331 = fadd double %328, %330
  %332 = getelementptr inbounds nuw i8, ptr %0, i64 888
  %333 = load i8, ptr %332, align 8, !tbaa !10
  %334 = uitofp i8 %333 to double
  %335 = fadd double %331, %334
  %336 = getelementptr inbounds nuw i8, ptr %0, i64 892
  %337 = load float, ptr %336, align 4, !tbaa !13
  %338 = fpext float %337 to double
  %339 = fadd double %335, %338
  %340 = getelementptr inbounds nuw i8, ptr %0, i64 896
  %341 = load i8, ptr %340, align 8, !tbaa !10
  %342 = uitofp i8 %341 to double
  %343 = fadd double %339, %342
  %344 = getelementptr inbounds nuw i8, ptr %0, i64 900
  %345 = load float, ptr %344, align 4, !tbaa !13
  %346 = fpext float %345 to double
  %347 = fadd double %343, %346
  %348 = getelementptr inbounds nuw i8, ptr %0, i64 904
  %349 = load i8, ptr %348, align 8, !tbaa !10
  %350 = uitofp i8 %349 to double
  %351 = fadd double %347, %350
  %352 = getelementptr inbounds nuw i8, ptr %0, i64 908
  %353 = load float, ptr %352, align 4, !tbaa !13
  %354 = fpext float %353 to double
  %355 = fadd double %351, %354
  %356 = getelementptr inbounds nuw i8, ptr %0, i64 912
  %357 = load i8, ptr %356, align 8, !tbaa !10
  %358 = uitofp i8 %357 to double
  %359 = fadd double %355, %358
  %360 = getelementptr inbounds nuw i8, ptr %0, i64 916
  %361 = load float, ptr %360, align 4, !tbaa !13
  %362 = fpext float %361 to double
  %363 = fadd double %359, %362
  %364 = getelementptr inbounds nuw i8, ptr %0, i64 920
  %365 = load i8, ptr %364, align 8, !tbaa !10
  %366 = uitofp i8 %365 to double
  %367 = fadd double %363, %366
  %368 = getelementptr inbounds nuw i8, ptr %0, i64 924
  %369 = load float, ptr %368, align 4, !tbaa !13
  %370 = fpext float %369 to double
  %371 = fadd double %367, %370
  %372 = getelementptr inbounds nuw i8, ptr %0, i64 928
  %373 = load i8, ptr %372, align 8, !tbaa !10
  %374 = uitofp i8 %373 to double
  %375 = fadd double %371, %374
  %376 = getelementptr inbounds nuw i8, ptr %0, i64 932
  %377 = load float, ptr %376, align 4, !tbaa !13
  %378 = fpext float %377 to double
  %379 = fadd double %375, %378
  %380 = getelementptr inbounds nuw i8, ptr %0, i64 936
  %381 = load i8, ptr %380, align 8, !tbaa !10
  %382 = uitofp i8 %381 to double
  %383 = fadd double %379, %382
  %384 = getelementptr inbounds nuw i8, ptr %0, i64 940
  %385 = load float, ptr %384, align 4, !tbaa !13
  %386 = fpext float %385 to double
  %387 = fadd double %383, %386
  %388 = getelementptr inbounds nuw i8, ptr %0, i64 944
  %389 = load i8, ptr %388, align 8, !tbaa !10
  %390 = uitofp i8 %389 to double
  %391 = fadd double %387, %390
  %392 = getelementptr inbounds nuw i8, ptr %0, i64 948
  %393 = load float, ptr %392, align 4, !tbaa !13
  %394 = fpext float %393 to double
  %395 = fadd double %391, %394
  %396 = getelementptr inbounds nuw i8, ptr %0, i64 952
  %397 = load i8, ptr %396, align 8, !tbaa !10
  %398 = uitofp i8 %397 to double
  %399 = fadd double %395, %398
  %400 = getelementptr inbounds nuw i8, ptr %0, i64 956
  %401 = load float, ptr %400, align 4, !tbaa !13
  %402 = fpext float %401 to double
  %403 = fadd double %399, %402
  %404 = getelementptr inbounds nuw i8, ptr %0, i64 960
  %405 = load i8, ptr %404, align 8, !tbaa !10
  %406 = uitofp i8 %405 to double
  %407 = fadd double %403, %406
  %408 = getelementptr inbounds nuw i8, ptr %0, i64 964
  %409 = load float, ptr %408, align 4, !tbaa !13
  %410 = fpext float %409 to double
  %411 = fadd double %407, %410
  ret double %411
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define dso_local void @InitializeMixed(ptr noundef writeonly captures(none) initializes((8, 889), (892, 897), (900, 905), (908, 913), (916, 921), (924, 929), (932, 937), (940, 945), (948, 953), (956, 961), (964, 968)) %0, i32 noundef %1) local_unnamed_addr #2 {
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %4 = sitofp i32 %1 to double
  store double %4, ptr %3, align 8, !tbaa !6
  %5 = insertelement <2 x i32> poison, i32 %1, i64 0
  %6 = shufflevector <2 x i32> %5, <2 x i32> poison, <2 x i32> zeroinitializer
  %7 = add <2 x i32> %6, <i32 1, i32 2>
  %8 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %9 = sitofp <2 x i32> %7 to <2 x double>
  store <2 x double> %9, ptr %8, align 8, !tbaa !6
  %10 = add <2 x i32> %6, <i32 3, i32 4>
  %11 = getelementptr inbounds nuw i8, ptr %0, i64 32
  %12 = sitofp <2 x i32> %10 to <2 x double>
  store <2 x double> %12, ptr %11, align 8, !tbaa !6
  %13 = add <2 x i32> %6, <i32 5, i32 6>
  %14 = getelementptr inbounds nuw i8, ptr %0, i64 48
  %15 = sitofp <2 x i32> %13 to <2 x double>
  store <2 x double> %15, ptr %14, align 8, !tbaa !6
  %16 = add <2 x i32> %6, <i32 7, i32 8>
  %17 = getelementptr inbounds nuw i8, ptr %0, i64 64
  %18 = sitofp <2 x i32> %16 to <2 x double>
  store <2 x double> %18, ptr %17, align 8, !tbaa !6
  %19 = add i32 %1, 9
  %20 = sitofp i32 %19 to double
  %21 = getelementptr inbounds nuw i8, ptr %0, i64 80
  store double %20, ptr %21, align 8, !tbaa !6
  %22 = getelementptr inbounds nuw i8, ptr %0, i64 88
  store double %4, ptr %22, align 8, !tbaa !6
  %23 = add <2 x i32> %6, <i32 1, i32 2>
  %24 = getelementptr inbounds nuw i8, ptr %0, i64 96
  %25 = sitofp <2 x i32> %23 to <2 x double>
  store <2 x double> %25, ptr %24, align 8, !tbaa !6
  %26 = add <2 x i32> %6, <i32 3, i32 4>
  %27 = getelementptr inbounds nuw i8, ptr %0, i64 112
  %28 = sitofp <2 x i32> %26 to <2 x double>
  store <2 x double> %28, ptr %27, align 8, !tbaa !6
  %29 = add <2 x i32> %6, <i32 5, i32 6>
  %30 = getelementptr inbounds nuw i8, ptr %0, i64 128
  %31 = sitofp <2 x i32> %29 to <2 x double>
  store <2 x double> %31, ptr %30, align 8, !tbaa !6
  %32 = add <2 x i32> %6, <i32 7, i32 8>
  %33 = getelementptr inbounds nuw i8, ptr %0, i64 144
  %34 = sitofp <2 x i32> %32 to <2 x double>
  store <2 x double> %34, ptr %33, align 8, !tbaa !6
  %35 = add <2 x i32> %6, <i32 9, i32 10>
  %36 = getelementptr inbounds nuw i8, ptr %0, i64 160
  %37 = sitofp <2 x i32> %35 to <2 x double>
  store <2 x double> %37, ptr %36, align 8, !tbaa !6
  %38 = add <2 x i32> %6, <i32 11, i32 12>
  %39 = getelementptr inbounds nuw i8, ptr %0, i64 176
  %40 = sitofp <2 x i32> %38 to <2 x double>
  store <2 x double> %40, ptr %39, align 8, !tbaa !6
  %41 = add <2 x i32> %6, <i32 13, i32 14>
  %42 = getelementptr inbounds nuw i8, ptr %0, i64 192
  %43 = sitofp <2 x i32> %41 to <2 x double>
  store <2 x double> %43, ptr %42, align 8, !tbaa !6
  %44 = add <2 x i32> %6, <i32 15, i32 16>
  %45 = getelementptr inbounds nuw i8, ptr %0, i64 208
  %46 = sitofp <2 x i32> %44 to <2 x double>
  store <2 x double> %46, ptr %45, align 8, !tbaa !6
  %47 = add <2 x i32> %6, <i32 17, i32 18>
  %48 = getelementptr inbounds nuw i8, ptr %0, i64 224
  %49 = sitofp <2 x i32> %47 to <2 x double>
  store <2 x double> %49, ptr %48, align 8, !tbaa !6
  %50 = add <2 x i32> %6, <i32 19, i32 20>
  %51 = getelementptr inbounds nuw i8, ptr %0, i64 240
  %52 = sitofp <2 x i32> %50 to <2 x double>
  store <2 x double> %52, ptr %51, align 8, !tbaa !6
  %53 = add <2 x i32> %6, <i32 21, i32 22>
  %54 = getelementptr inbounds nuw i8, ptr %0, i64 256
  %55 = sitofp <2 x i32> %53 to <2 x double>
  store <2 x double> %55, ptr %54, align 8, !tbaa !6
  %56 = add <2 x i32> %6, <i32 23, i32 24>
  %57 = getelementptr inbounds nuw i8, ptr %0, i64 272
  %58 = sitofp <2 x i32> %56 to <2 x double>
  store <2 x double> %58, ptr %57, align 8, !tbaa !6
  %59 = add <2 x i32> %6, <i32 25, i32 26>
  %60 = getelementptr inbounds nuw i8, ptr %0, i64 288
  %61 = sitofp <2 x i32> %59 to <2 x double>
  store <2 x double> %61, ptr %60, align 8, !tbaa !6
  %62 = add <2 x i32> %6, <i32 27, i32 28>
  %63 = getelementptr inbounds nuw i8, ptr %0, i64 304
  %64 = sitofp <2 x i32> %62 to <2 x double>
  store <2 x double> %64, ptr %63, align 8, !tbaa !6
  %65 = add <2 x i32> %6, <i32 29, i32 30>
  %66 = getelementptr inbounds nuw i8, ptr %0, i64 320
  %67 = sitofp <2 x i32> %65 to <2 x double>
  store <2 x double> %67, ptr %66, align 8, !tbaa !6
  %68 = add <2 x i32> %6, <i32 31, i32 32>
  %69 = getelementptr inbounds nuw i8, ptr %0, i64 336
  %70 = sitofp <2 x i32> %68 to <2 x double>
  store <2 x double> %70, ptr %69, align 8, !tbaa !6
  %71 = add <2 x i32> %6, <i32 33, i32 34>
  %72 = getelementptr inbounds nuw i8, ptr %0, i64 352
  %73 = sitofp <2 x i32> %71 to <2 x double>
  store <2 x double> %73, ptr %72, align 8, !tbaa !6
  %74 = add <2 x i32> %6, <i32 35, i32 36>
  %75 = getelementptr inbounds nuw i8, ptr %0, i64 368
  %76 = sitofp <2 x i32> %74 to <2 x double>
  store <2 x double> %76, ptr %75, align 8, !tbaa !6
  %77 = add <2 x i32> %6, <i32 37, i32 38>
  %78 = getelementptr inbounds nuw i8, ptr %0, i64 384
  %79 = sitofp <2 x i32> %77 to <2 x double>
  store <2 x double> %79, ptr %78, align 8, !tbaa !6
  %80 = add <2 x i32> %6, <i32 39, i32 40>
  %81 = getelementptr inbounds nuw i8, ptr %0, i64 400
  %82 = sitofp <2 x i32> %80 to <2 x double>
  store <2 x double> %82, ptr %81, align 8, !tbaa !6
  %83 = add <2 x i32> %6, <i32 41, i32 42>
  %84 = getelementptr inbounds nuw i8, ptr %0, i64 416
  %85 = sitofp <2 x i32> %83 to <2 x double>
  store <2 x double> %85, ptr %84, align 8, !tbaa !6
  %86 = add <2 x i32> %6, <i32 43, i32 44>
  %87 = getelementptr inbounds nuw i8, ptr %0, i64 432
  %88 = sitofp <2 x i32> %86 to <2 x double>
  store <2 x double> %88, ptr %87, align 8, !tbaa !6
  %89 = add <2 x i32> %6, <i32 45, i32 46>
  %90 = getelementptr inbounds nuw i8, ptr %0, i64 448
  %91 = sitofp <2 x i32> %89 to <2 x double>
  store <2 x double> %91, ptr %90, align 8, !tbaa !6
  %92 = add <2 x i32> %6, <i32 47, i32 48>
  %93 = getelementptr inbounds nuw i8, ptr %0, i64 464
  %94 = sitofp <2 x i32> %92 to <2 x double>
  store <2 x double> %94, ptr %93, align 8, !tbaa !6
  %95 = add <2 x i32> %6, <i32 49, i32 50>
  %96 = getelementptr inbounds nuw i8, ptr %0, i64 480
  %97 = sitofp <2 x i32> %95 to <2 x double>
  store <2 x double> %97, ptr %96, align 8, !tbaa !6
  %98 = add <2 x i32> %6, <i32 51, i32 52>
  %99 = getelementptr inbounds nuw i8, ptr %0, i64 496
  %100 = sitofp <2 x i32> %98 to <2 x double>
  store <2 x double> %100, ptr %99, align 8, !tbaa !6
  %101 = add <2 x i32> %6, <i32 53, i32 54>
  %102 = getelementptr inbounds nuw i8, ptr %0, i64 512
  %103 = sitofp <2 x i32> %101 to <2 x double>
  store <2 x double> %103, ptr %102, align 8, !tbaa !6
  %104 = add <2 x i32> %6, <i32 55, i32 56>
  %105 = getelementptr inbounds nuw i8, ptr %0, i64 528
  %106 = sitofp <2 x i32> %104 to <2 x double>
  store <2 x double> %106, ptr %105, align 8, !tbaa !6
  %107 = add <2 x i32> %6, <i32 57, i32 58>
  %108 = getelementptr inbounds nuw i8, ptr %0, i64 544
  %109 = sitofp <2 x i32> %107 to <2 x double>
  store <2 x double> %109, ptr %108, align 8, !tbaa !6
  %110 = add <2 x i32> %6, <i32 59, i32 60>
  %111 = getelementptr inbounds nuw i8, ptr %0, i64 560
  %112 = sitofp <2 x i32> %110 to <2 x double>
  store <2 x double> %112, ptr %111, align 8, !tbaa !6
  %113 = add <2 x i32> %6, <i32 61, i32 62>
  %114 = getelementptr inbounds nuw i8, ptr %0, i64 576
  %115 = sitofp <2 x i32> %113 to <2 x double>
  store <2 x double> %115, ptr %114, align 8, !tbaa !6
  %116 = add <2 x i32> %6, <i32 63, i32 64>
  %117 = getelementptr inbounds nuw i8, ptr %0, i64 592
  %118 = sitofp <2 x i32> %116 to <2 x double>
  store <2 x double> %118, ptr %117, align 8, !tbaa !6
  %119 = add <2 x i32> %6, <i32 65, i32 66>
  %120 = getelementptr inbounds nuw i8, ptr %0, i64 608
  %121 = sitofp <2 x i32> %119 to <2 x double>
  store <2 x double> %121, ptr %120, align 8, !tbaa !6
  %122 = add <2 x i32> %6, <i32 67, i32 68>
  %123 = getelementptr inbounds nuw i8, ptr %0, i64 624
  %124 = sitofp <2 x i32> %122 to <2 x double>
  store <2 x double> %124, ptr %123, align 8, !tbaa !6
  %125 = add <2 x i32> %6, <i32 69, i32 70>
  %126 = getelementptr inbounds nuw i8, ptr %0, i64 640
  %127 = sitofp <2 x i32> %125 to <2 x double>
  store <2 x double> %127, ptr %126, align 8, !tbaa !6
  %128 = add <2 x i32> %6, <i32 71, i32 72>
  %129 = getelementptr inbounds nuw i8, ptr %0, i64 656
  %130 = sitofp <2 x i32> %128 to <2 x double>
  store <2 x double> %130, ptr %129, align 8, !tbaa !6
  %131 = add <2 x i32> %6, <i32 73, i32 74>
  %132 = getelementptr inbounds nuw i8, ptr %0, i64 672
  %133 = sitofp <2 x i32> %131 to <2 x double>
  store <2 x double> %133, ptr %132, align 8, !tbaa !6
  %134 = add <2 x i32> %6, <i32 75, i32 76>
  %135 = getelementptr inbounds nuw i8, ptr %0, i64 688
  %136 = sitofp <2 x i32> %134 to <2 x double>
  store <2 x double> %136, ptr %135, align 8, !tbaa !6
  %137 = add <2 x i32> %6, <i32 77, i32 78>
  %138 = getelementptr inbounds nuw i8, ptr %0, i64 704
  %139 = sitofp <2 x i32> %137 to <2 x double>
  store <2 x double> %139, ptr %138, align 8, !tbaa !6
  %140 = add <2 x i32> %6, <i32 79, i32 80>
  %141 = getelementptr inbounds nuw i8, ptr %0, i64 720
  %142 = sitofp <2 x i32> %140 to <2 x double>
  store <2 x double> %142, ptr %141, align 8, !tbaa !6
  %143 = add <2 x i32> %6, <i32 81, i32 82>
  %144 = getelementptr inbounds nuw i8, ptr %0, i64 736
  %145 = sitofp <2 x i32> %143 to <2 x double>
  store <2 x double> %145, ptr %144, align 8, !tbaa !6
  %146 = add <2 x i32> %6, <i32 83, i32 84>
  %147 = getelementptr inbounds nuw i8, ptr %0, i64 752
  %148 = sitofp <2 x i32> %146 to <2 x double>
  store <2 x double> %148, ptr %147, align 8, !tbaa !6
  %149 = add <2 x i32> %6, <i32 85, i32 86>
  %150 = getelementptr inbounds nuw i8, ptr %0, i64 768
  %151 = sitofp <2 x i32> %149 to <2 x double>
  store <2 x double> %151, ptr %150, align 8, !tbaa !6
  %152 = add <2 x i32> %6, <i32 87, i32 88>
  %153 = getelementptr inbounds nuw i8, ptr %0, i64 784
  %154 = sitofp <2 x i32> %152 to <2 x double>
  store <2 x double> %154, ptr %153, align 8, !tbaa !6
  %155 = add <2 x i32> %6, <i32 89, i32 90>
  %156 = getelementptr inbounds nuw i8, ptr %0, i64 800
  %157 = sitofp <2 x i32> %155 to <2 x double>
  store <2 x double> %157, ptr %156, align 8, !tbaa !6
  %158 = add <2 x i32> %6, <i32 91, i32 92>
  %159 = getelementptr inbounds nuw i8, ptr %0, i64 816
  %160 = sitofp <2 x i32> %158 to <2 x double>
  store <2 x double> %160, ptr %159, align 8, !tbaa !6
  %161 = add <2 x i32> %6, <i32 93, i32 94>
  %162 = getelementptr inbounds nuw i8, ptr %0, i64 832
  %163 = sitofp <2 x i32> %161 to <2 x double>
  store <2 x double> %163, ptr %162, align 8, !tbaa !6
  %164 = add <2 x i32> %6, <i32 95, i32 96>
  %165 = getelementptr inbounds nuw i8, ptr %0, i64 848
  %166 = sitofp <2 x i32> %164 to <2 x double>
  store <2 x double> %166, ptr %165, align 8, !tbaa !6
  %167 = add <2 x i32> %6, <i32 97, i32 98>
  %168 = getelementptr inbounds nuw i8, ptr %0, i64 864
  %169 = sitofp <2 x i32> %167 to <2 x double>
  store <2 x double> %169, ptr %168, align 8, !tbaa !6
  %170 = add i32 %1, 99
  %171 = sitofp i32 %170 to double
  %172 = getelementptr inbounds nuw i8, ptr %0, i64 880
  store double %171, ptr %172, align 8, !tbaa !6
  %173 = getelementptr inbounds nuw i8, ptr %0, i64 888
  %174 = sitofp i32 %1 to float
  store i8 81, ptr %173, align 8, !tbaa !10
  %175 = getelementptr inbounds nuw i8, ptr %0, i64 892
  store float %174, ptr %175, align 4, !tbaa !13
  %176 = getelementptr inbounds nuw i8, ptr %0, i64 896
  store i8 81, ptr %176, align 8, !tbaa !10
  %177 = getelementptr inbounds nuw i8, ptr %0, i64 900
  store float %174, ptr %177, align 4, !tbaa !13
  %178 = getelementptr inbounds nuw i8, ptr %0, i64 904
  store i8 81, ptr %178, align 8, !tbaa !10
  %179 = getelementptr inbounds nuw i8, ptr %0, i64 908
  store float %174, ptr %179, align 4, !tbaa !13
  %180 = getelementptr inbounds nuw i8, ptr %0, i64 912
  store i8 81, ptr %180, align 8, !tbaa !10
  %181 = getelementptr inbounds nuw i8, ptr %0, i64 916
  store float %174, ptr %181, align 4, !tbaa !13
  %182 = getelementptr inbounds nuw i8, ptr %0, i64 920
  store i8 81, ptr %182, align 8, !tbaa !10
  %183 = getelementptr inbounds nuw i8, ptr %0, i64 924
  store float %174, ptr %183, align 4, !tbaa !13
  %184 = getelementptr inbounds nuw i8, ptr %0, i64 928
  store i8 81, ptr %184, align 8, !tbaa !10
  %185 = getelementptr inbounds nuw i8, ptr %0, i64 932
  store float %174, ptr %185, align 4, !tbaa !13
  %186 = getelementptr inbounds nuw i8, ptr %0, i64 936
  store i8 81, ptr %186, align 8, !tbaa !10
  %187 = getelementptr inbounds nuw i8, ptr %0, i64 940
  store float %174, ptr %187, align 4, !tbaa !13
  %188 = getelementptr inbounds nuw i8, ptr %0, i64 944
  store i8 81, ptr %188, align 8, !tbaa !10
  %189 = getelementptr inbounds nuw i8, ptr %0, i64 948
  store float %174, ptr %189, align 4, !tbaa !13
  %190 = getelementptr inbounds nuw i8, ptr %0, i64 952
  store i8 81, ptr %190, align 8, !tbaa !10
  %191 = getelementptr inbounds nuw i8, ptr %0, i64 956
  store float %174, ptr %191, align 4, !tbaa !13
  %192 = getelementptr inbounds nuw i8, ptr %0, i64 960
  store i8 81, ptr %192, align 8, !tbaa !10
  %193 = getelementptr inbounds nuw i8, ptr %0, i64 964
  store float %174, ptr %193, align 4, !tbaa !13
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readnone captures(none) %1) local_unnamed_addr #3 {
  %3 = alloca %struct.Mixed_struct, align 8
  %4 = alloca [4 x %struct.Mixed_struct], align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #5
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #5
  %5 = getelementptr inbounds nuw i8, ptr %3, i64 8
  store <2 x double> <double 1.000000e+02, double 1.010000e+02>, ptr %5, align 8, !tbaa !6
  %6 = getelementptr inbounds nuw i8, ptr %3, i64 24
  store <2 x double> <double 1.020000e+02, double 1.030000e+02>, ptr %6, align 8, !tbaa !6
  %7 = getelementptr inbounds nuw i8, ptr %3, i64 40
  store <2 x double> <double 1.040000e+02, double 1.050000e+02>, ptr %7, align 8, !tbaa !6
  %8 = getelementptr inbounds nuw i8, ptr %3, i64 56
  store <2 x double> <double 1.060000e+02, double 1.070000e+02>, ptr %8, align 8, !tbaa !6
  %9 = getelementptr inbounds nuw i8, ptr %3, i64 72
  store <2 x double> <double 1.080000e+02, double 1.090000e+02>, ptr %9, align 8, !tbaa !6
  %10 = getelementptr inbounds nuw i8, ptr %3, i64 88
  store <2 x double> <double 1.000000e+02, double 1.010000e+02>, ptr %10, align 8, !tbaa !6
  %11 = getelementptr inbounds nuw i8, ptr %3, i64 104
  store <2 x double> <double 1.020000e+02, double 1.030000e+02>, ptr %11, align 8, !tbaa !6
  %12 = getelementptr inbounds nuw i8, ptr %3, i64 120
  store <2 x double> <double 1.040000e+02, double 1.050000e+02>, ptr %12, align 8, !tbaa !6
  %13 = getelementptr inbounds nuw i8, ptr %3, i64 136
  store <2 x double> <double 1.060000e+02, double 1.070000e+02>, ptr %13, align 8, !tbaa !6
  %14 = getelementptr inbounds nuw i8, ptr %3, i64 152
  store <2 x double> <double 1.080000e+02, double 1.090000e+02>, ptr %14, align 8, !tbaa !6
  %15 = getelementptr inbounds nuw i8, ptr %3, i64 168
  store <2 x double> <double 1.100000e+02, double 1.110000e+02>, ptr %15, align 8, !tbaa !6
  %16 = getelementptr inbounds nuw i8, ptr %3, i64 184
  store <2 x double> <double 1.120000e+02, double 1.130000e+02>, ptr %16, align 8, !tbaa !6
  %17 = getelementptr inbounds nuw i8, ptr %3, i64 200
  store <2 x double> <double 1.140000e+02, double 1.150000e+02>, ptr %17, align 8, !tbaa !6
  %18 = getelementptr inbounds nuw i8, ptr %3, i64 216
  store <2 x double> <double 1.160000e+02, double 1.170000e+02>, ptr %18, align 8, !tbaa !6
  %19 = getelementptr inbounds nuw i8, ptr %3, i64 232
  store <2 x double> <double 1.180000e+02, double 1.190000e+02>, ptr %19, align 8, !tbaa !6
  %20 = getelementptr inbounds nuw i8, ptr %3, i64 248
  store <2 x double> <double 1.200000e+02, double 1.210000e+02>, ptr %20, align 8, !tbaa !6
  %21 = getelementptr inbounds nuw i8, ptr %3, i64 264
  store double 1.220000e+02, ptr %21, align 8, !tbaa !6
  %22 = getelementptr inbounds nuw i8, ptr %3, i64 272
  store <2 x double> <double 1.230000e+02, double 1.240000e+02>, ptr %22, align 8, !tbaa !6
  %23 = getelementptr inbounds nuw i8, ptr %3, i64 288
  store <2 x double> <double 1.250000e+02, double 1.260000e+02>, ptr %23, align 8, !tbaa !6
  %24 = getelementptr inbounds nuw i8, ptr %3, i64 304
  store <2 x double> <double 1.270000e+02, double 1.280000e+02>, ptr %24, align 8, !tbaa !6
  %25 = getelementptr inbounds nuw i8, ptr %3, i64 320
  store <2 x double> <double 1.290000e+02, double 1.300000e+02>, ptr %25, align 8, !tbaa !6
  %26 = getelementptr inbounds nuw i8, ptr %3, i64 336
  store <2 x double> <double 1.310000e+02, double 1.320000e+02>, ptr %26, align 8, !tbaa !6
  %27 = getelementptr inbounds nuw i8, ptr %3, i64 352
  store <2 x double> <double 1.330000e+02, double 1.340000e+02>, ptr %27, align 8, !tbaa !6
  %28 = getelementptr inbounds nuw i8, ptr %3, i64 368
  store <2 x double> <double 1.350000e+02, double 1.360000e+02>, ptr %28, align 8, !tbaa !6
  %29 = getelementptr inbounds nuw i8, ptr %3, i64 384
  store <2 x double> <double 1.370000e+02, double 1.380000e+02>, ptr %29, align 8, !tbaa !6
  %30 = getelementptr inbounds nuw i8, ptr %3, i64 400
  store <2 x double> <double 1.390000e+02, double 1.400000e+02>, ptr %30, align 8, !tbaa !6
  %31 = getelementptr inbounds nuw i8, ptr %3, i64 416
  store <2 x double> <double 1.410000e+02, double 1.420000e+02>, ptr %31, align 8, !tbaa !6
  %32 = getelementptr inbounds nuw i8, ptr %3, i64 432
  store <2 x double> <double 1.430000e+02, double 1.440000e+02>, ptr %32, align 8, !tbaa !6
  %33 = getelementptr inbounds nuw i8, ptr %3, i64 448
  store <2 x double> <double 1.450000e+02, double 1.460000e+02>, ptr %33, align 8, !tbaa !6
  %34 = getelementptr inbounds nuw i8, ptr %3, i64 464
  store <2 x double> <double 1.470000e+02, double 1.480000e+02>, ptr %34, align 8, !tbaa !6
  %35 = getelementptr inbounds nuw i8, ptr %3, i64 480
  store <2 x double> <double 1.490000e+02, double 1.500000e+02>, ptr %35, align 8, !tbaa !6
  %36 = getelementptr inbounds nuw i8, ptr %3, i64 496
  store <2 x double> <double 1.510000e+02, double 1.520000e+02>, ptr %36, align 8, !tbaa !6
  %37 = getelementptr inbounds nuw i8, ptr %3, i64 512
  store <2 x double> <double 1.530000e+02, double 1.540000e+02>, ptr %37, align 8, !tbaa !6
  %38 = getelementptr inbounds nuw i8, ptr %3, i64 528
  store <2 x double> <double 1.550000e+02, double 1.560000e+02>, ptr %38, align 8, !tbaa !6
  %39 = getelementptr inbounds nuw i8, ptr %3, i64 544
  store <2 x double> <double 1.570000e+02, double 1.580000e+02>, ptr %39, align 8, !tbaa !6
  %40 = getelementptr inbounds nuw i8, ptr %3, i64 560
  store <2 x double> <double 1.590000e+02, double 1.600000e+02>, ptr %40, align 8, !tbaa !6
  %41 = getelementptr inbounds nuw i8, ptr %3, i64 576
  store <2 x double> <double 1.610000e+02, double 1.620000e+02>, ptr %41, align 8, !tbaa !6
  %42 = getelementptr inbounds nuw i8, ptr %3, i64 592
  store <2 x double> <double 1.630000e+02, double 1.640000e+02>, ptr %42, align 8, !tbaa !6
  %43 = getelementptr inbounds nuw i8, ptr %3, i64 608
  store <2 x double> <double 1.650000e+02, double 1.660000e+02>, ptr %43, align 8, !tbaa !6
  %44 = getelementptr inbounds nuw i8, ptr %3, i64 624
  store <2 x double> <double 1.670000e+02, double 1.680000e+02>, ptr %44, align 8, !tbaa !6
  %45 = getelementptr inbounds nuw i8, ptr %3, i64 640
  store <2 x double> <double 1.690000e+02, double 1.700000e+02>, ptr %45, align 8, !tbaa !6
  %46 = getelementptr inbounds nuw i8, ptr %3, i64 656
  store <2 x double> <double 1.710000e+02, double 1.720000e+02>, ptr %46, align 8, !tbaa !6
  %47 = getelementptr inbounds nuw i8, ptr %3, i64 672
  store <2 x double> <double 1.730000e+02, double 1.740000e+02>, ptr %47, align 8, !tbaa !6
  %48 = getelementptr inbounds nuw i8, ptr %3, i64 688
  store <2 x double> <double 1.750000e+02, double 1.760000e+02>, ptr %48, align 8, !tbaa !6
  %49 = getelementptr inbounds nuw i8, ptr %3, i64 704
  store <2 x double> <double 1.770000e+02, double 1.780000e+02>, ptr %49, align 8, !tbaa !6
  %50 = getelementptr inbounds nuw i8, ptr %3, i64 720
  store <2 x double> <double 1.790000e+02, double 1.800000e+02>, ptr %50, align 8, !tbaa !6
  %51 = getelementptr inbounds nuw i8, ptr %3, i64 736
  store <2 x double> <double 1.810000e+02, double 1.820000e+02>, ptr %51, align 8, !tbaa !6
  %52 = getelementptr inbounds nuw i8, ptr %3, i64 752
  store <2 x double> <double 1.830000e+02, double 1.840000e+02>, ptr %52, align 8, !tbaa !6
  %53 = getelementptr inbounds nuw i8, ptr %3, i64 768
  store <2 x double> <double 1.850000e+02, double 1.860000e+02>, ptr %53, align 8, !tbaa !6
  %54 = getelementptr inbounds nuw i8, ptr %3, i64 784
  store <2 x double> <double 1.870000e+02, double 1.880000e+02>, ptr %54, align 8, !tbaa !6
  %55 = getelementptr inbounds nuw i8, ptr %3, i64 800
  store <2 x double> <double 1.890000e+02, double 1.900000e+02>, ptr %55, align 8, !tbaa !6
  %56 = getelementptr inbounds nuw i8, ptr %3, i64 816
  store <2 x double> <double 1.910000e+02, double 1.920000e+02>, ptr %56, align 8, !tbaa !6
  %57 = getelementptr inbounds nuw i8, ptr %3, i64 832
  store <2 x double> <double 1.930000e+02, double 1.940000e+02>, ptr %57, align 8, !tbaa !6
  %58 = getelementptr inbounds nuw i8, ptr %3, i64 848
  store <2 x double> <double 1.950000e+02, double 1.960000e+02>, ptr %58, align 8, !tbaa !6
  %59 = getelementptr inbounds nuw i8, ptr %3, i64 864
  store <2 x double> <double 1.970000e+02, double 1.980000e+02>, ptr %59, align 8, !tbaa !6
  %60 = getelementptr inbounds nuw i8, ptr %3, i64 880
  store double 1.990000e+02, ptr %60, align 8, !tbaa !6
  %61 = getelementptr inbounds nuw i8, ptr %3, i64 888
  store i8 81, ptr %61, align 8, !tbaa !10
  %62 = getelementptr inbounds nuw i8, ptr %3, i64 892
  store float 1.000000e+02, ptr %62, align 4, !tbaa !13
  %63 = getelementptr inbounds nuw i8, ptr %3, i64 896
  store i8 81, ptr %63, align 8, !tbaa !10
  %64 = getelementptr inbounds nuw i8, ptr %3, i64 900
  store float 1.000000e+02, ptr %64, align 4, !tbaa !13
  %65 = getelementptr inbounds nuw i8, ptr %3, i64 904
  store i8 81, ptr %65, align 8, !tbaa !10
  %66 = getelementptr inbounds nuw i8, ptr %3, i64 908
  store float 1.000000e+02, ptr %66, align 4, !tbaa !13
  %67 = getelementptr inbounds nuw i8, ptr %3, i64 912
  store i8 81, ptr %67, align 8, !tbaa !10
  %68 = getelementptr inbounds nuw i8, ptr %3, i64 916
  store float 1.000000e+02, ptr %68, align 4, !tbaa !13
  %69 = getelementptr inbounds nuw i8, ptr %3, i64 920
  store i8 81, ptr %69, align 8, !tbaa !10
  %70 = getelementptr inbounds nuw i8, ptr %3, i64 924
  store float 1.000000e+02, ptr %70, align 4, !tbaa !13
  %71 = getelementptr inbounds nuw i8, ptr %3, i64 928
  store i8 81, ptr %71, align 8, !tbaa !10
  %72 = getelementptr inbounds nuw i8, ptr %3, i64 932
  store float 1.000000e+02, ptr %72, align 4, !tbaa !13
  %73 = getelementptr inbounds nuw i8, ptr %3, i64 936
  store i8 81, ptr %73, align 8, !tbaa !10
  %74 = getelementptr inbounds nuw i8, ptr %3, i64 940
  store float 1.000000e+02, ptr %74, align 4, !tbaa !13
  %75 = getelementptr inbounds nuw i8, ptr %3, i64 944
  store i8 81, ptr %75, align 8, !tbaa !10
  %76 = getelementptr inbounds nuw i8, ptr %3, i64 948
  store float 1.000000e+02, ptr %76, align 4, !tbaa !13
  %77 = getelementptr inbounds nuw i8, ptr %3, i64 952
  store i8 81, ptr %77, align 8, !tbaa !10
  %78 = getelementptr inbounds nuw i8, ptr %3, i64 956
  store float 1.000000e+02, ptr %78, align 4, !tbaa !13
  %79 = getelementptr inbounds nuw i8, ptr %3, i64 960
  store i8 81, ptr %79, align 8, !tbaa !10
  %80 = getelementptr inbounds nuw i8, ptr %3, i64 964
  store float 1.000000e+02, ptr %80, align 4, !tbaa !13
  %81 = call double @AddMixed(ptr noundef nonnull %3)
  %82 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, double noundef %81)
  call void @InitializeMixed(ptr noundef nonnull %4, i32 noundef 200)
  %83 = call double @AddMixed(ptr noundef nonnull %4)
  %84 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef 0, double noundef %83)
  %85 = getelementptr inbounds nuw i8, ptr %4, i64 968
  call void @InitializeMixed(ptr noundef nonnull %85, i32 noundef 300)
  %86 = call double @AddMixed(ptr noundef nonnull %85)
  %87 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef 1, double noundef %86)
  %88 = getelementptr inbounds nuw i8, ptr %4, i64 1936
  call void @InitializeMixed(ptr noundef nonnull %88, i32 noundef 400)
  %89 = call double @AddMixed(ptr noundef nonnull %88)
  %90 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef 2, double noundef %89)
  %91 = getelementptr inbounds nuw i8, ptr %4, i64 2904
  call void @InitializeMixed(ptr noundef nonnull %91, i32 noundef 500)
  %92 = call double @AddMixed(ptr noundef nonnull %91)
  %93 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef 3, double noundef %92)
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #5
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #5
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #4

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"double", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!11, !8, i64 0}
!11 = !{!"Flat_struct", !8, i64 0, !12, i64 4}
!12 = !{!"float", !8, i64 0}
!13 = !{!11, !12, i64 4}
