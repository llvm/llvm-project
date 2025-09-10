; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/SignlessTypes/rem.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/SignlessTypes/rem.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@.str = private unnamed_addr constant [37 x i8] c"Test #%u, failed in iteration #: %u\0A\00", align 1
@.str.2 = private unnamed_addr constant [54 x i8] c"m=%u, x_u=%u, y_u=%u, z_u=%u, x_s=%d, y_s=%d, z_s=%d\0A\00", align 1
@str = private unnamed_addr constant [25 x i8] c"\0A *** REM test done! ***\00", align 4
@str.30 = private unnamed_addr constant [21 x i8] c"Failing test vector:\00", align 4

; Function Attrs: nofree norecurse nosync nounwind memory(none) uwtable
define dso_local i64 @gcd(i64 noundef %0, i64 noundef %1) local_unnamed_addr #0 {
  br label %3

3:                                                ; preds = %3, %2
  %4 = phi i64 [ %1, %2 ], [ %6, %3 ]
  %5 = phi i64 [ %0, %2 ], [ %4, %3 ]
  %6 = srem i64 %5, %4
  %7 = icmp eq i64 %6, 0
  br i1 %7, label %8, label %3

8:                                                ; preds = %3
  ret i64 %4
}

; Function Attrs: nounwind uwtable
define dso_local range(i32 0, 2) i32 @main(i32 noundef %0, ptr noundef readnone captures(none) %1) local_unnamed_addr #1 {
  br label %3

3:                                                ; preds = %2, %806
  %4 = phi i32 [ 0, %2 ], [ %807, %806 ]
  %5 = tail call i32 @rand() #6
  %6 = freeze i32 %5
  %7 = icmp eq i32 %6, 0
  br i1 %7, label %806, label %8

8:                                                ; preds = %3
  %9 = tail call i32 @rand() #6
  %10 = freeze i32 %9
  %11 = tail call i32 @rand() #6
  %12 = freeze i32 %11
  %13 = tail call i32 @rand() #6
  %14 = freeze i32 %13
  %15 = tail call i32 @rand() #6
  %16 = and i32 %15, 1
  %17 = icmp eq i32 %16, 0
  %18 = tail call i32 @rand() #6
  %19 = sub nsw i32 0, %18
  %20 = select i1 %17, i32 %19, i32 %18
  %21 = tail call i32 @rand() #6
  %22 = and i32 %21, 1
  %23 = icmp eq i32 %22, 0
  %24 = tail call i32 @rand() #6
  %25 = sub nsw i32 0, %24
  %26 = select i1 %23, i32 %25, i32 %24
  %27 = tail call i32 @rand() #6
  %28 = and i32 %27, 1
  %29 = icmp eq i32 %28, 0
  %30 = tail call i32 @rand() #6
  %31 = sub nsw i32 0, %30
  %32 = select i1 %29, i32 %31, i32 %30
  %33 = urem i32 %10, %6
  %34 = urem i32 %12, %6
  %35 = add i32 %34, %33
  %36 = icmp ult i32 %35, %33
  br i1 %36, label %48, label %37

37:                                               ; preds = %8
  %38 = add i32 %12, %10
  %39 = icmp ult i32 %38, %10
  br i1 %39, label %48, label %40

40:                                               ; preds = %37
  %41 = urem i32 %35, %6
  %42 = urem i32 %38, %6
  %43 = icmp eq i32 %41, %42
  br i1 %43, label %48, label %44

44:                                               ; preds = %40
  %45 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 5, i32 noundef %4)
  %46 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.30)
  %47 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef %6, i32 noundef %10, i32 noundef %12, i32 noundef %14, i32 noundef %20, i32 noundef %26, i32 noundef %32)
  br label %811

48:                                               ; preds = %40, %37, %8
  %49 = add i32 %14, %12
  %50 = icmp ult i32 %49, %12
  %51 = icmp ugt i32 %14, %10
  %52 = or i1 %50, %51
  br i1 %52, label %69, label %53

53:                                               ; preds = %48
  %54 = sub nuw i32 %10, %14
  %55 = urem i32 %49, %6
  %56 = icmp eq i32 %33, %55
  %57 = zext i1 %56 to i32
  %58 = icmp ugt i32 %6, %57
  %59 = select i1 %58, i32 0, i32 %6
  %60 = sub nuw i32 %57, %59
  %61 = urem i32 %54, %6
  %62 = icmp eq i32 %61, %34
  %63 = zext i1 %62 to i32
  %64 = icmp eq i32 %60, %63
  br i1 %64, label %69, label %65

65:                                               ; preds = %53
  %66 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 6, i32 noundef %4)
  %67 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.30)
  %68 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef %6, i32 noundef %10, i32 noundef %12, i32 noundef %14, i32 noundef %20, i32 noundef %26, i32 noundef %32)
  br label %811

69:                                               ; preds = %53, %48
  %70 = tail call { i32, i1 } @llvm.umul.with.overflow.i32(i32 %6, i32 %12)
  %71 = extractvalue { i32, i1 } %70, 1
  br i1 %71, label %83, label %72

72:                                               ; preds = %69
  %73 = extractvalue { i32, i1 } %70, 0
  %74 = add i32 %73, %10
  %75 = icmp ult i32 %74, %10
  br i1 %75, label %83, label %76

76:                                               ; preds = %72
  %77 = urem i32 %74, %6
  %78 = icmp eq i32 %33, %77
  br i1 %78, label %83, label %79

79:                                               ; preds = %76
  %80 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 7, i32 noundef %4)
  %81 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.30)
  %82 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef %6, i32 noundef %10, i32 noundef %12, i32 noundef %14, i32 noundef %20, i32 noundef %26, i32 noundef %32)
  br label %811

83:                                               ; preds = %76, %72, %69
  %84 = sext i32 %32 to i64
  %85 = zext i32 %6 to i64
  br label %86

86:                                               ; preds = %86, %83
  %87 = phi i64 [ %85, %83 ], [ %89, %86 ]
  %88 = phi i64 [ %84, %83 ], [ %87, %86 ]
  %89 = srem i64 %88, %87
  %90 = icmp eq i64 %89, 0
  br i1 %90, label %91, label %86

91:                                               ; preds = %86
  %92 = icmp ne i64 %87, 1
  %93 = icmp eq i32 %32, 0
  %94 = select i1 %92, i1 true, i1 %93
  br i1 %94, label %127, label %95

95:                                               ; preds = %91
  %96 = sext i32 %20 to i64
  br label %97

97:                                               ; preds = %97, %95
  %98 = phi i64 [ %84, %95 ], [ %100, %97 ]
  %99 = phi i64 [ %96, %95 ], [ %98, %97 ]
  %100 = srem i64 %99, %98
  %101 = icmp eq i64 %100, 0
  br i1 %101, label %102, label %97

102:                                              ; preds = %97
  %103 = icmp eq i64 %98, %84
  br i1 %103, label %104, label %127

104:                                              ; preds = %102
  %105 = sext i32 %26 to i64
  br label %106

106:                                              ; preds = %106, %104
  %107 = phi i64 [ %84, %104 ], [ %109, %106 ]
  %108 = phi i64 [ %105, %104 ], [ %107, %106 ]
  %109 = srem i64 %108, %107
  %110 = icmp eq i64 %109, 0
  br i1 %110, label %111, label %106

111:                                              ; preds = %106
  %112 = icmp eq i64 %107, %84
  br i1 %112, label %113, label %127

113:                                              ; preds = %111
  %114 = urem i32 %20, %6
  %115 = urem i32 %26, %6
  %116 = icmp eq i32 %114, %115
  %117 = sdiv i32 %20, %32
  %118 = urem i32 %117, %6
  %119 = sdiv i32 %26, %32
  %120 = urem i32 %119, %6
  %121 = icmp ne i32 %118, %120
  %122 = xor i1 %116, %121
  br i1 %122, label %127, label %123

123:                                              ; preds = %113
  %124 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 8, i32 noundef %4)
  %125 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.30)
  %126 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef %6, i32 noundef %10, i32 noundef %12, i32 noundef %14, i32 noundef %20, i32 noundef %26, i32 noundef %32)
  br label %811

127:                                              ; preds = %113, %111, %102, %91
  %128 = zext i32 %14 to i64
  br label %129

129:                                              ; preds = %129, %127
  %130 = phi i64 [ %85, %127 ], [ %132, %129 ]
  %131 = phi i64 [ %128, %127 ], [ %130, %129 ]
  %132 = urem i64 %131, %130
  %133 = icmp eq i64 %132, 0
  br i1 %133, label %134, label %129

134:                                              ; preds = %129
  %135 = icmp ne i64 %130, 1
  %136 = icmp eq i32 %14, 0
  %137 = or i1 %136, %135
  br i1 %137, label %138, label %139

138:                                              ; preds = %157, %155, %146, %134
  br label %171

139:                                              ; preds = %134
  %140 = sext i32 %20 to i64
  br label %141

141:                                              ; preds = %141, %139
  %142 = phi i64 [ %128, %139 ], [ %144, %141 ]
  %143 = phi i64 [ %140, %139 ], [ %142, %141 ]
  %144 = srem i64 %143, %142
  %145 = icmp eq i64 %144, 0
  br i1 %145, label %146, label %141

146:                                              ; preds = %141
  %147 = icmp eq i64 %142, %128
  br i1 %147, label %148, label %138

148:                                              ; preds = %146
  %149 = sext i32 %26 to i64
  br label %150

150:                                              ; preds = %150, %148
  %151 = phi i64 [ %128, %148 ], [ %153, %150 ]
  %152 = phi i64 [ %149, %148 ], [ %151, %150 ]
  %153 = srem i64 %152, %151
  %154 = icmp eq i64 %153, 0
  br i1 %154, label %155, label %150

155:                                              ; preds = %150
  %156 = icmp eq i64 %151, %128
  br i1 %156, label %157, label %138

157:                                              ; preds = %155
  %158 = urem i32 %20, %6
  %159 = urem i32 %26, %6
  %160 = icmp eq i32 %158, %159
  %161 = udiv i32 %20, %14
  %162 = urem i32 %161, %6
  %163 = udiv i32 %26, %14
  %164 = urem i32 %163, %6
  %165 = icmp ne i32 %162, %164
  %166 = xor i1 %160, %165
  br i1 %166, label %138, label %167

167:                                              ; preds = %157
  %168 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 9, i32 noundef %4)
  %169 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.30)
  %170 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef %6, i32 noundef %10, i32 noundef %12, i32 noundef %14, i32 noundef %20, i32 noundef %26, i32 noundef %32)
  br label %811

171:                                              ; preds = %138, %171
  %172 = phi i64 [ %174, %171 ], [ %85, %138 ]
  %173 = phi i64 [ %172, %171 ], [ %84, %138 ]
  %174 = srem i64 %173, %172
  %175 = icmp eq i64 %174, 0
  br i1 %175, label %176, label %171

176:                                              ; preds = %171
  %177 = icmp ne i64 %172, 1
  %178 = select i1 %177, i1 true, i1 %93
  br i1 %178, label %179, label %180

179:                                              ; preds = %198, %196, %187, %176
  br label %211

180:                                              ; preds = %176
  %181 = sext i32 %20 to i64
  br label %182

182:                                              ; preds = %182, %180
  %183 = phi i64 [ %84, %180 ], [ %185, %182 ]
  %184 = phi i64 [ %181, %180 ], [ %183, %182 ]
  %185 = srem i64 %184, %183
  %186 = icmp eq i64 %185, 0
  br i1 %186, label %187, label %182

187:                                              ; preds = %182
  %188 = icmp eq i64 %183, %84
  br i1 %188, label %189, label %179

189:                                              ; preds = %187
  %190 = zext i32 %12 to i64
  br label %191

191:                                              ; preds = %191, %189
  %192 = phi i64 [ %84, %189 ], [ %194, %191 ]
  %193 = phi i64 [ %190, %189 ], [ %192, %191 ]
  %194 = srem i64 %193, %192
  %195 = icmp eq i64 %194, 0
  br i1 %195, label %196, label %191

196:                                              ; preds = %191
  %197 = icmp eq i64 %192, %84
  br i1 %197, label %198, label %179

198:                                              ; preds = %196
  %199 = urem i32 %20, %6
  %200 = icmp eq i32 %199, %34
  %201 = sdiv i32 %20, %32
  %202 = urem i32 %201, %6
  %203 = udiv i32 %12, %32
  %204 = urem i32 %203, %6
  %205 = icmp ne i32 %202, %204
  %206 = xor i1 %200, %205
  br i1 %206, label %179, label %207

207:                                              ; preds = %198
  %208 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 10, i32 noundef %4)
  %209 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.30)
  %210 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef %6, i32 noundef %10, i32 noundef %12, i32 noundef %14, i32 noundef %20, i32 noundef %26, i32 noundef %32)
  br label %811

211:                                              ; preds = %179, %211
  %212 = phi i64 [ %214, %211 ], [ %85, %179 ]
  %213 = phi i64 [ %212, %211 ], [ %128, %179 ]
  %214 = urem i64 %213, %212
  %215 = icmp eq i64 %214, 0
  br i1 %215, label %216, label %211

216:                                              ; preds = %211
  %217 = icmp ne i64 %212, 1
  %218 = or i1 %136, %217
  br i1 %218, label %219, label %220

219:                                              ; preds = %238, %236, %227, %216
  br label %251

220:                                              ; preds = %216
  %221 = sext i32 %20 to i64
  br label %222

222:                                              ; preds = %222, %220
  %223 = phi i64 [ %128, %220 ], [ %225, %222 ]
  %224 = phi i64 [ %221, %220 ], [ %223, %222 ]
  %225 = srem i64 %224, %223
  %226 = icmp eq i64 %225, 0
  br i1 %226, label %227, label %222

227:                                              ; preds = %222
  %228 = icmp eq i64 %223, %128
  br i1 %228, label %229, label %219

229:                                              ; preds = %227
  %230 = zext i32 %12 to i64
  br label %231

231:                                              ; preds = %231, %229
  %232 = phi i64 [ %128, %229 ], [ %234, %231 ]
  %233 = phi i64 [ %230, %229 ], [ %232, %231 ]
  %234 = urem i64 %233, %232
  %235 = icmp eq i64 %234, 0
  br i1 %235, label %236, label %231

236:                                              ; preds = %231
  %237 = icmp eq i64 %232, %128
  br i1 %237, label %238, label %219

238:                                              ; preds = %236
  %239 = urem i32 %20, %6
  %240 = icmp eq i32 %239, %34
  %241 = udiv i32 %20, %14
  %242 = urem i32 %241, %6
  %243 = udiv i32 %12, %14
  %244 = urem i32 %243, %6
  %245 = icmp ne i32 %242, %244
  %246 = xor i1 %240, %245
  br i1 %246, label %219, label %247

247:                                              ; preds = %238
  %248 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 11, i32 noundef %4)
  %249 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.30)
  %250 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef %6, i32 noundef %10, i32 noundef %12, i32 noundef %14, i32 noundef %20, i32 noundef %26, i32 noundef %32)
  br label %811

251:                                              ; preds = %219, %251
  %252 = phi i64 [ %254, %251 ], [ %85, %219 ]
  %253 = phi i64 [ %252, %251 ], [ %84, %219 ]
  %254 = srem i64 %253, %252
  %255 = icmp eq i64 %254, 0
  br i1 %255, label %256, label %251

256:                                              ; preds = %251
  %257 = icmp ne i64 %252, 1
  %258 = select i1 %257, i1 true, i1 %93
  br i1 %258, label %259, label %260

259:                                              ; preds = %278, %276, %267, %256
  br label %291

260:                                              ; preds = %256
  %261 = zext i32 %10 to i64
  br label %262

262:                                              ; preds = %262, %260
  %263 = phi i64 [ %84, %260 ], [ %265, %262 ]
  %264 = phi i64 [ %261, %260 ], [ %263, %262 ]
  %265 = srem i64 %264, %263
  %266 = icmp eq i64 %265, 0
  br i1 %266, label %267, label %262

267:                                              ; preds = %262
  %268 = icmp eq i64 %263, %84
  br i1 %268, label %269, label %259

269:                                              ; preds = %267
  %270 = sext i32 %26 to i64
  br label %271

271:                                              ; preds = %271, %269
  %272 = phi i64 [ %84, %269 ], [ %274, %271 ]
  %273 = phi i64 [ %270, %269 ], [ %272, %271 ]
  %274 = srem i64 %273, %272
  %275 = icmp eq i64 %274, 0
  br i1 %275, label %276, label %271

276:                                              ; preds = %271
  %277 = icmp eq i64 %272, %84
  br i1 %277, label %278, label %259

278:                                              ; preds = %276
  %279 = urem i32 %26, %6
  %280 = icmp eq i32 %33, %279
  %281 = udiv i32 %10, %32
  %282 = urem i32 %281, %6
  %283 = sdiv i32 %26, %32
  %284 = urem i32 %283, %6
  %285 = icmp ne i32 %282, %284
  %286 = xor i1 %280, %285
  br i1 %286, label %259, label %287

287:                                              ; preds = %278
  %288 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 12, i32 noundef %4)
  %289 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.30)
  %290 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef %6, i32 noundef %10, i32 noundef %12, i32 noundef %14, i32 noundef %20, i32 noundef %26, i32 noundef %32)
  br label %811

291:                                              ; preds = %259, %291
  %292 = phi i64 [ %294, %291 ], [ %85, %259 ]
  %293 = phi i64 [ %292, %291 ], [ %128, %259 ]
  %294 = urem i64 %293, %292
  %295 = icmp eq i64 %294, 0
  br i1 %295, label %296, label %291

296:                                              ; preds = %291
  %297 = icmp ne i64 %292, 1
  %298 = or i1 %136, %297
  br i1 %298, label %299, label %300

299:                                              ; preds = %318, %316, %307, %296
  br label %331

300:                                              ; preds = %296
  %301 = zext i32 %10 to i64
  br label %302

302:                                              ; preds = %302, %300
  %303 = phi i64 [ %128, %300 ], [ %305, %302 ]
  %304 = phi i64 [ %301, %300 ], [ %303, %302 ]
  %305 = urem i64 %304, %303
  %306 = icmp eq i64 %305, 0
  br i1 %306, label %307, label %302

307:                                              ; preds = %302
  %308 = icmp eq i64 %303, %128
  br i1 %308, label %309, label %299

309:                                              ; preds = %307
  %310 = sext i32 %26 to i64
  br label %311

311:                                              ; preds = %311, %309
  %312 = phi i64 [ %128, %309 ], [ %314, %311 ]
  %313 = phi i64 [ %310, %309 ], [ %312, %311 ]
  %314 = srem i64 %313, %312
  %315 = icmp eq i64 %314, 0
  br i1 %315, label %316, label %311

316:                                              ; preds = %311
  %317 = icmp eq i64 %312, %128
  br i1 %317, label %318, label %299

318:                                              ; preds = %316
  %319 = urem i32 %26, %6
  %320 = icmp eq i32 %33, %319
  %321 = udiv i32 %10, %14
  %322 = urem i32 %321, %6
  %323 = udiv i32 %26, %14
  %324 = urem i32 %323, %6
  %325 = icmp ne i32 %322, %324
  %326 = xor i1 %320, %325
  br i1 %326, label %299, label %327

327:                                              ; preds = %318
  %328 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 13, i32 noundef %4)
  %329 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.30)
  %330 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef %6, i32 noundef %10, i32 noundef %12, i32 noundef %14, i32 noundef %20, i32 noundef %26, i32 noundef %32)
  br label %811

331:                                              ; preds = %299, %331
  %332 = phi i64 [ %334, %331 ], [ %85, %299 ]
  %333 = phi i64 [ %332, %331 ], [ %84, %299 ]
  %334 = srem i64 %333, %332
  %335 = icmp eq i64 %334, 0
  br i1 %335, label %336, label %331

336:                                              ; preds = %331
  %337 = icmp ne i64 %332, 1
  %338 = select i1 %337, i1 true, i1 %93
  br i1 %338, label %339, label %340

339:                                              ; preds = %358, %356, %347, %336
  br label %370

340:                                              ; preds = %336
  %341 = zext i32 %10 to i64
  br label %342

342:                                              ; preds = %342, %340
  %343 = phi i64 [ %84, %340 ], [ %345, %342 ]
  %344 = phi i64 [ %341, %340 ], [ %343, %342 ]
  %345 = srem i64 %344, %343
  %346 = icmp eq i64 %345, 0
  br i1 %346, label %347, label %342

347:                                              ; preds = %342
  %348 = icmp eq i64 %343, %84
  br i1 %348, label %349, label %339

349:                                              ; preds = %347
  %350 = zext i32 %12 to i64
  br label %351

351:                                              ; preds = %351, %349
  %352 = phi i64 [ %84, %349 ], [ %354, %351 ]
  %353 = phi i64 [ %350, %349 ], [ %352, %351 ]
  %354 = srem i64 %353, %352
  %355 = icmp eq i64 %354, 0
  br i1 %355, label %356, label %351

356:                                              ; preds = %351
  %357 = icmp eq i64 %352, %84
  br i1 %357, label %358, label %339

358:                                              ; preds = %356
  %359 = icmp eq i32 %33, %34
  %360 = udiv i32 %10, %32
  %361 = urem i32 %360, %6
  %362 = udiv i32 %12, %32
  %363 = urem i32 %362, %6
  %364 = icmp ne i32 %361, %363
  %365 = xor i1 %359, %364
  br i1 %365, label %339, label %366

366:                                              ; preds = %358
  %367 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 14, i32 noundef %4)
  %368 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.30)
  %369 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef %6, i32 noundef %10, i32 noundef %12, i32 noundef %14, i32 noundef %20, i32 noundef %26, i32 noundef %32)
  br label %811

370:                                              ; preds = %339, %370
  %371 = phi i64 [ %373, %370 ], [ %85, %339 ]
  %372 = phi i64 [ %371, %370 ], [ %128, %339 ]
  %373 = urem i64 %372, %371
  %374 = icmp eq i64 %373, 0
  br i1 %374, label %375, label %370

375:                                              ; preds = %370
  %376 = icmp ne i64 %371, 1
  %377 = or i1 %136, %376
  br i1 %377, label %408, label %378

378:                                              ; preds = %375
  %379 = zext i32 %10 to i64
  br label %380

380:                                              ; preds = %380, %378
  %381 = phi i64 [ %128, %378 ], [ %383, %380 ]
  %382 = phi i64 [ %379, %378 ], [ %381, %380 ]
  %383 = urem i64 %382, %381
  %384 = icmp eq i64 %383, 0
  br i1 %384, label %385, label %380

385:                                              ; preds = %380
  %386 = icmp eq i64 %381, %128
  br i1 %386, label %387, label %408

387:                                              ; preds = %385
  %388 = zext i32 %12 to i64
  br label %389

389:                                              ; preds = %389, %387
  %390 = phi i64 [ %128, %387 ], [ %392, %389 ]
  %391 = phi i64 [ %388, %387 ], [ %390, %389 ]
  %392 = urem i64 %391, %390
  %393 = icmp eq i64 %392, 0
  br i1 %393, label %394, label %389

394:                                              ; preds = %389
  %395 = icmp eq i64 %390, %128
  br i1 %395, label %396, label %408

396:                                              ; preds = %394
  %397 = icmp eq i32 %33, %34
  %398 = udiv i32 %10, %14
  %399 = urem i32 %398, %6
  %400 = udiv i32 %12, %14
  %401 = urem i32 %400, %6
  %402 = icmp ne i32 %399, %401
  %403 = xor i1 %397, %402
  br i1 %403, label %408, label %404

404:                                              ; preds = %396
  %405 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 15, i32 noundef %4)
  %406 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.30)
  %407 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef %6, i32 noundef %10, i32 noundef %12, i32 noundef %14, i32 noundef %20, i32 noundef %26, i32 noundef %32)
  br label %811

408:                                              ; preds = %396, %394, %385, %375
  br i1 %93, label %449, label %409

409:                                              ; preds = %408
  %410 = sext i32 %20 to i64
  br label %411

411:                                              ; preds = %411, %409
  %412 = phi i64 [ %84, %409 ], [ %414, %411 ]
  %413 = phi i64 [ %410, %409 ], [ %412, %411 ]
  %414 = srem i64 %413, %412
  %415 = icmp eq i64 %414, 0
  br i1 %415, label %416, label %411

416:                                              ; preds = %411
  %417 = icmp eq i64 %412, %84
  br i1 %417, label %418, label %449

418:                                              ; preds = %416
  %419 = sext i32 %26 to i64
  br label %420

420:                                              ; preds = %420, %418
  %421 = phi i64 [ %84, %418 ], [ %423, %420 ]
  %422 = phi i64 [ %419, %418 ], [ %421, %420 ]
  %423 = srem i64 %422, %421
  %424 = icmp eq i64 %423, 0
  br i1 %424, label %425, label %420

425:                                              ; preds = %420
  %426 = icmp eq i64 %421, %84
  br i1 %426, label %427, label %449

427:                                              ; preds = %425, %427
  %428 = phi i64 [ %430, %427 ], [ %84, %425 ]
  %429 = phi i64 [ %428, %427 ], [ %85, %425 ]
  %430 = srem i64 %429, %428
  %431 = icmp eq i64 %430, 0
  br i1 %431, label %432, label %427

432:                                              ; preds = %427
  %433 = icmp eq i64 %428, %84
  br i1 %433, label %434, label %449

434:                                              ; preds = %432
  %435 = urem i32 %20, %6
  %436 = urem i32 %26, %6
  %437 = icmp eq i32 %435, %436
  %438 = sdiv i32 %20, %32
  %439 = udiv i32 %6, %32
  %440 = urem i32 %438, %439
  %441 = sdiv i32 %26, %32
  %442 = urem i32 %441, %439
  %443 = icmp ne i32 %440, %442
  %444 = xor i1 %437, %443
  br i1 %444, label %449, label %445

445:                                              ; preds = %434
  %446 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 16, i32 noundef %4)
  %447 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.30)
  %448 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef %6, i32 noundef %10, i32 noundef %12, i32 noundef %14, i32 noundef %20, i32 noundef %26, i32 noundef %32)
  br label %811

449:                                              ; preds = %434, %432, %425, %416, %408
  br i1 %136, label %490, label %450

450:                                              ; preds = %449
  %451 = sext i32 %20 to i64
  br label %452

452:                                              ; preds = %452, %450
  %453 = phi i64 [ %128, %450 ], [ %455, %452 ]
  %454 = phi i64 [ %451, %450 ], [ %453, %452 ]
  %455 = srem i64 %454, %453
  %456 = icmp eq i64 %455, 0
  br i1 %456, label %457, label %452

457:                                              ; preds = %452
  %458 = icmp eq i64 %453, %128
  br i1 %458, label %459, label %490

459:                                              ; preds = %457
  %460 = sext i32 %26 to i64
  br label %461

461:                                              ; preds = %461, %459
  %462 = phi i64 [ %128, %459 ], [ %464, %461 ]
  %463 = phi i64 [ %460, %459 ], [ %462, %461 ]
  %464 = srem i64 %463, %462
  %465 = icmp eq i64 %464, 0
  br i1 %465, label %466, label %461

466:                                              ; preds = %461
  %467 = icmp eq i64 %462, %128
  br i1 %467, label %468, label %490

468:                                              ; preds = %466, %468
  %469 = phi i64 [ %471, %468 ], [ %128, %466 ]
  %470 = phi i64 [ %469, %468 ], [ %85, %466 ]
  %471 = urem i64 %470, %469
  %472 = icmp eq i64 %471, 0
  br i1 %472, label %473, label %468

473:                                              ; preds = %468
  %474 = icmp eq i64 %469, %128
  br i1 %474, label %475, label %490

475:                                              ; preds = %473
  %476 = urem i32 %20, %6
  %477 = urem i32 %26, %6
  %478 = icmp eq i32 %476, %477
  %479 = udiv i32 %20, %14
  %480 = udiv i32 %6, %14
  %481 = urem i32 %479, %480
  %482 = udiv i32 %26, %14
  %483 = urem i32 %482, %480
  %484 = icmp ne i32 %481, %483
  %485 = xor i1 %478, %484
  br i1 %485, label %490, label %486

486:                                              ; preds = %475
  %487 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 17, i32 noundef %4)
  %488 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.30)
  %489 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef %6, i32 noundef %10, i32 noundef %12, i32 noundef %14, i32 noundef %20, i32 noundef %26, i32 noundef %32)
  br label %811

490:                                              ; preds = %475, %473, %466, %457, %449
  br i1 %93, label %530, label %491

491:                                              ; preds = %490
  %492 = sext i32 %20 to i64
  br label %493

493:                                              ; preds = %493, %491
  %494 = phi i64 [ %84, %491 ], [ %496, %493 ]
  %495 = phi i64 [ %492, %491 ], [ %494, %493 ]
  %496 = srem i64 %495, %494
  %497 = icmp eq i64 %496, 0
  br i1 %497, label %498, label %493

498:                                              ; preds = %493
  %499 = icmp eq i64 %494, %84
  br i1 %499, label %500, label %530

500:                                              ; preds = %498
  %501 = zext i32 %12 to i64
  br label %502

502:                                              ; preds = %502, %500
  %503 = phi i64 [ %84, %500 ], [ %505, %502 ]
  %504 = phi i64 [ %501, %500 ], [ %503, %502 ]
  %505 = srem i64 %504, %503
  %506 = icmp eq i64 %505, 0
  br i1 %506, label %507, label %502

507:                                              ; preds = %502
  %508 = icmp eq i64 %503, %84
  br i1 %508, label %509, label %530

509:                                              ; preds = %507, %509
  %510 = phi i64 [ %512, %509 ], [ %84, %507 ]
  %511 = phi i64 [ %510, %509 ], [ %85, %507 ]
  %512 = srem i64 %511, %510
  %513 = icmp eq i64 %512, 0
  br i1 %513, label %514, label %509

514:                                              ; preds = %509
  %515 = icmp eq i64 %510, %84
  br i1 %515, label %516, label %530

516:                                              ; preds = %514
  %517 = urem i32 %20, %6
  %518 = icmp eq i32 %517, %34
  %519 = sdiv i32 %20, %32
  %520 = udiv i32 %6, %32
  %521 = urem i32 %519, %520
  %522 = udiv i32 %12, %32
  %523 = urem i32 %522, %520
  %524 = icmp ne i32 %521, %523
  %525 = xor i1 %518, %524
  br i1 %525, label %530, label %526

526:                                              ; preds = %516
  %527 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 18, i32 noundef %4)
  %528 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.30)
  %529 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef %6, i32 noundef %10, i32 noundef %12, i32 noundef %14, i32 noundef %20, i32 noundef %26, i32 noundef %32)
  br label %811

530:                                              ; preds = %516, %514, %507, %498, %490
  br i1 %136, label %570, label %531

531:                                              ; preds = %530
  %532 = sext i32 %20 to i64
  br label %533

533:                                              ; preds = %533, %531
  %534 = phi i64 [ %128, %531 ], [ %536, %533 ]
  %535 = phi i64 [ %532, %531 ], [ %534, %533 ]
  %536 = srem i64 %535, %534
  %537 = icmp eq i64 %536, 0
  br i1 %537, label %538, label %533

538:                                              ; preds = %533
  %539 = icmp eq i64 %534, %128
  br i1 %539, label %540, label %570

540:                                              ; preds = %538
  %541 = zext i32 %12 to i64
  br label %542

542:                                              ; preds = %542, %540
  %543 = phi i64 [ %128, %540 ], [ %545, %542 ]
  %544 = phi i64 [ %541, %540 ], [ %543, %542 ]
  %545 = urem i64 %544, %543
  %546 = icmp eq i64 %545, 0
  br i1 %546, label %547, label %542

547:                                              ; preds = %542
  %548 = icmp eq i64 %543, %128
  br i1 %548, label %549, label %570

549:                                              ; preds = %547, %549
  %550 = phi i64 [ %552, %549 ], [ %128, %547 ]
  %551 = phi i64 [ %550, %549 ], [ %85, %547 ]
  %552 = urem i64 %551, %550
  %553 = icmp eq i64 %552, 0
  br i1 %553, label %554, label %549

554:                                              ; preds = %549
  %555 = icmp eq i64 %550, %128
  br i1 %555, label %556, label %570

556:                                              ; preds = %554
  %557 = urem i32 %20, %6
  %558 = icmp eq i32 %557, %34
  %559 = udiv i32 %20, %14
  %560 = udiv i32 %6, %14
  %561 = urem i32 %559, %560
  %562 = udiv i32 %12, %14
  %563 = urem i32 %562, %560
  %564 = icmp ne i32 %561, %563
  %565 = xor i1 %558, %564
  br i1 %565, label %570, label %566

566:                                              ; preds = %556
  %567 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 19, i32 noundef %4)
  %568 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.30)
  %569 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef %6, i32 noundef %10, i32 noundef %12, i32 noundef %14, i32 noundef %20, i32 noundef %26, i32 noundef %32)
  br label %811

570:                                              ; preds = %556, %554, %547, %538, %530
  br i1 %93, label %610, label %571

571:                                              ; preds = %570
  %572 = zext i32 %10 to i64
  br label %573

573:                                              ; preds = %573, %571
  %574 = phi i64 [ %84, %571 ], [ %576, %573 ]
  %575 = phi i64 [ %572, %571 ], [ %574, %573 ]
  %576 = srem i64 %575, %574
  %577 = icmp eq i64 %576, 0
  br i1 %577, label %578, label %573

578:                                              ; preds = %573
  %579 = icmp eq i64 %574, %84
  br i1 %579, label %580, label %610

580:                                              ; preds = %578
  %581 = sext i32 %26 to i64
  br label %582

582:                                              ; preds = %582, %580
  %583 = phi i64 [ %84, %580 ], [ %585, %582 ]
  %584 = phi i64 [ %581, %580 ], [ %583, %582 ]
  %585 = srem i64 %584, %583
  %586 = icmp eq i64 %585, 0
  br i1 %586, label %587, label %582

587:                                              ; preds = %582
  %588 = icmp eq i64 %583, %84
  br i1 %588, label %589, label %610

589:                                              ; preds = %587, %589
  %590 = phi i64 [ %592, %589 ], [ %84, %587 ]
  %591 = phi i64 [ %590, %589 ], [ %85, %587 ]
  %592 = srem i64 %591, %590
  %593 = icmp eq i64 %592, 0
  br i1 %593, label %594, label %589

594:                                              ; preds = %589
  %595 = icmp eq i64 %590, %84
  br i1 %595, label %596, label %610

596:                                              ; preds = %594
  %597 = urem i32 %26, %6
  %598 = icmp eq i32 %33, %597
  %599 = udiv i32 %10, %32
  %600 = udiv i32 %6, %32
  %601 = urem i32 %599, %600
  %602 = sdiv i32 %26, %32
  %603 = urem i32 %602, %600
  %604 = icmp ne i32 %601, %603
  %605 = xor i1 %598, %604
  br i1 %605, label %610, label %606

606:                                              ; preds = %596
  %607 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 20, i32 noundef %4)
  %608 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.30)
  %609 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef %6, i32 noundef %10, i32 noundef %12, i32 noundef %14, i32 noundef %20, i32 noundef %26, i32 noundef %32)
  br label %811

610:                                              ; preds = %596, %594, %587, %578, %570
  br i1 %136, label %650, label %611

611:                                              ; preds = %610
  %612 = zext i32 %10 to i64
  br label %613

613:                                              ; preds = %613, %611
  %614 = phi i64 [ %128, %611 ], [ %616, %613 ]
  %615 = phi i64 [ %612, %611 ], [ %614, %613 ]
  %616 = urem i64 %615, %614
  %617 = icmp eq i64 %616, 0
  br i1 %617, label %618, label %613

618:                                              ; preds = %613
  %619 = icmp eq i64 %614, %128
  br i1 %619, label %620, label %650

620:                                              ; preds = %618
  %621 = sext i32 %26 to i64
  br label %622

622:                                              ; preds = %622, %620
  %623 = phi i64 [ %128, %620 ], [ %625, %622 ]
  %624 = phi i64 [ %621, %620 ], [ %623, %622 ]
  %625 = srem i64 %624, %623
  %626 = icmp eq i64 %625, 0
  br i1 %626, label %627, label %622

627:                                              ; preds = %622
  %628 = icmp eq i64 %623, %128
  br i1 %628, label %629, label %650

629:                                              ; preds = %627, %629
  %630 = phi i64 [ %632, %629 ], [ %128, %627 ]
  %631 = phi i64 [ %630, %629 ], [ %85, %627 ]
  %632 = urem i64 %631, %630
  %633 = icmp eq i64 %632, 0
  br i1 %633, label %634, label %629

634:                                              ; preds = %629
  %635 = icmp eq i64 %630, %128
  br i1 %635, label %636, label %650

636:                                              ; preds = %634
  %637 = urem i32 %26, %6
  %638 = icmp eq i32 %33, %637
  %639 = udiv i32 %10, %14
  %640 = udiv i32 %6, %14
  %641 = urem i32 %639, %640
  %642 = udiv i32 %26, %14
  %643 = urem i32 %642, %640
  %644 = icmp ne i32 %641, %643
  %645 = xor i1 %638, %644
  br i1 %645, label %650, label %646

646:                                              ; preds = %636
  %647 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 21, i32 noundef %4)
  %648 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.30)
  %649 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef %6, i32 noundef %10, i32 noundef %12, i32 noundef %14, i32 noundef %20, i32 noundef %26, i32 noundef %32)
  br label %811

650:                                              ; preds = %636, %634, %627, %618, %610
  br i1 %93, label %689, label %651

651:                                              ; preds = %650
  %652 = zext i32 %10 to i64
  br label %653

653:                                              ; preds = %653, %651
  %654 = phi i64 [ %84, %651 ], [ %656, %653 ]
  %655 = phi i64 [ %652, %651 ], [ %654, %653 ]
  %656 = srem i64 %655, %654
  %657 = icmp eq i64 %656, 0
  br i1 %657, label %658, label %653

658:                                              ; preds = %653
  %659 = icmp eq i64 %654, %84
  br i1 %659, label %660, label %689

660:                                              ; preds = %658
  %661 = zext i32 %12 to i64
  br label %662

662:                                              ; preds = %662, %660
  %663 = phi i64 [ %84, %660 ], [ %665, %662 ]
  %664 = phi i64 [ %661, %660 ], [ %663, %662 ]
  %665 = srem i64 %664, %663
  %666 = icmp eq i64 %665, 0
  br i1 %666, label %667, label %662

667:                                              ; preds = %662
  %668 = icmp eq i64 %663, %84
  br i1 %668, label %669, label %689

669:                                              ; preds = %667, %669
  %670 = phi i64 [ %672, %669 ], [ %84, %667 ]
  %671 = phi i64 [ %670, %669 ], [ %85, %667 ]
  %672 = srem i64 %671, %670
  %673 = icmp eq i64 %672, 0
  br i1 %673, label %674, label %669

674:                                              ; preds = %669
  %675 = icmp eq i64 %670, %84
  br i1 %675, label %676, label %689

676:                                              ; preds = %674
  %677 = icmp eq i32 %33, %34
  %678 = udiv i32 %10, %32
  %679 = udiv i32 %6, %32
  %680 = urem i32 %678, %679
  %681 = udiv i32 %12, %32
  %682 = urem i32 %681, %679
  %683 = icmp ne i32 %680, %682
  %684 = xor i1 %677, %683
  br i1 %684, label %689, label %685

685:                                              ; preds = %676
  %686 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 22, i32 noundef %4)
  %687 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.30)
  %688 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef %6, i32 noundef %10, i32 noundef %12, i32 noundef %14, i32 noundef %20, i32 noundef %26, i32 noundef %32)
  br label %811

689:                                              ; preds = %676, %674, %667, %658, %650
  br i1 %136, label %804, label %690

690:                                              ; preds = %689
  %691 = zext i32 %10 to i64
  br label %692

692:                                              ; preds = %692, %690
  %693 = phi i64 [ %128, %690 ], [ %695, %692 ]
  %694 = phi i64 [ %691, %690 ], [ %693, %692 ]
  %695 = urem i64 %694, %693
  %696 = icmp eq i64 %695, 0
  br i1 %696, label %697, label %692

697:                                              ; preds = %692
  %698 = icmp eq i64 %693, %128
  br i1 %698, label %699, label %724

699:                                              ; preds = %697
  %700 = zext i32 %12 to i64
  br label %701

701:                                              ; preds = %701, %699
  %702 = phi i64 [ %128, %699 ], [ %704, %701 ]
  %703 = phi i64 [ %700, %699 ], [ %702, %701 ]
  %704 = urem i64 %703, %702
  %705 = icmp eq i64 %704, 0
  br i1 %705, label %706, label %701

706:                                              ; preds = %701
  %707 = icmp eq i64 %702, %128
  br i1 %707, label %708, label %724

708:                                              ; preds = %706, %708
  %709 = phi i64 [ %711, %708 ], [ %128, %706 ]
  %710 = phi i64 [ %709, %708 ], [ %85, %706 ]
  %711 = urem i64 %710, %709
  %712 = icmp eq i64 %711, 0
  br i1 %712, label %713, label %708

713:                                              ; preds = %708
  %714 = icmp eq i64 %709, %128
  br i1 %714, label %715, label %724

715:                                              ; preds = %713
  %716 = icmp eq i32 %33, %34
  %717 = udiv i32 %10, %14
  %718 = udiv i32 %6, %14
  %719 = urem i32 %717, %718
  %720 = udiv i32 %12, %14
  %721 = urem i32 %720, %718
  %722 = icmp ne i32 %719, %721
  %723 = xor i1 %716, %722
  br i1 %723, label %724, label %725

724:                                              ; preds = %697, %706, %713, %715
  br label %729

725:                                              ; preds = %715
  %726 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 23, i32 noundef %4)
  %727 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.30)
  %728 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef %6, i32 noundef %10, i32 noundef %12, i32 noundef %14, i32 noundef %20, i32 noundef %26, i32 noundef %32)
  br label %811

729:                                              ; preds = %724, %729
  %730 = phi i64 [ %732, %729 ], [ %128, %724 ]
  %731 = phi i64 [ %730, %729 ], [ %85, %724 ]
  %732 = urem i64 %731, %730
  %733 = icmp eq i64 %732, 0
  br i1 %733, label %734, label %729

734:                                              ; preds = %729
  %735 = icmp eq i64 %730, %128
  br i1 %735, label %736, label %744

736:                                              ; preds = %734
  %737 = urem i32 %20, %6
  %738 = urem i32 %26, %6
  %739 = icmp eq i32 %737, %738
  %740 = urem i32 %20, %14
  %741 = urem i32 %26, %14
  %742 = icmp ne i32 %740, %741
  %743 = xor i1 %739, %742
  br i1 %743, label %744, label %745

744:                                              ; preds = %734, %736
  br label %749

745:                                              ; preds = %736
  %746 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 25, i32 noundef %4)
  %747 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.30)
  %748 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef %6, i32 noundef %10, i32 noundef %12, i32 noundef %14, i32 noundef %20, i32 noundef %26, i32 noundef %32)
  br label %811

749:                                              ; preds = %744, %749
  %750 = phi i64 [ %752, %749 ], [ %128, %744 ]
  %751 = phi i64 [ %750, %749 ], [ %85, %744 ]
  %752 = urem i64 %751, %750
  %753 = icmp eq i64 %752, 0
  br i1 %753, label %754, label %749

754:                                              ; preds = %749
  %755 = icmp eq i64 %750, %128
  br i1 %755, label %756, label %763

756:                                              ; preds = %754
  %757 = urem i32 %20, %6
  %758 = icmp eq i32 %757, %34
  %759 = urem i32 %20, %14
  %760 = urem i32 %12, %14
  %761 = icmp ne i32 %759, %760
  %762 = xor i1 %758, %761
  br i1 %762, label %763, label %764

763:                                              ; preds = %754, %756
  br label %768

764:                                              ; preds = %756
  %765 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 27, i32 noundef %4)
  %766 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.30)
  %767 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef %6, i32 noundef %10, i32 noundef %12, i32 noundef %14, i32 noundef %20, i32 noundef %26, i32 noundef %32)
  br label %811

768:                                              ; preds = %763, %768
  %769 = phi i64 [ %771, %768 ], [ %128, %763 ]
  %770 = phi i64 [ %769, %768 ], [ %85, %763 ]
  %771 = urem i64 %770, %769
  %772 = icmp eq i64 %771, 0
  br i1 %772, label %773, label %768

773:                                              ; preds = %768
  %774 = icmp eq i64 %769, %128
  br i1 %774, label %775, label %782

775:                                              ; preds = %773
  %776 = urem i32 %26, %6
  %777 = icmp eq i32 %33, %776
  %778 = urem i32 %10, %14
  %779 = urem i32 %26, %14
  %780 = icmp ne i32 %778, %779
  %781 = xor i1 %777, %780
  br i1 %781, label %782, label %783

782:                                              ; preds = %773, %775
  br label %787

783:                                              ; preds = %775
  %784 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 29, i32 noundef %4)
  %785 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.30)
  %786 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef %6, i32 noundef %10, i32 noundef %12, i32 noundef %14, i32 noundef %20, i32 noundef %26, i32 noundef %32)
  br label %811

787:                                              ; preds = %782, %787
  %788 = phi i64 [ %790, %787 ], [ %128, %782 ]
  %789 = phi i64 [ %788, %787 ], [ %85, %782 ]
  %790 = urem i64 %789, %788
  %791 = icmp eq i64 %790, 0
  br i1 %791, label %792, label %787

792:                                              ; preds = %787
  %793 = icmp eq i64 %788, %128
  br i1 %793, label %794, label %804

794:                                              ; preds = %792
  %795 = icmp eq i32 %33, %34
  %796 = urem i32 %10, %14
  %797 = urem i32 %12, %14
  %798 = icmp ne i32 %796, %797
  %799 = xor i1 %795, %798
  br i1 %799, label %804, label %800

800:                                              ; preds = %794
  %801 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 31, i32 noundef %4)
  %802 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.30)
  %803 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef %6, i32 noundef %10, i32 noundef %12, i32 noundef %14, i32 noundef %20, i32 noundef %26, i32 noundef %32)
  br label %811

804:                                              ; preds = %689, %794, %792
  %805 = add nuw nsw i32 %4, 1
  br label %806

806:                                              ; preds = %3, %804
  %807 = phi i32 [ %805, %804 ], [ %4, %3 ]
  %808 = icmp ult i32 %807, 100
  br i1 %808, label %3, label %809, !llvm.loop !6

809:                                              ; preds = %806
  %810 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str)
  br label %811

811:                                              ; preds = %44, %65, %79, %123, %167, %207, %247, %287, %327, %366, %404, %445, %486, %526, %566, %606, %646, %685, %725, %745, %764, %783, %800, %809
  %812 = phi i32 [ 0, %809 ], [ 1, %800 ], [ 1, %783 ], [ 1, %764 ], [ 1, %745 ], [ 1, %725 ], [ 1, %685 ], [ 1, %646 ], [ 1, %606 ], [ 1, %566 ], [ 1, %526 ], [ 1, %486 ], [ 1, %445 ], [ 1, %404 ], [ 1, %366 ], [ 1, %327 ], [ 1, %287 ], [ 1, %247 ], [ 1, %207 ], [ 1, %167 ], [ 1, %123 ], [ 1, %79 ], [ 1, %65 ], [ 1, %44 ]
  ret i32 %812
}

; Function Attrs: nounwind
declare i32 @rand() local_unnamed_addr #2

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #3

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr noundef readonly captures(none)) local_unnamed_addr #4

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare { i32, i1 } @llvm.umul.with.overflow.i32(i32, i32) #5

attributes #0 = { nofree norecurse nosync nounwind memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nofree nounwind }
attributes #5 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #6 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.mustprogress"}
