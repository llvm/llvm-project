; RUN: llc -regalloc=greedy -relocation-model=pic  < %s 2>&1 | FileCheck %s
; Without the last chance recoloring, this test fails with:
; "ran out of registers".

; NOTE: With the fix to PR18883, we don't actually run out of registers here
; any more, and so those checks are disabled. This test remains only for general coverage.
; XXX: not llc -regalloc=greedy -relocation-model=pic -lcr-max-depth=0  < %s 2>&1 | FileCheck %s --check-prefix=CHECK-DEPTH
; Test whether failure due to cutoff for depth is reported

; XXX: not llc -regalloc=greedy -relocation-model=pic -lcr-max-interf=1  < %s 2>&1 | FileCheck %s --check-prefix=CHECK-INTERF
; Test whether failure due to cutoff for interference is reported

; RUN: llc -regalloc=greedy -relocation-model=pic -lcr-max-interf=1 -lcr-max-depth=0 -exhaustive-register-search < %s > %t 2>&1
; RUN: FileCheck --input-file=%t %s --check-prefix=CHECK-EXHAUSTIVE
; Test whether exhaustive-register-search can bypass the depth and interference cutoffs of last chance recoloring 

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-n8:16:32-S128"
target triple = "i386-apple-macosx"

@fp_dh_36985b17790d59a27994eaab5dcb00ee = external constant [499 x i32]
@fp_dh_18716afa4a5354de0a302c8edb3b0ee1 = external global i32
@fp_dh_20a33cdeefab8f4c8887e82766cb9dcb = external global ptr
@fp_dh_9d93c897906e39883c58b034c8e786b2 = external global [5419648 x i8], align 16

; Function Attrs: nounwind ssp
; CHECK-NOT: ran out of registers during register allocation
; CHECK-INTERF: error: register allocation failed: maximum interference for recoloring reached
; CHECK-DEPTH: error: register allocation failed: maximum depth for recoloring reached
; CHECK-EXHAUSTIVE-NOT: error: register allocation failed: maximum {{depth|interference}} for recoloring reached
define void @fp_dh_f870bf31fd8ffe068450366e3f05389a(ptr %arg) #0 {
bb:
  indirectbr ptr undef, [label %bb85, label %bb206]

bb85:                                             ; preds = %bb222, %bb85, %bb
  store ptr blockaddress(@fp_dh_f870bf31fd8ffe068450366e3f05389a, %bb206), ptr undef, align 4
  indirectbr ptr undef, [label %bb439, label %bb85]

bb206:                                            ; preds = %bb
  %tmp = getelementptr [499 x i32], ptr @fp_dh_36985b17790d59a27994eaab5dcb00ee, i32 0, i32 undef
  %tmp207 = load i32, ptr %tmp
  %tmp208 = add i32 %tmp207, 1
  %tmp209 = inttoptr i32 %tmp208 to ptr
  indirectbr ptr %tmp209, [label %bb213]

bb213:                                            ; preds = %bb206
  %tmp214 = load i32, ptr @fp_dh_18716afa4a5354de0a302c8edb3b0ee1, align 4
  %tmp215 = load ptr, ptr @fp_dh_20a33cdeefab8f4c8887e82766cb9dcb, align 4
  %tmp216 = urem i32 -717428541, %tmp214
  %tmp217 = getelementptr i8, ptr %tmp215, i32 %tmp216
  %tmp219 = load i32, ptr %tmp217, align 4
  store i32 %tmp219, ptr undef, align 4
  %tmp220 = select i1 false, i32 359373646, i32 1677237955
  %tmp221 = add i32 %tmp220, 0
  indirectbr ptr undef, [label %bb432, label %bb222]

bb222:                                            ; preds = %bb213
  %tmp224 = load i32, ptr undef, align 4
  %tmp225 = load i32, ptr undef, align 4
  %tmp226 = xor i32 %tmp225, %tmp224
  %tmp227 = shl i32 %tmp226, 1
  %tmp228 = and i32 %tmp227, -2048880334
  %tmp229 = sub i32 0, %tmp228
  %tmp230 = add i32 0, %tmp229
  %tmp231 = xor i32 %tmp230, 1059356227
  %tmp232 = mul i32 %tmp231, 1603744721
  %tmp233 = urem i32 %tmp232, 259
  %tmp234 = getelementptr [259 x i8], ptr getelementptr inbounds ([5419648 x i8], ptr @fp_dh_9d93c897906e39883c58b034c8e786b2, i32 0, i32 2039075), i32 0, i32 %tmp233
  %tmp235 = load i8, ptr %tmp234, align 1
  %tmp236 = add i32 %tmp233, 2
  %tmp237 = getelementptr [264 x i8], ptr getelementptr inbounds ([5419648 x i8], ptr @fp_dh_9d93c897906e39883c58b034c8e786b2, i32 0, i32 3388166), i32 0, i32 %tmp236
  %tmp238 = load i8, ptr %tmp237, align 1
  %tmp239 = getelementptr [265 x i8], ptr getelementptr inbounds ([5419648 x i8], ptr @fp_dh_9d93c897906e39883c58b034c8e786b2, i32 0, i32 1325165), i32 0, i32 0
  %tmp240 = load i8, ptr %tmp239, align 1
  %tmp241 = add i32 %tmp233, 6
  %tmp242 = trunc i32 %tmp241 to i8
  %tmp243 = mul i8 %tmp242, -3
  %tmp244 = add i8 %tmp243, 3
  %tmp245 = mul i8 %tmp242, -6
  %tmp246 = and i8 %tmp245, 6
  %tmp247 = sub i8 0, %tmp246
  %tmp248 = add i8 %tmp244, %tmp247
  %tmp249 = load i8, ptr undef, align 1
  %tmp250 = xor i8 %tmp235, 17
  %tmp251 = xor i8 %tmp250, %tmp238
  %tmp252 = xor i8 %tmp251, %tmp240
  %tmp253 = xor i8 %tmp252, %tmp249
  %tmp254 = xor i8 %tmp253, %tmp248
  %tmp255 = zext i8 %tmp254 to i16
  %tmp256 = shl nuw i16 %tmp255, 8
  %tmp257 = load i8, ptr null, align 1
  %tmp258 = load i32, ptr @fp_dh_18716afa4a5354de0a302c8edb3b0ee1, align 4
  %tmp259 = load ptr, ptr @fp_dh_20a33cdeefab8f4c8887e82766cb9dcb, align 4
  %tmp260 = urem i32 -717428541, %tmp258
  %tmp261 = getelementptr i8, ptr %tmp259, i32 %tmp260
  %tmp263 = load i32, ptr %tmp261, align 4
  %tmp264 = xor i32 %tmp263, 0
  %tmp265 = shl i32 %tmp264, 1
  %tmp266 = and i32 %tmp265, -1312119832
  %tmp267 = sub i32 0, %tmp266
  %tmp268 = add i32 0, %tmp267
  %tmp269 = xor i32 %tmp268, 623994670
  %tmp270 = mul i32 %tmp269, 1603744721
  %tmp271 = urem i32 %tmp270, 259
  %tmp274 = add i32 %tmp271, 3
  %tmp275 = getelementptr [265 x i8], ptr getelementptr inbounds ([5419648 x i8], ptr @fp_dh_9d93c897906e39883c58b034c8e786b2, i32 0, i32 1325165), i32 0, i32 %tmp274
  %tmp276 = load i8, ptr %tmp275, align 1
  %tmp277 = add i32 %tmp271, 6
  %tmp278 = trunc i32 %tmp277 to i8
  %tmp279 = mul i8 %tmp278, -3
  %tmp280 = add i8 %tmp279, 31
  %tmp281 = add i8 %tmp280, 0
  %tmp282 = xor i8 %tmp257, 13
  %tmp283 = xor i8 %tmp282, 0
  %tmp284 = xor i8 %tmp283, 0
  %tmp285 = xor i8 %tmp284, %tmp276
  %tmp286 = xor i8 %tmp285, %tmp281
  %tmp287 = zext i8 %tmp286 to i16
  %tmp288 = or i16 %tmp287, %tmp256
  %tmp289 = xor i16 %tmp288, 14330
  %tmp290 = add i16 0, %tmp289
  %tmp291 = add i16 %tmp290, -14330
  %tmp292 = zext i16 %tmp291 to i32
  %tmp293 = add i16 %tmp290, -14330
  %tmp294 = lshr i16 %tmp293, 12
  %tmp295 = zext i16 %tmp294 to i32
  %tmp296 = sub i32 0, %tmp295
  %tmp297 = xor i32 %tmp296, 16
  %tmp298 = add i32 0, %tmp297
  %tmp299 = and i32 %tmp298, 31
  %tmp300 = and i32 %tmp292, 30864
  %tmp301 = shl i32 %tmp300, %tmp299
  %tmp302 = xor i32 0, %tmp301
  %tmp303 = add i32 0, %tmp302
  %tmp304 = and i32 %tmp298, 31
  %tmp305 = and i32 %tmp303, 25568
  %tmp306 = lshr i32 %tmp305, %tmp304
  %tmp307 = xor i32 0, %tmp306
  %tmp308 = add i32 0, %tmp307
  %tmp309 = trunc i32 %tmp308 to i16
  %tmp310 = shl i16 %tmp309, 1
  %tmp311 = and i16 %tmp310, -4648
  %tmp312 = shl i16 %tmp309, 1
  %tmp313 = and i16 %tmp312, 4646
  %tmp314 = xor i16 %tmp311, 17700
  %tmp315 = xor i16 %tmp313, 17700
  %tmp316 = add i16 %tmp314, %tmp315
  %tmp317 = and i16 %tmp314, %tmp315
  %tmp318 = shl nuw i16 %tmp317, 1
  %tmp319 = sub i16 0, %tmp318
  %tmp320 = add i16 %tmp316, %tmp319
  %tmp321 = and i16 %tmp320, 29906
  %tmp322 = xor i16 %tmp309, 14953
  %tmp323 = add i16 0, %tmp322
  %tmp324 = sub i16 0, %tmp321
  %tmp325 = xor i16 %tmp324, %tmp323
  %tmp326 = add i16 0, %tmp325
  %tmp327 = add i32 %tmp221, 1161362661
  %tmp333 = icmp eq i16 %tmp326, 14953
  %tmp334 = add i32 %tmp327, -1456704142
  %tmp335 = zext i1 %tmp333 to i32
  %tmp336 = add i32 %tmp334, %tmp335
  %tmp337 = getelementptr [499 x i32], ptr @fp_dh_36985b17790d59a27994eaab5dcb00ee, i32 0, i32 %tmp336
  %tmp338 = load i32, ptr %tmp337
  %tmp339 = add i32 %tmp338, 1
  %tmp340 = inttoptr i32 %tmp339 to ptr
  indirectbr ptr %tmp340, [label %bb85, label %bb439]

bb432:                                            ; preds = %bb432, %bb213
  %tmp433 = phi i32 [ %tmp221, %bb213 ], [ %tmp433, %bb432 ]
  %tmp434 = add i32 %tmp433, 1022523279
  %tmp435 = getelementptr [499 x i32], ptr @fp_dh_36985b17790d59a27994eaab5dcb00ee, i32 0, i32 %tmp434
  %tmp436 = load i32, ptr %tmp435
  %tmp437 = add i32 %tmp436, 1
  %tmp438 = inttoptr i32 %tmp437 to ptr
  indirectbr ptr %tmp438, [label %bb432]

bb439:                                            ; preds = %bb222, %bb85
  ret void
}

attributes #0 = { nounwind ssp "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
