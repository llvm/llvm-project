target datalayout = "e-p:64:64:64-S128-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f16:16:16-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-f128:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"
; RUN: opt < %s -basicaa -bb-vectorize -bb-vectorize-req-chain-depth=2 -instcombine -gvn -S | FileCheck %s
; The chain length is set to 2 so that this will do some vectorization; check that the order of the functions is unchanged.

%struct.descriptor_dimension = type { i64, i64, i64 }
%struct.__st_parameter_common = type { i32, i32, i8*, i32, i32, i8*, i32* }
%struct.__st_parameter_dt = type { %struct.__st_parameter_common, i64, i64*, i64*, i8*, i8*, i32, i32, i8*, i8*, i32, i32, i8*, [256 x i8], i32*, i64, i8*, i32, i32, i8*, i8*, i32, i32, i8*, i8*, i32, i32, i8*, i8*, i32, [4 x i8] }
%"struct.array4_real(kind=4)" = type { i8*, i64, i64, [4 x %struct.descriptor_dimension] }
%"struct.array4_integer(kind=4).73" = type { i8*, i64, i64, [4 x %struct.descriptor_dimension] }
%struct.array4_unknown = type { i8*, i64, i64, [4 x %struct.descriptor_dimension] }

@.cst4 = external unnamed_addr constant [11 x i8], align 8
@.cst823 = external unnamed_addr constant [214 x i8], align 64
@j.4580 = external global i32
@j1.4581 = external global i32
@nty1.4590 = external global [2 x i8]
@nty2.4591 = external global [2 x i8]
@xr1.4592 = external global float
@xr2.4593 = external global float
@yr1.4594 = external global float
@yr2.4595 = external global float

@__main1_MOD_iave = external unnamed_addr global i32
@__main1_MOD_igrp = external global i32
@__main1_MOD_iounit = external global i32
@__main1_MOD_ityp = external global i32
@__main1_MOD_mclmsg = external unnamed_addr global %struct.array4_unknown, align 32
@__main1_MOD_mxdate = external unnamed_addr global %"struct.array4_integer(kind=4).73", align 32
@__main1_MOD_rmxval = external unnamed_addr global %"struct.array4_real(kind=4)", align 32

declare void @_gfortran_st_write(%struct.__st_parameter_dt*)
declare void @_gfortran_st_write_done(%struct.__st_parameter_dt*)
declare void @_gfortran_transfer_character_write(%struct.__st_parameter_dt*, i8*, i32)
declare void @_gfortran_transfer_integer_write(%struct.__st_parameter_dt*, i8*, i32)
declare void @_gfortran_transfer_real_write(%struct.__st_parameter_dt*, i8*, i32)

define i1 @"prtmax__<bb 3>_<bb 34>"(%struct.__st_parameter_dt* %memtmp3, i32 %D.4627_188.reload) nounwind {
; CHECK: prtmax__
newFuncRoot:
  br label %"<bb 34>"

codeRepl80.exitStub:                              ; preds = %"<bb 34>"
  ret i1 true

"<bb 34>.<bb 25>_crit_edge.exitStub":             ; preds = %"<bb 34>"
  ret i1 false

"<bb 34>":                                        ; preds = %newFuncRoot
  %tmp128 = getelementptr inbounds %struct.__st_parameter_dt* %memtmp3, i32 0, i32 0
  %tmp129 = getelementptr inbounds %struct.__st_parameter_common* %tmp128, i32 0, i32 2
  store i8* getelementptr inbounds ([11 x i8]* @.cst4, i64 0, i64 0), i8** %tmp129, align 8
  %tmp130 = getelementptr inbounds %struct.__st_parameter_dt* %memtmp3, i32 0, i32 0
  %tmp131 = getelementptr inbounds %struct.__st_parameter_common* %tmp130, i32 0, i32 3
  store i32 31495, i32* %tmp131, align 4
  %tmp132 = getelementptr inbounds %struct.__st_parameter_dt* %memtmp3, i32 0, i32 5
  store i8* getelementptr inbounds ([214 x i8]* @.cst823, i64 0, i64 0), i8** %tmp132, align 8
  %tmp133 = getelementptr inbounds %struct.__st_parameter_dt* %memtmp3, i32 0, i32 6
  store i32 214, i32* %tmp133, align 4
  %tmp134 = getelementptr inbounds %struct.__st_parameter_dt* %memtmp3, i32 0, i32 0
  %tmp135 = getelementptr inbounds %struct.__st_parameter_common* %tmp134, i32 0, i32 0
  store i32 4096, i32* %tmp135, align 4
  %iounit.8748_288 = load i32* @__main1_MOD_iounit, align 4
  %tmp136 = getelementptr inbounds %struct.__st_parameter_dt* %memtmp3, i32 0, i32 0
  %tmp137 = getelementptr inbounds %struct.__st_parameter_common* %tmp136, i32 0, i32 1
  store i32 %iounit.8748_288, i32* %tmp137, align 4
  call void @_gfortran_st_write(%struct.__st_parameter_dt* %memtmp3) nounwind
  call void bitcast (void (%struct.__st_parameter_dt*, i8*, i32)* @_gfortran_transfer_integer_write to void (%struct.__st_parameter_dt*, i32*, i32)*)(%struct.__st_parameter_dt* %memtmp3, i32* @j.4580, i32 4) nounwind
; CHECK: @_gfortran_transfer_integer_write
  %D.75807_289 = load i8** getelementptr inbounds (%"struct.array4_real(kind=4)"* @__main1_MOD_rmxval, i64 0, i32 0), align 8
  %j.8758_290 = load i32* @j.4580, align 4
  %D.75760_291 = sext i32 %j.8758_290 to i64
  %iave.8736_292 = load i32* @__main1_MOD_iave, align 4
  %D.75620_293 = sext i32 %iave.8736_292 to i64
  %D.75808_294 = load i64* getelementptr inbounds (%"struct.array4_real(kind=4)"* @__main1_MOD_rmxval, i64 0, i32 3, i64 2, i32 0), align 8
  %D.75809_295 = mul nsw i64 %D.75620_293, %D.75808_294
  %igrp.8737_296 = load i32* @__main1_MOD_igrp, align 4
  %D.75635_297 = sext i32 %igrp.8737_296 to i64
  %D.75810_298 = load i64* getelementptr inbounds (%"struct.array4_real(kind=4)"* @__main1_MOD_rmxval, i64 0, i32 3, i64 1, i32 0), align 8
  %D.75811_299 = mul nsw i64 %D.75635_297, %D.75810_298
  %D.75812_300 = add nsw i64 %D.75809_295, %D.75811_299
  %D.75813_301 = add nsw i64 %D.75760_291, %D.75812_300
  %ityp.8750_302 = load i32* @__main1_MOD_ityp, align 4
  %D.75704_303 = sext i32 %ityp.8750_302 to i64
  %D.75814_304 = load i64* getelementptr inbounds (%"struct.array4_real(kind=4)"* @__main1_MOD_rmxval, i64 0, i32 3, i64 3, i32 0), align 8
  %D.75815_305 = mul nsw i64 %D.75704_303, %D.75814_304
  %D.75816_306 = add nsw i64 %D.75813_301, %D.75815_305
  %D.75817_307 = load i64* getelementptr inbounds (%"struct.array4_real(kind=4)"* @__main1_MOD_rmxval, i64 0, i32 1), align 8
  %D.75818_308 = add nsw i64 %D.75816_306, %D.75817_307
  %tmp138 = bitcast i8* %D.75807_289 to [0 x float]*
  %tmp139 = bitcast [0 x float]* %tmp138 to float*
  %D.75819_309 = getelementptr inbounds float* %tmp139, i64 %D.75818_308
  call void bitcast (void (%struct.__st_parameter_dt*, i8*, i32)* @_gfortran_transfer_real_write to void (%struct.__st_parameter_dt*, float*, i32)*)(%struct.__st_parameter_dt* %memtmp3, float* %D.75819_309, i32 4) nounwind
; CHECK: @_gfortran_transfer_real_write
  %D.75820_310 = load i8** getelementptr inbounds (%struct.array4_unknown* @__main1_MOD_mclmsg, i64 0, i32 0), align 8
  %j.8758_311 = load i32* @j.4580, align 4
  %D.75760_312 = sext i32 %j.8758_311 to i64
  %iave.8736_313 = load i32* @__main1_MOD_iave, align 4
  %D.75620_314 = sext i32 %iave.8736_313 to i64
  %D.75821_315 = load i64* getelementptr inbounds (%struct.array4_unknown* @__main1_MOD_mclmsg, i64 0, i32 3, i64 2, i32 0), align 8
  %D.75822_316 = mul nsw i64 %D.75620_314, %D.75821_315
  %igrp.8737_317 = load i32* @__main1_MOD_igrp, align 4
  %D.75635_318 = sext i32 %igrp.8737_317 to i64
  %D.75823_319 = load i64* getelementptr inbounds (%struct.array4_unknown* @__main1_MOD_mclmsg, i64 0, i32 3, i64 1, i32 0), align 8
  %D.75824_320 = mul nsw i64 %D.75635_318, %D.75823_319
  %D.75825_321 = add nsw i64 %D.75822_316, %D.75824_320
  %D.75826_322 = add nsw i64 %D.75760_312, %D.75825_321
  %ityp.8750_323 = load i32* @__main1_MOD_ityp, align 4
  %D.75704_324 = sext i32 %ityp.8750_323 to i64
  %D.75827_325 = load i64* getelementptr inbounds (%struct.array4_unknown* @__main1_MOD_mclmsg, i64 0, i32 3, i64 3, i32 0), align 8
  %D.75828_326 = mul nsw i64 %D.75704_324, %D.75827_325
  %D.75829_327 = add nsw i64 %D.75826_322, %D.75828_326
  %D.75830_328 = load i64* getelementptr inbounds (%struct.array4_unknown* @__main1_MOD_mclmsg, i64 0, i32 1), align 8
  %D.75831_329 = add nsw i64 %D.75829_327, %D.75830_328
  %tmp140 = bitcast i8* %D.75820_310 to [0 x [1 x i8]]*
  %tmp141 = bitcast [0 x [1 x i8]]* %tmp140 to [1 x i8]*
  %D.75832_330 = getelementptr inbounds [1 x i8]* %tmp141, i64 %D.75831_329
  call void bitcast (void (%struct.__st_parameter_dt*, i8*, i32)* @_gfortran_transfer_character_write to void (%struct.__st_parameter_dt*, [1 x i8]*, i32)*)(%struct.__st_parameter_dt* %memtmp3, [1 x i8]* %D.75832_330, i32 1) nounwind
; CHECK: @_gfortran_transfer_character_write
  %D.75833_331 = load i8** getelementptr inbounds (%"struct.array4_integer(kind=4).73"* @__main1_MOD_mxdate, i64 0, i32 0), align 8
  %j.8758_332 = load i32* @j.4580, align 4
  %D.75760_333 = sext i32 %j.8758_332 to i64
  %iave.8736_334 = load i32* @__main1_MOD_iave, align 4
  %D.75620_335 = sext i32 %iave.8736_334 to i64
  %D.75834_336 = load i64* getelementptr inbounds (%"struct.array4_integer(kind=4).73"* @__main1_MOD_mxdate, i64 0, i32 3, i64 2, i32 0), align 8
  %D.75835_337 = mul nsw i64 %D.75620_335, %D.75834_336
  %igrp.8737_338 = load i32* @__main1_MOD_igrp, align 4
  %D.75635_339 = sext i32 %igrp.8737_338 to i64
  %D.75836_340 = load i64* getelementptr inbounds (%"struct.array4_integer(kind=4).73"* @__main1_MOD_mxdate, i64 0, i32 3, i64 1, i32 0), align 8
  %D.75837_341 = mul nsw i64 %D.75635_339, %D.75836_340
  %D.75838_342 = add nsw i64 %D.75835_337, %D.75837_341
  %D.75839_343 = add nsw i64 %D.75760_333, %D.75838_342
  %ityp.8750_344 = load i32* @__main1_MOD_ityp, align 4
  %D.75704_345 = sext i32 %ityp.8750_344 to i64
  %D.75840_346 = load i64* getelementptr inbounds (%"struct.array4_integer(kind=4).73"* @__main1_MOD_mxdate, i64 0, i32 3, i64 3, i32 0), align 8
  %D.75841_347 = mul nsw i64 %D.75704_345, %D.75840_346
  %D.75842_348 = add nsw i64 %D.75839_343, %D.75841_347
  %D.75843_349 = load i64* getelementptr inbounds (%"struct.array4_integer(kind=4).73"* @__main1_MOD_mxdate, i64 0, i32 1), align 8
  %D.75844_350 = add nsw i64 %D.75842_348, %D.75843_349
  %tmp142 = bitcast i8* %D.75833_331 to [0 x i32]*
  %tmp143 = bitcast [0 x i32]* %tmp142 to i32*
  %D.75845_351 = getelementptr inbounds i32* %tmp143, i64 %D.75844_350
  call void bitcast (void (%struct.__st_parameter_dt*, i8*, i32)* @_gfortran_transfer_integer_write to void (%struct.__st_parameter_dt*, i32*, i32)*)(%struct.__st_parameter_dt* %memtmp3, i32* %D.75845_351, i32 4) nounwind
; CHECK: @_gfortran_transfer_integer_write
  call void bitcast (void (%struct.__st_parameter_dt*, i8*, i32)* @_gfortran_transfer_real_write to void (%struct.__st_parameter_dt*, float*, i32)*)(%struct.__st_parameter_dt* %memtmp3, float* @xr1.4592, i32 4) nounwind
; CHECK: @_gfortran_transfer_real_write
  call void bitcast (void (%struct.__st_parameter_dt*, i8*, i32)* @_gfortran_transfer_real_write to void (%struct.__st_parameter_dt*, float*, i32)*)(%struct.__st_parameter_dt* %memtmp3, float* @yr1.4594, i32 4) nounwind
; CHECK: @_gfortran_transfer_real_write
  call void bitcast (void (%struct.__st_parameter_dt*, i8*, i32)* @_gfortran_transfer_character_write to void (%struct.__st_parameter_dt*, [2 x i8]*, i32)*)(%struct.__st_parameter_dt* %memtmp3, [2 x i8]* @nty1.4590, i32 2) nounwind
; CHECK: @_gfortran_transfer_character_write
  call void bitcast (void (%struct.__st_parameter_dt*, i8*, i32)* @_gfortran_transfer_integer_write to void (%struct.__st_parameter_dt*, i32*, i32)*)(%struct.__st_parameter_dt* %memtmp3, i32* @j1.4581, i32 4) nounwind
; CHECK: @_gfortran_transfer_integer_write
  %D.75807_352 = load i8** getelementptr inbounds (%"struct.array4_real(kind=4)"* @__main1_MOD_rmxval, i64 0, i32 0), align 8
  %j1.8760_353 = load i32* @j1.4581, align 4
  %D.75773_354 = sext i32 %j1.8760_353 to i64
  %iave.8736_355 = load i32* @__main1_MOD_iave, align 4
  %D.75620_356 = sext i32 %iave.8736_355 to i64
  %D.75808_357 = load i64* getelementptr inbounds (%"struct.array4_real(kind=4)"* @__main1_MOD_rmxval, i64 0, i32 3, i64 2, i32 0), align 8
  %D.75809_358 = mul nsw i64 %D.75620_356, %D.75808_357
  %igrp.8737_359 = load i32* @__main1_MOD_igrp, align 4
  %D.75635_360 = sext i32 %igrp.8737_359 to i64
  %D.75810_361 = load i64* getelementptr inbounds (%"struct.array4_real(kind=4)"* @__main1_MOD_rmxval, i64 0, i32 3, i64 1, i32 0), align 8
  %D.75811_362 = mul nsw i64 %D.75635_360, %D.75810_361
  %D.75812_363 = add nsw i64 %D.75809_358, %D.75811_362
  %D.75846_364 = add nsw i64 %D.75773_354, %D.75812_363
  %ityp.8750_365 = load i32* @__main1_MOD_ityp, align 4
  %D.75704_366 = sext i32 %ityp.8750_365 to i64
  %D.75814_367 = load i64* getelementptr inbounds (%"struct.array4_real(kind=4)"* @__main1_MOD_rmxval, i64 0, i32 3, i64 3, i32 0), align 8
  %D.75815_368 = mul nsw i64 %D.75704_366, %D.75814_367
  %D.75847_369 = add nsw i64 %D.75846_364, %D.75815_368
  %D.75817_370 = load i64* getelementptr inbounds (%"struct.array4_real(kind=4)"* @__main1_MOD_rmxval, i64 0, i32 1), align 8
  %D.75848_371 = add nsw i64 %D.75847_369, %D.75817_370
  %tmp144 = bitcast i8* %D.75807_352 to [0 x float]*
  %tmp145 = bitcast [0 x float]* %tmp144 to float*
  %D.75849_372 = getelementptr inbounds float* %tmp145, i64 %D.75848_371
  call void bitcast (void (%struct.__st_parameter_dt*, i8*, i32)* @_gfortran_transfer_real_write to void (%struct.__st_parameter_dt*, float*, i32)*)(%struct.__st_parameter_dt* %memtmp3, float* %D.75849_372, i32 4) nounwind
; CHECK: @_gfortran_transfer_real_write
  %D.75820_373 = load i8** getelementptr inbounds (%struct.array4_unknown* @__main1_MOD_mclmsg, i64 0, i32 0), align 8
  %j1.8760_374 = load i32* @j1.4581, align 4
  %D.75773_375 = sext i32 %j1.8760_374 to i64
  %iave.8736_376 = load i32* @__main1_MOD_iave, align 4
  %D.75620_377 = sext i32 %iave.8736_376 to i64
  %D.75821_378 = load i64* getelementptr inbounds (%struct.array4_unknown* @__main1_MOD_mclmsg, i64 0, i32 3, i64 2, i32 0), align 8
  %D.75822_379 = mul nsw i64 %D.75620_377, %D.75821_378
  %igrp.8737_380 = load i32* @__main1_MOD_igrp, align 4
  %D.75635_381 = sext i32 %igrp.8737_380 to i64
  %D.75823_382 = load i64* getelementptr inbounds (%struct.array4_unknown* @__main1_MOD_mclmsg, i64 0, i32 3, i64 1, i32 0), align 8
  %D.75824_383 = mul nsw i64 %D.75635_381, %D.75823_382
  %D.75825_384 = add nsw i64 %D.75822_379, %D.75824_383
  %D.75850_385 = add nsw i64 %D.75773_375, %D.75825_384
  %ityp.8750_386 = load i32* @__main1_MOD_ityp, align 4
  %D.75704_387 = sext i32 %ityp.8750_386 to i64
  %D.75827_388 = load i64* getelementptr inbounds (%struct.array4_unknown* @__main1_MOD_mclmsg, i64 0, i32 3, i64 3, i32 0), align 8
  %D.75828_389 = mul nsw i64 %D.75704_387, %D.75827_388
  %D.75851_390 = add nsw i64 %D.75850_385, %D.75828_389
  %D.75830_391 = load i64* getelementptr inbounds (%struct.array4_unknown* @__main1_MOD_mclmsg, i64 0, i32 1), align 8
  %D.75852_392 = add nsw i64 %D.75851_390, %D.75830_391
  %tmp146 = bitcast i8* %D.75820_373 to [0 x [1 x i8]]*
  %tmp147 = bitcast [0 x [1 x i8]]* %tmp146 to [1 x i8]*
  %D.75853_393 = getelementptr inbounds [1 x i8]* %tmp147, i64 %D.75852_392
  call void bitcast (void (%struct.__st_parameter_dt*, i8*, i32)* @_gfortran_transfer_character_write to void (%struct.__st_parameter_dt*, [1 x i8]*, i32)*)(%struct.__st_parameter_dt* %memtmp3, [1 x i8]* %D.75853_393, i32 1) nounwind
; CHECK: @_gfortran_transfer_character_write
  %D.75833_394 = load i8** getelementptr inbounds (%"struct.array4_integer(kind=4).73"* @__main1_MOD_mxdate, i64 0, i32 0), align 8
  %j1.8760_395 = load i32* @j1.4581, align 4
  %D.75773_396 = sext i32 %j1.8760_395 to i64
  %iave.8736_397 = load i32* @__main1_MOD_iave, align 4
  %D.75620_398 = sext i32 %iave.8736_397 to i64
  %D.75834_399 = load i64* getelementptr inbounds (%"struct.array4_integer(kind=4).73"* @__main1_MOD_mxdate, i64 0, i32 3, i64 2, i32 0), align 8
  %D.75835_400 = mul nsw i64 %D.75620_398, %D.75834_399
  %igrp.8737_401 = load i32* @__main1_MOD_igrp, align 4
  %D.75635_402 = sext i32 %igrp.8737_401 to i64
  %D.75836_403 = load i64* getelementptr inbounds (%"struct.array4_integer(kind=4).73"* @__main1_MOD_mxdate, i64 0, i32 3, i64 1, i32 0), align 8
  %D.75837_404 = mul nsw i64 %D.75635_402, %D.75836_403
  %D.75838_405 = add nsw i64 %D.75835_400, %D.75837_404
  %D.75854_406 = add nsw i64 %D.75773_396, %D.75838_405
  %ityp.8750_407 = load i32* @__main1_MOD_ityp, align 4
  %D.75704_408 = sext i32 %ityp.8750_407 to i64
  %D.75840_409 = load i64* getelementptr inbounds (%"struct.array4_integer(kind=4).73"* @__main1_MOD_mxdate, i64 0, i32 3, i64 3, i32 0), align 8
  %D.75841_410 = mul nsw i64 %D.75704_408, %D.75840_409
  %D.75855_411 = add nsw i64 %D.75854_406, %D.75841_410
  %D.75843_412 = load i64* getelementptr inbounds (%"struct.array4_integer(kind=4).73"* @__main1_MOD_mxdate, i64 0, i32 1), align 8
  %D.75856_413 = add nsw i64 %D.75855_411, %D.75843_412
  %tmp148 = bitcast i8* %D.75833_394 to [0 x i32]*
  %tmp149 = bitcast [0 x i32]* %tmp148 to i32*
  %D.75857_414 = getelementptr inbounds i32* %tmp149, i64 %D.75856_413
  call void bitcast (void (%struct.__st_parameter_dt*, i8*, i32)* @_gfortran_transfer_integer_write to void (%struct.__st_parameter_dt*, i32*, i32)*)(%struct.__st_parameter_dt* %memtmp3, i32* %D.75857_414, i32 4) nounwind
; CHECK: @_gfortran_transfer_integer_write
  call void bitcast (void (%struct.__st_parameter_dt*, i8*, i32)* @_gfortran_transfer_real_write to void (%struct.__st_parameter_dt*, float*, i32)*)(%struct.__st_parameter_dt* %memtmp3, float* @xr2.4593, i32 4) nounwind
; CHECK: @_gfortran_transfer_real_write
  call void bitcast (void (%struct.__st_parameter_dt*, i8*, i32)* @_gfortran_transfer_real_write to void (%struct.__st_parameter_dt*, float*, i32)*)(%struct.__st_parameter_dt* %memtmp3, float* @yr2.4595, i32 4) nounwind
; CHECK: @_gfortran_transfer_real_write
  call void bitcast (void (%struct.__st_parameter_dt*, i8*, i32)* @_gfortran_transfer_character_write to void (%struct.__st_parameter_dt*, [2 x i8]*, i32)*)(%struct.__st_parameter_dt* %memtmp3, [2 x i8]* @nty2.4591, i32 2) nounwind
; CHECK: @_gfortran_transfer_character_write
  call void @_gfortran_st_write_done(%struct.__st_parameter_dt* %memtmp3) nounwind
; CHECK: @_gfortran_st_write_done
  %j.8758_415 = load i32* @j.4580, align 4
  %D.4634_416 = icmp eq i32 %j.8758_415, %D.4627_188.reload
  %j.8758_417 = load i32* @j.4580, align 4
  %j.8770_418 = add nsw i32 %j.8758_417, 1
  store i32 %j.8770_418, i32* @j.4580, align 4
  %tmp150 = icmp ne i1 %D.4634_416, false
  br i1 %tmp150, label %codeRepl80.exitStub, label %"<bb 34>.<bb 25>_crit_edge.exitStub"
}

