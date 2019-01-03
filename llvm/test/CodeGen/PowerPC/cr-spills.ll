; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7

target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; This test case triggers several functions related to cr spilling, both in
; frame lowering and to handle cr register pressure. When the register kill
; flags were not being set correctly, this would cause the register scavenger to
; assert.

@SetupFastFullPelSearch.orig_pels = external unnamed_addr global [768 x i16], align 2
@weight_luma = external global i32
@offset_luma = external global i32
@wp_luma_round = external global i32, align 4
@luma_log_weight_denom = external global i32, align 4

define void @SetupFastFullPelSearch() #0 {
entry:
  %mul10 = mul nsw i32 undef, undef
  br i1 undef, label %land.end, label %land.lhs.true

land.lhs.true:                                    ; preds = %entry
  switch i32 0, label %land.end [
    i32 0, label %land.rhs
    i32 3, label %land.rhs
  ]

land.rhs:                                         ; preds = %land.lhs.true, %land.lhs.true
  %tobool21 = icmp ne i32 undef, 0
  br label %land.end

land.end:                                         ; preds = %land.rhs, %land.lhs.true, %entry
  %0 = phi i1 [ %tobool21, %land.rhs ], [ false, %land.lhs.true ], [ false, %entry ]
  %cond = load i32*, i32** undef, align 8
  br i1 undef, label %if.then95, label %for.body.lr.ph

if.then95:                                        ; preds = %land.end
  %cmp.i4.i1427 = icmp slt i32 undef, undef
  br label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %if.then95, %land.end
  br label %for.body

for.body:                                         ; preds = %for.body, %for.body.lr.ph
  br i1 undef, label %for.body, label %for.body252

for.body252:                                      ; preds = %for.inc997, %for.body
  %shl263 = add i32 undef, 80
  br i1 %0, label %for.cond286.preheader, label %for.cond713.preheader

for.cond286.preheader:                            ; preds = %for.body252
  br label %for.cond290.preheader

for.cond290.preheader:                            ; preds = %for.end520, %for.cond286.preheader
  %srcptr.31595 = phi i16* [ getelementptr inbounds ([768 x i16], [768 x i16]* @SetupFastFullPelSearch.orig_pels, i64 0, i64 0), %for.cond286.preheader ], [ null, %for.end520 ]
  %1 = load i32, i32* undef, align 4
  %2 = load i32, i32* @weight_luma, align 4
  %3 = load i32, i32* @wp_luma_round, align 4
  %4 = load i32, i32* @luma_log_weight_denom, align 4
  %5 = load i32, i32* @offset_luma, align 4
  %incdec.ptr502.sum = add i64 undef, 16
  br label %for.body293

for.body293:                                      ; preds = %for.body293, %for.cond290.preheader
  %srcptr.41591 = phi i16* [ %srcptr.31595, %for.cond290.preheader ], [ undef, %for.body293 ]
  %refptr.11590 = phi i16* [ undef, %for.cond290.preheader ], [ %add.ptr517, %for.body293 ]
  %LineSadBlk0.01588 = phi i32 [ 0, %for.cond290.preheader ], [ %add346, %for.body293 ]
  %LineSadBlk1.01587 = phi i32 [ 0, %for.cond290.preheader ], [ %add402, %for.body293 ]
  %LineSadBlk3.01586 = phi i32 [ 0, %for.cond290.preheader ], [ %add514, %for.body293 ]
  %LineSadBlk2.01585 = phi i32 [ 0, %for.cond290.preheader ], [ %add458, %for.body293 ]
  %6 = load i16, i16* %refptr.11590, align 2
  %conv294 = zext i16 %6 to i32
  %mul295 = mul nsw i32 %conv294, %2
  %add296 = add nsw i32 %mul295, %3
  %shr = ashr i32 %add296, %4
  %add297 = add nsw i32 %shr, %5
  %cmp.i.i1513 = icmp sgt i32 %add297, 0
  %cond.i.i1514 = select i1 %cmp.i.i1513, i32 %add297, i32 0
  %cmp.i4.i1515 = icmp slt i32 %cond.i.i1514, %1
  %cond.i5.i1516 = select i1 %cmp.i4.i1515, i32 %cond.i.i1514, i32 %1
  %7 = load i16, i16* %srcptr.41591, align 2
  %conv300 = zext i16 %7 to i32
  %sub301 = sub nsw i32 %cond.i5.i1516, %conv300
  %idxprom302 = sext i32 %sub301 to i64
  %arrayidx303 = getelementptr inbounds i32, i32* %cond, i64 %idxprom302
  %8 = load i32, i32* %arrayidx303, align 4
  %add304 = add nsw i32 %8, %LineSadBlk0.01588
  %9 = load i32, i32* undef, align 4
  %add318 = add nsw i32 %add304, %9
  %10 = load i16, i16* undef, align 2
  %conv321 = zext i16 %10 to i32
  %mul322 = mul nsw i32 %conv321, %2
  %add323 = add nsw i32 %mul322, %3
  %shr324 = ashr i32 %add323, %4
  %add325 = add nsw i32 %shr324, %5
  %cmp.i.i1505 = icmp sgt i32 %add325, 0
  %cond.i.i1506 = select i1 %cmp.i.i1505, i32 %add325, i32 0
  %cmp.i4.i1507 = icmp slt i32 %cond.i.i1506, %1
  %cond.i5.i1508 = select i1 %cmp.i4.i1507, i32 %cond.i.i1506, i32 %1
  %sub329 = sub nsw i32 %cond.i5.i1508, 0
  %idxprom330 = sext i32 %sub329 to i64
  %arrayidx331 = getelementptr inbounds i32, i32* %cond, i64 %idxprom330
  %11 = load i32, i32* %arrayidx331, align 4
  %add332 = add nsw i32 %add318, %11
  %cmp.i.i1501 = icmp sgt i32 undef, 0
  %cond.i.i1502 = select i1 %cmp.i.i1501, i32 undef, i32 0
  %cmp.i4.i1503 = icmp slt i32 %cond.i.i1502, %1
  %cond.i5.i1504 = select i1 %cmp.i4.i1503, i32 %cond.i.i1502, i32 %1
  %incdec.ptr341 = getelementptr inbounds i16, i16* %srcptr.41591, i64 4
  %12 = load i16, i16* null, align 2
  %conv342 = zext i16 %12 to i32
  %sub343 = sub nsw i32 %cond.i5.i1504, %conv342
  %idxprom344 = sext i32 %sub343 to i64
  %arrayidx345 = getelementptr inbounds i32, i32* %cond, i64 %idxprom344
  %13 = load i32, i32* %arrayidx345, align 4
  %add346 = add nsw i32 %add332, %13
  %incdec.ptr348 = getelementptr inbounds i16, i16* %refptr.11590, i64 5
  %14 = load i16, i16* null, align 2
  %conv349 = zext i16 %14 to i32
  %mul350 = mul nsw i32 %conv349, %2
  %add351 = add nsw i32 %mul350, %3
  %shr352 = ashr i32 %add351, %4
  %add353 = add nsw i32 %shr352, %5
  %cmp.i.i1497 = icmp sgt i32 %add353, 0
  %cond.i.i1498 = select i1 %cmp.i.i1497, i32 %add353, i32 0
  %cmp.i4.i1499 = icmp slt i32 %cond.i.i1498, %1
  %cond.i5.i1500 = select i1 %cmp.i4.i1499, i32 %cond.i.i1498, i32 %1
  %incdec.ptr355 = getelementptr inbounds i16, i16* %srcptr.41591, i64 5
  %15 = load i16, i16* %incdec.ptr341, align 2
  %conv356 = zext i16 %15 to i32
  %sub357 = sub nsw i32 %cond.i5.i1500, %conv356
  %idxprom358 = sext i32 %sub357 to i64
  %arrayidx359 = getelementptr inbounds i32, i32* %cond, i64 %idxprom358
  %16 = load i32, i32* %arrayidx359, align 4
  %add360 = add nsw i32 %16, %LineSadBlk1.01587
  %incdec.ptr362 = getelementptr inbounds i16, i16* %refptr.11590, i64 6
  %17 = load i16, i16* %incdec.ptr348, align 2
  %conv363 = zext i16 %17 to i32
  %mul364 = mul nsw i32 %conv363, %2
  %add365 = add nsw i32 %mul364, %3
  %shr366 = ashr i32 %add365, %4
  %add367 = add nsw i32 %shr366, %5
  %cmp.i.i1493 = icmp sgt i32 %add367, 0
  %cond.i.i1494 = select i1 %cmp.i.i1493, i32 %add367, i32 0
  %cmp.i4.i1495 = icmp slt i32 %cond.i.i1494, %1
  %cond.i5.i1496 = select i1 %cmp.i4.i1495, i32 %cond.i.i1494, i32 %1
  %incdec.ptr369 = getelementptr inbounds i16, i16* %srcptr.41591, i64 6
  %18 = load i16, i16* %incdec.ptr355, align 2
  %conv370 = zext i16 %18 to i32
  %sub371 = sub nsw i32 %cond.i5.i1496, %conv370
  %idxprom372 = sext i32 %sub371 to i64
  %arrayidx373 = getelementptr inbounds i32, i32* %cond, i64 %idxprom372
  %19 = load i32, i32* %arrayidx373, align 4
  %add374 = add nsw i32 %add360, %19
  %incdec.ptr376 = getelementptr inbounds i16, i16* %refptr.11590, i64 7
  %20 = load i16, i16* %incdec.ptr362, align 2
  %conv377 = zext i16 %20 to i32
  %mul378 = mul nsw i32 %conv377, %2
  %add379 = add nsw i32 %mul378, %3
  %shr380 = ashr i32 %add379, %4
  %add381 = add nsw i32 %shr380, %5
  %cmp.i.i1489 = icmp sgt i32 %add381, 0
  %cond.i.i1490 = select i1 %cmp.i.i1489, i32 %add381, i32 0
  %cmp.i4.i1491 = icmp slt i32 %cond.i.i1490, %1
  %cond.i5.i1492 = select i1 %cmp.i4.i1491, i32 %cond.i.i1490, i32 %1
  %incdec.ptr383 = getelementptr inbounds i16, i16* %srcptr.41591, i64 7
  %21 = load i16, i16* %incdec.ptr369, align 2
  %conv384 = zext i16 %21 to i32
  %sub385 = sub nsw i32 %cond.i5.i1492, %conv384
  %idxprom386 = sext i32 %sub385 to i64
  %arrayidx387 = getelementptr inbounds i32, i32* %cond, i64 %idxprom386
  %22 = load i32, i32* %arrayidx387, align 4
  %add388 = add nsw i32 %add374, %22
  %23 = load i16, i16* %incdec.ptr376, align 2
  %conv391 = zext i16 %23 to i32
  %mul392 = mul nsw i32 %conv391, %2
  %add395 = add nsw i32 0, %5
  %cmp.i.i1485 = icmp sgt i32 %add395, 0
  %cond.i.i1486 = select i1 %cmp.i.i1485, i32 %add395, i32 0
  %cmp.i4.i1487 = icmp slt i32 %cond.i.i1486, %1
  %cond.i5.i1488 = select i1 %cmp.i4.i1487, i32 %cond.i.i1486, i32 %1
  %incdec.ptr397 = getelementptr inbounds i16, i16* %srcptr.41591, i64 8
  %24 = load i16, i16* %incdec.ptr383, align 2
  %conv398 = zext i16 %24 to i32
  %sub399 = sub nsw i32 %cond.i5.i1488, %conv398
  %idxprom400 = sext i32 %sub399 to i64
  %arrayidx401 = getelementptr inbounds i32, i32* %cond, i64 %idxprom400
  %25 = load i32, i32* %arrayidx401, align 4
  %add402 = add nsw i32 %add388, %25
  %incdec.ptr404 = getelementptr inbounds i16, i16* %refptr.11590, i64 9
  %cmp.i4.i1483 = icmp slt i32 undef, %1
  %cond.i5.i1484 = select i1 %cmp.i4.i1483, i32 undef, i32 %1
  %26 = load i16, i16* %incdec.ptr397, align 2
  %conv412 = zext i16 %26 to i32
  %sub413 = sub nsw i32 %cond.i5.i1484, %conv412
  %idxprom414 = sext i32 %sub413 to i64
  %arrayidx415 = getelementptr inbounds i32, i32* %cond, i64 %idxprom414
  %27 = load i32, i32* %arrayidx415, align 4
  %add416 = add nsw i32 %27, %LineSadBlk2.01585
  %incdec.ptr418 = getelementptr inbounds i16, i16* %refptr.11590, i64 10
  %28 = load i16, i16* %incdec.ptr404, align 2
  %conv419 = zext i16 %28 to i32
  %mul420 = mul nsw i32 %conv419, %2
  %add421 = add nsw i32 %mul420, %3
  %shr422 = ashr i32 %add421, %4
  %add423 = add nsw i32 %shr422, %5
  %cmp.i.i1477 = icmp sgt i32 %add423, 0
  %cond.i.i1478 = select i1 %cmp.i.i1477, i32 %add423, i32 0
  %cmp.i4.i1479 = icmp slt i32 %cond.i.i1478, %1
  %cond.i5.i1480 = select i1 %cmp.i4.i1479, i32 %cond.i.i1478, i32 %1
  %incdec.ptr425 = getelementptr inbounds i16, i16* %srcptr.41591, i64 10
  %sub427 = sub nsw i32 %cond.i5.i1480, 0
  %idxprom428 = sext i32 %sub427 to i64
  %arrayidx429 = getelementptr inbounds i32, i32* %cond, i64 %idxprom428
  %29 = load i32, i32* %arrayidx429, align 4
  %add430 = add nsw i32 %add416, %29
  %incdec.ptr432 = getelementptr inbounds i16, i16* %refptr.11590, i64 11
  %30 = load i16, i16* %incdec.ptr418, align 2
  %conv433 = zext i16 %30 to i32
  %mul434 = mul nsw i32 %conv433, %2
  %add435 = add nsw i32 %mul434, %3
  %shr436 = ashr i32 %add435, %4
  %add437 = add nsw i32 %shr436, %5
  %cmp.i.i1473 = icmp sgt i32 %add437, 0
  %cond.i.i1474 = select i1 %cmp.i.i1473, i32 %add437, i32 0
  %cmp.i4.i1475 = icmp slt i32 %cond.i.i1474, %1
  %cond.i5.i1476 = select i1 %cmp.i4.i1475, i32 %cond.i.i1474, i32 %1
  %31 = load i16, i16* %incdec.ptr425, align 2
  %conv440 = zext i16 %31 to i32
  %sub441 = sub nsw i32 %cond.i5.i1476, %conv440
  %idxprom442 = sext i32 %sub441 to i64
  %arrayidx443 = getelementptr inbounds i32, i32* %cond, i64 %idxprom442
  %32 = load i32, i32* %arrayidx443, align 4
  %add444 = add nsw i32 %add430, %32
  %incdec.ptr446 = getelementptr inbounds i16, i16* %refptr.11590, i64 12
  %33 = load i16, i16* %incdec.ptr432, align 2
  %conv447 = zext i16 %33 to i32
  %mul448 = mul nsw i32 %conv447, %2
  %add449 = add nsw i32 %mul448, %3
  %shr450 = ashr i32 %add449, %4
  %add451 = add nsw i32 %shr450, %5
  %cmp.i.i1469 = icmp sgt i32 %add451, 0
  %cond.i.i1470 = select i1 %cmp.i.i1469, i32 %add451, i32 0
  %cmp.i4.i1471 = icmp slt i32 %cond.i.i1470, %1
  %cond.i5.i1472 = select i1 %cmp.i4.i1471, i32 %cond.i.i1470, i32 %1
  %incdec.ptr453 = getelementptr inbounds i16, i16* %srcptr.41591, i64 12
  %34 = load i16, i16* undef, align 2
  %conv454 = zext i16 %34 to i32
  %sub455 = sub nsw i32 %cond.i5.i1472, %conv454
  %idxprom456 = sext i32 %sub455 to i64
  %arrayidx457 = getelementptr inbounds i32, i32* %cond, i64 %idxprom456
  %35 = load i32, i32* %arrayidx457, align 4
  %add458 = add nsw i32 %add444, %35
  %incdec.ptr460 = getelementptr inbounds i16, i16* %refptr.11590, i64 13
  %36 = load i16, i16* %incdec.ptr446, align 2
  %conv461 = zext i16 %36 to i32
  %mul462 = mul nsw i32 %conv461, %2
  %add463 = add nsw i32 %mul462, %3
  %shr464 = ashr i32 %add463, %4
  %add465 = add nsw i32 %shr464, %5
  %cmp.i.i1465 = icmp sgt i32 %add465, 0
  %cond.i.i1466 = select i1 %cmp.i.i1465, i32 %add465, i32 0
  %cmp.i4.i1467 = icmp slt i32 %cond.i.i1466, %1
  %cond.i5.i1468 = select i1 %cmp.i4.i1467, i32 %cond.i.i1466, i32 %1
  %incdec.ptr467 = getelementptr inbounds i16, i16* %srcptr.41591, i64 13
  %37 = load i16, i16* %incdec.ptr453, align 2
  %conv468 = zext i16 %37 to i32
  %sub469 = sub nsw i32 %cond.i5.i1468, %conv468
  %idxprom470 = sext i32 %sub469 to i64
  %arrayidx471 = getelementptr inbounds i32, i32* %cond, i64 %idxprom470
  %38 = load i32, i32* %arrayidx471, align 4
  %add472 = add nsw i32 %38, %LineSadBlk3.01586
  %incdec.ptr474 = getelementptr inbounds i16, i16* %refptr.11590, i64 14
  %add477 = add nsw i32 0, %3
  %shr478 = ashr i32 %add477, %4
  %add479 = add nsw i32 %shr478, %5
  %cmp.i.i1461 = icmp sgt i32 %add479, 0
  %cond.i.i1462 = select i1 %cmp.i.i1461, i32 %add479, i32 0
  %cmp.i4.i1463 = icmp slt i32 %cond.i.i1462, %1
  %cond.i5.i1464 = select i1 %cmp.i4.i1463, i32 %cond.i.i1462, i32 %1
  %incdec.ptr481 = getelementptr inbounds i16, i16* %srcptr.41591, i64 14
  %39 = load i16, i16* %incdec.ptr467, align 2
  %conv482 = zext i16 %39 to i32
  %sub483 = sub nsw i32 %cond.i5.i1464, %conv482
  %idxprom484 = sext i32 %sub483 to i64
  %arrayidx485 = getelementptr inbounds i32, i32* %cond, i64 %idxprom484
  %40 = load i32, i32* %arrayidx485, align 4
  %add486 = add nsw i32 %add472, %40
  %incdec.ptr488 = getelementptr inbounds i16, i16* %refptr.11590, i64 15
  %41 = load i16, i16* %incdec.ptr474, align 2
  %conv489 = zext i16 %41 to i32
  %mul490 = mul nsw i32 %conv489, %2
  %add491 = add nsw i32 %mul490, %3
  %shr492 = ashr i32 %add491, %4
  %add493 = add nsw i32 %shr492, %5
  %cmp.i.i1457 = icmp sgt i32 %add493, 0
  %cond.i.i1458 = select i1 %cmp.i.i1457, i32 %add493, i32 0
  %cmp.i4.i1459 = icmp slt i32 %cond.i.i1458, %1
  %cond.i5.i1460 = select i1 %cmp.i4.i1459, i32 %cond.i.i1458, i32 %1
  %incdec.ptr495 = getelementptr inbounds i16, i16* %srcptr.41591, i64 15
  %42 = load i16, i16* %incdec.ptr481, align 2
  %conv496 = zext i16 %42 to i32
  %sub497 = sub nsw i32 %cond.i5.i1460, %conv496
  %idxprom498 = sext i32 %sub497 to i64
  %arrayidx499 = getelementptr inbounds i32, i32* %cond, i64 %idxprom498
  %43 = load i32, i32* %arrayidx499, align 4
  %add500 = add nsw i32 %add486, %43
  %44 = load i16, i16* %incdec.ptr488, align 2
  %conv503 = zext i16 %44 to i32
  %mul504 = mul nsw i32 %conv503, %2
  %add505 = add nsw i32 %mul504, %3
  %shr506 = ashr i32 %add505, %4
  %add507 = add nsw i32 %shr506, %5
  %cmp.i.i1453 = icmp sgt i32 %add507, 0
  %cond.i.i1454 = select i1 %cmp.i.i1453, i32 %add507, i32 0
  %cmp.i4.i1455 = icmp slt i32 %cond.i.i1454, %1
  %cond.i5.i1456 = select i1 %cmp.i4.i1455, i32 %cond.i.i1454, i32 %1
  %45 = load i16, i16* %incdec.ptr495, align 2
  %conv510 = zext i16 %45 to i32
  %sub511 = sub nsw i32 %cond.i5.i1456, %conv510
  %idxprom512 = sext i32 %sub511 to i64
  %arrayidx513 = getelementptr inbounds i32, i32* %cond, i64 %idxprom512
  %46 = load i32, i32* %arrayidx513, align 4
  %add514 = add nsw i32 %add500, %46
  %add.ptr517 = getelementptr inbounds i16, i16* %refptr.11590, i64 %incdec.ptr502.sum
  %exitcond1692 = icmp eq i32 undef, 4
  br i1 %exitcond1692, label %for.end520, label %for.body293

for.end520:                                       ; preds = %for.body293
  store i32 %add346, i32* undef, align 4
  store i32 %add402, i32* undef, align 4
  store i32 %add458, i32* undef, align 4
  store i32 %add514, i32* null, align 4
  br i1 undef, label %for.end543, label %for.cond290.preheader

for.end543:                                       ; preds = %for.end520
  br i1 undef, label %for.inc997, label %for.body549

for.body549:                                      ; preds = %for.inc701, %for.end543
  %call554 = call i16* null(i16**** null, i32 signext undef, i32 signext %shl263) #1
  br label %for.cond559.preheader

for.cond559.preheader:                            ; preds = %for.cond559.preheader, %for.body549
  br i1 undef, label %for.inc701, label %for.cond559.preheader

for.inc701:                                       ; preds = %for.cond559.preheader
  br i1 undef, label %for.inc997, label %for.body549

for.cond713.preheader:                            ; preds = %for.end850, %for.body252
  br label %for.body716

for.body716:                                      ; preds = %for.body716, %for.cond713.preheader
  br i1 undef, label %for.end850, label %for.body716

for.end850:                                       ; preds = %for.body716
  br i1 undef, label %for.end873, label %for.cond713.preheader

for.end873:                                       ; preds = %for.end850
  br i1 undef, label %for.inc997, label %for.body879

for.body879:                                      ; preds = %for.inc992, %for.end873
  br label %for.cond889.preheader

for.cond889.preheader:                            ; preds = %for.end964, %for.body879
  br i1 undef, label %for.cond894.preheader.lr.ph, label %for.end964

for.cond894.preheader.lr.ph:                      ; preds = %for.cond889.preheader
  br label %for.body898.lr.ph.us

for.end957.us:                                    ; preds = %for.body946.us
  br i1 undef, label %for.body898.lr.ph.us, label %for.end964

for.body946.us:                                   ; preds = %for.body930.us, %for.body946.us
  br i1 false, label %for.body946.us, label %for.end957.us

for.body930.us:                                   ; preds = %for.body914.us, %for.body930.us
  br i1 undef, label %for.body930.us, label %for.body946.us

for.body914.us:                                   ; preds = %for.body898.us, %for.body914.us
  br i1 undef, label %for.body914.us, label %for.body930.us

for.body898.us:                                   ; preds = %for.body898.lr.ph.us, %for.body898.us
  br i1 undef, label %for.body898.us, label %for.body914.us

for.body898.lr.ph.us:                             ; preds = %for.end957.us, %for.cond894.preheader.lr.ph
  br label %for.body898.us

for.end964:                                       ; preds = %for.end957.us, %for.cond889.preheader
  %inc990 = add nsw i32 undef, 1
  br i1 false, label %for.inc992, label %for.cond889.preheader

for.inc992:                                       ; preds = %for.end964
  br i1 false, label %for.inc997, label %for.body879

for.inc997:                                       ; preds = %for.inc992, %for.end873, %for.inc701, %for.end543
  %cmp250 = icmp slt i32 undef, %mul10
  br i1 %cmp250, label %for.body252, label %for.end999

for.end999:                                       ; preds = %for.inc997
  ret void
}

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }
