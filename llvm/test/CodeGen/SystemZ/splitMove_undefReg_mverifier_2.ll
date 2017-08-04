; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 -verify-machineinstrs -disable-lsr | FileCheck %s
;
; Regression test for a machine verifier complaint discovered with llvm-stress.
; Test that splitting of a 128 bit store does not result in use of undef phys reg.
; This test case involved spilling of 128 bits, where the data operand was killed.

define void @autogen_SD15107(i8*, i32*, i64*, i32, i64, i8) {
; CHECK: .text
BB:
  %A4 = alloca double
  %A1 = alloca i32
  %L = load i8, i8* %0
  br label %CF331

CF331:                                            ; preds = %CF331, %BB
  %Shuff = shufflevector <8 x i8> zeroinitializer, <8 x i8> zeroinitializer, <8 x i32> <i32 undef, i32 undef, i32 1, i32 3, i32 5, i32 7, i32 9, i32 11>
  %L5 = load i8, i8* %0
  %FC9 = fptosi float 0xC59D259100000000 to i8
  %Shuff13 = shufflevector <8 x i64> zeroinitializer, <8 x i64> zeroinitializer, <8 x i32> <i32 10, i32 undef, i32 14, i32 0, i32 undef, i32 4, i32 6, i32 8>
  %Tr = trunc <8 x i16> zeroinitializer to <8 x i1>
  %Sl16 = select i1 true, i64 448097, i64 253977
  %E18 = extractelement <2 x i1> zeroinitializer, i32 1
  br i1 %E18, label %CF331, label %CF350

CF350:                                            ; preds = %CF331
  %Cmp22 = icmp slt i8 %L, -1
  br label %CF

CF:                                               ; preds = %CF333, %CF364, %CF, %CF350
  %Shuff25 = shufflevector <16 x i1> zeroinitializer, <16 x i1> zeroinitializer, <16 x i32> <i32 25, i32 27, i32 29, i32 31, i32 1, i32 undef, i32 undef, i32 7, i32 9, i32 11, i32 undef, i32 15, i32 17, i32 19, i32 21, i32 23>
  %B27 = mul <8 x i8> zeroinitializer, %Shuff
  %L31 = load i8, i8* %0
  store i8 %L5, i8* %0
  %E32 = extractelement <8 x i64> %Shuff13, i32 5
  %Sl37 = select i1 %E18, i64* %2, i64* %2
  %E40 = extractelement <8 x i64> %Shuff13, i32 4
  %I42 = insertelement <8 x i64> %Shuff13, i64 0, i32 1
  %Sl44 = select i1 true, double* %A4, double* %A4
  %L46 = load i64, i64* %Sl37
  br i1 undef, label %CF, label %CF335

CF335:                                            ; preds = %CF335, %CF
  %Shuff48 = shufflevector <8 x i16> zeroinitializer, <8 x i16> zeroinitializer, <8 x i32> <i32 undef, i32 15, i32 undef, i32 3, i32 5, i32 7, i32 9, i32 11>
  %B50 = sub <8 x i64> undef, zeroinitializer
  %Se = sext i1 %Cmp22 to i64
  %Cmp52 = icmp ule i64 %E40, 184653
  br i1 %Cmp52, label %CF335, label %CF364

CF364:                                            ; preds = %CF335
  store i64 %E32, i64* %Sl37
  %B57 = udiv <8 x i64> %I42, %B50
  %L61 = load i64, i64* %Sl37
  %Sl65 = select i1 undef, i1 %Cmp52, i1 true
  br i1 %Sl65, label %CF, label %CF333

CF333:                                            ; preds = %CF364
  %Cmp66 = fcmp uge float 0x474A237E00000000, undef
  br i1 %Cmp66, label %CF, label %CF324

CF324:                                            ; preds = %CF358, %CF360, %CF333
  %L67 = load i64, i64* %Sl37
  %Sl73 = select i1 %E18, i8 %L, i8 %L31
  %ZE = zext i1 true to i32
  %Cmp81 = icmp ult i64 184653, %L46
  br label %CF346

CF346:                                            ; preds = %CF363, %CF346, %CF324
  %L82 = load double, double* %Sl44
  store i64 %Se, i64* %Sl37
  br i1 undef, label %CF346, label %CF363

CF363:                                            ; preds = %CF346
  %I85 = insertelement <8 x i64> undef, i64 0, i32 4
  %Se86 = sext i1 %Cmp81 to i64
  %Cmp88 = icmp eq <16 x i1> zeroinitializer, undef
  %Shuff91 = shufflevector <8 x i64> %B57, <8 x i64> %I42, <8 x i32> <i32 1, i32 undef, i32 5, i32 7, i32 undef, i32 11, i32 13, i32 undef>
  %Sl95 = select i1 undef, i8 -1, i8 %5
  store i8 %FC9, i8* %0
  %Sl102 = select i1 %Sl65, float 0x3AAFABC380000000, float undef
  %L104 = load i64, i64* %Sl37
  store i8 %Sl95, i8* %0
  br i1 undef, label %CF346, label %CF360

CF360:                                            ; preds = %CF363
  %I107 = insertelement <16 x i1> undef, i1 %Sl65, i32 3
  %B108 = fdiv float undef, %Sl102
  %FC109 = sitofp <16 x i1> %Shuff25 to <16 x float>
  %Cmp111 = icmp slt i8 %Sl73, %Sl95
  br i1 %Cmp111, label %CF324, label %CF344

CF344:                                            ; preds = %CF344, %CF360
  store i64 %4, i64* %Sl37
  br i1 undef, label %CF344, label %CF358

CF358:                                            ; preds = %CF344
  %B116 = add i8 29, %5
  %Sl118 = select i1 %Cmp81, <8 x i1> undef, <8 x i1> %Tr
  %L120 = load i16, i16* undef
  store i8 %FC9, i8* %0
  %E121 = extractelement <16 x i1> %Shuff25, i32 3
  br i1 %E121, label %CF324, label %CF325

CF325:                                            ; preds = %CF362, %CF358
  %I123 = insertelement <8 x i16> undef, i16 %L120, i32 0
  %Sl125 = select i1 undef, i32 undef, i32 199785
  %Cmp126 = icmp ule <16 x i1> undef, %Cmp88
  br label %CF356

CF356:                                            ; preds = %CF356, %CF325
  %FC131 = sitofp <8 x i8> %B27 to <8 x double>
  store i8 %Sl73, i8* %0
  store i64 396197, i64* %Sl37
  %L150 = load i64, i64* %Sl37
  %Cmp157 = icmp ult i64 %L150, %L61
  br i1 %Cmp157, label %CF356, label %CF359

CF359:                                            ; preds = %CF359, %CF356
  %B162 = srem <8 x i64> %I85, %Shuff13
  %Tr163 = trunc i64 %Se to i8
  %Sl164 = select i1 %Cmp52, i32* %A1, i32* %1
  store i64 %E32, i64* undef
  %I168 = insertelement <8 x i16> %I123, i16 undef, i32 5
  %Se170 = sext i1 %Cmp81 to i32
  %Cmp172 = icmp uge i8 %Sl73, %Sl73
  br i1 %Cmp172, label %CF359, label %CF362

CF362:                                            ; preds = %CF359
  store i16 0, i16* undef
  store i64 448097, i64* %Sl37
  %E189 = extractelement <8 x i16> %Shuff48, i32 6
  %Sl194 = select i1 %Cmp111, i8 29, i8 0
  %Cmp195 = icmp eq i32 %ZE, %ZE
  br i1 %Cmp195, label %CF325, label %CF326

CF326:                                            ; preds = %CF342, %CF362
  store i64 %L104, i64* undef
  br label %CF342

CF342:                                            ; preds = %CF326
  %Cmp203 = icmp ule i1 %Cmp195, %E18
  br i1 %Cmp203, label %CF326, label %CF337

CF337:                                            ; preds = %CF342
  br label %CF327

CF327:                                            ; preds = %CF336, %CF355, %CF327, %CF337
  store i64 %Se86, i64* undef
  %Tr216 = trunc i64 184653 to i16
  %Sl217 = select i1 %Cmp157, <4 x i1> undef, <4 x i1> undef
  %Cmp218 = icmp slt i32 undef, %Se170
  br i1 %Cmp218, label %CF327, label %CF355

CF355:                                            ; preds = %CF327
  %E220 = extractelement <16 x i1> %Cmp126, i32 3
  br i1 %E220, label %CF327, label %CF340

CF340:                                            ; preds = %CF355
  %Sl224 = select i1 %Sl65, double undef, double 0xBE278346AB25A5C4
  br label %CF334

CF334:                                            ; preds = %CF343, %CF334, %CF340
  %L226 = load i64, i64* undef
  store i32 %3, i32* %Sl164
  %Cmp233 = icmp uge i16 %Tr216, %L120
  br i1 %Cmp233, label %CF334, label %CF354

CF354:                                            ; preds = %CF334
  store i64 %L226, i64* %Sl37
  %Cmp240 = icmp uge i1 %Cmp52, undef
  %Shuff243 = shufflevector <16 x i1> %I107, <16 x i1> undef, <16 x i32> <i32 28, i32 30, i32 undef, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 undef>
  %B245 = fmul <16 x float> %FC109, %FC109
  br label %CF343

CF343:                                            ; preds = %CF354
  %Cmp248 = icmp sgt i8 0, %B116
  br i1 %Cmp248, label %CF334, label %CF336

CF336:                                            ; preds = %CF343
  store i64 %E32, i64* undef
  br i1 undef, label %CF327, label %CF328

CF328:                                            ; preds = %CF345, %CF336
  br label %CF345

CF345:                                            ; preds = %CF328
  %E257 = extractelement <4 x i1> %Sl217, i32 2
  br i1 %E257, label %CF328, label %CF338

CF338:                                            ; preds = %CF345
  %Sl261 = select i1 %E121, <8 x i16> zeroinitializer, <8 x i16> undef
  %Cmp262 = icmp sgt i8 undef, %Sl194
  br label %CF329

CF329:                                            ; preds = %CF339, %CF348, %CF357, %CF338
  store i64 %L67, i64* %Sl37
  br label %CF357

CF357:                                            ; preds = %CF329
  %Cmp275 = icmp ne i1 %Cmp203, %Sl65
  br i1 %Cmp275, label %CF329, label %CF348

CF348:                                            ; preds = %CF357
  %Shuff286 = shufflevector <8 x i16> undef, <8 x i16> %Sl261, <8 x i32> <i32 6, i32 8, i32 10, i32 12, i32 undef, i32 0, i32 2, i32 4>
  %Cmp291 = icmp ne i32 %Sl125, undef
  br i1 %Cmp291, label %CF329, label %CF339

CF339:                                            ; preds = %CF348
  %Cmp299 = fcmp ugt double %L82, undef
  br i1 %Cmp299, label %CF329, label %CF330

CF330:                                            ; preds = %CF361, %CF330, %CF339
  %E301 = extractelement <8 x double> %FC131, i32 3
  store i64 %Sl16, i64* %Sl37
  %Se313 = sext <8 x i1> %Sl118 to <8 x i32>
  %Cmp315 = icmp sgt i8 %Tr163, %L
  br i1 %Cmp315, label %CF330, label %CF361

CF361:                                            ; preds = %CF330
  store i16 %L120, i16* undef
  %Shuff318 = shufflevector <8 x i64> %B162, <8 x i64> undef, <8 x i32> <i32 8, i32 10, i32 12, i32 14, i32 0, i32 2, i32 4, i32 6>
  %ZE321 = zext i16 %E189 to i64
  %Sl322 = select i1 %Cmp240, i1 %Cmp262, i1 %Cmp291
  br i1 %Sl322, label %CF330, label %CF351

CF351:                                            ; preds = %CF361
  store double %Sl224, double* %Sl44
  store i32 %ZE, i32* %Sl164
  ret void
}
