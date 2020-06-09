; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 -verify-machineinstrs | FileCheck %s
;
; Regression test for a machine verifier complaint discovered with llvm-stress.
; Test that splitting of a 128 bit store does not result in use of undef phys reg.

define void @autogen_SD29355(i8*, i32*, i64*, i32, i64, i8) {
; CHECK: .text
BB:
  %A4 = alloca double
  %A3 = alloca float
  %A2 = alloca i8
  %A1 = alloca double
  %A = alloca i64
  %L = load i8, i8* %0
  store i8 33, i8* %0
  %E = extractelement <8 x i1> zeroinitializer, i32 2
  br label %CF261

CF261:                                            ; preds = %BB
  %Shuff = shufflevector <2 x i16> zeroinitializer, <2 x i16> zeroinitializer, <2 x i32> <i32 undef, i32 3>
  %I = insertelement <8 x i8> zeroinitializer, i8 69, i32 3
  %B = udiv i8 -99, 33
  %Tr = trunc i64 -1 to i32
  %Sl = select i1 true, i64* %2, i64* %2
  %L5 = load i64, i64* %Sl
  store i64 %L5, i64* %2
  %E6 = extractelement <4 x i16> zeroinitializer, i32 3
  %Shuff7 = shufflevector <4 x i16> zeroinitializer, <4 x i16> zeroinitializer, <4 x i32> <i32 6, i32 0, i32 2, i32 4>
  %I8 = insertelement <4 x i16> %Shuff7, i16 27357, i32 0
  %B9 = xor <4 x i16> %Shuff7, %Shuff7
  %Tr10 = trunc i64 %4 to i1
  br label %CF239

CF239:                                            ; preds = %CF261
  %Sl11 = select i1 %Tr10, i16 -1, i16 27357
  %L12 = load i8, i8* %0
  store i64 %L5, i64* %A
  %E13 = extractelement <8 x i1> zeroinitializer, i32 0
  br label %CF238

CF238:                                            ; preds = %CF238, %CF239
  %Shuff14 = shufflevector <4 x i16> zeroinitializer, <4 x i16> zeroinitializer, <4 x i32> <i32 undef, i32 5, i32 7, i32 1>
  %I15 = insertelement <4 x i16> %Shuff7, i16 -1, i32 1
  %B16 = fsub double 0xDACBFCEAC1C99968, 0xDACBFCEAC1C99968
  %Sl17 = select i1 %E, i64* %Sl, i64* %Sl
  %Cmp = icmp ugt i16 %E6, 27357
  br i1 %Cmp, label %CF238, label %CF251

CF251:                                            ; preds = %CF238
  %L18 = load i64, i64* %Sl17
  store i64 0, i64* %Sl
  %E19 = extractelement <4 x i16> zeroinitializer, i32 1
  %Shuff20 = shufflevector <2 x i1> zeroinitializer, <2 x i1> zeroinitializer, <2 x i32> <i32 undef, i32 2>
  %I21 = insertelement <2 x i1> zeroinitializer, i1 true, i32 0
  %FC = fptoui float 0x3BE9BD7D80000000 to i1
  br label %CF237

CF237:                                            ; preds = %CF237, %CF271, %CF268, %CF251
  %Sl22 = select i1 true, i16 -1, i16 %E6
  %Cmp23 = icmp sgt i1 %E13, true
  br i1 %Cmp23, label %CF237, label %CF256

CF256:                                            ; preds = %CF256, %CF237
  %L24 = load i64, i64* %A
  store i64 %L5, i64* %Sl17
  %E25 = extractelement <4 x i16> zeroinitializer, i32 3
  %Shuff26 = shufflevector <4 x i16> %Shuff7, <4 x i16> zeroinitializer, <4 x i32> <i32 2, i32 4, i32 6, i32 undef>
  %I27 = insertelement <4 x i16> zeroinitializer, i16 %Sl22, i32 0
  %B28 = udiv i16 %Sl11, -1
  %ZE = zext i1 true to i32
  %Sl29 = select i1 true, i8 -99, i8 33
  %Cmp30 = fcmp ord double 0xC275146F92573C4, 0x16FB351AF5F9C998
  br i1 %Cmp30, label %CF256, label %CF271

CF271:                                            ; preds = %CF256
  %L31 = load i8, i8* %0
  store i64 %L5, i64* %Sl
  %E32 = extractelement <4 x i16> zeroinitializer, i32 2
  %Shuff33 = shufflevector <1 x i32> zeroinitializer, <1 x i32> zeroinitializer, <1 x i32> <i32 1>
  %I34 = insertelement <4 x i16> zeroinitializer, i16 %Sl11, i32 1
  %PC = bitcast double* %A4 to i1*
  %Sl35 = select i1 %FC, i32* %1, i32* %1
  %Cmp36 = icmp ult <2 x i1> %Shuff20, %Shuff20
  %L37 = load i64, i64* %Sl
  store i64 %L5, i64* %Sl
  %E38 = extractelement <2 x i32> zeroinitializer, i32 0
  %Shuff39 = shufflevector <4 x i16> zeroinitializer, <4 x i16> %Shuff7, <4 x i32> <i32 undef, i32 1, i32 3, i32 undef>
  %I40 = insertelement <4 x i16> %Shuff7, i16 %E19, i32 1
  %ZE41 = zext i1 true to i16
  %Sl42 = select i1 true, i1 true, i1 true
  br i1 %Sl42, label %CF237, label %CF246

CF246:                                            ; preds = %CF246, %CF271
  %Cmp43 = icmp uge i64 %L37, %L18
  br i1 %Cmp43, label %CF246, label %CF249

CF249:                                            ; preds = %CF249, %CF263, %CF246
  %L44 = load i64, i64* %A
  store i64 %L5, i64* %Sl17
  %E45 = extractelement <4 x i16> %Shuff14, i32 2
  %Shuff46 = shufflevector <1 x i32> zeroinitializer, <1 x i32> zeroinitializer, <1 x i32> <i32 1>
  %I47 = insertelement <4 x i16> %Shuff7, i16 %E6, i32 1
  %Sl48 = select i1 %FC, double 0xDACBFCEAC1C99968, double 0xDACBFCEAC1C99968
  %Cmp49 = fcmp ult double 0x9E8F85AE4F8D6C2C, 0x5A7FED9E637D2C1C
  br i1 %Cmp49, label %CF249, label %CF263

CF263:                                            ; preds = %CF249
  %L50 = load i64, i64* %Sl
  store i1 true, i1* %PC
  %E51 = extractelement <2 x i1> zeroinitializer, i32 0
  br i1 %E51, label %CF249, label %CF259

CF259:                                            ; preds = %CF259, %CF263
  %Shuff52 = shufflevector <4 x i64> zeroinitializer, <4 x i64> zeroinitializer, <4 x i32> <i32 undef, i32 5, i32 7, i32 1>
  %I53 = insertelement <4 x i16> zeroinitializer, i16 -1, i32 1
  %B54 = or <2 x i16> %Shuff, zeroinitializer
  %Sl55 = select i1 %Sl42, i16 %Sl22, i16 27357
  %Cmp56 = icmp uge i1 %Sl42, true
  br i1 %Cmp56, label %CF259, label %CF268

CF268:                                            ; preds = %CF259
  %L57 = load i8, i8* %0
  store i64 %L5, i64* %Sl
  %E58 = extractelement <4 x i16> %Shuff14, i32 1
  %Shuff59 = shufflevector <1 x i32> %Shuff33, <1 x i32> %Shuff33, <1 x i32> zeroinitializer
  %I60 = insertelement <2 x i1> %Shuff20, i1 true, i32 0
  %B61 = frem double 0x5A7FED9E637D2C1C, %B16
  %FC62 = sitofp i8 -99 to float
  %Sl63 = select i1 true, i16 %E19, i16 -1
  %Cmp64 = icmp slt i16 %Sl63, 27357
  br i1 %Cmp64, label %CF237, label %CF241

CF241:                                            ; preds = %CF241, %CF265, %CF268
  %L65 = load i1, i1* %PC
  br i1 %L65, label %CF241, label %CF262

CF262:                                            ; preds = %CF262, %CF270, %CF241
  store i64 %L37, i64* %Sl
  %E66 = extractelement <4 x i16> %Shuff14, i32 2
  %Shuff67 = shufflevector <4 x i16> %Shuff26, <4 x i16> %Shuff7, <4 x i32> <i32 1, i32 3, i32 undef, i32 7>
  %I68 = insertelement <2 x i32> zeroinitializer, i32 454413, i32 1
  %B69 = sub <4 x i16> %I8, %Shuff7
  %Tr70 = trunc i16 %E32 to i1
  br i1 %Tr70, label %CF262, label %CF270

CF270:                                            ; preds = %CF262
  %Sl71 = select i1 %Sl42, <8 x i1> zeroinitializer, <8 x i1> zeroinitializer
  %Cmp72 = icmp sge <2 x i16> %B54, zeroinitializer
  %L73 = load i64, i64* %Sl
  store i64 %L73, i64* %Sl
  %E74 = extractelement <8 x i1> %Sl71, i32 5
  br i1 %E74, label %CF262, label %CF265

CF265:                                            ; preds = %CF270
  %Shuff75 = shufflevector <2 x i32> %I68, <2 x i32> zeroinitializer, <2 x i32> <i32 undef, i32 2>
  %I76 = insertelement <2 x i1> %Cmp72, i1 %Sl42, i32 0
  %B77 = xor i16 27357, %B28
  %PC78 = bitcast i1* %PC to i32*
  %Sl79 = select i1 %Cmp64, <4 x i16> %Shuff14, <4 x i16> %Shuff7
  %Cmp80 = icmp slt <2 x i1> zeroinitializer, %Shuff20
  %L81 = load i1, i1* %PC
  br i1 %L81, label %CF241, label %CF245

CF245:                                            ; preds = %CF245, %CF265
  store i1 true, i1* %PC
  %E82 = extractelement <1 x i32> %Shuff33, i32 0
  %Shuff83 = shufflevector <4 x i16> zeroinitializer, <4 x i16> %Shuff14, <4 x i32> <i32 2, i32 4, i32 6, i32 0>
  %I84 = insertelement <2 x i1> %Shuff20, i1 %Sl42, i32 0
  %FC85 = uitofp i1 %Cmp to float
  %Sl86 = select i1 %Tr10, i16 -1, i16 %Sl63
  %Cmp87 = icmp ugt <2 x i1> %I76, %I60
  %L88 = load i32, i32* %PC78
  store i8 33, i8* %0
  %E89 = extractelement <2 x i32> zeroinitializer, i32 1
  %Shuff90 = shufflevector <4 x i64> zeroinitializer, <4 x i64> %Shuff52, <4 x i32> <i32 0, i32 undef, i32 4, i32 6>
  %I91 = insertelement <2 x i32> %Shuff75, i32 %ZE, i32 0
  %B92 = add i64 -1, %L73
  %Tr93 = trunc i64 0 to i16
  %Sl94 = select i1 %FC, i64 %L37, i64 %L5
  %Cmp95 = icmp sge i64 454853, %B92
  br i1 %Cmp95, label %CF245, label %CF257

CF257:                                            ; preds = %CF245
  %L96 = load i64, i64* %Sl
  store i1 true, i1* %PC
  %E97 = extractelement <2 x i1> %Shuff20, i32 1
  br label %CF

CF:                                               ; preds = %CF, %CF258, %CF257
  %Shuff98 = shufflevector <2 x i1> %Cmp80, <2 x i1> zeroinitializer, <2 x i32> <i32 undef, i32 0>
  %I99 = insertelement <2 x i1> %Shuff98, i1 %Cmp30, i32 0
  %B100 = sub <8 x i8> zeroinitializer, zeroinitializer
  %FC101 = uitofp <2 x i1> %I99 to <2 x double>
  %Sl102 = select i1 %FC, i16 %Sl63, i16 %E58
  %Cmp103 = fcmp ord double %B16, 0xDACBFCEAC1C99968
  br i1 %Cmp103, label %CF, label %CF240

CF240:                                            ; preds = %CF240, %CF260, %CF
  %L104 = load i32, i32* %1
  store i1 true, i1* %PC
  %E105 = extractelement <4 x i16> %I8, i32 1
  %Shuff106 = shufflevector <4 x i16> %Shuff7, <4 x i16> %I34, <4 x i32> <i32 4, i32 undef, i32 undef, i32 2>
  %I107 = insertelement <2 x i1> %Cmp87, i1 %FC, i32 0
  %ZE108 = zext <4 x i16> %B69 to <4 x i64>
  %Sl109 = select i1 %Cmp, i16 27357, i16 %Sl102
  %Cmp110 = icmp sge <4 x i16> %B9, zeroinitializer
  %L111 = load i64, i64* %Sl
  store i8 %L57, i8* %0
  %E112 = extractelement <2 x i1> %Shuff98, i32 0
  br i1 %E112, label %CF240, label %CF254

CF254:                                            ; preds = %CF254, %CF267, %CF264, %CF240
  %Shuff113 = shufflevector <2 x i32> %I68, <2 x i32> zeroinitializer, <2 x i32><i32 undef, i32 0>
  %I114 = insertelement <4 x i16> zeroinitializer, i16 27357, i32 3
  %B115 = and i16 %Sl102, %Sl11
  %FC116 = uitofp i16 %B115 to double
  %Sl117 = select i1 %L81, i32* %1, i32* %1
  %Cmp118 = icmp ne i64 %Sl94, %L50
  br i1 %Cmp118, label %CF254, label %CF267

CF267:                                            ; preds = %CF254
  %L119 = load i64, i64* %Sl
  store i32 %ZE, i32* %PC78
  %E120 = extractelement <4 x i16> zeroinitializer, i32 1
  %Shuff121 = shufflevector <1 x i32> %Shuff33, <1 x i32> %Shuff33, <1 x i32> zeroinitializer
  %I122 = insertelement <1 x i32> %Shuff121, i32 %E82, i32 0
  %B123 = mul <4 x i16> %I40, %I34
  %Sl124 = select i1 %FC, <4 x i1> %Cmp110, <4 x i1> %Cmp110
  %Cmp125 = icmp ne <4 x i64> %ZE108, zeroinitializer
  %L126 = load i64, i64* %Sl
  store i32 %ZE, i32* %Sl117
  %E127 = extractelement <2 x i1> %Cmp87, i32 1
  br i1 %E127, label %CF254, label %CF264

CF264:                                            ; preds = %CF267
  %Shuff128 = shufflevector <4 x i16> %Shuff83, <4 x i16> %I47, <4 x i32> <i32 undef, i32 2, i32 undef, i32 6>
  %I129 = insertelement <4 x i16> %Shuff67, i16 %Sl109, i32 2
  %B130 = add i32 %3, %E38
  %FC131 = sitofp i32 %3 to float
  %Sl132 = select i1 %Sl42, i64 %L24, i64 %L5
  %Cmp133 = icmp eq <2 x i1> %I99, %Shuff20
  %L134 = load i32, i32* %PC78
  store i32 %L104, i32* %1
  %E135 = extractelement <8 x i1> zeroinitializer, i32 4
  br i1 %E135, label %CF254, label %CF260

CF260:                                            ; preds = %CF264
  %Shuff136 = shufflevector <1 x i32> %Shuff59, <1 x i32> %Shuff121, <1 x i32> undef
  %I137 = insertelement <4 x i16> %Shuff67, i16 %Sl55, i32 3
  %B138 = lshr <1 x i32> %Shuff33, %Shuff59
  %Sl139 = select i1 %E135, i64 %L119, i64 %L126
  %Cmp140 = icmp slt i8 -99, %Sl29
  br i1 %Cmp140, label %CF240, label %CF247

CF247:                                            ; preds = %CF247, %CF272, %CF260
  %L141 = load i32, i32* %Sl117
  store i8 %5, i8* %0
  %E142 = extractelement <2 x i1> %Cmp36, i32 1
  br i1 %E142, label %CF247, label %CF272

CF272:                                            ; preds = %CF247
  %Shuff143 = shufflevector <4 x i64> %Shuff90, <4 x i64> %Shuff52, <4 x i32> <i32 6, i32 undef, i32 2, i32 undef>
  %I144 = insertelement <1 x i32> %Shuff121, i32 %L88, i32 0
  %Tr145 = trunc i64 %Sl139 to i16
  %Sl146 = select i1 %Cmp49, i32 %L134, i32 %L104
  %L147 = load i32, i32* %PC78
  store i32 %Tr, i32* %Sl117
  %E148 = extractelement <4 x i16> %Shuff67, i32 3
  %Shuff149 = shufflevector <4 x i16> zeroinitializer, <4 x i16> %Shuff67, <4 x i32> <i32 2, i32 4, i32 6, i32 0>
  %I150 = insertelement <2 x i1> zeroinitializer, i1 %E127, i32 0
  %B151 = fdiv double 0x16FB351AF5F9C998, 0xC275146F92573C4
  %FC152 = uitofp <1 x i32> %I144 to <1 x double>
  %Sl153 = select i1 %Cmp118, <1 x i32> %Shuff136, <1 x i32> %Shuff121
  %Cmp154 = icmp ule i8 %5, %Sl29
  br i1 %Cmp154, label %CF247, label %CF253

CF253:                                            ; preds = %CF253, %CF269, %CF272
  %L155 = load i32, i32* %Sl117
  store i32 %L141, i32* %PC78
  %E156 = extractelement <4 x i1> %Cmp125, i32 2
  br i1 %E156, label %CF253, label %CF269

CF269:                                            ; preds = %CF253
  %Shuff157 = shufflevector <1 x i32> %Shuff46, <1 x i32> %Shuff121, <1 x i32> <i32 1>
  %I158 = insertelement <4 x i16> %Shuff128, i16 %E66, i32 1
  %B159 = shl i64 %L119, %L73
  %Se = sext i16 %B77 to i32
  %Sl160 = select i1 %Cmp56, i16 %Sl63, i16 %B77
  %L161 = load i64, i64* %Sl
  store i32 %B130, i32* %Sl117
  %E162 = extractelement <1 x i32> %Shuff59, i32 0
  %Shuff163 = shufflevector <4 x i16> %Shuff7, <4 x i16> %Shuff67, <4 x i32> <i32 5, i32 7, i32 1, i32 3>
  %I164 = insertelement <4 x i16> %Shuff106, i16 27357, i32 3
  %Se165 = sext <4 x i1> %Sl124 to <4 x i8>
  %Sl166 = select i1 true, i1 %Cmp, i1 %Tr70
  br i1 %Sl166, label %CF253, label %CF255

CF255:                                            ; preds = %CF255, %CF266, %CF269
  %Cmp167 = icmp sge i64 %4, %L24
  br i1 %Cmp167, label %CF255, label %CF266

CF266:                                            ; preds = %CF255
  %L168 = load i8, i8* %0
  store i32 %E38, i32* %PC78
  %E169 = extractelement <2 x i16> zeroinitializer, i32 1
  %Shuff170 = shufflevector <4 x i16> %Sl79, <4 x i16> %I137, <4 x i32> <i32 6, i32 0, i32 2, i32 4>
  %I171 = insertelement <4 x i16> %Shuff163, i16 %ZE41, i32 0
  %Tr172 = trunc i16 %Tr145 to i1
  br i1 %Tr172, label %CF255, label %CF258

CF258:                                            ; preds = %CF266
  %Sl173 = select i1 true, <2 x i32> %I68, <2 x i32> %I91
  %Cmp174 = icmp ugt <2 x i1> %Cmp72, %I150
  %L175 = load i32, i32* %Sl117
  store i32 %L104, i32* %Sl117
  %E176 = extractelement <4 x i16> %Shuff67, i32 1
  %Shuff177 = shufflevector <1 x i32> %Shuff121, <1 x i32> %Shuff33, <1 x i32> zeroinitializer
  %I178 = insertelement <4 x i16> zeroinitializer, i16 27357, i32 0
  %FC179 = sitofp <4 x i16> %I47 to <4 x float>
  %Sl180 = select i1 %FC, i64 %L126, i64 %B92
  %Cmp181 = fcmp ugt double %B61, %B16
  br i1 %Cmp181, label %CF, label %CF236

CF236:                                            ; preds = %CF236, %CF258
  %L182 = load i8, i8* %0
  store i32 %E38, i32* %Sl117
  %E183 = extractelement <1 x i32> %Shuff121, i32 0
  %Shuff184 = shufflevector <4 x i64> zeroinitializer, <4 x i64> %Shuff90, <4 x i32> <i32 7, i32 undef, i32 3, i32 5>
  %I185 = insertelement <4 x i16> %Shuff106, i16 %Tr93, i32 1
  %ZE186 = zext i32 %E162 to i64
  %Sl187 = select i1 %Cmp95, <8 x i8> %B100, <8 x i8> %B100
  %Cmp188 = icmp uge i16 %B115, %Sl11
  br i1 %Cmp188, label %CF236, label %CF242

CF242:                                            ; preds = %CF242, %CF250, %CF248, %CF236
  %L189 = load i8, i8* %0
  store i8 %Sl29, i8* %0
  %E190 = extractelement <4 x i16> %B9, i32 3
  %Shuff191 = shufflevector <4 x i16> %Shuff26, <4 x i16> %Shuff26, <4 x i32> <i32 6, i32 0, i32 2, i32 4>
  %I192 = insertelement <1 x i32> %I122, i32 %3, i32 0
  %B193 = udiv i8 %5, %L168
  %Se194 = sext <8 x i1> %Sl71 to <8 x i32>
  %Sl195 = select i1 %Cmp188, i8 %L182, i8 %L168
  %Cmp196 = icmp slt i16 %B77, %Sl102
  br i1 %Cmp196, label %CF242, label %CF250

CF250:                                            ; preds = %CF242
  %L197 = load i64, i64* %Sl
  store i32 %ZE, i32* %Sl117
  %E198 = extractelement <2 x i1> %Shuff20, i32 1
  br i1 %E198, label %CF242, label %CF244

CF244:                                            ; preds = %CF244, %CF250
  %Shuff199 = shufflevector <1 x i32> %Shuff46, <1 x i32> %Shuff177, <1 x i32> zeroinitializer
  %I200 = insertelement <4 x i16> %Shuff191, i16 %Sl86, i32 0
  %B201 = mul i16 %ZE41, %E169
  %Se202 = sext <4 x i16> %I171 to <4 x i64>
  %Sl203 = select i1 %Sl166, i32 %E162, i32 %E82
  %Cmp204 = icmp ule i16 %E32, %E120
  br i1 %Cmp204, label %CF244, label %CF248

CF248:                                            ; preds = %CF244
  %L205 = load float, float* %A3
  store i32 %Tr, i32* %PC78
  %E206 = extractelement <2 x i1> %Shuff20, i32 1
  br i1 %E206, label %CF242, label %CF243

CF243:                                            ; preds = %CF243, %CF273, %CF248
  %Shuff207 = shufflevector <8 x i1> zeroinitializer, <8 x i1> %Sl71, <8 x i32> <i32 4, i32 6, i32 8, i32 undef, i32 12, i32 undef, i32 undef, i32 2>
  %I208 = insertelement <2 x i1> %Shuff20, i1 %E198, i32 0
  %B209 = xor <4 x i16> %I129, %I34
  %FC210 = uitofp <8 x i8> zeroinitializer to <8 x double>
  %Sl211 = select i1 %E74, i16 %Tr93, i16 %E19
  %Cmp212 = icmp ugt i32 %Se, %E38
  br i1 %Cmp212, label %CF243, label %CF273

CF273:                                            ; preds = %CF243
  %L213 = load i32, i32* %PC78
  store i8 %L168, i8* %0
  %E214 = extractelement <2 x i32> %Shuff113, i32 1
  %Shuff215 = shufflevector <4 x i16> %Shuff128, <4 x i16> %I137, <4 x i32> <i32 6, i32 0, i32 2, i32 4>
  %I216 = insertelement <2 x i1> %Shuff20, i1 %Cmp30, i32 0
  %B217 = sub <4 x i16> %Shuff83, %I185
  %Tr218 = trunc <4 x i16> %B9 to <4 x i1>
  %Sl219 = select i1 %Cmp154, i8 %B, i8 %5
  %Cmp220 = icmp uge <4 x i64> %Shuff52, %Shuff52
  %L221 = load i32, i32* %Sl117
  store i8 %L168, i8* %0
  %E222 = extractelement <4 x i16> %Shuff191, i32 0
  %Shuff223 = shufflevector <4 x i16> %Shuff26, <4 x i16> %I34, <4 x i32> <i32 undef, i32 1, i32 3, i32 5>
  %I224 = insertelement <4 x i16> %Shuff26, i16 %Tr145, i32 1
  %FC225 = sitofp i1 %Cmp56 to float
  %Sl226 = select i1 %E, i1 %Cmp154, i1 %Sl166
  br i1 %Sl226, label %CF243, label %CF252

CF252:                                            ; preds = %CF273
  %Cmp227 = icmp ugt <4 x i64> %Shuff143, zeroinitializer
  %L228 = load i32, i32* %Sl117
  store i32 %Tr, i32* %PC78
  %E229 = extractelement <4 x i16> %Shuff163, i32 2
  %Shuff230 = shufflevector <1 x i32> %Shuff199, <1 x i32> zeroinitializer, <1 x i32> <i32 1>
  %I231 = insertelement <4 x i16> %Shuff106, i16 %E32, i32 1
  %B232 = srem i32 %Sl203, %Sl203
  %FC233 = fptoui double 0x5A7FED9E637D2C1C to i32
  %Sl234 = select i1 %Cmp103, i8 %B193, i8 %L168
  %Cmp235 = icmp uge <2 x i16> zeroinitializer, zeroinitializer
  store i32 %ZE, i32* %PC78
  store i64 %L5, i64* %Sl
  store i8 33, i8* %0
  store i8 %L168, i8* %0
  store i1 %Sl226, i1* %PC
  ret void
}
