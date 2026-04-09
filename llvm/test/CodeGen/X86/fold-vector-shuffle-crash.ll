; RUN: llc < %s -mtriple=x86_64-unknown -mcpu=corei7

define void @autogen_SD13708(i32, i1 %arg) {
BB:
 %Shuff7 = shufflevector <8 x i32> zeroinitializer, <8 x i32> zeroinitializer, <8 x i32> <i32 8, i32 10, i32 12, i32 14, i32 undef, i32 2, i32 4, i32 undef>
 br label %CF

CF:
 %Tr = trunc <8 x i64> zeroinitializer to <8 x i32>
 %Shuff20 = shufflevector <8 x i32> %Shuff7, <8 x i32> %Tr, <8 x i32> <i32 13, i32 15, i32 1, i32 3, i32 5, i32 7, i32 undef, i32 11>
 br i1 %arg, label %CF, label %CF247

CF247:
 %I171 = insertelement <8 x i32> %Shuff20, i32 %0, i32 0
 br i1 %arg, label %CF, label %CF247
}

define void @autogen_SD13800(ptr, ptr, ptr, i32, i64, i8) {
BB:
  %A4 = alloca <8 x i1>
  %A3 = alloca i32
  %A2 = alloca <2 x float>
  %A1 = alloca <2 x double>
  %A = alloca <8 x float>
  %L = load <8 x i1>, ptr %A4
  store i8 %5, ptr %0
  %E = extractelement <2 x i64> zeroinitializer, i32 0
  %Shuff = shufflevector <4 x i64> zeroinitializer, <4 x i64> zeroinitializer, <4 x i32> <i32 7, i32 undef, i32 undef, i32 5>
  %I = insertelement <8 x i64> zeroinitializer, i64 419346, i32 1
  %B = shl i64 426618, 419346
  %Tr = trunc <8 x i64> %I to <8 x i16>
  %Sl = select i1 false, <4 x i64> zeroinitializer, <4 x i64> zeroinitializer
  %Cmp = icmp eq <16 x i64> zeroinitializer, zeroinitializer
  %L5 = load i8, ptr %0
  store i8 17, ptr %0
  %E6 = extractelement <4 x i64> zeroinitializer, i32 1
  %Shuff7 = shufflevector <2 x i64> zeroinitializer, <2 x i64> <i64 -1, i64 -1>, <2 x i32> <i32 0, i32 2>
  %I8 = insertelement <16 x i8> <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>, i8 %L5, i32 2
  %B9 = mul <8 x i16> %Tr, %Tr
  %FC = fptosi float 0xBDF7B90B80000000 to i32
  %Sl10 = select i1 false, float 0xBDF7B90B80000000, float 0xB875A90980000000
  %Cmp11 = icmp slt <2 x i64> zeroinitializer, %Shuff7
  %L12 = load <8 x float>, ptr %A
  store <2 x double> <double 0xFFFFFFFFFFFFFFFF, double 0.000000e+00>, ptr %A1
  %E13 = extractelement <4 x i64> zeroinitializer, i32 2
  %Shuff14 = shufflevector <2 x i32> zeroinitializer, <2 x i32> zeroinitializer, <2 x i32> <i32 1, i32 3>
  %I15 = insertelement <16 x i8> <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>, i8 %L5, i32 15
  %B16 = add <2 x i64> zeroinitializer, <i64 -1, i64 -1>
  %BC = bitcast i64 426618 to double
  %Sl17 = select i1 false, <2 x i32> zeroinitializer, <2 x i32> zeroinitializer
  %Cmp18 = icmp slt <8 x i1> %L, %L
  %L19 = load i8, ptr %0
  store i8 %L5, ptr %0
  %E20 = extractelement <16 x i8> %I8, i32 1
  %Shuff21 = shufflevector <8 x i64> zeroinitializer, <8 x i64> %I, <8 x i32> <i32 undef, i32 6, i32 8, i32 10, i32 12, i32 14, i32 0, i32 2>
  %I22 = insertelement <8 x i16> %Tr, i16 18460, i32 6
  %B23 = sub i64 419346, %4
  %FC24 = fptosi double 0xE603EE221901D6A0 to i32
  %Sl25 = select i1 false, i8 %L5, i8 %5
  %Cmp26 = icmp ugt i64 %B, %B23
  br label %CF253

CF253:                                            ; preds = %CF253, %CF271, %CF260, %BB
  %L27 = load i8, ptr %0
  store i8 %L5, ptr %0
  %E28 = extractelement <2 x i64> zeroinitializer, i32 0
  %Shuff29 = shufflevector <16 x i8> <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>, <16 x i8> %I8, <16 x i32> <i32 15, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 undef, i32 31, i32 undef, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13>
  %I30 = insertelement <8 x i1> %Cmp18, i1 false, i32 1
  %B31 = fsub double 0xE603EE221901D6A0, %BC
  %Tr32 = trunc <2 x i64> <i64 -1, i64 -1> to <2 x i32>
  %Sl33 = select i1 false, double %BC, double %B31
  %Cmp34 = icmp sgt <2 x i32> zeroinitializer, %Shuff14
  %L35 = load i8, ptr %0
  store i8 %L5, ptr %0
  %E36 = extractelement <16 x i8> %Shuff29, i32 5
  %Shuff37 = shufflevector <8 x i64> zeroinitializer, <8 x i64> %I, <8 x i32> <i32 8, i32 undef, i32 undef, i32 14, i32 0, i32 undef, i32 4, i32 6>
  %I38 = insertelement <4 x i64> zeroinitializer, i64 %E28, i32 2
  %FC39 = uitofp i8 %5 to double
  %Sl40 = select i1 %Cmp26, i32 %3, i32 %FC
  %Cmp41 = icmp sgt <2 x i64> zeroinitializer, <i64 -1, i64 -1>
  %L42 = load i8, ptr %0
  store i8 17, ptr %0
  %E43 = extractelement <2 x i1> %Cmp41, i32 1
  br i1 %E43, label %CF253, label %CF256

CF256:                                            ; preds = %CF256, %CF253
  %Shuff44 = shufflevector <8 x i64> zeroinitializer, <8 x i64> zeroinitializer, <8 x i32> <i32 14, i32 0, i32 2, i32 4, i32 6, i32 undef, i32 undef, i32 12>
  %I45 = insertelement <8 x i32> zeroinitializer, i32 %FC, i32 0
  %ZE = zext i8 %L19 to i32
  %Sl46 = select i1 %E43, i8 %L35, i8 %L35
  %Cmp47 = icmp ult i64 %E6, 426618
  br i1 %Cmp47, label %CF256, label %CF271

CF271:                                            ; preds = %CF256
  %L48 = load i8, ptr %0
  store i8 %L27, ptr %0
  %E49 = extractelement <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>, i32 2
  %Shuff50 = shufflevector <8 x i64> zeroinitializer, <8 x i64> zeroinitializer, <8 x i32> <i32 undef, i32 7, i32 undef, i32 11, i32 13, i32 15, i32 1, i32 3>
  %I51 = insertelement <8 x i64> zeroinitializer, i64 %4, i32 7
  %B52 = xor <8 x i32> %I45, zeroinitializer
  %BC53 = bitcast <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1> to <4 x float>
  %Sl54 = select i1 false, <2 x i64> <i64 -1, i64 -1>, <2 x i64> <i64 -1, i64 -1>
  %Cmp55 = icmp sgt i16 0, 18460
  br i1 %Cmp55, label %CF253, label %CF255

CF255:                                            ; preds = %CF255, %CF266, %CF270, %CF271
  %L56 = load i8, ptr %0
  store i8 %L35, ptr %0
  %E57 = extractelement <4 x i64> zeroinitializer, i32 3
  %Shuff58 = shufflevector <8 x i64> zeroinitializer, <8 x i64> %Shuff37, <8 x i32> <i32 undef, i32 undef, i32 10, i32 12, i32 14, i32 0, i32 2, i32 4>
  %I59 = insertelement <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>, i32 %FC, i32 0
  %B60 = lshr <4 x i64> %I38, zeroinitializer
  %FC61 = sitofp <8 x i1> %L to <8 x float>
  %Sl62 = select i1 false, i8 %L19, i8 17
  %Cmp63 = icmp ult i64 %E6, %E57
  br i1 %Cmp63, label %CF255, label %CF266

CF266:                                            ; preds = %CF255
  %L64 = load i64, ptr %2
  store i8 17, ptr %0
  %E65 = extractelement <8 x i64> %Shuff21, i32 6
  %Shuff66 = shufflevector <2 x i1> %Cmp11, <2 x i1> %Cmp41, <2 x i32> <i32 1, i32 3>
  %I67 = insertelement <8 x i1> %I30, i1 false, i32 7
  %FC68 = uitofp i8 %Sl62 to float
  %Sl69 = select i1 false, i8 %L42, i8 17
  %Cmp70 = icmp eq <2 x i32> zeroinitializer, zeroinitializer
  %L71 = load i8, ptr %0
  store i8 %5, ptr %0
  %E72 = extractelement <2 x i64> <i64 -1, i64 -1>, i32 1
  %Shuff73 = shufflevector <8 x i64> zeroinitializer, <8 x i64> %Shuff44, <8 x i32> <i32 undef, i32 14, i32 0, i32 2, i32 4, i32 6, i32 undef, i32 10>
  %I74 = insertelement <2 x i1> %Cmp70, i1 %Cmp55, i32 0
  %B75 = add <16 x i8> %I15, <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
  %FC76 = sitofp i8 33 to double
  %Sl77 = select i1 %E43, double %BC, double %B31
  %Cmp78 = icmp ult <8 x i64> %Shuff44, %I
  %L79 = load i8, ptr %0
  store i8 17, ptr %0
  %E80 = extractelement <2 x i64> %Shuff7, i32 0
  %Shuff81 = shufflevector <8 x i64> zeroinitializer, <8 x i64> %Shuff73, <8 x i32> <i32 undef, i32 5, i32 7, i32 9, i32 undef, i32 13, i32 15, i32 1>
  %I82 = insertelement <8 x i64> %Shuff81, i64 %E57, i32 5
  %FC83 = fptosi float %FC68 to i32
  %Sl84 = select i1 %Cmp26, <2 x i64> <i64 -1, i64 -1>, <2 x i64> <i64 -1, i64 -1>
  %Cmp85 = icmp ugt i64 %E6, %E57
  br i1 %Cmp85, label %CF255, label %CF261

CF261:                                            ; preds = %CF261, %CF266
  %L86 = load i8, ptr %0
  store i8 %L42, ptr %0
  %E87 = extractelement <16 x i8> <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>, i32 7
  %Shuff88 = shufflevector <16 x i8> %Shuff29, <16 x i8> %I15, <16 x i32> <i32 26, i32 28, i32 30, i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24>
  %I89 = insertelement <16 x i8> <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>, i8 %L35, i32 12
  %B90 = shl i32 %3, %E49
  %BC91 = bitcast <2 x i64> %Sl84 to <2 x double>
  %Sl92 = select i1 false, i8 %L5, i8 %L19
  %Cmp93 = icmp ugt i32 -1, %FC24
  br i1 %Cmp93, label %CF261, label %CF268

CF268:                                            ; preds = %CF268, %CF261
  %L94 = load i8, ptr %0
  store i8 %L5, ptr %0
  %E95 = extractelement <8 x i64> %Shuff58, i32 0
  %Shuff96 = shufflevector <8 x i64> %Shuff73, <8 x i64> %Shuff73, <8 x i32> <i32 3, i32 5, i32 undef, i32 9, i32 undef, i32 undef, i32 15, i32 1>
  %I97 = insertelement <4 x i64> zeroinitializer, i64 %B23, i32 1
  %B98 = or <8 x i64> %Shuff58, %Shuff50
  %FC99 = sitofp <2 x i1> %Cmp34 to <2 x float>
  %Sl100 = select i1 %Cmp85, i64 %4, i64 %E
  %Cmp101 = icmp ne <2 x i64> %B16, zeroinitializer
  %L102 = load i8, ptr %0
  store i8 %L56, ptr %0
  %E103 = extractelement <8 x i16> %I22, i32 6
  %Shuff104 = shufflevector <2 x double> %BC91, <2 x double> %BC91, <2 x i32> <i32 1, i32 3>
  %I105 = insertelement <8 x i64> %Shuff96, i64 198384, i32 7
  %B106 = sdiv <8 x i32> %B52, %I45
  %ZE107 = zext i16 0 to i32
  %Sl108 = select i1 %E43, <16 x i8> %Shuff29, <16 x i8> <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
  %Cmp109 = icmp slt <16 x i64> zeroinitializer, zeroinitializer
  %L110 = load <8 x float>, ptr %A
  store i8 %L56, ptr %0
  %E111 = extractelement <8 x i64> zeroinitializer, i32 3
  %Shuff112 = shufflevector <2 x i1> %Shuff66, <2 x i1> %Cmp11, <2 x i32> <i32 2, i32 0>
  %I113 = insertelement <2 x i64> %B16, i64 %E95, i32 0
  %B114 = mul i8 %E20, %Sl25
  %Tr115 = trunc <8 x i64> %I105 to <8 x i16>
  %Sl116 = select i1 %Cmp26, <16 x i32> <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>, <16 x i32> <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>
  %Cmp117 = icmp ult <8 x i16> %Tr, %Tr115
  %L118 = load i8, ptr %0
  store i8 %L19, ptr %0
  %E119 = extractelement <4 x i32> %I59, i32 3
  %Shuff120 = shufflevector <2 x i64> <i64 -1, i64 -1>, <2 x i64> %I113, <2 x i32> <i32 2, i32 0>
  %I121 = insertelement <2 x i1> %Shuff66, i1 %Cmp26, i32 0
  %B122 = fmul double 0.000000e+00, 0xE603EE221901D6A0
  %FC123 = sitofp i64 %E6 to float
  %Sl124 = select i1 false, <2 x i1> %Cmp41, <2 x i1> %Shuff66
  %Cmp125 = icmp ult <4 x i64> %I38, %I38
  %L126 = load i8, ptr %0
  store i8 %L126, ptr %0
  %E127 = extractelement <8 x i64> zeroinitializer, i32 7
  %Shuff128 = shufflevector <2 x i1> %Cmp101, <2 x i1> %Cmp11, <2 x i32> <i32 undef, i32 0>
  %I129 = insertelement <8 x i1> %Cmp18, i1 %E43, i32 0
  %B130 = lshr i8 %L71, %L56
  %FC131 = sitofp i32 %3 to float
  %Sl132 = select i1 false, <2 x i64> %Shuff7, <2 x i64> %Sl84
  %Cmp133 = icmp sge <8 x i16> %Tr, %Tr115
  %L134 = load i8, ptr %0
  store i8 %L102, ptr %0
  %E135 = extractelement <16 x i8> %Shuff88, i32 3
  %Shuff136 = shufflevector <8 x i64> %Shuff21, <8 x i64> zeroinitializer, <8 x i32> <i32 6, i32 8, i32 undef, i32 12, i32 14, i32 0, i32 2, i32 4>
  %I137 = insertelement <2 x i64> zeroinitializer, i64 %E111, i32 0
  %B138 = shl <8 x i64> %I51, %Shuff136
  %Se = sext <2 x i32> %Tr32 to <2 x i64>
  %Sl139 = select i1 %E43, <8 x i16> %Tr, <8 x i16> %Tr115
  %Cmp140 = icmp sge <2 x i32> %Sl17, %Tr32
  %L141 = load i8, ptr %0
  store i8 17, ptr %0
  %E142 = extractelement <8 x i16> %Tr115, i32 6
  %Shuff143 = shufflevector <8 x i64> zeroinitializer, <8 x i64> %Shuff21, <8 x i32> <i32 1, i32 3, i32 undef, i32 7, i32 undef, i32 11, i32 13, i32 15>
  %I144 = insertelement <4 x i64> %Shuff, i64 %4, i32 3
  %B145 = sub <2 x i64> <i64 -1, i64 -1>, %I113
  %Se146 = sext i8 %E135 to i32
  %Sl147 = select i1 %Cmp55, <2 x i32> %Tr32, <2 x i32> zeroinitializer
  %Cmp148 = icmp eq <8 x i1> %I30, %Cmp18
  %L149 = load i8, ptr %0
  store i8 %L56, ptr %0
  %E150 = extractelement <2 x i64> %I113, i32 0
  %Shuff151 = shufflevector <8 x i64> zeroinitializer, <8 x i64> %I, <8 x i32> <i32 9, i32 11, i32 13, i32 15, i32 1, i32 3, i32 undef, i32 7>
  %I152 = insertelement <8 x i64> %Shuff136, i64 %E6, i32 3
  %B153 = frem float %FC68, %FC123
  %Se154 = sext i1 false to i32
  %Sl155 = select i1 %Cmp26, i8 %E20, i8 %L19
  %Cmp156 = icmp eq i64 198384, %4
  br i1 %Cmp156, label %CF268, label %CF270

CF270:                                            ; preds = %CF268
  %L157 = load i8, ptr %0
  store i8 %L157, ptr %0
  %E158 = extractelement <8 x i1> %Cmp78, i32 1
  br i1 %E158, label %CF255, label %CF260

CF260:                                            ; preds = %CF270
  %Shuff159 = shufflevector <8 x i64> zeroinitializer, <8 x i64> %Shuff81, <8 x i32> <i32 undef, i32 6, i32 8, i32 undef, i32 12, i32 14, i32 0, i32 2>
  %I160 = insertelement <2 x i1> %Cmp11, i1 %Cmp156, i32 0
  %B161 = urem <2 x i32> zeroinitializer, %Sl147
  %Se162 = sext i8 %L48 to i16
  %Sl163 = select i1 %Cmp93, i32 %FC83, i32 378892
  %Cmp164 = fcmp uge double 0xE603EE221901D6A0, 0xE603EE221901D6A0
  br i1 %Cmp164, label %CF253, label %CF254

CF254:                                            ; preds = %CF254, %CF265, %CF263, %CF260
  %L165 = load i8, ptr %0
  store i8 %Sl62, ptr %0
  %E166 = extractelement <8 x i64> %Shuff58, i32 1
  %Shuff167 = shufflevector <8 x i64> %Shuff58, <8 x i64> %Shuff96, <8 x i32> <i32 12, i32 14, i32 0, i32 undef, i32 4, i32 undef, i32 8, i32 10>
  %I168 = insertelement <2 x double> %BC91, double %BC, i32 0
  %B169 = ashr i16 %E142, %E103
  %FC170 = sitofp <2 x i64> %Sl84 to <2 x float>
  %Sl171 = select i1 %Cmp156, i8 %L165, i8 %5
  %Cmp172 = icmp ugt i8 %E20, %L102
  br i1 %Cmp172, label %CF254, label %CF262

CF262:                                            ; preds = %CF262, %CF254
  %L173 = load i8, ptr %0
  store i8 %L94, ptr %0
  %E174 = extractelement <2 x i1> %Cmp70, i32 0
  br i1 %E174, label %CF262, label %CF264

CF264:                                            ; preds = %CF264, %CF262
  %Shuff175 = shufflevector <16 x i1> %Cmp, <16 x i1> %Cmp, <16 x i32> <i32 undef, i32 9, i32 undef, i32 13, i32 undef, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 undef, i32 1, i32 3, i32 5>
  %I176 = insertelement <8 x i64> %Shuff21, i64 419346, i32 1
  %B177 = lshr <2 x i32> %Sl17, zeroinitializer
  %FC178 = sitofp <8 x i32> %B106 to <8 x float>
  %Sl179 = select i1 %Cmp156, i8 %B114, i8 %Sl171
  %Cmp180 = icmp ugt <4 x i64> %B60, zeroinitializer
  %L181 = load i8, ptr %0
  store i8 %L102, ptr %0
  %E182 = extractelement <8 x i64> zeroinitializer, i32 0
  %Shuff183 = shufflevector <8 x i64> zeroinitializer, <8 x i64> %I176, <8 x i32> <i32 3, i32 5, i32 undef, i32 undef, i32 11, i32 13, i32 undef, i32 1>
  %I184 = insertelement <2 x i1> %Cmp34, i1 %Cmp63, i32 1
  %B185 = urem i32 %Sl163, %Se146
  %FC186 = sitofp i64 %E166 to float
  %Sl187 = select i1 %Cmp156, i1 %E43, i1 %Cmp26
  br i1 %Sl187, label %CF264, label %CF265

CF265:                                            ; preds = %CF264
  %Cmp188 = icmp uge <16 x i1> %Shuff175, %Cmp
  %L189 = load i8, ptr %0
  store i8 %L19, ptr %0
  %E190 = extractelement <2 x i1> %Cmp11, i32 0
  br i1 %E190, label %CF254, label %CF259

CF259:                                            ; preds = %CF259, %CF265
  %Shuff191 = shufflevector <8 x i1> %Cmp117, <8 x i1> %I30, <8 x i32> <i32 11, i32 13, i32 15, i32 1, i32 3, i32 5, i32 7, i32 9>
  %I192 = insertelement <16 x i1> %Cmp188, i1 %Cmp85, i32 13
  %B193 = urem <2 x i64> %Sl132, %Sl54
  %Tr194 = trunc i64 %E166 to i8
  %Sl195 = select i1 %Cmp93, <2 x i1> %I160, <2 x i1> %Shuff66
  %Cmp196 = icmp ult <2 x i1> %Shuff66, %Cmp11
  %L197 = load i8, ptr %0
  store i8 %L5, ptr %0
  %E198 = extractelement <8 x i64> %Shuff183, i32 0
  %Shuff199 = shufflevector <8 x i16> %I22, <8 x i16> %Tr115, <8 x i32> <i32 3, i32 5, i32 undef, i32 9, i32 11, i32 13, i32 15, i32 undef>
  %I200 = insertelement <16 x i8> %Shuff29, i8 %L197, i32 5
  %B201 = and <2 x i64> %B145, %I113
  %ZE202 = zext <2 x i1> %I74 to <2 x i64>
  %Sl203 = select i1 %Cmp26, i8 %L126, i8 %L102
  %Cmp204 = fcmp oeq <4 x float> %BC53, %BC53
  %L205 = load i8, ptr %0
  store i8 %5, ptr %0
  %E206 = extractelement <2 x double> %Shuff104, i32 0
  %Shuff207 = shufflevector <4 x i64> %I38, <4 x i64> zeroinitializer, <4 x i32> <i32 7, i32 undef, i32 3, i32 5>
  %I208 = insertelement <8 x i64> %I82, i64 323142, i32 1
  %B209 = lshr i8 %L56, %L5
  %FC210 = fptoui double 0xE603EE221901D6A0 to i1
  br i1 %FC210, label %CF259, label %CF263

CF263:                                            ; preds = %CF259
  %Sl211 = select i1 %E174, i32 %ZE, i32 %ZE107
  %Cmp212 = icmp ne i32 %Se154, %Sl163
  br i1 %Cmp212, label %CF254, label %CF257

CF257:                                            ; preds = %CF263
  %L213 = load i8, ptr %0
  store i8 %L213, ptr %0
  %E214 = extractelement <8 x i64> %Shuff81, i32 3
  %Shuff215 = shufflevector <8 x i64> %Shuff159, <8 x i64> %Shuff136, <8 x i32> <i32 14, i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12>
  %I216 = insertelement <8 x i64> %Shuff215, i64 323142, i32 0
  %Se217 = sext i8 %L71 to i64
  %Sl218 = select i1 %Cmp156, <8 x i16> %Tr115, <8 x i16> %Tr115
  %Cmp219 = fcmp ole <2 x float> %FC170, %FC99
  %L220 = load i8, ptr %0
  store i8 %L19, ptr %0
  %E221 = extractelement <8 x i64> zeroinitializer, i32 6
  %Shuff222 = shufflevector <4 x i1> %Cmp204, <4 x i1> %Cmp125, <4 x i32> <i32 1, i32 undef, i32 5, i32 7>
  %I223 = insertelement <8 x i1> %Cmp18, i1 %FC210, i32 3
  %B224 = lshr i32 %E49, %FC24
  %FC225 = sitofp <4 x i1> %Cmp180 to <4 x float>
  %Sl226 = select i1 %Cmp93, i64 %E28, i64 %B23
  %Cmp227 = icmp ugt <4 x i64> zeroinitializer, %B60
  %L228 = load i8, ptr %0
  store i8 %Sl46, ptr %0
  %E229 = extractelement <1 x i32> zeroinitializer, i32 0
  %Shuff230 = shufflevector <16 x i8> %Shuff29, <16 x i8> %I200, <16 x i32> <i32 11, i32 13, i32 15, i32 17, i32 19, i32 21, i32 23, i32 undef, i32 undef, i32 29, i32 31, i32 1, i32 undef, i32 5, i32 undef, i32 9>
  %I231 = insertelement <8 x i64> %Shuff183, i64 %L64, i32 5
  %B232 = fadd float %FC68, %FC68
  %Se233 = sext i1 %Cmp172 to i64
  %Sl234 = select i1 false, i1 %Cmp164, i1 %E43
  br label %CF

CF:                                               ; preds = %CF, %CF257
  %Cmp235 = icmp ule i32 %Sl163, %Sl211
  br i1 %Cmp235, label %CF, label %CF252

CF252:                                            ; preds = %CF252, %CF269, %CF
  %L236 = load i8, ptr %0
  store i8 %L19, ptr %0
  %E237 = extractelement <16 x i1> %Shuff175, i32 15
  br i1 %E237, label %CF252, label %CF269

CF269:                                            ; preds = %CF252
  %Shuff238 = shufflevector <2 x i1> %I160, <2 x i1> %Cmp101, <2 x i32> undef
  %I239 = insertelement <8 x i64> zeroinitializer, i64 %4, i32 0
  %B240 = add i8 %L56, %Sl155
  %Tr241 = trunc <2 x i32> %Sl147 to <2 x i1>
  %Sl242 = select i1 %Sl234, <2 x float> %FC99, <2 x float> %FC99
  %Cmp243 = icmp eq i8 %L5, %L118
  br i1 %Cmp243, label %CF252, label %CF258

CF258:                                            ; preds = %CF258, %CF269
  %L244 = load i8, ptr %0
  store i8 %L19, ptr %0
  %E245 = extractelement <2 x i64> %B201, i32 1
  %Shuff246 = shufflevector <4 x i64> zeroinitializer, <4 x i64> %I144, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %I247 = insertelement <8 x i64> %Shuff73, i64 %E182, i32 2
  %B248 = or i64 %Sl226, %E245
  %Tr249 = trunc <2 x i64> <i64 -1, i64 -1> to <2 x i16>
  %Sl250 = select i1 %FC210, i64 %E57, i64 %L64
  %Cmp251 = icmp eq i32 %FC24, %FC
  br i1 %Cmp251, label %CF258, label %CF267

CF267:                                            ; preds = %CF258
  store i8 %L42, ptr %0
  store i8 %Sl69, ptr %0
  store i8 %L5, ptr %0
  store i8 %L134, ptr %0
  store i8 %L141, ptr %0
  ret void
}
