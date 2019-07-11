; RUN: llc -march=mips < %s
; RUN: llc -march=mips -mattr=+msa,+fp64,+mips32r2 < %s
; RUN: llc -march=mipsel < %s
; RUN: llc -march=mipsel -mattr=+msa,+fp64,+mips32r2 < %s

; This test originally failed for MSA with a
; `Opc && "Cannot copy registers"' assertion.
; It should at least successfully build.

define void @autogen_SD1935737938(i8*, i32*, i64*, i32, i64, i8) {
BB:
  %A4 = alloca i64
  %A3 = alloca <4 x i32>
  %A2 = alloca i64
  %A1 = alloca i32
  %A = alloca <2 x i64>
  %L = load i8, i8* %0
  store i8 -1, i8* %0
  %E = extractelement <2 x i32> zeroinitializer, i32 0
  %Shuff = shufflevector <2 x i32> zeroinitializer, <2 x i32> zeroinitializer, <2 x i32> <i32 1, i32 3>
  %I = insertelement <1 x i64> <i64 -1>, i64 286689, i32 0
  %B = lshr i8 %L, -69
  %ZE = fpext float 0xBF2AA5FE80000000 to double
  %Sl = select i1 true, <1 x i64> <i64 -1>, <1 x i64> <i64 -1>
  %L5 = load i8, i8* %0
  store i8 -69, i8* %0
  %E6 = extractelement <16 x i64> <i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1>, i32 14
  %Shuff7 = shufflevector <2 x i32> zeroinitializer, <2 x i32> zeroinitializer, <2 x i32> <i32 1, i32 3>
  %I8 = insertelement <2 x i32> zeroinitializer, i32 135673, i32 1
  %B9 = udiv i8 %B, %B
  %FC = uitofp i32 %3 to double
  %Sl10 = select i1 true, <1 x i1> zeroinitializer, <1 x i1> zeroinitializer
  %Cmp = icmp ne <1 x i64> %I, <i64 -1>
  %L11 = load i8, i8* %0
  store i8 %L11, i8* %0
  %E12 = extractelement <1 x i64> <i64 -1>, i32 0
  %Shuff13 = shufflevector <1 x i64> %Sl, <1 x i64> <i64 -1>, <1 x i32> <i32 1>
  %I14 = insertelement <1 x i64> %I, i64 303290, i32 0
  %B15 = frem float 0.000000e+00, 0.000000e+00
  %Sl16 = select i1 true, <1 x i1> %Cmp, <1 x i1> zeroinitializer
  %Cmp17 = fcmp one float 0xBD946F9840000000, %B15
  br label %CF74

CF74:                                             ; preds = %CF74, %CF80, %CF76, %BB
  %L18 = load i8, i8* %0
  store i8 -69, i8* %0
  %E19 = extractelement <1 x i64> %Sl, i32 0
  %Shuff20 = shufflevector <8 x i8> <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>, <8 x i8> <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>, <8 x i32> <i32 12, i32 14, i32 0, i32 2, i32 4, i32 6, i32 8, i32 10>
  %I21 = insertelement <2 x i32> %Shuff, i32 135673, i32 0
  %B22 = urem i32 135673, %3
  %FC23 = sitofp i8 %L to float
  %Sl24 = select i1 true, i8 %B, i8 %L18
  %L25 = load i8, i8* %0
  store i8 %L, i8* %0
  %E26 = extractelement <2 x i32> %Shuff, i32 1
  %Shuff27 = shufflevector <2 x i32> zeroinitializer, <2 x i32> zeroinitializer, <2 x i32> <i32 2, i32 0>
  %I28 = insertelement <16 x i64> <i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1>, i64 %E12, i32 8
  %B29 = frem double %ZE, 0x235104F0E94F406E
  %Tr = trunc i64 286689 to i8
  %Sl30 = select i1 true, float 0x45B13EA500000000, float %B15
  %Cmp31 = icmp eq i32 %B22, %B22
  br i1 %Cmp31, label %CF74, label %CF80

CF80:                                             ; preds = %CF74
  %L32 = load i8, i8* %0
  store i8 -1, i8* %0
  %E33 = extractelement <2 x i32> zeroinitializer, i32 1
  %Shuff34 = shufflevector <1 x i64> %Shuff13, <1 x i64> <i64 -1>, <1 x i32> zeroinitializer
  %I35 = insertelement <8 x i8> <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>, i8 -1, i32 0
  %FC36 = sitofp <1 x i1> %Cmp to <1 x float>
  %Sl37 = select i1 true, <8 x i8> %Shuff20, <8 x i8> <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
  %Cmp38 = icmp sgt <2 x i32> %I21, %Shuff27
  %L39 = load i8, i8* %0
  store i8 %Sl24, i8* %0
  %E40 = extractelement <8 x i64> zeroinitializer, i32 1
  %Shuff41 = shufflevector <2 x i1> zeroinitializer, <2 x i1> %Cmp38, <2 x i32> <i32 0, i32 2>
  %I42 = insertelement <4 x i32> zeroinitializer, i32 414573, i32 2
  %B43 = srem i8 %L5, %L39
  %Sl44 = select i1 %Cmp17, i8 %L, i8 %L
  %Cmp45 = fcmp une float 0x3AFCE1A0C0000000, 0.000000e+00
  br i1 %Cmp45, label %CF74, label %CF76

CF76:                                             ; preds = %CF80
  %L46 = load i8, i8* %0
  store i8 %L39, i8* %0
  %E47 = extractelement <2 x i32> %Shuff27, i32 0
  %Shuff48 = shufflevector <1 x i1> %Sl10, <1 x i1> %Sl10, <1 x i32> <i32 1>
  %I49 = insertelement <1 x i64> <i64 -1>, i64 %E12, i32 0
  %FC50 = fptosi double 0x235104F0E94F406E to i32
  %Sl51 = select i1 %Cmp17, <16 x i64> %I28, <16 x i64> %I28
  %Cmp52 = icmp ne i8 %Tr, %Sl24
  br i1 %Cmp52, label %CF74, label %CF75

CF75:                                             ; preds = %CF75, %CF76
  %L53 = load i8, i8* %0
  store i8 %L18, i8* %0
  %E54 = extractelement <8 x i8> %Shuff20, i32 5
  %Shuff55 = shufflevector <2 x i32> %Shuff, <2 x i32> zeroinitializer, <2 x i32> <i32 0, i32 2>
  %I56 = insertelement <4 x i32> %I42, i32 %B22, i32 2
  %B57 = sub i64 %E40, %E6
  %Sl58 = select i1 true, i64 303290, i64 %E40
  %Cmp59 = icmp slt i64 %E40, %E6
  br i1 %Cmp59, label %CF75, label %CF78

CF78:                                             ; preds = %CF75
  %L60 = load i8, i8* %0
  store i8 -69, i8* %0
  %E61 = extractelement <2 x i32> zeroinitializer, i32 0
  %Shuff62 = shufflevector <2 x i32> %Shuff7, <2 x i32> %I21, <2 x i32> <i32 1, i32 3>
  %I63 = insertelement <1 x i1> %Sl16, i1 %Cmp45, i32 0
  %B64 = and i8 %Sl44, -69
  %ZE65 = zext <1 x i1> %Shuff48 to <1 x i64>
  %Sl66 = select i1 true, <1 x i64> %I, <1 x i64> %I49
  %Cmp67 = icmp ugt i64 286689, %E40
  br label %CF

CF:                                               ; preds = %CF, %CF78
  %L68 = load i8, i8* %0
  store i64 %B57, i64* %2
  %E69 = extractelement <2 x i1> %Shuff41, i32 1
  br i1 %E69, label %CF, label %CF77

CF77:                                             ; preds = %CF77, %CF
  %Shuff70 = shufflevector <1 x i64> %Shuff34, <1 x i64> <i64 -1>, <1 x i32> zeroinitializer
  %I71 = insertelement <2 x i32> %Shuff, i32 %E26, i32 0
  %Se = sext i8 %L60 to i32
  %Sl72 = select i1 %Cmp45, <2 x i32> %Shuff62, <2 x i32> %I71
  %Cmp73 = fcmp ugt double 0x235104F0E94F406E, 0x235104F0E94F406E
  br i1 %Cmp73, label %CF77, label %CF79

CF79:                                             ; preds = %CF77
  store i8 %L18, i8* %0
  store i8 %E54, i8* %0
  store i8 %L39, i8* %0
  store i8 %L39, i8* %0
  store i8 %B, i8* %0
  ret void
}
