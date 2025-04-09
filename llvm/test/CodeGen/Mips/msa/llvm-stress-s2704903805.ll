; RUN: llc -mtriple=mips < %s
; RUN: llc -mtriple=mips -mattr=+msa,+fp64,+mips32r2 < %s
; RUN: llc -mtriple=mipsel < %s
; RUN: llc -mtriple=mipsel -mattr=+msa,+fp64,+mips32r2 < %s

; This test originally failed for MSA after dereferencing a null this pointer.
; It should at least successfully build.

define void @autogen_SD2704903805(ptr, ptr, ptr, i32, i64, i8) {
BB:
  %A4 = alloca i32
  %A3 = alloca i32
  %A2 = alloca i8
  %A1 = alloca i32
  %A = alloca i8
  %L = load i8, ptr %0
  store i8 %5, ptr %0
  %E = extractelement <2 x i16> zeroinitializer, i32 0
  %Shuff = shufflevector <1 x i8> <i8 -1>, <1 x i8> <i8 -1>, <1 x i32> undef
  %I = insertelement <1 x i8> <i8 -1>, i8 85, i32 0
  %B = lshr <2 x i16> zeroinitializer, zeroinitializer
  %FC = sitofp <4 x i16> zeroinitializer to <4 x float>
  %Sl = select i1 true, float 0.000000e+00, float 0x401E76A240000000
  %Cmp = icmp ule i16 -25210, %E
  br label %CF83

CF83:                                             ; preds = %BB
  %L5 = load i8, ptr %0
  store i8 85, ptr %0
  %E6 = extractelement <1 x i8> <i8 -1>, i32 0
  %Shuff7 = shufflevector <2 x i16> zeroinitializer, <2 x i16> zeroinitializer, <2 x i32> <i32 1, i32 3>
  %I8 = insertelement <4 x i16> zeroinitializer, i16 %E, i32 3
  %B9 = ashr <2 x i16> %Shuff7, zeroinitializer
  %FC10 = sitofp i32 -1 to float
  %Sl11 = select i1 %Cmp, i32 -1, i32 -1
  %Cmp12 = icmp sgt i32 -1, -1
  br label %CF

CF:                                               ; preds = %CF, %CF81, %CF83
  %L13 = load i8, ptr %0
  store i8 0, ptr %0
  %E14 = extractelement <2 x i64> zeroinitializer, i32 0
  %Shuff15 = shufflevector <4 x i64> <i64 -1, i64 -1, i64 -1, i64 -1>, <4 x i64> <i64 -1, i64 -1, i64 -1, i64 -1>, <4 x i32> <i32 3, i32 5, i32 7, i32 undef>
  %I16 = insertelement <4 x i64> <i64 -1, i64 -1, i64 -1, i64 -1>, i64 81222, i32 1
  %B17 = lshr <2 x i16> zeroinitializer, %B
  %Tr = trunc i32 272597 to i1
  br i1 %Tr, label %CF, label %CF80

CF80:                                             ; preds = %CF80, %CF
  %Sl18 = select i1 %Cmp, <2 x i64> zeroinitializer, <2 x i64> zeroinitializer
  %Cmp19 = icmp ne i1 %Cmp12, %Cmp
  br i1 %Cmp19, label %CF80, label %CF81

CF81:                                             ; preds = %CF80
  %L20 = load i8, ptr %0
  store i8 85, ptr %0
  %E21 = extractelement <1 x i8> <i8 -1>, i32 0
  %Shuff22 = shufflevector <1 x i8> <i8 -1>, <1 x i8> %Shuff, <1 x i32> zeroinitializer
  %I23 = insertelement <1 x i8> <i8 -1>, i8 %L5, i32 0
  %FC24 = fptoui <4 x float> %FC to <4 x i16>
  %Sl25 = select i1 %Cmp, <2 x i32> zeroinitializer, <2 x i32> <i32 -1, i32 -1>
  %Cmp26 = icmp ult <4 x i64> %I16, %Shuff15
  %L27 = load i8, ptr %0
  store i8 %L, ptr %0
  %E28 = extractelement <1 x i8> <i8 -1>, i32 0
  %Shuff29 = shufflevector <8 x i16> zeroinitializer, <8 x i16> zeroinitializer, <8 x i32> <i32 11, i32 undef, i32 15, i32 1, i32 3, i32 5, i32 undef, i32 9>
  %I30 = insertelement <4 x i64> <i64 -1, i64 -1, i64 -1, i64 -1>, i64 %E14, i32 1
  %B31 = mul i8 %E28, 85
  %Sl32 = select i1 %Cmp12, float %FC10, float 0x4712BFE680000000
  %L33 = load i32, ptr %A3
  store i32 %L33, ptr %A3
  %E34 = extractelement <2 x i16> zeroinitializer, i32 1
  %Shuff35 = shufflevector <1 x i8> %Shuff, <1 x i8> <i8 -1>, <1 x i32> zeroinitializer
  %I36 = insertelement <1 x i8> <i8 -1>, i8 %L13, i32 0
  %B37 = xor i8 %L27, %L
  %Sl38 = select i1 %Cmp, i16 %E34, i16 %E
  %Cmp39 = icmp eq i1 %Cmp19, %Cmp
  br i1 %Cmp39, label %CF, label %CF77

CF77:                                             ; preds = %CF77, %CF81
  %L40 = load i32, ptr %A3
  store i32 %3, ptr %A3
  %E41 = extractelement <2 x i32> zeroinitializer, i32 0
  %Shuff42 = shufflevector <2 x i32> <i32 -1, i32 -1>, <2 x i32> zeroinitializer, <2 x i32> <i32 1, i32 3>
  %I43 = insertelement <1 x i8> <i8 -1>, i8 0, i32 0
  %B44 = or i16 %E, -25210
  %Se = sext i32 %3 to i64
  %Sl45 = select i1 true, <1 x i8> %Shuff, <1 x i8> %I43
  %Cmp46 = icmp sge <1 x i8> %I36, %Shuff
  %L47 = load i32, ptr %A3
  store i32 %L33, ptr %A3
  %E48 = extractelement <2 x i16> zeroinitializer, i32 0
  %Shuff49 = shufflevector <1 x i8> <i8 -1>, <1 x i8> <i8 -1>, <1 x i32> <i32 1>
  %I50 = insertelement <2 x i32> %Sl25, i32 47963, i32 1
  %B51 = srem <1 x i8> %I, %Shuff22
  %FC52 = sitofp i8 %5 to double
  %Sl53 = select i1 %Cmp39, i8 %L27, i8 85
  %Cmp54 = icmp slt i16 %E34, %E34
  br i1 %Cmp54, label %CF77, label %CF78

CF78:                                             ; preds = %CF78, %CF77
  %L55 = load i32, ptr %A3
  store i32 %L33, ptr %A3
  %E56 = extractelement <8 x i16> %Shuff29, i32 4
  %Shuff57 = shufflevector <1 x i8> <i8 -1>, <1 x i8> <i8 -1>, <1 x i32> <i32 1>
  %I58 = insertelement <1 x i8> %B51, i8 %Sl53, i32 0
  %ZE = fpext float %FC10 to double
  %Sl59 = select i1 %Cmp12, <2 x i16> %B9, <2 x i16> zeroinitializer
  %Cmp60 = fcmp ult double 0.000000e+00, 0.000000e+00
  br i1 %Cmp60, label %CF78, label %CF79

CF79:                                             ; preds = %CF79, %CF78
  %L61 = load i32, ptr %A3
  store i32 %L33, ptr %A3
  %E62 = extractelement <4 x i64> %Shuff15, i32 1
  %Shuff63 = shufflevector <8 x i16> %Shuff29, <8 x i16> %Shuff29, <8 x i32> <i32 undef, i32 10, i32 12, i32 undef, i32 undef, i32 undef, i32 4, i32 6>
  %I64 = insertelement <2 x i64> zeroinitializer, i64 %Se, i32 0
  %B65 = shl i8 %5, 85
  %ZE66 = zext <4 x i1> %Cmp26 to <4 x i32>
  %Sl67 = select i1 %Tr, <1 x i8> %Shuff, <1 x i8> %I23
  %Cmp68 = fcmp olt float 0x4712BFE680000000, 0x4712BFE680000000
  br i1 %Cmp68, label %CF79, label %CF82

CF82:                                             ; preds = %CF79
  %L69 = load i32, ptr %A3
  store i32 %L33, ptr %A3
  %E70 = extractelement <8 x i16> zeroinitializer, i32 3
  %Shuff71 = shufflevector <4 x i64> %Shuff15, <4 x i64> <i64 -1, i64 -1, i64 -1, i64 -1>, <4 x i32> <i32 6, i32 undef, i32 2, i32 4>
  %I72 = insertelement <1 x i8> <i8 -1>, i8 %L, i32 0
  %B73 = srem i64 %E62, %Se
  %ZE74 = zext <4 x i1> %Cmp26 to <4 x i32>
  %Sl75 = select i1 %Cmp, i32 463279, i32 %L61
  %Cmp76 = icmp sgt <1 x i8> %Shuff49, %Shuff22
  store i8 %B31, ptr %0
  store i8 85, ptr %0
  store i32 %L33, ptr %A3
  store i8 %B65, ptr %0
  store i8 %L5, ptr %0
  ret void
}
