; RUN: llc -march=mips < %s
; RUN: llc -march=mips -mattr=+msa,+fp64,+mips32r2 < %s
; RUN: llc -march=mipsel < %s
; RUN: llc -march=mipsel -mattr=+msa,+fp64,+mips32r2 < %s

; This test originally failed for MSA with a
; `Num < NumOperands && "Invalid child # of SDNode!"' assertion.
; It should at least successfully build.

define void @autogen_SD525530439(i8*, i32*, i64*, i32, i64, i8) {
BB:
  %A4 = alloca i32
  %A3 = alloca double
  %A2 = alloca <1 x double>
  %A1 = alloca <8 x double>
  %A = alloca i64
  %L = load i8, i8* %0
  store i64 33695, i64* %A
  %E = extractelement <4 x i32> zeroinitializer, i32 3
  %Shuff = shufflevector <2 x i32> <i32 -1, i32 -1>, <2 x i32> <i32 -1, i32 -1>, <2 x i32> <i32 2, i32 0>
  %I = insertelement <4 x i16> zeroinitializer, i16 -11642, i32 0
  %B = lshr <8 x i32> <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>, <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>
  %ZE = fpext float 0x3B64A2B880000000 to double
  %Sl = select i1 true, i16 -1, i16 -11642
  %L5 = load i8, i8* %0
  store i8 0, i8* %0
  %E6 = extractelement <4 x i32> zeroinitializer, i32 2
  %Shuff7 = shufflevector <8 x i1> zeroinitializer, <8 x i1> zeroinitializer, <8 x i32> <i32 undef, i32 7, i32 9, i32 11, i32 13, i32 15, i32 1, i32 undef>
  %I8 = insertelement <4 x i32> zeroinitializer, i32 %3, i32 3
  %B9 = sub i32 71140, 439732
  %BC = bitcast <2 x i32> <i32 -1, i32 -1> to <2 x float>
  %Sl10 = select i1 true, i32* %1, i32* %1
  %Cmp = icmp sge <8 x i64> zeroinitializer, zeroinitializer
  %L11 = load i32, i32* %Sl10
  store <1 x double> zeroinitializer, <1 x double>* %A2
  %E12 = extractelement <4 x i16> zeroinitializer, i32 0
  %Shuff13 = shufflevector <1 x i64> zeroinitializer, <1 x i64> zeroinitializer, <1 x i32> undef
  %I14 = insertelement <1 x i16> zeroinitializer, i16 %Sl, i32 0
  %B15 = or i16 -1, %E12
  %BC16 = bitcast <4 x i32> zeroinitializer to <4 x float>
  %Sl17 = select i1 true, i64 %4, i64 %4
  %Cmp18 = fcmp ugt float 0xC5ABB1BF80000000, 0x3EEF3D6300000000
  br label %CF75

CF75:                                             ; preds = %CF75, %BB
  %L19 = load i32, i32* %Sl10
  store i32 %L11, i32* %Sl10
  %E20 = extractelement <4 x i32> zeroinitializer, i32 1
  %Shuff21 = shufflevector <4 x i32> zeroinitializer, <4 x i32> %I8, <4 x i32> <i32 undef, i32 2, i32 4, i32 6>
  %I22 = insertelement <4 x float> %BC16, float 0x3EEF3D6300000000, i32 2
  %B23 = shl i32 71140, 439732
  %ZE24 = fpext <4 x float> %I22 to <4 x double>
  %Sl25 = select i1 %Cmp18, i32 %L11, i32 %L11
  %Cmp26 = icmp ne i32 %E20, %L19
  br i1 %Cmp26, label %CF75, label %CF76

CF76:                                             ; preds = %CF75
  %L27 = load i32, i32* %Sl10
  store i32 439732, i32* %Sl10
  %E28 = extractelement <4 x i32> %Shuff21, i32 3
  %Shuff29 = shufflevector <8 x i32> <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>, <8 x i32> <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>, <8 x i32> <i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 0>
  %I30 = insertelement <8 x i1> %Shuff7, i1 %Cmp18, i32 4
  %Sl31 = select i1 %Cmp18, i32 %3, i32 %B23
  %Cmp32 = icmp ugt i32 0, %3
  br label %CF74

CF74:                                             ; preds = %CF74, %CF80, %CF78, %CF76
  %L33 = load i64, i64* %2
  store i32 71140, i32* %Sl10
  %E34 = extractelement <4 x i32> zeroinitializer, i32 1
  %Shuff35 = shufflevector <1 x i16> zeroinitializer, <1 x i16> zeroinitializer, <1 x i32> undef
  %I36 = insertelement <4 x i16> zeroinitializer, i16 -11642, i32 0
  %B37 = mul <8 x i32> <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>, %Shuff29
  %Sl38 = select i1 %Cmp18, double 0.000000e+00, double 0x2BA9DB480DA732C6
  %Cmp39 = icmp sgt i16 -11642, %Sl
  br i1 %Cmp39, label %CF74, label %CF80

CF80:                                             ; preds = %CF74
  %L40 = load i8, i8* %0
  store i32 0, i32* %Sl10
  %E41 = extractelement <8 x i64> zeroinitializer, i32 1
  %Shuff42 = shufflevector <1 x i16> %I14, <1 x i16> %I14, <1 x i32> undef
  %I43 = insertelement <4 x i16> %I36, i16 -11642, i32 0
  %FC = fptoui float 0x455CA2B080000000 to i16
  %Sl44 = select i1 %Cmp18, i1 %Cmp18, i1 %Cmp39
  br i1 %Sl44, label %CF74, label %CF78

CF78:                                             ; preds = %CF80
  %L45 = load i32, i32* %Sl10
  store i8 %L5, i8* %0
  %E46 = extractelement <8 x i1> %Shuff7, i32 2
  br i1 %E46, label %CF74, label %CF77

CF77:                                             ; preds = %CF77, %CF78
  %Shuff47 = shufflevector <4 x i16> %I43, <4 x i16> zeroinitializer, <4 x i32> <i32 5, i32 undef, i32 1, i32 3>
  %I48 = insertelement <1 x i16> %Shuff42, i16 %Sl, i32 0
  %B49 = mul i8 0, %L40
  %FC50 = uitofp i32 %3 to double
  %Sl51 = select i1 %Sl44, i32 %L27, i32 0
  %Cmp52 = icmp sge i8 %B49, 0
  br i1 %Cmp52, label %CF77, label %CF79

CF79:                                             ; preds = %CF77
  %L53 = load i32, i32* %Sl10
  store i8 %L40, i8* %0
  %E54 = extractelement <4 x i32> zeroinitializer, i32 1
  %Shuff55 = shufflevector <4 x i32> %Shuff21, <4 x i32> %I8, <4 x i32> <i32 4, i32 6, i32 undef, i32 2>
  %I56 = insertelement <4 x i32> zeroinitializer, i32 %Sl51, i32 2
  %Tr = trunc <1 x i64> %Shuff13 to <1 x i16>
  %Sl57 = select i1 %Cmp18, <2 x i32> <i32 -1, i32 -1>, <2 x i32> <i32 -1, i32 -1>
  %Cmp58 = icmp uge <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>, %I56
  %L59 = load i8, i8* %0
  store <1 x double> zeroinitializer, <1 x double>* %A2
  %E60 = extractelement <4 x i32> zeroinitializer, i32 0
  %Shuff61 = shufflevector <4 x i32> %I8, <4 x i32> %I8, <4 x i32> <i32 undef, i32 1, i32 undef, i32 undef>
  %I62 = insertelement <4 x i16> zeroinitializer, i16 %E12, i32 1
  %B63 = and <4 x i32> %Shuff61, <i32 -1, i32 -1, i32 -1, i32 -1>
  %PC = bitcast double* %A3 to i32*
  %Sl64 = select i1 %Cmp18, <4 x i32> %Shuff61, <4 x i32> %Shuff55
  %Cmp65 = icmp sgt i32 439732, %3
  br label %CF

CF:                                               ; preds = %CF79
  %L66 = load i32, i32* %Sl10
  store i32 %E6, i32* %PC
  %E67 = extractelement <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>, i32 2
  %Shuff68 = shufflevector <4 x i32> %Sl64, <4 x i32> %I8, <4 x i32> <i32 5, i32 undef, i32 1, i32 undef>
  %I69 = insertelement <4 x i16> %Shuff47, i16 %Sl, i32 3
  %B70 = sdiv <4 x i64> zeroinitializer, zeroinitializer
  %FC71 = sitofp i32 %L66 to double
  %Sl72 = select i1 %Cmp18, i64 %4, i64 %4
  %Cmp73 = icmp eq <4 x i64> zeroinitializer, %B70
  store i32 %B23, i32* %PC
  store i32 %3, i32* %PC
  store i32 %3, i32* %Sl10
  store i32 %L27, i32* %1
  store i32 0, i32* %PC
  ret void
}
