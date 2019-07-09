; RUN: llc -march=mips < %s
; RUN: llc -march=mips -mattr=+msa,+fp64,+mips32r2 < %s
; RUN: llc -march=mipsel < %s
; RUN: llc -march=mipsel -mattr=+msa,+fp64,+mips32r2 < %s

; This test originally failed to select instructions for extract_vector_elt for
; v4f32 on MSA.
; It should at least successfully build.

define void @autogen_SD3997499501(i8*, i32*, i64*, i32, i64, i8) {
BB:
  %A4 = alloca <1 x double>
  %A3 = alloca double
  %A2 = alloca float
  %A1 = alloca double
  %A = alloca double
  %L = load i8, i8* %0
  store i8 97, i8* %0
  %E = extractelement <16 x i64> <i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1>, i32 14
  %Shuff = shufflevector <2 x i1> zeroinitializer, <2 x i1> zeroinitializer, <2 x i32> <i32 1, i32 3>
  %I = insertelement <4 x i64> zeroinitializer, i64 0, i32 3
  %Tr = trunc <1 x i64> zeroinitializer to <1 x i8>
  %Sl = select i1 false, double* %A1, double* %A
  %Cmp = icmp ne <2 x i64> zeroinitializer, zeroinitializer
  %L5 = load double, double* %Sl
  store float -4.374162e+06, float* %A2
  %E6 = extractelement <4 x i64> zeroinitializer, i32 3
  %Shuff7 = shufflevector <4 x i64> zeroinitializer, <4 x i64> %I, <4 x i32> <i32 2, i32 4, i32 6, i32 undef>
  %I8 = insertelement <2 x i1> %Shuff, i1 false, i32 0
  %B = ashr <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>, <i32 -1, i32 -1, i32 -1, i32 -1>
  %PC = bitcast float* %A2 to float*
  %Sl9 = select i1 false, i32 82299, i32 0
  %Cmp10 = icmp slt i8 97, %5
  br label %CF72

CF72:                                             ; preds = %CF72, %CF80, %CF78, %BB
  %L11 = load double, double* %Sl
  store double 0.000000e+00, double* %Sl
  %E12 = extractelement <2 x i1> zeroinitializer, i32 0
  br i1 %E12, label %CF72, label %CF80

CF80:                                             ; preds = %CF72
  %Shuff13 = shufflevector <2 x i1> zeroinitializer, <2 x i1> zeroinitializer, <2 x i32> <i32 3, i32 1>
  %I14 = insertelement <2 x i64> zeroinitializer, i64 %4, i32 1
  %B15 = fadd double %L5, 0.000000e+00
  %BC = bitcast i32 0 to float
  %Sl16 = select i1 %E12, float 0xC7957ED940000000, float %BC
  %Cmp17 = icmp eq i32 136082, 471909
  br i1 %Cmp17, label %CF72, label %CF77

CF77:                                             ; preds = %CF77, %CF80
  %L18 = load double, double* %Sl
  store double 0.000000e+00, double* %Sl
  %E19 = extractelement <2 x i1> zeroinitializer, i32 0
  br i1 %E19, label %CF77, label %CF78

CF78:                                             ; preds = %CF77
  %Shuff20 = shufflevector <2 x i1> zeroinitializer, <2 x i1> zeroinitializer, <2 x i32> <i32 1, i32 3>
  %I21 = insertelement <8 x i1> zeroinitializer, i1 %Cmp10, i32 7
  %B22 = sdiv <4 x i64> %Shuff7, zeroinitializer
  %FC = uitofp i8 97 to double
  %Sl23 = select i1 %Cmp10, <2 x i1> zeroinitializer, <2 x i1> zeroinitializer
  %L24 = load double, double* %Sl
  store float %Sl16, float* %PC
  %E25 = extractelement <2 x i1> %Shuff, i32 1
  br i1 %E25, label %CF72, label %CF76

CF76:                                             ; preds = %CF78
  %Shuff26 = shufflevector <4 x i64> zeroinitializer, <4 x i64> %B22, <4 x i32> <i32 undef, i32 undef, i32 0, i32 undef>
  %I27 = insertelement <4 x i64> zeroinitializer, i64 %E, i32 2
  %B28 = mul <4 x i64> %I27, zeroinitializer
  %ZE = zext <8 x i1> zeroinitializer to <8 x i64>
  %Sl29 = select i1 %Cmp17, float -4.374162e+06, float -4.374162e+06
  %L30 = load i8, i8* %0
  store double %L5, double* %Sl
  %E31 = extractelement <8 x i1> zeroinitializer, i32 5
  br label %CF

CF:                                               ; preds = %CF, %CF81, %CF76
  %Shuff32 = shufflevector <16 x i64> <i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1>, <16 x i64> <i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1>, <16 x i32> <i32 8, i32 undef, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 undef, i32 26, i32 28, i32 30, i32 undef, i32 2, i32 4, i32 6>
  %I33 = insertelement <8 x i1> zeroinitializer, i1 false, i32 2
  %BC34 = bitcast <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1> to <4 x float>
  %Sl35 = select i1 %E12, <2 x i1> %I8, <2 x i1> zeroinitializer
  %Cmp36 = fcmp oge double 0xC2C3BAE2D5C18360, 0xC2C3BAE2D5C18360
  br i1 %Cmp36, label %CF, label %CF74

CF74:                                             ; preds = %CF74, %CF
  %L37 = load float, float* %PC
  store double 0.000000e+00, double* %Sl
  %E38 = extractelement <2 x i1> %Sl23, i32 1
  br i1 %E38, label %CF74, label %CF75

CF75:                                             ; preds = %CF75, %CF82, %CF74
  %Shuff39 = shufflevector <2 x i1> %Shuff13, <2 x i1> zeroinitializer, <2 x i32> <i32 undef, i32 2>
  %I40 = insertelement <4 x i64> zeroinitializer, i64 %4, i32 2
  %Sl41 = select i1 %Cmp10, i32 0, i32 %3
  %Cmp42 = icmp ne <1 x i64> zeroinitializer, zeroinitializer
  %L43 = load double, double* %Sl
  store i64 %4, i64* %2
  %E44 = extractelement <2 x i1> %Shuff20, i32 1
  br i1 %E44, label %CF75, label %CF82

CF82:                                             ; preds = %CF75
  %Shuff45 = shufflevector <2 x i1> %Sl23, <2 x i1> %Sl23, <2 x i32> <i32 2, i32 0>
  %I46 = insertelement <4 x i64> zeroinitializer, i64 0, i32 0
  %B47 = sub i64 %E, %E6
  %Sl48 = select i1 %Cmp10, double %L5, double %L43
  %Cmp49 = icmp uge i64 %4, %B47
  br i1 %Cmp49, label %CF75, label %CF81

CF81:                                             ; preds = %CF82
  %L50 = load i8, i8* %0
  store double %L43, double* %Sl
  %E51 = extractelement <4 x i64> %Shuff7, i32 3
  %Shuff52 = shufflevector <4 x float> %BC34, <4 x float> %BC34, <4 x i32> <i32 2, i32 4, i32 6, i32 0>
  %I53 = insertelement <2 x i1> %Cmp, i1 %E25, i32 0
  %B54 = fdiv double %L24, %L43
  %BC55 = bitcast <4 x i64> zeroinitializer to <4 x double>
  %Sl56 = select i1 false, i8 %5, i8 97
  %L57 = load i8, i8* %0
  store i8 %L50, i8* %0
  %E58 = extractelement <2 x i1> %Shuff20, i32 1
  br i1 %E58, label %CF, label %CF73

CF73:                                             ; preds = %CF73, %CF81
  %Shuff59 = shufflevector <2 x i1> %Shuff13, <2 x i1> %Shuff45, <2 x i32> <i32 undef, i32 0>
  %I60 = insertelement <4 x float> %Shuff52, float -4.374162e+06, i32 0
  %B61 = mul <4 x i64> %I46, zeroinitializer
  %PC62 = bitcast double* %A3 to float*
  %Sl63 = select i1 %Cmp10, <1 x i64> zeroinitializer, <1 x i64> zeroinitializer
  %Cmp64 = icmp ne <2 x i1> %Cmp, %Shuff
  %L65 = load double, double* %A1
  store float -4.374162e+06, float* %PC62
  %E66 = extractelement <8 x i1> %I21, i32 3
  br i1 %E66, label %CF73, label %CF79

CF79:                                             ; preds = %CF79, %CF73
  %Shuff67 = shufflevector <8 x i1> %I21, <8 x i1> %I21, <8 x i32> <i32 6, i32 8, i32 10, i32 12, i32 14, i32 0, i32 undef, i32 4>
  %I68 = insertelement <1 x i1> %Cmp42, i1 %E25, i32 0
  %B69 = sdiv <16 x i64> <i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1>, <i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1>
  %Sl70 = select i1 %Cmp49, <2 x i1> %Sl23, <2 x i1> %Shuff45
  %Cmp71 = icmp ne i1 false, false
  br i1 %Cmp71, label %CF79, label %CF83

CF83:                                             ; preds = %CF79
  store double 0.000000e+00, double* %Sl
  store float %BC, float* %PC62
  store double %Sl48, double* %Sl
  store double %FC, double* %Sl
  store float %BC, float* %PC62
  ret void
}
