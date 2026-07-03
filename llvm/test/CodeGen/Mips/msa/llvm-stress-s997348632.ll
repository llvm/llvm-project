; RUN: llc -mtriple=mips < %s
; RUN: llc -mtriple=mips -mattr=+msa,+fp64,+mips32r2 < %s
; RUN: llc -mtriple=mipsel < %s
; RUN: llc -mtriple=mipsel -mattr=+msa,+fp64,+mips32r2 < %s

; This test originally failed to select instructions for extract_vector_elt for
; v2f64 on MSA.
; It should at least successfully build.

define void @autogen_SD997348632(ptr, ptr, ptr, i32, i64, i8) {
BB:
  %A4 = alloca <2 x i32>
  %A3 = alloca <16 x i16>
  %A2 = alloca <4 x i1>
  %A1 = alloca <4 x i16>
  %A = alloca <2 x i32>
  %L = load i8, ptr %0
  store i8 %L, ptr %0
  %E = extractelement <4 x i32> zeroinitializer, i32 0
  %Shuff = shufflevector <4 x i64> zeroinitializer, <4 x i64> zeroinitializer, <4 x i32> <i32 undef, i32 1, i32 3, i32 5>
  %I = insertelement <2 x i1> zeroinitializer, i1 false, i32 1
  %FC = sitofp <4 x i32> zeroinitializer to <4 x double>
  %Sl = select i1 false, <4 x i64> %Shuff, <4 x i64> %Shuff
  %L5 = load i8, ptr %0
  store i8 %5, ptr %0
  %E6 = extractelement <1 x i16> zeroinitializer, i32 0
  %Shuff7 = shufflevector <2 x i1> %I, <2 x i1> %I, <2 x i32> <i32 1, i32 undef>
  %I8 = insertelement <1 x i16> zeroinitializer, i16 0, i32 0
  %B = xor i32 376034, %3
  %FC9 = fptoui float 0x406DB70180000000 to i64
  %Sl10 = select i1 false, <8 x i32> <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>, <8 x i32> <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>
  %Cmp = icmp ult <4 x i64> zeroinitializer, zeroinitializer
  %L11 = load i8, ptr %0
  store i8 %L, ptr %0
  %E12 = extractelement <4 x i64> zeroinitializer, i32 2
  %Shuff13 = shufflevector <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <4 x i32> <i32 5, i32 7, i32 undef, i32 3>
  %I14 = insertelement <8 x i32> zeroinitializer, i32 -1, i32 7
  %B15 = fdiv <4 x double> %FC, %FC
  %Tr = trunc i32 376034 to i16
  %Sl16 = select i1 false, <8 x i32> %Sl10, <8 x i32> zeroinitializer
  %Cmp17 = icmp uge i32 233658, %E
  br label %CF

CF:                                               ; preds = %CF, %CF79, %CF84, %BB
  %L18 = load i8, ptr %0
  store i8 %L, ptr %0
  %E19 = extractelement <4 x i64> %Sl, i32 3
  %Shuff20 = shufflevector <2 x i1> %Shuff7, <2 x i1> %I, <2 x i32> <i32 2, i32 0>
  %I21 = insertelement <4 x i64> zeroinitializer, i64 %FC9, i32 0
  %B22 = xor <8 x i32> %I14, %I14
  %Tr23 = trunc i16 0 to i8
  %Sl24 = select i1 false, <8 x i32> <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>, <8 x i32> zeroinitializer
  %Cmp25 = icmp eq i1 false, false
  br i1 %Cmp25, label %CF, label %CF79

CF79:                                             ; preds = %CF
  %L26 = load i8, ptr %0
  store i8 %L26, ptr %0
  %E27 = extractelement <1 x i16> zeroinitializer, i32 0
  %Shuff28 = shufflevector <16 x i32> <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>, <16 x i32> <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>, <16 x i32> <i32 13, i32 15, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31, i32 1, i32 3, i32 5, i32 7, i32 9, i32 11>
  %I29 = insertelement <16 x i32> %Shuff28, i32 %B, i32 15
  %B30 = fdiv float 0.000000e+00, -6.749110e+06
  %Sl31 = select i1 false, i32 %3, i32 %3
  %Cmp32 = fcmp uno float 0.000000e+00, 0x406DB70180000000
  br i1 %Cmp32, label %CF, label %CF78

CF78:                                             ; preds = %CF78, %CF79
  %L33 = load i8, ptr %0
  store i8 %L, ptr %0
  %E34 = extractelement <16 x i32> %Shuff28, i32 1
  %Shuff35 = shufflevector <4 x i64> zeroinitializer, <4 x i64> %I21, <4 x i32> <i32 undef, i32 6, i32 0, i32 2>
  %I36 = insertelement <4 x double> %FC, double 0xA4A57F449CA36CC2, i32 2
  %Se = sext <4 x i1> %Cmp to <4 x i32>
  %Sl37 = select i1 %Cmp17, i32 0, i32 0
  %Cmp38 = icmp ne i32 440284, 376034
  br i1 %Cmp38, label %CF78, label %CF80

CF80:                                             ; preds = %CF80, %CF82, %CF78
  %L39 = load i8, ptr %0
  store i8 %L, ptr %0
  %E40 = extractelement <2 x i1> %Shuff20, i32 1
  br i1 %E40, label %CF80, label %CF82

CF82:                                             ; preds = %CF80
  %Shuff41 = shufflevector <2 x i1> zeroinitializer, <2 x i1> %Shuff20, <2 x i32> <i32 2, i32 0>
  %I42 = insertelement <2 x i1> %Shuff41, i1 false, i32 0
  %B43 = sub i32 %E, 0
  %Sl44 = select i1 %Cmp32, <16 x i32> %Shuff28, <16 x i32> %Shuff28
  %Cmp45 = icmp sgt <4 x i64> zeroinitializer, %I21
  %L46 = load i8, ptr %0
  store i8 %L11, ptr %0
  %E47 = extractelement <8 x i32> %Sl16, i32 4
  %Shuff48 = shufflevector <2 x i1> zeroinitializer, <2 x i1> %Shuff7, <2 x i32> <i32 undef, i32 1>
  %I49 = insertelement <2 x i1> %Shuff48, i1 %Cmp17, i32 1
  %B50 = and <8 x i32> %I14, %Sl10
  %FC51 = fptoui float -6.749110e+06 to i1
  br i1 %FC51, label %CF80, label %CF81

CF81:                                             ; preds = %CF81, %CF82
  %Sl52 = select i1 false, float -6.749110e+06, float 0x406DB70180000000
  %Cmp53 = icmp uge <2 x i32> <i32 -1, i32 -1>, <i32 -1, i32 -1>
  %L54 = load i8, ptr %0
  store i8 %L5, ptr %0
  %E55 = extractelement <8 x i32> zeroinitializer, i32 7
  %Shuff56 = shufflevector <4 x i64> zeroinitializer, <4 x i64> zeroinitializer, <4 x i32> <i32 undef, i32 4, i32 6, i32 0>
  %I57 = insertelement <2 x i1> %Shuff7, i1 false, i32 0
  %B58 = fmul <4 x double> %FC, %FC
  %FC59 = fptoui <4 x double> %I36 to <4 x i16>
  %Sl60 = select i1 %Cmp17, <2 x i1> %I, <2 x i1> %I57
  %Cmp61 = icmp ule <8 x i32> %B50, <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>
  %L62 = load i8, ptr %0
  store i8 %L33, ptr %0
  %E63 = extractelement <4 x i64> %Shuff, i32 2
  %Shuff64 = shufflevector <4 x i64> %Shuff56, <4 x i64> %Shuff56, <4 x i32> <i32 5, i32 7, i32 1, i32 undef>
  %I65 = insertelement <2 x i1> zeroinitializer, i1 false, i32 1
  %B66 = sdiv i32 %B, %E55
  %Tr67 = trunc i8 %L54 to i1
  br i1 %Tr67, label %CF81, label %CF83

CF83:                                             ; preds = %CF83, %CF81
  %Sl68 = select i1 %Cmp17, i1 %Cmp25, i1 %Tr67
  br i1 %Sl68, label %CF83, label %CF84

CF84:                                             ; preds = %CF83
  %Cmp69 = icmp uge i32 %E, %E34
  br i1 %Cmp69, label %CF, label %CF77

CF77:                                             ; preds = %CF84
  %L70 = load i8, ptr %0
  store i8 %L, ptr %0
  %E71 = extractelement <4 x i64> %Shuff, i32 0
  %Shuff72 = shufflevector <2 x i1> zeroinitializer, <2 x i1> %I, <2 x i32> <i32 3, i32 1>
  %I73 = insertelement <8 x i32> <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>, i32 %B66, i32 1
  %FC74 = uitofp i1 %Cmp32 to double
  %Sl75 = select i1 %FC51, i16 9704, i16 0
  %Cmp76 = icmp ugt <1 x i16> %I8, %I8
  store i8 %L39, ptr %0
  store i8 %5, ptr %0
  store i8 %Tr23, ptr %0
  store i8 %L, ptr %0
  store i8 %5, ptr %0
  ret void
}
