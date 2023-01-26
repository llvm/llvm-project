; RUN: llc -march=mips < %s
; RUN: llc -march=mips -mattr=+msa,+fp64,+mips32r2 < %s
; RUN: llc -march=mipsel < %s
; RUN: llc -march=mipsel -mattr=+msa,+fp64,+mips32r2 < %s

; This test originally failed for MSA with a
; "Unexpected illegal type!" assertion.
; It should at least successfully build.

define void @autogen_SD1704963983(ptr, ptr, ptr, i32, i64, i8) {
BB:
  %A4 = alloca <4 x double>
  %A3 = alloca <8 x i64>
  %A2 = alloca <1 x double>
  %A1 = alloca double
  %A = alloca i32
  %L = load i8, ptr %0
  store i8 77, ptr %0
  %E = extractelement <8 x i64> zeroinitializer, i32 2
  %Shuff = shufflevector <8 x i64> zeroinitializer, <8 x i64> zeroinitializer, <8 x i32> <i32 5, i32 7, i32 undef, i32 undef, i32 13, i32 15, i32 1, i32 3>
  %I = insertelement <8 x i64> zeroinitializer, i64 %E, i32 7
  %Sl = select i1 false, ptr %0, ptr %0
  %Cmp = icmp eq i32 434069, 272505
  br label %CF

CF:                                               ; preds = %CF, %CF78, %BB
  %L5 = load i8, ptr %Sl
  store i8 %L, ptr %Sl
  %E6 = extractelement <8 x i32> zeroinitializer, i32 2
  %Shuff7 = shufflevector <8 x i64> zeroinitializer, <8 x i64> %Shuff, <8 x i32> <i32 13, i32 15, i32 1, i32 3, i32 5, i32 7, i32 9, i32 undef>
  %I8 = insertelement <8 x i64> zeroinitializer, i64 %4, i32 7
  %B = shl <1 x i16> zeroinitializer, zeroinitializer
  %FC = sitofp <8 x i64> zeroinitializer to <8 x float>
  %Sl9 = select i1 %Cmp, i8 77, i8 77
  %Cmp10 = icmp uge <8 x i64> %Shuff, zeroinitializer
  %L11 = load i8, ptr %0
  store i8 %Sl9, ptr %0
  %E12 = extractelement <1 x i16> zeroinitializer, i32 0
  %Shuff13 = shufflevector <8 x i64> zeroinitializer, <8 x i64> %Shuff, <8 x i32> <i32 9, i32 11, i32 13, i32 15, i32 undef, i32 3, i32 5, i32 7>
  %I14 = insertelement <4 x i32> zeroinitializer, i32 %3, i32 3
  %B15 = udiv <1 x i16> %B, zeroinitializer
  %Tr = trunc <8 x i64> %Shuff to <8 x i32>
  %Sl16 = select i1 %Cmp, i8 77, i8 %5
  %Cmp17 = icmp ult <8 x i1> %Cmp10, %Cmp10
  %L18 = load i8, ptr %Sl
  store i8 -1, ptr %Sl
  %E19 = extractelement <8 x i32> zeroinitializer, i32 3
  %Shuff20 = shufflevector <8 x float> %FC, <8 x float> %FC, <8 x i32> <i32 6, i32 8, i32 undef, i32 12, i32 14, i32 0, i32 2, i32 undef>
  %I21 = insertelement <8 x i64> %Shuff13, i64 %E, i32 0
  %B22 = urem <8 x i64> %Shuff7, %I21
  %FC23 = sitofp i32 50347 to float
  %Sl24 = select i1 %Cmp, double 0.000000e+00, double 0.000000e+00
  %Cmp25 = icmp ugt i32 465489, 47533
  br i1 %Cmp25, label %CF, label %CF78

CF78:                                             ; preds = %CF
  %L26 = load i8, ptr %Sl
  store i32 50347, ptr %A
  %E27 = extractelement <8 x i1> %Cmp10, i32 2
  br i1 %E27, label %CF, label %CF77

CF77:                                             ; preds = %CF77, %CF81, %CF78
  %Shuff28 = shufflevector <8 x i64> zeroinitializer, <8 x i64> %Shuff, <8 x i32> <i32 13, i32 15, i32 1, i32 3, i32 5, i32 7, i32 9, i32 undef>
  %I29 = insertelement <1 x i16> zeroinitializer, i16 -1, i32 0
  %B30 = urem <8 x i32> %Tr, zeroinitializer
  %Tr31 = trunc i32 0 to i16
  %Sl32 = select i1 %Cmp, <2 x i1> zeroinitializer, <2 x i1> zeroinitializer
  %L33 = load i8, ptr %Sl
  store i8 %L26, ptr %Sl
  %E34 = extractelement <4 x i32> zeroinitializer, i32 0
  %Shuff35 = shufflevector <1 x i16> zeroinitializer, <1 x i16> %B, <1 x i32> undef
  %I36 = insertelement <8 x i64> %Shuff28, i64 %E, i32 7
  %B37 = srem <1 x i16> %I29, zeroinitializer
  %FC38 = sitofp <8 x i32> %B30 to <8 x double>
  %Sl39 = select i1 %Cmp, double 0.000000e+00, double %Sl24
  %L40 = load i8, ptr %Sl
  store i8 %Sl16, ptr %Sl
  %E41 = extractelement <1 x i16> zeroinitializer, i32 0
  %Shuff42 = shufflevector <8 x i1> %Cmp17, <8 x i1> %Cmp10, <8 x i32> <i32 14, i32 undef, i32 2, i32 4, i32 undef, i32 8, i32 10, i32 12>
  %I43 = insertelement <4 x i32> zeroinitializer, i32 272505, i32 0
  %B44 = urem <8 x i32> %B30, %Tr
  %Sl45 = select i1 %Cmp, <8 x i1> %Cmp10, <8 x i1> %Shuff42
  %Cmp46 = fcmp ugt float 0xB856238A00000000, 0x47DA795E40000000
  br i1 %Cmp46, label %CF77, label %CF80

CF80:                                             ; preds = %CF80, %CF77
  %L47 = load i64, ptr %0
  store i8 77, ptr %Sl
  %E48 = extractelement <8 x i64> zeroinitializer, i32 2
  %Shuff49 = shufflevector <8 x i64> zeroinitializer, <8 x i64> %Shuff7, <8 x i32> <i32 5, i32 7, i32 9, i32 undef, i32 undef, i32 undef, i32 undef, i32 3>
  %I50 = insertelement <8 x i64> zeroinitializer, i64 %L47, i32 7
  %B51 = fdiv float 0x46CC2D8000000000, %FC23
  %Sl53 = select i1 %Cmp, <8 x i64> %Shuff, <8 x i64> %Shuff
  %Cmp54 = fcmp ole float 0x47DA795E40000000, 0xB856238A00000000
  br i1 %Cmp54, label %CF80, label %CF81

CF81:                                             ; preds = %CF80
  %L55 = load i8, ptr %Sl
  store i8 %Sl16, ptr %Sl
  %E56 = extractelement <1 x i16> %B, i32 0
  %Shuff57 = shufflevector <1 x i16> zeroinitializer, <1 x i16> zeroinitializer, <1 x i32> <i32 1>
  %I58 = insertelement <8 x i64> zeroinitializer, i64 %L47, i32 7
  %B59 = srem i32 %E19, %E19
  %Sl60 = select i1 %Cmp, i8 77, i8 77
  %Cmp61 = icmp ult <1 x i16> zeroinitializer, %B
  %L62 = load i8, ptr %Sl
  store i64 %L47, ptr %A3
  %E63 = extractelement <4 x i32> %I43, i32 2
  %Shuff64 = shufflevector <4 x i1> zeroinitializer, <4 x i1> zeroinitializer, <4 x i32> <i32 undef, i32 undef, i32 1, i32 3>
  %I65 = insertelement <8 x i64> %B22, i64 %L47, i32 7
  %B66 = add <8 x i64> %I50, %I65
  %FC67 = uitofp i16 %E12 to float
  %Sl68 = select i1 %Cmp, <8 x i32> %B30, <8 x i32> zeroinitializer
  %Cmp69 = fcmp ord double 0.000000e+00, 0.000000e+00
  br i1 %Cmp69, label %CF77, label %CF79

CF79:                                             ; preds = %CF81
  %L70 = load i32, ptr %A
  store i64 %4, ptr %0
  %E71 = extractelement <4 x i32> zeroinitializer, i32 0
  %Shuff72 = shufflevector <8 x i32> zeroinitializer, <8 x i32> %B44, <8 x i32> <i32 11, i32 undef, i32 15, i32 1, i32 3, i32 undef, i32 7, i32 9>
  %I73 = insertelement <8 x i16> zeroinitializer, i16 %E12, i32 5
  %B74 = fsub double 0.000000e+00, 0.000000e+00
  %Sl75 = select i1 %Cmp46, i32 %E6, i32 %E19
  %Cmp76 = icmp ugt <4 x i32> %I43, zeroinitializer
  store i8 %L, ptr %Sl
  store i64 %L47, ptr %0
  store i64 %L47, ptr %0
  store i8 %L5, ptr %Sl
  store i8 %L5, ptr %0
  ret void
}
