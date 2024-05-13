; RUN: llc -march=mips < %s
; RUN: llc -march=mips -mattr=+msa,+fp64,+mips32r2 < %s
; RUN: llc -march=mipsel < %s
; RUN: llc -march=mipsel -mattr=+msa,+fp64,+mips32r2 < %s

; This test originally failed for MSA with a
; "Type for zero vector elements is not legal" assertion.
; It should at least successfully build.

define void @autogen_SD3926023935(ptr, ptr, ptr, i32, i64, i8) {
BB:
  %A4 = alloca i1
  %A3 = alloca float
  %A2 = alloca double
  %A1 = alloca float
  %A = alloca double
  %L = load i8, ptr %0
  store i8 -123, ptr %0
  %E = extractelement <4 x i64> zeroinitializer, i32 1
  %Shuff = shufflevector <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>, <4 x i32> zeroinitializer, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %I = insertelement <2 x i1> zeroinitializer, i1 false, i32 0
  %BC = bitcast i64 181325 to double
  %Sl = select i1 false, <2 x i32> zeroinitializer, <2 x i32> zeroinitializer
  %Cmp = icmp ne <4 x i64> zeroinitializer, zeroinitializer
  %L5 = load i8, ptr %0
  store i8 %L, ptr %0
  %E6 = extractelement <4 x i64> zeroinitializer, i32 3
  %Shuff7 = shufflevector <2 x i16> zeroinitializer, <2 x i16> zeroinitializer, <2 x i32> <i32 2, i32 0>
  %I8 = insertelement <8 x i64> <i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1>, i64 498254, i32 4
  %B = shl i32 0, 364464
  %Sl9 = select i1 false, i64 %E, i64 498254
  %Cmp10 = icmp sge i8 -123, %5
  br label %CF80

CF80:                                             ; preds = %BB
  %L11 = load i8, ptr %0
  store i8 -123, ptr %0
  %E12 = extractelement <2 x i16> zeroinitializer, i32 1
  %Shuff13 = shufflevector <4 x i64> zeroinitializer, <4 x i64> zeroinitializer, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %I14 = insertelement <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>, i32 %B, i32 2
  %B15 = sdiv i64 334618, -1
  %Sl16 = select i1 %Cmp10, <4 x i32> zeroinitializer, <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>
  %Cmp17 = icmp ule <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>, %Sl16
  %L18 = load double, ptr %A2
  store i64 498254, ptr %A4
  %E19 = extractelement <4 x i64> zeroinitializer, i32 0
  %Shuff20 = shufflevector <2 x i1> zeroinitializer, <2 x i1> %I, <2 x i32> <i32 3, i32 1>
  %I21 = insertelement <2 x i1> zeroinitializer, i1 false, i32 1
  %B22 = fadd double 0.000000e+00, %BC
  %ZE = zext <2 x i1> %Shuff20 to <2 x i32>
  %Sl23 = select i1 %Cmp10, <2 x i1> %Shuff20, <2 x i1> zeroinitializer
  %Cmp24 = icmp ult <2 x i32> zeroinitializer, zeroinitializer
  %L25 = load i8, ptr %0
  store i8 %L25, ptr %0
  %E26 = extractelement <4 x i8> <i8 -1, i8 -1, i8 -1, i8 -1>, i32 3
  %Shuff27 = shufflevector <4 x i32> %Shuff, <4 x i32> %I14, <4 x i32> <i32 6, i32 0, i32 undef, i32 4>
  %I28 = insertelement <4 x i32> zeroinitializer, i32 %3, i32 0
  %B29 = lshr i8 %E26, -43
  %Tr = trunc i8 %L5 to i1
  br label %CF79

CF79:                                             ; preds = %CF80
  %Sl30 = select i1 false, i8 %B29, i8 -123
  %Cmp31 = icmp sge <2 x i1> %I, %I
  %L32 = load i64, ptr %A4
  store i8 -123, ptr %0
  %E33 = extractelement <8 x i64> <i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1>, i32 2
  %Shuff34 = shufflevector <4 x i64> zeroinitializer, <4 x i64> %Shuff13, <4 x i32> <i32 5, i32 7, i32 1, i32 3>
  %I35 = insertelement <4 x i64> zeroinitializer, i64 498254, i32 3
  %B36 = sub <8 x i64> %I8, <i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1>
  %Sl38 = select i1 %Cmp10, i8 -43, i8 %L5
  %Cmp39 = icmp eq i64 498254, %B15
  br label %CF

CF:                                               ; preds = %CF, %CF79
  %L40 = load double, ptr %A
  store i1 %Cmp39, ptr %0
  %E41 = extractelement <4 x i64> zeroinitializer, i32 3
  %Shuff42 = shufflevector <2 x i32> zeroinitializer, <2 x i32> %ZE, <2 x i32> <i32 2, i32 undef>
  %I43 = insertelement <4 x i32> %Shuff, i32 %3, i32 0
  %B44 = shl i64 %E41, -1
  %Se = sext <2 x i1> %I to <2 x i32>
  %Sl45 = select i1 %Cmp10, i1 false, i1 false
  br i1 %Sl45, label %CF, label %CF77

CF77:                                             ; preds = %CF77, %CF
  %Cmp46 = fcmp uno double 0.000000e+00, 0.000000e+00
  br i1 %Cmp46, label %CF77, label %CF78

CF78:                                             ; preds = %CF78, %CF83, %CF82, %CF77
  %L47 = load i64, ptr %A4
  store i8 -123, ptr %0
  %E48 = extractelement <4 x i64> zeroinitializer, i32 3
  %Shuff49 = shufflevector <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>, <4 x i32> zeroinitializer, <4 x i32> <i32 2, i32 4, i32 6, i32 undef>
  %I50 = insertelement <2 x i1> zeroinitializer, i1 %Cmp10, i32 0
  %B51 = sdiv i64 %E19, 463132
  %Tr52 = trunc i64 %E48 to i32
  %Sl53 = select i1 %Tr, i1 %Cmp46, i1 %Cmp10
  br i1 %Sl53, label %CF78, label %CF83

CF83:                                             ; preds = %CF78
  %Cmp54 = fcmp uge double %L40, %L40
  br i1 %Cmp54, label %CF78, label %CF82

CF82:                                             ; preds = %CF83
  %L55 = load i64, ptr %A4
  store i64 %L32, ptr %A4
  %E56 = extractelement <2 x i16> %Shuff7, i32 1
  %Shuff57 = shufflevector <4 x i64> zeroinitializer, <4 x i64> zeroinitializer, <4 x i32> <i32 2, i32 4, i32 6, i32 0>
  %I58 = insertelement <2 x i32> %Sl, i32 %Tr52, i32 0
  %B59 = or i32 %B, %3
  %FC = sitofp i64 498254 to double
  %Sl60 = select i1 false, i64 %E6, i64 -1
  %Cmp61 = icmp sgt <4 x i32> %Shuff27, %I43
  %L62 = load i64, ptr %A4
  store i64 %Sl9, ptr %A4
  %E63 = extractelement <2 x i32> %ZE, i32 0
  %Shuff64 = shufflevector <4 x i64> zeroinitializer, <4 x i64> %Shuff13, <4 x i32> <i32 1, i32 3, i32 undef, i32 7>
  %I65 = insertelement <4 x i32> %Shuff, i32 %3, i32 3
  %B66 = sub i64 %L47, 53612
  %Tr67 = trunc i64 %4 to i32
  %Sl68 = select i1 %Cmp39, i1 %Cmp39, i1 false
  br i1 %Sl68, label %CF78, label %CF81

CF81:                                             ; preds = %CF82
  %Cmp69 = icmp ne <8 x i64> <i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1>, %B36
  %L70 = load i8, ptr %0
  store i64 %L55, ptr %A4
  %E71 = extractelement <4 x i32> %Shuff49, i32 1
  %Shuff72 = shufflevector <4 x i64> zeroinitializer, <4 x i64> %Shuff34, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %I73 = insertelement <4 x i64> %Shuff64, i64 %E, i32 2
  %B74 = lshr <8 x i64> <i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1>, %B36
  %Sl75 = select i1 %Sl68, i64 %B51, i64 %L55
  %Cmp76 = icmp sgt <8 x i64> %B74, %B36
  store i1 %Cmp39, ptr %0
  store i64 %E41, ptr %A4
  store i64 %L32, ptr %A4
  store i64 %Sl75, ptr %2
  store i64 %L32, ptr %A4
  ret void
}
