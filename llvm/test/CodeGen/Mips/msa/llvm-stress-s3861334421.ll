; RUN: llc -march=mips < %s
; RUN: llc -march=mips -mattr=+msa,+fp64,+mips32r2 < %s
; RUN: llc -march=mipsel < %s
; RUN: llc -march=mipsel -mattr=+msa,+fp64,+mips32r2 < %s

; This test originally failed for MSA with a
; "Don't know how to expand this condition!" unreachable.
; It should at least successfully build.

define void @autogen_SD3861334421(ptr, ptr, ptr, i32, i64, i8) {
BB:
  %A4 = alloca <2 x i32>
  %A3 = alloca <2 x double>
  %A2 = alloca i64
  %A1 = alloca i64
  %A = alloca double
  %L = load i8, ptr %0
  store i8 -101, ptr %0
  %E = extractelement <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>, i32 0
  %Shuff = shufflevector <8 x i64> zeroinitializer, <8 x i64> zeroinitializer, <8 x i32> <i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 undef, i32 1>
  %I = insertelement <8 x i64> zeroinitializer, i64 %4, i32 5
  %B = and i64 116376, 57247
  %FC = uitofp i8 7 to double
  %Sl = select i1 false, <8 x i8> <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>, <8 x i8> <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
  %L5 = load i8, ptr %0
  store i8 %L, ptr %0
  %E6 = extractelement <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>, i32 3
  %Shuff7 = shufflevector <4 x i64> zeroinitializer, <4 x i64> zeroinitializer, <4 x i32> <i32 2, i32 4, i32 6, i32 0>
  %I8 = insertelement <8 x i8> %Sl, i8 7, i32 4
  %B9 = or <8 x i64> zeroinitializer, zeroinitializer
  %Sl10 = select i1 false, i64 116376, i64 380809
  %Cmp = icmp sgt i32 394647, 17081
  br label %CF

CF:                                               ; preds = %CF, %BB
  %L11 = load i8, ptr %0
  store i8 -87, ptr %0
  %E12 = extractelement <4 x i64> zeroinitializer, i32 0
  %Shuff13 = shufflevector <8 x i64> zeroinitializer, <8 x i64> zeroinitializer, <8 x i32> <i32 7, i32 9, i32 11, i32 13, i32 undef, i32 1, i32 3, i32 5>
  %I14 = insertelement <4 x i64> zeroinitializer, i64 380809, i32 1
  %B15 = srem i64 %Sl10, 380809
  %FC16 = sitofp i64 57247 to float
  %Sl17 = select i1 false, double 0x87A9374869A78EC6, double 0.000000e+00
  %Cmp18 = icmp uge i8 %L, %5
  br i1 %Cmp18, label %CF, label %CF80

CF80:                                             ; preds = %CF80, %CF88, %CF
  %L19 = load i8, ptr %0
  store i8 -101, ptr %0
  %E20 = extractelement <4 x i64> zeroinitializer, i32 0
  %Shuff21 = shufflevector <4 x i64> zeroinitializer, <4 x i64> %Shuff7, <4 x i32> <i32 7, i32 1, i32 3, i32 5>
  %I22 = insertelement <4 x i64> zeroinitializer, i64 127438, i32 1
  %B23 = fdiv double %Sl17, 0.000000e+00
  %Sl24 = select i1 %Cmp18, i32 420510, i32 492085
  %Cmp25 = icmp ugt i1 %Cmp18, false
  br i1 %Cmp25, label %CF80, label %CF83

CF83:                                             ; preds = %CF83, %CF80
  %L26 = load i8, ptr %0
  store i8 -87, ptr %0
  %E27 = extractelement <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>, i32 0
  %Shuff28 = shufflevector <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>, <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>, <4 x i32> <i32 7, i32 1, i32 3, i32 5>
  %I29 = insertelement <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>, i32 492085, i32 1
  %B30 = lshr <8 x i8> %I8, %I8
  %FC31 = sitofp <4 x i32> %Shuff28 to <4 x double>
  %Sl32 = select i1 false, <8 x i8> %I8, <8 x i8> <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
  %Cmp33 = icmp eq i64 %B, 116376
  br i1 %Cmp33, label %CF83, label %CF88

CF88:                                             ; preds = %CF83
  %L34 = load i8, ptr %0
  store i8 -87, ptr %0
  %E35 = extractelement <8 x i64> %Shuff, i32 7
  %Shuff36 = shufflevector <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>, <4 x i32> %Shuff28, <4 x i32> <i32 2, i32 undef, i32 undef, i32 0>
  %I37 = insertelement <4 x i64> zeroinitializer, i64 380809, i32 0
  %B38 = xor <8 x i64> %B9, %B9
  %ZE = zext i32 0 to i64
  %Sl39 = select i1 %Cmp33, i8 %L11, i8 %L5
  %Cmp40 = icmp sgt i1 %Cmp, false
  br i1 %Cmp40, label %CF80, label %CF81

CF81:                                             ; preds = %CF81, %CF85, %CF87, %CF88
  %L41 = load i8, ptr %0
  store i8 %L34, ptr %0
  %E42 = extractelement <8 x i64> %Shuff13, i32 6
  %Shuff43 = shufflevector <4 x i64> zeroinitializer, <4 x i64> zeroinitializer, <4 x i32> <i32 undef, i32 undef, i32 undef, i32 7>
  %I44 = insertelement <4 x i64> zeroinitializer, i64 116376, i32 3
  %B45 = fsub float %FC16, 0x3AC86DCC40000000
  %Tr = trunc <4 x i64> %I14 to <4 x i32>
  %Sl46 = select i1 false, <8 x i64> %B38, <8 x i64> zeroinitializer
  %Cmp47 = icmp sgt i1 %Cmp18, %Cmp18
  br i1 %Cmp47, label %CF81, label %CF85

CF85:                                             ; preds = %CF81
  %L48 = load i8, ptr %0
  store i8 -101, ptr %0
  %E49 = extractelement <8 x i8> <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>, i32 2
  %Shuff50 = shufflevector <4 x i64> zeroinitializer, <4 x i64> zeroinitializer, <4 x i32> <i32 5, i32 7, i32 1, i32 3>
  %I51 = insertelement <4 x i64> zeroinitializer, i64 %E20, i32 3
  %B52 = or i32 336955, %Sl24
  %FC53 = uitofp i8 %L48 to double
  %Sl54 = select i1 %Cmp47, i32 %3, i32 %Sl24
  %Cmp55 = icmp ne <8 x i64> %Shuff13, zeroinitializer
  %L56 = load i8, ptr %0
  store i8 %L11, ptr %0
  %E57 = extractelement <4 x i64> %Shuff21, i32 1
  %Shuff58 = shufflevector <8 x i64> %Shuff, <8 x i64> zeroinitializer, <8 x i32> <i32 4, i32 6, i32 undef, i32 10, i32 12, i32 undef, i32 0, i32 2>
  %I59 = insertelement <4 x i64> zeroinitializer, i64 %E42, i32 2
  %B60 = udiv <8 x i8> %Sl, <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
  %Tr61 = trunc i8 49 to i1
  br i1 %Tr61, label %CF81, label %CF84

CF84:                                             ; preds = %CF84, %CF85
  %Sl62 = select i1 false, i8 %L, i8 %L48
  %Cmp63 = icmp ne <8 x i64> %I, zeroinitializer
  %L64 = load i8, ptr %0
  store i8 %5, ptr %0
  %E65 = extractelement <8 x i1> %Cmp55, i32 0
  br i1 %E65, label %CF84, label %CF87

CF87:                                             ; preds = %CF84
  %Shuff66 = shufflevector <4 x i64> %Shuff21, <4 x i64> %I14, <4 x i32> <i32 3, i32 undef, i32 7, i32 1>
  %I67 = insertelement <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>, i32 %Sl54, i32 1
  %B68 = frem double %B23, %Sl17
  %ZE69 = zext <8 x i8> %Sl32 to <8 x i64>
  %Sl70 = select i1 %Tr61, i64 %E20, i64 %E12
  %Cmp71 = icmp slt <8 x i64> %I, %Shuff
  %L72 = load i8, ptr %0
  store i8 %L72, ptr %0
  %E73 = extractelement <8 x i1> %Cmp55, i32 6
  br i1 %E73, label %CF81, label %CF82

CF82:                                             ; preds = %CF82, %CF87
  %Shuff74 = shufflevector <4 x i32> %I67, <4 x i32> %I29, <4 x i32> <i32 1, i32 3, i32 undef, i32 7>
  %I75 = insertelement <4 x i64> zeroinitializer, i64 380809, i32 3
  %B76 = fsub double 0.000000e+00, %FC53
  %Tr77 = trunc i32 %E to i8
  %Sl78 = select i1 %Cmp18, ptr %A2, ptr %2
  %Cmp79 = icmp eq i32 394647, 492085
  br i1 %Cmp79, label %CF82, label %CF86

CF86:                                             ; preds = %CF82
  store i64 %Sl70, ptr %Sl78
  store i64 %E57, ptr %Sl78
  store i64 %Sl70, ptr %Sl78
  store i64 %B, ptr %Sl78
  store i64 %Sl10, ptr %Sl78
  ret void
}
