; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 < %s
target triple = "powerpc64-unknown-linux-gnu"

define void @autogen_SD4932(i8) {
BB:
  %A4 = alloca i8
  %A = alloca <1 x ppc_fp128>
  %Shuff = shufflevector <16 x i32> <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>, <16 x i32> <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>, <16 x i32> <i32 7, i32 9, i32 11, i32 13, i32 15, i32 17, i32 19, i32 21, i32 23, i32 25, i32 undef, i32 29, i32 31, i32 1, i32 3, i32 5>
  br label %CF

CF:                                               ; preds = %CF80, %CF, %BB
  %L5 = load i64, ptr undef
  store i8 %0, ptr %A4
  %Shuff7 = shufflevector <16 x i32> <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>, <16 x i32> %Shuff, <16 x i32> <i32 28, i32 30, i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 undef, i32 20, i32 22, i32 24, i32 26>
  br i1 undef, label %CF, label %CF77

CF77:                                             ; preds = %CF81, %CF83, %CF77, %CF
  br i1 undef, label %CF77, label %CF82

CF82:                                             ; preds = %CF82, %CF77
  %L19 = load i64, ptr undef
  store <1 x ppc_fp128> zeroinitializer, ptr %A
  store i8 -65, ptr %A4
  br i1 undef, label %CF82, label %CF83

CF83:                                             ; preds = %CF82
  %L34 = load i64, ptr undef
  br i1 undef, label %CF77, label %CF81

CF81:                                             ; preds = %CF83
  %Shuff43 = shufflevector <16 x i32> %Shuff7, <16 x i32> undef, <16 x i32> <i32 15, i32 17, i32 19, i32 21, i32 23, i32 undef, i32 undef, i32 29, i32 31, i32 undef, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13>
  store ppc_fp128 0xM00000000000000000000000000000000, ptr %A4
  br i1 undef, label %CF77, label %CF78

CF78:                                             ; preds = %CF78, %CF81
  br i1 undef, label %CF78, label %CF79

CF79:                                             ; preds = %CF79, %CF78
  br i1 undef, label %CF79, label %CF80

CF80:                                             ; preds = %CF79
  store i64 %L19, ptr undef
  %Cmp75 = icmp uge i32 206779, undef
  br i1 %Cmp75, label %CF, label %CF76

CF76:                                             ; preds = %CF80
  store i64 %L5, ptr undef
  store i64 %L34, ptr undef
  ret void
}

define void @autogen_SD88042(ptr, ptr, i8) {
BB:
  %A4 = alloca <2 x i1>
  %A = alloca <16 x float>
  %L = load i8, ptr %0
  %Sl = select i1 false, ptr %A, ptr %A
  %Sl27 = select i1 false, i8 undef, i8 %L
  br label %CF

CF:                                               ; preds = %CF78, %CF, %BB
  br i1 undef, label %CF, label %CF77

CF77:                                             ; preds = %CF80, %CF77, %CF
  store <16 x float> zeroinitializer, ptr %Sl
  %L58 = load i32, ptr %1
  store i8 0, ptr %0
  br i1 undef, label %CF77, label %CF80

CF80:                                             ; preds = %CF77
  store i64 0, ptr %A4
  %E67 = extractelement <8 x i1> zeroinitializer, i32 1
  br i1 %E67, label %CF77, label %CF78

CF78:                                             ; preds = %CF80
  %Cmp73 = icmp eq i32 189865, %L58
  br i1 %Cmp73, label %CF, label %CF76

CF76:                                             ; preds = %CF78
  store i8 %2, ptr %0
  store i8 %Sl27, ptr %0
  ret void
}

define void @autogen_SD37497(ptr, ptr, ptr) {
BB:
  %A1 = alloca i1
  %I8 = insertelement <1 x i32> <i32 -1>, i32 454855, i32 0
  %Cmp = icmp ult <4 x i64> <i64 -1, i64 -1, i64 -1, i64 -1>, undef
  %L10 = load i64, ptr %2
  %E11 = extractelement <4 x i1> %Cmp, i32 2
  br label %CF72

CF72:                                             ; preds = %CF74, %CF72, %BB
  store double 0xB47BB29A53790718, ptr undef
  %E18 = extractelement <1 x i32> <i32 -1>, i32 0
  %FC22 = sitofp <1 x i32> %I8 to <1 x float>
  br i1 undef, label %CF72, label %CF74

CF74:                                             ; preds = %CF72
  store i8 0, ptr %0
  %L31 = load i64, ptr %A1
  store i64 477323, ptr %A1
  %Sl37 = select i1 false, ptr undef, ptr %1
  %Cmp38 = icmp ugt i1 undef, undef
  br i1 %Cmp38, label %CF72, label %CF73

CF73:                                             ; preds = %CF74
  store i64 %L31, ptr %A1
  %B55 = fdiv <1 x float> undef, %FC22
  %Sl63 = select i1 %E11, ptr undef, ptr %Sl37
  store i32 %E18, ptr %Sl63
  store i64 %L10, ptr %A1
  ret void
}
