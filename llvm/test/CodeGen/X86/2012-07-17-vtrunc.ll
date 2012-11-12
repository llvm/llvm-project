; RUN: llc < %s -march=x86-64 -mcpu=corei7

define void @autogen_SD33189483() {
BB:
  br label %CF76

CF76:                                             ; preds = %CF76, %BB
  %Shuff13 = shufflevector <4 x i64> zeroinitializer, <4 x i64> undef, <4 x i32> zeroinitializer
  %Tr16 = trunc <8 x i64> <i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1> to <8 x i1>
  %E19 = extractelement <8 x i1> %Tr16, i32 2
  br i1 %E19, label %CF76, label %CF78

CF78:                                             ; preds = %CF78, %CF76
  %BC = bitcast <4 x i64> %Shuff13 to <4 x double>
  br label %CF78
}
