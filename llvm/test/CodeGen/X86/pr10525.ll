; RUN: llc < %s -march=x86-64 -mattr=+sse2,+sse4.1

; No check in a crash test

define void @autogen_163411_5000() {
BB:
  %L = load <2 x i64>, <2 x i64>* undef
  %Shuff11 = shufflevector <2 x i64> %L, <2 x i64> %L, <2 x i32> <i32 2, i32 0>
  %I51 = insertelement <2 x i64> undef, i64 undef, i32 0
  %Shuff152 = shufflevector <2 x i64> %I51, <2 x i64> %Shuff11, <2 x i32> <i32 1, i32 3>
  store <2 x i64> %Shuff152, <2 x i64>* undef
  ret void
}
