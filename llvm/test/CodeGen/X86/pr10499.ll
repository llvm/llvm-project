; RUN: llc < %s -mtriple=x86_64-pc-linux -mcpu=corei7-avx -mattr=-sse2

; No check as PR10499 is a crashing bug.

define void @autogen_24438_500() {
BB:
  %I = insertelement <8 x i32> undef, i32 -1, i32 4
  %BC = bitcast <8 x i32> %I to <8 x float>
  br label %CF

CF:                                               ; preds = %CF, %BB
  %ZE = fpext <8 x float> %BC to <8 x double>
  br label %CF
}
