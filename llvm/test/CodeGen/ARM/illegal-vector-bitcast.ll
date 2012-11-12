; RUN: llc < %s -march=arm
; RUN: llc < %s -mtriple=arm-linux

define void @foo(<8 x float>* %f, <8 x float>* %g, <4 x i64>* %y)
{
  %h = load <8 x float>* %f
  %i = fmul <8 x float> %h, <float 0x3FF19999A0000000, float 0x400A666660000000, float 0x40119999A0000000, float 0x40159999A0000000, float 0.5, float 0x3FE3333340000000, float 0x3FE6666660000000, float 0x3FE99999A0000000>
  %m = bitcast <8 x float> %i to <4 x i64>
  %z = load <4 x i64>* %y
  %n = mul <4 x i64> %z, %m
  %p = bitcast <4 x i64> %n to <8 x float>
  store <8 x float> %p, <8 x float>* %g
  ret void
}
