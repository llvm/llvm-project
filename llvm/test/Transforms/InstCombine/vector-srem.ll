; RUN: opt < %s -instcombine -S | grep "srem <4 x i32>"

define <4 x i32> @foo(<4 x i32> %t, <4 x i32> %u)
{
  %k = sdiv <4 x i32> %t, %u
  %l = mul <4 x i32> %k, %u
  %m = sub <4 x i32> %t, %l
  ret <4 x i32> %m
}
