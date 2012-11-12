; RUN: llc -mtriple=armv7-- < %s -mattr=-neon

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:64:128-a0:0:64-n32"
target triple = "armv7-none-linux-gnueabi"

@x1 = common global <3 x i16> zeroinitializer
@y1 = common global <3 x i16> zeroinitializer
@z1 = common global <3 x i16> zeroinitializer
@x2 = common global <4 x i16> zeroinitializer
@y2 = common global <4 x i16> zeroinitializer
@z2 = common global <4 x i16> zeroinitializer

define void @f() {
  %1 = load <3 x i16>* @x1
  %2 = load <3 x i16>* @y1
  %3 = sdiv <3 x i16> %1, %2
  store <3 x i16> %3, <3 x i16>* @z1
  %4 = load <4 x i16>* @x2
  %5 = load <4 x i16>* @y2
  %6 = sdiv <4 x i16> %4, %5
  store <4 x i16> %6, <4 x i16>* @z2
  ret void
}
