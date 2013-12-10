; RUN: llc < %s -verify-machineinstrs -mtriple=aarch64-none-linux-gnu -mattr=+neon | FileCheck %s

define <8 x i8> @movi8b() {
;CHECK:  movi {{v[0-31]+}}.8b, #0x8
   ret <8 x i8> < i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8 >
}

define <16 x i8> @movi16b() {
;CHECK:  movi {{v[0-31]+}}.16b, #0x8
   ret <16 x i8> < i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8 >
}

define <2 x i32> @movi2s_lsl0() {
;CHECK:  movi {{v[0-31]+}}.2s, #0xff
   ret <2 x i32> < i32 255, i32 255 >
}

define <2 x i32> @movi2s_lsl8() {
;CHECK:  movi {{v[0-31]+}}.2s, #0xff, lsl #8
   ret <2 x i32> < i32 65280, i32 65280 >
}

define <2 x i32> @movi2s_lsl16() {
;CHECK:  movi {{v[0-31]+}}.2s, #0xff, lsl #16
   ret <2 x i32> < i32 16711680, i32 16711680 >

}

define <2 x i32> @movi2s_lsl24() {
;CHECK:  movi {{v[0-31]+}}.2s, #0xff, lsl #24
   ret <2 x i32> < i32 4278190080, i32 4278190080 >
}

define <4 x i32> @movi4s_lsl0() {
;CHECK:  movi {{v[0-31]+}}.4s, #0xff
   ret <4 x i32> < i32 255, i32 255, i32 255, i32 255 >
}

define <4 x i32> @movi4s_lsl8() {
;CHECK:  movi {{v[0-31]+}}.4s, #0xff, lsl #8
   ret <4 x i32> < i32 65280, i32 65280, i32 65280, i32 65280 >
}

define <4 x i32> @movi4s_lsl16() {
;CHECK:  movi {{v[0-31]+}}.4s, #0xff, lsl #16
   ret <4 x i32> < i32 16711680, i32 16711680, i32 16711680, i32 16711680 >

}

define <4 x i32> @movi4s_lsl24() {
;CHECK:  movi {{v[0-31]+}}.4s, #0xff, lsl #24
   ret <4 x i32> < i32 4278190080, i32 4278190080, i32 4278190080, i32 4278190080 >
}

define <4 x i16> @movi4h_lsl0() {
;CHECK:  movi {{v[0-31]+}}.4h, #0xff
   ret <4 x i16> < i16 255, i16 255, i16 255, i16 255 >
}

define <4 x i16> @movi4h_lsl8() {
;CHECK:  movi {{v[0-31]+}}.4h, #0xff, lsl #8
   ret <4 x i16> < i16 65280, i16 65280, i16 65280, i16 65280 >
}

define <8 x i16> @movi8h_lsl0() {
;CHECK:  movi {{v[0-31]+}}.8h, #0xff
   ret <8 x i16> < i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255 >
}

define <8 x i16> @movi8h_lsl8() {
;CHECK:  movi {{v[0-31]+}}.8h, #0xff, lsl #8
   ret <8 x i16> < i16 65280, i16 65280, i16 65280, i16 65280, i16 65280, i16 65280, i16 65280, i16 65280 >
}


define <2 x i32> @mvni2s_lsl0() {
;CHECK:  mvni {{v[0-31]+}}.2s, #0x10
   ret <2 x i32> < i32 4294967279, i32 4294967279 >
}

define <2 x i32> @mvni2s_lsl8() {
;CHECK:  mvni {{v[0-31]+}}.2s, #0x10, lsl #8
   ret <2 x i32> < i32 4294963199, i32 4294963199 >
}

define <2 x i32> @mvni2s_lsl16() {
;CHECK:  mvni {{v[0-31]+}}.2s, #0x10, lsl #16
   ret <2 x i32> < i32 4293918719, i32 4293918719 >
}

define <2 x i32> @mvni2s_lsl24() {
;CHECK:  mvni {{v[0-31]+}}.2s, #0x10, lsl #24
   ret <2 x i32> < i32 4026531839, i32 4026531839 >
}

define <4 x i32> @mvni4s_lsl0() {
;CHECK:  mvni {{v[0-31]+}}.4s, #0x10
   ret <4 x i32> < i32 4294967279, i32 4294967279, i32 4294967279, i32 4294967279 >
}

define <4 x i32> @mvni4s_lsl8() {
;CHECK:  mvni {{v[0-31]+}}.4s, #0x10, lsl #8
   ret <4 x i32> < i32 4294963199, i32 4294963199, i32 4294963199, i32 4294963199 >
}

define <4 x i32> @mvni4s_lsl16() {
;CHECK:  mvni {{v[0-31]+}}.4s, #0x10, lsl #16
   ret <4 x i32> < i32 4293918719, i32 4293918719, i32 4293918719, i32 4293918719 >

}

define <4 x i32> @mvni4s_lsl24() {
;CHECK:  mvni {{v[0-31]+}}.4s, #0x10, lsl #24
   ret <4 x i32> < i32 4026531839, i32 4026531839, i32 4026531839, i32 4026531839 >
}


define <4 x i16> @mvni4h_lsl0() {
;CHECK:  mvni {{v[0-31]+}}.4h, #0x10
   ret <4 x i16> < i16 65519, i16 65519, i16 65519, i16 65519 >
}

define <4 x i16> @mvni4h_lsl8() {
;CHECK:  mvni {{v[0-31]+}}.4h, #0x10, lsl #8
   ret <4 x i16> < i16 61439, i16 61439, i16 61439, i16 61439 >
}

define <8 x i16> @mvni8h_lsl0() {
;CHECK:  mvni {{v[0-31]+}}.8h, #0x10
   ret <8 x i16> < i16 65519, i16 65519, i16 65519, i16 65519, i16 65519, i16 65519, i16 65519, i16 65519 >
}

define <8 x i16> @mvni8h_lsl8() {
;CHECK:  mvni {{v[0-31]+}}.8h, #0x10, lsl #8
   ret <8 x i16> < i16 61439, i16 61439, i16 61439, i16 61439, i16 61439, i16 61439, i16 61439, i16 61439 >
}


define <2 x i32> @movi2s_msl8(<2 x i32> %a) {
;CHECK:  movi {{v[0-31]+}}.2s, #0xff, msl #8
	ret <2 x i32> < i32 65535, i32 65535 >
}

define <2 x i32> @movi2s_msl16() {
;CHECK:  movi {{v[0-31]+}}.2s, #0xff, msl #16
   ret <2 x i32> < i32 16777215, i32 16777215 >
}


define <4 x i32> @movi4s_msl8() {
;CHECK:  movi {{v[0-31]+}}.4s, #0xff, msl #8
   ret <4 x i32> < i32 65535, i32 65535, i32 65535, i32 65535 >
}

define <4 x i32> @movi4s_msl16() {
;CHECK:  movi {{v[0-31]+}}.4s, #0xff, msl #16
   ret <4 x i32> < i32 16777215, i32 16777215, i32 16777215, i32 16777215 >
}

define <2 x i32> @mvni2s_msl8() {
;CHECK:  mvni {{v[0-31]+}}.2s, #0x10, msl #8
   ret <2 x i32> < i32 18446744073709547264, i32 18446744073709547264>
}

define <2 x i32> @mvni2s_msl16() {
;CHECK:  mvni {{v[0-31]+}}.2s, #0x10, msl #16
   ret <2 x i32> < i32 18446744073708437504, i32 18446744073708437504>
}

define <4 x i32> @mvni4s_msl8() {
;CHECK:  mvni {{v[0-31]+}}.4s, #0x10, msl #8
   ret <4 x i32> < i32 18446744073709547264, i32 18446744073709547264, i32 18446744073709547264, i32 18446744073709547264>
}

define <4 x i32> @mvni4s_msl16() {
;CHECK:  mvni {{v[0-31]+}}.4s, #0x10, msl #16
   ret <4 x i32> < i32 18446744073708437504, i32 18446744073708437504, i32 18446744073708437504, i32 18446744073708437504>
}

define <2 x i64> @movi2d() {
;CHECK: movi {{v[0-31]+}}.2d, #0xff0000ff0000ffff
	ret <2 x i64> < i64 18374687574888349695, i64 18374687574888349695 >
}

define <1 x i64> @movid() {
;CHECK: movi {{d[0-31]+}}, #0xff0000ff0000ffff
	ret  <1 x i64> < i64 18374687574888349695 >
}

define <2 x float> @fmov2s() {
;CHECK:  fmov {{v[0-31]+}}.2s, #-12.00000000
	ret <2 x float> < float -1.2e1, float -1.2e1>
}

define <4 x float> @fmov4s() {
;CHECK:  fmov {{v[0-31]+}}.4s, #-12.00000000
	ret <4 x float> < float -1.2e1, float -1.2e1, float -1.2e1, float -1.2e1>
}

define <2 x double> @fmov2d() {
;CHECK:  fmov {{v[0-31]+}}.2d, #-12.00000000
	ret <2 x double> < double -1.2e1, double -1.2e1>
}

define <2 x i32> @movi1d_1() {
; CHECK: movi    d0, #0xffffffff0000
  ret <2 x i32> < i32  -65536, i32 65535>
}


declare <2 x i32> @test_movi1d(<2 x i32>, <2 x i32>)
define <2 x i32> @movi1d() {
; CHECK: movi     d1, #0xffffffff0000
  %1 = tail call <2 x i32> @test_movi1d(<2 x i32> <i32 -2147483648, i32 2147450880>, <2 x i32> <i32 -65536, i32 65535>)
  ret <2 x i32> %1
}

