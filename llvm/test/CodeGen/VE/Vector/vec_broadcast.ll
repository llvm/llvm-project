; RUN: llc < %s -mtriple=ve-unknown-unknown -mattr=+vpu | FileCheck %s

; ISA-compatible vector broadcasts
define fastcc <256 x i64> @brd_v256i64(i64 %s) {
; CHECK-LABEL: brd_v256i64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vbrd %v0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %val = insertelement <256 x i64> undef, i64 %s, i32 0
  %ret = shufflevector <256 x i64> %val, <256 x i64> undef, <256 x i32> zeroinitializer
  ret <256 x i64> %ret
}

define fastcc <256 x i64> @brdi_v256i64() {
; CHECK-LABEL: brdi_v256i64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vbrd %v0, 1
; CHECK-NEXT:    b.l.t (, %s10)
  %val = insertelement <256 x i64> undef, i64 1, i32 0
  %ret = shufflevector <256 x i64> %val, <256 x i64> undef, <256 x i32> zeroinitializer
  ret <256 x i64> %ret
}

define fastcc <256 x double> @brd_v256f64(double %s) {
; CHECK-LABEL: brd_v256f64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vbrd %v0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %val = insertelement <256 x double> undef, double %s, i32 0
  %ret = shufflevector <256 x double> %val, <256 x double> undef, <256 x i32> zeroinitializer
  ret <256 x double> %ret
}

define fastcc <256 x double> @brdi_v256f64() {
; CHECK-LABEL: brdi_v256f64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vbrd %v0, 0
; CHECK-NEXT:    b.l.t (, %s10)
  %val = insertelement <256 x double> undef, double 0.e+00, i32 0
  %ret = shufflevector <256 x double> %val, <256 x double> undef, <256 x i32> zeroinitializer
  ret <256 x double> %ret
}

define fastcc <256 x i32> @brd_v256i32(i32 %s) {
; CHECK-LABEL: brd_v256i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vbrd %v0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %val = insertelement <256 x i32> undef, i32 %s, i32 0
  %ret = shufflevector <256 x i32> %val, <256 x i32> undef, <256 x i32> zeroinitializer
  ret <256 x i32> %ret
}

define fastcc <256 x i32> @brdi_v256i32(i32 %s) {
; CHECK-LABEL: brdi_v256i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vbrd %v0, 13
; CHECK-NEXT:    b.l.t (, %s10)
  %val = insertelement <256 x i32> undef, i32 13, i32 0
  %ret = shufflevector <256 x i32> %val, <256 x i32> undef, <256 x i32> zeroinitializer
  ret <256 x i32> %ret
}

define fastcc <256 x float> @brd_v256f32(float %s) {
; CHECK-LABEL: brd_v256f32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vbrd %v0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %val = insertelement <256 x float> undef, float %s, i32 0
  %ret = shufflevector <256 x float> %val, <256 x float> undef, <256 x i32> zeroinitializer
  ret <256 x float> %ret
}

define fastcc <256 x float> @brdi_v256f32(float %s) {
; CHECK-LABEL: brdi_v256f32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vbrd %v0, 0
; CHECK-NEXT:    b.l.t (, %s10)
  %val = insertelement <256 x float> undef, float 0.e+00, i32 0
  %ret = shufflevector <256 x float> %val, <256 x float> undef, <256 x i32> zeroinitializer
  ret <256 x float> %ret
}


; Shorter vectors, we expect these to be widened (for now).
define fastcc <128 x i64> @brd_v128i64(i64 %s) {
; CHECK-LABEL: brd_v128i64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vbrd %v0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %val = insertelement <128 x i64> undef, i64 %s, i32 0
  %ret = shufflevector <128 x i64> %val, <128 x i64> undef, <128 x i32> zeroinitializer
  ret <128 x i64> %ret
}

define fastcc <128 x double> @brd_v128f64(double %s) {
; CHECK-LABEL: brd_v128f64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vbrd %v0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %val = insertelement <128 x double> undef, double %s, i32 0
  %ret = shufflevector <128 x double> %val, <128 x double> undef, <128 x i32> zeroinitializer
  ret <128 x double> %ret
}

define fastcc <128 x i32> @brd_v128i32(i32 %s) {
; CHECK-LABEL: brd_v128i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vbrd %v0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %val = insertelement <128 x i32> undef, i32 %s, i32 0
  %ret = shufflevector <128 x i32> %val, <128 x i32> undef, <128 x i32> zeroinitializer
  ret <128 x i32> %ret
}

define fastcc <128 x i32> @brdi_v128i32(i32 %s) {
; CHECK-LABEL: brdi_v128i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vbrd %v0, 13
; CHECK-NEXT:    b.l.t (, %s10)
  %val = insertelement <128 x i32> undef, i32 13, i32 0
  %ret = shufflevector <128 x i32> %val, <128 x i32> undef, <128 x i32> zeroinitializer
  ret <128 x i32> %ret
}

define fastcc <128 x float> @brd_v128f32(float %s) {
; CHECK-LABEL: brd_v128f32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vbrd %v0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %val = insertelement <128 x float> undef, float %s, i32 0
  %ret = shufflevector <128 x float> %val, <128 x float> undef, <128 x i32> zeroinitializer
  ret <128 x float> %ret
}

define fastcc <128 x float> @brdi_v128f32(float %s) {
; CHECK-LABEL: brdi_v128f32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, 256
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    vbrd %v0, 0
; CHECK-NEXT:    b.l.t (, %s10)
  %val = insertelement <128 x float> undef, float 0.e+00, i32 0
  %ret = shufflevector <128 x float> %val, <128 x float> undef, <128 x i32> zeroinitializer
  ret <128 x float> %ret
}

; Vectors with small element types and valid element count, we expect those to be promoted.
define fastcc <256 x i16> @brd_v256i16(i16 %s) {
; CHECK-LABEL: brd_v256i16:
; CHECK:       # %bb.0:
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    lea %s1, 256
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    vbrd %v0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %val = insertelement <256 x i16> undef, i16 %s, i32 0
  %ret = shufflevector <256 x i16> %val, <256 x i16> undef, <256 x i32> zeroinitializer
  ret <256 x i16> %ret
}

; Vectors with small element types and low element count, these are scalarized for now.
; FIXME Promote + Widen
define fastcc <128 x i16> @brd_v128i16(i16 %s) {
; CHECK-LABEL: brd_v128i16:
; CHECK:       # %bb.0:
; CHECK-NEXT:  adds.w.sx %s1, %s1, (0)1
; CHECK-NEXT:  st2b %s1, 254(, %s0)
; CHECK-NEXT:  st2b %s1, 252(, %s0)
; CHECK-NEXT:  st2b %s1, 250(, %s0)
; CHECK-NEXT:  st2b %s1, 248(, %s0)
; CHECK-NEXT:  st2b %s1, 246(, %s0)
; CHECK-NEXT:  st2b %s1, 244(, %s0)
; CHECK-NEXT:  st2b %s1, 242(, %s0)
; CHECK-NEXT:  st2b %s1, 240(, %s0)
; CHECK-NEXT:  st2b %s1, 238(, %s0)
; CHECK-NEXT:  st2b %s1, 236(, %s0)
; CHECK-NEXT:  st2b %s1, 234(, %s0)
; CHECK-NEXT:  st2b %s1, 232(, %s0)
; CHECK-NEXT:  st2b %s1, 230(, %s0)
; CHECK-NEXT:  st2b %s1, 228(, %s0)
; CHECK-NEXT:  st2b %s1, 226(, %s0)
; CHECK-NEXT:  st2b %s1, 224(, %s0)
; CHECK-NEXT:  st2b %s1, 222(, %s0)
; CHECK-NEXT:  st2b %s1, 220(, %s0)
; CHECK-NEXT:  st2b %s1, 218(, %s0)
; CHECK-NEXT:  st2b %s1, 216(, %s0)
; CHECK-NEXT:  st2b %s1, 214(, %s0)
; CHECK-NEXT:  st2b %s1, 212(, %s0)
; CHECK-NEXT:  st2b %s1, 210(, %s0)
; CHECK-NEXT:  st2b %s1, 208(, %s0)
; CHECK-NEXT:  st2b %s1, 206(, %s0)
; CHECK-NEXT:  st2b %s1, 204(, %s0)
; CHECK-NEXT:  st2b %s1, 202(, %s0)
; CHECK-NEXT:  st2b %s1, 200(, %s0)
; CHECK-NEXT:  st2b %s1, 198(, %s0)
; CHECK-NEXT:  st2b %s1, 196(, %s0)
; CHECK-NEXT:  st2b %s1, 194(, %s0)
; CHECK-NEXT:  st2b %s1, 192(, %s0)
; CHECK-NEXT:  st2b %s1, 190(, %s0)
; CHECK-NEXT:  st2b %s1, 188(, %s0)
; CHECK-NEXT:  st2b %s1, 186(, %s0)
; CHECK-NEXT:  st2b %s1, 184(, %s0)
; CHECK-NEXT:  st2b %s1, 182(, %s0)
; CHECK-NEXT:  st2b %s1, 180(, %s0)
; CHECK-NEXT:  st2b %s1, 178(, %s0)
; CHECK-NEXT:  st2b %s1, 176(, %s0)
; CHECK-NEXT:  st2b %s1, 174(, %s0)
; CHECK-NEXT:  st2b %s1, 172(, %s0)
; CHECK-NEXT:  st2b %s1, 170(, %s0)
; CHECK-NEXT:  st2b %s1, 168(, %s0)
; CHECK-NEXT:  st2b %s1, 166(, %s0)
; CHECK-NEXT:  st2b %s1, 164(, %s0)
; CHECK-NEXT:  st2b %s1, 162(, %s0)
; CHECK-NEXT:  st2b %s1, 160(, %s0)
; CHECK-NEXT:  st2b %s1, 158(, %s0)
; CHECK-NEXT:  st2b %s1, 156(, %s0)
; CHECK-NEXT:  st2b %s1, 154(, %s0)
; CHECK-NEXT:  st2b %s1, 152(, %s0)
; CHECK-NEXT:  st2b %s1, 150(, %s0)
; CHECK-NEXT:  st2b %s1, 148(, %s0)
; CHECK-NEXT:  st2b %s1, 146(, %s0)
; CHECK-NEXT:  st2b %s1, 144(, %s0)
; CHECK-NEXT:  st2b %s1, 142(, %s0)
; CHECK-NEXT:  st2b %s1, 140(, %s0)
; CHECK-NEXT:  st2b %s1, 138(, %s0)
; CHECK-NEXT:  st2b %s1, 136(, %s0)
; CHECK-NEXT:  st2b %s1, 134(, %s0)
; CHECK-NEXT:  st2b %s1, 132(, %s0)
; CHECK-NEXT:  st2b %s1, 130(, %s0)
; CHECK-NEXT:  st2b %s1, 128(, %s0)
; CHECK-NEXT:  st2b %s1, 126(, %s0)
; CHECK-NEXT:  st2b %s1, 124(, %s0)
; CHECK-NEXT:  st2b %s1, 122(, %s0)
; CHECK-NEXT:  st2b %s1, 120(, %s0)
; CHECK-NEXT:  st2b %s1, 118(, %s0)
; CHECK-NEXT:  st2b %s1, 116(, %s0)
; CHECK-NEXT:  st2b %s1, 114(, %s0)
; CHECK-NEXT:  st2b %s1, 112(, %s0)
; CHECK-NEXT:  st2b %s1, 110(, %s0)
; CHECK-NEXT:  st2b %s1, 108(, %s0)
; CHECK-NEXT:  st2b %s1, 106(, %s0)
; CHECK-NEXT:  st2b %s1, 104(, %s0)
; CHECK-NEXT:  st2b %s1, 102(, %s0)
; CHECK-NEXT:  st2b %s1, 100(, %s0)
; CHECK-NEXT:  st2b %s1, 98(, %s0)
; CHECK-NEXT:  st2b %s1, 96(, %s0)
; CHECK-NEXT:  st2b %s1, 94(, %s0)
; CHECK-NEXT:  st2b %s1, 92(, %s0)
; CHECK-NEXT:  st2b %s1, 90(, %s0)
; CHECK-NEXT:  st2b %s1, 88(, %s0)
; CHECK-NEXT:  st2b %s1, 86(, %s0)
; CHECK-NEXT:  st2b %s1, 84(, %s0)
; CHECK-NEXT:  st2b %s1, 82(, %s0)
; CHECK-NEXT:  st2b %s1, 80(, %s0)
; CHECK-NEXT:  st2b %s1, 78(, %s0)
; CHECK-NEXT:  st2b %s1, 76(, %s0)
; CHECK-NEXT:  st2b %s1, 74(, %s0)
; CHECK-NEXT:  st2b %s1, 72(, %s0)
; CHECK-NEXT:  st2b %s1, 70(, %s0)
; CHECK-NEXT:  st2b %s1, 68(, %s0)
; CHECK-NEXT:  st2b %s1, 66(, %s0)
; CHECK-NEXT:  st2b %s1, 64(, %s0)
; CHECK-NEXT:  st2b %s1, 62(, %s0)
; CHECK-NEXT:  st2b %s1, 60(, %s0)
; CHECK-NEXT:  st2b %s1, 58(, %s0)
; CHECK-NEXT:  st2b %s1, 56(, %s0)
; CHECK-NEXT:  st2b %s1, 54(, %s0)
; CHECK-NEXT:  st2b %s1, 52(, %s0)
; CHECK-NEXT:  st2b %s1, 50(, %s0)
; CHECK-NEXT:  st2b %s1, 48(, %s0)
; CHECK-NEXT:  st2b %s1, 46(, %s0)
; CHECK-NEXT:  st2b %s1, 44(, %s0)
; CHECK-NEXT:  st2b %s1, 42(, %s0)
; CHECK-NEXT:  st2b %s1, 40(, %s0)
; CHECK-NEXT:  st2b %s1, 38(, %s0)
; CHECK-NEXT:  st2b %s1, 36(, %s0)
; CHECK-NEXT:  st2b %s1, 34(, %s0)
; CHECK-NEXT:  st2b %s1, 32(, %s0)
; CHECK-NEXT:  st2b %s1, 30(, %s0)
; CHECK-NEXT:  st2b %s1, 28(, %s0)
; CHECK-NEXT:  st2b %s1, 26(, %s0)
; CHECK-NEXT:  st2b %s1, 24(, %s0)
; CHECK-NEXT:  st2b %s1, 22(, %s0)
; CHECK-NEXT:  st2b %s1, 20(, %s0)
; CHECK-NEXT:  st2b %s1, 18(, %s0)
; CHECK-NEXT:  st2b %s1, 16(, %s0)
; CHECK-NEXT:  st2b %s1, 14(, %s0)
; CHECK-NEXT:  st2b %s1, 12(, %s0)
; CHECK-NEXT:  st2b %s1, 10(, %s0)
; CHECK-NEXT:  st2b %s1, 8(, %s0)
; CHECK-NEXT:  st2b %s1, 6(, %s0)
; CHECK-NEXT:  st2b %s1, 4(, %s0)
; CHECK-NEXT:  st2b %s1, 2(, %s0)
; CHECK-NEXT:  st2b %s1, (, %s0)
; CHECK-NEXT:  b.l.t (, %s10)
  %val = insertelement <128 x i16> undef, i16 %s, i32 0
  %ret = shufflevector <128 x i16> %val, <128 x i16> undef, <128 x i32> zeroinitializer
  ret <128 x i16> %ret
}
