; RUN: llc -mtriple=arm64-linux-gnu -o - %s | FileCheck %s

define i32 @test_neg_i32(i32 %in) {
; CHECK-LABEL: test_neg_i32:
; CHECK: neg w0, w0
  %res = sub i32 0, %in
  ret i32 %res
}

define i64 @test_neg_i64(i64 %in) {
; CHECK-LABEL: test_neg_i64:
; CHECK: neg x0, x0
  %res = sub i64 0, %in
  ret i64 %res
}

define <8 x i8> @test_neg_v8i8(<8 x i8> %in) {
; CHECK-LABEL: test_neg_v8i8:
; CHECK: neg v0.8b, v0.8b
  %res = sub <8 x i8> zeroinitializer, %in
  ret <8 x i8> %res
}

define <4 x i16> @test_neg_v4i16(<4 x i16> %in) {
; CHECK-LABEL: test_neg_v4i16:
; CHECK: neg v0.4h, v0.4h
  %res = sub <4 x i16> zeroinitializer, %in
  ret <4 x i16> %res
}

define <2 x i32> @test_neg_v2i32(<2 x i32> %in) {
; CHECK-LABEL: test_neg_v2i32:
; CHECK: neg v0.2s, v0.2s
  %res = sub <2 x i32> zeroinitializer, %in
  ret <2 x i32> %res
}

define <16 x i8> @test_neg_v16i8(<16 x i8> %in) {
; CHECK-LABEL: test_neg_v16i8:
; CHECK: neg v0.16b, v0.16b
  %res = sub <16 x i8> zeroinitializer, %in
  ret <16 x i8> %res
}

define <8 x i16> @test_neg_v8i16(<8 x i16> %in) {
; CHECK-LABEL: test_neg_v8i16:
; CHECK: neg v0.8h, v0.8h
  %res = sub <8 x i16> zeroinitializer, %in
  ret <8 x i16> %res
}

define <4 x i32> @test_neg_v4i32(<4 x i32> %in) {
; CHECK-LABEL: test_neg_v4i32:
; CHECK: neg v0.4s, v0.4s
  %res = sub <4 x i32> zeroinitializer, %in
  ret <4 x i32> %res
}

define <2 x i64> @test_neg_v2i64(<2 x i64> %in) {
; CHECK-LABEL: test_neg_v2i64:
; CHECK: neg v0.2d, v0.2d
  %res = sub <2 x i64> zeroinitializer, %in
  ret <2 x i64> %res
}

define <1 x i64> @test_neg_v1i64(<1 x i64> %in) {
; CHECK-LABEL: test_neg_v1i64:
; CHECK: neg d0, d0
  %res = sub <1 x i64> zeroinitializer, %in
  ret <1 x i64> %res
}
