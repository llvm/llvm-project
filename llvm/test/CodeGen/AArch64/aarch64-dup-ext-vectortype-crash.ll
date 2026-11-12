; RUN: llc < %s -mtriple aarch64-none-linux-gnu | FileCheck %s

; This test covers a case where extended value types can't be converted to
; vector types, resulting in a crash. We don't care about the specific output
; here, only that this case no longer causes said crash.
; See https://reviews.llvm.org/D91255#2484399 for context
define <8 x i16> @extend_i7_v8i16(i7 %src, <8 x i8> %b) {
; CHECK-LABEL: extend_i7_v8i16:
entry:
    %in = sext i7 %src to i16
    %ext.b = sext <8 x i8> %b to <8 x i16>
    %broadcast.splatinsert = insertelement <8 x i16> undef, i16 %in, i16 0
    %broadcast.splat = shufflevector <8 x i16> %broadcast.splatinsert, <8 x i16> undef, <8 x i32> zeroinitializer
    %out = mul nsw <8 x i16> %broadcast.splat, %ext.b
    ret <8 x i16> %out
}
