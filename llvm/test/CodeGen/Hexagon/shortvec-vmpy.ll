
; RUN: llc -march=hexagon -mattr=+hvxv73,+hvx-length128b -hexagon-hvx-widen=32 -hexagon-widen-short-vector < %s | FileCheck %s

; CHECK: {{[0-9]+:[0-9]+}}.uh = vmpy(v{{[0-9]+}}.ub,v{{[0-9]+}}.ub)

define dllexport void @test_vmpy(i64 %seed, i8 %val, i8* %dst) local_unnamed_addr {
entry:
  ; Replace poison loads with args
  %t.1 = trunc i64 %seed to i16
  %0 = lshr i16 %t.1, 7
  %1 = and i16 %0, 255
  %broadcast.splatinsert44 = insertelement <64 x i16> poison, i16 %1, i32 0
  %broadcast.splat45 = shufflevector <64 x i16> %broadcast.splatinsert44, <64 x i16> poison, <64 x i32> zeroinitializer
  %3 = insertelement <64 x i8> poison, i8 %val, i32 57
  %4 = insertelement <64 x i8> %3, i8 %val, i32 58
  %5 = insertelement <64 x i8> %4, i8 %val, i32 59
  %6 = insertelement <64 x i8> %5, i8 %val, i32 60
  %7 = insertelement <64 x i8> %6, i8 %val, i32 61
  %8 = insertelement <64 x i8> %7, i8 %val, i32 62
  %9 = insertelement <64 x i8> %8, i8 %val, i32 63
  %10 = zext <64 x i8> %9 to <64 x i16>
  %11 = mul nuw <64 x i16> %broadcast.splat45, %10
  %12 = trunc <64 x i16> %11 to <64 x i8>
  %elem = extractelement <64 x i8> %12, i32 0
  store i8 %elem, i8* %dst, align 1
  ret void
}
