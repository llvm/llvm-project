
; RUN: llc -march=hexagon -hexagon-hvx-widen=32 -hexagon-widen-short-vector -mattr=+hvxv73,+hvx-length128b < %s | FileCheck %s

; CHECK-LABEL: test_vasr
; CHECK: = vasr{{.*}}:sat

define dllexport void @test_vasr(i64 %seed0, i64 %seed1, i8* %dst) local_unnamed_addr {
entry:
  %1 = trunc i64 %seed0 to i32
  %t.1 = trunc i64 %seed1 to i32
  %2 = lshr i32 %t.1, 23
  %3 = and i32 %2, 255
  %4 = icmp ugt i32 %3, 125
  %5 = select i1 %4, i32 %3, i32 125
  %6 = sub nsw i32 132, %5
  %7 = shl i32 %1, %6
  %8 = trunc i32 %7 to i16
  %9 = trunc i32 %6 to i16

  %broadcast.splatinsert50 = insertelement <64 x i16> poison, i16 %8, i32 0
  %broadcast.splat51 = shufflevector <64 x i16> %broadcast.splatinsert50, <64 x i16> poison, <64 x i32> zeroinitializer
  %broadcast.splatinsert52 = insertelement <64 x i16> poison, i16 %9, i32 0
  %broadcast.splat53 = shufflevector <64 x i16> %broadcast.splatinsert52, <64 x i16> poison, <64 x i32> zeroinitializer

  %11 = call <64 x i16> @llvm.sadd.sat.v64i16(<64 x i16> zeroinitializer, <64 x i16> %broadcast.splat51)
  %12 = ashr <64 x i16> %11, %broadcast.splat53
  %13 = icmp slt <64 x i16> %12, <i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255>
  %14 = select <64 x i1> %13, <64 x i16> %12, <64 x i16> <i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255>
  %15 = icmp sgt <64 x i16> %14, zeroinitializer
  %16 = select <64 x i1> %15, <64 x i16> %14, <64 x i16> zeroinitializer
  %17 = trunc <64 x i16> %16 to <64 x i8>
  %elem = extractelement <64 x i8> %17, i32 0
  store i8 %elem, i8* %dst, align 1
  ret void
}

; CHECK-LABEL: test_vasr_with_intrinsic
; CHECK: v{{[0-9:]+}}.ub = vasr(v{{[0-9]+}}.h,v{{[0-9]+}}.h,r{{[0-9]+}}):sat

define dllexport void @test_vasr_with_intrinsic(i64 %seed0, i64 %seed1, i8* %dst) local_unnamed_addr {
entry:
  %1 = trunc i64 %seed0 to i32
  %t.1 = trunc i64 %seed1 to i32
  %2 = lshr i32 %t.1, 23
  %3 = and i32 %2, 255
  %4 = icmp ugt i32 %3, 125
  %5 = select i1 %4, i32 %3, i32 125
  %6 = sub nsw i32 132, %5
  %7 = shl i32 %1, %6
  %8 = trunc i32 %7 to i16
  %9 = trunc i32 %6 to i16

  %broadcast.splatinsert50 = insertelement <64 x i16> poison, i16 %8, i32 0
  %broadcast.splat51 = shufflevector <64 x i16> %broadcast.splatinsert50, <64 x i16> poison, <64 x i32> zeroinitializer

  %11 = call <64 x i16> @llvm.sadd.sat.v64i16(<64 x i16> zeroinitializer, <64 x i16> %broadcast.splat51)
  %12 = ashr <64 x i16> %11, <i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4>
  %13 = call <64 x i16> @llvm.smin.v64i16(<64 x i16> %12, <64 x i16> <i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255>)
  %14 = call <64 x i16> @llvm.smax.v64i16(<64 x i16> %13, <64 x i16> zeroinitializer)
  %15 = trunc <64 x i16> %14 to <64 x i8>
  %elem = extractelement <64 x i8> %15, i32 0
  store i8 %elem, i8* %dst, align 1
  ret void
}

declare <64 x i16> @llvm.sadd.sat.v64i16(<64 x i16>, <64 x i16>)
declare <64 x i16> @llvm.smin.v64i16(<64 x i16>, <64 x i16>)
declare <64 x i16> @llvm.smax.v64i16(<64 x i16>, <64 x i16>)
