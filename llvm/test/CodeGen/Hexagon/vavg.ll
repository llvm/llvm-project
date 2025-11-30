; RUN: llc -march=hexagon -mv73 -mhvx -mattr=+hvx-length128b < %s | FileCheck %s
; Test for vmpa instruction.

; CHECK: = vavg(v{{[0-9:]+}}.uh,v{{[0-9]+}}.uh)

define dllexport void @test_vavg(float %f0, float %f1,
                                 <128 x i8> %src,
                                 i16* %dst) local_unnamed_addr {
entry:
  %0 = select i1 false, float %f0, float %f1
  %1 = fptosi float %0 to i16
  %2 = lshr i16 %1, 7
  %3 = and i16 %2, 255
  %4 = and i16 %1, 127
  %broadcast.splatinsert212.1 = insertelement <128 x i16> poison, i16 %4, i32 0
  %broadcast.splat213.1 = shufflevector <128 x i16> %broadcast.splatinsert212.1, <128 x i16> poison, <128 x i32> zeroinitializer
  %broadcast.splatinsert208.1 = insertelement <128 x i16> poison, i16 %3, i32 0
  %broadcast.splat209.1 = shufflevector <128 x i16> %broadcast.splatinsert208.1, <128 x i16> poison, <128 x i32> zeroinitializer
  %7 = zext <128 x i8> %src to <128 x i16>
  %8 = mul nuw <128 x i16> %broadcast.splat209.1, %7
  %9 = add <128 x i16> %8, zeroinitializer
  %10 = zext <128 x i16> %9 to <128 x i32>
  %11 = mul nuw nsw <128 x i16> %broadcast.splat213.1, %7
  %12 = add nuw <128 x i16> %11, zeroinitializer
  %13 = lshr <128 x i16> %12, <i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7>
  %14 = zext <128 x i16> %13 to <128 x i32>
  %15 = add nuw nsw <128 x i32> %14, %10
  %16 = lshr <128 x i32> %15, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %17 = trunc <128 x i32> %16 to <128 x i16>
  %19 = bitcast i16* %dst to <128 x i16>*
  store <128 x i16> %17, <128 x i16>* %19, align 1
  ret void
}
