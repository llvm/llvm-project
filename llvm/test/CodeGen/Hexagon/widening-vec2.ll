; RUN: llc -march=hexagon -mattr=+hvxv73,+hvx-length128b < %s | FileCheck %s

; Test to make sure that the widening vector instructions are being generated.

; CHECK: .uh = vmpy(v{{[0-9:]+}}.ub,v{{[0-9]+}}.ub)

define dllexport void @test1() local_unnamed_addr {
  %1 = load i64, i64* poison, align 8
  %2 = trunc i64 %1 to i16
  %3 = lshr i16 %2, 7
  %4 = and i16 %3, 255
  %broadcast.splatinsert.1 = insertelement <128 x i16> poison, i16 %4, i32 0
  %broadcast.splat.1 = shufflevector <128 x i16> %broadcast.splatinsert.1, <128 x i16> poison, <128 x i32> zeroinitializer
  %scevgep = getelementptr i8, i8* null, i32 128
  %lsr.iv13 = bitcast i8* %scevgep to <128 x i8>*
  %wide.load.1 = load <128 x i8>, <128 x i8>* poison, align 1
  %5 = zext <128 x i8> %wide.load.1 to <128 x i16>
  %6 = mul nuw <128 x i16> %broadcast.splat.1, %5
  %7 = add <128 x i16> zeroinitializer, %6
  %trun = trunc <128 x i16> %7 to <128 x i8>
  store <128 x i8> %trun, <128 x i8>* %lsr.iv13, align 1
  ret void
}
