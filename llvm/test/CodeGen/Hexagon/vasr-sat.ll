
; RUN: llc -march=hexagon -mattr=+hvxv73,+hvx-length128b < %s | FileCheck %s

; Test for saturating vasr instruction.

; CHECK-LABEL: test_vasr
; CHECK: = vasr{{.*}}:sat

define dllexport void @test_vasr(i64 %seed0, i64 %seed1,
                                 i8* %dst) local_unnamed_addr {
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

  ; Broadcast splats
  %broadcast.splatinsert216 = insertelement <128 x i16> poison, i16 %9, i32 0
  %broadcast.splat217 = shufflevector <128 x i16> %broadcast.splatinsert216, <128 x i16> poison, <128 x i32> zeroinitializer
  %broadcast.splatinsert214 = insertelement <128 x i16> poison, i16 %8, i32 0
  %broadcast.splat215 = shufflevector <128 x i16> %broadcast.splatinsert214, <128 x i16> poison, <128 x i32> zeroinitializer
  %11 = ashr <128 x i16> %broadcast.splat215, %broadcast.splat217
  %12 = icmp slt <128 x i16> %11, <i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255>
  %13 = select <128 x i1> %12, <128 x i16> %11, <128 x i16> <i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255>
  %14 = icmp sgt <128 x i16> %13, zeroinitializer
  %15 = select <128 x i1> %14, <128 x i16> %13, <128 x i16> zeroinitializer
  %16 = trunc <128 x i16> %15 to <128 x i8>
  %17 = bitcast i8* %dst to <128 x i8>*
  store <128 x i8> %16, <128 x i8>* %17, align 1
  ret void
}

; CHECK-LABEL: test_vasr_with_intrinsic
; CHECK: = vasr{{.*}}:sat

define dllexport void @test_vasr_with_intrinsic(i64 %seed0, i64 %seed1,
                                                i8* %dst) local_unnamed_addr {
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
  %broadcast.splatinsert214 = insertelement <128 x i16> poison, i16 %8, i32 0
  %broadcast.splat215 = shufflevector <128 x i16> %broadcast.splatinsert214, <128 x i16> poison, <128 x i32> zeroinitializer
  %11 = ashr <128 x i16> %broadcast.splat215, <i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4>
  %12 = call <128 x i16> @llvm.smin.v128i16(<128 x i16> %11, <128 x i16> <i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255>)
  %13 = call <128 x i16> @llvm.smax.v128i16(<128 x i16> %12, <128 x i16> zeroinitializer)
  %14 = trunc <128 x i16> %13 to <128 x i8>
  %15 = bitcast i8* %dst to <128 x i8>*
  store <128 x i8> %14, <128 x i8>* %15, align 1
  ret void
}

declare <128 x i16> @llvm.smin.v128i16(<128 x i16>, <128 x i16>)
declare <128 x i16> @llvm.smax.v128i16(<128 x i16>, <128 x i16>)
