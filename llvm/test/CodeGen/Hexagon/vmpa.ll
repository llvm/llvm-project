; RUN: llc -march=hexagon -mattr=+hvxv73,+hvx-length128b < %s | FileCheck %s

; Test for vmpa instruction.

; CHECK-LABEL: test_vmpa8
; CHECK: = vmpa(v{{[0-9:]+}}.ub,r{{[0-9]+}}.b)

; Function Attrs: nounwind
define dllexport void @test_vmpa8(i64 %seed0, i64 %seed1,
                                  <128 x i8> %srcA, <128 x i8> %srcB,
                                  i8* %dst) local_unnamed_addr {
entry:
  %1 = trunc i64 %seed0 to i16
  %3 = trunc i64 %seed1 to i8
  %4 = and i8 %3, 127
  %5 = insertelement <128 x i8> poison, i8 %4, i32 0
  %6 = shufflevector <128 x i8> %5, <128 x i8> poison, <128 x i32> zeroinitializer
  %7 = zext <128 x i8> %6 to <128 x i16>
  %8 = and i16 %1, 127
  %9 = insertelement <128 x i16> poison, i16 %8, i32 0
  %10 = shufflevector <128 x i16> %9, <128 x i16> poison, <128 x i32> zeroinitializer
  %11 = zext <128 x i8> %srcA to <128 x i16>
  %12 = zext <128 x i8> %srcB to <128 x i16>
  %13 = mul nuw nsw <128 x i16> %11, %7
  %14 = mul nuw nsw <128 x i16> %10, %12
  %15 = add nuw <128 x i16> %14, %13
  %16 = lshr <128 x i16> %15, <i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7, i16 7>
  %17 = add <128 x i16> zeroinitializer, %16
  %18 = trunc <128 x i16> %17 to <128 x i8>
  %21 = bitcast i8* %dst to <128 x i8>*
  store <128 x i8> %18, <128 x i8>* %21, align 128
  ret void
}

; CHECK-LABEL: test_vmpa16
; CHECK: = vmpa(v{{[0-9:]+}}.uh,r{{[0-9]+}}.b)

; Function Attrs: nounwind
define dllexport void @test_vmpa16(i64 %seed0, i64 %seed1,
                                   <64 x i16> %srcA16, <64 x i16> %srcB16,
                                   i16* %dst16) local_unnamed_addr {
entry:
  %1 = trunc i64 %seed0 to i32
  %3 = trunc i64 %seed1 to i32
  %4 = and i32 %3, 127
  %5 = insertelement <64 x i32> poison, i32 %4, i32 0
  %6 = shufflevector <64 x i32> %5, <64 x i32> poison, <64 x i32> zeroinitializer
  %7 = and i32 %3, 127
  %8 = and i32 %1, 127
  %9 = insertelement <64 x i32> poison, i32 %8, i32 0
  %10 = shufflevector <64 x i32> %9, <64 x i32> poison, <64 x i32> zeroinitializer
  %11 = zext <64 x i16> %srcA16 to <64 x i32>
  %12 = zext <64 x i16> %srcB16 to <64 x i32>
  %13 = mul nuw nsw <64 x i32> %11, %6
  %14 = mul nuw nsw <64 x i32> %10, %12
  %15 = add nuw <64 x i32> %14, %13
  %16 = lshr <64 x i32> %15, <i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7>
  ;, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7>
  %17 = add <64 x i32> zeroinitializer, %16
  %18 = trunc <64 x i32> %17 to <64 x i16>
  %21 = bitcast i16* %dst16 to <64 x i16>*
  store <64 x i16> %18, <64 x i16>* %21, align 128
  ret void
}
