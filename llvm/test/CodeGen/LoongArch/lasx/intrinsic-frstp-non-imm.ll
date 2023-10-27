; RUN: not llc --mtriple=loongarch64 --mattr=+lasx < %s 2>&1 | FileCheck %s

declare <32 x i8> @llvm.loongarch.lasx.xvfrstpi.b(<32 x i8>, <32 x i8>, i32)

define <32 x i8> @lasx_xvfrstpi_b(<32 x i8> %va, <32 x i8> %vb, i32 %c) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <32 x i8> @llvm.loongarch.lasx.xvfrstpi.b(<32 x i8> %va, <32 x i8> %vb, i32 %c)
  ret <32 x i8> %res
}

declare <16 x i16> @llvm.loongarch.lasx.xvfrstpi.h(<16 x i16>, <16 x i16>, i32)

define <16 x i16> @lasx_xvfrstpi_h(<16 x i16> %va, <16 x i16> %vb, i32 %c) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <16 x i16> @llvm.loongarch.lasx.xvfrstpi.h(<16 x i16> %va, <16 x i16> %vb, i32 %c)
  ret <16 x i16> %res
}
