; RUN: not llc --mtriple=loongarch64 --mattr=+lsx < %s 2>&1 | FileCheck %s

declare <16 x i8> @llvm.loongarch.lsx.vfrstpi.b(<16 x i8>, <16 x i8>, i32)

define <16 x i8> @lsx_vfrstpi_b(<16 x i8> %va, <16 x i8> %vb, i32 %c) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <16 x i8> @llvm.loongarch.lsx.vfrstpi.b(<16 x i8> %va, <16 x i8> %vb, i32 %c)
  ret <16 x i8> %res
}

declare <8 x i16> @llvm.loongarch.lsx.vfrstpi.h(<8 x i16>, <8 x i16>, i32)

define <8 x i16> @lsx_vfrstpi_h(<8 x i16> %va, <8 x i16> %vb, i32 %c) nounwind {
; CHECK: immarg operand has non-immediate parameter
entry:
  %res = call <8 x i16> @llvm.loongarch.lsx.vfrstpi.h(<8 x i16> %va, <8 x i16> %vb, i32 %c)
  ret <8 x i16> %res
}
