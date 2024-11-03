; RUN: not llc --mtriple=loongarch64 --mattr=+lsx < %s 2>&1 | FileCheck %s

declare <16 x i8> @llvm.loongarch.lsx.vfrstpi.b(<16 x i8>, <16 x i8>, i32)

define <16 x i8> @lsx_vfrstpi_b_lo(<16 x i8> %va, <16 x i8> %vb) nounwind {
; CHECK: llvm.loongarch.lsx.vfrstpi.b: argument out of range
entry:
  %res = call <16 x i8> @llvm.loongarch.lsx.vfrstpi.b(<16 x i8> %va, <16 x i8> %vb, i32 -1)
  ret <16 x i8> %res
}

define <16 x i8> @lsx_vfrstpi_b_hi(<16 x i8> %va, <16 x i8> %vb) nounwind {
; CHECK: llvm.loongarch.lsx.vfrstpi.b: argument out of range
entry:
  %res = call <16 x i8> @llvm.loongarch.lsx.vfrstpi.b(<16 x i8> %va, <16 x i8> %vb, i32 32)
  ret <16 x i8> %res
}

declare <8 x i16> @llvm.loongarch.lsx.vfrstpi.h(<8 x i16>, <8 x i16>, i32)

define <8 x i16> @lsx_vfrstpi_h_lo(<8 x i16> %va, <8 x i16> %vb) nounwind {
; CHECK: llvm.loongarch.lsx.vfrstpi.h: argument out of range
entry:
  %res = call <8 x i16> @llvm.loongarch.lsx.vfrstpi.h(<8 x i16> %va, <8 x i16> %vb, i32 -1)
  ret <8 x i16> %res
}

define <8 x i16> @lsx_vfrstpi_h_hi(<8 x i16> %va, <8 x i16> %vb) nounwind {
; CHECK: llvm.loongarch.lsx.vfrstpi.h: argument out of range
entry:
  %res = call <8 x i16> @llvm.loongarch.lsx.vfrstpi.h(<8 x i16> %va, <8 x i16> %vb, i32 32)
  ret <8 x i16> %res
}
