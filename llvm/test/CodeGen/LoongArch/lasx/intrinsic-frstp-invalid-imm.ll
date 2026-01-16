; RUN: not llc --mtriple=loongarch32 --mattr=+32s,+lasx < %s 2>&1 | FileCheck %s
; RUN: not llc --mtriple=loongarch64 --mattr=+lasx < %s 2>&1 | FileCheck %s

declare <32 x i8> @llvm.loongarch.lasx.xvfrstpi.b(<32 x i8>, <32 x i8>, i32)

define <32 x i8> @lasx_xvfrstpi_b_lo(<32 x i8> %va, <32 x i8> %vb) nounwind {
; CHECK: llvm.loongarch.lasx.xvfrstpi.b: argument out of range
entry:
  %res = call <32 x i8> @llvm.loongarch.lasx.xvfrstpi.b(<32 x i8> %va, <32 x i8> %vb, i32 -1)
  ret <32 x i8> %res
}

define <32 x i8> @lasx_xvfrstpi_b_hi(<32 x i8> %va, <32 x i8> %vb) nounwind {
; CHECK: llvm.loongarch.lasx.xvfrstpi.b: argument out of range
entry:
  %res = call <32 x i8> @llvm.loongarch.lasx.xvfrstpi.b(<32 x i8> %va, <32 x i8> %vb, i32 32)
  ret <32 x i8> %res
}

declare <16 x i16> @llvm.loongarch.lasx.xvfrstpi.h(<16 x i16>, <16 x i16>, i32)

define <16 x i16> @lasx_xvfrstpi_h_lo(<16 x i16> %va, <16 x i16> %vb) nounwind {
; CHECK: llvm.loongarch.lasx.xvfrstpi.h: argument out of range
entry:
  %res = call <16 x i16> @llvm.loongarch.lasx.xvfrstpi.h(<16 x i16> %va, <16 x i16> %vb, i32 -1)
  ret <16 x i16> %res
}

define <16 x i16> @lasx_xvfrstpi_h_hi(<16 x i16> %va, <16 x i16> %vb) nounwind {
; CHECK: llvm.loongarch.lasx.xvfrstpi.h: argument out of range
entry:
  %res = call <16 x i16> @llvm.loongarch.lasx.xvfrstpi.h(<16 x i16> %va, <16 x i16> %vb, i32 32)
  ret <16 x i16> %res
}
