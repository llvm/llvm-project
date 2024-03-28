; RUN: llc -mtriple=aarch64-unknown-linux-gnu < %s | FileCheck %s

define <8 x i16> @uhadd_in_disguise(<8 x i16> %a0, <8 x i16> %a1) {
; CHECK-LABEL: uhadd_in_disguise:
; CHECK:       and
; CHECK-NEXT:  uhadd
; CHECK-NEXT:  ret
  %m0 = and <8 x i16> %a0, <i16 510, i16 510, i16 510, i16 510, i16 510, i16 510, i16 510, i16 510>
  %m1 = and <8 x i16> %a1, <i16 510, i16 510, i16 510, i16 510, i16 510, i16 510, i16 510, i16 510>
  %r = call <8 x i16> @llvm.aarch64.neon.shadd.v8i16(<8 x i16> %m0, <8 x i16> %m1)
  ret <8 x i16> %r
}
declare <8 x i16> @llvm.aarch64.neon.shadd.v8i16(<8 x i16>, <8 x i16>)
