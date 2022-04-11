; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve -aarch64-sve-vector-bits-min=512  -o - -asm-verbose=0 < %s | FileCheck %s

; CHECK-LABEL: vls_sve_and_64xi8:
; CHECK-NEXT:  adrp    x[[ONE:[0-9]+]], .LCPI0_0
; CHECK-NEXT:  add     x[[TWO:[0-9]+]], x[[ONE]], :lo12:.LCPI0_0
; CHECK-NEXT:  ptrue   p0.b, vl64
; CHECK-NEXT:  ld1b    { z0.b }, p0/z, [x0]
; CHECK-NEXT:  ld1b    { z1.b }, p0/z, [x[[TWO]]]
; CHECK-NEXT:  and     z0.d, z0.d, z1.d
; CHECK-NEXT:  st1b    { z0.b }, p0, [x1]
; CHECK-NEXT:  ret
define void @vls_sve_and_64xi8(<64 x i8>* %ap, <64 x i8>* %out) nounwind {
 %a = load <64 x i8>, <64 x i8>* %ap
 %b = and <64 x i8> %a, <i8 0, i8 255, i8 0, i8 255, i8 0, i8 255, i8 0, i8 255, i8 0, i8 255, i8 0, i8 255, i8 0, i8 255, i8 0, i8 255,
                         i8 0, i8 255, i8 0, i8 255, i8 0, i8 255, i8 0, i8 255, i8 0, i8 255, i8 0, i8 255, i8 0, i8 255, i8 0, i8 255,
                         i8 0, i8 255, i8 0, i8 255, i8 0, i8 255, i8 0, i8 255, i8 0, i8 255, i8 0, i8 255, i8 0, i8 255, i8 0, i8 255,
                         i8 0, i8 255, i8 0, i8 255, i8 0, i8 255, i8 0, i8 255, i8 0, i8 255, i8 0, i8 255, i8 0, i8 255, i8 0, i8 255>
 store <64 x i8> %b, <64 x i8>* %out
 ret void
}

; CHECK-LABEL: vls_sve_and_16xi8:
; CHECK-NEXT:  bic     v0.8h, #255
; CHECK-NEXT:  ret
define <16 x i8> @vls_sve_and_16xi8(<16 x i8> %b, <16 x i8>* %out) nounwind {
 %c = and <16 x i8> %b, <i8 0, i8 255, i8 0, i8 255, i8 0, i8 255, i8 0, i8 255, i8 0, i8 255, i8 0, i8 255, i8 0, i8 255, i8 0, i8 255>
 ret <16 x i8> %c
}

; CHECK-LABEL: vls_sve_and_8xi8:
; CHECK-NEXT:  bic     v0.4h, #255
; CHECK-NEXT:  ret
define <8 x i8> @vls_sve_and_8xi8(<8 x i8> %b, <8 x i8>* %out) nounwind {
 %c = and <8 x i8> %b, <i8 0, i8 255, i8 0, i8 255, i8 0, i8 255, i8 0, i8 255>
 ret <8 x i8> %c
}

