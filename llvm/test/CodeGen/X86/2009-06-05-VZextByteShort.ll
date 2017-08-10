; RUN: llc < %s -mtriple=i686-- -mcpu=core2 | FileCheck %s

define <4 x i16> @a(i32* %x1) nounwind {
; CHECK-LABEL: a:
; CHECK:         shrl %[[R:[^,]+]]
; CHECK-NEXT:    movd %[[R]], %xmm0
; CHECK-NEXT:    retl

  %x2 = load i32, i32* %x1
  %x3 = lshr i32 %x2, 1
  %x = trunc i32 %x3 to i16
  %r = insertelement <4 x i16> zeroinitializer, i16 %x, i32 0
  ret <4 x i16> %r
}

define <8 x i16> @b(i32* %x1) nounwind {
; CHECK-LABEL: b:
; CHECK:         shrl %e[[R:.]]x
; CHECK-NEXT:    movzwl %[[R]]x, %e[[R]]x
; CHECK-NEXT:    movd %e[[R]]x, %xmm0
; CHECK-NEXT:    retl

  %x2 = load i32, i32* %x1
  %x3 = lshr i32 %x2, 1
  %x = trunc i32 %x3 to i16
  %r = insertelement <8 x i16> zeroinitializer, i16 %x, i32 0
  ret <8 x i16> %r
}

define <8 x i8> @c(i32* %x1) nounwind {
; CHECK-LABEL: c:
; CHECK:         shrl %e[[R:.]]x
; CHECK-NEXT:    movzwl %[[R]]x, %e[[R]]x
; CHECK-NEXT:    movd %e[[R]]x, %xmm0
; CHECK-NEXT:    retl

  %x2 = load i32, i32* %x1
  %x3 = lshr i32 %x2, 1
  %x = trunc i32 %x3 to i8
  %r = insertelement <8 x i8> zeroinitializer, i8 %x, i32 0
  ret <8 x i8> %r
}

define <16 x i8> @d(i32* %x1) nounwind {
; CHECK-LABEL: d:
; CHECK:         shrl %e[[R:.]]x
; CHECK-NEXT:    movzbl %[[R]]l, %e[[R]]x
; CHECK-NEXT:    movd %e[[R]]x, %xmm0
; CHECK-NEXT:    retl

  %x2 = load i32, i32* %x1
  %x3 = lshr i32 %x2, 1
  %x = trunc i32 %x3 to i8
  %r = insertelement <16 x i8> zeroinitializer, i8 %x, i32 0
  ret <16 x i8> %r
}

