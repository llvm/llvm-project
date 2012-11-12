; RUN: llc < %s -march=x86-64 -mcpu=corei7 | FileCheck %s

; Splat patterns below


define <4 x i32> @shl4(<4 x i32> %A) nounwind {
entry:
; CHECK:      shl4
; CHECK:      padd
; CHECK:      pslld
; CHECK:      ret
  %B = shl <4 x i32> %A,  < i32 2, i32 2, i32 2, i32 2>
  %C = shl <4 x i32> %A,  < i32 1, i32 1, i32 1, i32 1>
  %K = xor <4 x i32> %B, %C
  ret <4 x i32> %K
}

define <4 x i32> @shr4(<4 x i32> %A) nounwind {
entry:
; CHECK:      shr4
; CHECK:      psrld
; CHECK-NEXT: psrld
; CHECK:      ret
  %B = lshr <4 x i32> %A,  < i32 2, i32 2, i32 2, i32 2>
  %C = lshr <4 x i32> %A,  < i32 1, i32 1, i32 1, i32 1>
  %K = xor <4 x i32> %B, %C
  ret <4 x i32> %K
}

define <4 x i32> @sra4(<4 x i32> %A) nounwind {
entry:
; CHECK:      sra4
; CHECK:      psrad
; CHECK-NEXT: psrad
; CHECK:      ret
  %B = ashr <4 x i32> %A,  < i32 2, i32 2, i32 2, i32 2>
  %C = ashr <4 x i32> %A,  < i32 1, i32 1, i32 1, i32 1>
  %K = xor <4 x i32> %B, %C
  ret <4 x i32> %K
}

define <2 x i64> @shl2(<2 x i64> %A) nounwind {
entry:
; CHECK:      shl2
; CHECK:      psllq
; CHECK-NEXT: psllq
; CHECK:      ret
  %B = shl <2 x i64> %A,  < i64 2, i64 2>
  %C = shl <2 x i64> %A,  < i64 9, i64 9>
  %K = xor <2 x i64> %B, %C
  ret <2 x i64> %K
}

define <2 x i64> @shr2(<2 x i64> %A) nounwind {
entry:
; CHECK:      shr2
; CHECK:      psrlq
; CHECK-NEXT: psrlq
; CHECK:      ret
  %B = lshr <2 x i64> %A,  < i64 8, i64 8>
  %C = lshr <2 x i64> %A,  < i64 1, i64 1>
  %K = xor <2 x i64> %B, %C
  ret <2 x i64> %K
}


define <8 x i16> @shl8(<8 x i16> %A) nounwind {
entry:
; CHECK:      shl8
; CHECK:      padd
; CHECK:      psllw
; CHECK:      ret
  %B = shl <8 x i16> %A,  < i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2>
  %C = shl <8 x i16> %A,  < i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  %K = xor <8 x i16> %B, %C
  ret <8 x i16> %K
}

define <8 x i16> @shr8(<8 x i16> %A) nounwind {
entry:
; CHECK:      shr8
; CHECK:      psrlw
; CHECK-NEXT: psrlw
; CHECK:      ret
  %B = lshr <8 x i16> %A,  < i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2>
  %C = lshr <8 x i16> %A,  < i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  %K = xor <8 x i16> %B, %C
  ret <8 x i16> %K
}

define <8 x i16> @sra8(<8 x i16> %A) nounwind {
entry:
; CHECK:      sra8
; CHECK:      psraw
; CHECK-NEXT: psraw
; CHECK:      ret
  %B = ashr <8 x i16> %A,  < i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2>
  %C = ashr <8 x i16> %A,  < i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  %K = xor <8 x i16> %B, %C
  ret <8 x i16> %K
}

; non splat test


define <8 x i16> @sll8_nosplat(<8 x i16> %A) nounwind {
entry:
; CHECK: sll8_nosplat
; CHECK-NOT: psll
; CHECK-NOT: psll
; CHECK:      ret
  %B = shl <8 x i16> %A,  < i16 1, i16 2, i16 3, i16 6, i16 2, i16 2, i16 2, i16 2>
  %C = shl <8 x i16> %A,  < i16 9, i16 7, i16 5, i16 1, i16 4, i16 1, i16 1, i16 1>
  %K = xor <8 x i16> %B, %C
  ret <8 x i16> %K
}


define <2 x i64> @shr2_nosplat(<2 x i64> %A) nounwind {
entry:
; CHECK: shr2_nosplat
; CHECK-NOT:  psrlq
; CHECK-NOT:  psrlq
; CHECK:      ret
  %B = lshr <2 x i64> %A,  < i64 8, i64 1>
  %C = lshr <2 x i64> %A,  < i64 1, i64 0>
  %K = xor <2 x i64> %B, %C
  ret <2 x i64> %K
}


; Other shifts

define <2 x i32> @shl2_other(<2 x i32> %A) nounwind {
entry:
; CHECK: shl2_other
; CHECK: psllq
; CHECK: ret
  %B = shl <2 x i32> %A,  < i32 2, i32 2>
  %C = shl <2 x i32> %A,  < i32 9, i32 9>
  %K = xor <2 x i32> %B, %C
  ret <2 x i32> %K
}

define <2 x i32> @shr2_other(<2 x i32> %A) nounwind {
entry:
; CHECK: shr2_other
; CHECK: psrlq
; CHECK: ret
  %B = lshr <2 x i32> %A,  < i32 8, i32 8>
  %C = lshr <2 x i32> %A,  < i32 1, i32 1>
  %K = xor <2 x i32> %B, %C
  ret <2 x i32> %K
}

define <16 x i8> @shl9(<16 x i8> %A) nounwind {
  %B = shl <16 x i8> %A, <i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3>
  ret <16 x i8> %B
; CHECK: shl9:
; CHECK: psllw $3
; CHECK: pand
; CHECK: ret
}

define <16 x i8> @shr9(<16 x i8> %A) nounwind {
  %B = lshr <16 x i8> %A, <i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3>
  ret <16 x i8> %B
; CHECK: shr9:
; CHECK: psrlw $3
; CHECK: pand
; CHECK: ret
}

define <16 x i8> @sra_v16i8_7(<16 x i8> %A) nounwind {
  %B = ashr <16 x i8> %A, <i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7>
  ret <16 x i8> %B
; CHECK: sra_v16i8_7:
; CHECK: pxor
; CHECK: pcmpgtb
; CHECK: ret
}

define <16 x i8> @sra_v16i8(<16 x i8> %A) nounwind {
  %B = ashr <16 x i8> %A, <i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3>
  ret <16 x i8> %B
; CHECK: sra_v16i8:
; CHECK: psrlw $3
; CHECK: pand
; CHECK: pxor
; CHECK: psubb
; CHECK: ret
}
